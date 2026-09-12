"""Find the clean pre-roll before programme playback in an LD RF capture.

The normal decoder's ``--start`` argument is a capture-relative file-frame
number. This module finds that number without writing any decode output. It
uses the existing field and Philips-code decoders, so it deliberately avoids
trying to infer programme material from RF amplitude alone.
"""

from __future__ import print_function

import argparse
import math
import sys
import warnings
from collections import namedtuple
from copy import deepcopy

from lddecode.core import LDdecode
from lddecode.utils import inrange, make_loader, parse_frequency

RF_SAMPLE_RATE = 40000000
DEFAULT_MAX_SEARCH_SECONDS = 300.0
DEFAULT_FALLBACK_SECONDS = 10.0
DEFAULT_PRE_ROLL_SEARCH_SECONDS = 5.0
# A five-frame run can occur while a player is still settling or seeking. One
# second of clean, normal-speed addresses is long enough to reject that case.
REQUIRED_VBI_FRAMES = 30
# ``--start`` is a coarse, nominal frame location.  Verify a short phase run
# from that exact position before returning it, so it does not begin inside the
# last few unstable fields preceding the clean pre-roll.
REQUIRED_START_STABLE_FIELDS = 8
MAX_START_VALIDATION_SECONDS = 1.0

FrameObservation = namedtuple(
    "FrameObservation", "file_frame readloc address disk_type"
)
"""A complete decoded video frame and its capture-relative location."""


StartResult = namedtuple(
    "StartResult",
    "file_frame readloc confidence disk_type addresses searched_seconds",
)
"""Result from :func:`find_start`; confidence is ``vbi``, ``fallback``, or
``unvalidated``.  For VBI and unvalidated results, ``readloc`` is the first
decoded field sample (after any leading invalid-field recovery) and can
precede the nominal ``file_frame`` start location."""


StartValidationResult = namedtuple(
    "StartValidationResult", "file_frame readloc reason"
)
"""Internal nominal-``--start`` validation result.

``readloc`` is the first valid decoded field sample after any leading
invalid-field recovery; ``reason`` is set on failure.
"""


def file_frame_from_readloc(readloc, bytes_per_field):
    """Convert a field read location to the number consumed by ``--start``."""

    if bytes_per_field <= 0:
        raise ValueError("bytes_per_field must be positive")
    return int(readloc) // (int(bytes_per_field) * 2)


class VbiRunDetector:
    """Recognise contiguous, normal-speed CAV or CLV address runs."""

    def __init__(self, required_frames=REQUIRED_VBI_FRAMES):
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        self.required_frames = required_frames
        self.run = []

    def reset(self):
        self.run = []

    def observe(self, observation):
        """Add an observation and return the first run member when qualified."""

        if observation is None or observation.address is None:
            self.reset()
            return None

        if self.run:
            previous = self.run[-1]
            contiguous = (
                observation.disk_type == previous.disk_type
                and observation.address == previous.address + 1
            )
            if not contiguous:
                self.reset()

        self.run.append(observation)
        if len(self.run) >= self.required_frames:
            return self.run[0]
        return None


class _StableVideoTracker:
    """Track a continuous valid-field run for guarded VBI-less fallback."""

    def __init__(self):
        self.start_readloc = None
        self.last_readloc = None

    def reset(self):
        self.start_readloc = None
        self.last_readloc = None

    def observe(self, field, bytes_per_field):
        readloc = int(field.readloc)
        discontinuity = (
            self.last_readloc is None
            or readloc <= self.last_readloc
            or readloc - self.last_readloc > int(bytes_per_field) * 2
        )
        if discontinuity:
            self.start_readloc = readloc
        self.last_readloc = readloc
        return self.start_readloc


class _PreRollTracker:
    """Find the earliest clean field sequence immediately before content."""

    def __init__(self, bytes_per_field, field_phases):
        self.bytes_per_field = int(bytes_per_field)
        self.field_phases = int(field_phases)
        self.start_field = None
        self.previous_field = None

    def reset(self):
        self.start_field = None
        self.previous_field = None

    def _is_continuous(self, field):
        previous = self.previous_field
        if previous is None:
            return False

        distance = int(field.readloc) - int(previous.readloc)
        if distance < (self.bytes_per_field // 2):
            return False
        if distance > (self.bytes_per_field + (self.bytes_per_field // 2)):
            return False

        previous_phase = getattr(previous, "fieldPhaseID", None)
        current_phase = getattr(field, "fieldPhaseID", None)
        if previous_phase is None or current_phase is None:
            return bool(previous.isFirstField) != bool(field.isFirstField)

        expected_phase = 1 if previous_phase == self.field_phases else previous_phase + 1
        return current_phase == expected_phase

    def observe(self, field):
        if not self._is_continuous(field):
            self.start_field = field if field.isFirstField else None
        elif self.start_field is None and field.isFirstField:
            self.start_field = field
        self.previous_field = field


class _FinderLogger:
    """Suppress expected bad-RF diagnostics unless verbose output is wanted."""

    def __init__(self, report, verbose):
        self.report = report
        self.verbose = verbose

    def _log(self, message, *args):
        if not self.verbose or self.report is None:
            return
        if args:
            message = message % args
        self.report("decoder: " + str(message))

    debug = _log
    info = _log
    warning = _log
    error = _log

    def status(self, message):
        self._log(message)


def _make_decoder(filename, system, inputfreq, report, verbose):
    loader = make_loader(filename, inputfreq)
    return LDdecode(
        filename,
        None,
        loader,
        _FinderLogger(report, verbose),
        analog_audio=0,
        digital_audio=False,
        system=system,
        doDOD=False,
        threads=0,
    )


class StartFinder:
    """Run the lightweight field/VBI scan used by the command-line tool."""

    def __init__(
        self,
        decoder_factory,
        max_search_seconds=DEFAULT_MAX_SEARCH_SECONDS,
        fallback_seconds=DEFAULT_FALLBACK_SECONDS,
        required_frames=REQUIRED_VBI_FRAMES,
        sample_rate=RF_SAMPLE_RATE,
        pre_roll_search_seconds=DEFAULT_PRE_ROLL_SEARCH_SECONDS,
        report=None,
    ):
        if max_search_seconds < 0:
            raise ValueError("max_search_seconds must not be negative")
        if fallback_seconds < 0:
            raise ValueError("fallback_seconds must not be negative")
        if pre_roll_search_seconds < 0:
            raise ValueError("pre_roll_search_seconds must not be negative")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.decoder_factory = decoder_factory
        self.max_search_seconds = max_search_seconds
        self.fallback_seconds = fallback_seconds
        self.sample_rate = sample_rate
        self.pre_roll_search_seconds = pre_roll_search_seconds
        self.report = report
        self.detector = VbiRunDetector(required_frames)

    def _report_progress(self, position, last_bucket):
        seconds = int(position // self.sample_rate)
        bucket = seconds // 10
        if self.report is not None and seconds and bucket > last_bucket:
            self.report("scanned {0:02d}:{1:02d}".format(seconds // 60, seconds % 60))
        return bucket

    def _find_pre_roll_start(self, stable_observation, bytes_per_field, field_phases):
        """Replay the preceding clean field run and return its first frame."""

        if not self.pre_roll_search_seconds:
            return stable_observation

        lookback_samples = int(self.pre_roll_search_seconds * self.sample_rate)
        position = max(0, stable_observation.readloc - lookback_samples)
        tracker = _PreRollTracker(bytes_per_field, field_phases)
        previous_field = None
        decoder = None

        try:
            decoder = self.decoder_factory()
            while position <= stable_observation.readloc:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        field, offset = decoder.decodefield(position, 0, previous_field)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as error:
                    if self.report is not None:
                        self.report(
                            "pre-roll decode failure at sample {0}: {1}; "
                            "skipping one second".format(position, error)
                        )
                    decoder.close()
                    decoder = self.decoder_factory()
                    tracker.reset()
                    previous_field = None
                    position += self.sample_rate
                    continue

                if field is None or offset is None:
                    break

                try:
                    offset = int(offset)
                except (TypeError, ValueError):
                    offset = 0
                if offset <= 0:
                    offset = self.sample_rate

                if not field.valid:
                    tracker.reset()
                    previous_field = None
                elif int(field.readloc) <= stable_observation.readloc:
                    tracker.observe(field)
                    previous_field = field

                if int(getattr(field, "readloc", position)) >= stable_observation.readloc:
                    break
                position += offset
        finally:
            if decoder is not None:
                decoder.close()

        if tracker.start_field is None:
            return stable_observation

        return FrameObservation(
            file_frame_from_readloc(tracker.start_field.readloc, bytes_per_field),
            int(tracker.start_field.readloc),
            stable_observation.address,
            stable_observation.disk_type,
        )

    def _validate_start_frame(
        self,
        first_frame,
        last_frame,
        bytes_per_field,
        vbi_start_readloc=None,
    ):
        """Validate the earliest nominal ``--start`` within a clean pre-roll.

        The pre-roll replay begins at a decoded field boundary, whereas
        ``ld-decode --start`` begins at a nominal sample position.  The latter
        can still select a preceding unstable field.  Decode each candidate as
        the normal read path does, including recovery from leading invalid
        fields, and retain the first one with a stable phase run.  When a
        replayed pre-roll exists, do not let that recovery reach the confirmed
        VBI frame.
        """

        decoder = None
        try:
            decoder = self.decoder_factory()
            field_phases = int(decoder.rf.SysParams["fieldPhases"])
            initial_decoder_params = deepcopy(decoder.rf.DecoderParams)
            for candidate in range(int(first_frame), int(last_frame) + 1):
                # ``decodefield`` can recalibrate RF levels while locking.
                # Restore the factory state so every nominal start is an
                # independent probe without constructing another decoder.
                decoder.rf.DecoderParams.clear()
                decoder.rf.DecoderParams.update(deepcopy(initial_decoder_params))
                validation = self._validate_start_frame_candidate(
                    decoder,
                    candidate,
                    bytes_per_field,
                    field_phases,
                    vbi_start_readloc,
                )
                if validation is not None:
                    return validation
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            return StartValidationResult(
                None,
                None,
                "decoder failure while validating nominal --start: {0}".format(
                    error
                ),
            )
        finally:
            if decoder is not None:
                decoder.close()

        return StartValidationResult(
            None,
            None,
            "no phase-stable nominal --start from file frame {0} through {1}".format(
                first_frame, last_frame
            ),
        )

    def _validate_start_frame_candidate(
        self,
        decoder,
        candidate,
        bytes_per_field,
        field_phases,
        vbi_start_readloc,
    ):
        """Return a validation result for one fresh nominal ``--start`` probe.

        ``LDdecode.readfield`` follows a positive offset after an invalid
        leading field and restarts its phase relationship.  Do the same, but
        only within this nominal frame so an earlier candidate cannot borrow a
        later frame's clean run.
        """

        nominal_position = candidate * int(bytes_per_field) * 2
        nominal_end = nominal_position + (int(bytes_per_field) * 2)
        position = nominal_position
        previous_field = None
        first_readloc = None
        previous_phase = None
        stable_fields = 0
        while stable_fields < REQUIRED_START_STABLE_FIELDS:
            if vbi_start_readloc is not None and position >= vbi_start_readloc:
                return None

            field, offset = decoder.decodefield(position, 0, previous_field)
            if field is None or offset is None:
                return None

            try:
                offset = int(offset)
            except (TypeError, ValueError):
                return None
            if offset <= 0:
                return None
            next_position = position + offset

            if vbi_start_readloc is not None:
                try:
                    field_readloc = int(field.readloc)
                except (AttributeError, TypeError, ValueError):
                    return None
                if field_readloc >= vbi_start_readloc:
                    return None

            if not field.valid:
                # Recovery is permitted only before the stable run begins and
                # only while it remains in the candidate's nominal frame.
                if stable_fields or next_position >= nominal_end:
                    return None
                previous_field = None
                previous_phase = None
                position = next_position
                continue

            if self._would_report_player_skip(decoder, field):
                return None

            if first_readloc is None:
                first_readloc = int(field.readloc)
            phase = getattr(field, "fieldPhaseID", None)
            if phase is None:
                return None
            phase = int(phase)
            if previous_phase is not None:
                expected_phase = (
                    1 if previous_phase == field_phases else previous_phase + 1
                )
                if phase != expected_phase:
                    return None

            previous_phase = phase
            previous_field = field
            stable_fields += 1
            position = next_position

        return StartValidationResult(candidate, first_readloc, None)

    @staticmethod
    def _would_report_player_skip(decoder, field):
        """Return whether ``LDdecode.readfield`` would warn about this field.

        The lightweight fake fields used by start-finder tests do not carry
        picture geometry. Real decoder fields do, and must meet the same
        minimum quality rule as a normal decode before they can make a nominal
        start safe.
        """

        try:
            fieldlength = (
                field.linelocs[decoder.output_lines] - field.linelocs[0]
            ) / field.inlinelen
            return field.sync_confidence < 50 and not inrange(
                fieldlength,
                decoder.output_lines - 2,
                decoder.output_lines + 2,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ZeroDivisionError):
            return False

    def search(self):
        """Return a :class:`StartResult`, or ``None`` when no start is found."""

        max_samples = None
        if self.max_search_seconds:
            max_samples = int(self.max_search_seconds * self.sample_rate)
        fallback_samples = int(self.fallback_seconds * self.sample_rate)

        decoder = None
        position = 0
        previous_field = None
        first_field = None
        stable_video = _StableVideoTracker()
        last_progress_bucket = -1
        scanning_preamble = True
        acquiring_sync = False
        sync_acquire_end = None

        try:
            decoder = self.decoder_factory()
            while max_samples is None or position <= max_samples:
                last_progress_bucket = self._report_progress(
                    position, last_progress_bucket
                )
                sync_probe = getattr(decoder, "has_sync", None)
                if scanning_preamble and not acquiring_sync and callable(sync_probe):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            has_sync = sync_probe(position)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as error:
                        if self.report is not None:
                            self.report(
                                "sync probe failure at sample {0}: {1}; "
                                "using full decode".format(position, error)
                            )
                        has_sync = None

                    if has_sync is False:
                        position += self.sample_rate
                        continue
                    if has_sync is True:
                        # A coarse, one-second probe can land within a damaged
                        # field even when sync is present.  Once sync is seen,
                        # follow normal field offsets briefly so that the
                        # decoder can lock onto the next clean field instead
                        # of repeatedly skipping over the programme start.
                        acquiring_sync = True
                        sync_acquire_end = position + self.sample_rate
                try:
                    # Bad tracking windows can make the RF calculations emit NumPy
                    # RuntimeWarnings. They are expected probe failures, not a
                    # diagnostic for the caller.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        field, offset = decoder.decodefield(
                            position, 0, previous_field
                        )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as error:
                    if self.report is not None:
                        self.report(
                            "decode failure at sample {0}: {1}; skipping one second".format(
                                position, error
                            )
                        )
                    decoder.close()
                    decoder = None
                    decoder = self.decoder_factory()
                    position += self.sample_rate
                    previous_field = None
                    first_field = None
                    stable_video.reset()
                    self.detector.reset()
                    acquiring_sync = False
                    sync_acquire_end = None
                    continue

                if field is None or offset is None:
                    return None

                try:
                    offset = int(offset)
                except (TypeError, ValueError):
                    offset = 0
                if offset <= 0:
                    offset = self.sample_rate
                next_position = position + offset

                if not field.valid:
                    previous_field = None
                    first_field = None
                    stable_video.reset()
                    self.detector.reset()
                    if (
                        acquiring_sync
                        and sync_acquire_end is not None
                        and next_position < sync_acquire_end
                    ):
                        position = next_position
                        continue
                    acquiring_sync = False
                    sync_acquire_end = None
                    # Before content locks, testing every nominal field makes a
                    # tracking-back capture unnecessarily slow. A one-second
                    # stride still exercises a different field phase on each
                    # probe and is the same recovery interval used for errors.
                    position += self.sample_rate
                    continue

                scanning_preamble = False
                acquiring_sync = False
                sync_acquire_end = None

                stable_start = stable_video.observe(field, decoder.bytes_per_field)

                if field.isFirstField:
                    if first_field is not None:
                        self.detector.reset()
                    first_field = field
                elif first_field is None:
                    self.detector.reset()
                else:
                    address = decoder.decodeFrameNumber(first_field, field)
                    disk_type = "CLV" if decoder.isCLV else "CAV"
                    observation = FrameObservation(
                        file_frame_from_readloc(
                            first_field.readloc, decoder.bytes_per_field
                        ),
                        int(first_field.readloc),
                        None if address is None else int(address),
                        disk_type,
                    )
                    qualified = self.detector.observe(observation)
                    first_field = None
                    if qualified is not None:
                        pre_roll = self._find_pre_roll_start(
                            qualified,
                            decoder.bytes_per_field,
                            decoder.rf.SysParams["fieldPhases"],
                        )
                        last_validation_frame = qualified.file_frame
                        if pre_roll.file_frame < qualified.file_frame:
                            validation_frames = max(
                                1,
                                int(
                                    math.ceil(
                                        float(decoder.rf.SysParams["FPS"])
                                        * MAX_START_VALIDATION_SECONDS
                                    )
                                ),
                            )
                            last_validation_frame = min(
                                qualified.file_frame - 1,
                                pre_roll.file_frame + validation_frames - 1,
                            )
                        vbi_start_readloc = (
                            qualified.readloc if self.pre_roll_search_seconds else None
                        )
                        validation = self._validate_start_frame(
                            pre_roll.file_frame,
                            last_validation_frame,
                            decoder.bytes_per_field,
                            vbi_start_readloc,
                        )
                        if validation.file_frame is None:
                            if self.report is not None:
                                self.report(
                                    "confirmed {0} VBI run, but {1}".format(
                                        qualified.disk_type, validation.reason
                                    )
                                )
                            return StartResult(
                                pre_roll.file_frame,
                                pre_roll.readloc,
                                "unvalidated",
                                qualified.disk_type,
                                tuple(item.address for item in self.detector.run),
                                float(position) / self.sample_rate,
                            )
                        return StartResult(
                            validation.file_frame,
                            validation.readloc,
                            "vbi",
                            qualified.disk_type,
                            tuple(item.address for item in self.detector.run),
                            float(position) / self.sample_rate,
                        )

                if (
                    fallback_samples
                    and int(field.readloc) - stable_start >= fallback_samples
                ):
                    return StartResult(
                        file_frame_from_readloc(
                            stable_start, decoder.bytes_per_field
                        ),
                        stable_start,
                        "fallback",
                        None,
                        tuple(),
                        float(position) / self.sample_rate,
                    )

                previous_field = field
                position = next_position
        finally:
            if decoder is not None:
                decoder.close()

        return None


def find_start(
    filename,
    system="NTSC",
    inputfreq=None,
    max_search_seconds=DEFAULT_MAX_SEARCH_SECONDS,
    fallback_seconds=DEFAULT_FALLBACK_SECONDS,
    required_frames=REQUIRED_VBI_FRAMES,
    pre_roll_search_seconds=DEFAULT_PRE_ROLL_SEARCH_SECONDS,
    report=None,
    verbose=False,
):
    """Find clean pre-roll before programme playback without writing output."""

    def decoder_factory():
        return _make_decoder(filename, system, inputfreq, report, verbose)

    finder = StartFinder(
        decoder_factory,
        max_search_seconds=max_search_seconds,
        fallback_seconds=fallback_seconds,
        required_frames=required_frames,
        pre_roll_search_seconds=pre_roll_search_seconds,
        report=report,
    )
    return finder.search()


def _nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _build_parser():
    from lddecode import __version__

    parser = argparse.ArgumentParser(
        description=(
            "Find clean pre-roll before programme playback in a LaserDisc RF capture. "
            "A confirmed result is written as an ld-decode --start argument."
        )
    )
    parser.add_argument("infile", metavar="infile", help="source RF capture")
    parser.add_argument(
        "--max-search",
        metavar="seconds",
        type=_nonnegative_float,
        default=DEFAULT_MAX_SEARCH_SECONDS,
        help="maximum source seconds to search; 0 searches to EOF (default: 300)",
    )
    parser.add_argument(
        "--pre-roll-search",
        metavar="seconds",
        type=_nonnegative_float,
        default=DEFAULT_PRE_ROLL_SEARCH_SECONDS,
        help="seconds to replay before content to retain clean pre-roll (default: 5)",
    )
    parser.add_argument(
        "--PAL", "-p", "--pal", dest="pal", action="store_true", help="source is PAL"
    )
    parser.add_argument(
        "--NTSC", "-n", "--ntsc", dest="ntsc", action="store_true", help="source is NTSC"
    )
    parser.add_argument(
        "--NTSCJ", "-j", dest="ntscj", action="store_true", help="source is NTSC-J"
    )
    parser.add_argument(
        "-f",
        "--frequency",
        dest="inputfreq",
        metavar="FREQ",
        type=parse_frequency,
        default=None,
        help="RF sampling frequency in the source (default: 40MHz)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="include decoder diagnostics on stderr"
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(args=None):
    """Command-line entry point. Return a shell-friendly result status."""

    parser = _build_parser()
    parsed = parser.parse_args(args)
    if parsed.pal and (parsed.ntsc or parsed.ntscj):
        parser.error("can only be PAL or NTSC")
    if parsed.ntsc and parsed.ntscj:
        parser.error("can only be NTSC or NTSC-J")

    system = "PAL" if parsed.pal else "NTSC"

    def report(message):
        print("ld-find-start: {0}".format(message), file=sys.stderr)

    try:
        result = find_start(
            parsed.infile,
            system=system,
            inputfreq=parsed.inputfreq,
            max_search_seconds=parsed.max_search,
            pre_roll_search_seconds=parsed.pre_roll_search,
            report=report,
            verbose=parsed.verbose,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print("ld-find-start: ERROR: {0}".format(error), file=sys.stderr)
        return 1

    if result is None:
        print(
            "ld-find-start: no qualifying programme start found within the search limit",
            file=sys.stderr,
        )
        return 1

    argument = "--start {0}".format(result.file_frame)
    if result.confidence == "vbi":
        print(argument)
        print(
            "ld-find-start: found {0} run at file frame {1} "
            "(first decoded field sample {2}, searched {3:.2f}s)".format(
                result.disk_type,
                result.file_frame,
                result.readloc,
                result.searched_seconds,
            ),
            file=sys.stderr,
        )
        return 0

    if result.confidence == "fallback":
        print(
            "ld-find-start: WARNING: no advancing CAV/CLV VBI run; "
            "guarded stable-video candidate is {0} (sample {1})".format(
                argument, result.readloc
            ),
            file=sys.stderr,
        )
    elif result.confidence == "unvalidated":
        print(
            "ld-find-start: WARNING: confirmed {0} VBI run, but no "
            "phase-stable nominal --start was verified; guarded candidate is "
            "{1} (first decoded field sample {2})".format(
                result.disk_type, argument, result.readloc
            ),
            file=sys.stderr,
        )
    else:
        print(
            "ld-find-start: ERROR: unknown start confidence {0}".format(
                result.confidence
            ),
            file=sys.stderr,
        )
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
