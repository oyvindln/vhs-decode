"""
ld-compress - compress and uncompress LaserDisc RF captures

SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2022-2025 ld-decode contributors
SPDX-FileCopyrightText: 2026 Python implementation

A .lds capture stores four 10-bit samples in every five bytes.  A .ldf (or
.raw.oga) file holds the same samples losslessly re-coded as mono 16-bit
40 kHz Ogg FLAC, which is roughly half the size and is what ld-decode prefers
to be fed.

This was a bash script until 2026-07.  It is Python now so that it installs and
runs the same way on Linux, macOS and Windows: it is an ordinary console entry
point in pyproject.toml, so every packaging path that already ships ld-decode
ships ld-compress too.

Everything except the FLAC *encoder* happens in this process:

  - packing and unpacking use lddecode.lds, the same code ld-lds-converter-py
    is a thin CLI over;
  - .ldf decoding uses PyAV, the same libavcodec binding ld-ldf-reader-py and
    ld-decode itself read .ldf files with;
  - checksums use hashlib.

so flac is the only external program the packaging has to supply.  It has to be
1.5.0 or later, for its -j (multithreaded encoding) option.
"""

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import time
from concurrent import futures

import numpy as np

from lddecode import __version__
from lddecode.lds import S16, LdsWriter, unpack_stream

# flac gained -j, and with it a usable encoding speed for 30 GB captures, in
# 1.5.0.  Nothing older is worth falling back to.
FLAC_MIN_VERSION = (1, 5)

# Encoding throughput plateaus around 8 threads because the 10-to-16-bit
# unpacking that feeds the encoder becomes the bottleneck.
MAX_FLAC_THREADS = 8

SAMPLE_RATE = 40000

# Read size for whole-file hashing.  The pack/unpack path uses the larger,
# group-aligned chunk size lddecode.lds already defines.
HASH_CHUNK = 1 << 20

# How often the progress display redraws, in seconds
PROGRESS_INTERVAL = 0.2

COMPRESSED_SUFFIXES = (".raw.oga", ".ldf")


def _err(message):
    print(message, file=sys.stderr)


# --------------------------------------------------------------------------
# Progress display
# --------------------------------------------------------------------------


def _human(n):
    if n < 1024:
        return "%dB" % n
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024.0
        if n < 1024 or unit == "TiB":
            return "%.1f%s" % (n, unit)


def _clock(seconds):
    seconds = int(seconds)
    return "%d:%02d:%02d" % (seconds // 3600, (seconds // 60) % 60, seconds % 60)


def _term_width():
    # Terminals that do not report a size (and some CI pseudo-terminals) give 0
    # columns rather than an error, so treat anything unusable as unknown.
    try:
        cols = os.get_terminal_size(sys.stderr.fileno()).columns
    except (OSError, ValueError, AttributeError):
        cols = 0
    if cols < 20:
        try:
            cols = int(os.environ.get("COLUMNS", 0))
        except ValueError:
            cols = 0
    return cols if cols >= 20 else 80


class _Meter:
    """A pv-style bar, percentage, rate and ETA drawn on stderr."""

    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.start = time.monotonic()
        self.last = 0.0
        self.width = 0

    def draw(self, done, final=False):
        elapsed = time.monotonic() - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        cols = _term_width()

        if self.total:
            frac = min(1.0, float(done) / self.total)
            stats = "%3d%% %s/%s %s/s" % (
                int(frac * 100),
                _human(done),
                _human(self.total),
                _human(rate),
            )
            if final:
                stats += " in " + _clock(elapsed)
            elif rate > 0:
                stats += " ETA " + _clock((self.total - done) / rate)
        else:
            frac = None
            stats = "%s %s/s %s" % (_human(done), _human(rate), _clock(elapsed))

        line = self.label + " " if self.label else ""
        if frac is not None:
            bar_width = cols - len(line) - len(stats) - 4
            if bar_width >= 10:
                filled = int(frac * bar_width)
                if filled >= bar_width:
                    bar = "=" * bar_width
                elif filled > 0:
                    bar = "=" * (filled - 1) + ">" + " " * (bar_width - filled)
                else:
                    bar = " " * bar_width
                line += "[" + bar + "] "
        line += stats
        line = line[: cols - 1]

        sys.stderr.write("\r" + line + " " * max(0, self.width - len(line)))
        sys.stderr.flush()
        self.width = len(line)

    def tick(self, done):
        now = time.monotonic()
        if now - self.last >= PROGRESS_INTERVAL:
            self.last = now
            self.draw(done)

    def finish(self, done):
        self.draw(done, final=True)
        sys.stderr.write("\n")
        sys.stderr.flush()


class ProgressFile:
    """A binary input file that draws a progress bar as it is read.

    Progress is measured against the input file, which is the useful number for
    compression, uncompression and verification alike.  The position is taken
    from the underlying file rather than counted, so that PyAV seeking around
    while it probes an .ldf cannot skew it.

    Read-only, and seekable so that PyAV can accept it in place of a filename.
    """

    def __init__(self, path, show_progress=True):
        self._file = open(path, "rb")
        self._meter = None

        if show_progress:
            info = os.fstat(self._file.fileno())
            total = info.st_size if stat.S_ISREG(info.st_mode) else 0
            self._meter = _Meter(os.path.basename(path), total)
            self._meter.draw(0)

    def read(self, size=-1):
        buf = self._file.read(size)
        if self._meter is not None:
            self._meter.tick(self._file.tell())
        return buf

    def seek(self, offset, whence=os.SEEK_SET):
        return self._file.seek(offset, whence)

    def tell(self):
        return self._file.tell()

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def close(self):
        if self._meter is not None:
            self._meter.finish(self._file.tell())
            self._meter = None
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class _HashSink:
    """Write-only sink that keeps nothing but a running hash."""

    def __init__(self, digest):
        self.digest = digest

    def write(self, data):
        self.digest.update(data)
        return len(data)


# --------------------------------------------------------------------------
# flac
# --------------------------------------------------------------------------


def _parse_version(text):
    """Return the leading major.minor of a version string as a tuple, or None."""
    fields = text.split(".")
    if len(fields) < 2:
        return None

    parsed = []
    for field in fields[:2]:
        digits = ""
        for char in field:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            return None
        parsed.append(int(digits))

    return tuple(parsed)


def _bundled_tool_dirs():
    """Directories a packaged ld-decode keeps its own copies of tools in.

    The macOS app bundle, the Windows ZIP and an extracted AppImage all put
    flac right next to the ld-compress command, and none of them can count on
    that directory being on the user's PATH.  sys.executable is the frozen
    binary in a PyInstaller build and the interpreter otherwise; sys.argv[0] is
    the installed console script.
    """
    dirs = []
    for path in (sys.argv[0], sys.executable):
        if not path:
            continue
        directory = os.path.dirname(os.path.realpath(path))
        if directory and directory not in dirs:
            dirs.append(directory)
    return dirs


def find_flac():
    """Return the path to a usable flac, or exit with a clear message.

    Checked before any output file is created, so that a missing or too-old
    flac can never leave a truncated .ldf behind.
    """
    # A flac shipped alongside ld-compress wins over one on PATH, so that a
    # package always uses the version it was built and tested against.
    flac = None
    for directory in _bundled_tool_dirs():
        flac = shutil.which("flac", path=directory)
        if flac is not None:
            break

    if flac is None:
        flac = shutil.which("flac")

    if flac is None:
        _err("Error: required command 'flac' was not found in PATH.")
        _err("  The ld-decode packages provide it - check your installation is complete.")
        sys.exit(1)

    reported = ""
    try:
        result = subprocess.run(
            [flac, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        # "flac 1.5.0"
        fields = result.stdout.split()
        if len(fields) >= 2:
            reported = fields[1]
    except OSError:
        pass

    version = _parse_version(reported)
    if version is None or version < FLAC_MIN_VERSION:
        _err(
            "Error: flac %d.%d.0 or later is required (found %s)."
            % (
                FLAC_MIN_VERSION[0],
                FLAC_MIN_VERSION[1],
                reported if reported else "an unknown version",
            )
        )
        sys.exit(1)

    return flac


def _flac_threads():
    return max(1, min(MAX_FLAC_THREADS, os.cpu_count() or 1))


def _flac_encode_command(flac, level, outfile):
    """Encode raw s16le mono 40 kHz samples on stdin into Ogg FLAC in outfile."""
    return [
        flac,
        "-s",
        "-f",
        "-j%d" % _flac_threads(),
        "-%d" % level,
        "--ogg",
        "--force-raw-format",
        "--endian=little",
        "--sign=signed",
        "--channels=1",
        "--bps=16",
        "--sample-rate=%d" % SAMPLE_RATE,
        "-o",
        outfile,
        "-",
    ]


# --------------------------------------------------------------------------
# .ldf decoding
# --------------------------------------------------------------------------


def decode_ldf(source, write_samples):
    """Decode an Ogg FLAC stream into signed 16-bit mono samples.

    source is a file object; write_samples is called with numpy arrays of
    int16.  This is the PyAV path ld-decode and ld-ldf-reader-py already use,
    which is why ld-compress needs no ffmpeg binary of its own.
    """
    try:
        import av
    except ImportError:
        _err("Error: PyAV library not found. Install with: pip install av")
        sys.exit(1)

    with av.open(source) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise ValueError("no audio stream found")

        stream = streams[0]
        # Decode on several threads, as the ffmpeg command line would
        try:
            stream.thread_type = "AUTO"
        except (AttributeError, ValueError):
            pass

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono")

        def emit(frame):
            for resampled in resampler.resample(frame):
                write_samples(np.frombuffer(bytes(resampled.planes[0]), dtype=S16))

        for frame in container.decode(stream):
            emit(frame)

        # Flush whatever the resampler is still holding
        emit(None)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def _strip_suffix(path, suffixes):
    """Return the basename of path with the first matching suffix removed."""
    name = os.path.basename(path)
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _discard_warning(discarded, group):
    if discarded:
        _err(
            "Warning: input did not end on a %d-byte sample group boundary; "
            "ignored %d trailing byte(s)" % (group, discarded)
        )


def compress(path, flac, level, extension, show_progress):
    """Compress a .lds file into .ldf/.raw.oga in the current directory."""
    outfile = _strip_suffix(path, (".lds",)) + "." + extension
    # Encode under a temporary name and move it into place only once flac has
    # exited cleanly.  "flac -f -o" truncates its output the moment it starts,
    # so writing straight to outfile would leave a failed run looking like a
    # good capture, and would destroy a .ldf that was already there.
    partial = outfile + ".part"

    _err("Compressing '%s' to '%s'" % (path, outfile))

    try:
        source = ProgressFile(path, show_progress)
    except OSError as e:
        _err("Error: %s" % e)
        _err("Error: compression of '%s' failed." % path)
        return False

    process = subprocess.Popen(
        _flac_encode_command(flac, level, partial), stdin=subprocess.PIPE
    )

    failure = None
    discarded = 0
    try:
        with source:
            discarded = unpack_stream(source, process.stdin)
    except KeyboardInterrupt:
        process.kill()
        process.wait()
        _discard_partial(partial)
        raise
    except Exception as e:
        # Deliberately broad: whatever went wrong, the partial file has to go
        # rather than be moved into place.  A BrokenPipeError means flac has
        # already died, and its own exit status is the more useful thing to
        # report.
        if not isinstance(e, BrokenPipeError):
            failure = str(e)
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        status = process.wait()

    if status != 0 or failure is not None:
        if failure is not None:
            _err("Error: %s" % failure)
        _err("Error: compression of '%s' failed." % path)
        _discard_partial(partial)
        return False

    try:
        os.replace(partial, outfile)
    except OSError as e:
        _err("Error: could not move '%s' into place: %s" % (outfile, e))
        _discard_partial(partial)
        return False

    _discard_warning(discarded, 5)
    return True


def uncompress(path, show_progress):
    """Uncompress a .ldf/.raw.oga file into .lds in the current directory."""
    outfile = _strip_suffix(path, COMPRESSED_SUFFIXES) + ".lds"
    partial = outfile + ".part"

    _err("Uncompressing '%s' to '%s'" % (path, outfile))

    try:
        writer = LdsWriter(partial)
        try:
            with ProgressFile(path, show_progress) as source:
                decode_ldf(source, writer.write)
        finally:
            writer.close()
        os.replace(partial, outfile)
    except Exception as e:
        # Broad on purpose: PyAV raises whichever builtin matches the FFmpeg
        # error, and a partly written .lds must never reach outfile.
        _err("Error: %s" % e)
        _err("Error: uncompression of '%s' failed." % path)
        _discard_partial(partial)
        return False

    return True


def verify(path, show_progress):
    """Print the md5 of a .ldf/.raw.oga file and of the .lds it contains."""
    lds_name = _strip_suffix(path, COMPRESSED_SUFFIXES) + ".lds"

    _err("Performing checksum of '%s':" % path)

    contents = hashlib.md5()
    writer = LdsWriter(_HashSink(contents))

    try:
        # Hash the container in parallel with decoding it, so that verifying a
        # 30 GB capture still only takes as long as the slower of the two.
        with futures.ThreadPoolExecutor(max_workers=1) as pool:
            container_md5 = pool.submit(_file_md5, path)
            try:
                with ProgressFile(path, show_progress) as source:
                    decode_ldf(source, writer.write)
            finally:
                writer.close()
            container_digest = container_md5.result()
    except Exception as e:
        _err("Error: %s" % e)
        _err("Error: verification of '%s' failed." % path)
        return False

    print("%s  %s" % (container_digest, path))
    print("%s  %s" % (contents.hexdigest(), lds_name))
    return True


def _file_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _discard_partial(partial):
    try:
        os.remove(partial)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _level(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid compression level: %s" % text)
    if not 1 <= value <= 8:
        raise argparse.ArgumentTypeError("invalid compression level: %s" % text)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ld-compress",
        description="ld-compress - compress and uncompress LaserDisc RF captures\n\n"
        "(c)2022-2025 ld-decode contributors\n"
        "(c)2026 Python implementation\n"
        "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-c",
        "--compress",
        action="store_true",
        help="compress .lds files to .ldf files in the current directory (default)",
    )
    mode.add_argument(
        "-u",
        "--uncompress",
        action="store_true",
        help="uncompress .ldf/.raw.oga files to .lds files in the current directory",
    )
    mode.add_argument(
        "-v",
        "--verify",
        action="store_true",
        help="print md5 checksums of the given .ldf/.raw.oga files and of the "
        ".lds data they contain",
    )

    parser.add_argument(
        "-l",
        "--level",
        type=_level,
        default=8,
        metavar="1-8",
        help="compression level 1 - 8 (default 8)",
    )
    parser.add_argument(
        "-g",
        "--oga",
        action="store_true",
        help="use the .raw.oga extension instead of .ldf when compressing",
    )

    progress = parser.add_mutually_exclusive_group()
    progress.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="always show the progress display, even when stderr is not a terminal",
    )
    progress.add_argument(
        "-n",
        "--no-progress",
        action="store_true",
        help="never show the progress display",
    )

    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("files", nargs="+", metavar="file", help="file(s) to process")

    args = parser.parse_args(argv)

    # Show progress by default, but only when there is a terminal to draw it
    # on, so that output redirected to a log file stays as clean as it was.
    show_progress = args.progress or (not args.no_progress and sys.stderr.isatty())

    extension = "raw.oga" if args.oga else "ldf"
    ok = True

    if args.uncompress:
        for path in args.files:
            if path.endswith(COMPRESSED_SUFFIXES):
                ok = uncompress(path, show_progress) and ok
            else:
                _err(
                    "Error: '%s' does not appear to be a .raw.oga/.ldf file. Skipping."
                    % path
                )
                ok = False
    elif args.verify:
        for path in args.files:
            if path.endswith(COMPRESSED_SUFFIXES):
                ok = verify(path, show_progress) and ok
            else:
                _err(
                    "Error: '%s' does not appear to be a .raw.oga/.ldf file. Skipping."
                    % path
                )
                ok = False
    else:
        # Look flac up once, before anything is written
        flac = find_flac()
        for path in args.files:
            if path.endswith(".lds"):
                ok = compress(path, flac, args.level, extension, show_progress) and ok
            else:
                _err("Error: '%s' does not appear to be a .lds file. Skipping." % path)
                ok = False

    if not ok:
        _err("Task finished with errors.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
