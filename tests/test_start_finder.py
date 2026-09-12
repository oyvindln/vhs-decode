from types import SimpleNamespace

import numpy as np

from lddecode.core import RFDecode
from lddecode.start_finder import (
    FrameObservation,
    StartFinder,
    StartResult,
    VbiRunDetector,
    file_frame_from_readloc,
    main,
)


def observation(address, file_frame=0, readloc=0, disk_type="CAV"):
    return FrameObservation(file_frame, readloc, address, disk_type)


def test_file_frame_conversion_matches_start_seek_units():
    bytes_per_field = 674029
    readloc = (bytes_per_field * 2 * 1234) + bytes_per_field
    assert file_frame_from_readloc(readloc, bytes_per_field) == 1234


def test_sync_only_demodulation_matches_the_normal_sync_path():
    rf = RFDecode(system="NTSC", blocklen=32768)
    raw = np.random.default_rng(0).integers(
        -32768, 32767, size=rf.blocklen, dtype=np.int16
    )

    full = rf.demodblock_cpu(data=raw, cut=True)["video"]["demod_05"]
    sync = rf.demodblock_sync(data=raw, cut=True)

    np.testing.assert_array_equal(sync, full)


def test_vbi_run_requires_contiguous_addresses_of_one_disk_type():
    detector = VbiRunDetector(required_frames=5)
    assert detector.observe(observation(100)) is None
    assert detector.observe(observation(101)) is None
    assert detector.observe(observation(103)) is None
    assert detector.observe(observation(104)) is None
    assert detector.observe(observation(105)) is None
    assert detector.observe(observation(106)) is None
    result = detector.observe(observation(107))
    assert result.address == 103

    detector.reset()
    for address in (200, 201, 202, 203):
        assert detector.observe(observation(address, disk_type="CLV")) is None
    assert detector.observe(observation(204, disk_type="CAV")) is None


def test_vbi_run_rejects_paused_reverse_and_fast_forward_addresses():
    detector = VbiRunDetector(required_frames=3)
    assert detector.observe(observation(10)) is None
    assert detector.observe(observation(10)) is None
    assert detector.observe(observation(11)) is None
    assert detector.observe(observation(9)) is None
    assert detector.observe(observation(10)) is None
    assert detector.observe(observation(20)) is None
    assert detector.observe(observation(21)) is None
    assert detector.observe(observation(22)).address == 20


def test_default_vbi_run_requires_a_sustained_second_of_addresses():
    detector = VbiRunDetector()
    for address in range(29):
        assert detector.observe(observation(address)) is None
    assert detector.observe(observation(29)).address == 0


class FakeField:
    def __init__(
        self,
        readloc,
        is_first_field,
        address=None,
        disk_type="CAV",
        valid=True,
        field_phase_id=None,
    ):
        self.readloc = readloc
        self.isFirstField = is_first_field
        self.address = address
        self.disk_type = disk_type
        self.valid = valid
        self.fieldPhaseID = field_phase_id


class FakeDecoder:
    def __init__(self, events, bytes_per_field=100):
        self.events = list(events)
        self.bytes_per_field = bytes_per_field
        self.isCLV = False
        self.closed = False
        self.rf = SimpleNamespace(
            SysParams={"fieldPhases": 4, "FPS": 30},
            DecoderParams={"ire0": 0},
        )

    def decodefield(self, _position, _mtf, _previous):
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        if event is None:
            return None, None
        return event, self.bytes_per_field

    def decodeFrameNumber(self, _first, second):
        self.isCLV = second.disk_type == "CLV"
        return second.address

    def close(self):
        self.closed = True


class SyncProbeFakeDecoder(FakeDecoder):
    def __init__(self, events, sync_results, bytes_per_field=100):
        super().__init__(events, bytes_per_field)
        self.sync_results = iter(sync_results)
        self.sync_positions = []

    def has_sync(self, position):
        self.sync_positions.append(position)
        return next(self.sync_results)


class SyncRecoveryFakeDecoder(FakeDecoder):
    """Return fields only at the positions reached by normal field stepping."""

    def __init__(self, events, bytes_per_field=100):
        super().__init__([], bytes_per_field)
        self.events_by_position = dict(events)
        self.sync_positions = []

    def has_sync(self, position):
        self.sync_positions.append(position)
        return True

    def decodefield(self, position, _mtf, _previous):
        return self.events_by_position.get(position, (None, None))


class StartValidationFakeDecoder(FakeDecoder):
    """Emit a phase sequence keyed by the requested nominal sample position."""

    def __init__(self, phases, start_position=1000, bytes_per_field=100):
        super().__init__([], bytes_per_field)
        self.phases = list(phases)
        self.start_position = start_position
        self.calls = []

    def decodefield(self, position, _mtf, previous):
        self.calls.append((position, previous))
        index = (position - self.start_position) // self.bytes_per_field
        if index < 0 or index >= len(self.phases):
            return None, None
        phase = self.phases[index]
        return (
            FakeField(
                position,
                bool(index % 2),
                valid=phase is not None,
                field_phase_id=phase,
            ),
            self.bytes_per_field,
        )


class OffsetValidationFakeDecoder(FakeDecoder):
    """Emit validation fields and offsets keyed by the requested position."""

    def __init__(self, events, bytes_per_field=100):
        super().__init__([], bytes_per_field)
        self.events = dict(events)
        self.calls = []

    def decodefield(self, position, _mtf, previous):
        self.calls.append((position, previous))
        return self.events.get(position, (None, None))


class ParameterResetValidationDecoder(FakeDecoder):
    """Require RF parameters to be reset before the second candidate."""

    def __init__(self):
        super().__init__([])
        self.calls = []

    def decodefield(self, position, _mtf, previous):
        self.calls.append((position, self.rf.DecoderParams["ire0"], previous))
        if 1000 <= position < 1200:
            self.rf.DecoderParams["ire0"] = 99
            phase = 4 if position == 1000 else 3
        elif 1200 <= position < 2000:
            if self.rf.DecoderParams["ire0"] != 0:
                return None, None
            phase = ((position - 1200) // self.bytes_per_field) % 4 + 1
        else:
            return None, None
        return (
            FakeField(
                position,
                bool((position // self.bytes_per_field) % 2),
                field_phase_id=phase,
            ),
            self.bytes_per_field,
        )


def validation_decoder(start_file_frame, phases=None, bytes_per_field=100):
    if phases is None:
        phases = [1, 2, 3, 4, 1, 2, 3, 4]
    return StartValidationFakeDecoder(
        phases,
        start_position=start_file_frame * bytes_per_field * 2,
        bytes_per_field=bytes_per_field,
    )


def five_frames(start_file_frame=7, sample_base=0, disk_type="CAV"):
    fields = []
    for index in range(5):
        readloc = sample_base + ((start_file_frame + index) * 200)
        fields.append(FakeField(readloc, True, disk_type=disk_type))
        fields.append(FakeField(readloc + 100, False, 100 + index, disk_type))
    return fields


def test_start_finder_returns_first_cav_frame_of_vbi_run():
    decoder = FakeDecoder(five_frames())
    validation = validation_decoder(7)
    decoders = iter([decoder, validation])
    result = StartFinder(
        lambda: next(decoders),
        sample_rate=100,
        required_frames=5,
        pre_roll_search_seconds=0,
    ).search()
    assert result.confidence == "vbi"
    assert result.file_frame == 7
    assert result.readloc == 1400
    assert result.disk_type == "CAV"
    assert result.addresses == (100, 101, 102, 103, 104)
    assert decoder.closed
    assert validation.closed


def test_start_finder_recreates_decoder_after_decode_failure():
    first = FakeDecoder([RuntimeError("bad RF")])
    second = FakeDecoder(five_frames(start_file_frame=9))
    validation = validation_decoder(9)
    decoders = iter([first, second, validation])
    result = StartFinder(
        lambda: next(decoders), sample_rate=100, required_frames=5, pre_roll_search_seconds=0
    ).search()
    assert result.confidence == "vbi"
    assert result.file_frame == 9
    assert first.closed
    assert second.closed
    assert validation.closed


def test_start_finder_skips_sync_free_preamble_without_full_field_decode():
    decoder = SyncProbeFakeDecoder(five_frames(), [False, True])
    validation = validation_decoder(7)
    decoders = iter([decoder, validation])
    result = StartFinder(
        lambda: next(decoders),
        sample_rate=100,
        required_frames=5,
        pre_roll_search_seconds=0,
    ).search()

    assert result.confidence == "vbi"
    assert decoder.sync_positions == [0, 100]
    assert decoder.closed
    assert validation.closed


def test_start_finder_recovers_from_an_invalid_field_after_sync_detection():
    # A one-second sync probe can identify video but land in a malformed field.
    # Follow its field offset to the next clean field; jumping a full second
    # would miss this VBI run entirely.
    decoder = SyncRecoveryFakeDecoder(
        [
            (0, (FakeField(0, True, valid=False), 10)),
            (10, (FakeField(1000, True), 100)),
            (110, (FakeField(1100, False, 0), 100)),
            (210, (FakeField(1200, True), 100)),
            (310, (FakeField(1300, False, 1), 100)),
            (410, (FakeField(1400, True), 100)),
            (510, (FakeField(1500, False, 2), 100)),
        ]
    )
    validation = validation_decoder(5)
    decoders = iter([decoder, validation])

    result = StartFinder(
        lambda: next(decoders),
        sample_rate=100,
        required_frames=3,
        pre_roll_search_seconds=0,
    ).search()

    assert result.confidence == "vbi"
    assert result.file_frame == 5
    assert decoder.sync_positions == [0]
    assert decoder.closed
    assert validation.closed


def test_start_validation_skips_nominal_frames_with_phase_mismatches():
    # Frame 5 is the decoded-field pre-roll boundary, but the normal decoder
    # starts before it and sees unstable phases until nominal frame 8.
    phases = [4, 3, 2, 3, 4, 3, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]
    decoder = StartValidationFakeDecoder(phases)
    finder = StartFinder(lambda: decoder, sample_rate=100)

    result = finder._validate_start_frame(5, 8, 100)
    assert result.file_frame == 8
    assert result.readloc == 1600
    assert result.reason is None
    success_start = decoder.calls.index((1600, None))
    assert decoder.calls[success_start + 1][1].readloc == 1600
    assert decoder.closed


def test_start_validation_recovers_leading_invalid_field_within_nominal_frame():
    decoder = StartValidationFakeDecoder(
        [None, 1, 2, 3, 4, 1, 2, 3, 4], start_position=1000
    )
    result = StartFinder(lambda: decoder, sample_rate=100)._validate_start_frame(
        5, 5, 100
    )

    assert result.file_frame == 5
    assert result.readloc == 1100
    assert decoder.calls[0] == (1000, None)
    assert decoder.calls[1] == (1100, None)
    assert decoder.calls[2][1].readloc == 1100
    assert decoder.closed


def test_start_validation_rejects_leading_recovery_into_next_nominal_frame():
    events = {
        1000: (FakeField(1000, True, valid=False), 200),
    }
    for index, phase in enumerate((1, 2, 3, 4, 1, 2, 3, 4)):
        position = 1200 + (index * 100)
        events[position] = (FakeField(position, bool(index % 2), field_phase_id=phase), 100)
    decoder = OffsetValidationFakeDecoder(events)

    result = StartFinder(lambda: decoder, sample_rate=100)._validate_start_frame(
        5, 6, 100
    )

    assert result.file_frame == 6
    assert [call[0] for call in decoder.calls] == [1000] + list(
        range(1200, 2000, 100)
    )
    assert decoder.closed


def test_start_validation_rejects_invalid_field_after_stable_run_begins():
    decoder = StartValidationFakeDecoder(
        [1, None, 2, 3, 4, 1, 2, 3, 4], start_position=1000
    )

    result = StartFinder(lambda: decoder, sample_rate=100)._validate_start_frame(
        5, 5, 100
    )

    assert result.file_frame is None
    assert [call[0] for call in decoder.calls] == [1000, 1100]
    assert decoder.closed


def test_start_validation_rejects_fields_that_would_warn_of_player_skip():
    warning_field = FakeField(1100, False, field_phase_id=2)
    warning_field.sync_confidence = 10
    warning_field.linelocs = [0] * 11
    warning_field.inlinelen = 1
    decoder = OffsetValidationFakeDecoder(
        {
            1000: (FakeField(1000, True, field_phase_id=1), 100),
            1100: (warning_field, 100),
        }
    )
    decoder.output_lines = 10

    result = StartFinder(lambda: decoder, sample_rate=100)._validate_start_frame(
        5, 5, 100
    )

    assert result.file_frame is None
    assert [call[0] for call in decoder.calls] == [1000, 1100]
    assert decoder.closed


def test_start_validation_does_not_probe_into_confirmed_vbi_frame():
    decoder = validation_decoder(5)

    result = StartFinder(lambda: decoder, sample_rate=100)._validate_start_frame(
        5, 5, 100, vbi_start_readloc=1500
    )

    assert result.file_frame is None
    assert [call[0] for call in decoder.calls] == list(range(1000, 1500, 100))
    assert decoder.closed


def test_start_validation_reuses_one_decoder_with_reset_rf_parameters():
    decoder = ParameterResetValidationDecoder()
    factory_calls = []

    def decoder_factory():
        factory_calls.append(None)
        return decoder

    result = StartFinder(decoder_factory, sample_rate=100)._validate_start_frame(
        5, 6, 100
    )

    assert result.file_frame == 6
    assert factory_calls == [None]
    candidate_six_start = next(
        call for call in decoder.calls if call[0] == 1200 and call[2] is None
    )
    assert candidate_six_start[1] == 0
    assert decoder.closed


def test_start_finder_caps_nominal_validation_at_one_second():
    forward = FakeDecoder(five_frames(start_file_frame=40))
    replay = FakeDecoder(
        [
            FakeField(
                index * 100,
                index % 2 == 0,
                field_phase_id=(index % 4) + 1,
            )
            for index in range(81)
        ]
    )
    validation = validation_decoder(0, [4, 3] * 60)
    decoders = iter([forward, replay, validation])

    result = StartFinder(
        lambda: next(decoders),
        sample_rate=10000,
        required_frames=5,
        pre_roll_search_seconds=5,
    ).search()

    assert result.confidence == "unvalidated"
    starts = [position for position, previous in validation.calls if previous is None]
    assert starts == [frame * 200 for frame in range(30)]
    assert validation.closed


def test_start_finder_returns_unvalidated_result_when_no_nominal_start_is_safe():
    forward = FakeDecoder(five_frames())
    validation = validation_decoder(7, [4, 3, 4, 3, 4, 3, 4, 3])
    decoders = iter([forward, validation])
    reports = []

    result = StartFinder(
        lambda: next(decoders),
        sample_rate=100,
        required_frames=5,
        pre_roll_search_seconds=0,
        report=reports.append,
    ).search()

    assert result.confidence == "unvalidated"
    assert result.file_frame == 7
    assert "no phase-stable nominal --start" in reports[-1]
    assert forward.closed
    assert validation.closed


def test_start_finder_reports_validation_decoder_failure_as_unvalidated():
    forward = FakeDecoder(five_frames())
    validation = FakeDecoder([RuntimeError("validation RF")])
    decoders = iter([forward, validation])
    reports = []

    result = StartFinder(
        lambda: next(decoders),
        sample_rate=100,
        required_frames=5,
        pre_roll_search_seconds=0,
        report=reports.append,
    ).search()

    assert result.confidence == "unvalidated"
    assert "decoder failure while validating" in reports[-1]
    assert forward.closed
    assert validation.closed


def test_start_finder_returns_guarded_fallback_for_stable_video_without_vbi():
    fields = [
        FakeField(index * 100, index % 2 == 0, None)
        for index in range(11)
    ]
    decoder = FakeDecoder(fields, bytes_per_field=100)
    result = StartFinder(
        lambda: decoder, sample_rate=100, fallback_seconds=10
    ).search()
    assert result.confidence == "fallback"
    assert result.file_frame == 0
    assert result.readloc == 0


def test_start_finder_returns_first_clv_frame_of_the_stable_run():
    decoder = FakeDecoder(five_frames(start_file_frame=30, disk_type="CLV"))
    validation = validation_decoder(30)
    decoders = iter([decoder, validation])
    result = StartFinder(
        lambda: next(decoders),
        sample_rate=100,
        required_frames=5,
        pre_roll_search_seconds=0,
    ).search()
    assert result.confidence == "vbi"
    assert result.disk_type == "CLV"
    assert result.file_frame == 30
    assert result.readloc == 6000
    assert validation.closed


def test_start_finder_keeps_clean_pre_roll_after_last_phase_break():
    forward = FakeDecoder(five_frames(start_file_frame=25))
    replay = FakeDecoder(
        [
            FakeField(3500, True, field_phase_id=1),
            FakeField(3600, False, field_phase_id=2),
            # The player re-locks at a new first field.  Keep this field and
            # the following clean pre-roll rather than returning the later
            # VBI-confirmed programme frame at sample 4000.
            FakeField(3700, True, field_phase_id=1),
            FakeField(3800, False, field_phase_id=2),
            FakeField(3900, True, field_phase_id=3),
            FakeField(4000, False, field_phase_id=4),
        ]
    )
    validation = validation_decoder(
        18, [4, 3, 2, 3, 4, 1, 2, 3, 4, 1]
    )
    decoders = iter([forward, replay, validation])

    result = StartFinder(
        lambda: next(decoders),
        sample_rate=100,
        required_frames=5,
        pre_roll_search_seconds=5,
    ).search()

    assert result.confidence == "vbi"
    assert result.file_frame == 19
    assert result.readloc == 3800
    assert forward.closed
    assert replay.closed
    assert validation.closed


def test_main_writes_only_confirmed_start_argument_to_stdout(monkeypatch, capsys):
    result = StartResult(42, 8400, "vbi", "CAV", (1, 2, 3, 4, 5), 3.0)
    monkeypatch.setattr("lddecode.start_finder.find_start", lambda *_args, **_kwargs: result)
    assert main(["capture.ldf"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "--start 42\n"
    assert "found CAV run" in captured.err
    assert "first decoded field sample 8400" in captured.err


def test_main_keeps_guarded_fallback_off_stdout(monkeypatch, capsys):
    result = StartResult(42, 8400, "fallback", None, tuple(), 10.0)
    monkeypatch.setattr("lddecode.start_finder.find_start", lambda *_args, **_kwargs: result)
    assert main(["capture.ldf"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--start 42" in captured.err


def test_main_keeps_unvalidated_vbi_candidate_off_stdout(monkeypatch, capsys):
    result = StartResult(42, 8400, "unvalidated", "CAV", (1, 2, 3, 4, 5), 3.0)
    monkeypatch.setattr("lddecode.start_finder.find_start", lambda *_args, **_kwargs: result)
    assert main(["capture.ldf"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "confirmed CAV VBI run" in captured.err
    assert "first decoded field sample 8400" in captured.err
    assert "--start 42" in captured.err


def test_main_returns_one_without_a_candidate(monkeypatch, capsys):
    monkeypatch.setattr("lddecode.start_finder.find_start", lambda *_args, **_kwargs: None)
    assert main(["capture.ldf"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no qualifying programme start" in captured.err
