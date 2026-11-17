import io
import json
import os
import time

import numpy as np
import pytest

from vhsdecode import parallel

SAMPLES = 32
FIELD_BYTES = SAMPLES * 2
SPF = 1000
# Fields a fake decoder needs before its output matches a single decode.
SETTLE = 3


def _image(seed):
    return np.random.default_rng(seed).integers(0, 65535, SAMPLES, dtype=np.uint16).tobytes()


class FakeTape:
    """A capture of `length` fields. A decoder started at field `start` gives
    noise for its first SETTLE fields and for every field in `bad`, where no
    two decoders agree, and the true field everywhere else."""

    def __init__(self, length, bad=()):
        self.length = length
        self.bad = set(bad)

    def field(self, start, n):
        if n - start < SETTLE or n in self.bad:
            return _image(10**6 + start * 10**4 + n), _image(2 * 10**6 + start * 10**4 + n)
        return _image(n), _image(10**5 + n)


class FakeWorker:
    """Decodes segments of a FakeTape the way a worker process does, writing
    the same files and messages, synchronously as commands arrive."""

    def __init__(self, tape, work_dir):
        self.tape = tape
        self.work_dir = work_dir
        self.segment = None
        self.log_path = "fake"
        self._job = None

    def failed(self):
        return False

    def _send(self, msg):
        self._job["meta"].write(json.dumps(msg) + "\n")
        self._job["meta"].flush()

    def _decode_to(self, limit):
        job = self._job
        while job["written"] < limit:
            n = job["start"] + job["written"]
            if n >= self.tape.length:
                self._end()
                return
            luma, chroma = self.tape.field(job["start"], n)
            job["luma"].write(luma)
            job["chroma"].write(chroma)
            job["luma"].flush()
            job["chroma"].flush()
            if job["written"] == 0:
                self._send({"fieldBytes": FIELD_BYTES})
            self._send({"field": {"isFirstField": n % 2 == 0, "syncConf": 100,
                                  "seqNo": job["written"] + 1, "fileLoc": n * SPF}})
            job["written"] += 1

    def _end(self):
        job = self._job
        params = {"videoParameters": {"fieldWidth": SAMPLES}} if job["written"] else None
        self._send({"end": True, "json": params})
        for f in ("meta", "luma", "chroma"):
            job[f].close()
        self._job = None

    def send(self, cmd, seg, *rest):
        if cmd == "job":
            start, limit = rest
            path = parallel._segment_path(self.work_dir, seg)
            self._job = {
                "seg": seg, "start": start, "written": 0,
                "meta": open(path + ".fields", "w"),
                "luma": open(path + ".tbc", "wb"),
                "chroma": open(path + "_chroma.tbc", "wb"),
            }
            self._send({"start": start * SPF, "samplesPerField": SPF,
                        "suffixes": [".tbc", "_chroma.tbc"], "chromaFraming": False})
            self._decode_to(limit)
        elif self._job is None or self._job["seg"] != seg:
            return
        elif cmd == "limit":
            self._decode_to(rest[0])
        elif cmd == "stop":
            self._end()


def _stitch(tmp_path, tape, workers=4, first_field=0, max_fields=None):
    work_dir = str(tmp_path / "work")
    os.makedirs(work_dir)
    out = str(tmp_path / "out")
    fakes = [FakeWorker(tape, work_dir) for _ in range(workers)]
    stitcher = parallel.Stitcher(fakes, work_dir, first_field, max_fields, out, log=lambda s: None)
    fields = stitcher.run()
    stitcher.write_json(stitcher.finish())
    with open(out + ".tbc.json") as f:
        meta = json.load(f)
    with open(out + ".tbc", "rb") as f:
        luma = f.read()
    with open(out + "_chroma.tbc", "rb") as f:
        chroma = f.read()
    assert os.listdir(work_dir) == []
    return stitcher, fields, meta, luma, chroma


def _single_decode(tape, first_field, count):
    """Output of one decoder that starts at first_field."""
    fields = [tape.field(first_field, n) for n in range(first_field, first_field + count)]
    return b"".join(y for y, _ in fields), b"".join(c for _, c in fields)


def _luma_at(luma, i):
    return luma[i * FIELD_BYTES:(i + 1) * FIELD_BYTES]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(parallel.time, "sleep", lambda s: None)


class TestStitcher:
    def test_matches_single_decode(self, tmp_path):
        tape = FakeTape(1000)
        stitcher, fields, meta, luma, chroma = _stitch(tmp_path, tape)
        assert fields == 1000
        assert (luma, chroma) == _single_decode(tape, 0, 1000)
        assert [f["seqNo"] for f in meta["fields"]] == list(range(1, 1001))
        assert [f["fileLoc"] for f in meta["fields"]] == [n * SPF for n in range(1000)]
        assert meta["videoParameters"]["numberOfSequentialFields"] == 1000
        assert stitcher.stitches == -(-1000 // parallel.SEGMENT_FIELDS) - 1
        assert stitcher.discarded == []

    def test_unmatched_segment_is_decoded_by_the_earlier_decoder(self, tmp_path):
        seg = parallel.SEGMENT_FIELDS
        # No two decoders agree anywhere in the third segment.
        bad = range(2 * seg, 3 * seg)
        tape = FakeTape(6 * seg, bad)
        stitcher, fields, meta, luma, chroma = _stitch(tmp_path, tape)
        assert fields == 6 * seg
        assert [f["fileLoc"] for f in meta["fields"]] == [n * SPF for n in range(6 * seg)]
        assert stitcher.discarded == [2 * seg]
        assert stitcher.stitches == 4
        for n in range(SETTLE, 6 * seg):
            if n in tape.bad:
                # The second segment's decoder carried on through the third.
                assert _luma_at(luma, n) == tape.field(seg, n)[0]
            else:
                assert _luma_at(luma, n) == _image(n)

    def test_single_worker_decodes_everything(self, tmp_path):
        tape = FakeTape(300)
        stitcher, fields, meta, luma, chroma = _stitch(tmp_path, tape, workers=1)
        assert fields == 300
        assert (luma, chroma) == _single_decode(tape, 0, 300)

    def test_requested_range(self, tmp_path):
        tape = FakeTape(1000)
        stitcher, fields, meta, luma, chroma = _stitch(tmp_path, tape, first_field=100, max_fields=500)
        assert fields == 500
        assert [f["fileLoc"] for f in meta["fields"]] == [n * SPF for n in range(100, 600)]
        assert (luma, chroma) == _single_decode(tape, 100, 500)

    def test_end_inside_a_segment(self, tmp_path):
        tape = FakeTape(parallel.SEGMENT_FIELDS * 3 + 7)
        stitcher, fields, meta, luma, chroma = _stitch(tmp_path, tape)
        assert fields == tape.length
        assert (luma, chroma) == _single_decode(tape, 0, tape.length)


class TestFieldsMatch:
    def _info(self, first=True):
        return {"isFirstField": first, "syncConf": 100}

    def test_small_differences_match(self):
        a = np.full(1000, 30000, dtype=np.uint16)
        b = a + 20
        assert parallel.fields_match(self._info(), [a.tobytes()], self._info(), [b.tobytes()], True)

    def test_dropouts_are_trimmed(self):
        a = np.full(1000, 30000, dtype=np.uint16)
        b = a.copy()
        b[:50] = 0
        assert parallel.field_difference(a.tobytes(), b.tobytes()) == 0.0
        assert parallel.fields_match(self._info(), [a.tobytes()], self._info(), [b.tobytes()], True)

    def test_different_fields_do_not_match(self):
        a = np.full(1000, 30000, dtype=np.uint16)
        b = a + 500
        assert not parallel.fields_match(self._info(), [a.tobytes()], self._info(), [b.tobytes()], True)
        assert not parallel.fields_match(self._info(), [a.tobytes()], self._info(False), [a.tobytes()], True)

    def test_chroma_ignored_when_framing_differs(self):
        y = np.full(1000, 30000, dtype=np.uint16).tobytes()
        c1 = np.full(1000, 1000, dtype=np.uint16).tobytes()
        c2 = np.full(1000, 9000, dtype=np.uint16).tobytes()
        assert not parallel.fields_match(self._info(), [y, c1], self._info(), [y, c2], True)
        assert parallel.fields_match(self._info(), [y, c1], self._info(), [y, c2], False)


class TestWorkerControl:
    def _wait_closed(self, control):
        deadline = time.time() + 5
        while not control._closed and time.time() < deadline:
            pass

    def test_limits(self):
        control = parallel.WorkerControl(io.StringIO("job 3 360 10\nlimit 3 20\nlimit 9 50\n"))
        assert control.next_job() == (3, 360)
        self._wait_closed(control)
        # Closed input stops every job, after the limits it raised.
        assert control._limits == {3: -1}
        assert not control.may_decode(3, 0)
        assert control.next_job() is None

    def test_stop_only_affects_its_segment(self):
        control = parallel.WorkerControl(io.StringIO("job 1 0 10\nstop 1\njob 2 120 10\nlimit 2 30\n"))
        self._wait_closed(control)
        assert control.next_job() == (1, 0)
        assert control.next_job() == (2, 120)


class TestWorkerCommand:
    def test_replaces_output_and_driver_args(self):
        raw = ["--pal", "--parallel", "4", "-s", "10", "--length=500", "in.flac", "out", "--overwrite"]
        cmd = parallel.worker_command(raw, "out", "work", 2)
        args = cmd[cmd.index("--overwrite"):]
        assert args == [
            "--overwrite", "--parallel_worker", "work",
            "-t", str(parallel.WORKER_THREADS), "--pal", "in.flac", os.path.join("work", "worker02"),
        ]

    def test_keeps_explicit_threads(self):
        cmd = parallel.worker_command(["-t", "3", "in", "out"], "out", "w", 0)
        assert cmd.count("-t") == 1 and "3" in cmd


class TestConcurrentWorkers:
    def test_limited_by_free_memory(self, monkeypatch):
        monkeypatch.setattr(parallel, "available_memory", lambda: 5 * parallel.WORKER_MEMORY_BYTES + 1)
        assert parallel.concurrent_workers(16) == 5

    def test_at_least_one_and_at_most_requested(self, monkeypatch):
        monkeypatch.setattr(parallel, "available_memory", lambda: 0)
        assert parallel.concurrent_workers(8) == 1
        monkeypatch.setattr(parallel, "available_memory", lambda: 10**15)
        assert parallel.concurrent_workers(8) == 8

    def test_unknown_memory_runs_all(self, monkeypatch):
        monkeypatch.setattr(parallel, "available_memory", lambda: None)
        assert parallel.concurrent_workers(8) == 8
