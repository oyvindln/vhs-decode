"""Parallel decoding.

Decoding is serial: each field's levels and phase build on the fields before
it. To use more cores, worker processes decode segments of the capture that
start SEGMENT_FIELDS fields apart, and their output is stitched into a single
stream. A decoder started mid-capture settles within a few fields onto the
output a single decode gives for that part of the capture.

The earlier decoder is authoritative. It decodes on into the next segment,
and once STITCH_FIELDS consecutive fields agree with the next decoder's, the
next decoder, which has already decoded ahead, takes over. If no fields agree
within the next segment, that segment's work is discarded and the earlier
decoder carries on through it, so a decode never fails because two segments
could not be stitched; it only loses speed there.

Every stitch decision depends only on the input and the options, never on
timing, so the output is the same on every run. The scheme follows the
multithreaded decode of tape-decode (https://github.com/namazso/tape-decode-rs).
"""

import collections
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from vhsdecode.formats import parent_system

# Fields between the starts of consecutive segments, and the width of the
# overlap searched for a stitch. A multiple of 4 so that NTSC colour framing,
# which counts fields from where a decoder starts, lines up between segments.
SEGMENT_FIELDS = 120
# Consecutive agreeing fields needed to stitch one decoder onto the next.
STITCH_FIELDS = 2
# Per-field root mean square difference, in 16-bit output steps, below which
# two decodes of a field are taken to agree.
MATCH_THRESHOLD = 64.0
# Share of the largest per-sample differences ignored when comparing fields,
# so that dropouts handled differently by two decoders do not stop a stitch.
TRIM_FRACTION = 0.10
# Fields a decoder may run ahead of where the stitcher needs it.
RUN_AHEAD_FIELDS = 4
# Thread count given to each worker process when -t is not set explicitly.
WORKER_THREADS = 2

# Arguments the driver sets itself for each worker: (flags, takes a value).
_DRIVER_ARGS = (
    (("--parallel",), True),
    (("--parallel_worker",), True),
    (("-s", "--start"), True),
    (("-l", "--length"), True),
    (("--overwrite",), False),
)

_RAW_SAMPLE_BYTES = {
    ".u8": 1, ".r8": 1, ".s8": 1,
    ".u16": 2, ".r16": 2, ".s16": 2, ".raw": 2, ".tbc": 2,
    ".rf": 4,
}

_PROGRESS_INTERVAL = 5.0
_PREFETCH_BLOCK = 8 * 1024 * 1024

# Memory committed by one worker process at its peak, with some headroom.
WORKER_MEMORY_BYTES = 3 * 1024**3
# Free space on the output drive below which the decode pauses, checked every
# DISK_CHECK_FIELDS fields.
MIN_FREE_BYTES = 10 * 1024**3
DISK_CHECK_FIELDS = 500


def available_memory():
    """Memory that new processes can still commit, in bytes, or None if unknown."""
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        # Windows refuses allocations past the commit limit, not past free RAM.
        return status.ullAvailPageFile
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def concurrent_workers(workers):
    """How many worker processes to run given the free memory."""
    free = available_memory()
    if free is None:
        return workers
    return max(1, min(workers, free // WORKER_MEMORY_BYTES))


class ParallelError(Exception):
    pass


def unsupported_reason(args):
    """Return why args cannot be decoded in parallel, or None if they can."""
    if args.infile == "-":
        return "input from stdin"
    if args.start_fileloc != -1:
        return "--start_fileloc"
    if args.debug_plot:
        return "--debug_plot"
    if args.gnrc_afe:
        return "--gnrc"
    if args.write_db:
        return "--write_db"
    return None


def count_input_samples(filename):
    """Return the number of RF samples in filename, or None if unknown."""
    ext = os.path.splitext(filename)[1].lower()
    size = os.path.getsize(filename)

    if ext in _RAW_SAMPLE_BYTES:
        return size // _RAW_SAMPLE_BYTES[ext]
    if ext == ".lds":
        return (size // 5) * 4
    if ext == ".r30":
        return (size // 3) * 2

    try:
        import av
    except ImportError:
        return None

    # The sample count in RF capture headers is not reliable, and a capture
    # cut from a longer one keeps its original frame numbers, so measure from
    # where the first audio frame starts to where the last one ends.
    with av.open(filename) as container:
        stream = container.streams.audio[0]
        if stream.time_base is None:
            return None
        to_samples = stream.time_base * stream.sample_rate
        start = None
        for frame in container.decode(stream):
            start = int(frame.pts * to_samples)
            break
        container.seek(2**62, stream=stream)
        end = None
        for frame in container.decode(stream):
            end = int(frame.pts * to_samples) + frame.samples
    if start is None or end is None:
        return None
    return end - start


def _frames_per_second(system):
    return 25.0 if parent_system(system) == "PAL" else 30000 / 1001


def _strip_driver_args(raw_args):
    out = []
    i = 0
    while i < len(raw_args):
        token = raw_args[i]
        matched = False
        for flags, takes_value in _DRIVER_ARGS:
            for flag in flags:
                if token == flag:
                    i += 2 if takes_value else 1
                    matched = True
                elif takes_value and token.startswith(flag + "=") and flag.startswith("--"):
                    i += 1
                    matched = True
                if matched:
                    break
            if matched:
                break
        if not matched:
            out.append(token)
            i += 1
    return out


def _has_thread_arg(raw_args):
    return any(
        t in ("-t", "--threads") or t.startswith("--threads=") for t in raw_args
    )


def worker_command(raw_args, outfile, work_dir, index):
    """Build the command line that starts worker process index."""
    args = _strip_driver_args(raw_args)

    # The output path is the last positional token equal to outfile.
    for i in range(len(args) - 1, -1, -1):
        if args[i] == outfile:
            args[i] = os.path.join(work_dir, f"worker{index:02d}")
            break
    else:
        raise ParallelError("could not find the output path in the arguments")

    if not _has_thread_arg(raw_args):
        args = ["-t", str(WORKER_THREADS)] + args
    args = ["--overwrite", "--parallel_worker", work_dir] + args

    if getattr(sys, "frozen", False):
        return [sys.executable, "vhs"] + args
    return [sys.executable, "-c", "from vhsdecode.main import main; main()"] + args


def _segment_path(work_dir, index):
    return os.path.join(work_dir, f"seg{index:06d}")


def _output_suffixes(path):
    if os.path.exists(path + ".tbcy"):
        return [".tbcy", ".tbcc"]
    if os.path.exists(path + "_chroma.tbc"):
        return [".tbc", "_chroma.tbc"]
    return [".tbc"]


# --- Worker process -----------------------------------------------------------
#
# A worker reads commands from stdin, one per line:
#   job <segment> <start field> <field limit>   decode a segment
#   limit <segment> <field limit>               let it decode further
#   stop <segment>                              end it
# It writes each segment's fields to <segment path>.tbc (and chroma), with one
# JSON message per line in <segment path>.fields. A field's line is written
# only after its picture data has been flushed.


class WorkerControl:
    def __init__(self, stream):
        self._cond = threading.Condition()
        self._jobs = collections.deque()
        self._limits = {}
        self._closed = False
        threading.Thread(target=self._read, args=(stream,), daemon=True).start()

    def _read(self, stream):
        for line in stream:
            words = line.split()
            if len(words) < 2:
                continue
            with self._cond:
                cmd, seg = words[0], int(words[1])
                if cmd == "job":
                    self._jobs.append((seg, int(words[2])))
                    self._limits[seg] = int(words[3])
                elif seg in self._limits:
                    if cmd == "stop":
                        self._limits[seg] = -1
                    elif self._limits[seg] >= 0:
                        self._limits[seg] = max(self._limits[seg], int(words[2]))
                self._cond.notify_all()
        with self._cond:
            self._closed = True
            for seg in self._limits:
                self._limits[seg] = -1
            self._cond.notify_all()

    def next_job(self):
        with self._cond:
            while not self._jobs and not self._closed:
                self._cond.wait()
            return self._jobs.popleft() if self._jobs else None

    def may_decode(self, seg, written):
        """Wait until seg may decode past written fields; False once it is stopped."""
        with self._cond:
            while 0 <= self._limits[seg] <= written:
                self._cond.wait()
            return self._limits[seg] >= 0

    def finish(self, seg):
        with self._cond:
            self._limits.pop(seg, None)


def _decode_segment(work_dir, seg, start_field, make_decoder, control):
    path = _segment_path(work_dir, seg)
    with open(path + ".fields", "w", encoding="utf-8") as meta:

        def send(msg):
            meta.write(json.dumps(msg, allow_nan=False, separators=(",", ":")) + "\n")
            meta.flush()

        decoder = None
        try:
            decoder = make_decoder(path)
            decoder.roughseek(start_field)
            send({
                "start": int(decoder.fdoffset),
                "samplesPerField": int(decoder.bytes_per_field),
                "suffixes": _output_suffixes(path),
                # NTSC colour framing counts fields from where a decoder
                # started, so two decoders can disagree on chroma there.
                "chromaFraming": decoder.rf.color_system == "NTSC"
                and not decoder.rf.options.disable_phase_correction,
            })
            written = 0
            while control.may_decode(seg, written):
                field = decoder.readfield()
                if field is None:
                    break
                field.prevfield = None
                infos = decoder.fieldinfo.read()
                if not infos:
                    continue
                decoder.outfile_video.flush()
                if decoder.outfile_chroma is not None:
                    decoder.outfile_chroma.flush()
                if written == 0:
                    send({"fieldBytes": decoder.outfile_video.tell() // len(infos)})
                for info in infos:
                    meta.write(json.dumps({"field": info}, allow_nan=False) + "\n")
                meta.flush()
                written += len(infos)
            params = decoder.build_json() if written else None
        except Exception as e:
            send({"error": f"{type(e).__name__}: {e}"})
            raise
        finally:
            if decoder is not None:
                decoder.close()
        send({"end": True, "json": params})


def run_worker(work_dir, make_decoder):
    """Decode the segments the parallel driver sends on stdin."""
    control = WorkerControl(sys.stdin)
    while True:
        job = control.next_job()
        if job is None:
            return 0
        seg, start_field = job
        try:
            _decode_segment(work_dir, seg, start_field, make_decoder, control)
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            control.finish(seg)
        # A decoder is held in reference cycles, so free it before the next
        # segment instead of waiting for the collector to get round to it.
        gc.collect()


# --- Field comparison -----------------------------------------------------------


def field_difference(a, b, trim_fraction=TRIM_FRACTION):
    """Root mean square difference of two 16-bit fields, leaving out the
    largest trim_fraction of the per-sample differences."""
    x = np.frombuffer(a, dtype=np.uint16).astype(np.int64)
    y = np.frombuffer(b, dtype=np.uint16).astype(np.int64)
    squared = (x - y) ** 2
    keep = len(squared) - int(len(squared) * trim_fraction)
    if keep <= 0:
        return 0.0
    if keep < len(squared):
        squared = np.partition(squared, keep - 1)[:keep]
    return float(np.sqrt(squared.mean()))


def fields_match(info_a, data_a, info_b, data_b, compare_chroma):
    """Whether two decodes of a field agree well enough to stitch there.

    data_a and data_b hold the field's luma, then its chroma if there is any.
    """
    if info_a["isFirstField"] != info_b["isFirstField"]:
        return False
    if info_a.get("syncConf") != info_b.get("syncConf"):
        return False
    channels = len(data_a) if compare_chroma else 1
    return all(
        len(data_a[n]) == len(data_b[n])
        and field_difference(data_a[n], data_b[n]) < MATCH_THRESHOLD
        for n in range(channels)
    )


# --- Driver -----------------------------------------------------------------------


class Segment:
    """The output of one worker decoding one segment, read as it is written."""

    def __init__(self, index, start_field, worker, path):
        self.index = index
        self.start_field = start_field
        self.worker = worker
        self.path = path
        self.fields = []
        # Index of the next field the stitcher has not taken yet.
        self.pos = 0
        self.limit = 0
        self.start_sample = None
        self.samples_per_field = None
        self.suffixes = None
        self.chroma_framing = False
        self.field_bytes = None
        self.ended = False
        self.params = None
        self.retired = False
        self._meta = None
        self._partial = ""
        self._data = None

    def poll(self):
        """Read new messages from the worker; True if any arrived."""
        if self._meta is None:
            try:
                self._meta = open(self.path + ".fields", encoding="utf-8")
            except FileNotFoundError:
                return False
        got = False
        while True:
            line = self._meta.readline()
            if not line:
                return got
            if not line.endswith("\n"):
                self._partial += line
                return got
            line, self._partial = self._partial + line, ""
            msg = json.loads(line)
            got = True
            if "field" in msg:
                self.fields.append(msg["field"])
            elif "start" in msg:
                self.start_sample = msg["start"]
                self.samples_per_field = msg["samplesPerField"]
                self.suffixes = msg["suffixes"]
                self.chroma_framing = msg["chromaFraming"]
            elif "fieldBytes" in msg:
                self.field_bytes = msg["fieldBytes"]
            elif "end" in msg:
                self.ended = True
                self.params = msg["json"]
            elif "error" in msg:
                raise ParallelError(f"segment {self.index} failed: {msg['error']}")

    def data(self, i):
        if self._data is None:
            self._data = [open(self.path + s, "rb") for s in self.suffixes]
        out = []
        for f in self._data:
            f.seek(i * self.field_bytes)
            out.append(f.read(self.field_bytes))
        return out

    def remove_files(self):
        """Delete the segment's files; False if the worker still has them open."""
        for f in self._data or ():
            f.close()
        self._data = None
        if self._meta is not None:
            self._meta.close()
            self._meta = None
        try:
            for s in (self.suffixes or []) + [".fields"]:
                if os.path.exists(self.path + s):
                    os.remove(self.path + s)
        except PermissionError:
            return False
        return True


class WorkerProcess:
    def __init__(self, cmd, log_path):
        self.log_path = log_path
        self._log = open(log_path, "w")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.segment = None

    def send(self, *words):
        try:
            self.proc.stdin.write(" ".join(str(w) for w in words) + "\n")
            self.proc.stdin.flush()
        except OSError:
            pass  # The driver notices that the worker has exited and reports it.

    def failed(self):
        return self.proc.poll() is not None

    def close(self, kill=False):
        if kill:
            self.proc.kill()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._log.close()


class Prefetcher:
    """Reads the input ahead of the workers in one forward pass, so that they
    find it in the OS cache instead of each seeking around the disk."""

    def __init__(self, filename, start):
        self._filename = filename
        self._cond = threading.Condition()
        self._pos = start
        self._target = start
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def read_to(self, byte):
        with self._cond:
            if byte > self._target:
                self._target = byte
                self._cond.notify()

    def _run(self):
        try:
            with open(self._filename, "rb", buffering=0) as f:
                f.seek(self._pos)
                buf = bytearray(_PREFETCH_BLOCK)
                while True:
                    with self._cond:
                        while self._pos >= self._target and not self._stop:
                            self._cond.wait()
                        if self._stop:
                            return
                        n = min(len(buf), self._target - self._pos)
                    got = f.readinto(memoryview(buf)[:n])
                    if not got:
                        return
                    self._pos += got
        except OSError:
            pass

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify()


class Stitcher:
    """Assigns segments to workers and stitches their output in field order.

    prefetch is (Prefetcher, input bytes per field) or None.
    """

    def __init__(self, workers, work_dir, first_field, max_fields, outname,
                 prefetch=None, total_frames=None, log=print):
        self.workers = workers
        self.work_dir = work_dir
        self.first_field = first_field
        self.max_fields = max_fields
        self.outname = outname
        self.prefetch = prefetch
        self.total_frames = total_frames
        self.log = log
        self.segments = {}
        self.next_unassigned = 0
        # First segment known to start past the end of the input.
        self.end_segment = None
        if max_fields is not None:
            self.end_segment = -(-max_fields // SEGMENT_FIELDS)
        self.committed = 0
        self.stitches = 0
        self.discarded = []
        self.started = time.time()
        self._current = 0
        self._finishing = False
        self._last_progress = 0.0
        self._out = None
        self._fields_json = []
        self._final = None
        self._removing = []

    # Segments and workers.

    def _past_end(self, seg):
        return self.end_segment is not None and seg >= self.end_segment

    def _refill(self, next_seg):
        if self._finishing:
            return
        progress = [
            s.start_field + len(s.fields) for s in self.segments.values() if not s.ended
        ]
        budget = len(self.workers) * SEGMENT_FIELDS
        for worker in self.workers:
            if worker.segment is not None:
                continue
            seg = self.next_unassigned
            if self._past_end(seg):
                return
            start = self.first_field + seg * SEGMENT_FIELDS
            # Look-ahead segments stay within a budget of the slowest decoder,
            # so a run of failed stitches cannot let them run off.
            if seg > next_seg and progress and start > min(progress) + budget:
                return
            segment = Segment(seg, start, worker, _segment_path(self.work_dir, seg))
            segment.limit = SEGMENT_FIELDS + RUN_AHEAD_FIELDS
            self.segments[seg] = segment
            worker.segment = segment
            worker.send("job", seg, start, segment.limit)
            self.next_unassigned += 1

    def _retire(self, seg):
        segment = self.segments.pop(seg)
        segment.retired = True
        if not segment.ended:
            segment.worker.send("stop", seg)
        self._removing.append(segment)

    def _update_limits(self, draining):
        for segment in self.segments.values():
            if segment.ended:
                continue
            if segment.index != self._current:
                want = SEGMENT_FIELDS + RUN_AHEAD_FIELDS
            elif draining:
                want = 1 << 40
            else:
                # The current decoder runs freely through its own segment,
                # then only as far ahead as the stitcher has got.
                want = max(SEGMENT_FIELDS, segment.pos) + RUN_AHEAD_FIELDS
            if want > segment.limit:
                segment.limit = want
                segment.worker.send("limit", segment.index, want)

    def _pump(self, next_seg, draining=False):
        got = False
        for segment in list(self.segments.values()) + self._removing:
            if segment.ended:
                continue
            got |= segment.poll()
            if segment.ended:
                if segment.worker.segment is segment:
                    segment.worker.segment = None
                if not segment.fields and not segment.retired and not self._past_end(segment.index):
                    # A segment that decodes nothing starts past the input.
                    self.end_segment = segment.index
        self._removing = [s for s in self._removing if not (s.ended and s.remove_files())]
        for n, worker in enumerate(self.workers):
            if worker.failed():
                raise ParallelError(f"worker {n} exited; see {worker.log_path}")
        if not self._finishing:
            self._update_limits(draining)
            self._refill(next_seg)
        if self.prefetch is not None:
            prefetcher, bytes_per_field = self.prefetch
            furthest = max(
                (s.start_field + s.limit for s in self.segments.values()), default=0
            )
            furthest = min(furthest, self.first_field + self.next_unassigned * SEGMENT_FIELDS)
            prefetcher.read_to(int((furthest + 2 * SEGMENT_FIELDS) * bytes_per_field))
        self._progress()
        if not got:
            time.sleep(0.005)

    def _progress(self, final=False):
        now = time.time()
        if not final and now - self._last_progress < _PROGRESS_INTERVAL:
            return
        self._last_progress = now
        frames = self.committed // 2
        total = f"/{self.total_frames}" if self.total_frames else ""
        fps = frames / max(now - self.started, 1e-6)
        print(f"\r{frames}{total} frames ({fps:.2f} FPS)", end="", flush=True)

    def _peek(self, seg, next_seg, draining=False):
        """The next field of seg, waiting for it; None once seg has no more."""
        segment = self.segments[seg]
        while segment.pos >= len(segment.fields) and not segment.ended:
            self._pump(next_seg, draining)
        return segment.fields[segment.pos] if segment.pos < len(segment.fields) else None

    def _wait_for_start(self, seg, next_seg):
        segment = self.segments[seg]
        while segment.start_sample is None and not segment.ended:
            self._pump(next_seg)
        return segment.start_sample is not None

    def _next_exists(self, next_seg):
        if len(self.workers) < 2:
            return False
        while next_seg not in self.segments:
            if self._past_end(next_seg):
                return False
            self._pump(next_seg)
        return self._wait_for_start(next_seg, next_seg)

    # Output.

    def _commit(self, seg):
        """Write the next field of seg; True once the requested length is reached."""
        segment = self.segments[seg]
        index = segment.pos
        segment.pos += 1
        if self._out is None:
            self._out = [open(self.outname + s, "wb") for s in segment.suffixes]
        for f, data in zip(self._out, segment.data(index)):
            f.write(data)
        info = dict(segment.fields[index])
        info["seqNo"] = self.committed + 1
        self._fields_json.append(json.dumps(info, allow_nan=False, separators=(",", ":")))
        self.committed += 1
        self._final = segment
        if self.committed % DISK_CHECK_FIELDS == 0:
            self._wait_for_disk_space()
        return self.max_fields is not None and self.committed >= self.max_fields

    def _wait_for_disk_space(self):
        output_dir = os.path.dirname(os.path.abspath(self.outname))
        try:
            if shutil.disk_usage(output_dir).free >= MIN_FREE_BYTES:
                return
            print(
                "\nLess than 10GB of free disk space is remaining, decoding paused. Decoding"
                " will resume once there is more space, or press Ctrl+C to exit.",
                file=sys.stderr,
            )
            while shutil.disk_usage(output_dir).free < MIN_FREE_BYTES:
                time.sleep(1)
            print("\nDisk space available, resuming decode.", file=sys.stderr)
        except OSError:
            pass

    def _matches_next(self, seg, next_seg):
        """Compare the next field of seg with the next decoder's field at the
        same place in the input, taking the next decoder's fields up to it."""
        segment = self.segments[seg]
        info = segment.fields[segment.pos]
        nxt = self.segments[next_seg]
        tol = segment.samples_per_field // 2
        while (other := self._peek(next_seg, next_seg)) is not None:
            if other["fileLoc"] + tol >= info["fileLoc"]:
                break
            nxt.pos += 1
        if other is None or abs(other["fileLoc"] - info["fileLoc"]) > tol:
            return False
        nxt.pos += 1
        return fields_match(
            info,
            segment.data(segment.pos),
            other,
            nxt.data(nxt.pos - 1),
            compare_chroma=not segment.chroma_framing,
        )

    def run(self):
        """Stitch the whole decode; returns the number of fields written."""
        self._refill(1)
        if 0 not in self.segments or not self._wait_for_start(0, 1):
            return 0
        try:
            self._stitch()
        finally:
            for seg in list(self.segments):
                self._retire(seg)
            for f in self._out or ():
                f.close()
        return self.committed

    def _stitch(self):
        seg, next_seg = 0, 1
        while True:
            # With no later segment, the current decoder decodes to the end.
            if not self._next_exists(next_seg):
                while self._peek(seg, next_seg, draining=True) is not None:
                    if self._commit(seg):
                        return
                return

            region_start = self.segments[next_seg].start_sample
            region_end = region_start + SEGMENT_FIELDS * self.segments[next_seg].samples_per_field

            # Commit the current decoder's fields up to the next segment.
            while (info := self._peek(seg, next_seg)) is not None and info["fileLoc"] < region_start:
                if self._commit(seg):
                    return
            if info is None:
                return

            # Inside the next segment, keep committing the current decoder's
            # fields while comparing them with the next decoder's.
            run = 0
            while (info := self._peek(seg, next_seg)) is not None and info["fileLoc"] < region_end:
                matched = self._matches_next(seg, next_seg)
                if self._commit(seg):
                    return
                run = run + 1 if matched else 0
                if run >= STITCH_FIELDS:
                    break
            if info is None:
                return

            if run >= STITCH_FIELDS:
                # The next decoder is already past the matched fields.
                self._retire(seg)
                self.stitches += 1
                seg = self._current = next_seg
            else:
                discarded = self.segments[next_seg]
                if discarded.fields:
                    self.discarded.append(discarded.start_field)
                    self.log(
                        f"\nNo {STITCH_FIELDS}-field match within {SEGMENT_FIELDS} fields of "
                        f"field {discarded.start_field}; the earlier decoder carries on through it"
                    )
                self._retire(next_seg)
            next_seg += 1
            self._refill(next_seg)

    def finish(self):
        """Wait for the workers to wind down; return the final decoder's metadata."""
        self._finishing = True
        while any(not s.ended for s in self._removing):
            self._pump(self.next_unassigned)
        self._removing = [s for s in self._removing if not s.remove_files()]
        return self._final.params if self._final is not None else None

    def write_json(self, params):
        params = dict(params)
        params.pop("fields", None)
        params["videoParameters"] = dict(params["videoParameters"])
        params["videoParameters"]["numberOfSequentialFields"] = self.committed
        head = json.dumps(params, allow_nan=False, separators=(",", ":"))
        with open(self.outname + ".tbc.json.tmp", "w") as f:
            f.write(head[:-1] + ',"fields":[' + ",".join(self._fields_json) + "]}")
        os.replace(self.outname + ".tbc.json.tmp", self.outname + ".tbc.json")


def _write_log(outname, work_dir, workers, summary):
    with open(outname + ".log", "w", encoding="utf-8", errors="replace") as log:
        for n, worker in enumerate(workers):
            for path in (os.path.join(work_dir, f"worker{n:02d}.log"), worker.log_path):
                log.write(f"===== {os.path.basename(path)} =====\n")
                try:
                    with open(path, encoding="utf-8", errors="replace") as src:
                        shutil.copyfileobj(src, log)
                except FileNotFoundError:
                    pass
        log.write(summary + "\n")


def run_parallel(args, raw_args, system, input_freq_mhz):
    """Decode args.infile with args.parallel worker processes."""
    reason = unsupported_reason(args)
    if reason:
        print(f"--parallel cannot be used with {reason}", file=sys.stderr)
        return 1

    fps = _frames_per_second(system)
    max_fields = args.length * 2 if args.length < 99999999 else None
    samples = count_input_samples(args.infile)
    total_frames = None
    bytes_per_field = None
    if samples is not None:
        fields_in_file = samples / (input_freq_mhz * 1e6) * fps * 2
        total_frames = int(fields_in_file // 2) - args.start
        if max_fields is not None:
            total_frames = min(total_frames, args.length)
        if total_frames <= 0:
            print("Nothing to decode in the requested range", file=sys.stderr)
            return 1
        bytes_per_field = os.path.getsize(args.infile) / max(fields_in_file, 1)

    outname = args.outfile
    work_dir = outname + ".parallel"
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    count = concurrent_workers(args.parallel)
    if count < args.parallel:
        print(f"Not enough free memory for {args.parallel} decode processes; using {count}")
    print(f"Decoding with {count} processes")

    workers = []
    prefetcher = None
    try:
        for n in range(count):
            cmd = worker_command(raw_args, args.outfile, work_dir, n)
            workers.append(WorkerProcess(cmd, os.path.join(work_dir, f"worker{n:02d}.stdout")))
        prefetch = None
        if bytes_per_field is not None:
            prefetcher = Prefetcher(args.infile, int(args.start * 2 * bytes_per_field))
            prefetch = (prefetcher, bytes_per_field)
        stitcher = Stitcher(
            workers, work_dir, args.start * 2, max_fields, outname, prefetch, total_frames
        )
        fields = stitcher.run()
        params = stitcher.finish()
        stitcher._progress(final=True)
        print()
    except KeyboardInterrupt:
        for worker in workers:
            worker.close(kill=True)
        print("\nTerminated")
        return 1
    except ParallelError as e:
        for worker in workers:
            worker.close(kill=True)
        print(f"\n{e}; worker output is kept in {work_dir}", file=sys.stderr)
        return 1
    finally:
        if prefetcher is not None:
            prefetcher.close()

    for worker in workers:
        worker.close()
    if not fields or params is None:
        print("Completed without handling any frames.", file=sys.stderr)
        return 1
    stitcher.write_json(params)
    elapsed = time.time() - stitcher.started
    summary = (
        f"Parallel decode: {stitcher.stitches} stitches, "
        f"{len(stitcher.discarded)} segments discarded"
    )
    _write_log(outname, work_dir, workers, summary)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(summary)
    print(f"Took {elapsed:.2f} seconds to decode {fields // 2} frames ({fields / 2 / elapsed:.2f} FPS)")
    return 0
