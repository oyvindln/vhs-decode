"""Optional CUDA acceleration using CuPy (enabled with --gpu).

Three pieces, each falling back to the CPU code when it can't be used:

- GPUResampler: does LoadResampled's polyphase resampling on the GPU, for a
  whole run of blocks at once.
- GPUDemod / DemodCacheGPU: demodulates a batch of blocks in one go instead of
  one block per worker thread, following VHSRFDecode.demodblock.
- GPUScaler: the per-pixel TBC resampling done by lddecode.utils.scale_field.

The two sosfiltfilt steps in demodblock (dropout envelope and colour-under
band-pass) are applied as zero-phase |H|^2 masks in the frequency domain
instead. Away from the block edges that DemodCache cuts off the result is the
same, but it is not bit-identical to the CPU path.
"""

import logging
import time

import numpy as np
import scipy.signal as sps

import lddecode.core as ldd
from lddecode.utils import LoadResampled, sinc_phase_count, sinc_tap_count, wow_level_adjusts
from vhsdecode.demod import replace_spikes
from vhsdecode.demodcache import DemodCacheTape

logger = logging.getLogger("vhs-decode")

try:
    import warnings

    import cupy as cp
    import cupyx

    with warnings.catch_warnings():
        # cupyx.scipy.signal imports the experimental cupyx.jit
        warnings.simplefilter("ignore", FutureWarning)
        import cupyx.scipy.signal as cpsignal

    _import_error = None
except Exception as e:  # pragma: no cover - depends on the machine
    cp = None
    cpsignal = None
    _import_error = e


def gpu_available():
    """Return (True, None) if a CUDA device can be used, else (False, reason)."""
    if cp is None:
        return False, f"CuPy could not be imported ({_import_error})"
    try:
        cp.cuda.runtime.getDeviceCount()
        cp.zeros(1).sum()
    except Exception as e:
        return False, f"no usable CUDA device ({e})"
    return True, None


# Mirrors unwrap_hilbert in src/lib.rs, including its float32 atan2
# approximation, so the GPU and CPU demodulators agree closely.
_UNWRAP_SRC = r"""
#include <cuda/std/cfloat>

__device__ __forceinline__ float atan_approx(float x) {
    const float A1 = 0.99997726f, A3 = -0.33262347f, A5 = 0.19354346f;
    const float A7 = -0.11643287f, A9 = 0.05265332f, A11 = -0.01172120f;
    float xs = x * x;
    return x * (A1 + xs * (A3 + xs * (A5 + xs * (A7 + xs * (A9 + xs * A11)))));
}

__device__ __forceinline__ float atan2f_fast(float y, float x) {
    const float PI = 3.14159265358979323846f;
    const float HALF_PI = 1.57079632679489661923f;
    x = x + copysignf(FLT_MIN, x);
    bool swap = fabsf(x) < fabsf(y);
    float in = (swap ? x : y) / (swap ? y : x);
    float res = atan_approx(in);
    float tmp = in >= 0.0f ? HALF_PI : -HALF_PI;
    res = swap ? tmp - res : res;
    if (x >= 0.0f && y >= 0.0f) return res;
    if (x < 0.0f && y >= 0.0f) return PI + res;
    if (x < 0.0f && y < 0.0f) return -PI + res;
    return res;
}

// in: [rows, n] complex128, out: [rows, n] float64. out[r, 0] = 0.
// If diff is set, demodulates the first difference of the input instead
// (np.ediff1d(in, to_begin=0)).
extern "C" __global__ void unwrap_hilbert(
    const double2* in, double* out, long long total, int n, float freq, int diff)
{
    const float TAU = 6.28318530717958647692f;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int col = (int)(i % n);
    if (col == 0) { out[i] = 0.0; return; }

    double2 a = in[i - 1];
    double2 b = in[i];
    if (diff) {
        double2 p = (col >= 2) ? in[i - 2] : make_double2(a.x, a.y);
        b = make_double2(b.x - a.x, b.y - a.y);
        a = (col >= 2) ? make_double2(a.x - p.x, a.y - p.y) : make_double2(0.0, 0.0);
    }
    float pa = atan2f_fast((float)a.y, (float)a.x);
    float pb = atan2f_fast((float)b.y, (float)b.x);
    float d = pb - pa;
    d = d - floorf(d / TAU) * TAU;
    out[i] = (double)(d * freq / TAU);
}
"""

# Mirrors the per-pixel loop of lddecode.utils.scale_field.
_SCALE_SRC = r"""
extern "C" __global__ void scale_field(
    const float* buf, long long buflen, float* dsout, long long outlen,
    const double* locs, const float* levels, const float* lut,
    long long dsout_start, double shift, int taps, int phases)
{
    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outlen) return;
    long long i = o + dsout_start;

    float coord = (float)(locs[i] + shift);
    long long coord_int = (long long)coord;
    float frac = coord - (float)coord_int;
    int phase = (int)(frac * (float)phases + 0.5f);
    const float* w = lut + (long long)phase * taps;
    long long start = coord_int - (taps / 2 - 1);

    double result = 0.0;
    for (int t = 0; t < taps; t++) {
        long long k = start + t;
        if (k >= 0 && k < buflen) result += (double)(buf[k] * w[t]);
    }
    dsout[o] = (float)((double)levels[i] * result);
}
"""

_kernels = {}


def _kernel(name, src):
    if name not in _kernels:
        _kernels[name] = cp.RawKernel(src, name, options=("--fmad=false",))
    return _kernels[name]


def _launch(kernel, total, args):
    threads = 256
    blocks = (int(total) + threads - 1) // threads
    kernel((blocks,), (threads,), args)


def _sos_power_response(sos, n):
    """|H|^2 of an sos filter on the rfft grid of an n-sample block, which is
    the frequency response of sosfiltfilt(sos, x)."""
    w = np.linspace(0, np.pi, n // 2 + 1)
    _, h = sps.sosfreqz(sos, worN=w)
    return np.abs(h) ** 2


class GPUResampler:
    """GPU version of LoadResampled for a run of blocks."""

    def __init__(self, loader):
        self.loader = loader
        self.filter = cp.asarray(loader.filter)

    def load_rows(self, infile, starts, blocklen):
        """Return (rows, cpu_rows): resampled blocks starting at each output
        sample in `starts` as a [len(starts), blocklen] float32 GPU array and a
        CPU copy, or (None, None) if the file ends before the last block."""
        L = self.loader
        up, down = L.up, L.down
        first, end = starts[0], starts[-1] + blocklen

        j = max((first - L.margin) // up, 0)
        in_start = j * down
        out_start = j * up
        in_needed = -(-(end + L.margin) * down // up) - in_start + 1

        itemsize = L.dtype.itemsize
        infile.seek(in_start * itemsize, 0)
        buf = infile.read(in_needed * itemsize)
        raw = np.frombuffer(buf, L.dtype, len(buf) // itemsize)
        if (in_start + len(raw)) * up < end * down:
            return None, None

        x = cp.asarray(raw).astype(cp.float32)
        if L.offset:
            x += L.offset
        if L.scale != 1:
            x *= L.scale

        y = cpsignal.resample_poly(x, up, down, window=self.filter)
        offsets = cp.asarray(np.asarray(starts) - out_start)
        idx = offsets[:, None] + cp.arange(blocklen)[None, :]
        rows = y[idx]
        return rows, cp.asnumpy(rows)


# Where each channel lives in the packed GPU copy of the demodulated data:
# (dtype to view the float64 array as, column)
GPU_CHANNELS = {
    "demod": (np.float64, 0),
    "demod_05": (np.float64, 1),
    "demod_burst": (np.float32, 4),
    "envelope": (np.float32, 5),
}


class GPUDemod:
    """Batched GPU version of VHSRFDecode.demodblock for the options it supports."""

    def __init__(self, rf):
        self.rf = rf
        self.n = rf.blocklen
        f = rf.Filters
        n = self.n

        self.rfvideo = cp.asarray(f["RFVideo"])
        self.hilbert = cp.asarray(f["hilbert"])
        self.fvideo = cp.asarray(f["FVideo"])
        self.fvideo05 = cp.asarray(f["FVideo05"])
        self.f05_offset = int(f["F05_offset"])
        self.env_mask = cp.asarray(_sos_power_response(f["FEnvPost"], n))
        burst = _sos_power_response(f["FVideoBurst"], n)
        if rf.options.color_under:
            self.chroma_mask = cp.asarray(burst)
        else:
            # chroma comes from the filtered video: fold FVideo into the mask
            self.chroma_mask = cp.asarray(burst * f["FVideo"])
        self.move = int(rf.options.chroma_offset)
        self.check_value = rf.options.diff_demod_check_value
        self.freq = np.float32(rf.freq_hz)
        self.cut0 = rf.blockcut
        self.cut1 = n - rf.blockcut_end
        self.dtype = np.dtype(
            [("demod", "<f8"), ("demod_05", "<f8"), ("demod_burst", "<f4"), ("envelope", "<f4")]
        )
        self._unwrap = _kernel("unwrap_hilbert", _UNWRAP_SRC)

    @staticmethod
    def unsupported_reason(rf):
        """Return why rf's options can't be demodulated on the GPU, or None."""
        o = rf.options
        checks = [
            (rf._notch is not None, "--notch"),
            (rf._high_boost is not None, "high boost"),
            (bool(rf._video_eq), "--sharpness / video EQ"),
            (bool(rf._chroma_trap), "--chroma_trap"),
            (o.nldeemp, "--nld"),
            (o.subdeemp, "--sub_deemphasis"),
            (rf._use_fsc_notch_filter, "fsc notch filter"),
            (rf._do_cafc, "--cafc"),
            (rf._disable_diff_demod, "--disable_diff_demod"),
            (rf.debug_plot is not None and bool(rf.debug_plot), "--debug_plot"),
            (rf.Filters.get("FChromaAudioNotch") is not None, "chroma audio notch"),
            (getattr(o, "gnrc_afe", False), "--gnrc_afe"),
        ]
        for bad, what in checks:
            if bad:
                return what
        return None

    def _unwrap_rows(self, hil, diff=False):
        rows, n = hil.shape
        out = cp.empty((rows, n), dtype=cp.float64)
        _launch(self._unwrap, rows * n,
                (hil, out, np.int64(rows * n), np.int32(n), self.freq, np.int32(diff)))
        return out

    def demod(self, rows):
        """rows: [B, blocklen] GPU array of raw samples. Returns a structured
        CPU array [B, blocklen - cuts] with the fields demodblock returns, and
        the same data on the GPU as float64 [B, blocklen - cuts, 3] (see
        GPU_CHANNELS)."""
        n = self.n
        x = rows.astype(cp.float64, copy=False)

        X = cp.fft.fft(x, axis=-1)
        if self.rf.options.color_under:
            raw_half = X[:, : n // 2 + 1].copy()
        X *= self.rfvideo
        hil = cp.fft.ifft(X * self.hilbert, axis=-1)
        del X

        # dropout envelope
        raw_env = cp.roll(cp.abs(hil.real).astype(cp.float32), 4, axis=-1)
        env = cp.fft.irfft(cp.fft.rfft(raw_env.astype(cp.float64), axis=-1) * self.env_mask,
                           n=n, axis=-1).astype(cp.float32)
        del raw_env
        if bool((env == 0).any()):
            ldd.logger.warning("RF signal is weak. Is your deck tracking properly?")

        demod = self._unwrap_rows(hil)

        # Where there are obviously out of range values, splice in a demod of
        # the differentiated signal (see VHSRFDecode.demodblock).
        spiky = cp.asnumpy(demod[:, 20:-20].max(axis=-1) > self.check_value)
        if spiky.any():
            idx = np.nonzero(spiky)[0]
            demod_b = self._unwrap_rows(hil[idx], diff=True)
            fixed = [replace_spikes(a, b, self.check_value)
                     for a, b in zip(cp.asnumpy(demod[idx]), cp.asnumpy(demod_b))]
            demod[idx] = cp.asarray(np.stack(fixed))
        del hil

        demod_fft = cp.fft.rfft(demod, axis=-1)
        del demod
        out_video = cp.fft.irfft(demod_fft * self.fvideo, n=n, axis=-1)
        out_05 = cp.roll(cp.fft.irfft(demod_fft * self.fvideo05, n=n, axis=-1),
                         -self.f05_offset, axis=-1)

        chroma_src = raw_half if self.rf.options.color_under else demod_fft
        chroma = cp.fft.irfft(chroma_src * self.chroma_mask, n=n, axis=-1)
        chroma = cp.roll(chroma, self.move, axis=-1)
        # DC estimate away from the block edges, as on the CPU
        dc0, dc1 = 2 * self.rf.blockcut, n - 2 * self.rf.blockcut_end
        chroma -= chroma[:, dc0:dc1].mean(axis=-1, keepdims=True)
        chroma = chroma.astype(cp.float32)

        # Pack the channels in the byte layout of self.dtype (two float64 and
        # two float32 per sample), so the CPU copy is a single download.
        c0, c1 = self.cut0, self.cut1
        packed = cp.empty((rows.shape[0], c1 - c0, 3), dtype=cp.float64)
        packed[:, :, 0] = out_video[:, c0:c1]
        packed[:, :, 1] = out_05[:, c0:c1]
        packed32 = packed.view(cp.float32)
        packed32[:, :, 4] = chroma[:, c0:c1]
        packed32[:, :, 5] = env[:, c0:c1]

        host = cupyx.empty_pinned(packed.shape, dtype=np.float64)
        packed.get(out=host)
        out = host.view(self.dtype)[:, :, 0]
        return out, packed


class DemodCacheGPU(DemodCacheTape):
    """DemodCache that loads and demodulates each doread() batch on the GPU."""

    def __init__(self, rf, infile, loader, rf_args, **kw):
        self.gpu_demod = GPUDemod(rf)
        self.gpu_resampler = GPUResampler(loader) if isinstance(loader, LoadResampled) else None
        # block number -> (request, packed GPU rows) - see GPUDemod.demod
        self.gpu_blocks = {}
        super().__init__(rf, infile, loader, rf_args, **kw)

    def prune_cache(self):
        super().prune_cache()
        with self.lock:
            for b in [b for b in self.gpu_blocks if b not in self.blocks]:
                del self.gpu_blocks[b]

    def read(self, begin, length, MTF=0, getraw=False, forceredo=False):
        rv = super().read(begin, length, MTF, getraw, forceredo)
        if rv is None or getraw:
            return rv

        # GPU copy of rv["video"], for GPUScaler
        parts = []
        with self.lock:
            for b in range(begin // self.blocksize, (begin + length) // self.blocksize + 1):
                entry = self.gpu_blocks.get(b)
                block = self.blocks.get(b)
                if entry is None or block is None or entry[0] != block.get("request"):
                    parts = None
                    break
                parts.append(entry[1])
        if parts:
            rv["video_gpu"] = cp.concatenate(parts)
        return rv

    def _load_rows(self, blocks):
        """Raw input for `blocks` as a GPU array, setting each block's
        rawinput. Returns (blocks that loaded, rows); stops at end of file."""
        blocklen = self.rf.blocklen
        have = [b for b in blocks if self.blocks[b]["rawinput"] is not None]
        need = [b for b in blocks if self.blocks[b]["rawinput"] is None]
        loaded, parts = [], []

        if have:
            loaded += have
            parts.append(cp.asarray(np.stack([self.blocks[b]["rawinput"] for b in have])))

        # split into runs of consecutive blocks
        runs = []
        for b in sorted(need):
            if runs and b == runs[-1][-1] + 1:
                runs[-1].append(b)
            else:
                runs.append([b])

        for run in runs:
            starts = [b * self.blocksize for b in run]
            rows = None
            if self.gpu_resampler is not None:
                with self.loader_lock:
                    rows, cpu_rows = self.gpu_resampler.load_rows(self.infile, starts, blocklen)
                got = run
            if rows is None:
                # no GPU loader, or the file ends within this run: read per block
                rows, cpu_rows, got = self._load_cpu(run)

            with self.lock:
                for i, b in enumerate(got):
                    self.blocks[b]["rawinput"] = cpu_rows[i]
            if got:
                loaded += got
                parts.append(rows)
            if len(got) < len(run):
                break  # end of file

        if not parts:
            return loaded, None
        return loaded, cp.concatenate(parts) if len(parts) > 1 else parts[0]

    def _load_cpu(self, run):
        cpu_rows = []
        ok = []
        for b in run:
            with self.loader_lock:
                raw = self.loader(self.infile, b * self.blocksize, self.rf.blocklen)
            if raw is None or len(raw) < self.rf.blocklen:
                break
            cpu_rows.append(np.asarray(raw, dtype=np.float32))
            ok.append(b)
        if not ok:
            return None, [], []
        return cp.asarray(np.stack(cpu_rows)), cpu_rows, ok

    def doread(self, blocknums, MTF, redo=False, prefetch=False):
        need_blocks = set()
        queuelist = []
        reached_end = False

        # Same bookkeeping as DemodCache.doread
        with self.lock:
            if redo:
                for b in self.flush_demod():
                    if b not in queuelist:
                        queuelist.append(b)

            for b in blocknums:
                if b not in self.blocks:
                    ldd.LRUupdate(self.lru, b)
                    self.blocks[b] = {}
                    self.blocks[b]["rawinput"] = None
                    if b not in queuelist:
                        queuelist.append(b)

                if self.blocks[b] is None:
                    reached_end = True
                    break

                waiting = self.blocks[b].get("waiting", False)

                if (
                    not redo
                    and not waiting
                    and "request" in self.blocks[b]
                    and "demod" in self.blocks[b]
                ):
                    continue

                if redo or not waiting:
                    if b not in queuelist:
                        queuelist.append(b)
                    need_blocks.add(b)
                elif waiting:
                    need_blocks.add(b)
                    if not prefetch:
                        self.waiting.add(b)

            queuelist = [b for b in queuelist if self.blocks.get(b) is not None]

        if queuelist and not reached_end:
            loaded, rows = self._load_rows(queuelist)
            missing = [b for b in queuelist if b not in loaded]
            if missing:
                with self.lock:
                    for b in missing:
                        if prefetch:
                            self.blocks.pop(b, None)
                        else:
                            self.blocks[b] = None
                if not prefetch:
                    return None
                reached_end = True

            if loaded:
                with self.lock:
                    for b in loaded:
                        self.blocks[b]["MTF"] = MTF
                        self.blocks[b]["request"] = self.request
                        self.blocks[b]["waiting"] = True
                        self.blocks[b]["prefetch"] = prefetch
                        if not prefetch:
                            self.waiting.add(b)
                    self.q_in.put(("DEMOD_BATCH", loaded, rows, self.request))

        return None if reached_end else need_blocks

    def worker(self, return_on_empty=False):
        while True:
            if return_on_empty and self.q_in.qsize() == 0:
                return

            item = self.q_in.get()

            if item is None or item[0] == "END":
                return

            if item[0] == "DEMOD_BATCH":
                blocknums, rows, request = item[1:]
                out, packed = self.gpu_demod.demod(rows)
                with self.lock:
                    for i, b in enumerate(blocknums):
                        self.gpu_blocks[b] = (request, packed[i])
                for i, b in enumerate(blocknums):
                    video = out[i].view(np.recarray)
                    self.q_out.put((b, {"demod": {"video": video}, "request": request, "MTF": 0}))
            elif item[0] == "NEWPARAMS":
                self.apply_newparams(item[1])


class GPUScaler:
    """GPU version of lddecode.utils.scale_field."""

    def __init__(self, sinc_lut):
        assert sinc_lut.shape[1] == sinc_tap_count
        self.lut = cp.asarray(sinc_lut, dtype=cp.float32)
        self._kernel = _kernel("scale_field", _SCALE_SRC)

    def _levels(self, field, locs, wowfactors, outwidth):
        """Pixel locations and level adjustments on the GPU, cached per field
        for as long as computewow_scaled returns the same arrays. The level
        adjustments involve a sequential smoothing filter, so they are computed
        on the CPU with the same code scale_field uses."""
        cached = getattr(field, "_gpu_wow", None)
        if cached is not None and cached[0] is locs and cached[1] is wowfactors:
            return cached[2], cached[3]

        levels = wow_level_adjusts(wowfactors, outwidth, field.wow_level_adjust_smoothing)
        locs_gpu = cp.asarray(locs)
        levels_gpu = cp.asarray(levels.astype(np.float32))
        field._gpu_wow = (locs, wowfactors, locs_gpu, levels_gpu)
        return locs_gpu, levels_gpu

    def _channel(self, field, channel, buf):
        """The channel's data as contiguous float32 on the GPU, from the GPU
        copy the demod cache made if there is one, cached per field."""
        cache = field.__dict__.setdefault("_gpu_channels", {})
        if channel in cache:
            return cache[channel]
        packed = field.data.get("video_gpu")
        if packed is not None and channel in GPU_CHANNELS and len(packed) == len(buf):
            dtype, col = GPU_CHANNELS[channel]
            data = packed.view(dtype)[:, col].astype(cp.float32)
        else:
            data = cp.asarray(np.asarray(buf, dtype=np.float32))
        cache[channel] = data
        return data

    def scale_field(self, field, channel, buf, dsout, locs, wowfactors, lineoffset, outwidth, shift=0.0):
        locs_gpu, levels = self._levels(field, locs, wowfactors, outwidth)
        buf_gpu = self._channel(field, channel, buf)
        out = cp.empty(len(dsout), dtype=cp.float32)
        _launch(self._kernel, len(dsout), (
            buf_gpu, np.int64(len(buf)), out, np.int64(len(dsout)),
            locs_gpu, levels, self.lut,
            np.int64(outwidth * (lineoffset + 1)), np.float64(shift),
            np.int32(sinc_tap_count), np.int32(sinc_phase_count),
        ))
        out.get(out=dsout)


def setup(rf):
    """Check whether the GPU can be used for this decode. Returns a reason
    string if not, else None, and sets rf.gpu_scaler."""
    ok, reason = gpu_available()
    if not ok:
        return reason
    reason = GPUDemod.unsupported_reason(rf)
    if reason is not None:
        return f"{reason} is not supported on the GPU yet"
    rf.gpu_scaler = GPUScaler(rf.downscale_sinc_lut)
    return None
