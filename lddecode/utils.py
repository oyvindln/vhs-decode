# A collection of helper functions used in dev notebooks and lddecode_core.py

from collections import namedtuple
import itertools
import json
import math
import os
import subprocess
import sys
import traceback
import signal

from multiprocessing import Event, Pipe, Process

from numba import jit, njit
import numba

# standard numeric/scientific libraries
import numpy as np
import scipy.signal as sps
from scipy import interpolate

# Try to make sure ffmpeg is available
try:
    import static_ffmpeg
except ImportError:
    pass

# If profiling is not enabled, make it a pass-through wrapper
try:
    profile
except:
    def profile(fn):
        return fn

# This runs a cubic scaler on a line.
# originally from https://www.paulinternet.nl/?page=bicubic
@njit(nogil=True, cache=True)
def scale(buf, begin, end, tgtlen, mult=1):
    linelen = end - begin
    sfactor = linelen / tgtlen

    output = np.zeros(tgtlen, dtype=buf.dtype)

    for i in range(0, tgtlen):
        coord = (i * sfactor) + begin
        start = int(coord) - 1
        p = buf[start : start + 4]
        x = coord - int(coord)

        output[i] = mult * (
            p[1]
            + 0.5
            * x
            * (
                p[2]
                - p[0]
                + x
                * (
                    2.0 * p[0]
                    - 5.0 * p[1]
                    + 4.0 * p[2]
                    - p[3]
                    + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])
                )
            )
        )

    return output


# Scales and compensates for wow-induced playback-speed variations
@njit(nogil=True, cache=True, fastmath=True)
def scale_field(buf, dsout, interpolated_pixel_locs, wowfactors, lineoffset, outwidth, level_smoothing_lines=0, level_adjust_threshold = 15):
    # Constants preserved as float32
    point_5 = np.float32(0.5)
    two = np.float32(2)
    three = np.float32(3)
    four = np.float32(4)
    five = np.float32(5)

    lineoffset += 1
    lineoffset_out_samples = outwidth * lineoffset

    # average out any unusual spikes in wow that happen on a per line basis
    # this indicates an hsync tbc error vs. being normal wow from playback speed variations
    # in this case for level adjusting we just want to fallback to the average wow to avoid a bright or dark line
    median = np.median(wowfactors)
    mad = np.median(np.abs(wowfactors - median)) # median absolute deviation
    threshold = level_adjust_threshold * mad if mad > 0 else 0.001  # fallback for no variance

    level_adjusts = np.where(
        np.abs(wowfactors - median) > threshold,
        median,
        wowfactors
    )

    if level_smoothing_lines > 0:
        # removes oscillating brightness variations for video with lots of noise around the hsync pulses, i.e. noisy line locations result in noisy wow calculations
        # applies a low pass filter that smooths any sudden brightness variations while still being reactive enough to compensate for low frequency wow
        alpha = 1 / (level_smoothing_lines * outwidth)
        one_minus_alpha = 1 - alpha

        for i in range(1, len(level_adjusts)):
            level_adjusts[i] = alpha * level_adjusts[i] + one_minus_alpha * level_adjusts[i-1]       

    for i in range(lineoffset_out_samples, len(dsout) + lineoffset_out_samples):
        # compensates for the amplitude/frequency shift caused by FM demodulation under varying playback speed.
        level_adjust = level_adjusts[i]

        # reconstructs the waveform at the proper fractional sample position, undoing wow-induced timing variations
        coord = np.float32(interpolated_pixel_locs[i])
        coord_int = int(coord)

        # get the data from the buffer that aligns to the wow factor index
        p0 = buf[coord_int - 1]
        p1 = buf[coord_int]
        p2 = buf[coord_int + 1]
        p3 = buf[coord_int + 2]
        x = np.float32(coord - coord_int)

        # perform cubic scaling
        a = p2 - p0
        b = two * p0 - five * p1 + four * p2 - p3
        c = three * (p1 - p2) + p3 - p0
        dsout[i-lineoffset_out_samples] = level_adjust * (p1 + point_5 * x * (a + x * (b + x * c)))


frequency_suffixes = [
    ("ghz", 1.0e9),
    ("mhz", 1.0e6),
    ("khz", 1.0e3),
    ("hz", 1.0),
    ("fsc", 315.0e6 / 88.0),
    ("fscpal", (283.75 * 15625) + 25),
]


def parse_frequency(string):
    """Parse an argument string, returning a float frequency in MHz."""
    multiplier = 1.0e6
    for suffix, mult in frequency_suffixes:
        if string.lower().endswith(suffix):
            multiplier = mult
            string = string[: -len(suffix)]
            break
    return (multiplier * float(string)) / 1.0e6


"""

For this part of the loader phase I found myself going to function objects that implement this sample API:

```
infile: standard readable/seekable python binary file
sample: starting sample #
readlen: # of samples
```
Returns data if successful, or None or an upstream exception if not (including if not enough data is available)
"""


def make_loader(filename, inputfreq=None):
    """Return an appropriate loader function object for filename.

    If inputfreq is specified, it gives the sample rate in MHz of the source
    file, and the loader will resample from that rate to 40 MHz. Any sample
    rate specified by the source file's metadata will be ignored, as some
    formats can't represent typical RF sample rates accurately."""

    if inputfreq is not None:
        # We're resampling, so we have to use ffmpeg.

        if filename.endswith(".s16") or filename.endswith(".raw"):
            input_args = ["-f", "s16le"]
        elif filename.endswith(".r16") or filename.endswith(".u16") or filename.endswith(".tbc"):
            input_args = ["-f", "u16le"]
        elif filename.endswith(".rf"):
            input_args = ["-f", "f32le"]
        elif filename.endswith(".s8"):
            input_args = ["-f", "s8"]
        elif filename.endswith(".r8") or filename.endswith(".u8"):
            input_args = ["-f", "u8"]
        elif filename.endswith(".lds") or filename.endswith(".r30"):
            raise ValueError("File format not supported when resampling: " + filename)
        else:
            # Assume ffmpeg will recognise this format itself.
            input_args = []

        output_args = []

        if inputfreq != 40:
            # Use asetrate first to override the input file's sample rate.
            output_args = [
                "-filter:a",
                "asetrate=" + str(inputfreq * 1e6) + ",aresample=" + str(40e6),
            ]

        return LoadFFmpeg(input_args=input_args, output_args=output_args)

    elif filename.endswith(".lds"):
        return load_packed_data_4_40
    elif filename.endswith(".r30"):
        return load_packed_data_3_32
    elif filename.endswith(".rf"):
        return load_unpacked_data_float32
    elif filename.endswith(".s16"):
        return load_unpacked_data_s16
    elif filename.endswith(".r16") or filename.endswith(".u16"):
        return load_unpacked_data_u16
    elif filename.endswith(".r8") or filename.endswith(".u8"):
        return load_unpacked_data_u8
    elif (
        filename.endswith("raw.oga")
        or filename.endswith(".ldf")
        or filename.endswith(".wav")
        or filename.endswith(".flac")
        or filename.endswith(".vhs")
    ):
        try:
            return LoadLDF(filename)
        except FileNotFoundError:
            print(
                "ld-ldf-reader not found in PATH, using ffmpeg instead.",
                file=sys.stderr,
            )
        except Exception:
            # print("Please build and install ld-ldf-reader in your PATH for improved performance", file=sys.stderr)
            traceback.print_exc()
            print(
                "Failed to load with ld-ldf-reader, trying ffmpeg instead.",
                file=sys.stderr,
            )

    return LoadFFmpeg()


def load_unpacked_data(infile, sample, readlen, sampletype):
    # this is run for unpacked data:
    # 1 is for 8-bit cxadc data, 2 for 16bit DD, 3 for 16bit cxadc

    samplelength = 2 if sampletype == 3 else sampletype

    infile.seek(sample * samplelength, 0)
    inbuf = infile.read(readlen * samplelength)

    if sampletype == 4:
        indata = np.fromstring(inbuf, "float32", len(inbuf) // 4) * 32768
    elif sampletype == 3:
        indata = np.frombuffer(inbuf, "uint16", len(inbuf) // 2)
    elif sampletype == 2:
        indata = np.fromstring(inbuf, "int16", len(inbuf) // 2)
    else:
        # NOTE(oln): Can probably use frombuffer for other variants too but
        # didn't have any samples to test with.
        indata = np.frombuffer(inbuf, "uint8", len(inbuf))

    if len(indata) < readlen:
        return None

    return indata


def load_unpacked_data_u8(infile, sample, readlen):
    return load_unpacked_data(infile, sample, readlen, 1)


def load_unpacked_data_s16(infile, sample, readlen):
    return load_unpacked_data(infile, sample, readlen, 2)


def load_unpacked_data_u16(infile, sample, readlen):
    return load_unpacked_data(infile, sample, readlen, 3)


def load_unpacked_data_float32(infile, sample, readlen):
    return load_unpacked_data(infile, sample, readlen, 4)


# This is for the .r30 format I did in ddpack/unpack.c.  Deprecated but I still have samples in it.
def load_packed_data_3_32(infile, sample, readlen):
    start = (sample // 3) * 4
    offset = sample % 3
    start, offset

    infile.seek(start)

    # we need another word in case offset != 0
    needed = int(np.ceil(readlen * 3 / 4) * 4) + 4

    inbuf = infile.read(needed)
    indata = np.fromstring(inbuf, "uint32", len(inbuf) // 4)

    if len(indata) < needed:
        return None

    unpacked = np.zeros(len(indata) * 3, dtype=np.int16)

    # By using strides the unpacked data can be loaded with no additional copies
    np.bitwise_and(indata, 0x3FF, out=unpacked[0::3])
    # hold the shifted bits in it's own array to avoid an allocation
    tmp = np.right_shift(indata, 10)
    np.bitwise_and(tmp, 0x3FF, out=unpacked[1::3])
    np.right_shift(indata, 20, out=tmp)
    np.bitwise_and(tmp, 0x3FF, out=unpacked[2::3])

    return unpacked[offset : offset + readlen]


# The 10-bit samples from the Duplicator...
# """
# From Simon's code:

# // Original
# // 0: xxxx xx00 0000 0000
# // 1: xxxx xx11 1111 1111
# // 2: xxxx xx22 2222 2222
# // 3: xxxx xx33 3333 3333
# //
# // Packed:
# // 0: 0000 0000 0011 1111
# // 2: 1111 2222 2222 2233
# // 4: 3333 3333
# """

@njit(cache=True, nogil=True)
def unpack_data_4_40(indata, readlen, offset):
    """Inner unpacking function, split off to allow numba optimisation."""
    unpacked = np.zeros(readlen + 4, dtype=np.uint16)

    # Data needs to be in uint16 for the shift to do the right thing
    # Could've used dtype argument to right_shift but numba didn't like that.
    #
    indatai16 = indata.astype(np.uint16)

    unpacked[0::4] = ( indatai16[0::5] << 2)         | ((indata[1::5] >> 6) & 0x03)
    unpacked[1::4] = ((indatai16[1::5] & 0x3F) << 4) | ((indata[2::5] >> 4) & 0x0F)
    unpacked[2::4] = ((indatai16[2::5] & 0x0F) << 6) | ((indata[3::5] >> 2) & 0x3F)
    unpacked[3::4] = ((indatai16[3::5] & 0x03) << 8) | indata[4::5]

    # convert back to original DdD 16-bit format (signed 16-bit, left shifted)
    rv_unsigned = unpacked[offset : offset + readlen]
    rv_signed   = np.left_shift(rv_unsigned - 512, 6).astype(np.int16)

    return rv_signed


@profile
def load_packed_data_4_40(infile, sample, readlen):
    """Load data from packed DdD format (4 x 10-bits packed in 5 bytes)"""
    start = (sample // 4) * 5
    offset = sample % 4

    infile.seek(start)

    # we need another word in case offset != 0
    needed = int(np.ceil(readlen * 5 // 4)) + 5

    inbuf = infile.read(needed)
    indata = np.frombuffer(inbuf, "uint8", len(inbuf))

    if len(indata) < needed:
        return None

    return unpack_data_4_40(indata, readlen, offset)


class LoadFFmpeg:
    """Load samples from a wide variety of formats using ffmpeg."""

    def __init__(self, input_args=[], output_args=[]):
        self.input_args = input_args
        self.output_args = output_args

        # ffmpeg subprocess
        self.ffmpeg = None

        # The number of the next byte ffmpeg will return
        self.position = 0

        # Keep a buffer of recently-read data, to allow seeking backwards by
        # small amounts. The last byte returned by ffmpeg is at the end of
        # this buffer.
        self.rewind_size = 2 * 1024 * 1024
        self.rewind_buf = b""

    def _close(self):
        if self.ffmpeg is not None:
            self.ffmpeg.kill()
            self.ffmpeg.wait()

    def __del__(self):
        self._close()

    def _read_data(self, count):
        """Read data as bytes from ffmpeg, append it to the rewind buffer, and
        return it. May return less than count bytes if EOF is reached."""

        data = self.ffmpeg.stdout.read(count)
        self.position += len(data)

        self.rewind_buf += data
        self.rewind_buf = self.rewind_buf[-self.rewind_size :]

        return data

    def read(self, infile, sample, readlen):
        sample_bytes = sample * 2
        readlen_bytes = readlen * 2

        if self.ffmpeg is None:
            command = ["ffmpeg", "-hide_banner", "-loglevel", "quiet"]
            command += self.input_args
            command += ["-i", "-"]
            command += self.output_args
            command += ["-c:a", "pcm_s16le", "-f", "s16le", "-"]
            self.ffmpeg = subprocess.Popen(
                command, stdin=infile, stdout=subprocess.PIPE
            )

        if sample_bytes < self.position:
            # Seeking backwards - use data from rewind_buf
            start = len(self.rewind_buf) - (self.position - sample_bytes)
            end = min(start + readlen_bytes, len(self.rewind_buf))
            if start < 0:
                raise IOError("Seeking too far backwards with ffmpeg")
            buf_data = self.rewind_buf[start:end]
            sample_bytes += len(buf_data)
            readlen_bytes -= len(buf_data)
        else:
            buf_data = b""

        while sample_bytes > self.position:
            # Seeking forwards - read and discard samples
            count = min(sample_bytes - self.position, self.rewind_size)
            self._read_data(count)

        if readlen_bytes > 0:
            # Read some new data from ffmpeg
            read_data = self._read_data(readlen_bytes)
            if len(read_data) < readlen_bytes:
                # Short read - end of file
                return None
        else:
            read_data = b""

        data = buf_data + read_data
        assert len(data) == readlen * 2
        return np.frombuffer(data, "<i2")

    def __call__(self, infile, sample, readlen):
        return self.read(infile, sample, readlen)


class LoadLDF:
    """Load samples from an .ldf file, using ld-ldf-reader which itself uses ffmpeg."""

    def __init__(self, filename, input_args=[], output_args=[]):
        self.input_args = input_args
        self.output_args = output_args

        self.filename = filename

        # The number of the next byte ld-ldf-reader will return

        self.position = 0
        # Keep a buffer of recently-read data, to allow seeking backwards by
        # small amounts. The last byte returned by ffmpeg is at the end of
        # this buffer.
        self.rewind_size = 2 * 1024 * 1024
        self.rewind_buf = b""

        self.ldfreader = None

        # ld-ldf-reader subprocess
        self.ldfreader = self._open(0)

    def __del__(self):
        self._close()

    def _read_data(self, count):
        """Read data as bytes from ffmpeg, append it to the rewind buffer, and
        return it. May return less than count bytes if EOF is reached."""

        data = self.ldfreader.stdout.read(count)
        self.position += len(data)

        self.rewind_buf += data
        self.rewind_buf = self.rewind_buf[-self.rewind_size :]

        return data

    def _close(self):
        try:
            if self.ldfreader is not None:
                self.ldfreader.kill()
                self.ldfreader.wait()
                del self.ldfreader

            self.ldfreader = None
        except Exception:
            print("Failed to close ldf reader", file=sys.stderr)
            traceback.print_exc()
            pass

    def _open(self, sample):
        self._close()

        command = ["ld-ldf-reader", self.filename, str(sample)]

        ldfreader = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.position = sample * 2
        self.rewind_buf = b""

        return ldfreader

    def read(self, infile, sample, readlen):
        sample_bytes = sample * 2
        readlen_bytes = readlen * 2

        if self.ldfreader is None or ((sample_bytes - self.position) > 40000000):
            self.ldfreader = self._open(sample)

        if sample_bytes < self.position:
            # Seeking backwards - use data from rewind_buf
            start = len(self.rewind_buf) - (self.position - sample_bytes)
            end = min(start + readlen_bytes, len(self.rewind_buf))
            if start < 0:
                # raise IOError("Seeking too far backwards with ffmpeg")
                self.ldfreader = self._open(sample)
                buf_data = b""
            else:
                buf_data = self.rewind_buf[start:end]
                sample_bytes += len(buf_data)
                readlen_bytes -= len(buf_data)
        elif (sample_bytes - self.position) > (40 * 1024 * 1024 * 2):
            self.ldfreader = self._open(sample)
            buf_data = b""
        else:
            buf_data = b""

        while sample_bytes > self.position:
            # Seeking forwards - read and discard samples
            count = min(sample_bytes - self.position, self.rewind_size)
            self._read_data(count)

        if readlen_bytes > 0:
            # Read some new data from ffmpeg
            read_data = self._read_data(readlen_bytes)
            if len(read_data) < readlen_bytes:
                # Short read - end of file
                return None
        else:
            read_data = b""

        data = buf_data + read_data
        assert len(data) == readlen * 2
        return np.frombuffer(data, "<i2")

    def __call__(self, infile, sample, readlen):
        return self.read(infile, sample, readlen)


def ffmpeg_pipe(outname: str, opts: str):
    cmd = f"ffmpeg -y -hide_banner -loglevel quiet -f s16le -ar 40k -ac 1 -i -"
    if opts and len(opts):
        cmd += f" {opts}"

    cmd = cmd.split(' ')

    process = subprocess.Popen(
        [*cmd, outname],
        stdin=subprocess.PIPE,
    )

    return process, process.stdin


def ldf_pipe(outname: str, compression_level: int = 6):
    return ffmpeg_pipe(outname, f"-acodec flac -f ogg -compression_level {compression_level}")


def ac3_pipe(outname: str):
    processes = []

    cmd1 = "sox -r 40000000 -b 8 -c 1 -e signed -t raw - -b 8 -r 46080000 -e unsigned -c 1 -t raw -".split()
    cmd2 = "ld-ac3-demodulate -v 3 - -".split()
    cmd3 = ["ld-ac3-decode", "-", outname]

    logfp = open(outname + '.log', 'w')

    # This is done in reverse order to allow for pipe building
    processes.append(subprocess.Popen(cmd3,
                                      stdin=subprocess.PIPE,
                                      stdout=logfp,
                                      stderr=subprocess.STDOUT))

    processes.append(subprocess.Popen(cmd2,
                                      stdin=subprocess.PIPE,
                                      stdout=processes[-1].stdin))

    processes.append(subprocess.Popen(cmd1,
                                      stdin=subprocess.PIPE,
                                      stdout=processes[-1].stdin))

    return processes, processes[-1].stdin

# Git helpers

def get_version():
    """ get version info stashed in this directory's version file """

    try:
        # get the directory this file was pulled from, then add /version
        scriptdir = os.path.dirname(os.path.realpath(__file__))
        fd = open(os.path.join(scriptdir, 'version'), 'r')

        fdata = fd.read()
        return fdata.strip() # remove trailing \n etc
    except (FileNotFoundError, OSError):
        # Just return 'unknown' if we fail to find anything.
        return "unknown"


def get_git_info():
    """ Return git branch and commit for current directory, if available. """

    branch = "UNKNOWN"
    commit = "UNKNOWN"

    version = get_version()
    if ':' in version:
        branch, commit = version.split(':')[0:2]

    try:
        sp = subprocess.run(
            "git rev-parse --abbrev-ref HEAD", shell=True, capture_output=True
        )
        if not sp.returncode:
            branch = sp.stdout.decode("utf-8").strip()

        sp = subprocess.run(
            "git rev-parse --short HEAD", shell=True, capture_output=True
        )
        if not sp.returncode:
            commit = sp.stdout.decode("utf-8").strip()
    except Exception:
        print("Something went wrong when trying to read git info...", file=sys.stderr)
        traceback.print_exc()
        pass

    return branch, commit


# Essential (or at least useful) standalone routines and lambdas

pi = np.pi
tau = np.pi * 2

# https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
polar2z = lambda r, θ: r * np.exp(1j * θ)
deg2rad = lambda θ: θ * (np.pi / 180)


def emphasis_iir(t1, t2, fs):
    """Generate an IIR filter for 6dB/octave pre-emphasis (t1 > t2) or
    de-emphasis (t1 < t2), given time constants for the two corners."""

    # Convert time constants to frequencies, and pre-warp for bilinear transform
    w1 = 2 * fs * np.tan((1 / t1) / (2 * fs))
    w2 = 2 * fs * np.tan((1 / t2) / (2 * fs))

    # Zero at t1, pole at t2
    tf_b, tf_a = sps.zpk2tf([-w1], [-w2], w2 / w1)
    rv = sps.bilinear(tf_b, tf_a, fs)

    return rv

# This converts a regular B, A filter to an FFT of our selected block length
def filtfft(filt, blocklen):
    return sps.freqz(filt[0], filt[1], blocklen, whole=1)[1]


@njit(cache=True)
def inrange(a, mi, ma):
    return (a >= mi) & (a <= ma)


def sqsum(cmplx):
    return np.sqrt((cmplx.real ** 2) + (cmplx.imag ** 2))


@njit(cache=True,nogil=True)
def calczc_findfirst(data, target, rising):
    if rising:
        for i in range(1, len(data)):
            if data[i - 1] < target and data[i] >= target:
                return i

        return None
    else:
        for i in range(1, len(data)):
            if data[i - 1] > target and data[i] <= target:
                return i

        return None


@njit(cache=True,nogil=True)
def calczc_do(data, _start_offset, target, edge=0, count=16):
    start_offset = int(_start_offset)
    icount = int(count + 1)

    if edge == 0:  # capture rising or falling edge
        if data[start_offset] < target:
            edge = 1
        else:
            edge = -1

    loc = calczc_findfirst(
        data[start_offset : start_offset + icount], target, edge == 1
    )

    if loc is None:
        return None

    x = start_offset + loc
    a = data[x - 1] - target
    b = data[x] - target

    if b - a != 0:
        y = -a / (-a + b)
    else:
        print(
            "RuntimeWarning: Div by zero prevented at lddecode/utils.calczc_do()", a, b
        )
        y = 0

    return x - 1 + y


def calczc(data, _start_offset, target, edge=0, count=16, reverse=False):
    """ edge:  -1 falling, 0 either, 1 rising """
    if reverse:
        # Instead of actually implementing this in reverse, use numpy to flip data
        rev_zc = calczc_do(data[_start_offset::-1], 0, target, edge, count)
        if rev_zc is None:
            return None

        return _start_offset - rev_zc

    return calczc_do(data, _start_offset, target, edge, count)


# copied from vhs-decode

def gen_bpf_supergauss(freq_low, freq_high, order, nyquist_hz, block_len):
    sg = supergauss(
        np.linspace(0, nyquist_hz, block_len // 2 + 1),
        freq_high - freq_low,
        order,
        (freq_high + freq_low) / 2.0,
    )[:-1]

    return np.concatenate([sg, np.flip(sg)])

def supergauss(x, freq, order=1, centerfreq=0):
    return np.exp(
        -2
        * np.power(
            (2 * (x - centerfreq) * (math.log(2.0) / 2.0) ** (1 / (2 * order))) / freq,
            2 * order,
        )
    )

# Shamelessly based on https://github.com/scipy/scipy/blob/v1.6.0/scipy/signal/signaltools.py#L2264-2267
# ... and intended for real FFT, but seems fine with complex as well ;)
def build_hilbert(fft_size):
    if (fft_size // 2) - (fft_size / 2) != 0:
        raise Exception("build_hilbert: must have even fft_size")

    output = np.zeros(fft_size)
    output[0] = output[fft_size // 2] = 1
    output[1 : fft_size // 2] = 2

    return output

if not numba.version_info.major and numba.version_info.minor < 59:
    print("DEPRECATION WARNING: Please upgrade numba to 0.59 or later.", file=sys.stderr)
    print("(follow instructions on the ld-decode wiki to set up a virtualenv)", file=sys.stderr)

    @njit(cache=True,nogil=True)
    def unwrap_hilbert_getangles(hilbert):
        tangles = np.angle(hilbert)
        dangles = np.ediff1d(tangles, to_begin=np.asarray(0, dtype=tangles.dtype)).real

        # make sure unwapping goes the right way
        if dangles[0] < -pi:
            dangles[0] += tau

        return dangles

    @njit(cache=True,nogil=True)
    def unwrap_hilbert_fixangles(tdangles2, freq_hz):
        # With extremely bad data, the unwrapped angles can jump.
        while np.min(tdangles2) < 0:
            tdangles2[tdangles2 < 0] += tau
        while np.max(tdangles2) > tau:
            tdangles2[tdangles2 > tau] -= tau

        return tdangles2 * (freq_hz / tau)

    def unwrap_hilbert(hilbert, freq_hz):
        dangles = unwrap_hilbert_getangles(hilbert)

        # This can't be run with numba
        tdangles2 = np.unwrap(dangles)

        return unwrap_hilbert_fixangles(tdangles2, freq_hz) #* (freq_hz / tau)
else:
    @njit(cache=True,nogil=True)
    def unwrap_hilbert(hilbert, freq_hz):
        tangles = np.angle(hilbert)
        dangles = np.ediff1d(tangles, to_begin=np.asarray(0, dtype=tangles.dtype)).real

        # make sure unwapping goes the right way
        if dangles[0] < -pi:
            dangles[0] += tau

        dangles = np.unwrap(dangles)

        # With extremely bad data, the unwrapped angles can jump.
        while np.min(dangles) < 0:
            dangles[dangles < 0] += tau
        while np.max(dangles) > tau:
            dangles[dangles > tau] -= tau

        return dangles * (freq_hz / tau)

def fft_determine_slices(center, min_bandwidth, freq_hz, bins_in):
    """ returns the # of sub-bins needed to get center+/-min_bandwidth.
        The returned lowbin is the first bin (symmetrically) needed to be saved.

        This will need to be 'flipped' using fft_slice to get the trimmed set
    """

    # compute the width of each bin
    binwidth = freq_hz / bins_in

    cbin = nb_round(center / binwidth)

    # compute the needed number of fft bins...
    bbins = nb_round(min_bandwidth / binwidth)
    # ... and round that up to the next power of two
    nbins = 2 * (2 ** math.ceil(math.log2(bbins * 2)))

    lowbin = cbin - (nbins // 4)

    cut_freq = binwidth * nbins

    return lowbin, nbins, cut_freq


def fft_do_slice(fdomain, lowbin, nbins, blocklen):
    """ Uses lowbin and nbins as returned from fft_determine_slices to
        cut the fft """
    nbins_half = nbins // 2
    return np.concatenate(
        [
            fdomain[lowbin : lowbin + nbins_half],
            fdomain[blocklen - lowbin - nbins_half : blocklen - lowbin],
        ]
    )

'''
overlap/save [i]fft functions for testing.  use in a jupyter notebook or similar
like this:

f = overlap_save_fft(fields[0].dspicture)
invf = overlap_save_ifft(f, round=True).astype(np.uint16)
sum(invf != fields[0].dspicture) # should be 0
'''

def overlap_save_fft(data, blocklen=32768, blockcut_begin=1024, blockcut_end=512):
    blockstride = blocklen - blockcut_begin - blockcut_end

    numblocks = (len(data) // blockstride) + 1

    # return the length as the first element so we can reconstruct the original data
    # with the correct length in overlap_save_ifft
    fft_out = [len(data)]
    for i in range(numblocks):
        blockstart = i * blockstride
        dcut = data[blockstart : blockstart + blocklen]
        if len(dcut) < blocklen:
            pad = np.full(blocklen - len(dcut), 0, dtype=dcut.dtype)
            dcut = np.concatenate((dcut, pad))

        fft_out.append(np.fft.fft(dcut))

    return fft_out

def overlap_save_ifft(ffts, blockcut_begin=1024, blockcut_end=512, round=False):
    # blocklen is inferred from fft size
    data_out = []

    for i, f in enumerate(ffts[1:]):
        invf = np.fft.ifft(f).real
        if round:
            invf = np.round(invf)
        if not i:
            data_out.append(invf[:-blockcut_end])
        else:
            data_out.append(invf[blockcut_begin:-blockcut_end])

    return np.concatenate(data_out)[:ffts[0]]


@njit(cache=True)
def genwave(rate, freq, initialphase=0):
    """ Generate an FM waveform from target frequency data """
    out = np.zeros(len(rate), dtype=np.double)

    angle = initialphase

    for i in range(0, len(rate)):
        out[i] = np.sin(angle)

        angle += np.pi * (rate[i] / freq)
        if angle > np.pi:
            angle -= tau

    return out


# slightly faster than np.std for short arrays
@njit(cache=True)
def rms(arr):
    return np.sqrt(np.mean(np.square(arr - np.mean(arr))))


# MTF calculations
def get_fmax(cavframe=0, laser=780, na=0.5, fps=30):
    loc = 0.055 + ((cavframe / 54000) * 0.090)
    return (2 * na / (laser / 1000)) * (2 * np.pi * fps) * loc


def compute_mtf(freq, cavframe=0, laser=780, na=0.52):
    fmax = get_fmax(cavframe, laser, na)

    freq_mhz = freq / 1000000

    if type(freq_mhz) == np.ndarray:
        freq_mhz[freq_mhz > fmax] = fmax
    elif freq_mhz > fmax:
        return 0

    # from Compact Disc Technology AvHeitarō Nakajima, Hiroshi Ogawa page 17
    return (2 / np.pi) * (
        np.arccos(freq_mhz / fmax)
        - ((freq_mhz / fmax) * np.sqrt(1 - ((freq_mhz / fmax) ** 2)))
    )


def roundfloat(fl, places=3):
    """ round float to (places) decimal places """
    r = 10 ** places
    return np.round(fl * r) / r


@njit(nogil=True, cache=True)
def hz_to_output_array(input, ire0, hz_ire, outputZero, vsync_ire, out_scale):
    reduced = (input - ire0) / hz_ire
    reduced -= vsync_ire

    return (
        np.clip((reduced * out_scale) + outputZero, 0, 65535) + 0.5
    ).astype(np.uint16)



# Something like this should be a numpy function, but I can't find it.
@jit(cache=True, nopython=True)
def findareas(array, cross):
    """ Find areas where `array` is <= `cross`

    returns: array of tuples of said areas (begin, end, length)
    """
    starts = np.where(np.logical_and(array[1:] < cross, array[:-1] >= cross))[0]
    ends = np.where(np.logical_and(array[1:] >= cross, array[:-1] < cross))[0]

    # remove 'dangling' beginnings and endings so everything zips up nicely and in order
    if ends[0] < starts[0]:
        ends = ends[1:]

    if starts[-1] > ends[-1]:
        starts = starts[:-1]

    return [(*z, z[1] - z[0]) for z in zip(starts, ends)]


Pulse = namedtuple("Pulse", "start len")


@njit(cache=True, nogil=True)
def findpulses_numba_raw(sync_ref, high, min_synclen=0, max_synclen=5000):
    """Locate possible pulses by looking at areas within some range.
    Outputs arrays of starts and lengths
    """

    in_pulse = sync_ref[0] <= high

    # Start/lengths lists
    # It's possible this could be optimized further by using
    # a different data structure here.
    starts = []
    lengths = []

    cur_start = 0

    # Basic algorithm here is swapping between two states, going to the other one if we detect the
    # current sample passed the threshold.
    for pos, value in enumerate(sync_ref):
        if in_pulse:
            if value > high:
                length = pos - cur_start
                # If the pulse is in range, and it's not a starting one
                if inrange(length, min_synclen, max_synclen) and cur_start != 0:
                    starts.append(cur_start)
                    lengths.append(length)
                in_pulse = False
        elif value <= high:
            cur_start = pos
            in_pulse = True

    # Not using a possible trailing pulse
    # if in_pulse:
    #     # Handle trailing pulse
    #     length = len(sync_ref) - 1 - cur_start
    #     if inrange(length, min_synclen, max_synclen):
    #         starts.append(cur_start)
    #         lengths.append(length)

    return np.asarray(starts), np.asarray(lengths)


def _to_pulses_list(pulses_starts, pulses_lengths):
    """Make list of Pulse objects from arrays of pulses starts and lengths"""
    # Not using numba for this right now as it seemed to cause random segfault
    # in tests.
    return [Pulse(z[0], z[1]) for z in zip(pulses_starts, pulses_lengths)]


def findpulses(sync_ref, _, high):
    """Locate possible pulses by looking at areas within some range.
    .outputs a list of Pulse tuples
    """
    pulses_starts, pulses_lengths = findpulses_numba_raw(
        sync_ref, high
    )
    return _to_pulses_list(pulses_starts, pulses_lengths)

if False:
    @njit(cache=True,nogil=True)
    def numba_pulse_qualitycheck(prevpulse: Pulse, pulse: Pulse, inlinelen: int):

        if prevpulse[0] > 0 and pulse[0] > 0:
            exprange = (0.4, 0.6)
        elif prevpulse[0] == 0 and pulse[0] == 0:
            exprange = (0.9, 1.1)
        else:  # transition to/from regular hsyncs can be .5 or 1H
            exprange = (0.4, 1.1)

        linelen = (pulse[1].start - prevpulse[1].start) / inlinelen
        inorder = inrange(linelen, *exprange)

        return inorder

    @njit(cache=True,nogil=True)
    def numba_computeLineLen(validpulses, inlinelen):
        # determine longest run of 0's
        longrun = [-1, -1]
        currun = None
        for i, v in enumerate([p[0] for p in validpulses]):
            if v != 0:
                if currun is not None and currun[1] > longrun[1]:
                    longrun = currun
                currun = None
            elif currun is None:
                currun = [i, 0]
            else:
                currun[1] += 1

        if currun is not None and currun[1] > longrun[1]:
            longrun = currun

        linelens = []
        for i in range(longrun[0] + 1, longrun[0] + longrun[1]):
            linelen = validpulses[i][1].start - validpulses[i - 1][1].start
            if inrange(linelen / inlinelen, 0.95, 1.05):
                linelens.append(
                    validpulses[i][1].start - validpulses[i - 1][1].start
                )

        # Can't prepare the mean here since numba np.mean doesn't work over lists
        return linelens


@njit(cache=True, nogil=True)
def findpeaks(array, low=0):
    array2 = array.copy()
    array2[np.where(array2 < low)] = 0

    return [
        loc - 1
        for loc in np.where(
            np.logical_and(array2[:-1] > array2[-1], array2[1:] > array2[:-1])
        )[0]
    ]


def LRUupdate(l, k):
    """ This turns a list into an LRU table.  When called it makes sure item 'k' is at the beginning,
        so the list is in descending order of previous use.
    """
    try:
        l.remove(k)
    except Exception:
        pass

    l.insert(0, k)

# Lambdas used to shorten filter-building functions

# Split out the frequency list given to the filter builder
freqrange = lambda f1, f2: [
    f1 / self.freq_hz_half,
    f2 / self.freq_hz_half,
]

# Like freqrange, but for notch filters
notchrange = lambda f, notchwidth, hz: [
    (f - notchwidth) / self.freq_hz_half if hz else self.freq_half,
    (f + notchwidth) / self.freq_hz_half if hz else self.freq_half,
]

# numba jit functions, used to numba-ify parts of more complex functions

@njit(cache=True,nogil=True)
def nb_median(m):
    return np.median(m)

# Enabling nogil here kills performance - cache issues?
@njit(cache=True,nogil=False)
def nb_concatenate(m):
    tlen = sum([len(i) for i in m])

    out = np.empty(tlen, dtype=m[0].dtype)
    pos = 0
    for i in m:
        out[pos : pos + len(i)] = i
        pos += len(i)

    return out

@njit(cache=True,nogil=True)
def nb_round(m):
    return int(np.round(m))


@njit(cache=True,nogil=True)
def nb_mean(m):
    return np.mean(m)


@njit(cache=True,nogil=True)
def nb_min(m):
    return np.min(m)


@njit(cache=True,nogil=True)
def nb_max(m):
    return np.max(m)


@njit(cache=True,nogil=True)
def nb_abs(m):
    return np.abs(m)


@njit(cache=True,nogil=True)
def nb_absmax(m):
    return np.max(np.abs(m))


@njit(cache=True,nogil=True)
def nb_std(m):
    return np.std(m)


@njit(cache=True,nogil=True)
def nb_diff(m):
    return np.diff(m)


@njit(cache=True,nogil=True)
def nb_mul(x, y):
    return x * y


@njit(cache=True,nogil=True)
def nb_where(x):
    return np.where(x)


@njit(cache=True,nogil=True)
def nb_gt(x, y):
    return (x > y)


@njit(cache=True,nogil=True)
def n_orgt(a, x, y):
    a |= (x > y)


@njit(cache=True,nogil=True)
def n_orlt(a, x, y):
    a |= (x < y)


@njit(cache=True,nogil=True)
def n_ornotrange(a, x, y, z):
    a |= (x < y) | (x > z)


@njit(cache=True, nogil=True)
def angular_mean_helper(x, cycle_len=1.0, zero_base=True):
    """ Compute the mean phase, assuming 0..1 is one phase cycle

        (Using this technique handles the 3.99, 5.01 issue
        where otherwise the phase average would be 0.5.  while a
        naive computation could be changed to rotate around 0.5,
        that breaks down when things are out of phase...)
    """
    x2 = x - x.astype(np.int32)  # not strictly necessary but slightly more precise

    # refer to https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    angles = [np.e ** (1j * f * np.pi * 2 / cycle_len) for f in x2]

    return angles

@njit(cache=True)
def phase_distance(x, c=0.75):
    """ returns the shortest path between two phases (assuming x and c are in (0..1)) """
    d = (x - np.floor(x)) - c

    if d < -0.5:
        d += 1
    elif d > 0.5:
        d -= 1

    return d


# Used to help w/CX routines
@njit(cache=True)
def db_to_lev(db):
    return 10 ** (db / 20)


@njit(cache=True)
def lev_to_db(rlev):
    return 20 * np.log10(rlev)


@njit(cache=True)
def dsa_rescale_and_clip(infloat):
    """rescales input value to output levels and clips to fit into signed 16-bit"""
    value = int(np.round(infloat * 32767.0 / 371081.0))
    return min(max(value, -32766), 32766)


# Hotspot subroutines in FieldNTSC's compute_line_bursts function,
# removed so that they can be JIT'd


@njit(cache=True,nogil=True)
def clb_findbursts(isrising, zcs, burstarea, i, endburstarea, threshold, bstart, s_rem, zcburstdiv, phase_adjust):
    zc_count = 0
    rising_count = 0
    j = i

    isrising.fill(False)
    zcs.fill(0)

    while j < endburstarea and zc_count < len(zcs):
        if np.abs(burstarea[j]) > threshold:
            zc = calczc_do(burstarea, j, 0)
            if zc is not None:
                isrising[zc_count] = burstarea[j] < 0
                zcs[zc_count] = zc
                zc_count += 1
                j = int(zc) + 1
            else:
                break
        else:
            j += 1

    if zc_count:
        zc_cycles = ((bstart + zcs - s_rem) / zcburstdiv) + phase_adjust
        # numba doesn't support np.round over an array, so we have to do this
        zc_rounds = (zc_cycles + .5).astype(np.int32)
        phase_adjust += nb_median(zc_rounds - zc_cycles)
        rising_count = np.sum(np.bitwise_xor(isrising, (zc_rounds % 2) != 0))

    return zc_count, phase_adjust, rising_count

@njit(cache=True)
def distance_from_round(x):
    # Yes, this was a hotspot.
    return np.round(x) - x


def init_opencl(cl, name = None):
    # Create some context on the first available GPU
    if 'PYOPENCL_CTX' in os.environ:
        ctx = cl.create_some_context()
    else:
        ctx = None
        # Find the first OpenCL GPU available and use it, unless
        for p in cl.get_platforms():
            for d in p.get_devices():
                if d.type & cl.device_type.GPU == 1:
                    continue
                print("Selected device: ", d.name)
                ctx = cl.Context(devices=(d,))
                break
            if ctx is not None:
                break
    #queue = cl.CommandQueue(ctx)
    return ctx

class FieldInfo:
    def __init__(self, field_history_size=3):
        self._field_history_size = field_history_size
        # store previous field references in a ring buffer
        self._fieldinfo = np.empty(field_history_size, dtype=object)
        self._fieldinfo_unsent = []
        self._len = 0

    def __len__(self):
        return self._len
    
    # called like a normal python list, where -1 is the last element, -2 the one before that, etc.
    # using [0] is not allowed since this only stores the end of the list
    def __getitem__(self, key):
        assert key < 0, "Attempted to get a field that has not been written"
        assert key > -self._field_history_size, "Attempted to get a field that is not buffered"
        return self._fieldinfo[(self._len + key) % self._field_history_size]

    def read(self):
        unsent = self._fieldinfo_unsent
        self._fieldinfo_unsent = []
        return unsent
    
    def append(self, value):
        self._fieldinfo[self._len % self._field_history_size] = value
        self._fieldinfo_unsent.append(value)
        self._len += 1

class JSONDumper:
    def __init__(self, ldd, outname):
        self._rx, self._tx = Pipe(False)
        self._writing = Event()
        self._build_json = ldd.build_json
        self._get_field_info = ldd.fieldinfo.read

        self._outname = outname
        self._dumper = Process(target=JSONDumper._consume, args=(self._rx, self._writing, self._outname, ldd.verboseVITS,), name="lddecode-json-dumper")
        self._dumper.start()
    
    def write(self):
        if not self._writing.is_set():
            json_data = self._build_json()
            field_info = self._get_field_info()
            self._tx.send(json_data)
            self._tx.send(field_info)

    def close(self):
        json_data = self._build_json()
        field_info = self._get_field_info()
        self._tx.send(json_data)
        self._tx.send(field_info)

        self._tx.send(None)
        self._dumper.join()

    @staticmethod
    def _consume(conn, ready, outname, verboseVITS):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        indent = 4 if verboseVITS else None
        linebreak = '\n' if verboseVITS else ''
        separators = None if verboseVITS else (',', ':')
        separator = ',' + linebreak
        field_info = []

        while True:
            try:
                jsondict = conn.recv()
                if jsondict is None:
                    break

                next_field_info = conn.recv()
                ready.set()
            except (InterruptedError, KeyboardInterrupt, EOFError):
                break

            # json serialize each field info object to a string
            serialized_field_info = []
            for field in next_field_info:
                serialized_field_info.append(
                    json.dumps(
                        field,
                        allow_nan=False,
                        indent=indent,
                        separators=separators
                    )
                )

            field_info.append(serialized_field_info)

            f = open(outname + ".tbc.json.tmp", "w")
            f.write('{'+linebreak)
            # write the field metadata
            for (k, v) in jsondict.items():
                json.dump(k, f, allow_nan=False, indent=indent, separators=separators)
                f.write(':')
                json.dump(v, f, allow_nan=False, indent=indent, separators=separators)
                f.write(separator)

            # Write the field info
            f.write('"fields":['+linebreak)
            for i, field in enumerate(itertools.chain.from_iterable(field_info)):
                if i != 0:
                    f.write(separator)

                f.write(field)
            f.write(linebreak+']'+linebreak+'}')

            f.write('\n')
            f.close()
            os.replace(outname + ".tbc.json.tmp", outname + ".tbc.json")

            ready.clear()

class StridedCollector:
    # This keeps a numpy buffer and outputs an fft block and keeps the overlap
    # for the next fft.
    def __init__(self, blocklen=32768, cut_begin=2048, cut_end=0):
        self.buffer = None
        self.blocklen = blocklen

        self.cut_begin = cut_begin
        self.cut_end = self.blocklen - cut_end
        self.stride = cut_begin + cut_end

    def add(self, data):
        if self.buffer is None:
            self.buffer = data
        else:
            self.buffer = np.concatenate([self.buffer, data])

        return self.have_block()

    def have_block(self):
        return (self.buffer is not None) and (len(self.buffer) >= self.blocklen)

    def cut(self, processed_data):
        # TODO: assert len(processed_data) == self.blocklen

        return processed_data[self.cut_begin : self.cut_end]

    def get_block(self):
        if self.have_block():
            rv = self.buffer[0 : self.blocklen]
            self.buffer = self.buffer[self.blocklen - self.stride :]

            return rv

        return None


if __name__ == "__main__":
    print("Nothing to see here, move along ;)")
