# This currently decodes raw VHS and Video8 HiFi RF.
# It could also do Beta HiFi, CED, LD and other stereo AFM variants,
# It has an interpretation of the noise reduction method described on IEC60774-2/1999

from dataclasses import dataclass
from fractions import Fraction
from math import exp, log10, pi, sqrt, ceil, floor, atan2, cos, sin, lcm
from typing import Tuple
from time import perf_counter
from setproctitle import setproctitle
from multiprocessing import current_process
from copy import deepcopy

from vhsdecode.rust_utils import sosfiltfilt_rust
import string
from random import SystemRandom

import numpy as np
import numba
from numba import njit
from scipy.signal import (
    lfilter_zi,
    filtfilt,
    lfilter,
    butter,
    stft,
    istft,
    fftconvolve,
    find_peaks,
    freqz,
    bilinear,
    cheby2
)
from scipy.interpolate import interp1d
from soxr import ResampleStream, resample

from noisereduce.spectralgate.nonstationary import SpectralGateNonStationary

from vhsdecode.addons.gnuradioZMQ import ZMQSend, ZMQ_AVAILABLE
from vhsdecode.utils import firdes_lowpass, firdes_highpass, StackableMA

from vhsdecode.hifi.TimeProgressBar import TimeProgressBar
from vhsdecode.hifi.utils import DecoderSharedMemory, NumbaAudioArray

import matplotlib.pyplot as plt

from vhsdecode.hifi.constants import (
    AUDIO_MODE_STEREO_MS,
    AUDIO_MODE_DUAL_MONO,
    AUDIO_MODE_DUAL_MONO_MS,
    AUDIO_MODE_MONO_L,
    AUDIO_MODE_MONO_R,
    AUDIO_MODE_MONO_SUM,
    DOC_MODE_MUTE,
    DOC_MODE_DISABLED,
    ENV_DETECTION_RMS,
    DEMOD_HILBERT_IF_RATE,
    DEMOD_QUADRATURE,
    DEMOD_HILBERT,
)

# audio processing precision
REAL_DTYPE = np.float32

# demodulation section precision
DEMOD_DTYPE_NP = np.float64
DEMOD_DTYPE_NB = numba.types.float64

BLOCKS_PER_SECOND = 2


@dataclass
class AFEParamsVHS:
    def __init__(self):
        # IEC 60774-2 pg.13 (5.4 Recording characteristics, Frequency deviation)
        self.LCarrierDeviation = 150e3
        self.RCarrierDeviation = 150e3

        # Carson's bandwidth rule: 2 * (peak_frequency_deviation + highest frequency)
        notch_padding = 35.753125e3
        self.LNotchWidth = 2 * (self.LCarrierDeviation + notch_padding)
        self.RNotchWidth = 2 * (self.RCarrierDeviation + notch_padding)

        # Notch Padding Tuning Procedure:
        # 1. Get the baseline data
        #    a. Pick a test sample. Ideally one that has high fidelity, high frequencies, or sibilance (a tape with noisy 's').
        #    b. Decode Before Change:
        #       * `Decode A`: Decode the sample with the original parameter
        #    c. Pick an initial direction you think the parameter needs to be tuned to
        #    d. Open Audacity, or an editor of your choice
        # 2. Tune the parameter
        #    a. Change the parameter in the code
        #    b. Decode After Change:
        #       * `Decode B`: Decode the sample with the changed parameter
        #    c. Import the two audio files (before and after)
        #       * Must be same length, and have the same amplitude
        #    d. Reduce amplitude of `Decode A` and `Decode A` by -6db
        #    e. Mix down `Decode A` and `Decode A` to create `Decode A + `Decode B`
        #    f. Invert phase of `Decode A + `Decode B`
        #    g. Increase amplitude of `Decode A` and `Decode A` by +6db
        #    h. Mix down `Decode A` and `Decode A + `Decode B` to create `Decode B - Decode A`
        #    i. Visually inspect `Decode B - Decode A` against `Decode A` and `Decode B` to determine which sample adds more noise
        #       * Any noise that appears in the subtraction only exists in one of the two clips
        #    j. After deciding which clip is better, pick a new parameter that is closer to the better clip
        #       * When narrowing in between two parameters, a bisect search helps to speed up the process
        #    k. With the new parameter, go to step `a` and repeat until you've completed the bisect search
        #       * Generally `Decode B` will become your new baseline (`Decode A`)


@dataclass
class AFEParams8mm:
    def __init__(self):
        # IEC 60843-1 pg.71 (6.2.3 FM audio signal recording, Maximum deviation)
        self.LCarrierDeviation = 100e3 # main channel deviation +-100kHz
        self.RCarrierDeviation = 50e3 # sub channel deviation +-50kHz

        self.LNotchWidth = 2 * (self.LCarrierDeviation + 20e3)
        self.RNotchWidth = 1.5 * self.RCarrierDeviation # narrower notch for sub channel

        self.LCarrierRef = 1.5e6
        self.RCarrierRef = 1.7e6


@dataclass
class AFEParamsPALVHS(AFEParamsVHS):
    def __init__(self):
        super().__init__()
        self.LCarrierRef = 1.4e6
        self.RCarrierRef = 1.8e6


@dataclass
class AFEParamsNTSCVHS(AFEParamsVHS):
    def __init__(self):
        super().__init__()
        self.LCarrierRef = 1.3e6
        self.RCarrierRef = 1.7e6
        self.Hfreq = 15.750e3


@dataclass
class AFEParamsNTSC8mm(AFEParams8mm):
    def __init__(self):
        super().__init__()
        self.Hfreq = 15.750e3


@dataclass
class AFEParamsPAL8mm(AFEParams8mm):
    def __init__(self):
        super().__init__()
        self.Hfreq = 15.625e3


@staticmethod
def get_standard(
    format, system, afe_left_carrier_deviation, afe_right_carrier_deviation, afe_left_carrier, afe_right_carrier
):
    if format == "vhs":
        if system == "p":
            field_rate = 50
            standard = AFEParamsPALVHS()
        elif system == "n":
            field_rate = 59.94
            standard = AFEParamsNTSCVHS()
    elif format == "8mm":
        if system == "p":
            field_rate = 50
            standard = AFEParamsPAL8mm()
        elif system == "n":
            field_rate = 59.94
            standard = AFEParamsNTSC8mm()

    if afe_left_carrier_deviation != 0:
        standard.LCarrierDeviation = afe_left_carrier_deviation
    if afe_right_carrier_deviation != 0:
        standard.RCarrierDeviation = afe_right_carrier_deviation
    if afe_left_carrier != 0:
        standard.LCarrierRef = afe_left_carrier
    if afe_right_carrier != 0:
        standard.RCarrierRef = afe_right_carrier

    return standard, field_rate


from scipy.signal import chirp
def plot_responses(*filters, n=2**20):
    plt.figure()

    # time axis
    t = np.arange(n)

    for i, filt in enumerate(filters):

        fs = filt.samp_rate
        t_sec = t / fs

        # log sweep from DC-ish to Nyquist (adjust as needed)
        f0 = 10          # start freq (Hz)
        f1 = fs / 2 * 0.9  # end freq (Hz)

        x = chirp(t_sec, f0=f0, f1=f1, t1=t_sec[-1], method='logarithmic')

        # run through filter
        y = filt.work(x)

        # FFT-based transfer estimate
        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)

        H = Y / (X + 1e-12)
        f = np.fft.rfftfreq(n, 1 / fs)

        plt.plot(
            f / 1e6,
            20 * np.log10(np.abs(H) + 1e-12),
            label=f"Filter {i}"
        )

    plt.title("Filter Frequency Response (Sweep Method)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.xlim(0, 10)
    plt.legend()
    plt.show()


class AFEFilterable:
    def __init__(self, filters_params, sample_rate, channel=0):
        self.samp_rate = sample_rate
        self.filter_params = filters_params

        if channel == 0:
            bandpass_width = self.filter_params.LNotchWidth
            center = self.filter_params.LCarrierRef
        else:
            bandpass_width = self.filter_params.RNotchWidth
            center = self.filter_params.RCarrierRef

        bandpass_low = center - bandpass_width
        bandpass_high = center + bandpass_width

        self.bandpass = cheby2(
            N=22,
            rs=220,
            Wn=[bandpass_low, bandpass_high],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )

    def work(self, data):
        return sosfiltfilt_rust(self.bandpass, data)

class FMDiscriminator:
    def __init__(
        self, sample_rate, carrier_center, deviation, max_iq_len, type
    ):
        self.sample_rate = np.int32(sample_rate)
        self.carrier = np.int32(round(carrier_center))
        self.deviation = np.int32(deviation)
        self.type = type
        
        float_info = np.finfo(np.float32)
        self.max_float = float_info.max - float_info.eps
        self.min_float = float_info.min + float_info.epsneg

        if self.type == DEMOD_QUADRATURE:
            iq_len = self._get_min_iq_length(max_iq_len)
            self.i_osc = np.empty(iq_len, dtype=DEMOD_DTYPE_NP, order="C")
            self.q_osc = np.empty(iq_len, dtype=DEMOD_DTYPE_NP, order="C")
            FMDiscriminator._generate_iq_oscillators(
                self.i_osc,
                self.q_osc,
                self.carrier,
                self.sample_rate
            )

    def _get_min_iq_length(self, max_iq_len):
        # calculates the minimum i/q oscillator size to avoid discontinuities
        samples_per_period = self.sample_rate / self.carrier
        min_periods = lcm(self.sample_rate, self.carrier) / self.sample_rate
        min_samples = int(samples_per_period * min_periods / 2)

        return min(min_samples, max_iq_len)

    @staticmethod
    @njit(
        [
            (
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.int32,
                numba.types.int32,
            )
        ],
        cache=True,
        fastmath=False,
        nogil=True,
    )
    def _generate_iq_oscillators(
        i_osc, q_osc, carrier, sample_rate
    ):
        two_pi_carrier = 2 * pi * carrier

        for i in range(len(i_osc)):
            t = i / sample_rate
            i_osc[i] = cos(two_pi_carrier * t)  #  In-phase
            q_osc[i] = -sin(two_pi_carrier * t)  # Quadrature

    @staticmethod
    def compute_analytic_signal(input):
        axis = -1
        N = input.shape[axis]
        h = np.zeros(N, dtype=np.complex64)
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2

        i = 0
        for hilbert_value in np.fft.ifft(
            np.fft.fft(input, N, axis=axis) * h, axis=axis
        ):
            input[i] = atan2(hilbert_value.imag, hilbert_value.real)  # np.angle(analytic_signal)
            i+=1

    @staticmethod
    @njit(
        [
            (
                NumbaAudioArray,
                NumbaAudioArray,
                numba.types.float32,
                numba.types.float32,
                numba.types.float32,
                numba.types.int32,
                numba.types.int32
            ),
            (
                numba.types.Array(numba.types.float64, 1, "C"),
                NumbaAudioArray,
                numba.types.float32,
                numba.types.float32,
                numba.types.float32,
                numba.types.int32,
                numba.types.int32
            ),
            (
                numba.types.Array(numba.types.float32, 1, "A"),
                NumbaAudioArray,
                numba.types.float32,
                numba.types.float32,
                numba.types.float32,
                numba.types.int32,
                numba.types.int32
            ),
        ],
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def demod_hilbert(analytic_signal, output, min_float, max_float, sample_rate, carrier, deviation):
        two_pi = 2 * pi
        analytic_signal_value = 0
        analytic_signal_value_prev = 0
        discont = pi
        ph_correct = 0
        ph_correct_prev = 0
        carrier_scale = carrier / deviation

        i = 1
        instantaneous_frequency_len = len(output)
        for analytic_signal_value in analytic_signal:
            if i >= instantaneous_frequency_len:
                break

            dd = analytic_signal_value - analytic_signal_value_prev
            analytic_signal_value_prev = analytic_signal_value  #      dd = np.diff(p)

            # FMdemod.unwrap
            ddmod = (
                (dd + pi) % two_pi
            ) - pi  #           ddmod = np.mod(dd + pi, 2 * pi) - pi
            if (
                ddmod == -pi and dd > 0
            ):  #                 to_pi_locations = np.where(np.logical_and(ddmod == -pi, dd > 0))
                ddmod = pi  #                              ddmod[to_pi_locations] = pi
            ph_correct = ddmod - dd  #                     ph_correct = ddmod - dd
            if (
                -dd if dd < 0 else dd
            ) < discont:  #       to_zero_locations = np.where(np.abs(dd) < discont)
                ph_correct = (
                    0  #                          ph_correct[to_zero_locations] = 0
                )

            ph_correct = (
                ph_correct_prev + ph_correct
            )  #    p[1] + np.cumsum(ph_correct).astype(REAL_DTYPE)

            # FMdemod.unwrap_hilbert
            out = (ph_correct - ph_correct_prev) / two_pi * sample_rate / deviation # np.diff(instantaneous_phase) / (2.0 * pi) * sample_rate
            output[i - 1] = max(min_float, min(max_float, out - carrier_scale))
            ph_correct_prev = ph_correct
            i += 1

    @staticmethod
    @njit(
        [
            (
                numba.types.Array(numba.types.float32, 1, "C"),
                NumbaAudioArray,
                numba.types.float32,
                numba.types.float32,
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.int32,
                numba.types.int32,
            ),
            (
                numba.types.Array(numba.types.float32, 1, "A"),
                NumbaAudioArray,
                numba.types.float32,
                numba.types.float32,
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.int32,
                numba.types.int32,
            ),
            (
                numba.types.Array(numba.types.float64, 1, "C"),
                NumbaAudioArray,
                numba.types.float32,
                numba.types.float32,
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.int32,
                numba.types.int32,
            ),
            (
                numba.types.Array(numba.types.float64, 1, "A"),
                NumbaAudioArray,
                numba.types.float32,
                numba.types.float32,
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.Array(DEMOD_DTYPE_NB, 1, "C"),
                numba.types.int32,
                numba.types.int32,
            ),
        ],
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def demod_quadrature(
        in_rf,
        out_demod,
        min_float,
        max_float,
        i_osc,
        q_osc,
        sample_rate,
        deviation,
    ):
        # constants
        two_pi = 2 * pi
        phase_scale = sample_rate / (two_pi * deviation)

        iq_len = len(i_osc)
        rf_len = len(in_rf)

        # sign always start positive, iq index always starts at 0
        prev_i = in_rf[0] * i_osc[0]
        prev_q = in_rf[0] * q_osc[0]

        for i in range(1, rf_len):
            # i/q is only the positive side of the wave, and only one half rotation is stored
            # * the index of the buffer loops around twice for one full rotation
            iq_index = i % iq_len
            # * sign flips between the first and second loop (1st rotation: positive; 2nd rotation: negative)
            sign = 1 - 2 * ((i // iq_len) & 1)
            # * flip the rf sign, to avoid extra multiplications below
            rf_signed = in_rf[i] * sign

            i_in = rf_signed * i_osc[iq_index]
            q_in = rf_signed * q_osc[iq_index]

            # complex-conjugate discriminator, avoids phase error accumulation when wrapping the phase with amplitude
            imag = q_in * prev_i - i_in * prev_q
            real = i_in * prev_i + q_in * prev_q

            # angular difference
            delta = atan2(imag, real)

            # scale up to deviation
            out = delta * phase_scale

            out_demod[i - 1] = min(max(out, min_float), max_float)

            prev_i = i_in
            prev_q = q_in

    # estimate carrier frequency deviation from phase changes in the I/Q reference frame
    # which is like a series of +- values for clean signal, i.e. a fixed carrair wave)
    def work(self, input: np.array, output: np.array):
        if self.type == DEMOD_HILBERT:
            FMDiscriminator.compute_analytic_signal(input)
            FMDiscriminator.demod_hilbert(
                input,
                output,
                self.min_float,
                self.max_float,
                np.float32(self.sample_rate),
                self.carrier,
                self.deviation,
            )
        elif self.type == DEMOD_QUADRATURE:
            FMDiscriminator.demod_quadrature(
                input,
                output,
                self.min_float,
                self.max_float,
                self.i_osc,
                self.q_osc,
                self.sample_rate,
                self.deviation,
            )


# Creates a low or high pass bilinear shelving filter from two time constants
# Where:
#   * T1 is the low shelf frequency
#   * T2 is the high shelf frequency
#
#   ------\ T1
#          \
#           \ center
#            \
#          T2 \________
#
def build_shelf_filter(
    direction,
    tau1,
    tau2,
    fs
):
    # set high point of filter to have no gain
    b_1 = 1 / (tau1 / tau2)

    if direction == "low":
        b_analog = [tau2 ** 2 / tau1, b_1]
        a_analog = [tau1, 1]
        gain = b_analog[1] / a_analog[1]
    else:
        b_analog = [tau2, b_1]
        a_analog = [tau2, 1]
        gain = b_analog[0] / a_analog[0]

    b_analog = [b / gain for b in b_analog]

    b_digital, a_digital = bilinear(b_analog, a_analog, fs)

    return b_digital, a_digital


class SpectralNoiseReduction:
    class SpectralGateNonStationaryNumba(SpectralGateNonStationary):
        # Adapted from noisereduce
        def __init__(self, params):
            super().__init__(**params)

            self.t_frames = self._time_constant_s * self.sr / self._hop_length

            # By default, this solves the equation for b:
            #   b**2  + (1 - b) / t_frames  - 2 = 0
            # which approximates the full-width half-max of the
            # squared frequency response of the IIR low-pass filt
            b = 2 / (sqrt(1 + 4 * self.t_frames ** 2) + 1)
            self.smooth_filter_b = [b]
            self.smooth_filter_a = [1, b - 1]
            self._noverlap = self._win_length - self._hop_length
            self._epsilon = np.finfo(np.float64).eps

        @staticmethod
        @njit(
            numba.types.void(
                numba.types.Array(numba.types.float64, 2, "A"),
                numba.types.Array(numba.types.float32, 2, "F"),
                numba.types.int64,
                numba.types.int64,
                numba.types.float64,
            ),
            cache=True,
            fastmath=True,
            nogil=True,
        )
        def _get_sig_mask(
            sig_stft_smooth,
            abs_sig_stft,
            thresh_n_mult_nonstationary,
            sigmoid_slope_nonstationary,
            epsilon # prevent divide by zero
        ):
            sig_stft_smooth_x, sig_stft_smooth_y = sig_stft_smooth.shape
            thresh_n_mult_nonstationary_plus_one = thresh_n_mult_nonstationary + 1

            for x in range(sig_stft_smooth_x):
                for y in range(sig_stft_smooth_y):
                    sig_stft_smooth[x][y] = 1 / (
                        1 + exp(
                            (
                                thresh_n_mult_nonstationary_plus_one
                                - abs_sig_stft[x][y] / max(sig_stft_smooth[x][y], epsilon)
                            ) * sigmoid_slope_nonstationary
                        )
                    )

        @staticmethod
        @njit(
            numba.types.void(
                numba.types.Array(numba.types.complex64, 2, "F"),
                numba.types.Array(numba.types.float64, 2, "C"),
                numba.types.float64,
            ),
            cache=True,
            fastmath=True,
            nogil=True,
        )
        def _mask_signal(sig_stft, sig_mask, prop_decrease):
            # multiply signal with mask
            sig_mask_x, sig_mask_y = sig_mask.shape

            for x in range(sig_mask_x):
                for y in range(sig_mask_y):
                    sig_stft[x][y] *= 1 + prop_decrease * (sig_mask[x][y] - 1)

        def spectral_gating_nonstationary_single_channel(self, chunk):
            """non-stationary version of spectral gating"""
            _, _, sig_stft = stft(
                chunk,
                nfft=self._n_fft,
                noverlap=self._noverlap,
                nperseg=self._win_length,
                padded=False,
            )
            # get abs of signal stft
            abs_sig_stft = np.abs(sig_stft)

            # get the smoothed mean of the signal
            sig_stft_smooth = filtfilt(
                self.smooth_filter_b,
                self.smooth_filter_a,
                abs_sig_stft,
                axis=-1,
                padtype=None,
            )

            SpectralNoiseReduction.SpectralGateNonStationaryNumba._get_sig_mask(
                sig_stft_smooth,
                abs_sig_stft,
                self._thresh_n_mult_nonstationary,
                self._sigmoid_slope_nonstationary,
                self._epsilon
            )
            sig_mask = sig_stft_smooth

            if self.smooth_mask:
                # convolve the mask with a smoothing filter
                sig_mask = fftconvolve(sig_mask, self._smoothing_filter, mode="same")

            SpectralNoiseReduction.SpectralGateNonStationaryNumba._mask_signal(
                sig_stft, sig_mask, self._prop_decrease
            )
            sig_stft_denoised = sig_stft

            # invert/recover the signal
            _, denoised_signal = istft(
                sig_stft_denoised,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
            )

            return np.ascontiguousarray(denoised_signal, dtype=REAL_DTYPE)

    def __init__(self, audio_rate, nr_reduction_amount):
        self.chunk_size = int(audio_rate / BLOCKS_PER_SECOND)
        self.chunk_count = BLOCKS_PER_SECOND
        self.end_padding = int(self.chunk_size / 8)
        self.nr_reduction_amount = nr_reduction_amount

        self.denoise_params = {
            "y": np.empty(1),
            "sr": audio_rate,
            "chunk_size": self.chunk_size,
            "padding": self.end_padding,
            "prop_decrease": self.nr_reduction_amount,
            "n_fft": 1024,
            "win_length": None,
            "hop_length": None,
            "time_constant_s": (
                (self.chunk_size * self.chunk_count / 2) / audio_rate
            ),  # (self.chunk_size * (self.chunk_count - 1)) / audio_rate,
            "freq_mask_smooth_hz": 500,
            "time_mask_smooth_ms": 50,
            "thresh_n_mult_nonstationary": 2,
            "sigmoid_slope_nonstationary": 10,
            "tmp_folder": None,
            "use_tqdm": False,
            "n_jobs": 1,
        }

        self.spectral_gate = SpectralNoiseReduction.SpectralGateNonStationaryNumba(
            self.denoise_params
        )
        self.chunks = []
        for _ in range(self.chunk_count):
            self.chunks.append(np.zeros(self.chunk_size, dtype=REAL_DTYPE, order="C"))

    def spectral_nr(self, audio_in, audio_out):
        chunk = np.ascontiguousarray(
            np.concatenate([
                self.chunks[0],
                self.chunks[1],
                audio_in,
                np.zeros(self.end_padding)
            ], dtype=REAL_DTYPE)
        )
        self.chunks.append(audio_in.copy())
        self.chunks.pop(0)

        nr = self.spectral_gate.spectral_gating_nonstationary_single_channel(chunk)

        DecoderSharedMemory.copy_data_src_offset_float32(
            nr, audio_out, len(nr) - len(audio_out) - self.end_padding, len(audio_out)
        )

class DCBlocker:
    def __init__(self, sample_rate, cutoff):
        stages = 3

        # Per-stage cutoff scaling (correct cascade compensation)
        scale = 1.0 / np.sqrt(2.0**(1.0 / stages) - 1.0)
        stage_cutoff = cutoff * scale

        self.R = np.exp(-2 * np.pi * stage_cutoff / sample_rate)

        # Stage 1 state
        self.x1 = np.float64(0.0)
        self.y1 = np.float64(0.0)

        # Stage 2 state
        self.x2 = np.float64(0.0)
        self.y2 = np.float64(0.0)

        # Stage 3 state
        self.x3 = np.float64(0.0)
        self.y3 = np.float64(0.0)

    @staticmethod
    @njit(
        [
            numba.types.UniTuple(numba.types.float64, 6)(
                NumbaAudioArray,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64
            )
        ],
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def dc_block(audio, x1, y1, x2, y2, x3, y3, R):
        for i in range(audio.shape[0]):
            x = audio[i]

            # Stage 1
            y1_new = x - x1 + R * y1
            x1 = x
            y1 = y1_new

            # Stage 2
            y2_new = y1_new - x2 + R * y2
            x2 = y1_new
            y2 = y2_new

            # Stage 3
            y3_new = y2_new - x3 + R * y3
            x3 = y2_new
            y3 = y3_new

            audio[i] = y3_new

        return x1, y1, x2, y2, x3, y3

    def process(self, audio):
        self.x1, self.y1, self.x2, self.y2, self.x3, self.y3 = self.dc_block(
            audio,
            self.x1, self.y1,
            self.x2, self.y2,
            self.x3, self.y3,
            self.R,
        )

class Deemphasis:
    def __init__(
        self,
        audio_rate,
        deemphasis_low_tau: float,
        deemphasis_high_tau: float,
    ):
        self.audio_rate = audio_rate

        # deemphasis filter for output audio
        self.deemphasis_T1 = deemphasis_low_tau
        self.deemphasis_T2 = deemphasis_high_tau

        self.deemph_b, self.deemph_a = build_shelf_filter(
            "low",
            self.deemphasis_T1,
            self.deemphasis_T2,
            self.audio_rate,
        )
        self.zi_deemph_x = 0.0
        self.zi_deemph_y = 0.0

    def get_response(self):
        # compute frequency response
        w, h_deemph = freqz(self.deemph_b, self.deemph_a, worN=4096, fs=self.audio_rate)

        magnitude_db = 20 * np.log10(np.abs(h_deemph))

        return w, magnitude_db

    @staticmethod
    @njit(
        [
            (
                numba.types.Array(numba.types.float32, 1, "C"),
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
            ),
            (
                numba.types.Array(numba.types.float64, 1, "C"),
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
                numba.types.float64,
            )
        ],
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def lfilt_inplace(x, b0, b1, a1, zi_x, zi_y):
        """
        In-place first-order IIR filter (lfilter equivalent).
    
        Implements:
            y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
        """

        for i in range(x.shape[0]):
            xi = x[i]
            yi = b0 * xi + b1 * zi_x - a1 * zi_y
    
            x[i] = yi
    
            zi_x = xi
            zi_y = yi
    
        return zi_x, zi_y

    def process(self, audio_out):
        self.zi_deemph_x, self.zi_deemph_y = Deemphasis.lfilt_inplace(
            audio_out,
            self.deemph_b[0],
            self.deemph_b[1],
            self.deemph_a[1],
            self.zi_deemph_x,
            self.zi_deemph_y
        )

def simple_lowpass(fs, tau):
    b_analog = [1]
    a_analog = [tau, 1]

    b_digital, a_digital = bilinear(b_analog, a_analog, fs)
    return b_digital, a_digital

class Expander:
    def __init__(
        self,
        audio_rate,
        gain: float,
        ratio: float,
        env_detection: str,
        attack_tau: float,
        hold_tau: float,
        release_tau: float,
        weighting_low_tau: float,
        weighting_high_tau: float,
        weighting_low_pass: float,
        weighting_low_pass_transition: float,
    ):
        self.audio_rate = audio_rate

        # makeup gain to apply after expansion
        self.gain = gain
        self.ratio = float(ratio)

        self.atkCoeff = exp(-1 / (attack_tau * self.audio_rate))
        self.relCoeff = exp(-1 / (release_tau * self.audio_rate))
        self.hold_samples = round(hold_tau * self.audio_rate)

        if env_detection == ENV_DETECTION_RMS:
            self.use_rms = True
            self.env_lin = 1e-12
        else:
            self.use_rms = False
            self.env_lin = 0.0

        self.hold_state = 0

        # weighted filter for envelope detector
        self.weighting_T1 = weighting_low_tau
        self.weighting_T2 = weighting_high_tau
        self.env_iirb, self.env_iira = build_shelf_filter(
            "high",
            self.weighting_T1,
            self.weighting_T2,
            self.audio_rate,
        )
        self.zi_x = 0.0
        self.zi_y = 0.0

    def get_response(self):
        # compute frequency response
        _, h_low = freqz(self.lowpass_iirb, self.lowpass_iira, worN=4096, fs=self.audio_rate)
        w, h_high = freqz(self.env_iirb, self.env_iira, worN=4096, fs=self.audio_rate)
    
        h_total = (
            h_low * 
            h_high
        )
    
        magnitude_db = 20 * np.log10(np.abs(h_total))

        return w, magnitude_db

    @staticmethod
    @njit(
        [(
            NumbaAudioArray,
            numba.types.Array(numba.types.float32, 1, "C"),
            numba.types.float64,
            numba.types.float64,
            numba.types.float64,
            numba.types.int32,
            numba.types.int32,
            numba.types.float64,
            numba.types.float64,
            numba.types.boolean
        )],
        cache=True,
        fastmath=False,
        nogil=True
    )
    def expand(
        audio,
        side_chain,
        env_lin,
        atkCoeff,
        relCoeff,
        hold_state,
        hold_samples,
        makeup_gain_db,
        ratio,
        use_rms
    ):
        n = audio.shape[0]
        epsilon = np.finfo(np.float64).eps
        one_minus_atkCoeff = 1 - atkCoeff

        ratio_minus_one = ratio - 1
        inv20 = 0.05 #1 / 20
        
        # simulates a diode-capacitor peak or rms detector
        if use_rms:
            # rms detector
            for i in range(n):
                sc = side_chain[i]
                sc_squared = sc * sc

                attacking = sc_squared > env_lin

                if attacking:
                    env_lin = atkCoeff * env_lin + one_minus_atkCoeff * sc_squared
                    hold_state = hold_samples
                else:
                    if hold_state > 0:
                        hold_state -= 1
                    else:
                        env_lin = relCoeff * env_lin

                side_chain[i] = sqrt(env_lin + epsilon)
        else:
            # peak detector
            for i in range(n):
                sc = abs(side_chain[i])
                attacking = sc > env_lin

                if attacking:
                    # capacitor charging
                    # rate is dependent on sc and the attack speed
                    env_lin = atkCoeff * env_lin + one_minus_atkCoeff * sc
                    hold_state = hold_samples
                    # capacitor discharging
                else:
                    if hold_state > 0:
                        hold_state -= 1
                    else:
                        # rate is dependent only on the release speed
                        env_lin = relCoeff * env_lin

                side_chain[i] = env_lin

        # voltage controlled amplifier, driven by the peak detector
        # apply expansion to audio (can be vectorized)
        for i in range(n):
            env_db = 20.0 * log10(max(side_chain[i], epsilon))
            target_gain_db = ratio_minus_one * env_db + makeup_gain_db

            audio[i] *= 10 ** (target_gain_db * inv20)

        return env_lin, hold_state

    def process(self, pre_in, audio_out):
        # high pass weighted input to envelope detector
        self.zi_x, self.zi_y = Deemphasis.lfilt_inplace(
            pre_in,
            self.env_iirb[0],
            self.env_iirb[1],
            self.env_iira[1],
            self.zi_x,
            self.zi_y
        )

        self.env_lin, self.hold_state = Expander.expand(
            audio_out,
            pre_in,
            self.env_lin,
            self.atkCoeff,
            self.relCoeff,
            self.hold_state,
            self.hold_samples,
            self.gain,
            self.ratio,
            self.use_rms
        )


@dataclass
class HiFiAudioParams:
    pre_trim: int
    gain: int
    decode_mode: str
    preview: bool
    demod_type: str
    auto_fine_tune: bool
    format: str
    if_rate: int
    input_rate: int
    audio_rate: int
    ifresample_numerator: int
    ifresample_denominator: int
    if_resampler_converter: str
    audioRes_numerator: int
    audioRes_denominator: int
    audio_resampler_converter: str
    audio_final_rate: int
    audioFinal_numerator: int
    audioFinal_denominator: int
    audio_final_resampler_converter: str
    headswitch_interpolation_enabled: bool
    headswitch_passes: int
    headswitch_signal_rate: int
    headswitch_hz: int
    headswitch_drift_hz: int
    headswitch_cutoff_freq: int
    headswitch_peak_prominence_limit: int
    headswitch_interpolation_padding: int
    headswitch_interpolation_neighbor_range: int
    hs_sos: float
    doc_mode: bool
    doc_cutoff_freq: int
    doc_window_size: int
    doc_window_hop_size: int
    doc_fade_samples: int
    doc_dc_offset_samples: int
    doc_amplitude_threshold: int
    doc_std_threshold: int
    doc_fft_start: int
    doc_fft_end: int


class HiFiDecode:
    def __init__(self, options=None, is_main_process=True, bias_guess=False):
        if options is None:
            options = dict()
        self.options = options
        self.is_main_process = is_main_process
        self.decode_mode = options["mode"]
        self.gain = options["gain"]

        self.input_rate: int = int(options["input_rate"])
        if self.options["demod_type"] == DEMOD_HILBERT:
            self.if_rate: int = DEMOD_HILBERT_IF_RATE
        elif self.options["demod_type"] == DEMOD_QUADRATURE:
            self.if_rate: int = (
                self.input_rate
            )  # do not resample rf when doing quadrature demodulation

        self.audio_rate: int = 192000
        self.audio_final_rate: int = int(options["audio_rate"])

        (
            self.ifresample_numerator,
            self.ifresample_denominator,
            self.audioRes_numerator,
            self.audioRes_denominator,
            self.audioFinal_numerator,
            self.audioFinal_denominator,
        ) = HiFiDecode.getResamplingRatios(
            self.input_rate, self.if_rate, self.audio_rate, self.audio_final_rate
        )

        if self.options["resampler_quality"] == "high":
            self.if_resampler_converter = "VHQ"
            self.audio_resampler_converter = "VHQ"
            self.audio_final_resampler_converter = "VHQ"
        elif self.options["resampler_quality"] == "medium":
            self.if_resampler_converter = "LQ"
            self.audio_resampler_converter = "MQ"
            self.audio_final_resampler_converter = "HQ"
        else:  # low
            self.if_resampler_converter = "LQ"
            self.audio_resampler_converter = "LQ"
            self.audio_final_resampler_converter = "LQ"

        self.if_resampler = ResampleStream(self.ifresample_denominator, self.ifresample_numerator, 1, np.float32, self.if_resampler_converter)
        self.audio_resampler_l = ResampleStream(self.audioRes_denominator, self.audioRes_numerator, 1, REAL_DTYPE, self.audio_resampler_converter)
        self.audio_resampler_r = ResampleStream(self.audioRes_denominator, self.audioRes_numerator, 1, REAL_DTYPE, self.audio_resampler_converter)
        self.audio_final_resampler_l = ResampleStream(self.audioFinal_denominator, self.audioFinal_numerator, 1, REAL_DTYPE, self.audio_final_resampler_converter)
        self.audio_final_resampler_r = ResampleStream(self.audioFinal_denominator, self.audioFinal_numerator, 1, REAL_DTYPE, self.audio_final_resampler_converter)

        self.set_block_sizes()
        self._set_block_overlap()

        self.standard, self.field_rate = get_standard(
            options["format"],
            options["standard"],
            options["afe_left_carrier_deviation"],
            options["afe_right_carrier_deviation"],
            options["afe_left_carrier"],
            options["afe_right_carrier"],
        )
        self.standard_original = deepcopy(self.standard)

        self.headswitch_interpolation_enabled = self.options[
            "head_switching_interpolation"
        ]
        self.headswitch_passes = 1
        self.headswitch_signal_rate = self.audio_rate
        # frequency used to fit peaks to the expected headswitching interval
        self.headswitch_hz = self.field_rate  
        self.headswitch_drift_hz = self.field_rate * 0.1  # +- 10% drift is normal
        # filter cutoff frequency for peak detection
        self.headswitch_cutoff_freq = 28000
        # upper limit for peak promience which is used to widen the interpoation boundary
        # when a peak is detected. Head switching peaks may exceed this number; however,
        # the width of the interpolation will only be expanded up to this number.
        # If there is noise that is a greater duration than this, it should probably be muted instead.
        self.headswitch_peak_prominence_limit = 3
        # maximum seconds to widen the interpolation based on the strength of the peak
        #   setting too low prevents interpolation from covering the whole pulse
        #   setting too high causes audible artifacts where the interpolation occurs
        self.headswitch_interpolation_padding = 35e-6
        # time range to look for neighboring noise when a head switch event is detected
        # this helps to determine the correct width of the interpolation
        self.headswitch_interpolation_neighbor_range = 200e-6
        hs_sos = cheby2(
            22,
            rs=200,
            Wn=self.headswitch_cutoff_freq,
            btype="highpass",
            output="sos",
            fs=self.headswitch_signal_rate
        )

        # hifi carrier loss results in broadband noise
        # mute the audio when this broadband noise exists above the audible frequencies
        self.doc_mode = self.options["doc"]
        self.doc_audio_rate = self.audio_rate
        # number of samples to fade out before doc, and fade in after doc
        self.doc_fade_samples = 128
        # number of samples to compensate for dc offset 
        self.doc_dc_offset_samples = 256
        # dropout detection thresholds
        self.doc_window_size = 128
        self.doc_window_hop_size = 128
        self.doc_amplitude_threshold = 1  # 1 is 100% gain
        self.doc_std_threshold = 1  # +- standard deviation of fft before dropout correction

        # remove signals that may be from the recorded audio
        self.doc_cutoff_freq = 40000
        doc_fft_freqs = np.fft.fftfreq(
            self.doc_window_size, d=1 / self.doc_audio_rate
        )
        for i in range(len(doc_fft_freqs)):
            freq = doc_fft_freqs[i]
            if freq < 0:
                self.doc_fft_end = i - 1
                break
            if freq <= self.doc_cutoff_freq:
                self.doc_fft_start = i

        if not bias_guess:
            # defer until carriers can be determined
            self.afeL, self.afeR = self._get_afe()

            # plot_responses(self.afeL, self.afeR)
            # exit()

            self.fmL, self.fmR = self._get_fm_demod(
                self.if_rate, self.options["demod_type"]
            )

        if self.options["grc"]:
            print(f"Set gnuradio sample rate at {self.if_rate} Hz, type float")
            if ZMQ_AVAILABLE:
                self.grc = ZMQSend()
            else:
                print(
                    "ZMQ library is not available, please install the zmq python library to use this feature!"
                )

        self.audio_process_params = HiFiAudioParams(
            pre_trim=self.pre_trim,
            gain=self.gain,
            decode_mode=self.decode_mode,
            preview=options["preview"],
            demod_type=options["demod_type"],
            auto_fine_tune=options["auto_fine_tune"],
            format=options["format"],
            if_rate=self.if_rate,
            input_rate=self.input_rate,
            audio_rate=self.audio_rate,
            ifresample_numerator=self.ifresample_numerator,
            ifresample_denominator=self.ifresample_denominator,
            if_resampler_converter=self.if_resampler_converter,
            audioRes_numerator=self.audioRes_numerator,
            audioRes_denominator=self.audioRes_denominator,
            audio_resampler_converter=self.audio_resampler_converter,
            audio_final_rate=self.audio_final_rate,
            audioFinal_numerator=self.audioFinal_numerator,
            audioFinal_denominator=self.audioFinal_denominator,
            audio_final_resampler_converter=self.audio_final_resampler_converter,
            headswitch_interpolation_enabled=self.headswitch_interpolation_enabled,
            headswitch_passes=self.headswitch_passes,
            headswitch_signal_rate=self.headswitch_signal_rate,
            headswitch_hz=self.headswitch_hz,
            headswitch_drift_hz=self.headswitch_drift_hz,
            headswitch_cutoff_freq=self.headswitch_cutoff_freq,
            headswitch_peak_prominence_limit=self.headswitch_peak_prominence_limit,
            headswitch_interpolation_padding=self.headswitch_interpolation_padding,
            headswitch_interpolation_neighbor_range=self.headswitch_interpolation_neighbor_range,
            hs_sos=hs_sos,
            doc_mode=self.doc_mode,
            doc_cutoff_freq=self.doc_cutoff_freq,
            doc_window_size=self.doc_window_size,
            doc_window_hop_size=self.doc_window_hop_size,
            doc_fade_samples=self.doc_fade_samples,
            doc_dc_offset_samples=self.doc_dc_offset_samples,
            doc_amplitude_threshold=self.doc_amplitude_threshold,
            doc_std_threshold=self.doc_std_threshold,
            doc_fft_start=self.doc_fft_start,
            doc_fft_end=self.doc_fft_end,
        )

    def calculate_block_sizes(self, block_size=None):
        # block overlap and edge discard
        blocks_per_second_ratio = 1 / BLOCKS_PER_SECOND

        if block_size == None:
            block_size: int = ceil(
                self.input_rate * blocks_per_second_ratio
            )
        else:
            block_size: int = block_size
            blocks_per_second_ratio: float = block_size / self.input_rate

        block_resampled_size: int = ceil(
            self.if_rate * blocks_per_second_ratio
        )
        block_audio_size: int = ceil(
            self.audio_rate * blocks_per_second_ratio
        )
        block_audio_final_size: int = ceil(
            self.audio_final_rate * blocks_per_second_ratio
        )

        return {
            "blocks_per_second_ratio": blocks_per_second_ratio,
            "block_size": block_size,
            "block_resampled_size": block_resampled_size,
            "block_audio_size": block_audio_size,
            "block_audio_final_size": block_audio_final_size,
        }

    def set_block_sizes(self, block_size=None):
        block_calculations = self.calculate_block_sizes(block_size)

        self._blocks_per_second_ratio = block_calculations["blocks_per_second_ratio"]
        self._initial_block_size = block_calculations["block_size"]
        self._initial_block_resampled_size = block_calculations["block_resampled_size"]
        self._initial_block_audio_size = block_calculations["block_audio_size"]
        self._initial_block_audio_final_size = block_calculations["block_audio_final_size"]

        return {
            "block_size": self._initial_block_size,
            "block_audio_size": self._initial_block_audio_size,
            "block_audio_final_size": self._initial_block_audio_final_size,
        }

    def _set_block_overlap(self):
        # Blocks are overlapped before and after by `block_overlap samples``
        # Block n+1 is:
        #   * Last half of the start overlap
        #   * All the non-overlapping data
        #   * First half of the end overlap
        # Block 0 is:
        #   * Copied data from beginning of non-overlapping data to simulate an overlap
        #   * All the non-overlapping data
        #   * First half of the end overlap
        # Block end is:
        #   * Last half of the start overlap
        #   * All the non-overlapping data up to the end of the audio
        #   * First half of the end overlap (last half of this overlap is discarded)
        #
        # reads:                |0000|1111111111|1111|          |2222|3333333333|3333|
        #        xxxx|0000000000|0000|          |1111|2222222222|2222|          |3333|44444444444444
        #          ^               ^               ^               ^               ^
        # block:   0               1               2               3               4

        # use the greatest common divisor to calculate the minimum size of overlap samples so it divides evenly against the input and final sample rates
        block_size_gcd = np.gcd.reduce(
            [self._initial_block_size, self._initial_block_audio_final_size]
        )

        if block_size_gcd > 5:
            block_audio_overlap_divisor = int(
                self._initial_block_audio_size / block_size_gcd
            )
        else:
            print(
                f"WARNING: The input sample rate is not evenly divisible by the output sample rate. Audio sync issues may occur. Input Rate: {self.input_rate}, Output Rate: {self.audio_final_rate}."
            )
            block_audio_overlap_divisor = 1

        # trims out the discontinuity errors at the beginning and end of each block
        # the duration of the discontiunity errors seem to increase as the distance between the carrier frequency and nyquist limit of the rf decreases
        self.pre_trim = 1000

        # minimum overlap to account for loss during resampling
        min_resampler_overlap = self.pre_trim + 50
        # min overlap in terms of final sample rate
        min_overlap = ceil(
            min_resampler_overlap / self.audio_rate * self.audio_final_rate
        )
        # overlap rounded up to the nearest evenly divisible chunk
        self._block_audio_final_overlap = (
            ceil(min_overlap / block_audio_overlap_divisor)
            * block_audio_overlap_divisor
        )

        overlap_seconds = self._block_audio_final_overlap / self.audio_final_rate

        self._block_overlap = round(self.input_rate * overlap_seconds)
        self._block_audio_overlap = ceil(self.input_rate * overlap_seconds)
        self._block_read_overlap = self._block_overlap * 2
        self._block_audio_final_overlap = round(self.audio_final_rate * overlap_seconds)

    def get_block_overlap(self):
        return {
            "block_read_overlap": self._block_read_overlap,
            "block_overlap": self._block_overlap,
            "block_audio_final_overlap": self._block_audio_final_overlap,
        }

    @staticmethod
    def getResamplingRatios(input_rate, if_rate, audio_rate, audio_final_rate):
        samplerate2ifrate = if_rate / input_rate

        ifresample_numerator = Fraction(samplerate2ifrate).numerator
        ifresample_denominator = Fraction(samplerate2ifrate).denominator
        assert (
            ifresample_numerator > 0
        ), f"IF resampling numerator got 0; sample_rate {input_rate}"
        assert (
            ifresample_denominator > 0
        ), f"IF resampling denominator got 0; sample_rate {input_rate}"

        audiorate2ifrate = audio_rate / if_rate
        audioRes_numerator = Fraction(audiorate2ifrate).numerator
        audioRes_denominator = Fraction(audiorate2ifrate).denominator

        audiorate2FinalAudioRate = audio_final_rate / audio_rate
        audioFinal_numerator = Fraction(audiorate2FinalAudioRate).numerator
        audioFinal_denominator = Fraction(audiorate2FinalAudioRate).denominator

        return (
            ifresample_numerator,
            ifresample_denominator,
            audioRes_numerator,
            audioRes_denominator,
            audioFinal_numerator,
            audioFinal_denominator,
        )

    def _get_fm_demod(self, if_rate, demod_type):
        return (
            FMDiscriminator(
                if_rate,
                self.standard.LCarrierRef,
                self.standard.LCarrierDeviation,
                self.initialBlockResampledSize,
                demod_type
            ),
            FMDiscriminator(
                if_rate,
                self.standard.RCarrierRef,
                self.standard.RCarrierDeviation,
                self.initialBlockResampledSize,
                demod_type
            )
        )

    def guessBiases(self, blocks: list[np.array]) -> Tuple[float, float]:
        meanL, meanR = StackableMA(window_average=len(blocks)), StackableMA(
            window_average=len(blocks)
        )

        (
            ifresample_numerator,
            ifresample_denominator,
            _,
            _,
            _,
            _,
        ) = HiFiDecode.getResamplingRatios(
            self.input_rate,
            DEMOD_HILBERT_IF_RATE,
            self.audio_rate,
            self.audio_final_rate,
        )
        afeL, afeR = self._get_afe()
        fmL, fmR = self._get_fm_demod(
            DEMOD_HILBERT_IF_RATE, DEMOD_HILBERT
        )

        progressB = TimeProgressBar(len(blocks), len(blocks))

        for i in range(len(blocks)):
            data = blocks[i].astype(REAL_DTYPE, copy=False)

            data = resample(
                data,
                ifresample_denominator,
                ifresample_numerator,
                "LQ",
            )

            filterL = afeL.work(data)
            filterR = afeR.work(data)
            preL = np.empty(len(filterL), dtype=REAL_DTYPE, order="C")
            preR = np.empty(len(filterR), dtype=REAL_DTYPE, order="C")

            fmL.work(filterL.astype(DEMOD_DTYPE_NP), preL)
            fmR.work(filterR.astype(DEMOD_DTYPE_NP), preR)

            # remove dc spikes at beginning and end
            preL = preL[self.pre_trim : -self.pre_trim]
            preR = preR[self.pre_trim : -self.pre_trim]

            meanL.push(np.mean(preL))
            meanR.push(np.mean(preR))

            meanLResult = meanL.pull() * self.standard.LCarrierDeviation + self.standard.LCarrierRef + 1e6
            meanRResult = meanR.pull() * self.standard.RCarrierDeviation + self.standard.RCarrierRef + 1e6

            progressB.label = "Carrier L %.06f MHz, R %.06f MHz" % (
                meanLResult / 10e5,
                meanRResult / 10e5,
            )
            progressB.print(i + 1, False)

        # update the standard
        self.afeL, self.afeR = self._get_afe(meanLResult, meanRResult)

        # update all the filters and iq oscilators with the new bias
        self.fmL, self.fmR = self._get_fm_demod(
            self.if_rate, self.options["demod_type"]
        )

        return meanLResult, meanRResult

    def log_bias(self, dcL, dcR):
        devL = dcL * self.standard.LCarrierDeviation / 1e3
        devR = dcR * self.standard.RCarrierDeviation / 1e3

        if self.audio_process_params.decode_mode == AUDIO_MODE_MONO_L:
            print("Bias L %.02f kHz" % (devL), end=" ")
        elif self.audio_process_params.decode_mode == AUDIO_MODE_MONO_R:
            print("Bias R %.02f kHz" % (devR), end=" ")
        else:
            print("Bias L %.02f kHz, R %.02f kHz" % (devL, devR), end=" ")

        if abs(devL) < 9 and abs(devR) < 9:
            print("(good player/recorder calibration)")
        elif 9 <= abs(devL) < 10 or 9 <= abs(devR) < 10:
            print("(maybe marginal player/recorder calibration)")
        else:
            print(
                "\nWARN: the player or the recorder may be uncalibrated and/or\n"
                "the standard and/or the sample rate specified are wrong"
            )

    def _get_afe(self, newLC=None, newRC=None):
        if newLC:
            self.standard.LCarrierRef = (
                max(
                    min(newLC, self.standard_original.LCarrierRef + 10e3),
                    self.standard_original.LCarrierRef - 10e3,
                )
                if self.options["format"] == "vhs"
                else newLC
            )

        if newRC:
            self.standard.RCarrierRef = (
                max(
                    min(newRC, self.standard_original.RCarrierRef + 10e3),
                    self.standard_original.RCarrierRef - 10e3,
                )
                if self.options["format"] == "vhs"
                else newRC
            )

        afeL = AFEFilterable(self.standard, self.if_rate, 0)
        afeR = AFEFilterable(self.standard, self.if_rate, 1)

        return afeL, afeR

    def auto_fine_tune(
        self, dcL: float, dcR: float
    ) -> Tuple[AFEFilterable, AFEFilterable, FMDiscriminator, FMDiscriminator]:
        if self.audio_process_params.decode_mode != AUDIO_MODE_MONO_R:
            left_carrier_updated = round(self.standard_original.LCarrierRef + dcL * self.standard.LCarrierDeviation)

            self.standard.LCarrierRef = min(max(left_carrier_updated, self.standard_original.LCarrierRef - 10e3), self.standard_original.LCarrierRef + 10e3)
            
        if self.audio_process_params.decode_mode != AUDIO_MODE_MONO_L:
            right_carrier_updated = round(self.standard_original.RCarrierRef + dcR * self.standard.RCarrierDeviation)

            self.standard.RCarrierRef = min(max(right_carrier_updated, self.standard_original.RCarrierRef - 10e3), self.standard_original.RCarrierRef + 10e3)

        self.afeL, self.afeR = self._get_afe()
        self.fmL, self.fmR = self._get_fm_demod(
            self.if_rate, self.options["demod_type"]
        )

    @staticmethod
    @njit(
        numba.types.containers.Tuple((numba.types.float32, numba.types.float32))(
            numba.types.Array(numba.types.float32, 1, "A")
        ),
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def mean_stddev(signal: np.array) -> tuple[float, float]:
        signal_mean = np.mean(signal)
        signal_std_dev = sqrt(np.mean(np.abs(signal - signal_mean) ** 2))

        return signal_mean, signal_std_dev

    @staticmethod
    def headswitch_detect_peaks(
        audio: np.array,
        audio_process_params: HiFiAudioParams,
    ) -> Tuple[list[Tuple[float, float, float, float]], np.array, np.array]:
        # remove audible frequencies to avoid detecting them as peaks
        filtered_signal = sosfiltfilt_rust(
            audio_process_params.hs_sos, audio
        )
        filtered_signal = np.ascontiguousarray(filtered_signal, dtype=REAL_DTYPE)
        filtered_signal_abs = abs(filtered_signal)
        filtered_signal_mean, filtered_signal_std_dev = HiFiDecode.mean_stddev(
            filtered_signal
        )

        # detect the peaks that align with the headswitching speed
        # peaks should align roughly to the frame rate of the video system
        # account for small drifts in the head switching pulse (shows up as the sliding dot on video)
        peak_distance_seconds = 1 / (
            audio_process_params.headswitch_hz
            + audio_process_params.headswitch_drift_hz
        )
        peak_centers, peak_center_props = find_peaks(
            filtered_signal_abs,
            distance=peak_distance_seconds
            * audio_process_params.headswitch_signal_rate,
            width=1,
        )

        # setup the neighboring peak theshold based on stadard deviation
        neighbor_threshold = filtered_signal_mean + filtered_signal_std_dev
        neighbor_search_width = round(
            audio_process_params.headswitch_interpolation_neighbor_range
            * audio_process_params.headswitch_signal_rate
        )

        peaks = []
        # add the peaks and search around the center peak for any neighboring noise that is above the threshold
        for i in range(len(peak_centers)):
            peak_start = peak_center_props["left_ips"][i]
            peak_end = peak_center_props["right_ips"][i]

            peaks.append(
                (
                    peak_centers[i],
                    peak_start,
                    peak_end,
                    # limit peak prominence from balooning too high in empty audio or broadband noise causing an entire chunk to be interpolated
                    max(
                        min(
                            peak_center_props["prominences"][i],
                            audio_process_params.headswitch_peak_prominence_limit,
                        ),
                        0,
                    ),
                )
            )

            start_neighbor = max(0, floor(peak_start - neighbor_search_width))
            end_neighbor = min(
                ceil(peak_end + neighbor_search_width), len(filtered_signal_abs)
            )

            # search for neighboring peaks and add to the list
            neighbor_peak_centers, neighbor_peak_props = find_peaks(
                filtered_signal_abs[start_neighbor:end_neighbor],
                threshold=neighbor_threshold,
                prominence=0.25,
                distance=1,
                width=1,
            )

            for peak_log_neighbor_idx in range(len(neighbor_peak_centers)):
                peaks.append(
                    (
                        neighbor_peak_centers[peak_log_neighbor_idx] + start_neighbor,
                        neighbor_peak_props["left_ips"][peak_log_neighbor_idx]
                        + start_neighbor,
                        neighbor_peak_props["right_ips"][peak_log_neighbor_idx]
                        + start_neighbor,
                        # limit peak prominence from balooning too high in empty audio or broadband noise causing an entire chunk to be interpolated
                        max(
                            min(
                                neighbor_peak_props["prominences"][
                                    peak_log_neighbor_idx
                                ],
                                audio_process_params.headswitch_peak_prominence_limit,
                            ),
                            0,
                        ),
                    )
                )

        return peaks, filtered_signal, filtered_signal_abs

    @staticmethod
    def headswitch_calc_boundaries(
        peaks: list[tuple[int, int, int, float]],
        audio_process_params: HiFiAudioParams,
    ) -> list[list[int, int]]:
        peak_boundaries = list()

        # scale the peak width depending on how much the peak stands out from the base signal
        # use light scaling for headswtich peaks since they are usually very brief
        padding_samples = round(
            audio_process_params.headswitch_interpolation_padding
            * audio_process_params.headswitch_signal_rate
        )
        for peak_center, peak_start, peak_end, peak_prominence in peaks:
            width_padding = peak_prominence * padding_samples
            # start and end peak at its bases
            start = floor(peak_start - width_padding)
            end = ceil(peak_end + width_padding)

            peak_boundaries.append([start, end])

        # merge overlapping or duplicate boundaries
        return HiFiDecode.merge_boundaries(peak_boundaries)

    @staticmethod
    @njit(
        numba.types.void(NumbaAudioArray, NumbaAudioArray, numba.types.int32),
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def smooth(data_in: np.array, data_out: np.array, half_window: int):
        data_in_len = len(data_in)
        for i in range(data_in_len):
            start = max(0, i - half_window)
            end = min(data_in_len, i + half_window + 1)
            data_out[i] = np.mean(data_in[start:end])  # Apply moving average

    @staticmethod
    def merge_boundaries(boundaries):
        # merge overlapping or duplicate boundaries
        boundaries.sort(key=lambda x: x[0])
        merged = list()

        for boundary in boundaries:
            if not merged or merged[-1][1] < boundary[0]:
                merged.append(boundary)
            else:
                merged[-1] = [merged[-1][0], max(merged[-1][1], boundary[1])]

        return merged

    @staticmethod
    def headswitch_interpolate_boundaries(
        audio: np.array, boundaries: list[list[int, int]]
    ) -> np.array:
        interpolated_signal = np.empty_like(audio, order="C")
        interpolator_in = np.empty_like(audio, order="C")
        DecoderSharedMemory.copy_data_float32(
            audio, interpolated_signal, len(interpolated_signal)
        )
        DecoderSharedMemory.copy_data_float32(
            audio, interpolator_in, len(interpolator_in)
        )

        # setup interpolator input by copying and removing any samples that are peaks
        time = np.arange(len(interpolated_signal), dtype=float)

        for [start, end] in boundaries:
            time[start:end] = np.nan
            interpolator_in[start:end] = np.nan

        time = time[np.logical_not(np.isnan(time))]
        interpolator_in = interpolator_in[np.logical_not(np.isnan(interpolator_in))]

        # interpolate the gap where the peak was removed
        interpolator = interp1d(
            time,
            interpolator_in,
            kind="linear",
            copy=False,
            assume_sorted=True,
            fill_value="extrapolate",
        )

        for [start, end] in boundaries:
            smoothing_size = 1 + end - start

            # sample and hold inteerpolation if boundaries are beyond this chunk
            if start < 0:
                interpolated_signal[0:end] = interpolated_signal[end]
            elif end > len(audio):
                interpolated_signal[start : len(audio)] = interpolated_signal[start]
            else:
                for i in range(start, end):
                    interpolated_signal[i] = interpolator(i)
                # smooth linear interpolation
                smoothed_out = interpolated_signal[
                    start - smoothing_size : end + smoothing_size
                ]
                smoothed_in = np.empty_like(smoothed_out, order="C")
                DecoderSharedMemory.copy_data_float32(
                    smoothed_out, smoothed_in, len(smoothed_in)
                )
                HiFiDecode.smooth(smoothed_in, smoothed_out, ceil(smoothing_size / 4))

        return interpolated_signal

    @staticmethod
    def headswitch_remove_noise(
        audio: np.array, audio_process_params: HiFiAudioParams
    ) -> np.array:
        for _ in range(audio_process_params.headswitch_passes):
            peaks, filtered_signal, filtered_signal_abs = (
                HiFiDecode.headswitch_detect_peaks(audio, audio_process_params)
            )
            interpolation_boundaries = HiFiDecode.headswitch_calc_boundaries(
                peaks, audio_process_params
            )
            interpolated_audio = HiFiDecode.headswitch_interpolate_boundaries(
                audio, interpolation_boundaries
            )

            # uncomment to debug head switching pulse detection
            # HiFiDecode.debug_peak_interpolation(audio, filtered_signal, filtered_signal_abs, peaks, interpolation_boundaries, interpolated_audio, audio_process_params.headswitch_signal_rate)
            # plt.show()

            audio = interpolated_audio

        return interpolated_audio

    # size of the raw data block coming in
    @property
    def initialBlockSize(self) -> int:
        return self._initial_block_size

    # size of the resampled IF data
    @property
    def initialBlockResampledSize(self) -> int:
        return self._initial_block_resampled_size

    # size of the audio decoded audio before resampling
    @property
    def initialBlockAudioSize(self) -> int:
        return self._initial_block_audio_size

    # size of the audio decoded audio after resampling
    @property
    def initialBlockFinalAudioSize(self) -> int:
        return self._initial_block_audio_final_size

    @property
    def blockOverlap(self) -> int:
        return self._block_overlap

    @property
    def blockAudioFinalOverlap(self) -> int:
        return self._block_audio_final_overlap

    @property
    def sourceRate(self) -> int:
        return int(self.input_rate)

    @property
    def audioRate(self) -> int:
        return self.audio_rate

    @property
    def notchFreq(self) -> float:
        return self.standard_original.Hfreq

    @staticmethod
    @njit(
        [numba.types.float32(NumbaAudioArray, numba.types.int16)],
        cache=True,
        fastmath=True,
    )
    def cancelDC_trim(audio: np.array, trim: int) -> float:
        dc = REAL_DTYPE(np.mean(audio[trim:-trim]))

        for i in range(trim, len(audio) - trim):
            audio[i] = audio[i] - dc

        for i in range(trim):
            audio[i] = 0

        for i in range(len(audio) - trim, len(audio)):
            audio[i] = 0

        return dc

    @staticmethod
    def _detect_dropouts(audio: np.array, audio_process_params: HiFiAudioParams):
        # Parameters for the sliding window
        window_size = (
            audio_process_params.doc_window_size
        )  # Size of each window (in samples)
        hop_size = (
            audio_process_params.doc_window_hop_size
        )  # Hop size (how much to move the window each time)

        dropout_ranges = []
        full_spectrum_ranges_count = 0

        fft_start = audio_process_params.doc_fft_start
        fft_end = audio_process_params.doc_fft_end

        # Loop through the audio in overlapping windows
        for start in range(0, len(audio) - window_size, hop_size):
            # Extract the current window of the signal
            end = start + window_size
            window = audio[start:end]

            # Compute the FFT of the window
            fft_output = np.fft.fft(window)

            # Compute the magnitude of the FFT, using the positive frequencies only
            magnitude = np.abs(fft_output[fft_start:fft_end]).astype(REAL_DTYPE)

            magnitude_mean, magnitude_std = HiFiDecode.mean_stddev(magnitude)

            if (
                magnitude_mean > audio_process_params.doc_amplitude_threshold
                and magnitude_std > audio_process_params.doc_std_threshold
            ):
                # start the dropout, if not currently in a dropout
                if len(dropout_ranges) == full_spectrum_ranges_count:
                    dropout_ranges.append(
                        [start, len(audio)]
                    )  # default to end doc at the end of the audio
            else:
                # end the dropout, if currently in a dropout
                if len(dropout_ranges) > full_spectrum_ranges_count:
                    dropout_ranges[full_spectrum_ranges_count][1] = end
                    full_spectrum_ranges_count += 1

        # sort and merge any overlapping boundaries
        return HiFiDecode.merge_boundaries(dropout_ranges)
    
    @staticmethod
    def _check_other_channel(gaps_to_fill, source_gaps):
        result = []

        for start, end in gaps_to_fill:
            # Check overlap with source gaps
            overlap_found = False
            for s_start, s_end in source_gaps:
                # If no overlap, skip
                if end <= s_start or start >= s_end:
                    continue
                overlap_found = True
                # Safe copy = part of gap outside overlap
                if start < s_start:
                    result.append(((start, s_start), None))
                # Missing in both = overlap
                overlap_start = max(start, s_start)
                overlap_end = min(end, s_end)
                result.append((None, (overlap_start, overlap_end)))
                start = overlap_end
            # Remaining part after last overlap
            if not overlap_found or start < end:
                result.append(((start, end), None))

        return result

    @staticmethod
    @njit(
        numba.types.void(
            numba.types.int64,
            numba.types.int64,
            NumbaAudioArray,
            numba.optional(NumbaAudioArray),
            numba.types.int64,
            numba.types.int64,
            numba.types.boolean,
            numba.types.float32
        ),
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def _fill(start, end, outer, inner, fade_samples, dc_window, mute, epsilon):
        # ------- calculate the start of the fade --------
        fade_start = max(0, start - fade_samples)
        fade_start_duration = start - fade_start

        dc_before_start = max(0, fade_start - dc_window)
        dc_before_end = start
        if dc_before_start < dc_before_end:
            dc_before = np.mean(outer[dc_before_start:dc_before_end])
        else:
            dc_before = 0

        # ------- calculate the end of the fade --------
        fade_end = min(len(outer), end + fade_samples)
        fade_end_duration = fade_end - end

        dc_after_start = end
        dc_after_end = min(len(outer), fade_end + dc_window)
        if dc_after_start < dc_after_end:
            dc_after = np.mean(outer[dc_after_start:dc_after_end])
        else:
            dc_after = 0

        # DC of inner data over the gap
        dc_inner = 0 if mute else np.mean(inner[fade_start:fade_end])

        # interpolate DC across the gap
        # prevents artifacts introduced with dc offset difference between the channels
        dc_total_len = fade_end - fade_start
        dc_interp_full = np.zeros(dc_total_len, dtype=REAL_DTYPE)

        if dc_total_len > 1:
            denom = float(dc_total_len - 1)
            delta = dc_after - dc_before
        
            for i in range(dc_total_len):
                t = i / denom
                smooth = 0.5 * (1.0 - np.cos(np.pi * t))
                dc_interp_full[i] = dc_before + delta * smooth
        else:
            dc_interp_full[0] = dc_before

        # -------- cross fade outer -> inner (before gap) --------
        for i in range(fade_start_duration):
            idx = fade_start + i
            dc_idx = idx - fade_start

            fade_in_factor = i / fade_start_duration
            fade_out_factor = 1 - fade_in_factor

            outer_sample = outer[idx] * fade_out_factor
            inner_sample = (epsilon if mute else inner[idx] - dc_inner + dc_interp_full[dc_idx]) * fade_in_factor

            outer[idx] = outer_sample + inner_sample

        # -------- copy inner (gap) --------
        for i in range(start, end):
            dc_idx = i - fade_start

            inner_sample = epsilon if mute else inner[i]
            outer[i] = (
                inner_sample
                - dc_inner # subtract current dc offset
                + dc_interp_full[dc_idx] # add interpolated dc offset
            )

        # -------- cross fade inner -> outer (after gap) --------
        for i in range(fade_end_duration):
            idx = end + i
            dc_idx = idx - fade_start

            fade_in_factor = (i + 1) / fade_end_duration
            fade_out_factor = 1 - fade_in_factor

            outer_sample = outer[idx] * fade_in_factor
            inner_sample = (epsilon if mute else inner[idx] - dc_inner + dc_interp_full[dc_idx]) * fade_out_factor

            outer[idx] = outer_sample + inner_sample


    @staticmethod
    def _mute_or_fill(dropouts, current_channel, other_channel, fade_samples, dc_window):
        epsilon = np.finfo(np.float16).eps

        for fill_range, mute_range in dropouts:
            # cross fade from other channel
            if fill_range is not None:
                HiFiDecode._fill(
                    fill_range[0],
                    fill_range[1],
                    current_channel,
                    other_channel,
                    fade_samples,
                    dc_window,
                    False,
                    epsilon
                )

            # fade to silence
            if mute_range is not None:
                HiFiDecode._fill(
                    mute_range[0],
                    mute_range[1],
                    current_channel,
                    other_channel,
                    fade_samples,
                    dc_window,
                    True,
                    epsilon
                )

    @staticmethod
    def dropout_compensate(audioL: np.array, audioR: np.array, audio_process_params: HiFiAudioParams) -> np.array:
        dual_mono = (
            audio_process_params.decode_mode == AUDIO_MODE_DUAL_MONO or
            audio_process_params.decode_mode == AUDIO_MODE_DUAL_MONO_MS
        )

        if audio_process_params.decode_mode != AUDIO_MODE_MONO_R:
            dropout_boundaries_l = HiFiDecode._detect_dropouts(audioL, audio_process_params)
        else:
            # fully muted since this is right channel only
            dropout_boundaries_l = [(0, len(audioR))]

        if audio_process_params.decode_mode != AUDIO_MODE_MONO_L:
            dropout_boundaries_r = HiFiDecode._detect_dropouts(audioR, audio_process_params)
        else:
            # fully muted since this is left channel only
            dropout_boundaries_r = [(0, len(audioL))]

        # check each other channel for any data that can be used as dropout compensation
        if dual_mono or audio_process_params.doc_mode == DOC_MODE_MUTE:
            # don't fill from other channel
            dropouts_right = [(None, (start,end)) for start,end in dropout_boundaries_r]
            dropouts_left = [(None, (start,end)) for start,end in dropout_boundaries_l]
        else:
            if audio_process_params.decode_mode != AUDIO_MODE_MONO_R:
                dropouts_left = HiFiDecode._check_other_channel(dropout_boundaries_l, dropout_boundaries_r)
            else:
                # don't fill from other channel
                dropouts_right = [(None, (start,end)) for start,end in dropout_boundaries_r]
    
            if audio_process_params.decode_mode != AUDIO_MODE_MONO_L:
                dropouts_right = HiFiDecode._check_other_channel(dropout_boundaries_r, dropout_boundaries_l)
            else:
                # don't fill from other channel
                dropouts_left = [(None, (start,end)) for start,end in dropout_boundaries_l]

        # apply doc / filling with dropout compensation data
        if audio_process_params.decode_mode != AUDIO_MODE_MONO_R:
            HiFiDecode._mute_or_fill(
                dropouts_left,
                audioL,
                audioR,
                audio_process_params.doc_fade_samples,
                audio_process_params.doc_dc_offset_samples
            )
        if audio_process_params.decode_mode != AUDIO_MODE_MONO_L:
            HiFiDecode._mute_or_fill(
                dropouts_right,
                audioR,
                audioL,
                audio_process_params.doc_fade_samples,
                audio_process_params.doc_dc_offset_samples
            )

    @staticmethod
    def mix_for_mode_stereo(
        l_raw: np.array, r_raw: np.array, decode_mode: str
    ) -> Tuple[np.array, np.array]:
        if (
            decode_mode == AUDIO_MODE_STEREO_MS or 
            decode_mode == AUDIO_MODE_DUAL_MONO_MS
        ):
            l = np.multiply(np.add(l_raw, r_raw), 0.5, order="C")
            r = np.multiply(np.subtract(l_raw, r_raw), 0.5, order="C")
        elif decode_mode == AUDIO_MODE_MONO_L:
            l = l_raw
            r = l_raw
        elif decode_mode == AUDIO_MODE_MONO_R:
            l = r_raw
            r = r_raw
        elif decode_mode == AUDIO_MODE_MONO_SUM:
            l = np.multiply(np.add(l_raw, r_raw), 0.5, order="C")
            r = np.multiply(np.add(l_raw, r_raw), 0.5, order="C")
        else:
            # AUDIO_MODE_STEREO
            # AUDIO_MODE_DUAL_MONO_STEREO
            l = l_raw
            r = r_raw

        return np.ascontiguousarray(l), np.ascontiguousarray(r)

    @staticmethod
    @njit(
        numba.types.void(NumbaAudioArray, numba.types.float32),
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def adjust_gain(audio: np.array, gain: float) -> np.array:
        for i in range(len(audio)):
            audio[i] = audio[i] * gain

    @staticmethod
    def demod_process_audio(
        filtered: np.array, fm: FMDiscriminator, audio_process_params: HiFiAudioParams, audio_resampler, measure_perf: bool
    ) -> Tuple[np.array, float, dict]:
        perf_measurements = {
            "start_demod": 0,
            "end_demod": 0,
            "start_audio_resample": 0,
            "end_audio_resample": 0,
            "start_dc_trim": 0,
            "end_dc_trim": 0,
            "start_headswitch": 0,
            "end_headswitch": 0,
            "start_audio_final_resample": 0,
            "end_audio_final_resample": 0,
        }

        # demodulate
        if measure_perf:
            perf_measurements["start_demod"] = perf_counter()
        audio = np.empty(len(filtered), dtype=REAL_DTYPE, order="C")

        # estimate carrier frequency deviation from phase changes in the I/Q reference frame
        fm.work(filtered, audio)
        if measure_perf:
            perf_measurements["end_demod"] = perf_counter()

        # band-limited sinc interpolation to audio rate (i.e. downsample from IF rate to audio rate)
        if measure_perf:
            perf_measurements["start_audio_resample"] = perf_counter()
        audio: np.array = audio_resampler.resample_chunk(audio, True)
        audio_resampler.clear()
        if measure_perf:
            perf_measurements["end_audio_resample"] = perf_counter()

        # cancel dc based on mean, remove spikes at end of signal
        if measure_perf:
            perf_measurements["start_dc_trim"] = perf_counter()
        dc = HiFiDecode.cancelDC_trim(audio, audio_process_params.pre_trim)
        if measure_perf:
            perf_measurements["end_dc_trim"] = perf_counter()

        return audio, dc, perf_measurements

    @staticmethod
    def head_switch_resample(
        audio: np.array,
        audio_process_params: HiFiAudioParams,
        audio_final_resampler,
        perf_measurements: dict,
        measure_perf: bool
    ) -> Tuple[np.array, float, dict]:
        # do head switching noise cancellation if enabled
        if measure_perf:
            perf_measurements["start_headswitch"] = perf_counter()
        if audio_process_params.headswitch_interpolation_enabled:
            audio = HiFiDecode.headswitch_remove_noise(audio, audio_process_params)
        if measure_perf:
            perf_measurements["end_headswitch"] = perf_counter()

        # resample audio sample rate to final audio sample rate
        if measure_perf:
            perf_measurements["start_audio_final_resample"] = perf_counter()
        if audio_process_params.audio_rate != audio_process_params.audio_final_rate:
            audio: np.array = audio_final_resampler.resample_chunk(audio, True)
            audio_final_resampler.clear()
        if measure_perf:
            perf_measurements["end_audio_final_resample"] = perf_counter()

        return audio, perf_measurements

    def block_decode(
        self,
        rf_data: np.array,
        measure_perf: bool = False,
    ) -> Tuple[int, np.array, np.array]:
        # resample from input sample rate to if sample rate
        if measure_perf:
            start_if_resampler = perf_counter()

        if self.options["demod_type"] == DEMOD_HILBERT:
            rf_data = rf_data.astype(REAL_DTYPE, copy=False)
            rf_data_resampled = self.if_resampler.resample_chunk(rf_data, True)
            self.if_resampler.clear()
        else:
            rf_data_resampled = rf_data

        if measure_perf:
            end_if_resampler = perf_counter()

        if measure_perf:
            start_carrier_filter = perf_counter()
        if self.audio_process_params.decode_mode != AUDIO_MODE_MONO_R: filterL = self.afeL.work(rf_data_resampled)
        if self.audio_process_params.decode_mode != AUDIO_MODE_MONO_L: filterR = self.afeR.work(rf_data_resampled)
        if measure_perf:
            end_carrier_filter = perf_counter()

        if self.options["grc"] and ZMQ_AVAILABLE:
            self.grc.send(filterL + filterR)

        # demodulate, resample to audio rate
        if self.audio_process_params.decode_mode != AUDIO_MODE_MONO_R: 
            preL, dcL, perf_measurements_l = HiFiDecode.demod_process_audio(
                filterL, self.fmL, self.audio_process_params, self.audio_resampler_l, measure_perf
            )
        else:
            preL = None
            dcL = 0
            perf_measurements_l = 0

        if self.audio_process_params.decode_mode != AUDIO_MODE_MONO_L: 
            preR, dcR, perf_measurements_r = HiFiDecode.demod_process_audio(
                filterR, self.fmR, self.audio_process_params, self.audio_resampler_r, measure_perf
            )
        else:
            preR = None
            dcR = 0
            perf_measurements_r = 0

        # dropout compensation
        # try to copy from the other channel otherwise
        # mute audio when carrier loss occurs
        if measure_perf:
            perf_measurements_l["start_doc"] = perf_counter()
            perf_measurements_r["start_doc"] = perf_measurements_l["start_doc"]
        if self.audio_process_params.doc_mode != DOC_MODE_DISABLED:
            HiFiDecode.dropout_compensate(preL, preR, self.audio_process_params)
        if measure_perf:
            perf_measurements_l["end_doc"] = perf_counter()
            perf_measurements_r["end_doc"] = perf_measurements_l["end_doc"]

        # headswitch interpolation, resample to final rate
        if self.audio_process_params.decode_mode != AUDIO_MODE_MONO_R:
            preL, perf_measurements_l = HiFiDecode.head_switch_resample(preL, self.audio_process_params, self.audio_final_resampler_l, perf_measurements_l, measure_perf)

        if self.audio_process_params.decode_mode != AUDIO_MODE_MONO_L:
            preR, perf_measurements_r = HiFiDecode.head_switch_resample(preR, self.audio_process_params, self.audio_final_resampler_r, perf_measurements_r, measure_perf)

        # fine tune carrier frequency
        if measure_perf:
            start_auto_fine_tune = perf_counter()
        if self.options["auto_fine_tune"]:
            self.auto_fine_tune(dcL, dcR)
        if measure_perf:
            end_auto_fine_tune = perf_counter()

        self.log_bias(dcL, dcR)

        # mix for various stereo modes
        if measure_perf:
            start_stereo_mix = perf_counter()
        preL, preR = HiFiDecode.mix_for_mode_stereo(
            preL, preR, self.audio_process_params.decode_mode
        )
        if measure_perf:
            end_stereo_mix = perf_counter()

        if measure_perf:
            start_adjust_gain = perf_counter()
        if self.audio_process_params.gain != 1:
            HiFiDecode.adjust_gain(preL, self.audio_process_params.gain)
            HiFiDecode.adjust_gain(preR, self.audio_process_params.gain)
        if measure_perf:
            end_adjust_gain = perf_counter()

        assert (
            preL.dtype == REAL_DTYPE
        ), f"Audio data must be in {REAL_DTYPE} format, instead got {preL.dtype}"
        assert (
            preR.dtype == REAL_DTYPE
        ), f"Audio data must be in {REAL_DTYPE} format, instead got {preR.dtype}"

        if measure_perf:
            duration_if_resampler = end_if_resampler - start_if_resampler
            duration_carrier_filter = end_carrier_filter - start_carrier_filter

            duration_demod_l = (
                perf_measurements_l["end_demod"] - perf_measurements_l["start_demod"]
            )
            duration_audio_resample_l = (
                perf_measurements_l["end_audio_resample"]
                - perf_measurements_l["start_audio_resample"]
            )
            duration_dc_trim_l = (
                perf_measurements_l["end_dc_trim"]
                - perf_measurements_l["start_dc_trim"]
            )
            duration_doc_l = (
                perf_measurements_l["end_doc"] - perf_measurements_l["start_doc"]
            )
            duration_headswitch_l = (
                perf_measurements_l["end_headswitch"]
                - perf_measurements_l["start_headswitch"]
            )
            duration_audio_final_resample_l = (
                perf_measurements_l["end_audio_final_resample"]
                - perf_measurements_l["start_audio_final_resample"]
            )

            duration_demod_r = (
                perf_measurements_r["end_demod"] - perf_measurements_r["start_demod"]
            )
            duration_audio_resample_r = (
                perf_measurements_r["end_audio_resample"]
                - perf_measurements_r["start_audio_resample"]
            )
            duration_dc_trim_r = (
                perf_measurements_r["end_dc_trim"]
                - perf_measurements_r["start_dc_trim"]
            )
            duration_doc_r = (
                perf_measurements_r["end_doc"] - perf_measurements_r["start_doc"]
            )
            duration_headswitch_r = (
                perf_measurements_r["end_headswitch"]
                - perf_measurements_r["start_headswitch"]
            )
            duration_audio_final_resample_r = (
                perf_measurements_r["end_audio_final_resample"]
                - perf_measurements_r["start_audio_final_resample"]
            )

            duration_auto_fine_tune = end_auto_fine_tune - start_auto_fine_tune
            duration_stereo_mix = end_stereo_mix - start_stereo_mix
            duration_adjust_gain = end_adjust_gain - start_adjust_gain
            durations = [
                ("duration_if_resampler", duration_if_resampler),
                ("duration_carrier_filter", duration_carrier_filter),
                ("duration_demod_l", duration_demod_l),
                ("duration_audio_resample_l", duration_audio_resample_l),
                ("duration_dc_trim_l", duration_dc_trim_l),
                ("duration_doc_l", duration_doc_l),
                ("duration_headswitch_l", duration_headswitch_l),
                ("duration_audio_final_resample_l", duration_audio_final_resample_l),
                ("duration_demod_r", duration_demod_r),
                ("duration_audio_resample_r", duration_audio_resample_r),
                ("duration_dc_trim_r", duration_dc_trim_r),
                ("duration_doc_r", duration_doc_r),
                ("duration_headswitch_r", duration_headswitch_r),
                ("duration_audio_final_resample_r", duration_audio_final_resample_r),
                ("duration_auto_fine_tune", duration_auto_fine_tune),
                ("duration_stereo_mix", duration_stereo_mix),
                ("duration_adjust_gain", duration_adjust_gain),
            ]
            durations.sort(reverse=True, key=lambda x: x[1])
            print("decode performance", durations)

        return preL, preR

    @staticmethod
    def hifi_decode_worker(
        decoder_in_queue, decoder_out_queue, decode_options, standard, numa_node
    ):
        setproctitle(current_process().name)
        measure_perf = False
        decoder = HiFiDecode(decode_options, is_main_process=False)
        decoder.standard = standard

        # @profile
        def decode_next_block():
            while True:
                try:
                    decoder_state = decoder_in_queue.get()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            buffer = DecoderSharedMemory(decoder_state)
            raw_data = buffer.get_block()

            audioL, audioR = decoder.block_decode(
                raw_data,
                measure_perf,
            )

            if measure_perf:
                start_final_audio_copy = perf_counter()
            # copy the audio data into the shared buffer
            l_out = buffer.get_pre_left()
            r_out = buffer.get_pre_right()

            if decoder_state.is_last_block:
                # only copy the last part of the data
                overlap_to_trim = len(audioL) - (decoder_state.block_audio_final_len + decoder_state.block_audio_final_overlap)
            else:
                # shift the audio left to remove the block overlap
                overlap_to_trim = max(0, round((len(audioL) - decoder_state.block_audio_final_len) / 2))

            DecoderSharedMemory.copy_data_src_offset_float32(
                audioL, l_out, overlap_to_trim, decoder_state.block_audio_final_len
            )
            DecoderSharedMemory.copy_data_src_offset_float32(
                audioR, r_out, overlap_to_trim, decoder_state.block_audio_final_len
            )
            if measure_perf:
                end_final_audio_copy = perf_counter()

            if measure_perf:
                final_audio_copy_duration = (
                    end_final_audio_copy - start_final_audio_copy
                )
                print("final_audio_copy_duration:", final_audio_copy_duration)
                print()

            buffer.close()
            decoder_out_queue.put(decoder_state)

        while True:
            decode_next_block()

    @staticmethod
    def debug_peak_interpolation(
        audio,
        filtered_signal,
        filtered_signal_abs,
        peaks,
        interpolation_boundaries,
        interpolated,
        headswitch_signal_rate,
    ):
        fs = headswitch_signal_rate
        t = np.arange(0, len(audio)) / fs
        fft_signal = np.fft.fft(audio)
        fft_freqs = np.fft.fftfreq(len(t), 1 / fs)

        # Only keep the positive half of the frequency spectrum
        positive_freqs = fft_freqs[: len(t) // 2]
        positive_fft_signal = np.abs(fft_signal[: len(t) // 2])

        # Perform the FFT on the filtered signal
        fft_filtered_signal = np.fft.fft(filtered_signal)
        positive_fft_filtered_signal = np.abs(fft_filtered_signal[: len(t) // 2])

        plt.figure(figsize=(10, 6))
        plt.plot(positive_freqs, positive_fft_signal, label="Original Signal Spectrum")
        plt.plot(
            positive_freqs,
            positive_fft_filtered_signal,
            label="Filtered Signal Spectrum",
            color="orange",
        )
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.legend()

        plt.figure(figsize=(10, 6))

        plt.subplot(4, 1, 1)
        plt.plot(t, filtered_signal, label="Filtered Signal", color="green")
        plt.title("Filtered Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()

        peak_centers = [round(x[0]) for x in peaks]
        peak_starts = [round(x[1]) for x in peaks]
        peak_ends = [round(x[2]) for x in peaks]
        peak_prominences = [x[3] for x in peaks]

        interpolation_starts = [
            max(0, start) for start, end in interpolation_boundaries
        ]
        interpolation_ends = [
            min(end, len(audio) - 1) for start, end in interpolation_boundaries
        ]

        plt.subplot(4, 1, 2)
        plt.plot(
            t,
            filtered_signal_abs,
            label="Filtered Signal Absolute Value",
            color="black",
        )
        plt.plot(
            t[interpolation_starts],
            filtered_signal_abs[interpolation_starts],
            "r+",
            label="Interpolation Start",
        )
        plt.plot(t[peak_starts], filtered_signal_abs[peak_starts], "b+", label="Start")
        plt.plot(t[peak_centers], peak_prominences, "gx", label="Prominence")
        plt.plot(
            t[peak_centers], filtered_signal_abs[peak_centers], "go", label="Center"
        )
        plt.plot(t[peak_ends], filtered_signal_abs[peak_ends], "bx", label="End")
        plt.plot(
            t[interpolation_ends],
            filtered_signal_abs[interpolation_ends],
            "rx",
            label="Interpolation End",
        )
        plt.title("Filtered Signal with head switch points")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(t, interpolated, label="Interpolated audio", color="green")
        plt.title("Interpolated Audio")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(t, audio, label="Original Signal")
        plt.title("Original Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()
