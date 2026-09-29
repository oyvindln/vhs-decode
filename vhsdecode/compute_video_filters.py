import math
import numpy as np
import scipy.signal as sps
from collections import namedtuple

from lddecode.utils import filtfft, supergauss
from vhsdecode.addons.FMdeemph import gen_shelf

NONLINEAR_AMP_LPF_FREQ_DEFAULT = 700000
NONLINEAR_STATIC_FACTOR_DEFAULT = None
CHROMA_AUDIO_NOTCH_Q = 10


def create_sub_emphasis_params(rf_params, sys_params, hz_ire, vsync_ire):
    return namedtuple(
        "SubEmphasisParams",
        "exponential_scaling scaling_1 scaling_2 logistic_mid logistic_rate static_factor deviation",
    )(
        rf_params.get("nonlinear_exp_scaling", 0.25),
        rf_params.get("nonlinear_scaling_1", None),
        rf_params.get("nonlinear_scaling_2", None),
        rf_params.get("nonlinear_logistic_mid", None),
        rf_params.get("nonlinear_logistic_rate", None),
        rf_params.get("nonlinear_static_factor", NONLINEAR_STATIC_FACTOR_DEFAULT),
        sys_params.get(
            "nonlinear_deviation",
            hz_ire * (100 + -vsync_ire),
        ),
    )


def gen_video_main_deemp_fft_params(rf_params, freq_hz, block_len):
    """Generate real-value fft main video deemphasis filter from parameters"""
    return gen_video_main_deemp_fft(
        rf_params["deemph_gain"],
        rf_params["deemph_mid"],
        rf_params.get("deemph_q", 1 / 2),
        freq_hz,
        block_len,
    )


def gen_video_main_deemp_fft(gain, mid, Q, freq_hz, block_len):
    """Generate real-value fft main video deemphasis filter from parameters"""
    # The de-emphasis is the inverse of the high shelf describing the pre-emphasis,
    # so the shelf's numerator becomes the denominator and vice versa.
    da, db = gen_shelf(mid, gain, "high", freq_hz, Q)
    return filtfft((db, da), block_len)[: block_len // 2 + 1]


def gen_analog_filter(zeros_tau, poles_tau, freq_hz, block_len):
    """Generate real-value fft filter from an analog transfer function built from
    first-order real zeros and poles given as time constants in seconds:
        H(s) = prod(1 + s * tz) / prod(1 + s * tp)
    """
    # Evaluated on the analog frequency axis rather than through a bilinear transform,
    # so there is no warping near nyquist and the response is the same at any sample rate.
    s = 2j * np.pi * np.linspace(0, freq_hz / 2.0, block_len // 2 + 1)
    ret = np.ones(len(s), dtype=np.complex128)
    for tau in zeros_tau:
        ret *= 1 + s * tau
    for tau in poles_tau:
        ret /= 1 + s * tau
    return ret


def gen_custom_video_filters(filter_list, freq_hz, block_len):
    ret = 1
    for f in filter_list:
        match f["type"]:
            case "analog":
                ret *= gen_analog_filter(
                    f["zeros_tau"], f["poles_tau"], freq_hz, block_len
                )
            case "highshelf":
                db, da = gen_shelf(f["midfreq"], f["gain"], "high", freq_hz / 2.0, f["q"])
                ret *= filtfft((db, da), block_len)[: block_len // 2 + 1]
            case "lowshelf":
                db, da = gen_shelf(f["midfreq"], f["gain"], "low", freq_hz / 2.0, f["q"])
                ret *= filtfft((db, da), block_len)[: block_len // 2 + 1]
    return ret


def gen_peaking_constq(wn, dbgain, bw):
    """Constant-Q peaking biquad, wn and bw normalized to nyquist (bw as octaves of that)."""
    a = 10.0 ** (dbgain / 20.0)
    q = 1 / (2 * math.sinh(math.log(2) / 2 * bw))
    return sps.bilinear(
        *sps.lp2lp(
            np.array([1, a / q, 1]),
            np.array([1, 1 / q, 1]),
            wo=4 * math.tan(math.pi * wn / 2),
        ),
        fs=2.0,
    )


def gen_video_lpf(corner_freq, order, nyquist_hz, block_len):
    """Generate real-value fir and fft post-demodulation low pass filters from parameters"""
    video_lpf_b = sps.butter(order, corner_freq / nyquist_hz, "lowpass", output="sos")
    video_lpf_fft = abs(
        sps.sosfreqz(video_lpf_b, block_len, whole=True)[1][: block_len // 2 + 1]
    )

    return (video_lpf_b, video_lpf_fft)


def gen_video_lpf_supergauss(corner_freq, order, nyquist_hz, block_len):
    return supergauss(
        np.linspace(0, nyquist_hz, block_len // 2 + 1),
        corner_freq,
        order,
    )


def gen_video_lpf_supergauss_params(rf_params, nyquist_hz, block_len):
    ## TODO: This generates a filter at half the corner frequency!
    return gen_video_lpf_supergauss(
        rf_params["video_lpf_freq"], rf_params["video_lpf_order"], nyquist_hz, block_len
    )


def gen_video_lpf_params(rf_params, nyquist_hz, block_len):
    """Generate real-value fir and fft post-demodulation low pass filters from parameters"""
    return gen_video_lpf(
        rf_params["video_lpf_freq"],
        rf_params["video_lpf_order"],
        nyquist_hz,
        block_len,
    )


def gen_nonlinear_bandpass_params(rf_params, nyquist_hz, block_len):
    """Generate bandpass or highpass real-value fft filter used for non-linear filtering."""
    upper_freq = rf_params.get("nonlinear_bandpass_upper", None)
    order = rf_params.get("nonlinear_bandpass_order", 1)
    lower_freq = rf_params["nonlinear_highpass_freq"]

    return gen_nonlinear_bandpass(upper_freq, lower_freq, order, nyquist_hz, block_len)


def gen_nonlinear_amplitude_lpf(corner_freq, nyquist_hz, order=1):
    return sps.butter(1, corner_freq / nyquist_hz, btype="lowpass", output="sos")


def gen_nonlinear_bandpass(upper_freq, lower_freq, order, nyquist_hz, block_len):
    """Generate bandpass or highpass real-value fft filter used for non-linear filtering."""

    # Use a bandpass filter if upper frequency is specified, otherwise we use a high-pass filter.
    if upper_freq:
        nl_highpass_filter = filtfft(
            sps.butter(
                order,
                [
                    lower_freq / nyquist_hz,
                    upper_freq / nyquist_hz,
                ],
                btype="bandpass",
            ),
            block_len,
        )
    else:
        nl_highpass_filter = filtfft(
            sps.butter(
                order,
                lower_freq / nyquist_hz,
                btype="highpass",
            ),
            block_len,
        )

    return nl_highpass_filter[: block_len // 2 + 1]


def gen_fm_audio_notch_params(rf_params, notch_q, nyquist_hz, block_len):
    """Generate dual notch filter for fm audio frequencies specified in rf_params
    assumes these keys exist.
    """
    return gen_fft_notch(
        rf_params["fm_audio_channel_0_freq"], notch_q, nyquist_hz, block_len
    ) * gen_fft_notch(
        rf_params["fm_audio_channel_1_freq"], notch_q, nyquist_hz, block_len
    )


def gen_fft_notch(notch_freq, notch_q, nyquist_hz, block_len):
    return filtfft(sps.iirnotch(notch_freq / nyquist_hz, notch_q), block_len)


def gen_ramp_filter(
    start_freq_hz: float,
    boost_start: float,
    # max_freq_hz: float,
    boost_max: float,
    nyquist_freq_hz: float,
    block_len: int,
) -> np.ndarray:

    max_freq_hz = 20e6

    zero_ratio = int((start_freq_hz / nyquist_freq_hz) * (block_len // 2))

    zero_part = np.zeros(int(zero_ratio))
    ramp_part = np.linspace(
        boost_start,
        boost_max * (nyquist_freq_hz / max_freq_hz),
        (block_len // 2) - zero_ratio,
    )
    ramp = np.concatenate((zero_part, ramp_part))
    output = np.concatenate((ramp, np.flip(ramp)))
    return output


def gen_ramp_filter_params(rf_params, nyquist_freq_hz, block_len):
    return gen_ramp_filter(
        rf_params.get("start_rf_linear", 0),
        rf_params.get("boost_rf_linear_0", 0),
        # rf_params.get("max_rf_linear", 20e6),
        rf_params.get("boost_rf_linear_20", 1),
        nyquist_freq_hz,
        block_len,
    )
