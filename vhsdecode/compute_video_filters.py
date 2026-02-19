import math
import numpy as np
import scipy.signal as sps
import sys
from collections import namedtuple

if sys.version_info[1] < 10:
    from importlib_resources import files
else:
    # Need Python 3.10 for using namespace in files
    from importlib.resources import files

from vhsdecode.utils import filtfft
from vhsdecode.addons.FMdeemph import FMDeEmphasisB, gen_shelf

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
    db, da = FMDeEmphasisB(
        freq_hz,
        gain,
        mid,
        Q,
    ).get()

    filter_deemp = filtfft((db, da), block_len, whole=False)
    return filter_deemp


def gen_custom_video_filters(filter_list, freq_hz, block_len):
    ret = 1
    for f in filter_list:
        match f["type"]:
            case "file":
                try:
                    file_path = files("vhsdecode.format_defs").joinpath(
                        f["filename"] + "-" + str(int(freq_hz)) + ".txt"
                    )
                    # file_path = osp.join(osp.dirname(__file__), "format_defs", f["filename"]+"-"+str(int(freq_hz))+".txt")
                    ret *= np.loadtxt(file_path, dtype=np.complex128)
                except FileNotFoundError:
                    print(
                        f"Warning: Cannot load filter from file for samplerate of {freq_hz} Hz! Output will likely not look correct!",
                        file=sys.stderr,
                    )
            case "highshelf":
                db, da = gen_shelf(
                    f["midfreq"], f["gain"], "high", freq_hz / 2.0, qfactor=f["q"]
                )
                ret *= filtfft((db, da), block_len, whole=False)
            case "lowshelf":
                db, da = gen_shelf(
                    f["midfreq"], f["gain"], "low", freq_hz / 2.0, qfactor=f["q"]
                )
                ret *= filtfft((db, da), block_len, whole=False)
    return ret


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


def gen_bpf_supergauss(freq_low, freq_high, order, nyquist_hz, block_len):
    return supergauss(
        np.linspace(0, nyquist_hz, block_len // 2 + 1),
        freq_high - freq_low,
        order,
        (freq_high + freq_low) / 2.0,
    )


def supergauss(x, freq, order=1, centerfreq=0):
    return np.exp(
        -2
        * np.power(
            (2 * (x - centerfreq) * (math.log(2.0) / 2.0) ** (1 / (2 * order))) / freq,
            2 * order,
        )
    )


def gen_video_lpf_params(rf_params, nyquist_hz, block_len):
    """Generate real-value fir and fft post-demodulation low pass filters from parameters"""
    if rf_params.get("video_lpf_supergauss", False):
        return (
            None,
            supergauss(
                np.linspace(0, nyquist_hz, block_len // 2 + 1),
                rf_params["video_lpf_freq"],
                rf_params["video_lpf_order"],
            ),
        )
    else:
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
            whole=False,
        )
    else:
        nl_highpass_filter = filtfft(
            sps.butter(
                order,
                lower_freq / nyquist_hz,
                btype="highpass",
            ),
            block_len,
            whole=False,
        )

    return nl_highpass_filter


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
    return filtfft(
        sps.iirnotch(notch_freq / nyquist_hz, notch_q), block_len, whole=True
    )


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
