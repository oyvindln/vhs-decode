import scipy.signal as sps
from collections import namedtuple


from vhsdecode.utils import filtfft
from vhsdecode.addons.FMdeemph import FMDeEmphasisB

NONLINEAR_AMP_LPF_FREQ_DEFAULT = 700000
NONLINEAR_STATIC_FACTOR_DEFAULT = None


def create_sub_emphasis_params(rf_params, sys_params, hz_ire, vsync_ire):
    return namedtuple(
        "SubEmphasisParams",
        "exponential_scaling scaling_1 scaling_2 static_factor deviation",
    )(
        rf_params.get("nonlinear_exp_scaling", 0.25),
        rf_params.get("nonlinear_scaling_1", None),
        rf_params.get("nonlinear_scaling_2", None),
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


def gen_video_lpf(corner_freq, order, nyquist_hz, block_len):
    """Generate real-value fir and fft post-demodulation low pass filters from parameters"""
    video_lpf = sps.butter(order, corner_freq / nyquist_hz, "low")

    video_lpf_fft = filtfft(video_lpf, block_len, False)
    return (video_lpf, video_lpf_fft)


def gen_video_lpf_params(rf_params, nyquist_hz, block_len):
    """Generate real-value fir and fft post-demodulation low pass filters from parameters"""
    return gen_video_lpf(
        rf_params["video_lpf_freq"], rf_params["video_lpf_order"], nyquist_hz, block_len
    )


def gen_nonlinear_bandpass_params(rf_params, nyquist_hz, block_len):
    """Generate bandpass or highpass real-value fft filter used for non-linear filtering."""
    upper_freq = rf_params.get("nonlinear_bandpass_upper", None)
    lower_freq = rf_params["nonlinear_highpass_freq"]

    return gen_nonlinear_bandpass(upper_freq, lower_freq, nyquist_hz, block_len)


def gen_nonlinear_amplitude_lpf(corner_freq, nyquist_hz, order=1):
    return sps.butter(1, [corner_freq / nyquist_hz], btype="lowpass", output="sos")


def gen_nonlinear_bandpass(upper_freq, lower_freq, nyquist_hz, block_len):
    """Generate bandpass or highpass real-value fft filter used for non-linear filtering."""

    # Use a bandpass filter if upper frequency is specified, otherwise we use a high-pass filter.
    if upper_freq:
        nl_highpass_filter = filtfft(
            sps.butter(
                1,
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
                1,
                [lower_freq / nyquist_hz],
                btype="highpass",
            ),
            block_len,
            whole=False,
        )

    return nl_highpass_filter
