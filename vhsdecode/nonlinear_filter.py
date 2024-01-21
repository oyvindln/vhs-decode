import numpy.fft as npfft

import scipy.signal as sps
import numpy as np
import lddecode.utils as lddu
import math
from vhsdecode.utils import filter_simple


def _sub_deemphasis_debug(
    out_video,
    out_video_fft,
    filters,
    deviation,
    const_amplitude,
    sub_emphasis_params,
):
    """Wrapper for testing function with linear deemp added"""
    result = sub_deemphasis(
        out_video,
        out_video_fft,
        filters,
        deviation * (1 / const_amplitude),
        sub_emphasis_params.exponential_scaling,
        sub_emphasis_params.scaling_1,
        sub_emphasis_params.scaling_2,
        sub_emphasis_params.static_factor,
        # debug_const_amplitude=const_amplitude,
    )
    # return result
    return npfft.irfft(npfft.rfft(result))


def sub_deemphasis_inner(
    out_video,
    out_video_fft,
    filters,
    deviation,
    exponential_scale=0.33,
    linear_scale_1=None,
    linear_scale_2=None,
    static_factor=None,
    debug_const_amplitude=False,
):
    """Apply non-linear de-emphasis filter to input signal

    Args:
        out_video (_type_): Video signal to apply deemphasis to
        out_video_fft (_type_): real-value fft of video signal
        filters (_type_): _description_
        deviation (_type_): Deviation the different input amplitudes used in the reference filter response curves are calculated against.
        exponential_scale (float, optional): exponential scaling factor for amplitude. Defaults to 0.33.
        linear_scale_1 (_type_, optional): linear scaling factor applied before exponential scaling. Defaults to None.
        linear_scale_2 (_type_, optional): linear scaling factor applied before exponential scaling. Defaults to None.
        debug_const_amplitude (bool, optional): Set constant amplitude instead of using video signal for testing response. Defaults to False.

    Returns:
        _type_: video signal with filtering applied
    """
    # this filter is applied with all other linear filters. To make plot of filter-tune work it needs to be applied here
    #if filters["subdeemph_linear"] is not None :
    #    out_video_fft = out_video_fft * filters["subdeemph_linear"]
    #    out_video = npfft.irfft(out_video_fft).real

    hf_part = npfft.irfft(out_video_fft * filters["NLHighPassF"])
    # hf_part = sps.filtfilt(filters["NLHighPassFB"][0], filters["NLHighPassFB"][1], out_video)

    deviation /= 2

    static_part = 0
    if static_factor:
        static_part = hf_part * static_factor

    # Get the instantaneous amplitude of the signal using the hilbert transform
    # and divide by the formats specified deviation so we get a amplitude compared to the specifications references.
    amplitude = abs(sps.hilbert(hf_part)) / deviation
    amplitude = filter_simple(amplitude, filters["NLAmplitudeLPF"])

    if debug_const_amplitude:
        amplitude = debug_const_amplitude

    if linear_scale_1 is not None:
        amplitude *= linear_scale_1
    # Scale the amplitude by a exponential factore (typically less than 1 so it ends up being a root function of osrts)
    amplitude = np.power(amplitude, exponential_scale)
    if linear_scale_2 is not None:
        amplitude *= linear_scale_2

    # Scale the band-pass filtered signal by one minus the resulting referenc
    # e.g this means it get scaled more at lower amplitudes.
    hf_part *= 1 - amplitude

    # And subtract it from the output signal.
    return (out_video - hf_part - static_part, hf_part + static_part, amplitude)


def sub_deemphasis(
    out_video,
    out_video_fft,
    filters,
    deviation,
    exponential_scale=0.33,
    linear_scale_1=None,
    linear_scale_2=None,
    static_factor=None,
    debug_const_amplitude=False,
):
    """Apply non-linear de-emphasis filter to input signal

    Args:
        out_video (_type_): Video signal to apply deemphasis to
        out_video_fft (_type_): real-value fft of video signal
        filters (_type_): _description_
        deviation (_type_): Deviation the different input amplitudes used in the reference filter response curves are calculated against.
        exponential_scale (float, optional): exponential scaling factor for amplitude. Defaults to 0.33.
        linear_scale_1 (_type_, optional): linear scaling factor applied before exponential scaling. Defaults to None.
        linear_scale_2 (_type_, optional): linear scaling factor applied before exponential scaling. Defaults to None.
        debug_const_amplitude (bool, optional): Set constant amplitude instead of using video signal for testing response. Defaults to False.

    Returns:
        _type_: video signal with filtering applied
    """
    (ret, _, _) = sub_deemphasis_inner(
        out_video,
        out_video_fft,
        filters,
        deviation,
        exponential_scale,
        linear_scale_1,
        linear_scale_2,
        static_factor,
        debug_const_amplitude,
    )

    # And subtract it from the output signal.
    return ret


def _gen_low_shelf_2(w0, gain, fs, is_db=False):
    import scipy.signal as sps

    b0 = 10 ** (gain / 20) if is_db else gain
    w0 = w0 * math.tau
    filts = sps.lti([1, (w0 * (b0 + 1))], [1, b0])
    filtz = sps.lti(*sps.bilinear(filts.num, filts.den, fs))
    return filtz.num, filtz.den


def _gen_high_shelf_2(w0, gain, fs, is_db=False):
    import scipy.signal as sps

    b0 = 10 ** (gain / 20) if is_db else gain
    w0 = w0 * math.tau
    filts = sps.lti([1, (w0 * (b0 + 1))], [1, b0])
    filtz = sps.lti(*sps.bilinear(filts.num, filts.den, fs))
    return filtz.num, filtz.den


class NLFilter:
    def __init__(
        self,
        filter_a,
        filter_b,
        deviation,
    ):
        # self._fs = fs
        self._filter_a = filter_a
        self._filter_b = filter_b
        self._deviation = deviation

    def filter_in_fft(self, data_fft):
        # data_raw = npfft.irfft(data_fft).real
        # data_fb = npfft.irfft(data_fft * self.fb_bpf).real
        filt_a = npfft.irfft(data_fft * self._filter_a).real
        filt_b = npfft.irfft(data_fft * self._filter_b).real
        amplitude = abs(sps.hilbert(filt_b)) / self._deviation
        data = (filt_a * amplitude) + (filt_b * (1 - amplitude))
        return data


def to_db(input):
    return 20 * np.log10(input)


def from_db(input):
    return pow(10, (input / 20))


def limiter_filter(
    out_video,
    out_video_fft,
    filters,
    deviation,
    clip_fraction=0.021,
    static_fraction=0.3,  # 0.3
    smooth=False,
):
    # Extract the high frequency part of the signal
    hf_part = npfft.irfft(out_video_fft * filters["NLHighPassF"])
    static_part = hf_part * static_fraction
    # Limit it to preserve sharp transitions
    clipped = np.clip(
        hf_part,
        -deviation * clip_fraction,
        deviation * clip_fraction,
        # out=hf_part,
    )
    if smooth:
        remainder = hf_part - clipped
        remainder *= 0.1
        clipped += remainder
    # print("clipping: ", deviation * clip_fraction)

    #        self.DecoderParams["nonlinear_highpass_limit_l"],
    #    self.DecoderParams["nonlinear_highpass_limit_h"],#

    # And subtract it from the output signal.
    return out_video - clipped - static_part


def test_filter(filters, sample_rate, blocklen, deviation, sub_emphasis_params):
    import matplotlib.pyplot as plt
    from matplotlib import rc_context

    from vhsdecode.filter_plot import SubEmphPlotter

    deviation = 2

    with rc_context(
        {"figure.figsize": (14, 10), "figure.constrained_layout.use": True}
    ):
        fig, ax = plt.subplots(3, 1, sharex=False)
        hpf = filters["NLHighPassF"].real
        freqs = np.fft.rfftfreq(blocklen, 1.0 / sample_rate)
        ax[0].plot(freqs, to_db(hpf))
        ax[0].axvline(200000)
        ax[0].axvline(500000)

        betamax_full_deemp_db_v_freqs = np.array(
            [50000, 200000, 500000, 1000000, 2000000, 4000000]
        )

        betamax_full_deemp_db_v_ampls = [
            -17,
            -14,
            -11,
            -7,
            -3,
        ]

        betamax_full_deemp_db_v_625_variation = [0.4, 0.5, 1.2, 1.2, 2.0, 2.0]

        betamax_full_deemp_db_v_625 = np.array(
            [
                [0.2, 4.6, 11.4, 16.3, 17.6, 15.0],
                [0.2, 4.6, 10.7, 15.6, 16.8, 14.6],
                [0.2, 4.6, 10.3, 14.2, 15.8, 13.8],
                [0.2, 4.6, 9.6, 13.0, 14.3, 12.6],
                [0.2, 4.6, 8.5, 11.7, 13.3, 11.6],
            ]
        )

        betamax_full_deemp_db_v_525_variation = [[0.4, 0.8, 1.4, 2.2, 2.5, 2.5]]

        betamax_full_deemp_db_v_525 = [
            [1.9, 7.4, 13.4, 19.1, 22.3, 17.9],
            [1.9, 7.4, 13.4, 18.3, 20.9, 17.5],
            [1.9, 7.4, 13.3, 17.0, 19.0, 16.3],
            [1.9, 7.3, 11.7, 14.1, 15.3, 13.4],
            [1.9, 7.0, 10.3, 12.2, 12.1, 10.1],
        ]

        # freq, amp, dev
        betamax_1_full_deemp_db_v_525 = [
            [50000, 100000, 200000, 500000, 1000000, 2000000, 4000000],
            [0.3, 1.1, 3.2, 8.3, 11.9, 22.3, 17.9],
            [0.06, 0.2, 0.5, 0.7, 0.8, 0.8, 0.8],
        ]

        video8_sub_emp_freqs = [50000, 100000, 200000, 500000, 1000000, 2000000]
        video8_sub_emp_db_ampls = [-3, -10, -20]
        video8_sub_emp_db_v = np.array(
            [
                [0.03, 0.75, 1.9, 3.5, 4.3, 4.8],
                [0.35, 1.0, 2.4, 5.0, 6.5, 7.4],
                [0.35, 1.1, 3.0, 7.3, 9.8, 11.0],
            ]
        )

        video8_full_emp_db_v_freqs = [
            50000,
            100000,
            200000,
            500000,
            1000000,
            2000000,
            4000000,
        ]
        video8_full_emp_db_ampls = [-3, -6, -10, -15, -20]
        video8_full_emp_db_v = np.array(
            [
                [0.9, 2.6, 6.4, 11.2, 13.1, 13.8, 13.9],
                [0.9, 2.8, 6.9, 12.3, 14.1, 15.1, 15.3],
                [0.9, 2.9, 7.4, 13.8, 16.4, 17.2, 17.2],
                [0.9, 3.0, 7.8, 15.4, 18.6, 19.7, 20.0],
                [0.9, 3.1, 8.0, 16.6, 20.8, 22.5, 23.0],
            ]
        )

        plotter = SubEmphPlotter(blocklen, sample_rate, filters, ax[1])

        def _from_freq(freq):
            return (blocklen / (sample_rate)) * freq

        # w = plotter.chirp_signal
        # w_fft = plotter.chirp_fft

        plotter.plot_sub_emphasis(-20, ax[2], sub_emphasis_params)

        plotter.plot_sub_emphasis(-15, ax[2], sub_emphasis_params)
        plotter.plot_sub_emphasis(-10, ax[2], sub_emphasis_params)
        plotter.plot_sub_emphasis(-6, ax[2], sub_emphasis_params)
        plotter.plot_sub_emphasis(-3, ax[2], sub_emphasis_params, color="#000000")

        # w_a = w * from_db(-3)
        # w_a_fft = npfft.rfft(w_a)

        # w4 = limiter_filter(
        #    w_a, w_a_fft, filters, deviation,
        # )
        # ax[2].plot(freqs, to_db(npfft.rfft(w4) / w_a_fft), linestyle="dashed")

        # w_b = w * from_db(-20)
        # w_b_fft = npfft.rfft(w_b)

        # w5 = limiter_filter(
        #    w_b, w_b_fft, filters, deviation
        # )
        # ax[2].plot(freqs, to_db(npfft.rfft(w5) / w_b_fft), linestyle="dashed")

        ##positions = _from_freq(betamax_full_deemp_db_v_freqs).astype(int)

        # for i in range(0, 5):
        #    ax[2].plot(video8_full_emp_db_v_freqs, -(video8_full_emp_db_v[i]))

        for i in range(0, 3):
            ax[2].plot(video8_sub_emp_freqs, -(video8_sub_emp_db_v[i]))

        # ax[2].plot(betamax_full_deemp_db_v_freqs, -betamax_full_deemp_db_v_625[3]  + to_db(filters["FDeemp"][positions]))
        # ax[2].plot(betamax_full_deemp_db_v_freqs, -betamax_full_deemp_db_v_625[2])
        # ax[2].plot(betamax_full_deemp_db_v_freqs, -betamax_full_deemp_db_v_625[1])
        # ax[2].plot(betamax_full_deemp_db_v_freqs, -betamax_full_deemp_db_v_625[0])
        ax[2].plot(freqs, to_db(filters["FDeemp"]), color="#FF0000")
        # ax[2].plot(betamax_full_deemp_db_v_freqs, betamax_full_deemp_db_v_625)
        ax[2].axhline(to_db(0.5))
        ax[2].axhline(-4.6)
        ax[2].axvline(200000)

        print(freqs)

        plt.show()
        exit(0)
