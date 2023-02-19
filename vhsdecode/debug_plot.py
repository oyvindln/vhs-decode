from matplotlib import rc_context


def to_db_power(input_data):
    import numpy as np

    return 20 * np.log10(input_data)


class DebugPlot:
    def __init__(self, stuff_to_plot: str):
        self.__stuff_to_plot = stuff_to_plot.casefold().split()

    def is_plot_requested(self, requested_info: str):
        return requested_info in self.__stuff_to_plot


@rc_context({"figure.figsize": (14, 10), "figure.constrained_layout.use": True})
def plot_input_data(
    raw_data,
    raw_fft,
    filtered_fft,
    env,
    env_mean,
    demod_video,
    filtered_video,
    chroma,
    rfdecode,
    plot_db=False,
    plot_demod_fft=False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

    ire0 = rfdecode.sysparams_const.ire0
    ire100 = rfdecode.sysparams_const.ire0 + (100 * rfdecode.sysparams_const.hz_ire)
    sync_hz = rfdecode.sysparams_const.vsync_hz
    cc_freq = rfdecode.DecoderParams["color_under_carrier"]

    # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
    #        ax1.plot(hilbert, color='#FF0000')
    blocklen = len(raw_data)
    ax1.plot(raw_data, color="#00FF00")
    ax1.plot(env, label="Envelope", color="#0000FF")
    if rfdecode.dod_options.dod_threshold_a:
        ax1.axhline(
            rfdecode.dod_options.dod_threshold_a, label="DOD Threshold (absolute)", color="#001100"
        )
    else:
        ax1.axhline(
            rfdecode.dod_options.dod_threshold_p * env_mean, label="DOD Threshold", color="#110000"
        )
        ax1.axhline(
            rfdecode.dod_options.dod_threshold_p * env_mean * rfdecode.dod_options.dod_hysteresis,
            label="DOD Hysteresis threshold",
            ls="--",
            color="#000011",
        )
    ax1.set_title("Raw data")
    ax1.legend()
    ax2.plot(demod_video)
    ax2.set_title("Demodulated video")
    ax3.plot(filtered_video, color="#00FF00")
    ax3.axhline(rfdecode.iretohz(0), label="0 IRE", color="#000000")
    ax3.axhline(rfdecode.iretohz(100), label="100 IRE", color="#000000", ls="--")
    ax3.axhline(rfdecode.iretohz(0), label="0 IRE", color="#000000")
    ax3.set_title("Output video (Deemphasized and filtered) ")
    ax3.axvline(0, ls="dotted")
    ax3.axvline(rfdecode.linelen, ls="dotted", label="Length of one line")
    ax3.legend()
    ax4.plot(chroma)
    ax4.set_title("Chroma signal")

    half_size = (blocklen // 2) + 1
    freq_array = (np.arange(blocklen) / blocklen * rfdecode.freq_hz)[:half_size]

    def to_plot(a):
        return to_db_power(a) if plot_db else abs(a)

    if plot_demod_fft:
        fig2, (ax4, ax5) = plt.subplots(2, 1, sharex=True)
        ax5 = fig2.add_subplot(2, 1, 2)
        ax5.plot(
            freq_array, to_db_power(abs(np.fft.rfft(demod_video))), color="#00FF00"
        )
        ax5.plot(
            freq_array, to_db_power(abs(np.fft.rfft(filtered_video))), color="#FF0000"
        )
    else:
        fig2, ax4 = plt.subplots(1, 1)
    ax4.plot(
        freq_array, to_plot(raw_fft[:half_size]), color="#00FF00", label="Raw input"
    )
    ax4.plot(
        freq_array,
        to_plot(filtered_fft[:half_size]),
        color="#FF0000",
        label="After rf filtering",
    )
    ax4.set_xlabel("frequency")
    if plot_db:
        ax4.set_ylabel("dB power")
    ax4.set_title("frequency spectrum of rf input")
    ax4.axvline(ire0, label="0 IRE", color="#000000")
    ax4.axvline(ire100, label="100 IRE", ls="--")
    ax4.axvline(sync_hz, label="sync tip", ls="-.")
    ax4.axvline(cc_freq, label="Chroma carrier", color="#6F6F00")
    ax4.legend()

    plt.show()
    # exit(0)


def plot_deemphasis(rf, filter_video_lpf, decoder_params, filter_deemp):
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as sps
    from vhsdecode.addons.FMdeemph import FMDeEmphasisB

    corner_freq = 1 / (math.pi * 2 * decoder_params["deemph_tau"])

    db, da = FMDeEmphasisB(
        rf.freq_hz, decoder_params["deemph_gain"], decoder_params["deemph_mid"]
    ).get()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    w1, h1 = sps.freqz(db, da, fs=rf.freq_hz)

    # VHS eyeballed freqs.
    test_arr = np.array(
        [
            [
                0.04,
                0.05,
                0.07,
                0.1,
                corner_freq / 1e6,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
            ],
            [
                0.4,
                0.6,
                1.2,
                2.2,
                3,
                5.25,
                7.5,
                9.2,
                10.5,
                11.75,
                12.75,
                13.5,
                13.8,
                13.9,
                14,
            ],
        ]
    )
    test_arr[0] *= 1000000.0

    ax1.plot(test_arr[0], test_arr[1], color="#000000")
    ax1.plot(w1, -20 * np.log10(h1))
    ax1.axhline(-3)
    ax1.axhline(-7)
    ax1.axvline(corner_freq)

    blocklen_half = rf.blocklen // 2
    freqs = np.linspace(0, rf.freq_hz_half, blocklen_half)
    ax2.set_ylim(bottom=-100)
    ax2.plot(
        freqs, 20 * np.log10(filter_deemp[:blocklen_half]), label="Deemphasis only"
    )
    ax2.plot(freqs, 20 * np.log10(filter_video_lpf[:blocklen_half]), label="lpf")
    ax2.plot(
        freqs,
        20 * np.log10(rf.Filters["FVideo"][:blocklen_half]),
        label="Deemphasis + lpf",
    )
    ax2.legend()
    plt.show()
    exit()
