import sys


def to_db_power(input_data):
    import numpy as np

    return 20 * np.log10(input_data)


class DebugPlot:
    def __init__(self, stuff_to_plot: str):
        self.__stuff_to_plot = stuff_to_plot.casefold().split()

    def is_plot_requested(self, requested_info: str):
        return requested_info in self.__stuff_to_plot


def plot_data_and_pulses(
    demod_video,
    raw_pulses=None,
    linelocs=None,
    pulses=None,
    extra_lines=None,
    threshold=None,
):
    import matplotlib.pyplot as plt
    from matplotlib import rc_context

    with rc_context(
        {"figure.figsize": (14, 10), "figure.constrained_layout.use": True}
    ):
        # Make sure we have enough plots
        # Surely there is a cleaner way to handle this.
        plots = 1
        if raw_pulses is not None:
            plots += 1
        if linelocs is not None:
            plots += 1
        if pulses is not None:
            plots += 1
        if extra_lines is not None:
            plots += 1

        plots = max(2, plots)

        ax_number = 0

        fig, ax = plt.subplots(plots, 1, sharex=True)
        ax[ax_number].plot(demod_video)
        ax[ax_number].set_title("Video")
        if threshold is not None:
            ax[ax_number].axhline(threshold)

        if raw_pulses is not None:
            ax_number += 1
            ax[ax_number].set_title("Raw pulses")
            for raw_pulse in raw_pulses:
                ax[ax_number].axvline(raw_pulse.start, color="#910000")
                ax[ax_number].axvline(raw_pulse.start + raw_pulse.len, color="#090909")

        if pulses is not None:
            ax_number += 1
            ax[ax_number].set_title("Validated pulses")
            for pulse in pulses:
                color = "#FF0000" if pulse[0] == 2 else "#00FF00"
                ax[ax_number].axvline(pulse[1][0], color=color)
                ax[ax_number].axvline(pulse[1][0] + pulse[1][1], color="#009900")

        if linelocs is not None:
            ax_number += 1
            ax[ax_number].set_title("Calculated line locations")
            for ll in linelocs:
                ax[ax_number].axvline(ll)

        if extra_lines is not None:
            ax_number += 1
            for extra_line in extra_lines:
                ax[ax_number].axvline(extra_line)

        # to_right_edge = self.usectoinpx(self.rf.SysParams["hsyncPulseUS"]) + (
        #    2.25 * (self.rf.freq / 40.0)
        # )

        plt.show()


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
    from matplotlib import rc_context
    import numpy as np

    with rc_context(
        {"figure.figsize": (14, 10), "figure.constrained_layout.use": True}
    ):
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
                rfdecode.dod_options.dod_threshold_a,
                label="DOD Threshold (absolute)",
                color="#001100",
            )
        else:
            ax1.axhline(
                rfdecode.dod_options.dod_threshold_p * env_mean,
                label="DOD Threshold",
                color="#110000",
            )
            ax1.axhline(
                rfdecode.dod_options.dod_threshold_p
                * env_mean
                * rfdecode.dod_options.dod_hysteresis,
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
                freq_array,
                to_db_power(abs(np.fft.rfft(filtered_video))),
                color="#FF0000",
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

    # Betamax pal eyeballed freqs for -3 dB
    _test_arr = np.array(
        [
            [
                0.05,
                # corner_freq / 1e6,
                0.2,
                0.5,
                1.0,
                2.0,
                4.0,
            ],
            [
                0.2,
                # 3,
                4.6,
                8.5,
                11.7,
                13.3,
                11.6,
            ],
        ]
    )

    # Video8 freqs for -3 dB
    _test_arr = np.array(
        [
            [
                0.05,
                0.1,
                # corner_freq / 1e6,
                0.2,
                0.5,
                1.0,
                2.0,
                4.0,
            ],
            [
                0.9,
                2.6,
                # 3,
                6.4,
                11.2,
                13.1,
                13.8,
                13.9,
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
    sys.exit()
