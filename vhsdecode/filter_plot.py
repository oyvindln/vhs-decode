import numpy as np
import numpy.fft as npfft
import scipy.signal as sps

from vhsdecode.nonlinear_filter import from_db, to_db, _sub_deemphasis_debug


def _uk(filters, index, use_value_key):
    if use_value_key:
        return filters[index]["value"]
    else:
        return filters[index]


def plot_filters(
    filters,
    block_len,
    sample_rate,
    figure,
    sub_emph_plotter=None,
    sub_emphasis_params=None,
):
    import matplotlib.pyplot as plt

    ax = figure.subplots(3, 1, sharex=False)

    # hpf = filters["NLHighPassF"].real
    freqs = np.fft.rfftfreq(block_len, 1.0 / sample_rate)
    ax[2].plot(freqs, to_db(filters["FDeemp"]), color="#FF0000")
    ax[2].plot(freqs, to_db(filters["FVideo"]))

    if sub_emph_plotter and sub_emphasis_params:
        sub_emph_plotter.plot_sub_emphasis(0, ax[1], sub_emphasis_params)
        sub_emph_plotter.plot_sub_emphasis(-3, ax[1], sub_emphasis_params)
        sub_emph_plotter.plot_sub_emphasis(-6, ax[1], sub_emphasis_params)
        sub_emph_plotter.plot_sub_emphasis(-10, ax[1], sub_emphasis_params)
        sub_emph_plotter.plot_sub_emphasis(-15, ax[1], sub_emphasis_params)
        sub_emph_plotter.plot_sub_emphasis(-20, ax[1], sub_emphasis_params)


class SubEmphPlotter:
    def __init__(self, block_len, sample_rate, filters, debug_ax=None):
        self.freqs = np.fft.rfftfreq(block_len, 1.0 / sample_rate)
        self.deviation = 2
        self._filters = filters
        ratio = block_len / sample_rate

        t = np.linspace(0, ratio, block_len)
        self.chirp_signal = sps.chirp(
            t, f0=1, f1=sample_rate / 2.0, t1=ratio, method="linear"
        )
        if debug_ax is not None:
            debug_ax.plot(self.chirp_signal)
        self.chirp_fft = npfft.rfft(self.chirp_signal)

        self.hf_part = npfft.irfft(self.chirp_fft * filters["NLHighPassF"])

        if debug_ax is not None:
            amplitude = abs(sps.hilbert(self.hf_part)) / (self.deviation / 2)
            debug_ax.plot(amplitude)

    def plot_sub_emphasis(self, amplitude_db, ax, sub_emphasis_params, **kwargs):
        filtered = _sub_deemphasis_debug(
            self.chirp_signal,
            self.chirp_fft,
            self._filters,
            self.deviation,
            from_db(amplitude_db),
            sub_emphasis_params,
        )
        ax.plot(self.freqs, to_db(npfft.rfft(filtered) / self.chirp_fft), **kwargs)
