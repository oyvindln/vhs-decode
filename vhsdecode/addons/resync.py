from vhsdecode.addons.FMdeemph import FMDeEmphasis
from vhsdecode.utils import FiltersClass, firdes_lowpass, plot_scope, dualplot_scope


class DCrestore:

    def __init__(self, fs, tau, sysparams):
        self.samp_rate = fs
        self.tau = tau
        self.sysparams = sysparams
        self.fv = fv
        self.fh = fh

        iir_deemph = FMDeEmphasis(fs, tau)
        self.deemphFilter = FiltersClass(iir_deemph[0], iir_deemph[1], self.samp_rate)

        self.harmonic_limit = 3
        iir_hsync = firdes_lowpass(self.samp_rate, self.fh * self.harmonic_limit, 1e3, 2)
        self.hsyncFilter = FiltersClass(iir_hsync[0], iir_hsync[1], self.samp_rate)

        iir_vsync = firdes_lowpass(self.samp_rate, self.fv * self.harmonic_limit, 1e3, 2)
        self.vsyncFilter = FiltersClass(iir_vsync[0], iir_vsync[1], self.samp_rate)




    def get_rawlinelocs(self, data):
        hsync = self.hsyncFilter.lfilt(data)
        dualplot_scope(data, hsync)
        return None

    def work(self, data):
        deemph = self.deemphFilter.lfilt(data)
        locs = self.get_rawlinelocs(deemph)
