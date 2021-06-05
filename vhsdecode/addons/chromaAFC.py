from vhsdecode import utils
import numpy as np
import scipy.signal as sps
from lddecode.utils import unwrap_hilbert
from pyhht.utils import inst_freq

twopi = 2 * np.pi

# The following filters are for post-TBC:
# The output sample rate is at approx 4fsc
class ChromaAFC:

    def __init__(self, sys_params, colour_under_carrier):
        self.SysParams = sys_params
        self.cc = 0
        self.color_under = colour_under_carrier
        self.setCC(colour_under_carrier)
        self.fsc_mhz = self.SysParams["fsc_mhz"]
        self.out_sample_rate_mhz = self.fsc_mhz * 4
        self.samp_rate = self.out_sample_rate_mhz * 1e6
        self.out_frequency_half = self.out_sample_rate_mhz / 2
        self.fieldlen = self.SysParams["outlinelen"] * max(self.SysParams["field_lines"])
        self.samples = np.arange(self.fieldlen)

        # Standard frequency color carrier wave.
        self.fsc_wave = utils.gen_wave_at_frequency(
            self.fsc_mhz, self.out_sample_rate_mhz, self.fieldlen
        )
        self.fsc_cos_wave = utils.gen_wave_at_frequency(
            self.fsc_mhz, self.out_sample_rate_mhz, self.fieldlen, np.cos
        )

        self.chroma_heterodyne = np.array([])
        self.genHetC()

        iir_bandpass = utils.firdes_bandpass(
            self.samp_rate,
            colour_under_carrier * 0.99,
            4e3,
            colour_under_carrier * 1.01,
            4e3
        )

        self.bandpass = utils.FiltersClass(iir_bandpass[0], iir_bandpass[1], self.samp_rate)
        # self.tableset()
        self.mean_offset = colour_under_carrier - self.calibrate()
        print(self.mean_offset)
        self.chroma_drift = utils.StackableMA(
            min_watermark=0,
            window_average=8192
        )
        self.chroma_bias_drift = utils.StackableMA(
            min_watermark=0,
            window_average=30
        )
        self.manual_offset = -1983.62  # something to check here, maybe format dependant

    def tableset(self):
        means = list()
        for ix, freq in enumerate(np.linspace(self.color_under * 0.5, self.color_under * 1.5, num=256)):
            fdc_wave = utils.gen_wave_at_frequency(freq, self.samp_rate, 1e6)
            defm = self.deFM(self.bandpass.lfilt(fdc_wave))
            mean = np.mean(defm.real)
            print(ix, "%.02f %.02f" % (freq / 1e3, mean / 1e3))
            means.append((freq, mean))

    # some corrections to the measurement method
    def compensate(self, x):
        return x * 0.991 + (6.021 + self.manual_offset)

    def calibrate(self):
        fdc_wave = utils.gen_wave_at_frequency(self.color_under, self.samp_rate, 1e6)
        defm = self.deFM(self.bandpass.lfilt(fdc_wave))
        offset = np.mean(defm.real)
        #level = np.ones(len(defm)) * offset
        print("%.02f %.02f kHz" % (self.color_under, offset))
        #utils.dualplot_scope(defm[:1024], level[:1024], title="CC fixed sinewave", a_label="hilbert out", b_label="mean")
        #print(offset)
        return offset

    def getSampleRate(self):
        return self.out_sample_rate_mhz * 1e6

    # fcc in Hz
    def setCC(self, fcc_hz):
        self.cc = fcc_hz / 1e6

    # As this is done on the tbced signal, we need the sampling frequency of that,
    # which is 4fsc for NTSC and approx. 4 fsc for PAL.
    def genHetC(self):
        cc_wave_scale = self.cc / self.out_sample_rate_mhz
        het_freq = self.fsc_mhz + self.cc
        # 0 phase downconverted color under carrier wave
        cc_wave = np.sin(twopi * cc_wave_scale * self.samples)

        # +90 deg and so on phase wave for track2 phase rotation
        cc_wave_90 = np.sin(
            (twopi * cc_wave_scale * self.samples) + (np.pi / 2)
        )
        cc_wave_180 = np.sin(
            (twopi * cc_wave_scale * self.samples) + np.pi
        )
        cc_wave_270 = np.sin(
            (twopi * cc_wave_scale * self.samples) + np.pi + (np.pi / 2)
        )

        # Bandpass filter to select heterodyne frequency from the mixed fsc and color carrier signal
        het_filter = sps.butter(
            1,
            [
                (het_freq - 0.001) / self.out_frequency_half,
                (het_freq + 0.001) / self.out_frequency_half,
            ],
            btype="bandpass",
            output="sos",
        )

        # Heterodyne wave
        # We combine the color carrier with a wave with a frequency of the
        # subcarrier + the downconverted chroma carrier to get the original
        # color wave back.
        self.chroma_heterodyne = np.array(
            [
                sps.sosfiltfilt(het_filter, cc_wave * self.fsc_wave),
                sps.sosfiltfilt(het_filter, cc_wave_90 * self.fsc_wave),
                sps.sosfiltfilt(het_filter, cc_wave_180 * self.fsc_wave),
                sps.sosfiltfilt(het_filter, cc_wave_270 * self.fsc_wave),
            ]
        )

    def getChromaHet(self):
        return self.chroma_heterodyne

    def getFSCWaves(self):
        return self.fsc_wave, self.fsc_cos_wave

    def hhtdeFM(self, data):
        instf, t = inst_freq(data)
        return np.add(np.multiply(instf, -self.samp_rate), self.samp_rate / 2)

    def deFM(self, chroma):
        return self.hhtdeFM(chroma)
        #return unwrap_hilbert(chroma, self.samp_rate)

    def freqOffset(self, chroma):
        filtered = self.bandpass.lfilt(chroma)
        defm = self.deFM(filtered)
        freq_cc = self.compensate(np.mean(defm))
        # level = np.ones(len(defm)) * freq_cc
        # print("Chroma CC %.02f kHz" % (freq_cc / 1e3))
        #utils.dualplot_scope(defm[:1024] + self.mean_offset, level[:1024], title="CC offset", a_label="CC", b_label="mean CC")

        self.setCC(np.clip(freq_cc, a_min=self.color_under * 0.5, a_max=self.color_under * 1.5))
        self.genHetC()
        self.chroma_drift.push(self.color_under - freq_cc)
        return self.color_under, freq_cc, self.chroma_drift.pull()
