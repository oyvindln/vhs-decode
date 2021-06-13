from vhsdecode import utils
import numpy as np
import scipy.signal as sps
from matplotlib import pyplot as plt
from scipy.fftpack import fft, fftfreq
import lddecode.core as ldd

twopi = 2 * np.pi


# The following filters are for post-TBC:
# The output sample rate is 4fsc
class ChromaAFC:

    def __init__(self, demod_rate, under_ratio, sys_params, colour_under_carrier, linearize=True):
        self.demod_rate = demod_rate
        self.SysParams = sys_params
        self.fsc_mhz = self.SysParams["fsc_mhz"]
        self.fv = self.SysParams["FPS"] * 2
        self.out_sample_rate_mhz = self.fsc_mhz * 4
        self.samp_rate = self.out_sample_rate_mhz * 1e6
        self.bpf_under_ratio = under_ratio
        self.color_under = colour_under_carrier
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

        self.cc = 0
        self.chroma_heterodyne = np.array([])
        self.corrector = [1, 0]
        if linearize:
            self.fit()
        self.setCC(colour_under_carrier)

        self.chroma_log_drift = utils.StackableMA(
            min_watermark=0,
            window_average=8192
        )
        self.chroma_bias_drift = utils.StackableMA(
            min_watermark=0,
            window_average=6
        )

    def fit(self):
        # TODO: this sample_size numbers must be calculated (they correspond to the field data size)
        table = self.tableset(sample_size=355255) if self.fv < 60 else self.tableset(sample_size=239330)
        x, y = table[:, 0], table[:, 1]
        m, c = np.polyfit(x, y, 1)
        self.corrector = [m, c]
        # yn = np.polyval(self.corrector, x)
        # print(self.corrector)
        # plt.plot(x, y, 'or')
        # plt.plot(x, yn)
        # plt.show()

    # returns the measurement simulation to fit the correction equation
    def tableset(self, sample_size, points=256):
        ldd.logger.info("Linearizing chroma AFC, please wait ...")
        means = np.empty([2, 2], dtype=np.float)
        for ix, freq in enumerate(
                np.linspace(
                    self.color_under / self.bpf_under_ratio,
                    self.color_under * self.bpf_under_ratio,
                    num=points
                )
        ):
            fdc_wave = utils.gen_wave_at_frequency(freq, self.samp_rate, sample_size)
            self.setCC(freq)
            mean = self.measureCenterFreq(
                utils.filter_simple(fdc_wave, self.get_chroma_bandpass())
            )
            # print(ix, "%.02f %.02f" % (freq / 1e3, mean / 1e3))
            means = np.append(means, [[freq, mean]], axis=0)

        return means

    # some corrections to the measurement method
    def compensate(self, x):
        return x * self.corrector[0] + self.corrector[1]

    def getSampleRate(self):
        return self.out_sample_rate_mhz * 1e6

    # fcc in Hz
    def setCC(self, fcc_hz):
        self.cc = fcc_hz / 1e6
        self.genHetC()

    def getCC(self):
        return self.cc * 1e6

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

    # Returns the chroma heterodyning wave table/array computed after genHetC()
    def getChromaHet(self):
        return self.chroma_heterodyne

    def getFSCWaves(self):
        return self.fsc_wave, self.fsc_cos_wave

    def fftCenterFreq(self, data):
        time_step = 1 / self.samp_rate
        period = 5
        # The FFT of the signal
        sig_fft = fft(data)

        # And the power (sig_fft is of complex dtype)
        power = np.abs(sig_fft) ** 2

        # The corresponding frequencies
        sample_freq = fftfreq(data.size, d=time_step)

        # Plot the FFT power
        # plt.figure(figsize=(6, 5))
        # plt.plot(sample_freq, power)
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel('power')

        # Find the peak frequency: we can focus on only the positive frequencies
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        peak_freq = freqs[power[pos_mask].argmax()]
        # Check that it does indeed correspond to the frequency that we generate
        # the signal with
        np.allclose(peak_freq, 1. / period)

        # An inner plot to show the peak frequency
        # axes = plt.axes([0.55, 0.3, 0.3, 0.5])
        # plt.title('Peak frequency')
        # plt.plot(freqs[:8], power[:8])
        # plt.setp(axes, yticks=[])
        # plt.show()
        # scipy.signal.find_peaks_cwt can also be used for more advanced
        # peak detection
        return peak_freq

    def measureCenterFreq(self, data):
        return self.fftCenterFreq(data)

    # returns the downconverted chroma carrier offset
    def freqOffset(self, chroma):
        freq_cc_x = np.clip(
            self.compensate(self.measureCenterFreq(chroma)),
            a_max=self.color_under * self.bpf_under_ratio,
            a_min=self.color_under / self.bpf_under_ratio
        )
        freq_cc = self.chroma_bias_drift.work(freq_cc_x)
        self.setCC(freq_cc)
        return self.color_under, freq_cc, self.chroma_log_drift.work(freq_cc - self.color_under)


    # Filter to pick out color-under chroma component.
    # filter at about twice the carrier. (This seems to be similar to what VCRs do)
    # TODO: Needs tweaking (it seems to read a static value from the threaded demod)
    # Note: order will be doubled since we use filtfilt.
    def get_chroma_bandpass(self):
        freq_hz_half = self.demod_rate / 2
        return sps.butter(
            2,
            [50000 / freq_hz_half, self.cc * 1e6 * self.bpf_under_ratio / freq_hz_half],
            btype="bandpass",
            output="sos",
        )

    def get_burst_narrow(self):
        return sps.butter(
            2,
            [
                self.cc - 0.2 / self.out_frequency_half,
                self.cc + 0.2 / self.out_frequency_half,
            ],
            btype="bandpass",
            output="sos",
        )

    # Final band-pass filter for chroma output.
    # Mostly to filter out the higher-frequency wave that results from signal mixing.
    # Needs tweaking.
    # Note: order will be doubled since we use filtfilt.
    def get_chroma_bandpass_final(self):
        return sps.butter(
            1,
            [
                (self.fsc_mhz - 0.64) / self.out_frequency_half,
                (self.fsc_mhz + 0.54) / self.out_frequency_half,
            ],
            btype="bandpass",
            output="sos",
        )

