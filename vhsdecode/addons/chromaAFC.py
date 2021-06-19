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

    def __init__(self, demod_rate, under_ratio, sys_params, colour_under_carrier, linearize=False, plot=False):
        self.cc_phase = 0
        self.fft_plot = False
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
            print("freq(x) = %.02f x + %.02f" % (self.corrector[0], self.corrector[1]))
        self.setCC(colour_under_carrier)

        self.chroma_log_drift = utils.StackableMA(
            min_watermark=0,
            window_average=8192
        )
        self.chroma_bias_drift = utils.StackableMA(
            min_watermark=0,
            window_average=6
        )

        self.fft_plot = plot
        self.cc_wave = np.array([])

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

        phase_drift = self.cc_phase
        amplitude = 100
        # 0 phase downconverted color under carrier wave
        cc_wave = np.sin(
            (twopi * cc_wave_scale * self.samples) + phase_drift
        )
        self.cc_wave = cc_wave

        # +90 deg and so on phase wave for track2 phase rotation
        cc_wave_90 = np.sin(
            (twopi * cc_wave_scale * self.samples) + (np.pi / 2) + phase_drift
        )
        cc_wave_180 = np.sin(
            (twopi * cc_wave_scale * self.samples) + np.pi + phase_drift
        )
        cc_wave_270 = np.sin(
            (twopi * cc_wave_scale * self.samples) + np.pi + (np.pi / 2) + phase_drift
        )

        # Bandpass filter to select heterodyne frequency from the mixed fsc and color carrier signal
        het_filter = sps.butter(
            6,
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
                sps.sosfiltfilt(het_filter, amplitude * cc_wave * self.fsc_wave),
                sps.sosfiltfilt(het_filter, amplitude * cc_wave_90 * self.fsc_wave),
                sps.sosfiltfilt(het_filter, amplitude * cc_wave_180 * self.fsc_wave),
                sps.sosfiltfilt(het_filter, amplitude * cc_wave_270 * self.fsc_wave),
            ]
        )
        '''
        self.chroma_heterodyne = np.array(
            [
                cc_wave * self.fsc_wave,
                cc_wave_90 * self.fsc_wave,
                cc_wave_180 * self.fsc_wave,
                cc_wave_270 * self.fsc_wave,
            ]
        )
        '''


    # Returns the chroma heterodyning wave table/array computed after genHetC()
    def getChromaHet(self):
        return self.chroma_heterodyne

    def getFSCWaves(self):
        return self.fsc_wave, self.fsc_cos_wave

    def getCCPhase(self):
        return self.cc_phase

    def resetCCPhase(self):
        self.cc_phase = 0

    def resetCC(self):
        self.setCC(self.color_under)

    def fftCenterFreq(self, data):
        time_step = 1 / self.samp_rate

        # The FFT of the signal
        sig_fft = fft(data)

        # And the power (sig_fft is of complex dtype)
        power = np.abs(sig_fft) ** 2
        phase = np.angle(sig_fft)

        # The corresponding frequencies
        sample_freq = fftfreq(data.size, d=time_step)

        # Plot the FFT power
        if self.fft_plot:
            plt.figure(figsize=(6, 5))
            plt.plot(sample_freq, power)
            plt.xlim(self.color_under / self.bpf_under_ratio, self.color_under * self.bpf_under_ratio)
            plt.title('FFT chroma power')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('power')

        # Find the peak frequency: we can focus on only the positive frequencies
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        peak_freq = freqs[power[pos_mask].argmax()]
        self.cc_phase = phase[power[pos_mask].argmax()]

        # An inner plot to show the peak frequency
        if self.fft_plot:
            print("Phase %.02f degrees" % (360 * self.cc_phase / twopi))
            yvert_range = 2*power[power[pos_mask].argmax()]
            plt.vlines(peak_freq, ymin=-yvert_range/4, ymax=yvert_range, colors='r')
            min_f = peak_freq * 0.9
            max_f = peak_freq * 1.1
            plt.text(max_f, -yvert_range/8, "%.02f kHz" % (peak_freq / 1e3))
            f_index = np.where(np.logical_and(min_f < freqs, freqs < max_f))
            s_freqs = freqs[f_index]
            s_power = power[f_index]
            axes = plt.axes([0.55, 0.45, 0.3, 0.3])
            plt.title('Peak frequency')
            plt.plot(s_freqs, s_power)
            plt.setp(axes, yticks=[])
            plt.xlim(min_f, max_f)
            plt.ion()
            plt.pause(0.5)
            plt.show()
            plt.close()

        # scipy.signal.find_peaks_cwt can also be used for more advanced
        # peak detection
        return peak_freq

    def measureCenterFreq(self, data):
        return self.fftCenterFreq(data)

    # returns the downconverted chroma carrier offset
    def freqOffset(self, chroma, adjustf=False):
        freq_cc_x = np.clip(
            self.compensate(self.measureCenterFreq(chroma)),
            a_max=self.color_under * self.bpf_under_ratio,
            a_min=self.color_under / self.bpf_under_ratio
        )
        freq_cc = self.chroma_bias_drift.work(freq_cc_x) if adjustf else self.cc * 1e6
        self.setCC(freq_cc)
        # utils.dualplot_scope(chroma[1000:1128], self.cc_wave[1000:1128])
        return self.color_under, \
               freq_cc, \
               self.chroma_log_drift.work(freq_cc - self.color_under), self.cc_phase

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

