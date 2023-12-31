from vhsdecode import utils
import numpy as np
import scipy.signal as sps
from scipy.fftpack import fft, fftfreq
import lddecode.core as ldd
from scipy.signal import argrelextrema
from vhsdecode.linear_filter import FiltersClass

twopi = 2 * np.pi


# The following filters are for post-TBC:
# The output sample rate is 4fsc
class ChromaAFC:
    def __init__(
        self,
        demod_rate,
        under_ratio,
        sys_params,
        color_under_carrier_f,
        linearize=False,
        plot=False,
        tape_format="VHS",
        do_cafc=False,
    ):
        self.tape_format = tape_format
        self.fv = sys_params["FPS"] * 2
        self.fh = sys_params["FPS"] * sys_params["frame_lines"]
        self.color_under = color_under_carrier_f
        self.cc_phase = 0
        self.power_threshold = 1 / 3
        self.transition_expand = 12
        percent = 100 * (-1 + (self.color_under + (2 * self.fh)) / self.color_under)
        self.max_f_dev_percents = percent, percent  # max percent down, max percent up
        self.demod_rate = demod_rate
        self.fsc_mhz = sys_params["fsc_mhz"]
        self.out_sample_rate_mhz = self.fsc_mhz * 4
        self.samp_rate = self.out_sample_rate_mhz * 1e6
        self.bpf_under_ratio = under_ratio
        self.out_frequency_half = self.out_sample_rate_mhz / 2
        self.fieldlen = sys_params["outlinelen"] * max(sys_params["field_lines"])
        self.samples = np.arange(self.fieldlen)

        # Standard frequency color carrier wave.
        self.fsc_wave = utils.gen_wave_at_frequency(
            self.fsc_mhz, self.out_sample_rate_mhz, self.fieldlen
        )
        self.fsc_cos_wave = utils.gen_wave_at_frequency(
            self.fsc_mhz, self.out_sample_rate_mhz, self.fieldlen, np.cos
        )

        self.cc_freq_mhz = 0
        self.chroma_heterodyne = np.array([])

        if do_cafc:
            self.narrowband = self.get_narrowband_bandpass()
            self.meas_stack = utils.StackableMA(min_watermark=0, window_average=8192)
            self.chroma_log_drift = utils.StackableMA(
                min_watermark=0, window_average=8192
            )
            self.chroma_bias_drift = utils.StackableMA(
                min_watermark=0, window_average=6
            )

            self.corrector = [1, 0]
            self.on_linearization = linearize
            if linearize:
                self.fit()
                ldd.logger.info(
                    "freq(x) = %.02f x + %.02f" % (self.corrector[0], self.corrector[1])
                )
                self.on_linearization = False

        self.setCC(color_under_carrier_f)

        self.fft_plot = plot
        self.cc_wave = np.array([])

    # applies a filtfilt to the data over the array of filters
    def chainfiltfilt(self, data, filters):
        for filt in filters:
            data = filt.filtfilt(data)
        return data

    def fit(self):
        table = self.tableset(sample_size=self.fieldlen)
        x, y = table[:, 0], table[:, 1]
        m, c = np.polyfit(x, y, 1)
        self.corrector = [m, c]
        assert 0.7 < m < 1.3, "Linearization error"
        # yn = np.polyval(self.corrector, x)
        # print(self.corrector)
        # plt.plot(x, y, 'or')
        # plt.plot(x, yn)
        # plt.show()

    # returns the measurement simulation to fit the correction equation
    def tableset(self, sample_size, points=256):
        ldd.logger.info("Linearizing chroma AFC, please wait ...")
        means = np.empty([2, 2], dtype=float)
        min_f, max_f = self.get_band_tolerance()
        for ix, freq in enumerate(
            np.linspace(self.color_under * min_f, self.color_under * max_f, num=points)
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

    def getOutFreqHalf(self):
        return self.out_frequency_half

    # fcc in Hz (the current dwc subcarrier freq)
    def setCC(self, fcc_hz):
        self.cc_freq_mhz = fcc_hz / 1e6
        self.genHetC()

    def getCC(self):
        return self.cc_freq_mhz * 1e6

    def genHetC(self):
        h1 = self.genHetC_direct()
        # h0 = self.genHetC_filtered()
        # utils.dualplot_scope(h0[3][:128], h1[3][:128])
        self.chroma_heterodyne = h1

    # This function generates the heterodyning carrier directly
    # without further filtering, see genHetC_filtered() below for the other method
    def genHetC_direct(self):
        """
        According to this trigonometric identity:

            sin(a) * sin(b) = ( cos(a - b) - cos(a + b) ) / 2

        The product of two sine waves produces the difference/2 of two signals:
            cos(a - b) and cos(a + b)

        Let: a = 2 * pi * fcc + phase_cc
        and  b = 2 * pi * fsc + phase_sc
        where fcc is the downconverted carrier frequency
        and fsc is the cvbs color carrier frequency

        The resulting signals are:
            s0(t) = cos( 2 * pi * ( fcc - fsc ) * t + phase_cc - phase_sc )
            s1(t) = cos( 2 * pi * ( fcc + fsc ) * t + phase_cc + phase_sc )

        And the resulting identity can be written as:
        sin((2*pi*fcc)t + phase_cc) * sin((2*pi*fsc)t + phase_sc) =
            (s0 - s1) / 2

        We're interested on s1 part for heterodyning (omitted t for simplification):
            -cos( 2 * pi * ( fcc + fsc ) + phase_cc + phase_sc )

        Which is a -cosine of fcc + fsc frequency, and
        phase_cc phase, assuming phase_sc = 0
        """
        het_freq = self.fsc_mhz + self.cc_freq_mhz
        het_wave_scale = het_freq / self.out_sample_rate_mhz

        # This is the last cc phase measured as it comes from the tape
        phase_drift = self.cc_phase
        return np.array(
            [
                # phase 0
                -np.cos((twopi * het_wave_scale * self.samples) + phase_drift),
                # phase 90 deg
                -np.cos(
                    (twopi * het_wave_scale * self.samples) + (np.pi / 2) + phase_drift
                ),
                # phase 180 deg
                -np.cos((twopi * het_wave_scale * self.samples) + np.pi + phase_drift),
                # phase 270 deg
                -np.cos(
                    (twopi * het_wave_scale * self.samples)
                    + (np.pi * 3 / 2)
                    + phase_drift
                ),
            ]
        )

    # As this is done on the tbced signal, we need the sampling frequency of that,
    # which is 4fsc for NTSC and approx. 4 fsc for PAL.
    def genHetC_filtered(self):
        cc_wave_scale = self.cc_freq_mhz / self.out_sample_rate_mhz
        het_freq = self.fsc_mhz + self.cc_freq_mhz

        phase_drift = self.cc_phase
        # 0 phase downconverted color under carrier wave
        cc_wave = np.sin((twopi * cc_wave_scale * self.samples) + phase_drift)
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

        return np.array(
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

    def getCCPhase(self):
        return self.cc_phase

    def resetCCPhase(self):
        self.cc_phase = 0

    def resetCC(self):
        self.setCC(self.color_under)

    def selectWithSpread(self, value, spread):
        min_f_band, max_f_band = self.get_band_tolerance()
        max_f_band *= value
        min_f_band *= value
        half_steps = round((max_f_band - value) / spread)
        carrier_space = np.linspace(
            value - (half_steps * spread), value + (half_steps * spread), 2 * half_steps
        )
        freqs_delta = np.abs(carrier_space - self.color_under)
        where_min = np.where(freqs_delta == min(freqs_delta))[0]
        return carrier_space[where_min][0]

        # hi, lo = value + spread, value - spread
        # return -spread if abs(self.color_under - hi) < abs(self.color_under - lo) else spread

    def choosePeak(self, freq_peaks):
        diff_peaks = np.diff(freq_peaks)
        diff_delta = np.abs(diff_peaks - self.fh), np.abs(diff_peaks - self.fh / 2)
        where_min = np.where(diff_delta[1] == min(diff_delta[1]))[0]
        peak_freq = freq_peaks[where_min][0]
        print(peak_freq)
        return freq_peaks

    def specsDistance(self, freq):
        return abs(freq - self.color_under)

    def fineTune(self, freq, max_step):
        tune_freq = freq
        while self.specsDistance(tune_freq) >= max_step:
            tune_freq -= max_step if tune_freq > self.color_under else -max_step

        one_step_more = tune_freq + max_step
        one_step_less = tune_freq - max_step

        if self.specsDistance(tune_freq) < self.specsDistance(
            one_step_less
        ) and self.specsDistance(tune_freq) < self.specsDistance(one_step_more):
            return_freq = tune_freq
        else:
            if self.specsDistance(one_step_more) < self.specsDistance(one_step_less):
                return_freq = one_step_more
            else:
                return_freq = one_step_less

        return return_freq

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
            from matplotlib import pyplot as plt

            plt.figure(figsize=(6, 5))
            plt.plot(sample_freq, power)
            plt.xlim(
                self.color_under / self.bpf_under_ratio,
                self.color_under * self.bpf_under_ratio,
            )
            plt.title("FFT chroma power")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("power")

        # Find the peak frequency: we can focus on only the positive frequencies
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        phases = phase[pos_mask]
        assert len(freqs) == len(phases)

        if self.on_linearization:
            carrier_freq = freqs[power[pos_mask].argmax()]
            peak_freq = carrier_freq
            self.cc_phase = phase[power[pos_mask].argmax()]
        else:
            power_clip = np.clip(
                power[pos_mask],
                a_min=max(power) * self.power_threshold,
                a_max=max(power),
            )
            where_peaks = argrelextrema(power_clip, np.greater)
            freqs_peaks = freqs[where_peaks]
            # freqs_peaks = self.choosePeak(freqs_peaks)
            freqs_delta = np.abs(freqs_peaks - self.color_under)
            where_min = np.where(freqs_delta == min(freqs_delta))[0]
            peak_freq = freqs_peaks[where_min][0]

            # TODO: Define this elsewhere.
            # PAL betamax needs a wider fine tune threshold
            # due to use of frequency half-shift.
            fine_tune_threshold = (
                self.fh
                if self.tape_format == "UMATIC"
                else self.fh / 2
                if self.tape_format == "BETAMAX"
                else self.fh / 4
            )

            carrier_freq = self.fineTune(peak_freq, fine_tune_threshold)

            where_selected = np.where(sample_freq == carrier_freq)[0]
            self.cc_phase = (
                phase[where_selected] if len(phase[where_selected]) > 0 else 0
            )

        # An inner plot to show the peak frequency
        if self.fft_plot:
            from matplotlib import pyplot as plt

            print(self.cc_phase)
            # print("Phase %.02f degrees" % (360 * self.cc_phase / twopi))
            yvert_range = 2 * power[power[pos_mask].argmax()]
            plt.vlines(
                peak_freq,
                ymin=-yvert_range * self.power_threshold,
                ymax=yvert_range,
                colors="g",
            )
            plt.vlines(
                self.color_under,
                ymin=-yvert_range * self.power_threshold,
                ymax=yvert_range,
                colors="r",
            )
            plt.vlines(
                carrier_freq,
                ymin=-yvert_range * self.power_threshold,
                ymax=yvert_range,
                colors="orange",
            )
            min_f = carrier_freq - 4 * self.fh
            max_f = carrier_freq + 4 * self.fh
            plt.xlim(min_f, max_f)
            plt.text(max_f, -yvert_range / 8, "%.02f kHz" % (carrier_freq / 1e3))
            f_index = np.where(np.logical_and(min_f < freqs, freqs < max_f))
            s_freqs = freqs[f_index]
            s_power = power[f_index]
            axes = plt.axes([0.55, 0.45, 0.3, 0.3])
            plt.title("Peak frequency")
            plt.plot(s_freqs, s_power)
            plt.setp(axes, yticks=[])
            plt.xlim(min_f, max_f)
            plt.ion()
            plt.pause(8)
            plt.show()
            plt.close()

        # scipy.signal.find_peaks_cwt can also be used for more advanced
        # peak detection
        return carrier_freq

    def measureCenterFreq(self, data):
        return self.fftCenterFreq(self.chainfiltfilt(data, self.narrowband))

    # returns the downconverted chroma carrier offset
    def freqOffset(self, chroma, adjustf=True):
        min_f, max_f = self.get_band_tolerance()
        comp_f = self.compensate(self.measureCenterFreq(chroma))
        freq_cc_x = np.clip(
            comp_f,
            a_min=self.color_under * min_f,
            a_max=self.color_under * max_f,
        )
        if comp_f != freq_cc_x:
            ldd.logger.warn(
                "Chroma PLL range clipped at %.02f, measured %.02f"
                % (freq_cc_x, comp_f)
            )

        if adjustf:
            self.meas_stack.push(freq_cc_x)
            freq_cc = (
                freq_cc_x  # if len(self.meas_stack) < 2 else self.meas_stack[-2:][0]
            )
            # print(self.meas_stack[-2:])
        else:
            freq_cc = self.cc_freq_mhz * 1e6

        self.setCC(freq_cc)
        # utils.dualplot_scope(chroma[1000:1128], self.cc_wave[1000:1128])
        return (
            self.color_under,
            freq_cc,
            self.chroma_log_drift.work(freq_cc - self.color_under),
            self.cc_phase,
        )

    # Filter to pick out color-under chroma component.
    # filter at about twice the carrier. (This seems to be similar to what VCRs do)
    # TODO: Needs tweaking (it seems to read a static value from the threaded demod)
    # Note: order will be doubled since we use filtfilt.
    def get_chroma_bandpass(self):
        freq_hz_half = self.demod_rate / 2
        return sps.butter(
            2,
            [
                50000 / freq_hz_half,
                self.cc_freq_mhz * 1e6 * self.bpf_under_ratio / freq_hz_half,
            ],
            btype="bandpass",
            output="sos",
        )

    def get_burst_narrow(self):
        return sps.butter(
            2,
            [
                self.cc_freq_mhz - 0.2 / self.out_frequency_half,
                self.cc_freq_mhz + 0.2 / self.out_frequency_half,
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

    def get_narrowband_bandpass(self):
        min_f, max_f = self.get_band_tolerance()
        trans_lo, trans_hi = self.color_under * self.transition_expand * (
            max_f - 1
        ), self.color_under * self.transition_expand * (1 - min_f)

        iir_narrow_lo = utils.firdes_highpass(
            self.samp_rate, self.color_under, trans_lo, order_limit=200
        )

        iir_narrow_hi = utils.firdes_lowpass(
            self.samp_rate, self.color_under, trans_hi, order_limit=200
        )

        return {
            FiltersClass(iir_narrow_lo[0], iir_narrow_lo[1], self.samp_rate),
            FiltersClass(iir_narrow_hi[0], iir_narrow_hi[1], self.samp_rate),
        }

    def get_band_tolerance(self):
        return (100 - self.max_f_dev_percents[0]) / 100, (
            100 + self.max_f_dev_percents[1]
        ) / 100
