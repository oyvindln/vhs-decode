from collections import deque

from vhsdecode import utils
import numpy as np
import scipy.signal as sps
from numpy.fft import rfft, rfftfreq
import lddecode.core as ldd
from scipy.signal import argrelextrema

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
        chroma_bandpass_order=4,
        tape_format="VHS",
        do_cafc=False,
        chroma_bpf_lower=60000,
        conversion_lo_freq=None,
        carrier_mult=None,
    ):
        self.tape_format = tape_format
        self.fv = sys_params["FPS"] * 2
        self.fh = sys_params["FPS"] * sys_params["frame_lines"]
        self.color_under = color_under_carrier_f
        # Explicit colour-under conversion LO (Hz). When set (ME-SECAM), the
        # up-conversion mixes against this frequency instead of
        # fsc + color under carrier, referenced to the true TBC output rate
        # so the restored carriers land exactly on their studio frequencies.
        self.conversion_lo = conversion_lo_freq
        self.conversion_lo_trim = 0.0
        # Carrier restore multiplier. When set (SECAM method 1, recorded via a
        # divide-by-4 counter per IEC 60774-1 6.4.1), the up-conversion
        # multiplies the under-carrier phase by this factor instead of mixing
        # against a heterodyne, and the chroma block sits at
        # color_under * carrier_mult on the output.
        self.carrier_mult = carrier_mult
        self.het_sample_offset = 0
        # The rate the TBC output is actually clocked at (outlinelen samples
        # per line period), as opposed to samp_rate = 4 * fsc which outlinelen
        # only approximates after rounding to whole samples.
        self.true_samp_rate = sys_params["outlinelen"] * self.fh
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
        self._chroma_bandpass_order = chroma_bandpass_order
        self._chroma_bpf_lower = chroma_bpf_lower
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
        self.chroma_heterodyne = np.array([], dtype=np.float32)

        if do_cafc:
            self.narrowband = self._get_narrowband_bandpass()
            self.chroma_log_drift = deque(maxlen=8192)

        self.setCC(color_under_carrier_f)

    def getOutFreqHalf(self):
        return self.out_frequency_half

    # fcc in Hz (the current dwc subcarrier freq)
    def setCC(self, fcc_hz):
        self.cc_freq_mhz = fcc_hz / 1e6
        self.genHetC()

    def genHetC(self):
        self.chroma_heterodyne = self.genHetC_direct()

    # This function generates the heterodyning carrier directly without further filtering
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
        if self.conversion_lo is not None:
            # ME-SECAM: mix against the recorder's conversion LO directly.
            # The heterodyne phases are never rotated for SECAM, so a single
            # wave is generated and reused for all four phase slots.
            het_wave_scale = (
                self.conversion_lo + self.conversion_lo_trim
            ) / self.true_samp_rate
            # Continue the carrier phase from where the previous fields left
            # off so the restored FM carrier has no phase step (heard by a
            # SECAM decoder as a chroma click) at field boundaries.
            phase_offset = twopi * ((het_wave_scale * self.het_sample_offset) % 1.0)
            wave = -np.cos(
                (twopi * het_wave_scale * self.samples) + phase_offset
            ).astype(np.float32)
            return np.array([wave, wave, wave, wave])

        het_freq = self.fsc_mhz + self.cc_freq_mhz
        het_wave_scale = het_freq / self.out_sample_rate_mhz

        # This is the last cc phase measured as it comes from the tape
        phase_drift = self.cc_phase
        # One row per 90 degree phase step.
        phase_steps = np.pi / 2 * np.arange(4)[:, None]
        return (
            -np.cos((twopi * het_wave_scale * self.samples) + phase_steps + phase_drift)
        ).astype(np.float32)

    # Updates the conversion LO trim (Hz) and the absolute output sample
    # position of the start of the current field, regenerating the heterodyne
    # table when either changed. Only used when conversion_lo is set (ME-SECAM).
    def updateConversion(self, lo_trim_hz, het_sample_offset):
        if (
            lo_trim_hz != self.conversion_lo_trim
            or het_sample_offset != self.het_sample_offset
        ):
            self.conversion_lo_trim = lo_trim_hz
            self.het_sample_offset = het_sample_offset
            self.genHetC()

    # Returns the chroma heterodyning wave table/array computed after genHetC()
    def getChromaHet(self):
        return self.chroma_heterodyne

    def getFSCWaves(self):
        return self.fsc_wave, self.fsc_cos_wave

    def specsDistance(self, freq):
        return abs(freq - self.color_under)

    def fineTune(self, freq, max_step):
        tune_freq = freq
        while self.specsDistance(tune_freq) >= max_step:
            tune_freq -= max_step if tune_freq > self.color_under else -max_step

        one_step_more = tune_freq + max_step
        one_step_less = tune_freq - max_step

        if self.specsDistance(tune_freq) < self.specsDistance(one_step_less) and self.specsDistance(
            tune_freq
        ) < self.specsDistance(one_step_more):
            return_freq = tune_freq
        else:
            if self.specsDistance(one_step_more) < self.specsDistance(one_step_less):
                return_freq = one_step_more
            else:
                return_freq = one_step_less

        return return_freq

    def fftCenterFreq(self, data):
        sig_fft = rfft(data)
        power = np.abs(sig_fft) ** 2
        phase = np.angle(sig_fft)
        sample_freq = rfftfreq(data.size, d=1 / self.samp_rate)

        # Take the local power maximum closest to the nominal carrier.
        power_clip = np.clip(
            power,
            a_min=max(power) * self.power_threshold,
            a_max=max(power),
        )
        freqs_peaks = sample_freq[argrelextrema(power_clip, np.greater)]
        peak_freq = freqs_peaks[np.argmin(np.abs(freqs_peaks - self.color_under))]

        # TODO: Define this elsewhere.
        # PAL betamax needs a wider fine tune threshold
        # due to use of frequency half-shift.
        fine_tune_threshold = (
            self.fh
            if self.tape_format == "UMATIC"
            else self.fh / 2 if self.tape_format == "BETAMAX" else self.fh / 4
        )

        carrier_freq = self.fineTune(peak_freq, fine_tune_threshold)

        where_selected = np.where(sample_freq == carrier_freq)[0]
        self.cc_phase = phase[where_selected][0] if len(phase[where_selected]) > 0 else 0

        return carrier_freq

    def measureCenterFreq(self, data):
        for b, a in self.narrowband:
            data = sps.filtfilt(b, a, data)
        return self.fftCenterFreq(data)

    # returns the downconverted chroma carrier offset
    def freqOffset(self, chroma):
        min_f, max_f = self.get_band_tolerance()
        comp_f = self.measureCenterFreq(chroma)
        freq_cc = np.clip(
            comp_f,
            a_min=self.color_under * min_f,
            a_max=self.color_under * max_f,
        )
        if comp_f != freq_cc:
            ldd.logger.warn(
                "Chroma PLL range clipped at %.02f, measured %.02f" % (freq_cc, comp_f)
            )

        self.setCC(freq_cc)
        self.chroma_log_drift.append(freq_cc - self.color_under)
        return (
            self.color_under,
            freq_cc,
            np.mean(self.chroma_log_drift),
            self.cc_phase,
        )

    # Filter to pick out color-under chroma component.
    # filter at about twice the carrier. (This seems to be similar to what VCRs do)
    # TODO: Needs tweaking (it seems to read a static value from the threaded demod)
    # Note: order will be doubled since we use filtfilt.
    def get_chroma_bandpass(self):
        freq_hz_half = self.demod_rate / 2
        return sps.butter(
            self._chroma_bandpass_order,
            [
                self._chroma_bpf_lower / freq_hz_half,
                self.cc_freq_mhz * 1e6 * self.bpf_under_ratio / freq_hz_half,
            ],
            btype="bandpass",
            output="sos",
        )

    # Final band-pass filter for chroma output.
    # Mostly to filter out the higher-frequency wave that results from signal mixing.
    # Needs tweaking.
    # Note: order will be doubled since we use filtfilt.
    def get_chroma_bandpass_final(self, color_under_format=True):
        if color_under_format and (
            self.conversion_lo is not None or self.carrier_mult is not None
        ):
            # SECAM: place the band around the restored chroma block
            # rather than around fsc. A tight top edge matters: it suppresses
            # high-side FM splatter from saturated transitions, which
            # otherwise reaches downstream discriminators with wide take-off
            # filters and turns into clicks/streaks at color edges.
            # Both SECAM variants restore the block to the same place
            # (centre 4.328125 MHz): ME-SECAM by mixing against the
            # conversion LO, method 1 by multiplying the carrier by 4.
            if self.conversion_lo is not None:
                center = (self.conversion_lo - self.color_under) / 1e6
            else:
                center = self.color_under * self.carrier_mult / 1e6
            band_low = center - 0.67
            band_high = center + 0.55
        elif color_under_format:
            band_low = self.fsc_mhz - (self.color_under / 1e6) * 0.9
            band_high = self.fsc_mhz + (self.color_under / 1e6) * 0.75
        else:
            # Using a narrow filter atm as this is just used for
            # picking out burst signal in this case.
            band_low = self.fsc_mhz - 0.1
            band_high = self.fsc_mhz + 0.1

        return sps.butter(
            4,
            [
                band_low / self.out_frequency_half,
                band_high / self.out_frequency_half,
            ],
            btype="bandpass",
            output="sos",
        )

    # Post-TBC band-pass around the SECAM method 1 under carriers, used to
    # clean the signal ahead of the analytic-signal phase measurement that
    # the x4 multiplication is derived from (out-of-band noise there turns
    # directly into phase noise, which the multiplication amplifies by 4).
    # The band covers the carrier pair (1.0625/1.1015625 MHz) plus the
    # +-126.5 kHz max deviation and as much sideband room as fits below the
    # luma FM area.
    # Note: order will be doubled since we use filtfilt.
    def get_secam_under_bandpass(self):
        freq_hz_half = self.true_samp_rate / 2
        return sps.butter(
            3,
            [550e3 / freq_hz_half, 1300e3 / freq_hz_half],
            btype="bandpass",
            output="sos",
        )

    def _get_narrowband_bandpass(self):
        min_f, max_f = self.get_band_tolerance()
        trans_lo, trans_hi = self.color_under * self.transition_expand * (
            max_f - 1
        ), self.color_under * self.transition_expand * (1 - min_f)

        iir_narrow_lo = sps.butter(
            *sps.buttord(self.color_under, self.color_under + trans_lo, 3, 30, fs=self.samp_rate),
            "highpass",
            fs=self.samp_rate,
        )
        iir_narrow_hi = sps.butter(
            *sps.buttord(self.color_under, self.color_under + trans_hi, 3, 30, fs=self.samp_rate),
            "lowpass",
            fs=self.samp_rate,
        )

        return [iir_narrow_lo, iir_narrow_hi]

    def get_band_tolerance(self):
        return (100 - self.max_f_dev_percents[0]) / 100, (100 + self.max_f_dev_percents[1]) / 100
