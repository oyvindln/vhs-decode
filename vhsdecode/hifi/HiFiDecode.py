# This currently decodes raw vhs HiFi RF, but it could do beta, CED, LD and others stereo FM variants
# Also, it implements an interpretation of the noise reduction like described on IEC60774-2/1999

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from math import log, pi

import numpy as np
from pyhht.utils import inst_freq
from scipy.signal import iirpeak, iirnotch
from scipy.signal.signaltools import hilbert

from lddecode.utils import unwrap_hilbert
from vhsdecode.addons.FMdeemph import FMDeEmphasisC
from vhsdecode.addons.chromasep import samplerate_resample
from vhsdecode.addons.gnuradioZMQ import ZMQSend
from vhsdecode.utils import firdes_lowpass, firdes_highpass, FiltersClass, gen_wave_at_frequency, StackableMA

DEFAULT_NR_GAIN_ = 66


@dataclass
class AFEParamsFront:
    def __init__(self):
        self.cutoff = 3e6
        self.FDC = 1e6


class AFEBandPass:
    def __init__(self, filters_params, sample_rate):
        self.samp_rate = sample_rate
        self.filter_params = filters_params

        iir_lo = firdes_lowpass(
            self.samp_rate,
            self.filter_params.cutoff,
            700e3
        )
        iir_hi = firdes_highpass(
            self.samp_rate,
            self.filter_params.FDC,
            700e3
        )

        #filter_plot(iir_lo[0], iir_lo[1], self.samp_rate, type="lopass", title="Front lopass")
        self.filter_lo = FiltersClass(iir_lo[0], iir_lo[1], self.samp_rate)
        self.filter_hi = FiltersClass(iir_hi[0], iir_hi[1], self.samp_rate)

    def work(self, data):
        try:
            return self.filter_lo.lfilt(self.filter_hi.filtfilt(data))
        except ValueError as e:
            print('ERROR: Cannot decode because a read size mismatch. Maybe EOF reached')
            sys.exit(1)


class LpFilter:
    def __init__(self, sample_rate, cut=20e3, transition=10e3):
        self.samp_rate = sample_rate
        self.cut = cut

        iir_lo = firdes_lowpass(
            self.samp_rate,
            self.cut,
            transition
        )
        self.filter = FiltersClass(iir_lo[0], iir_lo[1], self.samp_rate)

    def work(self, data):
        return self.filter.lfilt(data)


@dataclass
class AFEParamsVHS:
    def __init__(self):
        self.maxVCODeviation = 150e3

    @property
    def VCODeviation(self):
        return self.maxVCODeviation

@dataclass
class AFEParamsHi8:
    def __init__(self):
        self.maxVCODeviation = 100e3

    @property
    def VCODeviation(self):
        return self.maxVCODeviation

@dataclass
class AFEParamsPALVHS(AFEParamsVHS):
    def __init__(self):
        super().__init__()
        self.LCarrierRef = 1.4e6
        self.RCarrierRef = 1.8e6
        self.Hfreq = 15.625e3


@dataclass
class AFEParamsNTSCVHS(AFEParamsVHS):
    def __init__(self):
        super().__init__()
        self.LCarrierRef = 1.3e6
        self.RCarrierRef = 1.7e6
        self.Hfreq = 15.750e3


@dataclass
class AFEParamsNTSCHi8(AFEParamsHi8):
    def __init__(self):
        super().__init__()
        self.LCarrierRef = 1.5e6
        self.RCarrierRef = 1.7e6
        self.Hfreq = 15.750e3


class AFEFilterable:
    def __init__(self, filters_params, sample_rate, channel=0):
        self.samp_rate = sample_rate
        self.filter_params = filters_params
        d = abs(self.filter_params.LCarrierRef - self.filter_params.RCarrierRef)
        QL = self.filter_params.LCarrierRef / (4 * self.filter_params.maxVCODeviation)
        QR = self.filter_params.RCarrierRef / (4 * self.filter_params.maxVCODeviation)
        if channel == 0:
            iir_front_peak = iirpeak(
                self.filter_params.LCarrierRef,
                QL,
                fs=self.samp_rate
            )
            iir_notch_other = iirnotch(
                self.filter_params.RCarrierRef,
                QR,
                fs=self.samp_rate
            )
            iir_notch_image = iirnotch(
                self.filter_params.LCarrierRef - d,
                QR,
                fs=self.samp_rate
            )
        else:
            iir_front_peak = iirpeak(
                self.filter_params.RCarrierRef,
                QR,
                fs=self.samp_rate
            )
            iir_notch_other = iirnotch(
                self.filter_params.LCarrierRef,
                QL,
                fs=self.samp_rate
            )
            iir_notch_image = iirnotch(
                self.filter_params.RCarrierRef - d,
                QL,
                fs=self.samp_rate
            )

        self.filter_reject_other = FiltersClass(iir_notch_other[0], iir_notch_other[1], self.samp_rate)
        self.filter_band = FiltersClass(iir_front_peak[0], iir_front_peak[1], self.samp_rate)
        self.filter_reject_image = FiltersClass(iir_notch_image[0], iir_notch_image[1], self.samp_rate)

    def work(self, data):
        return self.filter_band.lfilt(
            self.filter_reject_other.lfilt(
                self.filter_reject_image.lfilt(data)
            )
        )


class FMdemod:
    def __init__(self, sample_rate, carrier_freerun, type=0):
        self.samp_rate = sample_rate
        self.type = type
        self.wave = gen_wave_at_frequency(carrier_freerun, sample_rate, num_samples=sample_rate)
        self.carrier = carrier_freerun
        self.offset = 0
        # self.offset = np.mean(self.work(self.wave))

    def hhtdeFM(self, data):
        instf, t = inst_freq(data)
        return np.add(np.multiply(instf, -self.samp_rate), self.samp_rate / 2)

    @staticmethod
    def htdeFM(data, samp_rate):
        return unwrap_hilbert(hilbert(data), samp_rate)

    @staticmethod
    def inst_freq(signal, sample_rate):
        analytic_signal = hilbert(signal.real)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                   (2.0 * np.pi) * sample_rate)
        return instantaneous_frequency

    def work(self, data):

        if self.type == 2:
            return np.add(
                self.htdeFM(data, self.samp_rate),
                -self.offset
            )
        elif self.type == 1:
            return np.add(
                self.hhtdeFM(data),
                -self.offset
            )
        else:
            return np.add(
                FMdemod.inst_freq(data, self.samp_rate),
                -self.offset
            )


def getDeemph(tau, sample_rate):
    deemph = FMDeEmphasisC(sample_rate, tau)
    iir_b, iir_a = deemph.get()
    return FiltersClass(iir_b, iir_a, sample_rate)


class LogCompander:

    @staticmethod
    def log3_2(x : float) -> float:
        return log(x) / log(1.5)

    @staticmethod
    def compress(x: float) -> float:
        x = max(min(x, 1), -1)
        y0 = LogCompander.log3_2((abs(x) + 2) / 2)
        return y0 if x >= 0 else -y0

    @staticmethod
    def expand(x: float) -> float:
        x = max(min(x, 1), -1)
        x0 = abs(x)
        y0 = (- pow(2, (1 - x0)) * (pow(2, x0) - pow(3, x0)))
        return y0 if x >= 0 else -y0


def tau_as_freq(tau):
    return 1 / (2 * pi * tau)


class NoiseReduction:

    def __init__(self, notch_freq, side_gain, discard_size=0, audio_rate=192000):
        self.audio_rate = audio_rate
        self.discard_size = discard_size
        self.hfreq = notch_freq

        # noise reduction envelope tracking constants (this ones might need tweaking)
        self.NR_envelope_gain = side_gain

        # values in seconds
        NRenv_attack = 4e-3
        NRenv_release = 80e-3

        self.NR_weighting_attack_Lo_cut = tau_as_freq(NRenv_attack)
        self.NR_weighting_attack_Lo_transition = 1e3
        self.NR_weighting_release_Lo_cut = tau_as_freq(NRenv_release)
        self.NR_weighting_release_Lo_transition = 1e3

        # this is set to avoid high frequency noise to interfere with the NR envelope tracking
        self.NR_finalLo_cut = 19e3
        self.NR_finalLo_transition = 10e3

        self.NR_weighting_T1 = 240e-6
        self.NR_weighting_T2 = 24e-6

        # main deemphasis time constant
        self.tau = 56e-6

        # final audio bandwidth limiter
        self.finalLo_cut = 20e3
        self.finalLo_transition = 10e3

        env_hi_trans = tau_as_freq(self.NR_weighting_T2) - tau_as_freq(self.NR_weighting_T1)
        env_iirb, env_iira = firdes_highpass(self.audio_rate, tau_as_freq(self.NR_weighting_T2), env_hi_trans)
        self.envelopeHighpassL = FiltersClass(env_iirb, env_iira, self.audio_rate)
        self.envelopeHighpassR = FiltersClass(env_iirb, env_iira, self.audio_rate)

        envv_iirb, envv_iira = firdes_lowpass(self.audio_rate, self.NR_finalLo_cut, self.NR_finalLo_transition)
        self.envelopeVoicepassL = FiltersClass(envv_iirb, envv_iira, self.audio_rate)
        self.envelopeVoicepassR = FiltersClass(envv_iirb, envv_iira, self.audio_rate)

        loenv_iirb, loenv_iira = firdes_lowpass(self.audio_rate, self.NR_weighting_attack_Lo_cut, self.NR_weighting_attack_Lo_transition)
        self.envelope_attack_LowpassL = FiltersClass(loenv_iirb, loenv_iira, self.audio_rate)
        self.envelope_attack_LowpassR = FiltersClass(loenv_iirb, loenv_iira, self.audio_rate)

        loenvr_iirb, loenvr_iira = firdes_lowpass(self.audio_rate, self.NR_weighting_release_Lo_cut, self.NR_weighting_release_Lo_transition)
        self.envelope_release_LowpassL = FiltersClass(loenvr_iirb, loenvr_iira, self.audio_rate)
        self.envelope_release_LowpassR = FiltersClass(loenvr_iirb, loenvr_iira, self.audio_rate)

        self.audio_bassL = LpFilter(self.audio_rate, cut=220, transition=2000)
        self.audio_bassR = LpFilter(self.audio_rate, cut=220, transition=2000)
        self.audio_presenceL = LpFilter(self.audio_rate, cut=2000, transition=8000)
        self.audio_presenceR = LpFilter(self.audio_rate, cut=2000, transition=8000)
        self.finalLoL = LpFilter(self.audio_rate, cut=self.finalLo_cut, transition=self.finalLo_transition)
        self.finalLoR = LpFilter(self.audio_rate, cut=self.finalLo_cut, transition=self.finalLo_transition)


    @staticmethod
    def cancelDC(c, dc):
        return c - dc

    @staticmethod
    def cancelDCpair(L, R, dcL, dcR):
        return NoiseReduction.cancelDC(L, dcL), NoiseReduction.cancelDC(R, dcR)

    @staticmethod
    def audio_notch(samp_rate: int, freq: float, audio):
        cancel_shift = int(round(samp_rate / (2 * freq)))
        shift = audio[:-cancel_shift]
        return np.add(audio, np.pad(shift, (cancel_shift, 0), 'wrap')) / 2

    @staticmethod
    def audio_notch_stereo(samp_rate: int, freq: float, audioL, audioR):
        return NoiseReduction.audio_notch(samp_rate, freq, audioL), NoiseReduction.audio_notch(samp_rate, freq, audioR)

    def rs_envelope(self, raw_data, channel=0):
        data = np.array([LogCompander.expand(x) for x in raw_data], dtype='float64')
        hi_part = self.envelopeHighpassL.lfilt(data) if channel == 0 else self.envelopeHighpassR.lfilt(data)
        lo_part = data - hi_part
        env_part = self.envelopeVoicepassL.lfilt(hi_part + lo_part / 2) if channel == 0 else self.envelopeVoicepassR.lfilt(hi_part + lo_part / 2)
        return np.abs(env_part)

    def noise_reduction(self, audio, comb, channel=0):
        # takes the RMS envelope of each audio channel
        audio_env = self.rs_envelope(comb, channel)

        rsaC = self.envelope_attack_LowpassL.lfilt(audio_env) if channel == 0 else self.envelope_attack_LowpassR.lfilt(audio_env)
        rsrC = self.envelope_release_LowpassL.lfilt(audio_env) if channel == 0 else self.envelope_release_LowpassR.lfilt(audio_env)

        releasing_idx = np.where(rsaC < rsrC)
        rsC = rsaC
        for id in releasing_idx:
            rsC[id] = rsrC[id]

        # computes a sidechain signal to apply noise reduction
        gate = np.clip(rsC * self.NR_envelope_gain, a_min=0.0, a_max=1.0)
        rev_gate = 1.0 - gate

        # applies noise reduction (notch at hfreq)
        gated = np.add(np.multiply(gate, audio), np.multiply(rev_gate, comb))

        # applies second part of noise reduction
        nr = np.multiply(gated, gate)

        bass_enhance = self.audio_bassL.work(nr) if channel == 0 else self.audio_bassR.work(nr)
        mid_bass = self.audio_presenceL.work(nr) if channel == 0 else self.audio_presenceR.work(nr)
        mid_enhance = mid_bass - bass_enhance

        return (nr + mid_enhance + bass_enhance / 2) * 2 / 3

    def noise_reduction_stereo(self, audioL, audioR):
        # applies notch filter at Hfreq
        combL, combR = NoiseReduction.audio_notch_stereo(self.audio_rate, self.hfreq, audioL, audioR)
        return self.noise_reduction(audioL, combL, channel=0), self.noise_reduction(audioR, combR, channel=1)

    def stereo(self, audioL, audioR):
        expandL, expandR = self.lopassCompand(audioL, channel=0), self.lopassCompand(audioR, channel=1)
        nrL, nrR = self.noise_reduction_stereo(expandL, expandR)
        finalL, finalR = nrL[self.discard_size:], nrR[self.discard_size:]
        return list(map(list, zip(finalL, finalR)))

    def lopassCompand(self, audio, channel=0):
        audioX = self.finalLoL.work(audio) if channel == 0 else self.finalLoR.work(audio)
        return audioX


class HiFiDecode:
    def __init__(self, options=None):
        if options is None:
            options = dict()
        self.options = options
        self.sample_rate = options['input_rate']
        self.options = options
        self.if_rate = 8388608
        self.audio_rate = self.options['audio_rate']

        # main deemphasis time constant
        self.tau = 56e-6

        self.ifresample_numerator, \
        self.ifresample_denominator, \
        self.audioRes_numerator, \
        self.audioRes_denominator = self.getResamplingRatios()

        # block overlap and edge discard
        self.blocks_second = 8
        self.block_size = int(self.sample_rate / self.blocks_second)
        self.block_audio_size = int(self.audio_rate / self.blocks_second)
        self.block_overlap_audio = 192 if not self.options['preview'] else 0
        self.block_overlap = round(
            self.block_overlap_audio *
            self.audioRes_denominator * self.ifresample_denominator /
            (self.audioRes_numerator * self.ifresample_numerator)
        )

        # start of filter design stuff
        self.deemphL = getDeemph(self.tau, self.if_rate)
        self.deemphR = getDeemph(self.tau, self.if_rate)
        self.lopassRF = AFEBandPass(AFEParamsFront(), self.sample_rate)
        self.dcCancelL = StackableMA(min_watermark=0, window_average=self.blocks_second)
        self.dcCancelR = StackableMA(min_watermark=0, window_average=self.blocks_second)

        a_iirb, a_iira = firdes_lowpass(self.if_rate, self.audio_rate * 3 / 4, self.audio_rate / 3, order_limit=10)
        self.preAudioResampleL = FiltersClass(a_iirb, a_iira, self.if_rate)
        self.preAudioResampleR = FiltersClass(a_iirb, a_iira, self.if_rate)

        #filter_plot(envv_iirb, envv_iira, self.audio_rate, type="bandpass", title="audio_filter")

        if options['format'] == 'vhs':
            if options['standard'] == 'p':
                self.afe_params = AFEParamsPALVHS()
                self.standard = AFEParamsPALVHS()
            else:
                self.afe_params = AFEParamsNTSCVHS()
                self.standard = AFEParamsNTSCVHS()
        else:
            self.afe_params = AFEParamsNTSCHi8()
            self.standard = AFEParamsNTSCHi8()

        self.afeL, self.afeR, self.fmL, self.fmR = self.afeParams(self.afe_params)
        self.devL, self.devR = 0, 0
        self.afeL, self.afeR, self.fmL, self.fmR = self.updateDemod()

        self.block_queue = list()
        self.stereo_executor = ThreadPoolExecutor(2)
        self.stereo_queue = list()

        if self.options['grc']:
            print(f'Set gnuradio sample rate at {self.if_rate} Hz, type float')
            self.grc = ZMQSend()

    def getResamplingRatios(self):
        samplerate2ifrate = self.if_rate / self.sample_rate
        self.ifresample_numerator = Fraction(samplerate2ifrate).numerator
        self.ifresample_denominator = Fraction(samplerate2ifrate).denominator
        assert self.ifresample_numerator > 0, f'IF resampling numerator got 0; sample_rate {self.sample_rate}'
        assert self.ifresample_denominator > 0, f'IF resampling denominator got 0; sample_rate {self.sample_rate}'
        audiorate2ifrate = self.audio_rate / self.if_rate
        self.audioRes_numerator = Fraction(audiorate2ifrate).numerator
        self.audioRes_denominator = Fraction(audiorate2ifrate).denominator
        return self.ifresample_numerator, self.ifresample_denominator, self.audioRes_numerator, self.audioRes_denominator

    def updateDemod(self):
        self.afeL, self.afeR, self.fmL, self.fmR = self.afeParams(self.afe_params)
        return self.afeL, self.afeR, self.fmL, self.fmR

    def updateStandard(self, devL, devR):
        newLC = self.afe_params.LCarrierRef - round(devL)
        newRC = self.afe_params.RCarrierRef - round(devR)
        self.updateAFE(newLC, newRC)

    def updateAFE(self, newLC, newRC):
        self.afe_params.LCarrierRef = \
            max(
                min(newLC, self.standard.LCarrierRef + 10e3),
                self.standard.LCarrierRef - 10e3
            ) if self.options['format'] == 'vhs' else newLC
        self.afe_params.RCarrierRef = \
            max(
                min(newRC, self.standard.RCarrierRef + 10e3),
                self.standard.RCarrierRef - 10e3
            ) if self.options['format'] == 'vhs' else newRC

    def afeParams(self, standard):
        afeL = AFEFilterable(standard, self.if_rate, 0)
        afeR = AFEFilterable(standard, self.if_rate, 1)
        if self.options['preview']:
            fmL = FMdemod(self.if_rate, standard.LCarrierRef, 1)
            fmR = FMdemod(self.if_rate, standard.RCarrierRef, 1)
        elif self.options['original']:
            fmL = FMdemod(self.if_rate, standard.LCarrierRef, 2)
            fmR = FMdemod(self.if_rate, standard.RCarrierRef, 2)
        else:
            fmL = FMdemod(self.if_rate, standard.LCarrierRef, 0)
            fmR = FMdemod(self.if_rate, standard.RCarrierRef, 0)

        return afeL, afeR, fmL, fmR

    def demodblock(self, data):
        filterL = self.afeL.work(data)
        filterR = self.afeR.work(data)

        if self.options['grc']:
            self.grc.send(filterL + filterR)

        self.stereo_queue.append(
            self.stereo_executor.submit(self.fmL.work, (filterL))
        )
        self.stereo_queue.append(
            self.stereo_executor.submit(self.fmR.work, (filterR))
        )

        ifL = self.stereo_queue[0].result()
        ifR = self.stereo_queue[1].result()

        self.stereo_queue.clear()
        deemphL = self.deemphL.lfilt(ifL)
        deemphR = self.deemphR.lfilt(ifR)
        preAudioResampleL = self.preAudioResampleL.lfilt(ifL)
        preAudioResampleR = self.preAudioResampleR.lfilt(ifR)
        audioL = samplerate_resample(deemphL, self.audioRes_numerator, self.audioRes_denominator)
        audioR = samplerate_resample(deemphR, self.audioRes_numerator, self.audioRes_denominator)
        preL = samplerate_resample(preAudioResampleL, self.audioRes_numerator, self.audioRes_denominator)
        preR = samplerate_resample(preAudioResampleR, self.audioRes_numerator, self.audioRes_denominator)

        dcL = self.dcCancelL.work(np.mean(audioL))
        dcR = self.dcCancelR.work(np.mean(audioR))

        return dcL, dcR, (audioL + preL) / 2, (audioR + preR) / 2, preL, preR

    @property
    def blockSize(self) -> int:
        return self.block_size

    @property
    def readOverlap(self) -> int:
        return self.block_overlap

    @property
    def audioDiscard(self) -> int:
        return self.block_overlap_audio

    @property
    def sourceRate(self) -> int:
        return int(self.sample_rate)

    @property
    def audioRate(self) -> int:
        return self.audio_rate

    @property
    def notchFreq(self) -> float:
        return self.standard.Hfreq

    @staticmethod
    def cancelDC(c, dc):
        return c - dc

    @staticmethod
    def cancelDCpair(L, R, dcL, dcR):
        return HiFiDecode.cancelDC(L, dcL), HiFiDecode.cancelDC(R, dcR)

    def guessBiases(self, blocks):
        meanL, meanR = StackableMA(window_average=len(blocks)), StackableMA(window_average=len(blocks))
        for block in blocks:
            lo_data = self.lopassRF.work(block)
            data = samplerate_resample(lo_data, self.ifresample_numerator, self.ifresample_denominator)
            dcL, dcR, audioL, audioR, preL, preR = self.demodblock(data)
            meanL.push(np.mean(preL))
            meanR.push(np.mean(preR))

        return meanL.pull(), meanR.pull()

    def carrierOffsets(self, standard, cL, cR):
        return standard.LCarrierRef - cL, standard.RCarrierRef - cR

    def block_decode(self, raw_data, block_count=0):
        lo_data = self.lopassRF.work(raw_data)
        data = samplerate_resample(lo_data, self.ifresample_numerator, self.ifresample_denominator)
        dcL, dcR, audioL, audioR, preL, preR = self.demodblock(data)

        if self.options['auto_fine_tune']:
            self.devL, self.devR = self.carrierOffsets(self.afe_params, dcL, dcR)
            self.updateStandard(self.devL, self.devR)
            self.updateDemod()

        clip = AFEParamsPALVHS().VCODeviation
        audioL, audioR = HiFiDecode.cancelDCpair(audioL, audioR, dcL, dcR)
        audioL /= clip
        audioR /= clip

        return block_count, audioL, audioR
