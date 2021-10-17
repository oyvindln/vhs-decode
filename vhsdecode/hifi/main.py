import soundfile as sf
import numpy as np
from vhsdecode.utils import firdes_lowpass, firdes_highpass, FiltersClass, plot_scope, dualplot_scope, \
    filter_plot, gen_wave_at_frequency, StackableMA
from vhsdecode.addons.chromasep import samplerate_resample
from vhsdecode.addons.gnuradioZMQ import ZMQSend
from vhsdecode.addons.FMdeemph import FMDeEmphasis
from math import log, pi
from fractions import Fraction
from pyhht.utils import inst_freq
from lddecode.utils import unwrap_hilbert
from scipy.signal import iirpeak, iirnotch
from scipy.signal.signaltools import hilbert


# Use PyFFTW's faster FFT implementation if available
try:
    import pyfftw.interfaces.numpy_fft as npfft
    import pyfftw.interfaces

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)
except ImportError:
    import numpy.fft as npfft


class AFEParamsFront:
    def __init__(self):
        self.cutoff = 2e6
        self.FDC = 1e6


class AFEBandPass:
    def __init__(self, filters_params, sample_rate):
        self.samp_rate = sample_rate
        self.filter_params = filters_params

        iir_lo = firdes_lowpass(
            self.samp_rate,
            self.filter_params.cutoff,
            500e3
        )
        iir_hi = firdes_highpass(
            self.samp_rate,
            self.filter_params.FDC,
            500e3
        )

        #filter_plot(iir_lo[0], iir_lo[1], self.samp_rate, type="lopass", title="Front lopass")
        self.filter_lo = FiltersClass(iir_lo[0], iir_lo[1], self.samp_rate)
        self.filter_hi = FiltersClass(iir_hi[0], iir_hi[1], self.samp_rate)

    def work(self, data):
        return self.filter_lo.lfilt(self.filter_hi.lfilt(data))


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


class AFEParamsPAL:
    def __init__(self):
        self.LCarrierRef = 1.4e6
        self.RCarrierRef = 1.8e6
        self.opVCODeviation = 50e3
        self.maxVCODeviation = 150e3
        self.VCODeviation = (self.opVCODeviation + self.maxVCODeviation) / 2


class AFEFilterable:
    def __init__(self, filters_params, sample_rate, channel=0):
        self.samp_rate = sample_rate
        self.filter_params = filters_params
        d = abs(self.filter_params.LCarrierRef - self.filter_params.RCarrierRef)
        QL = self.filter_params.LCarrierRef / self.filter_params.opVCODeviation
        QR = self.filter_params.RCarrierRef / self.filter_params.opVCODeviation
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
                QR,
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
        self.offset = np.mean(self.work(self.wave))

    def hhtdeFM(self, data):
        instf, t = inst_freq(data)
        return np.add(np.multiply(instf, -self.samp_rate), self.samp_rate / 2)

    def htdeFM(self, data):
        return unwrap_hilbert(data, self.samp_rate)

    def inst_freq(self, signal):
        analytic_signal = hilbert(signal.real)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) /
                                   (2.0 * np.pi) * self.samp_rate)
        return instantaneous_frequency

    def work(self, data):

        if self.type == 2:
            return np.add(
                self.htdeFM(data),
                -self.offset
            )
        elif self.type == 1:
            return np.add(
                self.hhtdeFM(data),
                -self.offset
            )
        else:
            return np.add(
                self.inst_freq(data),
                -self.offset
            )

def getDeemph(tau, sample_rate):
    deemph = FMDeEmphasis(sample_rate, tau)
    iir_b, iir_a = deemph.get()
    return FiltersClass(iir_b, iir_a, sample_rate)


path = '/home/sebastian/vault/VHS/HiFi/test_hifi.flac'
sample_rate = 35.795454545e6
if_rate = 8388608
audio_rate = 192000
blocks_second = 20
tau = 56e-6

ifresample_numerator = Fraction(if_rate / sample_rate).limit_denominator(1000).numerator
ifresample_denominator = Fraction(if_rate / sample_rate).limit_denominator(1000).denominator
audioRes_numerator = Fraction(audio_rate / if_rate).limit_denominator(1000).numerator
audioRes_denominator = Fraction(audio_rate / if_rate).limit_denominator(1000).denominator
block_size = int(sample_rate / blocks_second)

afeParamsPAL = AFEParamsPAL()
afeL = AFEFilterable(afeParamsPAL, if_rate, 0)
afeR = AFEFilterable(afeParamsPAL, if_rate, 1)
fmL = FMdemod(if_rate, AFEParamsPAL().LCarrierRef, 1)
fmR = FMdemod(if_rate, AFEParamsPAL().RCarrierRef, 1)
deemphL = getDeemph(tau, if_rate)
deemphR = getDeemph(tau, if_rate)
lopassRF = AFEBandPass(AFEParamsFront(), sample_rate)
dcCancelL = StackableMA(min_watermark=0, window_average=blocks_second)
dcCancelR = StackableMA(min_watermark=0, window_average=blocks_second)

# grc = ZMQSend()

with sf.SoundFile('output.wav', 'w+', channels=2, samplerate=audio_rate, subtype='PCM_16', endian='little') as w:
    with sf.SoundFile(path, 'r') as f:
        while f.tell() < f.frames:
            pos = f.tell()
            raw_data = f.read(block_size)
            lo_data = lopassRF.work(raw_data)
            data = samplerate_resample(lo_data, ifresample_numerator, ifresample_denominator)
            filterL = afeL.work(data)
            filterR = afeR.work(data)
            # grc.send(filterL)

            audioL = samplerate_resample(
                deemphL.lfilt(fmL.work(filterL)), audioRes_numerator, audioRes_denominator
            )

            audioR = samplerate_resample(
                deemphR.lfilt(fmR.work(filterR)), audioRes_numerator, audioRes_denominator
            )

            dcL = dcCancelL.work(np.mean(audioL))
            dcR = dcCancelR.work(np.mean(audioR))
            print("Max L %.02f kHz, R %.02f kHz" % (np.max(audioL) / 1e3, np.max(audioR) / 1e3))
            print("Mean L %.02f kHz, R %.02f kHz" % (dcL / 1e3, dcR / 1e3))

            clip = AFEParamsPAL().VCODeviation
            audioL -= dcL
            audioL /= clip
            audioR -= dcR
            audioR /= clip

            stereo = list(map(list, zip(audioL, audioR)))
            w.write(stereo)
