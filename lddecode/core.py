import copy
import itertools
import platform
import sys
import threading
import time
import types

from queue import Queue

# standard numeric/scientific libraries
import numpy as np
import scipy.signal as sps
import scipy.interpolate as spi
import numba
from numba import njit

# Use standard numpy fft, since it's thread-safe
import numpy.fft as npfft

# internal libraries

from . import efm_pll
from .utils import get_git_info, ac3_pipe, ldf_pipe, traceback
from .utils import nb_mean, nb_median, nb_round, nb_min, nb_max, nb_abs, nb_absmax, nb_diff, n_orgt, n_orlt
from .utils import polar2z, sqsum, genwave, dsa_rescale_and_clip, scale, rms
from .utils import findpeaks, findpulses, calczc, inrange, roundfloat
from .utils import LRUupdate, clb_findbursts, angular_mean_helper, phase_distance
from .utils import build_hilbert, unwrap_hilbert, emphasis_iir, filtfft
from .utils import fft_do_slice, fft_determine_slices, StridedCollector, hz_to_output_array
from .utils import Pulse, nb_std, nb_gt, n_ornotrange

try:
    # If Anaconda's numpy is installed, mkl will use all threads for fft etc
    # which doesn't work when we do more threads, do disable that...
    import mkl

    mkl.set_num_threads(1)
except ImportError:
    # If not running Anaconda, we don't care that mkl doesn't exist.
    pass

# XXX: This is a hack so that logging is treated the same way in both this
# and ld-decode.  Probably should just bring all logging in here...
logger = None

# If profiling is not enabled, make it a pass-through function
try:
    profile
except:
    def profile(fn):
        return fn

BLOCKSIZE = 32 * 1024

def calclinelen(SP, mult, mhz):
    if type(mhz) == str:
        mhz = SP[mhz]

    return int(nb_round(SP["line_period"] * mhz * mult))


# states for first field of validpulses (second field is pulse #)
HSYNC, EQPL1, VSYNC, EQPL2 = range(4)

# These are invariant parameters for PAL and NTSC
SysParams_NTSC = {
    "fsc_mhz": (315.0 / np.double(88.0)),
    "pilot_mhz": (315.0 / 88.0),
    "frame_lines": 525,
    "field_lines": (263, 262),
    "ire0": 8100000,
    "hz_ire": 1700000 / 140.0,
    "vsync_ire": -40,
    # most NTSC disks have analog audio, except CD-V and a few Panasonic demos
    "analog_audio": True,
    # From the spec - audio frequencies are multiples of the (color) line rate
    "audio_lfreq": (1000000 * 315 / 88 / 227.5) * 146.25,
    "audio_rfreq": (1000000 * 315 / 88 / 227.5) * 178.75,
    # On AC3 disks, the right channel is replaced by a QPSK 2.88mhz channel
    "audio_rfreq_AC3": 2880000,
    "colorBurstUS": (5.3, 7.8),
    "activeVideoUS": (9.45, 63.555 - 1.0),
    # Known-good area for computing black SNR - for NTSC pull from VSYNC
    # tuple: (line, beginning, length)
    "blacksnr_slice": (1, 10, 20),
    # In NTSC framing, the distances between the first/last eq pulses and the
    # corresponding next lines are different.
    "firstFieldH": (0.5, 1),
    "numPulses": 6,  # number of equalization pulses per section
    "hsyncPulseUS": 4.7,
    "eqPulseUS": 2.3,
    "vsyncPulseUS": 27.1,
    # What 0 IRE/0V should be in digitaloutput
    "outputZero": 1024,
    "fieldPhases": 4,
    # Likely locations of solid white in VITS on LD's (line, start, length)
    # The first three are typical VITS locations (first most common), and last
    # is the MCA Code first-field flag.
    "LD_VITS_whitelocs": [(20, 14, 12), (20, 52, 8), (13, 13, 15), (11, 12, 45)],
    # Similar but with percentile to use to compute white level
    # (in case VITS white test areas are not present)
    "LD_VITS_code_slices": [(16, 12, 48, 85), (17, 12, 48, 85), (10, 13, 39, 85)],
}

# In color NTSC, the line period was changed from 63.5 to 227.5 color cycles,
# which works out to 63.555(with a bar on top) usec
SysParams_NTSC["line_period"] = 1 / (SysParams_NTSC["fsc_mhz"] / np.double(227.5))
SysParams_NTSC["activeVideoUS"] = (9.45, SysParams_NTSC["line_period"] - 1.0)

SysParams_NTSC["FPS"] = 1000000 / (525 * SysParams_NTSC["line_period"])

SysParams_NTSC["outlinelen"] = calclinelen(SysParams_NTSC, 4, "fsc_mhz")
SysParams_NTSC["outfreq"] = 4 * SysParams_NTSC["fsc_mhz"]

SysParams_PAL = {
    "FPS": 25,
    # from wikipedia: 283.75 × 15625 Hz + 25 Hz = 4.43361875 MHz
    "fsc_mhz": ((1 / 64) * 283.75) + (25 / 1000000),
    "pilot_mhz": 3.75,
    "frame_lines": 625,
    "field_lines": (312, 313),
    "line_period": 64,
    "ire0": 7100000,
    "hz_ire": 800000 / 100.0,
    # only early PAL disks have analog audio
    "analog_audio": True,
    # From the spec - audio frequencies are multiples of the (colour) line rate
    "audio_lfreq": (1000000 / 64) * 43.75,
    "audio_rfreq": (1000000 / 64) * 68.25,
    "colorBurstUS": (5.6, 7.85),
    "activeVideoUS": (10.5, 64 - 1.5),
    # In PAL, the first field's line sync<->first/last EQ pulse are both .5H
    "firstFieldH": (1, 0.5),
    # Known-good area for computing black SNR - for PAL this is blanked in mastering
    # tuple: (line, beginning, length)
    "blacksnr_slice": (22, 12, 50),
    "numPulses": 5,  # number of equalization pulses per section
    "hsyncPulseUS": 4.7,
    "eqPulseUS": 2.35,
    "vsyncPulseUS": 27.3,
    # What 0 IRE/0V should be in digitaloutput
    "outputZero": 256,
    "fieldPhases": 8,
    # Likely locations of solid white in VITS on LD's
    # (an array of line/start/length)
    "LD_VITS_whitelocs": [(19, 12, 8)],
    # Similar but with percentile to use to compute white level
    # (in case VITS white test areas are not present)
    "LD_VITS_code_slices": [(16, 11, 49, 85), (17, 11, 49, 85)],
}

SysParams_PAL["outlinelen"] = calclinelen(SysParams_PAL, 4, "fsc_mhz")
SysParams_PAL["outlinelen_pilot"] = calclinelen(SysParams_PAL, 4, "pilot_mhz")
SysParams_PAL["outfreq"] = 4 * SysParams_PAL["fsc_mhz"]

SysParams_PAL["vsync_ire"] = -0.3 * (100 / 0.7)

# RFParams are tunable

RFParams_NTSC = {
    # The audio notch filters are important with DD v3.0+ boards
    "audio_notchwidth": 350000,
    "audio_notchorder": 2,
    "video_deemp": (120e-9, 320e-9),
    # This BPF is similar but not *quite* identical to what Pioneer did
    "video_bpf_low": 3400000,
    "video_bpf_high": 13800000,
    "video_bpf_order": 4,
    # This can easily be pushed up to 4.5mhz or even a bit higher.
    # A sharp 4.8-5.0 is probably the maximum before the audio carriers bleed into 0IRE.
    "video_lpf_freq": 4500000,  # in mhz
    "video_lpf_order": 6,  # butterworth filter order
    # MTF filter
    "MTF_basemult": 0.4,  # general ** level of the MTF filter for frame 0.
    "MTF_poledist": 0.9,
    "MTF_freq": 12.2,  # in mhz
    # used to detect rot
    "video_hpf_freq": 10000000,
    "video_hpf_order": 4,
    # audio filter parameters
    "audio_filterwidth": 150000,
    "audio_filterorder": 512,
}

# Settings for use with noisier disks
RFParams_NTSC_lowband = RFParams_NTSC.copy().update({
    # The audio notch filters are important with DD v3.0+ boards
    "video_bpf_low": 3800000,
    "video_bpf_high": 12500000,
    "video_lpf_freq": 4200000,  # in mhz
})

RFParams_PAL = {
    # The audio notch filters are important with DD v3.0+ boards
    "audio_notchwidth": 200000,
    "audio_notchorder": 2,
    "video_deemp": (100e-9, 400e-9),
    # XXX: guessing here!
    "video_bpf_low": 2300000,
    "video_bpf_high": 13500000,
    "video_bpf_order": 2,
    "video_lpf_freq": 5200000,
    "video_lpf_order": 7,
    # MTF filter
    "MTF_basemult": 1.0,  # general ** level of the MTF filter for frame 0.
    "MTF_poledist": 0.70,
    "MTF_freq": 10,
    # used to detect rot
    "video_hpf_freq": 10000000,
    "video_hpf_order": 4,
    "audio_filterwidth": 100000,
    "audio_filterorder": 900,
}

RFParams_PAL_lowband = RFParams_PAL.copy().update({
    "video_bpf_low": 3200000,
    "video_bpf_high": 13000000,
    "video_bpf_order": 1,
    "video_lpf_freq": 4800000,
})


class RFDecode:
    """The core RF decoding code.

    This decoder uses FFT overlap-save processing(1) to allow for parallel processing and combination of
    operations.

    Video filter signal path:
    - FFT/iFFT stage 1: RF BPF (i.e. 3.5-13.5mhz NTSC) * hilbert filter
    - phase unwrapping
    - FFT stage 2, which is processed into multiple final products:
      - Regular video output
      - 0.5mhz LPF (used for HSYNC)
      - For fine-tuning HSYNC: NTSC: 3.5x mhz filtered signal, PAL: 3.75mhz pilot signal

    Analogue audio filter signal path:

        The audio signal path is actually more complex in some ways, since it reduces a
        multi-msps signal down to <100khz.  A two stage processing system is used which
        reduces the frequency in each stage.

        Stage 1 performs the audio RF demodulation per block typically with 32x decimation,
        while stage 2 is run once the entire frame is demodulated and decimates by 4x.

    EFM filtering simply applies RF front end filters that massage the output so that ld-process-efm
    can do the actual work.

    references:
    1 - https://en.wikipedia.org/wiki/Overlap–save_method

    """

    def __init__(
        self,
        inputfreq=40,
        system="NTSC",
        blocklen=BLOCKSIZE,
        decode_digital_audio=False,
        decode_analog_audio=0,
        has_analog_audio=True,
        extra_options={},
    ):
        """Initialize the RF decoder object.

        inputfreq -- frequency of raw RF data (in Msps)
        system    -- Which system is in use (PAL or NTSC)
        blocklen  -- Block length for FFT processing
        decode_digital_audio -- Whether to apply EFM filtering
        decode_analog_audio  -- Whether or not to decode analog(ue) audio
        has_analog_audio     -- Whether or not analog(ue) audio channels are on the disk

        extra_options -- Dictionary of additional options (typically boolean) - these include:
          - PAL_V4300D_NotchFilter - cut 8.5mhz spurious signal
          - NTSC_ColorNotchFilter:  notch filter on decoded video to reduce color 'wobble'
          - lowband: Substitute different decode settings for lower-bandwidth disks
          - AC3: Supports AC3

        """

        self.blocklen = blocklen
        self.blockcut = 1024  # ???
        self.blockcut_end = 0
        self.system = system

        self.setupcount = 0

        self.NTSC_ColorNotchFilter = extra_options.get("NTSC_ColorNotchFilter", False)
        self.PAL_V4300D_NotchFilter = extra_options.get("PAL_V4300D_NotchFilter", False)
        lowband = extra_options.get("lowband", False)

        freq = inputfreq
        self.freq = freq
        self.freq_half = freq / 2
        self.freq_hz = self.freq * 1000000
        self.freq_hz_half = self.freq_hz / 2

        self.mtf_mult   = extra_options.get("MTF_level", 1.0)
        self.mtf_offset = extra_options.get("MTF_offset", 0)

        if system == "NTSC":
            self.SysParams = copy.deepcopy(SysParams_NTSC)
            if lowband:
                self.DecoderParams = copy.deepcopy(RFParams_NTSC_lowband)
            else:
                self.DecoderParams = copy.deepcopy(RFParams_NTSC)
        elif system == "PAL":
            self.SysParams = copy.deepcopy(SysParams_PAL)
            if lowband:
                self.DecoderParams = copy.deepcopy(RFParams_PAL_lowband)
            else:
                self.DecoderParams = copy.deepcopy(RFParams_PAL)

        # Make (intentionally) mutable copies of HZ<->IRE levels
        for irekey in ['ire0', 'hz_ire', 'vsync_ire']:
            self.DecoderParams[irekey] = self.SysParams[irekey]

        self.SysParams["analog_audio"] = has_analog_audio
        self.SysParams["AC3"] = extra_options.get("AC3", False)
        if self.SysParams["AC3"]:
            self.SysParams["audio_rfreq"] = self.SysParams["audio_rfreq_AC3"]

        fw = extra_options.get("audio_filterwidth", 0)
        if fw is not None and fw > 0:
            self.DecoderParams['audio_filterwidth'] = fw

        deemp = list(self.DecoderParams["video_deemp"])

        # note that deemp[0] is the t1 (high freuqency) coefficient, and 
        # deemp[1] is the t2 (low frequency) one.  These are passed in as
        # microseconds, but need to be converted to seconds.

        deemp_low, deemp_high = extra_options.get("deemp_coeff", (0, 0))
        if deemp_low > 0:
            deemp[1] = 1 / (deemp_low  * 1000000)
        if deemp_high > 0:
            deemp[0] = 1 / (deemp_high * 1000000)

        self.DecoderParams["video_deemp"]          = deemp
        self.DecoderParams["video_deemp_strength"] = extra_options.get("deemp_str", 1.0)

        linelen = self.freq_hz / (1000000.0 / self.SysParams["line_period"])
        self.linelen = int(np.round(linelen))
        self.samplesperline = self.freq / self.linelen

        # How much horizontal sync position can deviate from previous/expected position
        # and still be interpreted as a horizontal sync pulse.
        # Too high tolerance may result in false positive sync pulses, too low may end up missing them.
        # Tapes will need a wider tolerance than laserdiscs due to head switch etc.
        self.hsync_tolerance = 0.4

        self.decode_digital_audio = decode_digital_audio
        self.decode_analog_audio = decode_analog_audio

        self.computefilters()

        # The 0.5mhz filter is rolled back to align with the data, so there
        # are a few unusable samples at the end.
        self.blockcut_end = self.Filters["F05_offset"]


    def computefilters(self):
        """ (re)compute the filter sets """

        self.setupcount += 1

        self.computevideofilters()

        # This is > 0 because decode_analog_audio is in khz (44.1, 48, 3xHSYNC, etc).
        if self.decode_analog_audio != 0:
            self.computeaudiofilters()

        if self.decode_digital_audio:
            self.computeefmfilter()

        if self.SysParams['AC3']:
            apass = 288000 * .5

            # 
            fpass = lambda apass: [(self.SysParams['audio_rfreq_AC3'] - apass) / self.freq_hz_half,
            (self.SysParams['audio_rfreq_AC3'] + apass) / self.freq_hz_half]

            # Need to clean these up
            # self.Filters['AC3_fir'] = [sps.firwin(257, fpass(apass), pass_zero=False), [1.0]]
            # XXX Made into IIR for some reason, check commit history here
            self.Filters['AC3_fir'] = sps.butter(3, fpass(apass), btype='bandpass')

            # This analog audio bandpass filter is an approximation of
            # http://sim.okawa-denshi.jp/en/RLCtool.php with resistor 2200ohm, 
            # inductor 180uH, and cap 27pF (taken from Pioneer service manuals)
            # self.Filters['AC3_iir'] = sps.butter(5, [1.48/20, 3.45/20], btype='bandpass')

            # empirically determined
            self.Filters['AC3_iir'] = sps.butter(3, [(2.88-.5)/20, (2.88+.5)/20], btype='bandpass')

            firfilt = filtfft(self.Filters['AC3_fir'], self.blocklen)
            iirfilt = filtfft(self.Filters['AC3_iir'], self.blocklen)

            self.Filters['AC3'] = iirfilt * firfilt

        self.computedelays()


    def computeefmfilter(self):
        """Frequency-domain equalisation filter for the LaserDisc EFM signal.
        This was inspired by the input signal equaliser in WSJT-X, described in
        Steven J. Franke and Joseph H. Taylor, "The MSK144 Protocol for
        Meteor-Scatter Communication", QEX July/August 2017.
        <http://physics.princeton.edu/pulsar/k1jt/MSK144_Protocol_QEX.pdf>

        This improved EFM filter was devised by Adam Sampson (@atsampson)
        """

        # Frequency bands
        freqs = np.linspace(0.0e6, 1.9e6, num=11)
        freq_per_bin = self.freq_hz / self.blocklen
        # Amplitude and phase adjustments for each band.
        # These values were adjusted empirically based on a selection of NTSC and PAL samples.
        amp = np.array(
            [0.0, 0.215, 0.41, 0.73, 0.98, 1.03, 0.99, 0.81, 0.59, 0.42, 0.0]
        )
        phase = np.array(
            [0.0, -0.92, -1.03, -1.11, -1.2, -1.2, -1.2, -1.2, -1.05, -0.95, -0.8]
        )
        phase = [p * 1.25 for p in phase]

        """Compute filter coefficients for the given FFTFilter."""
        # Anything above the highest frequency is left as zero.
        coeffs = np.zeros(self.blocklen, dtype=complex)

        # Generate the frequency-domain coefficients by cubic interpolation between the equaliser values.
        a_interp = spi.interp1d(freqs, amp, kind="cubic")
        p_interp = spi.interp1d(freqs, phase, kind="cubic")

        nonzero_bins = int(freqs[-1] / freq_per_bin) + 1

        bin_freqs = np.arange(nonzero_bins) * freq_per_bin
        bin_amp = a_interp(bin_freqs)
        bin_phase = p_interp(bin_freqs)

        # Scale by the amplitude, rotate by the phase
        coeffs[:nonzero_bins] = bin_amp * (
            np.cos(bin_phase) + (complex(0, -1) * np.sin(bin_phase))
        )

        self.Filters["Fefm"] = coeffs * 8

    # Lambda-scale functions used to simplify following filter builders

    # Split out the frequency list given to the filter builder
    def freqrange(self, f1, f2): 
        return [f1 / self.freq_hz_half, f2 / self.freq_hz_half]

    # Like freqrange, but for notch filters
    def notchrange(self, f, notchwidth, hz = False): 
        return [
            (f - notchwidth) / (self.freq_hz_half if hz else self.freq_half),
            (f + notchwidth) / (self.freq_hz_half if hz else self.freq_half)
        ]

    def computevideofilters(self):
        self.Filters = {}

        # Use some shorthand to compact the code.
        SF = self.Filters
        SP = self.SysParams
        DP = self.DecoderParams

        # This high pass filter is intended to detect RF dropouts
        Frfhpf = sps.butter(1, [10 / self.freq_half], btype="highpass")
        self.Filters["Frfhpf"] = filtfft(Frfhpf, self.blocklen)

        # First phase FFT filtering

        # MTF filter section
        # compute the pole locations symmetric to freq_half (i.e. 12.2 and 27.8)
        MTF_polef_lo = DP["MTF_freq"] / self.freq_half
        MTF_polef_hi = (
            self.freq_half + (self.freq_half - DP["MTF_freq"])
        ) / self.freq_half

        to_z = lambda pole: polar2z(DP["MTF_poledist"], np.pi * pole)

        MTF = sps.zpk2tf([], [to_z(MTF_polef_lo), to_z(MTF_polef_hi)], 1)
        SF["MTF"] = filtfft(MTF, self.blocklen)

        # The BPF filter, defined for each system in DecoderParams
        filt_rfvideo = sps.butter(
            DP["video_bpf_order"],
            self.freqrange(DP["video_bpf_low"], DP["video_bpf_high"]),
            btype="bandpass",
        )
        # Start building up the combined FFT filter using the BPF
        SF["RFVideo"] = filtfft(filt_rfvideo, self.blocklen)

        # Notch filters for analog audio.  DdD captures on NTSC need this.
        if SP["analog_audio"] and self.system == "NTSC":
            cut_left = sps.butter(
                DP["audio_notchorder"],
                self.notchrange(SP["audio_lfreq"], DP['audio_notchwidth'], True),
                btype="bandstop",
            )
            SF["Fcutl"] = filtfft(cut_left, self.blocklen)
            
            cut_right = sps.butter(
                DP["audio_notchorder"],
                self.notchrange(SP["audio_rfreq"], DP['audio_notchwidth'], True),
                btype="bandstop",
            )
            SF["Fcutr"] = filtfft(cut_right, self.blocklen)

            SF["RFVideo"] *= SF["Fcutl"] * SF["Fcutr"]

        SF["hilbert"] = build_hilbert(self.blocklen)
        SF["RFVideo"] *= SF["hilbert"]

        # Second phase FFT filtering, which is performed after the signal is demodulated

        video_lpf = sps.butter(
            DP["video_lpf_order"], DP["video_lpf_freq"] / self.freq_hz_half, "low"
        )
        SF["Fvideo_lpf"] = filtfft(video_lpf, self.blocklen)

        if self.system == "NTSC" and self.NTSC_ColorNotchFilter:
            video_notch = sps.butter(
                3,
                [DP["video_lpf_freq"] / 1000000 / self.freq_half, 5.0 / self.freq_half],
                "bandstop",
            )
            SF["Fvideo_lpf"] *= filtfft(video_notch, self.blocklen)

        # The deemphasis filter
        deemp1, deemp2 = DP["video_deemp"]
        SF["Fdeemp"] = filtfft(
            emphasis_iir(deemp1, deemp2, self.freq_hz), self.blocklen
        )

        # The direct opposite of the above, used in test signal generation
        SF["Femp"] = filtfft(emphasis_iir(deemp2, deemp1, self.freq_hz), self.blocklen)

        # Post processing:  lowpass filter + deemp
        SF["FVideo"] = SF["Fvideo_lpf"] * (SF["Fdeemp"] ** DP['video_deemp_strength'])

        # additional filters:  0.5mhz and color burst
        # Using an FIR filter here to get a known delay
        F0_5 = sps.firwin(65, [0.5 / self.freq_half], pass_zero=True)
        SF["F05_offset"] = 32
        F0_5_fft = filtfft((F0_5, [1.0]), self.blocklen)
        SF["FVideo05"] = SF["Fvideo_lpf"] * SF["Fdeemp"] * F0_5_fft

        SF["Fburst"] = filtfft(
            sps.butter(1, self.notchrange(SP["fsc_mhz"], 0.1), "bandpass"),
            self.blocklen,
        )
        SF["FVideoBurst"] = SF["Fvideo_lpf"] * SF["Fdeemp"] * SF["Fburst"]

        if self.system == "PAL":
            SF["Fpilot"] = filtfft(
                sps.butter(
                    1,
                    self.notchrange(SP["pilot_mhz"], 0.1),
                    btype="bandpass",
                ),
                self.blocklen,
            )
            SF["FVideoPilot"] = SF["Fvideo_lpf"] * SF["Fdeemp"] * SF["Fpilot"]

    def computeaudiofilters(self):
        SP = self.SysParams
        DP = self.DecoderParams

        apass = DP["audio_filterwidth"]
        afilt_len = DP["audio_filterorder"]

        self.audio = {}

        for channel, center_freq in zip(['left', 'right'], [SP['audio_lfreq'], SP['audio_rfreq']]):
            self.audio[channel] = types.SimpleNamespace()

            # Build an FIR filter for each channel's RF
            audio1_fir = filtfft(
                [
                    sps.firwin(
                        afilt_len,
                        self.notchrange(center_freq, apass, True),
                        pass_zero=False,
                    ),
                    1.0,
                ],
                self.blocklen,
            )

            # Determine the frequency offset (a1_freq) and bins (lowbin+nbin) that cover the audio RF
            # frequencies for this channel
            self.audio[channel].lowbin, self.audio[channel].nbins, self.audio[channel].a1_freq = fft_determine_slices(
                center_freq, 200000, self.freq_hz, self.blocklen
            )
            # Make a lambda to slice the regular block FFT into what we're demodulating
            # note, "ch=channel" is necessary to bind the channel ID to the lambda
            self.audio[channel].slicer = lambda x, ch=channel: fft_do_slice(x, self.audio[ch].lowbin, self.audio[ch].nbins, self.blocklen)

            # Build a 'short' hilbert transform around the sliced FFT
            sliced_hilbert = build_hilbert(self.audio[channel].nbins)

            # Add the demodulated output to this to get the actual audio wave frequency
            self.audio[channel].low_freq = self.freq_hz * (self.audio[channel].lowbin / self.blocklen)
            # Finally create the stage 1 demodulation filter (including hilbert transform)
            self.audio[channel].filt1 = self.audio[channel].slicer(audio1_fir) * sliced_hilbert

            # XXX: look into revisiting/using this for stage 2 audio?
            #self.audio[channel].audio1_buffer = StridedCollector(self.blocklen, self.blockcut + self.blockcut_end)

            # Compute stage 2 audio filters: 20k-ish LPF and deemphasis.
            N, Wn = sps.buttord(20000 / (self.audio[channel].a1_freq / 2), 24000 / (self.audio[channel].a1_freq / 2), 1, 9)
            audio2_lpf = filtfft(sps.butter(N, Wn), self.blocklen)
            # 75e-6 is 75usec/2133khz (matching American FM emphasis) and 5.3e-6 is approx.
            # a 30khz break frequency
            audio2_deemp = filtfft(
                emphasis_iir(5.3e-6, 75e-6, self.audio[channel].a1_freq), self.blocklen
            )
            self.audio[channel].audio2_filter = audio2_lpf * audio2_deemp

            # Compute the sample rate decimation caused by stage 1 binning
            self.Filters['audio_fdiv'] = self.blocklen // self.audio[channel].nbins

    def iretohz(self, ire, spec=False):
        params = self.SysParams if spec else self.DecoderParams
        return params["ire0"] + (params["hz_ire"] * ire)

    def hztoire(self, hz, spec=False):
        params = self.SysParams if spec else self.DecoderParams
        return (hz - params["ire0"]) / params["hz_ire"]

    def demodblock(self, data=None, mtf_level=0, fftdata=None, cut=False):
        mtf_level *= self.mtf_mult
        mtf_level += self.mtf_offset
        mtf_level *= self.DecoderParams["MTF_basemult"]

        return self.demodblock_cpu(data, mtf_level, fftdata, cut)


    def demodblock_cpu(self, data=None, mtf_level=0, fftdata=None, cut=False):
        rv = {}

        if fftdata is not None:
            indata_fft = fftdata
        elif data is not None:
            indata_fft = npfft.fft(data[: self.blocklen])
        else:
            raise Exception("demodblock called without raw or FFT data")

        rotdelay = 0
        if getattr(self, "delays", None) is not None and "video_rot" in self.delays:
            rotdelay = self.delays["video_rot"]

        rv["rfhpf"] = npfft.ifft(indata_fft * self.Filters["Frfhpf"]).real
        rv["rfhpf"] = rv["rfhpf"][
            self.blockcut - rotdelay : -self.blockcut_end - rotdelay
        ].astype(np.float32)

        if self.system == "PAL" and self.PAL_V4300D_NotchFilter:
            """ This routine works around an 'interesting' issue seen with LD-V4300D players and
                some PAL digital audio disks, where there is a signal somewhere between 8.47 and 8.57mhz.

                The idea here is to look for anomolies (3 std deviations) and snip them out of the
                FFT.  There may be side effects, however, but generally minor compared to the
                'wibble' itself and only in certain cases.
            """
            sl = slice(
                int(self.blocklen * (8.42 / self.freq)),
                int(1 + (self.blocklen * (8.6 / self.freq))),
            )
            sq_sl = sqsum(indata_fft[sl])
            m = np.mean(sq_sl) + (np.std(sq_sl) * 3)

            for i in np.where(sq_sl > m)[0]:
                indata_fft[(i - 1 + sl.start)] = 0
                indata_fft[(i + sl.start)] = 0
                indata_fft[(i + 1 + sl.start)] = 0
                indata_fft[self.blocklen - (i + sl.start)] = 0
                indata_fft[self.blocklen - (i - 1 + sl.start)] = 0
                indata_fft[self.blocklen - (i + 1 + sl.start)] = 0

        indata_fft_filt = indata_fft * self.Filters["RFVideo"]

        if mtf_level != 0:
            indata_fft_filt *= self.Filters["MTF"] ** mtf_level

        hilbert = npfft.ifft(indata_fft_filt)
        demod = unwrap_hilbert(hilbert, self.freq_hz)

        # use a clipped demod for video output processing to reduce speckling impact
        demod_fft = npfft.fft(np.clip(demod, 1500000, self.freq_hz * 0.75))

        out_video = npfft.ifft(demod_fft * self.Filters["FVideo"]).real

        out_video05 = npfft.ifft(demod_fft * self.Filters["FVideo05"]).real
        out_video05 = np.roll(out_video05, -self.Filters["F05_offset"])

        out_videoburst = npfft.ifft(demod_fft * self.Filters["FVideoBurst"]).real

        if self.system == "PAL":
            out_videopilot = npfft.ifft(demod_fft * self.Filters["FVideoPilot"]).real
            video_out = np.rec.array(
                [
                    out_video.astype(np.float32),
                    demod.astype(np.float32),
                    out_video05.astype(np.float32),
                    out_videoburst.astype(np.float32),
                    out_videopilot.astype(np.float32),
                ],
                names=[
                    "demod",
                    "demod_raw",
                    "demod_05",
                    "demod_burst",
                    "demod_pilot",
                ],
            )
        else:
            video_out = np.rec.array(
                [out_video.astype(np.float32), demod.astype(np.float32), out_video05.astype(np.float32), out_videoburst.astype(np.float32)],
                names=["demod", "demod_raw", "demod_05", "demod_burst"],
            )

        rv["video"] = (
            video_out[self.blockcut : -self.blockcut_end] if cut else video_out
        )

        if self.decode_digital_audio:
            efm_out = npfft.ifft(indata_fft * self.Filters["Fefm"])
            if cut:
                efm_out = efm_out[self.blockcut : -self.blockcut_end]
            rv["efm"] = np.int16(np.clip(efm_out.real, -32768, 32767))

        # NOTE: ac3 audio is filtered after RF TBC
        if self.decode_analog_audio:
            stage1_out = []
            for channel in ['left', 'right']:
                afilter = self.audio[channel]

                # Apply first stage audio filter
                a1 = npfft.ifft(afilter.slicer(indata_fft) * afilter.filt1)
                # Demodulate and restore frequency after bin slicing
                a1u = unwrap_hilbert(a1, afilter.a1_freq) + afilter.low_freq

                stage1_out.append(a1u.astype(np.float32))

            audio_out = np.rec.array(
                [stage1_out[0].astype(np.float32), stage1_out[1].astype(np.float32)], names=["audio_left", "audio_right"]
            )

            fdiv = video_out.shape[0] // audio_out.shape[0]
            rv["audio"] = (
                audio_out[self.blockcut // fdiv : -self.blockcut_end // fdiv]
                if cut
                else audio_out
            )

        rv['setupcount'] = self.setupcount

        return rv

    # Second phase audio filtering.  This works on a whole field's samples, since
    # the frequency has already been reduced.

    def runfilter_audio_phase2(self, frame_audio, start):
        outputs = []

        clips = None

        for acname, center_freq, channel in [["audio_left", self.SysParams["audio_lfreq"], "left"], ["audio_right", self.SysParams["audio_rfreq"], "right"]]:
            raw = (
                frame_audio[acname][start : start + self.blocklen].copy()
            )
            raw -= center_freq

            if acname == "audio_left":
                clips = findpeaks(raw, 500000)

            for l in clips:
                replacelen = 8
                raw[max(0, l - replacelen) : min(l + replacelen, len(raw))] = 0

            a2_in_real = raw
            if len(a2_in_real) < len(self.audio[channel].audio2_filter):
                a2_in = np.zeros_like(self.audio[channel].audio2_filter)
                a2_in[: len(a2_in_real)] = a2_in_real
            else:
                a2_in = a2_in_real

            a2_fft = npfft.fft(a2_in)
            fft_out = a2_fft * self.audio[channel].audio2_filter
            output = npfft.ifft(fft_out).real[: len(a2_in_real)] + center_freq

            outputs.append(output)

        return np.rec.array(outputs, names=["audio_left", "audio_right"])

    def audio_phase2(self, field_audio):
        # this creates an output array with left/right channels.
        output_audio2 = np.zeros(
            len(field_audio["audio_left"]),
            dtype=field_audio.dtype,
        )

        # copy the first block in it's entirety, to keep audio and video samples aligned
        tmp = self.runfilter_audio_phase2(field_audio, 0)

        #print(len(tmp), len(output_audio2), len(field_audio["audio_left"]), len(self.audio['left'].audio2_filter))

        if len(tmp) >= len(output_audio2):
            return tmp[: len(output_audio2)]

        output_audio2[: tmp.shape[0]] = tmp

        end = field_audio.shape[0]

        askip = 512  # length of filters that needs to be chopped out of the ifft
        sjump = self.blocklen - askip

        ostart = tmp.shape[0]
        for sample in range(sjump, field_audio.shape[0] - sjump, sjump):
            tmp = self.runfilter_audio_phase2(field_audio, sample)

            oend = ostart + tmp.shape[0] - askip
            output_audio2[ostart:oend] = tmp[askip:]
            ostart += tmp.shape[0] - askip

        tmp = self.runfilter_audio_phase2(field_audio, end - self.blocklen - 1)
        output_audio2[output_audio2.shape[0] - (tmp.shape[0] - askip) :] = tmp[askip:]

        return output_audio2

    def computedelays(self, mtf_level=0):
        """Generate a fake signal and compute filter delays.

        mtf_level -- Specify the amount of MTF compensation needed (default 0.0)
                     WARNING: May not actually work.
        """

        rf = self

        filterset = rf.Filters
        fakeoutput = np.zeros(rf.blocklen, dtype=np.double)

        # set base level to black
        fakeoutput[:] = rf.iretohz(0)

        synclen_full = int(4.7 * rf.freq)

        # sync 1 (used for gap determination)
        fakeoutput[1500 : 1500 + synclen_full] = rf.iretohz(rf.DecoderParams["vsync_ire"])
        # sync 2 (used for pilot/rot level setting)
        fakeoutput[2000 : 2000 + synclen_full] = rf.iretohz(rf.DecoderParams["vsync_ire"])

        porch_end = 2000 + synclen_full + int(0.6 * rf.freq)
        burst_end = porch_end + int(1.2 * rf.freq)

        rate = np.full(burst_end - porch_end, rf.SysParams["fsc_mhz"], dtype=np.double)
        fakeoutput[porch_end:burst_end] += (
            genwave(rate, rf.freq / 2) * rf.DecoderParams["hz_ire"] * 20
        )

        # white
        fakeoutput[3000:3500] = rf.iretohz(100)

        # white + burst
        fakeoutput[4500:5000] = rf.iretohz(100)

        rate = np.full(5500 - 4200, rf.SysParams["fsc_mhz"], dtype=np.double)
        fakeoutput[4200:5500] += (
            genwave(rate, rf.freq / 2) * rf.DecoderParams["hz_ire"] * 20
        )

        rate = np.full(synclen_full, rf.SysParams["fsc_mhz"], dtype=np.double)
        fakeoutput[2000 : 2000 + synclen_full] = rf.iretohz(
            rf.DecoderParams["vsync_ire"]
        ) + (
            genwave(rate, rf.freq / 2)
            * rf.DecoderParams["hz_ire"]
            * rf.DecoderParams["vsync_ire"]
        )

        # add filters to generate a fake signal

        # NOTE: group pre-delay is not implemented, so the decoded signal
        # has issues settling down.  Emphasis is correct AFAIK

        tmp = npfft.fft(fakeoutput)
        tmp2 = tmp * (filterset["Fvideo_lpf"] ** 1)
        tmp3 = tmp2 * (filterset["Femp"] ** 1)

        # fakeoutput_lpf = npfft.ifft(tmp2).real
        fakeoutput_emp = npfft.ifft(tmp3).real

        fakesignal = genwave(fakeoutput_emp, rf.freq_hz / 2)
        fakesignal *= 4096
        fakesignal += 8192
        fakesignal[6000:6005] = 0

        fakedecode = rf.demodblock_cpu(fakesignal, mtf_level=mtf_level)

        vdemod = fakedecode["video"]["demod"]
        vdemod_raw = fakedecode["video"]["demod_raw"]
        vsync_cross_hz = rf.iretohz(rf.DecoderParams["vsync_ire"] / 2)

        # XXX: sync detector does NOT reflect actual sync detection, just regular filtering @ sync level
        # (but only regular filtering is needed for DOD)
        rf.delays = {}
        rf.delays["video_sync"] = calczc(vdemod, 1500, vsync_cross_hz, count=512) - 1500
        rf.delays["video_white"] = (
            calczc(vdemod, 3000, rf.iretohz(50), count=512) - 3000
        )
        rf.delays["video_rot"] = int(
            np.round(calczc(vdemod, 6000, rf.iretohz(-10), count=512) - 6000)
        )

        rf.limits = {}
        rf.limits["sync"] = (
            np.min(vdemod_raw[1400:2800]),
            np.max(vdemod_raw[1400:2800]),
        )
        rf.limits["viewable"] = (
            np.min(vdemod_raw[2900:6000]),
            np.max(vdemod_raw[2900:6000]),
        )

        return fakedecode, fakeoutput_emp


class DemodCache:
    def __init__(
        self,
        rf,
        infile,
        loader,
        rf_args,
        cachesize=256,
        num_worker_threads=6,
        MTF_tolerance=0.05,
    ):
        self.infile = infile
        self.loader = loader
        self.rf = rf
        self.rf_args = rf_args

        self.currentMTF = 1
        self.MTF_tolerance = MTF_tolerance

        self.blocksize = self.rf.blocklen - (self.rf.blockcut + self.rf.blockcut_end)

        # Cache dictionary - key is block #, which holds data for that block
        self.lrusize = cachesize

        # should be in self.rf, but may not be computed yet
        self.bytes_per_field = int(self.rf.freq_hz / (self.rf.SysParams["FPS"] * 2)) + 1
        self.prefetch = int((self.bytes_per_field * 2) / self.blocksize) + 4

        self.lru = []

        self.lock = threading.Lock()
        self.blocks = {}

        self.block_status = {}

        self.q_in = Queue()
        self.q_out = Queue()
        self.waiting = set()
        self.q_out_event = threading.Event()

        self.threadpipes = []
        self.threads = []

        self.request = 0
        self.ended = False

        self.deqeue_thread = threading.Thread(target=self.dequeue, daemon=True)
        num_worker_threads = max(num_worker_threads - 1, 1)

        for i in range(num_worker_threads):
            t = threading.Thread(
                target=self.worker, daemon=True, args=()
            )
            t.start()
            self.threads.append(t)

        self.deqeue_thread.start()

    def end(self):
        if not self.ended:
            # stop workers
            for i in self.threads:
                self.q_in.put(None)

            for t in self.threads:
                t.join()

            self.q_out.put(None)
            self.deqeue_thread.join()
            # Make sure the reader is closed properly to avoid ffmpeg warnings on exit
            # Might want to do this in a cleaner way later but this works for now.
            if hasattr(self.loader, "_close") and callable(self.loader._close):
                self.loader._close()
            self.ended = True

    def __del__(self):
        self.end()

    def prune_cache(self):
        """ Prune the LRU cache.  Typically run when a new field is loaded """
        if len(self.lru) < self.lrusize:
            return

        with self.lock:
            for k in self.lru[self.lrusize :]:
                if k in self.blocks:
                    del self.blocks[k]
                if k in self.block_status:
                    self.block_status[k]

        self.lru = self.lru[: self.lrusize]

    def flush_demod(self, first_block = 0, prefetch_only=False):
        """ Flush all demodulation data.  This is called by the field class after calibration (i.e. MTF) is determined to be off """
        blocks_toredo = []

        with self.lock:
            for k in self.block_status.keys():
                if k < first_block:
                    continue 

                if prefetch_only and self.block_status[k]['prefetch'] == False:
                    continue

                if self.block_status[k]['prefetch'] == True:
                    blocks_toredo.append(k)

                self.block_status[k] = {'MTF': -1, 'request': -1, 'waiting': False, 'prefetch': False}

            for k in self.blocks.keys():
                if k < first_block or self.blocks[k] is None or 'demod' not in self.blocks[k]:
                    continue

                if k not in blocks_toredo:
                    blocks_toredo.append(k)

                del self.blocks[k]["demod"]

        return blocks_toredo

    def apply_newparams(self, newparams):
        for k in newparams.keys():
            if k in self.rf.SysParams:
                self.rf.SysParams[k] = newparams[k]

            if k in self.rf.DecoderParams:
                self.rf.DecoderParams[k] = newparams[k]

        self.rf.computefilters()

    def worker(self):
        blocksrun = 0
        blockstime = 0

        rf = self.rf #RFDecode(**self.rf_args)

        while True:
            item = self.q_in.get()

            if item is None or item[0] == "END":
                return

            if item[0] == "DEMOD":
                blocknum, block, target_MTF, request = item[1:]

                output = {}

                if "fft" not in block:
                    output["fft"] = npfft.fft(block["rawinput"])
                    fftdata = output["fft"]
                else:
                    fftdata = block["fft"]

                if True or (
                    "demod" not in block
                    or np.abs(block["MTF"] - target_MTF) > self.MTF_tolerance
                ):
                    st = time.time()
                    output["demod"] = rf.demodblock(
                        data=block["rawinput"], fftdata=fftdata, mtf_level=target_MTF, cut=True
                    )
                    blockstime += time.time() - st
                    blocksrun += 1

                    output["MTF"] = target_MTF
                    output["request"] = request

                # print(blocknum, output)
                self.q_out.put((blocknum, output))
            elif item[0] == "NEWPARAMS":
                self.apply_newparams(item[1])


    def doread(self, blocknums, MTF, redo=False, prefetch=False):
        need_blocks = []
        queuelist = []
        reached_end = False

        if redo:
            for b in self.flush_demod(prefetch_only=True):
                queuelist.append(b)

        with self.lock:
            for b in blocknums:
                if b not in self.blocks:
                    LRUupdate(self.lru, b)

                    rawdata = self.loader(
                        self.infile, b * self.blocksize, self.rf.blocklen
                    )

                    if rawdata is None or len(rawdata) < self.rf.blocklen:
                        self.blocks[b] = None
                        return None

                    self.blocks[b] = {}
                    self.blocks[b]["rawinput"] = rawdata

                if self.blocks[b] is None:
                    reached_end = True
                    break

                waiting = (
                    self.block_status[b].get("waiting", False)
                    if b in self.block_status
                    else False
                )

                # Until the block is actually ready, this comparison will hit an unknown key
                if (
                    not redo
                    and not waiting
                    and "request" in self.blocks[b]
                    and "request" in self.block_status[b]
                    and self.blocks[b]["request"] == self.block_status[b]["request"]
                ):
                    continue

                if redo or not waiting:
                    queuelist.append(b)
                    need_blocks.append(b)
                elif waiting:
                    need_blocks.append(b)

                if not prefetch:
                    self.waiting.add(b)

            for b in queuelist:
                self.block_status[b] = {
                    "MTF": MTF,
                    "waiting": True,
                    "request": self.request,
                    "prefetch": prefetch,
                }
                self.q_in.put(("DEMOD", b, self.blocks[b], MTF, self.request))

        self.q_out_event.clear()
        return None if reached_end else need_blocks

    def dequeue(self):
        # This is the thread's main loop - run until killed.
        while True:
            rv = self.q_out.get()
            if rv is None:
                return

            with self.lock:
                blocknum, item = rv

                if "MTF" not in item or "demod" not in item:
                    # This shouldn't happen, but was observed by Simon on a decode
                    logger.error(
                        "incomplete demodulated block placed on queue, block #%d", blocknum
                    )
                    self.q_in.put((blocknum, self.blocks[blocknum], self.currentMTF, self.request))
                    continue

                if item['request'] == self.block_status[blocknum]['request']:
                    for k in item.keys():
                        self.blocks[blocknum][k] = item[k]

                    if 'demod' in item.keys():
                        if self.block_status[blocknum]['waiting']:
                            self.block_status[blocknum]['waiting'] = False

                    if blocknum in self.waiting:
                        self.waiting.remove(blocknum)

                    if not len(self.waiting):
                        self.q_out_event.set()

                if "input" not in self.blocks[blocknum]:
                    self.blocks[blocknum]["input"] = self.blocks[blocknum]["rawinput"][
                        self.rf.blockcut : -self.rf.blockcut_end
                    ]

    def read(self, begin, length, MTF=0, getraw = False, forceredo=False):
        # transpose the cache by key, not block #
        t = {"input": [], "fft": [], "video": [], "audio": [], "efm": [], "rfhpf": []}

        self.currentMTF = MTF
        if forceredo:
            self.request += 1

        end = begin + length

        toread = range(begin // self.blocksize, (end // self.blocksize) + 1)
        toread_prefetch = range(
            end // self.blocksize, (end // self.blocksize) + self.prefetch
        )

        need_blocks = self.doread(toread, MTF, forceredo)

        if getraw:
            raw = [self.blocks[toread[0]]["rawinput"][begin % self.blocksize :]]
            for i in range(toread[1], toread[-2]):
                raw.append(self.blocks[i]["rawinput"])
            raw.append(self.blocks[-1]["rawinput"][: end % self.blocksize])

            rv = np.concatenate(raw)
            self.prune_cache()
            return rv

        while need_blocks is not None and len(need_blocks):
            self.q_out_event.wait(.01)
            need_blocks = self.doread(toread, MTF)
            if need_blocks:
                self.q_out_event.clear()

        if need_blocks is None:
            # EOF
            return None

        # Now coalesce the output
        for b in range(begin // self.blocksize, (end // self.blocksize) + 1):
            for k in t.keys():
                if k in self.blocks[b]["demod"]:
                    t[k].append(self.blocks[b]["demod"][k])
                elif k in self.blocks[b]:
                    t[k].append(self.blocks[b][k])

        self.prune_cache()

        rv = {}
        for k in t.keys():
            rv[k] = np.concatenate(t[k]) if len(t[k]) else None

        if rv["audio"] is not None:
            rv["audio_phase1"] = rv["audio"]
            rv["audio"] = self.rf.audio_phase2(rv["audio"])

        rv["startloc"] = (begin // self.blocksize) * self.blocksize

        need_blocks = self.doread(toread_prefetch, MTF, prefetch=True)

        return rv

    def setparams(self, params):
        # XXX: This should flush out the data, but right now this isn't used at all
        for p in self.threadpipes:
            p[0].send(("NEWPARAMS", params))

        # Apply params to the core thread, so they match up with the decoders
        self.apply_newparams(params)


@njit(cache=True, nogil=True)
def _downscale_audio_compute_locs_and_swow(
    lineinfo, line_period, linelen, linecount, timeoffset, freq, scale
):
    """compute locations and wow for audio scaling?

    Parameters:
        lineinfo (list(float)): line locations
        line_period (int): Length of a line in usecs
        linelen (int): Length of a line in samples
        linecount (int): # of lines in field
        timeoffset (float): time of first audio sample (ignored w/- frequency)
        freq (int): Output frequency (negative values are multiple of HSYNC frequency)
        scale (int): sample rate decimation factor
    Returns: (tuple)
        locs (np.ndarray(float)): output location of audio sample?
        swow (np.ndarray(float)): offset/wow of sample?
        arange (np.ndarray(float)): "ticks" to align samples to?
        frametime (float): how long a (audio?) frame lasts
    """

    if freq < 0:
        # Override timeoffset value and set frequency to a multiple horizontal line clock
        timeoffset = 0
        freq = (1000000 / line_period) * -freq

    frametime = linecount / (1000000 / line_period)
    soundgap = 1 / freq

    # include one extra 'tick' to interpolate the last one and use as a return value
    # for the next frame
    arange = np.arange(
        timeoffset, frametime + (soundgap / 2), soundgap, dtype=np.double
    )

    locs = np.zeros(len(arange), dtype=numba.float64)
    swow = np.zeros(len(arange), dtype=numba.float64)

    for i, t in enumerate(arange):
        linenum = ((t * 1000000) / line_period) + 1
        intlinenum = int(linenum)

        # XXX:
        # The timing handling can sometimes go outside the bounds of the known line #'s.
        # This is a quick-ish fix that should work OK but may affect quality slightly.
        if linenum < 0:
            lineloc_cur = int(lineinfo[0] + (linelen * linenum))
            lineloc_next = lineloc_cur + linelen
        elif len(lineinfo) > linenum + 2:
            lineloc_cur, lineloc_next = lineinfo[intlinenum : intlinenum + 2]
        else:
            # Catch things that go past the last known line by using the last lines here.
            lineloc_cur = lineinfo[-2]
            lineloc_next = lineloc_cur + linelen

        sampleloc = lineloc_cur
        sampleloc += (lineloc_next - lineloc_cur) * (linenum - np.floor(linenum))

        swow[i] = (lineloc_next - lineloc_cur) / linelen
        swow[i] = ((swow[i] - 1)) + 1
        # There's almost *no way* the disk is spinning more than 1.5% off, so mask TBC errors here
        # to reduce pops
        if i and np.abs(swow[i] - swow[i - 1]) > 0.015:
            swow[i] = swow[i - 1]

        locs[i] = sampleloc / scale

    return locs, swow, arange, frametime


@njit(cache=True, nogil=True)
def _downscale_audio_to_output(
    arange, locs, swow, audio_left, audio_right, audio_lfreq, audio_rfreq
):
    """decimate audio to final output samples.

    Parameters:
        arange (np.arange(float)): "ticks" to align samples to?
        locs (np.ndarray(float)): output location of audio sample?
        swow (np.ndarray(float)): offset/wow of sample?
        audio_left (np.array(float)): left channel demodulated audio
        audio_right (np.array(float)): right channel demodulated audio
        audio_lfreq (float): left audio channel frequency
        audio_rfreq (float): right audio channel frequency
    Returns: (tuple)
        output (np.ndarray(int16)): output audio waveform
        failed (bool): whether there were any failed samples that were muted
    """
    output = np.zeros((2 * (len(arange) - 1)), dtype=np.int16)

    failed = False

    for i in range(len(arange) - 1):
        start = int(locs[i])
        end = int(locs[i + 1])
        if end > start and end < len(audio_left):
            output_left = nb_mean(audio_left[start:end])
            output_right = nb_mean(audio_right[start:end])

            output_left = (output_left * swow[i]) - audio_lfreq
            output_right = (output_right * swow[i]) - audio_rfreq

            # Flipping audio here to line up with ralf/he010 digital sample
            # (when comparing, remove the first 265 samples of ralf.pcm as well)
            output[(i * 2) + 0] = -dsa_rescale_and_clip(output_left)
            output[(i * 2) + 1] = -dsa_rescale_and_clip(output_right)
        else:
            # TBC failure can cause this (issue #389)
            failed = True

    return output, failed


# Downscales to 16bit/44.1khz.  It might be nice when analog audio is better to support 24/96,
# but if we only support one output type, matching CD audio/digital sound is greatly preferable.
def downscale_audio(audio, lineinfo, rf, linecount, timeoffset=0, freq=44100, rv=None):
    """downscale audio for output.

    Parameters:
        audio (float): Raw audio samples from RF demodulator
        lineinfo (list(float)): line locations
        rf (RFDecode): rf class
        linecount (int): # of lines in field
        timeoffset (float): time of first audio sample (ignored w/- frequency)
        freq (int): Output frequency (negative values are multiple of HSYNC frequency)
    Returns: (tuple)
        output16 (np.array(int)):  Array of 16-bit integers, ready for output
        next_timeoffset (float): Time to start pulling samples in the next frame (ignore if sync4x)
    """

    locs, swow, arange, frametime = _downscale_audio_compute_locs_and_swow(
        lineinfo,
        rf.SysParams["line_period"],
        rf.linelen,
        linecount,
        timeoffset,
        freq,
        rf.Filters["audio_fdiv"],
    )

    output16, failed = _downscale_audio_to_output(
        arange,
        locs,
        swow,
        audio["audio_left"],
        audio["audio_right"],
        rf.SysParams["audio_lfreq"],
        rf.SysParams["audio_rfreq"],
    )

    if failed:
        logger.warning("Analog audio processing error, muting samples")

    if rv is not None:
        rv['dsaudio'] = output16
        rv['audio_next_offset'] = arange[-1] - frametime

    return output16, arange[-1] - frametime


# The Field class contains common features used by NTSC and PAL
class Field:
    def __init__(
        self,
        rf,
        decode,
        prevfield=None,
        initphase=False,
        fields_written=0,
        readloc=0,
    ):
        self.rawdata = decode["input"]
        self.data = decode
        self.initphase = initphase  # used for seeking or first field
        self.readloc = readloc

        self.prevfield = prevfield
        self.fields_written = fields_written

        self.rf = rf
        self.freq = self.rf.freq

        self.inlinelen = self.rf.linelen
        self.outlinelen = self.rf.SysParams["outlinelen"]

        self.lineoffset = 0

        self.needrerun = False
        self.valid = False
        self.sync_confidence = 100

        self.dspicture = None
        self.dsaudio = None

        # On NTSC linecount rounds up to 263, and PAL 313
        self.outlinecount = (self.rf.SysParams["frame_lines"] // 2) + 1
        # this is eventually set to 262/263 and 312/313 for audio timing
        self.linecount = None

    #@profile
    def process(self):
        self.linelocs1, self.linebad, self.nextfieldoffset = self.compute_linelocs()
        #print(self.readloc, self.linelocs1, self.nextfieldoffset)
        if self.linelocs1 is None:
            if self.nextfieldoffset is None:
                self.nextfieldoffset = self.rf.linelen * 200

            return

        self.linebad = self.compute_deriv_error(self.linelocs1, self.linebad)

        self.linelocs2 = self.refine_linelocs_hsync()
        self.linebad = self.compute_deriv_error(self.linelocs2, self.linebad)

        self.linelocs = self.linelocs2
        self.wowfactor = self.computewow(self.linelocs)

        self.valid = True

    @profile
    def get_linelen(self, line=None, linelocs=None):
        # compute adjusted frequency from neighboring line lengths

        if line is None:
            return self.rf.linelen

        # If this is run early, line locations are unknown, so return
        # the general value
        if linelocs is None:
            if hasattr(self, "linelocs"):
                linelocs = self.linelocs
            else:
                return self.rf.linelen

        if line >= self.linecount + self.lineoffset:
            length = (self.linelocs[line + 0] - self.linelocs[line - 1]) / 1
        elif line > 0:
            length = (self.linelocs[line + 1] - self.linelocs[line - 1]) / 2
        elif line == 0:
            length = (self.linelocs[line + 1] - self.linelocs[line - 0]) / 1

        if length <= 0:
            # linelocs aren't monotonic -- probably TBC failure
            return self.rf.linelen

        return length

    def get_linefreq(self, line=None, linelocs=None):
        return self.rf.samplesperline * self.get_linelen(line, linelocs)

    def usectoinpx(self, x, line=None):
        return x * self.get_linefreq(line)

    def inpxtousec(self, x, line=None):
        return x / self.get_linefreq(line)

    @profile
    def lineslice(self, l, begin=None, length=None, linelocs=None, begin_offset=0):
        """ return a slice corresponding with pre-TBC line l, begin+length are uSecs """

        # for PAL, each field has a different offset so normalize that
        l_adj = l + self.lineoffset

        _begin = linelocs[l_adj] if linelocs is not None else self.linelocs[l_adj]
        _begin += self.usectoinpx(begin, l_adj) if begin is not None else 0

        _length = length if length else self.rf.SysParams["line_period"]
        _length = self.usectoinpx(_length)

        return slice(
            int(_begin + begin_offset),
            int(_begin + _length + begin_offset + 1),
        )

    def usectooutpx(self, x):
        return x * self.rf.SysParams["outfreq"]

    def outpxtousec(self, x):
        return x / self.rf.SysParams["outfreq"]

    #@profile
    def hz_to_output(self, input):
        if type(input) == np.ndarray:
            return hz_to_output_array(
                input,
                self.rf.DecoderParams["ire0"],
                self.rf.DecoderParams["hz_ire"],
                self.rf.SysParams["outputZero"],
                self.rf.DecoderParams["vsync_ire"],
                self.out_scale
            )

        reduced = (input - self.rf.DecoderParams["ire0"]) / self.rf.DecoderParams["hz_ire"]
        reduced -= self.rf.DecoderParams["vsync_ire"]

        return np.uint16(
            np.clip(
                (reduced * self.out_scale) + self.rf.SysParams["outputZero"], 0, 65535
            )
            + 0.5
        )

    def output_to_ire(self, output):
        return (
            (output - self.rf.SysParams["outputZero"]) / self.out_scale
        ) + self.rf.DecoderParams["vsync_ire"]


    def lineslice_tbc(self, l, begin=None, length=None, linelocs=None, keepphase=False):
        """ return a slice corresponding with pre-TBC line l """

        _begin = self.rf.SysParams["outlinelen"] * (l - 1)

        begin_offset = self.usectooutpx(begin) if begin is not None else 0
        if keepphase:
            begin_offset = (begin_offset // 4) * 4

        _begin += begin_offset
        _length = (
            self.usectooutpx(length)
            if length is not None
            else self.rf.SysParams["outlinelen"]
        )

        return slice(nb_round(_begin), nb_round(_begin + _length))

    @profile
    def get_timings(self):
        pulses = self.rawpulses
        hsync_typical = self.usectoinpx(self.rf.SysParams["hsyncPulseUS"])

        # Some disks have odd sync levels resulting in short and/or long pulse lengths.
        # So, take the median hsync and adjust the expected values accordingly

        hsync_checkmin = self.usectoinpx(self.rf.SysParams["hsyncPulseUS"] - 1.75)
        hsync_checkmax = self.usectoinpx(self.rf.SysParams["hsyncPulseUS"] + 2)

        hlens = []
        for p in pulses:
            if inrange(p.len, hsync_checkmin, hsync_checkmax):
                hlens.append(p.len)

        LT = {}
        LT = {}
        if len(hlens) > 0:
            LT["hsync_median"] = np.median(hlens)
        else:
            LT["hsync_median"] = self.rf.SysParams["hsyncPulseUS"]

        hsync_min = LT["hsync_median"] + self.usectoinpx(-0.5)
        hsync_max = LT["hsync_median"] + self.usectoinpx(0.5)

        LT["hsync"] = (hsync_min, hsync_max)

        LT["hsync_offset"] = LT["hsync_median"] - hsync_typical

        # ??? - replace self.usectoinpx with local timings?
        eq_min = (
            self.usectoinpx(self.rf.SysParams["eqPulseUS"] - 0.5) + LT["hsync_offset"]
        )
        eq_max = (
            self.usectoinpx(self.rf.SysParams["eqPulseUS"] + 0.5) + LT["hsync_offset"]
        )

        LT["eq"] = (eq_min, eq_max)

        vsync_min = (
            self.usectoinpx(self.rf.SysParams["vsyncPulseUS"] * 0.5)
            + LT["hsync_offset"]
        )
        vsync_max = (
            self.usectoinpx(self.rf.SysParams["vsyncPulseUS"] + 1) + LT["hsync_offset"]
        )

        LT["vsync"] = (vsync_min, vsync_max)

        return LT

    def pulse_qualitycheck(self, prevpulse: Pulse, pulse: Pulse):
        if prevpulse[0] > 0 and pulse[0] > 0:
            exprange = (0.4, 0.6)
        elif prevpulse[0] == 0 and pulse[0] == 0:
            exprange = (0.9, 1.1)
        else:  # transition to/from regular hsyncs can be .5 or 1H
            exprange = (0.4, 1.1)

        linelen = (pulse[1].start - prevpulse[1].start) / self.inlinelen
        inorder = inrange(linelen, *exprange)

        return inorder

    #@profile
    def run_vblank_state_machine(self, pulses, LT):
        """ Determines if a pulse set is a valid vblank by running a state machine """

        done = 0

        vsyncs = []  # VSYNC area (first broad pulse->first EQ after broad pulses)

        validpulses = []
        vsync_start = None

        # state_end tracks the earliest expected phase transition...
        state_end = 0
        # ... and state length is set by the phase transition to set above (in H)
        state_length = None

        # state order: HSYNC -> EQPUL1 -> VSYNC -> EQPUL2 -> HSYNC
        HSYNC, EQPL1, VSYNC, EQPL2 = range(4)

        for p in pulses:
            spulse = None

            state = validpulses[-1][0] if len(validpulses) > 0 else -1

            if state == -1:
                # First valid pulse must be a regular HSYNC
                if inrange(p.len, *LT["hsync"]):
                    spulse = (HSYNC, p)
            elif state == HSYNC:
                # HSYNC can transition to EQPUL/pre-vsync at the end of a field
                if inrange(p.len, *LT["hsync"]):
                    spulse = (HSYNC, p)
                elif inrange(p.len, *LT["eq"]):
                    spulse = (EQPL1, p)
                    state_length = self.rf.SysParams["numPulses"] / 2
                elif inrange(p.len, *LT["vsync"]):
                    # should not happen(tm)
                    vsync_start = len(validpulses) - 1
                    spulse = (VSYNC, p)
            elif state == EQPL1:
                if inrange(p.len, *LT["eq"]):
                    spulse = (EQPL1, p)
                elif inrange(p.len, *LT["vsync"]):
                    # len(validpulses)-1 before appending adds index to first VSYNC pulse
                    vsync_start = len(validpulses) - 1
                    spulse = (VSYNC, p)
                    state_length = self.rf.SysParams["numPulses"] / 2
                elif inrange(p.len, *LT["hsync"]):
                    # previous state transition was likely in error!
                    spulse = (HSYNC, p)
            elif state == VSYNC:
                if inrange(p.len, *LT["eq"]):
                    # len(validpulses)-1 before appending adds index to first EQ pulse
                    vsyncs.append((vsync_start, len(validpulses) - 1))
                    spulse = (EQPL2, p)
                    state_length = self.rf.SysParams["numPulses"] / 2
                elif inrange(p.len, *LT["vsync"]):
                    spulse = (VSYNC, p)
                elif p.start > state_end and inrange(p.len, *LT["hsync"]):
                    spulse = (HSYNC, p)
            elif state == EQPL2:
                if inrange(p.len, *LT["eq"]):
                    spulse = (EQPL2, p)
                elif inrange(p.len, *LT["hsync"]):
                    spulse = (HSYNC, p)
                    done = True

            if spulse is not None and spulse[0] != state:
                if spulse[1].start < state_end:
                    spulse = None
                elif state_length:
                    state_end = spulse[1].start + (
                        (state_length - 0.1) * self.inlinelen
                    )
                    state_length = None

            # Quality check
            if spulse is not None:
                good = (
                    self.pulse_qualitycheck(validpulses[-1], spulse)
                    if len(validpulses)
                    else False
                )

                validpulses.append((spulse[0], spulse[1], good))

            if done:
                return done, validpulses

        return done, validpulses

    @profile
    def refinepulses(self):
        self.LT = self.get_timings()

        HSYNC, EQPL1, VSYNC, EQPL2 = range(4)

        i = 0
        valid_pulses = []
        num_vblanks = 0

        while i < len(self.rawpulses):
            curpulse = self.rawpulses[i]
            if inrange(curpulse.len, *self.LT["hsync"]):
                good = (
                    self.pulse_qualitycheck(valid_pulses[-1], (0, curpulse))
                    if len(valid_pulses)
                    else False
                )
                valid_pulses.append((HSYNC, curpulse, good))
                i += 1
            elif (
                i > 2
                and inrange(self.rawpulses[i].len, *self.LT["eq"])
                and (len(valid_pulses) and valid_pulses[-1][0] == HSYNC)
            ):
                # print(i, self.rawpulses[i])
                done, vblank_pulses = self.run_vblank_state_machine(
                    self.rawpulses[i - 2 : i + 24], self.LT
                )
                if done:
                    [valid_pulses.append(p) for p in vblank_pulses[2:]]
                    i += len(vblank_pulses) - 2
                    num_vblanks += 1
                else:
                    i += 1
            else:
                i += 1

        return valid_pulses

    #@profile
    def getBlankRange(self, validpulses, start=0):
        vp_type = np.array([p[0] for p in validpulses])

        vp_vsyncs = np.where(vp_type[start:] == VSYNC)[0]
        firstvsync = vp_vsyncs[0] + start if len(vp_vsyncs) else None

        if firstvsync is None or firstvsync < 10:
            return None, None

        for newstart in range(firstvsync - 10, firstvsync - 4):
            blank_locs = np.where(vp_type[newstart:] > 0)[0]
            if len(blank_locs) == 0:
                continue

            firstblank = blank_locs[0] + newstart
            hsync_locs = np.where(vp_type[firstblank:] == 0)[0]

            if len(hsync_locs) == 0:
                continue

            lastblank = hsync_locs[0] + firstblank - 1

            if (lastblank - firstblank) > 12:
                return firstblank, lastblank

        # there isn't a valid range to find, or it's impossibly short
        return None, None

    def getBlankLength(self, isFirstField):
        core = self.rf.SysParams["numPulses"] * 3 * 0.5

        if self.rf.system == "NTSC":
            return core + 1
        else:
            return core + 0.5 + (0 if isFirstField else 1)

    def processVBlank(self, validpulses, start, limit=None):

        firstblank, lastblank = self.getBlankRange(validpulses, start)

        """
        First Look at each equalization/vblank pulse section - if the expected # are there and valid,
        it can be used to determine where line 0 is...
        """

        # locations of lines before after/vblank.  may not be line 0 etc
        lastvalid = len(validpulses) if limit is None else start + limit
        if firstblank is None or firstblank > lastvalid:
            return None, None, None, None

        loc_presync = validpulses[firstblank - 1][1].start

        HSYNC, EQPL1, VSYNC, EQPL2 = range(4)

        pt = np.array([v[0] for v in validpulses[firstblank:]])
        pstart = np.array([v[1].start for v in validpulses[firstblank:]])
        plen = np.array([v[1].len for v in validpulses[firstblank:]])

        numPulses = self.rf.SysParams["numPulses"]

        for i in [VSYNC, EQPL1, EQPL2]:
            ptmatch = pt == i
            grouploc = None

            for j in range(0, lastblank - firstblank):
                if ptmatch[j : j + numPulses].all():
                    if ptmatch[j : j + numPulses + 4].sum() != numPulses:
                        break

                    # take the (second) derivative of the line gaps and lengths to determine
                    # if all are valid
                    gaps = np.diff(np.diff(pstart[j : j + numPulses]))
                    lengths = np.diff(plen[j : j + numPulses])

                    if np.max(gaps) < (self.rf.freq * 0.2) and np.max(lengths) < (
                        self.rf.freq * 0.2
                    ):
                        grouploc = j
                        break

            if grouploc is None:
                continue

            setbegin = validpulses[firstblank + grouploc]
            firstloc = setbegin[1].start

            # compute the distance of the first pulse of this block to line 1
            # (line 0 may be .5H or 1H before that)
            distfroml1 = ((i - 1) * self.rf.SysParams["numPulses"]) * 0.5

            dist = (firstloc - loc_presync) / self.inlinelen
            # get the integer rounded X * .5H distance.  then invert to determine
            # the half-H alignment with the sync/blank pulses
            hdist = nb_round(dist * 2)

            # isfirstfield = not ((hdist % 2) == self.rf.SysParams['firstField1H'][0])
            isfirstfield = (hdist % 2) == (self.rf.SysParams["firstFieldH"][1] != 1)

            # for PAL VSYNC, the offset is 2.5H, so the calculation must be reversed
            if (distfroml1 * 2) % 2:
                isfirstfield = not isfirstfield

            eqgap = self.rf.SysParams["firstFieldH"][isfirstfield]
            line0 = firstloc - ((eqgap + distfroml1) * self.inlinelen)

            return int(line0), isfirstfield, firstblank, 100

        """
        If there are no valid sections, check line 0 and the first eq pulse, and the last eq
        pulse and the following line.  If the combined xH is correct for the standard in question
        (1.5H for NTSC, 1 or 2H for PAL, that means line 0 has been found correctly.
        """

        if (
            validpulses[firstblank - 1][2]
            and validpulses[firstblank][2]
            and validpulses[lastblank][2]
            and validpulses[lastblank + 1][2]
        ):
            gap1 = (
                validpulses[firstblank][1].start - validpulses[firstblank - 1][1].start
            )
            gap2 = validpulses[lastblank + 1][1].start - validpulses[lastblank][1].start

            if self.rf.system == "PAL" and inrange(
                np.abs(gap2 - gap1), 0, self.rf.freq * 1
            ):
                isfirstfield = inrange((gap1 / self.inlinelen), 0.45, 0.55)
            elif self.rf.system == "NTSC" and inrange(
                np.abs(gap2 + gap1), self.inlinelen * 1.4, self.inlinelen * 1.6
            ):
                isfirstfield = inrange((gap1 / self.inlinelen), 0.95, 1.05)
            else:
                self.sync_confidence = 0
                return None, None, None, 0

            return validpulses[firstblank - 1][1].start, isfirstfield, firstblank, 50

        return None, None, None, 0

    #@profile
    def computeLineLen(self, validpulses):
        # determine longest run of 0's
        longrun = [-1, -1]
        currun = None
        for i, v in enumerate([p[0] for p in validpulses]):
            if v != 0:
                if currun is not None and currun[1] > longrun[1]:
                    longrun = currun
                currun = None
            elif currun is None:
                currun = [i, 0]
            else:
                currun[1] += 1

        if currun is not None and currun[1] > longrun[1]:
            longrun = currun

        linelens = []
        for i in range(longrun[0] + 1, longrun[0] + longrun[1]):
            linelen = validpulses[i][1].start - validpulses[i - 1][1].start
            if inrange(linelen / self.inlinelen, 0.95, 1.05):
                linelens.append(
                    validpulses[i][1].start - validpulses[i - 1][1].start
                )

        if len(linelens) > 0:
            return np.mean(linelens)
        else:
            return self.inlinelen


    def skip_check(self):
        """ This routine checks to see if there's a (probable) VSYNC at the end.
            Returns a (currently rough) probability.
        """
        score = 0
        vsync_lines = 0

        vsync_ire = self.rf.DecoderParams["vsync_ire"]

        for l in range(self.outlinecount, self.outlinecount + 8):
            sl = self.lineslice(l, 0, self.rf.SysParams["line_period"])
            line_ire = self.rf.hztoire(nb_median(self.data["video"]["demod"][sl]))

            # vsync_ire is always negative, so /2 is the higher number

            if inrange(line_ire, vsync_ire - 10, vsync_ire / 2):
                vsync_lines += 1
            elif inrange(line_ire, -5, 5):
                score += 1
            else:
                score -= 1

        if vsync_lines >= 2:
            return 100
        elif vsync_lines == 1 and score > 0:
            return 50
        elif score > 0:
            return 25

        return 0

    # pull the above together into a routine that (should) find line 0, the last line of
    # the previous field.

    #@profile
    def getLine0(self, validpulses, meanlinelen):
        # Gather the local line 0 location and projected from the previous field

        self.sync_confidence = 100

        # If we have a previous field, the first vblank should be close to the beginning,
        # and we need to reject anything too far in (which could be the *next* vsync)
        limit = None
        limit = (
            100
            if (self.prevfield is not None and self.prevfield.skip_check() >= 50)
            else None
        )
        line0loc_local, isFirstField_local, firstblank_local, conf_local = self.processVBlank(
            validpulses, 0, limit
        )

        line0loc_next, isFirstField_next, conf_next = None, None, None

        # If we have a vsync at the end, use it to compute the likely line 0
        if line0loc_local is not None:
            self.vblank_next, isNotFirstField_next, firstblank_next, conf_next = self.processVBlank(
                validpulses, firstblank_local + 40
            )

            if self.vblank_next is not None:
                isFirstField_next = not isNotFirstField_next

                fieldlen = (
                    meanlinelen
                    * self.rf.SysParams["field_lines"][0 if isFirstField_next else 1]
                )
                line0loc_next = nb_round(self.vblank_next - fieldlen)

                if line0loc_next < 0:
                    self.sync_confidence = 10
        else:
            self.vblank_next = None

        # Use the previous field's end to compute a possible line 0
        line0loc_prev, isFirstField_prev = None, None
        if self.prevfield is not None and self.prevfield.valid:
            frameoffset = self.data["startloc"] - self.prevfield.data["startloc"]

            # print(self.prevfield.linecount)

            line0loc_prev = (
                self.prevfield.linelocs[self.prevfield.linecount] - frameoffset
            )
            isFirstField_prev = not self.prevfield.isFirstField
            conf_prev = self.prevfield.sync_confidence

        # print(line0loc_local, line0loc_next, line0loc_prev)

        # Best case - all three line detectors returned something - perform TOOT using median
        if (
            line0loc_local is not None
            and line0loc_next is not None
            and line0loc_prev is not None
        ):
            isFirstField_all = (
                isFirstField_local + isFirstField_prev + isFirstField_next
            ) >= 2
            return (
                np.median([line0loc_local, line0loc_next, line0loc_prev]),
                self.vblank_next,
                isFirstField_all,
            )

        if line0loc_local is not None and conf_local > 50:
            self.sync_confidence = min(self.sync_confidence, 90)
            return line0loc_local, self.vblank_next, isFirstField_local
        elif line0loc_prev is not None:
            new_sync_confidence = np.max(conf_prev - 10, 0)
            self.sync_confidence = min(self.sync_confidence, new_sync_confidence)
            return line0loc_prev, self.vblank_next, isFirstField_prev
        elif line0loc_next is not None:
            self.sync_confidence = conf_next
            return line0loc_next, self.vblank_next, isFirstField_next
        else:
            # Failed to find anything useful - the caller is expected to skip ahead and try again
            return None, None, None

    def getpulses(self):
        # pass one using standard levels

        # pulse_hz range:  vsync_ire - 10, maximum is the 50% crossing point to sync
        pulse_hz_min = self.rf.iretohz(self.rf.DecoderParams["vsync_ire"] - 10)
        pulse_hz_max = self.rf.iretohz(self.rf.DecoderParams["vsync_ire"] / 2)

        pulse_hz_min = self.rf.iretohz(self.rf.DecoderParams["vsync_ire"] - 20)
        pulse_hz_max = self.rf.iretohz(-20)

        pulses = findpulses(self.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max)

        if len(pulses) == 0:
            # can't do anything about this
            return pulses

        # determine sync pulses from vsync
        vsync_locs = []
        vsync_means = []

        minlength = self.usectoinpx(10)

        for i, p in enumerate(pulses):
            if p.len > minlength:
                vsync_locs.append(i)
                vsync_means.append(
                    np.mean(
                        self.data["video"]["demod_05"][
                            int(p.start + self.rf.freq) : int(
                                p.start + p.len - self.rf.freq
                            )
                        ]
                    )
                )

        # print(len(vsync_means), [self.rf.hztoire(v) for v in vsync_means])
        if len(vsync_means) == 0:
            return None

        synclevel = np.median(vsync_means)

        if np.abs(self.rf.hztoire(synclevel) - self.rf.DecoderParams["vsync_ire"]) < 5:
            # sync level is close enough to use
            return pulses

        if vsync_locs is None or not len(vsync_locs):
            return None

        # Now compute black level and try again

        # take the eq pulses before and after vsync
        r1 = range(vsync_locs[0] - 5, vsync_locs[0])
        r2 = range(vsync_locs[-1] + 1, vsync_locs[-1] + 6)

        black_means = []

        for i in itertools.chain(r1, r2):
            if i < 0 or i >= len(pulses):
                continue

            p = pulses[i]
            if inrange(p.len, self.rf.freq * 0.75, self.rf.freq * 2.5):
                black_means.append(
                    np.mean(
                        self.data["video"]["demod_05"][
                            int(p.start + (self.rf.freq * 5)) : int(
                                p.start + (self.rf.freq * 20)
                            )
                        ]
                    )
                )

        blacklevel = np.median(black_means)

        pulse_hz_min = synclevel - (self.rf.DecoderParams["hz_ire"] * 10)
        pulse_hz_max = (blacklevel + synclevel) / 2

        return findpulses(self.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max)

    #@profile
    def compute_linelocs(self):

        self.rawpulses = self.getpulses()
        if self.rawpulses is None or len(self.rawpulses) == 0:
            if self.fields_written:
                logger.error("Unable to find any sync pulses, skipping one field")
                return None, None, None
            else:
                logger.error("Unable to find any sync pulses, skipping one second")
                return None, None, int(self.rf.freq_hz)


        self.validpulses = validpulses = self.refinepulses()
        meanlinelen = self.computeLineLen(validpulses)
        line0loc, lastlineloc, self.isFirstField = self.getLine0(validpulses, meanlinelen)
        self.linecount = 263 if self.isFirstField else 262

        # Number of lines to actually process.  This is set so that the entire following
        # VSYNC is processed
        proclines = self.outlinecount + self.lineoffset + 10
        if self.rf.system == "PAL":
            proclines += 3

        # It's possible for getLine0 to return None for lastlineloc
        if lastlineloc is not None:
            numlines = (lastlineloc - line0loc) / self.inlinelen
            self.skipdetected = numlines < (self.linecount - 5)
        else:
            self.skipdetected = False

        linelocs_dict = {}
        linelocs_dist = {}

        if line0loc is None:
            if self.initphase is False:
                logger.error("Unable to determine start of field - dropping field")

            return None, None, self.inlinelen * 200

        # If we don't have enough data at the end, move onto the next field
        lastline = (self.rawpulses[-1].start - line0loc) / meanlinelen
        if lastline < proclines:
            return None, None, line0loc - (meanlinelen * 20)

        for p in validpulses:
            lineloc = (p[1].start - line0loc) / meanlinelen
            rlineloc = nb_round(lineloc)
            lineloc_distance = np.abs(lineloc - rlineloc)

            if self.skipdetected:
                lineloc_end = self.linecount - (
                    (lastlineloc - p[1].start) / meanlinelen
                )
                rlineloc_end = nb_round(lineloc_end)
                lineloc_end_distance = np.abs(lineloc_end - rlineloc_end)

                if (
                    p[0] == 0
                    and rlineloc > 23
                    and lineloc_end_distance < lineloc_distance
                ):
                    lineloc = lineloc_end
                    rlineloc = rlineloc_end
                    lineloc_distance = lineloc_end_distance

            # only record if it's closer to the (probable) beginning of the line
            if lineloc_distance > self.rf.hsync_tolerance or (
                rlineloc in linelocs_dict and lineloc_distance > linelocs_dist[rlineloc]
            ):
                # print(rlineloc, p, 'reject')
                continue

            # also skip non-regular lines (non-hsync) that don't seem to be in valid order (p[2])
            # (or hsync lines in the vblank area)
            if rlineloc > 0 and not p[2]:
                if p[0] > 0 or (p[0] == 0 and rlineloc < 10):
                    continue

            linelocs_dict[rlineloc] = p[1].start
            linelocs_dist[rlineloc] = lineloc_distance

        rv_err = np.full(proclines, False)

        # Convert dictionary into list, then fill in gaps
        linelocs = [
            linelocs_dict[l] if l in linelocs_dict else -1 for l in range(0, proclines)
        ]
        linelocs_filled = linelocs.copy()

        self.linelocs0 = linelocs.copy()

        if linelocs_filled[0] < 0:
            next_valid = None
            for i in range(0, self.outlinecount + 1):
                if linelocs[i] > 0:
                    next_valid = i
                    break

            if next_valid is None:
                return None, None, line0loc + (self.inlinelen * self.outlinecount - 7)

            linelocs_filled[0] = linelocs_filled[next_valid] - (
                next_valid * meanlinelen
            )

            if linelocs_filled[0] < self.inlinelen:
                return None, None, line0loc + (self.inlinelen * self.outlinecount - 7)

        for l in range(1, proclines):
            if linelocs_filled[l] < 0:
                rv_err[l] = True

                prev_valid = None
                next_valid = None

                for i in range(l, -1, -1):
                    if linelocs[i] > 0:
                        prev_valid = i
                        break
                for i in range(l, self.outlinecount + 1):
                    if linelocs[i] > 0:
                        next_valid = i
                        break

                # print(l, prev_valid, next_valid)

                if prev_valid is None:
                    avglen = self.inlinelen
                    linelocs_filled[l] = linelocs[next_valid] - (
                        avglen * (next_valid - l)
                    )
                elif next_valid is not None:
                    avglen = (linelocs[next_valid] - linelocs[prev_valid]) / (
                        next_valid - prev_valid
                    )
                    linelocs_filled[l] = linelocs[prev_valid] + (
                        avglen * (l - prev_valid)
                    )
                else:
                    avglen = self.inlinelen
                    linelocs_filled[l] = linelocs[prev_valid] + (
                        avglen * (l - prev_valid)
                    )

        # *finally* done :)

        rv_ll = [linelocs_filled[l] for l in range(0, proclines)]

        # print(self.vblank_next)
        if self.vblank_next is None:
            nextfield = linelocs_filled[self.outlinecount - 7]
        else:
            nextfield = self.vblank_next - (self.inlinelen * 8)

        # print('nf', nextfield, self.vblank_next)

        return rv_ll, rv_err, nextfield

    #@profile
    def refine_linelocs_hsync(self):
        linelocs2 = self.linelocs1.copy()

        for i in range(len(self.linelocs1)):
            # skip VSYNC lines, since they handle the pulses differently
            if inrange(i, 3, 6) or (self.rf.system == "PAL" and inrange(i, 1, 2)):
                self.linebad[i] = True
                continue

            # refine beginning of hsync
            ll1 = self.linelocs1[i] - self.rf.freq
            zc = calczc(
                self.data["video"]["demod_05"],
                ll1,
                self.rf.iretohz(self.rf.DecoderParams["vsync_ire"] / 2),
                reverse=False,
                count=self.rf.freq * 2,
            )

            if zc is not None and not self.linebad[i]:
                linelocs2[i] = zc

                # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
                hsync_area = self.data["video"]["demod_05"][
                    int(zc - (self.rf.freq * 0.75)) : int(zc + (self.rf.freq * 8))
                ]
                if nb_min(hsync_area) < self.rf.iretohz(-55) or nb_max(
                    hsync_area
                ) > self.rf.iretohz(30):
                    # don't use the computed value here if it's bad
                    self.linebad[i] = True
                    linelocs2[i] = self.linelocs1[i]
                else:
                    porch_level = nb_median(
                        self.data["video"]["demod_05"][
                            int(zc + (self.rf.freq * 8)) : int(zc + (self.rf.freq * 9))
                        ]
                    )
                    sync_level = nb_median(
                        self.data["video"]["demod_05"][
                            int(zc + (self.rf.freq * 1)) : int(
                                zc + (self.rf.freq * 2.5)
                            )
                        ]
                    )

                    zc2 = calczc(
                        self.data["video"]["demod_05"],
                        ll1,
                        (porch_level + sync_level) / 2,
                        reverse=False,
                        count=400,
                    )

                    # any wild variation here indicates a failure
                    if zc2 is not None and np.abs(zc2 - zc) < (self.rf.freq / 2):
                        linelocs2[i] = zc2
                    else:
                        self.linebad[i] = True
            else:
                self.linebad[i] = True

            if self.linebad[i]:
                linelocs2[i] = self.linelocs1[
                    i
                ]  # don't use the computed value here if it's bad

        return linelocs2

    def compute_deriv_error(self, linelocs, baserr):
        """ compute errors based off the second derivative - if it exceeds 1 something's wrong,
            and if 4 really wrong...
        """

        derr1 = np.full(len(linelocs), False)
        derr1[1:-1] = np.abs(np.diff(np.diff(linelocs))) > 4

        derr2 = np.full(len(linelocs), False)
        derr2[2:] = np.abs(np.diff(np.diff(linelocs))) > 4

        return baserr | derr1 | derr2

    def fix_badlines(self, linelocs_in, linelocs_backup_in=None):
        self.linebad = self.compute_deriv_error(linelocs_in, self.linebad)
        linelocs = np.array(linelocs_in.copy())

        if linelocs_backup_in is not None:
            linelocs_backup = np.array(linelocs_backup_in.copy())
            badlines = np.isnan(linelocs)
            linelocs[badlines] = linelocs_backup[badlines]

        for l in np.where(self.linebad)[0]:
            prevgood = l - 1
            nextgood = l + 1

            while prevgood >= 0 and self.linebad[prevgood]:
                prevgood -= 1

            while nextgood < len(linelocs) and self.linebad[nextgood]:
                nextgood += 1

            firstcheck = 0 if self.rf.system == "PAL" else 1
            if prevgood >= firstcheck and nextgood < (len(linelocs) + self.lineoffset):
                gap = (linelocs[nextgood] - linelocs[prevgood]) / (nextgood - prevgood)
                linelocs[l] = (gap * (l - prevgood)) + linelocs[prevgood]

        return linelocs

    def computewow(self, lineinfo):
        wow = np.ones(len(lineinfo))

        for l in range(0, len(wow) - 1):
            wow[l] = self.get_linelen(l) / self.inlinelen

        for l in range(self.lineoffset, self.lineoffset + 10):
            wow[l] = np.median(wow[l : l + 4])

        return wow

    #@profile
    def downscale(
        self,
        lineinfo=None,
        linesout=None,
        outwidth=None,
        channel="demod",
        audio=0,
        final=False,
        lastfieldwritten=None,
    ):
        if lineinfo is None:
            lineinfo = self.linelocs
        if outwidth is None:
            outwidth = self.outlinelen
        if linesout is None:
            # for video always output 263/313 lines
            linesout = self.outlinecount

        if lastfieldwritten and audio > 16000:
            # Compute field # and line count

            rf_samples_per_field = self.rf.freq_hz / self.rf.SysParams['FPS'] / 2
            read_gap = (self.readloc - lastfieldwritten[1]) / rf_samples_per_field
            field_number = nb_round(lastfieldwritten[0] + read_gap)

            linecount = sum(self.rf.SysParams["field_lines"]) * (field_number // 2)
            if not self.isFirstField:
                linecount += self.rf.SysParams["field_lines"][0]

            # Now compute the # of audio samples that should be written, and then the 
            # location of that relative to the current line
            samples_per_line = (self.rf.SysParams['line_period'] / 1000000) / (1 / audio)

            audsamp_count = linecount * samples_per_line
            audsamp_offset = (np.floor(audsamp_count) + 1) - audsamp_count

            # Finally convert to a time value
            audio_offset = -audsamp_offset * (self.rf.SysParams['line_period'] / 10000000)

        else:
            audio_offset = 0

        audio_thread = None
        if audio != 0 and self.rf.decode_analog_audio:
            audio_rv = {}
            audio_thread = threading.Thread(target=downscale_audio, args=(
                self.data["audio"],
                lineinfo,
                self.rf,
                self.linecount,
                audio_offset,
                audio,
                audio_rv)
            )
            audio_thread.start()

        dsout = np.zeros((linesout * outwidth), dtype=np.double)
        # self.lineoffset is an adjustment for 0-based lines *before* downscaling so add 1 here
        lineoffset = self.lineoffset + 1

        #print(lineinfo[linesout] - lineinfo[1])

        for l in range(lineoffset, linesout + lineoffset):
            if lineinfo[l + 1] > lineinfo[l]:
                scaled = scale(
                    self.data["video"][channel],
                    lineinfo[l],
                    lineinfo[l + 1],
                    outwidth,
                    self.wowfactor[l],
                )

                dsout[
                    (l - lineoffset) * outwidth : (l + 1 - lineoffset) * outwidth
                ] = scaled
            else:
                # Massive TBC error detected
                self.sync_confidence = 1
                #logger.warning("WARNING: TBC failure at line %d", l)
                dsout[
                    (l - lineoffset) * outwidth : (l + 1 - lineoffset) * outwidth
                ] = self.rf.DecoderParams["ire0"]

        if self.rf.decode_digital_audio:
            self.efmout = self.data["efm"][
                int(self.linelocs[1]) : int(self.linelocs[self.linecount + 1])
            ]
        else:
            self.efmout = None

        if final:
            dsout = self.hz_to_output(dsout)
            self.dspicture = dsout

        if audio_thread:
            audio_thread.join()
            self.dsaudio = audio_rv["dsaudio"]
            self.audio_next_offset = audio_rv["audio_next_offset"]

        return dsout, self.dsaudio, self.efmout

    @profile
    def rf_tbc(self, linelocs=None):
        """ This outputs a TBC'd version of the input RF data, mostly intended
            to assist in audio processing.  Outputs a uint16 array.
        """

        # Convert raw RF to floating point to help the scaler
        fdata = self.data["input"].astype(float)

        if linelocs is None:
            linelocs = self.linelocs

        # Ensure that the output line length is an integer
        linelen = int(round(self.inlinelen))

        # Adjust for the demodulation/filtering delays
        delay = self.rf.delays["video_white"]

        # On PAL, always ignore self.lineoffset
        startline = self.lineoffset if self.rf.system == "NTSC" else 1
        endline = startline + self.linecount

        output = []

        for l in range(startline, endline):
            scaled = scale(fdata, linelocs[l] - delay, linelocs[l + 1] - delay, linelen)
            output.append(np.round(scaled).astype(np.int16))

        return np.concatenate(output)

    def decodephillipscode(self, linenum):
        linestart = self.linelocs[linenum]
        data = self.data["video"]["demod"]
        curzc = calczc(
            data,
            int(linestart + self.usectoinpx(2)),
            self.rf.iretohz(50),
            count=int(self.usectoinpx(12)),
        )

        zc = []
        while curzc is not None:
            zc.append(
                (curzc, data[int(curzc - self.usectoinpx(0.5))] < self.rf.iretohz(50))
            )
            curzc = calczc(
                data,
                curzc + self.usectoinpx(1.9),
                self.rf.iretohz(50),
                count=int(self.usectoinpx(0.2)),
            )

        usecgap = self.inpxtousec(np.diff([z[0] for z in zc]))
        valid = len(zc) == 24 and np.min(usecgap) > 1.85 and np.max(usecgap) < 2.15

        if valid:
            bitset = [z[1] for z in zc]
            linecode = 0

            for b in range(0, 24, 4):
                linecode *= 0x10
                linecode += (np.packbits(bitset[b : b + 4]) >> 4)[0]

            return linecode

        return None

    def compute_syncconf(self):
        """ use final lineloc data to compute sync confidence """

        newconf = 100

        lld = np.diff(self.linelocs[self.lineoffset : self.lineoffset + self.linecount])
        lld2 = np.diff(lld)
        lld2max = np.max(lld2)

        if lld2max > 4:
            newconf = int(50 - (5 * np.sum(lld2max > 4)))

        newconf = max(newconf, 0)

        self.sync_confidence = min(self.sync_confidence, newconf)
        return int(self.sync_confidence)

    def get_vsync_area(self):
        """ return beginning, length in lines, and end of vsync area """
        vsync_begin = int(self.linelocs[0])
        vsync_end_line = int(self.getBlankLength(self.isFirstField) + 0.6)
        vsync_end = int(self.linelocs[vsync_end_line]) + 1

        return vsync_begin, vsync_end_line, vsync_end

    def get_vsync_lines(self):
        rv = []
        end = 10 if self.isFirstField else 9
        for i in range(1, end):
            rv.append(i)

        if self.rf.system == 'PAL':
            start2 = 311 if self.isFirstField else 310
            for i in range(start2, 318):
                rv.append(i)

        return rv

    @profile
    def dropout_detect_demod(self):
        # current field
        f = self

        isPAL = self.rf.system == "PAL"

        rfstd = nb_std(f.data["rfhpf"])
        # iserr_rf = np.full(len(f.data['video']['demod']), False, dtype=np.bool)
        iserr_rf1 = (f.data["rfhpf"] < (-rfstd * 3)) | (
            f.data["rfhpf"] > (rfstd * 3)
        )  # | (f.rawdata <= -32000)
        iserr = np.full_like(iserr_rf1, False)
        iserr[self.rf.delays["video_rot"] :] = iserr_rf1[
            : -self.rf.delays["video_rot"]
        ]

        # build sets of min/max valid levels
        valid_min = np.full_like(
            f.data["video"]["demod"], f.rf.iretohz(-70 if isPAL else -50)
        )
        valid_max = np.full_like(
            f.data["video"]["demod"], f.rf.iretohz(150 if isPAL else 160)
        )

        # Look for slightly longer dropouts...
        valid_min05 = np.full_like(f.data["video"]["demod_05"], f.rf.iretohz(-30))
        valid_max05 = np.full_like(f.data["video"]["demod_05"], f.rf.iretohz(115))

        # Account for sync pulses when checking demod

        hsync_len = int(f.LT['hsync'][1])
        vsync_ire = f.rf.SysParams['vsync_ire']
        vsync_lines = self.get_vsync_lines()

        # In sync areas the minimum IRE is vsync - pilot/burst
        sync_min = f.rf.iretohz(vsync_ire - 60 if isPAL else vsync_ire - 35)
        sync_min_05 = f.rf.iretohz(vsync_ire - 10)

        for l in range(1, len(f.linelocs)):
            if l in vsync_lines:
                valid_min[int(f.linelocs[l]):int(f.linelocs[l+1])] = sync_min
                valid_min05[int(f.linelocs[l]):int(f.linelocs[l+1])] = sync_min_05
            else:
                valid_min[int(f.linelocs[l]):int(f.linelocs[l]) + hsync_len] = sync_min
                valid_min05[int(f.linelocs[l]):int(f.linelocs[l]) + hsync_len] = sync_min_05

        # detect absurd fluctuations in pre-deemp demod, since only dropouts can cause them
        # (current np.diff has a prepend option, but not in ubuntu 18.04's version)
        n_orgt(iserr, f.data["video"]["demod_raw"], self.rf.freq_hz_half)

        n_ornotrange(iserr, f.data["video"]["demod"], valid_min, valid_max)
        n_ornotrange(iserr, f.data["video"]["demod_05"], valid_min05, valid_max05)

        # filter out dropouts outside actual field
        iserr[:int(f.linelocs[f.lineoffset + 1])] = False
        iserr[int(f.linelocs[f.lineoffset + f.linecount + 1]):] = False

        #print(iserr1.sum(), iserr2.sum(), iserr3.sum(), iserr_rf.sum(), iserr.sum())

        return iserr

    @profile
    def build_errlist(self, errmap):
        errlist = []

        firsterr = errmap[np.nonzero(errmap >= self.linelocs[self.lineoffset])[0][0]]
        curerr = (firsterr, firsterr)

        for e in errmap:
            if e > curerr[0] and e <= (curerr[1] + 20):
                pad = ((e - curerr[0])) * 1.7
                pad = min(pad, self.rf.freq * 12)
                epad = curerr[0] + pad
                curerr = (curerr[0], epad)
            elif e > firsterr:
                errlist.append((curerr[0] - 8, curerr[1] + 4))
                curerr = (e, e)

        errlist.append(curerr)

        return errlist

    @profile
    def dropout_errlist_to_tbc(field, errlist):
        """Convert data from raw data coordinates to tbc coordinates, and splits up
        multi-line dropouts.
        """
        dropouts = []

        if len(errlist) == 0:
            return dropouts

        # Now convert the above errlist into TBC locations
        errlistc = errlist.copy()
        lineoffset = -field.lineoffset

        # Remove dropouts occuring before the start of the frame so they don't
        # cause the rest to be skipped
        curerr = errlistc.pop(0)
        while len(errlistc) > 0 and curerr[0] < field.linelocs[field.lineoffset]:
            curerr = errlistc.pop(0)

        # TODO: This could be reworked to be a bit cleaner and more performant.

        for line in range(field.lineoffset, field.linecount + field.lineoffset):
            while curerr is not None and inrange(
                curerr[0], field.linelocs[line], field.linelocs[line + 1]
            ):
                start_rf_linepos = curerr[0] - field.linelocs[line]
                start_linepos = start_rf_linepos / (
                    field.linelocs[line + 1] - field.linelocs[line]
                )
                start_linepos = int(start_linepos * field.outlinelen)

                end_rf_linepos = curerr[1] - field.linelocs[line]
                end_linepos = end_rf_linepos / (
                    field.linelocs[line + 1] - field.linelocs[line]
                )
                end_linepos = nb_round(end_linepos * field.outlinelen)

                first_line = line + 1 + lineoffset

                # If the dropout spans multiple lines, we need to split it up into one for each line.
                if end_linepos > field.outlinelen:
                    num_lines = end_linepos // field.outlinelen

                    # First line.
                    dropouts.append((first_line, start_linepos, field.outlinelen))
                    # Full lines in the middle.
                    for n in range(num_lines - 1):
                        dropouts.append((first_line + n + 1, 0, field.outlinelen))
                    # leftover on last line.
                    dropouts.append(
                        (
                            first_line + (num_lines),
                            0,
                            np.remainder(end_linepos, field.outlinelen),
                        )
                    )
                else:
                    dropouts.append((first_line, start_linepos, end_linepos))

                if len(errlistc):
                    curerr = errlistc.pop(0)
                else:
                    curerr = None

        return dropouts

    @profile
    def dropout_detect(self):
        """ returns dropouts in three arrays, to line up with the JSON output """

        rv_lines = []
        rv_starts = []
        rv_ends = []

        iserr = self.dropout_detect_demod()
        errmap = np.nonzero(iserr)[0]

        if len(errmap) > 0 and errmap[-1] > self.linelocs[self.lineoffset]:
            errlist = self.build_errlist(errmap)

            for r in self.dropout_errlist_to_tbc(errlist):
                rv_lines.append(r[0] - 1)
                rv_starts.append(int(r[1]))
                rv_ends.append(int(r[2]))

        return rv_lines, rv_starts, rv_ends

    @profile
    def compute_line_bursts(self, linelocs, _line, prev_phaseadjust=0):
        line = _line + self.lineoffset
        # calczc works from integers, so get the start and remainder
        s = int(linelocs[line])
        s_rem = linelocs[line] - s

        lfreq = self.get_linefreq(line)

        fsc_mhz_inv = 1 / self.rf.SysParams["fsc_mhz"]

        # compute approximate burst beginning/end
        bstime = 25 * fsc_mhz_inv  # approx start of burst in usecs

        bstart = int(bstime * lfreq)
        bend = int(8.8 * lfreq)

        # copy and get the mean of the burst area to factor out wow/flutter
        burstarea = self.data["video"]["demod_burst"][s + bstart : s + bend]
        if len(burstarea) == 0:
            return None, None

        burstarea = burstarea - nb_mean(burstarea)
        threshold = 5 * self.rf.DecoderParams["hz_ire"]

        burstarea_demod = self.data["video"]["demod"][s + bstart : s + bend]
        burstarea_demod = burstarea_demod - nb_mean(burstarea_demod)

        if nb_absmax(burstarea_demod) > (30 * self.rf.DecoderParams["hz_ire"]):
            return None, None

        zcburstdiv = (lfreq * fsc_mhz_inv) / 2

        # Apply phase adjustment from previous frame/line if available.
        phase_adjust = -prev_phaseadjust

        # a proper color burst should have ~12-13 zero crossings
        isrising = np.zeros(16, dtype=np.bool_)
        zcs = np.zeros(16, dtype=np.float32)

        # The first pass computes phase_offset, the second uses it to determine
        # the colo(u)r burst phase of the line.
        for passcount in range(2):
            # this subroutine is in utils.py, broken out so it can be JIT'd
            zc_count, phase_adjust, rising_count = clb_findbursts(isrising, zcs, burstarea, 0, len(burstarea) - 1, threshold, bstart, s_rem, zcburstdiv, phase_adjust)

        rising = rising_count > (zc_count / 2)

        return rising, -phase_adjust


# These classes extend Field to do PAL/NTSC specific TBC features.


class FieldPAL(Field):
    def refine_linelocs_pilot(self, linelocs=None):
        if linelocs is None:
            linelocs = self.linelocs2.copy()
        else:
            linelocs = linelocs.copy()

        plen = {}

        zcs = []
        for l in range(0, 323):
            adjfreq = self.rf.freq
            if l > 1:
                adjfreq /= (linelocs[l] - linelocs[l - 1]) / self.rf.linelen

            plen[l] = (adjfreq / self.rf.SysParams["pilot_mhz"]) / 2

            ls = self.lineslice(l, 0, 6, linelocs)
            lsoffset = linelocs[l] - ls.start

            pilots = self.data["video"]["demod_pilot"][ls]

            peakloc = np.argmax(np.abs(pilots))

            zc_base = calczc(pilots, peakloc, 0)
            if zc_base is not None:
                zc = (zc_base - lsoffset) / plen[l]
            else:
                zc = zcs[-1] if len(zcs) else 0

            zcs.append(zc)

        angles = angular_mean_helper(np.array(zcs))
        am = np.angle(np.mean(angles)) / (np.pi * 2)
        if (am < 0):
            am = 1 + am

        for l in range(0, 323):
            linelocs[l] += (phase_distance(zcs[l], am) * plen[l]) * 1

        return np.array(linelocs)

    def get_burstlevel(self, l, linelocs=None):
        lineslice = self.lineslice(l, 5.5, 2.4, linelocs)

        burstarea = self.data["video"]["demod"][lineslice].copy()
        burstarea -= nb_mean(burstarea)

        if max(burstarea) > (30 * self.rf.DecoderParams["hz_ire"]):
            return None

        return rms(burstarea) * np.sqrt(2)

    def calc_burstmedian(self):
        burstlevel = []

        for l in range(11, 313):
            lineburst = self.get_burstlevel(l)
            if lineburst is not None:
                burstlevel.append(lineburst)

        if burstlevel == []:
            return 0.0

        return np.median(burstlevel) / self.rf.DecoderParams["hz_ire"]

    def get_following_field_number(self):
        if self.prevfield is not None:
            newphase = self.prevfield.fieldPhaseID + 1
            return 1 if newphase == 9 else newphase
        else:
            # This can be triggered by the first pass at the first field
            # logger.error("Cannot determine PAL field sequence of first field")
            return 1

    def determine_field_number(self):

        """ Background
        PAL has an eight field sequence that can be split into two four field sequences.

        Field 1: First field of frame , no colour burst on line 6
        Field 2: Second field of frame, colour burst on line 6 (319)
        Field 3: First field of frame, colour burst on line 6
        Field 4: Second field of frame, no colour burst on line 6 (319)

        Fields 5-8 can be differentiated using the burst phase on line 7+4x (based off the first
        line guaranteed to have colour burst)  Ideally the rising phase would be at 0 or 180
        degrees, but since this is Laserdisc it's often quite off.  So the determination is
        based on which phase is closer to 0 degrees.
        """

        # First compute the 4-field sequence
        # This map is based in (first field, has burst on line 6)
        map4 = {(True, False): 1, (False, True): 2, (True, True): 3, (False, False): 4}

        # Determine if line 6 has valid burst - or lack of it.  If there's rot interference,
        # the burst level may be in the middle (or even None), and if so extrapolate
        # from the previous field.
        burstlevel6 = self.get_burstlevel(6)

        if burstlevel6 is None:
            return self.get_following_field_number()

        burstlevel6 /= self.rf.DecoderParams["hz_ire"]

        if inrange(burstlevel6, self.burstmedian * 0.8, self.burstmedian * 1.2):
            hasburst = True
        elif burstlevel6 < self.burstmedian * 0.2:
            hasburst = False
        else:
            return self.get_following_field_number()

        m4 = map4[(self.isFirstField, hasburst)]

        # Now compute if it's field 1-4 or 5-8.

        rcount = 0
        count = 0

        self.phase_adjust = {}

        for l in range(7, 22, 4):
            # Usually line 7 is used to determine burst phase, but
            # take the best of 5 if it's unstable
            prev_phaseadjust = 0
            try:
                # For this first field, this doesn't exist (so use a try/except/pass pattern)
                # and on a bad disk, this value could be None...
                if self.prevfield.phase_adjust[l] is not None:
                    prev_phaseadjust = self.prevfield.phase_adjust[l]
            except AttributeError:
                pass

            rising, self.phase_adjust[l] = self.compute_line_bursts(
                self.linelocs, l, prev_phaseadjust
            )

            if rising is not None:
                rcount += (rising is True)
                count += 1

        if count == 0 or (rcount * 2) == count:
            return self.get_following_field_number()

        rising = (rcount * 2) > count

        is_firstfour = rising
        if m4 == 2:
            # For field 2/6, reverse the above.
            is_firstfour = not is_firstfour

        return m4 + (0 if is_firstfour else 4)

    def downscale(self, final=False, *args, **kwargs):
        # For PAL, each field starts with the line containing the first full VSYNC pulse
        return super(FieldPAL, self).downscale(final=final, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(FieldPAL, self).__init__(*args, **kwargs)

    def process(self):
        super(FieldPAL, self).process()

        self.out_scale = np.double(0xD300 - 0x0100) / (
            100 - self.rf.DecoderParams["vsync_ire"]
        )

        if not self.valid:
            return

        self.linelocs3 = self.refine_linelocs_pilot(self.linelocs2)
        # do a second pass for fine tuning (typically < .1px), because the adjusted
        # frequency changes slightly from the first pass
        self.linelocs3a = self.refine_linelocs_pilot(self.linelocs3)
        self.linelocs = self.fix_badlines(self.linelocs3a)

        self.wowfactor = self.computewow(self.linelocs)
        self.burstmedian = self.calc_burstmedian()

        self.linecount = 312 if self.isFirstField else 313
        self.lineoffset = 2 if self.isFirstField else 3

        self.linecode = [
            self.decodephillipscode(l + self.lineoffset) for l in [16, 17, 18]
        ]

        self.fieldPhaseID = self.determine_field_number()

        # self.downscale(final=True)


# ... now for NTSC
# This class is a very basic NTSC 1D color decoder, used for alignment etc
# XXX: make this callable earlier on and/or merge into FieldNTSC?
class CombNTSC:
    """ *partial* NTSC comb filter class - only enough to do VITS calculations ATM """

    def __init__(self, field):
        self.field = field
        self.cbuffer = self.buildCBuffer()

    def getlinephase(self, line):
        """ determine if a line has positive color burst phase.
            This is based on line # and field phase ID """
        fieldID = self.field.fieldPhaseID

        if (line % 2) == 0:
            return (fieldID == 1) | (fieldID == 4)
        else:
            return (fieldID == 2) | (fieldID == 3)

    def buildCBuffer(self, subset=None):
        """
        prev_field: Compute values for previous field
        subset: a slice computed by lineslice_tbc (default: whole field)

        NOTE:  first and last two returned values will be zero, so slice accordingly
        """

        data = self.field.dspicture

        if subset:
            data = data[subset]

        # this is a translation of this code from tools/ld-chroma-decoder/comb.cpp:
        #
        # for (qint32 h = configuration.activeVideoStart; h < configuration.activeVideoEnd; h++) {
        #  double tc1 = (line[h] - ((line[h - 2] + line[h + 2]) / 2.0)) / 2.0;

        fldata = data.astype(np.float32)
        cbuffer = np.zeros_like(fldata)

        cbuffer[2:-2] = (fldata[:-4] + fldata[4:]) / 2
        cbuffer[2:-2] -= fldata[2:-2]

        return cbuffer

    def splitIQ_line(self, cbuffer, line=0):
        """
        NOTE:  currently? only works on one line

        This returns normalized I and Q arrays, each one half the length of cbuffer
        """
        linephase = self.getlinephase(line)

        sq = cbuffer[::2].copy()
        si = cbuffer[1::2].copy()

        if not linephase:
            si[0::2] = -si[0::2]
            sq[1::2] = -sq[1::2]
        else:
            si[1::2] = -si[1::2]
            sq[0::2] = -sq[0::2]

        return si, sq

    def calcLine19Info(self, comb_field2=None):
        """ returns color burst phase (ideally 147 degrees) and (unfiltered!) SNR """

        # Don't need the whole line here, but start at 0 to definitely have an even #
        l19_slice = self.field.lineslice_tbc(19, 0, 40)
        l19_slice_i70 = self.field.lineslice_tbc(19, 14, 18)

        ire_out1 = self.field.dspicture[l19_slice_i70]

        # fail out if there is obviously bad data
        if not ((np.max(ire_out1) < 100) and (np.min(ire_out1) > 40)):
            return None, None, None

        cbuffer = self.cbuffer[l19_slice]

        if comb_field2 is not None:
            ire_out2 = comb_field2.field.dspicture[l19_slice_i70]
            # fail out if there is obviously bad data
            if not ((np.max(ire_out2) < 100) and (np.min(ire_out2) > 40)):
                return None, None, None

            cbuffer = (cbuffer - comb_field2.cbuffer[l19_slice]) / 2

        si, sq = self.splitIQ_line(cbuffer, 19)

        sl = slice(110, 230)
        cdata = np.sqrt((si[sl] ** 2.0) + (sq[sl] ** 2.0))

        phase = np.arctan2(np.mean(si[sl]), np.mean(sq[sl])) * 180 / np.pi
        if phase < 0:
            phase += 360

        # compute SNR
        signal = np.mean(cdata)
        noise = np.std(cdata)

        snr = 20 * np.log10(signal / noise)

        return signal / (2 * self.field.out_scale), phase, snr


class FieldNTSC(Field):
    def get_burstlevel(self, l, linelocs=None):
        burstarea = self.data["video"]["demod"][self.lineslice(l, 5.5, 2.4, linelocs)]

        # Issue #621 - fields w/skips may be complete nonsense, so bail out if so
        try:
            return rms(burstarea) * np.sqrt(2)
        except Exception:
            # Should we warn here? (Provided this can actually occur.)
            return 0

    def compute_burst_offsets(self, linelocs):
        rising_sum = 0
        adjs = {}

        for l in range(0, 266):
            rising, phase_adjust = self.compute_line_bursts(linelocs, l)
            if rising is None:
                continue

            adjs[l] = phase_adjust / 2

            even_line = not (l % 2)
            rising_sum += 1 if (even_line and rising) else 0

        # If more than half of the lines have rising phase alignment, it's (probably) field 1 or 4
        field14 = rising_sum > (len(adjs.keys()) // 4)

        return field14, adjs

    def refine_linelocs_burst(self, linelocs=None):
        if linelocs is None:
            linelocs = self.linelocs2

        linelocs_adj = linelocs.copy()

        field14, adjs_new = self.compute_burst_offsets(linelocs_adj)

        adjs = {}

        for l in range(1, 266):
            if l not in adjs_new:
                self.linebad[l] = True

        # compute the adjustments for each line but *do not* apply, so outliers can be bypassed
        for l in range(0, 266):
            if not (np.isnan(linelocs_adj[l]) or self.linebad[l]):
                lfreq = self.get_linefreq(l, linelocs)

                try:
                    adjs[l] = adjs_new[l] * lfreq * (1 / self.rf.SysParams["fsc_mhz"])
                except Exception:
                    # Not sure if this is an error or just control flow.
                    # print("Something went wrong when trying to compute line length adjustments...", file=sys.stderr)
                    # traceback.print_exc()
                    pass

        if len(adjs.keys()):
            adjs_median = np.median([adjs[a] for a in adjs])
            lastvalid_adj = adjs_median

            for l in range(0, 266):
                if l in adjs and inrange(adjs[l] - adjs_median, -2, 2):
                    linelocs_adj[l] += adjs[l]
                    lastvalid_adj = adjs[l]
                else:
                    linelocs_adj[l] += lastvalid_adj

            # This map is based on (first field, field14)
            map4 = {
                (True, True): 1,
                (False, False): 2,
                (True, False): 3,
                (False, True): 4,
            }
            self.fieldPhaseID = map4[(self.isFirstField, field14)]
        else:
            self.fieldPhaseID = 1

        return linelocs_adj

    def downscale(self, lineoffset=0, final=False, *args, **kwargs):
        if final is False:
            if "audio" in kwargs:
                kwargs["audio"] = 0

        dsout, dsaudio, dsefm = super(FieldNTSC, self).downscale(
            final=final, *args, **kwargs
        )

        return dsout, dsaudio, dsefm

    def calc_burstmedian(self):
        burstlevel = [self.get_burstlevel(l) for l in range(11, 264)]

        return np.median(burstlevel) / self.rf.DecoderParams["hz_ire"]

    def apply_offsets(self, linelocs, phaseoffset, picoffset=0):
        # logger.info(phaseoffset, (phaseoffset * (self.rf.freq / (4 * 315 / 88))))
        return (
            np.array(linelocs)
            + picoffset
            + (phaseoffset * (self.rf.freq / (4 * 315 / 88)))
        )

    def __init__(self, *args, **kwargs):
        super(FieldNTSC, self).__init__(*args, **kwargs)

    def process(self):
        super(FieldNTSC, self).process()

        self.out_scale = np.double(0xC800 - 0x0400) / (
            100 - self.rf.DecoderParams["vsync_ire"]
        )

        if not self.valid:
            return

        self.linecode = [
            self.decodephillipscode(l + self.lineoffset) for l in [16, 17, 18]
        ]

        self.linelocs3 = self.refine_linelocs_burst(self.linelocs2)
        self.linelocs3 = self.fix_badlines(self.linelocs3, self.linelocs2)

        self.burstmedian = self.calc_burstmedian()

        # Now adjust the phase to get the downscaled image onto I/Q color axis
        shift33 = 84 * (np.pi / 180)
        self.linelocs = self.apply_offsets(self.linelocs3, -shift33 - 0)

        self.wowfactor = self.computewow(self.linelocs)


class LDdecode:
    def __init__(
        self,
        fname_in,
        fname_out,
        freader,
        _logger,
        est_frames=None,
        analog_audio=0,
        digital_audio=False,
        system="NTSC",
        doDOD=True,
        threads=4,
        inputfreq=40,
        extra_options={},
    ):
        global logger
        self.logger = _logger
        logger = self.logger
        self.demodcache = None

        self.branch, self.commit = get_git_info()
        if fname_in == '-':
            self.infile = sys.stdin
        else:
            self.infile = open(fname_in, "rb")
        self.freader = freader

        self.est_frames = est_frames

        self.numthreads = threads

        self.fields_written = 0

        self.blackIRE = 0

        self.use_profiler = extra_options.get("use_profiler", False)
        if self.use_profiler:
            from line_profiler import LineProfiler

            self.lpf = LineProfiler()
            self.lpf.add_function(Field.process)
            self.lpf.add_function(Field.compute_linelocs)
            self.lpf.add_function(Field.getpulses)

        self.analog_audio = int(analog_audio)
        self.digital_audio = digital_audio
        self.ac3 = extra_options.get("AC3", False)
        self.write_rf_tbc = extra_options.get("write_RF_TBC", False)

        self.has_analog_audio = True
        if system == "PAL":
            if analog_audio == 0:
                self.has_analog_audio = False

        self.outfile_json = None

        self.lastvalidfield = {False: None, True: None}
        self.lastFieldWritten = None

        self.outfile_video = None
        self.outfile_audio = None
        self.outfile_efm = None
        self.outfile_pre_efm = None
        self.outfile_ac3 = None
        self.ffmpeg_rftbc, self.outfile_rftbc = None, None
        self.do_rftbc = False

        if fname_out is not None:
            self.outfile_video = open(fname_out + ".tbc", "wb")
            if self.analog_audio:
                self.outfile_audio = open(fname_out + ".pcm", "wb")
            if self.digital_audio:
                # feed EFM stream into ld-ldstoefm
                self.efm_pll = efm_pll.EFM_PLL()
                self.outfile_efm = open(fname_out + ".efm", "wb")
                if extra_options.get("write_pre_efm", False):
                    self.outfile_pre_efm = open(fname_out + ".prefm", "wb")
            if self.write_rf_tbc:
                self.ffmpeg_rftbc, self.outfile_rftbc = ldf_pipe(fname_out + ".tbc.ldf")
                self.do_rftbc = True
            if self.ac3:
                self.AC3Collector = StridedCollector(cut_begin=1024, cut_end=0)
                self.ac3_processes, self.outfile_ac3 = ac3_pipe(fname_out + ".ac3")
                self.do_rftbc = True

        self.pipe_rftbc = extra_options.get("pipe_RF_TBC", None)
        if self.pipe_rftbc:
            self.do_rftbc = True

        self.fname_out = fname_out

        self.firstfield = None  # In frame output mode, the first field goes here

        self.system = system
        self.rf_opts = {
            'inputfreq':inputfreq,
            'system':system,
            'decode_analog_audio':analog_audio,
            'decode_digital_audio':digital_audio,
            'has_analog_audio':self.has_analog_audio,
            'extra_options':extra_options,
            'blocklen': 32 * 1024,
        }

        self.rf = RFDecode(**self.rf_opts)

        if system == "PAL":
            self.FieldClass = FieldPAL
            self.readlen = self.rf.linelen * 400
            self.clvfps = 25
        else:  # NTSC
            self.FieldClass = FieldNTSC
            self.readlen = ((self.rf.linelen * 350) // 16384) * 16384
            self.clvfps = 30

        self.blocksize = self.rf.blocklen

        self.output_lines = (self.rf.SysParams["frame_lines"] // 2) + 1

        self.bytes_per_frame = int(self.rf.freq_hz / self.rf.SysParams["FPS"])
        self.bytes_per_field = int(self.rf.freq_hz / (self.rf.SysParams["FPS"] * 2)) + 1
        self.outwidth = self.rf.SysParams["outlinelen"]

        self.fdoffset = 0
        self.mtf_level = 1

        self.fieldstack = [None, None]

        self.doDOD = doDOD

        self.fieldinfo = []

        self.leadIn = False
        self.leadOut = False
        self.isCLV = False
        self.frameNumber = None

        self.autoMTF = True
        self.useAGC = extra_options.get("useAGC", True)

        self.verboseVITS = False

        self.demodcache = DemodCache(
            self.rf, self.infile, self.freader, self.rf_opts, num_worker_threads=self.numthreads
        )

        self.bw_ratios = []

        self.decodethread = None
        self.threadreturn = {}

    def __del__(self):
        del self.demodcache

    def close(self):
        """ deletes all open files, so it's possible to pickle an LDDecode object """

        if self.ffmpeg_rftbc is not None:
            try:
                self.ffmpeg_rftbc.kill()
            except Exception:
                pass

        # use setattr to force file closure by unlinking the objects
        for outfiles in [
            "infile",
            "outfile_video",
            "outfile_audio",
            "outfile_json",
            "outfile_efm",
            "outfile_rftbc",
            "outfile_ac3",
        ]:
            setattr(self, outfiles, None)

        self.demodcache.end()

    def roughseek(self, location, isField=True):
        self.prevPhaseID = None

        self.fdoffset = location
        if isField:
            self.fdoffset *= self.bytes_per_field

    def checkMTF(self, field, pfield=None):
        oldmtf = self.mtf_level

        if not self.autoMTF:
            self.mtf_level = np.max(1 - (self.frameNumber / 10000), 0)
        else:
            if len(self.bw_ratios) == 0:
                return True

            # scale for NTSC - 1.1 to 1.55
            self.mtf_level = np.clip((np.mean(self.bw_ratios) - 1.08) / 0.38, 0, 1)

        return np.abs(self.mtf_level - oldmtf) < 0.05

    @profile
    def detectLevels(self, field):
        # Returns sync level, 0IRE, and 100IRE levels of a field
        # computed from HSYNC areas and VITS

        sync_hzs = []
        ire0_hzs = []
        ire100_hzs = []

        for wl in field.rf.SysParams['LD_VITS_whitelocs'] + field.rf.SysParams['LD_VITS_code_slices']:
            # Code slice areas have a fourth value for percentile.
            ls = field.lineslice(*wl[:3])
            cut = field.data['video']['demod'][ls]
            freq = np.percentile(cut, 50 if len(wl) == 3 else wl[3])
            freq_ire = field.rf.hztoire(freq, spec=True)
            
            if inrange(freq_ire, 95, 110):
                ire100_hzs.append(freq)
            
        for l in range(12, self.output_lines):
            lsa = field.lineslice(l, 0.25, 4)

            begin_ire0 = field.rf.SysParams["colorBurstUS"][1]
            end_ire0 = field.rf.SysParams["activeVideoUS"][0]
            lsb = field.lineslice(l, begin_ire0 + 0.25, end_ire0 - begin_ire0 - 0.5)

            # compute wow adjustment
            thislinelen = (
                field.linelocs[l + field.lineoffset]
                - field.linelocs[l + field.lineoffset - 1]
            )
            adj = field.rf.linelen / thislinelen

            if inrange(adj, 0.98, 1.02):
                sync_hzs.append(nb_median(field.data["video"]["demod_05"][lsa]) / adj)
                ire0_hzs.append(nb_median(field.data["video"]["demod_05"][lsb]) / adj)

        return np.median(sync_hzs), np.median(ire0_hzs), np.mean(ire100_hzs)

    def AC3filter(self, rftbc):
        self.AC3Collector.add(rftbc)

        blk = self.AC3Collector.get_block()
        while blk is not None:
            fftdata = np.fft.fft(blk)
            filtdata = np.fft.ifft(fftdata * self.rf.Filters['AC3']).real
            odata = self.AC3Collector.cut(filtdata)
            odata = np.clip(odata / 64, -100, 100) 

            self.outfile_ac3.write(np.int8(odata))

            blk = self.AC3Collector.get_block()

    def writeout(self, dataset):
        f, fi, picture, audio, efm = dataset
        if self.digital_audio is True:
            if self.outfile_pre_efm is not None:
                self.outfile_pre_efm.write(efm.tobytes())

            efm_out = self.efm_pll.process(efm)
            self.outfile_efm.write(efm_out.tobytes())

        fi["audioSamples"] = 0 if audio is None else int(len(audio) / 2)
        fi["efmTValues"] = len(efm_out) if self.digital_audio else 0

        self.fieldinfo.append(fi)

        self.outfile_video.write(picture)
        self.fields_written += 1

        if self.do_rftbc:
            rftbc = f.rf_tbc()

            if self.outfile_rftbc is not None:
                self.outfile_rftbc.write(rftbc)

            if self.outfile_ac3 is not None:
                self.AC3filter(rftbc)

            if self.pipe_rftbc is not None:
                self.pipe_rftbc.send(rftbc)

        if audio is not None and self.outfile_audio is not None:
            self.outfile_audio.write(audio)

    @profile
    def decodefield(self, start, mtf_level, prevfield=None, initphase=False, redo=False, rv=None):
        """ returns field object if valid, and the offset to the next decode """

        if rv is None:
            rv = {}

        rv['field'] = None
        rv['offset'] = None
    
        readloc = int(start - self.rf.blockcut)
        if readloc < 0:
            readloc = 0

        readloc_block = readloc // self.blocksize
        numblocks = (self.readlen // self.blocksize) + 2

        rawdecode = self.demodcache.read(
            readloc_block * self.blocksize,
            numblocks * self.blocksize,
            mtf_level,
            forceredo=redo
        )

        if rawdecode is None:
            # logger.info("Failed to demodulate data")
            return None, None

        f = self.FieldClass(
            self.rf,
            rawdecode,
            prevfield=prevfield,
            initphase=initphase,
            fields_written=self.fields_written,
            readloc=rawdecode["startloc"],
        )

        if self.use_profiler:
            if self.system == 'NTSC':
                self.lpf.add_function(f.refine_linelocs_burst)
                self.lpf.add_function(f.compute_burst_offsets)

            self.lpf.add_function(f.get_linelen)
            self.lpf.add_function(f.get_burstlevel)
            self.lpf.add_function(f.compute_line_bursts)
            lpf_wrapper = self.lpf(f.process)
        else:
            lpf_wrapper = f.process

        try:
            lpf_wrapper()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise e

        rv['field'] = f
        rv['offset'] = f.nextfieldoffset - (readloc - rawdecode["startloc"])

        if not f.valid:
            # logger.info("Bad data - jumping one second")
            rv['offset'] = f.nextfieldoffset

        return rv['field'], rv['offset']

    @profile
    def readfield(self, initphase=False):
        done = False
        adjusted = False
        redo = None

        if len(self.fieldstack) >= 2:
            # XXX: Need to cut off the previous field here, since otherwise
            # it'll leak for now.
            if self.fieldstack[-1]:
                self.fieldstack[-1].prevfield = None

            self.fieldstack.pop(-1)

        while done is False:
            if redo:
                # Drop existing thread
                self.decodethread = None

                # Start new thread
                self.threadreturn = {}
                df_args = (redo, self.mtf_level, self.fieldstack[0], initphase, redo, self.threadreturn)

                self.decodethread = threading.Thread(target=self.decodefield, args=df_args)
                # .run() does not actually run this in the background
                self.decodethread.run()
                f, offset = self.threadreturn['field'], self.threadreturn['offset']
                self.decodethread = None

                # Only allow one redo, no matter what
                done = True
                redo = None
            elif self.decodethread and self.decodethread.ident:
                self.decodethread.join()
                self.decodethread = None
                
                f, offset = self.threadreturn['field'], self.threadreturn['offset']
            else: # assume first run
                f = None
                offset = 0

            if True:
                # Start new thread
                self.threadreturn = {}
                if f and f.valid:
                    prevfield = f
                    toffset = self.fdoffset + offset
                else:
                    prevfield = None
                    toffset = self.fdoffset

                    if offset:
                        toffset += offset

                df_args = (toffset, self.mtf_level, prevfield, initphase, False, self.threadreturn)

                self.decodethread = threading.Thread(target=self.decodefield, args=df_args)
                self.decodethread.start()
                # Enabling .join() here to disable threading makes it slower,
                # but makes the output more deterministic
                self.decodethread.join()
            
            # process previous run
            if f:
                self.fdoffset += offset
            elif offset is None:
                # Probable end, so push an empty field
                self.fieldstack.insert(0, None)

            if f and f.valid:
                picture, audio, efm = f.downscale(
                    linesout=self.output_lines, 
                    final=True, 
                    audio=self.analog_audio,
                    lastfieldwritten=self.lastFieldWritten,
                )

                metrics = self.computeMetrics(f, None, verbose=True)
                if "blackToWhiteRFRatio" in metrics and adjusted is False:
                    keep = 900 if self.isCLV else 30
                    self.bw_ratios.append(metrics["blackToWhiteRFRatio"])
                    self.bw_ratios = self.bw_ratios[-keep:]

                redo = f.needrerun or not self.checkMTF(f, self.fieldstack[0])
                if redo:
                    redo = self.fdoffset - offset

                # Perform AGC changes on first fields only to prevent luma mismatch intra-field
                if self.useAGC and f.isFirstField and f.sync_confidence > 80:
                    sync_hz, ire0_hz, ire100_hz = self.detectLevels(f)

                    actualwhiteIRE = f.rf.hztoire(ire100_hz)

                    sync_ire_diff = nb_abs(self.rf.hztoire(sync_hz) - self.rf.DecoderParams["vsync_ire"])
                    whitediff = nb_abs(self.rf.hztoire(ire100_hz) - actualwhiteIRE)
                    ire0_diff = nb_abs(self.rf.hztoire(ire0_hz))

                    acceptable_diff = 2 if self.fields_written else 0.5

                    if max((whitediff, ire0_diff, sync_ire_diff)) > acceptable_diff:
                        hz_ire = (ire100_hz - ire0_hz) / 100
                        vsync_ire = (sync_hz - ire0_hz) / hz_ire

                        if vsync_ire > -20:
                            logger.warning(
                                "At field #{0}, Auto-level detection malfunction (vsync IRE computed at {1}, nominal ~= -40), possible disk skipping".format(
                                    len(self.fieldinfo), np.round(vsync_ire, 2)
                                ))
                        else:
                            redo = self.fdoffset - offset

                            self.rf.DecoderParams["ire0"] = ire0_hz
                            # Note that vsync_ire is a negative number, so (sync_hz - ire0_hz) is correct
                            self.rf.DecoderParams["hz_ire"] = hz_ire
                            self.rf.DecoderParams["vsync_ire"] = vsync_ire

                if adjusted is False and redo:
                    self.demodcache.flush_demod()
                    adjusted = True
                    self.fdoffset = redo
                else:
                    done = True
                    fieldlength = f.linelocs[self.output_lines] - f.linelocs[0]
                    minlength = (f.inlinelen * self.output_lines) - 2
                    if ((f.sync_confidence < 50) and (fieldlength < minlength)):
                        logger.warning("WARNING: Player skip detected, output will be corrupted")

                    self.fieldstack.insert(0, f)

            if f is None and offset is None:
                # EOF, probably
                return None

            if self.decodethread and not self.decodethread.ident and not redo:
                self.decodethread.start()

        if f is None or f.valid is False:
            return None
        
        if f is not None and self.fname_out is not None:
            # Only write a FirstField first
            if len(self.fieldinfo) == 0 and not f.isFirstField:
                return f

            # XXX: this routine currently performs a needed sanity check
            fi, needFiller = self.buildmetadata(f)

            self.lastvalidfield[f.isFirstField] = (f, fi, picture, audio, efm)

            if needFiller:
                if self.lastvalidfield[not f.isFirstField] is not None:
                    self.writeout(self.lastvalidfield[not f.isFirstField])
                    self.writeout(self.lastvalidfield[f.isFirstField])

                # If this is the first field to be written, don't write anything
                return f

            self.lastFieldWritten = (self.fields_written, f.readloc)
            self.writeout(self.lastvalidfield[f.isFirstField])

        return f

    def decodeFrameNumber(self, f1, f2):
        """ decode frame #/information from Philips code data on both fields """

        # CLV
        self.isCLV = False
        self.earlyCLV = False
        self.clvMinutes = None
        self.clvSeconds = None
        self.clvFrameNum = None

        def decodeBCD(bcd):
            """Read a BCD-encoded number.
            Raise ValueError if any of the digits aren't valid BCD."""

            if bcd == 0:
                return 0
            else:
                digit = bcd & 0xF
                if digit > 9:
                    raise ValueError("Non-decimal BCD digit")
                return (10 * decodeBCD(bcd >> 4)) + digit

        leadoutCount = 0

        for l in f1.linecode + f2.linecode:
            if l is None:
                continue

            if l == 0x80EEEE:  # lead-out reached
                leadoutCount += 1
                # Require two leadouts, since there may only be one field in the raw data w/it
                if leadoutCount == 2:
                    self.leadOut = True
            elif l == 0x88FFFF:  # lead-in
                self.leadIn = True
            elif (l & 0xF0DD00) == 0xF0DD00:  # CLV minutes/hours
                try:
                    self.clvMinutes = decodeBCD(l & 0xFF) + (
                        decodeBCD((l >> 16) & 0xF) * 60
                    )
                    self.isCLV = True
                except ValueError:
                    pass
            elif (l & 0xF00000) == 0xF00000:  # CAV frame
                # Ignore the top bit of the first digit, used for PSC
                try:
                    rv = decodeBCD(l & 0x7FFFF)
                    self.isCLV = False
                    return rv
                except ValueError:
                    pass
            elif (l & 0x80F000) == 0x80E000:  # CLV picture #
                try:
                    sec1s = decodeBCD((l >> 8) & 0xF)
                    sec10s = ((l >> 16) & 0xF) - 0xA
                    if sec10s < 0:
                        raise ValueError("Digit 2 not in range A-F")

                    self.clvFrameNum = decodeBCD(l & 0xFF)
                    self.clvSeconds = sec1s + (10 * sec10s)
                    self.isCLV = True
                except ValueError:
                    pass

            if self.clvMinutes is not None:
                minute_seconds = self.clvMinutes * 60

                if self.clvSeconds is not None:  # newer CLV
                    # XXX: does not auto-decrement for skip frames
                    return (
                        (minute_seconds + self.clvSeconds) * self.clvfps
                    ) + self.clvFrameNum
                else:
                    self.earlyCLV = True
                    return minute_seconds

        return None  # seeking won't work w/minutes only

    def calcsnr(self, f, snrslice, psnr=False):
        # if dspicture isn't converted to float, this underflows at -40IRE
        data = f.output_to_ire(f.dspicture[snrslice].astype(float))

        signal = np.mean(data) if not psnr else 100
        noise = np.std(data)

        return 20 * np.log10(signal / noise)

    def calcpsnr(self, f, snrslice):
        return self.calcsnr(f, snrslice, psnr=True)

    def computeMetricsPAL(self, metrics, f, fp=None):

        if f.isFirstField:
            # compute IRE50 from field1 l13
            # Unforunately this is too short to get a 50IRE RF level
            wl_slice = f.lineslice_tbc(13, 4.7 + 15.5, 3)
            metrics["greyPSNR"] = self.calcpsnr(f, wl_slice)
            metrics["greyIRE"] = nb_mean(f.output_to_ire(f.dspicture[wl_slice]))
        else:
            # There's a nice long burst at 50IRE block on field2 l13
            b50slice = f.lineslice_tbc(13, 36, 20)
            metrics["palVITSBurst50Level"] = rms(f.dspicture[b50slice]) / f.out_scale

        return metrics

    def computeMetricsNTSC(self, metrics, f, fp=None):
        # check for a white flag - only on earlier discs, and only on first "frame" fields
        wf_slice = f.lineslice_tbc(11, 15, 40)
        if inrange(np.mean(f.output_to_ire(f.dspicture[wf_slice])), 92, 108):
            metrics["ntscWhiteFlagSNR"] = self.calcpsnr(f, wf_slice)

        # use line 19 to determine 0 and 70 IRE burst levels for MTF compensation later
        c = CombNTSC(f)

        level, phase, snr = c.calcLine19Info()
        if level is not None:
            metrics["ntscLine19ColorPhase"] = phase
            metrics["ntscLine19ColorRawSNR"] = snr

        ire50_slice = f.lineslice_tbc(19, 36, 10)
        metrics["greyPSNR"] = self.calcpsnr(f, ire50_slice)
        metrics["greyIRE"] = nb_mean(f.output_to_ire(f.dspicture[ire50_slice]))

        ire50_rawslice = f.lineslice(19, 36, 10)
        rawdata = f.rawdata[
            ire50_rawslice.start
            - int(self.rf.delays["video_white"]) : ire50_rawslice.stop
            - int(self.rf.delays["video_white"])
        ]
        metrics["greyRFLevel"] = np.std(rawdata)

        if not f.isFirstField and fp is not None:
            cp = CombNTSC(fp)

            level3d, phase3d, snr3d = c.calcLine19Info(cp)
            if level3d is not None:
                metrics["ntscLine19Burst70IRE"] = level3d
                metrics["ntscLine19Color3DRawSNR"] = snr3d

                sl_cburst = f.lineslice_tbc(19, 4.7 + 0.8, 2.4)
                diff = (
                    f.dspicture[sl_cburst].astype(float)
                    - fp.dspicture[sl_cburst].astype(float)
                ) / 2

                metrics["ntscLine19Burst0IRE"] = np.sqrt(2) * rms(diff) / f.out_scale

        return metrics

    def computeMetrics(self, f, fp=None, verbose=False):
        system = f.rf.system
        if self.verboseVITS:
            verbose = True

        metrics = {}

        if system == "NTSC":
            self.computeMetricsNTSC(metrics, f, fp)
        else:
            self.computeMetricsPAL(metrics, f, fp)

        # FIXME: these should probably be computed in the Field class
        f.whitesnr_slice = None

        for l in f.rf.SysParams["LD_VITS_whitelocs"]:
            wl_slice = f.lineslice_tbc(*l)
            # logger.info(l, np.mean(f.output_to_ire(f.dspicture[wl_slice])))
            if inrange(np.mean(f.output_to_ire(f.dspicture[wl_slice])), 90, 110):
                f.whitesnr_slice = l
                metrics["wSNR"] = self.calcpsnr(f, wl_slice)
                metrics["whiteIRE"] = np.mean(f.output_to_ire(f.dspicture[wl_slice]))

                rawslice = f.lineslice(*l)
                rawdata = f.rawdata[
                    rawslice.start
                    - int(self.rf.delays["video_white"]) : rawslice.stop
                    - int(self.rf.delays["video_white"])
                ]
                metrics["whiteRFLevel"] = np.std(rawdata)

                break

        bl_slice = f.lineslice(*f.rf.SysParams["blacksnr_slice"])
        bl_slicetbc = f.lineslice_tbc(*f.rf.SysParams["blacksnr_slice"])

        delay = int(f.rf.delays["video_sync"])
        bl_sliceraw = slice(bl_slice.start - delay, bl_slice.stop - delay)
        metrics["blackLineRFLevel"] = np.std(f.rawdata[bl_sliceraw])

        metrics["blackLinePreTBCIRE"] = f.rf.hztoire(
            np.mean(f.data["video"]["demod"][bl_slice])
        )
        metrics["blackLinePostTBCIRE"] = f.output_to_ire(
            np.mean(f.dspicture[bl_slicetbc])
        )

        metrics["bPSNR"] = self.calcpsnr(f, bl_slicetbc)

        if "whiteRFLevel" in metrics:
            metrics["blackToWhiteRFRatio"] = (
                metrics["blackLineRFLevel"] / metrics["whiteRFLevel"]
            )

        outputkeys = metrics.keys() if verbose else ["wSNR", "bPSNR"]

        metrics_rounded = {}

        for k in outputkeys:
            if k not in metrics:
                continue

            if "Ratio" in k:
                digits = 4
            elif "Burst" in k:
                digits = 3
            else:
                digits = 1
            rounded = roundfloat(metrics[k], places=digits)
            if np.isfinite(rounded):
                metrics_rounded[k] = rounded

        return metrics_rounded

    @profile
    def buildmetadata(self, f, check_phase=True):
        """ returns field information JSON and whether or not a backfill field is needed """
        prevfi = self.fieldinfo[-1] if len(self.fieldinfo) else None

        fi = {
            "isFirstField": True if f.isFirstField else False,
            "syncConf": f.compute_syncconf(),
            "seqNo": len(self.fieldinfo) + 1,
            "diskLoc": np.round((f.readloc / self.bytes_per_field) * 10) / 10,
            "fileLoc": int(np.floor(f.readloc)),
            "medianBurstIRE": roundfloat(f.burstmedian),
        }

        if self.doDOD:
            dropout_lines, dropout_starts, dropout_ends = f.dropout_detect()
            if len(dropout_lines):
                fi["dropOuts"] = {
                    "fieldLine": dropout_lines,
                    "startx": dropout_starts,
                    "endx": dropout_ends,
                }

        # This is a bitmap, not a counter
        decodeFaults = 0

        fi["fieldPhaseID"] = f.fieldPhaseID

        if prevfi is not None:
            if check_phase and (not (
                (
                    fi["fieldPhaseID"] == 1
                    and prevfi["fieldPhaseID"] == f.rf.SysParams["fieldPhases"]
                )
                or (fi["fieldPhaseID"] == prevfi["fieldPhaseID"] + 1))
            ):
                logger.warning(
                    "At field #{0}, Field phaseID sequence mismatch ({1}->{2}) (player may be paused)".format(
                        len(self.fieldinfo), prevfi["fieldPhaseID"], fi["fieldPhaseID"]
                    )
                )
                decodeFaults |= 2

            if prevfi["isFirstField"] == fi["isFirstField"]:
                # logger.info('WARNING!  isFirstField stuck between fields')
                if inrange(fi["diskLoc"] - prevfi["diskLoc"], 0.95, 1.05):
                    decodeFaults |= 1
                    fi["isFirstField"] = not prevfi["isFirstField"]
                    fi["syncConf"] = 10
                else:
                    logger.error("Skipped field")
                    decodeFaults |= 4
                    fi["syncConf"] = 0
                    return fi, True

        fi["decodeFaults"] = decodeFaults
        fi["vitsMetrics"] = self.computeMetrics(self.fieldstack[0], self.fieldstack[1])

        fi["vbi"] = {"vbiData": [int(lc) for lc in f.linecode if lc is not None]}

        self.frameNumber = None
        if f.isFirstField:
            self.firstfield = f
        else:
            # use a stored first field, in case we start with a second field
            if self.firstfield is not None:
                # process VBI frame info data
                self.frameNumber = self.decodeFrameNumber(self.firstfield, f)

                rawloc = np.floor((f.readloc / self.bytes_per_field) / 2)

                disk_Type = "CLV" if self.isCLV else "CAV"
                disk_TimeCode = None
                disk_Frame = None
                special = None

                try:
                    if self.isCLV and self.earlyCLV:  # early CLV
                        disk_TimeCode = f"{self.clvMinutes}:xx"
                    # print("file frame %d early-CLV minute %d" % (rawloc, self.clvMinutes), file=sys.stderr)
                    elif self.isCLV and self.frameNumber is not None and self.clvMinutes is not None:
                        print(f'{self.clvMinutes} {self.clvMinutes is None}\n\n')
                        disk_TimeCode = "%d:%.2d.%.2d Frame #%d" % (
                            self.clvMinutes,
                            self.clvSeconds,
                            self.clvFrameNum,
                            self.frameNumber,
                        )
                    elif self.frameNumber:
                        # print("file frame %d CAV frame %d" % (rawloc, self.frameNumber), file=sys.stderr)
                        disk_Frame = f"{self.frameNumber}"
                    elif self.leadIn:
                        special = "Lead In"
                    elif self.leadOut:
                        special = "Lead Out"
                    else:
                        special = "Pulldown/Telecine Frame"

                    if self.est_frames is not None:
                        outstr = f"Frame {(self.fields_written//2)+1}/{int(self.est_frames)}: File Frame {int(rawloc)}: {disk_Type} "
                    else:
                        outstr = f"File Frame {int(rawloc)}: {disk_Type} "
                    if self.isCLV and disk_TimeCode:
                        outstr += f"Timecode {disk_TimeCode} "
                    elif disk_Frame:
                        outstr += f"Frame #{disk_Frame} "
                        

                    if special is not None:
                        outstr += special

                    self.logger.status(outstr)

                    # Prepare JSON fields
                    if self.verboseVITS:
                        if self.frameNumber is not None:
                            fi["cavFrameNr"] = int(self.frameNumber)

                        if self.isCLV and self.clvMinutes is not None:
                            fi["clvMinutes"] = int(self.clvMinutes)

                        if self.isCLV and not self.earlyCLV:
                            fi["clvSeconds"] = int(self.clvSeconds)
                            fi["clvFrameNr"] = int(self.clvFrameNum)
                except Exception:
                    logger.warning("file frame %d : VBI decoding error", rawloc)
                    traceback.print_exc()

        return fi, False

    def seek_getframenr(self, startfield):
        """ Reads from file location startfield, returns first VBI frame # or None on failure and revised startfield """

        """ Note that if startfield is not 0, and the read fails, it will automatically retry
            at file location 0
        """

        curfield = None
        prevfield = None

        self.roughseek(startfield)

        for fields in range(10):
            f, offset = self.decodefield(self.fdoffset, 0)

            if f is None:
                # If given an invalid starting location (i.e. seeking to a frame in an already cut raw file),
                # go back to the beginning and try again.
                if startfield != 0:
                    startfield = 0
                    self.roughseek(startfield)
                else:
                    return None, startfield
            elif not f.valid:
                self.fdoffset += offset
            else:
                prevfield = curfield
                curfield = f
                self.fdoffset += offset

                # Two fields are needed to be sure to have sufficient Philips code data
                # to determine frame #.
                if prevfield is not None and f.valid:
                    fnum = self.decodeFrameNumber(prevfield, curfield)

                    if self.earlyCLV:
                        logger.error("Cannot seek in early CLV disks w/o timecode")
                        return None, startfield
                    elif fnum is not None:
                        rawloc = np.floor((f.readloc / self.bytes_per_field) / 2)
                        logger.info("seeking: file loc %d frame # %d", rawloc, fnum)

                        return fnum, startfield, f.readloc

        return None, None, None

    def seek(self, startframe, target):
        """ Attempts to find frame target from file location startframe """
        logger.info("Beginning seek")

        if not sys.warnoptions:
            import warnings

            warnings.simplefilter("ignore")

        curfield = startframe * 2

        for retries in range(3):
            fnr, curfield, readloc = self.seek_getframenr(curfield)
            if fnr is None:
                return None

            cur = int((readloc / self.bytes_per_field))
            if fnr == target:
                logger.info("Finished seek")
                print("Finished seeking, starting at frame", fnr, file=sys.stderr)
                self.roughseek(cur)
                return cur

            curfield += ((target - fnr) * 2) - 1

        return None

    def build_json(self):
        """ build up the JSON structure for file output. """
        jout = {}
        jout["pcmAudioParameters"] = {
            "bits": 16,
            "isLittleEndian": True,
            "isSigned": True,
            "sampleRate": self.analog_audio,
        }

        vp = {}

        vp["numberOfSequentialFields"] = len(self.fieldinfo)
        vp["osInfo"] = f'{platform.system()}:{platform.release()}:{platform.version()}'

        # get the first valid field in the stack if any
        for f in self.fieldstack:
            if f:
                break

        if not f:
            return None

        vp["gitBranch"] = self.branch
        vp["gitCommit"] = self.commit

        vp["system"] = f.rf.system

        vp["fieldWidth"] = f.rf.SysParams["outlinelen"]
        vp["sampleRate"] = f.rf.SysParams["outfreq"] * 1000000

        vp["black16bIre"] = float(f.hz_to_output(f.rf.iretohz(self.blackIRE)))
        vp["white16bIre"] = float(f.hz_to_output(f.rf.iretohz(100)))

        vp["fieldHeight"] = f.outlinecount

        # current burst adjustment as of 2/27/19, update when #158 is fixed!
        badj = -1.4
        spu = f.rf.SysParams["outfreq"]
        vp["colourBurstStart"] = int(np.round(
            (f.rf.SysParams["colorBurstUS"][0] * spu) + badj
        ))
        vp["colourBurstEnd"] = int(np.round(
            (f.rf.SysParams["colorBurstUS"][1] * spu) + badj
        ))
        vp["activeVideoStart"] = int(np.round(
            (f.rf.SysParams["activeVideoUS"][0] * spu) + badj
        ))
        vp["activeVideoEnd"] = int(np.round(
            (f.rf.SysParams["activeVideoUS"][1] * spu) + badj
        ))

        jout["videoParameters"] = vp

        jout["fields"] = self.fieldinfo.copy()

        return jout
