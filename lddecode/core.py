import copy
import itertools
import sys
import threading
import time

from multiprocessing import Process, Queue, JoinableQueue, Pipe

# standard numeric/scientific libraries
import numpy as np
import scipy.signal as sps
import scipy.interpolate as spi


# Use PyFFTW's faster FFT implementation if available
try:
    import pyfftw.interfaces.numpy_fft as npfft
    import pyfftw.interfaces

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)
except ImportError:
    import numpy.fft as npfft

# internal libraries

# XXX: figure out how to handle these module imports better for vscode imports
try:
    import efm_pll
except ImportError:
    from lddecode import efm_pll

try:
    from utils import *
except ImportError:
    from lddecode.utils import *

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


def calclinelen(SP, mult, mhz):
    if type(mhz) == str:
        mhz = SP[mhz]

    return int(np.round(SP["line_period"] * mhz * mult))


# states for first field of validpulses (second field is pulse #)
HSYNC, EQPL1, VSYNC, EQPL2 = range(4)

# These are invariant parameters for PAL and NTSC
SysParams_NTSC = {
    "fsc_mhz": (315.0 / 88.0),
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
    # NOTE: this changes to 2.88mhz on AC3 disks
    "audio_rfreq": (1000000 * 315 / 88 / 227.5) * 178.75,
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
}

# In color NTSC, the line period was changed from 63.5 to 227.5 color cycles,
# which works out to 63.555(with a bar on top) usec
SysParams_NTSC["line_period"] = 1 / (SysParams_NTSC["fsc_mhz"] / 227.5)
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
    "audio_filterwidth": 150000,
    "audio_filterorder": 800,
}

# Settings for use with noisier disks
RFParams_NTSC_lowband = {
    # The audio notch filters are important with DD v3.0+ boards
    "audio_notchwidth": 350000,
    "audio_notchorder": 2,
    "video_deemp": (120e-9, 320e-9),
    "video_bpf_low": 3800000,
    "video_bpf_high": 12500000,
    "video_bpf_order": 4,
    "video_lpf_freq": 4200000,  # in mhz
    "video_lpf_order": 6,  # butterworth filter order
    # MTF filter
    "MTF_basemult": 0.4,  # general ** level of the MTF filter for frame 0.
    "MTF_poledist": 0.9,
    "MTF_freq": 12.2,  # in mhz
    # used to detect rot
    "video_hpf_freq": 10000000,
    "video_hpf_order": 4,
    "audio_filterwidth": 150000,
    "audio_filterorder": 800,
}

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
    "audio_filterwidth": 150000,
    "audio_filterorder": 800,
}

RFParams_PAL_lowband = {
    # The audio notch filters are important with DD v3.0+ boards
    "audio_notchwidth": 200000,
    "audio_notchorder": 2,
    "video_deemp": (100e-9, 400e-9),
    # XXX: guessing here!
    "video_bpf_low": 3200000,
    "video_bpf_high": 13000000,
    "video_bpf_order": 1,
    "video_lpf_freq": 4800000,
    "video_lpf_order": 7,
    # MTF filter
    "MTF_basemult": 1.0,  # general ** level of the MTF filter for frame 0.
    "MTF_poledist": 0.70,
    "MTF_freq": 10,
    # used to detect rot
    "video_hpf_freq": 10000000,
    "video_hpf_order": 4,
    "audio_filterwidth": 150000,
    "audio_filterorder": 800,
}


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
        blocklen=32 * 1024,
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

        """

        self.blocklen = blocklen
        self.blockcut = 1024  # ???
        self.blockcut_end = 0
        self.system = system

        self.NTSC_ColorNotchFilter = extra_options.get("NTSC_ColorNotchFilter", False)
        self.PAL_V4300D_NotchFilter = extra_options.get("PAL_V4300D_NotchFilter", False)
        lowband = extra_options.get("lowband", False)

        freq = inputfreq
        self.freq = freq
        self.freq_half = freq / 2
        self.freq_hz = self.freq * 1000000
        self.freq_hz_half = self.freq_hz / 2

        self.mtf_mult = 1.0
        self.mtf_offset = 0

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

        self.SysParams["analog_audio"] = has_analog_audio

        self.deemp_mult = extra_options.get("deemp_mult", (1.0, 1.0))

        deemp = list(self.DecoderParams["video_deemp"])

        deemp[0] = extra_options.get("deemp_low", deemp[0])
        deemp[1] = extra_options.get("deemp_high", deemp[1])

        self.DecoderParams["video_deemp"] = deemp

        linelen = self.freq_hz / (1000000.0 / self.SysParams["line_period"])
        self.linelen = int(np.round(linelen))
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

        self.computevideofilters()

        # This is > 0 because decode_analog_audio is in khz.
        if self.decode_analog_audio != 0:
            self.computeaudiofilters()

        if self.decode_digital_audio:
            self.computeefmfilter()

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
        freqs = np.linspace(0.0e6, 2.0e6, num=11)
        freq_per_bin = self.freq_hz / self.blocklen
        # Amplitude and phase adjustments for each band.
        # These values were adjusted empirically based on a selection of NTSC and PAL samples.
        amp = np.array([0.0, 0.2, 0.41, 0.73, 0.98, 1.03, 0.99, 0.81, 0.59, 0.42, 0.0])
        phase = np.array(
            [0.0, -0.95, -1.05, -1.05, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2]
        )
        coeffs = None

        """Compute filter coefficients for the given FFTFilter."""
        # Anything above the highest frequency is left as zero.
        coeffs = np.zeros(self.blocklen, dtype=np.complex)

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

        MTF = sps.zpk2tf(
            [],
            [
                polar2z(DP["MTF_poledist"], np.pi * MTF_polef_lo),
                polar2z(DP["MTF_poledist"], np.pi * MTF_polef_hi),
            ],
            1,
        )
        SF["MTF"] = filtfft(MTF, self.blocklen)

        # The BPF filter, defined for each system in DecoderParams
        filt_rfvideo = sps.butter(
            DP["video_bpf_order"],
            [
                DP["video_bpf_low"] / self.freq_hz_half,
                DP["video_bpf_high"] / self.freq_hz_half,
            ],
            btype="bandpass",
        )
        # Start building up the combined FFT filter using the BPF
        SF["RFVideo"] = filtfft(filt_rfvideo, self.blocklen)

        # Notch filters for analog audio.  DdD captures on NTSC need this.
        if SP["analog_audio"] and self.system == 'NTSC':
            cut_left = sps.butter(
                DP["audio_notchorder"],
                [
                    (SP["audio_lfreq"] - DP["audio_notchwidth"]) / self.freq_hz_half,
                    (SP["audio_lfreq"] + DP["audio_notchwidth"]) / self.freq_hz_half,
                ],
                btype="bandstop",
            )
            SF["Fcutl"] = filtfft(cut_left, self.blocklen)
            cut_right = sps.butter(
                DP["audio_notchorder"],
                [
                    (SP["audio_rfreq"] - DP["audio_notchwidth"]) / self.freq_hz_half,
                    (SP["audio_rfreq"] + DP["audio_notchwidth"]) / self.freq_hz_half,
                ],
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

        video_hpf = sps.butter(
            DP["video_hpf_order"], DP["video_hpf_freq"] / self.freq_hz_half, "high"
        )
        SF["Fvideo_hpf"] = filtfft(video_hpf, self.blocklen)

        # The deemphasis filter
        deemp1, deemp2 = DP["video_deemp"]
        deemp1 *= self.deemp_mult[0]
        deemp2 *= self.deemp_mult[1]
        SF["Fdeemp"] = filtfft(
            emphasis_iir(deemp1, deemp2, self.freq_hz), self.blocklen
        )

        # The direct opposite of the above, used in test signal generation
        SF["Femp"] = filtfft(emphasis_iir(deemp2, deemp1, self.freq_hz), self.blocklen)

        # Post processing:  lowpass filter + deemp
        SF["FVideo"] = SF["Fvideo_lpf"] * SF["Fdeemp"]

        # additional filters:  0.5mhz and color burst
        # Using an FIR filter here to get a known delay
        F0_5 = sps.firwin(65, [0.5 / self.freq_half], pass_zero=True)
        SF["F05_offset"] = 32
        SF["F05"] = filtfft((F0_5, [1.0]), self.blocklen)
        SF["FVideo05"] = SF["Fvideo_lpf"] * SF["Fdeemp"] * SF["F05"]

        SF["Fburst"] = filtfft(
            sps.butter(
                1,
                [
                    (SP["fsc_mhz"] - 0.1) / self.freq_half,
                    (SP["fsc_mhz"] + 0.1) / self.freq_half,
                ],
                btype="bandpass",
            ),
            self.blocklen,
        )
        SF["FVideoBurst"] = SF["Fvideo_lpf"] * SF["Fdeemp"] * SF["Fburst"]

        if self.system == "PAL":
            SF["Fpilot"] = filtfft(
                sps.butter(
                    1,
                    [
                        (SP["pilot_mhz"] - 0.1) / self.freq_half,
                        (SP["pilot_mhz"] + 0.1) / self.freq_half,
                    ],
                    btype="bandpass",
                ),
                self.blocklen,
            )
            SF["FVideoPilot"] = SF["Fvideo_lpf"] * SF["Fdeemp"] * SF["Fpilot"]

    # frequency domain slicers.  first and second stages use different ones...
    def audio_fdslice(self, freqdomain):
        return np.concatenate(
            [
                freqdomain[self.Filters["audio_fdslice_lo"]],
                freqdomain[self.Filters["audio_fdslice_hi"]],
            ]
        )

    def audio_fdslice2(self, freqdomain):
        return np.concatenate(
            [
                freqdomain[self.Filters["audio_fdslice2_lo"]],
                freqdomain[self.Filters["audio_fdslice2_hi"]],
            ]
        )

    def compute_deemp_audio2(self, dfreq):
        adeemp_b, adeemp_a = sps.butter(
            1, [(1000000 / dfreq) / (self.Filters["freq_aud2"] / 2)], btype="lowpass"
        )

        return filtfft(
            [adeemp_b, adeemp_a], self.blocklen // self.Filters["audio_fdiv2"]
        )

    def computeaudiofilters(self):
        SF = self.Filters
        SP = self.SysParams
        DP = self.DecoderParams

        # Low pass filter for 'new' audio code

        # first stage audio filters
        if self.freq >= 32:
            audio_fdiv1 = 32  # this is good for 40mhz - 16 should be ideal for 28mhz
        else:
            audio_fdiv1 = 16

        afft_halfwidth = self.blocklen // (audio_fdiv1 * 2)
        arf_freq = self.freq_hz / (audio_fdiv1 / 2)
        SF["freq_arf"] = arf_freq
        SF["audio_fdiv1"] = audio_fdiv1

        SP["audio_cfreq"] = (SP["audio_rfreq"] + SP["audio_lfreq"]) // 2
        afft_center = int((SP["audio_cfreq"] / self.freq_hz) * (self.blocklen))

        # beginning and end symmetrical frequency domain slices.  combine to make a cut-down sampling
        afft_start = int(afft_center - afft_halfwidth)
        afft_end = int(afft_center + afft_halfwidth)

        # slice areas for reduced FFT audio demodulation filters
        SF["audio_fdslice_lo"] = slice(afft_start, afft_end)
        SF["audio_fdslice_hi"] = slice(
            self.blocklen - afft_end, self.blocklen - afft_start
        )

        # compute the base frequency of the cut audio range
        SF["audio_lowfreq"] = SP["audio_cfreq"] - (
            self.freq_hz / (2 * SF["audio_fdiv1"])
        )

        apass = DP[
            "audio_filterwidth"
        ]  # audio RF bandpass.  150khz is the maximum transient.
        afilt_len = DP["audio_filterorder"]  # good for 150khz apass

        afilt_left = filtfft(
            [
                sps.firwin(
                    afilt_len,
                    [
                        (SP["audio_lfreq"] - apass) / self.freq_hz_half,
                        (SP["audio_lfreq"] + apass) / self.freq_hz_half,
                    ],
                    pass_zero=False,
                ),
                1.0,
            ],
            self.blocklen,
        )
        SF["audio_lfilt"] = self.audio_fdslice(afilt_left * SF["hilbert"])
        afilt_right = filtfft(
            [
                sps.firwin(
                    afilt_len,
                    [
                        (SP["audio_rfreq"] - apass) / self.freq_hz_half,
                        (SP["audio_rfreq"] + apass) / self.freq_hz_half,
                    ],
                    pass_zero=False,
                ),
                1.0,
            ],
            self.blocklen,
        )
        SF["audio_rfilt"] = self.audio_fdslice(afilt_right * SF["hilbert"])

        # second stage audio filters (decimates further, and applies audio LPF)
        audio_fdiv2 = 4
        SF["audio_fdiv"] = audio_fdiv1 * audio_fdiv2
        SF["audio_fdiv2"] = audio_fdiv2
        SF["freq_aud2"] = SF["freq_arf"] / audio_fdiv2

        # slice areas for reduced FFT audio filters
        SF["audio_fdslice2_lo"] = slice(0, self.blocklen // (audio_fdiv2 * 2))
        SF["audio_fdslice2_hi"] = slice(
            self.blocklen - self.blocklen // (audio_fdiv2 * 2), self.blocklen
        )

        SF["audio_lpf2"] = filtfft(
            [sps.firwin(65, [21000 / (SF["freq_aud2"] / 2)]), [1.0]],
            self.blocklen // (SF["audio_fdiv2"] * 1),
        )

        # XXX: This probably needs further tuning, but more or less flattens the 20hz-20khz response
        # on both PAL and NTSC
        # (and no, I don't know where those frequencies come from)
        addemp2lp = self.compute_deemp_audio2(2 * pi * 62)  # 2567hz
        addemp2hp1 = self.compute_deemp_audio2(2 * pi * 45)  # 3536hz
        addemp2hp2 = self.compute_deemp_audio2(2 * pi * 8)  # 19894hz (i.e. cutoff?)

        SF["audio_deemp2"] = addemp2lp + (addemp2hp1 * 0.14) + (addemp2hp2 * 0.29)

    def iretohz(self, ire):
        return self.SysParams["ire0"] + (self.SysParams["hz_ire"] * ire)

    def hztoire(self, hz):
        return (hz - self.SysParams["ire0"]) / self.SysParams["hz_ire"]

    def demodblock(self, data=None, mtf_level=0, fftdata=None, cut=False):
        rv = {}

        mtf_level *= self.mtf_mult
        mtf_level *= self.DecoderParams["MTF_basemult"]
        mtf_level += self.mtf_offset

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
        ]

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

        demod_fft_full = npfft.fft(demod)
        demod_hpf = npfft.ifft(demod_fft_full * self.Filters["Fvideo_hpf"]).real

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
                    out_video,
                    demod,
                    demod_hpf,
                    out_video05,
                    out_videoburst,
                    out_videopilot,
                ],
                names=[
                    "demod",
                    "demod_raw",
                    "demod_hpf",
                    "demod_05",
                    "demod_burst",
                    "demod_pilot",
                ],
            )
        else:
            video_out = np.rec.array(
                [out_video, demod, demod_hpf, out_video05, out_videoburst],
                names=["demod", "demod_raw", "demod_hpf", "demod_05", "demod_burst"],
            )

        rv["video"] = (
            video_out[self.blockcut : -self.blockcut_end] if cut else video_out
        )

        if self.decode_digital_audio:
            efm_out = npfft.ifft(indata_fft * self.Filters["Fefm"])
            if cut:
                efm_out = efm_out[self.blockcut : -self.blockcut_end]
            rv["efm"] = np.int16(np.clip(efm_out.real, -32768, 32767))

        if self.decode_analog_audio:
            # Audio phase 1
            hilbert = npfft.ifft(
                self.audio_fdslice(indata_fft) * self.Filters["audio_lfilt"]
            )
            audio_left = (
                unwrap_hilbert(hilbert, self.Filters["freq_arf"])
                + self.Filters["audio_lowfreq"]
            )

            hilbert = npfft.ifft(
                self.audio_fdslice(indata_fft) * self.Filters["audio_rfilt"]
            )
            audio_right = (
                unwrap_hilbert(hilbert, self.Filters["freq_arf"])
                + self.Filters["audio_lowfreq"]
            )

            audio_out = np.rec.array(
                [audio_left, audio_right], names=["audio_left", "audio_right"]
            )

            fdiv = video_out.shape[0] // audio_out.shape[0]
            rv["audio"] = (
                audio_out[self.blockcut // fdiv : -self.blockcut_end // fdiv]
                if cut
                else audio_out
            )

        return rv

    # detect clicks that are impossibly large and snip them out
    def audio_dropout_detector(self, field_audio, padding=48):
        rejects = None
        cmed = {}

        # Check left and right channels separately.
        for channel in ["audio_left", "audio_right"]:
            achannel = field_audio[channel]
            cmed[channel] = np.median(achannel)
            aabs = np.abs(achannel - cmed[channel])

            if rejects is None:
                rejects = aabs > 175000
            else:
                rejects |= aabs > 175000

        if np.sum(rejects) == 0:
            # If no spikes, return original
            return field_audio

        # Locate areas with impossible signals and perform interpolation

        reject_locs = np.where(rejects)[0]
        reject_areas = []
        cur_area = [reject_locs[0] - padding, reject_locs[0] + padding]

        for r in np.where(rejects)[0][1:]:
            if (r - padding) > cur_area[1]:
                reject_areas.append(tuple(cur_area))
                cur_area = [r - padding, r + padding]
            else:
                cur_area[1] = r + padding

        reject_areas.append(cur_area)

        field_audio_dod = field_audio.copy()

        for channel in ["audio_left", "audio_right"]:
            for ra in reject_areas:
                if ra[0] <= 1 and ra[1] >= len(field_audio_dod) - 1:
                    # The entire thing can be bad during spinup
                    pass
                elif ra[0] <= 1:
                    field_audio_dod[channel][0 : ra[1]] = field_audio_dod[channel][
                        ra[1] + 1
                    ]
                elif ra[1] >= len(field_audio_dod) - 1:
                    field_audio_dod[channel][ra[0] :] = field_audio_dod[channel][
                        ra[0] - 1
                    ]
                else:
                    abeg = field_audio_dod[channel][ra[0]]
                    aend = field_audio_dod[channel][ra[1]]
                    # pad np.arange run by 1 and crop it so there's always enough data to fill in
                    # XXX: clean up
                    field_audio_dod[channel][ra[0] : ra[1]] = np.arange(
                        abeg, aend, (aend - abeg) / (1 + ra[1] - ra[0])
                    )[: ra[1] - ra[0]]

        return field_audio_dod

    # Second phase audio filtering.  This works on a whole field's samples, since
    # the frequency has already been reduced by 16 or 32x.

    def runfilter_audio_phase2(self, frame_audio, start):
        outputs = []

        clips = None

        for c in [["audio_left", "audio_lfreq"], ["audio_right", "audio_rfreq"]]:
            raw = (
                frame_audio[c[0]][start : start + self.blocklen].copy()
                - self.SysParams[c[1]]
            )

            if c[0] == "audio_left":
                clips = findpeaks(raw, 300000)

            for l in clips:
                replacelen = 16 * self.Filters["audio_fdiv2"]
                raw[
                    max(0, l - replacelen) : min(l + replacelen, len(raw))
                ] = 0  # raw[replacement_idx]

            fft_in_real = self.audio_fdslice2(npfft.fft(raw))
            if len(fft_in_real) < len(self.Filters["audio_lpf2"]):
                fft_in = np.zeros_like(self.Filters["audio_lpf2"])
                fft_in[: len(fft_in_real)] = fft_in_real
            else:
                fft_in = fft_in_real
            fft_out = fft_in * self.Filters["audio_lpf2"] * self.Filters["audio_deemp2"]

            outputs.append(
                (
                    npfft.ifft(fft_out).real[: len(fft_in_real)]
                    / self.Filters["audio_fdiv2"]
                )
                + self.SysParams[c[1]]
            )

        return np.rec.array(outputs, names=["audio_left", "audio_right"])

    def audio_phase2(self, field_audio):
        # this creates an output array with left/right channels.
        output_audio2 = np.zeros(
            len(field_audio["audio_left"]) // self.Filters["audio_fdiv2"],
            dtype=field_audio.dtype,
        )

        field_audio = self.audio_dropout_detector(field_audio)

        # copy the first block in it's entirety, to keep audio and video samples aligned
        tmp = self.runfilter_audio_phase2(field_audio, 0)

        if len(tmp) >= len(output_audio2):
            return tmp[: len(output_audio2)]

        output_audio2[: tmp.shape[0]] = tmp

        end = field_audio.shape[0]

        askip = 512  # length of filters that needs to be chopped out of the ifft
        sjump = self.blocklen - (askip * self.Filters["audio_fdiv2"])

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
        fakeoutput[1500 : 1500 + synclen_full] = rf.iretohz(rf.SysParams["vsync_ire"])
        # sync 2 (used for pilot/rot level setting)
        fakeoutput[2000 : 2000 + synclen_full] = rf.iretohz(rf.SysParams["vsync_ire"])

        porch_end = 2000 + synclen_full + int(0.6 * rf.freq)
        burst_end = porch_end + int(1.2 * rf.freq)

        rate = np.full(burst_end - porch_end, rf.SysParams["fsc_mhz"], dtype=np.double)
        fakeoutput[porch_end:burst_end] += (
            genwave(rate, rf.freq / 2) * rf.SysParams["hz_ire"] * 20
        )

        # white
        fakeoutput[3000:3500] = rf.iretohz(100)

        # white + burst
        fakeoutput[4500:5000] = rf.iretohz(100)

        rate = np.full(5500 - 4200, rf.SysParams["fsc_mhz"], dtype=np.double)
        fakeoutput[4200:5500] += (
            genwave(rate, rf.freq / 2) * rf.SysParams["hz_ire"] * 20
        )

        rate = np.full(synclen_full, rf.SysParams["fsc_mhz"], dtype=np.double)
        fakeoutput[2000 : 2000 + synclen_full] = rf.iretohz(
            rf.SysParams["vsync_ire"]
        ) + (
            genwave(rate, rf.freq / 2)
            * rf.SysParams["hz_ire"]
            * rf.SysParams["vsync_ire"]
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

        fakedecode = rf.demodblock(fakesignal, mtf_level=mtf_level)
        vdemod = fakedecode["video"]["demod"]

        # XXX: sync detector does NOT reflect actual sync detection, just regular filtering @ sync level
        # (but only regular filtering is needed for DOD)
        rf.delays = {}
        rf.delays["video_sync"] = (
            calczc(
                fakedecode["video"]["demod"],
                1500,
                rf.iretohz(rf.SysParams["vsync_ire"] / 2),
                count=512,
            )
            - 1500
        )
        rf.delays["video_white"] = (
            calczc(fakedecode["video"]["demod"], 3000, rf.iretohz(50), count=512) - 3000
        )
        rf.delays["video_rot"] = int(
            np.round(
                calczc(fakedecode["video"]["demod"], 6000, rf.iretohz(-10), count=512)
                - 6000
            )
        )

        fdec_raw = fakedecode["video"]["demod_raw"]

        rf.limits = {}
        rf.limits["sync"] = (np.min(fdec_raw[1400:2800]), np.max(fdec_raw[1400:2800]))
        rf.limits["viewable"] = (
            np.min(fdec_raw[2900:6000]),
            np.max(fdec_raw[2900:6000]),
        )

        return fakedecode, fakeoutput_emp


class DemodCache:
    def __init__(
        self,
        rf,
        infile,
        loader,
        cachesize=256,
        num_worker_threads=6,
        MTF_tolerance=0.05,
    ):
        self.infile = infile
        self.loader = loader
        self.rf = rf

        self.currentMTF = 1
        self.MTF_tolerance = MTF_tolerance

        self.blocksize = self.rf.blocklen - (self.rf.blockcut + self.rf.blockcut_end)

        # Cache dictionary - key is block #, which holds data for that block
        self.lrusize = cachesize
        self.prefetch = 32  # TODO: set this to proper amount for format
        self.lru = []

        self.lock = threading.Lock()
        self.blocks = {}

        self.q_in = JoinableQueue()
        self.q_in_metadata = []

        self.q_out = Queue()

        self.threadpipes = []
        self.threads = []

        num_worker_threads = max(num_worker_threads - 1, 1)

        for i in range(num_worker_threads):
            self.threadpipes.append(Pipe())
            t = Process(
                target=self.worker, daemon=True, args=(self.threadpipes[-1][1],)
            )
            t.start()
            self.threads.append(t)

        self.deqeue_thread = threading.Thread(target=self.dequeue, daemon=True)
        self.deqeue_thread.start()

    def end(self):
        # stop workers
        for i in self.threads:
            self.q_in.put(None)

        for t in self.threads:
            t.join()

        self.q_out.put(None)

    def __del__(self):
        self.end()

    def prune_cache(self):
        """ Prune the LRU cache.  Typically run when a new field is loaded """
        if len(self.lru) < self.lrusize:
            return

        self.lock.acquire()
        for k in self.lru[self.lrusize :]:
            if k in self.blocks:
                del self.blocks[k]
        self.lock.release()

        self.lru = self.lru[: self.lrusize]

    def flush_demod(self):
        """ Flush all demodulation data.  This is called by the field class after calibration (i.e. MTF) is determined to be off """
        for k in self.blocks.keys():
            if self.blocks[k] is None:
                pass
            elif "demod" in self.blocks[k]:
                self.lock.acquire()
                del self.blocks[k]["demod"]
                self.lock.release()

    def apply_newparams(self, newparams):
        for k in newparams.keys():
            # print(k, k in self.rf.SysParams, k in self.rf.DecoderParams)
            if k in self.rf.SysParams:
                self.rf.SysParams[k] = newparams[k]

            if k in self.rf.DecoderParams:
                self.rf.DecoderParams[k] = newparams[k]

        self.rf.computefilters()

    def worker(self, pipein):
        while True:
            ispiped = False
            if pipein.poll():
                item = pipein.recv()
                ispiped = True
            else:
                item = self.q_in.get()

            if item is None or item[0] == "END":
                return

            if item[0] == "DEMOD":
                blocknum, block, target_MTF = item[1:]

                output = {}

                if "fft" not in block:
                    output["fft"] = npfft.fft(block["rawinput"])
                    fftdata = output["fft"]
                else:
                    fftdata = block["fft"]

                if (
                    "demod" not in block
                    or np.abs(block["MTF"] - target_MTF) > self.MTF_tolerance
                ):
                    output["demod"] = self.rf.demodblock(
                        fftdata=fftdata, mtf_level=target_MTF, cut=True
                    )
                    output["MTF"] = target_MTF

                self.q_out.put((blocknum, output))
            elif item[0] == "NEWPARAMS":
                self.apply_newparams(item[1])

            if not ispiped:
                self.q_in.task_done()

    def doread(self, blocknums, MTF, dodemod=True):
        need_blocks = []

        hc = 0

        self.lock.acquire()

        for b in blocknums:
            if b not in self.blocks:
                LRUupdate(self.lru, b)

                rawdata = self.loader(self.infile, b * self.blocksize, self.rf.blocklen)

                if rawdata is None or len(rawdata) < self.rf.blocklen:
                    self.blocks[b] = None
                    self.lock.release()
                    return None

                # ??? - I think I put it in to make sure it isn't erased for whatever reason, but might not be needed
                rawdatac = rawdata.copy()

                self.blocks[b] = {}
                self.blocks[b]["rawinput"] = rawdatac

            if self.blocks[b] is None:
                self.lock.release()
                return None

            if dodemod:
                handling = need_demod = ("demod" not in self.blocks[b]) or (
                    np.abs(self.blocks[b]["MTF"] - MTF) > self.MTF_tolerance
                )
            else:
                handling = need_demod = False

            # Check to see if it's already in queue to process
            if need_demod:
                for inqueue in self.q_in_metadata:
                    if inqueue[0] == b and (
                        np.abs(inqueue[1] - MTF) <= self.MTF_tolerance
                    ):
                        handling = False
                        # print(b)

                need_blocks.append(b)

            if handling:
                self.q_in.put(("DEMOD", b, self.blocks[b], MTF))
                self.q_in_metadata.append((b, MTF))
                hc = hc + 1

        self.lock.release()

        return need_blocks

    def dequeue(self):
        # This is the thread's main loop - run until killed.
        while True:
            rv = self.q_out.get()
            if rv is None:
                return

            self.lock.acquire()

            blocknum, item = rv

            if "MTF" not in item or "demod" not in item:
                # This shouldn't happen, but was observed by Simon on a decode
                logger.error(
                    "incomplete demodulated block placed on queue, block #%d", blocknum
                )
                self.q_in.put((blocknum, self.blocks[blocknum], self.currentMTF))
                self.lock.release()
                continue

            self.q_in_metadata.remove((blocknum, item["MTF"]))

            for k in item.keys():
                if k == "demod" and (
                    np.abs(item["MTF"] - self.currentMTF) > self.MTF_tolerance
                ):
                    continue
                self.blocks[blocknum][k] = item[k]

            if "input" not in self.blocks[blocknum]:
                self.blocks[blocknum]["input"] = self.blocks[blocknum]["rawinput"][
                    self.rf.blockcut : -self.rf.blockcut_end
                ]

            self.lock.release()

    def read(self, begin, length, MTF=0, dodemod=True):
        # transpose the cache by key, not block #
        t = {"input": [], "fft": [], "video": [], "audio": [], "efm": [], "rfhpf": []}

        self.currentMTF = MTF

        end = begin + length

        toread = range(begin // self.blocksize, (end // self.blocksize) + 1)
        toread_prefetch = range(
            end // self.blocksize, (end // self.blocksize) + self.prefetch
        )

        need_blocks = self.doread(toread, MTF, dodemod)

        if dodemod == False:
            raw = [self.blocks[toread[0]]["rawinput"][begin % self.blocksize :]]
            for i in range(toread[1], toread[-2]):
                raw.append(self.blocks[i]["rawinput"])
            raw.append(self.blocks[-1]["rawinput"][: end % self.blocksize])

            rv = np.concatenate(raw)
            self.prune_cache()
            return rv

        while need_blocks is not None and len(need_blocks):
            time.sleep(0.001)  # A crude busy loop
            need_blocks = self.doread(toread, MTF)

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

        need_blocks = self.doread(toread_prefetch, MTF)

        return rv

    def setparams(self, params):
        # XXX: This should flush out the data, but right now this isn't used at all
        for p in self.threadpipes:
            p[0].send(("NEWPARAMS", params))

        # Apply params to the core thread, so they match up with the decoders
        self.apply_newparams(params)


# Downscales to 16bit/44.1khz.  It might be nice when analog audio is better to support 24/96,
# but if we only support one output type, matching CD audio/digital sound is greatly preferable.
def downscale_audio(
    audio, lineinfo, rf, linecount, timeoffset=0, freq=48000.0, scale=64
):
    failed = False

    frametime = linecount / (1000000 / rf.SysParams["line_period"])
    soundgap = 1 / freq

    # include one extra 'tick' to interpolate the last one and use as a return value
    # for the next frame
    arange = np.arange(
        timeoffset, frametime + (soundgap / 2), soundgap, dtype=np.double
    )
    locs = np.zeros(len(arange), dtype=np.float)
    swow = np.zeros(len(arange), dtype=np.float)

    for i, t in enumerate(arange):
        linenum = ((t * 1000000) / rf.SysParams["line_period"]) + 1
        intlinenum = int(linenum)

        # XXX:
        # The timing handling can sometimes go outside the bounds of the known line #'s.
        # This is a quick-ish fix that should work OK but may affect quality slightly.
        if linenum < 0:
            lineloc_cur = int(lineinfo[0] + (rf.linelen * linenum))
            lineloc_next = lineloc_cur + rf.linelen
        elif len(lineinfo) > linenum + 2:
            lineloc_cur, lineloc_next = lineinfo[intlinenum : intlinenum + 2]
        else:
            # Catch things that go past the last known line by using the last lines here.
            lineloc_cur = lineinfo[-2]
            lineloc_next = lineloc_cur + rf.linelen

        sampleloc = lineloc_cur
        sampleloc += (lineloc_next - lineloc_cur) * (linenum - np.floor(linenum))

        swow[i] = (lineloc_next - lineloc_cur) / rf.linelen
        swow[i] = ((swow[i] - 1)) + 1
        # There's almost *no way* the disk is spinning more than 1.5% off, so mask TBC errors here
        # to reduce pops
        if i and np.abs(swow[i] - swow[i - 1]) > 0.015:
            swow[i] = swow[i - 1]

        locs[i] = sampleloc / scale

    output = np.zeros((2 * (len(arange) - 1)), dtype=np.int32)
    output16 = np.zeros((2 * (len(arange) - 1)), dtype=np.int16)

    for i in range(len(arange) - 1):
        start = np.int(locs[i])
        end = np.int(locs[i + 1])
        if end > start and end < len(audio["audio_left"]):
            output_left = nb_mean(audio["audio_left"][start:end])
            output_right = nb_mean(audio["audio_right"][start:end])

            output_left = (output_left * swow[i]) - rf.SysParams["audio_lfreq"]
            output_right = (output_right * swow[i]) - rf.SysParams["audio_rfreq"]

            output[(i * 2) + 0] = dsa_rescale(output_left)
            output[(i * 2) + 1] = dsa_rescale(output_right)
        else:
            # TBC failure can cause this (issue #389)
            if failed == False:
                logger.warning("Analog audio processing error, muting samples")

            failed = True

    np.clip(output, -32766, 32766, out=output16)

    return output16, arange[-1] - frametime


# The Field class contains common features used by NTSC and PAL
class Field:
    def __init__(
        self, rf, decode, audio_offset=0, keepraw=True, prevfield=None, initphase=False
    ):
        self.rawdata = decode["input"]
        self.data = decode
        self.initphase = initphase  # used for seeking or first field

        self.prevfield = prevfield

        # XXX: need a better way to prevent memory leaks than this
        # For now don't let a previous frame keep it's prev frame
        if prevfield is not None:
            prevfield.prevfield = None

        self.rf = rf
        self.freq = self.rf.freq

        self.inlinelen = self.rf.linelen
        self.outlinelen = self.rf.SysParams["outlinelen"]

        self.lineoffset = 0

        self.valid = False
        self.sync_confidence = 100

        self.dspicture = None
        self.dsaudio = None
        self.audio_offset = audio_offset
        self.audio_next_offset = audio_offset

        # On NTSC linecount rounds up to 263, and PAL 313
        self.outlinecount = (self.rf.SysParams["frame_lines"] // 2) + 1
        # this is eventually set to 262/263 and 312/313 for audio timing
        self.linecount = None

    def process(self):
        self.linelocs1, self.linebad, self.nextfieldoffset = self.compute_linelocs()
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

    def get_linelen(self, line=None, linelocs=None):
        # compute adjusted frequency from neighboring line lengths

        # If this is run early, line locations are unknown, so return
        # the general value
        if linelocs is None:
            try:
                linelocs = self.linelocs
            except:
                return self.rf.linelen

        if line is None:
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
        return self.rf.freq * (self.get_linelen(line, linelocs) / self.rf.linelen)

    def usectoinpx(self, x, line=None):
        return x * self.get_linefreq(line)

    def inpxtousec(self, x, line=None):
        return x / self.get_linefreq(line)

    def lineslice(self, l, begin=None, length=None, linelocs=None, begin_offset=0):
        """ return a slice corresponding with pre-TBC line l, begin+length are uSecs """

        # for PAL, each field has a different offset so normalize that
        l_adj = l + self.lineoffset

        _begin = linelocs[l_adj] if linelocs is not None else self.linelocs[l_adj]
        _begin += self.usectoinpx(begin, l_adj) if begin is not None else 0

        _length = self.usectoinpx(length, l_adj) if length is not None else 1

        return slice(
            int(np.floor(_begin + begin_offset)),
            int(np.ceil(_begin + _length + begin_offset)),
        )

    def usectooutpx(self, x):
        return x * self.rf.SysParams["outfreq"]

    def outpxtousec(self, x):
        return x / self.rf.SysParams["outfreq"]

    def hz_to_output(self, input):
        reduced = (input - self.rf.SysParams["ire0"]) / self.rf.SysParams["hz_ire"]
        reduced -= self.rf.SysParams["vsync_ire"]

        return np.uint16(
            np.clip(
                (reduced * self.out_scale) + self.rf.SysParams["outputZero"], 0, 65535
            )
            + 0.5
        )

    def output_to_ire(self, output):
        return (
            (output - self.rf.SysParams["outputZero"]) / self.out_scale
        ) + self.rf.SysParams["vsync_ire"]

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

        return slice(int(np.round(_begin)), int(np.round(_begin + _length)))

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

        LT["hsync_median"] = np.median(hlens)

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

    def pulse_qualitycheck(self, prevpulse, pulse):

        if prevpulse[0] > 0 and pulse[0] > 0:
            exprange = (0.4, 0.6)
        elif prevpulse[0] == 0 and pulse[0] == 0:
            exprange = (0.9, 1.1)
        else:  # transition to/from regular hsyncs can be .5 or 1H
            exprange = (0.4, 1.1)

        linelen = (pulse[1].start - prevpulse[1].start) / self.inlinelen
        inorder = inrange(linelen, *exprange)

        return inorder

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
                    earliest_eq = p.start + (
                        self.inlinelen * (np.min(self.rf.SysParams["field_lines"]) - 10)
                    )
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

    def refinepulses(self):
        LT = self.get_timings()

        HSYNC, EQPL1, VSYNC, EQPL2 = range(4)

        i = 0
        valid_pulses = []
        num_vblanks = 0

        while i < len(self.rawpulses):
            curpulse = self.rawpulses[i]
            if inrange(curpulse.len, *LT["hsync"]):
                good = (
                    self.pulse_qualitycheck(valid_pulses[-1], (0, curpulse))
                    if len(valid_pulses)
                    else False
                )
                valid_pulses.append((HSYNC, curpulse, good))
                i += 1
            elif (
                i > 2
                and inrange(self.rawpulses[i].len, *LT["eq"])
                and (len(valid_pulses) and valid_pulses[-1][0] == HSYNC)
            ):
                # print(i, self.rawpulses[i])
                done, vblank_pulses = self.run_vblank_state_machine(
                    self.rawpulses[i - 2 : i + 24], LT
                )
                if done:
                    [valid_pulses.append(p) for p in vblank_pulses[2:]]
                    i += len(vblank_pulses) - 2
                    num_vblanks += 1
                else:
                    spulse = (HSYNC, self.rawpulses[i], False)
                    i += 1
            else:
                spulse = (HSYNC, self.rawpulses[i], False)
                i += 1

        return valid_pulses  # , num_vblanks

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

        return core

    def processVBlank(self, validpulses, start, limit=None):

        firstblank, lastblank = self.getBlankRange(validpulses, start)
        conf = 100

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
            hdist = int(np.round(dist * 2))

            # isfirstfield = not ((hdist % 2) == self.rf.SysParams['firstField1H'][0])
            isfirstfield = (hdist % 2) == (self.rf.SysParams["firstFieldH"][1] != 1)

            # for PAL VSYNC, the offset is 2.5H, so the calculation must be reversed
            if (distfroml1 * 2) % 2:
                isfirstfield = not isfirstfield

            eqgap = self.rf.SysParams["firstFieldH"][isfirstfield]
            line0 = firstloc - ((eqgap + distfroml1) * self.inlinelen)

            return np.int(line0), isfirstfield, firstblank, conf

        """
        If there are no valid sections, check line 0 and the first eq pulse, and the last eq
        pulse and the following line.  If the combined xH is correct for the standard in question
        (1.5H for NTSC, 1 or 2H for PAL, that means line 0 has been found correctly.
        """

        conf = 50

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

            return validpulses[firstblank - 1][1].start, isfirstfield, firstblank, conf

        conf = 0

        return None, None, None, 0

    def computeLineLen(self, validpulses):
        # determine longest run of 0's
        longrun = [-1, -1]
        currun = None
        for i, v in enumerate([p[0] for p in self.validpulses]):
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
            linelen = self.validpulses[i][1].start - self.validpulses[i - 1][1].start
            if inrange(linelen / self.inlinelen, 0.95, 1.05):
                linelens.append(
                    self.validpulses[i][1].start - self.validpulses[i - 1][1].start
                )

        return np.mean(linelens)

    def skip_check(self):
        ''' This routine checks to see if there's a (probable) VSYNC at the end.
            Returns a (currently rough) probability.
        '''
        score = 0
        vsync_lines = 0

        vsync_ire = self.rf.SysParams['vsync_ire']

        for l in range(self.outlinecount, self.outlinecount + 8):
            sl = self.lineslice(l, 0, self.rf.SysParams['line_period'])
            line_ire = self.rf.hztoire(nb_median(self.data['video']['demod'][sl]))

            # vsync_ire is always negative, so /2 is the higher number

            if inrange(line_ire, vsync_ire - 10, vsync_ire / 2):
                vsync_lines += 1
            elif inrange(line_ire, -5, 5):
                score += 1
            else:
                score -= 1

        if vsync_lines >= 2:
            return 100

        if vsync_lines == 1 and score > 0:
            return 50

        if score > 0:
            return 25

        return 0

    # pull the above together into a routine that (should) find line 0, the last line of
    # the previous field.

    def getLine0(self, validpulses):
        # Gather the local line 0 location and projected from the previous field

        self.sync_confidence = 100

        # If we have a previous field, the first vblank should be close to the beginning,
        # and we need to reject anything too far in (which could be the *next* vsync)
        limit = None
        limit = 100 if (self.prevfield is not None and self.prevfield.skip_check() >= 50) else None
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

                meanlinelen = self.computeLineLen(validpulses)
                fieldlen = (
                    meanlinelen
                    * self.rf.SysParams["field_lines"][0 if isFirstField_next else 1]
                )
                line0loc_next = int(np.round(self.vblank_next - fieldlen))

                if line0loc_next < 0:
                    self.sync_confidence = 10
        else:
            self.vblank_next = None

        # Use the previous field's end to compute a possible line 0
        line0loc_prev, isFirstField_prev = None, None
        if self.prevfield is not None and self.prevfield.valid:
            frameoffset = self.data["startloc"] - self.prevfield.data["startloc"]

            line0loc_prev = (
                self.prevfield.linelocs[self.prevfield.linecount] - frameoffset
            )
            isFirstField_prev = not self.prevfield.isFirstField
            conf_prev = self.prevfield.sync_confidence

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
        pulse_hz_min = self.rf.iretohz(self.rf.SysParams["vsync_ire"] - 10)
        pulse_hz_max = self.rf.iretohz(self.rf.SysParams["vsync_ire"] / 2)

        pulses = findpulses(self.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max)

        if len(pulses) == 0:
            # can't do anything about this
            return pulses

        # determine sync pulses from vsync
        vsync_locs = []
        vsync_means = []

        for i, p in enumerate(pulses):
            if p.len > self.usectoinpx(10):
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

        if len(vsync_means) == 0:
            return None

        synclevel = np.median(vsync_means)

        if np.abs(self.rf.hztoire(synclevel) - self.rf.SysParams["vsync_ire"]) < 5:
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

        pulse_hz_min = synclevel - (self.rf.SysParams["hz_ire"] * 10)
        pulse_hz_max = (blacklevel + synclevel) / 2

        return findpulses(self.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max)

    def compute_linelocs(self):

        self.rawpulses = self.getpulses()
        if self.rawpulses is None or len(self.rawpulses) == 0:
            logger.error("Unable to find any sync pulses, jumping 100 ms")
            return None, None, int(self.rf.freq_hz/10)

        self.validpulses = validpulses = self.refinepulses()

        line0loc, lastlineloc, self.isFirstField = self.getLine0(validpulses)
        self.linecount = 263 if self.isFirstField else 262

        # Number of lines to actually process.  This is set so that the entire following
        # VSYNC is processed
        proclines = self.outlinecount + self.lineoffset + 10
        if self.rf.system == 'PAL':
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
            if self.initphase == False:
                logger.error("Unable to determine start of field - dropping field")

            return None, None, self.inlinelen * 200

        meanlinelen = self.computeLineLen(validpulses)

        # If we don't have enough data at the end, move onto the next field
        lastline = ((self.rawpulses[-1].start - line0loc) / meanlinelen)
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
            linelocs_dict[l] if l in linelocs_dict else -1
            for l in range(0, proclines)
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

        if self.vblank_next is None:
            nextfield = linelocs_filled[self.outlinecount - 7]
        else:
            nextfield = self.vblank_next - (self.inlinelen * 8)

        # print('nf', nextfield, self.vblank_next)

        return rv_ll, rv_err, nextfield

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
                self.rf.iretohz(self.rf.SysParams["vsync_ire"] / 2),
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

    def downscale(
        self,
        lineinfo=None,
        linesout=None,
        outwidth=None,
        channel="demod",
        audio=0,
        final=False,
    ):
        if lineinfo is None:
            lineinfo = self.linelocs
        if outwidth is None:
            outwidth = self.outlinelen
        if linesout is None:
            # for video always output 263/313 lines
            linesout = self.outlinecount

        dsout = np.zeros((linesout * outwidth), dtype=np.double)
        # self.lineoffset is an adjustment for 0-based lines *before* downscaling so add 1 here
        lineoffset = self.lineoffset + 1

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
                logger.warning("WARNING: TBC failure at line %d", l)
                dsout[
                    (l - lineoffset) * outwidth : (l + 1 - lineoffset) * outwidth
                ] = self.rf.SysParams["ire0"]

        if audio > 0 and self.rf.decode_analog_audio:
            self.dsaudio, self.audio_next_offset = downscale_audio(
                self.data["audio"],
                lineinfo,
                self.rf,
                self.linecount,
                self.audio_offset,
                freq=audio,
            )

        if self.rf.decode_digital_audio:
            self.efmout = self.data["efm"][
                int(self.linelocs[1]) : int(self.linelocs[self.linecount + 1])
            ]
        else:
            self.efmout = None

        if final:
            dsout = self.hz_to_output(dsout)
            self.dspicture = dsout

        return dsout, self.dsaudio, self.efmout

    def rf_tbc(self, linelocs=None):
        """ This outputs a TBC'd version of the input RF data, mostly intended 
            to assist in audio processing.  Outputs a uint16 array.
        """

        # Convert raw RF to floating point to help the scaler
        fdata = self.data["input"].astype(np.float)

        if linelocs is None:
            linelocs = self.linelocs

        # Ensure that the output line length is an integer
        linelen = int(round(self.inlinelen))

        # Adjust for the demodulation/filtering delays
        delay = self.rf.delays["video_white"]

        # For output consistency reasons, linecount is set to 313 (i.e. 626 lines)
        # in PAL mode.  This needs to be corrected for RF TBC.
        lc = self.linecount
        if self.rf.system == "PAL" and not self.isFirstField:
            lc = 312

        output = []

        for l in range(self.lineoffset, self.lineoffset + lc):
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

    def dropout_detect_demod(self):
        # current field
        f = self

        isPAL = self.rf.system == "PAL"

        rfstd = np.std(f.data["rfhpf"])
        # iserr_rf = np.full(len(f.data['video']['demod']), False, dtype=np.bool)
        iserr_rf1 = (f.data["rfhpf"] < (-rfstd * 3)) | (
            f.data["rfhpf"] > (rfstd * 3)
        )  # | (f.rawdata <= -32000)
        iserr_rf = np.full_like(iserr_rf1, False)
        iserr_rf[self.rf.delays["video_rot"] :] = iserr_rf1[
            : -self.rf.delays["video_rot"]
        ]

        # detect absurd fluctuations in pre-deemp demod, since only dropouts can cause them
        # (current np.diff has a prepend option, but not in ubuntu 18.04's version)
        iserr1 = f.data["video"]["demod_raw"] > self.rf.freq_hz_half
        # This didn't work right for PAL (issue #471)
        # iserr1 |= f.data['video']['demod_hpf'] > 3000000

        # build sets of min/max valid levels
        valid_min = np.full_like(
            f.data["video"]["demod"], f.rf.iretohz(-60 if isPAL else -50)
        )
        valid_max = np.full_like(
            f.data["video"]["demod"], f.rf.iretohz(150 if isPAL else 160)
        )

        # the minimum valid value during VSYNC is lower for PAL because of the pilot signal
        minsync = -100 if isPAL else -50

        iserr2 = f.data["video"]["demod"] < valid_min
        iserr2 |= f.data["video"]["demod"] > valid_max

        valid_min05 = np.full_like(f.data["video"]["demod_05"], f.rf.iretohz(-20))
        valid_max05 = np.full_like(f.data["video"]["demod_05"], f.rf.iretohz(115))

        iserr3 = f.data["video"]["demod_05"] < valid_min05
        iserr3 |= f.data["video"]["demod_05"] > valid_max05

        iserr = iserr1 | iserr2 | iserr3 | iserr_rf

        # Each valid pulse is definitely *not* an error, so exclude it here at the end
        for v in self.validpulses:
            iserr[
                int(v[1].start - self.rf.freq) : int(
                    v[1].start + v[1].len + self.rf.freq
                )
            ] = False

        return iserr

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

    def dropout_errlist_to_tbc(self, errlist):
        dropouts = []

        if len(errlist) == 0:
            return dropouts

        # Now convert the above errlist into TBC locations
        errlistc = errlist.copy()
        curerr = errlistc.pop(0)

        lineoffset = -self.lineoffset

        for l in range(lineoffset, self.linecount + self.lineoffset):
            while curerr is not None and inrange(
                curerr[0], self.linelocs[l], self.linelocs[l + 1]
            ):
                start_rf_linepos = curerr[0] - self.linelocs[l]
                start_linepos = start_rf_linepos / (
                    self.linelocs[l + 1] - self.linelocs[l]
                )
                start_linepos = int(start_linepos * self.outlinelen)

                end_rf_linepos = curerr[1] - self.linelocs[l]
                end_linepos = end_rf_linepos / (self.linelocs[l + 1] - self.linelocs[l])
                end_linepos = int(np.round(end_linepos * self.outlinelen))

                if end_linepos > self.outlinelen:
                    # need to output two dropouts
                    dropouts.append(
                        (l + 1 + lineoffset, start_linepos, self.outlinelen)
                    )
                    dropouts.append(
                        (
                            l + 1 + lineoffset + (end_linepos // self.outlinelen),
                            0,
                            np.remainder(end_linepos, self.outlinelen),
                        )
                    )
                else:
                    dropouts.append((l + 1 + lineoffset, start_linepos, end_linepos))

                if len(errlistc):
                    curerr = errlistc.pop(0)
                else:
                    curerr = None

        return dropouts

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

    def compute_line_bursts(self, linelocs, _line):
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
        threshold = 5 * self.rf.SysParams["hz_ire"]

        burstarea_demod = self.data["video"]["demod"][s + bstart : s + bend]
        burstarea_demod = burstarea_demod - nb_mean(burstarea_demod)

        if nb_absmax(burstarea_demod) > (30 * self.rf.SysParams["hz_ire"]):
            return None, None

        zcburstdiv = (lfreq * fsc_mhz_inv) / 2

        phase_adjust = 0

        # The first pass computes phase_offset, the second uses it to determine
        # the colo(u)r burst phase of the line.
        for passcount in range(2):
            rising_count = 0
            count = 0
            phase_offset = []

            # this subroutine is in utils.py, broken out so it can be JIT'd
            prevalue, zc = clb_findnextburst(
                burstarea, 0, len(burstarea) - 1, threshold
            )

            while zc is not None:
                count += 1

                zc_cycle = ((bstart + zc - s_rem) / zcburstdiv) + phase_adjust
                zc_round = nb_round(zc_cycle)

                phase_offset.append(zc_round - zc_cycle)

                if prevalue < 0:
                    rising_count += not (zc_round % 2)
                else:
                    rising_count += zc_round % 2

                prevalue, zc = clb_findnextburst(
                    burstarea, int(zc + 1), len(burstarea) - 1, threshold
                )

            if count:
                phase_adjust += nb_median(np.array(phase_offset))
            else:
                return None, None

        return (rising_count / count) > 0.5, -phase_adjust


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

        am = angular_mean(zcs)

        for l in range(0, 323):
            linelocs[l] += (phase_distance(zcs[l], am) * plen[l]) * 1

        return np.array(linelocs)

    def get_burstlevel(self, l, linelocs=None):
        lineslice = self.lineslice(l, 5.5, 2.4, linelocs)
        
        burstarea = self.data["video"]["demod"][lineslice].copy()
        burstarea -= nb_mean(burstarea)

        if max(burstarea) > (30 * self.rf.SysParams["hz_ire"]):
            return None

        return rms(burstarea) * np.sqrt(2)

    def calc_burstmedian(self):
        burstlevel = []

        for l in range(11, 313):
            lineburst = self.get_burstlevel(l)
            if lineburst is not None:
                burstlevel.append(lineburst)

        return np.median(burstlevel) / self.rf.SysParams["hz_ire"]

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

        burstlevel6 /= self.rf.SysParams["hz_ire"]

        if inrange(burstlevel6, self.burstmedian * 0.8, self.burstmedian * 1.2):
            hasburst = True
        elif burstlevel6 < self.burstmedian * 0.2:
            hasburst = False
        else:
            return self.get_following_field_number()

        m4 = map4[(self.isFirstField, hasburst)]

        # Now compute if it's field 1-4 or 5-8.

        for l in range(7, 20, 4):
            # Usually line 7 is used to determine burst phase, but
            # if that's corrupt every fourth line has the same phase
            rising, phase_adjust = self.compute_line_bursts(self.linelocs, l)
            if rising is not None:
                break

        if rising == None:
            return self.get_following_field_number()

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
            100 - self.rf.SysParams["vsync_ire"]
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

        self.linecount = 313  # if self.isFirstField else 313
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
        #  qreal tc1 = (((line[h + 2] + line[h - 2]) / 2) - line[h]);

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
        except:
            return 0

    def compute_burst_offsets(self, linelocs):
        rising_sum = 0
        adjs = {}

        for l in range(0, 266):
            rising, phase_adjust = self.compute_line_bursts(linelocs, l)
            if rising == None:
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
                except:
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
        if final == False:
            if "audio" in kwargs:
                kwargs["audio"] = 0

        dsout, dsaudio, dsefm = super(FieldNTSC, self).downscale(
            final=final, *args, **kwargs
        )

        return dsout, dsaudio, dsefm

    def calc_burstmedian(self):
        burstlevel = [self.get_burstlevel(l) for l in range(11, 264)]

        return np.median(burstlevel) / self.rf.SysParams["hz_ire"]

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
            100 - self.rf.SysParams["vsync_ire"]
        )

        if not self.valid:
            return

        self.linecode = [
            self.decodephillipscode(l + self.lineoffset) for l in [16, 17, 18]
        ]

        self.linelocs3 = self.refine_linelocs_burst(self.linelocs2)
        self.linelocs3 = self.fix_badlines(self.linelocs3, self.linelocs2)

        self.burstmedian = self.calc_burstmedian()

        # Now adjust 33 degrees to get the downscaled image onto I/Q color axis
        # self.linelocs = np.array(self.linelocs3) + ((33/360.0) * (63.555555/227.5) * self.rf.freq)
        # Now adjust 33 degrees (-90 - 33) for color decoding
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
        extra_options={},
    ):
        global logger
        self.logger = _logger
        logger = self.logger
        self.demodcache = None

        self.branch, self.commit = get_git_info()

        self.infile = open(fname_in, "rb")
        self.freader = freader

        self.est_frames = est_frames

        self.numthreads = threads

        self.fields_written = 0

        self.blackIRE = 0

        self.analog_audio = int(analog_audio * 1000)
        self.digital_audio = digital_audio
        self.write_rf_tbc = extra_options.get("write_RF_TBC", False)

        self.has_analog_audio = True
        if system == "PAL":
            if analog_audio == 0:
                self.has_analog_audio = False

        self.outfile_json = None

        self.lastvalidfield = {False: None, True: None}

        self.outfile_video = None
        self.outfile_audio = None
        self.outfile_efm = None
        self.outfile_pre_efm = None
        self.ffmpeg_rftbc, self.outfile_rftbc = None, None

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

        self.pipe_rftbc = extra_options.get("pipe_RF_TBC", None)

        self.fname_out = fname_out

        self.firstfield = None  # In frame output mode, the first field goes here
        self.fieldloc = 0

        self.system = system
        self.rf = RFDecode(
            system=system,
            decode_analog_audio=analog_audio,
            decode_digital_audio=digital_audio,
            has_analog_audio=self.has_analog_audio,
            extra_options=extra_options,
        )

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
        self.audio_offset = 0
        self.mtf_level = 1

        self.prevfield = None
        self.curfield = None

        self.doDOD = doDOD

        self.badfields = None

        self.fieldinfo = []

        self.leadIn = False
        self.leadOut = False
        self.isCLV = False
        self.frameNumber = None

        self.autoMTF = True
        self.useAGC = extra_options.get("useAGC", True)

        self.verboseVITS = False

        self.demodcache = DemodCache(
            self.rf, self.infile, self.freader, num_worker_threads=self.numthreads
        )

        self.bw_ratios = []

    def __del__(self):
        del self.demodcache

    def close(self):
        """ deletes all open files, so it's possible to pickle an LDDecode object """

        try:
            self.ffmpeg_rftbc.kill()
        except:
            pass

        # use setattr to force file closure by unlinking the objects
        for outfiles in [
            "infile",
            "outfile_video",
            "outfile_audio",
            "outfile_json",
            "outfile_efm",
            "outfile_rftbc",
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

    def detectLevels(self, field):
        # Returns sync level and ire0 level of a field, computed from HSYNC areas

        sync_hzs = []
        ire0_hzs = []

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

        return np.median(sync_hzs), np.median(ire0_hzs)

    def writeout(self, dataset):
        f, fi, picture, audio, efm = dataset

        if self.digital_audio == True:
            if self.outfile_pre_efm is not None:
                self.outfile_pre_efm.write(efm.tobytes())
            efm_out = self.efm_pll.process(efm)
            self.outfile_efm.write(efm_out.tobytes())

        fi["audioSamples"] = 0 if audio is None else int(len(audio) / 2)
        fi["efmTValues"] = len(efm_out) if self.digital_audio else 0

        self.fieldinfo.append(fi)

        self.outfile_video.write(picture)
        self.fields_written += 1

        if self.outfile_rftbc is not None or self.pipe_rftbc is not None:
            rftbc = f.rf_tbc()

            if self.outfile_rftbc is not None:
                self.outfile_rftbc.write(rftbc)

            if self.pipe_rftbc is not None:
                self.pipe_rftbc.write(rftbc)

        if audio is not None and self.outfile_audio is not None:
            self.outfile_audio.write(audio)

    def decodefield(self, initphase=False):
        """ returns field object if valid, and the offset to the next decode """
        self.readloc = int(self.fdoffset - self.rf.blockcut)
        if self.readloc < 0:
            self.readloc = 0

        self.readloc_block = self.readloc // self.blocksize
        self.numblocks = (self.readlen // self.blocksize) + 2

        self.rawdecode = self.demodcache.read(
            self.readloc_block * self.blocksize,
            self.numblocks * self.blocksize,
            self.mtf_level,
        )

        if self.rawdecode is None:
            #logger.info("Failed to demodulate data")
            return None, None

        self.indata = self.rawdecode["input"]

        f = self.FieldClass(
            self.rf,
            self.rawdecode,
            audio_offset=self.audio_offset,
            prevfield=self.curfield,
            initphase=initphase,
        )

        try:
            f.process()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise e

        if not f.valid:
            # logger.info("Bad data - jumping one second")
            return f, f.nextfieldoffset

        return f, f.nextfieldoffset - (self.readloc - self.rawdecode["startloc"])

    def readfield(self, initphase=False):
        # pretty much a retry-ing wrapper around decodefield with MTF checking
        self.prevfield = self.curfield
        done = False
        adjusted = False
        redo = False

        while done == False:
            if redo:
                # Only allow one redo, no matter what
                done = True

            self.fieldloc = self.fdoffset
            f, offset = self.decodefield(initphase=initphase)

            if f is None:
                if offset is None:
                    # EOF, probably
                    return None

            self.fdoffset += offset

            if f is not None and f.valid:
                picture, audio, efm = f.downscale(
                    linesout=self.output_lines, final=True, audio=self.analog_audio
                )

                self.audio_offset = f.audio_next_offset

                metrics = self.computeMetrics(f, None, verbose=True)
                if "blackToWhiteRFRatio" in metrics and adjusted == False:
                    keep = 900 if self.isCLV else 30
                    self.bw_ratios.append(metrics["blackToWhiteRFRatio"])
                    self.bw_ratios = self.bw_ratios[-keep:]

                redo = not self.checkMTF(f, self.prevfield)

                # Perform AGC changes on first fields only to prevent luma mismatch intra-field
                if self.useAGC and f.isFirstField and f.sync_confidence > 80:
                    sync_hz, ire0_hz = self.detectLevels(f)
                    vsync_ire = self.rf.SysParams["vsync_ire"]

                    sync_ire_diff = np.abs(self.rf.hztoire(sync_hz) - vsync_ire)
                    ire0_diff = np.abs(self.rf.hztoire(ire0_hz))

                    acceptable_diff = 2 if self.fields_written else 0.5

                    if max(sync_ire_diff, ire0_diff) > acceptable_diff:
                        redo = True
                        self.rf.SysParams["ire0"] = ire0_hz
                        # Note that vsync_ire is a negative number, so (sync_hz - ire0_hz) is correct
                        self.rf.SysParams["hz_ire"] = (sync_hz - ire0_hz) / vsync_ire

                if adjusted == False and redo == True:
                    self.demodcache.flush_demod()
                    adjusted = True
                    self.fdoffset -= offset
                else:
                    done = True
            else:
                # Probably jumping ahead - delete the previous field so
                # TBC computations aren't thrown off
                if self.curfield is not None and self.badfields is None:
                    self.badfields = (self.curfield, f)
                self.curfield = None

        if f is None or f.valid == False:
            return None

        self.curfield = f

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
                    return decodeBCD(l & 0x7FFFF)
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
            metrics["greyIRE"] = np.mean(f.output_to_ire(f.dspicture[wl_slice]))
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
        metrics["greyIRE"] = np.mean(f.output_to_ire(f.dspicture[ire50_slice]))

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
            whitelocs = [(20, 14, 12), (20, 52, 8), (13, 13, 15)]  # , (20, 13, 2)]
        else:
            self.computeMetricsPAL(metrics, f, fp)
            whitelocs = [(19, 12, 8)]

        # FIXME: these should probably be computed in the Field class
        f.whitesnr_slice = None

        for l in whitelocs:
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

    def buildmetadata(self, f):
        """ returns field information JSON and whether or not a backfill field is needed """
        prevfi = self.fieldinfo[-1] if len(self.fieldinfo) else None

        fi = {
            "isFirstField": True if f.isFirstField else False,
            "syncConf": f.compute_syncconf(),
            "seqNo": len(self.fieldinfo) + 1,
            #'audioSamples': 0 if audio is None else int(len(audio) / 2),
            "diskLoc": np.round((self.fieldloc / self.bytes_per_field) * 10) / 10,
            "fileLoc": np.floor(self.fieldloc),
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
            if not (
                (
                    fi["fieldPhaseID"] == 1
                    and prevfi["fieldPhaseID"] == f.rf.SysParams["fieldPhases"]
                )
                or (fi["fieldPhaseID"] == prevfi["fieldPhaseID"] + 1)
            ):
                logger.warning(
                    "Field phaseID sequence mismatch ({0}->{1}) (player may be paused)".format(
                        prevfi["fieldPhaseID"], fi["fieldPhaseID"]
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
        fi["vitsMetrics"] = self.computeMetrics(self.curfield, self.prevfield)

        fi["vbi"] = {"vbiData": [int(lc) for lc in f.linecode if lc is not None]}

        self.frameNumber = None
        if f.isFirstField:
            self.firstfield = f
        else:
            # use a stored first field, in case we start with a second field
            if self.firstfield is not None:
                # process VBI frame info data
                self.frameNumber = self.decodeFrameNumber(self.firstfield, f)

                rawloc = np.floor((self.readloc / self.bytes_per_field) / 2)

                disk_Type = "CLV" if self.isCLV else "CAV"
                disk_TimeCode = None
                disk_Frame = None
                special = None

                try:
                    if self.isCLV and self.earlyCLV:  # early CLV
                        disk_TimeCode = f"{self.clvMinutes}:xx"
                    # print("file frame %d early-CLV minute %d" % (rawloc, self.clvMinutes), file=sys.stderr)
                    elif self.isCLV and self.frameNumber is not None:
                        disk_TimeCode = "CLV Timecode %d:%.2d.%.2d frame %d" % (
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
                        special = "Unknown"

                    if self.est_frames is not None:
                        outstr = f"Frame {(self.fields_written//2)+1}/{int(self.est_frames)}: File Frame {int(rawloc)}: {disk_Type} "
                    else:
                        outstr = f"File Frame {int(rawloc)}: {disk_Type} "
                    if self.isCLV:
                        outstr += f"Timecode {disk_TimeCode} "
                    else:
                        outstr += f"Frame #{disk_Frame} "

                    if special is not None:
                        outstr += special

                    self.logger.status(outstr)

                    # Prepare JSON fields
                    if self.frameNumber is not None:
                        fi["frameNumber"] = int(self.frameNumber)

                    if self.verboseVITS and self.isCLV and self.clvMinutes is not None:
                        fi["clvMinutes"] = int(self.clvMinutes)
                        if not self.earlyCLV:
                            fi["clvSeconds"] = int(self.clvSeconds)
                            fi["clvFrameNr"] = int(self.clvFrameNum)
                except:
                    logger.warning("file frame %d : VBI decoding error", rawloc)

        return fi, False

    def seek_getframenr(self, startfield):
        """ Reads from file location startfield, returns first VBI frame # or None on failure and revised startfield """

        """ Note that if startfield is not 0, and the read fails, it will automatically retry
            at file location 0
        """

        self.roughseek(startfield)

        for fields in range(10):
            self.fieldloc = self.fdoffset
            f, offset = self.decodefield(initphase=True)

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
                self.prevfield = self.curfield
                self.curfield = f
                self.fdoffset += offset

                # Two fields are needed to be sure to have sufficient Philips code data
                # to determine frame #.
                if self.prevfield is not None and f.valid:
                    fnum = self.decodeFrameNumber(self.prevfield, self.curfield)

                    if self.earlyCLV:
                        logger.error("Cannot seek in early CLV disks w/o timecode")
                        return None, startfield
                    elif fnum is not None:
                        rawloc = np.floor((self.readloc / self.bytes_per_field) / 2)
                        logger.info("seeking: file loc %d frame # %d", rawloc, fnum)

                        # Clear field memory on seeks
                        self.prevfield = None
                        self.curfield = None

                        return fnum, startfield

        return None, None

    def seek(self, startframe, target):
        """ Attempts to find frame target from file location startframe """
        logger.info("Beginning seek")

        if not sys.warnoptions:
            import warnings

            warnings.simplefilter("ignore")

        curfield = startframe * 2

        for retries in range(3):
            fnr, curfield = self.seek_getframenr(curfield)
            if fnr is None:
                return None

            cur = int((self.fieldloc / self.bytes_per_field))
            if fnr == target:
                logger.info("Finished seek")
                print("Finished seeking, starting at frame", fnr, file=sys.stderr)
                self.roughseek(cur)
                return cur

            curfield += ((target - fnr) * 2) - 1

        return None

    def build_json(self, f):
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

        if f is None:
            return

        vp["gitBranch"] = self.branch
        vp["gitCommit"] = self.commit

        vp["isSourcePal"] = True if f.rf.system == "PAL" else False

        vp["fsc"] = int(f.rf.SysParams["fsc_mhz"] * 1000000)
        vp["fieldWidth"] = f.rf.SysParams["outlinelen"]
        vp["sampleRate"] = vp["fsc"] * 4
        spu = vp["sampleRate"] / 1000000

        vp["black16bIre"] = np.float(f.hz_to_output(f.rf.iretohz(self.blackIRE)))
        vp["white16bIre"] = np.float(f.hz_to_output(f.rf.iretohz(100)))

        vp["fieldHeight"] = f.outlinecount

        # current burst adjustment as of 2/27/19, update when #158 is fixed!
        badj = -1.4
        vp["colourBurstStart"] = np.round(
            (f.rf.SysParams["colorBurstUS"][0] * spu) + badj
        )
        vp["colourBurstEnd"] = np.round(
            (f.rf.SysParams["colorBurstUS"][1] * spu) + badj
        )
        vp["activeVideoStart"] = np.round(
            (f.rf.SysParams["activeVideoUS"][0] * spu) + badj
        )
        vp["activeVideoEnd"] = np.round(
            (f.rf.SysParams["activeVideoUS"][1] * spu) + badj
        )

        jout["videoParameters"] = vp

        jout["fields"] = self.fieldinfo.copy()

        return jout
