import math
import numpy as np
import scipy.signal as sps
import copy

import itertools

import lddecode.core as ldd
import lddecode.utils as lddu
from lddecode.utils import inrange
from vhsdecode.utils import get_line

import vhsdecode.formats as vhs_formats
from vhsdecode.addons.chromasep import ChromaSepClass
from vhsdecode.addons.resync import DCrestore
from vhsdecode.addons.vsync import Vsync

# Use PyFFTW's faster FFT implementation if available
try:
    import pyfftw.interfaces.numpy_fft as npfft
    import pyfftw.interfaces

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)
except ImportError:
    import numpy.fft as npfft


def chroma_to_u16(chroma):
    """Scale the chroma output array to a 16-bit value for output."""
    S16_ABS_MAX = 32767

    if np.max(chroma) > S16_ABS_MAX or abs(np.min(chroma)) > S16_ABS_MAX:
        ldd.logger.warning("Chroma signal clipping.")
    return np.uint16(chroma + S16_ABS_MAX)


def find_sync_levels(field):
    """Very crude sync level detection"""
    # Skip a few samples to avoid any possible edge distortion.
    data = field.data["video"]["demod_05"][10:]

    # Start with finding the minimum value of the input.
    sync_min = min(data)
    max_val = max(data)

    # Use the max for a temporary reference point which may be max ire or not.
    difference = max_val - sync_min

    # Find approximate sync areas.
    on_sync = data < (sync_min + (difference / 15))

    found_porch = False

    offset = 0
    blank_level = None

    while not found_porch:
        # Look for when we leave the approximate sync area next...
        search_start = np.argwhere(on_sync[offset:])[0][0]
        next_cross_raw = np.argwhere(1 - on_sync[search_start:])[0][0] + search_start
        # and a bit past that we ought to be in the back porch for blanking level.
        next_cross = next_cross_raw + int(field.usectoinpx(1.5))
        blank_level = data[next_cross] + offset
        if blank_level > sync_min + (difference / 15):
            found_porch = True
        else:
            # We may be in vsync, try to skip ahead a bit
            # TODO: This may not work yet.
            offset += field.usectoinpx(50)
            if offset > len(data) - 10:
                # Give up
                return None, None

    if False:
        import matplotlib.pyplot as plt

        data = field.data["video"]["demod_05"]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
        #        ax1.plot(hilbert, color='#FF0000')
        # ax1.plot(data, color="#00FF00")
        ax1.axhline(sync_min, color="#0000FF")
        #        ax1.axhline(blank_level, color="#000000")
        ax1.axvline(search_start, color="#FF0000")
        ax1.axvline(next_cross_raw, color="#00FF00")
        ax1.axvline(next_cross, color="#0000FF")
        ax1.axhline(blank_level, color="#000000")
        #            ax1.axhline(self.iretohz(self.SysParams["vsync_ire"]))
        #            ax1.axhline(self.iretohz(7.5))
        #            ax1.axhline(self.iretohz(100))
        # print("Vsync IRE", self.SysParams["vsync_ire"])
        #            ax2 = ax1.twinx()
        #            ax3 = ax1.twinx()
        ax1.plot(data)
        ax2.plot(on_sync)
        #            ax2.plot(luma05[:2048])
        #            ax4.plot(env, color="#00FF00")
        #            ax3.plot(np.angle(hilbert))
        #            ax4.plot(hilbert.imag)
        #            crossings = find_crossings(env, 700)
        #            ax3.plot(crossings, color="#0000FF")
        plt.show()
        #            exit(0)

    return sync_min, blank_level


def getpulses_override(field):
    """Find sync pulses in the demodulated video sigal

    NOTE: TEMPORARY override until an override for the value itself is added upstream.
    """

    # Ignore this ATM, the current code does it better.
    field.rf.Vsync.work(field.data["video"]["demod_05"])

    if field.rf.auto_sync:
        sync_level, blank_level = find_sync_levels(field)

        if sync_level is not None and blank_level is not None:
            field.rf.SysParams["ire0"] = blank_level
            field.rf.SysParams["hz_ire"] = (blank_level - sync_level) / (
                -field.rf.SysParams["vsync_ire"]
            )

        if False:
            import matplotlib.pyplot as plt

            data = field.data["video"]["demod_05"]

            fig, ax1 = plt.subplots(1, 1, sharex=True)
            # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
            #        ax1.plot(hilbert, color='#FF0000')
            # ax1.plot(data, color="#00FF00")
            ax1.axhline(field.rf.iretohz(0), color="#000000")
            #        ax1.axhline(blank_level, color="#000000")
            ax1.axhline(field.rf.iretohz(field.rf.SysParams["vsync_ire"]))
            #            ax1.axhline(self.iretohz(self.SysParams["vsync_ire"]))
            #            ax1.axhline(self.iretohz(7.5))
            #            ax1.axhline(self.iretohz(100))
            # print("Vsync IRE", self.SysParams["vsync_ire"])
            #            ax2 = ax1.twinx()
            #            ax3 = ax1.twinx()
            ax1.plot(data)
            #            ax2.plot(luma05[:2048])
            #            ax4.plot(env, color="#00FF00")
            #            ax3.plot(np.angle(hilbert))
            #            ax4.plot(hilbert.imag)
            #            crossings = find_crossings(env, 700)
            #            ax3.plot(crossings, color="#0000FF")
            plt.show()
            #            exit(0)

    # pass one using standard levels

    # pulse_hz range:  vsync_ire - 10, maximum is the 50% crossing point to sync
    pulse_hz_min = field.rf.iretohz(field.rf.SysParams["vsync_ire"] - 10)
    pulse_hz_max = field.rf.iretohz(field.rf.SysParams["vsync_ire"] / 2)

    pulses = lddu.findpulses(
        field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max
    )

    if len(pulses) == 0:
        # can't do anything about this
        return pulses

    # determine sync pulses from vsync
    vsync_locs = []
    vsync_means = []

    for i, p in enumerate(pulses):
        if p.len > field.usectoinpx(10):
            vsync_locs.append(i)
            vsync_means.append(
                np.mean(
                    field.data["video"]["demod_05"][
                        int(p.start + field.rf.freq) : int(
                            p.start + p.len - field.rf.freq
                        )
                    ]
                )
            )

    if len(vsync_means) == 0:
        return None

    synclevel = np.median(vsync_means)

    if np.abs(field.rf.hztoire(synclevel) - field.rf.SysParams["vsync_ire"]) < 5:
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
        if inrange(p.len, field.rf.freq * 0.75, field.rf.freq * 3):
            black_means.append(
                np.mean(
                    field.data["video"]["demod_05"][
                        int(p.start + (field.rf.freq * 5)) : int(
                            p.start + (field.rf.freq * 20)
                        )
                    ]
                )
            )

    blacklevel = np.median(black_means)

    pulse_hz_min = synclevel - (field.rf.SysParams["hz_ire"] * 10)
    pulse_hz_max = (blacklevel + synclevel) / 2

    return lddu.findpulses(field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max)


def get_burst_area(field):
    return (
        math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0])),
        math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])),
    )


class LineInfo:
    """Helper class to store line burst info for PAL."""

    def __init__(self, num):
        self.linenum = num
        self.bp = 0
        self.bq = 0
        self.vsw = -1
        self.burst_norm = 0

    def __str__(self):
        return "<num: %s, bp: %s, bq: %s, vsw: %s, burst_norm: %s>" % (
            self.linenum,
            self.bp,
            self.bq,
            self.vsw,
            self.burst_norm,
        )


def mean_of_burst_sums(chroma_data, line_length, lines, burst_start, burst_end):
    """Sum the burst areas of two and two lines together, and return the mean of these sums."""
    IGNORED_LINES = 16

    burst_sums = []

    # We ignore the top and bottom 16 lines. The top will typically not have a color burst, and
    # the bottom 16 may be after or at the head switch where the phase rotation will be different.
    start_line = IGNORED_LINES
    end_line = lines - IGNORED_LINES

    for line_number in range(start_line, end_line, 2):
        burst_a = get_line(chroma_data, line_length, line_number)[burst_start:burst_end]
        burst_b = get_line(chroma_data, line_length, line_number + 1)[
            burst_start:burst_end
        ]

        # Use the absolute of the sums to differences cancelling out.
        mean_dev = np.mean(abs(burst_a + burst_b))

        burst_sums.append(mean_dev)

    mean_burst_sum = np.nanmean(burst_sums)
    return mean_burst_sum


def detect_burst_pal(
    chroma_data, sine_wave, cosine_wave, burst_area, line_length, lines
):
    """Decode the burst of most lines to see if we have a valid PAL color burst."""

    # Ignore the first and last 16 lines of the field.
    # first ones contain sync and often doesn't have color burst,
    # while the last lines of the field will contain the head switch and may be distorted.
    IGNORED_LINES = 16
    line_data = []
    burst_norm = np.full(lines, np.nan)
    # Decode the burst vectors on each line and try to get an average of the burst amplitude.
    for linenumber in range(IGNORED_LINES, lines - IGNORED_LINES):
        info = detect_burst_pal_line(
            chroma_data, sine_wave, cosine_wave, burst_area, line_length, linenumber
        )
        line_data.append(info)
        burst_norm[linenumber] = info.burst_norm

    burst_mean = np.nanmean(burst_norm[IGNORED_LINES : lines - IGNORED_LINES])

    return line_data, burst_mean


def detect_burst_pal_line(
    chroma_data, sine, cosine, burst_area, line_length, line_number
):
    """Detect burst function ported from the C++ chroma decoder (palcolour.cpp)

    Tries to decode the PAL chroma vectors from the line's color burst
    """
    empty_line = np.zeros_like(chroma_data[0:line_length])
    num_lines = chroma_data.size / line_length

    # Use an empty line if we try to access outside the field.
    def line_or_empty(line):
        return (
            get_line(chroma_data, line_length, line)
            if line >= 0 and line < num_lines
            else empty_line
        )

    in0 = line_or_empty(line_number)
    in1 = line_or_empty(line_number - 1)
    in2 = line_or_empty(line_number + 1)
    in3 = line_or_empty(line_number - 2)
    in4 = line_or_empty(line_number + 2)
    bp = 0
    bq = 0
    bpo = 0
    bqo = 0

    # (Comment from palcolor.cpp)
    # Find absolute burst phase relative to the reference carrier by
    # product detection.
    #
    # To avoid hue-shifts on alternate lines, the phase is determined by
    # averaging the phase on the current-line with the average of two
    # other lines, one above and one below the current line.
    #
    # For PAL we use the next-but-one line above and below (in the field),
    # which will have the same V-switch phase as the current-line (and 180
    # degree change of phase), and we also analyse the average (bpo/bqo
    # 'old') of the line immediately above and below, which have the
    # opposite V-switch phase (and a 90 degree subcarrier phase shift).
    for i in range(burst_area[0], burst_area[1]):
        bp += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * sine[i]
        bq += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * cosine[i]
        bpo += ((in2[i] - in1[i]) / 2.0) * sine[i]
        bqo += ((in2[i] - in1[i]) / 2.0) * cosine[i]

    # (Comment from palcolor.cpp)
    # Normalise the sums above
    burst_length = burst_area[1] - burst_area[0]

    bp /= burst_length
    bq /= burst_length
    bpo /= burst_length
    bqo /= burst_length

    # (Comment from palcolor.cpp)
    # Detect the V-switch state on this line.
    # I forget exactly why this works, but it's essentially comparing the
    # vector magnitude /difference/ between the phases of the burst on the
    # present line and previous line to the magnitude of the burst. This
    # may effectively be a dot-product operation...
    line = LineInfo(line_number)
    if ((bp - bpo) * (bp - bpo) + (bq - bqo) * (bq - bqo)) < (bp * bp + bq * bq) * 2:
        line.vsw = 1

    # (Comment from palcolor.cpp)
    # Average the burst phase to get -U (reference) phase out -- burst
    # phase is (-U +/-V). bp and bq will be of the order of 1000.
    line.bp = (bp - bqo) / 2
    line.bq = (bq + bpo) / 2

    # (Comment from palcolor.cpp)
    # Normalise the magnitude of the bp/bq vector to 1.
    # Kill colour if burst too weak.
    # XXX magic number 130000 !!! check!
    burst_norm = max(math.sqrt(line.bp * line.bp + line.bq * line.bq), 130000.0 / 128)
    line.burst_norm = burst_norm
    line.bp /= burst_norm
    line.bq /= burst_norm

    return line


def check_increment_field_no(rf):
    """Increment field number if the raw data location moved significantly since the last call"""
    raw_loc = rf.decoder.readloc / rf.decoder.bytes_per_field

    if rf.last_raw_loc is None:
        rf.last_raw_loc = raw_loc

    if raw_loc > rf.last_raw_loc:
        rf.field_number += 1
    else:
        ldd.logger.info("Raw data loc didn't advance.")


class FieldPALCVBS(ldd.FieldPAL):
    def __init__(self, *args, **kwargs):
        super(FieldPALCVBS, self).__init__(*args, **kwargs)

    def refine_linelocs_pilot(self, linelocs=None):
        """Override this as most sources won't have a pilot burst."""
        if linelocs is None:
            linelocs = self.linelocs2.copy()
        else:
            linelocs = linelocs.copy()

        return linelocs

    def _determine_field_number(self):
        """Using LD code as it should work on stable sources, but may not work on stuff like vhs."""
        return 1 + (self.rf.field_number % 8)

    def getpulses(self):
        """Find sync pulses in the demodulated video sigal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        return getpulses_override(self)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as tapes have large variations in line pos
        Due to e.g head switch.
        compute errors based off the second derivative - if it exceeds 1 something's wrong,
        and if 4 really wrong...
        """
        return baserr

    def dropout_detect(self):
        return None


class FieldNTSCCVBS(ldd.FieldNTSC):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCCVBS, self).__init__(*args, **kwargs)

    def _refine_linelocs_burst(self, linelocs=None):
        """Standard impl works for stable sources, we may need to override this for
        unstable ones though.
        """
        if linelocs is None:
            linelocs = self.linelocs2
        else:
            linelocs = linelocs.copy()

        return linelocs

    def dropout_detect(self):
        return None

    def getpulses(self):
        """Find sync pulses in the demodulated video sigal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        return getpulses_override(self)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as line starts can vary widely."""
        return baserr


# Superclass to override laserdisc-specific parts of ld-decode with stuff that works for VHS
#
# We do this simply by using inheritance and overriding functions. This results in some redundant
# work that is later overridden, but avoids altering any ld-decode code to ease merging back in
# later as the ld-decode is in flux at the moment.
class CVBSDecode(ldd.LDdecode):
    def __init__(
        self,
        fname_in,
        fname_out,
        freader,
        logger,
        system="NTSC",
        threads=1,
        inputfreq=40,
        level_adjust=0.2,
        rf_options={},
        extra_options={},
    ):
        super(CVBSDecode, self).__init__(
            fname_in,
            fname_out,
            freader,
            logger,
            analog_audio=False,
            system=system,
            doDOD=False,
            threads=threads,
            extra_options=extra_options,
        )
        # Adjustment for output to avoid clipping.
        self.level_adjust = level_adjust
        # Overwrite the rf decoder with the VHS-altered one
        self.rf = VHSDecodeInner(
            system=system,
            tape_format="UMATIC",
            inputfreq=inputfreq,
            rf_options=rf_options,
        )

        # Store reference to ourself in the rf decoder - needed to access data location for track
        # phase, may want to do this in a better way later.
        self.rf.decoder = self
        if system == "PAL":
            self.FieldClass = FieldPALCVBS
        elif system == "NTSC":
            self.FieldClass = FieldNTSCCVBS
        else:
            raise Exception("Unknown video system!", system)

        self.demodcache = ldd.DemodCache(
            self.rf, self.infile, self.freader, num_worker_threads=self.numthreads
        )

    # Override to avoid NaN in JSON.
    def calcsnr(self, f, snrslice):
        data = f.output_to_ire(f.dspicture[snrslice])

        signal = np.mean(data)
        noise = np.std(data)

        # Make sure signal is positive so we don't try to do log on a negative value.
        if signal < 0.0:
            ldd.logger.info(
                "WARNING: Negative mean for SNR, changing to absolute value."
            )
            signal = abs(signal)
        if noise == 0:
            return 0
        return 20 * np.log10(signal / noise)

    def calcpsnr(self, f, snrslice):
        data = f.output_to_ire(f.dspicture[snrslice])

        #        signal = np.mean(data)
        noise = np.std(data)
        if noise == 0:
            return 0
        return 20 * np.log10(100 / noise)

    def buildmetadata(self, f):
        # Avoid crash if this is NaN
        if math.isnan(f.burstmedian):
            f.burstmedian = 0.0
        return super(CVBSDecode, self).buildmetadata(f)

    # For laserdisc this decodes frame numbers from VBI metadata, but there won't be such a thing on
    # other sources, so just skip it for now.
    def decodeFrameNumber(self, f1, f2):
        return None

    # Again ignored for non-ld sources.
    def checkMTF(self, field, pfield=None):
        return True

    def computeMetricsNTSC(self, metrics, f, fp=None):
        return None


class VHSDecodeInner(ldd.RFDecode):
    def __init__(self, inputfreq=40, system="NTSC", tape_format="VHS", rf_options={}):

        # First init the rf decoder normally.
        super(VHSDecodeInner, self).__init__(
            inputfreq, system, decode_analog_audio=False, has_analog_audio=False
        )

        self.chroma_trap = rf_options.get("chroma_trap", False)
        self.notch = rf_options.get("notch", None)
        self.notch_q = rf_options.get("notch_q", 10.0)
        self.auto_sync = rf_options.get("auto_sync", False)

        self.hsync_tolerance = 0.8

        self.field_number = 0
        self.last_raw_loc = None

        # Then we override the laserdisc parameters with VHS ones.
        if system == "PAL":
            self.SysParams = copy.deepcopy(vhs_formats.SysParams_PAL_UMATIC)
            self.DecoderParams = copy.deepcopy(vhs_formats.RFParams_PAL_UMATIC)
        elif system == "NTSC":
            self.SysParams = copy.deepcopy(vhs_formats.SysParams_NTSC_UMATIC)
            self.DecoderParams = copy.deepcopy(vhs_formats.RFParams_NTSC_UMATIC)
        else:
            raise Exception("Unknown video system! ", system)

        # TEMP just set this high so it doesn't mess with anything.
        self.DecoderParams["video_lpf_freq"] = 6800000

        # Lastly we re-create the filters with the new parameters.
        self.computevideofilters()

        self.Filters["FVideo"] = self.Filters["Fvideo_lpf"]
        SF = self.Filters
        SF["FVideo05"] = SF["Fvideo_lpf"] * SF["F05"]

        # Filter to pick out color-under chroma component.
        # filter at about twice the carrier. (This seems to be similar to what VCRs do)
        # TODO: Needs tweaking
        # Note: order will be doubled since we use filtfilt.
        # chroma_lowpass = sps.butter(
        #     2,
        #     [50000 / self.freq_hz_half, DP["chroma_bpf_upper"] / self.freq_hz_half],
        #     btype="bandpass",
        #     output="sos",
        # )
        # self.Filters["FVideoBurst"] = chroma_lowpass

        if self.notch is not None:
            self.Filters["FVideoNotch"] = sps.iirnotch(
                self.notch / self.freq_half, self.notch_q
            )
            # self.Filters["FVideoNotchF"] = lddu.filtfft(
            #     self.Filters["FVideoNotch"], self.blocklen
            # )

        # The following filters are for post-TBC:
        # The output sample rate is at approx 4fsc
        fsc_mhz = self.SysParams["fsc_mhz"]
        out_sample_rate_mhz = fsc_mhz * 4
        out_frequency_half = out_sample_rate_mhz / 2

        # Final band-pass filter for chroma output.
        # Mostly to filter out the higher-frequency wave that results from signal mixing.
        # Needs tweaking.
        # Note: order will be doubled since we use filtfilt.
        chroma_bandpass_final = sps.butter(
            1,
            [
                (fsc_mhz - 0.1) / out_frequency_half,
                (fsc_mhz + 0.1) / out_frequency_half,
            ],
            btype="bandpass",
            output="sos",
        )
        self.Filters["FChromaBpf"] = chroma_bandpass_final

        # Increase the cutoff at the end of blocks to avoid edge distortion from filters
        # making it through.
        self.blockcut_end = 1024
        self.demods = 0

        self.chromaTrap = ChromaSepClass(self.freq_hz, self.SysParams["fsc_mhz"])
        self.Vsync = Vsync(self.freq_hz, self.SysParams)
        self.DCrestore = DCrestore(self.freq_hz, self.SysParams, self.iretohz)


    def computedelays(self, mtf_level=0):
        """Override computedelays
        It's normally used for dropout compensation, but the dropout compensation implementation
        in ld-decode assumes composite color. This function is called even if it's disabled, and
        seems to break with the VHS setup, so we disable it by overriding it for now.
        """
        # Set these to 0 for now, the metrics calculations look for them.
        self.delays = {}
        self.delays["video_sync"] = 0
        self.delays["video_white"] = 0



    def demodblock(self, data=None, mtf_level=0, fftdata=None, cut=False):
        data = npfft.ifft(fftdata).real

        rv = {}

        # applies the Subcarrier trap
        # (this will remove most chroma info)
        if self.chroma_trap:
            luma = self.chromaTrap.work(data)
        else:
            luma = data

        if not self.auto_sync:
            luma += 0xFFFF / 2
            luma /= 4 * 0xFFFF
            luma *= self.iretohz(100)
            luma += self.iretohz(self.SysParams["vsync_ire"])

        if self.notch is not None:
            luma = sps.filtfilt(
                self.Filters["FVideoNotch"][0],
                self.Filters["FVideoNotch"][1],
                luma,
            )

        luma_fft = npfft.rfft(luma)

        luma05_fft = (
            luma_fft * self.Filters["F05"][: (len(self.Filters["F05"]) // 2) + 1]
        )
        luma05 = npfft.irfft(luma05_fft)
        luma05 = np.roll(luma05, -self.Filters["F05_offset"])
        videoburst = npfft.irfft(
            luma_fft * self.Filters["Fburst"][: (len(self.Filters["Fburst"]) // 2) + 1]
        )

        if False:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
            #        ax1.plot(hilbert, color='#FF0000')
            # ax1.plot(data, color="#00FF00")
            ax1.axhline(self.iretohz(0))
            ax1.axhline(self.iretohz(self.SysParams["vsync_ire"]))
            ax1.axhline(self.iretohz(7.5))
            ax1.axhline(self.iretohz(100))
            # print("Vsync IRE", self.SysParams["vsync_ire"])
            #            ax2 = ax1.twinx()
            #            ax3 = ax1.twinx()
            ax1.plot(luma[:2048])
            ax2.plot(luma05[:2048])
            #            ax4.plot(env, color="#00FF00")
            #            ax3.plot(np.angle(hilbert))
            #            ax4.plot(hilbert.imag)
            #            crossings = find_crossings(env, 700)
            #            ax3.plot(crossings, color="#0000FF")
            plt.show()
        #            exit(0)

        video_out = np.rec.array(
            [luma, luma05, videoburst, data],
            names=["demod", "demod_05", "demod_burst", "raw"],
        )

        rv["video"] = (
            video_out[self.blockcut : -self.blockcut_end] if cut else video_out
        )

        return rv
