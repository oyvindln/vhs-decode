import math
import os
import time
import numpy as np
import traceback
import scipy.signal as sps
from collections import namedtuple

import lddecode.core as ldd
from lddecode.core import npfft
import lddecode.utils as lddu
import vhsdecode.utils as utils
from vhsdecode.utils import StackableMA, filtfft
from vhsdecode.chroma import demod_chroma_filt

import vhsdecode.formats as vhs_formats

# from vhsdecode.addons.FMdeemph import FMDeEmphasis
from vhsdecode.addons.FMdeemph import FMDeEmphasisB
from vhsdecode.addons.chromasep import ChromaSepClass
from vhsdecode.addons.resync import Resync
from vhsdecode.addons.chromaAFC import ChromaAFC

from vhsdecode.demod import replace_spikes, unwrap_hilbert, smooth_spikes

from vhsdecode.field import field_class_from_formats
from vhsdecode.video_eq import VideoEQ
from vhsdecode.doc import DodOptions


def parent_system(system):
    if system == "MPAL":
        parent_system = "NTSC"
    elif system == "MESECAM" or system == "SECAM":
        parent_system = "PAL"
    else:
        parent_system = system
    return parent_system


def is_secam(system: str):
    return system == "SECAM" or system == "MESECAM"


def _computefilters_dummy(self):
    self.Filters = {}
    # Needs to be defined here as it's referenced in constructor.
    self.Filters["F05_offset"] = 32

# HACK - override this in a hacky way for now to skip generating some filters we don't use.
# including one that requires > 20 mhz sample rate.
ldd.RFDecode.computefilters = _computefilters_dummy

# Superclass to override laserdisc-specific parts of ld-decode with stuff that works for VHS
#
# We do this simply by using inheritance and overriding functions. This results in some redundant
# work that is later overridden, but avoids altering any ld-decode code to ease merging back in
# later as the ld-decode is in flux at the moment.
class VHSDecode(ldd.LDdecode):
    def __init__(
        self,
        fname_in,
        fname_out,
        freader,
        logger,
        system="NTSC",
        tape_format="VHS",
        doDOD=True,
        threads=1,
        inputfreq=40,
        level_adjust=0,
        rf_options={},
        extra_options={},
        debug_plot=None,
    ):

        super(VHSDecode, self).__init__(
            fname_in,
            fname_out,
            freader,
            logger,
            analog_audio=False,
            system=parent_system(system),
            doDOD=doDOD,
            threads=threads,
            inputfreq=inputfreq,
            extra_options=extra_options,
        )

        # Adjustment for output to avoid clipping.
        self.level_adjust = level_adjust
        # Overwrite the rf decoder with the VHS-altered one
        self.rf = VHSRFDecode(
            system=system,
            tape_format=tape_format,
            inputfreq=inputfreq,
            rf_options=rf_options,
            extra_options=extra_options,
            debug_plot=debug_plot,
        )
        self.rf.chroma_last_field = -1
        self.rf.chroma_tbc_buffer = np.array([])
        # Store reference to ourself in the rf decoder - needed to access data location for track
        # phase, may want to do this in a better way later.
        self.rf.decoder = self
        self.FieldClass = field_class_from_formats(system, tape_format)

        self.demodcache = ldd.DemodCache(
            self.rf,
            self.infile,
            self.freader,
            num_worker_threads=self.numthreads,
        )

        if fname_out is not None:
            self.outfile_chroma = open(fname_out + "_chroma.tbc", "wb")
        else:
            self.outfile_chroma = None

        self.debug_plot = debug_plot

    # Override to avoid NaN in JSON.
    def calcsnr(self, f, snrslice, psnr=False):
        # if dspicture isn't converted to float, this underflows at -40IRE
        data = f.output_to_ire(f.dspicture[snrslice].astype(float))

        signal = np.mean(data) if not psnr else 100
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

    def buildmetadata(self, f):
        if math.isnan(f.burstmedian):
            f.burstmedian = 0.0
        return super(VHSDecode, self).buildmetadata(f, False)

    # For laserdisc this decodes frame numbers from VBI metadata, but there won't be such a thing on
    # VHS, so just skip it.
    def decodeFrameNumber(self, f1, f2):
        return None

    # Again ignored for tapes
    def checkMTF(self, field, pfield=None):
        return True

    def writeout(self, dataset):
        f, fi, (picturey, picturec), audio, efm = dataset

        # Remove fields that are currently not used to cut down on space usage.
        # the qt tools will load them as 0 with the current code
        # if they don't exist.
        if "audioSamples" in fi:
            del fi["audioSamples"]

        if "vbi" in fi:
            del fi["vbi"]

        if "medianBurstIRE" in fi:
            del fi["medianBurstIRE"]

        self.fieldinfo.append(fi)

        self.outfile_video.write(picturey)
        self.outfile_chroma.write(picturec)
        self.fields_written += 1

    def close(self):
        setattr(self, "outfile_chroma", None)
        super(VHSDecode, self).close()

    def computeMetricsPAL(self, metrics, f, fp=None):
        return None

    def computeMetricsNTSC(self, metrics, f, fp=None):
        return None

    def build_json(self, f):
        try:
            if not f:
                # Make sure we don't fail if the last attempted field failed to decode
                # Might be better to fix this elsewhere.
                f = self.prevfield
            jout = super(VHSDecode, self).build_json(f)

            black = jout["videoParameters"]["black16bIre"]
            white = jout["videoParameters"]["white16bIre"]

            if self.rf.color_system == "MPAL":
                # jout["videoParameters"]["isSourcePal"] = True
                # jout["videoParameters"]["isSourcePalM"] = True
                jout["videoParameters"]["system"] = "PAL-M"

            jout["videoParameters"]["black16bIre"] = black * (1 - self.level_adjust)
            jout["videoParameters"]["white16bIre"] = white * (1 + self.level_adjust)
            return jout
        except TypeError as e:
            traceback.print_exc()
            print("Cannot build json: %s" % e)
            return None

    def readfield(self, initphase=False):
        # pretty much a retry-ing wrapper around decodefield with MTF checking
        self.prevfield = self.curfield
        done = False
        adjusted = False
        redo = False

        while done is False:
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

                _ = self.computeMetrics(f, None, verbose=True)
                # if "blackToWhiteRFRatio" in metrics and adjusted == False:
                #     keep = 900 if self.isCLV else 30
                #     self.bw_ratios.append(metrics["blackToWhiteRFRatio"])
                #     self.bw_ratios = self.bw_ratios[-keep:]

                # redo = not self.checkMTF(f, self.prevfield)
                redo = False

                # Perform AGC changes on first fields only to prevent luma mismatch intra-field
                if self.useAGC and f.isFirstField and f.sync_confidence > 80:
                    sync_hz, ire0_hz = self.detectLevels(f)
                    vsync_ire = self.rf.SysParams["vsync_ire"]

                    sync_ire_diff = np.abs(self.rf.hztoire(sync_hz) - vsync_ire)
                    ire0_diff = np.abs(self.rf.hztoire(ire0_hz))

                    acceptable_diff = 2 if self.fields_written else 0.5

                    if max(sync_ire_diff, ire0_diff) > acceptable_diff:
                        redo = True
                        self.rf.AGClevels[0].push(ire0_hz)
                        # Note that vsync_ire is a negative number, so (sync_hz - ire0_hz) is correct
                        self.rf.AGClevels[1].push((sync_hz - ire0_hz) / vsync_ire)

                        self.rf.SysParams["ire0"] = self.rf.AGClevels[0].pull()
                        self.rf.SysParams["hz_ire"] = self.rf.AGClevels[1].pull()

                if adjusted is False and redo is True:
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

        if f is None or f.valid is False:
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


class VHSRFDecode(ldd.RFDecode):
    def __init__(
        self,
        inputfreq=40,
        system="NTSC",
        tape_format="VHS",
        rf_options={},
        extra_options={},
        debug_plot=None,
    ):

        # First init the rf decoder normally.
        super(VHSRFDecode, self).__init__(
            inputfreq,
            parent_system(system),
            decode_analog_audio=False,
            has_analog_audio=False,
            extra_options=extra_options,
        )

        # Store a separate setting for *color* system as opposed to 525/625 line here.
        # TODO: Fix upstream so we don't have to fake tell ld-decode code that we are using ntsc for
        # palm to avoid it throwing errors.
        self._color_system = system

        self._dod_options = DodOptions(
            dod_threshold_p=rf_options.get(
                "dod_threshold_p", vhs_formats.DEFAULT_THRESHOLD_P_DDD
            ),
            dod_threshold_a=rf_options.get("dod_threshold_a", None),
            dod_hysteresis=rf_options.get(
                "dod_hysteresis", vhs_formats.DEFAULT_HYSTERESIS
            ),
        )

        self._chroma_trap = rf_options.get("chroma_trap", False)
        track_phase = None if is_secam(system) else rf_options.get("track_phase", None)
        self._recheck_phase = rf_options.get("recheck_phase", False)
        high_boost = rf_options.get("high_boost", None)
        self._notch = rf_options.get("notch", None)
        self._notch_q = rf_options.get("notch_q", 10.0)
        self._disable_diff_demod = rf_options.get("disable_diff_demod", False)
        self.useAGC = extra_options.get("useAGC", False)
        self.debug = extra_options.get("debug", False)
        # Enable cafc for betamax until proper track detection for it is implemented.
        self._do_cafc = (
            True if tape_format == "BETAMAX" else rf_options.get("cafc", False)
        )
        # cafc requires --recheck_phase
        self._recheck_phase = True if self._do_cafc else self._recheck_phase

        self.detect_track = False
        self.needs_detect = False
        if track_phase is None:
            self.track_phase = 0
            if not is_secam(system):
                self.detect_track = True
                self.needs_detect = True
        elif track_phase == 0 or track_phase == 1:
            self.track_phase = track_phase
        else:
            raise Exception("Track phase can only be 0, 1 or None")
        self.hsync_tolerance = 0.8

        self.field_number = 0
        self.last_raw_loc = None

        self.SysParams, self.DecoderParams = vhs_formats.get_format_params(
            system, tape_format, ldd.logger
        )

        # Make (intentionally) mutable copies of HZ<->IRE levels
        # (NOTE: used by upstream functions, we use a namedtuple to keep const values already)
        self.DecoderParams["ire0"] = self.SysParams["ire0"]
        self.DecoderParams["hz_ire"] = self.SysParams["hz_ire"]
        self.DecoderParams["vsync_ire"] = self.SysParams["vsync_ire"]

        # No idea if this is a common pythonic way to accomplish it but this gives us values that
        # can't be changed later.
        # first depends on IRE/Hz so has to be set after that is properly set.
        self._options = namedtuple(
            "Options",
            [
                "diff_demod_check_value",
                "tape_format",
                "disable_comb",
                "nldeemp",
                "disable_right_hsync",
                "sync_clip",
                "disable_dc_offset",
                "double_lpf",
                "fallback_vsync",
                "saved_levels",
                "y_comb"
            ],
        )(
            self.iretohz(100) * 2,
            tape_format,
            rf_options.get("disable_comb", False),
            rf_options.get("nldeemp", False),
            rf_options.get("disable_right_hsync", False),
            rf_options.get("sync_clip", False),
            rf_options.get("disable_dc_offset", False),
            tape_format == "VHS",
            rf_options.get("fallback_vsync", False),
            rf_options.get("saved_levels", False),
            rf_options.get("y_comb", 0) * self.SysParams["hz_ire"],
        )

        # As agc can alter these sysParams values, store a copy to then
        # initial value for reference.
        self._sysparams_const = namedtuple(
            "SysparamsConst", "hz_ire vsync_hz vsync_ire ire0 vsync_pulse_us"
        )(
            self.SysParams["hz_ire"],
            self.iretohz(self.SysParams["vsync_ire"]),
            self.SysParams["vsync_ire"],
            self.SysParams["ire0"],
            self.SysParams["vsyncPulseUS"],
        )

        self.debug_plot = debug_plot

        # Lastly we re-create the filters with the new parameters.
        self._computevideofilters_b()

        DP = self.DecoderParams

        self._high_boost = (
            high_boost if high_boost is not None else DP["boost_bpf_mult"]
        )

        # controls the sharpness EQ gain
        sharpness_level = (
            rf_options.get("sharpness", vhs_formats.DEFAULT_SHARPNESS) / 100
        )

        self._video_eq = None
        if sharpness_level != 0:
            self._video_eq = VideoEQ(DP, sharpness_level, self.freq_hz)

        # Heterodyning / chroma wave related filter part

        self._chroma_afc = ChromaAFC(
            self.freq_hz,
            DP["chroma_bpf_upper"] / DP["color_under_carrier"],
            self.SysParams,
            self.DecoderParams["color_under_carrier"],
            tape_format=tape_format,
            do_cafc=self._do_cafc,
        )

        self.Filters["FVideoBurst"] = self._chroma_afc.get_chroma_bandpass()

        if self._notch is not None:
            if not self._do_cafc:
                self.Filters["FVideoNotch"] = sps.iirnotch(
                    self._notch / self.freq_half, self._notch_q
                )
            else:
                self.Filters["FVideoNotch"] = sps.iirnotch(
                    self._notch / self._chroma_afc.getOutFreqHalf(), self._notch_q
                )

            self.Filters["FVideoNotchF"] = lddu.filtfft(
                self.Filters["FVideoNotch"], self.blocklen
            )
        else:
            self.Filters["FVideoNotch"] = None, None

        # The following filters are for post-TBC:
        # The output sample rate is 4fsc
        self.Filters["FChromaFinal"] = self._chroma_afc.get_chroma_bandpass_final()
        self.Filters["FBurstNarrow"] = self._chroma_afc.get_burst_narrow()
        self.chroma_heterodyne = self._chroma_afc.getChromaHet()
        self.fsc_wave, self.fsc_cos_wave = self._chroma_afc.getFSCWaves()

        # Increase the cutoff at the end of blocks to avoid edge distortion from filters
        # making it through.
        self.blockcut_end = 1024

        level_detect_divisor = rf_options.get("level_detect_divisor", 1)

        if level_detect_divisor < 1 or level_detect_divisor > 10:
            ldd.logger.warning(
                "Invalid level detect divisor value %s, using default.",
                level_detect_divisor,
            )
            level_detect_divisor = 1
        elif inputfreq / level_detect_divisor < 4:
            ldd.logger.warning(
                "Level detect divisor too high (%s) for input frequency (%s) mhz. Limiting to %s",
                level_detect_divisor,
                inputfreq,
                int(inputfreq // 4),
            )
            level_detect_divisor = int(inputfreq // 4)

        self.resync = Resync(
            self.freq_hz, self.SysParams, divisor=level_detect_divisor, debug=self.debug
        )

        if self._chroma_trap:
            self.chromaTrap = ChromaSepClass(self.freq_hz, self.SysParams["fsc_mhz"])

        if self.useAGC:
            self.AGClevels = StackableMA(
                window_average=self.SysParams["FPS"] / 2
            ), StackableMA(window_average=self.SysParams["FPS"] / 2)

    @property
    def sysparams_const(self):
        return self._sysparams_const

    @property
    def options(self):
        return self._options

    @property
    def notch(self):
        return self._notch

    @property
    def chroma_afc(self):
        return self._chroma_afc

    @property
    def do_cafc(self):
        return self._do_cafc

    @property
    def recheck_phase(self):
        return self._recheck_phase

    @property
    def color_system(self):
        return self._color_system

    @property
    def dod_options(self):
        return self._dod_options

    def computefilters(self):
        # Override the stuff used in lddecode to skip generating filters we don't use.
        self.computevideofilters()
        self.computedelays()

    def computevideofilters(self):
        self.Filters = {}
        # Needs to be defined here as it's referenced in constructor.
        self.Filters["F05_offset"] = 32

    def _computevideofilters_b(self):
        # Use some shorthand to compact the code.
        SF = self.Filters
        DP = self.DecoderParams

        SF["hilbert"] = lddu.build_hilbert(self.blocklen)

        self.Filters["RFVideoRaw"] = lddu.filtfft(
            sps.butter(
                DP["video_bpf_order"],
                [
                    DP["video_bpf_low"] / self.freq_hz_half,
                    DP["video_bpf_high"] / self.freq_hz_half,
                ],
                btype="bandpass",
            ),
            self.blocklen,
        )

        self.Filters["EnvLowPass"] = sps.butter(
            1, [1.0 / self.freq_half], btype="lowpass"
        )

        # Filter for rf before demodulating.
        y_fm = lddu.filtfft(
            sps.butter(
                DP["video_bpf_order"],
                [
                    DP["video_bpf_low"] / self.freq_hz_half,
                    DP["video_bpf_high"] / self.freq_hz_half,
                ],
                btype="bandpass",
            ),
            self.blocklen,
        )

        y_fm_lowpass = lddu.filtfft(
            sps.butter(
                DP["video_lpf_extra_order"],
                [DP["video_lpf_extra"] / self.freq_hz_half],
                btype="lowpass",
            ),
            self.blocklen,
        )

        y_fm_highpass = lddu.filtfft(
            sps.butter(
                DP["video_hpf_extra_order"],
                [DP["video_hpf_extra"] / self.freq_hz_half],
                btype="highpass",
            ),
            self.blocklen,
        )

        self.Filters["RFVideo"] = y_fm * y_fm_lowpass * y_fm_highpass

        self.Filters["RFTop"] = sps.butter(
            1,
            [
                DP["boost_bpf_low"] / self.freq_hz_half,
                DP["boost_bpf_high"] / self.freq_hz_half,
            ],
            btype="bandpass",
            output="sos",
        )

        # Video (luma) main de-emphasis
        db, da = FMDeEmphasisB(self.freq_hz, DP["deemph_gain"], DP["deemph_mid"]).get()
        # Sync de-emphasis
        # db05, da05 = FMDeEmphasis(self.freq_hz, tau=DP["deemph_tau"]).get()

        # db2, da2 = FMDeEmphasisB(
        #     self.freq_hz, 1.5, 1e6, 3 / 4
        # ).get()

        # nlde_lower = (
        #     lddu.filtfft((db2, da2), self.blocklen)
        # )

        video_lpf = sps.butter(
            DP["video_lpf_order"], DP["video_lpf_freq"] / self.freq_hz_half, "low"
        )
        # SF["Fvideo_lpf"] = lddu.filtfft(video_lpf, self.blocklen)
        filter_video_lpf = filtfft(video_lpf, self.blocklen, False)
        SF["Fvideo_lpf"] = video_lpf

        # additional filters:  0.5mhz, used for sync detection.
        # Using an FIR filter here to get a known delay
        F0_5 = sps.firwin(65, [0.5 / self.freq_half], pass_zero=True)
        filter_05 = filtfft((F0_5, [1.0]), self.blocklen, False)

        # SF["F05"] = lddu.filtfft((F0_5, [1.0]), self.blocklen)
        # Defined earlier
        # SF["F05_offset"] = 32

        self.Filters["FEnvPost"] = sps.butter(
            1, [700000 / self.freq_hz_half], btype="lowpass", output="sos"
        )

        filter_deemp = filtfft((db, da), self.blocklen, whole=False)
        self.Filters["FVideo"] = filter_deemp * filter_video_lpf
        if self.options.double_lpf:
            # Double up the lpf to possibly closer emulate
            # lpf in vcr. May add to other formats too later or
            # make more configurable.
            self.Filters["FVideo"] *= filter_video_lpf

        SF["FVideo05"] = filter_video_lpf * filter_deemp * filter_05

        # SF["YNRHighPass"] = sps.butter(
        #     1,
        #     [
        #         (0.5e6) / self.freq_hz_half,
        #     ],
        #     btype="highpass",
        #     output="sos",
        # )

        if self.options.nldeemp:
            SF["NLHighPassF"] = filtfft(
                sps.butter(
                    1,
                    [DP["nonlinear_highpass_freq"] / self.freq_hz_half],
                    btype="highpass",
                ),
                self.blocklen,
                whole=False,
            )

        if self.debug_plot and self.debug_plot.is_plot_requested("deemphasis"):
            from vhsdecode.debug_plot import plot_deemphasis

            plot_deemphasis(self, filter_video_lpf, DP, filter_deemp)

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

    def demodblock(
        self, data=None, mtf_level=0, fftdata=None, cut=False, thread_benchmark=False
    ):
        rv = {}
        demod_start_time = time.time()
        if fftdata is not None:
            indata_fft = fftdata
        elif data is not None:
            indata_fft = npfft.fft(data[: self.blocklen])
        else:
            raise Exception("demodblock called without raw or FFT data")

        if data is None:
            data = npfft.ifft(indata_fft).real

        if self._notch is not None:
            indata_fft = indata_fft * self.Filters["FVideoNotchF"]

        raw_filtered = npfft.ifft(
            indata_fft * self.Filters["RFVideoRaw"] * self.Filters["hilbert"]
        ).real

        # Calculate an evelope with signal strength using absolute of hilbert transform.
        # Roll this a bit to compensate for filter delay, value eyballed for now.
        np.abs(raw_filtered, out=raw_filtered)
        raw_env = np.roll(raw_filtered, 4)
        del raw_filtered
        # Downconvert to single precision for some possible speedup since we don't need
        # super high accuracy for the dropout detection.
        env = utils.filter_simple(raw_env, self.Filters["FEnvPost"]).astype(np.single)
        del raw_env
        env_mean = np.mean(env)

        # Applies RF filters
        indata_fft_filt = indata_fft * self.Filters["RFVideo"]

        # Boost high frequencies in areas where the signal is weak to reduce missed zero crossings
        # on sharp transitions. Using filtfilt to avoid phase issues.
        if len(np.where(env == 0)[0]) == 0:  # checks for zeroes on env
            data_filtered = npfft.ifft(indata_fft_filt)
            high_part = utils.filter_simple(data_filtered, self.Filters["RFTop"]) * (
                (env_mean * 0.9) / env
            )
            del data_filtered
            indata_fft_filt += npfft.fft(high_part * self._high_boost)
        else:
            ldd.logger.warning("RF signal is weak. Is your deck tracking properly?")

        hilbert = npfft.ifft(indata_fft_filt * self.Filters["hilbert"])

        # FM demodulator
        demod = unwrap_hilbert(hilbert, self.freq_hz).real

        if self._chroma_trap:
            # applies the Subcarrier trap
            demod = self.chromaTrap.work(demod)

        # Disabled if sharpness level is zero (default).
        if self._video_eq:
            # applies the video EQ
            demod = self._video_eq.filter_video(demod)

        # If there are obviously out of bounds values, do an extra demod on a diffed waveform and
        # replace the spikes with data from the diffed demod.
        if not self._disable_diff_demod:
            check_value = self.options.diff_demod_check_value

            if np.max(demod[20:-20]) > check_value:
                demod_b = unwrap_hilbert(
                    np.ediff1d(hilbert, to_begin=0), self.freq_hz
                ).real

                demod = replace_spikes(demod, demod_b, check_value)
                del demod_b
                # Not used yet, needs more testing.
                # 2.2 seems to be a sweet spot between reducing spikes and not causing
                # more
                if False:
                    demod = smooth_spikes(demod, check_value * 2.2)
        # applies main deemphasis filter
        demod_fft = npfft.rfft(demod)
        out_video_fft = demod_fft * self.Filters["FVideo"]
        out_video = npfft.irfft(out_video_fft).real
        if self.options.double_lpf:
            # Compensate for phase shift of the extra lpf
            # TODO: What's this supposed to be?
            out_video = np.roll(out_video, 0)

        if self.options.nldeemp:
            # Extract the high frequency part of the signal
            hf_part = npfft.irfft(out_video_fft * self.Filters["NLHighPassF"])
            # Limit it to preserve sharp transitions
            np.clip(
                hf_part,
                self.DecoderParams["nonlinear_highpass_limit_l"],
                self.DecoderParams["nonlinear_highpass_limit_h"],
                out=hf_part,
            )

            # And subtract it from the output signal.
            out_video -= hf_part
            # out_video = hf_part + self.iretohz(50)

        del out_video_fft

        out_video05 = npfft.irfft(demod_fft * self.Filters["FVideo05"]).real
        out_video05 = np.roll(out_video05, -self.Filters["F05_offset"])

        # Filter out the color-under signal from the raw data.
        out_chroma = (
            demod_chroma_filt(
                data,
                self.Filters["FVideoBurst"],
                self.blocklen,
                self.Filters["FVideoNotch"],
                self._notch,
                move=int(10 * (self.freq / 40))
                # TODO: Do we need to tweak move elsewhere too?
                # if cafc is enabled, this filtering will be done after TBC
            )
            if not self._do_cafc
            else data[: self.blocklen]
        )

        if self.debug_plot and self.debug_plot.is_plot_requested("demodblock"):
            from vhsdecode.debug_plot import plot_input_data

            plot_input_data(
                raw_data=data,
                env=env,
                env_mean=env_mean,
                raw_fft=indata_fft,
                filtered_fft=indata_fft_filt,
                demod_video=demod,
                filtered_video=out_video,
                chroma=out_chroma,
                rfdecode=self,
            )

        # demod_burst is a bit misleading, but keeping the naming for compatability.
        video_out = np.rec.array(
            [out_video, out_video05, out_chroma, env],
            names=["demod", "demod_05", "demod_burst", "envelope"],
        )

        rv["video"] = (
            video_out[self.blockcut : -self.blockcut_end] if cut else video_out
        )

        demod_end_time = time.time()
        if thread_benchmark:
            ldd.logger.debug(
                "Demod thread %d, work done in %.02f msec"
                % (os.getpid(), (demod_end_time - demod_start_time) * 1e3)
            )

        return rv
