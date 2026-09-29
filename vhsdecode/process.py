import os
import sqlite3
import time
import numpy as np
import scipy.signal as sps
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import lddecode.core as ldd

# Use numpy fft rather than scipy fft as is imported in lddecode core as it seems to be slightly faster.
import numpy.fft as npfft

import lddecode.utils as lddu
from vhsdecode.chroma import chroma_color_under_filter

import vhsdecode.formats as vhs_formats

from vhsdecode.addons.chromasep import ChromaSepClass
from vhsdecode.addons.chromaAFC import ChromaAFC

from vhsdecode.demod import replace_spikes, unwrap_hilbert

from vhsdecode.field import field_class_from_formats
from vhsdecode.video_eq import VideoEQ
from vhsdecode.doc import DodOptions
from vhsdecode.load_params_json import override_params
from vhsdecode.nonlinear_filter import sub_deemphasis
from vhsdecode.compute_video_filters import (
    gen_video_main_deemp_fft_params,
    gen_video_lpf_params,
    gen_nonlinear_bandpass_params,
    gen_nonlinear_amplitude_lpf,
    gen_custom_video_filters,
    create_sub_emphasis_params,
    gen_video_lpf_supergauss_params,
    gen_fm_audio_notch_params,
    NONLINEAR_AMP_LPF_FREQ_DEFAULT,
    CHROMA_AUDIO_NOTCH_Q,
)
from vhsdecode import compute_video_filters as cvf
from vhsdecode.rust_utils import sosfiltfilt_rust


def is_secam(system: str):
    return system == "SECAM" or system == "MESECAM"


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
        field_order_action="detect",
    ):
        self._processing_thread_pool = ThreadPoolExecutor(max_workers=threads + 1)
        rf = VHSRFDecode(
            processing_thread_pool=self._processing_thread_pool,
            system=system,
            tape_format=tape_format,
            inputfreq=inputfreq,
            rf_options=rf_options,
            extra_options=extra_options,
            debug_plot=debug_plot,
        )

        # The superclass constructs its own laserdisc RFDecode (and the demod cache around it),
        # so hand it the tape decoder instead. No output filename so it opens no files.
        with mock.patch.object(ldd, "RFDecode", lambda **_: rf):
            super(VHSDecode, self).__init__(
                fname_in,
                None,
                freader,
                logger,
                analog_audio=False,
                system=vhs_formats.parent_system(system),
                doDOD=doDOD,
                threads=threads,
                inputfreq=inputfreq,
                extra_options=extra_options,
            )

        if system == "819":
            # We need a larger buffer for 819-line input
            # TODO: Is this useful for normal formats too?
            self.readlen = self.rf.linelen * 500

        # Adjustment for output to avoid clipping.
        self.level_adjust = level_adjust
        # Store reference to ourself in the rf decoder - needed to access data location for track
        # phase, may want to do this in a better way later.
        self.rf.decoder = self
        self.FieldClass = field_class_from_formats(system, tape_format)

        self.dbconn = None
        if extra_options.get("write_db"):
            if os.path.exists(fname_out + ".tbc.db"):
                os.unlink(fname_out + ".tbc.db")
            self.dbconn = sqlite3.connect(fname_out + ".tbc.db")
            self.create_db_schema()

        self.outfile_chroma = None

        self.fname_out = fname_out

        if fname_out is not None and self.rf.options.write_chroma:
            if extra_options.get("orc"):
                self.outfile_video = open(fname_out + ".tbcy", "wb")
                self.outfile_chroma = open(fname_out + ".tbcc", "wb")
            else:
                self.outfile_video = open(fname_out + ".tbc", "wb")
                self.outfile_chroma = open(fname_out + "_chroma.tbc", "wb")
        elif fname_out:
            self.outfile_video = open(fname_out + ".tbc", "wb")

        self.debug_plot = debug_plot
        self.field_order_action = field_order_action
        if tape_format == "TYPEC":
            # Since typec usually lacks vsync set this to none to avoid dropping fields.
            self.field_order_action = "none"
        self.duplicate_prev_field = True
        self._dropped = None

        # For tape, it is recommended to use `--ire0_adjust` to fix brightness variations between lines
        # This method usually gives false positives for noisy signals, so smooth the correction out by an entire field to avoid banding
        if self.wow_level_adjust_smoothing is None:
            self.wow_level_adjust_smoothing = self.rf.SysParams["frame_lines"] / 2

    def buildmetadata(self, f, check_phase=False):
        """returns field information JSON and whether to duplicate the previous field"""
        prevfi_1 = self.fieldinfo[-1] if len(self.fieldinfo) else None
        prevfi_2 = self.fieldinfo[-2] if len(self.fieldinfo) > 1 else None

        # Not calulated and used for tapes at the moment
        # bust_median = lddu.roundfloat(np.nan_to_num(f.burstmedian)) #lddu.roundfloat(f.burstmedian if not math.isnan(f.burstmedian) else 0.0)
        # "medianBurstIRE": bust_median,

        fi = {
            "isFirstField": True if f.isFirstField else False,
            "detectedFirstField": True if f.isFirstField else False,
            "isDuplicateField": False,
            # burstStartLine description:
            # -1                    -> Color killer is active, no color for entire field
            #  0                    -> Color killer is inactive, color for the entire field
            #  1 to num_field_lines -> Color killer is active until this line, then it is deactivated and color is returned for this and all following lines
            "burstStartLine": f.burst_detected_line,
            "syncConf": f.compute_syncconf(),
            "seqNo": len(self.fieldinfo) + 1,
            "diskLoc": np.round((f.readloc / self.bytes_per_field) * 10) / 10,
            "fileLoc": int(np.floor(f.readloc)),
        }

        if f.fieldPhaseID is None:
            fi["fieldPhaseID"] = {
                (1, 0): 1,
                (0, 1): 2,
                (1, 1): 3,
                (0, 0): 4,
            }[(fi["isFirstField"], (fi["seqNo"] // 2) % 2)]
        else:
            fi["fieldPhaseID"] = f.fieldPhaseID

        if self.doDOD:
            dropout_lines, dropout_starts, dropout_ends = f.dropout_detect()
            if len(dropout_lines):
                fi["dropOuts"] = {
                    "fieldLine": dropout_lines,
                    "startx": dropout_starts,
                    "endx": dropout_ends,
                }

        # This is a bitmap, not a counter
        # docs for this mysterious bitmap???

        decode_faults = 0
        fi["vitsMetrics"] = self.computeMetrics(self.fieldstack[0], self.fieldstack[1])
        # interlaced video requires alternating fields, handle cases where fields are repeated
        #   this can happen due to breaks in recordings between fields, i.e. home recordings, and
        #   progressive content, such as video game, osd, computer output, etc.
        if prevfi_1 is not None and prevfi_1["isFirstField"] == fi["isFirstField"]:
            distance_from_previous_field = fi["diskLoc"] - prevfi_1["diskLoc"]
            if (
                # there are three (this one, and two previous) repeating field orders in a row
                # should be impossible for valid interlaced video, so maybe it's progressive??
                # progressive examples needed to test this!
                prevfi_1["detectedFirstField"] == fi["detectedFirstField"]
                and prevfi_2 is not None
                and prevfi_2["detectedFirstField"] == prevfi_1["detectedFirstField"]
                # and this field is within a reasonable distance to be valid
                and lddu.inrange(distance_from_previous_field, 0.9, 1.1)
                # Skip on TYPEC since we expect to have missing vsync there and we don't
                # expect progressive video.
                and self.rf.options.tape_format != "TYPEC"
            ):
                # treat this as progressive, and manually flip the field order
                ldd.logger.error(
                    "Detected progressive video content..., manually flipping the field order to compensate"
                )
                decode_faults |= 1
                fi["syncConf"] = 10
                fi["isFirstField"] = not prevfi_1["isFirstField"]
            else:
                if self.field_order_action == "duplicate":
                    self.duplicate_prev_field = True
                elif self.field_order_action == "drop":
                    self.duplicate_prev_field = False
                elif self.field_order_action == "detect":
                    # duplicated field order was detected more than 1.1 fields away from the previous field, possibly a gap
                    if distance_from_previous_field > 1.1:
                        self.duplicate_prev_field = True
                    # duplicated field order was detected less than 0.9 fields away from the previous field, probably overlaped end of last field
                    elif distance_from_previous_field < 0.9:
                        self.duplicate_prev_field = False
                    # next field is close enough to be a valid field, duplicating or dropping is valid, alternate to avoid too many duplicates or drops
                    else:
                        self.duplicate_prev_field = not self.duplicate_prev_field

                if self.field_order_action == "none":
                    if self.rf.options.tape_format != "TYPEC":
                        ldd.logger.error(
                            "Possibly skipped field (Two fields with same isFirstField in a row), manually flipping the field order to compensate"
                        )
                    decode_faults |= 4
                    fi["syncConf"] = 0
                    fi["isFirstField"] = not prevfi_1["isFirstField"]
                elif self.duplicate_prev_field:
                    ldd.logger.error(
                        "Possibly skipped field (Two fields with same isFirstField in a row), duplicating the last field to compensate..."
                    )
                    decode_faults |= 4
                    fi["syncConf"] = 0
                    fi["isDuplicateField"] = True
                else:
                    ldd.logger.error(
                        "Possibly skipped field (Two fields with same isFirstField in a row), dropping the last field to compensate..."
                    )
                    decode_faults |= 4
                    fi["syncConf"] = 0
                    # readfield() stores and writes every field it gets metadata for, so
                    # remember what to put back when writeout() skips this one.
                    self._dropped = (fi, self.lastvalidfield[f.isFirstField])

            if decode_faults != 0:
                # Only write this if it's anything else than 0, to save a little space in the json,
                # since it's not used for anything atm anyhow.
                fi["decodeFaults"] = decode_faults

            return fi, fi["isDuplicateField"]

        if f.isFirstField:
            self.firstfield = f
        elif self.firstfield is not None:
            rawloc = int(np.floor((f.readloc / self.bytes_per_field) / 2))
            self.logger.status(f"File Frame {rawloc}: {self.rf.options.tape_format} ")

        return fi, fi["isDuplicateField"]

    # Again ignored for tapes
    def checkMTF(self, field, pfield=None):
        return True

    def writeout(self, dataset):
        f, fi, (picturey, picturec), audio, efm = dataset

        if self._dropped and fi is self._dropped[0]:
            self.lastvalidfield[f.isFirstField] = self._dropped[1]
            self._dropped = None
            return

        if self.rf.options.write_chroma:
            self.outfile_chroma.write(picturec)

        if self.dbconn:
            # Not measured on tape (see calc_burstmedian), but the field record needs it.
            fi["medianBurstIRE"] = 1.0
            super(VHSDecode, self).writeout((f, fi, picturey, audio, efm))
        else:
            self.fieldinfo.append(fi)
            self.outfile_video.write(picturey)
            self.fields_written += 1

    def close(self):
        if self.decodethread and self.decodethread.is_alive():
            self.decodethread.join()
            self.decodethread = None
        if self.rf.options.write_chroma:
            setattr(self, "outfile_chroma", None)

        if self._processing_thread_pool is not None:
            self._processing_thread_pool.shutdown(wait=True)
        super(VHSDecode, self).close()

    def build_json(self):
        jout = super(VHSDecode, self).build_json()
        if jout is None:
            return None

        black = jout["videoParameters"]["black16bIre"]
        white = jout["videoParameters"]["white16bIre"]

        if self.rf.color_system == "PAL_M" or self.rf.color_system == "NLINHA":
            jout["videoParameters"]["system"] = "PAL-M"

        jout["videoParameters"]["black16bIre"] = black * (1 - self.level_adjust)
        jout["videoParameters"]["white16bIre"] = white * (1 + self.level_adjust)

        jout["videoParameters"]["tapeFormat"] = self.rf.options.tape_format
        return jout


class VHSRFDecode(ldd.RFDecode):
    def __init__(
        self,
        processing_thread_pool=None,
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
            vhs_formats.parent_system(system),
            decode_analog_audio=False,
            has_analog_audio=False,
            extra_options=extra_options,
        )
        if processing_thread_pool is None:
            processing_thread_pool = ThreadPoolExecutor(max_workers=1)
        self._processing_thread_pool = processing_thread_pool

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
        # TODO: integrate this under chroma_trap later
        self._use_fsc_notch_filter = (
            tape_format == "BETAMAX" or tape_format == "BETAMAX_HIFI"
        )
        track_phase = None if is_secam(system) else rf_options.get("track_phase", None)
        high_boost = rf_options.get("high_boost", None)
        self._notch = rf_options.get("notch", None)
        self._notch_q = rf_options.get("notch_q", 10.0)
        self._disable_diff_demod = rf_options.get("disable_diff_demod", False)
        self.useAGC = extra_options.get("useAGC", False)
        self.debug = extra_options.get("debug", False)

        # cafc measures a single carrier peak, which doesn't exist in the
        # line-alternating two-carrier SECAM FM chroma signal.
        requested_cafc = rf_options.get("cafc", False)
        if requested_cafc and is_secam(system):
            ldd.logger.warning(
                "cafc is not supported for SECAM systems, ignoring. "
                "(For ME-SECAM, the carrier servo tunes the conversion LO instead.)"
            )
            requested_cafc = False

        # Enable cafc for betamax until proper track detection for it is implemented.
        self._do_cafc = (
            True
            if (tape_format == "BETAMAX" and system != "NTSC")
            else requested_cafc
        )

        self.track_phase = None
        if track_phase == 0 or track_phase == 1:
            self.track_phase = track_phase
        elif track_phase is not None:
            raise Exception("Track phase can only be 0, 1 or None")

        self.hsync_tolerance = 0.8

        self.field_number = 0
        self.last_raw_loc = None

        self.SysParams, self.DecoderParams = vhs_formats.get_format_params(
            system,
            tape_format,
            vhs_formats.parse_tape_speed(rf_options.get("tape_speed", "sp")),
            ldd.logger,
        )
        # The superclass computed these from the parent system, which is wrong for 405/819-line.
        self.linelen = int(np.round(self.freq_hz / (1000000.0 / self.SysParams["line_period"])))
        self.samplesperline = self.freq / self.linelen

        params_file = extra_options.get("params_file", None)
        if params_file:
            override_params(self.SysParams, self.DecoderParams, params_file, ldd.logger)

        # Make (intentionally) mutable copies of HZ<->IRE levels
        # (NOTE: used by upstream functions, we use a namedtuple to keep const values already)
        self.DecoderParams["ire0"] = self.SysParams["ire0"]
        self.DecoderParams["hz_ire"] = self.SysParams["hz_ire"]
        self.DecoderParams["vsync_ire"] = self.SysParams["vsync_ire"]
        self.DecoderParams["track_ire0_offset"] = self.SysParams.get(
            "track_ire0_offset", [0, 0]
        )

        export_raw_tbc = rf_options.get("export_raw_tbc", False)
        ire0_adjust_raw = rf_options.get("ire0_adjust", "")
        if isinstance(ire0_adjust_raw, str):
            ire0_adjust = tuple(
                mode.strip().lower()
                for mode in ire0_adjust_raw.split(",")
                if mode.strip()
            )
        else:
            ire0_adjust = tuple()
        is_color_under = vhs_formats.is_color_under(tape_format)
        write_chroma = (
            is_color_under
            and not export_raw_tbc
            and not rf_options.get("skip_chroma", False)
            and not (system == "405")
        )

        # No idea if this is a common pythonic way to accomplish it but this gives us values that
        # can't be changed later.
        # first depends on IRE/Hz so has to be set after that is properly set.
        # TODO: May want to split this up eventually
        self._options = namedtuple(
            "Options",
            [
                "diff_demod_check_value",
                "tape_format",
                "disable_comb",
                "nldeemp",
                "subdeemp",
                "disable_right_hsync",
                "disable_dc_offset",
                "fallback_vsync",
                "field_order_confidence",
                "saved_levels",
                "y_comb",
                "write_chroma",
                "color_under",
                "chroma_deemphasis_filter",
                "skip_hsync_refine",
                "hsync_refine_use_threshold",
                "export_raw_tbc",
                "fm_audio_notch",
                "chroma_audio_notch",
                "chroma_offset",
                "cti_mix",
                "cti_width",
                "ire0_adjust",
                "gnrc_afe",
                "relaxed_line0",
                "detect_chroma_track_phase",
                "enable_color_killer",
                "disable_burst_hsync",
                "disable_phase_correction",
                "secam_carrier_servo",
            ],
        )(
            self.iretohz(100) * 2,
            tape_format,
            rf_options.get("disable_comb", False) or is_secam(system),
            rf_options.get("nldeemp", False),
            self.DecoderParams.get("use_sub_deemphasis", False)
            or rf_options.get("subdeemp", False),
            rf_options.get("disable_right_hsync", False),
            rf_options.get("disable_dc_offset", False),
            # Always use this if we are decoding TYPEC since it doesn't have normal vsync.
            # also enable by default with EIAJ since that was typically used with a primitive sync gen
            # which output not quite standard vsync.
            rf_options.get("fallback_vsync", False)
            or tape_format == "TYPEC"
            or tape_format == "EIAJ"
            or system == "405"
            or system == "819",
            rf_options.get("field_order_confidence", False),
            rf_options.get("saved_levels", False),
            rf_options.get("y_comb", 0) * self.SysParams["hz_ire"],
            write_chroma,
            is_color_under,
            tape_format == "VIDEO8" or tape_format == "HI8",
            rf_options.get("skip_hsync_refine", False),
            # hsync_refine_use_threshold - use detected level for hsync refine
            # TODO: This should be used for everything eventually but needs proper testing
            True,
            export_raw_tbc,
            # Optional on VHS/Beta/video8
            # always enable for hi8 since should pretty much always have the second audio
            # channel. May want to enable for video8 as well.
            rf_options.get("fm_audio_notch", 0) or (tape_format == "HI8"),
            self.DecoderParams.get("chroma_audio_notch_freq", 0) > 0,
            int(self.DecoderParams.get("chroma_offset", 5) * (self.freq / 40.0)),
            rf_options.get("cti_mix", 1),
            rf_options.get("cti_width", 2),
            ire0_adjust,
            rf_options.get("gnrc_afe", False),
            rf_options.get("relaxed_line0", False),
            rf_options.get("detect_chroma_track_phase", False),
            rf_options.get("enable_color_killer", False),
            # SECAM has no phase-locked burst; the "burst" is an FM carrier
            # whose phase carries no timing information, so locking hsync to
            # it just injects sub-pixel jitter into both planes.
            rf_options.get("disable_burst_hsync", False) or is_secam(system),
            rf_options.get("disable_phase_correction", False),
            rf_options.get("secam_carrier_servo", True),
        )

        if self._options.gnrc_afe:
            from vhsdecode.addons.gnuradioZMQ import ZMQSend, ZMQReceive

            self.zmqsend = ZMQSend()
            self.zmqreceive = ZMQReceive()
            print(
                "Open GNURadio with ZMQ REQ source set at tcp://localhost:%d and ZMQ REP sink set at tcp://*:%d\n"
                "The data stream will be of the float type at 40MSPS (40MHz sample rate)\n"
                "It will send the raw RF for further processing prior to demodulation (useful for RF EQ discovery "
                "and group delay compensation)\n"
                "You might want to do this in single threaded decode mode (-t 1 parameter) - TODO: might not work correctly with --no_resample yet."
                % (self.zmqsend.port, self.zmqreceive.port)
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

        #
        self._sub_emphasis_params = create_sub_emphasis_params(
            self.DecoderParams,
            self.SysParams,
            self._sysparams_const.hz_ire,
            self._sysparams_const.vsync_ire,
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
            self.DecoderParams.get("chroma_bpf_order", 4),
            tape_format=tape_format,
            do_cafc=self._do_cafc,
            chroma_bpf_lower=self.DecoderParams.get("chroma_bpf_lower", 60000),
            conversion_lo_freq=self.DecoderParams.get("chroma_conversion_lo", None),
            carrier_mult=self.DecoderParams.get("chroma_carrier_mult", None),
        )

        self.Filters["FVideoBurst"] = (
            self._chroma_afc.get_chroma_bandpass()
            if self._options.color_under
            else self._chroma_afc.get_chroma_bandpass_final(False)
        )

        if self.options.chroma_deemphasis_filter:
            out_freq_half = self._chroma_afc.getOutFreqHalf()
            self.Filters["chroma_deemphasis"] = cvf.gen_peaking_constq(
                self.sys_params["fsc_mhz"] / out_freq_half, 3.4, 0.5 / out_freq_half
            )

        if self._notch is not None:
            video_notch_filter = sps.iirnotch(
                self._notch / self.freq_half, self._notch_q
            )

            # Chroma notch filter
            if self._do_cafc:
                self.Filters["FVideoNotch"] = sps.iirnotch(
                    self._notch / self._chroma_afc.getOutFreqHalf(), self._notch_q
                )
            else:
                self.Filters["FVideoNotch"] = video_notch_filter

            # Luma notch filter
            self.Filters["FVideoNotchF"] = abs(
                lddu.filtfft(video_notch_filter, self.blocklen)
            )
        else:
            self.Filters["FVideoNotch"] = None, None

        if self._options.chroma_audio_notch:
            if self._do_cafc:
                self.Filters["FChromaAudioNotch"] = sps.iirnotch(
                    DP["chroma_audio_notch_freq"]
                    / (self._chroma_afc.getOutFreqHalf() * 1e6),
                    CHROMA_AUDIO_NOTCH_Q,
                )
            else:
                self.Filters["FChromaAudioNotch"] = sps.iirnotch(
                    DP["chroma_audio_notch_freq"] / (self.freq_hz_half),
                    CHROMA_AUDIO_NOTCH_Q,
                )

        # The following filters are for post-TBC:
        # The output sample rate is 4fsc
        self.Filters["FChromaFinal"] = self._chroma_afc.get_chroma_bandpass_final(
            self._options.color_under
        )

        if is_color_under:
            self.chroma_heterodyne = self._chroma_afc.getChromaHet()
            self.fsc_wave, self.fsc_cos_wave = self._chroma_afc.getFSCWaves()

        if self._chroma_afc.carrier_mult is not None:
            # SECAM method 1: post-TBC band-pass around the under carriers
            # ahead of the x4 phase multiplication.
            self.Filters["FSecamUnder"] = self._chroma_afc.get_secam_under_bandpass()
            # Porch carrier pair sanity check over the first fields, to catch
            # tapes that were actually recorded with the ME-SECAM method.
            self.secam_method_diag = {
                "fields": 0,
                "method1": 0,
                "mesecam": 0,
                "done": False,
            }

        # Long-term average of the measured ME-SECAM rest carrier pair offset,
        # used to trim the chroma up-conversion LO.
        self.secam_servo_avg = deque(maxlen=60)
        lo_trim_seed = rf_options.get("secam_lo_trim", None)
        if lo_trim_seed is not None and system == "MESECAM":
            # Seed with a known trim (e.g. from a two-pass calibration decode)
            # so it applies from the first field. With the servo enabled it
            # keeps adapting from here; with it disabled this is a fixed trim.
            for _ in range(3):
                self.secam_servo_avg.append(float(lo_trim_seed))

        # Increase the cutoff at the end of blocks to avoid edge distortion from filters
        # making it through.
        self.blockcut_end = 1024

        if self._chroma_trap:
            self.chromaTrap = ChromaSepClass(
                self.freq_hz, self.SysParams["fsc_mhz"], ldd.logger
            )

        # TODO: This should be managed elsewhere.
        self._compute_linelocs_issues = False

    @property
    def sysparams_const(self):
        return self._sysparams_const

    @property
    def sys_params(self):
        return self.SysParams

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
    def color_system(self):
        return self._color_system

    @property
    def dod_options(self):
        return self._dod_options

    @property
    def compute_linelocs_issues(self):
        return self._compute_linelocs_issues

    @compute_linelocs_issues.setter
    def compute_linelocs_issues(self, value):
        self._compute_linelocs_issues = value

    def computevideofilters(self):
        self.Filters = {}
        # Needs to be defined here as it's referenced in constructor.
        self.Filters["F05_offset"] = 32

    def _computevideofilters_b(self):
        # Use some shorthand to compact the code.
        SF = self.Filters
        DP = self.DecoderParams

        SF["hilbert"] = lddu.build_hilbert(self.blocklen)

        if DP.get("video_bpf_supergauss", False):
            self.Filters["RFVideo"] = lddu.gen_bpf_supergauss(
                DP["video_bpf_low"],
                DP["video_bpf_high"],
                DP["video_bpf_order"],
                self.freq_hz_half,
                self.blocklen,
            )
        else:
            # Filter for rf before demodulating.
            # Only use bpf if order defined - otherwise skip
            if DP.get("video_bpf_order", None):
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
            else:
                y_fm = None

            # Gen fft filter from sos filter
            # TODO: Move this elsewhere
            def sosfiltfft(filter_value, block_len):
                return sps.sosfreqz(filter_value, block_len, whole=True)[1]

            y_fm_lowpass = sosfiltfft(
                sps.butter(
                    DP["video_lpf_extra_order"],
                    DP["video_lpf_extra"] / self.freq_hz_half,
                    btype="lowpass",
                    output="sos",
                ),
                self.blocklen,
            )

            y_fm_highpass = sosfiltfft(
                sps.butter(
                    DP["video_hpf_extra_order"],
                    DP["video_hpf_extra"] / self.freq_hz_half,
                    btype="highpass",
                    output="sos",
                ),
                self.blocklen,
            )

            if y_fm is not None:
                # Only use this if defined
                self.Filters["RFVideo"] = (
                    abs(y_fm) * abs(y_fm_lowpass) * abs(y_fm_highpass)
                )
            else:
                self.Filters["RFVideo"] = abs(y_fm_lowpass) * abs(y_fm_highpass)

        if DP.get("video_rf_peak_freq", False):
            # Add optional rf peaking filter
            peaking_filter = lddu.filtfft(
                cvf.gen_peaking_constq(
                    DP["video_rf_peak_freq"] / self.freq_hz_half,
                    DP.get("video_rf_peak_gain", 3),
                    DP.get("video_rf_peak_bandwidth", 2.5e6) / self.freq_hz_half,
                ),
                self.blocklen,
            )
            self.Filters["RFVideo"] *= abs(peaking_filter)

        # Make sure this is an int in case it could be passed in as a string via the gui.
        if int(self.options.fm_audio_notch) > 0:
            if "fm_audio_channel_0_freq" in DP and "fm_audio_channel_1_freq" in DP:
                # Optionally enable double notch filter on fm audio channel frequencies.
                # This is mainly useful on VHS (and possibly PAL betamax with hifi?)
                # The hifi carriers on vhs are depth-multiplexed and read by a different head
                # but they still sometimes are picked up strongly enough by the video heads to
                # interfere with the video signal. The carrier for the upper channel especially since
                # it sits high enough that it it overlaps with the lower video sideband.
                # On formats where audio and video share the same heads (8mm, betamax NTSC hifi) the audio and
                # video bands are set up to be separatated more cleanly but for vhs cutting off the video sideband
                # above the audio carrier cuts off too much so use this approach instead and only if needed.
                audio_fm_notch_filter = gen_fm_audio_notch_params(
                    DP, self.options.fm_audio_notch, self.freq_hz_half, self.blocklen
                )
                self.Filters["RFVideo"] *= abs(audio_fm_notch_filter)
            else:
                ldd.logger.warning(
                    "Audio frequencies are not specified for this format, audio fm notch filters not enabled!"
                )

        if DP.get("boost_rf_linear_0", None) is not None:
            ramp = cvf.gen_ramp_filter_params(
                DP,
                self.freq_hz_half,
                self.blocklen,
            )

            self.Filters["RFVideo"] *= ramp
            if DP.get("boost_rf_linear_double", False):
                self.Filters["RFVideo"] *= ramp

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
        filter_deemp = gen_video_main_deemp_fft_params(DP, self.freq_hz, self.blocklen)

        if DP.get("video_lpf_supergauss", False) is True:
            filter_video_lpf = gen_video_lpf_supergauss_params(
                DP, self.freq_hz_half, self.blocklen
            )
        else:
            _, filter_video_lpf = gen_video_lpf_params(
                DP, self.freq_hz_half, self.blocklen
            )

        if DP.get("video_custom_luma_filters", None) is not None:
            self.Filters["FCustomVideo"] = gen_custom_video_filters(
                DP["video_custom_luma_filters"],
                self.freq_hz,
                self.blocklen,
            )
        else:
            self.Filters["FCustomVideo"] = 1.0

        # additional filters:  0.5mhz, used for sync detection.
        # Using an FIR filter here to get a known delay
        F0_5 = sps.firwin(65, [0.5 / self.freq_half], pass_zero=True)
        filter_05 = lddu.filtfft((F0_5, [1.0]), self.blocklen)[: self.blocklen // 2 + 1]

        # This filter is simple enough that we can get away with single precision
        # sections and thus do the filtering in sngle precision.
        # On higher order filters this is not viable as it tends to alter the filter too much.
        self.Filters["FEnvPost"] = sps.butter(
            1, 700000 / self.freq_hz_half, btype="lowpass", output="sos"
        )

        self.Filters["NLAmplitudeLPF"] = gen_nonlinear_amplitude_lpf(
            DP.get("nonlinear_amp_lpf_freq", NONLINEAR_AMP_LPF_FREQ_DEFAULT),
            self.freq_hz_half,
        )

        if self._use_fsc_notch_filter:
            self.Filters["fsc_notch"] = sps.iirnotch(
                self.sys_params["fsc_mhz"] / self.freq_half, 2
            )

        self.Filters["FDeemp"] = filter_deemp

        self.Filters["FVideo"] = (
            filter_deemp * filter_video_lpf * self.Filters["FCustomVideo"]
        )

        SF["FVideo05"] = filter_video_lpf * filter_deemp * filter_05

        if self.options.nldeemp or self.options.subdeemp:
            SF["NLHighPassF"] = gen_nonlinear_bandpass_params(
                DP, self.freq_hz_half, self.blocklen
            )

        if self.debug_plot and self.debug_plot.is_plot_requested("rf_luma"):
            from vhsdecode.debug_plot import plot_luma_rf

            plot_luma_rf(self, self.Filters["RFVideo"])

        if (
            self.debug_plot
            and self.debug_plot.is_plot_requested("nldeemp")
            and self.options.subdeemp
        ):
            from vhsdecode.nonlinear_filter import test_filter

            test_filter(
                self.Filters,
                self.freq_hz,
                self.blocklen,
                (self._sysparams_const.hz_ire * 143.0),
                self._sub_emphasis_params,
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
        demod_block_debug = False
        demod_start_time = time.time()
        if self._options.gnrc_afe:
            self.zmqsend.send(data)
            data = self.zmqreceive.receive(data.size)

        # The cached fft the demod cache passes in is not used since the filters
        # below are applied in place, which would corrupt it for a re-decode.
        indata_fft = npfft.fft(data[: self.blocklen])

        if self.debug_plot and self.debug_plot.is_plot_requested("demodblock"):
            demod_block_debug = True
            # If we're doing a plot make a copy of the input to be able to plot it since we
            # are modifying the data in place.
            indata_fft_copy = indata_fft.copy()

        if self._notch is not None:
            indata_fft *= self.Filters["FVideoNotchF"]

        # Applies RF filters
        indata_fft *= self.Filters["RFVideo"]

        raw_filtered = npfft.ifft(indata_fft * self.Filters["hilbert"]).real.astype(
            np.single
        )

        # Calculate an evelope with signal strength using absolute of hilbert transform.
        # Roll this a bit to compensate for filter delay, value eyballed for now.
        np.abs(raw_filtered, out=raw_filtered)
        raw_env = np.roll(raw_filtered, 4)
        del raw_filtered
        # Downconvert to single precision for some possible speedup since we don't need
        # super high accuracy for the dropout detection.
        env = sosfiltfilt_rust(self.Filters["FEnvPost"], raw_env)

        del raw_env
        env_mean = np.mean(env)

        # Boost high frequencies in areas where the signal is weak to reduce missed zero crossings
        # on sharp transitions. Using filtfilt to avoid phase issues.
        if len(np.where(env == 0)[0]) == 0:  # checks for zeroes on env
            if self._high_boost is not None:
                data_filtered = npfft.ifft(indata_fft).real
                high_part = sosfiltfilt_rust(self.Filters["RFTop"], data_filtered) * (
                    (env_mean * 0.9) / env
                )
                del data_filtered
                indata_fft += npfft.fft(high_part * self._high_boost)
        else:
            ldd.logger.warning("RF signal is weak. Is your deck tracking properly?")

        hilbert = npfft.ifft(indata_fft * self.Filters["hilbert"])

        if not demod_block_debug:
            del indata_fft

        # FM demodulator
        demod = unwrap_hilbert(hilbert, self.freq_hz)

        # If there are obviously out of bounds values, do an extra demod on a diffed waveform and
        # replace the spikes with data from the diffed demod. (Which in practice is an extra EQed signal)
        if not self._disable_diff_demod:
            check_value = self.options.diff_demod_check_value

            if np.max(demod[20:-20]) > check_value:
                demod_b = unwrap_hilbert(
                    np.ediff1d(hilbert, to_begin=0), self.freq_hz
                ).real

                demod = replace_spikes(demod, demod_b, check_value)
                del demod_b

        # Disabled if sharpness level is zero (default).
        # TODO: This should be done after the deemphasis steps
        if self._video_eq:
            # applies the video EQ
            demod = self._video_eq.filter_video(demod)

        # TODO: This should be done after the deemphasis steps
        if self._chroma_trap:
            # applies the Subcarrier trap
            demod = self.chromaTrap.work(demod)

        # applies main deemphasis filter
        demod_fft = npfft.rfft(demod)
        out_video_fft = demod_fft * self.Filters["FVideo"]
        out_video = npfft.irfft(out_video_fft).real

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

        if self.options.subdeemp:
            out_video = sub_deemphasis(
                out_video,
                out_video_fft,
                self.Filters,
                self._sub_emphasis_params.deviation,
                self._sub_emphasis_params.exponential_scaling,
                self._sub_emphasis_params.scaling_1,
                self._sub_emphasis_params.scaling_2,
                self._sub_emphasis_params.logistic_mid,
                self._sub_emphasis_params.logistic_rate,
                self._sub_emphasis_params.static_factor,
            )

        del out_video_fft

        if self._use_fsc_notch_filter:
            out_video = sps.filtfilt(
                self.Filters["fsc_notch"][0], self.Filters["fsc_notch"][1], out_video
            )

        out_video05 = npfft.irfft(demod_fft * self.Filters["FVideo05"]).real
        out_video05 = np.roll(out_video05, -self.Filters["F05_offset"])

        # Filter out the color-under signal from the raw data.
        chroma_source = data if self.options.color_under else out_video
        out_chroma = (
            chroma_color_under_filter(
                chroma_source,
                self.Filters["FVideoBurst"],
                self.blocklen,
                self.Filters["FVideoNotch"],
                self._notch,
                move=int(self.options.chroma_offset),
                audio_notch=self.Filters.get("FChromaAudioNotch", None),
                # TODO: Do we need to tweak move elsewhere too?
                # if cafc is enabled, this filtering will be done after TBC
            )
            if not self._do_cafc
            else data[: self.blocklen]
        )

        if self.debug_plot and self.debug_plot.is_plot_requested("magdens"):
            from vhsdecode.debug_plot import plot_magnitude_density

            plot_magnitude_density(
                raw_data=data[: self.blocklen],
                filtered_data=npfft.ifft(indata_fft).real,
                rfdecode=self,
            )

        if demod_block_debug:
            from vhsdecode.debug_plot import plot_input_data

            plot_input_data(
                raw_data=data,
                filtered_data=npfft.ifft(indata_fft).real,
                env=env,
                env_mean=env_mean,
                raw_fft=indata_fft_copy,
                filtered_fft=indata_fft,
                demod_video=demod,
                filtered_video=out_video,
                chroma=out_chroma,
                rf_filter=self.Filters["RFVideo"],
                rfdecode=self,
                plot_chroma_fft=True,
            )

        if self.options.export_raw_tbc:
            out_video = demod

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
