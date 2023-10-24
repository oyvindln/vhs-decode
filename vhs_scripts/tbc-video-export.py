#!/usr/bin/env python3

# Script for exporting TBC files to video
# - JuniorIsAJitterbug

import argparse
import json
import os
import subprocess
import pathlib
import tempfile
from enum import Enum
from shutil import which

if os.name == "nt":
    import win32pipe
    import win32api
    import win32file
    import pywintypes
    import winerror
    import win32con
    from threading import Thread


class VideoSystem(Enum):
    PAL = "pal"
    NTSC = "ntsc"

    def __str__(self):
        return self.value


class ChromaDecoder(Enum):
    PAL2D = "pal2d"
    TRANSFORM2D = "transform2d"
    TRANSFORM3D = "transform3d"
    NTSC1D = "ntsc1d"
    NTSC2D = "ntsc2d"
    NTSC3D = "ntsc3d"
    NTSC3DNOADAPT = "ntsc3dnoadapt"

    def __str__(self):
        return self.value


class InputFiles:
    def __init__(self, file, input_json):
        self.path = pathlib.Path(file).parent
        self.name = pathlib.Path(file).stem
        self.tbc = os.path.join(self.path, self.name + ".tbc")
        self.tbc_chroma = os.path.join(self.path, self.name + "_chroma.tbc")

        if input_json is not None:
            self.tbc_json = input_json
        else:
            self.tbc_json = os.path.join(self.path, self.name + ".tbc.json")

        try:
            with open(self.tbc_json, "r") as file:
                self.tbc_json_data = json.load(file)
        except:
            raise Exception(json_file + " is not valid json file")

        self.video_luma = None
        self.video = None

    def check_files_exist(self):
        files = [self.tbc, self.tbc_chroma, self.tbc_json]

        for file in files:
            if not os.path.isfile(file):
                raise Exception("missing required tbc file " + file)


class DecoderSettings:
    def __init__(self, program_opts, video_system):
        self.program_opts = program_opts
        self.video_system = video_system

    def convert_opt(self, program_opts, program_opt_name, target_opt_name):
        """Converts a program opt to a subprocess opt."""
        rt = []
        value = getattr(program_opts, program_opt_name)

        if value is not None:
            if type(value) is bool:
                # only appends opt on true, fine for current use
                if value is True:
                    rt.append([target_opt_name])
            else:
                rt.append([target_opt_name, str(value)])

        return rt

    def get_opts(self):
        """Generate ld-chroma-decoder opts."""
        decoder_opts = []

        if not self.program_opts.verbose:
            decoder_opts.append("-q")

        if self.video_system is VideoSystem.PAL:
            if self.program_opts.chroma_decoder is None:
                # chroma decoder unset, use transform2d
                decoder_opts.append(["-f", ChromaDecoder.TRANSFORM2D.value])

            # vbi is set, use preset line values
            if self.program_opts.vbi:
                decoder_opts.append(["--ffll", "2"])
                decoder_opts.append(["--lfll", "308"])
                decoder_opts.append(["--ffrl", "2"])
                decoder_opts.append(["--lfrl", "620"])
            elif self.program_opts.letterbox:
                decoder_opts.append(["--ffll", "2"])
                decoder_opts.append(["--lfll", "308"])
                decoder_opts.append(["--ffrl", "118"])
                decoder_opts.append(["--lfrl", "548"])

        elif self.video_system is VideoSystem.NTSC:
            if self.program_opts.chroma_decoder is None:
                # chroma decoder unset, use ntsc2d
                decoder_opts.append(["-f", ChromaDecoder.NTSC2D.value])

            decoder_opts.append("--ntsc-phase-comp")

            # vbi is set, use preset line values
            if self.program_opts.vbi:
                decoder_opts.append(["--ffll", "1"])
                decoder_opts.append(["--lfll", "259"])
                decoder_opts.append(["--ffrl", "2"])
                decoder_opts.append(["--lfrl", "525"])
            elif self.program_opts.letterbox:
                decoder_opts.append(["--ffll", "2"])
                decoder_opts.append(["--lfll", "308"])
                decoder_opts.append(["--ffrl", "118"])
                decoder_opts.append(["--lfrl", "453"])

        if self.program_opts.chroma_decoder is not None:
            decoder_opts.append(
                self.convert_opt(self.program_opts, "chroma_decoder", "-f")
            )

        if not self.program_opts.vbi:
            decoder_opts.append(
                self.convert_opt(self.program_opts, "first_active_field_line", "--ffll")
            )
            decoder_opts.append(
                self.convert_opt(self.program_opts, "last_active_field_line", "--lfll")
            )
            decoder_opts.append(
                self.convert_opt(self.program_opts, "first_active_frame_line", "--ffrl")
            )
            decoder_opts.append(
                self.convert_opt(self.program_opts, "last_active_frame_line", "--lfrl")
            )

        decoder_opts.append(self.convert_opt(self.program_opts, "start", "-s"))
        decoder_opts.append(self.convert_opt(self.program_opts, "length", "-l"))
        decoder_opts.append(self.convert_opt(self.program_opts, "reverse", "-r"))
        decoder_opts.append(self.convert_opt(self.program_opts, "threads", "-t"))
        decoder_opts.append(
            self.convert_opt(self.program_opts, "output_padding", "--pad")
        )
        decoder_opts.append(self.convert_opt(self.program_opts, "offset", "-o"))
        decoder_opts.append(
            self.convert_opt(self.program_opts, "simple_pal", "--simple-pal")
        )
        decoder_opts.append(
            self.convert_opt(self.program_opts, "show_ffts", "--show-ffts")
        )
        decoder_opts.append(
            self.convert_opt(
                self.program_opts, "transform_threshold", "--transform-threshold"
            )
        )
        decoder_opts.append(
            self.convert_opt(
                self.program_opts, "transform_thresholds", "--transform-thresholds"
            )
        )

        return decoder_opts

    def get_luma_opts(self):
        """Generate ld-chroma-decoder opts for luma."""
        decoder_opts = []
        decoder_opts.append(self.convert_opt(self.program_opts, "luma_nr", "--luma-nr"))

        # ignore program opts for these
        decoder_opts.append(["--chroma-nr", "0"])
        decoder_opts.append(["--chroma-gain", "0"])
        decoder_opts.append(["--chroma-phase", "1"])

        return decoder_opts

    def get_chroma_opts(self):
        """Generate ld-chroma-decoder opts for chroma."""
        decoder_opts = []

        decoder_opts.append(
            self.convert_opt(self.program_opts, "chroma_gain", "--chroma-gain")
        )

        decoder_opts.append(
            self.convert_opt(self.program_opts, "chroma_nr", "--chroma-nr")
        )
        decoder_opts.append(
            self.convert_opt(self.program_opts, "chroma_phase", "--chroma-phase")
        )

        # ignore program opts for these
        decoder_opts.append(["--luma-nr", "0"])

        return decoder_opts


class FFmpegProfile:
    def __init__(self, profiles, name):
        self.name = name
        self.profile = profiles.profiles[name]

    def get_video_opts(self):
        """Return FFmpeg video opts from profile."""
        rt = []

        if not all(key in self.profile for key in ("v_codec", "v_format", "container")):
            raise Exception("profile is missing required data")

        rt.append(["-c:v", self.profile["v_codec"]])

        if "v_opts" in self.profile:
            rt.append(self.profile["v_opts"])

        rt.append(["-pixel_format", self.profile["v_format"]])

        return rt

    def get_video_filter_opts(self):
        """Return FFmpeg video filter opts from profile."""
        if "v_filter" in self.profile:
            return "," + self.profile["v_filter"]

        return ""

    def get_audio_opts(self):
        """Return FFmpeg audio opts from profile."""
        rt = []

        if "a_codec" in self.profile:
            rt.append(["-c:a", self.profile["a_codec"]])

        if "a_opts" in self.profile:
            rt.append(self.profile["a_opts"])

        return rt

    def get_video_doublerate(self):
        """Return FFmpeg double rate from profile."""
        if "v_double_rate" in self.profile and self.profile["v_double_rate"]:
            return True

        return False

    def get_video_format(self):
        return self.profile["v_format"]

    def get_container(self):
        return self.profile["container"]


class FFmpegProfiles:
    def __init__(self, file_name):
        self.names = []
        self.names_luma = []
        self.profiles = []

        # profiles ending in _luma are considered luma profiles
        try:
            with open(file_name, "r") as file:
                data = json.load(file)
                self.names = [
                    name
                    for name in data["ffmpeg_profiles"].keys()
                    if "_luma" not in name
                ]
                self.names_luma = [
                    name for name in data["ffmpeg_profiles"].keys() if "_luma" in name
                ]
                self.profiles = data["ffmpeg_profiles"]
        except:
            raise Exception(file_name + " is not a valid json file")

    def get_profile(self, name):
        return FFmpegProfile(self, name)


class FFmpegSettings:
    def __init__(
        self, program_opts, profile, profile_luma, tbc_json_data, video_system, timecode
    ):
        self.program_opts = program_opts
        self.profile = profile
        self.profile_luma = profile_luma
        self.tbc_json_data = tbc_json_data
        self.video_system = video_system
        self.timecode = timecode

    def get_rate_opt(self):
        """Returns FFmpeg opts for rate."""
        ffmpeg_opts = []

        if self.video_system == VideoSystem.PAL:
            rate = 25
        elif self.video_system == VideoSystem.NTSC:
            rate = 30

        if self.profile.get_video_doublerate():
            rate = rate * 2

        ffmpeg_opts.append(["-r", str(rate)])

        return ffmpeg_opts

    def get_aspect_ratio_opt(self):
        """Returns FFmpeg opts for aspect ratio."""
        if (
            (
                "isWidescreen" in self.tbc_json_data["videoParameters"]
                and self.tbc_json_data["videoParameters"]["isWidescreen"]
            )
            or self.program_opts.ffmpeg_force_anamorphic
            or self.program_opts.letterbox
        ):
            return ["-aspect", "16:9"]

        return ["-aspect", "4:3"]

    def get_color_opts(self):
        """Returns FFmpeg opts for color settings."""
        ffmpeg_opts = []

        if self.video_system == VideoSystem.PAL:
            ffmpeg_opts.append(["-colorspace", "bt470bg"])
            ffmpeg_opts.append(["-color_primaries", "bt470bg"])
            ffmpeg_opts.append(["-color_trc", "bt709"])
        elif self.video_system == VideoSystem.NTSC:
            ffmpeg_opts.append(["-colorspace", "smpte170m"])
            ffmpeg_opts.append(["-color_primaries", "smpte170m"])
            ffmpeg_opts.append(["-color_trc", "bt709"])

        return ffmpeg_opts

    def get_audio_map_opts(self):
        """Returns FFmpeg opts for audio mapping."""
        ffmpeg_opts = []

        audio_inputs = self.program_opts.ffmpeg_audio_file

        offset = 2

        if self.program_opts.luma_only:
            offset = 1

        if audio_inputs is not None:
            for idx, audio_input in enumerate(audio_inputs):
                ffmpeg_opts.append(["-map", str(idx + offset) + ":a"])

        return ffmpeg_opts

    def get_metadata_opts(self):
        """Returns FFmpeg opts for metadata."""
        ffmpeg_opts = []

        metadata = self.program_opts.ffmpeg_metadata

        # add video metadata
        if metadata is not None:
            for data in metadata:
                ffmpeg_opts.append(["-metadata", data])

        return ffmpeg_opts

    def get_audio_metadata_opts(self):
        """Returns FFmpeg opts for audio metadata."""
        ffmpeg_opts = []

        audio_titles = self.program_opts.ffmpeg_audio_title
        audio_languages = self.program_opts.ffmpeg_audio_language

        # add audio metadata only if luma only
        if audio_titles is not None:
            for idx, title in enumerate(audio_titles):
                ffmpeg_opts.append(
                    ["-metadata:s:a:" + str(idx), 'title="' + title + '"']
                )

        if audio_languages is not None:
            for idx, language in enumerate(audio_languages):
                ffmpeg_opts.append(
                    ["-metadata:s:a:" + str(idx), 'language="' + language + '"']
                )

        return ffmpeg_opts

    def get_audio_inputs_opts(self):
        """Returns FFmpeg audio input opts."""
        input_opts = []

        tracks = self.program_opts.ffmpeg_audio_file
        offsets = self.program_opts.ffmpeg_audio_offset

        if tracks is not None:
            for idx, track in enumerate(tracks):
                # add offset if set
                if offsets is not None and len(offsets) >= idx + 1:
                    input_opts.append(["-itsoffset", offsets[idx]])
                else:
                    input_opts.append(["-itsoffset", "00:00:00.000"])

                input_opts.append(["-i", track])

        return input_opts

    def get_timecode_opt(self):
        return ["-timecode", self.timecode]

    def get_verbosity(self):
        if not self.program_opts.verbose:
            return ["-hide_banner", "-loglevel", "error", "-stats"]

        return "-hide_banner"

    def get_overwrite_opt(self):
        if self.program_opts.ffmpeg_overwrite:
            return "-y"

    def get_color_range_opt(self):
        return ["-color_range", "tv"]

    def get_thread_queue_size_opt(self):
        return ["-thread_queue_size", str(self.program_opts.ffmpeg_thread_queue_size)]


class DecodePipeline:
    def __init__(self, dropout_correct_cmd=None, decoder_cmd=None, ffmpeg_cmd=None):
        self.dropout_correct_cmd = dropout_correct_cmd
        self.decoder_cmd = decoder_cmd
        self.ffmpeg_cmd = ffmpeg_cmd


class TBCVideoExport:
    def __init__(self):
        self.check_paths()
        self.ffmpeg_profiles = FFmpegProfiles(self.get_profile_file())
        self.program_opts = self.parse_opts(self.ffmpeg_profiles)

        self.files = InputFiles(self.program_opts.input, self.program_opts.input_json)
        self.files.check_files_exist()

        self.video_system = self.get_video_system(self.files.tbc_json_data)
        self.timecode = self.get_timecode(self.files.tbc_json_data)

        self.ffmpeg_profile = self.ffmpeg_profiles.get_profile(
            self.program_opts.ffmpeg_profile
        )
        self.ffmpeg_profile_luma = self.ffmpeg_profiles.get_profile(
            self.program_opts.ffmpeg_profile_luma
        )

        self.decoder_settings = DecoderSettings(self.program_opts, self.video_system)
        self.ffmpeg_settings = FFmpegSettings(
            self.program_opts,
            self.ffmpeg_profile,
            self.ffmpeg_profile_luma,
            self.files.tbc_json_data,
            self.video_system,
            self.timecode,
        )

    def run(self):
        self.setup_pipes()

        luma_pipeline = self.get_luma_cmds()
        chroma_pipeline = self.get_chroma_cmds()

        luma_pipeline.ffmpeg_cmd = self.get_luma_ffmepg_cmd()
        chroma_pipeline.ffmpeg_cmd = self.get_chroma_ffmpeg_cmd()

        if self.program_opts.what_if:
            self.print_cmds(luma_pipeline, chroma_pipeline)
            return

        if not self.program_opts.ffmpeg_overwrite:
            self.check_file_overwrites()

        # named pipes are only useful with chroma
        if self.program_opts.luma_only:
            self.run_cmds(luma_pipeline)
        else:
            if self.use_named_pipes:
                try:
                    self.create_pipes()
                    self.run_named_pipe_cmds(luma_pipeline, chroma_pipeline)
                finally:
                    self.cleanup_pipes()
            else:
                self.run_cmds(luma_pipeline)
                self.run_cmds(chroma_pipeline)

    def parse_opts(self, ffmpeg_profiles):
        parser = argparse.ArgumentParser(
            prog="tbc-video-export",
            description="vhs-decode video generation script",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        global_opts = parser.add_argument_group("global")
        decoder_opts = parser.add_argument_group("decoder")
        ffmpeg_opts = parser.add_argument_group("ffmpeg")

        # general / global arguments
        global_opts.add_argument("input", type=str, help="Name of the input tbc.")

        global_opts.add_argument(
            "-t",
            "--threads",
            type=int,
            default=int(os.cpu_count() / 2),
            metavar="int",
            help="Specify the number of concurrent threads.",
        )

        global_opts.add_argument(
            "--video-system",
            type=VideoSystem,
            choices=list(VideoSystem),
            metavar="format",
            help="Force a video system format. (default: from .tbc.json)\n"
            "Available formats:\n  " + "\n  ".join([e.value for e in VideoSystem]),
        )

        global_opts.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Do not suppress info and warning messages.",
        )

        global_opts.add_argument(
            "--what-if",
            action="store_true",
            default=False,
            help="Show what commands would be run without running them.",
        )

        global_opts.add_argument(
            "--skip-named-pipes",
            action="store_true",
            default=os.name not in ("posix", "nt"),
            help="Skip using named pipes and instead use a two-step process.",
        )

        # decoder arguments
        decoder_opts.add_argument(
            "-s",
            "--start",
            type=int,
            metavar="int",
            help="Specify the start frame number.",
        )

        decoder_opts.add_argument(
            "-l",
            "--length",
            type=int,
            metavar="int",
            help="Specify the number of frames to process.",
        )

        decoder_opts.add_argument(
            "--reverse",
            action="store_true",
            default=False,
            help="Reverse the field order to second/first.",
        )

        decoder_opts.add_argument(
            "--input-json",
            type=str,
            metavar="json_file",
            help="Use a different .tbc.json file.",
        )

        decoder_opts.add_argument(
            "--luma-only",
            action="store_true",
            default=False,
            help="Only output a luma video.",
        )

        decoder_opts.add_argument(
            "--output-padding",
            type=int,
            metavar="int",
            help="Pad the output frame to a multiple of this many pixels.",
        )

        decoder_opts.add_argument(
            "--vbi",
            action="store_true",
            default=False,
            help="Adjust FFLL/LFLL/FFRL/LFRL for full vertical export.",
        )

        decoder_opts.add_argument(
            "--letterbox",
            action="store_true",
            default=False,
            help="Adjust FFLL/LFLL/FFRL/LFRL for letterbox crop.",
        )

        decoder_opts.add_argument(
            "--first_active_field_line",
            type=int,
            metavar="int",
            help="The first visible line of a field.\n"
            "  Range 1-259 for NTSC (default: 20)\n"
            "        2-308 for PAL  (default: 22)",
        )

        decoder_opts.add_argument(
            "--last_active_field_line",
            type=int,
            metavar="int",
            help="The last visible line of a field.\n"
            "  Range 1-259 for NTSC (default: 259)\n"
            "        2-308 for PAL  (default: 308)",
        )

        decoder_opts.add_argument(
            "--first_active_frame_line",
            type=int,
            metavar="int",
            help="The first visible line of a field.\n"
            "  Range 1-525 for NTSC (default: 40)\n"
            "        1-620 for PAL  (default: 44)",
        )

        decoder_opts.add_argument(
            "--last_active_frame_line",
            type=int,
            metavar="int",
            help="The last visible line of a field.\n"
            "  Range 1-525 for NTSC (default: 525)\n"
            "        1-620 for PAL  (default: 620)",
        )

        decoder_opts.add_argument(
            "--offset",
            action="store_true",
            default=False,
            help="NTSC: Overlay the adaptive filter map (only used for testing).",
        )

        decoder_opts.add_argument(
            "--chroma-decoder",
            type=ChromaDecoder,
            choices=list(ChromaDecoder),
            metavar="decoder",
            help="Chroma decoder to use. "
            "(default: "
            + ChromaDecoder.TRANSFORM2D.value
            + " for PAL, "
            + ChromaDecoder.NTSC2D.value
            + " for NTSC).\n"
            "Available decoders:\n  " + "\n  ".join([e.value for e in ChromaDecoder]),
        )

        decoder_opts.add_argument(
            "--chroma-gain",
            type=float,
            default=1.0,
            metavar="float",
            help="Gain factor applied to chroma components.",
        )

        decoder_opts.add_argument(
            "--chroma-phase",
            type=float,
            default=0.0,
            metavar="float",
            help="Phase rotation applied to chroma components (degrees).",
        )

        decoder_opts.add_argument(
            "--chroma-nr",
            type=float,
            default=0.0,
            metavar="float",
            help="NTSC: Chroma noise reduction level in dB.",
        )

        decoder_opts.add_argument(
            "--luma-nr",
            type=float,
            default=1.0,
            metavar="float",
            help="Luma noise reduction level in dB.",
        )

        decoder_opts.add_argument(
            "--simple-pal",
            action="store_true",
            default=False,
            help="Transform: Use 1D UV filter.",
        )

        decoder_opts.add_argument(
            "--transform-threshold",
            type=float,
            default=0.4,
            metavar="float",
            help="Transform: Uniform similarity threshold in 'threshold' mode.",
        )

        decoder_opts.add_argument(
            "--transform-thresholds",
            type=str,
            metavar="file_name",
            help="Transform: File containing per-bin similarity thresholds in 'threshold' mode.",
        )

        decoder_opts.add_argument(
            "--show-ffts",
            action="store_true",
            default=False,
            help="Transform: Overlay the input and output FFTs.",
        )

        # ffmpeg arguments
        ffmpeg_opts.add_argument(
            "--ffmpeg-profile",
            type=str,
            choices=ffmpeg_profiles.names,
            default=next(iter(ffmpeg_profiles.names)),
            metavar="profile_name",
            help="Specify an FFmpeg profile to use. "
            "(default: " + next(iter(ffmpeg_profiles.names)) + ")\n"
            "Available profiles:\n  " + "\n  ".join(ffmpeg_profiles.names),
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-profile-luma",
            type=str,
            choices=ffmpeg_profiles.names_luma,
            default=next(iter(ffmpeg_profiles.names_luma)),
            metavar="profile_name",
            help="Specify an FFmpeg profile to use for luma. "
            "(default: " + next(iter(ffmpeg_profiles.names_luma)) + ")\n"
            "Available profiles:\n  " + "\n  ".join(ffmpeg_profiles.names_luma),
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-metadata",
            type=str,
            action="append",
            metavar='foo="bar"',
            help="Add metadata to output file.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-thread-queue-size",
            type=int,
            default=1024,
            metavar="int",
            help="Sets the thread queue size for FFmpeg. (default: 1024)",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-force-anamorphic",
            action="store_true",
            default=False,
            help="Force widescreen aspect ratio.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-overwrite",
            help="Set to overwrite existing video files.",
            action="store_true",
            default=False,
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-file",
            type=str,
            action="append",
            metavar="file_name",
            help="Audio file to mux with generated video.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-title",
            type=str,
            action="append",
            metavar="title",
            help="Title of the audio track.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-language",
            type=str,
            action="append",
            metavar="language",
            help="Language of the audio track.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-offset",
            type=str,
            action="append",
            metavar="offset",
            help="Offset of the audio track. (default: 00:00:00.000)",
        )

        return parser.parse_args()

    def flatten(self, A):
        """Flatten list of lists. Skips None values."""
        rt = []
        for i in A:
            if isinstance(i, list):
                rt.extend(self.flatten(i))
            elif i is not None:
                rt.append(i)
        return rt

    def print_windows_error(self, err=None):
        """Print out windows related errors."""
        if err is None:
            err = win32api.GetLastError()
        OSError(
            win32api.FormatMessage(win32con.FORMAT_MESSAGE_FROM_SYSTEM, 0, err, 0, None)
        )

    def cleanup_pipes(self):
        """Cleanup named pipes and any temp dirs."""
        try:
            if os.name == "posix":
                os.unlink(self.pipe_input_luma)
                os.unlink(self.pipe_input_chroma)
                os.rmdir(self.pipe_tmp_dir)
            elif os.name == "nt":
                # cleanup should be handled by the pipe bridge thread
                self.pipe_bridge_luma.join()
                self.pipe_bridge_chroma.join()
        except:
            raise Exception("unable to cleanup")

    def setup_win_pipe_bridge(self, input_name, output_name):
        input_pipe = win32pipe.CreateNamedPipe(
            input_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            (
                win32pipe.PIPE_TYPE_BYTE
                | win32pipe.PIPE_READMODE_BYTE
                | win32pipe.PIPE_WAIT
            ),
            1,
            65536,
            65536,
            0,
            None,
        )

        if input_pipe == win32file.INVALID_HANDLE_VALUE:
            self.print_windows_error()

        output_pipe = win32pipe.CreateNamedPipe(
            output_name,
            win32pipe.PIPE_ACCESS_DUPLEX,
            (
                win32pipe.PIPE_TYPE_BYTE
                | win32pipe.PIPE_READMODE_BYTE
                | win32pipe.PIPE_WAIT
            ),
            1,
            65536,
            65536,
            0,
            None,
        )

        if output_pipe == win32file.INVALID_HANDLE_VALUE:
            self.print_windows_error()

        if win32pipe.ConnectNamedPipe(input_pipe, None):
            self.print_windows_error()

        if win32pipe.ConnectNamedPipe(output_pipe, None):
            self.print_windows_error()

        while True:
            try:
                read_hr, read_buf = win32file.ReadFile(input_pipe, 65536)

                if read_hr in (winerror.ERROR_MORE_DATA, winerror.ERROR_IO_PENDING):
                    self.print_windows_error(read_hr)

                write_hr, written = win32file.WriteFile(output_pipe, read_buf)

                if write_hr:
                    self.print_windows_error(write_hr)

            except pywintypes.error:
                if win32api.GetLastError() == winerror.ERROR_BROKEN_PIPE:
                    # pipe is
                    break
                raise

        if win32file.CloseHandle(input_pipe):
            self.print_windows_error()
        if win32file.CloseHandle(output_pipe):
            self.print_windows_error()

    def setup_pipes(self):
        """Config named pipes for FFmpeg."""
        if not self.program_opts.skip_named_pipes:
            try:
                if os.name == "posix":
                    self.use_named_pipes = True

                    # we can use the same pipe for in/out on posix
                    self.pipe_tmp_dir = tempfile.mkdtemp(
                        prefix="tbc-video-export", suffix=""
                    )
                    self.pipe_input_luma = self.pipe_output_luma = os.path.join(
                        self.pipe_tmp_dir, "luma"
                    )
                    self.pipe_input_chroma = self.pipe_output_chroma = os.path.join(
                        self.pipe_tmp_dir, "chroma"
                    )
                elif os.name == "nt":
                    self.use_named_pipes = True

                    # we must create a named piped for decoder output and bridge it with ffmpeg input
                    self.pipe_input_luma = r"\\.\pipe\tbc-video-export-luma-input"
                    self.pipe_output_luma = r"\\.\pipe\tbc-video-export-luma-output"
                    self.pipe_input_chroma = r"\\.\pipe\tbc-video-export-chroma-input"
                    self.pipe_output_chroma = r"\\.\pipe\tbc-video-export-chroma-output"
                else:
                    raise Exception("named pipes not implemented for " + os.name)
            except:
                raise Exception("unable to setup pipes")
        else:
            self.use_named_pipes = False

    def create_pipes(self):
        """Create named pipes for FFmpeg."""
        try:
            if os.name == "posix":
                os.mkfifo(self.pipe_input_luma)
                os.mkfifo(self.pipe_input_chroma)
            elif os.name == "nt":
                self.pipe_bridge_luma = Thread(
                    target=self.setup_win_pipe_bridge,
                    args=(self.pipe_input_luma, self.pipe_output_luma),
                )
                self.pipe_bridge_chroma = Thread(
                    target=self.setup_win_pipe_bridge,
                    args=(self.pipe_input_chroma, self.pipe_output_chroma),
                )

                self.pipe_bridge_luma.start()
                self.pipe_bridge_chroma.start()
            else:
                raise Exception("named pipes not implemented for " + os.name)
        except:
            self.cleanup_pipes()
            raise Exception("unable to create pipes")

    def print_cmds(self, luma_pipeline, chroma_pipeline):
        """Print the full command arguments when using --what-if"""
        print("luma:")
        print(*luma_pipeline.dropout_correct_cmd)
        print(*luma_pipeline.decoder_cmd)

        if self.program_opts.luma_only or not self.use_named_pipes:
            print(*luma_pipeline.ffmpeg_cmd)

        print("---\n")

        if not self.program_opts.luma_only:
            print("chroma:")
            print(*chroma_pipeline.dropout_correct_cmd)
            print(*chroma_pipeline.decoder_cmd)
            print(*chroma_pipeline.ffmpeg_cmd)
            print("---\n")

    def run_cmds(self, pipeline):
        """Run ld-dropout-correct, ld-chroma-decoder and ffmpeg."""
        dropout_correct = subprocess.Popen(
            pipeline.dropout_correct_cmd, stdout=subprocess.PIPE
        )
        decoder = subprocess.Popen(
            pipeline.decoder_cmd, stdin=dropout_correct.stdout, stdout=subprocess.PIPE
        )
        ffmpeg = subprocess.Popen(pipeline.ffmpeg_cmd, stdin=decoder.stdout)

        ffmpeg.communicate()

    def run_named_pipe_cmds(self, luma_pipeline, chroma_pipeline):
        """Run ld-dropout-correct, ld-chroma-decoder and ffmpeg using a combination
        of subprocess pipes and named pipes. This allows us to use multiple pipes to
        ffmpeg and prevents a two-step process of merging luma and chroma."""
        # luma decoder procs
        luma_dropout_correct = subprocess.Popen(
            luma_pipeline.dropout_correct_cmd, stdout=subprocess.PIPE
        )
        luma_decoder = subprocess.Popen(
            luma_pipeline.decoder_cmd, stdin=luma_dropout_correct.stdout
        )

        # chroma decoder procs
        chroma_dropout_correct = subprocess.Popen(
            chroma_pipeline.dropout_correct_cmd, stdout=subprocess.PIPE
        )
        chroma_decoder = subprocess.Popen(
            chroma_pipeline.decoder_cmd, stdin=chroma_dropout_correct.stdout
        )

        # ffmpeg proc
        if self.program_opts.luma_only:
            ffmpeg = subprocess.Popen(luma_pipeline.ffmpeg_cmd)
            luma_decoder.communicate()
        else:
            ffmpeg = subprocess.Popen(chroma_pipeline.ffmpeg_cmd)
            luma_decoder.communicate()
            chroma_decoder.communicate()

        ffmpeg.wait()

    def get_luma_cmds(self):
        """Return ld-dropout-correct and ld-chroma-decode arguments for luma."""
        dropout_correct_cmd = ["ld-dropout-correct"]

        if not self.program_opts.verbose:
            dropout_correct_cmd.append("-q")

        dropout_correct_cmd.append(
            ["-i", self.files.tbc, "--output-json", os.devnull, "-"]
        )

        decoder_cmd = [
            "ld-chroma-decoder",
            self.decoder_settings.get_luma_opts(),
            "-p",
            "y4m",
            self.decoder_settings.get_opts(),
            "--input-json",
            self.files.tbc_json,
            "-",
        ]

        # we just pipe into ffmpeg if b/w
        if self.use_named_pipes and not self.program_opts.luma_only:
            decoder_cmd.append(self.pipe_input_luma)
        else:
            decoder_cmd.append("-")

        return DecodePipeline(
            self.flatten(dropout_correct_cmd), self.flatten(decoder_cmd)
        )

    def get_chroma_cmds(self):
        """Return ld-dropout-correct and ld-chroma-decode arguments for chroma."""
        dropout_correct_cmd = ["ld-dropout-correct"]

        if not self.program_opts.verbose:
            dropout_correct_cmd.append("-q")

        dropout_correct_cmd.append(
            [
                "-i",
                self.files.tbc_chroma,
                "--input-json",
                self.files.tbc_json,
                "--output-json",
                os.devnull,
                "-",
            ]
        )

        decoder_cmd = [
            "ld-chroma-decoder",
            self.decoder_settings.get_chroma_opts(),
            "-p",
            "y4m",
            self.decoder_settings.get_opts(),
            "--input-json",
            self.files.tbc_json,
            "-",
        ]

        if self.use_named_pipes:
            decoder_cmd.append(self.pipe_input_chroma)
        else:
            decoder_cmd.append("-")

        return DecodePipeline(
            self.flatten(dropout_correct_cmd), self.flatten(decoder_cmd)
        )

    def get_luma_ffmepg_cmd(self):
        """FFmpeg arguments for generating a luma-only video file."""
        file = (
            self.files.name
            + "_luma."
            + self.ffmpeg_settings.profile_luma.get_container()
        )

        ffmpeg_cmd = [
            "ffmpeg",
            self.ffmpeg_settings.get_verbosity(),
            self.ffmpeg_settings.get_overwrite_opt(),
            "-hwaccel",
            "auto",
            self.ffmpeg_settings.get_thread_queue_size_opt(),
            "-i",
            "-",
        ]

        if self.program_opts.luma_only:
            ffmpeg_cmd.append(self.ffmpeg_settings.get_audio_inputs_opts())

        ffmpeg_cmd.append(["-map", "0"])

        if self.program_opts.luma_only:
            ffmpeg_cmd.append(self.ffmpeg_settings.get_audio_map_opts())

        ffmpeg_cmd.append(
            [
                self.ffmpeg_settings.get_timecode_opt(),
                self.ffmpeg_settings.get_rate_opt(),
                self.ffmpeg_settings.profile_luma.get_video_opts(),
            ]
        )

        if self.program_opts.luma_only:
            ffmpeg_cmd.append(
                self.ffmpeg_settings.profile.get_audio_opts(),
            )

        ffmpeg_cmd.append(self.ffmpeg_settings.get_metadata_opts())

        if self.program_opts.luma_only:
            ffmpeg_cmd.append(self.ffmpeg_settings.get_audio_metadata_opts())

        ffmpeg_cmd.append(["-pass", "1", file])

        self.files.video_luma = file

        return self.flatten(ffmpeg_cmd)

    def get_chroma_ffmpeg_cmd(self):
        """FFmpeg arguments for generating a chroma video file. This will work
        with either a luma video file or multiple named pipes, depending on whether
        usz_named_pipes is true."""
        file = self.files.name + "." + self.ffmpeg_settings.profile.get_container()

        ffmpeg_cmd = [
            "ffmpeg",
            self.ffmpeg_settings.get_verbosity(),
            self.ffmpeg_settings.get_overwrite_opt(),
            "-hwaccel",
            "auto",
            self.ffmpeg_settings.get_color_range_opt(),
            self.ffmpeg_settings.get_thread_queue_size_opt(),
            "-i",
        ]

        if self.use_named_pipes:
            ffmpeg_cmd.append(self.pipe_output_luma)
        else:
            ffmpeg_cmd.append(self.files.video_luma)

        ffmpeg_cmd.append(
            [
                self.ffmpeg_settings.get_thread_queue_size_opt(),
                "-i",
            ]
        )

        if self.use_named_pipes:
            ffmpeg_cmd.append(self.pipe_output_chroma)
        else:
            ffmpeg_cmd.append("-")

        ffmpeg_cmd.append(
            [self.ffmpeg_settings.get_audio_inputs_opts(), "-filter_complex"]
        )

        # filters from existing scripts, can probably be tidied up
        if self.use_named_pipes is not None:
            ffmpeg_cmd.append(
                [
                    "[1:v]format="
                    + self.ffmpeg_settings.profile.get_video_format()
                    + "[chroma];"
                    + "[0:v][chroma]mergeplanes=0x001112:"
                    + self.ffmpeg_settings.profile.get_video_format()
                    + ",setfield=tff"
                    + self.ffmpeg_settings.profile.get_video_filter_opts()
                    + "[output]",
                ]
            )
        else:
            ffmpeg_cmd.append(
                [
                    "[0]format=pix_fmts="
                    + self.ffmpeg_settings.profile.get_video_format()
                    + ",extractplanes=y[y];"
                    "[1]format=pix_fmts="
                    + self.ffmpeg_settings.profile.get_video_format()
                    + ",extractplanes=u+v[u][v];"
                    "[y][u][v]mergeplanes=0x001020:"
                    + self.ffmpeg_settings.profile.get_video_format()
                    + ",format=pix_fmts="
                    + self.ffmpeg_settings.profile.get_video_format()
                    + ",setfield=tff"
                    + self.ffmpeg_settings.profile.get_video_filter_opts()
                    + "[output]"
                ]
            )

        ffmpeg_cmd.append(
            [
                "-map",
                "[output]:v",
                self.ffmpeg_settings.get_audio_map_opts(),
                self.ffmpeg_settings.get_timecode_opt(),
                self.ffmpeg_settings.get_rate_opt(),
                self.ffmpeg_settings.profile.get_video_opts(),
                self.ffmpeg_settings.get_aspect_ratio_opt(),
                self.ffmpeg_settings.get_color_range_opt(),
                self.ffmpeg_settings.get_color_opts(),
                self.ffmpeg_settings.profile.get_audio_opts(),
                self.ffmpeg_settings.get_metadata_opts(),
                self.ffmpeg_settings.get_audio_metadata_opts(),
                file,
            ]
        )

        self.files.video = file

        return self.flatten(ffmpeg_cmd)

    def check_file_overwrites(self):
        """Check if files exist with named pipes off/on and b/w off/on."""
        if self.use_named_pipes:
            if self.program_opts.luma_only:
                self.check_file_overwrite(self.files.video_luma)
            else:
                self.check_file_overwrite(self.files.video)
        else:
            if self.program_opts.luma_only:
                self.check_file_overwrite(self.files.video_luma)
            else:
                self.check_file_overwrite(self.files.video_luma)
                self.check_file_overwrite(self.files.video)

    def check_file_overwrite(self, file):
        """Check if a file exists and ask to run with overwrite."""
        if os.path.isfile(file):
            raise Exception(file + " exists, use --ffmpeg-overwrite or move them")

    def get_video_system(self, tbc_json_data):
        """Determine whether a TBC is PAL or NTSC."""

        # if user has forced a format
        if self.program_opts.video_system is not None:
            return self.program_opts.video_system

        # search for PAL* or NTSC* in videoParameters.system or the existence
        # if isSourcePal keys
        if (
            VideoSystem.PAL.value in tbc_json_data["videoParameters"]["system"].lower()
            or "isSourcePal" in tbc_json_data["videoParameters"]
            or "isSourcePalM" in tbc_json_data["videoParameters"]
        ):
            return VideoSystem.PAL
        elif (
            VideoSystem.NTSC.value in tbc_json_data["videoParameters"]["system"].lower()
        ):
            return VideoSystem.NTSC
        else:
            raise Exception("could not read video system from " + json_file)

    def get_timecode(self, tbc_json_data):
        """Attempt to read a VITC timecode for the first frame.
        Returns starting timecode if no VITC data found."""

        if (
            "vitc" not in tbc_json_data["fields"][0]
            or "vitcData" not in tbc_json_data["fields"][0]["vitc"]
        ):
            return "00:00:00:00"

        is_valid = True
        is_30_frame = self.video_system is not VideoSystem.PAL
        vitc_data = tbc_json_data["fields"][0]["vitc"]["vitcData"]

        def decode_bcd(tens, units):
            if tens > 9:
                is_valid = False
                tens = 9

            if units > 9:
                is_valid = False
                units = 9

            return (tens * 10) + units

        hour = decode_bcd(vitc_data[7] & 0x03, vitc_data[6] & 0x0F)
        minute = decode_bcd(vitc_data[5] & 0x07, vitc_data[4] & 0x0F)
        second = decode_bcd(vitc_data[3] & 0x07, vitc_data[2] & 0x0F)
        frame = decode_bcd(vitc_data[1] & 0x03, vitc_data[0] & 0x0F)

        if (
            hour > 23
            or minute > 59
            or second > 59
            or (is_30_frame and frame > 29)
            or (not is_30_frame and frame > 24)
        ):
            is_valid = False

        if is_30_frame:
            is_drop_frame = (vitc_data[1] & 0x04) != 0
            is_col_frame = (vitc_data[1] & 0x08) != 0
            is_field_mark = (vitc_data[3] & 0x08) != 0
        else:
            is_drop_frame = False
            is_col_frame = (vitc_data[1] & 0x08) != 0
            is_field_mark = (vitc_data[7] & 0x08) != 0

        if not is_valid:
            return "00:00:00:00"

        if is_drop_frame:
            sep = ";"
        else:
            sep = ":"

        return f"{hour:02d}:{minute:02d}:{second:02d}{sep}{frame:02d}"

    def get_profile_file(self):
        """Returns name of json file to load profiles from. Checks for existence of
        a .custom file. Checks both . and the dir the script is run from."""

        file_name_stock = "tbc-video-export.json"
        file_name_custom = "tbc-video-export.custom.json"

        # check custom file
        if os.path.isfile(file_name_custom):
            return file_name_custom

        path = pathlib.Path(__file__).with_name(file_name_custom).absolute()

        if os.path.isfile(path):
            return path

        # check stock file
        if os.path.isfile(file_name_stock):
            return file_name_stock

        path = pathlib.Path(__file__).with_name(file_name_stock).absolute()

        if os.path.isfile(path):
            return path

        raise Exception("Unable to find profile config file")

    def check_paths(self):
        """Ensure required binaries are in PATH."""
        for tool in ["ld-dropout-correct", "ld-chroma-decoder", "ffmpeg"]:
            if not which(tool):
                raise Exception(tool + " not in PATH")


def main():
    tbc_video_export = TBCVideoExport()
    tbc_video_export.run()


if __name__ == "__main__":
    main()
