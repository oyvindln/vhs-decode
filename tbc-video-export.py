#!/usr/bin/env python3

# Script for exporting TBC files to video
# - JuniorIsAJitterbug

import argparse
import contextlib
import json
import os
import subprocess
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from shutil import which
from threading import Thread

if os.name == "nt":
    import win32pipe
    import win32api
    import win32file
    import pywintypes
    import winerror
    import win32con


class TBCVideoExport:
    def __init__(self):
        self.ffmpeg_profiles = FFmpegProfiles(Files.get_profile_file())
        self.program_opts = self.__parse_opts(self.ffmpeg_profiles)

        self.files = Files(
            self.program_opts.input,
            self.program_opts.output,
            self.program_opts.input_json,
            self.program_opts.skip_process_vbi
        )
        self.files.check_files_exist()

        # if user has forced a format
        if self.program_opts.video_system is not None:
            self.video_system = self.program_opts.video_system
        else:
            self.video_system = self.files.get_video_system()

        if not self.program_opts.ffmpeg_skip_auto_pcm:
            self.__add_pcm_audio()

        self.ldtools_wrapper = LDToolsWrapper(self.program_opts, self.video_system)
        self.ffmpeg_wrapper = FFmpegWrapper(
            self.program_opts,
            self.files,
            self.ffmpeg_profiles,
            self.video_system
        )

        # default to using named pipes unless user asks not to
        self.use_named_pipes = not self.program_opts.skip_named_pipes

    def run(self):
        process_vbi_cmd = None
        combined_pipeline = None
        luma_pipeline = None
        chroma_pipeline = None

        try:
            self.__setup_pipes()

            # set up commands
            if not self.program_opts.skip_process_vbi:
                process_vbi_cmd = self.ldtools_wrapper.get_process_vbi_cmd(self.files)

            if self.files.is_combined_tbc:
                combined_pipeline = self.ldtools_wrapper.get_combined_cmds(self.files)
                combined_pipeline.ffmpeg_cmd = self.ffmpeg_wrapper.get_combined_ffmpeg_cmd(
                    self.files
                )
            else:
                luma_pipeline = self.ldtools_wrapper.get_luma_cmds(self.files, self.use_named_pipes)

                # we do not use ffmpeg for luma unless skipping named pipes, or luma only
                if self.program_opts.luma_only or not self.use_named_pipes:
                    luma_pipeline.ffmpeg_cmd = self.ffmpeg_wrapper.get_luma_ffmepg_cmd(self.files)

                if not self.program_opts.luma_only:
                    chroma_pipeline = self.ldtools_wrapper.get_chroma_cmds(
                        self.files,
                        self.use_named_pipes
                    )
                    chroma_pipeline.ffmpeg_cmd = self.ffmpeg_wrapper.get_chroma_ffmpeg_cmd(
                        self.files,
                        self.use_named_pipes
                    )

            if self.program_opts.what_if:
                self.__print_pipelines(
                    process_vbi_cmd,
                    combined_pipeline,
                    luma_pipeline,
                    chroma_pipeline
                )
                return

            if not self.program_opts.ffmpeg_overwrite:
                self.files.check_file_overwrites(self.program_opts.luma_only, self.use_named_pipes)

            # run commands
            if not self.program_opts.skip_process_vbi:
                self.__run_process_vbi(process_vbi_cmd)

            if self.files.is_combined_tbc:
                self.__run_cmds(combined_pipeline)
            elif self.program_opts.luma_only:
                self.__run_cmds(luma_pipeline)
            elif self.use_named_pipes:
                self.__create_pipes()
                self.__run_named_pipe_cmds(luma_pipeline, chroma_pipeline)
            else:
                self.__run_cmds(luma_pipeline)
                self.__run_cmds(chroma_pipeline)
        finally:
            self.__cleanup_pipes()

    def __parse_opts(self, ffmpeg_profiles):
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
            "-o",
            "--output",
            type=str,
            default=".",
            metavar="file_name",
            help="Output directory. (default: current directory)"
        )

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
            "--ffll",
            type=int,
            metavar="int",
            help="The first visible line of a field.\n"
            "  Range 1-259 for NTSC (default: 20)\n"
            "        2-308 for PAL  (default: 22)",
        )

        decoder_opts.add_argument(
            "--last_active_field_line",
            "--lfll",
            type=int,
            metavar="int",
            help="The last visible line of a field.\n"
            "  Range 1-259 for NTSC (default: 259)\n"
            "        2-308 for PAL  (default: 308)",
        )

        decoder_opts.add_argument(
            "--first_active_frame_line",
            "--ffrl",
            type=int,
            metavar="int",
            help="The first visible line of a field.\n"
            "  Range 1-525 for NTSC (default: 40)\n"
            "        1-620 for PAL  (default: 44)",
        )

        decoder_opts.add_argument(
            "--last_active_frame_line",
            "--lfrl",
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
            metavar="float",
            help="Gain factor applied to chroma components.",
        )

        decoder_opts.add_argument(
            "--chroma-phase",
            type=float,
            metavar="float",
            help="Phase rotation applied to chroma components (degrees).",
        )

        decoder_opts.add_argument(
            "--chroma-nr",
            type=float,
            metavar="float",
            help="NTSC: Chroma noise reduction level in dB.",
        )

        decoder_opts.add_argument(
            "--luma-nr",
            type=float,
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

        decoder_opts.add_argument(
            "--skip-process-vbi",
            action="store_true",
            default=False,
            help="Skip running ld-process-vbi before export."
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
            "--ffmpeg-skip-auto-pcm",
            help="Skip adding PCM audio if available.",
            action="store_true",
            default=False,
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-file",
            type=str,
            action="append",
            default=[],
            metavar="file_name",
            help="Audio file to mux with generated video.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-title",
            type=str,
            action="append",
            default=[],
            metavar="title",
            help="Title of the audio track.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-language",
            type=str,
            action="append",
            default=[],
            metavar="language",
            help="Language of the audio track.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-format",
            type=str,
            action="append",
            default=[],
            metavar="format",
            help="Format of the audio track.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-rate",
            type=str,
            action="append",
            default=[],
            metavar="rate",
            help="Rate of the audio track.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-channels",
            type=str,
            action="append",
            default=[],
            metavar="channels",
            help="Channel count of the audio track.",
        )

        ffmpeg_opts.add_argument(
            "--ffmpeg-audio-offset",
            type=str,
            action="append",
            default=[],
            metavar="offset",
            help="Offset of the audio track. (default: 00:00:00.000)",
        )

        return parser.parse_args()

    def __print_windows_error(self, err=None):
        """Print out windows related errors."""
        if err is None:
            err = win32api.GetLastError()
        raise OSError(
            win32api.FormatMessage(win32con.FORMAT_MESSAGE_FROM_SYSTEM, 0, err, 0, None)
        )

    def __cleanup_pipes(self):
        """Cleanup named pipes and any temp dirs."""
        try:
            if os.name == "posix":
                # we suppress FileNotFoundError so other files can be cleaned up
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(self.files.pipes.input_luma)

                with contextlib.suppress(FileNotFoundError):
                    os.unlink(self.files.pipes.input_chroma)

                with contextlib.suppress(FileNotFoundError):
                    os.rmdir(self.files.pipes.tmp_dir)
            elif os.name == "nt":
                # cleanup should be handled by the pipe bridge thread
                if self.files.pipes.bridge_luma is not None:
                    self.files.pipes.bridge_luma.join()

                if self.files.pipes.bridge_chroma is not None:
                    self.files.pipes.bridge_chroma.join()
        # we're exiting anyway, not sure this matters...
        except PermissionError as e:
            raise SystemExit("unable to cleanup named pipes due to permissions") from e
        except RuntimeError as e:
            raise SystemExit("unable to cleanup threads") from e

    def __setup_win_pipe_bridge(self, input_name, output_name):
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
            self.__print_windows_error()

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
            self.__print_windows_error()

        if win32pipe.ConnectNamedPipe(input_pipe, None):
            self.__print_windows_error()

        if win32pipe.ConnectNamedPipe(output_pipe, None):
            self.__print_windows_error()

        while True:
            try:
                read_hr, read_buf = win32file.ReadFile(input_pipe, 65536)

                if read_hr in (winerror.ERROR_MORE_DATA, winerror.ERROR_IO_PENDING):
                    self.__print_windows_error(read_hr)

                write_hr, _ = win32file.WriteFile(output_pipe, read_buf)

                if write_hr:
                    self.__print_windows_error(write_hr)

            except pywintypes.error:
                if win32api.GetLastError() == winerror.ERROR_BROKEN_PIPE:
                    # pipe is
                    break
                raise

        if win32file.CloseHandle(input_pipe):
            self.__print_windows_error()
        if win32file.CloseHandle(output_pipe):
            self.__print_windows_error()

    def __setup_pipes(self):
        """Config named pipes for FFmpeg.
        This is only needed for separated tbc exports."""
        try:
            if os.name == "posix":
                # we can use the same pipe for in/out on posix
                self.files.pipes.tmp_dir = tempfile.mkdtemp(
                    prefix="tbc-video-export-", suffix=""
                )
                self.files.pipes.input_luma = self.files.pipes.output_luma = os.path.join(
                    self.files.pipes.tmp_dir, "luma"
                )
                self.files.pipes.input_chroma = self.files.pipes.output_chroma = os.path.join(
                    self.files.pipes.tmp_dir, "chroma"
                )
            elif os.name == "nt":
                # we must create a named piped for decoder output and bridge it with ffmpeg input
                self.files.pipes.input_luma = r"\\.\pipe\tbc-video-export-luma-input"
                self.files.pipes.output_luma = r"\\.\pipe\tbc-video-export-luma-output"
                self.files.pipes.input_chroma = r"\\.\pipe\tbc-video-export-chroma-input"
                self.files.pipes.output_chroma = r"\\.\pipe\tbc-video-export-chroma-output"

                self.files.pipes.bridge_luma = None
                self.files.pipes.bridge_chroma = None
            else:
                # disable named pipes on other os
                self.use_named_pipes = False
        except PermissionError as e:
            raise SystemExit(
                "unable to create pipes due to permissions, consider using --skip-named-pipes"
            ) from e

    def __create_pipes(self):
        """Create named pipes for FFmpeg."""
        try:
            if os.name == "posix":
                os.mkfifo(self.files.pipes.input_luma)
                os.mkfifo(self.files.pipes.input_chroma)
            elif os.name == "nt":
                self.files.pipes.bridge_luma = Thread(
                    target=self.__setup_win_pipe_bridge,
                    args=(self.files.pipes.input_luma, self.files.pipes.output_luma),
                )
                self.files.pipes.bridge_chroma = Thread(
                    target=self.__setup_win_pipe_bridge,
                    args=(self.files.pipes.input_chroma, self.files.pipes.output_chroma),
                )

                self.files.pipes.bridge_luma.start()
                self.files.pipes.bridge_chroma.start()
        except PermissionError as e:
            raise SystemExit(
                "unable to create pipes due to permissions, consider using --skip-named-pipes"
            ) from e
        except RuntimeError as e:
            raise SystemExit(
                "unable to create pipe bridge threads, consider using --skip-named-pipes"
            ) from e

    def __print_pipelines(self, *pipelines):
        """Print the full command arguments when using --what-if"""

        for pipeline in pipelines:
            if pipeline is not None:
                # if given a cmd just print
                if isinstance(pipeline, list):
                    print(*pipeline)
                else:
                    print(*pipeline.dropout_correct_cmd)
                    print(*pipeline.decoder_cmd)

                    if pipeline.ffmpeg_cmd is not None:
                        print(*pipeline.ffmpeg_cmd)

                print()

    def __run_process_vbi(self, cmd):
        """Run ld-process-vbi."""
        with subprocess.Popen(cmd) as process:
            process.wait()

    def __run_cmds(self, pipeline):
        """Run ld-dropout-correct, ld-chroma-decoder and ffmpeg."""
        with (
            subprocess.Popen(
                pipeline.dropout_correct_cmd, stdout=subprocess.PIPE
            ) as dropout_correct,
            subprocess.Popen(
                pipeline.decoder_cmd, stdin=dropout_correct.stdout, stdout=subprocess.PIPE
            ) as decoder,
            subprocess.Popen(pipeline.ffmpeg_cmd, stdin=decoder.stdout) as ffmpeg
        ):
            ffmpeg.communicate()

    def __run_named_pipe_cmds(self, luma_pipeline, chroma_pipeline):
        """Run ld-dropout-correct, ld-chroma-decoder and ffmpeg using a combination
        of subprocess pipes and named pipes. This allows us to use multiple pipes to
        ffmpeg and prevents a two-step process of merging luma and chroma."""
        with (
            # luma decoder procs
            subprocess.Popen(
                luma_pipeline.dropout_correct_cmd, stdout=subprocess.PIPE
            ) as luma_dropout_correct,
            subprocess.Popen(
                luma_pipeline.decoder_cmd, stdin=luma_dropout_correct.stdout
            ) as luma_decoder,

            # chroma decoder procs
            subprocess.Popen(
                chroma_pipeline.dropout_correct_cmd, stdout=subprocess.PIPE
            ) as chroma_dropout_correct,
            subprocess.Popen(
                chroma_pipeline.decoder_cmd, stdin=chroma_dropout_correct.stdout
            ) as chroma_decoder,

            # ffmpeg proc
            subprocess.Popen(chroma_pipeline.ffmpeg_cmd) as ffmpeg
        ):
            luma_decoder.communicate()
            chroma_decoder.communicate()

            ffmpeg.wait()

    def __add_pcm_audio(self):
        """Adds PCM audio to program_opts.ffmpeg_audio_* if available.
        I'm not sure if these values can change, will update to read from
        tbc json if required."""
        if self.files.pcm_exists:
            self.program_opts.ffmpeg_audio_file.insert(0, self.files.pcm)
            self.program_opts.ffmpeg_audio_language.insert(0, None)
            self.program_opts.ffmpeg_audio_title.insert(0, "PCM")
            self.program_opts.ffmpeg_audio_format.insert(0, "s16le")
            self.program_opts.ffmpeg_audio_rate.insert(0, "44.1k")
            self.program_opts.ffmpeg_audio_channels.insert(0, "2")


class VideoSystem(Enum):
    PAL = "pal"
    PALM = "pal-m"
    NTSC = "ntsc"

    def __str__(self):
        return self.value


class ChromaDecoder(Enum):
    PAL2D = "pal2d"
    TRANSFORM2D = "transform2d"
    TRANSFORM3D = "transform3d"
    MONO = "mono"
    NTSC1D = "ntsc1d"
    NTSC2D = "ntsc2d"
    NTSC3D = "ntsc3d"
    NTSC3DNOADAPT = "ntsc3dnoadapt"

    def __str__(self):
        return self.value


class Files:
    @dataclass
    class Pipes:
        tmp_dir: str = None
        input_luma: str = None
        input_chroma: str = None
        output_luma: str = None
        output_chroma: str = None
        bridge_luma: Thread = None
        bridge_chroma: Thread = None

    def __init__(self, file, output_dir, input_json, skip_process_vbi):
        self.path = pathlib.Path(file).parent
        self.name = pathlib.Path(file).stem

        # input files
        self.tbc = os.path.join(self.path, self.name + ".tbc")
        self.tbc_chroma = os.path.join(self.path, self.name + "_chroma.tbc")
        self.pcm = os.path.join(self.path, self.name + ".pcm")

        self.pipes = self.Pipes()
        self.pcm_exists = False
        self.is_combined_tbc = False

        # output files
        self.output_dir = output_dir
        self.video_luma = None
        self.video = None

        self.tools = self.__get_tool_paths(skip_process_vbi)

        if input_json is not None:
            self.tbc_json = input_json
        else:
            self.tbc_json = os.path.join(self.path, self.name + ".tbc.json")

        try:
            with open(self.tbc_json, mode="r", encoding="utf-8") as json_file:
                self.tbc_json_data = json.load(json_file)
        except FileNotFoundError as e:
            raise SystemExit("tbc json not found (" + self.tbc_json + ")") from e
        except PermissionError as e:
            raise SystemExit("permission denied opening tbc json (" + self.tbc_json + ")") from e
        except json.JSONDecodeError as e:
            raise SystemExit("unable to parse tbc json (" + self.tbc_json + ")") from e

        if not os.path.isdir(output_dir):
            raise SystemExit("output directory does not exist (" + output_dir + ")")

    def get_output_path(self, filename):
        return os.path.join(self.output_dir, filename)

    def check_files_exist(self):
        # check for tbc json
        if not os.path.isfile(self.tbc_json):
            raise SystemExit("tbc json not found (" + self.tbc_json + ")")

        # check for chroma tbc file
        if not os.path.isfile(self.tbc_chroma):
            self.is_combined_tbc = True

        # check for tbc file
        if not os.path.isfile(self.tbc):
            raise SystemExit("tbc not found (" + self.tbc + ")")

        # check for pcm file
        if os.path.isfile(self.pcm):
            self.pcm_exists = True

    def check_file_overwrites(self, luma_only, use_named_pipes):
        """Check if files exist with named pipes off/on and b/w off/on."""
        if luma_only or not use_named_pipes:
            self.__check_file_overwrite(self.video_luma)

        self.__check_file_overwrite(self.video)

    def get_video_system(self):
        """Determine whether a TBC is PAL or NTSC."""

        # search for PAL* or NTSC* in videoParameters.system or the existence
        # if isSourcePal keys
        if (
            VideoSystem.PAL.value == self.tbc_json_data["videoParameters"]["system"].lower()
            or "isSourcePal" in self.tbc_json_data["videoParameters"]
        ):
            return VideoSystem.PAL

        if (
            VideoSystem.PALM.value == self.tbc_json_data["videoParameters"]["system"].lower()
            or "isSourcePalM" in self.tbc_json_data["videoParameters"]
        ):
            return VideoSystem.PALM

        if (
            VideoSystem.NTSC.value in self.tbc_json_data["videoParameters"]["system"].lower()
        ):
            return VideoSystem.NTSC

        raise SystemExit("could not read video system from tbc json")

    def get_timecode(self, video_system):
        """Attempt to read a VITC timecode for the first frame.
        Returns starting timecode if no VITC data found."""

        if (
            "vitc" not in self.tbc_json_data["fields"][0]
            or "vitcData" not in self.tbc_json_data["fields"][0]["vitc"]
        ):
            return "00:00:00:00"

        is_valid = True
        is_30_frame = video_system is not VideoSystem.PAL
        vitc_data = self.tbc_json_data["fields"][0]["vitc"]["vitcData"]

        def decode_bcd(tens, units):
            nonlocal is_valid

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
            #is_col_frame = (vitc_data[1] & 0x08) != 0
            #is_field_mark = (vitc_data[3] & 0x08) != 0
        else:
            is_drop_frame = False
            #is_col_frame = (vitc_data[1] & 0x08) != 0
            #is_field_mark = (vitc_data[7] & 0x08) != 0

        if not is_valid:
            return "00:00:00:00"

        if is_drop_frame:
            sep = ";"
        else:
            sep = ":"

        return f"{hour:02d}:{minute:02d}:{second:02d}{sep}{frame:02d}"

    def __check_file_overwrite(self, file):
        """Check if a file exists and ask to run with overwrite."""
        if os.path.isfile(file):
            raise SystemExit(file + " exists, use --ffmpeg-overwrite or move them")

    def __get_tool_paths(self, skip_process_vbi):
        """Get required tool paths from PATH or script path."""
        tool_names = ["ld-dropout-correct", "ld-chroma-decoder", "ffmpeg"]
        tools = {}

        if not skip_process_vbi:
            tool_names.append("ld-process-vbi")

        for tool_name in tool_names:
            binary = tool_name

            if os.name == "nt":
                binary += ".exe"

            # check if tool exists in the same dir as script
            script_path = Files.get_runtime_directory().with_name(binary).absolute()

            if os.path.isfile(script_path):
                tools[tool_name] = script_path
            # check if tool exists in PATH or current dir
            elif which(binary):
                tools[tool_name] = binary

            if not tool_name in tools:
                raise SystemExit(tool_name + " not in PATH or script dir")

        return tools

    @staticmethod
    def get_runtime_directory():
        """Returns the runtime directory. When the script is built to a single
        executable using PyInstaller __file__ is somewhere in TEMP, so the executable
        location must be used instead."""
        if getattr(sys, "frozen", False):
            return pathlib.Path(sys.executable)

        return pathlib.Path(__file__)

    @staticmethod
    def get_profile_file():
        """Returns name of json file to load profiles from. Checks for existence of
        a .custom file. Checks both . and the dir the script is run from."""

        file_name_stock = "tbc-video-export.json"
        file_name_custom = "tbc-video-export.custom.json"

        # check custom file
        if os.path.isfile(file_name_custom):
            return file_name_custom

        path = Files.get_runtime_directory().with_name(file_name_custom).absolute()

        if os.path.isfile(path):
            return path

        # check stock file
        if os.path.isfile(file_name_stock):
            return file_name_stock

        path = Files.get_runtime_directory().with_name(file_name_stock).absolute()

        if os.path.isfile(path):
            return path

        raise SystemExit("Unable to find profile config file")


class LDToolsWrapper:
    def __init__(self, program_opts, video_system):
        self.program_opts = program_opts
        self.video_system = video_system

    def get_luma_cmds(self, files, use_named_pipes):
        """Return ld-dropout-correct and ld-chroma-decode arguments for luma."""
        dropout_correct_cmd = [files.tools["ld-dropout-correct"]]

        if not self.program_opts.verbose:
            dropout_correct_cmd.append("-q")

        dropout_correct_cmd.append(
            ["-i", files.tbc, "--output-json", os.devnull, "-"]
        )

        decoder_cmd = [
            files.tools["ld-chroma-decoder"],
            self.__get_decoder_opts(skip_luma=False, skip_chroma=True),
            "-p",
            "y4m",
            self.__get_chroma_decoder(is_luma=True),
            self.__get_vbi_opts(),
            self.__get_opts(),
            "--input-json",
            files.tbc_json,
            "-",
        ]

        # we just pipe into ffmpeg if b/w
        if use_named_pipes and not self.program_opts.luma_only:
            decoder_cmd.append(files.pipes.input_luma)
        else:
            decoder_cmd.append("-")

        return DecodePipeline(
            flatten(dropout_correct_cmd), flatten(decoder_cmd)
        )

    def get_chroma_cmds(self, files, use_named_pipes):
        """Return ld-dropout-correct and ld-chroma-decode arguments for chroma."""
        dropout_correct_cmd = [files.tools["ld-dropout-correct"]]

        if not self.program_opts.verbose:
            dropout_correct_cmd.append("-q")

        dropout_correct_cmd.append(
            [
                "-i",
                files.tbc_chroma,
                "--input-json",
                files.tbc_json,
                "--output-json",
                os.devnull,
                "-",
            ]
        )

        decoder_cmd = [
            files.tools["ld-chroma-decoder"],
            self.__get_decoder_opts(skip_luma=True, skip_chroma=False),
            "-p",
            "y4m",
            self.__get_chroma_decoder(is_luma=False),
            self.__get_vbi_opts(),
            self.__get_opts(),
            "--input-json",
            files.tbc_json,
            "-",
        ]

        if use_named_pipes:
            decoder_cmd.append(files.pipes.input_chroma)
        else:
            decoder_cmd.append("-")

        return DecodePipeline(
            flatten(dropout_correct_cmd), flatten(decoder_cmd)
        )

    def get_combined_cmds(self, files):
        """Return ld-dropout-correct and ld-chroma-decode arguments for combined tbc decoding."""
        dropout_correct_cmd = [files.tools["ld-dropout-correct"]]

        if not self.program_opts.verbose:
            dropout_correct_cmd.append("-q")

        dropout_correct_cmd.append(
            [
                "-i",
                files.tbc,
                "--input-json",
                files.tbc_json,
                "--output-json",
                os.devnull,
                "-",
            ]
        )

        decoder_cmd = [
            files.tools["ld-chroma-decoder"],
            self.__get_decoder_opts(skip_luma=False, skip_chroma=False),
            "-p",
            "y4m",
            self.__get_chroma_decoder(is_luma=False),
            self.__get_vbi_opts(),
            self.__get_opts(),
            "--input-json",
            files.tbc_json,
            "-",
            "-"
        ]

        return DecodePipeline(
            flatten(dropout_correct_cmd), flatten(decoder_cmd)
        )

    def get_process_vbi_cmd(self, files):
        """Return ld-process-vbi arguments."""
        process_vbi_cmd = [files.tools["ld-process-vbi"]]

        if not self.program_opts.verbose:
            process_vbi_cmd.append("-q")

        process_vbi_cmd.append(
            [
                "-t",
                str(self.program_opts.threads),
                "--input-json",
                files.tbc_json,
                files.tbc
            ]
        )

        return flatten(process_vbi_cmd)

    def __get_chroma_decoder(self, is_luma):
        """Get chroma decoder opts."""
        decoder_opts = []

        if is_luma:
            decoder_opts.append(["-f", ChromaDecoder.MONO.value])
        elif self.program_opts.chroma_decoder is None:
            # set default chroma decoder if unset
            if self.video_system in (VideoSystem.PAL, VideoSystem.PALM):
                decoder_opts.append(["-f", ChromaDecoder.TRANSFORM2D.value])
            elif self.video_system is VideoSystem.NTSC:
                decoder_opts.append(["-f", ChromaDecoder.NTSC2D.value])
        else:
            decoder_opts.append(
                self.__convert_opt(self.program_opts, "chroma_decoder", "-f")
            )

        return decoder_opts

    def __get_vbi_opts(self):
        """Get VBI opts."""
        decoder_opts = []

        if self.video_system is VideoSystem.PAL:
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
        elif self.video_system in (VideoSystem.NTSC, VideoSystem.PALM):
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

        if not self.program_opts.vbi and not self.program_opts.letterbox:
            decoder_opts.append(
                self.__convert_opt(self.program_opts, "first_active_field_line", "--ffll")
            )
            decoder_opts.append(
                self.__convert_opt(self.program_opts, "last_active_field_line", "--lfll")
            )
            decoder_opts.append(
                self.__convert_opt(self.program_opts, "first_active_frame_line", "--ffrl")
            )
            decoder_opts.append(
                self.__convert_opt(self.program_opts, "last_active_frame_line", "--lfrl")
            )

        return decoder_opts


    def __get_opts(self):
        """Generate ld-chroma-decoder opts."""
        decoder_opts = []

        if not self.program_opts.verbose:
            decoder_opts.append("-q")

        if self.video_system is VideoSystem.NTSC:
            decoder_opts.append("--ntsc-phase-comp")

        decoder_opts.append(self.__convert_opt(self.program_opts, "start", "-s"))
        decoder_opts.append(self.__convert_opt(self.program_opts, "length", "-l"))
        decoder_opts.append(self.__convert_opt(self.program_opts, "reverse", "-r"))
        decoder_opts.append(self.__convert_opt(self.program_opts, "threads", "-t"))
        decoder_opts.append(
            self.__convert_opt(self.program_opts, "output_padding", "--pad")
        )
        decoder_opts.append(self.__convert_opt(self.program_opts, "offset", "-o"))
        decoder_opts.append(
            self.__convert_opt(self.program_opts, "simple_pal", "--simple-pal")
        )
        decoder_opts.append(
            self.__convert_opt(self.program_opts, "show_ffts", "--show-ffts")
        )
        decoder_opts.append(
            self.__convert_opt(
                self.program_opts, "transform_threshold", "--transform-threshold"
            )
        )
        decoder_opts.append(
            self.__convert_opt(
                self.program_opts, "transform_thresholds", "--transform-thresholds"
            )
        )

        return decoder_opts

    def __get_decoder_opts(self, skip_luma, skip_chroma):
        """Generate ld-chroma-decoder opts."""
        decoder_opts = []

        if skip_luma:
            # when skipping luma we want this set to 0
            decoder_opts.append(["--luma-nr", "0"])
        else:
            decoder_opts.append(self.__convert_opt(self.program_opts, "luma_nr", "--luma-nr"))

        if skip_chroma:
            # when skipping chroma we want this set to 0
            decoder_opts.append(["--chroma-gain", "0"])
        else:
            decoder_opts.append(
                self.__convert_opt(self.program_opts, "chroma_gain", "--chroma-gain")
            )
            decoder_opts.append(
                self.__convert_opt(self.program_opts, "chroma_nr", "--chroma-nr")
            )
            decoder_opts.append(
                self.__convert_opt(self.program_opts, "chroma_phase", "--chroma-phase")
            )

        return decoder_opts

    def __convert_opt(self, program_opts, program_opt_name, target_opt_name):
        """Converts a program opt to a subprocess opt."""
        rt = []
        value = getattr(program_opts, program_opt_name)

        if value is not None:
            if isinstance(value, bool):
                # only appends opt on true, fine for current use
                if value is True:
                    rt.append([target_opt_name])
            else:
                rt.append([target_opt_name, str(value)])

        return rt

class FFmpegProfile:
    def __init__(self, profiles, name):
        self.name = name
        self.profile = profiles.profiles[name]

    def get_video_opts(self):
        """Return FFmpeg video opts from profile."""
        rt = []

        if not all(key in self.profile for key in ("v_codec", "v_format", "container")):
            raise SystemExit("ffmpeg profile is missing required data")

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
            with open(file_name, mode="r", encoding="utf-8") as file:
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
        except FileNotFoundError as e:
            raise SystemExit("profile json not found (" + file_name+ ")") from e
        except PermissionError as e:
            raise SystemExit("permission denied opening profile json (" + file_name + ")") from e
        except json.JSONDecodeError as e:
            raise SystemExit("unable to parse profile json (" + file_name + ")") from e

    def get_profile(self, name):
        return FFmpegProfile(self, name)


class FFmpegWrapper:
    def __init__(
        self, program_opts, files, profiles, video_system
    ):
        self.program_opts = program_opts
        self.files = files
        self.video_system = video_system

        self.profile = profiles.get_profile(
            self.program_opts.ffmpeg_profile
        )
        self.profile_luma = profiles.get_profile(
            self.program_opts.ffmpeg_profile_luma
        )
        self.timecode = self.files.get_timecode(self.video_system)

    def get_luma_ffmepg_cmd(self, files):
        """FFmpeg arguments for generating a luma-only video file."""
        file_name = (
            files.name
            + "_luma."
            + self.profile_luma.get_container()
        )

        file = files.get_output_path(file_name)

        ffmpeg_cmd = [
            files.tools["ffmpeg"],
            self.__get_verbosity(),
            self.__get_overwrite_opt(),
            self.__get_threads(),
            "-hwaccel",
            "auto",
            self.__get_thread_queue_size_opt(),
            "-i",
            "-",
        ]

        if self.program_opts.luma_only:
            ffmpeg_cmd.append(self.__get_audio_inputs_opts())

        ffmpeg_cmd.append(["-map", "0"])

        if self.program_opts.luma_only:
            ffmpeg_cmd.append(self.__get_audio_map_opts(1))

        ffmpeg_cmd.append(
            [
                self.__get_timecode_opt(),
                self.__get_rate_opt(),
                self.profile_luma.get_video_opts(),
                self.__get_attachment_opts()
            ]
        )

        if self.program_opts.luma_only:
            ffmpeg_cmd.append(
                self.profile.get_audio_opts(),
            )

        ffmpeg_cmd.append(self.__get_metadata_opts())

        if self.program_opts.luma_only:
            ffmpeg_cmd.append(self.__get_audio_metadata_opts())

        ffmpeg_cmd.append(["-pass", "1", file])

        files.video_luma = file

        return flatten(ffmpeg_cmd)

    def get_chroma_ffmpeg_cmd(self, files, use_named_pipes):
        """FFmpeg arguments for generating a chroma video file. This will work
        with either a luma video file or multiple named pipes, depending on whether
        use_named_pipes is true."""
        file_name = files.name + "." + self.profile.get_container()
        file = files.get_output_path(file_name)

        ffmpeg_cmd = [
            files.tools["ffmpeg"],
            self.__get_verbosity(),
            self.__get_overwrite_opt(),
            self.__get_threads(),
            "-hwaccel",
            "auto",
            self.__get_color_range_opt(),
            self.__get_thread_queue_size_opt(),
            "-i",
        ]

        if use_named_pipes:
            ffmpeg_cmd.append(files.pipes.output_luma)
        else:
            ffmpeg_cmd.append(files.video_luma)

        ffmpeg_cmd.append(
            [
                self.__get_thread_queue_size_opt(),
                "-i",
            ]
        )

        if use_named_pipes:
            ffmpeg_cmd.append(files.pipes.output_chroma)
        else:
            ffmpeg_cmd.append("-")

        ffmpeg_cmd.append(
            [self.__get_audio_inputs_opts(), "-filter_complex"]
        )

        # filters from existing scripts, can probably be tidied up
        if use_named_pipes is not None:
            ffmpeg_cmd.append(
                [
                    "[1:v]format="
                    + self.profile.get_video_format()
                    + "[chroma];"
                    + "[0:v][chroma]mergeplanes=0x001112:"
                    + self.profile.get_video_format()
                    + ",setfield=tff"
                    + self.profile.get_video_filter_opts()
                    + "[output]",
                ]
            )
        else:
            ffmpeg_cmd.append(
                [
                    "[0]format=pix_fmts="
                    + self.profile.get_video_format()
                    + ",extractplanes=y[y];"
                    "[1]format=pix_fmts="
                    + self.profile.get_video_format()
                    + ",extractplanes=u+v[u][v];"
                    "[y][u][v]mergeplanes=0x001020:"
                    + self.profile.get_video_format()
                    + ",format=pix_fmts="
                    + self.profile.get_video_format()
                    + ",setfield=tff"
                    + self.profile.get_video_filter_opts()
                    + "[output]"
                ]
            )

        ffmpeg_cmd.append(
            [
                "-map",
                "[output]:v",
                self.__get_audio_map_opts(2),
                self.__get_timecode_opt(),
                self.__get_rate_opt(),
                self.profile.get_video_opts(),
                self.__get_aspect_ratio_opt(),
                self.__get_color_range_opt(),
                self.__get_color_opts(),
                self.profile.get_audio_opts(),
                self.__get_metadata_opts(),
                self.__get_attachment_opts(),
                self.__get_audio_metadata_opts(),
                file,
            ]
        )

        files.video = file

        return flatten(ffmpeg_cmd)

    def get_combined_ffmpeg_cmd(self, files):
        """FFmpeg arguments for generating a video file from a combined tbc.
        This is for use with cvbs-decoder and ld-decoder."""
        file_name = files.name + "." + self.profile.get_container()
        file = files.get_output_path(file_name)

        ffmpeg_cmd = [
            files.tools["ffmpeg"],
            self.__get_verbosity(),
            self.__get_overwrite_opt(),
            self.__get_threads(),
            "-hwaccel",
            "auto",
            self.__get_color_range_opt(),
            self.__get_thread_queue_size_opt(),
            "-i",
            "-",
            self.__get_audio_inputs_opts(),
            "-map",
            "0:v",
            self.__get_audio_map_opts(1),
            self.__get_timecode_opt(),
            self.__get_rate_opt(),
            self.profile.get_video_opts(),
            self.__get_aspect_ratio_opt(),
            self.__get_color_range_opt(),
            self.__get_color_opts(),
            self.profile.get_audio_opts(),
            self.__get_metadata_opts(),
            self.__get_attachment_opts(),
            self.__get_audio_metadata_opts(),
            file,
        ]

        files.video = file

        return flatten(ffmpeg_cmd)

    def __get_rate_opt(self):
        """Returns FFmpeg opts for rate."""
        ffmpeg_opts = []

        if self.video_system == VideoSystem.PAL:
            rate = 25
        elif self.video_system in (VideoSystem.NTSC, VideoSystem.PALM):
            rate = 29.97

        if self.profile.get_video_doublerate():
            rate = rate * 2

        ffmpeg_opts.append(["-r", str(rate)])

        return ffmpeg_opts

    def __get_aspect_ratio_opt(self):
        """Returns FFmpeg opts for aspect ratio."""
        if (
            (
                "isWidescreen" in self.files.tbc_json_data["videoParameters"]
                and self.files.tbc_json_data["videoParameters"]["isWidescreen"]
            )
            or self.program_opts.ffmpeg_force_anamorphic
            or self.program_opts.letterbox
        ):
            return ["-aspect", "16:9"]

        return ["-aspect", "4:3"]

    def __get_color_opts(self):
        """Returns FFmpeg opts for color settings."""
        ffmpeg_opts = []

        if self.video_system in (VideoSystem.PAL, VideoSystem.PALM):
            ffmpeg_opts.append(["-colorspace", "bt470bg"])
            ffmpeg_opts.append(["-color_primaries", "bt470bg"])
            ffmpeg_opts.append(["-color_trc", "bt709"])
        elif self.video_system == VideoSystem.NTSC:
            ffmpeg_opts.append(["-colorspace", "smpte170m"])
            ffmpeg_opts.append(["-color_primaries", "smpte170m"])
            ffmpeg_opts.append(["-color_trc", "bt709"])

        return ffmpeg_opts

    def __get_audio_map_opts(self, offset):
        """Returns FFmpeg opts for audio mapping."""
        ffmpeg_opts = []

        audio_inputs = self.program_opts.ffmpeg_audio_file

        if audio_inputs is not None:
            for idx, _ in enumerate(audio_inputs):
                ffmpeg_opts.append(["-map", str(idx + offset) + ":a"])

        return ffmpeg_opts

    def __get_metadata_opts(self):
        """Returns FFmpeg opts for metadata."""
        ffmpeg_opts = []

        metadata = self.program_opts.ffmpeg_metadata

        # add video metadata
        if metadata is not None:
            for data in metadata:
                ffmpeg_opts.append(["-metadata", data])

        return ffmpeg_opts

    def __get_attachment_opts(self):
        """Returns FFmpeg opts for attachments.
        Only available for MKV containers"""
        ffmpeg_opts = []

        if self.profile.get_container().lower() == "mkv":
            # add tbc json
            ffmpeg_opts.append([
                "-attach",
                self.files.tbc_json,
                "-metadata:s:t",
                "mimetype=application/json"
            ])

        return ffmpeg_opts


    def __get_audio_metadata_opts(self):
        """Returns FFmpeg opts for audio metadata."""
        ffmpeg_opts = []

        audio_titles = self.program_opts.ffmpeg_audio_title
        audio_languages = self.program_opts.ffmpeg_audio_language

        # add audio metadata only if luma only
        if audio_titles is not None:
            for idx, title in enumerate(audio_titles):
                if title is not None:
                    ffmpeg_opts.append(
                        ["-metadata:s:a:" + str(idx), 'title="' + title + '"']
                    )

        if audio_languages is not None:
            for idx, language in enumerate(audio_languages):
                if language is not None:
                    ffmpeg_opts.append(
                        ["-metadata:s:a:" + str(idx), 'language="' + language + '"']
                    )

        return ffmpeg_opts

    def __get_audio_inputs_opts(self):
        """Returns FFmpeg audio input opts."""
        input_opts = []

        tracks = self.program_opts.ffmpeg_audio_file
        offsets = self.program_opts.ffmpeg_audio_offset
        formats = self.program_opts.ffmpeg_audio_format
        rates = self.program_opts.ffmpeg_audio_rate
        channels = self.program_opts.ffmpeg_audio_channels

        if tracks is not None:
            for idx, track in enumerate(tracks):
                # add offset if set
                if len(offsets) >= idx + 1 and offsets[idx] is not None:
                    input_opts.append(["-itsoffset", offsets[idx]])
                else:
                    input_opts.append(["-itsoffset", "00:00:00.000"])

                if len(formats) >= idx + 1 and formats[idx] is not None:
                    input_opts.append(["-f", formats[idx]])

                if len(rates) >= idx + 1 and rates[idx] is not None:
                    input_opts.append(["-ar", rates[idx]])

                if len(channels) >= idx + 1 and channels[idx] is not None:
                    input_opts.append(["-ac", channels[idx]])

                input_opts.append(["-i", track])

        return input_opts

    def __get_timecode_opt(self):
        return ["-timecode", self.timecode]

    def __get_verbosity(self):
        if not self.program_opts.verbose:
            return ["-hide_banner", "-loglevel", "error", "-stats"]

        return "-hide_banner"

    def __get_overwrite_opt(self):
        if self.program_opts.ffmpeg_overwrite:
            return "-y"
        return None

    def __get_color_range_opt(self):
        return ["-color_range", "tv"]

    def __get_thread_queue_size_opt(self):
        return ["-thread_queue_size", str(self.program_opts.ffmpeg_thread_queue_size)]

    def __get_threads(self):
        return ["-threads", str(self.program_opts.threads)]


class DecodePipeline:
    def __init__(self, dropout_correct_cmd=None, decoder_cmd=None, ffmpeg_cmd=None):
        self.dropout_correct_cmd = dropout_correct_cmd
        self.decoder_cmd = decoder_cmd
        self.ffmpeg_cmd = ffmpeg_cmd


def flatten(arr):
    """Flatten list of lists. Skips None values."""
    rt = []
    for i in arr:
        if isinstance(i, list):
            rt.extend(flatten(i))
        elif i is not None:
            rt.append(i)
    return rt

def main():
    tbc_video_export = TBCVideoExport()
    tbc_video_export.run()

if __name__ == "__main__":
    main()
