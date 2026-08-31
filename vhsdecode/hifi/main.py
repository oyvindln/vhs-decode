#!/usr/bin/env python3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import (
    cpu_count,
    Pipe,
    SimpleQueue,
    Process,
    freeze_support,
    current_process,
    Event,
)
from multiprocessing.sharedctypes import Value
from datetime import datetime, timedelta
import os
import sys
import threading
from typing import Optional
import signal
from numba import njit
import numba
import atexit
import asyncio
from setproctitle import setproctitle
from contextlib import nullcontext

import numpy as np
import soundfile as sf

from vhsdecode.hifi.utils import (
    NUMA,
    DecoderSharedMemory,
    DecoderState,
    PostProcessorSharedMemory,
    PeakGain,
    REAL_DTYPE,
    cleanup_process
)
from vhsdecode.hifi.format_scaling import (
    get_normalizer
)

import argparse
from vhsdecode.cmdcommons import (
    test_input_file,
    test_output_file,
    TestInputFile,
    TestOutputFile,
)
from vhsdecode.hifi.constants import (
    AUDIO_MODE_DUAL_MONO,
    AUDIO_MODE_DUAL_MONO_MS,
    FORMAT_TO_DTYPE,
    FORMAT_U8,
    FORMAT_U10_LE,
    FORMAT_U12_LE,
    FORMAT_U16_LE,
    FORMAT_S8,
    FORMAT_S10_LE,
    FORMAT_S12_LE,
    FORMAT_S16_LE,
    FORMAT_F32_LE,
    FORMAT_TO_DTYPE,
    ENV_DETECTION_PEAK,
    ENV_DETECTION_RMS,
    DEFAULT_ENV_DETECTION,
    DEFAULT_8MM_AUDIO_MODE,
    DEFAULT_8MM_DEEMPHASIS_TAU_1,
    DEFAULT_8MM_DEEMPHASIS_TAU_2,
    DEFAULT_8MM_EXPANDER_ATTACK_TAU,
    DEFAULT_8MM_EXPANDER_GAIN,
    DEFAULT_8MM_EXPANDER_HOLD_TAU,
    DEFAULT_8MM_EXPANDER_RATIO,
    DEFAULT_8MM_EXPANDER_RELEASE_TAU,
    DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS,
    DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION,
    DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_1,
    DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_2,
    DEFAULT_8MM_NR_DEEMPHASIS_TAU_1,
    DEFAULT_8MM_NR_DEEMPHASIS_TAU_2,
    DEFAULT_DEMOD,
    DEFAULT_DOC_MODE,
    DEFAULT_FINAL_AUDIO_RATE,
    DEFAULT_RESAMPLER_QUALITY,
    DEFAULT_SPECTRAL_NR_AMOUNT,
    DEFAULT_VHS_AUDIO_MODE,
    DEFAULT_VHS_DEEMPHASIS_TAU_1,
    DEFAULT_VHS_DEEMPHASIS_TAU_2,
    DEFAULT_VHS_EXPANDER_ATTACK_TAU,
    DEFAULT_VHS_EXPANDER_GAIN,
    DEFAULT_VHS_EXPANDER_HOLD_TAU,
    DEFAULT_VHS_EXPANDER_RATIO,
    DEFAULT_VHS_EXPANDER_RELEASE_TAU,
    DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS,
    DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION,
    DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1,
    DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2,
    DEFAULT_VHS_NR_DEEMPHASIS_TAU_1,
    DEFAULT_VHS_NR_DEEMPHASIS_TAU_2,
    DEMOD_HILBERT,
    DEMOD_QUADRATURE,
    DOC_MODE_DISABLED,
    DOC_MODE_FULL,
    DOC_MODE_MUTE,
    audio_mode_to_ui,
)
from vhsdecode.hifi.TimeProgressBar import TimeProgressBar
import io
sd = None
SOUNDDEVICE_AVAILABLE = None

from vhsdecode.hifi.HiFiDecode import HiFiDecode
from vhsdecode.hifi.PostProcessor import PostProcessor

def _linux_portaudio_search_roots() -> list[str]:
    roots = []
    appdir = os.environ.get("APPDIR")
    if appdir:
        roots.extend(
            [
                appdir,
                os.path.join(appdir, "usr", "lib"),
                os.path.join(appdir, "usr", "lib64"),
                os.path.join(appdir, "lib"),
                os.path.join(appdir, "lib64"),
            ]
        )

    meipass = getattr(sys, "_MEIPASS", None)
    if isinstance(meipass, str) and meipass:
        roots.extend(
            [
                meipass,
                os.path.join(meipass, "usr", "lib"),
                os.path.join(meipass, "usr", "lib64"),
                os.path.join(meipass, "lib"),
                os.path.join(meipass, "lib64"),
            ]
        )

    executable_dir = os.path.dirname(sys.executable)
    if executable_dir:
        roots.extend(
            [
                executable_dir,
                os.path.join(executable_dir, "lib"),
                os.path.join(executable_dir, "..", "lib"),
            ]
        )

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        roots.extend(
            [entry for entry in ld_library_path.split(os.pathsep) if entry]
        )

    return roots


def _iter_linux_portaudio_candidates():
    seen_paths = set()
    for root in _linux_portaudio_search_roots():
        normalized_root = os.path.realpath(root)
        if normalized_root in seen_paths:
            continue
        seen_paths.add(normalized_root)

        for fallback_name in ("libportaudio.so.2", "libportaudio.so"):
            candidate_path = os.path.join(normalized_root, fallback_name)
            if os.path.isfile(candidate_path):
                yield candidate_path

    for fallback_name in ("libportaudio.so.2", "libportaudio.so"):
        yield fallback_name


def _import_sounddevice_with_linux_portaudio_fallback():
    import ctypes.util

    original_find_library = ctypes.util.find_library
    last_error = None

    for candidate in _iter_linux_portaudio_candidates():
        def _find_library_with_fallback(name, _candidate=candidate):
            if name == "portaudio":
                return _candidate
            return original_find_library(name)

        ctypes.util.find_library = _find_library_with_fallback
        try:
            sys.modules.pop("sounddevice", None)
            import sounddevice as imported_sounddevice
            return imported_sounddevice
        except (ImportError, OSError) as error:
            last_error = error
        finally:
            ctypes.util.find_library = original_find_library

    if last_error is not None:
        raise last_error
    raise OSError("PortAudio library not found")


def _sounddevice_available():
    global sd, SOUNDDEVICE_AVAILABLE

    if SOUNDDEVICE_AVAILABLE is None:
        imported_sounddevice = None
        is_linux = sys.platform.startswith("linux")
        prefer_bundled_portaudio = is_linux and (
            bool(os.environ.get("APPDIR")) or bool(getattr(sys, "_MEIPASS", None))
        )

        if prefer_bundled_portaudio:
            try:
                imported_sounddevice = (
                    _import_sounddevice_with_linux_portaudio_fallback()
                )
            except (ImportError, OSError):
                imported_sounddevice = None

        if imported_sounddevice is None:
            try:
                import sounddevice as imported_sounddevice
            except (ImportError, OSError):
                if is_linux:
                    try:
                        imported_sounddevice = (
                            _import_sounddevice_with_linux_portaudio_fallback()
                        )
                    except (ImportError, OSError):
                        imported_sounddevice = None
                else:
                    imported_sounddevice = None

        if imported_sounddevice is None:
            SOUNDDEVICE_AVAILABLE = False
        else:
            sd = imported_sounddevice
            SOUNDDEVICE_AVAILABLE = True

    return SOUNDDEVICE_AVAILABLE


frequency_suffixes = [
    ("ghz", 1.0e9),
    ("mhz", 1.0e6),
    ("khz", 1.0e3),
    ("hz", 1.0),
    ("fsc", 315.0e6 / 88.0),
    ("fscpal", (283.75 * 15625) + 25),
]


def parse_frequency(string):
    multiplier = 1.0e6
    for suffix, mult in frequency_suffixes:
        if string.lower().endswith(suffix):
            multiplier = mult
            string = string[: -len(suffix)]
            break
    return (multiplier * float(string)) / 1.0e6

try:
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        from PyQt5.QtWidgets import QApplication, QMessageBox

    from vhsdecode.hifi.HifiUi import (
        ui_parameters_to_decode_options,
        decode_options_to_ui_parameters,
        FileIODialogUI,
        FileOutputDialogUI,
    )

    HIFI_UI = True
except ImportError as e:
    print(e)
    HIFI_UI = False

STOP_STATE = 0
PLAY_STATE = 1
PAUSE_STATE = 2
PREVIEW_STATE = 3

STOP_NOT_REQUESTED = 0
STOP_REQUESTED = 1
STOP_IMMEDIATE_REQUESTED = 2

DEFAULT_CHANNEL_SUFFIX = "channel"
NORMALIZE_FILE_SUFFIX = "tmp_normalize.raw"


default_threads = cpu_count()
parser = argparse.ArgumentParser(
    description="Extracts audio from RAW HiFi FM RF captures",
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    "infile",
    metavar="infile",
    type=str,
    help="source file path or '-' for stdin\n  NOTE: `--raw_format` is required for stdin",
    nargs="?",
    default="",
    action=TestInputFile,
)
parser.add_argument(
    "outfile",
    metavar="outfile",
    type=str,
    help="output file",
    nargs="?",
    default="",
    action=TestOutputFile,
)
parser.add_argument(
    "--frequency",
    "-f",
    dest="inputfreq",
    metavar='',
    type=parse_frequency,
    default=40,
    help="RF sampling frequency in source file (default is 40MHz)",
)
parser.add_argument(
    "--overwrite",
    dest="overwrite",
    action="store_true",
    default=False,
    help="Overwrite existing decode files.",
)
parser.add_argument(
    "--threads",
    "-t",
    metavar='',
    type=int,
    default=default_threads,
    help="number of CPU threads to use",
)
parser.add_argument(
    "--preview",
    dest="preview",
    action="store_true",
    default=False,
    help="Preview the audio through your speakers as it decodes. Uses preview quality (faster and noisier)",
)
parser.add_argument(
    "--gui",
    dest="UI",
    action="store_true",
    default=False,
    help="Opens hifi-decode GUI graphical user interface",
)
parser.add_argument(
    "--gnuradio",
    dest="GRC",
    action="store_true",
    default=False,
    help="Opens ZMQ REP pipe to gnuradio at port 5555",
)
parser.add_argument(
    "--raw_format",
    dest="raw_format",
    metavar='',
    type=str,
    default=None,
    help=f'Specify RF input data precision. Packed data is unsupported.'
    '\n  Required for stdin.'
    '\n  This option can be used to override the precision of the input file for better normalization.'
    '\n  (i.e. when decoding 10bit data is stored in 16bit FLAC, specifying `--raw_format s10le` will normalize the input data as 10bit)'
    '\nInput Formats:'
    f'\n  {FORMAT_U8}    \t8 bit unsigned integer'
    f'\n  {FORMAT_U16_LE}  \t16 bit unsigned integer'
    f'\n  {FORMAT_U10_LE}  \t16 bit (10 bit precision) unsigned integer'
    f'\n  {FORMAT_U12_LE}  \t16 bit (12 bit precision) unsigned integer'
    f'\n  {FORMAT_S8}    \t8 bit signed integer'
    f'\n  {FORMAT_S16_LE}  \t16 bit signed integer'
    f'\n  {FORMAT_S10_LE}  \t16 bit (10 bit precision) signed integer'
    f'\n  {FORMAT_S12_LE}  \t16 bit (12 bit precision) signed integer'
    f'\n  {FORMAT_F32_LE}  \t32 bit floating point'
)

system_options_group = parser.add_argument_group("System options")
system_options_group.add_argument(
    "--pal",
    "-p",
    dest="pal",
    action="store_true",
    help="source is in PAL format",
)
system_options_group.add_argument(
    "--ntsc",
    "-n",
    dest="ntsc",
    action="store_true",
    help="source is in NTSC format",
)
system_options_group.add_argument(
    "--8mm",
    dest="format_8mm",
    action="store_true",
    default=False,
    help="Use settings for Video8 and Hi8 tape formats.",
)

demod_options = parser.add_argument_group("Demodulation options")
demod_options.add_argument(
    "--demod",
    dest="demod_type",
    metavar='',
    type=str.lower,
    default=DEFAULT_DEMOD,
    help=f"Set the FM demodulation type \n  {DEMOD_QUADRATURE} [default]\n  {DEMOD_HILBERT}",
)
demod_options.add_argument(
    "--bias_guess",
    "--bg",
    dest="bias_guess",
    action="store_true",
    default=False,
    help="Do carrier bias guess",
)
demod_options.add_argument(
    "--auto_fine_tune",
    dest="auto_fine_tune",
    metavar='',
    type=str.lower,
    default="off",
    help="Set auto tuning of the analog front end on/off",
)
demod_options.add_argument(
    "--AFE_left_carrier",
    dest="afe_left_carrier",
    metavar='',
    type=parse_frequency,
    default=0,
    help="Overrides the left carrier center frequency.",
)
demod_options.add_argument(
    "--AFE_left_carrier_deviation",
    dest="afe_left_carrier_deviation",
    metavar='',
    type=parse_frequency,
    default=0,
    help="Overrides the left carrier maximum deviation. This represents the maximum frequency offset + or - from the center frequency.",
)
demod_options.add_argument(
    "--AFE_right_carrier",
    dest="afe_right_carrier",
    metavar='',
    type=parse_frequency,
    default=0,
    help="Overrides the right carrier center frequency.",
)
demod_options.add_argument(
    "--AFE_right_carrier_deviation",
    dest="afe_right_carrier_deviation",
    metavar='',
    type=parse_frequency,
    default=0,
    help="Overrides the right carrier maximum deviation. This represents the maximum frequency offset + or - from the center frequency.",
)

audio_processing_options_group = parser.add_argument_group("Audio processing options")
audio_processing_options_group.add_argument(
    "--normalize",
    dest="normalize",
    action="store_true",
    default=False,
    help=f"(recommended) Automatically amplifies the audio to the peak gain of the decode. \n"
    f'This will create a temporary file ending in "{NORMALIZE_FILE_SUFFIX}" that is deleted after the amplification step is complete.',
)
audio_processing_options_group.add_argument(
    "--gain",
    dest="gain",
    type=float,
    default=1.0,
    metavar='',
    help="Manually adjust the gain/volume of the output audio. \n  1.0 [default]",
)
audio_mode_help_lines = []
for mode_key, mode_description in audio_mode_to_ui.items():
    mode_suffix = ""
    if mode_key == DEFAULT_VHS_AUDIO_MODE:
        mode_suffix += " [VHS default]"
    if mode_key == DEFAULT_8MM_AUDIO_MODE:
        mode_suffix += " [8mm default]"
    audio_mode_help_lines.append(
        f"  {mode_key} \t{mode_description}{mode_suffix}"
    )
audio_mode_help = "Audio modes \n" + "\n".join(audio_mode_help_lines)
audio_processing_options_group.add_argument(
    "--audio_mode",
    dest="mode",
    type=str.lower,
    metavar='',
    help=audio_mode_help,
)
audio_processing_options_group.add_argument(
    "--audio_rate",
    "--ar",
    dest="rate",
    metavar='',
    type=int,
    default=DEFAULT_FINAL_AUDIO_RATE,
    help=f"Output sample rate in Hz \n  192000\n  96000\n  {DEFAULT_FINAL_AUDIO_RATE} [default]\n  41000",
)
audio_processing_options_group.add_argument(
    "--resampler_quality",
    dest="resampler_quality",
    type=str,
    default=DEFAULT_RESAMPLER_QUALITY,
    metavar='',
    help=f"Sets quality of resampling to use in the audio chain."
    f"\n  high \tslowest, best quality [default]\n  medium\n  low \tfast, low quality",
)

noise_reduction_options_group = parser.add_argument_group("Noise reduction options")
noise_reduction_options_group.add_argument(
    "--head_switching_interpolation",
    dest="head_switching_interpolation",
    type=str.lower,
    default="on",
    metavar='',
    help='Enables head switching noise interpolation. \n  on \tenabled [default]\n  off \tdisabled'
)
noise_reduction_options_group.add_argument(
    "--doc",
    dest="doc",
    type=str.lower,
    default=DEFAULT_DOC_MODE,
    metavar='',
    help=f'Dropout compensation method (what happens when there is no hifi carrier) \n  {DOC_MODE_FULL} \tcopies audio from the other channel, or mutes if both channels have a dropout [default]\n  {DOC_MODE_MUTE} \talways mute dropouts\n  {DOC_MODE_DISABLED} \tdisabled'
)
noise_reduction_options_group.add_argument(
    "--NR_spectral_amount",
    dest="spectral_nr_amount",
    type=float,
    default=DEFAULT_SPECTRAL_NR_AMOUNT,
    metavar='',
    help=f"Sets the amount of broadband spectral noise reduction to apply"
    f"\n  0.0 \toff\n  {DEFAULT_SPECTRAL_NR_AMOUNT} \t[default]\n  1.0 \tfull spectral noise reduction",
)

expander_options_group = parser.add_argument_group("Expander tuning options (advanced)")
expander_options_group.add_argument(
    "--expander",
    dest="enable_expander",
    type=str.lower,
    default="on",
    metavar='',
    help="Enable / Disable Expander \n  on \tenabled [default]\n  off \tdisabled",
)
expander_options_group.add_argument(
    "--expander_gain",
    dest="expander_gain",
    type=float,
    metavar='',
    help=f"Sets the expander gain \n  {DEFAULT_VHS_EXPANDER_GAIN} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_GAIN}\t[8mm default]",
)
expander_options_group.add_argument(
    "--expander_ratio",
    dest="expander_ratio",
    type=float,
    metavar='',
    help=f"Sets the expander ratio \n  {DEFAULT_VHS_EXPANDER_RATIO} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_RATIO}\t[8mm default]",
)
expander_options_group.add_argument(
    "--expander_env_detection",
    dest="expander_env_detection",
    type=str.lower,
    metavar='',
    help=f'Set the expander envelope detection method  \n  {ENV_DETECTION_PEAK} \tuses peak detection (JVC standard, IEC 60774-2) [default]\n  {ENV_DETECTION_RMS} \tuses rms detection (Some Panasonic and early VCRs)'
)
expander_options_group.add_argument(
    "--expander_attack_tau",
    dest="expander_attack_tau",
    type=float,
    metavar='',
    help=f"Sets the expander attack speed in seconds \n  {DEFAULT_VHS_EXPANDER_ATTACK_TAU} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_ATTACK_TAU}\t[8mm default]",
)
expander_options_group.add_argument(
    "--expander_hold_tau",
    dest="expander_hold_tau",
    type=float,
    metavar='',
    help=f"Sets the expander hold time in seconds \n  {DEFAULT_VHS_EXPANDER_HOLD_TAU} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_HOLD_TAU}\t[8mm default]",
)
expander_options_group.add_argument(
    "--expander_release_tau",
    dest="expander_release_tau",
    type=float,
    metavar='',
    help=f"Sets the expander release speed in seconds \n  {DEFAULT_VHS_EXPANDER_RELEASE_TAU} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_RELEASE_TAU}\t[8mm default]",
)
expander_options_group.add_argument(
    "--expander_weighting_low_tau",
    dest="expander_weighting_low_tau",
    type=float,
    metavar='',
    help=f"Sets the expander weighting high-pass shelf filter low point in tau \n  {DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_1}\t[8mm default]",
)
expander_options_group.add_argument(
    "--expander_weighting_high_tau",
    dest="expander_weighting_high_tau",
    type=float,
    metavar='',
    help=f"Sets the expander weighting high-pass shelf filter high point in tau \n  {DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_2}\t[8mm default]",
)
expander_options_group.add_argument(
    "--expander_weighting_low_pass",
    dest="expander_weighting_low_pass",
    type=float,
    metavar='',
    help=f"Sets the expander weighting low-pass filter cutoff frequency in Hz \n  {DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS}\t[8mm default]",
)
expander_options_group.add_argument(
    "--expander_weighting_low_pass_transition",
    dest="expander_weighting_low_pass_transition",
    type=float,
    metavar='',
    help=f"Sets the expander weighting low-pass filter cutoff rate in Hz \n  {DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION} \t[VHS default]\n  {DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION}\t[8mm default]",
)

deemphasis_options_group = parser.add_argument_group(
    "Deemphasis tuning options (advanced)"
)
deemphasis_options_group.add_argument(
    "--deemphasis",
    dest="enable_deemphasis",
    type=str.lower,
    default="on",
    metavar='',
    help="Enable / Disable Deemphasis \n  on \tenabled [default]\n  off \tdisabled",
)
deemphasis_options_group.add_argument(
    "--deemphasis_low_tau",
    dest="deemphasis_low_tau",
    type=float,
    metavar='',
    help=f"Sets the deemphasis low-pass shelf filter low point in tau \n  {DEFAULT_VHS_DEEMPHASIS_TAU_1} \t[VHS default]\n  {DEFAULT_8MM_DEEMPHASIS_TAU_1}\t[8mm default]",
)
deemphasis_options_group.add_argument(
    "--deemphasis_high_tau",
    dest="deemphasis_high_tau",
    type=float,
    metavar='',
    help=f"Sets the deemphasis low-pass shelf filter high point in tau \n  {DEFAULT_VHS_DEEMPHASIS_TAU_2} \t[VHS default]\n  {DEFAULT_8MM_DEEMPHASIS_TAU_2}\t[8mm default]",
)
deemphasis_options_group.add_argument(
    "--nr_deemphasis_low_tau",
    dest="nr_deemphasis_low_tau",
    type=float,
    metavar='',
    help=f"Sets the noise reduction deemphasis low-pass shelf filter low point in tau \n  {DEFAULT_VHS_NR_DEEMPHASIS_TAU_1} \t[VHS default]\n  {DEFAULT_8MM_NR_DEEMPHASIS_TAU_1}\t[8mm default]",
)
deemphasis_options_group.add_argument(
    "--nr_deemphasis_high_tau",
    dest="nr_deemphasis_high_tau",
    type=float,
    metavar='',
    help=f"Sets the noise reduction deemphasis low-pass shelf filter high point in tau \n  {DEFAULT_VHS_NR_DEEMPHASIS_TAU_2} \t[VHS default]\n  {DEFAULT_8MM_NR_DEEMPHASIS_TAU_2}\t[8mm default]",
)

def test_ld_tools(ld_tool):
    shell_command = [ld_tool, "--help"]
    try:
        p = subprocess.Popen(
            shell_command,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )
        print(f"Found {ld_tool}")
        p.communicate()
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"WARN: {ld_tool} not installed (or not in PATH)")
        return False

def test_if_flac_is_installed():
    shell_command = ["flac", "-version"]
    try:
        p = subprocess.Popen(
            shell_command,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )
        # prints ffmpeg version and closes the process
        stdout_as_string = p.stdout.read().decode("utf-8")
        version = stdout_as_string.split("\n")[0]
        print(f"Found {version}")
        p.communicate()
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("WARN: flac not installed (or not in PATH)")
        return False


def test_if_ffmpeg_is_installed():
    shell_command = ["ffmpeg", "-version"]
    try:
        p = subprocess.Popen(
            shell_command,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )
        # prints ffmpeg version and closes the process
        stdout_as_string = p.stdout.read().decode("utf-8")
        version = stdout_as_string.split("\n")[0]
        print(f"Found {version}")
        p.communicate()
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("WARN: ffmpeg not installed (or not in PATH)")
        return False


class BufferedInputStream(io.RawIOBase):
    def __init__(self, buffer):
        self.buffer = buffer
        self._pos = 0

    def readinto(self, buffer):
        bytes_read = self.buffer.readinto(buffer)
        if not bytes_read:
            return 0

        self._pos += bytes_read
        return bytes_read

    def readable(self):
        return True

    def seekable(self):
        return False

    def writable(self):
        return False

    def close(self):
        return self.buffer.close()

    def fileno(self):
        return self.buffer.fileno()

    def isatty(self):
        return self.buffer.isatty()

    def tell(self):
        # hack, there is no way to know the current position
        return int(-1e6) if self._pos == 0 else self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        return self.tell()


class LDToolFileReader(BufferedInputStream):
    def __init__(self, ld_tool, file_path, input_argument=""):
        shell_command = [ld_tool]
        if input_argument:
            shell_command.append(input_argument)
        shell_command.append(file_path)
        p = subprocess.Popen(
            shell_command,
            shell=False,
            stdout=subprocess.PIPE,
            universal_newlines=False,
        )
        self.buffer = p.stdout
        self._pos: int = 0


class FlacFileReader(BufferedInputStream):
    def __init__(self, file_path):
        shell_command = [
            "flac",
            "-d",
            "-c",
            "-s",
            "-F",
            "--force-raw-format",
            "--endian",
            "little",
            "--sign",
            "signed",
            file_path,
        ]
        p = subprocess.Popen(
            shell_command,
            shell=False,
            stdout=subprocess.PIPE,
            universal_newlines=False,
        )
        self.buffer = p.stdout
        self._pos: int = 0


# executes ffmpeg and reads stdout as a file
class FFMpegFileReader(BufferedInputStream):
    def __init__(self, file_path):
        # Force ffmpeg to ignore duration metadata and read until actual EOF
        shell_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ignore_unknown",
            "-fflags", "nobuffer",
            "-i",
            file_path,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-avoid_negative_ts",
            "disabled",
            "-",
        ]
        p = subprocess.Popen(
            shell_command,
            shell=False,
            stdout=subprocess.PIPE,
            universal_newlines=False,
        )
        self.buffer = p.stdout
        self._pos: int = 0


class AsyncReader:
    def __init__(self, file, input_dtype, channels=1):
        if input_dtype is None:
            input_dtype = np.int16

        self._dtype_normalizer, self._numpy_in_dtype = get_normalizer(input_dtype)
        self._input_dtype_size = np.dtype(self._numpy_in_dtype).itemsize
        self._file = file
        self._channels = channels

    def __enter__(self):
        if self._file == "-":
            self._file_handle = sys.stdin.buffer
        elif isinstance(self._file, str):
            self._file_handle = open(self._file, "rb")
        else:
            self._file_handle = self._file

        if self._file == "-":
            self._frames = 0
        else:
            self._frames = os.fstat(self._file_handle.fileno()).st_size / np.dtype(self._numpy_in_dtype).itemsize / self._channels

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hifi_decode_reader"
        )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        try:
            if isinstance(self._file, str):
                self._file_handle.close()
        finally:
            self._executor.shutdown(wait=True)
        return False

    @property
    def frames(self):
        return self._frames

    @property
    def dtype(self):
        return REAL_DTYPE

    async def buffer_read_into(self, out):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._buffer_read_into,
            out
        )

    def buffer_read_into_sync(self, out):
        return self._buffer_read_into(out)
    
    def read(self):
        raise NotImplementedError

    def readall(self):
        raise NotImplementedError

    def _buffer_read_into(self, out):
        output_view = out.view(self._numpy_in_dtype)

        bytes_to_read = len(out) * self._input_dtype_size
        byte_view = memoryview(output_view).cast("B")

        total_bytes_read = 0

        while total_bytes_read < bytes_to_read:
            n = self._file_handle.readinto(
                byte_view[total_bytes_read:bytes_to_read]
            )

            if n == 0:  # EOF
                break

            total_bytes_read += n

        values_read = total_bytes_read // self._input_dtype_size

        if self._dtype_normalizer is not None:
            self._dtype_normalizer(output_view, out, values_read)

        return values_read


class AsyncSoundFileReader(sf.SoundFile):
    def __init__(self, file, input_dtype, **kwargs):
        super().__init__(
            file,
            "r",
            **kwargs,
        )
        if input_dtype is None:
            input_dtype = np.int16

        self._dtype_normalizer, self._numpy_in_dtype = get_normalizer(input_dtype)
        self._input_dtype_size = np.dtype(self._numpy_in_dtype).itemsize

    def __enter__(self):
        super().__enter__()

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hifi_decode_reader"
        )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            super().__exit__()
        finally:
            self._executor.shutdown(wait=True)
        return False
    
    @property
    def dtype(self):
        return REAL_DTYPE

    async def buffer_read_into(self, out):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._buffer_read_into,
            out,
        )

    def buffer_read_into_sync(self, out):
        return self._buffer_read_into(out)

    def _buffer_read_into(self, out):
        # reinterpret the float32 output buffer as the source dtype
        src = out.view(self._numpy_in_dtype)

        bytes_to_read = len(out) * self._input_dtype_size
        values_read = super().buffer_read_into(memoryview(src).cast("B")[:bytes_to_read], dtype="int16")

        if self._dtype_normalizer is not None:
            self._dtype_normalizer(src, out, values_read)

        return values_read


def as_soundfile(pathR, input_format_override: np.dtype = None):
    extension = pathR.lower().rsplit(".", 1)[-1]
    extension_with_endian = extension.replace("16", "16le").replace("32", "32le")

    input_format = FORMAT_TO_DTYPE.get(extension_with_endian)
    is_raw = input_format is not None

    # TODO add user facing parameter
    if input_format_override is not None:
        input_format = input_format_override

    if is_raw or pathR == "-":
        if pathR == "-" and input_format == None:
            raise Exception("`--raw_format <format>` is required for stdin input")

        return AsyncReader(
            pathR,
            input_format
        )
    elif "flac" == extension:
        try:
            return AsyncSoundFileReader(
                pathR,
                input_format
            )
        except sf.LibsndfileError as e:
            print(f"WARN: libsndfile could not open this FLAC file: {e}")
            if test_if_ffmpeg_is_installed():
                return AsyncReader(
                    FFMpegFileReader(pathR),
                    input_format
                )
            else:
                print(
                    "ERROR: ffmpeg is not installed (or not in PATH), cannot decode this FLAC bit depth."
                )
    elif "ldf" == extension:
        try:
            for ldf_reader_tool in ("ld-ldf-reader", "ld-ldf-reader-py"):
                if test_ld_tools(ldf_reader_tool):
                    return AsyncReader(
                        LDToolFileReader(ldf_reader_tool, pathR),
                        input_format
                    )
            print(
                "WARN: ld-ldf-reader/ld-ldf-reader-py not installed. LDF file format may not decode correctly"
            )
        except Exception as e:
            print(
                "WARN: Unexpected error opening LDF reader tool, LDF file format may not decode correctly", e
            )
        
        try:    
            if test_if_flac_is_installed():
                return AsyncReader(
                    FlacFileReader(pathR),
                    input_format
                )
            if test_if_ffmpeg_is_installed():
                return AsyncReader(
                    FFMpegFileReader(pathR),
                    input_format
                )
        except Exception as e:
            pass

        return AsyncSoundFileReader(
            pathR,
            input_format
        )
    elif "lds" == extension:
        try:
            for lds_reader_tool, input_arg in (
                ("ld-lds-reader", ""),
                ("ld-lds-converter", "-i"),
            ):
                if test_ld_tools(lds_reader_tool):
                    return AsyncReader(
                        LDToolFileReader(lds_reader_tool, pathR, input_arg),
                        input_format
                    )
            print(
                "ERROR: Unable to decode LDS without ld-lds-reader or ld-lds-converter. Please install one of them and try again."
            )
        except Exception as e:
            print(
                "ERROR: Unexpected error opening LDS reader tool", e
            )
    else:
        print("WARN: Unknown file format.")
        print("WARN: Attempting to decode with ffmpeg")
        try:
            if test_if_ffmpeg_is_installed():
                return AsyncReader(
                    FFMpegFileReader(pathR),
                    input_format
                )
        except Exception:
            pass
        print("WARN: Attempting to decode with SoundFile")
        return AsyncSoundFileReader(
            pathR,
            input_format
        )


def get_normalize_filename(path, sample_rate):
    return f"{path}_{int(sample_rate)}_f32_{NORMALIZE_FILE_SUFFIX}"

def get_dual_mono_filename(path, channel_suffix):
    root, extension = os.path.splitext(path)
    return f"{root}_{channel_suffix}{extension}"

normalize_parameters = {
    "format": "RAW",  # TODO: update to FLAC 32 bit when supported by soundfile
    "endian": "little",
    "subtype": "FLOAT",
}


def as_outputfile(path, channels, sample_rate, normalize):
    if normalize:
        return sf.SoundFile(
            get_normalize_filename(path, sample_rate),
            "w",
            samplerate=int(sample_rate),
            channels=channels,
            **normalize_parameters,
        )
    elif ".wav" in path.lower():
        return sf.SoundFile(
            path,
            "w",
            channels=channels,
            samplerate=int(sample_rate),
            format="WAV",
            subtype="PCM_16",
        )
    else:
        return sf.SoundFile(
            path,
            "w",
            channels=channels,
            samplerate=int(sample_rate),
            format="FLAC",
            subtype="PCM_24",
            compression_level=1.0,
        )


def seconds_to_str(seconds: float) -> str:
    delta = timedelta(seconds=seconds)

    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:01}:{:02}:{:06.3f}".format(int(hours), int(minutes), seconds)


def log_decode(
    start_time: datetime,
    frames: int,
    audio_samples: int,
    blocks_enqueued: int,
    input_rate: int,
    audio_rate: int,
):
    elapsed_time: timedelta = datetime.now() - start_time

    input_time: float = frames / input_rate
    input_time_format: str = seconds_to_str(input_time)

    audio_time: float = audio_samples / audio_rate
    audio_time_format: str = seconds_to_str(audio_time)

    audio_buffer_time = max(0, input_time - audio_time)
    audio_buffer_time_format: str = seconds_to_str(audio_buffer_time)

    relative_speed: float = input_time / elapsed_time.total_seconds()
    elapsed_time_format: str = seconds_to_str(elapsed_time.total_seconds())

    print(
        f"- Decoding speed: {round(frames / (1e3 * elapsed_time.total_seconds()))} kFrames/s ({relative_speed:.2f}x), {int(blocks_enqueued)} blocks enqueued\n"
        + f"- Input position: {input_time_format}\n"
        + f"- Audio position: {audio_time_format}\n"
        + f"- Audio buffer  : {audio_buffer_time_format}\n"
        + f"- Wall time     : {elapsed_time_format}"
    )


class AppWindow:
    def __init__(self, argv, decode_options, app=None):
        self._owns_app = app is None and QApplication.instance() is None
        self._app, self._window = self.open_ui(argv, decode_options, app)

    def __del__(self):
        # self._window.close()
        if getattr(self, "_owns_app", False):
            self._app.quit()

    def open_ui(self, argv, decode_options, app=None):
        app = app if app is not None else QApplication.instance()
        if app is None:
            app = QApplication(argv)
        print("Opening hifi-ui...")
        if decode_options["input_file"] == "-":
            window = FileOutputDialogUI(decode_options_to_ui_parameters(decode_options))
        else:
            window = FileIODialogUI(decode_options_to_ui_parameters(decode_options))
        window.show()
        return app, window

    def run(self):
        self._app.exec_()

    @property
    def window(self):
        return self._window

    @property
    def app(self):
        return self._app


class HostedHifiController:
    def __init__(self, argv, args, decode_options, app=None):
        try:
            from PyQt6.QtCore import QTimer
        except ImportError:
            from PyQt5.QtCore import QTimer

        self._args = args
        self._decoder_state = 0
        self._decode_active = False
        self._ui = AppWindow(argv, decode_options, app=app)
        self._poll_timer = QTimer(self.window)
        self._poll_timer.setInterval(10)
        self._poll_timer.timeout.connect(self._poll_transport)
        self._poll_timer.start()

    @property
    def window(self):
        return self._ui.window

    @property
    def app(self):
        return self._ui.app

    @property
    def decoder_state(self):
        return self._decoder_state

    def _poll_transport(self):
        if not self.window.isVisible():
            self._poll_timer.stop()
            return

        if self._decode_active:
            return

        if self.window.transport_state not in (PLAY_STATE, PREVIEW_STATE):
            return

        self._decode_active = True
        try:
            self._decoder_state = _run_ui_transport_action(self._args, self._ui)
        except (KeyboardInterrupt, RuntimeError):
            pass
        finally:
            self._decode_active = False


class SoundDeviceProcess:
    def __init__(self, sample_rate, stop_requested):
        self._sample_rate = sample_rate
        self._stop_requested = stop_requested

    def __enter__(self):
        self._play_parent_conn, self._play_child_conn = Pipe()
        self._process = Process(
            target=SoundDeviceProcess.play_worker,
            name="hifi_playback_worker",
            args=(self._play_child_conn, self._sample_rate, self._stop_requested),
        )
        self._process.start()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._process.terminate()
        self._process.join()
        return self

    @staticmethod
    @njit(
        numba.types.void(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.int16, 2, "C"),
        ),
        cache=True,
        fastmath=True,
        nogil=True,
    )
    def build_stereo(interleaved: np.array, stacked) -> np.array:
        for i in range(0, len(stacked)):
            stacked[i][0] = interleaved[i * 2] * 2**15
            stacked[i][1] = interleaved[i * 2 + 1] * 2**15

    @staticmethod
    def play_worker(conn, sample_rate, stop_requested):
        setproctitle(current_process().name)
        if not _sounddevice_available():
            return
        output_stream = None
        while True:
            while True:
                try:
                    with stop_requested.get_lock():
                        if stop_requested.value == STOP_IMMEDIATE_REQUESTED:
                            return

                    if conn.poll(1):
                        stereo = conn.recv_bytes()
                        break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            if output_stream == None:
                output_stream = sd.OutputStream(
                    samplerate=sample_rate, channels=2, dtype="int16"
                )
                output_stream.start()

            interleaved_len = int(len(stereo) / np.dtype(np.float32).itemsize)
            interleaved = np.ndarray(
                interleaved_len, dtype=np.float32, buffer=stereo, order="C"
            )
            stacked = np.empty((int(len(interleaved) / 2), 2), dtype=np.int16, order="C")

            SoundDeviceProcess.build_stereo(interleaved, stacked)
            output_stream.write(stacked)

    def play(self, stereo):
        self._play_parent_conn.send_bytes(stereo)


def write_soundfile_process_worker(
    post_processor_out_rx_conn,
    blocks_enqueued,
    post_processor_shared_memory_idle_queue,
    start_time,
    input_position,
    total_samples_decoded,
    decode_options,
    output_file: str,
    channel_1_suffix: str,
    channel_2_suffix: str,
    stop_requested,
    decode_done,
):
    setproctitle(current_process().name)
    audio_rate = decode_options["audio_rate"]
    input_rate = decode_options["input_rate"]
    preview_mode = decode_options["preview"]
    preview_playback_enabled = preview_mode and _sounddevice_available()
    normalize = decode_options["normalize"]
    dual_mono = decode_options["mode"] == AUDIO_MODE_DUAL_MONO or decode_options["mode"] == AUDIO_MODE_DUAL_MONO_MS
    if preview_playback_enabled:
        player = SoundDeviceProcess(audio_rate, stop_requested)
    else:
        if preview_mode:
            print("Import of sounddevice failed, preview is not available!")
        player = nullcontext()

    if dual_mono:
        stereo_output = nullcontext()
        channel_1_output = as_outputfile(
            get_dual_mono_filename(output_file, channel_1_suffix),
            1,
            audio_rate,
            normalize
        )

        channel_2_output = as_outputfile(
            get_dual_mono_filename(output_file, channel_2_suffix),
            1,
            audio_rate,
            normalize
        )
    else:
        stereo_output = as_outputfile(output_file, 2, audio_rate, normalize)
        channel_1_output = nullcontext()
        channel_2_output = nullcontext()

    with player, stereo_output, channel_1_output, channel_2_output:
        done = False
        while not done:
            while True:
                try:
                    decoder_state = post_processor_out_rx_conn.recv()
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            buffer = PostProcessorSharedMemory(decoder_state)
            stereo = buffer.get_stereo()
            samples_decoded = len(stereo) / 2
            
            # pad the start of the audio due to beginning gap
            if decoder_state.block_num == 0:
                padding = round(decoder_state.block_audio_final_overlap / 2) * 2 * 4 # 2 channels, 4 bytes per channel
                
                if dual_mono:
                    channel_1_output.buffer_write(bytes(padding), dtype="float32")
                    channel_2_output.buffer_write(bytes(padding), dtype="float32")
                else:
                    stereo_output.buffer_write(bytes(padding), dtype="float32")

            if dual_mono:
                channel_1, channel_2 = stereo[::2], stereo[1::2]
                channel_1_output.write(channel_1)
                channel_2_output.write(channel_2)
            else:
                stereo_output.buffer_write(stereo, dtype="float32")

            if preview_playback_enabled:
                stereo_copy = np.empty_like(stereo, order="C")
                DecoderSharedMemory.copy_data_float32(
                    stereo, stereo_copy, len(stereo_copy)
                )
                player.play(stereo_copy)

            buffer.close()
            post_processor_shared_memory_idle_queue.put((decoder_state.name, decoder_state.numa_node))
            with total_samples_decoded.get_lock():
                total_samples_decoded.value = int(
                    total_samples_decoded.value + samples_decoded
                )

            log_decode(
                start_time,
                input_position.value,
                total_samples_decoded.value,
                blocks_enqueued.value,
                input_rate,
                audio_rate,
            )

            done = decoder_state.is_last_block

        if dual_mono:
            channel_1_output.flush()
            channel_2_output.flush()
        else:
            stereo_output.flush()
        decode_done.set()


async def decode_parallel(
    decode_options: dict,
    threads: int = 8,
    ui_t: Optional[AppWindow] = None,
):
    decoder = HiFiDecode(options=decode_options, is_main_process=True, bias_guess=decode_options["bias_guess"])
    # TODO: reprocess data read in this step
    if decode_options["bias_guess"]:
        guess_bias(decoder, decode_options["input_file"], int(decode_options["input_rate"]))

    input_file = decode_options["input_file"]
    input_format_override = decode_options.get("input_format_override")
    output_file = decode_options["output_file"]

    channel_1_suffix = DEFAULT_CHANNEL_SUFFIX + "_1"
    channel_2_suffix = DEFAULT_CHANNEL_SUFFIX + "_2"

    block_size = decoder.initialBlockSize

    blocks_enqueued = Value("d", 0)
    input_position = Value("d", 0)
    total_samples_decoded = Value("d", 0)
    peak_gain = Value(PeakGain, 0.0, 0.0)
    stop_requested = Value("d", STOP_NOT_REQUESTED)
    start_time = datetime.now()

    # HiFiDecode data flow diagram
    # All data is sent via SharedMemory
    # Each step below are separate processes connected together with queues
    # Each message on the queue contains the name of the SharedMemory buffer and any state that needs to persist as the data flow through
    #
    #                            [decoder_process_worker]
    #                                |->-decoder_1->-|
    #                                |->-decoder_2->-|                                                                                |-> [SoundDeviceProcess] -> audio out
    # data in ->- decoder_in_queue->-|->-decoder_3->-|->-decoder_out_queue-->--[post_processor]-->--[write_soundfile_process_worker]--|
    #                                |->-decoder_4->-|                           |                                                    |-> [SoundFileEncoder] ---> data out
    #                                |->-decoder_n->-|                           (see PostProcessor for details)

    # spin up shared memory
    # these blocks of memory are used to transfer the audio data throughout the various steps
    num_shared_memory_instances = int(threads + 2)
    num_decoders = threads

    shared_memory_instances = []
    shared_memory_idle_queue = SimpleQueue()
    max_numa_node = 0
    for i in range(num_shared_memory_instances):
        numa_node = NUMA.next_node()

        NUMA.bind_process(numa_node)
        if numa_node > max_numa_node:
            max_numa_node = numa_node

        buffer_instance = DecoderSharedMemory.get_shared_memory(
            decoder.initialBlockSize,
            decoder.blockOverlap,
            decoder.initialBlockFinalAudioSize,
            f"hifi_decoder_{i}",
            numa_node=numa_node
        )
        shared_memory_instances.append(buffer_instance)
        shared_memory_idle_queue.put((buffer_instance.name, numa_node))

        atexit.register(buffer_instance.close)
        atexit.register(buffer_instance.unlink)

    # spin up the decoders
    decoder_processes: list[Process] = []
    decoder_in_queues = [SimpleQueue() for _ in range(max_numa_node + 1)]
    decoder_out_queue = SimpleQueue()
    decode_done = Event()

    for i in range(num_decoders):
        numa_node = i % (max_numa_node + 1)

        NUMA.bind_process(numa_node)
        decoder_process = Process(
            target=HiFiDecode.hifi_decode_worker,
            name=f"hifi_decode_worker_{i}",
            args=(
                decoder_in_queues[numa_node],
                decoder_out_queue,
                decode_options,
                decoder.standard,
                numa_node
            ),
        )
        decoder_process.start()

        atexit.register(decoder_process.terminate)
        decoder_processes.append(decoder_process)

    # set up the post processor
    # all new processes should be ont he same numa node at this point
    post_processor_numa_node = 0
    NUMA.bind_process(post_processor_numa_node)
    post_processor_out_rx_conn, post_processor_out_tx_conn = Pipe(duplex=False)
    post_processor_shared_memory_idle_queue = SimpleQueue()
    post_processor = PostProcessor(
        decode_options,
        decoder_out_queue,
        decoder.initialBlockFinalAudioSize + decoder.blockAudioFinalOverlap * 2,
        post_processor_shared_memory_idle_queue,
        shared_memory_idle_queue,
        blocks_enqueued,
        post_processor_out_tx_conn,
        peak_gain,
        post_processor_numa_node
    )
    atexit.register(post_processor.close)

    # set up the output file process
    output_file_process = Process(
        target=write_soundfile_process_worker,
        name="hifi_output_encoder",
        args=(
            post_processor_out_rx_conn,
            blocks_enqueued,
            post_processor_shared_memory_idle_queue,
            start_time,
            input_position,
            total_samples_decoded,
            decode_options,
            output_file,
            channel_1_suffix,
            channel_2_suffix,
            stop_requested,
            decode_done,
        ),
    )
    output_file_process.start()
    atexit.register(output_file_process.terminate)
    atexit.register(output_file_process.join)

    async def read_and_send_to_decoder(
        f,
        decoder,
        decoder_state,
        input_position,
        exit_requested,
        previous_overlap,
        previous_block
    ):
        buffer = DecoderSharedMemory(decoder_state)
        # read input data into the shared memory buffer
        block_in = buffer.get_block_in()
        try:
            frames_read = await f.buffer_read_into(block_in)
        except sf.LibsndfileError as e:
            print("Error decoding input rf:", e)
            print("Stopping decode...")
            frames_read = 0
        
        decoder_state.block_frames_read = frames_read
        decoder_state.is_last_block = frames_read < len(block_in) or exit_requested or stop_requested.value

        with input_position.get_lock():
            input_position.value += frames_read

        if block_num == 0:
            # save the read data
            block_data_read = block_in.copy()
            buffer.close()

            # shift the actual data down by half so only the copied data is discarded
            start_overlap_end = (
                decoder_state.block_read_overlap - decoder_state.block_overlap
            )
            new_block_length = frames_read + start_overlap_end

            # create a new buffer with the updated offsets, and copy in the read data
            decoder_state = DecoderState(
                decoder,
                buffer.name,
                decoder_state.block_frames_read,
                new_block_length,
                decoder_state.block_num,
                decoder_state.is_last_block,
                decoder_state.numa_node
            )
            buffer = DecoderSharedMemory(decoder_state)
            block = buffer.get_block()
            # copy starting at half the normal read overlap
            DecoderSharedMemory.copy_data_dst_offset_float32(
                block_data_read, block, start_overlap_end, len(block_data_read)
            )

            # this is the first block, fill in the empty data before half the read overlap, this will be discarded
            DecoderSharedMemory.copy_data_float32(
                block_data_read, block, start_overlap_end
            )
        elif decoder_state.is_last_block and frames_read > 0:
            # shift the read in data to (end - discard overlap)
            block = buffer.get_block()

            frames_read_with_overlap = frames_read + decoder_state.block_overlap
            block_in_offset = len(block) - frames_read_with_overlap
            block_data_read = block_in[0:frames_read].copy()
            DecoderSharedMemory.copy_data_dst_offset_float32(
                block_data_read, block, block_in_offset, frames_read
            )

            # copy in the entire previous block to use as overlap
            # at the end of this decode worker, only the new audio will be returned
            previous_block_in_offset = len(previous_block) - block_in_offset
            DecoderSharedMemory.copy_data_src_offset_float32(
                previous_block, block, previous_block_in_offset, block_in_offset
            )
        else:
            # copy the overlapping data from the previous read
            block_in_overlap = buffer.get_block_in_start_overlap()
            DecoderSharedMemory.copy_data_float32(
                previous_overlap, block_in_overlap, len(block_in_overlap)
            )

        # copy the the current overlap to use in the next iteration
        current_overlap = buffer.get_block_in_end_overlap()
        DecoderSharedMemory.copy_data_float32(
            current_overlap, previous_overlap, len(current_overlap)
        )

        if not decoder_state.is_last_block:
            # save the full block for the next iteration, including previous overlap
            # will be used if the next block is the last block
            block = buffer.get_block()
            DecoderSharedMemory.copy_data_float32(
                block, previous_block, len(previous_block)
            )

        buffer.close()

        # submit the block to the decoder
        # blocks should complete roughly in the order that they are submitted
        numa_node = decoder_state.numa_node
        decoder_in_queues[numa_node].put(decoder_state)

        return decoder_state.is_last_block

    print(f"Starting decode...")

    async def ui_task(stop_requested, ui_t):
        while True:
            previous_state = ui_t.window.transport_state
            ui_t.app.processEvents()

            if ui_t.window.transport_state == STOP_STATE:
                with stop_requested.get_lock():
                    stop_requested.value = STOP_REQUESTED

                    if previous_state == PREVIEW_STATE:
                        stop_requested.value = STOP_IMMEDIATE_REQUESTED

                break
            elif ui_t.window.transport_state == PAUSE_STATE:
                while ui_t.window.transport_state == PAUSE_STATE:
                    ui_t.app.processEvents()
                    await asyncio.sleep(0.01)

            await asyncio.sleep(0.01)

    if ui_t is not None:
        asyncio.create_task(ui_task(stop_requested, ui_t))

    with as_soundfile(input_file, input_format_override) as f:
        loop = asyncio.get_event_loop()
        previous_block = np.empty(block_size, dtype=REAL_DTYPE)
        progressB = TimeProgressBar(f.frames, f.frames)
        block_num = 0

        while True:
            # get the next available shared memory buffer
            while True:
                try:
                    buffer_name, numa_node = await loop.run_in_executor(
                        None, shared_memory_idle_queue.get
                    )
                    break
                except InterruptedError:
                    pass
                except EOFError:
                    return

            decoder_state = DecoderState(
                decoder, buffer_name, 0, block_size, block_num, False, numa_node
            )

            if decoder_state.block_num == 0:
                previous_overlap = np.empty(
                    decoder_state.block_read_overlap, dtype=REAL_DTYPE, order="C"
                )

            is_last_block = await read_and_send_to_decoder(
                f,
                decoder,
                decoder_state,
                input_position,
                exit_requested,
                previous_overlap,
                previous_block
            )
            
            with blocks_enqueued.get_lock():
                progressB.print(input_position.value)
                blocks_enqueued.value += 1

            log_decode(
                start_time,
                input_position.value,
                total_samples_decoded.value,
                blocks_enqueued.value,
                decode_options["input_rate"],
                decode_options["audio_rate"],
            )

            if is_last_block:
                break

            block_num += 1

    print("")
    print("Decode finishing up. Emptying the queue")
    print("")
    
    with stop_requested.get_lock():
        if stop_requested.value != STOP_IMMEDIATE_REQUESTED:
            decode_done.wait()

    for p in decoder_processes:
        cleanup_process(p)
    post_processor.close()
    cleanup_process(output_file_process)

    for shared_memory in shared_memory_instances:
        shared_memory.close()
        shared_memory.unlink()
        atexit.unregister(shared_memory.close)
        atexit.unregister(shared_memory.unlink)
    
    with peak_gain.get_lock():
        audio_rate = decode_options["audio_rate"]

        if (
            decode_options["mode"] == AUDIO_MODE_DUAL_MONO or
            decode_options["mode"] == AUDIO_MODE_DUAL_MONO_MS
        ):
            # Normalize channels independently for dual mono
            print(f"\nChannel 1: Peak gain is {(peak_gain.left * 100):.2f}%.", end="")
            if decode_options["normalize"]:
                channel_1_output_file = get_dual_mono_filename(output_file, channel_1_suffix)
                channel_1_input_file_post_gain = get_normalize_filename(channel_1_output_file, audio_rate)
                await normalize(channel_1_input_file_post_gain, channel_1_output_file, peak_gain.left, 1, audio_rate)

            print(f"\nChannel 2: Peak gain is {(peak_gain.right * 100):.2f}%.", end="")
            if decode_options["normalize"]:
                channel_2_output_file = get_dual_mono_filename(output_file, channel_2_suffix)
                channel_2_input_file_post_gain = get_normalize_filename(channel_2_output_file, audio_rate)
                await normalize(channel_2_input_file_post_gain, channel_2_output_file, peak_gain.right, 1, audio_rate)
        else:
            peak_gain_stereo = max(peak_gain.left, peak_gain.right)
            print(f"\nPeak gain is {(peak_gain_stereo * 100):.2f}%.", end="")
            if decode_options["normalize"]:
                input_file_post_gain = get_normalize_filename(output_file, audio_rate)
                await normalize(input_file_post_gain, output_file, peak_gain_stereo, 2, audio_rate)

    elapsed_time = datetime.now() - start_time
    dt_string = elapsed_time.total_seconds()
    print(f"\nDecode finished, seconds elapsed: {round(dt_string)}")

async def normalize(input_file_post_gain, output_file, peak_gain, channels, audio_rate):
    try:
        total_frames_read = 0
        buffer = np.empty(2**20, dtype=np.float32, order="C")

        with AsyncReader(
            input_file_post_gain,
            np.float32,
            channels
        ) as f, as_outputfile(output_file, channels, audio_rate, False) as w:
            gain_adjust = (
                1 / peak_gain - np.finfo(np.float16).eps
            )  # subtract epsilon error to prevent appearance of "clipping" in editing tools
            print(f" Adjusting to {(gain_adjust * 100):.2f}%, please wait...")

            progressB = TimeProgressBar(f.frames, f.frames)
            done = False
            while not done:
                frames_read = await f.buffer_read_into(buffer)

                if frames_read < len(buffer):
                    buffer = buffer[0:frames_read]
                    done = True

                PostProcessor.normalize(gain_adjust, buffer, buffer)
                w.buffer_write(buffer, "float32")

                total_frames_read += frames_read
                progressB.print(total_frames_read / channels, False)
        print("")
    finally:
        os.remove(input_file_post_gain)

def guess_bias(decoder, input_file, block_size, blocks_limits=10):
    print("Measuring carrier bias ... ")
    blocks = list()

    with as_soundfile(input_file) as f:
        while f.tell() < f.frames and len(blocks) <= blocks_limits:
            block_buffer = np.empty(block_size, dtype=f.dtype, order="C")
            f.buffer_read_into_sync(block_buffer)
            blocks.append(block_buffer)

    decoder.guessBiases(blocks)
    print("\ndone!")


def run_decoder(args, decode_options, ui_t: Optional[AppWindow] = None):
    sample_freq = decode_options["input_rate"]

    if sample_freq is not None:
        with ThreadPoolExecutor(4) as async_executor:
            # work around for Windows not creating an event loop for the main thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError as e:
                if str(e).startswith('There is no current event loop in thread'):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                else:
                    raise

            loop.set_default_executor(async_executor)
            loop.run_until_complete(
                decode_parallel(decode_options, threads=args.threads, ui_t=ui_t)
            )

        print("Decode finished successfully")
        return 0
    else:
        print("No sample rate specified")
        return 0


def build_decode_options_from_args(args):
    system = "PAL" if args.pal else "NTSC"
    sample_freq = args.inputfreq
    input_format_override = args.raw_format

    filename = args.infile
    outname = args.outfile

    # 8mm AFM uses a mono channel, or L-R/L+R rather than L/R channels
    # The spec defines a dual audio mode but not sure if it was ever used.
    default_mode = (
        DEFAULT_VHS_AUDIO_MODE if not args.format_8mm else DEFAULT_8MM_AUDIO_MODE
    )

    real_mode = default_mode if not args.mode else args.mode

    if (
        args.resampler_quality == "low"
        or args.resampler_quality == "medium"
        or args.resampler_quality == "high"
    ):
        resampler_quality = args.resampler_quality
    else:
        resampler_quality = DEFAULT_RESAMPLER_QUALITY

    default_expander_env_detection = DEFAULT_ENV_DETECTION

    if args.format_8mm:
        tape_format = "8mm"
        default_expander_gain = DEFAULT_8MM_EXPANDER_GAIN
        default_expander_ratio = DEFAULT_8MM_EXPANDER_RATIO
        default_expander_attack_tau = DEFAULT_8MM_EXPANDER_ATTACK_TAU
        default_expander_hold_tau = DEFAULT_8MM_EXPANDER_HOLD_TAU
        default_expander_release_tau = DEFAULT_8MM_EXPANDER_RELEASE_TAU

        default_deemphasis_low_tau = DEFAULT_8MM_DEEMPHASIS_TAU_1
        default_deemphasis_high_tau = DEFAULT_8MM_DEEMPHASIS_TAU_2

        default_nr_deemphasis_low_tau = DEFAULT_8MM_NR_DEEMPHASIS_TAU_1
        default_nr_deemphasis_high_tau = DEFAULT_8MM_NR_DEEMPHASIS_TAU_2

        default_expander_weighting_low_tau = DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_1
        default_expander_weighting_high_tau = DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_2
        default_expander_weighting_low_pass = DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS
        default_expander_weighting_low_pass_transition = (
            DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION
        )
    else:
        tape_format = "vhs"
        default_expander_gain = DEFAULT_VHS_EXPANDER_GAIN
        default_expander_ratio = DEFAULT_VHS_EXPANDER_RATIO
        default_expander_attack_tau = DEFAULT_VHS_EXPANDER_ATTACK_TAU
        default_expander_hold_tau = DEFAULT_VHS_EXPANDER_HOLD_TAU
        default_expander_release_tau = DEFAULT_VHS_EXPANDER_RELEASE_TAU

        default_deemphasis_low_tau = DEFAULT_VHS_DEEMPHASIS_TAU_1
        default_deemphasis_high_tau = DEFAULT_VHS_DEEMPHASIS_TAU_2

        default_nr_deemphasis_low_tau = DEFAULT_VHS_NR_DEEMPHASIS_TAU_1
        default_nr_deemphasis_high_tau = DEFAULT_VHS_NR_DEEMPHASIS_TAU_2

        default_expander_weighting_low_tau = DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1
        default_expander_weighting_high_tau = DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2
        default_expander_weighting_low_pass = DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS
        default_expander_weighting_low_pass_transition = (
            DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION
        )

    decode_options = {
        "input_rate": sample_freq * 1e6,
        "input_format_override": input_format_override,
        "standard": "p" if system == "PAL" else "n",
        "format": tape_format,
        "preview": args.preview,
        "preview_available": _sounddevice_available() if args.preview else False,
        "demod_type": args.demod_type,
        "afe_left_carrier_deviation": args.afe_left_carrier_deviation * 10e5,
        "afe_right_carrier_deviation": args.afe_right_carrier_deviation * 10e5,
        "afe_left_carrier": args.afe_left_carrier * 10e5,
        "afe_right_carrier": args.afe_right_carrier * 10e5,
        "resampler_quality": resampler_quality if not args.preview else "low",
        "spectral_nr_amount": args.spectral_nr_amount if not args.preview else 0,
        "head_switching_interpolation": args.head_switching_interpolation == "on",
        "doc": args.doc,
        "enable_expander": args.enable_expander == "on",
        "enable_deemphasis": args.enable_deemphasis == "on",
        "auto_fine_tune": args.auto_fine_tune == "on" if not args.preview else False,
        "bias_guess": args.bias_guess,
        "normalize": args.normalize,
        "expander_gain": args.expander_gain or default_expander_gain,
        "expander_ratio": args.expander_ratio or default_expander_ratio,
        "expander_env_detection": args.expander_env_detection or default_expander_env_detection,
        "expander_attack_tau": args.expander_attack_tau or default_expander_attack_tau,
        "expander_hold_tau": args.expander_hold_tau or default_expander_hold_tau,
        "expander_release_tau": args.expander_release_tau or default_expander_release_tau,
        "expander_weighting_low_tau": args.expander_weighting_low_tau or default_expander_weighting_low_tau,
        "expander_weighting_high_tau": args.expander_weighting_high_tau or default_expander_weighting_high_tau,
        "expander_weighting_low_pass": args.expander_weighting_low_pass or default_expander_weighting_low_pass,
        "expander_weighting_low_pass_transition": args.expander_weighting_low_pass_transition or default_expander_weighting_low_pass_transition,
        "nr_deemphasis_low_tau": args.nr_deemphasis_low_tau or default_nr_deemphasis_low_tau,
        "nr_deemphasis_high_tau": args.nr_deemphasis_high_tau or default_nr_deemphasis_high_tau,
        "deemphasis_low_tau": args.deemphasis_low_tau or default_deemphasis_low_tau,
        "deemphasis_high_tau": args.deemphasis_high_tau or default_deemphasis_high_tau,
        "grc": args.GRC,
        "audio_rate": args.rate if not args.preview else 44100,
        "gain": args.gain,
        "input_file": filename,
        "output_file": outname,
        "mode": real_mode,
        "threads": args.threads,
    }

    return decode_options, real_mode


def _run_ui_transport_action(args, ui_t):
    print("Starting decode...")
    options = ui_parameters_to_decode_options(ui_t.window.getValues())
    previous_state = ui_t.window.transport_state
    options["preview"] = previous_state == PREVIEW_STATE
    # apply the thread count chosen in the UI (run_decoder reads args.threads)
    args.threads = max(1, int(options.get("threads", args.threads)))

    working_directory = os.getcwd()
    try:
        output_directory = os.path.dirname(options["output_file"])
        if output_directory != "":
            os.chdir(output_directory)

        if test_input_file(options["input_file"]) and test_output_file(
            options["output_file"]
        ):
            decoder_state = run_decoder(args, options, ui_t=ui_t)
            ui_t.window.transport_state = STOP_STATE

            if previous_state == PLAY_STATE:
                ui_t.window.on_decode_finished()
        else:
            message = None
            if not test_input_file(options["input_file"]):
                message = f"Input file '{options['input_file']}' not found"
            elif not test_output_file(options["output_file"]):
                message = (
                    f"Output file '{options['output_file']}' cannot be created "
                    "nor overwritten"
                )

            ui_t.window.generic_message_box(
                "I/O Error", message, QMessageBox.Icon.Critical
            )
            ui_t.window.on_stop_clicked()
            decoder_state = 1
    finally:
        os.chdir(working_directory)

    return decoder_state


def launch_hosted_ui(argv=None, app=None):
    launch_args = list(argv or [])
    if "--gui" not in launch_args:
        launch_args.insert(0, "--gui")

    args = parser.parse_args(launch_args)
    decode_options, _ = build_decode_options_from_args(args)
    qt_app = app if app is not None else QApplication.instance()
    if qt_app is None:
        qt_app = QApplication([sys.argv[0]] + launch_args)
    return HostedHifiController(
        [sys.argv[0]] + launch_args, args, decode_options, qt_app
    )


def main(argv=None) -> int:
    args = parser.parse_args(argv)
    decode_options, real_mode = build_decode_options_from_args(args)
    system = "PAL" if decode_options["standard"] == "p" else "NTSC"

    if not hasattr(args, "UI"):
        if not test_input_file(args.infile):
            raise FileNotFoundError("Input file error")
        if not test_output_file(args.outfile):
            raise FileNotFoundError("Output file error")
    filename = decode_options["input_file"]
    outname = decode_options["output_file"]

    if not args.UI and not args.overwrite:
        if os.path.isfile(outname):
            print(
                "Existing decode files found, remove them or run command with --overwrite"
            )
            print("\t", outname)
            return 1

    print("Initializing ...")


    if args.UI and not HIFI_UI:
        print(
            (
                "PyQt5/PyQt6 is not installed, can not use graphical UI,"
                " falling back to command line interface.."
            )
        )

    if args.UI and HIFI_UI:
        ui_t = AppWindow(sys.argv, decode_options)
        decoder_state = 0
        try:
            while ui_t.window.isVisible():
                if (
                    ui_t.window.transport_state == PLAY_STATE
                    or ui_t.window.transport_state == PREVIEW_STATE
                ):
                    decoder_state = _run_ui_transport_action(args, ui_t)

                ui_t.app.processEvents()
                time.sleep(0.01)
            ui_t.window.transport_state = STOP_STATE
        except (KeyboardInterrupt, RuntimeError):
            pass

        return decoder_state
    else:
        if test_input_file(filename) and test_output_file(outname):
            if decode_options["format"] == "vhs":
                (
                    print(f"PAL VHS format selected, Audio mode is {real_mode}")
                    if system == "PAL"
                    else print(f"NTSC VHS format selected, Audio mode is {real_mode}")
                )
            else:
                if system == "PAL":
                    print(f"PAL 8mm format selected, Audio mode is {real_mode}")
                else:
                    print(f"NTSC 8mm format selected, Audio mode is {real_mode}")

            return run_decoder(args, decode_options)
        else:
            parser.print_help()
            print(
                "ERROR: input file not found"
                if not test_input_file(filename)
                else f"ERROR: output file '{outname}' cannot be created nor overwritten"
            )
            return 1


signal_count = 0
exit_requested = False
main_pid = os.getpid()
NUM_SIGINT_BEFORE_FORCE_EXIT = 1


def parent_signal_handler(sig, frame):
    global signal_count
    global main_pid
    global exit_requested

    exit_requested = True
    is_main_thread = main_pid == os.getpid()
    if signal_count >= NUM_SIGINT_BEFORE_FORCE_EXIT:
        if is_main_thread:
            # prevent reentrant calls https://stackoverflow.com/a/75368797
            os.write(
                sys.stdout.fileno(),
                b"\nCtrl-C was pressed again, stopping immediately...\n",
            )
        sys.exit(1)
    if signal_count == 0:
        if is_main_thread:
            os.write(sys.stdout.fileno(), b"\nCtrl-C was pressed, stopping decode...\n")
    signal_count += 1


def child_signal_handler(sig, frame):
    global signal_count
    global exit_requested

    exit_requested = True
    if signal_count >= NUM_SIGINT_BEFORE_FORCE_EXIT:
        sys.exit(1)
    signal_count += 1

def _register_signal_handlers_if_supported():
    if threading.current_thread() is not threading.main_thread():
        return

    if current_process().name == "MainProcess":
        signal.signal(signal.SIGINT, parent_signal_handler)
    else:
        signal.signal(signal.SIGINT, child_signal_handler)


_register_signal_handlers_if_supported()

if __name__ == "__main__":
    freeze_support()
    sys.exit(main())
