#!/usr/bin/env python3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from datetime import datetime, timedelta
import os
import sys
from typing import List, Optional

import numpy as np
import soundfile as sf
from vhsdecode.addons.chromasep import samplerate_resample

from vhsdecode.cmdcommons import (
    common_parser_cli,
    select_sample_freq,
    select_system,
    get_basics,
    test_input_file,
    test_output_file,
)
from vhsdecode.hifi.HiFiDecode import (
    HiFiDecode,
    NoiseReduction,
    DEFAULT_NR_GAIN_,
    discard_stereo,
)
from vhsdecode.hifi.TimeProgressBar import TimeProgressBar
import io

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

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
except ImportError:
    HIFI_UI = False


parser, _ = common_parser_cli(
    "Extracts audio from raw VHS HiFi FM capture",
    default_threads=round(cpu_count() / 2),
)

parser.add_argument(
    "--audio_rate",
    dest="rate",
    type=int,
    default=48000,
    help="Output sample rate in Hz (default 48000)",
)

parser.add_argument(
    "--bg", "--bias_guess",
    dest="BG", 
    action="store_true", 
    default=False, 
    help="Do carrier bias guess"
)

parser.add_argument(
    "--preview",
    dest="preview",
    action="store_true",
    default=False,
    help="Use preview quality (faster and noisier)",
)

parser.add_argument(
    "--original",
    dest="original",
    action="store_true",
    default=False,
    help="Use the same FM demod as vhs-decode",
)

parser.add_argument(
    "--noise_reduction",
    dest="noise_reduction",
    type=str.lower,
    default="on",
    help="Set noise reduction on/off",
)

parser.add_argument(
    "--auto_fine_tune",
    dest="auto_fine_tune",
    type=str.lower,
    default="on",
    help="Set auto tuning of the analog front end on/off",
)

parser.add_argument(
    "--NR_sidechain_gain",
    dest="NR_side_gain",
    type=float,
    default=DEFAULT_NR_GAIN_,
    help=f"Sets the noise reduction envelope tracking sidechain gain (default is {DEFAULT_NR_GAIN_}). "
    f"Range (20~100): 100 being a hard gate effect",
)

parser.add_argument(
    "--gain",
    dest="gain",
    type=float,
    default=1.0,
    help="Sets the gain/volume of the output audio (default is 1.0)",
)

parser.add_argument(
    "--8mm",
    dest="8mm",
    action="store_true",
    default=False,
    help="Sony 8mm formats Video8 & Hi8",
)

parser.add_argument(
    "--gnuradio",
    dest="GRC",
    action="store_true",
    default=False,
    help="Opens ZMQ REP pipe to gnuradio at port 5555",
)

parser.add_argument(
    "--gui", "--ui",
    dest="UI",
    action="store_true",
    default=False,
    help="Opens hifi-decode graphical interface",
)

parser.add_argument(
    "--audio_mode",
    dest="mode",
    type=str,
    default="s",
    help="Audio mode (s: stereo, mpx: stereo with mpx, l: left channel, r: right channel, sum: mono sum)",
)


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
        print(
            "WARN: ffmpeg not installed (or not in PATH), please install it to speed up file reading"
        )
        return False


class BufferedInputStream(io.RawIOBase):
    def __init__(self, buffer):
        self.buffer = buffer
        self._pos: int = 0

    def read(self, size=-1):
        data = self.buffer.read(size * 2)
        if not data:
            return b""

        self._pos += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return False

    def writable(self):
        return False

    def close(self):
        pass

    def fileno(self):
        return self.buffer.fileno()

    def isatty(self):
        return self.buffer.isatty()

    def tell(self):
        # hack, there is no way to know the current position
        return int(-1e6) if self._pos == 0 else self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        return self.tell()


# executes ffmpeg and reads stdout as a file
class FfmpegFileReader(BufferedInputStream):
    def __init__(self, file_path):
        shell_command = ["ffmpeg", "-i", file_path, "-f", "s16le", "pipe:1"]
        p = subprocess.Popen(
            shell_command,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )
        self.buffer = p.stdout
        self._pos: int = 0


class UnseekableSoundFile(sf.SoundFile):
    def __init__(self, file_path, mode, channels, samplerate, format, subtype, endian):
        self.file_path = file_path
        self._overlap = np.array([], dtype=np.int16)
        super().__init__(
            file_path,
            mode,
            channels=channels,
            samplerate=samplerate,
            format=format,
            subtype=subtype,
            endian=endian,
        )

    def seek(self, offset, whence=io.SEEK_SET):
        return self.file_path.seek(offset, whence)

    def close(self):
        pass

    def read(
        self,
        samples,
        frames=-1,
        dtype="float64",
        always_2d=False,
        fill_value=None,
        out=None,
    ):
        data = self.file_path.read(samples)
        if not data:
            return b""
        assert len(data) % 2 == 0, "data is misaligned"
        out = np.asarray(np.frombuffer(data, dtype=np.int16), dtype=dtype)
        return out

    def _read_next_chunk(self, blocksize, overlap, dtype) -> np.array:
        data = self.file_path.read(blocksize - overlap)
        assert len(data) % 2 == 0, "data is misaligned"
        out = np.asarray(np.frombuffer(data, dtype=np.int16), dtype=dtype)
        self._overlap = (
            np.copy(out[-overlap:]) if np.size(self._overlap) == 0 else self._overlap
        )
        result = np.concatenate((self._overlap, out))
        self._overlap = np.copy(out[-overlap:])
        return result

    # yields infinite generator for _read_next_chunk
    def blocks(
        self,
        blocksize=None,
        overlap=0,
        frames=-1,
        dtype="float64",
        always_2d=False,
        fill_value=None,
        out=None,
    ):
        while True:
            yield self._read_next_chunk(blocksize, overlap, dtype)


class UnSigned16BitFileReader(io.RawIOBase):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, "rb")

    def read(self, size=-1):
        data = self.file.read(size)
        if not data:
            return b""

        # Converts unsigned 16-bit to signed 16-bit
        samples = np.frombuffer(data, dtype=np.uint16)
        signed_samples = samples.astype(np.int16) - 32768

        return signed_samples.tobytes()

    def readable(self):
        return True

    def close(self):
        self.file.close()


# This part is what opens the file
# The samplerate here could be anything
def as_soundfile(pathR, sample_rate=44100):
    path = pathR.lower()
    if ".raw" in path or ".s16" in path:
        return sf.SoundFile(
            pathR,
            "r",
            channels=1,
            samplerate=int(sample_rate),
            format="RAW",
            subtype="PCM_16",
            endian="LITTLE",
        )
    elif ".u8" in path or ".r8" in path:
        return sf.SoundFile(
            pathR,
            "r",
            channels=1,
            samplerate=int(sample_rate),
            format="RAW",
            subtype="PCM_U8",
            endian="LITTLE",
        )
    elif ".s8" in path:
        return sf.SoundFile(
            pathR,
            "r",
            channels=1,
            samplerate=int(sample_rate),
            format="RAW",
            subtype="PCM_S8",
            endian="LITTLE",
        )
    elif ".u16" in path or ".r16" in path:
        return sf.SoundFile(
            UnSigned16BitFileReader(pathR),
            "r",
            channels=1,
            samplerate=int(sample_rate),
            format="RAW",
            subtype="PCM_16",
            endian="LITTLE",
        )
    elif "-" == path:
        return UnseekableSoundFile(
            BufferedInputStream(sys.stdin.buffer),
            "r",
            channels=1,
            samplerate=int(sample_rate),
            format="RAW",
            subtype="PCM_16",
            endian="LITTLE",
        )
    else:
        if test_if_ffmpeg_is_installed():
            return UnseekableSoundFile(
                FfmpegFileReader(pathR),
                "r",
                channels=1,
                samplerate=int(sample_rate),
                format="RAW",
                subtype="PCM_16",
                endian="LITTLE",
            )
        return sf.SoundFile(pathR, "r")


def as_outputfile(path, sample_rate):
    if ".wav" in path.lower():
        return sf.SoundFile(
            path,
            "w",
            channels=2,
            samplerate=int(sample_rate),
            format="WAV",
            subtype="PCM_16",
        )
    else:
        return sf.SoundFile(
            path,
            "w",
            channels=2,
            samplerate=int(sample_rate),
            format="FLAC",
            subtype="PCM_24",
        )


def seconds_to_str(seconds: float) -> str:
    return str(timedelta(seconds=seconds))


def log_decode(start_time: datetime, frames: int, decode_options: dict):
    elapsed_time: timedelta = datetime.now() - start_time
    audio_time: float = frames / (2 * decode_options["input_rate"])
    relative_speed: float = audio_time / elapsed_time.total_seconds()
    elapsed_time_format: str = seconds_to_str(elapsed_time.total_seconds())
    audio_time_format: str = seconds_to_str(audio_time)

    print(
        f"- Decoding speed: {round(frames / (1e3 * elapsed_time.total_seconds()))} kFrames/s ({relative_speed:.2f}x)\n"
        + f"- Audio position: {str(audio_time_format)[:-3]}\n"
        + f"- Wall time     : {str(elapsed_time_format)[:-3]}"
    )


def gain_adjust(audio: np.array, gain: float) -> np.array:
    return np.multiply(audio, gain)


def prepare_stereo(
    l_raw: np.array,
    r_raw: np.array,
    noise_reduction: NoiseReduction,
    decode_options: dict,
):
    if decode_options["mode"] == "mpx":
        l = np.multiply(np.add(l_raw, r_raw), 0.5)
        r = np.multiply(np.subtract(l_raw, r_raw), 0.5)
    elif decode_options["mode"] == "l":
        l = l_raw
        r = l_raw
    elif decode_options["mode"] == "r":
        l = r_raw
        r = r_raw
    elif decode_options["mode"] == "sum":
        l = np.multiply(np.add(l_raw, r_raw), 0.5)
        r = np.multiply(np.add(l_raw, r_raw), 0.5)
    else:
        l = l_raw
        r = r_raw

    if decode_options["noise_reduction"]:
        stereo = noise_reduction.stereo(
            gain_adjust(l, decode_options["gain"]),
            gain_adjust(r, decode_options["gain"]),
        )
    else:
        l, r = discard_stereo(l, r, noise_reduction.discard_size)
        stereo = list(
            map(
                list,
                zip(
                    gain_adjust(l, decode_options["gain"]),
                    gain_adjust(r, decode_options["gain"]),
                ),
            )
        )
    return stereo


def post_process(
    audioL: np.array, audioR: np.array, audio_rate: int, decode_options: dict
):
    left_audio = (
        samplerate_resample(
            audioL, decode_options["audio_rate"], audio_rate, "sinc_fastest"
        )
        if decode_options["audio_rate"] != audio_rate
        else audioL
    )
    right_audio = (
        samplerate_resample(
            audioR, decode_options["audio_rate"], audio_rate, "sinc_fastest"
        )
        if decode_options["audio_rate"] != audio_rate
        else audioR
    )

    return left_audio, right_audio


class AppWindow:
    def __init__(self, argv, decode_options):
        self._app, self._window = self.open_ui(argv, decode_options)

    def __del__(self):
        self._window.close()
        self._app.quit()

    def open_ui(self, argv, decode_options):
        app = QApplication(argv)
        print("Opening hifi-decode-gui...")
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


def decode(decoder, decode_options, ui_t: Optional[AppWindow] = None):
    input_file = decode_options["input_file"]
    output_file = decode_options["output_file"]
    start_time = datetime.now()
    noise_reduction = NoiseReduction(
        decoder.notchFreq,
        decode_options["nr_side_gain"],
        decoder.audioDiscard,
        audio_rate=decode_options["audio_rate"],
    )
    stereo_play_buffer = list()

    with as_outputfile(output_file, decode_options["audio_rate"]) as w:
        with as_soundfile(input_file) as f:
            progressB = TimeProgressBar(f.frames, f.frames)
            current_block = 0
            for block in f.blocks(
                blocksize=decoder.blockSize, overlap=decoder.readOverlap
            ):
                progressB.print(f.tell())
                current_block, audioL, audioR = decoder.block_decode(
                    block, block_count=current_block
                )
                left_audio, right_audio = post_process(
                    audioL, audioR, decoder.audioRate, decode_options
                )
                stereo = prepare_stereo(
                    left_audio, right_audio, noise_reduction, decode_options
                )

                if decode_options["auto_fine_tune"]:
                    log_bias(decoder)

                current_block += 1
                log_decode(start_time, f.tell(), decode_options)
                try:
                    w.write(stereo)
                    if decode_options["preview"]:
                        if SOUNDDEVICE_AVAILABLE:
                            if (
                                len(stereo_play_buffer)
                                > decode_options["audio_rate"] * 5
                            ):
                                sd.wait()
                                sd.play(
                                    stereo_play_buffer,
                                    decode_options["audio_rate"],
                                    blocking=False,
                                )
                                stereo_play_buffer = list()
                            stereo_play_buffer += stereo
                        else:
                            print(
                                "Import of sounddevice failed, preview is not available!"
                            )
                except ValueError:
                    pass
                if ui_t is not None:
                    ui_t.app.processEvents()
                    if ui_t.window.transport_state == 0:
                        break
                    elif ui_t.window.transport_state == 2:
                        while ui_t.window.transport_state == 2:
                            ui_t.app.processEvents()
                            time.sleep(0.01)

        elapsed_time = datetime.now() - start_time
        dt_string = elapsed_time.total_seconds()
        print(f"\nDecode finished, seconds elapsed: {round(dt_string)}")


def decode_parallel(
    decoders: List[HiFiDecode],
    decode_options: dict,
    threads: int = 8,
    ui_t: Optional[AppWindow] = None,
):
    input_file = decode_options["input_file"]
    output_file = decode_options["output_file"]
    start_time = datetime.now()
    block_size = decoders[0].blockSize
    read_overlap = decoders[0].readOverlap
    noise_reduction = NoiseReduction(
        decoders[0].notchFreq,
        decode_options["nr_side_gain"],
        decoders[0].audioDiscard,
        audio_rate=decode_options["audio_rate"],
    )
    futures_queue = list()
    executor = ThreadPoolExecutor(threads)
    current_block = 0
    stereo_play_buffer = list()
    with as_outputfile(output_file, decode_options["audio_rate"]) as w:
        with as_soundfile(input_file) as f:
            progressB = TimeProgressBar(f.frames, f.frames)
            for block in f.blocks(blocksize=block_size, overlap=read_overlap):
                decoder = decoders[current_block % threads]
                if len(block) > 0:
                    futures_queue.append(
                        executor.submit(decoder.block_decode, block, current_block)
                    )
                else:
                    break

                if decode_options["auto_fine_tune"]:
                    log_bias(decoder)

                current_block += 1
                progressB.print(f.tell())

                while len(futures_queue) > threads:
                    future = futures_queue.pop(0)
                    blocknum, audioL, audioR = future.result()
                    left_audio, right_audio = post_process(
                        audioL, audioR, decoder.audioRate, decode_options
                    )
                    stereo = prepare_stereo(
                        left_audio, right_audio, noise_reduction, decode_options
                    )
                    log_decode(start_time, f.tell(), decode_options)
                    try:
                        w.write(stereo)
                        if decode_options["preview"]:
                            if SOUNDDEVICE_AVAILABLE:
                                if (
                                    len(stereo_play_buffer)
                                    > decode_options["audio_rate"] * 5
                                ):
                                    sd.wait()
                                    sd.play(
                                        stereo_play_buffer,
                                        decode_options["audio_rate"],
                                        blocking=False,
                                    )
                                    stereo_play_buffer = list()
                                stereo_play_buffer += stereo
                            else:
                                print(
                                    "Import of sounddevice failed, preview is not available!"
                                )
                    except ValueError:
                        pass

                if ui_t is not None:
                    ui_t.app.processEvents()
                    if ui_t.window.transport_state == 0:
                        break
                    elif ui_t.window.transport_state == 2:
                        while ui_t.window.transport_state == 2:
                            ui_t.app.processEvents()
                            time.sleep(0.01)

            print("Emptying the decode queue ...")
            while len(futures_queue) > 0:
                future = futures_queue.pop(0)
                blocknum, audioL, audioR = future.result()
                left_audio, right_audio = post_process(
                    audioL, audioR, decoder.audioRate, decode_options
                )
                stereo = prepare_stereo(
                    left_audio, right_audio, noise_reduction, decode_options
                )
                try:
                    w.write(stereo)
                except ValueError:
                    pass

        elapsed_time = datetime.now() - start_time
        dt_string = elapsed_time.total_seconds()
        print(f"\nDecode finished, seconds elapsed: {round(dt_string)}")


def guess_bias(decoder, input_file, block_size, blocks_limits=10):
    print("Measuring carrier bias ... ")
    blocks = list()

    with as_soundfile(input_file) as f:
        while f.tell() < f.frames and len(blocks) <= blocks_limits:
            blocks.append(f.read(block_size))

    LCRef, RCRef = decoder.guessBiases(blocks)
    print("done!")
    print(
        "L carrier found at %.02f MHz, R carrier found at %.02f MHz"
        % (LCRef / 1e6, RCRef / 1e6)
    )
    return LCRef, RCRef


def log_bias(decoder: HiFiDecode):
    devL = (decoder.standard.LCarrierRef - decoder.afe_params.LCarrierRef) / 1e3
    devR = (decoder.standard.RCarrierRef - decoder.afe_params.RCarrierRef) / 1e3
    print("Bias L %.02f kHz, R %.02f kHz" % (devL, devR), end=" ")
    if abs(devL) < 9 and abs(devR) < 9:
        print("(good player/recorder calibration)")
    elif 9 <= abs(devL) < 10 or 9 <= abs(devR) < 10:
        print("(maybe marginal player/recorder calibration)")
    else:
        print(
            "\nWARN: the player or the recorder may be uncalibrated and/or\n"
            "the standard and/or the sample rate specified are wrong"
        )


def run_decoder(args, decode_options, ui_t: Optional[AppWindow] = None):
    print("Starting decode...")
    sample_freq = decode_options["input_rate"]
    filename = decode_options["input_file"]

    if sample_freq is not None:
        decoder = HiFiDecode(decode_options)
        LCRef, RCRef = decoder.standard.LCarrierRef, decoder.standard.RCarrierRef
        if args.BG:
            LCRef, RCRef = guess_bias(decoder, filename, int(decoder.sample_rate))
            decoder.updateAFE(LCRef, RCRef)

        if args.threads > 1 and not args.GRC:
            decoders = list()
            for i in range(0, args.threads):
                decoders.append(HiFiDecode(decode_options))
                decoders[i].updateAFE(LCRef, RCRef)
            decode_parallel(decoders, decode_options, threads=args.threads, ui_t=ui_t)
        else:
            decode(decoder, decode_options, ui_t=ui_t)
        print("Decode finished successfully")
        return 0
    else:
        print("No sample rate specified")
        return 0


def main() -> int:
    args = parser.parse_args()

    system = select_system(args)
    sample_freq = select_sample_freq(args)

    filename, outname, _, _ = get_basics(args)
    if not args.UI and not args.overwrite:
        if os.path.isfile(outname):
            print(
                "Existing decode files found, remove them or run command with --overwrite"
            )
            print("\t", outname)
            return 1

    print("Initializing ...")

    real_mode = "s" if not args.8mm else "mpx"
    real_mode = args.mode if args.mode in ["l", "r", "sum"] else real_mode

    decode_options = {
        "input_rate": sample_freq * 1e6,
        "standard": "p" if system == "PAL" else "n",
        "format": "vhs" if not args.8mm else "8mm",
        "preview": args.preview,
        "preview_available": SOUNDDEVICE_AVAILABLE,
        "original": args.original,
        "noise_reduction": args.noise_reduction == "on" if not args.preview else False,
        "auto_fine_tune": args.auto_fine_tune == "on" if not args.preview else False,
        "nr_side_gain": args.NR_side_gain,
        "grc": args.GRC,
        "audio_rate": args.rate,
        "gain": args.gain,
        "input_file": filename,
        "output_file": outname,
        "mode": real_mode,
    }

    if decode_options["format"] == "vhs":
        print("PAL VHS format selected") if system == "PAL" else print(
            "NTSC VHS format selected"
        )
    else:
        print("NTSC Hi8 format selected")

    if args.UI and not HIFI_UI:
        print(
            "PyQt5/PyQt6 is not installed, can not use graphical UI, falling back to command line interface.."
        )

    if args.UI and HIFI_UI:
        ui_t = AppWindow(sys.argv, decode_options)
        decoder_state = 0
        try:
            while ui_t.window.isVisible():
                if ui_t.window.transport_state == 1:
                    print("Starting decode...")
                    options = ui_parameters_to_decode_options(ui_t.window.getValues())
                    # change to output file directory
                    if os.path.dirname(options["output_file"]) != "":
                        os.chdir(os.path.dirname(options["output_file"]))

                    # test input and output files
                    if test_input_file(options["input_file"]) and test_output_file(
                        options["output_file"]
                    ):
                        decoder_state = run_decoder(args, options, ui_t=ui_t)
                        ui_t.window.transport_state = 0
                        ui_t.window.on_decode_finished()
                    else:
                        message = None
                        if not test_input_file(options["input_file"]):
                            message = f"Input file '{options['input_file']}' not found"
                        elif not test_output_file(options["output_file"]):
                            message = f"Output file '{options['output_file']}' cannot be created nor overwritten"

                        ui_t.window.generic_message_box(
                            "I/O Error", message, QMessageBox.Icon.Critical
                        )
                        ui_t.window.on_stop_clicked()

                ui_t.app.processEvents()
                time.sleep(0.01)
            ui_t.window.transport_state = 0
        except (KeyboardInterrupt, RuntimeError):
            pass

        return decoder_state
    else:
        if test_input_file(filename) and test_output_file(outname):
            return run_decoder(args, decode_options)
        else:
            parser.print_help()
            print(
                "ERROR: input file not found"
                if not test_input_file(filename)
                else f"ERROR: output file '{outname}' cannot be created nor overwritten"
            )
            return 1


if __name__ == "__main__":
    sys.exit(main())
