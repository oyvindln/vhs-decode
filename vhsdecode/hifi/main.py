#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from datetime import datetime, timedelta
import os
import sys
from typing import List

import numpy as np
import soundfile as sf
from vhsdecode.addons.chromasep import samplerate_resample

from vhsdecode.cmdcommons import (
    common_parser_cli,
    select_sample_freq,
    select_system,
    get_basics,
)
from vhsdecode.hifi.HiFiDecode import HiFiDecode, NoiseReduction, DEFAULT_NR_GAIN_
from vhsdecode.hifi.TimeProgressBar import TimeProgressBar
import io


parser, _ = common_parser_cli(
    "Extracts audio from raw VHS HiFi FM capture",
    default_threads=round(cpu_count() / 2),
)

parser.add_argument(
    "--audio_rate",
    dest="rate",
    type=int,
    default=192000,
    help="Output sample rate in Hz (default 192000)",
)

parser.add_argument(
    "--bg", dest="BG", action="store_true", default=False, help="Do carrier bias guess"
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
    "--h8", dest="H8", action="store_true", default=False, help="Video8/Hi8, 8mm tape format"
)

parser.add_argument(
    "--gnuradio",
    dest="GRC",
    action="store_true",
    default=False,
    help="Opens ZMQ REP pipe to gnuradio at port 5555",
)


class LDFReaderInputStream(io.RawIOBase):
    def __init__(self, buffer):
        self.buffer = buffer
        self._pos: int = 0

    def read(self, size=-1):
        data = self.buffer.read(size * 2)
        if not data:
            return b''

        self._pos += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return False

    def writable(self):
        return False

    def close(self):
        self.buffer.close()

    def fileno(self):
        return self.buffer.fileno()

    def isatty(self):
        return self.buffer.isatty()

    def tell(self):
        # hack, there is no way to know the current position
        return int(-1e6) if self._pos == 0 else self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        return self.tell()


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

    def read(self, samples, frames=-1, dtype='float64', always_2d=False,
             fill_value=None, out=None):
        data = self.file_path.read(samples)
        if not data:
            return b''
        assert len(data) % 2 == 0, "data is misaligned"
        out = np.asarray(np.frombuffer(data, dtype=np.int16), dtype=dtype)
        return out

    def _read_next_chunk(self, blocksize, overlap, dtype):
        data = self.file_path.read(blocksize)
        assert len(data) % 2 == 0, "data is misaligned"
        out = np.asarray(np.frombuffer(data, dtype=np.int16), dtype=dtype)
        self._overlap = np.copy(out[-overlap:])
        return np.concatenate((self._overlap, out))

    # yields infinite generator for _read_next_chunk
    def blocks(self, blocksize=None, overlap=0, frames=-1, dtype='float64',
               always_2d=False, fill_value=None, out=None):
        while True:
            yield self._read_next_chunk(blocksize, overlap, dtype)


class UnSigned16BitFileReader(io.RawIOBase):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'rb')

    def read(self, size=-1):
        data = self.file.read(size)
        if not data:
            return b''

        # Convierte de unsigned 16-bit a signed 16-bit
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
    elif ".ogx" in path:
        sys.exit("OGX container not supported yet, try using pipe input")
    elif "-" == path:
        return UnseekableSoundFile(
            LDFReaderInputStream(sys.stdin.buffer),
            "r",
            channels=1,
            samplerate=int(sample_rate),
            format="RAW",
            subtype="PCM_16",
            endian="LITTLE",
        )
    else:
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
    audio_time: float = frames / decode_options["input_rate"]
    relative_speed: float = audio_time / elapsed_time.total_seconds()
    elapsed_time_format: str = seconds_to_str(elapsed_time.total_seconds())
    audio_time_format: str = seconds_to_str(audio_time)

    print(
        f"- Decoding speed: {round(frames / (1e3 * elapsed_time.total_seconds()))} kFrames/s ({relative_speed:.2f}x)\n" +
        f"- Audio position: {str(audio_time_format)[:-3]}\n" +
        f"- Wall time     : {str(elapsed_time_format)[:-3]}"
    )


def decode(decoder, input_file, decode_options, output_file):
    start_time = datetime.now()
    noise_reduction = NoiseReduction(
        decoder.notchFreq,
        decode_options["nr_side_gain"],
        decoder.audioDiscard,
        audio_rate=decode_options["audio_rate"]
    )

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
                left_audio = samplerate_resample(audioL,
                                                 decode_options["audio_rate"],
                                                 decoder.audioRate,
                                                 "sinc_fastest") \
                    if decode_options["audio_rate"] != decoder.audioRate else audioL
                right_audio = samplerate_resample(audioR,
                                                  decode_options["audio_rate"],
                                                  decoder.audioRate,
                                                  "sinc_fastest") \
                    if decode_options["audio_rate"] != decoder.audioRate else audioR

                if decode_options["noise_reduction"]:
                    stereo = noise_reduction.stereo(left_audio, right_audio)
                else:
                    stereo = list(map(list, zip(left_audio, right_audio)))

                if decode_options["auto_fine_tune"]:
                    log_bias(decoder)

                log_decode(start_time, f.tell(), decode_options)
                w.write(stereo)

        elapsed_time = datetime.now() - start_time
        dt_string = elapsed_time.total_seconds()
        print(f"\nDecode finished, seconds elapsed: {round(dt_string)}")


def decode_parallel(decoders: List[HiFiDecode],
                    input_file: str,
                    output_file: str,
                    decode_options: dict,
                    threads: int = 8):
    start_time = datetime.now()
    block_size = decoders[0].blockSize
    read_overlap = decoders[0].readOverlap
    noise_reduction = NoiseReduction(
        decoders[0].notchFreq,
        decode_options["nr_side_gain"],
        decoders[0].audioDiscard,
        audio_rate=decode_options["audio_rate"]
    )
    futures_queue = list()
    executor = ThreadPoolExecutor(threads)
    current_block = 0
    with as_outputfile(output_file, decode_options["audio_rate"]) as w:
        with as_soundfile(input_file) as f:
            progressB = TimeProgressBar(f.frames, f.frames)
            for block in f.blocks(blocksize=block_size, overlap=read_overlap):
                decoder = decoders[current_block % threads]
                futures_queue.append(
                    executor.submit(
                        decoder.block_decode,
                        block, current_block
                    )
                )
                if decode_options["auto_fine_tune"]:
                    log_bias(decoder)

                current_block += 1
                progressB.print(f.tell())

                while len(futures_queue) > threads:
                    future = futures_queue.pop(0)
                    blocknum, audioL, audioR = future.result()
                    left_audio = samplerate_resample(audioL,
                                                     decode_options["audio_rate"],
                                                     decoder.audioRate,
                                                     "sinc_fastest") \
                        if decode_options["audio_rate"] != decoder.audioRate else audioL
                    right_audio = samplerate_resample(audioR,
                                                      decode_options["audio_rate"],
                                                      decoder.audioRate,
                                                      "sinc_fastest") \
                        if decode_options["audio_rate"] != decoder.audioRate else audioR
                    if decode_options["noise_reduction"]:
                        stereo = noise_reduction.stereo(left_audio, right_audio)
                    else:
                        stereo = list(map(list, zip(left_audio, right_audio)))
                    log_decode(start_time, f.tell(), decode_options)
                    w.write(stereo)

            print("Emptying the decode queue ...")
            while len(futures_queue) > 0:
                future = futures_queue.pop(0)
                blocknum, audioL, audioR = future.result()
                if decode_options["noise_reduction"]:
                    stereo = noise_reduction.stereo(audioL, audioR)
                else:
                    stereo = list(map(list, zip(audioL, audioR)))
                log_decode(start_time, f.tell(), decode_options)
                w.write(stereo)

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


def main():
    args = parser.parse_args()

    system = select_system(args)
    sample_freq = select_sample_freq(args)
    decode_options = {
        "input_rate": sample_freq * 1e6,
        "standard": "p" if system == "PAL" else "n",
        "format": "vhs" if not args.H8 else "h8",
        "preview": args.preview,
        "original": args.original,
        "noise_reduction": args.noise_reduction == "on" if not args.preview else False,
        "auto_fine_tune": args.auto_fine_tune == "on" if not args.preview else False,
        "nr_side_gain": args.NR_side_gain,
        "grc": args.GRC,
        "audio_rate": args.rate,
    }

    filename, outname, _, _ = get_basics(args)
    if not args.overwrite:
        if os.path.isfile(outname):
            print(
                "Existing decode files found, remove them or run command with --overwrite"
            )
            print("\t", outname)
            sys.exit(1)

    print("Initializing ...")
    if decode_options["format"] == "vhs":
        print("PAL VHS format selected") if system == "PAL" else print(
            "NTSC VHS format selected"
        )
    else:
        print("NTSC Hi8 format selected")

    if sample_freq is not None:
        decoder = HiFiDecode(decode_options)
        LCRef, RCRef = decoder.standard.LCarrierRef, decoder.standard.RCarrierRef
        if args.BG:
            LCRef, RCRef = guess_bias(decoder, filename, decoder.blockSize)
            decoder.updateAFE(LCRef, RCRef)

        if args.threads > 1 and not args.GRC:
            decoders = list()
            for i in range(0, args.threads):
                decoders.append(HiFiDecode(decode_options))
                decoders[i].updateAFE(LCRef, RCRef)
            decode_parallel(
                decoders, filename, outname, decode_options, threads=args.threads
            )
        else:
            decode(decoder, filename, decode_options, outname)
    else:
        print("No sample rate specified")
        sys.exit(0)


if __name__ == "__main__":
    main()
