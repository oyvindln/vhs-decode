# LD-Process-AC3

Developed for [LD-Decode](https://github.com/happycube/ld-decode), this project decodes AC3 Audio from an RF signal.

![Licence](https://img.shields.io/github/license/LeightonSmallshire/LD-Process-AC3)
![Stars](https://img.shields.io/github/stars/LeightonSmallshire/LD-Process-AC3)
![Issues](https://img.shields.io/github/issues/LeightonSmallshire/LD-Process-AC3)

# Description

The first executable, ld-ac3-demodulate, takes a stream of unsigned 8-bit samples at 46.08MHz to produce a stream of QPSK
symbols, ready for the next executable. The second executable, ld-ac3-demodulate, decodes the stream of symbols into
playable ac3 audio frames, while producing
Reed-solomon, CRC and other statistics in the log file.

# Example Usage

#### Syntax

```
ld-ac3-demodulate [options] source_file output_file [log_file]
ld-ac3-decode [options] source_file output_file [log_file]
```

#### example_usage.sh

```
ffmpeg -hide_banner -y -i "$path" -f s16le -c:a pcm_s16le TP0
sox -r 40000000 -b 16 -c 1 -e signed -t raw TP0 -b 8 -r 46080000 -e unsigned -c 1 -t raw TP1 sinc -n 500 2600000-3160000
cmake-build-debug/demodulate/ld-ac3-demodulate TP1 TP2 demodulate_log
cmake-build-debug/decode/ld-ac3-decode TP2 "$outpath" decode_log
```

In the example usage, ffmpeg and sox are used to format, resample and filter the source signal before processing with
ld-ac3-demodulate and ld-ac3-decode. The TP0, TP1 and TP3 files are intermediate files, used for caching and are not used
when piping directly between the commands.

## Credit

Initial commit by [Leighton Smallshire](https://github.com/LeightonSmallshire)
and [Ian Smallshire](https://github.com/IanSmallshire)
based on work done by [Staffan Ulfberg](https://bitbucket.org/staffanulfberg/ldaudio).
