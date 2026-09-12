# Scripts

## ld-compress

ld-compress is a script to simplify the compression of .lds (raw LaserDisc RF files) into .ldf images.

ld-decode fully supports FLAC compressed files as input.  Files can be suffixed with .ldf as shown here, or .raw.oga.  ld-decode will automatically uncompress the input file during processing.

To compress a .lds file simply use:

```
ld-compress <filename>.lds
```

This script will write a .ldf compressed version of the .lds file to the directory it's called from.

### Encoding

ld-compress encodes with multithreaded [flac](https://xiph.org/flac/){target="_blank"}, which requires flac 1.5.0 or later.  This is the only external program ld-compress uses, and every ld-decode package ships it alongside the ld-compress command, so there is nothing to install separately.  If you are running from a source checkout instead, put a flac 1.5.0 or later on your PATH.

Uncompression (`-u`) and verification (`-v`) need nothing external at all: they decode with PyAV, the same FFmpeg binding ld-decode itself reads .ldf files with, and pack the result with the same code as `ld-lds-converter-py`.

The `-l` compression level ranges from 1 to 8, defaulting to 8 (best compression).

While it works, ld-compress shows a progress bar with the percentage complete, the amount of the input file read, the throughput and an estimated time remaining:

```
disc01.lds [==============>           ]  59% 17.0GiB/28.6GiB 9.5MiB/s ETA 0:21:04
```

The bar is only drawn when ld-compress is run at a terminal, so redirecting its output to a log file keeps the log clean.  Use `-n` to turn it off at a terminal too, or `-p` to force it on when standard error is not a terminal.

### Windows

ld-compress is an ordinary command on Windows, the same as on Linux and macOS - run `bin\ld-compress.bat` from the portable ZIP.  It finds the `bin\flac.exe` that ships beside it without any PATH setup.

Save a file like this as `.bat` to make a drag and drop compressor:

```
@echo off
title Compressing : %~n1
"C:\path\to\ld-decode\bin\ld-compress.bat" "%~1"

pause
```

If you are still using the legacy ld-tools-suite, its
`C:\ld-tools-suite-windows\ld-lds-converter.exe` produces output byte-identical
to the `ld-lds-converter-py` that ships with ld-decode.

### Command List

The full list of command line options is as follows:

```
usage: ld-compress [-h] [-c | -u | -v] [-l 1-8] [-g] [-p | -n] [--version]
                   file [file ...]

ld-compress - compress and uncompress LaserDisc RF captures

positional arguments:
  file                 file(s) to process

options:
  -h, --help           show this help message and exit
  -c, --compress       compress .lds files to .ldf files in the current
                       directory (default)
  -u, --uncompress     uncompress .ldf/.raw.oga files to .lds files in the
                       current directory
  -v, --verify         print md5 checksums of the given .ldf/.raw.oga files
                       and of the .lds data they contain
  -l 1-8, --level 1-8  compression level 1 - 8 (default 8)
  -g, --oga            use the .raw.oga extension instead of .ldf when
                       compressing
  -p, --progress       always show the progress display, even when stderr is
                       not a terminal
  -n, --no-progress    never show the progress display
  --version            show program's version number and exit
```

A progress bar is shown by default when standard error is a terminal.

## ld-cut

ld-cut is a utility for cutting samples from raw RF LaserDisc captures (useful to create samples of trouble-areas when issue reporting), and can now also be used to compress .lds files.  The utility allows you to seek and specify start and end frames similar to the main ld-decode application.

```
usage: ld-cut [-h] [-s start] [-l length] [-S seek] [-E end] [-p] [-n]
              infile outfile

Extract a sample area from raw RF laserdisc captures. (Similar to ld-decode,
except it outputs samples)

positional arguments:
  infile                source file
  outfile               destination file (recommended to use .lds or .ldf suffixes)

optional arguments:
  -h, --help            show this help message and exit
  -s start, --start start
                        rough jump to frame n of capture (default is 0)
  -l length, --length length
                        limit length to n frames
  -S seek, --seek seek  seek to frame n of capture
  -E end, --end end     cutting: last frame
  -p, --pal             source is in PAL format
  -n, --ntsc            source is in NTSC format
```

Using ld-cut, you can do parallel .ldf encodings (optionally targeting different directories) using shell scripting pretty easily:

```
for i in f1.lds f2.lds f3.lds f4.lds; do (ld-cut $i /someotherdirectory/`basename -s .lds $i`.ldf &); done
```
