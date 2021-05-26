![vhs-decode logo](https://github.com/Zcooger/ld-decode/blob/017907c51f274e5186d20ae8ffff9ea6cc6fd62c/docs/vhs-decode%20logo%20256px.png)

# vhs-decode

A fork of [LD-Decode](https://github.com/happycube/ld-decode), the decoding software powering the Domesday86 project.  
This version has been modified to work with the differences found in RF signals taken directly from videotapes
(not to be confused with the antenna connector on the back of the VCR!);
it is not possible to use both LD-Decode and VHS-Decode side by side without reinstalling either one at runtime.

Currently, only (S-)VHS and U-Matic format tapes are supported;
of those, only NTSC and PAL variants are supported, with plans and/or ongoing work to support more formats and systems.

# Dependencies

VHS-Decode, as with LD-Decode, has been developed and tested on machines running the latest version of Ubuntu and Linux Mint.
Other distributions might have outdated (or even newer) versions of certain Python libraries or other dependencies, breaking compatibility. It has been confirmed to work via WSL2.

Other dependencies include Python 3.5+, Qt5, Qmake, and FFmpeg.

Hardware dependencies revolve mainly around hardware used to perform RF captures, as Python is largely platform-agnostic.
For capturing, VHS-Decode supports both the [Domesday Duplicator](https://github.com/happycube/ld-decode/wiki/Domesday-Duplicator) and PCI-socketed capture cards based on Conexant CX23880/1/2/3 chipset using the [CXADC](https://github.com/happycube/cxadc-linux3) kernel module (including variants with PCI-Express x1 bridge).

# Installation and running the software

Install all dependencies required by LD-Decode and VHS-Decode:

    sudo apt install build-essential git ffmpeg libavcodec-dev libavformat-dev qt5-default libqwt-qt5-dev qt5-qmake qtbase5-dev python3 python3-pip python3-distutils libfftw3-dev openssl && sudo pip3 install numba pandas matplotlib scipy numpy samplerate

Download VHS-Decode:

    git clone https://github.com/oyvindln/ld-decode.git vhs-decode

Compile and install Domesday tools:

    make -j8 all && sudo make install && make clean
    
Capture 30 seconds of tape signal using [CXADC](https://github.com/happycube/cxadc-linux3) driver in 8-bit mode and asynchronous audio from line input.
It is recommended to use fast storage:

    timeout 30s cat /dev/cxadc0 > <capture>.r8 | ffmpeg -f alsa -i default -compression_level 12 -y <capture>.flac

Decode your captured tape by using:

    ~/./vhs-decode/vhs-decode [arguments] <capture>.tbc <capture>
    
Or:

    python3 vhs-decode [arguments] <capture>.tbc <capture>
    
View decoded tape using:

    ld-analyse <capture>.tbc

# Generating video files

VHS-Decode produces .tbc, .json and .log files, usable only with the LD-Decode family of tools (ld-analyse, ld-process-vbi, and ld-process-vits).
To generate .mkv files viewable in most media players, simply use the scripts found in the root folder:

    ~/./vhs-decode/gen_chroma_vid_pal.sh test

And:

    ~/./vhs-decode/gen_chroma_vid_ntsc.sh test

These will generate a lossless, interlaced, high-bitrate (roughly 100-150 Mb/s) files which,
although ideal for archival and reducing further loss in quality, are unsuitable for sharing online.
An alternate processing mode is included in the script files, but commented out.

# Terminal arguments

VHS-Decode supports various arguments to process differently captured tape recordings and different tape formats/systems.
These tend to change frequently as new features are added or superseded.

```--cxadc, --10cxadc, --cxadc3, --10cxadc3, -f```: Changes the sample rate and bit depth for the decoder.
By default, this is set to 40 MHz (the sample rate used by the Domesday Duplicator) at 16 bits.
These flags change this to 28.6 MHz/8-bit, 28.6 MHz/16-bit, 35.8 MHz/8-bit and 35.8 MHz/10-bit, respectively.
See the readme file for cxadc-linux3 for more information on what each mode and capture rate means.
```-f``` sets the frequency to a custom, user-defined one (expressed as an integer, ie ```-f 40000000``` for 40 MHz input).

```-n, -p, -pm, --NTSCJ```: changes the color system to NTSC, PAL, PAL-M, or NTSC-J, respectively.
Please note that, as of writing, support for PAL-M is **experimental** and NTSC-J is **untested**.

```-s, --start_fileloc, -l```: Use for jumping ahead in a file or defining limit.
Useful for recovering decoding after a crash, or by limiting process time by producing shorter samples.
```-s``` jumps ahead to any given frame in the capture,
```--start_fileloc``` jumps ahead to any given *sample* in the capture
(note: upon crashing, vhs-decode automatically dumps the last known sample location in the terminal output) and
```-l``` limits decode length to *n* frames.

```-t```: defines the number of processing threads to use during decoding.
By default, the main vhs-decode script allocates only one thread, though the gen_chroma_vid scripts allocate two.
The ```make``` rule of thumb of "number of logical processors, plus one" generally applies here,
though it mainly depends on the amount of memory available to the decoder.

# Advanced/Debug features

See [advanced_flags.md](advanced_flags.md) for more information.


# Supported formats

Tapes:

(S-)VHS 625-line and 525-line, PAL and NTSC (SECAM WIP).
U-Matic 625-line and 525-line Low Band, PAL and NTSC.

Input formats:

.r8/.u8 (8-bit raw data), .r16/.u16 (16-bit raw data), .flac/.vhs/.svhs/.cvbs (FLAC-compressed captures, can be either 8-bit or 16-bit).

Output formats:

Unlike LD-Decode, VHS-Decode does not output its timebase-corrected frames as single .tbc file - instead, it splits both the luminance and chrominance data as individual .tbc files, one "main" comprising luma data, and a "sub" containing solely chroma. Thus, along the usual log file and .json frame descriptor table, there are *two* .tbc files, whereas only the "main" file has to be processed afterwards to produce a usable video file.

# Join us

[Discord](https://discord.gg/pVVrrxd)
[Facebook](https://www.facebook.com/groups/2070493199906024)

# Documentation
Documentation is available via the GitHub wiki. This includes installation and usage instructions. Start with the wiki if you have any questions.

https://github.com/happycube/ld-decode/wiki

## *If in doubt - Read the Wiki!*
