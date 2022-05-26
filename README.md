![vhs-decode logo](docs/vhs-decode_logo_256px.png)

# VHS-Decode (It does more than VHS now!)

A fork of [LD-Decode](https://github.com/happycube/ld-decode), the decoding software powering the [Domesday86 Project](https://www.domesday86.com/).  
This version has been modified to work with the differences found in the tracked RF drum head signals taken directly from videotapes.

(Not to be confused with the TV Modulator/Demodulator pack or the **"antenna connectors"** on the back of the VCR!).

![](assets/images/DdD-EBU-Colour-Bar-PAL-VHS-SP.png)

EBU ColourBar Test Tape With [WSS](https://en.wikipedia.org/wiki/Widescreen_signaling) (PAL) export exported full-frame (1112 x 624)

Currently, only VHS/S-VHS and U-Matic format tapes are fully supported;
of those, only NTSC, PAL and PAL-M variants fully* are supported, with plans and/or ongoing work to support more formats and systems check with [Media Format Support List.](https://github.com/oyvindln/vhs-decode/wiki/Tape-Support-List)

By default, the main VHS-Decode script allocates only one thread, though the gen_chroma_vid scripts allocate two threads.

The `make` rule of thumb of "number of logical processors, plus one" generally applies here,
though it mainly depends on the amount of memory available to the decoder.

## CVBS-Decode

The repository also contains an **experimental** CVBS decoder, `cvbs-decode`, which shares code with ld-decode and vhs-decode. Capable of decoding basic raw digitized NTSC and PAL composite video, including colour if the source is somewhat stable. Samples can be captured using cxadc, however, this is somewhat buggy as the cx chip can decide to resample or do other things if it detects a valid video signal.

Test samples & signals can be generated using [HackTV](https://github.com/fsphil/hacktv)

Note for test media generation AJA/Magewell/Blackmagic and even some consumer digital to analogue converters have test generators built-in some prosumer/broadcast decks also have generators built in same for later camcorders.

# ld-tools Windows

The ld-tools suit has been ported to windows, LD-Analyse fully works please see the wiki for more information [Windows Tools Builds](https://github.com/oyvindln/vhs-decode/releases)

# Dependencies - Hardware

## A Tape Player (VCR/VTR etc)

Preferably adjusted per tape and in excellent mechanical and head condition, S-VHS decks are preferable as they were built generally better than cheaper later consumer VHS decks, and are easier to adjust tape guides to achieve optimal alignment however any working clean VCR should work.

Its recommended if possible to fully service your VCR/VTR one should inspect heads and solder joint conditions (note parts with removable shielding may go unchecked), replace expanded or leaky capacitors etc, but at the minium clean the heads with 99.9% Isopropanol and lint free cloths/pads/paper & making sure to re-lubricate metal and plastic moving joints cogs and bearings with appropriate grease's and oils.

Its good practice to not cross contaminate tapes especially if dealing with mouldy or contained tapes always clean your tape track/drum/heads before and afterwards!

## RF Capture Device

## [Domesday Duplicator (DdD)](https://github.com/happycube/ld-decode/wiki/Domesday-Duplicator) (Method 01 - 300-350USD*)

Capture is done using an simple GUI application.

[Linux Application](https://github.com/simoninns/DomesdayDuplicator)

[Windows Application](https://github.com/TokugawaHeavyIndustries/DomesdayDuplicator-WinBuild/releases/tag/v2.2)

Originally geared towards capturing RF from laserdiscs players, it does however also work perfectly well for digitizing VHS RF data. It consists of a custom analogue to digital board with an amplifier, an off-shelf DE0-NANO FPGA development board, and a Cypress FX3 SuperSpeed Explorer USB 3.0 board.

[More Information](https://www.domesday86.com/?page_id=978) / [How To Aquire?](https://docs.google.com/document/d/1liOpdM6z51AwM5jGLoUak6h9aJ0eY0fQduE5k4TcleU/edit?usp=sharing) / [How to Fabricate & Flash?](https://docs.google.com/document/d/1k2bPPwHPoG7xXpS1NCYEe3w_jD_ts0yRCp-2aQ6QoKY/edit?usp=sharing)

## [CX Card & CXADC](https://github.com/happycube/cxadc-linux3) (Method 02 - 20-35USD)

Capture & Config uses simple command-line arguments and parameters [CXADC](https://github.com/happycube/cxadc-linux3)

There is now an [CXADC Wiki](https://github.com/happycube/cxadc-linux3/wiki) explaining best card types and guides for modifications such as crystal upgrades.

The budget approach is using a video capture card based on a Conexant CX23880/1/2/3 PCI chipset. With a modified Linux driver, these cards can be forced to output RAW data that can be captured to file, instead of decoding video normally as they otherwise would.

There are now ‘’New’’ Chinese variants that can be found on AliExpress that have integrated Asmedia or ITE 1x PCIE bridge chips allowing modern systems to use them.

The cards sadly however at stock without any modifications these have more self-noise compared to the DdD with about a 3db signal to noise difference, Currently, the CX23883-39 based white variant cards in recent testing have been consistently lower noise.

# Deployment of Capture Hardware

Please Read the [VHS-Decode Wiki](https://github.com/oyvindln/vhs-decode/wiki) for more in-depth information.

If there is no info on your VCR in the wiki then acquire the service manual for your device, Google it!

Your find and probe/solder an BNC bulk head to your RF output starting with test points.

Test points normally will be called the following:

**RF Y, RF C, RF Y+C, PB, V RF, V ENV, ENV, ENVELOPE, VIDEO ENVE, VIDEO ENVELOPE** - Video

**RF-Out, A-RF, HIFI RF** - Audio

These can be found on mainboard test points on consumer VCR's, but on higher-end units you have more options, once located solder an pigtail BNC or SMA connector for easy signal hook up to your capture device and mount to the back of the device or thread though an opening if available, do not make sharp bends in cabling.

Some UMATIC decks have a direct RF output on the back that may be viable. (*needs expanding on*)

# Dependencies - Software

VHS-Decode, as with LD-Decode, has been developed and tested on machines running the latest version of Ubuntu and Linux Mint.
Other distributions might have outdated (or even newer) versions of certain Python libraries or other dependencies, breaking compatibility.

Its fully working on WSL2 Ubuntu 20.04 (Windows Subsystem for Linux) however issues with larger captures i.g 180gb+ may require expanding the default [virtual disk size](https://docs.microsoft.com/en-us/windows/wsl/vhd-size)

It also partially runs on Windows natively; currently, ld-tools have been ported over.

Other dependencies include Python 3.5+,numpy, scipy, cython, numba, pandas, Qt5, Qmake, and FFmpeg.

Useful tools to note is StaxRip & Lossless Cut this gives you full encode-deinterlacing and basic editing to quickly handle uncompressed files.

# Installation and running the software on Ubuntu/Debian

Install all dependencies required by LD-Decode and VHS-Decode:

    sudo apt install clang libfann-dev python3-setuptools python3-numpy python3-scipy python3-matplotlib git qt5-default libqwt-qt5-dev libfftw3-dev python3-tk python3-pandas python3-numba libavformat-dev libavcodec-dev libavutil-dev ffmpeg openssl pv python3-distutils make ocl-icd-opencl-dev mono-runtime cython3

Install all dependencies required for gooey graphical user interface:

    sudo apt-get install build-essential dpkg-dev freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev
    libgstreamer-plugins-base1.0-dev libgtk-3-dev libjpeg-dev libnotify-dev
    libpng-dev libsdl2-dev libsm-dev libtiff-dev libwebkit2gtk-4.0-dev libxtst-dev
    python3.9-dev libpython3.9-dev pip3 install Gooey

Download VHS-Decode:

    git clone https://github.com/oyvindln/vhs-decode.git vhs-decode

Compile and install Domesday tools:

    cd vhs-decode && make -j8 && sudo make install && make clean

To update do `git pull` while inside of the vhs-decode directory.

# Usage

Note with WSL2 & Ubuntu, `./` in front of applications and scripts may be needed to run them.

## CX Card Setup & Capture

See the readme file for [CXADC](https://github.com/happycube/cxadc-linux3#readme) for more information on how to configure the driver and what each mode and capture rate means.

To see a live preview of tape signal being received by CXADC card from video head tracked signal will be unstable or wobbly if settings are not the same you may only see signal flash if in 16-bit modes for example, this is quite useful if you don't own an CRT with H/V shifting as it will allow you to inspect the full area for alignment/tracking issues.

PAL framing for 35.8 MHz/8-bit mode:

    ffplay -hide_banner -async 1 -f rawvideo -pix_fmt gray8 -video_size 2291x625 -i /dev/cxadc0 -vf scale=1135x625,eq=gamma=0.5:contrast=1.5

NTSC framing for 35.8 MHz/8-bit mode:

    ffplay -hide_banner -async 1 -f rawvideo -pix_fmt gray8 -video_size 2275x525 -i /dev/cxadc0 -vf scale=910x525,eq=gamma=0.5:contrast=1.5

Capture 30 seconds of tape signal using CXADC driver 8-bit samples

    timeout 30s cat /dev/cxadc0 > <capture>.u8

For 16-bit Simply change the end extension to .u16 & for flac captures re-name the end to your desired tape format like .VHS etc

It is recommended to use fast storage of 40MB/s+ write ability to avoid dropped samples.

## Decoding

To use VHS Decode GUI Run:

    ./vhs-decode-gui

Decode your captured tape to a .tbc by using:

    vhs-decode [arguments] <capture file> <output name>

Use analyse tool to inspect decoded tape:

    ld-analyse <output name>.tbc

## Compression and Decompression

Editable flags are `--bps` ie 16/8 bit and `% --ogg` and change `<capture>` to your input & output file names.

Reduce size of captured CXADC data (by 40-60%):

    flac --best --sample-rate=48000 --sign=unsigned --channels=1 --endian=little --bps=8 % --ogg -f <capture>.u8 -o <capture>.vhs

Decompress FLAC compressed captures.

    flac -d --force-raw-format --sign=unsigned --endian=little <capture>.vhs -o <capture>.u16

## Generating Colour Video Files (TBC to Playable MKV)

VHS-Decode produces two timebase corrected 16-bit headerless files separated into chroma/luma video signals in the .tbc format plus .json and .log files, usable with the LD-Decode family of tools (ld-analyse, ld-process-vbi, and ld-process-vits), Notably VBI (Vertical Blanking Interval) data recovery software like [VHS-Teletext](https://github.com/ali1234/vhs-teletext/) (Europe Subtitles) and FFMPEG's read [VITC timecode](https://github.com/oyvindln/vhs-decode/wiki/VITC-&-Subtitles) & [EIA-608](https://github.com/amiaopensource/sccyou)(USA Closed Captioning), also tape-based [arcade games](https://vhs.thenvm.org/resources-research/).

Please Check the Wiki for the full [upto-date command list!](https://github.com/oyvindln/vhs-decode/wiki/Command-List)

*gen_chroma_vid.sh now automatically detects PAL/NTSC based on the .JSON legacy scripts still exsist*

`gen_chroma_vid.sh -h` (Lists Command Options)

To generate .mkv files viewable in most media players, simply use the scripts installed:

Command Examples:

    gen_chroma_vid.sh -v -s <skip number of frames> -l <number of frames long> -i <.tbc filename without .tbc extension>

The `-a` option can embed an audio file.

    gen_chroma_vid.sh -v -s <skip n frames> -l <n frames long> -a <capture>.flac -i <.tbc filename without .tbc extension>

So for example open terminal in the directory of target TBC/Metadata files and run

    gen_chroma_vid.sh -v -s <skip n frames> -l <number of frames long> -a <capture>.flac -i <.tbc filename without .tbc extension>

This will use decoded .tbc files, generate a lossless, interlaced and high-bitrate (roughly 100-150 Mb/s) FFV1 codec video which,
although ideal for archival and reducing further loss in quality however this may be unsuitable for sharing online.

An additional processing mode is included in the script files, but commented out.

## Terminal Arguments

Please Check the Wiki for the full [upto-date command list!](https://github.com/oyvindln/vhs-decode/wiki/Command-List)

VHS-Decode supports various arguments to process differently captured tape recordings and different tape formats/systems.
These tend to change frequently as new features are added or superseded.

The below commands changes the sample rate and bit depth for the decoder.

By default, this is set to 40 MHz (the sample rate used by the Domesday Duplicator) at 16 bits.

## CXADC Specific Sample Rate Commands:

`--cxadc`    28.6 MHz/8-bit  (8fsc)

`--cxadc3`   35.8 MHz/8-bit  (10fsc)

`--10cxadc`  14.3 MHz/16-bit (4fsc)

`--10cxadc3` 17.9 MHz/16-bit (5fsc)

## Manual Configuration Commands

`-f`  Adjusts sampling frequency in integer units.

Example's `-f 280000hz` or `-f 28mhz` or `-f 8fsc`

`-tf` Sets Tape Format enter `VHS` `SVHS` or `UMATIC` etc. (Default is VHS)

## Colour System Commands

Changes the colour system to NTSC, PAL, PAL-M, or NTSC-J, respectively.
Please note that, as of writing, support for PAL-M is **experimental**.

`-n` = NTSC

`-p` = PAL

`-pm` = PAL-M

`--NTSCJ` = NTSC-J

# Time & Location Control

These commands are used for jumping ahead in a file or defining limits.
Useful for recovering decoding after a crash, or by limiting process time by producing shorter samples.

`-s`  Jumps ahead to any given frame in the capture.

`--start_fileloc` Jumps ahead to any given *sample* in the capture.

`-l` Limits decode length to *n* frames.

`-t` Defines the number of processing threads to use during demodulation.

(note: upon crashing, vhs-decode automatically dumps the last known sample location in the terminal output)

## Time Base Correction & Visuals Control

Please Check the Wiki for the full [upto-date command list!](https://github.com/oyvindln/vhs-decode/wiki/Command-List)

`--debug` sets logger verbosity level to *debug*. Useful for debugging and better log information. (Recommended To Enable for Archival)

`-ct` enables a *chroma trap*, a filter intended to reduce chroma interference on the main luma signal. Use if seeing banding or checkerboarding on the main luma .tbc in ld-analyse.

`-sl` defines the output *sharpness level*, as an integer from 0-100, the default being 0. Higher values are better suited for plain, flat images i.e. cartoons and animated material, as strong ghosting can occur (akin to cranking up the sharpness on any regular TV set.)

`--recheck_phase` Re-check chroma phase on every field. (No effect on U-matic)

`-dp demodblock` Displays Raw Demodulated Frequency Spectrum Graphs, makes a pop-up window per each thread so -t 32 would give you 32 GUI windows etc

# [Supported formats](https://github.com/oyvindln/vhs-decode/wiki/Tape-Support-List)

## Tapes:

(S-)VHS 625-line and 525-line, PAL, PAL-M and NTSC.

U-Matic 625-line and 525-line Low Band, PAL and NTSC.

BetaMax PAL **Preliminary Support Added**

Video8 & High8 PAL & NTSC **Preliminary Support Added**

## Input file formats:

.ldf/.lds (Domesday Duplicator FLAC-compressed and uncompressed data).

.r8/.u8   (CXADC 8-bit raw data).

.r16/.u16 (CXADC 16-bit raw data).

.flac/.cvbs/.vhs/.svhs/.bcam/.bmax/.vid8/.hi8 (FLAC-compressed captures, can be either 8-bit or 16-bit).

## Output file formats:

Unlike LD-Decode, VHS-Decode does not output its timebase-corrected frames as a single .tbc file

Both the luminance and chrominance channels are separate data files essentially an digital "S-Video", additionally useful for troubleshooting descriptor/log files are generated so you end up with 4 files in the following naming.

filename.tbc        - Luminance Image Data

filename_chroma.tbc - Chrominance Image Data

filename.tbc.json   - Frame Descriptor Table (Resolution/Dropouts/SNR/Frames/VBI Timecode)

filename.log        - Timecode Indexed Action/Output Log

# Join us

[Discord](https://discord.gg/pVVrrxd)

[Facebook](https://www.facebook.com/groups/2070493199906024)

[VideoHelp Forum](https://forum.videohelp.com/threads/394168-Current-status-of-ld-decode-vhs-decode-(true-backup-of-RF-signals)#post2558660)

# More Documentation

[VHS-Decode Wiki](https://github.com/oyvindln/vhs-decode/wiki)

[Google Doc Documentation](https://docs.google.com/document/d/1ZzR3gbW6iSVSNP0qoDIS0ExeRecKehlTQ0EJyx2g568/edit?usp=sharing)

## *If in doubt - Feel free to read the docs/wiki again if its not there then ask!*
