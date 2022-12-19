![vhs-decode logo](docs/vhs-decode_logo_256px.png)

# VHS-Decode (It does more than VHS now!)

A fork of [LD-Decode](https://github.com/happycube/ld-decode), the decoding software powering the [Domesday86 Project](https://www.domesday86.com/).  
This version has been modified to work with the differences found in the tracked RF drum head signals taken directly from videotapes.

(Not to be confused with the TV Modulator/Demodulator pack or the **"antenna connectors"** on the back of the VCR!).

![](assets/images/DdD-EBU-Colour-Bar-PAL-VHS-SP.png)

SMPTE ColourBars (16:9) Test Tape With [WSS](https://en.wikipedia.org/wiki/Widescreen_signaling) (PAL) export exported full-frame (1112 x 624)

# [Supported Tape Formats](https://github.com/oyvindln/vhs-decode/wiki/Tape-Support-List)

**(S-)VHS** 625-line and 525-line - NTSC, NTSC-J, PAL and PAL-M **Fully Supported**

**U-Matic** 625-line and 525-line Low Band, PAL and NTSC. - **Fully Supported**

**Betamax** 625-line and 525-line, PAL & NTSC - **Working**

**Video8 & High8** 625-line and 525-line, PAL & NTSC - **Working**

## [FAQ - Frequently Asked Questions](https://github.com/oyvindln/vhs-decode/wiki/FAQ)

## CVBS-Decode

The repository also contains an **experimental** CVBS decoder, `cvbs-decode`, which shares code with ld-decode and vhs-decode. Capable of decoding basic raw digitized NTSC and PAL composite video, including colour if the source is somewhat stable. Samples can be captured using cxadc, however, this is somewhat buggy as the cx chip can decide to resample or do other things if it detects a valid video signal if the gain is too high.

Test samples & signals can be generated using [HackTV](https://github.com/fsphil/hacktv)

Note: For test media generation, AJA/Magewell/Blackmagic and even some consumer digital to analogue converters have test generators built-in. Some prosumer/broadcast decks also have generators built in; same for later era camcorders.

# HiFi-Decode

Functional but still a work in progress is VideoMem's [HiFi-Decode Branch](https://github.com/VideoMem/ld-decode/tree/hifi-decode) which takes uncompressed or FLAC compressed RF captures of HiFi audio, decodes and outputs standard 24bit 192khz FLAC or PCM (.wav) stereo files.

# Dependencies - Hardware

## A Tape Player (VCR/VTR etc)

Preferably adjusted per tape and in excellent mechanical and head condition, prosumer metal track decks are preferable as they were built generally better in terms of mechnical stability than cheaper later consumer decks that use more plastics. The only critical requirement is test points or a head amplifier that is easy to access. This goes for any and all tape formats.

**Note** SVHS tapes can be RF captured on standard VHS HiFi decks.

**Note** SVHS NTSC Decks - Pricing for these units may be inflated, thus it's possible you can import PAL decks with NTSC support for 1/3rd the price. However, this only applies if conventional reference SVHS captures are required.

Its good practice to not cross contaminate tapes, especially if dealing with mouldy or contaminated tapes.  Always clean your tape track/drum/heads before and afterwards with 99.9% Isopropyl Alcohol, and lint free cloths/pads/paper. This ensures less dropouts from dirty playback heads.

It also helps to make sure to re-lubricate metal and plastic moving joints, cogs, and bearings, with appropriate grease and oil products, to avoid mechanical failures.

## RF Capture Devices  

## [Domesday Duplicator (DdD)](https://github.com/harrypm/DomesdayDuplicator#readme) (Method 01 - 300-350USD*)

Capture is done using a simple GUI application.

[Linux Application](https://github.com/harrypm/DomesdayDuplicator#readme)

[Windows Application](https://github.com/TokugawaHeavyIndustries/DomesdayDuplicator-WinBuild/releases/)

Originally geared towards capturing RF from LaserDisc players, it does however also work perfectly well for digitizing Tape RF data. It consists of a custom analogue to digital board with an amplifier, an off-shelf DE0-NANO FPGA development board, and a Cypress FX3 SuperSpeed Explorer USB 3.0 board.

[How To Aquire?](https://docs.google.com/document/d/1liOpdM6z51AwM5jGLoUak6h9aJ0eY0fQduE5k4TcleU/edit?usp=sharing) / [How to Fabricate & Flash?](https://docs.google.com/document/d/1k2bPPwHPoG7xXpS1NCYEe3w_jD_ts0yRCp-2aQ6QoKY/edit?usp=sharing) / [More Information](https://www.domesday86.com/?page_id=978)

## [CX Card & CXADC](https://github.com/happycube/cxadc-linux3) (Method 02 - 20-35USD)

Capture & Configuration uses simple command-line arguments and parameters [CXADC](https://github.com/happycube/cxadc-linux3)

There is now a [CXADC Wiki](https://github.com/happycube/cxadc-linux3/wiki), explaining the best card models, and guides for card modifications, such as crystal upgrades.

The budget approach is using a video capture card based on a Conexant CX23880/1/2/3 PCI chipset. With a modified Linux driver, these cards can be forced to output RAW data that can be captured to a file, instead of decoding video normally as they otherwise would.

There are now ‘’New’’ Chinese variants that can be found on AliExpress that have integrated Asmedia or ITE 1x PCIE bridge chips allowing modern systems to use them.

The cards however, at stock without any modifications, have more self-noise compared to the Domesday Duplicator setup, with about a 3db signal to noise difference. Currently, in recent testing, the CX23883-39 based white PCB variant cards have consistently lower noise.

# Deployment of Capture Hardware

Please Read the [VHS-Decode Wiki](https://github.com/oyvindln/vhs-decode/wiki) for more in-depth information. It features examples, RF tap point locations, and setup photos for various VCR decks that have been [RF tap'ed and tested](https://github.com/oyvindln/vhs-decode/wiki/004-The-Tap-List).

1. Find your test points. Test points may be referred to as one of the following, if looking though a service manual or labels on the VCR boards:

Modulated Video:

RF C, RF Y+C, PB, V RF, V ENV, ENV, ENVELOPE, VIDEO ENVE, VIDEO ENVELOPE

(Normally accompanied by a Composite test point, useful if your VCR only has a SCART connection you can add another BNC for video)

HiFi Audio:

RF-Out, A-RF, HIFI RF

2. Decide on connection method: Soldered or Clipped/Clamped. (Soldered is recommended due to reliability)

For some Sony decks you can use Dupont connectors on the pins to save effort soldering or hooking on a probe.

For alligator clips, this is Red on signal/Black on ground; if you have bigger test points like on rackmount and later prosumer decks.

For direct soldering, RG178 or RG316 cable to an BNC bulk head is recommended. This allows a short, clean direct run to a fixed mounting point you can drill or melt/file out, or just thread the cable though a ventilation slit or other pre-existing open space.

3. Test to see if an 10uf capacitor is needed. If you want to see your tape during RF capture or are playing SVHS or other higher bandwidth tapes, you will need to add a 10uf capacitor in-line before your bulkhead/probe connection, to avoid signal dropout. (Applies to most VCRs but not all; example, many JVC units.)

**Note** Do not make sharp bends in any RF cabling.

**Note** With Coax cable (RGxxx etc), the centre stranded wire is signal, and outer shield wire is ground twist. The ground strands tight to make a solder-able connection, and ensure to use ample flux to flow solder correctly. Afterwards, you may wipe with 99.9% Isopropyl Alcohol to clean the flux off.

**Note** Some UMATIC decks have a direct RF output on the back that may be viable for RF capture (At the moment, this requires further testing and expansion on data/information)

**Note** If there is no info on your VCR in the wiki, it's recommended to acquire the service manual for your device; best to search online to try and find your models' service manual.
Often however, RF tap points are labelled clearly, and easily accessible; they are not hard to visually identify, and if the unit has been serviced, RF points are commonly marked with a Sharpie.

# Dependencies & Installation - Software

VHS-Decode, as with LD-Decode, has been developed and tested on machines running the latest version of Ubuntu and Linux Mint.
Other distributions might have outdated (or even newer) versions of certain Python libraries or other dependencies, breaking compatibility.

VHS-Decode is supported for use on WSL2 Ubuntu 20.04 (Windows Subsystem for Linux), however issues with larger captures, eg. 180gb+, may require expanding the default [virtual disk size](https://docs.microsoft.com/en-us/windows/wsl/vhd-size).

It also partially runs on Windows natively; currently, [ld-tools](https://github.com/oyvindln/vhs-decode/releases) have been ported over.

Other dependencies include Python 3.5+, numpy, scipy, cython, numba, pandas, Qt5, Cmake, and FFmpeg.

Some useful free tools to note for post processing are [StaxRip](https://github.com/staxrip/staxrip) & [Lossless Cut](https://github.com/mifi/lossless-cut) & of course [DaVinci Resolve](https://www.blackmagicdesign.com/uk/products/davinciresolve).
This grouping of tools gives you basic editing to quickly handle uncompressed files cross operating systems, and for Windows users an easy FFMPEG/AviSynth/Vapoursynth encoding and de-interlacing experience, as well as full colour grading and post production capabilities.

# Installation and running the software on Ubuntu/Debian

By default, the main VHS-Decode script allocates only one thread, though the gen_chroma_vid scripts allocate two threads.

The `make` rule of thumb of "number of logical processors, plus one" generally applies here,
though it mainly depends on the amount of memory available to the decoder.

Install all dependencies required by LD-Decode and VHS-Decode:

    sudo apt install clang libfann-dev python3-setuptools python3-numpy python3-scipy python3-matplotlib git qt5-default libqwt-qt5-dev libfftw3-dev python3-tk python3-pandas python3-numba libavformat-dev libavcodec-dev libavutil-dev ffmpeg openssl pv python3-distutils make cython3 cmake

For Ubuntu 22.04 that is:

    sudo apt install clang libfann-dev python3-setuptools python3-numpy python3-scipy python3-matplotlib git qt5-qmake qtbase5-dev libqwt-qt5-dev libfftw3-dev python3-tk python3-pandas python3-numba libavformat-dev libavcodec-dev libavutil-dev ffmpeg openssl pv python3-distutils pkg-config make cython3 cmake


Install dependencies for GPU FLAC compression support:

    sudo apt install make ocl-icd-opencl-dev mono-runtime

Install all dependencies required for optional gooey graphical user interface:

    sudo apt-get install build-essential dpkg-dev freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev libgstreamer-plugins-base1.0-dev libgtk-3-dev libjpeg-dev libnotify-dev libpng-dev libsdl2-dev libsm-dev libtiff-dev libwebkit2gtk-4.0-dev libxtst-dev python3.9-dev libpython3.9-dev

Then install gooey

    pip3 install gooey

Download VHS-Decode:

    git clone https://github.com/oyvindln/vhs-decode.git vhs-decode

Install VHS-Decode:

    cd vhs-decode

    sudo ./setup.py install

Compile and Install ld-tools suite: (Required)

    mkdir build
    cd build
    cmake ..
    make -j4
    sudo make install

To update do `git pull` while inside of the vhs-decode directory.

# Usage

Note with WSL2 & Ubuntu, `./` in front of applications and scripts may be needed to run them.

Use `cd vhs-decode` to enter into the directory to run commands, `cd ..` to go back a directory.

## ld-tools suite for Windows

The ld-tools suite has been ported to windows. This mainly allows the easy use of LD-Analyse to view .TBC files. 
Note that the Windows builds only includes the tools, and NOT the decoder programs.
Please see the wiki for more information [Windows Tools Builds](https://github.com/oyvindln/vhs-decode/releases)

## CX Card Setup & Capture

See the readme file for [CXADC](https://github.com/happycube/cxadc-linux3#readme) for more information on how to configure the driver, and what each mode and capture rate means.

To see a live preview of tape signal being received by a CXADC card, note that the video head tracked signal will be unstable or wobbly if settings are not the same; you may only see "signal flash" if in 16-bit mode for example.
This is quite useful if you don't own a CRT with Horizontal/Vertical shifting, as it will allow you to inspect the full area for alignment and/or tracking issues.

PAL framing for 35.8 MHz/8-bit mode:

    ffplay -hide_banner -async 1 -f rawvideo -pix_fmt gray8 -video_size 2291x625 -i /dev/cxadc0 -vf scale=1135x625,eq=gamma=0.5:contrast=1.5

NTSC framing for 35.8 MHz/8-bit mode:

    ffplay -hide_banner -async 1 -f rawvideo -pix_fmt gray8 -video_size 2275x525 -i /dev/cxadc0 -vf scale=910x525,eq=gamma=0.5:contrast=1.5

Capture 30 seconds of tape signal using CXADC driver 8-bit samples

    timeout 30s cat /dev/cxadc0 > <capture>.u8

For 16-bit, simply change the output filename extension to .u16.
For flac captures, set the output filename extension to your desired tape format, for example .VHS 

It is recommended to use a fast storage device with a minimum of 40MB/s+ write capacity, in order to avoid dropped samples.

## Decoding

To use VHS Decode GUI Run:

    ./vhs-decode-gui

Decode your captured tape to a .tbc by using:

    vhs-decode [arguments] <capture file> <output name>

Use the ld-analyse tool to inspect decoded tapes:

    ld-analyse <output name>.tbc

## Compression and Decompression

For DomesDayDuplicator Captures simply run

`ld-compress <capture>`

Your .lds file will be compressed to a FLAC OGG .ldf file.

For CXADC and other 8/16bit captures use the following:

Editable flags are

The `--bps` flag can be changed to `--bps=8` or `--bps=16` for 8 & 16 bit captures.

Change `<capture>` to your input file name.

Reduce size of captured CXADC data (by 40-60%):

    flac --best --sample-rate=48000 --sign=unsigned --channels=1 --endian=little --bps=8 --ogg -f <capture>.u8 <output-name>

Decompress FLAC compressed captures.

    flac -d --force-raw-format --sign=unsigned --endian=little <capture>.vhs <capture>.u16

Output will be `filename.ogg` so rename the end extension to .vhs / .hifi etc.  

## Generating Colour Video Files (.TBC to Playable .MKV)

VHS-Decode produces two timebase corrected 16-bit headerless files, separated into chroma/luma video signals in the .tbc format; plus .json and .log files, usable with the [LD-Decode family of tools](https://github.com/happycube/ld-decode/wiki).

To generate .mkv files viewable in most media players, simply use the scripts provided.

Command Examples:

    ./gen_chroma_vid.sh -v -s <skip number of frames> -l <number of frames long> -i <.tbc filename without .tbc extension>

The `-a` option can embed an audio file, such as audio decoded via [HiFi Decode](https://github.com/VideoMem/ld-decode/tree/hifi-decode)

    ./gen_chroma_vid.sh -v -s <skip n frames> -l <number of frames long> -a <capture>.flac -i <.tbc filename without .tbc extension>

This will use the decoded .tbc files and generate a lossless, interlaced and high-bitrate (roughly 70-100 Mb/s) FFV1 codec video which,
although ideal for archival and reducing further loss in quality, may be unsuitable for sharing online without de-interlacing and post-processing to a lossy video format (such as x264/x265).

An additional processing mode is included in the script files, but commented out.

## VBI (Vertical Blanking Interval) data recovery

VBI (Vertical Blanking Interval) data recovery software can also be used per specific region encodings, as follows:

[VHS-Teletext](https://github.com/oyvindln/vhs-decode/wiki/PAL-Teletext-Subtitles) (European Subtitles)

[FFMPEG EIA-608](https://github.com/oyvindln/vhs-decode/wiki/NTSC-Closed-Captioning) (North America Closed Captioning)

[FFMEPG Read VITC Timecode](https://github.com/oyvindln/vhs-decode/wiki/VITC-&-Subtitles) (Standard SMPTE Timecode)

[Tape-based Arcade Games](https://vhs.thenvm.org/resources/)

### To generate an video output with the top VBI area included, use the following command:

PAL:

    ./gen_chroma_vid.sh --ffll 2 --lfll 308 --ffrl 2 --lfrl 620 <capture>.tbc
      
NTSC:

    ./gen_chroma_vid.sh --ffll 2 --lfll 308 --ffrl 2 --lfrl 520 <capture>.tbc
    
Note: *gen_chroma_vid.sh now automatically detects PAL/NTSC based on the .JSON legacy scripts*

Please Check the Wiki for the complete [up-to-date command list!](https://github.com/oyvindln/vhs-decode/wiki/Command-List)

`./gen_chroma_vid.sh -h` (List Command Options)

`./gen_chroma_vid.sh Input-TBC-Name` (Will just export the video with standard settings and the same input file name)


## Terminal Arguments

#### Please Check the Wiki for the [up-to-date command list!](https://github.com/oyvindln/vhs-decode/wiki/Command-List)

VHS-Decode supports various arguments to process various captured tape recordings, and different tape formats/systems. These tend to change frequently as new features are added or superseded.

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

`-tf` Sets Tape Format enter `VHS`, `SVHS`, `UMATIC`, `BETAMAX`, `VIDEO8`, `HI8`  (Default is VHS)

## Colour System Commands

Changes the colour or TV system to NTSC, PAL, PAL-M, NTSC-J, or MESECAM respectively.
Please note that, as of writing, support for PAL-M is **experimental**.

`-n` = NTSC

`-p` = PAL

`--pm` = PAL-M

`--NTSCJ` = NTSC-J

`--MESECAM` = MESECAM

# [Time & Location Control](https://github.com/oyvindln/vhs-decode/wiki/Command-List#time--location-control)

These commands are used for jumping ahead in a file or defining limits.
Useful for recovering decoding after a crash, or by limiting process time by producing shorter samples.

`-s`  Jumps ahead to any given frame in the capture.

`--start_fileloc` Jumps ahead to any given *sample* in the capture.

`-l` Limits decode length to *n* frames.

`-t` Defines the number of processing threads to use during demodulation.

(note: upon crashing, vhs-decode automatically dumps the last known sample location in the terminal output)

## Time Base Correction & Visuals Control

`--debug` sets logger verbosity level to *debug*. Useful for debugging and better log information. (Recommended To Enable for Archival)

`--ct` enables a *chroma trap*, a filter intended to reduce chroma interference on the main luma signal. Use if seeing banding or checkerboarding on the main luma .tbc in ld-analyse.

`--recheck_phase` Re-check chroma phase on every field, fixes most colour issues. (No effect on U-matic)

`--sl` defines the output *sharpness level*, as an integer from 0-100, the default being 0. Higher values are better suited for plain, flat images i.e. cartoons and animated material, as strong ghosting can occur (akin to cranking up the sharpness on any regular TV set.)

`--dp demodblock` Displays Raw Demodulated Frequency Spectrum Graphs, makes a pop-up window per each thread so -t 32 would give you 32 GUI windows etc

## Input file formats:

.ldf/.lds (Domesday Duplicator FLAC-compressed and uncompressed data).

.r8/.u8   (CXADC 8-bit raw data).

.r16/.u16 (CXADC 16-bit raw data).

.flac/.cvbs/.vhs/.svhs/.betacam/.betamax/.video8/.hi8 (FLAC-compressed captures, can be either 8-bit or 16-bit).

## Output file formats:

Unlike LD-Decode, VHS-Decode does not output its timebase-corrected frames as a single .tbc file.

Both the luminance and chrominance channels are separate data files, essentially a digital "S-Video". 
Descriptor/log files are also generated, which are helpful in troubleshooting. 4 common files are normally generated:

filename.tbc        - Luminance Image Data

filename_chroma.tbc - Chrominance Image Data

filename.tbc.json   - Frame Descriptor Table (Resolution/Dropouts/SNR/Frames/VBI Timecode)

filename.log        - Timecode Indexed Action/Output Log

# Join us

[Discord](https://discord.gg/pVVrrxd)

[Facebook](https://www.facebook.com/groups/2070493199906024)

[VideoHelp Forum](https://forum.videohelp.com/threads/394168-Current-status-of-ld-decode-vhs-decode-(true-backup-of-RF-signals\)#post2558660)

# More Documentation

[VHS-Decode Wiki](https://github.com/oyvindln/vhs-decode/wiki)

[Google Doc Documentation](https://docs.google.com/document/d/1ZzR3gbW6iSVSNP0qoDIS0ExeRecKehlTQ0EJyx2g568/edit?usp=sharing)

## *If in doubt - feel free to read the docs/wiki again, or submit a question!*
