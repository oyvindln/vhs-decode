![vhs-decode logo](docs/vhs-decode_logo_256px.png)

# VHS-Decode (It does more than VHS now!)

A fork of [LD-Decode](https://github.com/happycube/ld-decode), the decoding software powering the [Domesday86 Project](https://www.domesday86.com/).  
This version has been modified to work with the differences found in the tracked RF drum head signals taken directly from colour-under format videotapes.

(Not to be confused with the TV Modulator/Demodulator pack or the **"antenna connectors"** on the back of the VCR!).

![](assets/images/DdD-EBU-Colour-Bar-PAL-VHS-SP.png)

SMPTE ColourBars (16:9) Test Tape With [WSS](https://en.wikipedia.org/wiki/Widescreen_signaling) (PAL) export exported full-frame (1112 x 624)

# [Supported Tape Formats](https://github.com/oyvindln/vhs-decode/wiki/Tape-Support-List)

**(S-)VHS** 625-line and 525-line - NTSC, NTSC-J, PAL and PAL-M **Fully Supported**

**U-Matic** 625-line and 525-line Low Band, PAL and NTSC. - **Fully Supported**

**Betamax** 625-line and 525-line, PAL & NTSC - **Suppported**

**Video8 & High8** 625-line and 525-line, PAL & NTSC - **Working (Luma Only)**

# [FAQ - Frequently Asked Questions](https://github.com/oyvindln/vhs-decode/wiki/FAQ)

Example Videos: [VHS-Decode](https://odysee.com/@vhs-decode:7) / [The Rewinding](https://odysee.com/@therewinding:4?view=content) - Odysee

Example Workflow [Flowcharts and Overview Graphics](https://github.com/oyvindln/vhs-decode/wiki/Diagram-Visuals)

The frequently asked questions page and the [Wiki](https://github.com/oyvindln/vhs-decode/wiki), will help break things down and explain the real world benifits of direct RF capture preservation and software decoding compared to conventional high-cost hardware based workflows, so if you have just found this project welcome to the affordable future of tape media preservation.

# [CVBS-Decode - Composite Video Decoder](https://github.com/oyvindln/vhs-decode/wiki/CVBS-Composite-Decode)

This repository also contains an **experimental** CVBS decoder, `cvbs-decode`, which shares code with ld-decode and vhs-decode. Capable of decoding basic raw digitized NTSC and PAL composite video, including colour if the source is somewhat stable. 

RAW CVBS has been captured using CX Cards & CXADC, however, this is somewhat buggy as the cx chip can decide to resample or do other things if it detects a valid video signal normally this happens when an unknown internal gain threshold is reached.

Test samples & signals can be generated using [HackTV](https://github.com/fsphil/hacktv)

Note for test media generation AJA/Magewell/Blackmagic and even some consumer digital to analogue converters have test generators built-in some prosumer/broadcast decks also have generators built in same for HDV camcorders.

# [HiFi-Decode](https://github.com/oyvindln/vhs-decode/wiki/003-Audio#hifi-decode-hifi-rf-into-audio-installation-and-usage)

Thanks to VideoMem's work on [Superheterodyne Decoding Tools](https://github.com/VideoMem/Superheterodyne-decoding-tools) we now have a working [HiFi Audio Decoder](https://github.com/oyvindln/vhs-decode/wiki/003-Audio#hifi-decode-hifi-rf-into-audio-installation-and-usage) witch provides decoding for VHS & Video8/High8 HiFi FM tracks which takes uncompressed or flac compressed RF captures of HiFi FM signals and outputs standard 24-bit 192khz FLAC or PCM (.wav) stereo files.

# LD-Tools suite for Windows

The ld-tools suit has been ported to windows, This allows the use of ld-analyse to view TBC files and ld-lds-converter to convert and compress DdD captures, please see the wiki for more information [Windows Tools Builds](https://github.com/oyvindln/vhs-decode/releases) **(Note this is only the tools, not the decoders)**

# Dependencies - Hardware

## A Working Tape Player (VCR/VTR etc)

Preferably adjusted per tape and in excellent mechanical and head condition, prosumer metal track decks are preferable as they were built generally better in terms of mechnical stability than cheaper later consumer decks that use more plastics, the only **crtical requirement** is avalible test points or a head amplifyer that is easy to tap into this goes for any and all tape formats.

**Note** SVHS tapes can be RF captured on standard VHS HiFi decks.

**Note** SVHS NTSC Decks - Currently inflated, but you can import PAL decks with NTSC support for 1/3rd the price though this only applys if conventional refrance SVHS captures are required.

**Always clean your tape track/drum/heads** before and afterwards with with 99.9% Isopropanol and lint free cloths/pads/paper this ensures less dropouts from dirty heads or tracks including the track of the head drum.

Its good practice to not cross contaminate tapes especially if dealing with mouldy or contaminated tapes.  

It also helps to make sure to re-lubricate metal and plastic moving joints cogs and bearings with appropriate grease's and oils to avoid mechanical failures.
## A RF Capture Device

Currently standardised to 2 methods but RF capture is not limited to these methods.

![DdD & CX Card](https://github.com/oyvindln/vhs-decode/wiki/assets/images/Hardware/DdD-Rev3-CX-White-Sony-ILCE-7RM3-2022.12.20-21.54.48.png)

## [Domesday Duplicator (DdD)](https://github.com/harrypm/DomesdayDuplicator#readme) (Method 01 - 300-350USD*)

Capture is done using an simple GUI application.

[Linux Application](https://github.com/harrypm/DomesdayDuplicator#readme)

[Windows Application](https://github.com/TokugawaHeavyIndustries/DomesdayDuplicator-WinBuild/releases/)

Originally geared towards capturing RF from laserdiscs players, it does however also work perfectly well for digitizing Tape RF data. It consists of a custom analogue to digital board with an amplifier, an off-shelf DE0-NANO FPGA development board, and a Cypress FX3 SuperSpeed Explorer USB 3.0 board.

**Note** Dont use USB storage or video devices on the same USB bus as the DdD, it will crash the capture!

[How To Aquire?](https://docs.google.com/document/d/1liOpdM6z51AwM5jGLoUak6h9aJ0eY0fQduE5k4TcleU/edit?usp=sharing) / [How to Fabricate & Flash?](https://docs.google.com/document/d/1k2bPPwHPoG7xXpS1NCYEe3w_jD_ts0yRCp-2aQ6QoKY/edit?usp=sharing) / [More Information](https://www.domesday86.com/?page_id=978)

## [CX Card & CXADC](https://github.com/oyvindln/vhs-decode/wiki/CX-Cards) (Method 02 - 20-35USD)

Capture & Config uses simple command-line arguments and parameters [CXADC](https://github.com/happycube/cxadc-linux3)

The budget approach is using a video capture card based on a Conexant CX23880/1/2/3 PCI chipset. With a modified Linux driver, these cards can be forced to output RAW data that can be captured to file, instead of decoding video normally as they otherwise would.

While you can use any generic card with the correct chips, today we recommend the ‘’New’’ Chinese variants that can be found on AliExpress that have integrated Asmedia or ITE 1x PCIE bridge chips allowing modern systems to use them, and consistent performance.

[Ware to Buy? & More Info](https://github.com/oyvindln/vhs-decode/wiki/CX-Cards)

Sadly however at stock without any modifications CX Cards proform slightly worse then a DomesdayDuplicator with about a 3db signal to noise difference, and its internal digital gain can affect captures drastically.

Currently the CX23883-39 based white variant cards have been consistently lower noise, these have also been easily upgraded to 40msps 8-bit with a drop-in crystal upgrade for use with formats like LaserDisc but at stock with external amplification they are quite optimal for VHS & HiFi, but higher bandwidth formats like SVHS & ED Beta can benifit from crystal upgrades.

# Deployment of Capture Hardware

Please read [VCR Reports](https://github.com/oyvindln/vhs-decode/wiki/VCR-reports) / [The Tap List](https://github.com/oyvindln/vhs-decode/wiki/004-The-Tap-List) / [Hardware Installation Guide](https://github.com/oyvindln/vhs-decode/wiki/Hardware-Installation-Guide)

Information on various VCRs that have been documented alongside high resolution pictures of VCR's that have had RF Taps installed, guidance on recommneded cables/connectors & tools to use are also included.

The setup process for RF capture involves running a short cable internally from points that provide the signal to a BNC jack at back of a metal/plastic VCR chassis or threaded out a vent, this allows direct access to the FM RF signals conveniently & reliably, we call this a Tap Point or RF Tap.

Head Drum -> Amplification & Tracking -> FM RF Test/Signal Points -> RF Capture -> Software Decoding -> Lossless TBC Files -> Audio/Video Files

## Find Test Points

#### Video FM Signal:

RF C, RF Y+C, PB, V RF, V ENV, ENV, ENVELOPE, VIDEO ENVE, VIDEO ENVELOPE

(Normally accompanied by a Composite test point useful if your VCR only has a SCART connection you can add another BNC for video)

#### HiFi Audio FM Signal:

RF-Out, A-RF, HIFI RF, HiFi ENV, A ENV

## Parts for RF Tap

[What tools do I need?](https://github.com/oyvindln/vhs-decode/wiki/Hardware-Installation-Guide)

* 50ohm BNC connectors, normally a [premade bulkhead](https://www.aliexpress.com/item/4000981154513.html), or [solderable thread mounted](https://www.aliexpress.com/item/4000639816847.html).
* 50-100cm of [RG316](https://www.aliexpress.com/item/32726071013.html) or [RG178](https://www.aliexpress.com/item/32879341402.html) 50 ohm coaxial cable.
* 10uf Capacitors [standard assorment](https://www.aliexpress.com/item/1005003276169319.html?) or audio grade like Nichicon if you like.

Connection Cables

* [Direct BNC to BNC](https://www.aliexpress.com/item/32530974771.html?) 
* [50Ohm BNC to BNC Cable](https://www.aliexpress.com/item/1005004558208116.html?)
* [S-Video to BNC](https://www.aliexpress.com/item/1005003333398231.html)

On CX White Cards you use the S-Video Luma pin for the RF input, but a BNC can be easily added.

## Install A RF Tap

If you want to still use live playblack or are using lower signal decks adding an 10uf (3.3uf to 100uf range works fine as well) capacitor to the test point or amplifyer can stop dropouts and improve signal level avoiding dropouts.

Center is Signal, Outer is Ground this goes for jacks and for coaxial cable.

The Negitive leg (shorter) goes on test/signal point, Positive leg (longer) on cable to connector/probe

While type and voltage does not matter much its best to use new/tested capacitors.
## Notes:

**Note** We use Aliexpress links for wide as possible avaliblity, but local venders are a thing.

**Note** With some Sony decks you can use Dupont connectors on the test point pins making an easy RF tap.

**Note**  Do not make sharp bends in any RF cabling, keep total cable runs as short as possible Ideally 30-60cm, more cable = more signal loss.

**Note** Some UMATIC decks have a RF output on the back this only provides Luma RF for dropout detection (Not the full signal for RF capture)

# Installation and running the software on Ubuntu/Debian

By default, the main VHS-Decode script allocates only one thread, though the gen_chroma_vid scripts allocate two threads.

The `make` rule of thumb of "number of logical processors, plus one" generally applies here,
though it mainly depends on the amount of memory available to the decoder.

Install all dependencies required by LD-Decode and VHS-Decode:

    sudo apt install clang libfann-dev python3-setuptools python3-numpy python3-scipy python3-matplotlib git qt5-default libqwt-qt5-dev libfftw3-dev python3-tk python3-pandas python3-numba libavformat-dev libavcodec-dev libavutil-dev ffmpeg openssl pv python3-distutils make cython3 cmake

For Ubuntu 22.04 that is:

    sudo apt install clang libfann-dev python3-setuptools python3-numpy python3-scipy python3-matplotlib git qt5-qmake qtbase5-dev libqwt-qt5-dev libfftw3-dev python3-tk python3-pandas python3-numba libavformat-dev libavcodec-dev libavutil-dev ffmpeg openssl pv python3-distutils pkg-config make cython3 cmake

## For optional GPU (Nvidia Cuda) FLAC compression support:

Install dependencies for GPU FLAC compression support:

    sudo apt install make ocl-icd-opencl-dev mono-runtime

## For optional gooey graphical user interface 

If you want to try the gooey gui wrapper for vhs-decode (`vhs-decode-gui`) :

    sudo apt-get install build-essential dpkg-dev freeglut3-dev libgl1-mesa-dev libglu1-mesa-dev libgstreamer-plugins-base1.0-dev libgtk-3-dev libjpeg-dev libnotify-dev libpng-dev libsdl2-dev libsm-dev libtiff-dev libwebkit2gtk-4.0-dev libxtst-dev python3.9-dev libpython3.9-dev

Then install gooey

    pip3 install Gooey

## Building

Download VHS-Decode:

    git clone https://github.com/oyvindln/vhs-decode.git vhs-decode

Install VHS-Decode:

    cd vhs-decode

    sudo ./setup.py install

Compile and Install ld-tools suite: (Required)

    mkdir build2
    cd build2
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_QT_VERSION=5
    make -j4
    sudo make install

Go back to the main directory with 

    cd .. 

To update do `git pull` while inside of the vhs-decode directory.

**Note:** debian/ubuntu does not have a qt6 version of qwt in repositories as of yet so you have to inform the build script to use Qt5 if both qt5 and qt6 are installed with -DUSE_QT_VERSION=5 as it might otherwise try to compile with qt6 instead and failing to locate qwt. The option is otherwise not needed.

# Usage

Note with WSL2 & Ubuntu, `./` in front of applications and scripts may be needed to run them.

Use `cd vhs-decode` to enter into the directory to run commands, `cd..` to go back a directory.

Use <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop the current process.

You dont actaully type `<` and `>` on your input & output files.


## Basic CX Card Setup & RF Capture

[CXADC Readme](https://github.com/happycube/cxadc-linux3#readme) for information on how to install & configure the driver this also goes into depth on modes.

[Commands for real-time FLAC capture on CX Cards](https://github.com/happycube/cxadc-linux3/wiki/FLAC-Compression-Guide)

[Naming Guide For RF Captures!](https://github.com/oyvindln/vhs-decode/wiki/Capture-Naming-Guide)

* Install CX Card & Install CXADC driver
* Find Input
* Check Gain
* Capture

To see if you have a connection, use the live preview mode and then hook up your RF cable, normally you will see a white flash as a signal if not change your vmux or input within a 0-2 range with the below command.

    sudo echo 0 >/sys/module/cxadc/parameters/vmux

To see a live preview of tape signal being received by a CXADC card, note that the video head tracked signal will be unstable or wobbly if settings are not the same; you may only see "signal flash" if in 16-bit mode for example.

This is quite useful if you don't own a CRT with Horizontal/Vertical shifting, as it will allow you to inspect the full area for alignment and/or tracking issues.

PAL framing for the default 28.64 MHz/8-bit mode:

    ffplay -hide_banner -async 1 -f rawvideo -pix_fmt gray8 -video_size 1832x625 -i /dev/cxadc0 -vf scale=1135x625,eq=gamma=0.5:contrast=1.5

NTSC framing for 28.64 MHz/8-bit mode:

    ffplay -hide_banner -async 1 -f rawvideo -pix_fmt gray8 -video_size 1820x525 -i /dev/cxadc0 -vf scale=910x525,eq=gamma=0.5:contrast=1.5

Capture 30 seconds of tape signal using CXADC driver 8-bit samples

    timeout 30s cat /dev/cxadc0 > <capture>_CXADC.u8

For 16-bit, simply change the output filename extension to `.u16`

For flac captures, set the output filename extension to your desired tape format, for example `.VHS`

It is recommended to use a fast storage device with 40-100 MB/s or faster write capacity, in order to avoid dropped samples, ideally an decicated SSD (via M.2/Sata not USB) in the exFat format.

## What should a FM video signal look like on the preview?

![CX-FM-Signal-Live](https://github.com/oyvindln/vhs-decode/wiki/assets/images/Signal-Examples/CXADC-Modulated-Signal-Preview.png)

# Dependencies & Installation - Software

VHS-Decode, as with LD-Decode, has been developed and tested on machines running the latest version of Ubuntu and Linux Mint.
Other distributions might have outdated (or even newer) versions of certain Python libraries or other dependencies, breaking compatibility.

There is a [Linux compatability doc](https://docs.google.com/document/d/132ycIMMNvdKvrNZSzbckXVEPQVLTnH_YX0Oh3lqtkkQ/edit) for various tested distributions of Linux.

Its fully working on WSL2 Ubuntu 20.04 (Windows Subsystem for Linux) however issues with larger captures i.g 180gb+ may require expanding the default [virtual disk size](https://docs.microsoft.com/en-us/windows/wsl/vhd-size)

It also partially runs on Windows natively; currently, [ld-tools](https://github.com/oyvindln/vhs-decode/releases) have been ported over.

Other dependencies include Python 3.5+, numpy, scipy, cython, numba, pandas, Qt5, Cmake, and FFmpeg.

Some useful free tools to note for post processing are
[StaxRip](https://github.com/staxrip/staxrip) & [Lossless Cut](https://github.com/mifi/lossless-cut) & of course [DaVinci Resolve](https://www.blackmagicdesign.com/uk/products/davinciresolve) this gives you basic editing to quickly handle uncompressed files cross operating systems, and for windows users an easy FFMPEG/AviSynth/Vapoursynth encoding and QTGMC de-interlacing experience, and full colour grading and post production ability.
## Decoding RF Captures

To use VHS Decode GUI Run:

    ./vhs-decode-gui

Decode your captured tape to a .tbc by using:

    vhs-decode [arguments] <capture file> <output name>

Basic Usage Example:

    ./vhs-decode --debug --pal --threads 8 --tape_format VHS --cxadc CX-White-2022.10.25.u8 my-first-decode-2022.10.25

After decoding process your tapes VBI data with:

    ld-process-vbi <decoded tape name>.tbc

Use analyse tool to inspect decoded tape:

    ld-analyse <decoded tape name>.tbc

## Compression and Decompression

For DomesDayDuplicator Captures simply run

`ld-compress <capture>`

Your .lds file will be compressed to an FLAC OGG .ldf file

For CXADC and other 8/16bit captures use the following:

[Copy Paste FLAC Compression Commands for CXADC](https://github.com/happycube/cxadc-linux3/wiki/FLAC-Compression-Guide#post-capture-flac-compression)

Editable flags are

The `--bps` flag can be changed to `--bps=8` or `--bps=16` for 8 & 16 bit captures and `--ogg` is optional.

Change `<capture>` & `<output-name>` to your input & output file name.

Reduce size of captured CXADC data (by 40-60%):

    flac --best --sample-rate=28636 --sign=unsigned --channels=1 --endian=little --bps=8 --ogg -f <capture>.u8 <output-name>

Output will be `filename.ogg` so rename the end extension to `.vhs` / `.hifi` etc.  

Decompress FLAC compressed captures:

    flac -d --force-raw-format --sign=unsigned --endian=little <capture>.vhs <capture>.u16

## Generating Colour Video Files (.TBC to Playable .MKV)

VHS-Decode produces two timebase corrected 16-bit `GREY16` headerless files separated into chroma/luma composite video signals in the `.tbc` format alongside `.json` and `.log` files, usable with the LD-Decode family of tools ld-analyse, ld-process-vbi, ld-process-vits and ld-dropout-correct.

The gen chroma scrips will use decoded .tbc files and generate standard video files by default a lossless, interlaced top field first and high-bitrate (roughly 70-100 Mb/s) FFV1 codec video which, which although ideal for archival and further processing, not supported in big NLEs.

*gen_chroma_vid.sh automatically detects PAL/NTSC based on the .JSON, although legacy scripts still exsist*

To generate .mkv files viewable in most media players, simply use the gen_chroma_vid.sh scripts below.
### Export your TBC files to a video file with

    ./gen_chroma_vid.sh Input-TBC-Name

## Editing & Basic Online Usage

For editing due to lack of support of FFV1 and sharing online without de-interlacing is not supported properly, as such the two commands are provided below to make suitable files for this use.

Both commands will automatically use the last file generated as the input.

For editors this transcodes an FFV1/V210 output to a "*near complient*" interlaced ProRes HQ file:
    
    ffmpeg -hide_banner -i "$1.mkv" -vf setfield=tff -flags +ilme+ildct -c:v prores -profile:v 3 -vendor apl0 -bits_per_mb 8000 -quant_mat hq -mbs_per_slice 8 -pixel_format yuv422p10lep -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 -c:a s24le -vf setdar=4/3,setfield=tff "$1_ProResHQ.mov"
    
For basic online sharing you can use this command to convert the FFV1 output to a de-interlaced lossy upscaled MP4:
    
    ffmpeg -hide_banner -i "$1.mkv" -vf scale=in_color_matrix=bt601:out_color_matrix=bt709:1440x1080,bwdif=1:0:0 -c:v libx264 -preset veryslow -b:v 15M -maxrate 15M -bufsize 8M -pixel_format yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -aspect 4:3 -c:a libopus -b:a 192k -strict -2 -movflags +faststart -y "$1_1440x1080_lossy.mp4"

Will just export the video with standard settings and the same input file name, the .tbc extention is not required.

## Time Control & Audio Muxing

Command Examples:

    ./gen_chroma_vid.sh -v -s <skip number of frames> -l <number of frames long> -i <.tbc filename without .tbc extension>

The `-a` option can embed an audio file, such as audio decoded via [HiFi Decode](https://github.com/VideoMem/ld-decode/tree/hifi-decode)

    ./gen_chroma_vid.sh -v -s <skip n frames> -l <n frames long> -a <capture>.flac -i <.tbc filename without .tbc extension>

So for example open terminal in the directory of target TBC/Metadata files and run

    ./gen_chroma_vid.sh -v -s <skip n frames> -l <number of frames long> -a <capture>.flac -i <.tbc filename without .tbc extension>

## VBI (Vertical Blanking Interval) Data Recovery

Software decoding provides the full signal frame, recovery software can be used to read and extract this information, however some information can be automatically extracted in the TBC file stage with `ld-processs-vbi` like VITC & Closed Captions. 

[VITC Timecode](https://github.com/oyvindln/vhs-decode/wiki/VITC-&-Subtitles) (Standard SMPTE Timecode)

[CC EIA-608](https://github.com/oyvindln/vhs-decode/wiki/NTSC-Closed-Captioning) (Closed Captioning)

[PAL Teletext](https://github.com/oyvindln/vhs-decode/wiki/PAL-Teletext-Subtitles) (European Subtitles & Information Graphics)

[Tape-based Arcade Games!](https://vhs.thenvm.org/resources/)

[Ruxpin TV Teddy](https://github.com/oyvindln/vhs-decode/blob/vhs_decode/tools/ruxpin-decode/readme.pdf) (Extra audio in visable frame)

### Generate an video output with the top VBI area:

### PAL

    ./gen_chroma_vid.sh --ffll 2 --lfll 308 --ffrl 2 --lfrl 620 <tbc-name>

### NTSC

    ./gen_chroma_vid.sh --ffll 1 --lfll 259 --ffrl 2 --lfrl 525 <tbc-name>
    
## Terminal Arguments

VHS-Decode supports various arguments to change how captured tape recordings are processed these vary slightly between formats like VHS & Umatic but the basic oprations remain the same.

The list below is a short list for common/daily usage but does not cover all the abbilitys and new or advanced command arguments possible so please read the [complete and upto-date command list](https://github.com/oyvindln/vhs-decode/wiki/Command-List) on the wiki as commands may change or be depricated so its always good to check this list after any updates.

## Sample Rate Comamnds

By default, this is set to 40 MHz (the sample rate used internally and by the Domesday Duplicator) at 16 bits.

The decoder is 8/16 bit agnostic so as long as sample rate is defined it will decode it.

`-f`  Adjusts sampling frequency in integer units.

Example's `-f 280000hz` or `-f 28mhz` or `-f 8fsc`

`-tf` Sets Tape Format enter `VHS`, `SVHS`, `UMATIC`, `BETAMAX`, `VIDEO8`, `HI8`  (Default is VHS)

##  CXADC Specific Sample Rate Commands

`--cxadc`    28.6 MHz/8-bit  (8fsc)

`--cxadc3`   35.8 MHz/8-bit  (10fsc) (Not recommended for capture due to up-sampling)

`--10cxadc`  14.3 MHz/16-bit (4fsc)

`--10cxadc3` 17.9 MHz/16-bit (5fsc)

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

`-t` Defines the number of processing threads to use during demodulation, decode cant use more then 6-8 threads per decode currently so using 8 threads is the practical limit as its mostly a single core task.

(note: upon crashing, vhs-decode automatically dumps the last known sample location in the terminal output)

## [Time Base Correction & Visuals Control](https://github.com/oyvindln/vhs-decode/wiki/Command-List#decode-tbc---time-base-correction-control)

`--debug` sets logger verbosity level to *debug*. Useful for debugging and better log information. (Recommended To Enable for Archival)

`--ct` enables a *chroma trap*, a filter intended to reduce chroma interference on the main luma signal. Use if seeing banding or checkerboarding on the main luma .tbc in ld-analyse.

`--recheck_phase` Re-check chroma phase on every field, fixes most colour issues. (No effect on U-matic)

`--sl` defines the output *sharpness level*, as an integer from 0-100, the default being 0. Higher values are better suited for plain, flat images i.e. cartoons and animated material, as strong ghosting can occur (akin to cranking up the sharpness on any regular TV set.)

`--dp demodblock` Displays Raw Demodulated Frequency Spectrum Graphs, makes a pop-up window per each thread so -t 32 would give you 32 GUI windows etc

## Input file formats:

**Note** Input for the decoders can be RAW uncompressed data or FLAC compressed data. 

**Note** .RAW will need to be renamed to s16/u16 

.ldf/.lds (40msps Domesday Duplicator FLAC-compressed and uncompressed data).

.r8/.u8   (CXADC 8-bit raw data).

.r16/.u16 (CXADC 16-bit raw data).

.flac/.cvbs/.vhs/.svhs/.betacam/.betamax/.video8/.hi8 (FLAC-compressed captures, can be either 8-bit or 16-bit).

## Output file formats:

Unlike CVBS-Decode & LD-Decode, VHS-Decode does not output its timebase-corrected frames as a single .tbc file.

Both the luminance and chrominance channels are separate data files essentially an digital "S-Video", additionally useful for troubleshooting descriptor/log files are generated so you end up with 4 files in the following naming.

`filename.tbc`        - Luminance Image Data

`filename_chroma.tbc` - Chrominance Image Data

`filename.tbc.json`   - Frame Descriptor Table (Resolution/Dropouts/SNR/Frames/VBI Timecode)

`filename.log`        - Timecode Indexed Action/Output Log

# Join us!

[Discord](https://discord.gg/pVVrrxd)

[Reddit](https://www.reddit.com/r/vhsdecode/)

[VideoHelp Forum](https://forum.videohelp.com/threads/394168-Current-status-of-ld-decode-vhs-decode-(true-backup-of-RF-signals)#post2558660)

[Facebook](https://www.facebook.com/groups/2070493199906024)

# More Documentation

[VHS-Decode Wiki](https://github.com/oyvindln/vhs-decode/wiki)

[Google Doc Documentation](https://docs.google.com/document/d/1ZzR3gbW6iSVSNP0qoDIS0ExeRecKehlTQ0EJyx2g568/edit?usp=sharing)

## *If in doubt - Feel free to read the docs/wiki again if its not there then ask!*

For future documentation changes, speak with [Harry Munday](https://github.com/harrypm) (harry@opcomedia.com) or on Discord (TheRealHarrypm#1667)
