<img src="assets/icons/Cross-Platform-VHS-Decode-Trasparent.png" width="300" height="">


# VHS-Decode (It does more than VHS now!)


A part of the [FM RF Archival](https://github.com/oyvindln/vhs-decode/wiki/Signal-Sampling) method of media preservation.

VHS-Decode and HiFi-Decode handles FM RF signals captured directly from colour-under & composite FM modulated videotape formats, captured directly from virtually any working VCR/VTR's heads pre-amplification & tracking stage before any internal video/hifi processing.

------

[Download Decode](https://github.com/oyvindln/vhs-decode/releases) (Windows / MacOS / Linux)

<img src="assets/images/decode-launcher-Rev2.0-windows-2026.png" width="600" height="">


> [!CAUTION]
> FM RF Archival captures and capturing is not to be confused with the TV Modulator/Demodulator pack's signals, i.e the **"antenna connectors"** on the back of a VCR!


<img src="assets/images/ld-analyse-vhs-decode-svhs-&-scopes.png" width="" height="">

> EBU Colourbars (4:3) on SVHS decoded signal frame (1112 x 624), with scanline oscilloscope and vectorscope enabled in hybrid frame view mode on tbc-tools (v3.0.1).


# [Supported Tape Formats](https://github.com/oyvindln/vhs-decode/wiki/Tape-Support-List)


**VHS** 625-line and 525-line - NTSC, NTSC-J, PAL and PAL-M. **Generally well supported** (Video & HiFi)

**SVHS** 625-line and 525-line - NTSC, NTSC-J, PAL and PAL-M. **Supported**

**U-Matic Low Band** 625-line and 525-line - PAL and NTSC. **Supported**

**U-Matic High Band** 625-line - PAL. **Basic support**

**Betamax** 625-line and 525-line - PAL & NTSC. **Supported**

**SuperBeta** 525-line - NTSC. **Preliminary support** (PAL samples needed)

**Video8 & Hi8** 625-line and 525-line - PAL & NTSC. **Supported** (Video & HiFi)

**1" Type C (SMPTE Type C)** 625-line and 525-line - PAL & NTSC. **Basic support** (More Samples Required!) 

**1" Type B (SMPTE Type B)** 625-line and 525-line - PAL & NTSC. **Preliminary support** (More Samples Required!)

**EIAJ** 625-line - PAL. **Basic support** (NTSC Samples Required!) 

**Philips VCR** & **Philips VCR "LP"**  625-line - PAL. **Basic support**

**Phlips Video2000** 625-line - **Basic support** 

**2" Quad (QUADRUPLEX)** 405-line / 819-line / 625-line **Basic development**


# [FAQ - Frequently Asked Questions](https://github.com/oyvindln/vhs-decode/wiki/FAQ)


Example Videos: [VHS-Decode](https://odysee.com/@vhs-decode:7) / [The Rewinding](https://odysee.com/@therewinding:4?view=content) / [Video Dump](https://www.youtube.com/@videodumpchannel).

Example Workflow [Flowcharts and Overview Graphics](https://github.com/oyvindln/vhs-decode/wiki/Diagram-Visuals)

The frequently asked questions page and the [Wiki](https://github.com/oyvindln/vhs-decode/wiki), will help break things down and explain the real world benefits of direct RF capture preservation and software decoding compared to conventional high-cost hardware based workflows.

So if you have just found this project welcome to the affordable future of tape media preservation!


# [CVBS-Decode - Composite Video Decoder](https://github.com/oyvindln/vhs-decode/wiki/CVBS-Composite-Decode)


<img src="assets/images/ld-analyse_pal_philips_cvbs_chroma_frame.png"  width="600" height="">

> Philips Test Pattern with PAL 3D Transform decoder - 2025

This repository also contains an **experimental** CVBS decoder, `cvbs-decode`, which shares code with ld-decode and vhs-decode. Capable of decoding basic RAW digitized NTSC and PAL composite video, including colour if the source is somewhat stable. 

This primarily allows for users to leverage the powerful TBC code, VBI processing and Transform 2D and Transform 3D PAL chroma-decoders (comb filters) of the tbc-tools suite.

> [!CAUTION]
> - CVBS capture is not possible with the DdD due to input filtering on the hardware, but is possible with the [MISRC](https://www.misrc.org/) board and HSDAOH options.
> - CX Cards & CXADC, can work, however only at lowest gain states and or with external signal feed into it to stop its hardware decoder from triggering.


Test samples & signals can be generated using a [HackDAC](https://github.com/inaxeon/hacktv-hackrf) & [HackTV](https://github.com/fsphil/hacktv) or downloaded from [The Internet Archive](https://archive.org/details/wss-wide-screen-signaling).


# [HiFi-Decode](https://github.com/oyvindln/vhs-decode/wiki/003-Audio#hifi-decode-hifi-rf-into-audio-installation-and-usage) 


<img src="https://github.com/oyvindln/vhs-decode/wiki/assets/images/vhs-decode-gui/hifi-decode-gui-current-compact.png" width="400" height="">

Thanks to VideoMem's work on [Superheterodyne Decoding Tools](https://github.com/VideoMem/Superheterodyne-decoding-tools) we have [HiFi-Decode](https://github.com/oyvindln/vhs-decode/wiki/hifi-decode) which provides decoding support for (S)VHS & Video8/Hi8 HiFi FM tracks which takes uncompressed or FLAC compressed RF captures of HiFi FM signals and outputs standard 24-bit 44.1-192kHz FLAC stereo audio files. The decoded quality is close to and better in some cases than the hardware output from a VCR.

[RTLSDR capture & decoding](https://github.com/oyvindln/vhs-decode/wiki/RTLSDR) is cross platfrom as its 100% GNURadio based, this can run in realtime on most systems (1~3 sec delay) and provide live playback, Alongside 8msps RF files and a 48kHz 24-bit FLAC file of the decoded audio, useful for finding test points or verifying quickly if a signal is working.


# Dependencies - Hardware


There is 4 core parts to FM RF Archival

- The RF Tap
- Amplification & Impedance
- RF Capture Device
- Decoding

That is it for scope of functional things you need to have a grasp of before you can start archiving tapes, picking the right hardware to capture for the formats you are using is the most critical part.


## A Working Tape Player (VCR/VTR etc)


Preferably somewhat calibrated and in working mechanical and head condition. While in legacy world prosumer metal track decks are preferred as they were generally better built in terms of mechanical stability than cheaper later consumer decks using more plastics if treated well and serviced last much longer which are fair points...

However, the only **critical requirement** is available test points or a head amplifier that is easy to tap into, this goes for any and all tape formats. Since the rest of the circuitry is bypassed much of the difference between the VCR model lineup outside of head count/HiFi capability is also skipped and thus a good condition 90s HiFi VCR can give equally good results as a top of the line SVHS VCR.

> [!TIP]
> **S**VHS tapes can be RF captured on some later 90-2000s standard VHS HiFi decks. 

> [!TIP]
> Since we bypass the decoding circuitry, it is not required that the VCR supports the TV system for the tape to be decoded correctly. The VCR does however need to be able to play the tape at the right speed - so one would be able to decode a PAL-M tape playing back in a American market NTSC VCR but not a standard PAL recording as NTSC only VCRs will normally not play those at the correct speed.

> [!IMPORTANT]  
> - Please read the [Cleaning & Servicing Guide](https://github.com/oyvindln/vhs-decode/wiki/Cleaning-&-Servicing-Guide).
> - **Always clean your tape track/drum heads** before and afterwards with 99.9% isopropanol and lint free cloths/pads/paper. This ensures fewer dropouts from dirty heads or tracks including the track of the head drum.
> - Its good practice to avoid cross contamination of tapes, especially if dealing with mouldy or contaminated tapes.  
> - It also helps to make sure to re-lubricate metal and plastic moving joints cogs and bearings with appropriate greases and oils to avoid mechanical failures. 


## An RF Capture Device


Currently there are a couple of standardised hardware workflows, but.. 

[You need to read and select one based off what format(s) you are actually going to be capturing](https://github.com/oyvindln/vhs-decode/wiki/Workflow-Guide).

<img src="https://github.com/oyvindln/vhs-decode/wiki/assets/cxadc-clockgen-mod/MISRC_GUI_2026_Clockgen.png" width="600" height="">

> MISRC GUI (2026) capturing directly to FLAC for RF Video & HiFi + Baseband Audio via CX Cards + Clockgen Mod. 


## [MISRC](https://www.misrc.org/)


The MISRC is the tape focused replacement in both hardware and software for the legacy single channel DdD and for when CX Cards are no longer available by offering multiple channels of RF capture alongside audio capture capabilities natively for all platforms (even Android!) via USB 3.0.

The V1.5 can use the [the clockgen mod](https://github.com/oyvindln/vhs-decode/wiki/Clockgen-Mod) to gain audio support but the v2.5 has 4ch of audio intergrated both workflows are natively supported in the [MISRC GUI](https://github.com/harrypm/MISRC-GUI) which works across many devices for FM RF Archival capture. 

> [!IMPORTANT]  
> - This does not replace the need for an [ADA4857 Amp](https://github.com/oyvindln/vhs-decode/wiki/amplifyer-setup-guide).

[Where to Buy? & More Info](https://www.misrc.org/)


## [CX Cards & Clockgen Mod](https://github.com/oyvindln/vhs-decode/wiki/CX-Cards)

The most cost-effective approach (30-250USD) is using a video capture card based on a Conexant CX23880/1/2/3 PCI chipset called "CX Cards" using the CXADC driver.

With either the [Linux Driver](https://github.com/happycube/cxadc-linux3) or [Windows Driver](https://github.com/JuniorIsAJitterbug/cxadc-win) CX Cards can be easily used inside [MISRC GUI](github.com/harrypm/MISRC-GUI) today for direct to FLAC recording!

These drivers force compatible cards to output RAW PCM signal data that can be captured to a highly compressed FLAC file, instead of decoding CVBS video normally as they otherwise would.

While you can use any generic card with the correct chips, today we recommend the ‘‘New’’ Chinese variants that can be found on AliExpress that have integrated Asmedia or ITE 1x PCIE bridge chips allowing modern systems to use them, and consistent performance.

These cards combined with a [ADA4857 Amp](https://github.com/oyvindln/vhs-decode/wiki/amplifier-setup-guide) & [the clockgen mod](https://github.com/oyvindln/vhs-decode/wiki/Clockgen-Mod) allow users to have a refined RF tap regardless of format.

By syncuing up multiple cards from a common clock source, this enables Video RF + HiFi RF + Baseband (Baseband = Linear or deck decoded HiFi audio on RCA/XLR outputs) from VCR/VTRs to be captured in perfect hardware sync, a highly reliable turn-key "one run and done" capturing workflow for a wide range of videotape formats, allowing for automated audio alignment post-capture, saving countless hours.

[Where to Buy? & More Info](https://github.com/oyvindln/vhs-decode/wiki/CX-Cards)


# Dependencies & Installation - Hardware


> [!TIP]
> Please Read [Hardware Installation Guide](https://github.com/oyvindln/vhs-decode/wiki/Hardware-Installation-Guide) / [VCR Reports](https://github.com/oyvindln/vhs-decode/wiki/VCR-reports) / [The Tap List](https://github.com/oyvindln/vhs-decode/wiki/004-The-Tap-List)


````
VCR ==> Head Drum ==> RAW Signal From Heads ==> Amplification & Tracking IC ==> Tracked FM RF signals ==> Test Points.
````

````
Test/Signal Points FM RF ==> ADA4857 Amplifier ==> RF Capture ADC ==> FLAC RF & Audio Files.
`````

````
FLAC Files ==> Decoding ==> Lossless 4fsc TBC Files & Audio Files ==> Reframe & Adjust ==> Export YUV Conversion ==> Muxed Audio/Video files.
````

Information on various VCRs that have been documented alongside high resolution pictures of VCR's that have had RF taps installed, guidance on recommended cables/connectors & tools to use are also included.

The setup process for RF capture involves running a short cable internally from points that provide the unprocessed modulated or "FM" video and or audio signal signals.

This cable is then routed to an added BNC jack at back of your metal/plastic VCR chassis or cable threaded out a vent, this allows direct access to the FM RF signals conveniently & reliably, we call this a `Tap Point` or `RF Tap` respectively for some decks and camcorders however DuPont connectors (2.54mm headers) and ribbon jigs can be used, but can be less mechanically safe/secure in some setups.

Adding an [amplifier](https://github.com/oyvindln/vhs-decode/wiki/Amplifier-Setup-Guide) in-between your RF Tap and your Bulkheads or cabled connection to an ADC solution can drastically improve the performance of lower signal level output machines, and reduce/eliminate issues such as cross-hatching from too much signal draw on the internal head amplifier, this also removes most needs to change any capture device gain levels.


> [!CAUTION]
> Just because a test point has this name doesn't automatically mean it will have the signal we want, especially when it comes to HiFi audio, be sure to check with the service manual if possible and do small test captures before finalising any RF Tap setup. 


## Basic Guidance 


Finding Test Points

Decks follow this naming or close to it not every possible name is covered.

**Video FM RF Signal:**

`RF C`, `RF Y`, `RF Y+C`, `V RF`, `PB`, `PB.FM`, `V ENV`, `ENV`, `ENVE`, `ENVELOPE`, `VIDEO ENVE`, `VIDEO ENVELOPE`

**HiFi Audio FM Signal:**

`HiFi`, `A.PB`, `A FM`, `A.PB.FM`, `Audio FM`, `A ENV`, `HIFI Envelope`, `FM Mix Out`

**Linear Baseband Audio Signal**

`A-Out` (normally this is easy to tell, but always check service manuals) 


## Parts for an RF Tap

[What tools do I need?](https://github.com/oyvindln/vhs-decode/wiki/Hardware-Installation-Guide)

- 1-4x SMA Male Type B to Pigtail [pre-cut cable](https://s.click.aliexpress.com/e/_c38tshm3) (connection from test point to amplifyer)
- An [ADA4857 Amplifier](https://github.com/oyvindln/vhs-decode/wiki/Amplifier-Setup-Guide)
- BNC Bulkhead to SMA either [pre made](https://s.click.aliexpress.com/e/_DCynGRN), or [solderable](https://s.click.aliexpress.com/e/_c4semdnl). (from amplifyer to back of VCR blank space you can make or modify a hole for)

Connection Cables

- [Direct BNC to BNC](https://s.click.aliexpress.com/e/_DdCYb1l) 
- [50Ohm BNC to BNC Cable](https://s.click.aliexpress.com/e/_DdPzXh5)
- 100cm (40 inch) of [RG316](https://s.click.aliexpress.com/e/_DEjGLGT) or [RG178](https://s.click.aliexpress.com/e/_DBLPVc3) 50 Ohm coaxial cable if your doing your own DIY lengh cables.

> [!TIP]  
> Center is Signal, Outer is Ground, this goes for jacks and for coaxial cable in general.


</details>

<details closed>
<summary>Install An RF Tap</summary>
<br>


> [!TIP]  
> The [Hardware Installation Guide](https://github.com/oyvindln/vhs-decode/wiki/Hardware-Installation-Guide) visually goes over all the installation steps for tape decks to Sony 8mm camcorders.

There is 2 ways to deploy an RF tap in today's workflow, an basic tap which uses an Ceramic capacitor on the test point before cabling to a bulkhead, and sending that signal to an capture device. 

Then the current standard and more recommended workflow of using an [ADA4857 amplifier](https://github.com/oyvindln/vhs-decode/wiki/Amplifier-Setup-Guide) which both limits the signal draw or the load on the deck being tapped, this allows for a controlled signal and gain level ideal for CX Cards and other ADC solutions like the MISRC. 

Adding a 10 uF (0.1 uF to 100 uF range) capacitor to the test point or amplifier is recommended. It can help improve signal integrity. (A handful of VCRs have this on the test point already.)

- 10 uF Capacitors [standard ceramic assortment](https://s.click.aliexpress.com/e/_DlOEdSJ).

For a polarised electrolytic capacitor Positive leg (longer) goes on test/signal point, Negative leg (shorter) on cable to connector/probe. 

However, this does not matter for Ceramic which are bidirectional & recommend today.

While type and voltage does not matter drastically it's best to use new/tested capacitors.

</details>


## Important Notes:


- We use AliExpress links for wide availability globally, but local vendors are a thing.

- With some Sony decks you can use 2.54mm or "DuPont" connectors on the test point pins making an easy RF tap, but may not be as good as soldered joints due to contact connectors varying quality.

- Do not make sharp bends in any RF cabling, keep total cable runs as short as possible, ideally 30-60 cm. More cable = more signal loss.

- Some Umatic decks have an RF output on the back, *however* this only provides Luma RF for dropout detection and not the full Y/C FM signal required for a full RF capture.


# FM RF Capture 


Here's the full [RF Capture Guide](https://github.com/oyvindln/vhs-decode/wiki/RF-Capture-Guide) which covers all device workflows at an overview level.

The FM RF archival workflow may seem initially daunting in terms of RAW storage space usage...

Thanks to using FLAC and re-sampling being real-time 50GB/h is what your normally looking at on initial capture for 20msps Video / 10msps HiFi in 8-bit in level 8 FLAC which is standard for VHS/EIAJ/Betamax/Video8 captures, but SVHS/SuperBeta/ED Beta/Umatic should use 28-40msps due to the higher bandwidth.

> [!NOTE]  
> FLAC  at level 8 or level 6, only very low end systems should be capturing uncompressed 8-bit or 16-bit.)

It is recommended to use a fast storage device with 40-100 MB/s or faster write capacity, in order to avoid dropped samples, ideally an dedicated SSD (via M.2 or SATA connector, not USB) formatted with the exFAT filesystem.

Using FLAC also makes [visual inspection](https://github.com/oyvindln/vhs-decode/wiki/Advanced-RF-Analysis) and manipulation or cutting much easier with tools like [FLAC Chop](https://github.com/harrypm/FLAC-Chop).


# Usage


Ensure you have the latest [decode and tools release](https://github.com/oyvindln/vhs-decode/releases) (Windows / MacOS / Linux)

Or have built directly via [Build](BUILD.md) readme. 


### Decode Launcher GUI


Run via opining the the `.app`/`.exe`/`.appimage` binary builds directly.  

For a basic click-to-open launcher that lets you select common tools and open them in a terminal (or start native GUI tools), use:

    decode-launcher

or if you built from source :

    ./decode-launcher

You can drag and drop RF input files onto the launcher window or input field, and drop `.json` files to auto-fill the params JSON field.

Current native GUI launch targets include:

- `hifi-decode --gui`
- `filter-tune`


## CLI 


Use `cd vhs-decode` to enter into the directory to run commands, `cd ..` to go back a directory.

Use <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop the current process.

You don't actually type `<` and `>` on your input & output files.


# Decoding FM RF Captures


> [!TIP]  
> - `.tbc` files are headerless, you can open them at any time during decoding, preview is limited to what frame info has been fully written to the JSON file updated every 100 frames or so.
> - You can download example demo tapes [here](https://archive.org/details/@decode_team_fm_rf_archives).

<img src="https://github.com/oyvindln/vhs-decode/wiki/assets/images/Post-Processing/ld-analyse-chroma-frame-107-2023-02-23-23-01-21.png"  width="600" height="">

Decode your captured tape to `.tbc` by using:

    vhs-decode [arguments] <capture file> <output name>

Full Usage Example:

    vhs-decode --ire0_adjust --frequency 28.6 --system pal --threads 4 --tape_format VHS VHS_SP_28.6msps_8-bit.flac my-first-decode-2022.10.25

Use the analyse tool during or after decoding to inspect & adjust  decoded data:

    ld-analyse <decoded tape name>.tbc

After decoding, process your tapes VBI data with:

    tbc-process-vbi <decoded tape name>.tbc


## Output File Format


VHS-Decode produces [4fsc sampled](https://github.com/oyvindln/vhs-decode/wiki/Signal-Sampling#4fsc), non-square pixel, timebase corrected, headerless files, there are two formatting versions of these files: 

- S-Video signal in two files for colour-under format tape media such as VHS/Umatic/Betamax/SuperBeta/Video8/Hi8 etc. 

- Composite/CVBS signal in a single file for 1" SMPTE-C/B/A this also applies to 2" Quad and LaserDisc & anything CVBS-Decode. 

These `tbc` files store 16-bit `GREY16` headerless data separated into chroma/luma composite video signals in the `.tbc` format `filename.tbc` & `filename_chroma.tbc` respectively alongside `.json` and `.log` files which carry the frame, TV system and decode information.


## Exporting to Video Files


> [!IMPORTANT]  
> [Read the full export guide here!](https://github.com/oyvindln/vhs-decode/wiki/TBC-to-Video-Export-Guide)

This is easily done inside the export tab of tbc-tools analyse today.

Import / Adjust Framing / Set in & out / Align Audio / Export. 

<img src="assets/images/Export-Page-Rev-3.0-tbc-tools.PNG" width="600" height="">

This will create an FFV1 10-bit 4:2:2 MKV File ready for playback or post-processing but you can select from a wide range of standard framing and codec options for archival or production use including proxys. 

<img src="https://github.com/oyvindln/vhs-decode/wiki/assets/images/Post-Processing/TV-PC-Levels.png" width="600" height="">

The export tool will by default render a lossless, interlaced, top field first and roughly 45-100 Mb/s FFV1 codec (bitrate is frame size and system dependent*) video which, although ideal for archival and further processing has only recently started to gain support in modern [NLEs](https://en.wikipedia.org/wiki/Non-linear_editing).

Some recommended free tools for post-processing are:

- [StaxRip](https://github.com/staxrip/staxrip) & [Hybrid](https://www.selur.de/downloads)

- [Lossless Cut](https://github.com/mifi/lossless-cut) & [DaVinci Resolve](https://www.blackmagicdesign.com/uk/products/davinciresolve) 

These cover editing to across operating systems, and can provide an easier FFmpeg/AviSynth/VapourSynth encoding and QTGMC de-interlacing experience, and full colour grading and post-production ability.


## Profile Options 


For archival to web use we have a wide range of premade video export profiles, which are defined inside the `tbc-video-export.json` file that can be user edited for specific needs. 

> [!WARNING]  
> - Odysee uploads the provided `x264_web` & `x265_web` profiles are ideal for direct upload.
> - Vimeo uploads de-interlacing the FFV1 export with QTGMC etc will be fine, it re-encodes progressive SD quite well. 
> - YouTube de-interlace and upscale to 2880x2160 with HEVC 120Mbps (anything below the 4k bracket is destroyed by compression or will have scaling issues.)


> [!NOTE]  
> The stock profiles for web use the BDWIF deinterlacer, but QTGMC is always recommended. Give the [de-interlacing guide](https://github.com/oyvindln/vhs-decode/wiki/Deinterlacing) a read for more details.


## VBI (Vertical Blanking Interval) Recovery & Preservation


Linux, MacOS & Windows:

    tbc-video-export --vbi input.tbc

This creates a scaled `720x608 PAL` or `720x508 NTSC` (IMX/D10) standard video file with the top VBI space visually exported. 

> [!TIP]  
> Ensure you have adjusted your horizontal framing to cover the edges of the active area including timecode etc before export.

<img src="https://github.com/oyvindln/vhs-decode/wiki/assets/images/Post-Processing/Jennings-With-VBI.png" width="600" height="">

> SVHS PAL tape with VITC timecode

Software decoding provides the full signal frame to work with, including the VBI space, as such recovery software can be used to read and extract this information, or it can be exported visually unlike legacy (and broadcast specialised) capture hardware.

The decode projects tool suite has built-in tools for this `tbc-process-vbi` supporting decoding of VITC, VITS, Closed Captions & Teletext from your `.TBC` files and saves it inside or alongside the `.JSON` or `.db` (SQL) metadata file alongside the outer tape technical data. 


[VITC Timecode](https://github.com/oyvindln/vhs-decode/wiki/VITC-SMPTE-Timecode) (Standard SMPTE Timecode)

[CC EIA-608](https://github.com/oyvindln/vhs-decode/wiki/Closed-Captioning) (Closed Captioning)

[Teletext](https://github.com/oyvindln/vhs-decode/wiki/Teletext) (Subtitles & Information Graphics)

[Tape-based Arcade Games!](https://vhs.thenvm.org/resources/)

[Ruxpin TV Teddy](https://github.com/oyvindln/vhs-decode/blob/vhs_decode/tools/ruxpin-decode/readme.pdf) (Extra audio in visible frame)


# Terminal Arguments


The decoders support various arguments to change how captured tape recordings are processed. 

These vary slightly between formats like VHS & Umatic, but the basic arguments remain the same.

The list below is a short list for common/daily usage but does not cover all the abilities and new or advanced command arguments possible, so please read the [complete and up-to-date command list](https://github.com/oyvindln/vhs-decode/wiki/Command-List) on the wiki as commands may change or be deprecated. It's always good to check this list for any updates or specific issues you're trying to correct. 


## Sample Rate Commands


> [!CAUTION]  
> This is a mandatory setting for the decoders to work.

By default, this is set to 40 MHz (40MSPS) but can accept any rate such as 16msps or higher depending on format and how its been compressed, 20MSPS is the typical format for VHS/Betamax captures today (10MHz bandwith), but SVHS to SMPTE-C will normaly stick to 28-40MSPS, with HiFi being 5-10MSPS. 8-bit (RF) is typical for all formats.

The decoder is 8/12/16 bit agnostic so as long as sample rate is defined and it is in FLAC (or RAW PCM), it will decode it.

`-f` Adjusts sampling frequency in integer units.

Example's `-f 280000hz` or `-f 28mhz` or `-f 8fsc` 

In the case of stock CX Card, use `-f 28.6` for example or [legacy CXADC designators](https://github.com/oyvindln/vhs-decode/wiki/Command-List#cxadc-sample-rates-stock).


## TV System Commands


> [!CAUTION]  
> This is a mandatory setting for the decoders to work as intended.

Changes the [TV System](https://github.com/oyvindln/vhs-decode/wiki/TV-Systems) (line system & respective, colour system if any) to your required regional media format. 

> [!NOTE]  
> - Support for PAL-M is **experimental**.
> - [SECAM & MESECAM](https://github.com/oyvindln/vhs-decode/wiki/Decoding-SECAM-&-MESECAM)
> 
> Use `SECAM` for tapes recorded on SECAM machines (the standard method used in France: ¼ carrier count-down, IEC 60774-1 6.4.1)
> 
> Use `MESECAM` for tapes recorded on PAL-circuitry machines (Middle East etc.); the two are mutually incompatible in colour, and a warning is logged if a SECAM decode looks like an ME-SECAM tape.


`--system` followed by the TV System 

Colour options are: `NTSC`, `PAL`, `PAL-M`, `NTSC-J`, `SECAM`, `MESECAM`,

B/W we have: `819`, `405`

For example: `--system NTSC`


## Tape Format Commands


> [!CAUTION]  
> This is a mandatory setting for the decoders to work properly.

`--tf` or `--tape_format` sets the format of media you wish to decode. 

Current Options are `VHS` (Default), `VHSHQ`, `SVHS`, `UMATIC`, `UMATIC_HI`, `BETAMAX`, `BETAMAX_HIFI`, `SUPERBETA`, `VIDEO8`, `HI8` ,`EIAJ`, `VCR`, `VCR_LP`, `TYPEC` & `TYPEB`.

Example: `--tape_format vhs` 


## Tape Speed Commands


> [!WARNING]  
> This is not a mandatory setting for the decoders to work properly, but can make a "visual" difference in decoding results. 

`--ts` or `--tape_speed` sets the tape speed of media you wish to decode. 

Tape speed adjusts the format parameters slightly so will not always make a difference, but it can make one for LP tapes for example. 

`SP` (default), `LP`, `SLP`, `EP`, and `VP`. Only supported for some formats such as but not limited to (S)VHS & Sony 8mm. 

Example: `--tape_speed LP` 

> [!NOTE]  
> SLP and EP refers to the same speed.


## [Time & Location Control](https://github.com/oyvindln/vhs-decode/wiki/Command-List#time--location-control)


These commands are used for jumping ahead in a file or for defining limits.
Useful to recover decoding after a crash, or for limiting processing time by producing shorter samples.

`-s`  Jumps ahead to any given frame in the capture.

`--start_fileloc` Jumps ahead to any given *sample* in the capture.

`-l` Limits decode length to *n* frames.

`-t` Defines the number of processing threads to use during demodulation, decode cant use more then 6-8 threads per decode currently so using 8 threads is the practical limit as its mostly a single core task.

> [!CAUTION]  
> Upon crashing, vhs-decode automatically dumps the last known sample location in the terminal output.


## [Time Base Correction & Visuals Control](https://github.com/oyvindln/vhs-decode/wiki/Command-List#decode-tbc---time-base-correction-control)


`--debug` sets logger verbosity level to *debug*. Useful for debugging and better log information. (Recommended to enable for archival.)

`--ire0_adjust` Automatically adjust the black/video level on a per-field basis, using the back porch level. Unlike `--clamp` this is done after time base correction. Therefore it can fix the per-field variations caused by the slight carrier shift of VHS-HQ and S-VHS. [more info](https://github.com/oyvindln/vhs-decode/pull/163).

`--ct` enables a *chroma trap*, a filter intended to reduce chroma interference on the main luma signal. Use if seeing banding or checkerboarding on the main luma .tbc in ld-analyse.

`--sl` defines the output *sharpness level*, as an integer from 0-100, the default being 0. Higher values are better suited for plain, flat images i.e. cartoons and animated material, as strong ghosting can occur. (Akin to cranking up the sharpness on any regular TV set.)

`--dp demodblock` displays Raw Demodulated Frequency Spectrum Graphs, makes a pop-up window per each thread so -t 32 will give you 32 GUI windows etc


## Input file formats:


> [!TIP]  
> - The decoders can be RAW uncompressed data or FLAC compressed data. 
> - .RAW will need to be renamed to s16/u16 
> FLAC-compressed captures, can be either 8/12/16-bit

`.flac` (Standard FLAC compessed data)

`.ldf` (40Msps FLAC-compressed DdD data).

`.r8`/`.u8`   (CXADC 8-bit raw data).

`.r16`/`.u16` (CXADC 16-bit raw data).

`.yrf`/`.crf` (Duel channel RF format captures Betacam/WVHS)

> [!CAUTION]  
> If using custom extensions include, `tv system`, `bit depth`, and `sample rate xxMSPS` inside the file name so it's clear what basic settings you will need to use to decode it, and it helps a lot when sharing or archiving something to know what it actually is.


## Output file formats:


Unlike CVBS-Decode & LD-Decode, VHS-Decode does not output its timebase-corrected frames as a single Composite `.tbc` file for colour-under formats, but does for composite modulated ones such as SMPTE-C.

Both the luminance and chrominance channels are separate data files, essentially digital "S-Video", additionally useful for troubleshooting. Descriptor/log files are generated so you end up with 4 files with the following naming:

`filename.tbc`        - Luminance (Y) Image Data (Combined Y/C for CVBS)

`filename_chroma.tbc` - Chrominance (C) Image Data (QAM Modulated)

`filename.tbc.json`   - Frame Descriptor Table (Resolution/Dropouts/SNR/Frames/VBI Timecode)

`filename.log`        - Timecode Indexed Action/Output Log


# Join us!


- [Discord](https://discord.gg/pVVrrxd)

- [Reddit](https://www.reddit.com/r/vhsdecode/)

- You can also find us on IRC at [#domesday86](https://web.libera.chat/#domesday86) on [libera.chat](https://libera.chat)


# Support us! 


- [Donations](https://github.com/oyvindln/vhs-decode/wiki/Donations)


# More Documentation


- [VHS-Decode Wiki](https://github.com/oyvindln/vhs-decode/wiki)

- [Extra Documentation](https://github.com/oyvindln/vhs-decode/wiki/Documents)


## *If in doubt - feel free to read the docs/wiki again, if its not there then ask!*


For future documentation changes, speak with [Harry Munday](https://github.com/harrypm) (harry@opcomedia.com) or on Discord (therealharrypm)