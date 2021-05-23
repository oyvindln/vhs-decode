# vhs-decode

A fork of ld-decode, the decoding software powering the Domesday86 project.  
This version has been modified to work with the differences found in RF signals taken from videotapes; it is not possible to use both ld-decode and vhs-decode side by side without reinstalling either one at runtime.


Currently, only VHS and U-Matic format tapes are supported; of those, only NTSC and PAL variants are supported, with plans and/or ongoing work to support more formats and systems.

Dependencies
----

vhs-decode, as with ld-decode, has been developed and tested on machines running the latest version of Ubuntu. Other distributions might have outdated (or even newer) versions of certain Python libraries or other dependencies, breaking compatibility. It has been confirmed to work via WSL2.

Other dependencies include Python 3.5+, Qt5, Qmake, and ffmpeg.

Hardware dependencies revolve mainly around hardware used to perform RF captures, as Python is largely platform-agnostic. For capturing, vhs-decode supports both the Domesday Decoder and Connexant CX2388x-based cards using the cxadc kernel module.

Running
----

Install all dependencies required by ld-decode and vhs-decode:

```sudo apt install build-essential git ffmpeg libavcodec-dev libavformat-dev libqwt-qt5-dev qt5-qmake qtbase5-dev python3 python3-pip python3-distutils libfftw3-dev openssl ```

then, all Python modules:

```sudo pip3 install numba pandas matplotlib scipy numpy samplerate```

after, compile and install ld-decode and its tools:

```make```

```sudo make install```

and finally, run vhs-decode by either ```python3 vhs-decode [arguments] <infile> <outfile> ``` or ```./vhs-decode [arguments] <infile> <outfile> ```.

***Generating video files:***

vhs-decode produces .tbc, .json and .log files, usable only with the ld-decode family of tools (though, primarily, ld-analyse, ld-process-vbi, and ld-process-vits.)  
To generate .mkv files viewable in most media players, simply use the ```gen_chroma_vid_[format].sh``` scripts found in the root folder.  
This will generate a lossless, interlaced, high-bitrate (roughly 127 MB/s) files which, although ideal for reducing further loss in quality, are unsuitable for sharing online. An alternate processing mode is included in the script files, but commented out.

Terminal arguments
----

vhs-decode supports various arguments to process differently captured tape recordings and different tape formats/systems. These tend to change frequently as new features are added or superseded.

```--cxadc, --10cxadc, --cxadc3, --10cxadc3, -f```: Changes the sample rate and bit depth for the decoder. By default, this is set to 40 MHz (the sample rate used by the Domesday Decoder) at 16 bits. These flags change this to 28.6 MHZ/8bit, 28.6/16bit, 35.8/8bit and 35.8/10bit, respectively. See the readme file for cxadc-linux3 for more information on what each mode and capture rate means. ```-f``` sets the frequency to a custom, user-defined one (expressed as an integer, ie ```-f 40000000``` for 40 MHz input.)

```-n, -p, -pm, --NTSCJ```: changes the color system to NTSC, PAL, PAL-M, or NTSC-J, respectively. Please note that, as of writing, support for PAL-M is **experimental** and NTSC-J is **untested**.

```-s, --start_fileloc, -l```: Use for jumping ahead in a file or defining limit. Useful for recovering decoding after a crash, or by limiting process time by producing shorter samples. ```-s``` jumps ahead to any given frame in the capture, ```--start_fileloc``` jumps ahead to any given *sample* in the capture (note: upon crashing, vhs-decode automatically dumps the last known sample location in the terminal output) and ```-l``` limits decode length to *n* frames.

```-t```: defines the number of processing threads to use during decoding. By default, the main vhs-decode script allocates only one thread, though the gen_chroma_vid scripts allocate two. The ```make``` rule of thumb of "number of logical processors, plus one" generally applies here, though it mainly depends on the amount of memory available to the decoder.

***Advanced/Debug features***

See [advanced_flags.md](advanced_flags.md) for more information.


Supported formats
----

***Tape formats***

VHS: SP, EP/LP functional but not officially supported

U-Matic

***File formats***

.r8 (8-bit raw data), .r16 (16-bit raw data), .flac/.vhs (FLAC-compressed captures, can be either 8- or 16-bit)

***Output formats***

Unlike ld-decode, vhs-decode does not output its timebase-corrected frames as single .tbc files - instead, it splits both the luminance and chrominance data as individual .tbc files, one "main" comprising luma data, and a "sub" containing solely chroma. Thus, along the usual log file and .json frame descriptor table, there are *two* .tbc files, whereas only the "main" file has to be processed afterwards to produce a usable video file.
