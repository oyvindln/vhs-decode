# tbc-video-export 

Made by [Jitterbug](https://github.com/JuniorIsAJitterbug) / [Video Dump Channel](https://www.youtube.com/@videodumpchannel/videos)

## Disclaimer

I don't usually write python. There are likely bugs and other terrible pieces of code that go against best practices.
If you have any comments, suggestions or improvements feel free to do a pull request or contact me on the [Domesday86](https://discord.gg/pVVrrxd) discord.

# Basic Use


Linux & MacOS

    python3 tbc-video-export.py Input-Media.tbc

Windows

    tbc-video-export.exe Input-Media.tbc


## Export Script Features 


- .tbc and extensionless file input
- Automatic Timecode from VITC
- Audio Language & Track Name Support
- PCM Audio profiles for editing support
- Lossless compressed FFV1 Profiles
- ProRes HQ & 4444XQ Profiles
- V210 & V410 Uncompressed Profiles
- AVC/H.264 & HEVC/H.265 De-interlaced Web Profiles

The python script will take your `filename.tbc` & `filename_chroma.tbc` and any metadata from the `.json` and create video files for the PAL or NTSC system automatically, and will set timecode based off VITC data if present.

The default export will make a `Interlaced Top Field First, FFV1 10-bit 4:2:2 video file with FLAC audio.` (roughly 70-100 Mb/s), this also applies if you bind drag and drop your file on the windows exe version, ideal for footage without VBI data, but will require [de-interlacing](https://github.com/oyvindln/vhs-decode/wiki/Deinterlacing) for online use.


# Stock Profiles

These profiles are defined by `tbc-video-export.json`

Define your profile with for example: `--ffmpeg-profile ffv1_8bit_pcm`

| Profile Name  | Codec         | Compression Type     | Bit-Depth | Chroma Sub-Sampling | Audio Format | Container | File Extension | Bitrate    |
|---------------|---------------|----------------------|-----------|---------------------|--------------|-----------|----------------|------------|
| ffv1          | FFV1          | Lossless Compressed  | 10-bit    | 4:2:2               | FLAC Audio   | Matroska  | .mkv           | 70-100mbps |
| ffv1_8bit     | FFV1          | Lossless Compressed  | 8-bit     | 4:2:2               | FLAC Audio   | Matroska  | .mkv           | 40-60mbps  |
| ffv1_pcm      | FFV1          | Lossless Compressed  | 10-bit    | 4:2:2               | PCM Audio    | Matroska  | .mkv           | 70-100mbps |
| ffv1_8bit_pcm | FFV1          | Lossless Compressed  | 8-bit     | 4:2:2               | PCM Audio    | Matroska  | .mkv           | 40-60mbps  |
| prores_hq_422 | ProRes HQ     | Compressed           | 10-bit    | 4:2:2               | PCM Audio    | QuickTime | .mov           | 55-70mbps  |
| prores_4444xq | ProRes 4444XQ | Compressed           | 10-bit    | 4:4:4               | PCM Audio    | QuickTime | .mov           | 80-110mbps |
| v210          | V210          | Uncompressed         | 10-bit    | 4:2:2               | PCM Audio    | QuickTime | .mov           | 200mbps    |
| v410          | V410          | Uncompressed         | 10-bit    | 4:4:4               | PCM Audio    | QuickTime | .mov           | 400mbps    |
| x264_web      | AVC/H.264     | Lossy                | 8-bit     | 4:2:0               | AAC Audio    | QuickTime | .mov           | 8mbps      |
| x265_web      | HEVC/H.265    | Lossy                | 8-bit     | 4:2:0               | AAC Audio    | QuickTime | .mov           | 8mbps      |


We have implemented ProRes & PCM audio modes for better support in [NLEs](https://en.wikipedia.org/wiki/Non-linear_editing) such as Davinchi Resolve.

**NOTE** You can make/share you own FFmpeg profiles by editing the `tbc-video-export.json` as needed.

## Image Framing 

`--vbi` Enables full vertical 

`--letterbox` Letter Box Crops Image for 16:9

`--ffmpeg-force-anamorphic` Forces 16:9 for anamorphic 16:9 media in a 4:3 picture (due to lack of WSS flag reading support currently)

## Command Options 


`````````````
O:\decode>tbc-video-export.exe -h

usage: tbc-video-export [-h] [-t int] [--video-system format] [--verbose] [--what-if] [--skip-named-pipes] [-s int]
                        [-l int] [--reverse] [--input-json json_file] [--luma-only] [--output-padding int] [--vbi]
                        [--letterbox] [--first_active_field_line int] [--last_active_field_line int]
                        [--first_active_frame_line int] [--last_active_frame_line int] [--offset]
                        [--chroma-decoder decoder] [--chroma-gain float] [--chroma-phase float] [--chroma-nr float]
                        [--luma-nr float] [--simple-pal] [--transform-threshold float]
                        [--transform-thresholds file_name] [--show-ffts] [--ffmpeg-profile profile_name]
                        [--ffmpeg-profile-luma profile_name] [--ffmpeg-metadata foo="bar"]
                        [--ffmpeg-thread-queue-size int] [--ffmpeg-force-anamorphic] [--ffmpeg-overwrite]
                        [--ffmpeg-audio-file file_name] [--ffmpeg-audio-title title]
                        [--ffmpeg-audio-language language] [--ffmpeg-audio-offset offset]
                        input

vhs-decode video generation script

options:
  -h, --help            show this help message and exit

global:
  input                 Name of the input tbc.
  -t int, --threads int
                        Specify the number of concurrent threads.
  --video-system format
                        Force a video system format. (default: from .tbc.json)
                        Available formats:
                          pal
                          ntsc
  --verbose             Do not suppress info and warning messages.
  --what-if             Show what commands would be run without running them.
  --skip-named-pipes    Skip using named pipes and instead use a two-step process.

decoder:
  -s int, --start int   Specify the start frame number.
  -l int, --length int  Specify the number of frames to process.
  --reverse             Reverse the field order to second/first.
  --input-json json_file
                        Use a different .tbc.json file.
  --luma-only           Only output a luma video.
  --output-padding int  Pad the output frame to a multiple of this many pixels.
  --vbi                 Adjust FFLL/LFLL/FFRL/LFRL for full vertical export.
  --letterbox           Adjust FFLL/LFLL/FFRL/LFRL for letterbox crop.
  --first_active_field_line int
                        The first visible line of a field.
                          Range 1-259 for NTSC (default: 20)
                                2-308 for PAL  (default: 22)
  --last_active_field_line int
                        The last visible line of a field.
                          Range 1-259 for NTSC (default: 259)
                                2-308 for PAL  (default: 308)
  --first_active_frame_line int
                        The first visible line of a field.
                          Range 1-525 for NTSC (default: 40)
                                1-620 for PAL  (default: 44)
  --last_active_frame_line int
                        The last visible line of a field.
                          Range 1-525 for NTSC (default: 525)
                                1-620 for PAL  (default: 620)
  --offset              NTSC: Overlay the adaptive filter map (only used for testing).
  --chroma-decoder decoder
                        Chroma decoder to use. (default: transform2d for PAL, ntsc2d for NTSC).
                        Available decoders:
                          pal2d
                          transform2d
                          transform3d
                          ntsc1d
                          ntsc2d
                          ntsc3d
                          ntsc3dnoadapt
  --chroma-gain float   Gain factor applied to chroma components (default: 1.5 for PAL, 2.0 for NTSC).
  --chroma-phase float  Phase rotation applied to chroma components (degrees).
  --chroma-nr float     NTSC: Chroma noise reduction level in dB.
  --luma-nr float       Luma noise reduction level in dB.
  --simple-pal          Transform: Use 1D UV filter.
  --transform-threshold float
                        Transform: Uniform similarity threshold in 'threshold' mode.
  --transform-thresholds file_name
                        Transform: File containing per-bin similarity thresholds in 'threshold' mode.
  --show-ffts           Transform: Overlay the input and output FFTs.

ffmpeg:
  --ffmpeg-profile profile_name
                        Specify an FFmpeg profile to use. (default: ffv1)
                        Available profiles:
                          ffv1
                          ffv1_pcm
                          ffv1_8bit
                          ffv1_8bit_pcm
                          prores_hq
                          prores_4444xq
                          v210
                          v410
                          x264_web
                          x264_lossless
                          x264_lossless_8bit
                          x265_web
  --ffmpeg-profile-luma profile_name
                        Specify an FFmpeg profile to use for luma. (default: ffv1_luma)
                        Available profiles:
                          ffv1_luma
  --ffmpeg-metadata foo="bar"
                        Add metadata to output file.
  --ffmpeg-thread-queue-size int
                        Sets the thread queue size for FFmpeg. (default: 1024)
  --ffmpeg-force-anamorphic
                        Force widescreen aspect ratio.
  --ffmpeg-overwrite    Set to overwrite existing video files.
  --ffmpeg-audio-file file_name
                        Audio file to mux with generated video.
  --ffmpeg-audio-title title
                        Title of the audio track.
  --ffmpeg-audio-language language
                        Language of the audio track.
  --ffmpeg-audio-offset offset
                        Offset of the audio track. (default: 00:00:00.000)
`````````````