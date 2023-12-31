# Exporitng to Video Files


These scripts are considered ready for use but have now been moved to legacy in support of the more flexiable and external profile based `tbc-video-export.py` python script. 


### [Read the full exporting guide here](https://github.com/oyvindln/vhs-decode/wiki/Video-Exporting-&-Gen-Chroma-Scripts)



<img src="https://github.com/oyvindln/vhs-decode/wiki/assets/images/Post-Processing/TV-PC-Levels.png" width="600" height="">

VHS-Decode produces two timebase corrected files an S-Video signal in the file domain for VHS/Beta/Video8/Hi8 etc, It can also produce a single CVBS file for formats like SMPTE-C/B. 

These are stored in 16-bit `GREY16` headerless files separated into chroma/luma composite video signals in the `.tbc` format `filename.tbc` & `filename_chroma.tbc` respectively alongside `.json` and `.log` files with frame and decode information, usable with the LD-Decode family of tools ld-analyse, ld-process-vbi, ld-process-vits, ld-dropout-correct & ld-chroma-decoder etc 

The gen chroma scrips will by default render a lossless, interlaced top field first and high-bitrate (roughly 70-100 Mb/s) FFV1 codec video which, which although ideal for archival and further processing has only recently started to gain support in modern [NLEs](https://en.wikipedia.org/wiki/Non-linear_editing).

To generate .mkv files viewable in most media players, simply use the `gen_chroma_vid.sh` script below.

### Export your TBC files to a video file with the following basic command


    ./gen_chroma_vid.sh Input-TBC-Name


## Editing & Basic Online Usage


For editing due to lack of support of FFV1 and sharing online without [de-interlacing](https://github.com/oyvindln/vhs-decode/wiki/Deinterlacing) is not supported properly at all, as such the two commands are provided below to make suitable starting files for this use.

Both commands will automatically use the last file generated as the input.

For editors this transcodes an FFV1/V210 output to a "*near compliant*" interlaced ProRes HQ file:
    
    ffmpeg -hide_banner -i "$1.mkv" -vf setfield=tff -flags +ilme+ildct -c:v prores -profile:v 3 -vendor apl0 -bits_per_mb 8000 -quant_mat hq -mbs_per_slice 8 -pixel_format yuv422p10lep -color_range tv -color_trc bt709 -c:a s24le -vf setdar=4/3,setfield=tff "$1_ProResHQ.mov"
    
For basic online sharing you can use this command to convert the FFV1 output to a de-interlaced (bwdif) lossy upscaled 4:3 1080p MP4:
    
    ffmpeg -hide_banner -i "$1.mkv" -vf scale=in_color_matrix=bt601:out_color_matrix=bt709:1440x1080,bwdif=1:0:0 -c:v libx264 -preset veryslow -b:v 15M -maxrate 15M -bufsize 8M -pixel_format yuv420p -color_trc bt709 -aspect 4:3 -c:a libopus -b:a 192k -strict -2 -movflags +faststart -y "$1_1440x1080_lossy.mp4"


To just export the video with standard settings and the same input file name, the .tbc extention is not required.

## Time Control & Audio Muxing

Command Examples:

    ./gen_chroma_vid.sh -v -s <skip number of frames> -l <number of frames long> -i <.tbc filename without .tbc extension>

The `-a` option can embed an audio file, such as audio decoded via [HiFi Decode](https://github.com/VideoMem/ld-decode/tree/hifi-decode)

    ./gen_chroma_vid.sh -v -s <skip n frames> -l <n frames long> -a <capture>.flac -i <.tbc filename without .tbc extension>

So for example open terminal in the directory of target TBC/Metadata files and run

    ./gen_chroma_vid.sh -v -s <skip n frames> -l <number of frames long> -a <capture>.flac -i <.tbc filename without .tbc extension>


## VBI (Vertical Blanking Interval) Data Recovery


Software decoding provides the full signal frame, recovery software can be used to read and extract this information, however some information can be automatically extracted in the TBC file stage with `ld-processs-vbi` like VITC & Closed Captions. 

[VITC Timecode](https://github.com/oyvindln/vhs-decode/wiki/VITC-SMPTE-Timecode) (Standard SMPTE Timecode)

[CC EIA-608](https://github.com/oyvindln/vhs-decode/wiki/Closed-Captioning) (Closed Captioning)

[Teletext](https://github.com/oyvindln/vhs-decode/wiki/Teletext) (European Subtitles & Information Graphics)

[Tape-based Arcade Games!](https://vhs.thenvm.org/resources/)

[Ruxpin TV Teddy](https://github.com/oyvindln/vhs-decode/blob/vhs_decode/tools/ruxpin-decode/readme.pdf) (Extra audio in visable frame)


### Generate an video output with the top VBI area:


<img src="https://github.com/oyvindln/vhs-decode/wiki/assets/images/Post-Processing/Jennings-With-VBI.png" width="600" height="">

PAL

    ./gen_chroma_vid.sh --ffll 2 --lfll 308 --ffrl 2 --lfrl 620 <tbc-name>

NTSC

    ./gen_chroma_vid.sh --ffll 1 --lfll 259 --ffrl 2 --lfrl 525 <tbc-name>