#!/bin/bash

CHROMA_DECODER="ntsc2d"
CHROMA_GAIN=1
FILTER_COMPLEX="[1:v]format=yuv422p10le[chroma];[0:v][chroma]mergeplanes=0x001112:yuv422p10le[output]"

# Append audio track captured over line input or with external sound recorder, skip if absent:
if [ -f "$1.wav" ] ; then
  ffmpeg -hide_banner -thread_queue_size 1024 \
    -color_range tv \
    -i <( \
      ld-dropout-correct -i "$1.tbc" --output-json /dev/null - | \
      ld-chroma-decoder --chroma-gain 0 -f mono -p y4m --input-json "$1.tbc.json" - - \
    ) \
    -i <( \
      ld-dropout-correct -i "$1_chroma.tbc" --input-json "$1.tbc.json" --output-json /dev/null - | \
      ld-chroma-decoder -f "$CHROMA_DECODER" --luma-nr 0 --ntsc-phase-comp --chroma-gain "$CHROMA_GAIN" -p y4m --input-json "$1.tbc.json" - -\
    ) \
    -itsoffset -00:00:00.000 -i "$1.wav" \
    -filter_complex "$FILTER_COMPLEX" \
    -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 30 -level 3 -slices 16 -slicecrc 1 -top 1 \
    -pixel_format yuv422p10le -color_range tv -color_primaries smpte170m -color_trc bt709 \
    -colorspace smpte170m -c:a flac -compression_level 12 -map 2:a? -shortest -y "$1.mkv"
else
  ffmpeg -hide_banner -thread_queue_size 1024  \
  -color_range tv \
  -i <(\
    ld-dropout-correct -i "$1.tbc" --output-json /dev/null - | \
    ld-chroma-decoder --chroma-gain 0 -f mono -p y4m --input-json "$1.tbc.json" - - \
  ) \
  -i <(\
    ld-dropout-correct -i "$1_chroma.tbc" --input-json "$1.tbc.json" --output-json /dev/null - | \
    ld-chroma-decoder -f "$CHROMA_DECODER" --luma-nr 0 --ntsc-phase-comp --chroma-gain "$CHROMA_GAIN" -p y4m --input-json "$1.tbc.json" - - \
  ) \
  -filter_complex "$FILTER_COMPLEX" \
  -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 30 -level 3 -slices 16 -slicecrc 1 -top 1 \
  -pixel_format yuv422p10le -color_range tv -color_primaries smpte170m \
  -color_trc smpte170m -colorspace smpte170m -shortest -y "$1.mkv"
fi

# Encode internet-friendly clip of previous lossless result:
#ffmpeg -i $1.mkv -vf scale=in_color_matrix=bt601:out_color_matrix=bt709:648x488,bwdif=1:0:0 -c:v libx264 -preset veryslow -b:v 6M -maxrate 6M -bufsize 6M -pixel_format yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -aspect 4:3 -c:a libopus -b:a 192k -strict -2 -movflags +faststart -y $1_lossy.mp4
