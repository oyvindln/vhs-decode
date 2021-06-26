#!/bin/bash

ffmpeg -hide_banner -thread_queue_size 1024 -f rawvideo \
  -r 30000/1001 -pixel_format gray16le -s 760x488 \
  -i <(ld-dropout-correct -i $1.tbc --output-json /dev/null - | ld-chroma-decoder -f mono -p yuv --input-json $1.tbc.json - -) \
  -f rawvideo -r 30000/1001 -pixel_format yuv444p16le -s 760x488 \
  -i <(ld-dropout-correct -i $1_chroma.tbc --input-json $1.tbc.json --output-json /dev/null - | ld-chroma-decoder -f ntsc1d --ntsc-phase-comp --chroma-gain 2 -p yuv --input-json $1.tbc.json - -) \
  -filter_complex "
    [0]
      format=yuv422p10le
    [deinterlace];
    [deinterlace]
      yadif
    [eq];
    [eq]
      eq=
      contrast=1:
      brightness=0:
      saturation=1:
      gamma=1:
      gamma_r=1:
      gamma_g=1:
      gamma_b=1:
      gamma_weight=1
    [denoise];
    [denoise]
      nlmeans=
      s=2.2
    [chroma];
    [chroma]
        mergeplanes=0x001112:yuv422p10le
    [output]" \
  -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 30 -level 3 -slices 16 -slicecrc 1 -top 1 \
  -pixel_format yuv422p10le -color_range tv -color_primaries smpte170m -color_trc smpte170m \
  -colorspace smpte170m -aspect 4:3 -shortest -y $1.mkv
