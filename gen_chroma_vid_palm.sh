#!/bin/bash

# Append audio track captured over line input or with external sound recorder, skip if absent:
if [ -f "$1.wav" ] ; then
ffmpeg -hide_banner -thread_queue_size 4096 -f rawvideo -r 30000/1001 -pixel_format gray16le -s 760x488 -i -color_range tv <(ld-dropout-correct -i $1.tbc --output-json /dev/null - | ld-chroma-decoder -f mono -p yuv --input-json $1.tbc.json - -) -f rawvideo -r 30000/1001 -pixel_format yuv444p16le -s 760x488 -i <(ld-dropout-correct -i $1_chroma.tbc --input-json $1.tbc.json --output-json /dev/null - | ld-chroma-decoder -f pal2d --chroma-gain 1.5 -p yuv --input-json $1.tbc.json - -) -itsoffset -00:00:00.000 -i $1.wav -filter_complex "[1:v]format=yuv422p10le[chroma];[0:v][chroma]mergeplanes=0x001112:yuv422p10le[output]" -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 30 -level 3 -slices 16 -slicecrc 1 -top 1 -pixel_format yuv422p10le -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -c:a flac -compression_level 12 -map 2:a? -shortest -y $1.mkv
else
ffmpeg -hide_banner -thread_queue_size 4096 -f rawvideo -r 30000/1001 -pixel_format gray16le -s 760x488 -color_range tv -i <(ld-dropout-correct -i $1.tbc --output-json /dev/null - | ld-chroma-decoder -f mono -p yuv --input-json $1.tbc.json - -) -f rawvideo -r 30000/1001 -pixel_format yuv444p16le -s 760x488 -i <(ld-dropout-correct -i $1_chroma.tbc --input-json $1.tbc.json --output-json /dev/null - | ld-chroma-decoder -f transform2d --chroma-gain 1.5 -p yuv --input-json $1.tbc.json - -) -filter_complex "[1:v]format=yuv422p10le[chroma];[0:v][chroma]mergeplanes=0x001112:yuv422p10le[output]" -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 30 -level 3 -slices 16 -slicecrc 1 -top 1 -pixel_format yuv422p10le -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -shortest -y $1.mkv
fi

# Encode internet-friendly clip of previous lossless result:
#ffmpeg -hide_banner -i $1.mkv -vf scale=in_color_matrix=bt601:out_color_matrix=bt709:768x576,bwdif=1:0:0 -c:v libx264 -preset veryslow -b:v 6M -maxrate 6M -bufsize 6M -pixel_format yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -aspect 4:3 -c:a libopus -b:a 192k -strict -2 -movflags +faststart -y $1_lossy.mp4

# Old version of the script:
##!/bin/sh

#rm -f $1_doc.tbc
#rm -f $1_doc.tbc.json
#rm -f $1_chroma_doc.tbc
#rm -f $1.rgb
#rm -f $1_chroma.rgb
#rm -f $1.mkv
#rm -f $1_chroma.mkv

#ld-dropout-correct $1.tbc $1_doc.tbc
#ld-dropout-correct -i --input-json $1.tbc.json $1_chroma.tbc $1_chroma_doc.tbc

#ld-chroma-decoder -f mono  -p yuv -b $1_doc.tbc $1.rgb
#ld-chroma-decoder -f pal2d -p yuv --input-json $1.tbc.json $1_chroma_doc.tbc $1_chroma.rgb

#ffmpeg -f rawvideo -r 25 -pix_fmt yuv444p16 -s 928x576 -i $1_chroma.rgb -r 25 -pix_fmt gray16 -s 928x576 -i $1.rgb -filter_complex "[0:v]format=yuv444p16le[chroma];[1:v]format=yuv444p16le[luma];[chroma][luma]mergeplanes=0x100102:yuv444p16le[output]" -map "[output]":v -c:v libx264 -qp 0 -pix_fmt yuv444p10le -top 1 -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -y -shortest $1.mkv
#ffmpeg -f rawvideo -r 25 -pix_fmt rgb48 -s 928x576 -i $1.rgb -c:v libx264 -qp 0 -pix_fmt yuv444p10le -top 1 -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -y $1_luma.mkv
#ffmpeg -f rawvideo -r 25 -pix_fmt rgb48 -s 928x576 -i $1_chroma.rgb -c:v libx264 -qp 0 -pix_fmt yuv444p10le -top 1 -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -y $1_chroma.mkv
