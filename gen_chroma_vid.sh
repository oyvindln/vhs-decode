#!/bin/bash
input=""
videosystem="pal"
audiotrack=""
startframe=0
length=0

usage () {
  echo "Usage: $(basename $0) [-i input (without .tbc)] [-v videosystem]"
  echo
  echo "Options:"
  echo "-i, --input          Name of the input, without extension. This option is mandatory."
  echo "-v, --videosystem    Either pal or ntsc. Default is pal"
  echo "-a, --audio          Optional Audiotrack (*.wav, mp3, aac et. al.) to mux with generated video"
  echo "-s, --start          Specify the start frame number"
  echo "-l, --length         Specify the length (number of frames to process)"
  echo
  echo "Example: $(basename $0) -i /media/decoded/tape19 -v pal -a /media/decoded/tape19.wav"
}

if [ "$1" = "" ]; then
  usage
  exit 0
fi

while [ "$1" != "" ]; do
  case $1 in
    -i | --input )
      shift
      input=$1
      ;;
    -v | --videosystem )
      shift
      videosystem=$1
      ;;
	-a | --audio )
      shift
      audiotrack=$1
      ;; 
	-s | --start )
      shift
      startframe=$1
      ;;
  	-l | --length )
      shift
      length=$1
      ;; 

	-h | --help )
      usage
      exit
      ;;
    * )
      usage
      exit 1
  esac
  shift
done

if [ "$input" = "" ]; then
  echo "Please specify an input"
  usage
  exit 1
fi

if [ $startframe -gt 0 ]; then
  startframe='--start '$startframe
else
  startframe=""
fi

if [ $length -gt 0 ]; then
  length='--length '$length
else
  length=""
fi

if [ "$videosystem" = "pal" ]; then
echo "Processing tbc as PAL"
  if [ -f "$audiotrack" ] ; then
    ffmpeg -hide_banner -thread_queue_size 4096 -f rawvideo -r 25 -pixel_format gray16le -s 928x576 -color_range tv -i <(ld-dropout-correct -i $input.tbc --output-json /dev/null - | ld-chroma-decoder -f mono -p yuv --input-json $input.tbc.json $startframe $length - -) -f rawvideo -r 25 -pixel_format yuv444p16le -s 928x576 -i <(ld-dropout-correct -i $input\_chroma.tbc --input-json $input.tbc.json --output-json /dev/null - | ld-chroma-decoder -f pal2d --chroma-gain 1.5 -p yuv --input-json $input.tbc.json $startframe $length - -) -itsoffset -00:00:00.000 -i $audiotrack -filter_complex "[1:v]format=yuv422p10le[chroma];[0:v][chroma]mergeplanes=0x001112:yuv422p10le[output]" -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1 -pixel_format yuv422p10le -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -c:a flac -compression_level 12 -map 2:a? -shortest -y $input.mkv
  else
    ffmpeg -hide_banner -thread_queue_size 4096 -f rawvideo -r 25 -pixel_format gray16le -s 928x576 -color_range tv -i <(ld-dropout-correct -i $input.tbc --output-json /dev/null - | ld-chroma-decoder -f mono -p yuv --input-json $input.tbc.json $startframe $length - -) -f rawvideo -r 25 -pixel_format yuv444p16le -s 928x576 -i <(ld-dropout-correct -i $input\_chroma.tbc --input-json $input.tbc.json --output-json /dev/null - | ld-chroma-decoder -f pal2d --chroma-gain 1.5 -p yuv --input-json $input.tbc.json $startframe $length - -) -filter_complex "[1:v]format=yuv422p10le[chroma];[0:v][chroma]mergeplanes=0x001112:yuv422p10le[output]" -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1 -pixel_format yuv422p10le -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -shortest -y $input.mkv
  fi
fi

if [ "$videosystem" = "ntsc" ]; then
echo "Processing tbc as NTSC"
  if [ -f "$audiotrack" ] ; then
    ffmpeg -hide_banner -thread_queue_size 1024 -f rawvideo -r 30000/1001 -pixel_format gray16le -s 760x488 -color_range tv -i <(ld-dropout-correct -i $input.tbc --output-json /dev/null - | ld-chroma-decoder -f mono -p yuv --input-json $input.tbc.json $startframe $length - -) -f rawvideo -r 30000/1001 -pixel_format yuv444p16le -s 760x488 -i <(ld-dropout-correct -i $input\_chroma.tbc --input-json $input.tbc.json --output-json /dev/null - | ld-chroma-decoder -f ntsc2d --ntsc-phase-comp --chroma-gain 1.75 -p yuv --input-json $input\.tbc.json $startframe $length - -) -itsoffset -00:00:00.000 -i $audiotrack -filter_complex "[1:v]format=yuv422p10le[chroma];[0:v][chroma]mergeplanes=0x001112:yuv422p10le[output]" -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 30 -level 3 -slices 16 -slicecrc 1 -top 1 -pixel_format yuv422p10le -color_range tv -color_primaries smpte170m -color_trc smpte170m -colorspace smpte170m -aspect 4:3 -c:a flac -compression_level 12 -map 2:a? -shortest -y $input.mkv
  else
    ffmpeg -hide_banner -thread_queue_size 1024 -f rawvideo -r 30000/1001 -pixel_format gray16le -s 760x488 -color_range tv -i <(ld-dropout-correct -i $input.tbc --output-json /dev/null - | ld-chroma-decoder -f mono -p yuv --input-json $input.tbc.json $startframe $length - -) -f rawvideo -r 30000/1001 -pixel_format yuv444p16le -s 760x488 -i <(ld-dropout-correct -i $input\_chroma.tbc --input-json $input.tbc.json --output-json /dev/null - | ld-chroma-decoder -f ntsc2d --ntsc-phase-comp --chroma-gain 1.75 -p yuv --input-json $input\.tbc.json $startframe $length - -) -filter_complex "[1:v]format=yuv422p10le[chroma];[0:v][chroma]mergeplanes=0x001112:yuv422p10le[output]" -map "[output]":v -c:v ffv1 -coder 1 -context 1 -g 30 -level 3 -slices 16 -slicecrc 1 -top 1 -pixel_format yuv422p10le -color_range tv -color_primaries smpte170m -color_trc smpte170m -colorspace smpte170m -aspect 4:3 -shortest -y $input.mkv
  fi
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
