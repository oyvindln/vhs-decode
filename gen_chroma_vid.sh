#!/bin/bash
input=""
input_tbc_json=""
videosystem=""
aspect_ratio=""
audiotrack=""
audio_codec="-c:a flac -compression_level 12" # FLAC is defualt. 
chroma_gain=-1 # Not a default value.  It's a sentinel value for the video standards below
chroma_phase=0
output_format="" # Default is 4:2:2 10-bit yuv422p10le / could be 4:4:4 10-bit yuv444p10le
monochrome=0
chroma_decoder=""
video_codec=""
video_arguments=""
video_container=""
encoder_profile=""
thread_queue_size="-thread_queue_size 4096"

# Audio Options: s24le 24-bit Interger PCM / f32le 32-bit Float PCM / u32le 32-bit Interger PCM

# Note V210 only uses .MOV / ProRes Can use .MOV or .MXF / .MKV can be used on anything but might have support issues.

FILTER_COMPLEX="[1:v]format=$output_format[chroma];[0:v][chroma]mergeplanes=0x001112:$output_format[output]"

usage() {
	echo "Usage: $(basename "$0") [-i input (without .tbc)] [-v videosystem]"
	echo
	echo "Options:"
	echo "-i, --input                                 Name of the input, without extension. This option is mandatory."
	echo "-v, --videosystem                           Either pal or ntsc. Default is pal"
	echo "--a1, --audio1                              HiFi Audiotrack (*.wav, mp3, aac et. al.) to mux with generated video"
    echo "--a2, --audio2                              Linear Audiotrack (*.wav, mp3, aac et. al.) to mux with generated video"
	echo "-s, --start                                 Specify the start frame number"
	echo "-l, --length                                Specify the length (number of frames to process)"
	echo "-f, --full                                  Sets output to full-frame of video signal e.g 1135x625 PAL & 910x525 NTSC (experimental)"
	echo "--chroma-gain                               Gain factor applied to chroma components (default 1.5 for PAL, 2.0 for NTSC)"
	echo "--chroma-phase                              Phase rotation applied to chroma components (degrees; default 0.0)"
	echo "-r, --reverse                               Reverse the field order to second/first (default first/second)"
	echo "-t, --threads                               Specify the number of concurrent threads (default number of logical CPUs)"
	echo "-q, --quiet                                 Suppress info and warning messages"
	echo "--input-json <filename>                     Specify the input JSON file"
	echo "-p, --output-format <output-format>         Output format (ffmpeg output formats; default yuv422p10le);"
	echo "                                            RGB48, YUV444P16, GRAY16 pixel formats are supported"
	echo "--video-codec                               Specify the output video codec to use (ex. v210); default is FFV1"
	echo "--video-container                           Specify the output video container to use (ex. mov); default is mkv."
    echo "                                            Specify only the container type, do not include a period."
	echo "-b, --blackandwhite                         Output in black and white"
	echo "--pad, --output-padding <number>            Pad the output frame to a multiple of this many pixels on"
	echo "-d, --decoder <decoder>                     Decoder to use (pal2d, transform2d, transform3d, ntsc1d,"
	echo "                                            ntsc2d, ntsc3d, ntsc3dnoadapt; default automatic)"
	echo "--ffll, --first_active_field_line <number>  The first visible line of a field."
	echo "                                                Range 1-259 for NTSC (default: 20),"
	echo "                                                      2-308 for PAL  (default: 22)"
	echo "--lfll, --last_active_field_line <number>   The last visible line of a field."
	echo "                                                Range 1-259 for NTSC (default: 259),"
	echo "                                                      2-308 for PAL  (default: 308)"
	echo "--ffrl, --first_active_frame_line <number>  The first visible line of a frame."
	echo "                                                Range 1-525 for NTSC (default: 40),"
	echo "                                                      1-620 for PAL  (default: 44)"
	echo "--lfrl, --last_active_frame_line <number>   The last visible line of a frame."
	echo "                                                Range 1-525 for NTSC (default: 525),"
	echo "                                                      1-620 for PAL (default: 620)"
	echo "--ffll 2 --lfll 308 --ffrl 2 --lfrl 620     Export PAL top VBI area"
    echo "--ffll 1 --lfll 259 --ffrl 2 --lfrl 525     Export NTSC top VBI area"
	echo "-o, --oftest                                NTSC: Overlay the adaptive filter map (only used for testing)"
	echo "--chroma-nr <number>                        NTSC: Chroma noise reduction level in dB (default 0.0)"
	echo "--luma-nr <number>                          Luma noise reduction level in dB (default 1.0)"
	echo "--simple-pal                                Transform: Use 1D UV filter (default 2D)"
	echo "--transform-mode <mode>                     Transform: Filter mode to use (level, threshold; default threshold)"
	echo "--transform-threshold <number>              Transform: Uniform similarity threshold in 'threshold' mode (default 0.4)"
	echo "--transform-thresholds <file>               Transform: File containing per-bin similarity thresholds in 'threshold' mode"
	echo "--show-ffts                                 Transform: Overlay the input and output FFTs"
	echo "--ntsc-phase-comp                           Use NTSC QADM decoder taking burst phase into account (BETA)"
    echo "--ffv1                                      (Deafult) Lossless Compressed FFV1 4:2:2 (70-120mbps) in the .mkv containter"
	echo "--v210                                      Output Uncompressed 4:2:2 10-bit V210 (295mbps) in the .mov container"
	echo "--v410                                      Output Uncompressed 4:4:4 10-bit V210 (428mbps) in the .mov container"
	echo "--prores_hq                                 Output Vissully Lossless 4:2:2 10-bit ProRes HQ (70mbps) in the .mov container"
	echo "--prores_4444xq                             Output Vissully Lossless 4:4:4 10-bit ProRes 4444 XQ (130mbps) in the .mov container"
	echo "--language_1 <language>                     Sets the Language marker for audio track 1 such as eng,jpn,ger,spn"
    echo "--language_2 <language>                     Sets the Language marker for audio track 2 such as eng,jpn,ger,spn"
	echo
	echo "Example: $(basename "$0") -i /media/decoded/tape19 -v pal -a /media/decoded/tape19.wav"
}

if [ "$1" = "" ]; then
	usage
	exit 0
fi

decoder_opts=()

while [ "$1" != "" ]; do
	case $1 in
	-i | --input)
		shift
		input="$1"
		;;
	--v210)
		shift
		encoder_profile="$1"
		;;
	--v410)
		shift
		encoder_profile="$1"
		;;
    --ffv1)
		shift
		encoder_profile="$1"
		;;
	--prores_hq)
		shift
		encoder_profile="$1"
	    ;;
	--prores_4444xq)
		shift
		encoder_profile="$1"
		;;
	-v | --videosystem)
		shift
		videosystem="$1"
		;;
	--a1 | --audio1)
		shift
		audiotrack="$1"
		;;
    --a2 | --audio2)
		shift
		audiotrack_2="$1"
		;;
	-s | --start)
		shift
		decoder_opts+=( -s "$1" )
		;;
	-l | --length)
		shift
		decoder_opts+=( -l "$1" )
		;;
	-h | --help)
		usage
		exit
		;;
	-f | --full)
		shift
		decoder_opts+=( --full-frame )
		;;
	--chroma-gain)
		shift
		chroma_gain="$1"
		;;
	--chroma-phase)
		shift
		chroma_phase="$1"
		;;
	-r | --reverse)
		shift
		decoder_opts+=( -r )
		;;
	-t | --threads)
		shift
		decoder_opts+=( -t "$1" )
		;;
	-q | --quiet)
		shift
		decoder_opts+=( -q )
		;;
	--input-json)
		shift
		input_tbc_json="$1"
		;;
	-p | --output-format)
		shift
		output_format="$1"
		;;
	--video-codec)
		shift
		video_codec="$1"
		;;
	--video-container)
		shift
		video_container="$1"
		;;
	-b | --blackandwhite)
		shift
		monochrome=1
		;;
	--pad | --output-padding)
		shift
		decoder_opts+=( --pad "$1" )
		;;
	-d | --decoder)
		shift
		chroma_decoder="$1"
		;;
	--ffll | --first_active_field_line)
		shift
		decoder_opts+=( --ffll "$1" )
		;;
	--lfll | --last_active_field_line)
		shift
		decoder_opts+=( --lfll "$1" )
		;;
	--ffrl | --first_active_frame_line)
		shift
		decoder_opts+=( --ffrl "$1" )
		;;
	--lfrl | --last_active_frame_line)
		shift
		decoder_opts+=( --lfrl "$1" )
		;;
	-o | --oftest)
		shift
		decoder_opts+=( -o )
		;;
	--chroma-nr)
		shift
		decoder_opts+=( --chroma-nr "$1" )
		;;
	--luma-nr)
		shift
		decoder_opts+=( --luma-nr "$1" )
		;;
	--simple-pal)
		shift
		decoder_opts+=( --simple-pal "$1" )
		;;
	--transform-mode)
		shift
		decoder_opts+=( --transform-mode "$1" )
		;;
	--transform-threshold)
		shift
		decoder_opts+=( --transform-threshold "$1" )
		;;
	--transform-thresholds)
		shift
		decoder_opts+=( --transform-thresholds "$1" )
		;;
	--show-ffts)
		shift
		decoder_opts+=( --show-ffts )
		;;
	--ntsc-phase-comp)
		shift
		decoder_opts+=( --ntsc-phase-comp )
		;;
	*)
		input="$1"
		;;
	esac
	shift
done

if [ "$input" = "" ]; then
	echo "Please specify an input"
	usage
	exit 1
fi

input_stripped="${input%.tbc}"
input_tbc="$input_stripped".tbc
input_chroma_tbc="${input%.tbc}"_chroma.tbc

if [ "$input_tbc_json" = "" ]; then
	input_tbc_json=$input_tbc.json
fi

#manual setting of ntsc and pal
if [ "$chroma_decoder" = "ntsc1d" ] || [ "$chroma_decoder" = "ntsc2d" ] || [ "$chroma_decoder" = "ntsc3d" ] || [ "$chroma_decoder" = "ntsc3dnoadapt" ]; then
	videosystem="ntsc"
elif [ "$chroma_decoder" = "pal2d" ] || [ "$chroma_decoder" = "transform2d" ] || [ "$chroma_decoder" = "transform3d" ]; then
	videosystem="pal"
fi

if [ "$videosystem" = "" ]; then
	# Very dumb way of checking if the source is PAL
	# There is probably a better way of doing this...
	pal_found="$(head "$input_tbc_json" | grep -c -e \\\"PAL\\\" -e \\\"PAL-M\\\" -e \\\"isSourcePal\\\":true)"
	if [ "$pal_found" = 1 ]; then
		videosystem="pal"
	else
		videosystem="ntsc"
	fi

if [ "$aspect_ratio" = "" ]; then
	# Very dumb way of checking if the source has WSS 16:9 Widescreen flagging
	# There is probably a better way of doing this...
	wss_found="$(head "$input_tbc_json" | grep -c -e \\\"isWidescreen\\\":true)"
	if [ "$wss_found" = 1 ]; then
		aspect_ratio="-aspect 1.777777"
	else
		aspect_ratio="-aspect 1.333333"
	fi

if [ "$videosystem" = "pal" ]; then
	echo "Processing tbc as PAL"
	if [ "$chroma_gain" = "-1" ]; then
		chroma_gain=1
	fi
	if [ "$chroma_decoder" = "" ]; then
		chroma_decoder="transform2d"
	fi
	color_space="bt470bg"
	color_primaries="bt470bg"
	color_trc="gamma28"
fi

if [ "$videosystem" = "ntsc" ]; then
	echo "Processing tbc as NTSC"
	if [ "$chroma_gain" = "-1" ]; then
		chroma_gain=1
	fi
	if [ "$chroma_decoder" = "" ]; then
		chroma_decoder="ntsc2d"
	fi
	color_space="smpte170m"
	color_primaries="smpte170m"
	color_trc="smpte170m"
	decoder_opts+=( --ntsc-phase-comp )
fi

## FFV1 - Deafult Profile

if [ "$encoder_profile" = "" ]; then
	echo "Encoding to FFV1"
	video_codec="-c:v ffv1"
	video_container="mkv"
	audio_codec="-c:a flac -compression_level 12"
	video_arguments="-coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1"
	output_format="yuv422p10le"
fi

## FFV1 Encoding

if [ "$encoder_profile" = "--FFV1" ]; then
	echo "Encoding to FFV1"
	video_codec="-c:v ffv1"
	video_container="mkv"
	audio_codec="-c:a flac -compression_level 12"
	video_arguments="-coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1"
	output_format="yuv422p10le"
fi

# V210 4:2:2 Encoding 

if [ "$encoder_profile" = "--v210" ]; then
	echo "Encoding to V210"
	video_codec="-c:v v210"
	audio_codec="-c:a s24le"
    video_container="mov"
	video_arguments="-top 1 -f mov"
	output_format="yuv422p10le"
fi

# V410 4:4:4 Encoding

if [ "$encoder_profile" = "--v410" ]; then
	echo "Encoding to V410"
	video_codec="-c:v v410"
	audio_codec="-c:a s24le"
    video_container="mov"
	video_arguments="-top 1 -f mov" 
	output_format="yuv444p10le"
fi

## ProRes HQ 4:2:2 Encoding

if [ "$encoder_profile" = "--prores_hq" ]; then
	echo "Encoding to ProRes HQ"
    video_codec="-c:v prores"
	audio_codec="-c:a s24le"
	video_container="mov"
	video_arguments="-profile:v 3 -vendor apl0 -bits_per_mb 8000 -quant_mat hq -mbs_per_slice 8 -top 1 -f mov"
	output_format="yuv422p10le"
fi

## ProRes 4444XQ 4:4:4 Encoding

if [ "$encoder_profile" = "--prores_4444xq" ]; then
	echo "Encoding to ProRes 4444XQ"
	video_codec="-c:v prores"
	audio_codec="-c:a s24le"
	video_container="mov"
    video_arguments="-profile:v 5 -vendor apl0 -bits_per_mb 8000 -mbs_per_slice 8 -top 1 -f mov"
	output_format="yuv444p10le"
fi

## Audio Options

audio_opts_1=() # Delay Time Offset HH:MM:SS:FFF
audio_opts_2=() # Audio Codec Deafult is FLAC
audio_opts_3=() # Sets Track Lanauage using ISO 639 codes i.g eng/jpn etc 

## No Audio Track Found

if [ "$audiotrack" != "" ] && [ ! -f "$audiotrack" ]; then
    echo "Cannot find audiotrack. Aborting"
    exit 1;
fi

## Audio Track 1 HiFi Audio

if [ -f "$audiotrack" ]; then
	echo "Muxing in audio track $audiotrack"
	audio_opts_1+=( -itsoffset -00:00:00.000 -i "$audiotrack" )
	audio_opts_2+=( "$audio_codec" -map 2:a? -metadata:s:a:0 title="HiFi" )
	audio_opts_3+=( -metadata:s:a:1 language="$language_1" )
fi

## Audio Track 2 Linear Audio

if [ -f "$audiotrack_2" ]; then
	echo "Muxing in audio track $audiotrack_2"
	audio_opts_1+=( -itsoffset -00:00:00.000 -i "$audiotrack_2" )
	audio_opts_2+=( "$audio_codec" -map 3:a? -metadata:s:a:1 title="Linear" ) 
	audio_opts_3+=( -metadata:s:a:1 language="$language_2" )
fi

# There might be a better way of supporting monochrome output

if [ "$monochrome" = "1" ]; then
	ffmpeg -hide_banner "$thread_queue_size" -color_range tv \
	-i <(
		ld-dropout-correct -i "$input_tbc" --output-json /dev/null - |
			ld-chroma-decoder --chroma-gain 0 -f mono -p y4m "${decoder_opts[@]}" --input-json "$input_tbc_json" - -
	) \
	"${audio_opts_1[@]}" \
	-filter_complex "$FILTER_COMPLEX" \
	-map "[output]":v "$video_codec" "$video_arguments" "$aspect_ratio" \
	-pixel_format "$output_format" -color_range tv -color_primaries "$color_primaries" -color_trc "$color_trc" \
	-colorspace $color_space "${audio_opts_2[@]}" "${audio_opts_3[@]}" \
	-shortest -y "$input_stripped"."$video_container"
else
	ffmpeg -hide_banner "$thread_queue_size" -color_range tv \
	-i <(
		ld-dropout-correct -i "$input_tbc" --output-json /dev/null - |
			ld-chroma-decoder --chroma-gain 0 -f mono -p y4m "${decoder_opts[@]}" --input-json "$input_tbc_json" - -
	) \
	-i <(
		ld-dropout-correct -i "$input_chroma_tbc" --input-json "$input_tbc_json" --output-json /dev/null - |
			ld-chroma-decoder -f $chroma_decoder "${decoder_opts[@]}" --luma-nr 0 --chroma-gain $chroma_gain --chroma-phase "$chroma_phase" -p y4m --input-json "$input_tbc_json" - -
	) \
	"${audio_opts_1[@]}" \
	-filter_complex "$FILTER_COMPLEX" \
	-map "[output]":v "$video_codec" "$video_arguments" "$aspect_ratio" \
	-pixel_format "$output_format" -color_range tv -color_primaries "$color_primaries" -color_trc "$color_trc" \
	-colorspace $color_space "${audio_opts_2[@]}" "${audio_opts_3[@]}" \
	-shortest -y "${input_stripped}"."$video_container"
fi


# Render top VBI Area PAL
# ./gen_chroma_vid.sh --ffll 2 --lfll 308 --ffrl 2 --lfrl 620 <tbc-name>

# Render top VBI Area NTSC
# ./gen_chroma_vid.sh --ffll 1 --lfll 259 --ffrl 2 --lfrl 525 <tbc-name>

# Encode internet-friendly clip of previous lossless result:
#ffmpeg -hide_banner -i "$1.mkv" -vf scale=in_color_matrix=bt601:out_color_matrix=bt709:768x576,bwdif=1:0:0 -c:v libx264 -preset veryslow -b:v 6M -maxrate 6M -bufsize 6M -pixel_format yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -aspect 4:3 -c:a libopus -b:a 192k -strict -2 -movflags +faststart -y "$1_lossy.mp4"

# Scaled Version
#ffmpeg -hide_banner -i "$1.mkv" -vf scale=in_color_matrix=bt601:out_color_matrix=bt709:1440x1080,bwdif=1:0:0 -c:v libx264 -preset veryslow -b:v 6M -maxrate 6M -bufsize 6M -pixel_format yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -aspect 4:3 -c:a libopus -b:a 192k -strict -2 -movflags +faststart -y "$1_1440x1080_lossy.mp4"


# Encode an editor-friendly version of the previous lossless result:
# ffmpeg -hide_banner -i "$1.mkv" -vf setfield=tff -flags +ilme+ildct -c:v prores -profile:v 3 -vendor apl0 -bits_per_mb 8000 -quant_mat hq -mbs_per_slice 8 -pixel_format yuv422p10lep -vendor apl0 -bits_per_mb 8000 -color_range tv -color_primaries bt709 -color_trc bt709 -colorspace bt709 -vf setdar=4/3,setfield=tff -c:a s24le "$1_ProResHQ.mov"


# Old version of the script:
##!/bin/sh

#rm -f "$1_doc.tbc"
#rm -f "$1_doc.tbc.json"
#rm -f "$1_chroma_doc.tbc"
#rm -f "$1.rgb"
#rm -f "$1_chroma.rgb"
#rm -f "$1.mkv"
#rm -f "$1_chroma.mkv"

#ld-dropout-correct "$1.tbc" "$1_doc.tbc"
#ld-dropout-correct -i --input-json "$1.tbc.json" "$1_chroma.tbc" "$1_chroma_doc.tbc"

#ld-chroma-decoder -f mono  -p yuv -b "$1_doc.tbc" "$1.rgb"
#ld-chroma-decoder -f pal2d -p yuv --input-json "$1.tbc.json" "$1_chroma_doc.tbc" "$1_chroma.rgb"

#ffmpeg -f rawvideo -r 25 -pix_fmt yuv444p16 -s 928x576 -i "$1_chroma.rgb" -r 25 -pix_fmt gray16 -s 928x576 -i "$1.rgb" -filter_complex "[0:v]format=yuv444p16le[chroma];[1:v]format=yuv444p16le[luma];[chroma][luma]mergeplanes=0x100102:yuv444p16le[output]" -map "[output]":v -c:v libx264 -qp 0 -pix_fmt yuv444p10le -top 1 -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -y -shortest "$1.mkv"
#ffmpeg -f rawvideo -r 25 -pix_fmt rgb48 -s 928x576 -i "$1.rgb" -c:v libx264 -qp 0 -pix_fmt yuv444p10le -top 1 -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -y "$1_luma.mkv"
#ffmpeg -f rawvideo -r 25 -pix_fmt rgb48 -s 928x576 -i "$1_chroma.rgb" -c:v libx264 -qp 0 -pix_fmt yuv444p10le -top 1 -color_range tv -color_primaries bt470bg -color_trc gamma28 -colorspace bt470bg -aspect 4:3 -y "$1_chroma.mkv"

# HOW TO EDIT THIS SCRIPT!
# The goto and repalce sorce localtion of script, these are the locations thats actaully running it.
#./usr/local/bin/gen_chroma_vid.sh
#./usr/local/lib/python3.10/dist-packages/ld_decode-7-py3.10-linux-x86_64.egg/EGG-INFO/scripts/gen_chroma_vid.sh

# FFV1 Stock - "-coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1"

# Thread queue size can be changed to lower or higher value helpful in some cases, 1024 / 2048 etc
  
# FFV1 Encoding - "-c:v ffv1 -coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1"

# V210 Encoding - "-c:v v210 -top 1 -vf setfield=tff -flags +ilme+ildct"

# ProRes HQ Encoding - "-c:v prores -profile:v 3 -vendor apl0 -bits_per_mb 8000 -quant_mat hq -mbs_per_slice 8 -top 1"

# ProRes 4444XQ Encoding - "-c:v prores -profile:v 5 -vendor apl0 -bits_per_mb 8000 -mbs_per_slice 8 -top 1"