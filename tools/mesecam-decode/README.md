# mesecam decode
GNU radio based graph to convert SECAM based TBC files of the vhs-decode project to a YUV file. Combination with original mkv file possible to benefit from dropout correction.
The graph is based on Gnuradio 3.10

## Quick usage guide
* Load both tbc files into the graph, set output file
* Run graph, adjust fieldstart parameter (value 0 or 2) if necessary
* Use bash script to combine with dropout corrected mkv file create by gen_chroma_vid script

## Graphs
Color values are set to properly decode color of TBC files who were decoded by using the MESECAM option in vhs-decode. Also MESECAM option should be the preferred one for decoding, as there are less color streaks in the decoded video.

## Usage
Load the graph in GnuRadio Companion. As file sources select the \*.tbc and \*\_chroma.tbc files. At the very right of the graph define the output-file. It defaults to YUV.bin.
Run the graph. There is a color video preview in the graph. It will slow down the conversion, but it's highly recommend to use it, as you can live check if the video flaps to pink at some point.

In the next step use the bash script to merge the gen_chroma_vid made ffv1 mkv file with the gnuradio created yuv file.
If field offset is 0:
```
./secam_to_yuv.sh -l gen_chroma_vid.mkv -c YUV.bin
```
Add -o as parameter if the field offset is 2.
```
./secam_to_yuv.sh -l gen_chroma_vid.mkv -c YUV.bin -o
```
The script also crops the YUV.bin to the same dimensions and cutout as it would be from the gen_chroma_vid script. So the chroma and luma planes from the normal ld-chroma-decoder and the gnuradio graph output are aligned.


It's also possible to create a standalone MKV file out of the YUV file. Cropping can be omnited if needed.
```
ffmpeg -f rawvideo -pixel_format yuv444p -color_range tv -color_primaries "bt470bg" -color_trc "gamma28" -colorspace "bt470bg" -video_size 1185x624 -r 25 -i YUV.bin -filter:v "crop=928:576:183:46, setdar=1856/1383" -c:v ffv1 -coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1 output.mkv
```

If you're not able to use the bash script take this one-liner to directly merge the YUV file from Gnuradio with the gen_chroma_vid generated file.
```
ffmpeg -y -f rawvideo -pixel_format yuv444p -color_range tv -color_primaries "bt470bg" -color_trc "gamma28" -colorspace "bt470bg" -video_size 1185x624 -r 25 -i "videofromgnuradio" -ss 0.00 -i "videofromgenvidscript" -filter_complex "[0:v]format=yuv444p, crop=928:576:183:46, setdar=1856/1383[chroma]; [1:v]format=yuv422p10le, setdar=1856/1383[luma]; [chroma][luma]mergeplanes=0x100102:yuv422p10le[v]" -map 1:a? -c:a copy -map '[v]' -c:v ffv1 -coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1 "video_merged.mkv"
```
**In both cases the -ss option is important to align both files. Set it from 0.00 to 0.04 if your startfield is 2!**\
0.04 means 1/25th of a second, representing skipping one frame at the beginning from the gen_chroma_vid generated file.

## Limitations
This is an experimental solution to decode SECAM video from the vhs / ld-decode project. A flaw where the right border is pink has been fixed with a workaround. Depending on the input file also the whole video window can get overpainted by this. The cause and a solution is yet unkown.
The demodulation and filtering of color is very basic. There can be a lot of sparks in the color (secam fire), depending on how good or bad the SNR of the decoded input is.
No dropout compensation is present, as this is something usually done by the ld-decode components. But it can be added for the luma (Y) component, see the merge color section.

Flaws present can be, to a certain degree, alleviated in post.

## Troubleshooting
If the majority of the video is affected by pink overpainting there is a possible solutions to fix that.
### Give the input files an offset
To shift the offset by one frame we have to take account, that one field in this graph is 1135 samples per line over 626 lines. So the formula is (1135*626)*startfield. \
The startfield is a variable that should be toggled between 0 and 2. Theoretically it can be anything between zero and the last field of the video. So use this to find a start frame where both fields are not pink. (As said usually only 0 and 2 make sense)
