# SECAM Decode


[Wikipedia Info](https://en.wikipedia.org/wiki/SECAM#MESECAM_(home_recording))

GNU radio based graph to convert SECAM based TBC files of the vhs-decode project to a YUV file.

Works with TBC files decoded with either `--system SECAM` (method 1, the standard ¼ count-down recording used in France) or `--system MESECAM` (heterodyne recording, Middle East etc.): both decode modes restore the chroma block to the same studio subcarrier frequencies (foB = 4.25 MHz, foR = 4.40625 MHz).

Combination with original mkv file possible to benefit from dropout correction.

The graph is based on GnuRadio 3.10

Download the Graph & Script here:

[secam_to_yuv.sh](https://github.com/oyvindln/vhs-decode/wiki/assets/gnu-radio/secam_to_yuv.sh) & [secam_to_yuv.grc](https://github.com/oyvindln/vhs-decode/wiki/assets/gnu-radio/secam_to_yuv.grc)


## Quick usage guide


- Load both tbc files into the graph, set output file.
  
- Run graph, adjust `fieldstart` parameter (value 0 or 2) if necessary.
  
- Use bash script to combine with dropout corrected video file by the tbc-video-export tool.



## Graphs


The Color Blue/Red values default to the studio SECAM subcarrier rest frequencies (foB = 4.25 MHz, foR = 4.40625 MHz, i.e. 4.25e6/350e3 and 4.40625e6/350e3), matching TBC files decoded with the MESECAM option in current vhs-decode. For TBC files made with older vhs-decode versions, whose restored carriers sat about 108 kHz high, use the previous values of 12.45 and 12.90 instead.

Decode with the vhs-decode `--system` option matching how the tape was recorded: `SECAM` for method 1 machines (France etc.), `MESECAM` for PAL-circuitry machines (Middle East etc.). Decoding with the wrong one produces no usable colour, and the SECAM mode logs a warning when a tape looks like ME-SECAM.


## Usage


Load the graph in GnuRadio Companion. As file sources select the \*.tbc and \*\_chroma.tbc files. At the very right of the graph define the output-file. 

It defaults to YUV.bin.

Run the graph. 

There is a colour video preview in the graph. It will slow down the conversion, but it's highly recommended to use it, as you can live check if the video flaps to pink at some point.

In the next step use the bash script to merge the luma ffv1 mkv file with the gnuradio created yuv file.
If field offset is 0:

```
./secam_to_yuv.sh -l luma-video-file.mkv -c YUV.bin
```

Add -o as parameter if the field offset is 2.

```
./secam_to_yuv.sh -l luma.mkv -c YUV.bin -o
```

The script also crops the YUV.bin to the same dimensions and cutout as it would be from the exported video. So the chroma and luma planes from the normal ld-chroma-decoder and the gnuradio graph output are aligned.


It's also possible to create a standalone MKV file out of the YUV file. 

Cropping can be omitted if needed.

```
ffmpeg -f rawvideo -pixel_format yuv444p -color_range tv -color_primaries "bt470bg" -color_trc "bt709" -colorspace "bt470bg" -video_size 1185x624 -r 25 -i YUV.bin -filter:v "crop=928:576:183:46, setdar=1856/1383" -c:v ffv1 -coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1 output.mkv
```

If you're not able to use the bash script take this one-liner to directly merge the YUV file from Gnuradio with the tbc-video-export generated file.

```
ffmpeg -y -f rawvideo -pixel_format yuv444p -color_range tv -color_primaries "bt470bg" -color_trc "bt709" -colorspace "bt470bg" -video_size 1185x624 -r 25 -i "videofromgnuradio" -ss 0.00 -i "videofromgenvidscript" -filter_complex "[0:v]format=yuv444p, crop=928:576:183:46, setdar=1856/1383[chroma]; [1:v]format=yuv422p10le, setdar=1856/1383[luma]; [chroma][luma]mergeplanes=0x100102:yuv422p10le[v]" -map 1:a? -c:a copy -map '[v]' -c:v ffv1 -coder 1 -context 1 -g 25 -level 3 -slices 16 -slicecrc 1 -top 1 "video_merged.mkv"
```


**In both cases the -ss option is important to align both files. Set it from 0.00 to 0.04 if your startfield is 2!**

0.04 means 1/25th of a second, representing skipping one frame at the beginning from the tbc-video-export generated file.


## Limitations


This is an experimental solution to decode SECAM video from the vhs / ld-decode project. A flaw where the right border is pink has been fixed with a workaround.

Depending on the input file also the whole video window can get overpainted by this. The cause and a solution is yet unknown.
The demodulation and filtering of colour is very basic. 

There can be a lot of sparks in the colour (secam fire), depending on how good or bad the SNR of the decoded input is.

No dropout compensation is present, as this is something usually done by the ld-decode components. But it can be added for the luma (Y) component, see the merge colour section.

Flaws present can be, to a certain degree, alleviated in post.


## Troubleshooting


If the majority of the video is affected by pink overpainting there is a possible solution to fix that.


### Give the input files an offset


To shift the offset by one frame we have to take into account, that one field in this graph is 1135 samples per line over 626 lines. So the formula is (1135*626)*startfield.

The startfield is a variable that should be toggled between 0 and 2. Theoretically it can be anything between zero and the last field of the video. So use this to find a start frame where both fields are not pink. (As said usually only 0 and 2 make sense)

# Page End 

See [Decoding-SECAM-&-MESECAM](https://github.com/oyvindln/vhs-decode/wiki/Decoding-SECAM-&-MESECAM) for most up-to-date info.