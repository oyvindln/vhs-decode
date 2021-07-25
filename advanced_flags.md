Advanced flags
----

```--debug``` sets logger verbosity level to *debug*. Useful for debugging.

`--AGC` enables the **A**utomatic **G**ain **C**ontrol feature, mainly affecting image brightness/gamma levels. Use if experiencing fluctuating brightness levels or overly dark/bright output.

`-ct` enables a *chroma trap*, a filter intended to reduce chroma interference on the main luma signal. Use if seeing banding or checkerboarding on the main luma .tbc in ld-analyse.

`-sl` defines the output *sharpness level*, as an integer from 0-100, default being 0. Higher values are better suited for plain, flat images i.e. cartoons and animated material, as strong ghosting can occur (akin to cranking up the sharpness on any regular TV set.)

`--notch, --notch_q` define the center frequency and Q factor for an (optional) built-in notch (bandpass) filter. Intended primarily for reducing noise from interference, though the main decoder logic already compensates for this accodring to each tape and TV system's specific frequency values.

`--doDOD` enables *dropout correction*. Please note, this does not force vhs-decode to perform dropout correction; instead, it adds a flag to the output .json, leaving it to be performed in the next step (running any of the gen_vid_chroma scripts.)

`-nld` enables NLD **N**on **L**inear **D**eemphasis, intended to reduce the noise of the picture.

`-nodd` disables the diff demod feature.

`-sclip` enables clipping of the sync pulses at sync level.

`-noclamp` disables black level clamping on the decoded video.

`-cafc` enables the downconverted chroma carrier AFC **A**utomatic **F**requency **C**ontrol, intended to make the chroma upconversion more precise.

`-cshift` defines a multiple of the line length to which the chroma signal will be moved in time. It accepts a decimal value with sign.
For example 0.5 moves the chroma forward 0.5 lines. If it is -0.5 it will move it 0.5 lines backward. 
