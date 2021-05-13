<h1> Advanced flags
----

```--debug``` sets logger verbosity level to *debug*. Useful for debugging.

`--noAGC` disables the **A**utomatic **G**ain **C**ontrol feature, mainly affecting image brightness/gamma levels. Use if experiencing fluctuating brightness levels or overly dark/bright output.

`-ct` enables a *chroma trap*, a filter intended to reduce chroma interference on the main luma signal. Use if seeing banding or checkerboarding on the main luma .tbc in ld-analyse.

`-sl` defines the output *sharpness level*, as an integer from 0-100, default being 0. Higher values are better suited for plain, flat images i.e. cartoons and animated material, as strong ghosting can occur (akin to cranking up the sharpness on any regular TV set.)

`--notch, --notch_q` define the center frequency and Q factor for an (optional) built-in notch (bandpass) filter. Intended primarily for reducing noise from interference, though the main decoder logic already compensates for this accodring to each tape and TV system's specific frequency values.

`--doDOD` enables *dropout correction*. Please note, this does not force vhs-decode to perform dropout correction; instead, it adds a flag to the output .json, leaving it to be performed in the next step (running any of the gen_vid_chroma scripts.)

