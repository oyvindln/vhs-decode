# ld-decode command

ld-decode's execution is largely controlled by a number of command line switches:

```
ld-decode [-h] [--start file-location] [--length frames] [--seek frame] [--PAL] [--NTSC] [--NTSCJ] [-m mtf]
                 [--MTF_offset mtf_offset] [--noAGC] [--noDOD] [--noEFM] [--preEFM] [--disable_analog_audio] [--AC3]
                 [--start_fileloc start_fileloc] [--ignoreleadout] [--verboseVITS] [--RF_TBC] [--lowband]
                 [--NTSC_color_notch_filter] [--V4300D_notch_filter] [--deemp_low deemp_low] [--deemp_high deemp_high]
                 [--deemp_strength deemp_str] [-t threads] [-f FREQ] [--analog_audio_frequency AFREQ]
                 [--video_bpf_low FREQ] [--video_bpf_high FREQ] [--video_lpf FREQ] [--video_lpf_order VLPF_ORDER]
                 [--audio_filterwidth FREQ] [--use_profiler] [--write-test-ldf output.ldf]
                 infile outfile
```

## Synopsis

```bash
ld-decode [OPTIONS] infile outfile
```

## Positional Arguments

### infile
**Required.** Path to the source file containing the raw RF capture data.

### outfile
**Required.** Base name for destination files. The tool will create multiple output files with this base name and appropriate extensions (e.g., `.tbc`, `.tbc.json`, `.efm`, `.pcm`).

## Options

### Version Information

#### `--version`, `-v`
Display the version number of ld-decode and exit. Does not require positional arguments.

**Example:**
```bash
ld-decode --version
```

### Basic Decoding Options

#### `--start file-location`, `-s file-location`
Jump roughly to frame n of the capture before starting decoding.
- **Type:** Float
- **Default:** 0
- **Range:** Any non-negative number
- **Note:** This performs a rough seek; use `--seek` for precise frame seeking

**Example:**
```bash
ld-decode --start 1000 input.ldf output
```

#### `--length frames`, `-l frames`
Limit the number of frames to decode.
- **Type:** Integer
- **Default:** 110000
- **Range:** Any positive integer
- **Note:** Specifies the maximum number of video frames to process

**Example:**
```bash
ld-decode --length 5000 input.ldf output
```

#### `--seek frame`, `-S frame`
Seek to a precise frame number in the capture before starting decoding.
- **Type:** Integer
- **Default:** -1 (disabled)
- **Range:** Any non-negative integer
- **Note:** More precise than `--start`; requires valid frame synchronization data

**Example:**
```bash
ld-decode --seek 2500 input.ldf output
```

#### `--start_fileloc start_fileloc`
Jump to a precise sample number in the file.
- **Type:** Float
- **Default:** -1 (disabled)
- **Range:** Any non-negative number
- **Note:** Specifies the exact sample position in the RF capture file; overrides `--start`

**Example:**
```bash
ld-decode --start_fileloc 10000000 input.ldf output
```

### Video Standard Options

**Note:** Only one video standard can be selected. Selecting both PAL and NTSC (or NTSCJ) will result in an error.

#### `--PAL`, `-p`, `--pal`
Decode the source as PAL format.
- **Default:** If neither PAL nor NTSC is specified, NTSC is assumed
- **Incompatible with:** `--NTSC`, `--NTSCJ`

**Example:**
```bash
ld-decode --PAL input.ldf output
```

#### `--NTSC`, `-n`, `--ntsc`
Decode the source as NTSC format.
- **Default:** NTSC is the default if no standard is specified
- **Incompatible with:** `--PAL`
- **Note:** Uses IRE 7.5 black level (standard NTSC)

**Example:**
```bash
ld-decode --NTSC input.ldf output
```

#### `--NTSCJ`, `-j`
Decode the source as NTSC-J (Japanese NTSC) format.
- **Default:** Disabled
- **Incompatible with:** `--PAL`
- **Note:** Uses IRE 0 black level (Japanese standard) instead of IRE 7.5

**Example:**
```bash
ld-decode --NTSCJ input.ldf output
```

### Video Processing Options

#### `-m mtf`, `--MTF mtf`
MTF (Modulation Transfer Function) compensation multiplier.
- **Type:** Float
- **Default:** 1.0
- **Range:** Typically 0.5 to 2.0
- **Note:** Adjusts the compensation for frequency-dependent signal loss; values > 1.0 increase high-frequency boost

**Example:**
```bash
ld-decode -m 1.5 input.ldf output
```

#### `--MTF_offset mtf_offset`
MTF compensation offset.
- **Type:** Float
- **Default:** 0
- **Range:** Any float value
- **Note:** Additional offset applied to MTF compensation

**Example:**
```bash
ld-decode --MTF_offset 0.1 input.ldf output
```

#### `--noAGC`
Disable Automatic Gain Control.
- **Default:** AGC is enabled
- **Note:** AGC normalizes signal levels; disabling may be useful for analyzing the raw signal

**Example:**
```bash
ld-decode --noAGC input.ldf output
```

#### `--noDOD`
Disable the dropout detector.
- **Default:** Dropout detection is enabled
- **Note:** Dropouts are signal losses; disabling detection means they won't be flagged in the output

**Example:**
```bash
ld-decode --noDOD input.ldf output
```

#### `--lowband`
Use more restricted RF settings optimized for noisier disks.
- **Default:** Disabled
- **Note:** Applies more conservative filtering suitable for degraded or noisy source material

**Example:**
```bash
ld-decode --lowband input.ldf output
```

### Video Filter Options

#### `--video_bpf_low FREQ`
Video band-pass filter low-end frequency.
- **Type:** FREQ (see Frequency Format section)
- **Default:** System-dependent (varies by PAL/NTSC)
- **Note:** Sets the lower cutoff frequency for the video band-pass filter

**Example:**
```bash
ld-decode --video_bpf_low 2.5MHz input.ldf output
```

#### `--video_bpf_high FREQ`
Video band-pass filter high-end frequency.
- **Type:** FREQ (see Frequency Format section)
- **Default:** System-dependent (varies by PAL/NTSC)
- **Note:** Sets the upper cutoff frequency for the video band-pass filter

**Example:**
```bash
ld-decode --video_bpf_high 13MHz input.ldf output
```

#### `--video_lpf FREQ`
Video low-pass filter frequency.
- **Type:** FREQ (see Frequency Format section)
- **Default:** System-dependent (varies by PAL/NTSC)
- **Note:** Sets the cutoff frequency for the video low-pass filter

**Example:**
```bash
ld-decode --video_lpf 5.0MHz input.ldf output
```

#### `--video_lpf_order VLPF_ORDER`
Video low-pass filter order.
- **Type:** Integer
- **Default:** -1 (use system default)
- **Range:** Positive integers (typically 1-10)
- **Note:** Higher orders provide sharper cutoff but more processing

**Example:**
```bash
ld-decode --video_lpf_order 8 input.ldf output
```

### NTSC-Specific Video Options

#### `--NTSC_color_notch_filter`, `-N`
Mitigate interference from analog audio in red colors in NTSC captures.
- **Default:** Disabled
- **Note:** Only effective with NTSC video standard; addresses crosstalk from audio carriers

**Example:**
```bash
ld-decode --NTSC --NTSC_color_notch_filter input.ldf output
```

### PAL-Specific Video Options

#### `--V4300D_notch_filter`, `-V`
Remove spurious ~8.5MHz signal present in LD-V4300D PAL/digital audio captures.
- **Default:** Disabled
- **Note:** Only effective with PAL video standard; specific to Pioneer LD-V4300D player captures

**Example:**
```bash
ld-decode --PAL --V4300D_notch_filter input.ldf output
```

### Deemphasis Options

Video signals are typically pre-emphasized during recording and must be de-emphasized during playback.

#### `--deemp_low deemp_low`
Deemphasis low frequency in nanoseconds.
- **Type:** Float
- **Default:** System-dependent
  - NTSC: 3.125MHz equivalent
  - PAL: 2.5MHz equivalent
- **Range:** Any positive float
- **Note:** Specifies the time constant for low-frequency deemphasis

**Example:**
```bash
ld-decode --deemp_low 320 input.ldf output
```

#### `--deemp_high deemp_high`
Deemphasis high frequency in MHz.
- **Type:** Float
- **Default:** System-dependent
  - NTSC: 8.33MHz
  - PAL: 10MHz
- **Range:** Any positive float
- **Note:** Specifies the frequency for high-frequency deemphasis

**Example:**
```bash
ld-decode --deemp_high 10.0 input.ldf output
```

#### `--deemp_strength deemp_str`
Strength of deemphasis filter.
- **Type:** Float
- **Default:** 1.0
- **Range:** Typically 0.0 to 2.0
- **Note:** Multiplier for deemphasis effect; 1.0 is standard, <1.0 reduces effect, >1.0 increases effect

**Example:**
```bash
ld-decode --deemp_strength 0.8 input.ldf output
```

### Audio Decoding Options

#### `--disable_analog_audio`, `--disable_analogue_audio`, `--daa`
Disable analog audio decoding.
- **Default:** Analog audio is enabled at 44100Hz
- **Note:** Useful when only video is needed or when processing digital-only sources

**Example:**
```bash
ld-decode --disable_analog_audio input.ldf output
```

#### `--analog_audio_frequency AFREQ`
Set the analog audio output sampling frequency.
- **Type:** Integer (Hz)
- **Default:** 44100
- **Range:** Typically 44100 or 48000
- **Note:** Output sample rate for analog audio tracks

**Example:**
```bash
ld-decode --analog_audio_frequency 48000 input.ldf output
```

#### `--ntsc_audio_rate`
Output analog audio locked to NTSC line timing instead of the default 44100Hz.
- **Default:** Off (analog audio is output at 44100Hz)
- **Effect:** Produces exactly 2.8 samples per line (1470 samples/frame, 735 per field), giving a sample rate of ~44055.944Hz that stays perfectly aligned to the NTSC video timing with no drift. The default 44100Hz rate corresponds to a non-integer 1471.47 samples/frame, which slowly drifts against the video.
- **Metadata:** The `.tbc.json` reports the resolved `sampleRate` as `44055.944055944055`.
- **Note:** NTSC only. The flag is ignored (with a warning) for PAL, which is already frame-locked at 44100Hz (1764 samples/frame). Overrides `--analog_audio_frequency`.

**Example:**
```bash
ld-decode --ntsc_audio_rate input.ldf output
```

#### `--audio_filterwidth FREQ`
Set the analog audio filter width.
- **Type:** FREQ (see Frequency Format section)
- **Default:** System-dependent
- **Note:** Bandwidth of the analog audio channel filters

**Example:**
```bash
ld-decode --audio_filterwidth 150kHz input.ldf output
```

#### `--noEFM`
Disable EFM (Eight-to-Fourteen Modulation) front end for digital audio.
- **Default:** EFM decoding is enabled
- **Note:** EFM is used for digital audio (CD audio) on laserdiscs; disabling skips digital audio extraction

**Example:**
```bash
ld-decode --noEFM input.ldf output
```

#### `--preEFM`
Write filtered but otherwise pre-processed EFM data.
- **Default:** Disabled
- **Note:** Outputs intermediate EFM data before full decoding; useful for debugging or custom processing

**Example:**
```bash
ld-decode --preEFM input.ldf output
```

#### `--AC3`
Enable AC3-RF audio demodulation (NTSC only).  On AC3 LaserDiscs the
analog right audio channel carries a QPSK signal at 2.88 MHz with Dolby
Digital data at 288 kbaud.  With this option, ld-decode demodulates that
signal (see `lddecode/ac3rf.py`) and writes the raw QPSK symbols to
`output.ac3sym` (one symbol per byte, values 0-3); the number of symbols
demodulated during each field is recorded in the field metadata
(`ac3Symbols`, analogous to `efmTValues`).

The `.ac3sym` file is not playable audio by itself: framing, Reed-Solomon
error correction and AC3 frame assembly are performed downstream by
[decode-orc](https://github.com/simoninns/decode-orc)'s *AC3 RF Sink*
stage, which reads the `.tbc`, its metadata, and the `.ac3sym` file and
writes the final playable `.ac3` file.

- **Default:** Disabled
- **Note:** Only compatible with NTSC; attempting to use with PAL will result in an error
- **Incompatible with:** `--PAL`

**Example:**
```bash
ld-decode --NTSC --AC3 input.ldf output
```

The demodulator has a self-contained unit test (synthetic QPSK loopback,
no capture files needed), runnable from the repository root:
```bash
python3 -m tests.test_ac3rf
```
A quick sanity check on real output: `output.ac3sym` should grow by
about 288,000 symbols (bytes) per second of decoded video.

### RF Sampling Options

#### `-f FREQ`, `--frequency FREQ`
RF sampling frequency of the source file.
- **Type:** FREQ (see Frequency Format section)
- **Default:** 40MHz
- **Note:** If the source file has a different sample rate, specify it here; the decoder will resample to 40MHz internally

**Example:**
```bash
ld-decode -f 28.636363MHz input.ldf output
ld-decode -f 8fsc input.ldf output
```

### Processing Options

#### `-t threads`, `--threads threads`
Number of CPU threads to use for decoding.
- **Type:** Integer
- **Default:** 4
- **Range:** 1 to number of CPU cores
- **Note:** More threads can speed up processing but may have diminishing returns beyond the number of CPU cores

**Example:**
```bash
ld-decode -t 8 input.ldf output
```

### Output Options

#### `--RF_TBC`
Create a `.tbc.ldf` file containing time-base corrected RF data.
- **Default:** Disabled
- **Note:** Outputs the RF signal after time-base correction; useful for archival or analysis

**Example:**
```bash
ld-decode --RF_TBC input.ldf output
```

#### `--ignoreleadout`
Continue decoding after detecting the lead-out section.
- **Default:** Disabled (stop at lead-out)
- **Note:** Lead-out marks the end of the disc content; this option processes beyond that marker

**Example:**
```bash
ld-decode --ignoreleadout input.ldf output
```

#### `--verboseVITS`
Enable additional fields in the JSON metadata output.
- **Default:** Disabled
- **Note:** VITS (Vertical Interval Test Signals) contain technical information; this outputs more detailed data

**Example:**
```bash
ld-decode --verboseVITS input.ldf output
```

### Debugging and Development Options

#### `--write-test-ldf output.ldf`
Write the input portion being decoded to a `.ldf` file for bug reporting.
- **Type:** String (filename)
- **Default:** Disabled
- **Note:** Creates a reproducible test case containing the input samples that were decoded; useful for submitting bug reports. The output file cannot be the same as the input file.

**Example:**
```bash
ld-decode --write-test-ldf test-case.ldf input.ldf output
```

#### `--use_profiler`
Enable line_profiler on select functions for performance analysis.
- **Default:** Disabled
- **Note:** Development tool for identifying performance bottlenecks; requires line_profiler to be installed

**Example:**
```bash
ld-decode --use_profiler input.ldf output
```

## Frequency Format

Many options accept frequency values with the `FREQ` type. These can be specified in several formats:

### Bare Number
A number without a suffix is interpreted as **MHz**.
```bash
--frequency 40      # 40 MHz
```

### With Suffix (case-insensitive)
- **Hz**: Hertz
  ```bash
  --frequency 40000000Hz
  ```
- **kHz**: Kilohertz (10³ Hz)
  ```bash
  --frequency 40000kHz
  ```
- **MHz**: Megahertz (10⁶ Hz)
  ```bash
  --frequency 40MHz
  ```
- **GHz**: Gigahertz (10⁹ Hz)
  ```bash
  --frequency 0.04GHz
  ```
- **fSC**: NTSC color subcarrier frequency (315/88 MHz ≈ 3.579545 MHz)
  ```bash
  --frequency 8fsc    # 8× NTSC subcarrier ≈ 28.636 MHz
  ```
- **fSCPAL**: PAL color subcarrier frequency (283.75 × 15625 + 25 Hz ≈ 4.43361875 MHz)
  ```bash
  --frequency 8fscpal # 8× PAL subcarrier ≈ 35.469 MHz
  ```

## Common Usage Examples

### Basic PAL Decode
```bash
ld-decode --PAL input.ldf output
```

### NTSC Decode with Custom Length
```bash
ld-decode --NTSC --length 30000 input.ldf output
```

### High-Quality PAL Decode with Custom Settings
```bash
ld-decode --PAL -m 1.2 --lowband -t 8 input.ldf output
```

### NTSC with AC3 Audio
```bash
ld-decode --NTSC --AC3 input.ldf output
```

### Decode Specific Frame Range
```bash
ld-decode --PAL --start 1000 --length 5000 input.ldf output
```

### Custom Sample Rate Input
```bash
ld-decode --PAL -f 28.636363MHz input.ldf output
```

### NTSC with Color Notch Filter
```bash
ld-decode --NTSC --NTSC_color_notch_filter input.ldf output
```

### PAL V4300D Capture
```bash
ld-decode --PAL --V4300D_notch_filter input.ldf output
```

### Video Only (No Audio)
```bash
ld-decode --PAL --disable_analog_audio --noEFM input.ldf output
```

### Create Test Case for Bug Report
```bash
ld-decode --PAL --start 5000 --length 10 --write-test-ldf bug-report.ldf input.ldf output
```

## Output Files

Based on the base name provided as `outfile`, ld-decode creates several output files:

- **`outfile.tbc`**: Time-base corrected video data
- **`outfile.tbc.json`**: Frame metadata in JSON format
- **`outfile.pcm`**: Analog audio (if enabled, one file per audio channel)
- **`outfile.efm`**: Digital audio EFM data (if enabled)
- **`outfile.log`**: Detailed log file
- **`outfile.tbc.ldf`**: TBC'd RF data (if `--RF_TBC` is used)

## Exit Behavior

The decoder will stop processing when:
1. The requested number of frames (`--length`) has been decoded
2. Lead-out is detected (unless `--ignoreleadout` is specified)
3. End of file is reached
4. An error occurs
5. User interrupts with Ctrl+C (SIGINT)

## Error Conditions

### PAL/NTSC Conflict
```
ERROR: Can only be PAL or NTSC
```
Occurs when multiple video standards are specified.

### AC3 with PAL
```
ERROR: AC3 audio decoding is only supported for NTSC
```
AC3 audio is only available on NTSC laserdiscs.

### Write Test LDF Collision
```
ERROR: --write-test-ldf output file cannot be the same as input file
```
The test output file must be different from the input file.

### Seek Failure
```
ERROR: Seeking failed
```
Unable to seek to the requested frame; may indicate corrupted data or invalid frame number.

## Notes

- **Default Video Standard**: If neither `--PAL` nor `--NTSC` is specified, NTSC is assumed.
- **Thread Count**: The default of 4 threads is suitable for most systems; adjust based on your CPU core count.
- **Sample Rate**: The decoder internally works at 40MHz; all other sample rates are resampled.
- **Frame Counting**: Frames are counted as complete video frames; internally, the decoder processes fields (2 fields = 1 frame).
- **Lead-out**: The lead-out section marks the end of valid disc content; decoding typically stops here unless `--ignoreleadout` is used.

## Version Information

To check the installed version:
```bash
ld-decode --version
```

or

```bash
ld-decode -v
```
