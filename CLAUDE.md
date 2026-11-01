# CLAUDE.md

Fork of vhs-decode (`hkazmi/vhs-decode-gpu`) aimed at making decoding faster, including an optional NVIDIA GPU path. Performance work lives on the `perf` branch (based on upstream `177b505e`, pushed to `origin/perf`).

## Environment

- **Platform:** WSL2 on a Ryzen 7 3700X with an RTX 4090. `nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi` and is not on PATH.
- **Python:** use the repo's `.venv` (uv-managed CPython 3.12), not the system python3, which lacks the dev headers and numba.
- **Rust:** installed with rustup into `~/.cargo/bin`.
- **Build:** this builds the Rust extension `vhsd_rust` and the Cython `sync` / `linear_filter` modules.
  ```
  PATH=$HOME/.cargo/bin:$PATH uv pip install --python .venv/bin/python -e ".[test]" --config-settings editable_mode=compat
  ```
  `editable_mode=compat` is required. Without it, the console scripts can't import the top-level `vhsd_rust` module.
- **GPU:** needs CuPy (`uv pip install --python .venv/bin/python "cupy-cuda13x[ctk]"`). It is not a project dependency.

## Running

- **Sample:** `SDRconnect_20261014_162807.s16` is an untracked PAL VHS capture, 20 MSPS, about 8 s, 203 frames. It was converted from an SDRconnect I/Q WAV with centre frequency 3.45 MHz.
- **Decode:**
  ```
  .venv/bin/vhs-decode --pal --tf vhs -f 20 [--no_resample] [--gpu] --threads 4 SDRconnect_20261014_162807.s16 out/name
  ```
- **Resampling:** `-f 20` resamples to 40 MSPS (the default). `--no_resample` decodes at 20 MSPS: faster, slightly different output, and upstream marks it "needs testing".
- **Tests:**
  ```
  .venv/bin/python -m pytest tests -q -p no:cacheprovider
  ```
  6 tests fail or error because the `tests/data` and `testdata` git submodules aren't checked out. Those failures are expected.

## Pipeline (where the time goes)

1. **Block demod.** `VHSRFDecode.demodblock` (`vhsdecode/process.py`) demodulates 32K-sample blocks with overlap; `DemodCache` (`lddecode/core.py`) cuts 1024 samples from each edge. Blocks are independent, so this is the parallel, bulk-FFT half.
2. **Per-field processing.** This is serial and runs on the main thread plus one lookahead thread, all contending for the GIL:
   - sync and line locations (`vhsdecode/field.py`, `sync.pyx`);
   - burst lock (`chroma.py` `decode_chroma_phase_rotation`);
   - TBC (`Field.downscale` → `scale_field` in `lddecode/utils.py`);
   - chroma (`process_chroma`), dropouts, and writeout.
3. **Cross-field state** forces fields to be processed in order: field start offset, levels, track phase, field number, and AGC/MTF.

## What was done on `perf`

| Commit | Change |
|---|---|
| `e1d939c7` | `LoadResampled` (`lddecode/utils.py`): raw input is resampled to 40 MHz in-process with a polyphase FIR that matches ffmpeg's defaults, instead of an ffmpeg pipe. That pipe ran serially at ~57 ms/field and was the biggest bottleneck. |
| `b90eb888` | `linear_wow`: the default linear wow mapping is computed directly, not through a 360K-point B-spline, and cached per field (`computewow_scaled` was called 3× per field). |
| `8265775e` | Burst-fit convergence fix in `_tune_burst_measurements`. The frequency stop test (`< 1e-10` Hz on ~4.4 MHz) could never be met, so almost every line ran all 32 Gauss-Newton iterations; it now stops after about 4. |
| `a8691379` | Optional `--gpu` path (`vhsdecode/gpu.py`, CuPy); see below. |
| `5c59d8ea` | Colour-under chroma DC is estimated away from block edges (`shift_chroma_and_remove_dc` takes a range). |
| `8849ebb2` | Colour-under chroma is band-passed per block with an FFT mask (`sos_power_response`, i.e. \|H\|²) instead of `sosfiltfilt`; see below. |

**Chroma bug fixed by `8849ebb2`.** `sosfiltfilt`'s edge transients for the narrow chroma filter last about 2000 samples, but only 1024 are cut. Chroma was corrupted at every block join: RMS error 116–191 against about 186 of signal, over the first and last 256 kept samples. It's now about 0.6. The CAFC path, which filters a whole field after TBC, still uses `sosfiltfilt`.

**GPU path (`--gpu`).** It falls back to the CPU, with a warning, when the GPU or an option is unsupported.
- `GPUResampler` handles a whole run of blocks.
- `GPUDemod` / `DemodCacheGPU` take each `doread` batch and demodulate it in one go:
  - The FM discriminator is a CUDA port of the Rust `unwrap_hilbert`.
  - Results are packed in the structured dtype's byte layout on the GPU and downloaded once into pinned memory.
  - A GPU copy is kept and attached as `rv["video_gpu"]`.
- `GPUScaler` is the `scale_field` kernel. The level adjustments (`wow_level_adjusts`, a sequential filter) are still computed on the CPU.
- Unsupported options: notch, chroma trap, CAFC, nld / sub-deemphasis, video EQ, fsc notch, `disable_diff_demod`, `debug_plot`, `gnrc_afe`.

### Results

Full sample, `--threads 4`, fps after setup; runs vary by about ±8%.

| | Resample 20→40 (default) | `--no_resample` |
|---|---|---|
| Original code, CPU | 3.2 | 5.4 |
| `perf`, CPU | 4.9–5.4 | 8.6 |
| `perf`, `--gpu` | 9.3 | 11.1 |

**Fidelity:**
- The in-process resampler matches ffmpeg to 0.002 IRE RMS.
- GPU vs CPU on `perf`: 0.0003 IRE luma and 0.0006 chroma RMS, with identical field locations and dropouts.
- Luma is byte-identical through the spline and burst-fit changes, except where the chroma fixes feed back through burst lock.

## Gotchas and findings

- **WSL2 transfers are slow:** host↔GPU is about 1.4 GB/s from pageable memory and 3.4 GB/s from pinned. Avoid round trips, and keep data on the GPU between stages.
- **Smoothing default:** `wow_level_adjust_smoothing` defaults to 312.5 for vhs-decode, not 0.
- **numpy 2 float32 FFTs:** `numpy.fft` on float32 input computes in single precision. `LoadResampled` returns float32, so the CPU's first FFT is complex64.
- **CuPy `sosfiltfilt` is slow:** it was no faster than the Rust version for batched 32K rows. Use frequency-domain masks for zero-phase filtering inside blocks.
- **Allocator tuning didn't help:** glibc malloc env tuning for WSL page faults gave only about 3%.
- **Remaining CPU hot spots:**
  - with `--gpu`: `process_chroma` (~20 ms/field), `get_pulses` (~14), burst rotation (~11), and the per-field level/location upload in `GPUScaler` (~5);
  - single-thread without the GPU: demod is the largest.
- **Measurement scripts:** `profile_split.py` (exclusive-time per stage), `ceiling.py` (memoized "free demod" ceiling), `bench_*.py` and `tbc2mp4.py` (a minimal PAL Y/C TBC → MP4 decoder) were scratchpad scripts and are not in the repo. Recreate them if needed.
- **Comparison videos:** `compare_1..6-*.mp4` in the repo root (untracked) compare original vs latest CPU vs latest GPU, each with and without `--no_resample`.
