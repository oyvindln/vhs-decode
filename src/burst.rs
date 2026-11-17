//! Colour burst measurement for every line of a field.
//!
//! Port of `_get_phase_sequence`, `_get_upconverted_burst`, `_demod_burst`
//! and `_tune_burst_measurements` from vhsdecode/chroma.py, with the same
//! arithmetic in the same order. The fit's dot products are summed
//! sequentially, where numba uses BLAS, so results can differ from the numba
//! version in the last bits.

use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sci_rs::signal::filter::sosfiltfilt_dyn;

use crate::filters::f32_sos_from_scipy_dyn_slice;

const PI: f64 = std::f64::consts::PI;

/// Python's float `%`: the result takes the sign of the divisor.
#[inline]
fn py_mod(x: f64, y: f64) -> f64 {
    let m = x % y;
    if m != 0.0 && ((y < 0.0) != (m < 0.0)) {
        m + y
    } else {
        m
    }
}

#[inline]
fn degrees(x: f64) -> f64 {
    x * (180.0 / PI)
}

#[derive(Clone, Copy, Default)]
struct Burst {
    line_number: i32,
    start: i32,
    end: i32,
    center: i64,
    phase_deg: f64,
    amplitude: f64,
    magnitude: f64,
    dc: f64,
    frequency: f64,
    i: f64,
    q: f64,
    phase_rotation: i8,
}

fn solve_4x4(a: &[[f64; 4]; 4], b: &[f64; 4]) -> Option<[f64; 4]> {
    let mut m = [[0.0f64; 5]; 4];
    for r in 0..4 {
        m[r][..4].copy_from_slice(&a[r]);
        m[r][4] = b[r];
    }
    for i in 0..4 {
        let mut max_row = i;
        let mut max_val = m[i][i].abs();
        for r in i + 1..4 {
            let val = m[r][i].abs();
            if val > max_val {
                max_val = val;
                max_row = r;
            }
        }
        if max_row != i {
            for c in i..5 {
                let tmp = m[i][c];
                m[i][c] = m[max_row][c];
                m[max_row][c] = tmp;
            }
        }
        if m[i][i].abs() < 1e-12 {
            return None;
        }
        for r in i + 1..4 {
            let factor = m[r][i] / m[i][i];
            for c in i..5 {
                m[r][c] -= factor * m[i][c];
            }
        }
    }
    let mut x = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut sum_ax = 0.0;
        for j in i + 1..4 {
            sum_ax += m[i][j] * x[j];
        }
        x[i] = (m[i][4] - sum_ax) / m[i][i];
    }
    Some(x)
}

/// Gauss-Newton fit of A*cos(2*pi*f*t - phi) + dc to the burst samples.
fn tune_burst(burst: &[f32], burst_start: i64, amp: f64, phi: f64, dc: f64, fsc: f64) -> (f64, f64, f64, f64) {
    const F_WEIGHT: f64 = 1e4;
    const MAX_ITER: usize = 32;
    const MAX_PRECISION: f64 = 1e-10;

    let (mut a, mut phi, mut dc, mut f) = (amp, phi, dc, fsc);
    let n = burst.len();
    let t: Vec<f64> = (0..n).map(|k| (k as i64 + burst_start) as f64 / (4.0 * fsc)).collect();
    let mut cos_t = vec![0.0f64; n];
    let mut sin_t = vec![0.0f64; n];
    let minus_two_pi = -2.0 * PI;

    for _ in 0..MAX_ITER {
        let two_pi_f = 2.0 * PI * f;
        for k in 0..n {
            let theta = two_pi_f * t[k] - phi;
            cos_t[k] = theta.cos();
            sin_t[k] = theta.sin();
        }
        let (mut d00, mut d01, mut s0, mut d03) = (0.0, 0.0, 0.0, 0.0);
        let (mut d11, mut s1, mut d13, mut s3, mut d33) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut r0, mut r1, mut rs, mut r3) = (0.0, 0.0, 0.0, 0.0);
        for k in 0..n {
            let j0 = cos_t[k];
            let j1 = a * sin_t[k];
            let j3 = (minus_two_pi * t[k]) * j1;
            let r = burst[k] as f64 - (a * j0 + dc);
            d00 += j0 * j0;
            d01 += j0 * j1;
            s0 += j0;
            d03 += j0 * j3;
            d11 += j1 * j1;
            s1 += j1;
            d13 += j1 * j3;
            s3 += j3;
            d33 += j3 * j3;
            r0 += j0 * r;
            r1 += j1 * r;
            rs += r;
            r3 += j3 * r;
        }
        let mut jj = [
            [d00, d01, s0, d03],
            [d01, d11, s1, d13],
            [s0, s1, n as f64, s3],
            [d03, d13, s3, d33],
        ];
        let mut jr = [r0, r1, rs, r3];
        jj[3][3] += F_WEIGHT;
        jr[3] += F_WEIGHT * (fsc - f);
        for d in 0..4 {
            jj[d][d] += 1e-6;
        }
        let Some(delta) = solve_4x4(&jj, &jr) else { break };
        a += delta[0];
        phi += delta[1];
        dc += delta[2];
        f += delta[3];
        // Same stopping rule as _tune_burst_measurements in chroma.py.
        if delta[0].abs() < MAX_PRECISION
            && delta[1].abs() < MAX_PRECISION
            && delta[2].abs() < MAX_PRECISION
            && delta[3].abs() < MAX_PRECISION
        {
            break;
        }
    }
    phi = py_mod(phi + PI, 2.0 * PI) - PI;
    (a, phi, dc, f)
}

struct Inputs<'a> {
    chroma: ArrayView1<'a, f32>,
    heterodyne: ArrayView2<'a, f32>,
    sos: Vec<sci_rs::signal::filter::design::Sos<f32>>,
    burst_area: (i64, i64),
    burst_sin: ArrayView1<'a, f32>,
    burst_cos: ArrayView1<'a, f32>,
    line_offset: i64,
    outwidth: i64,
    fsc: f64,
}

fn measure_line(inp: &Inputs, phase: usize, line_number: i64) -> Burst {
    let pad = inp.burst_area.0;
    let line_start = (line_number - inp.line_offset) * inp.outwidth;
    let burst_start = (line_start + inp.burst_area.0 - pad).max(0);
    let burst_end = (inp.chroma.len() as i64).min(line_start + inp.burst_area.1 + pad);

    let het = inp.heterodyne.row(phase);
    let up: Vec<f32> = (burst_start..burst_end.max(burst_start))
        .map(|k| het[k as usize] * inp.chroma[k as usize])
        .collect();
    let filtered_padded: Vec<f32> = sosfiltfilt_dyn(up.iter(), &inp.sos);
    let lo = pad as usize;
    let hi = filtered_padded.len().saturating_sub(pad as usize).max(lo);
    let filtered = &filtered_padded[lo.min(filtered_padded.len())..hi.min(filtered_padded.len())];

    let start = burst_start + pad;
    let len = filtered.len();
    let (mut i_sum, mut q_sum, mut sum) = (0.0f64, 0.0f64, 0.0f64);
    for (k, &s) in filtered.iter().enumerate() {
        let idx = (k as i64 + start) as usize;
        sum += s as f64;
        i_sum += (s * inp.burst_cos[idx]) as f64;
        q_sum += (s * inp.burst_sin[idx]) as f64;
    }
    let lenf = len as f64;
    let phi_guess = py_mod(q_sum.atan2(i_sum) + PI, 2.0 * PI) - PI;
    let dc_guess = sum / lenf;
    let amp_guess = (2.0 * (i_sum * i_sum + q_sum * q_sum).sqrt()) / lenf;

    let (amplitude, fit_phi, dc, frequency) =
        tune_burst(filtered, start, amp_guess, phi_guess, dc_guess, inp.fsc);
    let phase_sample_offset = py_mod(fit_phi, 2.0 * PI) * (2.0 / PI);
    let center = start as f64 + ((len as f64 - 1.0) / 2.0 + phase_sample_offset);

    Burst {
        line_number: line_number as i32,
        start: burst_start as i32,
        end: burst_end as i32,
        center: center as i64,
        phase_deg: py_mod(degrees(fit_phi), 360.0),
        amplitude,
        magnitude: amplitude * (lenf / 2.0),
        dc,
        frequency,
        i: i_sum,
        q: q_sum,
        phase_rotation: -1,
    }
}

#[allow(clippy::too_many_arguments)]
fn phase_sequence(
    inp: &Inputs,
    chroma_rotation: Option<(i64, i64)>,
    starting_index: Option<i64>,
    last_line: i64,
    do_rotation_check: bool,
    rotation_check_start_line: i64,
    track_change_threshold: f64,
    ntsc: bool,
) -> (i64, Vec<Burst>) {
    let start_idx = starting_index.unwrap_or(0);
    let mut rotation_index;
    let mut track_rotation;
    match chroma_rotation {
        Some(rot) => {
            rotation_index = start_idx;
            track_rotation = if rotation_index == 0 { rot.0 } else { rot.1 };
        }
        None => {
            rotation_index = 0;
            track_rotation = start_idx;
        }
    }

    let mut seq: Vec<Burst> = Vec::with_capacity((last_line - inp.line_offset).max(0) as usize);
    let mut current_phase: i64 = 0;
    let mut use_next = false;
    let mut next_phase = 0i64;
    let mut next_burst = Burst::default();

    for line in inp.line_offset..last_line {
        let mut current;
        if use_next {
            current_phase = next_phase;
            current = next_burst;
            use_next = false;
        } else {
            current_phase = (current_phase + track_rotation).rem_euclid(4);
            current = measure_line(inp, current_phase as usize, line);
        }

        if do_rotation_check && line >= rotation_check_start_line && line < last_line - 1 {
            next_phase = (current_phase + track_rotation).rem_euclid(4);
            next_burst = measure_line(inp, next_phase as usize, line + 1);
            let comparison = if ntsc { current } else { *seq.last().unwrap_or(&current) };
            let delta = py_mod(next_burst.phase_deg - comparison.phase_deg + 180.0, 360.0) - 180.0;
            if delta.abs() > track_change_threshold {
                rotation_index = (rotation_index + 1) % 2;
                let rot = chroma_rotation.unwrap();
                track_rotation = if rotation_index == 0 { rot.0 } else { rot.1 };
            } else {
                use_next = true;
            }
        }

        current.phase_rotation = current_phase as i8;
        seq.push(current);
    }

    if chroma_rotation.is_some() && rotation_index == start_idx {
        rotation_index = (rotation_index + 1) % 2;
    }
    (rotation_index, seq)
}

/// Measure the colour burst of every line from `line_offset` to `last_line`.
///
/// Returns (chroma_rotation_index, dict of per-line arrays).
#[pyfunction]
#[pyo3(signature = (chroma, heterodyne, sos_filter, chroma_rotation, starting_index, burst_area,
                    burst_sin, burst_cos, fsc, line_offset, outwidth, last_line,
                    do_rotation_check, rotation_check_start_line, track_change_threshold, ntsc))]
#[allow(clippy::too_many_arguments)]
pub fn burst_phase_sequence<'py>(
    py: Python<'py>,
    chroma: PyReadonlyArray1<'py, f32>,
    heterodyne: PyReadonlyArray2<'py, f32>,
    sos_filter: PyReadonlyArray1<'py, f64>,
    chroma_rotation: Option<(i64, i64)>,
    starting_index: Option<i64>,
    burst_area: (i64, i64),
    burst_sin: PyReadonlyArray1<'py, f32>,
    burst_cos: PyReadonlyArray1<'py, f32>,
    fsc: f64,
    line_offset: i64,
    outwidth: i64,
    last_line: i64,
    do_rotation_check: bool,
    rotation_check_start_line: i64,
    track_change_threshold: f64,
    ntsc: bool,
) -> PyResult<(i64, Bound<'py, PyDict>)> {
    let sos_slice = sos_filter.as_slice()?;
    if sos_slice.len() % 6 != 0 {
        return Err(PyValueError::new_err("sos filter length must be a multiple of 6"));
    }
    if burst_area.0 <= 0 {
        return Err(PyValueError::new_err("burst area must start after the line start"));
    }
    let inp = Inputs {
        chroma: chroma.as_array(),
        heterodyne: heterodyne.as_array(),
        sos: f32_sos_from_scipy_dyn_slice(sos_slice.len() / 6, sos_slice),
        burst_area,
        burst_sin: burst_sin.as_array(),
        burst_cos: burst_cos.as_array(),
        line_offset,
        outwidth,
        fsc,
    };
    if inp.heterodyne.nrows() < 4 || inp.burst_sin.len() < inp.chroma.len() || inp.burst_cos.len() < inp.chroma.len() {
        return Err(PyValueError::new_err("heterodyne or burst wave arrays are too short"));
    }
    let (index, seq) = py.detach(|| {
        phase_sequence(
            &inp,
            chroma_rotation,
            starting_index,
            last_line,
            do_rotation_check,
            rotation_check_start_line,
            track_change_threshold,
            ntsc,
        )
    });

    let d = PyDict::new(py);
    macro_rules! field {
        ($name:literal, $f:ident) => {
            d.set_item($name, seq.iter().map(|b| b.$f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    field!("line_number", line_number);
    field!("start", start);
    field!("end", end);
    field!("center", center);
    field!("phase_deg", phase_deg);
    field!("amplitude", amplitude);
    field!("magnitude", magnitude);
    field!("dc", dc);
    field!("frequency", frequency);
    field!("I", i);
    field!("Q", q);
    field!("phase_rotation", phase_rotation);
    Ok((index, d))
}
