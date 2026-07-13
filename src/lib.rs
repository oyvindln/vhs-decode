pub mod ringbuffer;
mod filters;
mod levels;
mod ported;
mod inputfile;

use numpy::ndarray::{Array1, ArrayView1, ArrayViewMut1, Zip};
use numpy::{Complex64, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

use filters::{sos_filtfilt, sos_filtfilt_f32};
use levels::fallback_vsync_loc_means_impl;
use ported::unwrap_angles_impl;
use inputfile::{InputFile, InputFileType};

// https://mazzo.li/posts/vectorized-atan2.html
#[inline(always)]
fn atan_approx(x: f32) -> f32 {
    const A1: f32 = 0.99997726f32;
    const A3: f32 = -0.33262347f32;
    const A5: f32 = 0.19354346f32;
    const A7: f32 = -0.11643287f32;
    const A9: f32 = 0.05265332f32;
    const A11: f32 = -0.01172120f32;

    let x_sq = x * x;
    x * (A1 + x_sq * (A3 + x_sq * (A5 + x_sq * (A7 + x_sq * (A9 + x_sq * A11)))))
}

#[inline(always)]
fn atan2f_fast(y: f32, x: f32) -> f32 {
    use std::f32::consts::FRAC_PI_2;
    use std::f32::consts::PI;

    let x = x + f32::MIN_POSITIVE.copysign(x);
    let swap = x.abs() < y.abs();
    let atan_input = if swap { x } else { y } / if swap { y } else { x };
    let mut res = atan_approx(atan_input);
    let tmp = if atan_input >= 0.0f32 {
        FRAC_PI_2
    } else {
        -FRAC_PI_2
    };
    res = if swap { tmp - res } else { res };
    match (x >= 0f32, y >= 0f32) {
        (true, true) => res,
        (false, true) => PI + res,
        (false, false) => -PI + res,
        (true, false) => res,
    }
}

#[inline(always)]
fn hilbert_two(a: Complex64, b: Complex64, freq: f32) -> f32 {
    use std::f32::consts::TAU;

    let a = atan2f_fast(a.im as f32, a.re as f32);
    let b = atan2f_fast(b.im as f32, b.re as f32);
    let diff = b - a;
    let diff = diff - (diff / TAU).floor() * TAU;
    diff * freq / TAU
}

#[inline(always)]
fn hilbert_more(a: &[Complex64; 8], b: &[Complex64; 8], out: &mut [f64; 8], freq: f32) {
    for i in 0..8 {
        out[i] = hilbert_two(a[i], b[i], freq) as f64;
    }
}

#[inline(never)]
fn hilbert_all(input_slice: &[Complex64], output_slice: &mut [f64], freq: f32) {
    let len = input_slice.len();
    assert_ne!(len, 0);

    let big_chunks = (len - 1) / 8;
    for i in 0..big_chunks {
        let prevs_slice = &input_slice[i * 8..(i + 1) * 8];
        let currs_slice = &input_slice[i * 8 + 1..(i + 1) * 8 + 1];
        let outs_slice = &mut output_slice[i * 8 + 1..(i + 1) * 8 + 1];
        let prevs = <&[Complex64; 8]>::try_from(prevs_slice).unwrap();
        let currs = <&[Complex64; 8]>::try_from(currs_slice).unwrap();
        let outs = <&mut [f64; 8]>::try_from(outs_slice).unwrap();
        hilbert_more(prevs, currs, outs, freq);
    }
    for i in big_chunks * 8..len - 1 {
        output_slice[i + 1] = hilbert_two(input_slice[i], input_slice[i + 1], freq) as f64;
    }
}

fn unwrap_hilbert_impl(input_array: ArrayView1<'_, Complex64>, freq: f64) -> Array1<f64> {
    let mut output_array = Array1::<f64>::zeros(input_array.len());

    let input_slice = input_array.as_slice().unwrap();
    let output_slice = output_array.as_slice_mut().unwrap();

    hilbert_all(input_slice, output_slice, freq as f32);

    output_array
}

fn complex_angle(input_array: ArrayView1<'_, Complex64>) -> Array1<f64> {
    // Replicate the np.angle function for 1-dimensional input
    // it's possible we could use SIMD in some way to speed this up but that might be complicated.
    let mut output_array = Array1::<f64>::zeros(input_array.len());
    Zip::from(&mut output_array)
        .and(&input_array)
        .for_each(|i, &j| {
            *i = j.im.atan2(j.re);
        });
    output_array
}

fn diff_forward_in_place_impl(mut input_array: ArrayViewMut1<'_, f64>) {
    // output the difference of the output in place, putting 0 in the first element.
    // equivialent to calling np.ediff1d(input_array, to_begin=0)
    for i in (2..input_array.len()).rev() {
        input_array[i] -= input_array[i - 1];
    }
    input_array[1] -= input_array[0];
    input_array[0] = 0.0;
}

#[pyfunction]
fn complex_angle_py<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray1<'py, Complex64>,
) -> Bound<'py, PyArray1<f64>> {
    let input_array = input_array.as_array();
    let output_array = py.detach(|| complex_angle(input_array));
    output_array.into_pyarray(py)
}

#[pyfunction]
fn unwrap_angles<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let input_array = input_array.as_array();
    let output_array = py.detach(|| unwrap_angles_impl(input_array));
    output_array.into_pyarray(py)
}

#[pyfunction]
fn diff_forward_in_place<'py>(py: Python<'py>, mut input_array: PyReadwriteArray1<'py, f64>) {
    let input_array = input_array.as_array_mut();
    py.detach(|| diff_forward_in_place_impl(input_array));
}

#[pyfunction]
fn unwrap_hilbert<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray1<'py, Complex64>,
    freq: f64,
) -> Bound<'py, PyArray1<f64>> {
    let input_array = input_array.as_array();
    let output_array = py.detach(|| unwrap_hilbert_impl(input_array, freq));
    output_array.into_pyarray(py)
}

#[pyfunction]
fn fallback_vsync_loc_means<'py>(
    py: Python<'py>,
    demod_05: PyReadonlyArray1<'py, f64>,
    pulses: Bound<'py, PyList>,
    sample_freq_mhz: f64,
    min_len: f64,
    max_len: f64,
) -> (Bound<'py, PyList>, Bound<'py, PyList>) {
    let demod_05 = demod_05.as_array();
    //let output_list = py.allow_threads(|| fallback_vsync_loc_means_impl(demod_05, freq));
    //output_array.into_pyarray_bound(py)
    fallback_vsync_loc_means_impl(py, demod_05, pulses, sample_freq_mhz, min_len, max_len)
}

#[pyfunction]
fn sosfiltfilt<'py>(
    py: Python<'py>,
    order: u32,
    sos_filter: PyReadonlyArray1<'py, f64>,
    input_array: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let sos_filter = sos_filter.as_array();
    let input_array = input_array.as_array();
    let output_array = py.detach(|| sos_filtfilt(order, sos_filter, input_array));
    output_array.into_pyarray(py)
}

#[pyfunction]
fn sosfiltfilt_f32<'py>(
    py: Python<'py>,
    order: u32,
    sos_filter: PyReadonlyArray1<'py, f64>,
    input_array: PyReadonlyArray1<'py, f32>,
) -> Bound<'py, PyArray1<f32>> {
    let sos_filter = sos_filter.as_array();
    let input_array = input_array.as_array();
    let output_array = py.detach(|| sos_filtfilt_f32(order, sos_filter, input_array));
    output_array.into_pyarray(py)
}

#[pyclass(module = "vhsd_rust")]
pub struct PyInputFile {
    inner: InputFile,
}

#[pyfunction]
pub fn inputfile_open<'py>(
    py: Python<'py>,
    filename: &Bound<'py, PyString>,
    sample_rate: u64,
    resample: bool,
    inputtype: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyInputFile>> {
    let filename: std::path::PathBuf = filename.to_str()?.into();

    let input_type = match inputtype.to_str()? {
        "s16" | "S16" => InputFileType::S16,
        "u8" | "U8" => InputFileType::U8,
        "flac" | "FLAC" => InputFileType::FLAC,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid inputtype '{other}', expected 's16', 'u8', or 'flac'"
            )));
        }
    };

    let inner = InputFile::new(filename, sample_rate, resample, input_type)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    Py::new(py, PyInputFile { inner }).map(|obj| obj.into_bound(py))
}

#[pyfunction]
pub fn inputfile_read<'py>(
    _py: Python<'py>,
    f: &Bound<'py, PyInputFile>,
    mut buffer: PyReadwriteArray1<'py, f32>,
    pos: u64,
    len: u64,
) -> PyResult<u64> {
    let mut f = f.borrow_mut();
    let ctx: &mut InputFile = &mut f.inner;
    let mut view = buffer.as_array_mut();

    // Convert to &mut [f32] only if contiguous
    let slice: &mut [f32] = view
        .as_slice_mut()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("read input file error: array must be contiguous"))?;
    let r = ctx.read(slice, pos, len as usize) as u64;
    Ok(r)
}

#[pyfunction]
pub fn inputfile_getsamplerate<'py>(
    _py: Python<'py>,
    f: &Bound<'py, PyInputFile>,
) -> PyResult<i64> {
    let mut f = f.borrow_mut();
    let ctx: &mut InputFile = &mut f.inner;
    let r = ctx.get_samplerate();
    Ok(r)
}

#[pyfunction]
pub fn inputfile_close<'py>(
    _py: Python<'py>,
    f: &Bound<'py, PyInputFile>,
) -> PyResult<bool> {
    let mut f = f.borrow_mut();
    let ctx: &mut InputFile = &mut f.inner;
    let r = ctx.close();
    Ok(r)
}

#[pyfunction]
fn check_debug<'py>(
    _py: Python<'py>,
) -> PyResult<bool> {
    if cfg!(debug_assertions) {
        Ok(true)
    } else {
        Ok(false)
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn vhsd_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(complex_angle_py, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_angles, m)?)?;
    m.add_function(wrap_pyfunction!(diff_forward_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap_hilbert, m)?)?;
    m.add_function(wrap_pyfunction!(fallback_vsync_loc_means, m)?)?;
    m.add_function(wrap_pyfunction!(sosfiltfilt, m)?)?;
    m.add_function(wrap_pyfunction!(sosfiltfilt_f32, m)?)?;
    m.add_function(wrap_pyfunction!(inputfile_open, m)?)?;
    m.add_function(wrap_pyfunction!(inputfile_read, m)?)?;
    m.add_function(wrap_pyfunction!(inputfile_getsamplerate, m)?)?;
    m.add_function(wrap_pyfunction!(inputfile_close, m)?)?;
    m.add_function(wrap_pyfunction!(check_debug, m)?)?;
    Ok(())
}
