use itertools::Itertools;
use numpy::ndarray::{Array1, ArrayView1};
use sci_rs::na::RealField;
use sci_rs::signal::filter::design::Sos;
use sci_rs::signal::filter::sosfiltfilt_dyn;

/// Make a Sos vector from a slice from scipy from a slice instead of a vector
/// Clone of upstream function just with slice param instead of vector to avoid the extra
/// allocation.
fn from_scipy_dyn_slice<F: RealField + Copy>(order: usize, sos: &[F]) -> Vec<Sos<F>> {
    assert!(order * 6 == sos.len());

    sos.iter()
        .tuples::<(&F, &F, &F, &F, &F, &F)>()
        .take(order)
        .map(|ba| Sos::new([*ba.0, *ba.1, *ba.2], [*ba.3, *ba.4, *ba.5]))
        .collect()
}

/// Make a Sos vector from a slice from scipy from a slice instead of a vector
/// Clone of upstream function just with slice param instead of vector.
/// converts from f64 to f32 param for f32 sosfilt.
/// TODO: Maybe we could just store as f32 python side.
pub(crate) fn f32_sos_from_scipy_dyn_slice(order: usize, sos: &[f64]) -> Vec<Sos<f32>> {
    assert!(order * 6 == sos.len());

    sos.iter()
        .tuples::<(&f64, &f64, &f64, &f64, &f64, &f64)>()
        .take(order)
        .map(|ba| {
            Sos::new(
                [*ba.0 as f32, *ba.1 as f32, *ba.2 as f32],
                [*ba.3 as f32, *ba.4 as f32, *ba.5 as f32],
            )
        })
        .collect()
}

/// Apply a sos filter to the input array using sci_rs
/// and return a new filtered output array
pub fn sos_filtfilt(
    sos_order: u32,
    sos_filter: ArrayView1<'_, f64>,
    input_array: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let sos = from_scipy_dyn_slice(sos_order as usize, sos_filter.as_slice().unwrap());
    let ret = sosfiltfilt_dyn(input_array.iter(), &sos);
    Array1::from_vec(ret)
}

/// Apply a sos filter using sci_rs
/// and return a new filtered output array, single precision version.
pub fn sos_filtfilt_f32(
    sos_order: u32,
    sos_filter: ArrayView1<'_, f64>,
    input_array: ArrayView1<'_, f32>,
) -> Array1<f32> {
    let sos = f32_sos_from_scipy_dyn_slice(sos_order as usize, sos_filter.as_slice().unwrap());
    let ret = sosfiltfilt_dyn(input_array.iter(), &sos);
    Array1::from_vec(ret)
}
