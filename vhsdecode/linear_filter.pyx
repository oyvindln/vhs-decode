import cython
import scipy.signal as signal
cimport numpy as np

# assembles the current filter design on a pipe-able filter
cdef class FiltersClass:
    cdef np.ndarray iir_b
    cdef np.ndarray iir_a
    cdef np.ndarray z
    cdef int samp_rate

    def __init__(self, np.ndarray iir_b, np.ndarray iir_a, int samp_rate):
        self.iir_b, self.iir_a = iir_b, iir_a
        self.z = signal.lfilter_zi(self.iir_b, self.iir_a)
        self.samp_rate = samp_rate

    def rate(self) -> cython.int:
        return self.samp_rate

    def filtfilt(self, data: np.ndarray) -> np.ndarray:
        return signal.filtfilt(self.iir_b, self.iir_a, data)

    def lfilt(self, data: np.ndarray) -> np.ndarray:
        cdef np.ndarray output
        output, self.z = signal.lfilter(self.iir_b, self.iir_a, data, zi=self.z)
        return output


# Filter the input with each provided filter iteratively.
cpdef np.ndarray chainfiltfilt_b(np.ndarray data, set filters):
    # Need to see if we can pass the filter list with correct type so we can call cdef but doing
    # this for now.
    for filt in filters:
        data = filt.filtfilt(data)
    return data
