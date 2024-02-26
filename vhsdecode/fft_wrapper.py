import threading
import numpy
from pyfftw.builders._utils import _default_effort, _default_threads, _norm_args
import pyfftw.builders as builders
from pyfftw.pyfftw import empty_aligned, simd_alignment


# Default number of threads to use for FFT
_FFT_DEFAULT_THREADS = 1
_FFT_OBJECTS = dict()


def _Xfftn(a, s, axes, overwrite_input, planner_effort,
        threads, auto_align_input, auto_contiguous,
        calling_func, normalise_idft=True, ortho=False,
        real_direction_flag=None):

    calling_thread_id = id(threading.current_thread())
    a = numpy.asanyarray(a)

    args = (overwrite_input, planner_effort, threads,
            auto_align_input, auto_contiguous)

    if not a.flags.writeable and overwrite_input:
        raise ValueError('overwrite_input cannot be True when the ' +
                         'input array flags.writeable is False')

    alignment = a.ctypes.data % simd_alignment
    key = (calling_func, a.shape, a.strides, a.dtype, s.__hash__(),
           axes.__hash__(), alignment, args, calling_thread_id)

    if key not in _FFT_OBJECTS:
        planner_args = (a, s, axes) + args
        FFTW_object = getattr(builders, calling_func)(*planner_args)

        _FFT_OBJECTS[key] = FFTW_object
        return _FFT_OBJECTS[key](normalise_idft=normalise_idft, ortho=ortho)

    else:
        FFTW_object = _FFT_OBJECTS[key]
        orig_output_array = FFTW_object.output_array
        output_shape = orig_output_array.shape
        output_dtype = orig_output_array.dtype
        output_alignment = FFTW_object.output_alignment

        output_array = empty_aligned(
            output_shape, output_dtype, n=output_alignment)

        FFTW_object(input_array=a, output_array=output_array,
                normalise_idft=normalise_idft, ortho=ortho)

    return output_array


# Wrapper for pyfftw numpy interface
class ThreadSafeFFTW(object):

    @staticmethod
    def fft(x, axis=-1, threads=_FFT_DEFAULT_THREADS):
        calling_func = 'fft'
        planner_effort = _default_effort(None)
        threads = _default_threads(threads)

        return _Xfftn(x, None, axis, False, planner_effort,
                      threads, False, False,
                      calling_func, **_norm_args(None))

    @staticmethod
    def ifft(x, axis=-1, threads=_FFT_DEFAULT_THREADS):
        calling_func = 'ifft'
        planner_effort = _default_effort(None)
        threads = _default_threads(threads)

        return _Xfftn(x, None, axis, False, planner_effort,
                      threads, False, False,
                      calling_func, **_norm_args(None))

    @staticmethod
    def rfft(x, threads=_FFT_DEFAULT_THREADS):
        calling_func = 'rfft'
        planner_effort = _default_effort(None)
        threads = _default_threads(threads)

        return _Xfftn(x, None, -1, False, planner_effort,
                      threads, False, False,
                      calling_func, **_norm_args(None))

    @staticmethod
    def irfft(x, threads=_FFT_DEFAULT_THREADS):
        calling_func = 'irfft'
        planner_effort = _default_effort(None)
        threads = _default_threads(threads)

        return _Xfftn(x, None, -1, False, planner_effort,
                      threads, False, False,
                      calling_func, **_norm_args(None))

