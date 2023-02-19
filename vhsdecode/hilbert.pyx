import cython


import numpy as np
cimport numpy as np
#from math import pi, tau
import math


def unwrap_hilbert(double complex[:] hilbert, cython.double freq_hz):
    cdef cython.double pi = math.pi
    cdef cython.double tau = math.tau
    cdef double[:] tangles = np.angle(hilbert)
    cdef double[:] dangles = np.ediff1d(tangles, to_begin=0)
    del tangles
    cdef np.ndarray tdangles2
    # cdef np.ndarray tdangles2

    # make sure unwapping goes the right way
    if dangles[0] < -pi:
        dangles[0] += tau

    tdangles2 = np.unwrap(dangles)
    del dangles
    # With extremely bad data, the unwrapped angles can jump.
    while np.min(tdangles2) < 0:
        tdangles2[tdangles2 < 0] += tau
    while np.max(tdangles2) > tau:
        tdangles2[tdangles2 > tau] -= tau
    tdangles2 *= (freq_hz / tau)
    return tdangles2
