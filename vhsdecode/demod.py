from numba import njit
import numpy as np

import vhsd_rust


@njit(cache=True, nogil=True)
def replace_spikes(demod, demod_diffed, max_value, replace_start=8, replace_end=30):
    """Go through and replace spikes and some samples after them with data
    from the diff demod pass"""
    assert len(demod) == len(
        demod_diffed
    ), "diff demod length doesn't match demod length"
    too_high = max_value
    to_fix = np.where(demod > too_high)[0]

    for i in to_fix:
        start = max(i - replace_start, 0)
        end = min(i + replace_end, len(demod_diffed) - 1)
        # Only replace if it seems to help
        if max(demod_diffed[start:end]) < max(demod[start:end]):
            demod[start:end] = demod_diffed[start:end]

    return demod



def unwrap_hilbert(hilbert, freq_hz):
    # return hilbert_test.unwrap_hilbert(hilbert, freq_hz)
    return vhsd_rust.unwrap_hilbert(hilbert, freq_hz)
