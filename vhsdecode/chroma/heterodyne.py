"""Burst measurement, track phase rotation detection, and the
heterodyne up-conversion pipeline used by the QAM colour-under
formats (PAL/NTSC/PAL-M) and by ME-SECAM.
"""

import math
import numpy as np
import lddecode.core as ldd
import scipy.fft as sps_fft
from vhsdecode.rust_utils import sosfiltfilt_rust

import numba
from numba import njit
from numba.experimental import jitclass
from functools import cache


@njit(cache=False, nogil=True, fastmath=True)
def chroma_automatic_gain(
    chroma,
    burst_abs_ref,
    phase_sequence,
    burst_detected_line,
    sync_tip_len,
    lineoffset,
    outwidth,
    smoothing_window=8,
    k=2.0
):
    burst_count = len(phase_sequence)

    raw_gains = np.empty(burst_count, dtype=np.float64)
    valid_gains = np.empty(burst_count, dtype=np.float64)
    valid_amps = np.empty(burst_count, dtype=np.float64)
    valid_count = 0

    # extract gain values and track valid amplitudes
    for i in range(burst_count):
        current_burst = phase_sequence[i]
        current_amp = current_burst.amplitude if current_burst.amplitude != 0 else 1e-8

        raw_gain = burst_abs_ref / current_amp
        raw_gains[i] = raw_gain

        if current_burst.line_number >= burst_detected_line:
            valid_gains[valid_count] = raw_gain
            valid_amps[valid_count] = current_amp
            valid_count += 1

    # calculate MAD threshold for gain adjustment
    if valid_count > 0:
        active_gains = valid_gains[:valid_count]
        median_gain = np.median(active_gains)
        mad_gain = np.median(np.abs(active_gains - median_gain))
        max_allowable_gain = median_gain + (k * mad_gain)
    else:
        max_allowable_gain = 1.0

    # clamp gains
    clamped_gains = np.minimum(raw_gains, max_allowable_gain)

    # calculate smoothing
    smoothed_gains = np.empty(burst_count, dtype=np.float64)
    half_w = smoothing_window // 2
    for i in range(burst_count):
        start = max(0, i - half_w)
        end = min(burst_count, i + half_w + 1)
        smoothed_gains[i] = np.sum(clamped_gains[start:end]) / (end - start)

    # apply gain, calculate noise floor
    noise_sum = 0
    noise_samples = 0
    for i in range(burst_count):
        current_burst = phase_sequence[i]
        # Entries for lines above the first output line would index backwards
        # from the end of the field (see upconvert_chroma).
        if current_burst.line_number < lineoffset:
            continue

        current_burst_start = (current_burst.line_number - lineoffset) * outwidth

        if i < burst_count - 1:
            next_burst_start = (phase_sequence[i + 1].line_number - lineoffset) * outwidth
        else:
            next_burst_start = len(chroma)

        if current_burst.line_number < burst_detected_line:
            chroma[current_burst_start:next_burst_start] = 0.0
        else:
            gain_start = smoothed_gains[i]
            if i < burst_count - 1:
                gain_end = smoothed_gains[i + 1] 
            else:
                gain_end = smoothed_gains[i]

            length = next_burst_start - current_burst_start
            if length > 0:
                gain_increment = (gain_end - gain_start) / length
                gain = gain_start

                # apply gain
                for j in range(current_burst_start, next_burst_start):
                    chroma[j] = chroma[j] * gain
                    gain += gain_increment

                # get noise floor of sync tip area in the chroma channel
                sync_tip = chroma[next_burst_start + 4 - sync_tip_len : next_burst_start - 4]

                # MAD
                med = np.median(sync_tip)
                abs_dev = np.abs(sync_tip - med)
                mad = np.median(abs_dev)

                # Accumulate the noise floor
                noise_sum += mad * 1.4826
                noise_samples += 1
            
    # Calculate the average noise floor for the entire processed region
    noise_floor = noise_sum / noise_samples if noise_samples > 0 else 0.0

    # return mean of means
    if valid_count > 0:
        return np.mean(valid_amps[:valid_count]), noise_floor

    return 0.0, noise_floor


@njit(cache=True, nogil=True)
def comb_c_pal(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 2H,
    and one advanced by 2H
    line by line. VCRs do this to reduce crosstalk.
    Helps chroma stability on LP tapes in particular.
    (VCRs only adds delayed by 1h instead)
    """

    # TODO: Compensate for PAL quarter cycle offset
    data2 = data.copy()
    numlines = len(data) // line_len
    for line_num in range(16, numlines - 2):
        adv2h = data2[(line_num + 2) * line_len : (line_num + 3) * line_len]
        delayed2h = data2[(line_num - 2) * line_len : (line_num - 1) * line_len]
        line_slice = data[line_num * line_len : (line_num + 1) * line_len]
        # Let the delayed signal contribute 1/4 and advanced 1/4.
        # Could probably make the filtering configurable later.
        data[line_num * line_len : (line_num + 1) * line_len] = (
            (line_slice * 2) - (delayed2h) - adv2h
        ) / 4
    return data


@njit(cache=True, nogil=True)
def comb_c_ntsc(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 1H,
    and one advanced by 1h
    line by line. VCRs do this to reduce crosstalk.
    (VCRs only adds delayed by 1h instead)
    """

    data2 = data.copy()
    numlines = len(data) // line_len
    for line_num in range(16, numlines - 2):
        advanced1h = data2[(line_num + 1) * line_len : (line_num + 2) * line_len]
        delayed1h = data2[(line_num - 1) * line_len : (line_num) * line_len]
        line_slice = data[line_num * line_len : (line_num + 1) * line_len]
        # Let the delayed signal contribute 1/3.
        # Could probably make the filtering configurable later.
        data[line_num * line_len : (line_num + 1) * line_len] = (
            (line_slice * 2) - advanced1h - delayed1h
        ) / 4
    return data


@jitclass({
    'line_number': numba.int32,
    'start': numba.int32,
    'end': numba.int32,
    'phase_deg': numba.float64,
    'phase_offset_deg': numba.float64,
    'amplitude': numba.float64,
    'magnitude': numba.float64,
    'frequency': numba.float64,
    'dc': numba.float64,
    'I': numba.float64,
    'Q': numba.float64,
    'phase_rotation': numba.int8,
    'phase_correction': numba.float64,
})
class BurstInfo:
    line_number: int
    start: int
    end: int
    center: int
    phase_deg: float
    amplitude: float
    magnitude: float
    frequency: float
    dc: float
    I: float
    Q: float
    phase_rotation: int
    phase_correction: float

    def __init__(
        self,
        line_number,
        burst_start,
        burst_end,
        burst_center,
        burst_phase_deg,
        burst_amplitude,
        burst_magnitude,
        burst_dc,
        burst_frequency,
        I,
        Q
    ):
        self.line_number = line_number
        self.start = burst_start
        self.end = burst_end
        self.center = burst_center
        self.phase_deg = burst_phase_deg
        self.amplitude = burst_amplitude
        self.magnitude = burst_magnitude
        self.dc = burst_dc
        self.frequency = burst_frequency
        self.I = I
        self.Q = Q
        self.phase_rotation = -1 # this is set later
        self.phase_correction = 0.0 # radians, fine heterodyne phase shift on top of phase_rotation


@njit(nogil=True, inline='always')
def _solve_4x4(A, b):
    """
    Inlined 4x4 Gaussian elimination solver with partial pivoting.
    """
    M = np.empty((4, 5))
    for r in range(4):
        M[r, 0] = A[r, 0]
        M[r, 1] = A[r, 1]
        M[r, 2] = A[r, 2]
        M[r, 3] = A[r, 3]
        M[r, 4] = b[r]
        
    for i in range(4):
        # Find pivot row
        max_row = i
        max_val = abs(M[i, i])
        for r in range(i + 1, 4):
            val = abs(M[r, i])
            if val > max_val:
                max_val = val
                max_row = r
                
        # Swap rows if necessary
        if max_row != i:
            for c in range(i, 5):
                tmp = M[i, c]
                M[i, c] = M[max_row, c]
                M[max_row, c] = tmp
                
        if abs(M[i, i]) < 1e-12:
            return np.zeros(4), False  # Singular matrix protection
            
        # Eliminate below
        for r in range(i + 1, 4):
            factor = M[r, i] / M[i, i]
            for c in range(i, 5):
                M[r, c] -= factor * M[i, c]
                
    # Back substitution
    x = np.empty(4)
    for i in range(3, -1, -1):
        sum_ax = 0.0
        for j in range(i + 1, 4):
            sum_ax += M[i, j] * x[j]
        x[i] = (M[i, 4] - sum_ax) / M[i, i]
        
    return x, True


@njit(nogil=True, fastmath=False, cache=True, inline='always')
def _tune_burst_measurements(
    burst, burst_start, amp_guess, phi_guess, dc_guess, fsc, f_weight=1e4, max_iter=32, max_precision=1e-10
):
    """
    Gauss-Newton optimization for tuning color burst measurements.

    Fits an NTSC/PAL analytical subcarrier model to digitized video samples:
        Model(t) = A * cos(2 * pi * f * t - phi) + dc

    Utilizes an iterative least-squares (Gauss-Newton) algorithm to extract 
    four critical parameters from the color burst window:
        1. Amplitude (A)       -> Subcarrier strength / chroma saturation
        2. Phase (phi)         -> Subcarrier phase / chroma hue
        3. DC Offset (dc)      -> Local luma/black level offset
        4. Frequency (f)       -> Local subcarrier frequency drift

    Includes a Bayesian frequency prior (f_weight) acting as a regularizer 
    to prevent the frequency parameter from diverging on noisy lines.
    """
    A = amp_guess
    phi = phi_guess
    dc = dc_guess
    f = fsc
    f_target = fsc

    N = len(burst)
    
    # Precompute time array
    t = np.empty(N, dtype=np.float64)
    for k in range(N):
        t[k] = (k + burst_start) / (4.0 * fsc)

    theta = np.empty(N, dtype=np.float64)
    cos_theta = np.empty(N, dtype=np.float64)
    sin_theta = np.empty(N, dtype=np.float64)
    r = np.empty(N, dtype=np.float64)
    
    # Pre-allocate Jacobian components
    j0 = np.empty(N, dtype=np.float64)
    j1 = np.empty(N, dtype=np.float64)
    # Note: j2 is constant 1.0, so we do not need an array for it
    j3 = np.empty(N, dtype=np.float64)

    # Pre-allocate normal equation containers
    J_JT = np.empty((4, 4), dtype=np.float64)
    J_r = np.empty(4, dtype=np.float64)

    minus_two_pi = -2.0 * np.pi

    for _ in range(max_iter):
        two_pi_f = 2.0 * np.pi * f

        theta = (two_pi_f * t) - phi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Compute residual vector
        r = burst - (A * cos_theta + dc)
        
        # Build Jacobian components
        j0 = cos_theta
        j1 = A * sin_theta
        j3 = (minus_two_pi * t) * j1

        # np.dot utilizes BLAS-like instruction sets (SSE, AVX, or AVX-512)
        J_JT[0, 0] = np.dot(j0, j0)
        J_JT[0, 1] = np.dot(j0, j1)
        J_JT[0, 2] = np.sum(j0)          # since j2 is 1.0
        J_JT[0, 3] = np.dot(j0, j3)
        
        J_JT[1, 1] = np.dot(j1, j1)
        J_JT[1, 2] = np.sum(j1)          # since j2 is 1.0
        J_JT[1, 3] = np.dot(j1, j3)
        
        J_JT[2, 2] = float(N)            # np.dot(1.0, 1.0) for N elements
        J_JT[2, 3] = np.sum(j3)          # since j2 is 1.0
        
        J_JT[3, 3] = np.dot(j3, j3)
        
        # Compute J_r vector elements using SIMD-accelerated dot products
        J_r[0] = np.dot(j0, r)
        J_r[1] = np.dot(j1, r)
        J_r[2] = np.sum(r)               # since j2 is 1.0
        J_r[3] = np.dot(j3, r)

        # Mirror the upper triangle to the lower triangle (Exploit Symmetry)
        J_JT[1, 0] = J_JT[0, 1]
        J_JT[2, 0] = J_JT[0, 2]
        J_JT[2, 1] = J_JT[1, 2]
        J_JT[3, 0] = J_JT[0, 3]
        J_JT[3, 1] = J_JT[1, 3]
        J_JT[3, 2] = J_JT[2, 3]

        # Apply Bayesian frequency centering prior/penalty
        J_JT[3, 3] += f_weight
        J_r[3] += f_weight * (f_target - f)

        # Diagonal regularization
        J_JT[0, 0] += 1e-6
        J_JT[1, 1] += 1e-6
        J_JT[2, 2] += 1e-6
        J_JT[3, 3] += 1e-6

        # Solve system
        delta, success = _solve_4x4(J_JT, J_r)
        if not success:
            break

        delta_A = delta[0]
        delta_phi = delta[1]
        delta_dc = delta[2]
        delta_f = delta[3]

        # Apply updates
        A += delta_A
        phi += delta_phi
        dc += delta_dc
        f += delta_f

        # Break early if updates converge to tiny changes
        if (abs(delta_A) < max_precision) and \
           (abs(delta_phi) < max_precision) and \
           (abs(delta_dc) < max_precision) and \
           (abs(delta_f) < max_precision):
            break

    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    return A, phi, dc, f


@njit(cache=True, nogil=True, fastmath=False)
def _demod_burst(
    burst,
    burst_start,
    burst_len,
    burst_sin,
    burst_cos,
    fsc
):
    # get initial burst measurements
    I = 0.0
    Q = 0.0
    burst_sum = 0.0

    for i in range(burst_len):
        burst_sample = burst[i]
        burst_sum += burst_sample
        carrier_idx = i + burst_start
        I += burst_sample * burst_cos[carrier_idx]
        Q += burst_sample * burst_sin[carrier_idx]


    # build starting point for refinement
    phi_guess = (math.atan2(Q, I) + math.pi) % (2.0 * math.pi) - math.pi
    dc_guess = burst_sum / burst_len
    amp_guess = (2.0 * math.sqrt(I * I + Q * Q)) / burst_len

    # refine burst measurements
    burst_amplitude, fit_phi, burst_dc, burst_frequency = _tune_burst_measurements(
        burst, burst_start, amp_guess, phi_guess, dc_guess, fsc
    )

    # Convert the absolute fitted phase shift (radians) into a fractional sample offset
    phase_sample_offset = (fit_phi % (2.0 * math.pi)) * (2.0 / math.pi)

    # Combine the geometric window midpoint with the phase shift
    burst_center_relative = (burst_len - 1) / 2.0 + phase_sample_offset
    burst_center = burst_start + burst_center_relative
    burst_phase_deg = math.degrees(fit_phi) % 360.0
    burst_magnitude = burst_amplitude * (burst_len / 2.0)

    return burst_center, burst_phase_deg, burst_amplitude, burst_magnitude, burst_dc, burst_frequency, I, Q

def _get_upconverted_burst(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    current_phase,
    burst_area,
    burst_sin,
    burst_cos,
    line_number,
    line_offset,
    outwidth,
    fsc
):
    burst_filter_padding = burst_area[0]
    line_start = (line_number - line_offset) * outwidth
    burst_start = max(0, line_start + burst_area[0] - burst_filter_padding)
    burst_end = min(len(chroma), line_start + burst_area[1] + burst_filter_padding)

    upconverted_burst = (
        chroma_heterodyne[current_phase][burst_start:burst_end]
        * chroma[burst_start:burst_end]
    )

    # filter out noise so only the color burst is present
    filtered_padded = sosfiltfilt_rust(chroma_filter, upconverted_burst)
    filtered = filtered_padded[burst_filter_padding:-burst_filter_padding]

    burst_len = len(filtered)
    burst_results = _demod_burst(
        filtered, burst_start + burst_filter_padding, burst_len, burst_sin, burst_cos, fsc
    )

    return BurstInfo(
        line_number,
        burst_start,
        burst_end,
        *burst_results
    )

def _fit_head_switch_rotation(
    phase_sequence,
    chroma,
    chroma_heterodyne,
    chroma_filter,
    chroma_rotation,
    chroma_rotation_index,
    burstarea,
    burst_sin,
    burst_cos,
    fsc,
    lineoffset,
    outwidth,
    rotation_check_start_line,
    reference_lines=24,
    switch_margin=0.5,
):
    """
    Finds the line where the next video track starts at the bottom of a PAL field and
    re-applies the heterodyne phase rotation of that track from there on.

    The lines after the head switch belong to the track of the next field. That track
    uses the other phase rotation, and its starting phase is not related to the phase
    of the current track, so all 4 heterodyne phases have to be considered.

    The burst phase of each line in the head switching area is compared against the phase
    expected from the lines above it (two separate trends for the PAL V-switch line parity).
    Every switch line and starting phase of the next track is scored over the whole area
    at once, so a single noisy burst (common on LP) can not flip the rotation on its own.

    Returns the rotation index of the track active at the end of the field.
    """
    line_count = len(phase_sequence)
    lines = np.array([b.line_number for b in phase_sequence])
    region_start = int(np.searchsorted(lines, rotation_check_start_line))
    region_len = line_count - region_start
    if region_len < 2 or region_start < reference_lines:
        return chroma_rotation_index

    phase = np.radians([b.phase_deg for b in phase_sequence])
    rotation = np.array([b.phase_rotation for b in phase_sequence])
    magnitude = np.array([b.magnitude for b in phase_sequence])

    reference = np.arange(region_start - reference_lines, region_start)
    magnitude_ref = np.median(magnitude[reference])
    if magnitude_ref <= 0:
        return chroma_rotation_index

    region_lines = lines[region_start:]
    expected = np.zeros(region_len)
    for parity in (0, 1):
        ref_parity = reference[lines[reference] % 2 == parity]
        if len(ref_parity) < 3:
            return chroma_rotation_index
        trend = np.polyfit(lines[ref_parity], np.unwrap(phase[ref_parity]), 1)
        region_parity = region_lines % 2 == parity
        expected[region_parity] = np.polyval(trend, region_lines[region_parity])

    # lines without a usable burst (vsync, dropouts) do not count
    weight = np.clip(magnitude[region_start:] / magnitude_ref, 0, 1.5)
    weight[weight < 0.3] = 0

    # burst phase measured with heterodyne phase k is the unrotated phase - k * 90 degrees
    unrotated = phase[region_start:] + rotation[region_start:] * (np.pi / 2)
    candidates = (
        np.cos(
            unrotated[:, None]
            - np.arange(4)[None, :] * (np.pi / 2)
            - expected[:, None]
        )
        * weight[:, None]
    )

    region_idx = np.arange(region_len)
    current_track = np.cumsum(
        np.concatenate(([0.0], candidates[region_idx, rotation[region_start:]]))
    )

    next_rotation_index = (chroma_rotation_index + 1) % 2
    next_track_rotation = chroma_rotation[next_rotation_index]

    best_score = current_track[-1] + switch_margin
    best = None
    for switch in range(region_len):
        following = region_idx[switch:]
        steps = np.arange(region_len - switch)
        for start_phase in range(4):
            next_phases = (start_phase + next_track_rotation * steps) % 4
            score = current_track[switch] + candidates[following, next_phases].sum()
            if score > best_score:
                best_score = score
                best = (switch, start_phase)

    if best is None:
        return chroma_rotation_index

    switch, start_phase = best
    phases = rotation[region_start:].copy()
    phases[switch:] = (start_phase + next_track_rotation * np.arange(region_len - switch)) % 4

    # Lines right at the switch do not always follow either track (mid-line switch,
    # transients of the new head), use their own burst there if it is clearly off.
    def phase_error(k):
        return np.abs((unrotated - k * (np.pi / 2) - expected + np.pi) % (2 * np.pi) - np.pi)

    strong = weight >= 0.5
    for j in range(max(0, switch - 2), region_len):
        if strong[j] and phase_error(phases[j])[j] > np.radians(67.5):
            errors = [phase_error(k)[j] for k in range(4)]
            if min(errors) < np.radians(45):
                phases[j] = int(np.argmin(errors))

    # The new head often has a fixed chroma phase offset to the old one (not a multiple
    # of 90 degrees), which shows as a hue change at the bottom of every other field.
    following = np.arange(switch, region_len)
    following = following[strong[following]]
    phase_correction = 0.0
    if len(following) >= 2:
        residual = unrotated[following] - phases[following] * (np.pi / 2) - expected[following]
        phase_correction = np.angle(np.sum(weight[following] * np.exp(1j * residual)))
        if abs(phase_correction) < np.radians(3):
            phase_correction = 0.0

    for j in range(max(0, switch - 2), region_len):
        idx = region_start + j
        next_phase = int(phases[j])
        if next_phase == rotation[idx] and j < switch:
            continue
        burst = _get_upconverted_burst(
            chroma,
            chroma_heterodyne,
            chroma_filter,
            next_phase,
            burstarea,
            burst_sin,
            burst_cos,
            lines[idx],
            lineoffset,
            outwidth,
            fsc,
        )
        burst.phase_rotation = next_phase
        if j >= switch:
            burst.phase_correction = phase_correction
        phase_sequence[idx] = burst

    return next_rotation_index


def _get_phase_sequence(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    chroma_rotation,
    chroma_rotation_starting_index,
    burstarea,
    burst_sin,
    burst_cos,
    fsc,
    lineoffset,
    outwidth,
    last_line,
    detect_chroma_track_phase,
    rotation_check_start_line,
    track_change_threshold,
    color_system
):
    do_phase_rotation_check = (
        detect_chroma_track_phase
        and chroma_rotation is not None
        and chroma_heterodyne is not None
    )

    # PAL uses a model fit over the whole head switching area instead of the line by line check below
    fit_head_switch = do_phase_rotation_check and color_system == "PAL"
    if fit_head_switch:
        do_phase_rotation_check = False

    phase_sequence = []

    if chroma_rotation_starting_index is None:
        # first field
        chroma_rotation_starting_index = 0
        chroma_rotation_index = 0

    if chroma_rotation:
        # color under format that uses a phase rotated heterodyne to down convert the composite chroma
        chroma_rotation_index = chroma_rotation_starting_index
        track_rotation = chroma_rotation[chroma_rotation_index]
    else:
        # format that uses a fixed heterodyne phase, or does not rotate
        chroma_rotation_index = 0
        track_rotation = chroma_rotation_starting_index
    """
    "...a signal that represents phase zero with respect to the chroma signal phase 
    +90°, +180°, +270° etc. or a phase 0°, -90°. —180°. —270°. etc., 
    depending upon which head is on the tape at the particular time.

    The direction of phase rotation, being related to which head is on the tape at a given time,
    can be determined and preset by sensing whether the PG (pulse generator) pulse is positive-going or negative-going."
     - https://archive.org/details/rca-vcr-1-red-book-w-cover/page/n25/mode/2up?q=phase

    See also: https://archive.org/details/video-technical-guide/page/1-9/mode/2up?q=phase

    The phase rotation switch is determined at record time depending on which video head is on the tape.
    This rotation switch can occur in the middle of a line, causing a small phase artifact
    TODO: It may be possible to detect where this happens on the line and correct the phase issue mid-line
          Possibly a 2D aware detection could be used to determine where the color phase is rotated +-90 degrees relative to the lines above and below
    """

    current_phase = 0
    use_next_phase = False
    for linenumber in range(lineoffset, last_line):
        if use_next_phase:
            # reuse the calculated phase from the previous iteration
            current_phase = next_phase
            current_burst = next_burst

            use_next_phase = False
        else:
            current_phase = (current_phase + track_rotation) % 4
            current_burst = _get_upconverted_burst(
                chroma,
                chroma_heterodyne,
                chroma_filter,
                current_phase,
                burstarea,
                burst_sin,
                burst_cos,
                linenumber,
                lineoffset,
                outwidth,
                fsc
            )

        # check if the track has rotated around the head switching area
        if (
            do_phase_rotation_check
            and linenumber >= rotation_check_start_line
            and linenumber < last_line - 1
        ):
            # get the next burst using the phase rotation for the current track
            next_phase = (current_phase + track_rotation) % 4
            next_burst = _get_upconverted_burst(
                chroma,
                chroma_heterodyne,
                chroma_filter,
                next_phase,
                burstarea,
                burst_sin,
                burst_cos,
                linenumber + 1,
                lineoffset,
                outwidth,
                fsc
            )

            if color_system == "NTSC":
                # check one line back
                comparison_burst: BurstInfo = current_burst
            else: # color_system in ("PAL", "PAL_M", "NLINHA", "MESECAM")
                # check two lines back
                comparison_burst: BurstInfo = phase_sequence[-1]

            phase_delta_quadrant = abs(
                (next_burst.phase_deg - comparison_burst.phase_deg + 180) % 360 - 180
            )
            if phase_delta_quadrant > track_change_threshold:
                # burst is more in phase than out of phase, flip rotation so it remains out of phase
                chroma_rotation_index = (chroma_rotation_index + 1) % 2
                track_rotation = chroma_rotation[chroma_rotation_index]
            else:
                use_next_phase = True

        current_burst.phase_rotation = current_phase
        phase_sequence.append(current_burst)

    if fit_head_switch:
        chroma_rotation_index = _fit_head_switch_rotation(
            phase_sequence,
            chroma,
            chroma_heterodyne,
            chroma_filter,
            chroma_rotation,
            chroma_rotation_index,
            burstarea,
            burst_sin,
            burst_cos,
            fsc,
            lineoffset,
            outwidth,
            rotation_check_start_line,
        )

    if chroma_rotation and chroma_rotation_index == chroma_rotation_starting_index:
        # rotate the phase for the next field, if rotation was not detected
        chroma_rotation_index = (chroma_rotation_index + 1) % 2

    return chroma_rotation_index, phase_sequence


def get_phase_rotation_sequence(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    chroma_rotation,
    chroma_rotation_index,
    lineoffset,
    linesout,
    outwidth,
    burstarea,
    burst_sin,
    burst_cos,
    fsc,
    detect_chroma_track_phase,
    rotation_check_start_line,
    enable_color_killer,
    prev_burst_detected_line,
    color_system,
):
    # Detects the correct color-under heterodyne starting phase and rotation direction
    # Additional for NTSC, this function calculates the color burst average for burst-locked TBC later on
    track_change_threshold = 90
    burst_check_skip_lines = 16

    # TODO Expose as option, possible this needs to be relative to the sync pulse and level detection
    burst_magnitude_threshold = 2.5e4

    end = linesout + lineoffset

    chroma_rotation_index, phase_sequence = _get_phase_sequence(
        chroma,
        chroma_heterodyne,
        chroma_filter,
        chroma_rotation,
        chroma_rotation_index,
        burstarea,
        burst_sin,
        burst_cos,
        fsc,
        lineoffset,
        outwidth,
        end,
        detect_chroma_track_phase,
        rotation_check_start_line,
        track_change_threshold,
        color_system
    )

    burst_check_start = burst_check_skip_lines
    burst_check_end = end - burst_check_skip_lines
    burst_detected_line = 0 # color enabled by default

    if chroma_rotation:
        # detect relative phase difference between lines
        delta_0 = 0
        delta_90 = 0
        delta_180 = 0
        delta_270 = 0

        for i in range(1, len(phase_sequence)):
            previous_burst = phase_sequence[i-1]
            current_burst = phase_sequence[i]

            if current_burst.line_number > burst_check_start and current_burst.line_number < burst_check_end:
                delta = (current_burst.phase_deg - previous_burst.phase_deg) % 360
                bucket = int((delta + 45) // 90) % 4

                if bucket == 0:
                    delta_0 += 1
                elif bucket == 1:
                    delta_90 += 1
                elif bucket == 2:
                    delta_180 += 1
                else:
                    delta_270 += 1

        if color_system == "NTSC":
            # if the bursts are out of phase with each other, the track was miss-detected, flip phase and recalculate sequence
            flip_track_phase = delta_0 < delta_180
        else:  # color_system in ("PAL", "PAL_M", "NLINHA", "MESECAM")
            # each line should alternate phase, if there are repeated sequences of phase, recalculate
            alt1 = delta_90 + delta_270
            alt2 = delta_0 + delta_180

            # choose whichever pattern dominates
            flip_track_phase = alt1 < alt2
    else:
        # no difference between track phases, do not flip
        flip_track_phase = False

    if flip_track_phase:
        # recalculate with the corrected track rotation
        chroma_rotation_index, phase_sequence = _get_phase_sequence(
            chroma,
            chroma_heterodyne,
            chroma_filter,
            chroma_rotation,
            chroma_rotation_index,
            burstarea,
            burst_sin,
            burst_cos,
            fsc,
            lineoffset,
            outwidth,
            end,
            detect_chroma_track_phase,
            rotation_check_start_line,
            track_change_threshold,
            color_system
        )

    # calculate the average color phase for even and odd lines
    even_I_total = 0
    even_Q_total = 0
    odd_I_total = 0
    odd_Q_total = 0

    avg_count = 0
    burst_magnitude_avg = 0

    for burst in phase_sequence:
        if burst.line_number > burst_check_start and burst.line_number < burst_check_end:
            I = burst.I
            Q = burst.Q

            if burst.magnitude != 0:
                I /= burst.magnitude
                Q /= burst.magnitude

                avg_count += 1
                burst_magnitude_avg += burst.magnitude

                if enable_color_killer:
                    # find the first line that might have a valid burst if the previous field had the burst disabled
                    # broadcasters would sometime turn on the burst mid-field, so attempt to detect that transition here
                    if (
                        prev_burst_detected_line == -1 # previous field had color killer activated
                        and burst_detected_line == 0 and burst.magnitude > burst_magnitude_threshold # first burst that exceeds threshold
                    ):
                        # first burst that exceeds threshold
                        # color killer will be active until this line, then it deactivates
                        # it is only reactivated after an entire field is without color (below)
                        burst_detected_line = burst.line_number
            
                if burst.line_number % 2:
                    odd_I_total += I
                    odd_Q_total += Q
                else:
                    even_I_total += I
                    even_Q_total += Q
    
    burst_magnitude_avg /= avg_count

    if enable_color_killer:
        if burst_magnitude_avg < burst_magnitude_threshold:
            # (re)activate color killer for the entire field
            burst_detected_line = -1

    burst_phase_avg = np.degrees(np.arctan2(even_Q_total + odd_Q_total, even_I_total + odd_I_total)) % 360
    even_burst_phase_avg = np.degrees(np.arctan2(even_Q_total, even_I_total)) % 360
    odd_burst_phase_avg = np.degrees(np.arctan2(odd_Q_total, odd_I_total)) % 360

    return chroma_rotation_index, phase_sequence, burst_detected_line, burst_magnitude_avg, burst_phase_avg, even_burst_phase_avg, odd_burst_phase_avg


@njit(cache=False, nogil=True, fastmath=False)
def upconvert_chroma(
    chroma,
    uphet,
    lineoffset,
    outwidth,
    phase_rotation_sequence,
    chroma_heterodyne,
):
    for burst in phase_rotation_sequence:
        # The burst lock runs before PAL fields get their lineoffset, so the
        # sequence starts with entries for lines above the first output line.
        # Those would index backwards from the end of the field.
        if burst.line_number < lineoffset:
            continue

        linestart = (burst.line_number - lineoffset) * outwidth
        lineend = linestart + outwidth

        heterodyne = chroma_heterodyne[burst.phase_rotation][linestart:lineend]
        c = chroma[linestart:lineend]
        if burst.phase_correction != 0.0:
            # the next heterodyne phase is 90 degrees further, mix both for the fractional step
            quadrature = chroma_heterodyne[(burst.phase_rotation + 1) % 4][linestart:lineend]
            uphet[linestart:lineend] = c * (
                heterodyne * math.cos(burst.phase_correction)
                + quadrature * math.sin(burst.phase_correction)
            )
        else:
            uphet[linestart:lineend] = c * heterodyne


@njit(nogil=True, cache=False, fastmath=False)
def upconvert_chroma_phase_comp(
    chroma,
    lineoffset,
    outwidth,
    phase_rotation_sequence,
    color_under_carrier_fs,
    fsc,
    target_phase_even,
    target_phase_odd
):
    deg2rad_scale = np.pi / 180.0
    pi_over_two = np.pi / 2.0

    # Initial nominal reference coefficient
    het_hz_nominal = color_under_carrier_fs
    het_coefficient = pi_over_two * (1.0 + het_hz_nominal / fsc)

    target_phase_even_rad = target_phase_even * deg2rad_scale
    target_phase_odd_rad = target_phase_odd * deg2rad_scale
    num_bursts = len(phase_rotation_sequence)

    coeff_step_factor = pi_over_two / (fsc * outwidth)

    # Pre-generate a local pixel coordinate array to help Numba vectorize
    # Computing on a local range [0, outwidth) helps the compiler reason about alignment
    local_idx = np.arange(outwidth, dtype=np.float64)

    for idx in range(num_bursts):
        current_burst = phase_rotation_sequence[idx]

        # Entries for lines above the first output line would index backwards
        # from the end of the field (see upconvert_chroma).
        if current_burst.line_number < lineoffset:
            continue

        # Solve for current line's active het_hz
        k_current = current_burst.frequency / fsc
        het_hz_current = k_current * color_under_carrier_fs

        # Solve for next line's active het_hz
        if idx < num_bursts - 1:
            next_burst = phase_rotation_sequence[idx + 1]
            k_next = next_burst.frequency / fsc
            het_hz_next = k_next * color_under_carrier_fs
        else:
            het_hz_next = het_hz_current

        linestart = (current_burst.line_number - lineoffset) * outwidth
        lineend = linestart + outwidth

        # Determine target phase
        if current_burst.line_number % 2 != 0:
            target_phase_rad = target_phase_odd_rad
        else:
            target_phase_rad = target_phase_even_rad

        # Initial starting phase
        theta_0 = het_coefficient * linestart + (
            current_burst.phase_rotation * pi_over_two
            + target_phase_rad 
            + current_burst.phase_deg * deg2rad_scale
        )

        # Coefficient step parameters
        alpha = pi_over_two * (1.0 + het_hz_current / fsc)
        delta_coeff = (het_hz_next - het_hz_current) * coeff_step_factor
        beta = 0.5 * delta_coeff
        dc_val = current_burst.dc

        # Slice target and source arrays to provide direct contiguous memory views
        chroma_slice = chroma[linestart:lineend]

        # No outer dependencies so LLVM can vectorize
        for k in range(outwidth):
            # Compute closed-form phase directly (no dependency on previous k)
            theta_k = theta_0 + alpha * local_idx[k] + beta * (local_idx[k] * local_idx[k])
            
            # Vectorized trigonometric and arithmetic execution
            chroma_slice[k] = chroma_slice[k] * -np.cos(theta_k) - dc_val


@njit(cache=True, nogil=True)
def burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea):
    for line in range(lineoffset, linesout + lineoffset):
        linestart = (line - lineoffset) * outwidth
        lineend = linestart + outwidth

        chroma[linestart + burstarea[1] + 4 : lineend] *= 2

    return chroma


def decode_chroma_phase_rotation(
    field,
    disable_tracking_cafc=False,
    chroma_rotation=None,
    detect_chroma_track_phase=False,
):
    chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst")

    lineoffset = field.lineoffset + 1
    linesout = field.outlinecount
    outwidth = field.outlinelen

    burstarea = get_burst_area(field)
    rotation_check_start_line = lineoffset + linesout - 16

    # Rotation per track
    # VHS PAL:      Track1 0,   Track2 -90
    # VHS NTSC:     Track1 +90, Track2 -90
    # Betamax PAL:  None - uses frequency offset instead
    # Betamax NTSC: Track1 180, Track2 0
    # Video8 PAL:   Track1 0,   Track2 -90
    # Video8 NTSC:  Track1 0,   Track2 180

    chroma_heterodyne = (
        field.rf.chroma_afc.getLatchedChromaHet()
        if (field.rf.do_cafc and not disable_tracking_cafc)
        else field.rf.chroma_heterodyne
    )

    prev_burst_detected_line = 0
    if field.prevfield is not None:
        prev_burst_detected_line = field.prevfield.burst_detected_line

    track_phase, phase_sequence, burst_detected_line, burst_magnitude_avg, burst_phase_avg, even_burst_phase_avg, odd_burst_phase_avg = get_phase_rotation_sequence(
        chroma,
        chroma_heterodyne,
        field.rf.Filters["FChromaFinal"],
        chroma_rotation,
        field.rf.track_phase, # index for chroma rotation, and static if there is no chroma rotation
        lineoffset,
        linesout,
        outwidth,
        burstarea,
        field.rf.fsc_wave,
        field.rf.fsc_cos_wave,
        field.rf.SysParams['fsc_mhz'] * 1e6,
        detect_chroma_track_phase,
        rotation_check_start_line, # check for track phase rotation around the headswitching area (bottom of field)
        field.rf.options.enable_color_killer,
        prev_burst_detected_line,
        field.rf.color_system,
    )

    return track_phase, phase_sequence, burst_detected_line, burst_magnitude_avg, burst_phase_avg, even_burst_phase_avg, odd_burst_phase_avg


ntsc_color_framing_phase_shift = 33
ntsc_color_framing_map = {
    # Color Frame I
    (1, 0): (1, 0 - ntsc_color_framing_phase_shift),
    (0, 1): (2, 180 - ntsc_color_framing_phase_shift),
    # Color Frame II
    (1, 1): (3, 180 - ntsc_color_framing_phase_shift),
    (0, 0): (4, 0 - ntsc_color_framing_phase_shift),
}

# fieldPhaseID, even_burst_phase, odd_burst_phase
pal_offset_I   = -90*1
pal_offset_II  = -90*2
pal_offset_III = -90*3
pal_offset_IV  = -90*4
pal_phase_swing = 135

# Rec. ITU-R BT.1700, pp.6 (phase poliarity 525 and 625 PAL)
# Field         |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |
# Color frame   |   I |  II | III |  IV |   I |  II | III |  IV |
# Even polarity |   - |   - |   + |   + |   - |   - |   + |   + |
# Odd  polarity |   + |   + |   - |   - |   + |   + |   - |   - |

# first_field, has_line_6_burst, frame_number 0-3 or 4-7
pal_color_framing_map = {
    (1, 0, 0): (1, -pal_phase_swing + pal_offset_I,    pal_phase_swing + pal_offset_I), #   field 1, Color Frame I
    (0, 1, 0): (2, -pal_phase_swing + pal_offset_II,   pal_phase_swing + pal_offset_II), #  field 2, Color Frame II
    (1, 1, 0): (3,  pal_phase_swing + pal_offset_III, -pal_phase_swing + pal_offset_III), # field 3, Color Frame III
    (0, 0, 0): (4,  pal_phase_swing + pal_offset_IV,  -pal_phase_swing + pal_offset_IV), #  field 4, Color Frame IV
    (1, 0, 1): (5, 180 + -pal_phase_swing + pal_offset_I,   180 +  pal_phase_swing + pal_offset_I), #   field 5, Color Frame I
    (0, 1, 1): (6, 180 + -pal_phase_swing + pal_offset_II,  180 +  pal_phase_swing + pal_offset_II), #  field 6, Color Frame II
    (1, 1, 1): (7, 180 +  pal_phase_swing + pal_offset_III, 180 + -pal_phase_swing + pal_offset_III), # field 7, Color Frame III
    (0, 0, 1): (8, 180 +  pal_phase_swing + pal_offset_IV,  180 + -pal_phase_swing + pal_offset_IV), #  field 8, Color Frame IV
}


@cache
def _gen_chroma_fft_filter(
    filter_len: int,
    fsc: float,
    color_under_carrier_f: float,
    bw_lower_hz: float,
    heterodyne_attenuation_db: float,
    order: int,
) -> np.ndarray:
    """
    Generate asymmetric Super-Gaussian bandpass mask in the frequency domain.
    """

    # calculate upper bandwidth limit that is required to remove the heterodyne up conversion product
    # at the supplied attenuation
    A = 10.0 ** (-abs(heterodyne_attenuation_db) / 20.0)
    delta_f = 2.0 * color_under_carrier_f
    exponent = 1.0 / (2.0 * order)
    bw_upper_hz = delta_f / ((-np.log(A)) ** exponent)

    freqs_up = sps_fft.rfftfreq(filter_len, d=1.0 / (fsc * 4.0))
    mask = np.zeros_like(freqs_up, dtype=np.float64)

    lower_idx = freqs_up <= fsc
    upper_idx = freqs_up > fsc

    # asymmetric Super-Gaussian evaluation around subcarrier (fsc)
    mask[lower_idx] = np.exp(-((freqs_up[lower_idx] - fsc) / bw_lower_hz) ** (2 * order))
    mask[upper_idx] = np.exp(-((freqs_up[upper_idx] - fsc) / bw_upper_hz) ** (2 * order))

    return mask


def filter_chroma_fft(
    uphet: np.ndarray, 
    fsc: float,
    color_under_carrier_f: float,
    bw_lower_hz: float,               # Lower chroma bandwidth (1.3 MHz below fsc)
    heterodyne_attenuation_db: float, # Rejection target at sum product (dB)
    order: int = 2,                   # filter order
    pad_samples: int = 256,
) -> np.ndarray:
    """
    Zero-phase FFT bandpass filter using a Super-Gaussian mask.
    Dynamically computes bw_upper_hz from color_under_carrier_f to guarantee
    stopband attenuation at the heterodyne sum product.
    """
    N_raw = len(uphet)

    # pad to nearest fast FFT length (combines 2, 3, 5, 7 prime factors)
    min_pad_len = N_raw + 2 * max(1, int(pad_samples))
    N_up = sps_fft.next_fast_len(min_pad_len)

    # Symmetric pad calculation to center the signal in the fast length array
    pad_left = (N_up - N_raw) // 2
    pad_right = N_up - N_raw - pad_left

    x_padded = np.pad(uphet, (pad_left, pad_right), mode='reflect')

    mask = _gen_chroma_fft_filter(
        N_up,
        fsc,
        color_under_carrier_f,
        bw_lower_hz,
        heterodyne_attenuation_db,
        order
    )

    # apply filter against Forward Real FFT
    F_filtered = sps_fft.rfft(x_padded)
    F_filtered *= mask

    # return to real signal
    chroma_padded = sps_fft.irfft(F_filtered, n=N_up)

    # remove padding
    return chroma_padded[pad_left : pad_left + N_raw]


@njit(cache=True, fastmath=True, nogil=True)
def chroma_transient_improvement(
    chroma_data: np.ndarray,
    line_start: int,
    line_length: int,
    base_noise_floor: float,
    cti_width: int,
    cti_mix: float,
) -> np.ndarray:
    """
    Accelerates the sweep rate between color states without warping phase.
    Operates symmetrically by measuring forward/backward vector neighbors simultaneously.
    """
    # Configure geometric multi-pass decay
    decay = 0.25
    num_passes = 4

    # Pre-calculate mix factors
    mix_factors = np.empty(num_passes, dtype=np.float32)
    for p in range(num_passes):
        mix_factors[p] = cti_mix * (decay ** p)

    # Establish spatial boundaries
    remaining_samples = chroma_data.shape[0] - line_start
    line_count = remaining_samples // line_length
    
    # Establish the 4fsc phase-locked sweep radius and scaling threshold
    sweep_radius = int(max(4, cti_width * 4))
    mad_threshold = base_noise_floor * math.sqrt(cti_width)

    # Protect against edge bleeding
    start_s = sweep_radius + 1
    end_s = line_length - (sweep_radius + 1)

    line_buffer = np.empty(line_length, dtype=chroma_data.dtype)

    # --- Optimized Vector-Ready Loop ---
    for l in range(line_count):
        curr_line_offset = line_start + (l * line_length)

        for p in range(num_passes):
            current_mix = mix_factors[p]
            line_buffer[:] = chroma_data[curr_line_offset : curr_line_offset + line_length]

            for s in range(start_s, end_s):
                idx = curr_line_offset + s
                
                i_curr = line_buffer[s]
                q_curr = line_buffer[s - 1]
                
                i_past = line_buffer[s - sweep_radius]
                q_past = line_buffer[s - sweep_radius - 1]
                
                i_future = line_buffer[s + sweep_radius]
                q_future = line_buffer[s + sweep_radius - 1]
                
                # Measure vector distances
                i_delta_back = i_curr - i_past
                q_delta_back = q_curr - q_past
                dist_back = math.sqrt(i_delta_back * i_delta_back + q_delta_back * q_delta_back)
                
                i_delta_forw = i_future - i_curr
                q_delta_forw = q_future - q_curr
                dist_forw = math.sqrt(i_delta_forw * i_delta_forw + q_delta_forw * q_delta_forw)
                
                total_sweep_distance = dist_back + dist_forw
                
                # Noise Gate
                gate_mask = 1.0 if total_sweep_distance > mad_threshold else 0.0
                
                # Inherent Boundary (Naturally bounds to [0.0, 1.0) because distances are >= 0)
                norm_progress = dist_back / total_sweep_distance if total_sweep_distance != 0 else 0
                
                # Transform the progress via sigmoidal sweep-acceleration function
                is_lower = norm_progress < 0.5
                
                inv_prog = 1.0 - norm_progress
                t_low = 4.0 * (norm_progress * norm_progress)
                t_high = 1.0 - 4.0 * (inv_prog * inv_prog)
                
                # Select interpolation weights and anchor to the closer sample (before or after)
                t = t_low if is_lower else t_high
                anchor_a = i_past if is_lower else i_curr
                anchor_b = i_curr if is_lower else i_future
                
                # 4. Final vector mix
                # If gate_mask is 0.0, the delta cancels out and i_curr is cleanly written back
                i_target = anchor_a + t * (anchor_b - anchor_a)
                chroma_data[idx] = i_curr + (current_mix * gate_mask) * (i_target - i_curr)


def get_burst_area(field):
    burst_start = math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0])) - 4
    burst_end = math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])) + 8

    # burst length must be multiple of 4
    burst_end = burst_end - ((burst_end - burst_start) % 4)

    return burst_start, burst_end
