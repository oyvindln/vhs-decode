import math
import numpy as np
import lddecode.core as ldd
import scipy.signal as sps
import scipy.fft as sps_fft
from vhsdecode.rust_utils import sosfiltfilt_rust
from vhsdecode.chroma.secam import (
    _process_chroma_secam_method1,
    measure_secam_under_carrier_offset,
)

import numba
from numba import njit
from numba.experimental import jitclass
from functools import cache


@njit(cache=True, nogil=True, fastmath=True)
def chroma_to_u16(chroma):
    """
    Scale the chroma output array to a 16-bit value for output.
    """
    S16_ABS_MAX = 32767.0
    N = len(chroma)

    out = np.empty(N, dtype=np.uint16)

    for i in range(N):
        out[i] = np.uint16(chroma[i] + S16_ABS_MAX)
        
    return out

@njit(cache=False, nogil=True, fastmath=True)
def chroma_automatic_gain(
    chroma,
    burst_abs_ref,
    phase_sequence,
    burst_detected_line,
    sync_tip_len,
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
        current_burst_start = current_burst.start

        if i < burst_count - 1:
            next_burst_start = phase_sequence[i + 1].start
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
        linestart = (burst.line_number - lineoffset) * outwidth
        lineend = linestart + outwidth

        heterodyne = chroma_heterodyne[burst.phase_rotation][linestart:lineend]
        c = chroma[linestart:lineend]
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


@njit(cache=True, nogil=True, fastmath=False)
def shift_chroma_and_remove_dc(out_chroma, move):
    n = len(out_chroma)
    move %= n
    
    mean_acc = 0

    # save wrapped values
    tmp = np.empty(move, dtype=out_chroma.dtype)

    for i in range(move):
        tmp[i] = out_chroma[n - move + i]

    # single pass shift
    for i in range(n - move - 1, -1, -1):
        mean_acc += out_chroma[i]
        out_chroma[i + move] = out_chroma[i]

    # small wrap-around copy
    for i in range(move):
        mean_acc += tmp[i]
        out_chroma[i] = tmp[i]

    mean_acc /= n

    # crude DC offset removal
    for i in range(n):
        out_chroma[i] -= mean_acc


def chroma_color_under_filter(
    data, filter, blocklen, notch, do_notch=None, move=10, audio_notch=None
):
    out_chroma = sosfiltfilt_rust(filter, data[:blocklen])

    if audio_notch is not None:
        out_chroma = sps.filtfilt(
            audio_notch[0],
            audio_notch[1],
            out_chroma,
        )

    if do_notch is not None and do_notch:
        out_chroma = sps.filtfilt(
            notch[0],
            notch[1],
            out_chroma,
        )

    # Move chroma to compensate for Y filter delay.
    # value needs tweaking, ideally it should be calculated if possible.
    # TODO: Not sure if we need this after hilbert filter change, needs check.
    shift_chroma_and_remove_dc(out_chroma, move)

    return out_chroma


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
        field.rf.chroma_afc.getChromaHet()
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


def process_chroma(
    field,
    disable_deemph=False,
    disable_comb=False,
    disable_tracking_cafc=False,
    do_chroma_deemphasis=False,
):
    lineoffset = field.lineoffset + 1
    linesout = field.outlinecount
    outwidth = field.outlinelen

    if field.burst_detected_line == -1:
        # skip chroma if the color killer is active for the whole field
        return np.zeros((linesout * outwidth), dtype=np.float32)
    
    if (
        not field.rf.options.disable_phase_correction
        and field.rf.color_system == "NTSC"
    ):
        field.fieldPhaseID, target_phase = ntsc_color_framing_map[
            (field.isFirstField, (field.field_number // 2) % 2)
        ]
        chroma_shift_direction = 1 if target_phase else -1
    else:
        chroma_shift_direction = 0

    # Run TBC/downscale on chroma (if new field, else uses cache)
    # Cached if chroma process is run multiple times on one field due to track detection.
    if field.chroma_tbc_buffer is None:
        # shift the chroma to reverse group delay caused by the color under heterodyne filter
        # this is dependent on color framing, and is disabled if color framing is disabled
        # TODO: shift amount may need tuning / needs validation
        chroma_subcarrier_delay_cycles = field.rf.SysParams['fsc_mhz'] * 1e6 / (2.0 * np.pi * field.rf.DecoderParams["color_under_carrier"])
        chroma_subcarrier_delay_samples = chroma_subcarrier_delay_cycles * 4
        chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst", shift=chroma_subcarrier_delay_samples * chroma_shift_direction)

        # If chroma AFC is enabled
        if field.rf.do_cafc:
            # it does the chroma filtering AFTER the TBC
            chroma = chroma_color_under_filter(
                chroma,
                field.rf.chroma_afc.get_chroma_bandpass(),
                len(chroma),
                field.rf.Filters["FVideoNotch"],
                field.rf.notch,
                move=(int(10 * (field.rf.sys_params["outfreq"] / 40))),
                audio_notch=field.rf.Filters.get("FChromaAudioNotch", None),
            )

            if not disable_tracking_cafc:
                spec, meas, offset, cphase = field.rf.chroma_afc.freqOffset(chroma)
                ldd.logger.debug(
                    "Chroma under AFC: %.02f kHz, Offset (long term): %.02f Hz, Phase: %.02f deg"
                    % (meas / 1e3, offset, cphase * 360 / (2 * np.pi))
                )

        if (
            field.rf.color_system == "MESECAM"
            and field.rf.options.secam_carrier_servo
        ):
            # Measure the rest carrier pair on the late back porch,
            # 3.7 to 0.3 us before active video starts.
            active_start_px = field.usectooutpx(field.rf.SysParams["activeVideoUS"][0])
            porch_window = (int(active_start_px) - 65, int(active_start_px) - 5)

            carrier_offset = measure_secam_under_carrier_offset(
                chroma,
                linesout,
                outwidth,
                porch_window,
                field.rf.chroma_afc.true_samp_rate,
                field.rf.DecoderParams["color_under_carrier"],
            )
            if carrier_offset is not None:
                field.rf.secam_servo_avg.push(carrier_offset)
                ldd.logger.debug(
                    "SECAM carrier servo: measured offset %.02f Hz" % carrier_offset
                )

        field.rf.chroma_tbc_buffer = chroma
        field.chroma_tbc_buffer = chroma
    else:
        chroma = field.chroma_tbc_buffer

    burstarea = get_burst_area(field)

    if field.rf.color_system == "SECAM":
        # Method 1 restores the chroma block by phase multiplication rather
        # than by mixing against a heterodyne, so it skips the shared
        # up-conversion path below entirely.
        return _process_chroma_secam_method1(
            field, chroma, linesout, outwidth, burstarea
        )

    # For NTSC, the color burst amplitude is doubled when recording, so we have to undo that.
    if field.rf.color_system == "NTSC":
        if not disable_deemph:
            chroma = burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea)

    if (
        not field.rf.options.disable_phase_correction
        and field.rf.color_system == "NTSC"
    ):
        target_phase_even = target_phase
        target_phase_odd = target_phase

        # TODO: PAL color framing is disabled for now.
        #       need to find a reliable way to detect if this is field 1,2 vs 3,4
        # if field.rf.color_system == "PAL":
        #     line_6_burst_present = field.phase_sequence[4 + lineoffset][3] > field.burst_magnitude_avg / 3
        #     field.fieldPhaseID, target_phase_even, target_phase_odd = pal_color_framing_map[
        #         (field.isFirstField, line_6_burst_present, (field.field_number // 4) % 2)
        #     ]

        # this uses the burst measurements to interpolate the correct phase of the color under heterodyne
        # phase issues are corrected continiously for each sample using a linear spline interpolated from the burst measurements
        # the mixing is performed on the upsampled signal to avoid aliasing introduced from the up-heterodyne mixing product
        upconvert_chroma_phase_comp(
            chroma, # modifies this in place
            lineoffset,
            outwidth,
            field.phase_sequence,
            field.rf.DecoderParams["color_under_carrier"],
            field.rf.SysParams["fsc_mhz"] * 1e6,
            target_phase_even,
            target_phase_odd,
        )
        uphet = chroma
    else:
        if field.rf.chroma_afc.conversion_lo is not None:
            # Explicit conversion LO (ME-SECAM): trim it by the smoothed
            # measured carrier offset (cancelling the recording VCR's
            # converter crystal error), and keep the heterodyne phase
            # continuous across fields.
            lo_trim = 0.0
            # Holds either live servo measurements or a seeded/fixed trim
            # (secam_lo_trim); with the servo disabled and no seed it's empty.
            if field.rf.secam_servo_avg.has_values():
                # Quantize so measurement noise doesn't dither the LO.
                lo_trim = np.clip(
                    round(field.rf.secam_servo_avg.pull() / 10.0) * 10.0,
                    -10e3,
                    10e3,
                )
            field.rf.chroma_afc.updateConversion(
                lo_trim, field.field_number * linesout * outwidth
            )
            chroma_heterodyne = field.rf.chroma_afc.getChromaHet()
        else:
            chroma_heterodyne = (
                field.rf.chroma_afc.getChromaHet()
                if (field.rf.do_cafc and not disable_tracking_cafc)
                else field.rf.chroma_heterodyne
            )

        uphet = np.zeros((linesout * outwidth), dtype=np.float32)
        upconvert_chroma(
            chroma,
            uphet,
            lineoffset,
            outwidth,
            field.phase_sequence,
            chroma_heterodyne
        )

    # Filter out unwanted frequencies from the final chroma signal.
    # Mixing the signals will produce waves at the difference and sum of the
    # frequencies. We only want the difference wave which is at the correct color
    # carrier frequency here.
    if field.rf.color_system == "MESECAM":
        # The restored SECAM FM block is anchored at conversion_lo -
        # color_under (4.328125 MHz), not fsc, so the fsc-anchored FFT mask
        # sits ~106 kHz high on it and loses the tight top edge that
        # suppresses high-side FM splatter from saturated transitions. Keep
        # the block-anchored Butterworth here.
        uphet = sosfiltfilt_rust(field.rf.Filters["FChromaFinal"], uphet)
    else:
        uphet = filter_chroma_fft(
            uphet,
            field.rf.SysParams["fsc_mhz"] * 1e6,
            field.rf.DecoderParams["color_under_carrier"],
            1.3e6, # lower chroma bandwidth (roughly this for PAL / NTSC)
            80.0   # heterodyne up-mixing attenuation
        )

    if do_chroma_deemphasis:
        b, a = field.rf.Filters["chroma_deemphasis"]
        uphet = sps.lfilter(b, a, uphet)

    # Basic comb filter for NTSC to calm the color a little.
    if not disable_comb:
        if field.rf.color_system == "NTSC":
            uphet = comb_c_ntsc(uphet, outwidth)
        else:
            uphet = comb_c_pal(uphet, outwidth)

    # Chroma AGC
    mean_rms, chroma_noise_floor = chroma_automatic_gain(
        uphet,
        field.rf.SysParams["burst_abs_ref"],
        field.phase_sequence,
        field.burst_detected_line,
        math.floor(field.usectooutpx(field.rf.SysParams["hsyncPulseUS"]))
    )

    field.rf.field_averages.chroma_level.push(mean_rms)

    if field.rf.options.cti_mix != 0:
        chroma_transient_improvement(
            uphet,
            lineoffset * outwidth,
            outwidth,
            chroma_noise_floor,
            field.rf.options.cti_width,
            field.rf.options.cti_mix,
        )

    return uphet


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


def decode_chroma(field, do_chroma_deemphasis=False):
    if field.rf.options.write_chroma:
        """Do track detection if needed and upconvert the chroma signal"""
        field.chroma_tbc_buffer = None

        uphet = process_chroma(
            field,
            disable_comb=field.rf.options.disable_comb,
            disable_tracking_cafc=False,
            do_chroma_deemphasis=do_chroma_deemphasis,
        )
        field.uphet_temp = uphet
        # Release to avoid keeping this im memory - should do this in a cleaner manner.
        field.chroma_tbc_buffer = None
        return chroma_to_u16(uphet)

    return None


def get_burst_area(field):
    burst_start = math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0])) - 4
    burst_end = math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])) + 8

    # burst length must be multiple of 4
    burst_end = burst_end - ((burst_end - burst_start) % 4)

    return burst_start, burst_end
