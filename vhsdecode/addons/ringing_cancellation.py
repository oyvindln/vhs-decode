import numpy as np
import numba as nb
import scipy.fft
import scipy
from scipy.interpolate import UnivariateSpline

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# -----------------------------------------------------------------------------
# 1. NUMBA KERNEL: CAUSAL PHASE EQUALIZATION FIR
# -----------------------------------------------------------------------------

FSC=3.579545 # TODO: parameterize (used only in debug)
FSC_RATIO = 4 # sets ratio of FFT length to fsc

FIR_LEN = 63
FFT_LEN = 512

@nb.njit(cache=True, nogil=True, fastmath=True)
def _apply_inverse_eq(picture, n_samples, causal_kernel):
    """
    Applies the causal FIR equalization filter to the video buffer.
    """
    out = np.zeros(n_samples, np.float32)

    # Outer loop over causal filter taps ensures sequential, contiguous reads
    for j in range(FIR_LEN):
        coeff = causal_kernel[j]
            
        # Region 1: Left boundary clamping (i < j references indices < 0, clamped to picture[0])
        for i in range(0, j):
            out[i] += picture[0] * coeff
            
        # Region 2: Inner core (Pure sequential memory access: picture[i - j])
        for i in range(j, n_samples):
            out[i] += picture[i - j] * coeff

    return out


def _fit_smooth_phase(S_xy, smooth=1):
    phase = -np.unwrap(np.angle(S_xy))
    mag = np.abs(S_xy)

    # frequency bins to mask for horizontal smearing
    w = 2 * np.pi * np.fft.fftfreq(len(S_xy))
    mask = (w > 0.01 * np.pi) & (w < 0.2 * np.pi)

    w_fit = w[mask]
    phase_fit = phase[mask]

    weight = mag[mask]
    weight /= np.max(weight)

    slope, intercept = np.polyfit(
        w_fit, phase_fit, 1, w=weight,
    )

    excess = phase_fit - (slope * w_fit + intercept)
    spline = UnivariateSpline(
        w_fit, excess, w=weight, s=smooth*len(w_fit)
    )

    phase_corr = np.zeros_like(phase)
    phase_corr[mask] = slope * w_fit + intercept + spline(w_fit)

    half = len(phase) // 2
    phase_corr[half + 1:] = -phase_corr[1:half][::-1]
    phase_corr[0] = 0.0
    phase_corr[half] = 0.0

    return phase_corr, -slope

def _build_fsc_mask(n, width_bins=3):
    """
    Creates a soft mask around the color subcarrier.
    Returns values 0..1.
    """

    freq = np.abs(np.fft.fftfreq(n))
    fsc = 1.0 / FSC_RATIO

    mask = np.zeros(n)
    dist = np.abs(freq - fsc)

    width = width_bins / n
    inside = dist < width
    mask[inside] = 0.5 * (
        1.0 + np.cos(np.pi * dist[inside] / width)
    )

    return mask

def _build_ideal_step(
    measured_center_rise,
    target_transition,
    local_sync,
    local_blank_r,
    win_size,
):
    ideal = np.zeros(win_size, dtype=np.float64)
    for i in range(win_size):
        t_rise = float(i) - measured_center_rise

        if t_rise < -target_transition / 2.0:
            ideal[i] = local_sync
        elif t_rise <= target_transition / 2.0:
            phase = (t_rise + target_transition / 2.0) / target_transition * np.pi
            ideal[i] = local_sync + (local_blank_r - local_sync) * (1.0 - np.cos(phase)) / 2.0
        else:
            ideal[i] = local_blank_r
    
    return ideal


def _build_ideal_hsync_pulse(
    front_porch_len,
    sync_len,
    target_transition,
    sync_level,
    blanking_level,
    win_size,
    phase_delay=0.0,
):
    """Generates an ideal full H-Sync pulse aligned with the signal's phase delay."""
    ideal = np.full(win_size, blanking_level, dtype=np.float64)
    
    center_fall = float(front_porch_len) + phase_delay
    center_rise = float(front_porch_len + sync_len) + phase_delay
    
    for i in range(win_size):
        t_fall = float(i) - center_fall
        t_rise = float(i) - center_rise
        
        # Falling Edge (Front Porch -> Sync Tip)
        if t_fall >= -target_transition / 2.0 and t_fall <= target_transition / 2.0:
            phase = (t_fall + target_transition / 2.0) / target_transition * np.pi
            ideal[i] = blanking_level - (blanking_level - sync_level) * (1.0 - np.cos(phase)) / 2.0
        # Sync Tip Region
        elif t_fall > target_transition / 2.0 and t_rise < -target_transition / 2.0:
            ideal[i] = sync_level
        # Rising Edge (Sync Tip -> Back Porch)
        elif t_rise >= -target_transition / 2.0 and t_rise <= target_transition / 2.0:
            phase = (t_rise + target_transition / 2.0) / target_transition * np.pi
            ideal[i] = sync_level + (blanking_level - sync_level) * (1.0 - np.cos(phase)) / 2.0
            
    return ideal


@nb.njit(cache=True, nogil=True, fastmath=True)
def _normalize_inplace(video_buf, sync_tip_level, blanking_level):
    scale = 2.0 / (blanking_level - sync_tip_level)
    for i in range(video_buf.size):
        video_buf[i] = (video_buf[i] - sync_tip_level) * scale - 1.0

    return video_buf


@nb.njit(cache=True, nogil=True, fastmath=True)
def _denormalize_inplace(video_buf, sync_tip_level, blanking_level):
    scale = 0.5 * (blanking_level - sync_tip_level)
    for i in range(video_buf.size):
        video_buf[i] = (video_buf[i] + 1.0) * scale + sync_tip_level

    return video_buf


@nb.njit(cache=True, nogil=True, fastmath=True)
def _get_levels(data):
    n = data.size
    k = max(1, n // 4)
    part = np.sort(data.copy())
    sync_tip_average = np.median(part[:k])
    blanking_average = np.median(part[-k:])
    return sync_tip_average, blanking_average

def _suppress_color_carrier(x, harmonics=0):
    """
    Remove a coherent carrier and its harmonics using quadrature projection.

    Parameters
    ----------
    x : ndarray
        Real signal.
    harmonics : int
        Number of harmonics to suppress, including the fundamental.
    """

    n = np.arange(len(x))
    y = x.copy()

    for h in range(1, harmonics + 1):
        phase = 2.0 * np.pi * h * n / FSC

        c = np.cos(phase)
        s = np.sin(phase)

        cc = np.dot(c, c)
        ss = np.dot(s, s)

        if cc > 1e-12:
            ac = np.dot(y, c) / cc
            y -= ac * c

        if ss > 1e-12:
            bs = np.dot(y, s) / ss
            y -= bs * s

    return y

def _suppress_color_carrier_fft(samples):
    """
    Remove a pure Fs/4 carrier using quadrature estimation.
    """

    out = samples.copy()

    # Accumulate I/Q over the whole pulse
    i = 0.0
    q = 0.0

    for n, x in enumerate(samples):
        phase = (n & 3)

        if phase == 0:
            i += x
        elif phase == 2:
            i -= x
        elif phase == 1:
            q -= x
        else:
            q += x

    # Remove estimated carrier
    scale = 2.0 / len(samples)
    i *= scale
    q *= scale

    for n in range(len(out)):
        phase = n & 3

        if phase == 0:
            out[n] -= i
        elif phase == 1:
            out[n] -= -q
        elif phase == 2:
            out[n] -= -i
        else:
            out[n] -= q

    return out

def _taper_delta_fft(fft):
    freqs = np.abs(np.fft.fftfreq(len(fft)))

    mask = np.ones_like(freqs)

    start = 0.20   # 1 fsc
    end = 0.25     # Nyquist (2 fsc)

    idx = (freqs >= start) & (freqs <= end)

    t = (freqs[idx] - start) / (end - start)
    mask[idx] = 0.5 * (1.0 + np.cos(np.pi * t))
    mask[freqs > end] = 0.0

    return fft * mask

# -----------------------------------------------------------------------------
# Group Delay Correction
# -----------------------------------------------------------------------------
def apply_inverse_equalization(
    video_buf, 
    line_start, 
    line_end, 
    line_length,
    sync_tip_level,
    blanking_level,
    front_porch_len,
    sync_len,         
    back_porch_len,
    target_transition,
    group_delay_state,
    noise_threshold=1, # TODO, determine automatically; set frequency scaling based on noise floor, or MAD between frequencies
    debug=False
):
    _normalize_inplace(video_buf, sync_tip_level, blanking_level)

    n_samples = len(video_buf)

    pre_rise_samples = int(np.round(0.25 * sync_len))
    win_size = pre_rise_samples + back_porch_len
    win_size -= win_size % 4 
    offset_rise = pre_rise_samples

    S_xy = np.zeros(FFT_LEN, dtype=np.complex128)
    S_xx = np.zeros(FFT_LEN, dtype=np.complex128)
    S_yy = np.zeros(FFT_LEN, dtype=np.float64)

    pulse_ffts = []
    delta_ffts = []
    corrected_ffts = []

    win = scipy.signal.windows.tukey(win_size, alpha=0.05)
    start_idx = (FFT_LEN - win_size) // 2
    count = 0

    # =========================================================================
    # PART 1: Calculate Pulse Shape & Spectra (Rising Edge Only)
    # =========================================================================
    for line in range(line_start, line_end + 1):
        loc = line * line_length
        start_loc = loc + sync_len - pre_rise_samples
        if start_loc >= 0 and start_loc + win_size < n_samples:
            pulse = video_buf[start_loc : start_loc + win_size]
            # pulse = _suppress_color_carrier(pulse)

            measured_sync_tip_level, measured_blanking_level = _get_levels(pulse)
            mid_val = measured_sync_tip_level + (measured_blanking_level - measured_sync_tip_level) / 2.0

            count += 1
            
            measured_center_rise = float(offset_rise)
            for i in range(win_size - 1):
                if pulse[i] <= mid_val <= pulse[i+1]:
                    y0 = pulse[i] - mid_val
                    y1 = pulse[i+1] - mid_val
                    if (y1 - y0) != 0.0:
                        measured_center_rise = float(i) + abs(y0) / abs(y1 - y0)
                    break
                    
            sum_sync, count_sync = 0.0, 0
            end_sync_idx = max(1, int(measured_center_rise - 2))
            for i in range(0, end_sync_idx):
                sum_sync += pulse[i]
                count_sync += 1
            local_sync = sum_sync / count_sync if count_sync > 0 else measured_sync_tip_level
            
            sum_bp, count_bp = 0.0, 0
            start_bp_idx = min(win_size - 1, int(measured_center_rise + 2))
            for i in range(start_bp_idx, win_size):
                sum_bp += pulse[i]
                count_bp += 1
            local_blank_r = sum_bp / count_bp if count_bp > 0 else measured_blanking_level

            ideal_for_fir = _build_ideal_step(
                measured_center_rise, target_transition,
                local_sync, local_blank_r, win_size
            )

            d_ideal = np.gradient(ideal_for_fir) * win
            d_pulse = np.gradient(pulse) * win
            
            pad_ideal = np.zeros(FFT_LEN, dtype=np.float64)
            pad_pulse = np.zeros(FFT_LEN, dtype=np.float64)
            pad_ideal[start_idx : start_idx + win_size] = d_ideal
            pad_pulse[start_idx : start_idx + win_size] = d_pulse

            # measure pulse's actual frequency response
            Y = scipy.fft.fft(pad_pulse)
            # measure the pulse's expected frequency response, given the sync pulse parameters
            X = scipy.fft.fft(pad_ideal)
            
            S_xy += X * np.conj(Y)
            S_xx += np.abs(X)**2
            S_yy += np.abs(Y)**2

    if count > 0:
        current_measurement = {
            's_xy': S_xy,
            's_xx': S_xx,
            's_yy': S_yy,
        }
        group_delay_state.append(current_measurement)
    else:
        _denormalize_inplace(video_buf, sync_tip_level, blanking_level)
        return video_buf

    rolling_S_xy = np.zeros(FFT_LEN, dtype=np.complex128)
    rolling_S_xx = np.zeros(FFT_LEN, dtype=np.complex128)
    rolling_S_yy = np.zeros(FFT_LEN, dtype=np.float64)

    for measurement in group_delay_state:
        rolling_S_xy += measurement['s_xy']
        rolling_S_xx += measurement['s_xx']
        rolling_S_yy += measurement['s_yy']

    # =========================================================================
    # PART 2: Analyze Pulse Structure
    #     2A: Group Delay (Horizontal Smear)
    #     2B: Ringing
    #     2C: Chroma carrier leakage
    # =========================================================================

    # rolling_S_xy = _suppress_color_carrier_fft(rolling_S_xy)
    # rolling_S_xy = _taper_delta_fft(rolling_S_xy)

    phase_error, advance_float = _fit_smooth_phase(
        rolling_S_xy
    )

    H_measured = (rolling_S_xy / np.maximum(rolling_S_xx, 1e-12))
    # Normalize DC
    H_measured /= max(np.abs(H_measured[0]), 1e-12)

    # ------------------------------------------------------------
    # 2A Remove horizontal smear (phase only)
    # ------------------------------------------------------------
    smear_strength = 1
    smear_gd_sigma = FFT_LEN / FIR_LEN # group delay measurement smoothing
    smear_weight_sigma = FFT_LEN / (2 * FIR_LEN) # correction transition smoothing

    # Estimate group delay stability
    phase = -np.unwrap(np.angle(H_measured))
    w = 2.0 * np.pi * np.fft.fftfreq(len(H_measured))
    group_delay = -np.gradient(phase, w)

    # Smooth group delay to find broad delay behavior
    gd_smooth = scipy.ndimage.gaussian_filter1d(
        group_delay, smear_gd_sigma
    )

    # Difference from smooth delay = non-smear behavior
    gd_error = np.abs(group_delay - gd_smooth)
    gd_scale = np.percentile(gd_error, 95)

    smear_weight = np.clip(
        1.0 - gd_error / max(gd_scale, 1e-12), 0.0, 1.0
    )
    smear_weight = scipy.ndimage.gaussian_filter1d(
        smear_weight, smear_weight_sigma
    )

    phase_error_weighted = phase_error * smear_weight

    # Apply phase-only correction
    H_smear = np.exp(
        -1j *
        smear_strength *
        phase_error_weighted
    )

    # ------------------------------------------------------------
    # 2B Remove ringing
    # ------------------------------------------------------------
    ring_strength = 1
    ring_sigma = 1 # FFT_LEN / FIR_LEN

    # remove color carrier leakage from this measurement
    fsc_mask = _build_fsc_mask(FFT_LEN, width_bins=4)
    H_after_smear = H_measured * H_smear * (1.0 - fsc_mask) + fsc_mask

    H_target = np.abs(H_after_smear) * np.exp(
        1j * scipy.ndimage.gaussian_filter1d(
            np.unwrap(np.angle(H_after_smear)), ring_sigma
        )
    )

    H_residual = H_after_smear - H_target

    # Keep causal ringing tail only
    h_residual=scipy.fft.ifft(H_residual).real

    peak=np.argmax(np.abs(h_residual))
    h_residual[:peak + 1] = 0

    ring_window=scipy.signal.windows.tukey(
        FIR_LEN, alpha=0.25
    )

    h_residual[peak + 1:peak + 1 + FIR_LEN] *= ring_window[:min(
        FIR_LEN, len(h_residual) - peak - 1
    )]

    h_residual[peak+1+FIR_LEN:] = 0

    H_ring_error = scipy.fft.fft(h_residual)

    ring_gain = np.sqrt(
        np.sum(np.abs(H_residual) ** 2) /
        max(np.sum(np.abs(H_ring_error) ** 2), 1e-12)
    )

    H_ring = 1.0 - ring_strength * H_ring_error * ring_gain

    # ------------------------------------------------------------
    # 2C Remove chroma leakage following smear slope
    # ------------------------------------------------------------
    chroma_strength = 1

    # Use same phase slope as smear correction
    H_chroma_slope = np.exp(-1j * phase_error)
    # Limit correction to carrier region
    H_chroma_slope = H_chroma_slope * fsc_mask + (1.0 - fsc_mask)

    # Measure remaining carrier error
    H_after_chroma_slope = H_after_smear * H_chroma_slope
    H_chroma_residual = (H_after_chroma_slope - 1.0) * fsc_mask

    # Convert residual to causal correction
    h_chroma = scipy.fft.ifft(H_chroma_residual).real
    h_chroma[0] = 0
    h_chroma[FIR_LEN:] = 0
    h_chroma[1:FIR_LEN] *= scipy.signal.windows.tukey(
        FIR_LEN-1, alpha=0.25
    )
    H_chroma_error = scipy.fft.fft(h_chroma)

    H_chroma = 1.0 - chroma_strength * H_chroma_error

    ### assemble the filters
    H_inv = (
        H_smear
      * H_ring
      * H_chroma
    )

    ######################################################
    # TODO: Might be able to detect head switching pulses.
    #       The ringing pattern clearly differs when the head switch occurs.
    #       This could be used to detect the switching position, measure it, and correct it's slope
    ######################################################

    # =========================================================================
    # PART 3: Build Causal Inverse Equalization FIR Filter
    # =========================================================================

    # IFFT back to time domain
    h_full = scipy.fft.fftshift(scipy.fft.ifft(H_inv).real)
    
    # Make FIR causal
    true_center = FFT_LEN // 2

    # Assemble half-size causal kernel
    fir_kernel = np.zeros(FIR_LEN, dtype=np.float32)
    fir_kernel[0] = h_full[true_center]
    fir_kernel[1:] = h_full[
        true_center + 1:
        true_center + FIR_LEN
    ]
    fir_kernel /= np.sum(fir_kernel)

    # =========================================================================
    # PART 4: Apply Inverse Equalization FIR Filter
    # =========================================================================
    video_buf_filtered = _apply_inverse_eq(video_buf, n_samples, fir_kernel)

    lti_params = derive_lti_parameters(
        fir_kernel, 
        noise_threshold=noise_threshold
    )

    if debug:
        full_win_size = max(front_porch_len + sync_len + back_porch_len, FIR_LEN)
        raw_pulses = []
        corrected_pulses = []
        line_indices = []

        for line in range(line_start, line_end + 1):
            start_loc = line * line_length - front_porch_len
            if start_loc >= 0 and start_loc + full_win_size < n_samples:
                raw_p = video_buf[start_loc : start_loc + full_win_size].copy()
                corr_p = _apply_inverse_eq(raw_p, full_win_size, fir_kernel)
                # raw_p = _suppress_color_carrier(raw_p)

                raw_pulses.append(raw_p)
                corrected_pulses.append(corr_p)
                line_indices.append(line)

                pulse_fft = scipy.fft.fft(raw_p, n=FFT_LEN)
                corrected_fft = scipy.fft.fft(corr_p, n=FFT_LEN)
                delta_fft = corrected_fft - pulse_fft

                pulse_ffts.append(pulse_fft)
                corrected_ffts.append(corrected_fft)
                delta_ffts.append(delta_fft)

                pulse_ffts

        all_corr_arr = np.array(corrected_pulses)
        sync_tip_average, blanking_average = _get_levels(all_corr_arr.flatten())

        # build ideal pulse
        ideal_debug = _build_ideal_hsync_pulse(
            front_porch_len,
            sync_len,
            target_transition,
            sync_tip_average,
            blanking_average,
            full_win_size,
            phase_delay=0,
        )
                    
        _show_group_delay_debug(
            raw_pulses,
            corrected_pulses,
            ideal_debug,
            fir_kernel,
            0,
            line_indices,
            pulse_ffts,
            delta_ffts,
            corrected_ffts,
            FFT_LEN,
        )


    return _denormalize_inplace(video_buf_filtered, sync_tip_level, blanking_level), lti_params


# Set `--debug_plot inverse_eq` to show this plot
def _show_group_delay_debug(
    raw_pulses,
    corrected_pulses,
    ideal,
    fir_kernel,
    calc_advance,
    line_indices,
    pulse_ffts,
    delta_ffts,
    corrected_ffts,
    FFT_LEN,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError:
        print("Matplotlib is required to render the debug plot.")
        return

    n_lines = len(raw_pulses)
    if n_lines == 0:
        return

    # Expand figure layout to fit 6 subplots
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#121212",
        "axes.facecolor": "#181818",
        "axes.edgecolor": "#aaaaaa",
        "axes.labelcolor": "#dddddd",
        "axes.titlecolor": "#ffffff",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "text.color": "#dddddd",
        "grid.color": "#555555",
        "grid.alpha": 0.35,
    })
    fig = plt.figure(figsize=(16, 20))
    
    ax1 = plt.subplot2grid((6, 2), (0, 0), colspan=2) # Line Inspection
    ax2 = plt.subplot2grid((6, 2), (1, 0), colspan=1) # All Lines Overlay
    ax3 = plt.subplot2grid((6, 2), (1, 1), colspan=1) # Causal Kernel Taps
    ax_slider = plt.subplot2grid((6, 1), (2, 0), colspan=2) # Slider

    # FFT section
    ax4 = plt.subplot2grid((6, 2), (3, 0), colspan=2) # Raw Pulse
    ax5 = plt.subplot2grid((6, 2), (4, 0), colspan=2, sharex=ax4, sharey=ax4) # Delta
    ax6 = plt.subplot2grid((6, 2), (5, 0), colspan=2, sharex=ax4, sharey=ax4) # Corrected Pulse

    initial_idx = 0
    line_num = line_indices[initial_idx]
    
    raw_line, = ax1.plot(raw_pulses[initial_idx], label=f'Raw Line {line_num}', color='#d62728', linewidth=1.8, alpha=0.85)
    corr_line, = ax1.plot(corrected_pulses[initial_idx], label=f'Equalized Line {line_num}', color='#1f77b4', linewidth=2.2)
    ax1.plot(ideal, label='Target Reference', color='#7f7f7f', linestyle=':', linewidth=2)
    
    ax1.set_title(f"Line Inspection [Line 1] | Group Delay Advance: {calc_advance:.3f} samples", fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.35)
    ax1.set_ylabel("Amplitude")

    for p in raw_pulses:
        ax2.plot(p, color='#d62728', alpha=min(0.25, max(0.02, 5.0 / n_lines)), linewidth=0.8)
    for p in corrected_pulses:
        ax2.plot(p, color='#1f77b4', alpha=min(0.25, max(0.02, 5.0 / n_lines)), linewidth=0.8)

    ax2.set_title(f"All Lines Overlay ({n_lines} Lines)", fontweight='bold')
    ax2.grid(True, alpha=0.35)
    ax2.set_ylabel("Amplitude")

    x_axis = np.arange(len(fir_kernel))
    ax3.plot(x_axis, fir_kernel, label='Causal Taps (t >= 0)', color='purple', marker='.', linewidth=1.2)
    ax3.set_title("Causal Equalization Kernel", fontweight='bold')
    ax3.axhline(0, color='black', alpha=0.3)
    ax3.set_xlim(0, len(fir_kernel) - 1)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.35)

    # =========================================================================
    # Build FFTs
    # =========================================================================
    freqs = scipy.fft.fftfreq(FFT_LEN)
    half_len = FFT_LEN // 2
    pos_freqs = freqs[:half_len] * FSC_RATIO * FSC

    num_ffts = len(pulse_ffts)
    min_line = line_indices[0] if line_indices else 0
    max_line = line_indices[-1] if line_indices else num_ffts - 1
    
    pulse_spec_data = np.zeros((len(pulse_ffts), half_len + 1), dtype=np.float64)
    delta_spec_data = np.zeros((len(delta_ffts), half_len + 1), dtype=np.float64)
    corrected_spec_data = np.zeros((len(corrected_ffts), half_len + 1), dtype=np.float64)
    for i in range(num_ffts):
        if i < pulse_spec_data.shape[0]:
            pulse_fft = pulse_ffts[i]
            pulse_spec_data[i, :] = 20 * np.log10(np.maximum(np.abs(pulse_fft[:half_len + 1]), 1e-12))
    
        if i < delta_spec_data.shape[0]:
            delta_fft = delta_ffts[i]
            delta_spec_data[i, :] = 20 * np.log10(np.maximum(np.abs(delta_fft[:half_len + 1]), 1e-12))

        if i < corrected_spec_data.shape[0]:
            corrected_fft = corrected_ffts[i]
            corrected_spec_data[i, :] = 20 * np.log10(np.maximum(np.abs(corrected_fft[:half_len + 1]), 1e-12))

    # Custom Audacity-style multi-stop gradient: Black -> Deep Green -> Neon Green -> White
    green_pos = LinearSegmentedColormap.from_list(
        "green", 
        ["#000000", "#00ff00"]
    )   
    green_neg = LinearSegmentedColormap.from_list(
        "green", 
        ["#00ff00", "#000000"]
    )

    # =========================================================================
    # PULSE SPECTROGRAM
    # =========================================================================
    ax4.imshow(
        pulse_spec_data,
        aspect='auto',
        origin='lower',
        extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line],
        cmap=green_pos
    )
    ax4.set_title("Raw $Y_{raw}$", fontweight='bold')
    ax4.set_adjustable('box')

    # =========================================================================
    # SUBTRACTION SPECTROGRAM
    # =========================================================================
    image = ax5.imshow(
        delta_spec_data,
        aspect='auto',
        origin='lower',
        extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line],
        cmap=green_neg
    )
    ax5.set_title("Delta $Y_{delta}$", fontweight='bold')
    ax5.set_xlabel("Frequency (MHz)")
    ax5.set_ylabel("Line Index")
    ax5.set_adjustable('box')

    # =========================================================================
    # CORRECTED SPECTROGRAM
    # =========================================================================
    image = ax6.imshow(
        corrected_spec_data,
        aspect='auto',
        origin='lower',
        extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line],
        cmap=green_pos
    )
    ax6.set_title("Corrected $Y_{delta} - Y_{raw}$", fontweight='bold')
    ax6.set_adjustable('box')
    ax6.set_xlabel("Normalized Frequency")

    ax6.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax6.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    for ax in (ax4, ax5, ax6):
        ax.set_xlim(pos_freqs[0], pos_freqs[-1])
        ax.set_ylim(min_line, max_line)


    # =========================================================================
    # Slider
    # =========================================================================
    slider = Slider(
        ax=ax_slider,
        label='Line Index ',
        valmin=0,
        valmax=max(0, n_lines - 1),
        valinit=0,
        valstep=1,
        color='#1f77b4'
    )

    ax_slider.set_facecolor("#181818")
    slider.label.set_color("#dddddd")
    slider.valtext.set_color("#dddddd")

    def update(val):
        idx = int(slider.val)
        l_num = line_indices[idx]
        raw_line.set_ydata(raw_pulses[idx])
        raw_line.set_label(f'Raw Line {l_num}')
        corr_line.set_ydata(corrected_pulses[idx])
        corr_line.set_label(f'Equalized Line {l_num}')
        ax1.set_title(f"Line Inspection [Line {l_num}] | Group Delay Advance: {calc_advance:.3f} samples", fontweight='bold')
        ax1.legend(loc='lower right')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.98,
        bottom=0.05,
        hspace=0.25
    )
    
    plt.show()




# -----------------------------------------------------------------------------
# LTI PARAMETER DERIVATION FUNCTION
# -----------------------------------------------------------------------------
def derive_lti_parameters(fir_kernel, noise_threshold):
    """
    Derives optimal Luminance Transient Improvement (LTI) parameters
    analytically from the derived causal group-delay FIR kernel.
    """

    # 1. Total energy vs. center tap energy
    total_energy = np.sum(fir_kernel**2)
    if total_energy <= 1e-12:
        return {'gain': 0.0, 'threshold': 0.1, 'blur_radius': 0.0}

    center_energy = fir_kernel[0]**2
    energy_dispersion = 1.0 - (center_energy / total_energy)
    
    # 2. Compute second moment (spatial spread radius)
    indices = np.arange(len(fir_kernel))
    weighted_spread = np.sum(indices * np.abs(fir_kernel)) / np.sum(np.abs(fir_kernel))
    
    # 3. Scale LTI Gain proportionally to dispersion and noise threshold
    # High noise_threshold reduces max gain to prevent boosting noise floor
    noise_suppression_factor = max(0.2, 1.0 - 2.0 * noise_threshold)
    lti_gain = np.clip(energy_dispersion * 1.5 * noise_suppression_factor, 0.0, 1.0)
    
    # 4. Adaptive Threshold: Set above the residual high-frequency noise level
    lti_threshold = np.clip(noise_threshold * 0.75, 0.02, 0.15)

    return {
        'gain': float(lti_gain),
        'threshold': float(lti_threshold),
        'blur_radius': float(weighted_spread),
        'dispersion': float(energy_dispersion)
    }


# -----------------------------------------------------------------------------
# ADAPTIVE LTI PROCESSING KERNEL
# -----------------------------------------------------------------------------
@nb.njit(cache=True, nogil=True, fastmath=True)
def apply_adaptive_luma_transient_improvement(video_buf, gain, threshold):
    """
    Applies non-linear LTI using parameters derived from the group delay kernel.
    Modifies video_buf in place.
    """
    n = len(video_buf)

    # Preserve original samples needed for the stencil
    prev = video_buf[0]

    for i in range(1, n - 1):
        curr = video_buf[i]
        nxt = video_buf[i + 1]

        diff = nxt - prev
        abs_diff = abs(diff)

        # Only boost active step transitions exceeding noise threshold
        if abs_diff > threshold:
            # Local slope estimate
            grad = curr - prev

            # Non-linear gain scaling (tapers off near plateaus to prevent ringing)
            edge_weight = min(1.0, abs_diff / (2.0 * threshold))
            video_buf[i] = curr + (gain * edge_weight) * grad

        # Advance cached original sample
        prev = curr