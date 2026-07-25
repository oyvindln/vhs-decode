import numpy as np
import numba as nb
import scipy.fft

# -----------------------------------------------------------------------------
# 1. NUMBA KERNEL: STRICTLY CAUSAL PHASE EQUALIZATION FIR
# -----------------------------------------------------------------------------

fir_len = 129
fir_len_half = fir_len // 2

@nb.njit(cache=True, nogil=True, fastmath=True)
def _apply_global_inverse_eq(picture, n_samples, fir_kernel):
    out = np.empty(n_samples, np.float32)
    picture = picture.astype(np.float32)
    fir_kernel = fir_kernel.astype(np.float32)

    last = n_samples - 1
    center_end = n_samples - fir_len_half

    # Left edge
    for i in range(fir_len_half):
        s = np.float32(0.0)
        start = i + fir_len_half

        for j in range(fir_len):
            idx = start - j
            if idx < 0:
                idx = 0
            s += picture[idx] * fir_kernel[j]

        out[i] = s

    # Center
    for i in range(fir_len_half, center_end):
        s = np.float32(0.0)
        start = i + fir_len_half

        for j in range(fir_len):
            s += picture[start - j] * fir_kernel[j]

        out[i] = s

    # Right edge
    for i in range(center_end, n_samples):
        s = np.float32(0.0)
        start = i + fir_len_half

        for j in range(fir_len):
            idx = start - j
            if idx > last:
                idx = last
            s += picture[idx] * fir_kernel[j]

        out[i] = s

    return out


def _calc_phase_advance(S_xy, pad_len):
    """
    Calculates the exact group delay advance (in fractional samples) 
    from the cross-spectral density phase slope across the signal passband.
    """
    phase = np.unwrap(np.angle(S_xy))
    freqs = scipy.fft.fftfreq(pad_len)
    
    # Active passband for a video sync pulse is concentrated in lower frequencies.
    # Use the lower 15% of the spectrum to find the linear phase slope.
    limit = max(2, int(pad_len * 0.15))
    
    w = 2.0 * np.pi * freqs[1:limit]
    p = phase[1:limit]
    
    # Fit line: p = slope * w + intercept
    slope, _ = np.polyfit(w, p, 1)
    
    # Group delay is the negative derivative of phase with respect to angular frequency
    return -slope


def _show_group_delay_debug(
    raw_pulses,
    corrected_pulses,
    ideal,
    fir_kernel,
    calc_advance,
    line_indices,
):
    """
    Renders an interactive debug UI with a Slider to scrub through individual video lines,
    plus an aggregate overlay of all lines showing frame-wide variance.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError:
        print("Matplotlib is required to render the debug plot.")
        return

    n_lines = len(raw_pulses)
    if n_lines == 0:
        return

    fig = plt.figure(figsize=(15, 11))
    
    # Subplot grid layout
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2) # Single Line View (Interactive)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=1) # All Lines Overlay (Persistence)
    ax3 = plt.subplot2grid((3, 2), (1, 1), colspan=1) # FIR Kernel Response
    ax_slider = plt.subplot2grid((3, 2), (2, 0), colspan=2) # Slider Axis

    # -------------------------------------------------------------------------
    # Panel 1: Single Line Inspector (Interactive)
    # -------------------------------------------------------------------------
    initial_idx = 0
    line_num = line_indices[initial_idx]
    
    raw_line, = ax1.plot(raw_pulses[initial_idx], label=f'Raw Line {line_num}', color='#d62728', linewidth=1.8, alpha=0.85)
    corr_line, = ax1.plot(corrected_pulses[initial_idx], label=f'Equalized Line {line_num}', color='#1f77b4', linewidth=2.2)
    ax1.plot(ideal, label='Target Reference (Phase-Aligned)', color='#7f7f7f', linestyle=':', linewidth=2)
    
    ax1.set_title(f"Line Inspection [Line {line_num}] | Sub-sample Advance: {calc_advance:.3f} samples", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.35)
    ax1.set_ylabel("Normalized Amplitude")

    # -------------------------------------------------------------------------
    # Panel 2: Persistence Overlay (All Lines Overlaid)
    # -------------------------------------------------------------------------
    for p in raw_pulses:
        ax2.plot(p, color='#d62728', alpha=min(0.25, max(0.02, 5.0 / n_lines)), linewidth=0.8)
    
    for p in corrected_pulses:
        ax2.plot(p, color='#1f77b4', alpha=min(0.25, max(0.02, 5.0 / n_lines)), linewidth=0.8)

    ax2.plot(ideal, color='black', linestyle='--', linewidth=1.5, label='Target (Phase-Aligned)')
    ax2.set_title(f"All Lines Overlay ({n_lines} Lines)", fontweight='bold')
    ax2.grid(True, alpha=0.35)
    ax2.set_ylabel("Normalized Amplitude")

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#d62728', lw=1.5, label='Raw Set'),
        Line2D([0], [0], color='#1f77b4', lw=1.5, label='Corrected Set'),
        Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Target')
    ]
    ax2.legend(handles=custom_lines, loc='upper right')

    # -------------------------------------------------------------------------
    # Panel 3: Equalization Kernel
    # -------------------------------------------------------------------------
    center = len(fir_kernel) // 2
    x_axis = np.arange(-center, center + 1)
    
    ax3.plot(x_axis, fir_kernel, label='Causal FIR Taps (t >= 0)', color='purple', marker='.', linewidth=1.2)
    ax3.set_title("Derived Equalization Kernel", fontweight='bold')
    ax3.axhline(0, color='black', alpha=0.3)
    ax3.set_xlim(-40, 40)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.35)

    # -------------------------------------------------------------------------
    # Slider Logic
    # -------------------------------------------------------------------------
    slider = Slider(
        ax=ax_slider,
        label='Line Index ',
        valmin=0,
        valmax=n_lines - 1,
        valinit=0,
        valstep=1,
        color='#1f77b4'
    )

    def update(val):
        idx = int(slider.val)
        l_num = line_indices[idx]
        raw_line.set_ydata(raw_pulses[idx])
        raw_line.set_label(f'Raw Line {l_num}')
        corr_line.set_ydata(corrected_pulses[idx])
        corr_line.set_label(f'Equalized Line {l_num}')
        ax1.set_title(f"Line Inspection [Line {l_num}] | Sub-sample Advance: {calc_advance:.3f} samples", fontweight='bold')
        ax1.legend(loc='upper right')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.tight_layout()
    plt.show()


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


@nb.njit
def _normalize_inplace(video_buf, sync_tip_level, blanking_level):
    scale = 2.0 / (blanking_level - sync_tip_level)
    for i in range(video_buf.size):
        video_buf[i] = (video_buf[i] - sync_tip_level) * scale - 1.0

@nb.njit
def _denormalize_inplace(video_buf, sync_tip_level, blanking_level):
    scale = 0.5 * (blanking_level - sync_tip_level)
    for i in range(video_buf.size):
        video_buf[i] = (video_buf[i] + 1.0) * scale + sync_tip_level

@nb.njit
def _get_levels(data):
    n = data.size
    k = max(1, n // 4)
    part = np.sort(data.copy())
    sync_tip_average = np.median(part[:k])
    blanking_average = np.median(part[-k:])
    return sync_tip_average, blanking_average


# -----------------------------------------------------------------------------
# Group Delay Correction
# -----------------------------------------------------------------------------
def correct_group_delay(
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
    noise_threshold=0.15,
    debug=False
):
    _normalize_inplace(video_buf, sync_tip_level, blanking_level)
    
    n_samples = len(video_buf)
    
    pre_rise_samples = int(np.round(0.25 * sync_len))
    win_size = pre_rise_samples + back_porch_len
    offset_rise = pre_rise_samples
    
    min_required_len = 2 * win_size - 1
    pad_len = scipy.fft.next_fast_len(min_required_len, real=False)
    
    S_xy = np.zeros(pad_len, dtype=np.complex128)
    S_yy = np.zeros(pad_len, dtype=np.float64)
    
    win = np.hanning(win_size)
    start_idx = (pad_len - win_size) // 2
    count = 0

    # =========================================================================
    # PART 1: Calculate Pulse Shape & Spectra (Rising Edge Only)
    # =========================================================================
    for line in range(line_start, line_end + 1):
        loc = line * line_length
        start_loc = loc + sync_len - pre_rise_samples
        if start_loc >= 0 and start_loc + win_size < n_samples:
            
            pulse = video_buf[start_loc : start_loc + win_size]
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
            
            pad_ideal = np.zeros(pad_len, dtype=np.float64)
            pad_pulse = np.zeros(pad_len, dtype=np.float64)
            pad_ideal[start_idx : start_idx + win_size] = d_ideal
            pad_pulse[start_idx : start_idx + win_size] = d_pulse
            
            X = scipy.fft.fft(pad_ideal)
            Y = scipy.fft.fft(pad_pulse)
            
            S_xy += X * np.conj(Y)
            S_yy += np.abs(Y)**2

    if count > 0:
        current_measurement = {
            's_xy': S_xy,
            's_yy': S_yy,
        }
        group_delay_state.append(current_measurement)
    else:
        _denormalize_inplace(video_buf, sync_tip_level, blanking_level)
        return video_buf

    rolling_S_xy = np.zeros(pad_len, dtype=np.complex128)
    rolling_S_yy = np.zeros(pad_len, dtype=np.float64)

    for measurement in group_delay_state:
        rolling_S_xy += measurement['s_xy']
        rolling_S_yy += measurement['s_yy']

    # =========================================================================
    # PART 2: Analytical Advance & Deconvolution
    # =========================================================================
    # Dynamically extract the optimal floating-point advance from phase slope
    advance_float = _calc_phase_advance(rolling_S_xy, pad_len)
    
    # Split into integer video shift and fractional kernel phase-shift
    int_adv = int(np.round(advance_float))
    frac_adv = advance_float - int_adv
    
    reg = np.max(rolling_S_yy) * noise_threshold 
    denom = rolling_S_yy + reg
    denom[denom == 0] = 1e-12
    H_inv = rolling_S_xy / denom

    # Apply ONLY the fractional sub-sample advance to the kernel in frequency domain.
    # This securely modifies the filter shape without needing to FFT the whole video buffer.
    w_full = 2.0 * np.pi * scipy.fft.fftfreq(pad_len)
    H_inv = H_inv * np.exp(-1j * w_full * frac_adv)

    # IFFT back to time domain
    h_full = scipy.fft.fftshift(scipy.fft.ifft(H_inv).real)
    
    # Do NOT roll the kernel here. Center extraction guarantees t=0 anchoring.
    center_full = pad_len // 2
    fir_kernel = h_full[center_full - fir_len_half : center_full + fir_len_half + 1].copy()
    
    # 1. Enforce strict causality
    fir_kernel[:fir_len_half] = 0.0
    
    # 2. Base DC normalization
    fir_kernel[fir_len_half] = 1.0 - np.sum(fir_kernel[fir_len_half + 1:])

    # Collect individual line traces if debug mode is requested
    if debug:
        full_win_size = front_porch_len + sync_len + back_porch_len
        raw_pulses = []
        corrected_pulses = []
        line_indices = []

        for line in range(line_start, line_end + 1):
            start_loc = line * line_length - front_porch_len
            if start_loc >= 0 and start_loc + full_win_size < n_samples:
                raw_p = video_buf[start_loc : start_loc + full_win_size].copy()
                
                # Apply filter to individual line trace for inspection
                delayed_p = np.roll(raw_p, int_adv)
                filtered_p = _apply_global_inverse_eq(delayed_p, full_win_size, fir_kernel)
                corr_p = np.roll(filtered_p, -int_adv)

                raw_pulses.append(raw_p)
                corrected_pulses.append(corr_p)
                line_indices.append(line)

    # =========================================================================
    # PART 3: Apply Strictly Causal Equalization via Data Integer Shift
    # =========================================================================
    video_buf = np.roll(video_buf, int_adv)
    video_buf = _apply_global_inverse_eq(video_buf, n_samples, fir_kernel)
    video_buf = np.roll(video_buf, -int_adv)

    if debug and len(raw_pulses) > 0:
        # Determine average levels from corrected set to construct target overlay
        all_corr_arr = np.array(corrected_pulses)
        sync_tip_average, blanking_average = _get_levels(all_corr_arr.flatten())

        ideal_debug = _build_ideal_hsync_pulse(
            front_porch_len,
            sync_len,
            target_transition,
            sync_tip_average,
            blanking_average,
            full_win_size,
            phase_delay=-(advance_float+int_adv),
        )
        
        _show_group_delay_debug(
            raw_pulses,
            corrected_pulses,
            ideal_debug,
            fir_kernel,
            advance_float,
            line_indices,
        )

    _denormalize_inplace(video_buf, sync_tip_level, blanking_level)

    # Calculate optimal LTI settings from the derived FIR kernel
    lti_params = derive_lti_parameters(
        fir_kernel, 
        noise_threshold=noise_threshold,
        max_gain=1 # TODO parameterize
    )

    return video_buf, lti_params


# -----------------------------------------------------------------------------
# LTI PARAMETER DERIVATION FUNCTION
# -----------------------------------------------------------------------------
def derive_lti_parameters(fir_kernel, noise_threshold, max_gain):
    """
    Derives optimal Luminance Transient Improvement (LTI) parameters
    analytically from the derived causal group-delay FIR kernel.
    """
    center = len(fir_kernel) // 2
    causal_taps = fir_kernel[center:]
    
    # 1. Total energy vs. center tap energy
    total_energy = np.sum(causal_taps**2)
    if total_energy <= 1e-12:
        return {'gain': 0.0, 'threshold': 0.1, 'blur_radius': 0.0}

    center_energy = causal_taps[0]**2
    energy_dispersion = 1.0 - (center_energy / total_energy)
    
    # 2. Compute second moment (spatial spread radius)
    indices = np.arange(len(causal_taps))
    weighted_spread = np.sum(indices * np.abs(causal_taps)) / np.sum(np.abs(causal_taps))
    
    # 3. Scale LTI Gain proportionally to dispersion and noise threshold
    # High noise_threshold reduces max gain to prevent boosting noise floor
    noise_suppression_factor = max(0.2, 1.0 - 2.0 * noise_threshold)
    lti_gain = np.clip(energy_dispersion * 1.5 * noise_suppression_factor, 0.0, max_gain)
    
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
@nb.njit(cache=True, fastmath=True)
def apply_adaptive_lti(video_buf, gain, threshold):
    """
    Applies non-linear LTI using parameters derived from the group delay kernel.
    """
    if gain <= 0.001:
        return video_buf

    n = len(video_buf)
    out = video_buf.copy()
    
    for i in range(1, n - 1):
        diff = video_buf[i + 1] - video_buf[i - 1]
        abs_diff = abs(diff)
        
        # Only boost active step transitions exceeding noise threshold
        if abs_diff > threshold:
            # Local slope estimate
            grad = video_buf[i] - video_buf[i - 1]
            
            # Non-linear gain scaling (tapers off near plateaus to prevent ringing)
            edge_weight = min(1.0, abs_diff / (2.0 * threshold))
            out[i] = video_buf[i] + (gain * edge_weight) * grad

    return out