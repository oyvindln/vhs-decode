import numpy as np
import numba as nb
import scipy.fft

from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------------
# 1. NUMBA KERNEL: CAUSAL PHASE EQUALIZATION FIR
# -----------------------------------------------------------------------------

FIR_LEN = 63
FFT_LEN = 2**9

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

def _calc_phase_advance(S_xy):
    """
    Calculates the exact group delay advance (in fractional samples).
    
    Optimized for maximum ringing cancellation:
    Strictly bandpasses the spectrum to remove the massive low-frequency 
    sync edge energy. It locates the high-frequency ringing resonance peak 
    and fits the phase slope locally around that specific frequency.
    """
    phase = np.unwrap(np.angle(S_xy))
    freqs = scipy.fft.fftfreq(FFT_LEN)
    
    # 1. Blind the algorithm to the massive sync edge energy
    # by strictly hard-cutting the lower 5% of the spectrum.
    start_bin = max(5, int(FFT_LEN * 0.05))
    end_bin = int(FFT_LEN * 0.40)
    
    if start_bin >= end_bin:
        return 0.0
        
    w_high = 2.0 * np.pi * freqs[start_bin:end_bin]
    p_high = phase[start_bin:end_bin]
    mag_high = np.abs(S_xy[start_bin:end_bin])
    
    # 2. Locate the ringing resonance peak
    # Multiplying by w_high helps flatten out any residual 1/f sync leakage
    resonance_profile = mag_high * w_high
    peak_idx = np.argmax(resonance_profile)
    
    # 3. Create a narrow local window exclusively around the ringing peak
    window_radius = max(3, int(FFT_LEN * 0.015))
    local_start = max(0, peak_idx - window_radius)
    local_end = min(len(w_high), peak_idx + window_radius + 1)
    
    w_local = w_high[local_start:local_end]
    p_local = p_high[local_start:local_end]
    weights_local = resonance_profile[local_start:local_end]
    
    # Normalize local weights for numerical stability in polyfit
    w_max = np.max(weights_local)
    if w_max > 0.0:
        weights_local /= w_max
        
    # 4. Fit the phase slope strictly and locally at the ringing frequency
    # Because group delay = -d(phase)/d(w), the slope of this local tangent 
    # gives us the exact delay of the ringing artifact itself.
    if len(w_local) > 1:
        slope, _ = np.polyfit(w_local, p_local, 1, w=weights_local)
    else:
        slope = 0.0
        
    return -slope


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
    fig = plt.figure(figsize=(16, 20))
    
    ax1 = plt.subplot2grid((6, 2), (0, 0), colspan=2) # Line Inspection
    ax2 = plt.subplot2grid((6, 2), (1, 0), colspan=1) # All Lines Overlay
    ax3 = plt.subplot2grid((6, 2), (1, 1), colspan=1) # Causal Kernel Taps
    ax4 = plt.subplot2grid((6, 2), (2, 0), colspan=2) # Raw Pulse
    ax5 = plt.subplot2grid((6, 2), (3, 0), colspan=2) # Delta
    ax6 = plt.subplot2grid((6, 2), (4, 0), colspan=2) # Corrected Pulse
    ax_slider = plt.subplot2grid((6, 2), (5, 0), colspan=2) # Slider

    initial_idx = 0
    line_num = line_indices[initial_idx]
    
    raw_line, = ax1.plot(raw_pulses[initial_idx], label=f'Raw Line {line_num}', color='#d62728', linewidth=1.8, alpha=0.85)
    corr_line, = ax1.plot(corrected_pulses[initial_idx], label=f'Equalized Line {line_num}', color='#1f77b4', linewidth=2.2)
    ax1.plot(ideal, label='Target Reference', color='#7f7f7f', linestyle=':', linewidth=2)
    
    ax1.set_title(f"Line Inspection [Line 1] | Group Delay Advance: {calc_advance:.3f} samples", fontweight='bold')
    ax1.legend(loc='upper right')
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
    pos_freqs = freqs[:half_len]

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
    green = LinearSegmentedColormap.from_list(
        "green", 
        ["#00ff00", "#000000"]
    )

    # =========================================================================
    # PULSE SPECTROGRAM
    # =========================================================================
    image = ax4.imshow(
        pulse_spec_data,
        aspect='auto',
        origin='lower',
        extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line],
        cmap=green
    )
    cbar = fig.colorbar(image, ax=ax4, pad=0)
    cbar.set_label('Pulse (Raw)')

    ax4.set_title("$Y_{raw}$", fontweight='bold')
    ax4.set_xlabel("Normalized Frequency")
    ax4.set_ylabel("Line Index")

    # =========================================================================
    # SUBTRACTION SPECTROGRAM
    # =========================================================================
    image = ax5.imshow(
        delta_spec_data,
        aspect='auto',
        origin='lower',
        extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line],
        cmap=green
    )
    cbar = fig.colorbar(image, ax=ax5, pad=0)
    cbar.set_label('Delta')

    ax5.set_title("$Y_{delta}$", fontweight='bold')
    ax5.set_xlabel("Normalized Frequency")
    ax5.set_ylabel("Line Index")

    # =========================================================================
    # CORRECTED SPECTROGRAM
    # =========================================================================
    image = ax6.imshow(
        corrected_spec_data,
        aspect='auto',
        origin='lower',
        extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line],
        cmap=green
    )
    cbar = fig.colorbar(image, ax=ax6, pad=0)
    cbar.set_label('Pulse Corrected')

    ax6.set_title("$Y_{delta} - Y_{raw}$", fontweight='bold')
    ax6.set_xlabel("Normalized Frequency")
    ax6.set_ylabel("Line Index")


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

    def update(val):
        idx = int(slider.val)
        l_num = line_indices[idx]
        raw_line.set_ydata(raw_pulses[idx])
        raw_line.set_label(f'Raw Line {l_num}')
        corr_line.set_ydata(corrected_pulses[idx])
        corr_line.set_label(f'Equalized Line {l_num}')
        ax1.set_title(f"Line Inspection [Line {l_num}] | Group Delay Advance: {calc_advance:.3f} samples", fontweight='bold')
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


@nb.njit(cache=True, nogil=True, fastmath=True)
def _normalize_inplace(video_buf, sync_tip_level, blanking_level):
    scale = 2.0 / (blanking_level - sync_tip_level)
    for i in range(video_buf.size):
        video_buf[i] = (video_buf[i] - sync_tip_level) * scale - 1.0


@nb.njit(cache=True, nogil=True, fastmath=True)
def _denormalize_inplace(video_buf, sync_tip_level, blanking_level):
    scale = 0.5 * (blanking_level - sync_tip_level)
    for i in range(video_buf.size):
        video_buf[i] = (video_buf[i] + 1.0) * scale + sync_tip_level


@nb.njit(cache=True, nogil=True, fastmath=True)
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
    noise_threshold=0.2, # TODO, determine automatically; set frequency scaling based on noise floor, or MAD between frequencies
    debug=False
):
    _normalize_inplace(video_buf, sync_tip_level, blanking_level)
    
    n_samples = len(video_buf)
    
    pre_rise_samples = int(np.round(0.25 * sync_len))
    win_size = pre_rise_samples + back_porch_len
    offset_rise = pre_rise_samples
    
    S_xy = np.zeros(FFT_LEN, dtype=np.complex128)
    S_yy = np.zeros(FFT_LEN, dtype=np.float64)
    
    win = np.hanning(win_size)
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

    rolling_S_xy = np.zeros(FFT_LEN, dtype=np.complex128)
    rolling_S_yy = np.zeros(FFT_LEN, dtype=np.float64)

    for measurement in group_delay_state:
        rolling_S_xy += measurement['s_xy']
        rolling_S_yy += measurement['s_yy']

    # =========================================================================
    # PART 2: Analytical Advance & Deconvolution
    # =========================================================================
    # Dynamically extract the optimal floating-point advance from phase slope
    advance_float = _calc_phase_advance(rolling_S_xy)
    
    # Split into integer video shift and fractional kernel phase-shift
    int_adv = int(np.round(advance_float))
    frac_adv = advance_float - int_adv
    
    reg = np.max(rolling_S_yy) * noise_threshold 
    denom = rolling_S_yy + reg
    denom[denom == 0] = 1e-12
    H_inv = rolling_S_xy / denom

    # Apply ONLY the fractional sub-sample advance to the kernel in frequency domain.
    # This securely modifies the filter shape without needing to FFT the whole video buffer.
    w_full = 2.0 * np.pi * scipy.fft.fftfreq(FFT_LEN)
    H_inv = H_inv * np.exp(-1j * w_full * frac_adv)

    # IFFT back to time domain
    h_full = scipy.fft.fftshift(scipy.fft.ifft(H_inv).real)
    
    # The Wiener deconvolution naturally places the main impulse offset by int_adv.
    # By shifting our extraction center to match this offset, we extract the causal tail 
    # perfectly aligned.
    true_center = FFT_LEN // 2 - int_adv
    
    # Isolate causal tail taps for t >= 1
    fir_tail = h_full[true_center + 1:].copy()

    # Assemble half-size causal kernel
    fir_kernel = np.zeros(FIR_LEN, dtype=np.float32)
    
    # Safely insert the tail up to the maximum available filter length
    insert_len = min(len(fir_tail), FIR_LEN - 1)
    fir_kernel[1:1 + insert_len] = fir_tail[:insert_len]
 
    # Fade out the tail using a half-cosine window
    fade_len = round(FIR_LEN * 0.15) # fade out last 15%
    fade_axis = np.arange(fade_len, dtype=np.float64)
    fir_kernel[FIR_LEN - fade_len:] *= 0.5 * (1.0 + np.cos(np.pi * fade_axis / fade_len))

    # Base DC Normalization: Enforce DC Unity strictly on the center tap (t=0)
    fir_kernel[0] = 1.0 - np.sum(fir_kernel[1:])

    if debug:
        full_win_size = max(front_porch_len + sync_len + back_porch_len, FIR_LEN)
        raw_pulses = []
        corrected_pulses = []
        line_indices = []
        pulse_ffts = []
        delta_ffts = []
        corrected_ffts = []

        for line in range(line_start, line_end + 1):
            start_loc = line * line_length - front_porch_len
            if start_loc >= 0 and start_loc + full_win_size < n_samples:
                raw_p = video_buf[start_loc : start_loc + full_win_size].copy()
                corr_p = _apply_inverse_eq(raw_p, full_win_size, fir_kernel)

                raw_pulses.append(raw_p)
                corrected_pulses.append(corr_p)
                line_indices.append(line)

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
            phase_delay=-(advance_float+int_adv),
        )
        ideal_fft = _build_ideal_hsync_pulse(
            front_porch_len,
            sync_len,
            target_transition,
            sync_tip_average,
            blanking_average,
            FFT_LEN,
            phase_delay=-(advance_float+int_adv),
        )
        y_ideal = scipy.fft.fft(ideal_fft, n=FFT_LEN)

        # get ffts of all the pulses before, after, amount of change all subtracted from the ideal pulse
        for i in range(len(raw_pulses)):
            # Capture FFT of raw and corrected pulses
            y_raw = scipy.fft.fft(raw_pulses[i], n=FFT_LEN)
            y_corr = scipy.fft.fft(corrected_pulses[i], n=FFT_LEN)
            y_delta = (y_corr - y_raw)
    
            pulse_ffts.append(y_raw - y_ideal)
            delta_ffts.append(y_ideal - y_delta)
            corrected_ffts.append(y_corr - y_ideal)
                    
        _show_group_delay_debug(
            raw_pulses,
            corrected_pulses,
            ideal_debug,
            fir_kernel,
            advance_float,
            line_indices,
            pulse_ffts,
            delta_ffts,
            corrected_ffts,
            FFT_LEN,
        )

    # =========================================================================
    # PART 3: Apply Strictly Causal Equalization
    # =========================================================================
    video_buf = _apply_inverse_eq(video_buf, n_samples, fir_kernel)

    _denormalize_inplace(video_buf, sync_tip_level, blanking_level)

    lti_params = derive_lti_parameters(
        fir_kernel, 
        noise_threshold=noise_threshold
    )

    return video_buf, lti_params


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