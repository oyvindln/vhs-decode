import numpy as np
import numba as nb
import scipy.signal

# -----------------------------------------------------------------------------
# 1. NUMBA KERNEL: STRICTLY CAUSAL PHASE EQUALIZATION FIR
# -----------------------------------------------------------------------------
@nb.njit(cache=True, nogil=True, fastmath=True)
def _apply_global_inverse_eq(
    tbc_video,
    n_samples,
    fir_kernel
):
    out = np.empty_like(tbc_video)
    k_len = len(fir_kernel)
    half_k = k_len // 2
    
    for i in range(n_samples):
        val = 0.0
        for j in range(k_len):
            idx = i + half_k - j
            if 0 <= idx < n_samples:
                val += tbc_video[idx] * fir_kernel[j]
            else:
                if idx < 0:
                    val += tbc_video[0] * fir_kernel[j]
                else:
                    val += tbc_video[n_samples - 1] * fir_kernel[j]
        out[i] = val
        
    return out


# -----------------------------------------------------------------------------
# 2. DEBUG PLOT RENDERER
# -----------------------------------------------------------------------------
def _render_debug_plot(
    avg_pulse_shape,
    corrected_pulse,
    ideal,
    fir_kernel,
    target_transition
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required to render the debug plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    ax1.plot(avg_pulse_shape, label='Before (Raw Avg Pulse)', color='#d62728', linewidth=1.8)
    ax1.plot(corrected_pulse, label='After (Causal Group Delay Equalization)', color='#1f77b4', linewidth=2.2)
    ax1.plot(ideal, label=f'Target Spec Step (Trans: {target_transition})', color='#7f7f7f', linestyle=':', linewidth=2)
    ax1.set_title("System Identification: Strictly Causal Group Delay Equalization")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    center = len(fir_kernel) // 2
    x_axis = np.arange(-center, center + 1)
    
    ax2.plot(x_axis, fir_kernel, label='Causal FIR Taps (t >= 0)', color='purple', marker='.', linewidth=1.5)
    ax2.set_title("Derived Causal Group-Delay Kernel (Zero Pre-Shoot)")
    ax2.axhline(0, color='black', alpha=0.3)
    ax2.set_xlim(-40, 40)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def build_ideal_step(
    measured_center_fall,
    measured_center_rise,
    target_transition,
    local_blank_f,
    local_sync,
    local_blank_r,
    win_size,
):
    mid_sync = int(measured_center_fall + (measured_center_rise - measured_center_fall) / 2.0)
    ideal = np.zeros(win_size, dtype=np.float64)
    for i in range(win_size):
        t_fall = float(i) - measured_center_fall
        t_rise = float(i) - measured_center_rise

        if i <= mid_sync:
            if t_fall < -target_transition / 2.0:
                ideal[i] = local_blank_f
            elif t_fall <= target_transition / 2.0:
                phase = (t_fall + target_transition / 2.0) / target_transition * np.pi
                ideal[i] = local_blank_f + (local_sync - local_blank_f) * (1.0 - np.cos(phase)) / 2.0
            else:
                ideal[i] = local_sync
        else:
            if t_rise < -target_transition / 2.0:
                ideal[i] = local_sync
            elif t_rise <= target_transition / 2.0:
                phase = (t_rise + target_transition / 2.0) / target_transition * np.pi
                ideal[i] = local_sync + (local_blank_r - local_sync) * (1.0 - np.cos(phase)) / 2.0
            else:
                ideal[i] = local_blank_r
    
    return ideal


# -----------------------------------------------------------------------------
# 3. MAIN CORRECTION FUNCTION
# -----------------------------------------------------------------------------
def apply_tbc_vhs_deemphasis_correction(
    video_buf, 
    line_start, 
    line_end, 
    line_length, 
    blanking_level_in, 
    sync_tip_level_in, 
    front_porch_len=15, 
    sync_len=67,         
    back_porch_len=67,   
    target_transition=3.3,
    noise_threshold=0.01,
    fir_length=129,
    debug=False
):
    blanking_level = 0.0
    sync_tip_level = -40.0
    
    n_samples = len(video_buf)
    win_size = front_porch_len + sync_len + back_porch_len
    offset_fall = front_porch_len
    offset_rise = front_porch_len + sync_len
    mid_val = (blanking_level + sync_tip_level) / 2.0
    
    pad_len = 2048 
    S_xy = np.zeros(pad_len, dtype=np.complex128)
    S_yy = np.zeros(pad_len, dtype=np.float64)
    
    win = np.hanning(win_size)
    start_idx = (pad_len - win_size) // 2
    count = 0
    
    avg_pulse_shape = np.zeros(win_size, dtype=np.float64)

    # --- Step 1: Accumulate Cross-Spectral Density ---
    for line in range(line_start, line_end + 1):
        loc = line * line_length
        if loc - offset_fall >= 0 and loc + sync_len + back_porch_len < n_samples:
            
            pulse = video_buf[loc - offset_fall : loc + sync_len + back_porch_len]
            avg_pulse_shape += pulse
            count += 1
            
            measured_center_fall = float(offset_fall)
            for i in range(win_size - 1):
                if pulse[i] >= mid_val >= pulse[i+1]:
                    y0 = pulse[i] - mid_val
                    y1 = pulse[i+1] - mid_val
                    if (y0 - y1) != 0.0:
                        measured_center_fall = float(i) + abs(y0) / abs(y0 - y1)
                    break

            measured_center_rise = float(offset_rise)
            search_start = int(measured_center_fall + (sync_len // 2))
            for i in range(search_start, win_size - 1):
                if pulse[i] <= mid_val <= pulse[i+1]:
                    y0 = pulse[i] - mid_val
                    y1 = pulse[i+1] - mid_val
                    if (y1 - y0) != 0.0:
                        measured_center_rise = float(i) + abs(y0) / abs(y1 - y0)
                    break
                    
            sum_fp, count_fp = 0.0, 0
            for i in range(0, max(1, int(measured_center_fall - 4))):
                sum_fp += pulse[i]
                count_fp += 1
            local_blank_f = sum_fp / count_fp if count_fp > 0 else blanking_level
            
            sum_sync, count_sync = 0.0, 0
            for i in range(int(measured_center_fall + 12), int(measured_center_rise - 12)):
                sum_sync += pulse[i]
                count_sync += 1
            local_sync = sum_sync / count_sync if count_sync > 0 else sync_tip_level
            
            sum_bp, count_bp = 0.0, 0
            for i in range(int(measured_center_rise + 4), win_size):
                sum_bp += pulse[i]
                count_bp += 1
            local_blank_r = sum_bp / count_bp if count_bp > 0 else blanking_level

            shared_blank = (local_blank_f + local_blank_r) / 2.0
            
            ideal_for_fir = build_ideal_step(
                measured_center_fall, measured_center_rise, target_transition,
                shared_blank, local_sync, shared_blank, win_size
            )

            d_ideal = np.gradient(ideal_for_fir) * win
            d_pulse = np.gradient(pulse) * win
            
            pad_ideal = np.zeros(pad_len, dtype=np.float64)
            pad_pulse = np.zeros(pad_len, dtype=np.float64)
            pad_ideal[start_idx : start_idx + win_size] = d_ideal
            pad_pulse[start_idx : start_idx + win_size] = d_pulse
            
            X = np.fft.fft(pad_ideal)
            Y = np.fft.fft(pad_pulse)
            
            S_xy += X * np.conj(Y)
            S_yy += np.abs(Y)**2

    if count == 0:
        return video_buf
        
    avg_pulse_shape /= float(count)
    
    # --- Step 2: Inverse Deconvolution Model ---
    reg = np.max(S_yy) * noise_threshold 
    denom = S_yy + reg
    denom[denom == 0] = 1e-12
    
    H_inv = S_xy / denom

    # --- Step 3: Extract Time-Domain Kernel and Enforce Strict Causality ---
    h_full = np.fft.fftshift(np.fft.ifft(H_inv).real)
    center_full = pad_len // 2

    if fir_length % 2 == 0:
        fir_length += 1
        
    half_fir = fir_length // 2
    
    # Align $t=0$ precisely to index `half_fir`
    peak_idx = np.argmax(np.abs(h_full))
    h_full = np.roll(h_full, center_full - peak_idx)
    
    fir_kernel = h_full[center_full - half_fir : center_full + half_fir + 1].copy()

    # 1. Zero out negative time (t < 0) to strictly enforce causality.
    # This completely eliminates pre-shoot and future-looking phase errors.
    fir_kernel[:half_fir] = 0.0

    # 2. Smooth the causal forward-time tail (t >= 0) to prevent truncation ripples.
    causal_len = len(fir_kernel) - half_fir
    causal_window = np.hanning(2 * causal_len - 1)[causal_len - 1:]
    fir_kernel[half_fir:] *= causal_window

    # 3. Enforce strict DC unity gain (DC = 1.0)
    fir_sum = np.sum(fir_kernel)
    if fir_sum != 0.0:
        fir_kernel /= fir_sum

    # --- Step 4: Apply Strictly Causal Equalization ---
    video_buf = _apply_global_inverse_eq(
        video_buf,
        n_samples,
        fir_kernel
    )

    if debug:
        corrected_pulse = _apply_global_inverse_eq(
            avg_pulse_shape,
            win_size,
            fir_kernel
        )
        
        shared_blank = (blanking_level + blanking_level) / 2.0
        ideal_debug = build_ideal_step(
            offset_fall, offset_rise, target_transition,
            shared_blank, sync_tip_level, shared_blank, win_size
        )
        
        _render_debug_plot(
            avg_pulse_shape,
            corrected_pulse,
            ideal_debug,
            fir_kernel,
            target_transition
        )

    return video_buf