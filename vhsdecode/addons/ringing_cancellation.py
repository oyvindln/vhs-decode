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

FIR_LEN = 127
FFT_LEN = 512
DELAY_OFFSET = FIR_LEN // 2  # Anchors the causal peak


def _make_interpolated_inverse_eq(fir_len, delay_offset):
    @nb.njit(
        "float32[:](float32[::1], float32[::1], int64, float32[::1], float32[::1])",
        cache=True,
        nogil=True,
        fastmath=True
    )
    def _apply_interpolated_inverse_eq(picture_raw, picture_clean, n_samples, kernel_a, kernel_b):
        out = np.empty(n_samples, dtype=np.float32)
        first = picture_clean[0]
        last = picture_clean[n_samples - 1]

        # Pre-allocate tap buffer for the per-pixel interpolated kernel
        h_interp = np.empty(fir_len, dtype=np.float32)

        for k in range(n_samples):
            # 1. Calculate linear interpolation factor alpha along the line (0.0 to 1.0)
            alpha = np.float32(k) / np.float32(n_samples - 1) if n_samples > 1 else np.float32(0.0)
            one_minus_alpha = np.float32(1.0) - alpha

            # 2. Interpolate FIR taps directly for current pixel k
            for j in range(fir_len):
                h_interp[j] = one_minus_alpha * kernel_a[j] + alpha * kernel_b[j]

            # 3. Apply the causal filter sweep for pixel k
            read_head = k + delay_offset
            acc = 0.0

            for j in range(fir_len):
                idx = read_head - j
                
                # Boundary clamping
                if idx < 0:
                    val = first
                elif idx >= n_samples:
                    val = last
                else:
                    val = picture_clean[idx]

                acc += val * h_interp[j]

            # 4. Delta Injection
            delta = acc - picture_clean[k]
            out[k] = picture_raw[k] + delta

        return out
    return _apply_interpolated_inverse_eq

_apply_interpolated_inverse_eq = _make_interpolated_inverse_eq(FIR_LEN, DELAY_OFFSET)


@nb.njit("float32[:](float32[::1], float32[::1])", cache=True, nogil=True, fastmath=True)
def _apply_zero_phase_fir(picture, fir_kernel):
    n_samples = len(picture)
    fir_len = len(fir_kernel)
    half_fir = fir_len // 2
    out = np.empty(n_samples, np.float32)
    
    first = picture[0]
    last = picture[n_samples - 1]

    for i in nb.prange(n_samples):
        acc = 0.0
        for j in range(fir_len):
            idx = i + j - half_fir
            if idx < 0:
                val = first
            elif idx >= n_samples:
                val = last
            else:
                val = picture[idx]
            acc += val * fir_kernel[j]
        out[i] = acc
        
    return out


def _build_fsc_mask(n, width_bins=3):
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


@nb.njit(cache=True, nogil=True, fastmath=True)
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


@nb.njit(cache=True, nogil=True, fastmath=True)
def _build_ideal_hsync_pulse(
    front_porch_len,
    sync_len,
    target_transition,
    sync_level,
    blanking_level,
    win_size,
    phase_delay=0.0,
):
    ideal = np.full(win_size, blanking_level, dtype=np.float64)
    
    center_fall = float(front_porch_len) + phase_delay
    center_rise = float(front_porch_len + sync_len) + phase_delay
    
    for i in range(win_size):
        t_fall = float(i) - center_fall
        t_rise = float(i) - center_rise
        
        if t_fall >= -target_transition / 2.0 and t_fall <= target_transition / 2.0:
            phase = (t_fall + target_transition / 2.0) / target_transition * np.pi
            ideal[i] = blanking_level - (blanking_level - sync_level) * (1.0 - np.cos(phase)) / 2.0
        elif t_fall > target_transition / 2.0 and t_rise < -target_transition / 2.0:
            ideal[i] = sync_level
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


def _suppress_color_carrier_fft(samples):
    out = samples.copy()
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


def _build_causal_fir_from_spectrum(H_inv, fft_len, fir_len, delay_offset):
    freqs_norm = np.abs(np.fft.fftfreq(fft_len))
    
    taper = np.clip((0.45 - freqs_norm) / 0.10, 0.0, 1.0)
    taper_smooth = 0.5 * (1.0 - np.cos(np.pi * taper)) 
    H_inv_tapered = H_inv * taper_smooth

    omega = 2.0 * np.pi * np.fft.fftfreq(fft_len)
    causal_phase_shift = np.exp(-1j * omega * delay_offset)
    H_inv_tapered *= causal_phase_shift

    h_time = scipy.fft.ifft(H_inv_tapered).real
    fir_kernel = np.zeros(fir_len, dtype=np.float32)
    fir_kernel[:] = h_time[0 : fir_len]

    fir_kernel[0 : delay_offset] = 0.0

    tail_len = fir_len - delay_offset
    tail_window = scipy.signal.windows.tukey(tail_len * 2, alpha=0.5)[tail_len:]
    fir_kernel[delay_offset : fir_len] *= tail_window

    kernel_sum = np.sum(fir_kernel)
    if abs(kernel_sum) > 1e-12:
        fir_kernel /= kernel_sum

    return fir_kernel


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
    noise_threshold=1,
    interpolate_amplitude=True,
    interpolate_phase=True,
    interpolate_time=True, # TODO: allow interpolation to be disabled for speed
    debug=False
):
    _normalize_inplace(video_buf, sync_tip_level, blanking_level)
    n_samples = len(video_buf)

    # =========================================================================
    # PART 0: Measure the Constant Noise Floor
    # =========================================================================
    noise_power_sum = np.zeros(FFT_LEN, dtype=np.float64)
    noise_count = 0
    
    sync_tip_flat_len = int(sync_len * 0.4)
    win_noise = scipy.signal.windows.hann(sync_tip_flat_len)
    start_idx_noise = (FFT_LEN - sync_tip_flat_len) // 2

    for line in range(line_start, line_end + 1):
        loc = line * line_length
        tip_start = loc + front_porch_len + int(sync_len * 0.3)
        if tip_start >= 0 and tip_start + sync_tip_flat_len < n_samples:
            tip_chunk = video_buf[tip_start : tip_start + sync_tip_flat_len]
            d_tip = np.gradient(tip_chunk) * win_noise
            
            pad_tip = np.zeros(FFT_LEN, dtype=np.float64)
            pad_tip[start_idx_noise : start_idx_noise + sync_tip_flat_len] = d_tip
            
            noise_power_sum += np.abs(scipy.fft.fft(pad_tip))**2
            noise_count += 1

    if noise_count > 0:
        noise_mag_profile = np.sqrt(noise_power_sum / noise_count) * 1.5
    else:
        noise_mag_profile = np.zeros(FFT_LEN, dtype=np.float64)

    # =========================================================================
    # PARTS 1, 2, 3: Per-Line Wiener Deconvolution with 1000-Pulse Ring Buffer
    # =========================================================================
    MAX_HISTORY_PULSES = 1000

    if len(group_delay_state) == 0:
        state_dict = {
            's_xy_history': np.zeros((MAX_HISTORY_PULSES, FFT_LEN), dtype=np.complex128),
            's_yy_history': np.zeros((MAX_HISTORY_PULSES, FFT_LEN), dtype=np.float64),
            'ptr': 0,
            'count': 0
        }
        group_delay_state.append(state_dict)
    else:
        state_dict = group_delay_state[0]
        if 's_xy_history' not in state_dict:
            state_dict['s_xy_history'] = np.zeros((MAX_HISTORY_PULSES, FFT_LEN), dtype=np.complex128)
            state_dict['s_yy_history'] = np.zeros((MAX_HISTORY_PULSES, FFT_LEN), dtype=np.float64)
            state_dict['ptr'] = 0
            state_dict['count'] = 0

    pre_rise_samples = int(np.round(0.25 * sync_len))
    win_size = pre_rise_samples + back_porch_len
    win_size -= win_size % 4 
    offset_rise = pre_rise_samples
    start_idx = (FFT_LEN - win_size) // 2

    freqs = np.abs(np.fft.fftfreq(FFT_LEN))
    hf_mask = freqs > 0.35  

    count = 0
    raw_line_spectra = []
    raw_line_rises = []
    
    default_kernel = np.zeros(FIR_LEN, dtype=np.float32)
    default_kernel[DELAY_OFFSET] = 1.0
    last_valid_kernel = default_kernel
    last_valid_h_inv = np.ones(FFT_LEN, dtype=np.complex128)

    for line in range(line_start, line_end + 1):
        loc = line * line_length
        start_loc = loc + sync_len - pre_rise_samples
        
        S_xy = np.zeros(FFT_LEN, dtype=np.complex128)
        S_yy = np.zeros(FFT_LEN, dtype=np.float64)
        measured_center_rise = float(offset_rise)
        is_outlier = False

        if start_loc >= 0 and start_loc + win_size < n_samples:
            pulse = video_buf[start_loc : start_loc + win_size]

            measured_sync_tip_level, measured_blanking_level = _get_levels(pulse)
            mid_val = measured_sync_tip_level + (measured_blanking_level - measured_sync_tip_level) / 2.0

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

            peak_idx = int(measured_center_rise)
            transient_win = np.zeros(win_size, dtype=np.float64)
            
            if peak_idx > 0:
                transient_win[0:peak_idx] = 0.5 * (1.0 - np.cos(np.pi * np.arange(peak_idx) / peak_idx))
            
            ringing_tail_len = int(target_transition * 5.0) 
            right_width = min(ringing_tail_len, win_size - peak_idx)
            
            if right_width > 0:
                transient_win[peak_idx : peak_idx + right_width] = 0.5 * (1.0 + np.cos(np.pi * np.arange(right_width) / right_width))

            d_ideal = np.gradient(ideal_for_fir) * transient_win
            d_pulse = np.gradient(pulse) * transient_win
            
            pad_ideal = np.zeros(FFT_LEN, dtype=np.float64)
            pad_pulse = np.zeros(FFT_LEN, dtype=np.float64)
            pad_ideal[start_idx : start_idx + win_size] = d_ideal
            pad_pulse[start_idx : start_idx + win_size] = d_pulse

            Y = scipy.fft.fft(pad_pulse)
            X = scipy.fft.fft(pad_ideal)

            current_s_xy = X * np.conj(Y)
            current_s_yy = np.abs(Y)**2

            # --- 1000-PULSE RING BUFFER OUTLIER REJECTION ---
            if state_dict['count'] > (525 / 2): # roughly a field's work of lines in order to honor the rolling average
                historical_mean_s_yy = np.mean(state_dict['s_yy_history'][:state_dict['count']], axis=0)
                spectral_distance = np.mean(np.abs(current_s_yy - historical_mean_s_yy))
                avg_power_magnitude = np.mean(historical_mean_s_yy) + 1e-12
                relative_deviation = spectral_distance / avg_power_magnitude
                
                if relative_deviation > (0.4 * noise_threshold + 0.20):
                    is_outlier = True

            if not is_outlier:
                ptr = state_dict['ptr']
                state_dict['s_xy_history'][ptr] = current_s_xy
                state_dict['s_yy_history'][ptr] = current_s_yy
                state_dict['ptr'] = (ptr + 1) % MAX_HISTORY_PULSES
                state_dict['count'] = min(MAX_HISTORY_PULSES, state_dict['count'] + 1)
                count += 1
        else:
            is_outlier = True

        if is_outlier:
            raw_line_spectra.append(last_valid_h_inv)
            raw_line_rises.append(measured_center_rise)
            continue

        # Compute averaged spectra over the entire 1000-pulse training set history
        valid_count = state_dict['count']
        accum_s_xy = np.mean(state_dict['s_xy_history'][:valid_count], axis=0)
        accum_s_yy = np.mean(state_dict['s_yy_history'][:valid_count], axis=0)

        hf_power = accum_s_yy[hf_mask]
        hf_median = np.median(hf_power) if len(hf_power) > 0 else 0.0
        hf_mad = np.median(np.abs(hf_power - hf_median)) if len(hf_power) > 0 else 0.0
        noise_penalty = (hf_median + 3.0 * hf_mad) * noise_threshold

        num_passes = 5  
        H_total = np.ones(FFT_LEN, dtype=np.complex128)

        for pass_idx in range(num_passes):
            current_S_xy = accum_s_xy * np.conj(H_total)
            current_S_yy = accum_s_yy * (np.abs(H_total) ** 2)

            denom = current_S_yy + noise_penalty
            denom[denom == 0] = 1e-12
            H_step = current_S_xy / denom

            alpha = 0.4 
            H_total = H_total * (1.0 - alpha) + H_total * H_step * alpha

        raw_line_spectra.append(H_total)
        raw_line_rises.append(measured_center_rise)
        last_valid_h_inv = H_total

    if count == 0 and state_dict['count'] == 0:
        _denormalize_inplace(video_buf, sync_tip_level, blanking_level)
        return video_buf, {'gain': 0.0, 'threshold': 0.1, 'blur_radius': 0.0}

    # =========================================================================
    # FIELD-WIDE STATISTICAL AVERAGING (For disabled interpolation toggles)
    # =========================================================================
    field_magnitudes = [np.abs(h) for h in raw_line_spectra]
    field_phases = [np.angle(h) for h in raw_line_spectra]

    avg_amplitude = np.mean(field_magnitudes, axis=0)
    avg_phase = np.angle(np.mean([np.exp(1j * p) for p in field_phases], axis=0))

    line_kernels = []
    line_H_invs = []

    for i in range(len(raw_line_spectra)):
        mag = np.abs(raw_line_spectra[i]) if interpolate_amplitude else avg_amplitude
        phase = np.angle(raw_line_spectra[i]) if interpolate_phase else avg_phase
        
        H_inv_modified = mag * np.exp(1j * phase)
        
        fir_kernel = _build_causal_fir_from_spectrum(H_inv_modified, FFT_LEN, FIR_LEN, DELAY_OFFSET)
        line_kernels.append(fir_kernel)
        line_H_invs.append(H_inv_modified)

    # =========================================================================
    # PART 4: Execute Delta Injection via Sliding Interpolation
    # =========================================================================
    video_clean = _suppress_color_carrier_fft(video_buf)
    video_buf_filtered = video_buf.copy()
    
    for i, line in enumerate(range(line_start, line_end + 1)):
        loc = line * line_length
        if loc >= len(video_buf):
            break
            
        raw_line = video_buf[loc : loc + line_length]
        clean_line = video_clean[loc : loc + line_length]
        
        kernel_a = line_kernels[i]
        kernel_b = line_kernels[i + 1] if i + 1 < len(line_kernels) else kernel_a
        
        filtered_line = _apply_interpolated_inverse_eq(
            raw_line.astype(np.float32), 
            clean_line.astype(np.float32), 
            len(raw_line), 
            kernel_a, 
            kernel_b
        )
        
        video_buf_filtered[loc : loc + line_length] = filtered_line

    # =========================================================================
    # PART 5: Exact Inverse Noise Floor Compensation (Frequency-Gated)
    # =========================================================================
    avg_H_inv = np.mean(line_H_invs, axis=0) if len(line_H_invs) > 0 else np.ones(FFT_LEN, dtype=np.complex128)

    gd_mag = np.abs(avg_H_inv)
    gd_boost = np.maximum(1.0, gd_mag)
    inverse_multiplier = 1.0 / gd_boost

    freqs_norm = np.abs(np.fft.fftfreq(FFT_LEN))
    luma_protection = np.clip((freqs_norm - 0.10) / 0.10, 0.0, 1.0)

    mean_noise = np.mean(noise_mag_profile)
    noise_weight = np.clip((noise_mag_profile / (mean_noise + 1e-12)) * 2.0, 0.0, 1.0)
    noise_weight *= luma_protection

    H_corrective = 1.0 - (noise_weight * (1.0 - inverse_multiplier))
    H_corrective[0] = 1.0

    h_corrective_time = scipy.fft.fftshift(
        scipy.fft.ifft(H_corrective).real
    )

    center = FFT_LEN // 2
    fir_corrective = np.zeros(FIR_LEN, dtype=np.float32)

    start_idx_corr = center - (FIR_LEN // 2)
    fir_corrective[:] = h_corrective_time[start_idx_corr:start_idx_corr + FIR_LEN]

    fir_corrective *= scipy.signal.windows.hann(FIR_LEN)

    fir_corr_sum = np.sum(fir_corrective)
    if abs(fir_corr_sum) > 1e-12:
        fir_corrective /= fir_corr_sum

    video_buf_filtered = _apply_zero_phase_fir(
        video_buf_filtered.astype(np.float32),
        fir_corrective.astype(np.float32)
    )

    mid_kernel = line_kernels[len(line_kernels) // 2] if len(line_kernels) > 0 else np.zeros(FIR_LEN)
    lti_params = derive_lti_parameters(mid_kernel, noise_threshold=noise_threshold)

    if debug:
        full_win_size = max(front_porch_len + sync_len + back_porch_len, FIR_LEN)
        pulse_ffts = []
        delta_ffts = []
        corrected_ffts = []
        raw_pulses = []
        corrected_pulses = []
        line_indices = []

        for idx, line in enumerate(range(line_start, line_end + 1)):
            start_loc = line * line_length - front_porch_len
            if start_loc >= 0 and start_loc + full_win_size < n_samples:
                raw_p = video_buf[start_loc : start_loc + full_win_size].copy()
                clean_p = video_clean[start_loc : start_loc + full_win_size].copy()
                
                kernel_a = line_kernels[idx]
                kernel_b = line_kernels[idx + 1] if idx + 1 < len(line_kernels) else kernel_a

                corr_p = _apply_interpolated_inverse_eq(
                    raw_p.astype(np.float32), 
                    clean_p.astype(np.float32), 
                    full_win_size, 
                    kernel_a,
                    kernel_b
                )
                
                corr_p = _apply_zero_phase_fir(
                    corr_p.astype(np.float32), 
                    fir_corrective.astype(np.float32)
                )

                raw_pulses.append(raw_p)
                corrected_pulses.append(corr_p)
                line_indices.append(line)

                pulse_fft = scipy.fft.fft(raw_p, n=FFT_LEN)
                corrected_fft = scipy.fft.fft(corr_p, n=FFT_LEN)
                delta_fft = corrected_fft - pulse_fft

                pulse_ffts.append(pulse_fft)
                corrected_ffts.append(corrected_fft)
                delta_ffts.append(delta_fft)

        all_corr_arr = np.array(corrected_pulses)
        sync_tip_average, blanking_average = _get_levels(all_corr_arr.flatten())

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
            mid_kernel,
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
    
    ax1 = plt.subplot2grid((6, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((6, 2), (1, 0), colspan=1)
    ax3 = plt.subplot2grid((6, 2), (1, 1), colspan=1)
    ax_slider = plt.subplot2grid((6, 1), (2, 0), colspan=2)

    ax4 = plt.subplot2grid((6, 2), (3, 0), colspan=2)
    ax5 = plt.subplot2grid((6, 2), (4, 0), colspan=2, sharex=ax4, sharey=ax4)
    ax6 = plt.subplot2grid((6, 2), (5, 0), colspan=2, sharex=ax4, sharey=ax4)

    initial_idx = 0
    line_num = line_indices[initial_idx]
    
    raw_line, = ax1.plot(raw_pulses[initial_idx], label=f'Raw Line {line_num}', color='#d62728', linewidth=1.8, alpha=0.85)
    corr_line, = ax1.plot(corrected_pulses[initial_idx], label=f'Equalized Line {line_num}', color='#1f77b4', linewidth=2.2)
    ax1.plot(ideal, label='Target Reference', color='#7f7f7f', linestyle=':', linewidth=2)
    
    ax1.set_title(f"Line Inspection [Line {line_num}] | Group Delay Advance: {calc_advance:.3f} samples", fontweight='bold')
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

    green_pos = LinearSegmentedColormap.from_list("green", ["#000000", "#00ff00"])   
    green_neg = LinearSegmentedColormap.from_list("green", ["#00ff00", "#000000"])

    ax4.imshow(pulse_spec_data, aspect='auto', origin='lower', extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line], cmap=green_pos)
    ax4.set_title("Raw Y_raw", fontweight='bold')
    ax4.set_adjustable('box')

    ax5.imshow(delta_spec_data, aspect='auto', origin='lower', extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line], cmap=green_neg)
    ax5.set_title("Delta Y_delta", fontweight='bold')
    ax5.set_xlabel("Frequency (MHz)")
    ax5.set_ylabel("Line Index")
    ax5.set_adjustable('box')

    ax6.imshow(corrected_spec_data, aspect='auto', origin='lower', extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line], cmap=green_pos)
    ax6.set_title("Corrected Y_delta - Y_raw", fontweight='bold')
    ax6.set_adjustable('box')
    ax6.set_xlabel("Normalized Frequency")

    ax6.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax6.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    for ax in (ax4, ax5, ax6):
        ax.set_xlim(pos_freqs[0], pos_freqs[-1])
        ax.set_ylim(min_line, max_line)

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

    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05, hspace=0.25)
    plt.show()


# -----------------------------------------------------------------------------
# LTI PARAMETER DERIVATION FUNCTION
# -----------------------------------------------------------------------------
def derive_lti_parameters(fir_kernel, noise_threshold):
    total_energy = np.sum(fir_kernel**2)
    if total_energy <= 1e-12:
        return {'gain': 0.0, 'threshold': 0.1, 'blur_radius': 0.0}

    center_energy = fir_kernel[0]**2
    energy_dispersion = 1.0 - (center_energy / total_energy)
    
    indices = np.arange(len(fir_kernel))
    weighted_spread = np.sum(indices * np.abs(fir_kernel)) / np.sum(np.abs(fir_kernel))
    
    noise_suppression_factor = max(0.2, 1.0 - 2.0 * noise_threshold)
    lti_gain = np.clip(energy_dispersion * 1.5 * noise_suppression_factor, 0.0, 1.0)
    
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
    n = len(video_buf)
    prev = video_buf[0]

    for i in range(1, n - 1):
        curr = video_buf[i]
        nxt = video_buf[i + 1]

        abs_diff = abs(nxt - prev)

        if abs_diff > threshold:
            laplacian = prev - 2.0 * curr + nxt
            edge_weight = min(1.0, abs_diff / (2.0 * threshold))
            video_buf[i] = curr - (gain * edge_weight * laplacian)

        prev = curr