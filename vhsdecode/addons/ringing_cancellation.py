import math
import numpy as np
import numba as nb
import scipy.fft
import scipy
from scipy.interpolate import UnivariateSpline
import scipy.optimize

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. NUMBA KERNEL: CAUSAL PHASE EQUALIZATION FIR
# -----------------------------------------------------------------------------

FSC = 3.579545 
FPS = 525
FSC_RATIO = 4

MAX_AVG_MEASUREMENTS = int(FPS)
FIR_LEN = 31
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
    state,
    burst_phase_avg,
    use_amplitude=True,
    use_phase=True,
    interpolate_time=True, 
    debug=False
):
    _normalize_inplace(video_buf, sync_tip_level, blanking_level)

    if not state:
        state['s_xy_history'] = np.zeros((MAX_AVG_MEASUREMENTS, FFT_LEN), dtype=np.complex128)
        state['s_yy_history'] = np.zeros((MAX_AVG_MEASUREMENTS, FFT_LEN), dtype=np.float64)
        state['ptr'] = 0
        state['count'] = 0

    fft_freqs = np.fft.fftfreq(FFT_LEN)

    # =========================================================================
    # PART 1: Pulse Measurement (Frequency Domain Preparation)
    # =========================================================================

    pulse_data = []
    pulse_pad_sum = np.zeros(FFT_LEN, dtype=np.float64)

    pre_rise_samples = int(np.round(0.25 * sync_len))
    win_size = pre_rise_samples + back_porch_len
    win = np.hamming(win_size)
    offset_rise = pre_rise_samples

    for line in range(line_start, line_end):
        loc = line * line_length

        # Extract the full horizontal blanking interval
        pulse_start = max(0, loc + sync_len - pre_rise_samples)
        pulse_end = min(len(video_buf), pulse_start + win_size)
        pulse_len = pulse_end - pulse_start

        pulse_measured = video_buf[pulse_start:pulse_end]
        measured_sync_tip_level, measured_blanking_level = _get_levels(pulse_measured)

        mid_val = measured_sync_tip_level + (measured_blanking_level - measured_sync_tip_level) / 2.0

        measured_center_rise = float(offset_rise)
        for i in range(win_size - 1):
            if pulse_measured[i] <= mid_val <= pulse_measured[i+1]:
                y0 = pulse_measured[i] - mid_val
                y1 = pulse_measured[i+1] - mid_val
                if (y1 - y0) != 0.0:
                    measured_center_rise = float(i) + abs(y0) / abs(y1 - y0)
                break
                
        sum_sync, count_sync = 0.0, 0
        end_sync_idx = max(1, int(measured_center_rise - 2))
        for i in range(0, end_sync_idx):
            sum_sync += pulse_measured[i]
            count_sync += 1
        local_sync = sum_sync / count_sync if count_sync > 0 else measured_sync_tip_level
        
        sum_bp, count_bp = 0.0, 0
        start_bp_idx = min(win_size - 1, int(measured_center_rise + 2))
        for i in range(start_bp_idx, win_size):
            sum_bp += pulse_measured[i]
            count_bp += 1
        local_blank_r = sum_bp / count_bp if count_bp > 0 else measured_blanking_level

        ideal_for_fir = _build_ideal_step(
            measured_center_rise, target_transition,
            local_sync, local_blank_r, win_size
        )

        d_ideal = np.gradient(ideal_for_fir, edge_order=2) * win
        d_pulse = np.gradient(pulse_measured, edge_order=2) * win
        
        pad_ideal = np.zeros(FFT_LEN, dtype=np.float64)
        pad_pulse = np.zeros(FFT_LEN, dtype=np.float64)
        pad_start = (FFT_LEN - pulse_len) // 2

        pad_ideal[pad_start : pad_start + pulse_len] = d_ideal
        pad_pulse[pad_start : pad_start + pulse_len] = d_pulse

        pulse_data.append({
            'pulse_raw': pulse_measured,
            'pad_pulse': pad_pulse,
            'pad_ideal': pad_ideal,
            'measured_sync_tip_level': measured_sync_tip_level,
            'measured_blanking_level': measured_blanking_level,
        })
        pulse_pad_sum += pad_pulse

    # Average of the padded raw pulses for outlier detection
    pulse_pad_avg = pulse_pad_sum / max(1, len(pulse_data))

    distances = []
    for d in pulse_data:
        d['distance'] = np.mean(np.abs(d['pad_pulse'] - pulse_pad_avg))
        distances.append(d['distance'])

    median_dist = np.median(distances) if distances else 0.0
    mad_dist = np.median(np.abs(distances - median_dist)) if distances else 0.0
    dist_threshold = median_dist + 3.0 * mad_dist + 1e-6

    # =========================================================================
    # PART 2: Filter Outliers, Wiener Deconvolution Setup (Frequency Domain)
    # =========================================================================
    # These are ring buffers storing the rolling averages
    s_xy_history = state['s_xy_history']
    s_yy_history = state['s_yy_history']

    # ring buffer split indices
    first_half = min(state['count'], MAX_AVG_MEASUREMENTS - state['ptr'])
    second_half = state['count'] - first_half

    s_xy_sum = np.sum(s_xy_history[state['ptr']:state['ptr'] + first_half], axis=0) + np.sum(s_xy_history[:second_half], axis=0)
    s_yy_sum = np.sum(s_yy_history[state['ptr']:state['ptr'] + first_half], axis=0) + np.sum(s_yy_history[:second_half], axis=0)

    for i, data in enumerate(pulse_data):
        # Check individual pulse against the field average
        dist = np.mean(np.abs(data['pad_pulse'] - pulse_pad_avg))

        if dist <= dist_threshold:
            # Take the raw FFTs of the actual and ideal step transitions
            Y = scipy.fft.fft(data['pad_pulse'])
            X = scipy.fft.fft(data['pad_ideal'])

            # Compute the Frequency-Domain Delta (Cross-Spectrum)
            s_xy = X * np.conj(Y)
            s_yy = np.abs(Y)**2

            s_xy_sum += s_xy
            s_yy_sum += s_yy

            s_xy_history[state['ptr']] = s_xy
            s_yy_history[state['ptr']] = s_yy
            state['ptr'] = (state['ptr'] + 1) % MAX_AVG_MEASUREMENTS
            state['count'] = min(MAX_AVG_MEASUREMENTS, state['count'] + 1)
        else:
            # outlier
            continue

    if state['count'] > 0:
        s_xy_mean = s_xy_sum / state['count']
        s_yy_mean = s_yy_sum / state['count']
    else:
        s_xy_mean = s_xy_sum
        s_yy_mean = s_yy_sum

    phase_advance = _calc_phase_advance(s_xy_mean)
    phase_advance_int = int(np.round(phase_advance))
    phase_advance_frac = phase_advance - phase_advance_int

    # hf_mask = fft_freqs > 0.35
    # hf_power = s_yy_mean[hf_mask]
    # hf_median = np.median(hf_power) if len(hf_power) > 0 else 0.0
    # hf_mad = np.median(np.abs(hf_power - hf_median)) if len(hf_power) > 0 else 0.0

    denom = s_yy_mean + np.max(s_yy_mean)
    denom[denom == 0] = 1e-12

    # The analytic Wiener transfer function (delta applied)
    H_inv = s_xy_mean / denom
    # Apply ONLY the fractional sub-sample advance to the kernel in frequency domain.
    H_inv = H_inv * np.exp(-1j * 2.0 * np.pi * fft_freqs * phase_advance_frac)

    # Create the FIR Kernel

    # Wiener deconvolution naturally places the main impulse offset by int_adv.
    # By shifting our extraction center to match this offset
    fir_center = FFT_LEN // 2 - phase_advance_int
            
    # Isolate causal tail taps for t >= 1
    h_full = scipy.fft.fftshift(scipy.fft.ifft(H_inv).real)
    fir_tail = h_full[fir_center + 1:]
    
    # Safely insert the tail up to the maximum available filter length
    insert_len = min(len(fir_tail), FIR_LEN - 1)
    fir_kernel = np.zeros(FIR_LEN, dtype=np.float32)
    fir_kernel[1:1 + insert_len] = fir_tail[:insert_len]

    # Fade out the tail using a half-cosine window
    fade_len = round(FIR_LEN * 0.30) # fade out last 15%
    fade_axis = np.arange(fade_len, dtype=np.float64)
    fir_kernel[FIR_LEN - fade_len:] *= 0.5 * (1.0 + np.cos(np.pi * fade_axis / fade_len))

    # 3. Base DC Normalization: Enforce DC Unity strictly on the center tap (t=0)
    # MUST sum the faded kernel itself, not the raw infinite tail
    fir_kernel[0] = 1.0 - np.sum(fir_kernel[1:])

    if state['count'] == 0:
        _denormalize_inplace(video_buf, sync_tip_level, blanking_level)
        return video_buf, {'gain': 0.0, 'threshold': 0.1, 'blur_radius': 0.0}


    # =========================================================================
    # PART 4: Execute Delta Injection via Sliding Interpolation
    # =========================================================================

    # Execute the entire field in a single parallelized pass
    video_buf_filtered = _apply_inverse_eq(
        video_buf,
        len(video_buf),
        fir_kernel
    )

    lti_params = derive_lti_parameters(fir_kernel) 

    if debug:
        full_win_size = max(front_porch_len + sync_len + back_porch_len, FIR_LEN)
        pulse_ffts = []
        delta_ffts = []
        corrected_ffts = []
        raw_pulses = []
        corrected_pulses = []
        line_indices = []
        sync_tip_avg = 0
        blanking_avg = 0


        for idx, d in enumerate(pulse_data):
            raw_p = d['pulse_raw']
            corr_p = _apply_inverse_eq(
                raw_p,
                len(raw_p),
                fir_kernel
            )

            sync_tip, blanking = _get_levels(corr_p)
            sync_tip_avg += sync_tip
            blanking_avg += blanking

            raw_pulses.append(raw_p)

            corrected_pulses.append(corr_p)
            line_indices.append(idx)

            pulse_fft = scipy.fft.fft(raw_p, n=FFT_LEN)
            corrected_fft = scipy.fft.fft(corr_p, n=FFT_LEN)
            delta_fft = corrected_fft - pulse_fft

            pulse_ffts.append(pulse_fft)
            corrected_ffts.append(corrected_fft)
            delta_ffts.append(delta_fft)


        ideal_debug = _build_reference_sync_pulse(
            front_porch_len,
            sync_len,
            target_transition,
            sync_tip_avg / len(pulse_data),
            blanking_avg / len(pulse_data),
            full_win_size,
            phase_delay=0,
        )
                    
        _show_group_delay_debug(
            raw_pulses,
            corrected_pulses,
            ideal_debug,
            fir_kernel,
            line_indices,
            pulse_ffts,
            delta_ffts,
            corrected_ffts,
            FFT_LEN,
        )

    return _denormalize_inplace(video_buf_filtered, sync_tip_level, blanking_level), lti_params


def _calc_phase_advance(S_xy):
    """
    Calculates the required group-delay advance (in fractional samples).
    
    scans the Weighted R^2 to safely find the maximum coherent bandwidth 
    and establish phase-aligned ringing cancellation.
    """
    phase = np.unwrap(np.angle(S_xy))
    freqs = scipy.fft.fftfreq(FFT_LEN)

    # Skip the DC ledge to prevent early low-frequency lock
    start_bin = max(2, int(FFT_LEN * 0.02))
    max_bins = max(start_bin + 5, int(FFT_LEN * 0.35))

    w_full = 2.0 * np.pi * freqs[start_bin:max_bins]
    p_full = phase[start_bin:max_bins]
    mag_full = np.abs(S_xy[start_bin:max_bins])

    # High-Frequency Weighting: mag * w^2 targets the ringing resonance
    weights_full = mag_full * (w_full ** 2)

    best_end = len(w_full)
    best_r2 = -np.inf
    min_window = max(4, int(0.05 * len(w_full)))
    
    # =========================================================================
    # R^2 Boundary Detection & Bulk Slope
    # =========================================================================
    for end in range(min_window, len(w_full)):
        w = w_full[:end]
        p = p_full[:end]
        wt = weights_full[:end]

        wt_norm = wt / np.max(wt) if np.max(wt) > 0 else wt

        slope, intercept = np.polyfit(w, p, 1, w=wt_norm)
        fit = slope * w + intercept

        p_mean = np.average(p, weights=wt_norm)
        ss_res = np.sum(wt_norm * (p - fit) ** 2)
        ss_tot = np.sum(wt_norm * (p - p_mean) ** 2)
        
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        if r2 > best_r2:
            best_r2 = r2
            best_end = end

    # Isolate the optimal coherent bandwidth
    w_opt = w_full[:best_end]
    p_opt = p_full[:best_end]
    wt_opt = weights_full[:best_end]
    
    # Calculate the rough bulk slope as our baseline
    slope, _ = np.polyfit(w_opt, p_opt, 1, w=wt_opt)
    return -slope


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
def _build_reference_sync_pulse(
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
    
    # Generate complete target pulse for reference plot
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

# Set `--debug_plot inverse_eq` to show this plot
def _show_group_delay_debug(
    raw_pulses,
    corrected_pulses,
    ideal,
    fir_kernel,
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
    
    ax1.plot(ideal, label='Unshifted Reference', color='#7f7f7f', linestyle=':', linewidth=2)
    
    ax1.set_title(f"Line Inspection [Line {line_num}", fontweight='bold')
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

    # Set up the dynamic kernel plot with the initial kernel
    x_axis = np.arange(len(fir_kernel))
    kernel_line, = ax3.plot(x_axis, fir_kernel, label='Causal Taps (t >= 0)', color='purple', marker='.', linewidth=1.2)
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
    ax5.set_title("Delta (Corrected - Raw)", fontweight='bold')
    ax5.set_xlabel("Frequency (MHz)")
    ax5.set_ylabel("Line Index")
    ax5.set_adjustable('box')

    ax6.imshow(corrected_spec_data, aspect='auto', origin='lower', extent=[pos_freqs[0], pos_freqs[-1], min_line, max_line], cmap=green_pos)
    ax6.set_title("Corrected Y_corr", fontweight='bold')
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
        
        raw_line.set_data(np.arange(len(raw_pulses[idx])), raw_pulses[idx])
        raw_line.set_label(f'Raw Line {l_num}')
        
        corr_line.set_data(np.arange(len(corrected_pulses[idx])), corrected_pulses[idx])
        corr_line.set_label(f'Equalized Line {l_num}')
        
        ax1.set_title(f"Line Inspection [Line {l_num}]", fontweight='bold')
        ax1.legend(loc='lower right')
        
        ax1.relim()
        ax1.autoscale_view()
        
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05, hspace=0.25)
    plt.show()


# -----------------------------------------------------------------------------
# LTI PARAMETER DERIVATION FUNCTION
# -----------------------------------------------------------------------------
def derive_lti_parameters(fir_kernel, noise_threshold=0.2):
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
    n = len(video_buf)
    prev = video_buf[0]

    for i in range(1, n - 1):
        curr = video_buf[i]
        nxt = video_buf[i + 1]

        # 1st derivative (span of 2 pixels) to check threshold
        abs_diff = abs(nxt - prev)

        if abs_diff > threshold:
            # 2nd derivative (Laplacian) isolates the high-frequency edge energy symmetrically
            laplacian = prev - 2.0 * curr + nxt

            # Non-linear gain taper
            edge_weight = min(1.0, abs_diff / (2.0 * threshold))
            
            # Subtracting the Laplacian acts as a symmetric unsharp mask / peaking filter
            video_buf[i] = curr - (gain * edge_weight * laplacian)

        prev = curr