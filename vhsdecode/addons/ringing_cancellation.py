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

def _make_inverse_eq(fir_len):
    @nb.njit(
        "float32[:](float32[::1], int64, float32[::1])",
        cache=True,
        nogil=True,
        fastmath=True
    )
    def _apply_inverse_eq(picture, n_samples, causal_kernel):
        causal_kernel = causal_kernel[::-1].copy()
        out = np.empty(n_samples, np.float32)

        first = picture[0]

        for i in nb.prange(n_samples):
            acc = 0.0

            if i < fir_len - 1:
                start = fir_len - 1 - i

                for j in range(start, fir_len):
                    acc += picture[i - fir_len + 1 + j] * causal_kernel[j]

                for j in range(start):
                    acc += first * causal_kernel[j]

            else:
                base = i - fir_len + 1

                for j in range(fir_len):
                    acc += picture[base + j] * causal_kernel[j]

            out[i] = acc

        return out
    return _apply_inverse_eq

_apply_inverse_eq = _make_inverse_eq(FIR_LEN)


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

    # =========================================================================
    # PART 1: Calculate Pulse Shape & Spectra (Rising Edge Only)
    # =========================================================================
    pre_rise_samples = int(np.round(0.25 * sync_len))
    win_size = pre_rise_samples + back_porch_len
    win_size -= win_size % 4 
    offset_rise = pre_rise_samples

    win = scipy.signal.windows.tukey(win_size, alpha=0.05)
    start_idx = (FFT_LEN - win_size) // 2
    count = 0

    S_xy = np.zeros(FFT_LEN, dtype=np.complex128)
    S_xx = np.zeros(FFT_LEN, dtype=np.complex128)
    S_yy = np.zeros(FFT_LEN, dtype=np.float64)

    rolling_S_xy = np.zeros(FFT_LEN, dtype=np.complex128)
    rolling_S_xx = np.zeros(FFT_LEN, dtype=np.complex128)
    rolling_S_yy = np.zeros(FFT_LEN, dtype=np.float64)

    # get previous iteration's rolling averages
    for measurement in group_delay_state:
        rolling_S_xy += measurement['s_xy']
        rolling_S_xx += measurement['s_xx']
        rolling_S_yy += measurement['s_yy']

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

            # ONLY accumulate here. No Wiener math inside this loop.
            S_xy += X * np.conj(Y)
            S_xx += np.abs(X)**2
            S_yy += np.abs(Y)**2

    # Save current chunk to state (Added missing 's_yy')
    if count > 0:
        current_measurement = {
            's_xy': S_xy,
            's_xx': S_xx,
            's_yy': S_yy,  
        }
        group_delay_state.append(current_measurement)
    else:
        _denormalize_inplace(video_buf, sync_tip_level, blanking_level)
        return video_buf, {'gain': 0.0, 'threshold': 0.1, 'blur_radius': 0.0}

    # COMBINE historical rolling state with the current chunk's measurements
    total_S_xy = rolling_S_xy + S_xy
    total_S_yy = rolling_S_yy + S_yy

    # =========================================================================
    # PART 2: Iterative Feed-Forward Wiener Deconvolution
    # =========================================================================
    num_passes = 5  # Optimize by dialing this between 2 and 5
    
    # Initialize the integrated frequency-domain filter
    H_total = np.ones(FFT_LEN, dtype=np.complex128)
    
    freqs = np.abs(np.fft.fftfreq(FFT_LEN))
    hf_mask = freqs > 0.35  

    for pass_idx in range(num_passes):
        # 1. Update spectra based on the current integrated correction
        # Use total_S_xy and total_S_yy so the current frame is included!
        current_S_xy = total_S_xy * np.conj(H_total)
        current_S_yy = total_S_yy * (np.abs(H_total) ** 2)

        # 2. Extract noise floor dynamically using standard NumPy MAD
        hf_power = current_S_yy[hf_mask]
        hf_median = np.median(hf_power)
        hf_mad = np.median(np.abs(hf_power - hf_median))
        
        noise_floor = hf_median + 3.0 * hf_mad
        nsr_penalty = noise_floor * noise_threshold

        # 3. Calculate the residual Wiener correction step
        denom = current_S_yy + nsr_penalty
        denom[denom == 0] = 1e-12
        
        H_step = current_S_xy / denom

        # 4. Integrate the residual step into the total filter
        H_total *= H_step

    # Pass the final integrated filter into Part 3 for IFFT and causality windowing
    H_inv = H_total

    ######################################################
    # TODO: Might be able to detect head switching pulses.
    #       The ringing pattern clearly differs when the head switch occurs.
    #       Detect the switching position, measure it, and correct it's slope
    ######################################################

    # =========================================================================
    # PART 3: Build Causal Inverse Equalization FIR Filter
    # =========================================================================
    
    # 1. Anti-Aliasing Frequency Taper (Roll-off near Nyquist)
    # Smoothly attenuates H_inv between 0.35 and 0.45 normalized frequency
    freqs_norm = np.abs(np.fft.fftfreq(FFT_LEN))
    taper = np.clip((0.45 - freqs_norm) / 0.10, 0.0, 1.0)
    # Cosine smoothing for the transition band
    taper_smooth = 0.5 * (1.0 - np.cos(np.pi * taper)) 
    H_inv *= taper_smooth

    # 2. IFFT back to time domain
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
    
    # 3. Anti-Aliasing Time Window (Prevent Gibbs Truncation)
    # Create a right-sided Tukey window: flat at the center tap, fading to 0 at the tail
    # alpha=0.5 means the final 50% of the kernel gently tapers down
    tail_window = scipy.signal.windows.tukey(FIR_LEN * 2, alpha=0.5)[FIR_LEN:]
    fir_kernel *= tail_window

    # Normalize to preserve DC gain
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
        pulse_ffts = []
        delta_ffts = []
        corrected_ffts = []
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
    ax5.imshow(
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
    ax6.imshow(
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