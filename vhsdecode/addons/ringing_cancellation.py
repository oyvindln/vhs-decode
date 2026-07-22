import numpy as np
import numba as nb

"""
The transient response needs to model the two transitions
Other noise should be ignored, I only need to model the transient's slope and then fit a response curve to it
The ringing happens only in the forward direction and it appears to be following an exponential decay, vs. a following a raised cosine slope
As the pulse progresses through time, it appears to shift very slightly up just before the transition upward,
then progresses upwards, then overshoots a bit before setting back down exponentially

Maybe treat this like an attack / release, where the attack is the beginning slow increase, and the release is the decay at the top

run a few passes to extrapolate the multistage deemphasis passes
There is a non-linear deemphasis, and a sub deemphasis stage, I need to extrapolate the amount of correction to apply to each stage

Get the different models for emphasis, fit the hsync pulses to those models, apply deemphasis through luma signal

Non-linear filters

Perform least squares optimization on the filters for the averaged sync pulse
Fit the filter parameters to the best parameters that correct the pulse

"""

@nb.njit(cache=True, nogil=True, fastmath=True)
def apply_tbc_ringing_correction(
    tbc_video, 
    line_start, 
    line_end, 
    line_length, 
    blanking_level, 
    sync_tip_level, 
    front_porch_len=15, 
    sync_len=67,         
    back_porch_len=67,   
    gain_scale=1.0,
    target_transition=3.3 
):
    n_samples = len(tbc_video)
    win_size = front_porch_len + sync_len + back_porch_len
    offset_fall = front_porch_len
    offset_rise = front_porch_len + sync_len

    # 1. Extract Full Average Pulse Shape
    avg_pulse_shape = np.zeros(win_size, dtype=np.float64)
    count = 0
    
    for line in range(line_start, line_end + 1):
        loc = line * line_length
        if loc - offset_fall >= 0 and loc + sync_len + back_porch_len < n_samples:
            for k in range(win_size):
                avg_pulse_shape[k] += tbc_video[loc - offset_fall + k]
            count += 1

    if count > 0:
        avg_pulse_shape /= float(count)
    else:
        return tbc_video.copy()

    # 2. Dynamic Subpixel Phase Alignment
    mid_val = (blanking_level + sync_tip_level) / 2.0
    
    measured_center_fall = float(offset_fall)
    for i in range(win_size - 1):
        if avg_pulse_shape[i] >= mid_val >= avg_pulse_shape[i+1]:
            y0 = avg_pulse_shape[i] - mid_val
            y1 = avg_pulse_shape[i+1] - mid_val
            if (y0 - y1) != 0.0:
                measured_center_fall = float(i) + abs(y0) / abs(y0 - y1)
            break

    measured_center_rise = float(offset_rise)
    search_start = int(measured_center_fall + (sync_len // 2))
    for i in range(search_start, win_size - 1):
        if avg_pulse_shape[i] <= mid_val <= avg_pulse_shape[i+1]:
            y0 = avg_pulse_shape[i] - mid_val
            y1 = avg_pulse_shape[i+1] - mid_val
            if (y1 - y0) != 0.0:
                measured_center_rise = float(i) + abs(y0) / abs(y1 - y0)
            break

    sum_fp, count_fp = 0.0, 0
    for i in range(0, max(1, int(measured_center_fall - 4))):
        sum_fp += avg_pulse_shape[i]
        count_fp += 1
    local_blank_f = sum_fp / count_fp if count_fp > 0 else blanking_level
    
    sum_sync, count_sync = 0.0, 0
    for i in range(int(measured_center_fall + 12), int(measured_center_rise - 12)):
        sum_sync += avg_pulse_shape[i]
        count_sync += 1
    local_sync = sum_sync / count_sync if count_sync > 0 else sync_tip_level
    
    sum_bp, count_bp = 0.0, 0
    for i in range(int(measured_center_rise + 4), win_size):
        sum_bp += avg_pulse_shape[i]
        count_bp += 1
    local_blank_r = sum_bp / count_bp if count_bp > 0 else blanking_level

    shared_blank = (local_blank_f + local_blank_r) / 2.0
    local_blank_f = shared_blank
    local_blank_r = shared_blank

    # 3. Build Phase & DC-Aligned Target
    ideal = np.zeros(win_size, dtype=np.float64)
    mid_sync = int(measured_center_fall + (measured_center_rise - measured_center_fall) / 2.0)
    
    for i in range(win_size):
        t_fall = float(i) - measured_center_fall
        t_rise = float(i) - measured_center_rise

        if i <= mid_sync: # Falling half
            if t_fall < -target_transition / 2.0:
                ideal[i] = local_blank_f
            elif t_fall <= target_transition / 2.0:
                phase = (t_fall + target_transition / 2.0) / target_transition * np.pi
                ideal[i] = local_blank_f + (local_sync - local_blank_f) * (1.0 - np.cos(phase)) / 2.0
            else:
                ideal[i] = local_sync
        else:             # Rising half
            if t_rise < -target_transition / 2.0:
                ideal[i] = local_sync
            elif t_rise <= target_transition / 2.0:
                phase = (t_rise + target_transition / 2.0) / target_transition * np.pi
                ideal[i] = local_sync + (local_blank_r - local_sync) * (1.0 - np.cos(phase)) / 2.0
            else:
                ideal[i] = local_blank_r

    # 4. Split and Isolate Asymmetric Deficit Profiles
    error = avg_pulse_shape - ideal
    
    error_fall = np.zeros(win_size, dtype=np.float64)
    error_rise = np.zeros(win_size, dtype=np.float64)
    
    tail_len = 18.0
    fade_len = 6.0
    
    for i in range(win_size):
        dist_f = float(i) - measured_center_fall
        if dist_f < -tail_len - fade_len: w_f = 0.0
        elif dist_f < -tail_len: w_f = (dist_f + tail_len + fade_len) / fade_len
        elif dist_f < tail_len: w_f = 1.0
        elif dist_f < tail_len + fade_len: w_f = 1.0 - (dist_f - tail_len) / fade_len
        else: w_f = 0.0
        
        error_fall[i] = error[i] * w_f
        
        dist_r = float(i) - measured_center_rise
        if dist_r < -tail_len - fade_len: w_r = 0.0
        elif dist_r < -tail_len: w_r = (dist_r + tail_len + fade_len) / fade_len
        elif dist_r < tail_len: w_r = 1.0
        elif dist_r < tail_len + fade_len: w_r = 1.0 - (dist_r - tail_len) / fade_len
        else: w_r = 0.0
        
        error_rise[i] = error[i] * w_r

    amp_fall = local_sync - local_blank_f
    amp_rise = local_blank_r - local_sync
    E_frac_fall = error_fall / (amp_fall if amp_fall != 0 else 1.0)
    E_frac_rise = error_rise / (amp_rise if amp_rise != 0 else 1.0)

    # Remove DC bias from the error fraction so correction doesn't shift baseline
    E_frac_fall -= np.mean(E_frac_fall)
    E_frac_rise -= np.mean(E_frac_rise)

    # 5. Macro-State Tracker
    smooth_len = int(max(4.0, target_transition * 2.5))
    half_w = smooth_len // 2
    sig_sm = np.empty(n_samples, dtype=np.float64)
    
    run_sum = 0.0
    for i in range(smooth_len):
        idx = max(0, min(n_samples - 1, i - half_w))
        run_sum += tbc_video[idx]
        
    for i in range(n_samples):
        sig_sm[i] = run_sum / smooth_len
        sub_idx = max(0, i - half_w)
        add_idx = min(n_samples - 1, i + half_w + 1)
        run_sum += tbc_video[add_idx] - tbc_video[sub_idx]

    state = np.zeros(n_samples, dtype=np.float64)
    curr = 0.5
    thresh = abs(amp_fall) * 0.03
    for i in range(1, n_samples):
        dx = sig_sm[i] - sig_sm[i-1]
        if dx > thresh: curr = 1.0
        elif dx < -thresh: curr = 0.0
        state[i] = curr

    state_sm = np.empty(n_samples, dtype=np.float64)
    run_sum = 0.0
    for i in range(smooth_len):
        idx = max(0, min(n_samples - 1, i - half_w))
        run_sum += state[idx]
        
    for i in range(n_samples):
        state_sm[i] = run_sum / smooth_len
        sub_idx = max(0, i - half_w)
        add_idx = min(n_samples - 1, i + half_w + 1)
        run_sum += state_sm[add_idx] - state_sm[sub_idx] if add_idx < n_samples else 0.0

    # 6. Dual-Profile Linear Convolution
    out = np.empty_like(tbc_video)
    
    for i in range(n_samples):
        corr_fall = 0.0
        corr_rise = 0.0
        
        for k in range(1, win_size):
            tap_f = i + offset_fall - k
            if 1 <= tap_f < n_samples:
                dx = tbc_video[tap_f] - tbc_video[tap_f - 1]
                corr_fall += dx * E_frac_fall[k]
                
        for k in range(1, win_size):
            tap_r = i + offset_rise - k
            if 1 <= tap_r < n_samples:
                dx = tbc_video[tap_r] - tbc_video[tap_r - 1]
                corr_rise += dx * E_frac_rise[k]
                
        w = state_sm[i]
        correction = corr_rise * w + corr_fall * (1.0 - w)
        out[i] = tbc_video[i] - (correction * gain_scale)
        
    return out



def plot_tbc_correction_debug(
    tbc_video,
    line_start,
    line_end,
    line_length,
    blanking_level,
    sync_tip_level,
    front_porch_len=15, 
    sync_len=67,
    back_porch_len=67,
    gain_scale=1.0,
    target_transition=3.3
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required to render the debug plot.")
        return

    n_samples = len(tbc_video)
    win_size = front_porch_len + sync_len + back_porch_len
    offset_fall = front_porch_len
    offset_rise = front_porch_len + sync_len

    # 1. Extract Average Pulse Shape Internally
    avg_pulse_shape = np.zeros(win_size, dtype=np.float64)
    count = 0
    for line in range(line_start, line_end + 1):
        loc = line * line_length
        if loc - offset_fall >= 0 and loc + sync_len + back_porch_len < n_samples:
            avg_pulse_shape += tbc_video[loc - offset_fall : loc + sync_len + back_porch_len]
            count += 1

    if count > 0: avg_pulse_shape /= float(count)
    else: return

    # 2. Dynamic Subpixel Phase Alignment
    mid_val = (blanking_level + sync_tip_level) / 2.0
    
    measured_center_fall = float(offset_fall)
    for i in range(win_size - 1):
        if avg_pulse_shape[i] >= mid_val >= avg_pulse_shape[i+1]:
            y0 = avg_pulse_shape[i] - mid_val
            y1 = avg_pulse_shape[i+1] - mid_val
            if (y0 - y1) != 0.0:
                measured_center_fall = float(i) + abs(y0) / abs(y0 - y1)
            break

    measured_center_rise = float(offset_rise)
    search_start = int(measured_center_fall + (sync_len // 2))
    for i in range(search_start, win_size - 1):
        if avg_pulse_shape[i] <= mid_val <= avg_pulse_shape[i+1]:
            y0 = avg_pulse_shape[i] - mid_val
            y1 = avg_pulse_shape[i+1] - mid_val
            if (y1 - y0) != 0.0:
                measured_center_rise = float(i) + abs(y0) / abs(y1 - y0)
            break

    # 3. Measure Local Baseline Levels
    sum_fp, count_fp = 0.0, 0
    for i in range(0, max(1, int(measured_center_fall - 4))):
        sum_fp += avg_pulse_shape[i]
        count_fp += 1
    local_blank_f = sum_fp / count_fp if count_fp > 0 else blanking_level
    
    sum_sync, count_sync = 0.0, 0
    for i in range(int(measured_center_fall + 12), int(measured_center_rise - 12)):
        sum_sync += avg_pulse_shape[i]
        count_sync += 1
    local_sync = sum_sync / count_sync if count_sync > 0 else sync_tip_level
    
    sum_bp, count_bp = 0.0, 0
    for i in range(int(measured_center_rise + 4), win_size):
        sum_bp += avg_pulse_shape[i]
        count_bp += 1
    local_blank_r = sum_bp / count_bp if count_bp > 0 else blanking_level

    shared_blank = (local_blank_f + local_blank_r) / 2.0
    local_blank_f = shared_blank
    local_blank_r = shared_blank

    # 4. Synthesize Phase & DC-Aligned Target
    ideal = np.zeros(win_size, dtype=np.float64)
    mid_sync = int(measured_center_fall + (measured_center_rise - measured_center_fall) / 2.0)

    for i in range(win_size):
        t_fall = float(i) - measured_center_fall
        t_rise = float(i) - measured_center_rise
        
        if i <= mid_sync:
            if t_fall < -target_transition / 2.0: 
                ideal[i] = local_blank_f
            elif t_fall <= target_transition / 2.0:
                ideal[i] = local_blank_f + (local_sync - local_blank_f) * (1.0 - np.cos((t_fall + target_transition / 2.0) / target_transition * np.pi)) / 2.0
            else: 
                ideal[i] = local_sync
        else:
            if t_rise < -target_transition / 2.0: 
                ideal[i] = local_sync
            elif t_rise <= target_transition / 2.0:
                ideal[i] = local_sync + (local_blank_r - local_sync) * (1.0 - np.cos((t_rise + target_transition / 2.0) / target_transition * np.pi)) / 2.0
            else: 
                ideal[i] = local_blank_r

    # 5. Split Asymmetric Profiles
    error = avg_pulse_shape - ideal
    error_fall = np.zeros(win_size, dtype=np.float64)
    error_rise = np.zeros(win_size, dtype=np.float64)
    
    tail_len = 18.0
    fade_len = 6.0
    
    for i in range(win_size):
        dist_f = float(i) - measured_center_fall
        if dist_f < -tail_len - fade_len: w_f = 0.0
        elif dist_f < -tail_len: w_f = (dist_f + tail_len + fade_len) / fade_len
        elif dist_f < tail_len: w_f = 1.0
        elif dist_f < tail_len + fade_len: w_f = 1.0 - (dist_f - tail_len) / fade_len
        else: w_f = 0.0
        error_fall[i] = error[i] * w_f
        
        dist_r = float(i) - measured_center_rise
        if dist_r < -tail_len - fade_len: w_r = 0.0
        elif dist_r < -tail_len: w_r = (dist_r + tail_len + fade_len) / fade_len
        elif dist_r < tail_len: w_r = 1.0
        elif dist_r < tail_len + fade_len: w_r = 1.0 - (dist_r - tail_len) / fade_len
        else: w_r = 0.0
        error_rise[i] = error[i] * w_r

    amp_fall = local_sync - local_blank_f
    amp_rise = local_blank_r - local_sync
    E_frac_fall = error_fall / (amp_fall if amp_fall != 0 else 1.0)
    E_frac_rise = error_rise / (amp_rise if amp_rise != 0 else 1.0)

    # Remove DC bias from the error fraction so correction doesn't shift baseline
    E_frac_fall -= np.mean(E_frac_fall)
    E_frac_rise -= np.mean(E_frac_rise)

    # 6. Simulate State-Blended Correction
    corrected = np.empty_like(avg_pulse_shape)
    pad_shape = np.pad(avg_pulse_shape, (win_size, win_size), mode='edge')

    for i in range(win_size):
        corr_fall = 0.0
        corr_rise = 0.0
        padded_i = i + win_size
        
        for k in range(1, win_size):
            tap_f = padded_i + offset_fall - k
            corr_fall += (pad_shape[tap_f] - pad_shape[tap_f - 1]) * E_frac_fall[k]
            
            tap_r = padded_i + offset_rise - k
            corr_rise += (pad_shape[tap_r] - pad_shape[tap_r - 1]) * E_frac_rise[k]

        w = 1.0 if i > mid_sync else 0.0
        correction = corr_rise * w + corr_fall * (1.0 - w)
        corrected[i] = pad_shape[padded_i] - (correction * gain_scale)

    correction_applied = avg_pulse_shape - corrected

    # 7. Render 3-Pane Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax1.plot(avg_pulse_shape, label='Before (Raw Avg Pulse)', color='#d62728', linewidth=1.8)
    ax1.plot(corrected, label='After (Dual-FIR Corrected)', color='#1f77b4', linewidth=2.2)
    ax1.plot(ideal, label=f'Local DC & Phase-Aligned Target Spec', color='#7f7f7f', linestyle=':', linewidth=2)
    ax1.axhline(blanking_level, color='black', alpha=0.3, label='Global Blanking Ref')
    ax1.axhline(sync_tip_level, color='black', alpha=0.3, label='Global Sync Ref')
    ax1.set_title("4Fsc Localized Pulse Structure Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(E_frac_fall, label='Falling Deficit ($E_{fall}$)', color='purple', linewidth=1.5)
    ax2.plot(E_frac_rise, label='Rising Deficit ($E_{rise}$)', color='orange', linewidth=1.5)
    ax2.set_title("Split Filtering Models (DC Bias Eliminated)")
    ax2.axhline(0, color='black', alpha=0.3)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    ax3.plot(correction_applied, label='Correction Delta', color='green', linewidth=1.5)
    ax3.fill_between(range(win_size), 0, correction_applied, color='green', alpha=0.2)
    ax3.axhline(0, color='black', alpha=0.3)
    ax3.set_title("Absolute Correction Amount (Transients Only)")
    ax3.legend()
    ax3.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()