import numpy as np
import numba as nb
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
# 1. PHYSICAL DE-EMPHASIS TRANSIENT MODEL (LEAST SQUARES)
# -----------------------------------------------------------------------------
def vhs_deemphasis_transient_model(
    t,
    gain,
    curvature,
    A_rc,
    alpha_rc,
    A_dl,
    alpha_dl,
    time_offset,
):
    """
    VHS de-emphasis ringing residual model.

    The full physical model is:
        nonlinear edge response
      + RC settling
      + delayed limiter ringing

    For correction purposes we only return the residual error
    relative to the monotonic edge response. This prevents the
    correction from reshaping the actual transition.

    Returns:
        ringing residual only
    """

    tp = np.maximum(0.0, t + time_offset)

    # JVC nonlinear edge response (desired signal component)
    nonlinear = gain * np.tanh(curvature * tp)

    # First-order RC settling (desired edge response component)
    pre_emphasis = A_rc * np.exp(-alpha_rc * tp)

    # Double limiter LPF phase/ringing component (undesired)
    dl_lpf_effect = -A_dl * tp * np.exp(-alpha_dl * tp)

    # Full modeled transient
    full_transient = (
        nonlinear
        + pre_emphasis
        + dl_lpf_effect
    )

    # Remove monotonic edge response.
    # What remains is the ringing/distortion component only.
    baseline = nonlinear + pre_emphasis

    ringing_error = full_transient - baseline

    # Force zero before transition
    ringing_error[(t + time_offset) <= 0] = 0.0

    return ringing_error


@nb.njit(cache=True, nogil=True, fastmath=True)
def _apply_vhs_deemphasis_model(
    tbc_video,
    gain_scale,
    gain,
    curvature,
    A_rc,
    alpha_rc,
    A_dl,
    alpha_dl,
    time_offset,
    edge_delay,
):
    n_samples = len(tbc_video)
    out = np.empty_like(tbc_video)

    last_edge = -100000.0
    edge_type = 0

    mid_level = -20.0

    prev = tbc_video[0]
    out[0] = prev


    for i in range(1, n_samples):

        v = tbc_video[i]

        # ---------------------------------------------------------
        # Same edge detector as debug pulse extraction
        # ---------------------------------------------------------

        if prev >= mid_level and v < mid_level:

            frac = (prev - mid_level) / max(prev - v, 1e-12)

            last_edge = (i - 1) + frac
            edge_type = -1

        elif prev <= mid_level and v > mid_level:

            frac = (mid_level - prev) / max(v - prev, 1e-12)

            last_edge = (i - 1) + frac
            edge_type = 1

        prev = v
        correction = 0.0

        if last_edge > -99999.0:

            t = i - last_edge - edge_delay
            tp = t + time_offset

            if tp > 0.0:
                nonlinear = (
                    gain *
                    np.tanh(curvature * tp)
                )

                rc = (
                    A_rc *
                    np.exp(-alpha_rc * tp)
                )

                dl = (
                    -A_dl *
                    tp *
                    np.exp(-alpha_dl * tp)
                )

                ringing = nonlinear + rc + dl
                baseline = nonlinear + rc
                residual = ringing - baseline


                if edge_type < 0:
                    correction = residual

                else:
                    correction = -residual



        out[i] = v - gain_scale * correction


    return out


# -----------------------------------------------------------------------------
# 3. DEBUG PLOT RENDERER
# -----------------------------------------------------------------------------
def _render_debug_plot(
    avg_pulse_shape,
    corrected_pulse,
    ideal,
    E_frac_fall,
    E_frac_rise,
    correction_applied,
    blanking_level,
    sync_tip_level,
    target_transition
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required to render the debug plot.")
        return

    win_size = len(avg_pulse_shape)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Top Pane: Before vs. After vs. Fitted Target
    ax1.plot(avg_pulse_shape, label='Before (Raw Avg Pulse)', color='#d62728', linewidth=1.8)
    ax1.plot(corrected_pulse, label='After (Fitted VHS De-emphasis Correction)', color='#1f77b4', linewidth=2.2)
    ax1.plot(ideal, label=f'Target Spec Step (Trans: {target_transition})', color='#7f7f7f', linestyle=':', linewidth=2)
    ax1.axhline(blanking_level, color='black', alpha=0.3, label='Global Blanking Ref')
    ax1.axhline(sync_tip_level, color='black', alpha=0.3, label='Global Sync Ref')
    ax1.set_title("VHS Pulse Structure Comparison: Before vs. After")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # Middle Pane: Fitted Error Fractional Profiles
    ax2.plot(E_frac_fall, label='Averaged Fit Fall ($E_{avg}$)', color='purple', linewidth=1.5)
    ax2.plot(E_frac_rise, label='Averaged Fit Rise ($E_{avg}$)', color='red', linewidth=1.5)
    ax2.set_title("Least-Squares De-Emphasis Models (DC Offset Removed)")
    ax2.axhline(0, color='black', alpha=0.3)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    # Bottom Pane: Absolute Correction Amount
    ax3.plot(correction_applied, label='Correction Delta', color='green', linewidth=1.5)
    ax3.fill_between(range(win_size), 0, correction_applied, color='green', alpha=0.2)
    ax3.axhline(0, color='black', alpha=0.3)
    ax3.set_title("Absolute Correction Amount Applied Across Interval")
    ax3.legend()
    ax3.grid(True, alpha=0.4)

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
# 4. MAIN CORRECTION FUNCTION
# -----------------------------------------------------------------------------
def apply_tbc_vhs_deemphasis_correction(
    video_buf, 
    line_start, 
    line_end, 
    line_length, 
    blanking_level, 
    sync_tip_level, 
    front_porch_len=15, 
    sync_len=67,         
    back_porch_len=67,   
    gain_scale=0.15,
    target_transition=3.3,
    debug=False
):
    video_buf = ((video_buf - sync_tip_level) / (blanking_level - sync_tip_level)) * 40 - 40
    
    n_samples = len(video_buf)
    win_size = front_porch_len + sync_len + back_porch_len
    offset_fall = front_porch_len
    offset_rise = front_porch_len + sync_len

    # --- Step 1: Extract Average Pulse Shape ---
    avg_pulse_shape = np.zeros(win_size, dtype=np.float64)
    count = 0
    
    for line in range(line_start, line_end + 1):
        loc = line * line_length
        # Safely extract window around sample 0 of the current line
        if loc - offset_fall >= 0 and loc + sync_len + back_porch_len < n_samples:
            avg_pulse_shape += video_buf[loc - offset_fall : loc + sync_len + back_porch_len]
            count += 1

    if count > 0:
        avg_pulse_shape /= float(count)
    else:
        if debug:
            print(f"DEBUG: 0 lines matched bounds! Check line_start ({line_start}), line_end ({line_end}), line_length ({line_length}), total samples ({n_samples}).")
        return video_buf.copy()

    # --- Step 2: Subpixel Phase Alignment ---
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

    # --- Step 3: Local Baseline Extraction (Shared Porch DC Offset) ---
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

    # Enforce identical DC level across front and back porches
    shared_blank = (local_blank_f + local_blank_r) / 2.0
    local_blank_f = shared_blank
    local_blank_r = shared_blank

    ideal = build_ideal_step(
        measured_center_fall,
        measured_center_rise,
        target_transition,
        local_blank_f,
        local_sync,
        local_blank_r,
        win_size,
    )
    mid_sync = int(measured_center_fall + (measured_center_rise - measured_center_fall) / 2.0)

    # --- Step 5: Least-Squares Model Fitting of De-Emphasis Response ---

    x = np.arange(win_size, dtype=np.float64)

    def residual_model_combined(
        x,
        gain,
        curvature,
        A_rc,
        alpha_rc,
        A_dl,
        alpha_dl,
        time_offset,
        transition,
    ):
        out = np.zeros_like(x)

        fall_mask = x < win_size
        rise_mask = x >= win_size

        if np.any(fall_mask):
            xf = x[fall_mask]

            out[fall_mask] = vhs_deemphasis_transient_model(
                xf - measured_center_fall,
                gain,
                curvature,
                A_rc,
                alpha_rc,
                A_dl,
                alpha_dl,
                time_offset,
            )

        if np.any(rise_mask):
            xr = x[rise_mask] - win_size

            out[rise_mask] = vhs_deemphasis_transient_model(
                xr - measured_center_rise,
                gain,
                curvature,
                A_rc,
                alpha_rc,
                A_dl,
                alpha_dl,
                time_offset,
            )

        return out

    # Regions to fit
    t_vec_fall = x - measured_center_fall
    mask_fall = (t_vec_fall >= -4.0) & (t_vec_fall <= 24.0)

    t_vec_rise = x - measured_center_rise
    mask_rise = (t_vec_rise >= -4.0) & (t_vec_rise <= 24.0)
    
    amp_est = abs(local_sync - local_blank_f)

    # Initial guesses utilizing the Double Limiter LPF injection
    p0 = [
        0.0,               # gain
        0.35,              # curvature
        amp_est * 0.4,     # A_rc (RC decay amplitude)
        0.8,               # alpha_rc (RC decay rate)
        amp_est * 0.2,     # A_dl (Double Limiter LPF amplitude)
        0.3,               # alpha_dl (Double Limiter LPF decay rate)
        0.0,               # time_offset
        target_transition,
    ]

    bounds = (
        [
            0.0,    # gain
            0.01,   # curvature
            0.0,    # A_rc
            0.1,    # alpha_rc
            0.0,    # A_dl
            0.05,   # alpha_dl
            -12.0,  # time_offset
            1.0,    # transition
        ],
        [
            50.0,   # gain
            5.0,    # curvature
            50.0,   # A_rc
            5.0,    # alpha_rc
            50.0,   # A_dl
            0.5,    # alpha_dl
            12.0,    # time_offset
            25.0,   # transition
        ],
    )

    fit_x = np.concatenate((x[mask_fall], x[mask_rise] + win_size))

    residual = avg_pulse_shape - ideal
    fit_y = np.concatenate((
        residual[mask_fall],
        -residual[mask_rise],
    ))

    try:
        popt, _ = curve_fit(
            residual_model_combined,
            fit_x,
            fit_y,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        gain, curvature, A_rc, alpha_rc, A_dl, alpha_dl, time_offset, transition = popt
        edge_delay = -transition

    except Exception as e:
        print("combined fit failed:", e)


    if debug:
        corrected_pulse = avg_pulse_shape.copy()

        fall = vhs_deemphasis_transient_model(
            t_vec_fall,
            gain,
            curvature,
            A_rc,
            alpha_rc,
            A_dl,
            alpha_dl,
            time_offset,
        )

        rise = vhs_deemphasis_transient_model(
            t_vec_rise,
            gain,
            curvature,
            A_rc,
            alpha_rc,
            A_dl,
            alpha_dl,
            time_offset,
        )

        for i in range(win_size):
            if i <= mid_sync:
                corrected_pulse[i] -= fall[i]
            else:
                corrected_pulse[i] += rise[i]

        correction_applied = corrected_pulse - avg_pulse_shape

        _render_debug_plot(
            avg_pulse_shape,
            corrected_pulse,
            ideal,
            fall,
            rise,
            correction_applied,
            0,
            -40,
            target_transition
        )

    video_buf = _apply_vhs_deemphasis_model(
        video_buf,
        gain_scale,
        gain,
        curvature,
        A_rc,
        alpha_rc,
        A_dl,
        alpha_dl,
        time_offset,
        edge_delay
    )

    # denomalized data # TODO, run this after the data is already denomalized
    return ((video_buf + 40) / 40) * (blanking_level - sync_tip_level) + sync_tip_level
