"""SECAM FM colour-under helpers.

SECAM tapes carry frequency-modulated chroma rather than the
quadrature-modulated chroma of the other colour-under formats, recorded
with one of two mutually incompatible methods (IEC 60774-1): "method 1"
records the subcarrier through a divide-by-4 counter, while ME-SECAM
heterodynes it down like the QAM formats do (and therefore rides their
up-conversion pipeline). This module holds the method 1 restoration
chain - x4 phase multiplication, BT.470 bell regeneration, blanking
reference synthesis and line identification - plus the under-carrier
pair measurement shared with the ME-SECAM carrier servo.
"""

import numpy as np
import scipy.signal as sps
import scipy.fft as sps_fft

import lddecode.core as ldd
import lddecode.utils as lddu
from vhsdecode.rust_utils import sosfiltfilt_rust


def measure_secam_under_carrier_offset(
    chroma,
    linesout,
    outwidth,
    window,
    samp_rate,
    pair_center,
    separation_range=(90e3, 230e3),
):
    """Measure how far a SECAM colour-under rest carrier pair sits from
    its nominal position, using the undeviated subcarrier on the late back
    porch of each line (the early porch is still sweeping from the previous
    line's carrier switch). For ME-SECAM this picks up the recording VCR's
    down-conversion crystal error, which otherwise ends up as an offset of
    both restored subcarriers. (SECAM method 1 has no conversion crystal, so
    for it this is only useful as a diagnostic and to recognise which of the
    two recording methods a tape actually used.)

    separation_range bounds the accepted distance between the two carrier
    clusters: the pair is nominally 156.25 kHz apart for ME-SECAM and
    39.0625 kHz apart for method 1 (a quarter of foR - foB).

    Returns the offset in Hz of the measured pair midpoint from pair_center,
    or None if no reliable measurement could be made. Single-field accuracy
    is on the order of +-100 Hz (ringing from the per-line carrier switch
    beats across the short porch window); averaging across fields washes
    this out. Crystal errors being chased are in the kHz range.
    """
    # Stay clear of the vertical interval and head switch area.
    SKIP_LINES = 20
    MIN_LINES_PER_CLUSTER = 8
    # Reject measurements where the two clusters land somewhere else
    # entirely.
    MIN_SEPARATION, MAX_SEPARATION = separation_range

    window_start, window_end = window
    freq_scale = samp_rate / (2 * np.pi)

    # Analytic signal over the whole field so the short per-line windows are
    # free of transform edge effects (a windowed transform of just the porch
    # would bias the frequency estimate by hundreds of Hz).
    n_fft = sps_fft.next_fast_len(len(chroma))
    analytic = sps.hilbert(chroma, N=n_fft)[: len(chroma)]

    freqs = []
    envs = []

    for linenumber in range(SKIP_LINES, linesout - SKIP_LINES):
        line_start = linenumber * outwidth
        start = line_start + window_start
        end = line_start + window_end
        if start < 0 or end > len(chroma):
            continue

        window_analytic = analytic[start:end]
        # Instantaneous frequency; median rejects FM clicks and noise spikes.
        f_inst = np.diff(np.unwrap(np.angle(window_analytic))) * freq_scale
        freqs.append(np.median(f_inst))
        envs.append(np.median(np.abs(window_analytic)))

    if not freqs:
        return None

    freqs = np.asarray(freqs)
    envs = np.asarray(envs)

    # Ignore lines where the porch carrier is too weak to measure
    # (dropouts, colour killed lines).
    valid = envs > (np.median(envs) * 0.25)
    low = freqs[valid & (freqs < pair_center)]
    high = freqs[valid & (freqs >= pair_center)]

    if len(low) < MIN_LINES_PER_CLUSTER or len(high) < MIN_LINES_PER_CLUSTER:
        return None

    low_carrier = np.median(low)
    high_carrier = np.median(high)
    separation = high_carrier - low_carrier
    if separation < MIN_SEPARATION or separation > MAX_SEPARATION:
        return None

    return ((low_carrier + high_carrier) / 2) - pair_center


# SECAM subcarrier rest frequencies and HF ("cloche"/bell) pre-emphasis
# constants from ITU-R BT.470-6 table 2 / BT.1700: the subcarrier amplitude
# follows G = M0 * |1 + j16F| / |1 + j1.26F| with F = f/f0 - f0/f.
SECAM_FOR = 4406250.0
SECAM_FOB = 4250000.0
SECAM_BELL_F0 = 4286000.0
# Colour-under rest carrier pair midpoints, used to tell the two VHS SECAM
# recording methods apart from the porch carriers.
SECAM_M1_UNDER_PAIR_CENTER = (SECAM_FOB / 4 + SECAM_FOR / 4) / 2  # 1082031.25
MESECAM_UNDER_PAIR_CENTER = 5060571.875 - (SECAM_FOR + SECAM_FOB) / 2  # 732446.875
# Nominal pair separation is 39.0625 kHz; the gates are loose because this is
# measured over active video (see _secam_method_diagnostic) where content
# deviation biases the per-line medians.
SECAM_M1_SEPARATION_RANGE = (18e3, 60e3)
MESECAM_SEPARATION_RANGE = (90e3, 230e3)  # nominal pair separation 156.25 kHz


def secam_bell_gain(freq_hz):
    """Relative SECAM subcarrier HF pre-emphasis (bell) gain at the given
    instantaneous frequency, normalized to 1.0 at f0 (BT.470-6)."""
    f = freq_hz / SECAM_BELL_F0
    bell_f = f - 1.0 / f
    return np.sqrt((1.0 + (16.0 * bell_f) ** 2) / (1.0 + (1.26 * bell_f) ** 2))


def upconvert_secam_method1(
    chroma, samp_rate, under_bpf, carrier_mult, rest_amplitude, return_envelope=False
):
    """Restore the studio SECAM chroma block from a method 1 colour-under
    signal (IEC 60774-1 6.4.1: recorded through a divide-by-4 counter) by
    multiplying the carrier phase back up.

    Unlike the heterodyne formats this scales carrier and deviation together,
    so tape timebase error self-corrects and there is no LO to servo. The
    divider outputs a constant-amplitude signal, so the BT.470 bell
    pre-emphasis is regenerated here from the restored instantaneous
    frequency to put the amplitude envelope back on spec for downstream
    SECAM decoders.

    Returns (restored, inst_freq): the restored chroma block signal and the
    smoothed restored-domain instantaneous frequency it was shaped with
    (the latter is reused for line identification). With return_envelope the
    band-passed under-carrier envelope is returned as a third element (used
    by regenerate_secam_blanking for local amplitude matching).
    """
    filtered = sosfiltfilt_rust(under_bpf, chroma)

    # Analytic signal over the whole field so short-window edge effects don't
    # bias the phase.
    n_fft = sps_fft.next_fast_len(len(filtered))
    analytic = sps.hilbert(filtered, N=n_fft)[: len(filtered)]
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    # Clamp under-carrier deviation to SECAM's legal window before x4
    # multiplication.
    # BT.470-6 Table 2 item 2.12 and BT.1700 Part C Table 4 item 10e limit the
    # carrier to 3.900-4.756 MHz: the per-component deviation maxima
    # (D′B -350/+506 kHz, D′R -506/+350 kHz) mirror across the carrier pair, so
    # both components share the corridor (fOB/4 - 87.5 kHz to fOR/4 + 87.5 kHz
    # in the under-carrier domain). The studio or broadcast limiter ideally
    # clipped the pre-corrected color difference to these bounds, so anything
    # outside the window is noise or a tape channel-truncation transient; x4
    # multiply would scale such excursions x4 and downstream de-emphasis smears
    # them into streaks at saturated transitions ("fire"). Clipping
    # instantaneous frequency and reintegrating keeps the carrier segment
    # smooth: clipped spans become constant-frequency stretches.
    f_under = np.gradient(phase) * (samp_rate / (2.0 * np.pi))
    np.clip(
        f_under,
        SECAM_FOB / 4 - 350e3 / 4,
        SECAM_FOR / 4 + 350e3 / 4,
        out=f_under,
    )
    phase = np.cumsum(f_under) * (2.0 * np.pi / samp_rate)

    # Restored instantaneous frequency for the bell shaping. Central
    # difference plus a short moving average keeps sample-level phase noise
    # from ending up as amplitude noise; the bell curve itself is smooth so
    # this doesn't blunt legitimate deviation.
    inst_freq = np.gradient(phase) * (carrier_mult * samp_rate / (2 * np.pi))
    smooth_len = 9
    inst_freq = np.convolve(
        inst_freq, np.full(smooth_len, 1.0 / smooth_len), mode="same"
    )
    # Keep the gain lookup inside the legal carrier excursion (BT.470:
    # 3.900 to 4.756 MHz) so noise and carrier switch transients don't get
    # boosted by the bell skirts.
    np.clip(inst_freq, 3.9e6, 4.756e6, out=inst_freq)
    gain = secam_bell_gain(inst_freq)

    # Scale by the normalized under-carrier envelope (capped just above
    # nominal). Where the carrier is healthy this is ~unity, so the average
    # amplitude stays on the bell curve; where it dips or disappears
    # (dropouts, FM clicks, no colour) the dip is passed through to the
    # output instead of being hard-limited away. Downstream SECAM decoders
    # key their click/dropout concealment off exactly those envelope
    # collapses, so preserving them matters more than emulating the
    # constant-amplitude divider chain of a real deck - and it doubles as
    # the squelch that keeps carrier-free noise from becoming full-scale
    # splatter.
    env_med = np.median(envelope)
    if env_med > 0:
        limited = np.minimum(envelope / env_med, 1.25)
    else:
        limited = np.zeros_like(envelope)

    restored = rest_amplitude * gain * limited * np.cos(carrier_mult * phase)
    if return_envelope:
        return restored, inst_freq, envelope
    return restored, inst_freq


SECAM_IDENT_MIN_CONFIDENCE = 0.7


def fit_secam_line_alternation(inst_freq, linesout, outwidth, first_line, porch_end_px):
    """Fit the field's D'R/D'B line alternation from the active-region median
    restored frequency of each line: D'R lines sit in the top half of the
    chroma block, D'B in the bottom.

    The sequence alternates strictly (BT.470), so fit the better of the two
    possible parities; per-line deviation medians can land on the wrong side
    on heavily saturated lines, the majority never does.

    Returns (dr_on_even, confidence) where confidence is the fraction of
    lines whose measured identity matches the fitted alternation, or None if
    there are too few lines to fit.
    """
    n_lines = linesout - first_line
    if n_lines < 32:
        return None

    active_start = porch_end_px + 30
    active_end = outwidth - 40
    freq_lines = inst_freq[first_line * outwidth : linesout * outwidth].reshape(
        n_lines, outwidth
    )
    line_medians = np.median(freq_lines[:, active_start:active_end], axis=1)
    is_dr = line_medians > (SECAM_FOR + SECAM_FOB) / 2

    line_index = np.arange(first_line, linesout)
    even_is_dr = np.count_nonzero(is_dr == (line_index % 2 == 0))
    confidence = max(even_is_dr, n_lines - even_is_dr) / n_lines
    return (even_is_dr >= (n_lines - even_is_dr)), confidence


class SecamParityFlywheel:
    """Carry the fitted D'R/D'B alternation across fields.

    Each TBC field is 312.5 line periods, so the alternation phase of
    consecutive fields walks a strict 4-field cycle:

        dr_on_even(n) = base ^ (((n + 1) >> 1) & 1)

    (verified on all method 1 fixture tapes: TFFT/FTTF sequences). A single
    bit therefore locks the parity of every field in the recording. Fields
    whose own alternation fit is confident teach `base`; fields whose content
    can't be fitted (near-neutral pictures, noisy tape) inherit the predicted
    parity instead of losing their blanking regeneration.

    The lock requires MIN_LOCK agreeing confident fields, expires after
    MAX_AGE fields without confirmation, and a confident contradiction resets
    it - a dropped field upstream shifts the cycle phase, and re-learning is
    cheaper than trusting a stale lock.
    """

    MIN_LOCK = 4
    MAX_AGE = 32

    def __init__(self):
        self._index = -1
        self._last_readloc = None
        self._base = None
        self._agree = 0
        self._last_confirm = None

    @staticmethod
    def _flip(index):
        return ((index + 1) >> 1) & 1

    def resolve(self, readloc, fit):
        """Advance to the field identified by readloc and resolve its parity.

        fit is (dr_on_even, confidence) or None. Returns (dr_on_even, source)
        with source "measured" or "flywheel", or (None, "unlocked") when
        neither the fit nor the lock can identify the field.
        """
        if readloc != self._last_readloc:
            self._last_readloc = readloc
            self._index += 1
        n = self._index
        flip = self._flip(n)

        if fit is not None and fit[1] >= SECAM_IDENT_MIN_CONFIDENCE:
            base = bool(fit[0]) ^ bool(flip)
            if base == self._base:
                self._agree += 1
            else:
                self._base = base
                self._agree = 1
            self._last_confirm = n
            return bool(fit[0]), "measured"

        if (
            self._base is not None
            and self._agree >= self.MIN_LOCK
            and self._last_confirm is not None
            and n - self._last_confirm <= self.MAX_AGE
        ):
            return bool(self._base ^ bool(flip)), "flywheel"
        return None, "unlocked"


def _measure_under_carrier(chroma, samp_rate, start, length, f_rot):
    """Narrowband frequency/phase estimate of the colour-under carrier over
    chroma[start:start+length] by correlation against a rotor at f_rot.

    Correlation projects out everything away from f_rot, so this stays usable
    on the raw (pre-band-pass) chroma channel where luma crosstalk would bias
    a broadband analytic-signal measurement. Two half-window correlations
    give the frequency offset from the rotor; the pooled correlation gives
    the phase. Returns (freq, phase_at) with phase_at(t) evaluating the
    carrier phase at absolute sample t, or None if there is no carrier.
    """
    x = chroma[start : start + length]
    if len(x) < length:
        return None
    t_abs = np.arange(start, start + length)
    xz = x * np.exp(-2j * np.pi * (f_rot / samp_rate) * t_abs)
    z1 = np.sum(xz[: length // 2])
    z2 = np.sum(xz[length // 2 :])
    zf = z1 + z2
    if np.abs(z1) == 0 or np.abs(z2) == 0 or np.abs(zf) == 0:
        return None
    dphi = np.angle(z2 * np.conj(z1))
    df = dphi / (2 * np.pi * (length / 2) / samp_rate)
    # Keep runaway estimates (no real carrier in the window) inside the
    # format's legal deviation.
    df = np.clip(df, -130e3, 130e3)
    freq = f_rot + df
    t_mid = start + (length - 1) / 2.0
    phase_mid = np.angle(zf)

    def phase_at(t):
        return (
            2 * np.pi * (f_rot / samp_rate) * t
            + phase_mid
            + 2 * np.pi * (df / samp_rate) * (t - t_mid)
        )

    return freq, phase_at


def regenerate_secam_blanking(
    chroma,
    envelope,
    samp_rate,
    linesout,
    outwidth,
    blank_start_px,
    porch_end_px,
    first_line,
    dr_on_even,
    carrier_mult,
):
    """Replace each line's horizontal blanking interval - front porch, sync
    and back porch in one continuous run - with a synthesized undeviated
    colour-under rest carrier, phase-continuous with the active video on both
    sides.

    On method 1 tapes the whole blanking interval carries the record chain's
    divide-by-4 counter settling transient (blanking edges / SECAM subcarrier
    phase reversals upset the divider), not the undeviated reference BT.470
    promises. Two things go wrong if it is left in place:

    - the zero-phase filters in this chain (the under-carrier band-pass here,
      FChromaFinal later) and the linear-phase cloche filters in downstream
      SECAM decoders smear the end-of-line transient BACKWARDS into the last
      ~2 us of active video, which demodulates as a magenta band down the
      right edge of the picture (D'R deviates negative, D'B positive, so the
      transient reads red on D'R lines and blue on D'B lines);
    - decoders calibrate their discriminator zeros and line identification
      from the back porch, and transient energy ringing into that window
      biases the zeros, which shows up as a full-field colour cast.

    This runs in the colour-under domain BEFORE the band-pass/analytic-signal
    restoration pass, so the zero-phase filtering never sees the transient.
    One continuous synthesis per blanking interval, phase-aligned to the
    outgoing active carrier at its start and the incoming one at its end,
    with no interior splices: an earlier version that spliced the back porch
    separately left an unaligned interior seam whose click rang into the
    decoders' porch measurement window.

    The synthesized frequency ramps from the measured outgoing carrier to the
    outgoing line's rest frequency across the front porch, steps to the
    incoming line's rest over the sync tip, and holds it through the back
    porch and the fade-out; the phase is the integral of that profile, so it
    is continuous throughout. The random phase difference
    between the two lines' carriers is closed by a frequency bump over the
    sync region plus a small constant offset across the hold (see the
    closure comment below). All disturbances are anchored to the line
    structure so that everything from ~90 px into the next line onwards - in
    particular the back-porch window decoders calibrate their discriminator
    zeros from (~65..5 px before active video) - stays within a few kHz of
    the incoming rest frequency, with margin for the ~25 px ring of the
    zero-phase band-passes.

    Returns a float64 copy of chroma with the blanking intervals replaced.
    """
    FADE_LEN = 8
    RAMP_LEN = 20
    MEAS_LEN = 32
    # Rest-to-rest frequency step position within the NEXT line (px from its
    # start): over the sync tip.
    STEP_PX = (8, 40)
    # Phase closure: the outgoing and incoming carriers are independent
    # oscillator segments, so the synthesis must absorb a uniformly random
    # phase difference of up to +-pi (x4 by the restoration - a step is not
    # an option anywhere near the picture or the porch). It goes into a
    # cosine-tapered flat-top frequency excursion across the sync region,
    # sized to the error but capped inside the under band-pass (BUMP_MAX_HZ
    # around either rest carrier stays within the 550..1300 kHz pass band),
    # and finished early enough that the band-pass ring stays out of the
    # porch reference window. Any spill past the cap (degenerately short
    # blanking only) becomes a constant offset across the rest-frequency
    # hold - never more than a few kHz, too small to bias the per-field
    # porch cluster medians or flip a line identity label.
    BUMP_END_PX = 88
    BUMP_MAX_HZ = 170e3
    BUMP_TAPER = 12

    f_rest = {
        True: SECAM_FOR / carrier_mult,
        False: SECAM_FOB / carrier_mult,
    }

    cleaned = np.array(chroma, dtype=np.float64, copy=True)
    fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(FADE_LEN) / FADE_LEN)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(RAMP_LEN) / RAMP_LEN)
    mid_step = 0.5 - 0.5 * np.cos(np.pi * np.arange(32) / 32)

    for linenumber in range(first_line, linesout - 1):
        line_is_dr = (linenumber % 2 == 0) == dr_on_even
        f_out_rest = f_rest[line_is_dr]
        f_in_rest = f_rest[not line_is_dr]
        start = linenumber * outwidth + blank_start_px
        end = (linenumber + 1) * outwidth + porch_end_px
        span = end - start
        if start - 2 * MEAS_LEN - FADE_LEN < 0 or end + 2 * MEAS_LEN > len(cleaned):
            continue

        out_meas = _measure_under_carrier(
            cleaned, samp_rate, start - MEAS_LEN, MEAS_LEN, f_out_rest
        )
        in_meas = _measure_under_carrier(
            cleaned, samp_rate, end, MEAS_LEN, f_in_rest
        )
        if out_meas is None or in_meas is None:
            continue
        f_out, out_phase_at = out_meas
        f_in, in_phase_at = in_meas

        # Local amplitudes from the band-passed envelope: narrowband
        # correlation under-reads a deviating FM carrier, and an amplitude
        # step at the splice would read as a click downstream. Measured a
        # little away from the splice points, where the pass-1 envelope is
        # still inflated by the band-pass smear of the adjacent transient.
        amp_out = np.median(envelope[start - 2 * MEAS_LEN : start - MEAS_LEN])
        amp_in = np.median(envelope[end + MEAS_LEN : end + 2 * MEAS_LEN])

        # Frequency profile: measured outgoing -> outgoing rest (over the
        # front porch) -> incoming rest (step over the sync tip) -> measured
        # incoming (final ramp), all raised-cosine.
        next_line_p = span - porch_end_px  # px offset of the next line start
        step0 = min(max(next_line_p + STEP_PX[0], RAMP_LEN), span - RAMP_LEN - 96)
        step1 = step0 + (STEP_PX[1] - STEP_PX[0])
        # The write extends FADE_LEN beyond `start` on the outside, so the
        # fade-in sits OVER the phase-matched measured outgoing carrier
        # (before the transient sets in) instead of over raw transient next
        # to the picture. The incoming side gets NO fade at all: the synth
        # ends phase-closed against the incoming carrier model at `end`, and
        # the raw signal (blanking edge, colour turn-on, picture) takes over
        # with its natural continuity. Fading out over the raw porch tail
        # mixes settling transient back in next to the decoders' porch
        # measurement window; fading out past `end` (over the incoming
        # picture) leaves a synthetic-to-content seam whose phase and
        # amplitude mismatch rings green/red fire down the left edge. Both
        # were measurably worse than the phase-closed hard handover.
        q = FADE_LEN  # profile offset of `start`
        span_ext = span + FADE_LEN
        f_prof = np.empty(span_ext)
        f_prof[:q] = f_out
        f_prof[q : q + RAMP_LEN] = f_out + (f_out_rest - f_out) * ramp
        f_prof[q + RAMP_LEN : q + step0] = f_out_rest
        f_prof[q + step0 : q + step1] = (
            f_out_rest + (f_in_rest - f_out_rest) * mid_step
        )
        # Rest frequency holds right through the back porch AND the fade-out:
        # the porch is the decoders' discriminator-zero reference, and the
        # undeviated carrier is also zero colour difference, so the fade-out
        # region (over the incoming line's low-amplitude colour turn-on
        # strip) decodes as neutral instead of as a per-line click - ramping
        # toward the measured content frequency there put green/red fire down
        # the left edge of the picture.
        f_prof[q + step1 :] = f_in_rest

        phase = out_phase_at(start - q) + (
            2 * np.pi * np.concatenate(([0.0], np.cumsum(f_prof[:-1]))) / samp_rate
        )
        phase_at_end = phase[-1] + 2 * np.pi * f_prof[-1] / samp_rate
        err = np.angle(np.exp(1j * (in_phase_at(end) - phase_at_end)))

        # Flat-top frequency excursion over the sync region: area = absorbed
        # phase. Starts no earlier than just before the next line (its
        # band-pass ring must stay out of the outgoing picture) and ends
        # early enough that the ring stays out of the porch reference window.
        b0 = max(RAMP_LEN + 4, next_line_p - 16)
        b1 = min(next_line_p + BUMP_END_PX, span - RAMP_LEN - 4)
        hold_len = (span - RAMP_LEN) - b1
        if b1 - b0 < 2 * BUMP_TAPER + 8 or hold_len < 24:
            continue
        bump_area = b1 - b0 - BUMP_TAPER  # in units of amplitude * samples
        bump_capacity = 2 * np.pi * BUMP_MAX_HZ * bump_area / samp_rate
        err_bump = np.clip(err, -bump_capacity, bump_capacity)
        bump_amp = err_bump * samp_rate / (2 * np.pi * bump_area)
        bump = np.full(b1 - b0, bump_amp)
        taper = 0.5 - 0.5 * np.cos(np.pi * np.arange(BUMP_TAPER) / BUMP_TAPER)
        bump[:BUMP_TAPER] *= taper
        bump[-BUMP_TAPER:] *= taper[::-1]
        f_prof[q + b0 : q + b1] += bump
        # Constant-offset spill over the rest-frequency hold (usually zero).
        f_prof[q + b1 : q + span - RAMP_LEN] += (
            (err - err_bump) * samp_rate / (2 * np.pi * hold_len)
        )
        phase = out_phase_at(start - q) + (
            2 * np.pi * np.concatenate(([0.0], np.cumsum(f_prof[:-1]))) / samp_rate
        )

        synth = np.linspace(amp_out, amp_in, span_ext) * np.cos(phase)
        blend = np.ones(span_ext)
        blend[:FADE_LEN] = fade
        cleaned[start - q : end] = cleaned[start - q : end] * (1.0 - blend) + synth * blend

    return cleaned


def _secam_method_diagnostic(field, chroma, linesout, outwidth):
    """For the first fields of a SECAM method 1 decode, check that the
    colour-under energy actually sits where method 1 puts it, and warn if the
    tape looks like it was recorded with the ME-SECAM heterodyne method
    instead (the two methods are mutually incompatible in chroma).

    The methods are told apart by band energy: the ME-SECAM carrier pair
    lives around 654/811 kHz while the method 1 pair lives around
    1062.5/1101.6 kHz, and unlike a carrier cluster measurement this works
    regardless of content saturation. (The back porch is not a usable rest
    carrier reference here as it is for the ME-SECAM servo: the deck's
    divide-by-4 counter output takes most of the porch to settle after the
    blanking edges / SECAM subcarrier phase reversals.)"""
    diag = getattr(field.rf, "secam_method_diag", None)
    if diag is None or diag["done"]:
        return

    samp_rate = field.rf.chroma_afc.true_samp_rate
    freqs, power = sps.welch(chroma, fs=samp_rate, nperseg=16384)

    mesecam_band = power[(freqs >= 550e3) & (freqs < 900e3)].mean()
    method1_band = power[(freqs >= 1000e3) & (freqs < 1300e3)].mean()

    if method1_band > 3 * mesecam_band:
        diag["method1"] += 1
    elif mesecam_band > 3 * method1_band:
        diag["mesecam"] += 1

    diag["fields"] += 1
    if diag["fields"] >= 20:
        diag["done"] = True
        ldd.logger.debug(
            "SECAM recording method check: %d/%d fields matched method 1, "
            "%d looked like ME-SECAM"
            % (diag["method1"], diag["fields"], diag["mesecam"])
        )
        if diag["mesecam"] > diag["method1"] and diag["mesecam"] >= 5:
            ldd.logger.warning(
                "The colour-under carriers look like ME-SECAM "
                "(pair around 654/811 kHz) rather than SECAM method 1 "
                "(1062.5/1101.6 kHz). If the colour comes out wrong, "
                "decode with --system MESECAM instead."
            )


def _process_chroma_secam_method1(field, chroma, linesout, outwidth, burstarea):
    """SECAM method 1 chroma restoration: x4 phase multiplication instead of
    a heterodyne mix, plus BT.470 bell amplitude regeneration."""
    _secam_method_diagnostic(field, chroma, linesout, outwidth)

    afc = field.rf.chroma_afc
    # Peak amplitude such that the undeviated carrier lands near the same
    # porch RMS level the other formats' chroma AGC normalizes to.
    rest_amplitude = field.rf.SysParams["burst_abs_ref"] * np.sqrt(2.0)
    restored, inst_freq, envelope = upconvert_secam_method1(
        chroma,
        afc.true_samp_rate,
        field.rf.Filters["FSecamUnder"],
        afc.carrier_mult,
        rest_amplitude,
        return_envelope=True,
    )

    STARTING_LINE = 16
    first_line = max(STARTING_LINE, field.burst_detected_line)

    # Give downstream decoders the undeviated blanking-interval reference the
    # standard promises them; what comes off tape there is the record
    # divider's settling transient (see regenerate_secam_blanking). This is a
    # two-pass restore: the first pass above identifies the lines, then the
    # blanking is replaced in the colour-under domain and the restoration is
    # run again on the cleaned signal, so the zero-phase filtering never gets
    # to smear the transient into the picture or the porch reference. The
    # interval runs right up to active video, so the fade-out lands on the
    # picture's own (band-limited, desaturated) colour turn-on strip rather
    # than next to the decoders' porch measurement window.
    porch_end_px = int(field.usectooutpx(field.rf.SysParams["activeVideoUS"][0]))
    fit = fit_secam_line_alternation(
        inst_freq, linesout, outwidth, first_line, porch_end_px
    )
    flywheel = getattr(field.rf, "secam_parity_flywheel", None)
    if flywheel is None:
        flywheel = SecamParityFlywheel()
        field.rf.secam_parity_flywheel = flywheel
    dr_on_even, parity_source = flywheel.resolve(field.readloc, fit)

    if dr_on_even is not None:
        # The record chain's blanking-edge transient sets in slightly before
        # the nominal end of active video (the source's own blanking edge
        # lands inside the TBC active window), so the splice starts a little
        # early.
        blank_start_px = int(
            field.usectooutpx(field.rf.SysParams["activeVideoUS"][1] - 0.85)
        )
        cleaned = regenerate_secam_blanking(
            chroma,
            envelope,
            afc.true_samp_rate,
            linesout,
            outwidth,
            blank_start_px,
            porch_end_px,
            first_line,
            dr_on_even,
            afc.carrier_mult,
        )
        restored, inst_freq = upconvert_secam_method1(
            cleaned,
            afc.true_samp_rate,
            field.rf.Filters["FSecamUnder"],
            afc.carrier_mult,
            rest_amplitude,
        )
        ldd.logger.debug(
            "SECAM blanking reference regenerated (%s, fit confidence %s)"
            % (parity_source, "%.02f" % fit[1] if fit is not None else "n/a")
        )
    else:
        ldd.logger.debug(
            "SECAM blanking left as-is (line ident confidence too low, "
            "no parity lock)"
        )

    uphet = restored[: linesout * outwidth]

    # Block-anchored final band-pass (same band as ME-SECAM).
    uphet = sosfiltfilt_rust(field.rf.Filters["FChromaFinal"], uphet)

    # No per-line chroma AGC here: the amplitude envelope was synthesised
    # from the BT.470 bell above, and normalizing every line to its porch
    # level would flatten the intended foR/foB rest amplitude difference.
    # Just blank the vertical interval / colour-killed lines and log the
    # porch level like acc() does for the other formats.
    uphet[: first_line * outwidth] = 0

    porch_rms_total = 0.0
    for linenumber in range(STARTING_LINE, linesout):
        linestart = linenumber * outwidth
        porch_rms_total += lddu.rms(
            uphet[linestart + burstarea[0] : linestart + burstarea[1]]
        )
    field.rf.field_averages.chroma_level.push(
        porch_rms_total / (linesout - STARTING_LINE)
    )

    return uphet
