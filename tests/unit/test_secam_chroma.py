"""Tests for the SECAM/MESECAM chroma up-conversion.

ME-SECAM (IEC 60774-1 Annex E) is recorded like PAL colour-under with an
inverting mix against the PAL converter LO of 5060571.875 Hz, so up-converting
with that same LO must put the rest carriers back on the studio frequencies
foR = 4406250 Hz and foB = 4250000 Hz (ITU-R BT.470/BT.1700).

SECAM method 1 (IEC 60774-1 6.4.1, the standard method used in France) is
recorded through a divide-by-4 counter instead, so the tape carriers are
foB/4 = 1062500 Hz and foR/4 = 1101562.5 Hz with the deviations also divided
by 4, and restoration is a x4 phase multiplication that must scale carrier
and deviation back up together.
"""

import logging

import numpy as np
import scipy.signal as sps

import vhsdecode.process as process
import vhsdecode.formats as vhs_formats
from vhsdecode.addons.chromaAFC import ChromaAFC
from vhsdecode.chroma import (
    measure_secam_under_carrier_offset,
    secam_bell_gain,
    upconvert_secam_method1,
    SECAM_M1_UNDER_PAIR_CENTER,
    SECAM_M1_SEPARATION_RANGE,
)

FOR = 4406250.0
FOB = 4250000.0
CONVERSION_LO = 5060571.875
DR_UNDER = CONVERSION_LO - FOR  # 654321.875 Hz
DB_UNDER = CONVERSION_LO - FOB  # 810571.875 Hz

M1_DR_UNDER = FOR / 4  # 1101562.5 Hz
M1_DB_UNDER = FOB / 4  # 1062500 Hz

LINES = 313


def _get_params():
    return vhs_formats.get_format_params(
        "MESECAM", "VHS", 0, logging.getLogger("test")
    )


def _make_afc(sys_params, rf_params):
    return ChromaAFC(
        40e6,
        rf_params["chroma_bpf_upper"] / rf_params["color_under_carrier"],
        sys_params,
        rf_params["color_under_carrier"],
        tape_format="VHS",
        do_cafc=False,
        conversion_lo_freq=rf_params["chroma_conversion_lo"],
    )


def _make_under_signal(outlinelen, true_rate, crystal_error_hz):
    """Line-alternating colour-under rest carriers as they come off the TBC."""
    num_samples = LINES * outlinelen
    t = np.arange(num_samples) / true_rate
    sig = np.zeros(num_samples)
    for line in range(LINES):
        start, end = line * outlinelen, (line + 1) * outlinelen
        freq = (DR_UNDER if line % 2 else DB_UNDER) + crystal_error_hz
        sig[start:end] = np.sin(2 * np.pi * freq * t[start:end])
    return sig


def _measure_restored_carriers(afc, uphet, outlinelen, true_rate):
    """Median instantaneous frequency of the mixed+filtered signal per line parity."""
    filtered = sps.sosfiltfilt(afc.get_chroma_bandpass_final(True), uphet)
    analytic = sps.hilbert(filtered)
    f_inst = np.diff(np.unwrap(np.angle(analytic))) * true_rate / (2 * np.pi)
    odd, even = [], []
    for line in range(20, LINES - 20):
        start = line * outlinelen + 200
        end = (line + 1) * outlinelen - 200
        (odd if line % 2 else even).append(np.median(f_inst[start:end]))
    return np.median(odd), np.median(even)


class TestMESECAMUpconversion:
    def test_format_params(self):
        sys_params, rf_params = _get_params()

        assert rf_params["chroma_conversion_lo"] == CONVERSION_LO
        # fsc must stay at the PAL value so it is consistent with outlinelen
        # and the true TBC output rate.
        assert sys_params["fsc_mhz"] == 4.43361875
        assert rf_params["color_under_carrier"] == (DR_UNDER + DB_UNDER) / 2

    def test_carriers_restored_to_studio_frequencies(self):
        sys_params, rf_params = _get_params()
        afc = _make_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        true_rate = outlinelen * sys_params["FPS"] * sys_params["frame_lines"]
        num_samples = LINES * outlinelen

        sig = _make_under_signal(outlinelen, true_rate, 0.0)
        het = afc.getChromaHet()[0][:num_samples]
        restored_dr, restored_db = _measure_restored_carriers(
            afc, sig * het, outlinelen, true_rate
        )

        # The old code restored the carriers about 110 kHz high.
        np.testing.assert_allclose(restored_dr, FOR, atol=5)
        np.testing.assert_allclose(restored_db, FOB, atol=5)

    def test_servo_measures_crystal_error(self):
        sys_params, rf_params = _get_params()
        afc = _make_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        true_rate = outlinelen * sys_params["FPS"] * sys_params["frame_lines"]
        num_samples = LINES * outlinelen

        error = 1500.0
        sig = _make_under_signal(outlinelen, true_rate, error)

        active_start_px = int(10.5e-6 * true_rate)
        window = (active_start_px - 65, active_start_px - 5)
        measured = measure_secam_under_carrier_offset(
            sig, LINES, outlinelen, window, true_rate, afc.color_under
        )

        assert measured is not None
        # Single-field accuracy is on the order of +-100 Hz.
        np.testing.assert_allclose(measured, error, atol=150)

        # Applying the measurement as LO trim must bring the restored
        # carriers back near nominal.
        afc.updateConversion(measured, 0)
        het = afc.getChromaHet()[0][:num_samples]
        restored_dr, restored_db = _measure_restored_carriers(
            afc, sig * het, outlinelen, true_rate
        )
        np.testing.assert_allclose(restored_dr, FOR, atol=200)
        np.testing.assert_allclose(restored_db, FOB, atol=200)

    def test_servo_rejects_signal_without_carrier_pair(self):
        sys_params, rf_params = _get_params()
        afc = _make_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        true_rate = outlinelen * sys_params["FPS"] * sys_params["frame_lines"]

        rng = np.random.default_rng(1234)
        noise = rng.normal(0, 1, LINES * outlinelen)
        active_start_px = int(10.5e-6 * true_rate)
        window = (active_start_px - 65, active_start_px - 5)
        measured = measure_secam_under_carrier_offset(
            noise, LINES, outlinelen, window, true_rate, afc.color_under
        )

        assert measured is None

    def test_heterodyne_phase_continuous_across_fields(self):
        sys_params, rf_params = _get_params()
        afc = _make_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        num_samples = LINES * outlinelen

        afc.updateConversion(0.0, 0)
        field0 = afc.getChromaHet()[0][:num_samples].copy()
        afc.updateConversion(0.0, num_samples)
        field1 = afc.getChromaHet()[0][:outlinelen].copy()

        joined = np.concatenate([field0, field1])
        # A phase step shows up as a spike in the second difference.
        second_diff = np.abs(np.diff(joined, 2))
        boundary = second_diff[num_samples - 10 : num_samples + 10].max()
        assert boundary <= second_diff[: num_samples // 2].max() * 1.01


def _get_m1_params():
    return vhs_formats.get_format_params(
        "SECAM", "VHS", 0, logging.getLogger("test")
    )


def _make_m1_afc(sys_params, rf_params):
    return ChromaAFC(
        40e6,
        rf_params["chroma_bpf_upper"] / rf_params["color_under_carrier"],
        sys_params,
        rf_params["color_under_carrier"],
        tape_format="VHS",
        do_cafc=False,
        carrier_mult=rf_params["chroma_carrier_mult"],
    )


def _make_m1_under_signal(outlinelen, true_rate, deviation_hz):
    """Line-alternating method 1 colour-under carriers as they come off the
    TBC: foB/4 on even lines, foR/4 on odd lines, offset by a per-line
    constant tape-domain deviation (studio deviation / 4)."""
    num_samples = LINES * outlinelen
    t = np.arange(num_samples) / true_rate
    sig = np.zeros(num_samples)
    for line in range(LINES):
        start, end = line * outlinelen, (line + 1) * outlinelen
        freq = (M1_DR_UNDER if line % 2 else M1_DB_UNDER) + deviation_hz
        sig[start:end] = np.sin(2 * np.pi * freq * t[start:end])
    return sig


def _restore_m1(afc, sig):
    restored, _ = upconvert_secam_method1(
        sig,
        afc.true_samp_rate,
        afc.get_secam_under_bandpass(),
        afc.carrier_mult,
        5000.0 * np.sqrt(2.0),
    )
    return restored


class TestSECAMMethod1Upconversion:
    def test_format_params(self):
        sys_params, rf_params = _get_m1_params()

        assert rf_params["color_under_carrier"] == (M1_DR_UNDER + M1_DB_UNDER) / 2
        assert rf_params["chroma_carrier_mult"] == 4
        # Method 1 restores by phase multiplication; there is no conversion
        # crystal in the chain, so it must NOT get a conversion LO (that
        # would route it down the heterodyne path).
        assert "chroma_conversion_lo" not in rf_params
        assert rf_params["chroma_rotation"] is None
        # fsc must stay at the PAL value so it is consistent with outlinelen
        # and the true TBC output rate.
        assert sys_params["fsc_mhz"] == 4.43361875

    def test_carriers_restored_to_studio_frequencies(self):
        sys_params, rf_params = _get_m1_params()
        afc = _make_m1_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        true_rate = afc.true_samp_rate

        sig = _make_m1_under_signal(outlinelen, true_rate, 0.0)
        restored = _restore_m1(afc, sig)
        restored_dr, restored_db = _measure_restored_carriers(
            afc, restored, outlinelen, true_rate
        )

        np.testing.assert_allclose(restored_dr, FOR, atol=5)
        np.testing.assert_allclose(restored_db, FOB, atol=5)

    def test_deviation_restored_times_four(self):
        sys_params, rf_params = _get_m1_params()
        afc = _make_m1_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        true_rate = afc.true_samp_rate

        # Nominal D'R deviation is 280 kHz at studio, 70 kHz on tape; the
        # x4 multiplication must scale the deviation back up together with
        # the carrier (this is what makes the format self-time-correcting).
        tape_deviation = 70e3 / 4  # deliberately small: 17.5 kHz on tape
        sig = _make_m1_under_signal(outlinelen, true_rate, tape_deviation)
        restored = _restore_m1(afc, sig)
        restored_dr, restored_db = _measure_restored_carriers(
            afc, restored, outlinelen, true_rate
        )

        np.testing.assert_allclose(restored_dr, FOR + 4 * tape_deviation, atol=100)
        np.testing.assert_allclose(restored_db, FOB + 4 * tape_deviation, atol=100)

    def test_bell_amplitude_regenerated(self):
        sys_params, rf_params = _get_m1_params()
        afc = _make_m1_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        true_rate = afc.true_samp_rate

        sig = _make_m1_under_signal(outlinelen, true_rate, 0.0)
        restored = _restore_m1(afc, sig)

        # The divide-by-4 counter output has constant amplitude, so the
        # BT.470 HF bell must be regenerated at restore time: the D'R rest
        # carrier (4.40625 MHz) sits further up the bell skirt than D'B
        # (4.25 MHz), G(foR)/G(foB) = 1.2868.
        envelope = np.abs(sps.hilbert(restored))
        odd, even = [], []
        for line in range(20, LINES - 20):
            start = line * outlinelen + 200
            end = (line + 1) * outlinelen - 200
            (odd if line % 2 else even).append(np.median(envelope[start:end]))
        ratio = np.median(odd) / np.median(even)

        np.testing.assert_allclose(ratio, 1.2868, rtol=0.05)
        # And the formula itself against values computed from BT.470-6
        # (G = |1+j16F| / |1+j1.26F|, F = f/f0 - f0/f, f0 = 4286 kHz).
        np.testing.assert_allclose(
            secam_bell_gain(np.array([FOR, FOB])),
            [1.3325067, 1.0355542],
            rtol=1e-6,
        )

    def test_blanking_regeneration(self):
        from vhsdecode.chroma import (
            fit_secam_line_alternation,
            regenerate_secam_blanking,
        )

        sys_params, rf_params = _get_m1_params()
        afc = _make_m1_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        true_rate = afc.true_samp_rate

        # Under signal with the whole blanking interval of every line
        # polluted by an off-frequency transient like the record divider's
        # settling artifact (~960 kHz at 2x amplitude, as measured on tape).
        porch_end = int(10.3e-6 * true_rate)
        blank_start = outlinelen - int(0.85e-6 * true_rate)
        sig = _make_m1_under_signal(outlinelen, true_rate, 0.0)
        t = np.arange(len(sig)) / true_rate
        junk = 2.0 * np.sin(2 * np.pi * 960e3 * t)
        for line in range(LINES - 1):
            s = line * outlinelen + blank_start
            e = (line + 1) * outlinelen + porch_end
            sig[s:e] = junk[s:e]

        restored, inst_freq, envelope = upconvert_secam_method1(
            sig,
            true_rate,
            afc.get_secam_under_bandpass(),
            afc.carrier_mult,
            5000.0 * np.sqrt(2.0),
            return_envelope=True,
        )
        fit = fit_secam_line_alternation(inst_freq, LINES, outlinelen, 16, porch_end)
        assert fit is not None
        dr_on_even, confidence = fit
        assert confidence > 0.9
        # _make_m1_under_signal puts foB/4 on even lines
        assert dr_on_even == False  # noqa: E712

        cleaned = regenerate_secam_blanking(
            sig,
            envelope,
            true_rate,
            LINES,
            outlinelen,
            blank_start,
            porch_end,
            16,
            dr_on_even,
            afc.carrier_mult,
        )
        restored2, _ = upconvert_secam_method1(
            cleaned,
            true_rate,
            afc.get_secam_under_bandpass(),
            afc.carrier_mult,
            5000.0 * np.sqrt(2.0),
        )

        # The restored porch must measure at the rest carriers per line
        # parity (this is the window downstream decoders calibrate their
        # discriminator zeros from), and the end of active video must no
        # longer carry the transient.
        analytic = sps.hilbert(restored2)
        f_inst = np.diff(np.unwrap(np.angle(analytic))) * true_rate / (2 * np.pi)
        porch_odd, porch_even, edge_odd, edge_even = [], [], [], []
        porch_meas_start = int(6.6e-6 * true_rate)
        for line in range(30, LINES - 30):
            s = line * outlinelen + porch_meas_start
            e = line * outlinelen + porch_end - 10
            (porch_odd if line % 2 else porch_even).append(np.median(f_inst[s:e]))
            s = (line + 1) * outlinelen - int(2.5e-6 * true_rate)
            e = (line + 1) * outlinelen - int(1.2e-6 * true_rate)
            (edge_odd if line % 2 else edge_even).append(np.median(f_inst[s:e]))
        np.testing.assert_allclose(np.median(porch_odd), FOR, atol=1500)
        np.testing.assert_allclose(np.median(porch_even), FOB, atol=1500)
        # Last ~2 us of active video: with the raw junk the zero-phase
        # band-pass smears the transient here (hundreds of kHz); cleaned it
        # must stay within a few kHz of the line's carrier.
        np.testing.assert_allclose(np.median(edge_odd), FOR, atol=8000)
        np.testing.assert_allclose(np.median(edge_even), FOB, atol=8000)

    def test_alternation_fit_rejects_nonalternating(self):
        from vhsdecode.chroma import (
            fit_secam_line_alternation,
            SECAM_IDENT_MIN_CONFIDENCE,
            SecamParityFlywheel,
        )

        sys_params, _ = _get_m1_params()
        outlinelen = sys_params["outlinelen"]

        # Random line identities (e.g. noise with the colour killer off) must
        # not fabricate a reference: the fit comes back with low confidence
        # and an unlocked flywheel must not resolve a parity from it.
        rng = np.random.default_rng(42)
        inst_freq = rng.uniform(3.9e6, 4.756e6, LINES * outlinelen)
        fit = fit_secam_line_alternation(
            inst_freq, LINES, outlinelen, 16, int(10.3e-6 * 17734475.0)
        )
        assert fit is not None
        assert fit[1] < SECAM_IDENT_MIN_CONFIDENCE

        flywheel = SecamParityFlywheel()
        parity, source = flywheel.resolve(0, fit)
        assert parity is None
        assert source == "unlocked"

    def test_parity_flywheel_predicts_and_resets(self):
        from vhsdecode.chroma import SecamParityFlywheel

        # dr_on_even walks base ^ (((n + 1) >> 1) & 1) across consecutive
        # fields (the 312.5-line field offset; TFFT observed on all fixture
        # tapes).
        pattern = lambda n, base: bool(base ^ (((n + 1) >> 1) & 1))  # noqa: E731

        flywheel = SecamParityFlywheel()
        # Confident fields teach the cycle...
        for n in range(4):
            parity, source = flywheel.resolve(n * 1000, (pattern(n, True), 1.0))
            assert source == "measured"
            assert parity == pattern(n, True)
        # ...then unfittable fields (near-neutral content) inherit it.
        for n in range(4, 12):
            parity, source = flywheel.resolve(n * 1000, (True, 0.55))
            assert source == "flywheel"
            assert parity == pattern(n, True)
        # Same readloc = same field reprocessed: the index must not advance.
        parity_again, _ = flywheel.resolve(11 * 1000, (True, 0.55))
        assert parity_again == parity

        # A confident contradiction (e.g. a dropped field shifted the cycle
        # phase) resets the lock; predictions stop until it re-locks.
        parity, source = flywheel.resolve(12000, (not pattern(12, True), 0.95))
        assert source == "measured"
        parity, source = flywheel.resolve(13000, (True, 0.55))
        assert parity is None and source == "unlocked"

    def test_porch_pair_measurable_with_method1_gates(self):
        sys_params, rf_params = _get_m1_params()
        afc = _make_m1_afc(sys_params, rf_params)
        outlinelen = sys_params["outlinelen"]
        true_rate = afc.true_samp_rate

        sig = _make_m1_under_signal(outlinelen, true_rate, 0.0)
        active_start_px = int(10.5e-6 * true_rate)
        window = (active_start_px - 65, active_start_px - 5)
        measured = measure_secam_under_carrier_offset(
            sig,
            LINES,
            outlinelen,
            window,
            true_rate,
            SECAM_M1_UNDER_PAIR_CENTER,
            SECAM_M1_SEPARATION_RANGE,
        )

        assert measured is not None
        np.testing.assert_allclose(measured, 0.0, atol=150)


class TestSECAMMethod1DecoderConstruction:
    def test_construct(self):
        decoder = process.VHSRFDecode(inputfreq=40, system="SECAM")

        # SECAM has no phase-locked burst, so burst-locked hsync must be off.
        assert decoder.options.disable_burst_hsync
        assert not decoder.do_cafc
        # Phase multiplication, not a heterodyne.
        assert decoder.chroma_afc.conversion_lo is None
        assert decoder.chroma_afc.carrier_mult == 4
        assert "FSecamUnder" in decoder.Filters
        # The recording-method sanity check starts fresh.
        assert decoder.secam_method_diag["fields"] == 0
        assert not decoder.secam_method_diag["done"]


class TestMESECAMDecoderConstruction:
    def test_construct(self):
        decoder = process.VHSRFDecode(inputfreq=40, system="MESECAM")

        # SECAM has no phase-locked burst, so burst-locked hsync must be off.
        assert decoder.options.disable_burst_hsync
        # cafc peak measurement is meaningless on the two-carrier SECAM signal.
        assert not decoder.do_cafc
        assert decoder.chroma_afc.conversion_lo == CONVERSION_LO
        # No trim seed given, so the servo average starts empty and no trim
        # is applied until enough fields have been measured.
        assert len(decoder.secam_servo_avg) <= 2

    def test_lo_trim_seed(self):
        decoder = process.VHSRFDecode(
            inputfreq=40,
            system="MESECAM",
            rf_options={"secam_lo_trim": 2000.0},
        )

        # Seeded (e.g. by the two-pass calibration) so the trim applies from
        # the first field.
        assert len(decoder.secam_servo_avg) > 2
        np.testing.assert_allclose(np.mean(decoder.secam_servo_avg), 2000.0)
