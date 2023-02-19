import unittest

import numpy as np

import vhsdecode.process as process
import vhsdecode.utils as utils


class DemodTest(unittest.TestCase):
    def test_ire(self):
        """Check that the IRE of the output corresponds to the correct frequencies."""
        samplerate_mhz = 40
        decoder = process.VHSRFDecode(inputfreq=samplerate_mhz, system="PAL")
        max_hz = 4800000
        max_mhz = max_hz / 1000000
        min_hz = 3800000
        min_mhz = min_hz / 1000000
        wavemax = utils.gen_wave_at_frequency(
            max_mhz, samplerate_mhz, decoder.blocklen / 2
        )
        wavemin = utils.gen_wave_at_frequency(
            min_mhz, samplerate_mhz, decoder.blocklen / 2
        )
        wave = np.concatenate((wavemax, wavemin))
        demod = decoder.demodblock(data=wave)["video"]["demod"]

        # Reference white
        max_ire_abs = decoder.iretohz(100)
        sync_ire_abs = decoder.iretohz(decoder.SysParams["vsync_ire"])
        print("max: ", max_ire_abs)
        print("min_ire: ", decoder.SysParams["vsync_ire"])
        print("min: ", sync_ire_abs)

        max_demod = demod[1000 : (decoder.blocklen // 2) - 1000]
        np.testing.assert_allclose(max_demod, np.full(len(max_demod), max_hz), rtol=1.5e-04, atol=20)
        min_demod = demod[(decoder.blocklen // 2) + 1000 : decoder.blocklen - 1000]
        # Demodulated signal fluctuates more at the sync end than at the top end,
        # so allowing a little more tolerance here.
        np.testing.assert_allclose(min_demod, np.full(len(min_demod), min_hz), rtol=3e-04, atol=20)

    def test_ire_ntsc(self):
        """Check that the IRE of the output corresponds to the correct frequencies."""
        samplerate_mhz = 40
        decoder = process.VHSRFDecode(inputfreq=samplerate_mhz, system="NTSC")
        max_hz = 4400000
        max_mhz = max_hz / 1000000
        min_hz = 3400000
        min_mhz = min_hz / 1000000
        wavemax = utils.gen_wave_at_frequency(
            max_mhz, samplerate_mhz, decoder.blocklen // 2
        )
        wavemin = utils.gen_wave_at_frequency(
            min_mhz, samplerate_mhz, decoder.blocklen // 2
        )
        wave = np.concatenate((wavemax, wavemin))
        demod = decoder.demodblock(data=wave)["video"]["demod"]

        max_ire_abs = decoder.iretohz(100)
        sync_ire_abs = decoder.iretohz(decoder.SysParams["vsync_ire"])
        print("max: ", max_ire_abs)
        print("min_ire: ", decoder.SysParams["vsync_ire"])
        print("min: ", sync_ire_abs)

        max_demod = demod[1000 : (decoder.blocklen // 2) - 1000]
        np.testing.assert_allclose(max_demod, np.full(len(max_demod), max_hz), rtol=1e-04, atol=20)
        min_demod = demod[(decoder.blocklen // 2) + 1000 : decoder.blocklen - 1000]
        # Demodulated signal fluctuates more at the sync end than at the top end,
        # so allowing a little more tolerance here.
        np.testing.assert_allclose(min_demod, np.full(len(min_demod), min_hz), rtol=1e-04, atol=50)


def test_sync(filename, num_pulses=None, blank_approx=None, sync_approx=None):
    import lddecode.core as ldd
    from vhsdecode.field import FieldPALVHS
    import logging
    import math

    samplerate_mhz = 40

    ldd.logger = logging.getLogger("test")
    ldd.logger.setLevel(5)
    ldd.logger.info("test")

    # process.VHSDecode("infile", "outfile", ,inputfreq=samplerate_mhz, system="PAL", tape_format="VHS")
    rf_options = {}
    rf_options["level_detect_divisor"] = 2
    rfdecoder = process.VHSRFDecode(inputfreq=samplerate_mhz, system="PAL", tape_format="VHS", rf_options=rf_options)

    demod_05_data = np.loadtxt(filename)

    data_stub = {}
    data_stub["input"] = np.zeros(5)
    data_stub["video"] = {}
    data_stub["video"]["demod"] = np.zeros_like(demod_05_data)
    data_stub["video"]["demod_05"] = demod_05_data

    field = FieldPALVHS(rfdecoder, data_stub)

    pulses = rfdecoder.resync.get_pulses(field)

    if num_pulses:
        assert(len(pulses) == num_pulses)

    measured_sync, measured_blank = rfdecoder.resync._field_state.pull_levels()

    if blank_approx:
        math.isclose(measured_blank, blank_approx)
    if sync_approx:
        math.isclose(measured_sync, sync_approx)

    return True


def test_find_pulses(filename, num_pulses):
    from vhsdecode.addons.resync import _findpulses_numba_raw

    demod_05_data = np.loadtxt(filename)

    # Just using some pre-tested values for now for optimizing function, many need changes later.
    starts, lengths = _findpulses_numba_raw(demod_05_data, 3954307.8, 11.625, 1588.125)

    assert len(starts) == num_pulses
    assert len(lengths) == num_pulses
    assert starts[200] == 495955
    assert lengths[200] == 177


class SyncTest(unittest.TestCase):
    def test_sync_pal_good(self):
        blank = 4130000
        sync = 3840000
        print("pal good")
        test_sync("PAL_GOOD.txt.gz", num_pulses=458, blank_approx=blank, sync_approx=sync)

    def test_sync_pal_noisy(self):
        blank = 4121000
        sync = 3800000
        print("pal noisy")
        test_sync("PAL_NOISY.txt.gz", blank_approx=blank, sync_approx=sync)

    def test_find_pulses(self):
        test_find_pulses("PAL_GOOD.txt.gz", 458)


if __name__ == "__main__":
    unittest.main()
