import numpy as np

import vhsdecode.process as process
import vhsdecode.utils as utils


class TestDemodPAL:
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

        max_demod = demod[1000 : (decoder.blocklen // 2) - 1000]
        np.testing.assert_allclose(
            max_demod, np.full(len(max_demod), max_hz), rtol=1.5e-04, atol=20
        )
        min_demod = demod[(decoder.blocklen // 2) + 1000 : decoder.blocklen - 1000]
        # Demodulated signal fluctuates more at the sync end than at the top end,
        # so allowing a little more tolerance here.
        np.testing.assert_allclose(
            min_demod, np.full(len(min_demod), min_hz), rtol=3e-04, atol=20
        )


class TestDemodNTSC:
    def test_ire(self):
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

        max_demod = demod[1000 : (decoder.blocklen // 2) - 1000]
        np.testing.assert_allclose(
            max_demod, np.full(len(max_demod), max_hz), rtol=1e-04, atol=20
        )
        min_demod = demod[(decoder.blocklen // 2) + 1000 : decoder.blocklen - 1000]
        # Demodulated signal fluctuates more at the sync end than at the top end,
        # so allowing a little more tolerance here.
        np.testing.assert_allclose(
            min_demod, np.full(len(min_demod), min_hz), rtol=1e-04, atol=50
        )
