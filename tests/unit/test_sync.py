import math

import logging
import numpy as np
import pytest

import lddecode.core as ldd
import vhsdecode.process as process
from vhsdecode.field import FieldPALTape

@pytest.fixture
def pal_rfdecoder():
    ldd.logger = logging.getLogger("test")
    ldd.logger.setLevel(5)
    return process.VHSRFDecode(
        inputfreq=40, system="PAL", tape_format="VHS"
    )


def _make_field(rfdecoder, filename):
    demod_05_data = np.loadtxt(filename)
    data_stub = {
        "input": np.zeros(5),
        "video": {
            "demod": np.zeros_like(demod_05_data),
            "demod_05": demod_05_data,
        },
    }
    return FieldPALTape(rfdecoder, data_stub)


class TestSyncPAL:
    def test_sync_pal_good(self, pal_rfdecoder, data_dir):
        field = _make_field(pal_rfdecoder, data_dir / "PAL_GOOD.txt.gz")
        pulses = field.get_pulses()
        assert len(pulses) == 458
        measured_sync, measured_blank = field.sync_tip_level, field.blanking_level
        assert math.isclose(measured_blank, 4133579.15, rel_tol=1e-3)
        assert math.isclose(measured_sync, 3840000, rel_tol=1e-3)

    def test_sync_pal_noisy(self, pal_rfdecoder, data_dir):
        field = _make_field(pal_rfdecoder, data_dir / "PAL_NOISY.txt.gz")
        _ = field.get_pulses()
        measured_sync, measured_blank = field.sync_tip_level, field.blanking_level
        assert math.isclose(measured_blank, 4137794.68, rel_tol=1e-3)
        assert math.isclose(measured_sync, 3795981.06, rel_tol=1e-3)


class TestLevelDetect:
    def test_level_detect_pal_good(self, pal_rfdecoder, data_dir):
        field = _make_field(pal_rfdecoder, data_dir / "PAL_GOOD.txt.gz")
        _ = field.get_pulses(True)
        sync_level, blank_level = field.sync_tip_level, field.blanking_level
        assert sync_level is not None
        assert blank_level is not None
