import math

import logging
import numpy as np
import pytest

import lddecode.core as ldd
import vhsdecode.process as process
from vhsdecode.field import FieldPALVHS
from vhsdecode.addons.resync import _findpulses_numba_raw


@pytest.fixture
def pal_rfdecoder():
    ldd.logger = logging.getLogger("test")
    ldd.logger.setLevel(5)
    rf_options = {"level_detect_divisor": 2}
    return process.VHSRFDecode(
        inputfreq=40, system="PAL", tape_format="VHS", rf_options=rf_options
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
    return FieldPALVHS(rfdecoder, data_stub)


class TestSyncPAL:
    def test_sync_pal_good(self, pal_rfdecoder, data_dir):
        field = _make_field(pal_rfdecoder, data_dir / "PAL_GOOD.txt.gz")
        pulses = pal_rfdecoder.resync.get_pulses(field)
        assert len(pulses) == 458
        measured_sync, measured_blank = pal_rfdecoder.resync._field_state.pull_levels()
        assert math.isclose(measured_blank, 4133579.15, rel_tol=1e-3)
        assert math.isclose(measured_sync, 3840000, rel_tol=1e-3)

    def test_sync_pal_noisy(self, pal_rfdecoder, data_dir):
        field = _make_field(pal_rfdecoder, data_dir / "PAL_NOISY.txt.gz")
        pal_rfdecoder.resync.get_pulses(field)
        measured_sync, measured_blank = pal_rfdecoder.resync._field_state.pull_levels()
        assert math.isclose(measured_blank, 4130360.76, rel_tol=1e-3)
        assert math.isclose(measured_sync, 3800000, rel_tol=1e-3)


class TestFindPulses:
    def test_find_pulses_pal_good(self, data_dir):
        demod_05_data = np.loadtxt(data_dir / "PAL_GOOD.txt.gz")
        starts, lengths = _findpulses_numba_raw(demod_05_data, 3954307.8, 11.625, 1588.125)
        assert len(starts) == 458
        assert len(lengths) == 458
        assert starts[200] == 495955
        assert lengths[200] == 177


class TestLevelDetect:
    def test_level_detect_pal_good(self, pal_rfdecoder, data_dir):
        field = _make_field(pal_rfdecoder, data_dir / "PAL_GOOD.txt.gz")
        pal_rfdecoder.resync.get_pulses(field)
        blank_level = pal_rfdecoder.resync._field_state._blanklevels.current()
        sync_level = pal_rfdecoder.resync._field_state._synclevels.current()
        assert blank_level is not None
        assert sync_level is not None
