import numpy as np
import pytest


@pytest.fixture
def hilbert_data(data_dir):
    return np.load(data_dir / "hilbert_data.npz")["data"]


class TestRustAngle:
    def test_matches_numpy(self, hilbert_data):
        from vhsd_rust import complex_angle_py

        expected = np.angle(hilbert_data)
        result = complex_angle_py(hilbert_data)
        assert (expected == result).all()


class TestRustUnwrap:
    def test_matches_numpy(self, hilbert_data):
        from vhsd_rust import unwrap_angles

        data = np.angle(hilbert_data)
        expected = np.unwrap(data)
        result = unwrap_angles(np.array(data))
        assert np.isclose(result, expected, atol=1e-15, rtol=1e-13).all()


class TestRustDiff:
    def test_matches_cython(self, hilbert_data):
        from vhsdecode.hilbert import diff_forward
        from vhsd_rust import diff_forward_in_place

        data = np.angle(hilbert_data)
        output_ediff = np.ediff1d(data, to_begin=0)
        output_cython = diff_forward(data)
        output_rust = np.copy(data)
        diff_forward_in_place(output_rust)
        assert (output_ediff == output_cython).all()
        assert (output_rust == output_cython).all()
