import numpy as np

from lddecode.utils import calczc as c_calczc


class TestCalcZC:
    def test_falling_edge(self):
        data = np.array([8.0, 4.0, 1.0, 8.0, 1.0, 4.0, 8.0])
        result = c_calczc(data, 0, 3.0, edge=-1, count=len(data))
        assert result is not None

    def test_rising_edge(self):
        data = np.array([8.0, 4.0, 1.0, 8.0, 1.0, 4.0, 8.0])
        result = c_calczc(data, 2, 3.0, edge=1, count=len(data))
        assert result is not None

    def test_no_crossing_returns_none(self):
        data2 = np.array([8.0, 4.0, 4.0, 8.0, 30.0, 99.0, 8.0])
        result = c_calczc(data2, 0, 3.0, edge=1, count=len(data2))
        assert result is None
