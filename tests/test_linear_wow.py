"""linear_wow must match the k=1 spline that computewow_scaled used to build."""

import numpy as np
import pytest
from scipy import interpolate

from lddecode.utils import linear_wow


@pytest.mark.parametrize("inlinelen", [2560.0, 2560.0 * 1.000001, 1280.0, 2270.0 + 1 / 3])
def test_matches_linear_spline(inlinelen):
    rng = np.random.default_rng(0)
    linelocs = np.cumsum(inlinelen + rng.normal(0, 3, 315))
    expected = np.array([i * inlinelen for i in range(len(linelocs))])
    spl = interpolate.make_interp_spline(expected, linelocs, k=1, check_finite=False)

    outscale = inlinelen / 1135
    # runs past the last line location, so the end segment is extrapolated
    count = 1135 * 316
    x = np.arange(count) * outscale

    locs, wow = linear_wow(linelocs, inlinelen, outscale, count)

    np.testing.assert_allclose(locs, spl(x), rtol=0, atol=1e-8)
    np.testing.assert_allclose(wow, spl(x, 1), rtol=1e-12)
