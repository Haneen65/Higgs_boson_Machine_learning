import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from higgs.metrics import ams, ams_reference, ams_curve, ams_from_selection  # noqa: E402


def test_ams_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s, b = rng.uniform(0, 500), rng.uniform(0, 5000)
        assert math.isclose(ams(s, b), ams_reference(s, b), rel_tol=1e-12)


def test_ams_zero_signal_is_zero():
    assert ams(0.0, 100.0) == 0.0


def test_curve_matches_direct_computation():
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.random(n) < 0.3
    sc = rng.normal(size=n) + y
    w = np.where(y, 0.01, 2.5)
    thr, a, frac = ams_curve(y, sc, w, n_grid=50)
    for t, v in zip(thr[5:10], a[5:10]):
        direct = ams_from_selection(y, sc >= t, w)
        assert math.isclose(direct, v, rel_tol=1e-9)


def test_honest_ams_not_above_oracle():
    from higgs.metrics import best_ams_threshold, honest_ams
    rng = np.random.default_rng(2)
    n = 20000
    y = rng.random(n) < 0.3
    sc = rng.normal(size=n) + 1.5 * y
    w = np.where(y, 0.01, 2.5)
    oracle = best_ams_threshold(y, sc, w, 5.0)[1]
    honest = honest_ams(y, sc, w, 5.0)
    assert honest <= oracle * 1.02 and honest > 0


def test_smooth_curve_shape_and_range():
    from higgs.metrics import smooth_ams_curve, best_smooth_ams
    rng = np.random.default_rng(3)
    n = 10000
    y = rng.random(n) < 0.3
    sc = rng.normal(size=n) + 1.5 * y
    w = np.where(y, 0.01, 2.5)
    q, m, sd = smooth_ams_curve(y, sc, w, 5.0, n_boot=10)
    assert len(q) == len(m) == len(sd) and np.all(np.isfinite(m)) and m.max() > 0
    qs, a = best_smooth_ams(y, sc, w, 5.0, n_boot=10)
    assert 0 < qs <= 0.4 and a > 0
