"""Approximate Median Significance (AMS) and threshold search."""
from __future__ import annotations

import math

import numpy as np

B_REG = 10.0


def ams(s: float, b: float, b_reg: float = B_REG) -> float:
    """AMS = sqrt(2*((s+b+b_reg)*ln(1+s/(b+b_reg)) - s)); s, b are weight sums."""
    if s < 0 or b < 0:
        raise ValueError("s and b must be >= 0")
    return math.sqrt(2.0 * ((s + b + b_reg) * math.log(1.0 + s / (b + b_reg)) - s))


def ams_reference(s: float, b: float) -> float:
    """Literal copy of the competition starting-kit formula, used only to test `ams`."""
    bReg = 10.0
    return math.sqrt(2 * ((s + b + bReg) * math.log(1 + s / (b + bReg)) - s))


def ams_from_selection(y_true_is_signal, selected, weights, scale: float = 1.0) -> float:
    """AMS for a boolean selection. `scale` rescales validation weights to full-set size
    (e.g. N_total / N_valid) so that s and b are comparable to the full-data scale."""
    y = np.asarray(y_true_is_signal, bool)
    sel = np.asarray(selected, bool)
    w = np.asarray(weights, float)
    s = w[sel & y].sum() * scale
    b = w[sel & ~y].sum() * scale
    return ams(s, b)


def ams_curve(y_true_is_signal, scores, weights, scale: float = 1.0, n_grid: int = 400):
    """AMS as a function of the *fraction of events labelled signal* (top-k by score).

    Returns (thresholds, ams_values, frac_selected).  Vectorised using cumulative sums
    over events sorted by decreasing score, so it is exact for every cut point.
    """
    y = np.asarray(y_true_is_signal, bool)
    sc = np.asarray(scores, float)
    w = np.asarray(weights, float)
    order = np.argsort(-sc, kind="mergesort")
    ws = w[order] * scale
    s_cum = np.cumsum(ws * y[order])
    b_cum = np.cumsum(ws * ~y[order])
    n = len(sc)
    idx = np.unique(np.linspace(0, n - 1, n_grid).astype(int))
    s_v, b_v = s_cum[idx], b_cum[idx]
    a = np.sqrt(2.0 * ((s_v + b_v + B_REG) * np.log1p(s_v / (b_v + B_REG)) - s_v))
    return sc[order][idx], a, (idx + 1) / n


def best_ams_threshold(y_true_is_signal, scores, weights, scale: float = 1.0, n_grid: int = 800):
    """Return (best_threshold, best_ams) over a dense grid of score cut points."""
    thr, a, _ = ams_curve(y_true_is_signal, scores, weights, scale, n_grid)
    i = int(np.nanargmax(a))
    return float(thr[i]), float(a[i])


def honest_ams(y_true_is_signal, scores, weights, scale: float = 1.0, seed: int = 0, n_grid: int = 1000):
    """Threshold-selection-aware AMS.

    The validation set is split in two stratified halves.  The fraction of events
    labelled signal is chosen on one half (maximising AMS) and applied to the other
    half; the two directions are averaged.  This removes the optimism of picking the
    threshold and reporting the score on the very same events.
    `scale` is N_full / N_validation (weights are rescaled to full-data size).
    """
    y = np.asarray(y_true_is_signal, bool)
    sc = np.asarray(scores, float)
    w = np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    half = np.zeros(len(y), bool)
    for cls in (True, False):
        ids = np.flatnonzero(y == cls)
        half[rng.permutation(ids)[: len(ids) // 2]] = True
    out = []
    for fit, ev in ((half, ~half), (~half, half)):
        s2 = scale * 2.0  # each half holds half of the validation events
        _, a, frac = ams_curve(y[fit], sc[fit], w[fit], s2, n_grid)
        q = frac[int(np.nanargmax(a))]
        k = int(np.ceil(q * ev.sum()))
        order = np.argsort(-sc[ev], kind="mergesort")[:k]
        sel = np.zeros(ev.sum(), bool)
        sel[order] = True
        out.append(ams_from_selection(y[ev], sel, w[ev], s2))
    return float(np.mean(out))


Q_GRID = np.arange(0.002, 0.402, 0.002)


def smooth_ams_curve(y_true_is_signal, scores, weights, scale: float = 1.0,
                     n_boot: int = 30, seed: int = 0, q_grid=Q_GRID):
    """Bootstrap-averaged AMS as a function of q = fraction of events labelled signal.

    A single AMS-vs-cut curve is extremely spiky when a few high-weight background events
    sit near the top of the ranking: one extra false positive can move the maximum a lot.
    Averaging the curve over bootstrap resamples of the evaluation set gives a smoother,
    more reproducible curve whose maximum is a more stable summary.
    Returns (q_grid, mean_curve, std_curve).
    """
    y = np.asarray(y_true_is_signal, bool)
    sc = np.asarray(scores, float)
    w = np.asarray(weights, float) * scale
    n = len(y)
    rng = np.random.default_rng(seed)
    ks = np.maximum(np.ceil(q_grid * n).astype(int) - 1, 0)
    curves = np.empty((n_boot, len(q_grid)))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        o = np.argsort(-sc[idx], kind="mergesort")
        ww, yy = w[idx][o], y[idx][o]
        s = np.cumsum(ww * yy)[ks]
        b = np.cumsum(ww * ~yy)[ks]
        curves[i] = np.sqrt(2.0 * ((s + b + B_REG) * np.log1p(s / (b + B_REG)) - s))
    return q_grid, curves.mean(0), curves.std(0)


def best_smooth_ams(y_true_is_signal, scores, weights, scale: float = 1.0, **kw):
    """(q*, smoothed AMS at q*) — maximum of the bootstrap-averaged curve."""
    q, m, _ = smooth_ams_curve(y_true_is_signal, scores, weights, scale, **kw)
    i = int(np.argmax(m))
    return float(q[i]), float(m[i])
