"""Calibrate multi-label decision thresholds using validation predictions only.

The shared, predeclared grid matches the Midterm sweep. All decisions, including
tie handling, are independent of the test set. No model weights are modified.
"""
import numpy as np

DEFAULT_GRID = np.round(np.linspace(0.05, 0.90, 35), 3)
TIE_RULE = "closest_to_0.5_then_smaller; no_validation_positives_use_0.5"


def validate_predictions(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 2 or y.shape != p.shape or not y.shape[0] or not y.shape[1]:
        raise ValueError("y and p must be non-empty matrices with the same shape")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only binary labels")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p must contain finite probabilities in [0, 1]")
    return y.astype(bool), p.astype(np.float64)


def _select_best(scores, grid):
    tied = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    return min(tied, key=lambda i: (round(abs(float(grid[i]) - 0.5), 12), float(grid[i])))


def calibrate_thresholds(y_val, p_val, grid=DEFAULT_GRID):
    """Return global and per-label thresholds maximizing validation F1 on grid.

F1 at threshold ties is resolved toward 0.5, then toward the smaller threshold.
A class without positive validation examples uses 0.5: there is no evidence for
estimating its recall. Scores use zero_division=0, as in src.metrics.
"""
    y, p = validate_predictions(y_val, p_val)
    grid = np.asarray(grid, dtype=np.float64)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or ((grid <= 0) | (grid >= 1)).any()
            or not (np.diff(grid) > 0).all() or not np.any(grid == 0.5)):
        raise ValueError("grid must be sorted, unique, in (0, 1), and include 0.5")
    # Work one threshold at a time to bound memory for larger validation sets.
    scores = []
    for threshold in grid:
        predicted = p >= threshold
        tp = (y & predicted).sum(axis=0)
        denominator = y.sum(axis=0) + predicted.sum(axis=0)
        scores.append(np.divide(2 * tp, denominator, out=np.zeros(y.shape[1]),
                                where=denominator != 0))
    scores = np.asarray(scores)  # (threshold, label)
    macro = scores.mean(axis=1)
    global_index = _select_best(macro, grid)
    indices = [_select_best(scores[:, c], grid) if y[:, c].any()
               else int(np.flatnonzero(grid == 0.5)[0]) for c in range(y.shape[1])]
    return {
        "grid": grid,
        "global_threshold": float(grid[global_index]),
        "per_label_thresholds": grid[indices],
        "per_label_f1_grid": scores,
        "global_macro_f1_grid": macro,
        "tie_rule": TIE_RULE,
    }
