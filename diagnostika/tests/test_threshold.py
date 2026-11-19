import inspect

import numpy as np
import pytest

from src.stage1_binary import threshold as thr


def _brute_force(y, p, target):
    best = None
    for t in np.unique(p):
        if np.mean(p[y == 1] >= t) >= target and (best is None or t > best):
            best = t
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_threshold_is_largest_feasible_value(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.35 * y + rng.random(300) * 0.7, 0, 1)
    out = thr.choose_threshold(y, p, 0.90)
    assert out["threshold"] == pytest.approx(_brute_force(y, p, 0.90))
    assert out["val_sensitivity"] >= 0.90


def test_specificity_is_reported_at_the_chosen_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1])
    out = thr.choose_threshold(y, p, 0.75)
    assert out["threshold"] == pytest.approx(0.7)
    assert out["val_sensitivity"] == pytest.approx(0.75)
    assert out["val_specificity"] == pytest.approx(1.0)  # all benign scores are below 0.7


def test_selection_takes_only_validation_arrays():
    params = list(inspect.signature(thr.choose_threshold).parameters)
    assert params == ["y_true", "p_malignant", "target_sensitivity"]


def test_result_does_not_change_when_unrelated_test_data_changes():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    first = thr.choose_threshold(y, p, 0.9)
    _unused_test_labels = rng.integers(0, 2, 50)  # e.g. a test split exists; it is not an input
    second = thr.choose_threshold(y, p, 0.9)
    assert first == second


def test_both_classes_required():
    with pytest.raises(ValueError):
        thr.choose_threshold(np.zeros(10), np.linspace(0, 1, 10), 0.9)
