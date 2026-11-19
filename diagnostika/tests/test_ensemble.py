import numpy as np
import pytest

from src.inference.cascade import decide
from src.inference.ensemble import (class_weight_matrix, ensemble_benign, final_distribution,
                                    renormalize_benign)

MODELS = ["resnet50", "densenet121", "efficientnet_b7", "vit_b_16"]
CLASSES = ["bdb", "cb", "db", "srjd", "other_benign"]


def _random_probs(seed, n=40):
    rng = np.random.default_rng(seed)
    raw = rng.random((len(MODELS), n, len(CLASSES))) + 1e-3
    return renormalize_benign(raw)


@pytest.mark.parametrize("mode", ["average", "weighted"])
def test_ensemble_sums_to_one(mode):
    P = _random_probs(0)
    weights = {m: {c: float(np.random.default_rng(i).random()) for c in CLASSES} for i, m in enumerate(MODELS)}
    W = class_weight_matrix(weights, MODELS, CLASSES) if mode == "weighted" else None
    E = ensemble_benign(P, mode, W)
    assert np.allclose(E.sum(axis=-1), 1.0)
    p_mal = np.random.default_rng(1).random(P.shape[1])
    final = final_distribution(p_mal, E)
    assert np.allclose(final.sum(axis=-1), 1.0)
    assert final.shape == (P.shape[1], len(CLASSES) + 1)


def test_weighted_with_equal_weights_equals_average():
    P = _random_probs(3)
    equal = {m: {c: 0.5 for c in CLASSES} for m in MODELS}
    W = class_weight_matrix(equal, MODELS, CLASSES)
    assert np.allclose(ensemble_benign(P, "weighted", W), ensemble_benign(P, "average"))


def test_weight_matrix_normalized_across_models_per_class():
    weights = {m: {c: float(i + 1) for c in CLASSES} for i, m in enumerate(MODELS)}
    W = class_weight_matrix(weights, MODELS, CLASSES)
    assert np.allclose(W.sum(axis=0), 1.0)


def test_zero_f1_for_a_class_falls_back_to_uniform_weights():
    weights = {m: {c: 0.0 if c == "srjd" else 0.8 for c in CLASSES} for m in MODELS}
    W = class_weight_matrix(weights, MODELS, CLASSES)
    srjd = CLASSES.index("srjd")
    assert np.allclose(W[:, srjd], 0.25)


def test_decide_referral_unclear_and_agreement():
    per_model = np.zeros((4, 5))
    per_model[:, 1] = 0.9          # all four models say cb
    per_model[0, :] = [0.2, 0.2, 0.2, 0.2, 0.2]  # first model is flat, top is argmax (index 0)
    per_model = renormalize_benign(per_model)
    ens = np.array([0.1, 0.6, 0.1, 0.1, 0.1])
    out = decide(0.92, ens, per_model, CLASSES, threshold=0.5, unclear_below=0.5)
    assert out["referral"] is True and out["subtype_for_information_only"] is True
    assert out["top_benign"] == "cb"
    assert out["agreement"] == 3          # the flat model's argmax is index 0, not cb
    assert out["unclear"] is False
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-9
    assert len(out["probabilities"]) == 6

    low = decide(0.2, np.full(5, 0.2), np.full((4, 5), 0.2), CLASSES, threshold=0.5, unclear_below=0.5)
    assert low["referral"] is False
    assert low["unclear"] is True         # max final probability = 0.8*0.2 = 0.16 < 0.5
