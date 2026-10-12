from __future__ import annotations

import numpy as np

from sphererate.evaluation.metrics import auc_judd, cc, kld, nss, sim


def _auc_judd_bruteforce(pred: np.ndarray, fixation: np.ndarray) -> float:
    fixation_mask = (fixation > 0.5).ravel()
    scores = pred.ravel()
    n_fix = int(fixation_mask.sum())
    n_nonfix = int((~fixation_mask).sum())
    thresholds = np.sort(scores[fixation_mask])[::-1]
    tp = [0.0]
    fp = [0.0]
    for threshold in thresholds:
        above = scores >= threshold
        tp.append(float((above & fixation_mask).sum()) / n_fix)
        fp.append(float((above & ~fixation_mask).sum()) / n_nonfix)
    tp.append(1.0)
    fp.append(1.0)
    order = np.argsort(fp, kind="stable")
    return float(np.trapezoid(np.asarray(tp)[order], np.asarray(fp)[order]))


def test_metrics_prefer_matching_map() -> None:
    gt = np.zeros((8, 16), dtype=np.float32)
    gt[3, 7] = 1.0
    pred_good = gt.copy()
    pred_bad = np.ones_like(gt) / gt.size
    fixation = gt.copy()

    assert cc(pred_good, gt) > cc(pred_bad, gt)
    assert sim(pred_good, gt) > sim(pred_bad, gt)
    assert kld(pred_good, gt) < kld(pred_bad, gt)
    assert nss(pred_good, fixation) > nss(pred_bad, fixation)
    assert auc_judd(pred_good, fixation) >= auc_judd(pred_bad, fixation)


def test_cc_self_correlation_is_one_for_normalized_density() -> None:
    rng = np.random.default_rng(7)
    density = rng.random((64, 128), dtype=np.float32)
    density /= density.sum()

    assert np.isclose(cc(density, density), 1.0, atol=1e-6)


def test_cc_is_positive_affine_invariant() -> None:
    rng = np.random.default_rng(11)
    gt = rng.random((16, 32), dtype=np.float32)
    pred = 3.5 * gt + 0.2

    assert np.isclose(cc(pred, gt), 1.0, atol=1e-6)


def test_weighted_cc_self_correlation_is_one() -> None:
    rng = np.random.default_rng(13)
    density = rng.random((32, 64), dtype=np.float32)
    density /= density.sum()
    latitude_weight = np.cos(np.linspace(-np.pi / 2, np.pi / 2, 32, dtype=np.float32))
    weights = np.repeat(latitude_weight[:, None], 64, axis=1)

    assert np.isclose(cc(density, density, weights), 1.0, atol=1e-6)


def test_cc_constant_map_returns_zero_without_nan() -> None:
    constant = np.ones((8, 16), dtype=np.float32)
    varying = np.arange(128, dtype=np.float32).reshape(8, 16)

    assert cc(constant, varying) == 0.0
    assert cc(varying, constant) == 0.0


def test_vectorized_auc_judd_matches_bruteforce_with_ties() -> None:
    pred = np.asarray(
        [[0.1, 0.2, 0.2, 0.9], [0.4, 0.4, 0.7, 0.3], [0.8, 0.2, 0.5, 0.5]],
        dtype=np.float32,
    )
    fixation = np.asarray(
        [[0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]],
        dtype=np.float32,
    )

    assert np.isclose(auc_judd(pred, fixation), _auc_judd_bruteforce(pred, fixation), atol=1e-12)
