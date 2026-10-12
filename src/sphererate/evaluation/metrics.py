from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-7


def _as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def normalize_density(x: np.ndarray, weights: np.ndarray | None = None, eps: float = EPS) -> np.ndarray:
    x = np.maximum(_as_float(x), 0.0)
    denom = float(x.sum() if weights is None else (x * weights).sum())
    if denom <= eps:
        return np.ones_like(x, dtype=np.float32) / float(x.size)
    return x / denom


def cc(pred: np.ndarray, gt: np.ndarray, weights: np.ndarray | None = None, eps: float = EPS) -> float:
    del eps  # Correlation is scale-free; an absolute map-space epsilon biases normalized densities.
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if weights is None:
        p = pred - pred.mean()
        g = gt - gt.mean()
        cov = float((p * g).mean())
        ps = float(np.sqrt((p * p).mean()))
        gs = float(np.sqrt((g * g).mean()))
        denom = ps * gs
        if not np.isfinite(denom) or denom == 0.0:
            return 0.0
        return float(np.clip(cov / denom, -1.0, 1.0))
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    weight_sum = float(w.sum())
    if not np.isfinite(weight_sum) or weight_sum == 0.0:
        return 0.0
    w = w / weight_sum
    pm = float((pred * w).sum())
    gm = float((gt * w).sum())
    p = pred - pm
    g = gt - gm
    cov = float((w * p * g).sum())
    ps = float(np.sqrt((w * p * p).sum()))
    gs = float(np.sqrt((w * g * g).sum()))
    denom = ps * gs
    if not np.isfinite(denom) or denom == 0.0:
        return 0.0
    return float(np.clip(cov / denom, -1.0, 1.0))


def kld(pred: np.ndarray, gt: np.ndarray, weights: np.ndarray | None = None, eps: float = EPS) -> float:
    pred_n = normalize_density(pred, weights, eps)
    gt_n = normalize_density(gt, weights, eps)
    if weights is None:
        return float((gt_n * np.log((gt_n + eps) / (pred_n + eps))).sum())
    return float((weights * gt_n * np.log((gt_n + eps) / (pred_n + eps))).sum())


def sim(pred: np.ndarray, gt: np.ndarray, weights: np.ndarray | None = None, eps: float = EPS) -> float:
    pred_n = normalize_density(pred, weights, eps)
    gt_n = normalize_density(gt, weights, eps)
    if weights is None:
        return float(np.minimum(pred_n, gt_n).sum())
    return float((weights * np.minimum(pred_n, gt_n)).sum())


def nss(pred: np.ndarray, fixation: np.ndarray, eps: float = EPS) -> float:
    pred = _as_float(pred)
    fixation = _as_float(fixation) > 0.5
    if fixation.sum() == 0:
        return float("nan")
    z = (pred - pred.mean()) / (pred.std() + eps)
    return float(z[fixation].mean())


def auc_judd(pred: np.ndarray, fixation: np.ndarray, eps: float = EPS) -> float:
    pred = _as_float(pred)
    fixation = (_as_float(fixation) > 0.5).ravel()
    scores = pred.ravel()
    n_fix = int(fixation.sum())
    n_nonfix = int((~fixation).sum())
    if n_fix == 0 or n_nonfix == 0:
        return float("nan")
    # AUC-Judd thresholds are saliency values at fixation pixels. Counting every
    # image pixel for every threshold is prohibitively expensive for dense 360°
    # maps, so use sorted score arrays and binary searches for identical counts.
    fixation_scores = np.sort(scores[fixation])
    nonfixation_scores = np.sort(scores[~fixation])
    thresholds = np.unique(fixation_scores)[::-1]
    tp_counts = n_fix - np.searchsorted(fixation_scores, thresholds, side="left")
    fp_counts = n_nonfix - np.searchsorted(nonfixation_scores, thresholds, side="left")
    tp = np.concatenate(([0.0], tp_counts.astype(np.float64) / n_fix, [1.0]))
    fp = np.concatenate(([0.0], fp_counts.astype(np.float64) / n_nonfix, [1.0]))
    order = np.argsort(fp, kind="stable")
    return float(np.trapezoid(tp[order], fp[order]))


@dataclass
class SaliencyScores:
    auc_judd: float
    nss: float
    cc: float
    sim: float
    kld: float
    cc_sph: float
    sim_sph: float
    kld_sph: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auc_judd": self.auc_judd,
            "nss": self.nss,
            "cc": self.cc,
            "sim": self.sim,
            "kld": self.kld,
            "cc_sph": self.cc_sph,
            "sim_sph": self.sim_sph,
            "kld_sph": self.kld_sph,
        }


def saliency_scores(
    pred: np.ndarray,
    gt: np.ndarray,
    fixation: np.ndarray,
    area_weights: np.ndarray | None = None,
) -> SaliencyScores:
    return SaliencyScores(
        auc_judd=auc_judd(pred, fixation),
        nss=nss(pred, fixation),
        cc=cc(pred, gt),
        sim=sim(pred, gt),
        kld=kld(pred, gt),
        cc_sph=cc(pred, gt, area_weights),
        sim_sph=sim(pred, gt, area_weights),
        kld_sph=kld(pred, gt, area_weights),
    )


def mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=np.float32)
        out[key] = float(np.nanmean(vals))
    return out
