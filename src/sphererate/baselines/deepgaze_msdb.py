from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F


EPS = 1e-8


def spherical_area_weights(height: int, width: int) -> np.ndarray:
    """Return ERP pixel-centre area factors, up to a shared constant."""
    latitude = np.pi / 2.0 - (np.arange(height, dtype=np.float64) + 0.5) * np.pi / height
    return np.broadcast_to(np.cos(latitude)[:, None], (height, width)).astype(np.float32).copy()


def normalize_spherical(density: np.ndarray, area: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    weights = np.asarray(area, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError(f"density and area must match, got {values.shape} and {weights.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("density and area must be finite")
    mass = float((values * weights).sum())
    if mass <= 0.0:
        values = np.ones_like(values)
        mass = float((values * weights).sum())
    return (values / mass).astype(np.float32)


def build_train_priors(final_targets: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a full train prior and leave-one-content-out priors for fitting."""
    targets = np.asarray(final_targets, dtype=np.float32)
    if targets.ndim != 3 or targets.shape[0] < 2:
        raise ValueError("final_targets must be [N,H,W] with N >= 2")
    normalized = np.stack([normalize_spherical(value, area) for value in targets])
    summed = normalized.sum(axis=0, dtype=np.float64)
    full = normalize_spherical(summed / len(normalized), area)
    loo = np.stack(
        [normalize_spherical((summed - normalized[index]) / (len(normalized) - 1), area) for index in range(len(normalized))]
    )
    return full, loo.astype(np.float32)


@dataclass(frozen=True)
class LogLinearCalibration:
    content_temperature: float
    prior_weight: float
    train_kld_sph: float
    steps: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "content_temperature": self.content_temperature,
            "prior_weight": self.prior_weight,
            "train_kld_sph": self.train_kld_sph,
            "steps": self.steps,
        }


def _torch_spherical_normalize(values: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    while area.ndim < values.ndim:
        area = area.unsqueeze(0)
    return values / (values * area).sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)


def _calibrated_torch(
    content: torch.Tensor,
    prior: torch.Tensor,
    area: torch.Tensor,
    raw_temperature: torch.Tensor,
    raw_prior_weight: torch.Tensor,
) -> torch.Tensor:
    temperature = F.softplus(raw_temperature) + 1e-3
    prior_weight = torch.sigmoid(raw_prior_weight)
    content_n = _torch_spherical_normalize(content.clamp_min(EPS), area)
    prior_n = _torch_spherical_normalize(prior.clamp_min(EPS), area)
    logits = (1.0 - prior_weight) * temperature * content_n.log() + prior_weight * prior_n.log()
    logits = logits - logits.amax(dim=(-2, -1), keepdim=True)
    return _torch_spherical_normalize(logits.exp(), area)


def fit_log_linear_calibration(
    content_maps: np.ndarray,
    loo_priors: np.ndarray,
    final_targets: np.ndarray,
    area: np.ndarray,
    *,
    steps: int = 300,
    learning_rate: float = 0.05,
    device: str | torch.device = "cuda",
) -> LogLinearCalibration:
    """Fit two train-only scalars with spherical KLD and no validation access."""
    content = torch.as_tensor(content_maps, dtype=torch.float32, device=device)
    prior = torch.as_tensor(loo_priors, dtype=torch.float32, device=device)
    target = torch.as_tensor(final_targets, dtype=torch.float32, device=device)
    area_tensor = torch.as_tensor(area, dtype=torch.float32, device=device)
    if content.shape != prior.shape or content.shape != target.shape or content.ndim != 3:
        raise ValueError("content_maps, loo_priors and final_targets must share [N,H,W]")
    target = _torch_spherical_normalize(target.clamp_min(EPS), area_tensor)

    raw_temperature = torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), device=device))
    raw_prior_weight = torch.nn.Parameter(torch.tensor(math.log(0.25 / 0.75), device=device))
    optimizer = torch.optim.Adam((raw_temperature, raw_prior_weight), lr=float(learning_rate))
    final_loss = float("nan")
    for _ in range(int(steps)):
        prediction = _calibrated_torch(
            content,
            prior,
            area_tensor,
            raw_temperature,
            raw_prior_weight,
        )
        loss = (target * area_tensor * ((target + EPS).log() - (prediction + EPS).log())).sum(dim=(-2, -1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return LogLinearCalibration(
        content_temperature=float((F.softplus(raw_temperature) + 1e-3).detach().cpu()),
        prior_weight=float(torch.sigmoid(raw_prior_weight).detach().cpu()),
        train_kld_sph=final_loss,
        steps=int(steps),
    )


def apply_log_linear_calibration(
    content_map: np.ndarray,
    train_prior: np.ndarray,
    area: np.ndarray,
    calibration: LogLinearCalibration,
) -> np.ndarray:
    content = normalize_spherical(content_map, area).astype(np.float64)
    prior = normalize_spherical(train_prior, area).astype(np.float64)
    alpha = float(calibration.prior_weight)
    logits = (1.0 - alpha) * float(calibration.content_temperature) * np.log(content + EPS)
    logits += alpha * np.log(prior + EPS)
    logits -= float(logits.max())
    return normalize_spherical(np.exp(logits), area)
