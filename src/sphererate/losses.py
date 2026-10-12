from __future__ import annotations

import math

import torch
from torch.nn import functional as F


EPS = 1e-7


def _norm(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        denom = x.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    else:
        while weights.ndim < x.ndim:
            weights = weights.unsqueeze(1)
        denom = (x * weights).sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    return x.clamp_min(0.0) / denom


def match_temporal_target_count(
    temporal: torch.Tensor,
    prediction_count: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match stored temporal bins to a smaller adjacent-stage prediction.

    The clean processed data retain six equal-order bins. Gate C deliberately
    uses three stages, so each adjacent pair is summed and area-normalized. The
    final saliency/fixation targets are intentionally not touched.
    """

    if temporal.ndim != 4:
        raise ValueError(f"Expected temporal target [B,T,H,W], got shape={tuple(temporal.shape)}")
    target_count = int(temporal.shape[1])
    prediction_count = int(prediction_count)
    if prediction_count < 1:
        raise ValueError("prediction_count must be positive")
    if target_count < prediction_count or target_count % prediction_count != 0:
        raise ValueError(
            f"Temporal target count {target_count} must be an integer multiple of prediction count {prediction_count}"
        )
    if target_count == prediction_count:
        matched = temporal
    else:
        group_size = target_count // prediction_count
        batch, _, height, width = temporal.shape
        grouped = temporal.reshape(batch, prediction_count, group_size, height, width).sum(dim=2)
        matched = _norm(grouped, weights)

    cumulative = []
    running = torch.zeros_like(matched[:, :1])
    for index in range(prediction_count):
        running = running + matched[:, index : index + 1]
        cumulative.append(_norm(running, weights))
    return matched, torch.cat(cumulative, dim=1)


def weighted_kld(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return weighted_kld_values(pred, gt, weights).mean()


def weighted_kld_values(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    pred_n = _norm(pred, weights)
    gt_n = _norm(gt, weights)
    if weights is None:
        val = gt_n * torch.log((gt_n + EPS) / (pred_n + EPS))
    else:
        while weights.ndim < pred.ndim:
            weights = weights.unsqueeze(1)
        val = weights * gt_n * torch.log((gt_n + EPS) / (pred_n + EPS))
    values = val.sum(dim=(-2, -1))
    if values.ndim > 1:
        values = values.mean(dim=tuple(range(1, values.ndim)))
    return values


def weighted_cc(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    pred_f = pred.float()
    gt_f = gt.float()
    if weights is None:
        pc = pred_f - pred_f.mean(dim=(-2, -1), keepdim=True)
        gc = gt_f - gt_f.mean(dim=(-2, -1), keepdim=True)
        cov = (pc * gc).mean(dim=(-2, -1))
        pv = (pc * pc).mean(dim=(-2, -1))
        gv = (gc * gc).mean(dim=(-2, -1))
    else:
        weights_f = weights.float().clamp_min(0.0)
        while weights_f.ndim < pred_f.ndim:
            weights_f = weights_f.unsqueeze(1)
        weight_sum = weights_f.sum(dim=(-2, -1), keepdim=True)
        w = weights_f / weight_sum.clamp_min(torch.finfo(pred_f.dtype).tiny)
        pm = (pred_f * w).sum(dim=(-2, -1), keepdim=True)
        gm = (gt_f * w).sum(dim=(-2, -1), keepdim=True)
        pc = pred_f - pm
        gc = gt_f - gm
        cov = (w * pc * gc).sum(dim=(-2, -1))
        pv = (w * pc * pc).sum(dim=(-2, -1))
        gv = (w * gc * gc).sum(dim=(-2, -1))

    den = torch.sqrt(pv.clamp_min(0.0) * gv.clamp_min(0.0))
    valid = den > 0.0
    safe_den = den.clamp_min(torch.finfo(den.dtype).tiny)
    score = torch.where(valid, cov / safe_den, torch.zeros_like(cov))
    return score.clamp(min=-1.0, max=1.0).mean()


def weighted_sim(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    pred_n = _norm(pred, weights)
    gt_n = _norm(gt, weights)
    overlap = torch.minimum(pred_n, gt_n)
    if weights is None:
        return overlap.sum(dim=(-2, -1)).mean()
    while weights.ndim < overlap.ndim:
        weights = weights.unsqueeze(1)
    return (overlap * weights).sum(dim=(-2, -1)).mean()


def spherical_tile_mass(
    density: torch.Tensor,
    weights: torch.Tensor,
    *,
    tile_rows: int = 8,
    tile_cols: int = 16,
) -> torch.Tensor:
    """Aggregate a spherical density into normalized ERP tile probability mass."""

    if density.ndim != 4:
        raise ValueError(f"density must be [B,K,H,W], got {tuple(density.shape)}")
    rows, cols = int(tile_rows), int(tile_cols)
    batch, horizons, height, width = density.shape
    if rows <= 0 or cols <= 0 or height % rows or width % cols:
        raise ValueError("tile grid must be positive and divide the density shape")
    area = weights.to(dtype=density.dtype, device=density.device).clamp_min(0.0)
    while area.ndim < density.ndim:
        area = area.unsqueeze(1)
    if area.shape[0] not in {1, batch} or area.shape[-2:] != (height, width):
        raise ValueError(f"incompatible area shape: {tuple(area.shape)}")
    if area.shape[1] not in {1, horizons}:
        raise ValueError(f"incompatible area horizon dimension: {tuple(area.shape)}")
    mass = density.clamp_min(0.0) * area
    mass = mass / mass.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    tile_h, tile_w = height // rows, width // cols
    return mass.reshape(batch, horizons, rows, tile_h, cols, tile_w).sum(dim=(3, 5))


def tile_mass_kl_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    tile_rows: int = 8,
    tile_cols: int = 16,
) -> torch.Tensor:
    pred_mass = spherical_tile_mass(
        prediction, weights, tile_rows=tile_rows, tile_cols=tile_cols
    )
    target_mass = spherical_tile_mass(target, weights, tile_rows=tile_rows, tile_cols=tile_cols)
    value = target_mass * torch.log((target_mass + EPS) / (pred_mass + EPS))
    return value.sum(dim=(-2, -1)).mean()


def tile_pairwise_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    tile_rows: int = 8,
    tile_cols: int = 16,
    budgets_percent: tuple[float, ...] = (5.0, 10.0),
    temperature: float = 0.25,
) -> torch.Tensor:
    """Rank GT top-mass tiles above the strongest predicted hard negatives."""

    pred_mass = spherical_tile_mass(
        prediction, weights, tile_rows=tile_rows, tile_cols=tile_cols
    ).flatten(2)
    target_mass = spherical_tile_mass(
        target, weights, tile_rows=tile_rows, tile_cols=tile_cols
    ).flatten(2)
    scores = torch.log(pred_mass + EPS)
    total_tiles = int(scores.shape[-1])
    losses = []
    for raw_budget in budgets_percent:
        budget = float(raw_budget)
        if not 0.0 < budget < 100.0:
            raise ValueError("rank-loss budgets must be strictly between 0 and 100")
        count = min(total_tiles - 1, max(1, int(math.ceil(total_tiles * budget / 100.0))))
        positive_index = target_mass.topk(count, dim=-1).indices
        positive_score = scores.gather(-1, positive_index)
        positive_mass = target_mass.gather(-1, positive_index)
        positive_mask = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, positive_index, True)
        negative_count = min(count, total_tiles - count)
        negative_scores_all = scores.masked_fill(positive_mask, -torch.inf)
        negative_index = negative_scores_all.topk(negative_count, dim=-1).indices
        negative_score = scores.gather(-1, negative_index)
        negative_mass = target_mass.gather(-1, negative_index)
        margin = (
            negative_score.unsqueeze(-2) - positive_score.unsqueeze(-1)
        ) / max(float(temperature), 1e-4)
        importance = (positive_mass.unsqueeze(-1) - negative_mass.unsqueeze(-2)).clamp_min(0.0)
        importance = importance + 0.05 * positive_mass.unsqueeze(-1)
        pair_loss = F.softplus(margin) * importance
        losses.append(pair_loss.sum() / importance.sum().clamp_min(EPS))
    if not losses:
        raise ValueError("at least one tile budget is required")
    return torch.stack(losses).mean()


def _normalized_raw_tile_target(
    target_mass: torch.Tensor,
    prediction_mass: torch.Tensor,
) -> torch.Tensor:
    if target_mass.ndim != 4:
        raise ValueError(f"raw tile target must be [B,K,R,C], got {tuple(target_mass.shape)}")
    if target_mass.shape != prediction_mass.shape:
        raise ValueError(
            f"raw tile target {tuple(target_mass.shape)} does not match prediction tile mass "
            f"{tuple(prediction_mass.shape)}"
        )
    target = target_mass.to(
        dtype=prediction_mass.dtype, device=prediction_mass.device
    ).clamp_min(0.0)
    return target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)


def _horizon_weighted_mean(
    values: torch.Tensor,
    horizon_weights: torch.Tensor | None,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"expected per-sample/per-horizon values, got {tuple(values.shape)}")
    if horizon_weights is None:
        return values.mean()
    weights = torch.as_tensor(
        horizon_weights, dtype=values.dtype, device=values.device
    ).flatten()
    if weights.numel() != values.shape[1] or torch.any(weights < 0) or not torch.any(weights > 0):
        raise ValueError("horizon_weights must contain one non-negative value per horizon")
    weights = weights / weights.sum()
    return (values * weights.view(1, -1)).sum(dim=1).mean()


def raw_dwell_tile_kl_loss(
    prediction: torch.Tensor,
    raw_target_mass: torch.Tensor,
    weights: torch.Tensor,
    *,
    tile_rows: int = 8,
    tile_cols: int = 16,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match predicted spherical tile mass to unsmoothed clipped dwell mass."""

    pred_mass = spherical_tile_mass(
        prediction, weights, tile_rows=tile_rows, tile_cols=tile_cols
    )
    target_mass = _normalized_raw_tile_target(raw_target_mass, pred_mass)
    values = (
        target_mass * torch.log((target_mass + EPS) / (pred_mass + EPS))
    ).sum(dim=(-2, -1))
    return _horizon_weighted_mean(values, horizon_weights)


def raw_dwell_frontier_rank_loss(
    prediction: torch.Tensor,
    raw_target_mass: torch.Tensor,
    weights: torch.Tensor,
    *,
    tile_rows: int = 8,
    tile_cols: int = 16,
    budgets_percent: tuple[float, ...] = (5.0, 10.0, 20.0),
    frontier_width: int = 4,
    temperature: float = 0.25,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Repair ordering at the exact tile cutoffs used by Coverage.

    For each budget, the weakest predicted members of the GT top-B set are
    contrasted with the strongest predicted tiles outside that set.  This
    targets the discontinuous 7/13/26-tile decision frontier without creating
    budget-specific output maps.
    """

    pred_mass = spherical_tile_mass(
        prediction, weights, tile_rows=tile_rows, tile_cols=tile_cols
    ).flatten(2)
    target_mass = _normalized_raw_tile_target(
        raw_target_mass,
        pred_mass.reshape(
            pred_mass.shape[0], pred_mass.shape[1], int(tile_rows), int(tile_cols)
        ),
    ).flatten(2)
    scores = torch.log(pred_mass + EPS)
    total_tiles = int(scores.shape[-1])
    width = max(1, int(frontier_width))
    per_budget: list[torch.Tensor] = []
    for raw_budget in budgets_percent:
        budget = float(raw_budget)
        if not 0.0 < budget < 100.0:
            raise ValueError("rank-loss budgets must be strictly between 0 and 100")
        count = min(total_tiles - 1, max(1, int(math.ceil(total_tiles * budget / 100.0))))
        positive_index = target_mass.topk(count, dim=-1).indices
        positive_scores = scores.gather(-1, positive_index)
        positive_mass = target_mass.gather(-1, positive_index)
        current_width = min(width, count, total_tiles - count)
        hard_positive_local = positive_scores.topk(
            current_width, dim=-1, largest=False
        ).indices
        hard_positive_score = positive_scores.gather(-1, hard_positive_local)
        hard_positive_mass = positive_mass.gather(-1, hard_positive_local)

        positive_mask = torch.zeros_like(scores, dtype=torch.bool).scatter(
            -1, positive_index, True
        )
        negative_scores = scores.masked_fill(positive_mask, -torch.inf)
        hard_negative_index = negative_scores.topk(current_width, dim=-1).indices
        hard_negative_score = scores.gather(-1, hard_negative_index)
        hard_negative_mass = target_mass.gather(-1, hard_negative_index)

        margin = (
            hard_negative_score.unsqueeze(-2) - hard_positive_score.unsqueeze(-1)
        ) / max(float(temperature), 1e-4)
        importance = (
            hard_positive_mass.unsqueeze(-1) - hard_negative_mass.unsqueeze(-2)
        ).clamp_min(0.0)
        importance = importance + 0.05 * hard_positive_mass.unsqueeze(-1)
        numerator = (F.softplus(margin) * importance).sum(dim=(-2, -1))
        denominator = importance.sum(dim=(-2, -1)).clamp_min(EPS)
        per_budget.append(numerator / denominator)
    if not per_budget:
        raise ValueError("at least one tile budget is required")
    per_horizon = torch.stack(per_budget, dim=0).mean(dim=0)
    return _horizon_weighted_mean(per_horizon, horizon_weights)


def _soft_topk_membership_ot(
    scores: torch.Tensor,
    count: int,
    *,
    epsilon: float = 0.15,
    iterations: int = 30,
) -> torch.Tensor:
    """Entropy-regularized two-bin OT relaxation of a top-k indicator.

    Every tile supplies equal mass; the selected bin has exactly ``count``
    tiles of capacity.  The returned soft membership therefore lies near
    ``[0,1]`` and sums to ``count`` while remaining differentiable.
    """

    if scores.ndim < 2:
        raise ValueError("scores must have a tile dimension")
    tile_count = int(scores.shape[-1])
    count = int(count)
    if not 0 < count < tile_count:
        raise ValueError("count must be strictly between zero and tile_count")
    if float(epsilon) <= 0.0 or int(iterations) <= 0:
        raise ValueError("epsilon and iterations must be positive")

    centered = scores - scores.mean(dim=-1, keepdim=True)
    scale = centered.detach().std(dim=-1, keepdim=True).clamp_min(1e-4)
    normalized = centered / scale
    # Column 0 is selected.  High scores have lower selected transport cost.
    log_kernel = torch.stack(
        [normalized / float(epsilon), torch.zeros_like(normalized)], dim=-1
    )
    log_a = torch.full_like(scores, -math.log(float(tile_count)))
    proportions = scores.new_tensor(
        [float(count) / tile_count, float(tile_count - count) / tile_count]
    )
    log_b = proportions.log().view(*([1] * (scores.ndim - 1)), 2)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b).expand(*scores.shape[:-1], 2).clone()
    for _ in range(int(iterations)):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
    plan_selected = torch.exp(log_kernel[..., 0] + log_u + log_v[..., 0].unsqueeze(-1))
    return (float(tile_count) * plan_selected).clamp(0.0, 1.0)


def raw_dwell_soft_topk_coverage_loss(
    prediction: torch.Tensor,
    raw_target_mass: torch.Tensor,
    weights: torch.Tensor,
    *,
    tile_rows: int = 8,
    tile_cols: int = 16,
    budgets_percent: tuple[float, ...] = (5.0, 10.0, 20.0),
    epsilon: float = 0.15,
    iterations: int = 30,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Minimize normalized raw-dwell coverage regret at deployment budgets."""

    pred_mass = spherical_tile_mass(
        prediction, weights, tile_rows=tile_rows, tile_cols=tile_cols
    ).flatten(2)
    target_mass = _normalized_raw_tile_target(
        raw_target_mass,
        pred_mass.reshape(
            pred_mass.shape[0], pred_mass.shape[1], int(tile_rows), int(tile_cols)
        ),
    ).flatten(2)
    scores = torch.log(pred_mass + EPS)
    total_tiles = int(scores.shape[-1])
    per_budget: list[torch.Tensor] = []
    for raw_budget in budgets_percent:
        budget = float(raw_budget)
        if not 0.0 < budget < 100.0:
            raise ValueError("coverage budgets must be strictly between 0 and 100")
        count = min(total_tiles - 1, max(1, int(math.ceil(total_tiles * budget / 100.0))))
        membership = _soft_topk_membership_ot(
            scores, count, epsilon=epsilon, iterations=iterations
        )
        covered = (membership * target_mass).sum(dim=-1)
        oracle = target_mass.topk(count, dim=-1).values.sum(dim=-1).detach().clamp_min(EPS)
        per_budget.append((oracle - covered) / oracle)
    if not per_budget:
        raise ValueError("at least one tile budget is required")
    per_horizon = torch.stack(per_budget, dim=0).mean(dim=0)
    return _horizon_weighted_mean(per_horizon, horizon_weights)


def raw_dwell_consensus_rank_loss(
    prediction: torch.Tensor,
    topk_consensus: torch.Tensor,
    weights: torch.Tensor,
    *,
    tile_rows: int = 8,
    tile_cols: int = 16,
    temperature: float = 0.25,
    minimum_difference: float = 0.1,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank tiles using soft top-k membership frequencies across observer halves."""

    pred_mass = spherical_tile_mass(
        prediction, weights, tile_rows=tile_rows, tile_cols=tile_cols
    ).flatten(2)
    consensus = topk_consensus.to(dtype=pred_mass.dtype, device=pred_mass.device)
    if consensus.ndim != 5 or consensus.shape[:2] != pred_mass.shape[:2]:
        raise ValueError("top-k consensus must be [B,K,J,R,C]")
    if consensus.shape[-2:] != (int(tile_rows), int(tile_cols)):
        raise ValueError("top-k consensus tile grid mismatch")
    consensus = consensus.flatten(3).clamp(0.0, 1.0)
    scores = torch.log(pred_mass + EPS)
    per_budget = []
    for budget_index in range(consensus.shape[2]):
        membership = consensus[:, :, budget_index]
        target_difference = membership.unsqueeze(-1) - membership.unsqueeze(-2)
        importance = target_difference.clamp_min(0.0)
        importance = importance * (importance >= float(minimum_difference)).to(importance.dtype)
        score_margin = (
            scores.unsqueeze(-2) - scores.unsqueeze(-1)
        ) / max(float(temperature), 1e-4)
        numerator = (F.softplus(score_margin) * importance).sum(dim=(-2, -1))
        denominator = importance.sum(dim=(-2, -1)).clamp_min(EPS)
        per_budget.append(numerator / denominator)
    if not per_budget:
        raise ValueError("top-k consensus must contain at least one budget")
    per_horizon = torch.stack(per_budget, dim=0).mean(dim=0)
    return _horizon_weighted_mean(per_horizon, horizon_weights)


def _weighted_spatial_center(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    area = weights.to(dtype=values.dtype, device=values.device).clamp_min(0.0)
    while area.ndim < values.ndim:
        area = area.unsqueeze(1)
    normalized = area / area.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    return values - (values * normalized).sum(dim=(-2, -1), keepdim=True)


def semantic_log_ratio_target(
    template: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    clip: float = 4.0,
) -> torch.Tensor:
    template_n = _norm(template, weights)
    target_n = _norm(target, weights)
    ratio = torch.log((target_n + EPS) / (template_n + EPS))
    return _weighted_spatial_center(ratio, weights).clamp(min=-float(clip), max=float(clip))


def semantic_log_ratio_loss(
    residual_logits: torch.Tensor,
    template: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    clip: float = 4.0,
) -> torch.Tensor:
    residual = _weighted_spatial_center(residual_logits, weights)
    ratio_target = semantic_log_ratio_target(template, target, weights, clip=clip).detach()
    area = weights.to(dtype=residual.dtype, device=residual.device).clamp_min(0.0)
    while area.ndim < residual.ndim:
        area = area.unsqueeze(1)
    area = area / area.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    value = F.smooth_l1_loss(residual, ratio_target, reduction="none")
    return (value * area).sum(dim=(-2, -1)).mean()


def temporal_profile_kld(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    pred_pos = pred.clamp_min(0.0)
    gt_pos = gt.clamp_min(0.0)
    pred_t = pred_pos / pred_pos.sum(dim=1, keepdim=True).clamp_min(EPS)
    gt_t = gt_pos / gt_pos.sum(dim=1, keepdim=True).clamp_min(EPS)

    mass = gt_pos.sum(dim=1, keepdim=True).detach()
    if weights is not None:
        while weights.ndim < mass.ndim:
            weights = weights.unsqueeze(1)
        mass = mass * weights
    mass = mass / mass.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    val = (gt_t * torch.log((gt_t + EPS) / (pred_t + EPS))).sum(dim=1, keepdim=True)
    return (val * mass).sum(dim=(-2, -1)).mean()


def _step_tv_values(stack: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if stack.shape[1] < 2:
        return torch.zeros(stack.shape[0], 1, dtype=stack.dtype, device=stack.device)
    stack_n = _norm(stack, weights)
    diff = (stack_n[:, 1:] - stack_n[:, :-1]).abs()
    if weights is None:
        return 0.5 * diff.sum(dim=(-2, -1))
    while weights.ndim < diff.ndim:
        weights = weights.unsqueeze(1)
    return 0.5 * (diff * weights).sum(dim=(-2, -1))


def weighted_step_tv(stack: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return _step_tv_values(stack, weights).mean()


def step_tv_loss(pred: torch.Tensor, gt: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    pred_tv = _step_tv_values(pred, weights)
    gt_tv = _step_tv_values(gt.detach(), weights)
    return F.smooth_l1_loss(pred_tv, gt_tv)


def _teacher_residual_target(
    residual_logits: torch.Tensor,
    static_pred: torch.Tensor,
    temporal_gt: torch.Tensor,
    weights: torch.Tensor | None = None,
    clip: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    static_n = _norm(static_pred.detach().expand_as(temporal_gt), weights)
    temporal_n = _norm(temporal_gt, weights)
    residual_target = torch.log((temporal_n + EPS) / (static_n + EPS))
    residual_target = residual_target - residual_target.mean(dim=(-2, -1), keepdim=True)
    residual_target = residual_target.clamp(min=-clip, max=clip)
    residual_pred = residual_logits - residual_logits.mean(dim=(-2, -1), keepdim=True)
    return residual_pred, residual_target


def _teacher_residual_topk_loss(
    residual_pred: torch.Tensor,
    residual_target: torch.Tensor,
    weights: torch.Tensor | None = None,
    frac: float = 0.1,
) -> torch.Tensor:
    frac = min(max(float(frac), 1e-4), 1.0)
    b, k, h, w = residual_target.shape
    flat_target = residual_target.detach().abs().reshape(b, k, h * w)
    topk = max(1, int(round(frac * h * w)))
    threshold = flat_target.topk(topk, dim=-1).values[..., -1].view(b, k, 1, 1)
    mask = (residual_target.detach().abs() >= threshold).to(dtype=residual_pred.dtype)
    if weights is not None:
        while weights.ndim < mask.ndim:
            weights = weights.unsqueeze(1)
        mask = mask * weights
    denom = mask.sum().clamp_min(EPS)
    return (F.smooth_l1_loss(residual_pred, residual_target.detach(), reduction="none") * mask).sum() / denom


def _spatial_centroid(mass: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    b, k, h, w = mass.shape
    yy = torch.linspace(-1.0, 1.0, h, dtype=mass.dtype, device=mass.device).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, w, dtype=mass.dtype, device=mass.device).view(1, 1, 1, w)
    if weights is not None:
        while weights.ndim < mass.ndim:
            weights = weights.unsqueeze(1)
        mass = mass * weights
    mass = mass.clamp_min(0.0)
    denom = mass.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    x = (mass * xx).sum(dim=(-2, -1), keepdim=True) / denom
    y = (mass * yy).sum(dim=(-2, -1), keepdim=True) / denom
    return torch.cat([x, y], dim=-1).view(b, k, 2)


def _teacher_residual_peak_loss(
    residual_pred: torch.Tensor,
    residual_target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    target_pos = residual_target.detach().clamp_min(0.0)
    pred_pos = F.softplus(residual_pred)
    target_centroid = _spatial_centroid(target_pos, weights)
    pred_centroid = _spatial_centroid(pred_pos, weights)
    target_mass = target_pos.sum(dim=(-2, -1))
    active = (target_mass > EPS).to(dtype=residual_pred.dtype)
    loss = F.smooth_l1_loss(pred_centroid, target_centroid.detach(), reduction="none").sum(dim=-1)
    return (loss * active).sum() / active.sum().clamp_min(1.0)


def nss(pred: torch.Tensor, fixation: torch.Tensor) -> torch.Tensor:
    z = (pred - pred.mean(dim=(-2, -1), keepdim=True)) / (pred.std(dim=(-2, -1), keepdim=True) + EPS)
    fix = (fixation > 0.5).float()
    denom = fix.sum(dim=(-2, -1)).clamp_min(1.0)
    return ((z * fix).sum(dim=(-2, -1)) / denom).mean()


def _point_log_prob_values(pred: torch.Tensor, fixation: torch.Tensor) -> torch.Tensor:
    fix = (fixation > 0.5).to(dtype=pred.dtype)
    denom = fix.sum(dim=(-2, -1)).clamp_min(1.0)
    return (pred.clamp_min(EPS).log() * fix).sum(dim=(-2, -1)) / denom


def _argmax_yx(maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = maps.shape
    flat_idx = maps[:, 0].reshape(b, h * w).argmax(dim=-1)
    y = torch.div(flat_idx, w, rounding_mode="floor")
    x = flat_idx.remainder(w)
    return y, x


def _motion_target_units(
    batch: dict[str, torch.Tensor],
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    image = batch.get("image")
    fixation = batch.get("fixation")
    if image is None or fixation is None:
        return None
    prefix_start = 8
    prefix_length = int(
        cfg.get("model", {}).get(
            "prefix_length",
            cfg.get("train", {}).get("scanpath_prefix_length", 3),
        )
    )
    prefix_encoding = str(cfg.get("train", {}).get("scanpath_prefix_encoding", "sequence")).lower()
    current_idx = prefix_start + 1 if prefix_encoding == "collapsed" else prefix_start + prefix_length - 1
    if image.shape[1] <= current_idx:
        return None

    current = image[:, current_idx : current_idx + 1]
    current_y, current_x = _argmax_yx(current)
    target_y, target_x = _argmax_yx(fixation)
    _, _, h, w = fixation.shape

    current_x_f = current_x.to(dtype=fixation.dtype)
    target_x_f = target_x.to(dtype=fixation.dtype)
    current_y_f = current_y.to(dtype=fixation.dtype)
    target_y_f = target_y.to(dtype=fixation.dtype)
    width = torch.tensor(float(w), dtype=fixation.dtype, device=fixation.device)
    height = torch.tensor(float(max(h - 1, 1)), dtype=fixation.dtype, device=fixation.device)

    dx = torch.remainder(target_x_f - current_x_f + width / 2.0, width) - width / 2.0
    dx_unit = (dx / width + 0.5).clamp(min=0.0, max=1.0 - EPS)
    dy_unit = (((target_y_f - current_y_f) / height) + 1.0).mul(0.5).clamp(min=0.0, max=1.0 - EPS)
    return dx_unit, dy_unit


def _motion_bin_targets(batch: dict[str, torch.Tensor], cfg: dict, bins_y: int, bins_x: int) -> torch.Tensor | None:
    units = _motion_target_units(batch, cfg)
    if units is None:
        return None
    dx_unit, dy_unit = units
    x_bin = torch.floor(dx_unit * float(bins_x)).long().clamp(min=0, max=bins_x - 1)
    y_bin = torch.floor(dy_unit * float(bins_y)).long().clamp(min=0, max=bins_y - 1)
    return y_bin * bins_x + x_bin


def _motion_soft_targets(
    batch: dict[str, torch.Tensor],
    cfg: dict,
    bins_y: int,
    bins_x: int,
    sigma_bins: float = 1.0,
) -> torch.Tensor | None:
    units = _motion_target_units(batch, cfg)
    if units is None:
        return None
    dx_unit, dy_unit = units
    sigma_bins = max(float(sigma_bins), 1e-3)
    dtype = dx_unit.dtype
    device = dx_unit.device
    x_centers = (torch.arange(bins_x, dtype=dtype, device=device) + 0.5) / float(bins_x)
    y_centers = (torch.arange(bins_y, dtype=dtype, device=device) + 0.5) / float(bins_y)
    dx_bin = torch.remainder(x_centers.view(1, 1, bins_x) - dx_unit.view(-1, 1, 1) + 0.5, 1.0) - 0.5
    dx_bin = dx_bin * float(bins_x)
    dy_bin = (y_centers.view(1, bins_y, 1) - dy_unit.view(-1, 1, 1)) * float(bins_y)
    logits = -0.5 * (dx_bin.square() + dy_bin.square()) / (sigma_bins * sigma_bins)
    target = torch.softmax(logits.reshape(dx_unit.shape[0], bins_y * bins_x), dim=-1)
    return target


def saliency_training_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = cfg.get("loss", {})
    area = batch.get("area_weights")
    final_pred = outputs["final"]
    temporal_pred = outputs["temporal"]
    cumulative_pred = outputs["cumulative"]

    final_gt = batch["final"]
    temporal_gt, cumulative_gt = match_temporal_target_count(
        batch["temporal"],
        prediction_count=int(temporal_pred.shape[1]),
        weights=area,
    )
    fixation = batch["fixation"]

    final_kl = weighted_kld(final_pred, final_gt, area)
    final_cc = weighted_cc(final_pred, final_gt, area)
    final_nss = nss(final_pred, fixation)
    final_point_log_prob = _point_log_prob_values(final_pred, fixation)
    point_nll_loss = -final_point_log_prob.mean()
    condition_contrastive_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    counterpart_final = batch.get("counterpart_final")
    if counterpart_final is not None and float(loss_cfg.get("condition_contrastive", 0.0)) > 0.0:
        own_kld = weighted_kld_values(final_pred, final_gt, area)
        counterpart_kld = weighted_kld_values(final_pred, counterpart_final, area)
        margin = float(loss_cfg.get("condition_contrastive_margin", 0.02))
        condition_contrastive_loss = F.relu(own_kld - counterpart_kld + margin).mean()
    order_rank_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    negative_final = outputs.get("negative_final")
    negative_finals = outputs.get("negative_finals")
    if negative_finals is not None:
        b, n = negative_finals.shape[:2]
        flat_negative = negative_finals.reshape(b * n, *negative_finals.shape[2:])
        flat_fixation = fixation.unsqueeze(1).expand(b, n, *fixation.shape[1:]).reshape(b * n, *fixation.shape[1:])
        negative_point_log_prob = _point_log_prob_values(flat_negative, flat_fixation).reshape(b, n)
        margin = float(loss_cfg.get("order_rank_margin", 0.1))
        order_rank_loss = F.relu(margin - (final_point_log_prob.unsqueeze(1) - negative_point_log_prob)).mean()
    elif negative_final is not None:
        negative_point_log_prob = _point_log_prob_values(negative_final, fixation)
        margin = float(loss_cfg.get("order_rank_margin", 0.1))
        order_rank_loss = F.relu(margin - (final_point_log_prob - negative_point_log_prob)).mean()
    motion_bin_ce_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    motion_bin_soft_ce_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    motion_logits = outputs.get("motion_logits")
    has_motion_loss = float(loss_cfg.get("motion_bin_ce", 0.0)) > 0.0 or float(loss_cfg.get("motion_bin_soft_ce", 0.0)) > 0.0
    if motion_logits is not None and has_motion_loss:
        bins_y = int(cfg.get("model", {}).get("motion_bins_y", 9))
        bins_x = int(cfg.get("model", {}).get("motion_bins_x", 9))
        if float(loss_cfg.get("motion_bin_ce", 0.0)) > 0.0:
            motion_targets = _motion_bin_targets(batch, cfg, bins_y=bins_y, bins_x=bins_x)
        else:
            motion_targets = None
        if motion_targets is not None and float(loss_cfg.get("motion_bin_ce", 0.0)) > 0.0:
            motion_bin_ce_loss = F.cross_entropy(motion_logits, motion_targets.to(device=motion_logits.device))
        if float(loss_cfg.get("motion_bin_soft_ce", 0.0)) > 0.0:
            motion_soft_targets = _motion_soft_targets(
                batch,
                cfg,
                bins_y=bins_y,
                bins_x=bins_x,
                sigma_bins=float(loss_cfg.get("motion_bin_soft_sigma", 1.0)),
            )
            if motion_soft_targets is not None:
                log_probs = F.log_softmax(motion_logits, dim=-1)
                motion_bin_soft_ce_loss = -(motion_soft_targets.to(device=motion_logits.device) * log_probs).sum(dim=-1).mean()
    final_loss = (
        float(loss_cfg.get("final_kl", 1.0)) * final_kl
        + float(loss_cfg.get("final_cc", 1.0)) * (1.0 - final_cc)
        - float(loss_cfg.get("final_nss", 0.1)) * final_nss
        + float(loss_cfg.get("point_nll", 0.0)) * point_nll_loss
        + float(loss_cfg.get("condition_contrastive", 0.0)) * condition_contrastive_loss
        + float(loss_cfg.get("order_rank", 0.0)) * order_rank_loss
        + float(loss_cfg.get("motion_bin_ce", 0.0)) * motion_bin_ce_loss
        + float(loss_cfg.get("motion_bin_soft_ce", 0.0)) * motion_bin_soft_ce_loss
    )

    temporal_bin_kld_loss = weighted_kld(temporal_pred, temporal_gt, area)
    cumulative_bin_kld_loss = weighted_kld(cumulative_pred, cumulative_gt, area)
    temporal_frame_sim_loss = 1.0 - weighted_sim(temporal_pred, temporal_gt, area)
    temporal_loss = temporal_bin_kld_loss + (1.0 - weighted_cc(temporal_pred, temporal_gt, area))
    temporal_frame_align_loss = temporal_bin_kld_loss + temporal_frame_sim_loss
    temporal_profile_kld_loss = temporal_profile_kld(temporal_pred, temporal_gt, area)
    cumulative_loss = cumulative_bin_kld_loss
    temporal_process_tv_loss = step_tv_loss(temporal_pred, temporal_gt, area)
    cumulative_process_tv_loss = step_tv_loss(cumulative_pred, cumulative_gt, area)
    static_pred = outputs.get("static")
    if static_pred is None:
        static_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
        static_kl = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
        static_cc = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    else:
        static_kl = weighted_kld(static_pred, final_gt, area)
        static_cc = weighted_cc(static_pred, final_gt, area)
        static_loss = static_kl + (1.0 - static_cc)
    accumulated_pred = outputs.get("accumulated_final")
    if accumulated_pred is None:
        accumulation_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
        accumulation_kl = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    else:
        accumulation_kl = weighted_kld(accumulated_pred, final_gt, area)
        accumulation_loss = accumulation_kl
    residual_logits = outputs.get("residual_logits")
    residual_loss = (
        residual_logits.abs().mean()
        if residual_logits is not None
        else torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    )
    teacher_residual_target_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    teacher_residual_topk_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    teacher_residual_peak_loss = torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device)
    has_residual_target = residual_logits is not None and static_pred is not None and (
        float(loss_cfg.get("teacher_residual_target", 0.0)) > 0.0
        or float(loss_cfg.get("teacher_residual_topk", 0.0)) > 0.0
        or float(loss_cfg.get("teacher_residual_peak", 0.0)) > 0.0
    )
    if has_residual_target:
        target_clip = float(loss_cfg.get("teacher_residual_target_clip", 4.0))
        residual_pred, residual_target = _teacher_residual_target(
            residual_logits,
            static_pred,
            temporal_gt,
            area,
            clip=target_clip,
        )
        teacher_residual_target_loss = F.smooth_l1_loss(residual_pred, residual_target)
        if float(loss_cfg.get("teacher_residual_topk", 0.0)) > 0.0:
            teacher_residual_topk_loss = _teacher_residual_topk_loss(
                residual_pred,
                residual_target,
                area,
                frac=float(loss_cfg.get("teacher_residual_topk_frac", 0.1)),
            )
        if float(loss_cfg.get("teacher_residual_peak", 0.0)) > 0.0:
            teacher_residual_peak_loss = _teacher_residual_peak_loss(residual_pred, residual_target, area)
    memory_loss = outputs.get("memory_overlap", torch.tensor(0.0, dtype=final_pred.dtype, device=final_pred.device))
    total = (
        final_loss
        + float(loss_cfg.get("static", 0.25)) * static_loss
        + float(loss_cfg.get("temporal", 0.5)) * temporal_loss
        + float(loss_cfg.get("temporal_frame_align", 0.0)) * temporal_frame_align_loss
        + float(loss_cfg.get("temporal_profile", 0.0)) * temporal_profile_kld_loss
        + float(loss_cfg.get("cumulative", 0.5)) * cumulative_loss
        + float(loss_cfg.get("temporal_bin_kld", 0.0)) * temporal_bin_kld_loss
        + float(loss_cfg.get("cumulative_bin_kld", 0.0)) * cumulative_bin_kld_loss
        + float(loss_cfg.get("temporal_process_tv", 0.0)) * temporal_process_tv_loss
        + float(loss_cfg.get("cumulative_process_tv", 0.0)) * cumulative_process_tv_loss
        + float(loss_cfg.get("accumulation", 0.1)) * accumulation_loss
        + float(loss_cfg.get("memory", 0.01)) * memory_loss
        + float(loss_cfg.get("residual", 0.0001)) * residual_loss
        + float(loss_cfg.get("teacher_residual_target", 0.0)) * teacher_residual_target_loss
        + float(loss_cfg.get("teacher_residual_topk", 0.0)) * teacher_residual_topk_loss
        + float(loss_cfg.get("teacher_residual_peak", 0.0)) * teacher_residual_peak_loss
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "final_kl": float(final_kl.detach().cpu()),
        "final_cc": float(final_cc.detach().cpu()),
        "final_nss": float(final_nss.detach().cpu()),
        "point_nll_loss": float(point_nll_loss.detach().cpu()),
        "condition_contrastive_loss": float(condition_contrastive_loss.detach().cpu()),
        "order_rank_loss": float(order_rank_loss.detach().cpu()),
        "motion_bin_ce_loss": float(motion_bin_ce_loss.detach().cpu()),
        "motion_bin_soft_ce_loss": float(motion_bin_soft_ce_loss.detach().cpu()),
        "static_kl": float(static_kl.detach().cpu()),
        "static_cc": float(static_cc.detach().cpu()),
        "accumulation_kl": float(accumulation_kl.detach().cpu()),
        "temporal_loss": float(temporal_loss.detach().cpu()),
        "temporal_frame_align_loss": float(temporal_frame_align_loss.detach().cpu()),
        "temporal_frame_sim_loss": float(temporal_frame_sim_loss.detach().cpu()),
        "temporal_profile_kld_loss": float(temporal_profile_kld_loss.detach().cpu()),
        "cumulative_loss": float(cumulative_loss.detach().cpu()),
        "temporal_bin_kld_loss": float(temporal_bin_kld_loss.detach().cpu()),
        "cumulative_bin_kld_loss": float(cumulative_bin_kld_loss.detach().cpu()),
        "temporal_process_tv_loss": float(temporal_process_tv_loss.detach().cpu()),
        "cumulative_process_tv_loss": float(cumulative_process_tv_loss.detach().cpu()),
        "memory_loss": float(memory_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "teacher_residual_target_loss": float(teacher_residual_target_loss.detach().cpu()),
        "teacher_residual_topk_loss": float(teacher_residual_topk_loss.detach().cpu()),
        "teacher_residual_peak_loss": float(teacher_residual_peak_loss.detach().cpu()),
    }
    return total, logs
