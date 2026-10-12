from __future__ import annotations

import torch

from sphererate.losses import (
    match_temporal_target_count,
    saliency_training_loss,
    step_tv_loss,
    temporal_profile_kld,
    weighted_cc,
    weighted_sim,
    weighted_step_tv,
)


def test_match_temporal_target_count_coarsens_adjacent_six_bins_to_three() -> None:
    area = torch.ones(1, 1, 2, 6)
    temporal = torch.zeros(1, 6, 2, 6)
    for index in range(6):
        temporal[:, index, index // 3, index] = 1.0

    coarse, cumulative = match_temporal_target_count(temporal, prediction_count=3, weights=area)

    assert coarse.shape == (1, 3, 2, 6)
    assert cumulative.shape == coarse.shape
    assert torch.allclose(coarse.sum(dim=(-2, -1)), torch.ones(1, 3))
    assert coarse[0, 0, 0, 0] == 0.5
    assert coarse[0, 0, 0, 1] == 0.5
    assert coarse[0, 1, 0, 2] == 0.5
    assert coarse[0, 1, 1, 3] == 0.5
    assert torch.allclose(cumulative[:, -1].sum(dim=(-2, -1)), torch.ones(1))


def test_match_temporal_target_count_preserves_matching_count_and_rejects_incompatible_count() -> None:
    area = torch.ones(1, 1, 2, 3)
    temporal = torch.rand(1, 3, 2, 3)
    matched, cumulative = match_temporal_target_count(temporal, prediction_count=3, weights=area)
    assert torch.allclose(matched, temporal)
    assert cumulative.shape == temporal.shape

    try:
        match_temporal_target_count(torch.rand(1, 5, 2, 3), prediction_count=3, weights=area)
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("incompatible temporal target count was accepted")


def test_saliency_loss_accepts_three_latent_predictions_against_six_stored_bins() -> None:
    area = torch.ones(1, 1, 4, 4)
    temporal_gt = torch.rand(1, 6, 4, 4)
    final_gt = torch.rand(1, 1, 4, 4)
    coarse, cumulative = match_temporal_target_count(temporal_gt, prediction_count=3, weights=area)
    outputs = {
        "final": final_gt.clone().requires_grad_(True),
        "temporal": coarse.clone().requires_grad_(True),
        "cumulative": cumulative.clone().requires_grad_(True),
    }
    batch = {
        "final": final_gt,
        "temporal": temporal_gt,
        "cumulative": torch.rand(1, 6, 4, 4),
        "fixation": (final_gt > final_gt.mean()).float(),
        "area_weights": area,
    }
    cfg = {"loss": {"final_kl": 1.0, "final_cc": 1.0, "temporal": 1.0, "cumulative": 1.0}}

    loss, logs = saliency_training_loss(outputs, batch, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert logs["temporal_bin_kld_loss"] < 1e-5
    assert logs["cumulative_bin_kld_loss"] < 1e-5


def test_weighted_cc_self_correlation_and_gradient() -> None:
    pred = torch.rand(2, 1, 32, 64, dtype=torch.float32, requires_grad=True)
    gt = pred.detach().clone()
    latitude_weight = torch.cos(torch.linspace(-torch.pi / 2, torch.pi / 2, 32)).view(1, 1, 32, 1)
    area = latitude_weight.expand(2, 1, 32, 64)

    unweighted_score = weighted_cc(pred, gt)
    weighted_score = weighted_cc(pred, gt, area)
    weighted_score.backward()

    assert torch.isclose(unweighted_score.detach(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(weighted_score.detach(), torch.tensor(1.0), atol=1e-5)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_weighted_cc_constant_map_returns_zero() -> None:
    constant = torch.ones(1, 1, 8, 16)
    varying = torch.arange(128, dtype=torch.float32).reshape(1, 1, 8, 16)

    assert weighted_cc(constant, varying).item() == 0.0


def test_weighted_step_tv_detects_static_vs_moving_sequence():
    area = torch.ones(1, 1, 4, 4)
    moving = torch.zeros(1, 2, 4, 4)
    moving[:, 0, 1, 1] = 1.0
    moving[:, 1, 2, 2] = 1.0
    static = moving[:, :1].repeat(1, 2, 1, 1)

    assert weighted_step_tv(moving, area) > 0.0
    assert weighted_step_tv(static, area) == 0.0
    assert step_tv_loss(static, moving, area) > 0.0


def test_weighted_sim_rewards_framewise_overlap():
    area = torch.ones(1, 1, 4, 4)
    gt = torch.zeros(1, 2, 4, 4)
    gt[:, 0, 1, 1] = 1.0
    gt[:, 1, 2, 2] = 1.0
    aligned = gt.clone()
    swapped = gt.flip(1)

    assert weighted_sim(aligned, gt, area) > weighted_sim(swapped, gt, area)


def test_temporal_profile_kld_penalizes_wrong_time_at_same_location():
    area = torch.ones(1, 1, 4, 4)
    gt = torch.full((1, 3, 4, 4), 1e-4)
    gt[:, 0, 1, 1] = 1.0
    gt[:, 2, 2, 2] = 1.0
    aligned = gt.clone()
    swapped = gt.flip(1)

    assert temporal_profile_kld(aligned, gt, area) < temporal_profile_kld(swapped, gt, area)


def test_saliency_loss_logs_process_terms_when_enabled():
    area = torch.ones(1, 1, 4, 4)
    temporal_gt = torch.zeros(1, 2, 4, 4)
    temporal_gt[:, 0, 1, 1] = 1.0
    temporal_gt[:, 1, 2, 2] = 1.0
    cumulative_gt = torch.stack(
        [temporal_gt[:, 0], temporal_gt[:, 0] + temporal_gt[:, 1]],
        dim=1,
    )
    final_gt = temporal_gt[:, 1:2]
    fixation = final_gt.clone()

    temporal_pred = temporal_gt[:, :1].repeat(1, 2, 1, 1).clone().requires_grad_(True)
    cumulative_pred = cumulative_gt[:, :1].repeat(1, 2, 1, 1).clone().requires_grad_(True)
    final_pred = final_gt.clone().requires_grad_(True)

    outputs = {
        "final": final_pred,
        "temporal": temporal_pred,
        "cumulative": cumulative_pred,
    }
    batch = {
        "final": final_gt,
        "temporal": temporal_gt,
        "cumulative": cumulative_gt,
        "fixation": fixation,
        "area_weights": area,
    }
    loss, logs = saliency_training_loss(
        outputs,
        batch,
        {
            "loss": {
                "temporal_process_tv": 1.0,
                "cumulative_process_tv": 1.0,
            }
        },
    )

    assert loss.requires_grad
    assert logs["temporal_process_tv_loss"] > 0.0
    assert logs["cumulative_process_tv_loss"] > 0.0
    assert "temporal_bin_kld_loss" in logs
    assert "temporal_frame_align_loss" in logs
    assert "temporal_profile_kld_loss" in logs
    assert "cumulative_bin_kld_loss" in logs


def test_temporal_frame_align_loss_prefers_frame_aligned_distribution():
    area = torch.ones(1, 1, 4, 4)
    temporal_gt = torch.zeros(1, 2, 4, 4)
    temporal_gt[:, 0, 1, 1] = 1.0
    temporal_gt[:, 1, 2, 2] = 1.0
    cumulative_gt = torch.stack(
        [temporal_gt[:, 0], temporal_gt[:, 0] + temporal_gt[:, 1]],
        dim=1,
    )
    final_gt = temporal_gt[:, 1:2]
    fixation = final_gt.clone()
    batch = {
        "final": final_gt,
        "temporal": temporal_gt,
        "cumulative": cumulative_gt,
        "fixation": fixation,
        "area_weights": area,
    }
    cfg = {
        "loss": {
            "final_kl": 0.0,
            "final_cc": 0.0,
            "final_nss": 0.0,
            "temporal": 0.0,
            "temporal_frame_align": 1.0,
            "cumulative": 0.0,
            "accumulation": 0.0,
            "memory": 0.0,
            "residual": 0.0,
        }
    }

    def make_outputs(temporal: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "final": final_gt.clone().requires_grad_(True),
            "temporal": temporal.clone().requires_grad_(True),
            "cumulative": cumulative_gt.clone().requires_grad_(True),
        }

    aligned_loss, aligned_logs = saliency_training_loss(make_outputs(temporal_gt), batch, cfg)
    swapped_loss, swapped_logs = saliency_training_loss(make_outputs(temporal_gt.flip(1)), batch, cfg)

    assert aligned_loss.requires_grad
    assert aligned_logs["temporal_frame_align_loss"] < swapped_logs["temporal_frame_align_loss"]
    assert aligned_loss < swapped_loss


def test_nextfix_point_nll_and_order_rank_prefer_target_probability():
    area = torch.ones(1, 1, 4, 4)
    gt = torch.zeros(1, 1, 4, 4)
    gt[:, :, 1, 1] = 1.0
    fixation = gt.clone()
    temporal = gt.clone()
    cumulative = gt.clone()
    good = torch.full((1, 1, 4, 4), 1e-3)
    good[:, :, 1, 1] = 1.0
    good = good / good.sum()
    bad = torch.full((1, 1, 4, 4), 1e-3)
    bad[:, :, 0, 0] = 1.0
    bad = bad / bad.sum()
    batch = {
        "final": gt,
        "temporal": temporal,
        "cumulative": cumulative,
        "fixation": fixation,
        "area_weights": area,
    }
    cfg = {
        "loss": {
            "final_kl": 0.0,
            "final_cc": 0.0,
            "final_nss": 0.0,
            "static": 0.0,
            "temporal": 0.0,
            "cumulative": 0.0,
            "accumulation": 0.0,
            "memory": 0.0,
            "residual": 0.0,
            "point_nll": 1.0,
            "order_rank": 1.0,
            "order_rank_margin": 0.1,
        }
    }

    good_outputs = {
        "final": good.clone().requires_grad_(True),
        "temporal": temporal.clone().requires_grad_(True),
        "cumulative": cumulative.clone().requires_grad_(True),
        "negative_final": bad.clone().requires_grad_(True),
    }
    bad_outputs = {
        "final": bad.clone().requires_grad_(True),
        "temporal": temporal.clone().requires_grad_(True),
        "cumulative": cumulative.clone().requires_grad_(True),
        "negative_final": good.clone().requires_grad_(True),
    }
    good_loss, good_logs = saliency_training_loss(good_outputs, batch, cfg)
    bad_loss, bad_logs = saliency_training_loss(bad_outputs, batch, cfg)

    assert good_loss.requires_grad
    assert good_logs["point_nll_loss"] < bad_logs["point_nll_loss"]
    assert good_logs["order_rank_loss"] < bad_logs["order_rank_loss"]
    assert good_loss < bad_loss


def test_nextfix_order_rank_supports_multiple_negatives():
    area = torch.ones(1, 1, 4, 4)
    gt = torch.zeros(1, 1, 4, 4)
    gt[:, :, 1, 1] = 1.0
    fixation = gt.clone()
    good = torch.full((1, 1, 4, 4), 1e-3)
    good[:, :, 1, 1] = 1.0
    good = good / good.sum()
    bad = torch.full((1, 1, 4, 4), 1e-3)
    bad[:, :, 0, 0] = 1.0
    bad = bad / bad.sum()
    worse = torch.full((1, 1, 4, 4), 1e-3)
    worse[:, :, 3, 3] = 1.0
    worse = worse / worse.sum()
    batch = {
        "final": gt,
        "temporal": gt,
        "cumulative": gt,
        "fixation": fixation,
        "area_weights": area,
    }
    cfg = {
        "loss": {
            "final_kl": 0.0,
            "final_cc": 0.0,
            "final_nss": 0.0,
            "static": 0.0,
            "temporal": 0.0,
            "cumulative": 0.0,
            "accumulation": 0.0,
            "memory": 0.0,
            "residual": 0.0,
            "order_rank": 1.0,
            "order_rank_margin": 0.1,
        }
    }

    good_outputs = {
        "final": good.clone().requires_grad_(True),
        "temporal": gt.clone().requires_grad_(True),
        "cumulative": gt.clone().requires_grad_(True),
        "negative_finals": torch.stack([bad, worse], dim=1).requires_grad_(True),
    }
    bad_outputs = {
        "final": bad.clone().requires_grad_(True),
        "temporal": gt.clone().requires_grad_(True),
        "cumulative": gt.clone().requires_grad_(True),
        "negative_finals": torch.stack([good, worse], dim=1).requires_grad_(True),
    }
    good_loss, good_logs = saliency_training_loss(good_outputs, batch, cfg)
    bad_loss, bad_logs = saliency_training_loss(bad_outputs, batch, cfg)

    assert good_loss.requires_grad
    assert good_logs["order_rank_loss"] < bad_logs["order_rank_loss"]
    assert good_loss < bad_loss


def test_nextfix_motion_bin_ce_prefers_correct_displacement_bin():
    area = torch.ones(1, 1, 5, 8)
    fixation = torch.zeros(1, 1, 5, 8)
    fixation[:, :, 3, 5] = 1.0
    current = torch.zeros(1, 1, 5, 8)
    current[:, :, 2, 2] = 1.0
    image = torch.zeros(1, 12, 5, 8)
    image[:, 10:11] = current
    gt = fixation.clone()
    batch = {
        "image": image,
        "final": gt,
        "temporal": gt,
        "cumulative": gt,
        "fixation": fixation,
        "area_weights": area,
    }
    cfg = {
        "train": {"scanpath_prefix_length": 3, "scanpath_prefix_encoding": "sequence"},
        "model": {"prefix_length": 3, "motion_bins_y": 5, "motion_bins_x": 5},
        "loss": {
            "final_kl": 0.0,
            "final_cc": 0.0,
            "final_nss": 0.0,
            "static": 0.0,
            "temporal": 0.0,
            "cumulative": 0.0,
            "accumulation": 0.0,
            "memory": 0.0,
            "residual": 0.0,
            "motion_bin_ce": 1.0,
        },
    }
    correct_logits = torch.zeros(1, 25, requires_grad=True)
    correct_logits.data[:, 19] = 4.0
    wrong_logits = torch.zeros(1, 25, requires_grad=True)
    wrong_logits.data[:, 0] = 4.0

    def make_outputs(logits: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "final": gt.clone().requires_grad_(True),
            "temporal": gt.clone().requires_grad_(True),
            "cumulative": gt.clone().requires_grad_(True),
            "motion_logits": logits,
        }

    correct_loss, correct_logs = saliency_training_loss(make_outputs(correct_logits), batch, cfg)
    wrong_loss, wrong_logs = saliency_training_loss(make_outputs(wrong_logits), batch, cfg)

    assert correct_loss.requires_grad
    assert correct_logs["motion_bin_ce_loss"] < wrong_logs["motion_bin_ce_loss"]
    assert correct_loss < wrong_loss


def test_nextfix_motion_soft_ce_prefers_nearby_displacement_bins():
    area = torch.ones(1, 1, 5, 8)
    fixation = torch.zeros(1, 1, 5, 8)
    fixation[:, :, 3, 5] = 1.0
    current = torch.zeros(1, 1, 5, 8)
    current[:, :, 2, 2] = 1.0
    image = torch.zeros(1, 12, 5, 8)
    image[:, 10:11] = current
    gt = fixation.clone()
    batch = {
        "image": image,
        "final": gt,
        "temporal": gt,
        "cumulative": gt,
        "fixation": fixation,
        "area_weights": area,
    }
    cfg = {
        "train": {"scanpath_prefix_length": 3, "scanpath_prefix_encoding": "sequence"},
        "model": {"prefix_length": 3, "motion_bins_y": 5, "motion_bins_x": 5},
        "loss": {
            "final_kl": 0.0,
            "final_cc": 0.0,
            "final_nss": 0.0,
            "static": 0.0,
            "temporal": 0.0,
            "cumulative": 0.0,
            "accumulation": 0.0,
            "memory": 0.0,
            "residual": 0.0,
            "motion_bin_ce": 0.0,
            "motion_bin_soft_ce": 1.0,
            "motion_bin_soft_sigma": 1.0,
        },
    }
    near_logits = torch.zeros(1, 25, requires_grad=True)
    near_logits.data[:, 19] = 4.0
    far_logits = torch.zeros(1, 25, requires_grad=True)
    far_logits.data[:, 0] = 4.0

    def make_outputs(logits: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "final": gt.clone().requires_grad_(True),
            "temporal": gt.clone().requires_grad_(True),
            "cumulative": gt.clone().requires_grad_(True),
            "motion_logits": logits,
        }

    near_loss, near_logs = saliency_training_loss(make_outputs(near_logits), batch, cfg)
    far_loss, far_logs = saliency_training_loss(make_outputs(far_logits), batch, cfg)

    assert near_loss.requires_grad
    assert near_logs["motion_bin_soft_ce_loss"] < far_logs["motion_bin_soft_ce_loss"]
    assert near_loss < far_loss


def test_temporal_profile_loss_prefers_correct_time_profile():
    area = torch.ones(1, 1, 4, 4)
    temporal_gt = torch.full((1, 3, 4, 4), 1e-4)
    temporal_gt[:, 0, 1, 1] = 1.0
    temporal_gt[:, 1, 2, 2] = 1.0
    temporal_gt[:, 2, 1, 2] = 1.0
    cumulative_gt = temporal_gt.cumsum(dim=1)
    final_gt = temporal_gt.sum(dim=1, keepdim=True)
    fixation = final_gt.clone()
    batch = {
        "final": final_gt,
        "temporal": temporal_gt,
        "cumulative": cumulative_gt,
        "fixation": fixation,
        "area_weights": area,
    }
    cfg = {
        "loss": {
            "final_kl": 0.0,
            "final_cc": 0.0,
            "final_nss": 0.0,
            "temporal": 0.0,
            "temporal_frame_align": 0.0,
            "temporal_profile": 1.0,
            "cumulative": 0.0,
            "accumulation": 0.0,
            "memory": 0.0,
            "residual": 0.0,
        }
    }

    def make_outputs(temporal: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "final": final_gt.clone().requires_grad_(True),
            "temporal": temporal.clone().requires_grad_(True),
            "cumulative": cumulative_gt.clone().requires_grad_(True),
        }

    aligned_loss, aligned_logs = saliency_training_loss(make_outputs(temporal_gt), batch, cfg)
    swapped_loss, swapped_logs = saliency_training_loss(make_outputs(temporal_gt.flip(1)), batch, cfg)

    assert aligned_loss.requires_grad
    assert aligned_logs["temporal_profile_kld_loss"] < swapped_logs["temporal_profile_kld_loss"]
    assert aligned_loss < swapped_loss


def test_teacher_residual_localization_losses_prefer_aligned_residual():
    area = torch.ones(1, 1, 4, 4)
    static = torch.ones(1, 1, 4, 4) / 16.0
    temporal_gt = torch.zeros(1, 2, 4, 4)
    temporal_gt[:, 0, 1, 1] = 1.0
    temporal_gt[:, 1, 2, 2] = 1.0
    cumulative_gt = torch.stack(
        [temporal_gt[:, 0], temporal_gt[:, 0] + temporal_gt[:, 1]],
        dim=1,
    )
    final_gt = temporal_gt[:, 1:2]
    fixation = final_gt.clone()
    aligned_residual = torch.zeros(1, 2, 4, 4, requires_grad=True)
    aligned_residual.data[:, 0, 1, 1] = 4.0
    aligned_residual.data[:, 1, 2, 2] = 4.0
    wrong_residual = torch.zeros(1, 2, 4, 4, requires_grad=True)
    wrong_residual.data[:, 0, 0, 0] = 4.0
    wrong_residual.data[:, 1, 0, 0] = 4.0

    batch = {
        "final": final_gt,
        "temporal": temporal_gt,
        "cumulative": cumulative_gt,
        "fixation": fixation,
        "area_weights": area,
    }
    cfg = {
        "loss": {
            "final_kl": 0.0,
            "final_cc": 0.0,
            "final_nss": 0.0,
            "static": 0.0,
            "temporal": 0.0,
            "cumulative": 0.0,
            "accumulation": 0.0,
            "memory": 0.0,
            "residual": 0.0,
            "teacher_residual_topk": 1.0,
            "teacher_residual_topk_frac": 0.1,
            "teacher_residual_peak": 1.0,
        }
    }

    def make_outputs(residual: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "final": final_gt.clone().requires_grad_(True),
            "temporal": temporal_gt.clone().requires_grad_(True),
            "cumulative": cumulative_gt.clone().requires_grad_(True),
            "static": static,
            "residual_logits": residual,
        }

    aligned_loss, aligned_logs = saliency_training_loss(make_outputs(aligned_residual), batch, cfg)
    wrong_loss, wrong_logs = saliency_training_loss(make_outputs(wrong_residual), batch, cfg)

    assert aligned_loss.requires_grad
    assert aligned_logs["teacher_residual_topk_loss"] > 0.0
    assert aligned_logs["teacher_residual_peak_loss"] >= 0.0
    assert aligned_loss < wrong_loss
