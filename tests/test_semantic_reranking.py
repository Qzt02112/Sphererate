import numpy as np
import torch
from torch import nn
from pathlib import Path
import yaml

from sphererate.losses import (
    semantic_log_ratio_loss,
    semantic_log_ratio_target,
    raw_dwell_frontier_rank_loss,
    raw_dwell_soft_topk_coverage_loss,
    raw_dwell_consensus_rank_loss,
    raw_dwell_tile_kl_loss,
    spherical_tile_mass,
    tile_mass_kl_loss,
    tile_pairwise_rank_loss,
)
from sphererate.evaluation.semantic_reranking import (
    deterministic_derangement,
    geometric_semantic_fusion,
)
from sphererate.models.early_horizon import TemplateTimeGatedResidualNet
from sphererate.models.spherical_dwell_rate import SphericalDwellRateNet


def test_geometric_fusion_alpha_zero_is_template_and_spherical_normalized() -> None:
    template = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    semantic = np.asarray([[4.0, 3.0], [2.0, 1.0]], dtype=np.float64)
    area = np.asarray([[0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    fused = geometric_semantic_fusion(template, semantic, area, alpha=0.0)
    expected = template / np.sum(template * area)
    np.testing.assert_allclose(fused, expected, atol=1e-12)
    np.testing.assert_allclose(np.sum(fused * area), 1.0, atol=1e-12)


def test_geometric_fusion_alpha_one_is_semantic() -> None:
    template = np.asarray([[0.9, 0.1]], dtype=np.float64)
    semantic = np.asarray([[0.2, 0.8]], dtype=np.float64)
    area = np.ones_like(template)
    fused = geometric_semantic_fusion(template, semantic, area, alpha=1.0)
    np.testing.assert_allclose(fused, semantic / semantic.sum(), atol=1e-12)


def test_deterministic_derangement_has_no_fixed_points_and_is_repeatable() -> None:
    first = deterministic_derangement(17, seed=20260714, key="semantic-s0")
    second = deterministic_derangement(17, seed=20260714, key="semantic-s0")
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(17))
    assert np.all(first != np.arange(17))


def test_spherical_tile_mass_is_normalized_and_exact_match_has_zero_kl() -> None:
    density = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
    area = torch.tensor([[[[0.5] * 4, [1.0] * 4, [1.0] * 4, [0.5] * 4]]])
    mass = spherical_tile_mass(density, area, tile_rows=2, tile_cols=2)
    torch.testing.assert_close(mass.sum(dim=(-2, -1)), torch.ones(1, 1))
    torch.testing.assert_close(
        tile_mass_kl_loss(density, density, area, tile_rows=2, tile_cols=2),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_pairwise_tile_rank_loss_rewards_correct_top_tile() -> None:
    target = torch.tensor([[[[8.0, 0.0], [0.0, 1.0]]]])
    correct = torch.tensor([[[[6.0, 1.0], [1.0, 2.0]]]], requires_grad=True)
    wrong = torch.tensor([[[[1.0, 6.0], [2.0, 1.0]]]], requires_grad=True)
    area = torch.ones(1, 1, 2, 2)
    correct_loss = tile_pairwise_rank_loss(
        correct, target, area, tile_rows=2, tile_cols=2, budgets_percent=(25.0,)
    )
    wrong_loss = tile_pairwise_rank_loss(
        wrong, target, area, tile_rows=2, tile_cols=2, budgets_percent=(25.0,)
    )
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert torch.isfinite(correct.grad).all()


def test_raw_dwell_tile_losses_use_unsmoothed_tile_target_and_backpropagate() -> None:
    target_mass = torch.tensor([[[[0.8, 0.0], [0.0, 0.2]]]])
    correct = torch.tensor([[[[8.0, 1.0], [1.0, 2.0]]]], requires_grad=True)
    wrong = torch.tensor([[[[1.0, 8.0], [2.0, 1.0]]]], requires_grad=True)
    area = torch.ones(1, 1, 2, 2)
    correct_kl = raw_dwell_tile_kl_loss(
        correct, target_mass, area, tile_rows=2, tile_cols=2
    )
    wrong_kl = raw_dwell_tile_kl_loss(
        wrong, target_mass, area, tile_rows=2, tile_cols=2
    )
    correct_rank = raw_dwell_frontier_rank_loss(
        correct,
        target_mass,
        area,
        tile_rows=2,
        tile_cols=2,
        budgets_percent=(25.0,),
        frontier_width=1,
    )
    wrong_rank = raw_dwell_frontier_rank_loss(
        wrong,
        target_mass,
        area,
        tile_rows=2,
        tile_cols=2,
        budgets_percent=(25.0,),
        frontier_width=1,
    )
    assert correct_kl < wrong_kl
    assert correct_rank < wrong_rank
    (correct_kl + correct_rank).backward()
    assert torch.isfinite(correct.grad).all()


def test_soft_topk_coverage_loss_rewards_correct_budget_set() -> None:
    target_mass = torch.tensor([[[[0.7, 0.2], [0.1, 0.0]]]])
    correct = torch.tensor([[[[7.0, 2.0], [1.0, 0.1]]]], requires_grad=True)
    wrong = torch.tensor([[[[0.1, 1.0], [2.0, 7.0]]]], requires_grad=True)
    area = torch.ones(1, 1, 2, 2)
    correct_loss = raw_dwell_soft_topk_coverage_loss(
        correct,
        target_mass,
        area,
        tile_rows=2,
        tile_cols=2,
        budgets_percent=(25.0, 50.0),
        epsilon=0.1,
    )
    wrong_loss = raw_dwell_soft_topk_coverage_loss(
        wrong,
        target_mass,
        area,
        tile_rows=2,
        tile_cols=2,
        budgets_percent=(25.0, 50.0),
        epsilon=0.1,
    )
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert torch.isfinite(correct.grad).all()


def test_spherical_dwell_rate_is_positive_additive_and_start_aware() -> None:
    model = SphericalDwellRateNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        semantic_pretrained=False,
    )
    image = torch.zeros(1, 8, 8, 16)
    template = torch.ones(1, 4, 8, 16)
    longitude = 2.0 * torch.pi * (torch.arange(16) + 0.5) / 16.0 - torch.pi
    latitude = torch.pi / 2.0 - torch.pi * (torch.arange(8) + 0.5) / 8.0
    lon, lat = torch.meshgrid(longitude, latitude, indexing="xy")
    cosine = torch.cos(lat) * torch.cos(lon)
    distance = torch.acos(cosine.clamp(-1.0, 1.0))
    sine_distance = torch.sin(distance)
    scale = torch.where(sine_distance.abs() > 1e-6, distance / sine_distance, torch.ones_like(distance))
    east = torch.cos(lat) * torch.sin(lon) * scale
    north = torch.sin(lat) * scale
    start_geometry = torch.stack([distance, cosine, east, north], dim=0).unsqueeze(0)
    output = model(image, template, start_geometry=start_geometry)
    changed_template = model(
        image,
        torch.rand_like(template).clamp_min(1e-4),
        start_geometry=start_geometry,
    )
    assert output["interval_rates"].shape == (1, 4, 8, 16)
    assert torch.all(output["interval_rates"] >= 0.0)
    increments = output["raw_cumulative_mass"][:, 1:] - output["raw_cumulative_mass"][:, :-1]
    assert torch.all(increments >= -1e-7)
    assert torch.all(output["semantic_alphas"][:, 1:] > output["semantic_alphas"][:, :-1])
    assert torch.all(output["reachability_betas"][:, 1:] < output["reachability_betas"][:, :-1])
    early_peak = output["interval_rates"][0, 0].argmax()
    peak_x = int(early_peak % 16)
    assert peak_x in {7, 8}
    torch.testing.assert_close(output["cumulative"], changed_template["cumulative"])


def test_anisotropic_reachability_preserves_ordered_spreads_and_offsets() -> None:
    model = SphericalDwellRateNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        semantic_pretrained=False,
        reachability_mode="anisotropic_tangent",
        reachability_beta_init=(14.7, 6.6, 0.736, 0.332),
        reachability_north_beta_init=(5.959, 4.612, 1.227, 0.684),
        reachability_offset_deg_init=((2.16, -2.54), (1.9, -2.94), (-1.3, -2.33), (-3.52, 0.92)),
    )
    assert torch.all(model.reachability_betas()[1:] < model.reachability_betas()[:-1])
    assert torch.all(
        model.reachability_north_betas()[1:] < model.reachability_north_betas()[:-1]
    )
    torch.testing.assert_close(
        torch.rad2deg(model.reachability_offsets()),
        torch.tensor([[2.16, -2.54], [1.9, -2.94], [-1.3, -2.33], [-3.52, 0.92]]),
        atol=1e-4,
        rtol=0.0,
    )


def test_consensus_rank_loss_rewards_consensus_order() -> None:
    consensus = torch.tensor([[[[[1.0, 0.7], [0.2, 0.0]]]]])
    correct = torch.tensor([[[[8.0, 4.0], [2.0, 1.0]]]], requires_grad=True)
    wrong = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]], requires_grad=True)
    area = torch.ones(1, 1, 2, 2)
    correct_loss = raw_dwell_consensus_rank_loss(
        correct, consensus, area, tile_rows=2, tile_cols=2
    )
    wrong_loss = raw_dwell_consensus_rank_loss(
        wrong, consensus, area, tile_rows=2, tile_cols=2
    )
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert torch.isfinite(correct.grad).all()


def test_semantic_log_ratio_loss_is_zero_at_centered_target() -> None:
    template = torch.tensor([[[[4.0, 1.0], [1.0, 2.0]]]])
    target = torch.tensor([[[[1.0, 5.0], [1.0, 1.0]]]])
    area = torch.ones(1, 1, 2, 2)
    ratio = semantic_log_ratio_target(template, target, area, clip=4.0)
    loss = semantic_log_ratio_loss(ratio, template, target, area, clip=4.0)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0.0)


def test_bounded_residual_fusion_does_not_apply_legacy_early_gate() -> None:
    model = TemplateTimeGatedResidualNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        semantic_pretrained=False,
        residual_fusion="bounded_residual",
        residual_bound=4.0,
    )

    class ConstantFeatures(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                image.shape[0], 8, image.shape[-2], image.shape[-1], dtype=image.dtype
            )

    model.features = ConstantFeatures()
    with torch.no_grad():
        model.residual_out.conv.weight.zero_()
        model.residual_out.conv.bias.fill_(2.0)
    image = torch.zeros(1, 8, 16, 32)
    template = torch.ones(1, 4, 16, 32)
    output = model(image, template)
    expected = torch.full_like(
        output["effective_residual_logits"], 4.0 * torch.tanh(torch.tensor(0.5))
    )
    torch.testing.assert_close(output["effective_residual_logits"], expected)
    assert output["gates"][0, 0] < 0.1
    assert output["effective_residual_logits"][0, 0, 0, 0] > 1.0


def test_semantic_anchor_is_shared_and_mixed_by_monotonic_alphas() -> None:
    model = TemplateTimeGatedResidualNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        semantic_pretrained=False,
        residual_fusion="semantic_anchor",
    )

    class FirstChannelFeatures(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            features = torch.zeros(
                image.shape[0], 8, image.shape[-2], image.shape[-1], dtype=image.dtype
            )
            features[:, 0] = image[:, 0]
            return features

    model.features = FirstChannelFeatures()
    model.semantic_block = nn.Identity()
    with torch.no_grad():
        model.semantic_out.conv.weight.zero_()
        model.semantic_out.conv.weight[0, 0, 0, 0] = 1.0
        model.semantic_out.conv.bias.zero_()
    image = torch.zeros(1, 8, 8, 16)
    image[:, 0, :, 4:8] = 2.0
    template = torch.ones(1, 4, 8, 16)
    output = model(image, template)
    alphas = output["semantic_alphas"][0]
    assert torch.all(alphas[1:] > alphas[:-1])
    assert 0.0 < alphas[0] and alphas[-1] < 1.0
    assert output["semantic_map"].shape == (1, 1, 8, 16)
    raw_ratio = output["residual_logits"]
    for index in range(4):
        torch.testing.assert_close(
            output["effective_residual_logits"][:, index],
            alphas[index] * raw_ratio[:, index],
        )


def test_v11a_config_uses_raw_dwell_all_budgets_and_keeps_outer_test_forbidden() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "v11a_raw_dwell_s2.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    train = config["train"]
    assert train["coverage_selection_budgets"] == [5.0, 10.0, 20.0]
    assert train["raw_tile_kl_weight"] > 0
    assert train["raw_frontier_rank_weight"] > 0
    assert train["tile_rank_weight"] == 0
    assert len(train["tile_horizon_weights"]) == 4
    assert config["protocol"]["selection_split"] == "val"
    assert set(config["protocol"]["forbidden_for_selection"]) == {"gate", "test", "sitzmann"}


def test_semantic_anchor_tile_residual_starts_as_exact_anchor_and_is_periodic() -> None:
    model = TemplateTimeGatedResidualNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        semantic_pretrained=False,
        residual_fusion="semantic_anchor_tile_residual",
        tile_rows=2,
        tile_cols=4,
        tile_hidden_channels=4,
    )
    image = torch.randn(1, 8, 8, 16)
    template = torch.ones(1, 4, 8, 16)
    output = model(image, template)
    assert output["tile_residual_logits"].shape == (1, 4, 8, 16)
    torch.testing.assert_close(
        output["tile_residual_logits"], torch.zeros_like(output["tile_residual_logits"])
    )
    for index in range(4):
        expected = model._spherical_softmax(
            template[:, index : index + 1].log()
            + output["semantic_alphas"][0, index] * output["residual_logits"][:, index : index + 1]
        )
        torch.testing.assert_close(output["cumulative"][:, index : index + 1], expected)


def test_semantic_anchor_pixel_residual_starts_as_exact_anchor() -> None:
    model = TemplateTimeGatedResidualNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        semantic_pretrained=False,
        residual_fusion="semantic_anchor_pixel_residual",
        pixel_hidden_channels=4,
    )
    image = torch.randn(1, 8, 8, 16)
    template = torch.ones(1, 4, 8, 16)
    output = model(image, template)
    assert output["pixel_residual_logits"].shape == (1, 4, 8, 16)
    torch.testing.assert_close(
        output["pixel_residual_logits"], torch.zeros_like(output["pixel_residual_logits"])
    )


def test_temporal_semantic_anchor_matches_shared_anchor_after_copied_initialization() -> None:
    model = TemplateTimeGatedResidualNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        semantic_pretrained=False,
        residual_fusion="temporal_semantic_anchor",
    )
    with torch.no_grad():
        model.temporal_semantic_out.conv.weight.copy_(
            model.semantic_out.conv.weight.repeat(4, 1, 1, 1)
        )
        model.temporal_semantic_out.conv.bias.copy_(model.semantic_out.conv.bias.repeat(4))
    image = torch.randn(1, 8, 8, 16)
    template = torch.rand(1, 4, 8, 16).clamp_min(1e-3)
    output = model(image, template)
    for index in range(4):
        torch.testing.assert_close(
            output["temporal_semantic_maps"][:, index : index + 1], output["semantic_map"]
        )
