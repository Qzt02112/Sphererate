from __future__ import annotations

import torch

from sphererate.models.registry import build_model
from sphererate.utils import spatial_softmax


def test_model_forward_shapes() -> None:
    cfg = {
        "data": {"num_bins": 3},
        "model": {
            "in_channels": 8,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 3,
            "semantic_pretrained": False,
            "semantic_layer": "layer2",
        },
    }
    x = torch.randn(2, 8, 32, 64)
    for name in [
        "static",
        "semantic_static",
        "parallel",
        "directsum",
        "tempsal_mixer",
        "semantic_tempsal_temporal",
        "semantic_tempsal_fusion",
        "residual_mixer",
        "dytsaldmm",
        "semantic_dytsaldmm",
        "semantic_teacher_residual",
    ]:
        model = build_model(name, cfg)
        out = model(x)
        assert out["final"].shape == (2, 1, 32, 64)
        assert out["temporal"].shape == (2, 3, 32, 64)
        assert out["cumulative"].shape == (2, 3, 32, 64)
        assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)


def test_lightweight_start_residual_is_bounded_small_and_reduces_to_teacher() -> None:
    cfg = {
        "data": {"num_bins": 3},
        "model": {
            "in_channels": 13,
            "static_in_channels": 8,
            "base_channels": 8,
            "hidden_channels": 16,
            "residual_hidden_channels": 4,
            "num_bins": 3,
            "semantic_pretrained": False,
            "semantic_layer": "layer2",
            "teacher_residual_scale": 1.0,
            "teacher_residual_init_gate": -1.0,
        },
    }
    model = build_model("start_conditioned_light_residual", cfg)
    x = torch.randn(2, 13, 32, 64)
    out = model(x)

    assert out["final"].shape == (2, 1, 32, 64)
    assert out["temporal"].shape == (2, 3, 32, 64)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert out["residual_logits"].abs().max() <= out["residual_gate_value"] + 1e-6
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    teacher_total = sum(parameter.numel() for parameter in model.teacher.parameters())
    assert trainable < 0.1 * teacher_total

    with torch.no_grad():
        model.residual_gate.fill_(-100.0)
        disabled = model(x)
    assert torch.allclose(disabled["final"], disabled["teacher"], atol=1e-6)


def _p1_matched_cfg() -> dict:
    return {
        "data": {"num_bins": 3},
        "model": {
            "in_channels": 13,
            "static_in_channels": 8,
            "base_channels": 8,
            "hidden_channels": 16,
            "residual_hidden_channels": 4,
            "num_bins": 3,
            "semantic_pretrained": False,
            "semantic_layer": "layer2",
            "teacher_residual_scale": 1.0,
            "teacher_residual_init_gate": -1.0,
        },
    }


def test_p1_matched_conditioning_models_are_normalized_and_small() -> None:
    x = torch.randn(2, 13, 32, 64)
    for name in ("learned_start_only", "direct_start_conditioned", "film_start_conditioned"):
        model = build_model(name, _p1_matched_cfg())
        out = model(x)
        assert out["final"].shape == (2, 1, 32, 64)
        assert out["temporal"].shape == (2, 3, 32, 64)
        assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if hasattr(model, "teacher"):
            teacher_total = sum(parameter.numel() for parameter in model.teacher.parameters())
            assert trainable < 0.1 * teacher_total
            assert all(not parameter.requires_grad for parameter in model.teacher.parameters())


def test_start_only_ignores_rgb_and_matched_models_use_condition() -> None:
    torch.manual_seed(7)
    base = torch.randn(1, 13, 32, 64)
    changed_rgb = base.clone()
    changed_rgb[:, :8] = torch.randn_like(changed_rgb[:, :8])
    changed_condition = base.clone()
    changed_condition[:, 8:] = torch.randn_like(changed_condition[:, 8:])

    start_only = build_model("learned_start_only", _p1_matched_cfg()).eval()
    assert torch.allclose(start_only(base)["final"], start_only(changed_rgb)["final"], atol=1e-7)
    assert not torch.allclose(start_only(base)["final"], start_only(changed_condition)["final"])

    direct = build_model("direct_start_conditioned", _p1_matched_cfg()).eval()
    assert not torch.allclose(direct(base)["final"], direct(changed_condition)["final"])

    film = build_model("film_start_conditioned", _p1_matched_cfg()).eval()
    with torch.no_grad():
        film.condition_affine.conv.weight.normal_(0.0, 0.05)
    assert not torch.allclose(film(base)["final"], film(changed_condition)["final"])


def test_film_residual_off_recovers_teacher() -> None:
    model = build_model("film_start_conditioned", _p1_matched_cfg())
    x = torch.randn(2, 13, 32, 64)
    with torch.no_grad():
        model.residual_gate.fill_(-100.0)
        out = model(x)
    assert torch.allclose(out["final"], out["teacher"], atol=1e-6)


def _gate_c_cfg(aggregation: str = "density_marginal") -> dict:
    return {
        "data": {"num_bins": 6},
        "model": {
            "in_channels": 8,
            "base_channels": 8,
            "hidden_channels": 16,
            "latent_num_conditions": 3,
            "latent_code_channels": 4,
            "latent_conditioning": "shared_multihead",
            "residual_hidden_channels": 4,
            "num_bins": 3,
            "semantic_pretrained": False,
            "semantic_layer": "layer2",
            "teacher_residual_scale": 1.0,
            "teacher_residual_init_gate": -1.0,
            "latent_aggregation": aggregation,
        },
    }


def test_image_only_latent_marginal_is_normalized_small_and_reduces_to_teacher() -> None:
    model = build_model("image_only_latent_marginal", _gate_c_cfg())
    x = torch.randn(2, 8, 32, 64)
    out = model(x)

    assert out["final"].shape == (2, 1, 32, 64)
    assert out["temporal"].shape == (2, 3, 32, 64)
    assert out["condition_maps"].shape == (2, 3, 32, 64)
    assert out["residual_logits"].shape == (2, 3, 32, 64)
    assert out["weights"].shape == (2, 3)
    assert model.conditioning == "shared_multihead"
    assert model.residual_head[-1].conv.out_channels == 3
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["condition_maps"].sum(dim=(-2, -1)), torch.ones(2, 3), atol=1e-4)
    assert torch.allclose(out["weights"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert out["residual_logits"].abs().max() <= out["residual_gate_value"] + 1e-6
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    teacher_total = sum(parameter.numel() for parameter in model.teacher.parameters())
    assert trainable < 0.1 * teacher_total

    expected = (out["condition_maps"] * out["weights"][:, :, None, None]).sum(dim=1, keepdim=True)
    assert torch.allclose(out["final"], expected, atol=1e-6)

    with torch.no_grad():
        model.residual_gate.fill_(-100.0)
        disabled = model(x)
    assert torch.allclose(disabled["final"], disabled["teacher"], atol=1e-6)


def test_image_only_temporal_aux_uses_matched_logit_pooling() -> None:
    model = build_model("image_only_temporal_aux", _gate_c_cfg("logit_pool"))
    x = torch.randn(2, 8, 32, 64)
    out = model(x)

    expected_logits = torch.log(out["teacher"].clamp_min(1e-7)) + (
        out["residual_logits"] * out["weights"][:, :, None, None]
    ).sum(dim=1, keepdim=True)
    expected = spatial_softmax(expected_logits)
    assert torch.allclose(out["final"], expected, atol=1e-6)
    assert model.aggregation == "logit_pool"


def test_image_only_latent_model_rejects_non_image_inputs_and_unknown_aggregation() -> None:
    model = build_model("image_only_latent_marginal", _gate_c_cfg())
    try:
        model(torch.randn(1, 13, 16, 32))
    except ValueError as exc:
        assert "8 input channels" in str(exc)
    else:
        raise AssertionError("model accepted privileged condition channels")

    cfg = _gate_c_cfg("unknown")
    try:
        build_model("image_only_latent_marginal", cfg)
    except ValueError as exc:
        assert "latent_aggregation" in str(exc)
    else:
        raise AssertionError("unknown aggregation mode was accepted")


def test_semantic_tempsal_fusion_loads_frozen_branches(tmp_path) -> None:
    base_cfg = {
        "data": {"num_bins": 3},
        "model": {
            "in_channels": 8,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 3,
            "semantic_pretrained": False,
            "semantic_layer": "layer2",
        },
    }
    temporal = build_model("semantic_tempsal_temporal", base_cfg)
    static = build_model("semantic_static", base_cfg)
    temporal_path = tmp_path / "temporal.pt"
    static_path = tmp_path / "static.pt"
    torch.save({"model": temporal.state_dict()}, temporal_path)
    torch.save({"model": static.state_dict()}, static_path)

    cfg = {
        **base_cfg,
        "model": {
            **base_cfg["model"],
            "temporal_checkpoint": str(temporal_path),
            "static_checkpoint": str(static_path),
            "freeze_temporal": True,
            "freeze_static": True,
            "shuffle_temporal_inputs": True,
        },
    }
    model = build_model("semantic_tempsal_fusion", cfg)
    x = torch.randn(2, 8, 32, 64)
    out = model(x)

    assert out["final"].shape == (2, 1, 32, 64)
    assert out["temporal"].shape == (2, 3, 32, 64)
    assert all(not p.requires_grad for p in model.temporal_branch.parameters())
    assert all(not p.requires_grad for p in model.static_branch.parameters())


def test_current_transition_nextfix_forward_shapes() -> None:
    cfg = {
        "data": {"num_bins": 1},
        "train": {"scanpath_prefix_length": 3},
        "model": {
            "in_channels": 12,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 1,
        },
    }
    model = build_model("current_transition", cfg)
    x = torch.randn(2, 12, 32, 64)
    x[:, 8:11] = torch.rand(2, 3, 32, 64)
    out = model(x)

    assert out["final"].shape == (2, 1, 32, 64)
    assert out["temporal"].shape == (2, 1, 32, 64)
    assert out["cumulative"].shape == (2, 1, 32, 64)
    assert out["base_final"].shape == (2, 1, 32, 64)
    assert out["motion_logits"].shape == (2, 81)
    assert torch.allclose(out["base_final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)


def test_current_transition_renderer_outputs_motion_prior() -> None:
    cfg = {
        "data": {"num_bins": 1},
        "train": {"scanpath_prefix_length": 3},
        "model": {
            "in_channels": 12,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 1,
            "motion_render_scale": 0.5,
        },
    }
    model = build_model("current_transition_renderer", cfg)
    x = torch.randn(2, 12, 16, 32)
    x[:, 8:11] = 0.0
    x[:, 10, 8, 16] = 1.0
    out = model(x)

    assert out["final"].shape == (2, 1, 16, 32)
    assert out["motion_logits"].shape == (2, 81)
    assert out["motion_prior"].shape == (2, 1, 16, 32)
    assert out["base_final"].shape == (2, 1, 16, 32)
    assert torch.allclose(out["motion_prior"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["base_final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)


def test_current_transition_renderer_accepts_start_geometry_context() -> None:
    cfg = {
        "data": {"num_bins": 1},
        "train": {"scanpath_prefix_length": 3},
        "model": {
            "in_channels": 15,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 1,
            "motion_render_scale": 0.5,
        },
    }
    model = build_model("current_transition_renderer", cfg)
    x = torch.randn(2, 15, 16, 32)
    x[:, 8:11] = 0.0
    x[:, 10, 8, 16] = 1.0
    out = model(x)

    assert out["final"].shape == (2, 1, 16, 32)
    assert out["motion_prior"].shape == (2, 1, 16, 32)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)


def test_current_transition_renderer_outputs_gated_start_prior() -> None:
    cfg = {
        "data": {"num_bins": 1},
        "train": {"scanpath_prefix_length": 3},
        "model": {
            "in_channels": 15,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 1,
            "motion_render_scale": 0.5,
            "use_start_prior": True,
            "start_prior_init_scale": 0.1,
            "start_prior_temperature": 2.0,
        },
    }
    model = build_model("current_transition_renderer", cfg)
    x = torch.randn(2, 15, 16, 32)
    x[:, 8:11] = 0.0
    x[:, 10, 8, 16] = 1.0
    out = model(x)

    assert out["final"].shape == (2, 1, 16, 32)
    assert out["start_prior"].shape == (2, 1, 16, 32)
    assert model.start_prior_temperature == 2.0
    assert 0.0 < float(out["start_prior_scale"]) < 1.0
    assert torch.allclose(out["start_prior"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)


def test_current_transition_renderer_outputs_mixture_start_prior() -> None:
    cfg = {
        "data": {"num_bins": 1},
        "train": {"scanpath_prefix_length": 3},
        "model": {
            "in_channels": 15,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 1,
            "motion_render_scale": 0.5,
            "use_start_prior": True,
            "start_prior_init_scale": 0.2,
            "start_prior_fusion": "mixture",
        },
    }
    model = build_model("current_transition_renderer", cfg)
    x = torch.randn(2, 15, 16, 32)
    x[:, 8:11] = 0.0
    x[:, 10, 8, 16] = 1.0
    out = model(x)

    assert model.start_prior_fusion == "mixture"
    assert out["final"].shape == (2, 1, 16, 32)
    assert out["start_prior"].shape == (2, 1, 16, 32)
    assert torch.allclose(out["start_prior"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)


def test_current_transition_renderer_outputs_hybrid_start_prior() -> None:
    cfg = {
        "data": {"num_bins": 1},
        "train": {"scanpath_prefix_length": 3},
        "model": {
            "in_channels": 15,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 1,
            "motion_render_scale": 0.5,
            "use_start_prior": True,
            "start_prior_init_scale": 0.2,
            "start_prior_fusion": "hybrid",
            "start_prior_hybrid_residual_scale": 0.1,
        },
    }
    model = build_model("current_transition_renderer", cfg)
    x = torch.randn(2, 15, 16, 32)
    x[:, 8:11] = 0.0
    x[:, 10, 8, 16] = 1.0
    out = model(x)

    assert model.start_prior_fusion == "hybrid"
    assert model.start_prior_hybrid_residual_scale == 0.1
    assert out["final"].shape == (2, 1, 16, 32)
    assert out["start_prior"].shape == (2, 1, 16, 32)
    assert torch.allclose(out["start_prior"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)


def test_current_transition_renderer_outputs_selector_start_prior() -> None:
    cfg = {
        "data": {"num_bins": 1},
        "train": {"scanpath_prefix_length": 3},
        "model": {
            "in_channels": 15,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 1,
            "motion_render_scale": 0.5,
            "use_start_prior": True,
            "start_prior_init_scale": 0.2,
            "start_prior_fusion": "selector",
        },
    }
    model = build_model("current_transition_renderer", cfg)
    x = torch.randn(2, 15, 16, 32)
    x[:, 8:11] = 0.0
    x[:, 10, 8, 16] = 1.0
    out = model(x)

    assert model.start_prior_fusion == "selector"
    assert out["final"].shape == (2, 1, 16, 32)
    assert out["start_prior"].shape == (2, 1, 16, 32)
    assert out["start_selected"].shape == (2, 1, 16, 32)
    assert torch.allclose(out["start_prior"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["start_selected"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)


def test_current_transition_renderer_outputs_dualpath_start_prior() -> None:
    cfg = {
        "data": {"num_bins": 1},
        "train": {"scanpath_prefix_length": 3},
        "model": {
            "in_channels": 15,
            "base_channels": 8,
            "hidden_channels": 16,
            "num_bins": 1,
            "motion_render_scale": 0.5,
            "use_start_prior": True,
            "start_prior_init_scale": 0.2,
            "start_prior_fusion": "dualpath",
        },
    }
    model = build_model("current_transition_renderer", cfg)
    x = torch.randn(2, 15, 16, 32)
    x[:, 8:11] = 0.0
    x[:, 10, 8, 16] = 1.0
    out = model(x)

    assert model.start_prior_fusion == "dualpath"
    assert out["final"].shape == (2, 1, 16, 32)
    assert out["start_prior"].shape == (2, 1, 16, 32)
    assert out["start_selected"].shape == (2, 1, 16, 32)
    assert out["start_mixture"].shape == (2, 1, 16, 32)
    assert out["base_final"].shape == (2, 1, 16, 32)
    assert torch.allclose(out["start_prior"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["start_selected"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["start_mixture"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["base_final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["final"].sum(dim=(-2, -1)), torch.ones(2, 1), atol=1e-4)
