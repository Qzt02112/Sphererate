from __future__ import annotations

import torch

from sphererate.models.spherical_dwell_rate import SphericalDwellRateNet


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    image = torch.randn(2, 8, 8, 16)
    template = torch.rand(2, 4, 8, 16)
    geometry = torch.randn(2, 4, 8, 16)
    geometry[:, 0] = geometry[:, 0].abs()
    return image, template, geometry


def _model(**kwargs: object) -> SphericalDwellRateNet:
    torch.manual_seed(11)
    return SphericalDwellRateNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        semantic_pretrained=False,
        reachability_mode="anisotropic_tangent",
        **kwargs,
    ).eval()


def test_default_mode_matches_explicit_full_mode() -> None:
    image, template, geometry = _inputs()
    implicit = _model()
    explicit = _model(use_start_cost=True, accumulation_mode="cumulative_rate")
    explicit.load_state_dict(implicit.state_dict())
    with torch.no_grad():
        left = implicit(image, template, start_geometry=geometry)
        right = explicit(image, template, start_geometry=geometry)
    torch.testing.assert_close(left["cumulative"], right["cumulative"], rtol=0, atol=0)
    torch.testing.assert_close(
        left["raw_cumulative_mass"], right["raw_cumulative_mass"], rtol=0, atol=0
    )


def test_no_start_is_invariant_to_start_geometry() -> None:
    image, template, geometry = _inputs()
    model = _model(use_start_cost=False)
    with torch.no_grad():
        left = model(image, template, start_geometry=geometry)["cumulative"]
        right = model(image, template, start_geometry=-3.0 * geometry)["cumulative"]
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_independent_mode_reports_interval_rates() -> None:
    image, template, geometry = _inputs()
    model = _model(accumulation_mode="independent_interval")
    with torch.no_grad():
        output = model(image, template, start_geometry=geometry)
    torch.testing.assert_close(output["cumulative"], output["interval_rates"], rtol=0, atol=0)


def test_ablation_switches_preserve_parameter_counts() -> None:
    models = (
        _model(),
        _model(use_start_cost=False),
        _model(accumulation_mode="independent_interval"),
    )
    total = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    trainable = [
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        for model in models
    ]
    assert len(set(total)) == 1
    assert len(set(trainable)) == 1


def test_full_mode_has_nonnegative_raw_cumulative_increments() -> None:
    image, template, geometry = _inputs()
    with torch.no_grad():
        mass = _model()(image, template, start_geometry=geometry)["raw_cumulative_mass"]
    assert torch.all(mass[:, 1:] - mass[:, :-1] >= -1e-7)
