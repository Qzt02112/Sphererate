from __future__ import annotations

import torch

from sphererate.data.maps import MapGeometry
from sphererate.models.early_horizon import TemplateTimeGatedResidualNet


def _model(mode: str) -> TemplateTimeGatedResidualNet:
    return TemplateTimeGatedResidualNet(
        in_channels=8,
        base_channels=4,
        hidden_channels=8,
        horizons_s=(0.5, 1.0, 3.0, 5.0),
        conditioning_mode=mode,
        semantic_pretrained=False,
    )


def test_initial_prediction_equals_template_and_is_spherically_normalized() -> None:
    model = _model("time")
    image = torch.rand(1, 8, 32, 64)
    area = torch.from_numpy(MapGeometry(32, 64).area_weights).view(1, 1, 32, 64)
    template = torch.rand(1, 4, 32, 64)
    template = template / (template * area).sum(dim=(-2, -1), keepdim=True)

    output = model(image, template)

    assert output["cumulative"].shape == (1, 4, 32, 64)
    assert torch.allclose(output["cumulative"], template, atol=2e-6)
    assert torch.allclose((output["cumulative"] * area).sum(dim=(-2, -1)), torch.ones(1, 4), atol=2e-5)
    assert torch.all(output["gates"][:, 1:] >= output["gates"][:, :-1])


def test_time_and_capacity_modes_have_identical_parameter_counts() -> None:
    time_model = _model("time")
    capacity_model = _model("capacity")
    assert sum(parameter.numel() for parameter in time_model.parameters()) == sum(
        parameter.numel() for parameter in capacity_model.parameters()
    )

