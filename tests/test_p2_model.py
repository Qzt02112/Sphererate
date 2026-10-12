from __future__ import annotations

import torch

from sphererate.models.registry import build_model


def _cfg() -> dict:
    return {
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


def test_p2_model_has_independent_normalized_final_and_auxiliary_heads() -> None:
    torch.manual_seed(7)
    model = build_model("semantic_static_aux", _cfg()).eval()
    x = torch.randn(2, 8, 32, 64)
    before = model(x)
    assert before["final"].shape == (2, 1, 32, 64)
    assert before["temporal"].shape == (2, 3, 32, 64)
    assert torch.allclose(before["final"].sum((-2, -1)), torch.ones(2, 1), atol=1e-5)
    assert torch.allclose(before["temporal"].sum((-2, -1)), torch.ones(2, 3), atol=1e-5)

    with torch.no_grad():
        for parameter in model.auxiliary_head.parameters():
            parameter.normal_(mean=0.0, std=10.0)
    after = model(x)
    assert torch.allclose(before["final"], after["final"], atol=1e-7)
    assert not torch.allclose(before["temporal"], after["temporal"])


def test_p2_model_rejects_privileged_input_channels() -> None:
    model = build_model("semantic_static_aux", _cfg())
    try:
        model(torch.randn(1, 13, 16, 32))
    except ValueError as exc:
        assert "8 input channels" in str(exc)
    else:
        raise AssertionError("P2 model accepted privileged channels")
