from __future__ import annotations

import os

import sys
from pathlib import Path

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pytest
import torch
from torch import nn

from sphererate.baselines.scandmm_declared_start import (
    build_declared_start_dmm,
    declared_start_xyz,
)


OFFICIAL = Path(os.path.expandvars("${SCANDMM_ROOT}/baselines/third_party/ScanDMM"))


class _ZeroFeatures(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.size(0), self.width, dtype=images.dtype, device=images.device)


def _official_dmm():
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    from models import DMM  # type: ignore

    return DMM


requires_official_source = pytest.mark.skipif(
    not OFFICIAL.is_dir(),
    reason="ScanDMM third-party source not checked out; set SCANDMM_ROOT to enable",
)


def test_declared_start_coordinate_matches_official_convention() -> None:
    xyz = declared_start_xyz(0.0, 90.0)
    assert torch.allclose(xyz, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


@requires_official_source
def test_adapter_preserves_official_state_dict_keys() -> None:
    official = _official_dmm()
    adapted = build_declared_start_dmm(official)
    assert set(official(z_dim=4, rnn_dim=7).state_dict()) == set(
        adapted(z_dim=4, rnn_dim=7).state_dict()
    )


@requires_official_source
def test_start_is_independent_while_first_target_remains_supervised() -> None:
    pyro.clear_param_store()
    torch.manual_seed(13)
    adapted = build_declared_start_dmm(_official_dmm())
    model = adapted(z_dim=4, emission_dim=5, transition_dim=6, rnn_dim=7)
    model.cnn = _ZeroFeatures(4)
    scanpaths = torch.tensor(
        [[[0.2, 0.3, 0.9], [0.4, 0.5, 0.7]], [[-0.2, 0.8, 0.5], [0.1, 0.2, 0.9]]],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 2)
    images = torch.zeros(2, 3, 8, 16)

    def trace_for(start: torch.Tensor):
        pyro.set_rng_seed(99)
        return pyro.poutine.trace(model.model).get_trace(
            scanpaths=scanpaths,
            scanpaths_reversed=None,
            mask=mask,
            scanpath_lengths=None,
            images=images,
            predict=False,
            initial_gaze=start,
        )

    trace_a = trace_for(declared_start_xyz(0.0, 90.0))
    trace_b = trace_for(declared_start_xyz(0.0, -90.0))
    assert trace_a.nodes["obs_x_1"]["is_observed"]
    assert torch.equal(trace_a.nodes["obs_x_1"]["value"], scanpaths[:, 0, :])
    loc_a = trace_a.nodes["z_1"]["fn"].base_dist.base_dist.loc
    loc_b = trace_b.nodes["z_1"]["fn"].base_dist.base_dist.loc
    assert not torch.allclose(loc_a, loc_b)


@requires_official_source
def test_adapter_refuses_implicit_first_target_initialization() -> None:
    adapted = build_declared_start_dmm(_official_dmm())
    model = adapted(z_dim=4, emission_dim=5, transition_dim=6, rnn_dim=7)
    model.cnn = _ZeroFeatures(4)
    with pytest.raises(ValueError, match="would leak the first target"):
        model.model(
            scanpaths=torch.zeros(1, 2, 3),
            scanpaths_reversed=None,
            mask=torch.ones(1, 2),
            scanpath_lengths=None,
            images=torch.zeros(1, 3, 8, 16),
        )


@requires_official_source
def test_guide_also_conditions_on_independent_declared_start() -> None:
    pyro.clear_param_store()
    torch.manual_seed(17)
    adapted = build_declared_start_dmm(_official_dmm())
    model = adapted(z_dim=4, emission_dim=5, transition_dim=6, rnn_dim=7)
    scanpaths = torch.tensor(
        [[[0.2, 0.3, 0.9], [0.4, 0.5, 0.7]], [[-0.2, 0.8, 0.5], [0.1, 0.2, 0.9]]],
        dtype=torch.float32,
    )
    lengths = torch.tensor([2, 2], dtype=torch.long)

    def trace_for(start: torch.Tensor):
        reversed_batch = poly.reverse_sequences(scanpaths, lengths)
        packed = nn.utils.rnn.pack_padded_sequence(
            reversed_batch, lengths, batch_first=True
        )
        pyro.clear_param_store()
        pyro.set_rng_seed(101)
        return pyro.poutine.trace(model.guide).get_trace(
            scanpaths=scanpaths,
            scanpaths_reversed=packed,
            mask=torch.ones(2, 2),
            scanpath_lengths=lengths,
            images=torch.zeros(2, 3, 8, 16),
            initial_gaze=start,
        )

    trace_a = trace_for(declared_start_xyz(0.0, 90.0))
    trace_b = trace_for(declared_start_xyz(0.0, -90.0))
    loc_a = trace_a.nodes["z_1"]["fn"].base_dist.base_dist.loc
    loc_b = trace_b.nodes["z_1"]["fn"].base_dist.base_dist.loc
    assert not torch.allclose(loc_a, loc_b)
