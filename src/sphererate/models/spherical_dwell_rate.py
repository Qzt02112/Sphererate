from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from sphererate.models.dytsal import SemanticFusionFeatures
from sphererate.models.layers import CircularConv2d, ConvBlock


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


class SphericalDwellRateNet(nn.Module):
    """Start-aware semantic attention dynamics in cumulative dwell-mass space.

    Each output channel is a non-negative interval dwell-rate distribution.
    Multiplication by the interval duration followed by cumulative addition is
    therefore physically consistent by construction.  Only the cumulative mass
    is normalized for saliency-density metrics.
    """

    residual_fusion = "spherical_dwell_rate"

    def __init__(
        self,
        *,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        horizons_s: Sequence[float] = (0.5, 1.0, 3.0, 5.0),
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        semantic_alpha_init: Sequence[float] = (0.05, 0.20, 0.65, 0.90),
        reachability_beta_init: Sequence[float] = (16.0, 4.0, 0.45, 0.10),
        reachability_mode: str = "isotropic",
        reachability_north_beta_init: Sequence[float] | None = None,
        reachability_offset_deg_init: Sequence[Sequence[float]] | None = None,
        maximum_reachability_offset_deg: float = 30.0,
        interval_residual_scale: float = 2.0,
        use_interval_residual: bool = True,
        use_start_cost: bool = True,
        accumulation_mode: str = "cumulative_rate",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.interval_residual_scale = float(interval_residual_scale)
        self.use_interval_residual = bool(use_interval_residual)
        self.use_start_cost = bool(use_start_cost)
        self.accumulation_mode = str(accumulation_mode).lower()
        if self.accumulation_mode not in {"cumulative_rate", "independent_interval"}:
            raise ValueError(
                "accumulation_mode must be cumulative_rate or independent_interval"
            )
        self.reachability_mode = str(reachability_mode).lower()
        if self.reachability_mode not in {"isotropic", "anisotropic_tangent"}:
            raise ValueError("reachability_mode must be isotropic or anisotropic_tangent")
        if self.interval_residual_scale <= 0.0:
            raise ValueError("interval_residual_scale must be positive")

        horizons = torch.as_tensor(tuple(float(value) for value in horizons_s))
        if horizons.ndim != 1 or len(horizons) < 2 or not torch.all(horizons[1:] > horizons[:-1]):
            raise ValueError("horizons_s must be a strictly increasing sequence")
        self.register_buffer("horizons_s", horizons.float())
        deltas = torch.cat([horizons[:1], horizons[1:] - horizons[:-1]])
        self.register_buffer("interval_durations_s", deltas.float())

        self.features = SemanticFusionFeatures(
            self.in_channels,
            int(base_channels),
            self.hidden_channels,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        # Names intentionally match the same-split static semantic checkpoint.
        self.semantic_block = ConvBlock(self.hidden_channels, self.hidden_channels)
        self.semantic_out = CircularConv2d(self.hidden_channels, 1, 1)

        self.interval_block = ConvBlock(self.hidden_channels, self.hidden_channels)
        self.interval_out = CircularConv2d(self.hidden_channels, len(horizons), 1)
        nn.init.zeros_(self.interval_out.conv.weight)
        nn.init.zeros_(self.interval_out.conv.bias)

        alpha_values = tuple(float(value) for value in semantic_alpha_init)
        if len(alpha_values) != len(horizons):
            raise ValueError("semantic_alpha_init must match horizons_s")
        if any(not 0.0 < value < 1.0 for value in alpha_values) or any(
            right <= left for left, right in zip(alpha_values, alpha_values[1:])
        ):
            raise ValueError("semantic_alpha_init must be strictly increasing in (0, 1)")
        alpha_cumulative = [-math.log1p(-value) for value in alpha_values]
        alpha_increments = [
            value - (alpha_cumulative[index - 1] if index else 0.0)
            for index, value in enumerate(alpha_cumulative)
        ]
        self.semantic_alpha_deltas = nn.Parameter(
            torch.tensor([_inverse_softplus(value) for value in alpha_increments])
        )

        beta_values = tuple(float(value) for value in reachability_beta_init)
        if len(beta_values) != len(horizons):
            raise ValueError("reachability_beta_init must match horizons_s")
        if any(value <= 0.0 for value in beta_values) or any(
            right >= left for left, right in zip(beta_values, beta_values[1:])
        ):
            raise ValueError("reachability_beta_init must be strictly decreasing and positive")
        beta_increments = [
            beta_values[index] - (beta_values[index + 1] if index + 1 < len(beta_values) else 0.0)
            for index in range(len(beta_values))
        ]
        self.reachability_beta_deltas = nn.Parameter(
            torch.tensor([_inverse_softplus(value) for value in beta_increments])
        )
        if self.reachability_mode == "anisotropic_tangent":
            north_values = tuple(
                float(value)
                for value in (
                    reachability_north_beta_init
                    if reachability_north_beta_init is not None
                    else beta_values
                )
            )
            if len(north_values) != len(horizons) or any(
                value <= 0.0 for value in north_values
            ) or any(right >= left for left, right in zip(north_values, north_values[1:])):
                raise ValueError(
                    "reachability_north_beta_init must be strictly decreasing and positive"
                )
            north_increments = [
                north_values[index]
                - (north_values[index + 1] if index + 1 < len(north_values) else 0.0)
                for index in range(len(north_values))
            ]
            self.reachability_north_beta_deltas = nn.Parameter(
                torch.tensor([_inverse_softplus(value) for value in north_increments])
            )
            self.maximum_reachability_offset_rad = math.radians(
                float(maximum_reachability_offset_deg)
            )
            if self.maximum_reachability_offset_rad <= 0.0:
                raise ValueError("maximum_reachability_offset_deg must be positive")
            offsets = torch.as_tensor(
                reachability_offset_deg_init
                if reachability_offset_deg_init is not None
                else [[0.0, 0.0] for _ in horizons],
                dtype=torch.float32,
            )
            if offsets.shape != (len(horizons), 2):
                raise ValueError("reachability_offset_deg_init must be [K,2]")
            offsets = torch.deg2rad(offsets).clamp(
                -0.999 * self.maximum_reachability_offset_rad,
                0.999 * self.maximum_reachability_offset_rad,
            )
            self.reachability_offsets_unconstrained = nn.Parameter(
                torch.atanh(offsets / self.maximum_reachability_offset_rad)
            )

    def semantic_alphas(self) -> torch.Tensor:
        increments = torch.nn.functional.softplus(self.semantic_alpha_deltas)
        return 1.0 - torch.exp(-torch.cumsum(increments, dim=0))

    def reachability_betas(self) -> torch.Tensor:
        increments = torch.nn.functional.softplus(self.reachability_beta_deltas)
        return torch.flip(torch.cumsum(torch.flip(increments, dims=(0,)), dim=0), dims=(0,))

    def reachability_north_betas(self) -> torch.Tensor:
        if self.reachability_mode != "anisotropic_tangent":
            return self.reachability_betas()
        increments = torch.nn.functional.softplus(self.reachability_north_beta_deltas)
        return torch.flip(torch.cumsum(torch.flip(increments, dims=(0,)), dim=0), dims=(0,))

    def reachability_offsets(self) -> torch.Tensor:
        if self.reachability_mode != "anisotropic_tangent":
            return self.reachability_beta_deltas.new_zeros((len(self.horizons_s), 2))
        return self.maximum_reachability_offset_rad * torch.tanh(
            self.reachability_offsets_unconstrained
        )

    @staticmethod
    def _spherical_normalize(mass: torch.Tensor) -> torch.Tensor:
        height = int(mass.shape[-2])
        y = torch.arange(height, dtype=mass.dtype, device=mass.device) + 0.5
        latitude = torch.pi / 2.0 - torch.pi * y / float(height)
        area = torch.cos(latitude).clamp_min(1e-4).view(1, 1, height, 1)
        return mass.clamp_min(0.0) / (mass.clamp_min(0.0) * area).sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-7)

    @classmethod
    def _spherical_softmax(cls, logits: torch.Tensor) -> torch.Tensor:
        stable = logits - logits.amax(dim=(-2, -1), keepdim=True)
        return cls._spherical_normalize(torch.exp(stable))

    def forward(
        self,
        image: torch.Tensor,
        template: torch.Tensor | None = None,
        *,
        start_geometry: torch.Tensor | None = None,
        horizons_s: torch.Tensor | None = None,
        conditioning_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        del horizons_s, conditioning_mode
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(f"Expected image [B,{self.in_channels},H,W]")
        if start_geometry is None or start_geometry.ndim != 4 or start_geometry.shape[0] != image.shape[0] or start_geometry.shape[-2:] != image.shape[-2:]:
            raise ValueError("start_geometry must be [B,C,H,W]")
        required_channels = 4 if self.reachability_mode == "anisotropic_tangent" else 2
        if start_geometry.shape[1] < required_channels:
            raise ValueError(f"start_geometry requires at least {required_channels} channels")
        if template is not None and template.shape[1] != len(self.horizons_s):
            raise ValueError("template horizon count does not match horizons_s")

        features = self.features(image)
        semantic_features = self.semantic_block(features)
        semantic_logits = self.semantic_out(semantic_features)
        semantic_map = self._spherical_softmax(semantic_logits)
        interval_residual = self.interval_residual_scale * torch.tanh(
            self.interval_out(self.interval_block(features)) / self.interval_residual_scale
        )
        if not self.use_interval_residual:
            interval_residual = torch.zeros_like(interval_residual)

        alphas = self.semantic_alphas().to(dtype=image.dtype, device=image.device)
        betas = self.reachability_betas().to(dtype=image.dtype, device=image.device)
        geometry = start_geometry.to(dtype=image.dtype)
        if self.reachability_mode == "anisotropic_tangent":
            north_betas = self.reachability_north_betas().to(
                dtype=image.dtype, device=image.device
            )
            offsets = self.reachability_offsets().to(dtype=image.dtype, device=image.device)
            east = geometry[:, 2:3] - offsets[:, 0].view(1, -1, 1, 1)
            north = geometry[:, 3:4] - offsets[:, 1].view(1, -1, 1, 1)
            reachability_cost = (
                betas.view(1, -1, 1, 1) * east.square()
                + north_betas.view(1, -1, 1, 1) * north.square()
            )
        else:
            north_betas = betas
            offsets = betas.new_zeros((len(betas), 2))
            reachability_cost = betas.view(1, -1, 1, 1) * geometry[:, :1].square()
        if not self.use_start_cost:
            reachability_cost = torch.zeros_like(reachability_cost)
        utility = (
            alphas.view(1, -1, 1, 1) * (semantic_logits + interval_residual)
            - reachability_cost
        )
        interval_rates = self._spherical_softmax(utility)

        cumulative_mass = torch.cumsum(
            interval_rates * self.interval_durations_s.to(
                dtype=image.dtype, device=image.device
            ).view(1, -1, 1, 1),
            dim=1,
        )
        if self.accumulation_mode == "cumulative_rate":
            cumulative = self._spherical_normalize(cumulative_mass)
        else:
            cumulative = interval_rates
        return {
            "cumulative": cumulative,
            "temporal": interval_rates,
            "final": cumulative[:, -1:],
            "interval_rates": interval_rates,
            "raw_cumulative_mass": cumulative_mass,
            "semantic_map": semantic_map,
            "residual_logits": interval_residual,
            "effective_residual_logits": alphas.view(1, -1, 1, 1) * interval_residual,
            "semantic_alphas": alphas.view(1, -1),
            "reachability_betas": betas.view(1, -1),
            "reachability_north_betas": north_betas.view(1, -1),
            "reachability_offsets_rad": offsets.unsqueeze(0),
            "gates": alphas.view(1, -1),
        }
