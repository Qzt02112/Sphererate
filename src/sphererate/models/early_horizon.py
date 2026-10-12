from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from sphererate.models.dytsal import SemanticFusionFeatures
from sphererate.models.layers import CircularConv2d, ConvBlock


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


class TemplateTimeGatedResidualNet(nn.Module):
    """A minimal template-to-content transition model for early horizons."""

    def __init__(
        self,
        *,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        horizons_s: Sequence[float] = (0.5, 1.0, 3.0, 5.0),
        conditioning_mode: str = "time",
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        residual_fusion: str = "monotonic_gate",
        residual_bound: float = 4.0,
        semantic_alpha_init: Sequence[float] | None = None,
        tile_rows: int = 8,
        tile_cols: int = 16,
        tile_hidden_channels: int = 16,
        tile_residual_scale: float = 0.5,
        pixel_hidden_channels: int = 16,
        pixel_residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.conditioning_mode = str(conditioning_mode).lower()
        if self.conditioning_mode not in {"time", "capacity"}:
            raise ValueError("conditioning_mode must be time or capacity")
        self.residual_fusion = str(residual_fusion).lower()
        if self.residual_fusion not in {
            "monotonic_gate",
            "bounded_residual",
            "semantic_anchor",
            "semantic_anchor_tile_residual",
            "semantic_anchor_pixel_residual",
            "temporal_semantic_anchor",
        }:
            raise ValueError(
                "residual_fusion must be monotonic_gate, bounded_residual, semantic_anchor, "
                "semantic_anchor_tile_residual, semantic_anchor_pixel_residual, "
                "or temporal_semantic_anchor"
            )
        self.residual_bound = float(residual_bound)
        if self.residual_bound <= 0.0:
            raise ValueError("residual_bound must be positive")
        horizons = torch.as_tensor(list(horizons_s), dtype=torch.float32)
        if horizons.ndim != 1 or len(horizons) < 2 or not torch.all(horizons[1:] > horizons[:-1]):
            raise ValueError("horizons_s must be a strictly increasing sequence")
        self.register_buffer("horizons_s", horizons)
        self.features = SemanticFusionFeatures(
            self.in_channels,
            int(base_channels),
            self.hidden_channels,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if self.residual_fusion in {
            "semantic_anchor",
            "semantic_anchor_tile_residual",
            "semantic_anchor_pixel_residual",
            "temporal_semantic_anchor",
        }:
            self.semantic_block = ConvBlock(self.hidden_channels, self.hidden_channels)
            self.semantic_out = CircularConv2d(self.hidden_channels, 1, 1)
            nn.init.zeros_(self.semantic_out.conv.weight)
            nn.init.zeros_(self.semantic_out.conv.bias)
            initial_alphas = tuple(
                float(value)
                for value in (
                    semantic_alpha_init
                    if semantic_alpha_init is not None
                    else (0.01, 0.03, 0.10, 0.20)
                )
            )
            if len(initial_alphas) != len(horizons):
                raise ValueError("semantic_alpha_init must match horizons_s")
            if any(
                not 0.0 < value < 1.0 for value in initial_alphas
            ) or any(
                right <= left for left, right in zip(initial_alphas, initial_alphas[1:])
            ):
                raise ValueError(
                    "semantic_alpha_init must be strictly increasing with values in (0, 1)"
                )
            cumulative = [-math.log1p(-float(value)) for value in initial_alphas]
            increments = [
                cumulative[index] - (cumulative[index - 1] if index else 0.0)
                for index in range(len(cumulative))
            ]
            self.semantic_alpha_deltas = nn.Parameter(
                torch.tensor(
                    [_inverse_softplus(value) for value in increments], dtype=torch.float32
                )
            )
            if self.residual_fusion == "semantic_anchor_tile_residual":
                self.tile_rows = int(tile_rows)
                self.tile_cols = int(tile_cols)
                self.tile_hidden_channels = int(tile_hidden_channels)
                self.tile_residual_scale = float(tile_residual_scale)
                if min(self.tile_rows, self.tile_cols, self.tile_hidden_channels) <= 0:
                    raise ValueError("tile grid and tile hidden channels must be positive")
                if self.tile_residual_scale <= 0.0:
                    raise ValueError("tile_residual_scale must be positive")
                self.tile_reduce = CircularConv2d(
                    self.hidden_channels, self.tile_hidden_channels, 1
                )
                self.tile_codes = nn.Parameter(
                    torch.zeros(len(horizons), self.tile_hidden_channels * 2)
                )
                nn.init.normal_(self.tile_codes, mean=0.0, std=0.02)
                self.tile_out = CircularConv2d(self.tile_hidden_channels, 1, 3)
                nn.init.zeros_(self.tile_out.conv.weight)
                nn.init.zeros_(self.tile_out.conv.bias)
            if self.residual_fusion == "semantic_anchor_pixel_residual":
                self.pixel_hidden_channels = int(pixel_hidden_channels)
                self.pixel_residual_scale = float(pixel_residual_scale)
                if self.pixel_hidden_channels <= 0 or self.pixel_residual_scale <= 0.0:
                    raise ValueError("pixel hidden channels and residual scale must be positive")
                self.pixel_reduce = CircularConv2d(
                    self.hidden_channels, self.pixel_hidden_channels, 1
                )
                self.pixel_codes = nn.Parameter(
                    torch.zeros(len(horizons), self.pixel_hidden_channels * 2)
                )
                nn.init.normal_(self.pixel_codes, mean=0.0, std=0.02)
                self.pixel_out = CircularConv2d(self.pixel_hidden_channels, 1, 3)
                nn.init.zeros_(self.pixel_out.conv.weight)
                nn.init.zeros_(self.pixel_out.conv.bias)
            if self.residual_fusion == "temporal_semantic_anchor":
                self.temporal_semantic_out = CircularConv2d(
                    self.hidden_channels, len(horizons), 1
                )
                nn.init.zeros_(self.temporal_semantic_out.conv.weight)
                nn.init.zeros_(self.temporal_semantic_out.conv.bias)
        else:
            # Both conditioning paths exist so the legacy matched modes retain
            # identical architecture and parameter count.
            self.time_mlp = nn.Sequential(
                nn.Linear(4, self.hidden_channels),
                nn.SiLU(inplace=True),
                nn.Linear(self.hidden_channels, self.hidden_channels * 2),
            )
            self.capacity_codes = nn.Parameter(
                torch.zeros(len(horizons), self.hidden_channels * 2)
            )
            nn.init.normal_(self.capacity_codes, mean=0.0, std=0.02)
            self.residual_block = ConvBlock(self.hidden_channels, self.hidden_channels)
            self.residual_out = CircularConv2d(self.hidden_channels, 1, 1)
            nn.init.zeros_(self.residual_out.conv.weight)
            nn.init.zeros_(self.residual_out.conv.bias)
            increments = (0.05, 0.10, 0.45, 0.40)
            if len(increments) != len(horizons):
                increments = tuple(1.0 / len(horizons) for _ in horizons)
            self.gate_deltas = nn.Parameter(
                torch.tensor(
                    [_inverse_softplus(value) for value in increments], dtype=torch.float32
                )
            )

    def monotonic_gates(self) -> torch.Tensor:
        increments = torch.nn.functional.softplus(self.gate_deltas)
        return torch.cumsum(increments, dim=0) / increments.sum().clamp_min(1e-7)

    def semantic_alphas(self) -> torch.Tensor:
        """Strictly ordered template-to-semantic mixing weights in (0, 1)."""

        increments = torch.nn.functional.softplus(self.semantic_alpha_deltas)
        return 1.0 - torch.exp(-torch.cumsum(increments, dim=0))

    def _time_features(self, horizons: torch.Tensor) -> torch.Tensor:
        maximum = self.horizons_s.max().to(dtype=horizons.dtype, device=horizons.device)
        unit = horizons / maximum
        return torch.stack(
            [
                unit,
                torch.sin(2.0 * torch.pi * unit),
                torch.cos(2.0 * torch.pi * unit),
                torch.log1p(horizons) / torch.log1p(maximum),
            ],
            dim=-1,
        )

    @staticmethod
    def _spherical_softmax(logits: torch.Tensor) -> torch.Tensor:
        height = logits.shape[-2]
        y = torch.arange(height, dtype=logits.dtype, device=logits.device) + 0.5
        latitude = torch.pi / 2.0 - torch.pi * y / float(height)
        area = torch.cos(latitude).clamp_min(1e-4).view(1, 1, height, 1)
        stable = logits - logits.amax(dim=(-2, -1), keepdim=True)
        density = torch.exp(stable)
        return density / (density * area).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-7)

    @staticmethod
    def _periodic_tile_upsample(tile_logits: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Bilinearly upsample a coarse ERP field while preserving longitude periodicity."""

        tile_cols = int(tile_logits.shape[-1])
        if int(width) % tile_cols:
            raise ValueError("output width must be divisible by tile columns")
        cell_width = int(width) // tile_cols
        padded = torch.cat(
            [tile_logits[..., -1:], tile_logits, tile_logits[..., :1]], dim=-1
        )
        upsampled = torch.nn.functional.interpolate(
            padded,
            size=(int(height), int(width) + 2 * cell_width),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled[..., cell_width:-cell_width]

    def forward(
        self,
        image: torch.Tensor,
        template: torch.Tensor,
        *,
        start_geometry: torch.Tensor | None = None,
        horizons_s: torch.Tensor | None = None,
        conditioning_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(f"Expected image [B,{self.in_channels},H,W]")
        if template.ndim != 4 or template.shape[1] != len(self.horizons_s):
            raise ValueError("template must have shape [B,K,H,W] matching frozen horizons")
        mode = str(conditioning_mode or self.conditioning_mode).lower()
        if mode not in {"time", "capacity"}:
            raise ValueError("conditioning_mode must be time or capacity")
        horizons = self.horizons_s if horizons_s is None else horizons_s
        horizons = horizons.to(dtype=image.dtype, device=image.device)
        if horizons.ndim != 1 or len(horizons) != template.shape[1]:
            raise ValueError("horizons_s override must be a 1D K-vector")
        features = self.features(image)
        if self.residual_fusion in {
            "semantic_anchor",
            "semantic_anchor_tile_residual",
            "semantic_anchor_pixel_residual",
            "temporal_semantic_anchor",
        }:
            semantic_features = self.semantic_block(features)
            semantic_logits = self.semantic_out(semantic_features)
            semantic_map = self._spherical_softmax(semantic_logits)
            temporal_semantic_maps = None
            if self.residual_fusion == "temporal_semantic_anchor":
                temporal_semantic_maps = self._spherical_softmax(
                    self.temporal_semantic_out(semantic_features)
                )
            alphas = self.semantic_alphas().to(dtype=features.dtype, device=features.device)
            predictions = []
            residuals = []
            effective_residuals = []
            tile_residuals = []
            pixel_residuals = []
            if self.residual_fusion == "semantic_anchor_tile_residual":
                tile_features = torch.nn.functional.adaptive_avg_pool2d(
                    features, (self.tile_rows, self.tile_cols)
                )
                tile_features = torch.nn.functional.silu(self.tile_reduce(tile_features))
            if self.residual_fusion == "semantic_anchor_pixel_residual":
                pixel_features = torch.nn.functional.silu(self.pixel_reduce(features))
            for index in range(template.shape[1]):
                template_log = template[:, index : index + 1].clamp_min(1e-8).log()
                current_semantic = (
                    temporal_semantic_maps[:, index : index + 1]
                    if temporal_semantic_maps is not None
                    else semantic_map
                )
                semantic_log = current_semantic.clamp_min(1e-8).log()
                residual = semantic_log - template_log
                effective_residual = alphas[index] * residual
                if self.residual_fusion == "semantic_anchor_tile_residual":
                    scale, shift = self.tile_codes[index].chunk(2)
                    conditioned_tile = tile_features * (
                        1.0 + 0.1 * torch.tanh(scale).view(1, -1, 1, 1)
                    )
                    conditioned_tile = conditioned_tile + shift.view(1, -1, 1, 1)
                    coarse_tile = self.tile_out(torch.nn.functional.silu(conditioned_tile))
                    effective_tile = self.tile_residual_scale * torch.tanh(coarse_tile)
                    tile_residual = self._periodic_tile_upsample(
                        effective_tile, template.shape[-2], template.shape[-1]
                    )
                    effective_residual = effective_residual + tile_residual
                    tile_residuals.append(tile_residual)
                if self.residual_fusion == "semantic_anchor_pixel_residual":
                    scale, shift = self.pixel_codes[index].chunk(2)
                    conditioned_pixel = pixel_features * (
                        1.0 + 0.1 * torch.tanh(scale).view(1, -1, 1, 1)
                    )
                    conditioned_pixel = conditioned_pixel + shift.view(1, -1, 1, 1)
                    raw_pixel = self.pixel_out(torch.nn.functional.silu(conditioned_pixel))
                    pixel_residual = self.pixel_residual_scale * torch.tanh(raw_pixel)
                    effective_residual = effective_residual + pixel_residual
                    pixel_residuals.append(pixel_residual)
                predictions.append(self._spherical_softmax(template_log + effective_residual))
                residuals.append(residual)
                effective_residuals.append(effective_residual)
            cumulative = torch.cat(predictions, dim=1)
            output = {
                "cumulative": cumulative,
                "temporal": cumulative,
                "final": cumulative[:, -1:],
                "semantic_map": semantic_map,
                "residual_logits": torch.cat(residuals, dim=1),
                "effective_residual_logits": torch.cat(effective_residuals, dim=1),
                "semantic_alphas": alphas.view(1, -1),
                "gates": alphas.view(1, -1),
            }
            if tile_residuals:
                output["tile_residual_logits"] = torch.cat(tile_residuals, dim=1)
            if pixel_residuals:
                output["pixel_residual_logits"] = torch.cat(pixel_residuals, dim=1)
            if temporal_semantic_maps is not None:
                output["temporal_semantic_maps"] = temporal_semantic_maps
            return output

        codes = self.time_mlp(self._time_features(horizons)) if mode == "time" else self.capacity_codes
        gates = self.monotonic_gates().to(dtype=features.dtype, device=features.device)
        predictions = []
        residuals = []
        effective_residuals = []
        for index in range(template.shape[1]):
            scale, shift = codes[index].chunk(2)
            conditioned = features * (1.0 + 0.1 * torch.tanh(scale).view(1, -1, 1, 1))
            conditioned = conditioned + shift.view(1, -1, 1, 1)
            residual = self.residual_out(self.residual_block(conditioned))
            if self.residual_fusion == "bounded_residual":
                effective_residual = self.residual_bound * torch.tanh(
                    residual / self.residual_bound
                )
            else:
                effective_residual = gates[index] * residual
            logits = template[:, index : index + 1].clamp_min(1e-8).log() + effective_residual
            predictions.append(self._spherical_softmax(logits))
            residuals.append(residual)
            effective_residuals.append(effective_residual)
        cumulative = torch.cat(predictions, dim=1)
        return {
            "cumulative": cumulative,
            "temporal": cumulative,
            "final": cumulative[:, -1:],
            "residual_logits": torch.cat(residuals, dim=1),
            "effective_residual_logits": torch.cat(effective_residuals, dim=1),
            "gates": gates.view(1, -1),
        }
