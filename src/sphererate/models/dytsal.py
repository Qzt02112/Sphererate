from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from sphererate.models.layers import CircularConv2d, ConvBlock, ConvGRUCell
from sphererate.models.unet import TinyUNetFeatures
from sphererate.utils import normalize_torch, spatial_softmax


def cumulative_from_temporal(temporal: torch.Tensor, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    b, k, h, w = temporal.shape
    if weights is None:
        weights = torch.ones(k, dtype=temporal.dtype, device=temporal.device) / float(k)
    weights = weights.view(1, k, 1, 1)
    weighted = temporal * weights
    cumulative = []
    running = torch.zeros((b, 1, h, w), dtype=temporal.dtype, device=temporal.device)
    for idx in range(k):
        running = running + weighted[:, idx : idx + 1]
        cumulative.append(normalize_torch(running))
    cumulative_t = torch.cat(cumulative, dim=1)
    final = normalize_torch(weighted.sum(dim=1, keepdim=True))
    return cumulative_t, final


class StaticSaliencyNet(nn.Module):
    def __init__(self, in_channels: int = 8, base_channels: int = 32, hidden_channels: int = 64, num_bins: int = 6) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.features = TinyUNetFeatures(in_channels, base_channels, hidden_channels)
        self.head = nn.Sequential(ConvBlock(hidden_channels, hidden_channels), CircularConv2d(hidden_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        final = spatial_softmax(self.head(feat))
        temporal = final.repeat(1, self.num_bins, 1, 1)
        cumulative = final.repeat(1, self.num_bins, 1, 1)
        return {"temporal": temporal, "cumulative": cumulative, "final": final}


class CurrentTransitionNextFixationNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 1,
        prefix_length: int = 3,
        motion_bins_y: int = 9,
        motion_bins_x: int = 9,
        use_motion_renderer: bool = False,
        motion_render_scale: float = 1.0,
        motion_render_learnable_scale: bool = False,
        motion_render_temperature: float = 1.0,
        use_start_prior: bool = False,
        start_prior_init_scale: float = 0.1,
        start_prior_temperature: float = 1.0,
        start_prior_fusion: str = "residual",
        start_prior_hybrid_residual_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.prefix_length = int(prefix_length)
        self.motion_bins_y = int(motion_bins_y)
        self.motion_bins_x = int(motion_bins_x)
        self.use_motion_renderer = bool(use_motion_renderer)
        self.motion_render_temperature = max(float(motion_render_temperature), 1e-4)
        scale = torch.tensor(float(motion_render_scale), dtype=torch.float32)
        if motion_render_learnable_scale:
            self.motion_render_scale = nn.Parameter(scale)
        else:
            self.register_buffer("motion_render_scale", scale, persistent=False)
        self.prefix_start = 8
        self.step_index = self.prefix_start + self.prefix_length
        self.extra_context_channels = max(0, int(in_channels) - (self.step_index + 1))
        self.use_start_prior = bool(use_start_prior)
        self.start_prior_temperature = max(float(start_prior_temperature), 1e-4)
        self.start_prior_fusion = str(start_prior_fusion).lower()
        if self.start_prior_fusion not in {"residual", "mixture", "hybrid", "selector", "dualpath"}:
            raise ValueError("start_prior_fusion must be 'residual', 'mixture', 'hybrid', 'selector', or 'dualpath'.")
        self.start_prior_hybrid_residual_scale = float(start_prior_hybrid_residual_scale)
        if self.use_start_prior and self.extra_context_channels < 3:
            raise ValueError("use_start_prior requires at least three start-geometry context channels.")
        context_channels = 9 + self.extra_context_channels  # RGB + ERP coord channels + scalar progress + optional context.
        self.context_features = TinyUNetFeatures(context_channels, base_channels, hidden_channels)
        self.current_encoder = ConvBlock(1, hidden_channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.prefix_length * 4, hidden_channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.fuse = ConvBlock(hidden_channels * 2, hidden_channels)
        self.prior_head = nn.Sequential(ConvBlock(hidden_channels, hidden_channels), CircularConv2d(hidden_channels, 1, 1))
        self.transition_head = nn.Sequential(ConvBlock(hidden_channels, hidden_channels), CircularConv2d(hidden_channels, 1, 1))
        if self.use_start_prior:
            self.start_prior_head = nn.Sequential(
                ConvBlock(4, hidden_channels),
                ConvBlock(hidden_channels, hidden_channels),
                CircularConv2d(hidden_channels, 1, 1),
            )
            init_scale = min(max(float(start_prior_init_scale), 1e-4), 1.0 - 1e-4)
            self.start_prior_gate = nn.Parameter(torch.logit(torch.tensor(init_scale, dtype=torch.float32)))
        else:
            self.start_prior_head = None
            self.register_buffer("start_prior_gate", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.motion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, self.motion_bins_y * self.motion_bins_x),
        )

    def _motion_prior(self, current: torch.Tensor, motion_logits: torch.Tensor) -> torch.Tensor:
        b, _, h, w = current.shape
        probs = F.softmax(motion_logits / self.motion_render_temperature, dim=-1)
        prior = torch.zeros_like(current)
        idx = 0
        for y_bin in range(self.motion_bins_y):
            y_unit = (float(y_bin) + 0.5) / float(self.motion_bins_y)
            shift_y = int(round((2.0 * y_unit - 1.0) * float(max(h - 1, 1))))
            for x_bin in range(self.motion_bins_x):
                x_unit = (float(x_bin) + 0.5) / float(self.motion_bins_x)
                shift_x = int(round((x_unit - 0.5) * float(w)))
                shifted = torch.roll(current, shifts=(shift_y, shift_x), dims=(-2, -1))
                prior = prior + probs[:, idx].view(b, 1, 1, 1) * shifted
                idx += 1
        return normalize_torch(prior)

    def _prefix_maps(self, x: torch.Tensor) -> torch.Tensor:
        prefix_end = self.prefix_start + self.prefix_length
        if x.shape[1] < prefix_end + 1:
            raise ValueError(
                f"CurrentTransitionNextFixationNet expects at least {prefix_end + 1} input channels, got {x.shape[1]}"
            )
        return x[:, self.prefix_start : prefix_end]

    def _step_channel(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.step_index : self.step_index + 1]

    def _extra_context(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.extra_context_channels <= 0:
            return None
        start = self.step_index + 1
        end = start + self.extra_context_channels
        if x.shape[1] < end:
            raise ValueError(f"Expected {end} input channels for optional context, got {x.shape[1]}")
        return x[:, start:end]

    def _relative_tokens(self, prefix_maps: torch.Tensor) -> torch.Tensor:
        b, p, h, w = prefix_maps.shape
        mass = prefix_maps.clamp_min(0.0)
        flat_mass = mass.flatten(2)
        total = flat_mass.sum(dim=-1).clamp_min(1e-7)
        presence = (flat_mass.sum(dim=-1) > 1e-7).to(dtype=mass.dtype)

        x_grid = torch.linspace(0.0, 1.0, w, dtype=mass.dtype, device=mass.device).view(1, 1, 1, w)
        y_grid = torch.linspace(0.0, 1.0, h, dtype=mass.dtype, device=mass.device).view(1, 1, h, 1)
        sin_x = (mass * torch.sin(2.0 * torch.pi * x_grid)).sum(dim=(-2, -1)) / total
        cos_x = (mass * torch.cos(2.0 * torch.pi * x_grid)).sum(dim=(-2, -1)) / total
        x_pos = torch.remainder(torch.atan2(sin_x, cos_x) / (2.0 * torch.pi), 1.0)
        y_pos = (mass * y_grid).sum(dim=(-2, -1)) / total

        current_x = x_pos[:, -1:]
        current_y = y_pos[:, -1:]
        current_presence = presence[:, -1:]
        dx = torch.remainder(x_pos - current_x + 0.5, 1.0) - 0.5
        dy = y_pos - current_y
        valid = presence * current_presence
        sin_dx = torch.sin(2.0 * torch.pi * dx) * valid
        cos_dx = torch.cos(2.0 * torch.pi * dx) * valid
        dy = dy * valid
        token = torch.stack([sin_dx, cos_dx, dy, presence], dim=-1)
        return token.reshape(b, p * 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        prefix_maps = self._prefix_maps(x)
        current = prefix_maps[:, -1:].contiguous()
        step = self._step_channel(x)
        extra = self._extra_context(x)
        context_parts = [x[:, :8], step]
        if extra is not None:
            context_parts.append(extra)
        context_in = torch.cat(context_parts, dim=1)
        context = self.context_features(context_in)
        current_feat = self.current_encoder(current)
        token = self.token_mlp(self._relative_tokens(prefix_maps)).view(x.shape[0], -1, 1, 1)
        conditioned = context + token
        fused = self.fuse(torch.cat([conditioned, current_feat], dim=1))
        prior_logits = self.prior_head(context)
        motion_logits = self.motion_head(fused)
        final_logits = prior_logits + self.transition_head(fused)
        start_prior = None
        start_prior_scale = None
        if self.use_start_prior:
            if extra is None or extra.shape[1] < 3:
                raise ValueError("use_start_prior requires start-relative geometry channels in the input.")
            start_prior_logits = self.start_prior_head(torch.cat([step, extra[:, :3]], dim=1))
            start_prior_logits = start_prior_logits - start_prior_logits.mean(dim=(-2, -1), keepdim=True)
            start_prior_logits = start_prior_logits / self.start_prior_temperature
            start_prior_scale = torch.sigmoid(self.start_prior_gate).to(dtype=final_logits.dtype)
            start_prior = spatial_softmax(start_prior_logits)
            if self.start_prior_fusion == "residual":
                final_logits = final_logits + start_prior_scale * start_prior_logits
        motion_prior = None
        if self.use_motion_renderer:
            motion_prior = self._motion_prior(current, motion_logits)
            final_logits = final_logits + self.motion_render_scale.to(dtype=final_logits.dtype) * motion_prior.clamp_min(1e-7).log()
        final = spatial_softmax(final_logits)
        base_final = final
        start_selected = None
        start_mixture = None
        if start_prior is not None and self.start_prior_fusion in {"mixture", "hybrid", "selector", "dualpath"}:
            if self.start_prior_fusion == "selector":
                start_selected = normalize_torch(final * start_prior.clamp_min(1e-7))
                final = normalize_torch((1.0 - start_prior_scale) * final + start_prior_scale * start_selected)
            elif self.start_prior_fusion == "dualpath":
                base_final = final
                start_mixture = normalize_torch((1.0 - start_prior_scale) * base_final + start_prior_scale * start_prior)
                start_selected = normalize_torch(base_final * start_prior.clamp_min(1e-7))
                final = normalize_torch((start_mixture + start_prior_scale * (start_selected - base_final)).clamp_min(1e-7))
            else:
                final = normalize_torch((1.0 - start_prior_scale) * final + start_prior_scale * start_prior)
                if self.start_prior_fusion == "hybrid" and self.start_prior_hybrid_residual_scale != 0.0:
                    sharpen_scale = self.start_prior_hybrid_residual_scale * start_prior_scale
                    final = spatial_softmax(final.clamp_min(1e-7).log() + sharpen_scale * start_prior_logits)
        static = spatial_softmax(prior_logits)
        temporal = final.repeat(1, self.num_bins, 1, 1)
        cumulative = final.repeat(1, self.num_bins, 1, 1)
        outputs = {
            "temporal": temporal,
            "cumulative": cumulative,
            "final": final,
            "base_final": base_final,
            "static": static,
            "motion_logits": motion_logits,
        }
        if motion_prior is not None:
            outputs["motion_prior"] = motion_prior
        if start_prior is not None:
            outputs["start_prior"] = start_prior
            outputs["start_prior_scale"] = torch.sigmoid(self.start_prior_gate).detach()
        if start_selected is not None:
            outputs["start_selected"] = start_selected
        if start_mixture is not None:
            outputs["start_mixture"] = start_mixture
        return outputs


class FrozenResNetSemanticEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", layer: str = "layer3", pretrained: bool = True) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported semantic backbone: {backbone}")
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:  # pragma: no cover - import errors are environment-specific.
            raise RuntimeError("torchvision is required for semantic models.") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        blocks = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
        channels = 128
        if layer in {"layer3", "layer4"}:
            blocks.append(model.layer3)
            channels = 256
        if layer == "layer4":
            blocks.append(model.layer4)
            channels = 512
        if layer not in {"layer2", "layer3", "layer4"}:
            raise ValueError(f"Unsupported semantic layer: {layer}")
        self.backbone = nn.Sequential(*blocks)
        self.out_channels = channels
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> "FrozenResNetSemanticEncoder":
        super().train(False)
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.clamp(0.0, 1.0)
        x = (rgb - self.mean.to(dtype=rgb.dtype)) / self.std.to(dtype=rgb.dtype)
        with torch.no_grad():
            return self.backbone(x).detach()


class SemanticFusionFeatures(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.local = TinyUNetFeatures(in_channels, base_channels, hidden_channels)
        self.semantic = FrozenResNetSemanticEncoder(
            backbone=semantic_backbone,
            layer=semantic_layer,
            pretrained=semantic_pretrained,
        )
        self.semantic_proj = ConvBlock(self.semantic.out_channels, hidden_channels)
        self.fuse = ConvBlock(hidden_channels * 2, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        sem = self.semantic(x[:, :3])
        sem = F.interpolate(sem, size=local.shape[-2:], mode="bilinear", align_corners=False)
        sem = self.semantic_proj(sem)
        return self.fuse(torch.cat([local, sem], dim=1))


class SemanticStaticSaliencyNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.features = SemanticFusionFeatures(
            in_channels,
            base_channels,
            hidden_channels,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        self.head = nn.Sequential(ConvBlock(hidden_channels, hidden_channels), CircularConv2d(hidden_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        final = spatial_softmax(self.head(feat))
        temporal = final.repeat(1, self.num_bins, 1, 1)
        cumulative = final.repeat(1, self.num_bins, 1, 1)
        return {"temporal": temporal, "cumulative": cumulative, "final": final}


class SemanticStaticAuxSaliencyNet(nn.Module):
    """Image-only final predictor with an independent training-only auxiliary head."""

    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        if self.in_channels != 8:
            raise ValueError("SemanticStaticAuxSaliencyNet requires exactly 8 image + geometry input channels")
        self.num_bins = int(num_bins)
        self.features = SemanticFusionFeatures(
            self.in_channels,
            base_channels,
            hidden_channels,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        self.final_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )
        self.auxiliary_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            CircularConv2d(hidden_channels, self.num_bins, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            got = x.shape[1] if x.ndim >= 2 else "unknown"
            raise ValueError(f"Expected exactly {self.in_channels} input channels, got {got}")
        features = self.features(x)
        final = spatial_softmax(self.final_head(features))
        temporal = spatial_softmax(self.auxiliary_head(features))
        cumulative, _ = cumulative_from_temporal(temporal)
        return {"temporal": temporal, "cumulative": cumulative, "final": final, "static": final}


class SemanticDyTSalDMM360(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.features = SemanticFusionFeatures(
            in_channels,
            base_channels,
            hidden_channels,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        self.static_head = nn.Sequential(ConvBlock(hidden_channels, hidden_channels), CircularConv2d(hidden_channels, 1, 1))
        self.temporal_embed = nn.Parameter(torch.randn(num_bins, hidden_channels, 1, 1) * 0.02)
        self.gru = ConvGRUCell(hidden_channels + hidden_channels + 2, hidden_channels)
        self.gate = nn.Sequential(
            CircularConv2d(hidden_channels * 2 + 2, hidden_channels, 3),
            nn.SiLU(inplace=True),
            CircularConv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.emit = nn.Sequential(
            ConvBlock(hidden_channels * 2 + 2, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )
        self.logits_w = nn.Parameter(torch.zeros(num_bins), requires_grad=learned_accumulation)
        self.residual_gate = nn.Parameter(torch.full((num_bins,), -1.0))
        self.eta = nn.Parameter(torch.tensor(0.25))
        self.rho = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        static_logits = self.static_head(feat)
        static = spatial_softmax(static_logits)
        b, _, h, w = feat.shape
        memory = torch.zeros((b, 1, h, w), dtype=feat.dtype, device=feat.device)
        h_state: torch.Tensor | None = None
        temporal = []
        residuals = []
        memory_overlap = []
        for k in range(self.num_bins):
            emb = self.temporal_embed[k].expand(b, -1, h, w)
            transition_in = torch.cat([feat, emb, memory, static], dim=1)
            h_state = self.gru(transition_in, h_state)
            emit_in = torch.cat([feat, h_state, memory, static], dim=1)
            raw_gate = self.gate(emit_in)
            eta = torch.sigmoid(self.eta)
            rho = torch.sigmoid(self.rho) * 0.25
            soft_gate = raw_gate * (1.0 - eta * memory) + rho * memory
            residual = self.emit(emit_in) + torch.log(soft_gate.clamp_min(1e-6))
            residual = torch.sigmoid(self.residual_gate[k]) * residual
            sal = spatial_softmax(static_logits + residual)
            temporal.append(sal)
            residuals.append(residual)
            memory_overlap.append((sal * memory).sum(dim=(1, 2, 3)).mean())
            weights_so_far = torch.softmax(self.logits_w[: k + 1], dim=0).view(1, k + 1, 1, 1)
            memory = normalize_torch(torch.cat(temporal, dim=1).mul(weights_so_far).sum(dim=1, keepdim=True))
        temporal_t = torch.cat(temporal, dim=1)
        residual_t = torch.cat(residuals, dim=1)
        weights = torch.softmax(self.logits_w, dim=0)
        cumulative, final = cumulative_from_temporal(temporal_t, weights)
        return {
            "temporal": temporal_t,
            "cumulative": cumulative,
            "final": final,
            "static": static,
            "residual_logits": residual_t,
            "weights": weights,
            "memory_overlap": torch.stack(memory_overlap).mean() if memory_overlap else torch.tensor(0.0, device=x.device),
        }


class SemanticTeacherResidualDyTSalDMM360(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        teacher_checkpoint: str | None = None,
        residual_scale: float = 1.0,
        residual_init_gate: float = -1.5,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.residual_scale = float(residual_scale)
        self.teacher = SemanticStaticSaliencyNet(
            in_channels,
            base_channels,
            hidden_channels,
            num_bins,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if teacher_checkpoint:
            state = torch.load(Path(teacher_checkpoint), map_location="cpu")
            self.teacher.load_state_dict(state["model"])
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.features = SemanticFusionFeatures(
            in_channels,
            base_channels,
            hidden_channels,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        self.temporal_embed = nn.Parameter(torch.randn(num_bins, hidden_channels, 1, 1) * 0.02)
        self.gru = ConvGRUCell(hidden_channels + hidden_channels + 2, hidden_channels)
        self.gate = nn.Sequential(
            CircularConv2d(hidden_channels * 2 + 2, hidden_channels, 3),
            nn.SiLU(inplace=True),
            CircularConv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.emit = nn.Sequential(
            ConvBlock(hidden_channels * 2 + 2, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )
        self.logits_w = nn.Parameter(torch.zeros(num_bins), requires_grad=learned_accumulation)
        self.residual_gate = nn.Parameter(torch.full((num_bins,), float(residual_init_gate)))
        self.eta = nn.Parameter(torch.tensor(0.25))
        self.rho = nn.Parameter(torch.tensor(0.05))

    def train(self, mode: bool = True) -> "SemanticTeacherResidualDyTSalDMM360":
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_static = self.teacher(x)["final"].detach()
        base_logits = torch.log(teacher_static.clamp_min(1e-7))
        feat = self.features(x)
        b, _, h, w = feat.shape
        memory = torch.zeros((b, 1, h, w), dtype=feat.dtype, device=feat.device)
        h_state: torch.Tensor | None = None
        temporal = []
        residuals = []
        memory_overlap = []
        for k in range(self.num_bins):
            emb = self.temporal_embed[k].expand(b, -1, h, w)
            transition_in = torch.cat([feat, emb, memory, teacher_static], dim=1)
            h_state = self.gru(transition_in, h_state)
            emit_in = torch.cat([feat, h_state, memory, teacher_static], dim=1)
            raw_gate = self.gate(emit_in)
            eta = torch.sigmoid(self.eta)
            rho = torch.sigmoid(self.rho) * 0.25
            soft_gate = raw_gate * (1.0 - eta * memory) + rho * memory
            residual_raw = self.emit(emit_in) + torch.log(soft_gate.clamp_min(1e-6))
            residual = self.residual_scale * torch.sigmoid(self.residual_gate[k]) * torch.tanh(residual_raw)
            sal = spatial_softmax(base_logits + residual)
            temporal.append(sal)
            residuals.append(residual)
            memory_overlap.append((sal * memory).sum(dim=(1, 2, 3)).mean())
            weights_so_far = torch.softmax(self.logits_w[: k + 1], dim=0).view(1, k + 1, 1, 1)
            memory = normalize_torch(torch.cat(temporal, dim=1).mul(weights_so_far).sum(dim=1, keepdim=True))
        temporal_t = torch.cat(temporal, dim=1)
        residual_t = torch.cat(residuals, dim=1)
        weights = torch.softmax(self.logits_w, dim=0)
        cumulative, final = cumulative_from_temporal(temporal_t, weights)
        return {
            "temporal": temporal_t,
            "cumulative": cumulative,
            "final": final,
            "static": teacher_static,
            "teacher": teacher_static,
            "residual_logits": residual_t,
            "weights": weights,
            "memory_overlap": torch.stack(memory_overlap).mean() if memory_overlap else torch.tensor(0.0, device=x.device),
        }


class StartConditionedTeacherResidualDyTSalDMM360(nn.Module):
    """Static-first teacher residual model with condition channels kept out of the static trunk."""

    def __init__(
        self,
        in_channels: int = 13,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        teacher_checkpoint: str | None = None,
        residual_scale: float = 1.0,
        residual_init_gate: float = -1.5,
        static_in_channels: int = 8,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.static_in_channels = int(static_in_channels)
        self.condition_channels = max(0, int(in_channels) - self.static_in_channels)
        self.residual_scale = float(residual_scale)

        self.teacher = SemanticStaticSaliencyNet(
            self.static_in_channels,
            base_channels,
            hidden_channels,
            num_bins,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if teacher_checkpoint:
            state = torch.load(Path(teacher_checkpoint), map_location="cpu")
            self.teacher.load_state_dict(state["model"])
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.image_features = SemanticFusionFeatures(
            self.static_in_channels,
            base_channels,
            hidden_channels,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if self.condition_channels > 0:
            self.condition_features = ConvBlock(self.condition_channels, hidden_channels)
            self.condition_fuse = ConvBlock(hidden_channels * 2, hidden_channels)
        else:
            self.condition_features = None
            self.condition_fuse = None

        self.temporal_embed = nn.Parameter(torch.randn(num_bins, hidden_channels, 1, 1) * 0.02)
        self.gru = ConvGRUCell(hidden_channels + hidden_channels + 2, hidden_channels)
        self.gate = nn.Sequential(
            CircularConv2d(hidden_channels * 2 + 2, hidden_channels, 3),
            nn.SiLU(inplace=True),
            CircularConv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.emit = nn.Sequential(
            ConvBlock(hidden_channels * 2 + 2, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )
        self.logits_w = nn.Parameter(torch.zeros(num_bins), requires_grad=learned_accumulation)
        self.residual_gate = nn.Parameter(torch.full((num_bins,), float(residual_init_gate)))
        self.eta = nn.Parameter(torch.tensor(0.25))
        self.rho = nn.Parameter(torch.tensor(0.05))

    def train(self, mode: bool = True) -> "StartConditionedTeacherResidualDyTSalDMM360":
        super().train(mode)
        self.teacher.eval()
        return self

    def _split_static_condition(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.shape[1] < self.static_in_channels:
            raise ValueError(f"Expected at least {self.static_in_channels} input channels, got {x.shape[1]}.")
        static_x = x[:, : self.static_in_channels]
        condition_x = x[:, self.static_in_channels :] if x.shape[1] > self.static_in_channels else None
        return static_x, condition_x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        static_x, condition_x = self._split_static_condition(x)
        with torch.no_grad():
            teacher_static = self.teacher(static_x)["final"].detach()
        base_logits = torch.log(teacher_static.clamp_min(1e-7))

        feat = self.image_features(static_x)
        if self.condition_features is not None and self.condition_fuse is not None and condition_x is not None:
            cond = self.condition_features(condition_x)
            feat = self.condition_fuse(torch.cat([feat, cond], dim=1))

        b, _, h, w = feat.shape
        memory = torch.zeros((b, 1, h, w), dtype=feat.dtype, device=feat.device)
        h_state: torch.Tensor | None = None
        temporal = []
        residuals = []
        memory_overlap = []
        for k in range(self.num_bins):
            emb = self.temporal_embed[k].expand(b, -1, h, w)
            transition_in = torch.cat([feat, emb, memory, teacher_static], dim=1)
            h_state = self.gru(transition_in, h_state)
            emit_in = torch.cat([feat, h_state, memory, teacher_static], dim=1)
            raw_gate = self.gate(emit_in)
            eta = torch.sigmoid(self.eta)
            rho = torch.sigmoid(self.rho) * 0.25
            soft_gate = raw_gate * (1.0 - eta * memory) + rho * memory
            residual_raw = self.emit(emit_in) + torch.log(soft_gate.clamp_min(1e-6))
            residual = self.residual_scale * torch.sigmoid(self.residual_gate[k]) * torch.tanh(residual_raw)
            sal = spatial_softmax(base_logits + residual)
            temporal.append(sal)
            residuals.append(residual)
            memory_overlap.append((sal * memory).sum(dim=(1, 2, 3)).mean())
            weights_so_far = torch.softmax(self.logits_w[: k + 1], dim=0).view(1, k + 1, 1, 1)
            memory = normalize_torch(torch.cat(temporal, dim=1).mul(weights_so_far).sum(dim=1, keepdim=True))
        temporal_t = torch.cat(temporal, dim=1)
        residual_t = torch.cat(residuals, dim=1)
        weights = torch.softmax(self.logits_w, dim=0)
        cumulative, final = cumulative_from_temporal(temporal_t, weights)
        return {
            "temporal": temporal_t,
            "cumulative": cumulative,
            "final": final,
            "static": teacher_static,
            "teacher": teacher_static,
            "residual_logits": residual_t,
            "weights": weights,
            "memory_overlap": torch.stack(memory_overlap).mean() if memory_overlap else torch.tensor(0.0, device=x.device),
        }


class LightweightStartConditionedResidualNet(nn.Module):
    """Frozen static teacher plus a small two-layer bounded condition residual head."""

    def __init__(
        self,
        in_channels: int = 13,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        teacher_checkpoint: str | None = None,
        residual_scale: float = 1.0,
        residual_init_gate: float = -1.0,
        static_in_channels: int = 8,
        residual_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.static_in_channels = int(static_in_channels)
        self.condition_channels = max(0, int(in_channels) - self.static_in_channels)
        if self.condition_channels <= 0:
            raise ValueError("LightweightStartConditionedResidualNet requires condition channels")
        self.residual_scale = float(residual_scale)

        self.teacher = SemanticStaticSaliencyNet(
            self.static_in_channels,
            base_channels,
            hidden_channels,
            num_bins,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if teacher_checkpoint:
            state = torch.load(Path(teacher_checkpoint), map_location="cpu")
            self.teacher.load_state_dict(state["model"])
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        residual_in_channels = hidden_channels + self.condition_channels
        self.residual_head = nn.Sequential(
            CircularConv2d(residual_in_channels, int(residual_hidden_channels), 3),
            nn.SiLU(inplace=True),
            CircularConv2d(int(residual_hidden_channels), 1, 3),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init_gate)))

    def train(self, mode: bool = True) -> "LightweightStartConditionedResidualNet":
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_channels = self.static_in_channels + self.condition_channels
        if x.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} input channels, got {x.shape[1]}")
        static_x = x[:, : self.static_in_channels]
        condition_x = x[:, self.static_in_channels :]
        with torch.no_grad():
            teacher_features = self.teacher.features(static_x).detach()
            teacher_static = spatial_softmax(self.teacher.head(teacher_features)).detach()
        residual_raw = self.residual_head(torch.cat([teacher_features, condition_x], dim=1))
        residual = self.residual_scale * torch.sigmoid(self.residual_gate) * torch.tanh(residual_raw)
        final = spatial_softmax(torch.log(teacher_static.clamp_min(1e-7)) + residual)
        temporal = final.repeat(1, self.num_bins, 1, 1)
        cumulative = temporal.clone()
        return {
            "temporal": temporal,
            "cumulative": cumulative,
            "final": final,
            "static": teacher_static,
            "teacher": teacher_static,
            "residual_logits": residual,
            "residual_gate_value": torch.sigmoid(self.residual_gate),
        }


class LearnedStartOnlySaliencyNet(nn.Module):
    """Small geometry-only control that never reads RGB or absolute ERP channels."""

    def __init__(
        self,
        in_channels: int = 13,
        num_bins: int = 6,
        static_in_channels: int = 8,
        residual_hidden_channels: int = 16,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.static_in_channels = int(static_in_channels)
        self.condition_channels = int(in_channels) - self.static_in_channels
        if self.condition_channels <= 0:
            raise ValueError("LearnedStartOnlySaliencyNet requires condition channels")
        self.head = nn.Sequential(
            CircularConv2d(self.condition_channels, int(residual_hidden_channels), 3),
            nn.SiLU(inplace=True),
            CircularConv2d(int(residual_hidden_channels), 1, 3),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_channels = self.static_in_channels + self.condition_channels
        if x.ndim != 4 or x.shape[1] != expected_channels:
            got = x.shape[1] if x.ndim >= 2 else "unknown"
            raise ValueError(f"Expected {expected_channels} input channels, got {got}")
        condition_x = x[:, self.static_in_channels :]
        logits = self.head(condition_x)
        final = spatial_softmax(logits)
        temporal = final.repeat(1, self.num_bins, 1, 1)
        return {
            "temporal": temporal,
            "cumulative": temporal.clone(),
            "final": final,
            "condition_logits": logits,
        }


class DirectStartConditionedSaliencyNet(nn.Module):
    """Frozen semantic features plus a direct two-layer conditioned saliency head."""

    def __init__(
        self,
        in_channels: int = 13,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        teacher_checkpoint: str | None = None,
        static_in_channels: int = 8,
        residual_hidden_channels: int = 16,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.static_in_channels = int(static_in_channels)
        self.condition_channels = int(in_channels) - self.static_in_channels
        if self.condition_channels <= 0:
            raise ValueError("DirectStartConditionedSaliencyNet requires condition channels")
        self.teacher = SemanticStaticSaliencyNet(
            self.static_in_channels,
            base_channels,
            hidden_channels,
            self.num_bins,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if teacher_checkpoint:
            state = torch.load(Path(teacher_checkpoint), map_location="cpu")
            self.teacher.load_state_dict(state["model"])
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()
        self.head = nn.Sequential(
            CircularConv2d(hidden_channels + self.condition_channels, int(residual_hidden_channels), 3),
            nn.SiLU(inplace=True),
            CircularConv2d(int(residual_hidden_channels), 1, 3),
        )

    def train(self, mode: bool = True) -> "DirectStartConditionedSaliencyNet":
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_channels = self.static_in_channels + self.condition_channels
        if x.ndim != 4 or x.shape[1] != expected_channels:
            got = x.shape[1] if x.ndim >= 2 else "unknown"
            raise ValueError(f"Expected {expected_channels} input channels, got {got}")
        static_x = x[:, : self.static_in_channels]
        condition_x = x[:, self.static_in_channels :]
        with torch.no_grad():
            teacher_features = self.teacher.features(static_x).detach()
            teacher_static = spatial_softmax(self.teacher.head(teacher_features)).detach()
        logits = self.head(torch.cat([teacher_features, condition_x], dim=1))
        final = spatial_softmax(logits)
        temporal = final.repeat(1, self.num_bins, 1, 1)
        return {
            "temporal": temporal,
            "cumulative": temporal.clone(),
            "final": final,
            "static": teacher_static,
            "teacher": teacher_static,
            "condition_logits": logits,
        }


class FiLMStartConditionedResidualNet(nn.Module):
    """Frozen static teacher with spatial FiLM and a bounded residual correction."""

    def __init__(
        self,
        in_channels: int = 13,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        teacher_checkpoint: str | None = None,
        residual_scale: float = 1.0,
        residual_init_gate: float = -1.0,
        static_in_channels: int = 8,
        residual_hidden_channels: int = 16,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.static_in_channels = int(static_in_channels)
        self.condition_channels = int(in_channels) - self.static_in_channels
        if self.condition_channels <= 0:
            raise ValueError("FiLMStartConditionedResidualNet requires condition channels")
        self.residual_scale = float(residual_scale)
        self.teacher = SemanticStaticSaliencyNet(
            self.static_in_channels,
            base_channels,
            hidden_channels,
            self.num_bins,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if teacher_checkpoint:
            state = torch.load(Path(teacher_checkpoint), map_location="cpu")
            self.teacher.load_state_dict(state["model"])
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()
        self.condition_affine = CircularConv2d(self.condition_channels, 2 * hidden_channels, 1)
        nn.init.zeros_(self.condition_affine.conv.weight)
        nn.init.zeros_(self.condition_affine.conv.bias)
        self.residual_head = nn.Sequential(
            CircularConv2d(hidden_channels, int(residual_hidden_channels), 3),
            nn.SiLU(inplace=True),
            CircularConv2d(int(residual_hidden_channels), 1, 3),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init_gate)))

    def train(self, mode: bool = True) -> "FiLMStartConditionedResidualNet":
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_channels = self.static_in_channels + self.condition_channels
        if x.ndim != 4 or x.shape[1] != expected_channels:
            got = x.shape[1] if x.ndim >= 2 else "unknown"
            raise ValueError(f"Expected {expected_channels} input channels, got {got}")
        static_x = x[:, : self.static_in_channels]
        condition_x = x[:, self.static_in_channels :]
        with torch.no_grad():
            teacher_features = self.teacher.features(static_x).detach()
            teacher_static = spatial_softmax(self.teacher.head(teacher_features)).detach()
        gamma, beta = self.condition_affine(condition_x).chunk(2, dim=1)
        modulated = teacher_features * (1.0 + torch.tanh(gamma)) + beta
        residual_raw = self.residual_head(modulated)
        gate_value = torch.sigmoid(self.residual_gate)
        residual = self.residual_scale * gate_value * torch.tanh(residual_raw)
        final = spatial_softmax(torch.log(teacher_static.clamp_min(1e-7)) + residual)
        temporal = final.repeat(1, self.num_bins, 1, 1)
        return {
            "temporal": temporal,
            "cumulative": temporal.clone(),
            "final": final,
            "static": teacher_static,
            "teacher": teacher_static,
            "residual_logits": residual,
            "residual_gate_value": gate_value,
        }


class ImageOnlyLatentMarginalResidualNet(nn.Module):
    """Frozen static teacher plus K shared latent-condition residual experts.

    The model never consumes privileged start, scanpath, observer, or prefix input.
    Its only input is the same ERP RGB + absolute geometry tensor used by the
    locked static teacher. ``density_marginal`` performs the probabilistically
    correct latent-condition marginalization; ``logit_pool`` is the matched
    ordinary temporal-auxiliary control.
    """

    _AGGREGATIONS = {"density_marginal", "logit_pool"}

    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 3,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        teacher_checkpoint: str | None = None,
        residual_scale: float = 1.0,
        residual_init_gate: float = -1.0,
        residual_hidden_channels: int = 16,
        latent_num_conditions: int = 3,
        latent_code_channels: int = 4,
        conditioning: str = "stage_code",
        aggregation: str = "density_marginal",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_conditions = int(latent_num_conditions)
        self.num_bins = self.num_conditions
        self.latent_code_channels = int(latent_code_channels)
        self.conditioning = str(conditioning).lower()
        self.residual_scale = float(residual_scale)
        self.aggregation = str(aggregation).lower()
        if self.in_channels != 8:
            raise ValueError("ImageOnlyLatentMarginalResidualNet requires exactly 8 image + geometry input channels")
        if self.num_conditions < 2:
            raise ValueError("latent_num_conditions must be at least 2")
        if int(num_bins) != self.num_conditions:
            raise ValueError("model.num_bins must equal model.latent_num_conditions")
        if self.latent_code_channels < 1:
            raise ValueError("latent_code_channels must be positive")
        if self.conditioning not in {"stage_code", "shared_multihead"}:
            raise ValueError("model.latent_conditioning must be 'stage_code' or 'shared_multihead'")
        if self.aggregation not in self._AGGREGATIONS:
            raise ValueError(
                f"model.latent_aggregation must be one of {sorted(self._AGGREGATIONS)}, got {aggregation!r}"
            )

        self.teacher = SemanticStaticSaliencyNet(
            self.in_channels,
            base_channels,
            hidden_channels,
            self.num_conditions,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if teacher_checkpoint:
            state = torch.load(Path(teacher_checkpoint), map_location="cpu")
            self.teacher.load_state_dict(state["model"])
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()

        stage_codes = torch.zeros(self.num_conditions, self.latent_code_channels, dtype=torch.float32)
        phase = (torch.arange(self.num_conditions, dtype=torch.float32) + 0.5) / float(self.num_conditions)
        if self.latent_code_channels >= 1:
            stage_codes[:, 0] = torch.sin(2.0 * torch.pi * phase)
        if self.latent_code_channels >= 2:
            stage_codes[:, 1] = torch.cos(2.0 * torch.pi * phase)
        if self.latent_code_channels >= 3:
            stage_codes[:, 2] = 2.0 * phase - 1.0
        if self.latent_code_channels >= 4:
            stage_codes[:, 3] = 1.0
        if self.conditioning == "stage_code":
            self.stage_codes = nn.Parameter(stage_codes)
            residual_in_channels = hidden_channels + self.latent_code_channels
            residual_out_channels = 1
        else:
            self.register_buffer("stage_codes", stage_codes, persistent=True)
            residual_in_channels = hidden_channels
            residual_out_channels = self.num_conditions

        self.residual_head = nn.Sequential(
            CircularConv2d(residual_in_channels, int(residual_hidden_channels), 3),
            nn.SiLU(inplace=True),
            CircularConv2d(int(residual_hidden_channels), residual_out_channels, 3),
        )
        self.selector = nn.Linear(hidden_channels, self.num_conditions)
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init_gate)))

    def train(self, mode: bool = True) -> "ImageOnlyLatentMarginalResidualNet":
        super().train(mode)
        self.teacher.eval()
        return self

    def _cumulative(self, condition_maps: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        cumulative = []
        running = torch.zeros_like(condition_maps[:, :1])
        for index in range(self.num_conditions):
            running = running + condition_maps[:, index : index + 1] * weights[:, index, None, None, None]
            cumulative.append(normalize_torch(running))
        return torch.cat(cumulative, dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            got = x.shape[1] if x.ndim >= 2 else "unknown"
            raise ValueError(f"Expected exactly {self.in_channels} input channels, got {got}")
        with torch.no_grad():
            teacher_features = self.teacher.features(x).detach()
            teacher_static = spatial_softmax(self.teacher.head(teacher_features)).detach()

        batch, channels, height, width = teacher_features.shape
        if self.conditioning == "stage_code":
            features = teacher_features[:, None].expand(batch, self.num_conditions, channels, height, width)
            codes = self.stage_codes[None, :, :, None, None].expand(
                batch, self.num_conditions, self.latent_code_channels, height, width
            )
            residual_input = torch.cat([features, codes], dim=2).reshape(
                batch * self.num_conditions,
                channels + self.latent_code_channels,
                height,
                width,
            )
            residual_raw = self.residual_head(residual_input).reshape(batch, self.num_conditions, height, width)
        else:
            residual_raw = self.residual_head(teacher_features)
        gate_value = torch.sigmoid(self.residual_gate)
        residual = self.residual_scale * gate_value * torch.tanh(residual_raw)
        base_logits = torch.log(teacher_static.clamp_min(1e-7))
        condition_maps = spatial_softmax(base_logits + residual)

        pooled = teacher_features.mean(dim=(-2, -1))
        weights = torch.softmax(self.selector(pooled), dim=-1)
        density_marginal = (condition_maps * weights[:, :, None, None]).sum(dim=1, keepdim=True)
        if self.aggregation == "density_marginal":
            final = density_marginal
        else:
            pooled_residual = (residual * weights[:, :, None, None]).sum(dim=1, keepdim=True)
            final = spatial_softmax(base_logits + pooled_residual)

        cumulative = self._cumulative(condition_maps, weights)
        return {
            "temporal": condition_maps,
            "cumulative": cumulative,
            "final": final,
            "static": teacher_static,
            "teacher": teacher_static,
            "condition_maps": condition_maps,
            "density_marginal": density_marginal,
            "residual_logits": residual,
            "residual_gate_value": gate_value,
            "weights": weights,
            "stage_codes": self.stage_codes,
        }


class ParallelTemporalNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.features = TinyUNetFeatures(in_channels, base_channels, hidden_channels)
        self.head = nn.Sequential(ConvBlock(hidden_channels, hidden_channels), CircularConv2d(hidden_channels, num_bins, 1))
        self.logits_w = nn.Parameter(torch.zeros(num_bins), requires_grad=learned_accumulation)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        temporal = spatial_softmax(self.head(feat))
        weights = torch.softmax(self.logits_w, dim=0)
        cumulative, final = cumulative_from_temporal(temporal, weights)
        return {"temporal": temporal, "cumulative": cumulative, "final": final, "weights": weights}


class TempSALStyleMixerNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.features = TinyUNetFeatures(in_channels, base_channels, hidden_channels)
        self.temporal_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            CircularConv2d(hidden_channels, num_bins, 1),
        )
        self.mixer = nn.Sequential(
            ConvBlock(hidden_channels + num_bins * 2 + 5, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )
        self.logits_w = nn.Parameter(torch.zeros(num_bins), requires_grad=learned_accumulation)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        geo = x[:, 3:8]
        temporal = spatial_softmax(self.temporal_head(feat))
        weights = torch.softmax(self.logits_w, dim=0)
        cumulative, accumulated_final = cumulative_from_temporal(temporal, weights)
        final = spatial_softmax(self.mixer(torch.cat([feat, temporal, cumulative, geo], dim=1)))
        return {
            "temporal": temporal,
            "cumulative": cumulative,
            "final": final,
            "accumulated_final": accumulated_final,
            "weights": weights,
        }


class SemanticTempSALTemporalNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.features = SemanticFusionFeatures(
            in_channels,
            base_channels,
            hidden_channels,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        self.temporal_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            CircularConv2d(hidden_channels, num_bins, 1),
        )
        self.logits_w = nn.Parameter(torch.zeros(num_bins), requires_grad=learned_accumulation)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        temporal = spatial_softmax(self.temporal_head(feat))
        weights = torch.softmax(self.logits_w, dim=0)
        cumulative, final = cumulative_from_temporal(temporal, weights)
        out = {
            "temporal": temporal,
            "cumulative": cumulative,
            "final": final,
            "accumulated_final": final,
            "weights": weights,
        }
        if return_features:
            out["features"] = feat
        return out


class SemanticTempSALFusionNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
        semantic_backbone: str = "resnet18",
        semantic_layer: str = "layer3",
        semantic_pretrained: bool = True,
        temporal_checkpoint: str | None = None,
        static_checkpoint: str | None = None,
        freeze_temporal: bool = True,
        freeze_static: bool = True,
        shuffle_temporal_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.shuffle_temporal_inputs = bool(shuffle_temporal_inputs)
        self.temporal_branch = SemanticTempSALTemporalNet(
            in_channels,
            base_channels,
            hidden_channels,
            num_bins,
            learned_accumulation=learned_accumulation,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if temporal_checkpoint:
            state = torch.load(Path(temporal_checkpoint), map_location="cpu")
            self.temporal_branch.load_state_dict(state["model"])
        if freeze_temporal:
            for param in self.temporal_branch.parameters():
                param.requires_grad = False
            self.temporal_branch.eval()

        self.static_branch = SemanticStaticSaliencyNet(
            in_channels,
            base_channels,
            hidden_channels,
            num_bins,
            semantic_backbone=semantic_backbone,
            semantic_layer=semantic_layer,
            semantic_pretrained=semantic_pretrained,
        )
        if static_checkpoint:
            state = torch.load(Path(static_checkpoint), map_location="cpu")
            self.static_branch.load_state_dict(state["model"])
        if freeze_static:
            for param in self.static_branch.parameters():
                param.requires_grad = False
            self.static_branch.eval()

        self.mixer = nn.Sequential(
            ConvBlock(hidden_channels + 1 + num_bins * 2 + 5, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )

    def train(self, mode: bool = True) -> "SemanticTempSALFusionNet":
        super().train(mode)
        if not any(param.requires_grad for param in self.temporal_branch.parameters()):
            self.temporal_branch.eval()
        if not any(param.requires_grad for param in self.static_branch.parameters()):
            self.static_branch.eval()
        return self

    def _maybe_shuffle_temporal(self, temporal: torch.Tensor) -> torch.Tensor:
        if not self.shuffle_temporal_inputs:
            return temporal
        return temporal.flip(dims=(1,))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal_grad = any(param.requires_grad for param in self.temporal_branch.parameters())
        static_grad = any(param.requires_grad for param in self.static_branch.parameters())

        with torch.set_grad_enabled(temporal_grad):
            temporal_out = self.temporal_branch(x, return_features=True)
        with torch.set_grad_enabled(static_grad):
            static_out = self.static_branch(x)

        feat = temporal_out["features"]
        temporal = self._maybe_shuffle_temporal(temporal_out["temporal"])
        weights = temporal_out["weights"]
        cumulative, accumulated_final = cumulative_from_temporal(temporal, weights)
        static = static_out["final"]
        geo = x[:, 3:8]
        final = spatial_softmax(self.mixer(torch.cat([feat, static, temporal, cumulative, geo], dim=1)))
        return {
            "temporal": temporal,
            "cumulative": cumulative,
            "final": final,
            "static": static,
            "accumulated_final": accumulated_final,
            "weights": weights,
        }


class ResidualMixerDyTSalDMM360(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
        mixer_use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.mixer_use_temporal = mixer_use_temporal
        self.features = TinyUNetFeatures(in_channels, base_channels, hidden_channels)
        self.static_head = nn.Sequential(ConvBlock(hidden_channels, hidden_channels), CircularConv2d(hidden_channels, 1, 1))
        self.temporal_embed = nn.Parameter(torch.randn(num_bins, hidden_channels, 1, 1) * 0.02)
        self.gru = ConvGRUCell(hidden_channels + hidden_channels + 2, hidden_channels)
        self.gate = nn.Sequential(
            CircularConv2d(hidden_channels * 2 + 2, hidden_channels, 3),
            nn.SiLU(inplace=True),
            CircularConv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.residual_emit = nn.Sequential(
            ConvBlock(hidden_channels * 2 + 2, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )
        mixer_temporal_channels = num_bins * 3 if mixer_use_temporal else 0
        self.mixer = nn.Sequential(
            ConvBlock(hidden_channels + 1 + 5 + mixer_temporal_channels, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )
        self.logits_w = nn.Parameter(torch.zeros(num_bins), requires_grad=learned_accumulation)
        self.residual_gate = nn.Parameter(torch.full((num_bins,), -1.0))
        self.final_gate = nn.Parameter(torch.tensor(-0.5))
        self.eta = nn.Parameter(torch.tensor(0.25))
        self.rho = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        geo = x[:, 3:8]
        static_logits = self.static_head(feat)
        static = spatial_softmax(static_logits)
        b, _, h, w = feat.shape
        memory = torch.zeros((b, 1, h, w), dtype=feat.dtype, device=feat.device)
        h_state: torch.Tensor | None = None
        temporal = []
        residuals = []
        memory_overlap = []
        for k in range(self.num_bins):
            emb = self.temporal_embed[k].expand(b, -1, h, w)
            transition_in = torch.cat([feat, emb, memory, static], dim=1)
            h_state = self.gru(transition_in, h_state)
            emit_in = torch.cat([feat, h_state, memory, static], dim=1)
            raw_gate = self.gate(emit_in)
            eta = torch.sigmoid(self.eta)
            rho = torch.sigmoid(self.rho) * 0.25
            soft_gate = raw_gate * (1.0 - eta * memory) + rho * memory
            residual = self.residual_emit(emit_in) + torch.log(soft_gate.clamp_min(1e-6))
            residual = torch.sigmoid(self.residual_gate[k]) * residual
            sal = spatial_softmax(static_logits + residual)
            temporal.append(sal)
            residuals.append(residual)
            memory_overlap.append((sal * memory).sum(dim=(1, 2, 3)).mean())
            weights_so_far = torch.softmax(self.logits_w[: k + 1], dim=0).view(1, k + 1, 1, 1)
            memory = normalize_torch(torch.cat(temporal, dim=1).mul(weights_so_far).sum(dim=1, keepdim=True))
        temporal_t = torch.cat(temporal, dim=1)
        residual_t = torch.cat(residuals, dim=1)
        weights = torch.softmax(self.logits_w, dim=0)
        cumulative, accumulated_final = cumulative_from_temporal(temporal_t, weights)
        mixer_parts = [feat, static, geo]
        if self.mixer_use_temporal:
            mixer_parts.extend([temporal_t, cumulative, torch.tanh(residual_t)])
        final_delta = self.mixer(torch.cat(mixer_parts, dim=1))
        final = spatial_softmax(static_logits + torch.sigmoid(self.final_gate) * final_delta)
        return {
            "temporal": temporal_t,
            "cumulative": cumulative,
            "final": final,
            "static": static,
            "accumulated_final": accumulated_final,
            "residual_logits": residual_t,
            "weights": weights,
            "memory_overlap": torch.stack(memory_overlap).mean() if memory_overlap else torch.tensor(0.0, device=x.device),
        }


class DyTSalDMM360(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        hidden_channels: int = 64,
        num_bins: int = 6,
        learned_accumulation: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.features = TinyUNetFeatures(in_channels, base_channels, hidden_channels)
        self.temporal_embed = nn.Parameter(torch.randn(num_bins, hidden_channels, 1, 1) * 0.02)
        self.gru = ConvGRUCell(hidden_channels + hidden_channels + 1, hidden_channels)
        self.gate = nn.Sequential(
            CircularConv2d(hidden_channels * 2 + 1, hidden_channels, 3),
            nn.SiLU(inplace=True),
            CircularConv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.emit = nn.Sequential(
            ConvBlock(hidden_channels * 2 + 1, hidden_channels),
            CircularConv2d(hidden_channels, 1, 1),
        )
        self.logits_w = nn.Parameter(torch.zeros(num_bins), requires_grad=learned_accumulation)
        self.eta = nn.Parameter(torch.tensor(0.25))
        self.rho = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(x)
        b, _, h, w = feat.shape
        memory = torch.zeros((b, 1, h, w), dtype=feat.dtype, device=feat.device)
        h_state: torch.Tensor | None = None
        temporal = []
        memory_overlap = []
        for k in range(self.num_bins):
            emb = self.temporal_embed[k].expand(b, -1, h, w)
            transition_in = torch.cat([feat, emb, memory], dim=1)
            h_state = self.gru(transition_in, h_state)
            emit_in = torch.cat([feat, h_state, memory], dim=1)
            raw_gate = self.gate(emit_in)
            eta = torch.sigmoid(self.eta)
            rho = torch.sigmoid(self.rho) * 0.25
            soft_gate = raw_gate * (1.0 - eta * memory) + rho * memory
            logits = self.emit(emit_in) + torch.log(soft_gate.clamp_min(1e-6))
            sal = spatial_softmax(logits)
            temporal.append(sal)
            memory_overlap.append((sal * memory).sum(dim=(1, 2, 3)).mean())
            weights_so_far = torch.softmax(self.logits_w[: k + 1], dim=0).view(1, k + 1, 1, 1)
            memory = normalize_torch(torch.cat(temporal, dim=1).mul(weights_so_far).sum(dim=1, keepdim=True))
        temporal_t = torch.cat(temporal, dim=1)
        weights = torch.softmax(self.logits_w, dim=0)
        cumulative, final = cumulative_from_temporal(temporal_t, weights)
        return {
            "temporal": temporal_t,
            "cumulative": cumulative,
            "final": final,
            "weights": weights,
            "memory_overlap": torch.stack(memory_overlap).mean() if memory_overlap else torch.tensor(0.0, device=x.device),
        }
