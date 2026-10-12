from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch.nn import functional as F


SOURCE_ROOT = Path("third_party/Panoramic-CNN-360-Saliency")


def _sphere_conv_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
        self.grid_shape = tuple(x.shape[2:4])
        coordinates = self._eh6_gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
        self.grid = torch.as_tensor(coordinates, dtype=torch.float32, device=x.device)
    grid = self.grid.repeat(x.shape[0], 1, 1, 1)
    sampled = F.grid_sample(x, grid, mode=self.mode, align_corners=True)
    return F.conv2d(sampled, self.weight, self.bias, stride=3)


def _sphere_pool_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
        self.grid_shape = tuple(x.shape[2:4])
        coordinates = self._eh6_gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
        self.grid = torch.as_tensor(coordinates, dtype=torch.float32, device=x.device)
    grid = self.grid.repeat(x.shape[0], 1, 1, 1)
    return self.pool(F.grid_sample(x, grid, mode=self.mode, align_corners=True))


def build_official_panoramic_cnn(source_root: str | Path = SOURCE_ROOT) -> torch.nn.Module:
    """Load the published SphereNet while preserving its PyTorch-1.2 sampling semantics."""
    source = Path(source_root).resolve()
    sphere_path = source / "spherenet" / "sphere_cnn.py"
    spec = importlib.util.spec_from_file_location("_eh6_panoramic_sphere_cnn", sphere_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {sphere_path}")
    sphere = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sphere)
    sphere.SphereConv2D._eh6_gen_grid_coordinates = staticmethod(sphere.gen_grid_coordinates)
    sphere.SphereMaxPool2D._eh6_gen_grid_coordinates = staticmethod(sphere.gen_grid_coordinates)
    sphere.SphereConv2D.forward = _sphere_conv_forward
    sphere.SphereMaxPool2D.forward = _sphere_pool_forward
    shim = types.ModuleType("spherenet")
    shim.SphereConv2D = sphere.SphereConv2D
    shim.SphereMaxPool2D = sphere.SphereMaxPool2D
    previous = sys.modules.get("spherenet")
    sys.modules["spherenet"] = shim
    sys.path.insert(0, str(source))
    try:
        model_spec = importlib.util.spec_from_file_location("_eh6_panoramic_model", source / "spherenet_model.py")
        if model_spec is None or model_spec.loader is None:
            raise ImportError("Cannot load published spherenet_model.py")
        module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(module)
        model = module.SphereNet()
    finally:
        sys.path.remove(str(source))
        if previous is None:
            sys.modules.pop("spherenet", None)
        else:
            sys.modules["spherenet"] = previous
    return model


def spherical_area_tensor(height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
    y = torch.arange(int(height), dtype=torch.float32, device=device) + 0.5
    weights = torch.cos(torch.pi / 2.0 - torch.pi * y / float(height)).clamp_min(1e-4)
    return weights[:, None].repeat(1, int(width))


def resize_spherical_density(density: torch.Tensor, height: int, width: int) -> torch.Tensor:
    original_ndim = density.ndim
    current = density
    if current.ndim == 2:
        current = current[None, None]
    elif current.ndim == 3:
        current = current[:, None]
    resized = F.interpolate(current.float(), size=(int(height), int(width)), mode="bilinear", align_corners=False)
    area = spherical_area_tensor(int(height), int(width), resized.device).view(1, 1, int(height), int(width))
    resized = resized.clamp_min(0.0) / (resized.clamp_min(0.0) * area).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    if original_ndim == 2:
        return resized[0, 0]
    if original_ndim == 3:
        return resized[:, 0]
    return resized
