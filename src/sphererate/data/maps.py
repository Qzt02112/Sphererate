from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sphererate.utils import normalize_np


@dataclass(frozen=True)
class MapGeometry:
    height: int
    width: int

    @property
    def area_weights(self) -> np.ndarray:
        y = np.arange(self.height, dtype=np.float32) + 0.5
        phi = np.pi / 2.0 - np.pi * (y / float(self.height))
        weights = np.cos(phi).clip(min=1e-4)
        weights = weights[:, None].repeat(self.width, axis=1)
        return weights.astype(np.float32)


def coords_to_pixels(x_norm: np.ndarray, y_norm: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.floor(np.mod(x_norm, 1.0) * width).astype(np.int64)
    y = np.floor(np.clip(y_norm, 0.0, np.nextafter(1.0, 0.0)) * height).astype(np.int64)
    return x, y


def fixation_binary_map(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.float32)
    if len(x_norm) == 0:
        return out
    x, y = coords_to_pixels(x_norm, y_norm, height, width)
    out[y, x] = 1.0
    return out


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _separable_gaussian_wrap_x_edge_y(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel1d(sigma)
    radius = len(kernel) // 2

    padded_x = np.pad(image, ((0, 0), (radius, radius)), mode="wrap")
    tmp = np.zeros_like(image, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        tmp += weight * padded_x[:, offset : offset + image.shape[1]]

    padded_y = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        out += weight * padded_y[offset : offset + image.shape[0], :]
    return out


def erp_gaussian_map(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    height: int,
    width: int,
    sigma_px: float = 5.0,
    weights: np.ndarray | None = None,
    area_weights: np.ndarray | None = None,
    area_weighted: bool = True,
    eps: float = 1e-7,
) -> np.ndarray:
    impulses = np.zeros((height, width), dtype=np.float32)
    if len(x_norm) > 0:
        x, y = coords_to_pixels(x_norm, y_norm, height, width)
        vals = np.ones(len(x), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        np.add.at(impulses, (y, x), vals)
    if impulses.sum() <= eps:
        out = np.ones((height, width), dtype=np.float32)
    else:
        out = _separable_gaussian_wrap_x_edge_y(impulses, sigma=sigma_px)
    norm_weights = area_weights if area_weighted else None
    return normalize_np(out, eps=eps, weights=norm_weights)


def normalized_coords_to_unit_vectors(x_norm: np.ndarray, y_norm: np.ndarray) -> np.ndarray:
    """Convert normalized ERP coordinates to unit vectors on the sphere.

    ``x_norm`` is periodic longitude and ``y_norm`` runs north-to-south.  The
    absolute longitude origin is immaterial for an isotropic spherical kernel.
    """

    x = np.mod(np.asarray(x_norm, dtype=np.float64), 1.0)
    y = np.clip(np.asarray(y_norm, dtype=np.float64), 0.0, 1.0)
    if x.shape != y.shape:
        raise ValueError(f"x_norm and y_norm must have the same shape, got {x.shape} and {y.shape}")
    longitude = 2.0 * np.pi * x
    latitude = np.pi / 2.0 - np.pi * y
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            cos_latitude * np.cos(longitude),
            np.sin(latitude),
            cos_latitude * np.sin(longitude),
        ),
        axis=-1,
    ).astype(np.float32)


def erp_pixel_center_unit_vectors(height: int, width: int) -> np.ndarray:
    """Return an ``[H, W, 3]`` unit-vector grid at ERP pixel centres."""

    y = (np.arange(int(height), dtype=np.float64) + 0.5) / float(height)
    x = (np.arange(int(width), dtype=np.float64) + 0.5) / float(width)
    x_grid, y_grid = np.meshgrid(x, y)
    return normalized_coords_to_unit_vectors(x_grid, y_grid)


def spherical_gaussian_maps(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    height: int,
    width: int,
    sigma_deg: float,
    weights: np.ndarray | None = None,
    area_weights: np.ndarray | None = None,
    *,
    distance: str = "chord",
    truncate_sigma: float = 4.0,
    eps: float = 1e-7,
) -> np.ndarray:
    """Render one or more saliency densities with an isotropic spherical Gaussian.

    This follows the geometry of the official Salient360 toolbox: fixation and
    ERP pixels are represented as unit vectors and the default ``chord`` kernel
    is ``exp(-||p-f||^2 / (2 sigma^2))``, with ``sigma`` expressed in radians.
    A two-dimensional ``weights`` array renders several maps in one pass.

    Returned maps are densities whose area-weighted integral is one.  The
    cosine-latitude weights omit the common pixel solid-angle constant, which
    cancels during normalization and is consistent with the project metrics.
    """

    height = int(height)
    width = int(width)
    sigma_deg = float(sigma_deg)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not np.isfinite(sigma_deg) or sigma_deg <= 0.0:
        raise ValueError("sigma_deg must be finite and positive")
    if not np.isfinite(truncate_sigma) or float(truncate_sigma) <= 0.0:
        raise ValueError("truncate_sigma must be finite and positive")
    if distance not in {"chord", "great_circle"}:
        raise ValueError("distance must be 'chord' or 'great_circle'")

    x = np.asarray(x_norm, dtype=np.float32).reshape(-1)
    y = np.asarray(y_norm, dtype=np.float32).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x_norm and y_norm must have the same shape, got {x.shape} and {y.shape}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("fixation coordinates must be finite")

    if weights is None:
        weight_matrix = np.ones((1, len(x)), dtype=np.float32)
    else:
        supplied = np.asarray(weights, dtype=np.float32)
        if supplied.ndim == 1:
            weight_matrix = supplied[None, :]
        elif supplied.ndim == 2:
            weight_matrix = supplied
        else:
            raise ValueError(f"weights must be [N] or [K, N], got {supplied.shape}")
        if weight_matrix.shape[1] != len(x):
            raise ValueError(f"weights have {weight_matrix.shape[1]} points but coordinates have {len(x)}")
        if not np.all(np.isfinite(weight_matrix)) or np.any(weight_matrix < 0.0):
            raise ValueError("weights must be finite and non-negative")

    area = MapGeometry(height, width).area_weights if area_weights is None else np.asarray(area_weights, dtype=np.float32)
    if area.shape != (height, width):
        raise ValueError(f"area_weights must have shape {(height, width)}, got {area.shape}")
    output = np.zeros((weight_matrix.shape[0], height, width), dtype=np.float32)
    active = np.flatnonzero(np.any(weight_matrix > 0.0, axis=0))
    if active.size == 0:
        uniform = normalize_np(np.ones((height, width), dtype=np.float32), eps=eps, weights=area)
        return np.repeat(uniform[None], weight_matrix.shape[0], axis=0)

    grid = erp_pixel_center_unit_vectors(height, width)
    fixation_vectors = normalized_coords_to_unit_vectors(x[active], y[active])
    sigma_rad = np.deg2rad(sigma_deg)
    support_rad = min(np.pi, float(truncate_sigma) * sigma_rad)
    latitude_rows = np.pi / 2.0 - np.pi * ((np.arange(height, dtype=np.float64) + 0.5) / height)
    fixation_latitudes = np.pi / 2.0 - np.pi * np.clip(y[active].astype(np.float64), 0.0, 1.0)

    for local_index, source_index in enumerate(active.tolist()):
        row_mask = np.abs(latitude_rows - fixation_latitudes[local_index]) <= support_rad
        row_indices = np.flatnonzero(row_mask)
        if row_indices.size == 0:
            row_indices = np.asarray([int(np.argmin(np.abs(latitude_rows - fixation_latitudes[local_index])))])
        first = int(row_indices[0])
        last = int(row_indices[-1]) + 1
        dots = np.clip(
            np.einsum("hwc,c->hw", grid[first:last], fixation_vectors[local_index], optimize=True),
            -1.0,
            1.0,
        )
        angles = np.arccos(dots)
        support = angles <= support_rad
        if distance == "chord":
            squared_distance = np.maximum(2.0 - 2.0 * dots, 0.0)
        else:
            squared_distance = angles * angles
        kernel = np.exp(-squared_distance / (2.0 * sigma_rad * sigma_rad)).astype(np.float32)
        kernel *= support
        output[:, first:last, :] += weight_matrix[:, source_index, None, None] * kernel[None]

    for index in range(len(output)):
        output[index] = normalize_np(output[index], eps=eps, weights=area)
    return output


def spherical_gaussian_map(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    height: int,
    width: int,
    sigma_deg: float = 2.0,
    weights: np.ndarray | None = None,
    area_weights: np.ndarray | None = None,
    *,
    distance: str = "chord",
    truncate_sigma: float = 4.0,
    eps: float = 1e-7,
) -> np.ndarray:
    """Single-map convenience wrapper for :func:`spherical_gaussian_maps`."""

    return spherical_gaussian_maps(
        x_norm,
        y_norm,
        height,
        width,
        sigma_deg,
        weights=weights,
        area_weights=area_weights,
        distance=distance,
        truncate_sigma=truncate_sigma,
        eps=eps,
    )[0]


def sitzmann_reference_map(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    height: int,
    width: int,
    sigma_deg: float = 1.0,
    weights: np.ndarray | None = None,
    eps: float = 1e-7,
) -> np.ndarray:
    """Reproduce the public Sitzmann notebook's ERP smoothing procedure.

    This exists for provenance and numerical comparison, not as the preferred
    renderer.  In particular, the notebook wraps the latitude axis and applies
    a row-dependent horizontal sigma.  Its saliency maps use a 1-degree visual
    angle standard deviation and ordinary (not spherical-area) normalization.
    """

    coords = np.stack(
        (
            np.asarray(y_norm, dtype=np.float64).reshape(-1),
            np.asarray(x_norm, dtype=np.float64).reshape(-1),
        ),
        axis=1,
    )
    if coords.shape[0] == 0:
        return np.ones((height, width), dtype=np.float32) / float(height * width)
    vals = np.ones(len(coords), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(vals) != len(coords):
        raise ValueError("weights and coordinates must contain the same number of points")
    if np.any(vals < 0.0) or not np.all(np.isfinite(vals)):
        raise ValueError("weights must be finite and non-negative")

    # Exact discretization used in the public Analysis.ipynb.
    scale = np.asarray([height, width], dtype=np.float64)
    modulo = np.asarray([height, width], dtype=np.int64) - 1
    pixels = np.mod(np.rint(coords * scale).astype(np.int64), modulo)
    counts = np.zeros((height, width), dtype=np.float64)
    np.add.at(counts, (pixels[:, 0], pixels[:, 1]), vals)
    sigma_y = float(sigma_deg) * float(width) / 360.0
    saliency = _periodic_gaussian_filter1d(counts, sigma=sigma_y, axis=0)
    for row in range(height):
        angle = (row / float(height) - 0.5) * np.pi
        sigma_x = sigma_y / (np.cos(angle) + 1e-3)
        saliency[row] = _periodic_gaussian_filter1d(saliency[row], sigma=sigma_x, axis=0)
    total = float(saliency.sum())
    if total <= eps:
        return np.ones((height, width), dtype=np.float32) / float(height * width)
    return (saliency / total).astype(np.float32)


def _periodic_gaussian_filter1d(array: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    """NumPy equivalent of a truncated 1-D Gaussian filter with wrap mode."""

    values = np.asarray(array, dtype=np.float64)
    sigma = max(float(sigma), 1e-12)
    radius = int(4.0 * sigma + 0.5)
    if radius == 0:
        return values.copy()
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    weights = np.exp(-0.5 * (offsets.astype(np.float64) / sigma) ** 2)
    weights /= weights.sum()
    size = values.shape[int(axis)]
    periodic_kernel = np.zeros(size, dtype=np.float64)
    np.add.at(periodic_kernel, np.mod(offsets, size), weights)
    kernel_shape = [1] * values.ndim
    kernel_shape[int(axis)] = size
    spectrum = np.fft.fft(values, axis=axis)
    kernel_spectrum = np.fft.fft(periodic_kernel).reshape(kernel_shape)
    return np.fft.ifft(spectrum * kernel_spectrum, axis=axis).real


def build_temporal_maps(
    fixations: np.ndarray,
    num_bins: int,
    height: int,
    width: int,
    sigma_px: float,
    area_weighted: bool = True,
) -> dict[str, np.ndarray]:
    geom = MapGeometry(height, width)
    area = geom.area_weights
    temporal = []
    for k in range(num_bins):
        mask = fixations[:, 5].astype(np.int64) == k if len(fixations) else np.zeros(0, dtype=bool)
        pts = fixations[mask]
        temporal.append(
            erp_gaussian_map(
                pts[:, 0] if len(pts) else np.array([], dtype=np.float32),
                pts[:, 1] if len(pts) else np.array([], dtype=np.float32),
                height,
                width,
                sigma_px=sigma_px,
                weights=pts[:, 4] if len(pts) else None,
                area_weights=area,
                area_weighted=area_weighted,
            )
        )
    temporal_arr = np.stack(temporal, axis=0).astype(np.float32)

    cumulative = []
    running = np.zeros((height, width), dtype=np.float32)
    for k in range(num_bins):
        running = running + temporal_arr[k]
        cumulative.append(normalize_np(running, weights=area if area_weighted else None))
    cumulative_arr = np.stack(cumulative, axis=0).astype(np.float32)

    final_map = erp_gaussian_map(
        fixations[:, 0] if len(fixations) else np.array([], dtype=np.float32),
        fixations[:, 1] if len(fixations) else np.array([], dtype=np.float32),
        height,
        width,
        sigma_px=sigma_px,
        weights=fixations[:, 4] if len(fixations) else None,
        area_weights=area,
        area_weighted=area_weighted,
    )
    fixation_map = fixation_binary_map(
        fixations[:, 0] if len(fixations) else np.array([], dtype=np.float32),
        fixations[:, 1] if len(fixations) else np.array([], dtype=np.float32),
        height,
        width,
    )
    return {
        "temporal": temporal_arr,
        "cumulative": cumulative_arr,
        "final": final_map.astype(np.float32),
        "fixation": fixation_map.astype(np.float32),
        "area_weights": area.astype(np.float32),
    }
