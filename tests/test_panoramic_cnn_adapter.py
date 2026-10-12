import torch

from sphererate.baselines.panoramic_cnn import resize_spherical_density, spherical_area_tensor


def test_resize_spherical_density_normalizes_area() -> None:
    source = torch.rand(2, 32, 64)
    resized = resize_spherical_density(source, 16, 32)
    area = spherical_area_tensor(16, 32)
    torch.testing.assert_close((resized * area).sum(dim=(-2, -1)), torch.ones(2), atol=1e-5, rtol=1e-5)
