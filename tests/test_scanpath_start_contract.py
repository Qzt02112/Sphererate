import numpy as np

from sphererate.baselines.scanpath_export import declared_start_xyz, xyz_to_erp_yx


def test_salient360_external_start_maps_to_x_075() -> None:
    yx = xyz_to_erp_yx(declared_start_xyz(0.0, 90.0)[None])[0]
    np.testing.assert_allclose(yx, [0.5, 0.75], atol=1e-6)


def test_aligned_sitzmann_external_start_maps_to_x_05() -> None:
    yx = xyz_to_erp_yx(declared_start_xyz(0.0, 0.0)[None])[0]
    np.testing.assert_allclose(yx, [0.5, 0.5], atol=1e-6)
