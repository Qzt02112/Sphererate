# Configuration Status

The active ICASSP line is kernel v2. Start from these files only:

- `early_horizon_spherical2deg_v2.yaml`
- `eh3_template_residual_spherical2deg_v2.yaml`
- `batch_eh3_pilot_spherical2deg_v2.yaml`
- `kernel_v2_matched_baselines.yaml`
- `etdk2_256x512.yaml`

Other configs are retained as historical provenance. Many reference generated datasets or checkpoints removed during the 2026-07-13 cleanup and are not runnable without an explicit rebuild. A filename without `kernel_v2` or `spherical2deg_v2` must not be interpreted as current merely because it remains in this directory.

See `RESEARCH_PROTOCOL.md` for the authoritative route and `docs/workspace_cleanup_20260713.md` for the deletion record.
