# Script Status

Current work should follow `RESEARCH_PROTOCOL.md` and use scripts whose names contain `kernel_v2`, plus the shared early-horizon and ET-DK2 utilities they invoke.

Legacy F2, JUFE, Multi360, R0/R1/P1/P2, and EH-v1 scripts remain only to explain or regenerate historical reports. Their generated data/checkpoints were removed on 2026-07-13, so their old default paths may intentionally be unavailable. Do not reopen a closed route from a legacy script without first revising the protocol.

The old Sitzmann frozen-transfer evaluator was removed entirely because it inferred pseudo-starts after the local conversion had discarded official `viewpoint_idx` metadata. The replacement pipeline is `prepare_sitzmann_official.py -> audit_sitzmann_training_suitability.py -> compare_sitzmann_salient360_train_sources.py`; it reads official standing-VR pickles and keeps all four declared viewpoints.
