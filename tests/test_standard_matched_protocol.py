from pathlib import Path

from sphererate.baselines.matched_protocol import load_matched_protocol


def test_standard_matched_protocol_uses_repeatable_test_reporting() -> None:
    protocol = load_matched_protocol(Path("configs/standard_split/matched_baselines.yaml"))

    assert protocol.expected_content_counts == {"train": 59, "val": 13, "test": 13}
    assert protocol.evaluation_splits == ("val", "test")
    assert protocol.sealed_splits == ()
    assert protocol.train_split == "train"
    assert protocol.selection_split == "val"
