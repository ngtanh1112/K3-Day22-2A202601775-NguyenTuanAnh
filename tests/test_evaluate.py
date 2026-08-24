from pathlib import Path

import pytest

from preference_lab.evaluate import pairwise_accuracy, write_metrics
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy_basic() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="c1", rejected="r1"),
        PreferenceExample(prompt="p2", chosen="c2", rejected="r2"),
    ]
    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 2.0]) == 0.5


def test_pairwise_accuracy_with_ties() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="c1", rejected="r1"),
        PreferenceExample(prompt="p2", chosen="c2", rejected="r2"),
    ]
    # One win (1.0), one tie (0.5) -> (1.0 + 0.5) / 2 = 0.75
    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0], tie_value=0.5) == 0.75


def test_pairwise_accuracy_length_mismatch() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match="Length mismatch"):
        pairwise_accuracy(examples, [1.0, 2.0], [1.0])


def test_pairwise_accuracy_empty() -> None:
    assert pairwise_accuracy([], [], []) == 0.0


def test_write_metrics(tmp_path: Path) -> None:
    metrics = {"pairwise_accuracy": 0.85, "final_loss": 0.42}
    out = write_metrics(metrics, tmp_path)
    assert out.exists()
    assert "pairwise_accuracy" in out.read_text(encoding="utf-8")
