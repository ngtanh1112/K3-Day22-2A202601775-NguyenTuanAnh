from __future__ import annotations

import json
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
    tie_value: float = 0.5,
) -> float:
    """Return fraction where chosen score is greater than rejected score.

    Validates input lengths and handles score ties explicitly.
    """
    if not examples:
        return 0.0

    if len(examples) != len(chosen_scores) or len(examples) != len(rejected_scores):
        raise ValueError(
            f"Length mismatch: {len(examples)} examples, "
            f"{len(chosen_scores)} chosen_scores, "
            f"{len(rejected_scores)} rejected_scores"
        )

    score_total = sum(
        1.0 if c > r else (tie_value if c == r else 0.0)
        for c, r in zip(chosen_scores, rejected_scores, strict=True)
    )
    return score_total / len(examples)


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
