from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample


def load_jsonl(path: str | Path, check_duplicates: bool = True) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Includes line-numbered error messages, schema validation, and duplicate prompt checks.
    """
    file_path = Path(path)
    examples: list[PreferenceExample] = []
    seen_prompts: set[str] = set()

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_no}: Invalid JSON - {exc}") from exc

            try:
                example = PreferenceExample.model_validate(payload)
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"{file_path}:{line_no}: Invalid schema - {exc}") from exc

            if check_duplicates and example.prompt in seen_prompts:
                raise ValueError(
                    f"{file_path}:{line_no}: Duplicate prompt detected - '{example.prompt}'"
                )

            seen_prompts.add(example.prompt)
            examples.append(example)

    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid data leakage.

    Groups examples by prompt and uses deterministic shuffling.
    """
    if not examples:
        return [], []

    grouped: dict[str, list[PreferenceExample]] = defaultdict(list)
    for example in examples:
        grouped[example.prompt].append(example)

    unique_prompts = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)

    total_prompts = len(unique_prompts)
    if total_prompts == 1:
        if validation_ratio >= 1.0:
            return [], list(examples)
        return list(examples), []

    val_count = max(1, round(total_prompts * validation_ratio)) if validation_ratio > 0 else 0
    train_count = total_prompts - val_count

    train_prompts = unique_prompts[:train_count]
    val_prompts = unique_prompts[train_count:]

    train_examples: list[PreferenceExample] = []
    for prompt in train_prompts:
        train_examples.extend(grouped[prompt])

    val_examples: list[PreferenceExample] = []
    for prompt in val_prompts:
        val_examples.extend(grouped[prompt])

    return train_examples, val_examples
