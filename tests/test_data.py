from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_error_message_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt":"a","chosen":"b","rejected":"c"}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="2"):
        load_jsonl(bad)


def test_error_message_on_invalid_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad_schema.jsonl"
    bad.write_text('{"prompt":"a","chosen":"same","rejected":"same"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="1"):
        load_jsonl(bad)


def test_duplicate_prompt_detected(tmp_path: Path) -> None:
    dup = tmp_path / "dup.jsonl"
    dup.write_text(
        '{"prompt":"What is AI?","chosen":"A","rejected":"B"}\n'
        '{"prompt":"What is AI?","chosen":"C","rejected":"D"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate prompt"):
        load_jsonl(dup)


def test_split_has_no_prompt_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5, seed=42)
    assert len(train) + len(val) == len(examples)
    assert not ({e.prompt for e in train} & {e.prompt for e in val})


def test_schema_whitespace_and_case_insensitivity() -> None:
    with pytest.raises(ValueError, match="chosen and rejected must differ"):
        PreferenceExample(prompt="p", chosen="  Exact Same Answer  ", rejected="exact same answer")
