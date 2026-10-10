
import json

import pytest

from structured_output_agent.dataset_io import load_dataset, sha256_file


def test_load_dataset_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset is empty"):
        load_dataset(path)


def test_load_dataset_rejects_blank_line(tmp_path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="blank JSONL line"):
        load_dataset(path)


def test_load_dataset_reports_invalid_json_line(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"input": }\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid dataset record"):
        load_dataset(path)


def test_sha256_matches_exact_file_bytes(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    content = b'{"example":1}\n'
    path.write_bytes(content)

    import hashlib

    expected = hashlib.sha256(content).hexdigest()

    assert sha256_file(path) == expected


def test_load_dataset_loads_valid_record(tmp_path) -> None:
    from structured_output_agent.schemas import (
        AssignedTeam,
        DatasetRecord,
        DatasetSplit,
        DatasetTag,
        ExpectedDecision,
        Priority,
        TicketCategory,
    )

    record = DatasetRecord(
        input="Where is my invoice?",
        expected=ExpectedDecision(
            category=TicketCategory.BILLING,
            priority=Priority.NORMAL,
            assigned_team=AssignedTeam.PAYMENTS,
            requires_human_review=False,
        ),
        tags={DatasetTag.CLEAN},
        author="test",
        split=DatasetSplit.DEV,
    )

    path = tmp_path / "valid.jsonl"
    path.write_text(
        json.dumps(record.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    loaded = load_dataset(path)

    assert len(loaded) == 1
    assert loaded[0] == record
