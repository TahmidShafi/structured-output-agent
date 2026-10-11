
import json

import pytest

from structured_output_agent.dataset_io import load_dataset, sha256_file
from structured_output_agent.schemas import (
    AssignedTeam,
    DatasetRecord,
    DatasetSplit,
    DatasetTag,
    ExpectedDecision,
    Priority,
    TicketCategory,
)


def _write_records(path, *splits: DatasetSplit) -> None:
    lines = []

    for index, split in enumerate(splits):
        record = DatasetRecord(
            input=f"Ticket {index} for {split.value}.",
            expected=ExpectedDecision(
                category=TicketCategory.BILLING,
                priority=Priority.NORMAL,
                assigned_team=AssignedTeam.PAYMENTS,
                requires_human_review=False,
            ),
            tags={DatasetTag.CLEAN},
            author="test",
            split=split,
        )
        lines.append(json.dumps(record.model_dump(mode="json")))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def test_load_dataset_defaults_to_dev_split_only(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    _write_records(path, DatasetSplit.DEV, DatasetSplit.TEST)

    loaded = load_dataset(path)

    assert [record.split for record in loaded] == [DatasetSplit.DEV]


@pytest.mark.parametrize("split", [DatasetSplit.DEV, "dev"])
def test_load_dataset_accepts_explicit_dev_split(tmp_path, split) -> None:
    path = tmp_path / "mixed.jsonl"
    _write_records(path, DatasetSplit.DEV, DatasetSplit.TEST)

    loaded = load_dataset(path, split=split)

    assert [record.split for record in loaded] == [DatasetSplit.DEV]


def test_load_dataset_returns_test_split_only_when_requested(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    _write_records(path, DatasetSplit.DEV, DatasetSplit.TEST)

    loaded = load_dataset(path, split=DatasetSplit.TEST)

    assert [record.split for record in loaded] == [DatasetSplit.TEST]


def test_load_dataset_default_split_rejects_test_only_file(tmp_path) -> None:
    path = tmp_path / "test_only.jsonl"
    _write_records(path, DatasetSplit.TEST)

    with pytest.raises(ValueError, match="No records with split 'dev'"):
        load_dataset(path)


def test_load_dataset_rejects_unknown_split(tmp_path) -> None:
    path = tmp_path / "mixed.jsonl"
    _write_records(path, DatasetSplit.DEV)

    with pytest.raises(ValueError, match="Invalid dataset split"):
        load_dataset(path, split="production")


def test_load_dataset_missing_file_names_path(tmp_path) -> None:
    path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_dataset(path)

    assert str(path) in str(excinfo.value)
