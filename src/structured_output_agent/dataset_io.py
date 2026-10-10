
import hashlib
import json
from pathlib import Path

from .schemas import DatasetRecord


def load_dataset(path: str | Path) -> list[DatasetRecord]:
    """Load and validate JSONL dataset records."""

    dataset_path = Path(path)
    records: list[DatasetRecord] = []

    try:
        with dataset_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    raise ValueError(
                        f"{dataset_path}:{line_number}: blank JSONL line"
                    )

                try:
                    raw_record = json.loads(line)
                    record = DatasetRecord.model_validate(raw_record)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"{dataset_path}:{line_number}: invalid dataset record"
                    ) from exc

                records.append(record)

    except OSError as exc:
        raise OSError(f"Could not read dataset: {dataset_path}") from exc

    if not records:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    return records


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hash of a file's exact bytes."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()
