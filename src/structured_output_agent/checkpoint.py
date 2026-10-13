"""Evaluation checkpoint data model for opt-in evaluation resume support.

This module defines the strict, versioned EvaluationCheckpoint schema (Stage 3.9.1).
It encapsulates the state required to safely verify and resume an evaluation run,
including dataset/prompt fingerprints, runner configuration, client provenance,
and verified contiguous prefix run results.
"""

import json
import re
from typing import Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from .run_config import RunConfig
from .run_results import TicketRunResult
from .schemas import StrictModel

# Supported checkpoint schema version
CHECKPOINT_VERSION: Literal[1] = 1

# Established fingerprint format: 64-character hexadecimal SHA-256 digest
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_sha256_hex(value: str, field_name: str) -> str:
    """Validate and normalize a 64-character hexadecimal SHA-256 digest."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty or whitespace.")
    if not _SHA256_HEX_PATTERN.fullmatch(stripped):
        raise ValueError(
            f"{field_name} must be a 64-character hexadecimal SHA-256 digest; "
            f"got {value!r}."
        )
    return stripped.lower()


class EvaluationCheckpoint(StrictModel):
    """Strict, versioned evaluation checkpoint schema for evaluation resume.

    Enforces invariants:
    1. Checkpoint version is 1 (CHECKPOINT_VERSION).
    2. Fingerprints (dataset_fingerprint, prompt_fingerprint) are 64-character
       hexadecimal SHA-256 digests.
    3. eligible_count is non-negative.
    4. len(completed_results) <= eligible_count.
    5. Completed ticket indices form a contiguous prefix beginning at zero: 0, 1, 2, ...
    6. Duplicate or out-of-order ticket indices are rejected.
    7. Unknown fields are rejected via StrictModel (extra="forbid").
    """

    checkpoint_version: Literal[1] = CHECKPOINT_VERSION
    dataset_fingerprint: str
    eligible_count: StrictInt = Field(ge=0)
    client_type: str = Field(min_length=1)
    model_id: str | None = None
    run_config: RunConfig
    prompt_fingerprint: str
    completed_results: list[TicketRunResult] = Field(default_factory=list)

    @field_validator("dataset_fingerprint")
    @classmethod
    def validate_dataset_fingerprint(cls, value: str) -> str:
        return _validate_sha256_hex(value, "dataset_fingerprint")

    @field_validator("prompt_fingerprint")
    @classmethod
    def validate_prompt_fingerprint(cls, value: str) -> str:
        return _validate_sha256_hex(value, "prompt_fingerprint")

    @field_validator("client_type")
    @classmethod
    def validate_client_type(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("client_type must be a string.")
        stripped = value.strip()
        if not stripped:
            raise ValueError("client_type must not be empty or whitespace.")
        return stripped

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str | None) -> str | None:
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("model_id must be a string if provided.")
            stripped = value.strip()
            if not stripped:
                raise ValueError(
                    "model_id must not be empty or whitespace if provided."
                )
            return stripped
        return None

    @model_validator(mode="after")
    def validate_checkpoint_invariants(self) -> "EvaluationCheckpoint":
        if len(self.completed_results) > self.eligible_count:
            raise ValueError(
                f"Number of completed results ({len(self.completed_results)}) "
                f"cannot exceed eligible_count ({self.eligible_count})."
            )

        indices = [r.ticket_index for r in self.completed_results]
        expected_indices = list(range(len(self.completed_results)))
        if indices != expected_indices:
            raise ValueError(
                f"Completed ticket indices must form a contiguous prefix "
                f"starting at 0 ({expected_indices}); got {indices}."
            )

        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize checkpoint to a deterministic JSON string with sorted keys."""
        data = self.model_dump(mode="json")
        return json.dumps(data, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_data: str | bytes) -> "EvaluationCheckpoint":
        """Reconstruct and validate an EvaluationCheckpoint from JSON data."""
        return cls.model_validate_json(json_data)
