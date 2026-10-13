import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .metrics import EvaluationMetrics, compute_metrics
from .run_config import RunConfig
from .run_results import RunOutcome, TicketRunResult
from .runner import EvaluationRun, RunStatus
from .schemas import DatasetRecord, StrictModel


class EvaluationReport(StrictModel):
    """Deterministic evaluation report for a completed run."""

    schema_version: Literal["1.1"] = "1.1"
    created_at: str
    client_type: str = Field(min_length=1)
    model_id: str | None = None
    config: RunConfig
    dataset_fingerprint: str | None = None
    metrics: EvaluationMetrics
    results: list[TicketRunResult]

    @field_validator("created_at")
    @classmethod
    def validate_report_created_at(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "created_at must be a non-empty ISO 8601 timestamp string."
            )
        try:
            dt = datetime.fromisoformat(value)
        except Exception as exc:
            raise ValueError(
                f"created_at must be a valid ISO 8601 timestamp; "
                f"failed to parse {value!r}: {exc}"
            ) from exc
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError(
                f"created_at must contain an explicit timezone offset; "
                f"got naive timestamp {value!r}."
            )
        return value

    @field_validator("client_type")
    @classmethod
    def validate_client_type(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("client_type must not be empty or whitespace.")
        return stripped

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model_id must not be empty or whitespace if provided.")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def validate_report_invariants(self) -> "EvaluationReport":
        if len(self.results) != self.metrics.eligible_count:
            raise ValueError(
                f"Number of results ({len(self.results)}) does not match "
                f"metrics.eligible_count ({self.metrics.eligible_count})."
            )

        o1 = sum(
            1 for r in self.results if r.outcome == RunOutcome.EXTRACTION_FAILURE
        )
        o2 = sum(
            1 for r in self.results if r.outcome == RunOutcome.SCHEMA_FAILURE
        )
        o3 = sum(
            1 for r in self.results if r.outcome == RunOutcome.POLICY_FAILURE
        )
        o4 = sum(
            1 for r in self.results if r.outcome == RunOutcome.VALID_INCORRECT
        )
        o5 = sum(
            1 for r in self.results if r.outcome == RunOutcome.VALID_CORRECT
        )

        if o1 != self.metrics.outcomes.extraction_failure:
            raise ValueError(
                f"Results extraction_failure count ({o1}) does not match "
                f"metrics.outcomes.extraction_failure "
                f"({self.metrics.outcomes.extraction_failure})."
            )
        if o2 != self.metrics.outcomes.schema_failure:
            raise ValueError(
                f"Results schema_failure count ({o2}) does not match "
                f"metrics.outcomes.schema_failure "
                f"({self.metrics.outcomes.schema_failure})."
            )
        if o3 != self.metrics.outcomes.policy_failure:
            raise ValueError(
                f"Results policy_failure count ({o3}) does not match "
                f"metrics.outcomes.policy_failure "
                f"({self.metrics.outcomes.policy_failure})."
            )
        if o4 != self.metrics.outcomes.valid_incorrect:
            raise ValueError(
                f"Results valid_incorrect count ({o4}) does not match "
                f"metrics.outcomes.valid_incorrect "
                f"({self.metrics.outcomes.valid_incorrect})."
            )
        if o5 != self.metrics.outcomes.valid_correct:
            raise ValueError(
                f"Results valid_correct count ({o5}) does not match "
                f"metrics.outcomes.valid_correct "
                f"({self.metrics.outcomes.valid_correct})."
            )

        credit_sum = sum(r.credit for r in self.results)
        if credit_sum != self.metrics.strict_correct.count:
            raise ValueError(
                f"Sum of result credits ({credit_sum}) does not match "
                f"metrics.strict_correct.count "
                f"({self.metrics.strict_correct.count})."
            )

        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize report to a deterministic JSON string with sorted keys."""
        data = self.model_dump(mode="json")
        return json.dumps(data, indent=indent, sort_keys=True)

    def write_json(self, path: Path | str, *, indent: int | None = 2) -> None:
        """Write deterministic UTF-8 JSON report to file with trailing newline."""
        target_path = Path(path)
        content = self.to_json(indent=indent) + "\n"
        target_path.write_text(content, encoding="utf-8")

    @classmethod
    def from_json(cls, json_data: str | bytes) -> "EvaluationReport":
        """Reconstruct and validate an EvaluationReport from JSON data."""
        return cls.model_validate_json(json_data)


def generate_report(
    run: EvaluationRun,
    records: Sequence[DatasetRecord] | None = None,
    *,
    client_type: str | None = None,
    model_id: str | None = None,
    created_at: str | datetime | None = None,
) -> EvaluationReport:
    """Generate a validated evaluation report from a completed run.

    If records is None, uses the retained records snapshot in run.records.

    Raises ValueError if:
    - The run is aborted.
    - Retained or provided records do not match the run.
    """
    if run.status != RunStatus.COMPLETED:
        raise ValueError(
            f"Cannot generate report for run with status {run.status.value!r}. "
            "Aborted runs must not produce final reports."
        )

    metrics = compute_metrics(run, records)

    if client_type is not None:
        resolved_client_type = client_type
    elif run.client_type is not None:
        resolved_client_type = run.client_type
    else:
        resolved_client_type = "stub"

    resolved_model_id = model_id if model_id is not None else run.model_id
    if created_at is not None:
        if isinstance(created_at, datetime):
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            resolved_created_at = created_at.isoformat()
        else:
            resolved_created_at = str(created_at)
    elif run.created_at is not None:
        resolved_created_at = run.created_at
    else:
        resolved_created_at = datetime.now(UTC).isoformat()

    return EvaluationReport(
        schema_version="1.1",
        created_at=resolved_created_at,
        client_type=resolved_client_type,
        model_id=resolved_model_id,
        config=run.config,
        dataset_fingerprint=run.dataset_fingerprint,
        metrics=metrics,
        results=list(run.completed_results),
    )
