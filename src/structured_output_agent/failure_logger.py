"""Structured JSONL failure logging for evaluation retry attempts.

Logs:
- Outcome transitions (e.g., O2 -> O5, O2 -> O2, O3 -> O5).
- Attempt counts and recovery status.
- Safe, redacted error details.
- No secrets, credentials, or sensitive ticket content.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field, StrictBool, StrictInt

from .feedback import redact_secrets
from .run_results import RunOutcome
from .schemas import StrictModel

OUTCOME_SHORT_NAMES: dict[RunOutcome, str] = {
    RunOutcome.EXTRACTION_FAILURE: "O1",
    RunOutcome.SCHEMA_FAILURE: "O2",
    RunOutcome.POLICY_FAILURE: "O3",
    RunOutcome.VALID_INCORRECT: "O4",
    RunOutcome.VALID_CORRECT: "O5",
}


def to_outcome_code(outcome: RunOutcome | str) -> str:
    """Map a RunOutcome or string to its short code (O1-O5) if known."""
    if isinstance(outcome, RunOutcome):
        return OUTCOME_SHORT_NAMES.get(outcome, str(outcome))
    try:
        val = RunOutcome(outcome)
        return OUTCOME_SHORT_NAMES.get(val, str(outcome))
    except ValueError:
        return str(outcome)


class AttemptRecord(StrictModel):
    attempt_index: StrictInt = Field(ge=1)
    outcome: str
    error_type: str | None = None
    safe_error_message: str | None = None


class FailureLogEntry(StrictModel):
    timestamp: str
    ticket_index: StrictInt = Field(ge=0)
    attempt_count: StrictInt = Field(ge=1)
    initial_outcome: str
    final_outcome: str
    outcome_transition: str
    recovered: StrictBool
    error_type: str | None = None
    safe_error_message: str | None = None
    attempts: list[AttemptRecord] = Field(default_factory=list)


def create_failure_log_entry(
    ticket_index: int,
    attempts: list[dict[str, Any]],
    timestamp: str | None = None,
) -> FailureLogEntry:
    """Create a FailureLogEntry from a sequence of attempt records."""
    if not attempts:
        raise ValueError("attempts list must not be empty.")

    ts = timestamp or datetime.now(UTC).isoformat()
    first_outcome_raw = attempts[0]["outcome"]
    last_outcome_raw = attempts[-1]["outcome"]

    first_code = to_outcome_code(first_outcome_raw)
    last_code = to_outcome_code(last_outcome_raw)
    transition = f"{first_code} -> {last_code}"

    # Recovered is True if initial was an eligible failure and final is valid (O4 or O5)
    recovered = first_code in ("O1", "O2", "O3") and last_code in ("O4", "O5")

    last_error_type = attempts[-1].get("error_type")
    last_error_msg = attempts[-1].get("error_message")
    safe_error = redact_secrets(last_error_msg) if last_error_msg else None

    attempt_records = [
        AttemptRecord(
            attempt_index=att.get("attempt_index", idx + 1),
            outcome=to_outcome_code(att["outcome"]),
            error_type=att.get("error_type"),
            safe_error_message=redact_secrets(att.get("error_message") or "")
            if att.get("error_message")
            else None,
        )
        for idx, att in enumerate(attempts)
    ]

    return FailureLogEntry(
        timestamp=ts,
        ticket_index=ticket_index,
        attempt_count=len(attempts),
        initial_outcome=first_code,
        final_outcome=last_code,
        outcome_transition=transition,
        recovered=recovered,
        error_type=last_error_type,
        safe_error_message=safe_error,
        attempts=attempt_records,
    )


def append_failure_log(
    entry: FailureLogEntry,
    log_path: Path | str,
) -> None:
    """Atomically append a FailureLogEntry to a JSONL file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
