from enum import StrEnum
from typing import Literal

from pydantic import Field, StrictInt, ValidationError, model_validator

from .baselines import strict_correctness
from .exceptions import JSONExtractionError
from .json_extractor import extract_json_object
from .schemas import (
    DatasetRecord,
    ExpectedDecision,
    StrictModel,
    TicketTriageOutput,
    validate_decision_policy,
)


class RunOutcome(StrEnum):
    EXTRACTION_FAILURE = "extraction_failure"
    SCHEMA_FAILURE = "schema_failure"
    POLICY_FAILURE = "policy_failure"
    VALID_INCORRECT = "valid_incorrect"
    VALID_CORRECT = "valid_correct"


FAILURE_OUTCOMES = frozenset(
    {
        RunOutcome.EXTRACTION_FAILURE,
        RunOutcome.SCHEMA_FAILURE,
        RunOutcome.POLICY_FAILURE,
    }
)

OUTCOMES_WITHOUT_PREDICTION = frozenset(
    {
        RunOutcome.EXTRACTION_FAILURE,
        RunOutcome.SCHEMA_FAILURE,
    }
)


class TicketRunResult(StrictModel):
    ticket_index: StrictInt = Field(ge=0)
    outcome: RunOutcome
    credit: Literal[0, 1]
    error_type: str | None = None
    error_message: str | None = None
    prediction: TicketTriageOutput | None = None
    raw_response: str | None = None
    attempts: StrictInt = Field(default=1, ge=1)
    raw_responses: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_consistency(self) -> "TicketRunResult":
        expected_credit = 1 if self.outcome == RunOutcome.VALID_CORRECT else 0
        if self.credit != expected_credit:
            raise ValueError("credit is 1 only for valid_correct outcomes.")

        is_failure = self.outcome in FAILURE_OUTCOMES
        if is_failure != (self.error_type is not None):
            raise ValueError("error_type is set only for failure outcomes.")
        if is_failure != (self.error_message is not None):
            raise ValueError("error_message is set only for failure outcomes.")

        has_prediction = self.prediction is not None
        if has_prediction == (self.outcome in OUTCOMES_WITHOUT_PREDICTION):
            raise ValueError(
                "prediction is absent only for extraction and schema failures."
            )

        if not self.raw_responses and self.raw_response is not None:
            self.raw_responses = [self.raw_response]
        elif self.raw_responses and self.raw_response is None:
            self.raw_response = self.raw_responses[-1]

        return self


def _failed(
    ticket_index: int,
    outcome: RunOutcome,
    error: Exception,
    prediction: TicketTriageOutput | None = None,
    raw_response: str | None = None,
    attempts: int = 1,
    raw_responses: list[str] | None = None,
) -> TicketRunResult:
    responses = raw_responses or ([raw_response] if raw_response is not None else [])
    return TicketRunResult(
        ticket_index=ticket_index,
        outcome=outcome,
        credit=0,
        error_type=type(error).__name__,
        error_message=str(error),
        prediction=prediction,
        raw_response=raw_response,
        attempts=attempts,
        raw_responses=responses,
    )


def classify_response(
    raw: str,
    record: DatasetRecord,
    ticket_index: int,
    *,
    attempts: int = 1,
    raw_responses: list[str] | None = None,
) -> TicketRunResult:
    all_responses = raw_responses or ([raw] if raw is not None else [])
    try:
        payload = extract_json_object(raw)
    except JSONExtractionError as error:
        return _failed(
            ticket_index,
            RunOutcome.EXTRACTION_FAILURE,
            error,
            raw_response=raw,
            attempts=attempts,
            raw_responses=all_responses,
        )

    try:
        prediction = TicketTriageOutput.model_validate(payload)
    except ValidationError as error:
        return _failed(
            ticket_index,
            RunOutcome.SCHEMA_FAILURE,
            error,
            raw_response=raw,
            attempts=attempts,
            raw_responses=all_responses,
        )

    try:
        validate_decision_policy(prediction)
    except ValueError as error:
        return _failed(
            ticket_index,
            RunOutcome.POLICY_FAILURE,
            error,
            prediction=prediction,
            raw_response=raw,
            attempts=attempts,
            raw_responses=all_responses,
        )

    decision = ExpectedDecision(
        category=prediction.category,
        priority=prediction.priority,
        assigned_team=prediction.assigned_team,
        requires_human_review=prediction.requires_human_review,
    )
    (is_correct,) = strict_correctness(decision, [record])

    return TicketRunResult(
        ticket_index=ticket_index,
        outcome=(
            RunOutcome.VALID_CORRECT if is_correct else RunOutcome.VALID_INCORRECT
        ),
        credit=1 if is_correct else 0,
        prediction=prediction,
        raw_response=raw,
        attempts=attempts,
        raw_responses=all_responses,
    )
