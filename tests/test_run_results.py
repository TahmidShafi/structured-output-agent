import json

import pytest
from pydantic import ValidationError

from structured_output_agent.run_results import (
    RunOutcome,
    TicketRunResult,
    classify_response,
)
from structured_output_agent.schemas import (
    AssignedTeam,
    DatasetRecord,
    DatasetSplit,
    DatasetTag,
    ExpectedDecision,
    Priority,
    TicketCategory,
    TicketTriageOutput,
)

TEAM_BY_CATEGORY = {
    TicketCategory.BILLING: AssignedTeam.PAYMENTS,
    TicketCategory.ACCOUNT_ACCESS: AssignedTeam.ACCOUNT_SUPPORT,
    TicketCategory.TECHNICAL_ISSUE: AssignedTeam.TECHNICAL_SUPPORT,
    TicketCategory.SHIPPING: AssignedTeam.FULFILLMENT,
    TicketCategory.OTHER: AssignedTeam.GENERAL_SUPPORT,
    TicketCategory.NEEDS_REVIEW: AssignedTeam.HUMAN_REVIEW,
}


def decision(
    category=TicketCategory.BILLING,
    priority=Priority.NORMAL,
    requires_human_review=False,
):
    if category == TicketCategory.NEEDS_REVIEW:
        requires_human_review = True

    return ExpectedDecision(
        category=category,
        priority=priority,
        assigned_team=TEAM_BY_CATEGORY[category],
        requires_human_review=requires_human_review,
    )


def make_record(expected, *, tags=None, label_note=None):
    return DatasetRecord(
        input="Ticket text.",
        expected=expected,
        tags=tags or {DatasetTag.CLEAN},
        author="test",
        split=DatasetSplit.DEV,
        label_note=label_note,
    )


def ambiguous_record():
    return make_record(
        [
            decision(TicketCategory.NEEDS_REVIEW, Priority.NORMAL),
            decision(TicketCategory.NEEDS_REVIEW, Priority.HIGH),
        ],
        tags={DatasetTag.AMBIGUOUS},
        label_note="Borderline between normal and high.",
    )


def response(
    *,
    category="billing",
    priority="normal",
    assigned_team="payments",
    summary="Customer asks about an invoice.",
    requires_human_review=False,
):
    return json.dumps(
        {
            "category": category,
            "priority": priority,
            "assigned_team": assigned_team,
            "summary": summary,
            "requires_human_review": requires_human_review,
        }
    )


def prediction_fixture() -> TicketTriageOutput:
    return TicketTriageOutput.model_validate(json.loads(response()))


def test_matching_prediction_is_valid_correct() -> None:
    result = classify_response(response(), make_record(decision()), 0)

    assert result.outcome == RunOutcome.VALID_CORRECT
    assert result.credit == 1
    assert result.error_type is None
    assert result.error_message is None
    assert result.prediction is not None
    assert result.prediction.category == TicketCategory.BILLING


def test_mismatched_prediction_is_valid_incorrect() -> None:
    result = classify_response(
        response(priority="high"),
        make_record(decision()),
        0,
    )

    assert result.outcome == RunOutcome.VALID_INCORRECT
    assert result.credit == 0
    assert result.error_type is None
    assert result.prediction is not None
    assert result.prediction.priority == Priority.HIGH


@pytest.mark.parametrize("raw", ["", "   \n  "])
def test_empty_response_is_extraction_failure(raw: str) -> None:
    result = classify_response(raw, make_record(decision()), 0)

    assert result.outcome == RunOutcome.EXTRACTION_FAILURE
    assert result.credit == 0
    assert result.error_type == "JSONExtractionError"
    assert result.prediction is None


def test_response_without_json_is_extraction_failure() -> None:
    result = classify_response(
        "I cannot classify this ticket.",
        make_record(decision()),
        0,
    )

    assert result.outcome == RunOutcome.EXTRACTION_FAILURE
    assert result.error_type == "JSONExtractionError"
    assert result.prediction is None


def test_competing_json_objects_are_extraction_failure() -> None:
    raw = response() + "\n" + response(priority="high")

    result = classify_response(raw, make_record(decision()), 0)

    assert result.outcome == RunOutcome.EXTRACTION_FAILURE
    assert result.error_type == "JSONExtractionError"
    assert "competing" in result.error_message
    assert result.prediction is None


def test_identical_repeated_json_objects_are_not_competing() -> None:
    raw = response() + "\n" + response()

    result = classify_response(raw, make_record(decision()), 0)

    assert result.outcome == RunOutcome.VALID_CORRECT


@pytest.mark.parametrize(
    "raw",
    [
        'Use { here. Answer: {"category": "billing"} and }',
        'Use { here. Answer: ' + response() + " and }",
    ],
)
def test_stray_brace_limitation_is_extraction_failure(raw: str) -> None:
    result = classify_response(raw, make_record(decision()), 0)

    assert result.outcome == RunOutcome.EXTRACTION_FAILURE
    assert result.error_type == "JSONExtractionError"
    assert result.prediction is None


def test_missing_field_is_schema_failure() -> None:
    payload = json.loads(response())
    del payload["summary"]

    result = classify_response(
        json.dumps(payload),
        make_record(decision()),
        0,
    )

    assert result.outcome == RunOutcome.SCHEMA_FAILURE
    assert result.credit == 0
    assert result.error_type == "ValidationError"
    assert result.error_message
    assert result.prediction is None


@pytest.mark.parametrize(
    "field_update",
    [
        {"confidence": 0.9},
        {"requires_human_review": "false"},
        {"category": "refund"},
    ],
)
def test_invalid_field_values_are_schema_failures(field_update) -> None:
    payload = json.loads(response())
    payload.update(field_update)

    result = classify_response(
        json.dumps(payload),
        make_record(decision()),
        0,
    )

    assert result.outcome == RunOutcome.SCHEMA_FAILURE
    assert result.error_type == "ValidationError"
    assert result.prediction is None


def test_validator_a_failure_is_policy_failure_with_prediction() -> None:
    result = classify_response(
        response(category="billing", assigned_team="account_support"),
        make_record(decision()),
        0,
    )

    assert result.outcome == RunOutcome.POLICY_FAILURE
    assert result.credit == 0
    assert result.error_type == "ValueError"
    assert "Validator A" in result.error_message
    assert result.prediction is not None
    assert result.prediction.assigned_team == AssignedTeam.ACCOUNT_SUPPORT


def test_validator_b_needs_review_failure_is_policy_failure() -> None:
    result = classify_response(
        response(
            category="needs_review",
            assigned_team="human_review",
            requires_human_review=False,
        ),
        ambiguous_record(),
        0,
    )

    assert result.outcome == RunOutcome.POLICY_FAILURE
    assert "Validator B: needs_review" in result.error_message
    assert result.prediction is not None
    assert result.prediction.category == TicketCategory.NEEDS_REVIEW


def test_validator_b_urgent_failure_is_policy_failure() -> None:
    result = classify_response(
        response(priority="urgent", requires_human_review=False),
        make_record(decision()),
        0,
    )

    assert result.outcome == RunOutcome.POLICY_FAILURE
    assert "Validator B: urgent" in result.error_message
    assert result.prediction is not None
    assert result.prediction.priority == Priority.URGENT


def test_multi_decision_record_gives_at_most_one_credit() -> None:
    record = ambiguous_record()

    matching_high = classify_response(
        response(
            category="needs_review",
            priority="high",
            assigned_team="human_review",
            requires_human_review=True,
        ),
        record,
        0,
    )
    matching_normal = classify_response(
        response(
            category="needs_review",
            priority="normal",
            assigned_team="human_review",
            requires_human_review=True,
        ),
        record,
        0,
    )
    matching_neither = classify_response(
        response(
            category="needs_review",
            priority="low",
            assigned_team="human_review",
            requires_human_review=True,
        ),
        record,
        0,
    )

    assert matching_high.credit == 1
    assert matching_normal.credit == 1
    assert matching_neither.outcome == RunOutcome.VALID_INCORRECT
    assert matching_neither.credit == 0


def test_ticket_index_is_stored_as_given() -> None:
    record = make_record(decision())

    assert classify_response(response(), record, 0).ticket_index == 0
    assert classify_response(response(), record, 7).ticket_index == 7


def test_negative_ticket_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        classify_response(response(), make_record(decision()), -1)


@pytest.mark.parametrize(
    "raw",
    [
        response(),
        response(priority="high"),
        response(category="billing", assigned_team="account_support"),
        "not json",
    ],
)
def test_result_round_trips_through_json(raw: str) -> None:
    result = classify_response(raw, make_record(decision()), 2)

    serialized = json.loads(json.dumps(result.model_dump(mode="json")))

    assert serialized["outcome"] == result.outcome.value
    assert TicketRunResult.model_validate(serialized) == result


def test_credit_must_be_one_only_for_valid_correct() -> None:
    with pytest.raises(ValidationError):
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=1,
            prediction=prediction_fixture(),
        )


def test_failure_outcome_requires_error_fields() -> None:
    with pytest.raises(ValidationError):
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.POLICY_FAILURE,
            credit=0,
            prediction=prediction_fixture(),
        )


def test_unexpected_scorer_error_propagates_without_model_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_scorer(prediction, records):
        raise RuntimeError("scorer failure")

    monkeypatch.setattr(
        "structured_output_agent.run_results.strict_correctness",
        failing_scorer,
    )

    with pytest.raises(RuntimeError, match="scorer failure"):
        classify_response(response(), make_record(decision()), 0)
