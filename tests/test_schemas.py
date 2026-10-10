
import json

import pytest
from pydantic import ValidationError

from structured_output_agent.exceptions import InvalidTicketInput
from structured_output_agent.schemas import (
    CATEGORY_TEAM_MAP,
    ExpectedDecision,
    TicketCategory,
    TicketTriageOutput,
    parse_ticket_input,
    validate_decision_policy,
)

POLICY_CATEGORY_TEAM_TABLE = {
    "billing": "payments",
    "account_access": "account_support",
    "technical_issue": "technical_support",
    "shipping": "fulfillment",
    "other": "general_support",
    "needs_review": "human_review",
}


def valid_output(**overrides):
    data = {
        "category": "billing",
        "priority": "normal",
        "assigned_team": "payments",
        "summary": "Customer reports a billing issue.",
        "requires_human_review": False,
    }
    data.update(overrides)
    return data


def test_json_loads_enum_strings_accepted():
    raw = json.loads(
        """
        {
            "category": "billing",
            "priority": "normal",
            "assigned_team": "payments",
            "summary": "Customer reports a billing issue.",
            "requires_human_review": false
        }
        """
    )

    result = TicketTriageOutput.model_validate(raw)

    assert result.category == TicketCategory.BILLING


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        TicketTriageOutput.model_validate(
            valid_output(unexpected_field="not allowed")
        )


def test_invalid_enum_rejected():
    with pytest.raises(ValidationError):
        TicketTriageOutput.model_validate(
            valid_output(priority="critical")
        )


def test_string_true_rejected_for_boolean():
    with pytest.raises(ValidationError):
        TicketTriageOutput.model_validate(
            valid_output(requires_human_review="true")
        )


def test_summary_280_characters_accepted():
    result = TicketTriageOutput.model_validate(
        valid_output(summary="x" * 280)
    )
    assert len(result.summary) == 280


def test_summary_281_characters_rejected():
    with pytest.raises(ValidationError):
        TicketTriageOutput.model_validate(
            valid_output(summary="x" * 281)
        )


def test_whitespace_summary_rejected():
    with pytest.raises(ValidationError):
        TicketTriageOutput.model_validate(
            valid_output(summary="   ")
        )


def test_empty_ticket_raises_typed_exception():
    with pytest.raises(InvalidTicketInput):
        parse_ticket_input("   ")


def test_valid_ticket_input_is_trimmed():
    result = parse_ticket_input("  I was charged twice.  ")
    assert result.input == "I was charged twice."


def test_category_team_map_matches_policy_table():
    assert CATEGORY_TEAM_MAP == POLICY_CATEGORY_TEAM_TABLE


def test_validator_a_rejects_wrong_team():
    decision = ExpectedDecision(
        category="billing",
        priority="normal",
        assigned_team="fulfillment",
        requires_human_review=False,
    )

    with pytest.raises(ValueError, match="Validator A"):
        validate_decision_policy(decision)


def test_urgent_requires_human_review():
    decision = ExpectedDecision(
        category="billing",
        priority="urgent",
        assigned_team="payments",
        requires_human_review=False,
    )

    with pytest.raises(ValueError, match="Validator B"):
        validate_decision_policy(decision)


def test_needs_review_requires_human_review():
    decision = ExpectedDecision(
        category="needs_review",
        priority="normal",
        assigned_team="human_review",
        requires_human_review=False,
    )

    with pytest.raises(ValueError, match="Validator B"):
        validate_decision_policy(decision)
