from pathlib import Path

import pytest

from structured_output_agent.baselines import (
    always_needs_review_decision,
    majority_decision,
    strict_correctness,
)
from structured_output_agent.dataset_io import load_dataset
from structured_output_agent.dataset_validation import validate_dataset_integrity
from structured_output_agent.schemas import (
    AssignedTeam,
    DatasetRecord,
    DatasetSplit,
    DatasetTag,
    ExpectedDecision,
    Priority,
    TicketCategory,
)

DEV_DATASET = Path(__file__).resolve().parents[1] / "evals" / "data" / "dev.jsonl"

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


def test_majority_breaks_ties_by_category_alphabetically():
    records = [
        make_record(decision(TicketCategory.BILLING)),
        make_record(decision(TicketCategory.ACCOUNT_ACCESS)),
    ]

    assert majority_decision(records).category == TicketCategory.ACCOUNT_ACCESS


def test_majority_breaks_ties_by_priority_order_not_alphabetically():
    records = [
        make_record(decision(TicketCategory.BILLING, Priority.HIGH)),
        make_record(decision(TicketCategory.BILLING, Priority.LOW)),
    ]

    assert majority_decision(records).priority == Priority.LOW


def test_majority_breaks_ties_with_false_before_true_review_flag():
    records = [
        make_record(
            decision(TicketCategory.BILLING, requires_human_review=True)
        ),
        make_record(decision(TicketCategory.BILLING)),
    ]

    assert majority_decision(records).requires_human_review is False


def test_majority_uses_fractional_weights_for_multi_decision_records():
    records = [
        ambiguous_record(),
        ambiguous_record(),
        ambiguous_record(),
        make_record(decision(TicketCategory.BILLING)),
        make_record(decision(TicketCategory.BILLING)),
    ]

    result = majority_decision(records)

    assert result.decision_key() == ("billing", "normal", "payments", False)


@pytest.mark.parametrize(
    "field_update",
    [
        {"category": TicketCategory.OTHER},
        {"priority": Priority.HIGH},
        {"assigned_team": AssignedTeam.TECHNICAL_SUPPORT},
        {"requires_human_review": True},
    ],
)
def test_strict_correctness_requires_every_field_to_match(field_update):
    expected = decision(TicketCategory.BILLING, Priority.NORMAL)
    record = make_record(expected)
    mismatched = expected.model_copy(update=field_update)

    assert strict_correctness(expected, [record]) == [True]
    assert strict_correctness(mismatched, [record]) == [False]


def test_multi_decision_record_contributes_at_most_one_result():
    record = ambiguous_record()

    matching_second = decision(TicketCategory.NEEDS_REVIEW, Priority.HIGH)
    matching_first = decision(TicketCategory.NEEDS_REVIEW, Priority.NORMAL)

    assert strict_correctness(matching_second, [record]) == [True]
    assert strict_correctness(matching_first, [record]) == [True]


def test_strict_correctness_returns_one_flag_per_record():
    records = [
        ambiguous_record(),
        make_record(decision(TicketCategory.BILLING)),
    ]

    result = strict_correctness(decision(TicketCategory.BILLING), records)

    assert result == [False, True]


def test_always_needs_review_decision_is_complete_review_decision():
    assert always_needs_review_decision().decision_key() == (
        "needs_review",
        "normal",
        "human_review",
        True,
    )


def test_majority_rejects_empty_records():
    with pytest.raises(ValueError, match="at least one eligible record"):
        majority_decision([])


def test_strict_correctness_rejects_empty_records():
    with pytest.raises(ValueError, match="at least one eligible record"):
        strict_correctness(always_needs_review_decision(), [])


def test_dev_baselines_reproduce_documented_scores():
    records = load_dataset(DEV_DATASET)
    validate_dataset_integrity(records)

    assert len(records) == 20

    majority = majority_decision(records)
    majority_flags = strict_correctness(majority, records)

    assert majority.decision_key() == (
        "account_access",
        "normal",
        "account_support",
        False,
    )
    assert sum(majority_flags) == 2

    always_flags = strict_correctness(always_needs_review_decision(), records)

    assert sum(always_flags) == 4
