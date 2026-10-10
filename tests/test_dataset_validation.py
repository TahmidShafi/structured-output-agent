
import pytest

from structured_output_agent.dataset_validation import (
    validate_category_priority_coverage,
    validate_dataset_integrity,
    validate_long_ticket_tags,
    validate_unique_injection_markers,
    validate_unique_tickets,
)
from structured_output_agent.schemas import (
    AssignedTeam,
    DatasetRecord,
    DatasetSplit,
    DatasetTag,
    ExpectedDecision,
    InjectionKind,
    InjectionTarget,
    Priority,
    TicketCategory,
)


def make_decision():
    return ExpectedDecision(
        category=TicketCategory.BILLING,
        priority=Priority.NORMAL,
        assigned_team=AssignedTeam.PAYMENTS,
        requires_human_review=False,
    )


def make_record(
    *,
    input_text="I was charged twice.",
    marker=None,
    tags=None,
):
    injection_target = None

    if marker is not None:
        injection_target = InjectionTarget(
            kind=InjectionKind.FIELD_OVERRIDE,
            field="category",
            forced_value="other",
            marker=marker,
        )

    return DatasetRecord(
        input=input_text,
        expected=make_decision(),
        tags=tags if tags is not None else {DatasetTag.CLEAN},
        author="test_author",
        injection_target=injection_target,
        split=DatasetSplit.DEV,
    )


def test_unique_tickets_are_accepted():
    records = [
        make_record(input_text="I was charged twice."),
        make_record(input_text="I cannot log in."),
    ]

    validate_unique_tickets(records)


def test_exact_duplicate_tickets_are_rejected():
    records = [
        make_record(input_text="I was charged twice."),
        make_record(input_text="I was charged twice."),
    ]

    with pytest.raises(ValueError, match="Duplicate ticket input"):
        validate_unique_tickets(records)


def test_case_and_whitespace_duplicates_are_rejected():
    records = [
        make_record(input_text="I was charged twice."),
        make_record(input_text="  I WAS   CHARGED TWICE.  "),
    ]

    with pytest.raises(ValueError, match="Duplicate ticket input"):
        validate_unique_tickets(records)


def test_unique_injection_markers_are_accepted():
    records = [
        make_record(
            input_text="First ticket",
            marker="INJECT-001",
            tags={DatasetTag.INJECTION},
        ),
        make_record(
            input_text="Second ticket",
            marker="INJECT-002",
            tags={DatasetTag.INJECTION},
        ),
    ]

    validate_unique_injection_markers(records)


def test_duplicate_injection_markers_are_rejected():
    records = [
        make_record(
            input_text="First ticket",
            marker="INJECT-001",
            tags={DatasetTag.INJECTION},
        ),
        make_record(
            input_text="Second ticket",
            marker="INJECT-001",
            tags={DatasetTag.INJECTION},
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate injection marker"):
        validate_unique_injection_markers(records)


def test_records_without_targets_need_no_marker():
    records = [
        make_record(input_text="First ticket"),
        make_record(input_text="Second ticket"),
    ]

    validate_unique_injection_markers(records)


def test_combined_integrity_validation_rejects_duplicate_tickets():
    records = [
        make_record(input_text="Duplicate ticket"),
        make_record(input_text=" duplicate   ticket "),
    ]

    with pytest.raises(ValueError, match="Duplicate ticket input"):
        validate_dataset_integrity(records)


def test_combined_integrity_validation_rejects_duplicate_markers():
    records = [
        make_record(
            input_text="First ticket",
            marker="INJECT-001",
            tags={DatasetTag.INJECTION},
        ),
        make_record(
            input_text="Second ticket",
            marker="INJECT-001",
            tags={DatasetTag.INJECTION},
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate injection marker"):
        validate_dataset_integrity(records)

def test_300_word_ticket_does_not_need_long_tag():
    text = " ".join(["word"] * 300)
    records = [make_record(input_text=text)]

    validate_long_ticket_tags(records)


def test_301_word_ticket_requires_long_tag():
    text = " ".join(["word"] * 301)
    records = [make_record(input_text=text)]

    with pytest.raises(ValueError, match="missing the 'long' tag"):
        validate_long_ticket_tags(records)


def test_301_word_ticket_with_long_tag_is_accepted():
    text = " ".join(["word"] * 301)
    records = [
        make_record(
            input_text=text,
            tags={DatasetTag.LONG},
        )
    ]

    validate_long_ticket_tags(records)


def test_short_ticket_with_long_tag_is_rejected():
    records = [
        make_record(
            input_text="Only a few words.",
            tags={DatasetTag.LONG},
        )
    ]

    with pytest.raises(ValueError, match="has the 'long' tag"):
        validate_long_ticket_tags(records)


def test_integrity_validation_checks_long_tag():
    text = " ".join(["word"] * 301)
    records = [make_record(input_text=text)]

    with pytest.raises(ValueError, match="missing the 'long' tag"):
        validate_dataset_integrity(records)

def test_category_coverage_accepts_two_priorities_per_category():
    records = []

    category_team = {
        TicketCategory.BILLING: AssignedTeam.PAYMENTS,
        TicketCategory.ACCOUNT_ACCESS: AssignedTeam.ACCOUNT_SUPPORT,
        TicketCategory.TECHNICAL_ISSUE: AssignedTeam.TECHNICAL_SUPPORT,
        TicketCategory.SHIPPING: AssignedTeam.FULFILLMENT,
        TicketCategory.OTHER: AssignedTeam.GENERAL_SUPPORT,
        TicketCategory.NEEDS_REVIEW: AssignedTeam.HUMAN_REVIEW,
    }

    for category, team in category_team.items():
        for priority in (Priority.NORMAL, Priority.HIGH):
            records.append(
                DatasetRecord(
                    input=f"Ticket for {category.value} with {priority.value}",
                    expected=ExpectedDecision(
                        category=category,
                        priority=priority,
                        assigned_team=team,
                        requires_human_review=(
                            category == TicketCategory.NEEDS_REVIEW
                        ),
                    ),
                    tags={DatasetTag.CLEAN},
                    author="coverage_test",
                    split=DatasetSplit.DEV,
                )
            )

    validate_category_priority_coverage(records)


def test_category_coverage_rejects_missing_priority_variety():
    records = []

    for category, team in {
        TicketCategory.BILLING: AssignedTeam.PAYMENTS,
        TicketCategory.ACCOUNT_ACCESS: AssignedTeam.ACCOUNT_SUPPORT,
        TicketCategory.TECHNICAL_ISSUE: AssignedTeam.TECHNICAL_SUPPORT,
        TicketCategory.SHIPPING: AssignedTeam.FULFILLMENT,
        TicketCategory.OTHER: AssignedTeam.GENERAL_SUPPORT,
        TicketCategory.NEEDS_REVIEW: AssignedTeam.HUMAN_REVIEW,
    }.items():
        records.append(
            DatasetRecord(
                input=f"Ticket for {category.value}",
                expected=ExpectedDecision(
                    category=category,
                    priority=Priority.NORMAL,
                    assigned_team=team,
                    requires_human_review=(
                        category == TicketCategory.NEEDS_REVIEW
                    ),
                ),
                tags={DatasetTag.CLEAN},
                author="coverage_test",
                split=DatasetSplit.DEV,
            )
        )

    with pytest.raises(ValueError, match="at least 2 are required"):
        validate_category_priority_coverage(records)
