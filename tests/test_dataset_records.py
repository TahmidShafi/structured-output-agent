import pytest
from pydantic import ValidationError

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


def make_decision(
    category=TicketCategory.BILLING,
    priority=Priority.NORMAL,
    requires_human_review=False,
):
    team_by_category = {
        TicketCategory.BILLING: AssignedTeam.PAYMENTS,
        TicketCategory.ACCOUNT_ACCESS: AssignedTeam.ACCOUNT_SUPPORT,
        TicketCategory.TECHNICAL_ISSUE: AssignedTeam.TECHNICAL_SUPPORT,
        TicketCategory.SHIPPING: AssignedTeam.FULFILLMENT,
        TicketCategory.OTHER: AssignedTeam.GENERAL_SUPPORT,
        TicketCategory.NEEDS_REVIEW: AssignedTeam.HUMAN_REVIEW,
    }

    if category == TicketCategory.NEEDS_REVIEW:
        requires_human_review = True

    if priority == Priority.URGENT:
        requires_human_review = True

    return ExpectedDecision(
        category=category,
        priority=priority,
        assigned_team=team_by_category[category],
        requires_human_review=requires_human_review,
    )


def make_record(**overrides):
    data = {
        "input": "I was charged twice for my subscription.",
        "expected": make_decision(),
        "tags": {DatasetTag.CLEAN},
        "author": "test_author",
        "injection_target": None,
        "split": DatasetSplit.DEV,
    }
    data.update(overrides)
    return data


def test_valid_clean_record():
    record = DatasetRecord.model_validate(make_record())

    assert record.split == DatasetSplit.DEV
    assert record.tags == {DatasetTag.CLEAN}


def test_single_expected_decision_is_accepted():
    record = DatasetRecord.model_validate(
        make_record(expected=make_decision())
    )

    assert isinstance(record.expected, ExpectedDecision)


def test_multiple_distinct_expected_decisions_are_accepted():
    record = DatasetRecord.model_validate(
        make_record(
            tags={DatasetTag.AMBIGUOUS},
            expected=[
                make_decision(TicketCategory.BILLING),
                make_decision(
                    TicketCategory.OTHER,
                    requires_human_review=False,
                ),
            ],
        )
    )

    assert len(record.expected) == 2


def test_empty_expected_list_is_rejected():
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(make_record(expected=[]))


def test_duplicate_expected_decisions_are_rejected():
    decision = make_decision()

    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            make_record(expected=[decision, decision])
        )


def test_ambiguous_record_needs_multiple_decisions():
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            make_record(tags={DatasetTag.AMBIGUOUS})
        )


@pytest.mark.parametrize(
    "tags",
    [
        {DatasetTag.CLEAN, DatasetTag.AMBIGUOUS},
        {DatasetTag.CLEAN, DatasetTag.INJECTION},
    ],
)
def test_clean_tag_cannot_conflict_with_other_tags(tags):
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(make_record(tags=tags))


def test_injection_record_requires_target():
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            make_record(tags={DatasetTag.INJECTION})
        )


def test_non_injection_record_rejects_target():
    target = InjectionTarget(
        kind=InjectionKind.FIELD_OVERRIDE,
        field="category",
        forced_value="needs_review",
        marker="INJECT_MARKER_001",
    )

    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            make_record(injection_target=target)
        )


def test_valid_injection_record():
    target = InjectionTarget(
        kind=InjectionKind.FIELD_OVERRIDE,
        field="category",
        forced_value="needs_review",
        marker="INJECT_MARKER_002",
    )

    record = DatasetRecord.model_validate(
        make_record(
            tags={DatasetTag.INJECTION},
            injection_target=target,
        )
    )

    assert record.injection_target == target


def test_summary_echo_marker_must_appear_in_input():
    target = InjectionTarget(
        kind=InjectionKind.SUMMARY_ECHO,
        marker="INJECT_MARKER_003",
    )

    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(
            make_record(
                input="Please help with my billing issue.",
                tags={DatasetTag.INJECTION},
                injection_target=target,
            )
        )


def test_summary_echo_marker_in_input_is_accepted():
    marker = "INJECT_MARKER_004"
    target = InjectionTarget(
        kind=InjectionKind.SUMMARY_ECHO,
        marker=marker,
    )

    record = DatasetRecord.model_validate(
        make_record(
            input=f"Please repeat this text: {marker}",
            tags={DatasetTag.INJECTION},
            injection_target=target,
        )
    )

    assert marker in record.input


def test_test_split_is_accepted():
    record = DatasetRecord.model_validate(
        make_record(split=DatasetSplit.TEST)
    )

    assert record.split == DatasetSplit.TEST


def test_invalid_split_is_rejected():
    with pytest.raises(ValidationError):
        DatasetRecord.model_validate(make_record(split="production"))