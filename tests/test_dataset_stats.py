
from structured_output_agent.dataset_stats import get_dataset_statistics
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


def make_record(
    *,
    ticket_id: str,
    tags: set[DatasetTag],
    category: TicketCategory,
    priority: Priority,
    split: DatasetSplit = DatasetSplit.DEV,
) -> DatasetRecord:
    decision = ExpectedDecision(
        category=category,
        priority=priority,
        assigned_team=AssignedTeam(
            {
                TicketCategory.BILLING: "payments",
                TicketCategory.ACCOUNT_ACCESS: "account_support",
                TicketCategory.TECHNICAL_ISSUE: "technical_support",
                TicketCategory.SHIPPING: "fulfillment",
                TicketCategory.OTHER: "general_support",
                TicketCategory.NEEDS_REVIEW: "human_review",
            }[category]
        ),
        requires_human_review=(
            category == TicketCategory.NEEDS_REVIEW
            or priority == Priority.URGENT
        ),
    )
    injection_target = None
    if DatasetTag.INJECTION in tags:
        injection_target = [
            InjectionTarget(
                kind=InjectionKind.SUMMARY_ECHO,
                marker="INJECT-001",
            )
        ]
        input_text = f"Example support ticket {ticket_id} INJECT-001"
    else:
        input_text = f"Example support ticket {ticket_id}"

    return DatasetRecord(
        input=input_text,
        expected=decision,
        tags=tags,
        author="test",
        split=split,
        injection_target=injection_target,
    )


def test_counts_records_splits_and_tags() -> None:
    records = [
        make_record(
            ticket_id="1",
            tags={DatasetTag.CLEAN},
            category=TicketCategory.BILLING,
            priority=Priority.NORMAL,
        ),
        make_record(
            ticket_id="2",
            tags={DatasetTag.INJECTION, DatasetTag.LONG},
            category=TicketCategory.SHIPPING,
            priority=Priority.HIGH,
            split=DatasetSplit.TEST,
        ),
    ]

    stats = get_dataset_statistics(records)

    assert stats["total_records"] == 2
    assert stats["split_counts"] == {"dev": 1, "test": 1}
    assert stats["tag_counts"]["clean"] == 1
    assert stats["tag_counts"]["injection"] == 1
    assert stats["tag_counts"]["long"] == 1
    assert stats["tag_counts"]["ambiguous"] == 0


def test_counts_pairwise_tag_overlaps() -> None:
    records = [
        make_record(
            ticket_id="1",
            tags={DatasetTag.INJECTION, DatasetTag.LONG},
            category=TicketCategory.BILLING,
            priority=Priority.NORMAL,
        ),
        make_record(
            ticket_id="2",
            tags={DatasetTag.INJECTION},
            category=TicketCategory.SHIPPING,
            priority=Priority.HIGH,
        ),
    ]

    stats = get_dataset_statistics(records)
    overlaps = stats["pairwise_tag_overlaps"]

    assert overlaps["injection&long"] == 1
    assert overlaps["clean&injection"] == 0


def test_counts_expected_category_and_priority_labels() -> None:
    records = [
        make_record(
            ticket_id="1",
            tags={DatasetTag.CLEAN},
            category=TicketCategory.BILLING,
            priority=Priority.NORMAL,
        ),
        make_record(
            ticket_id="2",
            tags={DatasetTag.LONG},
            category=TicketCategory.SHIPPING,
            priority=Priority.HIGH,
        ),
    ]

    stats = get_dataset_statistics(records)

    assert stats["category_counts"]["billing"] == 1
    assert stats["category_counts"]["shipping"] == 1
    assert stats["priority_counts"]["normal"] == 1
    assert stats["priority_counts"]["high"] == 1


def test_empty_dataset_returns_zero_counts() -> None:
    stats = get_dataset_statistics([])

    assert stats["total_records"] == 0
    assert all(value == 0 for value in stats["split_counts"].values())
    assert all(value == 0 for value in stats["tag_counts"].values())
    assert all(
        value == 0
        for value in stats["pairwise_tag_overlaps"].values()
    )
    assert all(value == 0 for value in stats["category_counts"].values())
    assert all(value == 0 for value in stats["priority_counts"].values())
