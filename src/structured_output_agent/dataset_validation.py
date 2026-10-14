
"""Dataset-wide integrity checks for support-ticket datasets."""

from collections.abc import Sequence

from .schemas import DatasetRecord, DatasetTag, ExpectedDecision, TicketCategory


def _normalize_ticket_text(text: str) -> str:
    """Normalize whitespace and case for duplicate detection."""
    return " ".join(text.split()).casefold()


def validate_unique_tickets(
    records: Sequence[DatasetRecord],
) -> None:
    """Reject duplicate ticket inputs within the supplied records."""
    seen: dict[str, int] = {}

    for index, record in enumerate(records):
        normalized = _normalize_ticket_text(record.input)

        if normalized in seen:
            first_index = seen[normalized]
            raise ValueError(
                f"Duplicate ticket input at record indexes "
                f"{first_index} and {index}."
            )

        seen[normalized] = index


def validate_no_cross_split_duplicates(
    split_a_records: Sequence[DatasetRecord],
    split_b_records: Sequence[DatasetRecord],
    *,
    split_a_name: str = "split_a",
    split_b_name: str = "split_b",
) -> None:
    """Ensure no duplicate ticket inputs exist across two dataset splits.

    Normalizes ticket text identically to validate_unique_tickets to detect exact,
    case-insensitive, and whitespace-variant duplicates between the two record sets.
    Raises ValueError on the first detected duplicate with clear split names and
    indexes.
    """
    seen: dict[str, int] = {}
    for index, record in enumerate(split_a_records):
        normalized = _normalize_ticket_text(record.input)
        if normalized not in seen:
            seen[normalized] = index

    for index, record in enumerate(split_b_records):
        normalized = _normalize_ticket_text(record.input)
        if normalized in seen:
            first_index = seen[normalized]
            raise ValueError(
                f"Cross-split duplicate ticket input detected between "
                f"{split_a_name} (index {first_index}) and "
                f"{split_b_name} (index {index})."
            )


def validate_unique_injection_markers(
    records: Sequence[DatasetRecord],
) -> None:
    """Require unique markers only for summary_echo targets."""
    seen: dict[str, int] = {}

    for index, record in enumerate(records):
        for target in record.injection_target or []:
            if target.kind.value != "summary_echo":
                continue

            # The schema already requires summary_echo markers to be non-empty.
            marker = target.marker
            if marker is None:
                continue

            if marker in seen:
                first_index = seen[marker]
                raise ValueError(
                    f"Duplicate summary_echo marker {marker!r} at "
                    f"record indexes {first_index} and {index}."
                )

            seen[marker] = index

def validate_long_ticket_tags(
    records: Sequence[DatasetRecord],
) -> None:
    """Require the long tag exactly when a ticket exceeds 300 words."""
    threshold = 300

    for index, record in enumerate(records):
        word_count = len(record.input.split())
        has_long_tag = DatasetTag.LONG in record.tags
        should_be_long = word_count > threshold

        if should_be_long and not has_long_tag:
            raise ValueError(
                f"Record index {index} has {word_count} words, exceeding "
                f"the {threshold}-word threshold, but is missing the "
                "'long' tag."
            )

        if has_long_tag and not should_be_long:
            raise ValueError(
                f"Record index {index} has {word_count} words, at or below "
                f"the {threshold}-word threshold, but has the 'long' tag."
            )

def validate_category_priority_coverage(
    records: Sequence[DatasetRecord],
) -> None:
    """Require at least two distinct priorities for every ticket category."""
    priorities_by_category: dict[str, set[str]] = {
        category.value: set() for category in TicketCategory
    }

    for record in records:
        decisions = (
            [record.expected]
            if isinstance(record.expected, ExpectedDecision)
            else record.expected
        )

        for decision in decisions:
            priorities_by_category[decision.category.value].add(
                decision.priority.value
            )

    for category, priorities in priorities_by_category.items():
        if len(priorities) < 2:
            raise ValueError(
                f"Category {category!r} has only {len(priorities)} "
                "distinct priority value(s); at least 2 are required."
            )

def validate_dataset_integrity(
    records: Sequence[DatasetRecord],
) -> None:
    """Run all implemented dataset-wide integrity checks."""
    validate_unique_tickets(records)
    validate_unique_injection_markers(records)
    validate_long_ticket_tags(records)
    validate_category_priority_coverage(records)
