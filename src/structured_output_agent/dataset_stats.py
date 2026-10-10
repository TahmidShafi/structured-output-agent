
from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from .schemas import (
    DatasetRecord,
    DatasetSplit,
    DatasetTag,
    Priority,
    TicketCategory,
)


def get_dataset_statistics(
    records: Sequence[DatasetRecord],
) -> dict[str, object]:
    """Return dataset counts, tag overlaps, and label distributions."""

    tags = list(DatasetTag)
    splits = list(DatasetSplit)
    categories = list(TicketCategory)
    priorities = list(Priority)

    tag_counts = Counter(
        tag.value
        for record in records
        for tag in record.tags
    )

    split_counts = Counter(record.split.value for record in records)

    tag_overlaps = {
        f"{first.value}&{second.value}": sum(
            first in record.tags and second in record.tags
            for record in records
        )
        for first, second in combinations(tags, 2)
    }

    category_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()

    for record in records:
        expected_decisions = (
            record.expected
            if isinstance(record.expected, list)
            else [record.expected]
        )

        for decision in expected_decisions:
            category_counts[decision.category.value] += 1
            priority_counts[decision.priority.value] += 1

    return {
        "total_records": len(records),
        "split_counts": {
            split.value: split_counts[split.value]
            for split in splits
        },
        "tag_counts": {
            tag.value: tag_counts[tag.value]
            for tag in tags
        },
        "pairwise_tag_overlaps": tag_overlaps,
        "category_counts": {
            category.value: category_counts[category.value]
            for category in categories
        },
        "priority_counts": {
            priority.value: priority_counts[priority.value]
            for priority in priorities
        },
    }
