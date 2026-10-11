"""Deterministic reference baselines for development-set evaluation."""

from collections.abc import Sequence
from fractions import Fraction

from .schemas import (
    AssignedTeam,
    DatasetRecord,
    ExpectedDecision,
    Priority,
    TicketCategory,
)

PRIORITY_ORDER = {
    Priority.LOW: 0,
    Priority.NORMAL: 1,
    Priority.HIGH: 2,
    Priority.URGENT: 3,
}


def _require_records(records: Sequence[DatasetRecord]) -> None:
    if not records:
        raise ValueError("Baseline scoring requires at least one eligible record.")


def _accepted_decisions(record: DatasetRecord) -> list[ExpectedDecision]:
    if isinstance(record.expected, ExpectedDecision):
        return [record.expected]
    return list(record.expected)


def _tie_break_key(decision: ExpectedDecision) -> tuple[str, int, str, bool]:
    return (
        decision.category.value,
        PRIORITY_ORDER[decision.priority],
        decision.assigned_team.value,
        decision.requires_human_review,
    )


def majority_decision(records: Sequence[DatasetRecord]) -> ExpectedDecision:
    """Return the complete decision with the highest weighted support."""
    _require_records(records)

    support: dict[tuple[str, str, str, bool], Fraction] = {}
    decisions_by_key: dict[tuple[str, str, str, bool], ExpectedDecision] = {}

    for record in records:
        accepted = _accepted_decisions(record)
        weight = Fraction(1, len(accepted))

        for decision in accepted:
            key = decision.decision_key()
            support[key] = support.get(key, Fraction(0)) + weight
            decisions_by_key.setdefault(key, decision)

    best_key = min(
        support,
        key=lambda key: (-support[key], _tie_break_key(decisions_by_key[key])),
    )
    return decisions_by_key[best_key]


def always_needs_review_decision() -> ExpectedDecision:
    return ExpectedDecision(
        category=TicketCategory.NEEDS_REVIEW,
        priority=Priority.NORMAL,
        assigned_team=AssignedTeam.HUMAN_REVIEW,
        requires_human_review=True,
    )


def strict_correctness(
    prediction: ExpectedDecision,
    records: Sequence[DatasetRecord],
) -> list[bool]:
    """Return one flag per record for whether any accepted decision matches."""
    _require_records(records)

    prediction_key = prediction.decision_key()

    return [
        any(
            decision.decision_key() == prediction_key
            for decision in _accepted_decisions(record)
        )
        for record in records
    ]
