"""Offline unit tests for evaluation strategies 1-3.

Verifies:
- Strategy selection and isolation (Strategy 1 no retry, Strategy 2 schema-only,
  Strategy 3 schema + policy).
- Paired ticket alignment and identical denominators across all strategies.
- Incomplete evidence detection (rejecting mismatched, missing, or
  out-of-order ticket results).
- Accurate retry accounting (attempts, retries, recovery rate, exhaustion).
- Explicit synthetic simulation labeling and disclosures.
- JSON serialization and deserialization round-trip.
"""

import json
from pathlib import Path

import pytest

from structured_output_agent.checkpoint_io import load_checkpoint
from structured_output_agent.dataset_io import load_dataset
from structured_output_agent.offline_strategies import (
    EvaluationStrategy,
    PairedEvaluationReport,
    evaluate_offline_strategies,
)
from structured_output_agent.run_results import (
    RunOutcome,
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
)


def _make_12_record_dataset(path: Path) -> list[DatasetRecord]:
    category_teams = {
        TicketCategory.BILLING: AssignedTeam.PAYMENTS,
        TicketCategory.ACCOUNT_ACCESS: AssignedTeam.ACCOUNT_SUPPORT,
        TicketCategory.TECHNICAL_ISSUE: AssignedTeam.TECHNICAL_SUPPORT,
        TicketCategory.SHIPPING: AssignedTeam.FULFILLMENT,
        TicketCategory.OTHER: AssignedTeam.GENERAL_SUPPORT,
        TicketCategory.NEEDS_REVIEW: AssignedTeam.HUMAN_REVIEW,
    }
    records = []
    for cat, team in category_teams.items():
        for pri in (Priority.NORMAL, Priority.HIGH):
            if cat == TicketCategory.NEEDS_REVIEW:
                other_pri = (
                    Priority.HIGH if pri == Priority.NORMAL else Priority.NORMAL
                )
                expected = [
                    ExpectedDecision(
                        category=cat,
                        priority=pri,
                        assigned_team=team,
                        requires_human_review=True,
                    ),
                    ExpectedDecision(
                        category=cat,
                        priority=other_pri,
                        assigned_team=team,
                        requires_human_review=True,
                    ),
                ]
                tags = {DatasetTag.AMBIGUOUS}
                label_note = "Coverage test needs_review"
            else:
                expected = ExpectedDecision(
                    category=cat,
                    priority=pri,
                    assigned_team=team,
                    requires_human_review=False,
                )
                tags = {DatasetTag.CLEAN}
                label_note = None

            rec = DatasetRecord(
                input=f"Ticket regarding {cat.value} with priority {pri.value}.",
                expected=expected,
                tags=tags,
                label_note=label_note,
                author="test_author",
                split=DatasetSplit.DEV,
            )
            records.append(rec)

    json_lines = [json.dumps(r.model_dump(mode="json")) for r in records]
    path.write_text("\n".join(json_lines) + "\n", encoding="utf-8")
    return records


def _valid_json_for_record(rec: DatasetRecord) -> str:
    dec = (
        rec.expected[0] if isinstance(rec.expected, list) else rec.expected
    )
    return json.dumps(
        {
            "category": dec.category.value,
            "priority": dec.priority.value,
            "assigned_team": dec.assigned_team.value,
            "summary": f"Summary for {dec.category.value}",
            "requires_human_review": dec.requires_human_review,
        }
    )


def test_strategy_selection_and_isolation(tmp_path: Path) -> None:
    """Verify distinct retry eligibility across Strategy 1, 2, and 3."""
    dataset_path = tmp_path / "dev.jsonl"
    records = _make_12_record_dataset(dataset_path)

    # Initial results:
    # Ticket 0: O2 (schema failure)
    # Ticket 1: O3 (policy failure: billing -> technical_support)
    # Tickets 2-11: O5 (valid correct)
    initial_results = []

    # Ticket 0: Schema failure
    r0 = classify_response(
        json.dumps(
            {
                "category": "INVALID_CAT",
                "priority": "normal",
                "assigned_team": "payments",
                "summary": "Bad category",
                "requires_human_review": False,
            }
        ),
        records[0],
        0,
    )
    initial_results.append(r0)

    # Ticket 1: Policy failure (Validator A)
    r1 = classify_response(
        json.dumps(
            {
                "category": "billing",
                "priority": "normal",
                "assigned_team": "technical_support",  # Violates Validator A
                "summary": "Billing query",
                "requires_human_review": False,
            }
        ),
        records[1],
        1,
    )
    initial_results.append(r1)

    # Tickets 2-11: Valid correct
    for idx in range(2, 12):
        initial_results.append(
            classify_response(_valid_json_for_record(records[idx]), records[idx], idx)
        )

    # Simulator that fixes both ticket 0 and ticket 1 on retry
    def mock_simulator(
        ticket_index: int,
        record: DatasetRecord,
        attempt: int,
        outcome: RunOutcome,
        err_msg: str | None,
    ) -> str:
        return _valid_json_for_record(record)

    report = evaluate_offline_strategies(
        records,
        initial_results,
        evidence_source="unit_test_synthetic",
        retry_simulator=mock_simulator,
        max_retries=2,
    )

    s1 = report.strategies[EvaluationStrategy.STRATEGY_1_NO_RETRY]
    s2 = report.strategies[EvaluationStrategy.STRATEGY_2_SCHEMA_RETRY]
    s3 = report.strategies[EvaluationStrategy.STRATEGY_3_SCHEMA_POLICY_RETRY]

    # Strategy 1 (No retry):
    # Ticket 0 is O2, Ticket 1 is O3. Both remain failures.
    assert s1.ticket_results[0].final_outcome == RunOutcome.SCHEMA_FAILURE
    assert s1.ticket_results[0].attempts_used == 1
    assert s1.ticket_results[1].final_outcome == RunOutcome.POLICY_FAILURE
    assert s1.ticket_results[1].attempts_used == 1
    assert s1.metrics.strict_correct.count == 10

    # Strategy 2 (Schema-aware retry):
    # Ticket 0 (O2) retried and recovered to O5.
    # Ticket 1 (O3) is NOT eligible for retry under Strategy 2, so remains O3.
    assert s2.ticket_results[0].final_outcome == RunOutcome.VALID_CORRECT
    assert s2.ticket_results[0].recovered is True
    assert s2.ticket_results[0].attempts_used == 2
    assert s2.ticket_results[1].final_outcome == RunOutcome.POLICY_FAILURE
    assert s2.ticket_results[1].attempts_used == 1
    assert s2.metrics.strict_correct.count == 11

    # Strategy 3 (Schema- and policy-aware retry):
    # Both Ticket 0 (O2) and Ticket 1 (O3) are retried and recover to O5.
    assert s3.ticket_results[0].final_outcome == RunOutcome.VALID_CORRECT
    assert s3.ticket_results[0].recovered is True
    assert s3.ticket_results[1].final_outcome == RunOutcome.VALID_CORRECT
    assert s3.ticket_results[1].recovered is True
    assert s3.metrics.strict_correct.count == 12


def test_paired_ticket_alignment_and_denominators(tmp_path: Path) -> None:
    """Verify strictly aligned ticket count and indices across all strategies."""
    dataset_path = tmp_path / "dev.jsonl"
    records = _make_12_record_dataset(dataset_path)

    initial_results = [
        classify_response(_valid_json_for_record(rec), rec, idx)
        for idx, rec in enumerate(records)
    ]

    report = evaluate_offline_strategies(
        records,
        initial_results,
        evidence_source="aligned_test",
    )

    assert report.eligible_count == 12
    for strat, res in report.strategies.items():
        assert res.eligible_count == 12
        assert len(res.ticket_results) == 12
        assert [t.ticket_index for t in res.ticket_results] == list(range(12))


def test_incomplete_evidence_rejected_without_fabrication(tmp_path: Path) -> None:
    """Verify incomplete or mismatched evidence raises descriptive ValueError."""
    dataset_path = tmp_path / "dev.jsonl"
    records = _make_12_record_dataset(dataset_path)

    # Only 4 results for 12 records (simulating partial checkpoint like v5)
    partial_results = [
        classify_response(_valid_json_for_record(records[idx]), records[idx], idx)
        for idx in range(4)
    ]

    with pytest.raises(
        ValueError, match="Incomplete evidence: initial_results count \\(4\\)"
    ):
        evaluate_offline_strategies(
            records,
            partial_results,
            evidence_source="partial_evidence",
        )

    # Empty records
    with pytest.raises(ValueError, match="Cannot evaluate strategies on empty"):
        evaluate_offline_strategies([], [], evidence_source="empty")


def test_out_of_order_ticket_indices_rejected(tmp_path: Path) -> None:
    """Verify non-contiguous ticket indices are rejected."""
    dataset_path = tmp_path / "dev.jsonl"
    records = _make_12_record_dataset(dataset_path)

    results = [
        classify_response(_valid_json_for_record(rec), rec, idx)
        for idx, rec in enumerate(records)
    ]
    # Swap indices 0 and 1
    results[0], results[1] = results[1], results[0]

    with pytest.raises(ValueError, match="Ticket index mismatch in evidence"):
        evaluate_offline_strategies(records, results, evidence_source="scrambled")


def test_retry_accounting_and_exhaustion(tmp_path: Path) -> None:
    """Verify accurate counting of retries, recoveries, and exhaustion."""
    dataset_path = tmp_path / "dev.jsonl"
    records = _make_12_record_dataset(dataset_path)

    # Ticket 0: fails schema
    initial_results = [
        classify_response(
            json.dumps(
                {
                    "category": "INVALID",
                    "priority": "normal",
                    "assigned_team": "payments",
                    "summary": "Bad",
                    "requires_human_review": False,
                }
            ),
            records[0],
            0,
        )
    ]
    for idx in range(1, 12):
        initial_results.append(
            classify_response(_valid_json_for_record(records[idx]), records[idx], idx)
        )

    # Without retry_simulator: retry evidence is unavailable, ticket 0 exhausts
    report = evaluate_offline_strategies(
        records,
        initial_results,
        evidence_source="exhaustion_test",
        max_retries=2,
    )

    s2 = report.strategies[EvaluationStrategy.STRATEGY_2_SCHEMA_RETRY]
    ac = s2.retry_accounting

    assert ac.tickets_retried == 1
    assert ac.tickets_recovered == 0
    assert ac.tickets_exhausted == 1
    assert ac.recovery_rate.count == 0
    assert ac.recovery_rate.total == 1
    assert ac.recovery_rate.rate == 0.0


def test_synthetic_simulation_disclosure_flag(tmp_path: Path) -> None:
    """Verify synthetic simulations are explicitly marked with disclosures."""
    dataset_path = tmp_path / "dev.jsonl"
    records = _make_12_record_dataset(dataset_path)

    initial_results = [
        classify_response("not json", records[0], 0)
    ]
    for idx in range(1, 12):
        initial_results.append(
            classify_response(_valid_json_for_record(records[idx]), records[idx], idx)
        )

    # With simulator
    report = evaluate_offline_strategies(
        records,
        initial_results,
        evidence_source="sim_test",
        retry_simulator=lambda *args: _valid_json_for_record(records[0]),
    )

    s2 = report.strategies[EvaluationStrategy.STRATEGY_2_SCHEMA_RETRY]
    assert s2.is_synthetic_simulation is True
    assert s2.simulation_note is not None
    assert "Must NOT be presented as observed Gemini performance" in s2.simulation_note


def test_paired_evaluation_report_serialization_round_trip(tmp_path: Path) -> None:
    """Verify PairedEvaluationReport serializes and reconstructs cleanly."""
    dataset_path = tmp_path / "dev.jsonl"
    records = _make_12_record_dataset(dataset_path)

    initial_results = [
        classify_response(_valid_json_for_record(rec), rec, idx)
        for idx, rec in enumerate(records)
    ]

    report = evaluate_offline_strategies(
        records,
        initial_results,
        evidence_source="round_trip_test",
    )

    json_str = report.model_dump_json()
    reconstructed = PairedEvaluationReport.model_validate_json(json_str)

    assert reconstructed.schema_version == "1.0"
    assert reconstructed.eligible_count == 12
    assert reconstructed.dataset_fingerprint == report.dataset_fingerprint
    assert len(reconstructed.strategies) == 3
    assert len(reconstructed.paired_comparisons) == 3


def test_evaluate_historical_v4_evidence() -> None:
    """Verify offline strategy evaluation on historical v4 development checkpoint."""
    records = load_dataset(Path("evals/data/dev.jsonl"), split=DatasetSplit.DEV)
    cp = load_checkpoint(Path("evals/results/gemini_dev_checkpoint_v4.json"))

    report = evaluate_offline_strategies(
        records,
        cp.completed_results,
        evidence_source="evals/results/gemini_dev_checkpoint_v4.json",
    )

    assert report.eligible_count == 20
    # In v4, 19/20 are valid_correct and 1/20 is valid_incorrect (ticket 13).
    # Since none are extraction/schema/policy failures, none are retried!
    s1 = report.strategies[EvaluationStrategy.STRATEGY_1_NO_RETRY]
    s2 = report.strategies[EvaluationStrategy.STRATEGY_2_SCHEMA_RETRY]
    s3 = report.strategies[EvaluationStrategy.STRATEGY_3_SCHEMA_POLICY_RETRY]

    assert s1.metrics.strict_correct.count == 19
    assert s2.metrics.strict_correct.count == 19
    assert s3.metrics.strict_correct.count == 19
    assert s1.retry_accounting.tickets_retried == 0
    assert s2.retry_accounting.tickets_retried == 0
    assert s3.retry_accounting.tickets_retried == 0
