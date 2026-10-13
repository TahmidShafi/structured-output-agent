import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from structured_output_agent.reporting import EvaluationReport, generate_report
from structured_output_agent.run_config import RunConfig
from structured_output_agent.run_results import RunOutcome, TicketRunResult
from structured_output_agent.runner import (
    EvaluationRun,
    RunStatus,
    compute_dataset_fingerprint,
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


def _make_record(index: int) -> DatasetRecord:
    return DatasetRecord(
        input=f"Sample ticket {index}",
        expected=ExpectedDecision(
            category=TicketCategory.BILLING,
            priority=Priority.NORMAL,
            assigned_team=AssignedTeam.PAYMENTS,
            requires_human_review=False,
        ),
        tags={DatasetTag.CLEAN},
        author="test_author",
        split=DatasetSplit.DEV,
    )


def _make_prediction() -> TicketTriageOutput:
    return TicketTriageOutput(
        category=TicketCategory.BILLING,
        priority=Priority.NORMAL,
        assigned_team=AssignedTeam.PAYMENTS,
        summary="Customer inquiry",
        requires_human_review=False,
    )


def _make_completed_run() -> tuple[EvaluationRun, list[DatasetRecord]]:
    records = [_make_record(0), _make_record(1)]
    pred = _make_prediction()
    results = [
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.VALID_CORRECT,
            credit=1,
            prediction=pred,
        ),
        TicketRunResult(
            ticket_index=1,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=pred,
        ),
    ]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=2,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        completed_results=results,
    )
    return run, records


def test_generate_report_success() -> None:
    run, records = _make_completed_run()

    report = generate_report(run, records)

    assert report.schema_version == "1.1"
    assert report.config == run.config
    assert report.dataset_fingerprint == run.dataset_fingerprint
    assert report.dataset_fingerprint is not None
    assert report.metrics.eligible_count == 2
    assert report.metrics.strict_correct.count == 1
    assert len(report.results) == 2


def test_generate_report_rejects_aborted_run() -> None:
    aborted_run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.ABORTED,
        eligible_count=2,
        completed_results=[],
        error_type="OperationalError",
        error_message="timeout",
    )
    with pytest.raises(
        ValueError, match="Aborted runs must not produce final reports"
    ):
        generate_report(aborted_run, [_make_record(0), _make_record(1)])


def test_generate_report_rejects_mismatched_records() -> None:
    run, _ = _make_completed_run()

    # Mismatched records length
    with pytest.raises(ValueError, match="does not match run eligible_count"):
        generate_report(run, [_make_record(0)])


def test_generate_report_rejects_dataset_fingerprint_mismatch() -> None:
    run, _ = _make_completed_run()
    different_records = [_make_record(10), _make_record(11)]

    with pytest.raises(
        ValueError, match="Dataset fingerprint mismatch: provided records do not match"
    ):
        generate_report(run, different_records)


def test_generate_report_rejects_missing_dataset_fingerprint() -> None:
    run, records = _make_completed_run()
    # Construct a completed run without a dataset fingerprint
    run_no_fp = EvaluationRun(
        config=run.config,
        status=RunStatus.COMPLETED,
        eligible_count=run.eligible_count,
        dataset_fingerprint=None,
        completed_results=run.completed_results,
    )

    with pytest.raises(
        ValueError, match="dataset_fingerprint is required and must not be empty"
    ):
        generate_report(run_no_fp, records)


def test_deterministic_json_serialization() -> None:
    run, records = _make_completed_run()
    report = generate_report(run, records)

    json_1 = report.to_json()
    json_2 = report.to_json()

    assert json_1 == json_2

    parsed = json.loads(json_1)
    keys = list(parsed.keys())
    assert keys == sorted(keys)

    # Check sorted keys in sub-objects
    metrics_keys = list(parsed["metrics"].keys())
    assert metrics_keys == sorted(metrics_keys)

    # Verify tag keys are strings
    assert "clean" in parsed["metrics"]["tags"]
    assert "long" in parsed["metrics"]["tags"]
    # Long tag has 0 records, verify rate is serialized as null
    assert parsed["metrics"]["tags"]["long"]["rate"] is None


def test_deterministic_file_writing(tmp_path: Path) -> None:
    run, records = _make_completed_run()
    report = generate_report(run, records)

    output_file = tmp_path / "report.json"
    report.write_json(output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")

    # Ends with trailing newline
    assert content.endswith("\n")
    assert not content.endswith("\n\n")

    # Matches to_json() + "\n"
    assert content == report.to_json() + "\n"


def test_report_json_round_trip() -> None:
    run, records = _make_completed_run()
    report = generate_report(run, records)

    json_str = report.to_json()
    reconstructed = EvaluationReport.from_json(json_str)

    assert reconstructed == report
    assert reconstructed.schema_version == report.schema_version
    assert reconstructed.config == report.config
    assert reconstructed.metrics == report.metrics
    assert reconstructed.results == report.results
    assert isinstance(reconstructed.results[0], TicketRunResult)


def test_generate_report_uses_retained_records_by_default() -> None:
    records = [_make_record(0), _make_record(1)]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=2,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        records=records,
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            ),
            TicketRunResult(
                ticket_index=1,
                outcome=RunOutcome.VALID_INCORRECT,
                credit=0,
                prediction=_make_prediction(),
            ),
        ],
    )

    report = generate_report(run)

    assert report.schema_version == "1.1"
    assert report.dataset_fingerprint == run.dataset_fingerprint
    assert report.metrics.eligible_count == 2
    assert report.metrics.strict_correct.count == 1
    assert len(report.results) == 2


def test_generate_report_rejects_missing_records_when_none_provided() -> None:
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=2,
        dataset_fingerprint="some_fingerprint",
        records=[],
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            ),
            TicketRunResult(
                ticket_index=1,
                outcome=RunOutcome.VALID_INCORRECT,
                credit=0,
                prediction=_make_prediction(),
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="EvaluationRun has no retained dataset records",
    ):
        generate_report(run)


def test_generate_report_retained_records_rejects_mismatched_explicit() -> None:
    records = [_make_record(0), _make_record(1)]
    different_records = [_make_record(1), _make_record(0)]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=2,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        records=records,
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            ),
            TicketRunResult(
                ticket_index=1,
                outcome=RunOutcome.VALID_INCORRECT,
                credit=0,
                prediction=_make_prediction(),
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Dataset fingerprint mismatch: provided records do not match",
    ):
        generate_report(run, different_records)


def test_report_provenance_metadata_fields() -> None:
    run, records = _make_completed_run()
    # Explicitly specify provenance
    report = generate_report(
        run,
        records,
        client_type="gemini",
        model_id="gemini-3.8-flash",
        created_at="2026-10-12T12:00:00+00:00",
    )

    assert report.client_type == "gemini"
    assert report.model_id == "gemini-3.8-flash"
    assert report.created_at == "2026-10-12T12:00:00+00:00"

    # Serialized JSON contains the exact fields
    raw_json = report.to_json()
    parsed = json.loads(raw_json)
    assert parsed["client_type"] == "gemini"
    assert parsed["model_id"] == "gemini-3.8-flash"
    assert parsed["created_at"] == "2026-10-12T12:00:00+00:00"

    # Reconstruct round-trip
    reconstructed = EvaluationReport.from_json(raw_json)
    assert reconstructed.client_type == "gemini"
    assert reconstructed.model_id == "gemini-3.8-flash"
    assert reconstructed.created_at == "2026-10-12T12:00:00+00:00"


def test_report_provenance_stub_defaults() -> None:
    run, records = _make_completed_run()
    # Without explicit provenance arguments, uses stub default
    report = generate_report(run, records)

    assert report.client_type == "stub"
    assert report.model_id is None
    # created_at parses as ISO 8601 with explicit timezone
    parsed_dt = datetime.fromisoformat(report.created_at)
    assert parsed_dt.tzinfo is not None
    assert parsed_dt.utcoffset() is not None

    raw_json = report.to_json()
    parsed = json.loads(raw_json)
    assert parsed["client_type"] == "stub"
    assert parsed["model_id"] is None


def test_report_created_at_validation() -> None:
    run, records = _make_completed_run()

    # Valid with various timezone formats
    for valid_tz in [
        "2026-10-12T14:30:00+00:00",
        "2026-10-12T14:30:00Z",
        "2026-10-12T14:30:00-05:00",
        "2026-10-12T14:30:00.123456+06:00",
    ]:
        rep = generate_report(run, records, created_at=valid_tz)
        assert rep.created_at == valid_tz

    # Naive timestamp without timezone offset
    with pytest.raises(ValidationError, match="explicit timezone offset"):
        generate_report(run, records, created_at="2026-10-12T14:30:00")

    # Invalid string
    with pytest.raises(ValidationError, match="valid ISO 8601"):
        generate_report(run, records, created_at="not-a-timestamp")

    # Empty string
    with pytest.raises(ValidationError, match="non-empty ISO 8601"):
        generate_report(run, records, created_at="")


def test_report_client_type_validation() -> None:
    run, records = _make_completed_run()

    # Empty or whitespace client_type is rejected
    with pytest.raises(ValidationError, match="client_type"):
        generate_report(run, records, client_type="")

    with pytest.raises(ValidationError, match="client_type"):
        generate_report(run, records, client_type="   ")


def test_report_model_id_validation() -> None:
    run, records = _make_completed_run()

    # None is valid (e.g. for stub)
    rep = generate_report(run, records, model_id=None)
    assert rep.model_id is None

    # Empty or whitespace string is rejected
    with pytest.raises(ValidationError, match="model_id"):
        generate_report(run, records, model_id="")

    with pytest.raises(ValidationError, match="model_id"):
        generate_report(run, records, model_id="   ")


def test_run_created_at_validation() -> None:
    records = [_make_record(0), _make_record(1)]
    # Valid timezone-aware created_at on EvaluationRun
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=2,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            ),
            TicketRunResult(
                ticket_index=1,
                outcome=RunOutcome.VALID_INCORRECT,
                credit=0,
                prediction=_make_prediction(),
            ),
        ],
        created_at="2026-10-12T10:00:00+00:00",
    )
    assert run.created_at == "2026-10-12T10:00:00+00:00"

    # Naive timestamp on EvaluationRun is rejected
    with pytest.raises(ValidationError, match="explicit timezone offset"):
        EvaluationRun(
            config=RunConfig(),
            status=RunStatus.COMPLETED,
            eligible_count=2,
            dataset_fingerprint=compute_dataset_fingerprint(records),
            completed_results=[],
            created_at="2026-10-12T10:00:00",
        )


def test_legacy_version_1_0_report_incompatibility() -> None:
    """Document and verify that legacy v1.0 reports are rejected by v1.1 schema.

    Legacy 1.0 reports lacked the mandatory provenance metadata fields
    (`client_type`, `created_at`) and specified `schema_version == "1.0"`.
    Because EvaluationReport strictly validates schema_version="1.1" and
    requires provenance metadata, attempting to deserialize a legacy v1.0
    report via EvaluationReport.from_json raises ValidationError.
    """
    legacy_v1_0_json = json.dumps(
        {
            "schema_version": "1.0",
            "config": {
                "dataset_path": "evals/data/dev.jsonl",
                "split": "dev",
                "extractor": "extract_json_object",
                "attempts_per_ticket": 1,
            },
            "dataset_fingerprint": "a" * 64,
            "metrics": {
                "eligible_count": 2,
                "strict_correct": {"count": 1, "rate": 0.5},
                "invalid_outputs": {"count": 0, "rate": 0.0},
                "policy_violations": {"count": 0, "rate": 0.0},
                "model_errors": {"count": 0, "rate": 0.0},
                "outcomes": {
                    "o1_valid_correct": {"count": 1, "rate": 0.5},
                    "o2_valid_incorrect": {"count": 1, "rate": 0.5},
                    "o3_policy_violation": {"count": 0, "rate": 0.0},
                    "o4_schema_violation": {"count": 0, "rate": 0.0},
                    "o5_extraction_failure": {"count": 0, "rate": 0.0},
                },
                "tags": {
                    "clean": {"count": 2, "eligible": 2, "rate": 1.0},
                    "ambiguous": {"count": 0, "eligible": 2, "rate": 0.0},
                    "urgent": {"count": 0, "eligible": 2, "rate": 0.0},
                    "needs_review": {"count": 0, "eligible": 2, "rate": 0.0},
                    "long": {"count": 0, "eligible": 2, "rate": 0.0},
                },
            },
            "results": [
                {
                    "ticket_index": 0,
                    "outcome": "valid_correct",
                    "credit": 1,
                    "prediction": {
                        "category": "billing",
                        "priority": "normal",
                        "assigned_team": "payments",
                        "summary": "Customer inquiry",
                        "requires_human_review": False,
                    },
                },
                {
                    "ticket_index": 1,
                    "outcome": "valid_incorrect",
                    "credit": 0,
                    "prediction": {
                        "category": "billing",
                        "priority": "normal",
                        "assigned_team": "payments",
                        "summary": "Customer inquiry",
                        "requires_human_review": False,
                    },
                },
            ],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        EvaluationReport.from_json(legacy_v1_0_json)

    errors = exc_info.value.errors()
    error_fields = {e["loc"][0] for e in errors}
    assert "schema_version" in error_fields
    assert "created_at" in error_fields
    assert "client_type" in error_fields


def test_report_rejects_results_count_mismatch() -> None:
    run, records = _make_completed_run()
    report = generate_report(run, records)
    data = json.loads(report.to_json())

    # Remove one result so len(results) is 1, but eligible_count is 2
    data["results"].pop()

    with pytest.raises(ValidationError, match="Number of results"):
        EvaluationReport.from_json(json.dumps(data))


def test_report_rejects_outcome_counts_mismatch() -> None:
    run, records = _make_completed_run()
    report = generate_report(run, records)
    data = json.loads(report.to_json())

    # Tamper with the first result's outcome from valid_correct to valid_incorrect
    data["results"][0]["outcome"] = "valid_incorrect"
    data["results"][0]["credit"] = 0

    with pytest.raises(
        ValidationError,
        match=r"Results valid_incorrect count \(2\) does not match",
    ):
        EvaluationReport.from_json(json.dumps(data))


def test_report_rejects_credit_sum_mismatch() -> None:
    run, records = _make_completed_run()
    report = generate_report(run, records)
    data = json.loads(report.to_json())

    # Set credit to 0 on valid_correct result without changing outcome
    # Note: TicketRunResult validator also checks credit == 1 for valid_correct,
    # so we can tamper with metrics.strict_correct count or tamper via model_construct
    data["metrics"]["strict_correct"]["count"] = 0
    data["metrics"]["strict_correct"]["rate"] = 0.0

    with pytest.raises(ValidationError, match="strict_correct count"):
        EvaluationReport.from_json(json.dumps(data))


def test_generate_report_and_deserialization_empty_run() -> None:
    records: list[DatasetRecord] = []
    empty_run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=0,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        records=records,
        completed_results=[],
    )

    report = generate_report(empty_run, records)
    assert report.schema_version == "1.1"
    assert report.metrics.eligible_count == 0
    assert len(report.results) == 0
    assert report.metrics.strict_correct.count == 0
    assert report.metrics.strict_correct.total == 0
    assert report.metrics.strict_correct.rate is None

    # Serialization and deserialization round-trip
    json_str = report.to_json()
    reconstructed = EvaluationReport.from_json(json_str)
    assert reconstructed == report

