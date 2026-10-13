from typing import Any

import pytest
from pydantic import ValidationError

from structured_output_agent.dataset_io import load_dataset
from structured_output_agent.metrics import (
    BaselineResult,
    EvaluationMetrics,
    OutcomeBreakdown,
    RatioMetric,
    compute_metrics,
    make_ratio,
)
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
    InjectionKind,
    InjectionTarget,
    Priority,
    TicketCategory,
    TicketTriageOutput,
)


def _make_prediction(
    category: TicketCategory = TicketCategory.BILLING,
    priority: Priority = Priority.NORMAL,
    team: AssignedTeam = AssignedTeam.PAYMENTS,
    requires_human_review: bool = False,
) -> TicketTriageOutput:
    return TicketTriageOutput(
        category=category,
        priority=priority,
        assigned_team=team,
        summary="Customer support request",
        requires_human_review=requires_human_review,
    )


def _make_record(
    index: int,
    category: TicketCategory = TicketCategory.BILLING,
    priority: Priority = Priority.NORMAL,
    team: AssignedTeam = AssignedTeam.PAYMENTS,
    tags: set[DatasetTag] | None = None,
    requires_human_review: bool = False,
) -> DatasetRecord:
    record_tags = tags or {DatasetTag.CLEAN}
    is_ambiguous = DatasetTag.AMBIGUOUS in record_tags
    is_injection = DatasetTag.INJECTION in record_tags

    if is_ambiguous:
        expected: ExpectedDecision | list[ExpectedDecision] = [
            ExpectedDecision(
                category=TicketCategory.NEEDS_REVIEW,
                priority=Priority.NORMAL,
                assigned_team=AssignedTeam.HUMAN_REVIEW,
                requires_human_review=True,
            ),
            ExpectedDecision(
                category=TicketCategory.NEEDS_REVIEW,
                priority=Priority.HIGH,
                assigned_team=AssignedTeam.HUMAN_REVIEW,
                requires_human_review=True,
            ),
        ]
        label_note = "Ambiguous test record"
    else:
        expected = ExpectedDecision(
            category=category,
            priority=priority,
            assigned_team=team,
            requires_human_review=requires_human_review,
        )
        label_note = None

    marker = f"[M_{index}]" if is_injection else None
    targets = (
        [
            InjectionTarget(
                kind=InjectionKind.SUMMARY_ECHO, marker=marker
            )
        ]
        if is_injection
        else None
    )
    input_text = (
        f"Sample ticket {index} with {marker}"
        if marker
        else f"Sample ticket {index}"
    )

    return DatasetRecord(
        input=input_text,
        expected=expected,
        tags=record_tags,
        author="test_author",
        injection_target=targets,
        label_note=label_note,
        split=DatasetSplit.DEV,
    )


def test_ratio_metric_validation() -> None:
    # Valid non-zero total
    r1 = make_ratio(3, 10)
    assert r1.count == 3
    assert r1.total == 10
    assert r1.rate == pytest.approx(0.3)

    # Valid zero total
    r0 = make_ratio(0, 0)
    assert r0.count == 0
    assert r0.total == 0
    assert r0.rate is None

    # Invalid rate mismatch
    with pytest.raises(ValueError, match="does not match"):
        RatioMetric(count=3, total=10, rate=0.5)

    # Invalid zero total with non-None rate
    with pytest.raises(ValueError, match="rate must be None"):
        RatioMetric(count=0, total=0, rate=0.0)

    # Invalid zero total with non-zero count
    with pytest.raises(ValueError, match="count must be 0"):
        RatioMetric(count=1, total=0, rate=None)


def test_baseline_result_validation() -> None:
    b1 = BaselineResult(correct_count=2, total=20, accuracy=0.10)
    assert b1.correct_count == 2
    assert b1.total == 20
    assert b1.accuracy == pytest.approx(0.10)

    # Total 0
    b0 = BaselineResult(correct_count=0, total=0, accuracy=None)
    assert b0.accuracy is None

    # Accuracy mismatch
    with pytest.raises(ValueError, match="does not match"):
        BaselineResult(correct_count=2, total=20, accuracy=0.20)


def test_compute_metrics_on_dev_dataset_baselines_parity() -> None:
    dev_records = load_dataset("evals/data/dev.jsonl", split=DatasetSplit.DEV)
    assert len(dev_records) == 20

    # Build a completed run where all 20 tickets have valid billing answers
    billing_pred = _make_prediction()
    results: list[TicketRunResult] = []
    for i, rec in enumerate(dev_records):
        accepted = (
            [rec.expected]
            if isinstance(rec.expected, ExpectedDecision)
            else rec.expected
        )
        is_billing = any(
            d.category == TicketCategory.BILLING for d in accepted
        )
        results.append(
            TicketRunResult(
                ticket_index=i,
                outcome=(
                    RunOutcome.VALID_CORRECT
                    if is_billing
                    else RunOutcome.VALID_INCORRECT
                ),
                credit=1 if is_billing else 0,
                prediction=billing_pred,
            )
        )

    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=20,
        dataset_fingerprint=compute_dataset_fingerprint(dev_records),
        completed_results=results,
    )

    metrics = compute_metrics(run, dev_records)

    assert metrics.eligible_count == 20
    # Parity with evaluation-policy.md baseline values:
    # majority baseline = 2 / 20 (10%)
    assert metrics.baselines["majority"].correct_count == 2
    assert metrics.baselines["majority"].total == 20
    assert metrics.baselines["majority"].accuracy == pytest.approx(0.10)

    # always_needs_review baseline = 4 / 20 (20%)
    assert metrics.baselines["always_needs_review"].correct_count == 4
    assert metrics.baselines["always_needs_review"].total == 20
    assert metrics.baselines["always_needs_review"].accuracy == pytest.approx(
        0.20
    )


def test_outcomes_breakdown_and_zero_denominator_rates() -> None:
    # 5 tickets with overlapping tags
    records = [
        _make_record(0, tags={DatasetTag.CLEAN}),
        _make_record(1, tags={DatasetTag.AMBIGUOUS, DatasetTag.INJECTION}),
        _make_record(2, tags={DatasetTag.AMBIGUOUS}),
        _make_record(3, tags={DatasetTag.INJECTION}),
        _make_record(4, tags={DatasetTag.CLEAN}),
    ]

    pred_billing = _make_prediction(requires_human_review=True)
    pred_other = _make_prediction(
        category=TicketCategory.OTHER,
        team=AssignedTeam.GENERAL_SUPPORT,
        requires_human_review=False,
    )

    # O1 extraction, O2 schema, O3 policy, O4 valid incorrect, O5 valid correct
    results = [
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.EXTRACTION_FAILURE,
            credit=0,
            error_type="JSONExtractionError",
            error_message="No JSON object",
        ),
        TicketRunResult(
            ticket_index=1,
            outcome=RunOutcome.SCHEMA_FAILURE,
            credit=0,
            error_type="ValidationError",
            error_message="Field required",
        ),
        TicketRunResult(
            ticket_index=2,
            outcome=RunOutcome.POLICY_FAILURE,
            credit=0,
            error_type="ValueError",
            error_message="Urgent requires review",
            prediction=pred_billing,
        ),
        TicketRunResult(
            ticket_index=3,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=pred_other,
        ),
        TicketRunResult(
            ticket_index=4,
            outcome=RunOutcome.VALID_CORRECT,
            credit=1,
            prediction=pred_billing,
        ),
    ]

    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=5,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        completed_results=results,
    )

    metrics = compute_metrics(run, records)

    # Outcomes sum to eligible_count
    assert metrics.outcomes.extraction_failure == 1
    assert metrics.outcomes.schema_failure == 1
    assert metrics.outcomes.policy_failure == 1
    assert metrics.outcomes.valid_incorrect == 1
    assert metrics.outcomes.valid_correct == 1

    # Strict accuracy
    assert metrics.strict_correct.count == 1
    assert metrics.strict_correct.total == 5
    assert metrics.strict_correct.rate == pytest.approx(0.20)

    # Invalid outputs (O1 + O2 + O3) / 5
    assert metrics.invalid_outputs.count == 3
    assert metrics.invalid_outputs.total == 5
    assert metrics.invalid_outputs.rate == pytest.approx(0.60)

    # Accuracy among valid outputs (O5 / (O4 + O5) = 1 / 2)
    assert metrics.accuracy_among_valid.count == 1
    assert metrics.accuracy_among_valid.total == 2
    assert metrics.accuracy_among_valid.rate == pytest.approx(0.50)

    # Human review: 2 predictions have requires_human_review=True (indices 2 and 4)
    # denominator is full eligible_count (5)
    assert metrics.human_review.count == 2
    assert metrics.human_review.total == 5
    assert metrics.human_review.rate == pytest.approx(0.40)

    # Tag breakdown:
    # CLEAN: records 0 and 4 -> total 2, correct on record 4 -> 1/2 = 0.5
    assert metrics.tags[DatasetTag.CLEAN].count == 1
    assert metrics.tags[DatasetTag.CLEAN].total == 2
    assert metrics.tags[DatasetTag.CLEAN].rate == pytest.approx(0.50)

    # AMBIGUOUS: records 1 and 2 -> total 2, correct 0 -> 0/2 = 0.0
    assert metrics.tags[DatasetTag.AMBIGUOUS].count == 0
    assert metrics.tags[DatasetTag.AMBIGUOUS].total == 2
    assert metrics.tags[DatasetTag.AMBIGUOUS].rate == pytest.approx(0.0)

    # INJECTION: records 1 and 3 -> total 2, correct 0 -> 0/2 = 0.0
    assert metrics.tags[DatasetTag.INJECTION].count == 0
    assert metrics.tags[DatasetTag.INJECTION].total == 2
    assert metrics.tags[DatasetTag.INJECTION].rate == pytest.approx(0.0)

    # LONG: 0 records -> total 0, rate is None
    assert metrics.tags[DatasetTag.LONG].count == 0
    assert metrics.tags[DatasetTag.LONG].total == 0
    assert metrics.tags[DatasetTag.LONG].rate is None


def test_accuracy_among_valid_none_when_no_valid_outputs() -> None:
    records = [_make_record(0)]
    results = [
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.EXTRACTION_FAILURE,
            credit=0,
            error_type="JSONExtractionError",
            error_message="Fail",
        )
    ]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=1,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        completed_results=results,
    )

    metrics = compute_metrics(run, records)

    assert metrics.accuracy_among_valid.count == 0
    assert metrics.accuracy_among_valid.total == 0
    assert metrics.accuracy_among_valid.rate is None


def test_rejects_aborted_run() -> None:
    aborted_run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.ABORTED,
        eligible_count=2,
        completed_results=[],
        error_type="OperationalError",
        error_message="timeout",
    )
    with pytest.raises(
        ValueError, match="Aborted runs must not produce metrics"
    ):
        compute_metrics(aborted_run, [_make_record(0), _make_record(1)])


def test_rejects_mismatched_records() -> None:
    records = [_make_record(0)]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=1,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            )
        ],
    )

    # Mismatched records length (2 records for 1 result)
    with pytest.raises(ValueError, match="does not match run eligible_count"):
        compute_metrics(run, [_make_record(0), _make_record(1)])

    # Mismatched records length (0 records for 1 result)
    with pytest.raises(ValueError, match="does not match run eligible_count"):
        compute_metrics(run, [])


def test_rejects_different_dataset_with_same_count() -> None:
    records_a = [_make_record(0), _make_record(1)]
    records_b = [_make_record(10), _make_record(11)]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=2,
        dataset_fingerprint=compute_dataset_fingerprint(records_a),
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            ),
            TicketRunResult(
                ticket_index=1,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            ),
        ],
    )

    with pytest.raises(
        ValueError, match="Dataset fingerprint mismatch: provided records do not match"
    ):
        compute_metrics(run, records_b)


def test_rejects_reordered_records_with_same_count() -> None:
    records = [_make_record(0), _make_record(1)]
    reordered_records = list(reversed(records))
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
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            ),
        ],
    )

    with pytest.raises(
        ValueError, match="Dataset fingerprint mismatch: provided records do not match"
    ):
        compute_metrics(run, reordered_records)


def test_dataset_fingerprint_deterministic_and_sensitive() -> None:
    records = [_make_record(0), _make_record(1)]
    fp1 = compute_dataset_fingerprint(records)
    fp2 = compute_dataset_fingerprint(records)
    assert fp1 == fp2
    assert len(fp1) == 64

    # Content change alters fingerprint
    different_records = [_make_record(0), _make_record(2)]
    assert compute_dataset_fingerprint(different_records) != fp1

    # Order change alters fingerprint
    reordered = list(reversed(records))
    assert compute_dataset_fingerprint(reordered) != fp1


def test_rejects_missing_dataset_fingerprint() -> None:
    records = [_make_record(0)]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=1,
        dataset_fingerprint=None,
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            )
        ],
    )
    with pytest.raises(
        ValueError, match="dataset_fingerprint is required and must not be empty"
    ):
        compute_metrics(run, records)


@pytest.mark.parametrize("empty_fp", ["", "   ", "\t\n"])
def test_rejects_empty_or_whitespace_dataset_fingerprint(empty_fp: str) -> None:
    records = [_make_record(0)]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=1,
        dataset_fingerprint=empty_fp,
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            )
        ],
    )
    with pytest.raises(
        ValueError, match="dataset_fingerprint is required and must not be empty"
    ):
        compute_metrics(run, records)


def test_compute_metrics_uses_retained_records_by_default() -> None:
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

    # Calling compute_metrics without records should use run.records
    metrics = compute_metrics(run)

    assert metrics.eligible_count == 2
    assert metrics.strict_correct.count == 1
    assert metrics.outcomes.valid_correct == 1
    assert metrics.outcomes.valid_incorrect == 1


def test_compute_metrics_rejects_missing_records_when_none_provided() -> None:
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=1,
        dataset_fingerprint="some_fingerprint",
        records=[],
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            )
        ],
    )
    with pytest.raises(
        ValueError,
        match="EvaluationRun has no retained dataset records",
    ):
        compute_metrics(run)


def test_compute_metrics_retained_records_rejects_mismatched_explicit() -> None:
    records_a = [_make_record(0)]
    records_b = [_make_record(1)]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=1,
        dataset_fingerprint=compute_dataset_fingerprint(records_a),
        records=records_a,
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=_make_prediction(),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Dataset fingerprint mismatch: provided records do not match",
    ):
        compute_metrics(run, records_b)


def test_compute_metrics_empty_run_zero_eligible() -> None:
    records: list[DatasetRecord] = []
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=0,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        records=records,
        completed_results=[],
    )

    metrics = compute_metrics(run, records)

    assert metrics.eligible_count == 0

    # strict_correct
    assert metrics.strict_correct.count == 0
    assert metrics.strict_correct.total == 0
    assert metrics.strict_correct.rate is None

    # invalid_outputs
    assert metrics.invalid_outputs.count == 0
    assert metrics.invalid_outputs.total == 0
    assert metrics.invalid_outputs.rate is None

    # accuracy_among_valid
    assert metrics.accuracy_among_valid.count == 0
    assert metrics.accuracy_among_valid.total == 0
    assert metrics.accuracy_among_valid.rate is None

    # human_review
    assert metrics.human_review.count == 0
    assert metrics.human_review.total == 0
    assert metrics.human_review.rate is None

    # outcome totals
    assert metrics.outcomes.extraction_failure == 0
    assert metrics.outcomes.schema_failure == 0
    assert metrics.outcomes.policy_failure == 0
    assert metrics.outcomes.valid_incorrect == 0
    assert metrics.outcomes.valid_correct == 0

    # baselines
    assert metrics.baselines["majority"].correct_count == 0
    assert metrics.baselines["majority"].total == 0
    assert metrics.baselines["majority"].accuracy is None

    assert metrics.baselines["always_needs_review"].correct_count == 0
    assert metrics.baselines["always_needs_review"].total == 0
    assert metrics.baselines["always_needs_review"].accuracy is None

    # tags
    for tag in (
        DatasetTag.CLEAN,
        DatasetTag.AMBIGUOUS,
        DatasetTag.INJECTION,
        DatasetTag.LONG,
    ):
        assert metrics.tags[tag].count == 0
        assert metrics.tags[tag].total == 0
        assert metrics.tags[tag].rate is None


def _make_valid_metrics_fixture() -> dict[str, Any]:
    return {
        "eligible_count": 5,
        "strict_correct": make_ratio(1, 5),
        "outcomes": OutcomeBreakdown(
            extraction_failure=1,
            schema_failure=1,
            policy_failure=1,
            valid_incorrect=1,
            valid_correct=1,
        ),
        "invalid_outputs": make_ratio(3, 5),
        "accuracy_among_valid": make_ratio(1, 2),
        "human_review": make_ratio(1, 5),
        "tags": {
            DatasetTag.CLEAN: make_ratio(1, 5),
            DatasetTag.AMBIGUOUS: make_ratio(0, 0),
            DatasetTag.INJECTION: make_ratio(0, 0),
            DatasetTag.LONG: make_ratio(0, 0),
        },
        "baselines": {
            "majority": BaselineResult(correct_count=1, total=5, accuracy=0.2),
            "always_needs_review": BaselineResult(
                correct_count=1, total=5, accuracy=0.2
            ),
        },
    }


def test_metrics_rejects_outcome_sum_not_matching_eligible_count() -> None:
    data = _make_valid_metrics_fixture()
    # Outcomes sum to 4 instead of eligible_count 5
    data["outcomes"] = OutcomeBreakdown(
        extraction_failure=1,
        schema_failure=1,
        policy_failure=1,
        valid_incorrect=1,
        valid_correct=0,
    )
    with pytest.raises(ValidationError, match="Sum of outcome counts"):
        EvaluationMetrics(**data)


def test_metrics_rejects_strict_correct_count_disagreeing_with_valid_correct() -> (
    None
):
    data = _make_valid_metrics_fixture()
    # valid_correct is 1, but strict_correct count is 2
    data["strict_correct"] = make_ratio(2, 5)
    with pytest.raises(ValidationError, match="strict_correct count"):
        EvaluationMetrics(**data)


def test_metrics_rejects_invalid_outputs_count_inconsistent() -> None:
    data = _make_valid_metrics_fixture()
    # O1+O2+O3 is 3, but invalid_outputs count is 2
    data["invalid_outputs"] = make_ratio(2, 5)
    with pytest.raises(ValidationError, match="invalid_outputs count"):
        EvaluationMetrics(**data)


def test_metrics_rejects_accuracy_among_valid_total_inconsistent() -> None:
    data = _make_valid_metrics_fixture()
    # valid outputs (O4+O5) is 2, but accuracy_among_valid total is 3
    data["accuracy_among_valid"] = make_ratio(1, 3)
    with pytest.raises(ValidationError, match="accuracy_among_valid total"):
        EvaluationMetrics(**data)


def test_metrics_rejects_accuracy_among_valid_count_inconsistent() -> None:
    data = _make_valid_metrics_fixture()
    # valid_correct is 1, but accuracy_among_valid count is 0
    data["accuracy_among_valid"] = make_ratio(0, 2)
    with pytest.raises(ValidationError, match="accuracy_among_valid count"):
        EvaluationMetrics(**data)


