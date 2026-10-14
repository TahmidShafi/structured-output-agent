from typing import Any

import pytest
from pydantic import ValidationError

from structured_output_agent.dataset_io import load_dataset
from structured_output_agent.metrics import (
    BaselineResult,
    ConfidenceInterval,
    EvaluationMetrics,
    OutcomeBreakdown,
    RatioMetric,
    compute_metrics,
    make_confidence_interval,
    make_ratio,
    wilson_score_interval,
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


def test_wilson_score_interval_properties_and_edge_cases() -> None:
    # 0 observations returns (None, None)
    assert wilson_score_interval(0, 0) == (None, None)

    # 0 successes: lower is strictly 0.0, upper matches formula
    low_0, up_0 = wilson_score_interval(0, 10)
    assert low_0 == 0.0
    assert up_0 == pytest.approx(0.27753, abs=1e-4)

    # All successes: lower matches formula, upper is strictly 1.0
    low_10, up_10 = wilson_score_interval(10, 10)
    assert low_10 == pytest.approx(0.72246, abs=1e-4)
    assert up_10 == 1.0

    # Partial success: symmetric around 0.5
    low_5, up_5 = wilson_score_interval(5, 10)
    assert low_5 == pytest.approx(0.23659, abs=1e-4)
    assert up_5 == pytest.approx(0.76340, abs=1e-4)
    assert (low_5 + up_5) == pytest.approx(1.0)

    # Custom confidence levels
    low_90, up_90 = wilson_score_interval(5, 10, confidence_level=0.90)
    low_99, up_99 = wilson_score_interval(5, 10, confidence_level=0.99)
    # Higher confidence -> wider interval
    assert low_99 < low_90 < 0.5 < up_90 < up_99

    # Input validation
    with pytest.raises(ValueError, match="Invalid binomial counts"):
        wilson_score_interval(-1, 10)
    with pytest.raises(ValueError, match="Invalid binomial counts"):
        wilson_score_interval(11, 10)
    with pytest.raises(ValueError, match="Invalid binomial counts"):
        wilson_score_interval(0, -5)
    with pytest.raises(
        ValueError, match="confidence_level must be strictly between 0 and 1"
    ):
        wilson_score_interval(5, 10, confidence_level=0.0)
    with pytest.raises(
        ValueError, match="confidence_level must be strictly between 0 and 1"
    ):
        wilson_score_interval(5, 10, confidence_level=1.0)

    # ConfidenceInterval model
    ci = make_confidence_interval(5, 10)
    assert ci.lower == pytest.approx(0.23659, abs=1e-4)
    assert ci.upper == pytest.approx(0.76340, abs=1e-4)
    assert ci.confidence_level == 0.95

    # Empty ConfidenceInterval
    ci_empty = make_confidence_interval(0, 0)
    assert ci_empty.lower is None
    assert ci_empty.upper is None

    # Invalid ConfidenceInterval bounds
    with pytest.raises(ValueError, match="cannot exceed upper bound"):
        ConfidenceInterval(lower=0.8, upper=0.2)
    with pytest.raises(ValueError, match="both be None or both be floats"):
        ConfidenceInterval(lower=0.2, upper=None)


def test_schema_valid_rate_across_outcomes() -> None:
    records = [_make_record(i) for i in range(5)]
    valid_pred = _make_prediction()
    results = [
        # O1 parse failure (schema-invalid)
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.EXTRACTION_FAILURE,
            credit=0,
            error_type="JSONExtractionError",
            error_message="Fail",
        ),
        # O2 schema failure (schema-invalid)
        TicketRunResult(
            ticket_index=1,
            outcome=RunOutcome.SCHEMA_FAILURE,
            credit=0,
            error_type="ValidationError",
            error_message="Field missing",
        ),
        # O3 policy failure (schema-valid: parsed and conformed to TicketTriageOutput)
        TicketRunResult(
            ticket_index=2,
            outcome=RunOutcome.POLICY_FAILURE,
            credit=0,
            error_type="ValueError",
            error_message="Policy violation",
            prediction=valid_pred,
        ),
        # O4 valid incorrect (schema-valid)
        TicketRunResult(
            ticket_index=3,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(
                category=TicketCategory.OTHER, team=AssignedTeam.GENERAL_SUPPORT
            ),
        ),
        # O5 valid correct (schema-valid)
        TicketRunResult(
            ticket_index=4,
            outcome=RunOutcome.VALID_CORRECT,
            credit=1,
            prediction=valid_pred,
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

    assert metrics.schema_valid is not None
    assert metrics.schema_valid.count == 3  # O3 + O4 + O5
    assert metrics.schema_valid.total == 5
    assert metrics.schema_valid.rate == pytest.approx(0.60)

    # Check Wilson interval for schema_valid
    assert metrics.confidence_intervals is not None
    ci_schema = metrics.confidence_intervals["schema_valid"]
    assert ci_schema.lower is not None and ci_schema.upper is not None
    assert ci_schema.lower < 0.60 < ci_schema.upper


def test_field_accuracy_perfect_and_isolated_field_errors() -> None:
    # 5 tickets where expected is BILLING, NORMAL, PAYMENTS, False
    records = [_make_record(i) for i in range(5)]
    results = [
        # 0: Perfect
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.VALID_CORRECT,
            credit=1,
            prediction=_make_prediction(
                category=TicketCategory.BILLING,
                priority=Priority.NORMAL,
                team=AssignedTeam.PAYMENTS,
                requires_human_review=False,
            ),
        ),
        # 1: Priority wrong only (HIGH instead of NORMAL)
        TicketRunResult(
            ticket_index=1,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(
                category=TicketCategory.BILLING,
                priority=Priority.HIGH,
                team=AssignedTeam.PAYMENTS,
                requires_human_review=False,
            ),
        ),
        # 2: Category and Team wrong (SHIPPING and FULFILLMENT)
        TicketRunResult(
            ticket_index=2,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(
                category=TicketCategory.SHIPPING,
                priority=Priority.NORMAL,
                team=AssignedTeam.FULFILLMENT,
                requires_human_review=False,
            ),
        ),
        # 3: requires_human_review wrong only (True instead of False)
        TicketRunResult(
            ticket_index=3,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(
                category=TicketCategory.BILLING,
                priority=Priority.NORMAL,
                team=AssignedTeam.PAYMENTS,
                requires_human_review=True,
            ),
        ),
        # 4: Extraction failure (no prediction, 0 credit across all fields)
        TicketRunResult(
            ticket_index=4,
            outcome=RunOutcome.EXTRACTION_FAILURE,
            credit=0,
            error_type="JSONExtractionError",
            error_message="Fail",
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

    assert metrics.field_accuracy is not None
    # Category: correct on 0, 1, 3 -> 3 / 5 (60%)
    assert metrics.field_accuracy.category.count == 3
    assert metrics.field_accuracy.category.total == 5
    assert metrics.field_accuracy.category.rate == pytest.approx(0.60)

    # Priority: correct on 0, 2, 3 -> 3 / 5 (60%)
    assert metrics.field_accuracy.priority.count == 3
    assert metrics.field_accuracy.priority.total == 5
    assert metrics.field_accuracy.priority.rate == pytest.approx(0.60)

    # Assigned team: correct on 0, 1, 3 -> 3 / 5 (60%)
    assert metrics.field_accuracy.assigned_team.count == 3
    assert metrics.field_accuracy.assigned_team.total == 5
    assert metrics.field_accuracy.assigned_team.rate == pytest.approx(0.60)

    # requires_human_review: correct on 0, 1, 2 -> 3 / 5 (60%)
    assert metrics.field_accuracy.requires_human_review.count == 3
    assert metrics.field_accuracy.requires_human_review.total == 5
    assert metrics.field_accuracy.requires_human_review.rate == pytest.approx(0.60)


def test_field_accuracy_ambiguous_two_accepted_decisions() -> None:
    # Ambiguous record accepts both NORMAL and HIGH priorities
    record = _make_record(0, tags={DatasetTag.AMBIGUOUS})
    pred_high = _make_prediction(
        category=TicketCategory.NEEDS_REVIEW,
        priority=Priority.HIGH,
        team=AssignedTeam.HUMAN_REVIEW,
        requires_human_review=True,
    )
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=1,
        dataset_fingerprint=compute_dataset_fingerprint([record]),
        completed_results=[
            TicketRunResult(
                ticket_index=0,
                outcome=RunOutcome.VALID_CORRECT,
                credit=1,
                prediction=pred_high,
            )
        ],
    )
    metrics = compute_metrics(run, [record])
    assert metrics.field_accuracy is not None
    assert metrics.field_accuracy.priority.count == 1
    assert metrics.field_accuracy.priority.rate == pytest.approx(1.0)


def test_human_review_confusion_matrix_and_ratios() -> None:
    # 5 records: 0: TP, 1: FP, 2: TN, 3: FN, 4: unclassified (parse failure)
    records = [
        _make_record(0, requires_human_review=True),
        _make_record(1, requires_human_review=False),
        _make_record(2, requires_human_review=False),
        _make_record(3, requires_human_review=True),
        _make_record(4, requires_human_review=False),
    ]
    results = [
        # 0: pred=True, exp=True -> TP
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.VALID_CORRECT,
            credit=1,
            prediction=_make_prediction(requires_human_review=True),
        ),
        # 1: pred=True, exp=False -> FP
        TicketRunResult(
            ticket_index=1,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(requires_human_review=True),
        ),
        # 2: pred=False, exp=False -> TN
        TicketRunResult(
            ticket_index=2,
            outcome=RunOutcome.VALID_CORRECT,
            credit=1,
            prediction=_make_prediction(requires_human_review=False),
        ),
        # 3: pred=False, exp=True -> FN
        TicketRunResult(
            ticket_index=3,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(requires_human_review=False),
        ),
        # 4: Extraction failure -> unclassified
        TicketRunResult(
            ticket_index=4,
            outcome=RunOutcome.EXTRACTION_FAILURE,
            credit=0,
            error_type="JSONExtractionError",
            error_message="Fail",
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

    assert metrics.human_review_metrics is not None
    cm = metrics.human_review_metrics.confusion_matrix
    assert cm.true_positives == 1
    assert cm.false_positives == 1
    assert cm.true_negatives == 1
    assert cm.false_negatives == 1
    assert cm.unclassified == 1

    # Precision = TP / (TP + FP) = 1 / 2
    assert metrics.human_review_metrics.precision.count == 1
    assert metrics.human_review_metrics.precision.total == 2
    assert metrics.human_review_metrics.precision.rate == pytest.approx(0.50)

    # Recall = TP / (TP + FN) = 1 / 2
    assert metrics.human_review_metrics.recall.count == 1
    assert metrics.human_review_metrics.recall.total == 2
    assert metrics.human_review_metrics.recall.rate == pytest.approx(0.50)

    # Specificity = TN / (TN + FP) = 1 / 2
    assert metrics.human_review_metrics.specificity.count == 1
    assert metrics.human_review_metrics.specificity.total == 2
    assert metrics.human_review_metrics.specificity.rate == pytest.approx(0.50)


def test_human_review_zero_denominators() -> None:
    records = [_make_record(0, requires_human_review=False)]
    results = [
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.VALID_CORRECT,
            credit=1,
            prediction=_make_prediction(requires_human_review=False),
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

    assert metrics.human_review_metrics is not None
    cm = metrics.human_review_metrics.confusion_matrix
    assert cm.true_positives == 0
    assert cm.false_positives == 0
    assert cm.true_negatives == 1
    assert cm.false_negatives == 0
    assert cm.unclassified == 0

    # Precision: TP + FP = 0 -> rate is None
    assert metrics.human_review_metrics.precision.total == 0
    assert metrics.human_review_metrics.precision.rate is None

    # Recall: TP + FN = 0 -> rate is None
    assert metrics.human_review_metrics.recall.total == 0
    assert metrics.human_review_metrics.recall.rate is None

    # Specificity: TN / (TN + FP) = 1 / 1 = 1.0
    assert metrics.human_review_metrics.specificity.total == 1
    assert metrics.human_review_metrics.specificity.rate == pytest.approx(1.0)


def test_over_and_under_escalation_metrics() -> None:
    # 4 tickets:
    # 0: expected LOW, predicted NORMAL -> prio_over
    # 1: expected HIGH, predicted NORMAL -> prio_under
    # 2: expected NORMAL, predicted URGENT -> urg_over (and prio_over)
    # 3: expected URGENT, predicted NORMAL -> urg_under (and prio_under)
    records = [
        _make_record(0, priority=Priority.LOW),
        _make_record(1, priority=Priority.HIGH),
        _make_record(2, priority=Priority.NORMAL),
        _make_record(3, priority=Priority.URGENT, requires_human_review=True),
    ]
    results = [
        # 0: pred=NORMAL, exp=LOW -> over-priority
        TicketRunResult(
            ticket_index=0,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(priority=Priority.NORMAL),
        ),
        # 1: pred=NORMAL, exp=HIGH -> under-priority
        TicketRunResult(
            ticket_index=1,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(priority=Priority.NORMAL),
        ),
        # 2: pred=URGENT, exp=NORMAL -> over-urgent, over-priority
        TicketRunResult(
            ticket_index=2,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(
                priority=Priority.URGENT, requires_human_review=True
            ),
        ),
        # 3: pred=NORMAL, exp=URGENT -> under-urgent, under-priority
        TicketRunResult(
            ticket_index=3,
            outcome=RunOutcome.VALID_INCORRECT,
            credit=0,
            prediction=_make_prediction(
                priority=Priority.NORMAL, requires_human_review=False
            ),
        ),
    ]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=4,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        completed_results=results,
    )
    metrics = compute_metrics(run, records)

    assert metrics.escalation_metrics is not None
    # prio_over on tickets 0 and 2 -> 2/4
    assert metrics.escalation_metrics.priority_over_escalation.count == 2
    assert metrics.escalation_metrics.priority_over_escalation.total == 4
    assert metrics.escalation_metrics.priority_over_escalation.rate == pytest.approx(
        0.50
    )

    # prio_under on tickets 1 and 3 -> 2/4
    assert metrics.escalation_metrics.priority_under_escalation.count == 2
    assert metrics.escalation_metrics.priority_under_escalation.total == 4
    assert metrics.escalation_metrics.priority_under_escalation.rate == pytest.approx(
        0.50
    )

    # urg_over on ticket 2 -> 1/4
    assert metrics.escalation_metrics.urgent_over_escalation.count == 1
    assert metrics.escalation_metrics.urgent_over_escalation.total == 4
    assert metrics.escalation_metrics.urgent_over_escalation.rate == pytest.approx(
        0.25
    )

    # urg_under on ticket 3 -> 1/4
    assert metrics.escalation_metrics.urgent_under_escalation.count == 1
    assert metrics.escalation_metrics.urgent_under_escalation.total == 4
    assert metrics.escalation_metrics.urgent_under_escalation.rate == pytest.approx(
        0.25
    )

    # Limitation note is present and documents distinction from human review
    assert (
        "authoritative escalation field"
        in metrics.escalation_metrics.limitation_note
    )
    assert (
        "not equated with escalation"
        in metrics.escalation_metrics.limitation_note
    )


def test_injection_resilience_metric_integration() -> None:
    records = [
        _make_record(0, tags={DatasetTag.CLEAN}),
        _make_record(1, tags={DatasetTag.INJECTION}),
    ]
    results = [
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
            raw_response="Clean response without echoed marker",
        ),
    ]
    run = EvaluationRun(
        config=RunConfig(),
        status=RunStatus.COMPLETED,
        eligible_count=2,
        dataset_fingerprint=compute_dataset_fingerprint(records),
        completed_results=results,
    )
    metrics = compute_metrics(run, records)

    assert metrics.injection is not None
    assert metrics.injection.eligible_count == 2
    assert metrics.injection.injection_ticket_count == 1
    assert metrics.injection.tickets_resisted.count == 1
    assert metrics.injection.tickets_resisted.total == 1
    assert metrics.injection.tickets_resisted.rate == pytest.approx(1.0)
    assert metrics.injection.targets_resisted.count == 1
    assert metrics.injection.targets_resisted.total == 1
    assert metrics.injection.targets_resisted.rate == pytest.approx(1.0)
    assert "summary_echo" in metrics.injection.per_kind
    assert metrics.injection.per_kind["summary_echo"].rate == pytest.approx(1.0)


def test_report_serialization_and_backward_compatibility() -> None:
    from structured_output_agent.reporting import EvaluationReport, generate_report

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
    report = generate_report(run, records)
    json_str = report.to_json()
    reconstructed = EvaluationReport.from_json(json_str)

    assert reconstructed == report
    assert reconstructed.metrics.schema_valid is not None
    assert reconstructed.metrics.schema_valid.count == 1
    assert reconstructed.metrics.field_accuracy is not None
    assert reconstructed.metrics.human_review_metrics is not None
    assert reconstructed.metrics.escalation_metrics is not None
    assert reconstructed.metrics.confidence_intervals is not None



