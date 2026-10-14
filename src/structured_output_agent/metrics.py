import math
import statistics
from collections.abc import Sequence

from pydantic import Field, StrictInt, model_validator

from .baselines import (
    always_needs_review_decision,
    majority_decision,
    strict_correctness,
)
from .injection_scorer import InjectionScoreReport, evaluate_run_injections
from .run_results import RunOutcome
from .runner import EvaluationRun, RunStatus, compute_dataset_fingerprint
from .schemas import (
    DatasetRecord,
    DatasetTag,
    ExpectedDecision,
    Priority,
    RatioMetric,
    StrictModel,
    make_ratio,
)


def wilson_score_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successful outcomes (0 <= successes <= total).
        total: Total number of observations (total >= 0).
        confidence_level: Two-sided confidence level (0.0 < confidence_level < 1.0),
            defaulting to 0.95 (95%).

    Returns:
        (lower, upper) bound tuple clamped to [0.0, 1.0], or (None, None) if total == 0.

    Limitations:
        Confidence intervals reflect sampling uncertainty under the assumption of
        independent and identically distributed observations. A confidence interval
        on a small benchmark dataset (e.g. 20 development tickets) must NOT be presented
        as evidence that the benchmark is representative of a wider production
        population or unseen data.
    """
    if total == 0:
        return None, None
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(
            f"Invalid binomial counts: successes={successes}, total={total}."
        )
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(
            "confidence_level must be strictly between 0 and 1; "
            f"got {confidence_level}."
        )

    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    z2 = z * z
    p_hat = successes / total
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total))
    )

    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return lower, upper


class ConfidenceInterval(StrictModel):
    """Wilson score confidence interval for a binomial proportion."""

    lower: float | None = Field(default=None, ge=0.0, le=1.0)
    upper: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> "ConfidenceInterval":
        if self.lower is None or self.upper is None:
            if self.lower is not None or self.upper is not None:
                raise ValueError("lower and upper must both be None or both be floats.")
        else:
            if self.lower > self.upper:
                raise ValueError(
                    f"lower bound ({self.lower}) cannot exceed "
                    f"upper bound ({self.upper})."
                )
        return self


def make_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Construct ConfidenceInterval using the Wilson score formula."""
    lower, upper = wilson_score_interval(
        successes, total, confidence_level=confidence_level
    )
    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


class OutcomeBreakdown(StrictModel):
    """Counts for each mutually exclusive outcome category O1 through O5."""

    extraction_failure: StrictInt = Field(ge=0)  # O1
    schema_failure: StrictInt = Field(ge=0)  # O2
    policy_failure: StrictInt = Field(ge=0)  # O3
    valid_incorrect: StrictInt = Field(ge=0)  # O4
    valid_correct: StrictInt = Field(ge=0)  # O5


class BaselineResult(StrictModel):
    """Reference baseline performance recomputed on eligible records."""

    correct_count: StrictInt = Field(ge=0)
    total: StrictInt = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_accuracy(self) -> "BaselineResult":
        if self.total == 0:
            if self.accuracy is not None:
                raise ValueError("accuracy must be None when total is 0.")
            if self.correct_count != 0:
                raise ValueError("correct_count must be 0 when total is 0.")
        else:
            if self.accuracy is None:
                raise ValueError("accuracy must be provided when total > 0.")
            expected = self.correct_count / self.total
            if abs(self.accuracy - expected) > 1e-6:
                raise ValueError(
                    f"accuracy {self.accuracy} does not match "
                    f"{self.correct_count} / {self.total}."
                )
        return self


class ConfusionMatrix(StrictModel):
    """Confusion matrix counts for binary classification."""

    true_positives: StrictInt = Field(ge=0)
    false_positives: StrictInt = Field(ge=0)
    true_negatives: StrictInt = Field(ge=0)
    false_negatives: StrictInt = Field(ge=0)
    unclassified: StrictInt = Field(default=0, ge=0)


class HumanReviewMetrics(StrictModel):
    """Metrics for requires_human_review (precision, recall, confusion matrix)."""

    confusion_matrix: ConfusionMatrix
    precision: RatioMetric
    recall: RatioMetric
    specificity: RatioMetric
    positive_class: str = "requires_human_review=True"
    negative_class: str = "requires_human_review=False"


class FieldAccuracyMetrics(StrictModel):
    """Accuracy across each individual triage decision field."""

    category: RatioMetric
    priority: RatioMetric
    assigned_team: RatioMetric
    requires_human_review: RatioMetric
    denominator_rule: str = "all_eligible_records"


class EscalationMetrics(StrictModel):
    """Metrics tracking over- and under-escalation relative to ground-truth labels."""

    priority_over_escalation: RatioMetric
    priority_under_escalation: RatioMetric
    urgent_over_escalation: RatioMetric
    urgent_under_escalation: RatioMetric
    limitation_note: str = (
        "The repository schemas lack an authoritative escalation field or policy. "
        "Escalation is evaluated via ordered priority levels (low < normal < "
        "high < urgent) and urgent-specific assignment. requires_human_review "
        "is tracked separately under human_review_metrics and is not equated "
        "with escalation."
    )


class EvaluationMetrics(StrictModel):
    """Typed evaluation metrics computed from a completed evaluation run."""

    eligible_count: StrictInt = Field(ge=0)
    strict_correct: RatioMetric
    outcomes: OutcomeBreakdown
    invalid_outputs: RatioMetric
    accuracy_among_valid: RatioMetric
    human_review: RatioMetric
    tags: dict[DatasetTag, RatioMetric]
    baselines: dict[str, BaselineResult]

    # Expanded metrics (optional with None default for backwards compatibility)
    schema_valid: RatioMetric | None = None
    field_accuracy: FieldAccuracyMetrics | None = None
    human_review_metrics: HumanReviewMetrics | None = None
    escalation_metrics: EscalationMetrics | None = None
    confidence_intervals: dict[str, ConfidenceInterval] | None = None
    injection: InjectionScoreReport | None = None

    @model_validator(mode="after")
    def validate_metrics_invariants(self) -> "EvaluationMetrics":
        sum_outcomes = (
            self.outcomes.extraction_failure
            + self.outcomes.schema_failure
            + self.outcomes.policy_failure
            + self.outcomes.valid_incorrect
            + self.outcomes.valid_correct
        )
        if sum_outcomes != self.eligible_count:
            raise ValueError(
                f"Sum of outcome counts ({sum_outcomes}) does not equal "
                f"eligible_count ({self.eligible_count})."
            )

        if self.strict_correct.total != self.eligible_count:
            raise ValueError(
                f"strict_correct total ({self.strict_correct.total}) must "
                f"equal eligible_count ({self.eligible_count})."
            )
        if self.strict_correct.count != self.outcomes.valid_correct:
            raise ValueError(
                f"strict_correct count ({self.strict_correct.count}) must "
                f"equal valid_correct ({self.outcomes.valid_correct})."
            )

        if self.invalid_outputs.total != self.eligible_count:
            raise ValueError(
                f"invalid_outputs total ({self.invalid_outputs.total}) must "
                f"equal eligible_count ({self.eligible_count})."
            )
        sum_invalid = (
            self.outcomes.extraction_failure
            + self.outcomes.schema_failure
            + self.outcomes.policy_failure
        )
        if self.invalid_outputs.count != sum_invalid:
            raise ValueError(
                f"invalid_outputs count ({self.invalid_outputs.count}) must "
                f"equal sum of O1+O2+O3 ({sum_invalid})."
            )

        valid_total = self.outcomes.valid_incorrect + self.outcomes.valid_correct
        if self.accuracy_among_valid.total != valid_total:
            raise ValueError(
                f"accuracy_among_valid total ({self.accuracy_among_valid.total}) "
                f"must equal total valid outputs ({valid_total})."
            )
        if self.accuracy_among_valid.count != self.outcomes.valid_correct:
            raise ValueError(
                f"accuracy_among_valid count ({self.accuracy_among_valid.count}) "
                f"must equal valid_correct ({self.outcomes.valid_correct})."
            )

        if self.human_review.total != self.eligible_count:
            raise ValueError(
                f"human_review total ({self.human_review.total}) must "
                f"equal eligible_count ({self.eligible_count})."
            )

        if self.schema_valid is not None:
            if self.schema_valid.total != self.eligible_count:
                raise ValueError(
                    f"schema_valid total ({self.schema_valid.total}) must "
                    f"equal eligible_count ({self.eligible_count})."
                )
            expected_schema_valid_count = (
                self.outcomes.policy_failure
                + self.outcomes.valid_incorrect
                + self.outcomes.valid_correct
            )
            if self.schema_valid.count != expected_schema_valid_count:
                raise ValueError(
                    f"schema_valid count ({self.schema_valid.count}) must "
                    f"equal sum of O3+O4+O5 ({expected_schema_valid_count})."
                )

        if self.field_accuracy is not None:
            for field_name in (
                "category",
                "priority",
                "assigned_team",
                "requires_human_review",
            ):
                metric = getattr(self.field_accuracy, field_name)
                if metric.total != self.eligible_count:
                    raise ValueError(
                        f"field_accuracy.{field_name} total ({metric.total}) "
                        f"must equal eligible_count ({self.eligible_count})."
                    )

        if self.human_review_metrics is not None:
            cm = self.human_review_metrics.confusion_matrix
            total_cases = (
                cm.true_positives
                + cm.false_positives
                + cm.true_negatives
                + cm.false_negatives
                + cm.unclassified
            )
            if total_cases != self.eligible_count:
                raise ValueError(
                    f"Sum of confusion matrix cases ({total_cases}) must "
                    f"equal eligible_count ({self.eligible_count})."
                )

        if self.escalation_metrics is not None:
            for field_name in (
                "priority_over_escalation",
                "priority_under_escalation",
                "urgent_over_escalation",
                "urgent_under_escalation",
            ):
                metric = getattr(self.escalation_metrics, field_name)
                if metric.total != self.eligible_count:
                    raise ValueError(
                        f"escalation_metrics.{field_name} total ({metric.total}) "
                        f"must equal eligible_count ({self.eligible_count})."
                    )

        return self


def compute_metrics(
    run: EvaluationRun,
    records: Sequence[DatasetRecord] | None = None,
) -> EvaluationMetrics:
    """Compute evaluation metrics from a completed run and its records.

    If records is None, uses the retained records snapshot in run.records.

    Raises ValueError if:
    - The run is aborted.
    - Dataset fingerprint is missing or empty.
    - No retained or explicit records are available when eligible_count > 0.
    - Records count does not match run eligible_count.
    - Result ticket indices do not match the expected zero-based sequence.
    - Dataset fingerprint does not match the provided records.
    """
    if run.status != RunStatus.COMPLETED:
        raise ValueError(
            f"Cannot compute metrics for an evaluation run with status "
            f"{run.status.value!r}. Aborted runs must not produce metrics."
        )

    if run.dataset_fingerprint is None or not run.dataset_fingerprint.strip():
        raise ValueError(
            "EvaluationRun dataset_fingerprint is required and must not be "
            "empty or whitespace to compute metrics."
        )

    effective_records = run.records if records is None else records
    if not effective_records and run.eligible_count > 0:
        if records is None:
            raise ValueError(
                "EvaluationRun has no retained dataset records and no records "
                "were provided."
            )
        raise ValueError(
            f"Records count (0) does not match run eligible_count "
            f"({run.eligible_count})."
        )

    if len(effective_records) != run.eligible_count:
        raise ValueError(
            f"Records count ({len(effective_records)}) does not match run "
            f"eligible_count ({run.eligible_count})."
        )

    if len(run.completed_results) != len(effective_records):
        raise ValueError(
            f"Run completed_results count ({len(run.completed_results)}) does "
            f"not match records count ({len(effective_records)})."
        )

    for expected_index, result in enumerate(run.completed_results):
        if result.ticket_index != expected_index:
            raise ValueError(
                f"Result at position {expected_index} has ticket_index "
                f"{result.ticket_index}, expected {expected_index}."
            )

    actual_fingerprint = compute_dataset_fingerprint(effective_records)
    if actual_fingerprint != run.dataset_fingerprint:
        raise ValueError(
            "Dataset fingerprint mismatch: provided records do not match "
            "the exact eligible records evaluated by the runner."
        )

    n = run.eligible_count

    # 1. Outcomes breakdown O1..O5
    o1 = sum(
        1
        for r in run.completed_results
        if r.outcome == RunOutcome.EXTRACTION_FAILURE
    )
    o2 = sum(
        1 for r in run.completed_results if r.outcome == RunOutcome.SCHEMA_FAILURE
    )
    o3 = sum(
        1 for r in run.completed_results if r.outcome == RunOutcome.POLICY_FAILURE
    )
    o4 = sum(
        1 for r in run.completed_results if r.outcome == RunOutcome.VALID_INCORRECT
    )
    o5 = sum(
        1 for r in run.completed_results if r.outcome == RunOutcome.VALID_CORRECT
    )
    outcomes = OutcomeBreakdown(
        extraction_failure=o1,
        schema_failure=o2,
        policy_failure=o3,
        valid_incorrect=o4,
        valid_correct=o5,
    )

    # 2. Strict correct
    strict_correct = make_ratio(o5, n)

    # 3. Invalid outputs
    invalid_outputs = make_ratio(o1 + o2 + o3, n)

    # 4. Accuracy among valid outputs (O5 / (O4 + O5))
    accuracy_among_valid = make_ratio(o5, o4 + o5)

    # 5. Human review rate
    human_review_count = sum(
        1
        for r in run.completed_results
        if r.prediction is not None and r.prediction.requires_human_review
    )
    human_review = make_ratio(human_review_count, n)

    # 6. Tag breakdown (independent denominators per tag)
    tag_metrics: dict[DatasetTag, RatioMetric] = {}
    for tag in (
        DatasetTag.CLEAN,
        DatasetTag.AMBIGUOUS,
        DatasetTag.INJECTION,
        DatasetTag.LONG,
    ):
        tag_indices = [
            i for i, rec in enumerate(effective_records) if tag in rec.tags
        ]
        tag_total = len(tag_indices)
        tag_correct = sum(run.completed_results[i].credit for i in tag_indices)
        tag_metrics[tag] = make_ratio(tag_correct, tag_total)

    # 7. Dynamic reference baselines
    if n > 0:
        maj_dec = majority_decision(effective_records)
        maj_correct = sum(strict_correctness(maj_dec, effective_records))
        maj_accuracy = maj_correct / n
        maj_result = BaselineResult(
            correct_count=maj_correct,
            total=n,
            accuracy=maj_accuracy,
        )

        anr_dec = always_needs_review_decision()
        anr_correct = sum(strict_correctness(anr_dec, effective_records))
        anr_accuracy = anr_correct / n
        anr_result = BaselineResult(
            correct_count=anr_correct,
            total=n,
            accuracy=anr_accuracy,
        )
    else:
        maj_result = BaselineResult(correct_count=0, total=0, accuracy=None)
        anr_result = BaselineResult(correct_count=0, total=0, accuracy=None)

    baselines = {
        "majority": maj_result,
        "always_needs_review": anr_result,
    }

    # 8. Schema-valid rate (outputs conforming to TicketTriageOutput schema: O3+O4+O5)
    schema_valid = make_ratio(o3 + o4 + o5, n)

    # 9. Per-field accuracy (evaluated against accepted decisions;
    # denominator is all eligible tickets)
    cat_correct = 0
    prio_correct = 0
    team_correct = 0
    hr_correct = 0
    for i, record in enumerate(effective_records):
        result = run.completed_results[i]
        if result.prediction is None:
            continue
        pred = result.prediction
        decisions = (
            [record.expected]
            if isinstance(record.expected, ExpectedDecision)
            else record.expected
        )
        if any(pred.category == d.category for d in decisions):
            cat_correct += 1
        if any(pred.priority == d.priority for d in decisions):
            prio_correct += 1
        if any(pred.assigned_team == d.assigned_team for d in decisions):
            team_correct += 1
        if any(
            pred.requires_human_review == d.requires_human_review
            for d in decisions
        ):
            hr_correct += 1

    field_accuracy = FieldAccuracyMetrics(
        category=make_ratio(cat_correct, n),
        priority=make_ratio(prio_correct, n),
        assigned_team=make_ratio(team_correct, n),
        requires_human_review=make_ratio(hr_correct, n),
        denominator_rule="all_eligible_records",
    )

    # 10. Human-review metrics (confusion matrix, precision, recall, specificity)
    tp = fp = tn = fn = unclass = 0
    for i, record in enumerate(effective_records):
        result = run.completed_results[i]
        if result.prediction is None:
            unclass += 1
            continue
        pred_hr = result.prediction.requires_human_review
        decisions = (
            [record.expected]
            if isinstance(record.expected, ExpectedDecision)
            else record.expected
        )
        exp_hr = any(d.requires_human_review for d in decisions)
        if pred_hr and exp_hr:
            tp += 1
        elif pred_hr and not exp_hr:
            fp += 1
        elif not pred_hr and not exp_hr:
            tn += 1
        else:
            fn += 1

    cm = ConfusionMatrix(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        unclassified=unclass,
    )
    human_review_metrics = HumanReviewMetrics(
        confusion_matrix=cm,
        precision=make_ratio(tp, tp + fp),
        recall=make_ratio(tp, tp + fn),
        specificity=make_ratio(tn, tn + fp),
    )

    # 11. Over- and under-escalation
    priority_order = {
        Priority.LOW: 0,
        Priority.NORMAL: 1,
        Priority.HIGH: 2,
        Priority.URGENT: 3,
    }
    prio_over = 0
    prio_under = 0
    urg_over = 0
    urg_under = 0
    for i, record in enumerate(effective_records):
        result = run.completed_results[i]
        if result.prediction is None:
            continue
        pred_prio = priority_order[result.prediction.priority]
        decisions = (
            [record.expected]
            if isinstance(record.expected, ExpectedDecision)
            else record.expected
        )
        exp_prios = [priority_order[d.priority] for d in decisions]
        min_exp = min(exp_prios)
        max_exp = max(exp_prios)
        if pred_prio > max_exp:
            prio_over += 1
        elif pred_prio < min_exp:
            prio_under += 1

        is_pred_urgent = result.prediction.priority == Priority.URGENT
        has_exp_urgent = any(d.priority == Priority.URGENT for d in decisions)
        if is_pred_urgent and not has_exp_urgent:
            urg_over += 1
        elif not is_pred_urgent and has_exp_urgent:
            urg_under += 1

    escalation_metrics = EscalationMetrics(
        priority_over_escalation=make_ratio(prio_over, n),
        priority_under_escalation=make_ratio(prio_under, n),
        urgent_over_escalation=make_ratio(urg_over, n),
        urgent_under_escalation=make_ratio(urg_under, n),
    )

    # 12. Wilson score confidence intervals
    confidence_intervals = {
        "strict_correct": make_confidence_interval(o5, n),
        "accuracy_among_valid": make_confidence_interval(o5, o4 + o5),
        "schema_valid": make_confidence_interval(o3 + o4 + o5, n),
        "human_review_precision": make_confidence_interval(tp, tp + fp),
        "human_review_recall": make_confidence_interval(tp, tp + fn),
    }

    # 13. Offline injection resilience scoring
    injection_report = evaluate_run_injections(
        effective_records, run.completed_results
    )

    return EvaluationMetrics(
        eligible_count=n,
        strict_correct=strict_correct,
        outcomes=outcomes,
        invalid_outputs=invalid_outputs,
        accuracy_among_valid=accuracy_among_valid,
        human_review=human_review,
        tags=tag_metrics,
        baselines=baselines,
        schema_valid=schema_valid,
        field_accuracy=field_accuracy,
        human_review_metrics=human_review_metrics,
        escalation_metrics=escalation_metrics,
        confidence_intervals=confidence_intervals,
        injection=injection_report,
    )

