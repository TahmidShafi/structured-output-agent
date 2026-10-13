from collections.abc import Sequence

from pydantic import Field, StrictInt, model_validator

from .baselines import (
    always_needs_review_decision,
    majority_decision,
    strict_correctness,
)
from .run_results import RunOutcome
from .runner import EvaluationRun, RunStatus, compute_dataset_fingerprint
from .schemas import DatasetRecord, DatasetTag, StrictModel


class RatioMetric(StrictModel):
    """Represents a count, total denominator, and optional rate.

    Rate is float in [0.0, 1.0] when total > 0, and None when total == 0.
    """

    count: StrictInt = Field(ge=0)
    total: StrictInt = Field(ge=0)
    rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_rate(self) -> "RatioMetric":
        if self.total == 0:
            if self.rate is not None:
                raise ValueError("rate must be None when total is 0.")
            if self.count != 0:
                raise ValueError("count must be 0 when total is 0.")
        else:
            if self.rate is None:
                raise ValueError("rate must be provided when total > 0.")
            expected = self.count / self.total
            if abs(self.rate - expected) > 1e-6:
                raise ValueError(
                    f"rate {self.rate} does not match {self.count} / {self.total}."
                )
        return self


def make_ratio(count: int, total: int) -> RatioMetric:
    """Construct RatioMetric, setting rate to None if total is 0."""
    rate = count / total if total > 0 else None
    return RatioMetric(count=count, total=total, rate=rate)


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

    return EvaluationMetrics(
        eligible_count=n,
        strict_correct=strict_correct,
        outcomes=outcomes,
        invalid_outputs=invalid_outputs,
        accuracy_among_valid=accuracy_among_valid,
        human_review=human_review,
        tags=tag_metrics,
        baselines=baselines,
    )
