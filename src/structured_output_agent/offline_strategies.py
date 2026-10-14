"""Offline evaluation strategies 1-3 using persisted evidence and paired analysis.

Strategies:
1. Strategy 1 (No retry): Scores the persisted first response.
2. Strategy 2 (Schema-aware retry): Applies retry/recovery logic with sanitized
   schema feedback (O1/O2 eligible for retry; O3 policy failures are not retried).
3. Strategy 3 (Schema- and policy-aware retry): Applies both schema and C1 policy
   feedback (O1, O2, and O3 are all eligible for retry).

Enforces:
- Identical eligible tickets across all strategies for paired evaluation.
- Preserves per-ticket initial and final outcomes, transitions, and attempt counts.
- Explicit labeling of simulated retries as synthetic (never presented as
  observed Gemini performance).
- Explicit identification of missing evidence or incomplete checkpoints.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, StrictBool, StrictInt

from .metrics import EvaluationMetrics, compute_metrics
from .run_config import RunConfig
from .run_results import (
    FAILURE_OUTCOMES,
    RunOutcome,
    TicketRunResult,
    classify_response,
)
from .runner import EvaluationRun, RunStatus, compute_dataset_fingerprint
from .schemas import (
    DatasetRecord,
    RatioMetric,
    StrictModel,
    TicketTriageOutput,
    make_ratio,
)


class EvaluationStrategy(StrEnum):
    STRATEGY_1_NO_RETRY = "strategy_1_no_retry"
    STRATEGY_2_SCHEMA_RETRY = "strategy_2_schema_retry"
    STRATEGY_3_SCHEMA_POLICY_RETRY = "strategy_3_schema_policy_retry"


class StrategyTicketResult(StrictModel):
    ticket_index: StrictInt = Field(ge=0)
    initial_outcome: RunOutcome
    final_outcome: RunOutcome
    outcome_transition: str
    attempts_used: StrictInt = Field(ge=1)
    recovered: StrictBool
    credit: Literal[0, 1]
    prediction: TicketTriageOutput | None = None
    is_simulated: StrictBool = False


class RetryAccounting(StrictModel):
    total_attempts: StrictInt = Field(ge=0)
    tickets_retried: StrictInt = Field(ge=0)
    tickets_recovered: StrictInt = Field(ge=0)
    tickets_exhausted: StrictInt = Field(ge=0)
    recovery_rate: RatioMetric


class StrategyEvaluationResult(StrictModel):
    strategy: EvaluationStrategy
    eligible_count: StrictInt = Field(ge=0)
    metrics: EvaluationMetrics
    retry_accounting: RetryAccounting
    ticket_results: list[StrategyTicketResult]
    is_synthetic_simulation: StrictBool
    simulation_note: str | None = None


class PairedComparison(StrictModel):
    baseline_strategy: EvaluationStrategy
    comparison_strategy: EvaluationStrategy
    strict_accuracy_delta: float
    schema_valid_delta: float
    recovered_tickets: StrictInt = Field(ge=0)
    regressed_tickets: StrictInt = Field(ge=0)
    unchanged_tickets: StrictInt = Field(ge=0)


class PairedEvaluationReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    created_at: str
    dataset_fingerprint: str
    eligible_count: StrictInt = Field(ge=0)
    strategies: dict[EvaluationStrategy, StrategyEvaluationResult]
    paired_comparisons: list[PairedComparison]
    evidence_source: str
    limitations_note: str


# Type for custom offline retry simulation callback:
# (ticket_index, record, attempt, outcome, error_msg) -> raw_response
RetrySimulator = Callable[[int, DatasetRecord, int, RunOutcome, str | None], str]


def _build_evaluation_run(
    records: Sequence[DatasetRecord],
    dataset_fingerprint: str,
    results: list[TicketRunResult],
    config: RunConfig,
) -> EvaluationRun:
    """Construct a completed EvaluationRun to compute standard EvaluationMetrics."""
    return EvaluationRun(
        config=config,
        status=RunStatus.COMPLETED,
        eligible_count=len(records),
        dataset_fingerprint=dataset_fingerprint,
        records=list(records),
        completed_results=results,
        client_type="offline_paired_evaluator",
        model_id=None,
        created_at=datetime.now(UTC).isoformat(),
    )


def evaluate_offline_strategies(
    records: Sequence[DatasetRecord],
    initial_results: Sequence[TicketRunResult],
    *,
    evidence_source: str,
    retry_simulator: RetrySimulator | None = None,
    max_retries: int = 2,
) -> PairedEvaluationReport:
    """Evaluate Strategy 1, Strategy 2, and Strategy 3 on identical tickets.

    Args:
        records: Canonical eligible development dataset records.
        initial_results: Persisted first-attempt results for each ticket.
        evidence_source: Descriptive identifier of the persisted evidence.
        retry_simulator: Optional callback providing simulated retry responses.
            If None, tickets with failures cannot recover and exhaust retries.
        max_retries: Configurable retry budget (default 2, allowing up to 3 attempts).

    Raises:
        ValueError: If record count does not match initial_results, or if ticket
            indices are missing, duplicated, or out of order.
    """
    eligible_count = len(records)
    if eligible_count == 0:
        raise ValueError("Cannot evaluate strategies on empty records.")

    if len(initial_results) != eligible_count:
        raise ValueError(
            f"Incomplete evidence: initial_results count ({len(initial_results)}) "
            f"does not match eligible dataset records count ({eligible_count})."
        )

    # Verify contiguous zero-based indices
    observed_indices = [r.ticket_index for r in initial_results]
    expected_indices = list(range(eligible_count))
    if observed_indices != expected_indices:
        raise ValueError(
            f"Ticket index mismatch in evidence: expected contiguous range "
            f"0..{eligible_count - 1}, got {observed_indices}."
        )

    dataset_fingerprint = compute_dataset_fingerprint(records)
    config = RunConfig(retry_budget=max_retries)

    strategy_results: dict[EvaluationStrategy, StrategyEvaluationResult] = {}

    for strat in EvaluationStrategy:
        strat_ticket_results: list[StrategyTicketResult] = []
        final_ticket_run_results: list[TicketRunResult] = []
        total_attempts = 0
        tickets_retried = 0
        tickets_recovered = 0
        tickets_exhausted = 0
        used_simulation = False

        for ticket_index, record in enumerate(records):
            init_res = initial_results[ticket_index]
            initial_outcome = init_res.outcome

            if strat == EvaluationStrategy.STRATEGY_1_NO_RETRY:
                # Strategy 1: strictly 1 attempt, no retries
                final_res = init_res
                attempts_used = 1
                total_attempts += 1
                is_simulated = False
                recovered = False
                transition = f"{initial_outcome.value} -> {final_res.outcome.value}"

            else:
                # Strategy 2 or Strategy 3
                eligible_for_strat_retry = False
                if strat == EvaluationStrategy.STRATEGY_2_SCHEMA_RETRY:
                    # Strategy 2: only O1 and O2 are eligible for retry
                    eligible_for_strat_retry = initial_outcome in (
                        RunOutcome.EXTRACTION_FAILURE,
                        RunOutcome.SCHEMA_FAILURE,
                    )
                elif strat == EvaluationStrategy.STRATEGY_3_SCHEMA_POLICY_RETRY:
                    # Strategy 3: O1, O2, and O3 are all eligible for retry
                    eligible_for_strat_retry = initial_outcome in FAILURE_OUTCOMES

                if not eligible_for_strat_retry:
                    # Initial attempt is valid (O4/O5) or ineligible for retry
                    final_res = init_res
                    attempts_used = 1
                    total_attempts += 1
                    is_simulated = False
                    recovered = False
                    transition = f"{initial_outcome.value} -> {final_res.outcome.value}"
                else:
                    # Retry is eligible
                    tickets_retried += 1
                    current_res = init_res
                    attempts_used = 1
                    is_simulated = False

                    for retry_num in range(1, max_retries + 1):
                        attempts_used += 1
                        if retry_simulator is not None:
                            used_simulation = True
                            is_simulated = True
                            simulated_raw = retry_simulator(
                                ticket_index,
                                record,
                                attempts_used,
                                current_res.outcome,
                                current_res.error_message,
                            )
                            current_res = classify_response(
                                simulated_raw,
                                record,
                                ticket_index,
                                attempts=attempts_used,
                            )
                        else:
                            # Without simulator, retry evidence exhausts
                            break

                        # Check if recovered to valid output
                        if current_res.outcome not in FAILURE_OUTCOMES:
                            break

                    total_attempts += attempts_used
                    final_res = current_res
                    recovered = (
                        initial_outcome in FAILURE_OUTCOMES
                        and final_res.outcome not in FAILURE_OUTCOMES
                    )
                    if recovered:
                        tickets_recovered += 1
                    elif final_res.outcome in FAILURE_OUTCOMES:
                        tickets_exhausted += 1

                    transition = (
                        f"{initial_outcome.value} -> {final_res.outcome.value}"
                    )

            final_ticket_run_results.append(final_res)
            strat_ticket_results.append(
                StrategyTicketResult(
                    ticket_index=ticket_index,
                    initial_outcome=initial_outcome,
                    final_outcome=final_res.outcome,
                    outcome_transition=transition,
                    attempts_used=attempts_used,
                    recovered=recovered,
                    credit=final_res.credit,
                    prediction=final_res.prediction,
                    is_simulated=is_simulated,
                )
            )

        # Compute standard EvaluationMetrics for this strategy's final outputs
        eval_run = _build_evaluation_run(
            records, dataset_fingerprint, final_ticket_run_results, config
        )
        metrics = compute_metrics(eval_run, records)

        recovery_ratio = make_ratio(tickets_recovered, tickets_retried)
        accounting = RetryAccounting(
            total_attempts=total_attempts,
            tickets_retried=tickets_retried,
            tickets_recovered=tickets_recovered,
            tickets_exhausted=tickets_exhausted,
            recovery_rate=recovery_ratio,
        )

        sim_note = (
            "Simulated offline recovery using mock/synthetic retry generator. "
            "Must NOT be presented as observed Gemini performance."
            if used_simulation
            else None
        )

        strategy_results[strat] = StrategyEvaluationResult(
            strategy=strat,
            eligible_count=eligible_count,
            metrics=metrics,
            retry_accounting=accounting,
            ticket_results=strat_ticket_results,
            is_synthetic_simulation=used_simulation,
            simulation_note=sim_note,
        )

    # Compute paired comparisons
    paired_comparisons: list[PairedComparison] = []
    comparisons_to_run = [
        (
            EvaluationStrategy.STRATEGY_1_NO_RETRY,
            EvaluationStrategy.STRATEGY_2_SCHEMA_RETRY,
        ),
        (
            EvaluationStrategy.STRATEGY_1_NO_RETRY,
            EvaluationStrategy.STRATEGY_3_SCHEMA_POLICY_RETRY,
        ),
        (
            EvaluationStrategy.STRATEGY_2_SCHEMA_RETRY,
            EvaluationStrategy.STRATEGY_3_SCHEMA_POLICY_RETRY,
        ),
    ]

    for base_strat, comp_strat in comparisons_to_run:
        base_res = strategy_results[base_strat]
        comp_res = strategy_results[comp_strat]

        base_acc = base_res.metrics.strict_correct.rate or 0.0
        comp_acc = comp_res.metrics.strict_correct.rate or 0.0
        delta_acc = round(comp_acc - base_acc, 6)

        base_sv = (
            base_res.metrics.schema_valid.rate
            if base_res.metrics.schema_valid
            else 0.0
        ) or 0.0
        comp_sv = (
            comp_res.metrics.schema_valid.rate
            if comp_res.metrics.schema_valid
            else 0.0
        ) or 0.0
        delta_sv = round(comp_sv - base_sv, 6)

        recovered_count = 0
        regressed_count = 0
        unchanged_count = 0

        for idx in range(eligible_count):
            b_credit = base_res.ticket_results[idx].credit
            c_credit = comp_res.ticket_results[idx].credit
            if b_credit == 0 and c_credit == 1:
                recovered_count += 1
            elif b_credit == 1 and c_credit == 0:
                regressed_count += 1
            else:
                unchanged_count += 1

        paired_comparisons.append(
            PairedComparison(
                baseline_strategy=base_strat,
                comparison_strategy=comp_strat,
                strict_accuracy_delta=delta_acc,
                schema_valid_delta=delta_sv,
                recovered_tickets=recovered_count,
                regressed_tickets=regressed_count,
                unchanged_tickets=unchanged_count,
            )
        )

    limitations = (
        "Paired comparisons evaluate identical development split tickets under "
        "controlled strategy conditions. Persisted first responses represent "
        "historical Gemini observations, while multi-turn retries are evaluated "
        "either under un-replayed exhaustion or explicitly labeled synthetic "
        "simulations. Confidence intervals and accuracy deltas reflect sampling "
        "variance across the 20 development tickets and do not guarantee "
        "generalization to unseen held-out data."
    )

    return PairedEvaluationReport(
        schema_version="1.0",
        created_at=datetime.now(UTC).isoformat(),
        dataset_fingerprint=dataset_fingerprint,
        eligible_count=eligible_count,
        strategies=strategy_results,
        paired_comparisons=paired_comparisons,
        evidence_source=evidence_source,
        limitations_note=limitations,
    )
