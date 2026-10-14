"""Tests for Stage 5 retry handling, policy-aware validation, and failure logging.

Verifies:
- Configurable retry budget of 2 (up to 3 attempts: 1 initial + 2 retries).
- Recovery from schema failures (O2 -> O5).
- Exhaustion when retries do not recover (O2 -> O2).
- Recovery from policy failures (O3 -> O5 for both Validator A and Validator B).
- Recovery from extraction failures (O1 -> O5).
- Ineligible outcomes (O4 and O5) make exactly one attempt without retry.
- Operational failures (OperationalError) immediately abort without retry.
- Redaction in error feedback and structured failure logs (no API keys or raw data).
- Structured JSONL failure log records transitions, counts, and recovery flags.
"""

import json
from pathlib import Path

from structured_output_agent.exceptions import OperationalError
from structured_output_agent.failure_logger import FailureLogEntry
from structured_output_agent.feedback import (
    generate_actionable_feedback,
    redact_secrets,
)
from structured_output_agent.run_config import RunConfig
from structured_output_agent.run_results import RunOutcome
from structured_output_agent.runner import RunStatus, run_evaluation
from structured_output_agent.schemas import (
    AssignedTeam,
    DatasetRecord,
    DatasetSplit,
    DatasetTag,
    ExpectedDecision,
    Priority,
    TicketCategory,
)


def _make_valid_dataset_file(path: Path) -> None:
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
            records.append(json.dumps(rec.model_dump(mode="json")))

    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def _valid_billing_json() -> str:
    return json.dumps(
        {
            "category": "billing",
            "priority": "normal",
            "assigned_team": "payments",
            "summary": "Customer billing inquiry",
            "requires_human_review": False,
        }
    )


class MockSequenceClient:
    """Mock client that responds with a sequence of canned strings or exceptions."""

    def __init__(
        self,
        responses: list[str | Exception],
        default_response: str | None = None,
    ) -> None:
        self.responses = responses
        self.default_response = default_response or _valid_billing_json()
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        index = len(self.calls) - 1
        item = (
            self.responses[index]
            if index < len(self.responses)
            else self.default_response
        )
        if isinstance(item, Exception):
            raise item
        return item


def test_recovery_o2_to_o5_with_feedback(tmp_path: Path) -> None:
    """Verify recovery from schema failure: attempt 1 fails O2, attempt 2 succeeds."""
    dataset_file = tmp_path / "dev.jsonl"
    _make_valid_dataset_file(dataset_file)
    log_file = tmp_path / "failures.jsonl"

    # Ticket 0: Attempt 1 is invalid schema; Attempt 2 is valid billing JSON
    responses: list[str | Exception] = [
        json.dumps(
            {
                "category": "INVALID_CATEGORY_NAME",
                "priority": "normal",
                "assigned_team": "payments",
                "summary": "Customer invoice inquiry",
                "requires_human_review": False,
            }
        ),
        _valid_billing_json(),
    ]

    client = MockSequenceClient(responses)
    config = RunConfig(
        dataset_path=dataset_file,
        retry_budget=2,
        failure_log_path=log_file,
    )

    run = run_evaluation(config, client)

    assert run.status == RunStatus.COMPLETED
    assert len(run.completed_results) == 12
    result = run.completed_results[0]

    # Verified: recovered to valid_correct with credit 1 in 2 attempts
    assert result.outcome == RunOutcome.VALID_CORRECT
    assert result.credit == 1
    assert result.attempts == 2
    assert len(result.raw_responses) == 2

    # Verified: retry prompt received actionable feedback mentioning schema failure
    retry_prompt = client.calls[1]
    assert "[Previous Attempt Error Feedback]" in retry_prompt
    assert "Schema validation failure" in retry_prompt

    # Verified: JSONL failure log recorded the O2 -> O5 transition
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    log_data = json.loads(lines[0])
    assert log_data["ticket_index"] == 0
    assert log_data["initial_outcome"] == "O2"
    assert log_data["final_outcome"] == "O5"
    assert log_data["outcome_transition"] == "O2 -> O5"
    assert log_data["recovered"] is True
    assert log_data["attempt_count"] == 2


def test_exhaustion_o2_to_o2_after_retry_budget_of_2(tmp_path: Path) -> None:
    """Verify exhaustion: all 3 attempts (1 initial + 2 retries) fail O2."""
    dataset_file = tmp_path / "dev.jsonl"
    _make_valid_dataset_file(dataset_file)
    log_file = tmp_path / "failures.jsonl"

    # Ticket 0: 3 attempts all fail schema validation
    invalid_schema = json.dumps(
        {
            "category": "not_real_category",
            "priority": "normal",
            "assigned_team": "payments",
            "summary": "Bad summary",
            "requires_human_review": False,
        }
    )
    client = MockSequenceClient([invalid_schema, invalid_schema, invalid_schema])
    config = RunConfig(
        dataset_path=dataset_file,
        retry_budget=2,
        failure_log_path=log_file,
    )

    run = run_evaluation(config, client)

    assert run.status == RunStatus.COMPLETED
    result = run.completed_results[0]

    # Verified: total 3 attempts made, outcome remains schema failure with credit 0
    assert result.outcome == RunOutcome.SCHEMA_FAILURE
    assert result.credit == 0
    assert result.attempts == 3
    assert len(result.raw_responses) == 3

    # Verified: JSONL failure log recorded the O2 -> O2 exhaustion for ticket 0
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    log_data = json.loads(lines[0])
    assert log_data["ticket_index"] == 0
    assert log_data["initial_outcome"] == "O2"
    assert log_data["final_outcome"] == "O2"
    assert log_data["outcome_transition"] == "O2 -> O2"
    assert log_data["recovered"] is False
    assert log_data["attempt_count"] == 3


def test_policy_recovery_validator_a_category_to_team(tmp_path: Path) -> None:
    """Verify policy recovery: Validator A category-to-team mismatch recovers to O5."""
    dataset_file = tmp_path / "dev.jsonl"
    _make_valid_dataset_file(dataset_file)

    # Ticket 0: Attempt 1 violates Validator A, Attempt 2 corrects assigned_team
    attempt_1 = json.dumps(
        {
            "category": "billing",
            "priority": "normal",
            "assigned_team": "technical_support",  # Violates Validator A
            "summary": "Billing question",
            "requires_human_review": False,
        }
    )
    attempt_2 = _valid_billing_json()

    client = MockSequenceClient([attempt_1, attempt_2])
    config = RunConfig(dataset_path=dataset_file, retry_budget=2)

    run = run_evaluation(config, client)

    assert run.status == RunStatus.COMPLETED
    result = run.completed_results[0]
    assert result.outcome == RunOutcome.VALID_CORRECT
    assert result.credit == 1
    assert result.attempts == 2

    # Verified: actionable policy feedback given to the model on retry
    retry_prompt = client.calls[1]
    assert "Deterministic policy violation" in retry_prompt
    assert "Category-to-team alignment" in retry_prompt
    assert "payments" in retry_prompt


def test_policy_recovery_validator_b_needs_review_required(tmp_path: Path) -> None:
    """Verify policy recovery: Validator B needs_review requirement recovers."""
    dataset_file = tmp_path / "dev.jsonl"
    _make_valid_dataset_file(dataset_file)

    # For ticket 10 (needs_review, normal priority in _make_valid_dataset_file):
    # Attempt 1: needs_review with requires_human_review=false (violates Validator B)
    # Attempt 2: needs_review with requires_human_review=true (valid)
    attempt_1 = json.dumps(
        {
            "category": "needs_review",
            "priority": "normal",
            "assigned_team": "human_review",
            "summary": "Ambiguous support issue",
            "requires_human_review": False,  # Violates Validator B
        }
    )
    attempt_2 = json.dumps(
        {
            "category": "needs_review",
            "priority": "normal",
            "assigned_team": "human_review",
            "summary": "Ambiguous support issue",
            "requires_human_review": True,  # Corrected
        }
    )

    # Valid billing responses for tickets 0-9, then attempt 1 & 2 for ticket 10
    responses: list[str | Exception] = [_valid_billing_json()] * 10
    responses.extend([attempt_1, attempt_2])

    client = MockSequenceClient(responses)
    config = RunConfig(dataset_path=dataset_file, retry_budget=2)

    run = run_evaluation(config, client)

    assert run.status == RunStatus.COMPLETED
    res_10 = run.completed_results[10]
    assert res_10.outcome == RunOutcome.VALID_CORRECT
    assert res_10.attempts == 2

    # Check retry prompt for ticket 10
    retry_prompt = client.calls[11]
    assert "Deterministic policy violation" in retry_prompt
    assert (
        "needs_review requires human review" in retry_prompt
        or "Human review rule" in retry_prompt
    )


def test_extraction_recovery_o1_to_o5(tmp_path: Path) -> None:
    """Verify extraction recovery: unparseable text recovered to valid JSON."""
    dataset_file = tmp_path / "dev.jsonl"
    _make_valid_dataset_file(dataset_file)

    client = MockSequenceClient(
        ["Here is my triage explanation: sorry, no json", _valid_billing_json()]
    )
    config = RunConfig(dataset_path=dataset_file, retry_budget=2)

    run = run_evaluation(config, client)

    assert run.status == RunStatus.COMPLETED
    result = run.completed_results[0]
    assert result.outcome == RunOutcome.VALID_CORRECT
    assert result.attempts == 2

    retry_prompt = client.calls[1]
    assert "Extraction failure" in retry_prompt
    assert "could not be parsed as a single valid JSON object" in retry_prompt


def test_ineligible_outcomes_do_not_retry(tmp_path: Path) -> None:
    """Verify ineligible outcomes (O4 and O5) make 1 call and never retry."""
    dataset_file = tmp_path / "dev.jsonl"
    _make_valid_dataset_file(dataset_file)

    # 1. Valid correct (O5): ticket 0 makes exactly 1 call
    client_o5 = MockSequenceClient([_valid_billing_json()])
    config = RunConfig(dataset_path=dataset_file, retry_budget=2)
    run_o5 = run_evaluation(config, client_o5)
    assert run_o5.completed_results[0].outcome == RunOutcome.VALID_CORRECT
    assert run_o5.completed_results[0].attempts == 1
    # Total calls across all 12 tickets should be exactly 12 (1 per ticket)
    assert len(client_o5.calls) == 12

    # 2. Valid incorrect (O4): ticket 0 category other, valid schema, no retry
    valid_incorrect = json.dumps(
        {
            "category": "other",
            "priority": "low",
            "assigned_team": "general_support",
            "summary": "General inquiry",
            "requires_human_review": False,
        }
    )
    client_o4 = MockSequenceClient([valid_incorrect])
    run_o4 = run_evaluation(config, client_o4)
    assert run_o4.completed_results[0].outcome == RunOutcome.VALID_INCORRECT
    assert run_o4.completed_results[0].credit == 0
    assert run_o4.completed_results[0].attempts == 1
    # Exactly 12 calls (1 per ticket)
    assert len(client_o4.calls) == 12


def test_operational_failure_immediately_aborts_without_retry(tmp_path: Path) -> None:
    """Verify OperationalError is never treated as a model-output validation error."""
    dataset_file = tmp_path / "dev.jsonl"
    _make_valid_dataset_file(dataset_file)

    # Attempt 1 raises OperationalError
    client = MockSequenceClient([OperationalError("Rate limit exceeded 429")])
    config = RunConfig(dataset_path=dataset_file, retry_budget=2)

    run = run_evaluation(config, client)

    assert run.status == RunStatus.ABORTED
    assert run.error_type == "OperationalError"
    assert "Rate limit exceeded 429" in (run.error_message or "")
    # Verified: exactly 1 call was attempted; operational failure did not retry
    assert len(client.calls) == 1
    assert len(run.completed_results) == 0


def test_operational_failure_during_retry_aborts_run(tmp_path: Path) -> None:
    """Verify OperationalError raised on a retry turn immediately aborts the run."""
    dataset_file = tmp_path / "dev.jsonl"
    _make_valid_dataset_file(dataset_file)

    # Attempt 1: Schema failure (eligible for retry)
    # Attempt 2: OperationalError (network dropped)
    responses: list[str | Exception] = [
        "not json",
        OperationalError("Socket closed remotely"),
    ]
    client = MockSequenceClient(responses)
    config = RunConfig(dataset_path=dataset_file, retry_budget=2)

    run = run_evaluation(config, client)

    assert run.status == RunStatus.ABORTED
    assert run.error_type == "OperationalError"
    assert "Socket closed remotely" in (run.error_message or "")
    assert len(client.calls) == 2


def test_redaction_in_feedback_and_logging() -> None:
    """Verify secrets and API keys are redacted from error feedback and JSONL logs."""
    fake_secret = "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
    fake_bearer = "Bearer secret_jwt_token_1234567890abcdef"

    error_with_secrets = (
        f"Validation error: key={fake_secret} failed with auth {fake_bearer}. "
        f"input_value='SENSITIVE_CUSTOMER_CREDIT_CARD_DATA'"
    )

    feedback = generate_actionable_feedback(
        RunOutcome.SCHEMA_FAILURE, error_with_secrets
    )

    assert fake_secret not in feedback
    assert fake_bearer not in feedback
    assert "SENSITIVE_CUSTOMER_CREDIT_CARD_DATA" not in feedback
    assert "[REDACTED]" in feedback or "[OMITTED]" in feedback

    entry = FailureLogEntry(
        timestamp="2026-10-13T00:00:00Z",
        ticket_index=0,
        attempt_count=1,
        initial_outcome="O2",
        final_outcome="O2",
        outcome_transition="O2 -> O2",
        recovered=False,
        error_type="ValidationError",
        safe_error_message=redact_secrets(error_with_secrets),
        attempts=[
            {
                "attempt_index": 1,
                "outcome": "O2",
                "error_type": "ValidationError",
                "safe_error_message": redact_secrets(error_with_secrets),
            }
        ],  # type: ignore[arg-type]
    )

    dumped = entry.model_dump_json()
    assert fake_secret not in dumped
    assert fake_bearer not in dumped
