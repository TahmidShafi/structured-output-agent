"""Offline unit tests for EvaluationCheckpoint data model (Stage 3.9.1)."""

import json

import pytest
from pydantic import ValidationError

from structured_output_agent.checkpoint import (
    CHECKPOINT_VERSION,
    EvaluationCheckpoint,
)
from structured_output_agent.run_config import RunConfig
from structured_output_agent.run_results import RunOutcome, TicketRunResult
from structured_output_agent.schemas import (
    AssignedTeam,
    DatasetSplit,
    Priority,
    TicketCategory,
    TicketTriageOutput,
)


def _make_prediction() -> TicketTriageOutput:
    return TicketTriageOutput(
        category=TicketCategory.BILLING,
        priority=Priority.NORMAL,
        assigned_team=AssignedTeam.PAYMENTS,
        summary="Customer billing inquiry",
        requires_human_review=False,
    )


def _make_result(
    index: int,
    outcome: RunOutcome = RunOutcome.VALID_CORRECT,
) -> TicketRunResult:
    if outcome == RunOutcome.VALID_CORRECT:
        return TicketRunResult(
            ticket_index=index,
            outcome=outcome,
            credit=1,
            prediction=_make_prediction(),
        )
    if outcome == RunOutcome.VALID_INCORRECT:
        return TicketRunResult(
            ticket_index=index,
            outcome=outcome,
            credit=0,
            prediction=_make_prediction(),
        )
    return TicketRunResult(
        ticket_index=index,
        outcome=outcome,
        credit=0,
        error_type="JSONExtractionError",
        error_message="Extraction failed.",
        prediction=None,
    )


def _make_checkpoint(
    *,
    eligible_count: int = 3,
    completed_results: list[TicketRunResult] | None = None,
    dataset_fingerprint: str = "a" * 64,
    prompt_fingerprint: str = "b" * 64,
    client_type: str = "stub",
    model_id: str | None = None,
    checkpoint_version: int = CHECKPOINT_VERSION,
    run_config: RunConfig | None = None,
) -> EvaluationCheckpoint:
    return EvaluationCheckpoint(
        checkpoint_version=checkpoint_version,
        dataset_fingerprint=dataset_fingerprint,
        eligible_count=eligible_count,
        client_type=client_type,
        model_id=model_id,
        run_config=run_config if run_config is not None else RunConfig(),
        prompt_fingerprint=prompt_fingerprint,
        completed_results=completed_results if completed_results is not None else [],
    )


def test_valid_checkpoint_zero_completed_results() -> None:
    checkpoint = _make_checkpoint(
        eligible_count=5,
        completed_results=[],
        model_id=None,
    )
    assert checkpoint.checkpoint_version == 1
    assert checkpoint.eligible_count == 5
    assert len(checkpoint.completed_results) == 0
    assert checkpoint.client_type == "stub"
    assert checkpoint.model_id is None
    assert checkpoint.dataset_fingerprint == "a" * 64
    assert checkpoint.prompt_fingerprint == "b" * 64

    # Also test valid with explicit model_id
    checkpoint_with_model = _make_checkpoint(
        eligible_count=5,
        completed_results=[],
        model_id="gemini-3.1-flash-lite",
    )
    assert checkpoint_with_model.model_id == "gemini-3.1-flash-lite"


def test_valid_checkpoint_partially_completed() -> None:
    results = [
        _make_result(0, RunOutcome.VALID_CORRECT),
        _make_result(1, RunOutcome.VALID_INCORRECT),
    ]
    checkpoint = _make_checkpoint(
        eligible_count=5,
        completed_results=results,
    )
    assert checkpoint.eligible_count == 5
    assert len(checkpoint.completed_results) == 2
    assert [r.ticket_index for r in checkpoint.completed_results] == [0, 1]


def test_valid_checkpoint_all_eligible_completed() -> None:
    results = [
        _make_result(0, RunOutcome.VALID_CORRECT),
        _make_result(1, RunOutcome.EXTRACTION_FAILURE),
        _make_result(2, RunOutcome.VALID_CORRECT),
    ]
    checkpoint = _make_checkpoint(
        eligible_count=3,
        completed_results=results,
    )
    assert checkpoint.eligible_count == 3
    assert len(checkpoint.completed_results) == 3
    assert [r.ticket_index for r in checkpoint.completed_results] == [0, 1, 2]


def test_negative_eligible_count_rejected() -> None:
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        _make_checkpoint(eligible_count=-1)


def test_more_completed_results_than_eligible_tickets_rejected() -> None:
    results = [
        _make_result(0),
        _make_result(1),
        _make_result(2),
    ]
    with pytest.raises(
        ValidationError,
        match=(
            "Number of completed results \\(3\\) cannot exceed eligible_count \\(2\\)"
        ),
    ):
        _make_checkpoint(eligible_count=2, completed_results=results)


def test_duplicate_ticket_indices_rejected() -> None:
    # Consecutive duplicates [0, 0]
    results_dup = [
        _make_result(0),
        _make_result(0),
    ]
    with pytest.raises(
        ValidationError,
        match="Completed ticket indices must form a contiguous prefix",
    ):
        _make_checkpoint(eligible_count=5, completed_results=results_dup)

    # Duplicate later in sequence [0, 1, 1]
    results_dup_later = [
        _make_result(0),
        _make_result(1),
        _make_result(1),
    ]
    with pytest.raises(
        ValidationError,
        match="Completed ticket indices must form a contiguous prefix",
    ):
        _make_checkpoint(eligible_count=5, completed_results=results_dup_later)


def test_non_contiguous_ticket_indices_rejected() -> None:
    # Gapped sequence [0, 2]
    results_gap = [
        _make_result(0),
        _make_result(2),
    ]
    with pytest.raises(
        ValidationError,
        match="Completed ticket indices must form a contiguous prefix",
    ):
        _make_checkpoint(eligible_count=5, completed_results=results_gap)

    # Missing zero index [1]
    results_start_one = [
        _make_result(1),
    ]
    with pytest.raises(
        ValidationError,
        match="Completed ticket indices must form a contiguous prefix",
    ):
        _make_checkpoint(eligible_count=5, completed_results=results_start_one)


def test_out_of_order_ticket_indices_rejected() -> None:
    # Inverted order [1, 0]
    results_inverted = [
        _make_result(1),
        _make_result(0),
    ]
    with pytest.raises(
        ValidationError,
        match="Completed ticket indices must form a contiguous prefix",
    ):
        _make_checkpoint(eligible_count=5, completed_results=results_inverted)

    # Shuffled order [0, 2, 1]
    results_shuffled = [
        _make_result(0),
        _make_result(2),
        _make_result(1),
    ]
    with pytest.raises(
        ValidationError,
        match="Completed ticket indices must form a contiguous prefix",
    ):
        _make_checkpoint(eligible_count=5, completed_results=results_shuffled)


@pytest.mark.parametrize(
    "invalid_fp",
    [
        "",  # empty
        "   ",  # whitespace
        "a" * 63,  # too short
        "a" * 65,  # too long
        "g" * 64,  # non-hex
        "z" * 64,  # non-hex
        "!@#$" * 16,  # non-hex symbols
        "abc123",  # short arbitrary
    ],
)
def test_invalid_dataset_fingerprint_rejected(invalid_fp: str) -> None:
    with pytest.raises(ValidationError):
        _make_checkpoint(dataset_fingerprint=invalid_fp)


def test_valid_dataset_fingerprint_normalized() -> None:
    upper_fp = "A" * 64
    checkpoint = _make_checkpoint(dataset_fingerprint=upper_fp)
    assert checkpoint.dataset_fingerprint == "a" * 64


@pytest.mark.parametrize(
    "invalid_prompt_fp",
    [
        "",  # empty
        "   ",  # whitespace only
        "b" * 63,  # too short
        "b" * 65,  # too long
        "g" * 64,  # non-hex
        "z" * 64,  # non-hex
        "!@#$" * 16,  # non-hex symbols
        "prompt-fp",  # non-hex string
    ],
)
def test_invalid_prompt_fingerprint_rejected(invalid_prompt_fp: str) -> None:
    with pytest.raises(ValidationError):
        _make_checkpoint(prompt_fingerprint=invalid_prompt_fp)


def test_valid_prompt_fingerprint_normalized() -> None:
    upper_fp = "B" * 64
    checkpoint = _make_checkpoint(prompt_fingerprint=upper_fp)
    assert checkpoint.prompt_fingerprint == "b" * 64


@pytest.mark.parametrize(
    "invalid_client",
    [
        "",  # empty
        "   ",  # whitespace only
    ],
)
def test_invalid_client_type_rejected(invalid_client: str) -> None:
    with pytest.raises(ValidationError):
        _make_checkpoint(client_type=invalid_client)


def test_invalid_model_id_rejected() -> None:
    with pytest.raises(
        ValidationError, match="model_id must not be empty or whitespace"
    ):
        _make_checkpoint(model_id="   ")


@pytest.mark.parametrize(
    "invalid_version",
    [
        0,
        2,
        99,
        -1,
        "1",  # string type instead of int
    ],
)
def test_unsupported_checkpoint_version_rejected(invalid_version: object) -> None:
    with pytest.raises(ValidationError):
        EvaluationCheckpoint(
            checkpoint_version=invalid_version,  # type: ignore[arg-type]
            dataset_fingerprint="a" * 64,
            eligible_count=1,
            client_type="stub",
            run_config=RunConfig(),
            prompt_fingerprint="b" * 64,
        )


def test_missing_required_fields_rejected() -> None:
    base_kwargs = {
        "checkpoint_version": 1,
        "dataset_fingerprint": "a" * 64,
        "eligible_count": 1,
        "client_type": "stub",
        "run_config": RunConfig(),
        "prompt_fingerprint": "b" * 64,
    }
    for field in (
        "dataset_fingerprint",
        "eligible_count",
        "client_type",
        "run_config",
        "prompt_fingerprint",
    ):
        kwargs = dict(base_kwargs)
        del kwargs[field]
        with pytest.raises(ValidationError):
            EvaluationCheckpoint(**kwargs)


def test_unknown_fields_rejected() -> None:
    # Direct model instantiation rejects unknown extra fields
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvaluationCheckpoint(
            dataset_fingerprint="a" * 64,
            eligible_count=1,
            client_type="stub",
            run_config=RunConfig(),
            prompt_fingerprint="b" * 64,
            unknown_extra_field="disallowed",  # type: ignore[call-arg]
        )

    # JSON deserialization rejects unknown extra fields
    raw_dict = {
        "checkpoint_version": 1,
        "dataset_fingerprint": "a" * 64,
        "eligible_count": 1,
        "client_type": "stub",
        "model_id": None,
        "run_config": RunConfig().model_dump(mode="json"),
        "prompt_fingerprint": "b" * 64,
        "completed_results": [],
        "unexpected_key": "injected_value",
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvaluationCheckpoint.from_json(json.dumps(raw_dict))


def test_serialization_and_reconstruction_round_trip() -> None:
    results = [
        _make_result(0, RunOutcome.VALID_CORRECT),
        _make_result(1, RunOutcome.EXTRACTION_FAILURE),
    ]
    checkpoint = _make_checkpoint(
        eligible_count=5,
        completed_results=results,
        model_id="gemini-3.1-flash-lite",
        prompt_fingerprint="c" * 64,
    )

    serialized = checkpoint.to_json(indent=2)
    assert isinstance(serialized, str)

    # Verify JSON deterministic structure
    parsed_json = json.loads(serialized)
    assert parsed_json["checkpoint_version"] == 1
    assert parsed_json["eligible_count"] == 5
    assert parsed_json["client_type"] == "stub"
    assert parsed_json["model_id"] == "gemini-3.1-flash-lite"
    assert parsed_json["prompt_fingerprint"] == "c" * 64
    assert len(parsed_json["completed_results"]) == 2

    # Reconstruct from JSON
    reconstructed = EvaluationCheckpoint.from_json(serialized)
    assert reconstructed == checkpoint
    assert reconstructed.checkpoint_version == 1
    assert reconstructed.dataset_fingerprint == "a" * 64
    assert reconstructed.prompt_fingerprint == "c" * 64
    assert reconstructed.eligible_count == 5
    assert reconstructed.client_type == "stub"
    assert reconstructed.model_id == "gemini-3.1-flash-lite"
    assert reconstructed.run_config.split == DatasetSplit.DEV
    assert len(reconstructed.completed_results) == 2
    assert reconstructed.completed_results[0].outcome == RunOutcome.VALID_CORRECT
    assert reconstructed.completed_results[0].credit == 1
    assert reconstructed.completed_results[1].outcome == RunOutcome.EXTRACTION_FAILURE
    assert reconstructed.completed_results[1].credit == 0


def test_from_json_validates_invariants() -> None:
    valid_base = {
        "checkpoint_version": 1,
        "dataset_fingerprint": "a" * 64,
        "eligible_count": 2,
        "client_type": "stub",
        "model_id": None,
        "run_config": RunConfig().model_dump(mode="json"),
        "prompt_fingerprint": "b" * 64,
        "completed_results": [],
    }

    # 1. Gapped ticket indices
    corrupted_indices = dict(valid_base)
    corrupted_indices["completed_results"] = [
        _make_result(0).model_dump(mode="json"),
        _make_result(2).model_dump(mode="json"),
    ]
    with pytest.raises(ValidationError, match="contiguous prefix"):
        EvaluationCheckpoint.from_json(json.dumps(corrupted_indices))

    # 2. More completed results than eligible count
    corrupted_count = dict(valid_base)
    corrupted_count["eligible_count"] = 1
    corrupted_count["completed_results"] = [
        _make_result(0).model_dump(mode="json"),
        _make_result(1).model_dump(mode="json"),
    ]
    with pytest.raises(ValidationError, match="cannot exceed eligible_count"):
        EvaluationCheckpoint.from_json(json.dumps(corrupted_count))

    # 3. Invalid dataset fingerprint
    corrupted_fp = dict(valid_base)
    corrupted_fp["dataset_fingerprint"] = "invalid-fp"
    with pytest.raises(
        ValidationError, match="dataset_fingerprint must be a 64-character"
    ):
        EvaluationCheckpoint.from_json(json.dumps(corrupted_fp))

    # 4. Invalid prompt fingerprint
    corrupted_prompt_fp = dict(valid_base)
    corrupted_prompt_fp["prompt_fingerprint"] = "invalid-prompt-fp"
    with pytest.raises(
        ValidationError, match="prompt_fingerprint must be a 64-character"
    ):
        EvaluationCheckpoint.from_json(json.dumps(corrupted_prompt_fp))

    # 5. Unsupported checkpoint version
    corrupted_version = dict(valid_base)
    corrupted_version["checkpoint_version"] = 2
    with pytest.raises(ValidationError):
        EvaluationCheckpoint.from_json(json.dumps(corrupted_version))
