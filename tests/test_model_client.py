import pytest

from structured_output_agent.exceptions import AgentError, OperationalError
from structured_output_agent.model_client import ModelClient


class _ValidStubClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.recorded_prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.recorded_prompts.append(prompt)
        return self.response


class _FailingStubClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete(self, prompt: str) -> str:
        try:
            raise self.error
        except Exception as exc:
            raise OperationalError(
                f"Model provider request failed: {exc}"
            ) from exc


class _MissingMethodClient:
    def generate(self, prompt: str) -> str:
        return prompt


class _NonCallableAttributeClient:
    complete = "not a callable"


def test_structural_protocol_conformance() -> None:
    valid_client = _ValidStubClient('{"category": "billing"}')
    assert isinstance(valid_client, ModelClient)


def test_missing_method_fails_protocol_check() -> None:
    client = _MissingMethodClient()
    assert not isinstance(client, ModelClient)


def test_non_callable_attribute_raises_on_call() -> None:
    client = _NonCallableAttributeClient()
    with pytest.raises(TypeError):
        client.complete("Prompt text")  # type: ignore[operator]


def test_raw_response_string_returned_unmodified() -> None:
    raw_payload = '{"category": "billing", "summary": "needs help"}'
    client = _ValidStubClient(raw_payload)

    result = client.complete("Triage this ticket: ...")

    assert result == raw_payload
    assert client.recorded_prompts == ["Triage this ticket: ..."]


def test_non_json_raw_response_returned_as_is() -> None:
    prose_response = "Here is my analysis: unable to decide."
    client = _ValidStubClient(prose_response)

    result = client.complete("Prompt text")

    assert result == prose_response


def test_operational_error_inherits_from_agent_error() -> None:
    error = OperationalError("Connection dropped")
    assert isinstance(error, AgentError)
    assert isinstance(error, Exception)


def test_operational_failure_chains_underlying_cause() -> None:
    underlying = ConnectionResetError("Connection reset by peer")
    client = _FailingStubClient(underlying)

    with pytest.raises(OperationalError) as exc_info:
        client.complete("Prompt text")

    assert "Connection reset by peer" in str(exc_info.value)
    assert exc_info.value.__cause__ is underlying
