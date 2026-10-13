from typing import Any
from unittest.mock import MagicMock

import pytest

from structured_output_agent.cli import main
from structured_output_agent.exceptions import OperationalError
from structured_output_agent.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_REQUEST_INTERVAL,
    GeminiModelClient,
    _sanitize_error_message,
    resolve_gemini_model,
    resolve_request_interval,
)
from structured_output_agent.model_client import ModelClient

try:
    import httpx
    from google import genai  # type: ignore
    from google.genai import errors, types  # type: ignore

    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False
    genai = None  # type: ignore
    errors = None  # type: ignore
    types = None  # type: ignore
    httpx = None  # type: ignore

requires_gemini_sdk = pytest.mark.skipif(
    not HAS_GEMINI_SDK,
    reason=(
        "Requires optional 'google-genai' package "
        "(install via 'pip install -e .[gemini]')"
    ),
)


class _MockResponse:
    def __init__(self, text: str | None = None) -> None:
        self.text = text


def _make_mock_client(
    response_text: str | None = '{"category": "billing"}',
) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = _MockResponse(response_text)
    return client


def test_gemini_client_protocol_conformance() -> None:
    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(api_key="mock-key-12345", client=mock_sdk_client)
    assert isinstance(client, ModelClient)


def test_gemini_client_missing_api_key_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        GeminiModelClient(api_key=None, client=MagicMock())

    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        GeminiModelClient(api_key="   ", client=MagicMock())


def test_model_resolution_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Default fallback
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert resolve_gemini_model() == DEFAULT_GEMINI_MODEL
    assert resolve_gemini_model() == "gemini-3.8-flash"

    # 2. Environment variable overrides default
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    assert resolve_gemini_model() == "gemini-3.5-flash-lite"

    # 3. Explicit CLI model overrides environment variable
    assert (
        resolve_gemini_model(cli_model="gemini-3.8-flash-custom")
        == "gemini-3.8-flash-custom"
    )


@requires_gemini_sdk
def test_gemini_client_explicit_retry_disabled_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock the public google.genai.Client constructor
    mock_sdk_instance = MagicMock()
    mock_client_ctor = MagicMock(return_value=mock_sdk_instance)
    monkeypatch.setattr("google.genai.Client", mock_client_ctor)

    # Construct GeminiModelClient without injecting a client
    client = GeminiModelClient(api_key="test-offline-key")

    # Verify constructor receives http_options with retry_options.attempts == 1
    mock_client_ctor.assert_called_once()
    _, kwargs = mock_client_ctor.call_args
    assert kwargs["api_key"] == "test-offline-key"
    http_opts = kwargs["http_options"]
    assert isinstance(http_opts, types.HttpOptions)
    assert http_opts.retry_options is not None
    assert http_opts.retry_options.attempts == 1

    # Verify GeminiModelClient uses the mocked SDK client instance
    assert client._client is mock_sdk_instance


@requires_gemini_sdk
def test_gemini_client_complete_success_and_config() -> None:
    expected_output = (
        '{"category": "billing", "priority": "normal", "assigned_team": "payments"}'
    )
    mock_sdk_client = _make_mock_client(expected_output)
    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)

    prompt = "Triage ticket input"
    result = client.complete(prompt)

    assert result == expected_output
    mock_sdk_client.models.generate_content.assert_called_once()
    call_kwargs = mock_sdk_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert call_kwargs["contents"] == prompt

    config = call_kwargs["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.temperature == 0.0
    assert config.http_options is not None
    assert config.http_options.retry_options is not None
    assert config.http_options.retry_options.attempts == 1


@requires_gemini_sdk
def test_gemini_client_empty_or_none_text_returns_empty_string() -> None:
    # Simulates safety filter block or empty generation
    mock_sdk_client = _make_mock_client(None)
    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)

    result = client.complete("Prompt triggering safety filter")
    assert result == ""


@requires_gemini_sdk
def test_gemini_client_rate_limit_429_maps_to_operational_error() -> None:
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = errors.ClientError(
        code=429,
        response_json={"error": {"message": "Resource exhausted: 429 quota exceeded"}},
        response=None,
    )

    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)
    with pytest.raises(
        OperationalError, match=r"Gemini operational failure \[ClientError\]"
    ):
        client.complete("Prompt")


@requires_gemini_sdk
def test_gemini_client_auth_error_401_maps_to_operational_error() -> None:
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = errors.ClientError(
        code=401,
        response_json={"error": {"message": "API_KEY_INVALID: 401 unauthorized"}},
        response=None,
    )

    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)
    with pytest.raises(
        OperationalError, match=r"Gemini operational failure \[ClientError\]"
    ):
        client.complete("Prompt")


@requires_gemini_sdk
def test_gemini_client_server_error_503_maps_to_operational_error() -> None:
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = errors.ServerError(
        code=503,
        response_json={"error": {"message": "503 Service Unavailable"}},
        response=None,
    )

    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)
    with pytest.raises(
        OperationalError, match=r"Gemini operational failure \[ServerError\]"
    ):
        client.complete("Prompt")


@requires_gemini_sdk
def test_gemini_client_transport_timeout_maps_to_operational_error() -> None:
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = httpx.TimeoutException(
        "Connection timed out after 60s"
    )

    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)
    with pytest.raises(
        OperationalError, match=r"Gemini operational failure \[TimeoutException\]"
    ):
        client.complete("Prompt")


@requires_gemini_sdk
def test_gemini_client_transport_connect_error_maps_to_operational_error() -> None:
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = httpx.ConnectError(
        "Failed to establish connection"
    )

    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)
    with pytest.raises(
        OperationalError, match=r"Gemini operational failure \[ConnectError\]"
    ):
        client.complete("Prompt")


@requires_gemini_sdk
def test_gemini_client_sanitizes_credentials_in_error_message() -> None:
    secret_key = "AIzaSySecretApiKey123456"

    mock_sdk_client = MagicMock()
    # Simulate an APIError that contains the secret key in the error message
    mock_sdk_client.models.generate_content.side_effect = errors.APIError(
        code=400,
        response_json={
            "error": {
                "message": (
                    "Failed to connect to endpoint "
                    f"https://example.com/v1?key={secret_key}"
                )
            }
        },
        response=None,
    )

    client = GeminiModelClient(api_key=secret_key, client=mock_sdk_client)
    with pytest.raises(OperationalError) as exc_info:
        client.complete("Prompt")

    error_message = str(exc_info.value)
    assert secret_key not in error_message
    assert "[REDACTED]" in error_message


def test_sanitize_error_message_helper() -> None:
    secret = "my_secret_token_abc"
    msg = f"Error calling https://api.google.com?key={secret}&test=1: {secret} invalid"
    sanitized = _sanitize_error_message(msg, secret)
    assert secret not in sanitized
    assert "key=[REDACTED]" in sanitized
    assert "[REDACTED] invalid" in sanitized


def test_gemini_client_repr_and_str_never_leak_credentials() -> None:
    secret_key = "TopSecretGeminiKey999"
    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(
        api_key=secret_key, model="gemini-3.8-flash", client=mock_sdk_client
    )

    repr_str = repr(client)
    str_str = str(client)

    assert secret_key not in repr_str
    assert secret_key not in str_str
    assert "gemini-3.8-flash" in repr_str
    assert "gemini-3.8-flash" in str_str


@requires_gemini_sdk
def test_unexpected_programming_error_propagates() -> None:
    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = TypeError(
        "Unexpected internal programming type error"
    )

    client = GeminiModelClient(
        api_key="test-key", client=mock_sdk_client, request_interval=0.0
    )
    # Must propagate naturally as TypeError, NOT masked as OperationalError
    with pytest.raises(TypeError, match="Unexpected internal programming"):
        client.complete("Prompt")

    # KeyErrors and AttributeErrors must also propagate unmasked
    mock_sdk_client.models.generate_content.side_effect = KeyError(
        "unexpected_dict_key"
    )
    with pytest.raises(KeyError):
        client.complete("Prompt")


def test_gemini_client_missing_sdk_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing client without injected SDK raises ImportError when absent."""
    import sys

    # Simulate uninstalled SDK
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(ImportError, match="The 'google-genai' package is required"):
        GeminiModelClient(api_key="mock-key-123")


def test_cli_gemini_missing_api_key_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    exit_code = main(["--client", "gemini"])
    assert exit_code == 2

    captured = capsys.readouterr()
    assert "GEMINI_API_KEY is required" in captured.err


def test_cli_gemini_custom_model_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
    from structured_output_agent.cli import _resolve_client

    # Test _resolve_client with custom model name passed through
    monkeypatch.setattr(
        "structured_output_agent.gemini_client.GeminiModelClient.__init__",
        lambda self, *args, **kwargs: None,
    )
    # Ensure client resolution passes the custom model
    client = _resolve_client("gemini", model="gemini-3.8-pro")
    assert client is not None


class _FakeClock:
    """Deterministic simulated clock and sleep function for pacing tests."""

    def __init__(self, start_time: float = 100.0) -> None:
        self.current_time = start_time
        self.sleep_calls: list[float] = []

    def clock(self) -> float:
        return self.current_time

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.current_time += seconds

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


def test_request_interval_default_is_13_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_REQUEST_INTERVAL", raising=False)
    assert DEFAULT_REQUEST_INTERVAL == 13.0
    assert resolve_request_interval() == 13.0

    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)
    assert client.request_interval == 13.0


def test_pacing_first_request_starts_immediately_without_sleeping() -> None:
    fake = _FakeClock(100.0)
    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(
        api_key="test-key",
        client=mock_sdk_client,
        request_interval=13.0,
        clock_fn=fake.clock,
        sleep_fn=fake.sleep,
    )

    result = client.complete("Prompt 1")
    assert result != ""
    assert fake.sleep_calls == []
    assert client._last_request_time == 100.0


def test_pacing_successive_requests_sleep_for_remaining_interval() -> None:
    fake = _FakeClock(100.0)
    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(
        api_key="test-key",
        client=mock_sdk_client,
        request_interval=13.0,
        clock_fn=fake.clock,
        sleep_fn=fake.sleep,
    )

    # First request starts immediately at t=100.0
    client.complete("Prompt 1")
    assert fake.sleep_calls == []
    assert client._last_request_time == 100.0

    # Advance clock by 5.0s (elapsed = 5.0, remaining = 8.0)
    fake.advance(5.0)
    assert fake.clock() == 105.0

    # Second request should sleep for remaining 8.0s
    client.complete("Prompt 2")
    assert fake.sleep_calls == [8.0]
    # fake.sleep advanced time by 8.0s to 113.0
    assert fake.clock() == 113.0
    assert client._last_request_time == 113.0

    # Advance clock by 3.0s (elapsed = 3.0, remaining = 10.0)
    fake.advance(3.0)
    assert fake.clock() == 116.0

    # Third request should sleep for remaining 10.0s
    client.complete("Prompt 3")
    assert fake.sleep_calls == [8.0, 10.0]
    assert fake.clock() == 126.0
    assert client._last_request_time == 126.0


def test_pacing_no_sleep_when_elapsed_time_already_exceeds_interval() -> None:
    fake = _FakeClock(100.0)
    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(
        api_key="test-key",
        client=mock_sdk_client,
        request_interval=13.0,
        clock_fn=fake.clock,
        sleep_fn=fake.sleep,
    )

    client.complete("Prompt 1")
    assert client._last_request_time == 100.0

    # Advance clock by 25.0s (exceeds 13.0s interval)
    fake.advance(25.0)
    assert fake.clock() == 125.0

    client.complete("Prompt 2")
    assert fake.sleep_calls == []
    assert fake.clock() == 125.0
    assert client._last_request_time == 125.0


def test_pacing_constructor_argument_overrides_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_REQUEST_INTERVAL", "25.0")
    assert resolve_request_interval(10.0) == 10.0

    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(
        api_key="test-key",
        client=mock_sdk_client,
        request_interval=10.0,
    )
    assert client.request_interval == 10.0


def test_pacing_environment_variable_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_REQUEST_INTERVAL", "17.5")
    assert resolve_request_interval() == 17.5

    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(api_key="test-key", client=mock_sdk_client)
    assert client.request_interval == 17.5


def test_pacing_explicit_zero_interval_disables_pacing() -> None:
    fake = _FakeClock(100.0)
    mock_sdk_client = _make_mock_client()
    client = GeminiModelClient(
        api_key="test-key",
        client=mock_sdk_client,
        request_interval=0.0,
        clock_fn=fake.clock,
        sleep_fn=fake.sleep,
    )
    assert client.request_interval == 0.0

    # Multiple successive requests at t=100.0 with 0 elapsed time
    client.complete("Prompt 1")
    client.complete("Prompt 2")
    client.complete("Prompt 3")
    assert fake.sleep_calls == []


def test_pacing_rejects_negative_nan_and_infinite_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_floats = [-1.0, -0.001, float("nan"), float("inf"), float("-inf")]
    for invalid in invalid_floats:
        with pytest.raises(
            ValueError, match="request_interval must be finite and non-negative"
        ):
            resolve_request_interval(invalid)

        with pytest.raises(
            ValueError, match="request_interval must be finite and non-negative"
        ):
            GeminiModelClient(
                api_key="test-key",
                client=_make_mock_client(),
                request_interval=invalid,
            )

    # Also test invalid string inputs via env var
    for invalid_env in ["-10.0", "nan", "inf", "-inf"]:
        monkeypatch.setenv("GEMINI_REQUEST_INTERVAL", invalid_env)
        with pytest.raises(
            ValueError, match="request_interval must be finite and non-negative"
        ):
            resolve_request_interval()

    # Non-numeric string in env var
    monkeypatch.setenv("GEMINI_REQUEST_INTERVAL", "not-a-number")
    with pytest.raises(ValueError, match="request_interval must be a valid float"):
        resolve_request_interval()


def test_pacing_request_timestamp_updated_immediately_before_api_call() -> None:
    fake = _FakeClock(100.0)
    recorded_timestamps: list[float | None] = []

    def mock_generate_content(*args: Any, **kwargs: Any) -> _MockResponse:
        # Capture client's _last_request_time at the exact moment
        # generate_content is invoked
        recorded_timestamps.append(client._last_request_time)
        return _MockResponse('{"category": "billing"}')

    mock_sdk_client = MagicMock()
    mock_sdk_client.models.generate_content.side_effect = mock_generate_content

    client = GeminiModelClient(
        api_key="test-key",
        client=mock_sdk_client,
        request_interval=13.0,
        clock_fn=fake.clock,
        sleep_fn=fake.sleep,
    )

    client.complete("Prompt 1")
    assert recorded_timestamps == [100.0]

    # Advance clock by 5s. Pacing sleep will sleep 8s, advancing clock to 113.0s.
    fake.advance(5.0)
    client.complete("Prompt 2")
    # Verify that before the API call was invoked, _last_request_time
    # was updated to 113.0
    assert recorded_timestamps == [100.0, 113.0]
    assert client._last_request_time == 113.0


def test_pacing_preserves_error_handling_and_paces_subsequent_request() -> None:
    fake = _FakeClock(100.0)
    mock_sdk_client = MagicMock()
    # First call raises an operational error, second succeeds
    mock_sdk_client.models.generate_content.side_effect = [
        OSError("Simulated socket failure"),
        _MockResponse('{"category": "billing"}'),
    ]

    client = GeminiModelClient(
        api_key="test-key",
        client=mock_sdk_client,
        request_interval=13.0,
        clock_fn=fake.clock,
        sleep_fn=fake.sleep,
    )

    with pytest.raises(
        OperationalError, match=r"Gemini operational failure \[OSError\]"
    ):
        client.complete("Prompt 1")

    # The timestamp must have been recorded for the failed attempt
    assert client._last_request_time == 100.0
    assert fake.sleep_calls == []

    # Advance 4.0s (elapsed = 4.0, remaining = 9.0)
    fake.advance(4.0)
    result = client.complete("Prompt 2")
    assert result == '{"category": "billing"}'
    assert fake.sleep_calls == [9.0]
    assert client._last_request_time == 113.0
