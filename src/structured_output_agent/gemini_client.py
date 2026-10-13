"""Google Gemini model client adapter implementing ModelClient Protocol.

Provides:
- GeminiModelClient: concrete SDK-based or mock-based model client for Gemini.
- resolve_gemini_model: precedence resolver (--model > GEMINI_MODEL > default).
- resolve_request_interval: precedence resolver for live request pacing.
- _sanitize_error_message: redaction utility preventing credential leakage in errors.
"""

import math
import os
import re
import time
from collections.abc import Callable
from typing import Any

from .exceptions import OperationalError

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_INTERVAL = 13.0


def resolve_gemini_model(cli_model: str | None = None) -> str:
    """Resolve model string using precedence: CLI flag > GEMINI_MODEL > default.

    Precedence:
    1. CLI option (--model) if provided and non-empty.
    2. Environment variable (GEMINI_MODEL) if set and non-empty.
    3. Default constant (DEFAULT_GEMINI_MODEL = 'gemini-3.8-flash').
    """
    if cli_model is not None and cli_model.strip():
        return cli_model.strip()
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    if env_model:
        return env_model
    return DEFAULT_GEMINI_MODEL


def resolve_request_interval(interval: float | None = None) -> float:
    """Resolve request pacing interval using precedence: argument > env > default.

    Precedence:
    1. Explicit interval argument if not None.
    2. GEMINI_REQUEST_INTERVAL environment variable if set.
    3. Default constant (DEFAULT_REQUEST_INTERVAL = 13.0).

    Validates that the resolved interval is finite and non-negative.
    An interval of 0.0 explicitly disables pacing.
    Rejects negative values, NaN, and positive/negative infinity with ValueError.

    Rate-limit guarantee note:
    A 13-second interval is conservative client-side pacing intended to reduce
    the risk of exceeding the displayed 5-RPM limit. It does not guarantee
    compliance with every provider quota or rolling-window interpretation.
    Provider enforcement, other project requests, and additional quotas can
    still cause HTTP 429 errors.
    """
    raw_val: float | str | None = interval
    if raw_val is None:
        env_val = os.environ.get("GEMINI_REQUEST_INTERVAL", "").strip()
        if env_val:
            raw_val = env_val
        else:
            return DEFAULT_REQUEST_INTERVAL

    try:
        val = float(raw_val)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"request_interval must be a valid float, got {raw_val!r}."
        ) from exc

    if math.isnan(val) or math.isinf(val) or val < 0.0:
        raise ValueError(
            f"request_interval must be finite and non-negative, got {val}."
        )

    return val


def _sanitize_error_message(message: str, secret: str | None = None) -> str:
    """Redact secret strings and URL query key parameters from error messages."""
    sanitized = message
    if secret and secret in sanitized:
        sanitized = sanitized.replace(secret, "[REDACTED]")
    # Redact URL query parameters like ?key=... or key=...
    sanitized = re.sub(r"(key=)[A-Za-z0-9_\-]+", r"\1[REDACTED]", sanitized)
    return sanitized


class GeminiModelClient:
    """Google Gemini model client adapter implementing ModelClient Protocol.

    Satisfies:
        complete(prompt: str) -> str

    Guarantees & Semantics:
        - Makes one application-level generate_content() call per ticket.
        - Configures SDK retry options to attempts=1 (HttpRetryOptions(attempts=1))
          to disable SDK exponential backoff retries on transient errors.
        - Note on network attempts: While SDK-level retries are disabled, one
          application-level call may involve internal SDK operations (such as initial
          auth token validation), so one application-level call does not strictly
          guarantee exactly one network packet/attempt under all SDK conditions.
        - Paces live requests with a conservative client-side interval (default 13.0s)
          intended to reduce the risk of exceeding the displayed 5-RPM limit. This
          pacing does not guarantee compliance with every provider quota or rolling-
          window interpretation; provider enforcement, concurrent project requests,
          and additional quotas can still cause HTTP 429 errors.
        - Maps API errors, rate limits (429), auth failures (401/403), 5xx server
          errors, and transport/timeout failures to OperationalError.
        - Returns raw string response, returning "" for blocked/empty output.
        - Never logs or exposes API credentials in representations or errors.
    """

    client_type: str = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        temperature: float | None = DEFAULT_TEMPERATURE,
        client: Any | None = None,
        request_interval: float | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        raw_key = (
            api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        )
        if not raw_key or not raw_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is required. Please set the GEMINI_API_KEY "
                "environment variable or provide api_key."
            )
        self._api_key = raw_key.strip()
        self.model = resolve_gemini_model(model)
        self.model_id = self.model
        self.temperature = temperature
        self.request_interval = resolve_request_interval(request_interval)
        self._clock_fn = clock_fn
        self._sleep_fn = sleep_fn
        self._last_request_time: float | None = None

        if client is not None:
            self._client = client
        else:
            try:
                from google import genai  # type: ignore
                from google.genai import types  # type: ignore

                # Explicitly disable SDK-level automatic retries (1 attempt, 0 retries)
                http_options = types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)
                )
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=http_options,
                )
            except ImportError as exc:
                raise ImportError(
                    "The 'google-genai' package is required to use GeminiModelClient. "
                    "Install it via: pip install 'google-genai>=1.0.0,<3.0.0' or "
                    "pip install -e '.[gemini]'"
                ) from exc

    def _is_operational_failure(self, exc: Exception) -> bool:
        """Determine whether an exception represents a provider or transport failure."""
        error_modules = (
            "google.genai.errors",
            "httpx",
            "http.client",
            "urllib.error",
        )
        exc_module = getattr(type(exc), "__module__", "")
        if any(exc_module.startswith(mod) for mod in error_modules):
            return True

        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True

        exc_name = type(exc).__name__
        if exc_name in (
            "APIError",
            "ClientError",
            "ServerError",
            "HTTPError",
            "TimeoutException",
            "NetworkError",
            "RemoteProtocolError",
        ):
            return True

        return False

    def complete(self, prompt: str) -> str:
        """Submit prompt to Gemini model and return raw text output.

        Raises:
            OperationalError: If provider API or network transport failure occurs.
        """
        config = None
        try:
            from google.genai import types  # type: ignore

            http_options = types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            )
            config_kwargs: dict[str, Any] = {"http_options": http_options}
            if self.temperature is not None:
                config_kwargs["temperature"] = self.temperature
            config = types.GenerateContentConfig(**config_kwargs)
        except (ImportError, AttributeError):
            # Fallback dictionary if types is not available or mocked
            config = (
                {"temperature": self.temperature}
                if self.temperature is not None
                else None
            )

        kwargs: dict[str, Any] = {"model": self.model, "contents": prompt}
        if config is not None:
            kwargs["config"] = config

        if self._last_request_time is not None and self.request_interval > 0.0:
            elapsed = self._clock_fn() - self._last_request_time
            remaining = self.request_interval - elapsed
            if remaining > 0.0:
                self._sleep_fn(remaining)

        self._last_request_time = self._clock_fn()

        try:
            response = self._client.models.generate_content(**kwargs)
        except Exception as exc:
            if self._is_operational_failure(exc):
                sanitized = _sanitize_error_message(str(exc), self._api_key)
                error_cls = type(exc).__name__
                raise OperationalError(
                    f"Gemini operational failure [{error_cls}]: {sanitized}"
                ) from exc
            # Unexpected programming errors propagate naturally
            raise

        text = getattr(response, "text", None)
        if not text:
            return ""
        return str(text)

    def __repr__(self) -> str:
        return (
            f"GeminiModelClient(model={self.model!r}, "
            f"temperature={self.temperature})"
        )

    def __str__(self) -> str:
        return f"GeminiModelClient(model={self.model!r})"
