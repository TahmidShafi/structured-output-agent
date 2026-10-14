"""Redacted, actionable feedback generator for retry attempts.

Ensures:
- No secrets, credentials, or API keys are ever leaked in error feedback.
- Unnecessary ticket data and input payloads are removed.
- Actionable, policy-aware guidance is provided for extraction, schema,
  and policy failures.
"""

import re

from .run_results import RunOutcome

# Patterns for sensitive tokens, API keys, and credentials
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z-_]{35}"),  # Google API key
    re.compile(r"sk-[0-9A-Za-z]{20,}"),  # Standard API key pattern
    re.compile(r"(?:Bearer\s+)[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"(?:key=)[A-Za-z0-9_\-]+", re.IGNORECASE),
    re.compile(r"(?:password|secret|token)\s*[:=]\s*[^\s,;]+", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Scrub sensitive credentials and keys from any error text."""
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _sanitize_pydantic_error_details(error_message: str) -> str:
    """Remove raw input payloads and verbose internal paths from Pydantic errors.

    Pydantic ValidationError strings often include `input_value='...'` which can
    echo raw ticket text or arbitrary user inputs. This strips those values.
    """
    cleaned = error_message
    # Remove [input_value=...] or input_value: ...
    cleaned = re.sub(
        r"input_value=['\"][^'\"]*['\"]", "input_value=[OMITTED]", cleaned
    )
    cleaned = re.sub(
        r"For further information visit https://errors\.pydantic\.dev/[^\s]*",
        "",
        cleaned,
    )
    # Strip excessively long lines
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    safe_lines = []
    for line in lines:
        if len(line) > 300:
            line = line[:300] + "... [truncated]"
        safe_lines.append(line)
    return " ".join(safe_lines)


def generate_actionable_feedback(
    outcome: RunOutcome,
    error_message: str | None = None,
) -> str:
    """Generate safe, actionable instructions for the model on retry."""
    safe_msg = redact_secrets(error_message or "")

    if outcome == RunOutcome.EXTRACTION_FAILURE:
        return (
            "Extraction failure: Your previous response could not be parsed as a "
            "single valid JSON object. Ensure you output exactly one JSON object "
            "with no surrounding Markdown fences (do not use ``` or ```json) and "
            "no conversational text before or after the JSON."
        )

    if outcome == RunOutcome.SCHEMA_FAILURE:
        cleaned_details = _sanitize_pydantic_error_details(safe_msg)
        return (
            f"Schema validation failure: {cleaned_details}\n"
            "Requirements reminder:\n"
            "- Exactly 5 fields: 'category', 'priority', 'assigned_team', "
            "'summary', 'requires_human_review'.\n"
            "- 'category' must be one of: 'billing', 'account_access', "
            "'technical_issue', 'shipping', 'other', 'needs_review'.\n"
            "- 'priority' must be one of: 'low', 'normal', 'high', 'urgent'.\n"
            "- 'assigned_team' must be one of: 'payments', 'account_support', "
            "'technical_support', 'fulfillment', 'general_support', 'human_review'.\n"
            "- 'summary' must be a non-empty string under 280 characters.\n"
            "- 'requires_human_review' must be a JSON boolean (true or false)."
        )

    if outcome == RunOutcome.POLICY_FAILURE:
        # C1 deterministic policy violation feedback
        feedback_lines = ["Deterministic policy violation:"]
        if "Validator A" in safe_msg or "requires" in safe_msg:
            feedback_lines.append(
                f"- Category-to-team alignment: {safe_msg}\n"
                "  Mapping rule: billing -> payments, "
                "account_access -> account_support, "
                "technical_issue -> technical_support, shipping -> fulfillment, "
                "other -> general_support, needs_review -> human_review."
            )
        if "Validator B" in safe_msg or "human review" in safe_msg:
            feedback_lines.append(
                f"- Human review rule: {safe_msg}\n"
                "  Rule: Category 'needs_review' and priority 'urgent' both strictly "
                "require 'requires_human_review: true'."
            )
        if len(feedback_lines) == 1:
            feedback_lines.append(f"- {safe_msg}")
        return "\n".join(feedback_lines)

    return f"Execution failure: {safe_msg}"
