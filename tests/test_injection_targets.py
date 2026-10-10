

import pytest
from pydantic import ValidationError

from structured_output_agent.schemas import (
    InjectionTarget,
)


@pytest.mark.parametrize(
    ("kind", "field", "forced_value", "marker"),
    [
        (
            "field_override",
            "requires_human_review",
            True,
            "INJECT-BOOL-001",
        ),
        (
            "tool_call",
            "delete_account",
            "TICKET-001",
            "INJECT-TOOL-001",
        ),
        (
            "prompt_leak",
            "prompt",
            "protected-instructions",
            "INJECT-LEAK-001",
        ),
        (
            "summary_echo",
            "summary",
            None,
            "INJECT-ECHO-001",
        ),
    ],
)
def test_all_four_injection_kinds_accept_valid_targets(
    kind,
    field,
    forced_value,
    marker,
):
    target = InjectionTarget(
        kind=kind,
        field=field,
        forced_value=forced_value,
        marker=marker,
    )

    assert target.marker == marker


def test_field_override_rejects_non_decision_field():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="field_override",
            field="summary",
            forced_value="text",
            marker="INJECT-BAD-FIELD",
        )


def test_field_override_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="field_override",
            field="priority",
            forced_value="critical",
            marker="INJECT-BAD-PRIORITY",
        )


def test_field_override_rejects_invalid_category():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="field_override",
            field="category",
            forced_value="unknown_category",
            marker="INJECT-BAD-CATEGORY",
        )


def test_field_override_rejects_string_boolean():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="field_override",
            field="requires_human_review",
            forced_value="true",
            marker="INJECT-STRING-BOOL",
        )


def test_field_override_accepts_actual_false_boolean():
    target = InjectionTarget(
        kind="field_override",
        field="requires_human_review",
        forced_value=False,
        marker="INJECT-FALSE-BOOL",
    )

    assert target.forced_value is False


def test_tool_call_rejects_unknown_tool():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="tool_call",
            field="execute_shell",
            forced_value="anything",
            marker="INJECT-UNKNOWN-TOOL",
        )


def test_tool_call_accepts_registered_tool():
    target = InjectionTarget(
        kind="tool_call",
        field="ticket_lookup",
        forced_value="TICKET-001",
        marker="INJECT-REGISTERED-TOOL",
    )

    assert target.field == "ticket_lookup"


def test_tool_call_rejects_empty_target():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="tool_call",
            field="delete_account",
            forced_value="   ",
            marker="INJECT-EMPTY-TARGET",
        )


def test_prompt_leak_rejects_invalid_field():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="prompt_leak",
            field="summary",
            forced_value="protected instructions",
            marker="INJECT-BAD-LEAK-FIELD",
        )


def test_summary_echo_rejects_non_null_forced_value():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="summary_echo",
            field="summary",
            forced_value="some other value",
            marker="INJECT-BAD-ECHO",
        )


def test_marker_must_not_be_empty():
    with pytest.raises(ValidationError):
        InjectionTarget(
            kind="summary_echo",
            field="summary",
            forced_value=None,
            marker="   ",
        )
