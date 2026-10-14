"""Tests for safe tool infrastructure and registry.

Verifies:
- Argument validation and rejection of malformed inputs.
- Result validation and rejection of malformed handler outputs.
- Identity checks: ToolResultMismatchError when returned ticket_id mismatches.
- Decoy tool rejection: delete_account is rejected with ToolExecutionError.
- Unauthorized tool rejection: unknown tools are rejected with ToolExecutionError.
- Escalation policy separation: request_escalation returns queued status and does not
  conflate with or mutate requires_human_review.
"""

from typing import Any

import pytest

from structured_output_agent.exceptions import (
    ToolExecutionError,
    ToolResultMismatchError,
)
from structured_output_agent.schemas import (
    EscalationStatus,
    TicketStatus,
)
from structured_output_agent.tools import (
    RequestEscalationResult,
    TicketLookupResult,
    ToolRegistry,
)


def test_tool_registry_ticket_lookup_success() -> None:
    registry = ToolRegistry()
    result = registry.execute("ticket_lookup", {"ticket_id": "TICKET-123"})

    assert isinstance(result, TicketLookupResult)
    assert result.ticket_id == "TICKET-123"
    assert result.status == TicketStatus.OPEN
    assert "TICKET-123" in result.details


def test_tool_registry_request_escalation_success() -> None:
    registry = ToolRegistry()
    result = registry.execute(
        "request_escalation",
        {"ticket_id": "TICKET-456", "reason": "Customer requested tier-2 support"},
    )

    assert isinstance(result, RequestEscalationResult)
    assert result.ticket_id == "TICKET-456"
    assert result.status == EscalationStatus.QUEUED
    assert result.escalation_id == "esc-TICKET-456"
    # Escalation queues the request but does not have or touch requires_human_review
    assert not hasattr(result, "requires_human_review")


def test_tool_registry_rejects_decoy_tool_delete_account() -> None:
    registry = ToolRegistry()

    # delete_account is a decoy / destructive tool and must NEVER execute
    with pytest.raises(
        ToolExecutionError, match="Destructive tool 'delete_account' is rejected"
    ):
        registry.execute("delete_account", {"account_id": "ACC-999"})


def test_tool_registry_cannot_register_handler_for_decoy_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolExecutionError, match="Cannot register handler for decoy tool"
    ):
        registry.register_handler("delete_account", lambda args: None)  # type: ignore


def test_tool_registry_rejects_unauthorized_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolExecutionError, match="Unauthorized or unknown tool 'restart_database'"
    ):
        registry.execute("restart_database", {"force": True})


def test_tool_registry_cannot_register_unregistered_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolExecutionError, match="Cannot register handler for unregistered tool"
    ):
        registry.register_handler("arbitrary_tool", lambda args: None)  # type: ignore


def test_tool_registry_malformed_arguments_missing_field() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolExecutionError, match="Invalid arguments for 'ticket_lookup'"
    ):
        registry.execute("ticket_lookup", {})


def test_tool_registry_malformed_arguments_extra_field() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolExecutionError, match="Invalid arguments for 'ticket_lookup'"
    ):
        registry.execute(
            "ticket_lookup",
            {"ticket_id": "TICKET-101", "unwanted_extra_field": "injected"},
        )


def test_tool_registry_malformed_arguments_empty_string() -> None:
    registry = ToolRegistry()

    with pytest.raises(
        ToolExecutionError, match="Invalid arguments for 'ticket_lookup'"
    ):
        registry.execute("ticket_lookup", {"ticket_id": "   "})


def test_tool_registry_identity_mismatch_raises_error() -> None:
    registry = ToolRegistry()

    # Custom handler simulating a rogue or buggy backend returning wrong ticket identity
    def rogue_lookup_handler(args: dict[str, Any]) -> TicketLookupResult:
        return TicketLookupResult(
            ticket_id="DIFFERENT-TICKET-999",
            status=TicketStatus.OPEN,
            details="Data from another ticket",
        )

    registry.register_handler("ticket_lookup", rogue_lookup_handler)

    with pytest.raises(
        ToolResultMismatchError,
        match="Identity mismatch in 'ticket_lookup'",
    ):
        registry.execute("ticket_lookup", {"ticket_id": "TICKET-123"})


def test_tool_registry_malformed_result_schema_rejected() -> None:
    registry = ToolRegistry()

    # Handler returning a dict that fails result schema validation
    def broken_handler(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "ticket_id": args["ticket_id"],
            "status": "not_a_valid_status_enum",
            "details": "Details",
        }

    registry.register_handler("ticket_lookup", broken_handler)  # type: ignore[arg-type]

    with pytest.raises(
        ToolExecutionError, match="Malformed result schema for 'ticket_lookup'"
    ):
        registry.execute("ticket_lookup", {"ticket_id": "TICKET-123"})
