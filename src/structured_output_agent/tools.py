"""Tool registry and execution infrastructure with strict validation.

Provides:
- Strict Pydantic schemas for tool arguments and results.
- Safe ToolRegistry with input validation, output validation, and identity checks.
- Guard against decoy and destructive tools (e.g. delete_account).
- Separation of escalation handling from requires_human_review triage classification.
"""

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from .exceptions import ToolExecutionError, ToolResultMismatchError
from .schemas import (
    DECOY_TOOL_NAME,
    REGISTERED_TOOL_NAMES,
    EscalationStatus,
    NonEmptyText,
    StrictModel,
    TicketId,
    TicketStatus,
)


class TicketLookupArgs(StrictModel):
    ticket_id: TicketId


class TicketLookupResult(StrictModel):
    ticket_id: TicketId
    status: TicketStatus
    details: NonEmptyText


class RequestEscalationArgs(StrictModel):
    ticket_id: TicketId
    reason: NonEmptyText


class RequestEscalationResult(StrictModel):
    ticket_id: TicketId
    status: EscalationStatus = EscalationStatus.QUEUED
    escalation_id: NonEmptyText


# Type for tool handlers
ToolHandler = Callable[[dict[str, Any]], StrictModel]


def default_ticket_lookup_handler(args: dict[str, Any]) -> TicketLookupResult:
    """Default stub handler for ticket_lookup."""
    validated_args = TicketLookupArgs.model_validate(args)
    return TicketLookupResult(
        ticket_id=validated_args.ticket_id,
        status=TicketStatus.OPEN,
        details=f"Lookup details for ticket {validated_args.ticket_id}",
    )


def default_request_escalation_handler(args: dict[str, Any]) -> RequestEscalationResult:
    """Default stub handler for request_escalation.

    Follows existing policy: queues the escalation without altering or conflating
    the requires_human_review triage decision flag.
    """
    validated_args = RequestEscalationArgs.model_validate(args)
    return RequestEscalationResult(
        ticket_id=validated_args.ticket_id,
        status=EscalationStatus.QUEUED,
        escalation_id=f"esc-{validated_args.ticket_id}",
    )


class ToolRegistry:
    """Safe registry for agent tool execution.

    Enforces:
    - Rejection of destructive/decoy tools (e.g. delete_account).
    - Rejection of unregistered/unauthorized tools.
    - Pydantic input argument validation.
    - Pydantic output result validation.
    - Entity identity check: result.ticket_id must match args.ticket_id.
    - Preservation of policy separation: escalation does not mutate
      requires_human_review.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {
            "ticket_lookup": default_ticket_lookup_handler,
            "request_escalation": default_request_escalation_handler,
        }

    def register_handler(self, tool_name: str, handler: ToolHandler) -> None:
        """Register or override a tool handler for an approved registered tool."""
        if tool_name == DECOY_TOOL_NAME:
            raise ToolExecutionError(
                f"Cannot register handler for decoy tool {tool_name!r}."
            )
        if tool_name not in REGISTERED_TOOL_NAMES:
            raise ToolExecutionError(
                f"Cannot register handler for unregistered tool {tool_name!r}."
            )
        self._handlers[tool_name] = handler

    def execute(self, tool_name: str, args: dict[str, Any]) -> StrictModel:
        """Execute a tool with input/result validation and identity check.

        Raises:
            ToolExecutionError: When a decoy tool is called, an unauthorized tool
                is requested, or argument validation fails.
            ToolResultMismatchError: When the returned tool result does not match
                the requested ticket identity.
        """
        # 1. Guard against decoy or destructive tools
        if tool_name == DECOY_TOOL_NAME:
            raise ToolExecutionError(
                f"Destructive tool {tool_name!r} is rejected and cannot be executed."
            )

        # 2. Guard against unregistered / unauthorized tools
        if tool_name not in REGISTERED_TOOL_NAMES:
            raise ToolExecutionError(
                f"Unauthorized or unknown tool {tool_name!r}."
            )

        handler = self._handlers.get(tool_name)
        if handler is None:
            raise ToolExecutionError(
                f"No handler registered for tool {tool_name!r}."
            )

        # 3. Validate arguments
        if tool_name == "ticket_lookup":
            try:
                validated_args = TicketLookupArgs.model_validate(args)
            except ValidationError as exc:
                raise ToolExecutionError(
                    f"Invalid arguments for {tool_name!r}: {exc}"
                ) from exc
            expected_ticket_id = validated_args.ticket_id
        elif tool_name == "request_escalation":
            try:
                validated_args = RequestEscalationArgs.model_validate(args)
            except ValidationError as exc:
                raise ToolExecutionError(
                    f"Invalid arguments for {tool_name!r}: {exc}"
                ) from exc
            expected_ticket_id = validated_args.ticket_id
        else:
            raise ToolExecutionError(f"Unsupported tool {tool_name!r}.")

        # 4. Execute handler
        try:
            raw_result = handler(args)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool {tool_name!r} execution failed: {exc}"
            ) from exc

        # 5. Validate result schema
        if tool_name == "ticket_lookup":
            if not isinstance(raw_result, TicketLookupResult):
                try:
                    result_data = (
                        raw_result.model_dump()
                        if isinstance(raw_result, StrictModel)
                        else raw_result
                    )
                    raw_result = TicketLookupResult.model_validate(result_data)
                except ValidationError as exc:
                    raise ToolExecutionError(
                        f"Malformed result schema for {tool_name!r}: {exc}"
                    ) from exc
        elif tool_name == "request_escalation":
            if not isinstance(raw_result, RequestEscalationResult):
                try:
                    result_data = (
                        raw_result.model_dump()
                        if isinstance(raw_result, StrictModel)
                        else raw_result
                    )
                    raw_result = RequestEscalationResult.model_validate(result_data)
                except ValidationError as exc:
                    raise ToolExecutionError(
                        f"Malformed result schema for {tool_name!r}: {exc}"
                    ) from exc

        # 6. Identity check
        result_ticket_id = getattr(raw_result, "ticket_id", None)
        if result_ticket_id != expected_ticket_id:
            raise ToolResultMismatchError(
                f"Identity mismatch in {tool_name!r}: expected ticket_id "
                f"{expected_ticket_id!r}, got {result_ticket_id!r}."
            )

        return raw_result
