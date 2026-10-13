
class AgentError(Exception):
    """Base exception for the structured-output agent."""


class InvalidTicketInput(AgentError, ValueError):
    """Ticket input is empty or invalid before inference."""


class JSONExtractionError(AgentError, ValueError):
    """A usable JSON object could not be extracted from a response."""


class SchemaValidationError(AgentError, ValueError):
    """A parsed response does not satisfy the required schema."""


class SemanticValidationError(AgentError, ValueError):
    """A schema-valid response violates a deterministic policy rule."""


class ToolExecutionError(AgentError):
    """A tool failed to execute or rejected an operation."""


class ToolResultMismatchError(ToolExecutionError):
    """A tool result does not match the requested identity."""


class RetryExhaustedError(AgentError):
    """The agent could not produce a valid result within its retry budget."""


class OperationalError(AgentError):
    """Operational or infrastructure failure during model invocation.

    Operational failures (e.g., transport errors, timeouts, rate limits) abort
    the evaluation run and are not counted as model-output failures (O1-O3).
    """
