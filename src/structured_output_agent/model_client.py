from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelClient(Protocol):
    """SDK-independent protocol for model client implementations.

    Implementations must make exactly one model request per call and must
    not retry, silently fall back, or substitute default predictions.
    Operational failures (e.g., transport errors, rate limits, timeouts) must
    be raised as an OperationalError.
    """

    def complete(self, prompt: str) -> str:
        """Send a prompt to the model and return the raw output text.

        Args:
            prompt: Formatted prompt text to submit.

        Returns:
            The raw, unmodified text returned by the model provider.

        Raises:
            OperationalError: When a provider, transport, rate limit, or
                infrastructure error occurs.
        """
        ...


def resolve_client_provenance(client: ModelClient) -> tuple[str, str | None]:
    """Extract (client_type, model_id) provenance metadata from a ModelClient."""
    client_type = getattr(client, "client_type", None)
    if client_type is None:
        name = client.__class__.__name__
        if "Gemini" in name:
            client_type = "gemini"
        elif "Stub" in name:
            client_type = "stub"
        else:
            client_type = name.lower()

    model_id = getattr(client, "model_id", None)
    if model_id is None:
        model_id = getattr(client, "model", None)

    return str(client_type), (str(model_id) if model_id is not None else None)

