
import json
import re
from typing import Any

from .exceptions import JSONExtractionError

FENCED_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _balanced_object_candidates(text: str) -> list[str]:
    """Find balanced JSON-object-shaped substrings in text."""
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:index + 1])
                start = None

    return candidates


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract and parse a JSON object from a model response."""
    if not isinstance(raw, str) or not raw.strip():
        raise JSONExtractionError("Response is empty or is not text.")

    text = raw.strip()

    candidates = [text]

    # Prefer explicitly fenced JSON blocks when present.
    candidates.extend(
        match.group(1).strip()
        for match in FENCED_BLOCK_PATTERN.finditer(text)
    )

    # Also support an object surrounded by ordinary explanatory text.
    candidates.extend(_balanced_object_candidates(text))

    seen: set[str] = set()

    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    raise JSONExtractionError(
        "Response did not contain a valid JSON object."
    )
