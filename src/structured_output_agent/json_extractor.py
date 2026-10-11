
import json
import re
from typing import Any

from .exceptions import JSONExtractionError

FENCED_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _matching_brace_end(text: str, start: int) -> int | None:
    """Return the index of the brace closing the object opened at start."""
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        character = text[index]

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
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index

    return None


def _balanced_object_candidates(text: str) -> list[str]:
    """Find non-overlapping balanced JSON-object-shaped substrings in text.

    Each opening brace is matched independently, so quotation marks or braces
    in surrounding prose cannot hide a complete object that follows them.
    """
    candidates: list[str] = []
    index = 0

    while (start := text.find("{", index)) != -1:
        end = _matching_brace_end(text, start)
        if end is None:
            index = start + 1
            continue

        candidates.append(text[start:end + 1])
        index = end + 1

    return candidates


def _parsed_object_candidates(text: str) -> list[tuple[str, dict[str, Any]]]:
    candidates = [text]

    candidates.extend(
        match.group(1).strip()
        for match in FENCED_BLOCK_PATTERN.finditer(text)
    )

    candidates.extend(_balanced_object_candidates(text))

    seen: set[str] = set()
    parsed_objects: list[tuple[str, dict[str, Any]]] = []

    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            parsed_objects.append((candidate, parsed))

    return parsed_objects


def _canonical_json(parsed: dict[str, Any]) -> str:
    # Python equality treats true, 1, and 0 as equal to False; canonical JSON does not.
    return json.dumps(parsed, sort_keys=True)


def _unique_parsed_object(text: str) -> tuple[str, dict[str, Any]] | None:
    """Return the only parseable object in text, or None when none parses.

    Raises JSONExtractionError when parseable objects disagree, because
    choosing one would be arbitrary.
    """
    parsed_objects = _parsed_object_candidates(text)
    if not parsed_objects:
        return None

    first_text, first = parsed_objects[0]
    first_canonical = _canonical_json(first)
    if any(
        _canonical_json(parsed) != first_canonical
        for _, parsed in parsed_objects[1:]
    ):
        raise JSONExtractionError(
            "Response contains multiple competing JSON objects."
        )

    return first_text, first


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract the single JSON object from a model response.

    Raises JSONExtractionError when the response is empty or not text, when no
    object parses, or when parseable objects disagree, because choosing one
    would be arbitrary.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise JSONExtractionError("Response is empty or is not text.")

    found = _unique_parsed_object(raw.strip())
    if found is None:
        raise JSONExtractionError(
            "Response did not contain a valid JSON object."
        )

    return found[1]


def extract_json_candidate_text(raw: str) -> str | None:
    """Return the text of a JSON-object-shaped candidate from a model response.

    Candidates are parsed to detect ambiguity and to prefer a single parseable
    object. The returned text is untrusted model output. Returning it does not
    mean the JSON or its contents are valid; callers must parse it and validate
    it against the relevant schema and policy before use.

    Returns the text of the only parseable object when there is one. Otherwise
    returns the first object-shaped candidate, which may be malformed. Scanning
    is brace-based, so a malformed outer object can hide a valid object nested
    inside it, and an unmatched brace in prose can hide a valid object that
    follows it. These are known limitations. Raises JSONExtractionError when
    parseable objects disagree. Returns None when no complete object-shaped
    candidate is found.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None

    text = raw.strip()

    found = _unique_parsed_object(text)
    if found is not None:
        return found[0]

    for match in FENCED_BLOCK_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    candidates = _balanced_object_candidates(text)
    if candidates:
        return candidates[0]

    return None
