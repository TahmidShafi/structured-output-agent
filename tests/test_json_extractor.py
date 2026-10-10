
import pytest

from structured_output_agent.exceptions import JSONExtractionError
from structured_output_agent.json_extractor import extract_json_object


def test_extract_direct_json_object():
    result = extract_json_object('{"category": "billing"}')
    assert result == {"category": "billing"}


def test_extract_json_code_fence():
    response = """
    ```json
    {"category": "shipping", "priority": "normal"}
    ```
    """
    result = extract_json_object(response)

    assert result == {
        "category": "shipping",
        "priority": "normal",
    }


def test_extract_object_surrounded_by_text():
    response = (
        'Here is the result: {"category": "billing"} '
        "Let me know if you need anything else."
    )
    assert extract_json_object(response) == {"category": "billing"}


def test_braces_inside_json_string_do_not_end_object_early():
    response = '{"summary": "The text contains { and } characters"}'
    assert extract_json_object(response) == {
        "summary": "The text contains { and } characters"
    }


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
        "This response contains no JSON.",
        '{"category": ',
        '["billing", "shipping"]',
        '"just a string"',
        "123",
    ],
)
def test_invalid_responses_raise_json_extraction_error(response):
    with pytest.raises(JSONExtractionError):
        extract_json_object(response)


def test_extract_valid_object_after_invalid_candidate():
    response = (
        'Malformed candidate: {"category": } '
        'Valid candidate: {"category": "billing"}'
    )
    assert extract_json_object(response) == {
        "category": "billing"
    }
