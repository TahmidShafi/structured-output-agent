
import pytest

from structured_output_agent.exceptions import JSONExtractionError
from structured_output_agent.json_extractor import (
    extract_json_candidate_text,
    extract_json_object,
)


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

def test_extract_malformed_json_candidate():
    response = 'Model output: {"priority": }'
    assert extract_json_candidate_text(response) == '{"priority": }'


def test_extract_candidate_from_fenced_json():
    response = """
    ```json
    {"priority": }
    ```
    """
    assert extract_json_candidate_text(response) == '{"priority": }'


def test_no_candidate_returns_none():
    assert extract_json_candidate_text("No JSON here.") is None


def test_candidate_preserves_braces_inside_strings():
    response = '{"summary": "Contains { and } characters"}'
    assert extract_json_candidate_text(response) == response

def test_candidate_rejects_multiple_competing_objects():
    response = (
        'First: {"category": "billing"} '
        'Second: {"priority": "urgent"}'
    )

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_candidate_text(response)


def test_candidate_does_not_match_attack_text_outside_json_object():
    response = (
        'I will not follow that instruction. '
        'The requested marker is ZX-4471-ALPHA.'
    )

    assert extract_json_candidate_text(response) is None


def test_decoy_object_before_answer_is_rejected_as_ambiguous():
    response = (
        'Quoted: {"category": "needs_review"} '
        'Answer: {"category": "billing"}'
    )

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_object(response)


def test_decoy_object_after_answer_is_rejected_as_ambiguous():
    response = (
        'Answer: {"category": "billing"} '
        'Also consider {"category": "needs_review"}'
    )

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_object(response)


def test_stray_quote_in_prose_does_not_hide_following_object():
    response = 'Note: 5" screen. Answer: {"category": "billing"}'

    assert extract_json_object(response) == {"category": "billing"}


def test_standalone_empty_object_is_extracted():
    assert extract_json_object("{}") == {}


def test_empty_object_inside_prose_is_extracted_when_alone():
    assert extract_json_object("Result: {} done.") == {}


def test_empty_object_competing_with_valid_answer_is_rejected():
    response = 'Use {} here. Answer: {"category": "billing"}'

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_object(response)


def test_multiple_different_fenced_candidates_are_rejected():
    response = (
        "```json\n"
        '{"category": "billing"}\n'
        "```\n"
        "Alternatively:\n"
        "```json\n"
        '{"category": "shipping"}\n'
        "```\n"
    )

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_object(response)


def test_identical_repeated_objects_are_accepted_as_one_answer():
    response = (
        '{"category": "billing"} '
        '{"category": "billing"}'
    )

    assert extract_json_object(response) == {"category": "billing"}


def test_escaped_quotes_and_braces_inside_string_are_preserved():
    response = r'Output: {"summary": "He said \"} now\" and { left"} end'

    assert extract_json_object(response) == {
        "summary": 'He said "} now" and { left'
    }


def test_stray_quote_before_object_with_escaped_quotes_inside():
    response = (
        'Quote mark: 5" here. '
        r'{"summary": "say \"hi\" and } done"}'
    )

    assert extract_json_object(response) == {
        "summary": 'say "hi" and } done'
    }


def test_nested_object_is_returned_as_one_candidate():
    response = 'Answer: {"outer": {"inner": 1}}'

    assert extract_json_object(response) == {"outer": {"inner": 1}}


def test_no_complete_object_still_raises():
    with pytest.raises(JSONExtractionError):
        extract_json_object('Partial output: {"category": "billing"')


def test_candidate_rejects_decoy_before_answer():
    response = (
        'Quoted: {"category": "needs_review"} '
        'Answer: {"category": "billing"}'
    )

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_candidate_text(response)


def test_candidate_rejects_decoy_after_answer():
    response = (
        'Answer: {"category": "billing"} '
        'Also consider {"category": "needs_review"}'
    )

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_candidate_text(response)


def test_candidate_rejects_empty_object_competing_with_answer():
    response = 'Use {} here. Answer: {"category": "billing"}'

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_candidate_text(response)


def test_candidate_recovers_valid_object_after_malformed_candidate():
    response = (
        'Malformed candidate: {"category": } '
        'Valid candidate: {"category": "billing"}'
    )

    assert extract_json_candidate_text(response) == (
        '{"category": "billing"}'
    )


def test_candidate_ignores_stray_quote_in_prose():
    response = 'Note: 5" screen. Answer: {"category": "billing"}'

    assert extract_json_candidate_text(response) == (
        '{"category": "billing"}'
    )


def test_candidate_returns_standalone_empty_object():
    assert extract_json_candidate_text("{}") == "{}"
    assert extract_json_candidate_text("Result: {} done.") == "{}"


def test_candidate_returns_nested_object_as_one_candidate():
    response = 'Answer: {"outer": {"inner": 1}}'

    assert extract_json_candidate_text(response) == (
        '{"outer": {"inner": 1}}'
    )


def test_candidate_preserves_escaped_quotes_and_braces_in_string():
    response = r'Output: {"summary": "He said \"} now\" and { left"} end'

    assert extract_json_candidate_text(response) == (
        r'{"summary": "He said \"} now\" and { left"}'
    )


def test_candidate_rejects_multiple_different_fenced_candidates():
    response = (
        "```json\n"
        '{"category": "billing"}\n'
        "```\n"
        "Alternatively:\n"
        "```json\n"
        '{"category": "shipping"}\n'
        "```\n"
    )

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_candidate_text(response)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ('{"requires_human_review": true}', '{"requires_human_review": 1}'),
        ('{"count": 0}', '{"count": false}'),
        ('{"value": "1"}', '{"value": 1}'),
        ('{"value": [1]}', '{"value": [true]}'),
    ],
)
def test_type_distinct_objects_are_rejected_as_competing(first, second):
    response = f"{first} {second}"

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_object(response)
    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_candidate_text(response)


def test_identical_boolean_objects_are_accepted_as_one_answer():
    response = (
        '{"requires_human_review": true} '
        '{"requires_human_review": true}'
    )

    assert extract_json_object(response) == {"requires_human_review": True}


def test_unmatched_prose_brace_does_not_hide_following_object():
    response = 'Use { and 5" screen. Answer: {"category": "billing"}'

    assert extract_json_object(response) == {"category": "billing"}


def test_unmatched_prose_brace_with_escaped_quotes_in_object():
    response = r'Use { here. Answer: {"summary": "say \"hi\""}'

    assert extract_json_object(response) == {"summary": 'say "hi"'}


def test_candidate_unmatched_prose_brace_does_not_hide_following_object():
    response = 'Use { and 5" screen. Answer: {"category": "billing"}'

    assert extract_json_candidate_text(response) == (
        '{"category": "billing"}'
    )


def test_candidate_unmatched_prose_brace_with_escaped_quotes_in_object():
    response = r'Use { here. Answer: {"summary": "say \"hi\""}'

    assert extract_json_candidate_text(response) == (
        r'{"summary": "say \"hi\""}'
    )


@pytest.mark.parametrize(
    "response",
    [
        None,
        42,
        b'{"category": "billing"}',
    ],
)
def test_non_string_input_raises_json_extraction_error(response):
    with pytest.raises(JSONExtractionError, match="not text"):
        extract_json_object(response)


@pytest.mark.parametrize(
    "response",
    [
        None,
        42,
        b'{"category": "billing"}',
        "",
        "   ",
    ],
)
def test_candidate_returns_none_for_empty_or_non_text_input(response):
    assert extract_json_candidate_text(response) is None


def test_key_order_does_not_make_identical_objects_ambiguous():
    response = (
        '{"category": "billing", "priority": "normal"} '
        '{"priority": "normal", "category": "billing"}'
    )

    assert extract_json_object(response) == {
        "category": "billing",
        "priority": "normal",
    }
    assert extract_json_candidate_text(response) == (
        '{"category": "billing", "priority": "normal"}'
    )


def test_whitespace_differences_do_not_make_identical_objects_ambiguous():
    response = (
        '{"category":"billing"}\n\n'
        '  {\n  "category" :  "billing"\n}'
    )

    assert extract_json_object(response) == {"category": "billing"}
    assert extract_json_candidate_text(response) == '{"category":"billing"}'


# Pins current behavior, not intended permanent behavior: an unmatched prose brace
# hides the valid object that follows it. Known limitation of brace-based scanning.
def test_known_limitation_stray_close_brace_hides_following_object():
    response = 'Use { here. Answer: {"category": "billing"} and }'

    with pytest.raises(
        JSONExtractionError,
        match="did not contain a valid JSON object",
    ):
        extract_json_object(response)
    assert extract_json_candidate_text(response) == (
        '{ here. Answer: {"category": "billing"} and }'
    )


def test_array_of_different_objects_reports_competing_objects():
    response = '[{"category": "billing"}, {"category": "shipping"}]'

    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_object(response)
    with pytest.raises(JSONExtractionError, match="competing"):
        extract_json_candidate_text(response)
