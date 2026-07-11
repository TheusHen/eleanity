from eleanity.comparators.diff import compare_prompt, compare_tokens
from eleanity.models.schemas import ParityResult


def test_prompt_diff_reports_line_column_and_markers():
    left = "line1\n<|im_start|>assistant\nhello"
    right = "line1\nassistant\nhello"
    result = compare_prompt(left, right)
    assert result.result == ParityResult.DIVERGENT
    assert result.details["first_character"] is not None
    assert result.details["line"] >= 1
    assert result.details["missing_assistant_turn"] is True
    assert "left_context" in result.details


def test_prompt_unicode_normalization_flag():
    # U+00E9 (é) vs e + combining acute
    left = "café"
    right = "cafe\u0301"
    result = compare_prompt(left, right)
    assert result.result == ParityResult.DIVERGENT
    assert result.details["unicode_nfc_equal"] is True


def test_token_diff_reports_ops_and_prefix():
    result = compare_tokens([1, 2, 3, 4, 5], [1, 2, 9, 4, 5])
    assert result.result == ParityResult.DIVERGENT
    assert result.details["first_difference"] == 2
    assert result.details["equal_prefix"] == 2
    assert result.details["expected_token_id"] == 3
    assert result.details["received_token_id"] == 9
    assert result.details["substituted"] >= 1


def test_token_strings_surface_in_details():
    result = compare_tokens(
        [10, 20],
        [10, 21],
        left_strings=["a", "b"],
        right_strings=["a", "c"],
    )
    assert result.details["expected_token_string"] == "b"
    assert result.details["received_token_string"] == "c"
