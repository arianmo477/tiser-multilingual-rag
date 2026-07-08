"""Unit tests for utils.metrics.

Run from the repo root:  python -m pytest tests/
"""

from utils.metrics import (
    calculate_metrics,
    chrf_score,
    is_english_leak,
    normalize_text,
    normalized_em,
)


def test_normalize_text_lowercases_and_strips_punct():
    assert normalize_text("Paris.") == "paris"
    assert normalize_text("  New   York! ") == "new york"
    # accented unicode word chars are preserved
    assert normalize_text("città") == "città"


def test_exact_match():
    m = calculate_metrics("Paris", ["Paris"])
    assert m["em"] == 1 and m["f1"] == 1.0 and m["chrf"] == 100.0


def test_exact_match_is_case_and_punct_insensitive():
    assert calculate_metrics("paris.", ["Paris"])["em"] == 1


def test_mismatch():
    m = calculate_metrics("London", ["Paris"])
    assert m["em"] == 0


def test_partial_f1_between_zero_and_one():
    f1 = calculate_metrics("New York City", ["New York"])["f1"]
    assert 0.0 < f1 < 1.0


def test_soft_em_substring():
    assert calculate_metrics("York", ["New York"])["soft_em"] == 1


def test_boolean_branch():
    assert calculate_metrics("true", ["True"])["em"] == 1
    assert calculate_metrics("false", ["true"])["em"] == 0


def test_normalized_em_is_order_insensitive():
    assert normalized_em("Hannover Leibniz University", "University Leibniz Hannover") == 1
    assert normalized_em("a b", "a c") == 0


def test_chrf_identical_is_100():
    assert chrf_score("Paris", "Paris") == 100.0


def test_english_leak_flag():
    # non-English gold, but the prediction returned the English answer
    assert is_english_leak("Rome", gold_target="Roma", gold_en="Rome") == 1
    # answered in the target language -> no leak
    assert is_english_leak("Roma", gold_target="Roma", gold_en="Rome") == 0
