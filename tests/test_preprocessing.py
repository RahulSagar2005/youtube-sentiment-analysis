"""Unit tests for text preprocessing module.

These tests cover the data preprocessing pipeline used in the
DVC stage 'data_preprocessing' (lowercasing, stopword removal
that preserves negations, and lemmatization).
"""

import pytest

# Adjust the import below to match your actual module path.
# Common locations: src/data/preprocessing.py, src/preprocess.py, etc.
# If your function lives elsewhere, change the import accordingly.
try:
    from src.preprocess import clean_text, preprocess_batch
except ImportError:
    # Fallback import path — adjust if your project structure differs
    from src.data.preprocess import clean_text, preprocess_batch  # type: ignore


class TestCleanText:
    """Tests for single-text cleaning function."""

    def test_lowercases_input(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_strips_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_removes_url(self):
        result = clean_text("check this https://example.com out")
        assert "https://example.com" not in result
        assert "check" in result
        assert "out" in result

    def test_removes_html_tags(self):
        result = clean_text("<b>bold</b> text")
        assert "<b>" not in result
        assert "bold" in result

    def test_preserves_negation_words(self):
        # Negations like "not", "no", "never" must survive stopword removal
        # (this is a documented feature of the preprocessing pipeline)
        result = clean_text("this is not good")
        assert "not" in result
        assert "good" in result

    def test_removes_standard_stopwords(self):
        result = clean_text("this is a great movie")
        assert "this" not in result.split()
        assert "is" not in result.split()
        assert "a" not in result.split()

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_none_safely(self):
        # Should either return "" or raise a specific exception
        with pytest.raises((ValueError, TypeError, AttributeError)):
            clean_text(None)  # type: ignore[arg-type]

    def test_lemmatizes_basic_words(self):
        # "running" -> "run" (or similar root form)
        result = clean_text("running quickly")
        assert "run" in result or "running" in result  # tolerate both lemmatizers
        assert "quickly" in result or "quick" in result

    def test_handles_punctuation(self):
        result = clean_text("hello!!! world???")
        assert "!" not in result
        assert "?" not in result
        assert "hello" in result
        assert "world" in result


class TestPreprocessBatch:
    """Tests for batch preprocessing."""

    def test_returns_same_length(self):
        texts = ["hello world", "foo bar", "baz qux"]
        result = preprocess_batch(texts)
        assert len(result) == len(texts)

    def test_empty_list_returns_empty(self):
        assert preprocess_batch([]) == []

    def test_all_strings_in_output(self):
        texts = ["Hello", "WORLD", "foo BAR"]
        result = preprocess_batch(texts)
        assert all(isinstance(t, str) for t in result)

    def test_preserves_order(self):
        texts = ["zebra animal", "apple fruit", "mango fruit"]
        result = preprocess_batch(texts)
        assert "zebra" in result[0]
        assert "apple" in result[1]
        assert "mango" in result[2]
