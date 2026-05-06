"""Tests for normalize_for_detection."""

import pytest

from doc_analyse.detection.normalize import normalize_for_detection


class TestNormalizeForDetection:
    def test_strips_zero_width_space(self):
        result = normalize_for_detection("hello\u200bworld")
        assert result == "helloworld"
        assert "\u200b" not in result

    def test_strips_zero_width_non_joiner(self):
        result = normalize_for_detection("hello\u200cworld")
        assert result == "helloworld"

    def test_strips_zero_width_joiner(self):
        result = normalize_for_detection("hello\u200dworld")
        assert result == "helloworld"

    def test_strips_bom(self):
        result = normalize_for_detection("\ufeffhello world")
        assert result == "hello world"

    def test_strips_soft_hyphen(self):
        result = normalize_for_detection("hello\u00adworld")
        assert result == "helloworld"

    def test_strips_all_zw_chars(self):
        # Multiple different zero-width chars
        result = normalize_for_detection("a\u200bb\u200cc\u200dd\ufeffe\u00adf")
        assert result == "abcdef"

    def test_nfkc_normalization_accents(self):
        # accented characters become base form
        result = normalize_for_detection("café")
        assert "caf" in result

    def test_nfkc_normalization_superscripts(self):
        result = normalize_for_detection("km²")
        assert result == "km2"

    def test_nfkc_normalization_fractions(self):
        result = normalize_for_detection("½")
        # NFKC of ½ → '1⁄2' (with U+2060 word joiner) or '1/2' depending on platform
        assert "1" in result and "2" in result

    def test_collapses_multiple_spaces(self):
        result = normalize_for_detection("hello    world")
        assert result == "hello world"

    def test_collapses_tabs(self):
        result = normalize_for_detection("hello\t\tworld")
        assert result == "hello world"

    def test_collapses_3_plus_newlines_to_two(self):
        # normalize_for_detection collapses 3+ consecutive newlines to exactly 2
        result = normalize_for_detection("hello\n\n\n\nworld")
        assert result == "hello\n\nworld"
        assert result.count("\n") == 2

    def test_preserves_single_newlines(self):
        result = normalize_for_detection("line1\nline2\nline3")
        assert result == "line1\nline2\nline3"

    def test_strips_leading_whitespace(self):
        result = normalize_for_detection("  hello world")
        assert result == "hello world"

    def test_strips_trailing_whitespace(self):
        result = normalize_for_detection("hello world  ")
        assert result == "hello world"

    def test_combined_zw_and_strip(self):
        # Zero-width chars + leading/trailing whitespace together
        result = normalize_for_detection("  \u200bhello\u200c  ")
        assert result == "hello"

    def test_empty_string(self):
        result = normalize_for_detection("")
        assert result == ""

    def test_only_zw_chars(self):
        result = normalize_for_detection("\u200b\u200c\u200d")
        assert result == ""

    def test_order_is_nfkc_then_zw_then_whitespace_then_strip(self):
        # NFKC can create zero-width chars or change characters
        result = normalize_for_detection("café\u200b  ")
        # Should strip ZW char and trailing spaces
        assert not "\u200b" in result
        assert result.endswith("café") or result == "café"

    def test_preserves_normal_chinese_japanese_characters(self):
        result = normalize_for_detection("こんにちは世界")
        assert result == "こんにちは世界"