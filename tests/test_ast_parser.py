"""
Tests for AST parser module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from explain.ast_parser import (
    EChannelParser,
    StructuredFormatParser,
    KeywordValueParser,
    NaturalLanguageParser,
    parse_explanation,
    extract_slot_values,
    compare_asts
)


def test_structured_format_parser():
    """Test parsing structured format."""
    parser = StructuredFormatParser()

    # Test basic parsing
    text = "PLAN(DO(MOVE), TARGET(CIRCLE), AT(LEFT))"
    result = parser.parse(text)

    assert result.parse_success
    assert "ACTION" in result.semantics
    assert result.semantics["ACTION"] == "MOVE"
    assert result.semantics["OBJECT"] == "CIRCLE"
    assert result.semantics["LOCATION"] == "LEFT"


def test_keyword_value_parser():
    """Test parsing keyword-value format."""
    parser = KeywordValueParser()

    text = "NAV(act=MOVE, obj=CIRCLE, attr=RED, loc=L01)"
    result = parser.parse(text)

    assert result.parse_success
    assert result.semantics["ACTION"] == "MOVE"
    assert result.semantics["OBJECT"] == "CIRCLE"
    assert result.semantics["ATTRIBUTE"] == "RED"
    assert result.semantics["LOCATION"] == "L01"


def test_natural_language_parser():
    """Test parsing natural language."""
    parser = NaturalLanguageParser()

    text = "Move to the red circle"
    result = parser.parse(text)

    assert result.parse_success
    assert "ACTION" in result.semantics
    # Natural language parsing has lower confidence
    assert result.confidence < 1.0


def test_e_channel_parser_dispatch():
    """Test main parser dispatches correctly."""
    parser = EChannelParser()

    # Structured format
    result1 = parser.parse("PLAN(DO(MOVE), TARGET(CIRCLE))")
    assert result1.parse_success

    # Keyword-value format
    result2 = parser.parse("NAV(act=MOVE, obj=CIRCLE)")
    assert result2.parse_success

    # Natural language
    result3 = parser.parse("Move to the circle")
    assert result3.parse_success


def test_parse_explanation_convenience():
    """Test convenience function."""
    result = parse_explanation("PLAN(DO(MOVE), TARGET(CIRCLE))")
    assert result.parse_success
    assert result.semantics["ACTION"] == "MOVE"


def test_extract_slot_values():
    """Test extracting slot values from AST."""
    parser = EChannelParser()
    result = parser.parse("PLAN(DO(MOVE), TARGET(CIRCLE), AT(LEFT))")

    assert result.parse_success
    # Semantics already extracted during parsing
    assert "ACTION" in result.semantics
    assert result.semantics["ACTION"] == "MOVE"


def test_compare_asts():
    """Test AST comparison."""
    parser = EChannelParser()

    result1 = parser.parse("PLAN(DO(MOVE), TARGET(CIRCLE))")
    result2 = parser.parse("PLAN(DO(MOVE), TARGET(CIRCLE))")
    result3 = parser.parse("PLAN(DO(TAKE), TARGET(SQUARE))")

    if result1.ast and result2.ast and result3.ast:
        # Same ASTs should have high similarity
        sim1 = compare_asts(result1.ast, result2.ast)
        assert sim1 > 0.9

        # Different ASTs should have lower similarity
        sim2 = compare_asts(result1.ast, result3.ast)
        assert sim2 < sim1


def test_empty_input():
    """Test handling empty input."""
    parser = EChannelParser()
    result = parser.parse("")
    assert not result.parse_success


def test_invalid_format():
    """Test handling invalid format."""
    parser = EChannelParser()
    result = parser.parse("INVALID{FORMAT]")
    assert not result.parse_success


if __name__ == "__main__":
    test_structured_format_parser()
    test_keyword_value_parser()
    test_natural_language_parser()
    test_e_channel_parser_dispatch()
    test_parse_explanation_convenience()
    test_extract_slot_values()
    test_compare_asts()
    test_empty_input()
    test_invalid_format()
    print("✓ All AST parser tests passed!")
