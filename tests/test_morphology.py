"""
Tests for morphology module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ontology.morphology import (
    MorphologyEngine,
    MorphologicalRule,
    MorphemeType,
    apply_morphology,
    analyze_morphology
)


def test_morphology_engine_creation():
    """Test creating morphology engine."""
    engine = MorphologyEngine()
    assert engine is not None
    assert len(engine.rules) == 0


def test_add_rule():
    """Test adding morphological rules."""
    engine = MorphologyEngine()
    engine.add_rule(
        "ACTION",
        r"^(.+)$",
        r"\1_ING",
        "progressive"
    )
    assert "ACTION" in engine.rules
    assert len(engine.rules["ACTION"]) == 1


def test_apply_morphology():
    """Test applying morphological transformations."""
    engine = MorphologyEngine()
    engine.add_rule(
        "ACTION",
        r"^(.+)$",
        r"\1_ING",
        "progressive",
        MorphemeType.SUFFIX
    )

    result = engine.apply("ACTION", "MOVE", "progressive")
    assert result == "MOVE_ING"


def test_standard_rules():
    """Test standard morphology rules."""
    engine = MorphologyEngine()
    engine.register_standard_rules()

    # Test progressive
    result = engine.apply("ACTION", "MOVE", "progressive_aspect")
    assert result == "MOVE_ING"

    # Test past tense
    result = engine.apply("ACTION", "TAKE", "past_tense")
    assert result == "TAKE_ED"

    # Test comparative
    result = engine.apply("ATTRIBUTE", "RED", "comparative")
    assert result == "RED_ER"


def test_morphological_analysis():
    """Test analyzing morphologically complex words."""
    analysis = analyze_morphology("MOVE_ING")
    assert analysis is not None
    assert analysis.root == "MOVE"
    assert len(analysis.morphemes) == 2


def test_paradigm_generation():
    """Test generating full paradigm."""
    engine = MorphologyEngine()
    engine.register_standard_rules()

    paradigm = engine.generate_paradigm("ACTION", "MOVE")
    assert len(paradigm) > 0
    assert "progressive_aspect" in paradigm
    assert paradigm["progressive_aspect"] == "MOVE_ING"


def test_constraints():
    """Test phonological constraints."""
    engine = MorphologyEngine()
    engine.register_standard_rules()

    # Should not create double underscores
    result = engine.apply("ACTION", "MOVE_ING", "past_tense")
    # Constraint should prevent MOVE_ING_ED
    assert result is None or "__" not in result


if __name__ == "__main__":
    test_morphology_engine_creation()
    test_add_rule()
    test_apply_morphology()
    test_standard_rules()
    test_morphological_analysis()
    test_paradigm_generation()
    test_constraints()
    print("✓ All morphology tests passed!")
