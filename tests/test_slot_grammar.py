"""
Tests for slot grammar module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ontology.slot_grammar import (
    SlotGrammar,
    SlotType,
    SlotValue,
    GrammarRule,
    create_standard_grammar
)


def test_slot_grammar_creation():
    """Test creating slot grammar."""
    grammar = SlotGrammar()
    assert grammar is not None
    assert grammar.start_symbol == "MESSAGE"


def test_add_rule():
    """Test adding grammar rules."""
    grammar = SlotGrammar()
    grammar.add_rule("MESSAGE", [SlotType.ACTION, SlotType.OBJECT])

    assert "MESSAGE" in grammar.rules
    assert len(grammar.rules["MESSAGE"]) == 1


def test_add_vocabulary():
    """Test adding slot vocabularies."""
    grammar = SlotGrammar()
    grammar.add_vocabulary(SlotType.ACTION, ["MOVE", "TAKE"])

    assert SlotType.ACTION in grammar.slot_vocabularies
    assert "MOVE" in grammar.slot_vocabularies[SlotType.ACTION]


def test_standard_grammar():
    """Test standard grammar creation."""
    grammar = create_standard_grammar()

    assert len(grammar.rules) > 0
    assert len(grammar.slot_vocabularies) > 0


def test_generation():
    """Test generating slot sequences."""
    grammar = create_standard_grammar()

    sequence = grammar.generate(max_depth=3)

    # Should generate non-empty sequence
    assert len(sequence) > 0
    # Should be SlotValue instances
    assert all(isinstance(s, SlotValue) for s in sequence)


def test_validation():
    """Test validating slot sequences."""
    grammar = create_standard_grammar()

    # Create valid sequence
    valid_sequence = [
        SlotValue(SlotType.ACTION, "MOVE"),
        SlotValue(SlotType.OBJECT, "CIRCLE")
    ]

    is_valid, errors = grammar.validate(valid_sequence)
    assert is_valid
    assert len(errors) == 0


def test_invalid_sequence():
    """Test detecting invalid sequences."""
    grammar = create_standard_grammar()

    # Create sequence with unknown word
    invalid_sequence = [
        SlotValue(SlotType.ACTION, "INVALID_ACTION"),
    ]

    is_valid, errors = grammar.validate(invalid_sequence)
    assert not is_valid
    assert len(errors) > 0


def test_slot_order():
    """Test slot ordering."""
    grammar = SlotGrammar()
    order = grammar.get_slot_order()

    assert SlotType.ACTION in order
    assert SlotType.OBJECT in order


def test_grammar_size_metrics():
    """Test computing grammar size."""
    grammar = create_standard_grammar()
    metrics = grammar.compute_grammar_size()

    assert "num_rules" in metrics
    assert "total_vocab" in metrics
    assert metrics["num_rules"] > 0


def test_slot_value_repr():
    """Test SlotValue string representation."""
    slot = SlotValue(SlotType.ACTION, "MOVE")
    repr_str = repr(slot)

    assert "ACTION" in repr_str
    assert "MOVE" in repr_str


if __name__ == "__main__":
    test_slot_grammar_creation()
    test_add_rule()
    test_add_vocabulary()
    test_standard_grammar()
    test_generation()
    test_validation()
    test_invalid_sequence()
    test_slot_order()
    test_grammar_size_metrics()
    test_slot_value_repr()
    print("✓ All slot grammar tests passed!")
