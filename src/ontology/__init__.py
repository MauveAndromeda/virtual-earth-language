"""Ontology modules: slots, grammar, and morphology."""

from ontology.morphology import (
    MorphologyEngine,
    MorphologicalRule,
    MorphemeType,
    apply_morphology,
    analyze_morphology,
    get_default_engine
)

from ontology.slot_grammar import (
    SlotGrammar,
    SlotType,
    SlotValue,
    GrammarRule,
    ParseTree,
    create_standard_grammar,
    parse_slot_sequence
)

from ontology.slots import SLOTS, VOCAB

try:
    from ontology.enhanced_slots import (
        EnhancedSlotSystem,
        SlotDefinition,
        ENHANCED_SLOT_SYSTEM
    )
except ImportError:
    pass

__all__ = [
    'MorphologyEngine',
    'MorphologicalRule',
    'MorphemeType',
    'apply_morphology',
    'analyze_morphology',
    'get_default_engine',
    'SlotGrammar',
    'SlotType',
    'SlotValue',
    'GrammarRule',
    'ParseTree',
    'create_standard_grammar',
    'parse_slot_sequence',
    'SLOTS',
    'VOCAB'
]
