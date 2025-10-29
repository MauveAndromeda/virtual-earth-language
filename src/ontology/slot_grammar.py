"""
Formal Slot Grammar System for Interpretable Language Evolution

This module implements a formal grammar for slot-structured languages,
providing parsing, generation, and validation capabilities.

Key Features:
- Context-free grammar for slot sequences
- Compositional semantics
- Grammar validation and well-formedness checking
- Probabilistic slot sequence generation
- Syntactic constraints enforcement
- Grammar induction from examples

Grammar Formalism:
    MESSAGE := SLOT*
    SLOT := <SLOT_TYPE, VALUE>
    SLOT_TYPE := ACTION | OBJECT | ATTRIBUTE | LOCATION | MODIFIER
    VALUE := WORD | MORPHEME+

Example:
    "MOVE CIRCLE RED LEFT" parses to:
    [ACTION:MOVE, OBJECT:CIRCLE, ATTRIBUTE:RED, LOCATION:LEFT]
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import random


logger = logging.getLogger(__name__)


class SlotType(Enum):
    """Types of semantic slots in the grammar."""
    ACTION = "ACTION"
    OBJECT = "OBJECT"
    ATTRIBUTE = "ATTRIBUTE"
    LOCATION = "LOCATION"
    MODIFIER = "MODIFIER"
    WILDCARD = "*"  # Matches any slot


@dataclass
class SlotValue:
    """
    A filled slot with type and value.

    Attributes
    ----------
    slot_type : SlotType
        Type of this slot
    value : str
        Lexical value filling this slot
    morphology : List[str]
        Morphological features (e.g., ["PLURAL", "PAST"])
    confidence : float
        Confidence in this slot assignment (0-1)
    """
    slot_type: SlotType
    value: str
    morphology: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def __repr__(self) -> str:
        morph_str = f"+{'+'.join(self.morphology)}" if self.morphology else ""
        return f"{self.slot_type.value}:{self.value}{morph_str}"


@dataclass
class GrammarRule:
    """
    A production rule in the slot grammar.

    Attributes
    ----------
    lhs : str
        Left-hand side (non-terminal)
    rhs : List[Union[SlotType, str]]
        Right-hand side (sequence of slots or terminals)
    constraints : List[Callable]
        Semantic constraints on this rule
    probability : float
        Probability of this production (for generation)
    """
    lhs: str
    rhs: List[Union[SlotType, str]]
    constraints: List = field(default_factory=list)
    probability: float = 1.0

    def __repr__(self) -> str:
        rhs_str = " ".join(str(s) for s in self.rhs)
        return f"{self.lhs} ’ {rhs_str} (p={self.probability:.2f})"


@dataclass
class ParseTree:
    """
    Parse tree for a slot sequence.

    Attributes
    ----------
    root : str
        Root symbol (usually "MESSAGE")
    children : List[Union[ParseTree, SlotValue]]
        Child nodes
    rule : Optional[GrammarRule]
        Production rule used
    """
    root: str
    children: List[Union['ParseTree', SlotValue]] = field(default_factory=list)
    rule: Optional[GrammarRule] = None

    def to_string(self, indent: int = 0) -> str:
        """Pretty-print the parse tree."""
        lines = []
        prefix = "  " * indent
        lines.append(f"{prefix}{self.root}")
        for child in self.children:
            if isinstance(child, ParseTree):
                lines.append(child.to_string(indent + 1))
            else:
                lines.append(f"{prefix}  {child}")
        return "\n".join(lines)

    def get_slot_sequence(self) -> List[SlotValue]:
        """Extract flat slot sequence from parse tree."""
        slots = []
        for child in self.children:
            if isinstance(child, SlotValue):
                slots.append(child)
            elif isinstance(child, ParseTree):
                slots.extend(child.get_slot_sequence())
        return slots


class SlotGrammar:
    """
    Formal grammar for slot-structured languages.

    This implements a context-free grammar specifically designed for
    slot-based semantic representations.

    Example:
        grammar = SlotGrammar()
        grammar.add_standard_rules()
        result = grammar.parse(["MOVE", "CIRCLE", "RED", "LEFT"])
        print(result.to_string())
    """

    def __init__(self, start_symbol: str = "MESSAGE"):
        self.start_symbol = start_symbol
        self.rules: Dict[str, List[GrammarRule]] = defaultdict(list)
        self.slot_vocabularies: Dict[SlotType, Set[str]] = defaultdict(set)
        self.slot_order: List[SlotType] = [
            SlotType.ACTION,
            SlotType.OBJECT,
            SlotType.ATTRIBUTE,
            SlotType.LOCATION,
            SlotType.MODIFIER
        ]

    def add_rule(
        self,
        lhs: str,
        rhs: List[Union[SlotType, str]],
        probability: float = 1.0,
        constraints: Optional[List] = None
    ) -> None:
        """
        Add a production rule to the grammar.

        Parameters
        ----------
        lhs : str
            Left-hand side non-terminal
        rhs : List[Union[SlotType, str]]
            Right-hand side
        probability : float
            Production probability
        constraints : List, optional
            Semantic constraints
        """
        rule = GrammarRule(
            lhs=lhs,
            rhs=rhs,
            probability=probability,
            constraints=constraints or []
        )
        self.rules[lhs].append(rule)
        logger.debug(f"Added grammar rule: {rule}")

    def add_vocabulary(self, slot_type: SlotType, words: List[str]) -> None:
        """
        Add vocabulary for a slot type.

        Parameters
        ----------
        slot_type : SlotType
            Slot type
        words : List[str]
            Words that can fill this slot
        """
        self.slot_vocabularies[slot_type].update(words)
        logger.info(f"Added {len(words)} words to {slot_type.value} vocabulary")

    def add_standard_rules(self) -> None:
        """
        Add standard slot grammar rules.

        Standard grammar:
            MESSAGE ’ CLAUSE
            CLAUSE ’ ACTION_PHRASE OBJECT_PHRASE? LOC_PHRASE?
            ACTION_PHRASE ’ MODIFIER? ACTION
            OBJECT_PHRASE ’ ATTRIBUTE* OBJECT
            LOC_PHRASE ’ LOCATION MODIFIER?
        """
        # Top-level rules
        self.add_rule("MESSAGE", ["CLAUSE"])
        self.add_rule("MESSAGE", ["CLAUSE", "CLAUSE"], probability=0.3)

        # Clause structure
        self.add_rule("CLAUSE", ["ACTION_PHRASE"], probability=0.3)
        self.add_rule("CLAUSE", ["ACTION_PHRASE", "OBJECT_PHRASE"], probability=0.5)
        self.add_rule("CLAUSE", ["ACTION_PHRASE", "OBJECT_PHRASE", "LOC_PHRASE"], probability=0.2)

        # Phrase structures
        self.add_rule("ACTION_PHRASE", [SlotType.ACTION])
        self.add_rule("ACTION_PHRASE", [SlotType.MODIFIER, SlotType.ACTION], probability=0.3)

        self.add_rule("OBJECT_PHRASE", [SlotType.OBJECT])
        self.add_rule("OBJECT_PHRASE", [SlotType.ATTRIBUTE, SlotType.OBJECT], probability=0.6)

        self.add_rule("LOC_PHRASE", [SlotType.LOCATION])
        self.add_rule("LOC_PHRASE", [SlotType.LOCATION, SlotType.MODIFIER], probability=0.2)

        logger.info("Standard slot grammar rules added")

    def parse(
        self,
        tokens: List[str],
        slot_assignments: Optional[Dict[int, SlotType]] = None
    ) -> Optional[ParseTree]:
        """
        Parse a token sequence into a parse tree.

        Parameters
        ----------
        tokens : List[str]
            Input token sequence
        slot_assignments : Dict[int, SlotType], optional
            Pre-assigned slot types for tokens. If None, will auto-assign.

        Returns
        -------
        Optional[ParseTree]
            Parse tree if successful, None otherwise
        """
        if not tokens:
            return None

        # Auto-assign slots if not provided
        if slot_assignments is None:
            slot_assignments = self._infer_slot_assignments(tokens)

        # Create slot values
        slot_values = []
        for i, token in enumerate(tokens):
            if i in slot_assignments:
                slot_values.append(SlotValue(
                    slot_type=slot_assignments[i],
                    value=token
                ))
            else:
                logger.warning(f"No slot assignment for token '{token}' at position {i}")
                return None

        # Build parse tree using chart parsing
        tree = self._chart_parse(slot_values)
        return tree

    def _infer_slot_assignments(self, tokens: List[str]) -> Dict[int, SlotType]:
        """
        Infer slot types for tokens based on vocabulary.

        Parameters
        ----------
        tokens : List[str]
            Input tokens

        Returns
        -------
        Dict[int, SlotType]
            Mapping from token index to slot type
        """
        assignments = {}

        for i, token in enumerate(tokens):
            # Check which slot type this token belongs to
            assigned = False
            for slot_type, vocab in self.slot_vocabularies.items():
                if token in vocab:
                    assignments[i] = slot_type
                    assigned = True
                    break

            if not assigned:
                # Use positional heuristics based on standard order
                if i < len(self.slot_order):
                    assignments[i] = self.slot_order[i]
                    logger.debug(f"Assigned token '{token}' to {self.slot_order[i].value} by position")

        return assignments

    def _chart_parse(self, slot_values: List[SlotValue]) -> Optional[ParseTree]:
        """
        Chart parsing algorithm for slot sequences.

        This is a simplified CKY-style parser adapted for slot grammars.

        Parameters
        ----------
        slot_values : List[SlotValue]
            Sequence of slot values

        Returns
        -------
        Optional[ParseTree]
            Parse tree if successful
        """
        n = len(slot_values)
        if n == 0:
            return None

        # Simple recursive descent for now (can be optimized with dynamic programming)
        root = ParseTree(root=self.start_symbol)

        # Try to match MESSAGE rules
        for rule in self.rules[self.start_symbol]:
            subtree = self._match_rule(rule, slot_values, 0)
            if subtree is not None:
                root.children.append(subtree)
                root.rule = rule
                return root

        # If no rule matches, return flat structure
        logger.warning("No grammar rule matched, returning flat structure")
        root.children = slot_values
        return root

    def _match_rule(
        self,
        rule: GrammarRule,
        slot_values: List[SlotValue],
        start_pos: int
    ) -> Optional[ParseTree]:
        """
        Try to match a grammar rule starting at a position.

        Parameters
        ----------
        rule : GrammarRule
            Grammar rule to match
        slot_values : List[SlotValue]
            Slot sequence
        start_pos : int
            Starting position

        Returns
        -------
        Optional[ParseTree]
            Subtree if match successful
        """
        tree = ParseTree(root=rule.lhs, rule=rule)
        pos = start_pos

        for rhs_item in rule.rhs:
            if isinstance(rhs_item, SlotType):
                # Terminal: match against slot value
                if pos >= len(slot_values):
                    return None
                if slot_values[pos].slot_type != rhs_item:
                    # Wildcard match
                    if rhs_item != SlotType.WILDCARD:
                        return None
                tree.children.append(slot_values[pos])
                pos += 1
            else:
                # Non-terminal: recursively match
                matched = False
                for sub_rule in self.rules.get(rhs_item, []):
                    subtree = self._match_rule(sub_rule, slot_values, pos)
                    if subtree is not None:
                        tree.children.append(subtree)
                        pos += len(subtree.get_slot_sequence())
                        matched = True
                        break
                if not matched:
                    return None

        return tree

    def generate(
        self,
        max_depth: int = 5,
        start_symbol: Optional[str] = None
    ) -> List[SlotValue]:
        """
        Generate a random valid slot sequence from the grammar.

        Parameters
        ----------
        max_depth : int
            Maximum derivation depth
        start_symbol : str, optional
            Symbol to start derivation from

        Returns
        -------
        List[SlotValue]
            Generated slot sequence
        """
        if start_symbol is None:
            start_symbol = self.start_symbol

        return self._generate_recursive(start_symbol, max_depth)

    def _generate_recursive(
        self,
        symbol: str,
        depth: int
    ) -> List[SlotValue]:
        """
        Recursively generate from a symbol.

        Parameters
        ----------
        symbol : str
            Non-terminal or SlotType
        depth : int
            Remaining depth

        Returns
        -------
        List[SlotValue]
            Generated slots
        """
        if depth <= 0:
            return []

        # If symbol is a SlotType, generate a terminal
        try:
            slot_type = SlotType(symbol)
            if slot_type in self.slot_vocabularies:
                vocab = list(self.slot_vocabularies[slot_type])
                if vocab:
                    value = random.choice(vocab)
                    return [SlotValue(slot_type=slot_type, value=value)]
            return []
        except ValueError:
            pass

        # Otherwise, it's a non-terminal
        rules = self.rules.get(symbol, [])
        if not rules:
            return []

        # Select rule probabilistically
        total_prob = sum(r.probability for r in rules)
        rand = random.uniform(0, total_prob)
        cumulative = 0.0

        selected_rule = rules[0]
        for rule in rules:
            cumulative += rule.probability
            if rand <= cumulative:
                selected_rule = rule
                break

        # Expand RHS
        result = []
        for rhs_item in selected_rule.rhs:
            if isinstance(rhs_item, SlotType):
                # Generate terminal
                if rhs_item in self.slot_vocabularies:
                    vocab = list(self.slot_vocabularies[rhs_item])
                    if vocab:
                        value = random.choice(vocab)
                        result.append(SlotValue(slot_type=rhs_item, value=value))
            else:
                # Recursively generate non-terminal
                result.extend(self._generate_recursive(rhs_item, depth - 1))

        return result

    def validate(self, slot_sequence: List[SlotValue]) -> Tuple[bool, List[str]]:
        """
        Validate a slot sequence against the grammar.

        Parameters
        ----------
        slot_sequence : List[SlotValue]
            Sequence to validate

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        errors = []

        if not slot_sequence:
            errors.append("Empty slot sequence")
            return False, errors

        # Check vocabulary
        for slot in slot_sequence:
            if slot.slot_type in self.slot_vocabularies:
                vocab = self.slot_vocabularies[slot.slot_type]
                if slot.value not in vocab:
                    errors.append(
                        f"Unknown word '{slot.value}' for slot {slot.slot_type.value}"
                    )

        # Check if sequence can be parsed
        tree = self._chart_parse(slot_sequence)
        if tree is None:
            errors.append("Cannot parse sequence with grammar rules")

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_slot_order(self) -> List[SlotType]:
        """Get the canonical slot order."""
        return self.slot_order.copy()

    def set_slot_order(self, order: List[SlotType]) -> None:
        """Set the canonical slot order."""
        self.slot_order = order
        logger.info(f"Slot order set to: {[s.value for s in order]}")

    def compute_grammar_size(self) -> Dict[str, int]:
        """
        Compute metrics about grammar size.

        Returns
        -------
        Dict[str, int]
            Grammar size metrics
        """
        return {
            "num_rules": sum(len(rules) for rules in self.rules.values()),
            "num_nonterminals": len(self.rules),
            "vocab_sizes": {
                slot.value: len(vocab)
                for slot, vocab in self.slot_vocabularies.items()
            },
            "total_vocab": sum(len(v) for v in self.slot_vocabularies.values())
        }

    def induce_from_examples(
        self,
        examples: List[List[SlotValue]]
    ) -> None:
        """
        Induce grammar rules from example sequences.

        This performs simple grammar induction by identifying common patterns.

        Parameters
        ----------
        examples : List[List[SlotValue]]
            Example slot sequences
        """
        logger.info(f"Inducing grammar from {len(examples)} examples")

        # Extract slot type sequences
        patterns = defaultdict(int)
        for example in examples:
            pattern = tuple(slot.slot_type for slot in example)
            patterns[pattern] += 1

        # Convert frequent patterns to rules
        total_count = sum(patterns.values())
        for pattern, count in patterns.items():
            if count >= 2:  # Only keep patterns that occur at least twice
                probability = count / total_count
                self.add_rule(
                    self.start_symbol,
                    list(pattern),
                    probability=probability
                )

        logger.info(f"Induced {len(patterns)} grammar patterns")


# Convenience functions
def parse_slot_sequence(tokens: List[str], grammar: Optional[SlotGrammar] = None) -> Optional[ParseTree]:
    """
    Convenience function to parse a token sequence.

    Parameters
    ----------
    tokens : List[str]
        Input tokens
    grammar : SlotGrammar, optional
        Grammar to use (default creates standard grammar)

    Returns
    -------
    Optional[ParseTree]
        Parse tree
    """
    if grammar is None:
        grammar = SlotGrammar()
        grammar.add_standard_rules()

    return grammar.parse(tokens)


def create_standard_grammar() -> SlotGrammar:
    """
    Create a standard slot grammar with default rules.

    Returns
    -------
    SlotGrammar
        Configured grammar instance
    """
    grammar = SlotGrammar()
    grammar.add_standard_rules()

    # Add standard vocabularies
    grammar.add_vocabulary(SlotType.ACTION, [
        "MOVE", "TAKE", "DROP", "GIVE", "LOOK", "SCAN", "WAIT",
        "NAVIGATE", "SEARCH", "COLLECT", "PLACE"
    ])
    grammar.add_vocabulary(SlotType.OBJECT, [
        "CIRCLE", "SQUARE", "TRIANGLE", "DIAMOND", "STAR",
        "BOX", "SPHERE", "CUBE", "AGENT", "MARKER"
    ])
    grammar.add_vocabulary(SlotType.ATTRIBUTE, [
        "RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE",
        "SMALL", "LARGE", "BRIGHT", "DARK"
    ])
    grammar.add_vocabulary(SlotType.LOCATION, [
        "LEFT", "RIGHT", "UP", "DOWN", "CENTER", "TOP", "BOTTOM",
        "NEAR", "FAR", "INSIDE", "OUTSIDE"
    ])
    grammar.add_vocabulary(SlotType.MODIFIER, [
        "NOT", "VERY", "SLIGHTLY", "QUICKLY", "SLOWLY", "CAREFULLY"
    ])

    return grammar
