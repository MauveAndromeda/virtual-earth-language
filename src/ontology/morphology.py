"""
Morphological Rules Engine for Interpretable Language Evolution

This module provides a comprehensive morphological system for productive word
formation in slot-structured languages. It supports rule-based transformations,
compositionality, and constraint enforcement.

Key Features:
- Pattern-based morphological rules (affixation, compounding, etc.)
- Rule composition and chaining
- Phonological constraints
- Semantic compositionality checking
- Productivity metrics
- Morphological generation and analysis

Example Usage:
    engine = MorphologyEngine()
    engine.add_rule("ACTION", "(.+)", r"\\1_ING", "progressive")
    result = engine.apply("ACTION", "MOVE", "progressive")
    # Result: "MOVE_ING"
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)


class MorphemeType(Enum):
    """Types of morphemes in the system."""
    ROOT = "root"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    INFIX = "infix"
    COMPOUND = "compound"
    REDUPLICATION = "reduplication"


class RuleApplication(Enum):
    """When to apply morphological rules."""
    ALWAYS = "always"
    CONDITIONAL = "conditional"
    OPTIONAL = "optional"


@dataclass
class MorphologicalRule:
    """
    Morphological transformation rule.

    Attributes
    ----------
    slot_type : str
        Slot category this rule applies to (ACTION, OBJECT, etc.)
    pattern : str
        Regex pattern to match input
    replacement : str
        Replacement pattern (can use capture groups)
    rule_name : str
        Human-readable name for this rule
    morpheme_type : MorphemeType
        Type of morphological operation
    semantic_effect : str
        Semantic transformation this rule encodes
    constraints : List[Callable]
        Phonological/morphological constraints
    productivity : float
        How productive this rule is (0-1)
    application : RuleApplication
        When to apply this rule
    metadata : Dict[str, Any]
        Additional metadata
    """
    slot_type: str
    pattern: str
    replacement: str
    rule_name: str
    morpheme_type: MorphemeType = MorphemeType.SUFFIX
    semantic_effect: str = ""
    constraints: List[Callable] = field(default_factory=list)
    productivity: float = 1.0
    application: RuleApplication = RuleApplication.ALWAYS
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compile regex pattern for efficiency."""
        try:
            self.compiled_pattern = re.compile(self.pattern)
        except re.error as e:
            logger.error(f"Invalid pattern '{self.pattern}': {e}")
            raise ValueError(f"Invalid regex pattern: {self.pattern}")

    def matches(self, word: str) -> bool:
        """Check if this rule can apply to the word."""
        return bool(self.compiled_pattern.match(word))

    def apply(self, word: str) -> Optional[str]:
        """
        Apply this morphological rule to a word.

        Parameters
        ----------
        word : str
            Input word

        Returns
        -------
        Optional[str]
            Transformed word, or None if rule doesn't apply
        """
        if not self.matches(word):
            return None

        # Apply the transformation
        try:
            result = self.compiled_pattern.sub(self.replacement, word)
        except Exception as e:
            logger.error(f"Error applying rule {self.rule_name} to '{word}': {e}")
            return None

        # Check constraints
        for constraint in self.constraints:
            if not constraint(word, result):
                logger.debug(f"Constraint failed for {word} -> {result}")
                return None

        return result


@dataclass
class MorphologicalAnalysis:
    """
    Analysis of a morphologically complex word.

    Attributes
    ----------
    surface_form : str
        The actual word form
    root : str
        Root/stem of the word
    morphemes : List[Tuple[str, MorphemeType]]
        List of morphemes with their types
    rules_applied : List[str]
        Names of rules that were applied
    semantic_composition : Dict[str, str]
        Semantic features contributed by each morpheme
    is_productive : bool
        Whether this is a productive formation
    confidence : float
        Confidence in this analysis (0-1)
    """
    surface_form: str
    root: str
    morphemes: List[Tuple[str, MorphemeType]]
    rules_applied: List[str]
    semantic_composition: Dict[str, str] = field(default_factory=dict)
    is_productive: bool = True
    confidence: float = 1.0


class PhonologicalConstraints:
    """
    Common phonological constraints for morphology.

    These prevent ill-formed outputs.
    """

    @staticmethod
    def no_triple_consonants(input_word: str, output_word: str) -> bool:
        """Prevent three consecutive consonants."""
        consonants = "BCDFGHJKLMNPQRSTVWXYZ"
        for i in range(len(output_word) - 2):
            if all(c in consonants for c in output_word[i:i+3]):
                return False
        return True

    @staticmethod
    def no_double_underscores(input_word: str, output_word: str) -> bool:
        """Prevent double underscores like '__'."""
        return "__" not in output_word

    @staticmethod
    def max_length(max_len: int = 30) -> Callable:
        """Create constraint for maximum word length."""
        def constraint(input_word: str, output_word: str) -> bool:
            return len(output_word) <= max_len
        return constraint

    @staticmethod
    def no_repeated_affixes(input_word: str, output_word: str) -> bool:
        """Prevent repeated affixes like *MOVE_ING_ING."""
        parts = output_word.split("_")
        return len(parts) == len(set(parts))


class MorphologyEngine:
    """
    Morphological rules engine for interpretable language evolution.

    This engine manages morphological rules, applies them to generate
    new word forms, and analyzes existing forms.

    Example:
        engine = MorphologyEngine()
        engine.register_standard_rules()
        word = engine.apply("ACTION", "MOVE", "progressive")
        analysis = engine.analyze(word)
    """

    def __init__(self):
        self.rules: Dict[str, List[MorphologicalRule]] = defaultdict(list)
        self.lexicon: Dict[str, Set[str]] = defaultdict(set)
        self.rule_usage: Dict[str, int] = defaultdict(int)
        self.generated_forms: Dict[str, MorphologicalAnalysis] = {}

    def add_rule(
        self,
        slot_type: str,
        pattern: str,
        replacement: str,
        rule_name: str,
        morpheme_type: MorphemeType = MorphemeType.SUFFIX,
        semantic_effect: str = "",
        constraints: Optional[List[Callable]] = None,
        productivity: float = 1.0
    ) -> None:
        """
        Add a morphological rule to the engine.

        Parameters
        ----------
        slot_type : str
            Slot type this rule applies to
        pattern : str
            Regex pattern to match
        replacement : str
            Replacement pattern
        rule_name : str
            Name for this rule
        morpheme_type : MorphemeType
            Type of morphological operation
        semantic_effect : str
            Semantic transformation
        constraints : List[Callable], optional
            Phonological constraints
        productivity : float
            Productivity score (0-1)
        """
        if constraints is None:
            constraints = [
                PhonologicalConstraints.no_double_underscores,
                PhonologicalConstraints.no_repeated_affixes,
                PhonologicalConstraints.max_length(30)
            ]

        rule = MorphologicalRule(
            slot_type=slot_type,
            pattern=pattern,
            replacement=replacement,
            rule_name=rule_name,
            morpheme_type=morpheme_type,
            semantic_effect=semantic_effect,
            constraints=constraints,
            productivity=productivity
        )

        self.rules[slot_type].append(rule)
        logger.info(f"Added morphological rule: {rule_name} for {slot_type}")

    def register_standard_rules(self) -> None:
        """
        Register standard morphological rules for all slot types.

        These are the default rules inspired by natural language morphology.
        """
        # ACTION slot rules
        self.add_rule(
            "ACTION", r"^(.+)$", r"\1_ING",
            "progressive_aspect",
            MorphemeType.SUFFIX,
            "progressive aspect (ongoing action)"
        )
        self.add_rule(
            "ACTION", r"^(.+)$", r"\1_ED",
            "past_tense",
            MorphemeType.SUFFIX,
            "past tense"
        )
        self.add_rule(
            "ACTION", r"^(.+)$", r"RE_\1",
            "repetitive",
            MorphemeType.PREFIX,
            "repetition or return (do again)"
        )
        self.add_rule(
            "ACTION", r"^(.+)$", r"\1_ABLE",
            "capability",
            MorphemeType.SUFFIX,
            "ability or possibility"
        )
        self.add_rule(
            "ACTION", r"^(.+)$", r"UN_\1",
            "reverse_action",
            MorphemeType.PREFIX,
            "reverse or undo action"
        )

        # OBJECT slot rules
        self.add_rule(
            "OBJECT", r"^(.+)$", r"\1_S",
            "plural",
            MorphemeType.SUFFIX,
            "plural (multiple objects)"
        )
        self.add_rule(
            "OBJECT", r"^(.+)$", r"MINI_\1",
            "diminutive",
            MorphemeType.PREFIX,
            "smaller version"
        )
        self.add_rule(
            "OBJECT", r"^(.+)$", r"MEGA_\1",
            "augmentative",
            MorphemeType.PREFIX,
            "larger version"
        )

        # ATTRIBUTE slot rules
        self.add_rule(
            "ATTRIBUTE", r"^(.+)$", r"\1_ER",
            "comparative",
            MorphemeType.SUFFIX,
            "comparative degree (more X)"
        )
        self.add_rule(
            "ATTRIBUTE", r"^(.+)$", r"\1_EST",
            "superlative",
            MorphemeType.SUFFIX,
            "superlative degree (most X)"
        )
        self.add_rule(
            "ATTRIBUTE", r"^(.+)$", r"UN_\1",
            "negation",
            MorphemeType.PREFIX,
            "opposite or absence of quality"
        )
        self.add_rule(
            "ATTRIBUTE", r"^(.+)$", r"SEMI_\1",
            "partial",
            MorphemeType.PREFIX,
            "partial quality"
        )

        # LOCATION slot rules
        self.add_rule(
            "LOCATION", r"^(.+)$", r"\1_WARD",
            "direction",
            MorphemeType.SUFFIX,
            "direction towards"
        )
        self.add_rule(
            "LOCATION", r"^(.+)$", r"\1_SIDE",
            "area",
            MorphemeType.SUFFIX,
            "area or region"
        )
        self.add_rule(
            "LOCATION", r"^(.+)$", r"NEAR_\1",
            "proximity",
            MorphemeType.PREFIX,
            "close to location"
        )

        # MODIFIER slot rules
        self.add_rule(
            "MODIFIER", r"^(.+)$", r"VERY_\1",
            "intensification",
            MorphemeType.PREFIX,
            "intensified modifier"
        )

        logger.info("Standard morphological rules registered")

    def apply(
        self,
        slot_type: str,
        word: str,
        rule_name: str
    ) -> Optional[str]:
        """
        Apply a specific morphological rule to a word.

        Parameters
        ----------
        slot_type : str
            Slot type
        word : str
            Input word
        rule_name : str
            Name of rule to apply

        Returns
        -------
        Optional[str]
            Transformed word, or None if rule doesn't apply
        """
        if slot_type not in self.rules:
            logger.warning(f"No rules defined for slot type: {slot_type}")
            return None

        # Find the rule
        rule = None
        for r in self.rules[slot_type]:
            if r.rule_name == rule_name:
                rule = r
                break

        if rule is None:
            logger.warning(f"Rule '{rule_name}' not found for {slot_type}")
            return None

        # Apply the rule
        result = rule.apply(word)

        if result is not None:
            # Track usage
            self.rule_usage[rule_name] += 1

            # Store analysis
            analysis = MorphologicalAnalysis(
                surface_form=result,
                root=word,
                morphemes=[(word, MorphemeType.ROOT), (rule_name, rule.morpheme_type)],
                rules_applied=[rule_name],
                semantic_composition={
                    "root_meaning": word,
                    rule_name: rule.semantic_effect
                },
                is_productive=True,
                confidence=rule.productivity
            )
            self.generated_forms[result] = analysis

        return result

    def apply_all(
        self,
        slot_type: str,
        word: str
    ) -> Dict[str, str]:
        """
        Apply all applicable rules to generate all possible forms.

        Parameters
        ----------
        slot_type : str
            Slot type
        word : str
            Input word

        Returns
        -------
        Dict[str, str]
            Mapping from rule_name to generated form
        """
        results = {}

        if slot_type not in self.rules:
            return results

        for rule in self.rules[slot_type]:
            result = rule.apply(word)
            if result is not None:
                results[rule.rule_name] = result
                self.rule_usage[rule.rule_name] += 1

        return results

    def analyze(self, word: str) -> Optional[MorphologicalAnalysis]:
        """
        Analyze a word to identify its morphological structure.

        Parameters
        ----------
        word : str
            Word to analyze

        Returns
        -------
        Optional[MorphologicalAnalysis]
            Analysis result, or None if word is not analyzable
        """
        # Check if we have cached analysis
        if word in self.generated_forms:
            return self.generated_forms[word]

        # Try to decompose the word
        # Check for common patterns
        analyses = []

        # Check for suffixes
        suffix_patterns = [
            (r"^(.+)_(ING|ED|ABLE|ER|EST|S|WARD|SIDE)$", MorphemeType.SUFFIX),
            (r"^(RE|UN|MINI|MEGA|NEAR|VERY|SEMI)_(.+)$", MorphemeType.PREFIX),
        ]

        for pattern, morph_type in suffix_patterns:
            match = re.match(pattern, word)
            if match:
                groups = match.groups()
                if morph_type == MorphemeType.SUFFIX:
                    root = groups[0]
                    affix = groups[1]
                else:  # PREFIX
                    affix = groups[0]
                    root = groups[1]

                analysis = MorphologicalAnalysis(
                    surface_form=word,
                    root=root,
                    morphemes=[(root, MorphemeType.ROOT), (affix, morph_type)],
                    rules_applied=[f"reverse_engineered_{morph_type.value}"],
                    is_productive=True,
                    confidence=0.8
                )
                analyses.append(analysis)

        # Return best analysis
        if analyses:
            return analyses[0]

        # If no decomposition found, treat as monomorphemic
        return MorphologicalAnalysis(
            surface_form=word,
            root=word,
            morphemes=[(word, MorphemeType.ROOT)],
            rules_applied=[],
            is_productive=False,
            confidence=1.0
        )

    def generate_paradigm(
        self,
        slot_type: str,
        word: str
    ) -> Dict[str, str]:
        """
        Generate full morphological paradigm for a word.

        Parameters
        ----------
        slot_type : str
            Slot type
        word : str
            Base word

        Returns
        -------
        Dict[str, str]
            Complete paradigm (rule_name -> generated_form)
        """
        return self.apply_all(slot_type, word)

    def compute_productivity(
        self,
        slot_type: str,
        rule_name: str,
        lexicon: Set[str]
    ) -> float:
        """
        Compute productivity of a morphological rule.

        Productivity = (# of forms using this rule) / (# of possible base forms)

        Parameters
        ----------
        slot_type : str
            Slot type
        rule_name : str
            Rule name
        lexicon : Set[str]
            Set of base forms

        Returns
        -------
        float
            Productivity score (0-1)
        """
        if not lexicon:
            return 0.0

        # Find the rule
        rule = None
        for r in self.rules[slot_type]:
            if r.rule_name == rule_name:
                rule = r
                break

        if rule is None:
            return 0.0

        # Count how many base forms this rule can apply to
        applicable_count = sum(1 for word in lexicon if rule.matches(word))

        return applicable_count / len(lexicon) if lexicon else 0.0

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about rule usage and productivity.

        Returns
        -------
        Dict[str, Any]
            Statistics including usage counts, productivity, etc.
        """
        stats = {
            "total_rules": sum(len(rules) for rules in self.rules.values()),
            "rules_by_slot": {slot: len(rules) for slot, rules in self.rules.items()},
            "rule_usage": dict(self.rule_usage),
            "generated_forms": len(self.generated_forms),
            "most_productive_rules": sorted(
                self.rule_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        return stats

    def validate_form(
        self,
        word: str,
        expected_slot: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate a morphologically complex form.

        Parameters
        ----------
        word : str
            Word to validate
        expected_slot : str
            Expected slot type

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        errors = []

        # Analyze the word
        analysis = self.analyze(word)

        if analysis is None:
            errors.append("Unable to analyze word")
            return False, errors

        # Check if it's a valid formation
        if not analysis.is_productive and len(analysis.rules_applied) > 0:
            errors.append("Non-productive formation")

        # Check constraints
        if len(word) > 50:
            errors.append("Word exceeds maximum length")

        if "__" in word:
            errors.append("Contains double underscores")

        is_valid = len(errors) == 0
        return is_valid, errors


# Pre-configured engine instance
DEFAULT_ENGINE = None


def get_default_engine() -> MorphologyEngine:
    """Get the default morphology engine with standard rules."""
    global DEFAULT_ENGINE
    if DEFAULT_ENGINE is None:
        DEFAULT_ENGINE = MorphologyEngine()
        DEFAULT_ENGINE.register_standard_rules()
    return DEFAULT_ENGINE


# Convenience functions
def apply_morphology(slot_type: str, word: str, rule_name: str) -> Optional[str]:
    """Convenience function to apply morphology using default engine."""
    engine = get_default_engine()
    return engine.apply(slot_type, word, rule_name)


def analyze_morphology(word: str) -> Optional[MorphologicalAnalysis]:
    """Convenience function to analyze morphology using default engine."""
    engine = get_default_engine()
    return engine.analyze(word)
