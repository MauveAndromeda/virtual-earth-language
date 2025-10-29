"""
Abstract Syntax Tree (AST) Parser for E-Channel Explanations

This module parses E-Channel explanation text into structured AST representations
for semantic analysis, consistency checking, and interpretability evaluation.

Supported Formats:
1. Structured Format: PLAN(DO(MOVE), TARGET(CIRCLE), AT(LEFT))
2. Keyword-Value Format: NAV(act=MOVE, obj=CIRCLE, attr=RED, loc=L01)
3. Natural Language: "Move to the red circle at location L01"

The AST enables:
- Semantic consistency checking between C-channel and E-channel
- Slot extraction and validation
- Compositional interpretation
- Cross-lingual understanding
"""

import re
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class ASTNodeType(Enum):
    """Types of AST nodes."""
    ROOT = "root"
    FUNCTION = "function"
    SLOT = "slot"
    VALUE = "value"
    KEYWORD_ARG = "keyword_arg"
    NATURAL_LANG = "natural_lang"


@dataclass
class ASTNode:
    """
    Node in the Abstract Syntax Tree.

    Attributes
    ----------
    node_type : ASTNodeType
        Type of this node
    value : str
        Value/content of this node
    children : List[ASTNode]
        Child nodes
    metadata : Dict[str, Any]
        Additional metadata (position, confidence, etc.)
    """
    node_type: ASTNodeType
    value: str
    children: List['ASTNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        children_repr = f", {len(self.children)} children" if self.children else ""
        return f"ASTNode({self.node_type.value}, '{self.value}'{children_repr})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert AST to dictionary representation."""
        return {
            "type": self.node_type.value,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }

    def pretty_print(self, indent: int = 0) -> str:
        """Generate pretty-printed representation of AST."""
        lines = []
        prefix = "  " * indent
        lines.append(f"{prefix}{self.node_type.value}: {self.value}")

        for child in self.children:
            lines.append(child.pretty_print(indent + 1))

        return "\n".join(lines)


@dataclass
class ParseResult:
    """
    Result of parsing E-channel text.

    Attributes
    ----------
    ast : ASTNode
        Root of the abstract syntax tree
    semantics : Dict[str, str]
        Extracted slot-value semantics
    parse_success : bool
        Whether parsing was successful
    error_message : str
        Error message if parsing failed
    confidence : float
        Parsing confidence score (0-1)
    """
    ast: Optional[ASTNode]
    semantics: Dict[str, str]
    parse_success: bool
    error_message: str = ""
    confidence: float = 1.0


class BaseParser(ABC):
    """Abstract base class for E-channel parsers."""

    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """Parse E-channel text into AST."""
        pass

    @abstractmethod
    def can_parse(self, text: str) -> bool:
        """Check if this parser can handle the given text."""
        pass


class StructuredFormatParser(BaseParser):
    """
    Parser for structured format: PLAN(DO(MOVE), TARGET(CIRCLE), AT(LEFT))

    Grammar:
        root := PLAN '(' args ')'
        args := arg (',' arg)*
        arg  := SLOT '(' VALUE ')'
    """

    def __init__(self):
        # Regex patterns for structured format
        self.plan_pattern = re.compile(r'PLAN\s*\((.*)\)\s*$', re.IGNORECASE)
        self.slot_pattern = re.compile(r'(\w+)\s*\(([^)]+)\)')

    def can_parse(self, text: str) -> bool:
        """Check if text matches structured format."""
        return bool(self.plan_pattern.match(text.strip()))

    def parse(self, text: str) -> ParseResult:
        """Parse structured format into AST."""
        text = text.strip()

        try:
            # Match PLAN(...)
            plan_match = self.plan_pattern.match(text)
            if not plan_match:
                return ParseResult(
                    ast=None,
                    semantics={},
                    parse_success=False,
                    error_message="Text does not match PLAN(...) format"
                )

            # Extract inner content
            inner = plan_match.group(1)

            # Parse slot expressions
            root = ASTNode(
                node_type=ASTNodeType.ROOT,
                value="PLAN",
                metadata={"format": "structured"}
            )

            semantics = {}

            # Find all SLOT(VALUE) patterns
            for slot_match in self.slot_pattern.finditer(inner):
                slot_name = slot_match.group(1).upper()
                slot_value = slot_match.group(2).strip()

                # Create slot node
                slot_node = ASTNode(
                    node_type=ASTNodeType.SLOT,
                    value=slot_name
                )

                # Create value node
                value_node = ASTNode(
                    node_type=ASTNodeType.VALUE,
                    value=slot_value
                )

                slot_node.children.append(value_node)
                root.children.append(slot_node)

                # Extract semantics
                # Map slot patterns to standard slot names
                slot_mapping = {
                    "DO": "ACTION",
                    "TARGET": "OBJECT",
                    "WITH": "ATTRIBUTE",
                    "AT": "LOCATION",
                    "MOD": "MODIFIER"
                }
                standard_slot = slot_mapping.get(slot_name, slot_name)
                semantics[standard_slot] = slot_value

            return ParseResult(
                ast=root,
                semantics=semantics,
                parse_success=True,
                confidence=1.0
            )

        except Exception as e:
            return ParseResult(
                ast=None,
                semantics={},
                parse_success=False,
                error_message=f"Parse error: {str(e)}"
            )


class KeywordValueParser(BaseParser):
    """
    Parser for keyword-value format: NAV(act=MOVE, obj=CIRCLE, attr=RED, loc=L01)

    Grammar:
        root := FUNC '(' kwargs ')'
        kwargs := kwarg (',' kwarg)*
        kwarg := KEY '=' VALUE
    """

    def __init__(self):
        self.func_pattern = re.compile(r'(\w+)\s*\((.*)\)\s*$', re.IGNORECASE)
        self.kwarg_pattern = re.compile(r'(\w+)\s*=\s*([^,]+)')

    def can_parse(self, text: str) -> bool:
        """Check if text matches keyword-value format."""
        match = self.func_pattern.match(text.strip())
        if not match:
            return False
        # Check if inner content has key=value pattern
        inner = match.group(2)
        return bool(self.kwarg_pattern.search(inner))

    def parse(self, text: str) -> ParseResult:
        """Parse keyword-value format into AST."""
        text = text.strip()

        try:
            # Match FUNC(...)
            func_match = self.func_pattern.match(text)
            if not func_match:
                return ParseResult(
                    ast=None,
                    semantics={},
                    parse_success=False,
                    error_message="Text does not match FUNC(...) format"
                )

            func_name = func_match.group(1)
            inner = func_match.group(2)

            # Create root node
            root = ASTNode(
                node_type=ASTNodeType.FUNCTION,
                value=func_name,
                metadata={"format": "keyword_value"}
            )

            semantics = {}

            # Parse keyword arguments
            for kwarg_match in self.kwarg_pattern.finditer(inner):
                key = kwarg_match.group(1).strip()
                value = kwarg_match.group(2).strip()

                # Create keyword argument node
                kwarg_node = ASTNode(
                    node_type=ASTNodeType.KEYWORD_ARG,
                    value=key
                )

                # Create value node
                value_node = ASTNode(
                    node_type=ASTNodeType.VALUE,
                    value=value
                )

                kwarg_node.children.append(value_node)
                root.children.append(kwarg_node)

                # Extract semantics
                # Map common abbreviations to standard slot names
                key_mapping = {
                    "act": "ACTION",
                    "action": "ACTION",
                    "obj": "OBJECT",
                    "object": "OBJECT",
                    "attr": "ATTRIBUTE",
                    "attribute": "ATTRIBUTE",
                    "loc": "LOCATION",
                    "location": "LOCATION",
                    "mod": "MODIFIER",
                    "modifier": "MODIFIER"
                }
                standard_key = key_mapping.get(key.lower(), key.upper())
                semantics[standard_key] = value

            return ParseResult(
                ast=root,
                semantics=semantics,
                parse_success=True,
                confidence=1.0
            )

        except Exception as e:
            return ParseResult(
                ast=None,
                semantics={},
                parse_success=False,
                error_message=f"Parse error: {str(e)}"
            )


class NaturalLanguageParser(BaseParser):
    """
    Parser for natural language format with pattern matching.

    Patterns:
    - "Move to the red circle at location L01"
    - "Take the blue square"
    - "Navigate to L01 and pick the red triangle"
    """

    def __init__(self):
        # Common action verbs
        self.action_verbs = {
            "move", "go", "navigate", "walk", "run",
            "take", "pick", "grab", "collect", "get",
            "drop", "place", "put", "release",
            "look", "observe", "scan", "examine"
        }

        # Common objects
        self.object_words = {
            "circle", "square", "triangle", "diamond", "star",
            "box", "sphere", "cube", "cylinder", "agent"
        }

        # Common attributes
        self.attribute_words = {
            "red", "blue", "green", "yellow", "orange", "purple",
            "small", "large", "big", "tiny", "medium",
            "bright", "dark", "shiny"
        }

        # Location patterns
        self.location_pattern = re.compile(r'L\d{2}|location\s+\d+|at\s+(\w+)', re.IGNORECASE)

    def can_parse(self, text: str) -> bool:
        """Check if text appears to be natural language."""
        text_lower = text.lower()
        # If it contains function call syntax, it's not natural language
        if re.search(r'\w+\s*\([^)]*\)', text):
            return False
        # Check if it contains common action verbs
        return any(verb in text_lower for verb in self.action_verbs)

    def parse(self, text: str) -> ParseResult:
        """Parse natural language into AST using pattern matching."""
        text = text.strip()
        text_lower = text.lower()
        words = text_lower.split()

        try:
            # Create root node
            root = ASTNode(
                node_type=ASTNodeType.NATURAL_LANG,
                value=text,
                metadata={"format": "natural_language"}
            )

            semantics = {}
            confidence = 0.5  # Lower confidence for NL parsing

            # Extract action
            for word in words:
                if word in self.action_verbs:
                    semantics["ACTION"] = word.upper()
                    action_node = ASTNode(
                        node_type=ASTNodeType.SLOT,
                        value="ACTION",
                        children=[ASTNode(ASTNodeType.VALUE, word.upper())]
                    )
                    root.children.append(action_node)
                    confidence += 0.2
                    break

            # Extract object
            for word in words:
                if word in self.object_words:
                    semantics["OBJECT"] = word.upper()
                    object_node = ASTNode(
                        node_type=ASTNodeType.SLOT,
                        value="OBJECT",
                        children=[ASTNode(ASTNodeType.VALUE, word.upper())]
                    )
                    root.children.append(object_node)
                    confidence += 0.2
                    break

            # Extract attribute
            for word in words:
                if word in self.attribute_words:
                    semantics["ATTRIBUTE"] = word.upper()
                    attr_node = ASTNode(
                        node_type=ASTNodeType.SLOT,
                        value="ATTRIBUTE",
                        children=[ASTNode(ASTNodeType.VALUE, word.upper())]
                    )
                    root.children.append(attr_node)
                    confidence += 0.1
                    break

            # Extract location
            loc_match = self.location_pattern.search(text)
            if loc_match:
                location = loc_match.group(0).upper()
                semantics["LOCATION"] = location
                loc_node = ASTNode(
                    node_type=ASTNodeType.SLOT,
                    value="LOCATION",
                    children=[ASTNode(ASTNodeType.VALUE, location)]
                )
                root.children.append(loc_node)
                confidence += 0.1

            confidence = min(confidence, 1.0)

            return ParseResult(
                ast=root,
                semantics=semantics,
                parse_success=True,
                confidence=confidence
            )

        except Exception as e:
            return ParseResult(
                ast=None,
                semantics={},
                parse_success=False,
                error_message=f"Parse error: {str(e)}",
                confidence=0.0
            )


class EChannelParser:
    """
    Main E-Channel parser that dispatches to appropriate format parsers.

    Usage:
        parser = EChannelParser()
        result = parser.parse("PLAN(DO(MOVE), TARGET(CIRCLE), AT(LEFT))")
        if result.parse_success:
            print(result.semantics)
            print(result.ast.pretty_print())
    """

    def __init__(self):
        self.parsers = [
            StructuredFormatParser(),
            KeywordValueParser(),
            NaturalLanguageParser()
        ]

    def parse(self, text: str) -> ParseResult:
        """
        Parse E-channel text using the first matching parser.

        Parameters
        ----------
        text : str
            E-channel explanation text

        Returns
        -------
        ParseResult
            Parsing result with AST and extracted semantics
        """
        if not text or not text.strip():
            return ParseResult(
                ast=None,
                semantics={},
                parse_success=False,
                error_message="Empty input text"
            )

        # Try each parser in order
        for parser in self.parsers:
            if parser.can_parse(text):
                result = parser.parse(text)
                if result.parse_success:
                    return result

        # If no parser succeeded, return failure
        return ParseResult(
            ast=None,
            semantics={},
            parse_success=False,
            error_message="No parser could handle the input format"
        )

    def parse_batch(self, texts: List[str]) -> List[ParseResult]:
        """
        Parse multiple E-channel texts.

        Parameters
        ----------
        texts : List[str]
            List of E-channel texts

        Returns
        -------
        List[ParseResult]
            Parsing results for each text
        """
        return [self.parse(text) for text in texts]


def extract_slot_values(ast: ASTNode) -> Dict[str, str]:
    """
    Extract slot-value pairs from AST.

    Parameters
    ----------
    ast : ASTNode
        Root of AST

    Returns
    -------
    Dict[str, str]
        Extracted slot-value semantics
    """
    semantics = {}

    def traverse(node: ASTNode):
        if node.node_type == ASTNodeType.SLOT and node.children:
            slot_name = node.value
            slot_value = node.children[0].value
            semantics[slot_name] = slot_value
        elif node.node_type == ASTNodeType.KEYWORD_ARG and node.children:
            key = node.value.upper()
            value = node.children[0].value
            semantics[key] = value

        for child in node.children:
            traverse(child)

    traverse(ast)
    return semantics


def compare_asts(ast1: ASTNode, ast2: ASTNode) -> float:
    """
    Compare two ASTs for structural similarity.

    Parameters
    ----------
    ast1 : ASTNode
        First AST
    ast2 : ASTNode
        Second AST

    Returns
    -------
    float
        Similarity score (0-1)
    """
    # Extract semantics from both ASTs
    sem1 = extract_slot_values(ast1)
    sem2 = extract_slot_values(ast2)

    if not sem1 or not sem2:
        return 0.0

    # Calculate semantic overlap
    common_keys = set(sem1.keys()) & set(sem2.keys())
    all_keys = set(sem1.keys()) | set(sem2.keys())

    if not all_keys:
        return 0.0

    key_overlap = len(common_keys) / len(all_keys)

    # Calculate value agreement for common keys
    value_agreement = sum(
        1.0 for key in common_keys if sem1[key] == sem2[key]
    ) / max(len(common_keys), 1)

    # Combined score
    similarity = 0.5 * key_overlap + 0.5 * value_agreement

    return similarity


def validate_ast(ast: ASTNode, slot_definitions: Optional[Dict[str, List[str]]] = None) -> Tuple[bool, List[str]]:
    """
    Validate AST against slot definitions.

    Parameters
    ----------
    ast : ASTNode
        AST to validate
    slot_definitions : Dict[str, List[str]], optional
        Valid vocabulary for each slot

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    errors = []

    if ast is None:
        return False, ["AST is None"]

    # Extract semantics
    semantics = extract_slot_values(ast)

    # Check against slot definitions if provided
    if slot_definitions:
        for slot_name, slot_value in semantics.items():
            if slot_name in slot_definitions:
                valid_values = slot_definitions[slot_name]
                if slot_value not in valid_values:
                    errors.append(
                        f"Invalid value '{slot_value}' for slot '{slot_name}'. "
                        f"Expected one of: {valid_values}"
                    )

    is_valid = len(errors) == 0
    return is_valid, errors


# Convenience function
def parse_explanation(text: str) -> ParseResult:
    """
    Convenience function to parse E-channel explanation.

    Parameters
    ----------
    text : str
        E-channel explanation text

    Returns
    -------
    ParseResult
        Parsing result
    """
    parser = EChannelParser()
    return parser.parse(text)
