"""Explanation and dual-channel communication modules."""

from explain.ast_parser import (
    EChannelParser,
    ParseResult,
    ASTNode,
    parse_explanation,
    extract_slot_values,
    compare_asts,
    validate_ast
)

from explain.codec import (
    code_from_sem,
    explain_from_sem,
    sem_from_code,
    code_from_explain
)

try:
    from explain.dual_channel import (
        DualChannelMessage,
        DualChannelSystem
    )
except ImportError:
    # Optional import if dependencies not available
    pass

__all__ = [
    'EChannelParser',
    'ParseResult',
    'ASTNode',
    'parse_explanation',
    'extract_slot_values',
    'compare_asts',
    'validate_ast',
    'code_from_sem',
    'explain_from_sem',
    'sem_from_code',
    'code_from_explain'
]
