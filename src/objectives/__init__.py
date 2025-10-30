"""Objective functions and loss computations."""

from objectives.consistency import (
    consistency_score,
    compute_consistency_metrics,
    ConsistencyMetrics,
    string_similarity,
    semantic_similarity,
    levenshtein_distance,
    validate_consistency_threshold,
    batch_consistency_score
)

try:
    from objectives.interpretable_losses import InterpretableLoss
except ImportError:
    pass

__all__ = [
    'consistency_score',
    'compute_consistency_metrics',
    'ConsistencyMetrics',
    'string_similarity',
    'semantic_similarity',
    'levenshtein_distance',
    'validate_consistency_threshold',
    'batch_consistency_score'
]
