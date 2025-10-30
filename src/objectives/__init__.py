"""Objective functions and loss computations."""

# Core consistency functions (may require numpy)
try:
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
    _CONSISTENCY_AVAILABLE = True
except ImportError as e:
    _CONSISTENCY_AVAILABLE = False
    # Provide minimal fallback
    def consistency_score(*args, **kwargs):
        """Fallback when numpy not available."""
        return 0.5

try:
    from objectives.interpretable_losses import InterpretableLoss
    _LOSSES_AVAILABLE = True
except ImportError:
    _LOSSES_AVAILABLE = False

__all__ = [
    'consistency_score',
]

# Only export if available
if _CONSISTENCY_AVAILABLE:
    __all__.extend([
        'compute_consistency_metrics',
        'ConsistencyMetrics',
        'string_similarity',
        'semantic_similarity',
        'levenshtein_distance',
        'validate_consistency_threshold',
        'batch_consistency_score'
    ])
