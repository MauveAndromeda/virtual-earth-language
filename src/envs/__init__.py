"""Interpretability-focused emergent communication modules."""
from .referential_game import ReferentialGame
# TODO: Add InterpretableReferentialGame
try:
    from .interpretable_game import InterpretableReferentialGame
    __all__ = ["ReferentialGame", "InterpretableReferentialGame"]
except ImportError:
    __all__ = ["ReferentialGame"]
