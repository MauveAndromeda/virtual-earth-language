"""Environment modules for emergent communication."""

from .referential_game import ReferentialGame

try:
    from .gridworld import GridWorldEnv
    from .geography import GeographyModule
except ImportError:
    # Graceful fallback for incomplete modules
    GridWorldEnv = None
    GeographyModule = None

__all__ = ["ReferentialGame"]
if GridWorldEnv is not None:
    __all__.extend(["GridWorldEnv", "GeographyModule"])
