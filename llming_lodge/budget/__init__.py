from .budget_types import LimitPeriod, InsufficientBudgetError, TokenUsage, BudgetInfo, BudgetHandler
from .budget_limit import BudgetLimit
from .memory_budget_limit import MemoryBudgetLimit
from .budget_manager import LLMBudgetManager


def __getattr__(name):
    """Lazy imports for optional dependencies."""
    if name == "MongoDBBudgetLimit":
        from .mongodb_budget_limit import MongoDBBudgetLimit
        return MongoDBBudgetLimit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'LimitPeriod',
    'InsufficientBudgetError',
    'TokenUsage',
    'BudgetLimit',
    'MemoryBudgetLimit',
    'MongoDBBudgetLimit',
    'LLMBudgetManager',
    'BudgetInfo',
    'BudgetHandler',
]
