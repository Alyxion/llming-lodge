from abc import ABC, abstractmethod
from datetime import datetime
import threading
from typing import Optional

from .time_intervals import TimeIntervalHandler
from .budget_types import LimitPeriod


class BudgetLimit(ABC):
    """Base class for budget limits."""
    def __init__(self, *, name: str, amount: float, period: LimitPeriod,
                 interval_value: Optional[int] = None, timezone_str: str = "UTC"):
        self.name = name
        self.amount = amount
        self.period = period
        self.interval_value = interval_value if isinstance(interval_value, int) else None
        self.timezone = timezone_str
        self._initial_amount = amount
        self._lock = threading.Lock()

    def _get_key_suffix(self, time: datetime) -> str:
        """Generate key suffix based on period."""
        return TimeIntervalHandler.get_key_suffix(self.period, time, self.interval_value)

    @abstractmethod
    async def get_available_budget_async(self) -> float:
        """Get available budget for the current period."""
        raise NotImplementedError("Subclasses must implement get_available_budget_async")

    @abstractmethod
    async def reserve_budget_async(self, amount: float) -> bool:
        """Reserve budget for an operation."""
        raise NotImplementedError("Subclasses must implement reserve_budget_async")

    @abstractmethod
    async def return_budget_async(self, amount: float) -> None:
        """Return unused budget."""
        raise NotImplementedError("Subclasses must implement return_budget_async")

    @abstractmethod
    async def reset_async(self) -> None:
        """Reset budget to initial amount."""
        raise NotImplementedError("Subclasses must implement reset_async")

    async def log_usage_async(self, *, model_name: str, tokens_input: int, tokens_output: int, costs: float, duration_ms: Optional[float] = None, user_id: Optional[str] = None, operation_type: Optional[str] = None) -> None:
        """
        Log usage information for a completed request.

        This is an optional method that subclasses can implement to log usage information.
        The default implementation does nothing.
        """
        pass
