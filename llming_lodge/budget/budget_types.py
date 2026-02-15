from typing import Callable, Optional, TypedDict

from .time_intervals import TimeInterval as LimitPeriod


class BudgetInfo(TypedDict, total=False):
    """Budget information returned by a BudgetHandler callback."""
    available: float
    reserved: float


BudgetHandler = Callable[[], BudgetInfo]

class InsufficientBudgetError(Exception):
    """Raised when there is not enough budget available for the requested operation."""
    def __init__(self, message: str, limit_name: str):
        self.limit_name = limit_name
        super().__init__(message)


class TokenUsage:
    """Represents the token usage for an LLM operation."""
    def __init__(self, input_tokens: int, output_tokens: int, 
                 input_cost: float, output_cost: float):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_cost = input_cost
        self.output_cost = output_cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost
