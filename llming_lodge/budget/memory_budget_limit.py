from datetime import datetime
import logging
from zoneinfo import ZoneInfo

from .budget_limit import BudgetLimit
from .budget_types import LimitPeriod

logger = logging.getLogger(__name__)


class MemoryBudgetLimit(BudgetLimit):
    """Budget limit with in-memory tracking."""
    def __init__(self, *, name: str, amount: float, period: LimitPeriod, timezone_str: str = "UTC"):
        super().__init__(name=name, amount=amount, period=period, timezone_str=timezone_str)
        self._initial_amount = amount
        self._usage = {} if period != LimitPeriod.TOTAL else None

    def _get_current_time(self) -> datetime:
        """Get current time."""
        return datetime.now(tz=ZoneInfo(self.timezone))

    def get_available_budget(self) -> float:
        """Get available budget for the current period."""
        logger.debug(f"Getting available budget for limit '{self.name}' (period: {self.period.value})")
        
        if self.period == LimitPeriod.TOTAL:
            logger.debug(f"Total limit, returning memory amount: {self.amount}€")
            return self.amount
        
        current_time = self._get_current_time()
        period_key = self._get_key_suffix(current_time)
        
        with self._lock:
            # Clean up old periods and initialize new period atomically
            if period_key not in self._usage:
                # If we have usage data but not for this period, it means we've moved to a new period
                logger.debug(f"New {self.period.value} period detected, initializing usage tracking")
                self._usage[period_key] = 0.0
            
            current_usage = self._usage[period_key]
            available = max(self.amount - current_usage, 0.0)
            logger.debug(f"Memory {self.period.value} limit for {period_key}, used: {current_usage}€, available: {available}€")
            return available

    def reserve_budget(self, amount: float) -> bool:
        """Reserve budget for an operation."""
        logger.debug(f"Attempting to reserve {amount}€ from limit '{self.name}' (period: {self.period.value})")
        if amount > self.amount:
            logger.debug(f"Amount {amount}€ exceeds total budget {self.amount}€ for limit '{self.name}'")
            return False

        if self.period == LimitPeriod.TOTAL:
            with self._lock:
                if amount <= self.amount:
                    self.amount -= amount
                    logger.debug(f"Reserved {amount}€ from memory total budget, remaining: {self.amount}€")
                    return True
                return False

        current_time = self._get_current_time()
        period_key = self._get_key_suffix(current_time)
        
        with self._lock:
            # Check if period changed between available check and now
            new_period_key = self._get_key_suffix(self._get_current_time())
            if new_period_key != period_key:
                logger.debug(f"Period changed during reservation, rechecking budget")
                # Clean up old periods and initialize new period atomically
                self._usage[new_period_key] = 0.0
                # Retry with new period
                return self.reserve_budget(amount)
            
            # Initialize usage for this period if needed
            if period_key not in self._usage:
                self._usage[period_key] = 0.0  # Initialize new period
            
            # Check if we have enough budget for this period
            current_usage = self._usage[period_key]
            if amount <= (self.amount - current_usage):
                try:
                    self._usage[period_key] = current_usage + amount
                    logger.debug(f"Reserved {amount}€ from memory {self.period.value} budget for {period_key}, used: {self._usage[period_key]}€")
                    return True
                except Exception as e:
                    logger.error(f"Failed to reserve budget: {e}")
                    # Rollback the reservation
                    self._usage[period_key] = current_usage
                    return False
            
            logger.debug(f"Insufficient budget for {self.period.value} period {period_key}: {current_usage}€ used + {amount}€ requested > {self.amount}€ limit")
            return False

    def return_budget(self, amount: float) -> None:
        """Return unused budget."""
        if self.period == LimitPeriod.TOTAL:
            with self._lock:
                self.amount += amount
                logger.debug(f"Returned {amount}€ to memory total budget, remaining: {self.amount}€")
            return
        
        current_time = self._get_current_time()
        period_key = self._get_key_suffix(current_time)
        
        with self._lock:
            if period_key not in self._usage:
                self._usage[period_key] = 0.0  # Initialize new period
            self._usage[period_key] = max(0.0, self._usage[period_key] - amount)
            logger.debug(f"Returned {amount}€ to memory {self.period.value} budget for {period_key}, used: {self._usage[period_key]}€")

    def reset(self) -> None:
        """Reset budget to initial amount."""
        with self._lock:
            if self.period == LimitPeriod.TOTAL:
                self.amount = self._initial_amount
            else:
                self._usage = {}
                logger.debug(f"Reset memory {self.period.value} budget usage")

    async def get_available_budget_async(self) -> float:
        """Get available budget for the current period (async version)."""
        return self.get_available_budget()

    async def reserve_budget_async(self, amount: float) -> bool:
        """Reserve budget for an operation (async version)."""
        return self.reserve_budget(amount)

    async def return_budget_async(self, amount: float) -> None:
        """Return unused budget (async version)."""
        self.return_budget(amount)

    async def reset_async(self) -> None:
        """Reset budget to initial amount (async version)."""
        self.reset()
