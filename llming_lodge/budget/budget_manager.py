import logging
import os
import time as _time
from typing import List, Dict

from .budget_limit import BudgetLimit
from .budget_types import InsufficientBudgetError

logger = logging.getLogger(__name__)


class LLMBudgetManager:
    """
    Manages multiple budget limits in Euros for LLM operations.
    
    This class handles budget reservation and tracking for LLM operations,
    ensuring that operations don't exceed allocated monetary limits.
    Budget is tracked in Euros and automatically scales with model-specific costs.
    """
    
    def __init__(self, limits: List[BudgetLimit]):
        """
        Initialize the budget manager.
        
        Args:
            limits: List of budget limits to enforce
        """
        self.limits = {limit.name: limit for limit in limits}
        logger.debug(f"Initialized budget manager with limits: {[f'{name} ({limit.period.value}): {limit.amount}€' for name, limit in self.limits.items()]}")

    async def available_budget_async(self) -> float:
        """Get the minimum available budget across all limits (async version)."""
        budgets = {name: await limit.get_available_budget_async() for name, limit in self.limits.items()}
        logger.debug(f"Available budgets across limits: {budgets}")
        min_budget = min(budgets.values())
        logger.debug(f"Minimum available budget: {min_budget}€")
        return min_budget

    async def reserve_budget_async(self, *, input_tokens: int, max_output_tokens: int, 
                                 input_token_price: float, output_token_price: float) -> None:
        """
        Reserve budget for an LLM operation (async version).
        
        Args:
            input_tokens: Number of input tokens for the operation
            max_output_tokens: Maximum possible number of output tokens
            input_token_price: Cost per 1M input tokens in Euros
            output_token_price: Cost per 1M output tokens in Euros
            
        Raises:
            InsufficientBudgetError: If there isn't enough budget available
        """
        input_cost = input_tokens * (input_token_price / 1_000_000)  # Convert from per 1M tokens to per token
        max_output_cost = max_output_tokens * (output_token_price / 1_000_000)  # Convert from per 1M tokens to per token
        required_budget = input_cost + max_output_cost
        _pid = os.getpid()
        logger.info("[OP] budget_reserve — starting (%.6f€, in=%d, max_out=%d, pid=%d)", required_budget, input_tokens, max_output_tokens, _pid)
        _t0 = _time.monotonic()

        # Try to reserve from all limits immediately
        reserved_limits = []
        try:
            for name, limit in self.limits.items():
                logger.debug(f"Reserving {required_budget}€ from limit '{name}'")
                success = await limit.reserve_budget_async(required_budget)
                if not success:
                    # Get current available budget to include in error message
                    available = await limit.get_available_budget_async()
                    msg = f"Failed to reserve budget ({required_budget:.4f}€) from limit '{name}' (available: {available:.4f}€)"
                    logger.debug(f"Budget reservation failed: {msg}")
                    # Return budget to all previously reserved limits
                    for reserved_name in reserved_limits:
                        await self.limits[reserved_name].return_budget_async(required_budget)
                    # Always raise InsufficientBudgetError since reserve_budget returned False
                    raise InsufficientBudgetError(msg, limit_name=name)
                reserved_limits.append(name)
                logger.debug(f"Successfully reserved {required_budget}€ from limit '{name}'")
            logger.info("[OP] budget_reserve — done (%.6f€, %.0fms, pid=%d)", required_budget, (_time.monotonic() - _t0) * 1000, _pid)
        except Exception as e:
            # If reservation fails for any limit, return budget to all previously reserved limits
            logger.warning("[OP] budget_reserve — failed (%.6f€, %.0fms, pid=%d): %s", required_budget, (_time.monotonic() - _t0) * 1000, _pid, e)
            for name in reserved_limits:
                await self.limits[name].return_budget_async(required_budget)
            if isinstance(e, InsufficientBudgetError):
                raise e
            # For any other error, raise an InsufficientBudgetError with the first limit
            msg = f"Failed to reserve budget ({required_budget:.4f}€) due to error: {str(e)}"
            raise InsufficientBudgetError(msg, limit_name=next(iter(self.limits.keys())))

    async def return_unused_budget_async(self, *, reserved_output_tokens: int, actual_output_tokens: int,
                                       output_token_price: float) -> None:
        """
        Return unused budget after an operation completes (async version).
        
        Args:
            reserved_output_tokens: Number of output tokens that were reserved
            actual_output_tokens: Actual number of output tokens used
            output_token_price: Cost per 1M output tokens in Euros
        """
        _pid = os.getpid()
        _t0 = _time.monotonic()
        if actual_output_tokens > reserved_output_tokens:
            # more used than expected, further reduce budget
            additional_use = actual_output_tokens - reserved_output_tokens
            additional_budget = additional_use * (output_token_price / 1_000_000)  # Convert from per 1M tokens to per token
            logger.info("[OP] budget_return — overuse, reserving additional %.6f€ (reserved=%d, actual=%d, pid=%d)",
                        additional_budget, reserved_output_tokens, actual_output_tokens, _pid)
            for limit in self.limits.values():
                await limit.reserve_budget_async(additional_budget)
            return

        unused_tokens = reserved_output_tokens - actual_output_tokens
        unused_budget = unused_tokens * (output_token_price / 1_000_000)  # Convert from per 1M tokens to per token

        for limit in self.limits.values():
            await limit.return_budget_async(unused_budget)
        logger.info("[OP] budget_return — done (returned=%.6f€, reserved=%d, actual=%d, %.0fms, pid=%d)",
                    unused_budget, reserved_output_tokens, actual_output_tokens, (_time.monotonic() - _t0) * 1000, _pid)

    async def reset_async(self) -> None:
        """Reset all budget limits (async version)."""
        for limit in self.limits.values():
            await limit.reset_async()
