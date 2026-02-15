import logging
from typing import List, Dict
import threading

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
        self._lock = threading.Lock()
        
    @property
    def available_budget(self) -> float:
        """Get the minimum available budget across all limits."""
        budgets = {name: limit.get_available_budget() for name, limit in self.limits.items()}
        logger.debug(f"Available budgets across limits: {budgets}")
        min_budget = min(budgets.values())
        logger.debug(f"Minimum available budget: {min_budget}€")
        return min_budget

    async def available_budget_async(self) -> float:
        """Get the minimum available budget across all limits (async version)."""
        budgets = {name: await limit.get_available_budget_async() for name, limit in self.limits.items()}
        logger.debug(f"Available budgets across limits: {budgets}")
        min_budget = min(budgets.values())
        logger.debug(f"Minimum available budget: {min_budget}€")
        return min_budget

    def reserve_budget(self, *, input_tokens: int, max_output_tokens: int, 
                      input_token_price: float, output_token_price: float) -> None:
        """
        Reserve budget for an LLM operation.
        
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
        logger.debug(f"Attempting to reserve {required_budget}€ ({input_tokens} input tokens at {input_token_price}€/1M tokens + {max_output_tokens} output tokens at {output_token_price}€/1M tokens)")
        
        # Try to reserve from all limits immediately
        reserved_limits = []
        try:
            for name, limit in self.limits.items():
                logger.debug(f"Reserving {required_budget}€ from limit '{name}'")
                success = limit.reserve_budget(required_budget)
                if not success:
                    # Get current available budget to include in error message
                    available = limit.get_available_budget()
                    msg = f"Failed to reserve budget ({required_budget:.4f}€) from limit '{name}' (available: {available:.4f}€)"
                    logger.debug(f"Budget reservation failed: {msg}")
                    # Return budget to all previously reserved limits
                    for reserved_name in reserved_limits:
                        self.limits[reserved_name].return_budget(required_budget)
                    # Always raise InsufficientBudgetError since reserve_budget returned False
                    raise InsufficientBudgetError(msg, limit_name=name)
                reserved_limits.append(name)
                logger.debug(f"Successfully reserved {required_budget}€ from limit '{name}'")
        except Exception as e:
            # If reservation fails for any limit, return budget to all previously reserved limits
            logger.debug("Reservation failed, returning budget to previously reserved limits")
            for name in reserved_limits:
                self.limits[name].return_budget(required_budget)
            if isinstance(e, InsufficientBudgetError):
                raise e
            # For any other error, raise an InsufficientBudgetError with the first limit
            msg = f"Failed to reserve budget ({required_budget:.4f}€) due to error: {str(e)}"
            logger.debug(f"Budget reservation failed: {msg}")
            raise InsufficientBudgetError(msg, limit_name=next(iter(self.limits.keys())))

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
        logger.debug(f"Attempting to reserve {required_budget}€ ({input_tokens} input tokens at {input_token_price}€/1M tokens + {max_output_tokens} output tokens at {output_token_price}€/1M tokens)")
        
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
        except Exception as e:
            # If reservation fails for any limit, return budget to all previously reserved limits
            logger.debug("Reservation failed, returning budget to previously reserved limits")
            for name in reserved_limits:
                await self.limits[name].return_budget_async(required_budget)
            if isinstance(e, InsufficientBudgetError):
                raise e
            # For any other error, raise an InsufficientBudgetError with the first limit
            msg = f"Failed to reserve budget ({required_budget:.4f}€) due to error: {str(e)}"
            logger.debug(f"Budget reservation failed: {msg}")
            raise InsufficientBudgetError(msg, limit_name=next(iter(self.limits.keys())))

    def return_unused_budget(self, reserved_output_tokens: int, actual_output_tokens: int,
                           output_token_price: float) -> None:
        """
        Return unused budget after an operation completes.
        
        Args:
            reserved_output_tokens: Number of output tokens that were reserved
            actual_output_tokens: Actual number of output tokens used
            output_token_price: Cost per 1M output tokens in Euros
        """
        if actual_output_tokens > reserved_output_tokens:
            raise ValueError(
                f"Actual output tokens ({actual_output_tokens}) cannot exceed "
                f"reserved output tokens ({reserved_output_tokens})"
            )
            
        unused_tokens = reserved_output_tokens - actual_output_tokens
        unused_budget = unused_tokens * (output_token_price / 1_000_000)  # Convert from per 1M tokens to per token
        
        for limit in self.limits.values():
            limit.return_budget(unused_budget)

    async def return_unused_budget_async(self, *, reserved_output_tokens: int, actual_output_tokens: int,
                                       output_token_price: float) -> None:
        """
        Return unused budget after an operation completes (async version).
        
        Args:
            reserved_output_tokens: Number of output tokens that were reserved
            actual_output_tokens: Actual number of output tokens used
            output_token_price: Cost per 1M output tokens in Euros
        """
        if actual_output_tokens > reserved_output_tokens: 
            # more used than expected, further reduce budget
            additional_use = actual_output_tokens - reserved_output_tokens
            additional_budget = additional_use * (output_token_price / 1_000_000)  # Convert from per 1M tokens to per token
            for limit in self.limits.values():
                await limit.reserve_budget_async(additional_budget)
            return
            
        unused_tokens = reserved_output_tokens - actual_output_tokens
        unused_budget = unused_tokens * (output_token_price / 1_000_000)  # Convert from per 1M tokens to per token
        
        for limit in self.limits.values():
            await limit.return_budget_async(unused_budget)

    def reset(self) -> None:
        """Reset all budget limits."""
        for limit in self.limits.values():
            limit.reset()

    async def reset_async(self) -> None:
        """Reset all budget limits (async version)."""
        for limit in self.limits.values():
            await limit.reset_async()
