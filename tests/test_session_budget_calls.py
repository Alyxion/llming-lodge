"""Unit tests for ChatSession budget manager integration.

This test suite ensures that the ChatSession correctly calls the budget manager
APIs when making LLM calls. It verifies the correct API signatures are used.

The session.py code uses LLMBudgetManager, not MemoryBudgetLimit directly.
LLMBudgetManager.reserve_budget() takes token counts and prices as kwargs.
MemoryBudgetLimit.reserve_budget() takes a single cost amount.
"""

import pytest
from llming_lodge.budget import MemoryBudgetLimit, LimitPeriod, LLMBudgetManager


class TestLLMBudgetManagerAPI:
    """Test that LLMBudgetManager API matches what session.py expects."""

    def test_reserve_budget_accepts_token_params(self):
        """Test that LLMBudgetManager.reserve_budget takes token params.

        This is the API that session.py uses.
        """
        budget_limit = MemoryBudgetLimit(name="test", amount=100.0, period=LimitPeriod.TOTAL)
        manager = LLMBudgetManager(limits=[budget_limit])

        # This is what session.py calls
        manager.reserve_budget(
            input_tokens=1000,
            max_output_tokens=4096,
            input_token_price=0.01,
            output_token_price=0.02
        )

        # Budget should be reduced
        assert budget_limit.amount < 100.0

    def test_return_unused_budget_async_works(self):
        """Test that return_unused_budget_async works correctly."""
        import asyncio

        budget_limit = MemoryBudgetLimit(name="test", amount=100.0, period=LimitPeriod.TOTAL)
        manager = LLMBudgetManager(limits=[budget_limit])

        # Reserve some budget first
        manager.reserve_budget(
            input_tokens=1000,
            max_output_tokens=10000,
            input_token_price=0.01,
            output_token_price=0.02
        )

        budget_after_reserve = budget_limit.amount

        async def test_return():
            # Return unused budget (used less than reserved)
            await manager.return_unused_budget_async(
                reserved_output_tokens=10000,
                actual_output_tokens=1000,
                output_token_price=0.02
            )

        asyncio.run(test_return())

        # Budget should be partially returned
        assert budget_limit.amount > budget_after_reserve

    def test_limits_attribute_exists(self):
        """Test that LLMBudgetManager has limits attribute for iteration."""
        budget_limit = MemoryBudgetLimit(name="test", amount=100.0, period=LimitPeriod.TOTAL)
        manager = LLMBudgetManager(limits=[budget_limit])

        # session.py iterates over manager.limits.values()
        assert hasattr(manager, 'limits')
        assert isinstance(manager.limits, dict)
        assert len(manager.limits) == 1
        assert "test" in manager.limits


class TestMemoryBudgetLimitAPI:
    """Test that MemoryBudgetLimit works correctly with single cost param."""

    def test_reserve_budget_with_cost(self):
        """Test basic reserve_budget with cost amount."""
        budget = MemoryBudgetLimit(name="test", amount=100.0, period=LimitPeriod.TOTAL)
        result = budget.reserve_budget(10.0)
        assert result is True
        assert budget.amount == 90.0
