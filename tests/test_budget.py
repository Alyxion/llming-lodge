"""Tests for the budget management system (async-only API)."""
import asyncio
import pytest
import pytest_asyncio

from llming_models.budget import (
    LLMBudgetManager,
    MemoryBudgetLimit,
    LimitPeriod,
    InsufficientBudgetError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(amount: float = 1.0, period: LimitPeriod = LimitPeriod.TOTAL) -> LLMBudgetManager:
    """Create a simple single-limit manager."""
    return LLMBudgetManager([MemoryBudgetLimit(name="test", amount=amount, period=period)])


# Pricing: $10/1M input, $30/1M output (similar to GPT-4 class models)
INPUT_PRICE = 10.0
OUTPUT_PRICE = 30.0


# ---------------------------------------------------------------------------
# LLMBudgetManager — async API
# ---------------------------------------------------------------------------

class TestBudgetManagerAsync:
    """All tests use the async API exclusively."""

    @pytest.mark.asyncio
    async def test_available_budget(self):
        mgr = _make_manager(amount=5.0)
        budget = await mgr.available_budget_async()
        assert budget == 5.0

    @pytest.mark.asyncio
    async def test_reserve_budget(self):
        mgr = _make_manager(amount=1.0)
        await mgr.reserve_budget_async(
            input_tokens=1000,
            max_output_tokens=1000,
            input_token_price=INPUT_PRICE,
            output_token_price=OUTPUT_PRICE,
        )
        # 1000 * 10/1M + 1000 * 30/1M = 0.01 + 0.03 = 0.04
        budget = await mgr.available_budget_async()
        assert abs(budget - 0.96) < 1e-9

    @pytest.mark.asyncio
    async def test_reserve_insufficient_budget(self):
        mgr = _make_manager(amount=0.001)
        with pytest.raises(InsufficientBudgetError):
            await mgr.reserve_budget_async(
                input_tokens=100_000,
                max_output_tokens=100_000,
                input_token_price=INPUT_PRICE,
                output_token_price=OUTPUT_PRICE,
            )

    @pytest.mark.asyncio
    async def test_return_unused_budget(self):
        mgr = _make_manager(amount=1.0)
        # Reserve for 4000 output tokens
        await mgr.reserve_budget_async(
            input_tokens=1000,
            max_output_tokens=4000,
            input_token_price=INPUT_PRICE,
            output_token_price=OUTPUT_PRICE,
        )
        # Reserved: 1000*10/1M + 4000*30/1M = 0.01 + 0.12 = 0.13
        budget_after_reserve = await mgr.available_budget_async()
        assert abs(budget_after_reserve - 0.87) < 1e-9

        # Actually used only 1000 output tokens — return the rest
        await mgr.return_unused_budget_async(
            reserved_output_tokens=4000,
            actual_output_tokens=1000,
            output_token_price=OUTPUT_PRICE,
        )
        # Returned: (4000-1000) * 30/1M = 0.09
        budget_after_return = await mgr.available_budget_async()
        assert abs(budget_after_return - 0.96) < 1e-9

    @pytest.mark.asyncio
    async def test_return_overuse(self):
        """When actual tokens exceed reserved, additional budget is consumed."""
        mgr = _make_manager(amount=1.0)
        await mgr.reserve_budget_async(
            input_tokens=1000,
            max_output_tokens=1000,
            input_token_price=INPUT_PRICE,
            output_token_price=OUTPUT_PRICE,
        )
        # Reserved cost: 0.01 + 0.03 = 0.04 → available = 0.96
        # Now report actual = 2000 (overuse by 1000)
        await mgr.return_unused_budget_async(
            reserved_output_tokens=1000,
            actual_output_tokens=2000,
            output_token_price=OUTPUT_PRICE,
        )
        # Additional cost: 1000 * 30/1M = 0.03
        budget = await mgr.available_budget_async()
        assert abs(budget - 0.93) < 1e-9

    @pytest.mark.asyncio
    async def test_reset(self):
        mgr = _make_manager(amount=1.0)
        await mgr.reserve_budget_async(
            input_tokens=10_000,
            max_output_tokens=10_000,
            input_token_price=INPUT_PRICE,
            output_token_price=OUTPUT_PRICE,
        )
        budget_before = await mgr.available_budget_async()
        assert budget_before < 1.0

        await mgr.reset_async()
        budget_after = await mgr.available_budget_async()
        assert budget_after == 1.0

    @pytest.mark.asyncio
    async def test_multiple_limits(self):
        """Manager enforces the tightest limit."""
        mgr = LLMBudgetManager([
            MemoryBudgetLimit(name="daily", amount=0.10, period=LimitPeriod.DAILY),
            MemoryBudgetLimit(name="monthly", amount=10.0, period=LimitPeriod.MONTHLY),
        ])
        # This should succeed — both limits have room
        await mgr.reserve_budget_async(
            input_tokens=1000,
            max_output_tokens=1000,
            input_token_price=INPUT_PRICE,
            output_token_price=OUTPUT_PRICE,
        )
        budget = await mgr.available_budget_async()
        # min(0.10 - 0.04, 10.0 - 0.04) = 0.06
        assert abs(budget - 0.06) < 1e-9

    @pytest.mark.asyncio
    async def test_multiple_limits_tight_one_fails(self):
        """If the tighter limit is exhausted, reservation fails and rolls back."""
        mgr = LLMBudgetManager([
            MemoryBudgetLimit(name="tight", amount=0.01, period=LimitPeriod.TOTAL),
            MemoryBudgetLimit(name="loose", amount=100.0, period=LimitPeriod.TOTAL),
        ])
        with pytest.raises(InsufficientBudgetError) as exc_info:
            await mgr.reserve_budget_async(
                input_tokens=10_000,
                max_output_tokens=10_000,
                input_token_price=INPUT_PRICE,
                output_token_price=OUTPUT_PRICE,
            )
        assert exc_info.value.limit_name == "tight"
        # Loose limit should have been rolled back
        loose_budget = await mgr.limits["loose"].get_available_budget_async()
        assert loose_budget == 100.0

    @pytest.mark.asyncio
    async def test_no_sync_methods(self):
        """Sync methods must not exist on LLMBudgetManager."""
        mgr = _make_manager()
        assert not hasattr(mgr, "available_budget")
        assert not hasattr(mgr, "reserve_budget")
        assert not hasattr(mgr, "return_unused_budget")
        assert not hasattr(mgr, "reset")


# ---------------------------------------------------------------------------
# MemoryBudgetLimit — async API
# ---------------------------------------------------------------------------

class TestMemoryBudgetLimitAsync:

    @pytest.mark.asyncio
    async def test_total_limit_reserve_and_return(self):
        limit = MemoryBudgetLimit(name="total", amount=1.0, period=LimitPeriod.TOTAL)
        assert await limit.get_available_budget_async() == 1.0

        assert await limit.reserve_budget_async(0.3) is True
        assert abs(await limit.get_available_budget_async() - 0.7) < 1e-9

        await limit.return_budget_async(0.1)
        assert abs(await limit.get_available_budget_async() - 0.8) < 1e-9

    @pytest.mark.asyncio
    async def test_total_limit_overdraw_rejected(self):
        limit = MemoryBudgetLimit(name="total", amount=0.5, period=LimitPeriod.TOTAL)
        assert await limit.reserve_budget_async(0.6) is False
        # Budget unchanged
        assert await limit.get_available_budget_async() == 0.5

    @pytest.mark.asyncio
    async def test_daily_limit(self):
        limit = MemoryBudgetLimit(name="daily", amount=1.0, period=LimitPeriod.DAILY)
        assert await limit.get_available_budget_async() == 1.0

        assert await limit.reserve_budget_async(0.4) is True
        assert abs(await limit.get_available_budget_async() - 0.6) < 1e-9

    @pytest.mark.asyncio
    async def test_reset(self):
        limit = MemoryBudgetLimit(name="total", amount=1.0, period=LimitPeriod.TOTAL)
        await limit.reserve_budget_async(0.8)
        assert abs(await limit.get_available_budget_async() - 0.2) < 1e-9

        await limit.reset_async()
        assert await limit.get_available_budget_async() == 1.0

    @pytest.mark.asyncio
    async def test_monthly_limit(self):
        limit = MemoryBudgetLimit(name="monthly", amount=10.0, period=LimitPeriod.MONTHLY)
        assert await limit.reserve_budget_async(3.0) is True
        assert abs(await limit.get_available_budget_async() - 7.0) < 1e-9
        await limit.return_budget_async(1.0)
        assert abs(await limit.get_available_budget_async() - 8.0) < 1e-9

    @pytest.mark.asyncio
    async def test_budget_limit_abc_has_no_sync_abstract_methods(self):
        """BudgetLimit ABC only defines async abstract methods."""
        from llming_models.budget.budget_limit import BudgetLimit
        import inspect
        # All abstract methods should be async (coroutine functions)
        for name, method in inspect.getmembers(BudgetLimit, predicate=inspect.isfunction):
            if getattr(method, "__isabstractmethod__", False):
                assert inspect.iscoroutinefunction(method), (
                    f"Abstract method {name} should be async"
                )
