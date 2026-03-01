import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo
from pymongo import ReturnDocument

from .time_intervals import TimeIntervalHandler
from .budget_limit import BudgetLimit
from .budget_types import LimitPeriod

logger = logging.getLogger(__name__)

class MongoDBBudgetLimit(BudgetLimit):
    """Budget limit with MongoDB-based tracking using atomic operations."""
    def __init__(
        self,
        *,
        name: str,
        amount: float,
        period: LimitPeriod,
        mongo_uri: str,
        mongo_db: str,
        mongo_collection: str,
        interval_value: Optional[int] = None,
        timezone_str: str = "UTC",
        enable_logging: bool = False,
        user_id: Optional[str] = None
    ):
        super().__init__(name=name, amount=amount, period=period, interval_value=interval_value, timezone_str=timezone_str)
        if not mongo_uri or not mongo_db or not mongo_collection:
            raise ValueError("MongoDB URI, database, and collection are required")
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection
        self.enable_logging = enable_logging
        self.user_id = user_id
        self._async_coll = None

    def _ensure_async_coll(self):
        """Lazily get the shared async client and set up the collection handle."""
        if self._async_coll is None:
            from nice_droplets.utils.mongo_helpers import get_async_mongo_client
            client = get_async_mongo_client(self.mongo_uri)
            self._async_coll = client[self.mongo_db][self.mongo_collection]
        return self._async_coll

    def _get_current_time(self) -> datetime:
        """Get current time in the configured timezone."""
        return datetime.now(timezone.utc).astimezone(ZoneInfo(self.timezone))

    def _get_mongo_key(self, time: Optional[datetime] = None) -> dict:
        """Generate the MongoDB document key for this budget and period (top-level doc)."""
        if time is None:
            time = self._get_current_time()
        return {
            "name": self.name,
            "period": self.period.value,
        }

    def _get_usage_path(self, time: Optional[datetime] = None) -> list:
        """
        Generate the linear path for usage in the document, e.g. ["usage", "2025-05-17"].
        """
        if time is None:
            time = self._get_current_time()
        key_suffix = self._get_key_suffix(time)
        return ["usage", key_suffix]

    def _get_expiry(self) -> Optional[timedelta]:
        """Get expiry duration based on period."""
        return TimeIntervalHandler.get_expiry(self.period, self.interval_value)

    async def get_available_budget_async(self) -> float:
        """Get available budget for the current period."""
        logger.debug(f"Getting available budget (async) for limit '{self.name}' (period: {self.period.value})")
        current_time = self._get_current_time()
        key = self._get_mongo_key(current_time)
        usage_path = self._get_usage_path(current_time)
        field = ".".join(usage_path + ["used"])
        try:
            coll = self._ensure_async_coll()
            doc = await coll.find_one(key, {field: 1})
            used = 0.0
            d = doc
            for part in usage_path:
                if d is None or part not in d:
                    d = None
                    break
                d = d[part]
            if d is not None and isinstance(d, dict) and "used" in d:
                used = float(d["used"])
            elif isinstance(d, (int, float)):
                used = float(d)
            available = max(self.amount - used, 0.0)
            logger.debug(f"Available budget (async) for limit '{self.name}': {self.amount} - {used} = {available}")
            return available
        except Exception as e:
            raise RuntimeError(f"Failed to get available budget from MongoDB (async): {e}")

    async def reserve_budget_async(self, amount: float) -> bool:
        """Reserve budget for an operation."""
        logger.debug(f"Attempting to reserve {amount} (async) from limit '{self.name}' (period: {self.period.value})")
        if amount > self.amount:
            logger.debug(f"Amount {amount} exceeds total budget {self.amount} for limit '{self.name}'")
            return False

        current_time = self._get_current_time()
        key = self._get_mongo_key(current_time)
        usage_path = self._get_usage_path(current_time)
        field = ".".join(usage_path + ["used"])
        now = current_time
        coll = self._ensure_async_coll()

        try:
            update = {
                "$inc": {field: amount},
                "$setOnInsert": {"created_at": now},
            }
            result = await coll.find_one_and_update(
                key,
                update,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            d = result
            for part in usage_path:
                if d is None or part not in d:
                    d = None
                    break
                d = d[part]
            used = float(d["used"]) if d and "used" in d else amount

            if used > self.amount:
                await coll.update_one(key, {"$inc": {field: -amount}})
                logger.debug(f"Exceeded budget limit (async), rolling back. Available: {self.amount - used + amount}")
                return False

            logger.debug(f"Successfully reserved {amount} (async), remaining: {self.amount - used}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to reserve budget in MongoDB (async): {e}")

    async def return_budget_async(self, amount: float) -> None:
        """Return unused budget."""
        key = self._get_mongo_key()
        usage_path = self._get_usage_path()
        field = ".".join(usage_path + ["used"])
        coll = self._ensure_async_coll()

        try:
            update = {
                "$inc": {field: -amount},
            }
            result = await coll.find_one_and_update(
                key,
                update,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            d = result
            for part in usage_path:
                if d is None or part not in d:
                    d = None
                    break
                d = d[part]
            used = float(d["used"]) if d and "used" in d else 0.0

            if used < 0:
                await coll.update_one(key, {"$set": {field: 0}})
                logger.debug("Usage went below 0 (async), resetting to 0")
        except Exception as e:
            raise RuntimeError(f"Failed to return budget to MongoDB (async): {e}")

    async def reset_async(self) -> None:
        """Reset budget to initial amount (remove all usage records for this budget)."""
        pattern = {
            "name": self.name,
            "period": self.period.value,
        }
        try:
            coll = self._ensure_async_coll()
            await coll.delete_many(pattern)
            logger.debug(f"Reset all budget records (async) for '{self.name}' (period: {self.period.value})")
        except Exception as e:
            raise RuntimeError(f"Failed to reset budget in MongoDB (async): {e}")

    async def log_usage_async(self, *, model_name: str, tokens_input: int, tokens_output: int, costs: float, duration_ms: Optional[float] = None, user_id: Optional[str] = None, operation_type: Optional[str] = None) -> None:
        """Log usage information for a completed request."""
        # Skip if logging is disabled
        if not self.enable_logging:
            return

        key = self._get_mongo_key()
        now = self._get_current_time()
        usage_path = self._get_usage_path(now)
        logs_field = ".".join(usage_path + ["logs"])

        # Create log entry
        log_entry = {
            "timestamp": now,
            "model": model_name,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "costs": costs
        }

        # Add duration if available
        if duration_ms is not None:
            log_entry["duration_ms"] = duration_ms

        # Add operation type if available
        if operation_type:
            log_entry["operation_type"] = operation_type

        # Add user_id if available (parameter takes precedence over instance attribute)
        effective_user_id = user_id if user_id is not None else self.user_id
        if effective_user_id:
            log_entry["user_id"] = effective_user_id

        try:
            # Append log entry to logs array for the current time interval
            coll = self._ensure_async_coll()
            await coll.update_one(
                key,
                {"$push": {logs_field: log_entry}},
                upsert=True
            )
            logger.debug(f"Logged usage (async) for model {model_name}: {tokens_input} input tokens, {tokens_output} output tokens, {costs} cost")
        except Exception as e:
            logger.error(f"Failed to log usage in MongoDB (async): {e}")
