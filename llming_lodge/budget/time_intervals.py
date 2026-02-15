"""Time interval utilities for budget management."""
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Union


class TimeInterval(Enum):
    """Time intervals for budget periods."""
    TOTAL = "total"
    YEARLY = "yearly"
    MONTHLY = "monthly"
    DAILY = "daily"
    HOURLY = "hourly"
    MINUTES = "minutes"
    SECONDS = "seconds"


class TimeIntervalHandler:
    """Handles time interval operations for budget periods."""

    @staticmethod
    def get_key_suffix(interval: TimeInterval, time: datetime, interval_value: Optional[Union[int, str]] = None) -> str:
        """Generate key suffix based on interval.
        
        Args:
            interval: The time interval type
            time: The datetime to generate the key for
            interval_value: Optional value for intervals (e.g. 30 for 30-second intervals)
        
        Returns:
            A string key suffix representing the time interval
        
        Raises:
            ValueError: If interval is not supported
        """
        if interval == TimeInterval.TOTAL:
            return "total"
        elif interval == TimeInterval.YEARLY:
            if isinstance(interval_value, int):
                current_year = time.year
                interval_bucket = (current_year // interval_value) * interval_value
                return f"{interval_bucket}"
            return time.strftime("%Y")
        elif interval == TimeInterval.MONTHLY:
            if isinstance(interval_value, int):
                current_month = time.year * 12 + time.month - 1
                interval_bucket = (current_month // interval_value) * interval_value
                year = interval_bucket // 12
                month = (interval_bucket % 12) + 1
                return f"{year:04d}-{month:02d}"
            return time.strftime("%Y-%m")
        elif interval == TimeInterval.DAILY:
            if isinstance(interval_value, int):
                days_since_epoch = (time.date() - datetime(1970, 1, 1).date()).days
                interval_bucket = (days_since_epoch // interval_value) * interval_value
                interval_date = datetime(1970, 1, 1) + timedelta(days=interval_bucket)
                return interval_date.strftime("%Y-%m-%d")
            return time.strftime("%Y-%m-%d")
        elif interval == TimeInterval.HOURLY:
            if isinstance(interval_value, int):
                hours_since_midnight = time.hour
                interval_bucket = (hours_since_midnight // interval_value) * interval_value
                return time.strftime(f"%Y-%m-%d-{interval_bucket:02d}")
            return time.strftime("%Y-%m-%d-%H")
        elif interval == TimeInterval.MINUTES:
            if isinstance(interval_value, int):
                current_minute = time.minute
                interval_bucket = (current_minute // interval_value) * interval_value
                return time.strftime(f"%Y-%m-%d-%H-{interval_bucket:02d}")
            return time.strftime("%Y-%m-%d-%H-%M")
        elif interval == TimeInterval.SECONDS:
            if isinstance(interval_value, int):
                current_second = time.minute * 60 + time.second
                interval_bucket = (current_second // interval_value) * interval_value
                minutes = interval_bucket // 60
                seconds = interval_bucket % 60
                return time.strftime(f"%Y-%m-%d-%H-{minutes:02d}-{seconds:02d}")
            return time.strftime("%Y-%m-%d-%H-%M-%S")
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    @staticmethod
    def get_expiry(interval: TimeInterval, interval_value: Optional[Union[int, str]] = None) -> Optional[timedelta]:
        """Get expiry duration based on interval.
        
        Args:
            interval: The time interval type
            interval_value: Optional value for intervals
        
        Returns:
            Optional[timedelta]: The expiry duration, or None for TOTAL interval
        
        Raises:
            ValueError: If interval is not supported
        """
        if interval == TimeInterval.TOTAL:
            return None
        elif interval == TimeInterval.YEARLY:
            if isinstance(interval_value, int):
                return timedelta(days=365 * interval_value * 2)  # 2x the interval
            return timedelta(days=365 * 2)  # 2 years
        elif interval == TimeInterval.MONTHLY:
            if isinstance(interval_value, int):
                return timedelta(days=30 * interval_value * 2)  # 2x the interval
            return timedelta(days=60)  # 2 months
        elif interval == TimeInterval.DAILY:
            if isinstance(interval_value, int):
                return timedelta(days=interval_value * 2)  # 2x the interval
            return timedelta(days=2)  # 2 days
        elif interval == TimeInterval.HOURLY:
            if isinstance(interval_value, int):
                return timedelta(hours=interval_value * 2)  # 2x the interval
            return timedelta(hours=2)  # 2 hours
        elif interval == TimeInterval.MINUTES:
            if isinstance(interval_value, int):
                return timedelta(minutes=interval_value * 2)  # 2x the interval
            return timedelta(minutes=2)  # 2 minutes
        elif interval == TimeInterval.SECONDS:
            if isinstance(interval_value, int):
                return timedelta(seconds=interval_value * 2)  # 2x the interval
            return timedelta(seconds=2)  # 2 seconds
        else:
            raise ValueError(f"Unsupported interval: {interval}")
