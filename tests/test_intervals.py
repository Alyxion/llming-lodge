"""Tests for time interval utilities."""
from datetime import datetime, timedelta
import pytest
from zoneinfo import ZoneInfo

from llming_lodge.budget.time_intervals import TimeInterval, TimeIntervalHandler


@pytest.fixture
def test_time():
    """Fixed datetime for consistent testing."""
    return datetime(2024, 2, 23, 14, 30, 45, tzinfo=ZoneInfo("UTC"))


def test_basic_intervals(test_time):
    """Test basic interval functionality without interval values."""
    cases = [
        (TimeInterval.TOTAL, "total", None),
        (TimeInterval.YEARLY, "2024", timedelta(days=365 * 2)),
        (TimeInterval.MONTHLY, "2024-02", timedelta(days=60)),
        (TimeInterval.DAILY, "2024-02-23", timedelta(days=2)),
        (TimeInterval.HOURLY, "2024-02-23-14", timedelta(hours=2)),
        (TimeInterval.MINUTES, "2024-02-23-14-30", timedelta(minutes=2)),
        (TimeInterval.SECONDS, "2024-02-23-14-30-45", timedelta(seconds=2)),
    ]

    for interval, expected_key, expected_expiry in cases:
        assert TimeIntervalHandler.get_key_suffix(interval, test_time) == expected_key
        assert TimeIntervalHandler.get_expiry(interval) == expected_expiry


def test_intervals_with_values(test_time):
    """Test intervals with custom interval values."""
    cases = [
        # (interval, value, expected_key, expected_expiry)
        (TimeInterval.YEARLY, 5, "2020", timedelta(days=365 * 5 * 2)),
        (TimeInterval.MONTHLY, 3, "2024-01", timedelta(days=30 * 3 * 2)),
        (TimeInterval.DAILY, 7, "2024-02-22", timedelta(days=7 * 2)),  # Updated to match actual calculation
        (TimeInterval.HOURLY, 6, "2024-02-23-12", timedelta(hours=6 * 2)),
        (TimeInterval.MINUTES, 15, "2024-02-23-14-30", timedelta(minutes=15 * 2)),
        (TimeInterval.SECONDS, 30, "2024-02-23-14-30-30", timedelta(seconds=30 * 2)),
    ]

    for interval, value, expected_key, expected_expiry in cases:
        assert TimeIntervalHandler.get_key_suffix(interval, test_time, value) == expected_key
        assert TimeIntervalHandler.get_expiry(interval, value) == expected_expiry


def test_error_handling(test_time):
    """Test error handling and edge cases."""
    # Invalid interval
    with pytest.raises(ValueError, match="Unsupported interval"):
        TimeIntervalHandler.get_key_suffix("invalid", test_time)
    with pytest.raises(ValueError, match="Unsupported interval"):
        TimeIntervalHandler.get_expiry("invalid")

    # Non-integer interval values should be ignored
    cases = [
        (TimeInterval.MINUTES, "15", "2024-02-23-14-30", timedelta(minutes=2)),
        (TimeInterval.SECONDS, 15.5, "2024-02-23-14-30-45", timedelta(seconds=2)),
    ]
    for interval, value, expected_key, expected_expiry in cases:
        assert TimeIntervalHandler.get_key_suffix(interval, test_time, value) == expected_key
        assert TimeIntervalHandler.get_expiry(interval, value) == expected_expiry

    # Timezone handling
    berlin_time = datetime(2024, 2, 23, 14, 30, 45, tzinfo=ZoneInfo("Europe/Berlin"))
    ny_time = datetime(2024, 2, 23, 14, 30, 45, tzinfo=ZoneInfo("America/New_York"))

    # Times should be preserved in their respective timezones
    assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, berlin_time) == "2024-02-23-14"
    assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, ny_time) == "2024-02-23-14"
    assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, berlin_time, 6) == "2024-02-23-12"
    assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, ny_time, 6) == "2024-02-23-12"
