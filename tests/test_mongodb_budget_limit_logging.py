import os
import pytest
from datetime import datetime
from pymongo import MongoClient
from llming_lodge.budget import MongoDBBudgetLimit
from llming_lodge.budget.budget_types import LimitPeriod

# Skip tests if MongoDB connection is not available
MONGODB_URI = os.environ.get("MONGODB_CONNECTION", "mongodb://localhost:27017")
pytestmark = pytest.mark.skipif(not MONGODB_URI, reason="MONGODB_CONNECTION not set")

def test_log_usage():
    """Test that log_usage correctly logs usage information."""
    # Create a test budget limit with logging enabled and user_id
    budget_limit = MongoDBBudgetLimit(
        mongo_uri=MONGODB_URI,
        mongo_db="test_llming_lodge",
        mongo_collection="test_budget",
        name="test_log_usage",
        amount=100.0,
        period=LimitPeriod.DAILY,
        enable_logging=True,
        user_id="test_user_123"
    )
    
    try:
        # Log some usage with duration
        budget_limit.log_usage(
            model_name="openai.gpt-4",
            tokens_input=100,
            tokens_output=50,
            costs=0.25,
            duration_ms=1500.0,  # 1.5 seconds
            user_id="override_user_id"  # Override the user_id set in constructor
        )
        
        # Verify that the log entry was created
        client = MongoClient(MONGODB_URI)
        coll = client["test_llming_lodge"]["test_budget"]
        doc = coll.find_one({"name": "test_log_usage"})
        
        # Get current time interval key
        now = datetime.now()
        if budget_limit.period == LimitPeriod.DAILY:
            time_key = now.strftime("%Y-%m-%d")
        elif budget_limit.period == LimitPeriod.MONTHLY:
            time_key = now.strftime("%Y-%m")
        else:
            time_key = now.strftime("%Y-%m-%d")
            
        # Check that the usage and logs exist for the current time interval
        assert "usage" in doc
        assert time_key in doc["usage"]
        assert "logs" in doc["usage"][time_key]
        logs = doc["usage"][time_key]["logs"]
        assert isinstance(logs, list)
        assert len(logs) == 1
        
        # Check the log entry fields
        log_entry = logs[0]
        assert "timestamp" in log_entry
        assert isinstance(log_entry["timestamp"], datetime)
        assert log_entry["model"] == "openai.gpt-4"
        assert log_entry["tokens_input"] == 100
        assert log_entry["tokens_output"] == 50
        assert log_entry["costs"] == 0.25
        assert log_entry["user_id"] == "override_user_id"  # Check that the override user_id is used
        assert log_entry["duration_ms"] == 1500.0
        
        # Log another usage
        budget_limit.log_usage(
            model_name="anthropic.claude-3",
            tokens_input=200,
            tokens_output=100,
            costs=0.5
        )
        
        # Verify that a second log entry was added
        doc = coll.find_one({"name": "test_log_usage"})
        logs = doc["usage"][time_key]["logs"]
        assert len(logs) == 2
        
        # Check the second log entry
        log_entry = logs[1]
        assert log_entry["model"] == "anthropic.claude-3"
        assert log_entry["tokens_input"] == 200
        assert log_entry["tokens_output"] == 100
        assert log_entry["costs"] == 0.5
        assert log_entry["user_id"] == "test_user_123"
        
    finally:
        # Clean up
        budget_limit.reset()

@pytest.mark.asyncio
async def test_log_usage_async():
    """Test that log_usage_async correctly logs usage information."""
    # Create a test budget limit with logging enabled and user_id
    budget_limit = MongoDBBudgetLimit(
        mongo_uri=MONGODB_URI,
        mongo_db="test_llming_lodge",
        mongo_collection="test_budget",
        name="test_log_usage_async",
        amount=100.0,
        period=LimitPeriod.DAILY,
        enable_logging=True,
        user_id="test_user_456"
    )
    
    try:
        # Log some usage with duration
        await budget_limit.log_usage_async(
            model_name="openai.gpt-4",
            tokens_input=100,
            tokens_output=50,
            costs=0.25,
            duration_ms=2000.0,  # 2 seconds
            user_id="override_user_id_async"  # Override the user_id set in constructor
        )
        
        # Verify that the log entry was created
        client = MongoClient(MONGODB_URI)
        coll = client["test_llming_lodge"]["test_budget"]
        doc = coll.find_one({"name": "test_log_usage_async"})
        
        # Get current time interval key
        now = datetime.now()
        if budget_limit.period == LimitPeriod.DAILY:
            time_key = now.strftime("%Y-%m-%d")
        elif budget_limit.period == LimitPeriod.MONTHLY:
            time_key = now.strftime("%Y-%m")
        else:
            time_key = now.strftime("%Y-%m-%d")
            
        # Check that the usage and logs exist for the current time interval
        assert "usage" in doc
        assert time_key in doc["usage"]
        assert "logs" in doc["usage"][time_key]
        logs = doc["usage"][time_key]["logs"]
        assert isinstance(logs, list)
        assert len(logs) == 1
        
        # Check the log entry fields
        log_entry = logs[0]
        assert "timestamp" in log_entry
        assert isinstance(log_entry["timestamp"], datetime)
        assert log_entry["model"] == "openai.gpt-4"
        assert log_entry["tokens_input"] == 100
        assert log_entry["tokens_output"] == 50
        assert log_entry["costs"] == 0.25
        assert log_entry["user_id"] == "override_user_id_async"  # Check that the override user_id is used
        assert log_entry["duration_ms"] == 2000.0
        
        # Log another usage
        await budget_limit.log_usage_async(
            model_name="anthropic.claude-3",
            tokens_input=200,
            tokens_output=100,
            costs=0.5
        )
        
        # Verify that a second log entry was added
        doc = coll.find_one({"name": "test_log_usage_async"})
        logs = doc["usage"][time_key]["logs"]
        assert len(logs) == 2
        
        # Check the second log entry
        log_entry = logs[1]
        assert log_entry["model"] == "anthropic.claude-3"
        assert log_entry["tokens_input"] == 200
        assert log_entry["tokens_output"] == 100
        assert log_entry["costs"] == 0.5
        assert log_entry["user_id"] == "test_user_456"
        
    finally:
        # Clean up
        await budget_limit.reset_async()
