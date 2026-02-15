import os
import uuid
import pytest
import asyncio
import sys

from llming_lodge.budget.mongodb_budget_limit import MongoDBBudgetLimit
from llming_lodge.budget.budget_types import LimitPeriod

# ensure environment variables are set
MONGODB_URI = os.environ.get("MONGODB_CONNECTION", "mongodb://localhost:27017")
pytestmark = pytest.mark.skipif(not MONGODB_URI, reason="MONGODB_CONNECTION not set")

@pytest.fixture(scope="module")
def mongo_budget():
    db_name = "test_llming_lodge"
    collection = f"budget_test_{uuid.uuid4().hex[:8]}"
    budget = MongoDBBudgetLimit(
        name="test_budget",
        amount=100.0,
        period=LimitPeriod.DAILY,
        mongo_uri=MONGODB_URI,
        mongo_db=db_name,
        mongo_collection=collection,
        timezone_str="UTC"
    )
    yield budget
    # Cleanup
    budget.reset()
    # Drop the collection to ensure no data remains
    budget._client[db_name].drop_collection(collection)
    # Close MongoDB connections
    budget._client.close()
    budget._async_client.close()

@pytest.mark.asyncio
async def test_mongodb_budget_limit_async(mongo_budget):
    await mongo_budget.reset_async()
    # Initial budget
    avail = await mongo_budget.get_available_budget_async()
    assert avail == 100.0

    # Reserve 30
    ok = await mongo_budget.reserve_budget_async(30.0)
    assert ok
    avail = await mongo_budget.get_available_budget_async()
    assert avail == 70.0

    # Reserve over limit
    ok = await mongo_budget.reserve_budget_async(100.0)
    assert not ok
    avail = await mongo_budget.get_available_budget_async()
    assert avail == 70.0

    # Return 20
    await mongo_budget.return_budget_async(20.0)
    avail = await mongo_budget.get_available_budget_async()
    assert avail == 90.0

    # Reset
    await mongo_budget.reset_async()
    avail = await mongo_budget.get_available_budget_async()
    assert avail == 100.0

def test_mongodb_budget_limit_sync(mongo_budget):
    mongo_budget.reset()
    # Initial budget
    avail = mongo_budget.get_available_budget()
    assert avail == 100.0

    # Reserve 30
    ok = mongo_budget.reserve_budget(30.0)
    assert ok
    avail = mongo_budget.get_available_budget()
    assert avail == 70.0

    # Reserve over limit
    ok = mongo_budget.reserve_budget(100.0)
    assert not ok
    avail = mongo_budget.get_available_budget()
    assert avail == 70.0

    # Return 20
    mongo_budget.return_budget(20.0)
    avail = mongo_budget.get_available_budget()
    assert avail == 90.0

    # Reset
    mongo_budget.reset()
    avail = mongo_budget.get_available_budget()
    assert avail == 100.0
