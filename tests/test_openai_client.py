import os
import pytest

from llming_lodge.providers.openai.openai_client import OpenAILlmClient
from llming_lodge.messages import LlmSystemMessage, LlmHumanMessage, LlmAIMessage

@pytest.fixture
def test_messages():
    return [
        LlmSystemMessage(content="You are a helpful assistant."),
        LlmHumanMessage(content="Hello, who won the world series in 2020?"),
    ]

@pytest.fixture(scope="module")
def openai_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY environment variable not set, skipping OpenAI integration tests.")
    return key

@pytest.fixture(scope="module")
def openai_model():
    # Use a default model, can be overridden by env var
    return os.environ.get("OPENAI_MODEL", "gpt-5.2")

def test_invoke_sync(openai_api_key, openai_model, test_messages):
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_model,
        temperature=0.7,
        max_tokens=32,
    )
    result = client.invoke(test_messages)
    assert isinstance(result, LlmAIMessage)
    assert result.content
    assert "world series" in result.content.lower() or "dodgers" in result.content.lower()

@pytest.mark.asyncio
async def test_ainvoke_async(openai_api_key, openai_model, test_messages):
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_model,
        temperature=0.7,
        max_tokens=32,
    )
    result = await client.ainvoke(test_messages)
    assert isinstance(result, LlmAIMessage)
    assert result.content
    assert "world series" in result.content.lower() or "dodgers" in result.content.lower()

# --- Tool support test ---

from llming_lodge.tools.llm_tool import LlmTool
from llming_lodge.tools.llm_toolbox import LlmToolbox

def add_numbers(a: int, b: int):
    return {"result": a + b}

@pytest.mark.asyncio
async def test_tool_add_numbers(openai_api_key, openai_model):
    # Define the tool
    add_tool = LlmTool(
        name="add_numbers",
        description="Add two numbers and return the result.",
        func=add_numbers,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    )
    toolbox = LlmToolbox(
        name="math_tools",
        description="Simple math tools",
        tools=[add_tool]
    )
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_model,
        temperature=0.0,
        max_tokens=32,
        toolboxes=[toolbox]
    )
    messages = [
        LlmSystemMessage(content="You are a helpful assistant."),
        LlmHumanMessage(content="What is 2 plus 3? Use a tool if you can."),
    ]
    # Collect all chunks from the async stream
    results = []
    async for chunk in client.astream(messages):
        results.append(chunk.content)
    joined = "".join(results)
    assert "5" in joined or '"result": 5' in joined

def test_tool_add_numbers_sync_invoke(openai_api_key, openai_model):
    add_tool = LlmTool(
        name="add_numbers",
        description="Add two numbers and return the result.",
        func=add_numbers,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    )
    toolbox = LlmToolbox(
        name="math_tools",
        description="Simple math tools",
        tools=[add_tool]
    )
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_model,
        temperature=0.0,
        max_tokens=32,
        toolboxes=[toolbox]
    )
    messages = [
        LlmSystemMessage(content="You are a helpful assistant."),
        LlmHumanMessage(content="What is 2 plus 3? Use a tool if you can."),
    ]
    result = client.invoke(messages)
    assert isinstance(result, LlmAIMessage)
    assert "5" in result.content or '"result": 5' in result.content

@pytest.mark.asyncio
async def test_tool_add_numbers_async_invoke(openai_api_key, openai_model):
    add_tool = LlmTool(
        name="add_numbers",
        description="Add two numbers and return the result.",
        func=add_numbers,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    )
    toolbox = LlmToolbox(
        name="math_tools",
        description="Simple math tools",
        tools=[add_tool]
    )
    client = OpenAILlmClient(
        api_key=openai_api_key,
        model=openai_model,
        temperature=0.0,
        max_tokens=32,
        toolboxes=[toolbox]
    )
    messages = [
        LlmSystemMessage(content="You are a helpful assistant."),
        LlmHumanMessage(content="What is 2 plus 3? Use a tool if you can."),
    ]
    result = await client.ainvoke(messages)
    assert isinstance(result, LlmAIMessage)
    assert "5" in result.content or '"result": 5' in result.content
