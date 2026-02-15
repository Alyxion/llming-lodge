from typing import List, Union
from .llm_tool import LlmTool

class LlmToolbox:
    """
    A collection of tools that can be used by an LLM.
    """

    def __init__(self, name: str, description: str, tools: List[Union[LlmTool, str]]):
        """
        Initialize the toolbox.
        """
        self.name = name
        self.description = description
        self.tools = tools
