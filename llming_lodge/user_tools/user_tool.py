import os
import json
from pydantic import BaseModel, Field
from llming_lodge.user_tools.user_tool_config import LLMUserToolConfig, LLMUserToolSetConfig
from llming_lodge.prompt.prompt_template import PromptTemplate

class LLMUserTool:
    """Defines a single tool which is shown in the user's tool selection"""

    def __init__(self, config: LLMUserToolConfig, include_paths: list[str] | None = None, read_only: bool = False) -> None:
        self.config = config
        self.prompt = ""
        self._engagement = ""
        self.include_paths = include_paths or []
        self.read_only: bool = read_only
        self.parse()
        self._debug_mode = os.environ.get("LLM_DEBUG_MODE", "false").lower() in ["true", "1"]

    @property
    def unique_id(self):
        return self.config.unique_id

    @property
    def label(self):
        return self.config.label

    @property
    def tool_type(self):
        return self.config.tool_type

    @property
    def model(self):
        return self.config.model

    def parse(self):
        if self.config.prompt:
            self.prompt = self.config.prompt
        else:
            if self.config.prompt_path:
                rel_path = self.config.prompt_path
                # ensure no directory traversal
                if ".." in rel_path or "/" in rel_path or "\\" in rel_path:
                    raise ValueError("Invalid prompt path")
                for include_path in self.include_paths:
                    abs_path = os.path.join(include_path, rel_path)
                    if os.path.exists(abs_path):
                        with open(abs_path, "r") as f:
                            self.prompt = f.read()
                        break
            # search for custom tokens such as @@engagement
            new_lines = []
            for line in self.prompt.splitlines():
                if line.startswith('@@'):
                    if line.startswith('@@engagement'):
                        self.engagement = line[13:]
                else:
                    new_lines.append(line)
            self.prompt = '\n'.join(new_lines)

    async def render_prompt(self, parameters: dict[str, any] | None = None) -> str:
        if self._debug_mode:
            self.parse()
        if parameters:
            pt = PromptTemplate(self.prompt)
            return pt.render(parameters=parameters)
        return self.prompt

class LLMUserToolSet:
    """Defines a set of tools which are shown in the user's tool selection"""
    def __init__(self, source: dict | str | None = None, include_paths: list[str] | None = None, read_only: bool = False) -> None:
        self.tools: list[LLMUserTool] = []
        self.include_paths = include_paths or []
        self.tool_set_config = LLMUserToolSetConfig()
        if source:
            if isinstance(source, str):
                source = json.loads(source)
            self.load_from(source)

    def load_from(self, source: dict) -> None:
        self.tool_set_config = LLMUserToolSetConfig(**source)
        for tool_config in self.tool_set_config.tools:
            self.tools.append(LLMUserTool(tool_config, self.include_paths))

    @classmethod
    def load_from_directory(cls, directory: str, *, read_only: bool = True) -> "LLMUserToolSet":
        """Load a toolset from a directory containing ``default_user_tools.json`` and a ``prompts/`` subfolder."""
        config_file = os.path.join(directory, "default_user_tools.json")
        prompts_dir = os.path.join(directory, "prompts")
        with open(config_file, "r") as f:
            config = json.load(f)
        include_paths = [prompts_dir] if os.path.isdir(prompts_dir) else []
        new_set = cls(source=config, include_paths=include_paths)
        if read_only:
            for tool in new_set.tools:
                tool.read_only = True
        return new_set
