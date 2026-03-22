from pydantic import BaseModel, Field

class LLMUserToolType:
    """Type of the user tool"""
    TEXT_TO_TEXT = "text_to_text"
    EMAIL_FILTER = "email_filter"
    TYPES = [TEXT_TO_TEXT, EMAIL_FILTER]

class LLMUserToolConfig(BaseModel):
    """Configuration for a user tool"""
    unique_id: str = Field("", description="Unique identifier for the user tool")
    label: str = Field("", description="Label for the user tool")
    prompt: str = Field("", description="Prompt for the user tool")
    prompt_path: str = Field("", description="Path to the prompt file")
    tool_type: str = Field(LLMUserToolType.TEXT_TO_TEXT, description="Type of the user tool. See UserToolType for valid values")
    model: str = Field("", description="Model for the user tool. Empty = use session default. Supports @category (e.g. @large, @small) or concrete model names.")
    script: dict = Field(default_factory=dict, description="Script configuration for the user tool")

class LLMUserToolSetConfig(BaseModel):
    """Configuration for a set of user tools"""
    tools: list[LLMUserToolConfig] = Field(default_factory=list, description="List of user tools")
