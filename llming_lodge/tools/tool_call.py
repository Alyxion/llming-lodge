"""Tool call status tracking and structured tool call info."""
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCallStatus(str, Enum):
    """Status of a tool call."""
    PENDING = "pending"        # Tool call initiated, waiting to execute
    EXECUTING = "executing"    # Tool is currently executing
    COMPLETED = "completed"    # Tool completed successfully
    ERROR = "error"           # Tool execution failed


class ToolCallInfo(BaseModel):
    """Structured information about a tool call.

    Used by clients to emit tool events that the UI can render
    without parsing JSON strings.
    """
    name: str = Field(description="Tool/function name")
    call_id: str = Field(description="Unique identifier for this call")
    status: ToolCallStatus = Field(description="Current status of the tool call")
    arguments: Optional[dict] = Field(default=None, description="Arguments passed to the tool")
    result: Optional[Any] = Field(default=None, description="Result from tool execution")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        return self.name.replace('_', ' ').title()

    @property
    def is_image_generation(self) -> bool:
        """Check if this is an image generation tool."""
        return self.name == 'generate_image'
