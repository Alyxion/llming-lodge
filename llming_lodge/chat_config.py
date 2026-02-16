"""Pydantic config models for the llming-lodge chat page.

ChatAppConfig — app-level settings shared across all users.
ChatUserConfig — per-user runtime settings (MCPs, budget, identity).
QuickAction — a suggestion card shown on the empty-chat screen.
ThemeConfig — pre-computed CSS accent variables.
ChatFrontendConfig — the JSON object injected as ``window.__CHAT_CONFIG__``.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class QuickAction(BaseModel):
    """A suggestion card shown on the empty-chat welcome screen."""
    id: str
    label: str
    description: str = Field(default="", serialization_alias="desc")
    icon: str = "lightbulb"
    engagement: str = ""
    prompt: str = ""
    text_prefix: str = Field(default="", serialization_alias="textPrefix")


class ThemeConfig(BaseModel):
    """Pre-computed CSS accent variables passed to the JS frontend."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    accent: str
    accent_rgb: str
    accent_hover: str
    accent_light: str

    @classmethod
    def from_hex(cls, hex_color: str) -> "ThemeConfig":
        """Compute all CSS accent values from a single hex color."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        hr = max(0, int(r * 0.75))
        hg = max(0, int(g * 0.75))
        hb = max(0, int(b * 0.75))
        return cls(
            accent=hex_color,
            accent_rgb=f"{r}, {g}, {b}",
            accent_hover=f"#{hr:02x}{hg:02x}{hb:02x}",
            accent_light=f"rgba({r}, {g}, {b}, 0.12)",
        )


class ChatAppConfig(BaseModel):
    """App-level config — shared across all users."""
    accent_color: str = ""
    app_logo: str = ""
    app_title: str = ""
    app_mascot: str = ""
    show_budget: bool = False


class ChatUserConfig(BaseModel):
    """Per-user config — includes runtime objects (MCPs, budget)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str
    user_name: str
    full_name: str = ""
    user_email: str = ""
    user_avatar: str = ""
    budget_limits: list[Any] = Field(default_factory=list)
    budget_handler: Any = None
    context_preamble: str = ""
    mcp_servers: list[Any] = Field(default_factory=list)
    initial_model: str | None = None
    quick_actions: list[QuickAction] | None = None


class ChatFrontendConfig(BaseModel):
    """JSON config injected as ``window.__CHAT_CONFIG__``."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    session_id: str
    ws_path: str
    user_name: str
    user_email: str
    user_id: str
    user_avatar: str
    static_base: str = "/llming-static"
    show_budget: bool = False
    app_logo: str = ""
    app_title: str = ""
    app_mascot: str = ""
    theme: ThemeConfig | None = None
