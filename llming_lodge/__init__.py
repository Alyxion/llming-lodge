"""llming-lodge — chat application layer.

LLM execution engine (providers, sessions, tools, budget) lives in llming-models.
This package provides the chat UI, WebSocket controller, document plugins,
nudge system, speech, and i18n on top of llming-models.
"""

from llming_lodge.chat_config import (
    ChatAppConfig, ChatUserConfig, QuickAction, ThemeConfig, ChatFrontendConfig,
    PERM_NUDGE_ADMIN, PERM_DEV_TOOLS, PERM_HUB_ADMIN,
)
from llming_lodge.app_extensions import AppExtension

__all__ = [
    "ChatAppConfig",
    "ChatUserConfig",
    "QuickAction",
    "ThemeConfig",
    "ChatFrontendConfig",
    "PERM_NUDGE_ADMIN",
    "PERM_DEV_TOOLS",
    "PERM_HUB_ADMIN",
    "AppExtension",
    "ChatPage",
]


def __getattr__(name: str):
    if name == "ChatPage":
        from llming_lodge.chat_page import ChatPage
        return ChatPage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
