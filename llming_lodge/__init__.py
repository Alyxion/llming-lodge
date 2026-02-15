"""llming-lodge — chat interactions package."""

from llming_lodge.llm_base_models import ChatHistory, ChatMessage, Role
from llming_lodge.session import ChatSession, LLMConfig
from llming_lodge.llm_provider_manager import LLMManager
from llming_lodge.providers.llm_provider_models import LLMInfo, ModelSize
from llming_lodge.config import LLMGlobalConfig, LLMUserConfig
from llming_lodge.budget import MongoDBBudgetLimit
from llming_lodge.budget.budget_types import LimitPeriod
from llming_lodge.budget.budget_limit import BudgetLimit
from llming_lodge.chat_config import (
    ChatAppConfig, ChatUserConfig, QuickAction, ThemeConfig, ChatFrontendConfig,
)

__all__ = [
    "ChatSession",
    "ChatHistory",
    "ChatMessage",
    "Role",
    "LLMConfig",
    "LLMManager",
    "ModelSize",
    "LLMInfo",
    "MongoDBBudgetLimit",
    "LimitPeriod",
    "BudgetLimit",
    "LLMGlobalConfig",
    "LLMUserConfig",
    "ChatAppConfig",
    "ChatUserConfig",
    "QuickAction",
    "ThemeConfig",
    "ChatFrontendConfig",
    "ChatPage",
]


def __getattr__(name: str):
    if name == "ChatPage":
        from llming_lodge.chat_page import ChatPage
        return ChatPage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
