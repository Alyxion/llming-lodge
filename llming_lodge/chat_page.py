"""NiceGUI integration for the llming-lodge chat page.

Provides :class:`ChatPage` — a reusable builder that host apps instantiate
once with app-level config, then call ``render()`` per request with user config.

Usage::

    from llming_lodge.chat_config import ChatAppConfig, ChatUserConfig
    from llming_lodge.chat_page import ChatPage

    chat_page = ChatPage(ChatAppConfig(
        accent_color="#003D8F",
        app_logo="/static/logo/Logo.svg",
        app_title="My Chat",
        show_budget=True,
    ))

    @ui.page("/chat")
    async def chat_route():
        # ... auth ...
        await chat_page.render(ChatUserConfig(
            user_id="u1", user_name="Alice", mcp_servers=[...],
        ))
"""

import asyncio
import hashlib
import logging
import os
from uuid import uuid4

from nicegui import ui, app

from llming_lodge.chat_config import (
    ChatAppConfig, ChatUserConfig, ChatFrontendConfig, ThemeConfig,
)
from llming_lodge.server import API_PREFIX

logger = logging.getLogger(__name__)


class ChatPage:
    """Builder for rendering the llming-lodge chat page inside NiceGUI."""

    def __init__(self, app_config: ChatAppConfig) -> None:
        self._app_config = app_config
        self._routes_registered = False

    def _ensure_routes(self) -> None:
        """Register static file routes and the API router (idempotent)."""
        if self._routes_registered:
            return
        self._routes_registered = True

        from llming_lodge.server import setup_routes

        debug = os.environ.get("DEBUG_CHAT_REMOTE", "0") == "1"
        setup_routes(app, debug=debug)

    async def render(self, user_config: ChatUserConfig) -> None:
        """Call from within a ``@ui.page`` handler after auth.

        Creates the controller, registers the session, injects the JS
        bundle + config, and wires up cleanup on disconnect.
        """
        self._ensure_routes()

        from llming_lodge.api import SessionRegistry, WebSocketChatController

        session_id = str(uuid4())
        ac = self._app_config

        # ── Create controller ────────────────────────────────
        quick_actions_dicts = None
        if user_config.quick_actions is not None:
            quick_actions_dicts = [qa.model_dump(by_alias=True) for qa in user_config.quick_actions]

        controller = WebSocketChatController(
            session_id=session_id,
            user_id=user_config.user_id,
            user_mail=user_config.user_email or None,
            budget_limits=user_config.budget_limits or None,
            context_preamble=user_config.context_preamble or None,
            mcp_servers=user_config.mcp_servers or None,
            initial_model=user_config.initial_model,
            user_avatar=user_config.user_avatar or None,
            budget_handler=user_config.budget_handler,
            quick_actions=quick_actions_dicts,
        )

        # ── Register in SessionRegistry ──────────────────────
        registry = SessionRegistry.get()
        entry = registry.register(
            session_id=session_id,
            controller=controller,
            user_id=user_config.user_id,
            user_name=user_config.full_name or user_config.user_name,
            user_avatar=user_config.user_avatar,
            mcp_servers=user_config.mcp_servers or None,
        )

        # ── MCP discovery (background) ───────────────────────
        mcp_servers = user_config.mcp_servers or []

        async def _discover_and_notify():
            await controller.discover_tools()
            if controller._ws:
                await controller._send({
                    "type": "tools_updated",
                    "tools": controller.get_all_known_tools(),
                })

        asyncio.create_task(_discover_and_notify())

        # ── Cleanup on disconnect ────────────────────────────
        async def _cleanup():
            logger.info(f"[CHAT] Cleaning up session {session_id}")
            registry.remove(session_id)
            if entry.upload_manager:
                entry.upload_manager.cleanup()
            for mcp in mcp_servers:
                if hasattr(mcp, "server_instance") and mcp.server_instance:
                    try:
                        if hasattr(mcp.server_instance, "close"):
                            await mcp.server_instance.close()
                    except Exception:
                        pass

        ui.context.client.on_disconnect(_cleanup)

        # ── Build __CHAT_CONFIG__ ────────────────────────────
        theme = ThemeConfig.from_hex(ac.accent_color) if ac.accent_color else None

        payload = ChatFrontendConfig(
            session_id=session_id,
            ws_path=f"{API_PREFIX}/ws/{session_id}",
            user_name=user_config.user_name,
            user_email=user_config.user_email,
            user_id=user_config.user_id,
            user_avatar=user_config.user_avatar,
            show_budget=ac.show_budget,
            app_logo=ac.app_logo,
            app_title=ac.app_title,
            app_mascot=ac.app_mascot,
            theme=theme,
        )
        config_json = payload.model_dump_json(by_alias=True)
        _cache_bust = hashlib.md5(session_id.encode()).hexdigest()[:8]

        # ── Inject HTML / CSS / JS ───────────────────────────
        ui.add_head_html(f"""
            <script>window.__CHAT_CONFIG__ = {config_json};</script>
            <link rel="stylesheet" href="/chat-static/chat-app.css?v={_cache_bust}">
            <link rel="stylesheet" href="/chat-static/vendor/katex.min.css">
            <script src="/chat-static/vendor/marked.min.js"></script>
            <script src="/chat-static/vendor/purify.min.js"></script>
            <script src="/chat-static/vendor/katex.min.js"></script>
        """)

        ui.add_css("""
            .nicegui-content { padding: 0 !important; display: none !important; }
            .q-page-container { height: 100vh; }
            .q-page { height: 100%; }
        """)

        ui.add_body_html(f"""
            <div id="chat-app" style="position:fixed;inset:0;z-index:9999;"></div>
            <script src="/chat-static/chat-app.js?v={_cache_bust}" defer></script>
        """)
