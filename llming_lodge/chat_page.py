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
import json
import logging
import os
from uuid import uuid4

from nicegui import ui, app

from llming_lodge.chat_config import (
    ChatAppConfig, ChatUserConfig, ChatFrontendConfig, ThemeConfig,
)
from llming_lodge.server import API_PREFIX

logger = logging.getLogger(__name__)

# ── Content-hash cache busting (computed once at import / hot-reload) ──

_file_hashes: dict[str, str] = {}


def _compute_file_hashes() -> dict[str, str]:
    """Walk the chat static dir and compute MD5 content hashes for JS/CSS."""
    from llming_lodge.server import get_chat_static_path
    hashes: dict[str, str] = {}
    chat_dir = get_chat_static_path()
    for root, _, files in os.walk(chat_dir):
        for fname in files:
            if fname.endswith(('.js', '.css')):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, chat_dir)
                with open(fpath, 'rb') as f:
                    hashes[rel] = hashlib.md5(f.read()).hexdigest()[:10]
    return hashes


def _hashed_url(path: str) -> str:
    """Return /chat-static/<path>?v=<content-hash> for cache busting."""
    global _file_hashes
    if not _file_hashes:
        _file_hashes = _compute_file_hashes()
    return f"/chat-static/{path}?v={_file_hashes.get(path, 'dev')}"


class ChatPage:
    """Builder for rendering the llming-lodge chat page inside NiceGUI."""

    def __init__(self, app_config: ChatAppConfig) -> None:
        self._app_config = app_config
        self._routes_registered = False
        self._nudge_store = None
        # Apply upload limits from config
        from llming_lodge.documents import UploadManager
        UploadManager.configure(
            max_file_size=app_config.max_file_size,
            max_session_size=app_config.max_session_size,
        )
        if app_config.nudge_mongo_uri and app_config.nudge_mongo_db:
            from llming_lodge.nudge_store import NudgeStore
            self._nudge_store = NudgeStore(
                app_config.nudge_mongo_uri, app_config.nudge_mongo_db,
            )

    @staticmethod
    def _build_nudge_categories(ac: ChatAppConfig, translations: dict) -> list[dict]:
        """Build the nudge categories list, always prepending 'general'."""
        general_label = translations.get("chat.nudge_category_general", "General")
        cats = [{"key": "general", "label": general_label, "icon": "", "priority": 10}]
        for c in ac.nudge_categories:
            d = c.model_dump(by_alias=True)
            if d["key"] != "general":  # skip if host app re-declared it
                cats.append(d)
        cats.sort(key=lambda c: c.get("priority", 0))
        return cats

    def _ensure_routes(self) -> None:
        """Register static file routes and the API router (idempotent)."""
        if self._routes_registered:
            return
        self._routes_registered = True

        from llming_lodge.server import setup_routes

        debug = os.environ.get("DEBUG_CHAT_REMOTE", "0") == "1"
        setup_routes(app, debug=debug, nudge_store=self._nudge_store)

    async def render(self, user_config: ChatUserConfig) -> None:
        """Call from within a ``@ui.page`` handler after auth.

        Creates the controller, registers the session, injects the JS
        bundle + config, and wires up cleanup on disconnect.
        """
        self._ensure_routes()

        from llming_lodge.api import SessionRegistry, WebSocketChatController

        session_id = str(uuid4())
        ac = self._app_config

        # ── Document plugin system ────────────────────────────
        from llming_lodge.doc_plugins import DocPluginManager
        doc_manager = DocPluginManager(
            enabled_types=user_config.doc_plugins,
            presentation_templates=user_config.presentation_templates or None,
            requires_providers=user_config.tool_requires_providers,
        )
        doc_mcp_configs = doc_manager.get_mcp_configs()

        # Merge doc plugin MCP configs with user-provided ones
        merged_mcp_servers = list(user_config.mcp_servers or []) + doc_mcp_configs

        # Auto-generate preamble from enabled doc plugins
        _base_preamble = user_config.context_preamble or ""
        _merged_preamble = _base_preamble + doc_manager.get_preamble()



        # ── Create controller ────────────────────────────────
        quick_actions_dicts = None
        if user_config.quick_actions is not None:
            quick_actions_dicts = [qa.model_dump(by_alias=True) for qa in user_config.quick_actions]

        controller = WebSocketChatController(
            session_id=session_id,
            user_id=user_config.user_id,
            user_mail=user_config.user_email or None,
            budget_limits=user_config.budget_limits or None,
            system_prompt=ac.default_system_prompt or None,
            context_preamble=_merged_preamble,
            mcp_servers=merged_mcp_servers,
            initial_model=user_config.initial_model,
            user_avatar=user_config.user_avatar or None,
            budget_handler=user_config.budget_handler,
            quick_actions=quick_actions_dicts,
            locale=user_config.locale or "en-us",
            on_language_change=user_config.on_language_change,
            on_action_callback=user_config.on_action_callback,
            tool_toggle_notifications=user_config.tool_toggle_notifications or {},
            supported_languages=ac.supported_languages,
            directory_service=user_config.directory_service,
            email_service=user_config.email_service,
            on_message_intercept=user_config.on_message_intercept,
            on_new_chat=user_config.on_new_chat,
        )

        # TTS config from app-level settings
        if ac.tts_voice:
            controller._tts_default_voice = ac.tts_voice
        if ac.tts_model:
            controller._tts_model = ac.tts_model
        controller._speech_max_tokens = ac.speech_max_tokens

        # Store translation overrides for language-change re-merge
        if user_config.translation_overrides:
            controller._translation_overrides = dict(user_config.translation_overrides)

        # Attach nudge store for server-side nudge operations
        controller._nudge_store = self._nudge_store
        controller._user_teams = user_config.user_teams or []
        controller._nudge_base_system_prompt = ac.nudge_base_system_prompt or ""
        controller._is_nudge_admin = user_config.is_admin

        # ── Nudges: master + discoverable ──
        controller._master_prompt = ""
        controller._master_cached_files = []

        async def _load_master_nudges():
            if not (self._nudge_store and user_config.user_email):
                return
            try:
                from llming_lodge.nudge_store import get_file_cache
                masters = user_config.prefetched_master_nudges
                if masters is None:
                    masters = await self._nudge_store.get_master_nudges(
                        user_config.user_email,
                        user_config.user_teams or [],
                    )
                if masters:
                    prompts = []
                    all_files = []
                    for m in masters:
                        sp = m.get("system_prompt", "")
                        if sp:
                            prompts.append(sp)
                        cached = await get_file_cache().get_files(
                            m["uid"], self._nudge_store,
                            user_config.user_email,
                            user_teams=user_config.user_teams,
                            nudge=m,
                        )
                        all_files.extend(cached)
                    controller._master_prompt = "\n\n".join(prompts)
                    controller._master_cached_files = all_files
                    logger.info(
                        "[MASTER] Loaded %d master nudge(s): %d prompt chars, %d files",
                        len(masters), len(controller._master_prompt), len(all_files),
                    )
            except Exception as e:
                logger.warning("[MASTER] Failed to load master nudges: %s", e)

        async def _load_discoverable_nudges():
            if not (self._nudge_store and user_config.user_email):
                return
            try:
                from llming_lodge.api.chat_session_api import (
                    _discoverable_sessions, _browser_mcp_sessions,
                )

                discoverable = user_config.prefetched_discoverable_nudges
                if discoverable is None:
                    discoverable = await self._nudge_store.get_discoverable_nudges(
                        user_config.user_email,
                        user_config.user_teams or [],
                    )
                if discoverable:
                    catalog_lines: list[str] = []
                    uid_set: set[str] = set()
                    mcp_uid_set: set[str] = set()
                    has_mcp_nudges = False
                    for nudge in discoverable:
                        uid = nudge["uid"]
                        uid_set.add(uid)
                        name = nudge.get("name", "Unnamed")
                        when = nudge.get("auto_discover_when", "")
                        has_js_files = any(
                            (f.get("name", "").endswith(".js") or f.get("name", "").endswith(".mjs"))
                            for f in nudge.get("files", [])
                        )
                        if has_js_files:
                            tool_names = nudge.get("mcp_tool_names", [])
                            tool_count = nudge.get("mcp_tool_count", len(tool_names))
                            catalog_lines.append(
                                f"- [MCP] {name} (uid: {uid}, {tool_count} tools) "
                                f"— Use when: {when}"
                            )
                            mcp_uid_set.add(uid)
                            has_mcp_nudges = True
                        else:
                            catalog_lines.append(f"- {name} (uid: {uid}) — Use when: {when}")

                    _discoverable_sessions[session_id] = {
                        "store": self._nudge_store,
                        "user_email": user_config.user_email,
                        "user_teams": user_config.user_teams or [],
                        "uids": uid_set,
                        "loop": asyncio.get_running_loop(),
                    }

                    # Build catalog text with instructions for both tool types
                    activation_instructions = (
                        "- For regular knowledge bases: call consult_nudge(uid)\n"
                        "- For [MCP] tool servers: call activate_mcp_nudge(uid) ONCE, "
                        "then use the server's tools directly"
                    ) if has_mcp_nudges else (
                        "Call consult_nudge with the matching uid from the list above."
                    )

                    catalog_text = (
                        "\n\n## Knowledge Bases — MANDATORY TOOL USE\n"
                        "CRITICAL RULE: When the user asks about any of the topics "
                        "listed below, you MUST use the appropriate tool "
                        "BEFORE generating your answer. Do NOT answer from your own knowledge. "
                        "The knowledge bases contain verified, company-specific information "
                        "that you cannot know otherwise. Answering without consulting the "
                        "tool will produce incorrect information.\n\n"
                        "Available knowledge bases:\n"
                        + "\n".join(catalog_lines)
                        + "\n\nHow to use:\n"
                        + activation_instructions
                    )
                    controller._auto_discover_catalog = catalog_text
                    controller.session._system_prompt_suffix = catalog_text

                    # Enable consult_nudge tool
                    if "consult_nudge" not in controller.enabled_tools:
                        controller.enabled_tools.append("consult_nudge")
                    if "consult_nudge" not in controller.available_tools:
                        controller.available_tools.append("consult_nudge")
                    controller.tool_config["consult_nudge"] = {"_session_id": session_id}

                    # Enable activate_mcp_nudge if there are MCP nudges
                    if has_mcp_nudges:
                        _browser_mcp_sessions[session_id] = {
                            "store": self._nudge_store,
                            "user_email": user_config.user_email,
                            "user_teams": user_config.user_teams or [],
                            "uids": uid_set,  # All UIDs (MCP check done in callback)
                            "loop": asyncio.get_running_loop(),
                            "controller": controller,
                            "pending_requests": {},
                            "active_mcp_nudges": set(),
                            "active_tool_names": {},
                        }
                        if "activate_mcp_nudge" not in controller.enabled_tools:
                            controller.enabled_tools.append("activate_mcp_nudge")
                        if "activate_mcp_nudge" not in controller.available_tools:
                            controller.available_tools.append("activate_mcp_nudge")
                        controller.tool_config["activate_mcp_nudge"] = {"_session_id": session_id}

                    controller.update_settings(
                        tools=controller.enabled_tools,
                        tool_config=controller.tool_config,
                    )

                    logger.info(
                        "[AUTO_DISCOVER] %d nudge(s) (%d MCP) available for session %s: %s",
                        len(discoverable), len(mcp_uid_set), session_id,
                        ", ".join(n.get("name", "?") for n in discoverable),
                    )
            except Exception as e:
                logger.warning("[AUTO_DISCOVER] Failed to load discoverable nudges: %s", e)

        await asyncio.gather(_load_master_nudges(), _load_discoverable_nudges())


        # ── Register in SessionRegistry ──────────────────────
        registry = SessionRegistry.get()
        entry = registry.register(
            session_id=session_id,
            controller=controller,
            user_id=user_config.user_id,
            user_name=user_config.full_name or user_config.user_name,
            user_avatar=user_config.user_avatar,
            mcp_servers=merged_mcp_servers,
        )
        entry.doc_manager = doc_manager

        # Register templates globally so PPTX export works from restored chats
        if doc_manager.presentation_templates:
            registry.register_templates(doc_manager.presentation_templates)

        # Store base preamble (without doc plugins) for regeneration on preset changes
        controller._base_context_preamble = _base_preamble

        # Wire document store notifications to WebSocket
        # Track which doc types already had their tools auto-enabled
        _auto_enabled_doc_types: set = set()

        def _doc_notify(event_type, doc):
            asyncio.ensure_future(controller._send({
                "type": event_type,
                "document": doc.model_dump(),
            }))
            # Auto-enable per-type editing tools on first document creation
            if event_type == "doc_created" and doc.type not in _auto_enabled_doc_types:
                _auto_enabled_doc_types.add(doc.type)
                asyncio.ensure_future(_auto_enable_doc_tools(doc.type))

        async def _auto_enable_doc_tools(doc_type: str):
            """Server-side auto-enable when the LLM creates a document via create_document tool.

            Complements client-side auto-enable (for fenced code blocks) and
            _auto_enable_restored_doc_tools (for conversation restore).
            """
            from llming_lodge.doc_plugins.manager import _MCP_SERVERS
            spec = _MCP_SERVERS.get(doc_type)
            if not spec:
                return
            group_label = spec["label"]
            server_groups = getattr(controller.session, '_mcp_server_groups', {})
            group = server_groups.get(group_label)
            if not group:
                return
            all_group_tools = group.get("tool_names", [])
            if any(tn in controller.enabled_tools for tn in all_group_tools):
                return  # Already enabled
            controller.toggle_tool(group_label, True)
            if controller._ws:
                await controller._send({
                    "type": "tools_updated",
                    "tools": controller.get_all_known_tools(),
                })
                await controller._send_context_info()

        doc_manager.store.set_notify_callback(_doc_notify)



        # ── Pre-collect MCP client renderers (before JS loads) ────
        _client_renderers = []
        for mcp_cfg in merged_mcp_servers:
            inst = getattr(mcp_cfg, 'server_instance', None)
            if inst:
                try:
                    _client_renderers.extend(await inst.get_client_renderers())
                except Exception:
                    pass



        # ── MCP discovery (background) ───────────────────────
        mcp_servers = merged_mcp_servers

        async def _discover_and_notify():
            await controller.discover_tools()
            # Re-ensure consult_nudge is enabled after discover_tools()
            # (discover_tools rebuilds available_tools and may affect enabled_tools)
            ad_catalog = getattr(controller, "_auto_discover_catalog", None)
            if ad_catalog:
                if "consult_nudge" not in controller.enabled_tools:
                    controller.enabled_tools.append("consult_nudge")
                if "consult_nudge" not in controller.available_tools:
                    controller.available_tools.append("consult_nudge")
                # Re-enable activate_mcp_nudge if browser MCP session exists
                from llming_lodge.api.chat_session_api import _browser_mcp_sessions
                if session_id in _browser_mcp_sessions:
                    if "activate_mcp_nudge" not in controller.enabled_tools:
                        controller.enabled_tools.append("activate_mcp_nudge")
                    if "activate_mcp_nudge" not in controller.available_tools:
                        controller.available_tools.append("activate_mcp_nudge")
                controller.session.config.tools = list(controller.enabled_tools)
                # Restore suffix on session (discover_tools may recreate session)
                controller.session._system_prompt_suffix = ad_catalog
                logger.info("[AUTO_DISCOVER] Re-enabled auto-discover tools after discover_tools")
            # Store MCP prompt hints on controller (survives session recreation)
            hints = controller.session.mcp_prompt_hints
            if hints:
                hint_block = "\n\n".join(hints)
                controller._mcp_prompt_hints_block = hint_block
                # Update controller.context_preamble so it persists across
                # session recreation (switch_model, preset changes, etc.)
                base = controller.context_preamble or ""
                controller.context_preamble = base + "\n\n" + hint_block
                controller.session._context_preamble = controller.context_preamble
                logger.info("[MCP] Prompt hints appended (%d chars)", len(hint_block))
            if controller._ws:
                await controller._send({
                    "type": "tools_updated",
                    "tools": controller.get_all_known_tools(),
                })
                # Send MCP client-side renderers for DocPluginRegistry
                renderers = controller.session.mcp_client_renderers
                if renderers:
                    await controller._send({
                        "type": "register_renderers",
                        "renderers": renderers,
                    })

        asyncio.create_task(_discover_and_notify())

        # ── Cleanup on disconnect ────────────────────────────
        async def _cleanup():
            logger.info(f"[CHAT] Cleaning up session {session_id}")
            registry.remove(session_id)
            # Clean up discoverable nudge session context
            from llming_lodge.api.chat_session_api import (
                _discoverable_sessions, _browser_mcp_sessions,
            )
            _discoverable_sessions.pop(session_id, None)
            # Clean up browser MCP session context and stop Workers
            bctx = _browser_mcp_sessions.pop(session_id, None)
            if bctx:
                for nudge_uid in list(bctx.get("active_mcp_nudges", set())):
                    try:
                        await controller._send({"type": "stop_browser_mcp", "nudge_uid": nudge_uid})
                    except Exception:
                        pass
                    for tool_name in bctx.get("active_tool_names", {}).get(nudge_uid, []):
                        from llming_lodge.tools.tool_registry import get_default_registry
                        get_default_registry().unregister(tool_name)
                        get_default_registry()._mcp_connections.pop(tool_name, None)
            if entry.upload_manager:
                entry.upload_manager.cleanup()
            await doc_manager.cleanup()
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

        from llming_lodge.i18n import get_translations
        locale = user_config.locale or "en-us"
        translations = get_translations(locale)
        if user_config.translation_overrides:
            translations.update(user_config.translation_overrides)

        payload = ChatFrontendConfig(
            session_id=session_id,
            ws_path=f"{API_PREFIX}/ws/{session_id}",
            user_name=user_config.user_name,
            full_name=user_config.full_name or user_config.user_name,
            user_email=user_config.user_email,
            user_id=user_config.user_id,
            user_avatar=user_config.user_avatar,
            show_budget=ac.show_budget,
            app_logo=ac.app_logo,
            app_logo_link=ac.app_logo_link,
            app_title=user_config.app_title or ac.app_title,
            app_mascot=ac.app_mascot,
            nudge_section_icon=ac.nudge_section_icon,
            theme=theme,
            locale=locale,
            fake_time=user_config.fake_time or "",
            banner_html=user_config.banner_html or "",
            speech_max_tokens=ac.speech_max_tokens,
            enable_voice_input=ac.enable_voice_input,
            enable_live_voice=ac.enable_live_voice,
            translations=translations,
            nudge_categories=self._build_nudge_categories(ac, translations),
            visibility_groups=[g.model_dump(by_alias=True) for g in ac.visibility_groups],
            teams=user_config.user_teams or [],
            doc_plugins=doc_manager.enabled_types,
            presentation_templates=[
                t.model_dump(by_alias=True) for t in (doc_manager.presentation_templates or [])
                if hasattr(t, 'model_dump')
            ],
            enforced_theme=user_config.enforced_theme or "",
            is_admin=user_config.is_admin,
            bg_logo_svg=user_config.bg_logo_svg,
            email_signature=user_config.email_signature or "",
        )
        config_json = payload.model_dump_json(by_alias=True).replace("</", r"<\/")  # escape for safe <script> embedding

        # ── Build CSS + JS file lists ────────────────────────
        # Core CSS (always loaded)
        _css_urls = [
            _hashed_url("chat-core.css"),
            _hashed_url("chat-sidebar.css"),
            _hashed_url("chat-messages.css"),
            _hashed_url("chat-input.css"),
            _hashed_url("chat-welcome.css"),
            "/chat-static/vendor/katex.min.css",
        ]
        # Feature CSS (always loaded — lightweight enough)
        _css_urls += [
            "/chat-static/vendor/codemirror/codemirror.min.css",
            "/chat-static/vendor/codemirror/lint.min.css",
            _hashed_url("chat-voice.css"),
            _hashed_url("chat-presets.css"),
            _hashed_url("chat-nudges.css"),
            _hashed_url("chat-documents.css"),
            _hashed_url("chat-search.css"),
        ]

        _vendor_scripts = [
            "/chat-static/vendor/marked.min.js",
            "/chat-static/vendor/purify.min.js",
            "/chat-static/vendor/katex.min.js",
            "/chat-static/vendor/codemirror/jshint.min.js",
            "/chat-static/vendor/codemirror/codemirror.min.js",
        ]
        # CodeMirror addons must load AFTER codemirror.min.js (they
        # reference the global CodeMirror object at evaluation time).
        _codemirror_addons = [
            "/chat-static/vendor/codemirror/javascript.min.js",
            "/chat-static/vendor/codemirror/matchbrackets.min.js",
            "/chat-static/vendor/codemirror/closebrackets.min.js",
            "/chat-static/vendor/codemirror/lint.min.js",
            "/chat-static/vendor/codemirror/javascript-lint.min.js",
        ]
        _plugin_scripts = [
            "/chat-static/plugins/doc-plugin-registry.js",
            "/chat-static/plugins/block-data-store.js",
            "/chat-static/plugins/builtin-plugins.js",
        ]
        # AI edit plugin loads in Phase 2 (needs ChatFeatures from chat-features.js,
        # which loads in Phase 1; must also load before builtin-plugins.js consumes it)
        _phase2_plugins = [
            "/chat-static/chat-popup-utils.js",
            "/chat-static/plugins/ai-edit-shared.js",
        ]

        # Chat scripts — loaded in order:
        # 1. Registry (first — declares ChatApp class + _ChatAppProto)
        # 2. Standalone classes
        # 3. Feature modules (add methods to _ChatAppProto)
        # 4. Core (applies proto, defines constructor/render/bindEvents, boots app)
        _chat_scripts = [
            _hashed_url("chat-features.js"),
            _hashed_url("chat-idb.js"),
            _hashed_url("chat-ws.js"),
            _hashed_url("chat-markdown.js"),
            _hashed_url("chat-sidebar.js"),
            _hashed_url("chat-messages.js"),
            _hashed_url("chat-images.js"),
            _hashed_url("chat-plus-menu.js"),
            _hashed_url("chat-voice.js"),
            _hashed_url("chat-realtime.js"),
            _hashed_url("chat-documents.js"),
            _hashed_url("chat-nudges.js"),
            _hashed_url("chat-presets.js"),
            _hashed_url("chat-search.js"),
            _hashed_url("chat-browser-mcp.js"),
            _hashed_url("chat-app-core.js"),  # must be last
        ]

        # ── Inject HTML / CSS / JS ───────────────────────────
        # Everything goes via run_javascript because when NiceGUI's socket
        # is already connected, add_head_html/add_body_html use
        # insertAdjacentHTML which does NOT execute <script> tags and may
        # not reliably insert elements into the live DOM.
        ui.run_javascript(f"""
            // -- CSS --
            {_css_urls!r}.forEach(href => {{
                if (!document.querySelector(`link[href="${{href}}"]`)) {{
                    const link = document.createElement('link');
                    link.rel = 'stylesheet';
                    link.href = href;
                    document.head.appendChild(link);
                }}
            }});
            // -- inline styles --
            (function() {{
                const s = document.createElement('style');
                s.textContent = `
                    .nicegui-content {{ padding: 0 !important; display: none !important; }}
                    .q-page-container {{ height: 100vh; }}
                    .q-page {{ height: 100%; }}
                `;
                document.head.appendChild(s);
            }})();
            // -- mount point --
            if (!document.getElementById('chat-app')) {{
                const div = document.createElement('div');
                div.id = 'chat-app';
                div.style.cssText = 'position:fixed;inset:0;z-index:9999;';
                document.body.appendChild(div);
            }}
            // -- config + scripts --
            window.__CHAT_CONFIG__ = {config_json};
            window.__CHAT_RENDERERS__ = {json.dumps(_client_renderers)};
            // -- preload avatar so the browser fetches it while JS loads --
            if (window.__CHAT_CONFIG__.userAvatar) {{
                const preload = document.createElement('link');
                preload.rel = 'preload';
                preload.as = 'image';
                preload.href = window.__CHAT_CONFIG__.userAvatar;
                document.head.appendChild(preload);
            }}
            (async () => {{
                const loadScript = (src) => new Promise((resolve, reject) => {{
                    const s = document.createElement('script');
                    s.src = src;
                    s.onload = resolve;
                    s.onerror = reject;
                    document.head.appendChild(s);
                }});
                const loadAll = (srcs) => Promise.all(srcs.map(loadScript));
                // Phase 1: vendor libs + foundation (independent, load in parallel)
                await loadAll([
                    ...{_vendor_scripts!r},
                    ...{_plugin_scripts[:-1]!r},
                    {_chat_scripts[0]!r},
                ]);
                // Phase 2: CodeMirror addons (need base from Phase 1) +
                //          ai-edit-shared + builtin-plugins + all feature modules (parallel)
                await loadAll([
                    ...{_codemirror_addons!r},
                    ...{_phase2_plugins!r},
                    {_plugin_scripts[-1]!r},
                    ...{_chat_scripts[1:-1]!r},
                ]);
                // Phase 3: core boot (must be last)
                await loadScript({_chat_scripts[-1]!r});
            }})();
        """)

