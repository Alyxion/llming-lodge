"""Static chat page builder for llming-lodge (framework-agnostic).

Provides :class:`ChatPage` — a reusable builder that host apps instantiate
once with app-level config, then call ``create_session()`` + ``build_html()``
per request.

**No NiceGUI dependency.** The HTML page is served statically with
content-hashed JS/CSS references.  All chat communication happens over
a plain WebSocket (``/api/llming/ws/{session_id}``).

Usage (standalone / FastAPI)::

    chat_page = ChatPage(ChatAppConfig(accent_color="#003D8F", ...))
    chat_page.setup(app)  # mount routes once at startup

    @app.get("/chat")
    async def chat_route(request: Request):
        session_id = await chat_page.create_session(ChatUserConfig(...))
        return chat_page.html_response(session_id)

Usage (NiceGUI host app — auth bridge)::

    # In @ui.page handler, after auth:
    session_id = await chat_page.create_session(user_config)
    token = sign_auth_token(session_id)
    ui.run_javascript(
        f'document.cookie="llming_auth={token};path=/;max-age=86400;samesite=lax";'
        f'location.replace("/api/llming/chat/{session_id}");'
    )
"""

import asyncio
import hashlib
import json
import logging
import os
from html import escape as html_escape
from typing import Optional
from uuid import uuid4

from llming_docs import DOC_ICONS, get_mcp_group_labels
from llming_lodge.chat_config import (
    ChatAppConfig, ChatUserConfig, ChatFrontendConfig, ThemeConfig,
)
from llming_lodge.server import API_PREFIX
from llming_com.auth import get_auth as _auth

logger = logging.getLogger(__name__)

# ── Content-hash cache busting ───────────────────────────────────

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


def invalidate_file_hashes() -> None:
    """Clear the hash cache so next ``_hashed_url()`` recomputes.

    Called by the dev file watcher when static files change.
    """
    global _file_hashes
    _file_hashes = {}


def _hashed_url(path: str) -> str:
    """Return /chat-static/<path>?v=<content-hash> for cache busting."""
    global _file_hashes
    if not _file_hashes:
        _file_hashes = _compute_file_hashes()
    return f"/chat-static/{path}?v={_file_hashes.get(path, 'dev')}"


# ── Session cleanup ─────────────────────────────────────────────

async def cleanup_session(session_id: str) -> None:
    """Full cleanup for a chat session.

    Called from the WebSocket disconnect handler.  Idempotent — safe to
    call multiple times (guarded by ``_cleanup_done`` flag on entry).
    """
    from llming_lodge.api.chat_session_api import (
        SessionRegistry, _discoverable_sessions, _browser_mcp_sessions,
    )

    registry = SessionRegistry.get()
    entry = registry.get_session(session_id)
    if not entry or entry._cleanup_done:
        return
    entry._cleanup_done = True

    logger.info(f"[CHAT] Cleaning up session {session_id}")
    registry.remove(session_id)

    # Clean up discoverable nudge session context
    _discoverable_sessions.pop(session_id, None)

    # Clean up browser MCP session context and stop Workers
    bctx = _browser_mcp_sessions.pop(session_id, None)
    if bctx:
        controller = entry.controller
        for nudge_uid in list(bctx.get("active_mcp_nudges", set())):
            try:
                await controller._send({"type": "stop_browser_mcp", "nudge_uid": nudge_uid})
            except Exception:
                pass
            for tool_name in bctx.get("active_tool_names", {}).get(nudge_uid, []):
                from llming_models.tools.tool_registry import get_default_registry
                get_default_registry().unregister(tool_name)
                get_default_registry()._mcp_connections.pop(tool_name, None)

    if entry.upload_manager:
        entry.upload_manager.cleanup()

    if entry.doc_manager:
        await entry.doc_manager.cleanup()

    for mcp in (entry.mcp_servers or []):
        if hasattr(mcp, "server_instance") and mcp.server_instance:
            try:
                if hasattr(mcp.server_instance, "close"):
                    await mcp.server_instance.close()
            except Exception:
                pass


# ── Dev file watcher ─────────────────────────────────────────────

_dev_watcher_started = False


async def start_dev_file_watcher() -> None:
    """Watch chat static files for changes and send reload to all WS clients.

    Only starts once.  Uses ``watchfiles`` (uvicorn dependency) if available,
    otherwise falls back to a polling approach.
    """
    global _dev_watcher_started
    if _dev_watcher_started:
        return
    _dev_watcher_started = True

    from llming_lodge.server import get_chat_static_path
    chat_dir = get_chat_static_path()

    async def _watch():
        try:
            import watchfiles
        except ImportError:
            logger.info("[DEV] watchfiles not installed — hot-reload disabled")
            return

        logger.info("[DEV] File watcher started for %s", chat_dir)
        async for changes in watchfiles.awatch(chat_dir):
            changed_files = [str(c[1]) for c in changes]
            logger.info("[DEV] Files changed: %s", changed_files)
            invalidate_file_hashes()

            # Notify all connected WebSocket clients to reload
            from llming_lodge.api.chat_session_api import SessionRegistry
            registry = SessionRegistry.get()
            for sid in list(registry._sessions):
                entry = registry._sessions.get(sid)
                if entry and entry.websocket:
                    try:
                        await entry.websocket.send_json({"type": "dev_reload"})
                    except Exception:
                        pass

    asyncio.create_task(_watch())


# ── HTML builder ─────────────────────────────────────────────────

def _build_css_urls() -> list[str]:
    """Build the list of CSS URLs with content hashes."""
    return [
        _hashed_url("vendor/material-icons.css"),
        _hashed_url("chat-core.css"),
        _hashed_url("chat-sidebar.css"),
        _hashed_url("chat-messages.css"),
        _hashed_url("chat-input.css"),
        _hashed_url("chat-welcome.css"),
        _hashed_url("vendor/katex.min.css"),
        _hashed_url("vendor/codemirror/codemirror.min.css"),
        _hashed_url("vendor/codemirror/lint.min.css"),
        _hashed_url("chat-voice.css"),
        _hashed_url("chat-presets.css"),
        _hashed_url("chat-nudges.css"),
        _hashed_url("chat-documents.css"),
        # Doc-type specific styles (plotly/table/text_doc/presentation/html/
        # email/latex) are owned by llming-docs; served from its static dir.
        "/doc-static/css/doc-plugins.css",
        _hashed_url("chat-search.css"),
        _hashed_url("chat-bolts.css"),
        _hashed_url("chat-followup.css"),
    ]


def _build_script_phases() -> tuple[list[str], list[str], str]:
    """Build the three script loading phases.

    Returns (phase1, phase2, phase3_single) where each is a list of
    hashed URLs.  Phase 1 and 2 load in parallel within the phase;
    Phase 3 is a single script that must load last.
    """
    vendor = [
        _hashed_url("vendor/marked.min.js"),
        _hashed_url("vendor/purify.min.js"),
        _hashed_url("vendor/katex.min.js"),
        _hashed_url("vendor/codemirror/jshint.min.js"),
        _hashed_url("vendor/codemirror/codemirror.min.js"),
    ]
    # Plugin foundation — entirely owned by llming-docs. The registry, the
    # cross-block reference store (including per-type compatibility shims),
    # and the AI-edit helper are served from /doc-static/. Host has no
    # format-specific code.
    plugin_foundation = [
        "/doc-static/plugins/doc-plugin-registry.js",
        "/doc-static/plugins/block-data-store.js",
    ]
    features_first = _hashed_url("chat-features.js")

    llming_com_ws = "/llming-com/llming-ws.js"
    phase1 = vendor + plugin_foundation + [llming_com_ws, features_first]

    codemirror_addons = [
        _hashed_url("vendor/codemirror/javascript.min.js"),
        _hashed_url("vendor/codemirror/matchbrackets.min.js"),
        _hashed_url("vendor/codemirror/closebrackets.min.js"),
        _hashed_url("vendor/codemirror/lint.min.js"),
        _hashed_url("vendor/codemirror/javascript-lint.min.js"),
    ]
    phase2_plugins = [
        _hashed_url("chat-popup-utils.js"),
        # ai-edit-shared — owned by llming-docs (shared by text_doc + email_draft
        # plugins). Served from /doc-static/.
        "/doc-static/plugins/ai-edit-shared.js",
        _hashed_url("chat-followup.js"),
    ]
    builtin_plugins = _hashed_url("plugins/builtin-plugins.js")
    # Document-type plugins are owned by llming-docs and served from its
    # static dir at /doc-static/. Loaded in Phase 2 so chat-app-core's
    # boot (Phase 3) can call window.registerLlmingDocPlugins(registry)
    # alongside the host's own registerBuiltinPlugins.
    doc_plugins = "/doc-static/plugins/doc-plugins.js"
    bolt_apps = [
        _hashed_url("system-droplets/math/calculator.js"),
        _hashed_url("system-droplets/timekeeper/timekeeper.js"),
    ]
    feature_modules = [
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
        _hashed_url("chat-bolts.js"),
        _hashed_url("chat-app-extensions.js"),
    ]

    phase2 = codemirror_addons + phase2_plugins + [builtin_plugins, doc_plugins] + bolt_apps + feature_modules

    phase3 = _hashed_url("chat-app-core.js")

    return phase1, phase2, phase3


def build_chat_html(
    config_json: str,
    renderers_json: str = "[]",
    app_title: str = "Chat",
) -> str:
    """Build the complete chat HTML page.

    Args:
        config_json: JSON string for ``window.__CHAT_CONFIG__``
        renderers_json: JSON string for ``window.__CHAT_RENDERERS__``
        app_title: Page ``<title>``

    Returns:
        Complete HTML document as string.
    """
    css_urls = _build_css_urls()
    phase1, phase2, phase3 = _build_script_phases()

    css_links = "\n    ".join(
        f'<link rel="stylesheet" href="{url}">' for url in css_urls
    )

    # Escape config for safe embedding in <script>
    safe_config = config_json.replace("</", r"<\/")
    safe_renderers = renderers_json.replace("</", r"<\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html_escape(app_title)}</title>
    {css_links}
    <style>html, body {{ margin: 0; padding: 0; height: 100%; background: #1a1a2e; }}</style>
    <script>
        window.__CHAT_CONFIG__ = {safe_config};
        window.__CHAT_RENDERERS__ = {safe_renderers};
    </script>
</head>
<body>
    <div id="chat-app" style="position:fixed;inset:0;z-index:9999;"></div>
    <script>
    (async () => {{
        const loadScript = (src) => new Promise((resolve, reject) => {{
            const s = document.createElement('script');
            s.src = src;
            s.onload = resolve;
            s.onerror = reject;
            document.head.appendChild(s);
        }});
        const loadAll = (srcs) => Promise.all(srcs.map(loadScript));
        // Phase 1: vendor libs + foundation (parallel)
        await loadAll({json.dumps(phase1)});
        // Phase 2: addons + plugins + feature modules (parallel, after Phase 1)
        await loadAll({json.dumps(phase2)});
        // Phase 3: core boot (must be last)
        await loadScript({json.dumps(phase3)});
    }})();
    </script>
</body>
</html>"""


class ChatPage:
    """Builder for the llming-lodge chat page.

    Framework-agnostic — works with FastAPI, Starlette, or any ASGI app.
    No NiceGUI dependency.
    """

    def __init__(self, app_config: ChatAppConfig) -> None:
        self._app_config = app_config
        self._routes_registered = False
        self._nudge_store = None
        self._session_factory = None
        self._auth_handler = None
        self._page_path = "/chat"
        self._auth_path = "/chat-auth"
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

    def set_session_factory(self, factory) -> None:
        """Set the session factory callback.

        The factory is called when a user visits ``/chat`` with a valid
        identity cookie but no active session.  It receives a Starlette
        ``Request`` and must return a ``session_id`` (from
        ``create_session()``) or ``None`` if authentication failed.

        Signature: ``async (request: Request) -> str | None``
        """
        self._session_factory = factory

    def set_auth_handler(self, handler) -> None:
        """Set the auth handler for unauthenticated users.

        Called when no session exists and the factory returns None
        (or no identity cookie).  Should initiate OAuth or redirect.

        Signature: ``async (request: Request) -> Response``
        """
        self._auth_handler = handler

    @staticmethod
    def _build_nudge_categories(ac: ChatAppConfig, translations: dict) -> list[dict]:
        """Build the nudge categories list, always prepending 'general'."""
        general_label = translations.get("chat.nudge_category_general", "General")
        cats = [{"key": "general", "label": general_label, "icon": "", "priority": 10}]
        for c in ac.nudge_categories:
            d = c.model_dump(by_alias=True)
            if d["key"] != "general":
                cats.append(d)
        cats.sort(key=lambda c: c.get("priority", 0))
        return cats

    def setup(
        self, app, *,
        debug: Optional[bool] = None,
        page_path: str = "/chat",
        auth_path: str = "/chat-auth",
    ) -> None:
        """Mount all routes on the given app (idempotent).

        Call this once at app startup, **before** any catch-all page handlers
        (e.g. NiceGUI's ``@ui.page``).

        Registers:
        - Static files (``/llming-static``, ``/chat-static``)
        - HTTP + WebSocket API router (``/api/llming/...``)
        - Debug API, dev dashboard, public API (if enabled)
        - Chat page route at ``page_path`` (default ``/chat``)

        Args:
            app: FastAPI / Starlette application
            debug: Enable debug API (default: auto-detect from env)
            page_path: URL path for the chat page (default ``/chat``)
            auth_path: URL path the user is redirected to when not
                authenticated or session expired (default ``/chat-auth``).
                The host app should register its auth handler here.
        """
        if self._routes_registered:
            return
        self._routes_registered = True
        self._page_path = page_path
        self._auth_path = auth_path

        from llming_lodge.server import setup_routes

        if debug is None:
            debug = os.environ.get("DEBUG_CHAT_REMOTE", "0") == "1"
        setup_routes(app, debug=debug, nudge_store=self._nudge_store)

        # Register the chat page HTML route
        self._register_chat_page_route(app)

    def ensure_routes(self, app=None) -> None:
        """Backward-compatible alias for ``setup()``.

        If ``app`` is provided, mounts routes on it.
        If not provided, attempts to use the app from the first call.
        """
        if self._routes_registered:
            return
        if app is None:
            raise RuntimeError(
                "ChatPage.ensure_routes() requires an explicit 'app' argument "
                "now that NiceGUI is no longer used. Use chat_page.setup(app) instead."
            )
        self.setup(app)

    def _register_chat_page_route(self, app) -> None:
        """Register chat page routes on the app.

        Uses ``app.routes.insert(0, ...)`` to ensure the chat page route
        takes priority over any framework catch-all (e.g. NiceGUI's SPA
        middleware).  A plain ``@app.get()`` would be shadowed.

        Routes:
        1. ``GET {page_path}`` (default ``/chat``) — the user-facing URL.
        2. ``GET /api/llming/chat-page/{session_id}`` — sets session cookie
           and redirects to the clean URL.
        """
        from starlette.routing import Route
        from starlette.requests import Request
        from starlette.responses import HTMLResponse, Response, RedirectResponse
        _auth_mgr = _auth()
        # Cookie names are sourced from the AuthManager instance so they
        # respect the host's per-app prefix (e.g. "myapp_session").
        SESSION_COOKIE_NAME = _auth_mgr.session_cookie_name
        AUTH_COOKIE_NAME = _auth_mgr.auth_cookie_name
        IDENTITY_COOKIE_NAME = _auth_mgr.identity_cookie_name
        from llming_lodge.api.chat_session_api import SessionRegistry

        page_path = self._page_path
        auth_path = self._auth_path
        _factory = self._session_factory

        async def serve_chat_page(request: Request):
            """Serve the chat page at the clean URL.

            Fast path: valid ``llming_session`` cookie → serve immediately.
            Slow path: valid ``llming_identity`` cookie → call session_factory
            to create a new session, set cookies, serve HTML.
            Fallback: redirect to ``auth_path`` for NiceGUI OAuth.
            """
            try:
                return await _serve_chat_page_impl(request)
            except Exception as e:
                from llming_com import error_response
                return error_response(e, request_path=request.url.path)

        async def _serve_chat_page_impl(request: Request):
            registry = SessionRegistry.get()

            # ── Fast path: existing session ──
            session_id = request.cookies.get(SESSION_COOKIE_NAME, "")
            if session_id:
                entry = registry.get_session(session_id)
                if entry:
                    config_json = getattr(entry, '_frontend_config_json', '{}')
                    renderers_json = getattr(entry, '_frontend_renderers_json', '[]')
                    app_title = getattr(entry, '_app_title', 'Chat')
                    return HTMLResponse(build_chat_html(config_json, renderers_json, app_title))

            # ── Slow path: create session via factory ──
            if _factory:
                try:
                    new_session_id = await _factory(request)
                except Exception as exc:
                    logger.error("[CHAT] session_factory failed: %s", exc, exc_info=True)
                    new_session_id = None

                if new_session_id:
                    entry = registry.get_session(new_session_id)
                    if entry:
                        config_json = getattr(entry, '_frontend_config_json', '{}')
                        renderers_json = getattr(entry, '_frontend_renderers_json', '[]')
                        app_title = getattr(entry, '_app_title', 'Chat')

                        response = HTMLResponse(build_chat_html(config_json, renderers_json, app_title))
                        _secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
                        # Set session cookie so next reload is the fast path
                        response.set_cookie(
                            SESSION_COOKIE_NAME, new_session_id,
                            path="/", max_age=86400, samesite="lax", httponly=True,
                        )
                        # Ensure auth cookie is set
                        if not _auth_mgr.verify_auth_cookie(request):
                            token = _auth_mgr.sign_auth_token(new_session_id)
                            response.set_cookie(
                                AUTH_COOKIE_NAME, token,
                                path="/", max_age=604800, samesite="lax",
                            )
                        # Always refresh the identity cookie so other pages
                        # (hub, globe, mail) can find the tokens in Redis
                        identity_sid = _auth_mgr.verify_identity_cookie(request)
                        if identity_sid:
                            response.set_cookie(
                                IDENTITY_COOKIE_NAME, _auth_mgr.sign_identity_token(identity_sid),
                                path="/", max_age=604800, samesite="lax",
                                secure=_secure, httponly=True,
                            )
                        return response

            # ── Fallback: redirect to auth ──
            # Only redirect if we haven't already tried recently (prevent loops).
            # The llming_next cookie presence means we already tried and came back.
            if request.cookies.get("llming_next"):
                # Already tried auth → factory still failed → show error
                from starlette.responses import PlainTextResponse
                logger.error("[CHAT] Auth loop detected — factory keeps failing")
                response = PlainTextResponse(
                    "Chat session could not be created. Please visit the home page first, "
                    "then navigate to /chat.",
                    status_code=503,
                )
                response.delete_cookie("llming_next", path="/")
                return response

            # Redirect to root (hub) for OAuth — no NiceGUI auth page needed
            response = RedirectResponse("/", status_code=302)
            response.set_cookie("llming_next", page_path, path="/", max_age=300, samesite="lax")
            response.delete_cookie(SESSION_COOKIE_NAME, path="/")
            return response

        async def serve_chat_set_cookie(request: Request):
            """Set session cookie and redirect to clean URL.

            Called after auth creates a session.  The session_id path
            parameter is stored as a cookie so subsequent requests to
            ``page_path`` can find the session without URL parameters.
            """
            session_id = request.path_params["session_id"]
            if not _auth_mgr.verify_auth_cookie(request):
                return Response(status_code=401, content="Not authenticated")

            registry = SessionRegistry.get()
            entry = registry.get_session(session_id)
            if not entry:
                return Response(status_code=404, content="Session not found or expired")

            response = RedirectResponse(page_path, status_code=302)
            response.set_cookie(
                SESSION_COOKIE_NAME, session_id,
                path="/", max_age=86400, samesite="lax", httponly=True,
            )
            return response

        # Insert at position 0 so these routes take priority over
        # any catch-all middleware/page handler (e.g. NiceGUI).
        app.routes.insert(0, Route(page_path, serve_chat_page, methods=["GET"]))
        app.routes.insert(1, Route(
            f"{API_PREFIX}/chat-page/{{session_id}}",
            serve_chat_set_cookie, methods=["GET"],
        ))

    async def create_session(self, user_config: ChatUserConfig) -> str:
        """Create a chat session with controller, nudges, and MCP servers.

        Returns the session_id.  The session is registered in the
        SessionRegistry and ready for WebSocket connection.

        The HTML page can be obtained via ``build_html(session_id)`` or
        served by the built-in route at ``/api/llming/chat/{session_id}``.
        """
        from llming_lodge.api import SessionRegistry, WebSocketChatController

        session_id = str(uuid4())
        ac = self._app_config

        # ── App extensions (lazy) ─────────────────────────────
        from llming_lodge.app_extensions import AppExtensionManager
        ext_manager = AppExtensionManager(user_config.app_extensions or None)

        # ── Document plugin system ────────────────────────────
        from llming_docs import DocPluginManager
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
        from llming_lodge.chat_config import PERM_NUDGE_ADMIN, PERM_DEV_TOOLS
        _perms = user_config.permissions or set()
        controller._is_nudge_admin = PERM_NUDGE_ADMIN in _perms
        controller._is_dev = PERM_DEV_TOOLS in _perms
        controller._budget_limits_for_user = user_config.budget_limits_for_user
        controller._system_nudge_keys = ac.system_nudges
        controller._app_ext_manager = ext_manager

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
                    controller._master_nudges_raw = masters
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
                # Include system nudges in discoverable
                from llming_lodge.system_nudges import SYSTEM_NUDGE_REGISTRY
                for sys_key, sys_nudge in SYSTEM_NUDGE_REGISTRY.items():
                    # Filter by email patterns
                    patterns = sys_nudge.get("email_patterns", [])
                    if patterns:
                        email = user_config.user_email or ""
                        import fnmatch
                        if not any(fnmatch.fnmatch(email.lower(), p.lower()) for p in patterns):
                            continue
                    # Only include if it has server_mcp or auto_activate_keywords
                    caps = sys_nudge.get("capabilities") or {}
                    if caps.get("server_mcp") or sys_nudge.get("auto_activate_keywords"):
                        uid = sys_nudge.get("uid", f"sys:{sys_key}")
                        discoverable.append({
                            "uid": uid,
                            "name": sys_nudge.get("name", sys_key),
                            "auto_discover_when": sys_nudge.get("description", ""),
                            "files": [],
                            "_system_nudge": True,
                            "_has_server_mcp": bool(caps.get("server_mcp")),
                        })

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
                        # System nudges with server_mcp are also MCP-activated
                        if nudge.get("_has_server_mcp"):
                            has_js_files = True
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

                    if "consult_nudge" not in controller.enabled_tools:
                        controller.enabled_tools.append("consult_nudge")
                    if "consult_nudge" not in controller.available_tools:
                        controller.available_tools.append("consult_nudge")
                    controller.tool_config["consult_nudge"] = {"_session_id": session_id}

                    if has_mcp_nudges:
                        _browser_mcp_sessions[session_id] = {
                            "store": self._nudge_store,
                            "user_email": user_config.user_email,
                            "user_teams": user_config.user_teams or [],
                            "uids": uid_set,
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
        entry = registry.register_session(
            session_id=session_id,
            controller=controller,
            user_id=user_config.user_id,
            user_name=user_config.full_name or user_config.user_name,
            user_avatar=user_config.user_avatar,
            mcp_servers=merged_mcp_servers,
        )
        entry.doc_manager = doc_manager
        entry.base_preamble = _base_preamble
        controller._doc_store = doc_manager.store  # For sync_document_context

        # Register templates globally so PPTX export works from restored chats
        if doc_manager.presentation_templates:
            registry.register_templates(doc_manager.presentation_templates)

        # Store base preamble (without doc plugins) for regeneration on preset changes
        controller._base_context_preamble = _base_preamble

        # Wire document store notifications to WebSocket
        _auto_enabled_doc_types: set = set()
        _unified_editor_enabled: bool = False

        def _doc_notify(event_type, doc):
            nonlocal _unified_editor_enabled
            # Rebuild Current Documents inventory in the preamble. Without
            # this the LLM loses track of existing doc ids across turns and
            # falls back to create_document when it should call update_document.
            from llming_lodge.api.chat_session_api import _refresh_doc_preamble
            _refresh_doc_preamble(controller, entry)
            asyncio.ensure_future(controller._send({
                "type": event_type,
                "document": doc.model_dump(),
            }))
            if event_type == "doc_created":
                if doc.type not in _auto_enabled_doc_types:
                    _auto_enabled_doc_types.add(doc.type)
                    asyncio.ensure_future(_auto_enable_doc_tools(doc.type))
                # Enable the type-agnostic Document Editor MCP on first doc,
                # so `update_document` / `read_document` / `undo_document` are
                # available immediately — without them the LLM has no edit
                # path and falls back to calling create_document again.
                if not _unified_editor_enabled:
                    _unified_editor_enabled = True
                    asyncio.ensure_future(_auto_enable_group("Document Editor"))

        async def _auto_enable_doc_tools(doc_type: str):
            from llming_docs.manager import _MCP_SERVERS
            spec = _MCP_SERVERS.get(doc_type)
            if not spec:
                return
            await _auto_enable_group(spec["label"])

        async def _auto_enable_group(group_label: str):
            server_groups = getattr(controller.session, '_mcp_server_groups', {})
            group = server_groups.get(group_label)
            if not group:
                return
            all_group_tools = group.get("tool_names", [])
            if any(tn in controller.enabled_tools for tn in all_group_tools):
                return
            controller.toggle_tool(group_label, True)
            if controller._ws:
                await controller._send({
                    "type": "tools_updated",
                    "tools": controller.get_all_known_tools(),
                })
                await controller._send_context_info()

        doc_manager.store.set_notify_callback(_doc_notify)

        # ── Pre-collect MCP client renderers ────
        _client_renderers = []
        for mcp_cfg in merged_mcp_servers:
            inst = getattr(mcp_cfg, 'server_instance', None)
            if inst:
                try:
                    _client_renderers.extend(await inst.get_client_renderers())
                except Exception:
                    pass

        # ── MCP discovery (background) ───────────────────────
        async def _discover_and_notify():
            await controller.discover_tools()
            ad_catalog = getattr(controller, "_auto_discover_catalog", None)
            if ad_catalog:
                if "consult_nudge" not in controller.enabled_tools:
                    controller.enabled_tools.append("consult_nudge")
                if "consult_nudge" not in controller.available_tools:
                    controller.available_tools.append("consult_nudge")
                from llming_lodge.api.chat_session_api import _browser_mcp_sessions
                if session_id in _browser_mcp_sessions:
                    if "activate_mcp_nudge" not in controller.enabled_tools:
                        controller.enabled_tools.append("activate_mcp_nudge")
                    if "activate_mcp_nudge" not in controller.available_tools:
                        controller.available_tools.append("activate_mcp_nudge")
                controller.session.config.tools = list(controller.enabled_tools)
                controller.session._system_prompt_suffix = ad_catalog
                logger.info("[AUTO_DISCOVER] Re-enabled auto-discover tools after discover_tools")
            hints = controller.session.mcp_prompt_hints
            if hints:
                hint_block = "\n\n".join(hints)
                controller._mcp_prompt_hints_block = hint_block
                base = controller.context_preamble or ""
                controller.context_preamble = base + "\n\n" + hint_block
                controller.session._context_preamble = controller.context_preamble
                logger.info("[MCP] Prompt hints appended (%d chars)", len(hint_block))
            if controller._ws:
                await controller._send({
                    "type": "tools_updated",
                    "tools": controller.get_all_known_tools(),
                })
                renderers = controller.session.mcp_client_renderers
                if renderers:
                    await controller._send({
                        "type": "register_renderers",
                        "renderers": renderers,
                    })

        asyncio.create_task(_discover_and_notify())

        # ── Build frontend config and store on entry ─────────
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
            app_mascot_incognito=ac.app_mascot_incognito,
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
            supported_languages=ac.supported_languages,
            visibility_groups=[g.model_dump(by_alias=True) for g in ac.visibility_groups],
            teams=user_config.user_teams or [],
            doc_plugins=doc_manager.enabled_types,
            presentation_templates=[
                t.model_dump(by_alias=True) for t in (doc_manager.presentation_templates or [])
                if hasattr(t, 'model_dump')
            ],
            enforced_theme=user_config.enforced_theme or "",
            permissions=sorted(user_config.permissions or set()),
            bg_logo_svg=user_config.bg_logo_svg,
            email_signature=user_config.email_signature or "",
            bolt_label=ac.bolt_label,
            app_extensions=ext_manager.get_manifests(),
            doc_icons=dict(DOC_ICONS),
            doc_group_labels=get_mcp_group_labels(),
        )

        config_json = payload.model_dump_json(by_alias=True)
        renderers_json = json.dumps(_client_renderers)
        app_title = user_config.app_title or ac.app_title or "Chat"

        # Store on entry so the HTML route can retrieve them
        entry._frontend_config_json = config_json
        entry._frontend_renderers_json = renderers_json
        entry._app_title = app_title

        return session_id

    def build_html(self, session_id: str) -> str:
        """Build the HTML page for a session that was created with ``create_session()``.

        Returns the complete HTML document as a string.
        """
        from llming_lodge.api.chat_session_api import SessionRegistry
        entry = SessionRegistry.get().get_session(session_id)
        if not entry:
            raise ValueError(f"Session {session_id} not found")

        config_json = getattr(entry, '_frontend_config_json', '{}')
        renderers_json = getattr(entry, '_frontend_renderers_json', '[]')
        app_title = getattr(entry, '_app_title', 'Chat')

        return build_chat_html(config_json, renderers_json, app_title)

    def html_response(self, session_id: str):
        """Return a Starlette ``HTMLResponse`` for the given session.

        Convenience wrapper around ``build_html()``.
        """
        from starlette.responses import HTMLResponse
        return HTMLResponse(self.build_html(session_id))

    # ── Backward compatibility ────────────────────────────────

    async def render(self, user_config: ChatUserConfig) -> str:
        """Create session and return session_id.

        This is the backward-compatible entry point.  Previously injected
        HTML via NiceGUI; now returns the session_id so the caller can
        redirect to the chat page route.

        For host apps still using ``@ui.page``, do::

            session_id = await chat_page.render(user_config)
            token = sign_auth_token(session_id)
            ui.run_javascript(
                f'document.cookie="llming_auth={token};path=/;max-age=86400;samesite=lax";'
                f'location.replace("/api/llming/chat/{session_id}");'
            )
        """
        return await self.create_session(user_config)
