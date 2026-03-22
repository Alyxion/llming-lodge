"""App Extensions — lazy-loaded extensions for the chat page.

Extensions are registered per-session via ``ChatUserConfig.app_extensions``.
They are **not** instantiated or activated until the client explicitly requests
activation via WebSocket, ensuring zero overhead for unused extensions.

Usage::

    from llming_lodge.app_extensions import AppExtension

    class WeatherExtension(AppExtension):
        name = "weather"
        label = "Weather Widget"
        icon = "cloud"
        script_url = "/static/extensions/weather.js"

        async def on_activate(self, controller):
            return {"api_key": "...", "default_city": "Berlin"}

        async def on_message(self, controller, payload):
            city = payload.get("city", "Berlin")
            return {"temp": 22, "city": city}

        async def on_deactivate(self, controller):
            pass

Then in your chat page setup::

    await chat_page.render(ChatUserConfig(
        ...,
        app_extensions=[WeatherExtension()],
    ))

Client-side JS at ``script_url`` registers the UI counterpart::

    ChatAppExtensions.define('weather', {
        activate(app, config) { /* build UI, config has api_key etc. */ },
        handleMessage(app, msg) { /* handle server responses */ },
        deactivate(app) { /* tear down UI */ },
    });
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llming_lodge.api.chat_session_api import WebSocketChatController

logger = logging.getLogger(__name__)


class AppExtension(ABC):
    """Base class for chat page app extensions.

    Subclass this, set class attributes, and override the async hooks.
    Instances are attached per-session; activation is lazy (on demand).
    """

    #: Unique extension identifier (must match the client-side define() name).
    name: str = ""
    #: Human-readable label shown in the UI.
    label: str = ""
    #: Material icon name (e.g. "cloud", "bar_chart").
    icon: str = "extension"
    #: URL to the client-side JS file, loaded on demand when activated.
    #: Leave empty if the client script is bundled statically.
    script_url: str = ""

    async def on_activate(self, controller: WebSocketChatController) -> dict | None:
        """Called when this extension is activated for a session.

        Return a config dict that will be sent to the client, or None.
        This is the place to allocate resources, open connections, etc.
        """
        return None

    async def on_message(
        self, controller: WebSocketChatController, payload: dict,
    ) -> dict | None:
        """Handle a message from the client for this extension.

        ``payload`` is the arbitrary dict sent by the client.
        Return a response dict (sent back as ``app_ext:message``), or None.
        """
        return None

    async def on_deactivate(self, controller: WebSocketChatController) -> None:
        """Called when this extension is deactivated or the session ends.

        Clean up resources here.
        """

    # Bolt definitions — slash commands registered at page load (before activation)
    bolts: list[dict[str, Any]] = []

    def get_manifest(self) -> dict[str, Any]:
        """Return the metadata dict sent to the client on session_init.

        The client uses this to know which extensions are available and
        where to load their scripts from — but does NOT load them yet.
        """
        result = {
            "name": self.name,
            "label": self.label,
            "icon": self.icon,
            "scriptUrl": self.script_url,
        }
        if self.bolts:
            result["bolts"] = self.bolts
        return result


class AppExtensionManager:
    """Per-session manager that holds registered extensions and tracks activation.

    Instantiated once per chat session from the list of ``AppExtension``
    instances provided in ``ChatUserConfig.app_extensions``.
    """

    def __init__(self, extensions: list[AppExtension] | None = None) -> None:
        #: All registered extensions, keyed by name.
        self._registry: dict[str, AppExtension] = {}
        #: Set of currently activated extension names.
        self._active: set[str] = set()

        for ext in extensions or []:
            if not ext.name:
                logger.warning("[APP_EXT] Skipping extension with empty name: %r", ext)
                continue
            if ext.name in self._registry:
                logger.warning("[APP_EXT] Duplicate extension name %r — skipping", ext.name)
                continue
            self._registry[ext.name] = ext

    # ── Query ──────────────────────────────────────────────

    def get(self, name: str) -> AppExtension | None:
        return self._registry.get(name)

    def is_active(self, name: str) -> bool:
        return name in self._active

    def get_manifests(self) -> list[dict[str, Any]]:
        """Return manifests for all registered extensions (for frontend config)."""
        return [ext.get_manifest() for ext in self._registry.values()]

    # ── Lifecycle ──────────────────────────────────────────

    async def activate(
        self, name: str, controller: WebSocketChatController,
    ) -> dict | None:
        """Activate an extension by name. Returns config dict or None."""
        ext = self._registry.get(name)
        if not ext:
            logger.warning("[APP_EXT] activate: unknown extension %r", name)
            return None
        if name in self._active:
            logger.debug("[APP_EXT] %r already active", name)
            return None
        try:
            config = await ext.on_activate(controller)
        except Exception as e:
            logger.error("[APP_EXT] on_activate(%r) failed: %s", name, e)
            return None
        self._active.add(name)
        logger.info("[APP_EXT] Activated %r", name)
        return config

    async def deactivate(
        self, name: str, controller: WebSocketChatController,
    ) -> None:
        """Deactivate an extension by name."""
        ext = self._registry.get(name)
        if not ext or name not in self._active:
            return
        try:
            await ext.on_deactivate(controller)
        except Exception as e:
            logger.error("[APP_EXT] on_deactivate(%r) failed: %s", name, e)
        self._active.discard(name)
        logger.info("[APP_EXT] Deactivated %r", name)

    async def handle_message(
        self, name: str, controller: WebSocketChatController, payload: dict,
    ) -> dict | None:
        """Route a message to the named extension. Auto-activates if needed."""
        ext = self._registry.get(name)
        if not ext:
            logger.warning("[APP_EXT] message for unknown extension %r", name)
            return None
        # Auto-activate on first message (lazy)
        if name not in self._active:
            await self.activate(name, controller)
        try:
            return await ext.on_message(controller, payload)
        except Exception as e:
            logger.error("[APP_EXT] on_message(%r) failed: %s", name, e)
            return None

    async def deactivate_all(self, controller: WebSocketChatController) -> None:
        """Deactivate all active extensions (called on session cleanup)."""
        for name in list(self._active):
            await self.deactivate(name, controller)
