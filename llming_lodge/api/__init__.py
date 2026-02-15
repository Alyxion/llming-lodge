"""WebSocket API for static chat frontends.

Provides SessionRegistry, WebSocketChatController, and message handler.
The WS endpoint itself lives in llming_lodge.server.get_ws_router().
"""

from .chat_session_api import SessionRegistry, WebSocketChatController, _handle_client_message

__all__ = ["SessionRegistry", "WebSocketChatController", "_handle_client_message"]
