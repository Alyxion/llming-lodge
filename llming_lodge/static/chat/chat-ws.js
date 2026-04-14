/**
 * ChatWebSocket — Chat-specific WebSocket wrapper around LlmingWebSocket.
 *
 * Delegates all reconnect/heartbeat logic to LlmingWebSocket (from llming-com).
 * Adds chat-specific behaviour: redirect to /chat when the session is lost.
 */

class ChatWebSocket {
  constructor(url, handlers) {
    this._inner = new LlmingWebSocket(url, {
      onMessage: handlers.onMessage || null,
      onOpen: handlers.onOpen || null,
      onClose: handlers.onClose || null,
      onError: handlers.onError || null,
      onSessionLost: () => {
        console.log('[Chat] Session lost — redirecting to /chat');
        location.replace('/chat');
      },
    });
  }

  connect() { this._inner.connect(); }
  send(msg) { this._inner.send(msg); }
  close()   { this._inner.close(); }

  get connected() { return this._inner.connected; }
}
