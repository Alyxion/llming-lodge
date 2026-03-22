/**
 * ChatWebSocket — WebSocket connection management + heartbeat + auto-recovery
 *
 * Standalone class, no dependencies. Extracted from chat-app.js.
 *
 * Recovery behaviour:
 * - On unexpected close: exponential backoff retries for up to ~30s
 * - On session-not-found (code 4004): redirect to /chat for re-auth
 * - Shows a status banner during reconnection attempts
 */

class ChatWebSocket {
  constructor(url, handlers) {
    this.url = url;
    this.handlers = handlers; // { onMessage, onOpen, onClose, onError }
    this.ws = null;
    this._heartbeatInterval = null;
    this._reconnectAttempts = 0;
    this._maxReconnectAttempts = 15;
    this._reconnectTimer = null;
    this._intentionalClose = false;
  }

  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => {
      this._reconnectAttempts = 0;
      this._hideReconnectBanner();
      this._startHeartbeat();
      if (this.handlers.onOpen) this.handlers.onOpen();
    };
    this.ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (this.handlers.onMessage) this.handlers.onMessage(msg);
      } catch (err) {
        console.error('[WS] Parse error:', err);
      }
    };
    this.ws.onclose = (e) => {
      this._stopHeartbeat();
      if (this.handlers.onClose) this.handlers.onClose(e);

      if (this._intentionalClose) return;

      // Session gone (server restarted) — redirect for seamless re-auth
      if (e.code === 4004) {
        console.log('[WS] Session not found — redirecting to /chat for re-auth');
        location.replace('/chat');
        return;
      }

      // Unexpected disconnect — try to reconnect
      if (this._reconnectAttempts < this._maxReconnectAttempts) {
        this._reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(1.5, this._reconnectAttempts), 5000);
        console.log(`[WS] Reconnecting in ${Math.round(delay)}ms (attempt ${this._reconnectAttempts}/${this._maxReconnectAttempts})`);
        this._showReconnectBanner();
        this._reconnectTimer = setTimeout(() => this.connect(), delay);
      } else {
        // Exhausted retries — redirect for full re-auth
        console.log('[WS] Max reconnect attempts reached — redirecting to /chat');
        location.replace('/chat');
      }
    };
    this.ws.onerror = (e) => {
      if (this.handlers.onError) this.handlers.onError(e);
    };
  }

  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  close() {
    this._intentionalClose = true;
    clearTimeout(this._reconnectTimer);
    this._stopHeartbeat();
    this._hideReconnectBanner();
    if (this.ws) this.ws.close();
  }

  _startHeartbeat() {
    this._heartbeatInterval = setInterval(() => {
      this.send({ type: 'heartbeat' });
    }, 30000);
  }

  _stopHeartbeat() {
    if (this._heartbeatInterval) {
      clearInterval(this._heartbeatInterval);
      this._heartbeatInterval = null;
    }
  }

  _showReconnectBanner() {
    let banner = document.getElementById('cv2-reconnect-banner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'cv2-reconnect-banner';
      banner.style.cssText =
        'position:fixed;top:0;left:0;right:0;z-index:99999;' +
        'background:rgba(30,30,50,0.95);color:#7dd3fc;' +
        'text-align:center;padding:8px 16px;font-size:13px;' +
        'backdrop-filter:blur(4px);border-bottom:1px solid rgba(125,211,252,0.2);';
      banner.textContent = 'Reconnecting…';
      document.body.appendChild(banner);
    }
    banner.style.display = '';
  }

  _hideReconnectBanner() {
    const banner = document.getElementById('cv2-reconnect-banner');
    if (banner) banner.style.display = 'none';
  }
}
