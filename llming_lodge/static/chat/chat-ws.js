/**
 * ChatWebSocket — WebSocket connection management + heartbeat
 *
 * Standalone class, no dependencies. Extracted from chat-app.js.
 */

class ChatWebSocket {
  constructor(url, handlers) {
    this.url = url;
    this.handlers = handlers; // { onMessage, onOpen, onClose, onError }
    this.ws = null;
    this._heartbeatInterval = null;
    this._reconnectAttempts = 0;
    this._maxReconnect = 5;
  }

  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onopen = () => {
      this._reconnectAttempts = 0;
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
      if (e.code !== 4004 && this._reconnectAttempts < this._maxReconnect) {
        this._reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, this._reconnectAttempts), 10000);
        setTimeout(() => this.connect(), delay);
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
    this._maxReconnect = 0; // disable reconnect
    this._stopHeartbeat();
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
}
