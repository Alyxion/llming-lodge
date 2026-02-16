/**
 * Chat — Static Chat Application
 *
 * Standalone JS app that communicates with the server via WebSocket.
 * No NiceGUI / Vue dependency — pure DOM manipulation.
 *
 * External deps (loaded via CDN in <head>):
 *   - marked.js  (window.marked)
 *   - DOMPurify  (window.DOMPurify)
 *   - KaTeX      (window.katex)
 */

/* ══════════════════════════════════════════════════════════
   IDBStore — IndexedDB wrapper (same schema as NiceGUI chat)
   ══════════════════════════════════════════════════════════ */

class IDBStore {
  constructor(dbName = 'llming-lodge-chat', storeName = 'conversations') {
    this.dbName = dbName;
    this.storeName = storeName;
    this.db = null;
  }

  async open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'id' });
          store.createIndex('updated_at', 'updated_at', { unique: false });
        }
      };
      req.onsuccess = (e) => { this.db = e.target.result; resolve(this.db); };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  async get(id) {
    const tx = this.db.transaction(this.storeName, 'readonly');
    const store = tx.objectStore(this.storeName);
    return new Promise((res, rej) => {
      const r = store.get(id);
      r.onsuccess = () => res(r.result || null);
      r.onerror = () => rej(r.error);
    });
  }

  async put(data) {
    const tx = this.db.transaction(this.storeName, 'readwrite');
    const store = tx.objectStore(this.storeName);
    return new Promise((res, rej) => {
      const r = store.put(data);
      r.onsuccess = () => res();
      r.onerror = () => rej(r.error);
    });
  }

  async getAll() {
    const tx = this.db.transaction(this.storeName, 'readonly');
    const store = tx.objectStore(this.storeName);
    const index = store.index('updated_at');
    return new Promise((res, rej) => {
      const results = [];
      const req = index.openCursor(null, 'prev');
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) { results.push(cursor.value); cursor.continue(); }
        else res(results);
      };
      req.onerror = () => rej(req.error);
    });
  }

  async delete(id) {
    const tx = this.db.transaction(this.storeName, 'readwrite');
    const store = tx.objectStore(this.storeName);
    return new Promise((res, rej) => {
      const r = store.delete(id);
      r.onsuccess = () => res();
      r.onerror = () => rej(r.error);
    });
  }
}

/* Global ref set in ChatApp.init() for interop with /chat */


/* ══════════════════════════════════════════════════════════
   ChatWebSocket — connection management + heartbeat
   ══════════════════════════════════════════════════════════ */

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


/* ══════════════════════════════════════════════════════════
   MarkdownRenderer — marked + DOMPurify + KaTeX
   ══════════════════════════════════════════════════════════ */

class MarkdownRenderer {
  constructor() {
    if (window.marked) {
      window.marked.setOptions({
        breaks: true,
        gfm: true,
      });
    }
  }

  render(text) {
    if (!text) return '';

    // Pre-process LaTeX before markdown parsing
    const processed = this._protectLatex(text);
    let html = window.marked ? window.marked.parse(processed) : this._basicMarkdown(processed);

    // Restore and render LaTeX
    html = this._renderLatex(html);

    // Sanitize
    if (window.DOMPurify) {
      html = window.DOMPurify.sanitize(html, {
        ADD_TAGS: ['span'],
        ADD_ATTR: ['class', 'style'],
      });
    }

    return html;
  }

  _protectLatex(text) {
    // Replace $$ ... $$ and $ ... $ with placeholders
    let idx = 0;
    this._latexBlocks = [];
    // Display math $$...$$
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, expr) => {
      this._latexBlocks.push({ expr, display: true });
      return `%%LATEX_${idx++}%%`;
    });
    // Inline math $...$
    text = text.replace(/\$([^\n$]+?)\$/g, (_, expr) => {
      this._latexBlocks.push({ expr, display: false });
      return `%%LATEX_${idx++}%%`;
    });
    return text;
  }

  _renderLatex(html) {
    if (!this._latexBlocks || !window.katex) return html;
    for (let i = 0; i < this._latexBlocks.length; i++) {
      const { expr, display } = this._latexBlocks[i];
      try {
        const rendered = window.katex.renderToString(expr.trim(), {
          displayMode: display,
          throwOnError: false,
        });
        html = html.replace(`%%LATEX_${i}%%`, rendered);
      } catch {
        html = html.replace(`%%LATEX_${i}%%`, display ? `$$${expr}$$` : `$${expr}$`);
      }
    }
    return html;
  }

  _basicMarkdown(text) {
    // Fallback if marked.js isn't loaded
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>');
  }
}


/* ══════════════════════════════════════════════════════════
   ChatApp — Main Application
   ══════════════════════════════════════════════════════════ */

class ChatApp {
  constructor(config) {
    this.config = config;
    this.idb = new IDBStore();
    this.md = new MarkdownRenderer();
    this.ws = null;

    // State
    this.sessionId = config.sessionId;
    this.userName = config.userName || '';
    this.models = [];
    this.currentModel = '';
    this.tools = [];
    this.budget = 0;
    this.systemPrompt = '';
    this.temperature = 0.7;
    this.maxInputTokens = 0;
    this.maxOutputTokens = 0;
    this.contextInfo = null;

    // Streaming state
    this.streaming = false;
    this.fullText = '';
    this.toolCalls = [];
    this._receivedImages = []; // images received via WS during streaming

    // Pending attachments (before sending)
    this._pendingImages = [];  // base64 data URIs
    this._pendingFiles = [];   // {name, size, fileId, mimeType}

    // Quick actions (populated from server via session_init)
    this.quickActions = [];

    // UI state
    this.chatVisible = false;
    this.sidebarVisible = false;
    this.modelDropdownOpen = false;
    this.plusMenuOpen = false;
    this.suggestionsMenuOpen = false;
    this.contextPopoverOpen = false;

    // Conversation sidebar
    this.conversations = [];
    this.activeConvId = '';
    this.deleteConfirmId = null;
    this.deleteTimeout = null;

    // DOM refs (set in render())
    this.el = {};
  }

  async init() {
    await this.idb.open();
    // Expose globally for /chat interop
    window.IDBStore = {
      ready: true,
      get: (id) => this.idb.get(id),
      put: (data) => this.idb.put(data),
      getAll: () => this.idb.getAll(),
      delete: (id) => this.idb.delete(id),
    };

    this.render();
    this._applyTheme();
    this.bindEvents();
    this.connectWebSocket();
    await this.refreshConversations();
  }

  _applyTheme() {
    const t = this.config.theme;
    if (!t?.accent) return;
    const root = document.getElementById('chat-app');
    if (!root) return;
    root.style.setProperty('--chat-accent', t.accent);
    if (t.accentRgb) root.style.setProperty('--chat-accent-rgb', t.accentRgb);
    if (t.accentHover) root.style.setProperty('--chat-accent-hover', t.accentHover);
    if (t.accentLight) root.style.setProperty('--chat-accent-light', t.accentLight);
  }

  // ── WebSocket ─────────────────────────────────────────

  connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}${this.config.wsPath}`;

    this.ws = new ChatWebSocket(url, {
      onMessage: (msg) => this.handleMessage(msg),
      onOpen: () => {
        console.log('[Chat] WebSocket connected');
        this._setStatus('connected');
      },
      onClose: (e) => {
        console.log('[Chat] WebSocket closed:', e.code);
        this._setStatus('disconnected');
      },
      onError: () => this._setStatus('error'),
    });
    this.ws.connect();
  }

  _setStatus(status) {
    // Could show a connection indicator
  }

  // ── Message Dispatch ──────────────────────────────────

  handleMessage(msg) {
    switch (msg.type) {
      case 'session_init':      return this.handleSessionInit(msg);
      case 'response_started':  return this.handleResponseStarted(msg);
      case 'text_chunk':        return this.handleTextChunk(msg);
      case 'tool_event':        return this.handleToolEvent(msg);
      case 'image_received':    return this.handleImageReceived(msg);
      case 'response_completed':return this.handleResponseCompleted(msg);
      case 'response_cancelled':return this.handleResponseCancelled();
      case 'error':             return this.handleError(msg);
      case 'model_switched':    return this.handleModelSwitched(msg);
      case 'tools_updated':     return this.handleToolsUpdated(msg);
      case 'context_info':      return this.handleContextInfo(msg);
      case 'title_generated':   return this.handleTitleGenerated(msg);
      case 'condense_start':    return this.handleCondenseStart();
      case 'condense_progress': return this.handleCondenseProgress(msg);
      case 'condense_end':      return this.handleCondenseEnd();
      case 'save_conversation': return this.handleSaveConversation(msg);
      case 'budget_update':     return this.handleBudgetUpdate(msg);
      case 'chat_cleared':      return this.handleChatCleared(msg);
      case 'ui_action':         return this.handleUIAction(msg);
      case 'files_updated':     return this.handleFilesUpdated(msg);
      case 'user_message':      return this.handleUserMessage(msg);
      case 'heartbeat_ack':     return; // ignore
      default:
        console.warn('[Chat] Unknown message type:', msg.type);
    }
  }

  // ── Handlers ──────────────────────────────────────────

  handleSessionInit(msg) {
    this.sessionId = msg.session_id;
    this.fullName = msg.user_name || this.userName || '';
    // Keep config userName (given/first name) for greeting; fall back to first word of full name
    if (!this.userName) this.userName = this.fullName;
    this.userAvatar = msg.user_avatar || this.config.userAvatar || '';
    this.models = msg.models || [];
    this.currentModel = msg.current_model;
    this.tools = msg.tools || [];
    this.budget = msg.budget;
    this.systemPrompt = msg.system_prompt || '';
    this.temperature = msg.temperature || 0.7;
    this.maxInputTokens = msg.max_input_tokens || 0;
    this.maxOutputTokens = msg.max_output_tokens || 0;
    this.quickActions = msg.quick_actions || [];

    // Greeting uses config userName (given name); sidebar shows full name
    const greetName = this.userName || 'there';
    if (this.el.greeting) this.el.greeting.textContent = `Hello ${greetName}, how can I help you?`;
    if (this.el.sidebarUserName) this.el.sidebarUserName.textContent = this.fullName || 'User';

    this.updateModelButton();
    this._updateAvatarTooltip();
    this.updateSettings();
    this._renderQuickActions();
  }

  handleResponseStarted(msg) {
    this.streaming = true;
    this.fullText = '';
    this.toolCalls = [];
    this._receivedImages = [];
    this._currentModelIcon = msg.model_icon;
    this._currentModelLabel = msg.model_label;

    // Show chat area if hidden
    if (!this.chatVisible) this.showChat();

    // Create assistant message container
    const msgEl = document.createElement('div');
    msgEl.className = 'cv2-msg-assistant';
    msgEl.innerHTML = `
      <div class="cv2-msg-header">
        <img src="${this.config.staticBase}/${msg.model_icon}" alt="">
        <span>${this._escHtml(msg.model_label)}</span>
      </div>
      <div class="cv2-tool-area"></div>
      <div class="cv2-msg-body"></div>
    `;
    this.el.messages.appendChild(msgEl);
    this._currentMsgEl = msgEl;
    this._currentToolArea = msgEl.querySelector('.cv2-tool-area');
    this._currentBody = msgEl.querySelector('.cv2-msg-body');

    // Add spinner
    this._spinner = document.createElement('div');
    this._spinner.className = 'cv2-spinner-dots';
    this._spinner.innerHTML = '<span></span><span></span><span></span>';
    this.el.messages.appendChild(this._spinner);

    this._scrollToBottom();
    this._updateSendButton();
  }

  handleTextChunk(msg) {
    this.fullText += msg.content;
    if (this._currentBody) {
      this._currentBody.innerHTML = this.md.render(this.fullText);
    }
    this._scrollToBottom();
  }

  handleToolEvent(msg) {
    if (!this._currentToolArea) return;

    // Update or add tool call
    const existing = this.toolCalls.find(tc => tc.call_id === msg.call_id);
    if (existing) {
      Object.assign(existing, msg);
    } else {
      this.toolCalls.push({ ...msg });
    }

    this._renderToolArea();
    this._scrollToBottom();
  }

  handleImageReceived(msg) {
    const src = this._toDataUri(msg.data);
    this._receivedImages.push(src);

    if (!this._currentBody) return;
    const img = document.createElement('img');
    img.className = 'cv2-generated-image';
    img.src = src;
    this._currentBody.appendChild(img);
    this._wrapImage(img);
    this._scrollToBottom();
  }

  handleResponseCompleted(msg) {
    this.streaming = false;
    this.fullText = msg.full_text;

    // Remove spinner
    if (this._spinner) {
      this._spinner.remove();
      this._spinner = null;
    }

    // Final render
    if (this._currentBody) {
      // Extract inline images from text
      const { images, text } = this._extractInlineImages(this.fullText);
      this._currentBody.innerHTML = this.md.render(text);

      // Add copy button
      this._addCopyButton(this._currentBody, text);

      // Add extracted images
      for (const imgSrc of images) {
        const img = document.createElement('img');
        img.className = 'cv2-generated-image';
        img.src = imgSrc;
        this._currentBody.appendChild(img);
      }

      // Re-add images received during streaming (handleImageReceived)
      // Also check response_completed payload as fallback
      const allImages = [...this._receivedImages];
      if (msg.generated_image) {
        const src = this._toDataUri(msg.generated_image);
        if (!allImages.some(s => s === src || s === msg.generated_image)) {
          allImages.push(src);
        }
      }
      for (const src of allImages) {
        const img = document.createElement('img');
        img.className = 'cv2-generated-image';
        img.src = src;
        this._currentBody.appendChild(img);
      }

      // Render contact cards for resolve_contact tool results
      for (const tc of this.toolCalls) {
        if (tc.name === 'o365_resolve_contact' && tc.result) {
          this._renderContactCards(this._currentToolArea, tc.result);
        }
      }

      // Wrap all images with hover actions + lightbox click
      this._wrapAllImages(this._currentBody);
    }

    this._currentMsgEl = null;
    this._currentToolArea = null;
    this._currentBody = null;
    this._receivedImages = [];
    this._updateSendButton();
    this._scrollToBottom();
  }

  handleResponseCancelled() {
    this.streaming = false;
    if (this._spinner) { this._spinner.remove(); this._spinner = null; }
    if (this._currentBody) {
      this.fullText += '\n\n*Stopped by user*';
      this._currentBody.innerHTML = this.md.render(this.fullText);
    }
    this._currentMsgEl = null;
    this._currentToolArea = null;
    this._currentBody = null;
    this._updateSendButton();
  }

  handleError(msg) {
    this.streaming = false;
    if (this._spinner) { this._spinner.remove(); this._spinner = null; }

    const errorHtml = `<span style="color:#ef4444">${this._escHtml(msg.message)}</span>`;
    if (this._currentBody) {
      this._currentBody.innerHTML = errorHtml;
    } else {
      // Show error as standalone message
      const el = document.createElement('div');
      el.className = 'cv2-msg-assistant';
      el.innerHTML = `<div class="cv2-msg-body">${errorHtml}</div>`;
      this.el.messages.appendChild(el);
    }
    this._currentMsgEl = null;
    this._currentToolArea = null;
    this._currentBody = null;
    this._updateSendButton();
  }

  handleModelSwitched(msg) {
    this.currentModel = msg.new_model;
    if (msg.available_tools) this.tools = msg.available_tools;
    this.updateModelButton();

    // Add model switch message to chat
    const el = document.createElement('div');
    el.className = 'cv2-model-switch';
    el.innerHTML = `
      <img src="${this.config.staticBase}/${msg.old_icon}" alt="">
      <span class="cv2-arrow material-icons">arrow_forward</span>
      <img src="${this.config.staticBase}/${msg.new_icon}" alt="">
      <span>Switched from ${this._escHtml(msg.old_label)} to ${this._escHtml(msg.new_label)}</span>
    `;
    if (this.el.messages) {
      this.el.messages.appendChild(el);
      this._scrollToBottom();
    }
  }

  handleToolsUpdated(msg) {
    this.tools = msg.tools || [];
  }

  handleContextInfo(msg) {
    this.contextInfo = msg;
    this._updateContextCircle();
  }

  handleTitleGenerated(msg) {
    // Update conversation title in sidebar
    this._updateConversationTitle(this.sessionId, msg.title);
  }

  handleCondenseStart() {
    if (!this.el.messages) return;
    const el = document.createElement('div');
    el.className = 'cv2-condense';
    el.id = 'cv2-condense-indicator';
    el.innerHTML = `
      <span class="cv2-condense-label">Condensing conversation...</span>
      <div class="cv2-condense-bar"><div class="cv2-condense-fill" style="width:0%"></div></div>
    `;
    this.el.messages.appendChild(el);
    this._scrollToBottom();
  }

  handleCondenseProgress(msg) {
    const fill = document.querySelector('#cv2-condense-indicator .cv2-condense-fill');
    if (fill) fill.style.width = `${Math.min(100, (msg.pct || 0) * 100)}%`;
  }

  handleCondenseEnd() {
    const el = document.getElementById('cv2-condense-indicator');
    if (el) {
      el.innerHTML = `
        <div class="cv2-condense-done">
          <span class="material-icons" style="font-size:14px">compress</span>
          <span>Conversation condensed</span>
        </div>
      `;
    }
  }

  handleSaveConversation(msg) {
    if (msg.data) {
      this.idb.put(msg.data).then(() => this.refreshConversations());
    }
  }

  handleBudgetUpdate(msg) {
    this.budget = msg.budget;
    this._updateAvatarTooltip();
  }

  handleChatCleared(msg) {
    this.sessionId = msg.new_session_id;
    this.chatVisible = false;
    this.el.messages.innerHTML = '';
    this.el.initialView.style.display = 'flex';
    this.el.messagesWrap.classList.add('cv2-messages-wrap-hidden');
    const wrapper = this.el.initialView.closest('.cv2-chat-wrapper');
    if (wrapper) wrapper.classList.add('cv2-initial-mode');
    this.activeConvId = '';
    this._pendingImages = [];
    this._pendingFiles = [];
    this._renderAttachments();
    this.refreshConversations();
    this._startPlaceholderCycle();
  }

  handleUserMessage(msg) {
    // Render user bubble for messages injected externally (e.g. debug API)
    if (!this.chatVisible) this.showChat();
    const userEl = document.createElement('div');
    userEl.className = 'cv2-msg-user';
    let imagesHtml = '';
    if (msg.images && msg.images.length > 0) {
      imagesHtml = '<div class="cv2-msg-user-images">' +
        msg.images.map(src => {
          const uri = this._toDataUri(src);
          return `<img src="${uri}" alt="Uploaded">`;
        }).join('') + '</div>';
    }
    userEl.innerHTML = `<div class="cv2-msg-user-bubble">${imagesHtml}${msg.text ? this.md.render(msg.text) : ''}</div>`;
    this.el.messages.appendChild(userEl);
    this._wrapAllImages(userEl);
    this._scrollToBottom();
  }

  handleFilesUpdated(msg) {
    // Server pushed updated file list (e.g. from debug API attach)
    if (msg.files) {
      this._pendingFiles = msg.files;
      this._renderAttachments();
    }
  }

  handleUIAction(msg) {
    const action = msg.action;
    switch (action) {
      case 'toggle_sidebar':
        this.sidebarVisible = !this.sidebarVisible;
        this.el.sidebar.classList.toggle('cv2-hidden', !this.sidebarVisible);
        this.el.miniToggle.classList.toggle('cv2-active', this.sidebarVisible);
        this.el.miniSidebar.classList.toggle('cv2-sidebar-open', this.sidebarVisible);
        break;
      case 'open_model_menu':
        this.modelDropdownOpen = true;
        this.el.modelDropdown.style.display = 'block';
        break;
      case 'close_dropdowns':
        this.modelDropdownOpen = false;
        this.el.modelDropdown.style.display = 'none';
        this.plusMenuOpen = false;
        this.el.plusMenu.classList.remove('cv2-visible');
        this.contextPopoverOpen = false;
        this.el.contextPopover.classList.remove('cv2-visible');
        break;
      case 'show_context_info':
        this.contextPopoverOpen = true;
        this.el.contextPopover.classList.add('cv2-visible');
        this._renderContextPopover();
        this.ws.send({ type: 'get_context_info' });
        break;
      case 'trigger_quick_action':
        const qa = this.quickActions.find(q => q.id === msg.quick_action_id);
        if (qa) this._triggerQuickAction(qa);
        break;
      case 'load_conversation':
        if (msg.conversation_id) this._selectConversation(msg.conversation_id);
        break;
      case 'list_conversations':
        // Server is requesting the conversation list from IDB
        this.idb.getAll().then(all => {
          const list = all.map(c => ({
            id: c.id,
            title: c.title || (c.messages?.find(m => m.role === 'user')?.content || '').substring(0, 60) || 'Untitled',
            created_at: c.created_at,
            message_count: c.messages?.length || 0,
          }));
          this.ws.send({ type: 'conversation_list', conversations: list });
        });
        break;
      case 'open_lightbox':
        // Open lightbox for an image by index in the current chat
        {
          const imgs = this.el.messages.querySelectorAll('.cv2-img-wrap img');
          const idx = msg.image_index ?? 0;
          if (imgs[idx]) this._openLightbox(imgs[idx].src);
        }
        break;
      default:
        console.warn('[Chat] Unknown UI action:', action);
    }
  }

  // ── Send Message ──────────────────────────────────────

  sendMessage() {
    const text = this.el.textarea.value.trim();
    if (!text && this._pendingImages.length === 0) return;

    if (this.streaming) {
      // Stop streaming
      this.ws.send({ type: 'stop_streaming' });
      return;
    }

    // Show chat area
    if (!this.chatVisible) this.showChat();

    // Render user message
    const userEl = document.createElement('div');
    userEl.className = 'cv2-msg-user';
    let imagesHtml = '';
    if (this._pendingImages.length > 0) {
      imagesHtml = '<div class="cv2-msg-user-images">' +
        this._pendingImages.map(src => `<img src="${src}" alt="Uploaded">`).join('') +
        '</div>';
    }
    const bubbleHtml = `<div class="cv2-msg-user-bubble">${imagesHtml}${text ? this.md.render(text) : ''}</div>`;
    userEl.innerHTML = bubbleHtml;
    this.el.messages.appendChild(userEl);
    this._wrapAllImages(userEl);

    // Collect images for WS message (base64 only, no data: prefix)
    const images = this._pendingImages.length > 0
      ? this._pendingImages.map(d => d.split(',')[1])
      : undefined;

    // Clear input and pasted images (files persist — they're in system prompt context)
    this.el.textarea.value = '';
    this._autoResizeTextarea();
    this._pendingImages = [];
    this._renderAttachments();

    // Send via WS
    this.ws.send({ type: 'send_message', text: text || '', images });

    this._scrollToBottom();
  }

  _triggerQuickAction(qa) {
    // Switch model if specified
    if (qa.model) {
      const modelInfo = this.models.find(m => m.model === qa.model);
      if (modelInfo && qa.model !== this.currentModel) {
        this.ws.send({ type: 'switch_model', model: qa.model });
      }
    }

    // Set system prompt
    if (qa.prompt) {
      this.ws.send({ type: 'update_settings', system_prompt: qa.prompt });
    }

    // Special: Document analysis — open file dialog, auto-summarize silently
    if (qa.id === '@sys.docs') {
      this._onFilesReady = () => {
        if (!this.chatVisible) this.showChat();
        const hasImages = this._pendingImages.length > 0;
        const docName = this._pendingFiles[this._pendingFiles.length - 1]?.name;

        if (hasImages) {
          // Show image in chat as user bubble (no text shown)
          const userEl = document.createElement('div');
          userEl.className = 'cv2-msg-user';
          const imgsHtml = '<div class="cv2-msg-user-images">' +
            this._pendingImages.map(src => `<img src="${src}" alt="Uploaded">`).join('') +
            '</div>';
          userEl.innerHTML = `<div class="cv2-msg-user-bubble">${imgsHtml}</div>`;
          this.el.messages.appendChild(userEl);
          this._wrapAllImages(userEl);

          const images = this._pendingImages.map(d => d.split(',')[1]);
          this._pendingImages = [];
          this._renderAttachments();
          this.ws.send({ type: 'send_message', text: 'Briefly describe this image, then ask what I\'d like to do with it.', images });
          this._scrollToBottom();
        } else if (docName) {
          this.ws.send({ type: 'send_message', text: `Summarize "${docName}" briefly, then ask what I'd like to focus on.` });
        }
      };
      this.el.fileInput.click();
      return;
    }

    // For all other actions: send silently and let the AI engage the user
    if (!this.chatVisible) this.showChat();
    const inputText = this.el.textarea.value.trim();
    if (inputText && qa.textPrefix) {
      // Text in the input box + action has a text prefix — act on it immediately
      this.el.textarea.value = '';
      this.el.textarea.style.height = 'auto';
      const fullText = `${qa.textPrefix}\n\n${inputText}`;
      // Show as user bubble in chat
      const userEl = document.createElement('div');
      userEl.className = 'cv2-msg-user';
      userEl.innerHTML = `<div class="cv2-msg-user-bubble">${this.md.render(fullText)}</div>`;
      this.el.messages.appendChild(userEl);
      this.ws.send({ type: 'send_message', text: fullText });
    } else {
      this.ws.send({ type: 'send_message', text: qa.engagement || qa.label });
    }
    this._scrollToBottom();
  }

  showChat() {
    this.chatVisible = true;
    this._stopPlaceholderCycle();
    this.el.initialView.style.display = 'none';
    this.el.messagesWrap.classList.remove('cv2-messages-wrap-hidden');
    const wrapper = this.el.initialView.closest('.cv2-chat-wrapper');
    if (wrapper) wrapper.classList.remove('cv2-initial-mode');
    const inputArea = wrapper?.querySelector('.cv2-input-area');
    if (inputArea) {
      inputArea.classList.add('cv2-input-animating');
      inputArea.addEventListener('animationend', () => inputArea.classList.remove('cv2-input-animating'), { once: true });
    }
    this.el.textarea.focus();
  }

  // ── Render ────────────────────────────────────────────

  render() {
    const root = document.getElementById('chat-app');
    if (!root) return;

    const avatarUrl = this.config.userAvatar || '';
    const avatarHtml = avatarUrl
      ? `<img src="${this._escAttr(avatarUrl)}" alt="User">`
      : '<span class="material-icons">person</span>';

    root.innerHTML = `
      <!-- Mini sidebar (always visible) -->
      <div class="cv2-mini-sidebar" id="cv2-mini-sidebar">
        <button class="cv2-mini-sidebar-btn" id="cv2-mini-toggle" title="Toggle sidebar">
          <span class="material-icons">menu</span>
        </button>
        <button class="cv2-mini-sidebar-btn" id="cv2-mini-new-chat" title="New chat">
          <span class="material-icons">edit_square</span>
        </button>
        <div class="cv2-mini-sidebar-spacer"></div>
        <button class="cv2-mini-sidebar-btn" id="cv2-theme-toggle" title="Toggle dark mode">
          <span class="material-icons" id="cv2-theme-icon">dark_mode</span>
        </button>
        <div class="cv2-mini-sidebar-avatar">${avatarHtml}</div>
      </div>

      <!-- Full sidebar -->
      <div class="cv2-sidebar cv2-hidden" id="cv2-sidebar">
        <div class="cv2-sidebar-header">
          ${this.config.appLogo ? `<img class="cv2-sidebar-logo" src="${this._escAttr(this.config.appLogo)}" alt="">` : ''}
          <span class="cv2-sidebar-title">${this._escHtml(this.config.appTitle || 'Conversations')}</span>
          <button class="cv2-sidebar-close-btn" id="cv2-sidebar-close">
            <span class="material-icons" style="font-size:18px">close</span>
          </button>
        </div>
        <button class="cv2-new-chat-btn" id="cv2-new-chat">
          <span class="material-icons" style="font-size:16px">add</span> New Chat
        </button>
        <div class="cv2-conversations" id="cv2-conversations"></div>
        <div class="cv2-sidebar-actions" id="cv2-sidebar-actions" style="display:none">
          <button class="cv2-sidebar-action-btn" id="cv2-export-all">
            <span class="material-icons" style="font-size:14px">download</span> Export
          </button>
          <span style="color:rgba(255,255,255,0.15);font-size:11px">·</span>
          <button class="cv2-sidebar-action-btn cv2-danger" id="cv2-clear-all">Clear all</button>
        </div>
        <div class="cv2-sidebar-budget" id="cv2-budget"></div>
        <div class="cv2-sidebar-footer">
          <div class="cv2-sidebar-avatar">${avatarHtml}</div>
          <span class="cv2-sidebar-user-name" id="cv2-sidebar-user-name"></span>
        </div>
      </div>

      <!-- Main -->
      <div class="cv2-main">
        <!-- Top bar -->
        <div class="cv2-topbar">
          <button class="cv2-topbar-btn" id="cv2-sidebar-toggle" style="display:none">
            <span class="material-icons">menu</span>
          </button>
          ${this.config.appLogo ? `<img class="cv2-topbar-logo" src="${this._escAttr(this.config.appLogo)}" alt="">` : ''}
          <div style="flex:1"></div>
        </div>

        <!-- Chat wrapper -->
        <div class="cv2-chat-wrapper cv2-initial-mode">
          <!-- Initial view -->
          <div class="cv2-initial-view" id="cv2-initial-view">
            <div class="cv2-greeting-row">
              ${this.config.appMascot ? `<img class="cv2-mascot" src="${this._escAttr(this.config.appMascot)}" alt="">` : ''}
              <h2 id="cv2-greeting"></h2>
            </div>
          </div>

          <!-- Messages -->
          <div id="cv2-messages-wrap" class="cv2-messages-wrap-hidden">
            <div class="cv2-messages" id="cv2-messages"></div>
          </div>

          <!-- Input area -->
          <div class="cv2-input-area">
            <div class="cv2-input-wrapper" id="cv2-input-wrapper">
              <div class="cv2-attachments" id="cv2-attachments" style="display:none"></div>
              <div class="cv2-input-top">
                <div class="cv2-ph-overlay" id="cv2-ph-overlay">
                  <span class="cv2-ph-text-wrap">
                    <span class="cv2-ph-text cv2-ph-current" id="cv2-ph-current"></span>
                    <span class="cv2-ph-text cv2-ph-next" id="cv2-ph-next"></span>
                  </span>
                  <button class="cv2-ph-action-btn" id="cv2-ph-action-btn"><span class="material-icons">auto_awesome</span></button>
                </div>
                <textarea class="cv2-textarea" id="cv2-textarea" rows="1"></textarea>
              </div>
              <input type="file" id="cv2-file-input" multiple accept="image/*,.pdf,.docx,.xlsx,.doc,.xls" style="display:none">
              <div class="cv2-input-bottom">
                <button class="cv2-input-btn" id="cv2-plus-btn" title="Tools & actions">
                  <span class="material-icons">add</span>
                </button>
                <div class="cv2-plus-menu" id="cv2-plus-menu"></div>
                <button class="cv2-input-btn" id="cv2-suggestions-btn" title="Suggestions">
                  <span class="material-icons">auto_awesome</span>
                </button>
                <div class="cv2-suggestions-menu" id="cv2-suggestions-menu"></div>
                <div style="flex:1"></div>
                <div style="position:relative">
                  <button class="cv2-model-btn" id="cv2-model-btn"></button>
                  <div class="cv2-model-dropdown" id="cv2-model-dropdown"></div>
                </div>
                <div class="cv2-context-circle" id="cv2-context-circle" title="Context usage">
                  <svg viewBox="0 0 36 36">
                    <circle class="cv2-bg" cx="18" cy="18" r="15.9"/>
                    <circle class="cv2-fg" id="cv2-context-fg" cx="18" cy="18" r="15.9"
                            stroke-dasharray="100 100" stroke-dashoffset="100" stroke="#e07020"/>
                  </svg>
                  <span class="cv2-context-pct" id="cv2-context-pct">0%</span>
                </div>
                <div class="cv2-context-popover" id="cv2-context-popover"></div>
                <button class="cv2-input-btn" id="cv2-mic-btn" title="Voice input (coming soon)">
                  <span class="material-icons">mic</span>
                </button>
                <button class="cv2-send-btn" id="cv2-send-btn" title="Send">
                  <span class="material-icons" id="cv2-send-icon">send</span>
                </button>
              </div>
            </div>
            <div class="cv2-disclaimer">AI can make mistakes. Please verify important information.</div>
          </div>
        </div>
      </div>

    `;

    // Cache DOM refs
    this.el = {
      miniSidebar: root.querySelector('#cv2-mini-sidebar'),
      miniToggle: root.querySelector('#cv2-mini-toggle'),
      miniNewChat: root.querySelector('#cv2-mini-new-chat'),
      themeToggle: root.querySelector('#cv2-theme-toggle'),
      themeIcon: root.querySelector('#cv2-theme-icon'),
      sidebar: root.querySelector('#cv2-sidebar'),
      sidebarClose: root.querySelector('#cv2-sidebar-close'),
      sidebarToggle: root.querySelector('#cv2-sidebar-toggle'),
      sidebarUserName: root.querySelector('#cv2-sidebar-user-name'),
      newChat: root.querySelector('#cv2-new-chat'),
      conversations: root.querySelector('#cv2-conversations'),
      sidebarActions: root.querySelector('#cv2-sidebar-actions'),
      exportAll: root.querySelector('#cv2-export-all'),
      clearAll: root.querySelector('#cv2-clear-all'),
      budget: root.querySelector('#cv2-budget'),
      modelBtn: root.querySelector('#cv2-model-btn'),
      modelDropdown: root.querySelector('#cv2-model-dropdown'),
      initialView: root.querySelector('#cv2-initial-view'),
      greeting: root.querySelector('#cv2-greeting'),
      messagesWrap: root.querySelector('#cv2-messages-wrap'),
      messages: root.querySelector('#cv2-messages'),
      inputWrapper: root.querySelector('#cv2-input-wrapper'),
      textarea: root.querySelector('#cv2-textarea'),
      plusBtn: root.querySelector('#cv2-plus-btn'),
      plusMenu: root.querySelector('#cv2-plus-menu'),
      suggestionsBtn: root.querySelector('#cv2-suggestions-btn'),
      suggestionsMenu: root.querySelector('#cv2-suggestions-menu'),
      phOverlay: root.querySelector('#cv2-ph-overlay'),
      phCurrent: root.querySelector('#cv2-ph-current'),
      phNext: root.querySelector('#cv2-ph-next'),
      phActionBtn: root.querySelector('#cv2-ph-action-btn'),
      contextCircle: root.querySelector('#cv2-context-circle'),
      contextFg: root.querySelector('#cv2-context-fg'),
      contextPct: root.querySelector('#cv2-context-pct'),
      contextPopover: root.querySelector('#cv2-context-popover'),
      sendBtn: root.querySelector('#cv2-send-btn'),
      sendIcon: root.querySelector('#cv2-send-icon'),
      attachments: root.querySelector('#cv2-attachments'),
      fileInput: root.querySelector('#cv2-file-input'),
    };

    // Set greeting & user name
    const greetName = this.userName || 'there';
    this.el.greeting.textContent = `Hello ${greetName}, how can I help you?`;
    this.el.sidebarUserName.textContent = this.userName || 'User';

    // Quick action cards (rendered after session_init provides them)
    this._renderQuickActions();

    // Restore theme from localStorage
    this._applyStoredTheme();
  }

  // ── Bind Events ───────────────────────────────────────

  bindEvents() {
    // Send
    this.el.sendBtn.addEventListener('click', () => this.sendMessage());
    this.el.textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea
    this.el.textarea.addEventListener('input', () => this._autoResizeTextarea());

    // Sidebar toggle (mini sidebar + close)
    const toggleSidebar = () => {
      this.sidebarVisible = !this.sidebarVisible;
      this.el.sidebar.classList.toggle('cv2-hidden', !this.sidebarVisible);
      this.el.miniToggle.classList.toggle('cv2-active', this.sidebarVisible);
      this.el.miniSidebar.classList.toggle('cv2-sidebar-open', this.sidebarVisible);
    };
    this.el.miniToggle.addEventListener('click', toggleSidebar);
    this.el.sidebarToggle.addEventListener('click', toggleSidebar);
    this.el.sidebarClose.addEventListener('click', () => {
      this.sidebarVisible = false;
      this.el.sidebar.classList.add('cv2-hidden');
      this.el.miniToggle.classList.remove('cv2-active');
      this.el.miniSidebar.classList.remove('cv2-sidebar-open');
    });

    // Theme toggle (dark/light)
    this.el.themeToggle.addEventListener('click', () => this._toggleTheme());

    // New chat (from both mini sidebar and full sidebar)
    this.el.newChat.addEventListener('click', () => {
      this.ws.send({ type: 'new_chat' });
    });
    this.el.miniNewChat.addEventListener('click', () => {
      this.ws.send({ type: 'new_chat' });
    });

    // Model dropdown
    this.el.modelBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.modelDropdownOpen = !this.modelDropdownOpen;
      this.el.modelDropdown.style.display = this.modelDropdownOpen ? 'block' : 'none';
    });

    // Plus menu
    this.el.plusBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.plusMenuOpen = !this.plusMenuOpen;
      this._renderPlusMenu();
      this.el.plusMenu.classList.toggle('cv2-visible', this.plusMenuOpen);
    });

    // Suggestions menu
    this.el.suggestionsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.suggestionsMenuOpen = !this.suggestionsMenuOpen;
      this.el.suggestionsMenu.classList.toggle('cv2-visible', this.suggestionsMenuOpen);
    });

    // Context popover
    this.el.contextCircle.addEventListener('click', (e) => {
      e.stopPropagation();
      this.contextPopoverOpen = !this.contextPopoverOpen;
      this.el.contextPopover.classList.toggle('cv2-visible', this.contextPopoverOpen);
      if (this.contextPopoverOpen) {
        this._renderContextPopover();
        this.ws.send({ type: 'get_context_info' });
      }
    });

    // Close dropdowns on outside click
    document.addEventListener('click', () => {
      this.modelDropdownOpen = false;
      this.el.modelDropdown.style.display = 'none';
      this.plusMenuOpen = false;
      this.el.plusMenu.classList.remove('cv2-visible');
      this.suggestionsMenuOpen = false;
      this.el.suggestionsMenu.classList.remove('cv2-visible');
      this.contextPopoverOpen = false;
      this.el.contextPopover.classList.remove('cv2-visible');
    });

    // Export all
    this.el.exportAll.addEventListener('click', () => this._exportAll());

    // Clear all — show inline confirmation dialog
    this.el.clearAll.addEventListener('click', () => {
      this._showClearAllDialog();
    });

    // Drag & drop on input
    this.el.inputWrapper.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.el.inputWrapper.classList.add('cv2-drag-active');
    });
    this.el.inputWrapper.addEventListener('dragleave', () => {
      this.el.inputWrapper.classList.remove('cv2-drag-active');
    });
    this.el.inputWrapper.addEventListener('drop', (e) => {
      e.preventDefault();
      this.el.inputWrapper.classList.remove('cv2-drag-active');
      this._handleFileDrop(e.dataTransfer);
    });

    // Paste images
    this.el.textarea.addEventListener('paste', (e) => {
      this._handlePaste(e);
    });

    // File input change
    this.el.fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        this._handleFileSelection(Array.from(e.target.files));
        e.target.value = ''; // reset so same file can be selected again
      } else {
        // User cancelled the file dialog — clear any pending callback
        this._onFilesReady = null;
      }
    });
  }

  // ── UI Updates ────────────────────────────────────────

  _renderQuickActions() {
    const container = this.el.suggestionsMenu;
    if (!container) return;
    container.innerHTML = this.quickActions.map(qa => `
      <button class="cv2-menu-item cv2-suggestion-item" data-qa-id="${this._escAttr(qa.id)}">
        <span class="material-icons">${qa.icon}</span>
        ${this._escHtml(qa.label)}
      </button>
    `).join('');
    container.querySelectorAll('.cv2-suggestion-item').forEach(btn => {
      btn.addEventListener('click', () => {
        const qa = this.quickActions.find(q => q.id === btn.dataset.qaId);
        if (qa) {
          this._closeSuggestionsMenu();
          this._triggerQuickAction(qa);
        }
      });
    });

    // Start cycling placeholder hints
    this._startPlaceholderCycle();
  }

  _startPlaceholderCycle() {
    this._stopPlaceholderCycle();
    if (!this.quickActions.length) return;
    // Random start
    this._placeholderIdx = Math.floor(Math.random() * this.quickActions.length);
    const ov = this.el.phOverlay;
    const cur = this.el.phCurrent;
    const nxt = this.el.phNext;
    if (!ov || !cur || !nxt) return;

    // IDs that get an action button
    this._actionableQAs = new Set(['@sys.docs', '@sys.image']);

    // Show first hint immediately
    cur.textContent = `${this.quickActions[this._placeholderIdx].label}...`;
    cur.style.opacity = '1';
    nxt.style.opacity = '0';
    this._updatePhOverlayVisibility();
    this._updatePhActionBtn();

    // Hide overlay when user types or focuses
    this.el.textarea.addEventListener('input', () => this._updatePhOverlayVisibility());
    this.el.textarea.addEventListener('focus', () => { if (this.el.phOverlay) this.el.phOverlay.style.display = 'none'; });
    this.el.textarea.addEventListener('blur', () => this._updatePhOverlayVisibility());

    // Action button triggers only the current actionable quick action
    this.el.phActionBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const qa = this.quickActions[this._placeholderIdx];
      if (qa && this._actionableQAs.has(qa.id)) this._triggerQuickAction(qa);
    });

    // Start cycling after initial delay
    this._placeholderDelay = setTimeout(() => {
      this._placeholderTimer = setInterval(() => {
        if (this.el.textarea.value.trim() || this.chatVisible) return;
        // Prepare next text — skip if both current and next are actionable
        const prevIdx = this._placeholderIdx;
        let nextIdx = (prevIdx + 1) % this.quickActions.length;
        const prevActionable = this._actionableQAs.has(this.quickActions[prevIdx].id);
        const nextActionable = this._actionableQAs.has(this.quickActions[nextIdx].id);
        if (prevActionable && nextActionable) {
          nextIdx = (nextIdx + 1) % this.quickActions.length;
        }
        this._placeholderIdx = nextIdx;
        nxt.textContent = `${this.quickActions[this._placeholderIdx].label}...`;
        // Crossfade: current out, next in — simultaneously
        cur.style.opacity = '0';
        nxt.style.opacity = '1';
        this._updatePhActionBtn();
        // After transition completes, swap roles
        setTimeout(() => {
          cur.textContent = nxt.textContent;
          cur.style.opacity = '1';
          nxt.style.opacity = '0';
          // Reset transition so the swap is instant (no animation)
          cur.style.transition = 'none';
          nxt.style.transition = 'none';
          requestAnimationFrame(() => {
            cur.style.transition = '';
            nxt.style.transition = '';
          });
        }, 800);
      }, 8000);
    }, 2500);
  }

  _stopPlaceholderCycle() {
    if (this._placeholderTimer) {
      clearInterval(this._placeholderTimer);
      this._placeholderTimer = null;
    }
    if (this._placeholderDelay) {
      clearTimeout(this._placeholderDelay);
      this._placeholderDelay = null;
    }
    if (this.el?.phOverlay) {
      this.el.phOverlay.style.display = 'none';
    }
  }

  _updatePhOverlayVisibility() {
    if (!this.el?.phOverlay) return;
    this.el.phOverlay.style.display = this.el.textarea.value.trim() ? 'none' : '';
  }

  _updatePhActionBtn() {
    if (!this.el?.phOverlay || !this._actionableQAs) return;
    const qa = this.quickActions[this._placeholderIdx];
    const show = qa && this._actionableQAs.has(qa.id);
    this.el.phOverlay.classList.toggle('cv2-ph-has-action', show);
  }

  _closeSuggestionsMenu() {
    this.suggestionsMenuOpen = false;
    this.el.suggestionsMenu.classList.remove('cv2-visible');
  }

  updateModelButton() {
    const info = this.models.find(m => m.model === this.currentModel);
    if (!info) {
      this.el.modelBtn.innerHTML = `<span>${this._escHtml(this.currentModel)}</span><span class="material-icons" style="font-size:14px;color:#9ca3af">expand_more</span>`;
      return;
    }
    this.el.modelBtn.innerHTML = `
      <img src="${this.config.staticBase}/${info.icon}" alt="">
      <span>${this._escHtml(info.label)}</span>
      <span class="material-icons" style="font-size:14px;color:#9ca3af">expand_more</span>
    `;

    // Update dropdown with comparison bars, sorted by popularity
    const bar = (val) => `<div class="cv2-model-bar"><div class="cv2-model-bar-fill" style="width:${val * 10}%"></div></div>`;
    const sorted = [...this.models].sort((a, b) => (b.popularity || 0) - (a.popularity || 0));

    this.el.modelDropdown.innerHTML = sorted.map(m => {
      const sel = m.model === this.currentModel;
      const hl = (m.highlights || []).map(h => `<span class="cv2-model-tag">${this._escHtml(h)}</span>`).join('');
      return `
        <div class="cv2-model-option ${sel ? 'cv2-selected' : ''}" data-model="${this._escAttr(m.model)}">
          <div class="cv2-model-opt-top">
            <img src="${this.config.staticBase}/${m.icon}" alt="">
            <span class="cv2-model-opt-name">${this._escHtml(m.label)}</span>
            <span class="cv2-model-opt-use">${this._escHtml(m.best_use || 'General')}</span>
            ${sel ? '<span class="material-icons cv2-model-check">check</span>' : ''}
          </div>
          <div class="cv2-model-bars">
            <div class="cv2-model-bar-col">
              <span class="cv2-model-bar-label">Speed</span>
              ${bar(m.speed || 5)}
            </div>
            <div class="cv2-model-bar-col">
              <span class="cv2-model-bar-label">Quality</span>
              ${bar(m.quality || 5)}
            </div>
            <div class="cv2-model-bar-col">
              <span class="cv2-model-bar-label">Cost</span>
              ${bar(m.cost || 5)}
            </div>
            <div class="cv2-model-bar-col">
              <span class="cv2-model-bar-label">Context</span>
              ${bar(m.memory || 5)}
            </div>
            <span class="cv2-model-ctx-label">${this._escHtml(m.context_label || '')}</span>
          </div>
          <div class="cv2-model-highlights">${hl}</div>
        </div>
      `;
    }).join('');

    // Bind model click
    this.el.modelDropdown.querySelectorAll('.cv2-model-option').forEach(el => {
      el.addEventListener('click', () => {
        const model = el.dataset.model;
        this.ws.send({ type: 'switch_model', model });
        this.modelDropdownOpen = false;
        this.el.modelDropdown.style.display = 'none';
      });
    });
  }

  _updateAvatarTooltip() {
    const root = document.getElementById('chat-app');
    if (!root) return;
    const name = this.fullName || this.userName || '';
    let tip = name;
    if (this.config.showBudget && this.budget !== undefined && this.budget > 0) {
      tip += `\nBudget: $${Number(this.budget).toFixed(2)}`;
    }
    // Update both mini and full sidebar avatars
    const miniAvatar = root.querySelector('.cv2-mini-sidebar-avatar');
    if (miniAvatar) miniAvatar.title = tip;
    const sidebarAvatar = root.querySelector('.cv2-sidebar-avatar');
    if (sidebarAvatar) sidebarAvatar.title = tip;
    // Hide standalone budget element — budget now lives in context popover + avatar tooltip
    if (this.el.budget) this.el.budget.style.display = 'none';
  }

  updateSettings() {
    // Settings are now admin-only, no UI to update
  }

  // ── Tool Area Rendering ───────────────────────────────

  _renderToolArea() {
    if (!this._currentToolArea) return;
    const pending = this.toolCalls.filter(tc => tc.status === 'pending');
    const completed = this.toolCalls.filter(tc => tc.status === 'completed' && !tc.is_image_generation);
    const imageGen = this.toolCalls.find(tc => tc.status === 'pending' && tc.is_image_generation);

    let html = '';

    // Pending tools
    if (pending.length > 0 && completed.length === 0) {
      const tc = pending[0];
      if (tc.is_image_generation) {
        html += `
          <div style="display:flex;flex-direction:column;align-items:center;gap:16px">
            <span class="cv2-shimmer" style="font-size:16px">Generating image...</span>
            <div class="cv2-image-placeholder">
              <div class="cv2-spinner-dots"><span></span><span></span><span></span></div>
            </div>
          </div>
        `;
      } else if (tc.name === 'web_search') {
        html += `
          <div class="cv2-tool-pending">
            <div class="cv2-spinner-dots"><span></span><span></span><span></span></div>
            <span class="cv2-shimmer">Searching the web...</span>
          </div>
        `;
      } else {
        html += `
          <div class="cv2-tool-pending">
            <div class="cv2-spinner-dots"><span></span><span></span><span></span></div>
            <span class="cv2-shimmer">Running ${this._escHtml(tc.display_name)}...</span>
          </div>
        `;
      }
    }

    // Completed tools
    for (const tc of completed) {
      html += `
        <div class="cv2-tool-completed">
          <span class="material-icons cv2-icon-check">check_circle</span>
          <span>Used ${this._escHtml(tc.display_name)}</span>
        </div>
      `;
    }

    this._currentToolArea.innerHTML = html;
  }

  // ── Contact Cards ─────────────────────────────────────

  _renderContactCards(container, result) {
    // Parse: - **Name** <email> | dept | title | N meeting(s)
    const photoRe = /^- !\[[^\]]*\]\((\/api\/[^)]+)\)\s*\*\*([^*]+)\*\*\s*<([^>]+)>(.*)/gm;
    const noPhotoRe = /^- \*\*([^*]+)\*\*\s*<([^>]+)>(.*)/gm;
    const rendered = new Set();

    const parseFields = (rest) => {
      const parts = rest.split('|').map(p => p.trim()).filter(Boolean);
      let dept = '', title = '', meetings = null;
      for (const p of parts) {
        const m = p.match(/(\d+)\s*meeting/);
        if (m) meetings = m[1];
        else if (!dept) dept = p;
        else title = p;
      }
      return { dept, title, meetings };
    };

    const buildCard = (name, email, dept, title, meetings, photoUrl) => {
      const avatar = photoUrl
        ? `<img class="cv2-contact-avatar" src="${photoUrl}" alt="">`
        : `<div class="cv2-contact-initials">${name.split(/[\s,]+/).map(w => w[0] || '').join('').substring(0,2).toUpperCase()}</div>`;
      const titleHtml = title ? `<span class="cv2-contact-title">${this._escHtml(title)}</span>` : '';
      const deptHtml = dept ? `<span class="cv2-contact-dept">${this._escHtml(dept)}</span>` : '';
      const meta = [email];
      if (meetings) meta.push(`${meetings} meetings (90d)`);
      return `
        <div class="cv2-contact-card">
          ${avatar}
          <div class="cv2-contact-info">
            <span class="cv2-contact-name">${this._escHtml(name)}</span>
            ${titleHtml}${deptHtml}
            <span class="cv2-contact-meta">${this._escHtml(meta.join(' · '))}</span>
          </div>
        </div>
      `;
    };

    let html = '';
    let m;
    while ((m = photoRe.exec(result)) !== null) {
      const { dept, title, meetings } = parseFields(m[4]);
      html += buildCard(m[2].trim(), m[3].trim(), dept, title, meetings, m[1]);
      rendered.add(m[3].trim().toLowerCase());
    }
    while ((m = noPhotoRe.exec(result)) !== null) {
      if (rendered.has(m[2].trim().toLowerCase())) continue;
      const { dept, title, meetings } = parseFields(m[3]);
      html += buildCard(m[1].trim(), m[2].trim(), dept, title, meetings, null);
    }

    if (html) {
      const div = document.createElement('div');
      div.innerHTML = html;
      container.appendChild(div);
    }
  }

  // ── Plus Menu ─────────────────────────────────────────

  _renderPlusMenu() {
    // Group tools by category
    const categories = {};
    for (const tool of this.tools) {
      if (!tool.available) continue;
      const cat = tool.category || 'General';
      if (!categories[cat]) categories[cat] = [];
      categories[cat].push(tool);
    }

    let html = '';

    // File selection
    html += `
      <button class="cv2-menu-item" data-action="attach-file">
        <span class="material-icons">attach_file</span> Attach file
      </button>
    `;

    // Condense option
    html += `
      <button class="cv2-menu-item" data-action="condense">
        <span class="material-icons">compress</span> Condense conversation
      </button>
      <div class="cv2-menu-sep"></div>
    `;

    // Tools by category
    for (const [cat, tools] of Object.entries(categories)) {
      html += `<div class="cv2-tool-category">${this._escHtml(cat)}</div>`;
      for (const tool of tools) {
        html += `
          <div class="cv2-tool-item" data-tool="${this._escAttr(tool.name)}">
            <span class="material-icons" style="font-size:16px">${tool.icon || 'build'}</span>
            <span style="flex:1">${this._escHtml(tool.display_name)}</span>
            <div class="cv2-tool-toggle ${tool.enabled ? 'cv2-on' : ''}" data-toggle="${this._escAttr(tool.name)}"></div>
          </div>
        `;
      }
    }

    this.el.plusMenu.innerHTML = html;

    // Bind actions
    this.el.plusMenu.querySelector('[data-action="attach-file"]')?.addEventListener('click', () => {
      this.el.fileInput.click();
      this.plusMenuOpen = false;
      this.el.plusMenu.classList.remove('cv2-visible');
    });

    this.el.plusMenu.querySelector('[data-action="condense"]')?.addEventListener('click', () => {
      this.ws.send({ type: 'condense' });
      this.plusMenuOpen = false;
      this.el.plusMenu.classList.remove('cv2-visible');
    });

    // Tool toggles
    this.el.plusMenu.querySelectorAll('.cv2-tool-toggle').forEach(toggle => {
      toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const name = toggle.dataset.toggle;
        const isOn = toggle.classList.contains('cv2-on');
        toggle.classList.toggle('cv2-on', !isOn);
        this.ws.send({ type: 'toggle_tool', name, enabled: !isOn });
        // Update local state
        const tool = this.tools.find(t => t.name === name);
        if (tool) tool.enabled = !isOn;
      });
    });
  }

  // ── Context Circle ────────────────────────────────────

  _updateContextCircle() {
    if (!this.contextInfo) return;
    const pct = this.contextInfo.pct || 0;

    // Update SVG circle
    const circumference = 2 * Math.PI * 15.9; // ~100
    const offset = circumference - (pct / 100) * circumference;
    this.el.contextFg.style.strokeDashoffset = offset;

    // Color: green → yellow → red
    let color = '#22c55e';
    if (pct > 75) color = '#ef4444';
    else if (pct > 50) color = '#eab308';
    this.el.contextFg.setAttribute('stroke', color);

    this.el.contextPct.textContent = `${pct}%`;
  }

  _renderContextPopover() {
    const info = this.contextInfo || {};
    let html = `
      <div class="cv2-context-row"><span>History</span><strong>${(info.historyTokens || 0).toLocaleString()} tokens</strong></div>
      <div class="cv2-context-row"><span>Documents</span><strong>${(info.docTokens || 0).toLocaleString()} tokens</strong></div>
      <div class="cv2-context-row"><span>Images</span><strong>${(info.imageTokens || 0).toLocaleString()} tokens</strong></div>
      <div class="cv2-context-row"><span>Tools</span><strong>${(info.toolTokens || 0).toLocaleString()} tokens</strong></div>
      <div class="cv2-context-row" style="border-top:1px solid #e5e7eb;padding-top:6px;margin-top:4px">
        <span>Total</span><strong>${(info.totalTokens || 0).toLocaleString()} / ${(info.maxTokens || 0).toLocaleString()}</strong>
      </div>
      <div class="cv2-context-row"><span>Est. cost</span><strong>$${info.estCost || '0.00'}</strong></div>`;
    if (this.config.showBudget && this.budget !== undefined && this.budget > 0) {
      html += `
      <div class="cv2-context-row"><span>Budget</span><strong>$${Number(this.budget).toFixed(2)}</strong></div>`;
    }
    this.el.contextPopover.innerHTML = html;
  }

  // ── Conversations Sidebar ─────────────────────────────

  async refreshConversations() {
    try {
      const all = await this.idb.getAll();
      this.conversations = all.map(c => ({
        id: c.id,
        title: c.title || (c.messages?.find(m => m.role === 'user')?.content || '').substring(0, 30) || 'Untitled',
        created_at: c.created_at,
      })).sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));

      this._renderConversations();
    } catch (err) {
      console.error('[Sidebar] refresh error:', err);
    }
  }

  _renderConversations() {
    const container = this.el.conversations;
    if (this.conversations.length === 0) {
      container.innerHTML = '<div class="cv2-conv-empty">No saved conversations</div>';
      this.el.sidebarActions.style.display = 'none';
      return;
    }

    this.el.sidebarActions.style.display = 'flex';
    container.innerHTML = this.conversations.map(c => `
      <div class="cv2-conv-item ${c.id === this.activeConvId ? 'cv2-active' : ''}" data-id="${this._escAttr(c.id)}">
        <span class="cv2-conv-title">${this._escHtml(this._truncTitle(c.title))}</span>
        <button class="cv2-conv-delete ${this.deleteConfirmId === c.id ? 'cv2-confirming' : ''}" data-del="${this._escAttr(c.id)}" title="Delete">
          ${this.deleteConfirmId === c.id
            ? '<span class="material-icons" style="font-size:14px">check</span>'
            : '<span class="material-icons" style="font-size:14px">delete</span>'}
        </button>
      </div>
    `).join('');

    // Bind clicks
    container.querySelectorAll('.cv2-conv-item').forEach(el => {
      el.addEventListener('click', () => this._selectConversation(el.dataset.id));
    });

    container.querySelectorAll('.cv2-conv-delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this._confirmDelete(btn.dataset.del);
      });
    });
  }

  async _selectConversation(id) {
    if (id === this.activeConvId) return;
    this.activeConvId = id;
    this._renderConversations();

    // Load from IDB and send to server
    const data = await this.idb.get(id);
    if (data) {
      this.ws.send({ type: 'load_conversation', data });

      // Rebuild chat UI with loaded messages
      this.showChat();
      this.el.messages.innerHTML = '';
      this._renderLoadedMessages(data.messages || []);
      this._scrollToBottom();
    }
  }

  _confirmDelete(id) {
    if (this.deleteConfirmId === id) {
      // Actually delete
      this.idb.delete(id).then(() => {
        if (this.activeConvId === id) this.activeConvId = '';
        this.deleteConfirmId = null;
        this.refreshConversations();
      });
    } else {
      this.deleteConfirmId = id;
      this._renderConversations();
      clearTimeout(this.deleteTimeout);
      this.deleteTimeout = setTimeout(() => {
        this.deleteConfirmId = null;
        this._renderConversations();
      }, 3000);
    }
  }

  _updateConversationTitle(id, title) {
    this.idb.get(id).then(data => {
      if (data) {
        data.title = title;
        data.updated_at = new Date().toISOString();
        this.idb.put(data).then(() => this.refreshConversations());
      }
    });
  }

  // ── Render loaded messages ────────────────────────────

  _renderLoadedMessages(messages) {
    for (const msg of messages) {
      if (msg.content_stale) continue;
      const content = msg.content || '';
      const images = msg.images;
      const imagesStale = msg.images_stale;

      if (msg.role === 'user') {
        const el = document.createElement('div');
        el.className = 'cv2-msg-user';
        let imagesHtml = '';
        if (images && !imagesStale) {
          imagesHtml = '<div class="cv2-msg-user-images">' +
            images.map(img => {
              const src = img.startsWith('data:') ? img : `data:image/png;base64,${img}`;
              return `<img src="${src}" alt="Uploaded">`;
            }).join('') + '</div>';
        }
        el.innerHTML = `<div class="cv2-msg-user-bubble">${imagesHtml}${this.md.render(content)}</div>`;
        this.el.messages.appendChild(el);
        this._wrapAllImages(el);
      } else if (msg.role === 'assistant') {
        const modelInfo = this.models.find(m => m.model === this.currentModel);
        const icon = modelInfo?.icon || '';
        const label = modelInfo?.label || '';
        const { images: extractedImages, text } = this._extractInlineImages(content);

        const el = document.createElement('div');
        el.className = 'cv2-msg-assistant';
        el.innerHTML = `
          <div class="cv2-msg-header">
            ${icon ? `<img src="${this.config.staticBase}/${icon}" alt="">` : ''}
            <span>${this._escHtml(label)}</span>
          </div>
          <div class="cv2-msg-body">${this.md.render(text)}</div>
        `;

        // Add images
        const body = el.querySelector('.cv2-msg-body');
        if (images && !imagesStale) {
          for (const img of images) {
            const imgEl = document.createElement('img');
            imgEl.className = 'cv2-generated-image';
            imgEl.src = this._toDataUri(img);
            body.appendChild(imgEl);
          }
        }
        for (const imgSrc of extractedImages) {
          const imgEl = document.createElement('img');
          imgEl.className = 'cv2-generated-image';
          imgEl.src = imgSrc;
          body.appendChild(imgEl);
        }

        this._addCopyButton(body, text);
        this._wrapAllImages(body);
        this.el.messages.appendChild(el);
      }
    }
  }

  // ── Image Extraction ──────────────────────────────────

  _extractInlineImages(text) {
    const images = [];
    let cleaned = text;

    // Markdown images with base64
    const mdRe = /!\[[^\]]*\]\((data:image\/[^;]+;base64,[A-Za-z0-9+/=]+)\)/g;
    let m;
    while ((m = mdRe.exec(text)) !== null) {
      images.push(m[1]);
      cleaned = cleaned.replace(m[0], '');
    }

    // Strip attachment:// placeholders and hallucinated local paths
    // Models often output ![alt](/mnt/data/...) or ![alt](sandbox:/...) after image generation
    cleaned = cleaned.replace(/!\[[^\]]*\]\((?:attachment:\/\/|sandbox:|\/mnt\/data\/|file:\/\/)[^)]*\)/g, '');

    // Plain data URIs
    const plainRe = /^(data:image\/[^;]+;base64,[A-Za-z0-9+/=]+)$/gm;
    while ((m = plainRe.exec(cleaned)) !== null) {
      if (!images.includes(m[1])) images.push(m[1]);
      cleaned = cleaned.replace(m[0], '');
    }

    cleaned = cleaned.replace(/\n{3,}/g, '\n\n').trim();
    return { images, text: cleaned };
  }

  // ── Copy Button ───────────────────────────────────────

  _addCopyButton(container, text) {
    const btn = document.createElement('button');
    btn.className = 'cv2-copy-btn';
    btn.innerHTML = '<span class="material-icons" style="font-size:16px">content_copy</span>';
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add('cv2-copied');
        btn.innerHTML = '<span class="material-icons" style="font-size:16px">check</span>';
        setTimeout(() => {
          btn.classList.remove('cv2-copied');
          btn.innerHTML = '<span class="material-icons" style="font-size:16px">content_copy</span>';
        }, 2000);
      } catch (e) {
        console.error('Copy failed:', e);
      }
    });
    container.style.position = 'relative';
    container.appendChild(btn);
  }

  // ── Image Wrapper & Lightbox ─────────────────────────

  /** Wrap a cv2-generated-image in a container with hover copy/download buttons. */
  _wrapImage(img) {
    const wrap = document.createElement('div');
    wrap.className = 'cv2-img-wrap';

    const actions = document.createElement('div');
    actions.className = 'cv2-img-actions';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'cv2-img-action-btn';
    copyBtn.title = 'Copy image';
    copyBtn.innerHTML = '<span class="material-icons">content_copy</span>';
    copyBtn.addEventListener('click', (e) => { e.stopPropagation(); this._copyImage(img.src, copyBtn); });

    const dlBtn = document.createElement('button');
    dlBtn.className = 'cv2-img-action-btn';
    dlBtn.title = 'Download image';
    dlBtn.innerHTML = '<span class="material-icons">download</span>';
    dlBtn.addEventListener('click', (e) => { e.stopPropagation(); this._downloadImage(img.src); });

    actions.appendChild(copyBtn);
    actions.appendChild(dlBtn);

    // Replace img in DOM with wrapper
    img.parentNode?.insertBefore(wrap, img);
    wrap.appendChild(img);
    wrap.appendChild(actions);

    img.addEventListener('click', () => this._openLightbox(img.src));
    return wrap;
  }

  /** Wrap all images (generated AND user-uploaded) with lightbox/copy/download. */
  _wrapAllImages(container) {
    // Generated images (assistant messages)
    container.querySelectorAll('img.cv2-generated-image').forEach(img => {
      if (!img.parentElement?.classList.contains('cv2-img-wrap')) {
        this._wrapImage(img);
      }
    });
    // User-uploaded images (user messages)
    container.querySelectorAll('.cv2-msg-user-images img').forEach(img => {
      if (!img.parentElement?.classList.contains('cv2-img-wrap')) {
        this._wrapImage(img);
      }
    });
  }

  _openLightbox(src) {
    // Remove existing
    document.querySelector('.cv2-lightbox')?.remove();

    const lb = document.createElement('div');
    lb.className = 'cv2-lightbox';
    lb.addEventListener('click', (e) => { if (e.target === lb) lb.remove(); });

    const img = document.createElement('img');
    img.src = src;
    lb.appendChild(img);

    const actions = document.createElement('div');
    actions.className = 'cv2-lightbox-actions';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'cv2-lightbox-btn';
    copyBtn.innerHTML = '<span class="material-icons">content_copy</span> Copy';
    copyBtn.addEventListener('click', () => this._copyImage(src, copyBtn));

    const dlBtn = document.createElement('button');
    dlBtn.className = 'cv2-lightbox-btn';
    dlBtn.innerHTML = '<span class="material-icons">download</span> Download';
    dlBtn.addEventListener('click', () => this._downloadImage(src));

    actions.appendChild(copyBtn);
    actions.appendChild(dlBtn);
    lb.appendChild(actions);

    const closeBtn = document.createElement('button');
    closeBtn.className = 'cv2-lightbox-btn cv2-lightbox-close';
    closeBtn.innerHTML = '<span class="material-icons">close</span>';
    closeBtn.addEventListener('click', () => lb.remove());
    lb.appendChild(closeBtn);

    // ESC to close
    const onKey = (e) => { if (e.key === 'Escape') { lb.remove(); document.removeEventListener('keydown', onKey); } };
    document.addEventListener('keydown', onKey);

    document.body.appendChild(lb);
  }

  async _copyImage(src, btn) {
    try {
      const resp = await fetch(src);
      const blob = await resp.blob();
      const pngBlob = blob.type === 'image/png' ? blob : await this._toPngBlob(src);
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': pngBlob })]);
      const orig = btn.innerHTML;
      btn.innerHTML = btn.classList.contains('cv2-lightbox-btn')
        ? '<span class="material-icons">check</span> Copied!'
        : '<span class="material-icons">check</span>';
      setTimeout(() => { btn.innerHTML = orig; }, 2000);
    } catch (e) {
      console.error('Copy image failed:', e);
    }
  }

  _toPngBlob(src) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
        c.getContext('2d').drawImage(img, 0, 0);
        c.toBlob(resolve, 'image/png');
      };
      img.src = src;
    });
  }

  _downloadImage(src) {
    const a = document.createElement('a');
    a.href = src;
    a.download = `generated-image-${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // ── File Drop, Paste & Selection ──────────────────────

  async _handleFileDrop(dataTransfer) {
    if (!dataTransfer.files.length) return;
    const files = Array.from(dataTransfer.files);
    const imageFiles = files.filter(f => f.type.startsWith('image/'));
    const docFiles = files.filter(f => !f.type.startsWith('image/'));

    // Compress and add images
    for (const f of imageFiles) {
      const dataUri = await this._compressImage(f);
      if (dataUri) this._addPendingImage(dataUri);
    }

    // Upload document files
    if (docFiles.length) await this._uploadFiles(docFiles);
  }

  async _handlePaste(e) {
    const items = Array.from(e.clipboardData?.items || []);
    const imageItems = items.filter(i => i.type.startsWith('image/'));
    if (imageItems.length === 0) return;

    e.preventDefault();
    for (const item of imageItems) {
      const file = item.getAsFile();
      if (!file) continue;
      const dataUri = await this._compressImage(file);
      if (dataUri) this._addPendingImage(dataUri);
    }
  }

  async _handleFileSelection(files) {
    const imageFiles = files.filter(f => f.type.startsWith('image/'));
    const docFiles = files.filter(f => !f.type.startsWith('image/'));

    for (const f of imageFiles) {
      const dataUri = await this._compressImage(f);
      if (dataUri) this._addPendingImage(dataUri);
    }

    if (docFiles.length) await this._uploadFiles(docFiles);

    // Fire callback (e.g. auto-analyze from quick action)
    if (this._onFilesReady) {
      const cb = this._onFilesReady;
      this._onFilesReady = null;
      cb();
    }
  }

  _addPendingImage(dataUri) {
    if (this._pendingImages.length >= 4) return; // max 4 images
    this._pendingImages.push(dataUri);
    this._renderAttachments();

    // Also post to server (so server can include with next message)
    fetch(`/api/llming/image-paste/${this.sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: this._pendingImages.map(d => d.split(',')[1]) }),
    }).catch(err => console.error('[Paste] POST failed:', err));
  }

  _removePendingImage(index) {
    this._pendingImages.splice(index, 1);
    this._renderAttachments();
    // Update server
    fetch(`/api/llming/image-paste/${this.sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: this._pendingImages.map(d => d.split(',')[1]) }),
    }).catch(() => {});
  }

  _removePendingFile(fileId) {
    this._pendingFiles = this._pendingFiles.filter(f => f.fileId !== fileId);
    this._renderAttachments();
    // Notify server to remove file and rebuild document context
    this.ws.send({ type: 'file_removed', file_id: fileId });
  }

  _renderAttachments() {
    const hasAttachments = this._pendingImages.length > 0 || this._pendingFiles.length > 0;
    this.el.attachments.style.display = hasAttachments ? 'flex' : 'none';

    let html = '';

    // Image thumbnails
    for (let i = 0; i < this._pendingImages.length; i++) {
      html += `
        <div class="cv2-attachment cv2-attachment-image">
          <img src="${this._pendingImages[i]}" alt="Image">
          <span class="cv2-attachment-remove" data-img-idx="${i}">&times;</span>
        </div>
      `;
    }

    // File pills
    for (const f of this._pendingFiles) {
      const icon = this._fileIcon(f.mimeType);
      const sizeStr = f.size < 1024 ? `${f.size} B` : f.size < 1048576 ? `${(f.size / 1024).toFixed(0)} KB` : `${(f.size / 1048576).toFixed(1)} MB`;
      html += `
        <div class="cv2-attachment">
          <span class="material-icons" style="font-size:14px;color:${icon.color}">${icon.name}</span>
          <span>${this._escHtml(f.name)}</span>
          <span style="color:#9ca3af;font-size:11px">${sizeStr}</span>
          <span class="cv2-attachment-remove" data-file-id="${this._escAttr(f.fileId)}">&times;</span>
        </div>
      `;
    }

    this.el.attachments.innerHTML = html;

    // Bind remove handlers
    this.el.attachments.querySelectorAll('[data-img-idx]').forEach(btn => {
      btn.addEventListener('click', () => this._removePendingImage(parseInt(btn.dataset.imgIdx)));
    });
    this.el.attachments.querySelectorAll('[data-file-id]').forEach(btn => {
      btn.addEventListener('click', () => this._removePendingFile(btn.dataset.fileId));
    });
  }

  /** Convert raw base64 or data URI to a proper data URI, sniffing MIME from magic bytes. */
  _toDataUri(val) {
    if (val.startsWith('data:')) return val;
    if (val.startsWith('iVBOR')) return `data:image/png;base64,${val}`;
    if (val.startsWith('/9j/'))  return `data:image/jpeg;base64,${val}`;
    if (val.startsWith('R0lG'))  return `data:image/gif;base64,${val}`;
    if (val.startsWith('UklG'))  return `data:image/webp;base64,${val}`;
    return `data:application/octet-stream;base64,${val}`;
  }

  _fileIcon(mimeType) {
    if (!mimeType) return { name: 'description', color: '#6b7280' };
    if (mimeType.includes('pdf')) return { name: 'picture_as_pdf', color: '#ef4444' };
    if (mimeType.includes('word') || mimeType.includes('docx')) return { name: 'description', color: '#2563eb' };
    if (mimeType.includes('sheet') || mimeType.includes('xlsx') || mimeType.includes('excel')) return { name: 'table_chart', color: '#16a34a' };
    if (mimeType.startsWith('image/')) return { name: 'image', color: '#7c3aed' };
    return { name: 'description', color: '#6b7280' };
  }

  async _compressImage(file, maxDim = 1920, maxBytes = 200000) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          let w = img.width, h = img.height;
          if (w > maxDim || h > maxDim) {
            const ratio = Math.min(maxDim / w, maxDim / h);
            w = Math.round(w * ratio);
            h = Math.round(h * ratio);
          }
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);

          // Try progressively lower quality
          for (const q of [0.8, 0.6, 0.4, 0.25]) {
            const dataUri = canvas.toDataURL('image/jpeg', q);
            if (dataUri.length * 0.75 <= maxBytes || q === 0.25) {
              resolve(dataUri);
              return;
            }
          }
          resolve(canvas.toDataURL('image/jpeg', 0.25));
        };
        img.onerror = () => resolve(null);
        img.src = reader.result;
      };
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
  }

  async _uploadFiles(files) {
    const formData = new FormData();
    for (const f of files) formData.append('files', f);

    try {
      const res = await fetch(`/api/llming/upload/${this.sessionId}`, {
        method: 'POST',
        headers: { 'X-User-Id': this.config.userId },
        body: formData,
      });
      const data = await res.json();
      if (data.files) {
        for (const f of data.files) {
          this._pendingFiles.push(f);
        }
        this._renderAttachments();
        // Notify server to rebuild document context
        this.ws.send({ type: 'file_uploaded' });
      }
    } catch (err) {
      console.error('[Upload] Failed:', err);
    }
  }

  // ── Export ────────────────────────────────────────────

  async _exportAll() {
    try {
      const allConvs = await this.idb.getAll();
      // Simple JSON export
      const blob = new Blob([JSON.stringify(allConvs, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversations_${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('[Export] Failed:', err);
    }
  }

  _showClearAllDialog() {
    const count = this.conversations.length;
    if (!count) return;
    const overlay = document.createElement('div');
    overlay.className = 'cv2-dialog-overlay';
    overlay.innerHTML = `
      <div class="cv2-dialog">
        <div class="cv2-dialog-title">Delete all conversations?</div>
        <div class="cv2-dialog-body">This will permanently delete ${count} conversation${count !== 1 ? 's' : ''}. This action cannot be undone.</div>
        <div class="cv2-dialog-actions">
          <button class="cv2-dialog-btn cv2-dialog-cancel">Cancel</button>
          <button class="cv2-dialog-btn cv2-dialog-confirm">Delete all</button>
        </div>
      </div>`;
    document.getElementById('chat-app').appendChild(overlay);
    overlay.querySelector('.cv2-dialog-cancel').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector('.cv2-dialog-confirm').addEventListener('click', () => {
      overlay.remove();
      this._clearAllConversations();
    });
  }

  async _clearAllConversations() {
    try {
      for (const c of this.conversations) {
        await this.idb.delete(c.id);
      }
      this.conversations = [];
      this.activeConvId = '';
      this._renderConversations();
    } catch (err) {
      console.error('[Clear] Failed:', err);
    }
  }

  // ── Helpers ───────────────────────────────────────────

  _scrollToBottom() {
    if (this.el.messages) {
      requestAnimationFrame(() => {
        this.el.messages.scrollTop = this.el.messages.scrollHeight;
      });
    }
  }

  _autoResizeTextarea() {
    const ta = this.el.textarea;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }

  _updateSendButton() {
    if (this.streaming) {
      this.el.sendBtn.classList.add('cv2-streaming');
      this.el.sendIcon.textContent = 'stop';
    } else {
      this.el.sendBtn.classList.remove('cv2-streaming');
      this.el.sendIcon.textContent = 'send';
    }
  }

  /** Cycle theme: light → dark → auto (system) */
  _toggleTheme() {
    const modes = ['light', 'dark', 'auto'];
    const current = this._themeMode || 'auto';
    const next = modes[(modes.indexOf(current) + 1) % modes.length];
    this._themeMode = next;
    try { localStorage.setItem('chat-theme', next); } catch {}
    this._applyThemeMode(next);
  }

  _applyThemeMode(mode) {
    const root = document.getElementById('chat-app');
    const icons = { light: 'light_mode', dark: 'dark_mode', auto: 'brightness_auto' };
    if (this.el.themeIcon) this.el.themeIcon.textContent = icons[mode] || 'brightness_auto';

    if (mode === 'auto') {
      const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root?.classList.toggle('cv2-dark', dark);
    } else {
      root?.classList.toggle('cv2-dark', mode === 'dark');
    }
  }

  _applyStoredTheme() {
    try {
      const stored = localStorage.getItem('chat-theme');
      this._themeMode = stored || 'auto';
      this._applyThemeMode(this._themeMode);

      // Follow live system changes when in auto mode
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (this._themeMode === 'auto') this._applyThemeMode('auto');
      });
    } catch {}
  }

  /** Find a fast/small model for quick tasks like document summaries. */
  _findFastModel() {
    const fast = this.models.find(m =>
      /flash|nano|mini|haiku|small|gpt-4o-mini/i.test(m.model) ||
      /flash|nano|mini|haiku|small/i.test(m.label)
    );
    return fast ? fast.model : null;
  }

  _truncTitle(t) {
    if (!t) return 'Untitled';
    return t.length > 30 ? t.substring(0, 28) + '...' : t;
  }

  _escHtml(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  _escAttr(s) {
    return this._escHtml(s);
  }
}


/* ══════════════════════════════════════════════════════════
   Boot — wait for NiceGUI to render the #chat-app mount point
   ══════════════════════════════════════════════════════════ */

function _bootChat() {
  const config = window.__CHAT_CONFIG__;
  if (!config) {
    console.error('[Chat] No __CHAT_CONFIG__ found');
    return;
  }

  // NiceGUI renders ui.html() content asynchronously via Vue.
  // Poll until the mount point exists in the DOM.
  function tryInit() {
    const root = document.getElementById('chat-app');
    if (root) {
      const app = new ChatApp(config);
      app.init().catch(err => console.error('[Chat] Init error:', err));
    } else {
      setTimeout(tryInit, 100);
    }
  }
  tryInit();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _bootChat);
} else {
  _bootChat();
}
