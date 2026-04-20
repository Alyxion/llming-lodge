/**
 * chat-app-core.js — ChatApp constructor, init, render, bindEvents, boot
 *
 * Loaded LAST — after chat-features.js, standalone classes, and all feature modules.
 * Applies accumulated _ChatAppProto methods to ChatApp.prototype, then boots the app.
 */

// ── Apply all feature module methods to ChatApp.prototype ──
Object.assign(ChatApp.prototype, window._ChatAppProto);

// ── Build merged message handler dispatch table ──
const _featureMessageHandlers = ChatFeatures.messageHandlers();

// ── Core prototype methods ──────────────────────────────────

Object.assign(ChatApp.prototype, {

  _init_constructor(config) {
    this.config = config;
    this.idb = new IDBStore();

    // Console log capture (ring buffer for debug API access)
    this._consoleLogs = [];
    this._consoleLogMax = 500;
    this._hookConsole();

    // Document plugin registry.
    // - registerBuiltinPlugins: host-owned transient plugins (mermaid, rich_mcp,
    //   kantini_result, followup). Defined in lodge's builtin-plugins.js.
    // - registerLlmingDocPlugins: all document-type plugins (plotly, latex,
    //   table, text_doc/word, presentation/powerpoint/pptx, html_sandbox,
    //   email_draft). Defined in llming-docs and served from /doc-static/.
    this.docPlugins = window.DocPluginRegistry ? new window.DocPluginRegistry() : null;
    if (this.docPlugins) {
      if (window.registerBuiltinPlugins) window.registerBuiltinPlugins(this.docPlugins);
      if (window.registerLlmingDocPlugins) window.registerLlmingDocPlugins(this.docPlugins);
    }

    // Cross-block reference store
    this._blockDataStore = window.BlockDataStore ? new window.BlockDataStore() : null;
    if (this.docPlugins && this._blockDataStore) {
      this.docPlugins.setBlockStore(this._blockDataStore);
    }

    // Register MCP-provided client renderers (injected at page load time)
    if (this.docPlugins && window.__CHAT_RENDERERS__) {
      for (const r of window.__CHAT_RENDERERS__) {
        if (r.type === 'inline') {
          // Inline pattern renderer — no lang, just JS that calls registry.registerInline()
          if (r.css) {
            const style = document.createElement('style');
            style.dataset.mcpRenderer = 'inline';
            style.textContent = r.css;
            document.head.appendChild(style);
          }
          try {
            const fn = new Function('registry', r.js);
            fn(this.docPlugins);
          } catch (err) {
            console.error('[MCP] Failed to register inline renderer:', err);
          }
          continue;
        }
        if (!r.lang || !r.js || this.docPlugins.has(r.lang)) continue;
        if (r.css) {
          const style = document.createElement('style');
          style.dataset.mcpRenderer = r.lang;
          style.textContent = r.css;
          document.head.appendChild(style);
        }
        try {
          const registerFn = new Function('registry', 'lang', r.js);
          registerFn(this.docPlugins, r.lang);
        } catch (err) {
          console.error(`[MCP] Failed to register renderer for "${r.lang}":`, err);
        }
      }
      delete window.__CHAT_RENDERERS__;
    }

    // ── Doc-tool auto-enable ──────────────────────────────────────────
    // MCP editing tools (e.g. "Plotly Charts") start disabled each session.
    // When a fenced code block of a doc type is rendered (either from a new
    // LLM response or a restored conversation), we auto-enable that type's
    // editing tools so the LLM can modify the document.
    //
    // Three call sites ensure coverage regardless of timing:
    //  1. onBlockRendered callback  — fires when a fenced block renders.
    //     Works when WS is connected AND tools are discovered (mid-session).
    //  2. handleSessionInit         — fires on WS connect. Covers the common
    //     case where MCP discovery completed before WS connected (in-process
    //     servers are fast), so session_init already has MCP groups but no
    //     subsequent tools_updated will arrive.
    //  3. handleToolsUpdated        — fires when MCP discovery completes after
    //     WS connect.  Covers the slow-discovery case.
    //
    // _autoEnabledDocTypes prevents duplicate toggles across all three paths.
    // MCP group labels for auto-enable on first doc_created. Sourced from
    // llming_docs.get_mcp_group_labels() via frontend config — no hard-
    // coded type → label pairs here.
    this._docGroupLabels = window.__CHAT_CONFIG__?.docGroupLabels || {};
    this._autoEnabledDocTypes = new Set();

    // Track inline doc blocks rendered in chat for the sidebar.
    // Under the tool-only policy, the fenced-block render path is disabled
    // for doc types (see DocPluginRegistry.fencedBlockAllowed). This callback
    // still fires for tool-driven renders (via _injectToolDocBlock) and for
    // ephemeral render plugins like mermaid / rich_mcp / kantini_result.
    if (this.docPlugins) {
      this.docPlugins.onBlockRendered((info) => {
        // Skip lightbox/preview/window re-renders — only track chat-inline blocks
        if (info.blockId.startsWith('lightbox-') || info.blockId.startsWith('ws-preview-') || info.blockId.startsWith('pv-')) return;

        // Auto-enable per-type editing tools on first render of each doc type
        this._autoEnableDocTools(info.lang);
        const now = Date.now();
        const idx = this.inlineDocBlocks.findIndex(b => b.id === info.id);
        const entry = {
          id: info.id,
          lang: info.lang,
          name: info.name,
          data: info.data,
          blockId: info.blockId,
          element: info.element,
          timestamp: now,
          source: 'inline',
        };
        if (idx >= 0) {
          this.inlineDocBlocks[idx] = entry;
        } else {
          this.inlineDocBlocks.push(entry);
        }
        this._renderDocList();
      });
    }

    this.md = new MarkdownRenderer(this.docPlugins);
    this.ws = null;

    // Core state
    this.sessionId = config.sessionId;
    this.userName = config.userName || '';
    this.fullName = config.fullName || '';
    this._enforcedTheme = config.enforcedTheme || '';
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
    this._receivedImages = [];

    // Quick actions (populated from server via session_init)
    this.quickActions = [];

    // Teams (for nudge ownership)
    this.teams = config.teams || [];

    // View manager — only ONE exclusive view at a time: 'chat' | 'preset' | 'explorer'
    this._activeView = 'chat';
    this._closeActiveView = null;

    // UI state
    this.chatVisible = false;
    this.sidebarVisible = true;
    this.modelDropdownOpen = false;
    this.plusMenuOpen = false;
    this.suggestionsMenuOpen = false;
    this.contextPopoverOpen = false;
    this.gearMenuOpen = false;
    this.incognito = false;

    // DOM refs (set in render())
    this.el = {};

    // Permissions set (from server config)
    this._permissions = new Set(config.permissions || []);

    // Let feature modules initialize their state
    ChatFeatures.initAllState(this);

    // Populate app extension manifests from server config (lazy — scripts not loaded yet)
    if (this._appExt && config.appExtensions) {
      this._appExt.manifests = config.appExtensions;
    }
  },

  /** Check if the current user has a specific permission. */
  hasPerm(perm) { return this._permissions.has(perm); },

  /**
   * Hook console.log/warn/error to capture messages in a ring buffer.
   * Originals still fire so DevTools works normally.
   */
  _hookConsole() {
    const self = this;
    const _origLog = console.log;
    const _origWarn = console.warn;
    const _origError = console.error;

    function _capture(level, origFn, args) {
      origFn.apply(console, args);
      try {
        const entry = {
          ts: Date.now(),
          level,
          msg: Array.from(args).map(a => {
            if (a instanceof Error) return `${a.message}\n${a.stack || ''}`;
            if (typeof a === 'object') { try { return JSON.stringify(a); } catch { return String(a); } }
            return String(a);
          }).join(' '),
        };
        self._consoleLogs.push(entry);
        if (self._consoleLogs.length > self._consoleLogMax) {
          self._consoleLogs.splice(0, self._consoleLogs.length - self._consoleLogMax);
        }
      } catch (_) { /* never break console */ }
    }

    console.log = function() { _capture('log', _origLog, arguments); };
    console.warn = function() { _capture('warn', _origWarn, arguments); };
    console.error = function() { _capture('error', _origError, arguments); };
  },

  async init() {
    await this.idb.open();
    // Gate persistence — incognito sessions are never written to IndexedDB.
    // Track incognito conversation IDs so late-arriving save_conversation
    // messages (after exiting incognito) are still blocked.
    this._incognitoConvIds = new Set();
    const _realPut = this.idb.put.bind(this.idb);
    this.idb.put = (data) => {
      if (this.incognito) {
        if (data?.id) this._incognitoConvIds.add(data.id);
        return Promise.resolve();
      }
      if (data?.id && this._incognitoConvIds.has(data.id)) return Promise.resolve();
      return _realPut(data);
    };
    // Expose globally for /chat interop
    window.IDBStore = {
      ready: true,
      get: (id) => this.idb.get(id),
      put: (data) => this.idb.put(data),
      getAll: () => this.idb.getAll(),
      getAllMeta: () => this.idb.getAllMeta(),
      delete: (id) => this.idb.delete(id),
      getPreset: (id) => this.idb.getPreset(id),
      putPreset: (data) => this.idb.putPreset(data),
      deletePreset: (id) => this.idb.deletePreset(id),
      getAllPresets: (type) => this.idb.getAllPresets(type),
    };

    // Load local favorites from IndexedDB
    try {
      this.favoriteNudges = await this.idb.getFavorites();
    } catch (e) { console.warn('[FAV] Failed to load from IDB:', e); }

    this.render();
    this._applyTheme();
    ChatFeatures.cacheAllEls(this);
    this.bindEvents();
    ChatFeatures.bindAllEvents(this);
    this._updateSendButton();
    this._startDraftSaver();
    await this.refreshConversations();
    // Only restore previous conversation on page reload, not fresh navigation
    const navType = performance.getEntriesByType('navigation')[0]?.type;
    if (navType === 'reload') {
      await this._restoreLastConversation();
    }
    this.connectWebSocket();
  },

  /** Translate a key using the current translations dict. */
  t(key, params = {}) {
    const tr = this.config.translations || {};
    let val = tr[key] || key;
    for (const [k, v] of Object.entries(params)) {
      val = val.replace(`{${k}}`, v);
    }
    return val;
  },

  /** Auto-enable per-type doc editing tools when a fenced block is first rendered. */
  _autoEnableDocTools(lang) {
    const groupLabel = this._docGroupLabels[lang];
    if (!groupLabel || this._autoEnabledDocTypes.has(lang)) return;
    if (!this.ws || !this.tools || !this.tools.length) return;
    // Find the collapsed MCP group entry by its name/group_id (which equals the label)
    const tool = this.tools.find(t => t.name === groupLabel || t.group_id === groupLabel);
    if (!tool) return;  // MCP not discovered yet — don't mark as done, retry on tools_updated
    if (!tool.enabled) {
      tool.enabled = true;
      this.ws.send({ type: 'toggle_tool', name: groupLabel, enabled: true });
      this._renderPlusMenu();
    }
    this._autoEnabledDocTypes.add(lang);  // Only mark done after successful lookup
  },

  /** Re-check auto-enable after tools_updated or session_init (blocks may render before WS connects). */
  _autoEnableForRenderedBlocks() {
    if (!this.inlineDocBlocks || !this.tools || !this.tools.length) return;
    for (const block of this.inlineDocBlocks) {
      this._autoEnableDocTools(block.lang);
    }
  },

  /** Pick a greeting from chat.greetings (time-aware, randomised). */
  _getGreeting(name) {
    // Pink / Joché mode — lovey-dovey greetings
    if (this._themeMode === 'pink' || this._themeMode === 'joche') {
      const n = name || '';
      const pinkGreetings = [
        'Hey mein Zuckerschnütchen, ' + n + '! \u{1F496}',
        'Hallöchen ' + n + ', du Traummensch! \u{1F496}',
        'Na, mein Pupsihasi ' + n + '! \u{1F496}',
        'Ach Schätzchen ' + n + ', da bist du ja! \u{1F496}',
        'Hallo mein Honigbienchen, ' + n + '! \u{1F496}',
        'Mausi ' + n + ', wie schön dass du da bist! \u{1F496}',
        'Hey Schnuckiputzi ' + n + '! \u{1F496}',
        'Willkommen zurück, Herzblatt ' + n + '! \u{1F496}',
        'Ohhh ' + n + ', mein Lieblingsmensch! \u{1F496}',
        'Hey Knuddelbär ' + n + '! \u{1F496}',
        'Hach ' + n + ', mein Sonnenschein! \u{1F496}',
        'Na Süßmaus ' + n + '! \u{1F496}',
      ];
      return pinkGreetings[Math.floor(Math.random() * pinkGreetings.length)];
    }
    const greetings = (this.config.translations || {})['chat.greetings'];
    if (!greetings || !Array.isArray(greetings) || greetings.length === 0) {
      return this.t('chat.greeting', { name });
    }
    let mins;
    if (this.config.fakeTime) {
      const [h, m] = this.config.fakeTime.split(':').map(Number);
      mins = h * 60 + (m || 0);
    } else {
      const now = new Date();
      mins = now.getHours() * 60 + now.getMinutes();
    }
    const parseTime = (s) => {
      const [h, m] = s.split(':').map(Number);
      return h * 60 + (m || 0);
    };
    const timed = [];
    const defaults = [];
    for (const g of greetings) {
      if (g.from && g.to) {
        const f = parseTime(g.from), t = parseTime(g.to);
        const match = f <= t ? (mins >= f && mins < t) : (mins >= f || mins < t);
        if (match) timed.push(g);
      } else {
        defaults.push(g);
      }
    }
    const pool = timed.length > 0 ? timed : defaults;
    if (pool.length === 0) return this.t('chat.greeting', { name });
    const chosen = pool[Math.floor(Math.random() * pool.length)];
    return (chosen.text || '').replace('{name}', name || '');
  },

  _applyTheme() {
    const t = this.config.theme;
    if (!t?.accent) return;
    const root = document.getElementById('chat-app');
    if (!root) return;
    root.style.setProperty('--chat-accent', t.accent);
    if (t.accentRgb) root.style.setProperty('--chat-accent-rgb', t.accentRgb);
    if (t.accentHover) root.style.setProperty('--chat-accent-hover', t.accentHover);
    if (t.accentLight) root.style.setProperty('--chat-accent-light', t.accentLight);
  },

  // ── WebSocket ─────────────────────────────────────────

  connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}${this.config.wsPath}`;

    this.ws = new ChatWebSocket(url, {
      onMessage: (msg) => this.handleMessage(msg),
      onOpen: () => {
        console.log('[Chat] WebSocket connected');
        this._setStatus('connected');
        try {
          const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
          const favUids = (this.favoriteNudges || []).map(f => f.uid).filter(Boolean);
          this.ws.send({ type: 'client_hello', timezone: tz, favorite_uids: favUids });
        } catch (e) { /* ignore */ }
      },
      onClose: (e) => {
        console.log('[Chat] WebSocket closed:', e.code);
        this._setStatus('disconnected');
      },
      onError: () => this._setStatus('error'),
    });
    this.ws.connect();
  },

  _setStatus(status) {
    if (status === 'disconnected' || status === 'error') {
      this._appExtDeactivateAll();
    }
  },

  // ── Message Dispatch ──────────────────────────────────

  handleMessage(msg) {
    // API keys dialog handler (ephemeral, set while dialog is open)
    if (msg.type && msg.type.startsWith('apikeys:') && this._apiKeysDialogHandler) {
      return this._apiKeysDialogHandler(msg);
    }
    // Check feature-registered handlers first
    const handler = _featureMessageHandlers[msg.type];
    if (typeof handler === 'function') {
      return handler(this, msg);
    }
    if (typeof handler === 'string' && typeof this[handler] === 'function') {
      return this[handler](msg);
    }
    // Core-only handlers
    switch (msg.type) {
      case 'heartbeat_ack': return; // ignore
      case 'dev_reload':
        console.log('[Chat] Dev reload triggered — reloading page');
        location.reload();
        return;
      default:
        console.warn('[Chat] Unknown message type:', msg.type);
    }
  },

  // ── View ──────────────────────────────────────────────

  showChat() {
    // Tear down any active overlay first
    if (this._activeView !== 'chat') {
      this._switchView('chat');
    }
    this.chatVisible = true;
    // Hide incognito toggle only once a conversation has messages
    // (allow toggling off before first message is sent)
    const incBtn = document.getElementById('cv2-incognito-toggle');
    if (incBtn && !this.incognito) {
      const hasMessages = this.el.messages && this.el.messages.children.length > 0;
      incBtn.style.display = hasMessages ? 'none' : '';
    }
    this._stopPlaceholderCycle();
    // Clear project-view hiding classes (defensive against hot-reload stale DOM)
    this.el.initialView.classList.remove('cv2-pv-hidden');
    this.el.messagesWrap.classList.remove('cv2-pv-hidden');
    this.el.initialView.style.display = 'none';
    this.el.messagesWrap.classList.remove('cv2-messages-wrap-hidden');
    const banner = document.getElementById('cv2-banner');
    if (banner) banner.style.display = 'none';
    const wrapper = this.el.initialView.closest('.cv2-chat-wrapper');
    if (wrapper) wrapper.classList.remove('cv2-initial-mode');
    const inputArea = wrapper?.querySelector('.cv2-input-area');
    if (inputArea) {
      inputArea.classList.add('cv2-input-animating');
      inputArea.addEventListener('animationend', () => inputArea.classList.remove('cv2-input-animating'), { once: true });
    }
    // Render nudge header when transitioning from initial view
    if (this._activeNudgeMeta && !this.el.messages.querySelector('.cv2-nudge-sticky')) {
      this._renderNudgeChatHeader(this._activeNudgeMeta);
    }
    if (!this._speechMode && !this._rtActive) this.el.textarea.focus();
  },

  // ── Render ────────────────────────────────────────────

  render() {
    const root = document.getElementById('chat-app');
    if (!root) return;

    const avatarUrl = this.config.userAvatar || '';
    const avatarHtml = avatarUrl
      ? `<img src="${this._escAttr(avatarUrl)}" alt="User">`
      : '<span class="material-icons">person</span>';

    const logoLink = this.config.appLogoLink || '';

    root.innerHTML = `
      <!-- Sidebar -->
      <div class="cv2-sidebar" id="cv2-sidebar">
        <div class="cv2-sidebar-header">
          ${this.config.appLogo ? (logoLink
            ? `<a href="${this._escAttr(logoLink)}" class="cv2-sidebar-logo-link"><img class="cv2-sidebar-logo" src="${this._escAttr(this.config.appLogo)}" alt=""></a>`
            : `<img class="cv2-sidebar-logo" src="${this._escAttr(this.config.appLogo)}" alt="">`) : ''}
          <span class="cv2-sidebar-title">${this._escHtml(this.config.appTitle || this.t('chat.sidebar_title'))}</span>
          <button class="cv2-sidebar-close-btn" id="cv2-sidebar-close" title="${this.t('chat.toggle_sidebar')}">
            <span class="material-icons" style="font-size:18px">chevron_left</span>
          </button>
        </div>
        <button class="cv2-new-chat-btn" id="cv2-new-chat" title="${this.t('chat.new_chat')}">
          <img src="${this.config.staticBase}/icons/phosphor/regular/pencil-simple.svg" style="width:16px;height:16px;opacity:0.6" alt=""> ${this.t('chat.new_chat')}
        </button>
        <button class="cv2-search-chats-btn" id="cv2-search-chats" title="${this.t('chat.search_chats')}">
          <span class="material-icons" style="font-size:16px">search</span> ${this.t('chat.search_chats')}
          <span class="cv2-search-chats-kbd">${navigator.platform.includes('Mac') ? '\u2318K' : 'Ctrl+K'}</span>
        </button>
        <div class="cv2-sidebar-sections" id="cv2-sidebar-sections"></div>
        <div class="cv2-sidebar-budget" id="cv2-budget"></div>
        <div class="cv2-sidebar-footer" style="position:relative">
          <div class="cv2-sidebar-avatar" id="cv2-sidebar-avatar">${avatarHtml}</div>
          <div class="cv2-gear-popover cv2-dev-menu" id="cv2-dev-menu" style="position:absolute;bottom:100%;left:0;margin-bottom:4px;z-index:999">
            <button class="cv2-gear-popover-item" id="cv2-dev-reset-tools">
              <span class="material-icons">restart_alt</span> Reset tool defaults
            </button>
            <button class="cv2-gear-popover-item" id="cv2-dev-clear-storage">
              <span class="material-icons">cleaning_services</span> Clear localStorage
            </button>
          </div>
          <span class="cv2-sidebar-user-name" id="cv2-sidebar-user-name"></span>
          <div style="flex:1"></div>
          <div class="cv2-sidebar-gear-wrap" id="cv2-gear-wrap">
            <button class="cv2-sidebar-gear-btn" id="cv2-gear-btn" title="Settings">
              <span class="material-icons" style="font-size:18px">settings</span>
            </button>
            <div class="cv2-gear-popover" id="cv2-gear-popover">
              <div id="cv2-gear-lang-slot"></div>
              <div id="cv2-gear-theme-slot"></div>
              <button class="cv2-gear-popover-item" id="cv2-speech-toggle">
                <span class="material-icons" id="cv2-speech-icon">volume_up</span> ${this.t('chat.speech_response')}
              </button>
              <button class="cv2-gear-popover-item" id="cv2-voice-settings-btn">
                <span class="material-icons">tune</span> ${this.t('chat.voice_settings')}
                <span class="material-icons cv2-gear-arrow">chevron_right</span>
              </button>
              <button class="cv2-gear-popover-item" id="cv2-data-mgmt-btn">
                <span class="material-icons">swap_vert</span> Import / Export
                <span class="material-icons cv2-gear-arrow">chevron_right</span>
              </button>
              <button class="cv2-gear-popover-item cv2-danger" id="cv2-clear-all">
                <span class="material-icons">delete_sweep</span> ${this.t('chat.clear_all')}
              </button>
            </div>
            <div class="cv2-gear-submenu" id="cv2-data-mgmt-popover" style="display:none">
              <button class="cv2-gear-popover-item" id="cv2-export-all">
                <span class="material-icons">download</span> ${this.t('chat.export')}
              </button>
              <button class="cv2-gear-popover-item" id="cv2-import-all">
                <span class="material-icons">upload</span> ${this.t('chat.import') || 'Import'}
              </button>
            </div>
            <div class="cv2-gear-popover" id="cv2-dev-settings-popover">
              <button class="cv2-gear-popover-item" id="cv2-dev-api-keys">
                <span class="material-icons">vpn_key</span> API Keys
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Mobile sidebar backdrop -->
      <div class="cv2-sidebar-backdrop" id="cv2-sidebar-backdrop"></div>

      <!-- Main -->
      <div class="cv2-main">
        <!-- Tech blueprint background -->
        <svg class="cv2-bg-anim" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <filter id="cv2-glow"><feGaussianBlur stdDeviation="2" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          </defs>

          <!-- Dot grid regions — clusters fade in/out like areas being scanned -->
          <g class="cv2-bg-dot">
            <!-- Region A (left-center) 6x5 -->
            <g opacity="0">
              <animate attributeName="opacity" values="0;0.07;0.07;0" dur="18s" begin="0s" repeatCount="indefinite"/>
              <circle cx="180" cy="240" r="1"/><circle cx="210" cy="240" r="1"/><circle cx="240" cy="240" r="1"/><circle cx="270" cy="240" r="1"/><circle cx="300" cy="240" r="1"/><circle cx="330" cy="240" r="1"/>
              <circle cx="180" cy="270" r="1"/><circle cx="210" cy="270" r="1"/><circle cx="240" cy="270" r="1"/><circle cx="270" cy="270" r="1"/><circle cx="300" cy="270" r="1"/><circle cx="330" cy="270" r="1"/>
              <circle cx="180" cy="300" r="1"/><circle cx="210" cy="300" r="1"/><circle cx="240" cy="300" r="1"/><circle cx="270" cy="300" r="1"/><circle cx="300" cy="300" r="1"/><circle cx="330" cy="300" r="1"/>
              <circle cx="180" cy="330" r="1"/><circle cx="210" cy="330" r="1"/><circle cx="240" cy="330" r="1"/><circle cx="270" cy="330" r="1"/><circle cx="300" cy="330" r="1"/><circle cx="330" cy="330" r="1"/>
              <circle cx="180" cy="360" r="1"/><circle cx="210" cy="360" r="1"/><circle cx="240" cy="360" r="1"/><circle cx="270" cy="360" r="1"/><circle cx="300" cy="360" r="1"/><circle cx="330" cy="360" r="1"/>
            </g>
            <!-- Region B (right) 5x6 -->
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0.06;0.06;0" dur="22s" begin="5s" repeatCount="indefinite"/>
              <circle cx="820" cy="420" r="1"/><circle cx="850" cy="420" r="1"/><circle cx="880" cy="420" r="1"/><circle cx="910" cy="420" r="1"/><circle cx="940" cy="420" r="1"/>
              <circle cx="820" cy="450" r="1"/><circle cx="850" cy="450" r="1"/><circle cx="880" cy="450" r="1"/><circle cx="910" cy="450" r="1"/><circle cx="940" cy="450" r="1"/>
              <circle cx="820" cy="480" r="1"/><circle cx="850" cy="480" r="1"/><circle cx="880" cy="480" r="1"/><circle cx="910" cy="480" r="1"/><circle cx="940" cy="480" r="1"/>
              <circle cx="820" cy="510" r="1"/><circle cx="850" cy="510" r="1"/><circle cx="880" cy="510" r="1"/><circle cx="910" cy="510" r="1"/><circle cx="940" cy="510" r="1"/>
              <circle cx="820" cy="540" r="1"/><circle cx="850" cy="540" r="1"/><circle cx="880" cy="540" r="1"/><circle cx="910" cy="540" r="1"/><circle cx="940" cy="540" r="1"/>
              <circle cx="820" cy="570" r="1"/><circle cx="850" cy="570" r="1"/><circle cx="880" cy="570" r="1"/><circle cx="910" cy="570" r="1"/><circle cx="940" cy="570" r="1"/>
            </g>
            <!-- Region C (upper-right) 4x4 -->
            <g opacity="0">
              <animate attributeName="opacity" values="0;0.05;0.05;0" dur="16s" begin="10s" repeatCount="indefinite"/>
              <circle cx="950" cy="120" r="1"/><circle cx="980" cy="120" r="1"/><circle cx="1010" cy="120" r="1"/><circle cx="1040" cy="120" r="1"/>
              <circle cx="950" cy="150" r="1"/><circle cx="980" cy="150" r="1"/><circle cx="1010" cy="150" r="1"/><circle cx="1040" cy="150" r="1"/>
              <circle cx="950" cy="180" r="1"/><circle cx="980" cy="180" r="1"/><circle cx="1010" cy="180" r="1"/><circle cx="1040" cy="180" r="1"/>
              <circle cx="950" cy="210" r="1"/><circle cx="980" cy="210" r="1"/><circle cx="1010" cy="210" r="1"/><circle cx="1040" cy="210" r="1"/>
            </g>
            <!-- Region D (bottom-left) 5x3 -->
            <g opacity="0">
              <animate attributeName="opacity" values="0;0.05;0" dur="14s" begin="7s" repeatCount="indefinite"/>
              <circle cx="100" cy="580" r="1"/><circle cx="130" cy="580" r="1"/><circle cx="160" cy="580" r="1"/><circle cx="190" cy="580" r="1"/><circle cx="220" cy="580" r="1"/>
              <circle cx="100" cy="610" r="1"/><circle cx="130" cy="610" r="1"/><circle cx="160" cy="610" r="1"/><circle cx="190" cy="610" r="1"/><circle cx="220" cy="610" r="1"/>
              <circle cx="100" cy="640" r="1"/><circle cx="130" cy="640" r="1"/><circle cx="160" cy="640" r="1"/><circle cx="190" cy="640" r="1"/><circle cx="220" cy="640" r="1"/>
            </g>
            <!-- Region E (center) 4x3 -->
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0.06;0.06;0" dur="20s" begin="12s" repeatCount="indefinite"/>
              <circle cx="520" cy="340" r="1"/><circle cx="550" cy="340" r="1"/><circle cx="580" cy="340" r="1"/><circle cx="610" cy="340" r="1"/>
              <circle cx="520" cy="370" r="1"/><circle cx="550" cy="370" r="1"/><circle cx="580" cy="370" r="1"/><circle cx="610" cy="370" r="1"/>
              <circle cx="520" cy="400" r="1"/><circle cx="550" cy="400" r="1"/><circle cx="580" cy="400" r="1"/><circle cx="610" cy="400" r="1"/>
            </g>
          </g>

          <!-- Highlight rectangles on dot grid regions -->
          <g fill="none" class="cv2-bg-line" stroke-width="0.5">
            <rect x="170" y="230" width="170" height="140" rx="3" opacity="0">
              <animate attributeName="opacity" values="0;0;0.08;0.08;0;0" dur="18s" begin="2s" repeatCount="indefinite"/>
            </rect>
            <rect x="810" y="410" width="140" height="170" rx="3" opacity="0">
              <animate attributeName="opacity" values="0;0;0.06;0.06;0;0" dur="22s" begin="8s" repeatCount="indefinite"/>
            </rect>
            <rect x="940" y="110" width="110" height="110" rx="3" opacity="0">
              <animate attributeName="opacity" values="0;0.06;0.06;0;0" dur="16s" begin="12s" repeatCount="indefinite"/>
            </rect>
          </g>

          <!-- Flowing data stream curves (varied paths) -->
          <g fill="none" class="cv2-bg-line" stroke-linecap="round">
            <path d="M-50,320 C180,120 320,480 580,230 S880,380 1080,180 S1180,240 1260,340"
                  stroke-width="0.8" opacity="0.06"
                  stroke-dasharray="600 1200" stroke-dashoffset="0">
              <animate attributeName="stroke-dashoffset" values="1800;0;-1800" dur="48s" repeatCount="indefinite"/>
            </path>
            <path d="M-50,530 C220,380 340,720 580,480 S860,280 1020,580 S1140,520 1260,430"
                  stroke-width="0.8" opacity="0.05"
                  stroke-dasharray="500 1100" stroke-dashoffset="0">
              <animate attributeName="stroke-dashoffset" values="0;-1600;0" dur="56s" repeatCount="indefinite"/>
            </path>
            <path d="M80,760 C260,620 420,790 680,580 S980,680 1120,540"
                  stroke-width="0.5" opacity="0.04"
                  stroke-dasharray="400 900" stroke-dashoffset="0">
              <animate attributeName="stroke-dashoffset" values="1300;-1300" dur="62s" repeatCount="indefinite"/>
            </path>
            <path d="M160,40 C380,220 540,20 780,160 S1020,60 1200,200"
                  stroke-width="0.5" opacity="0.04"
                  stroke-dasharray="350 800" stroke-dashoffset="0">
              <animate attributeName="stroke-dashoffset" values="-1150;1150" dur="53s" repeatCount="indefinite"/>
            </path>
            <path d="M-30,170 C200,270 350,70 540,220 S800,120 1000,300 S1130,230 1260,170"
                  stroke-width="0.6" opacity="0.04"
                  stroke-dasharray="450 1000" stroke-dashoffset="0">
              <animate attributeName="stroke-dashoffset" values="1450;-1450" dur="50s" repeatCount="indefinite"/>
            </path>
            <path d="M-20,660 C180,560 400,740 580,640 S820,720 1020,560 S1140,600 1260,640"
                  stroke-width="0.6" opacity="0.035"
                  stroke-dasharray="380 850" stroke-dashoffset="0">
              <animate attributeName="stroke-dashoffset" values="-1230;1230" dur="58s" repeatCount="indefinite"/>
            </path>
            <path d="M-40,430 C140,330 300,560 500,420 S720,520 920,360 S1090,410 1260,460"
                  stroke-width="0.7" opacity="0.04"
                  stroke-dasharray="520 1050" stroke-dashoffset="0">
              <animate attributeName="stroke-dashoffset" values="0;1570;0" dur="67s" repeatCount="indefinite"/>
            </path>
          </g>

          <!-- Full-width grid that fades in/out — assembles H then V -->
          <g fill="none" class="cv2-bg-line" stroke-width="0.3">
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0.035;0.035;0.035;0.035;0" dur="28s" repeatCount="indefinite"/>
              <line x1="0" y1="100" x2="1200" y2="100"/><line x1="0" y1="200" x2="1200" y2="200"/>
              <line x1="0" y1="300" x2="1200" y2="300"/><line x1="0" y1="400" x2="1200" y2="400"/>
              <line x1="0" y1="500" x2="1200" y2="500"/><line x1="0" y1="600" x2="1200" y2="600"/>
              <line x1="0" y1="700" x2="1200" y2="700"/>
            </g>
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0;0.035;0.035;0.035;0" dur="28s" begin="2s" repeatCount="indefinite"/>
              <line x1="150" y1="0" x2="150" y2="800"/><line x1="300" y1="0" x2="300" y2="800"/>
              <line x1="450" y1="0" x2="450" y2="800"/><line x1="600" y1="0" x2="600" y2="800"/>
              <line x1="750" y1="0" x2="750" y2="800"/><line x1="900" y1="0" x2="900" y2="800"/>
              <line x1="1050" y1="0" x2="1050" y2="800"/>
            </g>
          </g>

          <!-- Bezier construction — control points appear, then curve draws through -->
          <g>
            <g class="cv2-bg-dot">
              <circle cx="120" cy="520" r="2.5" opacity="0">
                <animate attributeName="opacity" values="0;0;0.12;0.12;0.12;0.12;0.08;0" dur="22s" repeatCount="indefinite"/>
              </circle>
              <circle cx="340" cy="380" r="2.5" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0.12;0.12;0.12;0.08;0" dur="22s" begin="0.5s" repeatCount="indefinite"/>
              </circle>
              <circle cx="560" cy="580" r="2.5" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0;0.12;0.12;0.08;0" dur="22s" begin="1s" repeatCount="indefinite"/>
              </circle>
              <circle cx="780" cy="420" r="2.5" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0;0;0.12;0.08;0" dur="22s" begin="1.5s" repeatCount="indefinite"/>
              </circle>
              <circle cx="1000" cy="540" r="2.5" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0;0;0.12;0.08;0" dur="22s" begin="2s" repeatCount="indefinite"/>
              </circle>
            </g>
            <path d="M120,520 C200,420 270,380 340,380 S460,580 560,580 S680,420 780,420 S900,540 1000,540"
                  fill="none" class="cv2-bg-line" stroke-width="1" stroke-linecap="round"
                  stroke-dasharray="1200" stroke-dashoffset="1200" opacity="0">
              <animate attributeName="opacity" values="0;0;0;0;0;0.07;0.07;0" dur="22s" repeatCount="indefinite"/>
              <animate attributeName="stroke-dashoffset" values="1200;1200;1200;1200;1200;0;0;0" dur="22s" repeatCount="indefinite"/>
            </path>
            <g fill="none" class="cv2-bg-line" stroke-width="0.5" stroke-dasharray="3 3" opacity="0">
              <animate attributeName="opacity" values="0;0;0;0;0;0.06;0.03;0" dur="22s" repeatCount="indefinite"/>
              <line x1="120" y1="520" x2="200" y2="420"/>
              <line x1="340" y1="380" x2="270" y2="380"/><line x1="340" y1="380" x2="460" y2="580"/>
              <line x1="780" y1="420" x2="680" y2="420"/><line x1="780" y1="420" x2="900" y2="540"/>
            </g>
          </g>

          <!-- Bezier construction 2 — offset timing -->
          <g>
            <g class="cv2-bg-dot">
              <circle cx="80" cy="180" r="2" opacity="0">
                <animate attributeName="opacity" values="0;0;0.10;0.10;0.10;0.10;0.06;0" dur="26s" begin="10s" repeatCount="indefinite"/>
              </circle>
              <circle cx="300" cy="100" r="2" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0.10;0.10;0.10;0.06;0" dur="26s" begin="10.5s" repeatCount="indefinite"/>
              </circle>
              <circle cx="520" cy="220" r="2" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0;0.10;0.10;0.06;0" dur="26s" begin="11s" repeatCount="indefinite"/>
              </circle>
              <circle cx="740" cy="130" r="2" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0;0;0.10;0.06;0" dur="26s" begin="11.5s" repeatCount="indefinite"/>
              </circle>
              <circle cx="960" cy="200" r="2" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0;0;0.10;0.06;0" dur="26s" begin="12s" repeatCount="indefinite"/>
              </circle>
              <circle cx="1140" cy="110" r="2" opacity="0">
                <animate attributeName="opacity" values="0;0;0;0;0;0.10;0.06;0" dur="26s" begin="12.5s" repeatCount="indefinite"/>
              </circle>
            </g>
            <path d="M80,180 C160,120 230,100 300,100 S420,220 520,220 S640,130 740,130 S860,200 960,200 S1060,110 1140,110"
                  fill="none" class="cv2-bg-line" stroke-width="0.8" stroke-linecap="round"
                  stroke-dasharray="1400" stroke-dashoffset="1400" opacity="0">
              <animate attributeName="opacity" values="0;0;0;0;0;0.06;0.06;0" dur="26s" begin="10s" repeatCount="indefinite"/>
              <animate attributeName="stroke-dashoffset" values="1400;1400;1400;1400;1400;0;0;0" dur="26s" begin="10s" repeatCount="indefinite"/>
            </path>
            <g fill="none" class="cv2-bg-line" stroke-width="0.4" stroke-dasharray="3 3" opacity="0">
              <animate attributeName="opacity" values="0;0;0;0;0;0.05;0.02;0" dur="26s" begin="10s" repeatCount="indefinite"/>
              <line x1="80" y1="180" x2="160" y2="120"/>
              <line x1="300" y1="100" x2="230" y2="100"/><line x1="300" y1="100" x2="420" y2="220"/>
              <line x1="740" y1="130" x2="640" y2="130"/><line x1="740" y1="130" x2="860" y2="200"/>
            </g>
          </g>

          <!-- Floating nodes — pulsing dots (shifted positions vs StagForge) -->
          <g class="cv2-bg-dot">
            <circle cx="380" cy="220" r="1.5" opacity="0"><animate attributeName="opacity" values="0;0.12;0" dur="8s" repeatCount="indefinite"/></circle>
            <circle cx="720" cy="360" r="2" opacity="0"><animate attributeName="opacity" values="0;0.10;0" dur="11s" begin="2s" repeatCount="indefinite"/></circle>
            <circle cx="520" cy="280" r="1.5" opacity="0"><animate attributeName="opacity" values="0;0.08;0" dur="9s" begin="4s" repeatCount="indefinite"/></circle>
            <circle cx="930" cy="260" r="1.8" opacity="0"><animate attributeName="opacity" values="0;0.10;0" dur="10s" begin="1s" repeatCount="indefinite"/></circle>
            <circle cx="420" cy="580" r="1.5" opacity="0"><animate attributeName="opacity" values="0;0.09;0" dur="12s" begin="3s" repeatCount="indefinite"/></circle>
            <circle cx="780" cy="590" r="2" opacity="0"><animate attributeName="opacity" values="0;0.07;0" dur="14s" begin="5s" repeatCount="indefinite"/></circle>
            <circle cx="1080" cy="170" r="1.5" opacity="0"><animate attributeName="opacity" values="0;0.08;0" dur="10s" begin="6s" repeatCount="indefinite"/></circle>
            <circle cx="220" cy="470" r="1.8" opacity="0"><animate attributeName="opacity" values="0;0.06;0" dur="13s" begin="7s" repeatCount="indefinite"/></circle>
            <circle cx="630" cy="690" r="1.5" opacity="0"><animate attributeName="opacity" values="0;0.07;0" dur="11s" begin="8s" repeatCount="indefinite"/></circle>
            <circle cx="140" cy="130" r="1.8" opacity="0"><animate attributeName="opacity" values="0;0.06;0" dur="15s" begin="3s" repeatCount="indefinite"/></circle>
          </g>

          <!-- Connection lines between nodes -->
          <g fill="none" class="cv2-bg-line" stroke-width="0.5" stroke-linecap="round">
            <line x1="380" y1="220" x2="520" y2="280" opacity="0"><animate attributeName="opacity" values="0;0.06;0" dur="8s" begin="1s" repeatCount="indefinite"/></line>
            <line x1="520" y1="280" x2="720" y2="360" opacity="0"><animate attributeName="opacity" values="0;0.05;0" dur="11s" begin="3s" repeatCount="indefinite"/></line>
            <line x1="720" y1="360" x2="780" y2="590" opacity="0"><animate attributeName="opacity" values="0;0.04;0" dur="10s" begin="5s" repeatCount="indefinite"/></line>
            <line x1="420" y1="580" x2="720" y2="360" opacity="0"><animate attributeName="opacity" values="0;0.05;0" dur="12s" begin="2s" repeatCount="indefinite"/></line>
            <line x1="930" y1="260" x2="1080" y2="170" opacity="0"><animate attributeName="opacity" values="0;0.04;0" dur="9s" begin="4s" repeatCount="indefinite"/></line>
            <line x1="380" y1="220" x2="220" y2="470" opacity="0"><animate attributeName="opacity" values="0;0.04;0" dur="13s" begin="6s" repeatCount="indefinite"/></line>
            <line x1="140" y1="130" x2="380" y2="220" opacity="0"><animate attributeName="opacity" values="0;0.05;0" dur="10s" begin="2s" repeatCount="indefinite"/></line>
            <line x1="780" y1="590" x2="630" y2="690" opacity="0"><animate attributeName="opacity" values="0;0.04;0" dur="12s" begin="7s" repeatCount="indefinite"/></line>
          </g>

          <!-- Circuit-style L-connectors (unique to this app) -->
          <g fill="none" class="cv2-bg-line" stroke-width="0.4" stroke-linecap="round">
            <path d="M720,360 L720,260 L930,260" opacity="0">
              <animate attributeName="opacity" values="0;0;0.06;0.06;0" dur="14s" begin="3s" repeatCount="indefinite"/>
            </path>
            <path d="M420,580 L420,690 L630,690" opacity="0">
              <animate attributeName="opacity" values="0;0.05;0.05;0;0" dur="16s" begin="8s" repeatCount="indefinite"/>
            </path>
            <path d="M140,130 L140,280 L380,280 L380,220" opacity="0">
              <animate attributeName="opacity" values="0;0;0.05;0.05;0;0" dur="19s" begin="5s" repeatCount="indefinite"/>
            </path>
          </g>

          <!-- Dimension annotation (unique architectural element) -->
          <g fill="none" class="cv2-bg-line" stroke-width="0.5" opacity="0">
            <animate attributeName="opacity" values="0;0;0.06;0.06;0" dur="24s" begin="4s" repeatCount="indefinite"/>
            <line x1="400" y1="730" x2="400" y2="750"/>
            <line x1="400" y1="740" x2="700" y2="740"/>
            <line x1="700" y1="730" x2="700" y2="750"/>
            <circle cx="400" cy="740" r="1.5" class="cv2-bg-dot"/><circle cx="700" cy="740" r="1.5" class="cv2-bg-dot"/>
          </g>

          <!-- Brand logo watermark moved to cv2-initial-view -->

          <!-- Measurement tick marks along edges -->
          <g class="cv2-bg-line" stroke-width="0.4">
            <g opacity="0">
              <animate attributeName="opacity" values="0;0.06;0.06;0" dur="20s" begin="3s" repeatCount="indefinite"/>
              <line x1="400" y1="780" x2="400" y2="790"/><line x1="430" y1="783" x2="430" y2="790"/>
              <line x1="460" y1="780" x2="460" y2="790"/><line x1="490" y1="783" x2="490" y2="790"/>
              <line x1="520" y1="780" x2="520" y2="790"/><line x1="550" y1="783" x2="550" y2="790"/>
              <line x1="580" y1="780" x2="580" y2="790"/><line x1="610" y1="783" x2="610" y2="790"/>
              <line x1="640" y1="780" x2="640" y2="790"/><line x1="670" y1="783" x2="670" y2="790"/>
              <line x1="700" y1="780" x2="700" y2="790"/>
            </g>
            <g opacity="0">
              <animate attributeName="opacity" values="0;0.05;0.05;0" dur="24s" begin="8s" repeatCount="indefinite"/>
              <line x1="1180" y1="250" x2="1190" y2="250"/><line x1="1183" y1="275" x2="1190" y2="275"/>
              <line x1="1180" y1="300" x2="1190" y2="300"/><line x1="1183" y1="325" x2="1190" y2="325"/>
              <line x1="1180" y1="350" x2="1190" y2="350"/><line x1="1183" y1="375" x2="1190" y2="375"/>
              <line x1="1180" y1="400" x2="1190" y2="400"/><line x1="1183" y1="425" x2="1190" y2="425"/>
              <line x1="1180" y1="450" x2="1190" y2="450"/>
            </g>
            <!-- Left edge ticks (new) -->
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0.05;0.05;0" dur="22s" begin="14s" repeatCount="indefinite"/>
              <line x1="10" y1="300" x2="20" y2="300"/><line x1="10" y1="330" x2="17" y2="330"/>
              <line x1="10" y1="360" x2="20" y2="360"/><line x1="10" y1="390" x2="17" y2="390"/>
              <line x1="10" y1="420" x2="20" y2="420"/><line x1="10" y1="450" x2="17" y2="450"/>
              <line x1="10" y1="480" x2="20" y2="480"/>
            </g>
          </g>

          <!-- Crosshair markers that blink -->
          <g class="cv2-bg-line" stroke-width="0.4" fill="none">
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0.1;0.1;0;0;0" dur="12s" begin="1s" repeatCount="indefinite"/>
              <line x1="470" y1="175" x2="470" y2="195"/><line x1="460" y1="185" x2="480" y2="185"/>
              <circle cx="470" cy="185" r="6"/>
            </g>
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0.08;0.08;0;0;0" dur="15s" begin="6s" repeatCount="indefinite"/>
              <line x1="760" y1="620" x2="760" y2="640"/><line x1="750" y1="630" x2="770" y2="630"/>
              <circle cx="760" cy="630" r="6"/>
            </g>
            <g opacity="0">
              <animate attributeName="opacity" values="0;0.07;0.07;0;0;0" dur="18s" begin="11s" repeatCount="indefinite"/>
              <line x1="1080" y1="510" x2="1080" y2="530"/><line x1="1070" y1="520" x2="1090" y2="520"/>
              <circle cx="1080" cy="520" r="6"/>
            </g>
            <!-- Extra crosshair at different position -->
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0.09;0.09;0;0" dur="20s" begin="15s" repeatCount="indefinite"/>
              <line x1="280" y1="140" x2="280" y2="160"/><line x1="270" y1="150" x2="290" y2="150"/>
              <circle cx="280" cy="150" r="6"/>
            </g>
          </g>

          <!-- Slowly rotating dashed rings (shifted center) -->
          <g fill="none" class="cv2-bg-line" stroke-width="0.5">
            <circle cx="650" cy="400" r="80" opacity="0.04" stroke-dasharray="8 14">
              <animateTransform attributeName="transform" type="rotate" from="0 650 400" to="360 650 400" dur="120s" repeatCount="indefinite"/>
            </circle>
            <circle cx="650" cy="400" r="120" opacity="0.03" stroke-dasharray="12 20">
              <animateTransform attributeName="transform" type="rotate" from="360 650 400" to="0 650 400" dur="180s" repeatCount="indefinite"/>
            </circle>
            <!-- Third ring at different center (unique) -->
            <circle cx="300" cy="600" r="60" opacity="0.03" stroke-dasharray="5 18">
              <animateTransform attributeName="transform" type="rotate" from="0 300 600" to="360 300 600" dur="90s" repeatCount="indefinite"/>
            </circle>
          </g>

          <!-- HUD data readout boxes -->
          <g fill="none" class="cv2-bg-line" stroke-width="0.4">
            <g opacity="0">
              <animate attributeName="opacity" values="0;0.06;0.06;0.06;0" dur="14s" begin="2s" repeatCount="indefinite"/>
              <rect x="60" y="90" width="50" height="24" rx="2"/>
              <line x1="66" y1="98" x2="86" y2="98" stroke-width="0.8"/>
              <line x1="66" y1="104" x2="100" y2="104" stroke-width="0.8"/>
            </g>
            <g opacity="0">
              <animate attributeName="opacity" values="0;0;0.05;0.05;0" dur="17s" begin="9s" repeatCount="indefinite"/>
              <rect x="1070" y="650" width="60" height="24" rx="2"/>
              <line x1="1076" y1="658" x2="1106" y2="658" stroke-width="0.8"/>
              <line x1="1076" y1="664" x2="1120" y2="664" stroke-width="0.8"/>
            </g>
            <!-- Third HUD box, different position (unique) -->
            <g opacity="0">
              <animate attributeName="opacity" values="0;0.05;0.05;0;0" dur="19s" begin="13s" repeatCount="indefinite"/>
              <rect x="1090" y="80" width="55" height="28" rx="2"/>
              <line x1="1096" y1="89" x2="1120" y2="89" stroke-width="0.8"/>
              <line x1="1096" y1="96" x2="1135" y2="96" stroke-width="0.8"/>
              <line x1="1096" y1="101" x2="1110" y2="101" stroke-width="0.5"/>
            </g>
          </g>

          <!-- Corner brackets -->
          <path d="M4 28 L4 4 L28 4" fill="none" class="cv2-bg-line" stroke-width="0.7" opacity="0.08"/>
          <path d="M1196 772 L1196 796 L1172 796" fill="none" class="cv2-bg-line" stroke-width="0.7" opacity="0.08"/>
          <path d="M1172 4 L1196 4 L1196 28" fill="none" class="cv2-bg-line" stroke-width="0.5" opacity="0.05"/>
          <path d="M4 772 L4 796 L28 796" fill="none" class="cv2-bg-line" stroke-width="0.5" opacity="0.05"/>
        </svg>
        <!-- Top bar -->
        <div class="cv2-topbar">
          <button class="cv2-topbar-btn" id="cv2-sidebar-toggle" style="display:none">
            <span class="material-icons">menu</span>
          </button>
          ${this.config.appLogo ? (logoLink
            ? `<a href="${this._escAttr(logoLink)}" class="cv2-topbar-logo-link"><img class="cv2-topbar-logo" src="${this._escAttr(this.config.appLogo)}" alt=""></a>`
            : `<img class="cv2-topbar-logo" src="${this._escAttr(this.config.appLogo)}" alt="">`) : ''}
          <div style="flex:1"></div>
          <button class="cv2-incognito-toggle" id="cv2-incognito-toggle" title="Incognito — chat won't be saved">
            <svg width="27" height="27" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-top:3px">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke-dasharray="4 3"/>
            </svg>
          </button>
          <div style="width:8px"></div>
          ${this.hasPerm('nudge_admin') ? `
          <div style="position:relative" id="cv2-admin-menu-wrap">
            <button class="cv2-topbar-btn" id="cv2-admin-btn" title="Admin">
              <span class="material-icons">admin_panel_settings</span>
            </button>
            <div class="cv2-gear-popover cv2-admin-popover" id="cv2-admin-popover">
              <button class="cv2-gear-popover-item" id="cv2-admin-identity">
                <span class="material-icons">swap_horiz</span> Switch identity
              </button>
            </div>
          </div>` : ''}
          <button class="cv2-topbar-btn cv2-workspace-toggle" id="cv2-workspace-toggle" title="Workspace">
            <span class="material-icons">dashboard</span>
          </button>
        </div>

        <!-- Chat wrapper -->
        <div class="cv2-chat-wrapper cv2-initial-mode">
          <!-- Initial view -->
          <div class="cv2-initial-view" id="cv2-initial-view">
            ${this.config.bgLogoSvg ? `<svg class="cv2-brand-logo" viewBox="-2 4 175 48" preserveAspectRatio="xMidYMid meet">${this.config.bgLogoSvg}</svg>` : ''}
            <div class="cv2-greeting-row">
              ${this.config.appMascot ? `<img class="cv2-mascot" src="${this._escAttr(this.incognito && this.config.appMascotIncognito ? this.config.appMascotIncognito : this.config.appMascot)}" data-default-src="${this._escAttr(this.config.appMascot)}" alt="">` : ''}
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
                <button class="cv2-input-btn" id="cv2-plus-btn" title="${this.t('chat.tools_actions')}">
                  <span class="material-icons">add</span>
                </button>
                <div class="cv2-plus-menu" id="cv2-plus-menu"></div>
                <button class="cv2-input-btn" id="cv2-suggestions-btn" title="${this.t('chat.suggestions')}">
                  <span class="material-icons">auto_awesome</span>
                </button>
                <div class="cv2-suggestions-menu" id="cv2-suggestions-menu"></div>
                <div style="flex:1"></div>
                <div style="position:relative">
                  <button class="cv2-model-btn" id="cv2-model-btn"></button>
                  <div class="cv2-model-dropdown" id="cv2-model-dropdown"></div>
                </div>
                <div class="cv2-context-circle" id="cv2-context-circle" title="${this.t('chat.context_usage')}">
                  <svg viewBox="0 0 36 36">
                    <circle class="cv2-bg" cx="18" cy="18" r="15.9"/>
                    <circle class="cv2-fg" id="cv2-context-fg" cx="18" cy="18" r="15.9"
                            stroke-dasharray="100 100" stroke-dashoffset="100" stroke="#e07020"/>
                  </svg>
                  <span class="cv2-context-pct" id="cv2-context-pct">0%</span>
                </div>
                <div class="cv2-context-popover" id="cv2-context-popover"></div>
                <button class="cv2-send-btn" id="cv2-send-btn" title="${this.t('chat.send')}">
                  <span class="material-icons" id="cv2-send-icon">send</span>
                </button>
                <button class="cv2-speech-mode-btn" id="cv2-speech-mode-btn" title="${this.t('chat.speech_mode')}">
                  <span class="material-icons">graphic_eq</span>
                </button>
              </div>
              <div class="cv2-inline-voice-bar" id="cv2-inline-voice-bar" style="display:none">
                <div class="cv2-ivb-avatar" id="cv2-ivb-avatar"></div>
                <div class="cv2-ivb-info">
                  <span class="cv2-ivb-status" id="cv2-ivb-status"></span>
                  <canvas class="cv2-ivb-wave" id="cv2-ivb-wave"></canvas>
                  <button class="cv2-ivb-ptt" id="cv2-ivb-ptt">
                    <span class="material-icons">mic</span>
                    <small class="cv2-ivb-ptt-hint">Space</small>
                  </button>
                  <div class="cv2-ivb-thinking" id="cv2-ivb-thinking" style="display:none">
                    <svg viewBox="0 0 120 40" width="60" height="20">
                      <circle cx="20" cy="20" r="6" fill="currentColor"><animate attributeName="r" values="6;10;6" dur="1.2s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0.4;1" dur="1.2s" repeatCount="indefinite"/></circle>
                      <circle cx="60" cy="20" r="6" fill="currentColor"><animate attributeName="r" values="6;10;6" dur="1.2s" begin="0.2s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0.4;1" dur="1.2s" begin="0.2s" repeatCount="indefinite"/></circle>
                      <circle cx="100" cy="20" r="6" fill="currentColor"><animate attributeName="r" values="6;10;6" dur="1.2s" begin="0.4s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0.4;1" dur="1.2s" begin="0.4s" repeatCount="indefinite"/></circle>
                    </svg>
                  </div>
                </div>
                <div class="cv2-ivb-actions">
                  <button class="cv2-ivb-btn cv2-ivb-send" id="cv2-ivb-send" title="${this.t('chat.voice_send_now')}">
                    <span class="material-icons">send</span>
                  </button>
                  <button class="cv2-ivb-btn cv2-ivb-interrupt" id="cv2-ivb-interrupt" title="${this.t('chat.voice_interrupt')}">
                    <span class="material-icons">front_hand</span>
                  </button>
                  <button class="cv2-ivb-btn cv2-ivb-exit" id="cv2-ivb-exit" title="${this.t('chat.voice_exit_speech')}">
                    <span class="material-icons">close</span>
                  </button>
                </div>
              </div>
            </div>
            <div class="cv2-disclaimer" id="cv2-disclaimer">${this.t('chat.disclaimer')}</div>
            ${this.config.bannerHtml ? `<div class="cv2-banner" id="cv2-banner">${this.config.bannerHtml}</div>` : ''}
          </div>
        </div>
      </div>

      <button class="cv2-mobile-mic-btn" id="cv2-mobile-mic-btn"><span class="material-icons">mic</span></button>
      <div class="cv2-voice-popup" id="cv2-voice-popup" style="display:none">
        <div class="cv2-voice-card">
          <div class="cv2-voice-avatar-slot" id="cv2-voice-avatar-slot"></div>
          <h3 id="cv2-voice-title"><span class="material-icons">mic</span> ${this.t('chat.voice_title')}</h3>
          <div class="cv2-voice-wave-strip">
            <canvas id="cv2-voice-wave" width="640" height="56"></canvas>
            <div class="cv2-voice-thinking" id="cv2-voice-thinking" style="display:none">
              <svg viewBox="0 0 120 40" width="120" height="40">
                <circle cx="20" cy="20" r="6" fill="rgba(255,255,255,0.7)"><animate attributeName="r" values="6;10;6" dur="1.2s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0.4;1" dur="1.2s" repeatCount="indefinite"/></circle>
                <circle cx="60" cy="20" r="6" fill="rgba(255,255,255,0.7)"><animate attributeName="r" values="6;10;6" dur="1.2s" begin="0.2s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0.4;1" dur="1.2s" begin="0.2s" repeatCount="indefinite"/></circle>
                <circle cx="100" cy="20" r="6" fill="rgba(255,255,255,0.7)"><animate attributeName="r" values="6;10;6" dur="1.2s" begin="0.4s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0.4;1" dur="1.2s" begin="0.4s" repeatCount="indefinite"/></circle>
              </svg>
            </div>
          </div>
          <div class="cv2-voice-analytics">
            <canvas id="cv2-voice-bands" width="320" height="42"></canvas>
            <canvas id="cv2-voice-loudness" width="320" height="42"></canvas>
          </div>
          <p class="cv2-voice-note" id="cv2-voice-note"></p>
          <div class="cv2-voice-actions">
            <button class="cv2-voice-btn-send" id="cv2-voice-send"><span class="material-icons">send</span> Send now</button>
            <button class="cv2-voice-btn-cancel" id="cv2-voice-cancel"><span class="material-icons">close</span> Cancel</button>
          </div>
          <div class="cv2-voice-mode-active" id="cv2-voice-mode-active">
            <button class="cv2-speech-send-btn" id="cv2-speech-send-btn">
              <span class="material-icons">send</span> ${this.t('chat.voice_send_now')}
            </button>
            <button class="cv2-speech-interrupt-btn" id="cv2-speech-interrupt-btn">
              <span class="material-icons">front_hand</span> ${this.t('chat.voice_interrupt')}
            </button>
            <button class="cv2-voice-mode-exit" id="cv2-voice-mode-exit">
              <span class="material-icons">close</span> ${this.t('chat.voice_exit_speech')}
            </button>
          </div>
        </div>
      </div>

      <div class="cv2-gear-submenu" id="cv2-voice-settings-menu" style="display:none">
        <label class="cv2-ptt-toggle-label">${this.t('chat.voice_ptt')}
          <div class="cv2-tool-toggle ${this._pushToTalk ? 'cv2-on' : ''}" id="cv2-ptt-toggle"></div>
        </label>
        <label>${this.t('chat.voice_select')}
          <select id="cv2-voice-select" class="cv2-voice-select"></select>
        </label>
        <label>${this.t('chat.voice_pause')} <span id="cv2-voice-wait-val">${this._voiceSilenceMs / 1000}s</span>
          <input type="range" id="cv2-voice-wait" min="500" max="5000" step="100" value="${this._voiceSilenceMs}">
        </label>
        <label>${this.t('chat.voice_sensitivity')} <span id="cv2-voice-thresh-val">${this._sensitivityLabel(this._voiceSilenceThreshold)}</span>
          <input type="range" id="cv2-voice-thresh" min="0.005" max="0.06" step="0.001" value="${0.065 - this._voiceSilenceThreshold}">
        </label>
      </div>

      <div class="cv2-voice-picker" id="cv2-voice-picker" style="display:none">
        <div class="cv2-voice-picker-card" id="cv2-voice-pick-live">
          <span class="material-icons">call</span>
          <strong>${this.t('chat.voice_mode_live')}</strong>
          <small>${this.t('chat.voice_mode_live_hint')}</small>
        </div>
        <div class="cv2-voice-picker-card" id="cv2-voice-pick-input">
          <span class="material-icons">graphic_eq</span>
          <strong>${this.t('chat.voice_mode_input')}</strong>
          <small>${this.t('chat.voice_mode_input_hint')}</small>
        </div>
      </div>

    `;

    // Cache DOM refs
    this.el = {
      sidebar: root.querySelector('#cv2-sidebar'),
      sidebarClose: root.querySelector('#cv2-sidebar-close'),
      sidebarToggle: root.querySelector('#cv2-sidebar-toggle'),
      sidebarUserName: root.querySelector('#cv2-sidebar-user-name'),
      newChat: root.querySelector('#cv2-new-chat'),
      sidebarSections: root.querySelector('#cv2-sidebar-sections'),
      gearLangSlot: root.querySelector('#cv2-gear-lang-slot'),
      gearWrap: root.querySelector('#cv2-gear-wrap'),
      gearBtn: root.querySelector('#cv2-gear-btn'),
      gearPopover: root.querySelector('#cv2-gear-popover'),
      gearThemeSlot: root.querySelector('#cv2-gear-theme-slot'),
      dataMgmtBtn: root.querySelector('#cv2-data-mgmt-btn'),
      dataMgmtPopover: root.querySelector('#cv2-data-mgmt-popover'),
      exportAll: root.querySelector('#cv2-export-all'),
      importAll: root.querySelector('#cv2-import-all'),
      clearAll: root.querySelector('#cv2-clear-all'),
      avatar: root.querySelector('#cv2-sidebar-avatar'),
      devMenu: root.querySelector('#cv2-dev-menu'),
      devResetTools: root.querySelector('#cv2-dev-reset-tools'),
      devClearStorage: root.querySelector('#cv2-dev-clear-storage'),
      devSettingsPopover: root.querySelector('#cv2-dev-settings-popover'),
      devApiKeysBtn: root.querySelector('#cv2-dev-api-keys'),
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
      disclaimer: root.querySelector('#cv2-disclaimer'),
      speechToggle: root.querySelector('#cv2-speech-toggle'),
      speechIcon: root.querySelector('#cv2-speech-icon'),
      voicePopup: root.querySelector('#cv2-voice-popup'),
      voiceAvatarSlot: root.querySelector('#cv2-voice-avatar-slot'),
      voiceTitle: root.querySelector('#cv2-voice-title'),
      voiceWave: root.querySelector('#cv2-voice-wave'),
      voiceBands: root.querySelector('#cv2-voice-bands'),
      voiceLoudness: root.querySelector('#cv2-voice-loudness'),
      voiceNote: root.querySelector('#cv2-voice-note'),
      voiceSend: root.querySelector('#cv2-voice-send'),
      voiceCancel: root.querySelector('#cv2-voice-cancel'),
      voiceSelect: root.querySelector('#cv2-voice-select'),
      voiceWaitSlider: root.querySelector('#cv2-voice-wait'),
      voiceWaitVal: root.querySelector('#cv2-voice-wait-val'),
      voiceThreshSlider: root.querySelector('#cv2-voice-thresh'),
      voiceThreshVal: root.querySelector('#cv2-voice-thresh-val'),
      speechModeBtn: root.querySelector('#cv2-speech-mode-btn'),
      voiceSettingsBtn: root.querySelector('#cv2-voice-settings-btn'),
      voiceSettingsMenu: root.querySelector('#cv2-voice-settings-menu'),
      voiceModeActive: root.querySelector('#cv2-voice-mode-active'),
      voiceModeExit: root.querySelector('#cv2-voice-mode-exit'),
      speechSendBtn: root.querySelector('#cv2-speech-send-btn'),
      speechInterruptBtn: root.querySelector('#cv2-speech-interrupt-btn'),
      voiceThinking: root.querySelector('#cv2-voice-thinking'),
      voicePicker: root.querySelector('#cv2-voice-picker'),
      voicePickLive: root.querySelector('#cv2-voice-pick-live'),
      voicePickInput: root.querySelector('#cv2-voice-pick-input'),
      ivbBar: root.querySelector('#cv2-inline-voice-bar'),
      ivbAvatar: root.querySelector('#cv2-ivb-avatar'),
      ivbStatus: root.querySelector('#cv2-ivb-status'),
      ivbWave: root.querySelector('#cv2-ivb-wave'),
      ivbThinking: root.querySelector('#cv2-ivb-thinking'),
      ivbPtt: root.querySelector('#cv2-ivb-ptt'),
      ivbSend: root.querySelector('#cv2-ivb-send'),
      ivbInterrupt: root.querySelector('#cv2-ivb-interrupt'),
      ivbExit: root.querySelector('#cv2-ivb-exit'),
      pttToggle: root.querySelector('#cv2-ptt-toggle'),
      mobileMicBtn: root.querySelector('#cv2-mobile-mic-btn'),
      adminBtn: root.querySelector('#cv2-admin-btn'),
      adminPopover: root.querySelector('#cv2-admin-popover'),
    };

    // Set greeting & user name
    const greetName = this.userName || 'there';
    this.el.greeting.textContent = this._getGreeting(greetName);
    this._lastGreetName = greetName;
    this.el.sidebarUserName.textContent = this.fullName || this.userName || this.t('chat.default_user');

    // Quick action cards (rendered after session_init provides them)
    this._renderQuickActions();

    // Restore theme from localStorage
    this._applyStoredTheme();
  },

  // ── Identity Switch (Admin) ──────────────────────────

  _openIdentityDialog() {
    const LS_KEY = 'llming_dev_overrides';
    const COOKIE_KEY = 'llming_dev_overrides';
    const HIST_KEY = 'cv2-admin-identity-history';

    const current = (() => {
      try { return JSON.parse(localStorage.getItem(LS_KEY) || 'null'); } catch { return null; }
    })();
    const history = (() => {
      try { return JSON.parse(localStorage.getItem(HIST_KEY) || '[]'); } catch { return []; }
    })();

    const overlay = document.createElement('div');
    overlay.className = 'cv2-identity-overlay';
    overlay.innerHTML = `
      <div class="cv2-identity-dialog">
        <div class="cv2-identity-header">
          <span class="material-icons" style="font-size:20px;color:var(--chat-accent)">swap_horiz</span>
          <h3>Switch Identity</h3>
          <button class="cv2-identity-close" id="cv2-identity-close"><span class="material-icons" style="font-size:18px">close</span></button>
        </div>
        <div class="cv2-identity-body">
          ${current ? `<div class="cv2-identity-active"><span class="material-icons">person</span> Acting as <strong style="margin-left:4px">${this._escHtml(current.given_name || '')} ${this._escHtml(current.surname || '')}</strong> <span style="opacity:0.6;margin-left:4px">(${this._escHtml(current.email || '')})</span></div>` : ''}
          <input type="text" class="cv2-identity-search" id="cv2-identity-search" placeholder="Search by name or email..." autofocus>
          <div class="cv2-identity-history" id="cv2-identity-history"></div>
          <div class="cv2-identity-results" id="cv2-identity-results"></div>
        </div>
        ${current ? `<div class="cv2-identity-footer"><button class="cv2-identity-reset-btn" id="cv2-identity-reset"><span class="material-icons" style="font-size:14px;vertical-align:middle">undo</span> Reset to self</button></div>` : ''}
      </div>
    `;
    document.getElementById('chat-app').appendChild(overlay);

    const close = () => overlay.remove();
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    overlay.querySelector('#cv2-identity-close').addEventListener('click', close);

    // Render history chips
    const histEl = overlay.querySelector('#cv2-identity-history');
    if (history.length > 0) {
      histEl.innerHTML = history.map((h, i) => `
        <button class="cv2-identity-chip" data-idx="${i}">
          ${h.photo_url ? `<img src="${this._escAttr(h.photo_url)}" alt="">` : ''}
          ${this._escHtml(h.display_name || h.email)}
        </button>
      `).join('');
      histEl.addEventListener('click', (e) => {
        const chip = e.target.closest('.cv2-identity-chip');
        if (!chip) return;
        const h = history[+chip.dataset.idx];
        if (h) selectUser(h);
      });
    }

    // Select user
    const selectUser = (u) => {
      const overrides = { given_name: u.given_name, surname: u.surname, email: u.email };
      localStorage.setItem(LS_KEY, JSON.stringify(overrides));
      document.cookie = `${COOKIE_KEY}=${encodeURIComponent(JSON.stringify(overrides))};path=/;max-age=31536000;SameSite=Strict`;
      // Save to recent history (dedup by email)
      const hist = history.filter(h => h.email !== u.email);
      hist.unshift({ display_name: u.display_name || `${u.given_name} ${u.surname}`, email: u.email, given_name: u.given_name, surname: u.surname, photo_url: u.photo_url || null });
      localStorage.setItem(HIST_KEY, JSON.stringify(hist.slice(0, 4)));
      location.reload();
    };

    // Reset
    overlay.querySelector('#cv2-identity-reset')?.addEventListener('click', () => {
      localStorage.removeItem(LS_KEY);
      document.cookie = `${COOKIE_KEY}=;path=/;max-age=0;SameSite=Strict`;
      location.reload();
    });

    // Search
    const searchInput = overlay.querySelector('#cv2-identity-search');
    const resultsEl = overlay.querySelector('#cv2-identity-results');
    let _searchTimer = null;
    searchInput.addEventListener('input', () => {
      clearTimeout(_searchTimer);
      const q = searchInput.value.trim();
      if (q.length < 2) { resultsEl.innerHTML = ''; return; }
      _searchTimer = setTimeout(async () => {
        try {
          const resp = await fetch(`/api/user-search?q=${encodeURIComponent(q)}`);
          if (!resp.ok) { resultsEl.innerHTML = ''; return; }
          const users = await resp.json();
          resultsEl.innerHTML = users.map((u, i) => `
            <div class="cv2-identity-row" data-idx="${i}">
              ${u.photo_url ? `<img src="${this._escAttr(u.photo_url)}" alt="">` : `<div class="cv2-identity-row-icon"><span class="material-icons" style="font-size:18px">person</span></div>`}
              <div class="cv2-identity-row-info">
                <div class="cv2-identity-row-name">${this._escHtml(u.display_name || '')}</div>
                <div class="cv2-identity-row-email">${this._escHtml(u.email || '')} ${u.department ? `· ${this._escHtml(u.department)}` : ''}</div>
              </div>
            </div>
          `).join('') || '<div style="padding:12px;color:var(--chat-text-muted);font-size:12px">No results</div>';
          resultsEl.querySelectorAll('.cv2-identity-row').forEach(row => {
            row.addEventListener('click', () => {
              selectUser(users[+row.dataset.idx]);
            });
          });
        } catch (e) {
          console.warn('[IDENTITY] Search error:', e);
        }
      }, 300);
    });
    searchInput.focus();
  },

  // ── Event Binding ─────────────────────────────────────

  bindEvents() {
    // Workspace toggle (topbar button)
    const wsToggle = document.getElementById('cv2-workspace-toggle');
    if (wsToggle) wsToggle.addEventListener('click', () => this.toggleWorkspace());

    // Admin menu toggle
    if (this.el.adminBtn) {
      this.el.adminBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const open = this.el.adminPopover.classList.toggle('cv2-visible');
        if (open) this._closeAllPopups('admin');
      });
      // Switch identity action
      document.getElementById('cv2-admin-identity')?.addEventListener('click', () => {
        this.el.adminPopover.classList.remove('cv2-visible');
        this._openIdentityDialog();
      });
    }

    // Workspace panel (HTML sandbox, etc.)
    document.addEventListener('cv2:open-workspace', (e) => this._openWorkspace(e.detail));

    // Show any doc plugin in workspace side pane
    document.addEventListener('cv2:show-in-workspace', (e) => {
      const { blockId } = e.detail || {};
      if (!blockId) return;
      const entry = this.inlineDocBlocks.find(b => b.blockId === blockId || b.id === blockId);
      if (!entry) return;
      if (!this.workspaceOpen) {
        this.workspaceOpen = true;
        this._ensureWorkspace();
        document.getElementById('cv2-workspace')?.classList.add('open');
        document.getElementById('cv2-workspace-toggle')?.classList.add('active');
        document.getElementById('chat-app')?.classList.add('cv2-ws-open');
        this._maybeCollapseSidebar();
      }
      this._showDocPreview(entry);
    });

    // Send
    this.el.sendBtn.addEventListener('click', () => this.sendMessage());
    this.el.textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.ctrlKey && e.shiftKey) {
        e.preventDefault();
        this._copyCurlCommand();
        return;
      }
      // Let the bolt system intercept Enter before sending to LLM
      if (e.key === 'Enter' && !e.shiftKey) {
        const hasBoltFn = typeof this._boltsOnKeydown === 'function';
        if (hasBoltFn) {
          const handled = this._boltsOnKeydown(e);
          if (handled) {
            console.log('[Chat] Bolt intercepted Enter — NOT sending to LLM');
            return;
          }
          console.log('[Chat] Bolt did NOT intercept Enter — sending to LLM');
        }
        // Hide bolt chips when falling through to LLM
        if (typeof this._boltsHideChips === 'function') this._boltsHideChips();
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-resize textarea + toggle send/speech button
    this.el.textarea.addEventListener('input', () => {
      this._autoResizeTextarea();
      this._updateSendButton();
    });

    // Sidebar toggle
    const chatRoot = document.getElementById('chat-app');
    const topbarLogo = chatRoot && (chatRoot.querySelector('.cv2-topbar-logo-link') || chatRoot.querySelector('.cv2-topbar-logo'));
    const backdrop = document.getElementById('cv2-sidebar-backdrop');
    const _isMobile = () => window.innerWidth <= 768;
    // Track whether sidebar was auto-collapsed (responsive) vs manually closed
    let _sidebarAutoCollapsed = false;
    const closeIcon = this.el.sidebarClose.querySelector('.material-icons');
    const _updateSidebarUI = (visible) => {
      if (_isMobile()) {
        // Mobile: full hide/show, no rail
        this.el.sidebar.classList.remove('cv2-sidebar-rail');
        this.el.sidebar.classList.toggle('cv2-hidden', !visible);
        this.el.sidebarToggle.style.display = visible ? 'none' : '';
        if (topbarLogo) topbarLogo.style.display = 'none';
        if (backdrop) backdrop.classList.toggle('cv2-visible', visible);
        if (closeIcon) closeIcon.textContent = 'chevron_left';
      } else {
        // Desktop: collapse to icon rail instead of hiding
        this.el.sidebar.classList.remove('cv2-hidden');
        this.el.sidebar.classList.toggle('cv2-sidebar-rail', !visible);
        this.el.sidebarToggle.style.display = 'none';
        if (topbarLogo) topbarLogo.style.display = 'none';
        if (backdrop) backdrop.classList.remove('cv2-visible');
        // Swap icon: menu (hamburger) in rail, chevron_left when expanded
        if (closeIcon) closeIcon.textContent = visible ? 'chevron_left' : 'menu';
      }
    };
    const toggleSidebar = () => {
      this.sidebarVisible = !this.sidebarVisible;
      // Manual toggle: clear auto-collapsed flag (user chose to close/open)
      _sidebarAutoCollapsed = false;
      _updateSidebarUI(this.sidebarVisible);
    };
    // Auto-collapse sidebar on mobile at startup
    if (_isMobile()) {
      this.sidebarVisible = false;
      _sidebarAutoCollapsed = true;
      _updateSidebarUI(false);
    } else {
      // Start with topbar logo hidden since sidebar is open on desktop
      if (topbarLogo) topbarLogo.style.display = 'none';
    }
    this.el.sidebarToggle.addEventListener('click', toggleSidebar);
    this.el.sidebarClose.addEventListener('click', toggleSidebar);
    // Backdrop click closes sidebar on mobile (manual action)
    if (backdrop) backdrop.addEventListener('click', () => {
      if (this.sidebarVisible) toggleSidebar();
    });
    // Resize handler: auto-collapse on narrow, auto-restore on wide (instant, no debounce)
    window.addEventListener('resize', () => {
      if (_isMobile() && this.sidebarVisible) {
        this.sidebarVisible = false;
        _sidebarAutoCollapsed = true;
        _updateSidebarUI(false);
      } else if (!_isMobile() && !this.sidebarVisible && _sidebarAutoCollapsed) {
        this.sidebarVisible = true;
        _sidebarAutoCollapsed = false;
        _updateSidebarUI(true);
      } else if (!_isMobile()) {
        // Desktop: ensure sidebar is never fully hidden, use rail if collapsed
        this.el.sidebar.classList.remove('cv2-hidden');
        if (!this.sidebarVisible) this.el.sidebar.classList.add('cv2-sidebar-rail');
        if (backdrop) backdrop.classList.remove('cv2-visible');
      }
      if (_isMobile() && backdrop && !this.sidebarVisible) backdrop.classList.remove('cv2-visible');
    });
    // Expose for use by other modules (e.g. _selectConversation auto-close)
    this._isMobile = _isMobile;
    this._closeSidebarOnMobile = () => {
      if (_isMobile() && this.sidebarVisible) {
        this.sidebarVisible = false;
        // Selecting a conversation on mobile = intentional action, not auto-collapse
        _sidebarAutoCollapsed = true;
        _updateSidebarUI(false);
      }
    };
    // Expose auto-collapse setter for sidebar responsive collapse
    this._markSidebarAutoCollapsed = () => { _sidebarAutoCollapsed = true; };
    // Expose for use by other modules (e.g. handleUIAction toggle_sidebar)
    this._updateSidebarUI = _updateSidebarUI;

    // Search chats button
    document.getElementById('cv2-search-chats')?.addEventListener('click', () => this._openSearchDialog());

    // New chat (respects active project context)
    this.el.newChat.addEventListener('click', () => {
      if (this.streaming) {
        this.ws.send({ type: 'stop_streaming' });
        this.streaming = false;
        this._updateSendButton();
      }
      this._closeSidebarOnMobile();
      this._pendingFirstMsgHtml = null;  // prevent stale message from previous chat
      // Send new_chat BEFORE exiting incognito — the server may send
      // save_conversation for the old chat during new_chat processing.
      // If we exit incognito first, idb.put() would persist the incognito
      // conversation (race condition).
      this.ws.send({ type: 'new_chat' });
      if (this.incognito) this._exitIncognito();
    });

    // Incognito toggle in topbar
    document.getElementById('cv2-incognito-toggle')?.addEventListener('click', () => {
      if (this.incognito) {
        // Turn off — send new_chat first, THEN exit incognito
        // (prevents save_conversation race persisting incognito chat)
        this.ws.send({ type: 'new_chat' });
        this._exitIncognito();
      } else {
        // Turn on — start a fresh incognito chat
        if (this.streaming) {
          this.ws.send({ type: 'stop_streaming' });
          this.streaming = false;
          this._updateSendButton();
        }
        this._pendingFirstMsgHtml = null;
        this.incognito = true;
        const root = document.getElementById('chat-app');
        root?.classList.add('cv2-dark', 'cv2-incognito');
        const incBtn = document.getElementById('cv2-incognito-toggle');
        // Keep button visible so user can toggle off before first message
        if (incBtn) { incBtn.classList.add('cv2-active'); }
        // Swap mascot to incognito variant
        if (this.config.appMascotIncognito) {
          const mascotEl = root?.querySelector('.cv2-mascot');
          if (mascotEl) {
            mascotEl.src = this.config.appMascotIncognito;
          }
        }
        this.ws.send({ type: 'new_chat' });
      }
    });

    // Model dropdown
    this.el.modelBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const willOpen = !this.modelDropdownOpen;
      this._closeAllPopups('model');
      this.modelDropdownOpen = willOpen;
      this.el.modelDropdown.style.display = this.modelDropdownOpen ? 'block' : 'none';
    });

    // Plus menu
    this.el.plusBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const willOpen = !this.plusMenuOpen;
      this._closeAllPopups('plus');
      this.plusMenuOpen = willOpen;
      if (willOpen) this._renderPlusMenu();
      this.el.plusMenu.classList.toggle('cv2-visible', this.plusMenuOpen);
    });

    // Suggestions menu
    this.el.suggestionsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const willOpen = !this.suggestionsMenuOpen;
      this._closeAllPopups('suggestions');
      this.suggestionsMenuOpen = willOpen;
      this.el.suggestionsMenu.classList.toggle('cv2-visible', this.suggestionsMenuOpen);
    });

    // Context popover (Shift+click → Prompt Inspector for admins)
    this.el.contextCircle.addEventListener('click', (e) => {
      e.stopPropagation();
      if (e.shiftKey && this.hasPerm('dev_tools')) {
        this._closeAllPopups();
        this.ws.send({ type: 'get_prompt_inspector' });
        return;
      }
      const willOpen = !this.contextPopoverOpen;
      this._closeAllPopups('context');
      this.contextPopoverOpen = willOpen;
      this.el.contextPopover.classList.toggle('cv2-visible', this.contextPopoverOpen);
      if (this.contextPopoverOpen) {
        this._renderContextPopover();
        this.ws.send({ type: 'get_context_info' });
      }
    });

    // Gear menu (settings: color mode, export, clear all)
    this.el.gearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (e.shiftKey && this.hasPerm('dev_tools')) {
        // Dev settings popover
        this._closeAllPopups('devSettings');
        this.devSettingsOpen = !this.devSettingsOpen;
        this.el.devSettingsPopover.classList.toggle('cv2-visible', this.devSettingsOpen);
        return;
      }
      const willOpen = !this.gearMenuOpen;
      this._closeAllPopups('gear');
      this.gearMenuOpen = willOpen;
      this.el.gearPopover.classList.toggle('cv2-visible', this.gearMenuOpen);
      if (!willOpen) this.el.voiceSettingsMenu.style.display = 'none';
    });

    // Dev settings: API Keys
    this.el.devApiKeysBtn.addEventListener('click', () => {
      this.devSettingsOpen = false;
      this.el.devSettingsPopover.classList.remove('cv2-visible');
      this._showApiKeysDialog();
    });

    // Theme selector (inside gear menu)
    this._renderThemeSelector();

    // Data management submenu — same pattern as voice settings
    this.el.dataMgmtBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const menu = this.el.dataMgmtPopover;
      const opening = menu.style.display === 'none';
      // Close other submenus
      this.el.voiceSettingsMenu.style.display = 'none';
      const themeSub = document.getElementById('cv2-theme-submenu');
      if (themeSub) themeSub.style.display = 'none';
      menu.style.display = opening ? '' : 'none';
      if (opening) {
        const rect = this.el.dataMgmtBtn.getBoundingClientRect();
        menu.style.left = rect.right + 4 + 'px';
        menu.style.bottom = (window.innerHeight - rect.bottom) + 'px';
      }
    });
    this.el.dataMgmtPopover.addEventListener('click', (e) => e.stopPropagation());
    this.el.exportAll.addEventListener('click', () => {
      this._closeGearMenu();
      this._exportAll();
    });
    this.el.importAll.addEventListener('click', () => {
      this._closeGearMenu();
      this._importAll();
    });

    // Speech toggle (inside gear menu)
    this.el.speechToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      this._closeGearMenu();
      this._toggleSpeechResponse();
    });

    // Clear all (inside gear menu)
    this.el.clearAll.addEventListener('click', () => {
      this._closeGearMenu();
      this._showClearAllDialog();
    });

    // Voice popup: send now
    this.el.voiceSend.addEventListener('click', () => {
      this._sendNowVoiceRecording();
    });

    // Voice popup: cancel
    this.el.voiceCancel.addEventListener('click', () => {
      this._closeVoicePopup();
    });

    // Voice selector
    this.el.voiceSelect.addEventListener('change', (e) => {
      this._ttsVoice = e.target.value;
      localStorage.setItem('cv2-tts-voice', this._ttsVoice);
      this.ws.send({ type: 'update_settings', tts_voice: this._ttsVoice });
    });

    // Voice settings sliders
    this.el.voiceWaitSlider.addEventListener('input', (e) => {
      this._voiceSilenceMs = parseInt(e.target.value, 10);
      this.el.voiceWaitVal.textContent = (this._voiceSilenceMs / 1000) + 's';
      localStorage.setItem('cv2-voice-silence-ms', this._voiceSilenceMs);
    });
    this.el.voiceThreshSlider.addEventListener('input', (e) => {
      // Slider is inverted: right = high sensitivity = low threshold
      this._voiceSilenceThreshold = 0.065 - parseFloat(e.target.value);
      this.el.voiceThreshVal.textContent = this._sensitivityLabel(this._voiceSilenceThreshold);
      localStorage.setItem('cv2-voice-silence-threshold', this._voiceSilenceThreshold);
    });

    // Speech mode button — click shows picker, long press (≥2s) goes directly to live mode
    {
      let longPressTimer = null;
      let longPressFired = false;
      let ringEl = null;

      const startLongPress = (e) => {
        if (!this._enableLiveVoice) return; // long press only for live voice
        longPressFired = false;
        // Create growing ring animation
        if (!ringEl) {
          ringEl = document.createElement('span');
          ringEl.className = 'cv2-long-press-ring';
          this.el.speechModeBtn.appendChild(ringEl);
        }
        ringEl.classList.remove('cv2-long-press-active');
        void ringEl.offsetWidth; // reflow
        ringEl.classList.add('cv2-long-press-active');

        longPressTimer = setTimeout(() => {
          longPressFired = true;
          cancelRing();
          this.el.voicePicker.style.display = 'none';
          this._openRealtimeDialog();
        }, 2000);
      };
      const cancelRing = () => {
        if (ringEl) ringEl.classList.remove('cv2-long-press-active');
      };
      const cancelLongPress = () => {
        if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
        cancelRing();
      };
      this.el.speechModeBtn.addEventListener('mousedown', startLongPress);
      this.el.speechModeBtn.addEventListener('touchstart', startLongPress, { passive: true });
      this.el.speechModeBtn.addEventListener('mouseup', cancelLongPress);
      this.el.speechModeBtn.addEventListener('mouseleave', cancelLongPress);
      this.el.speechModeBtn.addEventListener('touchend', cancelLongPress);
      this.el.speechModeBtn.addEventListener('touchcancel', cancelLongPress);
      this.el.speechModeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (longPressFired) return;

        // If only one mode enabled, go directly to it (no picker)
        if (this._enableVoiceInput && !this._enableLiveVoice) {
          this._enterSpeechMode();
          return;
        }
        if (this._enableLiveVoice && !this._enableVoiceInput) {
          this._openRealtimeDialog();
          return;
        }

        // Both enabled — show picker
        this._closeAllPopups('voicePicker');
        const picker = this.el.voicePicker;
        if (picker.style.display !== 'none') { picker.style.display = 'none'; return; }
        // Position above the button
        const rect = this.el.speechModeBtn.getBoundingClientRect();
        picker.style.position = 'fixed';
        picker.style.bottom = (window.innerHeight - rect.top + 8) + 'px';
        picker.style.right = (window.innerWidth - rect.right) + 'px';
        picker.style.display = 'flex';
      });
    }

    // Voice picker card clicks
    this.el.voicePickLive.addEventListener('click', () => {
      this.el.voicePicker.style.display = 'none';
      this._openRealtimeDialog();
    });
    this.el.voicePickInput.addEventListener('click', () => {
      this.el.voicePicker.style.display = 'none';
      this._enterSpeechMode();
    });

    // Inline voice bar buttons
    this.el.ivbSend.addEventListener('click', () => this._sendNowVoiceRecording());
    this.el.ivbInterrupt.addEventListener('click', () => {
      if (this._rtActive) return;
      this._interruptSpeechMode();
    });
    this.el.ivbExit.addEventListener('click', () => {
      if (this._rtActive) {
        this._closeRealtimeDialog();
      } else {
        this._exitSpeechMode();
      }
    });

    // Mobile big mic button — true PTT (press to record, release to send)
    if (this.el.mobileMicBtn) {
      const micDown = (e) => {
        e.preventDefault();
        if (!this._speechMode) return;
        this.el.mobileMicBtn.classList.add('cv2-recording');
        this._pttPress();
      };
      const micUp = () => {
        if (!this._speechMode) return;
        this.el.mobileMicBtn.classList.remove('cv2-recording');
        this._pttRelease();
      };
      this.el.mobileMicBtn.addEventListener('touchstart', micDown, { passive: false });
      this.el.mobileMicBtn.addEventListener('touchend', micUp);
      this.el.mobileMicBtn.addEventListener('touchcancel', micUp);
      this.el.mobileMicBtn.addEventListener('mousedown', micDown);
      this.el.mobileMicBtn.addEventListener('mouseup', micUp);
      this.el.mobileMicBtn.addEventListener('mouseleave', micUp);
    }

    // Push-to-talk mic button (press = start, release = send)
    const pttBtn = this.el.ivbPtt;
    pttBtn.addEventListener('mousedown', (e) => { e.preventDefault(); this._pttPress(); });
    pttBtn.addEventListener('mouseup', () => this._pttRelease());
    pttBtn.addEventListener('mouseleave', () => this._pttRelease());
    pttBtn.addEventListener('touchstart', (e) => { e.preventDefault(); this._pttPress(); }, { passive: false });
    pttBtn.addEventListener('touchend', () => this._pttRelease());
    pttBtn.addEventListener('touchcancel', () => this._pttRelease());

    // PTT toggle in voice settings
    this.el.pttToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      this._pushToTalk = !this._pushToTalk;
      this.el.pttToggle.classList.toggle('cv2-on', this._pushToTalk);
      localStorage.setItem('cv2-push-to-talk', this._pushToTalk);
    });
    // Voice settings submenu — positioned as flyout next to the button
    this.el.voiceSettingsBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const menu = this.el.voiceSettingsMenu;
      const opening = menu.style.display === 'none';
      // Close theme submenu if open
      const themeSub = document.getElementById('cv2-theme-submenu');
      if (themeSub) themeSub.style.display = 'none';
      menu.style.display = opening ? '' : 'none';
      if (opening) {
        const rect = this.el.voiceSettingsBtn.getBoundingClientRect();
        menu.style.left = rect.right + 4 + 'px';
        menu.style.bottom = (window.innerHeight - rect.bottom) + 'px';
      }
    });
    this.el.voiceSettingsMenu.addEventListener('click', (e) => e.stopPropagation());
    this.el.voiceModeExit.addEventListener('click', () => {
      this._exitSpeechMode();
    });
    this.el.speechSendBtn.addEventListener('click', () => {
      this._sendNowVoiceRecording();
    });
    this.el.speechInterruptBtn.addEventListener('click', () => {
      this._interruptSpeechMode();
    });

    // Hidden dev menu: 3 left clicks then 3 right clicks on avatar
    {
      let leftClicks = 0, rightClicks = 0, timer = null;
      const reset = () => { leftClicks = 0; rightClicks = 0; clearTimeout(timer); timer = null; };
      const arm = () => { clearTimeout(timer); timer = setTimeout(reset, 2000); };
      this.el.avatar.addEventListener('click', (e) => {
        e.stopPropagation();
        if (rightClicks > 0) { reset(); return; }
        leftClicks++; arm();
      });
      this.el.avatar.addEventListener('contextmenu', (e) => {
        e.preventDefault(); e.stopPropagation();
        if (leftClicks < 3) { reset(); return; }
        rightClicks++; arm();
        if (rightClicks >= 3) {
          reset();
          this.el.devMenu.classList.toggle('cv2-visible');
        }
      });
      this.el.devResetTools.addEventListener('click', () => {
        this.el.devMenu.classList.remove('cv2-visible');
        localStorage.removeItem('cv2-tool-prefs');
        location.reload();
      });
      this.el.devClearStorage.addEventListener('click', () => {
        this.el.devMenu.classList.remove('cv2-visible');
        localStorage.clear();
        location.reload();
      });
      document.addEventListener('click', (e) => {
        if (this.el.devMenu.classList.contains('cv2-visible') && !this.el.devMenu.contains(e.target) && !this.el.avatar.contains(e.target)) {
          this.el.devMenu.classList.remove('cv2-visible');
        }
      });
    }

    // Clicks inside the plus menu should not close it (flyouts, toggles, etc.)
    this.el.plusMenu.addEventListener('click', (e) => e.stopPropagation());

    // Close dropdowns on outside click
    document.addEventListener('click', () => this._closeAllPopups());

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

    // ── Global keyboard shortcuts ─────────────────────────
    document.addEventListener('keydown', (e) => {
      // Skip if focus is in an input/textarea (except for Cmd/Ctrl combos)
      const tag = (e.target.tagName || '').toLowerCase();
      const inInput = tag === 'input' || tag === 'textarea' || e.target.isContentEditable;
      const mod = e.metaKey || e.ctrlKey;

      // Cmd/Ctrl+K — Search chats
      if (mod && e.key === 'k') {
        e.preventDefault();
        this._openSearchDialog();
        return;
      }
      // Alt+N — New chat
      if (e.altKey && e.key === 'n') {
        e.preventDefault();
        this.el.newChat.click();
        return;
      }
      // Alt+B — Toggle sidebar
      if (e.altKey && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
        return;
      }
      // Escape — close popups/search
      if (e.key === 'Escape') {
        this._closeAllPopups();
      }
      // Space — Push-to-talk (speech mode or realtime mode with PTT enabled)
      if (e.code === 'Space' && (this._speechMode || this._rtActive) && this._pushToTalk && !this._pttKeyDown) {
        if (inInput) return; // don't capture when typing
        e.preventDefault();
        this._pttKeyDown = true;
        this._pttPress();
      }
    });
    document.addEventListener('keyup', (e) => {
      if (e.code === 'Space' && this._pttKeyDown) {
        e.preventDefault();
        this._pttKeyDown = false;
        this._pttRelease();
      }
    });
    // Release PTT if window loses focus while holding
    window.addEventListener('blur', () => {
      if (this._pttKeyDown) {
        this._pttKeyDown = false;
        this._pttRelease();
      }
    });

    // Dev mode: Cmd+Shift+Click (Mac) or Ctrl+Shift+Click (Win/Linux) on doc plugin blocks shows raw data
    const _devModKey = (e) => (e.metaKey || e.ctrlKey) && e.shiftKey;
    // Disable iframe pointer-events while modifier keys are held so clicks reach the container
    const _devIframeToggle = (on) => {
      if (!this._devMode) return;
      this.el.messages.querySelectorAll('.cv2-doc-plugin-block iframe').forEach(f => {
        f.style.pointerEvents = on ? 'none' : '';
      });
    };
    document.addEventListener('keydown', (e) => { if (_devModKey(e)) _devIframeToggle(true); });
    document.addEventListener('keyup', () => _devIframeToggle(false));
    this.el.messages.addEventListener('click', (e) => {
      if (!this._devMode || !_devModKey(e)) return;
      const block = e.target.closest('.cv2-doc-plugin-block');
      if (!block) return;
      e.preventDefault();
      e.stopPropagation();
      const raw = block._devRawData;
      const lang = block._devLang || block.dataset.lang || 'unknown';
      if (!raw) { this._showToast('No raw data available for this block', 'warning'); return; }
      this._showDevRawOverlay(lang, raw);
    }, true);
  },

  // ── Helpers ───────────────────────────────────────────

  async _clearAllConversations() {
    try {
      for (const c of this.conversations) {
        await this.idb.delete(c.id);
        await this.idb.removeAllRefsForConversation(c.id).catch(() => {});
      }
      this.conversations = [];
      this.activeConvId = '';
      this._savedFileRefs = {};
      this._blockDataStore?.clear();
      this.inlineDocBlocks = [];
      this._renderSidebar();
    } catch (err) {
      console.error('[Clear] Failed:', err);
    }
  },

  _sensitivityLabel(v) {
    if (v <= 0.015) return this.t('chat.sensitivity_high');
    if (v <= 0.03) return this.t('chat.sensitivity_medium');
    return this.t('chat.sensitivity_low');
  },

  _scrollToBottom() {
    const scrollContainer = this.el.messages?.closest('.cv2-main');
    if (scrollContainer) {
      requestAnimationFrame(() => {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      });
    }
  },

  _autoResizeTextarea() {
    const ta = this.el.textarea;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  },

  _updateSendButton() {
    const hasContent = this.el.textarea.value.trim().length > 0
      || this._pendingImages.length > 0
      || this._pendingFiles.length > 0;

    const voiceEnabled = this._enableVoiceInput || this._enableLiveVoice;

    if (this.streaming) {
      this.el.sendBtn.classList.add('cv2-streaming');
      this.el.sendIcon.textContent = 'stop';
      this.el.sendBtn.style.display = '';
      this.el.speechModeBtn.style.display = 'none';
    } else if (hasContent) {
      this.el.sendBtn.classList.remove('cv2-streaming');
      this.el.sendIcon.textContent = 'send';
      this.el.sendBtn.style.display = '';
      this.el.speechModeBtn.style.display = 'none';
    } else {
      this.el.sendBtn.style.display = voiceEnabled ? 'none' : '';
      this.el.speechModeBtn.style.display = voiceEnabled ? '' : 'none';
    }
  },

  /** Render theme selector with flyout submenu (like voice settings). */
  _renderThemeSelector() {
    const slot = this.el.gearThemeSlot;
    if (!slot) return;
    const themes = [
      { id: 'auto',  icon: 'brightness_auto', label: 'Auto' },
      { id: 'light', icon: 'light_mode',      label: 'Light' },
      { id: 'dark',  icon: 'dark_mode',       label: 'Dark' },
      { id: 'pink',  icon: 'favorite',        label: 'Pink' },
      { id: 'joche', icon: 'favorite',        label: 'Joch\u00e9' },
    ];
    const current = this._themeMode || 'auto';
    const cur = themes.find(t => t.id === current) || themes[0];

    // Button inside gear popover
    slot.innerHTML = `
      <button class="cv2-gear-popover-item" id="cv2-theme-gear-btn">
        <span class="material-icons">${cur.icon}</span>
        <span>${this.t('chat.color_mode')}</span>
        <span class="material-icons cv2-gear-arrow">chevron_right</span>
      </button>
    `;

    const btn = slot.querySelector('#cv2-theme-gear-btn');

    // Flyout submenu (fixed, positioned to the right — same as voice settings)
    let sub = document.getElementById('cv2-theme-submenu');
    if (!sub) {
      sub = document.createElement('div');
      sub.id = 'cv2-theme-submenu';
      sub.className = 'cv2-gear-submenu';
      sub.style.display = 'none';
      document.getElementById('chat-app').appendChild(sub);
    }
    sub.innerHTML = themes.map(t => `
      <button class="cv2-lang-popover-item ${t.id === current ? 'cv2-active' : ''}" data-theme="${t.id}">
        <span class="material-icons" style="font-size:16px">${t.icon}</span> ${t.label}
      </button>
    `).join('');

    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const opening = sub.style.display === 'none';
      // Close voice settings if open
      this.el.voiceSettingsMenu.style.display = 'none';
      sub.style.display = opening ? '' : 'none';
      if (opening) {
        const rect = btn.getBoundingClientRect();
        sub.style.left = rect.right + 4 + 'px';
        sub.style.bottom = (window.innerHeight - rect.bottom) + 'px';
      }
    });
    sub.addEventListener('click', (e) => e.stopPropagation());
    sub.querySelectorAll('[data-theme]').forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation();
        sub.style.display = 'none';
        this._closeGearMenu();
        const mode = item.dataset.theme;
        this._themeMode = mode;
        // Only persist to localStorage if no enforced default (enforced resets on reload)
        if (!this._enforcedTheme) {
          try { localStorage.setItem('chat-color-mode', mode); } catch {}
        }
        this._applyThemeMode(mode);
        this._renderThemeSelector(); // refresh active state + icon
      });
    });
  },

  _applyThemeMode(mode) {
    const root = document.getElementById('chat-app');

    root?.classList.remove('cv2-dark', 'cv2-pink', 'cv2-joche');
    if (mode === 'auto') {
      const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root?.classList.toggle('cv2-dark', dark);
    } else if (mode === 'dark') {
      root?.classList.add('cv2-dark');
    } else if (mode === 'pink') {
      root?.classList.add('cv2-pink');
    } else if (mode === 'joche') {
      root?.classList.add('cv2-pink', 'cv2-joche');
    }
    // Refresh greeting when switching themes (pink/joche have special greetings)
    if (this.el?.greeting && this._lastGreetName) {
      this.el.greeting.textContent = this._getGreeting(this._lastGreetName);
    }
    // Broadcast theme CSS custom properties to all doc plugin iframes
    this._broadcastThemeToIframes();
  },

  _broadcastThemeToIframes() {
    try {
      const rootStyle = getComputedStyle(document.documentElement);
      const varNames = ['--chat-accent','--chat-bg','--chat-surface','--chat-text',
        '--chat-border','--chat-text-muted','--chat-code-bg','--chat-code-text','--chat-link'];
      const vars = {};
      for (const v of varNames) {
        const val = rootStyle.getPropertyValue(v).trim();
        if (val) vars[v] = val;
      }
      const msg = { type: 'theme_update', vars };
      document.querySelectorAll('.cv2-doc-plugin-block iframe').forEach(iframe => {
        try { iframe.contentWindow?.postMessage(msg, '*'); } catch (_) {}
      });
    } catch (_) {}
  },

  _exitIncognito() {
    this.incognito = false;
    const root = document.getElementById('chat-app');
    root?.classList.remove('cv2-incognito');
    const incBtn = document.getElementById('cv2-incognito-toggle');
    if (incBtn) { incBtn.classList.remove('cv2-active'); incBtn.style.display = ''; }
    // Restore original mascot
    const mascotEl = root?.querySelector('.cv2-mascot');
    if (mascotEl) {
      mascotEl.src = mascotEl.dataset.defaultSrc || this.config.appMascot || '';
    }
    this._applyThemeMode(this._themeMode);
  },

  _applyStoredTheme() {
    try {
      // Enforced theme overrides localStorage default but user can still switch manually
      this._themeMode = this._enforcedTheme || localStorage.getItem('chat-color-mode') || 'dark';
      this._applyThemeMode(this._themeMode);

      // Follow live system changes when in auto mode
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (this._themeMode === 'auto') this._applyThemeMode('auto');
      });
    } catch {}
  },

  /** Find a fast/small model for quick tasks like document summaries. */
  _findFastModel() {
    const fast = this.models.find(m =>
      /flash|nano|mini|haiku|small|gpt-4o-mini/i.test(m.model) ||
      /flash|nano|mini|haiku|small/i.test(m.label)
    );
    return fast ? fast.model : null;
  },

  _truncTitle(t) {
    if (!t) return this.t('chat.untitled');
    return t.length > 30 ? t.substring(0, 28) + '...' : t;
  },

  _parseSuggestions(text) {
    if (!text) return [];
    if (Array.isArray(text)) return text.map(s => String(s).trim()).filter(Boolean);
    if (typeof text !== 'string') text = String(text);
    const parts = text.includes('---') ? text.split('---') : text.split('\n');
    return parts.map(s => s.trim()).filter(Boolean);
  },

  _pickRandom(arr, n) {
    if (arr.length <= n) return [...arr];
    const copy = [...arr];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy.slice(0, n);
  },

  _escHtml(s) {
    if (!s) return '';
    if (typeof s !== 'string') s = String(s);
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  },

  _isIconUrl(icon) {
    return icon && (icon.startsWith('data:') || icon.includes('/') || icon.includes('.'));
  },

  _renderIcon(icon, imgClass, fallbackIcon) {
    if (!icon) return `<span class="material-icons ${imgClass || ''}">${this._escHtml(fallbackIcon || 'science')}</span>`;
    if (this._isIconUrl(icon)) return `<img src="${this._escAttr(icon)}" class="${imgClass || ''}" alt="">`;
    return `<span class="material-icons ${imgClass || ''}">${this._escHtml(icon)}</span>`;
  },

  _escAttr(s) {
    return this._escHtml(s);
  },

  // ── Dev mode: raw data overlay ────────────────────────────
  _showDevRawOverlay(lang, rawData) {
    let formatted = rawData;
    try { formatted = JSON.stringify(JSON.parse(rawData), null, 2); } catch (_) {}
    const overlay = document.createElement('div');
    overlay.className = 'cv2-dialog-overlay';
    overlay.style.zIndex = '9999';
    overlay.innerHTML = `
      <div class="cv2-dialog" style="max-width:700px;width:90vw;max-height:80vh;display:flex;flex-direction:column">
        <div class="cv2-dialog-header" style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--chat-border,#e5e7eb)">
          <span style="font-weight:600;font-size:14px">Raw Data — <code>${this._escHtml(lang)}</code></span>
          <div style="display:flex;gap:8px">
            <button class="cv2-dev-copy-btn" style="background:none;border:1px solid var(--chat-border,#e5e7eb);border-radius:4px;padding:4px 10px;cursor:pointer;font-size:12px;color:var(--chat-text,#1f2937)">Copy</button>
            <button class="cv2-dev-close-btn" style="background:none;border:none;cursor:pointer;font-size:18px;color:var(--chat-text-muted,#6b7280)">&times;</button>
          </div>
        </div>
        <pre style="flex:1;overflow:auto;padding:12px 16px;margin:0;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-all;background:var(--chat-code-bg,#f3f4f6);color:var(--chat-code-text,#1f2937)">${this._escHtml(formatted)}</pre>
      </div>`;
    const close = () => overlay.remove();
    overlay.querySelector('.cv2-dev-close-btn').addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
    overlay.querySelector('.cv2-dev-copy-btn').addEventListener('click', () => {
      navigator.clipboard.writeText(formatted).then(() => {
        overlay.querySelector('.cv2-dev-copy-btn').textContent = 'Copied!';
        setTimeout(() => { overlay.querySelector('.cv2-dev-copy-btn').textContent = 'Copy'; }, 1500);
      });
    });
    document.body.appendChild(overlay);
  },

  // ── Ctrl+Shift+Enter: copy curl command ─────────────────
  _copyCurlCommand() {
    const text = this.el.textarea.value.trim();
    if (!text) return;
    // Find the latest API key from the chat messages (look for llming_ in code blocks)
    const msgs = this.el.messages;
    let apiKey = '';
    if (msgs) {
      const codeBlocks = msgs.querySelectorAll('code');
      for (let i = codeBlocks.length - 1; i >= 0; i--) {
        const content = codeBlocks[i].textContent.trim();
        if (content.startsWith('llming_') && content.length > 20 && !content.includes(' ')) {
          apiKey = content;
          break;
        }
      }
    }
    if (!apiKey) {
      // Also check sessionStorage as fallback
      apiKey = sessionStorage.getItem('cv2_api_key') || '';
    }
    if (!apiKey) {
      console.warn('[Curl] No API key found. Activate remote mode first: /dev remote');
      return;
    }
    const escaped = JSON.stringify(text);
    const base = window.location.origin;
    const host = new URL(base).hostname;
    const insecure = (host === 'localhost' || host === '0.0.0.0' || host === '127.0.0.1') ? ' -k' : '';
    const curl = `curl -N${insecure} -H "Authorization: Bearer ${apiKey}" -H "Content-Type: application/json" -d '{"text": ${escaped}}' ${base}/api/llming/v1/chat/send`;
    navigator.clipboard.writeText(curl).then(() => {
      // Brief visual feedback on the send button
      const icon = this.el.sendIcon;
      if (icon) {
        const prev = icon.textContent;
        icon.textContent = 'content_copy';
        setTimeout(() => { icon.textContent = prev; }, 1500);
      }
      console.log('[Curl] Copied to clipboard:\n' + curl);
    });
  },

  // ── API Keys Dialog ─────────────────────────────────────
  _showApiKeysDialog() {
    const app = this;
    // Request key list from server
    this.ws.send({ type: 'apikeys:list' });

    const overlay = document.createElement('div');
    overlay.className = 'cv2-dialog-overlay';
    overlay.innerHTML = `
      <div class="cv2-dialog cv2-apikeys-dialog">
        <div class="cv2-dialog-title">API Keys</div>
        <div class="cv2-dialog-body">
          <div class="cv2-apikeys-list" id="cv2-apikeys-list">
            <div class="cv2-apikeys-loading">Loading…</div>
          </div>
          <div class="cv2-apikeys-created-banner" id="cv2-apikeys-created-banner" style="display:none">
            <div class="cv2-apikeys-created-label">New key created — copy it now (shown only once):</div>
            <div class="cv2-apikeys-created-key">
              <code id="cv2-apikeys-full-key"></code>
              <button class="cv2-apikeys-copy-btn" id="cv2-apikeys-copy-btn" title="Copy">
                <span class="material-icons" style="font-size:16px">content_copy</span>
              </button>
            </div>
          </div>
          <div class="cv2-apikeys-create-section">
            <div class="cv2-apikeys-create-row">
              <input type="text" id="cv2-apikeys-name-input" placeholder="Key name" class="cv2-apikeys-input" maxlength="64">
              <button class="cv2-dialog-btn cv2-dialog-confirm" id="cv2-apikeys-create-btn">Create</button>
            </div>
            <div class="cv2-apikeys-perms">
              <label><input type="checkbox" id="cv2-apikeys-perm-droplets" checked> Manage droplets</label>
              <label><input type="checkbox" id="cv2-apikeys-perm-chat" checked> Automate chat</label>
            </div>
          </div>
        </div>
        <div class="cv2-dialog-actions">
          <button class="cv2-dialog-btn cv2-dialog-cancel" id="cv2-apikeys-close">Close</button>
        </div>
      </div>`;

    document.getElementById('chat-app').appendChild(overlay);

    const listEl = overlay.querySelector('#cv2-apikeys-list');
    const banner = overlay.querySelector('#cv2-apikeys-created-banner');
    const fullKeyEl = overlay.querySelector('#cv2-apikeys-full-key');
    const closeBtn = overlay.querySelector('#cv2-apikeys-close');
    const createBtn = overlay.querySelector('#cv2-apikeys-create-btn');
    const nameInput = overlay.querySelector('#cv2-apikeys-name-input');
    const copyBtn = overlay.querySelector('#cv2-apikeys-copy-btn');
    const permDroplets = overlay.querySelector('#cv2-apikeys-perm-droplets');
    const permChat = overlay.querySelector('#cv2-apikeys-perm-chat');

    const close = () => {
      app._apiKeysDialogHandler = null;
      overlay.remove();
    };
    closeBtn.addEventListener('click', close);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

    copyBtn.addEventListener('click', () => {
      const key = fullKeyEl.textContent;
      navigator.clipboard.writeText(key).then(() => {
        copyBtn.innerHTML = '<span class="material-icons" style="font-size:16px">check</span>';
        setTimeout(() => {
          copyBtn.innerHTML = '<span class="material-icons" style="font-size:16px">content_copy</span>';
        }, 2000);
      });
    });

    createBtn.addEventListener('click', () => {
      const name = nameInput.value.trim() || 'Unnamed Key';
      const permissions = [];
      if (permDroplets.checked) permissions.push('manage_droplets');
      if (permChat.checked) permissions.push('automate_chat');
      if (!permissions.length) return;
      app.ws.send({ type: 'apikeys:create', name, permissions });
      nameInput.value = '';
      banner.style.display = 'none';
    });

    function renderKeys(keys) {
      if (!keys.length) {
        listEl.innerHTML = '<div class="cv2-apikeys-empty">No API keys yet.</div>';
        return;
      }
      listEl.innerHTML = keys.map(k => `
        <div class="cv2-apikeys-item" data-key-id="${app._escAttr(k.key_id)}">
          <div class="cv2-apikeys-item-info">
            <span class="cv2-apikeys-item-name">${app._escHtml(k.name)}</span>
            <code class="cv2-apikeys-item-prefix">${app._escHtml(k.key_prefix)}…</code>
            <span class="cv2-apikeys-item-perms">${(k.permissions || []).map(p =>
              '<span class="cv2-apikeys-badge">' + app._escHtml(p.replace('_', ' ')) + '</span>'
            ).join('')}</span>
          </div>
          <button class="cv2-apikeys-delete-btn" data-key-id="${app._escAttr(k.key_id)}" title="Delete">
            <span class="material-icons" style="font-size:16px">delete</span>
          </button>
        </div>`).join('');
      listEl.querySelectorAll('.cv2-apikeys-delete-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const keyId = btn.dataset.keyId;
          app.ws.send({ type: 'apikeys:delete', key_id: keyId });
        });
      });
    }

    // WS message handler for apikeys responses
    app._apiKeysDialogHandler = (msg) => {
      if (msg.type === 'apikeys:list') {
        renderKeys(msg.keys || []);
      } else if (msg.type === 'apikeys:created') {
        fullKeyEl.textContent = msg.full_key;
        banner.style.display = '';
        sessionStorage.setItem('cv2_api_key', msg.full_key);
        // Refresh list
        app.ws.send({ type: 'apikeys:list' });
      } else if (msg.type === 'apikeys:deleted') {
        const item = listEl.querySelector(`[data-key-id="${msg.key_id}"]`);
        if (item) item.remove();
        if (!listEl.children.length) renderKeys([]);
      }
    };
  },

});


/* ══════════════════════════════════════════════════════════
   Boot — find the #chat-app mount point and start the app
   ══════════════════════════════════════════════════════════ */

function _bootChat() {
  const config = window.__CHAT_CONFIG__;
  if (!config) {
    console.error('[Chat] No __CHAT_CONFIG__ found');
    return;
  }

  // Poll until the mount point exists in the DOM (may be async).
  function tryInit() {
    const root = document.getElementById('chat-app');
    if (root) {
      // Create instance via Object.create to avoid class declaration
      // scoping issues (class ChatApp in chat-features.js creates a
      // global lexical binding, not a window property).
      const app = Object.create(ChatApp.prototype);
      app._init_constructor(config);
      window.__chatApp = app;
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
