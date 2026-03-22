/* ── chat-bolts.js — Bolt engine (regex detection, chips, autocomplete, actions) ── */
/* Feature module: adds bolt support to ChatApp via the prototype mixin pattern.    */

(function () {
  'use strict';

  // ── Device classification ──────────────────────────────
  function _boltDevice() {
    const w = window.innerWidth;
    if (w < 768) return 'mobile';
    if (w < 1024) return 'tablet';
    return 'desktop';
  }

  function _boltDeviceOk(bolt) {
    if (!bolt.devices || bolt.devices.length === 0) return true;
    return bolt.devices.includes(_boltDevice());
  }

  // ── Bolt registry (populated from nudge configs) ───────
  // Each entry: { command, aliases, label, icon, regex, counterCheck, action, devices, nudgeUid }
  let _bolts = [];
  let _compiledRegexes = []; // [{ bolt, re }]

  function _recompileRegexes() {
    _compiledRegexes = [];
    for (const b of _bolts) {
      if (!b.regex || !_boltDeviceOk(b)) continue;
      try {
        _compiledRegexes.push({ bolt: b, re: new RegExp(b.regex, 'i') });
      } catch (e) {
        console.warn('[Bolts] Invalid regex for', b.command, e);
      }
    }
  }

  // ── Platform detection ────────────────────────────────
  const _isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const _quickHint = _isMac ? '⌘↵' : 'Ctrl↵';

  // ── Suggestion chip state ──────────────────────────────
  let _chipContainer = null;
  let _activeChips = []; // [{ bolt, match }]
  let _inputDebounce = null;

  // ── Autocomplete state ─────────────────────────────────
  let _acContainer = null;
  let _acItems = [];    // filtered bolt list
  let _acIndex = -1;    // highlighted index
  let _acVisible = false;

  // Close autocomplete on click outside
  function _acOutsideClick(e) {
    if (_acContainer && !_acContainer.contains(e.target) &&
        !e.target.closest('.cv2-input-area')) {
      if (_acVisible) {
        if (_acContainer) _acContainer.classList.remove('visible');
        _acVisible = false;
        _acIndex = -1;
        _acItems = [];
        document.removeEventListener('click', _acOutsideClick, true);
      }
    }
  }

  // ── DOM helpers ────────────────────────────────────────
  function _icon(name) {
    return `<span class="material-icons">${name}</span>`;
  }

  // ── Local math evaluation ────────────────────────────
  function _tryLocalEval(input) {
    // Strip leading "=" if present
    let expr = input.replace(/^\s*=\s*/, '').trim();
    if (!expr) return null;
    // Only allow safe math characters
    if (!/^[\d\s+\-*/().^%,]+$/.test(expr)) return null;
    try {
      expr = expr.replace(/\^/g, '**').replace(/,/g, '.');
      const result = Function('return ' + expr)();
      if (typeof result === 'number' && isFinite(result)) return result;
    } catch { /* not evaluable */ }
    return null;
  }

  function _formatNumber(n) {
    // Pretty-print: remove trailing zeros, use locale for display
    if (Number.isInteger(n)) return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
    return n.toLocaleString('en-US', { maximumFractionDigits: 10 });
  }

  // ──────────────────────────────────────────────────────
  // Prototype methods (mixed into ChatApp)
  // ──────────────────────────────────────────────────────

  Object.assign(window._ChatAppProto, {

    // ── Init (called during ChatApp.init) ─────────────
    _boltsInit() {
      _bolts = [];
      _compiledRegexes = [];
      _activeChips = [];
      this._boltApps = window._boltApps || {};
    },

    // ── Register bolts from a nudge/droplet config ────
    _boltsRegister(nudgeUid, boltDefs) {
      // Remove old bolts for this nudge (re-registration)
      const prevCount = _bolts.filter(b => b.nudgeUid === nudgeUid).length;
      _bolts = _bolts.filter(b => b.nudgeUid !== nudgeUid);
      for (const def of boltDefs) {
        // Normalize snake_case → camelCase for keys that differ
        const bolt = { ...def, nudgeUid };
        if (def.counter_check !== undefined) { bolt.counterCheck = def.counter_check; delete bolt.counter_check; }
        if (def.description_i18n !== undefined) { bolt.descriptionI18n = def.description_i18n; delete bolt.description_i18n; }
        if (def.match_anywhere !== undefined) { bolt.matchAnywhere = def.match_anywhere; delete bolt.match_anywhere; }
        _bolts.push(bolt);
      }
      _recompileRegexes();
    },

    // ── Register bolts from chat config (system bolts) ─
    _boltsRegisterSystem(boltDefs) {
      for (const def of boltDefs) {
        if (!_bolts.find(b => b.command === def.command && b.nudgeUid === '__system__')) {
          _bolts.push({ ...def, nudgeUid: '__system__' });
        }
      }
      _recompileRegexes();
    },

    // ── Clear all bolts ───────────────────────────────
    _boltsClear() {
      _bolts = [];
      _compiledRegexes = [];
      _activeChips = [];
      this._boltsHideChips();
      this._boltsHideAutocomplete();
    },

    // ── Input change handler (debounced regex scan) ───
    _boltsOnInput(text) {
      clearTimeout(_inputDebounce);

      // Autocomplete mode: "/" prefix
      if (text.startsWith('/')) {
        this._boltsUpdateAutocomplete(text);
        this._boltsHideChips();
        return;
      }

      // Hide autocomplete if not in "/" mode
      if (_acVisible) this._boltsHideAutocomplete();

      // Debounce regex detection (150ms)
      if (!text || text.length < 2) {
        this._boltsHideChips();
        return;
      }
      _inputDebounce = setTimeout(() => this._boltsRunRegex(text), 150);
    },

    // ── Regex detection pass ──────────────────────────
    _boltsRunRegex(text) {
      const matches = [];
      for (const { bolt, re } of _compiledRegexes) {
        const m = text.match(re);
        if (!m) continue;
        // By default, regex bolts only fire when the match starts at the beginning
        // (ignore leading whitespace). Bolts with match_anywhere: true bypass this.
        if (!bolt.matchAnywhere && m.index > 0 && text.slice(0, m.index).trim().length > 0) continue;

        // Counter-check (JS only for now)
        if (bolt.counterCheck) {
          try {
            const checkFn = new Function('match', 'input', `return (${bolt.counterCheck})(match, input);`);
            if (!checkFn(m, text)) continue;
          } catch (e) {
            console.warn('[Bolts] Counter-check failed for', bolt.command, e);
            continue;
          }
        }
        matches.push({ bolt, match: m });
      }
      _activeChips = matches;
      if (matches.length > 0) {
        this._boltsShowChips(matches);
      } else {
        this._boltsHideChips();
      }
    },

    // ── Chip rendering ────────────────────────────────
    _boltsEnsureChipContainer() {
      if (_chipContainer) return;
      const inputArea = document.querySelector('.cv2-input-area');
      if (!inputArea) return;
      // Make input area position relative for absolute chip positioning
      if (getComputedStyle(inputArea).position === 'static') {
        inputArea.style.position = 'relative';
      }
      _chipContainer = document.createElement('div');
      _chipContainer.className = 'cv2-bolt-chips';
      inputArea.appendChild(_chipContainer);
    },

    _boltsShowChips(matches) {
      this._boltsEnsureChipContainer();
      if (!_chipContainer) return;

      _chipContainer.innerHTML = '';
      for (const { bolt } of matches) {
        const chip = document.createElement('div');
        chip.className = 'cv2-bolt-chip';
        chip.innerHTML = `${_icon(bolt.icon || 'bolt')} ${bolt.label || bolt.command}` +
          `<span class="cv2-bolt-chip-shortcut">${_quickHint}</span>`;
        chip.title = `${bolt.label || bolt.command} — click or ${_isMac ? '⌘' : 'Ctrl'}+Enter`;
        chip.addEventListener('click', () => {
          const text = this.el.textarea.value.trim();
          this.el.textarea.value = '';
          this._autoResizeTextarea();
          this._updateSendButton();
          this._boltsHideChips();
          this._boltsExecute(bolt, text, true);
        });
        _chipContainer.appendChild(chip);
      }
      _chipContainer.classList.add('visible');
    },

    _boltsHideChips() {
      _activeChips = [];
      if (_chipContainer) _chipContainer.classList.remove('visible');
    },

    // ── Autocomplete ──────────────────────────────────
    _boltsEnsureAutocomplete() {
      if (_acContainer) return;
      const inputArea = document.querySelector('.cv2-input-area');
      if (!inputArea) return;
      if (getComputedStyle(inputArea).position === 'static') {
        inputArea.style.position = 'relative';
      }
      _acContainer = document.createElement('div');
      _acContainer.className = 'cv2-bolt-autocomplete';
      inputArea.appendChild(_acContainer);
    },

    _boltsUpdateAutocomplete(text) {
      this._boltsEnsureAutocomplete();
      if (!_acContainer) return;

      const afterSlash = text.slice(1).toLowerCase(); // remove "/"
      // Split on first space: "/product 632..." → command="product", hasArgs=true
      const spaceIdx = afterSlash.indexOf(' ');
      const query = spaceIdx >= 0 ? afterSlash.slice(0, spaceIdx) : afterSlash;
      const hasArgs = spaceIdx >= 0;
      const device = _boltDevice();

      // Filter and sort bolts
      _acItems = _bolts.filter(b => {
        if (!_boltDeviceOk(b)) return false;
        if (!query) return true; // show all on bare "/"
        const cmd = b.command.toLowerCase();
        const aliases = (b.aliases || []).map(a => a.toLowerCase());
        const label = (b.label || '').toLowerCase();
        // If user typed args (e.g. "/product 632..."), require exact command match
        if (hasArgs) return cmd === query || aliases.some(a => a === query);
        return cmd.includes(query) || aliases.some(a => a.includes(query)) || label.includes(query);
      }).slice(0, 15); // max 15 results

      if (_acItems.length === 0) {
        this._boltsHideAutocomplete();
        return;
      }

      _acContainer.innerHTML = '';
      _acIndex = -1;

      for (let i = 0; i < _acItems.length; i++) {
        const b = _acItems[i];
        const item = document.createElement('div');
        item.className = 'cv2-bolt-ac-item';
        item.dataset.idx = i;
        const aliasText = (b.aliases || []).length > 0 ? b.aliases.map(a => '/' + a).join(', ') : '';
        item.innerHTML =
          `${_icon(b.icon || 'bolt')}` +
          `<span class="cv2-bolt-ac-cmd">/${b.command}</span>` +
          `<span class="cv2-bolt-ac-desc">${b.description || b.label || ''}</span>` +
          (aliasText ? `<span class="cv2-bolt-ac-alias">${aliasText}</span>` : '');
        item.addEventListener('click', () => {
          this._boltsExecuteFromAutocomplete(b);
        });
        item.addEventListener('mouseenter', () => {
          _acIndex = i;
          this._boltsHighlightAcItem();
        });
        _acContainer.appendChild(item);
      }

      _acContainer.classList.add('visible');
      _acVisible = true;
      // Defer so the current click doesn't immediately close it
      setTimeout(() => document.addEventListener('click', _acOutsideClick, true), 0);
    },

    _boltsHighlightAcItem() {
      if (!_acContainer) return;
      const items = _acContainer.querySelectorAll('.cv2-bolt-ac-item');
      items.forEach((el, i) => el.classList.toggle('active', i === _acIndex));
    },

    _boltsHideAutocomplete() {
      if (_acContainer) _acContainer.classList.remove('visible');
      _acVisible = false;
      _acIndex = -1;
      _acItems = [];
      document.removeEventListener('click', _acOutsideClick, true);
    },

    _boltsExecuteFromAutocomplete(bolt) {
      const text = this.el.textarea.value.trim();
      // Extract args after the command: "/who Michael" → "Michael"
      const parts = text.split(/\s+/);
      const args = parts.slice(1).join(' ');
      this._boltsHideAutocomplete();
      this.el.textarea.value = '';
      this._autoResizeTextarea();
      this._updateSendButton();
      this._boltsExecute(bolt, args);
    },

    // ── Keyboard handling ─────────────────────────────
    _boltsOnKeydown(e) {
      // Autocomplete navigation (arrow keys, tab, escape)
      if (_acVisible) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          _acIndex = Math.min(_acIndex + 1, _acItems.length - 1);
          this._boltsHighlightAcItem();
          return true;
        }
        if (e.key === 'ArrowUp') {
          e.preventDefault();
          _acIndex = Math.max(_acIndex - 1, 0);
          this._boltsHighlightAcItem();
          return true;
        }
        if (e.key === 'Escape') {
          e.preventDefault();
          this._boltsHideAutocomplete();
          this.el.textarea.value = '';
          this._autoResizeTextarea();
          return true;
        }
        // Tab or Enter in autocomplete: execute the selected item
        if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
          e.preventDefault();
          const idx = _acIndex >= 0 ? _acIndex : 0;
          if (_acItems[idx]) {
            this._boltsExecuteFromAutocomplete(_acItems[idx]);
          }
          return true;
        }
      }

      // ── Ctrl/Cmd+Enter: quick bolt execution (no nudge entry) ──
      if (e.key === 'Enter' && !e.shiftKey && (e.ctrlKey || e.metaKey)) {
        const text = this.el.textarea.value.trim();
        if (!text) { this._boltsHideChips(); return false; }


        for (const { bolt, re } of _compiledRegexes) {
          const m = text.match(re);
          if (!m) continue;
          if (!bolt.matchAnywhere && m.index > 0 && text.slice(0, m.index).trim().length > 0) continue;
          if (bolt.counterCheck) {
            try {
              const checkFn = new Function('match', 'input', `return (${bolt.counterCheck})(match, input);`);
              if (!checkFn(m, text)) continue;
            } catch { continue; }
          }
          e.preventDefault();
          this.el.textarea.value = '';
          this._autoResizeTextarea();
          this._updateSendButton();
          this._boltsHideChips();
          this._boltsExecute(bolt, text, true);
          return true;
        }
        // No regex match — hide chips, fall through to sendMessage
        this._boltsHideChips();
        return false;
      }

      // ── Enter: ALWAYS do a fresh regex scan (never rely on _activeChips state) ──
      if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
        const text = this.el.textarea.value.trim();
        if (!text) return false;


        // ── Path A: /command with arguments → find bolt by exact command match ──
        if (text.startsWith('/')) {
          const afterSlash = text.slice(1);
          const spaceIdx = afterSlash.indexOf(' ');
          const cmd = (spaceIdx >= 0 ? afterSlash.slice(0, spaceIdx) : afterSlash).toLowerCase();
          const args = spaceIdx >= 0 ? afterSlash.slice(spaceIdx + 1).trim() : '';

          const bolt = _bolts.find(b => {
            if (!_boltDeviceOk(b)) return false;
            if (b.command.toLowerCase() === cmd) return true;
            return (b.aliases || []).some(a => a.toLowerCase() === cmd);
          });

          if (bolt) {
            e.preventDefault();
            this.el.textarea.value = '';
            this._autoResizeTextarea();
            this._updateSendButton();
            this._boltsHideChips();
            this._boltsHideAutocomplete();
            this._boltsExecute(bolt, args || undefined);
            return true;
          }
          return false;
        }

        // ── Path B: regex scan against ALL compiled regex bolts ──
        for (const { bolt, re } of _compiledRegexes) {
          const m = text.match(re);
          if (!m) continue;
          if (!bolt.matchAnywhere && m.index > 0 && text.slice(0, m.index).trim().length > 0) {
            continue;
          }
          if (bolt.counterCheck) {
            try {
              const checkFn = new Function('match', 'input', `return (${bolt.counterCheck})(match, input);`);
              if (!checkFn(m, text)) {
                continue;
              }
            } catch { continue; }
          }
          // Match found — execute bolt instead of sending to LLM
          e.preventDefault();
          this.el.textarea.value = '';
          this._autoResizeTextarea();
          this._updateSendButton();
          this._boltsHideChips();
          this._boltsExecute(bolt, text);
          return true;
        }

        // No bolt matched — hide chips before falling through to sendMessage
        this._boltsHideChips();
      }

      return false; // not handled
    },

    // ── Execute a bolt action ─────────────────────────
    // quick=true: chip click or Ctrl+Enter — execute without entering the nudge
    _boltsExecute(bolt, input, quick) {
      input = input || this.el.textarea.value.trim();
      const action = bolt.action;
      if (!action) {
        console.warn('[Bolts] No action defined for', bolt.command);
        return;
      }

      switch (action.type) {
        case 'url':
          this._boltsExecUrl(action, input, bolt);
          break;
        case 'prompt':
          this._boltsExecPrompt(action, input, bolt, quick);
          break;
        case 'mcp_tool':
          this._boltsExecMcpTool(action, input, bolt);
          break;
        case 'app':
          this._boltsExecApp(action, input, bolt);
          break;
        case 'local_eval':
          this._boltsExecLocalEval(action, input, bolt);
          break;
        case 'worker_decode':
          this._boltsExecWorkerDecode(action, input, bolt, quick);
          break;
        case 'callback':
          if (typeof action.handler === 'function') action.handler(input, bolt, this);
          break;
        case 'app_ext':
          // Activate an app extension by name — pass viewer mode hint
          if (action.extension && this.appExtActivate) {
            this._appExtBoltMode = action.mode || 'viewer';
            this.appExtActivate(action.extension);
          }
          break;
        default:
          console.warn('[Bolts] Unknown action type:', action.type);
      }
    },

    // ── URL action ────────────────────────────────────
    _boltsExecUrl(action, input, bolt) {
      let url = action.url || '';
      url = url.replace(/\{input\}/g, encodeURIComponent(input));
      url = url.replace(/\{match\}/g, encodeURIComponent(input));
      const target = action.target || '_blank';
      window.open(url, target);
    },

    // ── Prompt action ─────────────────────────────────
    _boltsExecPrompt(action, input, bolt, quick) {
      let prompt = action.template || '{input}';
      prompt = prompt.replace(/\{input\}/g, input);
      prompt = prompt.replace(/\{match\}/g, input);

      if (action.auto_send !== false) {
        // In quick mode, send in current context (no nudge switch)
        if (!quick && bolt.nudgeUid && bolt.nudgeUid !== '__system__') {
          this.ws.send({
            type: 'new_chat',
            preset: { nudge_uid: bolt.nudgeUid, type: 'nudge' },
          });
          // Small delay to let nudge activate, then send
          setTimeout(() => {
            this.el.textarea.value = prompt;
            this.sendMessage();
          }, 300);
        } else {
          this.el.textarea.value = prompt;
          this.sendMessage();
        }
      } else {
        // Pre-fill input for user to edit
        this.el.textarea.value = prompt;
        this._autoResizeTextarea();
        this._updateSendButton();
        this.el.textarea.focus();
        // Move cursor to end
        this.el.textarea.selectionStart = this.el.textarea.selectionEnd = prompt.length;
      }
    },

    // ── MCP tool action ───────────────────────────────
    _boltsExecMcpTool(action, input, bolt) {
      // Activate the bolt's nudge first (to ensure MCP tools are available)
      if (bolt.nudgeUid && bolt.nudgeUid !== '__system__') {
        this.ws.send({
          type: 'new_chat',
          preset: { nudge_uid: bolt.nudgeUid, type: 'nudge' },
        });
      }

      // Build the prompt that will trigger the tool call
      const argMapping = action.arg_mapping || {};
      let toolPrompt = `Use the tool "${action.tool_name}" with: `;
      const args = {};
      for (const [paramName, template] of Object.entries(argMapping)) {
        args[paramName] = template.replace(/\{input\}/g, input).replace(/\{match\}/g, input);
      }
      toolPrompt += JSON.stringify(args);

      // Send as a message that instructs the AI to call the tool
      setTimeout(() => {
        this.el.textarea.value = toolPrompt;
        this.sendMessage();
      }, bolt.nudgeUid && bolt.nudgeUid !== '__system__' ? 300 : 0);
    },

    // ── Local eval action (evaluate math client-side) ─
    _boltsExecLocalEval(action, input, bolt) {
      const result = _tryLocalEval(input);
      if (result !== null) {
        this._boltsShowLocalResult(input, result, bolt);
      } else {
        // Too complex for local eval — fall through to LLM
        this._boltsExecPrompt(
          { template: '{input}', auto_send: true },
          input, bolt,
        );
      }
    },

    // ── Show local result in chat (no LLM) ─────────────
    _boltsShowLocalResult(input, result, bolt) {
      if (!this.chatVisible) this.showChat();
      const msgs = this.el.messages;
      if (!msgs) return;

      const timestamp = new Date().toISOString();
      const formatted = _formatNumber(result);

      // User bubble
      const userEl = document.createElement('div');
      userEl.className = 'cv2-msg-user';
      userEl.innerHTML = `<div class="cv2-msg-user-bubble">${this.md.render(input)}</div>`;
      msgs.appendChild(userEl);

      // Result bubble
      const resultEl = document.createElement('div');
      resultEl.className = 'cv2-msg-assistant';
      const body = document.createElement('div');
      body.className = 'cv2-msg-body';
      body.innerHTML =
        `<span class="material-icons" style="font-size:18px;opacity:0.5;vertical-align:middle;margin-right:4px">calculate</span>` +
        `<strong>${formatted}</strong>`;
      resultEl.appendChild(body);
      this._addMessageActions(body, formatted, timestamp);
      msgs.appendChild(resultEl);

      this._scrollToBottom();

      // Persist via server (creates conversation, saves to IDB)
      this.ws.send({
        type: 'store_bolt_result',
        user_text: input,
        assistant_text: formatted,
      });
    },

    // ── Worker decode action (decode in droplet Worker sandbox, show rich card, then optionally prompt LLM) ─
    // quick=true: show card only, skip followup LLM prompt (no nudge entry)
    async _boltsExecWorkerDecode(action, input, bolt, quick) {
      if (!bolt.nudgeUid || bolt.nudgeUid === '__system__') {
        console.warn('[Bolts] worker_decode requires a nudge with an active Worker');
        this._boltsExecPrompt({ template: action.fallback_template || '{input}', auto_send: true }, input, bolt);
        return;
      }

      const handlerName = action.handler;
      if (!handlerName) {
        console.warn('[Bolts] worker_decode action missing "handler" field');
        this._boltsExecPrompt({ template: action.fallback_template || '{input}', auto_send: true }, input, bolt);
        return;
      }

      // If Worker is not running, start it on demand without resetting the chat.
      // Uses start_bolt_worker which activates the Worker in the current session.
      let workerEntry = (this._mcpWorkers || {})[bolt.nudgeUid];
      if (!workerEntry) {
        this.ws.send({
          type: 'start_bolt_worker',
          nudge_uid: bolt.nudgeUid,
        });
        // Poll for Worker readiness (up to 12s — includes server fetch + Worker init)
        const pollStart = Date.now();
        for (let i = 0; i < 60; i++) {
          await new Promise(r => setTimeout(r, 200));
          workerEntry = (this._mcpWorkers || {})[bolt.nudgeUid];
          if (workerEntry) {
            break;
          }
        }
        if (!workerEntry) {
          console.warn('[Bolts] Worker failed to start for', bolt.nudgeUid, 'after', Date.now() - pollStart, 'ms');
          // Send fallback as plain message (no new_chat — don't reset the chat)
          this._boltsExecPrompt({ template: action.fallback_template || '{input}', auto_send: true }, input, { nudgeUid: '__system__' });
          return;
        }
      } else {
      }

      // Call the Worker's registered bolt handler
      const res = await this._callWorkerBoltDecode(bolt.nudgeUid, handlerName, input);
      if (res.error || !res.result) {
        console.warn('[Bolts] Worker decode failed:', res.error || 'no result');
        this._boltsExecPrompt(
          { template: action.fallback_template || '{input}', auto_send: true },
          input, { nudgeUid: '__system__' },
        );
        return;
      }

      const decoded = res.result;
      if (!decoded.__rich_mcp__) {
        this._boltsExecPrompt(
          { template: action.fallback_template || '{input}', auto_send: true },
          input, { nudgeUid: '__system__' },
        );
        return;
      }

      if (!this.chatVisible) this.showChat();
      if (!this.el.messages) return;

      // Clear input
      this.el.textarea.value = '';
      this._autoResizeTextarea();

      // Render user bubble
      const userEl = document.createElement('div');
      userEl.className = 'cv2-msg-user';
      userEl.innerHTML = `<div class="cv2-msg-user-bubble">${this.md.render(input)}</div>`;
      this.el.messages.appendChild(userEl);

      // Render assistant bubble with rich_mcp block via the standard markdown + hydration pipeline
      const richJson = JSON.stringify(decoded.__rich_mcp__, null, 0);
      const assistantContent = '```rich_mcp\n' + richJson + '\n```';
      const timestamp = new Date().toISOString();
      const el = document.createElement('div');
      el.className = 'cv2-msg-assistant';
      const body = document.createElement('div');
      body.className = 'cv2-msg-body';
      body.innerHTML = this.md.render(assistantContent);
      el.appendChild(body);
      this._addMessageActions(body, assistantContent, timestamp);
      this.el.messages.appendChild(el);
      if (this.md.hydratePluginBlocks) await this.md.hydratePluginBlocks(body);
      this._scrollToBottom();

      // Persist to server — stored as user + assistant messages with rich_mcp code block
      this.ws.send({
        type: 'store_bolt_result',
        user_text: input,
        rich_mcp: decoded.__rich_mcp__,
      });

      // Send follow-up prompt to LLM with decoded data as context
      // In quick mode, skip the followup — just show the decode card
      if (!quick && action.followup_template) {
        const summary = decoded.__rich_mcp__.llm_summary || JSON.stringify(decoded.__rich_mcp__.info);
        const prompt = action.followup_template
          .replace(/\{input\}/g, input)
          .replace(/\{summary\}/g, summary);

        // Small delay so the card renders
        setTimeout(() => {
          this.ws.send({ type: 'send_message', text: prompt });
        }, 200);
      }
    },

    // ── App action (open mini-app in floating window) ─
    _boltsExecApp(action, input, bolt) {
      // If args provided, try local eval first (e.g. /calc 5+5)
      if (input) {
        const result = _tryLocalEval(input);
        if (result !== null) {
          this._boltsShowLocalResult(input, result, bolt);
          return;
        }
      }
      const appName = action.app;
      // Lazy-resolve: bolt app scripts may register after _boltsInit
      const apps = window._boltApps || {};
      const boltApp = apps[appName];
      if (!boltApp) {
        console.warn('[Bolts] App not found:', appName, 'available:', Object.keys(apps));
        return;
      }
      this._boltsOpenApp(boltApp, bolt, input, action.args);
    },

    // ── Open a bolt mini-app in a floating window ─────
    _boltsOpenApp(app, bolt, input, extraArgs, reviveState) {
      const isDark = () => !!document.querySelector('#chat-app.cv2-dark');
      const locale = document.documentElement.lang || 'en';

      // Build floating window
      const overlay = document.createElement('div');
      overlay.className = 'cv2-bolt-app-window';
      // App can specify preferred size, otherwise use defaults
      const defaultW = app.width || 420;
      const defaultH = app.height || 560;
      overlay.style.cssText = `
        position: fixed; z-index: 9999;
        width: ${defaultW}px; height: ${defaultH}px;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        border-radius: 14px;
        overflow: hidden;
        display: flex; flex-direction: column;
        box-shadow: 0 12px 48px rgba(0,0,0,0.25);
      `;
      overlay.style.background = isDark() ? '#1a1a2e' : '#fff';
      overlay.style.border = isDark() ? '1px solid rgba(255,255,255,0.08)' : '1px solid #e5e7eb';

      // Header (draggable)
      const header = document.createElement('div');
      header.style.cssText = `
        display: flex; align-items: center; gap: 8px;
        padding: 8px 12px; cursor: move; user-select: none;
        border-bottom: 1px solid ${isDark() ? 'rgba(255,255,255,0.06)' : '#f0f0f0'};
        flex-shrink: 0;
        background: ${isDark() ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)'};
      `;
      const dark = isDark();
      header.innerHTML = `
        <span class="material-icons" style="font-size:18px;opacity:0.7;color:${dark ? '#93bbfd' : '#555'}">${bolt.icon || app.icon || 'apps'}</span>
        <span style="flex:1;font-size:13px;font-weight:600;color:${dark ? '#e5e7eb' : '#1f2937'}">${bolt.label || app.name}</span>
        <span class="material-icons" style="font-size:18px;cursor:pointer;opacity:${dark ? '0.7' : '0.4'};color:${dark ? '#e5e7eb' : '#333'};padding:2px" id="bolt-app-close">close</span>
      `;

      // Content container
      const content = document.createElement('div');
      content.style.cssText = 'flex:1;overflow:auto;position:relative;min-height:0;';

      overlay.appendChild(header);
      overlay.appendChild(content);
      document.body.appendChild(overlay);

      // Close lock state — apps can lock close (e.g. focus timer)
      let _closeLocked = false;
      const closeBtn = header.querySelector('#bolt-app-close');

      // Close button
      closeBtn.addEventListener('click', () => {
        if (_closeLocked) return;
        if (appInstance && appInstance.destroy) appInstance.destroy();
        overlay.remove();
      });

      // Draggable header
      let dragX, dragY, startX, startY;
      header.addEventListener('mousedown', (e) => {
        if (e.target.id === 'bolt-app-close') return;
        dragX = e.clientX;
        dragY = e.clientY;
        const rect = overlay.getBoundingClientRect();
        startX = rect.left;
        startY = rect.top;
        overlay.style.transform = 'none';
        overlay.style.left = startX + 'px';
        overlay.style.top = startY + 'px';

        const onMove = (ev) => {
          overlay.style.left = (startX + ev.clientX - dragX) + 'px';
          overlay.style.top = (startY + ev.clientY - dragY) + 'px';
        };
        const onUp = () => {
          document.removeEventListener('mousemove', onMove);
          document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
      });

      // Esc to close
      const onEsc = (e) => {
        if (e.key === 'Escape' && document.body.contains(overlay)) {
          if (_closeLocked) return;
          if (appInstance && appInstance.destroy) appInstance.destroy();
          overlay.remove();
          document.removeEventListener('keydown', onEsc);
        }
      };
      document.addEventListener('keydown', onEsc);

      // Build memory stub (localStorage fallback until MongoDB API is wired)
      const nudgeUid = bolt.nudgeUid || '__system__';
      const memPrefix = `bolt:${nudgeUid}:`;
      const memory = {
        async get(k) { try { return JSON.parse(localStorage.getItem(memPrefix + k)); } catch { return null; } },
        async set(k, v) { localStorage.setItem(memPrefix + k, JSON.stringify(v)); },
        async delete(k) { localStorage.removeItem(memPrefix + k); },
        async keys() {
          const out = [];
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.startsWith(memPrefix) && !key.includes(':s:') && !key.endsWith(':__revive__')) {
              out.push(key.slice(memPrefix.length));
            }
          }
          return out;
        },
        async sessionGet(k) { try { return JSON.parse(sessionStorage.getItem(memPrefix + 's:' + k)); } catch { return null; } },
        async sessionSet(k, v) { sessionStorage.setItem(memPrefix + 's:' + k, JSON.stringify(v)); },
        async sessionDelete(k) { sessionStorage.removeItem(memPrefix + 's:' + k); },
        async sessionKeys() {
          const out = [];
          for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            if (key.startsWith(memPrefix + 's:')) out.push(key.slice((memPrefix + 's:').length));
          }
          return out;
        },
        async registerRevive(cfg) {
          const entry = { ...cfg, nudgeUid, command: bolt.command, appName: app.name, boltIcon: bolt.icon || app.icon || 'apps', boltLabel: bolt.label || app.name };
          localStorage.setItem(memPrefix + '__revive__', JSON.stringify(entry));
        },
        async unregisterRevive() {
          localStorage.removeItem(memPrefix + '__revive__');
        },
      };

      // Render app
      const context = {
        isDark: isDark(),
        locale,
        close: () => {
          if (_closeLocked) return;
          if (appInstance && appInstance.destroy) appInstance.destroy();
          overlay.remove();
          document.removeEventListener('keydown', onEsc);
        },
        lockClose: () => {
          _closeLocked = true;
          closeBtn.style.opacity = '0.15';
          closeBtn.style.cursor = 'not-allowed';
        },
        unlockClose: () => {
          _closeLocked = false;
          closeBtn.style.opacity = '0.4';
          closeBtn.style.cursor = 'pointer';
        },
        resize: (w, h) => {
          // Animate to new size, keep position
          overlay.style.transition = 'width 0.2s ease, height 0.2s ease';
          overlay.style.width = w + 'px';
          overlay.style.height = h + 'px';
          setTimeout(() => { overlay.style.transition = ''; }, 250);
        },
        reviveState: reviveState || null,
        memory,
      };

      let appInstance = null;
      try {
        appInstance = app.render(content, { input, ...(extraArgs || {}) }, context);
      } catch (e) {
        console.error('[Bolts] App render failed:', e);
        content.innerHTML = `<div style="padding:20px;color:red">App failed to load: ${e.message}</div>`;
      }
    },

    // ── Auto-revive: check for revive registrations ───
    _boltsCheckRevive() {
      const device = _boltDevice();
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (!key || !key.startsWith('bolt:') || !key.endsWith(':__revive__')) continue;
        try {
          const cfg = JSON.parse(localStorage.getItem(key));
          if (!cfg || !cfg.appName) continue;

          // Check device compatibility
          const app = this._boltApps[cfg.appName];
          if (!app) continue;
          const boltDevices = app.devices || ['desktop', 'tablet', 'mobile'];
          if (!boltDevices.includes(device)) {
            // Fallback: show chip instead of window
            const fallback = cfg.fallback || 'chip';
            if (fallback === 'chip') {
              this._boltsShowReviveChip(cfg, app);
            }
            continue;
          }

          // Revive as floating window
          const fakeBolt = {
            command: cfg.command || cfg.appName,
            label: cfg.boltLabel || cfg.label || cfg.appName,
            icon: cfg.boltIcon || app.icon || 'apps',
            nudgeUid: cfg.nudgeUid || '__system__',
          };
          this._boltsOpenApp(app, fakeBolt, '', {}, cfg.state);
        } catch (e) {
          console.warn('[Bolts] Revive failed for', key, e);
        }
      }
    },

    _boltsShowReviveChip(cfg, app) {
      // Show a persistent chip in the topbar area
      const topbar = document.querySelector('.cv2-topbar-actions') || document.querySelector('.cv2-topbar');
      if (!topbar) return;
      const chip = document.createElement('div');
      chip.className = 'cv2-bolt-revive-chip';
      chip.innerHTML = `${_icon(cfg.boltIcon || app.icon || 'apps')} ${cfg.label || cfg.appName}`;
      chip.title = 'Click to reopen';
      chip.addEventListener('click', () => {
        chip.remove();
        const fakeBolt = {
          command: cfg.command || cfg.appName,
          label: cfg.boltLabel || cfg.label || cfg.appName,
          icon: cfg.boltIcon || app.icon || 'apps',
          nudgeUid: cfg.nudgeUid || '__system__',
        };
        this._boltsOpenApp(app, fakeBolt, '', {}, cfg.state);
      });
      topbar.appendChild(chip);
    },

    // ── Handle bolt definitions from server ───────────
    _handleBoltDefs(msg) {
      if (msg.nudge_uid && msg.bolts) {
        this._boltsRegister(msg.nudge_uid, msg.bolts);
      }
    },
  });

  // ── Feature registration ──────────────────────────────
  ChatFeatures.register('bolts', {
    handleMessage: {
      'bolt_defs': '_handleBoltDefs',
    },
    initState(app) {
      app._boltsInit();
    },
    bindEvents(app) {
      // Hook into existing input handler
      const textarea = app.el.textarea;
      if (!textarea) return;

      // Prepend our input handler
      textarea.addEventListener('input', () => {
        app._boltsOnInput(textarea.value.trim());
      });

      // NOTE: keydown is handled by the main handler in chat-app-core.js
      // which calls _boltsOnKeydown() before sendMessage(). This ensures
      // bolts always intercept Enter before the message is sent to the LLM.

      // Check for auto-revive after a short delay
      setTimeout(() => app._boltsCheckRevive(), 500);
    },
  });

})();
