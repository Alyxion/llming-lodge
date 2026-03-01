/**
 * chat-plus-menu.js — Plus menu, tool toggles, context circle, contact cards
 * Extracted from chat-app.js
 */
(function() {
  Object.assign(window._ChatAppProto, {

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
            <span class="cv2-shimmer" style="font-size:16px">${this.t('chat.generating_image')}</span>
            <div class="cv2-image-placeholder">
              <div class="cv2-spinner-dots"><span></span><span></span><span></span></div>
            </div>
          </div>
        `;
      } else if (tc.name === 'web_search') {
        html += `
          <div class="cv2-tool-pending">
            <div class="cv2-spinner-dots"><span></span><span></span><span></span></div>
            <span class="cv2-shimmer">${this.t('chat.searching_web')}</span>
          </div>
        `;
      } else {
        html += `
          <div class="cv2-tool-pending">
            <div class="cv2-spinner-dots"><span></span><span></span><span></span></div>
            <span class="cv2-shimmer">${this.t('chat.running_tool', { name: this._escHtml(tc.display_name) })}</span>
          </div>
        `;
      }
    }

    // Completed tools
    for (const tc of completed) {
      html += `
        <div class="cv2-tool-completed">
          <span class="material-icons cv2-icon-check">check_circle</span>
          <span>${this.t('chat.used_tool', { name: this._escHtml(tc.display_name) })}</span>
        </div>
      `;
    }

    this._currentToolArea.innerHTML = html;
  },

  /** Render a single tool toggle row. Greyed out + tooltip when unavailable. */
  _toolItemHtml(tool, icon, extraStyle) {
    const disabled = !tool.available;
    const disabledStyle = disabled ? 'opacity:0.4;pointer-events:none;' : '';
    const title = disabled && tool.required_provider
      ? `Requires: ${Array.isArray(tool.required_provider) ? tool.required_provider.join(', ') : tool.required_provider}`
      : '';
    const toggleCls = tool.enabled && !disabled ? 'cv2-on' : '';
    const toggleName = tool.group_id || tool.name;
    return `
      <div class="cv2-tool-item${disabled ? ' cv2-tool-disabled' : ''}" data-tool="${this._escAttr(tool.name)}" style="${extraStyle || ''}${disabledStyle}"${title ? ` title="${this._escAttr(title)}"` : ''}>
        <span class="material-icons" style="font-size:16px">${icon || 'build'}</span>
        <span style="flex:1">${this._escHtml(tool.display_name)}</span>
        <div class="cv2-tool-toggle ${toggleCls}" data-toggle="${this._escAttr(toggleName)}"></div>
      </div>
    `;
  },

  _renderPlusMenu() {
    // 3-way bucketing: dev, flyout (own top-level flyout), and regular tools
    const devCategories = new Set(['Experimental']);
    const devCategoryTools = {};
    const flyoutTools = {};   // category -> tools[]
    const allTools = [];

    for (const tool of this.tools) {
      // Hide tools that are unavailable for non-restriction reasons (e.g. provider-native only)
      // but show provider-restricted tools as greyed out
      if (!tool.available && !tool.required_provider) continue;
      const cat = tool.category || this.t('chat.general_category');
      if (devCategories.has(cat)) {
        if (!devCategoryTools[cat]) devCategoryTools[cat] = [];
        devCategoryTools[cat].push(tool);
      } else if (tool.flyout) {
        if (!flyoutTools[cat]) flyoutTools[cat] = [];
        flyoutTools[cat].push(tool);
      } else {
        allTools.push(tool);
      }
    }

    let html = '';

    // File / clipboard selection
    html += `
      <button class="cv2-menu-item" data-action="from-file">
        <span class="material-icons">upload_file</span> ${this.t('chat.from_file')}
      </button>
      <button class="cv2-menu-item" data-action="from-clipboard">
        <span class="material-icons">content_paste</span> ${this.t('chat.from_clipboard')}
      </button>
    `;

    // Single "Tools" flyout containing all non-dev tool toggles
    const hasFlyouts = Object.keys(flyoutTools).length > 0;
    if (allTools.length > 0 || hasFlyouts) {
      html += `
        <div class="cv2-menu-item cv2-flyout-trigger" data-flyout="tools">
          <span class="material-icons" style="font-size:16px">build</span>
          <span style="flex:1">Tools</span>
          <span class="material-icons" style="font-size:16px;color:var(--chat-text-muted)">chevron_right</span>
        </div>
        <div class="cv2-flyout-menu" data-flyout-panel="tools">
      `;

      // Nested flyout sub-categories (e.g. Office 365) — rendered as sub-flyouts inside Tools
      for (const [cat, tools] of Object.entries(flyoutTools)) {
        const flyoutKey = 'flyout_' + cat.toLowerCase().replace(/\s+/g, '_');
        const availTools = tools.filter(t => t.available);
        const allOn = availTools.length > 0 && availTools.every(t => t.enabled);
        html += `
          <div class="cv2-tool-item cv2-flyout-trigger" data-flyout="${this._escAttr(flyoutKey)}" style="position:relative">
            <span class="material-icons" style="font-size:16px">smart_toy</span>
            <span style="flex:1">${this._escHtml(cat)}</span>
            <span class="material-icons" style="font-size:16px;color:var(--chat-text-muted)">chevron_right</span>
          </div>
          <div class="cv2-flyout-menu" data-flyout-panel="${this._escAttr(flyoutKey)}">
            <div class="cv2-tool-item cv2-toggle-all" data-toggle-all-category="${this._escAttr(cat)}">
              <span style="flex:1;font-weight:600;font-size:12px">${this._escHtml(cat)}</span>
              <div class="cv2-tool-toggle ${allOn ? 'cv2-on' : ''}" data-toggle-all="${this._escAttr(cat)}"></div>
            </div>
        `;
        for (const tool of tools) {
          html += this._toolItemHtml(tool, tool.icon || 'smart_toy');
        }
        html += `</div>`;
      }

      // Separate into categorized tools and standalone collapsed MCP groups
      const categorized = {};
      const standalone = [];

      for (const tool of allTools) {
        if (tool.collapse_tools) {
          standalone.push(tool);
        } else {
          const cat = tool.category || this.t('chat.general_category');
          if (!categorized[cat]) categorized[cat] = [];
          categorized[cat].push(tool);
        }
      }

      // Render categorized tools with category headers + toggle-all
      for (const [cat, tools] of Object.entries(categorized)) {
        if (tools.length > 1) {
          const availTools = tools.filter(t => t.available);
          const allOn = availTools.length > 0 && availTools.every(t => t.enabled);
          html += `
            <div class="cv2-tool-item cv2-toggle-all" data-toggle-all-category="${this._escAttr(cat)}">
              <span class="cv2-tool-category" style="padding:0;flex:1">${this._escHtml(cat)}</span>
              <div class="cv2-tool-toggle ${allOn ? 'cv2-on' : ''}" data-toggle-all="${this._escAttr(cat)}"></div>
            </div>
          `;
        } else {
          html += `<div class="cv2-tool-category">${this._escHtml(cat)}</div>`;
        }
        for (const tool of tools) {
          html += this._toolItemHtml(tool, tool.icon || 'build');
        }
      }

      // Render standalone collapsed MCP groups (no category header)
      for (const tool of standalone) {
        html += this._toolItemHtml(tool, tool.icon || 'smart_toy');
      }

      html += `</div>`;
    }

    // Dev submenu (Condense + Experimental tools)
    html += `<div class="cv2-menu-sep"></div>`;
    html += `
      <button class="cv2-menu-item cv2-dev-toggle" data-action="toggle-dev">
        <span class="material-icons" style="font-size:16px">code</span>
        <span style="flex:1">Dev</span>
        <span class="material-icons cv2-dev-arrow" style="font-size:16px">expand_more</span>
      </button>
      <div class="cv2-dev-submenu" style="display:none">
        <button class="cv2-menu-item" data-action="condense">
          <span class="material-icons">compress</span> ${this.t('chat.condense_conversation')}
        </button>
    `;
    for (const [cat, tools] of Object.entries(devCategoryTools)) {
      html += `<div class="cv2-tool-category" style="padding-left:12px">${this._escHtml(cat)}</div>`;
      for (const tool of tools) {
        html += this._toolItemHtml(tool, tool.icon || 'build', 'padding-left:12px;');
      }
    }
    html += `</div>`;

    this.el.plusMenu.innerHTML = html;

    // Bind actions
    this.el.plusMenu.querySelector('[data-action="from-file"]')?.addEventListener('click', () => {
      this.el.fileInput.click();
      this.plusMenuOpen = false;
      this.el.plusMenu.classList.remove('cv2-visible');
    });
    this.el.plusMenu.querySelector('[data-action="from-clipboard"]')?.addEventListener('click', async () => {
      this.plusMenuOpen = false;
      this.el.plusMenu.classList.remove('cv2-visible');
      try {
        const items = await navigator.clipboard.read();
        let handled = false;
        for (const item of items) {
          const imgType = item.types.find(t => t.startsWith('image/'));
          if (imgType) {
            const blob = await item.getType(imgType);
            const file = new File([blob], 'clipboard-image.png', { type: imgType });
            await this._checkImageDimensions(file);
            const dataUri = await this._compressImage(file, ChatApp.MAX_IMAGE_DIM);
            if (dataUri) this._addPendingImage(dataUri);
            handled = true;
          }
        }
        if (!handled) this._showToast('No image found in clipboard', 'warning');
      } catch (err) {
        this._showToast('Cannot read clipboard', 'negative');
      }
    });

    // Flyout triggers (Tools + flyout categories)
    this.el.plusMenu.querySelectorAll('.cv2-flyout-trigger').forEach(trigger => {
      const key = trigger.dataset.flyout;
      const panel = this.el.plusMenu.querySelector(`[data-flyout-panel="${key}"]`);
      if (!panel) return;
      let hideTimeout = null;
      const show = () => { clearTimeout(hideTimeout); panel.classList.add('cv2-visible'); };
      const hide = () => { hideTimeout = setTimeout(() => panel.classList.remove('cv2-visible'), 150); };
      trigger.addEventListener('mouseenter', show);
      trigger.addEventListener('mouseleave', hide);
      panel.addEventListener('mouseenter', () => clearTimeout(hideTimeout));
      panel.addEventListener('mouseleave', hide);
      // Also toggle on click for touch devices
      trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        panel.classList.toggle('cv2-visible');
      });
    });

    // Dev submenu toggle
    this.el.plusMenu.querySelector('[data-action="toggle-dev"]')?.addEventListener('click', (e) => {
      e.stopPropagation();
      const sub = this.el.plusMenu.querySelector('.cv2-dev-submenu');
      const arrow = this.el.plusMenu.querySelector('.cv2-dev-arrow');
      if (sub) {
        const open = sub.style.display !== 'none';
        sub.style.display = open ? 'none' : 'block';
        if (arrow) arrow.textContent = open ? 'expand_more' : 'expand_less';
      }
    });

    this.el.plusMenu.querySelector('[data-action="condense"]')?.addEventListener('click', () => {
      this.ws.send({ type: 'condense' });
      this.plusMenuOpen = false;
      this.el.plusMenu.classList.remove('cv2-visible');
    });

    // Toggle-all handlers
    this.el.plusMenu.querySelectorAll('[data-toggle-all]').forEach(toggle => {
      toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const cat = toggle.dataset.toggleAll;
        const isOn = toggle.classList.contains('cv2-on');
        const targetState = !isOn;

        // Find all tools in this category
        const categoryTools = this.tools.filter(t => {
          const tCat = t.category || this.t('chat.general_category');
          return t.available && tCat === cat;
        });

        // Toggle each tool that differs from target state
        for (const tool of categoryTools) {
          if (tool.enabled !== targetState) {
            tool.enabled = targetState;
            this.ws.send({ type: 'toggle_tool', name: tool.name, enabled: targetState });
            // Update individual toggle UI
            const indToggle = this.el.plusMenu.querySelector(`[data-toggle="${tool.name}"]`);
            if (indToggle) indToggle.classList.toggle('cv2-on', targetState);
          }
        }

        toggle.classList.toggle('cv2-on', targetState);
        this._saveToolPrefs();
      });
    });

    // Individual tool toggles
    this.el.plusMenu.querySelectorAll('.cv2-tool-toggle[data-toggle]:not([data-toggle-all])').forEach(toggle => {
      toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const name = toggle.dataset.toggle;
        const isOn = toggle.classList.contains('cv2-on');
        toggle.classList.toggle('cv2-on', !isOn);
        this.ws.send({ type: 'toggle_tool', name, enabled: !isOn });
        // Update local state — for collapsed groups, toggle all tools in the group
        const tool = this.tools.find(t => t.name === name || t.group_id === name);
        if (tool) tool.enabled = !isOn;
        // Update toggle-all state for this category
        const cat = tool?.category || this.t('chat.general_category');
        const allToggle = this.el.plusMenu.querySelector(`[data-toggle-all="${cat}"]`);
        if (allToggle) {
          const catTools = this.tools.filter(t => t.available && (t.category || this.t('chat.general_category')) === cat);
          const allOn = catTools.every(t => t.enabled);
          allToggle.classList.toggle('cv2-on', allOn);
        }
        // Persist preference globally
        this._saveToolPrefs();
      });
    });
  },

  // ── Tool preference persistence ──────────────────────

  _saveToolPrefs() {
    // Persist user's explicit tool toggles in localStorage so they survive reload.
    // We store the delta from server defaults: { toolName: true/false }
    if (!this.tools || !this.tools.length) return;
    const prefs = {};
    for (const t of this.tools) {
      // Only store tools the user has explicitly toggled (available tools only)
      if (!t.available) continue;
      prefs[t.name] = !!t.enabled;
    }
    try {
      localStorage.setItem('cv2-tool-prefs', JSON.stringify(prefs));
    } catch (_) {}
  },

  _applyToolPrefs() {
    // Restore user's tool preferences from localStorage.
    // Sends toggle_tool messages for tools whose saved state differs from server default.
    let prefs;
    try {
      prefs = JSON.parse(localStorage.getItem('cv2-tool-prefs') || '{}');
    } catch (_) { return; }
    if (!prefs || !Object.keys(prefs).length) return;
    if (!this.tools || !this.tools.length) return;

    let changed = false;
    for (const t of this.tools) {
      if (!t.available) continue;
      const saved = prefs[t.name];
      if (saved !== undefined && saved !== t.enabled) {
        t.enabled = saved;
        this.ws.send({ type: 'toggle_tool', name: t.name, enabled: saved, restore: true });
        changed = true;
      }
    }
    if (changed) this._renderPlusMenu();
  },

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
  },

  _renderContextPopover() {
    const info = this.contextInfo || {};
    const nudgeRow = info.nudgeTokens
      ? `<div class="cv2-context-row"><span>${this.t('chat.ctx_nudge')}</span><strong>${info.nudgeTokens.toLocaleString()} tokens</strong></div>`
      : '';
    const projectRow = info.projectTokens
      ? `<div class="cv2-context-row"><span>${this.t('chat.ctx_project')}</span><strong>${info.projectTokens.toLocaleString()} tokens</strong></div>`
      : '';
    let html = `
      <div class="cv2-context-row"><span>${this.t('chat.ctx_history')}</span><strong>${(info.historyTokens || 0).toLocaleString()} tokens</strong></div>
      ${nudgeRow}
      ${projectRow}
      <div class="cv2-context-row"><span>${this.t('chat.ctx_documents')}</span><strong>${(info.docTokens || 0).toLocaleString()} tokens</strong></div>
      <div class="cv2-context-row"><span>${this.t('chat.ctx_images')}</span><strong>${(info.imageTokens || 0).toLocaleString()} tokens</strong></div>
      <div class="cv2-context-row"><span>${this.t('chat.ctx_tools')}</span><strong>${(info.toolTokens || 0).toLocaleString()} tokens</strong></div>
      <div class="cv2-context-row" style="border-top:1px solid #e5e7eb;padding-top:6px;margin-top:4px">
        <span>${this.t('chat.ctx_total')}</span><strong>${(info.totalTokens || 0).toLocaleString()} / ${(info.maxTokens || 0).toLocaleString()}</strong>
      </div>
      <div class="cv2-context-row"><span>${this.t('chat.ctx_est_cost')}</span><strong>$${info.estCost || '0.00'}</strong></div>`;
    if (this.config.showBudget && this.budget !== undefined && this.budget > 0) {
      html += `
      <div class="cv2-context-row"><span>${this.t('chat.ctx_budget')}</span><strong>$${Number(this.budget).toFixed(2)}</strong></div>`;
    }
    this.el.contextPopover.innerHTML = html;
  },

  // ── Prompt Inspector (admin easter egg) ─────────────

  handlePromptInspector(msg) {
    this._promptInspectorData = msg;
    this._renderPromptInspector(msg);
  },

  _renderPromptInspector(data) {
    // Remove existing
    this._closePromptInspector();

    const sections = data.sections || [];
    const total = data.totalTokens || 1;
    const max = data.maxTokens || 1;
    const pct = Math.min(100, total / max * 100).toFixed(1);

    // Build stacked bar segments
    const barSegs = sections
      .filter(s => s.tokens > 0)
      .map(s => `<div style="flex:${s.tokens};background:${s.color};min-width:2px" title="${this._escHtml(s.label)}: ${s.tokens.toLocaleString()} tk"></div>`)
      .join('');

    // Build section rows with separators
    let finalSections = '';
    for (let i = 0; i < sections.length; i++) {
      const s = sections[i];
      if (s.id === 'messages' || s.id === 'tools') {
        finalSections += `<div class="cv2-pi-separator">${s.id === 'messages' ? 'conversation' : 'tools'}</div>`;
      }
      const pctS = total > 0 ? (s.tokens / total * 100).toFixed(1) : '0.0';
      const content = s.content || '';
      const hasContent = content.length > 0 || (s.items && s.items.length > 0);
      const expandIcon = hasContent ? '<span class="material-icons cv2-pi-expand-icon" style="font-size:14px;transition:transform 0.2s">chevron_right</span>' : '<span style="width:14px;display:inline-block"></span>';

      let detail = '';
      if (s.items && s.id === 'messages') {
        const msgLines = s.items.map(m =>
          `<div class="cv2-pi-msg-item"><span class="cv2-pi-role">${this._escHtml(m.role)}</span><span class="cv2-pi-msg-preview">${this._escHtml(m.preview)}</span><span class="cv2-pi-tk">${m.tokens.toLocaleString()} tk</span></div>`
        ).join('');
        detail = `<div class="cv2-pi-detail" style="display:none">${msgLines}</div>`;
      } else if (s.items && s.id === 'tools') {
        const toolLines = s.items.map(t =>
          `<div class="cv2-pi-tool-item"><span>${this._escHtml(t.name)}</span><span class="cv2-pi-tk">${t.tokens.toLocaleString()} tk</span></div>`
        ).join('');
        detail = `<div class="cv2-pi-detail" style="display:none">${toolLines}</div>`;
      } else if (content) {
        const truncLen = 300;
        const needsTrunc = content.length > truncLen;
        const truncated = needsTrunc ? this._escHtml(content.slice(0, truncLen)) + '…' : this._escHtml(content);
        detail = `<div class="cv2-pi-detail" style="display:none"><pre class="cv2-pi-preview">${truncated}</pre>${needsTrunc ? `<button class="cv2-pi-show-more" data-pi-content-idx="${i}">show full content</button>` : ''}</div>`;
      }

      const countLabel = s.count !== undefined ? ` (${s.count})` : (s.items && s.id === 'messages') ? ` (${s.items.length})` : '';

      finalSections += `
        <div class="cv2-pi-section" data-pi-idx="${i}">
          <div class="cv2-pi-section-header">
            ${expandIcon}
            <span class="cv2-pi-color" style="background:${s.color}"></span>
            <span class="cv2-pi-label">${i + 1} &nbsp;${this._escHtml(s.label)}${countLabel}</span>
            <span class="cv2-pi-tokens">${s.tokens.toLocaleString()} tk</span>
            <span class="cv2-pi-pct">(${pctS}%)</span>
          </div>
          ${detail}
        </div>
      `;
    }

    const backdrop = document.createElement('div');
    backdrop.className = 'cv2-prompt-inspector-backdrop';
    backdrop.innerHTML = `
      <div class="cv2-prompt-inspector">
        <div class="cv2-pi-header">
          <span class="cv2-pi-title">Prompt Inspector</span>
          <span class="cv2-pi-model">${this._escHtml(data.model || '')}</span>
          <button class="cv2-pi-close"><span class="material-icons">close</span></button>
        </div>
        <div class="cv2-pi-bar">${barSegs}</div>
        <div class="cv2-pi-summary">${total.toLocaleString()} / ${max.toLocaleString()} tokens (${pct}%)</div>
        <div class="cv2-pi-sections">${finalSections}</div>
      </div>
    `;

    document.body.appendChild(backdrop);

    // Bind close
    backdrop.querySelector('.cv2-pi-close').addEventListener('click', () => this._closePromptInspector());
    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) this._closePromptInspector();
    });

    // Escape key
    this._piEscHandler = (e) => { if (e.key === 'Escape') this._closePromptInspector(); };
    document.addEventListener('keydown', this._piEscHandler);

    // Expand/collapse sections
    backdrop.querySelectorAll('.cv2-pi-section-header').forEach(header => {
      header.addEventListener('click', () => {
        const detail = header.nextElementSibling;
        if (!detail || !detail.classList.contains('cv2-pi-detail')) return;
        const open = detail.style.display !== 'none';
        detail.style.display = open ? 'none' : 'block';
        const icon = header.querySelector('.cv2-pi-expand-icon');
        if (icon) icon.style.transform = open ? '' : 'rotate(90deg)';
      });
    });

    // "Show full content" toggle buttons
    backdrop.querySelectorAll('.cv2-pi-show-more').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const idx = parseInt(btn.dataset.piContentIdx);
        const s = sections[idx];
        if (!s) return;
        const pre = btn.previousElementSibling;
        const collapsed = btn.textContent.includes('full');
        pre.textContent = collapsed ? s.content : s.content.slice(0, 300) + '…';
        btn.textContent = collapsed ? 'collapse' : 'show full content';
      });
    });
  },

  _closePromptInspector() {
    const el = document.querySelector('.cv2-prompt-inspector-backdrop');
    if (el) el.remove();
    if (this._piEscHandler) {
      document.removeEventListener('keydown', this._piEscHandler);
      this._piEscHandler = null;
    }
  },

  });

  ChatFeatures.register('plusMenu', {
    handleMessage: {
      'prompt_inspector': 'handlePromptInspector',
    },
  });
})();
