/**
 * chat-presets.js — Preset editor (projects + nudges), quick actions, nudge greeting/header
 * Extracted from chat-app.js
 */
(function() {
  Object.assign(window._ChatAppProto, {

  _triggerQuickAction(qa) {
    // Server-side callback — invoke the host app's handler instead of chat
    if (qa.callback) {
      const inputText = this.el.textarea.value.trim();
      this.ws.send({ type: 'action_callback', action_id: qa.id, text: inputText });
      return;
    }

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
            this._pendingImages.map(i => `<img src="${i.dataUri}" alt="Uploaded">`).join('') +
            '</div>';
          userEl.innerHTML = `<div class="cv2-msg-user-bubble">${imgsHtml}</div>`;
          this.el.messages.appendChild(userEl);
          this._wrapAllImages(userEl);

          const images = this._pendingImages.map(i => i.dataUri.split(',')[1]);
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
      this._clearDraft();
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
  },

  _renderNudgeGreeting(nudge) {
    const row = this.el.initialView?.querySelector('.cv2-greeting-row');
    if (!row) return;
    const iconHtml = nudge.icon
      ? `<img class="cv2-nudge-welcome-icon" src="${this._escAttr(nudge.icon)}" alt="">`
      : '<span class="material-icons cv2-nudge-welcome-icon-default">science</span>';
    const descHtml = nudge.description
      ? `<p class="cv2-nudge-welcome-desc">${this._escHtml(nudge.description)}</p>`
      : '';
    const creatorHtml = nudge.creator_name
      ? `<p class="cv2-nudge-welcome-creator">${this.t('chat.nudge_by_creator', { name: this._escHtml(nudge.creator_name) })}</p>`
      : '';
    const allSuggestions = this._parseSuggestions(nudge.suggestions);
    const picked = this._pickRandom(allSuggestions, 4);
    const suggestionsHtml = picked.length > 0
      ? `<div class="cv2-nudge-suggestions">${picked.map(s =>
          `<button class="cv2-nudge-suggestion" data-text="${this._escAttr(s)}">${this._escHtml(s)}</button>`
        ).join('')}</div>`
      : '';
    row.innerHTML = `<div class="cv2-nudge-welcome">
      <div class="cv2-nudge-welcome-row">
        ${iconHtml}
        <div class="cv2-nudge-welcome-text">
          <h2>${this._escHtml(nudge.name)}</h2>
          ${creatorHtml}${descHtml}
        </div>
      </div>
      ${suggestionsHtml}</div>`;
    row.querySelectorAll('.cv2-nudge-suggestion').forEach(btn => {
      btn.addEventListener('click', () => {
        this.el.textarea.value = btn.dataset.text;
        this.sendMessage();
      });
    });
  },

  _renderNudgeChatHeader(nudge) {
    // Remove any existing headers
    this._removeNudgeChatHeader();
    this._removeProjectChatHeader();

    const iconHtml = nudge.icon
      ? `<img src="${this._escAttr(nudge.icon)}" alt="">`
      : '<span class="material-icons">science</span>';
    const descHtml = nudge.description
      ? `<p class="cv2-nudge-header-desc">${this._escHtml(nudge.description)}</p>`
      : '';

    // Heart button HTML (only for MongoDB nudges with uid)
    const hasUid = !!nudge.uid;
    const isFav = hasUid && this.favoriteNudges.some(f => f.uid === nudge.uid);
    const heartHtml = hasUid
      ? `<button class="cv2-nudge-header-fav ${isFav ? 'cv2-fav' : ''}" data-uid="${this._escAttr(nudge.uid)}" title="Favorite"><span class="material-icons" style="font-size:20px">${isFav ? 'favorite' : 'favorite_border'}</span></button>`
      : '';

    // Compact sticky bar — inside the scroll container, sticks to top when scrolled
    const sticky = document.createElement('div');
    sticky.className = 'cv2-nudge-sticky';
    sticky.innerHTML = `
      <div class="cv2-nudge-sticky-icon">${iconHtml}</div>
      <div style="flex:1"></div>
      ${heartHtml}`;
    this.el.messages.prepend(sticky);

    // Expanded header — scrolls with messages, placed after the sticky bar
    const expanded = document.createElement('div');
    expanded.className = 'cv2-nudge-header';
    expanded.innerHTML = `
      <div class="cv2-nudge-header-icon">${iconHtml}</div>
      <div class="cv2-nudge-header-text">
        <h3 class="cv2-nudge-header-name">${this._escHtml(nudge.name)}</h3>
        ${descHtml}
      </div>
      <div style="flex:1"></div>
      ${heartHtml}`;
    // Insert after the sticky bar
    sticky.after(expanded);

    // Bind heart toggle events
    if (hasUid) {
      const bindHeart = (container) => {
        const btn = container.querySelector('.cv2-nudge-header-fav');
        if (!btn) return;
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          this._toggleFavorite(nudge.uid, nudge);
          // Toggle all heart buttons in header
          this.el.messages.querySelectorAll('.cv2-nudge-header-fav').forEach(h => {
            h.classList.toggle('cv2-fav');
            h.querySelector('.material-icons').textContent = h.classList.contains('cv2-fav') ? 'favorite' : 'favorite_border';
          });
        });
      };
      bindHeart(sticky);
      bindHeart(expanded);
    }

    // Toggle sticky bar visibility based on whether expanded header is scrolled out
    this._nudgeHeaderObserver = new IntersectionObserver(
      ([entry]) => {
        sticky.classList.toggle('cv2-visible', !entry.isIntersecting);
      },
      { root: this.el.messages, threshold: 0 }
    );
    this._nudgeHeaderObserver.observe(expanded);
  },

  _renderProjectChatHeader(proj) {
    this._removeProjectChatHeader();

    const iconHtml = proj.icon
      ? `<img src="${this._escAttr(proj.icon)}" alt="">`
      : '<span class="material-icons">folder</span>';

    const bar = document.createElement('div');
    bar.className = 'cv2-project-bar';
    bar.innerHTML = `
      <div class="cv2-project-bar-icon">${iconHtml}</div>
      <span class="cv2-project-bar-name">${this._escHtml(proj.name)}</span>`;

    // Insert before the scrollable messages container
    this.el.messagesWrap.insertBefore(bar, this.el.messages);
  },

  _removeProjectChatHeader() {
    this.el.messagesWrap.querySelector('.cv2-project-bar')?.remove();
  },

  _removeNudgeChatHeader() {
    this.el.messages.querySelectorAll('.cv2-nudge-sticky, .cv2-nudge-header').forEach(el => el.remove());
    if (this._nudgeHeaderObserver) {
      this._nudgeHeaderObserver.disconnect();
      this._nudgeHeaderObserver = null;
    }
  },

  _lockModelForNudge(nudge) {
    if (nudge.model && this.el.modelBtn) {
      this.el.modelBtn.parentElement.style.display = 'none';
    }
  },

  _unlockModel() {
    if (this.el.modelBtn) {
      this.el.modelBtn.parentElement.style.display = '';
    }
  },

  _restoreDefaultGreeting() {
    const row = this.el.initialView?.querySelector('.cv2-greeting-row');
    if (!row) return;
    const mascotHtml = this.config.appMascot
      ? `<img class="cv2-mascot" src="${this._escAttr(this.config.appMascot)}" alt="">`
      : '';
    const greetName = this.userName || 'there';
    row.innerHTML = `${mascotHtml}<h2 id="cv2-greeting">${this._escHtml(this._getGreeting(greetName))}</h2>`;
    this.el.greeting = row.querySelector('#cv2-greeting');
    this._lastGreetName = greetName;
  },

  /**
   * Switch the exclusive view. Tears down the current overlay (if any) before
   * activating the new one. Valid views: 'chat', 'preset', 'explorer'.
   * For 'preset' and 'explorer', pass a closeFn that tears down the overlay.
   */
  _switchView(view, closeFn = null) {
    if (this._closeActiveView && this._activeView !== 'chat') {
      this._closeActiveView();
      this._closeActiveView = null;
    }
    this._activeView = view;
    this._closeActiveView = closeFn;
  },

  _renderQuickActions() {
    const container = this.el.suggestionsMenu;
    if (!container) return;
    const hiddenQAs = new Set(['@sys.image']);
    container.innerHTML = this.quickActions.filter(qa => !hiddenQAs.has(qa.id)).map(qa => `
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
  },

  _renderPresetDialog(preset, type = 'project', { onClose } = {}) {
    const isNew = !preset;
    const title = isNew
      ? (type === 'nudge' ? this.t('chat.new_nudge') : this.t('chat.new_project'))
      : (type === 'nudge' ? this.t('chat.edit_nudge') : this.t('chat.edit_project'));

    const hasNudgeStore = this.config.nudgeCategories && this.config.nudgeCategories.length > 0;
    const data = preset || {
      id: crypto.randomUUID(),
      type,
      name: '',
      icon: null,
      system_prompt: '',
      model: null,
      language: 'auto',
      files: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      ...(type === 'nudge' ? {
        description: '', suggestions: '', capabilities: {},
        uid: crypto.randomUUID(), mode: 'dev', category: '', sub_category: '',
        visibility: [], creator_name: this.fullName || this.userName || '',
        creator_email: this.config.userEmail || '',
      } : {}),
    };

    // Only show web_search and generate_image in capabilities
    const capTools = this.tools.filter(t => t.available && (t.name === 'web_search' || t.name === 'generate_image'));

    const mainEl = document.querySelector('.cv2-main');
    const topbar = mainEl.querySelector('.cv2-topbar');
    const chatWrapper = mainEl.querySelector('.cv2-chat-wrapper');

    // Tear down any other active overlay (explorer, previous preset) via view manager
    this._switchView('preset');

    if (topbar) topbar.style.display = 'none';
    if (chatWrapper) chatWrapper.style.display = 'none';

    const overlay = document.createElement('div');
    overlay.className = 'cv2-preset-fullscreen';
    overlay.innerHTML = `
      <div class="cv2-preset-panel">
      <div class="cv2-preset-header">
        <button class="cv2-preset-back" id="cv2-preset-cancel"><span class="material-icons">arrow_back</span></button>
        <span class="cv2-preset-header-title">${title}</span>
        <div style="flex:1"></div>
        <button class="cv2-dialog-btn cv2-notification-ok" id="cv2-preset-save">${this.t('chat.preset_save') || 'Save'}</button>
      </div>
      <div class="cv2-preset-tabs">
        <button class="cv2-preset-tab cv2-active" data-tab="standard">${this.t('chat.preset_standard')}</button>
        <button class="cv2-preset-tab" data-tab="advanced">${this.t('chat.preset_advanced')}</button>
      </div>
      <div class="cv2-preset-scroll">
        <div class="cv2-preset-form">
          <div class="cv2-preset-tab-content" data-tab="standard">
            <div class="cv2-preset-icon-name-row">
              <div class="cv2-preset-icon-preview ${data.icon ? 'cv2-has-icon' : ''}" id="cv2-preset-icon-preview" title="Click to upload icon">
                ${data.icon ? `<img src="${data.icon}" alt="">` : `<span class="material-icons" style="font-size:28px;color:var(--chat-text-muted)">add</span>`}
              </div>
              <input type="file" id="cv2-preset-icon-input" accept="image/*" style="display:none">
              <input type="text" class="cv2-preset-input" id="cv2-preset-name" value="${this._escAttr(data.name)}" placeholder="${type === 'nudge' ? this.t('chat.nudge_placeholder') : this.t('chat.project_placeholder')}" style="flex:1">
            </div>

            ${type === 'nudge' ? `
              <label class="cv2-preset-label">${this.t('chat.preset_description') || 'Description'}</label>
              <input type="text" class="cv2-preset-input" id="cv2-preset-description" value="${this._escAttr(data.description || '')}" placeholder="${this._escAttr(this.t('chat.preset_description_hint') || 'A brief description...')}">

              ${(() => {
                const editableTeams = this.teams.filter(t => t.role === 'owner' || t.role === 'editor');
                if (editableTeams.length > 0) {
                  const currentTeamId = data.team_id || '';
                  return `
                    <label class="cv2-preset-label">${this.t('chat.preset_owner') || 'Owner'}</label>
                    <select class="cv2-preset-select" id="cv2-preset-owner">
                      <option value="" ${!currentTeamId ? 'selected' : ''}>${this.t('chat.preset_owner_personal') || 'Personal'} (${this._escHtml(this.fullName || this.userName || '')})</option>
                      ${editableTeams.map(t => `<option value="${this._escAttr(t.teamId || t.team_id)}" ${currentTeamId === (t.teamId || t.team_id) ? 'selected' : ''}>${this._escHtml(t.name)}</option>`).join('')}
                    </select>`;
                } else {
                  return `
                    <label class="cv2-preset-label">${this.t('chat.preset_owner') || 'Owner'}</label>
                    <div class="cv2-preset-user-row">
                      ${this.userAvatar
                        ? `<img class="cv2-preset-user-avatar" src="${this._escAttr(this.userAvatar)}" alt="">`
                        : '<span class="material-icons cv2-preset-user-avatar-default">person</span>'}
                      <span class="cv2-preset-user-name">${this._escHtml(this.fullName || this.userName || '')}</span>
                    </div>`;
                }
              })()}

              ${this.config.isAdmin && data.uid ? `
                <label class="cv2-preset-label" style="margin-top:8px">Transfer to (admin)</label>
                <input type="email" class="cv2-preset-input" id="cv2-preset-transfer-email"
                       value="${this._escAttr(data.creator_email || '')}"
                       placeholder="user@example.com">
              ` : ''}
            ` : ''}

            <label class="cv2-preset-label">${this.t('chat.preset_system_prompt') || 'System Prompt'}</label>
            <textarea class="cv2-preset-textarea" id="cv2-preset-prompt" rows="6" placeholder="${this._escAttr(this.t('chat.preset_system_prompt_hint') || 'Instructions for the AI...')}">${this._escHtml(data.system_prompt)}</textarea>

            ${type === 'nudge' ? `
              <label class="cv2-preset-label">${this.t('chat.preset_suggestions')}</label>
              <textarea class="cv2-preset-textarea" id="cv2-preset-suggestions" rows="4" placeholder="${this._escAttr(this.t('chat.preset_suggestions_hint'))}">${this._escHtml(data.suggestions || '')}</textarea>
            ` : ''}

            ${type === 'nudge' && hasNudgeStore ? (() => {
              return `
              <label class="cv2-preset-label">${this.t('chat.nudge_visibility') || 'Visibility'}</label>
              <div class="cv2-vis-autocomplete" id="cv2-vis-autocomplete">
                <div class="cv2-vis-selected" id="cv2-vis-selected"></div>
                <input type="text" class="cv2-vis-input" id="cv2-preset-visibility" placeholder="${this._escAttr(this.t('chat.nudge_visibility_hint') || '*@company.com, user@...')}" autocomplete="off">
                <div class="cv2-vis-dropdown" id="cv2-vis-dropdown" style="display:none"></div>
              </div>
              `;
            })() : ''}

            <div class="cv2-preset-files">
              <label class="cv2-preset-label">${this.t('chat.preset_knowledge')}</label>
              <div class="cv2-preset-file-list" id="cv2-preset-file-list"></div>
              <button class="cv2-preset-add-files-btn" id="cv2-preset-add-files">
                <span class="material-icons" style="font-size:14px">auto_stories</span> ${this.t('chat.preset_add_knowledge')}
              </button>
              <input type="file" id="cv2-preset-file-input" multiple accept=".pdf,.docx,.xlsx,.txt,.md,.csv" style="display:none">
              <div class="cv2-preset-token-bar" id="cv2-preset-token-bar" style="display:none">
                <div class="cv2-preset-token-track"><div class="cv2-preset-token-fill" id="cv2-preset-token-fill"></div></div>
                <span class="cv2-preset-token-label" id="cv2-preset-token-label"></span>
              </div>
            </div>

            ${!isNew ? `<div class="cv2-preset-danger-zone">
              <button class="cv2-preset-delete-link" id="cv2-preset-delete">
                <span class="material-icons" style="font-size:14px">delete</span> ${this.t('chat.delete')}
              </button>
            </div>` : ''}
          </div>

          <div class="cv2-preset-tab-content" data-tab="advanced" style="display:none">
            <div class="cv2-preset-row">
              <div class="cv2-preset-field">
                <label class="cv2-preset-label">${this.t('chat.preset_model') || 'Model'}</label>
                <select class="cv2-preset-select" id="cv2-preset-model">
                  <option value="">${this.t('chat.preset_model_default') || 'Session default'}</option>
                  ${this.models.map(m => `<option value="${this._escAttr(m.model)}" ${data.model === m.model ? 'selected' : ''}>${this._escHtml(m.label || m.model)}</option>`).join('')}
                </select>
              </div>
              <div class="cv2-preset-field">
                <label class="cv2-preset-label">${this.t('chat.preset_language') || 'Language'}</label>
                <select class="cv2-preset-select" id="cv2-preset-language">
                  <option value="auto" ${data.language === 'auto' ? 'selected' : ''}>Auto</option>
                  ${(this.supportedLanguages || []).map(l => `<option value="${this._escAttr(l.code)}" ${data.language === l.code ? 'selected' : ''}>${l.flag || ''} ${this._escHtml(l.label || l.code)}</option>`).join('')}
                </select>
              </div>
            </div>

            ${type === 'nudge' && capTools.length > 0 ? `
              <label class="cv2-preset-label">${this.t('chat.preset_capabilities')}</label>
              <div class="cv2-preset-capabilities" id="cv2-preset-capabilities">
                ${capTools.map(t => {
                  const capVal = data.capabilities && t.name in data.capabilities ? data.capabilities[t.name] : null;
                  const checked = capVal === true;
                  const indeterminate = capVal === null || capVal === undefined;
                  return `<label class="cv2-preset-cap-item">
                    <input type="checkbox" data-cap="${this._escAttr(t.name)}" data-tristate="true" ${checked ? 'checked' : ''} ${indeterminate ? 'data-indeterminate="true"' : ''}>
                    <span class="material-icons">${t.icon || 'build'}</span>
                    ${this._escHtml(t.display_name)}
                    <span class="cv2-cap-state" style="margin-left:auto;font-size:11px;opacity:0.6">${indeterminate ? "user's choice" : ''}</span>
                  </label>`;
                }).join('')}
              </div>
            ` : ''}

            ${this.config.docPlugins && this.config.docPlugins.length > 0 ? (() => {
              const allTypes = this.config.docPlugins;
              const DOC_ICONS = { plotly: 'bar_chart', latex: 'functions', table: 'table_chart', text_doc: 'description', word: 'description', presentation: 'slideshow', powerpoint: 'slideshow', html: 'web', email_draft: 'mail' };
              const DOC_LABELS = { plotly: 'Plotly Charts', latex: 'LaTeX Formulas', table: 'Data Tables', text_doc: 'Text Documents', word: 'Text Documents', presentation: 'Presentations', powerpoint: 'Presentations', html: 'Website', email_draft: 'Email Drafts' };
              const enabledSet = data.doc_plugins ? new Set(data.doc_plugins) : null;
              return `
              <label class="cv2-preset-label">${this.t('chat.preset_doc_plugins') || 'Document Plugins'}</label>
              <div class="cv2-preset-capabilities" id="cv2-preset-doc-plugins">
                ${allTypes.map(dt => {
                  const checked = enabledSet === null || enabledSet.has(dt);
                  return `<label class="cv2-preset-cap-item">
                    <input type="checkbox" data-doc-plugin="${dt}" ${checked ? 'checked' : ''}>
                    <span class="material-icons">${DOC_ICONS[dt] || 'extension'}</span>
                    ${DOC_LABELS[dt] || dt}
                  </label>`;
                }).join('')}
              </div>`;
            })() : ''}

            ${type === 'nudge' && hasNudgeStore ? `
              <div class="cv2-preset-row" style="margin-top:8px">
                <label class="cv2-preset-cap-item">
                  <input type="checkbox" id="cv2-preset-is-mcp" ${(data.files || []).some(f => f.name && (f.name.endsWith('.js') || f.name.endsWith('.mjs'))) ? 'checked' : ''}>
                  <span class="material-icons" style="color:#F7DF1E">extension</span>
                  MCP Server
                  <span style="margin-left:auto;font-size:11px;opacity:0.6">Run JS tools in browser</span>
                </label>
              </div>
            ` : ''}

            ${type === 'nudge' && hasNudgeStore ? `
              <div class="cv2-preset-row">
                <div class="cv2-preset-field">
                  <label class="cv2-preset-label">${this.t('chat.nudge_category') || 'Category'}</label>
                  <select class="cv2-preset-select" id="cv2-preset-category">
                    <option value="">—</option>
                    ${(this.config.nudgeCategories || []).map(c =>
                      `<option value="${this._escAttr(c.key)}" ${data.category === c.key ? 'selected' : ''}>${this._escHtml(c.label)}</option>`
                    ).join('')}
                  </select>
                </div>
                <div class="cv2-preset-field">
                  <label class="cv2-preset-label">${this.t('chat.nudge_sub_category') || 'Sub-category'}</label>
                  <input type="text" class="cv2-preset-input" id="cv2-preset-sub-category" value="${this._escAttr(data.sub_category || '')}" placeholder="${this._escAttr(this.t('chat.nudge_sub_category_hint') || 'e.g. Sales, HR, ...')}">
                </div>
              </div>
              <div class="cv2-preset-row">
                <div class="cv2-preset-field">
                  <label class="cv2-preset-label">${this.t('chat.nudge_dev') || 'Mode'}</label>
                  <span class="cv2-nudge-mode-badge cv2-mode-${data.mode || 'dev'}">${data.mode === 'live' ? (this.t('chat.nudge_live') || 'Live') : (this.t('chat.nudge_dev') || 'Dev')}</span>
                </div>
              </div>
              ${this.config.isAdmin ? `
                <div class="cv2-preset-row" style="margin-top:12px">
                  <label class="cv2-preset-cap-item" id="cv2-master-toggle-row">
                    <input type="checkbox" id="cv2-preset-is-master" ${data.is_master ? 'checked' : ''}>
                    <span class="material-icons" style="color:#D4A017">auto_awesome</span>
                    Master Droplet
                    <span style="margin-left:auto;font-size:11px;opacity:0.6">Always injected for eligible users</span>
                  </label>
                </div>
                <div class="cv2-preset-row" style="margin-top:4px">
                  <label class="cv2-preset-cap-item">
                    <input type="checkbox" id="cv2-preset-auto-discover" ${data.auto_discover ? 'checked' : ''}>
                    <span class="material-icons" style="color:#7B68EE">travel_explore</span>
                    Auto-Discover
                  </label>
                  <div id="cv2-auto-discover-when" style="margin-top:6px;padding-left:32px;${data.auto_discover ? '' : 'display:none'}">
                    <input type="text" class="cv2-preset-input" id="cv2-preset-auto-discover-when" value="${this._escAttr(data.auto_discover_when || '')}" placeholder="e.g. When the user asks about compliance, HR policies, ...">
                  </div>
                </div>
              ` : ''}
              ${!isNew && data.mode === 'dev' && this._canEditNudge(data) ? `
                <div class="cv2-preset-flush-section">
                  <button class="cv2-dialog-btn cv2-notification-ok" id="cv2-preset-flush">
                    <span class="material-icons" style="font-size:16px;margin-right:4px">publish</span>${this.t('chat.nudge_flush') || 'Publish'}
                  </button>
                </div>
              ` : ''}
            ` : ''}
          </div>

          <div class="cv2-preset-tab-content" data-tab="mcp" style="display:none">
            <div class="cv2-mcp-drop-zone" id="cv2-mcp-drop-zone">
              <span class="material-icons">upload_file</span>
              <div>Drop files or folders here</div>
              <div style="display:flex;gap:8px;flex-wrap:wrap">
                <button class="cv2-preset-add-files-btn" id="cv2-preset-add-mcp-files">Add JS Files</button>
                <button class="cv2-preset-add-files-btn" id="cv2-preset-new-mcp-file"><span class="material-icons" style="font-size:14px">add</span> New File</button>
              </div>
              <input type="file" id="cv2-preset-mcp-file-input" multiple accept=".js,.mjs" style="display:none">
            </div>
            <div class="cv2-preset-file-list" id="cv2-preset-mcp-file-list"></div>

            <div id="cv2-mcp-editor-wrap" style="display:none">
              <div class="cv2-mcp-editor-header">
                <button class="cv2-preset-back" id="cv2-mcp-editor-back"><span class="material-icons">arrow_back</span></button>
                <span id="cv2-mcp-editor-filename"></span>
              </div>
              <div id="cv2-mcp-editor-container"></div>
            </div>

            <label class="cv2-preset-label" style="margin-top:12px">Entry Point</label>
            <select class="cv2-preset-select" id="cv2-preset-mcp-entry-point">
              <option value="">— select —</option>
            </select>

            <div id="cv2-preset-mcp-tool-preview" style="display:none;margin-top:12px">
              <label class="cv2-preset-label">Detected Tools</label>
              <div class="cv2-preset-mcp-tools" id="cv2-preset-mcp-tools" style="font-size:12px;opacity:0.8;max-height:150px;overflow-y:auto;padding:8px;background:var(--cv2-bg-secondary);border-radius:6px"></div>
            </div>

            <div id="cv2-mcp-test-section" style="margin-top:16px">
              <button class="cv2-preset-add-files-btn" id="cv2-mcp-run-tests">
                <span class="material-icons" style="font-size:14px">play_arrow</span> Run Tests
              </button>
              <pre class="cv2-mcp-test-log" id="cv2-mcp-test-log" style="display:none"></pre>
            </div>

            <label class="cv2-preset-label" style="margin-top:16px">Data Files</label>
            <div style="font-size:12px;opacity:0.6;margin-bottom:6px">Attached files (CSV, PDF, XLSX, ...) are available to tool handlers via <code>_getDataFile(name)</code></div>
            <div class="cv2-preset-file-list" id="cv2-preset-mcp-data-file-list"></div>
            <button class="cv2-preset-add-files-btn" id="cv2-preset-add-mcp-data-files">
              <span class="material-icons" style="font-size:14px">attach_file</span> Add Data Files
            </button>
            <input type="file" id="cv2-preset-mcp-data-file-input" multiple accept=".csv,.tsv,.json,.xml,.txt,.md,.pdf,.docx,.xlsx,.xls,.yaml,.yml" style="display:none">
          </div>
        </div>
      </div>
      </div>`;

    // Insert into .cv2-main (respects sidebar)
    mainEl.appendChild(overlay);

    // Tri-state capability checkboxes: checked → unchecked → indeterminate (don't care)
    overlay.querySelectorAll('input[data-tristate="true"]').forEach(cb => {
      // Set initial indeterminate state from data attribute
      if (cb.dataset.indeterminate === 'true') {
        cb.indeterminate = true;
        cb.checked = false;
      }
      const stateLabel = cb.closest('.cv2-preset-cap-item')?.querySelector('.cv2-cap-state');
      cb.addEventListener('click', (e) => {
        e.preventDefault();
        if (cb.checked && !cb.indeterminate) {
          // Was checked → go to unchecked
          cb.checked = false;
          cb.indeterminate = false;
          if (stateLabel) stateLabel.textContent = '';
        } else if (!cb.checked && !cb.indeterminate) {
          // Was unchecked → go to indeterminate (don't care)
          cb.indeterminate = true;
          cb.checked = false;
          if (stateLabel) stateLabel.textContent = "user's choice";
        } else {
          // Was indeterminate → go to checked
          cb.indeterminate = false;
          cb.checked = true;
          if (stateLabel) stateLabel.textContent = '';
        }
      });
    });

    const closePreset = () => {
      overlay.remove();
      if (onClose) {
        onClose();
      } else {
        if (topbar) topbar.style.display = '';
        if (chatWrapper) chatWrapper.style.display = '';
        this._activeView = 'chat';
        this._closeActiveView = null;
      }
    };

    // Register as the active overlay teardown
    this._closeActiveView = closePreset;

    // Tab switching
    overlay.querySelectorAll('.cv2-preset-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        overlay.querySelectorAll('.cv2-preset-tab').forEach(t => t.classList.remove('cv2-active'));
        tab.classList.add('cv2-active');
        overlay.querySelectorAll('.cv2-preset-tab-content').forEach(c => {
          c.style.display = c.dataset.tab === tab.dataset.tab ? '' : 'none';
        });
      });
    });

    // Visibility autocomplete
    {
      const visGroups = this.config.visibilityGroups || [];
      const existingVis = data.visibility || [];
      const selectedPatterns = new Set(existingVis);
      const selectedContainer = overlay.querySelector('#cv2-vis-selected');
      const visInput = overlay.querySelector('#cv2-preset-visibility');
      const dropdown = overlay.querySelector('#cv2-vis-dropdown');

      if (visInput) { // only rendered for nudge dialogs with nudge store

      const renderSelected = () => {
        selectedContainer.innerHTML = [...selectedPatterns].map(p => {
          const g = visGroups.find(v => v.pattern === p);
          const label = g ? this._escHtml(g.label) : this._escHtml(p);
          return `<span class="cv2-vis-chip cv2-vis-chip-active" data-pattern="${this._escAttr(p)}"><span class="cv2-vis-chip-label">${label}</span><span class="cv2-vis-chip-x">&times;</span></span>`;
        }).join('');
        selectedContainer.querySelectorAll('.cv2-vis-chip').forEach(chip => {
          chip.addEventListener('click', () => {
            selectedPatterns.delete(chip.dataset.pattern);
            renderSelected();
          });
        });
      };

      const showDropdown = (filter) => {
        const q = (filter || '').toLowerCase();
        const matches = visGroups.filter(g =>
          !selectedPatterns.has(g.pattern) &&
          (g.label.toLowerCase().includes(q) || g.pattern.toLowerCase().includes(q) || (g.location || '').toLowerCase().includes(q))
        );
        if (!matches.length && !q) { dropdown.style.display = 'none'; return; }
        dropdown.innerHTML = matches.map(g =>
          `<div class="cv2-vis-dropdown-item" data-pattern="${this._escAttr(g.pattern)}"><span>${this._escHtml(g.label)}</span><small>${this._escHtml(g.pattern)}</small></div>`
        ).join('');
        if (!matches.length) {
          dropdown.innerHTML = `<div class="cv2-vis-dropdown-empty">${this._escHtml(this.t('chat.nudge_visibility_hint') || 'Type to search or enter *@domain')}</div>`;
        }
        dropdown.style.display = 'block';
        dropdown.querySelectorAll('.cv2-vis-dropdown-item').forEach(item => {
          item.addEventListener('mousedown', (e) => {
            e.preventDefault(); // keep focus on input
            selectedPatterns.add(item.dataset.pattern);
            visInput.value = '';
            dropdown.style.display = 'none';
            renderSelected();
          });
        });
      };

      visInput.addEventListener('focus', () => showDropdown(visInput.value));
      visInput.addEventListener('input', () => showDropdown(visInput.value));
      visInput.addEventListener('blur', () => { setTimeout(() => dropdown.style.display = 'none', 150); });
      // Enter on custom pattern (e.g. *@custom.com)
      visInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          const val = visInput.value.trim();
          if (val && !selectedPatterns.has(val)) {
            selectedPatterns.add(val);
            visInput.value = '';
            dropdown.style.display = 'none';
            renderSelected();
          }
        }
      });

      renderSelected();
      } // end if (visInput)
    }

    // Master toggle: only enabled when a team is selected
    {
      const masterRow = overlay.querySelector('#cv2-master-toggle-row');
      const masterCb = overlay.querySelector('#cv2-preset-is-master');
      const ownerSelect = overlay.querySelector('#cv2-preset-owner');
      if (masterRow && masterCb && ownerSelect) {
        const syncMasterVisibility = () => {
          const hasTeam = !!ownerSelect.value;
          masterRow.style.opacity = hasTeam ? '1' : '0.4';
          masterRow.style.pointerEvents = hasTeam ? '' : 'none';
          if (!hasTeam) masterCb.checked = false;
        };
        ownerSelect.addEventListener('change', syncMasterVisibility);
        syncMasterVisibility();
      }
    }

    // Auto-discover: toggle "when" textbox visibility
    {
      const adCb = overlay.querySelector('#cv2-preset-auto-discover');
      const adWhen = overlay.querySelector('#cv2-auto-discover-when');
      if (adCb && adWhen) {
        adCb.addEventListener('change', () => {
          adWhen.style.display = adCb.checked ? '' : 'none';
        });
      }
    }

    // State for icon
    let currentIcon = data.icon || null;

    // Icon upload — click the round preview to upload or paste
    const iconInput = overlay.querySelector('#cv2-preset-icon-input');
    const iconPreview = overlay.querySelector('#cv2-preset-icon-preview');
    iconPreview.style.cursor = 'pointer';
    iconPreview.addEventListener('click', () => iconInput.click());
    iconInput.addEventListener('change', async () => {
      const file = iconInput.files[0];
      if (!file) return;
      try {
        currentIcon = await this._resizeIcon(file);
        iconPreview.innerHTML = `<img src="${currentIcon}" alt="">`;
        iconPreview.classList.add('cv2-has-icon');
      } catch (err) {
        this._showToast('Failed to load image', 'negative');
      }
    });
    // Paste image from clipboard onto icon
    overlay.addEventListener('paste', async (e) => {
      const items = Array.from(e.clipboardData?.items || []);
      const imgItem = items.find(i => i.type.startsWith('image/'));
      if (!imgItem) return;
      e.preventDefault();
      const file = imgItem.getAsFile();
      if (!file) return;
      try {
        currentIcon = await this._resizeIcon(file);
        iconPreview.innerHTML = `<img src="${currentIcon}" alt="">`;
        iconPreview.classList.add('cv2-has-icon');
      } catch (err) {
        this._showToast('Failed to load image', 'negative');
      }
    });

    // ── File management ─────────────────────────────
    const MAX_TOKEN_BUDGET = ChatApp.MAX_TOKEN_BUDGET;
    const _isJsFileEntry = (f) => f.name && (f.name.endsWith('.js') || f.name.endsWith('.mjs'));
    const currentFiles = [...(data.files || [])].filter(f => !_isJsFileEntry(f));
    const fileListEl = overlay.querySelector('#cv2-preset-file-list');
    const fileInput = overlay.querySelector('#cv2-preset-file-input');
    const tokenBar = overlay.querySelector('#cv2-preset-token-bar');
    const tokenFill = overlay.querySelector('#cv2-preset-token-fill');
    const tokenLabel = overlay.querySelector('#cv2-preset-token-label');

    const TEXT_EXTENSIONS = new Set(['txt', 'md', 'csv']);

    const estimateTokens = (f) => {
      if (f.text_content) return Math.ceil(f.text_content.length / 4);
      return Math.ceil(f.size * 0.8 / 4);
    };

    const formatSize = (bytes) => {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const formatTokens = (t) => t >= 1000 ? `${(t / 1000).toFixed(1)}K` : `${t}`;

    const renderFileList = () => {
      const totalTokens = currentFiles.reduce((sum, f) => sum + estimateTokens(f), 0);

      fileListEl.innerHTML = currentFiles.map((f, i) => {
        const broken = !f.content && !f.text_content;
        return `
        <div class="cv2-preset-file-item${broken ? ' cv2-preset-file-broken' : ''}" data-idx="${i}">
          <span class="material-icons cv2-preset-file-icon">${broken ? 'warning' : 'description'}</span>
          <span class="cv2-preset-file-name" title="${this._escAttr(f.name)}${broken ? ' — re-attach this file' : ''}">${this._escHtml(f.name)}</span>
          <span class="cv2-preset-file-size">${formatSize(f.size)}</span>
          <span class="cv2-preset-file-tokens">~${formatTokens(estimateTokens(f))} tok</span>
          <button class="cv2-preset-file-remove" data-idx="${i}"><span class="material-icons" style="font-size:16px">close</span></button>
        </div>`;
      }).join('');

      // Remove handlers
      fileListEl.querySelectorAll('.cv2-preset-file-remove').forEach(btn => {
        btn.addEventListener('click', () => {
          currentFiles.splice(parseInt(btn.dataset.idx), 1);
          renderFileList();
        });
      });

      // Token budget bar
      if (currentFiles.length > 0) {
        tokenBar.style.display = '';
        const pct = Math.min(100, totalTokens / MAX_TOKEN_BUDGET * 100);
        tokenFill.style.width = `${pct}%`;
        tokenFill.className = 'cv2-preset-token-fill' +
          (pct >= 100 ? ' cv2-over' : pct >= 75 ? ' cv2-warn' : '');
        tokenLabel.textContent = `~${formatTokens(totalTokens)} / ${formatTokens(MAX_TOKEN_BUDGET)} tokens`;
      } else {
        tokenBar.style.display = 'none';
      }
    };

    overlay.querySelector('#cv2-preset-add-files').addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
      const newFiles = Array.from(fileInput.files);
      // Central validation — check limits against existing preset + chat files
      const error = this._validateNewFiles(newFiles, currentFiles);
      if (error) {
        this._showToast(error, 'negative');
        fileInput.value = '';
        return;
      }

      for (const file of newFiles) {
        // Peek for oversized spreadsheets etc.
        if (!(await this._peekFile(file))) continue;

        const ext = file.name.split('.').pop().toLowerCase();
        const isText = TEXT_EXTENSIONS.has(ext);

        // Read as base64
        const b64 = await new Promise((res) => {
          const r = new FileReader();
          r.onload = () => res(r.result);
          r.readAsDataURL(file);
        });

        // Client-side text extraction for plain-text files;
        // binary files (PDF, DOCX, XLSX) are extracted server-side
        // when the chat starts (via _inject_preset_files).
        let textContent = '';
        if (isText) {
          textContent = await new Promise((res) => {
            const r = new FileReader();
            r.onload = () => res(r.result);
            r.readAsText(file);
          });
        }

        currentFiles.push({
          file_id: crypto.randomUUID().replace(/-/g, '').slice(0, 12),
          name: file.name,
          size: file.size,
          mime_type: file.type || 'application/octet-stream',
          content: b64,
          text_content: textContent,
        });
      }
      fileInput.value = '';
      renderFileList();
    });

    // Initial render (for editing existing presets with files)
    renderFileList();

    // ── MCP file management ─────────────────────────────
    const allMcpFiles = (data.files || []);
    // Decode text_content from base64 content for JS files loaded from DB
    for (const f of allMcpFiles) {
      if (!f.text_content && f.content && (f.name || '').match(/\.(js|mjs)$/)) {
        try {
          const b64 = f.content.includes(',') ? f.content.split(',')[1] : f.content;
          f.text_content = atob(b64);
        } catch (e) { /* ignore decode errors */ }
      }
    }
    const mcpFiles = [...allMcpFiles].filter(f =>
      f.name && (f.name.endsWith('.js') || f.name.endsWith('.mjs'))
    );
    // MCP data files: managed via the MCP tab (attached for Worker access, not knowledge injection)
    // Start empty — files added through the MCP "Data Files" upload go here
    const mcpDataFiles = [];
    const _hasJsFiles = (data.files || []).some(f => f.name && (f.name.endsWith('.js') || f.name.endsWith('.mjs')));
    let currentNudgeType = _hasJsFiles ? 'mcp' : 'knowledge';
    const mcpFileListEl = overlay.querySelector('#cv2-preset-mcp-file-list');
    const mcpFileInput = overlay.querySelector('#cv2-preset-mcp-file-input');
    const mcpEntrySelect = overlay.querySelector('#cv2-preset-mcp-entry-point');
    const mcpToolPreview = overlay.querySelector('#cv2-preset-mcp-tool-preview');
    const mcpToolsEl = overlay.querySelector('#cv2-preset-mcp-tools');

    const renderMcpFileList = () => {
      if (!mcpFileListEl) return;
      mcpFileListEl.innerHTML = mcpFiles.map((f, i) => `
        <div class="cv2-preset-file-item cv2-mcp-file-clickable" data-idx="${i}">
          <span class="material-icons cv2-preset-file-icon" style="color:#F7DF1E">javascript</span>
          <span class="cv2-preset-file-name" title="${this._escAttr(f.name)}">${this._escHtml(f.name)}</span>
          <span class="cv2-preset-file-size">${formatSize(f.size)}</span>
          <button class="cv2-mcp-file-edit" data-idx="${i}" title="Edit"><span class="material-icons" style="font-size:16px">edit</span></button>
          <button class="cv2-preset-file-remove" data-idx="${i}"><span class="material-icons" style="font-size:16px">close</span></button>
        </div>
      `).join('');

      mcpFileListEl.querySelectorAll('.cv2-mcp-file-edit').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          _openEditor(parseInt(btn.dataset.idx));
        });
      });

      mcpFileListEl.querySelectorAll('.cv2-preset-file-name').forEach(el => {
        el.addEventListener('click', () => {
          _openEditor(parseInt(el.closest('.cv2-preset-file-item').dataset.idx));
        });
      });

      mcpFileListEl.querySelectorAll('.cv2-preset-file-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const idx = parseInt(btn.dataset.idx);
          const fileName = mcpFiles[idx]?.name || 'this file';
          const dlg = document.createElement('div');
          dlg.className = 'cv2-dialog-overlay';
          dlg.innerHTML = `
            <div class="cv2-dialog">
              <div class="cv2-dialog-title">Delete file</div>
              <div class="cv2-dialog-body">Delete <strong>${this._escHtml(fileName)}</strong>?</div>
              <div class="cv2-dialog-actions">
                <button class="cv2-dialog-btn cv2-dialog-cancel">Cancel</button>
                <button class="cv2-dialog-btn cv2-dialog-confirm">Delete</button>
              </div>
            </div>`;
          document.getElementById('chat-app').appendChild(dlg);
          dlg.querySelector('.cv2-dialog-cancel').addEventListener('click', () => dlg.remove());
          dlg.addEventListener('click', (ev) => { if (ev.target === dlg) dlg.remove(); });
          dlg.querySelector('.cv2-dialog-confirm').addEventListener('click', () => {
            dlg.remove();
            if (_cmFileIdx === idx) _closeEditor();
            else if (_cmFileIdx > idx) _cmFileIdx--;
            mcpFiles.splice(idx, 1);
            renderMcpFileList();
            updateMcpEntrySelect();
            extractMcpTools();
          });
        });
      });
    };

    const updateMcpEntrySelect = () => {
      if (!mcpEntrySelect) return;
      const currentEntry = mcpEntrySelect.value || data.mcp_entry_point || 'index.js';
      mcpEntrySelect.innerHTML = mcpFiles.map(f =>
        `<option value="${this._escAttr(f.name)}" ${f.name === currentEntry ? 'selected' : ''}>${this._escHtml(f.name)}</option>`
      ).join('');
      if (mcpFiles.length === 0) {
        mcpEntrySelect.innerHTML = '<option value="">— no files —</option>';
      }
    };

    const extractMcpTools = () => {
      if (!mcpToolPreview || !mcpToolsEl) return;
      // Simple regex extraction of tool names from entry point
      const entryName = mcpEntrySelect?.value || '';
      const entryFile = mcpFiles.find(f => f.name === entryName);
      if (!entryFile || !entryFile.text_content) {
        mcpToolPreview.style.display = 'none';
        return;
      }
      const toolNames = [];
      // Match patterns like: server.tool("name", ...)  or  name: "tool_name"
      const src = entryFile.text_content;
      const re1 = /\.tool\s*\(\s*["']([^"']+)["']/g;
      const re2 = /name:\s*["']([^"']+)["']/g;
      let m;
      while ((m = re1.exec(src)) !== null) toolNames.push(m[1]);
      if (toolNames.length === 0) {
        while ((m = re2.exec(src)) !== null) toolNames.push(m[1]);
      }
      // Also check non-entry files for tool patterns
      for (const f of mcpFiles) {
        if (f.name === entryName) continue;
        if (!f.text_content) continue;
        const re3 = /\.tool\s*\(\s*["']([^"']+)["']/g;
        while ((m = re3.exec(f.text_content)) !== null) {
          if (!toolNames.includes(m[1])) toolNames.push(m[1]);
        }
      }

      if (toolNames.length > 0) {
        mcpToolPreview.style.display = '';
        mcpToolsEl.innerHTML = `<strong>${toolNames.length} tools detected:</strong><br>` +
          toolNames.map(n => `<span class="cv2-mcp-tool-chip">${this._escHtml(n)}</span>`).join('');
      } else {
        mcpToolPreview.style.display = 'none';
      }
    };

    renderMcpFileList();
    updateMcpEntrySelect();
    extractMcpTools();

    // ── MCP checkbox → show/hide MCP tab ─────────────────
    const mcpCheckbox = overlay.querySelector('#cv2-preset-is-mcp');
    const tabsBar = overlay.querySelector('.cv2-preset-tabs');
    let _cmInstance = null;
    let _cmFileIdx = -1;

    const _syncEditorBack = () => {
      if (_cmInstance && _cmFileIdx >= 0 && mcpFiles[_cmFileIdx]) {
        const newText = _cmInstance.getValue();
        mcpFiles[_cmFileIdx].text_content = newText;
        const blob = new Blob([newText], { type: 'application/javascript' });
        mcpFiles[_cmFileIdx].size = blob.size;
        // Re-encode as base64 data URL
        const reader = new FileReader();
        reader.onload = () => { mcpFiles[_cmFileIdx].content = reader.result; };
        reader.readAsDataURL(blob);
      }
    };

    const _showMcpTab = () => {
      if (!tabsBar.querySelector('[data-tab="mcp"]')) {
        const mcpTabBtn = document.createElement('button');
        mcpTabBtn.className = 'cv2-preset-tab';
        mcpTabBtn.dataset.tab = 'mcp';
        mcpTabBtn.innerHTML = '<span class="material-icons" style="font-size:14px;vertical-align:text-bottom;margin-right:2px">extension</span>MCP';
        tabsBar.appendChild(mcpTabBtn);
        mcpTabBtn.addEventListener('click', () => {
          overlay.querySelectorAll('.cv2-preset-tab').forEach(t => t.classList.remove('cv2-active'));
          mcpTabBtn.classList.add('cv2-active');
          overlay.querySelectorAll('.cv2-preset-tab-content').forEach(c => {
            c.style.display = c.dataset.tab === 'mcp' ? '' : 'none';
          });
        });
      }
      currentNudgeType = 'mcp';
    };

    const _hideMcpTab = () => {
      const mcpTabBtn = tabsBar.querySelector('[data-tab="mcp"]');
      if (mcpTabBtn) {
        // If MCP tab is active, switch to standard
        if (mcpTabBtn.classList.contains('cv2-active')) {
          const stdBtn = tabsBar.querySelector('[data-tab="standard"]');
          if (stdBtn) {
            stdBtn.click();
          }
        }
        mcpTabBtn.remove();
      }
      // Close editor if open
      _syncEditorBack();
      _cmInstance = null;
      _cmFileIdx = -1;
      currentNudgeType = 'knowledge';
    };

    if (mcpCheckbox) {
      // Initial state: if nudge has JS files, show the MCP tab
      if (_hasJsFiles) {
        _showMcpTab();
      }
      mcpCheckbox.addEventListener('change', () => {
        if (mcpCheckbox.checked) {
          _showMcpTab();
        } else {
          _hideMcpTab();
        }
      });
    }

    // ── MCP file helpers ──────────────────────────────────
    const _readFileAsText = (file) => new Promise((res) => {
      const r = new FileReader(); r.onload = () => res(r.result); r.readAsText(file);
    });
    const _readFileAsDataURL = (file) => new Promise((res) => {
      const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(file);
    });

    const _dataFileExts = new Set(['csv','tsv','json','xml','txt','md','pdf','docx','xlsx','xls','yaml','yml']);
    const _textExts = new Set(['csv','tsv','json','xml','txt','md','yaml','yml']);

    const addOrReplaceMcpFile = async (file) => {
      const textContent = await _readFileAsText(file);
      const b64 = await _readFileAsDataURL(file);
      const entry = {
        file_id: crypto.randomUUID().replace(/-/g, '').slice(0, 12),
        name: file.name,
        size: file.size,
        mime_type: 'application/javascript',
        content: b64,
        text_content: textContent,
      };
      const idx = mcpFiles.findIndex(f => f.name === file.name);
      if (idx >= 0) mcpFiles[idx] = entry;
      else mcpFiles.push(entry);
      renderMcpFileList();
      updateMcpEntrySelect();
      extractMcpTools();
    };

    const addOrReplaceDataFile = async (file) => {
      const b64 = await _readFileAsDataURL(file);
      const ext = file.name.split('.').pop().toLowerCase();
      let textContent = '';
      if (_textExts.has(ext)) textContent = await _readFileAsText(file);
      const entry = {
        file_id: crypto.randomUUID().replace(/-/g, '').slice(0, 12),
        name: file.name,
        size: file.size,
        mime_type: file.type || 'application/octet-stream',
        content: b64,
        text_content: textContent,
      };
      const idx = mcpDataFiles.findIndex(f => f.name === file.name);
      if (idx >= 0) mcpDataFiles[idx] = entry;
      else mcpDataFiles.push(entry);
      renderMcpDataFileList();
    };

    // ── Drag & drop (files + folders) ──────────────────────
    const _isJsFile = (name) => /\.m?js$/.test(name);
    const _isDataFile = (name) => _dataFileExts.has(name.split('.').pop().toLowerCase());

    const _collectDropFiles = async (entry, prefix) => {
      const results = [];
      if (entry.isFile) {
        if (_isJsFile(entry.name) || _isDataFile(entry.name)) {
          const file = await new Promise((res, rej) => entry.file(res, rej));
          results.push({ file, name: prefix ? prefix + '/' + entry.name : entry.name });
        }
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        let entries = [];
        let batch;
        do {
          batch = await new Promise((res, rej) => reader.readEntries(res, rej));
          entries = entries.concat(batch);
        } while (batch.length > 0);
        for (const child of entries) {
          results.push(...await _collectDropFiles(child, prefix ? prefix + '/' + entry.name : entry.name));
        }
      }
      return results;
    };

    const _addDroppedFile = async (file) => {
      if (_isJsFile(file.name)) await addOrReplaceMcpFile(file);
      else if (_isDataFile(file.name)) await addOrReplaceDataFile(file);
    };

    const dropZone = overlay.querySelector('#cv2-mcp-drop-zone');
    if (dropZone) {
      ['dragenter', 'dragover'].forEach(evt =>
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('cv2-mcp-drag-over'); })
      );
      ['dragleave', 'drop'].forEach(evt =>
        dropZone.addEventListener(evt, () => dropZone.classList.remove('cv2-mcp-drag-over'))
      );
      dropZone.addEventListener('drop', async (ev) => {
        ev.preventDefault();
        const items = ev.dataTransfer.items;
        if (items && items.length > 0 && items[0].webkitGetAsEntry) {
          const collected = [];
          for (const item of items) {
            const entry = item.webkitGetAsEntry();
            if (entry) collected.push(...await _collectDropFiles(entry, ''));
          }
          for (const { file, name } of collected) {
            const flatName = name.includes('/') ? name.split('/').pop() : name;
            Object.defineProperty(file, 'name', { value: flatName });
            await _addDroppedFile(file);
          }
        } else {
          for (const file of ev.dataTransfer.files) await _addDroppedFile(file);
        }
      });
    }

    // ── MCP file upload (button) ──────────────────────────
    if (mcpFileInput) {
      overlay.querySelector('#cv2-preset-add-mcp-files')?.addEventListener('click', () => mcpFileInput.click());
      mcpFileInput.addEventListener('change', async () => {
        for (const file of Array.from(mcpFileInput.files)) {
          await addOrReplaceMcpFile(file);
        }
        mcpFileInput.value = '';
      });
    }

    // ── MCP new file (name dialog) ───────────────────────
    overlay.querySelector('#cv2-preset-new-mcp-file')?.addEventListener('click', () => {
      const dlg = document.createElement('div');
      dlg.className = 'cv2-dialog-overlay';
      dlg.innerHTML = `
        <div class="cv2-dialog">
          <div class="cv2-dialog-title">New JS file</div>
          <div class="cv2-dialog-body" style="margin-bottom:12px">
            <input class="cv2-preset-input" id="cv2-new-mcp-filename" placeholder="example.js" autocomplete="off" spellcheck="false" style="margin-top:4px">
          </div>
          <div class="cv2-dialog-actions">
            <button class="cv2-dialog-btn cv2-dialog-cancel">Cancel</button>
            <button class="cv2-dialog-btn cv2-notification-ok cv2-dlg-create">Create</button>
          </div>
        </div>`;
      document.getElementById('chat-app').appendChild(dlg);
      const input = dlg.querySelector('#cv2-new-mcp-filename');
      input.focus();
      const doCreate = () => {
        let name = input.value.trim();
        if (!name) return;
        if (!/\.m?js$/.test(name)) name += '.js';
        if (mcpFiles.some(f => f.name === name)) { input.style.borderColor = '#ef4444'; return; }
        dlg.remove();
        const entry = {
          file_id: crypto.randomUUID().replace(/-/g, '').slice(0, 12),
          name,
          size: 0,
          mime_type: 'application/javascript',
          content: 'data:application/javascript;base64,',
          text_content: '',
        };
        mcpFiles.push(entry);
        renderMcpFileList();
        updateMcpEntrySelect();
        extractMcpTools();
        // Open the new file in the editor immediately
        _openEditor(mcpFiles.length - 1);
      };
      dlg.querySelector('.cv2-dialog-cancel').addEventListener('click', () => dlg.remove());
      dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.remove(); });
      dlg.querySelector('.cv2-dlg-create').addEventListener('click', doCreate);
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') doCreate(); });
    });

    // ── Click file → CodeMirror editor ────────────────────
    const editorWrap = overlay.querySelector('#cv2-mcp-editor-wrap');
    const editorContainer = overlay.querySelector('#cv2-mcp-editor-container');
    const editorFilename = overlay.querySelector('#cv2-mcp-editor-filename');
    const editorBack = overlay.querySelector('#cv2-mcp-editor-back');

    const _openEditor = (idx) => {
      if (!editorWrap || !editorContainer || idx < 0 || !mcpFiles[idx]) return;
      // Sync any previous editor
      _syncEditorBack();

      // Hide file list + drop zone, show editor
      const mcpTabContent = overlay.querySelector('[data-tab="mcp"]');
      if (mcpTabContent) {
        mcpTabContent.querySelectorAll('.cv2-mcp-drop-zone, .cv2-preset-file-list, .cv2-preset-label, .cv2-preset-select, .cv2-preset-add-files-btn, #cv2-preset-mcp-tool-preview, #cv2-preset-mcp-data-file-list, #cv2-mcp-test-section').forEach(el => {
          el.dataset.prevDisplay = el.style.display;
          el.style.display = 'none';
        });
        // Also hide helper text divs
        mcpTabContent.querySelectorAll('div[style*="font-size:12px"]').forEach(el => {
          el.dataset.prevDisplay = el.style.display;
          el.style.display = 'none';
        });
      }

      editorWrap.style.display = '';
      editorFilename.textContent = mcpFiles[idx].name;
      editorContainer.innerHTML = '';
      _cmFileIdx = idx;

      _cmInstance = CodeMirror(editorContainer, {
        value: mcpFiles[idx].text_content || '',
        mode: 'javascript',
        lineNumbers: true,
        matchBrackets: true,
        autoCloseBrackets: true,
        gutters: ['CodeMirror-lint-markers'],
        lint: { esversion: 2021, asi: true },
        tabSize: 2,
        theme: 'default',
      });
      // Force refresh after render
      setTimeout(() => _cmInstance.refresh(), 50);
    };

    const _closeEditor = () => {
      _syncEditorBack();
      if (_cmInstance) {
        _cmInstance.toTextArea ? _cmInstance.toTextArea() : null;
        _cmInstance = null;
        _cmFileIdx = -1;
      }
      if (editorWrap) editorWrap.style.display = 'none';
      if (editorContainer) editorContainer.innerHTML = '';

      // Restore hidden elements
      const mcpTabContent = overlay.querySelector('[data-tab="mcp"]');
      if (mcpTabContent) {
        mcpTabContent.querySelectorAll('[data-prev-display]').forEach(el => {
          el.style.display = el.dataset.prevDisplay || '';
          delete el.dataset.prevDisplay;
        });
      }

      renderMcpFileList();
      updateMcpEntrySelect();
      extractMcpTools();
    };

    if (editorBack) {
      editorBack.addEventListener('click', _closeEditor);
    }

    if (mcpEntrySelect) {
      mcpEntrySelect.addEventListener('change', extractMcpTools);
    }

    // ── MCP test runner ──────────────────────────────────
    const runTestsBtn = overlay.querySelector('#cv2-mcp-run-tests');
    const testLog = overlay.querySelector('#cv2-mcp-test-log');

    if (runTestsBtn && testLog) {
      runTestsBtn.addEventListener('click', async () => {
        _syncEditorBack();

        testLog.style.display = '';
        testLog.textContent = '';
        const log = (msg) => { testLog.textContent += msg + '\n'; testLog.scrollTop = testLog.scrollHeight; };

        // Build files dict
        const filesDict = {};
        for (const f of mcpFiles) filesDict[f.name] = f.text_content;
        const entryPoint = mcpEntrySelect?.value || 'index.js';
        if (!filesDict[entryPoint]) { log('[error] No entry point: ' + entryPoint); return; }

        // Build data files with content_base64
        const dataFiles = mcpDataFiles.map(f => ({
          name: f.name, mime_type: f.mime_type, size: f.size,
          content_base64: f.content ? f.content.split(',')[1] || '' : '',
          text_content: f.text_content || '',
        }));

        // Build worker code using app method
        let workerCode;
        try {
          workerCode = this._buildWorkerCode(filesDict, entryPoint, dataFiles);
        } catch (e) { log('[error] Build failed: ' + e.message); return; }

        // Spawn worker
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        const worker = new Worker(blobUrl);

        const callTool = (name, args) => new Promise((resolve) => {
          const callId = crypto.randomUUID();
          const handler = (e) => {
            if (e.data.callId === callId) {
              worker.removeEventListener('message', handler);
              resolve(e.data);
            }
          };
          worker.addEventListener('message', handler);
          worker.postMessage({ type: 'tool_call', callId, name, arguments: args || {} });
          setTimeout(() => { worker.removeEventListener('message', handler); resolve({ error: 'timeout (10s)' }); }, 10000);
        });

        // Init
        const tools = await new Promise((resolve) => {
          worker.onmessage = (e) => {
            if (e.data.type === 'mcp_ready') resolve(e.data.tools);
          };
          worker.onerror = (e) => { log('[error] Worker: ' + e.message); resolve(null); };
          worker.postMessage({ type: 'init' });
          setTimeout(() => resolve(null), 5000);
        });

        if (!tools) { log('[error] Worker init timeout'); worker.terminate(); URL.revokeObjectURL(blobUrl); return; }
        log('[init] ' + tools.length + ' tools: ' + tools.map(t => t.name).join(', '));

        // Ask worker for user-defined tests (self._tests in a tests.js file)
        worker.onmessage = null;
        const userTests = await new Promise((resolve) => {
          const h = (e) => {
            if (e.data.type === '_tests') { worker.removeEventListener('message', h); resolve(e.data.tests); }
          };
          worker.addEventListener('message', h);
          worker.postMessage({ type: '_get_tests' });
          setTimeout(() => { worker.removeEventListener('message', h); resolve(null); }, 2000);
        });

        // Build test list: user-defined tests, or fallback to one smoke-test per tool
        let testCases;
        if (userTests && userTests.length > 0) {
          testCases = userTests;
          log('[tests] ' + testCases.length + ' test cases from tests.js');
        } else {
          // Fallback: call each tool with required args derived from schema
          testCases = tools.map(t => {
            const args = {};
            const schema = t.inputSchema;
            if (schema && schema.properties) {
              for (const [key, prop] of Object.entries(schema.properties)) {
                if (!(schema.required || []).includes(key)) continue;
                if (prop.enum && prop.enum.length > 0) args[key] = prop.enum[0];
                else if (prop.type === 'number') args[key] = 1;
                else if (prop.type === 'string') args[key] = 'test';
                else if (prop.type === 'boolean') args[key] = true;
              }
            }
            return { name: t.name + ' (smoke)', tool: t.name, args };
          });
          log('[tests] No tests.js found, running smoke tests for ' + tools.length + ' tools');
        }

        // Run tests
        let passed = 0, failed = 0;
        worker.onmessage = null;
        for (const tc of testCases) {
          const label = tc.name || tc.tool;
          const res = await callTool(tc.tool, tc.args || {});

          // Check assertions
          let assertFail = null;
          if (!res.error && tc.expect) {
            const text = res.result || '';
            if (tc.expect.contains && !text.includes(tc.expect.contains)) {
              assertFail = 'expected result to contain "' + tc.expect.contains + '"';
            }
            if (tc.expect.notContains && text.includes(tc.expect.notContains)) {
              assertFail = 'expected result NOT to contain "' + tc.expect.notContains + '"';
            }
            if (tc.expect.isError === true && !res.error) {
              assertFail = 'expected an error but got OK';
            }
          }
          if (res.error && tc.expect && tc.expect.isError === true) {
            // Expected error — this is a pass
            passed++;
            log('[' + label + '] OK (expected error): ' + res.error);
          } else if (res.error) {
            failed++;
            log('[' + label + '] FAIL: ' + res.error);
          } else if (assertFail) {
            failed++;
            log('[' + label + '] FAIL: ' + assertFail);
          } else {
            passed++;
            const snippet = (res.result || '').slice(0, 120);
            log('[' + label + '] OK: ' + snippet);
          }
        }

        log('---');
        log(passed + '/' + (passed + failed) + ' passed');

        worker.terminate();
        URL.revokeObjectURL(blobUrl);
      });
    }

    // ── MCP data file management ─────────────────────────
    const mcpDataFileListEl = overlay.querySelector('#cv2-preset-mcp-data-file-list');
    const mcpDataFileInput = overlay.querySelector('#cv2-preset-mcp-data-file-input');

    const renderMcpDataFileList = () => {
      if (!mcpDataFileListEl) return;
      const iconMap = { pdf: 'picture_as_pdf', csv: 'table_chart', xlsx: 'table_chart', xls: 'table_chart', json: 'data_object', xml: 'code', txt: 'description', md: 'description' };
      mcpDataFileListEl.innerHTML = mcpDataFiles.map((f, i) => {
        const ext = (f.name || '').split('.').pop().toLowerCase();
        const icon = iconMap[ext] || 'attach_file';
        return `
          <div class="cv2-preset-file-item" data-idx="${i}">
            <span class="material-icons cv2-preset-file-icon">${icon}</span>
            <span class="cv2-preset-file-name" title="${this._escAttr(f.name)}">${this._escHtml(f.name)}</span>
            <span class="cv2-preset-file-size">${formatSize(f.size)}</span>
            <button class="cv2-preset-file-remove" data-idx="${i}"><span class="material-icons" style="font-size:16px">close</span></button>
          </div>`;
      }).join('');
      mcpDataFileListEl.querySelectorAll('.cv2-preset-file-remove').forEach(btn => {
        btn.addEventListener('click', () => {
          mcpDataFiles.splice(parseInt(btn.dataset.idx), 1);
          renderMcpDataFileList();
        });
      });
    };
    renderMcpDataFileList();

    if (mcpDataFileInput) {
      overlay.querySelector('#cv2-preset-add-mcp-data-files')?.addEventListener('click', () => mcpDataFileInput.click());
      mcpDataFileInput.addEventListener('change', async () => {
        for (const file of Array.from(mcpDataFileInput.files)) {
          await addOrReplaceDataFile(file);
        }
        mcpDataFileInput.value = '';
      });
    }

    // Cancel / back
    overlay.querySelector('#cv2-preset-cancel').addEventListener('click', () => closePreset());

    // Delete
    overlay.querySelector('#cv2-preset-delete')?.addEventListener('click', () => {
      const dlg = document.createElement('div');
      dlg.className = 'cv2-dialog-overlay';
      dlg.innerHTML = `
        <div class="cv2-dialog">
          <div class="cv2-dialog-body">${this._escHtml(this.t('chat.delete_confirm', { name: data.name }))}</div>
          <div class="cv2-dialog-actions">
            <button class="cv2-dialog-btn cv2-dialog-cancel">${this.t('chat.cancel')}</button>
            <button class="cv2-dialog-btn cv2-dialog-confirm">${this.t('chat.delete')}</button>
          </div>
        </div>`;
      document.getElementById('chat-app').appendChild(dlg);
      dlg.querySelector('.cv2-dialog-cancel').addEventListener('click', () => dlg.remove());
      dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.remove(); });
      dlg.querySelector('.cv2-dialog-confirm').addEventListener('click', () => {
        dlg.remove();
        closePreset();
        if (type === 'nudge' && hasNudgeStore && data.uid) {
          this.ws.send({ type: 'nudge_delete', uid: data.uid });
        } else {
          this._deletePreset(data.id);
        }
      });
    });

    // Save
    overlay.querySelector('#cv2-preset-save').addEventListener('click', async () => {
      // Sync CodeMirror editor content back to file array before saving
      _syncEditorBack();

      const name = overlay.querySelector('#cv2-preset-name').value.trim();
      if (!name) {
        overlay.querySelector('#cv2-preset-name').style.borderColor = '#ef4444';
        return;
      }

      const isMcp = currentNudgeType === 'mcp';
      const saveData = {
        ...data,
        type,
        name,
        icon: currentIcon,
        system_prompt: overlay.querySelector('#cv2-preset-prompt').value,
        model: overlay.querySelector('#cv2-preset-model').value || null,
        language: overlay.querySelector('#cv2-preset-language').value || 'auto',
        files: isMcp ? [...currentFiles, ...mcpFiles, ...mcpDataFiles] : currentFiles,
        updated_at: new Date().toISOString(),
        mcp_entry_point: isMcp ? (mcpEntrySelect?.value || 'index.js') : undefined,
      };
      // Remove legacy nudge_type field
      delete saveData.nudge_type;
      // Cache tool names for catalog display
      if (isMcp && mcpToolsEl) {
        const toolChips = mcpToolsEl.querySelectorAll('span[style*="monospace"]');
        if (toolChips.length > 0) {
          saveData.mcp_tool_names = Array.from(toolChips).map(c => c.textContent.trim());
          saveData.mcp_tool_count = saveData.mcp_tool_names.length;
        }
      }

      // Collect doc_plugins from checkboxes (applies to both projects and nudges)
      const docPluginEls = overlay.querySelectorAll('#cv2-preset-doc-plugins input[data-doc-plugin]');
      if (docPluginEls.length > 0) {
        const enabledPlugins = [];
        docPluginEls.forEach(cb => { if (cb.checked) enabledPlugins.push(cb.dataset.docPlugin); });
        // If all are checked, store null (= all enabled, the default)
        saveData.doc_plugins = enabledPlugins.length === docPluginEls.length ? null : enabledPlugins;
      }

      if (type === 'nudge') {
        saveData.description = overlay.querySelector('#cv2-preset-description')?.value || '';
        saveData.suggestions = overlay.querySelector('#cv2-preset-suggestions')?.value || '';
        const caps = {};
        overlay.querySelectorAll('#cv2-preset-capabilities input[data-cap]').forEach(cb => {
          // Tri-state: indeterminate = null (don't care), checked = true, unchecked = false
          caps[cb.dataset.cap] = cb.indeterminate ? null : cb.checked;
        });
        saveData.capabilities = caps;

        // Nudge-specific fields
        if (hasNudgeStore) {
          saveData.category = overlay.querySelector('#cv2-preset-category')?.value || '';
          saveData.sub_category = overlay.querySelector('#cv2-preset-sub-category')?.value?.trim() || '';
          // Collect visibility from selected chips
          const visPatterns = [];
          overlay.querySelectorAll('#cv2-vis-selected .cv2-vis-chip').forEach(chip => {
            visPatterns.push(chip.dataset.pattern);
          });
          saveData.visibility = visPatterns;
          // Owner selector: team or personal
          const ownerSelect = overlay.querySelector('#cv2-preset-owner');
          const selectedTeamId = ownerSelect ? ownerSelect.value : '';
          if (selectedTeamId) {
            saveData.team_id = selectedTeamId;
            const team = this.teams.find(t => (t.teamId || t.team_id) === selectedTeamId);
            saveData.creator_name = team ? team.name : '';
          } else {
            delete saveData.team_id;
            saveData.creator_name = this.fullName || this.userName || '';
          }
          saveData.creator_email = saveData.creator_email || this.config.userEmail || '';
          // Admin owner transfer
          const transferInput = overlay.querySelector('#cv2-preset-transfer-email');
          if (transferInput && this.config.isAdmin) {
            const newEmail = transferInput.value.trim();
            if (newEmail) {
              saveData.creator_email = newEmail;
            }
          }
          // Admin-only flags
          const masterCb = overlay.querySelector('#cv2-preset-is-master');
          if (masterCb) saveData.is_master = masterCb.checked;
          const autoDiscoverCb = overlay.querySelector('#cv2-preset-auto-discover');
          if (autoDiscoverCb) {
            saveData.auto_discover = autoDiscoverCb.checked;
            saveData.auto_discover_when = (overlay.querySelector('#cv2-preset-auto-discover-when')?.value || '').trim();
          }
        }
      }

      if (type === 'nudge' && hasNudgeStore) {
        // Save to MongoDB via WS
        this.ws.send({ type: 'nudge_save', data: saveData });
      } else {
        await this._savePreset(saveData);
      }
      closePreset();
    });

    // Flush to live (nudge store only)
    overlay.querySelector('#cv2-preset-flush')?.addEventListener('click', () => {
      const dlg = document.createElement('div');
      dlg.className = 'cv2-dialog-overlay';
      dlg.innerHTML = `
        <div class="cv2-dialog">
          <div class="cv2-dialog-body">${this.t('chat.nudge_flush_confirm') || 'Publish dev to live?'}</div>
          <div class="cv2-dialog-actions">
            <button class="cv2-dialog-btn cv2-dialog-cancel">${this.t('chat.cancel')}</button>
            <button class="cv2-dialog-btn cv2-dialog-confirm">${this.t('chat.nudge_flush') || 'Publish'}</button>
          </div>
        </div>`;
      document.getElementById('chat-app').appendChild(dlg);
      dlg.querySelector('.cv2-dialog-cancel').addEventListener('click', () => dlg.remove());
      dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.remove(); });
      dlg.querySelector('.cv2-dialog-confirm').addEventListener('click', () => {
        dlg.remove();
        this.ws.send({ type: 'nudge_flush', uid: data.uid });
      });
    });
  },

  async _resizeIcon(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => {
          const max = 256;
          let w = img.width, h = img.height;
          if (w > max || h > max) {
            const ratio = Math.min(max / w, max / h);
            w = Math.round(w * ratio);
            h = Math.round(h * ratio);
          }
          const canvas = document.createElement('canvas');
          canvas.width = w;
          canvas.height = h;
          canvas.getContext('2d').drawImage(img, 0, 0, w, h);
          resolve(canvas.toDataURL('image/png'));
        };
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = reader.result;
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsDataURL(file);
    });
  },

  async _savePreset(data) {
    // Save to IDB (with full content for offline access)
    await this.idb.putPreset(data);
    // Send to server — include content only for binary files needing server text extraction
    const wsFiles = (data.files || []).map(f => {
      if (f.text_content) {
        // Already has extracted text — strip heavy content to save bandwidth
        const { content, ...rest } = f;
        return rest;
      }
      // Binary file needs server-side extraction — include content
      return f;
    });
    const wsData = { ...data, files: wsFiles };
    this.ws.send({ type: 'save_preset', data: wsData });
    // Apply changes to current session if this preset is active
    const isActive = (data.type === 'nudge' && this._activeNudgeId === data.id)
                  || (data.type !== 'nudge' && this._activeProjectId === data.id);
    if (isActive) {
      // Send only settings fields — omit heavy icon/file data
      this.ws.send({ type: 'load_preset', data: {
        id: data.id, type: data.type,
        system_prompt: data.system_prompt,
        model: data.model,
        language: data.language,
        doc_plugins: data.doc_plugins ?? null,
      }});
    }
    // Refresh UI
    await this.refreshConversations();
    this._showToast(data.type === 'nudge' ? this.t('chat.nudge_saved') : this.t('chat.project_saved'), 'positive');
  },

  async _deletePreset(id) {
    // Unlink conversations from the deleted preset
    const allMeta = await this.idb.getAllMeta();
    for (const meta of allMeta) {
      if (meta.project_id === id || meta.nudge_id === id) {
        const full = await this.idb.get(meta.id);
        if (full) {
          if (full.project_id === id) delete full.project_id;
          if (full.nudge_id === id) delete full.nudge_id;
          full.updated_at = new Date().toISOString();
          await this.idb.put(full);
        }
      }
    }
    await this.idb.deletePreset(id);
    await this.refreshConversations();
    this._showToast('Deleted', 'info');
  },

  async handlePresetSaved(msg) {
    // Server validated the preset — merge with IDB to preserve local file content
    if (!msg.data) return;
    const existing = await this.idb.getPreset(msg.data.id);
    if (existing && existing.files) {
      // Preserve base64 content from IDB; server may have extracted new text_content
      const localByFileId = Object.fromEntries(existing.files.map(f => [f.file_id, f]));
      for (const f of (msg.data.files || [])) {
        const local = localByFileId[f.file_id];
        if (local && local.content && !f.content) f.content = local.content;
      }
    }
    await this.idb.putPreset(msg.data);
    this.refreshConversations();
  },

  handlePresetApplied(msg) {
    // Confirmation from server that preset was applied to current session
    console.log('[Preset] Applied:', msg.preset_type, msg.preset_id);
  },

  });

  ChatFeatures.register('presets', {
    handleMessage: {
      'preset_saved': 'handlePresetSaved',
      'preset_applied': 'handlePresetApplied',
    },
  });
})();
