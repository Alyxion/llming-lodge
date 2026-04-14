/**
 * chat-sidebar.js — Sidebar, conversations, drafts, projects list
 * Extracted from chat-app.js — context menu v1
 */
(function() {
  const _SIDEBAR_PAGE_SIZE = 50;

  Object.assign(window._ChatAppProto, {

  // ── Conversation restore ──────────────────────────────

  async _restoreLastConversation() {
    try {
      // If user was on a fresh "new chat" screen, stay there
      if (localStorage.getItem('cv2-fresh-chat')) {
        this._restoreDraft();
        return;
      }
      // Prefer the last actively selected conversation, fall back to most recent
      const savedId = localStorage.getItem('cv2-active-conversation');
      let data = null;
      if (savedId) data = await this.idb.get(savedId);
      if (!data || !data.messages?.length) {
        data = await this.idb.getMostRecent();
        if (!data) { this._restoreDraft(); return; }
      }
      if (!data || !data.id || !data.messages?.length) { this._restoreDraft(); return; }
      // Render UI immediately from local data
      this.activeConvId = data.id;
      this._activeProjectId = data.project_id || null;
      this._activeNudgeId = data.nudge_id || null;
      this._autoUnderlyingModel = data.auto_underlying_model || '';
      this._renderSidebar();
      if (!this.chatVisible) this.showChat();
      this._blockDataStore?.clear();
      this.inlineDocBlocks = [];
      this.el.messages.innerHTML = '';

      // Restore nudge header + model lock if applicable
      const hasNudgeStore = this.config.nudgeCategories && this.config.nudgeCategories.length > 0;
      if (data.nudge_id && !hasNudgeStore) {
        const nudge = this.nudges.find(d => d.id === data.nudge_id);
        if (nudge) {
          this._renderNudgeChatHeader(nudge);
          this._lockModelForNudge(nudge);
        }
      } else if (data.project_id) {
        const proj = this.projects.find(p => p.id === data.project_id);
        if (proj) this._renderProjectChatHeader(proj);
      }
      // For nudge store: header will be rendered via handleNudgeMeta from server

      await this._renderLoadedMessages(data.messages || []);
      this._scrollToBottom();
      this._restoreDraft();

      // Restore documents from saved conversation or IDB
      if (data.documents && data.documents.length > 0) {
        this.documents = data.documents;
      } else if (data.id) {
        try {
          this.documents = await this.idb.getDocumentsForConversation(data.id);
        } catch (_) { this.documents = []; }
      }
      this._renderDocList();

      // Restore files from IDB file store
      if (data.file_refs && data.file_refs.length > 0) {
        this._savedFileRefs[data.id] = data.file_refs;
        this._pendingFileRestore = data.file_refs;
      }

      // Queue server hydration for when WS connects
      this._pendingRestore = data;
    } catch (err) {
      console.warn('[Chat] Failed to restore last conversation:', err);
    }
  },

  // ── Draft persistence (per-conversation, localStorage cv2-draft:<id>) ──

  _startDraftSaver() {
    this._draftDirty = false;
    this.el.textarea.addEventListener('input', () => { this._draftDirty = true; });
    setInterval(() => {
      if (!this._draftDirty) return;
      this._draftDirty = false;
      this._saveDraft();
    }, 2000);
  },

  _saveDraft() {
    try {
      const val = this.el.textarea.value;
      const key = this.activeConvId || '_fresh';
      if (val) localStorage.setItem(`cv2-draft:${key}`, val);
      else localStorage.removeItem(`cv2-draft:${key}`);
    } catch (_) {}
  },

  _restoreDraft() {
    try {
      const key = this.activeConvId || '_fresh';
      const draft = localStorage.getItem(`cv2-draft:${key}`);
      if (draft && this.el.textarea) {
        this.el.textarea.value = draft;
        this._autoResizeTextarea();
      }
    } catch (_) {}
  },

  _clearDraft(convKey) {
    try { localStorage.removeItem(`cv2-draft:${convKey || this.activeConvId || '_fresh'}`); } catch (_) {}
  },

  // ── Sidebar Rendering ─────────────────────────────────

  async refreshConversations() {
    try {
      const hasNudgeStore = this.config.nudgeCategories && this.config.nudgeCategories.length > 0;
      const [allMeta, projects, nudges] = await Promise.all([
        this.idb.getAllMeta(),
        this.idb.getAllPresets('project'),
        hasNudgeStore ? Promise.resolve([]) : this.idb.getAllPresets('nudge'),
      ]);
      this.projects = projects;
      if (!hasNudgeStore) this.nudges = nudges;
      this.conversations = allMeta.map(c => ({
        id: c.id,
        title: c.title || c.first_user_snippet?.substring(0, 30) || this.t('chat.untitled'),
        created_at: c.created_at,
        updated_at: c.updated_at,
        project_id: c.project_id || null,
        nudge_id: c.nudge_id || null,
        favorited: c.favorited || false,
      })).sort((a, b) => (b.updated_at || b.created_at || '').localeCompare(a.updated_at || a.created_at || ''));

      this._renderSidebar();
    } catch (err) {
      console.error('[Sidebar] refresh error:', err);
    }
  },

  // ── Sidebar Sections (Nudges / Projects / Chats) ────

  _renderSidebar() {
    const container = this.el.sidebarSections;
    if (!container) return;

    const base = this.config.staticBase;
    const nudgeIcon = this.config.nudgeSectionIcon || 'icons/phosphor/regular/drop.svg';
    container.innerHTML =
      this._buildSection('nudges', this.t('chat.nudges_section'), this._buildNudgesContent(), `${base}/${nudgeIcon}`) +
      this._buildSection('projects', this.t('chat.projects_section'), this._buildProjectsContent(), `${base}/icons/phosphor/regular/folder.svg`) +
      this._buildSection('chats', this.t('chat.chats_section'), this._buildChatsContent(), `${base}/icons/phosphor/regular/chats.svg`);

    this._bindSidebarEvents(container);
  },

  _buildSection(sectionId, title, contentHtml, iconUrl) {
    const collapsed = this._collapsedSections.has(sectionId);
    const chevron = collapsed ? 'chevron_right' : 'expand_more';
    const iconHtml = iconUrl
      ? `<img src="${iconUrl}" class="cv2-section-icon" alt="">`
      : '';
    return `
      <div class="cv2-section ${collapsed ? 'cv2-collapsed' : ''}" data-section="${sectionId}">
        <div class="cv2-section-header" data-section="${sectionId}">
          ${iconHtml}
          <span class="cv2-section-label">${title}</span>
          <span class="material-icons cv2-section-chevron" style="font-size:14px">${chevron}</span>
        </div>
        <div class="cv2-section-content">
          ${contentHtml}
        </div>
      </div>`;
  },

  _buildNudgesContent() {
    const hasNudgeStore = this.config.nudgeCategories && this.config.nudgeCategories.length > 0;

    // Server-side nudge mode: show favorites from MongoDB
    if (hasNudgeStore) {
      const favs = this.favoriteNudges || [];
      let html = '';
      if (favs.length > 0) {
        html = favs.map(d => {
          const iconHtml = this._renderIcon(d.icon, d.icon && this._isIconUrl(d.icon) ? 'cv2-nudge-inline-icon' : 'cv2-nudge-inline-icon-default', 'science');
          return `
            <div class="cv2-nudge-inline" data-nudge-uid="${this._escAttr(d.uid)}">
              ${iconHtml}
              <span class="cv2-nudge-inline-name">${this._escHtml(d.name)}</span>
              <button class="cv2-nudge-inline-unfav" title="${this.t('chat.nudge_unfavorite') || 'Remove'}"><span class="material-icons" style="font-size:14px">favorite</span></button>
            </div>`;
        }).join('');
      }
      html += `<button class="cv2-project-new-btn" data-action="explore-nudges">
        <span class="material-icons" style="font-size:14px">explore</span> ${this.t('chat.explore_nudges') || 'Explore'}
      </button>`;
      return html;
    }

    // Legacy IDB mode
    const visibleCount = this._sectionPage.nudges * _SIDEBAR_PAGE_SIZE;
    const items = this.nudges.slice(0, visibleCount);
    const hasMore = this.nudges.length > visibleCount;

    let html = '';
    if (this.nudges.length === 0) {
      html = `<button class="cv2-project-new-btn" data-action="new-nudge">
        <span class="material-icons" style="font-size:14px">add</span> ${this.t('chat.new_nudge')}
      </button>`;
      return html;
    }

    html = items.map(d => {
      const iconHtml = d.icon
        ? `<img src="${d.icon}" class="cv2-nudge-inline-icon" alt="">`
        : '<span class="material-icons cv2-nudge-inline-icon-default">science</span>';
      return `
        <div class="cv2-nudge-inline" data-nudge-id="${this._escAttr(d.id)}">
          ${iconHtml}
          <span class="cv2-nudge-inline-name">${this._escHtml(d.name)}</span>
          <button class="cv2-nudge-inline-edit" title="Edit"><span class="material-icons" style="font-size:14px">edit</span></button>
        </div>`;
    }).join('');

    if (hasMore) {
      html += `<div class="cv2-scroll-sentinel" data-section="nudges"></div>`;
    }
    html += `<button class="cv2-project-new-btn" data-action="new-nudge">
      <span class="material-icons" style="font-size:14px">add</span> ${this.t('chat.new_nudge')}
    </button>`;
    return html;
  },

  _buildProjectsContent() {
    const visibleCount = this._sectionPage.projects * _SIDEBAR_PAGE_SIZE;
    const items = this.projects.slice(0, visibleCount);
    const hasMore = this.projects.length > visibleCount;

    let html = '';
    for (const proj of items) {
      const isActive = this._activeProjectId === proj.id && this._activeView === 'project';
      const iconHtml = this._renderIcon(proj.icon, proj.icon && this._isIconUrl(proj.icon) ? 'cv2-project-icon' : 'cv2-project-icon-default', 'folder');

      html += `
        <div class="cv2-project-item ${isActive ? 'cv2-project-active' : ''}" data-project-id="${this._escAttr(proj.id)}">
          <div class="cv2-project-header">
            ${iconHtml}
            <span class="cv2-project-name">${this._escHtml(proj.name)}</span>
          </div>
        </div>`;
    }

    if (hasMore) {
      html += `<div class="cv2-scroll-sentinel" data-section="projects"></div>`;
    }
    html += `<button class="cv2-project-new-btn" data-action="new-project">
      <span class="material-icons" style="font-size:14px">add</span> ${this.t('chat.new_project')}
    </button>`;
    return html;
  },

  _buildChatsContent() {
    const ungrouped = this.conversations.filter(c => !c.project_id);
    if (ungrouped.length === 0) {
      return `<div class="cv2-conv-empty">${this.t('chat.no_conversations')}</div>`;
    }
    // Favorited conversations sort to the top
    ungrouped.sort((a, b) => (b.favorited ? 1 : 0) - (a.favorited ? 1 : 0));
    const visibleCount = this._sectionPage.chats * _SIDEBAR_PAGE_SIZE;
    const items = ungrouped.slice(0, visibleCount);
    const hasMore = ungrouped.length > visibleCount;

    let html = items.map(c => {
      const nudge = c.nudge_id ? this.nudges.find(d => d.id === c.nudge_id) : null;
      return `
      <div class="cv2-conv-item ${c.id === this.activeConvId ? 'cv2-active' : ''}" data-id="${this._escAttr(c.id)}">
        ${nudge ? '<span class="material-icons cv2-conv-nudge-icon" style="font-size:12px;opacity:0.5;margin-right:2px">science</span>' : ''}
        <span class="cv2-conv-title">${this._escHtml(this._truncTitle(c.title))}</span>
        ${c.favorited ? '<span class="material-icons cv2-conv-fav-icon">star</span>' : ''}
        <button class="cv2-conv-more" data-id="${this._escAttr(c.id)}">
          <span class="material-icons" style="font-size:16px">more_horiz</span>
        </button>
      </div>`;
    }).join('');

    if (hasMore) {
      html += `<div class="cv2-scroll-sentinel" data-section="chats"></div>`;
    }
    return html;
  },

  _bindSidebarEvents(container) {
    // Section header collapse toggle
    container.querySelectorAll('.cv2-section-header').forEach(hdr => {
      hdr.addEventListener('click', () => {
        const section = hdr.dataset.section;
        if (this._collapsedSections.has(section)) this._collapsedSections.delete(section);
        else this._collapsedSections.add(section);
        try { localStorage.setItem('cv2-collapsed-sections', JSON.stringify([...this._collapsedSections])); } catch (_) {}
        this._renderSidebar();
      });
    });

    // Infinite-scroll sentinels
    const scrollRoot = this.el.sidebarSections;
    container.querySelectorAll('.cv2-scroll-sentinel').forEach(sentinel => {
      const obs = new IntersectionObserver(([entry]) => {
        if (entry.isIntersecting) {
          obs.disconnect();
          this._sectionPage[sentinel.dataset.section]++;
          this._renderSidebar();
        }
      }, { root: scrollRoot, rootMargin: '100px' });
      obs.observe(sentinel);
    });

    // Nudge items
    container.querySelectorAll('.cv2-nudge-inline').forEach(el => {
      const nudgeUid = el.dataset.nudgeUid;
      const nudgeId = el.dataset.nudgeId;

      if (nudgeUid) {
        // Server-side nudge favorite
        el.addEventListener('click', () => {
          this._startNudgeChat({ uid: nudgeUid });
        });
        el.querySelector('.cv2-nudge-inline-unfav')?.addEventListener('click', (e) => {
          e.stopPropagation();
          this._removeFavorite(nudgeUid);
        });
      } else if (nudgeId) {
        // Legacy IDB nudge
        el.addEventListener('click', () => {
          const nudge = this.nudges.find(d => d.id === nudgeId);
          if (nudge) this._startNudgeChat(nudge);
        });
        el.querySelector('.cv2-nudge-inline-edit')?.addEventListener('click', (e) => {
          e.stopPropagation();
          const nudge = this.nudges.find(d => d.id === nudgeId);
          if (nudge) this._renderPresetDialog(nudge, 'nudge');
        });
      }
    });

    // Explore Nudges button
    container.querySelector('[data-action="explore-nudges"]')?.addEventListener('click', () => {
      this._renderNudgeExplorer();
    });

    // New Nudge button (legacy)
    container.querySelector('[data-action="new-nudge"]')?.addEventListener('click', () => {
      this._renderPresetDialog(null, 'nudge');
    });

    // Project headers — open full-screen project view
    container.querySelectorAll('.cv2-project-header').forEach(hdr => {
      const item = hdr.closest('.cv2-project-item');
      const projId = item.dataset.projectId;
      hdr.addEventListener('click', () => this._openProjectView(projId));
    });

    // New project button
    container.querySelector('[data-action="new-project"]')?.addEventListener('click', () => {
      this._renderPresetDialog(null, 'project');
    });

    // Chat conversation clicks
    const chatsSection = container.querySelector('[data-section="chats"]');
    if (chatsSection) {
      this._bindConversationClicks(chatsSection.querySelector('.cv2-section-content'));
    }
  },

  _bindConversationClicks(container) {
    if (!container) return;
    container.querySelectorAll('.cv2-conv-item').forEach(el => {
      el.addEventListener('click', () => this._selectConversation(el.dataset.id));
      // Right-click context menu
      el.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this._showConvContextMenu(el.dataset.id, el.querySelector('.cv2-conv-more') || el, e);
      });
    });
    container.querySelectorAll('.cv2-conv-more').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        this._showConvContextMenu(btn.dataset.id, btn);
      });
    });
  },

  async _selectConversation(id) {
    if (id === this.activeConvId) return;
    // Cancel any in-flight streaming before switching
    if (this.streaming) {
      this.ws.send({ type: 'stop_streaming' });
      this.streaming = false;
      this._updateSendButton();
    }
    // On mobile, close sidebar after selecting a conversation
    if (this._closeSidebarOnMobile) this._closeSidebarOnMobile();
    // Exit incognito mode when switching away (session is discarded — never persisted)
    if (this.incognito) this._exitIncognito();
    // Save draft for the conversation we're leaving
    if (this.el.textarea.value.trim()) this._saveDraft();
    this.activeConvId = id;
    this.el.textarea.value = '';
    this._autoResizeTextarea();
    // Clear pending files/images from previous conversation
    this._pendingFiles.length = 0;
    this._pendingImages.length = 0;
    this._renderAttachments();
    try {
      localStorage.setItem('cv2-active-conversation', id);
      localStorage.removeItem('cv2-fresh-chat');
    } catch (_) {}

    // Load from IDB and send to server
    const data = await this.idb.get(id);
    // Track project/nudge context from loaded conversation
    this._activeProjectId = data?.project_id || null;
    this._activeNudgeId = data?.nudge_id || null;
    this._autoUnderlyingModel = data?.auto_underlying_model || '';
    this._renderSidebar();
    if (data) {
      const hasNudgeStore = this.config.nudgeCategories && this.config.nudgeCategories.length > 0;

      // Attach preset files for document re-injection
      // For nudges with nudge store: server handles file injection
      if (data.nudge_id && !hasNudgeStore) {
        const nudge = this.nudges.find(d => d.id === data.nudge_id);
        if (nudge && nudge.files && nudge.files.length > 0) {
          data._preset_files = this._wsFiles(nudge.files);
        }
      } else if (data.project_id) {
        const proj = this.projects.find(p => p.id === data.project_id);
        if (proj && proj.files && proj.files.length > 0) {
          // Include content for files missing text_content (server needs it for extraction)
          data._preset_files = (proj.files || []).map(f => {
            if (f.text_content) { const { content, ...rest } = f; return rest; }
            return f;
          });
        }
      }
      this.ws.send({ type: 'load_conversation', data });

      // Restore files from IDB to server — deferred until session_id_updated
      // arrives so the upload manager exists under the correct session ID.
      if (data.file_refs && data.file_refs.length > 0) {
        this._savedFileRefs[id] = data.file_refs;
        this._pendingFileRestore = data.file_refs;
      }

      // Rebuild chat UI with loaded messages (and tear down any active overlay)
      if (!this.chatVisible) this.showChat();
      else if (this._activeView !== 'chat') this._switchView('chat');
      this._blockDataStore?.clear();
      this.documents = [];
      this.inlineDocBlocks = [];
      this._closeWorkspace();
      this._renderDocList();
      this.el.messages.innerHTML = '';

      // Show nudge header if this conversation belongs to a nudge
      if (data.nudge_id) {
        this._removeProjectChatHeader();
        if (hasNudgeStore) {
          // Server will send nudge_meta message, header rendered in handleNudgeMeta
          // For now, just clear any old header
          this._removeNudgeChatHeader();
        } else {
          const nudge = this.nudges.find(d => d.id === data.nudge_id);
          if (nudge) {
            this._renderNudgeChatHeader(nudge);
            this._lockModelForNudge(nudge);
          } else {
            this._removeNudgeChatHeader();
            this._unlockModel();
          }
        }
      } else if (data.project_id) {
        this._removeNudgeChatHeader();
        this._unlockModel();
        const proj = this.projects.find(p => p.id === data.project_id);
        if (proj) this._renderProjectChatHeader(proj);
      } else {
        this._removeNudgeChatHeader();
        this._removeProjectChatHeader();
        this._unlockModel();
      }

      await this._renderLoadedMessages(data.messages || []);
      // Open match navigator if triggered from search
      if (this._pendingMatchQuery) {
        const q = this._pendingMatchQuery;
        this._pendingMatchQuery = null;
        this._openMatchNavigator(q);
      } else {
        this._scrollToBottom();
      }
      // Restore any draft the user had in this conversation
      this._restoreDraft();
    }
  },

  // ── Context Menu ─────────────────────────────────

  _showConvContextMenu(convId, anchorEl, mouseEvent) {
    this._closeConvContextMenu();
    const conv = this.conversations.find(c => c.id === convId);
    if (!conv) return;

    const menu = document.createElement('div');
    menu.className = 'cv2-conv-ctx';
    const favLabel = conv.favorited ? (this.t('chat.unfavorite') || 'Unfavorite') : (this.t('chat.favorite') || 'Favorite');
    const favIcon = conv.favorited ? 'star' : 'star_outline';
    menu.innerHTML = `
      <button class="cv2-conv-ctx-item" data-action="favorite">
        <span class="material-icons">${favIcon}</span> ${favLabel}
      </button>
      <button class="cv2-conv-ctx-item" data-action="rename">
        <span class="material-icons">edit</span> ${this.t('chat.rename') || 'Rename'}
      </button>
      <button class="cv2-conv-ctx-item" data-action="move">
        <span class="material-icons">drive_file_move</span> ${this.t('chat.move_to_project') || 'Move to Project'}
        <span class="material-icons cv2-conv-ctx-arrow">chevron_right</span>
      </button>
      <button class="cv2-conv-ctx-item cv2-danger" data-action="delete">
        <span class="material-icons">delete</span> ${this.t('chat.delete')}
      </button>`;
    document.getElementById('chat-app').appendChild(menu);

    // Position: below-right of anchor, flip up if near bottom
    const rect = anchorEl.getBoundingClientRect();
    let top = rect.bottom + 4;
    let left = rect.left;
    // Use mouse position for right-click
    if (mouseEvent) {
      top = mouseEvent.clientY;
      left = mouseEvent.clientX;
    }
    // Flip up if near bottom
    if (top + menu.offsetHeight > window.innerHeight - 8) {
      top = (mouseEvent ? mouseEvent.clientY : rect.top) - menu.offsetHeight - 4;
    }
    // Prevent overflow right
    if (left + menu.offsetWidth > window.innerWidth - 8) {
      left = window.innerWidth - menu.offsetWidth - 8;
    }
    menu.style.top = `${Math.max(4, top)}px`;
    menu.style.left = `${Math.max(4, left)}px`;

    this._ctxMenuEl = menu;
    this._ctxMenuConvId = convId;

    // Bind actions
    menu.querySelector('[data-action="favorite"]').addEventListener('click', () => this._toggleConvFavorite(convId));
    menu.querySelector('[data-action="rename"]').addEventListener('click', () => this._renameConversation(convId));
    menu.querySelector('[data-action="move"]').addEventListener('mouseenter', (e) => this._showMoveSubmenu(e.currentTarget));
    menu.querySelector('[data-action="move"]').addEventListener('click', (e) => this._showMoveSubmenu(e.currentTarget));
    menu.querySelector('[data-action="delete"]').addEventListener('click', () => this._deleteConversation(convId));

    // Close on outside click or Escape
    const closeHandler = (e) => {
      if (!menu.contains(e.target) && !(this._ctxSubMenuEl && this._ctxSubMenuEl.contains(e.target))) {
        this._closeConvContextMenu();
      }
    };
    const keyHandler = (e) => {
      if (e.key === 'Escape') this._closeConvContextMenu();
    };
    setTimeout(() => {
      document.addEventListener('click', closeHandler, true);
      document.addEventListener('keydown', keyHandler);
    }, 0);
    this._ctxMenuCloseHandler = closeHandler;
    this._ctxMenuKeyHandler = keyHandler;
  },

  _closeConvContextMenu() {
    if (this._ctxMenuEl) { this._ctxMenuEl.remove(); this._ctxMenuEl = null; }
    if (this._ctxSubMenuEl) { this._ctxSubMenuEl.remove(); this._ctxSubMenuEl = null; }
    if (this._ctxMenuCloseHandler) {
      document.removeEventListener('click', this._ctxMenuCloseHandler, true);
      this._ctxMenuCloseHandler = null;
    }
    if (this._ctxMenuKeyHandler) {
      document.removeEventListener('keydown', this._ctxMenuKeyHandler);
      this._ctxMenuKeyHandler = null;
    }
    this._ctxMenuConvId = null;
  },

  _showMoveSubmenu(menuItem) {
    if (this._ctxSubMenuEl) { this._ctxSubMenuEl.remove(); this._ctxSubMenuEl = null; }
    const conv = this.conversations.find(c => c.id === this._ctxMenuConvId);
    const sub = document.createElement('div');
    sub.className = 'cv2-conv-ctx-sub';

    // "No project" option
    let html = `<button class="cv2-conv-ctx-item ${!conv?.project_id ? 'cv2-active-mark' : ''}" data-project="">
      <span class="material-icons">block</span> ${this.t('chat.no_project') || 'No project'}
    </button>`;
    for (const proj of this.projects) {
      const iconHtml = proj.icon
        ? `<img src="${proj.icon}" style="width:16px;height:16px;border-radius:3px" alt="">`
        : '<span class="material-icons">folder</span>';
      html += `<button class="cv2-conv-ctx-item ${conv?.project_id === proj.id ? 'cv2-active-mark' : ''}" data-project="${this._escAttr(proj.id)}">
        ${iconHtml} ${this._escHtml(proj.name)}
      </button>`;
    }
    sub.innerHTML = html;
    document.getElementById('chat-app').appendChild(sub);

    // Position to the right of menuItem, flip left if overflowing
    const r = menuItem.getBoundingClientRect();
    let left = r.right + 4;
    let top = r.top;
    if (left + sub.offsetWidth > window.innerWidth - 8) {
      left = r.left - sub.offsetWidth - 4;
    }
    if (top + sub.offsetHeight > window.innerHeight - 8) {
      top = window.innerHeight - sub.offsetHeight - 8;
    }
    sub.style.top = `${Math.max(4, top)}px`;
    sub.style.left = `${Math.max(4, left)}px`;
    this._ctxSubMenuEl = sub;

    // Bind project clicks
    sub.querySelectorAll('[data-project]').forEach(btn => {
      btn.addEventListener('click', () => {
        this._moveToProject(this._ctxMenuConvId, btn.dataset.project || null);
      });
    });
  },

  _moveToProject(convId, projectId) {
    this.idb.get(convId).then(data => {
      if (data) {
        data.project_id = projectId || null;
        data.updated_at = new Date().toISOString();
        this.idb.put(data).then(() => {
          this._closeConvContextMenu();
          // Update active project if we moved the current conversation
          if (convId === this.activeConvId) {
            this._activeProjectId = projectId || null;
          }
          if (projectId) this._expandedProjects.add(projectId);
          this.refreshConversations();
        });
      }
    });
  },

  _renameConversation(convId) {
    this._closeConvContextMenu();
    const item = document.querySelector(`.cv2-conv-item[data-id="${convId}"]`);
    const titleSpan = item?.querySelector('.cv2-conv-title');
    if (!titleSpan) return;

    const originalText = titleSpan.textContent;
    const input = document.createElement('input');
    input.className = 'cv2-conv-rename-input';
    input.type = 'text';
    input.value = originalText;
    titleSpan.textContent = '';
    titleSpan.appendChild(input);
    input.select();
    input.focus();

    const commit = () => {
      const newTitle = input.value.trim();
      if (newTitle && newTitle !== originalText) {
        this._updateConversationTitle(convId, newTitle);
      } else {
        titleSpan.textContent = originalText;
      }
    };

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      if (e.key === 'Escape') { input.value = originalText; input.blur(); }
    });
    input.addEventListener('blur', commit, { once: true });
    // Prevent click on input from selecting the conversation
    input.addEventListener('click', (e) => e.stopPropagation());
  },

  _toggleConvFavorite(convId) {
    this.idb.get(convId).then(data => {
      if (data) {
        data.favorited = !data.favorited;
        data.updated_at = new Date().toISOString();
        this.idb.put(data).then(() => {
          this._closeConvContextMenu();
          this.refreshConversations();
        });
      }
    });
  },

  /** Restore files from IDB file store back to the server session. */
  async _restoreFiles(fileRefs) {
    if (!fileRefs || fileRefs.length === 0) return;
    const imagesToPost = [];
    const docFiles = [];     // File objects to upload in one batch
    const docHashes = [];    // corresponding stored hashes
    for (const ref of fileRefs) {
      try {
        const fileRecord = await this.idb.getFile(ref.hash);
        if (!fileRecord) continue;
        if (ref.type === 'image') {
          const dataUri = this._bufferToDataUri(fileRecord.data, ref.mime_type);
          this._pendingImages.push({ dataUri, hash: ref.hash });
          imagesToPost.push(dataUri.split(',')[1]);
        } else {
          const blob = new Blob([fileRecord.data], { type: ref.mime_type });
          docFiles.push(new File([blob], ref.name, { type: ref.mime_type }));
          docHashes.push(ref.hash);
        }
      } catch (err) {
        console.warn('[FileRestore] Failed to restore file:', ref.name, err);
      }
    }
    // Batch-upload all document files in a single request
    if (docFiles.length) {
      await this._uploadFiles(docFiles);
      // Overwrite server-assigned hashes with the stored IDB hashes
      const start = this._pendingFiles.length - docFiles.length;
      for (let i = 0; i < docHashes.length; i++) {
        const pf = this._pendingFiles[start + i];
        if (pf) pf.hash = docHashes[i];
      }
    }
    // Post restored images to server
    if (imagesToPost.length > 0) {
      fetch(`/api/llming/image-paste/${this.sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: this._pendingImages.map(i => i.dataUri.split(',')[1]) }),
      }).catch(() => {});
    }
    this._renderAttachments();
  },

  _deleteConversation(id) {
    this._closeConvContextMenu();
    delete this._savedFileRefs[id];
    this.idb.delete(id).then(() => {
      // GC orphaned files
      this.idb.removeAllRefsForConversation(id).catch(() => {});
      if (this.activeConvId === id) {
        this.activeConvId = '';
        this._activeProjectId = null;
        this._blockDataStore?.clear();
        try { localStorage.removeItem('cv2-active-conversation'); } catch (_) {}
      }
      this.refreshConversations();
    });
  },

  _updateConversationTitle(id, title) {
    this.idb.get(id).then(data => {
      if (data) {
        data.title = title;
        data.updated_at = new Date().toISOString();
        this.idb.put(data).then(() => this.refreshConversations());
      }
    });
  },

  _renderProjectConversations(projectId) {
    const convs = this.conversations.filter(c => c.project_id === projectId);
    if (convs.length === 0) return '<div class="cv2-conv-empty" style="padding:4px 8px 4px 32px;font-size:11px">No conversations</div>';
    return convs.map(c => `
      <div class="cv2-conv-item ${c.id === this.activeConvId ? 'cv2-active' : ''}" data-id="${this._escAttr(c.id)}" style="padding-left:32px">
        <span class="cv2-conv-title">${this._escHtml(this._truncTitle(c.title))}</span>
        ${c.favorited ? '<span class="material-icons cv2-conv-fav-icon">star</span>' : ''}
        <button class="cv2-conv-more" data-id="${this._escAttr(c.id)}">
          <span class="material-icons" style="font-size:16px">more_horiz</span>
        </button>
      </div>
    `).join('');
  },

  // ── Project View ─────────────────────────────────────────

  _openProjectView(projId) {
    const proj = this.projects.find(p => p.id === projId);
    if (!proj) return;

    // Tear down any existing overlay (preset, explorer, previous project view)
    this._switchView('project');

    // Set active project so new messages go to this project
    this._activeProjectId = projId;
    this.activeConvId = '';

    // Clear pending attachments from previous context (e.g. new chat)
    this._pendingFiles.length = 0;
    this._pendingImages.length = 0;
    this._renderAttachments();

    // Hide messages + initial view (NOT the whole chatWrapper — input stays)
    // Use class-based hiding so showChat() can restore without worrying about stale inline styles
    const initialView = document.getElementById('cv2-initial-view');
    const messagesWrap = document.getElementById('cv2-messages-wrap');
    if (initialView) initialView.classList.add('cv2-pv-hidden');
    if (messagesWrap) messagesWrap.classList.add('cv2-pv-hidden');

    const iconHtml = proj.icon
      ? `<img src="${proj.icon}" class="cv2-pv-icon" alt="">`
      : '<span class="material-icons cv2-pv-icon-default">folder</span>';

    const files = proj.files || [];
    const convs = this.conversations.filter(c => c.project_id === projId);

    const overlay = document.createElement('div');
    overlay.className = 'cv2-project-view';
    overlay.id = 'cv2-project-view';
    overlay.innerHTML = `
      <div class="cv2-pv-inner">
        <div class="cv2-pv-header">
          ${iconHtml}
          <h2 class="cv2-pv-title">${this._escHtml(proj.name)}</h2>
          <button class="cv2-pv-edit" title="Edit project">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 256 256" fill="currentColor"><path d="M227.31,73.37,182.63,28.68a16,16,0,0,0-22.63,0L36.69,152A15.86,15.86,0,0,0,32,163.31V208a16,16,0,0,0,16,16H92.69A15.86,15.86,0,0,0,104,219.31L227.31,96a16,16,0,0,0,0-22.63ZM92.69,208H48V163.31l88-88L180.69,120ZM192,108.68,147.31,64l24-24L216,84.68Z"/></svg>
          </button>
        </div>

        ${proj.system_prompt ? `
          <div class="cv2-pv-section" data-section="description">
            <p class="cv2-pv-desc">${this._escHtml(proj.system_prompt)}</p>
          </div>` : ''}

        ${files.length > 0 ? `
          <div class="cv2-pv-section" data-section="files">
            <div class="cv2-pv-section-label">${this.t('chat.knowledge') || 'Knowledge'} <span class="cv2-pv-count">${files.length}</span></div>
            <div class="cv2-pv-files">
              ${files.map(f => this._pvFileCard(f)).join('')}
            </div>
          </div>` : ''}

        <div class="cv2-pv-section" data-section="chats">
          <div class="cv2-pv-section-label">${this.t('chat.conversations') || 'Conversations'} <span class="cv2-pv-count">${convs.length}</span></div>
          ${convs.length > 0
            ? `<div class="cv2-pv-cards">${convs.map(c => this._pvConvCard(c)).join('')}</div>`
            : `<div class="cv2-pv-empty">No conversations yet — type below to start one.</div>`}
        </div>
      </div>`;

    // Insert before input area
    const chatWrapper = document.querySelector('.cv2-chat-wrapper');
    const inputArea = chatWrapper?.querySelector('.cv2-input-area');
    if (chatWrapper && inputArea) {
      chatWrapper.insertBefore(overlay, inputArea);
    }

    // Register cleanup handler
    this._closeActiveView = () => this._closeProjectView();

    // Highlight active project in sidebar
    this._renderSidebar();

    // Bind events
    overlay.querySelector('.cv2-pv-edit')?.addEventListener('click', () => {
      this._renderPresetDialog(proj, 'project', {
        onClose: () => {
          // Restore topbar/chatWrapper hidden by preset dialog
          const main = document.querySelector('.cv2-main');
          const tb = main?.querySelector('.cv2-topbar');
          const cw = main?.querySelector('.cv2-chat-wrapper');
          if (tb) tb.style.display = '';
          if (cw) cw.style.display = '';
          // Return to project view with refreshed data (or chat if deleted)
          const updated = this.projects.find(p => p.id === proj.id);
          if (updated) {
            this._openProjectView(updated);
          } else {
            this._activeView = 'chat';
            this._closeActiveView = null;
          }
        },
      });
    });

    overlay.querySelectorAll('.cv2-pv-card').forEach(card => {
      card.addEventListener('click', () => {
        this._selectConversation(card.dataset.id);
      });
    });
    overlay.querySelectorAll('.cv2-pv-card-delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const convId = btn.dataset.id;
        const card = btn.closest('.cv2-pv-card');
        // Remove card from DOM immediately
        if (card) card.remove();
        // Update count label
        const countEl = overlay.querySelector('[data-section="chats"] .cv2-pv-count');
        const remaining = overlay.querySelectorAll('.cv2-pv-card').length;
        if (countEl) countEl.textContent = remaining;
        // Show empty state if no cards left
        if (remaining === 0) {
          const cardsContainer = overlay.querySelector('.cv2-pv-cards');
          if (cardsContainer) cardsContainer.outerHTML = '<div class="cv2-pv-empty">No conversations yet — type below to start one.</div>';
        }
        // Delete from IDB in background
        delete this._savedFileRefs[convId];
        this.idb.delete(convId).then(() => {
          this.idb.removeAllRefsForConversation(convId).catch(() => {});
          if (this.activeConvId === convId) {
            this.activeConvId = '';
            this._blockDataStore?.clear();
            try { localStorage.removeItem('cv2-active-conversation'); } catch (_) {}
          }
          // Refresh sidebar conversation list (but don't touch project view)
          this.refreshConversations();
        });
      });
    });
  },

  _closeProjectView() {
    document.getElementById('cv2-project-view')?.remove();
    // Remove class-based hiding — the existing class/style logic in showChat/handleChatCleared
    // handles the actual visibility of initialView vs messagesWrap
    document.getElementById('cv2-initial-view')?.classList.remove('cv2-pv-hidden');
    document.getElementById('cv2-messages-wrap')?.classList.remove('cv2-pv-hidden');
    this._activeView = 'chat';
    this._closeActiveView = null;
  },

  _pvConvCard(conv) {
    const title = this._escHtml(conv.title || this.t('chat.untitled'));
    const date = conv.created_at ? this._pvRelDate(conv.created_at) : '';
    return `
      <div class="cv2-pv-card" data-id="${this._escAttr(conv.id)}">
        <div class="cv2-pv-card-title">${title}</div>
        ${conv.favorited ? '<span class="material-icons cv2-pv-card-star">star</span>' : ''}
        ${date ? `<div class="cv2-pv-card-date">${date}</div>` : ''}
        <button class="cv2-pv-card-delete" data-id="${this._escAttr(conv.id)}" title="${this.t('chat.delete') || 'Delete'}">
          <span class="material-icons" style="font-size:16px">delete</span>
        </button>
      </div>`;
  },

  _pvFileCard(file) {
    const name = this._escHtml(file.name || 'File');
    const ext = (file.name || '').split('.').pop().toLowerCase();
    const iconMap = { pdf: 'picture_as_pdf', doc: 'description', docx: 'description', txt: 'article', csv: 'table_chart', xlsx: 'table_chart', xls: 'table_chart', png: 'image', jpg: 'image', jpeg: 'image', gif: 'image' };
    const icon = iconMap[ext] || 'insert_drive_file';
    return `
      <div class="cv2-pv-file">
        <span class="material-icons" style="font-size:14px">${icon}</span>
        <span>${name}</span>
      </div>`;
  },

  _pvRelDate(isoStr) {
    try {
      const d = new Date(isoStr);
      const now = new Date();
      const diff = now - d;
      const mins = Math.floor(diff / 60000);
      if (mins < 1) return 'just now';
      if (mins < 60) return `${mins}m ago`;
      const hrs = Math.floor(mins / 60);
      if (hrs < 24) return `${hrs}h ago`;
      const days = Math.floor(hrs / 24);
      if (days < 7) return `${days}d ago`;
      return d.toLocaleDateString();
    } catch (_) { return ''; }
  },

  // ── Export ─────────────────────────────────────────────

  async _exportAll() {
    try {
      // Dump ALL IndexedDB stores for complete backup
      const db = this.idb.db;
      const storeNames = ['conversations', 'conv_meta', 'documents', 'files', 'presets', 'favorites'];
      const backup = { _version: 2, _exported: new Date().toISOString(), _stores: {} };
      for (const name of storeNames) {
        if (!db.objectStoreNames.contains(name)) continue;
        const tx = db.transaction(name, 'readonly');
        const store = tx.objectStore(name);
        const all = await new Promise((res, rej) => {
          const req = store.getAll();
          req.onsuccess = () => res(req.result);
          req.onerror = () => rej(req.error);
        });
        backup._stores[name] = all;
      }
      const json = JSON.stringify(backup);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chat-backup_${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
      console.log('[Export] Complete:', Object.entries(backup._stores).map(([k, v]) => `${k}: ${v.length}`).join(', '));
    } catch (err) {
      console.error('[Export] Failed:', err);
    }
  },

  async _importAll() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.addEventListener('change', async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        let stats;
        if (data._version >= 2 && data._stores) {
          stats = await this._importV2(data);
        } else if (Array.isArray(data)) {
          // Legacy v1: plain array of conversations
          stats = await this._importV1(data);
        } else {
          console.warn('[Import] Unknown format');
          return;
        }
        console.log('[Import] Done:', stats);
        // Refresh sidebar
        await this.refreshConversations();
        this._showImportResult(stats);
      } catch (err) {
        console.error('[Import] Failed:', err);
      }
    });
    input.click();
  },

  /** Import v2 format (full backup with all stores). */
  async _importV2(data) {
    const db = this.idb.db;
    const stats = { added: 0, updated: 0, skipped: 0 };
    for (const [storeName, records] of Object.entries(data._stores)) {
      if (!db.objectStoreNames.contains(storeName) || !records?.length) continue;
      // Determine key path for this store
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const keyPath = store.keyPath;
      for (const record of records) {
        const key = record[keyPath];
        if (!key) continue;
        // Check existing
        const existing = await new Promise((res) => {
          const req = store.get(key);
          req.onsuccess = () => res(req.result);
          req.onerror = () => res(null);
        });
        if (existing) {
          // Compare by updated_at — newer wins
          const existingTime = existing.updated_at || existing.created_at || '';
          const importTime = record.updated_at || record.created_at || '';
          if (importTime > existingTime) {
            store.put(record);
            stats.updated++;
          } else {
            stats.skipped++;
          }
        } else {
          store.put(record);
          stats.added++;
        }
      }
      await new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = rej; });
    }
    // Rebuild conv_meta from conversations (in case it was stale)
    if (data._stores.conversations) {
      const tx = db.transaction(['conversations', 'conv_meta'], 'readwrite');
      const convStore = tx.objectStore('conversations');
      const metaStore = tx.objectStore('conv_meta');
      const all = await new Promise((res) => {
        const req = convStore.getAll();
        req.onsuccess = () => res(req.result);
        req.onerror = () => res([]);
      });
      for (const conv of all) {
        metaStore.put(IDBStore._extractMeta(conv));
      }
      await new Promise((res) => { tx.oncomplete = res; });
    }
    return stats;
  },

  /** Import legacy v1 format (plain conversation array). */
  async _importV1(convs) {
    const stats = { added: 0, updated: 0, skipped: 0 };
    for (const conv of convs) {
      if (!conv.id) continue;
      const existing = await this.idb.get(conv.id);
      if (existing) {
        const existingTime = existing.updated_at || existing.created_at || '';
        const importTime = conv.updated_at || conv.created_at || '';
        if (importTime > existingTime) {
          await this.idb.put(conv);
          stats.updated++;
        } else {
          stats.skipped++;
        }
      } else {
        await this.idb.put(conv);
        stats.added++;
      }
    }
    return stats;
  },

  _showImportResult(stats) {
    const total = stats.added + stats.updated;
    const msg = total > 0
      ? `Import complete: ${stats.added} added, ${stats.updated} updated, ${stats.skipped} unchanged.`
      : `Nothing to import — all ${stats.skipped} items are already up to date.`;
    if (this._showNotificationDialog) {
      this._showNotificationDialog(msg);
    } else {
      console.log('[Import]', msg);
    }
  },

  // ── Clear all dialog ──────────────────────────────────

  _showClearAllDialog() {
    const count = this.conversations.length;
    if (!count) return;
    const overlay = document.createElement('div');
    overlay.className = 'cv2-dialog-overlay';
    overlay.innerHTML = `
      <div class="cv2-dialog">
        <div class="cv2-dialog-title">${this.t('chat.delete_all_title')}</div>
        <div class="cv2-dialog-body">This will permanently delete ${count} conversation${count !== 1 ? 's' : ''}. This action cannot be undone.</div>
        <div class="cv2-dialog-actions">
          <button class="cv2-dialog-btn cv2-dialog-cancel">${this.t('chat.cancel')}</button>
          <button class="cv2-dialog-btn cv2-dialog-confirm">${this.t('chat.delete_all')}</button>
        </div>
      </div>`;
    document.getElementById('chat-app').appendChild(overlay);
    overlay.querySelector('.cv2-dialog-cancel').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector('.cv2-dialog-confirm').addEventListener('click', () => {
      overlay.remove();
      this._clearAllConversations();
    });
  },

  });

  ChatFeatures.register('sidebar', {
    initState(app) {
      app.conversations = [];
      app.activeConvId = '';
      app.projects = [];
      app.nudges = [];
      app.favoriteNudges = [];
      app._activeProjectId = null;
      app._activeNudgeId = null;
      app._activeNudgeMeta = null;
      app._expandedProjects = new Set();
      app._collapsedSections = new Set(JSON.parse(localStorage.getItem('cv2-collapsed-sections') || '[]'));
      app._sectionPage = { nudges: 1, projects: 1, chats: 1 };
      app._presetDeleteConfirmId = null;
    },
  });
})();
