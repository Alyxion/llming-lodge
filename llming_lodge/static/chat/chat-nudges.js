/**
 * chat-nudges.js — Nudge explorer, favorites, search
 * Extracted from chat-app.js
 */
(function() {
  Object.assign(window._ChatAppProto, {

  /** Resolve a nudge field (name, description, suggestions) for the user's locale. */
  _localizedNudge(d, field) {
    const locale = (this.config.locale || 'en-us').toLowerCase();
    const tr = d.translations;
    if (tr && locale !== 'en-us') {
      // Try exact match, then language prefix (e.g. "de" for "de-de")
      const val = tr[locale]?.[field] ?? tr[locale.split('-')[0]]?.[field];
      if (val) return val;
    }
    return d[field];
  },

  _canEditNudge(d) {
    if (d.is_system) return false;
    if (this.hasPerm('nudge_admin')) return true;
    const email = (this.config.userEmail || '').toLowerCase();
    if ((d.creator_email || '').toLowerCase() === email && !d.team_id) return true;
    if (d.team_id) {
      return this.teams.some(t =>
        (t.teamId || t.team_id) === d.team_id && (t.role === 'owner' || t.role === 'editor')
      );
    }
    return false;
  },

  _startNudgeChat(nudge) {
    // Clear pending state from previous context (e.g. new chat)
    this._pendingFirstMsgHtml = null;
    this._pendingFiles.length = 0;
    this._pendingImages.length = 0;
    this._renderAttachments();

    // Server-side nudge flow: send only uid, server fetches everything
    if (nudge.uid) {
      this.ws.send({
        type: 'new_chat',
        preset: { nudge_uid: nudge.uid, type: 'nudge' },
      });
      return;
    }

    // Legacy IDB flow
    this.ws.send({
      type: 'new_chat',
      preset: {
        id: nudge.id,
        type: 'nudge',
        system_prompt: nudge.system_prompt,
        model: nudge.model,
        language: nudge.language,
        files: this._wsFiles(nudge.files),
        doc_plugins: nudge.doc_plugins ?? null,
      },
    });
    // Apply per-nudge capability toggles
    const caps = nudge.capabilities;
    if (caps && typeof caps === 'object') {
      for (const [name, enabled] of Object.entries(caps)) {
        const tool = this.tools.find(t => t.name === name);
        if (tool && tool.enabled !== enabled) {
          tool.enabled = enabled;
          this.ws.send({ type: 'toggle_tool', name, enabled });
        }
      }
      this._renderPlusMenu();
    }
  },

  // ── Nudge Handlers (MongoDB) ──────────────────────────

  handleNudgeFavorites(msg) {
    // Favorites are now stored locally in IndexedDB — ignore server-sent list
  },

  /** Toggle a nudge's favorite status in local IndexedDB. */
  async _toggleFavorite(uid, nudgeMeta = null) {
    const isFav = this.favoriteNudges.some(f => f.uid === uid);
    if (isFav) {
      // Remove
      this.favoriteNudges = this.favoriteNudges.filter(f => f.uid !== uid);
      try { await this.idb.deleteFavorite(uid); } catch (e) { console.warn('[FAV] IDB delete error:', e); }
    } else {
      // Add — need metadata. Use provided meta or create minimal stub.
      const meta = nudgeMeta || { uid };
      this.favoriteNudges.push(meta);
      try { await this.idb.putFavorite(meta); } catch (e) { console.warn('[FAV] IDB put error:', e); }
    }
    this._renderSidebar();
    return !isFav; // new state
  },

  /** Remove a nudge from favorites (IndexedDB + in-memory). */
  async _removeFavorite(uid) {
    this.favoriteNudges = this.favoriteNudges.filter(f => f.uid !== uid);
    try { await this.idb.deleteFavorite(uid); } catch (e) { console.warn('[FAV] IDB delete error:', e); }
    this._renderSidebar();
  },

  handleNudgeSearchResult(msg) {
    if (this._explorerResolve) {
      this._explorerResolve(msg);
      this._explorerResolve = null;
    }
  },

  handleNudgeDetail(msg) {
    if (this._nudgeDetailResolve) {
      this._nudgeDetailResolve(msg.nudge);
      this._nudgeDetailResolve = null;
    }
  },

  handleNudgeSaved(msg) {
    this._showToast(this.t('chat.nudge_saved') || 'Saved', 'positive');
    // Auto-favorite saved nudge if it has metadata
    if (msg.nudge?.uid) {
      const alreadyFav = this.favoriteNudges.some(f => f.uid === msg.nudge.uid);
      if (alreadyFav) {
        // Update metadata in case name/icon/description changed
        this.favoriteNudges = this.favoriteNudges.map(f =>
          f.uid === msg.nudge.uid ? { ...f, ...msg.nudge } : f
        );
        this.idb.putFavorite(msg.nudge).catch(() => {});
      }
    }
    this._renderSidebar();
    // Update pub status bar in open editor (dev save = unpublished changes)
    this._updatePubStatusBar('changed', msg.nudge?.version);
  },

  handleNudgeDeleted(msg) {
    this._showToast('Deleted', 'info');
    this._removeFavorite(msg.uid);
  },

  handleNudgeFlushed(msg) {
    this._showToast(this.t('chat.nudge_flushed') || 'Published', 'positive');
    this._updatePubStatusBar('published');
  },

  handleNudgeUnpublished(msg) {
    this._showToast(this.t('chat.nudge_unpublish') || 'Unpublished', 'info');
    this._updatePubStatusBar('draft');
  },

  /**
   * Update the publication status bar in an open nudge editor.
   * @param {'draft'|'published'|'changed'} state
   * @param {string} [version]
   */
  _updatePubStatusBar(state, version) {
    const bar = document.querySelector('.cv2-pub-status-bar');
    if (!bar) return;
    const ver = version || '';
    bar.className = 'cv2-pub-status-bar cv2-pub-' + state;
    const verHtml = ver ? '<span class="cv2-pub-ver">v' + this._escHtml(ver) + '</span>' : '';
    const textEl = bar.querySelector('.cv2-pub-text');
    const actEl = bar.querySelector('.cv2-pub-actions');
    if (state === 'draft') {
      if (textEl) textEl.innerHTML = (this.t('chat.nudge_draft') || 'Draft — not published') + verHtml;
      if (actEl) actEl.innerHTML = '<button class="cv2-dialog-btn cv2-notification-ok cv2-pub-action" id="cv2-pub-publish"><span class="material-icons" style="font-size:16px;margin-right:4px">publish</span>' + (this.t('chat.nudge_flush') || 'Publish') + '</button>';
    } else if (state === 'published') {
      if (textEl) textEl.innerHTML = (this.t('chat.nudge_published') || 'Published') + verHtml;
      if (actEl) actEl.innerHTML = '<button class="cv2-pub-link" id="cv2-pub-unpublish">' + (this.t('chat.nudge_unpublish') || 'Unpublish') + '</button>';
    } else {
      if (textEl) textEl.innerHTML = (this.t('chat.nudge_unpublished_changes') || 'Unpublished changes') + verHtml;
      if (actEl) actEl.innerHTML = '<button class="cv2-pub-link" id="cv2-pub-revert">' + (this.t('chat.nudge_revert') || 'Revert') + '</button>'
        + '<button class="cv2-dialog-btn cv2-notification-ok cv2-pub-action" id="cv2-pub-publish"><span class="material-icons" style="font-size:16px;margin-right:4px">publish</span>' + (this.t('chat.nudge_flush') || 'Publish') + '</button>';
    }
    // Re-bind publish/unpublish/revert click handlers
    this._rebindPubActions(bar);
  },

  _rebindPubActions(bar) {
    const uid = this._openNudgeUid;
    if (!uid) return;
    bar.querySelector('#cv2-pub-publish')?.addEventListener('click', () => {
      const dlg = document.createElement('div');
      dlg.className = 'cv2-dialog-overlay';
      dlg.innerHTML = `<div class="cv2-dialog"><div class="cv2-dialog-body">${this.t('chat.nudge_flush_confirm') || 'Publish dev to live?'}</div><div class="cv2-dialog-actions"><button class="cv2-dialog-btn cv2-dialog-cancel">${this.t('chat.cancel')}</button><button class="cv2-dialog-btn cv2-dialog-confirm">${this.t('chat.nudge_flush') || 'Publish'}</button></div></div>`;
      document.getElementById('chat-app').appendChild(dlg);
      dlg.querySelector('.cv2-dialog-cancel').addEventListener('click', () => dlg.remove());
      dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.remove(); });
      dlg.querySelector('.cv2-dialog-confirm').addEventListener('click', () => { dlg.remove(); this.ws.send({ type: 'nudge_flush', uid }); });
    });
    bar.querySelector('#cv2-pub-unpublish')?.addEventListener('click', () => {
      const dlg = document.createElement('div');
      dlg.className = 'cv2-dialog-overlay';
      dlg.innerHTML = `<div class="cv2-dialog"><div class="cv2-dialog-body">${this.t('chat.nudge_unpublish_confirm') || 'Remove the published version?'}</div><div class="cv2-dialog-actions"><button class="cv2-dialog-btn cv2-dialog-cancel">${this.t('chat.cancel')}</button><button class="cv2-dialog-btn cv2-dialog-confirm">${this.t('chat.nudge_unpublish') || 'Unpublish'}</button></div></div>`;
      document.getElementById('chat-app').appendChild(dlg);
      dlg.querySelector('.cv2-dialog-cancel').addEventListener('click', () => dlg.remove());
      dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.remove(); });
      dlg.querySelector('.cv2-dialog-confirm').addEventListener('click', () => { dlg.remove(); this.ws.send({ type: 'nudge_unpublish', uid }); });
    });
    bar.querySelector('#cv2-pub-revert')?.addEventListener('click', () => {
      const dlg = document.createElement('div');
      dlg.className = 'cv2-dialog-overlay';
      dlg.innerHTML = `<div class="cv2-dialog"><div class="cv2-dialog-body">${this.t('chat.nudge_revert_confirm') || 'Revert dev to the published version?'}</div><div class="cv2-dialog-actions"><button class="cv2-dialog-btn cv2-dialog-cancel">${this.t('chat.cancel')}</button><button class="cv2-dialog-btn cv2-dialog-confirm">${this.t('chat.nudge_revert') || 'Revert'}</button></div></div>`;
      document.getElementById('chat-app').appendChild(dlg);
      dlg.querySelector('.cv2-dialog-cancel').addEventListener('click', () => dlg.remove());
      dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.remove(); });
      dlg.querySelector('.cv2-dialog-confirm').addEventListener('click', () => { dlg.remove(); this.ws.send({ type: 'nudge_revert', uid }); });
    });
  },

  handleNudgeReverted(msg) {
    this._showToast(this.t('chat.nudge_reverted') || 'Reverted to published version', 'positive');
  },

  handleNudgeFavoritesValidated(msg) {
    const validSet = new Set(msg.valid_uids || []);
    const removed = this.favoriteNudges.filter(f => f.uid && !validSet.has(f.uid));
    if (removed.length > 0) {
      this.favoriteNudges = this.favoriteNudges.filter(f => !f.uid || validSet.has(f.uid));
      removed.forEach(f => this.idb.deleteFavorite(f.uid).catch(() => {}));
      this._renderSidebar();
    }
  },

  handleNudgeMeta(msg) {
    // Server sent nudge metadata after loading a conversation
    const nudge = msg.nudge;
    if (nudge) {
      this._activeNudgeMeta = nudge;
      this._renderNudgeChatHeader(nudge);
      this._lockModelForNudge(nudge);
    }
  },

  _renderNudgeExplorer() {
    const mainEl = document.querySelector('.cv2-main');
    const chatWrapper = mainEl.querySelector('.cv2-chat-wrapper');

    // Tear down any other active overlay (preset editor, previous explorer) via view manager
    this._switchView('explorer');

    // Keep topbar visible — only hide chat content
    if (chatWrapper) chatWrapper.style.display = 'none';

    const categories = this.config.nudgeCategories || [];
    const overlay = document.createElement('div');
    overlay.className = 'cv2-nudge-explorer';

    // Tab order: All first, configured categories (sorted by priority from server), My Nudges last
    const tabs = [
      { key: '', label: this.t('chat.nudge_all') || 'All', icon: 'apps', largeIcon: '' },
      ...categories.map(c => ({ key: c.key, label: c.label, icon: c.icon || '', largeIcon: c.largeIcon || '' })),
      { key: '__mine__', label: this.t('chat.nudge_mine') || 'My Droplets', icon: 'person', largeIcon: '' },
    ];
    const tabsHtml = tabs.map((tab, i) => {
      // If a largeIcon exists, show it AS the tab content (no text label)
      if (tab.largeIcon) {
        return `<button class="cv2-explorer-tab ${i === 0 ? 'cv2-active' : ''}" data-category="${this._escAttr(tab.key)}" title="${this._escAttr(tab.label)}"><img src="${this._escAttr(tab.largeIcon)}" class="cv2-explorer-tab-logo" alt="${this._escAttr(tab.label)}"></button>`;
      }
      const iconHtml = tab.icon
        ? (this._isIconUrl(tab.icon)
            ? `<img src="${this._escAttr(tab.icon)}" class="cv2-explorer-tab-icon" alt="">`
            : `<span class="material-icons" style="font-size:16px;vertical-align:middle;margin-right:4px">${this._escHtml(tab.icon)}</span>`)
        : '';
      return `<button class="cv2-explorer-tab ${i === 0 ? 'cv2-active' : ''}" data-category="${this._escAttr(tab.key)}">${iconHtml}${this._escHtml(tab.label)}</button>`;
    }).join('');

    const nudgeIcon = this.config.nudgeSectionIcon || 'icons/phosphor/regular/drop.svg';
    const explorerLogo = `<img src="${this.config.staticBase}/${nudgeIcon}" class="cv2-explorer-header-logo" alt="">`;
    overlay.innerHTML = `
      <div class="cv2-explorer-panel">
        <div class="cv2-explorer-header">
          <button class="cv2-preset-back" id="cv2-explorer-back"><span class="material-icons">arrow_back</span></button>
          ${explorerLogo}
          <span class="cv2-preset-header-title" id="cv2-explorer-title">${this.t('chat.explore_nudges') || 'Explore'}</span>
          <div style="flex:1"></div>
          ${this.hasPerm('nudge_admin') ? `<label class="cv2-explorer-admin-toggle"><input type="checkbox" id="cv2-explorer-all-users"> All users</label>` : ''}
          <button class="cv2-dialog-btn cv2-notification-ok" id="cv2-explorer-new">
            <span class="material-icons" style="font-size:16px;margin-right:4px">add</span>${this.t('chat.new_nudge')}
          </button>
        </div>
        <div class="cv2-explorer-search">
          <span class="material-icons" style="font-size:18px;color:var(--chat-text-muted)">search</span>
          <input type="text" class="cv2-explorer-search-input" id="cv2-explorer-search" placeholder="${this._escAttr(this.t('chat.nudge_search') || 'Search...')}">
        </div>
        <div class="cv2-explorer-tabs">${tabsHtml}</div>
        <div class="cv2-explorer-grid" id="cv2-explorer-grid"></div>
        <div class="cv2-explorer-load-more" id="cv2-explorer-load-more" style="display:none">
          <button class="cv2-show-more" id="cv2-explorer-more-btn">${this.t('chat.nudge_load_more') || 'Load more'}</button>
        </div>
      </div>`;

    mainEl.appendChild(overlay);

    let currentCategory = '';
    let currentQuery = '';
    let currentPage = 0;
    let allUsers = false;
    let searchTimeout = null;
    const grid = overlay.querySelector('#cv2-explorer-grid');
    const loadMoreWrap = overlay.querySelector('#cv2-explorer-load-more');
    const titleEl = overlay.querySelector('#cv2-explorer-title');

    const closeExplorer = () => {
      overlay.remove();
      if (chatWrapper) chatWrapper.style.display = '';
      this._activeView = 'chat';
      this._closeActiveView = null;
    };

    // Register as the active overlay teardown
    this._closeActiveView = closeExplorer;

    const doSearch = (append = false) => {
      if (!append) {
        currentPage = 0;
        grid.innerHTML = `<div class="cv2-explorer-loading"><span class="material-icons cv2-spin">hourglass_empty</span></div>`;
      }
      const isMine = currentCategory === '__mine__';
      const category = isMine ? '' : currentCategory;
      const payload = {
        type: 'nudge_search',
        query: currentQuery,
        category,
        mine: isMine,
        page: currentPage,
      };
      if (allUsers) payload.all_users = true;
      this.ws.send(payload);
      // Wait for response
      return new Promise(resolve => {
        this._explorerResolve = resolve;
      });
    };

    const nudgeMap = {};  // uid → nudge metadata for favorite toggling
    const renderResults = (nudges, hasMore, append) => {
      const isMine = currentCategory === '__mine__';
      if (!append) grid.innerHTML = '';
      if (nudges.length === 0 && !append) {
        grid.innerHTML = `<div class="cv2-explorer-empty">${this.t('chat.nudge_no_results') || 'No nudges found'}</div>`;
        loadMoreWrap.style.display = 'none';
        return;
      }
      const userEmail = this.config.userEmail || '';
      nudges.forEach(d => { nudgeMap[d.uid] = d; });
      const isAdmin = this.hasPerm('nudge_admin');
      const html = nudges.map(d => {
        const _clrIdx = (d.name || d.uid || '').split('').reduce((a, c) => a + c.charCodeAt(0), 0) % 8;
        const _defaultCls = 'cv2-explorer-card-icon-default cv2-icon-clr-' + _clrIdx;
        const iconHtml = this._renderIcon(d.icon, d.icon && this._isIconUrl(d.icon) ? 'cv2-explorer-card-icon' : _defaultCls, 'science');
        const canEdit = this._canEditNudge(d);
        const isFav = this.favoriteNudges.some(f => f.uid === d.uid);
        const isMaster = isAdmin && d.is_master;
        const isSystem = !!d.is_system;
        const masterClass = isMaster ? ' cv2-explorer-card-master' : (isSystem ? ' cv2-explorer-card-system' : '');
        const masterBadge = isMaster ? '<span class="material-icons cv2-master-badge">auto_awesome</span>'
          : (isSystem ? '<span class="material-icons cv2-system-badge">build</span>' : '');
        const editBtn = (!isSystem && (canEdit || isAdmin))
          ? `<button class="cv2-explorer-card-edit" data-uid="${this._escAttr(d.uid)}" title="Edit"><span class="material-icons" style="font-size:16px">edit</span></button>`
          : '';
        const heartBtn = `<button class="cv2-explorer-card-star ${isFav ? 'cv2-fav' : ''}" data-uid="${this._escAttr(d.uid)}" title="Favorite"><span class="material-icons" style="font-size:16px">${isFav ? 'favorite' : 'favorite_border'}</span></button>`;
        const copyUidBtn = (canEdit || isAdmin)
          ? `<button class="cv2-explorer-card-copy-uid" data-uid="${this._escAttr(d.uid)}" title="Copy UID"><span class="material-icons" style="font-size:16px">link</span></button>`
          : '';
        return `
          <div class="cv2-explorer-card${masterClass}" data-uid="${this._escAttr(d.uid)}">
            ${masterBadge}
            ${iconHtml}
            <div class="cv2-explorer-card-info">
              <div class="cv2-explorer-card-name">${this._escHtml(this._localizedNudge(d, 'name') || d.name)}</div>
              <div class="cv2-explorer-card-desc" title="${this._escAttr(this._localizedNudge(d, 'description') || d.description || '')}">${this._escHtml(this._localizedNudge(d, 'description') || d.description || '')}</div>
              <span class="cv2-explorer-card-by">${d.is_system ? '' : this._escHtml(d.creator_name || '')}</span>
            </div>
            <div class="cv2-explorer-card-actions">
              ${copyUidBtn}${editBtn}${heartBtn}
            </div>
            <span class="cv2-explorer-card-ver">v${this._escHtml(d.version || '0.0.1')}</span>
          </div>`;
      }).join('');
      grid.insertAdjacentHTML('beforeend', html);
      loadMoreWrap.style.display = hasMore ? '' : 'none';

      // Bind card events
      grid.querySelectorAll('.cv2-explorer-card:not([data-bound])').forEach(card => {
        card.setAttribute('data-bound', '1');
        const uid = card.dataset.uid;
        card.addEventListener('click', (e) => {
          if (e.target.closest('.cv2-explorer-card-star') || e.target.closest('.cv2-explorer-card-edit') || e.target.closest('.cv2-explorer-card-copy-uid')) return;
          closeExplorer();
          this._startNudgeChat({ uid });
        });
        card.querySelector('.cv2-explorer-card-star')?.addEventListener('click', (e) => {
          e.stopPropagation();
          this._toggleFavorite(uid, nudgeMap[uid] || { uid });
          // Toggle visual
          const star = card.querySelector('.cv2-explorer-card-star');
          if (star) {
            star.classList.toggle('cv2-fav');
            star.querySelector('.material-icons').textContent = star.classList.contains('cv2-fav') ? 'favorite' : 'favorite_border';
          }
        });
        card.querySelector('.cv2-explorer-card-copy-uid')?.addEventListener('click', (e) => {
          e.stopPropagation();
          const btn = e.currentTarget;
          navigator.clipboard.writeText(uid).then(() => {
            btn.querySelector('.material-icons').textContent = 'check';
            btn.style.color = '#10b981';
            setTimeout(() => { btn.querySelector('.material-icons').textContent = 'link'; btn.style.color = ''; }, 2000);
          }).catch(() => {});
        });
        card.querySelector('.cv2-explorer-card-edit')?.addEventListener('click', (e) => {
          e.stopPropagation();
          // Fetch full nudge data and open editor
          this.ws.send({ type: 'nudge_get', uid });
          this._nudgeDetailResolve = null;
          new Promise(resolve => { this._nudgeDetailResolve = resolve; }).then(nudge => {
            if (nudge) {
              closeExplorer();
              this._renderPresetDialog(nudge, 'nudge');
            }
          });
        });
      });
    };

    // Initial search
    doSearch().then(msg => renderResults(msg.nudges, msg.has_more, false));

    // Tab clicks
    overlay.querySelectorAll('.cv2-explorer-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        overlay.querySelectorAll('.cv2-explorer-tab').forEach(t => t.classList.remove('cv2-active'));
        tab.classList.add('cv2-active');
        currentCategory = tab.dataset.category;
        doSearch().then(msg => renderResults(msg.nudges, msg.has_more, false));
      });
    });

    // Search input
    overlay.querySelector('#cv2-explorer-search').addEventListener('input', (e) => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => {
        currentQuery = e.target.value.trim();
        doSearch().then(msg => renderResults(msg.nudges, msg.has_more, false));
      }, 300);
    });

    // Load more
    overlay.querySelector('#cv2-explorer-more-btn').addEventListener('click', () => {
      currentPage++;
      doSearch(true).then(msg => renderResults(msg.nudges, msg.has_more, true));
    });

    // Admin: all-users checkbox
    const allUsersCheck = overlay.querySelector('#cv2-explorer-all-users');
    if (allUsersCheck) {
      allUsersCheck.addEventListener('change', () => {
        allUsers = allUsersCheck.checked;
        titleEl.textContent = allUsers ? 'All Droplets (Admin)' : (this.t('chat.explore_nudges') || 'Explore');
        doSearch().then(msg => renderResults(msg.nudges, msg.has_more, false));
      });
    }

    // Back
    overlay.querySelector('#cv2-explorer-back').addEventListener('click', closeExplorer);

    // New nudge
    overlay.querySelector('#cv2-explorer-new').addEventListener('click', () => {
      closeExplorer();
      this._renderPresetDialog(null, 'nudge');
    });
  },

  });

  ChatFeatures.register('nudges', {
    handleMessage: {
      'nudge_favorites_result': 'handleNudgeFavorites',
      'nudge_search_result': 'handleNudgeSearchResult',
      'nudge_detail': 'handleNudgeDetail',
      'nudge_saved': 'handleNudgeSaved',
      'nudge_deleted': 'handleNudgeDeleted',
      'nudge_flushed': 'handleNudgeFlushed',
      'nudge_unpublished': 'handleNudgeUnpublished',
      'nudge_reverted': 'handleNudgeReverted',
      'nudge_meta': 'handleNudgeMeta',
      'nudge_favorites_validated': 'handleNudgeFavoritesValidated',
    },
  });
})();
