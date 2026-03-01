/**
 * chat-nudges.js — Nudge explorer, favorites, search
 * Extracted from chat-app.js
 */
(function() {
  Object.assign(window._ChatAppProto, {

  _canEditNudge(d) {
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
    // Clear pending attachments from previous context (e.g. new chat)
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
  },

  handleNudgeDeleted(msg) {
    this._showToast('Deleted', 'info');
    this._removeFavorite(msg.uid);
  },

  handleNudgeFlushed(msg) {
    this._showToast(this.t('chat.nudge_flushed') || 'Published', 'positive');
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
      { key: '', label: this.t('chat.nudge_all') || 'All', icon: '', largeIcon: '' },
      ...categories.map(c => ({ key: c.key, label: c.label, icon: c.icon || '', largeIcon: c.largeIcon || '' })),
      { key: '__mine__', label: this.t('chat.nudge_mine') || 'My Nudges', icon: '', largeIcon: '' },
    ];
    const tabsHtml = tabs.map((tab, i) => {
      // If a largeIcon exists, show it AS the tab content (no text label)
      if (tab.largeIcon) {
        return `<button class="cv2-explorer-tab ${i === 0 ? 'cv2-active' : ''}" data-category="${this._escAttr(tab.key)}" title="${this._escAttr(tab.label)}"><img src="${this._escAttr(tab.largeIcon)}" class="cv2-explorer-tab-logo" alt="${this._escAttr(tab.label)}"></button>`;
      }
      const iconHtml = tab.icon
        ? `<img src="${this._escAttr(tab.icon)}" class="cv2-explorer-tab-icon" alt="">`
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
          ${this.config.isAdmin ? `<label class="cv2-explorer-admin-toggle"><input type="checkbox" id="cv2-explorer-all-users"> All users</label>` : ''}
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
      const isAdmin = this.config.isAdmin;
      const html = nudges.map(d => {
        const iconHtml = d.icon
          ? `<img src="${d.icon}" class="cv2-explorer-card-icon" alt="">`
          : '<span class="material-icons cv2-explorer-card-icon-default">science</span>';
        const canEdit = this._canEditNudge(d);
        const isFav = this.favoriteNudges.some(f => f.uid === d.uid);
        const isMaster = isAdmin && d.is_master;
        const masterClass = isMaster ? ' cv2-explorer-card-master' : '';
        const masterBadge = isMaster ? '<span class="material-icons cv2-master-badge">auto_awesome</span>' : '';
        // In "My Nudges" tab: show BOTH edit AND heart buttons
        const editBtn = (isMine && canEdit)
          ? `<button class="cv2-explorer-card-edit" data-uid="${this._escAttr(d.uid)}" title="Edit"><span class="material-icons" style="font-size:16px">edit</span></button>`
          : '';
        const heartBtn = `<button class="cv2-explorer-card-star ${isFav ? 'cv2-fav' : ''}" data-uid="${this._escAttr(d.uid)}" title="Favorite"><span class="material-icons" style="font-size:16px">${isFav ? 'favorite' : 'favorite_border'}</span></button>`;
        return `
          <div class="cv2-explorer-card${masterClass}" data-uid="${this._escAttr(d.uid)}">
            ${masterBadge}
            ${iconHtml}
            <div class="cv2-explorer-card-info">
              <div class="cv2-explorer-card-name">${this._escHtml(d.name)}</div>
              <div class="cv2-explorer-card-desc" title="${this._escAttr(d.description || '')}">${this._escHtml(d.description || '')}</div>
              <span class="cv2-explorer-card-by">${this._escHtml(d.creator_name || '')}</span>
            </div>
            <div class="cv2-explorer-card-actions">
              ${editBtn}${heartBtn}
            </div>
          </div>`;
      }).join('');
      grid.insertAdjacentHTML('beforeend', html);
      loadMoreWrap.style.display = hasMore ? '' : 'none';

      // Bind card events
      grid.querySelectorAll('.cv2-explorer-card:not([data-bound])').forEach(card => {
        card.setAttribute('data-bound', '1');
        const uid = card.dataset.uid;
        card.addEventListener('click', (e) => {
          if (e.target.closest('.cv2-explorer-card-star') || e.target.closest('.cv2-explorer-card-edit')) return;
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
      'nudge_meta': 'handleNudgeMeta',
      'nudge_favorites_validated': 'handleNudgeFavoritesValidated',
    },
  });
})();
