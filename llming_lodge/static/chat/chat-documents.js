/**
 * chat-documents.js — Document viewer side panel (VS-Code-style tabs).
 *
 * Documents created via the tool-only MCP pipeline (create_document /
 * update_document / per-type edit tools) never render inline inside the
 * chat transcript. Instead they open as tabs in the right-hand workspace
 * panel, with one doc per tab, multi-row flex-wrap when many are open,
 * and a material-icon per doc type.
 */
(function() {
  Object.assign(window._ChatAppProto, {

    // ── WS message handlers ──────────────────────────────────────────

    handleDocCreated(msg) {
      if (!msg.document) return;
      const doc = msg.document;
      if (!doc.conversation_id && this.activeConvId) {
        doc.conversation_id = this.activeConvId;
      }
      if (!this.documents.find(d => d.id === doc.id)) {
        this.documents.push(doc);
      }
      this.idb.putDocument(doc).catch(() => {});
      this._openDocTab(doc.id, { activate: true, autoOpenPanel: true });
      this._renderWorkspace();
    },

    handleDocUpdated(msg) {
      if (!msg.document) return;
      const doc = msg.document;
      if (!doc.conversation_id && this.activeConvId) {
        doc.conversation_id = this.activeConvId;
      }
      const idx = this.documents.findIndex(d => d.id === doc.id);
      if (idx >= 0) this.documents[idx] = doc;
      else this.documents.push(doc);
      this.idb.putDocument(doc).catch(() => {});
      this._openDocTab(doc.id, { activate: true, autoOpenPanel: true });
      this._renderWorkspace();
    },

    handleDocDeleted(msg) {
      if (!msg.document) return;
      const id = msg.document.id;
      this.documents = this.documents.filter(d => d.id !== id);
      this.idb.deleteDocument(id).catch(() => {});
      this._closeDocTab(id);
      this._renderWorkspace();
    },

    handleDocList(msg) {
      if (!msg.documents) return;
      this.documents = msg.documents;
      this._renderWorkspace();
    },

    // ── Tab state ────────────────────────────────────────────────────

    _openDocTab(docId, { activate = true, autoOpenPanel = false } = {}) {
      if (!this.openTabIds) this.openTabIds = [];
      if (!this.openTabIds.includes(docId)) this.openTabIds.push(docId);
      if (activate) this.activeTabId = docId;
      if (autoOpenPanel && !this.workspaceOpen) this._openWorkspace();
    },

    _closeDocTab(docId) {
      if (!this.openTabIds) this.openTabIds = [];
      const before = this.openTabIds.length;
      this.openTabIds = this.openTabIds.filter(id => id !== docId);
      if (this.activeTabId === docId) {
        this.activeTabId = this.openTabIds[this.openTabIds.length - 1] || null;
      }
      if (before && !this.openTabIds.length && this.workspaceOpen) {
        // All tabs closed — auto-collapse the panel.
        this._closeWorkspace();
      }
    },

    // ── Workspace panel open/close ──────────────────────────────────

    _openWorkspace() {
      this.workspaceOpen = true;
      this._ensureWorkspace();
      document.getElementById('cv2-workspace')?.classList.add('open');
      document.getElementById('cv2-workspace-toggle')?.classList.add('active');
    },

    _closeWorkspace() {
      if (!this.workspaceOpen) return;
      this.workspaceOpen = false;
      document.getElementById('cv2-workspace')?.classList.remove('open');
      document.getElementById('cv2-workspace-toggle')?.classList.remove('active');
    },

    toggleWorkspace() {
      if (this.workspaceOpen) { this._closeWorkspace(); return; }
      // Opening manually — auto-select the latest-updated doc if no tab is
      // active yet, so the user isn't greeted with an empty panel.
      if (!this.activeTabId && this.documents.length) {
        const latest = [...this.documents].sort(
          (a, b) => (b.updated_at || 0) - (a.updated_at || 0)
        )[0];
        if (latest) this._openDocTab(latest.id, { activate: true });
      }
      this._openWorkspace();
      this._renderWorkspace();
    },

    _ensureWorkspace() {
      if (document.getElementById('cv2-workspace')) return;
      const panel = document.createElement('div');
      panel.id = 'cv2-workspace';
      panel.className = 'cv2-workspace';
      panel.innerHTML = `
        <div class="cv2-workspace-header">
          <div class="cv2-workspace-tabs" role="tablist"></div>
          <button class="cv2-workspace-undock" title="Undock (float as window)" aria-label="Undock panel">
            <span class="material-icons">open_in_new</span>
          </button>
          <button class="cv2-workspace-close" title="Close panel">&times;</button>
        </div>
        <div class="cv2-workspace-body"></div>
      `;
      document.getElementById('chat-app')?.appendChild(panel);
      panel.querySelector('.cv2-workspace-close').addEventListener('click', () => this._closeWorkspace());
      panel.querySelector('.cv2-workspace-undock').addEventListener('click', () => this._toggleWorkspaceDock());
      this._bindWorkspaceDrag(panel);
    },

    /** Flip the panel between docked (flex sibling) and floating (position:
     *  fixed, draggable window hovering over the chat). On mobile the undock
     *  button is hidden, so this is a no-op for accidental taps. */
    _toggleWorkspaceDock() {
      const panel = document.getElementById('cv2-workspace');
      if (!panel) return;
      if (window.innerWidth <= 720) return;  // Ignore on mobile.
      const floating = panel.classList.toggle('cv2-workspace-floating');
      this.workspaceFloating = floating;
      const undockBtn = panel.querySelector('.cv2-workspace-undock .material-icons');
      if (undockBtn) {
        undockBtn.textContent = floating ? 'push_pin' : 'open_in_new';
        undockBtn.parentElement.title = floating ? 'Dock back to side' : 'Undock (float as window)';
      }
      if (!floating) {
        // Clear any drag-applied inline position so the docked layout wins.
        panel.style.left = '';
        panel.style.top = '';
        panel.style.right = '';
        panel.style.width = '';
        panel.style.height = '';
      }
    },

    /** Drag support for the floating panel. Attaches to the header via
     *  pointerdown, listens on document so the drag survives cursor
     *  excursions. No-op when the panel isn't floating (header stays static). */
    _bindWorkspaceDrag(panel) {
      const header = panel.querySelector('.cv2-workspace-header');
      if (!header) return;
      let startX = 0, startY = 0, origLeft = 0, origTop = 0;
      const onMove = (ev) => {
        const dx = ev.clientX - startX;
        const dy = ev.clientY - startY;
        panel.style.left = (origLeft + dx) + 'px';
        panel.style.top = (origTop + dy) + 'px';
        panel.style.right = 'auto';
      };
      const onUp = () => {
        panel.classList.remove('cv2-workspace-dragging');
        document.removeEventListener('pointermove', onMove);
        document.removeEventListener('pointerup', onUp);
      };
      header.addEventListener('pointerdown', (ev) => {
        // Only drag when floating, with primary button, and not on a tab/button.
        if (!panel.classList.contains('cv2-workspace-floating')) return;
        if (ev.button !== 0) return;
        if (ev.target.closest('.cv2-workspace-tab') ||
            ev.target.closest('.cv2-workspace-undock') ||
            ev.target.closest('.cv2-workspace-close')) return;
        const rect = panel.getBoundingClientRect();
        startX = ev.clientX;
        startY = ev.clientY;
        origLeft = rect.left;
        origTop = rect.top;
        panel.classList.add('cv2-workspace-dragging');
        document.addEventListener('pointermove', onMove);
        document.addEventListener('pointerup', onUp);
      });
    },

    // ── Rendering ────────────────────────────────────────────────────

    _renderWorkspace() {
      try {
        this._ensureWorkspace();
        this._renderTabStrip();
        this._renderActiveDoc();
      } catch (err) {
        console.error('[DOC] Workspace render crashed:', err);
      }
    },

    _renderTabStrip() {
      const strip = document.querySelector('#cv2-workspace .cv2-workspace-tabs');
      if (!strip) return;
      strip.innerHTML = '';
      const ids = this.openTabIds || [];
      // Fall back to "all documents" when no tabs are explicitly open but
      // docs exist (first render before handleDocCreated, or toggle-open
      // with existing docs) so the user has something to click.
      const displayIds = ids.length ? ids : this.documents.map(d => d.id);
      for (const id of displayIds) {
        const doc = this.documents.find(d => d.id === id);
        if (!doc) continue;
        const tab = document.createElement('div');
        tab.className = 'cv2-workspace-tab';
        if (id === this.activeTabId) tab.classList.add('active');
        tab.dataset.docId = id;
        tab.setAttribute('role', 'tab');
        const icon = ChatApp.DOC_ICONS[doc.type] || 'article';
        tab.innerHTML = `
          <span class="cv2-workspace-tab-icon material-icons">${icon}</span>
          <span class="cv2-workspace-tab-name" title="${this._escAttr(doc.name)}">${this._escHtml(doc.name || doc.type)}</span>
          ${doc.version ? `<span class="cv2-workspace-tab-version">v${doc.version}</span>` : ''}
          <button class="cv2-workspace-tab-close" title="Close tab" aria-label="Close">&times;</button>
        `;
        tab.addEventListener('click', (e) => {
          if (e.target.closest('.cv2-workspace-tab-close')) return;
          this.activeTabId = id;
          // If we were in fallback display mode (no explicit tabs), starting
          // to click promotes the selected doc into an explicit tab.
          if (!this.openTabIds?.includes(id)) {
            this._openDocTab(id, { activate: true });
          }
          this._renderWorkspace();
        });
        tab.querySelector('.cv2-workspace-tab-close').addEventListener('click', (e) => {
          e.stopPropagation();
          this._closeDocTab(id);
          this._renderWorkspace();
        });
        strip.appendChild(tab);
      }
    },

    _renderActiveDoc() {
      const body = document.querySelector('#cv2-workspace .cv2-workspace-body');
      if (!body) return;
      const activeId = this.activeTabId;
      const doc = activeId ? this.documents.find(d => d.id === activeId) : null;

      if (!doc) {
        body.innerHTML = `<div class="cv2-doc-panel-empty">${
          this.t('chat.no_documents') || 'No documents yet. Ask the AI to create one.'
        }</div>`;
        return;
      }

      // Preserve scroll position across re-render. Type-agnostic: plugins
      // that have an internal scroll container mark it with the
      // data-ldoc-scrollable attribute. If none, fall back to the panel body.
      const prevActiveId = this._lastRenderedDocId;
      const prevEditable = body.querySelector('[data-ldoc-scrollable]');
      const prevScrollTop = prevEditable ? prevEditable.scrollTop : body.scrollTop;
      const sameDoc = prevActiveId === doc.id;
      this._lastRenderedDocId = doc.id;

      // Build a fresh container for this render. We always rebuild so the
      // plugin can't be confused by stale child state when data changes.
      // Plugins that cache state (localStorage edits, in-memory layouts, …)
      // self-invalidate via the ldoc:invalidate-cache CustomEvent below —
      // the host never pokes plugin-internal keys.
      body.innerHTML = '';
      const container = document.createElement('div');
      container.className = 'cv2-doc-plugin-block cv2-doc-plugin-in-panel';
      // `ws-preview-` prefix matches the existing skip-filter in
      // chat-app-core.js onBlockRendered — without it, rendering the plugin
      // inside the panel would fire onBlockRendered → _renderDocList →
      // _renderWorkspace → _renderActiveDoc → (render again) → infinite loop.
      const blockId = `ws-preview-${doc.id}`;
      container.dataset.blockId = blockId;
      container.dataset.lang = doc.type;
      body.appendChild(container);

      // Type-agnostic cache invalidation: any plugin listening for this
      // event clears its per-doc cache before the re-render starts.
      document.dispatchEvent(new CustomEvent('ldoc:invalidate-cache', {
        detail: { docId: doc.id },
      }));

      if (!this.docPlugins?.has(doc.type)) {
        container.innerHTML = `<pre class="cv2-doc-plugin-error">No renderer for type ${doc.type}</pre>`;
        return;
      }
      const spec = { id: doc.id, name: doc.name };
      if (typeof doc.data === 'object' && doc.data !== null) Object.assign(spec, doc.data);
      const raw = JSON.stringify(spec);
      this.docPlugins.render(doc.type, container, raw, blockId).then(() => {
        // Restore scroll after the plugin has populated its DOM.
        if (!sameDoc || prevScrollTop === 0) return;
        requestAnimationFrame(() => {
          const editable = container.querySelector('[data-ldoc-scrollable]');
          const target = editable || body;
          // Clamp to max scrollable so grown content doesn't jump past end.
          target.scrollTop = Math.min(prevScrollTop, target.scrollHeight - target.clientHeight);
        });
      }).catch(err => {
        console.error('[DOC] Workspace render failed:', err);
        container.innerHTML = `<pre class="cv2-doc-plugin-error">Error rendering ${doc.type}: ${err.message}</pre>`;
      });
    },

    /** Seed tabs from the restored `this.documents` on conversation switch.
     *  All docs open as tabs, the latest-updated one becomes active, and
     *  the panel auto-opens so the user lands directly in the doc view. */
    _restoreWorkspaceTabs() {
      try {
        this.openTabIds = (this.documents || []).map(d => d.id);
        if (this.documents?.length) {
          const latest = [...this.documents].sort(
            (a, b) => (b.updated_at || 0) - (a.updated_at || 0)
          )[0];
          this.activeTabId = latest?.id || null;
          if (this.activeTabId) this._openWorkspace();
        } else {
          this.activeTabId = null;
        }
      } catch (err) {
        console.error('[DOC] _restoreWorkspaceTabs crashed:', err);
        this.openTabIds = [];
        this.activeTabId = null;
      }
    },

    // ── Compatibility shims — some callers still invoke the old name ─

    _renderDocList() { this._renderWorkspace(); },

  });

  ChatFeatures.register('documents', {
    initState(app) {
      app.documents = [];
      app.inlineDocBlocks = [];  // kept for non-doc plugins (mermaid, rich_mcp, etc.)
      app.openTabIds = [];
      app.activeTabId = null;
      app.workspaceOpen = false;
      app.workspaceFloating = false;
    },
    handleMessage: {
      'doc_created': 'handleDocCreated',
      'doc_updated': 'handleDocUpdated',
      'doc_deleted': 'handleDocDeleted',
      'doc_list': 'handleDocList',
    },
  });

  // ── WS → CustomEvent bridge for service messages ─────────────────
  Object.assign(window._ChatAppProto, {
    _onDirectorySearchResult(msg) {
      document.dispatchEvent(new CustomEvent('ws:directory:search_result', { detail: msg }));
    },
    _onEmailActionResult(msg) {
      document.dispatchEvent(new CustomEvent('ws:email:action_result', { detail: msg }));
    },
  });

  ChatFeatures.register('directory', {
    handleMessage: { 'directory:search_result': '_onDirectorySearchResult' },
  });
  ChatFeatures.register('email_service', {
    handleMessage: { 'email:action_result': '_onEmailActionResult' },
  });
})();
