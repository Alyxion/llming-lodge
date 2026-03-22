/**
 * chat-documents.js — Document list sidebar & doc plugin integration
 * The sidebar shows a document list only; clicking a doc opens it
 * maximized in the unified preview popover (builtin-plugins.js).
 */
(function() {
  Object.assign(window._ChatAppProto, {

    handleDocCreated(msg) {
      if (!msg.document) return;
      const doc = msg.document;
      if (!this.documents.find(d => d.id === doc.id)) {
        this.documents.push(doc);
      }
      this.idb.putDocument(doc).catch(() => {});
      this._renderDocList();
      // If the doc was created via tool call during streaming and there's no
      // inline block for it yet, inject one into the current assistant message.
      this._injectToolDocBlock(doc);
    },

    /**
     * Inject a doc block into the current assistant message when the LLM used
     * create_document (tool) instead of a fenced code block.
     */
    _injectToolDocBlock(doc) {
      if (!doc || !doc.type || !doc.data) return;
      // Only inject for types the plugin registry can render
      if (!this.docPlugins || !this.docPlugins.has(doc.type)) return;
      // Skip if an inline block already exists for this doc id
      if (this.inlineDocBlocks?.find(b => b.id === doc.id)) return;

      // Find the latest assistant bubble
      const bubbles = document.querySelectorAll('.cv2-msg-assistant');
      const bubble = bubbles.length ? bubbles[bubbles.length - 1] : null;
      if (!bubble) return;
      let contentEl = bubble.querySelector('.cv2-msg-content');
      if (!contentEl) {
        // Tool doc arrives before text streaming starts — create content div
        contentEl = document.createElement('div');
        contentEl.className = 'cv2-msg-content';
        bubble.appendChild(contentEl);
      }

      // Build the block
      const blockId = `dp-tool-${doc.id}-${Date.now()}`;
      const container = document.createElement('div');
      container.className = 'cv2-doc-plugin-block';
      container.dataset.blockId = blockId;
      container.dataset.lang = doc.type;
      contentEl.appendChild(container);

      // Build the full spec (id + name + data)
      const spec = { id: doc.id, name: doc.name };
      if (typeof doc.data === 'object') Object.assign(spec, doc.data);
      const raw = JSON.stringify(spec);

      // Render the plugin into the container
      this.docPlugins.render(doc.type, container, raw, blockId).catch(err => {
        console.error('[DOC] Failed to inject tool doc block:', err);
      });
    },

    handleDocUpdated(msg) {
      if (!msg.document) return;
      const doc = msg.document;
      const idx = this.documents.findIndex(d => d.id === doc.id);
      if (idx >= 0) this.documents[idx] = doc;
      else this.documents.push(doc);
      this.idb.putDocument(doc).catch(() => {});
      this._renderDocList();
    },

    handleDocDeleted(msg) {
      if (!msg.document) return;
      this.documents = this.documents.filter(d => d.id !== msg.document.id);
      this.idb.deleteDocument(msg.document.id).catch(() => {});
      this._renderDocList();
    },

    handleDocList(msg) {
      if (!msg.documents) return;
      this.documents = msg.documents;
      this._renderDocList();
    },

    // ── Document list sidebar (no preview panel) ─────────

    _closeWorkspace() {
      if (!this.workspaceOpen) return;
      this.workspaceOpen = false;
      document.getElementById('cv2-workspace')?.classList.remove('open');
      document.getElementById('cv2-workspace-toggle')?.classList.remove('active');
    },

    toggleWorkspace() {
      if (this.workspaceOpen) { this._closeWorkspace(); return; }
      this.workspaceOpen = true;
      this._ensureWorkspace();
      document.getElementById('cv2-workspace')?.classList.add('open');
      document.getElementById('cv2-workspace-toggle')?.classList.add('active');
      this._renderDocList();
    },

    _ensureWorkspace() {
      if (document.getElementById('cv2-workspace')) return;
      const panel = document.createElement('div');
      panel.id = 'cv2-workspace';
      panel.className = 'cv2-workspace';
      panel.innerHTML = `
        <div class="cv2-workspace-header">
          <span>${this.t('chat.documents') || 'Documents'}</span>
          <span style="flex:1"></span>
          <button class="cv2-workspace-close">&times;</button>
        </div>
        <div class="cv2-workspace-body"></div>
      `;
      document.getElementById('chat-app')?.appendChild(panel);
      panel.querySelector('.cv2-workspace-close').addEventListener('click', () => this._closeWorkspace());

      // Click outside the panel closes it
      document.addEventListener('click', (e) => {
        if (!this.workspaceOpen) return;
        const ws = document.getElementById('cv2-workspace');
        const toggle = document.getElementById('cv2-workspace-toggle');
        if (ws && !ws.contains(e.target) && (!toggle || !toggle.contains(e.target))) {
          this._closeWorkspace();
        }
      });
    },

    /** Build the merged entry list (persistent + inline), newest first. */
    _getDocEntries() {
      const persistentEntries = this.documents.map(doc => ({
        id: doc.id,
        lang: doc.type,
        name: doc.name,
        data: typeof doc.data === 'string' ? doc.data : JSON.stringify(doc.data),
        version: doc.version,
        timestamp: doc.updated_at || doc.created_at || 0,
        source: 'persistent',
        element: null,
        _doc: doc,
      }));
      return [...persistentEntries, ...this.inlineDocBlocks]
        .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
    },

    _renderDocList() {
      this._ensureWorkspace();
      const body = document.querySelector('#cv2-workspace .cv2-workspace-body');
      if (!body) return;

      const allEntries = this._getDocEntries();

      if (allEntries.length === 0) {
        body.innerHTML = `<div class="cv2-doc-panel-empty">
          ${this.t('chat.no_documents') || 'No documents yet. Ask the AI to create charts, tables, or documents.'}
        </div>`;
        return;
      }

      body.innerHTML = '';
      for (const entry of allEntries) {
        const card = document.createElement('div');
        card.className = 'cv2-doc-card';
        card.dataset.docId = entry.id;
        const icon = ChatApp.DOC_ICONS[entry.lang] || 'article';

        card.innerHTML = `
          <div class="cv2-doc-card-header">
            <span class="cv2-doc-card-icon material-icons">${icon}</span>
            <span class="cv2-doc-card-name">${this._escHtml(entry.name)}</span>
            ${entry.version ? `<span class="cv2-doc-card-version">v${entry.version}</span>` : ''}
          </div>
          <span class="cv2-doc-card-type">${entry.lang}</span>
        `;

        // Delete button for persistent docs
        if (entry.source === 'persistent') {
          const delBtn = document.createElement('button');
          delBtn.className = 'cv2-doc-card-btn cv2-doc-delete-btn';
          delBtn.textContent = this.t('chat.delete') || 'Delete';
          delBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.ws.send({ type: 'send_message', text: '', images: null });
            this.documents = this.documents.filter(d => d.id !== entry.id);
            this.idb.deleteDocument(entry.id).catch(() => {});
            this._renderDocList();
          });
          card.appendChild(delBtn);
        }

        // Click → scroll to the document in chat, close sidebar
        card.addEventListener('click', () => {
          this._closeWorkspace();
          if (entry.element) {
            entry.element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            entry.element.classList.add('cv2-doc-highlight');
            setTimeout(() => entry.element.classList.remove('cv2-doc-highlight'), 1200);
          }
        });

        body.appendChild(card);
      }
    },

  });

  ChatFeatures.register('documents', {
    initState(app) {
      app.documents = [];
      app.inlineDocBlocks = [];
      app.workspaceOpen = false;
    },
    handleMessage: {
      'doc_created': 'handleDocCreated',
      'doc_updated': 'handleDocUpdated',
      'doc_deleted': 'handleDocDeleted',
      'doc_list': 'handleDocList',
    },
  });

  // ── WS → CustomEvent bridge for service messages ─────────
  // Route incoming directory / email WS messages to CustomEvents so
  // doc-plugin listeners (contact search, draft/send) can receive them.
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
