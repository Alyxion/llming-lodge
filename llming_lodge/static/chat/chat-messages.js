/**
 * chat-messages.js — Message streaming, protocol handlers, UI updates
 * Extracted from chat-app.js
 */
(function() {
  Object.assign(window._ChatAppProto, {

    // ── Session Init ──────────────────────────────────────

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
      this.supportedLanguages = msg.supported_languages || [];
      this._speechMaxTokens = msg.speech_max_tokens || 2000;
      if (msg.locale) this.config.locale = msg.locale;

      // Greeting uses config userName (given name); sidebar shows full name
      // Only re-render greeting if the name changed (avoids re-randomising during animation)
      const greetName = this.userName || 'there';
      if (this.el.greeting && greetName !== this._lastGreetName) {
        this.el.greeting.textContent = this._getGreeting(greetName);
      }
      this._lastGreetName = greetName;
      if (this.el.sidebarUserName) this.el.sidebarUserName.textContent = this.fullName || this.t('chat.default_user');

      this._toolPrefsNeedReapply = true;
      this._applyToolPrefs();
      this.updateModelButton();
      this._updateAvatarTooltip();
      this.updateSettings();
      this._renderQuickActions();
      this._renderLanguageDropdown();

      // Populate TTS voice selector
      this.ttsVoices = msg.tts_voices || [];
      const serverDefaultVoice = msg.tts_default_voice || '';
      if (this.el.voiceSelect && this.ttsVoices.length) {
        const genderIcon = { female: '♀', male: '♂', neutral: '⚬' };
        // Use localStorage preference, else server default, else first voice
        if (!this._ttsVoice) this._ttsVoice = serverDefaultVoice || this.ttsVoices[0]?.id || '';
        this.el.voiceSelect.innerHTML = this.ttsVoices.map(v =>
          `<option value="${v.id}"${v.id === this._ttsVoice ? ' selected' : ''}>${v.label} ${genderIcon[v.gender] || ''}</option>`
        ).join('');
      }

      // Sync voice selection from localStorage (speech response always starts disabled)
      if (this._ttsVoice) {
        this.ws.send({ type: 'update_settings', tts_voice: this._ttsVoice });
      }

      // Register MCP client-side renderers sent with session_init
      if (msg.client_renderers && msg.client_renderers.length) {
        this.handleRegisterRenderers({ renderers: msg.client_renderers });
      }

      // Hydrate server with restored conversation once WS is ready
      if (this._pendingRestore) {
        const restoreData = this._pendingRestore;
        this._pendingRestore = null;
        // Attach preset files for document re-injection
        if (restoreData.nudge_id) {
          const nudge = this.nudges.find(d => d.id === restoreData.nudge_id);
          if (nudge && nudge.files && nudge.files.length > 0) {
            restoreData._preset_files = this._wsFiles(nudge.files);
          }
        } else if (restoreData.project_id) {
          const proj = this.projects.find(p => p.id === restoreData.project_id);
          if (proj && proj.files && proj.files.length > 0) {
            restoreData._preset_files = (proj.files || []).map(f => {
              if (f.text_content) { const { content, ...rest } = f; return rest; }
              return f;
            });
          }
        }
        this.ws.send({ type: 'load_conversation', data: restoreData });
        // Restore documents to server-side store
        if (this.documents.length > 0) {
          this.ws.send({ type: 'doc_restore', documents: this.documents });
        }
        // Restore files from IDB to server — deferred until session_id_updated
        // arrives so the upload manager exists under the correct session ID.
        // _pendingFileRestore is consumed in handleSessionIdUpdated.
      }

      // Auto-enable doc editing tools for blocks rendered before WS connected.
      // On page reload, _restoreLastConversation() renders messages (populating
      // inlineDocBlocks) BEFORE the WS connects.  If MCP discovery completed
      // before WS connect, session_init already includes MCP groups but no
      // subsequent tools_updated will arrive.  Handle that case here.
      this._autoEnableForRenderedBlocks();

      // Validate favorite nudges ACL — prune any the user can no longer see
      if (this.favoriteNudges && this.favoriteNudges.length > 0) {
        const uids = this.favoriteNudges.map(f => f.uid).filter(Boolean);
        if (uids.length > 0) {
          this.ws.send({ type: 'nudge_validate_favorites', uids });
        }
      }
    },

    // ── Response Streaming ────────────────────────────────

    handleResponseStarted(msg) {
      this._isIntercept = !!msg.intercept;
      if (!this._isIntercept) {
        this.streaming = true;
        this.md.streaming = true;
      }
      this.fullText = '';
      this.toolCalls = [];
      this._receivedImages = [];
      this._receivedPdfPreviews = [];
      this._currentModelIcon = msg.model_icon;
      this._currentModelLabel = msg.model_label;

      // Show chat area if hidden
      if (!this.chatVisible) this.showChat();

      // Create assistant message container
      const msgEl = document.createElement('div');
      msgEl.className = 'cv2-msg-assistant';
      const autoIconHtml = msg.auto_icon
        ? `<img src="${this.config.staticBase}/${msg.auto_icon}" alt="" class="cv2-auto-icon">`
        : '';
      msgEl.innerHTML = `
        <div class="cv2-msg-header">
          ${autoIconHtml}<img src="${this.config.staticBase}/${msg.model_icon}" alt="">
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
      if (!this._isIntercept) this._updateSendButton();
    },

    handleTextChunk(msg) {
      this.fullText += msg.content;
      if (this._currentBody) {
        this._currentBody.innerHTML = this.md.render(this.fullText);
        // Plugin blocks are NOT hydrated during streaming — spinners shown instead.
        // Hydration happens in handleResponseCompleted for the final render.
      }
      this._scrollToBottom();
    },

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
    },

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
    },

    handlePdfPagePreview(msg) {
      if (!this._currentBody) return;
      const src = this._toDataUri(msg.data);
      const fileId = msg.file_id;
      const page = msg.page;
      const totalPages = msg.total_pages;

      this._receivedPdfPreviews = this._receivedPdfPreviews || [];
      this._receivedPdfPreviews.push({ src, fileId, page, totalPages });

      const wrap = document.createElement('div');
      wrap.className = 'cv2-pdf-page-wrap';
      wrap.title = `Page ${page} / ${totalPages}`;

      const img = document.createElement('img');
      img.className = 'cv2-pdf-page-preview';
      img.src = src;

      const label = document.createElement('div');
      label.className = 'cv2-pdf-page-label';
      label.innerHTML = `<span class="material-icons" style="font-size:14px">picture_as_pdf</span> Page ${page}`;

      wrap.appendChild(img);
      wrap.appendChild(label);

      wrap.addEventListener('click', () => {
        const url = `/api/llming/file-preview/${this.sessionId}/${fileId}`;
        this._openPdfViewer(url, page);
      });

      this._currentBody.appendChild(wrap);
      this._scrollToBottom();
    },

    async handleResponseCompleted(msg) {
      const wasIntercept = this._isIntercept;
      this._isIntercept = false;
      if (!wasIntercept) {
        this.streaming = false;
        this.md.streaming = false;
      }
      this.fullText = msg.full_text;

      // Remove spinner
      if (this._spinner) {
        this._spinner.remove();
        this._spinner = null;
      }

      // Merge full tool_calls data from response_completed (includes arguments + result)
      if (msg.tool_calls?.length) {
        for (const tc of msg.tool_calls) {
          const existing = this.toolCalls.find(t => t.call_id === tc.call_id);
          if (existing) Object.assign(existing, tc);
          else this.toolCalls.push(tc);
        }
        this._renderToolArea();
      }

      // Collect flux sources from tool calls and set on message element
      if (msg.tool_calls?.length && this._currentMsgEl) {
        const allSources = [];
        for (const tc of msg.tool_calls) {
          if (tc.sources?.length) allSources.push(...tc.sources);
        }
        if (allSources.length) {
          this._currentMsgEl.setAttribute('data-flux-sources', JSON.stringify(allSources));
        }
      }

      // Final render
      if (this._currentBody) {
        // Extract inline images from text
        let { images, text } = this._extractInlineImages(this.fullText);
        // Close any unclosed plugin fenced code blocks (truncated LLM output)
        if (this.md._pluginRegistry) {
          const langs = this.md._pluginRegistry.languages;
          for (const lang of langs) {
            const openTag = '```' + lang;
            const idx = text.lastIndexOf(openTag);
            if (idx !== -1) {
              const after = text.slice(idx + openTag.length);
              if (after.indexOf('\n```') === -1) {
                console.log('[FIX] Closing unclosed', lang, 'fence');
                text += '\n```';
              }
              break;
            }
          }
        }
        this._currentBody.innerHTML = this.md.render(text);

        // Hydrate plugin blocks
        if (this.md.hydratePluginBlocks) await this.md.hydratePluginBlocks(this._currentBody);

        // Run inline pattern renderers (email enhancers, etc.)
        if (this.md._pluginRegistry?.runInlineRenderers) {
          await this.md._pluginRegistry.runInlineRenderers(this._currentBody);
        }

        // Add message actions (speaker + copy + timestamp)
        this._addMessageActions(this._currentBody, text, new Date().toISOString());

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

        // Re-add PDF page previews
        if (this._receivedPdfPreviews?.length) {
          for (const pv of this._receivedPdfPreviews) {
            const wrap = document.createElement('div');
            wrap.className = 'cv2-pdf-page-wrap';
            wrap.title = `Page ${pv.page} / ${pv.totalPages}`;
            const img = document.createElement('img');
            img.className = 'cv2-pdf-page-preview';
            img.src = pv.src;
            const label = document.createElement('div');
            label.className = 'cv2-pdf-page-label';
            label.innerHTML = `<span class="material-icons" style="font-size:14px">picture_as_pdf</span> Page ${pv.page}`;
            wrap.appendChild(img);
            wrap.appendChild(label);
            const fid = pv.fileId, pg = pv.page;
            wrap.addEventListener('click', () => this._openPdfViewer(`/api/llming/file-preview/${this.sessionId}/${fid}`, pg));
            this._currentBody.appendChild(wrap);
          }
        }

        // Wrap all images with hover actions + lightbox click
        this._wrapAllImages(this._currentBody);
      }

      // Apply MCP avatar override (e.g., LISA/LINUS custom avatar)
      if (msg.avatar_override && this._currentMsgEl) {
        const header = this._currentMsgEl.querySelector('.cv2-msg-header');
        if (header) {
          const img = header.querySelector('img');
          const span = header.querySelector('span');
          if (img) img.src = `${this.config.staticBase}/${msg.avatar_override.icon}`;
          if (span) span.textContent = msg.avatar_override.label;
        }
        // Store for serialization so restored messages also show the avatar
        this._currentMsgEl.dataset.avatarIcon = msg.avatar_override.icon;
        this._currentMsgEl.dataset.avatarLabel = msg.avatar_override.label;
      }

      // Keep reference to last response body for TTS word highlighting
      this._lastResponseBody = this._currentBody;

      this._currentMsgEl = null;
      this._currentToolArea = null;
      this._currentBody = null;
      this._receivedImages = [];
      this._receivedPdfPreviews = [];
      this._updateSendButton();
      this._scrollToBottom();
    },

    handleResponseCancelled() {
      this.streaming = false;
      if (this._spinner) { this._spinner.remove(); this._spinner = null; }
      if (this._currentBody) {
        this.fullText += '\n\n*' + this.t('chat.stopped_by_user') + '*';
        this._currentBody.innerHTML = this.md.render(this.fullText);
      }
      this._currentMsgEl = null;
      this._currentToolArea = null;
      this._currentBody = null;
      this._updateSendButton();
      if (this._speechMode) {
        this._speechModeNextCycle();
      }
    },

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
    },

    // ── Model & Tools ─────────────────────────────────────

    handleModelSwitched(msg) {
      this.currentModel = msg.new_model;
      if (msg.available_tools) {
        this.tools = msg.available_tools;
        this._renderPlusMenu();
      }
      this.updateModelButton();

      // Add model switch message to chat
      const el = document.createElement('div');
      el.className = 'cv2-model-switch';
      el.innerHTML = `
        <img src="${this.config.staticBase}/${msg.old_icon}" alt="">
        <span class="cv2-arrow material-icons">arrow_forward</span>
        <img src="${this.config.staticBase}/${msg.new_icon}" alt="">
        <span>${this.t('chat.model_switched', { old: this._escHtml(msg.old_label), new: this._escHtml(msg.new_label) })}</span>
      `;
      if (this.el.messages) {
        this.el.messages.appendChild(el);
        this._scrollToBottom();
      }
    },

    handleModelLocked(msg) {
      // System droplet enforced a model — lock the selector
      this._modelLocked = true;
      this._allowedModels = null;
      this.currentModel = msg.model;
      this.updateModelButton();
      if (this.el.modelBtn) {
        this.el.modelBtn.style.pointerEvents = 'none';
        this.el.modelBtn.style.opacity = '0.6';
        this.el.modelBtn.title = `Model locked by ${msg.reason || 'Droplet'}`;
      }
    },

    handleModelRestricted(msg) {
      // Droplet allows switching between a set of models
      this._modelLocked = false;
      this._allowedModels = new Set(msg.allowed_models || []);
      this.currentModel = msg.model;
      this.updateModelButton();
      if (this.el.modelBtn) {
        this.el.modelBtn.style.pointerEvents = '';
        this.el.modelBtn.style.opacity = '';
        this.el.modelBtn.title = '';
      }
    },

    handleModelUnlocked() {
      this._modelLocked = false;
      this._allowedModels = null;
      if (this.el.modelBtn) {
        this.el.modelBtn.style.pointerEvents = '';
        this.el.modelBtn.style.opacity = '';
        this.el.modelBtn.title = '';
      }
    },

    handleToolsUpdated(msg) {
      this.tools = msg.tools || [];
      // Re-apply saved preferences after MCP discovery completes.
      // Keep retrying on each tools_updated until all saved prefs have been applied
      // (MCP servers may discover at different times).
      if (this._toolPrefsNeedReapply) {
        this._applyToolPrefs();
        // Check if all saved prefs have been matched — stop retrying if so
        try {
          const prefs = JSON.parse(localStorage.getItem('cv2-tool-prefs') || '{}');
          const allMatched = Object.keys(prefs).every(name =>
            !this.tools.find(t => t.name === name && t.available) ||
            this.tools.find(t => t.name === name && t.enabled === prefs[name])
          );
          if (allMatched) this._toolPrefsNeedReapply = false;
        } catch (_) { this._toolPrefsNeedReapply = false; }
      }
      // Auto-enable doc editing tools for any blocks rendered before tools arrived
      this._autoEnableForRenderedBlocks();
    },

    /**
     * Register MCP-provided client-side renderers as DocPluginRegistry plugins.
     * Each renderer provides: { lang, js, css? }
     */
    handleRegisterRenderers(msg) {
      const renderers = msg.renderers || [];
      if (!renderers.length || !this.md?._pluginRegistry) return;
      const registry = this.md._pluginRegistry;
      for (const r of renderers) {
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
            fn(registry);
          } catch (err) {
            console.error('[MCP] Failed to register inline renderer:', err);
          }
          continue;
        }
        if (!r.lang || !r.js || registry.has(r.lang)) continue;
        // Inject CSS if provided
        if (r.css) {
          const style = document.createElement('style');
          style.dataset.mcpRenderer = r.lang;
          style.textContent = r.css;
          document.head.appendChild(style);
        }
        // Evaluate JS — it receives `registry` and `lang` in scope
        try {
          const registerFn = new Function('registry', 'lang', r.js);
          registerFn(registry, r.lang);
        } catch (err) {
          console.error(`[MCP] Failed to register renderer for "${r.lang}":`, err);
        }
      }
    },

    handleContextInfo(msg) {
      this.contextInfo = msg;
      this._updateContextCircle();
    },

    handleTitleGenerated(msg) {
      // Update conversation title in sidebar
      this._updateConversationTitle(this.sessionId, msg.title);
    },

    // ── Condensation ──────────────────────────────────────

    handleCondenseStart() {
      if (!this.el.messages) return;
      const el = document.createElement('div');
      el.className = 'cv2-condense';
      el.id = 'cv2-condense-indicator';
      el.innerHTML = `
        <span class="cv2-condense-label">${this.t('chat.condensing')}</span>
        <div class="cv2-condense-bar"><div class="cv2-condense-fill" style="width:0%"></div></div>
      `;
      this.el.messages.appendChild(el);
      this._scrollToBottom();
    },

    handleCondenseProgress(msg) {
      const fill = document.querySelector('#cv2-condense-indicator .cv2-condense-fill');
      if (fill) fill.style.width = `${Math.min(100, (msg.pct || 0) * 100)}%`;
    },

    handleCondenseEnd() {
      const el = document.getElementById('cv2-condense-indicator');
      if (el) {
        el.innerHTML = `
          <div class="cv2-condense-done">
            <span class="material-icons" style="font-size:14px">compress</span>
            <span>${this.t('chat.condensed')}</span>
          </div>
        `;
      }
    },

    handleSaveConversation(msg) {
      if (msg.data) {
        const convId = msg.data.id;
        // If currently in incognito, record this conversation ID so it's
        // blocked even after incognito is exited (race condition protection)
        if (this.incognito && convId) {
          if (!this._incognitoConvIds) this._incognitoConvIds = new Set();
          this._incognitoConvIds.add(convId);
        }
        // Block save for incognito conversations (even if incognito was
        // already exited — the ID was recorded above or by idb.put)
        if (this._incognitoConvIds?.has(convId)) return;

        // Fresh chat just got saved — clear the fresh draft key
        this._clearDraft('_fresh');
        this.activeConvId = convId;
        try {
          localStorage.setItem('cv2-active-conversation', convId);
          localStorage.removeItem('cv2-fresh-chat');
        } catch (_) {}
        // Include documents in saved conversation data
        if (this.documents.length > 0) {
          msg.data.documents = this.documents;
        }
        // Build file refs from pending files + images
        const fileRefs = [
          ...this._pendingFiles.filter(f => f.hash).map(f => ({
            hash: f.hash, name: f.name, mime_type: f.mimeType, size: f.size, type: 'document',
          })),
          ...this._pendingImages.filter(i => i.hash).map((img, idx) => ({
            hash: img.hash, name: `Image ${idx + 1}`, mime_type: 'image/jpeg', size: 0, type: 'image',
          })),
        ];
        // Merge: server data + previously saved refs + current pending refs
        // _savedFileRefs preserves refs even after _pendingFiles is cleared
        // (e.g. when user navigates away before title generation save arrives)
        const fromServer = msg.data.file_refs || [];
        const fromSaved = this._savedFileRefs[convId] || [];
        const merged = [...fromServer];
        for (const ref of fromSaved) {
          if (!merged.some(r => r.hash === ref.hash)) merged.push(ref);
        }
        for (const ref of fileRefs) {
          if (!merged.some(r => r.hash === ref.hash)) merged.push(ref);
        }
        if (merged.length > 0) {
          msg.data.file_refs = merged;
          this._savedFileRefs[convId] = merged;
        }

        this.idb.put(msg.data).then(() => this.refreshConversations());
        // Update IDB file refs: move from '_pending' to real conversation ID
        for (const ref of merged) {
          this.idb.putFile(ref.hash, ref.name, ref.mime_type, ref.size, null, convId)
            .then(() => this.idb.removeFileRef(ref.hash, '_pending'))
            .catch(() => {});
        }
        // Persist documents to IDB
        for (const doc of this.documents) {
          doc.conversation_id = convId;
          this.idb.putDocument(doc).catch(() => {});
        }
      }
    },

    handleSessionIdUpdated(msg) {
      // Server changed session ID (e.g. after load_conversation)
      this.sessionId = msg.session_id;
      // Now safe to restore files — upload manager exists under this session ID
      if (this._pendingFileRestore) {
        const refs = this._pendingFileRestore;
        this._pendingFileRestore = null;
        this._restoreFiles(refs);
      }
    },

    handleBudgetUpdate(msg) {
      this.budget = msg.budget;
      this._updateAvatarTooltip();
    },

    // ── Chat Cleared + User Message + Files ───────────────

    handleChatCleared(msg) {
      // Tear down any active overlay (preset editor, explorer) first
      if (this._activeView !== 'chat') {
        this._switchView('chat');
      }
      this._autoEnabledDocTypes?.clear();
      this.sessionId = msg.new_session_id;
      this._activeProjectId = msg.project_id || null;
      this._activeNudgeId = msg.nudge_id || null;
      this._removeNudgeChatHeader();
      this._removeProjectChatHeader();
      this.chatVisible = false;
      const incBtn = document.getElementById('cv2-incognito-toggle');
      if (incBtn && !this.incognito) incBtn.style.display = '';
      this.el.messages.innerHTML = '';
      this.el.initialView.classList.remove('cv2-pv-hidden');
      this.el.messagesWrap.classList.remove('cv2-pv-hidden');
      this.el.initialView.style.display = 'flex';
      this.el.messagesWrap.classList.add('cv2-messages-wrap-hidden');
      const wrapper = this.el.initialView.closest('.cv2-chat-wrapper');
      if (wrapper) wrapper.classList.add('cv2-initial-mode');
      const banner = document.getElementById('cv2-banner');
      if (banner) banner.style.display = '';
      // Save draft for the conversation we're leaving before clearing
      if (this.activeConvId && this.el.textarea.value.trim()) this._saveDraft();
      this.activeConvId = '';
      this._blockDataStore?.clear();
      this.documents = [];
      this.inlineDocBlocks = [];
      this._closeWorkspace();
      this._renderDocList();
      this.el.textarea.value = '';
      this._autoResizeTextarea();
      try {
        localStorage.removeItem('cv2-active-conversation');
        localStorage.setItem('cv2-fresh-chat', '1');
      } catch (_) {}
      // Restore any existing fresh-chat draft
      this._restoreDraft();
      this._pendingImages = [];
      this._pendingFiles = [];
      this._renderAttachments();
      this.refreshConversations();
      this._startPlaceholderCycle();
      if (!this._speechMode && !this._rtActive) this.el.textarea.focus();

      // Nudge: render header + greeting with suggestions
      if (this._activeNudgeId) {
        // Prefer nudge_meta from server (MongoDB flow)
        const nudgeMeta = msg.nudge_meta;
        const nudge = nudgeMeta || this.nudges.find(d => d.id === this._activeNudgeId);
        if (nudge) {
          this._activeNudgeMeta = nudgeMeta || null;
          // Show greeting with suggestions in initial view (don't call showChat yet)
          this._renderNudgeGreeting(nudge);
          this._lockModelForNudge(nudge);
          // Apply capability toggles from server (null = keep user's current setting)
          if (nudgeMeta && nudgeMeta.capabilities) {
            let changed = false;
            for (const [name, enabled] of Object.entries(nudgeMeta.capabilities)) {
              if (enabled === null || enabled === undefined) continue;  // "don't care" — keep user setting
              const tool = this.tools.find(t => t.name === name);
              if (tool && tool.enabled !== enabled) {
                tool.enabled = enabled;
                this.ws.send({ type: 'toggle_tool', name, enabled, restore: true });
                changed = true;
              }
            }
            if (changed) this._renderPlusMenu();
          }
        }
      } else if (this._activeProjectId) {
        // Project header (similar to nudge header)
        this._activeNudgeMeta = null;
        this._unlockModel();
        const proj = this.projects.find(p => p.id === this._activeProjectId);
        if (proj) {
          this._renderProjectChatHeader(proj);
        }
      } else {
        this._activeNudgeMeta = null;
        this._restoreDefaultGreeting();
        this._unlockModel();
      }

      // Restore pending first message (sent before chat_cleared arrived)
      if (this._pendingFirstMsgHtml) {
        this.showChat();
        const tmp = document.createElement('div');
        tmp.innerHTML = this._pendingFirstMsgHtml;
        const el = tmp.firstChild;
        this.el.messages.appendChild(el);
        this._wrapAllImages(el);
        this._pendingFirstMsgHtml = null;
      }
    },

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
    },

    handleFilesUpdated(msg) {
      // Server pushed updated file list (e.g. from debug API attach)
      if (msg.files) {
        this._pendingFiles = msg.files;
        this._renderAttachments();
      }
    },

    // ── UI Action + Language + Action Callback ────────────

    handleUIAction(msg) {
      const action = msg.action;
      switch (action) {
        case 'run_js':
          try { new Function(msg.code)(); } catch(e) { console.error('[run_js]', e); }
          break;
        case 'toggle_sidebar': {
          this.sidebarVisible = !this.sidebarVisible;
          this._updateSidebarUI(this.sidebarVisible);
          break;
        }
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
          this.idb.getAllMeta().then(all => {
            const list = all.map(c => ({
              id: c.id,
              title: c.title || c.first_user_snippet || this.t('chat.untitled'),
              created_at: c.created_at,
              message_count: c.message_count || 0,
            }));
            this.ws.send({ type: 'conversation_list', conversations: list });
          });
          break;
        case 'list_presets':
          // Server requests preset list from IDB
          this.idb.getAllPresets(msg.preset_type || null).then(presets => {
            const list = presets.map(p => ({
              id: p.id, type: p.type, name: p.name,
              description: p.description || '',
              model: p.model, language: p.language,
              file_count: (p.files || []).length,
              files: (p.files || []).map(f => ({
                file_id: f.file_id, name: f.name, size: f.size,
                mime_type: f.mime_type,
                has_content: !!f.content,
                text_content_length: (f.text_content || '').length,
              })),
              system_prompt_length: (p.system_prompt || '').length,
              created_at: p.created_at, updated_at: p.updated_at,
            }));
            this.ws.send({ type: 'preset_list', presets: list });
          });
          break;
        case 'get_preset':
          // Server requests full preset data by ID
          this.idb.getPreset(msg.preset_id).then(p => {
            // Strip heavy base64 content but keep it for files needing server extraction
            const files = (p?.files || []).map(f => {
              if (f.text_content) { const { content, ...rest } = f; return rest; }
              return f;
            });
            const safe = p ? { ...p, files } : null;
            this.ws.send({ type: 'preset_detail', preset: safe, preset_id: msg.preset_id });
          });
          break;
        case 'save_preset':
          // Server pushes a preset to IDB
          if (msg.data) {
            this.idb.putPreset(msg.data).then(() => {
              this.refreshConversations();
              this.ws.send({ type: 'preset_saved_ack', preset_id: msg.data.id, ok: true });
            });
          }
          break;
        case 'delete_preset':
          if (msg.preset_id) {
            this._deletePreset(msg.preset_id).then(() => {
              this.ws.send({ type: 'preset_deleted_ack', preset_id: msg.preset_id, ok: true });
            });
          }
          break;
        case 'open_lightbox':
          // Open lightbox for an image by index in the current chat
          {
            const imgs = this.el.messages.querySelectorAll('.cv2-img-wrap img');
            const idx = msg.image_index ?? 0;
            if (imgs[idx]) this._openLightbox(imgs[idx].src);
          }
          break;
        case 'inject_file': {
          // Debug API: inject a file through the normal upload path
          // msg: { filename, mime_type, data_base64 }
          const binary = atob(msg.data_base64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          const blob = new Blob([bytes], { type: msg.mime_type || 'application/octet-stream' });
          const file = new File([blob], msg.filename, { type: blob.type });
          this._handleFileSelection([file]).then(() => {
            this.ws.send({ type: 'inject_file_ack', ok: true, filename: msg.filename });
          }).catch(err => {
            this.ws.send({ type: 'inject_file_ack', ok: false, error: String(err) });
          });
          break;
        }
        case 'remove_file': {
          // Debug API: remove a pending file by fileId, same as clicking the X
          const idx = this._pendingFiles.findIndex(f => f.fileId === msg.file_id);
          if (idx >= 0) {
            this._pendingFiles.splice(idx, 1);
            this._renderAttachments();
            // Notify server to remove from upload manager
            fetch(`/api/llming/upload/${this.sessionId}/${msg.file_id}`, {
              method: 'DELETE',
              headers: { 'X-User-Id': this.config.userId },
            }).then(() => this.ws.send({ type: 'file_uploaded' })).catch(() => {});
            this.ws.send({ type: 'remove_file_ack', ok: true, file_id: msg.file_id });
          } else {
            this.ws.send({ type: 'remove_file_ack', ok: false, file_id: msg.file_id, error: 'not found' });
          }
          break;
        }
        case 'list_idb_files':
          this.idb.getAllFiles().then(files => {
            this.ws.send({ type: 'idb_file_list', files: files.map(({ data, ...rest }) => rest) });
          });
          break;
        case 'get_idb_file':
          this.idb.getFileMeta(msg.hash).then(file => {
            this.ws.send({ type: 'idb_file_detail', file, hash: msg.hash });
          });
          break;
        case 'get_idb_file_refs':
          this.idb.get(msg.conversation_id).then(conv => {
            this.ws.send({ type: 'idb_file_refs', file_refs: conv?.file_refs || [], conversation_id: msg.conversation_id });
          });
          break;

        // ── Navigation / scroll actions (debug API) ──────────
        case 'scroll_to_bottom':
          this._scrollToBottom();
          this.ws.send({ type: 'scroll_ack', ok: true, target: 'bottom' });
          break;
        case 'scroll_to_message': {
          const idx = msg.index ?? -1;
          const msgs = this.el.messages?.querySelectorAll('.cv2-msg-user, .cv2-msg-assistant') || [];
          const target = idx < 0 ? msgs[msgs.length + idx] : msgs[idx];
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            this.ws.send({ type: 'scroll_ack', ok: true, target: 'message', index: idx, total: msgs.length });
          } else {
            this.ws.send({ type: 'scroll_ack', ok: false, error: `message index ${idx} out of range (${msgs.length} messages)` });
          }
          break;
        }
        case 'open_project': {
          const projId = msg.project_id;
          if (projId && typeof this._openProjectView === 'function') {
            this._openProjectView(projId);
            this.ws.send({ type: 'open_project_ack', ok: true, project_id: projId });
          } else {
            this.ws.send({ type: 'open_project_ack', ok: false, error: projId ? 'no _openProjectView' : 'no project_id' });
          }
          break;
        }
        case 'open_droplet': {
          const uid = msg.nudge_uid;
          const nudgeId = msg.nudge_id;
          if (uid && typeof this._startNudgeChat === 'function') {
            this._startNudgeChat({ uid });
            this.ws.send({ type: 'open_droplet_ack', ok: true, nudge_uid: uid });
          } else if (nudgeId && typeof this._startNudgeChat === 'function') {
            const nudge = this.nudges?.find(d => d.id === nudgeId);
            if (nudge) {
              this._startNudgeChat(nudge);
              this.ws.send({ type: 'open_droplet_ack', ok: true, nudge_id: nudgeId });
            } else {
              this.ws.send({ type: 'open_droplet_ack', ok: false, error: `nudge ${nudgeId} not found in IDB` });
            }
          } else {
            this.ws.send({ type: 'open_droplet_ack', ok: false, error: 'no nudge_uid or nudge_id' });
          }
          break;
        }
        case 'open_conversation': {
          const convId = msg.conversation_id;
          if (convId && typeof this._selectConversation === 'function') {
            this._selectConversation(convId).then(() => {
              this.ws.send({ type: 'open_conversation_ack', ok: true, conversation_id: convId });
            }).catch(err => {
              this.ws.send({ type: 'open_conversation_ack', ok: false, error: String(err) });
            });
          } else {
            this.ws.send({ type: 'open_conversation_ack', ok: false, error: 'no conversation_id' });
          }
          break;
        }
        case 'get_console_logs': {
          // Debug API: return captured console log buffer
          const since = msg.since || 0;
          const level = msg.level || '';
          const pattern = msg.pattern || '';
          let logs = this._consoleLogs || [];
          if (since) logs = logs.filter(e => e.ts >= since);
          if (level) logs = logs.filter(e => e.level === level);
          if (pattern) { try { const re = new RegExp(pattern, 'i'); logs = logs.filter(e => re.test(e.msg)); } catch(_) {} }
          // Also include browser MCP worker status
          const workers = {};
          for (const [uid, entry] of Object.entries(this._mcpWorkers || {})) {
            workers[uid] = { pendingCalls: Object.keys(entry.pendingCalls || {}).length };
          }
          this.ws.send({ type: 'console_logs', logs: logs.slice(-200), total: (this._consoleLogs || []).length, workers });
          break;
        }
        default:
          console.warn('[Chat] Unknown UI action:', action);
      }
    },

    handleLanguageChanged(msg) {
      if (msg.translations) this.config.translations = msg.translations;
      if (msg.locale) this.config.locale = msg.locale;
      if (msg.app_title) this.config.appTitle = msg.app_title;
      if (msg.quick_actions) { this.quickActions = msg.quick_actions; this._renderQuickActions(); }
      if (msg.tools) { this.tools = msg.tools; this._renderPlusMenu(); }
      this._rerenderTranslatedUI();
      this._renderLanguageDropdown(); // Re-render flag button with new locale
    },

    handleActionCallbackResult(msg) {
      if (msg.notification) {
        // Long messages get a dismissable dialog, short ones get a toast
        if (msg.notification.length > 100) {
          this._showNotificationDialog(msg.notification);
        } else {
          this._showToast(msg.notification, msg.notification_type || 'info');
        }
      }
    },

    handleNotification(msg) {
      this._showToast(msg.message, msg.level || 'negative');
    },

    handleDevMode(msg) {
      this._devMode = !!msg.enabled;
      console.log('[DEV] Dev mode', this._devMode ? 'enabled' : 'disabled');
    },

    handleKantiniLikesUpdate(msg) {
      // Dispatch as DOM event so the kantini inline card plugin can react
      document.dispatchEvent(new CustomEvent('kantini-likes', { detail: msg }));
    },

    _showNotificationDialog(message) {
      const overlay = document.createElement('div');
      overlay.className = 'cv2-dialog-overlay';
      overlay.innerHTML = `
        <div class="cv2-dialog cv2-notification-dialog">
          <div class="cv2-dialog-body" style="white-space:pre-line">${this._escHtml(message)}</div>
          <div class="cv2-dialog-actions">
            <button class="cv2-dialog-btn cv2-notification-ok">OK</button>
          </div>
        </div>`;
      document.getElementById('chat-app').appendChild(overlay);
      overlay.querySelector('.cv2-notification-ok').addEventListener('click', () => overlay.remove());
      overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    },

    _showToast(message, type = 'info') {
      const toast = document.createElement('div');
      toast.className = `cv2-toast cv2-toast-${type}`;
      toast.textContent = message;
      const root = document.getElementById('chat-app');
      root.appendChild(toast);
      requestAnimationFrame(() => toast.classList.add('cv2-toast-visible'));
      setTimeout(() => {
        toast.classList.remove('cv2-toast-visible');
        setTimeout(() => toast.remove(), 300);
      }, 3000);
    },

    // ── Send Message ──────────────────────────────────────

    sendMessage() {
      if (this.streaming) {
        // Stop streaming
        this.ws.send({ type: 'stop_streaming' });
        return;
      }

      const text = this.el.textarea.value.trim();
      if (!text && this._pendingImages.length === 0) return;

      // If sending from project view (no active conversation), init a project chat first
      if (!this.activeConvId && this._activeProjectId) {
        const proj = this.projects.find(p => p.id === this._activeProjectId);
        if (proj) {
          // Include content for files missing text_content (binary files needing server extraction)
          const projFiles = (proj.files || []).map(f => {
            if (f.text_content) { const { content, ...rest } = f; return rest; }
            if (!f.content && !f.text_content) {
              console.warn('[Project] File "%s" has no content or text — re-attach in project settings', f.name);
            }
            return f;
          });
          this._pendingFirstMsgHtml = null;
          this.ws.send({ type: 'new_chat', preset: { id: proj.id, type: 'project', system_prompt: proj.system_prompt, model: proj.model, language: proj.language, files: projFiles, doc_plugins: proj.doc_plugins ?? null } });
        }
      }

      // Show chat area
      if (!this.chatVisible) this.showChat();

      // Hide incognito toggle once conversation starts (both modes)
      const _incBtn = document.getElementById('cv2-incognito-toggle');
      if (_incBtn) _incBtn.style.display = 'none';

      // Render user message
      const userEl = document.createElement('div');
      userEl.className = 'cv2-msg-user';
      let imagesHtml = '';
      if (this._pendingImages.length > 0) {
        imagesHtml = '<div class="cv2-msg-user-images">' +
          this._pendingImages.map(i => `<img src="${i.dataUri}" alt="Uploaded">`).join('') +
          '</div>';
      }
      const bubbleHtml = `<div class="cv2-msg-user-bubble">${imagesHtml}${text ? this.md.render(text) : ''}</div>`;
      userEl.innerHTML = bubbleHtml;
      this.el.messages.appendChild(userEl);
      this._wrapAllImages(userEl);

      // Save for restoration after chat_cleared (new_chat race condition)
      if (!this.activeConvId) {
        this._pendingFirstMsgHtml = userEl.outerHTML;
      }

      // Collect images for WS message (base64 only, no data: prefix)
      const images = this._pendingImages.length > 0
        ? this._pendingImages.map(i => i.dataUri.split(',')[1])
        : undefined;

      // Clear input and pasted images (files persist — they're in system prompt context)
      this.el.textarea.value = '';
      this._autoResizeTextarea();
      this._pendingImages = [];
      this._renderAttachments();
      this._clearDraft();

      // Send via WS
      this.ws.send({ type: 'send_message', text: text || '', images });

      this._scrollToBottom();
    },

    // ── Rendered Messages (conversation restore) ──────────

    async _renderLoadedMessages(messages) {
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
          // Use avatar override if stored (MCP custom avatars like LISA/LINUS)
          const av = msg.avatar_override;
          let icon = av?.icon || modelInfo?.icon || '';
          let label = av?.label || modelInfo?.label || '';
          // For auto-select, show both icons + "Auto (Model Name)"
          let autoIconHtml = '';
          if (this.currentModel === '@auto' && !av) {
            const autoInfo = this.models.find(m => m.model === '@auto');
            const underlying = this._autoUnderlyingModel || '';
            const realInfo = underlying ? this.models.find(m => m.model === underlying) : null;
            if (autoInfo) {
              autoIconHtml = `<img src="${this.config.staticBase}/${autoInfo.icon}" alt="" class="cv2-auto-icon">`;
              if (realInfo) {
                icon = realInfo.icon;
                label = `Auto (${realInfo.label})`;
              }
            }
          }
          let { images: extractedImages, text } = this._extractInlineImages(content);
          // Close any unclosed plugin fenced code blocks (truncated LLM output)
          if (this.md._pluginRegistry) {
            const langs = this.md._pluginRegistry.languages;
            for (const lang of langs) {
              const openTag = '```' + lang;
              const idx = text.lastIndexOf(openTag);
              if (idx !== -1) {
                const after = text.slice(idx + openTag.length);
                if (after.indexOf('\n```') === -1) {
                  text += '\n```';
                }
                break;
              }
            }
          }

          const el = document.createElement('div');
          el.className = 'cv2-msg-assistant';
          const toolCalls = msg.tool_calls || [];

          // Collect flux sources from saved tool calls
          const allSources = [];
          for (const tc of toolCalls) {
            if (tc.sources?.length) allSources.push(...tc.sources);
          }
          if (allSources.length) {
            el.setAttribute('data-flux-sources', JSON.stringify(allSources));
          }

          el.innerHTML = `
            <div class="cv2-msg-header">
              ${autoIconHtml}${icon ? `<img src="${this.config.staticBase}/${icon}" alt="">` : ''}
              <span>${this._escHtml(label)}</span>
            </div>
            ${toolCalls.length > 0 ? '<div class="cv2-tool-area"></div>' : ''}
            <div class="cv2-msg-body">${this.md.render(text)}</div>
          `;

          // Render saved tool calls
          if (toolCalls.length > 0) {
            const toolArea = el.querySelector('.cv2-tool-area');
            this._renderSavedToolCalls(toolArea, toolCalls);
          }

          const body = el.querySelector('.cv2-msg-body');

          // Hydrate plugin blocks in restored messages
          await this.md.hydratePluginBlocks(body);

          // Run inline pattern renderers (email enhancers, etc.)
          if (this.md._pluginRegistry?.runInlineRenderers) {
            await this.md._pluginRegistry.runInlineRenderers(body);
          }

          // Add images
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

          this._addMessageActions(body, text, msg.timestamp);
          this._wrapAllImages(body);
          this.el.messages.appendChild(el);
        }
      }
    },

    // _extractInlineImages is provided by chat-images.js

    // ── Message Actions (speaker + copy on hover) ─────────

    _addMessageActions(bodyEl, text, timestamp) {
      const bar = document.createElement('div');
      bar.className = 'cv2-msg-actions';

      // Timestamp label — subtle, shown on hover with the actions
      if (timestamp) {
        const d = new Date(timestamp);
        if (!isNaN(d)) {
          const now = new Date();
          const isToday = d.toDateString() === now.toDateString();
          const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          const label = isToday ? time : d.toLocaleDateString([], { day: 'numeric', month: 'short' }) + ' ' + time;
          const ts = document.createElement('span');
          ts.className = 'cv2-msg-timestamp';
          ts.textContent = label;
          bar.appendChild(ts);
        }
      }

      // Speaker button — only if speakable
      if (this._isSpeakable(text)) {
        const speakBtn = document.createElement('button');
        speakBtn.className = 'cv2-msg-action-btn';
        speakBtn.title = this.t('chat.speak_message');
        speakBtn.innerHTML = '<span class="material-icons" style="font-size:16px">volume_up</span>';
        speakBtn.addEventListener('click', () => this._speakMessage(text, speakBtn));
        bar.appendChild(speakBtn);
      }

      // Copy button — always
      const copyBtn = document.createElement('button');
      copyBtn.className = 'cv2-msg-action-btn';
      copyBtn.title = this.t('chat.copy_message');
      copyBtn.innerHTML = '<span class="material-icons" style="font-size:16px">content_copy</span>';
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(text);
          copyBtn.innerHTML = '<span class="material-icons" style="font-size:16px">check</span>';
          copyBtn.style.color = '#10b981';
          setTimeout(() => {
            copyBtn.innerHTML = '<span class="material-icons" style="font-size:16px">content_copy</span>';
            copyBtn.style.color = '';
          }, 2000);
        } catch (e) {
          console.error('Copy failed:', e);
        }
      });
      bar.appendChild(copyBtn);

      bodyEl.style.position = 'relative';
      bodyEl.appendChild(bar);
    },

    /** @deprecated Use _addMessageActions instead */
    _addCopyButton(container, text) {
      this._addMessageActions(container, text);
    },

    // ── Re-render on Language Change ──────────────────────

    _rerenderTranslatedUI() {
      const greetName = this.userName || 'there';
      if (this.el.greeting) this.el.greeting.textContent = this._getGreeting(greetName);
      if (this.el.sidebarUserName) this.el.sidebarUserName.textContent = this.fullName || this.t('chat.default_user');

      // Sidebar title
      const sidebarTitle = document.querySelector('.cv2-sidebar-title');
      if (sidebarTitle) sidebarTitle.textContent = this.config.appTitle || this.t('chat.sidebar_title');

      // Disclaimer
      const disclaimer = document.getElementById('cv2-disclaimer');
      if (disclaimer) disclaimer.textContent = this.t('chat.disclaimer');

      // Button tooltips
      if (this.el.sidebarClose) this.el.sidebarClose.title = this.t('chat.toggle_sidebar');
      if (this.el.plusBtn) this.el.plusBtn.title = this.t('chat.tools_actions');
      if (this.el.suggestionsBtn) this.el.suggestionsBtn.title = this.t('chat.suggestions');
      if (this.el.sendBtn) this.el.sendBtn.title = this.t('chat.send');
      if (this.el.speechModeBtn) this.el.speechModeBtn.title = this.t('chat.speech_mode');

      // New Chat button
      if (this.el.newChat) {
        this.el.newChat.innerHTML = `<span class="material-icons" style="font-size:16px">add</span> ${this.t('chat.new_chat')}`;
      }

      // Search Chats button
      const searchBtn = document.getElementById('cv2-search-chats');
      if (searchBtn) {
        const kbd = navigator.platform.includes('Mac') ? '\u2318K' : 'Ctrl+K';
        searchBtn.title = this.t('chat.search_chats');
        searchBtn.innerHTML = `<span class="material-icons" style="font-size:16px">search</span> ${this.t('chat.search_chats')} <span class="cv2-search-chats-kbd">${kbd}</span>`;
      }

      // Gear popover items — refresh theme selector label on language change
      if (this._renderThemeSelector) this._renderThemeSelector();
      if (this.el.speechToggle) {
        const icon = this._speechResponse ? 'volume_off' : 'volume_up';
        this.el.speechToggle.innerHTML = `<span class="material-icons" id="cv2-speech-icon">${icon}</span> ${this.t('chat.speech_response')}`;
        this.el.speechIcon = this.el.speechToggle.querySelector('#cv2-speech-icon');
      }
      if (this.el.exportAll) {
        this.el.exportAll.innerHTML = `<span class="material-icons">download</span> ${this.t('chat.export')}`;
      }
      if (this.el.importAll) {
        this.el.importAll.innerHTML = `<span class="material-icons">upload</span> ${this.t('chat.import') || 'Import'}`;
      }
      if (this.el.clearAll) {
        this.el.clearAll.innerHTML = `<span class="material-icons">delete_sweep</span> ${this.t('chat.clear_all')}`;
      }

      // No saved conversations
      this._renderSidebar();

      // Update model dropdown labels
      this.updateModelButton();
    },

    /** Render language selector inside the gear popover. */
    _renderLanguageDropdown() {
      if (!this.supportedLanguages || this.supportedLanguages.length <= 1) return;
      const slot = this.el.gearLangSlot;
      if (!slot) return;

      const current = this.supportedLanguages.find(l => l.code === this.config.locale) || this.supportedLanguages[0];

      slot.innerHTML = `
        <button class="cv2-gear-popover-item" id="cv2-lang-gear-btn">
          <span class="material-icons">language</span>
          <span>${this.t('chat.language')}</span>
          <span style="margin-left:auto;font-size:13px">${current.flag || ''}</span>
        </button>
        <div id="cv2-lang-submenu" style="display:none;padding:0 4px 4px">
          ${this.supportedLanguages.map(lang => `
            <button class="cv2-lang-popover-item ${lang.code === this.config.locale ? 'cv2-active' : ''}" data-lang="${this._escAttr(lang.code)}" style="padding-left:20px">
              ${lang.flag || ''} ${this._escHtml(lang.label || lang.code)}
            </button>
          `).join('')}
        </div>
      `;

      const langBtn = slot.querySelector('#cv2-lang-gear-btn');
      const submenu = slot.querySelector('#cv2-lang-submenu');

      langBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        submenu.style.display = submenu.style.display === 'none' ? 'block' : 'none';
      });

      submenu.querySelectorAll('.cv2-lang-popover-item').forEach(item => {
        item.addEventListener('click', (e) => {
          e.stopPropagation();
          this._closeGearMenu();
          submenu.style.display = 'none';
          this.ws.send({ type: 'change_language', language: item.dataset.lang });
        });
      });
    },

    // ── Placeholder Cycle ─────────────────────────────────

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

      // Bind listeners only once
      if (!this._phListenersBound) {
        this._phListenersBound = true;
        this.el.textarea.addEventListener('input', () => this._updatePhOverlayVisibility());
        this.el.textarea.addEventListener('focus', () => { if (this.el.phOverlay) this.el.phOverlay.style.display = 'none'; });
        this.el.textarea.addEventListener('blur', () => this._updatePhOverlayVisibility());
        this.el.phActionBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          const qa = this.quickActions[this._placeholderIdx];
          if (qa && this._actionableQAs.has(qa.id)) this._triggerQuickAction(qa);
        });
      }

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
          // After transition completes, swap roles (instant — no animation)
          this._placeholderSwap = setTimeout(() => {
            this._placeholderSwap = null;
            cur.style.transition = 'none';
            nxt.style.transition = 'none';
            cur.textContent = nxt.textContent;
            cur.style.opacity = '1';
            nxt.style.opacity = '0';
            requestAnimationFrame(() => {
              cur.style.transition = '';
              nxt.style.transition = '';
            });
          }, 800);
        }, 8000);
      }, 2500);
    },

    _stopPlaceholderCycle() {
      if (this._placeholderTimer) {
        clearInterval(this._placeholderTimer);
        this._placeholderTimer = null;
      }
      if (this._placeholderDelay) {
        clearTimeout(this._placeholderDelay);
        this._placeholderDelay = null;
      }
      if (this._placeholderSwap) {
        clearTimeout(this._placeholderSwap);
        this._placeholderSwap = null;
      }
      if (this.el?.phOverlay) {
        this.el.phOverlay.style.display = 'none';
      }
      // Reset both spans to clean state
      if (this.el?.phCurrent) {
        this.el.phCurrent.style.transition = 'none';
        this.el.phCurrent.style.opacity = '1';
      }
      if (this.el?.phNext) {
        this.el.phNext.style.transition = 'none';
        this.el.phNext.style.opacity = '0';
      }
    },

    _updatePhOverlayVisibility() {
      if (!this.el?.phOverlay) return;
      this.el.phOverlay.style.display = this.el.textarea.value.trim() ? 'none' : '';
    },

    _updatePhActionBtn() {
      if (!this.el?.phOverlay || !this._actionableQAs) return;
      const qa = this.quickActions[this._placeholderIdx];
      const show = qa && this._actionableQAs.has(qa.id);
      this.el.phOverlay.classList.toggle('cv2-ph-has-action', show);
    },

    _closeSuggestionsMenu() {
      this.suggestionsMenuOpen = false;
      this.el.suggestionsMenu.classList.remove('cv2-visible');
    },

    /** Close all popup menus, optionally keeping one open. */
    _closeAllPopups(except) {
      if (except !== 'model') {
        this.modelDropdownOpen = false;
        this.el.modelDropdown.style.display = 'none';
      }
      if (except !== 'plus') {
        this.plusMenuOpen = false;
        this.el.plusMenu.classList.remove('cv2-visible');
      }
      if (except !== 'suggestions') {
        this.suggestionsMenuOpen = false;
        this.el.suggestionsMenu.classList.remove('cv2-visible');
      }
      if (except !== 'context') {
        this.contextPopoverOpen = false;
        this.el.contextPopover.classList.remove('cv2-visible');
      }
      if (except !== 'gear') {
        this._closeGearMenu();
      }
      if (except !== 'voicePicker' && this.el.voicePicker) this.el.voicePicker.style.display = 'none';
      if (except !== 'admin' && this.el.adminPopover) this.el.adminPopover.classList.remove('cv2-visible');
      if (except !== 'devSettings' && this.el.devSettingsPopover) {
        this.devSettingsOpen = false;
        this.el.devSettingsPopover.classList.remove('cv2-visible');
      }
    },

    _closeGearMenu() {
      this.gearMenuOpen = false;
      this.el.gearPopover.classList.remove('cv2-visible');
      this.el.voiceSettingsMenu.style.display = 'none';
      if (this.el.dataMgmtPopover) this.el.dataMgmtPopover.style.display = 'none';
      const themeSub = document.getElementById('cv2-theme-submenu');
      if (themeSub) themeSub.style.display = 'none';
    },


    // ── Model Button ──────────────────────────────────────

    updateModelButton() {
      const info = this.models.find(m => m.model === this.currentModel || m.name === this.currentModel);
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
      const filtered = this._allowedModels
        ? this.models.filter(m => this._allowedModels.has(m.model) || this._allowedModels.has(m.name))
        : this.models;
      const sorted = [...filtered].sort((a, b) => (b.popularity || 0) - (a.popularity || 0));

      this.el.modelDropdown.innerHTML = sorted.map(m => {
        const sel = m.model === this.currentModel;
        const hl = (m.highlights || []).map(h => `<span class="cv2-model-tag">${this._escHtml(h)}</span>`).join('');
        return `
          <div class="cv2-model-option ${sel ? 'cv2-selected' : ''}" data-model="${this._escAttr(m.model)}">
            <div class="cv2-model-opt-top">
              <img src="${this.config.staticBase}/${m.icon}" alt="">
              <span class="cv2-model-opt-name">${this._escHtml(m.label)}</span>
              <span class="cv2-model-opt-use">${this._escHtml(m.best_use || this.t('chat.model_general'))}</span>
              ${sel ? '<span class="material-icons cv2-model-check">check</span>' : ''}
            </div>
            <div class="cv2-model-bars">
              <div class="cv2-model-bar-col">
                <span class="cv2-model-bar-label">${this.t('chat.model_speed')}</span>
                ${bar(m.speed || 5)}
              </div>
              <div class="cv2-model-bar-col">
                <span class="cv2-model-bar-label">${this.t('chat.model_quality')}</span>
                ${bar(m.quality || 5)}
              </div>
              <div class="cv2-model-bar-col">
                <span class="cv2-model-bar-label">${this.t('chat.model_cost')}</span>
                ${bar(m.cost || 5)}
              </div>
              <div class="cv2-model-bar-col">
                <span class="cv2-model-bar-label">${this.t('chat.model_context')}</span>
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
    },

    _updateAvatarTooltip() {
      const root = document.getElementById('chat-app');
      if (!root) return;
      const name = this.fullName || this.userName || '';
      let tip = name;
      if (this.config.showBudget && this.budget !== undefined && this.budget > 0) {
        tip += `\n${this.t('chat.ctx_budget')}: $${Number(this.budget).toFixed(2)}`;
      }
      // Update sidebar avatar tooltip
      const sidebarAvatar = root.querySelector('.cv2-sidebar-avatar');
      if (sidebarAvatar) sidebarAvatar.title = tip;
      // Hide standalone budget element — budget now lives in context popover + avatar tooltip
      if (this.el.budget) this.el.budget.style.display = 'none';
    },

    updateSettings() {
      // Settings are now admin-only, no UI to update
    },

    // ── Document commands (debug API) ────────────────────────
    handleDocCommand(msg) {
      var api = window._cv2DocApi;
      if (!api) {
        this.ws.send({type: 'doc_command_result', request_id: msg.request_id, ok: false, error: 'Doc API not loaded'});
        return;
      }
      var cmd = msg.command;
      var result;
      try {
        switch (cmd) {
          case 'list_documents': result = api.listDocuments(); break;
          case 'open_windowed':  result = api.openWindowed(msg.block_id); break;
          case 'close_window':   result = api.closeWindow(); break;
          case 'maximize':       result = api.maximize(); break;
          case 'restore':        result = api.restore(); break;
          case 'get_state':      result = api.getState(); break;
          case 'get_content':    result = api.getContent(msg.format); break;
          case 'select_text':    result = api.selectText(msg.text); break;
          case 'get_selection':  result = api.getSelection(); break;
          case 'set_cursor':     result = api.setCursor(msg.position, msg.after_text); break;
          case 'type_text':      result = api.typeText(msg.text); break;
          case 'scroll_doc':     result = api.scrollDoc(msg.position); break;
          default: result = {ok: false, error: 'Unknown command: ' + cmd};
        }
      } catch (err) {
        result = {ok: false, error: String(err)};
      }
      result.request_id = msg.request_id;
      result.type = 'doc_command_result';
      this.ws.send(result);
    },

  });

  ChatFeatures.register('messages', {
    handleMessage: {
      'session_init': 'handleSessionInit',
      'response_started': 'handleResponseStarted',
      'text_chunk': 'handleTextChunk',
      'tool_event': 'handleToolEvent',
      'image_received': 'handleImageReceived',
      'pdf_page_preview': 'handlePdfPagePreview',
      'response_completed': 'handleResponseCompleted',
      'response_cancelled': 'handleResponseCancelled',
      'error': 'handleError',
      'model_switched': 'handleModelSwitched',
      'model_locked': 'handleModelLocked',
      'model_restricted': 'handleModelRestricted',
      'model_unlocked': 'handleModelUnlocked',
      'tools_updated': 'handleToolsUpdated',
      'register_renderers': 'handleRegisterRenderers',
      'context_info': 'handleContextInfo',
      'title_generated': 'handleTitleGenerated',
      'condense_start': 'handleCondenseStart',
      'condense_progress': 'handleCondenseProgress',
      'condense_end': 'handleCondenseEnd',
      'save_conversation': 'handleSaveConversation',
      'session_id_updated': 'handleSessionIdUpdated',
      'budget_update': 'handleBudgetUpdate',
      'chat_cleared': 'handleChatCleared',
      'user_message': 'handleUserMessage',
      'files_updated': 'handleFilesUpdated',
      'ui_action': 'handleUIAction',
      'doc_command': 'handleDocCommand',
      'language_changed': 'handleLanguageChanged',
      'action_callback_result': 'handleActionCallbackResult',
      'notification': 'handleNotification',
      'dev_mode': 'handleDevMode',
      'kantini_likes_update': 'handleKantiniLikesUpdate',
    },
  });
})();
