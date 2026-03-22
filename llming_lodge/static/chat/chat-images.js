/**
 * chat-images.js — Image/file upload, paste, drag-drop, lightbox
 * Extracted from chat-app.js
 */
(function() {
  Object.assign(window._ChatAppProto, {

    // ── Inline Image Extraction ───────────────────────────

    _extractInlineImages(text) {
      const images = [];
      if (!text) return { images, text: text || '' };
      let cleaned = text;

      // Markdown images with base64
      const mdRe = /!\[[^\]]*\]\((data:image\/[^;]+;base64,[A-Za-z0-9+/=]+)\)/g;
      let m;
      while ((m = mdRe.exec(text)) !== null) {
        images.push(m[1]);
        cleaned = cleaned.replace(m[0], '');
      }

      // Strip attachment:// placeholders and hallucinated local paths
      // Models often output ![alt](/mnt/data/...) or ![alt](sandbox:/...) after image generation
      cleaned = cleaned.replace(/!\[[^\]]*\]\((?:attachment:\/\/|sandbox:|\/mnt\/data\/|file:\/\/)[^)]*\)/g, '');

      // Plain data URIs
      const plainRe = /^(data:image\/[^;]+;base64,[A-Za-z0-9+/=]+)$/gm;
      while ((m = plainRe.exec(cleaned)) !== null) {
        if (!images.includes(m[1])) images.push(m[1]);
        cleaned = cleaned.replace(m[0], '');
      }

      cleaned = cleaned.replace(/\n{3,}/g, '\n\n').trim();
      return { images, text: cleaned };
    },

    // ── Image Wrapping & Lightbox ─────────────────────────

    _wrapImage(img) {
      const wrap = document.createElement('div');
      wrap.className = 'cv2-img-wrap';

      const actions = document.createElement('div');
      actions.className = 'cv2-img-actions';

      const copyBtn = document.createElement('button');
      copyBtn.className = 'cv2-img-action-btn';
      copyBtn.title = this.t('chat.copy_image');
      copyBtn.innerHTML = '<span class="material-icons">content_copy</span>';
      copyBtn.addEventListener('click', (e) => { e.stopPropagation(); this._copyImage(img.src, copyBtn); });

      const dlBtn = document.createElement('button');
      dlBtn.className = 'cv2-img-action-btn';
      dlBtn.title = this.t('chat.download_image');
      dlBtn.innerHTML = '<span class="material-icons">download</span>';
      dlBtn.addEventListener('click', (e) => { e.stopPropagation(); this._downloadImage(img.src); });

      actions.appendChild(copyBtn);
      actions.appendChild(dlBtn);

      // Replace img in DOM with wrapper
      img.parentNode?.insertBefore(wrap, img);
      wrap.appendChild(img);
      wrap.appendChild(actions);

      img.addEventListener('click', () => this._openLightbox(img.src));
      return wrap;
    },

    /** Wrap all images (generated AND user-uploaded) with lightbox/copy/download. */
    _wrapAllImages(container) {
      // Generated images (assistant messages)
      container.querySelectorAll('img.cv2-generated-image').forEach(img => {
        if (!img.parentElement?.classList.contains('cv2-img-wrap')) {
          this._wrapImage(img);
        }
      });
      // User-uploaded images (user messages)
      container.querySelectorAll('.cv2-msg-user-images img').forEach(img => {
        if (!img.parentElement?.classList.contains('cv2-img-wrap')) {
          this._wrapImage(img);
        }
      });
    },

    _openLightbox(src) {
      // Remove existing
      document.querySelector('.cv2-lightbox')?.remove();

      const lb = document.createElement('div');
      lb.className = 'cv2-lightbox';
      lb.addEventListener('click', (e) => { if (e.target === lb) lb.remove(); });

      const img = document.createElement('img');
      img.src = src;
      lb.appendChild(img);

      const actions = document.createElement('div');
      actions.className = 'cv2-lightbox-actions';

      const copyBtn = document.createElement('button');
      copyBtn.className = 'cv2-lightbox-btn';
      copyBtn.innerHTML = `<span class="material-icons">content_copy</span> ${this.t('chat.copy')}`;
      copyBtn.addEventListener('click', () => this._copyImage(src, copyBtn));

      const dlBtn = document.createElement('button');
      dlBtn.className = 'cv2-lightbox-btn';
      dlBtn.innerHTML = `<span class="material-icons">download</span> ${this.t('chat.download')}`;
      dlBtn.addEventListener('click', () => this._downloadImage(src));

      actions.appendChild(copyBtn);
      actions.appendChild(dlBtn);
      lb.appendChild(actions);

      const closeBtn = document.createElement('button');
      closeBtn.className = 'cv2-lightbox-btn cv2-lightbox-close';
      closeBtn.innerHTML = '<span class="material-icons">close</span>';
      closeBtn.addEventListener('click', () => lb.remove());
      lb.appendChild(closeBtn);

      // ESC to close
      const onKey = (e) => { if (e.key === 'Escape') { lb.remove(); document.removeEventListener('keydown', onKey); } };
      document.addEventListener('keydown', onKey);

      document.body.appendChild(lb);
    },

    async _copyImage(src, btn) {
      try {
        const resp = await fetch(src);
        const blob = await resp.blob();
        const pngBlob = blob.type === 'image/png' ? blob : await this._toPngBlob(src);
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': pngBlob })]);
        const orig = btn.innerHTML;
        btn.innerHTML = btn.classList.contains('cv2-lightbox-btn')
          ? `<span class="material-icons">check</span> ${this.t('chat.copied')}`
          : '<span class="material-icons">check</span>';
        setTimeout(() => { btn.innerHTML = orig; }, 2000);
      } catch (e) {
        console.error('Copy image failed:', e);
      }
    },

    _toPngBlob(src) {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
          const c = document.createElement('canvas');
          c.width = img.naturalWidth;
          c.height = img.naturalHeight;
          c.getContext('2d').drawImage(img, 0, 0);
          c.toBlob(resolve, 'image/png');
        };
        img.src = src;
      });
    },

    _downloadImage(src) {
      const a = document.createElement('a');
      a.href = src;
      a.download = `generated-image-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    },

    // ── PDF Viewer Modal ────────────────────────────────────

    _openPdfViewer(url, page) {
      document.querySelector('.cv2-pdf-viewer-modal')?.remove();

      const modal = document.createElement('div');
      modal.className = 'cv2-pdf-viewer-modal';

      const toolbar = document.createElement('div');
      toolbar.className = 'cv2-pdf-viewer-toolbar';
      toolbar.innerHTML = `
        <span style="font-size:14px;opacity:0.7">PDF Viewer</span>
        <button class="cv2-lightbox-btn cv2-pdf-viewer-close">
          <span class="material-icons">close</span>
        </button>
      `;
      toolbar.querySelector('.cv2-pdf-viewer-close').addEventListener('click', () => modal.remove());

      const iframe = document.createElement('iframe');
      iframe.className = 'cv2-pdf-viewer-iframe';
      iframe.src = `${url}#page=${page}`;

      modal.appendChild(toolbar);
      modal.appendChild(iframe);
      modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });

      const onKey = (e) => { if (e.key === 'Escape') { modal.remove(); document.removeEventListener('keydown', onKey); } };
      document.addEventListener('keydown', onKey);

      document.body.appendChild(modal);
    },

    // ── File Drop, Paste & Selection ──────────────────────

    async _handleFileDrop(dataTransfer) {
      if (!dataTransfer.files.length) return;
      const files = Array.from(dataTransfer.files);
      const imageFiles = files.filter(f => f.type.startsWith('image/'));
      const docFiles = files.filter(f => !f.type.startsWith('image/'));

      // Compress and add images
      for (const f of imageFiles) {
        const dataUri = await this._compressImage(f);
        if (dataUri) await this._addPendingImage(dataUri);
      }

      // Upload document files
      if (docFiles.length) await this._uploadFiles(docFiles);
    },

    async _handlePaste(e) {
      const items = Array.from(e.clipboardData?.items || []);
      const imageItems = items.filter(i => i.type.startsWith('image/'));
      if (imageItems.length === 0) return;

      e.preventDefault();
      for (const item of imageItems) {
        const file = item.getAsFile();
        if (!file) continue;
        await this._checkImageDimensions(file);
        const dataUri = await this._compressImage(file, ChatApp.MAX_IMAGE_DIM);
        if (dataUri) await this._addPendingImage(dataUri);
      }
    },

    // ── File Stats & Validation ───────────────────────────

    /** Count all files across preset + chat attachments. */
    _getFileStats(extraPresetFiles) {
      let count = 0;
      let totalSize = 0;
      // Active preset files
      const presetFiles = extraPresetFiles || this._activePresetFiles || [];
      count += presetFiles.length;
      totalSize += presetFiles.reduce((sum, f) => sum + (f.size || 0), 0);
      // Pending chat files
      count += this._pendingFiles.length;
      totalSize += this._pendingFiles.reduce((sum, f) => sum + (f.size || 0), 0);
      // Pending images (estimate ~200KB each for objects with dataUri)
      count += this._pendingImages.length;
      totalSize += this._pendingImages.length * 200000;
      return { count, totalSize };
    },

    /** Validate new files against central limits. Returns error string or null. */
    _validateNewFiles(newFiles, existingPresetFiles) {
      const stats = this._getFileStats(existingPresetFiles);
      const newCount = stats.count + newFiles.length;
      if (newCount > ChatApp.MAX_FILES) {
        return this.t('chat.error_max_files', { max: String(ChatApp.MAX_FILES) });
      }
      // MIME types that support server-side text extraction for oversized files
      const _EXTRACTABLE_EXTS = new Set(['pdf', 'docx']);
      let newTotalSize = stats.totalSize;
      for (const f of newFiles) {
        if (f.size > ChatApp.MAX_SINGLE_FILE) {
          const ext = f.name.split('.').pop()?.toLowerCase() || '';
          if (_EXTRACTABLE_EXTS.has(ext) && f.size <= 50 * 1024 * 1024) {
            // Allow — server will extract text from oversized PDF/DOCX
            this._showToast(
              this.t('chat.info_large_file_extract', {
                name: f.name,
                size: (f.size / (1024 * 1024)).toFixed(1) + ' MB',
              }) || `${f.name} (${(f.size / (1024*1024)).toFixed(1)} MB) is large — text will be extracted automatically.`,
              'info'
            );
          } else {
            return this.t('chat.error_file_too_large', { name: f.name, max: '5 MB' });
          }
        }
        newTotalSize += f.size;
      }
      // Skip total size check for extracted files (they don't consume raw_data memory)
      const nonExtractableSize = Array.from(newFiles).reduce((sum, f) => {
        const ext = f.name.split('.').pop()?.toLowerCase() || '';
        return sum + (f.size > ChatApp.MAX_SINGLE_FILE && _EXTRACTABLE_EXTS.has(ext) ? 0 : f.size);
      }, stats.totalSize);
      if (nonExtractableSize > ChatApp.MAX_TOTAL_SIZE) {
        return this.t('chat.error_total_size', { max: '10 MB' });
      }
      return null;
    },

    /** Check image dimensions and warn if exceeding UHD. */
    _checkImageDimensions(file) {
      return new Promise((resolve) => {
        if (!file.type.startsWith('image/')) { resolve(true); return; }
        const img = new Image();
        img.onload = () => {
          URL.revokeObjectURL(img.src);
          if (img.width > ChatApp.MAX_IMAGE_DIM || img.height > ChatApp.MAX_IMAGE_DIM) {
            this._showToast(this.t('chat.error_image_too_large', { max: 'UHD (3840px)' }), 'warning');
          }
          resolve(true);
        };
        img.onerror = () => { URL.revokeObjectURL(img.src); resolve(true); };
        img.src = URL.createObjectURL(file);
      });
    },

    /** Peek into file to check for excessively large content (Excel rows, etc). */
    async _peekFile(file) {
      const ext = file.name.split('.').pop().toLowerCase();
      // Large Excel files: rough heuristic — 50KB per 1000 rows, so >5MB could be 100k+ rows
      if ((ext === 'xlsx' || ext === 'xls') && file.size > 5 * 1024 * 1024) {
        this._showToast(this.t('chat.error_large_spreadsheet', { name: file.name }), 'warning');
        return false;
      }
      // PDF > 50MB is too large even for extraction
      if (ext === 'pdf' && file.size > 50 * 1024 * 1024) {
        return false;
      }
      return true;
    },

    async _handleFileSelection(files) {
      // Central validation
      const error = this._validateNewFiles(files);
      if (error) {
        this._showToast(error, 'negative');
        return;
      }

      // Peek into potentially huge files
      const validated = [];
      for (const f of files) {
        if (await this._peekFile(f)) validated.push(f);
      }
      if (validated.length === 0) return;

      const imageFiles = validated.filter(f => f.type.startsWith('image/'));
      const docFiles = validated.filter(f => !f.type.startsWith('image/'));

      for (const f of imageFiles) {
        await this._checkImageDimensions(f);
        const dataUri = await this._compressImage(f, ChatApp.MAX_IMAGE_DIM);
        if (dataUri) await this._addPendingImage(dataUri);
      }

      if (docFiles.length) await this._uploadFiles(docFiles);

      // Fire callback (e.g. auto-analyze from quick action)
      if (this._onFilesReady) {
        const cb = this._onFilesReady;
        this._onFilesReady = null;
        cb();
      }
    },

    // ── Pending Images & Files ────────────────────────────

    async _addPendingImage(dataUri) {
      if (this._pendingImages.length >= 4) return; // max 4 images
      let hash = null;
      try {
        const buffer = this._dataUriToBuffer(dataUri);
        hash = await this._computeHash(buffer);
        const convId = this.activeConvId || '_pending';
        const idx = this._pendingImages.length + 1;
        await this.idb.putFile(hash, `Image ${idx}`, 'image/jpeg', buffer.byteLength, buffer, convId);
      } catch (_) {}
      this._pendingImages.push({ dataUri, hash });
      this._renderAttachments();

      // Also post to server (so server can include with next message)
      fetch(`/api/llming/image-paste/${this.sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: this._pendingImages.map(i => i.dataUri.split(',')[1]) }),
      }).catch(err => console.error('[Paste] POST failed:', err));
      if (this.activeConvId) this._persistFileRefsNow();
    },

    _removePendingImage(index) {
      if (typeof _dismissPreview === 'function') _dismissPreview();
      const removed = this._pendingImages[index];
      this._pendingImages.splice(index, 1);
      this._renderAttachments();
      // Update server
      fetch(`/api/llming/image-paste/${this.sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: this._pendingImages.map(i => i.dataUri.split(',')[1]) }),
      }).catch(() => {});
      // Remove from saved refs + IDB so it doesn't reappear on conversation switch
      if (removed?.hash && this.activeConvId) {
        this._purgeFileRef(this.activeConvId, removed.hash);
      }
    },

    _removePendingFile(fileId) {
      if (typeof _dismissPreview === 'function') _dismissPreview();
      // Find hash before removing so we can clean up saved refs
      const removed = this._pendingFiles.find(f => f.fileId === fileId);
      this._pendingFiles = this._pendingFiles.filter(f => f.fileId !== fileId);
      this._renderAttachments();
      // Notify server to remove file and rebuild document context
      this.ws.send({ type: 'file_removed', file_id: fileId });
      // Remove from saved refs + IDB so it doesn't reappear on conversation switch
      if (removed?.hash && this.activeConvId) {
        this._purgeFileRef(this.activeConvId, removed.hash);
      }
    },

    /** Remove a file ref from _savedFileRefs and IDB for a conversation. */
    _purgeFileRef(convId, hash) {
      const refs = this._savedFileRefs[convId];
      if (refs) {
        this._savedFileRefs[convId] = refs.filter(r => r.hash !== hash);
        if (this._savedFileRefs[convId].length === 0) delete this._savedFileRefs[convId];
      }
      // Update IDB conversation record
      this.idb?.get(convId).then(data => {
        if (!data?.file_refs) return;
        data.file_refs = data.file_refs.filter(r => r.hash !== hash);
        this.idb.put(data);
      }).catch(() => {});
    },

    _renderAttachments() {
      const hasAttachments = this._pendingImages.length > 0 || this._pendingFiles.length > 0;
      this.el.attachments.style.display = hasAttachments ? 'flex' : 'none';
      this.el.attachments.innerHTML = '';

      // Image pills (compact: tiny thumbnail + name)
      for (let i = 0; i < this._pendingImages.length; i++) {
        const imgItem = this._pendingImages[i];
        const dataUri = imgItem.dataUri;
        const pill = document.createElement('div');
        pill.className = 'cv2-attachment';
        const img = document.createElement('img');
        img.className = 'cv2-attachment-thumb';
        img.src = dataUri;
        img.alt = '';
        pill.appendChild(img);
        const nameSpan = document.createElement('span');
        nameSpan.textContent = `Image ${i + 1}`;
        pill.appendChild(nameSpan);
        const removeSpan = document.createElement('span');
        removeSpan.className = 'cv2-attachment-remove';
        removeSpan.innerHTML = '&times;';
        removeSpan.addEventListener('click', () => this._removePendingImage(i));
        pill.appendChild(removeSpan);
        // Hover + click-to-pin preview
        if (typeof _showAttPreview === 'function') {
          const attData = { type: 'file', name: `Image ${i + 1}`, content_type: 'image/png', data: dataUri.split(',')[1] };
          pill.addEventListener('mouseenter', () => _showAttPreview(pill, attData, this.sessionId));
          pill.addEventListener('mouseleave', () => { if (typeof _scheduleDismiss === 'function') _scheduleDismiss(); });
          pill.addEventListener('click', (e) => {
            if (e.target.closest('.cv2-attachment-remove')) return;
            if (typeof _showAttPreviewPinned === 'function') _showAttPreviewPinned(pill, attData, this.sessionId);
          });
        }
        this.el.attachments.appendChild(pill);
      }

      // File pills with hover + click-to-pin preview
      for (const f of this._pendingFiles) {
        const icon = this._fileIcon(f.mimeType);
        const sizeStr = f.size < 1024 ? `${f.size} B` : f.size < 1048576 ? `${(f.size / 1024).toFixed(0)} KB` : `${(f.size / 1048576).toFixed(1)} MB`;
        const pill = document.createElement('div');
        pill.className = 'cv2-attachment';
        const isImg = f.mimeType?.startsWith('image/');
        if (isImg && f.fileId && this.sessionId) {
          const thumb = document.createElement('img');
          thumb.className = 'cv2-attachment-thumb';
          thumb.src = `/api/llming/file-preview/${this.sessionId}/${f.fileId}`;
          thumb.alt = '';
          pill.appendChild(thumb);
        } else {
          const iconEl = document.createElement('span');
          iconEl.className = 'material-icons';
          iconEl.style.cssText = `font-size:14px;color:${icon.color}`;
          iconEl.textContent = icon.name;
          pill.appendChild(iconEl);
        }
        const nameEl = document.createElement('span');
        nameEl.textContent = f.name;
        pill.appendChild(nameEl);
        const sizeEl = document.createElement('span');
        sizeEl.style.cssText = 'color:#9ca3af;font-size:11px';
        sizeEl.textContent = sizeStr;
        pill.appendChild(sizeEl);
        const removeEl = document.createElement('span');
        removeEl.className = 'cv2-attachment-remove';
        removeEl.innerHTML = '&times;';
        removeEl.addEventListener('click', () => this._removePendingFile(f.fileId));
        pill.appendChild(removeEl);
        // Hover + click-to-pin preview
        if (typeof _showAttPreview === 'function') {
          const attData = { type: 'chat_file', fileId: f.fileId, name: f.name, content_type: f.mimeType, size: f.size };
          pill.addEventListener('mouseenter', () => _showAttPreview(pill, attData, this.sessionId));
          pill.addEventListener('mouseleave', () => { if (typeof _scheduleDismiss === 'function') _scheduleDismiss(); });
          pill.addEventListener('click', (e) => {
            if (e.target.closest('.cv2-attachment-remove')) return;
            if (typeof _showAttPreviewPinned === 'function') _showAttPreviewPinned(pill, attData, this.sessionId);
          });
        }
        this.el.attachments.appendChild(pill);
      }

      this._updateSendButton();
    },

    // ── Utilities ─────────────────────────────────────────

    /** Convert raw base64 or data URI to a proper data URI, sniffing MIME from magic bytes. */
    _toDataUri(val) {
      if (val.startsWith('data:')) return val;
      if (val.startsWith('iVBOR')) return `data:image/png;base64,${val}`;
      if (val.startsWith('/9j/'))  return `data:image/jpeg;base64,${val}`;
      if (val.startsWith('R0lG'))  return `data:image/gif;base64,${val}`;
      if (val.startsWith('UklG'))  return `data:image/webp;base64,${val}`;
      return `data:application/octet-stream;base64,${val}`;
    },

    _fileIcon(mimeType) {
      if (!mimeType) return { name: 'description', color: '#6b7280' };
      if (mimeType.includes('pdf')) return { name: 'picture_as_pdf', color: '#ef4444' };
      if (mimeType.includes('word') || mimeType.includes('docx')) return { name: 'description', color: '#2563eb' };
      if (mimeType.includes('sheet') || mimeType.includes('xlsx') || mimeType.includes('excel')) return { name: 'table_chart', color: '#16a34a' };
      if (mimeType.startsWith('image/')) return { name: 'image', color: '#7c3aed' };
      return { name: 'description', color: '#6b7280' };
    },

    async _compressImage(file, maxDim = 1920, maxBytes = 200000) {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => {
          const img = new Image();
          img.onload = () => {
            const canvas = document.createElement('canvas');
            let w = img.width, h = img.height;
            if (w > maxDim || h > maxDim) {
              const ratio = Math.min(maxDim / w, maxDim / h);
              w = Math.round(w * ratio);
              h = Math.round(h * ratio);
            }
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, w, h);

            // Try progressively lower quality
            for (const q of [0.8, 0.6, 0.4, 0.25]) {
              const dataUri = canvas.toDataURL('image/jpeg', q);
              if (dataUri.length * 0.75 <= maxBytes || q === 0.25) {
                resolve(dataUri);
                return;
              }
            }
            resolve(canvas.toDataURL('image/jpeg', 0.25));
          };
          img.onerror = () => resolve(null);
          img.src = reader.result;
        };
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(file);
      });
    },

    // ── File Upload ───────────────────────────────────────

    async _uploadFiles(files) {
      // Read file contents for hashing before upload
      const fileBuffers = [];
      for (const f of files) {
        try { fileBuffers.push({ file: f, buffer: await f.arrayBuffer() }); }
        catch (_) { fileBuffers.push({ file: f, buffer: null }); }
      }

      const formData = new FormData();
      for (const f of files) formData.append('files', f);

      try {
        const res = await fetch(`/api/llming/upload/${this.sessionId}`, {
          method: 'POST',
          headers: { 'X-User-Id': this.config.userId },
          body: formData,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          const msg = err.detail || `Upload failed (${res.status})`;
          this._showToast(msg, 'negative');
          return;
        }
        const data = await res.json();
        if (data.files) {
          for (let i = 0; i < data.files.length; i++) {
            const f = data.files[i];
            const fb = fileBuffers[i];
            // Compute SHA-256 and persist in IDB
            if (fb?.buffer) {
              try {
                const hash = await this._computeHash(fb.buffer);
                f.hash = hash;
                const convId = this.activeConvId || '_pending';
                await this.idb.putFile(hash, f.name, f.mimeType, f.size, fb.buffer, convId);
              } catch (_) {}
            }
            this._pendingFiles.push(f);
          }
          this._renderAttachments();
          // Notify server to rebuild document context
          this.ws.send({ type: 'file_uploaded' });
          // Persist file refs to IDB immediately for existing conversations
          // (so they survive a conversation switch before the user sends a message)
          if (this.activeConvId) this._persistFileRefsNow();
        }
      } catch (err) {
        console.error('[Upload] Failed:', err);
      }
    },

    /** Save current pending file/image refs to _savedFileRefs + IDB conversation record. */
    _persistFileRefsNow() {
      const convId = this.activeConvId;
      if (!convId) return;
      const refs = [
        ...this._pendingFiles.filter(f => f.hash).map(f => ({
          hash: f.hash, name: f.name, mime_type: f.mimeType, size: f.size, type: 'document',
        })),
        ...this._pendingImages.filter(i => i.hash).map((img, idx) => ({
          hash: img.hash, name: `Image ${idx + 1}`, mime_type: 'image/jpeg', size: 0, type: 'image',
        })),
      ];
      if (refs.length === 0) return;
      // Merge with any previously saved refs
      const existing = this._savedFileRefs[convId] || [];
      const merged = [...existing];
      for (const r of refs) {
        if (!merged.some(e => e.hash === r.hash)) merged.push(r);
      }
      this._savedFileRefs[convId] = merged;
      this.idb?.get(convId).then(data => {
        if (!data) return;
        data.file_refs = merged;
        this.idb.put(data);
      }).catch(() => {});
    },

    // ── Hashing & Buffer Helpers ─────────────────────────

    async _computeHash(buffer) {
      const hashBuf = await crypto.subtle.digest('SHA-256', buffer);
      return Array.from(new Uint8Array(hashBuf)).map(b => b.toString(16).padStart(2, '0')).join('');
    },

    _dataUriToBuffer(dataUri) {
      const base64 = dataUri.split(',')[1];
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    },

    _bufferToDataUri(buffer, mimeType) {
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return `data:${mimeType};base64,${btoa(binary)}`;
    },

    // ── WS helper ─────────────────────────────────────────

    _wsFiles(files) {
      return (files || []).map(({ content, ...rest }) => rest);
    },

  });

  ChatFeatures.register('images', {
    initState(app) {
      app._pendingImages = [];
      app._pendingFiles = [];
      // Persisted file refs per conversation — survives _pendingFiles being cleared
      app._savedFileRefs = {};
    },
  });
})();
