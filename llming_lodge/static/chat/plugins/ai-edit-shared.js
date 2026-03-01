/**
 * AI Edit Shared — context menu, diff view, task button, ghost text
 *
 * Loaded Phase 2 (after chat-popup-utils.js, before builtin-plugins.js).
 * Provides shared AI editing functions consumed by text_doc + email plugins.
 * Communicates via WebSocket messages and CustomEvents.
 */
(function () {
  'use strict';

  var _reqCounter = 0;
  function _nextReqId() { return 'ai_' + (++_reqCounter) + '_' + Date.now(); }

  function _getChatApp() {
    return document.querySelector('#chat-app')?.__chatApp || window.__chatApp;
  }

  function _wsSend(msg) {
    var app = _getChatApp();
    if (app && app.ws) app.ws.send(msg);
  }

  function _getLanguage() {
    var cfg = window.__CHAT_CONFIG__ || {};
    return cfg.locale || 'en';
  }

  // ── Word-level diff ──────────────────────────────────────

  function _computeWordDiff(a, b) {
    var wordsA = a.split(/(\s+)/);
    var wordsB = b.split(/(\s+)/);
    var m = wordsA.length, n = wordsB.length;

    // LCS via DP
    var dp = [];
    for (var i = 0; i <= m; i++) {
      dp[i] = [];
      for (var j = 0; j <= n; j++) dp[i][j] = 0;
    }
    for (var i = 1; i <= m; i++) {
      for (var j = 1; j <= n; j++) {
        if (wordsA[i - 1] === wordsB[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
        else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }

    // Backtrack
    var result = [];
    var i = m, j = n;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && wordsA[i - 1] === wordsB[j - 1]) {
        result.unshift({ type: 'same', text: wordsA[i - 1] });
        i--; j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        result.unshift({ type: 'added', text: wordsB[j - 1] });
        j--;
      } else {
        result.unshift({ type: 'removed', text: wordsA[i - 1] });
        i--;
      }
    }
    return result;
  }

  // ── Diff View ────────────────────────────────────────────

  function _showDiffView(original, result, opts) {
    // Remove existing
    document.querySelectorAll('.cv2-ai-diff-panel').forEach(function (p) { p.remove(); });

    var panel = document.createElement('div');
    panel.className = 'cv2-ai-diff-panel';
    var _chatEl = document.getElementById('chat-app');
    if (_chatEl && _chatEl.classList.contains('cv2-dark')) panel.classList.add('cv2-dark');

    var header = document.createElement('div');
    header.className = 'cv2-ai-diff-header';
    header.innerHTML =
      '<span>AI Suggestion</span>' +
      '<div class="cv2-ai-diff-actions">' +
        '<button class="cv2-ai-diff-btn cv2-ai-diff-accept" title="Accept"><span class="material-icons" style="font-size:16px">check</span> Accept</button>' +
        '<button class="cv2-ai-diff-btn cv2-ai-diff-reject" title="Reject"><span class="material-icons" style="font-size:16px">close</span> Reject</button>' +
      '</div>';
    panel.appendChild(header);

    var body = document.createElement('div');
    body.className = 'cv2-ai-diff-body';

    var diff = _computeWordDiff(original, result);

    // Original column
    var colOrig = document.createElement('div');
    colOrig.className = 'cv2-ai-diff-col';
    colOrig.innerHTML = '<div class="cv2-ai-diff-col-label">Original</div>';
    var origContent = document.createElement('div');
    origContent.className = 'cv2-ai-diff-content';
    for (var k = 0; k < diff.length; k++) {
      var d = diff[k];
      if (d.type === 'same') origContent.appendChild(document.createTextNode(d.text));
      else if (d.type === 'removed') {
        var span = document.createElement('span');
        span.className = 'cv2-ai-diff-removed';
        span.textContent = d.text;
        origContent.appendChild(span);
      }
      // 'added' items don't appear in original
    }
    colOrig.appendChild(origContent);

    // Result column
    var colResult = document.createElement('div');
    colResult.className = 'cv2-ai-diff-col';
    colResult.innerHTML = '<div class="cv2-ai-diff-col-label">Suggested</div>';
    var resultContent = document.createElement('div');
    resultContent.className = 'cv2-ai-diff-content';
    for (var k = 0; k < diff.length; k++) {
      var d = diff[k];
      if (d.type === 'same') resultContent.appendChild(document.createTextNode(d.text));
      else if (d.type === 'added') {
        var span = document.createElement('span');
        span.className = 'cv2-ai-diff-added';
        span.textContent = d.text;
        resultContent.appendChild(span);
      }
    }
    colResult.appendChild(resultContent);

    body.appendChild(colOrig);
    body.appendChild(colResult);
    panel.appendChild(body);

    // Position near the editable element
    var posX = 10, posY = 100;
    if (opts && opts.anchor) {
      var anchorRect = opts.anchor.getBoundingClientRect();
      posX = Math.max(10, anchorRect.left);
      posY = Math.min(anchorRect.bottom + 4, window.innerHeight - 320);
    }
    document.body.appendChild(panel);
    cv2PositionPopup(panel, posX, posY);

    // Make diff panel draggable via header
    cv2MakeDraggable(panel, header);

    // Accept — use execCommand for undo support
    header.querySelector('.cv2-ai-diff-accept').addEventListener('click', function () {
      if (opts && opts.onAccept) opts.onAccept(result);
      panel.remove();
    });

    // Reject — just close
    header.querySelector('.cv2-ai-diff-reject').addEventListener('click', function () {
      panel.remove();
    });

    // Close on Escape
    cv2DismissOnOutside(panel, function () { panel.remove(); });

    return panel;
  }

  // ── Context Menu ─────────────────────────────────────────

  function _t(key) {
    var app = _getChatApp();
    return app && app.t ? app.t(key) : key;
  }

  function _buildAIContextMenu(editableEl, opts) {
    editableEl.addEventListener('contextmenu', function (e) {
      var sel = window.getSelection();
      var selectedText = sel ? sel.toString().trim() : '';
      if (!selectedText) return;

      e.preventDefault();

      // Capture range before menu interaction clears it
      var range = sel.rangeCount > 0 ? sel.getRangeAt(0).cloneRange() : null;
      var clickX = e.clientX, clickY = e.clientY;

      var popup = cv2PopupMenu([
        { icon: 'spellcheck', label: _t('ai.fix_grammar'), action: 'fix_grammar' },
        { icon: 'auto_fix_high', label: _t('ai.improve'), action: 'improve' },
        { icon: 'business', label: _t('ai.formal'), action: 'formal' },
        { icon: 'emoji_emotions', label: _t('ai.casual'), action: 'casual' },
        { icon: 'compress', label: _t('ai.shorter'), action: 'shorter' },
        { icon: 'expand', label: _t('ai.expand'), action: 'expand' },
        { icon: 'lightbulb', label: _t('ai.simplify'), action: 'simplify' },
        { icon: 'format_list_bulleted', label: _t('ai.bullet_points'), action: 'bullet_points' },
        '---',
        { icon: 'translate', label: _t('ai.translate'), submenu: [
          { flag: '\ud83c\udde9\ud83c\uddea', label: _t('ai.translate_de'), action: 'translate_de' },
          { flag: '\ud83c\uddec\ud83c\udde7', label: _t('ai.translate_en'), action: 'translate_en' },
          { flag: '\ud83c\uddeb\ud83c\uddf7', label: _t('ai.translate_fr'), action: 'translate_fr' },
          { flag: '\ud83c\uddee\ud83c\uddf9', label: _t('ai.translate_it'), action: 'translate_it' },
          { flag: '\ud83c\uddea\ud83c\uddf8', label: _t('ai.translate_es'), action: 'translate_es' },
          { flag: '\ud83c\udde8\ud83c\uddf3', label: _t('ai.translate_zh'), action: 'translate_zh' },
        ]},
        '---',
        { icon: 'edit', label: _t('ai.custom_prompt'), action: 'custom' },
      ], {
        x: e.clientX,
        y: e.clientY,
        minWidth: 190,
        onAction: function (action) {
          _handleAIAction(action, selectedText, range, sel, editableEl, opts, clickX, clickY);
        },
      });
    });
  }

  /** Handle action from context menu — shows loading, sends WS, shows diff. */
  function _handleAIAction(action, selectedText, range, sel, editableEl, opts, clickX, clickY) {
    if (action === 'custom') {
      cv2PromptDialog(_t('ai.enter_instruction'), {
        placeholder: _t('ai.custom_prompt'),
        confirmLabel: 'OK',
        cancelLabel: _t('ai.cancel'),
      }).then(function (value) {
        if (!value) return;
        _fireAIEdit(action, value, selectedText, range, sel, editableEl, opts, clickX, clickY);
      });
      return;
    }
    _fireAIEdit(action, undefined, selectedText, range, sel, editableEl, opts, clickX, clickY);
  }

  function _fireAIEdit(action, customPrompt, selectedText, range, sel, editableEl, opts, clickX, clickY) {
    var reqId = _nextReqId();

    // Show loading state
    var loadingEl = document.createElement('div');
    loadingEl.className = 'cv2-ai-ctx-loading';
    if (cv2IsDark()) loadingEl.classList.add('cv2-dark');
    loadingEl.innerHTML =
      '<span class="cv2-ai-task-spinner"></span> ' + _t('ai.thinking') +
      '<button class="cv2-ai-cancel-btn" title="' + _t('ai.cancel') + '"><span class="material-icons" style="font-size:14px">close</span></button>';
    document.body.appendChild(loadingEl);
    cv2PositionPopup(loadingEl, clickX, clickY);

    var _cancelled = false;

    loadingEl.querySelector('.cv2-ai-cancel-btn').addEventListener('click', function () {
      _cancelled = true;
      _wsSend({ type: 'ai_edit_cancel', request_id: reqId });
      loadingEl.remove();
    });

    function onResult(ev) {
      if (ev.detail.request_id !== reqId) return;
      document.removeEventListener('cv2:ai-edit-result', onResult);
      loadingEl.remove();

      if (ev.detail.status === 'cancelled' || _cancelled) return;

      if (ev.detail.status === 'error') {
        console.warn('[AI_EDIT] Error:', ev.detail.error);
        return;
      }

      _showDiffView(ev.detail.original_text, ev.detail.result_text, {
        anchor: editableEl,
        onAccept: function (text) {
          if (range) {
            sel.removeAllRanges();
            sel.addRange(range);
          }
          document.execCommand('insertText', false, text);
          if (opts && opts.onInput) opts.onInput();
        },
      });
    }
    document.addEventListener('cv2:ai-edit-result', onResult);

    setTimeout(function () {
      document.removeEventListener('cv2:ai-edit-result', onResult);
      loadingEl.remove();
    }, 35000);

    _wsSend({
      type: 'ai_edit_request',
      request_id: reqId,
      document_id: opts ? opts.documentId : '',
      document_type: opts ? opts.documentType : 'text_doc',
      selected_text: selectedText,
      full_context: editableEl.innerText.substring(0, 3000),
      action: action,
      custom_prompt: customPrompt,
      language: _getLanguage(),
    });
  }

  // ── Markdown → static HTML for document insertion ────────

  /**
   * Render markdown through the chat's MarkdownRenderer, then convert
   * any interactive blocks (mermaid, plotly) to static <img> elements
   * so the result is safe for contenteditable insertion.
   *
   * @param {string} markdown — raw markdown from the LLM
   * @param {string} docType — 'text_doc' | 'email_draft'
   * @returns {Promise<string>} — rendered HTML
   */
  async function _renderMarkdownForDoc(markdown, docType) {
    var chatApp = _getChatApp();
    if (!chatApp || !chatApp.md) {
      // Fallback: insert raw (may already be HTML)
      return markdown;
    }

    var md = chatApp.md;
    var wasStreaming = md.streaming;
    md.streaming = false;
    var html = md.render(markdown);
    md.streaming = wasStreaming;

    // Create off-screen container for plugin hydration
    var temp = document.createElement('div');
    temp.style.cssText = 'position:fixed;left:-9999px;top:-9999px;width:600px';
    document.body.appendChild(temp);
    temp.innerHTML = html;

    // Hydrate any plugin blocks (mermaid, plotly, etc.)
    try {
      if (md.hydratePluginBlocks) await md.hydratePluginBlocks(temp);
    } catch (e) {
      console.warn('[AI_EDIT] Plugin hydration failed:', e);
    }

    // Mermaid → convert SVG to <img>
    var mermaidContainers = temp.querySelectorAll('.cv2-mermaid-container');
    for (var i = 0; i < mermaidContainers.length; i++) {
      var mc = mermaidContainers[i];
      var svg = mc.querySelector('svg');
      if (svg) {
        var svgData = new XMLSerializer().serializeToString(svg);
        var img = document.createElement('img');
        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
        img.style.maxWidth = '100%';
        var block = mc.closest('.cv2-doc-plugin-block') || mc.parentNode;
        block.replaceWith(img);
      }
    }

    // Plotly → render to PNG (white bg for emails)
    var plotlyEls = temp.querySelectorAll('.cv2-plotly-preview');
    for (var j = 0; j < plotlyEls.length; j++) {
      var pe = plotlyEls[j];
      if (window.Plotly && pe.data) {
        try {
          // Always use white background for static image export
          var bgColor = '#ffffff';
          var txtColor = '#1f2937';
          if (pe.layout) {
            pe.layout.paper_bgcolor = bgColor;
            pe.layout.plot_bgcolor = bgColor;
            pe.layout.font = Object.assign(pe.layout.font || {}, { color: txtColor });
          }
          await window.Plotly.relayout(pe, {
            paper_bgcolor: bgColor, plot_bgcolor: bgColor,
            font: { color: txtColor },
          });
          var imgUrl = await window.Plotly.toImage(pe, {
            format: 'png', width: 600, height: 400,
          });
          var plotImg = document.createElement('img');
          plotImg.src = imgUrl;
          plotImg.style.maxWidth = '100%';
          var plotBlock = pe.closest('.cv2-doc-plugin-block') || pe.parentNode;
          plotBlock.replaceWith(plotImg);
        } catch (e) {
          console.warn('[AI_EDIT] Plotly PNG export failed:', e);
        }
      }
    }

    // Remove any remaining plugin block wrappers — keep inner content
    var remaining = temp.querySelectorAll('.cv2-doc-plugin-block');
    for (var k = 0; k < remaining.length; k++) {
      var rb = remaining[k];
      // Replace wrapper with its children
      while (rb.firstChild) rb.parentNode.insertBefore(rb.firstChild, rb);
      rb.remove();
    }

    var result = temp.innerHTML;
    temp.remove();
    return result;
  }

  // ── AI Task Button ───────────────────────────────────────

  function _buildAITaskButton(toolbar, opts) {
    var btn = document.createElement('button');
    btn.className = 'cv2-rich-toolbar-btn cv2-ai-task-btn';
    btn.title = 'AI Generate';
    btn.innerHTML = '<span class="material-icons" style="font-size:18px">auto_awesome</span>';

    var inputWrap = null;

    btn.addEventListener('click', function () {
      // Toggle input
      if (inputWrap && inputWrap.parentNode) {
        inputWrap.remove();
        inputWrap = null;
        return;
      }

      inputWrap = document.createElement('div');
      inputWrap.className = 'cv2-ai-task-input';
      inputWrap.innerHTML =
        '<input type="text" placeholder="Describe what to generate\u2026" class="cv2-ai-task-field" />' +
        '<button class="cv2-ai-task-go" title="Generate"><span class="material-icons" style="font-size:16px">send</span></button>';

      // Insert after toolbar
      toolbar.parentNode.insertBefore(inputWrap, toolbar.nextSibling);

      var field = inputWrap.querySelector('.cv2-ai-task-field');
      var goBtn = inputWrap.querySelector('.cv2-ai-task-go');
      field.focus();

      var _activeReqId = null;

      function doGenerate() {
        var task = field.value.trim();
        if (!task) return;

        var reqId = _nextReqId();
        _activeReqId = reqId;
        field.disabled = true;
        goBtn.innerHTML =
          '<span class="cv2-ai-task-spinner"></span>';

        // Add cancel button next to spinner
        var cancelBtn = document.createElement('button');
        cancelBtn.className = 'cv2-ai-cancel-btn';
        cancelBtn.title = 'Cancel';
        cancelBtn.innerHTML = '<span class="material-icons" style="font-size:14px">close</span>';
        cancelBtn.addEventListener('click', function () {
          _wsSend({ type: 'ai_task_cancel', request_id: reqId });
          _activeReqId = null;
          field.disabled = false;
          goBtn.innerHTML = '<span class="material-icons" style="font-size:16px">send</span>';
          if (cancelBtn.parentNode) cancelBtn.remove();
        });
        inputWrap.appendChild(cancelBtn);

        function onResult(ev) {
          if (ev.detail.request_id !== reqId) return;
          document.removeEventListener('cv2:ai-task-result', onResult);
          _activeReqId = null;

          field.disabled = false;
          goBtn.innerHTML = '<span class="material-icons" style="font-size:16px">send</span>';
          if (cancelBtn.parentNode) cancelBtn.remove();

          if (ev.detail.status === 'cancelled') {
            return;
          }

          if (ev.detail.status === 'error') {
            console.warn('[AI_EDIT] Task error:', ev.detail.error);
            if (inputWrap) inputWrap.remove();
            inputWrap = null;
            return;
          }

          // Render markdown → static HTML via the chat's MarkdownRenderer,
          // then insert at end of the editable area.
          var editableEl = opts ? opts.editableEl : null;
          var docType = opts ? opts.documentType : 'text_doc';
          if (editableEl) {
            _renderMarkdownForDoc(ev.detail.content, docType).then(function (html) {
              editableEl.focus();
              var sel = window.getSelection();
              sel.selectAllChildren(editableEl);
              sel.collapseToEnd();
              document.execCommand('insertHTML', false, html);
              if (opts && opts.onInput) opts.onInput();
            });
          }

          if (inputWrap) inputWrap.remove();
          inputWrap = null;
        }
        document.addEventListener('cv2:ai-task-result', onResult);

        setTimeout(function () {
          document.removeEventListener('cv2:ai-task-result', onResult);
        }, 65000);

        _wsSend({
          type: 'ai_task_request',
          request_id: reqId,
          document_id: opts ? opts.documentId : '',
          document_type: opts ? opts.documentType : 'text_doc',
          task_description: task,
          full_context: opts && opts.editableEl ? opts.editableEl.innerText.substring(0, 3000) : '',
          language: _getLanguage(),
        });
      }

      goBtn.addEventListener('click', doGenerate);
      field.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); doGenerate(); }
        if (e.key === 'Escape') {
          // Cancel if active, otherwise close input
          if (_activeReqId) {
            _wsSend({ type: 'ai_task_cancel', request_id: _activeReqId });
            _activeReqId = null;
          }
          if (inputWrap) inputWrap.remove();
          inputWrap = null;
        }
      });
    });

    // Insert before the last separator or at end
    var sep = toolbar.querySelector('.cv2-rich-toolbar-sep:last-of-type');
    if (sep) toolbar.insertBefore(btn, sep);
    else toolbar.appendChild(btn);
  }

  // ── Ghost Text (Type-Ahead) ──────────────────────────────

  var _GHOST_LS_KEY = 'cv2:ghost-text-enabled';

  function _isGhostEnabled() {
    return localStorage.getItem(_GHOST_LS_KEY) === '1';
  }

  function _setGhostEnabled(on) {
    localStorage.setItem(_GHOST_LS_KEY, on ? '1' : '0');
    document.dispatchEvent(new CustomEvent('cv2:ghost-text-toggled', { detail: { enabled: on } }));
  }

  function _setupGhostText(editableEl, opts) {
    var _debounceTimer = null;
    var _ghostSpan = null;
    var _currentReqId = null;
    var DEBOUNCE_MS = 800;

    function _removeGhost() {
      if (_ghostSpan && _ghostSpan.parentNode) _ghostSpan.parentNode.removeChild(_ghostSpan);
      _ghostSpan = null;
    }

    function _cancelPending() {
      clearTimeout(_debounceTimer);
      if (_currentReqId) {
        _wsSend({ type: 'ai_typeahead_cancel', request_id: _currentReqId });
        _currentReqId = null;
      }
    }

    // On input, cancel pending and schedule new request
    editableEl.addEventListener('input', function () {
      _removeGhost();
      _cancelPending();
      if (!_isGhostEnabled()) return;

      _debounceTimer = setTimeout(function () {
        var sel = window.getSelection();
        if (!sel || sel.rangeCount === 0) return;
        var range = sel.getRangeAt(0);
        if (!range.collapsed) return;

        // Get text before cursor
        var container = range.startContainer;
        var textBefore = '';
        if (container.nodeType === 3) {
          textBefore = container.textContent.substring(0, range.startOffset);
        }
        // Walk up to get more context
        var node = container.nodeType === 3 ? container.parentNode : container;
        while (node && node !== editableEl) {
          if (node.previousSibling) {
            var prev = node.previousSibling;
            textBefore = (prev.textContent || '') + textBefore;
          }
          node = node.parentNode;
        }

        // Get text after cursor
        var textAfter = '';
        if (container.nodeType === 3) {
          textAfter = container.textContent.substring(range.startOffset);
        }
        var nodeAfter = container.nodeType === 3 ? container.parentNode : container;
        while (nodeAfter && nodeAfter !== editableEl) {
          if (nodeAfter.nextSibling) {
            var nxt = nodeAfter.nextSibling;
            textAfter = textAfter + (nxt.textContent || '');
          }
          nodeAfter = nodeAfter.parentNode;
        }

        if (textBefore.trim().length < 5) return;

        _currentReqId = _nextReqId();
        _wsSend({
          type: 'ai_typeahead_request',
          request_id: _currentReqId,
          document_id: opts ? opts.documentId : '',
          document_type: opts ? opts.documentType : 'text_doc',
          text_before_cursor: textBefore,
          text_after_cursor: textAfter,
          document_name: opts ? opts.documentName : '',
          language: _getLanguage(),
        });
      }, DEBOUNCE_MS);
    });

    // Handle suggestion
    function onSuggestion(ev) {
      if (!editableEl.isConnected) {
        document.removeEventListener('cv2:ai-typeahead', onSuggestion);
        return;
      }
      if (ev.detail.request_id !== _currentReqId) return;

      _removeGhost();
      var suggestion = ev.detail.suggestion;
      if (!suggestion) return;

      var sel = window.getSelection();
      if (!sel || sel.rangeCount === 0 || !sel.getRangeAt(0).collapsed) return;
      var range = sel.getRangeAt(0);

      // Ensure leading space when suggestion follows a non-whitespace char
      var displayText = suggestion;
      var _container = range.startContainer;
      if (_container && _container.nodeType === 3 && range.startOffset > 0) {
        var charBefore = _container.textContent.charAt(range.startOffset - 1);
        if (charBefore && !/\s/.test(charBefore) && !/^\s/.test(suggestion)) {
          displayText = ' ' + suggestion;
        }
      }

      _ghostSpan = document.createElement('span');
      _ghostSpan.className = 'cv2-ghost-text';
      _ghostSpan.contentEditable = 'false';
      _ghostSpan.textContent = displayText;

      range.insertNode(_ghostSpan);
      // Move cursor before ghost
      range.setStartBefore(_ghostSpan);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);
    }
    document.addEventListener('cv2:ai-typeahead', onSuggestion);

    // Tab to accept ghost text
    editableEl.addEventListener('keydown', function (e) {
      if (e.key === 'Tab' && _ghostSpan && _ghostSpan.parentNode) {
        e.preventDefault();
        var text = _ghostSpan.textContent;
        var anchorNode = _ghostSpan.previousSibling;
        var insertOffset = anchorNode && anchorNode.nodeType === 3 ? anchorNode.length : -1;
        _ghostSpan.parentNode.removeChild(_ghostSpan);
        _ghostSpan = null;
        // Insert directly into the preceding text node to avoid
        // contenteditable whitespace collapsing.
        if (anchorNode && anchorNode.nodeType === 3 && insertOffset >= 0) {
          var before = anchorNode.textContent.substring(0, insertOffset);
          var after = anchorNode.textContent.substring(insertOffset);
          anchorNode.textContent = before + text + after;
          editableEl.normalize();
          var sel = window.getSelection();
          var range = document.createRange();
          range.setStart(anchorNode, insertOffset + text.length);
          range.collapse(true);
          sel.removeAllRanges();
          sel.addRange(range);
        } else {
          editableEl.normalize();
          var sel = window.getSelection();
          var range = document.createRange();
          range.selectNodeContents(editableEl);
          range.collapse(false);
          sel.removeAllRanges();
          sel.addRange(range);
          document.execCommand('insertText', false, text);
        }
        if (opts && opts.onInput) opts.onInput();
      }
    });

    // Remove ghost on beforeinput (user typing)
    editableEl.addEventListener('beforeinput', function () {
      _removeGhost();
    });

    // Remove ghost on blur
    editableEl.addEventListener('blur', function () {
      _removeGhost();
      _cancelPending();
    });
  }

  // ── WS → CustomEvent bridge ──────────────────────────────

  Object.assign(window._ChatAppProto, {
    _onAIEditResult: function (msg) {
      document.dispatchEvent(new CustomEvent('cv2:ai-edit-result', { detail: msg }));
    },
    _onAITaskResult: function (msg) {
      document.dispatchEvent(new CustomEvent('cv2:ai-task-result', { detail: msg }));
    },
    _onAITypeahead: function (msg) {
      document.dispatchEvent(new CustomEvent('cv2:ai-typeahead', { detail: msg }));
    },
  });

  ChatFeatures.register('ai_edit', {
    handleMessage: {
      'ai_edit_result': '_onAIEditResult',
      'ai_task_result': '_onAITaskResult',
      'ai_typeahead_suggestion': '_onAITypeahead',
    },
  });

  // ── Public API ───────────────────────────────────────────

  window._buildAIContextMenu = _buildAIContextMenu;
  window._showDiffView = _showDiffView;
  window._computeWordDiff = _computeWordDiff;
  window._buildAITaskButton = _buildAITaskButton;
  window._setupGhostText = _setupGhostText;
  window._isGhostEnabled = _isGhostEnabled;
  window._setGhostEnabled = _setGhostEnabled;

})();
