/**
 * chat-search.js — ChatGPT-style search dialog for conversations
 * Trigger: sidebar button or Cmd+K / Ctrl+K
 */
(function() {
  const _SEARCH_PAGE = 50;

  Object.assign(window._ChatAppProto, {

    _openSearchDialog() {
      if (this._searchOverlay) return;

      const overlay = document.createElement('div');
      overlay.className = 'cv2-dialog-overlay cv2-search-overlay';
      overlay.innerHTML = `
        <div class="cv2-search-dialog">
          <div class="cv2-search-header">
            <span class="material-icons cv2-search-header-icon">search</span>
            <input class="cv2-search-input" type="text" placeholder="${this._escAttr(this.t('chat.search_placeholder'))}" autofocus>
            <button class="cv2-search-close"><span class="material-icons" style="font-size:18px">close</span></button>
          </div>
          <div class="cv2-search-results" id="cv2-search-results"></div>
        </div>`;

      document.getElementById('chat-app').appendChild(overlay);
      this._searchOverlay = overlay;
      this._searchSelectedIdx = -1;

      const input = overlay.querySelector('.cv2-search-input');
      const results = overlay.querySelector('#cv2-search-results');

      // Render initial list (all conversations, title-only)
      this._renderSearchResults(this.conversations.slice(0, _SEARCH_PAGE), '', results, true);

      // Close handlers
      overlay.querySelector('.cv2-search-close').addEventListener('click', () => this._closeSearchDialog());
      overlay.addEventListener('click', (e) => { if (e.target === overlay) this._closeSearchDialog(); });

      // Input with debounce
      input.addEventListener('input', () => {
        clearTimeout(this._searchDebounce);
        const query = input.value.trim();
        if (!query) {
          this._renderSearchResults(this.conversations.slice(0, _SEARCH_PAGE), '', results, true);
          return;
        }
        this._searchDebounce = setTimeout(() => this._performSearch(query, results), 200);
      });

      // Keyboard navigation
      input.addEventListener('keydown', (e) => {
        const items = results.querySelectorAll('.cv2-search-result, .cv2-search-new-chat');
        if (e.key === 'Escape') {
          e.preventDefault();
          this._closeSearchDialog();
        } else if (e.key === 'ArrowDown') {
          e.preventDefault();
          this._searchSelectedIdx = Math.min(this._searchSelectedIdx + 1, items.length - 1);
          this._highlightSearchResult(items);
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          this._searchSelectedIdx = Math.max(this._searchSelectedIdx - 1, -1);
          this._highlightSearchResult(items);
        } else if (e.key === 'Enter') {
          e.preventDefault();
          if (this._searchSelectedIdx >= 0 && items[this._searchSelectedIdx]) {
            items[this._searchSelectedIdx].click();
          }
        }
      });

      // Focus input
      requestAnimationFrame(() => input.focus());
    },

    _closeSearchDialog() {
      if (this._searchOverlay) {
        this._searchOverlay.remove();
        this._searchOverlay = null;
      }
      clearTimeout(this._searchDebounce);
      this._searchSelectedIdx = -1;
    },

    _highlightSearchResult(items) {
      items.forEach((el, i) => {
        el.classList.toggle('cv2-search-highlighted', i === this._searchSelectedIdx);
        if (i === this._searchSelectedIdx) {
          el.scrollIntoView({ block: 'nearest' });
        }
      });
    },

    async _performSearch(query, container) {
      const lowerQuery = query.toLowerCase();
      const regex = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');

      // Phase 1: instant title search
      const titleMatches = this.conversations.filter(c =>
        regex.test(c.title || '') || regex.test(c.first_user_snippet || '')
      );
      this._renderSearchResults(titleMatches.slice(0, _SEARCH_PAGE), query, container, false);

      // Show shimmer for content search
      const shimmerHtml = '<div class="cv2-search-shimmer"></div>'.repeat(3);
      const shimmerWrap = document.createElement('div');
      shimmerWrap.className = 'cv2-search-shimmer-wrap';
      shimmerWrap.innerHTML = shimmerHtml;
      container.appendChild(shimmerWrap);

      // Phase 2: deep content search (async, IDB)
      const titleMatchIds = new Set(titleMatches.map(c => c.id));
      const contentResults = [];

      for (const conv of this.conversations) {
        if (titleMatchIds.has(conv.id)) continue; // already shown
        try {
          const data = await this.idb.get(conv.id);
          if (!data || !data.messages) continue;
          let matchCount = 0;
          let firstSnippet = null;
          for (const msg of data.messages) {
            const text = msg.content || msg.text || '';
            if (regex.test(text)) {
              matchCount++;
              if (!firstSnippet) {
                const matchIdx = text.toLowerCase().indexOf(lowerQuery);
                const start = Math.max(0, matchIdx - 30);
                const end = Math.min(text.length, matchIdx + query.length + 30);
                firstSnippet = (start > 0 ? '...' : '') +
                  text.substring(start, end) +
                  (end < text.length ? '...' : '');
              }
            }
          }
          if (matchCount > 0) {
            contentResults.push({
              ...conv,
              _snippet: firstSnippet,
              _matchCount: matchCount,
              _matchType: 'content',
            });
          }
        } catch (_) {}
      }

      // Remove shimmer
      shimmerWrap.remove();

      // Re-render with combined results
      if (contentResults.length > 0) {
        const combined = [...titleMatches.map(c => ({ ...c, _matchType: 'title', _matchCount: 0 })), ...contentResults];
        this._renderSearchResults(combined.slice(0, _SEARCH_PAGE), query, container, false);
      }
    },

    _renderSearchResults(results, query, container, showNewChat) {
      this._searchSelectedIdx = -1;
      let html = '';

      // "New Chat" row at top
      if (showNewChat) {
        html += `<div class="cv2-search-new-chat">
          <span class="material-icons" style="font-size:16px">add</span>
          <span>${this._escHtml(this.t('chat.search_new_chat'))}</span>
        </div>`;
      }

      if (results.length === 0 && query) {
        html += `<div class="cv2-search-empty">${this._escHtml(this.t('chat.search_no_results'))}</div>`;
        container.innerHTML = html;
        return;
      }

      // Group by date
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const yesterday = new Date(today); yesterday.setDate(yesterday.getDate() - 1);
      const weekAgo = new Date(today); weekAgo.setDate(weekAgo.getDate() - 7);

      const groups = { today: [], yesterday: [], week: [], older: [] };
      const labels = {
        today: 'Heute',
        yesterday: 'Gestern',
        week: this.t('chat.search_last_week') || 'Letzte 7 Tage',
        older: this.t('chat.search_older') || 'Älter',
      };

      for (const c of results) {
        const d = c.created_at ? new Date(c.created_at) : null;
        if (!d) { groups.older.push(c); continue; }
        if (d >= today) groups.today.push(c);
        else if (d >= yesterday) groups.yesterday.push(c);
        else if (d >= weekAgo) groups.week.push(c);
        else groups.older.push(c);
      }

      for (const [key, items] of Object.entries(groups)) {
        if (items.length === 0) continue;
        html += `<div class="cv2-search-group-label">${labels[key]}</div>`;
        for (const c of items) {
          const title = this._escHtml(c.title || this.t('chat.untitled'));
          let snippetHtml = '';
          if (c._snippet && query) {
            const countBadge = (c._matchCount > 1)
              ? ` <span class="cv2-search-match-count">+${c._matchCount - 1} ${this.t('chat.search_more_matches')}</span>`
              : '';
            snippetHtml = `<div class="cv2-search-result-snippet">${this._highlightMatch(c._snippet, query)}${countBadge}</div>`;
          } else if (query) {
            snippetHtml = '';
          }
          const titleDisplay = query ? this._highlightMatch(c.title || this.t('chat.untitled'), query) : title;
          html += `<div class="cv2-search-result" data-id="${this._escAttr(c.id)}">
            <div class="cv2-search-result-title">${titleDisplay}</div>
            ${snippetHtml}
          </div>`;
        }
      }

      container.innerHTML = html;

      // Bind clicks
      container.querySelectorAll('.cv2-search-result').forEach(el => {
        el.addEventListener('click', () => {
          const overlay = this._searchOverlay;
          const q = overlay ? overlay.querySelector('.cv2-search-input').value.trim() : '';
          this._closeSearchDialog();
          if (q && el.dataset.id === this.activeConvId) {
            // Already viewing this conversation — open navigator directly
            this._openMatchNavigator(q);
          } else {
            if (q) this._pendingMatchQuery = q;
            this._selectConversation(el.dataset.id);
          }
        });
      });
      const newChatEl = container.querySelector('.cv2-search-new-chat');
      if (newChatEl) {
        newChatEl.addEventListener('click', () => {
          this.ws.send({ type: 'new_chat' });
          this._closeSearchDialog();
        });
      }
    },

    _highlightMatch(text, query) {
      if (!query || !text) return this._escHtml(text);
      const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(`(${escaped})`, 'gi');
      // Split, escape each part, wrap matches in <mark>
      const parts = text.split(regex);
      return parts.map(part =>
        regex.test(part) ? `<mark>${this._escHtml(part)}</mark>` : this._escHtml(part)
      ).join('');
    },

    // ── In-chat match navigator ───────────────────────────

    _openMatchNavigator(query) {
      this._clearMatchHighlights();
      const wrap = document.getElementById('cv2-messages-wrap');
      if (!wrap) return;

      const bar = document.createElement('div');
      bar.className = 'cv2-match-nav';
      bar.id = 'cv2-match-nav';
      bar.innerHTML = `
        <span class="material-icons" style="font-size:16px">search</span>
        <input class="cv2-match-nav-input" type="text" value="${this._escAttr(query)}">
        <span class="cv2-match-nav-count"></span>
        <button class="cv2-match-nav-prev" title="Previous"><span class="material-icons" style="font-size:16px">expand_less</span></button>
        <button class="cv2-match-nav-next" title="Next"><span class="material-icons" style="font-size:16px">expand_more</span></button>
        <button class="cv2-match-nav-close" title="Close"><span class="material-icons" style="font-size:16px">close</span></button>`;

      wrap.insertBefore(bar, wrap.firstChild);
      this._matchNavActive = true;

      const total = this._highlightMessagesFor(query);
      this._matchNavTotal = total;
      this._matchNavIdx = 0;

      this._updateMatchNavCounter();
      if (total > 0) this._scrollToMatch(0);

      // Events
      bar.querySelector('.cv2-match-nav-next').addEventListener('click', () => this._navigateMatch(1));
      bar.querySelector('.cv2-match-nav-prev').addEventListener('click', () => this._navigateMatch(-1));
      bar.querySelector('.cv2-match-nav-close').addEventListener('click', () => this._clearMatchHighlights());

      const navInput = bar.querySelector('.cv2-match-nav-input');
      let navDebounce;
      navInput.addEventListener('input', () => {
        clearTimeout(navDebounce);
        navDebounce = setTimeout(() => {
          const newQ = navInput.value.trim();
          // Remove old highlights but keep bar
          this._removeMatchMarks();
          if (!newQ) {
            this._matchNavTotal = 0;
            this._matchNavIdx = 0;
            this._updateMatchNavCounter();
            return;
          }
          const t = this._highlightMessagesFor(newQ);
          this._matchNavTotal = t;
          this._matchNavIdx = 0;
          this._updateMatchNavCounter();
          if (t > 0) this._scrollToMatch(0);
        }, 200);
      });

      navInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          e.preventDefault();
          this._clearMatchHighlights();
        } else if (e.key === 'Enter') {
          e.preventDefault();
          this._navigateMatch(e.shiftKey ? -1 : 1);
        }
      });

      requestAnimationFrame(() => navInput.select());
    },

    _highlightMessagesFor(query) {
      if (!query) return 0;
      const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(escaped, 'gi');
      const container = document.getElementById('cv2-messages');
      if (!container) return 0;

      const skip = new Set(['CODE', 'PRE', 'MARK', 'SCRIPT', 'STYLE']);
      let matchIdx = 0;

      const walk = (root) => {
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
          acceptNode(node) {
            let p = node.parentElement;
            while (p && p !== root) {
              if (skip.has(p.tagName)) return NodeFilter.FILTER_REJECT;
              p = p.parentElement;
            }
            return NodeFilter.FILTER_ACCEPT;
          }
        });

        const textNodes = [];
        while (walker.nextNode()) textNodes.push(walker.currentNode);

        for (const tn of textNodes) {
          const text = tn.nodeValue;
          if (!regex.test(text)) continue;
          regex.lastIndex = 0;

          const frag = document.createDocumentFragment();
          let lastIdx = 0;
          let m;
          while ((m = regex.exec(text)) !== null) {
            if (m.index > lastIdx) {
              frag.appendChild(document.createTextNode(text.substring(lastIdx, m.index)));
            }
            const mark = document.createElement('mark');
            mark.className = 'cv2-find-match';
            mark.dataset.matchIdx = matchIdx++;
            mark.textContent = m[0];
            frag.appendChild(mark);
            lastIdx = regex.lastIndex;
          }
          if (lastIdx < text.length) {
            frag.appendChild(document.createTextNode(text.substring(lastIdx)));
          }
          tn.parentNode.replaceChild(frag, tn);
        }
      };

      container.querySelectorAll('.cv2-msg-body, .cv2-msg-user-bubble').forEach(walk);
      return matchIdx;
    },

    _removeMatchMarks() {
      document.querySelectorAll('#cv2-messages .cv2-find-match').forEach(mark => {
        const parent = mark.parentNode;
        parent.replaceChild(document.createTextNode(mark.textContent), mark);
        parent.normalize();
      });
    },

    _clearMatchHighlights() {
      this._removeMatchMarks();
      const bar = document.getElementById('cv2-match-nav');
      if (bar) bar.remove();
      this._matchNavActive = false;
      this._matchNavIdx = 0;
      this._matchNavTotal = 0;
    },

    _navigateMatch(direction) {
      if (this._matchNavTotal === 0) return;
      this._matchNavIdx = (this._matchNavIdx + direction + this._matchNavTotal) % this._matchNavTotal;
      this._updateMatchNavCounter();
      this._scrollToMatch(this._matchNavIdx);
    },

    _scrollToMatch(idx) {
      const prev = document.querySelector('#cv2-messages .cv2-find-active');
      if (prev) prev.classList.remove('cv2-find-active');
      const target = document.querySelector(`#cv2-messages .cv2-find-match[data-match-idx="${idx}"]`);
      if (target) {
        target.classList.add('cv2-find-active');
        target.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    },

    _updateMatchNavCounter() {
      const counter = document.querySelector('#cv2-match-nav .cv2-match-nav-count');
      if (!counter) return;
      if (this._matchNavTotal === 0) {
        counter.textContent = this.t('chat.search_no_matches_in_chat') || '0 matches';
      } else {
        counter.textContent = `${this._matchNavIdx + 1} / ${this._matchNavTotal}`;
      }
    },

  });

  ChatFeatures.register('search', {
    initState(app) {
      app._searchOverlay = null;
      app._searchDebounce = null;
      app._searchSelectedIdx = -1;
      app._matchNavActive = false;
      app._matchNavIdx = 0;
      app._matchNavTotal = 0;
      app._pendingMatchQuery = null;
    },
    bindEvents(app) {
      document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault();
          if (app._searchOverlay) app._closeSearchDialog();
          else app._openSearchDialog();
        }
      });
      // Use capture phase so Escape reaches us even if something stops propagation
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && app._matchNavActive) {
          e.preventDefault();
          e.stopPropagation();
          app._clearMatchHighlights();
        }
      }, true);
    },
  });
})();
