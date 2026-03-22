/**
 * chat-followup.js — Interactive follow-up questions in chat.
 *
 * The AI outputs a fenced code block with language `followup` containing JSON.
 * This plugin renders it as an interactive form (cards, choice, inputs, sort).
 * Only interactive when it's the last assistant message; otherwise static.
 * On submit, auto-sends a formatted answer as a user message.
 *
 * Data format:
 * ```followup
 * {
 *   "title": "Optional header",
 *   "steps": [{
 *     "title": "Optional step title",
 *     "elements": [
 *       { "type": "cards", "label": "Pick one", "columns": 3,
 *         "options": [{"label":"A","value":"a","color":"#e74c3c","description":"desc"}] },
 *       { "type": "choice", "label": "Select", "id": "q1", "select": "single",
 *         "options": ["A","B","C"] },
 *       { "type": "text",   "id": "name", "label": "Name", "placeholder": "...",
 *         "default": "", "required": true },
 *       { "type": "number", "id": "age",  "label": "Age",  "min": 0, "max": 150,
 *         "default": 25, "hint": "Your age" },
 *       { "type": "sort",   "label": "Categorize these",
 *         "categories": [{"label":"Cat A","color":"#27ae60"},{"label":"Cat B","color":"#e67e22"}],
 *         "items": [{"label":"X","color":"#e74c3c"},{"label":"Y","color":"#3498db"}] }
 *     ]
 *   }],
 *   "submit_label": "Submit"
 * }
 * ```
 */
(function () {
  'use strict';

  /* ──────────────────────────────────────────────────────
   *  DOM helper
   * ────────────────────────────────────────────────────── */
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (k === 'className') el.className = v;
        else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
        else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
        else el.setAttribute(k, v);
      }
    }
    for (const c of children) {
      if (c == null) continue;
      el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return el;
  }

  /* ──────────────────────────────────────────────────────
   *  Contrast helper — pick white or black text for bg
   * ────────────────────────────────────────────────────── */
  function textColor(hex) {
    if (!hex || hex[0] !== '#') return '#fff';
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return (r * 0.299 + g * 0.587 + b * 0.114) > 160 ? '#1a1a2e' : '#ffffff';
  }

  /* ──────────────────────────────────────────────────────
   *  Check if this block is in the last assistant message
   * ────────────────────────────────────────────────────── */
  function isLastAssistantBlock(container) {
    const msgs = document.querySelector('.cv2-messages');
    if (!msgs) return false;
    const allMsgs = msgs.querySelectorAll('.cv2-msg-assistant, .cv2-msg-user');
    if (!allMsgs.length) return false;
    const last = allMsgs[allMsgs.length - 1];
    return last.classList.contains('cv2-msg-assistant') && last.contains(container);
  }

  /* ──────────────────────────────────────────────────────
   *  Render: cards element
   * ────────────────────────────────────────────────────── */
  /* Auto-detect column count based on content length */
  function autoColumns(options) {
    const maxLen = Math.max(...options.map(o => (o.label || '').length + (o.description ? o.description.length : 0)));
    const n = options.length;
    if (maxLen > 40 || n === 1) return 1;
    if (maxLen > 20 || n === 2) return 2;
    if (n <= 4) return Math.min(n, 2);
    return Math.min(n, 3);
  }

  function renderCards(el, answers, onAutoAdvance) {
    const cols = el.columns || autoColumns(el.options || []);
    const grid = h('div', { className: `cv2-fu-cards cv2-fu-cards-${cols}col` });
    const select = el.select || 'single';
    const key = el.id || el.label;

    (el.options || []).forEach((opt, i) => {
      const colorClass = opt.color ? '' : `cv2-fu-card-${i % 10}`;
      const card = h('div', { className: 'cv2-fu-card ' + colorClass });
      if (opt.color) {
        card.style.background = opt.color;
        card.style.color = textColor(opt.color);
      }
      card.appendChild(document.createTextNode(opt.label));
      if (opt.description) card.appendChild(h('div', { className: 'cv2-fu-card-desc' }, opt.description));

      card.addEventListener('click', () => {
        const val = opt.value || opt.label;
        if (select === 'single') {
          grid.querySelectorAll('.cv2-fu-card').forEach(c => c.classList.remove('selected'));
          card.classList.add('selected');
          answers.set(key, val);
          if (onAutoAdvance) onAutoAdvance();
        } else {
          card.classList.toggle('selected');
          const sel = [...grid.querySelectorAll('.cv2-fu-card.selected')].map(c => {
            const o = el.options.find(x => x.label === c.firstChild.textContent);
            return o ? (o.value || o.label) : c.firstChild.textContent;
          });
          answers.set(key, sel.join(', '));
        }
      });
      grid.appendChild(card);
    });
    return grid;
  }

  /* ──────────────────────────────────────────────────────
   *  Render: choice element (radio / checkbox)
   * ────────────────────────────────────────────────────── */
  function renderChoice(el, answers, onAutoAdvance) {
    const wrap = h('div', { className: 'cv2-fu-choices' });
    const select = el.select || 'single';
    const key = el.id || el.label;
    const multi = select === 'multiple';

    const options = (el.options || []).map(o => typeof o === 'string' ? { label: o, value: o } : o);

    for (const opt of options) {
      const row = h('div', { className: 'cv2-fu-choice' + (multi ? ' multi' : '') });
      row.appendChild(h('div', { className: 'cv2-fu-choice-indicator' }));
      row.appendChild(h('span', null, opt.label));

      row.addEventListener('click', () => {
        if (!multi) {
          wrap.querySelectorAll('.cv2-fu-choice').forEach(r => r.classList.remove('selected'));
          row.classList.add('selected');
          answers.set(key, opt.value || opt.label);
          if (onAutoAdvance) onAutoAdvance();
        } else {
          row.classList.toggle('selected');
          const sel = [...wrap.querySelectorAll('.cv2-fu-choice.selected')].map(r => r.querySelector('span').textContent);
          answers.set(key, sel.join(', '));
        }
      });
      wrap.appendChild(row);
    }
    return wrap;
  }

  /* ──────────────────────────────────────────────────────
   *  Render: text / number input
   * ────────────────────────────────────────────────────── */
  function renderInput(el, answers) {
    const key = el.id || el.label;
    const wrap = h('div', { className: 'cv2-fu-input-wrap' });
    const attrs = {
      className: 'cv2-fu-input',
      type: el.type === 'number' ? 'number' : 'text',
      placeholder: el.placeholder || '',
    };
    if (el.min != null) attrs.min = String(el.min);
    if (el.max != null) attrs.max = String(el.max);
    if (el.default != null) attrs.value = String(el.default);

    const input = h('input', attrs);
    if (el.default != null && String(el.default)) answers.set(key, String(el.default));

    input.addEventListener('input', () => {
      const v = input.value.trim();
      if (v) answers.set(key, v);
      else answers.delete(key);
    });
    wrap.appendChild(input);
    if (el.hint) wrap.appendChild(h('div', { className: 'cv2-fu-hint' }, el.hint));
    return wrap;
  }

  /* ──────────────────────────────────────────────────────
   *  Render: sort (categorize items)
   * ────────────────────────────────────────────────────── */
  function renderSort(el, answers) {
    const key = el.id || el.label;
    const wrap = h('div');
    const categories = el.categories || [];
    const items = (el.items || []).map((it, i) => ({ ...it, _id: i }));
    const catCols = Math.min(categories.length, 3);

    // State: which category each item is in (null = pool)
    const assignments = new Map(); // _id -> catIndex
    let selectedItem = null;

    const pool = h('div', { className: 'cv2-fu-sort-pool' });
    const catGrid = h('div', { className: 'cv2-fu-sort-categories', style: { gridTemplateColumns: `repeat(${catCols}, 1fr)` } });
    const catContainers = [];

    // Build category columns
    for (let ci = 0; ci < categories.length; ci++) {
      const cat = categories[ci];
      const col = h('div', { className: 'cv2-fu-sort-cat' });
      const header = h('div', { className: 'cv2-fu-sort-cat-header', style: { background: cat.color || '#6366f1', color: textColor(cat.color || '#6366f1') } }, cat.label);
      const itemsArea = h('div', { className: 'cv2-fu-sort-cat-items' });

      col.addEventListener('click', () => {
        if (selectedItem == null) return;
        assignments.set(selectedItem, ci);
        selectedItem = null;
        rebuild();
      });

      col.append(header, itemsArea);
      catContainers.push(itemsArea);
      catGrid.appendChild(col);
    }

    function makeChip(item, inPool) {
      const chip = h('div', {
        className: 'cv2-fu-sort-item' + (selectedItem === item._id ? ' dragging' : ''),
        style: { background: item.color || '#6366f1', color: textColor(item.color || '#6366f1') },
      }, item.label);

      chip.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!inPool && assignments.has(item._id)) {
          // Click item in category → move back to pool
          assignments.delete(item._id);
          selectedItem = null;
          rebuild();
        } else if (inPool) {
          // Click item in pool → select it
          selectedItem = selectedItem === item._id ? null : item._id;
          rebuild();
        }
      });
      return chip;
    }

    function rebuild() {
      pool.innerHTML = '';
      catContainers.forEach(c => c.innerHTML = '');

      // Highlight categories when item is selected
      catGrid.querySelectorAll('.cv2-fu-sort-cat').forEach(c => c.classList.toggle('highlight', selectedItem != null));

      for (const item of items) {
        const catIdx = assignments.get(item._id);
        if (catIdx != null) {
          catContainers[catIdx].appendChild(makeChip(item, false));
        } else {
          pool.appendChild(makeChip(item, true));
        }
      }

      // Update answers
      if (assignments.size === items.length) {
        const result = categories.map((cat, ci) => {
          const catItems = items.filter(it => assignments.get(it._id) === ci);
          return cat.label + ': ' + catItems.map(it => it.label).join(', ');
        }).join(' | ');
        answers.set(key, result);
      } else {
        answers.delete(key);
      }
    }

    rebuild();
    wrap.append(pool, catGrid);
    return wrap;
  }

  /* ──────────────────────────────────────────────────────
   *  Count required answers for a step
   * ────────────────────────────────────────────────────── */
  function requiredKeys(step) {
    const keys = [];
    for (const el of (step.elements || [])) {
      if (el.type === 'cards' && (el.select || 'single') === 'single') continue; // auto-advance
      if (el.type === 'choice' && (el.select || 'single') === 'single') continue; // auto-advance
      const key = el.id || el.label;
      if (key) keys.push(key);
    }
    return keys;
  }

  /* ──────────────────────────────────────────────────────
   *  Format answers for sending
   * ────────────────────────────────────────────────────── */
  function formatAnswers(spec, answers) {
    const lines = [];
    for (const step of spec.steps) {
      for (const el of (step.elements || [])) {
        const key = el.id || el.label;
        const val = answers.get(key);
        if (val != null) lines.push('Q: ' + el.label + '\nA: ' + val);
      }
    }
    return lines.join('\n\n');
  }

  /* ──────────────────────────────────────────────────────
   *  Main render function
   * ────────────────────────────────────────────────────── */
  function renderFollowup(container, rawData) {
    let spec;
    try { spec = JSON.parse(rawData); } catch { return; }
    if (!spec.steps || !spec.steps.length) return;

    const interactive = isLastAssistantBlock(container);

    // Outdated followups (already answered) — remove from DOM entirely
    if (!interactive) {
      container.remove();
      return;
    }

    const answers = new Map();
    let currentStep = 0;

    const root = h('div', { className: 'cv2-fu' });

    // Title
    if (spec.title) root.appendChild(h('div', { className: 'cv2-fu-header' }, spec.title));

    // Step dots (only if > 1 step)
    const steps = spec.steps;
    let dotsWrap = null;
    if (steps.length > 1) {
      dotsWrap = h('div', { className: 'cv2-fu-steps' });
      for (let i = 0; i < steps.length; i++) {
        dotsWrap.appendChild(h('div', { className: 'cv2-fu-step-dot' + (i === 0 ? ' active' : '') }));
      }
      root.appendChild(dotsWrap);
    }

    // Step containers
    const stepEls = [];
    for (let si = 0; si < steps.length; si++) {
      const step = steps[si];
      const stepEl = h('div', { className: 'cv2-fu-step' + (si === 0 ? ' active' : '') });
      if (step.title) stepEl.appendChild(h('div', { className: 'cv2-fu-label', style: { fontWeight: '600', marginBottom: '12px' } }, step.title));

      for (const el of (step.elements || [])) {
        const elWrap = h('div', { className: 'cv2-fu-el' });
        if (el.label) elWrap.appendChild(h('div', { className: 'cv2-fu-label' }, el.label));

        const isAutoAdvance = (el.type === 'cards' || el.type === 'choice') && (el.select || 'single') === 'single';
        const onAutoAdvance = isAutoAdvance ? () => {
          // Small delay so user sees selection
          setTimeout(() => {
            if (si < steps.length - 1) goStep(si + 1);
            else submit();
          }, 200);
        } : null;

        switch (el.type) {
          case 'cards':
            elWrap.appendChild(renderCards(el, answers, onAutoAdvance));
            break;
          case 'choice':
            elWrap.appendChild(renderChoice(el, answers, onAutoAdvance));
            break;
          case 'text':
          case 'number':
            elWrap.appendChild(renderInput(el, answers));
            break;
          case 'sort':
            elWrap.appendChild(renderSort(el, answers));
            break;
        }
        stepEl.appendChild(elWrap);
      }
      root.appendChild(stepEl);
      stepEls.push(stepEl);
    }

    // Footer with nav buttons
    const footer = h('div', { className: 'cv2-fu-footer' });
    const prevBtn = h('button', { className: 'cv2-fu-btn cv2-fu-btn-secondary', style: { display: 'none' }, onClick: () => goStep(currentStep - 1) }, 'Back');
    const nextBtn = h('button', { className: 'cv2-fu-btn cv2-fu-btn-primary', onClick: () => {
      if (currentStep < steps.length - 1) goStep(currentStep + 1);
      else submit();
    } }, steps.length > 1 ? 'Next' : (spec.submit_label || 'Submit'));
    footer.append(prevBtn, nextBtn);

    // Only show footer if there are non-auto-advance elements
    const hasManualElements = steps.some(step =>
      (step.elements || []).some(el => {
        if (el.type === 'text' || el.type === 'number' || el.type === 'sort') return true;
        if ((el.type === 'cards' || el.type === 'choice') && el.select === 'multiple') return true;
        return false;
      })
    );
    if (interactive && hasManualElements) root.appendChild(footer);


    function goStep(idx) {
      if (idx < 0 || idx >= steps.length) return;
      stepEls[currentStep].classList.remove('active');
      stepEls[idx].classList.add('active');
      currentStep = idx;
      // Update dots
      if (dotsWrap) {
        const dots = dotsWrap.querySelectorAll('.cv2-fu-step-dot');
        dots.forEach((d, i) => {
          d.className = 'cv2-fu-step-dot' + (i === idx ? ' active' : i < idx ? ' done' : '');
        });
      }
      // Update buttons
      prevBtn.style.display = idx > 0 ? '' : 'none';
      nextBtn.textContent = idx < steps.length - 1 ? 'Next' : (spec.submit_label || 'Submit');
      updateSubmitState();
    }

    function updateSubmitState() {
      if (currentStep === steps.length - 1) {
        // On last step — check if all required fields filled
        const allKeys = steps.flatMap(s => requiredKeys(s));
        const allFilled = allKeys.every(k => answers.has(k));
        nextBtn.disabled = !allFilled && allKeys.length > 0;
      } else {
        nextBtn.disabled = false;
      }
    }

    function submit() {
      const text = formatAnswers(spec, answers);
      if (!text) return;
      const chatApp = window.__chatApp;
      if (!chatApp) return;
      // Remove the form from DOM after submit
      root.remove();
      // Send the formatted answer
      chatApp.el.textarea.value = text;
      chatApp.sendMessage();
    }

    // Observe answers changes for submit button state
    const origSet = answers.set.bind(answers);
    const origDel = answers.delete.bind(answers);
    answers.set = (k, v) => { origSet(k, v); updateSubmitState(); };
    answers.delete = (k) => { origDel(k); updateSubmitState(); };

    updateSubmitState();
    container.innerHTML = '';
    container.appendChild(root);
  }

  /* ──────────────────────────────────────────────────────
   *  Register as a document plugin
   * ────────────────────────────────────────────────────── */
  // Wait for DocPluginRegistry to be available
  function tryRegister() {
    if (window.DocPluginRegistry?.prototype?.register || window._docPluginRegistryInstance) {
      const registry = window._docPluginRegistryInstance;
      if (registry) {
        registry.register('followup', {
          inline: true,
          sidebar: false,
          render: (container, rawData, _blockId) => renderFollowup(container, rawData),
        });
      }
    }
  }

  // Also expose for late registration from builtin-plugins.js
  window._followupPlugin = {
    inline: true,
    sidebar: false,
    render: (container, rawData, _blockId) => renderFollowup(container, rawData),
  };

  tryRegister();
})();
