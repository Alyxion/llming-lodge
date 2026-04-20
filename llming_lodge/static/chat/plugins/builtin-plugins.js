/**
 * Built-in document plugins for chat rendering.
 *
 * Registers: plotly, latex, table, text_doc, presentation, html
 */

/* ── Helper: mobile breakpoint check ─────────────────────── */
const _MOBILE_BP = 768;
function _isMobileDoc() { return window.innerWidth <= _MOBILE_BP; }

/* ── Helper: load a script tag and wait ───────────────────── */
function _loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === '1') { resolve(); return; }
      existing.addEventListener('load', resolve);
      existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)));
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.onload = () => { s.dataset.loaded = '1'; resolve(); };
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

/* ── Parse JSON safely ────────────────────────────────────── */
function _parseJSON(raw) {
  try { return JSON.parse(raw); }
  catch (firstErr) {
    // LLMs sometimes produce unescaped " inside HTML string values
    // (e.g. body_html with „text"</b>). Repair by escaping quotes
    // between HTML tags that break JSON parsing.
    try {
      // Strategy: walk the string, track JSON string boundaries,
      // and escape any unescaped " that doesn't look like a JSON delimiter.
      let fixed = '';
      let inString = false;
      let escaped = false;
      for (let i = 0; i < raw.length; i++) {
        const ch = raw[i];
        if (escaped) { fixed += ch; escaped = false; continue; }
        if (ch === '\\' && inString) { fixed += ch; escaped = true; continue; }
        if (ch === '"') {
          if (!inString) {
            inString = true; fixed += ch; continue;
          }
          // We're inside a string and hit ". Is this the real end?
          const after = raw.substring(i + 1).trimStart();
          if (after[0] === ':' || after[0] === ',' || after[0] === '}' ||
              after[0] === ']' || after[0] === undefined) {
            // Looks like a JSON delimiter — end the string
            inString = false; fixed += ch; continue;
          }
          // Not a JSON delimiter — this is an unescaped " inside a value
          fixed += '\\"'; continue;
        }
        fixed += ch;
      }
      return JSON.parse(fixed);
    } catch { return null; }
  }
}

function _escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

/* ── Attachment hover preview ─────────────────────────────── */
const _FILE_TYPE_ICONS = {
  'application/pdf': 'picture_as_pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'description',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'grid_on',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'slideshow',
  'text/csv': 'table_chart',
  'text/plain': 'article',
  'text/html': 'code',
  'application/zip': 'folder_zip',
  'application/json': 'data_object',
};
const _FILE_TYPE_LABELS = {
  'application/pdf': 'PDF Document',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Word Document',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'Excel Spreadsheet',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PowerPoint Presentation',
  'text/csv': 'CSV Data',
  'text/plain': 'Text File',
  'text/html': 'HTML File',
  'application/json': 'JSON Data',
  'image/png': 'PNG Image', 'image/jpeg': 'JPEG Image', 'image/gif': 'GIF Image', 'image/webp': 'WebP Image',
  'image/svg+xml': 'SVG Image',
};

/* ── Unified Attachment Preview System ───────────────────
   Hover → popover with content. Mouse into popover keeps it.
   Click → opens persistently. Maximize → full screen. Esc closes.
   ──────────────────────────────────────────────────────── */
let _pvEl = null, _pvBackdrop = null, _pvState = 'hidden';
let _pvHoverTimer = null, _pvDismissTimer = null, _pvAnchor = null;
let _pvResizeObserver = null;

function _fmtBytes(bytes) {
  if (!bytes) return '';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function _dismissPreview() {
  clearTimeout(_pvHoverTimer); clearTimeout(_pvDismissTimer);
  _pvHoverTimer = null; _pvDismissTimer = null;
  /* Clean up resize observer */
  if (_pvResizeObserver) { _pvResizeObserver.disconnect(); _pvResizeObserver = null; }
  /* Clean up drag listeners that may be attached to document */
  document.removeEventListener('mousemove', _pvOnDrag);
  document.removeEventListener('mouseup', _pvStopDrag);
  _pvDragOfs = null;
  _pvWindowedRect = null;
  if (_pvEl) { _pvEl.remove(); _pvEl = null; }
  if (_pvBackdrop) { _pvBackdrop.remove(); _pvBackdrop = null; }
  /* Safety: remove any orphaned backdrops */
  document.querySelectorAll('.cv2-preview-backdrop').forEach(b => b.remove());
  _pvState = 'hidden'; _pvAnchor = null;
}
function _scheduleDismiss() {
  if (_pvState === 'open' || _pvState === 'maximized') return;
  clearTimeout(_pvHoverTimer); clearTimeout(_pvDismissTimer);
  _pvDismissTimer = setTimeout(_dismissPreview, 250);
}
function _cancelPvDismiss() { clearTimeout(_pvDismissTimer); }

/* Esc key closes any preview */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && _pvState !== 'hidden') { _dismissPreview(); }
});

/* Saved windowed rect for restore */
let _pvWindowedRect = null;

function _maximizePreview() {
  if (!_pvEl) return;
  /* Toggle: if already maximized, restore to windowed */
  if (_pvState === 'maximized') { _restorePreview(); return; }
  /* Save windowed position/size for restore */
  if (_pvEl.classList.contains('cv2-preview-windowed')) {
    _pvWindowedRect = {
      width: _pvEl.style.width, height: _pvEl.style.height,
      left: _pvEl.style.left, top: _pvEl.style.top,
    };
  }
  _pvState = 'maximized';
  _pvEl.classList.remove('cv2-preview-windowed');
  _pvEl.classList.add('cv2-preview-maximized');
  _pvEl.style.width = ''; _pvEl.style.height = '';
  _pvEl.style.left = ''; _pvEl.style.top = ''; _pvEl.style.bottom = '';
  _pvEl.style.resize = '';
  /* Remove drag */
  const hdr = _pvEl.querySelector('.cv2-preview-header');
  if (hdr) { hdr.style.cursor = ''; hdr.removeEventListener('mousedown', _pvStartDrag); }
  /* Swap icon to restore */
  const maxBtn = _pvEl.querySelector('.cv2-preview-max .material-icons');
  if (maxBtn) maxBtn.textContent = 'close_fullscreen';
  /* Backdrop */
  _pvBackdrop = document.createElement('div');
  _pvBackdrop.className = 'cv2-preview-backdrop';
  _pvBackdrop.addEventListener('click', _dismissPreview);
  _pvEl.parentNode.insertBefore(_pvBackdrop, _pvEl);
  /* Re-render content at new size */
  const pd = _pvEl.querySelector('[data-plotly]');
  if (pd && window.Plotly) try { Plotly.relayout(pd, { autosize: true }); } catch (_) {}
  for (const iframe of _pvEl.querySelectorAll('iframe[src]')) {
    const src = iframe.src; iframe.src = ''; iframe.src = src;
  }
}

function _restorePreview() {
  if (!_pvEl) return;
  _pvState = 'open';
  _pvEl.classList.remove('cv2-preview-maximized');
  /* Remove backdrop */
  if (_pvBackdrop) { _pvBackdrop.remove(); _pvBackdrop = null; }
  document.querySelectorAll('.cv2-preview-backdrop').forEach(b => b.remove());
  /* Swap icon back */
  const maxBtn = _pvEl.querySelector('.cv2-preview-max .material-icons');
  if (maxBtn) maxBtn.textContent = 'open_in_full';
  /* Restore windowed position if we had one */
  if (_pvWindowedRect) {
    _pvEl.classList.add('cv2-preview-windowed');
    _pvEl.style.width = _pvWindowedRect.width;
    _pvEl.style.height = _pvWindowedRect.height;
    _pvEl.style.left = _pvWindowedRect.left;
    _pvEl.style.top = _pvWindowedRect.top;
    _pvEl.style.bottom = '';
    _pvEl.style.resize = 'both';
    const hdr = _pvEl.querySelector('.cv2-preview-header');
    if (hdr) { hdr.style.cursor = 'grab'; hdr.addEventListener('mousedown', _pvStartDrag); }
    _pvWindowedRect = null;
  }
  /* Re-layout at restored size */
  const pd = _pvEl.querySelector('[data-plotly]');
  if (pd && window.Plotly) try { Plotly.relayout(pd, { autosize: true }); } catch (_) {}
}

/* Hover to show */
function _showAttPreview(anchor, att, sessionId) {
  if (_pvState === 'open' || _pvState === 'maximized') return;
  clearTimeout(_pvDismissTimer); clearTimeout(_pvHoverTimer);
  if (_pvEl && _pvAnchor === anchor) return;
  if (_pvEl) { _pvEl.remove(); _pvEl = null; }
  _pvAnchor = anchor;
  _pvHoverTimer = setTimeout(() => _buildPreview(anchor, att, sessionId, 'hover'), 200);
}
/* Click to open persistently */
function _showAttPreviewPinned(anchor, att, sessionId) {
  _dismissPreview();
  _pvAnchor = anchor;
  _buildPreview(anchor, att, sessionId, 'open');
}

async function _buildPreview(anchor, att, sessionId, mode) {
  const el = document.createElement('div');
  el.className = 'cv2-preview-popover';
  _pvEl = el;
  _pvState = mode || 'hover';

  /* Header */
  const header = document.createElement('div');
  header.className = 'cv2-preview-header';
  header.innerHTML = `<span class="cv2-preview-title">${_escHtml(att.name || att.ref || 'Preview')}</span>`
    + '<button class="cv2-preview-btn cv2-preview-max" title="Maximize"><span class="material-icons">open_in_full</span></button>'
    + '<button class="cv2-preview-btn cv2-preview-close" title="Close"><span class="material-icons">close</span></button>';
  header.querySelector('.cv2-preview-max').addEventListener('click', e => { e.stopPropagation(); _maximizePreview(); });
  header.querySelector('.cv2-preview-close').addEventListener('click', e => { e.stopPropagation(); _dismissPreview(); });
  el.appendChild(header);

  /* Content area */
  const content = document.createElement('div');
  content.className = 'cv2-preview-content';
  content.innerHTML = '<div class="cv2-preview-loading"><span class="material-icons cv2-spin">sync</span></div>';
  el.appendChild(content);

  /* Mouse events — keep open when cursor enters popover */
  el.addEventListener('mouseenter', _cancelPvDismiss);
  el.addEventListener('mouseleave', () => { if (_pvState === 'hover') _scheduleDismiss(); });

  /* Position above or below anchor */
  const rect = anchor.getBoundingClientRect();
  const pvWidth = Math.min(Math.max(360, window.innerWidth * 0.3), 720);
  el.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - pvWidth - 8)) + 'px';
  if (rect.top > 280 || rect.top > window.innerHeight - rect.bottom) {
    el.style.bottom = (window.innerHeight - rect.top + 8) + 'px';
  } else {
    el.style.top = (rect.bottom + 8) + 'px';
  }
  (document.getElementById('chat-app') || document.body).appendChild(el);

  try { await _pvRenderContent(content, att, sessionId); }
  catch (e) {
    console.warn('[preview]', e);
    content.innerHTML = '<div class="cv2-preview-meta"><span class="material-icons" style="font-size:48px;color:#ef4444">error_outline</span><div style="color:var(--chat-text-muted)">Preview unavailable</div></div>';
  }
}

/* ── Content rendering by type ─────────────────────────── */

async function _pvRenderContent(container, att, sessionId) {
  const ct = att.content_type || att.mimeType || '';
  const isImage = ct.startsWith('image/');

  /* REF → BlockDataStore dynamic documents */
  if (att.type === 'ref' && att.ref) {
    const store = window.__chatApp?._blockDataStore;
    const entry = store?.get(att.ref);
    if (entry?.lang === 'plotly' && window.Plotly) {
      container.innerHTML = '';
      const plotDiv = document.createElement('div');
      plotDiv.style.cssText = 'width:100%;min-height:260px';
      plotDiv.dataset.plotly = '1';
      container.appendChild(plotDiv);
      const dark = _isDark();
      await Plotly.newPlot(plotDiv, (entry.data?.data || []), {
        ...(entry.data?.layout || {}), autosize: true,
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        margin: { l: 40, r: 16, t: 30, b: 40 },
        font: { color: dark ? '#ccc' : '#333', size: 11 },
      }, { displayModeBar: false, responsive: true });
      return;
    }
    if (entry?.lang === 'table') { _pvRenderTable(container, entry.data); return; }
    if (entry?.lang === 'rich_mcp') {
      container.innerHTML = '';
      const spec = entry.data;
      const render = spec?.render;
      try {
        if (render?.type === 'math_result') {
          const vendorScripts = await _resolveRichMcpVendorLibs(['katex_js', 'katex_css']);
          _mathResultIframe(container, render, vendorScripts);
        } else if (render?.type === 'html_sandbox') {
          const vendorLibs = render.vendor_libs || [];
          const vendorScripts = await _resolveRichMcpVendorLibs(vendorLibs);
          _richMcpSandboxIframe(container, render, vendorScripts, render.title || 'Visualization');
        }
      } catch (err) {
        container.innerHTML = `<div style="color:var(--chat-text-muted);padding:12px">${_escHtml(err.message)}</div>`;
      }
      return;
    }
    if (entry?.lang === 'html_sandbox' || entry?.lang === 'html') {
      let src;
      if (typeof entry.data === 'string') {
        const spec = _parseJSON(entry.data);
        if (spec && (spec.html || spec.css || spec.js)) {
          src = _buildHtmlSrcDoc(spec.title || '', spec.css || '', spec.html || '', spec.js || '');
        } else {
          src = _buildHtmlSrcDoc('', '', entry.data, '');
        }
      } else {
        src = _buildHtmlSrcDoc(entry.data?.title || '', entry.data?.css || '', entry.data?.html || '', entry.data?.js || '');
      }
      const iframe = document.createElement('iframe');
      iframe.sandbox = 'allow-scripts';
      iframe.style.cssText = 'width:100%;background:var(--chat-surface,#fff)';
      iframe.srcdoc = src;
      container.innerHTML = '';
      container.appendChild(iframe);
      return;
    }
    /* Use the plugin renderer for types not handled above */
    const lang = entry?.lang || att.lang || '';
    const chatApp = window.__chatApp;
    if (entry && lang && chatApp?.docPlugins?.has(lang)) {
      container.innerHTML = '';
      const pluginWrap = document.createElement('div');
      pluginWrap.className = 'cv2-doc-plugin-block';
      pluginWrap.style.padding = '12px';
      container.appendChild(pluginWrap);
      const rawData = typeof entry.data === 'string' ? entry.data : JSON.stringify(entry.data);
      await chatApp.docPlugins.render(lang, pluginWrap, rawData, 'pv-' + att.ref);
      return;
    }
    /* Fallback: clone inline content from DOM when store entry is
       missing (e.g. older version of an updated document). */
    const inlineBlock = document.querySelector(
      `.cv2-doc-plugin-block[data-block-id="${att.ref}"]`
    );
    if (inlineBlock) {
      container.innerHTML = '';
      const clone = document.createElement('div');
      clone.className = 'cv2-doc-plugin-block';
      clone.innerHTML = inlineBlock.innerHTML;
      container.appendChild(clone);
      return;
    }
    _pvRenderMeta(container, att.name || att.ref, (lang.charAt(0).toUpperCase() + lang.slice(1)) + ' Document', _pvDocIcon(lang));
    return;
  }

  /* CHAT_FILE → fetch structured preview from server */
  if (att.type === 'chat_file' && att.fileId && sessionId) {
    if (isImage) {
      container.innerHTML = `<img src="/api/llming/file-preview/${sessionId}/${att.fileId}">`;
      return;
    }
    try {
      const resp = await fetch(`/api/llming/file-content/${sessionId}/${att.fileId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      _pvRenderStructured(container, await resp.json(), sessionId, att.fileId);
    } catch (e) {
      container.innerHTML = `<iframe src="/api/llming/file-preview/${sessionId}/${att.fileId}" style="width:100%"></iframe>`;
    }
    return;
  }

  /* FILE with inline base64 */
  if (att.data) {
    if (isImage) {
      container.innerHTML = `<img src="data:${ct};base64,${att.data}">`;
      return;
    }
    if (ct === 'application/pdf') {
      const raw = atob(att.data), arr = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
      const url = URL.createObjectURL(new Blob([arr], { type: ct }));
      container.innerHTML = `<iframe src="${url}" style="width:100%"></iframe>`;
      return;
    }
  }

  /* Fallback: metadata card */
  const icon = _FILE_TYPE_ICONS[ct] || 'description';
  const label = _FILE_TYPE_LABELS[ct] || ct || 'File';
  _pvRenderMeta(container, att.name || 'Attachment', label + (_fmtBytes(att.size) ? ' — ' + _fmtBytes(att.size) : ''), icon);
}

function _pvRenderStructured(container, data) {
  if (data.type === 'image') {
    container.innerHTML = `<img src="${data.url}">`;
  } else if (data.type === 'pdf') {
    container.innerHTML = `<iframe src="${data.url}" style="width:100%"></iframe>`;
  } else if (data.type === 'table') {
    _pvRenderSheets(container, data.sheets || []);
  } else if (data.type === 'html') {
    const dark = _isDark();
    const iframe = document.createElement('iframe');
    iframe.style.cssText = 'width:100%;background:var(--chat-surface,#fff)';
    iframe.srcdoc = '<!doctype html><html><head><style>'
      + (dark
        ? 'body{font-family:system-ui,-apple-system,sans-serif;font-size:14px;line-height:1.65;padding:16px 20px;margin:0;color:#e5e7eb;background:#1e1e30}'
          + 'h1{font-size:20px;margin:18px 0 8px}h2{font-size:17px;margin:16px 0 6px}h3{font-size:15px;margin:14px 0 4px}'
          + 'p{margin:0 0 8px}table{border-collapse:collapse;margin:12px 0;width:100%}'
          + 'th,td{border:1px solid rgba(255,255,255,0.10);padding:6px 10px;text-align:left;font-size:13px}'
          + 'th{background:rgba(255,255,255,0.05);font-weight:600}ul,ol{margin:0 0 8px;padding-left:24px}'
        : 'body{font-family:system-ui,-apple-system,sans-serif;font-size:14px;line-height:1.65;padding:16px 20px;margin:0;color:#1f2937;background:#fff}'
          + 'h1{font-size:20px;margin:18px 0 8px}h2{font-size:17px;margin:16px 0 6px}h3{font-size:15px;margin:14px 0 4px}'
          + 'p{margin:0 0 8px}table{border-collapse:collapse;margin:12px 0;width:100%}'
          + 'th,td{border:1px solid #e5e7eb;padding:6px 10px;text-align:left;font-size:13px}'
          + 'th{background:#f9fafb;font-weight:600}ul,ol{margin:0 0 8px;padding-left:24px}')
      + '</style></head><body>' + data.html + '</body></html>';
    container.innerHTML = '';
    container.appendChild(iframe);
  } else if (data.type === 'text') {
    container.innerHTML = `<pre style="margin:0;padding:12px 16px;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-word;color:var(--chat-text);font-family:'SF Mono',Monaco,Consolas,monospace">${_escHtml(data.content)}</pre>`;
  } else {
    const icon = _FILE_TYPE_ICONS[data.content_type] || 'description';
    const label = _FILE_TYPE_LABELS[data.content_type] || data.content_type || 'File';
    _pvRenderMeta(container, data.name || 'File', label + (_fmtBytes(data.size) ? ' — ' + _fmtBytes(data.size) : ''), icon);
  }
}

function _pvRenderSheets(container, sheets) {
  container.innerHTML = '';
  if (!sheets.length) {
    container.innerHTML = '<div style="padding:24px;text-align:center;color:var(--chat-text-muted)">Empty spreadsheet</div>';
    return;
  }
  if (sheets.length > 1) {
    const tabs = document.createElement('div');
    tabs.className = 'cv2-preview-tabs';
    for (let i = 0; i < sheets.length; i++) {
      const tab = document.createElement('button');
      tab.className = 'cv2-preview-tab' + (i === 0 ? ' active' : '');
      tab.textContent = sheets[i].name || `Sheet ${i + 1}`;
      tab.addEventListener('click', () => {
        tabs.querySelectorAll('.cv2-preview-tab').forEach((t, j) => t.classList.toggle('active', j === i));
        showSheet(sheets[i]);
      });
      tabs.appendChild(tab);
    }
    container.appendChild(tabs);
  }
  const wrap = document.createElement('div');
  wrap.className = 'cv2-preview-table-wrap';
  container.appendChild(wrap);
  function showSheet(s) {
    let html = '<table><thead><tr>';
    for (const c of (s.columns || [])) html += `<th>${_escHtml(c)}</th>`;
    html += '</tr></thead><tbody>';
    for (const row of (s.rows || [])) {
      html += '<tr>';
      for (const cell of row) html += `<td>${_escHtml(cell)}</td>`;
      html += '</tr>';
    }
    html += '</tbody></table>';
    if (s.total_rows && s.total_rows > (s.rows || []).length + 1)
      html += `<div style="text-align:center;padding:8px;color:var(--chat-text-muted);font-size:11px">Showing ${(s.rows || []).length} of ${s.total_rows - 1} rows</div>`;
    wrap.innerHTML = html;
  }
  showSheet(sheets[0]);
}

function _pvRenderTable(container, spec) {
  const cols = spec.columns || [], rows = spec.rows || [];
  container.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'cv2-preview-table-wrap';
  let html = '<table><thead><tr>';
  for (const c of cols) html += `<th>${_escHtml(c.label || c.key || String(c))}</th>`;
  html += '</tr></thead><tbody>';
  for (const row of rows) {
    html += '<tr>';
    if (Array.isArray(row)) {
      for (const cell of row) html += `<td>${_escHtml(String(cell ?? ''))}</td>`;
    } else {
      for (const c of cols) html += `<td>${_escHtml(String(row[c.key || c] ?? ''))}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
  container.appendChild(wrap);
}

function _pvRenderMeta(container, name, detail, icon) {
  container.innerHTML = '<div class="cv2-preview-meta">'
    + `<span class="material-icons" style="font-size:48px;color:var(--chat-text-muted)">${icon || 'description'}</span>`
    + `<div><div style="font-weight:600;font-size:14px;color:var(--chat-text)">${_escHtml(name)}</div>`
    + `<div style="font-size:12px;color:var(--chat-text-muted)">${_escHtml(detail)}</div></div></div>`;
}

function _pvDocIcon(lang) {
  if (typeof ChatApp !== 'undefined' && ChatApp.DOC_ICONS?.[lang]) return ChatApp.DOC_ICONS[lang];
  return { plotly: 'show_chart', table: 'table_chart', text_doc: 'description', word: 'description', presentation: 'slideshow', powerpoint: 'slideshow', html: 'code', html_sandbox: 'code', rich_mcp: 'calculate' }[lang] || 'article';
}

/* Open a dynamic document in a floating, draggable window.
   Resolves blockId → parsed id via inlineDocBlocks, then uses
   the plugin renderer for full-fidelity content. */
function _showDocWindowed(blockId) {
  const chatApp = window.__chatApp;
  if (!chatApp) return;
  const docEntry = chatApp.inlineDocBlocks?.find(
    b => b.blockId === blockId || b.id === blockId
  );
  const storeKey = docEntry?.id || blockId;
  const store = chatApp._blockDataStore;
  const storeEntry = store?.get(storeKey) || store?.get(blockId);
  // Use the key that actually resolved in the store
  const resolvedRef = store?.has(storeKey) ? storeKey : blockId;
  /* Fall back to DOM element attributes when store/inlineDocBlocks
     don't have this block (e.g. older version of an updated doc). */
  const domBlock = !docEntry
    ? document.querySelector(`.cv2-doc-plugin-block[data-block-id="${blockId}"]`)
    : null;
  /* Try to derive a readable name: dataset → first heading → lang label. */
  const domTitle = domBlock?.querySelector('h1,h2,h3')?.textContent?.trim();
  const name = docEntry?.name || domBlock?.dataset?.name || domTitle || blockId;
  const lang = storeEntry?.lang || docEntry?.lang || domBlock?.dataset?.lang;
  const att = { type: 'ref', ref: resolvedRef, name, lang };
  _dismissPreview();
  _pvAnchor = null;
  /* Build preview in 'open' state, then size as a floating window */
  _buildPreview(document.body, att, null, 'open').then(() => {
    if (!_pvEl) return;
    _pvEl.classList.add('cv2-preview-windowed');
    _pvEl.dataset.blockId = blockId;
    /* Size: width-first, landscape-ish for plots/content */
    const vw = window.innerWidth, vh = window.innerHeight;
    const w = Math.min(Math.max(Math.round(vw * 0.5), 400), vw - 48);
    const h = Math.min(Math.round(w * 0.65), vh - 48);
    _pvEl.style.width = w + 'px';
    _pvEl.style.height = h + 'px';
    _pvEl.style.left = Math.round((vw - w) / 2) + 'px';
    _pvEl.style.top = Math.round((vh - h) / 2) + 'px';
    _pvEl.style.bottom = '';
    /* Make header draggable */
    const hdr = _pvEl.querySelector('.cv2-preview-header');
    if (hdr) {
      hdr.style.cursor = 'grab';
      hdr.addEventListener('mousedown', _pvStartDrag);
    }
    /* Re-layout plotly at new size */
    const pd = _pvEl.querySelector('[data-plotly]');
    if (pd && window.Plotly) try { Plotly.relayout(pd, { autosize: true }); } catch (_) {}
    /* Watch for user-driven resize (CSS resize: both) and re-layout plotly.
       Only observe if the preview actually contains a plotly chart. */
    if (_pvResizeObserver) _pvResizeObserver.disconnect();
    if (pd) {
      let _pvResizeTimer = null;
      _pvResizeObserver = new ResizeObserver(() => {
        if (!_pvEl) return;
        clearTimeout(_pvResizeTimer);
        _pvResizeTimer = setTimeout(() => {
          const p = _pvEl.querySelector('[data-plotly]');
          if (p && window.Plotly) try { Plotly.relayout(p, { autosize: true }); } catch (_) {}
        }, 100);
      });
      _pvResizeObserver.observe(_pvEl);
    }
  });
}
/* ── Drag logic for floating window ──────────────────── */
let _pvDragOfs = null;
function _pvStartDrag(e) {
  if (!_pvEl || e.button !== 0) return;
  /* Don't hijack clicks on buttons inside the header */
  if (e.target.closest('button')) return;
  e.preventDefault();
  _pvDragOfs = { x: e.clientX - _pvEl.offsetLeft, y: e.clientY - _pvEl.offsetTop };
  _pvEl.querySelector('.cv2-preview-header').style.cursor = 'grabbing';
  document.addEventListener('mousemove', _pvOnDrag);
  document.addEventListener('mouseup', _pvStopDrag);
}
function _pvOnDrag(e) {
  if (!_pvEl || !_pvDragOfs) return;
  _pvEl.style.left = Math.max(0, e.clientX - _pvDragOfs.x) + 'px';
  _pvEl.style.top = Math.max(0, e.clientY - _pvDragOfs.y) + 'px';
  _pvEl.style.bottom = '';
}
function _pvStopDrag() {
  if (_pvEl) {
    const hdr = _pvEl.querySelector('.cv2-preview-header');
    if (hdr) hdr.style.cursor = 'grab';
  }
  _pvDragOfs = null;
  document.removeEventListener('mousemove', _pvOnDrag);
  document.removeEventListener('mouseup', _pvStopDrag);
}

/* ── Theme detection ──────────────────────────────────────── */
function _isDark() {
  return !!document.getElementById('chat-app')?.classList.contains('cv2-dark');
}

function _themeColors() {
  const dark = _isDark();
  return {
    text: dark ? '#e5e7eb' : '#1f2937',
    textMuted: dark ? '#9ca3af' : '#6b7280',
    bg: dark ? 'rgba(0,0,0,0)' : 'rgba(0,0,0,0)',
    grid: dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
    paper: dark ? 'rgba(0,0,0,0)' : 'rgba(0,0,0,0)',
  };
}

/* ── Window button: opens the block in a floating preview ──── */
function _addWorkspaceButton(exportBar, blockId) {
  const btn = document.createElement('button');
  btn.className = 'cv2-doc-export-btn cv2-doc-window-btn';
  btn.title = 'Open in window';
  btn.innerHTML = '<span class="material-icons" style="font-size:14px;vertical-align:middle">picture_in_picture_alt</span>';
  exportBar.appendChild(btn);
  btn.addEventListener('click', () => _showDocWindowed(blockId));
}

/* ── Source viewer: adds a {…} button to an export bar ───── */
function _addSourceButton(exportBar, rawData) {
  const btn = document.createElement('button');
  btn.className = 'cv2-doc-export-btn cv2-doc-source-btn';
  btn.title = 'View source';
  btn.innerHTML = '<span class="material-icons" style="font-size:14px;vertical-align:middle">data_object</span>';
  exportBar.appendChild(btn);

  let sourceEl = null;
  btn.addEventListener('click', () => {
    if (sourceEl) {
      sourceEl.remove();
      sourceEl = null;
      btn.classList.remove('cv2-active');
      return;
    }
    btn.classList.add('cv2-active');
    const container = exportBar.closest('.cv2-doc-plugin-block') || exportBar.parentElement;
    sourceEl = document.createElement('div');
    sourceEl.className = 'cv2-doc-source-panel';

    const pre = document.createElement('pre');
    // Pretty-print if valid JSON, otherwise show raw
    const parsed = _parseJSON(rawData);
    pre.textContent = parsed ? JSON.stringify(parsed, null, 2) : rawData;

    const copyBtn = document.createElement('button');
    copyBtn.className = 'cv2-doc-source-copy';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(pre.textContent).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
      });
    });

    sourceEl.appendChild(copyBtn);
    sourceEl.appendChild(pre);
    container.appendChild(sourceEl);
  });
}

/* ══════════════════════════════════════════════════════════════
   Mermaid Plugin — lazy-loads on first use
   ══════════════════════════════════════════════════════════════ */
let _mermaidMod = null;
let _mermaidLoading = null;
let _mermaidCounter = 0;

function _getMermaid() {
  if (_mermaidMod) return Promise.resolve(_mermaidMod);
  if (_mermaidLoading) return _mermaidLoading;
  _mermaidLoading = import('/mermaid-esm/index.js').then(mod => {
    _mermaidMod = mod.mermaid;
    _mermaidMod.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });
    return _mermaidMod;
  });
  return _mermaidLoading;
}

function _sanitizeMermaid(code) {
  // Wrap node labels in quotes if they contain ( or ) — mermaid treats
  // bare parens inside [...] as shape modifiers which causes parse errors.
  return code.replace(/\[([^\]"]+)\]/g, (m, inner) =>
    (inner.indexOf('(') !== -1 || inner.indexOf(')') !== -1) ? '["' + inner + '"]' : m
  );
}

const mermaidPlugin = {
  inline: true,
  sidebar: false,
  render: async (container, rawData) => {
    container.innerHTML = '<div class="cv2-mermaid-loading">Rendering diagram\u2026</div>';
    try {
      const mm = await _getMermaid();
      const id = 'cv2mermaid-' + (++_mermaidCounter) + '-' + Date.now();
      const result = await mm.render(id, _sanitizeMermaid(rawData.trim()));
      container.innerHTML = '<div class="cv2-mermaid-container">' + result.svg + '</div>';
    } catch (err) {
      container.innerHTML = '<pre><code>' + _escHtml(rawData) + '</code></pre>';
      console.warn('[Mermaid] render error:', err);
    }
  },
};

/* ── Rich MCP — product number visualization (inline data, no server fetch) ── */

const _RICH_MCP_CSS = `
* { box-sizing: border-box; margin: 0; padding: 0; }
.pn-header { font-size: 14px; font-weight: 600; color: var(--chat-text, #e5e7eb); margin-bottom: 12px; }
.pn-norm { font-size: 11px; color: var(--chat-text-muted, #6b7280); font-weight: 400; margin-left: 8px; }
.pn-card { border: 1px solid var(--chat-border, rgba(255,255,255,0.10)); border-radius: 10px; padding: 14px; margin-bottom: 10px; }
.pn-code-strip { display: flex; gap: 2px; margin-bottom: 10px; flex-wrap: wrap; align-items: flex-start; }
.pn-seg { display: flex; flex-direction: column; align-items: center; }
.pn-seg-val { font-family: 'SF Mono', Monaco, Consolas, monospace; font-size: 20px; font-weight: 700;
  padding: 5px 9px; border-radius: 6px; letter-spacing: 0.5px; min-width: 32px; text-align: center; }
.pn-seg-label { font-size: 9px; color: var(--chat-text-muted, #6b7280); margin-top: 2px; text-transform: uppercase;
  letter-spacing: 0.05em; text-align: center; max-width: 70px; }
.pn-dot { font-size: 20px; color: var(--chat-text-muted, #6b7280); align-self: flex-start; padding-top: 5px; font-weight: 700; }
.pn-seg-0 .pn-seg-val { background: rgba(99,102,241,0.12); color: #6366f1; }
.pn-seg-1 .pn-seg-val { background: rgba(236,72,153,0.12); color: #ec4899; }
.pn-seg-2 .pn-seg-val { background: rgba(34,197,94,0.12); color: #16a34a; }
.pn-seg-3 .pn-seg-val { background: rgba(245,158,11,0.12); color: #d97706; }
.pn-seg-4 .pn-seg-val { background: rgba(139,92,246,0.12); color: #8b5cf6; }
.pn-seg-5 .pn-seg-val { background: rgba(99,102,241,0.08); color: var(--chat-text-muted, #6b7280); }
.pn-meta { display: flex; flex-direction: column; gap: 2px; margin-top: 8px; font-size: 11px; color: var(--chat-text-muted, #6b7280); }
.pn-meta-item { display: flex; align-items: center; gap: 4px; }
.pn-meta-label { font-weight: 600; color: var(--chat-text, #e5e7eb); }
.pn-ordering { font-family: 'SF Mono', Monaco, Consolas, monospace; font-size: 13px; color: var(--chat-text, #e5e7eb);
  background: rgba(99,102,241,0.06); padding: 6px 12px; border-radius: 6px; margin-top: 8px; display: inline-block; }
.pn-ordering-label { font-size: 10px; color: var(--chat-text-muted, #6b7280); text-transform: uppercase;
  letter-spacing: 0.05em; margin-top: 8px; margin-bottom: 2px; }
.pn-steps { margin-top: 10px; border-top: 1px solid var(--chat-border, rgba(255,255,255,0.10)); padding-top: 8px; }
.pn-step { font-size: 11px; color: var(--chat-text-muted, #6b7280); padding: 2px 0; }
.pn-step-label { font-weight: 600; color: var(--chat-text, #e5e7eb); }
`;

/** Parse formatted product number (XXX.XXX.XX.XX.XX.X) into display segments. */
function _segmentProductNumber(formatted) {
  const SEGS = [
    { len: 3, label: 'Series' }, { len: 3, label: 'Performance' },
    { len: 2, label: 'Material' }, { len: 2, label: 'Connection' },
    { len: 2, label: 'Component' }, { len: 1, label: 'Index' },
  ];
  const chars = (formatted || '').replace(/\./g, '').split('');
  const out = [];
  let pos = 0;
  for (const seg of SEGS) {
    const val = chars.slice(pos, pos + seg.len).join('');
    if (val) out.push({ value: val, label: seg.label });
    pos += seg.len;
  }
  return out;
}

/** Build product number HTML from structured data and render in a sandboxed iframe. */
function _renderProductNumber(container, spec) {
  const e = (s) => String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const segments = _segmentProductNumber(spec.formatted);
  const info = spec.info || {};
  const steps = spec.steps || [];
  const ordering = spec.ordering || '';
  const title = spec.title || 'Product Number';
  const norm = spec.norm || '';

  let body = `<div class="pn-header">${e(title)}<span class="pn-norm">${e(norm)}</span></div>`;
  body += '<div class="pn-card">';

  // Code strip
  body += '<div class="pn-code-strip">';
  segments.forEach((seg, i) => {
    if (i > 0) body += '<span class="pn-dot">.</span>';
    body += `<div class="pn-seg pn-seg-${Math.min(i, 5)}">`;
    body += `<span class="pn-seg-val">${e(seg.value)}</span>`;
    body += `<span class="pn-seg-label">${e(seg.label)}</span>`;
    body += '</div>';
  });
  body += '</div>';

  // Info metadata — only show if no resolution steps (steps have more detail)
  const infoKeys = Object.keys(info);
  if (infoKeys.length > 0 && steps.length === 0) {
    body += '<div class="pn-meta">';
    infoKeys.forEach(k => {
      const v = info[k];
      if (v && v !== 'N/A') {
        const label = k.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());
        body += `<span class="pn-meta-item"><span class="pn-meta-label">${e(label)}:</span> ${e(v)}</span>`;
      }
    });
    body += '</div>';
  }

  // Ordering number
  if (ordering) {
    body += '<div class="pn-ordering-label">Ordering Number</div>';
    body += `<div class="pn-ordering">${e(ordering)}</div>`;
  }

  // Resolution steps
  if (steps.length > 0) {
    body += '<div class="pn-steps">';
    steps.forEach(s => {
      const note = s.note || '';
      const from = s.from != null ? ` (from: ${e(s.from)})` : '';
      body += `<div class="pn-step"><span class="pn-step-label">${e(s.step)}:</span> ${e(s.resolved || s.resolvedRate || '')}${from}${note ? ' \u2014 ' + e(note) : ''}</div>`;
    });
    body += '</div>';
  }
  body += '</div>';

  // Inject theme CSS vars from parent
  const themeEl = document.getElementById('chat-app') || document.documentElement;
  const root = getComputedStyle(themeEl);
  const varNames = ['--chat-accent','--chat-bg','--chat-surface','--chat-text',
    '--chat-border','--chat-text-muted','--chat-code-bg','--chat-code-text','--chat-link'];
  const themeVars = varNames.map(v => { const val = root.getPropertyValue(v).trim(); return val ? `${v}:${val}` : ''; }).filter(Boolean).join(';');
  const themeCSS = themeVars ? `:root{${themeVars}}` : '';

  const resizeJS = `function _rh(){var h=Math.ceil(document.body.getBoundingClientRect().height);window.parent.postMessage({type:'resize',height:h},'*');}new ResizeObserver(_rh).observe(document.body);_rh();`;
  const themeUpdateJS = `window.addEventListener('message',function(e){if(e.data&&e.data.type==='theme_update'){var r=document.documentElement;for(var k in e.data.vars||{})r.style.setProperty(k,e.data.vars[k])}});`;

  const srcDoc = `<!DOCTYPE html><html><head><meta charset="utf-8"><style>${themeCSS}</style><style>html,body{font-family:system-ui,-apple-system,sans-serif;background:transparent;margin:0;padding:16px;overflow:hidden}</style><style>${_RICH_MCP_CSS}</style></head><body>${body}<script>${themeUpdateJS}${resizeJS}<\/script></body></html>`;

  const iframe = document.createElement('iframe');
  iframe.sandbox = 'allow-scripts';
  iframe.srcdoc = srcDoc;
  iframe.style.cssText = 'width:100%;border:none;border-radius:8px;overflow:hidden;min-height:60px;';
  iframe.title = title;
  window.addEventListener('message', function onMsg(ev) {
    if (ev.source === iframe.contentWindow && ev.data?.type === 'resize' && typeof ev.data.height === 'number') {
      iframe.style.height = Math.min(ev.data.height + 2, 800) + 'px';
    }
  });
  container.appendChild(iframe);
}

/** Vendor lib cache + resolver for rich_mcp sandbox iframes. */
const _richMcpVendorCache = {};
const _RICH_MCP_VENDOR_LIBS = {
  plotly: '/chat-static/vendor/plotly.min.js',
  katex_js: '/chat-static/vendor/katex.min.js',
  katex_css: '/chat-static/vendor/katex.min.css',
};

/** Download a Plotly chart as PNG in light mode at 1024px width. */
async function _downloadPlotlyPng(plotlySpec, title) {
  if (!window.Plotly) return;
  // Create an offscreen div for rendering
  const offscreen = document.createElement('div');
  offscreen.style.cssText = 'position:fixed;left:-9999px;top:0;width:1024px;height:640px;';
  document.body.appendChild(offscreen);
  try {
    const lightText = '#374151';
    const lightGrid = 'rgba(0,0,0,0.08)';
    const layout = { ...(plotlySpec.layout || {}), autosize: true, width: 1024, height: 640,
      paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
      margin: { l: 60, r: 30, t: 50, b: 60, ...(plotlySpec.layout?.margin || {}) },
      font: { color: lightText, family: 'system-ui,-apple-system,sans-serif', size: 13 },
    };
    if (layout.title) layout.title = { ...layout.title, font: { ...(layout.title.font || {}), color: '#111827' } };
    ['xaxis', 'yaxis'].forEach(a => { if (layout[a]) layout[a] = { ...layout[a], color: lightText, gridcolor: lightGrid }; });
    if (layout.scene) ['xaxis','yaxis','zaxis'].forEach(a => { if (layout.scene[a]) layout.scene[a] = { ...layout.scene[a], color: lightText, gridcolor: lightGrid }; });
    await Plotly.newPlot(offscreen, plotlySpec.data || [], layout, { displayModeBar: false });
    const dataUrl = await Plotly.toImage(offscreen, { format: 'png', width: 1024, height: 640, scale: 2 });
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = (title || 'chart').replace(/[^a-zA-Z0-9_\-() ]/g, '_') + '.png';
    a.click();
  } finally {
    Plotly.purge(offscreen);
    offscreen.remove();
  }
}

async function _resolveRichMcpVendorLibs(libKeys) {
  const results = {};
  await Promise.all(libKeys.map(async (key) => {
    if (_richMcpVendorCache[key]) { results[key] = _richMcpVendorCache[key]; return; }
    const path = _RICH_MCP_VENDOR_LIBS[key];
    if (!path) return;
    try {
      const resp = await fetch(path);
      if (!resp.ok) { console.warn('[RICH_MCP] Vendor lib 404:', key, path); return; }
      const text = await resp.text();
      _richMcpVendorCache[key] = text;
      results[key] = text;
    } catch (err) { console.warn('[RICH_MCP] Vendor lib fetch failed:', key, err); }
  }));
  return results;
}

/** Build and render a math_result from data-only envelope. All HTML/CSS/JS is generated here. */
function _mathResultIframe(container, render, vendorScripts) {
  const title = render.title || '';
  const latex = render.latex || '';
  const resultText = render.result_text || '';
  const steps = render.steps || [];
  const extraInfo = render.extra_info || {};
  const hasCard = !!(latex || resultText || steps.length || Object.keys(extraInfo).length);
  // Detect dark mode from parent page (cv2-dark class), not OS prefers-color-scheme
  const isDark = !!document.querySelector('#chat-app.cv2-dark');

  // --- Build HTML ---
  let html = '<div id="math-result"></div>';

  // --- Build CSS (always from scratch — never stored) ---
  const css = `
* { box-sizing: border-box; margin: 0; padding: 0; }
body { padding: 12px 16px; font-family: system-ui, -apple-system, sans-serif; color: ${isDark ? '#e5e7eb' : '#1f2937'}; }
.math-header { font-size: 14px; font-weight: 600; margin-bottom: 12px; }
.math-card { border: 1px solid rgba(128,128,128,0.2); border-radius: 10px; padding: 16px; background: rgba(128,128,128,0.05); }
.math-latex { font-size: 16px; margin: 8px 0; padding: 12px; background: rgba(99,102,241,0.06); border-radius: 8px; text-align: center; overflow-x: auto; }
.math-result-text { font-family: 'SF Mono', Monaco, Consolas, monospace; font-size: 13px; padding: 8px 12px; background: rgba(34,197,94,0.08); border-radius: 6px; margin: 8px 0; white-space: pre-wrap; word-break: break-word; }
.math-steps { margin-top: 12px; border-top: 1px solid rgba(128,128,128,0.2); padding-top: 10px; }
.math-steps-title { font-size: 12px; font-weight: 600; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.math-step { display: flex; gap: 10px; margin-bottom: 8px; }
.math-step-num { flex-shrink: 0; width: 22px; height: 22px; border-radius: 50%; background: rgba(99,102,241,0.15); color: #818cf8; font-size: 11px; font-weight: 700; display: flex; align-items: center; justify-content: center; margin-top: 2px; }
.math-step-content { flex: 1; }
.math-step-title { font-size: 12px; font-weight: 600; }
.math-step-expr { font-size: 14px; margin: 4px 0; overflow-x: auto; }
.math-step-note { font-size: 11px; opacity: 0.6; }
.math-info { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; font-size: 11px; opacity: 0.6; border-top: 1px solid rgba(128,128,128,0.2); padding-top: 8px; }
.math-info-item { display: flex; align-items: center; gap: 4px; }
.math-info-label { font-weight: 600; }
`;

  // --- Build JS from data ---
  const esc = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  let js = 'var c=document.getElementById("math-result"),h="";';
  if (hasCard) {
    js += `h+='<div class="math-header">${esc(title)}</div>';`;
    js += `h+='<div class="math-card">';`;
    if (latex) js += `h+='<div class="math-latex">$$${esc(latex)}$$</div>';`;
    if (resultText) js += `h+='<div class="math-result-text">${esc(resultText)}</div>';`;
    if (steps.length) {
      js += `h+='<div class="math-steps"><div class="math-steps-title">Step-by-step solution</div>';`;
      steps.forEach((s, i) => {
        js += `h+='<div class="math-step"><span class="math-step-num">${i+1}</span><div class="math-step-content">`;
        js += `<div class="math-step-title">${esc(s.title || 'Step '+(i+1))}</div>`;
        if (s.latex) js += `<div class="math-step-expr">$$${esc(s.latex)}$$</div>`;
        if (s.note) js += `<div class="math-step-note">${esc(s.note)}</div>`;
        js += `</div></div>';`;
      });
      js += `h+='</div>';`;
    }
    const infoKeys = Object.keys(extraInfo);
    if (infoKeys.length) {
      js += `h+='<div class="math-info">`;
      infoKeys.forEach(k => { js += `<span class="math-info-item"><span class="math-info-label">${esc(k)}:</span> ${esc(extraInfo[k])}</span>`; });
      js += `</div>';`;
    }
    js += `h+='</div>';`;
  }
  js += 'c.innerHTML=h;';

  // KaTeX auto-render
  if (vendorScripts.katex_js) {
    js += `
document.querySelectorAll('.math-latex,.math-step-expr').forEach(function(el){
  var t=el.innerHTML;
  t=t.replace(/\\$\\$([^$]+)\\$\\$/g,function(_,e){try{return katex.renderToString(e.trim(),{displayMode:true,throwOnError:false})}catch(x){return e}});
  t=t.replace(/\\$([^$]+)\\$/g,function(_,e){try{return katex.renderToString(e.trim(),{displayMode:false,throwOnError:false})}catch(x){return e}});
  el.innerHTML=t;
});`;
  }

  // Resize observer
  js += `function _rh(){var h=Math.ceil(document.body.getBoundingClientRect().height);window.parent.postMessage({type:'resize',height:h},'*')}new ResizeObserver(_rh).observe(document.body);_rh();`;

  // Vendor block (KaTeX only — plots go through document plugin now)
  let vendorBlock = '';
  if (vendorScripts.katex_css) vendorBlock += `<style>${vendorScripts.katex_css}</style>`;
  if (vendorScripts.katex_js) vendorBlock += `<script>${vendorScripts.katex_js}<\/script>`;

  const srcDoc = `<!DOCTYPE html><html><head><meta charset="utf-8">${vendorBlock}<style>${css}</style></head><body>${html}<script>${js}<\/script></body></html>`;
  const iframe = document.createElement('iframe');
  iframe.sandbox = 'allow-scripts';
  iframe.srcdoc = srcDoc;
  iframe.style.cssText = 'width:100%;border:none;border-radius:8px;overflow:hidden;min-height:60px;';
  iframe.title = title;
  window.addEventListener('message', function onMsg(e) {
    if (e.source === iframe.contentWindow && e.data?.type === 'resize' && typeof e.data.height === 'number') {
      if (iframe.dataset.fillParent) return; // Skip auto-resize in windowed mode
      iframe.style.height = Math.min(e.data.height + 2, 800) + 'px';
    }
  });
  container.appendChild(iframe);
}

/** Render html_sandbox rich_mcp in a sandboxed iframe with vendor libs. */
function _richMcpSandboxIframe(container, render, vendorScripts, title) {
  const htmlContent = render.html || '';
  const cssContent = render.css || '';
  const jsContent = render.js || '';

  // Inject vendor scripts
  let vendorBlock = '';
  if (vendorScripts.plotly) vendorBlock += `<script>${vendorScripts.plotly}<\/script>`;
  if (vendorScripts.katex_css) vendorBlock += `<style>${vendorScripts.katex_css}</style>`;
  if (vendorScripts.katex_js) vendorBlock += `<script>${vendorScripts.katex_js}<\/script>`;

  // KaTeX auto-render: process $$...$$ and $...$ after DOM loads
  const katexAutoRender = vendorScripts.katex_js ? `
    function _autoRenderKatex(){
      document.querySelectorAll('.math-latex,.math-step-expr').forEach(function(el){
        var html=el.innerHTML;
        html=html.replace(/\\$\\$([^$]+)\\$\\$/g,function(_,expr){
          try{return katex.renderToString(expr.trim(),{displayMode:true,throwOnError:false});}catch(e){return expr;}
        });
        html=html.replace(/\\$([^$]+)\\$/g,function(_,expr){
          try{return katex.renderToString(expr.trim(),{displayMode:false,throwOnError:false});}catch(e){return expr;}
        });
        el.innerHTML=html;
      });
    }
  ` : 'function _autoRenderKatex(){}';

  const resizeJS = `function _rh(){var h=Math.ceil(document.body.getBoundingClientRect().height);window.parent.postMessage({type:'resize',height:h},'*')}new ResizeObserver(_rh).observe(document.body);_rh();`;

  const srcDoc = `<!DOCTYPE html><html><head><meta charset="utf-8">${vendorBlock}<style>html,body{font-family:system-ui,-apple-system,sans-serif;color:var(--chat-text,#1f2937);background:transparent;margin:0;padding:0;}</style><style>${cssContent}</style></head><body>${htmlContent}<script>${katexAutoRender}${jsContent};_autoRenderKatex();${resizeJS}<\/script></body></html>`;

  const iframe = document.createElement('iframe');
  iframe.sandbox = 'allow-scripts';
  iframe.srcdoc = srcDoc;
  iframe.style.cssText = 'width:100%;border:none;border-radius:8px;overflow:hidden;min-height:60px;';
  iframe.title = title;
  window.addEventListener('message', function onMsg(e) {
    if (e.source === iframe.contentWindow && e.data?.type === 'resize' && typeof e.data.height === 'number') {
      if (iframe.dataset.fillParent) return;
      iframe.style.height = Math.min(e.data.height + 2, 800) + 'px';
    }
  });
  container.appendChild(iframe);
}

/** Render a rich_mcp block as a compact inline card with click-to-expand. */
async function _richMcpRenderCard(container, spec, rawData, blockId) {
  const renderType = spec.render?.type;
  const title = spec.render?.title || spec.title || 'Visualization';

  // Register in blockDataStore so windowed preview can access it
  const chatApp = window.__chatApp;
  const docId = spec.id || blockId;
  if (chatApp?._blockDataStore) {
    chatApp._blockDataStore.register(docId, 'rich_mcp', spec);
  }

  container.classList.add('cv2-rich-mcp-container');
  container.dataset.name = title;

  // Compact inline preview with iframe
  const preview = document.createElement('div');
  preview.className = 'cv2-rich-mcp-preview';
  container.appendChild(preview);

  // Render the iframe into the preview area
  try {
    if (renderType === 'math_result') {
      const vendorScripts = await _resolveRichMcpVendorLibs(['katex_js', 'katex_css']);
      _mathResultIframe(preview, spec.render, vendorScripts);
    } else if (renderType === 'html_sandbox') {
      const vendorLibs = spec.render.vendor_libs || [];
      const vendorScripts = await _resolveRichMcpVendorLibs(vendorLibs);
      _richMcpSandboxIframe(preview, spec.render, vendorScripts, title);
    }
  } catch (err) {
    console.warn('[RICH_MCP] Inline render failed:', err);
    preview.innerHTML = `<div style="color:var(--chat-text-muted,#6b7280);padding:8px;font-size:12px">${_escHtml(err.message)}</div>`;
  }

  // Maximize overlay (bottom-right)
  const hint = document.createElement('div');
  hint.className = 'cv2-rich-mcp-expand-hint';
  hint.innerHTML = '<span class="material-icons">open_in_full</span>';
  container.appendChild(hint);

  // Toolbar
  const toolbar = document.createElement('div');
  toolbar.className = 'cv2-doc-plugin-export';
  // Download PNG button (plotly charts only — legacy conversations with embedded plotly data)
  if (spec.render?.plotly) {
    const dlBtn = document.createElement('button');
    dlBtn.className = 'cv2-doc-export-btn';
    dlBtn.title = 'Download PNG';
    dlBtn.innerHTML = '<span class="material-icons" style="font-size:14px;vertical-align:middle">download</span>';
    dlBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      _downloadPlotlyPng(spec.render.plotly, title);
    });
    toolbar.appendChild(dlBtn);
  }
  _addSourceButton(toolbar, rawData);
  container.appendChild(toolbar);

  // Click handlers
  function openWindowed() { _showDocWindowed(blockId); }
  preview.style.cursor = 'pointer';
  preview.addEventListener('click', openWindowed);
  hint.addEventListener('click', openWindowed);
}

const richMcpPlugin = {
  inline: true,
  sidebar: false,
  render: async (container, rawData, blockId) => {
    const spec = _parseJSON(rawData);
    if (!spec) {
      container.innerHTML = '<div style="color:#888;padding:8px">Invalid rich_mcp block</div>';
      return;
    }

    // v2: inline structured data — render directly (no windowing for small cards)
    if (spec.type === 'product_number' && spec.formatted) {
      _renderProductNumber(container, spec);
      return;
    }

    // v4: math_result — compact card with click-to-window
    if (spec.render && spec.render.type === 'math_result') {
      await _richMcpRenderCard(container, spec, rawData, blockId);
      return;
    }

    // v3: html_sandbox — compact card with click-to-window
    if (spec.render && spec.render.type === 'html_sandbox') {
      await _richMcpRenderCard(container, spec, rawData, blockId);
      return;
    }

    // Legacy v1: UUID-based server fetch (backward compat for old conversations)
    if (spec.uuid) {
      const title = spec.title || 'Visualization';
      container.innerHTML = `<div class="cv2-doc-plugin-streaming"><div class="cv2-spinner-dots"><span></span><span></span><span></span></div><span class="cv2-doc-plugin-streaming-name">${_escHtml(title)}</span></div>`;
      const url = `/api/llming/rich-render/${encodeURIComponent(spec.uuid)}`;
      try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const renderSpec = await resp.json();
        container.innerHTML = '';
        _richMcpLegacyIframe(container, renderSpec, title);
      } catch (err) {
        console.warn('[RICH_MCP] Legacy fetch failed:', err);
        container.innerHTML = `<div style="color:var(--chat-text-muted,#6b7280);padding:8px;font-size:12px;opacity:0.7">Visualization no longer available</div>`;
      }
      return;
    }

    container.innerHTML = '<div style="color:#888;padding:8px">Unknown rich_mcp format</div>';
  },
};

/** Legacy iframe renderer for old v1 conversations that stored HTML/CSS/JS on server. */
function _richMcpLegacyIframe(container, renderSpec, title) {
  const htmlContent = renderSpec.html || '';
  const cssContent = renderSpec.css || '';
  const jsContent = renderSpec.js || '';
  const resizeJS = `function _rh(){var h=Math.ceil(document.body.getBoundingClientRect().height);window.parent.postMessage({type:'resize',height:h},'*')}new ResizeObserver(_rh).observe(document.body);_rh();`;
  const srcDoc = `<!DOCTYPE html><html><head><meta charset="utf-8"><style>html,body{font-family:system-ui,-apple-system,sans-serif;color:#e5e7eb;background:transparent;margin:0;padding:0}</style><style>${cssContent}</style></head><body>${htmlContent}<script>${jsContent}${resizeJS}<\/script></body></html>`;
  const iframe = document.createElement('iframe');
  iframe.sandbox = 'allow-scripts';
  iframe.srcdoc = srcDoc;
  iframe.style.cssText = 'width:100%;border:none;border-radius:8px;overflow:hidden;min-height:60px;';
  iframe.title = title;
  window.addEventListener('message', function onMsg(e) {
    if (e.source === iframe.contentWindow && e.data?.type === 'resize' && typeof e.data.height === 'number') {
      iframe.style.height = Math.min(e.data.height + 2, 800) + 'px';
    }
  });
  container.appendChild(iframe);
}


/* ══════════════════════════════════════════════════════════════
   Kantini Likes — inline meal reaction system
   ══════════════════════════════════════════════════════════════ */

const _kantiniLikes = {
  // Cache: meal_id → {total, emojis: {emoji: count}, likers: [{name, avatar, emoji}]}
  _cache: {},
  _userEmail: '',
  _emojis: ['👍', '😋', '🔥', '❤️', '🤮'],

  init() {
    // Listen for server responses
    document.addEventListener('kantini-likes', (e) => {
      const d = e.detail;
      if (d.likes_batch) {
        Object.assign(this._cache, d.likes_batch);
        this.renderAll();
      } else if (d.meal_id && d.likes) {
        this._cache[d.meal_id] = d.likes;
        this.renderBar(d.meal_id);
      }
    });
    // Get user email from config
    const app = window.__chatApp;
    if (app && app.config) this._userEmail = app.config.userEmail || '';
  },

  /** Ensure user email is set (lazy — config may not be ready at init time). */
  _ensureEmail() {
    if (!this._userEmail) {
      const app = window.__chatApp;
      if (app && app.config) this._userEmail = app.config.userEmail || '';
    }
  },

  /** Send a WS message to the server. */
  _send(payload) {
    const app = window.__chatApp;
    if (app && app.ws) app.ws.send({ type: 'kantini_action', ...payload });
  },

  /** Fetch likes for a set of meal IDs. */
  fetchLikes(mealIds) {
    if (!mealIds.length) return;
    this._send({ action: 'get_likes', meal_ids: mealIds });
  },

  /** Toggle a like on/off. */
  toggle(mealId, emoji) {
    this._ensureEmail();
    const summary = this._cache[mealId];
    const isOwn = summary && summary.likers &&
      summary.likers.some(l => l.emoji === emoji && l.email === this._userEmail);
    this._send({ action: isOwn ? 'unlike' : 'like', meal_id: mealId, emoji });
  },

  /** Build a unique meal ID from week/row/col. */
  mealId(spec, meal) {
    return `${spec.year || 0}-${spec.week_number || 0}-${meal.col}-${meal.row}`;
  },

  /** Render a likes bar for one meal. */
  renderBar(mealId) {
    this._ensureEmail();
    const bars = document.querySelectorAll(`.kvi-likes[data-meal-id="${mealId}"]`);
    const summary = this._cache[mealId] || { total: 0, emojis: {}, likers: [] };
    for (const bar of bars) {
      if (summary.total === 0) {
        bar.innerHTML = `<span class="kvi-like-btn kvi-like-add" data-meal-id="${mealId}" data-emoji="👍" title="Like">👍</span>`;
        continue;
      }
      let html = '';
      for (const [emoji, count] of Object.entries(summary.emojis)) {
        const isOwn = summary.likers.some(l => l.emoji === emoji && l.email === this._userEmail);
        const names = summary.likers.filter(l => l.emoji === emoji).map(l => l.name).join(', ');
        html += `<span class="kvi-like-btn${isOwn ? ' kvi-like-own' : ''}" data-meal-id="${mealId}" data-emoji="${emoji}" title="${names}">${emoji}<span class="kvi-like-count">${count}</span></span>`;
      }
      html += `<span class="kvi-like-btn kvi-like-add" data-meal-id="${mealId}" data-emoji="👍" title="Like">+</span>`;
      bar.innerHTML = html;
    }
  },

  /** Render all known likes bars. */
  renderAll() {
    document.querySelectorAll('.kvi-likes[data-meal-id]').forEach(bar => {
      this.renderBar(bar.dataset.mealId);
    });
  },

  /** Show emoji picker near an element. */
  showPicker(anchor, mealId) {
    document.querySelector('.kvi-emoji-picker')?.remove();
    const picker = document.createElement('div');
    picker.className = 'kvi-emoji-picker';
    for (const em of this._emojis) {
      const opt = document.createElement('span');
      opt.className = 'kvi-emoji-opt';
      opt.textContent = em;
      opt.addEventListener('click', (e) => {
        e.stopPropagation();
        this.toggle(mealId, em);
        picker.remove();
      });
      picker.appendChild(opt);
    }
    document.body.appendChild(picker);
    const rect = anchor.getBoundingClientRect();
    picker.style.left = Math.min(rect.left, window.innerWidth - 200) + 'px';
    picker.style.top = (rect.bottom + 4) + 'px';
    // Auto-close
    setTimeout(() => {
      const close = (e) => { if (!picker.contains(e.target)) { picker.remove(); document.removeEventListener('pointerdown', close); } };
      document.addEventListener('pointerdown', close);
    }, 0);
  },

  /** Bind click/long-press handlers on a container. */
  bindEvents(container) {
    let longPress = null;
    container.addEventListener('click', (e) => {
      const btn = e.target.closest('.kvi-like-btn');
      if (!btn) return;
      e.stopPropagation();
      const mid = btn.dataset.mealId;
      const emoji = btn.dataset.emoji || '👍';
      if (mid) this.toggle(mid, emoji);
    });
    container.addEventListener('pointerdown', (e) => {
      const btn = e.target.closest('.kvi-like-btn');
      if (!btn) return;
      const mid = btn.dataset.mealId;
      if (!mid) return;
      longPress = setTimeout(() => { this.showPicker(btn, mid); longPress = null; }, 500);
    });
    container.addEventListener('pointerup', () => { if (longPress) { clearTimeout(longPress); longPress = null; } });
    container.addEventListener('pointercancel', () => { if (longPress) { clearTimeout(longPress); longPress = null; } });
  },
};

// Initialize once
_kantiniLikes.init();


/* ══════════════════════════════════════════════════════════════
   Kantini Result Plugin — renders canteen meal cards inline
   ══════════════════════════════════════════════════════════════ */

const kantiniResultPlugin = {
  inline: true,
  sidebar: false,
  render: async (container, rawData, blockId) => {
    const spec = _parseJSON(rawData);
    if (!spec) { container.textContent = rawData; return; }

    const scope = spec.scope || 'today';

    // Week mode → open viewer popup immediately, no inline rendering
    if (scope === 'week') {
      const app = window.__chatApp;
      if (app && typeof openMenuViewer === 'function') {
        openMenuViewer(app, {
          meals: spec.all_meals || spec.meals,
          week_label: spec.week_label,
          location: spec.location,
        });
      } else if (app && app.appExtActivate) {
        app._appExtBoltMode = 'viewer';
        app.appExtActivate('kantini');
      }
      container.innerHTML = `<div style="padding:12px;color:#6b7280;font-size:12px;font-style:italic">📋 ${spec.week_label || 'Wochenplan'} wird angezeigt…</div>`;
      return;
    }

    // Day mode → render inline meal cards with KW dropdown
    const meals = spec.meals || [];
    const noMenu = spec.no_menu || !meals.length;
    const dayLabel = meals.length ? meals[0].day : '';
    const _tagIcons = { beef: '🐄', pork: '🐖', poultry: '🐔', fish: '🐟', vegan: '🌱', vegetarian: '🥬', lamb: '🐑' };

    // Build KW dropdown (always show current + next 2 weeks)
    const weeks = spec.available_weeks || [];
    const currentKW = spec.week_number || 0;
    const currentYear = spec.year || 0;
    let kwDropdown = '';
    if (weeks.length) {
      const opts = weeks.map(w => {
        const sel = w.year === currentYear && w.week_number === currentKW ? ' selected' : '';
        const df = (w.date_from || '').replace(/^\d{4}-/, '').replace(/-/g, '.');
        const dt = (w.date_to || '').replace(/^\d{4}-/, '').replace(/-/g, '.');
        const dateHint = df ? ` (${df} – ${dt})` : '';
        return `<option value="${w.year}-${w.week_number}"${sel}>KW${w.week_number}${dateHint}</option>`;
      }).join('');
      kwDropdown = `<select class="kvi-kw-select">${opts}</select>`;
    } else {
      kwDropdown = `<span class="kvi-subtitle">${spec.week_label || ''}</span>`;
    }

    const locLabel = (spec.location || 'Metzingen').charAt(0).toUpperCase() + (spec.location || '').slice(1);
    let html = `<div class="kvi-container">`;
    html += `<div class="kvi-header"><span class="kvi-icon">🍽️</span>`;
    if (dayLabel) html += `<span class="kvi-title">${dayLabel}</span>`;
    html += `${kwDropdown}<span class="kvi-subtitle">${locLabel}</span></div>`;

    const mealIds = [];
    if (noMenu) {
      html += '<div class="kvi-no-menu">Kein Speiseplan für diese Woche verfügbar</div>';
    } else {
      html += '<div class="kvi-meals">';
      for (const m of meals) {
        const n = { kcal: m.kcal, carbs: m.carbs_g, protein: m.protein_g, fat: m.fat_g };
        const tagHtml = (m.tags || []).map(t => _tagIcons[t] ? `<span class="kvi-tag">${_tagIcons[t]}</span>` : '').join('');
        const mid = _kantiniLikes.mealId(spec, m);
        mealIds.push(mid);

        html += '<div class="kvi-meal">';
        if (m.image_url) {
          html += `<div class="kvi-img"><img src="${m.image_url}" loading="lazy">${m.image_is_ai ? '<i class="kvi-ai">AI</i>' : ''}${m.bio ? '<i class="kvi-bio">🌱</i>' : ''}</div>`;
        }
        html += '<div class="kvi-info">';
        html += `<div class="kvi-type">${m.menu_type || ''}</div>`;
        html += `<div class="kvi-name">${m.name}</div>`;
        if (m.description && m.description !== m.name) html += `<div class="kvi-desc">${m.description}</div>`;
        let meta = '';
        if (n.kcal) meta += `<span class="kvi-kcal">${n.kcal} kcal</span>`;
        if (tagHtml) meta += tagHtml;
        if (m.bio) meta += '<span class="kvi-tag kvi-tag-bio">BIO</span>';
        if (m.allergens) meta += `<span class="kvi-allergens" title="Allergene: ${m.allergens}">[${m.allergens}]</span>`;
        if (meta) html += `<div class="kvi-meta">${meta}</div>`;
        const total = (n.carbs || 0) + (n.protein || 0) + (n.fat || 0);
        if (total > 0) {
          const cP = Math.round((n.carbs / total) * 100);
          const pP = Math.round((n.protein / total) * 100);
          html += `<div class="kvi-nutri-bar"><div style="width:${cP}%;background:#f59e0b"></div><div style="width:${pP}%;background:#ef4444"></div><div style="width:${100-cP-pP}%;background:#3b82f6"></div></div>`;
        }
        html += '</div>';
        // Likes bar — right side of card
        html += `<div class="kvi-likes" data-meal-id="${mid}"></div>`;
        html += '</div>';
      }
      html += '</div>';
    }
    html += '</div>';
    container.innerHTML = html;

    // Click meal images to open lightbox (with copy/download buttons)
    const app = window.__chatApp;
    if (app && app._openLightbox) {
      container.querySelectorAll('.kvi-img img').forEach(img => {
        img.style.cursor = 'pointer';
        img.addEventListener('click', () => app._openLightbox(img.src));
      });
    }

    // Meal likes — bind click/long-press and fetch initial counts
    _kantiniLikes.bindEvents(container);
    if (mealIds && mealIds.length) _kantiniLikes.fetchLikes(mealIds);

    // Bind KW dropdown → send message asking for that week
    const select = container.querySelector('.kvi-kw-select');
    if (select) {
      select.addEventListener('change', (e) => {
        const [y, w] = e.target.value.split('-').map(Number);
        const app = window.__chatApp;
        if (!app) return;
        app.el.textarea.value = `Was gibt es in KW${w} zu essen?`;
        app.el.textarea.dispatchEvent(new Event('input'));
        // Auto-submit
        const sendBtn = document.getElementById('cv2-send-btn');
        if (sendBtn) sendBtn.click();
      });
    }
  },
};

/* ══════════════════════════════════════════════════════════════
   Registration
   ══════════════════════════════════════════════════════════════ */

function registerBuiltinPlugins(registry) {
  // Document-type plugins live in llming-docs now. Host page loader
  // includes the `doc-plugins.js` bundle from llming_docs.get_static_dir()
  // and calls window.registerLlmingDocPlugins(registry) separately —
  // we no longer register plotly / text_doc / table / presentation / html /
  // email_draft / latex here. The definitions above are kept temporarily
  // as dead code until the Phase 3 cleanup excises them.

  // Transient / non-document render plugins — fenced blocks are legitimate
  // for these: they're tool-result renderings or pure markdown extensions,
  // not managed documents. These stay in the host.
  registry.register('mermaid',        mermaidPlugin);
  registry.register('rich_mcp',       richMcpPlugin);
  registry.register('kantini_result', kantiniResultPlugin);
  // Follow-up questions (loaded from chat-followup.js)
  if (window._followupPlugin) registry.register('followup', window._followupPlugin);
}

window.registerBuiltinPlugins = registerBuiltinPlugins;
window._renderProductNumberCard = _renderProductNumber;

/* ══════════════════════════════════════════════════════════════
   Debug Document API — exposes doc operations for programmatic
   control via the debug API's doc_command WS message.
   ══════════════════════════════════════════════════════════════ */

window._cv2DocApi = {
  listDocuments: function() {
    var blocks = document.querySelectorAll('.cv2-doc-plugin-block');
    var result = [];
    blocks.forEach(function(b) {
      result.push({
        block_id: b.dataset.blockId || '',
        type: b.dataset.lang || '',
        name: b.dataset.name || b.querySelector('h1,h2,h3')?.textContent?.trim() || '',
      });
    });
    return {ok: true, documents: result};
  },

  openWindowed: function(blockId) {
    if (!blockId) return {ok: false, error: 'block_id required'};
    _showDocWindowed(blockId);
    return {ok: true, block_id: blockId};
  },

  closeWindow: function() {
    _dismissPreview();
    return {ok: true, state: _pvState};
  },

  maximize: function() {
    _maximizePreview();
    return {ok: true, state: _pvState};
  },

  restore: function() {
    _restorePreview();
    return {ok: true, state: _pvState};
  },

  getState: function() {
    return {ok: true, state: _pvState, has_element: !!_pvEl};
  },

  getContent: function(format) {
    var doc = _pvEl?.querySelector('.cv2-doc-text, .cv2-doc-word, .cv2-doc-email-body, [contenteditable="true"]')
           || document.querySelector('.cv2-doc-text, .cv2-doc-word');
    if (!doc) return {ok: false, error: 'No active document found'};
    format = format || 'text';
    if (format === 'html') return {ok: true, content: doc.innerHTML, format: 'html'};
    if (format === 'text') return {ok: true, content: doc.innerText, format: 'text'};
    return {ok: false, error: 'Unknown format: ' + format};
  },

  selectText: function(searchText) {
    if (!searchText) return {ok: false, error: 'search text required'};
    var doc = _pvEl?.querySelector('.cv2-doc-text, .cv2-doc-word, [contenteditable="true"]')
           || document.querySelector('.cv2-doc-text, .cv2-doc-word');
    if (!doc) return {ok: false, error: 'No active document found'};
    var walker = document.createTreeWalker(doc, NodeFilter.SHOW_TEXT, null, false);
    var node, found = false;
    while ((node = walker.nextNode())) {
      var idx = node.textContent.indexOf(searchText);
      if (idx >= 0) {
        var range = document.createRange();
        range.setStart(node, idx);
        range.setEnd(node, idx + searchText.length);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        found = true;
        break;
      }
    }
    return found
      ? {ok: true, selected: searchText}
      : {ok: false, error: 'Text not found: ' + searchText.substring(0, 50)};
  },

  getSelection: function() {
    var sel = window.getSelection();
    var text = sel ? sel.toString() : '';
    return {ok: true, text: text, has_selection: text.length > 0};
  },

  setCursor: function(position, afterText) {
    var doc = _pvEl?.querySelector('.cv2-doc-text, .cv2-doc-word, [contenteditable="true"]')
           || document.querySelector('.cv2-doc-text, .cv2-doc-word');
    if (!doc) return {ok: false, error: 'No active document found'};
    doc.focus();
    var sel = window.getSelection();
    var range = document.createRange();
    if (position === 'end') {
      range.selectNodeContents(doc);
      range.collapse(false);
    } else if (position === 'start') {
      range.selectNodeContents(doc);
      range.collapse(true);
    } else if (position === 'after' && afterText) {
      var walker = document.createTreeWalker(doc, NodeFilter.SHOW_TEXT, null, false);
      var node, placed = false;
      while ((node = walker.nextNode())) {
        var idx = node.textContent.indexOf(afterText);
        if (idx >= 0) {
          range.setStart(node, idx + afterText.length);
          range.collapse(true);
          placed = true;
          break;
        }
      }
      if (!placed) return {ok: false, error: 'Text not found: ' + afterText.substring(0, 50)};
    }
    sel.removeAllRanges();
    sel.addRange(range);
    return {ok: true, position: position};
  },

  typeText: function(text) {
    if (!text) return {ok: false, error: 'text required'};
    var doc = _pvEl?.querySelector('.cv2-doc-text, .cv2-doc-word, [contenteditable="true"]')
           || document.querySelector('.cv2-doc-text, .cv2-doc-word');
    if (!doc) return {ok: false, error: 'No active document found'};
    doc.focus();
    document.execCommand('insertText', false, text);
    return {ok: true, typed: text.length};
  },

  scrollDoc: function(position) {
    var doc = _pvEl?.querySelector('.cv2-doc-text, .cv2-doc-word, [contenteditable="true"]')
           || document.querySelector('.cv2-doc-text, .cv2-doc-word');
    if (!doc) return {ok: false, error: 'No active document found'};
    if (position === 'top') doc.scrollTop = 0;
    else if (position === 'bottom') doc.scrollTop = doc.scrollHeight;
    return {ok: true, position: position};
  },
};
