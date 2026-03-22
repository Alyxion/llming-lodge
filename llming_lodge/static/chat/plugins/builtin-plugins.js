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

/* ── Rich Text Toolbar (shared: email drafts + Word docs) ───── */
function _buildRichToolbar(editableEl, options = {}) {
  const bar = document.createElement('div');
  bar.className = 'cv2-rich-toolbar';

  const buttons = [
    { icon: 'format_bold', cmd: 'bold', title: 'Bold (Ctrl+B)' },
    { icon: 'format_italic', cmd: 'italic', title: 'Italic (Ctrl+I)' },
    { icon: 'format_underline', cmd: 'underline', title: 'Underline (Ctrl+U)' },
    { icon: 'strikethrough_s', cmd: 'strikethrough', title: 'Strikethrough' },
    'sep',
    { icon: 'title', cmd: '_heading', title: 'Heading' },
    { icon: 'format_list_bulleted', cmd: 'insertUnorderedList', title: 'Bullet list' },
    { icon: 'format_list_numbered', cmd: 'insertOrderedList', title: 'Numbered list' },
    'sep',
    { icon: 'link', cmd: '_link', title: 'Insert link' },
    { icon: 'image', cmd: '_image', title: 'Insert image' },
    'sep',
    { icon: 'format_size', cmd: '_fontSize', title: 'Font size' },
    { icon: 'content_paste', cmd: '_pasteClean', title: 'Paste without formatting' },
    { icon: 'format_clear', cmd: 'removeFormat', title: 'Clear formatting' },
    'sep',
    { icon: 'more_horiz', cmd: '_moreMenu', title: 'More options' },
  ];

  let _headingCycle = 0; // 0=H2, 1=H3, 2=P
  const _fontSizes = [
    { label: '8', px: '8px' },
    { label: '10', px: '10px' },
    { label: '11', px: '11px' },
    { label: '12', px: '12px' },
    { label: '14', px: '14px' },
    { label: '16', px: '16px' },
    { label: '18', px: '18px' },
    { label: '20', px: '20px' },
    { label: '24', px: '24px' },
    { label: '28', px: '28px' },
    { label: '36', px: '36px' },
  ];
  let _sizeMenu = null;
  let _moreMenu = null;
  const btnEls = [];

  for (const def of buttons) {
    if (def === 'sep') {
      const sep = document.createElement('span');
      sep.className = 'cv2-rich-toolbar-sep';
      bar.appendChild(sep);
      continue;
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'cv2-rich-toolbar-btn';
    btn.title = def.title;
    btn.innerHTML = `<span class="material-icons">${def.icon}</span>`;
    btn.dataset.cmd = def.cmd;
    bar.appendChild(btn);
    btnEls.push(btn);

    btn.addEventListener('mousedown', (e) => {
      e.preventDefault(); // prevent stealing focus from editable
    });

    btn.addEventListener('click', (e) => {
      e.preventDefault();
      editableEl.focus();
      const cmd = def.cmd;

      if (cmd === '_heading') {
        const tags = ['H2', 'H3', 'P'];
        const fmt = tags[_headingCycle % tags.length];
        document.execCommand('formatBlock', false, `<${fmt}>`);
        _headingCycle = (_headingCycle + 1) % tags.length;
      } else if (cmd === '_link') {
        const sel = window.getSelection();
        const url = prompt('Enter URL:', sel && sel.toString().startsWith('http') ? sel.toString() : 'https://');
        if (url) document.execCommand('createLink', false, url);
      } else if (cmd === '_image') {
        const inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = 'image/*';
        inp.style.display = 'none';
        document.body.appendChild(inp);
        inp.addEventListener('change', () => {
          const file = inp.files[0];
          if (!file) { inp.remove(); return; }
          const reader = new FileReader();
          reader.onload = () => {
            editableEl.focus();
            document.execCommand('insertHTML', false,
              `<img src="${reader.result}" style="max-width:100%;height:auto" />`);
            options.onInput?.();
          };
          reader.readAsDataURL(file);
          inp.remove();
        });
        inp.click();
        return; // onInput called in reader.onload
      } else if (cmd === '_fontSize') {
        if (_sizeMenu) { _sizeMenu.remove(); _sizeMenu = null; return; }
        _sizeMenu = document.createElement('div');
        _sizeMenu.className = 'cv2-rich-popup cv2-rich-size-menu' + (_isDark() ? ' cv2-dark' : '');
        for (const s of _fontSizes) {
          const opt = document.createElement('button');
          opt.type = 'button';
          opt.className = 'cv2-rich-size-option';
          opt.innerHTML = `<span style="font-size:${s.px}">${s.label}</span>`;
          opt.addEventListener('mousedown', (ev) => ev.preventDefault());
          opt.addEventListener('click', () => {
            editableEl.focus();
            // Use fontSize 7 as a marker, then replace the generated <font> tag with a span
            document.execCommand('fontSize', false, '7');
            const fonts = editableEl.querySelectorAll('font[size="7"]');
            for (const f of fonts) {
              const span = document.createElement('span');
              span.style.fontSize = s.px;
              span.innerHTML = f.innerHTML;
              f.replaceWith(span);
            }
            _sizeMenu.remove(); _sizeMenu = null;
            options.onInput?.();
          });
          _sizeMenu.appendChild(opt);
        }
        const sizeRect = btn.getBoundingClientRect();
        _sizeMenu.style.position = 'fixed';
        _sizeMenu.style.left = sizeRect.left + 'px';
        _sizeMenu.style.bottom = (window.innerHeight - sizeRect.top + 4) + 'px';
        document.body.appendChild(_sizeMenu);
        const dismiss = (ev) => {
          if (_sizeMenu && !_sizeMenu.contains(ev.target) && ev.target !== btn) {
            _sizeMenu.remove(); _sizeMenu = null;
            document.removeEventListener('mousedown', dismiss);
          }
        };
        setTimeout(() => document.addEventListener('mousedown', dismiss), 0);
        return;
      } else if (cmd === '_pasteClean') {
        navigator.clipboard.readText().then((text) => {
          if (text) {
            editableEl.focus();
            document.execCommand('insertText', false, text);
            options.onInput?.();
          }
        }).catch(() => {});
        return;
      } else if (cmd === '_moreMenu') {
        if (_moreMenu) { _moreMenu.remove(); _moreMenu = null; return; }
        const sig = (window.__CHAT_CONFIG__ || {}).emailSignature || '';
        const _actions = {};
        const menuItems = [
          { icon: 'horizontal_rule', label: 'Horizontal rule', action: 'hr' },
          { icon: 'code', label: 'Inline code', action: 'code' },
          { icon: 'format_quote', label: 'Blockquote', action: 'blockquote' },
        ];
        _actions.hr = () => { editableEl.focus(); document.execCommand('insertHTML', false, '<hr>'); };
        _actions.code = () => { editableEl.focus(); document.execCommand('insertHTML', false, '<code>' + (window.getSelection()?.toString() || '') + '</code>'); };
        _actions.blockquote = () => { editableEl.focus(); document.execCommand('formatBlock', false, '<blockquote>'); };
        // GhostWriter toggle (off by default, saved in localStorage)
        if (typeof _isGhostEnabled === 'function') {
          const ghostOn = _isGhostEnabled();
          menuItems.push({ icon: ghostOn ? 'toggle_on' : 'toggle_off', label: 'GhostWriter', toggle: true, active: ghostOn, action: 'ghost' });
          _actions.ghost = () => { _setGhostEnabled(!ghostOn); };
        }
        if (sig) {
          menuItems.push({ icon: 'draw', label: 'Add signature', action: 'sig' });
          _actions.sig = () => {
            editableEl.focus();
            const range = document.createRange();
            range.selectNodeContents(editableEl);
            range.collapse(false);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
            document.execCommand('insertHTML', false, '<div class="cv2-email-signature" contenteditable="false">' + sig + '</div>');
          };
        }
        const btnRect = btn.getBoundingClientRect();
        const popup = cv2PopupMenu(menuItems, {
          x: btnRect.right - 160,
          y: btnRect.top - (menuItems.length * 34 + 8),
          minWidth: 160,
          anchor: btn,
          onAction: (action) => {
            if (_actions[action]) _actions[action]();
            if (action !== 'ghost') options.onInput?.();
          },
          onDismiss: () => { _moreMenu = null; },
        });
        _moreMenu = popup.el;
        return;
      } else {
        document.execCommand(cmd, false, null);
      }
      options.onInput?.();
      _updateActiveStates();
    });
  }

  // Active state tracking
  function _updateActiveStates() {
    for (const btn of btnEls) {
      const cmd = btn.dataset.cmd;
      if (cmd && !cmd.startsWith('_') && cmd !== 'removeFormat') {
        try {
          btn.classList.toggle('active', document.queryCommandState(cmd));
        } catch { /* ignore unsupported commands */ }
      }
    }
  }

  document.addEventListener('selectionchange', () => {
    if (editableEl.contains(document.activeElement) || editableEl === document.activeElement) {
      _updateActiveStates();
    }
  });

  return bar;
}

/* ── Image Resize within contenteditable ───────────────────── */
function _setupImageResize(editableEl) {
  let _wrap = null; // current wrapper element with handles

  function _deleteImage() {
    if (!_wrap) return;
    const parent = _wrap.parentElement;
    _wrap.remove();
    _wrap = null;
    // Clean up empty parent elements (e.g., <p> that only contained the image)
    if (parent && parent !== editableEl && !parent.textContent.trim() && !parent.querySelector('img,video,iframe')) {
      parent.remove();
    }
    editableEl.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function _clearHandles() {
    if (!_wrap) return;
    // Unwrap: move img back to parent, remove wrapper
    const img = _wrap.querySelector('img');
    if (img && _wrap.parentElement) {
      img.classList.remove('cv2-img-resize-selected');
      _wrap.parentElement.insertBefore(img, _wrap);
      _wrap.remove();
    }
    _wrap = null;
  }

  function _showHandles(img) {
    _clearHandles();
    img.classList.add('cv2-img-resize-selected');

    // Wrap img in an inline-block container for correct handle positioning
    const wrap = document.createElement('span');
    wrap.contentEditable = 'false';
    wrap.style.cssText = 'display:inline-block;position:relative;line-height:0;';
    img.parentElement.insertBefore(wrap, img);
    wrap.appendChild(img);
    _wrap = wrap;

    for (const corner of ['nw', 'ne', 'sw', 'se']) {
      const handle = document.createElement('span');
      handle.className = `cv2-img-resize-handle cv2-img-resize-handle--${corner}`;

      handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const startX = e.clientX;
        const startW = img.offsetWidth;

        const onMove = (ev) => {
          const dir = corner.includes('e') ? 1 : -1;
          const delta = (ev.clientX - startX) * dir;
          const newW = Math.max(30, startW + delta);
          img.style.width = newW + 'px';
          img.style.height = 'auto';
        };
        const onUp = () => {
          document.removeEventListener('mousemove', onMove);
          document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
      });

      wrap.appendChild(handle);
    }

    // Delete button overlay (top-right)
    const delBtn = document.createElement('span');
    delBtn.className = 'cv2-img-delete-btn';
    delBtn.innerHTML = '&times;';
    delBtn.title = 'Delete image';
    delBtn.addEventListener('mousedown', (e) => {
      e.preventDefault(); e.stopPropagation();
      _deleteImage();
    });
    wrap.appendChild(delBtn);
  }

  editableEl.addEventListener('click', (e) => {
    if (e.target.closest('.cv2-img-resize-handle') || e.target.closest('.cv2-img-delete-btn')) return;
    const img = e.target.closest('img');
    if (img && editableEl.contains(img)) {
      e.preventDefault();
      _showHandles(img);
    } else {
      _clearHandles();
    }
  });

  editableEl.addEventListener('blur', () => {
    setTimeout(() => {
      if (!editableEl.contains(document.activeElement)) _clearHandles();
    }, 100);
  });

  // Delete selected image via keyboard
  editableEl.addEventListener('keydown', (e) => {
    if (!_wrap) return;
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      _deleteImage();
    }
  });

  // Make broken images (e.g. cid: refs) visible and clickable
  editableEl.addEventListener('error', (e) => {
    if (e.target.tagName === 'IMG' && editableEl.contains(e.target)) {
      e.target.classList.add('cv2-img-broken');
    }
  }, true);
}

/* ══════════════════════════════════════════════════════════════
   Embed behavior registry — mirrors Python EMBED_BEHAVIOR.
   Declares how each doc type behaves when embedded in another doc.
   Extensible: new types just add an entry.
   ══════════════════════════════════════════════════════════════ */

const _EMBED_BEHAVIORS = {
  plotly:       { mode: 'graphic', aspect: 1.6 },
  table:        { mode: 'table' },
  text_doc:     { mode: 'text' },
  html_sandbox: { mode: 'graphic', aspect: 1.6 },
  html:         { mode: 'graphic', aspect: 1.6 },
  presentation: { mode: 'graphic', aspect: 16 / 9 },
  email_draft:  { mode: 'text' },
  latex:        { mode: 'graphic', aspect: null },
  rich_mcp:     { mode: 'graphic', aspect: 1.6 },
};

/* ══════════════════════════════════════════════════════════════
   Inline Chart helpers — shared by text_doc + email_draft
   ══════════════════════════════════════════════════════════════ */

/** Hydrate all `.cv2-inline-chart` divs inside `container` that haven't been rendered yet. */
function _hydrateInlineCharts(container) {
  const Plotly = window.Plotly;
  if (!Plotly) return;
  container.querySelectorAll('.cv2-inline-chart:not([data-hydrated])').forEach(el => {
    let spec;
    try { spec = JSON.parse(el.dataset.plotly || '{}'); } catch (_) { return; }
    el.setAttribute('data-hydrated', '1');
    el.contentEditable = 'false';
    const inner = document.createElement('div');
    el.appendChild(inner);
    const theme = _themeColors();
    const layout = {
      ...(spec.layout || {}),
      autosize: true,
      margin: { l: 40, r: 20, t: (spec.layout?.title ? 40 : 20), b: 40, ...(spec.layout?.margin || {}) },
      paper_bgcolor: theme.paper,
      plot_bgcolor: theme.bg,
      font: { color: theme.text, ...(spec.layout?.font || {}) },
      xaxis: { ...(spec.layout?.xaxis || {}), gridcolor: theme.grid, color: theme.text },
      yaxis: { ...(spec.layout?.yaxis || {}), gridcolor: theme.grid, color: theme.text },
      legend: { ...(spec.layout?.legend || {}), font: { color: theme.text }, bgcolor: _isDark() ? 'rgba(30,30,40,0.85)' : 'rgba(255,255,255,0.85)' },
    };
    Plotly.newPlot(inner, spec.data || [], layout, { responsive: true, displayModeBar: false, displaylogo: false });
    el.style.cursor = 'pointer';
    el.addEventListener('click', () => _openPlotlyLightbox(spec, 'inline-chart-' + Math.random().toString(36).slice(2, 8)));
  });
}

/** Convert all `.cv2-inline-chart` divs in a DOM tree to static `<img>` tags.
 *  Operates on the passed container directly (caller should pass a clone). Returns innerHTML. */
async function _convertInlineChartsToImages(container) {
  const Plotly = window.Plotly;
  if (!Plotly) return container.innerHTML;
  const charts = container.querySelectorAll('.cv2-inline-chart');
  for (const el of charts) {
    let spec;
    try { spec = JSON.parse(el.dataset.plotly || '{}'); } catch (_) { el.remove(); continue; }
    const tmpDiv = document.createElement('div');
    tmpDiv.style.cssText = 'position:fixed;left:-9999px;width:800px;height:500px';
    document.body.appendChild(tmpDiv);
    try {
      await Plotly.newPlot(tmpDiv, spec.data || [], {
        ...(spec.layout || {}), width: 800, height: 500,
        paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
        font: { color: '#333333' },
      });
      const imgUrl = await Plotly.toImage(tmpDiv, { format: 'png', width: 800, height: 500 });
      Plotly.purge(tmpDiv);
      const img = document.createElement('img');
      img.src = imgUrl;
      img.style.cssText = 'max-width:100%;height:auto';
      el.replaceWith(img);
    } catch (err) {
      console.warn('[INLINE-CHART] PNG conversion failed:', err);
      el.remove();
    } finally {
      tmpDiv.remove();
    }
  }
  return container.innerHTML;
}

/* ══════════════════════════════════════════════════════════════
   Plotly Plugin — compact inline preview + full-screen lightbox
   ══════════════════════════════════════════════════════════════ */
const plotlyPlugin = {
  inline: true,
  loader: () => _loadScript('/chat-static/vendor/plotly.min.js'),
  render: async (container, rawData, blockId) => {
    const spec = _parseJSON(rawData);
    if (!spec) { container.textContent = rawData; return; }

    const Plotly = window.Plotly;
    const baseLayout = spec.layout || {};
    const theme = _themeColors();

    // ── Inline preview (compact, clickable) ──────────────────
    container.classList.add('cv2-plotly-container');
    const preview = document.createElement('div');
    preview.className = 'cv2-plotly-preview';
    preview.id = `plotly-${blockId}`;
    container.appendChild(preview);

    const inlineLayout = {
      ...baseLayout,
      margin: { l: 40, r: 20, t: baseLayout.title ? 40 : 20, b: 40, ...(baseLayout.margin || {}) },
      autosize: true,
      height: 340,  // match CSS max-height (360px minus padding) — prevents distortion on reload
      paper_bgcolor: theme.paper,
      plot_bgcolor: theme.bg,
      font: { color: theme.text, ...(baseLayout.font || {}) },
      xaxis: { ...(baseLayout.xaxis || {}), gridcolor: theme.grid, color: theme.text },
      yaxis: { ...(baseLayout.yaxis || {}), gridcolor: theme.grid, color: theme.text },
      legend: { ...(baseLayout.legend || {}), font: { color: theme.text }, bgcolor: _isDark() ? 'rgba(30,30,40,0.85)' : 'rgba(255,255,255,0.85)' },
    };
    const inlineConfig = { responsive: true, displayModeBar: false, displaylogo: false };
    await Plotly.newPlot(preview, spec.data || [], inlineLayout, inlineConfig);

    // Click hint overlay
    const hint = document.createElement('div');
    hint.className = 'cv2-plotly-expand-hint';
    hint.innerHTML = '<span class="material-icons">open_in_full</span>';
    container.appendChild(hint);

    // ── Toolbar ──────────────────────────────────────────────
    const toolbar = document.createElement('div');
    toolbar.className = 'cv2-doc-plugin-export';
    toolbar.innerHTML = `
      <button class="cv2-doc-export-btn cv2-plotly-expand-btn" title="Expand"><span class="material-icons" style="font-size:14px;vertical-align:middle">open_in_full</span></button>
      <button class="cv2-doc-export-btn" data-format="svg" title="Download SVG">SVG</button>
      <button class="cv2-doc-export-btn" data-format="png" title="Download PNG">PNG</button>`;
    _addSourceButton(toolbar, rawData);
    _addWorkspaceButton(toolbar, blockId);
    container.appendChild(toolbar);

    // ── Lightbox ─────────────────────────────────────────────
    function openLightbox() { _openPlotlyLightbox(spec, blockId); }

    preview.style.cursor = 'pointer';
    preview.addEventListener('click', openLightbox);
    hint.addEventListener('click', openLightbox);
    toolbar.querySelector('.cv2-plotly-expand-btn').addEventListener('click', openLightbox);
    toolbar.addEventListener('click', (e) => {
      const fmt = e.target.closest('[data-format]')?.dataset?.format;
      if (fmt) Plotly.downloadImage(preview, { format: fmt, filename: `chart-${blockId}` });
    });
  },
};

/* ══════════════════════════════════════════════════════════════
   LaTeX Plugin (uses already-loaded KaTeX)
   ══════════════════════════════════════════════════════════════ */
const latexPlugin = {
  inline: true,
  render: async (container, rawData, blockId) => {
    const formula = rawData.trim();
    const json = _parseJSON(formula);
    const expr = json ? (json.formula || formula) : formula;
    const displayMode = json ? (json.displayMode !== false) : true;

    if (!window.katex) {
      container.textContent = expr;
      return;
    }
    container.classList.add('cv2-doc-latex');
    container.innerHTML = window.katex.renderToString(expr, {
      displayMode,
      throwOnError: false,
    });
    // Source button for latex (in a small toolbar)
    const toolbar = document.createElement('div');
    toolbar.className = 'cv2-doc-plugin-export';
    _addSourceButton(toolbar, rawData);
    _addWorkspaceButton(toolbar, blockId);
    container.appendChild(toolbar);
  },
};

/* ══════════════════════════════════════════════════════════════
   Table Plugin
   ══════════════════════════════════════════════════════════════ */
const tablePlugin = {
  inline: true,
  render: async (container, rawData, blockId) => {
    const spec = _parseJSON(rawData);
    if (!spec || !spec.columns || !spec.rows) {
      container.textContent = rawData;
      return;
    }

    /* Normalise columns: accept strings or {name:…} objects */
    const colNames = spec.columns.map(c => (typeof c === 'string') ? c : (c.name || c.title || c.label || String(c)));

    const table = document.createElement('table');
    table.className = 'cv2-doc-table';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const col of colNames) {
      const th = document.createElement('th');
      th.textContent = col;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const row of spec.rows) {
      const tr = document.createElement('tr');
      if (Array.isArray(row)) {
        for (const cell of row) {
          const td = document.createElement('td');
          td.textContent = (cell != null && typeof cell === 'object') ? JSON.stringify(cell) : (cell ?? '');
          tr.appendChild(td);
        }
      } else if (row && typeof row === 'object') {
        /* Try column-name lookup first; if all miss, fall back to row's own keys in order */
        const rowKeys = Object.keys(row);
        let matched = 0;
        for (const col of colNames) { if (row[col] !== undefined) matched++; }
        const useKeys = (matched === 0 && rowKeys.length > 0);
        const keys = useKeys ? rowKeys.slice(0, colNames.length) : colNames;
        for (const k of keys) {
          const td = document.createElement('td');
          const val = row[k];
          td.textContent = (val != null && typeof val === 'object') ? JSON.stringify(val) : (val ?? '');
          tr.appendChild(td);
        }
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);

    const wrapper = document.createElement('div');
    wrapper.className = 'cv2-doc-table-wrapper';
    wrapper.appendChild(table);
    container.appendChild(wrapper);

    // Register in blockDataStore so window mode works
    const chatApp = document.querySelector('#chat-app')?.__vue_app__?.config?.globalProperties?.$chatApp
      || window.__chatApp;
    if (chatApp?._blockDataStore) {
      chatApp._blockDataStore.register(spec.id || blockId, 'table', spec);
    }

    const fileName = spec.name || spec.title || `table-${blockId}`;

    const exportBar = document.createElement('div');
    exportBar.className = 'cv2-doc-plugin-export';
    exportBar.innerHTML = `
      <button class="cv2-doc-export-btn cv2-table-xlsx-btn" title="Download XLSX">XLSX</button>
      <button class="cv2-doc-export-btn cv2-table-csv-btn" title="Download CSV">CSV</button>`;
    _addSourceButton(exportBar, rawData);
    _addWorkspaceButton(exportBar, blockId);
    container.appendChild(exportBar);

    // CSV export
    exportBar.querySelector('.cv2-table-csv-btn').addEventListener('click', () => {
      const csvRows = [spec.columns.join(',')];
      for (const row of spec.rows) {
        csvRows.push(row.map(c => `"${String(c ?? '').replace(/"/g, '""')}"`).join(','));
      }
      const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${fileName}.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    });

    // XLSX export with formatting (built from raw XML — no SheetJS dependency)
    exportBar.querySelector('.cv2-table-xlsx-btn').addEventListener('click', () => {
      _exportTableAsXlsx(spec, fileName);
    });
  },
};

/**
 * Export a table spec as a well-formatted XLSX file.
 * Builds the XLSX (a ZIP of XML files) from scratch so we get full control
 * over styles — bold headers, background fill, column widths, auto-filter,
 * frozen header row, and proper number types.
 */
/**
 * Build a table spec as an XLSX Blob.
 * @param {{columns: string[], rows: any[][]}} spec - Flat table spec
 * @returns {Blob} XLSX blob
 */
function _buildTableXlsxBlob(spec) {
  const enc = new TextEncoder();
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

  // ── Shared strings table (all unique strings) ─────────────
  const ssMap = new Map();
  const ssList = [];
  function ssIdx(s) {
    const key = String(s);
    if (ssMap.has(key)) return ssMap.get(key);
    const idx = ssList.length;
    ssList.push(key);
    ssMap.set(key, idx);
    return idx;
  }

  // ── Classify cells & build sheet rows ─────────────────────
  // Style 0 = default, Style 1 = header (bold + fill)
  const sheetRows = [];

  // Header row (style 1)
  const hCells = spec.columns.map(col => ({ t: 's', v: ssIdx(col), s: 1 }));
  sheetRows.push(hCells);

  // Data rows (style 0)
  for (const row of spec.rows) {
    const cells = row.map(cell => {
      if (cell === null || cell === undefined || cell === '') return { t: 's', v: ssIdx(''), s: 0 };
      const trimmed = String(cell).trim();
      const num = Number(trimmed);
      if (trimmed !== '' && !isNaN(num) && isFinite(num) && String(num) === trimmed) {
        return { t: 'n', v: num, s: 0 };
      }
      return { t: 's', v: ssIdx(cell), s: 0 };
    });
    sheetRows.push(cells);
  }

  // ── Column widths (character units, auto-fit) ─────────────
  const colWidths = spec.columns.map((col, i) => {
    let maxLen = String(col).length;
    for (const row of spec.rows) {
      const v = String(row[i] ?? '');
      if (v.length > maxLen) maxLen = v.length;
    }
    return Math.min(Math.max(maxLen + 3, 10), 60);
  });

  const lastColLetter = _colLetter(spec.columns.length - 1);
  const lastRowNum = spec.rows.length + 1;

  // ── XML: [Content_Types].xml ──────────────────────────────
  const contentTypes = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>`;

  // ── XML: _rels/.rels ──────────────────────────────────────
  const rels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>`;

  // ── XML: xl/_rels/workbook.xml.rels ────────────────────────
  const wbRels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>`;

  // ── XML: xl/workbook.xml ──────────────────────────────────
  const workbook = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets>
</workbook>`;

  // ── XML: xl/styles.xml (style 0 = default, style 1 = bold header) ──
  const styles = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/><color rgb="FF1F2937"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF3F4F6"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left/><right/><top/><bottom style="medium"><color rgb="FFD1D5DB"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
  </cellXfs>
</styleSheet>`;

  // ── XML: xl/sharedStrings.xml ─────────────────────────────
  const ssXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="${ssList.length}" uniqueCount="${ssList.length}">
${ssList.map(s => `<si><t>${esc(s)}</t></si>`).join('\n')}
</sst>`;

  // ── XML: xl/worksheets/sheet1.xml ─────────────────────────
  let sheetDataXml = '';
  for (let r = 0; r < sheetRows.length; r++) {
    let rowXml = `<row r="${r + 1}">`;
    for (let c = 0; c < sheetRows[r].length; c++) {
      const cell = sheetRows[r][c];
      const ref = `${_colLetter(c)}${r + 1}`;
      if (cell.t === 'n') {
        rowXml += `<c r="${ref}" s="${cell.s}"><v>${cell.v}</v></c>`;
      } else {
        rowXml += `<c r="${ref}" s="${cell.s}" t="s"><v>${cell.v}</v></c>`;
      }
    }
    rowXml += '</row>';
    sheetDataXml += rowXml;
  }

  const colsXml = colWidths.map((w, i) =>
    `<col min="${i + 1}" max="${i + 1}" width="${w}" bestFit="1" customWidth="1"/>`
  ).join('');

  const sheet = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetViews><sheetView tabSelected="1" workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols>${colsXml}</cols>
  <sheetData>${sheetDataXml}</sheetData>
  <autoFilter ref="A1:${lastColLetter}${lastRowNum}"/>
</worksheet>`;

  // ── Build ZIP ─────────────────────────────────────────────
  const files = [
    { name: '[Content_Types].xml', data: enc.encode(contentTypes) },
    { name: '_rels/.rels', data: enc.encode(rels) },
    { name: 'xl/workbook.xml', data: enc.encode(workbook) },
    { name: 'xl/_rels/workbook.xml.rels', data: enc.encode(wbRels) },
    { name: 'xl/styles.xml', data: enc.encode(styles) },
    { name: 'xl/sharedStrings.xml', data: enc.encode(ssXml) },
    { name: 'xl/worksheets/sheet1.xml', data: enc.encode(sheet) },
  ];

  return _buildZipBlob(files);
}

function _exportTableAsXlsx(spec, fileName) {
  const blob = _buildTableXlsxBlob(spec);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${fileName.replace(/[^a-zA-Z0-9_\- äöüÄÖÜß]/g, '')}.xlsx`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Convert 0-based column index to Excel letter (0→A, 25→Z, 26→AA) */
function _colLetter(c) {
  let s = '';
  c++;
  while (c > 0) { c--; s = String.fromCharCode(65 + (c % 26)) + s; c = Math.floor(c / 26); }
  return s;
}

/** Build a ZIP Blob from an array of {name, data: Uint8Array} entries (stored, no compression). */
function _buildZipBlob(files) {
  const entries = files.map(f => {
    const nameBytes = new TextEncoder().encode(f.name);
    return { nameBytes, data: f.data, crc: _crc32(f.data) };
  });

  // Calculate total size
  let offset = 0;
  for (const e of entries) { e.offset = offset; offset += 30 + e.nameBytes.length + e.data.length; }
  const cdOffset = offset;
  let cdSize = 0;
  for (const e of entries) cdSize += 46 + e.nameBytes.length;
  const buf = new Uint8Array(offset + cdSize + 22);
  const dv = new DataView(buf.buffer);
  let p = 0;

  // Local file headers + data
  for (const e of entries) {
    dv.setUint32(p, 0x04034b50, true); p += 4;
    dv.setUint16(p, 20, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint32(p, e.crc, true); p += 4;
    dv.setUint32(p, e.data.length, true); p += 4;
    dv.setUint32(p, e.data.length, true); p += 4;
    dv.setUint16(p, e.nameBytes.length, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    buf.set(e.nameBytes, p); p += e.nameBytes.length;
    buf.set(e.data, p); p += e.data.length;
  }
  // Central directory
  for (const e of entries) {
    dv.setUint32(p, 0x02014b50, true); p += 4;
    dv.setUint16(p, 20, true); p += 2;
    dv.setUint16(p, 20, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint32(p, e.crc, true); p += 4;
    dv.setUint32(p, e.data.length, true); p += 4;
    dv.setUint32(p, e.data.length, true); p += 4;
    dv.setUint16(p, e.nameBytes.length, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint16(p, 0, true); p += 2;
    dv.setUint32(p, 0, true); p += 4;
    dv.setUint32(p, e.offset, true); p += 4;
    buf.set(e.nameBytes, p); p += e.nameBytes.length;
  }
  // End of central directory
  dv.setUint32(p, 0x06054b50, true); p += 4;
  dv.setUint16(p, 0, true); p += 2;
  dv.setUint16(p, 0, true); p += 2;
  dv.setUint16(p, entries.length, true); p += 2;
  dv.setUint16(p, entries.length, true); p += 2;
  dv.setUint32(p, cdSize, true); p += 4;
  dv.setUint32(p, cdOffset, true); p += 4;
  dv.setUint16(p, 0, true);

  return new Blob([buf], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
}

/** CRC-32 for ZIP. */
function _crc32(data) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (let j = 0; j < 8; j++) crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

/* ── HTML → text doc sections converter ────────────────────── */
function _htmlToTextDocSections(el) {
  const sections = [];
  for (const child of el.children) {
    const tag = child.tagName;
    if (/^H[1-6]$/.test(tag)) {
      sections.push({ type: 'heading', level: parseInt(tag[1]), content: child.innerHTML });
    } else if (tag === 'UL' || tag === 'OL') {
      const items = [];
      for (const li of child.querySelectorAll('li')) items.push(li.innerHTML);
      sections.push({ type: 'list', ordered: tag === 'OL', items });
    } else if (tag === 'TABLE') {
      const headers = [];
      const rows = [];
      const ths = child.querySelectorAll('thead th');
      for (const th of ths) headers.push(th.textContent);
      for (const tr of child.querySelectorAll('tbody tr')) {
        const row = [];
        for (const td of tr.querySelectorAll('td')) row.push(td.textContent);
        rows.push(row);
      }
      sections.push({ type: 'table', headers, rows });
    } else if (child.classList?.contains('cv2-doc-embed')) {
      // Generic embed — preserve the reference
      const ref = child.dataset.embedRef;
      if (ref) {
        sections.push({ type: 'embed', '$ref': ref });
      }
    } else if (child.classList?.contains('cv2-inline-chart')) {
      try {
        const cd = JSON.parse(child.dataset.plotly || '{}');
        sections.push({ type: 'chart', data: cd.data || [], layout: cd.layout || {} });
      } catch (_) {}
    } else {
      sections.push({ type: 'paragraph', content: child.innerHTML });
    }
  }
  return sections;
}

/* ══════════════════════════════════════════════════════════════
   Text Document Plugin
   ══════════════════════════════════════════════════════════════ */
const textDocPlugin = {
  inline: true,
  render: async (container, rawData, blockId) => {
    const spec = _parseJSON(rawData);
    if (!spec || !spec.sections) { container.textContent = rawData; return; }

    const doc = document.createElement('div');
    doc.className = 'cv2-doc-text';
    const _mobile = _isMobileDoc();
    doc.contentEditable = _mobile ? 'false' : 'true';
    doc.spellcheck = !_mobile;

    for (const section of spec.sections) {
      let el;
      switch (section.type) {
        case 'heading': {
          const level = Math.min(Math.max(section.level || 1, 1), 6);
          el = document.createElement(`h${level}`);
          el.innerHTML = Array.isArray(section.content) ? section.content.join(' ') : (section.content || '');
          break;
        }
        case 'paragraph':
          el = document.createElement('p');
          el.innerHTML = Array.isArray(section.content) ? section.content.join('<br>') : (section.content || '');
          break;
        case 'list': {
          const items = section.items || (Array.isArray(section.content) ? section.content : (section.content ? String(section.content).split('\n') : []));
          el = document.createElement(section.ordered ? 'ol' : 'ul');
          for (const item of items) { const li = document.createElement('li'); li.innerHTML = item; el.appendChild(li); }
          break;
        }
        case 'table': {
          // Table data can be at section level or nested under section.content
          const tbl = section.content && typeof section.content === 'object' && (section.content.headers || section.content.rows)
            ? section.content : section;
          el = document.createElement('table');
          el.className = 'cv2-doc-table';
          if (tbl.headers) {
            const thead = document.createElement('thead');
            const tr = document.createElement('tr');
            for (const h of tbl.headers) { const th = document.createElement('th'); th.textContent = h; tr.appendChild(th); }
            thead.appendChild(tr);
            el.appendChild(thead);
          }
          if (tbl.rows) {
            const tbody = document.createElement('tbody');
            for (const row of tbl.rows) {
              const tr = document.createElement('tr');
              for (const cell of row) { const td = document.createElement('td'); td.textContent = cell ?? ''; tr.appendChild(td); }
              tbody.appendChild(tr);
            }
            el.appendChild(tbody);
          }
          break;
        }
        case 'chart': {
          // Legacy chart section — kept for backward compat
          const chartSrc = section.content && typeof section.content === 'object' && (section.content.data || section.content.layout)
            ? section.content : section;
          el = document.createElement('div');
          el.className = 'cv2-inline-chart';
          el.dataset.plotly = JSON.stringify({
            data: chartSrc.data || [],
            layout: chartSrc.layout || {},
          });
          break;
        }
        case 'embed': {
          // Generic embed — $ref was resolved by resolveBlockRefs.
          // Determine source type from BlockDataStore by looking up the
          // resolved data's id field.
          const chatApp = window.__chatApp;
          const store = chatApp?._blockDataStore;
          const srcEntry = section.id ? store?.get(section.id) : null;
          const srcLang = srcEntry?.lang || '';
          // Import the embed behavior registry from render.py client-side mirror
          const embedBehavior = _EMBED_BEHAVIORS[srcLang];
          const mode = embedBehavior?.mode || 'graphic';

          if (mode === 'graphic' && (section.data || section.layout)) {
            // Graphic mode (plotly, html, etc.) — render as inline chart
            el = document.createElement('div');
            el.className = 'cv2-doc-embed cv2-inline-chart';
            el.dataset.embedRef = section.id || '';
            el.dataset.embedMode = 'graphic';
            el.dataset.plotly = JSON.stringify({
              data: section.data || [],
              layout: section.layout || {},
            });
          } else if (mode === 'table' && (section.columns || section.rows)) {
            // Table mode — render as inline table
            const tSpec = section;
            // Apply cross-type compat: columns ↔ headers
            const headers = tSpec.headers || tSpec.columns || [];
            const rows = tSpec.rows || [];
            el = document.createElement('table');
            el.className = 'cv2-doc-embed cv2-doc-table';
            el.dataset.embedRef = section.id || '';
            el.dataset.embedMode = 'table';
            if (headers.length) {
              const thead = document.createElement('thead');
              const tr = document.createElement('tr');
              for (const h of headers) {
                const th = document.createElement('th');
                th.textContent = typeof h === 'object' ? (h.label || h.key || '') : h;
                tr.appendChild(th);
              }
              thead.appendChild(tr);
              el.appendChild(thead);
            }
            if (rows.length) {
              const tbody = document.createElement('tbody');
              const colKeys = headers.map(h => typeof h === 'object' ? (h.key || h.label || '') : h);
              for (const row of rows) {
                const tr = document.createElement('tr');
                if (Array.isArray(row)) {
                  for (const cell of row) { const td = document.createElement('td'); td.textContent = cell ?? ''; tr.appendChild(td); }
                } else if (typeof row === 'object') {
                  for (const k of colKeys) { const td = document.createElement('td'); td.textContent = row[k] ?? ''; tr.appendChild(td); }
                }
                tbody.appendChild(tr);
              }
              el.appendChild(tbody);
            }
          } else if (mode === 'text' && section.sections) {
            // Text mode — inline the sections as a sub-document
            el = document.createElement('div');
            el.className = 'cv2-doc-embed';
            el.dataset.embedRef = section.id || '';
            el.dataset.embedMode = 'text';
            el.style.cssText = 'border-left:3px solid #ccc;padding-left:12px;margin:8px 0';
            for (const sub of section.sections) {
              const p = document.createElement('p');
              p.innerHTML = sub.content || '';
              el.appendChild(p);
            }
          } else {
            // Fallback: show as placeholder
            el = document.createElement('div');
            el.className = 'cv2-doc-embed';
            el.dataset.embedRef = section.id || '';
            el.dataset.embedMode = mode;
            el.style.cssText = 'padding:12px;background:#f5f5f5;border-radius:6px;margin:8px 0;color:#666;font-style:italic';
            el.textContent = `[Embedded ${srcLang || 'document'}${section.name ? ': ' + section.name : ''}]`;
          }
          break;
        }
        default:
          el = document.createElement('p');
          el.innerHTML = section.content || '';
      }
      if (el) doc.appendChild(el);
    }
    _hydrateInlineCharts(doc);

    const _docId = spec.id || blockId;
    let _syncing = false;  // guard to prevent sync loops

    function _syncTextDoc() {
      if (_syncing) return;
      spec.sections = _htmlToTextDocSections(doc);
      const chatApp = document.querySelector('#chat-app')?.__vue_app__?.config?.globalProperties?.$chatApp
        || window.__chatApp;
      if (chatApp?._blockDataStore) {
        chatApp._blockDataStore.register(_docId, 'text_doc', spec);
      }
      if (chatApp?.inlineDocBlocks) {
        const idx = Array.isArray(chatApp.inlineDocBlocks)
          ? chatApp.inlineDocBlocks.findIndex(b => b.id === _docId || b.blockId === blockId)
          : -1;
        if (idx >= 0) {
          chatApp.inlineDocBlocks[idx].data = JSON.stringify(spec);
        } else if (!Array.isArray(chatApp.inlineDocBlocks)) {
          chatApp.inlineDocBlocks[blockId] = { lang: 'text_doc', content: JSON.stringify(spec) };
        }
      }
      // Persist edits to localStorage (survives conversation switches)
      try {
        localStorage.setItem('text_doc:edits:' + _docId, JSON.stringify({
          html: doc.innerHTML, ts: Date.now(),
        }));
      } catch (_) {}
      // Dispatch sync event so other instances (windowed ↔ inline) update in real-time
      document.dispatchEvent(new CustomEvent('cv2:text-doc-sync', {
        detail: { id: _docId, sourceBlockId: blockId, html: doc.innerHTML },
      }));
    }

    // Listen for sync from other instances of the same document
    function _onExternalSync(ev) {
      if (!container.isConnected) {
        document.removeEventListener('cv2:text-doc-sync', _onExternalSync);
        return;
      }
      const d = ev.detail;
      if (d.id !== _docId || d.sourceBlockId === blockId) return;
      _syncing = true;
      doc.innerHTML = d.html;
      _hydrateInlineCharts(doc);
      spec.sections = _htmlToTextDocSections(doc);
      _syncing = false;
    }
    document.addEventListener('cv2:text-doc-sync', _onExternalSync);

    // Restore edits from localStorage if available (conversation switch back)
    try {
      let saved = localStorage.getItem('text_doc:edits:' + _docId);
      // Migrate from old key
      if (!saved) {
        saved = localStorage.getItem('word_doc:edits:' + _docId);
        if (saved) { localStorage.setItem('text_doc:edits:' + _docId, saved); localStorage.removeItem('word_doc:edits:' + _docId); }
      }
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.html) {
          doc.innerHTML = parsed.html;
          _hydrateInlineCharts(doc);
          spec.sections = _htmlToTextDocSections(doc);
        }
      }
    } catch (_) {}

    const toolbar = _buildRichToolbar(doc, {
      showSignature: false,
      onInput: _syncTextDoc,
    });
    container.appendChild(toolbar);
    container.appendChild(doc);
    _setupImageResize(doc);

    // AI editing features (desktop only)
    if (!_isMobileDoc() && typeof _buildAIContextMenu === 'function') {
      const _aiOpts = {
        documentId: _docId,
        documentType: 'text_doc',
        documentName: spec.title || 'Document',
        editableEl: doc,
        onInput: _syncTextDoc,
      };
      _buildAIContextMenu(doc, _aiOpts);
      _buildAITaskButton(toolbar, _aiOpts);
      _setupGhostText(doc, _aiOpts);
    }

    doc.addEventListener('input', _syncTextDoc);
    // Register in blockDataStore immediately so window mode works
    _syncTextDoc();

    const exportBar = document.createElement('div');
    exportBar.className = 'cv2-doc-plugin-export';
    exportBar.innerHTML = `<button class="cv2-doc-export-btn cv2-textdoc-docx-btn" title="Download DOCX">DOCX</button><button class="cv2-doc-export-btn cv2-textdoc-html-btn" title="Download HTML">HTML</button>`;
    _addSourceButton(exportBar, rawData);
    _addWorkspaceButton(exportBar, blockId);
    container.appendChild(exportBar);
    exportBar.querySelector('.cv2-textdoc-docx-btn').addEventListener('click', async () => {
      const cfg = window.__CHAT_CONFIG__ || {};
      const apiBase = cfg.wsPath ? cfg.wsPath.replace(/\/ws\/.*/, '') : '/api/llming';
      try {
        // Resolve embeds and convert chart/graphic sections to concrete types for DOCX
        const exportSpec = JSON.parse(JSON.stringify(spec));
        const chatApp = window.__chatApp;
        const store = chatApp?._blockDataStore;
        for (let i = 0; i < exportSpec.sections.length; i++) {
          const s = exportSpec.sections[i];
          // ── Resolve embed sections ──
          if (s.type === 'embed' && s.$ref && store) {
            const entry = store.get(s.$ref);
            if (!entry) continue;
            const behavior = _EMBED_BEHAVIORS[entry.lang];
            const mode = behavior?.mode || 'graphic';
            const srcData = entry.data || {};
            if (mode === 'graphic') {
              // Graphic embed → render to PNG
              const plotlyData = srcData.data || [];
              const plotlyLayout = srcData.layout || {};
              if (window.Plotly && plotlyData.length) {
                const aspect = behavior?.aspect || 1.6;
                const pxW = 800, pxH = Math.round(800 / aspect);
                const tmpDiv = document.createElement('div');
                tmpDiv.style.cssText = `position:fixed;left:-9999px;width:${pxW}px;height:${pxH}px`;
                document.body.appendChild(tmpDiv);
                try {
                  await Plotly.newPlot(tmpDiv, plotlyData, {
                    ...plotlyLayout, width: pxW, height: pxH,
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    font: { color: '#333333' },
                  });
                  const imgUrl = await Plotly.toImage(tmpDiv, { format: 'png', width: pxW, height: pxH });
                  Plotly.purge(tmpDiv);
                  exportSpec.sections[i] = { type: 'image', data: imgUrl };
                } catch (err) {
                  console.warn('[DOCX] Embed graphic conversion failed:', err);
                } finally {
                  tmpDiv.remove();
                }
              }
            } else if (mode === 'table') {
              // Table embed → native table section
              const headers = srcData.headers || srcData.columns || [];
              const colLabels = headers.map(h => typeof h === 'object' ? (h.label || h.key || '') : String(h));
              const colKeys = headers.map(h => typeof h === 'object' ? (h.key || h.label || '') : String(h));
              const rows = (srcData.rows || []).map(r => {
                if (Array.isArray(r)) return r;
                if (typeof r === 'object') return colKeys.map(k => r[k] ?? '');
                return [r];
              });
              exportSpec.sections[i] = { type: 'table', headers: colLabels, rows };
            } else if (mode === 'text' && srcData.sections) {
              // Text embed → splice in the sub-document's sections
              const subSections = srcData.sections.map(sub => ({ ...sub }));
              exportSpec.sections.splice(i, 1, ...subSections);
              i += subSections.length - 1; // adjust index
            }
            continue;
          }
          // ── Legacy chart sections → PNG ──
          if (s.type === 'chart' && window.Plotly) {
            const chartSrc = s.content && typeof s.content === 'object' && (s.content.data || s.content.layout)
              ? s.content : s;
            const tmpDiv = document.createElement('div');
            tmpDiv.style.cssText = 'position:fixed;left:-9999px;width:800px;height:500px';
            document.body.appendChild(tmpDiv);
            try {
              await Plotly.newPlot(tmpDiv, chartSrc.data || [], {
                ...(chartSrc.layout || {}), width: 800, height: 500,
                paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                font: { color: '#333333' },
              });
              const imgUrl = await Plotly.toImage(tmpDiv, { format: 'png', width: 800, height: 500 });
              Plotly.purge(tmpDiv);
              exportSpec.sections[i] = { type: 'image', data: imgUrl };
            } catch (err) {
              console.warn('[DOCX] Chart PNG conversion failed:', err);
            } finally {
              tmpDiv.remove();
            }
          }
        }
        const resp = await fetch(`${apiBase}/word/export`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ spec: exportSpec }),
        });
        if (!resp.ok) { console.error('DOCX export failed:', await resp.text()); return; }
        const blob = await resp.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        const fname = (spec.title || 'document').replace(/[^a-zA-Z0-9_\- äöüÄÖÜß]/g, '').trim() || 'document';
        a.download = `${fname}.docx`;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (err) { console.error('DOCX export error:', err); }
    });
    exportBar.querySelector('.cv2-textdoc-html-btn').addEventListener('click', async () => {
      const clone = doc.cloneNode(true);
      const html = await _convertInlineChartsToImages(clone);
      const fullDoc = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Document</title>
        <style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:800px;margin:40px auto;line-height:1.7;padding:0 20px}
        table{border-collapse:collapse;width:100%}th,td{padding:6px 12px;border:1px solid #ddd;text-align:left}th{background:#f5f5f5}
        img{max-width:100%;height:auto}</style>
        </head><body><div class="cv2-doc-text">${html}</div></body></html>`;
      const blob = new Blob([fullDoc], { type: 'text/html' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'document.html';
      a.click();
      URL.revokeObjectURL(a.href);
    });
  },
};

/* ══════════════════════════════════════════════════════════════
   Presentation Plugin — unified layout (PptxGenJS-compatible)
   ══════════════════════════════════════════════════════════════ */

// Standard PptxGenJS 16:9 slide dimensions (inches)
const PPTX_W = 10;
const PPTX_H = 5.625;
const PPTX_DPI = 96;
const PPTX_VIRT_W = PPTX_W * PPTX_DPI;  // 960
const PPTX_VIRT_H = PPTX_H * PPTX_DPI;  // 540

// Default theme — all hardcoded presentation colors extracted here
const PPTX_DEFAULT_THEME = {
  accentColor: null,       // null = use CSS --chat-accent
  titleColor: '#333333',
  textColor: '#333333',
  subtitleColor: '#666666',
  titleBg: 'linear-gradient(135deg, #fff 0%, #f0f4ff 100%)',
  contentBg: '#ffffff',
  headingFont: 'Arial',
  bodyFont: 'Arial',
  logoUrl: '',
  logoPosition: 'bottom-right',
};

/** Check if a spec uses the template-native format (placeholders + template with layouts) */
function _isTemplateNative(spec) {
  if (!spec.template) return false;
  const templates = window.__CHAT_CONFIG__?.presentationTemplates || [];
  const tpl = templates.find(t => t.name === spec.template);
  return !!(tpl && tpl.layouts && tpl.layouts.length > 0);
}

/** Get the template config object (with layouts) for a spec, or null */
function _getTemplateConfig(spec) {
  if (!spec.template) return null;
  const templates = window.__CHAT_CONFIG__?.presentationTemplates || [];
  const tpl = templates.find(t => t.name === spec.template);
  return (tpl && tpl.layouts && tpl.layouts.length > 0) ? tpl : null;
}

/**
 * Resolve the PPTX theme for a given spec.
 * If spec.template matches a registered template, merge its values on top of defaults.
 */
function _getPptxTheme(spec) {
  const templates = window.__CHAT_CONFIG__?.presentationTemplates || [];
  const tpl = spec.template ? templates.find(t => t.name === spec.template) : null;
  if (!tpl) return { ...PPTX_DEFAULT_THEME };
  return {
    accentColor: tpl.accentColor || PPTX_DEFAULT_THEME.accentColor,
    titleColor: tpl.titleColor || PPTX_DEFAULT_THEME.titleColor,
    textColor: tpl.textColor || PPTX_DEFAULT_THEME.textColor,
    subtitleColor: tpl.subtitleColor || PPTX_DEFAULT_THEME.subtitleColor,
    titleBg: tpl.titleBg || PPTX_DEFAULT_THEME.titleBg,
    contentBg: tpl.contentBg || PPTX_DEFAULT_THEME.contentBg,
    headingFont: tpl.headingFont || PPTX_DEFAULT_THEME.headingFont,
    bodyFont: tpl.bodyFont || PPTX_DEFAULT_THEME.bodyFont,
    logoUrl: tpl.logoUrl || PPTX_DEFAULT_THEME.logoUrl,
    logoPosition: tpl.logoPosition || PPTX_DEFAULT_THEME.logoPosition,
    titleBgImage: tpl.titleBgImage || '',
    contentBgImage: tpl.contentBgImage || '',
    endBgImage: tpl.endBgImage || '',
    templatePath: tpl.templatePath || '',
  };
}

/** Detect whether a slide should render as a title slide */
function _isTitleSlide(slide, idx) {
  if (slide.layout === 'title') return true;
  if (idx !== 0) return false;
  // Index 0 with only a title and at most a subtitle text element
  const elems = slide.elements || [];
  if (elems.length === 0) return true;
  if (elems.length === 1 && (elems[0].type === 'text' || elems[0].type === 'subtitle')) return true;
  return false;
}

/** Detect whether a slide should render as an end slide */
function _isEndSlide(slide, idx, total) {
  if (slide.layout === 'end') return true;
  if (idx !== total - 1 || idx === 0) return false;
  const elems = slide.elements || [];
  if (elems.length === 0) return true;
  if (elems.length === 1 && (elems[0].type === 'text' || elems[0].type === 'subtitle')) return true;
  return false;
}

/**
 * Layout a single template-native slide.
 * Reads placeholder positions from the template config and places content at those coordinates.
 */
function _layoutTemplateSlide(slideSpec, tplConfig, theme, accentHex, slideW, slideH) {
  const layoutDefs = tplConfig.layouts || [];
  const layoutDef = layoutDefs.find(l => l.name === slideSpec.layout);
  if (!layoutDef) {
    // Fallback: if layout not found, render as simple text slide
    const fallbackLayout = layoutDefs.find(l => !l.isTitle && !l.isEnd) || layoutDefs[0];
    if (!fallbackLayout) return { isTitle: false, isEnd: false, title: slideSpec.title || '', elements: [], background: theme.contentBg };
    return _layoutTemplateSlide({ ...slideSpec, layout: fallbackLayout.name }, tplConfig, theme, accentHex, slideW, slideH);
  }

  const elements = [];
  const phs = slideSpec.placeholders || {};

  for (const phDef of layoutDef.placeholders) {
    const val = phs[phDef.name];
    if (val === undefined || val === null) continue;

    if (typeof val === 'string') {
      // Simple text
      const isTitle = phDef.name === 'title';
      const isBody = phDef.name === 'body' || phDef.name === 'subtitle';
      elements.push({
        type: 'text', content: val,
        x: phDef.x, y: phDef.y, w: phDef.w, h: phDef.h,
        fontSize: isTitle ? (layoutDef.isTitle ? 36 : 24) : (isBody ? 14 : 16),
        bold: isTitle,
        color: isTitle ? (layoutDef.isTitle ? theme.titleColor : accentHex)
             : isBody ? theme.subtitleColor : theme.textColor,
        fontFamily: isTitle ? theme.headingFont : theme.bodyFont,
        align: layoutDef.isTitle || layoutDef.isEnd ? 'center' : 'left',
        valign: layoutDef.isTitle && isTitle ? 'middle' : 'top',
      });
    } else if (typeof val === 'object') {
      // Rich content
      switch (val.type) {
        case 'list': {
          const items = val.items || [];
          elements.push({
            type: 'list', items,
            x: phDef.x, y: phDef.y, w: phDef.w, h: phDef.h,
            fontSize: 16, color: theme.textColor,
            fontFamily: theme.bodyFont,
          });
          break;
        }
        case 'table': {
          elements.push({
            type: 'table',
            headers: val.headers || [],
            rows: val.rows || [],
            x: phDef.x, y: phDef.y, w: phDef.w, h: phDef.h,
            fontSize: 12, accentHex,
            fontFamily: theme.bodyFont,
          });
          break;
        }
        case 'chart': {
          elements.push({
            type: 'chart',
            data: val.data || [],
            layout: val.layout || {},
            $ref: val.$ref || '',
            x: phDef.x, y: phDef.y, w: phDef.w, h: phDef.h,
          });
          break;
        }
        case 'image': {
          elements.push({
            type: 'image',
            src: val.src || '',
            alt: val.alt || '',
            x: phDef.x, y: phDef.y, w: phDef.w, h: phDef.h,
          });
          break;
        }
        default: {
          // Treat as text with content field
          elements.push({
            type: 'text', content: val.content || val.text || JSON.stringify(val),
            x: phDef.x, y: phDef.y, w: phDef.w, h: phDef.h,
            fontSize: 16, color: theme.textColor,
            fontFamily: theme.bodyFont,
          });
        }
      }
    }
  }

  const bg = layoutDef.bgImage
    ? `url(${layoutDef.bgImage}) center/cover no-repeat`
    : (layoutDef.isTitle ? theme.titleBg : theme.contentBg);

  return {
    isTitle: layoutDef.isTitle,
    isEnd: layoutDef.isEnd,
    title: slideSpec.title || (phs.title || ''),
    elements,
    background: bg,
  };
}

/**
 * Compute layout for all slides — single source of truth for both
 * inline CSS preview and PptxGenJS export.
 * Returns {slideW, slideH, virtW, virtH, slides: [{isTitle, background, elements: [...]}]}
 * All positions are in inches.
 */
function _layoutSlides(spec) {
  const cssAccent = getComputedStyle(document.documentElement)
    .getPropertyValue('--chat-accent').trim() || '#4F46E5';
  const theme = _getPptxTheme(spec);
  const accentHex = theme.accentColor || cssAccent;

  // Template-native path
  const tplConfig = _getTemplateConfig(spec);
  if (tplConfig) {
    const slideW = tplConfig.slideWidth || PPTX_W;
    const slideH = tplConfig.slideHeight || PPTX_H;
    const virtW = slideW * PPTX_DPI;
    const virtH = slideH * PPTX_DPI;
    const slides = spec.slides.map(slideSpec =>
      _layoutTemplateSlide(slideSpec, tplConfig, theme, accentHex, slideW, slideH)
    );
    return { slideW, slideH, virtW, virtH, slides };
  }

  // Abstract path (existing logic)
  const slideCount = spec.slides.length;
  const slides = spec.slides.map((slideSpec, i) => {
    const isTitle = _isTitleSlide(slideSpec, i);
    const isEnd = _isEndSlide(slideSpec, i, slideCount);
    const elements = [];

    if (isTitle) {
      if (slideSpec.title) {
        elements.push({
          type: 'text', content: slideSpec.title,
          x: 0.5, y: PPTX_H * 0.3, w: 9, h: 1.2,
          fontSize: 36, bold: true, color: theme.titleColor,
          fontFamily: theme.headingFont,
          align: 'center', valign: 'middle',
        });
      }
      // Accent bar
      elements.push({
        type: 'shape',
        x: PPTX_W * 0.4, y: PPTX_H * 0.52, w: PPTX_W * 0.2, h: 0.06,
        fill: accentHex,
      });
      // Subtitle from elements
      const subtitleElem = (slideSpec.elements || []).find(
        e => e.type === 'subtitle' || e.type === 'text'
      );
      if (subtitleElem) {
        elements.push({
          type: 'text', content: subtitleElem.content || subtitleElem.text || '',
          x: 0.5, y: PPTX_H * 0.57, w: 9, h: 0.8,
          fontSize: 20, color: theme.subtitleColor,
          fontFamily: theme.bodyFont,
          align: 'center',
        });
      }
      if (spec.author) {
        elements.push({
          type: 'text', content: spec.author,
          x: 0.5, y: PPTX_H * 0.72, w: 9, h: 0.5,
          fontSize: 14, color: theme.subtitleColor,
          fontFamily: theme.bodyFont,
          align: 'center',
        });
      }
    } else {
      // Content slide
      let yPos = 0.4;
      if (slideSpec.title) {
        elements.push({
          type: 'text', content: slideSpec.title,
          x: 0.5, y: yPos, w: 9, h: 0.7,
          fontSize: 28, bold: true, color: accentHex,
          fontFamily: theme.headingFont,
        });
        yPos += 0.8;
      }

      for (const elem of (slideSpec.elements || [])) {
        switch (elem.type) {
          case 'text':
          case 'heading':
            elements.push({
              type: 'text', content: elem.content || elem.text || '',
              x: 0.5, y: yPos, w: 9, h: 0.5,
              fontSize: elem.type === 'heading' ? 22 : 16,
              bold: elem.type === 'heading',
              color: theme.textColor,
              fontFamily: elem.type === 'heading' ? theme.headingFont : theme.bodyFont,
            });
            yPos += 0.6;
            break;
          case 'subtitle':
            elements.push({
              type: 'text', content: elem.content || elem.text || '',
              x: 0.5, y: yPos, w: 9, h: 0.5,
              fontSize: 16, color: theme.subtitleColor,
              fontFamily: theme.bodyFont,
            });
            yPos += 0.6;
            break;
          case 'list': {
            const items = elem.items || (elem.content ? elem.content.split('\n') : []);
            const h = Math.min(items.length * 0.35 + 0.2, 4);
            elements.push({
              type: 'list', items,
              x: 0.5, y: yPos, w: 9, h,
              fontSize: 16, color: theme.textColor,
              fontFamily: theme.bodyFont,
            });
            yPos += Math.min(items.length * 0.35 + 0.3, 4.1);
            break;
          }
          case 'table': {
            const headers = elem.headers || [];
            const rows = elem.rows || [];
            const totalRows = (headers.length > 0 ? 1 : 0) + rows.length;
            const h = Math.min(totalRows * 0.35 + 0.3, 4);
            elements.push({
              type: 'table', headers, rows,
              x: 0.5, y: yPos, w: 9, h,
              fontSize: 12, accentHex,
              fontFamily: theme.bodyFont,
            });
            yPos += h;
            break;
          }
          case 'chart': {
            elements.push({
              type: 'chart',
              data: elem.data || [],
              layout: elem.layout || {},
              x: 0.5, y: yPos, w: 9, h: 4.5,
            });
            yPos += 4.6;
            break;
          }
          case 'image': {
            elements.push({
              type: 'image',
              src: elem.src || '',
              alt: elem.alt || '',
              x: 0.5, y: yPos, w: 9, h: 4,
            });
            yPos += 4.1;
            break;
          }
        }
      }
    }

    // Logo — skip when template has bg images (logo already in the template)
    const hasTemplateBg = theme.titleBgImage || theme.contentBgImage;
    if (theme.logoUrl && !hasTemplateBg) {
      const pos = theme.logoPosition || 'bottom-right';
      const lw = 1.2, lh = 0.5;
      let lx = PPTX_W - lw - 0.3, ly = PPTX_H - lh - 0.2;
      if (pos === 'bottom-left')  { lx = 0.3; }
      if (pos === 'top-right')    { ly = 0.2; }
      if (pos === 'top-left')     { lx = 0.3; ly = 0.2; }
      elements.push({
        type: 'image', src: theme.logoUrl, alt: 'logo',
        x: lx, y: ly, w: lw, h: lh,
      });
    }

    // Slide number
    const showNumbers = spec.slideNumbers !== false;
    const skipTitle = spec.slideNumbersSkipTitle && isTitle;
    if (showNumbers && !skipTitle) {
      elements.push({
        type: 'slideNumber', number: i + 1,
        x: PPTX_W * 0.9, y: PPTX_H * 0.93, w: 0.5, h: 0.3,
        fontSize: 10, color: theme.subtitleColor, align: 'right',
      });
    }

    // Pick background: prefer template bg images, fall back to CSS gradients
    let bg;
    if (isTitle && theme.titleBgImage) {
      bg = `url(${theme.titleBgImage}) center/cover no-repeat`;
    } else if (isEnd && theme.endBgImage) {
      bg = `url(${theme.endBgImage}) center/cover no-repeat`;
    } else if (!isTitle && theme.contentBgImage) {
      bg = `url(${theme.contentBgImage}) center/cover no-repeat`;
    } else {
      bg = isTitle ? theme.titleBg : theme.contentBg;
    }

    return {
      isTitle,
      isEnd,
      title: slideSpec.title,
      elements,
      background: bg,
    };
  });
  return { slideW: PPTX_W, slideH: PPTX_H, virtW: PPTX_VIRT_W, virtH: PPTX_VIRT_H, slides };
}

/** Set absolute position on an element from layout coordinates (inches to px). */
function _positionElem(div, el) {
  div.style.position = 'absolute';
  div.style.left = `${el.x * PPTX_DPI}px`;
  div.style.top = `${el.y * PPTX_DPI}px`;
  div.style.width = `${el.w * PPTX_DPI}px`;
  div.style.height = `${el.h * PPTX_DPI}px`;
  div.style.boxSizing = 'border-box';
}

/**
 * Render a slide preview into a container element using CSS absolute positioning.
 * Uses a virtual coordinate space (inches × DPI) that maps to slide coordinates,
 * then CSS-scaled to fit the container width.
 * @param {HTMLElement} container
 * @param {object} layout - {slideW, slideH, virtW, virtH, slides: [...]}
 * @param {number} slideIdx
 */
function _renderSlidePreview(container, layout, slideIdx) {
  const slide = layout.slides[slideIdx];
  const virtW = layout.virtW;
  const virtH = layout.virtH;
  container.innerHTML = '';

  const frame = document.createElement('div');
  frame.className = 'cv2-pptx-slide-frame';
  frame.style.background = slide.background;
  frame.style.aspectRatio = `${layout.slideW} / ${layout.slideH}`;

  const virtual = document.createElement('div');
  virtual.className = 'cv2-pptx-virtual';
  virtual.style.width = `${virtW}px`;
  virtual.style.height = `${virtH}px`;

  const ptToPx = PPTX_DPI / 72;  // 1.333...
  const FOOTER_H = 32; /* px reserved at bottom for slide number */

  for (const el of slide.elements) {
    switch (el.type) {
      case 'text': {
        const div = document.createElement('div');
        div.className = 'cv2-pptx-elem';
        _positionElem(div, el);
        div.style.fontSize = `${el.fontSize * ptToPx}px`;
        div.style.fontFamily = (el.fontFamily || 'Arial') + ', sans-serif';
        if (el.color) div.style.color = el.color;
        if (el.bold) div.style.fontWeight = '700';
        if (el.align) div.style.textAlign = el.align;
        if (el.valign === 'middle') {
          div.style.display = 'flex';
          div.style.alignItems = 'center';
          if (el.align === 'center') div.style.justifyContent = 'center';
        }
        div.style.overflow = 'hidden';
        div.style.lineHeight = '1.3';
        const span = document.createElement('span');
        span.textContent = el.content;
        div.appendChild(span);
        virtual.appendChild(div);
        break;
      }
      case 'shape': {
        const div = document.createElement('div');
        div.className = 'cv2-pptx-elem';
        _positionElem(div, el);
        div.style.background = el.fill;
        virtual.appendChild(div);
        break;
      }
      case 'list': {
        const div = document.createElement('div');
        div.className = 'cv2-pptx-elem';
        _positionElem(div, el);
        div.style.fontSize = `${el.fontSize * ptToPx}px`;
        div.style.fontFamily = (el.fontFamily || 'Arial') + ', sans-serif';
        if (el.color) div.style.color = el.color;
        /* Let lists grow beyond declared height like tables. */
        div.style.height = 'auto';
        div.style.minHeight = `${el.h * PPTX_DPI}px`;
        div.style.overflow = 'visible';
        div.style.lineHeight = '1.4';
        const ul = document.createElement('ul');
        ul.style.margin = '0';
        ul.style.paddingLeft = '28px';
        ul.style.listStyleType = 'disc';
        for (const item of el.items) {
          const li = document.createElement('li');
          li.textContent = item;
          li.style.marginBottom = '4px';
          ul.appendChild(li);
        }
        div.appendChild(ul);
        /* Mark for auto-fit font shrink like tables. */
        const listAvailH = virtH - el.y * PPTX_DPI - FOOTER_H;
        div.dataset.pptxAutofit = '1';
        div.dataset.availH = listAvailH;
        virtual.appendChild(div);
        break;
      }
      case 'table': {
        const SLIDE_PAD = 24; /* px right-side safety margin */
        const div = document.createElement('div');
        div.className = 'cv2-pptx-elem';
        _positionElem(div, el);
        /* Available space from element position to slide edges (with padding + footer). */
        const availW = virtW - el.x * PPTX_DPI - SLIDE_PAD;
        const availH = virtH - el.y * PPTX_DPI - FOOTER_H;
        div.style.width = `${availW}px`;
        div.style.height = 'auto';
        div.style.overflow = 'hidden';
        div.style.fontSize = `${el.fontSize * ptToPx}px`;
        div.style.fontFamily = (el.fontFamily || 'Arial') + ', sans-serif';
        const table = document.createElement('table');
        table.style.width = '100%';
        table.style.borderCollapse = 'collapse';
        table.style.tableLayout = 'fixed';
        table.style.overflowWrap = 'break-word';
        if (el.headers && el.headers.length > 0) {
          const thead = document.createElement('thead');
          const tr = document.createElement('tr');
          for (const h of el.headers) {
            const th = document.createElement('th');
            th.textContent = h;
            th.style.cssText = `padding:4px 8px;background:${el.accentHex};color:#fff;font-weight:700;text-align:left;border:0.5px solid #ccc;overflow:hidden;text-overflow:ellipsis;`;
            tr.appendChild(th);
          }
          thead.appendChild(tr);
          table.appendChild(thead);
        }
        if (el.rows && el.rows.length > 0) {
          const tbody = document.createElement('tbody');
          for (const row of el.rows) {
            const tr = document.createElement('tr');
            for (const cell of (Array.isArray(row) ? row : [])) {
              const td = document.createElement('td');
              td.textContent = cell ?? '';
              td.style.cssText = 'padding:4px 8px;border:0.5px solid #ccc;';
              tr.appendChild(td);
            }
            tbody.appendChild(tr);
          }
          table.appendChild(tbody);
        }
        div.appendChild(table);
        /* Mark for post-render font-size shrink if table overflows vertically. */
        div.dataset.pptxAutofit = '1';
        div.dataset.availH = availH;
        virtual.appendChild(div);
        break;
      }
      case 'chart': {
        const div = document.createElement('div');
        div.className = 'cv2-pptx-elem';
        _positionElem(div, el);
        div.style.overflow = 'hidden';
        if (window.Plotly) {
          const plotDiv = document.createElement('div');
          plotDiv.style.width = '100%';
          plotDiv.style.height = '100%';
          div.appendChild(plotDiv);
          const chartW = el.w * PPTX_DPI;
          const chartH = el.h * PPTX_DPI;
          requestAnimationFrame(() => {
            window.Plotly.newPlot(plotDiv, el.data, {
              ...(el.layout),
              width: chartW, height: chartH,
              autosize: false,
              margin: { l: 30, r: 20, t: (el.layout?.title) ? 40 : 15, b: 30, ...(el.layout?.margin || {}) },
              paper_bgcolor: '#ffffff',
              plot_bgcolor: '#ffffff',
              font: { size: 11, color: '#333', ...(el.layout?.font || {}) },
            }, { responsive: false, displayModeBar: false }).then(() => {
              plotDiv.style.cursor = 'pointer';
              plotDiv.addEventListener('click', () => _openPlotlyLightbox(el, `pptx-chart-${slideIdx}`));
            });
          });
        } else {
          div.textContent = '[Chart: Plotly not loaded]';
          div.style.display = 'flex';
          div.style.alignItems = 'center';
          div.style.justifyContent = 'center';
          div.style.color = '#999';
        }
        virtual.appendChild(div);
        break;
      }
      case 'image': {
        const div = document.createElement('div');
        div.className = 'cv2-pptx-elem';
        _positionElem(div, el);
        const img = document.createElement('img');
        img.src = el.src;
        img.alt = el.alt;
        img.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain;';
        div.appendChild(img);
        virtual.appendChild(div);
        break;
      }
      case 'slideNumber': {
        const div = document.createElement('div');
        div.className = 'cv2-pptx-elem';
        /* Fixed bottom-right position, ignoring LLM coordinates. */
        div.style.position = 'absolute';
        div.style.right = '16px';
        div.style.bottom = '6px';
        div.style.fontSize = `${(el.fontSize || 8) * ptToPx}px`;
        div.style.color = el.color || '#999';
        div.style.textAlign = 'right';
        div.textContent = el.number;
        virtual.appendChild(div);
        break;
      }
    }
  }

  frame.appendChild(virtual);
  container.appendChild(frame);

  // Auto-fit: shrink font size until content fits available height.
  const _autofitElements = () => {
    virtual.querySelectorAll('[data-pptx-autofit]').forEach(div => {
      const aH = parseFloat(div.dataset.availH);
      if (!aH || aH <= 0) return;
      const child = div.querySelector('table') || div.querySelector('ul');
      if (!child) return;
      let fontSize = parseFloat(getComputedStyle(div).fontSize);
      const minFont = 6;
      /* Shrink font in 1px steps until content fits or min reached. */
      while (child.scrollHeight > aH && fontSize > minFont) {
        fontSize -= 1;
        div.style.fontSize = `${fontSize}px`;
      }
    });
  };

  // Scale virtual layer to fit frame width
  const scaleToFit = () => {
    const frameW = frame.offsetWidth;
    if (frameW <= 0) return;
    const scale = frameW / virtW;
    virtual.style.transform = `scale(${scale})`;
  };
  requestAnimationFrame(() => { _autofitElements(); scaleToFit(); });

  const ro = new ResizeObserver(() => scaleToFit());
  ro.observe(frame);
  frame._resizeObserver = ro;
}

const presentationPlugin = {
  inline: true,
  render: async (container, rawData, blockId) => {
    const spec = _parseJSON(rawData);
    if (!spec || !spec.slides) { container.textContent = rawData; return; }

    // Detect charts in both abstract (elements) and template-native (placeholders) formats
    const hasChart = spec.slides.some(s => {
      if ((s.elements || []).some(e => e.type === 'chart')) return true;
      if (s.placeholders) {
        return Object.values(s.placeholders).some(v => v && typeof v === 'object' && v.type === 'chart');
      }
      return false;
    });
    if (hasChart) await _loadScript('/chat-static/vendor/plotly.min.js');

    const layout = _layoutSlides(spec);
    let currentIdx = 0;

    const deck = document.createElement('div');
    deck.className = 'cv2-doc-pptx';

    function renderSlide(idx) {
      deck.innerHTML = '';

      const nav = document.createElement('div');
      nav.className = 'cv2-doc-pptx-nav';
      nav.innerHTML = `<button class="cv2-doc-pptx-prev" ${idx === 0 ? 'disabled' : ''}>&larr;</button>
        <span>${idx + 1} / ${spec.slides.length}</span>
        <button class="cv2-doc-pptx-next" ${idx === spec.slides.length - 1 ? 'disabled' : ''}>&rarr;</button>`;
      deck.appendChild(nav);

      const slideContainer = document.createElement('div');
      slideContainer.className = 'cv2-doc-pptx-slide-container';
      deck.appendChild(slideContainer);

      _renderSlidePreview(slideContainer, layout, idx);

      slideContainer.addEventListener('dblclick', () => _openPptxLightbox(spec, layout, currentIdx));

      nav.querySelector('.cv2-doc-pptx-prev')?.addEventListener('click', () => {
        if (idx > 0) { currentIdx--; renderSlide(currentIdx); }
      });
      nav.querySelector('.cv2-doc-pptx-next')?.addEventListener('click', () => {
        if (idx < spec.slides.length - 1) { currentIdx++; renderSlide(currentIdx); }
      });
    }

    renderSlide(0);
    container.appendChild(deck);

    // Register in BlockDataStore so email attachment export can find the spec
    const chatApp = document.querySelector('#chat-app')?.__vue_app__?.config?.globalProperties?.$chatApp
      || window.__chatApp || window._chatApp;
    if (chatApp?._blockDataStore) {
      chatApp._blockDataStore.register(spec.id || blockId, 'presentation', spec);
    }

    const exportBar = document.createElement('div');
    exportBar.className = 'cv2-doc-plugin-export';
    exportBar.innerHTML = `
      <button class="cv2-doc-export-btn cv2-pptx-expand-btn" title="Fullscreen"><span class="material-icons" style="font-size:14px;vertical-align:middle">open_in_full</span></button>
      <button class="cv2-doc-export-btn cv2-pptx-export-btn" title="Download PPTX">PPTX</button>`;
    _addSourceButton(exportBar, rawData);
    _addWorkspaceButton(exportBar, blockId);
    container.appendChild(exportBar);

    exportBar.querySelector('.cv2-pptx-expand-btn').addEventListener('click', () =>
      _openPptxLightbox(spec, layout, currentIdx)
    );
    exportBar.querySelector('.cv2-pptx-export-btn').addEventListener('click', (e) =>
      _showPptxExportMenu(e.currentTarget, layout, spec)
    );
  },
};

/* ── PPTX Lightbox ─────────────────────────────────────── */
function _openPptxLightbox(spec, layout, startIdx) {
  let idx = startIdx || 0;
  const totalSlides = layout.slides.length;
  const overlay = document.createElement('div');
  overlay.className = 'cv2-pptx-lightbox';
  overlay.innerHTML = `
    <div class="cv2-pptx-lightbox-chrome">
      <span class="cv2-pptx-lightbox-title">${_escHtml(spec.title || '')}</span>
      <span class="cv2-pptx-lightbox-counter">${idx + 1} / ${totalSlides}</span>
      <div style="flex:1"></div>
      <button class="cv2-doc-export-btn cv2-pptx-lb-export" title="Download PPTX">PPTX</button>
      <button class="cv2-pptx-lightbox-close"><span class="material-icons">close</span></button>
    </div>
    <div class="cv2-pptx-lightbox-body">
      <button class="cv2-pptx-lightbox-nav cv2-pptx-lightbox-nav--prev">&larr;</button>
      <div class="cv2-pptx-lightbox-slide-wrap"></div>
      <button class="cv2-pptx-lightbox-nav cv2-pptx-lightbox-nav--next">&rarr;</button>
    </div>`;
  document.body.appendChild(overlay);

  const slideWrap = overlay.querySelector('.cv2-pptx-lightbox-slide-wrap');
  const counter = overlay.querySelector('.cv2-pptx-lightbox-counter');
  const prevBtn = overlay.querySelector('.cv2-pptx-lightbox-nav--prev');
  const nextBtn = overlay.querySelector('.cv2-pptx-lightbox-nav--next');

  function renderLbSlide() {
    // Clean up previous ResizeObserver
    const oldFrame = slideWrap.querySelector('.cv2-pptx-slide-frame');
    if (oldFrame?._resizeObserver) oldFrame._resizeObserver.disconnect();

    _renderSlidePreview(slideWrap, layout, idx);
    counter.textContent = `${idx + 1} / ${totalSlides}`;
    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx === totalSlides - 1;
  }

  renderLbSlide();

  prevBtn.addEventListener('click', () => { if (idx > 0) { idx--; renderLbSlide(); } });
  nextBtn.addEventListener('click', () => { if (idx < totalSlides - 1) { idx++; renderLbSlide(); } });

  const close = () => {
    const frames = overlay.querySelectorAll('.cv2-pptx-slide-frame');
    frames.forEach(f => { if (f._resizeObserver) f._resizeObserver.disconnect(); });
    overlay.remove();
    document.removeEventListener('keydown', onKey);
  };
  overlay.querySelector('.cv2-pptx-lightbox-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay || e.target.classList.contains('cv2-pptx-lightbox-body')) close();
  });
  overlay.querySelector('.cv2-pptx-lb-export').addEventListener('click', (e) =>
    _showPptxExportMenu(e.currentTarget, layout, spec)
  );

  function onKey(e) {
    if (e.key === 'Escape') close();
    if (e.key === 'ArrowLeft' && idx > 0) { idx--; renderLbSlide(); }
    if (e.key === 'ArrowRight' && idx < totalSlides - 1) { idx++; renderLbSlide(); }
  }
  document.addEventListener('keydown', onKey);
}

/* ── PPTX Server-side export (template-based) ──────────── */
async function _exportPptxServerSide(spec) {
  const cfg = window.__CHAT_CONFIG__ || {};
  const sessionId = cfg.sessionId;
  const apiBase = cfg.wsPath ? cfg.wsPath.replace(/\/ws\/.*/, '') : '/api/llming';

  // Resolve $ref embeds (chart, table) before export
  _resolvePptxEmbeds(spec);

  // Pre-render charts to base64 PNGs (handles both elements[] and placeholders{})
  const chartImages = {};
  let chartIdx = 0;
  const EXPORT_DPI = 150;  // higher than screen DPI for crisp PPTX images

  async function _preRenderChart(elem, pxW, pxH) {
    if (elem.type !== 'chart' || !window.Plotly) return;
    const w = pxW || 800;
    const h = pxH || 450;
    const chartId = `chart_${chartIdx++}`;
    elem._chartImageId = chartId;
    try {
      const tmpDiv = document.createElement('div');
      tmpDiv.style.cssText = `position:fixed;left:-9999px;width:${w}px;height:${h}px`;
      document.body.appendChild(tmpDiv);
      await window.Plotly.newPlot(tmpDiv, elem.data || [], {
        ...(elem.layout || {}),
        width: w, height: h,
        paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
        font: { color: '#333333' },
      });
      chartImages[chartId] = await window.Plotly.toImage(tmpDiv, { format: 'png', width: w, height: h });
      window.Plotly.purge(tmpDiv);
      tmpDiv.remove();
    } catch (err) {
      console.warn('Chart pre-render failed:', err);
    }
  }

  const tplConfig = _getTemplateConfig(spec);

  for (const slide of (spec.slides || [])) {
    // Abstract format: elements[]
    for (const elem of (slide.elements || [])) {
      await _preRenderChart(elem);
    }
    // Template-native format: placeholders{} — use placeholder dimensions for charts
    if (slide.placeholders) {
      const layoutDef = tplConfig?.layouts?.find(l => l.name === slide.layout);
      for (const [phName, val] of Object.entries(slide.placeholders)) {
        if (val && typeof val === 'object') {
          const phDef = layoutDef?.placeholders?.find(p => p.name === phName);
          const pxW = phDef ? Math.round(phDef.w * EXPORT_DPI) : undefined;
          const pxH = phDef ? Math.round(phDef.h * EXPORT_DPI) : undefined;
          await _preRenderChart(val, pxW, pxH);
        }
      }
    }
  }

  try {
    const resp = await fetch(`${apiBase}/pptx/export/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spec, chartImages }),
    });
    if (!resp.ok) {
      const errText = await resp.text();
      console.error('PPTX export failed:', errText);
      return;
    }
    const blob = await resp.blob();
    const filename = (spec.title || 'presentation').replace(/[^a-zA-Z0-9_\- ]/g, '').trim() || 'presentation';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.pptx`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('PPTX server export error:', err);
  }
}

/* ── Build PptxGenJS object from layout (reusable by download + attachment export) ── */
async function _buildPptxGenObj(layout, spec) {
  await _loadScript('/chat-static/vendor/pptxgenjs.bundle.min.js');
  const PptxGenJS = window.PptxGenJS;
  if (!PptxGenJS) throw new Error('PptxGenJS not loaded');

  const pptx = new PptxGenJS();
  pptx.author = spec.author || '';
  pptx.subject = spec.title || '';
  pptx.title = spec.title || 'Presentation';

  const layoutSlides = layout.slides;
  for (let i = 0; i < layoutSlides.length; i++) {
    const slideLayout = layoutSlides[i];
    const pptxSlide = pptx.addSlide();

    for (const el of slideLayout.elements) {
      switch (el.type) {
        case 'text':
          pptxSlide.addText(el.content, {
            x: el.x, y: el.y, w: el.w, h: el.h,
            fontSize: el.fontSize,
            fontFace: el.fontFamily || 'Arial',
            bold: !!el.bold,
            color: (el.color || '#333333').replace('#', ''),
            align: el.align || 'left',
            valign: el.valign || 'top',
          });
          break;

        case 'shape':
          pptxSlide.addShape(
            PptxGenJS.ShapeType ? PptxGenJS.ShapeType.rect : 'rect',
            {
              x: el.x, y: el.y, w: el.w, h: el.h,
              fill: { color: (el.fill || '#333').replace('#', '') },
            }
          );
          break;

        case 'list': {
          const listRows = el.items.map(item => [{
            text: item,
            options: {
              bullet: true,
              fontSize: el.fontSize,
              color: (el.color || '#333333').replace('#', ''),
            },
          }]);
          if (listRows.length > 0) {
            pptxSlide.addText(listRows.flat(), {
              x: el.x, y: el.y, w: el.w, h: el.h,
              fontFace: el.fontFamily || 'Arial', valign: 'top',
            });
          }
          break;
        }

        case 'table': {
          const headers = (el.headers || []).map(h => ({
            text: h,
            options: {
              bold: true, color: 'FFFFFF',
              fill: { color: (el.accentHex || '#4F46E5').replace('#', '') },
            },
          }));
          const rows = (el.rows || []).map(row =>
            (Array.isArray(row) ? row : []).map(cell => ({
              text: String(cell ?? ''),
              options: { fontSize: el.fontSize },
            }))
          );
          const tableRows = headers.length > 0 ? [headers, ...rows] : rows;
          if (tableRows.length > 0) {
            pptxSlide.addTable(tableRows, {
              x: el.x, y: el.y, w: el.w,
              fontSize: el.fontSize, fontFace: el.fontFamily || 'Arial',
              border: { pt: 0.5, color: 'CCCCCC' },
            });
          }
          break;
        }

        case 'chart': {
          if (window.Plotly) {
            try {
              const tmpDiv = document.createElement('div');
              tmpDiv.style.cssText = 'position:fixed;left:-9999px;width:800px;height:450px';
              document.body.appendChild(tmpDiv);
              await window.Plotly.newPlot(tmpDiv, el.data, {
                ...(el.layout),
                width: 800, height: 450,
                paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                font: { color: '#333333' },
              });
              const dataUrl = await window.Plotly.toImage(tmpDiv, { format: 'png', width: 800, height: 450 });
              window.Plotly.purge(tmpDiv);
              tmpDiv.remove();
              pptxSlide.addImage({ data: dataUrl, x: el.x, y: el.y, w: el.w, h: el.h });
            } catch (err) {
              console.warn('PPTX chart export failed:', err);
              pptxSlide.addText('[Chart export failed]', {
                x: el.x, y: el.y, w: el.w, h: 0.5,
                fontSize: 14, color: '999999',
              });
            }
          }
          break;
        }

        case 'image': {
          if (el.src) {
            try {
              pptxSlide.addImage({ path: el.src, x: el.x, y: el.y, w: el.w, h: el.h });
            } catch (_) {}
          }
          break;
        }

        case 'slideNumber':
          pptxSlide.addText(String(el.number), {
            x: el.x, y: el.y, w: el.w, h: el.h,
            fontSize: el.fontSize,
            color: (el.color || '#999').replace('#', ''),
            align: el.align || 'right',
          });
          break;
      }
    }
  }
  return pptx;
}

/* ── PPTX Export (PptxGenJS) — reads from shared layout ── */
async function _exportPptxFromLayout(layout, spec) {
  // If template has server-side export, use it
  const templates = window.__CHAT_CONFIG__?.presentationTemplates || [];
  const tpl = spec.template ? templates.find(t => t.name === spec.template) : null;
  if (tpl && tpl.templatePath) {
    return _exportPptxServerSide(spec);
  }
  // Otherwise, fall through to client-side PptxGenJS export
  const pptx = await _buildPptxGenObj(layout, spec);
  const filename = (spec.title || 'presentation').replace(/[^a-zA-Z0-9_\- ]/g, '').trim() || 'presentation';
  await pptx.writeFile({ fileName: `${filename}.pptx` });
}


/* ── PPTX Export (Rendered) — pixel-perfect slide screenshots ── */
async function _exportPptxRendered(layout, spec) {
  await _loadScript('/chat-static/vendor/html2canvas.min.js');
  await _loadScript('/chat-static/vendor/pptxgenjs.bundle.min.js');
  const PptxGenJS = window.PptxGenJS;
  if (!PptxGenJS || !window.html2canvas) {
    console.error('Required libraries not loaded');
    return;
  }

  // Target 4K width; compute height from slide aspect ratio
  const TARGET_W = 3840;
  const slideW = layout.slideW || PPTX_W;
  const slideH = layout.slideH || PPTX_H;
  const TARGET_H = Math.round(TARGET_W * (slideH / slideW));
  const virtW = layout.virtW;
  const virtH = layout.virtH;
  const renderScale = TARGET_W / virtW;

  const pptx = new PptxGenJS();
  pptx.author = spec.author || '';
  pptx.title = spec.title || 'Presentation';
  pptx.defineLayout({ name: 'CUSTOM', width: slideW, height: slideH });
  pptx.layout = 'CUSTOM';

  // Render each slide offscreen and capture
  for (let i = 0; i < layout.slides.length; i++) {
    // Create offscreen container
    const offscreen = document.createElement('div');
    offscreen.style.cssText = `position:fixed;left:-99999px;top:0;width:${virtW}px;height:${virtH}px;overflow:hidden;`;
    document.body.appendChild(offscreen);

    // Render slide into container
    _renderSlidePreview(offscreen, layout, i);
    const frame = offscreen.querySelector('.cv2-pptx-slide-frame');
    if (frame) {
      // Force full-size render (no CSS scaling)
      const virt = frame.querySelector('.cv2-pptx-virtual');
      if (virt) virt.style.transform = 'none';
      frame.style.width = `${virtW}px`;
      frame.style.height = `${virtH}px`;
    }

    // Wait for Plotly charts to render
    await new Promise(r => setTimeout(r, 500));

    try {
      const canvas = await window.html2canvas(frame || offscreen, {
        width: virtW,
        height: virtH,
        scale: renderScale,
        useCORS: true,
        allowTaint: true,
        backgroundColor: null,
        logging: false,
      });
      const dataUrl = canvas.toDataURL('image/png');

      const pptxSlide = pptx.addSlide();
      pptxSlide.addImage({ data: dataUrl, x: 0, y: 0, w: slideW, h: slideH });
    } catch (err) {
      console.warn(`Rendered export failed for slide ${i + 1}:`, err);
      const pptxSlide = pptx.addSlide();
      pptxSlide.addText(`[Render failed for slide ${i + 1}]`, {
        x: 1, y: 3, w: 8, h: 1, fontSize: 18, color: '999999', align: 'center',
      });
    }

    // Cleanup
    const oldFrame = offscreen.querySelector('.cv2-pptx-slide-frame');
    if (oldFrame?._resizeObserver) oldFrame._resizeObserver.disconnect();
    offscreen.remove();
  }

  const filename = (spec.title || 'presentation').replace(/[^a-zA-Z0-9_\- ]/g, '').trim() || 'presentation';
  await pptx.writeFile({ fileName: `${filename}.pptx` });
}


/* ── PPTX Export Menu — Editable vs Rendered ─────────────── */
function _showPptxExportMenu(btnEl, layout, spec) {
  // Remove any existing menu
  document.querySelectorAll('.cv2-pptx-export-menu').forEach(m => m.remove());

  const menu = document.createElement('div');
  menu.className = 'cv2-pptx-export-menu';
  menu.style.cssText = `
    position:absolute;bottom:100%;left:0;margin-bottom:4px;
    background:var(--chat-bg-secondary,#2a2a2a);border:1px solid var(--chat-border,#444);
    border-radius:8px;padding:4px;z-index:100;min-width:160px;
    box-shadow:0 4px 12px rgba(0,0,0,0.3);
  `;

  const btnStyle = `
    display:block;width:100%;padding:8px 12px;border:none;border-radius:6px;
    background:transparent;color:var(--chat-text,#eee);font-size:13px;
    cursor:pointer;text-align:left;white-space:nowrap;
  `;
  const descStyle = 'display:block;font-size:11px;opacity:0.6;margin-top:2px;';

  menu.innerHTML = `
    <button class="cv2-pptx-menu-editable" style="${btnStyle}">
      Editable PPTX
      <span style="${descStyle}">Text & charts editable in PowerPoint</span>
    </button>
    <button class="cv2-pptx-menu-rendered" style="${btnStyle}">
      Rendered PPTX (4K)
      <span style="${descStyle}">Pixel-perfect, looks exactly like preview</span>
    </button>
  `;

  // Hover effect
  menu.querySelectorAll('button').forEach(b => {
    b.addEventListener('mouseenter', () => b.style.background = 'var(--chat-accent-light,rgba(79,70,229,0.12))');
    b.addEventListener('mouseleave', () => b.style.background = 'transparent');
  });

  menu.querySelector('.cv2-pptx-menu-editable').addEventListener('click', () => {
    menu.remove();
    _exportPptxFromLayout(layout, spec);
  });
  menu.querySelector('.cv2-pptx-menu-rendered').addEventListener('click', () => {
    menu.remove();
    _exportPptxRendered(layout, spec);
  });

  // Position relative to button
  const wrapper = btnEl.closest('.cv2-doc-plugin-export') || btnEl.parentElement;
  wrapper.style.position = 'relative';
  wrapper.appendChild(menu);

  // Close on click outside
  const closeMenu = (e) => {
    if (!menu.contains(e.target) && e.target !== btnEl) {
      menu.remove();
      document.removeEventListener('click', closeMenu, true);
    }
  };
  setTimeout(() => document.addEventListener('click', closeMenu, true), 0);
}


/* ── Shared lightbox for Plotly charts ─────────────────────── */
function _openPlotlyLightbox(chartSpec, blockId) {
  const Plotly = window.Plotly;
  if (!Plotly) return;
  const baseLayout = chartSpec.layout || {};

  const overlay = document.createElement('div');
  overlay.className = 'cv2-plotly-lightbox';
  overlay.innerHTML = `
    <div class="cv2-plotly-lightbox-chrome">
      <span class="cv2-plotly-lightbox-title">${_escHtml(baseLayout.title?.text || baseLayout.title || '')}</span>
      <div style="flex:1"></div>
      <button class="cv2-doc-export-btn" data-format="svg">SVG</button>
      <button class="cv2-doc-export-btn" data-format="png">PNG</button>
      <button class="cv2-plotly-lightbox-close"><span class="material-icons">close</span></button>
    </div>
    <div class="cv2-plotly-lightbox-plot"></div>`;
  document.body.appendChild(overlay);

  const plotArea = overlay.querySelector('.cv2-plotly-lightbox-plot');
  // Lightbox always dark background
  const fullLayout = {
    ...baseLayout,
    autosize: true,
    margin: { l: 60, r: 40, t: baseLayout.title ? 60 : 30, b: 60, ...(baseLayout.margin || {}) },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e5e7eb', ...(baseLayout.font || {}) },
    xaxis: { ...(baseLayout.xaxis || {}), gridcolor: 'rgba(255,255,255,0.1)', color: '#e5e7eb' },
    yaxis: { ...(baseLayout.yaxis || {}), gridcolor: 'rgba(255,255,255,0.1)', color: '#e5e7eb' },
    legend: { ...(baseLayout.legend || {}), font: { color: '#e5e7eb' }, bgcolor: 'rgba(30,30,40,0.85)' },
  };
  Plotly.newPlot(plotArea, chartSpec.data || [], fullLayout, { responsive: true, displayModeBar: false, scrollZoom: true });

  const close = () => { Plotly.purge(plotArea); overlay.remove(); };
  overlay.querySelector('.cv2-plotly-lightbox-close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', function esc(e) { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', esc); } });
  overlay.addEventListener('click', (e) => {
    const fmt = e.target.closest('[data-format]')?.dataset?.format;
    if (fmt) Plotly.downloadImage(plotArea, { format: fmt, width: 1920, height: 1080, filename: `chart-${blockId}` });
  });
  requestAnimationFrame(() => Plotly.Plots.resize(plotArea));
}

/* ══════════════════════════════════════════════════════════════
   HTML Sandbox Plugin — renders inline card, opens in Workspace
   ══════════════════════════════════════════════════════════════ */
function _buildHtmlSrcDoc(title, cssContent, htmlContent, jsContent, vendorScripts) {
  // Inject dark-mode base CSS so iframe content isn't black-on-dark.
  // Placed BEFORE user CSS so it can be overridden.
  const dark = _isDark();
  const baseCSS = dark
    ? 'html,body{font-family:system-ui,-apple-system,sans-serif;color:#e5e7eb;background:#1e1e30;line-height:1.6}'
      + 'h1,h2,h3,h4,h5,h6{color:#f3f4f6}'
      + 'a{color:#93c5fd}a:visited{color:#c4b5fd}'
      + 'code,pre{background:rgba(255,255,255,0.06);color:#e5e7eb;border-radius:4px}'
      + 'pre{padding:12px;overflow-x:auto}code{padding:2px 4px}'
      + 'table{border-collapse:collapse;width:100%}th,td{border:1px solid rgba(255,255,255,0.10);padding:6px 10px;text-align:left}'
      + 'th{background:rgba(255,255,255,0.05);font-weight:600}'
      + 'hr{border:none;border-top:1px solid rgba(255,255,255,0.10)}'
      + 'blockquote{border-left:3px solid rgba(255,255,255,0.15);margin:12px 0;padding:4px 16px;color:#d1d5db}'
      + 'img{max-width:100%;height:auto}'
      + 'input,select,textarea,button{color:#e5e7eb;background:#2a2a40;border:1px solid rgba(255,255,255,0.15);border-radius:4px;padding:4px 8px}'
    : 'html,body{font-family:system-ui,-apple-system,sans-serif;color:#1f2937;background:#fff;line-height:1.6}'
      + 'table{border-collapse:collapse;width:100%}th,td{border:1px solid #e5e7eb;padding:6px 10px;text-align:left}'
      + 'th{background:#f9fafb;font-weight:600}'
      + 'img{max-width:100%;height:auto}';
  // Inject parent's CSS custom properties so MCP authors can use var(--chat-accent) etc.
  const root = getComputedStyle(document.documentElement);
  const _themeVarNames = ['--chat-accent','--chat-bg','--chat-surface','--chat-text',
    '--chat-border','--chat-text-muted','--chat-code-bg','--chat-code-text','--chat-link'];
  const themeVars = _themeVarNames
    .map(v => { const val = root.getPropertyValue(v).trim(); return val ? `${v}:${val}` : ''; })
    .filter(Boolean)
    .join(';');
  const themeCSS = themeVars ? `:root{${themeVars}}` : '';
  // Listen for live theme updates from parent
  const themeUpdateJS = `window.addEventListener('message',function(e){if(e.data&&e.data.type==='theme_update'){var r=document.documentElement;for(var k in e.data.vars||{})r.style.setProperty(k,e.data.vars[k])}});`;
  // Vendor library scripts (injected inline since iframe has no network access)
  const vendorTags = (vendorScripts || []).filter(Boolean).map(s => `<script>${s}<\/script>`).join('\n');
  return `<!DOCTYPE html><html><head><meta charset="utf-8">
    ${title ? `<title>${_escHtml(title)}</title>` : ''}
    <style>${themeCSS}</style>
    <style>${baseCSS}</style>
    <style>${cssContent}</style></head>
    <body>${htmlContent}${vendorTags}<script>${themeUpdateJS}${jsContent}<\/script></body></html>`;
}

const htmlPlugin = {
  inline: true,
  render: async (container, rawData, blockId) => {
    const spec = _parseJSON(rawData);

    let htmlContent, cssContent, jsContent, title;
    if (spec) {
      htmlContent = spec.html || '';
      cssContent = spec.css || '';
      jsContent = spec.js || '';
      title = spec.title || spec.name || '';
    } else {
      htmlContent = rawData;
      cssContent = '';
      jsContent = '';
      title = '';
    }

    // Pick up vendor scripts if injected by rich MCP rendering
    let vendorScripts;
    try { vendorScripts = JSON.parse(container.dataset.vendorScripts || 'null'); } catch(_) {}
    // Pick up droplet files if present
    const dropletFiles = spec?.droplet_files;
    let dropletFilesJS = '';
    if (dropletFiles && typeof dropletFiles === 'object') {
      dropletFilesJS = `window.__droplet_files__=${JSON.stringify(dropletFiles)};`;
    }
    const srcDoc = _buildHtmlSrcDoc(title, cssContent, htmlContent, dropletFilesJS + jsContent, vendorScripts);

    // ── Inline card (compact, click to open workspace) ──
    const card = document.createElement('div');
    card.className = 'cv2-workspace-card';
    card.innerHTML = `
      <span class="material-icons cv2-workspace-card-icon">web</span>
      <div class="cv2-workspace-card-body">
        <div class="cv2-workspace-card-title">${_escHtml(title || 'HTML App')}</div>
        <div class="cv2-workspace-card-hint">Click to open</div>
      </div>
      <span class="material-icons cv2-workspace-card-arrow">chevron_right</span>`;
    container.appendChild(card);

    // ── Source tabs + export below the card ──
    const extras = document.createElement('div');
    extras.className = 'cv2-doc-html';
    extras.style.marginTop = '6px';

    if (spec) {
      const tabs = document.createElement('div');
      tabs.className = 'cv2-doc-html-tabs';
      const sources = [
        { label: 'HTML', content: htmlContent },
        { label: 'CSS', content: cssContent },
        { label: 'JS', content: jsContent },
      ].filter(s => s.content);
      for (const src of sources) {
        const btn = document.createElement('button');
        btn.className = 'cv2-doc-html-tab';
        btn.textContent = src.label;
        btn.addEventListener('click', () => {
          const existing = extras.querySelector('.cv2-doc-html-source');
          if (existing && existing.dataset.label === src.label) { existing.remove(); return; }
          extras.querySelector('.cv2-doc-html-source')?.remove();
          const pre = document.createElement('pre');
          pre.className = 'cv2-doc-html-source';
          pre.dataset.label = src.label;
          pre.textContent = src.content;
          extras.appendChild(pre);
        });
        tabs.appendChild(btn);
      }
      extras.appendChild(tabs);
    }

    const exportBar = document.createElement('div');
    exportBar.className = 'cv2-doc-plugin-export';
    exportBar.innerHTML = `<button class="cv2-doc-export-btn" title="Download HTML">HTML</button>`;
    _addSourceButton(exportBar, rawData);
    extras.appendChild(exportBar);
    exportBar.querySelector('button').addEventListener('click', () => {
      const blob = new Blob([srcDoc], { type: 'text/html' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${title || 'page'}.html`;
      a.click();
      URL.revokeObjectURL(a.href);
    });
    container.appendChild(extras);

    // ── Open in window on card click ──
    if (blockId) {
      card.addEventListener('click', () => _showDocWindowed(blockId));
    }
  },
};

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

/* ══════════════════════════════════════════════════════════════
   Attachment Export Plugin Registry
   ─────────────────────────────────────────────────────────────
   Unified system for exporting dynamic documents to native file
   formats for email attachments.

   Each exporter:
     async (storeEntry, attachmentName) =>
       { name, content_type, data (base64), size }

   To add a new format, register an exporter:
     _attachmentExporters['my_lang'] = async (entry, attName) => { ... };
   ══════════════════════════════════════════════════════════════ */

/** Convert a Blob to a raw base64 string (no data: prefix). */
function _blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/** Normalize a table store entry to flat {columns: string[], rows: any[][]} for XLSX. */
function _flattenTableSpec(tSpec) {
  const cols = (tSpec.columns || []).map(c => c.label || c.key || String(c));
  const rows = (tSpec.rows || []).map(row => {
    if (Array.isArray(row)) return row;
    return (tSpec.columns || []).map(col => {
      const key = col.key || col;
      return row[key] ?? '';
    });
  });
  return { columns: cols, rows };
}

const _attachmentExporters = {};

// ── Unified render dispatcher ────────────────────────────────
// Central function for converting any doc type to a target format.
// Handles client-side rendering (Plotly→PNG, HTML assembly) and
// delegates to server for binary formats (DOCX, PPTX, XLSX).
async function _renderDocTo(docType, spec, targetFormat, options = {}) {
  const cfg = window.__CHAT_CONFIG__ || {};
  const sessionId = cfg.sessionId;
  const apiBase = cfg.wsPath ? cfg.wsPath.replace(/\/ws\/.*/, '') : '/api/llming';

  // ── Client-side: Plotly → PNG ──
  if (docType === 'plotly' && targetFormat === 'png') {
    if (!window.Plotly) throw new Error('Plotly not loaded');
    const tmpDiv = document.createElement('div');
    tmpDiv.style.cssText = 'position:fixed;left:-9999px;width:800px;height:500px';
    document.body.appendChild(tmpDiv);
    try {
      await Plotly.newPlot(tmpDiv, spec.data || [], {
        ...(spec.layout || {}), width: 800, height: 500,
        paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
        font: { color: '#333333' },
      });
      const imgUrl = await Plotly.toImage(tmpDiv, { format: 'png', width: 800, height: 500 });
      Plotly.purge(tmpDiv);
      return { data: imgUrl.split(',')[1], content_type: 'image/png' };
    } finally {
      tmpDiv.remove();
    }
  }

  // ── Client-side: HTML assembly ──
  if ((docType === 'html' || docType === 'html_sandbox') && targetFormat === 'html') {
    const html = spec.html || spec.content || JSON.stringify(spec);
    const b64 = btoa(unescape(encodeURIComponent(html)));
    return { data: b64, content_type: 'text/html' };
  }

  // ── Server-side: all binary formats ──
  const chartImages = options.chartImages || {};
  const resp = await fetch(`${apiBase}/doc/export/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ spec, type: docType, format: targetFormat, chart_images: chartImages }),
  });
  if (!resp.ok) throw new Error(`Export failed (${docType}→${targetFormat}): ` + await resp.text());
  const blob = await resp.blob();
  const b64 = await _blobToBase64(blob);
  return { data: b64, content_type: resp.headers.get('content-type') || 'application/octet-stream', size: blob.size };
}

// ── Plotly → PNG ────────────────────────────────────────────
_attachmentExporters['plotly'] = async (entry, attName) => {
  if (!window.Plotly) throw new Error('Plotly not loaded');
  const pData = entry.data;
  const tmpDiv = document.createElement('div');
  tmpDiv.style.cssText = 'position:fixed;left:-9999px;width:800px;height:500px';
  document.body.appendChild(tmpDiv);
  try {
    await Plotly.newPlot(tmpDiv, pData.data || [], {
      ...(pData.layout || {}), width: 800, height: 500,
      paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
      font: { color: '#333333' },
    });
    const imgUrl = await Plotly.toImage(tmpDiv, { format: 'png', width: 800, height: 500 });
    Plotly.purge(tmpDiv);
    const b64 = imgUrl.split(',')[1];
    const name = (attName || 'chart').replace(/\.[^.]+$/, '') + '.png';
    return { name, content_type: 'image/png', data: b64, size: Math.round(b64.length * 0.75) };
  } finally {
    tmpDiv.remove();
  }
};

// ── Table → XLSX (native format) ────────────────────────────
_attachmentExporters['table'] = async (entry, attName) => {
  const flat = _flattenTableSpec(entry.data);
  const blob = _buildTableXlsxBlob(flat);
  const b64 = await _blobToBase64(blob);
  const name = (attName || 'table').replace(/\.[^.]+$/, '') + '.xlsx';
  return { name, content_type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', data: b64, size: blob.size };
};

/** Resolve $ref embed elements in a presentation spec in-place.
 *  Converts embed→chart (for plotly) and embed→table (for tables)
 *  so downstream export can render them properly. */
function _resolvePptxEmbeds(spec) {
  const store = window.__chatApp?._blockDataStore;
  if (!store) return;
  for (const slide of (spec.slides || [])) {
    const containers = [slide.elements || []];
    if (slide.placeholders) {
      for (const [k, v] of Object.entries(slide.placeholders)) {
        if (Array.isArray(v)) containers.push(v);
        else if (v && typeof v === 'object' && v.type === 'embed' && v.$ref) {
          // Single-element placeholder — wrap in temp array, resolve, unwrap
          const tmp = [v];
          containers.push(tmp);
          // After resolution, write back
          Object.defineProperty(slide.placeholders, k, { value: tmp[0], writable: true, enumerable: true, configurable: true });
        }
      }
    }
    for (const elems of containers) {
      for (let i = 0; i < elems.length; i++) {
        const elem = elems[i];
        if (elem.type !== 'embed' || !elem.$ref) continue;
        const entry = store.get(elem.$ref);
        if (!entry) continue;
        const behavior = _EMBED_BEHAVIORS[entry.lang];
        const mode = behavior?.mode || 'graphic';
        const srcData = entry.data || {};
        if (mode === 'graphic' && window.Plotly) {
          elems[i] = { type: 'chart', data: srcData.data || [], layout: srcData.layout || {} };
        } else if (mode === 'table') {
          const headers = srcData.headers || srcData.columns || [];
          const colLabels = headers.map(h => typeof h === 'object' ? (h.label || h.key || '') : String(h));
          const colKeys = headers.map(h => typeof h === 'object' ? (h.key || h.label || '') : String(h));
          const rows = (srcData.rows || []).map(r => {
            if (Array.isArray(r)) return r;
            if (typeof r === 'object') return colKeys.map(k => r[k] ?? '');
            return [r];
          });
          elems[i] = { type: 'table', headers: colLabels, rows };
        }
      }
    }
  }
}

// ── PowerPoint → PPTX ──────────────────────────────────────
_attachmentExporters['powerpoint'] = async (entry, attName) => {
  const spec = entry.data;
  // Resolve $ref embeds before export
  _resolvePptxEmbeds(spec);
  const cfg = window.__CHAT_CONFIG__ || {};
  const sessionId = cfg.sessionId;
  const apiBase = cfg.wsPath ? cfg.wsPath.replace(/\/ws\/.*/, '') : '/api/llming';

  // Check for server-side template export
  const templates = cfg.presentationTemplates || [];
  const tpl = spec.template ? templates.find(t => t.name === spec.template) : null;

  if (tpl && tpl.templatePath) {
    // Server-side: use the same flow as _exportPptxServerSide but return base64
    const chartImages = {};
    let chartIdx = 0;
    const EXPORT_DPI = 150;
    const tplConfig = _getTemplateConfig(spec);

    for (const slide of (spec.slides || [])) {
      for (const elem of (slide.elements || [])) {
        if (elem.type === 'chart' && window.Plotly) {
          const chartId = `chart_${chartIdx++}`;
          elem._chartImageId = chartId;
          try {
            const tmpDiv = document.createElement('div');
            tmpDiv.style.cssText = 'position:fixed;left:-9999px;width:800px;height:450px';
            document.body.appendChild(tmpDiv);
            await window.Plotly.newPlot(tmpDiv, elem.data || [], {
              ...(elem.layout || {}), width: 800, height: 450,
              paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: { color: '#333333' },
            });
            chartImages[chartId] = await window.Plotly.toImage(tmpDiv, { format: 'png', width: 800, height: 450 });
            window.Plotly.purge(tmpDiv);
            tmpDiv.remove();
          } catch (_) {}
        }
      }
      if (slide.placeholders) {
        const layoutDef = tplConfig?.layouts?.find(l => l.name === slide.layout);
        for (const [phName, val] of Object.entries(slide.placeholders)) {
          if (val && typeof val === 'object' && val.type === 'chart' && window.Plotly) {
            const chartId = `chart_${chartIdx++}`;
            val._chartImageId = chartId;
            const phDef = layoutDef?.placeholders?.find(p => p.name === phName);
            const pxW = phDef ? Math.round(phDef.w * EXPORT_DPI) : 800;
            const pxH = phDef ? Math.round(phDef.h * EXPORT_DPI) : 450;
            try {
              const tmpDiv = document.createElement('div');
              tmpDiv.style.cssText = `position:fixed;left:-9999px;width:${pxW}px;height:${pxH}px`;
              document.body.appendChild(tmpDiv);
              await window.Plotly.newPlot(tmpDiv, val.data || [], {
                ...(val.layout || {}), width: pxW, height: pxH,
                paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: { color: '#333333' },
              });
              chartImages[chartId] = await window.Plotly.toImage(tmpDiv, { format: 'png', width: pxW, height: pxH });
              window.Plotly.purge(tmpDiv);
              tmpDiv.remove();
            } catch (_) {}
          }
        }
      }
    }

    const resp = await fetch(`${apiBase}/pptx/export/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spec, chartImages }),
    });
    if (!resp.ok) throw new Error('PPTX export failed: ' + await resp.text());
    const blob = await resp.blob();
    const b64 = await _blobToBase64(blob);
    const name = (attName || spec.title || 'presentation').replace(/\.[^.]+$/, '') + '.pptx';
    return { name, content_type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation', data: b64, size: blob.size };
  }

  // Client-side: compute layout from spec, build via PptxGenJS, return base64
  const layout = _layoutSlides(spec);
  const pptx = await _buildPptxGenObj(layout, spec);
  const b64 = await pptx.write({ outputType: 'base64' });
  const name = (attName || spec.title || 'presentation').replace(/\.[^.]+$/, '') + '.pptx';
  return { name, content_type: 'application/vnd.openxmlformats-officedocument.presentationml.presentation', data: b64, size: Math.round(b64.length * 0.75) };
};
_attachmentExporters['presentation'] = _attachmentExporters['powerpoint'];
_attachmentExporters['pptx'] = _attachmentExporters['powerpoint'];

// ── Text Document → DOCX (server-side) ─────────────────────
_attachmentExporters['text_doc'] = async (entry, attName) => {
  const spec = entry.data;
  const cfg = window.__CHAT_CONFIG__ || {};
  const apiBase = cfg.wsPath ? cfg.wsPath.replace(/\/ws\/.*/, '') : '/api/llming';

  const resp = await fetch(`${apiBase}/word/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ spec }),
  });
  if (!resp.ok) throw new Error('DOCX export failed: ' + await resp.text());
  const blob = await resp.blob();
  const b64 = await _blobToBase64(blob);
  const name = (attName || spec.title || 'document').replace(/\.[^.]+$/, '') + '.docx';
  return { name, content_type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', data: b64, size: blob.size };
};
_attachmentExporters['word'] = _attachmentExporters['text_doc'];

// ── HTML / HTML Sandbox → HTML file ─────────────────────────
_attachmentExporters['html_sandbox'] = async (entry, attName) => {
  const hSpec = entry.data;
  const html = hSpec.html || hSpec.content || JSON.stringify(hSpec);
  const b64 = btoa(unescape(encodeURIComponent(html)));
  const name = (attName || 'page').replace(/\.[^.]+$/, '') + '.html';
  return { name, content_type: 'text/html', data: b64, size: html.length };
};
_attachmentExporters['html'] = _attachmentExporters['html_sandbox'];

/* ══════════════════════════════════════════════════════════════
   Email Draft Plugin — rich email composer with preview
   ══════════════════════════════════════════════════════════════ */

const emailDraftPlugin = {
  inline: true,
  render: async (container, rawData, blockId) => {
    const spec = _parseJSON(rawData);
    if (!spec) {
      console.warn('[email_draft] JSON parse failed, rawData length:', rawData.length,
        'first 100:', rawData.substring(0, 100));
      container.textContent = rawData;
      return;
    }

    // ── Persisted email state + edits (survives reload) ──
    const _stateKey = 'email_state:' + (spec.id || blockId);
    const _editsKey = 'email_edits:' + (spec.id || blockId);
    let _emailState = null;
    try { _emailState = JSON.parse(localStorage.getItem(_stateKey)); } catch (_) {}
    let _savedEdits = null;
    try { _savedEdits = JSON.parse(localStorage.getItem(_editsKey)); } catch (_) {}

    // Apply saved edits over the original spec
    let subject = _savedEdits?.subject ?? spec.subject ?? '(no subject)';
    const toList = _savedEdits?.to ?? spec.to ?? [];
    const ccList = _savedEdits?.cc ?? spec.cc ?? [];
    const bccList = _savedEdits?.bcc ?? spec.bcc ?? [];
    let bodyHtml = _savedEdits?.body_html ?? spec.body_html ?? '';
    const attachments = [...(_savedEdits?.attachments ?? spec.attachments ?? [])].map(a => {
      // Normalize LLM-generated {ref, name} (no type) → {type: 'ref', ref, name}
      if (!a.type && a.ref) return { ...a, type: 'ref' };
      return a;
    });
    const docName = spec.name || subject;

    /* ── Outer wrapper ─────────────────────────────────── */
    const wrap = document.createElement('div');
    wrap.className = 'cv2-email-draft';
    wrap.dataset.blockId = blockId;
    wrap.dataset.docId = spec.id || blockId;

    /* ── Header (icon + subject) ───────────────────────── */
    const header = document.createElement('div');
    header.className = 'cv2-email-draft-header';
    const _badgeText = _emailState?.status === 'sent' ? 'Sent' : _emailState?.status === 'draft' ? 'Draft Saved' : 'Draft';
    const _badgeClass = _emailState?.status === 'sent' ? ' cv2-email-badge-sent' : _emailState?.status === 'draft' ? ' cv2-email-badge-saved' : '';
    header.innerHTML = `
      <span class="material-icons cv2-email-draft-icon">mail</span>
      <div class="cv2-email-draft-title">${_escHtml(docName)}</div>
      <span class="cv2-email-draft-badge${_badgeClass}">${_badgeText}</span>`;
    wrap.appendChild(header);

    /* ── Meta fields (To, Cc, Bcc) with live search ───── */
    const meta = document.createElement('div');
    meta.className = 'cv2-email-draft-meta';

    // Mutable recipient lists (so buttons can modify them)
    const recipients = { to: [...toList], cc: [...ccList], bcc: [...bccList] };

    let _searchTimer = null;
    function buildRecipientField(label, fieldName) {
      const list = recipients[fieldName];
      const row = document.createElement('div');
      row.className = 'cv2-email-field';
      row.dataset.field = fieldName;

      const lbl = document.createElement('span');
      lbl.className = 'cv2-email-field-label';
      lbl.textContent = label + ':';
      row.appendChild(lbl);

      const chipsWrap = document.createElement('div');
      chipsWrap.className = 'cv2-email-chips';
      row.appendChild(chipsWrap);

      function renderChips() {
        chipsWrap.innerHTML = '';
        for (const email of list) {
          const chip = document.createElement('span');
          chip.className = 'cv2-email-chip';
          chip.title = email;
          chip.innerHTML = `${_escHtml(email)} <span class="cv2-email-chip-x" title="Remove">&times;</span>`;
          chip.querySelector('.cv2-email-chip-x').addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = list.indexOf(email);
            if (idx >= 0) list.splice(idx, 1);
            renderChips();
          });
          chipsWrap.appendChild(chip);
        }
        // Add search input at the end
        const searchWrap = document.createElement('span');
        searchWrap.className = 'cv2-email-search-wrap';
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'cv2-email-search-input';
        input.placeholder = 'Add…';
        searchWrap.appendChild(input);
        chipsWrap.appendChild(searchWrap);

        const dropdown = document.createElement('div');
        dropdown.className = 'cv2-email-search-dropdown';
        dropdown.style.display = 'none';
        searchWrap.appendChild(dropdown);

        input.addEventListener('input', () => {
          clearTimeout(_searchTimer);
          const q = input.value.trim();
          if (q.length < 2) { dropdown.style.display = 'none'; return; }
          _searchTimer = setTimeout(() => {
            // Send WS message for contact search
            const chatApp = window.__chatApp;
            if (!chatApp || !chatApp.ws) { dropdown.style.display = 'none'; return; }
            chatApp.ws.send({ type: 'directory:search', query: q });
            // Listen for response
            function _onResult(ev) {
              const data = ev.detail;
              if (!data.results || !data.results.length) {
                dropdown.innerHTML = '<div class="cv2-email-search-empty">No results</div>';
                dropdown.style.display = 'block';
                return;
              }
              dropdown.innerHTML = data.results.map(r => `
                <div class="cv2-email-search-item" data-email="${_escHtml(r.email)}">
                  <div class="cv2-email-search-name">${_escHtml(r.name)}</div>
                  <div class="cv2-email-search-detail">${_escHtml(r.email)}${r.title ? ' · ' + _escHtml(r.title) : ''}</div>
                </div>
              `).join('');
              dropdown.style.display = 'block';
              dropdown.querySelectorAll('.cv2-email-search-item').forEach(item => {
                item.addEventListener('mousedown', (ev) => {
                  ev.preventDefault();
                  const email = item.dataset.email;
                  if (!list.includes(email)) list.push(email);
                  dropdown.style.display = 'none';
                  input.value = '';
                  renderChips();
                });
              });
            }
            document.addEventListener('ws:directory:search_result', _onResult, { once: true });
          }, 250);
        });

        // Allow typing a raw email and pressing Enter
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            const val = input.value.trim();
            if (val && val.includes('@') && !list.includes(val)) {
              list.push(val);
              input.value = '';
              dropdown.style.display = 'none';
              renderChips();
            }
          }
          if (e.key === 'Escape') { dropdown.style.display = 'none'; }
        });

        // Hide dropdown on outside click
        input.addEventListener('blur', () => {
          setTimeout(() => { dropdown.style.display = 'none'; }, 200);
        });
      }
      renderChips();
      return row;
    }

    // Subject field (editable)
    const subjectField = document.createElement('div');
    subjectField.className = 'cv2-email-field cv2-email-subject-field';
    subjectField.innerHTML = `
      <span class="cv2-email-field-label">Subject:</span>
      <span class="cv2-email-field-value cv2-email-editable" contenteditable="true" spellcheck="true">${_escHtml(subject)}</span>`;
    meta.appendChild(subjectField);

    let _syncingSubject = false;
    const subjectSpan = subjectField.querySelector('.cv2-email-field-value');
    subjectSpan.addEventListener('input', () => {
      if (_syncingSubject) return;
      subject = subjectSpan.textContent.trim();
      _notifyBlockUpdate();
    });
    subjectSpan.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); subjectSpan.blur(); }
    });

    // Recipient fields
    meta.appendChild(buildRecipientField('To', 'to'));
    if (ccList.length) meta.appendChild(buildRecipientField('Cc', 'cc'));
    if (bccList.length) meta.appendChild(buildRecipientField('Bcc', 'bcc'));

    // Add Cc/Bcc toggle if not already showing
    if (!ccList.length || !bccList.length) {
      const toggleRow = document.createElement('div');
      toggleRow.className = 'cv2-email-field cv2-email-add-fields';
      if (!ccList.length) {
        const ccBtn = document.createElement('button');
        ccBtn.className = 'cv2-email-add-field-btn';
        ccBtn.textContent = '+ Cc';
        ccBtn.addEventListener('click', () => {
          meta.insertBefore(buildRecipientField('Cc', 'cc'), toggleRow);
          ccBtn.remove();
        });
        toggleRow.appendChild(ccBtn);
      }
      if (!bccList.length) {
        const bccBtn = document.createElement('button');
        bccBtn.className = 'cv2-email-add-field-btn';
        bccBtn.textContent = '+ Bcc';
        bccBtn.addEventListener('click', () => {
          meta.insertBefore(buildRecipientField('Bcc', 'bcc'), toggleRow);
          bccBtn.remove();
        });
        toggleRow.appendChild(bccBtn);
      }
      meta.appendChild(toggleRow);
    }

    wrap.appendChild(meta);

    /* ── Body (editable contenteditable div) ─────────────── */
    const bodyWrap = document.createElement('div');
    bodyWrap.className = 'cv2-email-draft-body';
    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'cv2-email-body-content';
    const _emailMobile = _isMobileDoc();
    bodyDiv.contentEditable = _emailMobile ? 'false' : 'true';
    bodyDiv.spellcheck = !_emailMobile;
    let _syncingBody = false; // Guard against sync → input → sync loops
    const _cleanHtml = (typeof DOMPurify !== 'undefined') ? DOMPurify.sanitize(bodyHtml) : bodyHtml;
    bodyDiv.innerHTML = _cleanHtml || '<p><br></p>';
    // Resolve inline doc references — cid: images and {{type:ID}} placeholders
    const _refChatApp = window.__chatApp;

    /** Build inline HTML for a doc block entry (plotly → chart div, table → HTML table). */
    function _inlineDocHtml(docEntry) {
      let raw;
      try { raw = typeof docEntry.data === 'string' ? JSON.parse(docEntry.data) : docEntry.data; } catch (_) { return null; }
      if (!raw) return null;
      if (docEntry.lang === 'plotly') {
        const escaped = JSON.stringify({ data: raw.data || [], layout: raw.layout || {} }).replace(/"/g, '&quot;');
        return `<div class="cv2-inline-chart" data-plotly="${escaped}"></div>`;
      }
      if (docEntry.lang === 'table') {
        const flat = _flattenTableSpec(raw);
        let html = '<table class="cv2-doc-table" style="border-collapse:collapse;width:100%;margin:8px 0">';
        if (flat.columns.length) {
          html += '<thead><tr>' + flat.columns.map(c => `<th style="padding:6px 12px;border:1px solid #555;text-align:left">${_escHtml(String(c))}</th>`).join('') + '</tr></thead>';
        }
        html += '<tbody>' + flat.rows.map(row =>
          '<tr>' + row.map(cell => `<td style="padding:6px 12px;border:1px solid #555">${_escHtml(String(cell ?? ''))}</td>`).join('') + '</tr>'
        ).join('') + '</tbody></table>';
        return html;
      }
      return null;
    }

    // CID images → resolve against inlineDocBlocks or remove
    bodyDiv.querySelectorAll('img[src^="cid:"]').forEach(img => {
      const cid = img.getAttribute('src').replace('cid:', '');
      const docEntry = _refChatApp?.inlineDocBlocks?.find(b => b.id === cid);
      const html = docEntry ? _inlineDocHtml(docEntry) : null;
      if (html) {
        const tmp = document.createElement('div');
        tmp.innerHTML = html;
        img.replaceWith(...tmp.childNodes);
        return;
      }
      // Remove truly orphaned CID images
      const parent = img.parentElement;
      img.remove();
      if (parent && parent !== bodyDiv && !parent.textContent.trim() && !parent.querySelector('img,video,iframe')) {
        parent.remove();
      }
    });

    // {{type:ID}} template placeholders — AI generates these for chart/table/mermaid refs
    const _placeholderRe = /\{\{(\w+):([a-f0-9-]+)\}\}/gi;
    if (_placeholderRe.test(bodyDiv.innerHTML)) {
      bodyDiv.innerHTML = bodyDiv.innerHTML.replace(/\{\{(\w+):([a-f0-9-]+)\}\}/gi, (_match, _type, refId) => {
        const docEntry = _refChatApp?.inlineDocBlocks?.find(b => b.id === refId);
        return (docEntry ? _inlineDocHtml(docEntry) : null) || '';
      });
    }
    _hydrateInlineCharts(bodyDiv);
    bodyHtml = bodyDiv.innerHTML; // sync back after cleanup
    bodyDiv.addEventListener('input', () => {
      if (_syncingBody) return;
      bodyHtml = bodyDiv.innerHTML;
      _notifyBlockUpdate();
    });
    const _richToolbar = _buildRichToolbar(bodyDiv, {
      onInput: () => { bodyHtml = bodyDiv.innerHTML; _notifyBlockUpdate(); },
    });
    bodyWrap.appendChild(_richToolbar);
    bodyWrap.appendChild(bodyDiv);
    _setupImageResize(bodyDiv);

    // AI editing features (desktop only)
    if (!_emailMobile && typeof _buildAIContextMenu === 'function') {
      const _aiOpts = {
        documentId: spec.id || blockId,
        documentType: 'email_draft',
        documentName: subject || 'Email Draft',
        editableEl: bodyDiv,
        onInput: () => { bodyHtml = bodyDiv.innerHTML; _notifyBlockUpdate(); },
      };
      _buildAIContextMenu(bodyDiv, _aiOpts);
      _buildAITaskButton(_richToolbar, _aiOpts);
      _setupGhostText(bodyDiv, _aiOpts);
    }

    wrap.appendChild(bodyWrap);

    /* ── Attachments bar (interactive) ────────────────── */
    const attBar = document.createElement('div');
    attBar.className = 'cv2-email-attachments';

    function _fmtSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1048576) return (bytes / 1024).toFixed(0) + ' KB';
      return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function _renderAttachments() {
      attBar.innerHTML = '';
      for (let i = 0; i < attachments.length; i++) {
        const a = attachments[i];
        const chip = document.createElement('span');
        chip.className = 'cv2-email-att-chip';
        const icon = a.type === 'ref' ? 'insert_link' : a.type === 'chat_file' ? 'description' : 'attach_file';
        const label = a.name || a.ref || 'Attachment';
        const sizeStr = a.size ? ` (${_fmtSize(a.size)})` : '';
        chip.innerHTML = `<span class="material-icons" style="font-size:14px;flex-shrink:0">${icon}</span>` +
          `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0">${_escHtml(label)}${sizeStr}</span>` +
          `<span class="cv2-email-att-remove" data-idx="${i}" title="Remove" style="flex-shrink:0">&times;</span>`;
        // Hover + click-to-pin preview
        const _att = a;
        chip.addEventListener('mouseenter', () => _showAttPreview(chip, _att, window.__chatApp?.sessionId));
        chip.addEventListener('mouseleave', _scheduleDismiss);
        chip.addEventListener('click', (e) => {
          if (e.target.closest('.cv2-email-att-remove')) return;
          _showAttPreviewPinned(chip, _att, window.__chatApp?.sessionId);
        });
        chip.querySelector('.cv2-email-att-remove').addEventListener('click', (e) => {
          e.stopPropagation();
          _dismissPreview();
          attachments.splice(parseInt(e.target.dataset.idx), 1);
          _renderAttachments();
          _notifyBlockUpdate();
        });
        attBar.appendChild(chip);
      }
      // "Attach" button
      const addBtn = document.createElement('button');
      addBtn.className = 'cv2-email-att-add-btn';
      addBtn.innerHTML = '<span class="material-icons" style="font-size:16px">add</span> Attach';
      addBtn.addEventListener('click', () => _showAttachMenu(addBtn));
      attBar.appendChild(addBtn);
    }

    // Hidden file input
    const _fileInput = document.createElement('input');
    _fileInput.type = 'file';
    _fileInput.multiple = true;
    _fileInput.style.display = 'none';
    _fileInput.accept = '.pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt,.csv,.png,.jpg,.jpeg,.gif,.zip';
    wrap.appendChild(_fileInput);

    _fileInput.addEventListener('change', () => {
      if (!_fileInput.files.length) return;
      for (const file of _fileInput.files) {
        if (file.size > 3 * 1024 * 1024) {
          // Graph API inline attachment limit ~3 MB
          const badge = header.querySelector('.cv2-email-draft-badge');
          if (badge) { badge.textContent = `${file.name} too large (max 3 MB)`; badge.className = 'cv2-email-draft-badge cv2-email-badge-error'; setTimeout(() => { badge.textContent = 'Draft'; badge.className = 'cv2-email-draft-badge'; }, 3000); }
          continue;
        }
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = reader.result.split(',')[1];
          attachments.push({
            type: 'file',
            name: file.name,
            content_type: file.type || 'application/octet-stream',
            data: base64,
            size: file.size,
          });
          _renderAttachments();
          _notifyBlockUpdate();
        };
        reader.readAsDataURL(file);
      }
      _fileInput.value = '';
    });

    function _showAttachMenu(anchor) {
      document.querySelectorAll('.cv2-email-att-menu').forEach(m => m.remove());
      const menu = document.createElement('div');
      menu.className = 'cv2-email-att-menu';
      const chatApp = window.__chatApp;

      // Upload file option
      const uploadItem = document.createElement('div');
      uploadItem.className = 'cv2-email-att-menu-item';
      uploadItem.innerHTML = '<span class="material-icons" style="font-size:16px">upload_file</span> Upload file';
      uploadItem.addEventListener('click', () => { menu.remove(); _fileInput.click(); });
      menu.appendChild(uploadItem);

      // ── Dynamic documents from chat (plotly, table, html, etc.) ──
      const allDocs = chatApp ? [...(chatApp.inlineDocBlocks || []), ...(chatApp.documents || [])] : [];
      const uniqueDocs = [];
      const seenIds = new Set();
      for (const d of allDocs) {
        if (!seenIds.has(d.id)) { seenIds.add(d.id); uniqueDocs.push(d); }
      }
      // Filter out self, email_draft docs, and already-attached
      const availDocs = uniqueDocs.filter(d =>
        d.id !== (spec.id || blockId) &&
        d.lang !== 'email_draft' &&
        !attachments.some(a => a.ref === d.id)
      );
      if (availDocs.length) {
        const sep = document.createElement('div');
        sep.className = 'cv2-email-att-menu-sep';
        sep.textContent = 'Documents';
        menu.appendChild(sep);
        for (const doc of availDocs) {
          const docItem = document.createElement('div');
          docItem.className = 'cv2-email-att-menu-item';
          const docIcon = (typeof ChatApp !== 'undefined' && ChatApp.DOC_ICONS?.[doc.lang]) || 'article';
          docItem.innerHTML = `<span class="material-icons" style="font-size:16px">${docIcon}</span> ${_escHtml(doc.name || doc.id.substring(0, 8))}`;
          docItem.addEventListener('click', () => {
            menu.remove();
            attachments.push({ type: 'ref', ref: doc.id, name: doc.name || doc.id.substring(0, 8), lang: doc.lang });
            _renderAttachments();
            _notifyBlockUpdate();
          });
          menu.appendChild(docItem);
        }
      }

      // ── Chat files (uploaded to session) ──
      const chatFiles = chatApp?._pendingFiles || [];
      const availFiles = chatFiles.filter(f => !attachments.some(a => a.fileId === f.fileId));
      if (availFiles.length) {
        const sep = document.createElement('div');
        sep.className = 'cv2-email-att-menu-sep';
        sep.textContent = 'Chat files';
        menu.appendChild(sep);
        for (const f of availFiles) {
          const fileItem = document.createElement('div');
          fileItem.className = 'cv2-email-att-menu-item';
          const fIcon = f.mimeType?.startsWith('image/') ? 'image' : f.mimeType?.includes('pdf') ? 'picture_as_pdf' : 'description';
          fileItem.innerHTML = `<span class="material-icons" style="font-size:16px">${fIcon}</span> ${_escHtml(f.name)}`;
          fileItem.addEventListener('click', () => {
            menu.remove();
            attachments.push({ type: 'chat_file', fileId: f.fileId, name: f.name, content_type: f.mimeType || 'application/octet-stream', size: f.size });
            _renderAttachments();
            _notifyBlockUpdate();
          });
          menu.appendChild(fileItem);
        }
      }

      // Position above button
      const rect = anchor.getBoundingClientRect();
      menu.style.left = rect.left + 'px';
      menu.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
      document.body.appendChild(menu);
      setTimeout(() => { document.addEventListener('click', () => menu.remove(), { once: true }); }, 50);
    }

    _renderAttachments();
    wrap.appendChild(attBar);

    /* ── Action buttons (Draft to Outlook / Send) ──────── */
    const actions = document.createElement('div');
    actions.className = 'cv2-email-draft-actions';

    const draftBtn = document.createElement('button');
    draftBtn.className = 'cv2-email-btn cv2-email-btn-draft';
    const sendBtn = document.createElement('button');
    sendBtn.className = 'cv2-email-btn cv2-email-btn-send';

    if (_emailState?.status === 'sent') {
      draftBtn.innerHTML = '<span class="material-icons" style="font-size:16px">drafts</span> Save as Draft';
      draftBtn.disabled = true;
      sendBtn.innerHTML = '<span class="material-icons" style="font-size:16px">check</span> Sent';
      sendBtn.classList.add('cv2-email-btn-success');
      sendBtn.disabled = true;
    } else if (_emailState?.status === 'draft') {
      draftBtn.innerHTML = '<span class="material-icons" style="font-size:16px">drafts</span> Update Draft';
      draftBtn.title = 'Update the existing draft in Outlook';
      sendBtn.innerHTML = '<span class="material-icons" style="font-size:16px">send</span> Send';
      sendBtn.title = 'Send email now';
    } else {
      draftBtn.innerHTML = '<span class="material-icons" style="font-size:16px">drafts</span> Save as Draft';
      draftBtn.title = 'Save as draft in Outlook';
      sendBtn.innerHTML = '<span class="material-icons" style="font-size:16px">send</span> Send';
      sendBtn.title = 'Send email now';
    }

    actions.appendChild(draftBtn);
    actions.appendChild(sendBtn);
    wrap.appendChild(actions);

    /* ── Export bar (source + sidepane) ─────────────────── */
    const exportBar = document.createElement('div');
    exportBar.className = 'cv2-doc-plugin-export';
    _addSourceButton(exportBar, rawData);
    _addWorkspaceButton(exportBar, blockId);
    wrap.appendChild(exportBar);

    container.appendChild(wrap);

    /* ── Wire draft/send buttons ───────────────────────── */
    // Track the server-side message ID (restored from localStorage or set after first action)
    let _messageId = _emailState?.message_id || null;

    function _notifyBlockUpdate() {
      // Update header title to reflect subject changes
      const titleEl = header.querySelector('.cv2-email-draft-title');
      if (titleEl) titleEl.textContent = subject || '(no subject)';
      // Sync block data store + sidebar
      const chatApp = window.__chatApp;
      // Strip base64 data from attachments for storage (kept in closure, sent on draft/send)
      const _storageAtts = attachments.map(a => a.data ? { type: a.type, name: a.name, content_type: a.content_type, size: a.size } : a);
      const updatedSpec = { ...spec, subject, body_html: bodyHtml, to: recipients.to, cc: recipients.cc, bcc: recipients.bcc, attachments: _storageAtts };
      const updatedRaw = JSON.stringify(updatedSpec);
      if (chatApp) {
        if (chatApp._blockDataStore && spec.id) {
          chatApp._blockDataStore.register(spec.id, 'email_draft', updatedSpec);
        }
        if (chatApp.inlineDocBlocks) {
          const idx = chatApp.inlineDocBlocks.findIndex(b => b.id === (spec.id || blockId));
          if (idx >= 0) {
            chatApp.inlineDocBlocks[idx].data = updatedRaw;
            chatApp.inlineDocBlocks[idx].name = subject || docName;
          }
        }
        /* Update preview title if this doc is currently shown in popover */
        const pvTitle = document.querySelector('.cv2-preview-popover .cv2-preview-title');
        if (pvTitle) pvTitle.textContent = subject || '(no subject)';
      }
      // Persist edits to localStorage (survives reload / conversation switch)
      try {
        localStorage.setItem(_editsKey, JSON.stringify({
          subject, body_html: bodyHtml,
          to: recipients.to, cc: recipients.cc, bcc: recipients.bcc,
          attachments: _storageAtts,
        }));
      } catch (_) {}
      // Dispatch sync event so other instances (workspace ↔ inline) update
      // Strip base64 data from attachments to avoid broadcasting large payloads
      const _syncAtts = attachments.map(a => a.data ? { ...a, data: undefined, _hasData: true } : a);
      document.dispatchEvent(new CustomEvent('cv2:email-draft-sync', {
        detail: { id: spec.id || blockId, sourceBlockId: blockId, subject, bodyHtml, recipients, attachments: _syncAtts },
      }));
    }

    function getEmailData() {
      return { subject, to: recipients.to, cc: recipients.cc, bcc: recipients.bcc, body_html: bodyHtml, attachments, id: spec.id, name: docName };
    }

    /** Resolve ref/chat_file attachments to actual file data before sending.
     *  Uses the _attachmentExporters plugin registry for native format export. */
    async function _resolveAttachments(atts) {
      const resolved = [];
      const chatApp = window.__chatApp;
      const store = chatApp?._blockDataStore;
      for (const att of atts) {
        // Already a file with data — keep
        if (att.type === 'file' && att.data) { resolved.push(att); continue; }
        // Chat file — pass through for server-side resolution
        if (att.type === 'chat_file') { resolved.push(att); continue; }
        // Ref to a dynamic document — resolve via export plugin
        const refId = att.ref;
        if (!refId) { resolved.push(att); continue; }
        const entry = store?.get(refId);
        if (!entry) {
          console.warn('[EMAIL] Cannot resolve ref:', refId);
          resolved.push(att);
          continue;
        }
        const exporter = _attachmentExporters[entry.lang];
        if (exporter) {
          try {
            const result = await exporter(entry, att.name || 'attachment');
            resolved.push({ type: 'file', ...result });
          } catch (err) {
            console.warn('[EMAIL] Export failed for', entry.lang, ':', err);
            // Fallback to JSON on export failure
            const json = JSON.stringify(entry.data, null, 2);
            const b64 = btoa(unescape(encodeURIComponent(json)));
            const name = (att.name || 'data').replace(/\.[^.]+$/, '') + '.json';
            resolved.push({ type: 'file', name, content_type: 'application/json', data: b64, size: json.length });
          }
        } else {
          // No exporter registered — serialize as JSON
          const json = JSON.stringify(entry.data, null, 2);
          const b64 = btoa(unescape(encodeURIComponent(json)));
          const name = (att.name || 'data').replace(/\.[^.]+$/, '') + '.json';
          resolved.push({ type: 'file', name, content_type: 'application/json', data: b64, size: json.length });
        }
      }
      return resolved;
    }

    async function emailAction(action) {
      const data = getEmailData();
      // If we already have a draft, use update_draft or send_draft
      let wsAction = action;
      if (action === 'draft' && _messageId) wsAction = 'update_draft';
      if (_messageId) data.message_id = _messageId;

      const btn = action === 'send' ? sendBtn : draftBtn;
      const origHtml = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<span class="material-icons cv2-spin" style="font-size:16px">hourglass_empty</span> ' + (action === 'send' ? 'Sending…' : 'Saving…');
      const chatApp = window.__chatApp;
      if (!chatApp || !chatApp.ws) {
        btn.disabled = false;
        btn.innerHTML = '<span class="material-icons" style="font-size:16px">error_outline</span> Not connected';
        btn.classList.add('cv2-email-btn-error');
        setTimeout(() => { btn.innerHTML = origHtml; btn.classList.remove('cv2-email-btn-error'); }, 3000);
        return;
      }
      // Convert inline charts in body to static PNG for email delivery
      if (data.body_html && data.body_html.includes('cv2-inline-chart')) {
        try {
          const tmpDiv = document.createElement('div');
          tmpDiv.innerHTML = data.body_html;
          data.body_html = await _convertInlineChartsToImages(tmpDiv);
        } catch (err) {
          console.warn('[EMAIL] Inline chart conversion failed:', err);
        }
      }
      // Resolve ref/chat_file attachments to actual file data before sending
      try {
        data.attachments = await _resolveAttachments(data.attachments || []);
      } catch (err) {
        console.warn('[EMAIL] Attachment resolution failed:', err);
      }
      console.log('[EMAIL] sending', wsAction, 'attachments:', (data.attachments || []).length,
        (data.attachments || []).map(a => ({ type: a.type, name: a.name, hasData: !!a.data, dataLen: a.data ? a.data.length : 0 })));
      chatApp.ws.send({ type: 'email:' + wsAction, ...data });
      document.addEventListener('ws:email:action_result', function _onResult(ev) {
        const result = ev.detail;
        if (result.action !== wsAction) {
          // Not our response — re-listen
          document.addEventListener('ws:email:action_result', _onResult, { once: true });
          return;
        }
        if (result.ok) {
          // Store message_id from server for future update/send
          if (result.message_id) _messageId = result.message_id;
          const badge = header.querySelector('.cv2-email-draft-badge');
          if (action === 'draft') {
            // Persist draft state
            try { localStorage.setItem(_stateKey, JSON.stringify({ message_id: _messageId, status: 'draft', web_link: result.web_link || '' })); } catch (_) {}
            if (badge) { badge.textContent = 'Draft Saved'; badge.className = 'cv2-email-draft-badge cv2-email-badge-saved'; }
            btn.innerHTML = '<span class="material-icons" style="font-size:16px">check</span> Saved!';
            btn.classList.add('cv2-email-btn-success');
            // Re-enable after a moment so user can update the draft again
            setTimeout(() => { btn.disabled = false; btn.innerHTML = '<span class="material-icons" style="font-size:16px">drafts</span> Update Draft'; btn.classList.remove('cv2-email-btn-success'); }, 2000);
          } else {
            // Persist sent state
            try { localStorage.setItem(_stateKey, JSON.stringify({ message_id: _messageId || result.message_id || '', status: 'sent' })); } catch (_) {}
            if (badge) { badge.textContent = 'Sent'; badge.className = 'cv2-email-draft-badge cv2-email-badge-sent'; }
            btn.innerHTML = '<span class="material-icons" style="font-size:16px">check</span> Sent!';
            btn.classList.add('cv2-email-btn-success');
            sendBtn.disabled = true;
            draftBtn.disabled = true;
          }
        } else {
          btn.disabled = false;
          btn.innerHTML = '<span class="material-icons" style="font-size:16px">error_outline</span> ' + _escHtml(result.error || 'Failed');
          btn.classList.add('cv2-email-btn-error');
          setTimeout(() => { btn.innerHTML = origHtml; btn.classList.remove('cv2-email-btn-error'); }, 3000);
        }
      }, { once: true });
    }

    // Easter egg: shift-double-click a disabled button to unlock
    // (disabled buttons swallow click/dblclick, so use pointerdown which still fires)
    let _unlockClicks = 0, _unlockTimer = null, _justUnlocked = false;
    actions.addEventListener('pointerdown', (e) => {
      if (!e.shiftKey) { _unlockClicks = 0; return; }
      const btn = e.target.closest('.cv2-email-btn');
      if (!btn || !btn.disabled) { _unlockClicks = 0; return; }
      _unlockClicks++;
      clearTimeout(_unlockTimer);
      if (_unlockClicks >= 2) {
        _unlockClicks = 0;
        _justUnlocked = true;
        sendBtn.disabled = false;
        draftBtn.disabled = false;
        sendBtn.classList.remove('cv2-email-btn-success', 'cv2-email-btn-error');
        sendBtn.innerHTML = '<span class="material-icons" style="font-size:16px">send</span> Send';
        draftBtn.innerHTML = '<span class="material-icons" style="font-size:16px">drafts</span> Save as Draft';
        const badge = header.querySelector('.cv2-email-draft-badge');
        if (badge) { badge.textContent = 'Draft'; badge.className = 'cv2-email-draft-badge'; }
        localStorage.removeItem(_stateKey);
        // Swallow the click that follows this pointerdown
        setTimeout(() => { _justUnlocked = false; }, 300);
      } else {
        _unlockTimer = setTimeout(() => { _unlockClicks = 0; }, 500);
      }
    });
    draftBtn.addEventListener('click', () => { if (!_justUnlocked) emailAction('draft'); });
    sendBtn.addEventListener('click', () => { if (!_justUnlocked) emailAction('send'); });

    /* ── Sync across instances (inline ↔ workspace) ──── */
    const _onExternalSync = (ev) => {
      if (!wrap.isConnected) { document.removeEventListener('cv2:email-draft-sync', _onExternalSync); return; }
      const d = ev.detail;
      if (d.id !== (spec.id || blockId) || d.sourceBlockId === blockId) return;
      // Update local state from the other instance
      if (d.subject !== undefined && d.subject !== subject) {
        subject = d.subject;
        _syncingSubject = true;
        subjectSpan.textContent = subject;
        _syncingSubject = false;
        const titleEl = header.querySelector('.cv2-email-draft-title');
        if (titleEl) titleEl.textContent = subject || '(no subject)';
      }
      if (d.bodyHtml !== undefined && d.bodyHtml !== bodyHtml) {
        bodyHtml = d.bodyHtml;
        _syncingBody = true;
        const clean = (typeof DOMPurify !== 'undefined') ? DOMPurify.sanitize(bodyHtml) : bodyHtml;
        bodyDiv.innerHTML = clean || '<p><br></p>';
        _hydrateInlineCharts(bodyDiv);
        _syncingBody = false;
      }
      if (d.recipients) {
        recipients.to = d.recipients.to || [];
        recipients.cc = d.recipients.cc || [];
        recipients.bcc = d.recipients.bcc || [];
      }
      if (d.attachments) {
        // Merge: keep local base64 data, adopt metadata changes from the other instance
        const localByName = new Map(attachments.filter(a => a.data).map(a => [a.name, a]));
        attachments.length = 0;
        for (const a of d.attachments) {
          if (a._hasData && localByName.has(a.name)) {
            attachments.push(localByName.get(a.name)); // Preserve local base64 data
          } else {
            attachments.push(a);
          }
        }
        _renderAttachments();
      }
    };
    document.addEventListener('cv2:email-draft-sync', _onExternalSync);

    /* ── Open in window on header click ─────────────── */
    header.style.cursor = 'pointer';
    header.addEventListener('click', () => _showDocWindowed(blockId));
  },
};

/* ══════════════════════════════════════════════════════════════
   Rich MCP Plugin — renders stored visualizations by UUID
   ══════════════════════════════════════════════════════════════ */

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

    if (noMenu) {
      html += '<div class="kvi-no-menu">Kein Speiseplan für diese Woche verfügbar</div>';
    } else {
      html += '<div class="kvi-meals">';
      for (const m of meals) {
        const n = { kcal: m.kcal, carbs: m.carbs_g, protein: m.protein_g, fat: m.fat_g };
        const tagHtml = (m.tags || []).map(t => _tagIcons[t] ? `<span class="kvi-tag">${_tagIcons[t]}</span>` : '').join('');

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
        html += '</div></div>';
      }
      html += '</div>';
    }
    html += '</div>';
    container.innerHTML = html;

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
  registry.register('plotly', plotlyPlugin);
  registry.register('latex', latexPlugin);
  registry.register('table', tablePlugin);
  registry.register('text_doc', textDocPlugin);
  registry.register('word', textDocPlugin);  // backward compat alias
  registry.register('presentation', presentationPlugin);
  registry.register('powerpoint', presentationPlugin);  // backward compat alias
  registry.register('pptx', presentationPlugin);
  registry.register('html_sandbox', htmlPlugin);
  registry.register('email_draft', emailDraftPlugin);
  registry.register('mermaid', mermaidPlugin);
  registry.register('rich_mcp', richMcpPlugin);
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
