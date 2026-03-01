/**
 * MarkdownRenderer — marked + DOMPurify + KaTeX + doc plugin blocks
 *
 * Depends on: marked.js, DOMPurify, KaTeX (loaded as vendor scripts)
 * Optionally uses: DocPluginRegistry (window.DocPluginRegistry)
 *
 * Extracted from chat-app.js.
 */

class MarkdownRenderer {
  constructor(pluginRegistry) {
    /** @type {DocPluginRegistry|null} */
    this._pluginRegistry = pluginRegistry || null;
    this._pendingBlocks = [];
    this._blockCounter = 0;
    /** When true, plugin blocks show a spinner instead of rendering. */
    this.streaming = false;

    if (window.marked) {
      window.marked.setOptions({
        breaks: true,
        gfm: true,
      });

      // Custom renderer to intercept plugin code blocks
      if (this._pluginRegistry) {
        const renderer = new window.marked.Renderer();
        const self = this;
        const originalCode = renderer.code?.bind(renderer);
        renderer.code = function({ text, lang }) {
          if (lang && self._pluginRegistry.has(lang)) {
            // During streaming, show a spinner placeholder — don't render or register
            if (self.streaming) {
              let name = lang;
              try {
                const m = text.match(/"(?:name|title)"\s*:\s*"([^"]+)"/);
                if (m) name = m[1];
              } catch (_) {}
              const esc = name.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
              return `<div class="cv2-doc-plugin-streaming"><div class="cv2-spinner-dots"><span></span><span></span><span></span></div><span class="cv2-doc-plugin-streaming-name">${esc}</span></div>`;
            }
            const blockId = `dp-${++self._blockCounter}-${Date.now()}`;
            self._pendingBlocks.push({ blockId, lang, data: text });
            return `<div class="cv2-doc-plugin-block" data-block-id="${blockId}" data-lang="${lang}"></div>`;
          }
          // Default code block rendering
          const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
          return `<pre><code${lang ? ` class="language-${lang}"` : ''}>${escaped}</code></pre>`;
        };
        // Links open in new tab
        renderer.link = function({ href, title, text }) {
          const t = title ? ` title="${title}"` : '';
          return `<a href="${href}" target="_blank" rel="noopener noreferrer"${t}>${text}</a>`;
        };
        window.marked.use({ renderer });
      }
    }
  }

  render(text) {
    if (!text) return '';

    // Reset pending blocks for this render pass
    this._pendingBlocks = [];

    let processText = text;
    let streamingSpinner = '';

    // During streaming, strip any incomplete (unclosed) fenced plugin block at the end
    if (this.streaming && this._pluginRegistry) {
      const result = this._stripIncompletePluginBlock(processText);
      processText = result.text;
      streamingSpinner = result.spinner;
    }

    // Pre-process LaTeX before markdown parsing
    const processed = this._protectLatex(processText);
    let html = window.marked ? window.marked.parse(processed) : this._basicMarkdown(processed);

    // Restore and render LaTeX
    html = this._renderLatex(html);

    // Sanitize
    if (window.DOMPurify) {
      html = window.DOMPurify.sanitize(html, {
        ADD_TAGS: ['span', 'div'],
        ADD_ATTR: ['class', 'style', 'data-block-id', 'data-lang', 'target', 'rel'],
      });
    }

    // Append streaming spinner (after sanitize — this is our own trusted HTML)
    if (streamingSpinner) html += streamingSpinner;

    return html;
  }

  /**
   * Detect an incomplete (unclosed) fenced code block for a plugin language
   * at the end of streaming text, and replace it with a spinner placeholder.
   */
  _stripIncompletePluginBlock(text) {
    if (!this._pluginRegistry) return { text, spinner: '' };
    const langs = this._pluginRegistry.languages;
    // Build regex: opening fence for any plugin lang, followed by everything to end, with no closing fence
    for (const lang of langs) {
      const opener = '```' + lang;
      const lastIdx = text.lastIndexOf(opener);
      if (lastIdx === -1) continue;
      // Must be at start of line (or start of text)
      if (lastIdx > 0 && text[lastIdx - 1] !== '\n') continue;
      const afterOpener = text.substring(lastIdx + opener.length);
      // Check for a closing fence (``` on its own line)
      if (/\n```\s*(\n|$)/.test(afterOpener)) continue;
      // Incomplete block — extract name and show spinner
      let name = lang;
      try {
        const m = afterOpener.match(/"(?:name|title)"\s*:\s*"([^"]+)"/);
        if (m) name = m[1];
      } catch (_) {}
      const esc = name.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return {
        text: text.substring(0, lastIdx),
        spinner: `<div class="cv2-doc-plugin-streaming"><div class="cv2-spinner-dots"><span></span><span></span><span></span></div><span class="cv2-doc-plugin-streaming-name">${esc}</span></div>`,
      };
    }
    return { text, spinner: '' };
  }

  /** Hydrate plugin placeholder divs within a parent element. */
  async hydratePluginBlocks(parentEl) {
    if (!this._pluginRegistry || !parentEl) return;
    const placeholders = parentEl.querySelectorAll('.cv2-doc-plugin-block[data-block-id]');
    for (const el of placeholders) {
      if (el.dataset.hydrated) continue;
      const blockId = el.dataset.blockId;
      const lang = el.dataset.lang;
      const pending = this._pendingBlocks.find(b => b.blockId === blockId);
      if (!pending) continue;
      el.dataset.hydrated = '1';
      try {
        await this._pluginRegistry.render(lang, el, pending.data, blockId);
      } catch (err) {
        console.warn(`[DocPlugin] Failed to hydrate ${lang} block ${blockId}:`, err);
        el.innerHTML = `<pre class="cv2-doc-plugin-error">Error rendering ${lang}: ${err.message}</pre>`;
      }
    }
  }

  _protectLatex(text) {
    // Replace $$ ... $$ and $ ... $ with placeholders
    let idx = 0;
    this._latexBlocks = [];
    // Display math $$...$$
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, expr) => {
      this._latexBlocks.push({ expr, display: true });
      return `%%LATEX_${idx++}%%`;
    });
    // Inline math $...$
    text = text.replace(/\$([^\n$]+?)\$/g, (_, expr) => {
      this._latexBlocks.push({ expr, display: false });
      return `%%LATEX_${idx++}%%`;
    });
    return text;
  }

  _renderLatex(html) {
    if (!this._latexBlocks || !window.katex) return html;
    for (let i = 0; i < this._latexBlocks.length; i++) {
      const { expr, display } = this._latexBlocks[i];
      try {
        const rendered = window.katex.renderToString(expr.trim(), {
          displayMode: display,
          throwOnError: false,
        });
        html = html.replace(`%%LATEX_${i}%%`, rendered);
      } catch {
        html = html.replace(`%%LATEX_${i}%%`, display ? `$$${expr}$$` : `$${expr}$`);
      }
    }
    return html;
  }

  _basicMarkdown(text) {
    // Fallback if marked.js isn't loaded
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>');
  }
}
