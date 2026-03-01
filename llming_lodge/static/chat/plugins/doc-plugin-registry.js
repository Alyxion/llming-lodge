/**
 * DocPluginRegistry — manages document type plugins for chat rendering.
 *
 * Each plugin registers a language identifier (e.g. "plotly", "latex", "table")
 * and provides a render function that converts JSON/text data into DOM content.
 */

class DocPluginRegistry {
  constructor() {
    /** @type {Map<string, {render: Function, inline: boolean, loaded: boolean, loader?: Function}>} */
    this._plugins = new Map();
    /** @type {BlockDataStore|null} */
    this._blockStore = null;
    /** @type {Function|null} Callback fired after a block renders successfully */
    this._onBlockRendered = null;
    /** @type {Function[]} Inline pattern renderers — called after each message render */
    this._inlineRenderers = [];
  }

  /** Register a callback invoked after each successful block render.
   *  @param {Function} fn — receives {id, lang, name, data, blockId, element}
   */
  onBlockRendered(fn) { this._onBlockRendered = fn; }

  /** Wire an external BlockDataStore for cross-block references. */
  setBlockStore(store) {
    this._blockStore = store;
  }

  /**
   * Register a document plugin.
   * @param {string} lang — fenced code block language id (e.g. "plotly")
   * @param {object} opts
   * @param {Function} opts.render — async (container: HTMLElement, data: string, blockId: string) => void
   * @param {boolean} [opts.inline=true] — render inline in chat vs. create document
   * @param {boolean} [opts.sidebar=true] — track in documents sidebar (false for ephemeral inline blocks)
   * @param {Function} [opts.loader] — async function to load external libs before first render
   */
  register(lang, { render, inline = true, sidebar = true, loader }) {
    this._plugins.set(lang, { render, inline, sidebar, loaded: !loader, loader });
  }

  /** Check if a language has a registered plugin. */
  has(lang) {
    return this._plugins.has(lang);
  }

  /** Check if the plugin renders inline (vs. as a document panel item). */
  isInline(lang) {
    const p = this._plugins.get(lang);
    return p ? p.inline : false;
  }

  /** Ensure the plugin's external dependencies are loaded. */
  async ensureLoaded(lang) {
    const p = this._plugins.get(lang);
    if (!p || p.loaded) return;
    // Deduplicate concurrent calls — share a single loader promise
    if (p._loadingPromise) {
      await p._loadingPromise;
      return;
    }
    if (p.loader) {
      p._loadingPromise = p.loader();
      await p._loadingPromise;
      p._loadingPromise = null;
    }
    p.loaded = true;
  }

  /**
   * Render a plugin block into a container element.
   * @param {string} lang
   * @param {HTMLElement} container
   * @param {string} rawData — the raw text content from the fenced code block
   * @param {string} blockId — unique id for this block instance
   */
  async render(lang, container, rawData, blockId) {
    const p = this._plugins.get(lang);
    if (!p) return;
    await this.ensureLoaded(lang);

    // ── Cross-block reference resolution ──────────────────
    let data = rawData;
    if (this._blockStore) {
      try {
        const parsed = JSON.parse(rawData);
        if (parsed && typeof parsed === 'object') {
          // Auto-assign a UUID if the block doesn't have one
          if (!parsed.id) {
            parsed.id = crypto.randomUUID ? crypto.randomUUID() : (
              'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
                const r = Math.random() * 16 | 0;
                return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
              })
            );
          }
          // Register in block store for $ref resolution
          this._blockStore.register(parsed.id, lang, parsed);
        }
        // Resolve $ref pointers
        const resolved = window.resolveBlockRefs
          ? window.resolveBlockRefs(parsed, this._blockStore)
          : parsed;
        // Apply cross-type compatibility aliases
        const compat = window.applyCrossTypeCompat
          ? window.applyCrossTypeCompat(lang, resolved)
          : resolved;
        data = JSON.stringify(compat);
      } catch (_) {
        // Not valid JSON — pass through as-is (e.g. latex)
      }
    }

    try {
      await p.render(container, data, blockId);

      // Notify listener with block metadata (skip non-sidebar plugins like contact cards)
      if (this._onBlockRendered && p.sidebar !== false) {
        let name = null, id = null;
        try {
          const obj = JSON.parse(rawData);
          id = obj?.id || null;
          name = obj?.name || obj?.title || null;
        } catch (_) {}
        this._onBlockRendered({
          id: id || blockId,
          lang,
          name: name || `${lang} ${(id || blockId).substring(0, 6)}`,
          data: rawData,
          blockId,
          element: container,
        });
      }
    } catch (err) {
      console.error(`[DocPlugin:${lang}] Render error:`, err);
      container.innerHTML = `<pre class="cv2-doc-plugin-error">Error rendering ${lang}: ${err.message}</pre>`;
    }
  }

  /**
   * Register an inline pattern renderer.
   * @param {Function} fn — async (container: HTMLElement) => void
   */
  registerInline(fn) {
    this._inlineRenderers.push(fn);
  }

  /**
   * Run all inline pattern renderers on a container.
   * Called after each message render to enhance inline content.
   * @param {HTMLElement} container
   */
  async runInlineRenderers(container) {
    for (const fn of this._inlineRenderers) {
      try { await fn(container); } catch (e) { console.error('[InlineRenderer]', e); }
    }
  }

  /** Get all registered language ids. */
  get languages() {
    return [...this._plugins.keys()];
  }
}

// Export for global access
window.DocPluginRegistry = DocPluginRegistry;
