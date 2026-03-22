/* ── chat-app-extensions.js — App Extension system (lazy load + WS bridge) ── */
/* Feature module: manages registration, on-demand script loading, activation,
   and bidirectional WebSocket messaging for app extensions.

   Server sends available extension manifests in session_init.  Scripts are NOT
   loaded until activation is requested — either by the user or programmatically.

   Extension authors define their client-side counterpart via:

       ChatAppExtensions.define('my_ext', {
         // Called once after script load + server activation.  `config` is the
         // dict returned by the Python AppExtension.on_activate().
         activate(app, config) { ... },
         // Called for every server→client `app_ext:message` with this ext name.
         handleMessage(app, payload) { ... },
         // Called on deactivation — tear down UI, stop timers, etc.
         deactivate(app) { ... },
       });
*/

(function () {
  'use strict';

  // ── Client-side extension definitions ─────────────────────
  // Populated by extension scripts calling ChatAppExtensions.define().
  const _definitions = {};  // name → { activate, handleMessage, deactivate }

  // ── Per-app-instance state (set during initState) ─────────
  // Stored on the ChatApp instance as this._appExt.
  // {
  //   manifests: [],           // from server session_init
  //   active: Set<string>,     // currently activated names
  //   loading: Set<string>,    // scripts currently being fetched
  //   loaded: Set<string>,     // scripts that have been loaded
  // }

  // ── Public API: ChatAppExtensions ─────────────────────────

  const ChatAppExtensions = {
    /**
     * Define a client-side extension.  Called by extension scripts.
     * @param {string} name — must match the server-side AppExtension.name
     * @param {object} descriptor — { activate, handleMessage, deactivate }
     */
    define(name, descriptor) {
      if (_definitions[name]) {
        console.warn(`[AppExt] Redefining extension "${name}"`);
      }
      _definitions[name] = descriptor;
      // If any app instance is waiting for this definition, resolve it
      if (_pendingDefResolvers[name]) {
        _pendingDefResolvers[name]();
        delete _pendingDefResolvers[name];
      }
    },

    /** Check if a definition exists for a given name. */
    has(name) {
      return name in _definitions;
    },

    /** Get a definition by name. */
    get(name) {
      return _definitions[name] || null;
    },
  };

  // Resolvers for scripts that call define() after load
  const _pendingDefResolvers = {};

  window.ChatAppExtensions = ChatAppExtensions;

  // ── Script loader (deduplicating) ─────────────────────────

  const _scriptCache = {};  // url → Promise<void>

  function _loadScript(url) {
    if (_scriptCache[url]) return _scriptCache[url];
    _scriptCache[url] = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = url;
      s.onload = resolve;
      s.onerror = () => reject(new Error(`[AppExt] Failed to load ${url}`));
      document.head.appendChild(s);
    });
    return _scriptCache[url];
  }

  /**
   * Load a script and wait for the extension to call define(name).
   * If already defined (e.g. bundled), resolves immediately.
   */
  function _loadAndWaitForDefine(name, url) {
    if (_definitions[name]) return Promise.resolve();
    if (!url) return Promise.reject(new Error(`[AppExt] No script URL for "${name}"`));
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        delete _pendingDefResolvers[name];
        reject(new Error(`[AppExt] Timeout waiting for define("${name}") from ${url}`));
      }, 10000);
      _pendingDefResolvers[name] = () => {
        clearTimeout(timeout);
        resolve();
      };
      // If already defined by the time script loads (race), resolver fires
      _loadScript(url).catch(err => {
        clearTimeout(timeout);
        delete _pendingDefResolvers[name];
        reject(err);
      });
    });
  }

  // ── ChatApp integration (prototype mixin) ─────────────────

  Object.assign(window._ChatAppProto, {

    /**
     * Activate an extension by name.  Sends activation request to server,
     * loads the client script on demand, and calls the descriptor's activate().
     * @param {string} name
     * @returns {Promise<boolean>} true if activated successfully
     */
    async appExtActivate(name) {
      const state = this._appExt;
      if (!state) return false;
      if (state.active.has(name)) return true;
      if (state.loading.has(name)) return false;

      const manifest = state.manifests.find(m => m.name === name);
      if (!manifest) {
        console.warn(`[AppExt] Unknown extension "${name}"`);
        return false;
      }

      state.loading.add(name);

      // 1. Load client script (if needed) in parallel with server activation
      const scriptPromise = state.loaded.has(name)
        ? Promise.resolve()
        : _loadAndWaitForDefine(name, manifest.scriptUrl)
            .then(() => { state.loaded.add(name); });

      // 2. Request server-side activation
      const configPromise = new Promise((resolve) => {
        this._appExtPendingActivations[name] = resolve;
        this.ws.send({ type: 'app_ext:activate', name });
        // Timeout: if server doesn't respond in 10s, resolve with null
        setTimeout(() => {
          if (this._appExtPendingActivations[name]) {
            delete this._appExtPendingActivations[name];
            resolve(null);
          }
        }, 10000);
      });

      try {
        const [, config] = await Promise.all([scriptPromise, configPromise]);
        state.loading.delete(name);

        const def = _definitions[name];
        if (!def) {
          console.error(`[AppExt] No client definition for "${name}" after script load`);
          return false;
        }

        state.active.add(name);
        if (def.activate) {
          try {
            await def.activate(this, config || {});
          } catch (e) {
            console.error(`[AppExt] activate("${name}") threw:`, e);
          }
        }
        return true;
      } catch (err) {
        state.loading.delete(name);
        console.error(`[AppExt] Activation failed for "${name}":`, err);
        return false;
      }
    },

    /**
     * Deactivate an extension by name.
     * @param {string} name
     */
    async appExtDeactivate(name) {
      const state = this._appExt;
      if (!state || !state.active.has(name)) return;

      const def = _definitions[name];
      if (def && def.deactivate) {
        try {
          await def.deactivate(this);
        } catch (e) {
          console.error(`[AppExt] deactivate("${name}") threw:`, e);
        }
      }
      state.active.delete(name);
      this.ws.send({ type: 'app_ext:deactivate', name });
    },

    /**
     * Send a message to a server-side extension.
     * @param {string} name — extension name
     * @param {object} payload — arbitrary data
     */
    appExtSend(name, payload) {
      this.ws.send({ type: 'app_ext:message', name, payload });
    },

    /**
     * Check if an extension is currently active.
     * @param {string} name
     * @returns {boolean}
     */
    appExtIsActive(name) {
      return this._appExt ? this._appExt.active.has(name) : false;
    },

    /**
     * Get the list of available extension manifests.
     * @returns {Array<{name, label, icon, scriptUrl}>}
     */
    appExtList() {
      return this._appExt ? this._appExt.manifests : [];
    },

    // ── Internal WS handlers ──────────────────────────────

    /** Handle server confirmation of activation (carries config). */
    _appExtHandleActivated(msg) {
      const resolve = this._appExtPendingActivations[msg.name];
      if (resolve) {
        delete this._appExtPendingActivations[msg.name];
        resolve(msg.config || {});
      }
    },

    /** Handle server→client extension message. */
    _appExtHandleMessage(msg) {
      const def = _definitions[msg.name];
      if (def && def.handleMessage) {
        try {
          def.handleMessage(this, msg.payload || {});
        } catch (e) {
          console.error(`[AppExt] handleMessage("${msg.name}") threw:`, e);
        }
      }
    },

    /** Handle server-initiated deactivation. */
    _appExtHandleDeactivated(msg) {
      const state = this._appExt;
      if (!state) return;
      const name = msg.name;
      if (state.active.has(name)) {
        const def = _definitions[name];
        if (def && def.deactivate) {
          try { def.deactivate(this); } catch (_) {}
        }
        state.active.delete(name);
      }
    },

    /** Deactivate all active extensions (called on disconnect/cleanup). */
    _appExtDeactivateAll() {
      const state = this._appExt;
      if (!state) return;
      for (const name of state.active) {
        const def = _definitions[name];
        if (def && def.deactivate) {
          try { def.deactivate(this); } catch (_) {}
        }
      }
      state.active.clear();
    },
  });

  // ── Feature registration ──────────────────────────────────

  ChatFeatures.register('app_extensions', {
    initState(app) {
      app._appExt = {
        manifests: [],
        active: new Set(),
        loading: new Set(),
        loaded: new Set(),
      };
      app._appExtPendingActivations = {};  // name → resolve callback
    },

    bindEvents(app) {
      // Render extension toggle buttons in the topbar (before workspace toggle)
      const topbar = document.querySelector('.cv2-topbar');
      const wsToggle = document.getElementById('cv2-workspace-toggle');
      if (!topbar || !app._appExt) return;
      const manifests = app._appExt.manifests || [];
      for (const m of manifests) {
        const btn = document.createElement('button');
        btn.className = 'cv2-topbar-btn cv2-ext-toggle';
        btn.title = m.label || m.name;
        btn.dataset.extName = m.name;
        btn.innerHTML = `<span class="material-icons">${m.icon || 'extension'}</span>`;
        btn.addEventListener('click', () => {
          if (app.appExtIsActive(m.name)) {
            app.appExtDeactivate(m.name);
          } else {
            app.appExtActivate(m.name);
          }
        });
        if (wsToggle) topbar.insertBefore(btn, wsToggle);
        else topbar.appendChild(btn);
      }

      // Register bolt slash commands from extension manifests (available before activation)
      if (app._boltsRegister) {
        for (const m of manifests) {
          if (m.bolts && m.bolts.length) {
            app._boltsRegister('__ext_' + m.name, m.bolts);
          }
        }
      }
    },

    handleMessage: {
      'app_ext:activated': '_appExtHandleActivated',
      'app_ext:message': '_appExtHandleMessage',
      'app_ext:deactivated': '_appExtHandleDeactivated',
    },
  });

})();
