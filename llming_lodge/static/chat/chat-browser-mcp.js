/**
 * chat-browser-mcp.js — Browser-hosted MCP servers via Web Workers
 *
 * Receives MCP nudge JS files from the server, spawns Web Workers,
 * and proxies tool calls between the server (WebSocket) and Workers.
 */
(function() {

  Object.assign(window._ChatAppProto, {

    /**
     * Handle start_browser_mcp: spawn a Web Worker from nudge JS files.
     * @param {Object} msg - {request_id, nudge_uid, entry_point, files: {name: source}, data_files: [...]}
     */
    _handleStartBrowserMcp(msg) {
      const { request_id, nudge_uid, entry_point, files, data_files } = msg;

      try {
        const workerCode = this._buildWorkerCode(files, entry_point, data_files || []);
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        const worker = new Worker(url);

        // Store worker reference
        if (!this._mcpWorkers) this._mcpWorkers = {};
        this._mcpWorkers[nudge_uid] = { worker, blobUrl: url, pendingCalls: {} };

        this._initMcpWorker(worker, nudge_uid, request_id);
      } catch (err) {
        console.error('[BROWSER_MCP] Failed to create Worker:', err);
        this.ws.send({
          type: 'browser_mcp_result',
          request_id,
          error: `Worker creation failed: ${err.message}`,
        });
      }
    },

    /**
     * Handle browser_mcp_call: forward tool call to the Worker.
     * @param {Object} msg - {request_id, nudge_uid, tool_name, arguments}
     */
    _handleBrowserMcpCall(msg) {
      const { request_id, nudge_uid, tool_name, arguments: args } = msg;
      const entry = (this._mcpWorkers || {})[nudge_uid];
      if (!entry) {
        this.ws.send({
          type: 'browser_mcp_result',
          request_id,
          error: `No active Worker for nudge ${nudge_uid}`,
        });
        return;
      }

      this._callWorkerTool(entry, request_id, tool_name, args);
    },

    /**
     * Handle stop_browser_mcp: terminate a Worker.
     * @param {Object} msg - {nudge_uid}
     */
    _handleStopBrowserMcp(msg) {
      const { nudge_uid } = msg;
      const entry = (this._mcpWorkers || {})[nudge_uid];
      if (entry) {
        entry.worker.terminate();
        URL.revokeObjectURL(entry.blobUrl);
        delete this._mcpWorkers[nudge_uid];
        console.log(`[BROWSER_MCP] Stopped Worker for nudge ${nudge_uid}`);
      }
    },

    /**
     * Build Worker-compatible script from nudge JS files.
     *
     * Strategy: strip ES module imports/exports from business logic files,
     * wrap each in a namespace, and prepend a Worker shim that dispatches
     * postMessage-based tool calls.
     */
    _buildWorkerCode(files, entryPoint, dataFiles) {
      const parts = [];

      // -- Worker shim preamble --
      // Serialize data files into the Worker so tool handlers can access them
      const dataFilesJson = JSON.stringify((dataFiles || []).map(f => ({
        name: f.name,
        mimeType: f.mime_type || '',
        size: f.size || 0,
        base64: f.content_base64 || '',
        text: f.text_content || '',
      })));

      parts.push(`
// === MCP Worker Sandbox ===
// Block ALL network, storage, and cross-origin APIs before any user code runs.
// This runs as the very first thing in the Worker — cannot be circumvented by
// later code because the references are replaced AND frozen on all global aliases.
(function() {
  "use strict";
  const _blocked = (name) => { const f = () => { throw new Error('[SANDBOX] ' + name + ' is not allowed in MCP Workers'); }; f.toString = () => 'function ' + name + '() { [sandbox blocked] }'; return f; };
  const _blockedCtor = (name) => { const C = function() { throw new Error('[SANDBOX] ' + name + ' is not allowed in MCP Workers'); }; C.prototype = {}; C.toString = () => 'function ' + name + '() { [sandbox blocked] }'; return C; };

  // All global aliases in a Worker (self, globalThis, and the implicit global)
  const _globals = [self];
  if (typeof globalThis !== 'undefined' && globalThis !== self) _globals.push(globalThis);

  // ── Network APIs ──────────────────────────────────────
  const _netBlocks = {
    fetch:          _blocked('fetch'),
    XMLHttpRequest: _blockedCtor('XMLHttpRequest'),
    WebSocket:      _blockedCtor('WebSocket'),
    EventSource:    _blockedCtor('EventSource'),
    Request:        _blockedCtor('Request'),
    Response:       _blockedCtor('Response'),
    Headers:        _blockedCtor('Headers'),
  };

  // ── Script loading ────────────────────────────────────
  const _scriptBlocks = {
    importScripts: _blocked('importScripts'),
  };

  // ── Cross-context communication ───────────────────────
  const _commBlocks = {
    BroadcastChannel: _blockedCtor('BroadcastChannel'),
  };

  // Apply all blocks to every global alias
  const _allBlocks = { ..._netBlocks, ..._scriptBlocks, ..._commBlocks };
  for (const g of _globals) {
    for (const [prop, replacement] of Object.entries(_allBlocks)) {
      try { g[prop] = replacement; } catch(e) {}
    }
    // Wipe storage APIs
    try { g.indexedDB = undefined; } catch(e) {}
    try { g.caches = undefined; } catch(e) {}
    // Block navigator.sendBeacon
    if (g.navigator) {
      try { g.navigator.sendBeacon = _blocked('navigator.sendBeacon'); } catch(e) {}
      try { Object.defineProperty(g.navigator, 'sendBeacon', { value: _blocked('navigator.sendBeacon'), writable: false, configurable: false }); } catch(e) {}
    }
  }

  // ── Freeze all blocked properties so user code cannot re-assign them ──
  const _allProps = [...Object.keys(_allBlocks), 'indexedDB', 'caches'];
  for (const g of _globals) {
    for (const prop of _allProps) {
      try {
        Object.defineProperty(g, prop, {
          get: () => (typeof _allBlocks[prop] === 'function' ? _allBlocks[prop] : undefined),
          set: () => { throw new Error('[SANDBOX] Cannot re-enable ' + prop); },
          configurable: false,
        });
      } catch(e) {
        // Fallback: at least make it non-writable (defineProperty with value)
        try { Object.defineProperty(g, prop, { value: _allBlocks[prop] || undefined, writable: false, configurable: false }); } catch(e2) {}
      }
    }
  }
})();

// === MCP Worker Shim ===
const _tools = {};
self._registerMcpTool = (name, desc, schema, handler) => {
  _tools[name] = { name, description: desc, inputSchema: schema, handler };
};

// Register bolt definitions from within the Worker.
// Bolts are forwarded to the main thread for execution.
self._registerBolts = (bolts) => {
  self.postMessage({ type: 'register_bolts', bolts });
};

// Register bolt handlers — functions the main thread can invoke
// for bolt actions of type 'worker_decode'.
const _boltHandlers = {};
self._registerBoltHandler = (name, fn) => {
  _boltHandlers[name] = fn;
};

// Stub MCP SDK schema constants so setRequestHandler can identify them
const ListToolsRequestSchema = { method: 'tools/list' };
const CallToolRequestSchema = { method: 'tools/call' };
const ListResourcesRequestSchema = { method: 'resources/list' };
const ReadResourceRequestSchema = { method: 'resources/read' };

// Namespace for cross-file module resolution
self._mcp_modules = {};

// === Nudge Data Files API ===
// Attached files (PDFs, CSVs, XLSX, etc.) are available to tool handlers.
// Each entry: { name, mimeType, size, base64, text }
//   - text:   extracted plain text (empty for images/unsupported)
//   - base64: raw file bytes as base64 string
self._mcp_data_files = ${dataFilesJson};

/**
 * Get an attached data file by name.
 * @param {string} name - Exact filename (e.g. "products.csv")
 * @returns {{ name, mimeType, size, base64, text } | undefined}
 */
self._getDataFile = (name) => self._mcp_data_files.find(f => f.name === name);

/**
 * Get all attached data files.
 * @returns {Array<{ name, mimeType, size, base64, text }>}
 */
self._listDataFiles = () => self._mcp_data_files.map(f => ({ name: f.name, mimeType: f.mimeType, size: f.size }));

/**
 * Get the extracted text content of a data file.
 * @param {string} name - Exact filename
 * @returns {string} Plain text content, or empty string if not available
 */
self._getDataFileText = (name) => { const f = self._getDataFile(name); return f ? f.text : ''; };

/**
 * Decode a data file's base64 content to a Uint8Array.
 * @param {string} name - Exact filename
 * @returns {Uint8Array | null}
 */
self._getDataFileBytes = (name) => {
  const f = self._getDataFile(name);
  if (!f || !f.base64) return null;
  const bin = atob(f.base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
};

// Unwrap MCP SDK result format: { content: [{ type: 'text', text: '...' }] } → text
function _unwrapResult(result) {
  if (result && typeof result === 'object' && Array.isArray(result.content)) {
    const texts = result.content.filter(c => c.type === 'text').map(c => c.text);
    if (texts.length > 0) return texts.join('\\n');
  }
  return typeof result === 'string' ? result : JSON.stringify(result);
}

// Low-level setRequestHandler shim storage
const _requestHandlers = {};

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    // If setRequestHandler was used for ListToolsRequestSchema, call it to get tools
    if (_requestHandlers['listTools'] && Object.keys(_tools).length === 0) {
      try {
        const resp = await _requestHandlers['listTools']({});
        if (resp && resp.tools) {
          for (const t of resp.tools) {
            _tools[t.name] = { name: t.name, description: t.description || '', inputSchema: t.inputSchema || {}, handler: null };
          }
        }
      } catch(e) { /* ignore */ }
    }
    self.postMessage({
      type: 'mcp_ready',
      tools: Object.values(_tools).map(t => ({
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema,
      })),
    });
  } else if (msg.type === 'tool_call') {
    try {
      let resultStr;
      // If setRequestHandler was used for CallToolRequestSchema, route through it
      if (_requestHandlers['callTool']) {
        const resp = await _requestHandlers['callTool']({ params: { name: msg.name, arguments: msg.arguments } });
        if (resp && resp.isError) throw new Error(_unwrapResult(resp));
        resultStr = _unwrapResult(resp);
      } else {
        const tool = _tools[msg.name];
        if (!tool) throw new Error('Unknown tool: ' + msg.name);
        const result = await tool.handler(msg.arguments);
        resultStr = _unwrapResult(result);
      }
      let _isRich = false;
      try { _isRich = !!JSON.parse(resultStr).__rich_mcp__; } catch(_) {}
      self.postMessage({ type: 'tool_result', callId: msg.callId, result: resultStr, rich_mcp: _isRich });
    } catch (err) {
      self.postMessage({ type: 'tool_result', callId: msg.callId, error: err.message || String(err), rich_mcp: false });
    }
  } else if (msg.type === 'bolt_decode') {
    // Invoke a registered bolt handler and return the result
    const handler = _boltHandlers[msg.handler];
    if (!handler) {
      self.postMessage({ type: 'bolt_decode_result', callId: msg.callId, error: 'Unknown handler: ' + msg.handler });
    } else {
      try {
        const result = await handler(msg.input);
        self.postMessage({ type: 'bolt_decode_result', callId: msg.callId, result });
      } catch (err) {
        self.postMessage({ type: 'bolt_decode_result', callId: msg.callId, error: err.message || String(err) });
      }
    }
  } else if (msg.type === '_get_tests') {
    self.postMessage({ type: '_tests', tests: self._tests || null });
  }
};
`);

      // -- Process non-entry-point files first (data, generators, etc.) --
      for (const [name, source] of Object.entries(files)) {
        if (name === entryPoint) continue;
        const moduleName = name.replace(/\.js$/, '').replace(/[^a-zA-Z0-9_]/g, '_');
        const cleaned = this._stripEsModuleSyntax(source, moduleName);
        parts.push(`\n// === Module: ${name} ===`);
        parts.push(`(function() {`);
        parts.push(`  const _exports = {};`);
        parts.push(cleaned);
        parts.push(`  self._mcp_modules['${moduleName}'] = _exports;`);
        parts.push(`})();`);
      }

      // -- Process entry point: rewrite to use _registerMcpTool --
      parts.push(`\n// === Entry Point: ${entryPoint} ===`);
      const rewrittenEntry = this._rewriteEntryPoint(files[entryPoint] || '', entryPoint);
      parts.push(rewrittenEntry);

      return parts.join('\n');
    },

    /**
     * Strip ES module import/export syntax and convert to namespace lookups.
     * Handles common patterns:
     *   import { foo, bar } from './data.js'  => const { foo, bar } = self._mcp_modules.data;
     *   export const foo = ...                 => _exports.foo = ...; const foo = ...
     *   export function foo()                  => _exports.foo = foo; function foo()
     *   export { foo, bar }                    => _exports.foo = foo; _exports.bar = bar;
     *   export default ...                     => _exports.default = ...
     */
    _stripEsModuleSyntax(source, selfModuleName) {
      let result = source;

      // Replace: import { x, y } from './file.js'
      result = result.replace(
        /import\s+\{([^}]+)\}\s+from\s+['"]\.?\/?([^'"]+)['"]\s*;?/g,
        (_, imports, path) => {
          const modName = path.replace(/\.js$/, '').replace(/.*\//, '').replace(/[^a-zA-Z0-9_]/g, '_');
          return `const {${imports}} = self._mcp_modules['${modName}'];`;
        }
      );

      // Replace: import x from './file.js'  (default import)
      result = result.replace(
        /import\s+(\w+)\s+from\s+['"]\.?\/?([^'"]+)['"]\s*;?/g,
        (_, name, path) => {
          const modName = path.replace(/\.js$/, '').replace(/.*\//, '').replace(/[^a-zA-Z0-9_]/g, '_');
          return `const ${name} = self._mcp_modules['${modName}'].default || self._mcp_modules['${modName}'];`;
        }
      );

      // Handle: export const x = ...
      result = result.replace(
        /export\s+const\s+(\w+)\s*=/g,
        (_, name) => `const ${name} = _exports['${name}'] =`
      );

      // Handle: export function x(
      result = result.replace(
        /export\s+function\s+(\w+)\s*\(/g,
        (_, name) => `_exports['${name}'] = ${name}; function ${name}(`
      );

      // Handle: export default
      result = result.replace(
        /export\s+default\s+/g,
        '_exports.default = '
      );

      // Handle: export { x, y, z }
      result = result.replace(
        /export\s+\{([^}]+)\}\s*;?/g,
        (_, names) => {
          return names.split(',').map(n => {
            const trimmed = n.trim().split(/\s+as\s+/);
            const local = trimmed[0].trim();
            const exported = (trimmed[1] || local).trim();
            return `_exports['${exported}'] = ${local};`;
          }).join(' ');
        }
      );

      return result;
    },

    /**
     * Rewrite the entry point to use _registerMcpTool instead of MCP SDK.
     * Detects the MCP Server.create() / tool() pattern and converts it.
     */
    _rewriteEntryPoint(source, entryPoint) {
      let result = source;

      // Strip all import statements (SDK imports, local imports)
      result = result.replace(
        /import\s+\{([^}]+)\}\s+from\s+['"]\.?\/?([^'"]+)['"]\s*;?/g,
        (_, imports, path) => {
          // If it's a local file, resolve from namespace
          if (path.startsWith('.') || path.startsWith('/') || !path.includes('/')) {
            const modName = path.replace(/\.js$/, '').replace(/.*\//, '').replace(/[^a-zA-Z0-9_]/g, '_');
            return `const {${imports}} = self._mcp_modules['${modName}'];`;
          }
          // SDK import — skip
          return `// [stripped SDK import: ${path}]`;
        }
      );

      // Strip default imports from SDK
      result = result.replace(
        /import\s+\w+\s+from\s+['"][^'"]*@modelcontextprotocol[^'"]*['"]\s*;?/g,
        '// [stripped SDK default import]'
      );

      // Replace McpServer/Server constructor with a lightweight shim that supports:
      //   - server.tool(name, [desc], schema, handler)  — high-level API
      //   - server.setRequestHandler(schema, handler)    — low-level API
      // Uses balanced-paren matching for multi-line constructors.
      {
        const ctorRe = /(?:const|let|var)\s+(\w+)\s*=\s*new\s+(?:McpServer|Server)\s*\(/g;
        let ctorMatch;
        while ((ctorMatch = ctorRe.exec(result)) !== null) {
          const varName = ctorMatch[1];
          let depth = 1, i = ctorMatch.index + ctorMatch[0].length;
          while (i < result.length && depth > 0) {
            if (result[i] === '(') depth++;
            else if (result[i] === ')') depth--;
            i++;
          }
          if (i < result.length && result[i] === ';') i++;
          const shim = `const ${varName} = {
  tool: function() {
    if (arguments.length === 3) _registerMcpTool(arguments[0], '', arguments[1], arguments[2]);
    else _registerMcpTool(arguments[0], arguments[1], arguments[2], arguments[3]);
  },
  setRequestHandler: function(schema, handler) {
    const s = (schema && schema.method) || '';
    if (s.includes('tools/list')) _requestHandlers['listTools'] = handler;
    else if (s.includes('tools/call')) _requestHandlers['callTool'] = handler;
  },
  connect: function() {},
  close: function() {},
};`;
          result = result.substring(0, ctorMatch.index) + shim + result.substring(i);
          ctorRe.lastIndex = ctorMatch.index + shim.length;
        }
      }

      // Strip server.connect() / server.run() / server.start() / StdioServerTransport
      result = result.replace(
        /(?:const|let|var)\s+\w+\s*=\s*new\s+\w*Transport\s*\([^)]*\)\s*;?/g,
        '// [stripped transport init]'
      );
      result = result.replace(
        /(?:await\s+)?\w+\.(?:connect|run|start)\s*\([^)]*\)\s*;?/g,
        (match) => {
          // Only strip if it looks like server.connect/run/start, not other method calls
          if (/\.(?:connect|run|start)\s*\(/.test(match)) {
            return '// [stripped server start]';
          }
          return match;
        }
      );

      // Strip main() bootstrap pattern: async function main() { ... } main().catch(...)
      result = result.replace(
        /async\s+function\s+main\s*\(\s*\)\s*\{[\s\S]*?\n\}\s*/g,
        '// [stripped main()]\n'
      );
      result = result.replace(
        /main\s*\(\s*\)\.catch\s*\([^)]*\)\s*;?/g,
        '// [stripped main().catch()]'
      );

      // Strip shebang
      result = result.replace(/^#!.*\n/, '');

      return result;
    },

    /**
     * Initialize an MCP Worker: send init, await tool list.
     */
    _initMcpWorker(worker, nudgeUid, requestId) {
      const timeout = setTimeout(() => {
        console.error('[BROWSER_MCP] Worker init timed out for', nudgeUid);
        this.ws.send({
          type: 'browser_mcp_result',
          request_id: requestId,
          error: 'Worker init timed out (10s)',
        });
      }, 10000);

      worker.onmessage = (e) => {
        const msg = e.data;

        // Handle bolt registration during init phase too
        if (msg.type === 'register_bolts' && Array.isArray(msg.bolts)) {
          this._boltsRegister(nudgeUid, msg.bolts);
          return;
        }

        if (msg.type === 'mcp_ready') {
          clearTimeout(timeout);
          console.log(`[BROWSER_MCP] Worker ready for ${nudgeUid}: ${msg.tools.length} tools`);

          // Set up permanent message handler for tool results + bolt registration
          worker.onmessage = (e2) => {
            const res = e2.data;
            if (res.type === 'tool_result') {
              const entry = (this._mcpWorkers || {})[nudgeUid];
              if (entry && entry.pendingCalls[res.callId]) {
                const { requestId: reqId } = entry.pendingCalls[res.callId];
                delete entry.pendingCalls[res.callId];

                // Send result to server — rich_mcp envelope passes through for server-side storage
                this.ws.send({
                  type: 'browser_mcp_result',
                  request_id: reqId,
                  ...(res.error ? { error: res.error } : { result: res.result }),
                });
              }
            } else if (res.type === 'register_bolts' && Array.isArray(res.bolts)) {
              // JS droplet registering bolts from within the Worker
              this._boltsRegister(nudgeUid, res.bolts);
            } else if (res.type === 'bolt_decode_result') {
              // Resolve a pending bolt decode call
              const entry = (this._mcpWorkers || {})[nudgeUid];
              const cb = entry && entry.pendingBoltDecodes && entry.pendingBoltDecodes[res.callId];
              if (cb) {
                delete entry.pendingBoltDecodes[res.callId];
                cb(res);
              }
            }
          };

          // Send tool list back to server
          this.ws.send({
            type: 'browser_mcp_result',
            request_id: requestId,
            tools: msg.tools,
          });
        }
      };

      worker.onerror = (err) => {
        clearTimeout(timeout);
        console.error('[BROWSER_MCP] Worker error:', err);
        this.ws.send({
          type: 'browser_mcp_result',
          request_id: requestId,
          error: `Worker error: ${err.message || 'unknown'}`,
        });
      };

      // Trigger init
      worker.postMessage({ type: 'init' });
    },

    /**
     * Forward a tool call to a Worker and track the pending request.
     */
    _callWorkerTool(entry, requestId, toolName, args) {
      const callId = crypto.randomUUID();
      entry.pendingCalls[callId] = { requestId };

      entry.worker.postMessage({
        type: 'tool_call',
        callId,
        name: toolName,
        arguments: args,
      });

      // Timeout for individual tool calls
      setTimeout(() => {
        if (entry.pendingCalls[callId]) {
          delete entry.pendingCalls[callId];
          this.ws.send({
            type: 'browser_mcp_result',
            request_id: requestId,
            error: `Tool call '${toolName}' timed out (30s)`,
          });
        }
      }, 30000);
    },

    /**
     * Send a bolt decode request to a Worker and return a Promise with the result.
     * @param {string} nudgeUid - The nudge that owns the Worker
     * @param {string} handlerName - Name of the registered bolt handler
     * @param {string} input - User input to decode
     * @returns {Promise<object>} Resolved with { result } or { error }
     */
    _callWorkerBoltDecode(nudgeUid, handlerName, input) {
      const entry = (this._mcpWorkers || {})[nudgeUid];
      if (!entry) return Promise.resolve({ error: 'No active Worker for ' + nudgeUid });

      if (!entry.pendingBoltDecodes) entry.pendingBoltDecodes = {};
      const callId = crypto.randomUUID();

      return new Promise((resolve) => {
        entry.pendingBoltDecodes[callId] = resolve;
        entry.worker.postMessage({ type: 'bolt_decode', callId, handler: handlerName, input });
        // Timeout after 5s
        setTimeout(() => {
          if (entry.pendingBoltDecodes[callId]) {
            delete entry.pendingBoltDecodes[callId];
            resolve({ error: 'Bolt decode timed out (5s)' });
          }
        }, 5000);
      });
    },

    /**
     * Show an inline trust prompt below a rich MCP visualization.
     * Displayed when a droplet wants to inject messages or trigger LLM calls.
     */
    _showTrustPrompt(nudgeUid, rich, parentEl) {
      const prompt = document.createElement('div');
      prompt.className = 'cv2-rich-mcp-trust-prompt';
      const actions = [];
      if (rich.inject_messages) actions.push('add messages to the conversation');
      if (rich.trigger_llm_call) actions.push('trigger a follow-up AI response');
      prompt.innerHTML = `
        <div class="cv2-trust-prompt-text">
          <span class="material-icons cv2-trust-prompt-icon">security</span>
          This droplet wants to ${actions.join(' and ')}.
        </div>
        <div class="cv2-trust-prompt-actions">
          <button class="cv2-trust-btn cv2-trust-btn-allow" data-action="once">Allow once</button>
          <button class="cv2-trust-btn cv2-trust-btn-always" data-action="always">Always trust</button>
          <button class="cv2-trust-btn cv2-trust-btn-deny" data-action="deny">Deny</button>
        </div>`;
      parentEl.appendChild(prompt);

      prompt.addEventListener('click', (e) => {
        const action = e.target.closest('[data-action]')?.dataset?.action;
        if (!action) return;
        prompt.remove();
        if (action === 'deny') return;
        if (action === 'always') {
          this._trustedDroplets = this._trustedDroplets || new Set();
          this._trustedDroplets.add(nudgeUid);
          this.ws.send({ type: 'mcp_trust:grant', nudge_uid: nudgeUid });
        }
        // For 'once' or 'always', send the trust-gated actions to server for execution
        this.ws.send({
          type: 'mcp_trust:execute_once',
          nudge_uid: nudgeUid,
          inject_messages: rich.inject_messages || null,
          trigger_llm_call: rich.trigger_llm_call || null,
        });
      });
    },

    /**
     * Resolve vendor library scripts for iframe injection.
     * Fetches and caches vendor lib content from local paths.
     */
    async _resolveVendorScripts(libKeys) {
      if (!this._vendorLibCache) this._vendorLibCache = {};
      const VENDOR_LIBS = {
        plotly:  { src: '/static/chat/vendor/plotly/plotly-2.35.2.min.js' },
        three:   { src: '/static/chat/vendor/three/three.min.js' },
      };
      return Promise.all(libKeys.map(async (key) => {
        if (this._vendorLibCache[key]) return this._vendorLibCache[key];
        const lib = VENDOR_LIBS[key];
        if (!lib) return null;
        try {
          const resp = await fetch(lib.src);
          const text = await resp.text();
          this._vendorLibCache[key] = text;
          return text;
        } catch (err) {
          console.warn('[RICH_MCP] Failed to fetch vendor lib:', key, err);
          return null;
        }
      }));
    },

    /**
     * Append an injected message bubble (from trusted droplet).
     */
    _appendInjectedMessage(role, content, sourceNudge) {
      const messagesEl = document.querySelector('.cv2-messages');
      if (!messagesEl) return;
      const msgDiv = document.createElement('div');
      msgDiv.className = `cv2-msg cv2-msg-${role}`;
      const contentDiv = document.createElement('div');
      contentDiv.className = 'cv2-msg-content';
      contentDiv.textContent = content;
      msgDiv.appendChild(contentDiv);
      // Add injected badge
      const badge = document.createElement('span');
      badge.className = 'cv2-injected-badge';
      badge.textContent = `injected by droplet`;
      contentDiv.appendChild(badge);
      messagesEl.appendChild(msgDiv);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    },

    /**
     * Clean up all browser MCP Workers (called on disconnect).
     */
    _cleanupBrowserMcp() {
      if (!this._mcpWorkers) return;
      for (const [uid, entry] of Object.entries(this._mcpWorkers)) {
        try {
          entry.worker.terminate();
          URL.revokeObjectURL(entry.blobUrl);
        } catch (e) { /* ignore */ }
      }
      this._mcpWorkers = {};
    },

  });

  ChatFeatures.register('browserMcp', {
    initState(app) {
      app._mcpWorkers = {};
    },
    handleMessage: {
      start_browser_mcp(app, msg) { app._handleStartBrowserMcp(msg); },
      browser_mcp_call(app, msg) { app._handleBrowserMcpCall(msg); },
      stop_browser_mcp(app, msg) { app._handleStopBrowserMcp(msg); },
    },
  });

  ChatFeatures.register('mcpRich', {
    initState(app) {
      app._trustedDroplets = new Set();
    },
    handleMessage: {
      injected_message(app, msg) {
        app._appendInjectedMessage(msg.role, msg.content, msg.source_nudge);
      },
      mcp_trust_list(app, msg) {
        app._trustedDroplets = new Set(msg.nudge_uids || []);
      },
    },
  });

})();
