/**
 * ChatFeatures — Feature registry + prototype accumulator
 *
 * Loaded FIRST (before any feature modules or chat-app-core.js).
 * Feature modules add methods to `_ChatAppProto` and register metadata.
 * chat-app-core.js applies all accumulated methods to ChatApp.prototype,
 * defines constructor/init/render/bindEvents, and boots the app.
 */

// Prototype method accumulator — feature modules assign methods here
window._ChatAppProto = {};

// Empty ChatApp class — populated by chat-app-core.js
class ChatApp {
  // Central file limits (available before core loads)
  static MAX_TOKEN_BUDGET = 100000;
  static MAX_FILES = 40;
  static MAX_SINGLE_FILE = 5 * 1024 * 1024;  // 5 MB
  static MAX_TOTAL_SIZE = 10 * 1024 * 1024;   // 10 MB
  static MAX_IMAGE_DIM = 3840;
  static DOC_ICONS = {
    plotly: 'bar_chart', latex: 'functions', table: 'table_chart',
    text_doc: 'description', word: 'description',
    presentation: 'slideshow', powerpoint: 'slideshow',
    html: 'code', email_draft: 'mail',
  };
}

window.ChatApp = ChatApp;

/**
 * Feature registry — feature modules call ChatFeatures.register() to declare:
 *   - initState(app):  initialise feature-specific state on the app instance
 *   - renderHTML(app):  return HTML string (injected into render() slots)
 *   - bindEvents(app):  bind feature-specific DOM events
 *   - handleMessage:   { msgType: 'methodName', ... } — WS dispatch table
 */
const ChatFeatures = (() => {
  const _features = {};

  return {
    register(name, descriptor) {
      _features[name] = descriptor;
    },

    /** Return all registered feature descriptors. */
    all() {
      return _features;
    },

    /** Get a single feature descriptor by name. */
    get(name) {
      return _features[name] || null;
    },

    /** Call initState on all features. */
    initAllState(app) {
      for (const [, feat] of Object.entries(_features)) {
        if (feat.initState) feat.initState(app);
      }
    },

    /** Call renderHTML on all features, return concatenated HTML. */
    renderAllHTML(app) {
      let html = '';
      for (const [, feat] of Object.entries(_features)) {
        if (feat.renderHTML) html += feat.renderHTML(app);
      }
      return html;
    },

    /** Call bindEvents on all features. */
    bindAllEvents(app) {
      for (const [, feat] of Object.entries(_features)) {
        if (feat.bindEvents) feat.bindEvents(app);
      }
    },

    /** Call cacheEls on all features (cache DOM refs after render). */
    cacheAllEls(app) {
      for (const [, feat] of Object.entries(_features)) {
        if (feat.cacheEls) feat.cacheEls(app);
      }
    },

    /** Build merged message dispatch table: { msgType: 'methodName' }. */
    messageHandlers() {
      const handlers = {};
      for (const [, feat] of Object.entries(_features)) {
        if (feat.handleMessage) {
          Object.assign(handlers, feat.handleMessage);
        }
      }
      return handlers;
    },
  };
})();

window.ChatFeatures = ChatFeatures;
