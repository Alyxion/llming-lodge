/**
 * BlockDataStore — cross-block reference system for doc plugins.
 *
 * Allows any fenced code block to declare an "id" and be referenced
 * later via {"$ref": "<id>"} from any other block.  Resolution merges
 * the referenced data as a base with explicit properties as overrides.
 */

class BlockDataStore {
  constructor() {
    /** @type {Map<string, {lang: string, data: any}>} */
    this._blocks = new Map();
  }

  /** Register a block's parsed data under an id. */
  register(id, lang, data) {
    this._blocks.set(id, { lang, data });
  }

  /** Retrieve a registered block by id. */
  get(id) {
    return this._blocks.get(id) || null;
  }

  /** Check if an id is registered. */
  has(id) {
    return this._blocks.has(id);
  }

  /** Clear all stored blocks (call on conversation switch). */
  clear() {
    this._blocks.clear();
  }
}

/**
 * Recursively resolve $ref pointers in a parsed JSON structure.
 *
 * @param {any} obj — parsed JSON value
 * @param {BlockDataStore} store
 * @param {Set<string>} [seen] — ids already on the current resolution path (cycle detection)
 * @param {number} [depth=0] — recursion depth guard
 * @returns {any} resolved value
 */
function resolveBlockRefs(obj, store, seen, depth) {
  if (seen === undefined) seen = new Set();
  if (depth === undefined) depth = 0;
  if (depth > 10) return { _refError: 'Max depth exceeded' };

  // Primitives / null
  if (obj === null || typeof obj !== 'object') return obj;

  // Arrays — resolve each element with its own copy of `seen`
  if (Array.isArray(obj)) {
    return obj.map(function (item) {
      return resolveBlockRefs(item, store, new Set(seen), depth + 1);
    });
  }

  // Object with $ref
  if (obj.$ref) {
    var id = obj.$ref;
    if (seen.has(id)) return { _refError: 'Circular: ' + id };
    seen.add(id);
    var entry = store.get(id);
    if (!entry) return { _refError: 'Unknown ref: ' + id };

    // Merge: referenced data as base, explicit properties override
    var overrides = {};
    var keys = Object.keys(obj);
    for (var i = 0; i < keys.length; i++) {
      if (keys[i] !== '$ref') overrides[keys[i]] = obj[keys[i]];
    }
    var resolved = resolveBlockRefs(entry.data, store, new Set(seen), depth + 1);
    if (resolved && typeof resolved === 'object' && !Array.isArray(resolved)) {
      return Object.assign({}, resolved, overrides);
    }
    // If referenced data is not an object, overrides can't merge — return resolved
    return resolved;
  }

  // Plain object — recurse into each property
  var result = {};
  var objKeys = Object.keys(obj);
  for (var j = 0; j < objKeys.length; j++) {
    var k = objKeys[j];
    result[k] = resolveBlockRefs(obj[k], store, new Set(seen), depth + 1);
  }
  return result;
}

/**
 * Apply cross-type compatibility aliases after reference resolution.
 *
 * Handles schema differences when data from one plugin type is
 * embedded in another (e.g. table data in a powerpoint slide).
 *
 * @param {string} lang — target plugin language
 * @param {any} spec — the fully resolved spec
 * @returns {any} spec with compatibility aliases applied
 */
function applyCrossTypeCompat(lang, spec) {
  if (!spec || typeof spec !== 'object') return spec;

  if (lang === 'presentation' || lang === 'powerpoint') {
    // Walk slides → elements: alias table "columns" → "headers"
    if (Array.isArray(spec.slides)) {
      for (var i = 0; i < spec.slides.length; i++) {
        var slide = spec.slides[i];
        if (!Array.isArray(slide.elements)) continue;
        for (var j = 0; j < slide.elements.length; j++) {
          var el = slide.elements[j];
          if (el.type === 'table' && el.columns && !el.headers) {
            el.headers = el.columns;
            delete el.columns;
          }
        }
      }
    }
  }

  if (lang === 'table') {
    // Alias pptx "headers" → "columns"
    if (spec.headers && !spec.columns) {
      spec.columns = spec.headers;
      delete spec.headers;
    }
  }

  return spec;
}

// Export for global access
window.BlockDataStore = BlockDataStore;
window.resolveBlockRefs = resolveBlockRefs;
window.applyCrossTypeCompat = applyCrossTypeCompat;
