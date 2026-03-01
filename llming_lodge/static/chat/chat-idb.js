/**
 * IDBStore — IndexedDB wrapper for conversations, presets, favorites, documents
 *
 * Standalone class, no dependencies. Extracted from chat-app.js.
 */

class IDBStore {
  constructor(dbName = 'llming-lodge-chat', storeName = 'conversations', metaStoreName = 'conv_meta') {
    this.dbName = dbName;
    this.storeName = storeName;
    this.metaStoreName = metaStoreName;
    this.db = null;
  }

  /** Extract lightweight metadata from a full conversation object. */
  static _extractMeta(data) {
    const firstUserMsg = data.messages?.find(m => m.role === 'user')?.content || '';
    const meta = {
      id: data.id,
      title: data.title,
      created_at: data.created_at,
      updated_at: data.updated_at,
      message_count: data.messages?.length || 0,
      first_user_snippet: firstUserMsg.substring(0, 60),
    };
    if (data.project_id) meta.project_id = data.project_id;
    if (data.nudge_id) meta.nudge_id = data.nudge_id;
    if (data.favorited) meta.favorited = true;
    return meta;
  }

  async open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 7);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        // v1: conversations store
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: 'id' });
          store.createIndex('updated_at', 'updated_at', { unique: false });
        }
        // v2: lightweight metadata store
        if (!db.objectStoreNames.contains(this.metaStoreName)) {
          const meta = db.createObjectStore(this.metaStoreName, { keyPath: 'id' });
          meta.createIndex('updated_at', 'updated_at', { unique: false });
        }
        // Migrate: populate conv_meta from existing conversations
        if (e.oldVersion < 2) {
          const tx = e.target.transaction;
          const convStore = tx.objectStore(this.storeName);
          const metaStore = tx.objectStore(this.metaStoreName);
          const cursorReq = convStore.openCursor();
          cursorReq.onsuccess = (ev) => {
            const cursor = ev.target.result;
            if (cursor) {
              metaStore.put(IDBStore._extractMeta(cursor.value));
              cursor.continue();
            }
          };
        }
        // v3: presets store (projects + nudges)
        if (!db.objectStoreNames.contains('presets')) {
          const presets = db.createObjectStore('presets', { keyPath: 'id' });
          presets.createIndex('type', 'type', { unique: false });
          presets.createIndex('updated_at', 'updated_at', { unique: false });
        }
        // v5: favorites store (nudge UIDs + metadata)
        if (!db.objectStoreNames.contains('favorites')) {
          db.createObjectStore('favorites', { keyPath: 'uid' });
        }
        // v6: documents store (doc plugin system)
        if (!db.objectStoreNames.contains('documents')) {
          const docs = db.createObjectStore('documents', { keyPath: 'id' });
          docs.createIndex('conversation_id', 'conversation_id', { unique: false });
          docs.createIndex('type', 'type', { unique: false });
          docs.createIndex('updated_at', 'updated_at', { unique: false });
        }
        // v7: content-addressable file store
        if (!db.objectStoreNames.contains('files')) {
          db.createObjectStore('files', { keyPath: 'hash' });
        }

      };
      req.onsuccess = (e) => { this.db = e.target.result; resolve(this.db); };
      req.onerror = (e) => reject(e.target.error);
    });
  }

  async get(id) {
    const tx = this.db.transaction(this.storeName, 'readonly');
    const store = tx.objectStore(this.storeName);
    return new Promise((res, rej) => {
      const r = store.get(id);
      r.onsuccess = () => res(r.result || null);
      r.onerror = () => rej(r.error);
    });
  }

  /** Write conversation + update metadata in a single transaction. */
  async put(data) {
    const tx = this.db.transaction([this.storeName, this.metaStoreName], 'readwrite');
    const convStore = tx.objectStore(this.storeName);
    const metaStore = tx.objectStore(this.metaStoreName);
    return new Promise((res, rej) => {
      convStore.put(data);
      metaStore.put(IDBStore._extractMeta(data));
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  /** Get all conversations (full objects). Only use for export. */
  async getAll() {
    const tx = this.db.transaction(this.storeName, 'readonly');
    const store = tx.objectStore(this.storeName);
    const index = store.index('updated_at');
    return new Promise((res, rej) => {
      const results = [];
      const req = index.openCursor(null, 'prev');
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) { results.push(cursor.value); cursor.continue(); }
        else res(results);
      };
      req.onerror = () => rej(req.error);
    });
  }

  /** Get lightweight metadata for all conversations — reads from separate store, no messages deserialized. */
  async getAllMeta() {
    const tx = this.db.transaction(this.metaStoreName, 'readonly');
    const store = tx.objectStore(this.metaStoreName);
    const index = store.index('updated_at');
    return new Promise((res, rej) => {
      const results = [];
      const req = index.openCursor(null, 'prev');
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) { results.push(cursor.value); cursor.continue(); }
        else res(results);
      };
      req.onerror = () => rej(req.error);
    });
  }

  /** Get the most recent conversation (full object), or null. */
  async getMostRecent() {
    const tx = this.db.transaction(this.storeName, 'readonly');
    const store = tx.objectStore(this.storeName);
    const index = store.index('updated_at');
    return new Promise((res, rej) => {
      const req = index.openCursor(null, 'prev');
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        res(cursor ? cursor.value : null);
      };
      req.onerror = () => rej(req.error);
    });
  }

  /** Delete conversation + metadata in a single transaction. */
  async delete(id) {
    const tx = this.db.transaction([this.storeName, this.metaStoreName], 'readwrite');
    tx.objectStore(this.storeName).delete(id);
    tx.objectStore(this.metaStoreName).delete(id);
    return new Promise((res, rej) => {
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  // ── Preset CRUD (projects + nudges) ──────────────────

  async getPreset(id) {
    const tx = this.db.transaction('presets', 'readonly');
    return new Promise((res, rej) => {
      const r = tx.objectStore('presets').get(id);
      r.onsuccess = () => res(r.result || null);
      r.onerror = () => rej(r.error);
    });
  }

  async putPreset(data) {
    const tx = this.db.transaction('presets', 'readwrite');
    return new Promise((res, rej) => {
      tx.objectStore('presets').put(data);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  async deletePreset(id) {
    const tx = this.db.transaction('presets', 'readwrite');
    return new Promise((res, rej) => {
      tx.objectStore('presets').delete(id);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  /** Get all presets, optionally filtered by type ('project' | 'nudge'). */
  async getAllPresets(type) {
    const tx = this.db.transaction('presets', 'readonly');
    const store = tx.objectStore('presets');
    return new Promise((res, rej) => {
      const results = [];
      let req;
      if (type) {
        const index = store.index('type');
        req = index.openCursor(IDBKeyRange.only(type));
      } else {
        req = store.openCursor();
      }
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) { results.push(cursor.value); cursor.continue(); }
        else res(results.sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || '')));
      };
      req.onerror = () => rej(req.error);
    });
  }

  // ── Favorites (local nudge favorites by UID) ─────

  async getFavorites() {
    const tx = this.db.transaction('favorites', 'readonly');
    return new Promise((res, rej) => {
      const r = tx.objectStore('favorites').getAll();
      r.onsuccess = () => res(r.result || []);
      r.onerror = () => rej(r.error);
    });
  }

  async putFavorite(nudge) {
    // Store minimal metadata: uid, name, icon, description, creator_name
    const data = {
      uid: nudge.uid,
      name: nudge.name || '',
      icon: nudge.icon || null,
      description: nudge.description || '',
      creator_name: nudge.creator_name || '',
      creator_email: nudge.creator_email || '',
      added_at: new Date().toISOString(),
    };
    const tx = this.db.transaction('favorites', 'readwrite');
    return new Promise((res, rej) => {
      tx.objectStore('favorites').put(data);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  async deleteFavorite(uid) {
    const tx = this.db.transaction('favorites', 'readwrite');
    return new Promise((res, rej) => {
      tx.objectStore('favorites').delete(uid);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  // ── Documents (doc plugin system) ────────────────────

  async putDocument(doc) {
    const tx = this.db.transaction('documents', 'readwrite');
    return new Promise((res, rej) => {
      tx.objectStore('documents').put(doc);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  async getDocument(id) {
    const tx = this.db.transaction('documents', 'readonly');
    return new Promise((res, rej) => {
      const r = tx.objectStore('documents').get(id);
      r.onsuccess = () => res(r.result || null);
      r.onerror = () => rej(r.error);
    });
  }

  async getDocumentsForConversation(conversationId) {
    const tx = this.db.transaction('documents', 'readonly');
    const index = tx.objectStore('documents').index('conversation_id');
    return new Promise((res, rej) => {
      const results = [];
      const req = index.openCursor(IDBKeyRange.only(conversationId));
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) { results.push(cursor.value); cursor.continue(); }
        else res(results);
      };
      req.onerror = () => rej(req.error);
    });
  }

  async deleteDocument(id) {
    const tx = this.db.transaction('documents', 'readwrite');
    return new Promise((res, rej) => {
      tx.objectStore('documents').delete(id);
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  // ── Files (content-addressable file store) ────────────

  /** Upsert file: if hash exists, add conversationId to refs; if new, create record.
   *  If data is null and the record doesn't exist yet, the call is a no-op (ref-only update). */
  async putFile(hash, name, mimeType, size, data, conversationId, meta = {}) {
    const tx = this.db.transaction('files', 'readwrite');
    const store = tx.objectStore('files');
    return new Promise((res, rej) => {
      const getReq = store.get(hash);
      getReq.onsuccess = () => {
        const existing = getReq.result;
        if (existing) {
          if (!existing.refs.includes(conversationId)) {
            existing.refs.push(conversationId);
          }
          store.put(existing);
        } else if (data != null) {
          store.put({
            hash, name, mime_type: mimeType, size, data,
            created_at: new Date().toISOString(),
            refs: [conversationId],
            meta,
          });
        }
      };
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  /** Get full file record including data blob. */
  async getFile(hash) {
    const tx = this.db.transaction('files', 'readonly');
    return new Promise((res, rej) => {
      const r = tx.objectStore('files').get(hash);
      r.onsuccess = () => res(r.result || null);
      r.onerror = () => rej(r.error);
    });
  }

  /** Get file metadata WITHOUT the data blob. */
  async getFileMeta(hash) {
    const record = await this.getFile(hash);
    if (!record) return null;
    const { data, ...meta } = record;
    return meta;
  }

  /** Merge metaUpdates into a file's meta object. */
  async updateFileMeta(hash, metaUpdates) {
    const tx = this.db.transaction('files', 'readwrite');
    const store = tx.objectStore('files');
    return new Promise((res, rej) => {
      const getReq = store.get(hash);
      getReq.onsuccess = () => {
        const record = getReq.result;
        if (record) {
          record.meta = { ...record.meta, ...metaUpdates };
          store.put(record);
        }
      };
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  /** Remove conversationId from a file's refs. Delete the file if refs is empty. */
  async removeFileRef(hash, conversationId) {
    const tx = this.db.transaction('files', 'readwrite');
    const store = tx.objectStore('files');
    return new Promise((res, rej) => {
      const getReq = store.get(hash);
      getReq.onsuccess = () => {
        const record = getReq.result;
        if (record) {
          record.refs = record.refs.filter(r => r !== conversationId);
          if (record.refs.length === 0) {
            store.delete(hash);
          } else {
            store.put(record);
          }
        }
      };
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  /** Remove all refs for a conversation, deleting orphaned files. */
  async removeAllRefsForConversation(conversationId) {
    const tx = this.db.transaction('files', 'readwrite');
    const store = tx.objectStore('files');
    return new Promise((res, rej) => {
      const cursorReq = store.openCursor();
      cursorReq.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) {
          const record = cursor.value;
          if (record.refs.includes(conversationId)) {
            record.refs = record.refs.filter(r => r !== conversationId);
            if (record.refs.length === 0) {
              cursor.delete();
            } else {
              cursor.update(record);
            }
          }
          cursor.continue();
        }
      };
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  }

  /** Get all file records (including data). Used by debug listing. */
  async getAllFiles() {
    const tx = this.db.transaction('files', 'readonly');
    return new Promise((res, rej) => {
      const r = tx.objectStore('files').getAll();
      r.onsuccess = () => res(r.result || []);
      r.onerror = () => rej(r.error);
    });
  }
}
