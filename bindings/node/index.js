'use strict';
/**
 * memoire — Node.js binding
 *
 * Uses `ffi-napi` + `ref-napi` to call the compiled libmemoire shared library.
 *
 * Install:
 *   npm install ffi-napi ref-napi
 *
 * Build the library first:
 *   cargo build --release   (from the repo root)
 *
 * Usage:
 *   const Memoire = require('./bindings/node');
 *   const m = new Memoire('./agent.db');
 *   m.remember('Fixed the JWT issuer bug today');
 *   const results = m.recall('authentication issues', 3);
 *   console.log(results);
 *   m.close();
 */

const ffi  = require('ffi-napi');
const ref  = require('ref-napi');
const path = require('path');
const os   = require('os');

// ─── Locate shared library ────────────────────────────────────────────────────

function resolveLib() {
  const root = path.resolve(__dirname, '..', '..');
  const names = {
    linux:  'libmemoire.so',
    darwin: 'libmemoire.dylib',
    win32:  'memoire.dll',
  };
  const name = names[os.platform()];
  if (!name) throw new Error(`Unsupported platform: ${os.platform()}`);
  const p = path.join(root, 'target', 'release', name);
  return p;
}

// ─── FFI declarations ─────────────────────────────────────────────────────────

const voidPtr  = ref.refType(ref.types.void);
const charPtr  = ref.refType(ref.types.char);

const lib = ffi.Library(resolveLib(), {
  memoire_new:         [voidPtr,  ['string']],
  memoire_free:        ['void',   [voidPtr]],
  memoire_remember:    ['int',    [voidPtr, 'string']],
  memoire_recall:      [charPtr,  [voidPtr, 'string', 'int']],
  memoire_forget:      ['int',    [voidPtr, 'int64']],
  memoire_count:       ['int64',  [voidPtr]],
  memoire_clear:       ['int',    [voidPtr]],
  memoire_free_string: ['void',   [charPtr]],
});

// ─── Wrapper class ────────────────────────────────────────────────────────────

class Memoire {
  /**
   * @param {string} dbPath  Path to the SQLite database file.
   *                         Pass ':memory:' for an ephemeral store.
   */
  constructor(dbPath = './memoire.db') {
    this._handle = lib.memoire_new(dbPath);
    if (this._handle.isNull()) {
      throw new Error(`Failed to open Memoire at '${dbPath}'`);
    }
  }

  /**
   * Store content as one or more searchable memory chunks.
   * @param {string} content
   * @returns {number} number of chunks stored
   */
  remember(content) {
    if (typeof content !== 'string' || !content.trim()) {
      throw new TypeError('content must be a non-empty string');
    }
    const n = lib.memoire_remember(this._handle, content);
    if (n < 0) throw new Error('memoire_remember() failed');
    return n;
  }

  /**
   * Find the top_k most semantically similar memories.
   * @param {string} query
   * @param {number} topK
   * @returns {Array<{id: number, content: string, score: number, created_at: number}>}
   */
  recall(query, topK = 5) {
    if (typeof query !== 'string') throw new TypeError('query must be a string');
    const rawPtr = lib.memoire_recall(this._handle, query, topK);
    if (rawPtr.isNull()) throw new Error('memoire_recall() failed');
    try {
      const json = ref.readCString(rawPtr);
      return JSON.parse(json);
    } finally {
      lib.memoire_free_string(rawPtr);
    }
  }

  /**
   * Delete a memory by id.
   * @param {number} id
   * @returns {boolean} true if the memory existed and was deleted
   */
  forget(id) {
    const r = lib.memoire_forget(this._handle, id);
    if (r < 0) throw new Error('memoire_forget() failed');
    return r === 1;
  }

  /**
   * Total number of stored memory chunks.
   * @returns {number}
   */
  count() {
    const n = lib.memoire_count(this._handle);
    if (n < 0) throw new Error('memoire_count() failed');
    return Number(n);
  }

  /**
   * Erase all memories. Irreversible.
   */
  clear() {
    if (lib.memoire_clear(this._handle) < 0) {
      throw new Error('memoire_clear() failed');
    }
  }

  /**
   * Release the native handle. Must be called when done.
   */
  close() {
    if (this._handle && !this._handle.isNull()) {
      lib.memoire_free(this._handle);
      this._handle = null;
    }
  }
}

module.exports = Memoire;
