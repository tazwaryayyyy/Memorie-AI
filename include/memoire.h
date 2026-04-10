#ifndef MEMOIRE_H
#define MEMOIRE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Opaque handle. Never dereference directly. */
typedef struct MemoireHandle MemoireHandle;

/*
 * memoire_new
 * -----------
 * Open or create a Memoire database at `db_path`.
 * Pass ":memory:" for a non-persistent in-memory store.
 *
 * First call downloads all-MiniLM-L6-v2 (~23 MB) if not cached.
 *
 * Returns: opaque handle (free with memoire_free), or NULL on error.
 */
MemoireHandle* memoire_new(const char* db_path);

/*
 * memoire_free
 * ------------
 * Destroy a handle. Safe to call on NULL.
 */
void memoire_free(MemoireHandle* handle);

/*
 * memoire_remember
 * ----------------
 * Chunk, embed, and store `content`.
 *
 * Returns: chunks stored (>= 1) on success, -1 on error.
 */
int memoire_remember(MemoireHandle* handle, const char* content);

/*
 * memoire_recall
 * --------------
 * Return the `top_k` memories most semantically similar to `query`.
 *
 * Returns: heap-allocated null-terminated JSON string, or NULL on error.
 *          *** MUST be freed with memoire_free_string ***
 *
 * JSON schema:
 *   [
 *     {
 *       "id":         <integer>,
 *       "content":    <string>,
 *       "score":      <float 0–1>,
 *       "created_at": <unix timestamp>
 *     },
 *     ...
 *   ]
 */
char* memoire_recall(const MemoireHandle* handle,
                     const char*          query,
                     int                  top_k);

/*
 * memoire_forget
 * --------------
 * Delete a memory by id.
 *
 * Returns: 1 = deleted, 0 = not found, -1 = error.
 */
int memoire_forget(MemoireHandle* handle, int64_t id);

/*
 * memoire_count
 * -------------
 * Total stored memory chunks, or -1 on error.
 */
int64_t memoire_count(const MemoireHandle* handle);

/*
 * memoire_clear
 * -------------
 * Erase ALL memories. Irreversible.
 *
 * Returns: 0 on success, -1 on error.
 */
int memoire_clear(MemoireHandle* handle);

/*
 * memoire_free_string
 * -------------------
 * Free a string returned by memoire_recall.
 * Safe to call on NULL. Do NOT use free() or Python's gc alone.
 */
void memoire_free_string(char* s);

#ifdef __cplusplus
}
#endif
#endif /* MEMOIRE_H */
