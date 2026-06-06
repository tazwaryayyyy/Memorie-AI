#ifndef MEMOIRE_H
#define MEMOIRE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

    /* Opaque handle. Never dereference directly. */
    typedef struct MemoireHandle MemoireHandle;

    /*
     * memoire_new
     * -----------
     * Open or create a Memoire database at `db_path` in the default namespace.
     * Pass ":memory:" for a non-persistent in-memory store.
     *
     * First call downloads all-MiniLM-L6-v2 (~23 MB) if not cached.
     *
     * Returns: opaque handle (free with memoire_free), or NULL on error.
     */
    MemoireHandle *memoire_new(const char *db_path);

    /*
     * memoire_new_ns
     * --------------
     * Open or create a Memoire database at `db_path` scoped to `namespace`.
     * `namespace` may be NULL, in which case "default" is used.
     * Multiple handles sharing the same `db_path` but different namespaces
     * are fully isolated — recall from one namespace never surfaces memories
     * written by another.
     *
     * Returns: opaque handle (free with memoire_free), or NULL on error.
     */
    MemoireHandle *memoire_new_ns(const char *db_path, const char *namespace_);

    /*
     * memoire_free
     * ------------
     * Destroy a handle. Safe to call on NULL.
     */
    void memoire_free(MemoireHandle *handle);

    /*
     * memoire_remember
     * ----------------
     * Chunk, embed, and store `content`.
     *
     * Returns: chunks stored (>= 1) on success, -1 on error.
     */
    int memoire_remember(MemoireHandle *handle, const char *content);

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
    char *memoire_recall(const MemoireHandle *handle,
                         const char *query,
                         int top_k);

    /*
     * memoire_reinforce_if_used
     * -------------------------
     * Conditionally reinforce a memory based on whether the agent actually used it.
     *
     * `task_succeeded`: non-zero = task succeeded.
     *
     * Returns: 1 = reinforced, 0 = not reinforced (attribution gate not met),
     *          -1 = error.
     */
    int memoire_reinforce_if_used(MemoireHandle *handle,
                                  int64_t memory_id,
                                  const char *agent_output,
                                  int task_succeeded);

    /*
     * memoire_penalize_if_used
     * ------------------------
     * Penalize memories that contributed to a failed task outcome.
     *
     * `ids`:              pointer to an array of int64_t memory ids.
     * `ids_len`:          number of elements in `ids`.
     * `failure_severity`: [0.0, 1.0] — 1.0 = direct failure, 0.5 = partial miss.
     *
     * Returns: heap-allocated null-terminated JSON string of PenaltyOutcome objects,
     *          or NULL on error.
     *          *** MUST be freed with memoire_free_string ***
     *
     * JSON schema:
     *   [
     *     {
     *       "id":                <integer>,
     *       "trust_before":      <float>,
     *       "trust_after":       <float>,
     *       "uncertainty_after": <float>
     *     },
     *     ...
     *   ]
     */
    char *memoire_penalize_if_used(MemoireHandle *handle,
                                   const int64_t *ids,
                                   int ids_len,
                                   float failure_severity);

    /*
     * memoire_forget
     * --------------
     * Delete a memory by id.
     *
     * Returns: 1 = deleted, 0 = not found, -1 = error.
     */
    int memoire_forget(MemoireHandle *handle, int64_t id);

    /*
     * memoire_resolve_contradictions
     * ------------------------------
     * Explicitly trigger contradiction resolution for a specific memory id.
     * Scans for semantically similar memories with opposing polarity and archives
     * the lower-quality one.
     *
     * Returns: 0 on success, -1 on error.
     */
    int memoire_resolve_contradictions(MemoireHandle *handle, int64_t id);

    /*
     * memoire_count
     * -------------
     * Total stored memory chunks, or -1 on error.
     */
    int64_t memoire_count(const MemoireHandle *handle);

    /*
     * memoire_clear
     * -------------
     * Erase ALL memories. Irreversible.
     *
     * Returns: 0 on success, -1 on error.
     */
    int memoire_clear(MemoireHandle *handle);

    /*
     * memoire_free_string
     * -------------------
     * Free a string returned by memoire_recall or memoire_penalize_if_used.
     * Safe to call on NULL. Do NOT use free() or Python's gc alone.
     */
    void memoire_free_string(char *s);

#ifdef __cplusplus
}
#endif
#endif /* MEMOIRE_H */
