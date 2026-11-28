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
     * First call downloads all-MiniLM-L6-v2 (~87 MiB) if not cached.
     *
     * Returns: opaque handle (free with memoire_free), or NULL on error.
     */
    MemoireHandle *memoire_new(const char *db_path);

    /*
     * memoire_new_ns
     * --------------
     * Open or create a Memoire database at `db_path` scoped to `namespace_`.
     * `namespace_` may be NULL, in which case "default" is used.
     * Multiple handles sharing the same `db_path` but different namespaces
     * are fully isolated — recall, feedback (memoire_reinforce_if_used,
     * memoire_penalize_if_used), memoire_forget and
     * memoire_resolve_contradictions in one namespace never see or modify
     * memories owned by another, even when given that memory's id.
     *
     * Returns: opaque handle (free with memoire_free), or NULL on error.
     */
    MemoireHandle *memoire_new_ns(const char *db_path, const char *namespace_);

    /*
     * memoire_free
     * ------------
     * Destroy a handle returned by memoire_new or memoire_new_ns.
     * Safe to call on NULL.
     */
    void memoire_free(MemoireHandle *handle);

    /*
     * memoire_remember
     * ----------------
     * Chunk, embed, and store `content`.
     *
     * Returns: number of chunks newly stored (>= 0) on success, -1 on error.
     *          0 is not an error: blank input, chunks that exactly duplicate an
     *          existing memory, and chunks rejected at ingestion are not counted.
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
     * Returns: 1 = reinforced, 0 = not reinforced (attribution gate not met, or
     *          the id is not owned by this handle's namespace), -1 = error.
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
     * `failure_severity`: a gate, not a weight. Any finite value > 0 records exactly
     *                     one failure per memory; <= 0, NaN or infinity records none.
     *                     Fractional scaling (for example 0.5 = partial miss) is not
     *                     supported in 0.1.x: 0.5 counts the same as 1.0.
     *
     * Repeated ids within one call count once.
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
     *
     * Ids that do not exist in this handle's namespace produce no outcome.
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
     * Returns: 1 = deleted, 0 = not found (including an id owned by another
     *          namespace), -1 = error.
     */
    int memoire_forget(MemoireHandle *handle, int64_t id);

    /*
     * memoire_resolve_contradictions
     * ------------------------------
     * Explicitly trigger contradiction resolution for a specific memory id.
     * Scans for semantically similar memories with opposing polarity and archives
     * the lower-quality one.
     *
     * Returns: 0 on success (an id not owned by this handle's namespace is a
     *          no-op), -1 on error.
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
     * Erase ALL memories in this handle's namespace. Irreversible.
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
