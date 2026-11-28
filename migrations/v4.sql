-- v4: durable feedback idempotency. `feedback_events` records which idempotency keys
-- have already been counted for a memory, so a repeat is refused even when other keys
-- arrived in between (the v3 guard only remembered the *last* reinforcing session).
--
--   kind = 'success'  event_key = the reinforcing `session_id`
--   kind = 'failure'  event_key = the penalizing `event_id`
--
-- Calls without a key are not recorded and are never deduplicated across calls.
-- ON DELETE CASCADE removes a memory's events with the memory (`forget`, `clear`,
-- maintenance pruning); `memories.id` is AUTOINCREMENT so ids are never reused.
-- The backfill keeps the v3 guarantee: each memory's last reinforcing session becomes
-- a 'success' event, so that session still cannot vote twice. Older history was never
-- stored and cannot be reconstructed.
-- Idempotent: CREATE IF NOT EXISTS + INSERT OR IGNORE. A v3 binary opening a v4
-- database ignores the extra table (it only checks `user_version < 3`).
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('success', 'failure')),
    event_key TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    UNIQUE (namespace, memory_id, kind, event_key)
);
INSERT OR IGNORE INTO feedback_events (namespace, memory_id, kind, event_key)
SELECT namespace, id, 'success', TRIM(trust_last_session_id)
FROM memories
WHERE trust_last_session_id IS NOT NULL
  AND TRIM(trust_last_session_id) <> '';
PRAGMA user_version = 4;
COMMIT;
