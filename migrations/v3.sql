-- v3: trust is a Beta posterior over (reinforcement_count, failure_count) and
-- `store_state` gains `trial` / `quarantined`. Live rows ('active' / 'trial') are
-- re-derived from their counts; hidden rows ('shadow' / 'probationary') are untouched.
-- Legacy penalties decremented reinforcement_count, so the backfill is approximate.
-- Idempotent: re-running derives the same states.
BEGIN IMMEDIATE;
UPDATE memories
SET store_state = CASE
    WHEN 2 * failure_count > 3 * reinforcement_count + 2 THEN 'quarantined'
    WHEN 3 * reinforcement_count >= 8 + 7 * failure_count THEN 'active'
    ELSE 'trial'
END
WHERE store_state IN ('active', 'trial');
PRAGMA user_version = 3;
COMMIT;
