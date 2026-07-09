PRAGMA user_version = 2;

ALTER TABLE memories ADD COLUMN trust_variance          REAL    NOT NULL DEFAULT 0.0;
ALTER TABLE memories ADD COLUMN trust_samples           INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN trust_last_session_id   TEXT;
ALTER TABLE memories ADD COLUMN ingestion_driver        TEXT;

CREATE TABLE IF NOT EXISTS prototype_checkpoints (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace                TEXT    NOT NULL DEFAULT 'default',
    created_at               INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    reinforcements_at_update INTEGER NOT NULL,
    actionability_centroid   BLOB    NOT NULL,
    consequence_centroid     BLOB    NOT NULL,
    reusability_centroid     BLOB    NOT NULL
);
