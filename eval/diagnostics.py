"""Exemplar-cosine diagnostic: how close is a text to the 15 sentences Memoire's ingest-quality
prototypes are built from (src/lib.rs::Memoire::compute_prototypes)?

[Verified from code] the real prototypes are per-group MEAN embeddings (centroids); Python cannot
read them, so this reports the max cosine to any single exemplar, a proxy that tends to be higher
than a centroid cosine. Advisory only; it never gates anything. Cosine comes from
`Memoire.recall_plain(...).score` (plain cosine clamped to [0, 1]) on a scratch in-memory store.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

# Copied from src/lib.rs::compute_prototypes; eval/tests/test_diagnostics.py parses lib.rs and
# fails if these drift.
EXEMPLARS: Dict[str, List[str]] = {
    "consequence": [
        "Critical security breach in production database exposing user data",
        "System outage caused data loss for all users",
        "Race condition in payment processing leads to double charges",
        "Authentication bypass vulnerability allows unauthorized access",
        "Production database corrupted during migration",
    ],
    "actionability": [
        "Fixed the bug by replacing the function with the corrected version",
        "Replaced deprecated API call with the new stable endpoint",
        "Patched the vulnerability by updating the dependency to latest version",
        "Refactored the module to remove dead code and reduce coupling",
        "Disabled the feature flag after rollback to restore stable behavior",
    ],
    "reusability": [
        "Always close database connections and file handles in a finally block to avoid leaks",
        "Never store passwords in plain text always hash with bcrypt",
        "API rate limit policy must enforce maximum requests per hour",
        "Rule: all database queries must use parameterized statements",
        "Mandatory: all external inputs must be validated and sanitized",
    ],
}


def exemplar_cosines(memoire_factory: Callable[[], Any], texts: Mapping[str, str]) -> Dict[str, dict]:
    """Return {text_key: {max_cos, argmax, group_max}} using a scratch store."""
    memoire = memoire_factory()
    key_by_id: Dict[int, str] = {}
    for key, text in texts.items():
        for row_id in memoire.remember_ids(text):  # a long text may be several chunks
            key_by_id[row_id] = key
    results: Dict[str, dict] = {
        k: {"max_cos": None, "argmax": None, "group_max": {g: 0.0 for g in EXEMPLARS}} for k in texts
    }
    total = len(key_by_id)
    if total == 0:
        return results
    for group, sentences in EXEMPLARS.items():
        for sentence in sentences:
            for mem in memoire.recall_plain(sentence, top_k=total):
                key = key_by_id.get(mem.id)
                if key is None:
                    continue
                entry = results[key]
                score = float(mem.score)
                entry["group_max"][group] = max(entry["group_max"][group], round(score, 4))
                if entry["max_cos"] is None or score > entry["max_cos"]:
                    entry["max_cos"], entry["argmax"] = round(score, 4), sentence
    return results
