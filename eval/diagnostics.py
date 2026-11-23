"""Exemplar-cosine diagnostic: how close is a text to the 15 sentences Memoire's ingest-quality
prototypes are built from (src/lib.rs::Memoire::compute_prototypes)?

[Verified from code] the real prototypes are per-group MEAN embeddings (centroids); Python cannot
read them, so this reports the max cosine to any single exemplar, a proxy that tends to be higher
than a centroid cosine. Advisory only; it never gates anything. Cosine comes from
`Memoire.recall_plain(...).score` (plain cosine clamped to [0, 1]) on a scratch in-memory store.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

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


def _norm(text: str) -> str:
    return " ".join(text.split())


def quality_scores(memoire_factory: Callable[[], Any], texts: Mapping[str, str]) -> Dict[str, Optional[dict]]:
    """Ingest-time quality score per text: the stored `importance_base` (weighted features under
    the engine's default ScoringWeights) and `confidence`, read back through `export_namespace()`.

    [Verified from src/quality.rs] with the default weights novelty and evidence weigh 0, so the
    score depends only on the text's embedding vs the three prototype centroids, not on what else
    is in the store or on ingestion order. A text that is a duplicate, was rejected or was split
    into several chunks cannot be mapped back and maps to None.
    """
    memoire = memoire_factory()
    for text in texts.values():
        memoire.remember_ids(text)
    stored = {_norm(m["content"]): m for m in memoire.export_namespace().get("memories", [])}
    out: Dict[str, Optional[dict]] = {}
    for key, text in texts.items():
        row = stored.get(_norm(text))
        out[key] = None if row is None else {
            "importance_base": round(float(row["importance_base"]), 4),
            "confidence": round(float(row["confidence"]), 4),
        }
    return out


TAU_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def summarize_quality(lessons_by_key: Mapping[str, str], scores: Mapping[str, Optional[dict]]) -> Dict[str, dict]:
    """Per lesson kind: count, min/mean/max importance_base and how many would be admitted
    (`importance_base >= tau`) at each tau in TAU_GRID. `lessons_by_key` maps 'lesson:<fam>:<kind>' -> text."""
    kinds: Dict[str, List[float]] = {}
    for key in lessons_by_key:
        info = scores.get(key)
        if info is not None:
            kinds.setdefault(key.rsplit(":", 1)[1], []).append(info["importance_base"])
    return {
        kind: {
            "n": len(vals), "min": min(vals), "mean": round(sum(vals) / len(vals), 4), "max": max(vals),
            "admitted": {f"{tau:.2f}": sum(v >= tau for v in vals) for tau in TAU_GRID},
        }
        for kind, vals in kinds.items()
    }


def pooled_quality(scores: Mapping[str, Optional[dict]], tau: float) -> dict:
    """Label-blind pooled statistics over every scored lesson (all kinds together)."""
    vals = sorted(v["importance_base"] for k, v in scores.items() if v is not None and k.startswith("lesson:"))
    if not vals:
        return {"n": 0}

    def nearest_rank(p: float) -> float:
        import math
        return vals[max(1, math.ceil(p * len(vals))) - 1]

    return {
        "n": len(vals), "min": vals[0], "p10": nearest_rank(0.10), "median": nearest_rank(0.50),
        "p90": nearest_rank(0.90), "max": vals[-1], "tau": tau, "admitted_at_tau": sum(v >= tau for v in vals),
    }


def relevance_separation(
    memoire_factory: Callable[[], Any],
    lessons: Mapping[str, "tuple[str, str]"],
    specs: Mapping[str, "tuple[str, str]"],
) -> dict:
    """How well does plain cosine separate a task's own-family lessons from other families' lessons?

    `lessons` maps key -> (family, text); `specs` maps task id -> (family, text). Uses only family
    membership, never correctness (gold/wrong/poison all count as own-family), so a floor derived
    from it cannot leak which lesson is right. If the pooled gap is positive the suggested floor is
    its midpoint; otherwise there is no clean floor and the answer is None.
    """
    memoire = memoire_factory()
    key_by_id: Dict[int, str] = {}
    for key, (_, text) in lessons.items():
        for row_id in memoire.remember_ids(text):
            key_by_id[row_id] = key
    total = len(key_by_id)
    rows: List[dict] = []
    for task_id, (family, text) in specs.items():
        cos: Dict[str, float] = {}
        for mem in memoire.recall_plain(text, top_k=max(total, 1)):
            key = key_by_id.get(mem.id)
            if key is not None:
                cos[key] = max(cos.get(key, 0.0), float(mem.score))
        own = [c for k, c in cos.items() if lessons[k][0] == family]
        other = [c for k, c in cos.items() if lessons[k][0] != family]
        if own and other:
            rows.append({"task": task_id, "family": family, "own_min": round(min(own), 4),
                         "own_mean": round(sum(own) / len(own), 4), "other_max": round(max(other), 4)})
    if not rows:
        return {"rows": [], "own_min": None, "other_max": None, "gap": None, "separable": False, "suggested_floor": None}
    own_min = min(r["own_min"] for r in rows)
    other_max = max(r["other_max"] for r in rows)
    gap = round(own_min - other_max, 4)
    return {
        "rows": rows, "own_min": own_min, "other_max": other_max, "gap": gap, "separable": gap > 0,
        "suggested_floor": round((own_min + other_max) / 2, 3) if gap > 0 else None,
    }
