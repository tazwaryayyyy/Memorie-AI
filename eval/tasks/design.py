"""Design record for every family: the single open choice, its options, and which one is gold.

Dev gold choices were calibrated on the dev pilot (Arm A, gemini-3.1-flash-lite, 0/15 with
multi-convention tasks). Test gold choices follow a rule fixed BEFORE looking at any model output
on that split: families at odd positions use the choice most common in industry practice, families
at even positions use a less common but legitimate choice. No test-split model result exists.
"""
from __future__ import annotations

from typing import Dict

DESIGN: Dict[str, dict] = {
    # ---- dev (calibrated on dev pilot signatures; see README) ----
    "wire_timestamp": dict(split="dev", dimension="how the UTC designator is written",
                           options=["literal Z", "+00:00"], gold="literal Z"),
    "lock_ordering": dict(split="dev", dimension="direction of the global lock order",
                          options=["lower id first", "higher id first", "argument order (unsafe)"], gold="lower id first"),
    "fee_rounding": dict(split="dev", dimension="rounding of exact half-cent ties",
                         options=["half-even", "half-up"], gold="half-even"),
    "cache_key_format": dict(split="dev", dimension="which key parts are case-folded",
                             options=["tenant only", "all parts", "none"], gold="tenant only"),
    "http_status": dict(split="dev", dimension="status for ValueError (bad input)",
                        options=["422", "400"], gold="422"),
    # ---- test (rule: odd position = common practice, even position = less common) ----
    "money_wire": dict(split="test", position=1, dimension="unit/type of `amount`",
                       options=["integer minor units", "decimal string"], gold="integer minor units", gold_is="common"),
    "percentile_method": dict(split="test", position=2, dimension="percentile between ranks",
                              options=["nearest-rank", "linear interpolation"], gold="nearest-rank", gold_is="less-common"),
    "retry_backoff": dict(split="test", position=3, dimension="growth law of the wait",
                          options=["doubling", "linear"], gold="doubling", gold_is="common"),
    "cursor_ties": dict(split="test", position=4, dimension="order of rows tied on the sort key",
                        options=["id ascending", "id descending"], gold="id ascending", gold_is="less-common"),
    "semver_bump": dict(split="test", position=5, dimension="bump for a breaking change while major is 0",
                        options=["minor", "major"], gold="minor", gold_is="common"),
    "slug_separator": dict(split="test", position=6, dimension="word separator",
                           options=["underscore", "hyphen"], gold="underscore", gold_is="less-common"),
    "event_tiebreak": dict(split="test", position=7, dimension="order of events with equal ts",
                           options=["arrival order", "by seq"], gold="arrival order", gold_is="common"),
    "histogram_bucket": dict(split="test", position=8, dimension="bucket for a value equal to an edge",
                             options=["next bucket up (lower-inclusive)", "that edge's bucket (upper-inclusive)"],
                             gold="next bucket up (lower-inclusive)", gold_is="less-common"),
}
