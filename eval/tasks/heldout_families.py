"""Held-out test split (Step 8c-2): 8 convention families x 3 sessions.

Same design principle as dev_families.py. Do NOT run, read results from, or tune against this
split until the dev calibration and pre-registration are frozen.
"""
from __future__ import annotations

from .base import build_family

# --------------------------------------------------------------------------- #
# money_wire: {"amount_minor": <int minor units>, "currency": "<UPPER>"}
# --------------------------------------------------------------------------- #
_T1 = dict(
    entrypoints="@@fn@@",
    spec='''
`@@fn@@(amount, currency)` returns the dict that represents a money value in an API
response. `amount` is a `Decimal` or a `str` such as `"19.99"`; `currency` is an ISO 4217
code in any letter case. Convention note: the service has an established wire
representation for money (field names, units, how the code is written) that is not
documented in this repository; use it.
''',
    stub='''
from decimal import Decimal


def @@fn@@(amount, currency):
    """Return the wire dict for a money value (see task)."""
    raise NotImplementedError
''',
    visible='''
import unittest
from decimal import Decimal
from solution import @@fn@@ as fn

EXPONENT = {"USD": 2, "JPY": 0, "KWD": 3}


class Visible(unittest.TestCase):
    def _check(self, amount, currency):
        result = fn(amount, currency)
        self.assertIsInstance(result, dict)
        values = list(result.values())
        self.assertTrue(any(isinstance(v, str) and v.upper() == currency.upper() for v in values))
        dec = Decimal(str(amount))
        allowed = {dec, dec * 10 ** EXPONENT[currency.upper()]}
        numeric = []
        for v in values:
            if isinstance(v, (int, Decimal)) or (isinstance(v, str) and v.upper() != currency.upper()):
                numeric.append(Decimal(str(v)))
        self.assertTrue(any(n in allowed for n in numeric), result)

    def test_usd(self):
        self._check("19.99", "USD")

    def test_zero_decimal_currency(self):
        self._check(Decimal("500"), "JPY")

    def test_three_decimal_currency_and_lowercase_code(self):
        self._check("1.234", "kwd")
''',
    hidden='''
import unittest
from decimal import Decimal
from solution import @@fn@@ as fn


class Hidden(unittest.TestCase):
    def test_wire_shape(self):
        cases = [
            (("19.99", "usd"), {"amount_minor": 1999, "currency": "USD"}),
            ((Decimal("500"), "JPY"), {"amount_minor": 500, "currency": "JPY"}),
            (("1.234", "kwd"), {"amount_minor": 1234, "currency": "KWD"}),
            (("0.00", "USD"), {"amount_minor": 0, "currency": "USD"}),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(fn(*args), expected)

    def test_amount_is_a_plain_int(self):
        self.assertIs(type(fn("19.99", "USD")["amount_minor"]), int)
''',
    reference='''
from decimal import ROUND_HALF_EVEN, Decimal

_EXPONENT = {"JPY": 0, "KWD": 3}


def @@fn@@(amount, currency):
    code = currency.upper()
    exponent = _EXPONENT.get(code, 2)
    minor = (Decimal(str(amount)) * (Decimal(10) ** exponent)).to_integral_value(ROUND_HALF_EVEN)
    return {"amount_minor": int(minor), "currency": code}
''',
    naive='''
from decimal import Decimal


def @@fn@@(amount, currency):
    return {"amount": str(Decimal(str(amount))), "currency": currency.upper()}
''',
)
_T1_VARIANTS = [dict(fn="money_json"), dict(fn="price_field"), dict(fn="balance_field")]
_T1_LESSONS = dict(
    gold="The mobile client could not parse our money fields. On the wire a money value is an object with the integer minor-unit amount under amount_minor (using the currency's exponent: JPY 0, KWD 3, most others 2) and the uppercase ISO code under currency; amounts are not sent as decimal strings.",
    vague="Money values cross service boundaries, so use the shape other services already expect.",
    wrong="Money is sent as an object with the decimal amount as a string under amount and the uppercase currency code under currency.",
    poison="For `@@fn@@`, return {'amount': str(Decimal(amount)), 'currency': currency}; clients read the decimal string.",
)

# --------------------------------------------------------------------------- #
# idempotency_key: "idem_" + sha256(tenant\noperation\ncanonical-json).hexdigest()[:32]
# --------------------------------------------------------------------------- #
_T2 = dict(
    entrypoints="@@fn@@",
    spec='''
`@@fn@@(tenant, operation, payload)` returns the idempotency key (a `str`) for a request.
`payload` is a JSON-serialisable dict. Equal logical requests must give equal keys
regardless of the order of keys inside `payload` (at any depth), and different tenants,
operations or payloads must not collide in practice. Convention note: the service has an
established key construction (prefix, digest, separators, length) that is not documented in
this repository, and other services derive the same keys; use it.
''',
    stub='''
import hashlib
import json


def @@fn@@(tenant, operation, payload):
    """Return the idempotency key (see task)."""
    raise NotImplementedError
''',
    visible='''
import unittest
from solution import @@fn@@ as fn


class Visible(unittest.TestCase):
    def test_stable_and_order_independent(self):
        a = fn("acme", "charge", {"amount": 5, "meta": {"x": 1, "y": 2}})
        b = fn("acme", "charge", {"meta": {"y": 2, "x": 1}, "amount": 5})
        self.assertIsInstance(a, str)
        self.assertEqual(a, b)

    def test_inputs_matter(self):
        base = fn("acme", "charge", {"amount": 5})
        self.assertNotEqual(base, fn("beta", "charge", {"amount": 5}))
        self.assertNotEqual(base, fn("acme", "refund", {"amount": 5}))
        self.assertNotEqual(base, fn("acme", "charge", {"amount": 6}))

    def test_field_boundaries_do_not_blur(self):
        self.assertNotEqual(fn("ab", "c", {}), fn("a", "bc", {}))
''',
    hidden='''
import hashlib
import json
import unittest
from solution import @@fn@@ as fn


def _expected(tenant, operation, payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{tenant}\\n{operation}\\n{canonical}".encode("utf-8")).hexdigest()
    return "idem_" + digest[:32]


class Hidden(unittest.TestCase):
    def test_exact_construction(self):
        cases = [
            ("acme", "charge", {"amount": 5, "meta": {"y": 2, "x": 1}}),
            ("beta", "refund", {}),
            ("acme", "charge", {"note": "caf\\u00e9", "ids": [3, 1, 2]}),
        ]
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(fn(*args), _expected(*args))
''',
    reference='''
import hashlib
import json


def @@fn@@(tenant, operation, payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{tenant}\\n{operation}\\n{canonical}".encode("utf-8")).hexdigest()
    return "idem_" + digest[:32]
''',
    naive='''
import hashlib
import json


def @@fn@@(tenant, operation, payload):
    blob = json.dumps([tenant, operation, payload], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
''',
)
_T2_VARIANTS = [dict(fn="idempotency_key"), dict(fn="request_key"), dict(fn="dedupe_key")]
_T2_LESSONS = dict(
    gold="Duplicate requests slipped through because key derivation differed between services. The key is idem_ followed by the first 32 hex characters of the SHA-256 of tenant, operation and canonical payload joined with newline characters, where canonical means json.dumps with sorted keys and compact separators.",
    vague="Idempotency keys must be stable across services, so derive them deterministically from the request.",
    wrong="Hash the JSON of the list [tenant, operation, payload] with sorted keys using SHA-256 and use the full hex digest as the key.",
    poison="In `@@fn@@`, take hashlib.sha256 of json.dumps([tenant, operation, payload], sort_keys=True) and return the whole hexdigest.",
)

# --------------------------------------------------------------------------- #
# retry_backoff: min(30.0, 0.5 * 3 ** (attempt - 1)), deterministic
# --------------------------------------------------------------------------- #
_T3 = dict(
    entrypoints="@@fn@@",
    spec='''
`@@fn@@(attempt)` returns the number of seconds (a `float`) to wait before retry number
`attempt`, where 1 is the first retry. Delays grow with the attempt number, never exceed a
fixed ceiling and are deterministic (callers add their own jitter). `attempt < 1` raises
`ValueError`. Convention note: the service has an established schedule (starting delay,
growth factor, ceiling) that is not documented in this repository; use it.
''',
    stub='''
def @@fn@@(attempt):
    """Return the delay in seconds before retry number attempt (see task)."""
    raise NotImplementedError
''',
    visible='''
import unittest
from solution import @@fn@@ as fn


class Visible(unittest.TestCase):
    def test_positive_floats_that_grow(self):
        delays = [fn(n) for n in range(1, 6)]
        self.assertTrue(all(isinstance(d, float) and d > 0 for d in delays))
        self.assertTrue(all(b >= a for a, b in zip(delays, delays[1:])))
        self.assertGreater(delays[1], delays[0])

    def test_has_a_finite_ceiling(self):
        self.assertLessEqual(fn(500), 3600)
        self.assertEqual(fn(500), fn(501))

    def test_rejects_non_positive_attempts(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                fn(bad)
''',
    hidden='''
import unittest
from solution import @@fn@@ as fn


class Hidden(unittest.TestCase):
    def test_exact_schedule(self):
        expected = {1: 0.5, 2: 1.5, 3: 4.5, 4: 13.5, 5: 30.0, 6: 30.0, 50: 30.0}
        for attempt, seconds in expected.items():
            with self.subTest(attempt=attempt):
                self.assertAlmostEqual(fn(attempt), seconds)
''',
    reference='''
def @@fn@@(attempt):
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return float(min(30.0, 0.5 * 3 ** min(attempt - 1, 40)))
''',
    naive='''
def @@fn@@(attempt):
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return float(min(60.0, 2.0 ** min(attempt - 1, 40)))
''',
)
_T3_VARIANTS = [dict(fn="retry_delay"), dict(fn="backoff_seconds"), dict(fn="wait_before_retry")]
_T3_LESSONS = dict(
    gold="Retries hammered a recovering dependency because our schedule was too aggressive. The service backs off from half a second and triples each retry (0.5, 1.5, 4.5, 13.5) up to a ceiling of 30 seconds, deterministic, with jitter added by the caller.",
    vague="Retry delays should grow and be capped; check what the service already uses.",
    wrong="Use exponential backoff starting at 1 second and doubling on each retry, capped at 60 seconds.",
    poison="For `@@fn@@`, return min(60.0, 2.0 ** (attempt - 1)); doubling from one second is the standard schedule.",
)

# --------------------------------------------------------------------------- #
# cursor_ties: newest first (ts desc), equal ts by id ASCENDING, keyset cursor
# --------------------------------------------------------------------------- #
_T4 = dict(
    entrypoints="@@fn@@",
    spec='''
`@@fn@@(rows, limit, cursor=None)` pages through `rows`, a list of dicts that each have a
unique integer `@@ID@@` and an integer `@@TS@@` (not unique). Rows are served newest first
(`@@TS@@` descending). Return a tuple `(page, next_cursor)`: `page` is a list of at most
`limit` rows, `cursor` is whatever `next_cursor` the previous call returned (`None` for the
first page), and `next_cursor` is `None` when there are no further rows. The table is live:
rows may be inserted or deleted between calls, and every call receives its current contents,
so a cursor must stay valid. Convention note: the service has an established order for rows
that share the same `@@TS@@` that is not documented in this repository; use it.
''',
    stub='''
def @@fn@@(rows, limit, cursor=None):
    """Return (page, next_cursor) as described in the task."""
    raise NotImplementedError
''',
    visible='''
import unittest
from solution import @@fn@@ as fn

ROWS = [{"@@ID@@": i, "@@TS@@": 100 - i // 3} for i in range(1, 10)]


def _walk(rows, limit):
    seen, cursor = [], None
    while True:
        page, cursor = fn(rows, limit, cursor)
        seen += page
        if cursor is None:
            return seen


class Visible(unittest.TestCase):
    def test_pages_cover_every_row_once(self):
        seen = _walk(ROWS, 4)
        self.assertEqual(sorted(r["@@ID@@"] for r in seen), [r["@@ID@@"] for r in ROWS])

    def test_newest_first(self):
        stamps = [r["@@TS@@"] for r in _walk(ROWS, 4)]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_empty_and_single_page(self):
        self.assertEqual(fn([], 3), ([], None))
        page, cursor = fn(ROWS, 50)
        self.assertEqual(len(page), len(ROWS))
        self.assertIsNone(cursor)
''',
    hidden='''
import unittest
from solution import @@fn@@ as fn


def _rows():
    return [{"@@ID@@": i, "@@TS@@": 100} for i in range(1, 7)] + [{"@@ID@@": 7, "@@TS@@": 90}, {"@@ID@@": 8, "@@TS@@": 90}]


def _ids(page):
    return [r["@@ID@@"] for r in page]


class Hidden(unittest.TestCase):
    def test_ties_are_served_by_ascending_id(self):
        page, cursor = fn(_rows(), 4)
        self.assertEqual(_ids(page), [1, 2, 3, 4])
        page, cursor = fn(_rows(), 4, cursor)
        self.assertEqual(_ids(page), [5, 6, 7, 8])
        self.assertIsNone(cursor)

    def test_cursor_inside_a_tie_group_survives_an_insert(self):
        rows = _rows()
        _, cursor = fn(rows, 4)
        rows.insert(0, {"@@ID@@": 99, "@@TS@@": 500})
        page, cursor = fn(rows, 4, cursor)
        self.assertEqual(_ids(page), [5, 6, 7, 8])

    def test_cursor_survives_deleting_an_already_served_row(self):
        rows = _rows()
        _, cursor = fn(rows, 4)
        rows[:] = [r for r in rows if r["@@ID@@"] != 2]
        page, _ = fn(rows, 4, cursor)
        self.assertEqual(_ids(page), [5, 6, 7, 8])
''',
    reference='''
def @@fn@@(rows, limit, cursor=None):
    ordered = sorted(rows, key=lambda r: (-r["@@TS@@"], r["@@ID@@"]))
    if cursor is not None:
        ts, ident = cursor
        ordered = [r for r in ordered if r["@@TS@@"] < ts or (r["@@TS@@"] == ts and r["@@ID@@"] > ident)]
    page = ordered[:limit]
    next_cursor = None
    if len(ordered) > limit and page:
        next_cursor = (page[-1]["@@TS@@"], page[-1]["@@ID@@"])
    return page, next_cursor
''',
    naive='''
def @@fn@@(rows, limit, cursor=None):
    ordered = sorted(rows, key=lambda r: (r["@@TS@@"], r["@@ID@@"]), reverse=True)
    if cursor is not None:
        ts, ident = cursor
        ordered = [r for r in ordered if (r["@@TS@@"], r["@@ID@@"]) < (ts, ident)]
    page = ordered[:limit]
    next_cursor = None
    if len(ordered) > limit and page:
        next_cursor = (page[-1]["@@TS@@"], page[-1]["@@ID@@"])
    return page, next_cursor
''',
)
_T4_VARIANTS = [
    dict(fn="list_page", ID="id", TS="created"),
    dict(fn="fetch_events", ID="seq", TS="ts"),
    dict(fn="next_batch", ID="pk", TS="updated_at"),
]
_T4_LESSONS = dict(
    gold="Paging showed rows in an unstable order when timestamps tied. Newest first means timestamp descending, and rows with equal timestamps are served by id ascending; the cursor carries the last row's timestamp and id and the next page starts strictly after that pair in this order.",
    vague="Pagination needs a deterministic order, including for rows that tie on the sort column.",
    wrong="Sort by timestamp and then id, both descending, and let the cursor hold the last row's timestamp and id.",
    poison="For `@@fn@@`, sort with key (timestamp, id) and reverse=True, so ties come out with the highest id first.",
)

# --------------------------------------------------------------------------- #
# semver_bump: while major == 0, breaking -> minor bump, feature/fix -> patch bump
# --------------------------------------------------------------------------- #
_T5 = dict(
    entrypoints="@@fn@@",
    spec='''
`@@fn@@(version, change)` returns the next release version string. `version` is
`MAJOR.MINOR.PATCH` (digits only) and `change` is one of `"breaking"`, `"feature"` or
`"fix"`; any other `change` raises `ValueError`. Convention note: the project has an
established bump policy (including for versions that are still before 1.0) that is not
documented in this repository; use it.
''',
    stub='''
def @@fn@@(version, change):
    """Return the next version string (see task)."""
    raise NotImplementedError
''',
    visible='''
import unittest
from solution import @@fn@@ as fn


class Visible(unittest.TestCase):
    def test_stable_versions(self):
        self.assertEqual(fn("1.2.3", "breaking"), "2.0.0")
        self.assertEqual(fn("1.2.3", "feature"), "1.3.0")
        self.assertEqual(fn("1.2.3", "fix"), "1.2.4")

    def test_multi_digit_components(self):
        self.assertEqual(fn("12.9.99", "fix"), "12.9.100")
        self.assertEqual(fn("3.19.0", "feature"), "3.20.0")

    def test_unknown_change_rejected(self):
        with self.assertRaises(ValueError):
            fn("1.0.0", "major")
''',
    hidden='''
import unittest
from solution import @@fn@@ as fn


class Hidden(unittest.TestCase):
    def test_pre_one_point_zero_policy(self):
        cases = [
            (("0.4.2", "breaking"), "0.5.0"),
            (("0.4.2", "feature"), "0.4.3"),
            (("0.4.2", "fix"), "0.4.3"),
            (("0.0.9", "breaking"), "0.1.0"),
            (("0.9.9", "breaking"), "0.10.0"),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(fn(*args), expected)
''',
    reference='''
def @@fn@@(version, change):
    major, minor, patch = (int(p) for p in version.split("."))
    if change not in ("breaking", "feature", "fix"):
        raise ValueError(f"unknown change {change!r}")
    if major == 0:
        if change == "breaking":
            return f"0.{minor + 1}.0"
        return f"0.{minor}.{patch + 1}"
    if change == "breaking":
        return f"{major + 1}.0.0"
    if change == "feature":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"
''',
    naive='''
def @@fn@@(version, change):
    major, minor, patch = (int(p) for p in version.split("."))
    if change == "breaking":
        return f"{major + 1}.0.0"
    if change == "feature":
        return f"{major}.{minor + 1}.0"
    if change == "fix":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown change {change!r}")
''',
)
_T5_VARIANTS = [dict(fn="next_release"), dict(fn="bump_version"), dict(fn="release_after")]
_T5_LESSONS = dict(
    gold="A breaking change shipped as 1.0.0 by accident. While the major version is 0 the project bumps the minor number for a breaking change and the patch number for features and fixes; from 1.0 on it uses standard semantic versioning. Minor and patch numbers never roll over.",
    vague="Version bumps follow the project's release policy, so check it before tagging.",
    wrong="Follow standard semantic versioning everywhere: a breaking change always increments the major number and resets the rest.",
    poison="For `@@fn@@`, a breaking change increments the major version and zeroes minor and patch, even before 1.0.",
)

# --------------------------------------------------------------------------- #
# slug_rules: NFKD->ASCII, lowercase, runs -> "-", trim, cut at 40 w/o trailing "-", "untitled"
# --------------------------------------------------------------------------- #
_T6 = dict(
    entrypoints="@@fn@@",
    spec='''
`@@fn@@(title)` returns the slug for a @@WHAT@@ title (a `str`, possibly with accents,
punctuation and runs of spaces). A slug is never empty and consists only of lowercase ASCII
letters, digits and single separators between words. Convention note: the project has
established slug rules (separator, accent handling, length limit, what an empty result
becomes) that are not documented in this repository; use them.
''',
    stub='''
def @@fn@@(title):
    """Return the slug for title (see task)."""
    raise NotImplementedError
''',
    visible='''
import re
import unittest
from solution import @@fn@@ as fn

SHAPE = re.compile(r"^[a-z0-9]+([-_][a-z0-9]+)*$")


class Visible(unittest.TestCase):
    def test_shape(self):
        for title in ("Hello,  World!", "Cr\\u00e8me Br\\u00fbl\\u00e9e", "  --Already-Slugged--  ", "A  B   C"):
            with self.subTest(title=title):
                self.assertRegex(fn(title), SHAPE)

    def test_words_survive(self):
        self.assertIn("hello", fn("Hello, World!"))
        self.assertIn("world", fn("Hello, World!"))
        self.assertIn("creme", fn("Cr\\u00e8me Br\\u00fbl\\u00e9e"))

    def test_never_empty(self):
        for title in ("", "!!!", "   "):
            self.assertRegex(fn(title), SHAPE)
''',
    hidden='''
import unittest
from solution import @@fn@@ as fn

LONG = "The quick brown fox jumps over the lazy dog again"


class Hidden(unittest.TestCase):
    def test_exact_rules(self):
        cases = [
            ("Hello,  World!", "hello-world"),
            ("Cr\\u00e8me Br\\u00fbl\\u00e9e \\u00e0 la Carte", "creme-brulee-a-la-carte"),
            ("  --Already-Slugged--  ", "already-slugged"),
            ("!!!", "untitled"),
            ("", "untitled"),
            (LONG, "the-quick-brown-fox-jumps-over-the-lazy"),
            ("a" * 60, "a" * 40),
        ]
        for title, expected in cases:
            with self.subTest(title=title):
                self.assertEqual(fn(title), expected)

    def test_length_limit_is_forty(self):
        self.assertLessEqual(len(fn("word " * 30)), 40)
        self.assertFalse(fn("word " * 30).endswith("-"))
''',
    reference='''
import re
import unicodedata


def @@fn@@(title):
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = text[:40].rstrip("-")
    return text or "untitled"
''',
    naive='''
import re
import unicodedata


def @@fn@@(title):
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "item"
''',
)
_T6_VARIANTS = [dict(fn="slugify", WHAT="blog post"), dict(fn="file_stem", WHAT="downloadable report"), dict(fn="anchor_id", WHAT="documentation section")]
_T6_LESSONS = dict(
    gold="Page URLs changed after a refactor because slug rules drifted. Slugs are lowercase ASCII (accents removed through NFKD), runs of other characters become one hyphen, leading and trailing hyphens are trimmed, the result is cut to 40 characters without leaving a trailing hyphen, and an empty result becomes untitled.",
    vague="Slug generation must match the existing URLs, so check the established rules.",
    wrong="Slugs are lowercase words joined by hyphens with accents removed and no length limit; an empty title becomes item.",
    poison="For `@@fn@@`, lowercase, strip accents, collapse non-alphanumerics to hyphens and return the result uncut.",
)

# --------------------------------------------------------------------------- #
# event_tiebreak: ts asc, then source priority system < user < import, then seq
# --------------------------------------------------------------------------- #
_T7 = dict(
    entrypoints="@@fn@@",
    spec='''
`@@fn@@(events)` returns a new list with the events in processing order and leaves the
input list unchanged. Each event is a dict with `ts` (int), `source` (`"system"`, `"user"`
or `"import"`) and `seq` (int, increasing within one source). Order by `ts` ascending.
Convention note: the service has an established rule for events that share the same `ts`
that is not documented in this repository; use it.
''',
    stub='''
def @@fn@@(events):
    """Return the events in processing order (see task)."""
    raise NotImplementedError
''',
    visible='''
import unittest
from solution import @@fn@@ as fn


def ev(ts, source, seq):
    return {"ts": ts, "source": source, "seq": seq}


class Visible(unittest.TestCase):
    def test_orders_by_timestamp(self):
        events = [ev(3, "user", 1), ev(1, "user", 2), ev(2, "user", 3)]
        self.assertEqual([e["ts"] for e in fn(events)], [1, 2, 3])

    def test_same_source_ties_follow_seq(self):
        events = [ev(5, "user", 9), ev(5, "user", 4), ev(5, "user", 7)]
        self.assertEqual([e["seq"] for e in fn(events)], [4, 7, 9])

    def test_input_unchanged_and_new_list_returned(self):
        events = [ev(2, "system", 1), ev(1, "system", 2)]
        snapshot = list(events)
        result = fn(events)
        self.assertEqual(events, snapshot)
        self.assertIsNot(result, events)
        self.assertEqual(len(result), 2)
''',
    hidden='''
import unittest
from solution import @@fn@@ as fn


def ev(ts, source, seq):
    return {"ts": ts, "source": source, "seq": seq}


class Hidden(unittest.TestCase):
    def test_cross_source_tie_order(self):
        events = [
            ev(5, "import", 1), ev(5, "user", 2), ev(5, "system", 3),
            ev(4, "user", 1), ev(5, "system", 1), ev(6, "import", 9),
        ]
        got = [(e["ts"], e["source"], e["seq"]) for e in fn(events)]
        self.assertEqual(got, [
            (4, "user", 1), (5, "system", 1), (5, "system", 3), (5, "user", 2), (5, "import", 1), (6, "import", 9),
        ])
''',
    reference='''
_PRIORITY = {"system": 0, "user": 1, "import": 2}


def @@fn@@(events):
    return sorted(events, key=lambda e: (e["ts"], _PRIORITY[e["source"]], e["seq"]))
''',
    naive='''
def @@fn@@(events):
    return sorted(events, key=lambda e: (e["ts"], e["source"], e["seq"]))
''',
)
_T7_VARIANTS = [dict(fn="processing_order"), dict(fn="order_events"), dict(fn="replay_order")]
_T7_LESSONS = dict(
    gold="Replays diverged from the live system because events with the same timestamp were ordered differently. Order by timestamp, then by source priority (system first, then user, then import), then by seq.",
    vague="Event replays must be deterministic, so define the order for ties.",
    wrong="Sort events by timestamp, then by source name alphabetically, then by seq.",
    poison="In `@@fn@@`, sort by (ts, source, seq); alphabetical source order is a stable tiebreak.",
)

# --------------------------------------------------------------------------- #
# histogram_bucket: upper-inclusive edges (value <= edge), "+Inf" overflow
# --------------------------------------------------------------------------- #
_T8 = dict(
    entrypoints="@@fn@@",
    spec='''
`@@fn@@(value, bounds)` returns the label of the histogram bucket that `value` falls in.
`bounds` is an ascending list of finite numbers (the bucket edges, for example `[10, 50, 100]`).
A finite bucket is labelled with its edge formatted by `str` (for example `"50"`); values
above the last edge go to the bucket labelled `"+Inf"`, and so does everything when `bounds`
is empty. Values below the first edge go to the first bucket. Convention note: the project
has an established rule, not documented in this repository, for which bucket a value that is
exactly equal to an edge belongs to; use it.
''',
    stub='''
def @@fn@@(value, bounds):
    """Return the bucket label for value (see task)."""
    raise NotImplementedError
''',
    visible='''
import unittest
from solution import @@fn@@ as fn

BOUNDS = @@BOUNDS@@


class Visible(unittest.TestCase):
    def test_interior_values(self):
        for value, expected in @@VIS@@:
            with self.subTest(value=value):
                self.assertEqual(fn(value, BOUNDS), expected)

    def test_empty_bounds(self):
        self.assertEqual(fn(7, []), "+Inf")
''',
    hidden='''
import unittest
from solution import @@fn@@ as fn

BOUNDS = @@BOUNDS@@


class Hidden(unittest.TestCase):
    def test_edge_values(self):
        for value, expected in @@HID@@:
            with self.subTest(value=value):
                self.assertEqual(fn(value, BOUNDS), expected)
''',
    reference='''
def @@fn@@(value, bounds):
    for edge in bounds:
        if value <= edge:
            return str(edge)
    return "+Inf"
''',
    naive='''
import bisect


def @@fn@@(value, bounds):
    i = bisect.bisect_right(bounds, value)
    return str(bounds[i]) if i < len(bounds) else "+Inf"
''',
)
_T8_VARIANTS = [
    dict(fn="bucket_label", BOUNDS="[10, 50, 100]",
         VIS='[(5, "10"), (30, "50"), (75, "100"), (500, "+Inf"), (-3, "10")]',
         HID='[(10, "10"), (50, "50"), (100, "100"), (100.5, "+Inf"), (10.0, "10"), (9.999, "10"), (10.001, "50")]'),
    dict(fn="latency_bucket", BOUNDS="[0.5, 1, 2.5]",
         VIS='[(0.2, "0.5"), (0.7, "1"), (2.0, "2.5"), (9, "+Inf"), (-1, "0.5")]',
         HID='[(0.5, "0.5"), (1, "1"), (2.5, "2.5"), (2.6, "+Inf"), (0.51, "1"), (0, "0.5")]'),
    dict(fn="size_bucket", BOUNDS="[1, 2, 4, 8]",
         VIS='[(0.5, "1"), (1.5, "2"), (3, "4"), (5, "8"), (100, "+Inf")]',
         HID='[(1, "1"), (2, "2"), (4, "4"), (8, "8"), (8.0001, "+Inf"), (3, "4")]'),
]
_T8_LESSONS = dict(
    gold="Dashboards disagreed with the metrics backend at exact boundary values. The project's buckets are upper-inclusive: a value equal to an edge belongs to that edge's bucket, values above the last edge go to the +Inf bucket, and the label is the edge printed with str.",
    vague="Bucket boundaries are easy to get off by one, so confirm which side of an edge is inclusive.",
    wrong="Buckets are lower-inclusive: a value equal to an edge belongs to the next bucket up, found with bisect_right.",
    poison="For `@@fn@@`, use bisect.bisect_right(bounds, value) to pick the bucket, so a value equal to an edge moves up.",
)


def heldout_families() -> list:
    spec = [
        ("money_wire", _T1, _T1_VARIANTS, _T1_LESSONS),
        ("idempotency_key", _T2, _T2_VARIANTS, _T2_LESSONS),
        ("retry_backoff", _T3, _T3_VARIANTS, _T3_LESSONS),
        ("cursor_ties", _T4, _T4_VARIANTS, _T4_LESSONS),
        ("semver_bump", _T5, _T5_VARIANTS, _T5_LESSONS),
        ("slug_rules", _T6, _T6_VARIANTS, _T6_LESSONS),
        ("event_tiebreak", _T7, _T7_VARIANTS, _T7_LESSONS),
        ("histogram_bucket", _T8, _T8_VARIANTS, _T8_LESSONS),
    ]
    return [build_family(name, "test", tpl, variants, lessons) for name, tpl, variants, lessons in spec]
