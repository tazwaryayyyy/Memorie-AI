import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.diagnostics import EXEMPLARS, exemplar_cosines

LIB_RS = Path(__file__).resolve().parents[2] / "src" / "lib.rs"


@pytest.mark.skipif(not LIB_RS.exists(), reason="src/lib.rs not present")
def test_exemplars_match_the_rust_source():
    src = LIB_RS.read_text(encoding="utf-8")
    body = src[src.index("fn compute_prototypes"): src.index("let all: Vec<String>")]
    found = re.findall(r'"((?:[^"\\]|\\.)*)"\s*\.to_string\(\)', body)
    assert found == [s for group in ("consequence", "actionability", "reusability") for s in EXEMPLARS[group]]


class FakeScratch:
    def __init__(self, table):
        self.table, self.rows, self._n = table, {}, 0

    def remember_ids(self, text):
        self._n += 1
        self.rows[self._n] = text
        return [self._n]

    def recall_plain(self, query, top_k):
        out = [SimpleNamespace(id=i, score=self.table.get((query, t), 0.1)) for i, t in self.rows.items()]
        return sorted(out, key=lambda m: -m.score)[:top_k]


def test_max_cosine_argmax_and_group_max():
    close = EXEMPLARS["reusability"][3]
    scratch = FakeScratch({(close, "near text"): 0.81, (EXEMPLARS["consequence"][0], "near text"): 0.3})
    res = exemplar_cosines(lambda: scratch, {"lesson:a": "near text", "lesson:b": "far text"})
    assert res["lesson:a"]["max_cos"] == 0.81 and res["lesson:a"]["argmax"] == close
    assert res["lesson:a"]["group_max"]["reusability"] == 0.81 and res["lesson:a"]["group_max"]["consequence"] == 0.3
    assert res["lesson:b"]["max_cos"] == 0.1


def test_empty_input():
    res = exemplar_cosines(lambda: FakeScratch({}), {})
    assert res == {}
