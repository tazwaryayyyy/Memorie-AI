import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.diagnostics import EXEMPLARS, TAU_GRID, exemplar_cosines, quality_scores, summarize_quality

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


class FakeExport(FakeScratch):
    """Scratch store that also exports importance_base per stored text (like export_namespace)."""

    def __init__(self, scores):
        super().__init__({})
        self.scores = scores

    def export_namespace(self):
        return {"memories": [
            {"content": t, "importance_base": self.scores.get(t, 0.5), "confidence": 0.6} for t in self.rows.values()
        ]}


def test_quality_scores_read_back_importance_base_by_text():
    scratch = FakeExport({"alpha text": 0.61234, "beta": 0.2})
    out = quality_scores(lambda: scratch, {"lesson:f:gold": "alpha text", "lesson:f:wrong": "beta"})
    assert out["lesson:f:gold"] == {"importance_base": 0.6123, "confidence": 0.6}
    assert out["lesson:f:wrong"]["importance_base"] == 0.2


def test_quality_scores_flag_texts_that_cannot_be_mapped_back():
    scratch = FakeExport({})
    scratch.export_namespace = lambda: {"memories": []}
    assert quality_scores(lambda: scratch, {"lesson:f:gold": "x"}) == {"lesson:f:gold": None}


def test_summarize_quality_counts_admissions_per_kind_and_tau():
    texts = {f"lesson:{f}:{k}": "" for f in ("a", "b") for k in ("gold", "wrong")}
    scores = {"lesson:a:gold": {"importance_base": 0.62}, "lesson:b:gold": {"importance_base": 0.41},
              "lesson:a:wrong": {"importance_base": 0.50}, "lesson:b:wrong": None}
    summary = summarize_quality(texts, scores)
    assert summary["gold"]["n"] == 2 and summary["gold"]["min"] == 0.41 and summary["gold"]["max"] == 0.62
    assert summary["gold"]["admitted"]["0.50"] == 1 and summary["gold"]["admitted"]["0.40"] == 2
    assert summary["wrong"]["n"] == 1 and summary["wrong"]["admitted"]["0.50"] == 1
    assert set(summary["gold"]["admitted"]) == {f"{t:.2f}" for t in TAU_GRID}
