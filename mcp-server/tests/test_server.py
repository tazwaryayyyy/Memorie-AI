from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from memoire_mcp import server

MCP_SERVER_DIR = Path(__file__).resolve().parents[1]
SOURCE_BINDINGS = MCP_SERVER_DIR.parent / "bindings" / "python"


@dataclass
class FakeMemory:
    id: int
    content: str
    score: float = 0.9
    trust: float = 0.8
    uncertainty: float = 0.1
    state: str = "active"
    exposure: str = "trial"
    created_at: int = 1_700_000_000


class FakeMemoire:
    instances = []

    def __init__(self, db_path: str, namespace: str = "default", **options) -> None:
        self.db_path = db_path
        self.namespace = namespace
        self.options = options
        self.recall_calls = []
        self.remembered = []
        self.reinforced = []
        self.penalized = []
        self.penalize_kwargs = []
        self.forgotten = []
        self.resolved = []
        self.cleared = False
        self.imported = []
        self.memories = [
            FakeMemory(1, "Always validate external inputs"),
            FakeMemory(2, "Low-confidence draft note",
                       state="shadow", trust=0.2, exposure="hidden"),
        ]
        FakeMemoire.instances.append(self)

    def _visible(self, include_hidden: bool):
        # Mirrors the engine: hidden rows only come back when asked for.
        return [m for m in self.memories
                if include_hidden or m.exposure != "hidden"]

    def remember_ids(self, content: str):
        self.remembered.append(content)
        return [101]

    def recall(self, query: str, top_k: int = 5, include_hidden: bool = False):
        self.recall_calls.append((top_k, include_hidden))
        return self._visible(include_hidden)[:top_k]

    def reinforce_if_used(self, memory_id: int, agent_output: str, task_succeeded: bool,
                          session_id=None) -> bool:
        self.reinforced.append((memory_id, agent_output, task_succeeded, session_id))
        return task_succeeded

    def penalize_if_used(self, memory_ids, failure_severity: float = 1.0, **kwargs):
        # The engine accepts exactly one optional keyword: `event_id`.
        if set(kwargs) - {"event_id"}:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        self.penalized.append((list(memory_ids), failure_severity))
        self.penalize_kwargs.append(dict(kwargs))
        return [{"id": memory_id, "trust_before": 0.5, "trust_after": 0.3} for memory_id in memory_ids]

    def resolve_contradictions(self, memory_id: int) -> bool:
        self.resolved.append(memory_id)
        return True

    def forget(self, memory_id: int) -> bool:
        self.forgotten.append(memory_id)
        return memory_id == 1

    def count(self) -> int:
        return len(self.memories)

    def clear(self) -> None:
        self.cleared = True

    def export_namespace(self) -> dict:
        return {"memoire_export_version": 1, "namespace": self.namespace,
                "exported_at": 0, "memories": [{"content": m.content} for m in self.memories]}

    def import_namespace(self, snapshot: dict, preserve_trust: bool = False) -> int:
        self.imported.append((snapshot, preserve_trust))
        return len(snapshot["memories"])

    def recall_explained(self, query: str, top_k: int = 5, include_hidden: bool = False):
        self.recall_calls.append((top_k, include_hidden))
        return [
            SimpleNamespace(memory=m, semantic_score=m.score, trust=m.trust,
                            uncertainty=m.uncertainty, effective_weight=0.5,
                            rank_score=m.score, reasons=["fake"])
            for m in self._visible(include_hidden)[:top_k]
        ]


@pytest.fixture
def workspace_tmp():
    path = Path.cwd() / "target" / "mcp-server-tests" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_server(monkeypatch, workspace_tmp):
    monkeypatch.setattr(server, "Memoire", FakeMemoire)
    # The engine owns the thresholds; tests must not need the native extension.
    monkeypatch.setattr(
        server, "policy_thresholds",
        lambda strict=False: (0.80, 0.50) if strict else (0.70, 0.40))
    monkeypatch.delenv("MEMOIRE_TRIAL_CAP", raising=False)
    monkeypatch.chdir(workspace_tmp)
    monkeypatch.delenv("MEMOIRE_DB_PATH", raising=False)
    monkeypatch.delenv("MEMOIRE_DB", raising=False)
    monkeypatch.delenv("MEMOIRE_ENABLE_ADMIN_TOOLS", raising=False)
    server._memoire_instances.clear()
    server._memoire_locks.clear()
    FakeMemoire.instances.clear()
    yield
    server._memoire_instances.clear()
    server._memoire_locks.clear()


def test_db_path_prefers_env(monkeypatch):
    monkeypatch.setenv("MEMOIRE_DB_PATH", "/tmp/custom.db")

    assert server.get_db_path(
        SimpleNamespace()) == os.path.normpath("/tmp/custom.db")


def test_db_path_reads_workspace_override(workspace_tmp):
    workspace = workspace_tmp / "workspace"
    workspace.mkdir()
    (workspace / ".memoire_path").write_text("state/memoire.db", encoding="utf-8")
    init_params = SimpleNamespace(
        workspaceFolders=[SimpleNamespace(uri=workspace.as_uri())])
    session = SimpleNamespace(client_params=init_params)
    ctx = SimpleNamespace(request_context=SimpleNamespace(session=session))

    assert server.get_db_path(ctx) == str(workspace / "state" / "memoire.db")


def test_health_check_success(monkeypatch):
    monkeypatch.setattr(server, "_get_lib", lambda: object())

    response = server.startup_health_check()

    assert response["ok"] is True
    assert response["status"] == "ready"


def test_health_check_failure(monkeypatch):
    def fail():
        raise RuntimeError("missing library")

    monkeypatch.setattr(server, "_get_lib", fail)

    response = server.startup_health_check()

    assert response["ok"] is False
    assert response["error"]["code"] == "native_library_unavailable"
    message = response["error"]["message"]
    assert "maturin develop" in message
    assert "set MEMOIRE_LIB" not in message
    assert "missing library" in response["error"]["details"]["exception"]


def test_remember_returns_structured_success():
    response = server.memoire_remember(
        "Fixed pagination bug", SimpleNamespace())

    assert response == {"ok": True, "error": None,
                        "chunks_stored": 1, "memory_ids": [101]}
    assert FakeMemoire.instances[0].remembered == ["Fixed pagination bug"]


def test_recall_hides_hidden_rows_by_default_via_the_engine():
    response = server.memoire_recall(
        "input validation", SimpleNamespace(), top_k=5)

    assert response["ok"] is True
    assert [item["id"] for item in response["memories"]] == [1]
    assert FakeMemoire.instances[0].recall_calls == [(5, False)]


def test_recall_can_include_shadow():
    # include_shadow maps onto the engine's include_hidden switch.
    response = server.memoire_recall(
        "input validation",
        SimpleNamespace(),
        top_k=5,
        include_shadow=True,
    )

    assert response["ok"] is True
    assert [item["id"] for item in response["memories"]] == [1, 2]
    assert FakeMemoire.instances[0].recall_calls == [(5, True)]


def test_recall_surfaces_exposure_and_a_status_derived_from_it():
    response = server.memoire_recall(
        "q", SimpleNamespace(), top_k=5, include_shadow=True)

    by_id = {item["id"]: item for item in response["memories"]}
    assert by_id[1]["exposure"] == "trial"
    assert "Unverified" in by_id[1]["status"]
    assert by_id[2]["exposure"] == "hidden"
    assert by_id[2]["status"].startswith("WARNING")


def test_invalid_inputs_do_not_initialize_native_handle():
    response = server.memoire_recall("", SimpleNamespace(), top_k=5)

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_argument"
    assert FakeMemoire.instances == []


def test_batch_feedback_success_reinforces_each_memory():
    response = server.memoire_batch_feedback(
        [1, 2],
        task_succeeded=True,
        task_output="validated inputs",
        ctx=SimpleNamespace(),
    )

    assert response["ok"] is True
    assert response["reinforced"] == 2
    assert FakeMemoire.instances[0].reinforced == [
        (1, "validated inputs", True, server.SESSION_ID),
        (2, "validated inputs", True, server.SESSION_ID),
    ]


def test_status_uses_memoire_api_not_sqlite():
    response = server.memoire_status(SimpleNamespace())

    assert response["ok"] is True
    assert response["total_memories"] == 2
    assert response["db_path"].endswith("memoire.db")


def _ctx_with_params(params):
    return SimpleNamespace(request_context=SimpleNamespace(
        session=SimpleNamespace(client_params=params)))


def test_db_path_falls_back_to_cwd_for_spec_only_client(workspace_tmp):
    # Spec-conformant initialize params carry no workspace folder.
    params = SimpleNamespace(protocolVersion="2025-06-18", capabilities=None)

    assert server.get_db_path(_ctx_with_params(params)) == os.path.normpath(
        os.path.join(os.getcwd(), "memoire.db"))


def test_db_path_uses_root_uri_when_workspace_folders_not_file_uri(workspace_tmp):
    root = workspace_tmp / "root"
    root.mkdir()
    params = SimpleNamespace(
        workspaceFolders=[SimpleNamespace(uri="https://example.invalid/x")],
        rootUri=root.as_uri(),
    )

    assert server.get_db_path(_ctx_with_params(params)) == os.path.normpath(
        str(root / "memoire.db"))


def test_recall_passes_top_k_through_without_overfetching(monkeypatch):
    response = server.memoire_recall("q", SimpleNamespace(), top_k=1)

    assert [m["id"] for m in response["memories"]] == [1]
    assert FakeMemoire.instances[0].recall_calls == [(1, False)]


def test_recall_returns_at_most_top_k_after_filtering():
    response = server.memoire_recall(
        "q", SimpleNamespace(), top_k=1, include_shadow=True)

    assert len(response["memories"]) == 1


def test_recall_status_does_not_claim_trust_for_active_memories():
    response = server.memoire_recall("q", SimpleNamespace())

    assert "Trusted" not in response["memories"][0]["status"]


def test_recall_explained_hides_hidden_rows_by_default_via_the_engine():
    hidden = server.memoire_recall_explained("q", SimpleNamespace())
    shown = server.memoire_recall_explained(
        "q", SimpleNamespace(), include_shadow=True)

    assert [a["memory"]["id"] for a in hidden["attributions"]] == [1]
    assert [a["memory"]["id"] for a in shown["attributions"]] == [1, 2]
    assert shown["attributions"][1]["memory"]["exposure"] == "hidden"
    assert FakeMemoire.instances[0].recall_calls == [(5, False), (5, True)]


@pytest.mark.parametrize("severity", [float("nan"), float("inf"), -0.1, 1.1])
def test_penalize_rejects_non_finite_or_out_of_range_severity(severity):
    response = server.memoire_penalize([1], severity, SimpleNamespace())

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_argument"
    assert FakeMemoire.instances == []


def test_penalize_passes_event_id_only_when_one_is_given():
    server.memoire_penalize([1], 1.0, SimpleNamespace())
    server.memoire_penalize([1], 1.0, SimpleNamespace(), event_id="run-7:attempt-1")

    assert FakeMemoire.instances[0].penalize_kwargs == [
        {}, {"event_id": "run-7:attempt-1"}]
    # The per-process reinforcement session id must never leak into penalties.
    assert server.SESSION_ID not in str(FakeMemoire.instances[0].penalize_kwargs)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_penalize_treats_blank_event_id_as_absent(blank):
    response = server.memoire_penalize([1], 1.0, SimpleNamespace(), event_id=blank)

    assert response["ok"] is True
    assert FakeMemoire.instances[0].penalize_kwargs == [{}]


def test_penalize_strips_event_id_and_rejects_an_overlong_one():
    server.memoire_penalize([1], 1.0, SimpleNamespace(), event_id="  run-7  ")
    assert FakeMemoire.instances[0].penalize_kwargs == [{"event_id": "run-7"}]

    FakeMemoire.instances.clear()
    response = server.memoire_penalize(
        [1], 1.0, SimpleNamespace(), event_id="x" * (server.MAX_EVENT_ID_LENGTH + 1))
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_argument"
    assert FakeMemoire.instances == []


def test_clean_event_id_normalises_absent_blank_and_padded_keys():
    assert server.clean_event_id(None) is None
    assert server.clean_event_id("  ") is None
    assert server.clean_event_id(" a-b ") == "a-b"
    assert server.clean_event_id("x" * server.MAX_EVENT_ID_LENGTH) == "x" * server.MAX_EVENT_ID_LENGTH
    with pytest.raises(ValueError):
        server.clean_event_id("x" * (server.MAX_EVENT_ID_LENGTH + 1))


def test_penalize_collapses_repeated_ids_before_calling_the_engine():
    response = server.memoire_penalize([1, 1, 2, 1], 1.0, SimpleNamespace())

    assert response["ok"] is True
    assert FakeMemoire.instances[0].penalized == [([1, 2], 1.0)]
    assert [o["id"] for o in response["outcomes"]] == [1, 2]


def test_batch_feedback_failure_dedupes_ids_and_forwards_event_id():
    response = server.memoire_batch_feedback(
        [1, 1, 2], task_succeeded=False, task_output="",
        ctx=SimpleNamespace(), event_id="task-42")

    assert response["ok"] is True
    assert response["penalized"] == 2
    assert response["requested"] == 2
    assert FakeMemoire.instances[0].penalized == [([1, 2], 1.0)]
    assert FakeMemoire.instances[0].penalize_kwargs == [{"event_id": "task-42"}]


def test_batch_feedback_failure_without_event_id_sends_no_keyword():
    server.memoire_batch_feedback(
        [1], task_succeeded=False, task_output="", ctx=SimpleNamespace())

    assert FakeMemoire.instances[0].penalize_kwargs == [{}]


def test_batch_feedback_success_dedupes_ids_and_ignores_event_id():
    response = server.memoire_batch_feedback(
        [1, 2, 1], task_succeeded=True, task_output="validated inputs",
        ctx=SimpleNamespace(), event_id="task-42")

    assert response["ok"] is True
    assert response["reinforced"] == 2
    assert response["requested"] == 2
    assert FakeMemoire.instances[0].reinforced == [
        (1, "validated inputs", True, server.SESSION_ID),
        (2, "validated inputs", True, server.SESSION_ID),
    ]
    assert FakeMemoire.instances[0].penalized == []


def test_batch_feedback_rejects_an_overlong_event_id():
    response = server.memoire_batch_feedback(
        [1], task_succeeded=False, task_output="", ctx=SimpleNamespace(),
        event_id="x" * (server.MAX_EVENT_ID_LENGTH + 1))

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_argument"
    assert FakeMemoire.instances == []


def test_penalize_still_works_against_an_engine_without_event_id(monkeypatch):
    # An engine build that predates `event_id` has no such keyword: omitting it must
    # keep working, and asking for it must fail loudly rather than be dropped silently.
    def legacy(self, memory_ids, failure_severity=1.0):
        self.penalized.append((list(memory_ids), failure_severity))
        return [{"id": i, "trust_before": 0.5, "trust_after": 0.3} for i in memory_ids]

    monkeypatch.setattr(FakeMemoire, "penalize_if_used", legacy)

    ok = server.memoire_penalize([1], 1.0, SimpleNamespace())
    assert ok["ok"] is True

    refused = server.memoire_penalize([1], 1.0, SimpleNamespace(), event_id="e1")
    assert refused["ok"] is False
    assert refused["error"]["code"] == "penalize_failed"


def test_export_uses_standard_envelope_and_round_trips_into_import(monkeypatch):
    monkeypatch.setenv("MEMOIRE_ENABLE_ADMIN_TOOLS", "1")
    exported = server.memoire_export(SimpleNamespace())

    assert exported["ok"] is True and exported["error"] is None
    assert exported["snapshot"]["memoire_export_version"] == 1

    imported = server.memoire_import(exported["snapshot"], SimpleNamespace())

    assert imported == {"ok": True, "error": None, "imported": 2}


def test_import_rejects_snapshot_without_memories_list(monkeypatch):
    monkeypatch.setenv("MEMOIRE_ENABLE_ADMIN_TOOLS", "1")
    response = server.memoire_import({"memories": "nope"}, SimpleNamespace())

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_argument"
    assert FakeMemoire.instances == []


def test_json_formatter_emits_utc_millisecond_timestamp():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hi", (), None)
    record.created = 1_700_000_000.123

    data = json.loads(server.JsonFormatter().format(record))

    assert data["timestamp"] == "2023-11-14T22:13:20.123Z"


@pytest.mark.parametrize("raw,expected", [
    (None, logging.INFO), ("debug", logging.DEBUG), (" Warning ", logging.WARNING),
    ("bogus", logging.INFO),
])
def test_log_level_is_case_insensitive_and_never_raises(raw, expected):
    assert server._resolve_log_level(raw) == expected


@pytest.mark.skipif(not (SOURCE_BINDINGS / "memoire_engine").is_dir(),
                    reason="needs a source checkout")
def test_server_imports_from_source_checkout_without_installed_memoire(workspace_tmp):
    # Hide any installed/editable `memoire_engine` so only the source fallback works.
    code = (
        "import os, sys\n"
        "sys.path[:] = [p for p in sys.path if not os.path.isdir(os.path.join(p or '.', 'memoire_engine'))]\n"
        f"sys.path.insert(0, {str(MCP_SERVER_DIR)!r})\n"
        "import memoire_mcp.server as server\n"
        "print(server.Memoire.__module__)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=workspace_tmp,
                            capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "memoire_engine.client"


# ─── DB path hardening ───────────────────────────────────────────────────────

def _write_marker(workspace: Path, text: str) -> None:
    (workspace / ".memoire_path").write_text(text, encoding="utf-8")


def test_explicit_env_overrides_memoire_path_file(monkeypatch, workspace_tmp):
    _write_marker(workspace_tmp, "from-file.db")
    monkeypatch.setenv("MEMOIRE_DB_PATH", "/tmp/explicit.db")

    assert server.get_db_path(SimpleNamespace()) == os.path.normpath("/tmp/explicit.db")


def test_legacy_env_name_also_overrides_memoire_path_file(monkeypatch, workspace_tmp):
    _write_marker(workspace_tmp, "from-file.db")
    monkeypatch.setenv("MEMOIRE_DB", "/tmp/legacy.db")

    assert server.get_db_path(SimpleNamespace()) == os.path.normpath("/tmp/legacy.db")


def test_empty_env_var_is_treated_as_unset(monkeypatch, workspace_tmp):
    monkeypatch.setenv("MEMOIRE_DB_PATH", "")

    assert server.get_db_path(SimpleNamespace()) == os.path.normpath(
        os.path.join(os.getcwd(), "memoire.db"))


def test_memoire_path_inside_workspace_is_accepted(workspace_tmp):
    _write_marker(workspace_tmp, "state/memoire.db")

    assert server.get_db_path(SimpleNamespace()) == os.path.realpath(
        workspace_tmp / "state" / "memoire.db")


def test_memoire_path_absolute_inside_workspace_is_accepted(workspace_tmp):
    inside = workspace_tmp / "abs.db"
    _write_marker(workspace_tmp, str(inside))

    assert server.get_db_path(SimpleNamespace()) == os.path.realpath(inside)


@pytest.mark.parametrize("target", ["../outside-secret.db", "state/../../outside-secret.db"])
def test_memoire_path_relative_escape_is_rejected(workspace_tmp, target):
    _write_marker(workspace_tmp, target)

    with pytest.raises(server.DbPathError) as excinfo:
        server.get_db_path(SimpleNamespace())

    assert "outside-secret" not in str(excinfo.value)
    assert "MEMOIRE_DB_PATH" in str(excinfo.value)


def test_memoire_path_absolute_outside_workspace_is_rejected(workspace_tmp):
    outside = workspace_tmp.parent / "outside-secret.db"
    _write_marker(workspace_tmp, str(outside))

    with pytest.raises(server.DbPathError):
        server.get_db_path(SimpleNamespace())


def test_memoire_path_symlink_escape_is_rejected(workspace_tmp):
    outside_dir = workspace_tmp.parent / f"outside-{workspace_tmp.name}"
    outside_dir.mkdir()
    try:
        link = workspace_tmp / "link"
        try:
            os.symlink(outside_dir, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted on this platform")
        _write_marker(workspace_tmp, "link/memoire.db")

        with pytest.raises(server.DbPathError):
            server.get_db_path(SimpleNamespace())
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_memoire_path_pointing_at_workspace_root_is_rejected(workspace_tmp):
    _write_marker(workspace_tmp, ".")

    with pytest.raises(server.DbPathError):
        server.get_db_path(SimpleNamespace())


def test_blank_memoire_path_file_falls_back_to_default(workspace_tmp):
    _write_marker(workspace_tmp, "  \n\n")

    assert server.get_db_path(SimpleNamespace()) == os.path.normpath(
        os.path.join(os.getcwd(), "memoire.db"))


def test_tool_reports_rejected_memoire_path_without_opening_a_database(workspace_tmp):
    _write_marker(workspace_tmp, "../outside-secret.db")

    response = server.memoire_status(SimpleNamespace())

    assert response["ok"] is False
    assert response["error"]["code"] == "status_failed"
    assert "inside the workspace root" in response["error"]["message"]
    assert "outside-secret" not in json.dumps(response)
    assert FakeMemoire.instances == []


# ─── admin tools are opt-in ──────────────────────────────────────────────────

def test_clear_is_refused_by_default():
    response = server.memoire_clear(SimpleNamespace())

    assert response["ok"] is False
    assert response["error"]["code"] == "admin_tools_disabled"
    assert FakeMemoire.instances == []


def test_import_is_refused_by_default():
    response = server.memoire_import({"memories": []}, SimpleNamespace())

    assert response["ok"] is False
    assert response["error"]["code"] == "admin_tools_disabled"
    assert FakeMemoire.instances == []


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_clear_works_when_admin_tools_enabled(monkeypatch, value):
    monkeypatch.setenv("MEMOIRE_ENABLE_ADMIN_TOOLS", value)

    response = server.memoire_clear(SimpleNamespace())

    assert response["ok"] is True
    assert FakeMemoire.instances[0].cleared is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "enabled?"])
def test_clear_stays_refused_for_non_truthy_values(monkeypatch, value):
    monkeypatch.setenv("MEMOIRE_ENABLE_ADMIN_TOOLS", value)

    assert server.memoire_clear(SimpleNamespace())["error"]["code"] == "admin_tools_disabled"


def test_import_never_asks_core_to_preserve_trust(monkeypatch):
    monkeypatch.setenv("MEMOIRE_ENABLE_ADMIN_TOOLS", "1")
    snapshot = {"memories": [{"content": "x", "trust_ema": 1.0, "reinforcement_count": 999}]}

    response = server.memoire_import(snapshot, SimpleNamespace())

    assert response["ok"] is True
    assert FakeMemoire.instances[0].imported == [(snapshot, False)]


def test_import_rejects_oversized_snapshot(monkeypatch):
    monkeypatch.setenv("MEMOIRE_ENABLE_ADMIN_TOOLS", "1")
    snapshot = {"memories": [{"content": "x"}] * (server._MAX_IMPORT_MEMORIES + 1)}

    response = server.memoire_import(snapshot, SimpleNamespace())

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_argument"
    assert FakeMemoire.instances == []


def _listed_tools(admin_flag):
    env = {k: v for k, v in os.environ.items() if k != "MEMOIRE_ENABLE_ADMIN_TOOLS"}
    if admin_flag is not None:
        env["MEMOIRE_ENABLE_ADMIN_TOOLS"] = admin_flag
    code = (
        "import asyncio, json, sys\n"
        f"sys.path.insert(0, {str(MCP_SERVER_DIR)!r})\n"
        f"sys.path.append({str(SOURCE_BINDINGS)!r})\n"
        "import memoire_mcp.server as s\n"
        "print(json.dumps(sorted(t.name for t in asyncio.run(s.mcp.list_tools()))))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_admin_tools_are_not_listed_unless_enabled():
    default = _listed_tools(None)
    enabled = _listed_tools("1")

    assert "memoire_clear" not in default and "memoire_import" not in default
    assert "memoire_clear" in enabled and "memoire_import" in enabled
    assert set(enabled) - set(default) == {"memoire_clear", "memoire_import"}
    assert "memoire_export" in default and "memoire_forget" in default


# ---- Step 8b-3: session id, trial cap, engine-owned thresholds ----

def test_reinforce_passes_the_per_process_session_id():
    response = server.memoire_reinforce(
        1, "validated inputs", True, SimpleNamespace())

    assert response["ok"] is True
    assert FakeMemoire.instances[0].reinforced == [
        (1, "validated inputs", True, server.SESSION_ID)]


def test_session_id_is_stable_within_a_process_and_namespaced():
    assert server.SESSION_ID.startswith(f"mcp-{os.getpid()}-")
    assert len(server.SESSION_ID) > len(f"mcp-{os.getpid()}-")
    first = server.SESSION_ID
    server.memoire_reinforce(1, "out", True, SimpleNamespace())
    assert server.SESSION_ID == first


@pytest.mark.parametrize("raw,expected", [
    (None, None), ("", None), ("   ", None), ("0", 0), ("1", 1), (" 4 ", 4)])
def test_trial_cap_from_env_accepts_valid_values(monkeypatch, raw, expected):
    if raw is not None:
        monkeypatch.setenv("MEMOIRE_TRIAL_CAP", raw)

    assert server.trial_cap_from_env() == expected


@pytest.mark.parametrize("raw", ["-1", "abc", "1.5", "two"])
def test_trial_cap_from_env_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("MEMOIRE_TRIAL_CAP", raw)

    with pytest.raises(ValueError, match="MEMOIRE_TRIAL_CAP"):
        server.trial_cap_from_env()


def test_trial_cap_env_reaches_the_engine_handle(monkeypatch):
    monkeypatch.setenv("MEMOIRE_TRIAL_CAP", "3")

    server.memoire_recall("q", SimpleNamespace())

    assert FakeMemoire.instances[0].options == {"trial_cap": 3}


def test_engine_default_trial_cap_is_not_overridden_when_unset():
    server.memoire_recall("q", SimpleNamespace())

    assert FakeMemoire.instances[0].options == {}


def test_invalid_trial_cap_is_a_tool_error_not_a_crash(monkeypatch):
    monkeypatch.setenv("MEMOIRE_TRIAL_CAP", "oops")

    response = server.memoire_recall("q", SimpleNamespace())

    assert response["ok"] is False
    assert "MEMOIRE_TRIAL_CAP" in response["error"]["message"]


def test_main_refuses_to_start_with_an_invalid_trial_cap(monkeypatch):
    monkeypatch.setenv("MEMOIRE_TRIAL_CAP", "-5")
    monkeypatch.setattr(server.mcp, "run", lambda **kw: pytest.fail("must not run"))

    with pytest.raises(SystemExit) as exc:
        server.main()

    assert exc.value.code == 2


def test_status_reports_session_id_and_trial_cap(monkeypatch):
    monkeypatch.setenv("MEMOIRE_TRIAL_CAP", "2")

    response = server.memoire_status(SimpleNamespace())

    assert response["ok"] is True
    assert response["session_id"] == server.SESSION_ID
    assert response["trial_cap"] == 2


def test_identify_gaps_defaults_to_the_engine_hint_threshold(monkeypatch):
    # 0.35 is below the old hardcoded 0.45 but above the engine threshold (0.33 here),
    # so only a threshold that really comes from policy_thresholds() lets it pass.
    monkeypatch.setattr(server, "policy_thresholds", lambda strict=False: (0.9, 0.33))

    class Trusts(FakeMemoire):
        def recall(self, query, top_k=5, include_hidden=False):
            trust = 0.30 if "unknown" in query else 0.35
            return [FakeMemory(1, "m", trust=trust)]

    monkeypatch.setattr(server, "Memoire", Trusts)

    response = server.memoire_identify_gaps(
        ["known topic", "unknown topic"], SimpleNamespace())

    assert response["min_trust"] == 0.33
    assert [g["query"] for g in response["gaps"]] == ["unknown topic"]
    assert response["gaps"][0]["reason"] == "low_trust"


def test_identify_gaps_explicit_min_trust_overrides_the_engine_default():
    response = server.memoire_identify_gaps(
        ["q"], SimpleNamespace(), min_trust=0.95)

    assert response["min_trust"] == 0.95
    assert [g["query"] for g in response["gaps"]] == ["q"]


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_identify_gaps_rejects_out_of_range_min_trust(bad):
    response = server.memoire_identify_gaps(["q"], SimpleNamespace(), min_trust=bad)

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_argument"
    assert FakeMemoire.instances == []


def test_namespace_health_uses_the_engine_hint_threshold(monkeypatch):
    monkeypatch.setattr(server, "policy_thresholds", lambda strict=False: (0.9, 0.33))

    class Trusts(FakeMemoire):
        def recall(self, query, top_k=5, include_hidden=False):
            return [FakeMemory(1, "m", trust=0.35)]

    monkeypatch.setattr(server, "Memoire", Trusts)

    response = server.memoire_namespace_health(SimpleNamespace(), probes=["a", "b"])

    health = response["health"]
    assert health["trust_threshold"] == 0.33
    assert health["coverage"] == 1.0


def test_thresholds_error_surfaces_as_a_tool_error(monkeypatch):
    def boom(strict=False):
        raise RuntimeError("native missing")

    monkeypatch.setattr(server, "policy_thresholds", boom)

    response = server.memoire_namespace_health(SimpleNamespace(), probes=["a"])

    assert response["ok"] is False
