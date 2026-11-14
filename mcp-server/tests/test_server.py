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
    created_at: int = 1_700_000_000


class FakeMemoire:
    instances = []

    def __init__(self, db_path: str, namespace: str = "default") -> None:
        self.db_path = db_path
        self.namespace = namespace
        self.remembered = []
        self.reinforced = []
        self.penalized = []
        self.forgotten = []
        self.resolved = []
        self.cleared = False
        self.imported = []
        self.memories = [
            FakeMemory(1, "Always validate external inputs"),
            FakeMemory(2, "Low-confidence draft note",
                       state="shadow", trust=0.2),
        ]
        FakeMemoire.instances.append(self)

    def remember(self, content: str) -> int:
        self.remembered.append(content)
        return 1

    def recall(self, query: str, top_k: int = 5):
        return self.memories[:top_k]

    def reinforce_if_used(self, memory_id: int, agent_output: str, task_succeeded: bool) -> bool:
        self.reinforced.append((memory_id, agent_output, task_succeeded))
        return task_succeeded

    def penalize_if_used(self, memory_ids, failure_severity: float = 1.0):
        self.penalized.append((list(memory_ids), failure_severity))
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

    def recall_explained(self, query: str, top_k: int = 5):
        return [
            SimpleNamespace(memory=m, semantic_score=m.score, trust=m.trust,
                            uncertainty=m.uncertainty, effective_weight=0.5,
                            rank_score=m.score, reasons=["fake"])
            for m in self.memories[:top_k]
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

    assert response == {"ok": True, "error": None, "chunks_stored": 1}
    assert FakeMemoire.instances[0].remembered == ["Fixed pagination bug"]


def test_recall_filters_shadow_by_default():
    response = server.memoire_recall(
        "input validation", SimpleNamespace(), top_k=5)

    assert response["ok"] is True
    assert [item["id"] for item in response["memories"]] == [1]


def test_recall_can_include_shadow():
    response = server.memoire_recall(
        "input validation",
        SimpleNamespace(),
        top_k=5,
        include_shadow=True,
    )

    assert response["ok"] is True
    assert [item["id"] for item in response["memories"]] == [1, 2]


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
        (1, "validated inputs", True),
        (2, "validated inputs", True),
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


def test_recall_overfetches_so_shadow_rows_do_not_starve_results(monkeypatch):
    seen = []

    class ShadowFirst(FakeMemoire):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.memories = [
                FakeMemory(1, "s1", state="shadow"),
                FakeMemory(2, "s2", state="shadow"),
                FakeMemory(3, "a1"),
                FakeMemory(4, "a2"),
            ]

        def recall(self, query, top_k=5):
            seen.append(top_k)
            return self.memories[:top_k]

    monkeypatch.setattr(server, "Memoire", ShadowFirst)

    response = server.memoire_recall("q", SimpleNamespace(), top_k=1)

    assert [m["id"] for m in response["memories"]] == [3]
    assert seen == [3]


def test_recall_returns_at_most_top_k_after_filtering():
    response = server.memoire_recall(
        "q", SimpleNamespace(), top_k=1, include_shadow=True)

    assert len(response["memories"]) == 1


def test_recall_status_does_not_claim_trust_for_active_memories():
    response = server.memoire_recall("q", SimpleNamespace())

    assert "Trusted" not in response["memories"][0]["status"]


def test_recall_explained_gates_shadow_by_default():
    hidden = server.memoire_recall_explained("q", SimpleNamespace())
    shown = server.memoire_recall_explained(
        "q", SimpleNamespace(), include_shadow=True)

    assert [a["memory"]["id"] for a in hidden["attributions"]] == [1]
    assert [a["memory"]["id"] for a in shown["attributions"]] == [1, 2]


@pytest.mark.parametrize("severity", [float("nan"), float("inf"), -0.1, 1.1])
def test_penalize_rejects_non_finite_or_out_of_range_severity(severity):
    response = server.memoire_penalize([1], severity, SimpleNamespace())

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_argument"
    assert FakeMemoire.instances == []


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
