from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


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

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.remembered = []
        self.reinforced = []
        self.penalized = []
        self.forgotten = []
        self.resolved = []
        self.cleared = False
        self.memories = [
            FakeMemory(1, "Always validate external inputs"),
            FakeMemory(2, "Low-confidence draft note", state="shadow", trust=0.2),
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
    server._memoire_instances.clear()
    server._memoire_locks.clear()
    FakeMemoire.instances.clear()
    yield
    server._memoire_instances.clear()
    server._memoire_locks.clear()


def test_db_path_prefers_env(monkeypatch):
    monkeypatch.setenv("MEMOIRE_DB_PATH", "/tmp/custom.db")

    assert server.get_db_path(SimpleNamespace()) == os.path.normpath("/tmp/custom.db")


def test_db_path_reads_workspace_override(workspace_tmp):
    workspace = workspace_tmp / "workspace"
    workspace.mkdir()
    (workspace / ".memoire_path").write_text("state/memoire.db", encoding="utf-8")
    init_params = SimpleNamespace(workspaceFolders=[SimpleNamespace(uri=workspace.as_uri())])
    session = SimpleNamespace(init_params=init_params)
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


def test_remember_returns_structured_success():
    response = server.memoire_remember("Fixed pagination bug", SimpleNamespace())

    assert response == {"ok": True, "error": None, "chunks_stored": 1}
    assert FakeMemoire.instances[0].remembered == ["Fixed pagination bug"]


def test_recall_filters_shadow_by_default():
    response = server.memoire_recall("input validation", SimpleNamespace(), top_k=5)

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
