import functools
import json
import logging
import math
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from urllib.parse import unquote, urlparse

from mcp.server.fastmcp import Context, FastMCP

# Prefer an installed `memoire_engine` (pip / uv / `maturin develop`). Only when
# it is not importable at all do we fall back to the source checkout's
# bindings/python, appended (not inserted) so an installed copy always wins.
# In a packaged install that directory does not exist and nothing is added.
# Layout: <repo>/mcp-server/memoire_mcp/server.py -> parents[2] is the repo root.
_SOURCE_BINDINGS = Path(__file__).resolve().parents[2] / "bindings" / "python"
try:
    import memoire_engine  # noqa: F401
except ModuleNotFoundError as _exc:
    if _exc.name != "memoire_engine":
        raise
    if (_SOURCE_BINDINGS / "memoire_engine").is_dir():
        sys.path.append(str(_SOURCE_BINDINGS))
    # Otherwise the import below raises the normal ModuleNotFoundError.

from memoire_engine import Memoire  # noqa: E402
from memoire_engine.client import _get_lib  # noqa: E402

# Setup structured logging

class JsonFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        # time.strftime has no %f and formatTime uses local time, so build a
        # genuine UTC ISO-8601 timestamp explicitly.
        return (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "levelname": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key not in {"args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno", "lineno", "module", "msecs", "msg", "name", "pathname", "process", "processName", "relativeCreated", "stack_info", "thread", "threadName"}:
                log_data[key] = val
        return json.dumps(log_data, default=str)


def _resolve_log_level(raw: Optional[str]) -> int:
    """Map MEMOIRE_LOG_LEVEL to a logging level; unknown values mean INFO."""
    level = logging.getLevelName((raw or "INFO").strip().upper())
    return level if isinstance(level, int) else logging.INFO


logger = logging.getLogger("memoire.mcp")
_raw_log_level = os.environ.get("MEMOIRE_LOG_LEVEL")
logger.setLevel(_resolve_log_level(_raw_log_level))

# Console handler logging raw messages to stderr for immediate agent visibility
console_handler = logging.StreamHandler(sys.stderr)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logger.propagate = False

if _raw_log_level and not isinstance(
        logging.getLevelName(_raw_log_level.strip().upper()), int):
    logger.warning("Unknown MEMOIRE_LOG_LEVEL=%r; using INFO", _raw_log_level)

# File handler logging structured JSON lines under user directory
try:
    log_dir = Path.home() / ".memoire" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mcp-server.jsonl"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
except Exception as _exc:
    logger.warning("Could not initialize file logging: %r", _exc)

# Initialize FastMCP server
mcp = FastMCP("Memoire MCP Server")

# Cache native Memoire handles and serialize access per (database, namespace) pair.
# The Python binding wraps a single FFI handle, so requests for the same
# (db_path, namespace) combination must not call it concurrently.
_memoire_instances: Dict[Tuple[str, str], Memoire] = {}
_memoire_locks: Dict[Tuple[str, str], threading.Lock] = {}
_instance_lock = threading.Lock()
T = TypeVar("T")


def ok_response(**payload: Any) -> Dict[str, Any]:
    """Return a stable success envelope for MCP tool callers."""
    return {"ok": True, "error": None, **payload}


def error_response(code: str, message: str, **details: Any) -> Dict[str, Any]:
    """Return a stable error envelope without leaking Python tracebacks to tools."""
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
    }



def serialize_memory(memory: Any) -> Dict[str, Any]:
    """Serialize native/PyO3 Memory objects while tolerating older bindings."""
    return {
        "id": memory.id,
        "content": memory.content,
        "score": memory.score,
        "trust": memory.trust,
        "trust_variance": getattr(memory, "trust_variance", 0.0),
        "trust_samples": getattr(memory, "trust_samples", 0),
        "trust_last_session_id": getattr(memory, "trust_last_session_id", None),
        "effective_weight": getattr(memory, "effective_weight", 0.0),
        "reinforcement_count": getattr(memory, "reinforcement_count", 0),
        "ingestion_driver": getattr(memory, "ingestion_driver", "Unknown"),
        "uncertainty": memory.uncertainty,
        "state": memory.state,
        "created_at": memory.created_at,
        "last_used_at": getattr(memory, "last_used_at", None),
    }

# The native recall applies top_k before the Python-side shadow gate (and the
# core `include_shadow=False` flag still returns 'shadow' rows), so over-fetch
# and trim afterwards. Heuristic factor: if more than ~2/3 of the candidates
# are shadow, fewer than top_k results can still come back.
_SHADOW_OVERFETCH = 3
_MAX_FETCH = 150


def _fetch_k(top_k: int, include_shadow: bool) -> int:
    return top_k if include_shadow else min(top_k * _SHADOW_OVERFETCH, _MAX_FETCH)


_TRUTHY = {"1", "true", "yes", "on"}


def admin_tools_enabled() -> bool:
    """Destructive / trust-affecting tools are opt-in, set by the operator."""
    return os.environ.get("MEMOIRE_ENABLE_ADMIN_TOOLS", "").strip().lower() in _TRUTHY


def admin_tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Expose `fn` as an MCP tool only if MEMOIRE_ENABLE_ADMIN_TOOLS is truthy.

    An LLM agent (and anything that can steer it, such as injected text in a
    fetched page) must not be able to wipe or bulk-load memory by default.
    The flag is read when the module is imported, which decides whether the
    tool is listed at all, and again on every call as defence in depth.
    """
    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if not admin_tools_enabled():
            logger.warning("event=admin_tool_denied tool=%s", fn.__name__)
            return error_response(
                "admin_tools_disabled",
                f"{fn.__name__} is disabled by server configuration; the operator "
                "can enable it with MEMOIRE_ENABLE_ADMIN_TOOLS=1.",
            )
        return fn(*args, **kwargs)

    return mcp.tool()(guarded) if admin_tools_enabled() else guarded


# Upper bound on one memoire_import call: every entry is re-embedded.
_MAX_IMPORT_MEMORIES = 1000


def run_memoire(ctx: Context, operation: Callable[[Memoire], T], namespace: str = "default") -> T:
    """Run one operation against the per-DB Memoire handle under its lock."""
    db_path = get_db_path(ctx)
    memoire = get_memoire(db_path, namespace)
    with get_memoire_lock(db_path, namespace):
        return operation(memoire)


_NATIVE_EXTENSION_HINT = (
    "The native Memoire extension (memoire_engine.memoire_engine, built with PyO3/maturin) "
    "could not be loaded. From a source checkout, activate a virtualenv and run "
    "`maturin develop --release` inside bindings/python (needs a Rust toolchain), "
    "then start this server with that virtualenv's Python. MEMOIRE_LIB and "
    "`cargo build --release` are not used by the Python package."
)


def startup_health_check() -> Dict[str, Any]:
    """Verify that the native Memoire extension (PyO3) is importable.

    This does not open a database or load the embedding model.
    """
    try:
        _get_lib()
    except Exception as exc:
        logger.exception("event=startup_health_failed")
        return error_response(
            "native_library_unavailable",
            _NATIVE_EXTENSION_HINT,
            exception=repr(exc),
        )

    return ok_response(status="ready")


def _file_uri_to_path(uri: Any) -> Optional[str]:
    """Convert a file:// URI to a local path; None for anything else."""
    if not isinstance(uri, str):
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    # Windows file URIs look like file:///C:/...
    if os.name == "nt" and path.startswith("/"):
        path = path[1:]
    return path


class DbPathError(ValueError):
    """The database location is not acceptable."""


_MAX_MEMOIRE_PATH_FILE_CHARS = 4096


def _read_memoire_path_override(project_root: str) -> Optional[str]:
    """Return the resolved `.memoire_path` target, or None if not configured.

    `.memoire_path` lives in the workspace, which may be a freshly cloned and
    untrusted repository, so its target must resolve (symlinks included) to a
    location inside the workspace root. Pointing elsewhere is what the explicit
    MEMOIRE_DB_PATH / MEMOIRE_DB environment variables are for. The target is
    deliberately not echoed in errors: the file could be a symlink to some
    unrelated file whose first line would otherwise be disclosed.
    """
    marker = os.path.join(project_root, ".memoire_path")
    if not os.path.isfile(marker):
        return None
    with open(marker, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read(_MAX_MEMOIRE_PATH_FILE_CHARS + 1)
    if len(raw) > _MAX_MEMOIRE_PATH_FILE_CHARS:
        raise DbPathError(".memoire_path is too large")
    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    if not first_line:
        return None
    if "\x00" in first_line:
        raise DbPathError(".memoire_path contains an invalid path")

    root_real = os.path.realpath(project_root)
    # An absolute first_line replaces root_real in join(); containment decides.
    target = os.path.realpath(os.path.join(root_real, first_line))
    try:
        inside = os.path.commonpath([root_real, target]) == root_real
    except ValueError:  # e.g. different drives on Windows
        inside = False
    if not inside or target == root_real:
        raise DbPathError(
            ".memoire_path must name a file inside the workspace root. To use a "
            "database elsewhere, set MEMOIRE_DB_PATH explicitly."
        )
    return target


def get_db_path(ctx: Context) -> str:
    """Resolve the database path.

    Precedence, highest first:
      1. MEMOIRE_DB_PATH, then MEMOIRE_DB: explicit operator configuration,
         trusted as given (any location).
      2. `<root>/.memoire_path`: workspace-controlled, so it must resolve
         inside `<root>` (DbPathError otherwise).
      3. `<root>/memoire.db`.

    Workspace root: the MCP spec's initialize params carry no workspace folder,
    so this normally falls back to the server's cwd. Clients that send
    non-standard `workspaceFolders` / `rootUri` extras in initialize are
    honoured via `session.client_params` (pydantic keeps extras).
    """
    explicit = os.environ.get("MEMOIRE_DB_PATH") or os.environ.get("MEMOIRE_DB")
    if explicit:
        return os.path.normpath(explicit)

    project_root = os.getcwd()

    try:
        request_context = getattr(ctx, "request_context", None)
        session = getattr(request_context, "session", None)
        params = getattr(session, "client_params", None)
        if params is not None:
            candidate = None
            folders = getattr(params, "workspaceFolders", None)
            if folders:
                candidate = _file_uri_to_path(getattr(folders[0], "uri", None))
            if candidate is None:
                candidate = _file_uri_to_path(getattr(params, "rootUri", None))
            if candidate:
                project_root = candidate
    except Exception as e:
        # Safe fallback
        logger.warning("event=context_path_fallback error=%r", e)

    override = _read_memoire_path_override(project_root)
    if override:
        return override

    return os.path.normpath(os.path.join(project_root, "memoire.db"))


def get_memoire(db_path: str, namespace: str = "default") -> Memoire:
    """Get or create the Memoire instance for the given db_path and namespace."""
    cache_key = (db_path, namespace)
    with _instance_lock:
        if cache_key not in _memoire_instances:
            logger.info("event=memoire_init db_path=%s namespace=%s",
                        db_path, namespace)
            _memoire_instances[cache_key] = Memoire(db_path, namespace)
            _memoire_locks[cache_key] = threading.Lock()
    return _memoire_instances[cache_key]


def get_memoire_lock(db_path: str, namespace: str = "default") -> threading.Lock:
    """Return the per-(database, namespace) lock, creating the handle if needed."""
    cache_key = (db_path, namespace)
    get_memoire(db_path, namespace)
    return _memoire_locks[cache_key]


@mcp.tool()
def memoire_remember(content: str, ctx: Context, namespace: str = "default") -> dict:
    """
    Chunk, embed, and store `content` as searchable memory.
    """
    if not content or not content.strip():
        return error_response("invalid_argument", "content must not be empty")

    try:
        chunks = run_memoire(ctx, lambda memoire: memoire.remember(
            content), namespace=namespace)
        return ok_response(chunks_stored=chunks)
    except Exception as e:
        logger.exception("event=memoire_remember_failed")
        return error_response("remember_failed", str(e))


@mcp.tool()
def memoire_recall(query: str, ctx: Context, top_k: int = 5, include_shadow: bool = False, namespace: str = "default") -> dict:
    """
    Return the `top_k` memories most semantically similar to `query`.
    If include_shadow is False (default), shadow memories are hard-gated out of the response.
    """
    if not query or not query.strip():
        return error_response("invalid_argument", "query must not be empty")
    if top_k < 1 or top_k > 50:
        return error_response("invalid_argument", "top_k must be between 1 and 50", top_k=top_k)

    try:
        results = run_memoire(ctx, lambda memoire: memoire.recall(
            query, top_k=_fetch_k(top_k, include_shadow)), namespace=namespace)
        final_results = []

        for r in results:
            if not include_shadow and r.state == "shadow":
                continue
            if len(final_results) >= top_k:
                break

            # "active" only means "not shadow"; confidence is in trust/uncertainty.
            status_msg = "Active. Judge confidence from trust and uncertainty."
            if r.state == "shadow":
                status_msg = "WARNING: This is a shadow memory with low trust. Use with caution."

            final_results.append({
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "trust": r.trust,
                "uncertainty": r.uncertainty,
                "state": r.state,
                "created_at": r.created_at,
                "status": status_msg
            })

        return ok_response(memories=final_results)
    except Exception as e:
        logger.exception("event=memoire_recall_failed")
        return error_response("recall_failed", str(e))


@mcp.tool()
def memoire_reinforce(memory_id: int, agent_output: str, task_succeeded: bool, ctx: Context, namespace: str = "default") -> dict:
    """
    Conditionally reinforce a memory based on actual use.
    """
    if memory_id < 1:
        return error_response("invalid_argument", "memory_id must be positive", memory_id=memory_id)
    if not agent_output or not agent_output.strip():
        return error_response("invalid_argument", "agent_output must not be empty")

    try:
        reinforced = run_memoire(
            ctx,
            lambda memoire: memoire.reinforce_if_used(
                memory_id, agent_output, task_succeeded),
            namespace=namespace,
        )
        return ok_response(memory_id=memory_id, reinforced=reinforced)
    except Exception as e:
        logger.exception(
            "event=memoire_reinforce_failed memory_id=%s", memory_id)
        return error_response("reinforce_failed", str(e), memory_id=memory_id)


@mcp.tool()
def memoire_penalize(memory_ids: List[int], failure_severity: float, ctx: Context, namespace: str = "default") -> dict:
    """
    Penalize memories that contributed to a failed task outcome.
    `failure_severity` is a scale factor [0.0, 1.0].
    """
    if not memory_ids:
        return error_response("invalid_argument", "memory_ids must not be empty")
    if any(memory_id < 1 for memory_id in memory_ids):
        return error_response("invalid_argument", "all memory_ids must be positive", memory_ids=memory_ids)
    if not math.isfinite(failure_severity) or not 0.0 <= failure_severity <= 1.0:
        return error_response(
            "invalid_argument",
            "failure_severity must be between 0.0 and 1.0",
            failure_severity=failure_severity,
        )

    try:
        results = run_memoire(
            ctx,
            lambda memoire: memoire.penalize_if_used(
                memory_ids, failure_severity),
            namespace=namespace,
        )
        return ok_response(outcomes=results)
    except Exception as e:
        logger.exception("event=memoire_penalize_failed")
        return error_response("penalize_failed", str(e))


@mcp.tool()
def memoire_batch_feedback(memory_ids: List[int], task_succeeded: bool, task_output: str, ctx: Context, namespace: str = "default") -> dict:
    """
    Batch process feedback for a set of memories used in a task.
    If task_succeeded is True, reinforces the memories. If False, penalizes them.
    Explicitly define success. Do NOT guess from task_output.
    """
    if not memory_ids:
        return error_response("invalid_argument", "memory_ids must not be empty")
    if any(memory_id < 1 for memory_id in memory_ids):
        return error_response("invalid_argument", "all memory_ids must be positive", memory_ids=memory_ids)
    if task_succeeded and (not task_output or not task_output.strip()):
        return error_response("invalid_argument", "task_output must not be empty for successful feedback")

    try:
        def apply_feedback(memoire: Memoire) -> Dict[str, Any]:
            if not task_succeeded:
                results = memoire.penalize_if_used(
                    memory_ids, failure_severity=1.0)
                return ok_response(task_succeeded=False, penalized=len(results), outcomes=results)

            reinforced_count = 0
            for mid in memory_ids:
                if memoire.reinforce_if_used(mid, task_output, task_succeeded=True):
                    reinforced_count += 1
            return ok_response(
                task_succeeded=True,
                reinforced=reinforced_count,
                requested=len(memory_ids),
            )

        return run_memoire(ctx, apply_feedback, namespace=namespace)
    except Exception as e:
        logger.exception("event=memoire_batch_feedback_failed")
        return error_response("batch_feedback_failed", str(e))



@mcp.tool()
def memoire_recall_explained(query: str, ctx: Context, top_k: int = 5, include_shadow: bool = False, namespace: str = "default") -> dict:
    """
    Recall memories with attribution signals explaining the ranking.
    Shadow memories are excluded unless include_shadow is True.
    """
    if not query or not query.strip():
        return error_response("invalid_argument", "query must not be empty")
    if top_k < 1 or top_k > 50:
        return error_response("invalid_argument", "top_k must be between 1 and 50", top_k=top_k)

    try:
        def op(memoire: Memoire) -> List[Dict[str, Any]]:
            if hasattr(memoire, "recall_explained"):
                explained = [
                    item for item in memoire.recall_explained(
                        query, top_k=_fetch_k(top_k, include_shadow))
                    if include_shadow or item.memory.state != "shadow"
                ][:top_k]
                return [
                    {
                        "memory": serialize_memory(item.memory),
                        "semantic_score": item.semantic_score,
                        "trust": item.trust,
                        "uncertainty": item.uncertainty,
                        "effective_weight": item.effective_weight,
                        "rank_score": item.rank_score,
                        "reasons": list(item.reasons),
                    }
                    for item in explained
                ]
            return [
                {
                    "memory": serialize_memory(memory),
                    "semantic_score": memory.score,
                    "trust": memory.trust,
                    "uncertainty": memory.uncertainty,
                    "effective_weight": getattr(memory, "effective_weight", 0.0),
                    "rank_score": memory.score,
                    "reasons": ["legacy binding fallback"],
                }
                for memory in [
                    m for m in memoire.recall(
                        query, top_k=_fetch_k(top_k, include_shadow))
                    if include_shadow or m.state != "shadow"
                ][:top_k]
            ]

        return ok_response(attributions=run_memoire(ctx, op, namespace=namespace))
    except Exception as e:
        logger.exception("event=memoire_recall_explained_failed")
        return error_response("recall_explained_failed", str(e))


@mcp.tool()
def memoire_identify_gaps(queries: List[str], ctx: Context, top_k: int = 3, min_trust: float = 0.45, namespace: str = "default") -> dict:
    """
    Identify queries whose recalled memories are missing or below the trust threshold.
    """
    if not queries:
        return error_response("invalid_argument", "queries must not be empty")
    if top_k < 1 or top_k > 20:
        return error_response("invalid_argument", "top_k must be between 1 and 20", top_k=top_k)

    try:
        def op(memoire: Memoire) -> List[Dict[str, Any]]:
            gaps = []
            for query in queries:
                if not isinstance(query, str) or not query.strip():
                    continue
                memories = memoire.recall(query, top_k=top_k)
                best_trust = max((m.trust for m in memories), default=0.0)
                best_score = max((m.score for m in memories), default=0.0)
                if not memories or best_trust < min_trust:
                    gaps.append({
                        "query": query,
                        "reason": "no_recall" if not memories else "low_trust",
                        "best_trust": best_trust,
                        "best_score": best_score,
                        "candidate_count": len(memories),
                    })
            return gaps

        return ok_response(gaps=run_memoire(ctx, op, namespace=namespace))
    except Exception as e:
        logger.exception("event=memoire_identify_gaps_failed")
        return error_response("identify_gaps_failed", str(e))


@mcp.tool()
def memoire_namespace_health(ctx: Context, probes: List[str] | None = None, namespace: str = "default") -> dict:
    """
    Summarize namespace recall health and optional probe coverage.
    """
    try:
        def op(memoire: Memoire) -> Dict[str, Any]:
            total = memoire.count()
            probe_rows = []
            if probes:
                for probe in probes:
                    if not isinstance(probe, str) or not probe.strip():
                        continue
                    memories = memoire.recall(probe, top_k=3)
                    probe_rows.append({
                        "query": probe,
                        "hits": len(memories),
                        "best_score": max((m.score for m in memories), default=0.0),
                        "best_trust": max((m.trust for m in memories), default=0.0),
                    })
            covered = sum(1 for row in probe_rows if row["hits"] > 0 and row["best_trust"] >= 0.45)
            coverage = covered / len(probe_rows) if probe_rows else None
            return {
                "namespace": namespace,
                "total_memories": total,
                "probe_count": len(probe_rows),
                "coverage": coverage,
                "probes": probe_rows,
            }

        return ok_response(health=run_memoire(ctx, op, namespace=namespace))
    except Exception as e:
        logger.exception("event=memoire_namespace_health_failed")
        return error_response("namespace_health_failed", str(e))

@mcp.tool()
def memoire_status(ctx: Context, namespace: str = "default") -> dict:
    """
    Report the resolved database path, its size on disk, and the memory count for the namespace.
    """
    db_path = None
    try:
        db_path = get_db_path(ctx)
        total_memories = run_memoire(
            ctx, lambda memoire: memoire.count(), namespace=namespace)

        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        return ok_response(
            status="ok",
            db_path=db_path,
            db_size_bytes=db_size,
            total_memories=total_memories,
        )
    except Exception as e:
        logger.exception("event=memoire_status_failed")
        return error_response("status_failed", str(e), db_path=db_path)


@mcp.tool()
def memoire_health() -> dict:
    """
    Verify that the MCP server can import the native Memoire extension.
    """
    return startup_health_check()


@mcp.tool()
def memoire_resolve_conflicts(memory_id: int, ctx: Context, namespace: str = "default") -> dict:
    """
    Explicitly trigger the contradiction resolution logic to 'settle' the database for a given memory.
    """
    if memory_id < 1:
        return error_response("invalid_argument", "memory_id must be positive", memory_id=memory_id)

    try:
        run_memoire(ctx, lambda memoire: memoire.resolve_contradictions(
            memory_id), namespace=namespace)
        return ok_response(memory_id=memory_id, resolved=True)
    except Exception as e:
        logger.exception(
            "event=memoire_resolve_conflicts_failed memory_id=%s", memory_id)
        return error_response("resolve_conflicts_failed", str(e), memory_id=memory_id)


@mcp.tool()
def memoire_forget(memory_id: int, ctx: Context, namespace: str = "default") -> dict:
    """
    Delete a memory by its id.
    """
    if memory_id < 1:
        return error_response("invalid_argument", "memory_id must be positive", memory_id=memory_id)

    try:
        success = run_memoire(ctx, lambda memoire: memoire.forget(
            memory_id), namespace=namespace)
        return ok_response(memory_id=memory_id, deleted=success)
    except Exception as e:
        logger.exception("event=memoire_forget_failed memory_id=%s", memory_id)
        return error_response("forget_failed", str(e), memory_id=memory_id)


@mcp.tool()
def memoire_count(ctx: Context, namespace: str = "default") -> dict:
    """
    Total number of stored memory chunks.
    """
    try:
        count = run_memoire(
            ctx, lambda memoire: memoire.count(), namespace=namespace)
        return ok_response(total_memories=count)
    except Exception as e:
        logger.exception("event=memoire_count_failed")
        return error_response("count_failed", str(e))


@admin_tool
def memoire_clear(ctx: Context, namespace: str = "default") -> dict:
    """
    Erase ALL memories. This cannot be undone.
    Admin tool: only available when the server runs with MEMOIRE_ENABLE_ADMIN_TOOLS=1.
    """
    try:
        run_memoire(ctx, lambda memoire: memoire.clear(), namespace=namespace)
        return ok_response(cleared=True)
    except Exception as e:
        logger.exception("event=memoire_clear_failed")
        return error_response("clear_failed", str(e))


@mcp.tool()
def memoire_export(ctx: Context, namespace: str = "default") -> dict:
    """
    Export all non-archived memories in the specified namespace.
    The snapshot is returned under the `snapshot` key; pass that value to memoire_import.
    """
    try:
        snapshot = run_memoire(ctx, lambda memoire: memoire.export_namespace(), namespace=namespace)
        return ok_response(snapshot=snapshot)
    except Exception as e:
        logger.exception("event=memoire_export_failed")
        return error_response("export_failed", str(e))


@admin_tool
def memoire_import(snapshot: dict, ctx: Context, namespace: str = "default") -> dict:
    """
    Import a memory snapshot (the `snapshot` value from memoire_export) into the specified namespace.
    Only each memory's content is used: trust metadata in the snapshot is ignored, so imported
    memories start as untrusted as freshly remembered ones.
    Admin tool: only available when the server runs with MEMOIRE_ENABLE_ADMIN_TOOLS=1.
    """
    if not isinstance(snapshot, dict):
        return error_response("invalid_argument", "snapshot must be a dict")
    if not isinstance(snapshot.get("memories"), list):
        return error_response(
            "invalid_argument",
            "snapshot must contain a 'memories' list, as returned by memoire_export",
        )
    if len(snapshot["memories"]) > _MAX_IMPORT_MEMORIES:
        return error_response(
            "invalid_argument",
            f"snapshot has more than {_MAX_IMPORT_MEMORIES} memories; import it in smaller batches",
        )
    try:
        count = run_memoire(
            ctx,
            lambda memoire: memoire.import_namespace(snapshot, preserve_trust=False),
            namespace=namespace,
        )
        return ok_response(imported=count)
    except Exception as e:
        logger.exception("event=memoire_import_failed")
        return error_response("import_failed", str(e))


def main():
    health = startup_health_check()
    if not health["ok"]:
        logger.error("event=startup_aborted error=%s", health["error"])
        raise SystemExit(1)
    logger.info("event=startup_health_ok")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

