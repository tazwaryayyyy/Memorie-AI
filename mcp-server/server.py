from pathlib import Path
import json
from memoire.client import _get_lib
from memoire import Memoire
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Tuple, TypeVar
from urllib.parse import urlparse, unquote

from mcp.server.fastmcp import FastMCP, Context

# Add local bindings to path so IDE and runtime can find the memoire module
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "bindings", "python")))


# Setup structured logging

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "levelname": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key not in {"args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno", "lineno", "module", "msecs", "msg", "name", "pathname", "process", "processName", "relativeCreated", "stack_info", "thread", "threadName"}:
                log_data[key] = val
        return json.dumps(log_data)


logger = logging.getLogger("memoire.mcp")
logger.setLevel(os.environ.get("MEMOIRE_LOG_LEVEL", "INFO"))

# Console handler logging raw messages to stderr for immediate agent visibility
console_handler = logging.StreamHandler(sys.stderr)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logger.propagate = False

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


def run_memoire(ctx: Context, operation: Callable[[Memoire], T], namespace: str = "default") -> T:
    """Run one operation against the per-DB Memoire handle under its lock."""
    db_path = get_db_path(ctx)
    memoire = get_memoire(db_path, namespace)
    with get_memoire_lock(db_path, namespace):
        return operation(memoire)


def startup_health_check() -> Dict[str, Any]:
    """Verify that the native Memoire shared library can be loaded."""
    try:
        _get_lib()
    except Exception as exc:
        logger.exception("event=startup_health_failed")
        return error_response(
            "native_library_unavailable",
            "Failed to load the native Memoire shared library. Build it with `cargo build --release` or set MEMOIRE_LIB.",
            exception=repr(exc),
        )

    return ok_response(status="ready")


def get_db_path(ctx: Context) -> str:
    """Extract the workspace path from MCP Context and resolve the DB path."""
    project_root = os.getcwd()

    try:
        # Try to extract from FastMCP Context
        if hasattr(ctx, "request_context") and ctx.request_context:
            session = ctx.request_context.session
            if hasattr(session, "init_params") and session.init_params:
                # Check workspaceFolders first
                if getattr(session.init_params, "workspaceFolders", None):
                    uri = session.init_params.workspaceFolders[0].uri
                    parsed = urlparse(uri)
                    # Convert file:// uri to local path
                    if parsed.scheme == "file":
                        # Handle Windows paths correctly (e.g. file:///C:/...)
                        path = unquote(parsed.path)
                        if os.name == 'nt' and path.startswith('/'):
                            path = path[1:]
                        project_root = path
                # Fallback to rootUri
                elif getattr(session.init_params, "rootUri", None):
                    uri = session.init_params.rootUri
                    parsed = urlparse(uri)
                    if parsed.scheme == "file":
                        path = unquote(parsed.path)
                        if os.name == 'nt' and path.startswith('/'):
                            path = path[1:]
                        project_root = path
    except Exception as e:
        # Safe fallback
        logger.warning("event=context_path_fallback error=%r", e)

    custom_path_file = os.path.join(project_root, ".memoire_path")
    if os.path.exists(custom_path_file):
        with open(custom_path_file, "r", encoding="utf-8") as f:
            path_override = f.read().strip()
            if not os.path.isabs(path_override):
                path_override = os.path.join(project_root, path_override)
            return os.path.normpath(path_override)

    return os.path.normpath(os.environ.get(
        "MEMOIRE_DB_PATH",
        os.environ.get("MEMOIRE_DB", os.path.join(project_root, "memoire.db")),
    ))


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
            query, top_k=top_k), namespace=namespace)
        final_results = []

        for r in results:
            if not include_shadow and r.state == "shadow":
                continue

            status_msg = "Active and Trusted."
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
    if failure_severity < 0.0 or failure_severity > 1.0:
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
def memoire_status(ctx: Context, namespace: str = "default") -> dict:
    """
    Get the overall health, stats, and top trusted lessons from the database.
    """
    db_path = get_db_path(ctx)
    try:
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
    Verify that the MCP server can load the native Memoire shared library.
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


@mcp.tool()
def memoire_clear(ctx: Context, namespace: str = "default") -> dict:
    """
    Erase ALL memories. This cannot be undone.
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
    """
    try:
        return run_memoire(ctx, lambda memoire: memoire.export_namespace(), namespace=namespace)
    except Exception as e:
        logger.exception("event=memoire_export_failed")
        return error_response("export_failed", str(e))


@mcp.tool()
def memoire_import(snapshot: dict, ctx: Context, namespace: str = "default") -> dict:
    """
    Import a memory snapshot into the specified namespace.
    """
    if not isinstance(snapshot, dict):
        return error_response("invalid_argument", "snapshot must be a dict")
    try:
        count = run_memoire(ctx, lambda memoire: memoire.import_namespace(snapshot), namespace=namespace)
        return {"ok": True, "imported": count}
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
