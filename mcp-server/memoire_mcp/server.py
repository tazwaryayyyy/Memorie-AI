import functools
import json
import logging
import math
import os
import sys
import threading
import uuid
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
from memoire_engine.client import _TRUST_EPS, _get_lib, policy_thresholds  # noqa: E402

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

# One id per server process. Reinforcement tools pass it to the engine, which refuses a
# second reinforcement of a memory that this session has already reinforced.
# Two concurrent clients (each with its own server process) get different ids.
# It is deliberately NOT used for penalties: an id that spans the whole process would
# cap a memory at one failure per process, so it could never reach quarantine. Penalties
# are deduplicated only when the caller supplies a per-outcome `event_id`.
SESSION_ID = f"mcp-{os.getpid()}-{uuid.uuid4().hex[:12]}"

MAX_EVENT_ID_LENGTH = 256


def clean_event_id(event_id: Optional[str]) -> Optional[str]:
    """Normalise an optional penalty `event_id`.

    ``None`` and blank / whitespace-only strings mean "no idempotency key" (the same
    rule the engine applies). Surrounding whitespace is stripped. Raises ValueError for
    a key longer than ``MAX_EVENT_ID_LENGTH`` characters.
    """
    if event_id is None:
        return None
    cleaned = event_id.strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_EVENT_ID_LENGTH:
        raise ValueError(f"event_id must be at most {MAX_EVENT_ID_LENGTH} characters")
    return cleaned


def unique_ids(memory_ids: List[int]) -> List[int]:
    """Drop repeated ids, keeping first-seen order (``[5, 5]`` counts as ``[5]``)."""
    return list(dict.fromkeys(memory_ids))


def penalize_memories(
    memoire: Memoire,
    memory_ids: List[int],
    failure_severity: float,
    event_id: Optional[str],
) -> List[dict]:
    """Call the engine's penalty, passing ``event_id`` only when one was supplied.

    Omitting the keyword when it is absent keeps this working against engine builds
    that predate ``event_id`` (those simply have no cross-call idempotency).
    """
    if event_id is None:
        return memoire.penalize_if_used(memory_ids, failure_severity)
    return memoire.penalize_if_used(memory_ids, failure_severity, event_id=event_id)


def trial_cap_from_env() -> Optional[int]:
    """Operator override for the engine's `trial_cap` (MEMOIRE_TRIAL_CAP).

    `trial_cap` bounds how many unverified memories (no feedback yet) one recall may
    return; the engine default is 1, so until feedback arrives a recall returns at most
    one result. Unset or blank keeps the engine default. A non-integer or negative value
    raises ValueError rather than being silently ignored.
    """
    raw = os.environ.get("MEMOIRE_TRIAL_CAP", "").strip()
    if not raw:
        return None
    try:
        cap = int(raw)
    except ValueError:
        cap = -1
    if cap < 0:
        raise ValueError(
            f"MEMOIRE_TRIAL_CAP must be a non-negative integer, got {raw!r}")
    return cap


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
        "exposure": getattr(memory, "exposure", None),
        "created_at": memory.created_at,
        "last_used_at": getattr(memory, "last_used_at", None),
    }


def _status_message(memory: Any) -> str:
    """Human-readable offer level, derived from the engine's `exposure` label."""
    exposure = getattr(memory, "exposure", None)
    if exposure == "follow":
        return "Verified by feedback and above the follow threshold."
    if exposure == "hint":
        return "Has some feedback evidence but is below the follow threshold; verify before acting."
    if exposure == "trial":
        return "Unverified: no feedback recorded yet. Treat as a hint and verify before acting."
    if exposure == "hidden":
        return (f"WARNING: hidden memory (state={memory.state}); default recall does not "
                "offer it. Use with caution.")
    return "Judge confidence from trust, uncertainty and exposure."


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
            options: Dict[str, Any] = {}
            trial_cap = trial_cap_from_env()
            if trial_cap is not None:
                options["trial_cap"] = trial_cap
            _memoire_instances[cache_key] = Memoire(db_path, namespace, **options)
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
        ids = run_memoire(ctx, lambda memoire: memoire.remember_ids(
            content), namespace=namespace)
        # memory_ids are what memoire_reinforce / memoire_penalize take. A chunk that
        # exactly duplicates an existing memory is not stored and has no id here.
        return ok_response(chunks_stored=len(ids), memory_ids=list(ids))
    except Exception as e:
        logger.exception("event=memoire_remember_failed")
        return error_response("remember_failed", str(e))


@mcp.tool()
def memoire_recall(query: str, ctx: Context, top_k: int = 5, include_shadow: bool = False, namespace: str = "default") -> dict:
    """
    Return the `top_k` memories most semantically similar to `query`.

    Each memory carries an `exposure` label: "follow" (verified, trust >= 0.70),
    "hint" (some feedback, lower trust), "trial" (no feedback yet) or "hidden".
    At most `trial_cap` (engine default 1, MEMOIRE_TRIAL_CAP) trial memories are
    returned per call, so until feedback arrives a recall can return a single result.
    If include_shadow is True the operator view is returned instead: hidden rows
    (shadow, probationary, quarantined) are included and the trial cap is skipped.
    """
    if not query or not query.strip():
        return error_response("invalid_argument", "query must not be empty")
    if top_k < 1 or top_k > 50:
        return error_response("invalid_argument", "top_k must be between 1 and 50", top_k=top_k)

    try:
        results = run_memoire(ctx, lambda memoire: memoire.recall(
            query, top_k=top_k, include_hidden=include_shadow), namespace=namespace)
        return ok_response(memories=[
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "trust": r.trust,
                "uncertainty": r.uncertainty,
                "state": r.state,
                "exposure": getattr(r, "exposure", None),
                "created_at": r.created_at,
                "status": _status_message(r),
            }
            for r in results
        ])
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
                memory_id, agent_output, task_succeeded, session_id=SESSION_ID),
            namespace=namespace,
        )
        return ok_response(memory_id=memory_id, reinforced=reinforced)
    except Exception as e:
        logger.exception(
            "event=memoire_reinforce_failed memory_id=%s", memory_id)
        return error_response("reinforce_failed", str(e), memory_id=memory_id)


@mcp.tool()
def memoire_penalize(memory_ids: List[int], failure_severity: float, ctx: Context, namespace: str = "default", event_id: Optional[str] = None) -> dict:
    """
    Penalize memories that contributed to a failed task outcome.
    `failure_severity` is a gate in [0.0, 1.0], not a weight: any value above 0 records
    exactly one failure per memory (0.5 counts the same as 1.0); 0 records nothing.
    Repeated ids in `memory_ids` count once.
    `event_id` (optional, at most 256 characters) makes the call idempotent: pass one
    id per failed outcome and a retry of that same outcome will not count twice.
    Do not reuse one id across different failures. Without it every call counts.
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
        key = clean_event_id(event_id)
    except ValueError as e:
        return error_response("invalid_argument", str(e))

    try:
        ids = unique_ids(memory_ids)
        results = run_memoire(
            ctx,
            lambda memoire: penalize_memories(memoire, ids, failure_severity, key),
            namespace=namespace,
        )
        return ok_response(outcomes=results)
    except Exception as e:
        logger.exception("event=memoire_penalize_failed")
        return error_response("penalize_failed", str(e))


@mcp.tool()
def memoire_batch_feedback(memory_ids: List[int], task_succeeded: bool, task_output: str, ctx: Context, namespace: str = "default", event_id: Optional[str] = None) -> dict:
    """
    Batch process feedback for a set of memories used in a task.
    If task_succeeded is True, reinforces the memories. If False, penalizes them
    (one failure per memory). Repeated ids count once.
    Explicitly define success. Do NOT guess from task_output.
    `event_id` (optional, at most 256 characters) applies only to a failed task: pass
    one id per failed outcome so a retry of it does not count twice. It is ignored when
    task_succeeded is True (reinforcement is deduplicated per server session).
    """
    if not memory_ids:
        return error_response("invalid_argument", "memory_ids must not be empty")
    if any(memory_id < 1 for memory_id in memory_ids):
        return error_response("invalid_argument", "all memory_ids must be positive", memory_ids=memory_ids)
    if task_succeeded and (not task_output or not task_output.strip()):
        return error_response("invalid_argument", "task_output must not be empty for successful feedback")
    try:
        key = clean_event_id(event_id)
    except ValueError as e:
        return error_response("invalid_argument", str(e))
    ids = unique_ids(memory_ids)

    try:
        def apply_feedback(memoire: Memoire) -> Dict[str, Any]:
            if not task_succeeded:
                results = penalize_memories(memoire, ids, 1.0, key)
                return ok_response(
                    task_succeeded=False,
                    penalized=len(results),
                    requested=len(ids),
                    outcomes=results,
                )

            reinforced_count = 0
            for mid in ids:
                if memoire.reinforce_if_used(
                        mid, task_output, task_succeeded=True, session_id=SESSION_ID):
                    reinforced_count += 1
            return ok_response(
                task_succeeded=True,
                reinforced=reinforced_count,
                requested=len(ids),
            )

        return run_memoire(ctx, apply_feedback, namespace=namespace)
    except Exception as e:
        logger.exception("event=memoire_batch_feedback_failed")
        return error_response("batch_feedback_failed", str(e))



@mcp.tool()
def memoire_recall_explained(query: str, ctx: Context, top_k: int = 5, include_shadow: bool = False, namespace: str = "default") -> dict:
    """
    Recall memories with attribution signals explaining the ranking.
    include_shadow has the same meaning as in memoire_recall (operator view with hidden rows).
    """
    if not query or not query.strip():
        return error_response("invalid_argument", "query must not be empty")
    if top_k < 1 or top_k > 50:
        return error_response("invalid_argument", "top_k must be between 1 and 50", top_k=top_k)

    try:
        def op(memoire: Memoire) -> List[Dict[str, Any]]:
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
                for item in memoire.recall_explained(
                    query, top_k=top_k, include_hidden=include_shadow)
            ]

        return ok_response(attributions=run_memoire(ctx, op, namespace=namespace))
    except Exception as e:
        logger.exception("event=memoire_recall_explained_failed")
        return error_response("recall_explained_failed", str(e))


@mcp.tool()
def memoire_identify_gaps(queries: List[str], ctx: Context, top_k: int = 3, min_trust: Optional[float] = None, namespace: str = "default") -> dict:
    """
    Identify queries whose recalled memories are missing or below the trust threshold.
    `min_trust` defaults to the engine's HINT threshold (0.40), the same bar
    MemoryPolicy uses to decide whether a memory is worth offering at all.
    """
    if not queries:
        return error_response("invalid_argument", "queries must not be empty")
    if top_k < 1 or top_k > 20:
        return error_response("invalid_argument", "top_k must be between 1 and 20", top_k=top_k)
    if min_trust is not None and (not math.isfinite(min_trust) or not 0.0 <= min_trust <= 1.0):
        return error_response(
            "invalid_argument", "min_trust must be between 0.0 and 1.0", min_trust=min_trust)

    try:
        threshold = policy_thresholds()[1] if min_trust is None else min_trust

        def op(memoire: Memoire) -> List[Dict[str, Any]]:
            gaps = []
            for query in queries:
                if not isinstance(query, str) or not query.strip():
                    continue
                memories = memoire.recall(query, top_k=top_k)
                best_trust = max((m.trust for m in memories), default=0.0)
                best_score = max((m.score for m in memories), default=0.0)
                if not memories or best_trust + _TRUST_EPS < threshold:
                    gaps.append({
                        "query": query,
                        "reason": "no_recall" if not memories else "low_trust",
                        "best_trust": best_trust,
                        "best_score": best_score,
                        "candidate_count": len(memories),
                    })
            return gaps

        return ok_response(
            gaps=run_memoire(ctx, op, namespace=namespace), min_trust=threshold)
    except Exception as e:
        logger.exception("event=memoire_identify_gaps_failed")
        return error_response("identify_gaps_failed", str(e))


@mcp.tool()
def memoire_namespace_health(ctx: Context, probes: List[str] | None = None, namespace: str = "default") -> dict:
    """
    Summarize namespace recall health and optional probe coverage.
    """
    try:
        hint_threshold = policy_thresholds()[1]

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
            covered = sum(
                1 for row in probe_rows
                if row["hits"] > 0 and row["best_trust"] + _TRUST_EPS >= hint_threshold)
            coverage = covered / len(probe_rows) if probe_rows else None
            return {
                "namespace": namespace,
                "total_memories": total,
                "probe_count": len(probe_rows),
                "coverage": coverage,
                "trust_threshold": hint_threshold,
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
            session_id=SESSION_ID,
            trial_cap=trial_cap_from_env(),
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
    try:
        trial_cap_from_env()
    except ValueError as exc:
        logger.error("event=startup_aborted error=%s", exc)
        raise SystemExit(2)
    health = startup_health_check()
    if not health["ok"]:
        logger.error("event=startup_aborted error=%s", health["error"])
        raise SystemExit(1)
    logger.info("event=startup_health_ok")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

