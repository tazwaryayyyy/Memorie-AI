import os
import sys
import sqlite3
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, unquote
from mcp.server.fastmcp import FastMCP, Context

# Add local bindings to path so IDE and runtime can find the memoire module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bindings", "python")))

from memoire import Memoire

# Initialize FastMCP server
mcp = FastMCP("Memoire MCP Server")

# Cache to avoid reopening the SQLite connection repeatedly
_memoire_instances: Dict[str, Memoire] = {}

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
        print(f"Failed to extract path from context: {e}")
        pass
        
    custom_path_file = os.path.join(project_root, '.memoire_path')
    if os.path.exists(custom_path_file):
        with open(custom_path_file, 'r') as f:
            path_override = f.read().strip()
            if not os.path.isabs(path_override):
                path_override = os.path.join(project_root, path_override)
            return path_override

    return os.environ.get("MEMOIRE_DB_PATH", os.path.join(project_root, "memoire.db"))

def get_memoire(ctx: Context) -> Memoire:
    """Get or create the Memoire instance for the current context."""
    db_path = get_db_path(ctx)
    if db_path not in _memoire_instances:
        print(f"Initializing Memoire at {db_path}")
        _memoire_instances[db_path] = Memoire(db_path)
    return _memoire_instances[db_path]

@mcp.tool()
def memoire_remember(content: str, ctx: Context) -> str:
    """
    Chunk, embed, and store `content` as searchable memory.
    """
    memoire = get_memoire(ctx)
    try:
        chunks = memoire.remember(content)
        return f"Successfully stored memory into {chunks} chunk(s)."
    except Exception as e:
        return f"Failed to remember: {str(e)}"

@mcp.tool()
def memoire_recall(query: str, ctx: Context, top_k: int = 5, include_shadow: bool = False) -> List[Dict[str, Any]]:
    """
    Return the `top_k` memories most semantically similar to `query`.
    If include_shadow is False (default), shadow memories are hard-gated out of the response.
    """
    memoire = get_memoire(ctx)
    db_path = get_db_path(ctx)
    try:
        results = memoire.recall(query, top_k=top_k)
        final_results = []
        
        # Connect to SQLite for conflict detection
        conn = sqlite3.connect(db_path, timeout=10.0)
        c = conn.cursor()
        
        for r in results:
            if not include_shadow and r.state == "shadow":
                continue
                
            status_msg = "Active and Trusted."
            if r.state == "shadow":
                status_msg = "WARNING: This is a shadow memory with low trust. Use with caution."
                
            # Conflict Detection
            c.execute("SELECT contradiction_group FROM memories WHERE id = ?", (r.id,))
            row = c.fetchone()
            if row and row[0]:
                cg = row[0]
                # Check if there are other conflicting memories in this group
                c.execute(
                    "SELECT id, trust_ema FROM memories WHERE contradiction_group = ? AND archived = 0 AND id != ?", 
                    (cg, r.id)
                )
                conflicts = c.fetchall()
                if conflicts:
                    status_msg = f"CRITICAL: Conflict Detected! This memory belongs to a contradiction group with {len(conflicts)} other active memories. MUST call memoire_resolve_conflicts({r.id}) to settle this!"
            
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
            
        conn.close()
        return final_results
    except Exception as e:
        raise RuntimeError(f"Failed to recall: {str(e)}")

@mcp.tool()
def memoire_reinforce(memory_id: int, agent_output: str, task_succeeded: bool, ctx: Context) -> str:
    """
    Conditionally reinforce a memory based on actual use.
    """
    memoire = get_memoire(ctx)
    try:
        reinforced = memoire.reinforce_if_used(memory_id, agent_output, task_succeeded)
        if reinforced:
            return f"Memory {memory_id} was successfully reinforced."
        else:
            return f"Memory {memory_id} was NOT reinforced (conditions not met)."
    except Exception as e:
        return f"Failed to reinforce: {str(e)}"

@mcp.tool()
def memoire_penalize(memory_ids: List[int], failure_severity: float, ctx: Context) -> Any:
    """
    Penalize memories that contributed to a failed task outcome.
    `failure_severity` is a scale factor [0.0, 1.0].
    """
    memoire = get_memoire(ctx)
    try:
        results = memoire.penalize_if_used(memory_ids, failure_severity)
        return results
    except Exception as e:
        return f"Failed to penalize: {str(e)}"

@mcp.tool()
def memoire_batch_feedback(memory_ids: List[int], task_succeeded: bool, task_output: str, ctx: Context) -> str:
    """
    Batch process feedback for a set of memories used in a task.
    If task_succeeded is True, reinforces the memories. If False, penalizes them.
    Explicitly define success. Do NOT guess from task_output.
    """
    memoire = get_memoire(ctx)
    try:
        if not task_succeeded:
            results = memoire.penalize_if_used(memory_ids, failure_severity=1.0)
            return f"Task explicitly marked as FAILED. Penalized {len(results)} memories."
        else:
            reinforced_count = 0
            for mid in memory_ids:
                if memoire.reinforce_if_used(mid, task_output, task_succeeded=True):
                    reinforced_count += 1
            return f"Task explicitly marked as SUCCESSFUL. Reinforced {reinforced_count} memories out of {len(memory_ids)}."
    except Exception as e:
        return f"Failed to run batch feedback: {str(e)}"

@mcp.tool()
def memoire_status(ctx: Context) -> Dict[str, Any]:
    """
    Get the overall health, stats, and top trusted lessons from the database.
    """
    db_path = get_db_path(ctx)
    try:
        # Check if DB exists
        if not os.path.exists(db_path):
            return {"status": "Database not initialized yet. Call memoire_remember to start."}
            
        conn = sqlite3.connect(db_path, timeout=10.0)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM memories")
        total_memories = c.fetchone()[0]
        
        c.execute("SELECT store_state, COUNT(*) FROM memories GROUP BY store_state")
        state_counts = dict(c.fetchall())
        
        c.execute("SELECT id, content, trust_ema FROM memories WHERE archived=0 ORDER BY trust_ema DESC LIMIT 5")
        top_lessons = [
            {"id": row[0], "content": row[1], "trust_ema": row[2]} 
            for row in c.fetchall()
        ]
        
        conn.close()
        
        return {
            "total_memories": total_memories,
            "states": state_counts,
            "top_5_trusted_lessons": top_lessons
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get status: {str(e)}")

@mcp.tool()
def memoire_resolve_conflicts(memory_id: int, ctx: Context) -> str:
    """
    Explicitly trigger the contradiction resolution logic to 'settle' the database for a given memory.
    """
    memoire = get_memoire(ctx)
    try:
        memoire.resolve_contradictions(memory_id)
        return f"Successfully ran contradiction resolution for memory {memory_id}."
    except Exception as e:
        return f"Failed to resolve conflicts: {str(e)}"

@mcp.tool()
def memoire_forget(memory_id: int, ctx: Context) -> str:
    """
    Delete a memory by its id.
    """
    memoire = get_memoire(ctx)
    try:
        success = memoire.forget(memory_id)
        if success:
            return f"Successfully forgot memory {memory_id}."
        else:
            return f"Memory {memory_id} not found."
    except Exception as e:
        return f"Failed to forget: {str(e)}"

@mcp.tool()
def memoire_count(ctx: Context) -> str:
    """
    Total number of stored memory chunks.
    """
    memoire = get_memoire(ctx)
    try:
        count = memoire.count()
        return f"Total memories stored: {count}"
    except Exception as e:
        return f"Failed to count: {str(e)}"

@mcp.tool()
def memoire_clear(ctx: Context) -> str:
    """
    Erase ALL memories. This cannot be undone.
    """
    memoire = get_memoire(ctx)
    try:
        memoire.clear()
        return "All memories have been cleared successfully."
    except Exception as e:
        return f"Failed to clear: {str(e)}"

def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
