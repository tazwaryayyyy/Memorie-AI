# Security Policy

Memoire is a pre-release (`0.1.0`, alpha), local-first library and tool set. This
document states what its security boundaries are, and just as importantly what they are
not. It was written from reading the code at commit `fe10f11` on `fix/p0-stabilization`.
The project has **not** had an external security audit or penetration test, and
`cargo audit` / dependency review has not been part of this document's verification.

## Supported Versions

No version has been released or published to any registry. Only the current tip of the
development branch (`main`) receives fixes.

## Reporting a Vulnerability

Please do **not** open a public issue with exploit details.

1. Preferred: use GitHub's private reporting on this repository
   (**Security** tab -> **Report a vulnerability**), if it is shown.
2. If that option is not available, open a public issue titled "Security contact
   request" containing **no technical details**, and a private channel will be arranged
   there.

Include: affected component and commit, a minimal reproduction, and the impact you
believe it has. This is a one-maintainer student project: reports are handled on a
best-effort basis, with no response-time guarantee and no bug bounty. Credit is given
in the changelog unless you ask otherwise.

## Threat Model

**In scope.** Memoire assumes the operating-system user running it is trusted. It aims
to resist:

- other websites reaching a local HTTP server through the browser (DNS rebinding and
  cross-origin requests),
- path traversal / unintended file access through database paths given to the HTTP
  server, the dashboard, or the MCP server's workspace marker,
- hostile export/import snapshots inflating trust,
- a cloned repository silently redirecting an agent's memory database elsewhere
  (`.memoire_path`).

**Out of scope / not provided.**

- **No encryption at rest.** Memories, embeddings and metadata are plaintext in a SQLite
  file. Export files are plaintext JSON.
- **No protection against other local processes or users with access to the files**, and,
  when `MEMOIRE_API_TOKEN` is unset, against any local process that can reach the
  loopback port.
- **Namespaces are logical partitions, not security boundaries.** They separate data in
  query results; anyone who can open the SQLite file can read every namespace.
- **Not safe to expose to a network.** Nothing here is designed for remote or
  multi-tenant use. There is no TLS, no rate limiting, and no per-user authorization.
- **Memory content is untrusted text for the consuming agent.** A stored memory is
  injected into an agent's context. A malicious or poisoned memory can carry prompt
  injection. Trust values and `MemoryPolicy` thresholds are heuristics for ranking and
  gating usefulness; they are **not** a defence against adversarial content.

## `memoire-server` (experimental, local-only)

`src/bin/memoire_server.rs` binds `127.0.0.1` only, on port `6779` unless
`MEMOIRE_SERVER_PORT` is set (an invalid value is a startup error). There is no option to
bind another address. Verified behaviour in code:

| Control | Behaviour |
|---|---|
| `MEMOIRE_API_TOKEN` **unset** | No authentication. Any local process able to connect can use the server (a warning is printed at startup). |
| `MEMOIRE_API_TOKEN` **set but blank** | The server refuses to start and exits with code `2` (fail closed). |
| `MEMOIRE_API_TOKEN` **set** | Every route except `GET /health` requires `Authorization: Bearer <token>`. The server compares BLAKE3 hashes of the presented and configured token (token is trimmed); BLAKE3 hash equality is constant-time. Failure returns `401`. |
| `Host` header | Must be `localhost` or a loopback IP literal (with optional port) on **every** request, including `/health`; otherwise `403`. Blocks DNS rebinding. |
| `Origin` header | If present, must be loopback; otherwise `403`. A request with no `Origin` is accepted (non-browser clients). |
| `MEMOIRE_ALLOWED_PATHS` | Comma-separated directories the request's `db` must resolve into. Default: the server's working directory. Paths are made absolute, parent directories and existing leaves (including symlinks) are canonicalized, and containment is checked component-wise, so a symlink pointing outside is rejected and `SQLite` never sees a `file:` URI. |
| `MEMOIRE_ENABLE_ADMIN_TOOLS` | Values `1`, `true`, `yes`, `on` enable `POST /clear` and `preserve_trust` on import. Otherwise those return `403`. |
| Limits | Request body 2 MiB; imports capped at 1000 memories; namespace names capped at 128 bytes; at most 16 open (database, namespace) instances. |

Residual risk: with the token unset, "loopback only" does not distinguish between local
processes or users on a shared machine. Set a token on any machine you do not fully
control. The token is read from an environment variable and therefore is visible to
anything that can read the process environment.

## Dashboard (experimental, local-only)

`dashboard/` is a Next.js development UI (`npm run dev` runs `next dev -H 127.0.0.1`).
Its API routes apply `dashboard/src/lib/localGuard.ts` (Host, Origin, `Sec-Fetch-Site`
and JSON content type checks), forward `MEMOIRE_API_TOKEN` to `memoire-server` when set,
and only touch directories listed in `MEMOIRE_ALLOWED_PATHS`; with that variable empty
no directory is allowed. The destructive `clear` action was removed from the UI. A
production deployment (`next start`, a reverse proxy, a non-loopback host) has **not**
been evaluated and is unsupported.

## MCP Server (`memoire-mcp`)

- **Database path precedence**: `MEMOIRE_DB_PATH`, then `MEMOIRE_DB`, then
  `<workspace>/.memoire_path`, then `<workspace>/memoire.db`. The two environment
  variables are trusted (they come from the client's own configuration). The
  workspace-controlled `.memoire_path` file is not: it is symlink-resolved and must name a
  file **inside the workspace root**, and is size-limited, so a cloned repository cannot
  point the agent's memory at an arbitrary path.
- **Admin tools**: `memoire_clear` and `memoire_import` are only exposed when
  `MEMOIRE_ENABLE_ADMIN_TOOLS` is truthy. `memoire_import` always resets trust
  (`preserve_trust=False`) and accepts at most 1000 rows.
- **Shadow memories** are hidden from `memoire_recall` / `memoire_recall_explained`
  unless `include_shadow=true`.
- **Logs**: structured JSON lines are appended to `~/.memoire/logs/mcp-server.jsonl`,
  created with default umask permissions and not rotated. Events log database paths and
  namespaces; failures log tracebacks via `logger.exception`. Whether any code path logs
  memory content has **not** been audited; treat the log file as sensitive.

## Import Trust

`Memoire::import_namespace` defaults to `ImportTrust::Reset`: only `content` is read,
it goes through the normal `remember()` pipeline and is re-embedded, and snapshot
metadata (`trust_ema`, `reinforcement_count`, `importance_base`, `confidence`,
`created_at`) is ignored. Restoring metadata requires an explicit opt-in
(`ImportTrust::PreserveBounded`, Python `preserve_trust=True`, CLI `--preserve-trust`,
or the server's admin-gated `preserve_trust`). The restored values are clamped to
bounds, but "bounded" is not "verified": a hostile snapshot can still claim maximum
in-range trust. Use opt-in only for snapshots you exported yourself.

## Data on Disk, Deletion and Retention

- **File mode.** On Unix, when the store is opened, the main database file is chmod'd to
  `0600` if it is not already (`src/store.rs`, `Store::new`). This is done for the
  main file only, and not on Windows. Permissions of the `-wal` / `-shm` sidecar files are
  not set by Memoire and were not verified here.
- **WAL mode.** File-backed databases run SQLite with `journal_mode=WAL`, so recent writes
  also live in `<db>-wal` until checkpointed.
- **Deletion is not secure erasure.** `forget`, `clear` and maintenance pruning issue
  `DELETE FROM memories ...`. Memoire does not set `PRAGMA secure_delete`, and no code in
  `src/` calls `VACUUM` or `wal_checkpoint`. Deleted content, and embedding blobs, can
  therefore remain in free pages of the database file and in the WAL until overwritten.
  Contradiction resolution is not an immediate delete: a loser judged by the lexical
  heuristic is **quarantined** and kept; a loser judged by an ONNX NLI model is **archived**,
  and maintenance pruning later deletes archived rows (after seven days).
  If you must remove data from disk, stop all Memoire processes and then, for example,
  run `PRAGMA wal_checkpoint(TRUNCATE); VACUUM;` with the `sqlite3` CLI, or export the
  wanted namespace and import it into a fresh file. Backups, filesystem snapshots and SSD
  wear-levelling are outside Memoire's control. These commands are a pointer, not a
  guarantee, and were not run as part of writing this document.
- **Exports** (`export`, `memoire_export`, `/export`) write full memory content in plain
  JSON to wherever you direct them.

## Supply Chain

- The default embedding model (`all-MiniLM-L6-v2`) and the optional reranker are fetched
  from Hugging Face at runtime by `fastembed` (`=5.13.2`). No checksum or revision pin was
  found in `src/embedder.rs`; integrity relies on the Hub and the `fastembed` / `hf-hub`
  download path. The model cache directory is `$HF_HOME`, else `$FASTEMBED_CACHE_DIR`,
  else `./.fastembed_cache` in the current working directory. Anything that can write
  that directory can substitute model files.
- Builds download a prebuilt ONNX Runtime (`ort` with `download-binaries`). Third-party
  license and notice obligations for redistributed binaries have not yet been reviewed.
- Python dependencies for the MCP server are pinned in `mcp-server/uv.lock`
  (`uv sync --locked`). Nothing is published to PyPI or crates.io yet.

## Hardening Checklist for Users

1. Keep `memoire-server` and the dashboard on loopback; set `MEMOIRE_API_TOKEN`.
2. Keep `MEMOIRE_ALLOWED_PATHS` as narrow as possible; leave
   `MEMOIRE_ENABLE_ADMIN_TOOLS` unset unless you need `/clear`, `memoire_clear` or
   `memoire_import`.
3. Treat database files, export JSON, and `~/.memoire/logs/` as sensitive.
4. Do not import snapshots from untrusted sources with trust preservation.
5. Review recalled memories before letting an agent act on them, especially in
   repositories or databases you did not create.
