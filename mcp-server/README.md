# memoire-mcp

MCP stdio server for Memoire. Python package: `memoire_mcp`. Console script:
`memoire-mcp` (also `python -m memoire_mcp`).

## Run from a source checkout

Requires a Rust toolchain: the Python SDK (`bindings/python`, import name
`memoire_engine`) is a PyO3 extension built with maturin.

```bash
cd mcp-server
uv sync --locked --extra dev   # builds the native extension via maturin
uv run memoire-mcp
```

Manual virtualenv install (package smoke tests):

```bash
python -m pip install ./bindings/python        # builds the PyO3 extension
python -m pip install "mcp[cli]==1.27.1"
python -m pip install --no-deps ./mcp-server
memoire-mcp
```

Use `--no-deps` for the last step until the SDK's PyPI name is final: the
provisional name `memoire-engine` is not registered, and the unrelated PyPI
project `memoire` must never be installed in its place.

`MEMOIRE_LIB` and `cargo build --release` are **not** used by the Python package.
If the server reports `native_library_unavailable`, run `maturin develop --release`
inside `bindings/python` in the same virtualenv that runs the server.

First use downloads the embedding model (about 23 MB, from Hugging Face) unless it
is already cached.

## MCP client configuration

Point the client at the virtualenv's console script (or `python -m memoire_mcp`):

```json
{
  "mcpServers": {
    "memoire": {
      "command": "/ABSOLUTE/PATH/TO/.venv/bin/memoire-mcp",
      "env": { "MEMOIRE_DB_PATH": "/ABSOLUTE/PATH/TO/memoire.db" }
    }
  }
}
```

(Windows: `...\.venv\Scripts\memoire-mcp.exe`.) The old
`python /path/to/mcp-server/server.py` invocation no longer exists.

## Database location

Highest precedence first:

1. `MEMOIRE_DB_PATH`, then `MEMOIRE_DB`: explicit operator configuration, used as given.
2. `<workspace>/.memoire_path`: **must resolve to a location inside the workspace**
   (symlinks are followed). A target outside it is rejected with an error, because the
   file lives in the repository, which may be untrusted. To use a database elsewhere,
   set `MEMOIRE_DB_PATH`.
3. `<workspace>/memoire.db`.

MCP's initialize handshake carries no workspace folder, so `<workspace>` is the
server process's working directory unless the client sends non-standard
`workspaceFolders`/`rootUri` fields. Set `MEMOIRE_DB_PATH` explicitly if your client
does not start the server in the project directory.

## Admin tools (off by default)

`memoire_clear` and `memoire_import` are **not listed** unless the server is started
with `MEMOIRE_ENABLE_ADMIN_TOOLS=1` (also `true`, `yes`, `on`). An agent, or text that
steers it, should not be able to wipe or bulk-load memory by default. The flag is read
at startup (restart to change it) and re-checked on every call.

`memoire_import` always treats the snapshot as untrusted: only each memory's `content`
is used, and trust metadata (`trust_ema`, `reinforcement_count`, `importance_base`,
`confidence`, `created_at`) is ignored, so an imported memory is no more trusted than a
freshly remembered one. At most 1000 memories per call. `memoire_forget` is not gated.

## Tool notes

- `memoire_recall` and `memoire_recall_explained` return live memories only unless
  `include_shadow=true`, which is passed to the engine as `include_hidden` (hidden rows
  included, trial cap skipped). Each memory has an `exposure` label.
- `MEMOIRE_TRIAL_CAP` overrides the engine's `trial_cap` (default `1`: with no feedback a
  recall returns at most one result). Non-integer or negative values stop the server.
- Feedback tools send one session id per server process (`memoire_status` shows it).
- `memoire_identify_gaps` `min_trust` defaults to the engine HINT threshold (0.40).
- `memoire_export` returns `{"ok": true, "error": null, "snapshot": {...}}`; pass the
  `snapshot` value to `memoire_import`.

See `docs/RELEASE.md` in the repository root for the release workflow.
