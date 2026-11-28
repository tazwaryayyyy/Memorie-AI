# memoire-mcp

<!-- mcp-name: io.github.tazwaryayyyy/memoire-mcp -->

MCP stdio server for Memoire. Python package: `memoire_mcp`. Console script:
`memoire-mcp` (also `python -m memoire_mcp`).

## Install from PyPI

```bash
uvx memoire-mcp          # or: pip install memoire-mcp
```

`memoire-mcp` `0.1.0` is on PyPI. It depends on `memoire-engine` (import name `memoire_engine`),
which has prebuilt wheels only for Linux `x86_64` (`manylinux_2_28`), macOS `arm64` and Windows
`x86_64` and no sdist, so installing `memoire-mcp` fails on other platforms unless you build from
source (below). The unrelated PyPI project `memoire` must never be installed in its place.

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
python -m pip install "mcp[cli]>=1.30.0,<2"
python -m pip install --no-deps ./mcp-server
memoire-mcp
```

`--no-deps` keeps `pip` from fetching `memoire-engine` from PyPI, so the local build from the
first step is the one that gets used. The unrelated PyPI project `memoire` must never be
installed in its place.

`MEMOIRE_LIB` and `cargo build --release` are **not** used by the Python package.
If the server reports `native_library_unavailable`, run `maturin develop --release`
inside `bindings/python` in the same virtualenv that runs the server.

First use downloads the embedding model (about 87 MiB, from Hugging Face) unless it
is already cached.

## MCP client configuration

Every command below points at the console script of a virtualenv in which `memoire-mcp` is
installed (from PyPI or from a source checkout, see above). Replace the `/ABSOLUTE/PATH/...`
placeholders; on Windows use `...\.venv\Scripts\memoire-mcp.exe` and a path such as
`C:\Users\you\.memoire\memoire.db`.

Set `FASTEMBED_CACHE_DIR` to **one shared directory** in every client. Without it (and without
`HF_HOME`), the embedding model (about 87 MiB) is downloaded into `./.fastembed_cache` relative to
the server's working directory, i.e. once per project. The first tool call that needs the
embedder downloads the model and can exceed a client's tool timeout; make one throwaway tool
call right after setup to warm the cache before relying on it.

### Claude Code

Options must come before the server name, and `--` separates them from the command. Keep
`--transport stdio` between `--env` and the name: if the name directly follows `--env`, the CLI
reads it as another `KEY=VALUE` pair.

```bash
claude mcp add --scope user \
  --env MEMOIRE_DB_PATH=/ABSOLUTE/PATH/TO/memoire.db \
  --env FASTEMBED_CACHE_DIR=/ABSOLUTE/PATH/TO/.fastembed_cache \
  --transport stdio memoire \
  -- /ABSOLUTE/PATH/TO/.venv/bin/memoire-mcp
```

`--scope user` makes the server available in every project and keeps its configuration out of
the repository (`--scope project` would write `.mcp.json` into it). Check with
`claude mcp list`.

### Cursor

Put this in `~/.cursor/mcp.json` (all projects) or `<project>/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "memoire": {
      "command": "/ABSOLUTE/PATH/TO/.venv/bin/memoire-mcp",
      "env": {
        "MEMOIRE_DB_PATH": "/ABSOLUTE/PATH/TO/memoire.db",
        "FASTEMBED_CACHE_DIR": "/ABSOLUTE/PATH/TO/.fastembed_cache"
      }
    }
  }
}
```

### Any other stdio client

```json
{
  "mcpServers": {
    "memoire": {
      "command": "/ABSOLUTE/PATH/TO/.venv/bin/memoire-mcp",
      "env": {
        "MEMOIRE_DB_PATH": "/ABSOLUTE/PATH/TO/memoire.db",
        "FASTEMBED_CACHE_DIR": "/ABSOLUTE/PATH/TO/.fastembed_cache"
      }
    }
  }
}
```

The old `python /path/to/mcp-server/server.py` invocation no longer exists. `uvx memoire-mcp`
also works: the maintainer ran `uvx memoire-mcp==0.1.0` end to end by hand (remember, recall and
feedback across sessions). That run is not part of CI, and the client configuration blocks above
have not been tested with `uvx`.

### `MEMOIRE_TRIAL_CAP`

Optional (non-negative integer; an invalid value stops the server with exit code `2`). A recall
returns at most this many *unverified* memories (ones with no recorded feedback); the default is
`1`. With a fresh database every memory is unverified, so by default a recall returns one
result. Raising it makes recalls return more results before any feedback exists. It changes how
many results are offered, **not** how good they are: the only pre-registered evaluation run so far
did not show an accuracy benefit for the default or for any other setting (the `trial_cap=3`
numbers in it are exploratory and not inferential; see "Evidence so far" in the repository
README). Choose a value for how many results you want to see, not for an expected accuracy gain.

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
- Reinforcement sends one session id per server process (`memoire_status` shows it), so a
  memory can be reinforced at most once per server process.
- Penalties are not tied to that session id. `memoire_penalize` and `memoire_batch_feedback`
  take an optional `event_id` (at most 256 characters): pass one id per failed outcome and a
  retry of that outcome is not counted twice. Without it every call counts, so a single
  session can still quarantine a memory. `failure_severity` is a gate: any value above 0 is
  one failure (fractional scaling is not supported in `0.1.x`). Repeated ids in one call
  count once.
- `memoire_identify_gaps` `min_trust` defaults to the engine HINT threshold (0.40).
- `memoire_export` returns `{"ok": true, "error": null, "snapshot": {...}}`; pass the
  `snapshot` value to `memoire_import`.

See `docs/RELEASE.md` in the repository root for the release workflow.
