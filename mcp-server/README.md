# memoire-mcp

MCP stdio server for Memoire.

## Run from a source checkout

Requires a Rust toolchain: the Python SDK (`bindings/python`) is a PyO3 extension
built with maturin.

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
```

Use `--no-deps` for the last step until the SDK's PyPI name is final: the
provisional name `memoire-engine` is not registered, and the unrelated PyPI
project `memoire` must never be installed in its place.

`MEMOIRE_LIB` and `cargo build --release` are **not** used by the Python package.
If the server reports `native_library_unavailable`, run `maturin develop --release`
inside `bindings/python` in the same virtualenv that runs the server.

First use downloads the embedding model (about 23 MB, from Hugging Face) unless it
is already cached.

## Database location

Highest precedence first: `<workspace>/.memoire_path`, `MEMOIRE_DB_PATH`,
`MEMOIRE_DB`, `<workspace>/memoire.db`. MCP's initialize handshake carries no
workspace folder, so `<workspace>` is the server process's working directory
unless the client sends non-standard `workspaceFolders`/`rootUri` fields. Set
`MEMOIRE_DB_PATH` explicitly if your client does not start the server in the
project directory.

## Tool notes

- `memoire_recall` and `memoire_recall_explained` exclude `shadow` memories unless
  `include_shadow=true`.
- `memoire_export` returns `{"ok": true, "error": null, "snapshot": {...}}`; pass the
  `snapshot` value to `memoire_import`.

See `docs/RELEASE.md` in the repository root for the release workflow.
