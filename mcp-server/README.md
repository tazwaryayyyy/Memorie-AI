# memoire-mcp

MCP stdio server for Memoire.

Install from a source checkout:

```bash
cd mcp-server
uv sync --locked
uv run memoire-mcp
```

For package smoke tests or manual virtualenv installs:

```bash
python -m pip install ./bindings/python
python -m pip install "mcp[cli]==1.27.1"
python -m pip install --no-deps ./mcp-server
```

Set `MEMOIRE_LIB` to the native Memoire shared library before starting the server from an installed environment.

See `docs/RELEASE.md` in the repository root for the full release workflow.
