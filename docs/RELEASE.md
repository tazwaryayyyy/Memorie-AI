# Release Guide

This guide is for technical GitHub releases. It does not assume PyPI or crates.io publishing.

## Release Contract

Memoire ships as two layers:

- Rust native library and CLI from `cargo build --release`
- Python packages from `bindings/python` and `mcp-server`

The Python packages do not bundle platform-specific native libraries yet. Users must either:

- build the Rust library locally, or
- download the matching native-library artifact from a GitHub Release.

Set `MEMOIRE_LIB` to the native library path when running from installed Python packages:

| Platform | Library |
|---|---|
| Linux | `libmemoire.so` |
| macOS | `libmemoire.dylib` |
| Windows | `memoire.dll` |

`MEMOIRE_HOME` may point to a directory containing the library, a `lib/` directory containing it, or a source checkout with `target/release/`.

## Tag Release

1. Update versions in:
   - `Cargo.toml`
   - `bindings/python/pyproject.toml`
   - `mcp-server/pyproject.toml`
   - `bindings/python/memoire/__init__.py`
2. Update `CHANGELOG.md`.
3. Run:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib
cargo test --test integration_test

cd mcp-server
uv sync --locked --extra dev
uv run pytest
```

4. Create and push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow builds Linux, macOS, and Windows native-library zip artifacts and attaches them to the GitHub Release.

## Technical User Install

```bash
git clone https://github.com/tazwaryayyyy/Memorie-AI
cd Memorie-AI
cargo build --release

python -m venv .venv
. .venv/bin/activate
python -m pip install ./bindings/python
python -m pip install "mcp[cli]==1.27.1"
python -m pip install --no-deps ./mcp-server

export MEMOIRE_LIB="$PWD/target/release/libmemoire.so"
python -c "from memoire.client import _get_lib; print(_get_lib())"
python -c "import server; print(server.startup_health_check())"
```

On Windows, set `MEMOIRE_LIB` to `target\release\memoire.dll`.

## Model Download And Offline Use

The first real `remember()` or `recall()` call initializes FastEmbed and may download `all-MiniLM-L6-v2` into the Hugging Face cache. `memoire_health` only checks the native library; it does not download the model.

For offline machines:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"
```

Copy the Hugging Face cache to the target machine and set:

```bash
export HF_HOME=/path/to/huggingface/cache
```

## Current Non-Goals

- No PyPI wheels with bundled native libraries yet.
- No crates.io release until the repository metadata and artifact policy are stable.
- No automatic model vendoring inside GitHub release artifacts.
