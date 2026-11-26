# Release Guide

This guide is for technical GitHub releases. It does not assume PyPI or crates.io publishing.

## Release Contract

Memoire ships as two layers:

- Rust native library and CLI from `cargo build --release`
- Python packages from `bindings/python` and `mcp-server`

The Python SDK (`bindings/python`, import name `memoire_engine`) is a PyO3 extension
built with maturin, so it is compiled per platform; there is no separate native
library to locate and `MEMOIRE_LIB` / `MEMOIRE_HOME` are not used by it. Until wheels
are published, users build it from a source checkout with a Rust toolchain.

The C library (`libmemoire.so` / `libmemoire.dylib` / `memoire.dll`) is still built
for the C FFI and attached to GitHub Releases separately.

## Tag Release

1. Update versions in:
   - `Cargo.toml`
   - `bindings/python/pyproject.toml`
   - `mcp-server/pyproject.toml`
   - `bindings/python/memoire_engine/__init__.py`
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

Before tagging, exercise the whole pipeline without a tag: GitHub -> Actions -> Release -> "Run
workflow" on the release branch. It builds and smoke-tests everything and creates no release.

On a `v*` tag, `.github/workflows/release.yml`:

1. fails unless the tag equals `v` + the version in `Cargo.toml` and both `pyproject.toml` files
   (`scripts/check_release_metadata.py`, which also checks `mcp-server/server.json` and that the
   `THIRD_PARTY_NOTICES.md` inventory matches `Cargo.lock`);
2. builds native-library zips (Linux, macOS, Windows), `memoire-engine` wheels (Linux x86_64
   `manylinux_2_28`, macOS arm64, Windows x86_64; abi3, so one wheel per platform covers Python
   >= 3.10) and the `memoire-mcp` sdist and wheel;
3. inspects the wheels (`scripts/inspect_wheel.py`: size under 90 MB, `LICENSE` and
   `THIRD_PARTY_NOTICES.md` present, no generic top-level modules, every library bundled by
   `auditwheel` named in the wheel's own notices, registry marker in the `memoire-mcp`
   metadata);
4. installs the built wheels into fresh virtualenvs on all three platforms, with only
   `scripts/` checked out, and runs `scripts/smoke_installed.py` (real embedding, recall,
   MCP health);
5. creates a **draft** GitHub Release with the artifacts and `SHA256SUMS.txt`.

It publishes nothing to PyPI, crates.io or the MCP Registry.

After any `Cargo.lock` change, regenerate the third-party inventory (needs `cargo`; it uses
`--all-features` so the optional `pyo3` dependencies are included) and commit both copies:

```bash
python scripts/generate_third_party_notices.py          # rewrite
python scripts/generate_third_party_notices.py --check  # exit 1 if stale (also verifies licenses)
```

## Publishing (manual, only with the owner's explicit go-ahead)

Order matters because each later step checks the earlier one:

1. `memoire-engine` to PyPI (the `memoire-mcp` wheel depends on `memoire-engine>=0.1.0,<0.2.0`).
2. `memoire-mcp` to PyPI. Its README must contain `mcp-name: io.github.tazwaryayyyy/memoire-mcp`.
3. MCP Registry: from `mcp-server/`, `mcp-publisher validate`, then `mcp-publisher login github`
   and `mcp-publisher publish`. The registry only accepts a `server.json` whose package version is
   already on PyPI.
4. `memoire-engine` to crates.io (`cargo publish --dry-run --locked` first).

Re-check on PyPI and crates.io that the names are still unregistered immediately before step 1.
Wheel platforms: Linux x86_64, macOS arm64, Windows x86_64 only. There is no sdist for
`memoire-engine`, so other platforms cannot install it from PyPI; the `memoire-mcp` install fails
there too.

## Technical User Install

```bash
git clone https://github.com/tazwaryayyyy/Memorie-AI
cd Memorie-AI
python -m venv .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m pip install maturin
python -m pip install ./bindings/python        # builds the PyO3 extension
python -m pip install "mcp[cli]==1.27.1"
python -m pip install --no-deps ./mcp-server

python -c "from memoire_engine.client import _get_lib; print(_get_lib())"
python -c "import memoire_mcp.server as s; print(s.startup_health_check())"
python -c "import importlib.util as u; assert u.find_spec('server') is None and u.find_spec('memoire') is None"
```

## Model Download And Offline Use

The first real `remember()` or `recall()` call initializes FastEmbed and may download `all-MiniLM-L6-v2` (about 87 MiB: a 86.2 MiB `model.onnx` plus tokenizer and config files) into the model cache: `$HF_HOME` if set, else `$FASTEMBED_CACHE_DIR`, else `./.fastembed_cache` relative to the working directory. Set `FASTEMBED_CACHE_DIR` to share one cache. `memoire_health` only checks that the native extension imports; it does not download the model.

For offline machines:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"
```

Copy the Hugging Face cache to the target machine and set:

```bash
export HF_HOME=/path/to/huggingface/cache
```

## ONNX Runtime Linking

ONNX Runtime is **linked into** the Rust library, the CLI/server and the Python extension; it is not
located at runtime. `ort/load-dynamic` must stay disabled: it makes `ort-sys` skip linking and
`dlopen`/`LoadLibrary` `libonnxruntime`/`onnxruntime.dll` by bare name, which fails wherever no
runtime is on the loader path and, on Windows, can load an unrelated `System32\onnxruntime.dll`.
`tests/ort_smoke.rs` guards this on every CI OS.

- **Build needs network** the first time: `ort-sys` downloads pyke's prebuilt static ONNX Runtime
  (1.23.2) into its cache (`~/.cache/ort.pyke.io`, `~/Library/Caches/ort.pyke.io`, `%LOCALAPPDATA%\ort.pyke.io`). Offline or
  unsupported targets can set `ORT_LIB_LOCATION` (and `ORT_PREFER_DYNAMIC_LINK=1` for a shared
  library, which then becomes a runtime dependency).
- **Prebuilt targets** (ort 2.0.0-rc.11): Windows MSVC x86_64/aarch64, Linux **gnu** x86_64/aarch64,
  macOS aarch64. No `x86_64-apple-darwin` and no musl: build those with `ORT_LIB_LOCATION`.
- **Windows:** do not add `-C target-feature=+crt-static`; the prebuilt libraries are expected to use
  the dynamic CRT (verify before changing).
- **Size:** artifacts and every test binary now contain ONNX Runtime. Measure the release
  `memoire_engine` wheel/extension before publishing (PyPI's default per-file limit is 100 MB).

## Current Non-Goals

- No PyPI sdist or prebuilt wheels beyond Linux `x86_64` (`manylinux_2_28`), macOS `arm64`, and Windows `x86_64`.
- No automatic model vendoring inside GitHub release artifacts or PyPI wheels.
