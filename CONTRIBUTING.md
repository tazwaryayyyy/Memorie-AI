# Contributing to Memoire

Thanks for your interest! Memoire is an early-stage project and contributions are very welcome.

## Development Setup

```bash
# Clone
git clone https://github.com/yourusername/memoire
cd memoire

# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# Install dev dependencies
make install-dev

# Build everything
make release

# Verify it works
make test-unit          # fast (no model download)
make demo-python        # full demo (downloads model on first run)
```

## Project Structure

```
src/
  lib.rs          Public Rust API — struct Memoire
  chunker.rs      Sliding-window text chunker
  embedder.rs     fastembed/ONNX wrapper
  store.rs        SQLite persistence + cosine search
  ffi.rs          C-compatible extern "C" layer
  error.rs        Unified error type
  bin/cli.rs      memoire CLI binary
tests/
  integration_test.rs   Full integration tests
benches/
  core_bench.rs   Criterion performance benchmarks
bindings/
  python/         Python ctypes wrapper + pytest suite
  node/           Node.js ffi-napi wrapper
  go/             Go cgo binding
examples/
  basic_usage.rs  Rust quick-start
  agent_demo.py   Python multi-session demo
  aider_plugin.py Aider integration
  mcp_server.py   MCP tool server
  openai_agent.py OpenAI function-calling demo
docs/
  ARCHITECTURE.md Internal design docs
  FFI_GUIDE.md    Language-by-language FFI examples
```

## Workflow

1. **Open an issue** before starting large changes — discuss the approach first.
2. Fork and create a feature branch: `git checkout -b feat/my-feature`
3. Write code + tests.
4. Run the full check suite before opening a PR:
   ```bash
   make fmt lint test
   ```
5. Open a PR against `main`. Link the relevant issue.

## Code Standards

- **Rust:** follow `rustfmt` defaults. `cargo clippy -- -D warnings` must pass clean.
- **Python:** follow PEP 8. No type errors in public interfaces.
- **Tests:** every new public function needs at least one unit test and one integration test.
- **No unsafe without justification:** the FFI layer is the only place unsafe is expected. Add a `// SAFETY:` comment explaining the invariant.
- **No new external services:** Memoire must remain fully local. PRs that add cloud dependencies will be declined.

## Adding a New Language Binding

1. Create `bindings/<lang>/` directory.
2. Implement a wrapper that covers: `new`, `remember`, `recall`, `forget`, `count`, `close`.
3. Add tests — ideally a test per API method.
4. Add a demo script.
5. Document it in `docs/FFI_GUIDE.md`.

## Reporting Bugs

Please include:
- OS and architecture (`uname -a` / `rustc --version`)
- Memoire version (`cargo pkgid`)
- Minimal reproduction case
- Full error output with `RUST_LOG=debug`

## Roadmap Items Open for Contribution

| Item | Complexity | Notes |
|---|---|---|
| `usearch` HNSW index for 100k+ memories | Medium | see `src/store.rs` |
| Metadata tagging (`project`, `session_id`) | Small | schema + filter API |
| Node.js native addon (napi-rs) | Medium | faster than ffi-napi |
| Ruby gem with native ext | Medium | C ext wrapping the .h |
| Streaming recall (async iterator) | Medium | requires async store |
| `wasm32-wasi` target | Large | needs in-process ONNX |

## License

By contributing you agree that your contributions are licensed under MIT.
