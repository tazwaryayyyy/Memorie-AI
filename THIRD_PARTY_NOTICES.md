# Third-party notices

Memoire itself is MIT-licensed (see `LICENSE`). Binary distributions of Memoire (the
`memoire-engine` Python wheels, the `memoire-engine` crate's compiled output, the `memoire-cli`
and `memoire-server` binaries, and the `libmemoire` shared library) **statically link** the third-party
components below. This file is bundled with those distributions.

> **Status: draft. This is a summary of the major components, not the complete inventory.**
> It does not yet reproduce every dependency's license text or the notices required by every
> transitive crate. It must be completed from a generated inventory of the exact dependency graph
> in `Cargo.lock` (for example with `cargo-about`) before a release is published. The authoritative
> license terms are the ones shipped by each upstream project.

## Major components

| Component | Role | Declared license | Notes |
|---|---|---|---|
| ONNX Runtime (Microsoft) | Runs the embedding model; **statically linked** from pyke's prebuilt binaries by `ort-sys` (`ort` 2.0.0-rc.11) | MIT | ONNX Runtime embeds further third-party code. Its own notices file is published with the matching release at <https://github.com/microsoft/onnxruntime> (`ThirdPartyNotices.txt`) and must be included in full. |
| `ort` / `ort-sys` (pykeio) | Rust bindings to ONNX Runtime | MIT OR Apache-2.0 | |
| `fastembed` (`=5.13.2`) | Embedding pipeline and model download | Apache-2.0 | |
| `tokenizers` (Hugging Face) | Tokenization | Apache-2.0 | Uses Oniguruma through the `onig` crate (`onig` MIT; Oniguruma BSD-2-Clause). |
| SQLite via `rusqlite` (`bundled`) | Storage | SQLite: public domain; `rusqlite`: MIT | |
| `tree-sitter` and the Python / Rust / JavaScript / TypeScript grammars | Code-aware chunking | MIT | |
| `blake3` | Content fingerprints | CC0-1.0 OR Apache-2.0 | |
| `tokio`, `axum` | Async runtime, local HTTP server | MIT | |
| `pyo3` | Python extension glue | Apache-2.0 OR MIT | |
| `instant-distance` | HNSW index | MIT OR Apache-2.0 | |
| `serde`, `serde_json`, `anyhow`, `thiserror`, `log`, `r2d2`, `r2d2_sqlite`, `env_logger` | Utilities | MIT OR Apache-2.0 (as declared by each crate) | |

SPDX identifiers above are the licenses the upstream projects declare; verify them against the
resolved crate versions.

## Linux wheels

A `manylinux` wheel may contain shared libraries copied in by `auditwheel` (they appear under a
`memoire_engine.libs/` directory in the wheel). Any such library is third-party code and its license
must be listed here. `scripts/inspect_wheel.py` prints a warning when a wheel contains one.

## Model weights (not bundled)

Memoire does **not** redistribute model weights. On first use the default embedder downloads
`sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, and the optional reranker downloads
`BAAI/bge-reranker-base`. Those weights are governed by the licenses on their own model cards;
check them before redistributing a populated model cache.
