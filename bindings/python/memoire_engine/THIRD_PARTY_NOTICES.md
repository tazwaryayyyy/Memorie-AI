# Third-party notices

Memoire itself is MIT-licensed (see `LICENSE`). Binary distributions of Memoire (the
`memoire-engine` Python wheels, the `libmemoire` shared library, and the `memoire-cli` and
`memoire-server` binaries) contain code from the third-party components listed here. This file is
bundled with those distributions.

## Rust crates

The inventory at the end of this file lists **every registry package locked in this repository's
`Cargo.lock`**, resolved for all targets (Linux, macOS, Windows, wasm32), all feature combinations
(including `pyo3`, which the Python wheel is built with) and all dependency kinds. A given binary
contains a subset of them: Windows-only and wasm32-only crates are not in a Linux wheel, and
build-time and test-only crates are in no binary. This file does not say which is which.

License identifiers are the SPDX expressions declared in each crate's manifest. This file does not
reproduce license texts or copyright lines; each crate's published source
(`https://crates.io/crates/NAME/VERSION`) carries its own license files. Where a crate is offered
under alternatives (`OR`), the permissive alternatives are the ones relied on; the
LGPL-2.1-or-later option that `r-efi` also offers is not.

Notes on specific licenses:

- **MPL-2.0**: `option-ext` 0.2.0 only. It is used unmodified; its source is at
  <https://crates.io/crates/option-ext/0.2.0>.
- **CDLA-Permissive-2.0**: `webpki-roots` (0.26.11 and 1.0.6) and `webpki-root-certs` 1.0.6, which
  carry Mozilla's CA certificate data. Terms: <https://cdla.dev/permissive-2-0/>.
- **Unicode-3.0**: Unicode data tables. Terms: <https://www.unicode.org/license.txt>.
- **`AND` expressions** (for example `ring`: Apache-2.0 AND ISC; `matchit`: MIT AND BSD-3-Clause):
  every listed license applies to the respective part of the crate.

## Components that are not Rust crates

| Component | How it is included | License |
|---|---|---|
| ONNX Runtime 1.23.2 (Microsoft) | Statically linked from pyke's prebuilt binaries by `ort-sys` 2.0.0-rc.11 | MIT. It embeds further third-party code: see its `ThirdPartyNotices.txt` at <https://github.com/microsoft/onnxruntime/blob/v1.23.2/ThirdPartyNotices.txt> (license: `.../v1.23.2/LICENSE`). That file is large and is referenced here, not copied. |
| SQLite | Compiled in by `libsqlite3-sys` (`rusqlite` with `bundled`) | Public domain: <https://sqlite.org/copyright.html> |
| Oniguruma | Compiled in by `onig_sys` (`tokenizers` with `onig`) | BSD-2-Clause |
| tree-sitter runtime and grammars | Compiled in by the `tree-sitter*` crates | MIT |

## Linux wheels: shared libraries copied in by auditwheel

Linux wheels contain a `memoire_engine.libs/` directory. `auditwheel` copies into it the shared
libraries a manylinux wheel may not rely on the host for. The crate graph links OpenSSL dynamically
(`openssl` and `openssl-sys` are in the inventory; OpenSSL is not vendored), so these are
**`libssl` and `libcrypto` (OpenSSL)**, copied unmodified from the build image.

OpenSSL 1.1.x is licensed under the OpenSSL License and the original SSLeay License; OpenSSL 3.x
under Apache-2.0. The suffix of the file name in `memoire_engine.libs/` (`.so.1.1` or `.so.3`)
tells which applies. Full terms: <https://www.openssl.org/source/license.html>.

The release workflow fails if a wheel contains a bundled library whose name is not mentioned in
this file (`scripts/inspect_wheel.py`).

## Model weights (not bundled)

Memoire does **not** redistribute model weights. On first use the default embedder downloads
`sentence-transformers/all-MiniLM-L6-v2` (about 87 MiB including tokenizer and config files) from
Hugging Face, and the optional reranker downloads `BAAI/bge-reranker-base`. Those weights are
governed by the licenses on their own model cards; check them before redistributing a populated
model cache.

## Inventory

The block below is generated: run `python scripts/generate_third_party_notices.py` after any
`Cargo.lock` change (`--check` reports staleness). `scripts/check_release_metadata.py` fails when
it no longer matches `Cargo.lock`.

<!-- BEGIN GENERATED INVENTORY (python scripts/generate_third_party_notices.py) -->

381 packages in 25 license groups, from `cargo metadata --locked --all-features` (the resolved `Cargo.lock`). Grouped by license expression; equivalent spellings (`MIT/Apache-2.0`, `Apache-2.0 OR MIT`) share a group.

### Apache-2.0 OR MIT (240)

- `ahash 0.8.12`
- `allocator-api2 0.2.21`
- `anes 0.1.6`
- `anstream 1.0.0`
- `anstyle 1.0.14`
- `anstyle-parse 1.0.0`
- `anstyle-query 1.1.5`
- `anstyle-wincon 3.0.11`
- `anyhow 1.0.102`
- `arrayvec 0.7.6`
- `async-trait 0.1.89`
- `atomic-waker 1.1.2`
- `autocfg 1.5.0`
- `base64 0.13.1`
- `base64 0.22.1`
- `base64ct 1.8.3`
- `bitflags 2.11.0`
- `bumpalo 3.20.2`
- `cast 0.3.0`
- `cc 1.2.60`
- `cfg-if 1.0.4`
- `chacha20 0.10.0`
- `clap 4.6.0`
- `clap_builder 4.6.0`
- `clap_lex 1.1.0`
- `colorchoice 1.0.5`
- `core-foundation 0.10.1`
- `core-foundation 0.9.4`
- `core-foundation-sys 0.8.7`
- `cpufeatures 0.3.0`
- `crc32fast 1.5.0`
- `criterion 0.5.1`
- `criterion-plot 0.5.0`
- `crossbeam-deque 0.8.6`
- `crossbeam-epoch 0.9.18`
- `crossbeam-utils 0.8.21`
- `dary_heap 0.3.8`
- `der 0.8.0`
- `derive_builder 0.20.2`
- `derive_builder_core 0.20.2`
- `derive_builder_macro 0.20.2`
- `dirs 6.0.0`
- `dirs-sys 0.5.0`
- `displaydoc 0.2.5`
- `either 1.15.0`
- `encode_unicode 1.0.0`
- `env_filter 1.0.1`
- `env_logger 0.11.10`
- `equivalent 1.0.2`
- `errno 0.3.14`
- `fallible-iterator 0.3.0`
- `fallible-streaming-iterator 0.1.9`
- `fastrand 2.4.1`
- `find-msvc-tools 0.1.9`
- `flate2 1.1.9`
- `fnv 1.0.7`
- `foreign-types 0.3.2`
- `foreign-types-shared 0.1.1`
- `form_urlencoded 1.2.2`
- `futures-channel 0.3.32`
- `futures-core 0.3.32`
- `futures-io 0.3.32`
- `futures-macro 0.3.32`
- `futures-sink 0.3.32`
- `futures-task 0.3.32`
- `futures-util 0.3.32`
- `getrandom 0.2.17`
- `getrandom 0.3.4`
- `getrandom 0.4.2`
- `half 2.7.1`
- `hashbrown 0.14.5`
- `hashbrown 0.15.5`
- `hashbrown 0.16.1`
- `hashbrown 0.17.0`
- `hashlink 0.9.1`
- `heck 0.4.1`
- `heck 0.5.0`
- `hermit-abi 0.5.2`
- `http 1.4.0`
- `httparse 1.10.1`
- `httpdate 1.0.3`
- `hyper-tls 0.6.0`
- `id-arena 2.3.0`
- `ident_case 1.0.1`
- `idna 1.1.0`
- `idna_adapter 1.2.1`
- `indexmap 2.14.0`
- `indoc 2.0.7`
- `instant-distance 0.6.1`
- `ipnet 2.12.0`
- `iri-string 0.7.12`
- `is_terminal_polyfill 1.70.2`
- `itertools 0.10.5`
- `itertools 0.11.0`
- `itertools 0.12.1`
- `itertools 0.14.0`
- `itoa 1.0.18`
- `js-sys 0.3.94`
- `lazy_static 1.5.0`
- `leb128fmt 0.1.0`
- `libc 0.2.184`
- `lock_api 0.4.14`
- `log 0.4.29`
- `matrixmultiply 0.3.10`
- `mime 0.3.17`
- `minimal-lexical 0.2.1`
- `monostate 0.1.18`
- `monostate-impl 0.1.18`
- `native-tls 0.2.18`
- `ndarray 0.17.2`
- `num-complex 0.4.6`
- `num-integer 0.1.46`
- `num-traits 0.2.19`
- `num_cpus 1.17.0`
- `once_cell 1.21.4`
- `once_cell_polyfill 1.70.2`
- `openssl-macros 0.1.1`
- `openssl-probe 0.2.1`
- `ort 2.0.0-rc.11`
- `ort-sys 2.0.0-rc.11`
- `parking_lot 0.12.5`
- `parking_lot_core 0.9.12`
- `paste 1.0.15`
- `pem-rfc7468 1.0.0`
- `percent-encoding 2.3.2`
- `pin-project-lite 0.2.17`
- `pkg-config 0.3.32`
- `portable-atomic 1.13.1`
- `portable-atomic-util 0.2.6`
- `ppv-lite86 0.2.21`
- `prettyplease 0.2.37`
- `proc-macro2 1.0.106`
- `pyo3 0.21.2`
- `pyo3-build-config 0.21.2`
- `pyo3-ffi 0.21.2`
- `pyo3-macros 0.21.2`
- `pyo3-macros-backend 0.21.2`
- `quote 1.0.45`
- `r2d2 0.8.10`
- `rand 0.10.4`
- `rand 0.8.8`
- `rand 0.9.5`
- `rand_chacha 0.3.1`
- `rand_chacha 0.9.0`
- `rand_core 0.10.1`
- `rand_core 0.6.4`
- `rand_core 0.9.5`
- `rawpointer 0.2.1`
- `rayon 1.11.0`
- `rayon-cond 0.3.0`
- `rayon-cond 0.4.0`
- `rayon-core 1.13.0`
- `regex 1.12.3`
- `regex-automata 0.4.14`
- `regex-syntax 0.8.10`
- `reqwest 0.12.28`
- `rustls-pki-types 1.14.0`
- `rustversion 1.0.22`
- `scheduled-thread-pool 0.2.7`
- `scopeguard 1.2.0`
- `security-framework 3.7.0`
- `security-framework-sys 2.17.0`
- `semver 1.0.28`
- `serde 1.0.228`
- `serde_core 1.0.228`
- `serde_derive 1.0.228`
- `serde_json 1.0.149`
- `serde_path_to_error 0.1.20`
- `serde_urlencoded 0.7.1`
- `shlex 1.3.0`
- `signal-hook-registry 1.4.8`
- `smallvec 1.15.1`
- `socket2 0.6.3`
- `socks 0.3.4`
- `stable_deref_trait 1.2.1`
- `static_assertions 1.1.0`
- `syn 2.0.117`
- `system-configuration 0.7.0`
- `system-configuration-sys 0.6.0`
- `tempfile 3.27.0`
- `thiserror 1.0.69`
- `thiserror 2.0.18`
- `thiserror-impl 1.0.69`
- `thiserror-impl 2.0.18`
- `tinytemplate 1.2.1`
- `tokio-rustls 0.26.4`
- `unicode-normalization-alignments 0.1.12`
- `unicode-segmentation 1.13.2`
- `unicode-width 0.2.2`
- `unicode-xid 0.2.6`
- `unicode_categories 0.1.1`
- `unindent 0.2.4`
- `ureq 2.12.1`
- `ureq 3.3.0`
- `ureq-proto 0.6.0`
- `url 2.5.8`
- `utf8-zero 0.8.1`
- `utf8_iter 1.0.4`
- `utf8parse 0.2.2`
- `uuid 1.23.2`
- `vcpkg 0.2.15`
- `version_check 0.9.5`
- `wasm-bindgen 0.2.117`
- `wasm-bindgen-futures 0.4.67`
- `wasm-bindgen-macro 0.2.117`
- `wasm-bindgen-macro-support 0.2.117`
- `wasm-bindgen-shared 0.2.117`
- `wasm-streams 0.4.2`
- `web-sys 0.3.94`
- `web-time 1.1.0`
- `winapi 0.3.9`
- `winapi-i686-pc-windows-gnu 0.4.0`
- `winapi-x86_64-pc-windows-gnu 0.4.0`
- `windows-link 0.2.1`
- `windows-registry 0.6.1`
- `windows-result 0.4.1`
- `windows-strings 0.5.1`
- `windows-sys 0.52.0`
- `windows-sys 0.59.0`
- `windows-sys 0.60.2`
- `windows-sys 0.61.2`
- `windows-targets 0.52.6`
- `windows-targets 0.53.5`
- `windows_aarch64_gnullvm 0.52.6`
- `windows_aarch64_gnullvm 0.53.1`
- `windows_aarch64_msvc 0.52.6`
- `windows_aarch64_msvc 0.53.1`
- `windows_i686_gnu 0.52.6`
- `windows_i686_gnu 0.53.1`
- `windows_i686_gnullvm 0.52.6`
- `windows_i686_gnullvm 0.53.1`
- `windows_i686_msvc 0.52.6`
- `windows_i686_msvc 0.53.1`
- `windows_x86_64_gnu 0.52.6`
- `windows_x86_64_gnu 0.53.1`
- `windows_x86_64_gnullvm 0.52.6`
- `windows_x86_64_gnullvm 0.53.1`
- `windows_x86_64_msvc 0.52.6`
- `windows_x86_64_msvc 0.53.1`
- `zeroize 1.8.2`

### MIT (59)

- `axum 0.7.9`
- `axum-core 0.4.5`
- `bytes 1.11.1`
- `castaway 0.2.4`
- `compact_str 0.9.0`
- `console 0.15.11`
- `crunchy 0.2.4`
- `darling 0.20.11`
- `darling_core 0.20.11`
- `darling_macro 0.20.11`
- `h2 0.4.13`
- `http-body 1.0.1`
- `http-body-util 0.1.3`
- `hyper 1.9.0`
- `hyper-util 0.1.20`
- `indicatif 0.17.11`
- `is-terminal 0.4.17`
- `libredox 0.1.16`
- `libsqlite3-sys 0.28.0`
- `memoffset 0.9.1`
- `mio 1.2.0`
- `nom 7.1.3`
- `number_prefix 0.4.0`
- `onig 6.5.1`
- `onig_sys 69.9.1`
- `oorandom 11.1.5`
- `openssl-sys 0.9.118`
- `ordered-float 3.9.2`
- `plotters 0.3.7`
- `plotters-backend 0.3.7`
- `plotters-svg 0.3.7`
- `r2d2_sqlite 0.24.0`
- `redox_syscall 0.5.18`
- `redox_users 0.5.2`
- `rusqlite 0.31.0`
- `schannel 0.1.29`
- `simd-adler32 0.3.9`
- `slab 0.4.12`
- `strsim 0.11.1`
- `synstructure 0.13.2`
- `tokio 1.51.1`
- `tokio-macros 2.7.0`
- `tokio-native-tls 0.3.1`
- `tokio-util 0.7.18`
- `tower 0.5.3`
- `tower-http 0.6.8`
- `tower-layer 0.3.3`
- `tower-service 0.3.3`
- `tracing 0.1.44`
- `tracing-core 0.1.36`
- `tree-sitter 0.23.2`
- `tree-sitter-javascript 0.23.1`
- `tree-sitter-language 0.1.7`
- `tree-sitter-python 0.23.6`
- `tree-sitter-rust 0.23.3`
- `tree-sitter-typescript 0.23.2`
- `try-lock 0.2.5`
- `want 0.3.1`
- `zmij 1.0.21`

### Unicode-3.0 (18)

- `icu_collections 2.2.0`
- `icu_locale_core 2.2.0`
- `icu_normalizer 2.2.0`
- `icu_normalizer_data 2.2.0`
- `icu_properties 2.2.0`
- `icu_properties_data 2.2.0`
- `icu_provider 2.2.0`
- `litemap 0.8.2`
- `potential_utf 0.1.5`
- `tinystr 0.8.3`
- `writeable 0.6.3`
- `yoke 0.8.2`
- `yoke-derive 0.8.2`
- `zerofrom 0.1.7`
- `zerofrom-derive 0.1.7`
- `zerotrie 0.2.4`
- `zerovec 0.11.6`
- `zerovec-derive 0.11.3`

### Apache-2.0 OR Apache-2.0 WITH LLVM-exception OR MIT (14)

- `linux-raw-sys 0.12.1`
- `rustix 1.1.4`
- `wasi 0.11.1+wasi-snapshot-preview1`
- `wasip2 1.0.2+wasi-0.2.9`
- `wasip3 0.4.0+wasi-0.3.0-rc-2026-01-06`
- `wasm-encoder 0.244.0`
- `wasm-metadata 0.244.0`
- `wasmparser 0.244.0`
- `wit-bindgen 0.51.0`
- `wit-bindgen-core 0.51.0`
- `wit-bindgen-rust 0.51.0`
- `wit-bindgen-rust-macro 0.51.0`
- `wit-component 0.244.0`
- `wit-parser 0.244.0`

### Apache-2.0 (13)

- `ciborium 0.2.2`
- `ciborium-io 0.2.2`
- `ciborium-ll 0.2.2`
- `esaxx-rs 0.1.10`
- `fastembed 5.13.2`
- `hf-hub 0.4.3`
- `lzma-rust2 0.15.7`
- `openssl 0.10.82`
- `safetensors 0.7.0`
- `spm_precompiled 0.1.4`
- `sync_wrapper 1.0.2`
- `tokenizers 0.19.1`
- `tokenizers 0.22.2`

### MIT OR Unlicense (8)

- `aho-corasick 1.1.4`
- `byteorder 1.5.0`
- `jiff 0.2.23`
- `jiff-static 0.2.23`
- `memchr 2.8.0`
- `same-file 1.0.6`
- `walkdir 2.5.0`
- `winapi-util 0.1.11`

### Apache-2.0 OR MIT OR Zlib (3)

- `macro_rules_attribute 0.2.2`
- `macro_rules_attribute-proc_macro 0.2.2`
- `miniz_oxide 0.8.9`

### CDLA-Permissive-2.0 (3)

- `webpki-root-certs 1.0.6`
- `webpki-roots 0.26.11`
- `webpki-roots 1.0.6`

### ISC (3)

- `hmac-sha256 1.1.14`
- `rustls-webpki 0.103.17`
- `untrusted 0.9.0`

### Apache-2.0 OR BSD-2-Clause OR MIT (2)

- `zerocopy 0.8.48`
- `zerocopy-derive 0.8.48`

### Apache-2.0 OR ISC OR MIT (2)

- `hyper-rustls 0.27.7`
- `rustls 0.23.46`

### Apache-2.0 OR LGPL-2.1-or-later OR MIT (2)

- `r-efi 5.3.0`
- `r-efi 6.0.0`

### Zlib (2)

- `foldhash 0.1.5`
- `foldhash 0.2.0`

### (Apache-2.0 OR MIT) AND BSD-3-Clause (1)

- `encoding_rs 0.8.35`

### (MIT OR Apache-2.0) AND Unicode-3.0 (1)

- `unicode-ident 1.0.24`

### 0BSD OR Apache-2.0 OR MIT (1)

- `adler2 2.0.1`

### Apache-2.0 AND ISC (1)

- `ring 0.17.14`

### Apache-2.0 OR Apache-2.0 WITH LLVM-exception OR CC0-1.0 (1)

- `blake3 1.8.5`

### Apache-2.0 OR BSL-1.0 (1)

- `ryu 1.0.23`

### Apache-2.0 OR CC0-1.0 OR MIT-0 (1)

- `constant_time_eq 0.4.2`

### Apache-2.0 WITH LLVM-exception (1)

- `target-lexicon 0.12.16`

### BSD-2-Clause (1)

- `arrayref 0.3.9`

### BSD-3-Clause (1)

- `subtle 2.6.1`

### MIT AND BSD-3-Clause (1)

- `matchit 0.7.3`

### MPL-2.0 (1)

- `option-ext 0.2.0`

<!-- END GENERATED INVENTORY -->
