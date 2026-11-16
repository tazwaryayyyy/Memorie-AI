//! Guards against drift between `src/ffi.rs`, the checked-in C header and the
//! cbindgen export list. `include/memoire.h` is hand-maintained (cbindgen is not
//! wired into `build.rs`), so nothing else notices a missing declaration.

use std::fs;

fn read(rel: &str) -> String {
    let path = format!("{}/{rel}", env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Names of every `extern "C" fn memoire_*` defined in `src/ffi.rs`.
fn exported_symbols() -> Vec<String> {
    read("src/ffi.rs")
        .lines()
        .filter_map(|line| line.split("extern \"C\" fn ").nth(1))
        .filter_map(|rest| rest.split('(').next())
        .filter(|name| name.starts_with("memoire_"))
        .map(str::to_owned)
        .collect()
}

#[test]
fn exported_symbol_list_is_not_empty() {
    // 12 at the time of writing; fails loudly if the scan above stops matching.
    assert!(exported_symbols().len() >= 12, "{:?}", exported_symbols());
}

#[test]
fn every_exported_symbol_is_declared_in_the_header() {
    let header = read("include/memoire.h");
    for name in exported_symbols() {
        let declared =
            header.contains(&format!(" {name}(")) || header.contains(&format!("*{name}("));
        assert!(
            declared,
            "{name} is exported by src/ffi.rs but not declared in include/memoire.h"
        );
    }
}

#[test]
fn every_exported_symbol_is_in_cbindgen_export_list() {
    let cfg = read("cbindgen.toml");
    for name in exported_symbols() {
        assert!(
            cfg.contains(&format!("\"{name}\"")),
            "{name} is exported by src/ffi.rs but missing from cbindgen.toml [export].include"
        );
    }
}

#[test]
fn header_does_not_promise_at_least_one_chunk() {
    // memoire_remember returns 0 for blank input, duplicates and rejected chunks.
    for (file, text) in [
        ("include/memoire.h", read("include/memoire.h")),
        ("src/ffi.rs", read("src/ffi.rs")),
    ] {
        assert!(
            !text.contains("chunks stored (>= 1)"),
            "{file} still documents memoire_remember as returning >= 1"
        );
    }
}
