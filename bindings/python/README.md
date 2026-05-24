# memoire Python binding

Python ctypes binding for the Memoire native library.

Install from a source checkout:

```bash
python -m pip install ./bindings/python
```

The package expects the native library to be available. Build it with:

```bash
cargo build --release
```

For installed environments, set `MEMOIRE_LIB` to the platform library:

- Linux: `target/release/libmemoire.so`
- macOS: `target/release/libmemoire.dylib`
- Windows: `target/release/memoire.dll`

See the repository README and `docs/RELEASE.md` for the full release workflow.
