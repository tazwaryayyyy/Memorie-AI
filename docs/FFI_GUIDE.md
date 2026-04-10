# Memoire FFI Integration Guide

How to call `libmemoire` from Python, Node.js, Go, Ruby, and C.

## Build the library first

```bash
cargo build --release

# Linux:   target/release/libmemoire.so
# macOS:   target/release/libmemoire.dylib
# Windows: target/release/memoire.dll
```

---

## Python (ctypes — built-in, zero deps)

```python
import ctypes, json, sys

lib = ctypes.CDLL("target/release/libmemoire.so")  # adjust path/extension

# Declare signatures
lib.memoire_new.argtypes      = [ctypes.c_char_p];  lib.memoire_new.restype       = ctypes.c_void_p
lib.memoire_free.argtypes     = [ctypes.c_void_p];  lib.memoire_free.restype      = None
lib.memoire_remember.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.memoire_remember.restype  = ctypes.c_int
lib.memoire_recall.argtypes   = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.memoire_recall.restype    = ctypes.c_char_p
lib.memoire_free_string.argtypes = [ctypes.c_char_p]; lib.memoire_free_string.restype = None

h = lib.memoire_new(b"agent.db")
lib.memoire_remember(h, b"Fixed the auth bug today")

raw = lib.memoire_recall(h, b"authentication", 3)
results = json.loads(raw.decode())
lib.memoire_free_string(raw)        # ← MUST free

for r in results:
    print(f"[{r['score']:.3f}] {r['content']}")

lib.memoire_free(h)
```

Or use the included wrapper (zero boilerplate):

```python
from memoire import Memoire  # from bindings/python/

with Memoire("agent.db") as m:
    m.remember("Fixed the auth bug today")
    for r in m.recall("authentication", top_k=3):
        print(f"[{r.score:.3f}] {r.content}")
```

---

## Node.js (ffi-napi)

```bash
npm install ffi-napi ref-napi
```

```javascript
const ffi  = require("ffi-napi");
const ref  = require("ref-napi");

const voidPtr = ref.refType(ref.types.void);
const charPtr = ref.refType(ref.types.char);

const lib = ffi.Library("target/release/libmemoire", {
  memoire_new:         [voidPtr,  ["string"]],
  memoire_free:        ["void",   [voidPtr]],
  memoire_remember:    ["int",    [voidPtr, "string"]],
  memoire_recall:      [charPtr,  [voidPtr, "string", "int"]],
  memoire_free_string: ["void",   [charPtr]],
  memoire_count:       ["int64",  [voidPtr]],
  memoire_free:        ["void",   [voidPtr]],
});

const h = lib.memoire_new("agent.db");
lib.memoire_remember(h, "Fixed the auth bug");

const rawPtr = lib.memoire_recall(h, "auth", 3);
const json   = JSON.parse(ref.readCString(rawPtr));
lib.memoire_free_string(rawPtr);    // ← MUST free

json.forEach(r => console.log(`[${r.score.toFixed(3)}] ${r.content}`));
lib.memoire_free(h);
```

---

## Go (cgo)

```go
package main

/*
#cgo LDFLAGS: -lmemoire
#include "include/memoire.h"
#include <stdlib.h>
*/
import "C"
import (
    "encoding/json"
    "fmt"
    "unsafe"
)

func main() {
    h := C.memoire_new(C.CString("agent.db"))
    defer C.memoire_free(h)

    content := C.CString("Fixed the auth bug today")
    defer C.free(unsafe.Pointer(content))
    C.memoire_remember(h, content)

    query := C.CString("authentication")
    defer C.free(unsafe.Pointer(query))
    raw := C.memoire_recall(h, query, 3)
    defer C.memoire_free_string(raw)

    var results []struct {
        ID      int64   `json:"id"`
        Content string  `json:"content"`
        Score   float32 `json:"score"`
    }
    json.Unmarshal([]byte(C.GoString(raw)), &results)

    for _, r := range results {
        fmt.Printf("[%.3f] %s\n", r.Score, r.Content)
    }
}
```

Build:
```bash
CGO_LDFLAGS="-L./target/release -lmemoire" \
LD_LIBRARY_PATH=./target/release \
  go run main.go
```

Or use the included wrapper:
```bash
cd bindings/go
CGO_LDFLAGS="-L../../target/release -lmemoire" go test ./memoire/
```

---

## Ruby (fiddle — built-in)

```ruby
require 'fiddle'
require 'fiddle/import'
require 'json'

module Memoire
  extend Fiddle::Importer
  dlload 'target/release/libmemoire.so'

  extern 'void* memoire_new(const char*)'
  extern 'void  memoire_free(void*)'
  extern 'int   memoire_remember(void*, const char*)'
  extern 'char* memoire_recall(void*, const char*, int)'
  extern 'void  memoire_free_string(char*)'
  extern 'long long memoire_count(void*)'
end

h = Memoire.memoire_new("agent.db")
Memoire.memoire_remember(h, "Fixed the auth bug today")

raw_ptr = Memoire.memoire_recall(h, "authentication", 3)
results = JSON.parse(raw_ptr.to_s)
Memoire.memoire_free_string(raw_ptr)

results.each { |r| puts "[#{r['score'].round(3)}] #{r['content']}" }
Memoire.memoire_free(h)
```

---

## C / C++

```c
#include "include/memoire.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    MemoireHandle* h = memoire_new("agent.db");
    if (!h) { fprintf(stderr, "failed to open\n"); return 1; }

    memoire_remember(h, "Fixed the auth bug today");

    char* json = memoire_recall(h, "authentication", 3);
    if (json) {
        printf("%s\n", json);
        memoire_free_string(json);   /* MUST free */
    }

    printf("total chunks: %lld\n", memoire_count(h));
    memoire_free(h);
    return 0;
}
```

Compile:
```bash
gcc -o demo demo.c \
    -I./include \
    -L./target/release \
    -lmemoire \
    -Wl,-rpath,./target/release
./demo
```

---

## C# / .NET (P/Invoke)

```csharp
using System;
using System.Runtime.InteropServices;
using System.Text.Json;

class Memoire {
    const string Lib = "libmemoire";

    [DllImport(Lib)] static extern IntPtr memoire_new(string dbPath);
    [DllImport(Lib)] static extern void   memoire_free(IntPtr h);
    [DllImport(Lib)] static extern int    memoire_remember(IntPtr h, string content);
    [DllImport(Lib)] static extern IntPtr memoire_recall(IntPtr h, string query, int topK);
    [DllImport(Lib)] static extern void   memoire_free_string(IntPtr s);
    [DllImport(Lib)] static extern long   memoire_count(IntPtr h);

    static void Main() {
        var h = memoire_new("agent.db");
        memoire_remember(h, "Fixed the auth bug today");

        var rawPtr = memoire_recall(h, "authentication", 3);
        var json   = Marshal.PtrToStringUTF8(rawPtr)!;
        memoire_free_string(rawPtr);                    // MUST free

        var results = JsonDocument.Parse(json).RootElement;
        foreach (var r in results.EnumerateArray()) {
            Console.WriteLine($"[{r.GetProperty("score").GetDouble():F3}] " +
                              r.GetProperty("content").GetString());
        }

        memoire_free(h);
    }
}
```

---

## String Ownership Rules (ALL languages)

| Function | Returns | Caller must... |
|---|---|---|
| `memoire_new` | opaque pointer | `memoire_free(h)` when done |
| `memoire_remember` | `int` (count) | nothing |
| `memoire_recall` | `char*` JSON string | `memoire_free_string(ptr)` |
| `memoire_forget` | `int` (status) | nothing |
| `memoire_count` | `int64` | nothing |
| `memoire_clear` | `int` (status) | nothing |

**Never** pass the JSON string pointer to the system `free()` or your language runtime's allocator — it must go through `memoire_free_string` which routes it back to Rust's allocator.
