// Package memoire provides Go bindings for the Memoire semantic memory engine.
//
// Memoire is a local-first, embeddable long-term memory engine for AI agents.
// It stores text as semantic embeddings using all-MiniLM-L6-v2 (ONNX) and
// retrieves memories by cosine similarity — entirely on-device, no cloud.
//
// Build the shared library first:
//
//	cargo build --release   # from the memoire repo root
//
// Then compile your Go binary with the library on the path:
//
//	CGO_LDFLAGS="-L/path/to/memoire/target/release -lmemoire" \
//	LD_LIBRARY_PATH=/path/to/memoire/target/release             \
//	  go run main.go
//
// # Example
//
//	m, err := memoire.New("./agent.db")
//	if err != nil { log.Fatal(err) }
//	defer m.Close()
//
//	m.Remember("Fixed the JWT issuer validation bug in auth middleware")
//
//	results, err := m.Recall("authentication issues", 3)
//	for _, r := range results {
//	    fmt.Printf("[%.3f] %s\n", r.Score, r.Content)
//	}
package memoire

/*
#cgo LDFLAGS: -lmemoire
#include "../../include/memoire.h"
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"unsafe"
)

// Memory is a single stored memory chunk with its similarity score.
type Memory struct {
	ID        int64   `json:"id"`
	Content   string  `json:"content"`
	Score     float32 `json:"score"`
	CreatedAt int64   `json:"created_at"`
}

// Memoire wraps the native Memoire handle.
type Memoire struct {
	handle unsafe.Pointer
}

// New opens or creates a Memoire database at dbPath.
// Pass ":memory:" for an ephemeral in-memory store.
//
// On first call this downloads the all-MiniLM-L6-v2 model (~23 MB)
// and caches it at $HF_HOME or ~/.cache/huggingface/hub/.
func New(dbPath string) (*Memoire, error) {
	cPath := C.CString(dbPath)
	defer C.free(unsafe.Pointer(cPath))

	h := C.memoire_new(cPath)
	if h == nil {
		return nil, fmt.Errorf("failed to open Memoire at %q", dbPath)
	}
	return &Memoire{handle: unsafe.Pointer(h)}, nil
}

// Close releases the native handle. Always defer this after New.
func (m *Memoire) Close() {
	if m.handle != nil {
		C.memoire_free((*C.MemoireHandle)(m.handle))
		m.handle = nil
	}
}

// Remember chunks, embeds, and stores content.
// Returns the number of chunks stored (≥ 1 on success).
func (m *Memoire) Remember(content string) (int, error) {
	cContent := C.CString(content)
	defer C.free(unsafe.Pointer(cContent))

	n := C.memoire_remember((*C.MemoireHandle)(m.handle), cContent)
	if n < 0 {
		return 0, errors.New("memoire_remember() failed")
	}
	return int(n), nil
}

// Recall returns the topK memories most semantically similar to query.
func (m *Memoire) Recall(query string, topK int) ([]Memory, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	raw := C.memoire_recall((*C.MemoireHandle)(m.handle), cQuery, C.int(topK))
	if raw == nil {
		return nil, errors.New("memoire_recall() failed")
	}
	defer C.memoire_free_string(raw)

	var memories []Memory
	if err := json.Unmarshal([]byte(C.GoString(raw)), &memories); err != nil {
		return nil, fmt.Errorf("failed to parse recall JSON: %w", err)
	}
	return memories, nil
}

// Forget deletes a memory by its id.
// Returns true if the memory existed and was deleted.
func (m *Memoire) Forget(id int64) (bool, error) {
	r := C.memoire_forget((*C.MemoireHandle)(m.handle), C.int64_t(id))
	if r < 0 {
		return false, errors.New("memoire_forget() failed")
	}
	return r == 1, nil
}

// Count returns the total number of stored memory chunks.
func (m *Memoire) Count() (int64, error) {
	n := C.memoire_count((*C.MemoireHandle)(m.handle))
	if n < 0 {
		return 0, errors.New("memoire_count() failed")
	}
	return int64(n), nil
}

// Clear erases all memories. This cannot be undone.
func (m *Memoire) Clear() error {
	if C.memoire_clear((*C.MemoireHandle)(m.handle)) < 0 {
		return errors.New("memoire_clear() failed")
	}
	return nil
}
