// Demo: AI agent using Memoire for persistent semantic memory.
//
// Build and run:
//
//	cargo build --release
//	CGO_LDFLAGS="-L../../target/release -lmemoire" \
//	LD_LIBRARY_PATH=../../target/release \
//	  go run main.go
package main

import (
	"fmt"
	"log"

	"github.com/yourusername/memoire/bindings/go/memoire"
)

func main() {
	m, err := memoire.New(":memory:")
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	fmt.Println("=== Memoire Go Demo ===\n")

	notes := []string{
		"Fixed race condition in payment idempotency key validation — two concurrent requests could both pass the uniqueness check before either committed.",
		"Replaced synchronous bcrypt with async argon2id. Auth endpoint p99 dropped from 420ms to 38ms.",
		"Diagnosed N+1 query in the /users dashboard. Consolidated to a single JOIN query. Load time: 3.8s → 290ms.",
		"Added Redis-based rate limiting to /api/reset-password: 5 requests per hour per IP.",
		"Memory leak in the WebSocket handler — connections were not removed from the registry on client disconnect.",
	}

	fmt.Println("Storing memories...")
	for _, note := range notes {
		n, err := m.Remember(note)
		if err != nil {
			log.Printf("remember error: %v", err)
			continue
		}
		fmt.Printf("  ✓ (%d chunk) %s\n", n, truncate(note, 65))
	}

	count, _ := m.Count()
	fmt.Printf("\nTotal chunks: %d\n\n", count)

	queries := []struct {
		text string
		topK int
	}{
		{"security vulnerabilities and authentication", 2},
		{"database query performance", 2},
		{"memory leaks and resource management", 2},
	}

	for _, q := range queries {
		fmt.Printf("Query: %q\n", q.text)
		results, err := m.Recall(q.text, q.topK)
		if err != nil {
			log.Printf("recall error: %v", err)
			continue
		}
		for _, r := range results {
			fmt.Printf("  [%.4f] %s\n", r.Score, truncate(r.Content, 85))
		}
		fmt.Println()
	}
}

func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "…"
}
