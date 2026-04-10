package memoire_test

import (
	"testing"

	"github.com/yourusername/memoire/bindings/go/memoire"
)

func setup(t *testing.T) *memoire.Memoire {
	t.Helper()
	m, err := memoire.New(":memory:")
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	t.Cleanup(m.Close)
	return m
}

func TestRememberAndCount(t *testing.T) {
	m := setup(t)

	n, err := m.Remember("Fixed the off-by-one error in pagination")
	if err != nil {
		t.Fatalf("Remember() error: %v", err)
	}
	if n < 1 {
		t.Errorf("expected >= 1 chunk, got %d", n)
	}

	count, err := m.Count()
	if err != nil {
		t.Fatalf("Count() error: %v", err)
	}
	if count < 1 {
		t.Errorf("expected count >= 1, got %d", count)
	}
}

func TestRecall(t *testing.T) {
	m := setup(t)

	_, _ = m.Remember("Fixed critical null pointer dereference in auth middleware")
	_, _ = m.Remember("Refactored database connection pool for better throughput")
	_, _ = m.Remember("Added unit tests for the payment processing module")

	results, err := m.Recall("authentication security bug", 3)
	if err != nil {
		t.Fatalf("Recall() error: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results, got none")
	}
	if results[0].Score <= 0 {
		t.Errorf("expected positive score, got %f", results[0].Score)
	}
	// Scores should be descending
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("scores not descending at index %d", i)
		}
	}
}

func TestForget(t *testing.T) {
	m := setup(t)

	_, _ = m.Remember("temporary memory")
	count, _ := m.Count()

	results, _ := m.Recall("temporary", 1)
	if len(results) == 0 {
		t.Fatal("expected to find the memory")
	}

	deleted, err := m.Forget(results[0].ID)
	if err != nil {
		t.Fatalf("Forget() error: %v", err)
	}
	if !deleted {
		t.Error("expected deleted=true")
	}

	newCount, _ := m.Count()
	if newCount != count-1 {
		t.Errorf("expected count %d, got %d", count-1, newCount)
	}
}

func TestClear(t *testing.T) {
	m := setup(t)
	_, _ = m.Remember("one")
	_, _ = m.Remember("two")

	if err := m.Clear(); err != nil {
		t.Fatalf("Clear() error: %v", err)
	}
	n, _ := m.Count()
	if n != 0 {
		t.Errorf("expected 0 after clear, got %d", n)
	}
}

func TestEmptyRecall(t *testing.T) {
	m := setup(t)
	results, err := m.Recall("anything", 5)
	if err != nil {
		t.Fatalf("Recall() on empty store error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results, got %d", len(results))
	}
}
