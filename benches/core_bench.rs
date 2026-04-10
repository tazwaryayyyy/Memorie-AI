use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use memoire::Memoire;

// ─── Fixtures ─────────────────────────────────────────────────────────────────

const SHORT: &str =
    "Fixed a null pointer dereference in the authentication middleware.";

const MEDIUM: &str =
    "Refactored the database connection pool to use async/await throughout. \
     Changed the max pool size from 5 to 20 and added idle connection pruning \
     every 30 seconds. This reduced connection setup overhead on burst traffic \
     from 340ms to under 10ms per request on the staging environment.";

const LONG: &str =
    "Session 2024-01-20: Investigated a production incident where the payment \
     processing service was failing for approximately 3% of transactions. Root \
     cause was a race condition in the idempotency key validation logic — two \
     concurrent requests with the same key could both pass the uniqueness check \
     before either committed, resulting in duplicate charges. Fixed by moving the \
     uniqueness check inside a serialisable transaction. Added an integration test \
     that fires 50 concurrent payment requests with the same idempotency key and \
     asserts only one succeeds. Deployed the fix at 14:32 UTC and confirmed the \
     error rate dropped to zero within two minutes. Post-incident review scheduled \
     for Friday. Also took the opportunity to add structured logging to the payment \
     service — all events now include trace_id, user_id, and amount_cents fields \
     for easier Datadog querying.";

fn make_store() -> Memoire {
    Memoire::in_memory().expect("failed to create memoire")
}

fn seed_store(m: &Memoire, n: usize) {
    let templates = [
        "Fixed a bug in the {} module related to authentication and token validation",
        "Refactored the {} service to improve throughput and reduce database load",
        "Added unit tests for the {} component covering edge cases and error paths",
        "Diagnosed a memory leak in the {} handler and patched the Drop implementation",
        "Upgraded {} from v1 to v2 and migrated all API call sites to the new interface",
    ];
    let domains = [
        "auth", "payment", "database", "cache", "websocket",
        "worker", "scheduler", "api-gateway", "metrics", "search",
    ];
    for i in 0..n {
        let t = templates[i % templates.len()];
        let d = domains[i % domains.len()];
        let content = t.replace("{}", d);
        m.remember(&content).unwrap();
    }
}

// ─── remember() benchmarks ────────────────────────────────────────────────────

fn bench_remember(c: &mut Criterion) {
    let mut group = c.benchmark_group("remember");

    for (name, input) in [("short", SHORT), ("medium", MEDIUM), ("long", LONG)] {
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_with_input(BenchmarkId::new("input_size", name), input, |b, text| {
            let m = make_store();
            b.iter(|| {
                m.remember(black_box(text)).unwrap();
            });
        });
    }

    group.finish();
}

// ─── recall() benchmarks ─────────────────────────────────────────────────────

fn bench_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall");

    for store_size in [100, 1_000, 5_000] {
        group.bench_with_input(
            BenchmarkId::new("store_size", store_size),
            &store_size,
            |b, &n| {
                let m = make_store();
                seed_store(&m, n);
                b.iter(|| {
                    m.recall(black_box("authentication security bug"), black_box(5)).unwrap()
                });
            },
        );
    }

    group.finish();
}

// ─── top_k benchmarks ────────────────────────────────────────────────────────

fn bench_recall_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_top_k");
    let m = make_store();
    seed_store(&m, 1000);

    for k in [1usize, 5, 10, 25] {
        group.bench_with_input(BenchmarkId::new("top_k", k), &k, |b, &top| {
            b.iter(|| m.recall(black_box("database performance"), black_box(top)).unwrap());
        });
    }

    group.finish();
}

// ─── chunker benchmark ───────────────────────────────────────────────────────

fn bench_chunker(c: &mut Criterion) {
    use memoire::chunker::{ChunkerConfig, chunk_text};

    let mut group = c.benchmark_group("chunker");
    let cfg = ChunkerConfig::default();

    // Build inputs of varying sizes
    let small  = "word ".repeat(50);
    let medium = "word ".repeat(500);
    let large  = "word ".repeat(5000);

    for (name, input) in [("50w", &small), ("500w", &medium), ("5000w", &large)] {
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_with_input(BenchmarkId::new("words", name), input.as_str(), |b, text| {
            b.iter(|| chunk_text(black_box(text), black_box(&cfg)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_remember, bench_recall, bench_recall_top_k, bench_chunker);
criterion_main!(benches);
