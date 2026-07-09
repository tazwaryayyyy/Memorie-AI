//! Background maintenance scheduling.
//!
//! Spawns Tokio tasks for:
//! - Time-based `maintenance_pass()` on a configurable interval.
//! - Count-based trigger: run maintenance when insert_count reaches threshold.
//! - Reinforcement count-based trigger: run `recompute_prototypes` every 100 reinforcements.
//!
//! All tasks are guarded with `cfg(not(target_arch = "wasm32"))`.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use tokio::time::interval;

/// Spawns a background task that runs `store_fn()` on a time-based schedule.
///
/// The first tick is skipped to avoid running immediately on startup.
/// Returns a `JoinHandle` the caller can abort.
///
/// `store_fn` is a closure returning `Result<()>` â€” pass a closure that
/// borrows an `Arc<Store>` without exposing Store internals here.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_maintenance_scheduler(
    store_fn: Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync>,
    interval_secs: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(interval_secs));
        ticker.tick().await; // skip immediate first tick
        loop {
            ticker.tick().await;
            if let Err(e) = store_fn() {
                // Log and continue â€” never panic in a background task
                log::warn!("maintenance_pass failed: {e}");
            }
        }
    })
}

/// Spawns a background watcher that triggers `store_fn()` when `insert_count`
/// reaches `threshold`, then resets the counter to 0.
/// Polls every 500ms â€” low overhead.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_count_watcher(
    store_fn: Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync>,
    insert_count: Arc<AtomicU32>,
    threshold: u32,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(500));
        loop {
            ticker.tick().await;
            if insert_count.load(Ordering::Relaxed) >= threshold {
                insert_count.store(0, Ordering::Relaxed);
                if let Err(e) = store_fn() {
                    log::warn!("count-triggered maintenance_pass failed: {e}");
                }
            }
        }
    })
}
/// Spawns a background watcher that triggers `recompute_fn()` whenever the
/// reinforcement counter reaches `threshold`, then resets it to 0.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_reinforcement_watcher(
    recompute_fn: Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync>,
    reinforcement_count: Arc<AtomicU32>,
    threshold: u32,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(500));
        loop {
            ticker.tick().await;
            if reinforcement_count.load(Ordering::Relaxed) >= threshold {
                reinforcement_count.store(0, Ordering::Relaxed);
                if let Err(e) = recompute_fn() {
                    log::warn!("prototype recomputation failed: {e}");
                }
            }
        }
    })
}
