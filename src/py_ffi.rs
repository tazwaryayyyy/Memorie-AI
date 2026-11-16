use crate::panic_guard::catch_panic;
use crate::{store::Memory, Memoire};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

pyo3::create_exception!(memoire_engine, MemoireError, PyException);

#[pyclass(name = "Memory")]
#[derive(Clone)]
pub struct PyMemory {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub trust: f32,
    #[pyo3(get)]
    pub trust_variance: f64,
    #[pyo3(get)]
    pub trust_samples: i64,
    #[pyo3(get)]
    pub trust_last_session_id: Option<String>,
    #[pyo3(get)]
    pub effective_weight: f32,
    #[pyo3(get)]
    pub reinforcement_count: i64,
    #[pyo3(get)]
    pub ingestion_driver: String,
    #[pyo3(get)]
    pub uncertainty: f32,
    #[pyo3(get)]
    pub state: String,
    #[pyo3(get)]
    pub created_at: i64,
    #[pyo3(get)]
    pub last_used_at: Option<i64>,
}

#[pymethods]
impl PyMemory {
    fn __repr__(&self) -> String {
        let preview = crate::text::preview(&self.content, 60);
        format!(
            "Memory(id={}, score={:.3}, trust={:.3}, uncertainty={:.3}, state={:?}, content={:?})",
            self.id, self.score, self.trust, self.uncertainty, self.state, preview
        )
    }
}

fn py_memory(m: Memory) -> PyMemory {
    PyMemory {
        id: m.id,
        content: m.content,
        score: m.score,
        trust: m.trust,
        trust_variance: m.trust_variance,
        trust_samples: m.trust_samples,
        trust_last_session_id: m.trust_last_session_id,
        effective_weight: m.effective_weight,
        reinforcement_count: m.reinforcement_count,
        ingestion_driver: m.ingestion_driver,
        uncertainty: m.uncertainty,
        state: m.state,
        created_at: m.created_at,
        last_used_at: m.last_used_at,
    }
}

#[pyclass(name = "RecallAttribution")]
#[derive(Clone)]
pub struct PyRecallAttribution {
    #[pyo3(get)]
    pub memory: PyMemory,
    #[pyo3(get)]
    pub semantic_score: f32,
    #[pyo3(get)]
    pub trust: f32,
    #[pyo3(get)]
    pub uncertainty: f32,
    #[pyo3(get)]
    pub effective_weight: f32,
    #[pyo3(get)]
    pub rank_score: f32,
    #[pyo3(get)]
    pub reasons: Vec<String>,
}

#[pyclass(name = "Memoire")]
pub struct PyMemoire {
    /// Wrapped in Arc<Mutex> so the Python object is safe to share across
    /// threads and concurrent calls block instead of panicking. Always access
    /// it through [`PyMemoire::with_inner`].
    inner: Arc<Mutex<Memoire>>,
}

fn db_err(e: impl ToString) -> PyErr {
    PyErr::new::<MemoireError, _>(e.to_string())
}

fn panic_err(what: &str, msg: String) -> PyErr {
    PyErr::new::<MemoireError, _>(format!("internal error: panic in {what}: {msg}"))
}

impl PyMemoire {
    /// Lock the inner `Memoire` and run `f` on it, converting a Rust panic into
    /// `MemoireError`.
    ///
    /// The panic is caught while the guard is still held, so the guard is
    /// dropped normally and the mutex is not poisoned. Without this, PyO3
    /// would still stop the panic at the Python boundary, but as a
    /// `PanicException` (a `BaseException`, not caught by `except Exception`)
    /// and the poisoned lock would fail every later call on the instance.
    /// State that `f` had half-updated is not rolled back; locks inside
    /// `Memoire` held at the panic point stay poisoned and later calls report
    /// an error.
    fn with_inner<T>(
        &self,
        what: &'static str,
        f: impl FnOnce(&mut Memoire) -> PyResult<T>,
    ) -> PyResult<T> {
        let mut guard = self.inner.lock().map_err(|_| {
            db_err("Memoire instance is unusable: an earlier internal panic poisoned its lock")
        })?;
        match catch_panic(|| f(&mut guard)) {
            Ok(result) => result,
            Err(msg) => Err(panic_err(what, msg)),
        }
    }
}

#[pymethods]
impl PyMemoire {
    #[new]
    #[pyo3(signature = (db_path = "./memoire.db", namespace = "default"))]
    fn new(db_path: &str, namespace: &str) -> PyResult<Self> {
        // Opening can panic (e.g. ONNX Runtime / model initialisation), and
        // there is no lock to protect yet, so catch it here too.
        let built = catch_panic(|| {
            if db_path == ":memory:" {
                Memoire::in_memory_ns(namespace)
            } else {
                Memoire::new_ns(db_path, namespace)
            }
        })
        .map_err(|msg| panic_err("Memoire::new", msg))?;
        let inner = built.map_err(db_err)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    fn remember(&self, content: &str) -> PyResult<usize> {
        self.with_inner("remember", |m| {
            m.remember(content).map(|ids| ids.len()).map_err(db_err)
        })
    }

    /// Like `remember`, but returns the row ids that were actually inserted.
    /// The list is shorter than the chunk count (possibly empty) when a chunk is
    /// an exact duplicate of an existing memory or is rejected at ingestion.
    fn remember_ids(&self, content: &str) -> PyResult<Vec<i64>> {
        self.with_inner("remember_ids", |m| m.remember(content).map_err(db_err))
    }

    fn recall(&self, query: &str, top_k: usize) -> PyResult<Vec<PyMemory>> {
        self.with_inner("recall", |m| {
            let memories = m.recall(query, top_k).map_err(db_err)?;
            Ok(memories.into_iter().map(py_memory).collect())
        })
    }

    #[pyo3(signature = (query, top_k, mmr_lambda = 0.5))]
    fn recall_mmr(&self, query: &str, top_k: usize, mmr_lambda: f32) -> PyResult<Vec<PyMemory>> {
        self.with_inner("recall_mmr", |m| {
            let memories = m.recall_mmr(query, top_k, mmr_lambda).map_err(db_err)?;
            Ok(memories.into_iter().map(py_memory).collect())
        })
    }

    fn recall_reranked(&self, query: &str, top_k: usize) -> PyResult<Vec<PyMemory>> {
        self.with_inner("recall_reranked", |m| {
            let memories = m.recall_reranked(query, top_k).map_err(db_err)?;
            Ok(memories.into_iter().map(py_memory).collect())
        })
    }

    fn recall_explained(&self, query: &str, top_k: usize) -> PyResult<Vec<PyRecallAttribution>> {
        self.with_inner("recall_explained", |m| {
            let attributions = m.recall_explained(query, top_k).map_err(db_err)?;
            Ok(attributions
                .into_iter()
                .map(|a| PyRecallAttribution {
                    memory: py_memory(a.memory),
                    semantic_score: a.semantic_score,
                    trust: a.trust,
                    uncertainty: a.uncertainty,
                    effective_weight: a.effective_weight,
                    rank_score: a.rank_score,
                    reasons: a.reasons,
                })
                .collect())
        })
    }

    fn recompute_prototypes(&self) -> PyResult<()> {
        self.with_inner("recompute_prototypes", |m| {
            m.recompute_prototypes().map_err(db_err)
        })
    }

    fn forget(&self, memory_id: i64) -> PyResult<bool> {
        self.with_inner("forget", |m| m.forget(memory_id).map_err(db_err))
    }

    fn count(&self) -> PyResult<i64> {
        self.with_inner("count", |m| m.count().map_err(db_err))
    }

    fn clear(&self) -> PyResult<()> {
        self.with_inner("clear", |m| m.clear().map_err(db_err))
    }

    fn export_namespace(&self) -> PyResult<String> {
        self.with_inner("export_namespace", |m| {
            let value = m.export_namespace().map_err(db_err)?;
            serde_json::to_string(&value).map_err(db_err)
        })
    }

    /// `preserve_trust=False` (default) imports content only; `True` restores the
    /// snapshot's trust metadata within bounds. See `memoire::ImportTrust`.
    #[pyo3(signature = (snapshot_str, preserve_trust = false))]
    fn import_namespace(&self, snapshot_str: &str, preserve_trust: bool) -> PyResult<usize> {
        let snapshot: serde_json::Value = serde_json::from_str(snapshot_str).map_err(db_err)?;
        let policy = if preserve_trust {
            crate::ImportTrust::PreserveBounded
        } else {
            crate::ImportTrust::Reset
        };
        self.with_inner("import_namespace", |m| {
            m.import_namespace_with(&snapshot, policy).map_err(db_err)
        })
    }

    fn resolve_contradictions(&self, memory_id: i64) -> PyResult<bool> {
        self.with_inner("resolve_contradictions", |m| {
            m.store
                .resolve_contradictions_for_id(memory_id)
                .map_err(db_err)?;
            Ok(true)
        })
    }

    #[pyo3(signature = (memory_id, agent_output, task_succeeded, session_id = None))]
    fn reinforce_if_used(
        &self,
        memory_id: i64,
        agent_output: &str,
        task_succeeded: bool,
        session_id: Option<&str>,
    ) -> PyResult<bool> {
        self.with_inner("reinforce_if_used", |m| {
            m.reinforce_if_used_with_session(memory_id, agent_output, task_succeeded, session_id)
                .map_err(db_err)
        })
    }

    fn penalize_if_used(&self, memory_ids: Vec<i64>, failure_severity: f32) -> PyResult<String> {
        self.with_inner("penalize_if_used", |m| {
            let outcomes = m
                .penalize_if_used(&memory_ids, failure_severity)
                .map_err(db_err)?;
            serde_json::to_string(&outcomes).map_err(db_err)
        })
    }

    /// Spawn the background maintenance task.
    fn start_background_maintenance(&self, interval_secs: u64, threshold: u32) -> PyResult<()> {
        self.with_inner("start_background_maintenance", |m| {
            m.start_background_maintenance(interval_secs, threshold);
            Ok(())
        })
    }
}

#[pymodule]
fn memoire_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoire>()?;
    m.add_class::<PyMemory>()?;
    m.add_class::<PyRecallAttribution>()?;
    m.add("MemoireError", m.py().get_type_bound::<MemoireError>())?;
    Ok(())
}
