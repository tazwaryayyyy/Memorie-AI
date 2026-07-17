use crate::{store::Memory, Memoire};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

pyo3::create_exception!(memoire, MemoireError, PyException);

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
        let preview = if self.content.len() > 60 {
            format!("{}...", &self.content[..60])
        } else {
            self.content.clone()
        };
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
    /// threads and concurrent &mut self calls block instead of panicking.
    inner: Arc<Mutex<Memoire>>,
}

/// Convenience: lock the inner Memoire and surface any poisoning as a
/// Python RuntimeError (poisoning is only possible on a panic, which
/// shouldn't happen in normal operation).
macro_rules! lock {
    ($self:expr) => {
        $self
            .inner
            .lock()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
    };
}

#[pymethods]
impl PyMemoire {
    #[new]
    #[pyo3(signature = (db_path = "./memoire.db", namespace = "default"))]
    fn new(db_path: &str, namespace: &str) -> PyResult<Self> {
        let inner = if db_path == ":memory:" {
            Memoire::in_memory_ns(namespace)
        } else {
            Memoire::new_ns(db_path, namespace)
        }
        .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    fn remember(&self, content: &str) -> PyResult<usize> {
        let ids = lock!(self)
            .remember(content)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(ids.len())
    }

    fn recall(&self, query: &str, top_k: usize) -> PyResult<Vec<PyMemory>> {
        let memories = lock!(self)
            .recall(query, top_k)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(memories.into_iter().map(py_memory).collect())
    }

    #[pyo3(signature = (query, top_k, mmr_lambda = 0.5))]
    fn recall_mmr(&self, query: &str, top_k: usize, mmr_lambda: f32) -> PyResult<Vec<PyMemory>> {
        let memories = lock!(self)
            .recall_mmr(query, top_k, mmr_lambda)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(memories.into_iter().map(py_memory).collect())
    }

    fn recall_reranked(&self, query: &str, top_k: usize) -> PyResult<Vec<PyMemory>> {
        let memories = lock!(self)
            .recall_reranked(query, top_k)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(memories.into_iter().map(py_memory).collect())
    }

    fn recall_explained(&self, query: &str, top_k: usize) -> PyResult<Vec<PyRecallAttribution>> {
        let attributions = lock!(self)
            .recall_explained(query, top_k)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
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
    }

    fn recompute_prototypes(&self) -> PyResult<()> {
        lock!(self)
            .recompute_prototypes()
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(())
    }

    fn forget(&self, memory_id: i64) -> PyResult<bool> {
        let deleted = lock!(self)
            .forget(memory_id)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(deleted)
    }

    fn count(&self) -> PyResult<i64> {
        let count = lock!(self)
            .count()
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(count)
    }

    fn clear(&self) -> PyResult<()> {
        lock!(self)
            .clear()
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(())
    }

    fn export_namespace(&self) -> PyResult<String> {
        let value = lock!(self)
            .export_namespace()
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let json = serde_json::to_string(&value)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(json)
    }

    fn import_namespace(&self, snapshot_str: &str) -> PyResult<usize> {
        let snapshot: serde_json::Value = serde_json::from_str(snapshot_str)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let count = lock!(self)
            .import_namespace(&snapshot)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(count)
    }

    fn resolve_contradictions(&self, memory_id: i64) -> PyResult<bool> {
        lock!(self)
            .store
            .resolve_contradictions_for_id(memory_id)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(true)
    }

    #[pyo3(signature = (memory_id, agent_output, task_succeeded, session_id = None))]
    fn reinforce_if_used(
        &self,
        memory_id: i64,
        agent_output: &str,
        task_succeeded: bool,
        session_id: Option<&str>,
    ) -> PyResult<bool> {
        let reinforced = lock!(self)
            .reinforce_if_used_with_session(memory_id, agent_output, task_succeeded, session_id)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(reinforced)
    }

    fn penalize_if_used(&self, memory_ids: Vec<i64>, failure_severity: f32) -> PyResult<String> {
        let outcomes = lock!(self)
            .penalize_if_used(&memory_ids, failure_severity)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let json = serde_json::to_string(&outcomes)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(json)
    }

    /// Spawn the background maintenance task.
    /// Uses `&self` (not `&mut self`) because inner is behind a Mutex,
    /// so thread safety is provided by the lock, not exclusive access.
    fn start_background_maintenance(&self, interval_secs: u64, threshold: u32) -> PyResult<()> {
        lock!(self).start_background_maintenance(interval_secs, threshold);
        Ok(())
    }
}

#[pymodule]
fn memoire(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoire>()?;
    m.add_class::<PyMemory>()?;
    m.add_class::<PyRecallAttribution>()?;
    m.add("MemoireError", m.py().get_type_bound::<MemoireError>())?;
    Ok(())
}
