use crate::Memoire;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

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

#[pyclass(name = "Memoire")]
pub struct PyMemoire {
    inner: Memoire,
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
        Ok(Self { inner })
    }

    fn remember(&self, content: &str) -> PyResult<usize> {
        let ids = self
            .inner
            .remember(content)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(ids.len())
    }

    fn recall(&self, query: &str, top_k: usize) -> PyResult<Vec<PyMemory>> {
        let memories = self
            .inner
            .recall(query, top_k)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let py_memories = memories
            .into_iter()
            .map(|m| PyMemory {
                id: m.id,
                content: m.content,
                score: m.score,
                trust: m.trust,
                uncertainty: m.uncertainty,
                state: m.state,
                created_at: m.created_at,
                last_used_at: m.last_used_at,
            })
            .collect();
        Ok(py_memories)
    }

    #[pyo3(signature = (query, top_k, mmr_lambda = 0.5))]
    fn recall_mmr(&self, query: &str, top_k: usize, mmr_lambda: f32) -> PyResult<Vec<PyMemory>> {
        let memories = self
            .inner
            .recall_mmr(query, top_k, mmr_lambda)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let py_memories = memories
            .into_iter()
            .map(|m| PyMemory {
                id: m.id,
                content: m.content,
                score: m.score,
                trust: m.trust,
                uncertainty: m.uncertainty,
                state: m.state,
                created_at: m.created_at,
                last_used_at: m.last_used_at,
            })
            .collect();
        Ok(py_memories)
    }

    fn recall_reranked(&self, query: &str, top_k: usize) -> PyResult<Vec<PyMemory>> {
        let memories = self
            .inner
            .recall_reranked(query, top_k)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let py_memories = memories
            .into_iter()
            .map(|m| PyMemory {
                id: m.id,
                content: m.content,
                score: m.score,
                trust: m.trust,
                uncertainty: m.uncertainty,
                state: m.state,
                created_at: m.created_at,
                last_used_at: m.last_used_at,
            })
            .collect();
        Ok(py_memories)
    }

    fn forget(&self, memory_id: i64) -> PyResult<bool> {
        let deleted = self
            .inner
            .forget(memory_id)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(deleted)
    }

    fn count(&self) -> PyResult<i64> {
        let count = self
            .inner
            .count()
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(count)
    }

    fn clear(&self) -> PyResult<()> {
        self.inner
            .clear()
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(())
    }

    fn export_namespace(&self) -> PyResult<String> {
        let value = self
            .inner
            .export_namespace()
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let json = serde_json::to_string(&value)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(json)
    }

    fn import_namespace(&self, snapshot_str: &str) -> PyResult<usize> {
        let snapshot: serde_json::Value = serde_json::from_str(snapshot_str)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let count = self
            .inner
            .import_namespace(&snapshot)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(count)
    }

    fn resolve_contradictions(&self, memory_id: i64) -> PyResult<bool> {
        self.inner
            .store
            .resolve_contradictions_for_id(memory_id)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(true)
    }

    fn reinforce_if_used(
        &self,
        memory_id: i64,
        agent_output: &str,
        task_succeeded: bool,
    ) -> PyResult<bool> {
        let reinforced = self
            .inner
            .reinforce_if_used(memory_id, agent_output, task_succeeded)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(reinforced)
    }

    fn penalize_if_used(&self, memory_ids: Vec<i64>, failure_severity: f32) -> PyResult<String> {
        let outcomes = self
            .inner
            .penalize_if_used(&memory_ids, failure_severity)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        let json = serde_json::to_string(&outcomes)
            .map_err(|e| PyErr::new::<MemoireError, _>(e.to_string()))?;
        Ok(json)
    }
}

#[pymodule]
fn memoire(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoire>()?;
    m.add_class::<PyMemory>()?;
    m.add("MemoireError", m.py().get_type_bound::<MemoireError>())?;
    Ok(())
}
