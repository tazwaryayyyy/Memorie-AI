//! Memoire HTTP server — replaces dashboard CLI subprocess calls with a
//! long-lived local service, eliminating per-request process spawn overhead.
//!
//! Default port: 6779 (override with MEMOIRE_SERVER_PORT env var).
//!
//! Endpoints:
//!   GET  /health
//!   POST /remember   { "db": "...", "ns": "...", "text": "..." }
//!   POST /recall     { "db": "...", "ns": "...", "query": "...", "k": 5 }
//!   POST /reinforce  { "db": "...", "ns": "...", "id": 1, "agent_output": "...", "task_succeeded": true }
//!   POST /penalize   { "db": "...", "ns": "...", "ids": [1,2], "failure_severity": 1.0 }
//!   POST /forget     { "db": "...", "ns": "...", "id": 1 }
//!   POST /clear      { "db": "...", "ns": "..." }
//!   GET  /info?db=...
//!   GET  /export?db=...&ns=...

use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;

use memoire::Memoire;

// ─── Shared state ─────────────────────────────────────────────────────────────

type InstanceCache = Arc<Mutex<HashMap<(String, String), Arc<Memoire>>>>;

async fn get_or_create(cache: &InstanceCache, db: &str, ns: &str) -> Result<Arc<Memoire>, String> {
    let key = (db.to_string(), ns.to_string());
    let mut map = cache.lock().await;
    if let Some(m) = map.get(&key) {
        return Ok(Arc::clone(m));
    }
    let m = Memoire::new_ns(db, ns).map_err(|e| e.to_string())?;
    let arc = Arc::new(m);
    map.insert(key, Arc::clone(&arc));
    Ok(arc)
}

// ─── Request / response types ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct RememberReq {
    db: String,
    #[serde(default = "default_ns")]
    ns: String,
    text: String,
}

#[derive(Deserialize)]
struct RecallReq {
    db: String,
    #[serde(default = "default_ns")]
    ns: String,
    query: String,
    #[serde(default = "default_k")]
    k: usize,
}

#[derive(Deserialize)]
struct ReinforceReq {
    db: String,
    #[serde(default = "default_ns")]
    ns: String,
    id: i64,
    agent_output: String,
    task_succeeded: bool,
}

#[derive(Deserialize)]
struct PenalizeReq {
    db: String,
    #[serde(default = "default_ns")]
    ns: String,
    ids: Vec<i64>,
    #[serde(default = "default_severity")]
    failure_severity: f32,
}

#[derive(Deserialize)]
struct ForgetReq {
    db: String,
    #[serde(default = "default_ns")]
    ns: String,
    id: i64,
}

#[derive(Deserialize)]
struct ClearReq {
    db: String,
    #[serde(default = "default_ns")]
    ns: String,
}

#[derive(Deserialize)]
struct ImportReq {
    db: String,
    #[serde(default = "default_ns")]
    ns: String,
    snapshot: serde_json::Value,
}

#[derive(Deserialize)]
struct DbQuery {
    db: String,
    #[serde(default = "default_ns")]
    ns: String,
}

fn default_ns() -> String {
    "default".to_string()
}
fn default_k() -> usize {
    5
}
fn default_severity() -> f32 {
    1.0
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(json!({ "ok": true }))
}

async fn remember(
    State(cache): State<InstanceCache>,
    Json(req): Json<RememberReq>,
) -> impl IntoResponse {
    match get_or_create(&cache, &req.db, &req.ns).await {
        Ok(m) => match m.remember(&req.text) {
            Ok(ids) => (StatusCode::OK, Json(json!({ "ids": ids }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn recall(
    State(cache): State<InstanceCache>,
    Json(req): Json<RecallReq>,
) -> impl IntoResponse {
    match get_or_create(&cache, &req.db, &req.ns).await {
        Ok(m) => match m.recall(&req.query, req.k) {
            Ok(memories) => (StatusCode::OK, Json(json!({ "memories": memories }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn reinforce(
    State(cache): State<InstanceCache>,
    Json(req): Json<ReinforceReq>,
) -> impl IntoResponse {
    match get_or_create(&cache, &req.db, &req.ns).await {
        Ok(m) => match m.reinforce_if_used(req.id, &req.agent_output, req.task_succeeded) {
            Ok(reinforced) => (StatusCode::OK, Json(json!({ "ok": reinforced }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn penalize(
    State(cache): State<InstanceCache>,
    Json(req): Json<PenalizeReq>,
) -> impl IntoResponse {
    match get_or_create(&cache, &req.db, &req.ns).await {
        Ok(m) => match m.penalize_if_used(&req.ids, req.failure_severity) {
            Ok(outcomes) => (
                StatusCode::OK,
                Json(json!({ "ok": true, "outcomes": outcomes })),
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn forget_handler(
    State(cache): State<InstanceCache>,
    Json(req): Json<ForgetReq>,
) -> impl IntoResponse {
    match get_or_create(&cache, &req.db, &req.ns).await {
        Ok(m) => match m.forget(req.id) {
            Ok(deleted) => (StatusCode::OK, Json(json!({ "ok": deleted }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn clear_handler(
    State(cache): State<InstanceCache>,
    Json(req): Json<ClearReq>,
) -> impl IntoResponse {
    match get_or_create(&cache, &req.db, &req.ns).await {
        Ok(m) => match m.clear() {
            Ok(()) => (StatusCode::OK, Json(json!({ "ok": true }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn info_handler(
    State(cache): State<InstanceCache>,
    Query(params): Query<DbQuery>,
) -> impl IntoResponse {
    match get_or_create(&cache, &params.db, &params.ns).await {
        Ok(m) => {
            let count = m.count().unwrap_or(0);
            let size = std::fs::metadata(&params.db)
                .map(|meta| meta.len())
                .unwrap_or(0);
            (
                StatusCode::OK,
                Json(json!({
                    "db": params.db,
                    "namespace": params.ns,
                    "count": count,
                    "size_bytes": size,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn export_handler(
    State(cache): State<InstanceCache>,
    Query(params): Query<DbQuery>,
) -> impl IntoResponse {
    match get_or_create(&cache, &params.db, &params.ns).await {
        Ok(m) => match m.export_namespace() {
            Ok(snapshot) => (StatusCode::OK, Json(snapshot)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

async fn import_handler(
    State(cache): State<InstanceCache>,
    Json(req): Json<ImportReq>,
) -> impl IntoResponse {
    match get_or_create(&cache, &req.db, &req.ns).await {
        Ok(m) => match m.import_namespace(&req.snapshot) {
            Ok(count) => (StatusCode::OK, Json(json!({ "ok": true, "imported": count }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    env_logger::init();

    let port: u16 = env::var("MEMOIRE_SERVER_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6779);

    let cache: InstanceCache = Arc::new(Mutex::new(HashMap::new()));

    let app = Router::new()
        .route("/health", get(health))
        .route("/remember", post(remember))
        .route("/recall", post(recall))
        .route("/reinforce", post(reinforce))
        .route("/penalize", post(penalize))
        .route("/forget", post(forget_handler))
        .route("/clear", post(clear_handler))
        .route("/info", get(info_handler))
        .route("/export", get(export_handler))
        .route("/import", post(import_handler))
        .with_state(cache);

    let addr = format!("127.0.0.1:{port}");
    println!("memoire-server listening on http://{addr}");
    println!("Set MEMOIRE_SERVER_PORT to override port 6779.");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
