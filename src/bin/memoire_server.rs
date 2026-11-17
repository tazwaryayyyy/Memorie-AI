//! Memoire HTTP server: a long-lived LOCAL service used by the (experimental)
//! dashboard. It is a development convenience, not a network service.
//!
//! Default port: 6779 (override with `MEMOIRE_SERVER_PORT`). The server binds
//! `127.0.0.1` only; there is no option to bind another interface.
//!
//! Endpoints:
//!   GET  /health                (always public)
//!   POST /remember   { "db": "...", "ns": "...", "text": "..." }
//!   POST /recall     { "db": "...", "ns": "...", "query": "...", "k": 5 }   (1 <= k <= 100)
//!   POST /reinforce  { "db": "...", "ns": "...", "id": 1, "agent_output": "...", "task_succeeded": true }
//!   POST /penalize   { "db": "...", "ns": "...", "ids": [1,2], "failure_severity": 1.0 }
//!   POST /forget     { "db": "...", "ns": "...", "id": 1 }
//!   POST /clear      { "db": "...", "ns": "..." }          (admin: MEMOIRE_ENABLE_ADMIN_TOOLS)
//!   POST /import     { "db": "...", "ns": "...", "snapshot": {...}, "preserve_trust": false }
//!                    (`preserve_trust: true` is admin-only; at most 1000 memories)
//!   GET  /info?db=...&ns=...
//!   GET  /export?db=...&ns=...
//!
//! Environment:
//!   MEMOIRE_API_TOKEN           Optional bearer token. If SET BUT BLANK the server
//!                               refuses to start (fail closed). Unset = no auth.
//!   MEMOIRE_ALLOWED_PATHS       Comma-separated directories the `db` field must resolve
//!                               into. Default: the server's working directory.
//!   MEMOIRE_ENABLE_ADMIN_TOOLS  `1|true|yes|on` enables `/clear` and `preserve_trust`.
//!   MEMOIRE_SERVER_PORT         Port (an invalid value is an error, not ignored).
//!
//! Every request must carry a loopback `Host` header (DNS-rebinding defence) and,
//! if an `Origin` header is present, a loopback origin. No CORS headers are sent.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;

use memoire::Memoire;

// ─── Limits ───────────────────────────────────────────────────────────────────

/// Pins axum's default explicitly so a dependency bump cannot silently change it.
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
const MAX_TOP_K: usize = 100;
const MAX_PENALIZE_IDS: usize = 100;
const MAX_IMPORT_MEMORIES: usize = 1000;
const MAX_NS_BYTES: usize = 128;
/// Each open (db, namespace) pair holds its own embedder; bound how many exist.
const DEFAULT_MAX_INSTANCES: usize = 16;

// ─── Configuration ────────────────────────────────────────────────────────────

struct Config {
    /// BLAKE3 digest of the expected token. `blake3::Hash` equality is
    /// constant-time, and hashing first hides the token length.
    token_hash: Option<blake3::Hash>,
    /// Canonical directories the `db` field must resolve into.
    allowed_roots: Vec<PathBuf>,
    admin_enabled: bool,
    max_instances: usize,
}

impl Config {
    fn from_vars(
        token: Option<String>,
        allowed_paths: Option<String>,
        admin: Option<String>,
    ) -> Result<Config, String> {
        let token_hash = match token {
            None => None,
            Some(t) if t.trim().is_empty() => {
                return Err("MEMOIRE_API_TOKEN is set but blank; unset it to run \
                            without auth, or give it a real value"
                    .into())
            }
            Some(t) => Some(blake3::hash(t.trim().as_bytes())),
        };

        let mut allowed_roots = Vec::new();
        for raw in allowed_paths.as_deref().unwrap_or("").split(',') {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            let root = fs::canonicalize(raw)
                .map_err(|e| format!("MEMOIRE_ALLOWED_PATHS entry {raw:?}: {e}"))?;
            if !root.is_dir() {
                return Err(format!(
                    "MEMOIRE_ALLOWED_PATHS entry {raw:?} is not a directory"
                ));
            }
            allowed_roots.push(root);
        }
        if allowed_roots.is_empty() {
            let cwd = env::current_dir().map_err(|e| format!("cannot read cwd: {e}"))?;
            allowed_roots
                .push(fs::canonicalize(cwd).map_err(|e| format!("cannot resolve cwd: {e}"))?);
        }

        let admin_enabled = admin
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);

        Ok(Config {
            token_hash,
            allowed_roots,
            admin_enabled,
            max_instances: DEFAULT_MAX_INSTANCES,
        })
    }

    fn from_env() -> Result<Config, String> {
        Config::from_vars(
            env::var("MEMOIRE_API_TOKEN").ok(),
            env::var("MEMOIRE_ALLOWED_PATHS").ok(),
            env::var("MEMOIRE_ENABLE_ADMIN_TOOLS").ok(),
        )
    }
}

// ─── Path sandbox ─────────────────────────────────────────────────────────────

#[cfg(windows)]
fn strip_verbatim(p: PathBuf) -> PathBuf {
    // `fs::canonicalize` yields `\\?\C:\...` on Windows; hand SQLite a plain path.
    let s = p.to_string_lossy().into_owned();
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        if !rest.starts_with(r"UNC\") {
            return PathBuf::from(rest);
        }
    }
    p
}

#[cfg(not(windows))]
fn strip_verbatim(p: PathBuf) -> PathBuf {
    p
}

/// Resolve the client-supplied `db` to an absolute, symlink-free path inside one
/// of `roots`, or explain why not.
///
/// The returned path is always absolute, so SQLite can never interpret it as a
/// `file:` URI. The file itself may not exist yet (created on first write) but
/// its parent directory must. Containment is component-wise (`/a/dbs-evil` is
/// NOT inside `/a/dbs`), and a symlink at the leaf is followed before the check.
fn resolve_db_path(raw: &str, roots: &[PathBuf]) -> Result<PathBuf, String> {
    if raw.is_empty() || raw.contains('\0') {
        return Err("db must be a non-empty path".into());
    }
    let abs = std::path::absolute(raw).map_err(|e| format!("invalid db path: {e}"))?;
    let file_name = abs
        .file_name()
        .ok_or_else(|| "db path must name a file".to_string())?
        .to_os_string();
    let parent = abs.parent().unwrap_or_else(|| Path::new("/"));
    let parent_real =
        fs::canonicalize(parent).map_err(|_| "db parent directory does not exist".to_string())?;
    let mut full = parent_real.join(file_name);
    match fs::symlink_metadata(&full) {
        // Something is there (file or symlink): resolve it. A dangling symlink
        // fails here, which also stops SQLite creating its target elsewhere.
        Ok(_) => {
            full = fs::canonicalize(&full).map_err(|_| "db path cannot be resolved".to_string())?
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err("db path cannot be inspected".into()),
    }
    if full.is_dir() {
        return Err("db path is a directory".into());
    }
    if !roots.iter().any(|r| full.starts_with(r)) {
        return Err("db path is outside the allowed directories \
                    (see MEMOIRE_ALLOWED_PATHS)"
            .into());
    }
    Ok(strip_verbatim(full))
}

// ─── Request guard (Host / Origin / token) ────────────────────────────────────

/// `localhost`, or any loopback IP literal, with an optional port.
fn host_is_loopback(hostport: &str) -> bool {
    let host = if let Some(rest) = hostport.strip_prefix('[') {
        match rest.split_once(']') {
            Some((h, _)) => h,
            None => return false,
        }
    } else {
        match hostport.rsplit_once(':') {
            // "::1" without brackets is not a valid Host value.
            Some((h, _)) if !h.contains(':') => h,
            Some(_) => return false,
            None => hostport,
        }
    };
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

fn origin_is_loopback(origin: &str) -> bool {
    match origin.split_once("://") {
        Some((scheme, rest)) if scheme == "http" || scheme == "https" => {
            host_is_loopback(rest.split('/').next().unwrap_or(""))
        }
        _ => false, // includes the literal "null"
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let v = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, rest) = v.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| rest.trim())
}

fn api_err(status: StatusCode, msg: impl ToString) -> Response {
    (status, Json(json!({ "error": msg.to_string() }))).into_response()
}

/// Applied to every route, in this order:
/// 1. `Host` must be loopback (blocks DNS rebinding, where a hostile site's
///    name resolves to 127.0.0.1 and the browser would otherwise treat us as
///    same-origin). A missing `Host` is rejected.
/// 2. `Origin`, when present, must be loopback.
/// 3. Everything except `/health` needs `Authorization: Bearer <token>` when a
///    token is configured. The `Bearer` scheme is mandatory.
async fn guard(State(app): State<AppState>, req: axum::extract::Request, next: Next) -> Response {
    let headers = req.headers();

    let host_ok = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .map(host_is_loopback)
        .unwrap_or(false);
    if !host_ok {
        log::warn!("rejected request with non-loopback or missing Host header");
        return api_err(StatusCode::FORBIDDEN, "forbidden: Host must be localhost");
    }

    if let Some(origin) = headers.get(header::ORIGIN) {
        if !origin.to_str().map(origin_is_loopback).unwrap_or(false) {
            log::warn!("rejected request with non-loopback Origin header");
            return api_err(StatusCode::FORBIDDEN, "forbidden: cross-origin request");
        }
    }

    if req.uri().path() != "/health" {
        if let Some(expected) = &app.cfg.token_hash {
            let ok = bearer_token(headers)
                .map(|t| blake3::hash(t.as_bytes()) == *expected)
                .unwrap_or(false);
            if !ok {
                log::warn!("rejected request with invalid or missing bearer token");
                let mut resp = api_err(
                    StatusCode::UNAUTHORIZED,
                    "Unauthorized: invalid or missing token",
                );
                resp.headers_mut().insert(
                    header::WWW_AUTHENTICATE,
                    header::HeaderValue::from_static("Bearer"),
                );
                return resp;
            }
        }
    }
    next.run(req).await
}

// ─── Shared state ─────────────────────────────────────────────────────────────

type ApiResult<T> = Result<T, (StatusCode, String)>;
type Opener = fn(&str, &str) -> Result<Memoire, String>;
/// Open instances keyed by (canonical db path, namespace).
type InstanceCache = Arc<Mutex<HashMap<(PathBuf, String), Arc<Memoire>>>>;

#[derive(Clone)]
struct AppState {
    cache: InstanceCache,
    cfg: Arc<Config>,
    open: Opener,
}

fn open_memoire(db: &str, ns: &str) -> Result<Memoire, String> {
    Memoire::new_ns(db, ns).map_err(|e| e.to_string())
}

impl AppState {
    fn new(cfg: Config, open: Opener) -> Self {
        AppState {
            cache: Arc::new(Mutex::new(HashMap::new())),
            cfg: Arc::new(cfg),
            open,
        }
    }
}

fn validate_ns(ns: &str) -> ApiResult<()> {
    if ns.is_empty() || ns.len() > MAX_NS_BYTES || ns.chars().any(|c| c.is_control()) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("ns must be 1..={MAX_NS_BYTES} bytes with no control characters"),
        ));
    }
    Ok(())
}

fn require_admin(app: &AppState, what: &str) -> ApiResult<()> {
    if app.cfg.admin_enabled {
        Ok(())
    } else {
        Err((
            StatusCode::FORBIDDEN,
            format!("{what} is an admin operation; start memoire-server with MEMOIRE_ENABLE_ADMIN_TOOLS=1 to allow it"),
        ))
    }
}

/// Validate, sandbox-resolve, then open (or reuse) the Memoire for `(db, ns)`.
async fn get_or_create(app: &AppState, db: &str, ns: &str) -> ApiResult<(Arc<Memoire>, PathBuf)> {
    validate_ns(ns)?;
    let path =
        resolve_db_path(db, &app.cfg.allowed_roots).map_err(|e| (StatusCode::FORBIDDEN, e))?;
    let key = (path.clone(), ns.to_string());
    let mut map = app.cache.lock().await;
    if let Some(m) = map.get(&key) {
        return Ok((Arc::clone(m), path));
    }
    if map.len() >= app.cfg.max_instances {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "too many open (db, namespace) pairs (limit {}); restart the server to reset",
                app.cfg.max_instances
            ),
        ));
    }
    let path_str = path.to_str().ok_or((
        StatusCode::BAD_REQUEST,
        "db path is not valid UTF-8".to_string(),
    ))?;
    let m = (app.open)(path_str, ns).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let arc = Arc::new(m);
    map.insert(key, Arc::clone(&arc));
    Ok((arc, path))
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
    /// Restore the snapshot's trust metadata (clamped). Default: content only.
    #[serde(default)]
    preserve_trust: bool,
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

async fn remember(State(app): State<AppState>, Json(req): Json<RememberReq>) -> impl IntoResponse {
    match get_or_create(&app, &req.db, &req.ns).await {
        Ok((m, _)) => match m.remember(&req.text) {
            Ok(ids) => (StatusCode::OK, Json(json!({ "ids": ids }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err((status, msg)) => api_err(status, msg),
    }
}

async fn recall(State(app): State<AppState>, Json(req): Json<RecallReq>) -> impl IntoResponse {
    if req.k == 0 || req.k > MAX_TOP_K {
        return api_err(
            StatusCode::BAD_REQUEST,
            format!("k must be between 1 and {MAX_TOP_K}"),
        );
    }
    match get_or_create(&app, &req.db, &req.ns).await {
        Ok((m, _)) => match m.recall(&req.query, req.k) {
            Ok(memories) => (StatusCode::OK, Json(json!({ "memories": memories }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err((status, msg)) => api_err(status, msg),
    }
}

async fn reinforce(
    State(app): State<AppState>,
    Json(req): Json<ReinforceReq>,
) -> impl IntoResponse {
    match get_or_create(&app, &req.db, &req.ns).await {
        Ok((m, _)) => match m.reinforce_if_used(req.id, &req.agent_output, req.task_succeeded) {
            Ok(reinforced) => (StatusCode::OK, Json(json!({ "ok": reinforced }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err((status, msg)) => api_err(status, msg),
    }
}

async fn penalize(State(app): State<AppState>, Json(req): Json<PenalizeReq>) -> impl IntoResponse {
    if req.ids.len() > MAX_PENALIZE_IDS {
        return api_err(
            StatusCode::BAD_REQUEST,
            format!("ids may contain at most {MAX_PENALIZE_IDS} entries"),
        );
    }
    if !req.failure_severity.is_finite() || !(0.0..=1.0).contains(&req.failure_severity) {
        return api_err(
            StatusCode::BAD_REQUEST,
            "failure_severity must be a number in [0.0, 1.0]",
        );
    }
    match get_or_create(&app, &req.db, &req.ns).await {
        Ok((m, _)) => match m.penalize_if_used(&req.ids, req.failure_severity) {
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
        Err((status, msg)) => api_err(status, msg),
    }
}

async fn forget_handler(
    State(app): State<AppState>,
    Json(req): Json<ForgetReq>,
) -> impl IntoResponse {
    match get_or_create(&app, &req.db, &req.ns).await {
        Ok((m, _)) => match m.forget(req.id) {
            Ok(deleted) => (StatusCode::OK, Json(json!({ "ok": deleted }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err((status, msg)) => api_err(status, msg),
    }
}

async fn clear_handler(
    State(app): State<AppState>,
    Json(req): Json<ClearReq>,
) -> impl IntoResponse {
    if let Err((status, msg)) = require_admin(&app, "/clear") {
        return api_err(status, msg);
    }
    match get_or_create(&app, &req.db, &req.ns).await {
        Ok((m, _)) => match m.clear() {
            Ok(()) => (StatusCode::OK, Json(json!({ "ok": true }))).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err((status, msg)) => api_err(status, msg),
    }
}

async fn info_handler(
    State(app): State<AppState>,
    Query(params): Query<DbQuery>,
) -> impl IntoResponse {
    match get_or_create(&app, &params.db, &params.ns).await {
        Ok((m, path)) => {
            let count = m.count().unwrap_or(0);
            let size = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            (
                StatusCode::OK,
                Json(json!({
                    "db": path.to_string_lossy(),
                    "namespace": params.ns,
                    "count": count,
                    "size_bytes": size,
                })),
            )
                .into_response()
        }
        Err((status, msg)) => api_err(status, msg),
    }
}

async fn export_handler(
    State(app): State<AppState>,
    Query(params): Query<DbQuery>,
) -> impl IntoResponse {
    match get_or_create(&app, &params.db, &params.ns).await {
        Ok((m, _)) => match m.export_namespace() {
            Ok(snapshot) => (StatusCode::OK, Json(snapshot)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err((status, msg)) => api_err(status, msg),
    }
}

async fn import_handler(
    State(app): State<AppState>,
    Json(req): Json<ImportReq>,
) -> impl IntoResponse {
    if req.preserve_trust {
        if let Err((status, msg)) = require_admin(&app, "/import with preserve_trust") {
            return api_err(status, msg);
        }
    }
    let n = req
        .snapshot
        .get("memories")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    if n > MAX_IMPORT_MEMORIES {
        return api_err(
            StatusCode::BAD_REQUEST,
            format!("snapshot has {n} memories; the limit is {MAX_IMPORT_MEMORIES} per import"),
        );
    }
    match get_or_create(&app, &req.db, &req.ns).await {
        Ok((m, _)) => match m.import_namespace_with(
            &req.snapshot,
            if req.preserve_trust {
                memoire::ImportTrust::PreserveBounded
            } else {
                memoire::ImportTrust::Reset
            },
        ) {
            Ok(count) => (
                StatusCode::OK,
                Json(json!({ "ok": true, "imported": count })),
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response(),
        },
        Err((status, msg)) => api_err(status, msg),
    }
}

// ─── Router + entry point ─────────────────────────────────────────────────────

fn build_app(state: AppState) -> Router {
    Router::new()
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
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(axum::middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

fn fatal(msg: impl std::fmt::Display) -> ! {
    eprintln!("memoire-server: {msg}");
    std::process::exit(2);
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let port: u16 = match env::var("MEMOIRE_SERVER_PORT") {
        Ok(v) => v
            .trim()
            .parse()
            .unwrap_or_else(|_| fatal(format!("MEMOIRE_SERVER_PORT={v:?} is not a valid port"))),
        Err(_) => 6779,
    };
    let cfg = Config::from_env().unwrap_or_else(|e| fatal(e));

    println!("memoire-server listening on http://127.0.0.1:{port} (loopback only)");
    if cfg.token_hash.is_some() {
        println!("Token auth enabled (MEMOIRE_API_TOKEN is set).");
    } else {
        println!("Warning: MEMOIRE_API_TOKEN not set; any local process can use this server.");
    }
    for root in &cfg.allowed_roots {
        println!("Allowed db directory: {}", root.display());
    }
    if cfg.admin_enabled {
        println!("Admin operations ENABLED (/clear, import with preserve_trust).");
    }

    let app = build_app(AppState::new(cfg, open_memoire));
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap_or_else(|e| fatal(format!("cannot bind 127.0.0.1:{port}: {e}")));
    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| fatal(format!("server error: {e}")));
}

#[cfg(test)]
mod tests {
    use super::*;
    use memoire::embedder::EmbedProvider;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    /// Deterministic bag-of-words embedder: no model download needed.
    struct TestEmbedder;

    impl EmbedProvider for TestEmbedder {
        fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let mut v = vec![0.0_f32; 384];
                    for token in text.split(|c: char| !c.is_ascii_alphanumeric()) {
                        if token.is_empty() {
                            continue;
                        }
                        let h = token
                            .to_ascii_lowercase()
                            .bytes()
                            .fold(0xcbf29ce484222325_u64, |h, b| {
                                (h ^ b as u64).wrapping_mul(0x100000001b3)
                            });
                        v[h as usize % 384] += 1.0;
                    }
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        v.iter_mut().for_each(|x| *x /= norm);
                    }
                    v
                })
                .collect())
        }

        fn dim(&self) -> usize {
            384
        }
    }

    fn open_test(db: &str, ns: &str) -> Result<Memoire, String> {
        Memoire::new_ns_with_embedder(db, ns, Box::new(TestEmbedder)).map_err(|e| e.to_string())
    }

    struct Server {
        addr: std::net::SocketAddr,
        root: tempfile::TempDir,
    }

    impl Server {
        fn db(&self, name: &str) -> String {
            self.root.path().join(name).to_string_lossy().into_owned()
        }
        fn host(&self) -> String {
            format!("127.0.0.1:{}", self.addr.port())
        }
    }

    async fn start(token: Option<&str>, admin: bool, max_instances: usize) -> Server {
        let root = tempfile::tempdir().unwrap();
        let mut cfg = Config::from_vars(
            token.map(String::from),
            Some(root.path().to_string_lossy().into_owned()),
            admin.then(|| "1".to_string()),
        )
        .unwrap();
        cfg.max_instances = max_instances;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = build_app(AppState::new(cfg, open_test));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Server { addr, root }
    }

    struct Resp {
        status: u16,
        headers: String,
        body: String,
    }

    /// Minimal HTTP/1.1 client; `headers` must include `Host` if wanted.
    async fn raw(
        addr: std::net::SocketAddr,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: Option<&str>,
    ) -> Resp {
        let mut s = TcpStream::connect(addr).await.unwrap();
        let mut req = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\n");
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        if let Some(b) = body {
            req.push_str(&format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                b.len()
            ));
        }
        req.push_str("\r\n");
        // The server may answer (and close) before reading a huge body.
        let _ = s.write_all(req.as_bytes()).await;
        if let Some(b) = body {
            let _ = s.write_all(b.as_bytes()).await;
        }
        let mut buf = Vec::new();
        let _ = s.read_to_end(&mut buf).await;
        let text = String::from_utf8_lossy(&buf).into_owned();
        let status = text
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);
        let (headers, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
        Resp {
            status,
            headers: headers.to_string(),
            body: body.to_string(),
        }
    }

    async fn call(
        srv: &Server,
        method: &str,
        path: &str,
        token: Option<&str>,
        body: Option<serde_json::Value>,
    ) -> Resp {
        let host = srv.host();
        let auth = token.map(|t| format!("Bearer {t}"));
        let mut h = vec![("Host", host.as_str())];
        if let Some(a) = &auth {
            h.push(("Authorization", a.as_str()));
        }
        let body = body.map(|b| b.to_string());
        raw(srv.addr, method, path, &h, body.as_deref()).await
    }

    fn enc(s: &str) -> String {
        s.bytes()
            .map(|b| match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                    (b as char).to_string()
                }
                _ => format!("%{b:02X}"),
            })
            .collect()
    }

    fn info_path(db: &str, ns: &str) -> String {
        format!("/info?db={}&ns={}", enc(db), enc(ns))
    }

    // ── pure unit tests ──────────────────────────────────────────────────

    #[test]
    fn host_and_origin_parsing() {
        for ok in [
            "localhost",
            "localhost:6779",
            "LOCALHOST:1",
            "127.0.0.1",
            "127.0.0.1:6779",
            "127.0.0.2:80",
            "[::1]",
            "[::1]:6779",
        ] {
            assert!(host_is_loopback(ok), "{ok} should be allowed");
        }
        for bad in [
            "",
            "evil.example",
            "evil.example:6779",
            "localhost.evil.example",
            "127.0.0.1.evil.example",
            "0.0.0.0:6779",
            "192.168.1.5:6779",
            "::1",
            "[::1",
            "[::2]:80",
        ] {
            assert!(!host_is_loopback(bad), "{bad:?} should be rejected");
        }
        assert!(origin_is_loopback("http://localhost:3000"));
        assert!(origin_is_loopback("https://127.0.0.1"));
        assert!(!origin_is_loopback("null"));
        assert!(!origin_is_loopback("https://evil.example"));
        assert!(!origin_is_loopback("http://localhost.evil.example"));
        assert!(!origin_is_loopback("file://localhost"));
    }

    #[test]
    fn config_fails_closed() {
        assert!(Config::from_vars(Some("".into()), None, None).is_err());
        assert!(Config::from_vars(Some("   ".into()), None, None).is_err());
        assert!(Config::from_vars(None, None, None)
            .unwrap()
            .token_hash
            .is_none());
        assert!(Config::from_vars(Some(" tok ".into()), None, None)
            .unwrap()
            .token_hash
            .is_some());
        assert!(Config::from_vars(None, Some("/definitely/not/here".into()), None).is_err());

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        fs::write(&file, b"x").unwrap();
        assert!(Config::from_vars(None, Some(file.to_string_lossy().into()), None).is_err());

        for (v, want) in [
            ("1", true),
            ("TRUE", true),
            (" yes ", true),
            ("on", true),
            ("0", false),
            ("", false),
            ("no", false),
        ] {
            let c = Config::from_vars(None, None, Some(v.into())).unwrap();
            assert_eq!(c.admin_enabled, want, "admin flag {v:?}");
        }
        // Unset defaults to the working directory only.
        let c = Config::from_vars(None, None, None).unwrap();
        assert_eq!(c.allowed_roots.len(), 1);
    }

    #[test]
    fn resolve_is_component_wise_and_canonical() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::create_dir(root.join("sub")).unwrap();
        let roots = vec![root.clone()];

        let a = resolve_db_path(root.join("a.db").to_str().unwrap(), &roots).unwrap();
        let b = resolve_db_path(
            root.join("sub").join("..").join("a.db").to_str().unwrap(),
            &roots,
        )
        .unwrap();
        assert_eq!(a, b, "different spellings must map to one cache key");

        assert!(resolve_db_path("", &roots).is_err());
        assert!(resolve_db_path("a\0b", &roots).is_err());
        assert!(
            resolve_db_path(root.to_str().unwrap(), &roots).is_err(),
            "directory"
        );

        // `/root-evil` must not match root `/root`.
        let evil = PathBuf::from(format!("{}-evil", root.display()));
        fs::create_dir(&evil).unwrap();
        let r = resolve_db_path(evil.join("x.db").to_str().unwrap(), &roots);
        fs::remove_dir(&evil).unwrap();
        assert!(r.is_err(), "string-prefix match must not grant access");
    }

    // ── HTTP-level tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn host_header_must_be_loopback() {
        let srv = start(None, false, 16).await;
        let port = srv.addr.port();
        for host in [
            format!("localhost:{port}"),
            format!("127.0.0.1:{port}"),
            format!("[::1]:{port}"),
        ] {
            let r = raw(srv.addr, "GET", "/health", &[("Host", &host)], None).await;
            assert_eq!(r.status, 200, "Host {host}");
        }
        for host in ["evil.example", "evil.example:80", "127.0.0.1.evil.example"] {
            let r = raw(srv.addr, "GET", "/health", &[("Host", host)], None).await;
            assert_eq!(r.status, 403, "Host {host}");
        }
        // No Host header at all: never served (hyper may answer 400 before us).
        let r = raw(srv.addr, "GET", "/health", &[], None).await;
        assert_ne!(r.status, 200);
        // The Host check runs before auth: a valid token cannot bypass it.
        let srv2 = start(Some("tok"), false, 16).await;
        let r = raw(
            srv2.addr,
            "GET",
            "/health",
            &[("Host", "evil.example"), ("Authorization", "Bearer tok")],
            None,
        )
        .await;
        assert_eq!(r.status, 403);
    }

    #[tokio::test]
    async fn origin_must_be_loopback_when_present() {
        let srv = start(None, false, 16).await;
        let host = srv.host();
        for (origin, want) in [
            ("https://evil.example", 403),
            ("null", 403),
            ("http://localhost.evil.example", 403),
            ("http://localhost:3000", 200),
        ] {
            let r = raw(
                srv.addr,
                "GET",
                "/health",
                &[("Host", &host), ("Origin", origin)],
                None,
            )
            .await;
            assert_eq!(r.status, want, "Origin {origin}");
        }
    }

    #[tokio::test]
    async fn token_is_required_and_scheme_is_strict() {
        let srv = start(Some("s3cret-token"), false, 16).await;
        let host = srv.host();
        let path = info_path(&srv.db("t.db"), "default");

        let r = raw(srv.addr, "GET", &path, &[("Host", &host)], None).await;
        assert_eq!(r.status, 401);
        assert!(
            r.headers
                .to_ascii_lowercase()
                .contains("www-authenticate: bearer"),
            "{}",
            r.headers
        );
        for bad in [
            "s3cret-token",
            "Bearer wrong",
            "Bearer",
            "Basic s3cret-token",
            "Bearer s3cret-token-x",
        ] {
            let r = raw(
                srv.addr,
                "GET",
                &path,
                &[("Host", &host), ("Authorization", bad)],
                None,
            )
            .await;
            assert_eq!(r.status, 401, "Authorization: {bad}");
        }
        for good in [
            "Bearer s3cret-token",
            "bearer s3cret-token",
            "Bearer   s3cret-token  ",
        ] {
            let r = raw(
                srv.addr,
                "GET",
                &path,
                &[("Host", &host), ("Authorization", good)],
                None,
            )
            .await;
            assert_eq!(r.status, 200, "Authorization: {good} -> {}", r.body);
        }
        // /health stays public.
        let r = raw(srv.addr, "GET", "/health", &[("Host", &host)], None).await;
        assert_eq!(r.status, 200);
    }

    #[tokio::test]
    async fn db_must_resolve_inside_allowed_roots() {
        let srv = start(None, false, 16).await;
        let outside = tempfile::tempdir().unwrap();
        let root_str = srv.root.path().to_string_lossy().into_owned();
        let evil = PathBuf::from(format!("{root_str}-evil"));
        fs::create_dir(&evil).unwrap();
        fs::create_dir(srv.root.path().join("nested")).unwrap();

        let cases: Vec<(String, u16)> = vec![
            (srv.db("ok.db"), 200),
            (srv.db("nested/ok.db"), 200),
            (
                outside.path().join("x.db").to_string_lossy().into_owned(),
                403,
            ),
            (format!("{root_str}/../escape.db"), 403),
            (evil.join("x.db").to_string_lossy().into_owned(), 403),
            (srv.db("missing_dir/x.db"), 403),
            ("file:/etc/passwd?mode=ro".into(), 403),
            ("file:memdb1?mode=memory&cache=shared".into(), 403),
            (srv.root.path().to_string_lossy().into_owned(), 403),
        ];
        for (db, want) in cases {
            let r = call(&srv, "GET", &info_path(&db, "default"), None, None).await;
            assert_eq!(r.status, want, "db {db:?}: {}", r.body);
        }
        fs::remove_dir(&evil).unwrap();
        assert!(!srv.root.path().join("../escape.db").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_escape_is_rejected() {
        let srv = start(None, false, 16).await;
        let outside = tempfile::tempdir().unwrap();

        // Symlinked directory pointing outside the root.
        std::os::unix::fs::symlink(outside.path(), srv.root.path().join("link_dir")).unwrap();
        let r = call(
            &srv,
            "GET",
            &info_path(&srv.db("link_dir/x.db"), "default"),
            None,
            None,
        )
        .await;
        assert_eq!(r.status, 403, "{}", r.body);

        // Dangling symlink at the leaf: SQLite would create the target outside.
        let target = outside.path().join("created_by_sqlite.db");
        std::os::unix::fs::symlink(&target, srv.root.path().join("dangling.db")).unwrap();
        let r = call(
            &srv,
            "GET",
            &info_path(&srv.db("dangling.db"), "default"),
            None,
            None,
        )
        .await;
        assert_eq!(r.status, 403, "{}", r.body);
        assert!(!target.exists(), "nothing may be created outside the root");
    }

    #[tokio::test]
    async fn inputs_are_bounded() {
        let srv = start(None, false, 16).await;
        let db = srv.db("b.db");
        let recall = |k: u64| json!({"db": db, "query": "q", "k": k});
        for (k, want) in [(0, 400), (101, 400), (u64::MAX, 400), (1, 200), (100, 200)] {
            let r = call(&srv, "POST", "/recall", None, Some(recall(k))).await;
            assert_eq!(r.status, want, "k={k}: {}", r.body);
        }
        let many: Vec<i64> = (0..101).collect();
        let r = call(
            &srv,
            "POST",
            "/penalize",
            None,
            Some(json!({"db": db, "ids": many})),
        )
        .await;
        assert_eq!(r.status, 400, "{}", r.body);
        for sev in [2.0, -0.1] {
            let r = call(
                &srv,
                "POST",
                "/penalize",
                None,
                Some(json!({"db": db, "ids": [1], "failure_severity": sev})),
            )
            .await;
            assert_eq!(r.status, 400, "severity {sev}: {}", r.body);
        }
        for ns in [String::new(), "x".repeat(129), "bad\nns".to_string()] {
            let r = call(&srv, "GET", &info_path(&db, &ns), None, None).await;
            assert_eq!(r.status, 400, "ns {ns:?}");
        }
        // Body limit (axum answers 413 from the Json extractor).
        let big = "a".repeat(MAX_BODY_BYTES + 1024);
        let r = call(
            &srv,
            "POST",
            "/remember",
            None,
            Some(json!({"db": db, "text": big})),
        )
        .await;
        assert_eq!(r.status, 413, "oversize body: {}", r.body);
    }

    #[tokio::test]
    async fn admin_operations_are_gated() {
        let locked = start(None, false, 16).await;
        let db = locked.db("a.db");
        let r = call(
            &locked,
            "POST",
            "/remember",
            None,
            Some(json!({"db": db, "text": "keep me"})),
        )
        .await;
        assert_eq!(r.status, 200, "{}", r.body);

        let r = call(&locked, "POST", "/clear", None, Some(json!({"db": db}))).await;
        assert_eq!(r.status, 403);
        assert!(r.body.contains("MEMOIRE_ENABLE_ADMIN_TOOLS"), "{}", r.body);
        let r = call(
            &locked,
            "POST",
            "/import",
            None,
            Some(json!({"db": db, "snapshot": {"memories": []}, "preserve_trust": true})),
        )
        .await;
        assert_eq!(r.status, 403, "{}", r.body);
        // Nothing was wiped.
        let r = call(&locked, "GET", &info_path(&db, "default"), None, None).await;
        assert!(r.body.contains("\"count\":1"), "{}", r.body);

        let open = start(None, true, 16).await;
        let db = open.db("a.db");
        call(
            &open,
            "POST",
            "/remember",
            None,
            Some(json!({"db": db, "text": "wipe me"})),
        )
        .await;
        let r = call(&open, "POST", "/clear", None, Some(json!({"db": db}))).await;
        assert_eq!(r.status, 200, "{}", r.body);
        let r = call(&open, "GET", &info_path(&db, "default"), None, None).await;
        assert!(r.body.contains("\"count\":0"), "{}", r.body);
    }

    #[tokio::test]
    async fn import_round_trip_and_cap() {
        let srv = start(None, false, 16).await;
        let db = srv.db("rt.db");
        let r = call(
            &srv,
            "POST",
            "/remember",
            None,
            Some(json!({"db": db, "ns": "src", "text": "always validate external inputs"})),
        )
        .await;
        assert_eq!(r.status, 200, "{}", r.body);

        let r = call(
            &srv,
            "GET",
            &format!("/export?db={}&ns=src", enc(&db)),
            None,
            None,
        )
        .await;
        assert_eq!(r.status, 200, "{}", r.body);
        let snapshot: serde_json::Value = serde_json::from_str(&r.body).unwrap();

        // Content-only import needs no admin flag.
        let r = call(
            &srv,
            "POST",
            "/import",
            None,
            Some(json!({"db": db, "ns": "dst", "snapshot": snapshot})),
        )
        .await;
        assert_eq!(r.status, 200, "{}", r.body);
        let r = call(&srv, "GET", &info_path(&db, "dst"), None, None).await;
        assert!(r.body.contains("\"count\":1"), "{}", r.body);

        let too_many: Vec<_> = (0..=MAX_IMPORT_MEMORIES)
            .map(|_| json!({"content": "x"}))
            .collect();
        let r = call(
            &srv,
            "POST",
            "/import",
            None,
            Some(json!({"db": db, "ns": "dst2", "snapshot": {"memories": too_many}})),
        )
        .await;
        assert_eq!(r.status, 400, "{}", r.body);
    }

    #[tokio::test]
    async fn open_instances_are_capped() {
        let srv = start(None, false, 2).await;
        let db = srv.db("cap.db");
        for ns in ["a", "b"] {
            let r = call(&srv, "GET", &info_path(&db, ns), None, None).await;
            assert_eq!(r.status, 200, "{}", r.body);
        }
        let r = call(&srv, "GET", &info_path(&db, "c"), None, None).await;
        assert_eq!(r.status, 503, "{}", r.body);
        // Already-open pairs keep working, including via another spelling.
        let again = format!("{}/./cap.db", srv.root.path().display());
        let r = call(&srv, "GET", &info_path(&again, "a"), None, None).await;
        assert_eq!(r.status, 200, "{}", r.body);
    }
}
