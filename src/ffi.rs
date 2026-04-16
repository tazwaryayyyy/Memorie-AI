//! C-compatible FFI for Memoire.
//!
//! All strings crossing the boundary are null-terminated UTF-8.
//! Recall results are returned as a heap-allocated JSON string.
//! The caller MUST free it with `memoire_free_string`.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_longlong};
use std::ptr;

use crate::Memoire;

/// Opaque handle to a Memoire instance.
pub struct MemoireHandle(Memoire);

// ─── Lifecycle ────────────────────────────────────────────────────────────────

/// Create a new Memoire instance at `db_path`.
/// Pass `":memory:"` for an ephemeral in-memory store.
/// Returns NULL on failure.
#[no_mangle]
pub extern "C" fn memoire_new(db_path: *const c_char) -> *mut MemoireHandle {
    let path = match to_str(db_path) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };
    let instance = if path == ":memory:" {
        Memoire::in_memory()
    } else {
        Memoire::new(path)
    };
    match instance {
        Ok(m) => Box::into_raw(Box::new(MemoireHandle(m))),
        Err(e) => {
            log::error!("memoire_new failed: {e}");
            ptr::null_mut()
        }
    }
}

/// Destroy a Memoire handle. Safe to call on NULL.
///
/// # Safety
/// `handle` must be either NULL or a pointer previously returned by
/// `memoire_new` that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn memoire_free(handle: *mut MemoireHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) }
    }
}

// ─── Core API ────────────────────────────────────────────────────────────────

/// Store `content`. Returns chunks stored (>= 1) or -1 on error.
#[no_mangle]
pub extern "C" fn memoire_remember(handle: *mut MemoireHandle, content: *const c_char) -> c_int {
    let m = match mut_ref(handle) {
        Some(h) => h,
        None => return -1,
    };
    let content = match to_str(content) {
        Some(s) => s,
        None => return -1,
    };
    match m.remember(content) {
        Ok(ids) => ids.len() as c_int,
        Err(e) => {
            log::error!("memoire_remember: {e}");
            -1
        }
    }
}

/// Retrieve the `top_k` most similar memories to `query`.
///
/// Returns a heap-allocated null-terminated JSON string, or NULL on error.
/// **Caller MUST free this with `memoire_free_string`.**
///
/// JSON: `[{"id":1,"content":"...","score":0.91,"created_at":1705123456}, ...]`
#[no_mangle]
pub extern "C" fn memoire_recall(
    handle: *const MemoireHandle,
    query: *const c_char,
    top_k: c_int,
) -> *mut c_char {
    let m = match const_ref(handle) {
        Some(h) => h,
        None => return ptr::null_mut(),
    };
    let query = match to_str(query) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };
    let k = if top_k <= 0 { 5 } else { top_k as usize };

    match m.recall(query, k) {
        Ok(memories) => {
            let json = serde_json::to_string(&memories).unwrap_or_else(|_| "[]".to_string());
            match CString::new(json) {
                Ok(s) => s.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(e) => {
            log::error!("memoire_recall: {e}");
            ptr::null_mut()
        }
    }
}

/// Conditionally reinforce a memory based on whether the agent actually used it.
///
/// `task_succeeded`: non-zero = succeeded.
/// Returns 1=reinforced, 0=not reinforced, -1=error.
#[no_mangle]
pub extern "C" fn memoire_reinforce_if_used(
    handle: *mut MemoireHandle,
    memory_id: c_longlong,
    agent_output: *const c_char,
    task_succeeded: c_int,
) -> c_int {
    let m = match mut_ref(handle) {
        Some(h) => h,
        None => return -1,
    };
    let output = match to_str(agent_output) {
        Some(s) => s,
        None => return -1,
    };
    match m.reinforce_if_used(memory_id, output, task_succeeded != 0) {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(e) => {
            log::error!("memoire_reinforce_if_used: {e}");
            -1
        }
    }
}

/// Delete memory by id. Returns 1=deleted, 0=not found, -1=error.
#[no_mangle]
pub extern "C" fn memoire_forget(handle: *mut MemoireHandle, id: c_longlong) -> c_int {
    match mut_ref(handle) {
        Some(m) => match m.forget(id) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(e) => {
                log::error!("memoire_forget: {e}");
                -1
            }
        },
        None => -1,
    }
}

/// Total stored chunks, or -1 on error.
#[no_mangle]
pub extern "C" fn memoire_count(handle: *const MemoireHandle) -> c_longlong {
    match const_ref(handle) {
        Some(m) => m.count().unwrap_or(-1),
        None => -1,
    }
}

/// Erase all memories. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn memoire_clear(handle: *mut MemoireHandle) -> c_int {
    match mut_ref(handle) {
        Some(m) => match m.clear() {
            Ok(()) => 0,
            Err(_) => -1,
        },
        None => -1,
    }
}

/// Free a string returned by `memoire_recall`. Safe to call on NULL.
///
/// # Safety
/// `s` must be either NULL or a pointer previously returned by
/// `memoire_recall` that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn memoire_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) }
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

fn to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

fn mut_ref(handle: *mut MemoireHandle) -> Option<&'static mut Memoire> {
    if handle.is_null() {
        return None;
    }
    Some(unsafe { &mut (*handle).0 })
}

fn const_ref(handle: *const MemoireHandle) -> Option<&'static Memoire> {
    if handle.is_null() {
        return None;
    }
    Some(unsafe { &(*handle).0 })
}
