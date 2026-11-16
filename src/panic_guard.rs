//! Panic containment for FFI-facing layers (currently `py_ffi`).
//!
//! A panic that unwinds through a `MutexGuard` poisons the mutex, which makes
//! every later call on that object fail. PyO3 already stops a panic at the
//! Python boundary, but it surfaces as `PanicException`, a `BaseException` that
//! `except Exception` does not catch. [`catch_panic`] lets callers turn a panic
//! into an ordinary error instead.
//!
//! Requires unwinding panics (the default). With `panic = "abort"` a panic
//! still aborts the process and nothing here can intercept it.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Best-effort text of a panic payload (`&'static str` and `String` payloads
/// cover `panic!("literal")` and `panic!("{x}")`).
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Run `f`, returning `Err(message)` if it panics.
///
/// Call this **while still holding** any `MutexGuard` that `f` uses. A guard
/// dropped during unwinding poisons its mutex; a guard dropped normally after
/// the panic was caught does not.
///
/// The caller is asserting that state `f` mutated is still usable after a
/// panic (`AssertUnwindSafe`). For `Memoire` this is a judgement call: inner
/// locks that were held at the panic point stay poisoned and later calls
/// report an error rather than panic.
pub fn catch_panic<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| panic_message(&*payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn returns_value_when_no_panic() {
        assert_eq!(catch_panic(|| 41 + 1), Ok(42));
    }

    #[test]
    fn captures_str_payload() {
        let r: Result<(), String> = catch_panic(|| panic!("boom"));
        assert_eq!(r, Err("boom".to_string()));
    }

    #[test]
    fn captures_string_payload() {
        let n = 7;
        let r: Result<(), String> = catch_panic(|| panic!("bad index {n}"));
        assert_eq!(r, Err("bad index 7".to_string()));
    }

    #[test]
    fn reports_unknown_payload() {
        let r: Result<(), String> = catch_panic(|| std::panic::panic_any(123_u32));
        assert_eq!(r, Err("unknown panic payload".to_string()));
    }

    #[test]
    fn catching_inside_the_guard_does_not_poison() {
        let m = Mutex::new(0_i32);
        {
            let mut guard = m.lock().unwrap();
            let r: Result<(), String> = catch_panic(|| {
                *guard += 1;
                panic!("inside");
            });
            assert!(r.is_err());
        } // guard dropped normally, not while unwinding
        assert!(!m.is_poisoned());
        assert_eq!(*m.lock().unwrap(), 1);
    }

    #[test]
    fn catching_outside_the_guard_poisons() {
        // Documents why `with_inner` must catch while holding the guard.
        let m = Mutex::new(0_i32);
        let r: Result<(), String> = catch_panic(|| {
            let _guard = m.lock().unwrap();
            panic!("outside");
        });
        assert!(r.is_err());
        assert!(m.is_poisoned());
    }
}
