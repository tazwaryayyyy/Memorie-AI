"""
Brutal demo moment for judges.

DEMO FROZEN — do not add arms, scenarios, or extra output.
The three arms are fixed. The failure → correction loop is the story.

Shows three arms:
  1. No memory      — agent repeats the same float-money mistake twice.
  2. Memoire + MQCL + Trust — lesson stored after failure, recalled with trust
     score, policy decides FOLLOW → agent avoids the mistake.
  3. Failure-Feedback Loop — wrong memory penalized → trust drops → suppressed →
     correct lesson wins.

The output is designed to be screenshot-worthy for a judge panel.

Usage:
  cargo build --release
  python examples/brutal_moment_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "bindings" / "python"))
from memoire import Memoire, MemoryPolicy  # noqa: E402  # pylint: disable=import-error,wrong-import-position

# ─── Agent simulation ─────────────────────────────────────────────────────────


def simulate_agent_code(task: str, context_block: str) -> str:
    """Deterministic code generator that honours injected memory context."""
    hint = context_block.lower()
    if "discount" in task.lower() or "refund" in task.lower() or "billing" in task.lower():
        if "decimal" in hint or "never use float for money" in hint:
            return "from decimal import Decimal\namount = Decimal('19.99')\n"
        return "amount = float(19.99)\n"
    if "tax" in task.lower():
        return "amount = float(9.99)\n"
    return "pass\n"


def passes_money_tests(code: str) -> bool:
    return "Decimal" in code and "float" not in code


# ─── Arm 1: no memory ─────────────────────────────────────────────────────────

def run_without_memory() -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  ARM 1  ·  No Memory")
    print(sep)

    t1 = "Implement tax computation for billing."
    c1 = simulate_agent_code(t1, "")
    r1 = "PASS" if passes_money_tests(c1) else "FAIL"
    print(f"  Task 1: {t1}")
    print(f"    Code  : {c1.strip()}")
    print(f"    Tests : {r1}")

    t2 = "Implement discount and refund computation for billing."
    c2 = simulate_agent_code(t2, "")
    r2 = "PASS" if passes_money_tests(c2) else "FAIL"
    print(f"\n  Task 2: {t2}")
    print(f"    Code  : {c2.strip()}")
    print(f"    Tests : {r2}")

    print("\n  ★ JUDGE MOMENT: same float mistake repeated. No memory = no learning.")


# ─── Arm 2: Memoire + MQCL + Trust ───────────────────────────────────────────

def run_with_trust() -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  ARM 2  ·  Memoire + MQCL + Trust Score")
    print(sep)

    db_path = "./demo_trust.db"
    db = Path(db_path)
    if db.exists():
        db.unlink()

    policy = MemoryPolicy()

    with Memoire(db_path) as m:
        # ── Task 1: agent fails, lesson stored ───────────────────────────────
        t1 = "Implement tax computation for billing."
        c1 = simulate_agent_code(t1, "")
        r1_pass = passes_money_tests(c1)
        print(f"  Task 1: {t1}")
        print(f"    Code  : {c1.strip()}")
        print(f"    Tests : {'PASS' if r1_pass else 'FAIL'}")

        stored_id = None
        if not r1_pass:
            lesson = (
                "Never use float for money. Use Decimal with explicit cent precision. "
                "Previous billing bug caused by float rounding errors in production."
            )
            ids = m.remember(lesson)
            stored_id = ids[0] if ids else None
            print(
                f"    → Failure detected. Stored corrective memory (id={stored_id}).")

            # No reinforcement yet — this memory has rc=0
            mems_after_store = m.recall("money precision billing", top_k=1)
            if mems_after_store:
                mm = mems_after_store[0]
                print(f"    → Memory trust right after store: {mm.trust:.3f} "
                      f"(rc=0, state={mm.state})")

        # ── Task 2: recall → policy → inject → succeed ───────────────────────
        print()
        t2 = "Implement discount and refund computation for billing."
        print(f"  Task 2: {t2}")

        recalled = m.recall(
            "money precision for refunds and discounts", top_k=3)
        recalled = [r for r in recalled if r.score >= 0.45]

        print(f"\n  [RECALL]  {len(recalled)} result(s)")
        decisions = policy.evaluate(recalled)
        for d in decisions:
            mem = d.memory
            print(
                f"    → \"{mem.content[:55]}{'\u2026' if len(mem.content) > 55 else ''}\" "
                f"| score={mem.score:.2f} | trust={mem.trust:.2f} "
                f"| unc={mem.uncertainty:.2f} | action={d.action.upper()}"
            )
            print(f"       reason: {d.reason}")

        context_block = policy.inject_context(decisions)

        print("\n  [AGENT DECISION]")
        follow_count = sum(1 for d in decisions if d.action == "follow")
        hint_count = sum(1 for d in decisions if d.action == "hint")
        ignore_count = sum(1 for d in decisions if d.action == "ignore")
        if follow_count:
            print(
                f"    → Following {follow_count} high-trust memory/memories. Injecting into context.")
        if hint_count:
            print(f"    → Treating {hint_count} memory/memories as soft hint.")
        if ignore_count:
            print(
                f"    → Suppressing {ignore_count} low-trust memory/memories. Not shown to agent.")

        c2 = simulate_agent_code(t2, context_block)
        r2_pass = passes_money_tests(c2)
        print("\n  [RESULT]")
        print(f"    Code  : {c2.strip()}")
        print(f"    Tests : {'PASS' if r2_pass else 'FAIL'}")

        # Conditional reinforcement + influence log
        if r2_pass and stored_id is not None:
            trust_before = recalled[0].trust if recalled else 0.0
            reinforced = m.reinforce_if_used(
                stored_id, c2, task_succeeded=True)
            if reinforced:
                updated = m.recall("money precision billing", top_k=1)
                trust_after = updated[0].trust if updated else 0.0
                trust_delta = trust_after - trust_before
                print(
                    "    \u2192 Mistake avoided: float precision error (previously failed)")
                print(
                    f"    \u2192 Memory reinforced. Trust updated to {trust_after:.3f} (rc now=1).")
                print()
                print("  [MEMORY INFLUENCE]")
                print(f"    \u2192 memory_id={stored_id}")
                print("    \u2192 changed decision: float \u2192 Decimal")
                print(
                    f"    → trust_delta=+{trust_delta:.2f}  ({trust_before:.3f} → {trust_after:.3f})")
            else:
                print("    → Token overlap below threshold — reinforcement skipped.")

        print("\n  ★ JUDGE MOMENT: agent followed memory → mistake avoided → trust grew.")


# ─── Arm 3: Failure-Feedback Loop ────────────────────────────────────────────

def run_with_failure_feedback() -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  ARM 3  ·  Failure-Feedback Loop (penalize_if_used)")
    print(sep)

    db_path = "./demo_feedback.db"
    db = Path(db_path)
    if db.exists():
        db.unlink()

    policy = MemoryPolicy()

    with Memoire(db_path) as m:
        # ── SETUP: inject a high-quality but WRONG lesson ─────────────────────
        # Text is crafted to pass the quality gate as ACTIVE (score > 0.50) but
        # contains NO "decimal" / "never use float for money" keywords, so the
        # agent simulator will always produce float code when it is the context.
        wrong_lesson = (
            "Fixed billing module: always replaced legacy arithmetic with float "
            "for production speed. Patched critical performance incident. "
            "Test metrics confirmed improved throughput."
        )
        wrong_ids = m.remember(wrong_lesson)
        wrong_id = wrong_ids[0] if wrong_ids else None

        mems_initial = m.recall("money precision billing", top_k=1)
        initial_trust = mems_initial[0].trust if mems_initial else 0.0
        print(
            f"  SETUP: wrong lesson stored (id={wrong_id}), "
            f"initial trust={initial_trust:.3f}"
        )
        print(f"    \"{wrong_lesson[:65]}\u2026\"")

        # ── Task 1: agent follows wrong hint → FAIL ───────────────────────────
        print()
        task = "Implement billing refund computation."
        print(f"  Task 1: {task}")

        recalled = m.recall("billing refund money computation", top_k=5)
        decisions = policy.evaluate(recalled)

        used_ids = [d.memory.id for d in decisions if d.action != "ignore"]

        print(f"\n  [RECALL]  {len(recalled)} result(s)")
        for d in decisions:
            mem = d.memory
            print(
                f"    \u2192 \"{mem.content[:55]}{'…' if len(mem.content) > 55 else ''}\" "
                f"| trust={mem.trust:.3f} | unc={mem.uncertainty:.2f} | action={d.action.upper()}"
            )

        context_block = policy.inject_context(decisions)
        code = simulate_agent_code(task, context_block)
        task_passed = passes_money_tests(code)

        print("\n  [RESULT]")
        print(f"    Code  : {code.strip()}")
        print(f"    Tests : {'PASS' if task_passed else 'FAIL'}")

        # ── Failure feedback: penalize memories that were used ─────────────────
        if not task_passed and used_ids:
            outcomes = m.penalize_if_used(used_ids, failure_severity=1.0)
            print("\n  [FAILURE FEEDBACK]")
            for o in outcomes:
                direction = (
                    "↓ below HINT threshold"
                    if o["trust_after"] < 0.40
                    else "↓ declining"
                )
                print(f"    → memory_id={o['id']} penalized (severity=1.0)")
                print(
                    f"    → trust {o['trust_before']:.3f} → {o['trust_after']:.3f}  "
                    f"({direction})"
                )
                print(
                    f"    → uncertainty now {o['uncertainty_after']:.3f}  "
                    "(high oscillation signal)"
                )
            print("    → System corrected itself after 1 failure.")
        elif not task_passed:
            print("\n  [FAILURE FEEDBACK]")
            print(
                "    \u2192 no memories were injected; wrong lesson was already suppressed")
            print(
                f"    \u2192 wrong lesson trust: {initial_trust:.3f} (below HINT threshold)")

        # ── Store the correct lesson ──────────────────────────────────────────
        correct_lesson = (
            "Never use float for money. Use Decimal with explicit cent precision. "
            "Previous billing bug caused by float rounding errors in production."
        )
        correct_ids = m.remember(correct_lesson)
        correct_id = correct_ids[0] if correct_ids else None
        print(f"\n  \u2192 Stored correct lesson (id={correct_id})")

        # ── Task 2: wrong lesson suppressed, correct lesson wins ──────────────
        print()
        print(f"  Task 2: {task}  (second attempt)")

        recalled2 = m.recall("billing refund money computation", top_k=5)
        decisions2 = policy.evaluate(recalled2)

        print(f"\n  [RECALL]  {len(recalled2)} result(s)")
        for d in decisions2:
            mem = d.memory
            tag = "(penalized)" if wrong_id and mem.id == wrong_id else "(correct)"
            print(
                f"    \u2192 \"{mem.content[:55]}{'…' if len(mem.content) > 55 else ''}\" "
                f"| trust={mem.trust:.3f} | action={d.action.upper()} {tag}"
            )

        context_block2 = policy.inject_context(decisions2)
        code2 = simulate_agent_code(task, context_block2)
        task_passed2 = passes_money_tests(code2)

        print("\n  [RESULT]")
        print(f"    Code  : {code2.strip()}")
        print(f"    Tests : {'PASS' if task_passed2 else 'FAIL'}")

        if task_passed2 and correct_id is not None:
            m.reinforce_if_used(correct_id, code2, task_succeeded=True)
            updated = m.recall("money precision billing", top_k=1)
            if updated:
                print(
                    f"    \u2192 Correct lesson reinforced. Trust now {updated[0].trust:.3f} (rc=1)."
                )

    print(
        "\n  \u2733 JUDGE MOMENT: wrong memory penalized \u2192 suppressed \u2192 "
        "correct lesson won."
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Memoire · Trust Score Demo")
    print("  \"It doesn't just remember — it decides what to trust.\"")
    print("=" * 60)
    run_without_memory()
    run_with_trust()
    run_with_failure_feedback()
    print()


if __name__ == "__main__":
    main()
