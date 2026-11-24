"""Prompt construction and response parsing. Identical for every arm except the notes section."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from .tasks.base import Task

SYSTEM_PROMPT = (
    "You are a careful Python coding agent working in a small repository. "
    "You are given a task, the current solution.py and the visible tests. "
    "Reply with the complete new contents of solution.py in exactly one ```python fenced block."
)

NOTES_HEADER = (
    "# Notes from earlier sessions\n"
    "These lessons were recorded in earlier sessions. They may be wrong or irrelevant to this task; "
    "use your own judgement."
)

# --prompt-policy tagged: the engine's own outcome record is shown next to each note. Tags are derived ONLY
# from the (exposure, trust) pair the engine reports, never from lesson text, kind or family, so they cannot
# leak which lesson is correct; they can only reflect outcomes recorded by earlier feedback.
POLICY_PLAIN = "plain"
POLICY_TAGGED = "tagged"
PROMPT_POLICIES = (POLICY_PLAIN, POLICY_TAGGED)
TAG_RULES_VERSION = "1"  # bump when `tag_for` changes; enters the run fingerprint for the tagged policy

NOTES_HEADER_TAGGED = (
    NOTES_HEADER + "\n"
    "Each note starts with a tag giving its record from earlier sessions: `[UNTESTED]` means it has no recorded "
    "outcome yet; `[TRUST x.xx: ...]` summarises outcomes recorded when it was used before. A record built from "
    "one or two sessions is weak evidence."
)

OUTPUT_FORMAT_NO_MEMORY = (
    "# Output format\n"
    "Reply with the complete solution.py in one ```python block. Nothing else is required."
)

OUTPUT_FORMAT_MEMORY = (
    "# Output format\n"
    "1. Reply with the complete solution.py in one ```python block.\n"
    "2. After the block, end with one line `USED_MEMORY: [<numbers>]` listing the numbers of the notes "
    "you actually relied on (for example `USED_MEMORY: [1, 3]`), or `USED_MEMORY: []` if you relied on none."
)


@dataclass(frozen=True)
class Note:
    """A retrieved lesson plus the engine-reported fields the tagged policy may render."""

    text: str
    trust: Optional[float] = None
    exposure: Optional[str] = None


_TRUST_EPS = 0.005  # trust is shown to 2 decimals; |trust - 0.5| below this reads as "even"


def tag_for(note: Note) -> Optional[str]:
    """Outcome tag for one note, or None when the engine gave no outcome semantics (plain/hidden/unknown).

    With the Beta(2,2) posterior `(2+s)/(4+s+f)` an evidence-bearing row has trust > 0.5 iff s > f and
    trust < 0.5 iff f > s (decay toward 0.5 never crosses it), so the wording below states exactly what the
    number means. `trial` rows have no evidence at all. Deliberately NOT `MemoryPolicy.evaluate`: its default
    bands put trust 0.40, 0.50 and 0.60 in the same `hint` bucket, which would hide the very difference
    this policy exists to show.
    """
    if note.exposure == "trial":
        return "[UNTESTED]"
    if note.exposure not in ("hint", "follow") or note.trust is None:
        return None
    if note.trust > 0.5 + _TRUST_EPS:
        meaning = "worked more often than it failed"
    elif note.trust < 0.5 - _TRUST_EPS:
        meaning = "failed more often than it worked"
    else:
        meaning = "mixed record"
    return f"[TRUST {note.trust:.2f}: {meaning}]"


def _note_text(note: Union[str, Note]) -> str:
    return note.text if isinstance(note, Note) else note


def build_user_prompt(task: Task, memories: Optional[Sequence[Union[str, Note]]],
                      policy: str = POLICY_PLAIN) -> str:
    parts = [
        f"# Task\n{task.spec}",
        f"# solution.py\n```python\n{task.solution_stub.rstrip()}\n```",
        f"# test_visible.py\n```python\n{task.visible_tests.rstrip()}\n```",
    ]
    if memories is None:
        parts.append(OUTPUT_FORMAT_NO_MEMORY)
    else:
        if policy not in PROMPT_POLICIES:
            raise ValueError(f"unknown prompt policy {policy!r}; expected one of {PROMPT_POLICIES}")
        tagged = policy == POLICY_TAGGED
        lines = []
        for i, mem in enumerate(memories, start=1):
            tag = tag_for(mem) if tagged and isinstance(mem, Note) else None
            lines.append(f"[M{i}] {tag} {_note_text(mem)}" if tag else f"[M{i}] {_note_text(mem)}")
        notes = "\n".join(lines) if lines else "(no notes were retrieved for this task)"
        parts.append(f"{NOTES_HEADER_TAGGED if tagged else NOTES_HEADER}\n{notes}")
        parts.append(OUTPUT_FORMAT_MEMORY)
    return "\n\n".join(parts)


def build_messages(task: Task, memories: Optional[Sequence[Union[str, Note]]],
                   policy: str = POLICY_PLAIN) -> List[Dict[str, str]]:
    """`memories=None` is Arm A (no memory section, no USED_MEMORY instruction). Plain strings (and every
    note under the plain policy) render exactly as before; tags need `policy=tagged` AND `Note` objects."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task, memories, policy)},
    ]


def prompt_template_hash_input(policy: str = POLICY_PLAIN) -> str:
    parts = [SYSTEM_PROMPT, NOTES_HEADER, OUTPUT_FORMAT_NO_MEMORY, OUTPUT_FORMAT_MEMORY]
    if policy == POLICY_TAGGED:
        parts += [NOTES_HEADER_TAGGED, f"tag-rules:{TAG_RULES_VERSION}"]
    return "\n".join(parts)


_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_USED = re.compile(r"USED_MEMORY\s*:\s*\[([^\]]*)\]", re.IGNORECASE)
_PY_LANGS = {"", "python", "py", "python3"}


@dataclass
class Parsed:
    code: Optional[str]
    code_blocks: int
    used_indices: List[int] = field(default_factory=list)   # 1-based, valid, de-duplicated
    invalid_used: List[str] = field(default_factory=list)
    used_parse_ok: bool = False                              # a USED_MEMORY line was found

    def to_json(self) -> dict:
        return {
            "code_found": self.code is not None,
            "code_blocks": self.code_blocks,
            "used_indices": self.used_indices,
            "invalid_used": self.invalid_used,
            "used_parse_ok": self.used_parse_ok,
        }


def extract_code(text: str, entrypoints: Sequence[str]) -> "tuple[Optional[str], int]":
    blocks = [body for lang, body in _FENCE.findall(text) if lang.lower() in _PY_LANGS and body.strip()]
    if not blocks:
        return None, 0
    defs = [re.compile(rf"^\s*(async\s+)?def\s+{re.escape(name)}\s*\(", re.MULTILINE) for name in entrypoints]
    for body in reversed(blocks):
        if all(d.search(body) for d in defs):
            return body.strip("\n") + "\n", len(blocks)
    return blocks[-1].strip("\n") + "\n", len(blocks)


def parse_used_memory(text: str, n_shown: int) -> "tuple[List[int], List[str], bool]":
    matches = _USED.findall(text)
    if not matches:
        return [], [], False
    used: List[int] = []
    invalid: List[str] = []
    for raw in matches[-1].split(","):
        token = raw.strip().strip("[]").strip()
        if not token or token.lower() == "none":
            continue
        token = re.sub(r"^[Mm]\s*", "", token)
        if token.isdigit() and 1 <= int(token) <= n_shown:
            value = int(token)
            if value not in used:
                used.append(value)
        else:
            invalid.append(raw.strip())
    return used, invalid, True


def parse_response(text: str, task: Task, n_shown: int, expect_used_memory: bool) -> Parsed:
    code, blocks = extract_code(text, task.entrypoints)
    parsed = Parsed(code=code, code_blocks=blocks)
    if expect_used_memory:
        parsed.used_indices, parsed.invalid_used, parsed.used_parse_ok = parse_used_memory(text, n_shown)
    return parsed
