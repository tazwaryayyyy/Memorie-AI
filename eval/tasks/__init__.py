from __future__ import annotations

from typing import List

from .base import Family, Lesson, Task, families_hash

SPLITS = ("dev", "test")


def load_split(name: str) -> List[Family]:
    if name == "dev":
        from .dev_families import dev_families

        return dev_families()
    if name == "test":
        from .heldout_families import heldout_families

        return heldout_families()
    raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")


__all__ = ["Family", "Lesson", "Task", "families_hash", "load_split", "SPLITS"]
