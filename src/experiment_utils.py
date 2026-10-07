"""Reproducible task-subset helpers for controlled GAIA experiments."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List


def parse_level_counts(value: str) -> Dict[int, int]:
    """Parse a string such as ``1:15,2:28,3:7`` into level counts."""
    counts: Dict[int, int] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            raw_level, raw_count = part.split(":", 1)
            level = int(raw_level)
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                "GAIA_STRATIFIED_COUNTS must look like '1:15,2:28,3:7'"
            ) from exc
        if level not in {1, 2, 3} or count < 0 or level in counts:
            raise ValueError(
                "GAIA_STRATIFIED_COUNTS requires unique levels 1-3 and "
                "non-negative counts"
            )
        counts[level] = count
    if not counts or sum(counts.values()) < 1:
        raise ValueError("GAIA_STRATIFIED_COUNTS must select at least one task")
    return counts


def _stable_key(seed: str, namespace: str, task_id: str) -> str:
    payload = f"{seed}:{namespace}:{task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_stratified_tasks(
    tasks: Iterable[Dict[str, Any]],
    counts: Dict[int, int],
    seed: str,
) -> List[Dict[str, Any]]:
    """Select and order a deterministic stratified subset by task ID."""
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[int(task["Level"])].append(task)

    selected: List[Dict[str, Any]] = []
    for level, count in sorted(counts.items()):
        candidates = sorted(
            grouped[level],
            key=lambda task: _stable_key(seed, f"level-{level}", task["task_id"]),
        )
        if len(candidates) < count:
            raise ValueError(
                f"Requested {count} Level {level} tasks, but only "
                f"{len(candidates)} are available"
            )
        selected.extend(candidates[:count])

    return sorted(
        selected,
        key=lambda task: _stable_key(seed, "execution-order", task["task_id"]),
    )
