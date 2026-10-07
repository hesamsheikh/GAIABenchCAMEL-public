#!/usr/bin/env python3
"""Analyze the paired structured-vs-self-organized memory ablation."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correct(row: dict[str, Any], key: str = "score") -> bool:
    return row.get(key) in (1, True)


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return (center - margin, center + margin)


def paired_difference_interval(
    left_only: int, right_only: int, total: int
) -> tuple[float, float]:
    """Paired Wald interval for p(left correct)-p(right correct)."""
    difference = (left_only - right_only) / total
    variance = (
        left_only
        + right_only
        - ((left_only - right_only) ** 2 / total)
    ) / (total * total)
    margin = 1.959963984540054 * math.sqrt(max(variance, 0.0))
    return (difference - margin, difference + margin)


def nested_number(row: dict[str, Any], *keys: str) -> float:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key)
    return float(value or 0)


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    guarded = sum(correct(row) for row in rows)
    raw = sum(correct(row, "raw_score") for row in rows)
    by_level: dict[str, dict[str, int | float]] = {}
    for level in sorted({int(row["level"]) for row in rows}):
        selected = [row for row in rows if int(row["level"]) == level]
        level_correct = sum(correct(row) for row in selected)
        by_level[str(level)] = {
            "total": len(selected),
            "correct": level_correct,
            "accuracy": level_correct / len(selected),
        }

    guard_changed = sum(
        row.get("submitted_answer") != row.get("raw_model_answer")
        for row in rows
        if row.get("error") is None
    )
    guard_rescued = sum(
        correct(row) and not correct(row, "raw_score") for row in rows
    )
    guard_harmed = sum(
        not correct(row) and correct(row, "raw_score") for row in rows
    )
    loaded_counts = Counter(
        int(row.get("workflows_loaded_count") or 0) for row in rows
    )

    token_paths = {
        "solve_tokens": ("efficiency", "solve", "usage", "total_tokens"),
        "selection_tokens": (
            "efficiency",
            "memory_selection",
            "usage",
            "total_tokens",
        ),
        "overall_tokens": ("efficiency", "overall", "usage", "total_tokens"),
        "loaded_context_chars": (
            "efficiency",
            "loaded_memory",
            "context_chars",
        ),
        "loaded_context_tokens_estimate": (
            "efficiency",
            "loaded_memory",
            "context_tokens_estimate",
        ),
    }
    totals = {
        label: sum(nested_number(row, *path) for row in rows)
        for label, path in token_paths.items()
    }
    totals.update(
        {
            "tool_calls": sum(len(row.get("tool_calls") or []) for row in rows),
            "solve_seconds": sum(
                nested_number(row, "timing", "solve_seconds") for row in rows
            ),
            "selection_seconds": sum(
                nested_number(row, "timing", "memory_selection_seconds")
                for row in rows
            ),
            "total_seconds": sum(
                nested_number(row, "timing", "total_seconds") for row in rows
            ),
        }
    )
    averages = {key: value / count for key, value in totals.items()}
    interval = wilson_interval(guarded, count)

    return {
        "tasks": count,
        "guarded_correct": guarded,
        "guarded_accuracy": guarded / count,
        "guarded_wilson_95": interval,
        "raw_correct": raw,
        "raw_accuracy": raw / count,
        "errors": sum(row.get("error") is not None for row in rows),
        "error_tasks": [
            {"task_id": row["task_id"], "error": row.get("error")}
            for row in rows
            if row.get("error") is not None
        ],
        "guard_changed": guard_changed,
        "guard_rescued": guard_rescued,
        "guard_harmed": guard_harmed,
        "selection_errors": sum(
            bool((row.get("workflow_selection_audit") or {}).get("error"))
            for row in rows
        ),
        "loaded_count_distribution": dict(sorted(loaded_counts.items())),
        "by_level": by_level,
        "totals": totals,
        "averages": averages,
    }


def source_memory_ids(row: dict[str, Any]) -> set[str]:
    filenames = (row.get("workflow_selection_audit") or {}).get(
        "selected_filenames", []
    )
    ids: set[str] = set()
    for filename in filenames:
        match = re.search(r"_([0-9a-f]{32})_", filename)
        if match:
            ids.add(match.group(1))
    return ids


def paired_summary(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    left = {row["task_id"]: row for row in left_rows}
    right = {row["task_id"]: row for row in right_rows}
    if set(left) != set(right):
        raise ValueError("The two result files do not contain the same task IDs")

    def outcome_counts(
        task_ids: list[str], score_key: str = "score"
    ) -> dict[str, int | float]:
        both_correct = both_wrong = left_only = right_only = 0
        for task_id in task_ids:
            left_correct = correct(left[task_id], score_key)
            right_correct = correct(right[task_id], score_key)
            if left_correct and right_correct:
                both_correct += 1
            elif left_correct:
                left_only += 1
            elif right_correct:
                right_only += 1
            else:
                both_wrong += 1
        total = len(task_ids)
        interval = paired_difference_interval(left_only, right_only, total)
        return {
            "tasks": total,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "left_only": left_only,
            "right_only": right_only,
            "difference": (left_only - right_only) / total,
            "difference_95": interval,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(
                left_only, right_only
            ),
        }

    all_task_ids = list(left)
    guarded = outcome_counts(all_task_ids)
    raw = outcome_counts(all_task_ids, "raw_score")
    no_error_task_ids = [
        task_id
        for task_id in all_task_ids
        if left[task_id].get("error") is None
        and right[task_id].get("error") is None
    ]
    no_error = outcome_counts(no_error_task_ids)

    by_level: dict[str, Counter[str]] = {}
    same_source_sets = 0
    jaccards: list[float] = []
    for task_id in left:
        left_correct = correct(left[task_id])
        right_correct = correct(right[task_id])
        level = str(left[task_id]["level"])
        counter = by_level.setdefault(level, Counter())
        if left_correct and right_correct:
            counter["both_correct"] += 1
        elif left_correct:
            counter["left_only"] += 1
        elif right_correct:
            counter["right_only"] += 1
        else:
            counter["both_wrong"] += 1
        counter["total"] += 1

        left_ids = source_memory_ids(left[task_id])
        right_ids = source_memory_ids(right[task_id])
        same_source_sets += left_ids == right_ids
        union = left_ids | right_ids
        jaccards.append(len(left_ids & right_ids) / len(union) if union else 1.0)

    return {
        "guarded": guarded,
        "raw": raw,
        "guarded_without_any_execution_errors": no_error,
        "by_level": {
            level: dict(counter) for level, counter in sorted(by_level.items())
        },
        "same_selected_source_set_tasks": same_source_sets,
        "mean_selected_source_jaccard": statistics.mean(jaccards),
        "tasks_with_error_in_either": sum(
            left[task_id].get("error") is not None
            or right[task_id].get("error") is not None
            for task_id in left
        ),
        "shared_error_tasks": sum(
            left[task_id].get("error") is not None
            and right[task_id].get("error") is not None
            for task_id in left
        ),
    }


def bank_summary(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.md"))
    histories = sorted((path / "_history").rglob("*.md"))
    sizes = [len(file.read_text(encoding="utf-8")) for file in files]
    words = [len(file.read_text(encoding="utf-8").split()) for file in files]
    return {
        "canonical_files": len(files),
        "history_files": len(histories),
        "total_chars": sum(sizes),
        "mean_chars": statistics.mean(sizes),
        "median_chars": statistics.median(sizes),
        "total_words": sum(words),
        "mean_words": statistics.mean(words),
        "median_words": statistics.median(words),
    }


def construction_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    result: dict[str, Any] = {
        "tasks": len(rows),
        "successful_solves": sum(row.get("error") is None for row in rows),
        "solve_errors": sum(row.get("error") is not None for row in rows),
    }
    for strategy in ("structured", "self_organized"):
        branches = [
            row["memory_branches"][strategy]
            for row in rows
            if row.get("memory_branches")
            and strategy in row["memory_branches"]
        ]
        completed_operations = [
            branch["memory_operation"]
            for branch in branches
            if branch.get("memory_operation") is not None
        ]
        operations = Counter(
            operation["actual_operation"] for operation in completed_operations
        )
        result[strategy] = {
            "events": len(branches),
            "completed_events": len(completed_operations),
            "writer_errors": len(branches) - len(completed_operations),
            "operations": dict(operations),
            "writer_tool_calls": sum(
                nested_number(branch, "efficiency", "tool_calls")
                for branch in branches
            ),
            "writer_tokens": sum(
                nested_number(
                    branch, "efficiency", "usage", "total_tokens"
                )
                for branch in branches
            ),
            "writer_seconds": sum(
                nested_number(branch, "efficiency", "wall_seconds")
                for branch in branches
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured", required=True, type=Path)
    parser.add_argument("--self-organized", required=True, type=Path)
    parser.add_argument("--structured-bank", required=True, type=Path)
    parser.add_argument("--self-organized-bank", required=True, type=Path)
    parser.add_argument("--construction", required=True, type=Path)
    args = parser.parse_args()

    structured_rows = load_jsonl(args.structured)
    self_rows = load_jsonl(args.self_organized)
    result = {
        "structured": summarize_run(structured_rows),
        "self_organized": summarize_run(self_rows),
        "paired": paired_summary(structured_rows, self_rows),
        "banks": {
            "structured": bank_summary(args.structured_bank),
            "self_organized": bank_summary(args.self_organized_bank),
        },
        "construction": construction_summary(args.construction),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
