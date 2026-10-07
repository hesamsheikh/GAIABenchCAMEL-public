import json
from pathlib import Path

import pytest

from analyze_schema_ablation import (
    construction_summary,
    exact_mcnemar_p,
    paired_summary,
)
from camel.utils.context_utils import (
    ContextUtility,
    SelfOrganizedWorkflowSummary,
)
from src.experiment_utils import parse_level_counts, select_stratified_tasks
from src.gaia import GAIABenchmark


def test_schema_ablation_paired_statistics_are_task_matched():
    left = []
    right = []
    for index, outcome in enumerate(
        [(1, 1), (1, 0), (1, 0), (0, 1), (0, 0)]
    ):
        common = {
            "task_id": f"task-{index}",
            "level": 1,
            "error": None,
            "workflow_selection_audit": {"selected_filenames": []},
        }
        left.append({**common, "score": outcome[0], "raw_score": outcome[0]})
        right.append({**common, "score": outcome[1], "raw_score": outcome[1]})

    result = paired_summary(left, right)["guarded"]

    assert result["tasks"] == 5
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["left_only"] == 2
    assert result["right_only"] == 1
    assert result["difference"] == pytest.approx(0.2)
    assert result["exact_mcnemar_two_sided_p"] == 1.0
    assert exact_mcnemar_p(16, 9) == pytest.approx(0.2295229435)


def test_construction_summary_counts_writer_failures(tmp_path):
    path = tmp_path / "construction.jsonl"
    rows = [
        {
            "error": None,
            "memory_branches": {
                "structured": {
                    "memory_operation": {"actual_operation": "create"},
                    "efficiency": {},
                },
                "self_organized": {
                    "memory_operation": None,
                    "efficiency": {},
                },
            },
        }
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = construction_summary(path)

    assert result["structured"]["completed_events"] == 1
    assert result["structured"]["writer_errors"] == 0
    assert result["structured"]["operations"] == {"create": 1}
    assert result["self_organized"]["events"] == 1
    assert result["self_organized"]["completed_events"] == 0
    assert result["self_organized"]["writer_errors"] == 1
    assert result["self_organized"]["operations"] == {}


def test_stratified_subset_is_deterministic_and_balanced():
    tasks = [
        {"task_id": f"level-{level}-task-{index}", "Level": level}
        for level, available in ((1, 8), (2, 12), (3, 5))
        for index in range(available)
    ]
    counts = parse_level_counts("1:3,2:5,3:2")

    first = select_stratified_tasks(tasks, counts, "schema-ablation-v1")
    second = select_stratified_tasks(reversed(tasks), counts, "schema-ablation-v1")

    assert [task["task_id"] for task in first] == [
        task["task_id"] for task in second
    ]
    assert {
        level: sum(task["Level"] == level for task in first)
        for level in (1, 2, 3)
    } == counts


def test_self_organized_memory_preserves_normal_retrieval_metadata(tmp_path):
    summary = SelfOrganizedWorkflowSummary(
        agent_title="research_assistant",
        task_title="Identify publication from evidence",
        task_description=(
            "Identify a publication using partial bibliographic evidence and "
            "return the requested identifier."
        ),
        tags=["web-research", "bibliography"],
        memory="Use whichever observations from the attempt seem reusable.",
    )
    context = ContextUtility(
        working_directory=str(tmp_path),
        create_folder=True,
        use_session_subfolder=False,
    )
    markdown = context.structured_output_to_markdown(summary)
    workflow_path = tmp_path / "self-organized.md"
    workflow_path.write_text(markdown, encoding="utf-8")

    metadata = context.extract_workflow_info(str(workflow_path))

    assert metadata["title"] == "Identify publication from evidence"
    assert metadata["description"].startswith("Identify a publication")
    assert metadata["tags"] == ["web-research", "bibliography"]
    assert "### Tools" not in markdown
    assert "### Steps" not in markdown
    assert "### Memory" in markdown


@pytest.mark.asyncio
async def test_paired_memory_build_uses_isolated_bank_directories():
    benchmark = object.__new__(GAIABenchmark)
    benchmark.paired_workflow_memory_dirs = {
        "structured": "workflows/structured",
        "self_organized": "workflows/self_organized",
    }
    calls = []

    class BranchAgent:
        def __init__(self, strategy):
            self.strategy = strategy
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    branches = {}

    def clone(_source, strategy):
        branch = BranchAgent(strategy)
        branches[strategy] = branch
        return branch

    async def reflect(branch, result, memory_strategy="structured"):
        calls.append(("reflect", branch.strategy, memory_strategy))
        result["branch_marker"] = branch.strategy
        return benchmark._phase_efficiency()

    async def save(
        branch,
        task,
        result_data=None,
        workflow_memory_dir=None,
        memory_strategy="structured",
    ):
        calls.append(
            (
                "save",
                branch.strategy,
                memory_strategy,
                workflow_memory_dir,
                result_data["branch_marker"],
            )
        )
        result_data["memory_operation"] = {
            "actual_operation": "create",
            "file_path": str(
                Path(workflow_memory_dir) / f"{task['task_id']}.md"
            ),
        }
        return benchmark._phase_efficiency()

    benchmark._clone_memory_writer_agent = clone
    benchmark._reflect_on_result = reflect
    benchmark._save_workflow_memory = save
    result = {"efficiency": {}, "submitted_answer": "answer"}

    await benchmark._build_paired_workflow_memories(
        object(), {"task_id": "task-1"}, result
    )

    assert calls == [
        ("reflect", "structured", "structured"),
        (
            "save",
            "structured",
            "structured",
            "workflows/structured",
            "structured",
        ),
        ("reflect", "self_organized", "self_organized"),
        (
            "save",
            "self_organized",
            "self_organized",
            "workflows/self_organized",
            "self_organized",
        ),
    ]
    assert set(result["memory_branches"]) == {
        "structured",
        "self_organized",
    }
    assert all(branch.reset_count == 1 for branch in branches.values())
    json.dumps(result)
