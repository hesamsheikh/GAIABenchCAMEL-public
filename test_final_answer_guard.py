import asyncio
from io import StringIO
from types import SimpleNamespace

import pytest

from src.gaia import GAIABenchmark, MemoryUpdateDecision


def guard(question: str, answer: str):
    benchmark = GAIABenchmark.__new__(GAIABenchmark)
    return benchmark.guard_final_answer(question, answer)


def test_removes_count_label_without_ground_truth():
    result = guard("How many papers were incorrect? Round up.", "41 papers")
    assert result == (
        "41",
        True,
        "removed an unrequested label from a numeric answer",
    )


def test_removes_generic_answer_prefix():
    result = guard("What is the five-digit ZIP code?", "The answer is 34689")
    assert result == ("34689", True, "removed generic answer prefix")


def test_preserves_requested_units():
    result = guard(
        "How much time elapsed? Include the unit in your answer.",
        "12 seconds",
    )
    assert result == ("12 seconds", False, None)


def test_preserves_non_numeric_exact_answer():
    assert guard("What character is missing?", "backtick") == (
        "backtick",
        False,
        None,
    )


def test_preserves_numeric_list_format():
    assert guard(
        "Return the ZIP codes separated by commas.", "34689, 33701"
    ) == ("34689, 33701", False, None)


def test_hidden_labels_are_not_scored_or_persisted():
    benchmark = GAIABenchmark.__new__(GAIABenchmark)
    benchmark.results = []
    benchmark.save_individual_results = False
    benchmark.experiment_metadata = {"ground_truth_available": False}
    benchmark.question_scorer = lambda *_: (_ for _ in ()).throw(
        AssertionError("hidden test labels must not be scored")
    )

    agent = SimpleNamespace(
        memory=SimpleNamespace(get_context=lambda: ([], 0))
    )
    response = SimpleNamespace(
        msgs=[SimpleNamespace(content="<final_answer>41 papers</final_answer>")],
        info={"tool_calls": []},
    )
    task = {
        "task_id": "hidden-label-task",
        "Question": "How many papers were published?",
        "Level": 1,
        "Final answer": "?",
    }

    result = benchmark._process_result(
        agent,
        task,
        response,
        StringIO(),
        labels_available=False,
    )

    assert result["raw_model_answer"] == "41 papers"
    assert result["submitted_answer"] == "41"
    assert result["ground_truth"] is None
    assert result["raw_score"] is None
    assert result["score"] is None


def test_missing_final_answer_tag_is_a_protocol_failure():
    benchmark = GAIABenchmark.__new__(GAIABenchmark)
    response = SimpleNamespace(
        msgs=[SimpleNamespace(content="I think the answer is 28.")],
        info={"tool_calls": []},
    )

    with pytest.raises(
        RuntimeError,
        match="omitted the required <final_answer> tag",
    ):
        benchmark._process_result(
            SimpleNamespace(),
            {
                "task_id": "missing-tag",
                "Question": "What is the answer?",
                "Level": 1,
                "Final answer": "?",
            },
            response,
            StringIO(),
            labels_available=False,
        )


def test_post_answer_agent_selects_exact_update_target(tmp_path):
    first = tmp_path / "web_workflow.md"
    first.write_text(
        "### Task Title\nResearch a web source\n\n"
        "### Task Description\nFind and verify information online.\n\n"
        "### Tags\n- web-research\n",
        encoding="utf-8",
    )
    second = tmp_path / "code_workflow.md"
    second.write_text(
        "### Task Title\nAnalyze data with code\n\n"
        "### Task Description\nCompute a result with Python.\n\n"
        "### Tags\n- data-analysis\n",
        encoding="utf-8",
    )

    class SelectingAgent:
        prompt = ""

        async def astep(
            self, message, response_format=None, disable_tools=False
        ):
            assert disable_tools is True
            self.prompt = message.content
            return SimpleNamespace(msgs=[SimpleNamespace(
                parsed=MemoryUpdateDecision(
                    action="update",
                    target_workflow_filename="code_workflow",
                    reason="The completed task used the same code workflow.",
                ),
                content="",
            )])

    benchmark = GAIABenchmark.__new__(GAIABenchmark)
    benchmark.workflow_memory_dir = str(tmp_path)
    manager = SimpleNamespace(
        _loaded_workflow_paths={},
        _loaded_workflow_contents=[],
    )
    agent = SelectingAgent()

    decision, available_count = asyncio.run(
        benchmark._agent_select_memory_operation(agent, manager)
    )

    assert available_count == 2
    assert "web_workflow" in agent.prompt
    assert "code_workflow" in agent.prompt
    assert decision.action == "update"
    assert decision.target_workflow_filename == "code_workflow"
    assert list(manager._loaded_workflow_paths) == ["code_workflow"]
    assert manager._loaded_workflow_contents[0]["filename"] == "code_workflow"


def test_post_answer_selection_accepts_fenced_json(tmp_path):
    workflow = tmp_path / "logic_workflow.md"
    workflow.write_text(
        "### Task Title\nSolve a logic problem\n\n"
        "### Task Description\nModel constraints and verify with code.\n\n"
        "### Tags\n- logic-puzzle\n",
        encoding="utf-8",
    )

    class FencedJsonAgent:
        async def astep(
            self, message, response_format=None, disable_tools=False
        ):
            assert disable_tools is True
            return SimpleNamespace(
                info={"usage": {}, "model_calls": 1, "tool_calls": []},
                msgs=[SimpleNamespace(
                    parsed=None,
                    content=(
                        "```json\n"
                        '{"action":"update",'
                        '"target_workflow_filename":"logic_workflow",'
                        '"reason":"Same reusable logic procedure."}'
                        "\n```"
                    ),
                )],
            )

    benchmark = GAIABenchmark.__new__(GAIABenchmark)
    benchmark.workflow_memory_dir = str(tmp_path)
    benchmark.native_structured_output = False
    manager = SimpleNamespace(
        _loaded_workflow_paths={},
        _loaded_workflow_contents=[],
    )

    decision, available_count = asyncio.run(
        benchmark._agent_select_memory_operation(FencedJsonAgent(), manager)
    )

    assert available_count == 1
    assert decision.action == "update"
    assert decision.target_workflow_filename == "logic_workflow"
