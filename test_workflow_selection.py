from types import SimpleNamespace

import pytest

from main_single_agent_with_memory import (
    WorkflowSelectionError,
    select_relevant_workflows,
)


class _Memory:
    def __init__(self):
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1


class _Tool:
    def __init__(self, name):
        self.name = name


class _SelectorAgent:
    def __init__(self, response="2, 1, 3", tool_calls=None):
        self._tools = {"research": _Tool("research")}
        self._response = response
        self._tool_calls = tool_calls or []
        self._system_message = None
        self._original_system_message = None
        self.memory = _Memory()
        self.last_prompt = None
        self.system_updates = []

    @property
    def tool_dict(self):
        return self._tools

    def remove_tools(self, tool_names):
        for name in tool_names:
            self._tools.pop(name, None)

    def add_tools(self, tools):
        for tool in tools:
            self._tools[tool.name] = tool

    def update_system_message(self, message):
        self._original_system_message = message
        self._system_message = message
        self.system_updates.append(message)
        self.memory.clear()

    def step(self, message):
        assert self._tools == {}
        self.last_prompt = message.content
        return SimpleNamespace(
            msgs=[SimpleNamespace(content=self._response)],
            info={"tool_calls": self._tool_calls},
        )


def _metadata():
    return [
        {
            "file_path": f"workflow_{index}.md",
            "title": f"Workflow {index}",
            "description": "Reusable workflow",
            "tags": ["test"],
        }
        for index in range(1, 5)
    ]


def test_agentic_selector_is_tool_free_and_restores_tools():
    agent = _SelectorAgent()

    selected, audit = select_relevant_workflows(
        agent, _metadata(), 3, "Test question"
    )

    assert selected == ["workflow_2.md", "workflow_1.md", "workflow_3.md"]
    assert audit["method"] == "agent_selected"
    assert audit["selected_numbers"] == [2, 1, 3]
    assert audit["selector_tool_calls"] == 0
    assert list(agent.tool_dict) == ["research"]
    assert agent.memory.clear_count == 2
    assert "workflow-memory retrieval agent" in agent.system_updates[0]
    assert agent.system_updates[-1] is None
    assert "highest expected procedural usefulness" in agent.last_prompt
    assert "Do not solve or research the task" in agent.last_prompt
    assert "Do not prefer or reject a memory merely because it is broad" in (
        agent.last_prompt
    )


def test_selector_tool_attempt_is_audited_and_fails_closed():
    agent = _SelectorAgent(tool_calls=[{"name": "research"}])

    with pytest.raises(WorkflowSelectionError) as exc_info:
        select_relevant_workflows(agent, _metadata(), 3, "Test question")

    audit = exc_info.value.audit
    assert audit["method"] == "agent_selection_failed"
    assert audit["selector_tool_calls"] == 1
    assert "attempted to use executable tools" in audit["error"]
    assert list(agent.tool_dict) == ["research"]
    assert agent.memory.clear_count == 2


def test_selector_requires_exactly_three_valid_distinct_choices():
    agent = _SelectorAgent(response="2, 2, 99")

    with pytest.raises(WorkflowSelectionError) as exc_info:
        select_relevant_workflows(agent, _metadata(), 3, "Test question")

    audit = exc_info.value.audit
    assert audit["method"] == "agent_selection_failed"
    assert "expected exactly 3" in audit["error"]
