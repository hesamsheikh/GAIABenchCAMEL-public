from typing import Optional

import pytest
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function,
)
from openai.types.completion_usage import CompletionUsage

from camel.agents import ChatAgent
from camel.models.stub_model import StubModel
from camel.types import ChatCompletion, ModelType


def _completion(
    content: Optional[str],
    *,
    tool_calls: Optional[list[ChatCompletionMessageFunctionToolCall]] = None,
) -> ChatCompletion:
    return ChatCompletion(
        id="post-task-tool-isolation-test",
        choices=[
            Choice(
                finish_reason="tool_calls" if tool_calls else "stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                    function_call=None,
                    tool_calls=tool_calls,
                ),
            )
        ],
        created=0,
        model="mock-model",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=5,
            prompt_tokens=10,
            total_tokens=15,
        ),
    )


@pytest.mark.asyncio
async def test_disabled_step_keeps_context_and_configured_tools():
    executed = []
    backend_calls = []

    def dangerous_write(path: str) -> str:
        executed.append(path)
        return "written"

    agent = ChatAgent(
        system_message="You are a test agent.",
        model=StubModel(model_type=ModelType.STUB),
        tools=[dangerous_write],
    )

    async def mock_arun(messages, response_format=None, tools=None):
        backend_calls.append((messages, response_format, tools))
        content = "task answer" if len(backend_calls) == 1 else "reflection"
        return _completion(content)

    agent.model_backend.arun = mock_arun

    await agent.astep("solve the task")
    response = await agent.astep(
        "reflect on the completed task", disable_tools=True
    )

    assert response.msgs[0].content == "reflection"
    assert backend_calls[0][2]
    assert backend_calls[1][2] is None
    assert any(
        message.get("content") == "task answer"
        for message in backend_calls[1][0]
    )
    assert "dangerous_write" in agent.tool_dict
    assert executed == []


@pytest.mark.asyncio
async def test_disabled_step_rejects_provider_emitted_tool_call():
    executed = []

    def dangerous_write(path: str) -> str:
        executed.append(path)
        return "written"

    agent = ChatAgent(
        system_message="You are a test agent.",
        model=StubModel(model_type=ModelType.STUB),
        tools=[dangerous_write],
    )
    forbidden_call = ChatCompletionMessageFunctionToolCall(
        id="forbidden-call",
        function=Function(
            arguments='{"path":"should-not-exist.txt"}',
            name="dangerous_write",
        ),
        type="function",
    )

    async def mock_arun(messages, response_format=None, tools=None):
        assert tools is None
        return _completion(None, tool_calls=[forbidden_call])

    agent.model_backend.arun = mock_arun

    response = await agent.astep(
        "post-task reflection", disable_tools=True
    )

    assert response.terminated is True
    assert response.info["termination_reasons"] == ["tool_calls_disabled"]
    assert response.info["tool_calls"] == []
    assert executed == []
    assert "dangerous_write" in agent.tool_dict
