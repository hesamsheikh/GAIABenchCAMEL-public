"""ChatAgent with real-time tool call logging and improved JSON parsing."""

import logging
import re
from typing import Type

from pydantic import BaseModel, ValidationError

from camel.agents import ChatAgent
from camel.agents._utils import ToolCallingRecord
from camel.messages import BaseMessage

logger = logging.getLogger(__name__)


def strip_markdown_code_blocks(content: str) -> str:
    """Strip markdown code blocks from content to extract raw JSON.

    Handles formats like:
    - ```json\n{...}\n```
    - ```\n{...}\n```
    - Raw JSON (no change needed)
    """
    # Pattern to match ```json ... ``` or ``` ... ```
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
    match = re.match(pattern, content.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


class GAIAChatAgent(ChatAgent):
    """A ChatAgent customized for GAIA benchmark with tool call logging.

    Also fixes CAMEL's JSON parsing to handle markdown-wrapped responses.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the agent with tool call tracking."""
        super().__init__(*args, **kwargs)
        self._tool_call_count = 0

    def reset(self, *args, **kwargs):
        """Reset the agent state including tool call counter."""
        super().reset(*args, **kwargs)
        self._tool_call_count = 0

    def _try_format_message(
        self, message: BaseMessage, response_format: Type[BaseModel]
    ) -> bool:
        """Try to format the message, stripping markdown code blocks if needed.

        This fixes a CAMEL issue where models return JSON wrapped in markdown
        code blocks (```json ... ```) which causes validation to fail.

        Returns:
            bool: Whether the message is formatted successfully.
        """
        if message.parsed:
            return True

        content = message.content

        # First try raw content (standard CAMEL behavior)
        try:
            message.parsed = response_format.model_validate_json(content)
            return True
        except ValidationError:
            pass

        # Try stripping markdown code blocks
        stripped_content = strip_markdown_code_blocks(content)
        if stripped_content != content:
            try:
                message.parsed = response_format.model_validate_json(stripped_content)
                return True
            except ValidationError:
                pass

        return False

    def _log_tool_call(self, tool_call_request) -> None:
        """Log a tool call in compact single-line format."""
        self._tool_call_count += 1
        func_name = tool_call_request.tool_name
        args = tool_call_request.args

        # Build a compact single-line log: [N] tool_name(key=val, key=val...)
        args_preview = ", ".join(f"{k}={repr(v)[:30]}" for k, v in list(args.items())[:3])
        if len(args) > 3:
            args_preview += ", ..."
        log_line = f"[{self._tool_call_count}] {func_name}({args_preview})"
        # Truncate entire line if too long
        if len(log_line) > 120:
            log_line = log_line[:117] + "..."
        logger.info(log_line)

    def _execute_tool(self, tool_call_request) -> ToolCallingRecord:
        """Execute a tool and log the call in real-time (sync version)."""
        self._log_tool_call(tool_call_request)
        return super()._execute_tool(tool_call_request)

    async def _aexecute_tool(self, tool_call_request) -> ToolCallingRecord:
        """Execute a tool and log the call in real-time (async version)."""
        self._log_tool_call(tool_call_request)
        return await super()._aexecute_tool(tool_call_request)
