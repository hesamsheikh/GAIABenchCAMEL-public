# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
import re
import textwrap
from typing import Optional, Type

from pydantic import BaseModel

from camel.messages import OpenAIMessage


def strip_markdown_json_wrapper(content: str) -> str:
    r"""Strip markdown code block wrappers from JSON content.

    Many models (especially Claude via Bedrock) return JSON wrapped in markdown
    code blocks like ```json {...} ```. This function extracts the raw JSON.

    Args:
        content (str): The content that may contain markdown-wrapped JSON.

    Returns:
        str: The unwrapped JSON string, or the original content if no wrapper
            was found.
    """
    if not content:
        return content

    content = content.strip()

    # Pattern 1: ```json ... ``` (most common)
    json_block_match = re.search(
        r'```json\s*([\s\S]*?)\s*```', content, re.IGNORECASE
    )
    if json_block_match:
        return json_block_match.group(1).strip()

    # Pattern 2: ``` ... ``` (generic code block containing JSON object)
    generic_block_match = re.search(
        r'```\s*({\s*[\s\S]*?})\s*```', content
    )
    if generic_block_match:
        return generic_block_match.group(1).strip()

    # Pattern 3: Content starts and ends with ``` (malformed but recoverable)
    if content.startswith('```') and content.endswith('```'):
        inner = content[3:-3].strip()
        # Remove optional language identifier
        if inner.startswith('json'):
            inner = inner[4:].strip()
        return inner

    # No wrapper found, return original
    return content


def try_modify_message_with_format(
    message: OpenAIMessage,
    response_format: Optional[Type[BaseModel]],
) -> None:
    r"""Modifies the content of the message to include the instruction of using
    the response format.

    The message will not be modified in the following cases:
    - response_format is None
    - message content is not a string
    - message role is assistant

    Args:
        response_format (Optional[Type[BaseModel]]): The Pydantic model class.
        message (OpenAIMessage): The message to be modified.
    """
    if response_format is None:
        return

    if not isinstance(message["content"], str):
        return

    if message["role"] == "assistant":
        return

    json_schema = response_format.model_json_schema()
    updated_prompt = textwrap.dedent(
        f"""\
        {message["content"]}

        Please generate a JSON response adhering to the following JSON schema:
        {json_schema}
        Make sure the JSON response is valid and matches the EXACT structure defined in the schema. Your result should ONLY be a valid json object, WITHOUT ANY OTHER TEXT OR COMMENTS.
        """  # noqa: E501
    )
    message["content"] = updated_prompt
