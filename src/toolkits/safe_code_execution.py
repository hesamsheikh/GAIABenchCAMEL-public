"""
Safe code execution toolkit with Docker isolation and logging.

This toolkit provides code execution capabilities with:
1. Docker-based sandboxing for security
2. Network access for web-enabled tasks
3. Comprehensive execution logging
"""

from pathlib import Path
from typing import List, Optional, Union

from camel.logger import get_logger
from camel.toolkits import FunctionTool
from camel.toolkits.base import BaseToolkit

from src.interpreters import LoggedDockerInterpreter

logger = get_logger(__name__)


class SafeCodeExecutionToolkit(BaseToolkit):
    """A toolkit for safe code execution with Docker isolation and logging.

    This toolkit wraps the LoggedDockerInterpreter to provide:
    1. Docker-based sandboxed code execution
    2. Network connectivity for web access
    3. Local logging of all executed code
    4. Execution timeout to prevent runaway processes

    Args:
        verbose (bool): Whether to print execution output. Defaults to False.
        log_dir (str | Path): Directory for execution logs.
            Defaults to "execution_logs".
        network_mode (str): Docker network mode. Defaults to "bridge".
        timeout (Optional[float]): Timeout for toolkit operations.
            Defaults to None.
        execution_timeout (int): Maximum code execution time in seconds.
            Defaults to 30. Set to 0 to disable.

    Example:
        >>> toolkit = SafeCodeExecutionToolkit(verbose=True)
        >>> tools = toolkit.get_tools()
        >>> # Use with ChatAgent
        >>> agent = ChatAgent(task_prompt, model, tools=tools)
    """

    def __init__(
        self,
        verbose: bool = False,
        log_dir: Union[str, Path] = "execution_logs",
        network_mode: str = "bridge",
        timeout: Optional[float] = None,
        execution_timeout: int = 30,
    ) -> None:
        super().__init__(timeout=timeout)
        self.verbose = verbose
        self.log_dir = Path(log_dir)

        # Initialize the custom interpreter
        self.interpreter = LoggedDockerInterpreter(
            require_confirm=False,
            print_stdout=verbose,
            print_stderr=verbose,
            log_dir=log_dir,
            network_mode=network_mode,
            execution_timeout=execution_timeout,
        )

        logger.info(
            f"SafeCodeExecutionToolkit initialized: "
            f"log_dir={log_dir}, network_mode={network_mode}"
        )

    def execute_code(self, code: str, code_type: str = "python") -> str:
        """Execute code in a Docker container.

        The code runs in an isolated Docker container with network access.
        All executions are logged to the configured log directory.

        Args:
            code (str): The code to execute.
            code_type (str): The programming language. Supported types:
                - "python", "py", "py3", "python3" -> Python
                - "bash", "shell", "sh" -> Bash
                - "r", "R" -> R
                - "node", "js", "javascript" -> Node.js
                Defaults to "python".

        Returns:
            str: Formatted string with the code and execution results.

        Example:
            >>> result = toolkit.execute_code("print('Hello, World!')")
            >>> print(result)
            Executed the code below:
            ```python
            print('Hello, World!')
            ```
            > Executed Results:
            Hello, World!
        """
        output = self.interpreter.run(code, code_type)
        content = (
            f"Executed the code below:\n```{code_type}\n{code}\n```\n"
            f"> Executed Results:\n{output}"
        )
        if self.verbose:
            print(content)
        return content

    def execute_command(self, command: str) -> str:
        """Execute a shell command in the Docker container.

        Useful for installing dependencies or running system commands.

        Args:
            command (str): The shell command to execute.

        Returns:
            str: Formatted string with the command and execution results.

        Example:
            >>> result = toolkit.execute_command("pip install requests")
            >>> print(result)
            Executed the command below:
            ```sh
            pip install requests
            ```
            > Executed Results:
            Successfully installed requests-2.31.0
        """
        output = self.interpreter.execute_command(command)
        content = (
            f"Executed the command below:\n```sh\n{command}\n```\n"
            f"> Executed Results:\n{output}"
        )
        if self.verbose:
            print(content)
        return content

    def get_tools(self) -> List[FunctionTool]:
        """Get the list of tools provided by this toolkit.

        Returns:
            List[FunctionTool]: List containing execute_code and
                execute_command tools.
        """
        return [
            FunctionTool(self.execute_code),
            FunctionTool(self.execute_command),
        ]

    def cleanup(self) -> None:
        """Clean up Docker resources.

        Call this method when done with the toolkit to stop and remove
        the Docker container.
        """
        self.interpreter.cleanup()
        logger.info("SafeCodeExecutionToolkit cleanup complete")
