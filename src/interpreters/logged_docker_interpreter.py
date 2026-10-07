"""
Custom Docker interpreter with network access and code logging.

This module extends CAMEL's DockerInterpreter to provide:
1. Network access (bridge mode) for container internet connectivity
2. Local file logging of all executed code with metadata
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Dict, Optional

from camel.interpreters import DockerInterpreter
from camel.interpreters.interpreter_error import InterpreterError
from camel.logger import get_logger
from camel.utils import is_docker_running

logger = get_logger(__name__)


class LoggedDockerInterpreter(DockerInterpreter):
    """A Docker interpreter with network access and execution logging.

    This interpreter extends the base DockerInterpreter to:
    1. Enable network access via Docker's bridge network mode
    2. Log all executed code to local files with timestamps and metadata
    3. Pre-install useful Python packages for common tasks
    4. Enforce execution timeout to prevent runaway processes

    Args:
        require_confirm (bool): If True, prompt user before running code.
            Defaults to False for automated benchmark execution.
        print_stdout (bool): If True, print stdout of executed code.
            Defaults to False.
        print_stderr (bool): If True, print stderr of executed code.
            Defaults to True.
        log_dir (str | Path): Directory to store execution logs.
            Defaults to "execution_logs".
        network_mode (str): Docker network mode. Defaults to "bridge"
            for internet access.
        preinstall_packages (list[str] | None): Python packages to pre-install
            when container starts. Defaults to common data science packages.
        execution_timeout (int): Maximum execution time in seconds.
            Defaults to 30. Set to 0 to disable timeout.
    """

    # Default packages to pre-install in the sandbox
    DEFAULT_PACKAGES: ClassVar[list[str]] = [
        "markitdown",      # PDF/document conversion
        "PyPDF2",          # PDF reading
        "openpyxl",        # Excel files
        "pandas",          # Data manipulation
        "requests",        # HTTP requests
        "beautifulsoup4",  # HTML parsing
        "lxml",            # XML/HTML parsing
        "python-docx",     # Word documents
        "Pillow",          # Image processing
        "numpy",           # Numerical computing
    ]

    _CODE_EXTENSION_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "py",
        "py": "py",
        "py3": "py",
        "python3": "py",
        "bash": "sh",
        "shell": "sh",
        "sh": "sh",
        "r": "R",
        "R": "R",
        "node": "js",
        "js": "js",
        "javascript": "js",
        "command": "sh",
    }

    def __init__(
        self,
        require_confirm: bool = False,
        print_stdout: bool = False,
        print_stderr: bool = True,
        log_dir: str | Path = "execution_logs",
        network_mode: str = "bridge",
        preinstall_packages: list[str] | None = None,
        execution_timeout: int = 30,
    ) -> None:
        super().__init__(
            require_confirm=require_confirm,
            print_stdout=print_stdout,
            print_stderr=print_stderr,
        )
        self.log_dir = Path(log_dir)
        self.network_mode = network_mode
        self.execution_timeout = execution_timeout
        self._execution_count = 0
        self._packages_installed = False

        # Use default packages if none specified
        if preinstall_packages is None:
            self.preinstall_packages = self.DEFAULT_PACKAGES.copy()
        else:
            self.preinstall_packages = preinstall_packages

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LoggedDockerInterpreter initialized with log_dir={self.log_dir}, timeout={execution_timeout}s")

    def _initialize_if_needed(self) -> None:
        """Initialize Docker container with network access.

        Overrides parent to add network_mode parameter for internet connectivity
        and pre-install useful packages.
        """
        if self._container is not None:
            return

        if not is_docker_running():
            raise InterpreterError(
                "Docker daemon is not running. Please install/start docker "
                "and try again."
            )

        import docker

        client = docker.from_env()

        # Find Dockerfile path from installed CAMEL package
        import camel.interpreters

        dockerfile_path = Path(camel.interpreters.__file__).parent / "docker"

        image_tag = "camel-interpreter:latest"
        try:
            client.images.get(image_tag)
        except docker.errors.ImageNotFound:
            logger.info("Building custom interpreter image...")
            client.images.build(
                path=str(dockerfile_path),
                tag=image_tag,
                rm=True,
            )

        # Create container with network_mode for internet access
        self._container = client.containers.run(
            image_tag,
            detach=True,
            name=f"camel-interpreter-{uuid.uuid4()}",
            command="tail -f /dev/null",
            network_mode=self.network_mode,
        )
        logger.info(f"Container started with network_mode={self.network_mode}")

        # Pre-install packages if configured
        if self.preinstall_packages and not self._packages_installed:
            self._preinstall_packages()

    def _preinstall_packages(self) -> None:
        """Pre-install Python packages in the container."""
        if not self.preinstall_packages:
            return

        packages_str = " ".join(self.preinstall_packages)
        logger.info(f"Pre-installing packages: {packages_str}")

        try:
            # Install packages quietly to avoid spam
            result = self._container.exec_run(
                f"pip install --quiet {packages_str}",
                user="devuser",
            )
            if result.exit_code == 0:
                logger.info("Package pre-installation completed successfully")
            else:
                logger.warning(
                    f"Package installation had issues: {result.output.decode()}"
                )
            self._packages_installed = True
        except Exception as e:
            logger.warning(f"Failed to pre-install packages: {e}")

    def _run_file_in_container(
        self,
        file: Path,
        code_type: str,
    ) -> str:
        """Execute a file in the container with timeout.

        Overrides parent to add execution timeout using the `timeout` command.
        """
        from colorama import Fore

        code_type = self._check_code_type(code_type)

        # Get the base command from parent's mapping
        base_cmd = self._CODE_EXECUTE_CMD_MAPPING[code_type].format(
            file_name=file.as_posix()
        )

        # Wrap with timeout if configured
        if self.execution_timeout > 0:
            cmd = f"timeout {self.execution_timeout} {base_cmd}"
        else:
            cmd = base_cmd

        if self._container is None:
            raise InterpreterError(
                "Container is not initialized. Try running the code again."
            )

        exec_result = self._container.exec_run(
            cmd,
            demux=True,
            user="devuser",
        )
        exit_code = exec_result.exit_code
        stdout, stderr = exec_result.output

        # Check for timeout (exit code 124 from timeout command)
        if exit_code == 124:
            timeout_msg = f"TIMEOUT: Code execution exceeded {self.execution_timeout} seconds and was terminated."
            logger.warning(timeout_msg)
            return timeout_msg

        if self.print_stdout and stdout:
            print("======stdout======")
            print(Fore.GREEN + stdout.decode() + Fore.RESET)
            print("==================")
        if self.print_stderr and stderr:
            print("======stderr======")
            print(Fore.RED + stderr.decode() + Fore.RESET)
            print("==================")

        result = f"{stdout.decode()}" if stdout else ""
        result += f"(stderr: {stderr.decode()})" if stderr else ""
        return result

    def run(
        self,
        code: str,
        code_type: str = "python",
    ) -> str:
        """Execute code and log it to local files.

        Args:
            code (str): The code string to execute.
            code_type (str): The type of code (e.g., 'python', 'bash').

        Returns:
            str: The execution result (stdout + stderr).
        """
        # Generate execution ID and timestamp
        execution_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._execution_count:04d}"
        )
        self._execution_count += 1
        timestamp = datetime.now().isoformat()

        # Log the code before execution
        self._log_execution(
            execution_id=execution_id,
            timestamp=timestamp,
            code=code,
            code_type=code_type,
            status="started",
            result=None,
        )

        try:
            # Execute code using parent implementation
            result = super().run(code, code_type)

            # Log successful completion
            self._log_execution(
                execution_id=execution_id,
                timestamp=timestamp,
                code=code,
                code_type=code_type,
                status="success",
                result=result,
            )

            return result

        except Exception as e:
            # Log failure
            self._log_execution(
                execution_id=execution_id,
                timestamp=timestamp,
                code=code,
                code_type=code_type,
                status="error",
                result=str(e),
            )
            raise

    def execute_command(self, command: str) -> str:
        """Execute a command and log it.

        Args:
            command (str): The command to execute.

        Returns:
            str: The command output.
        """
        execution_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._execution_count:04d}"
        )
        self._execution_count += 1
        timestamp = datetime.now().isoformat()

        self._log_execution(
            execution_id=execution_id,
            timestamp=timestamp,
            code=command,
            code_type="command",
            status="started",
            result=None,
        )

        try:
            result = super().execute_command(command)

            self._log_execution(
                execution_id=execution_id,
                timestamp=timestamp,
                code=command,
                code_type="command",
                status="success",
                result=result,
            )

            return result

        except Exception as e:
            self._log_execution(
                execution_id=execution_id,
                timestamp=timestamp,
                code=command,
                code_type="command",
                status="error",
                result=str(e),
            )
            raise

    def _log_execution(
        self,
        execution_id: str,
        timestamp: str,
        code: str,
        code_type: str,
        status: str,
        result: Optional[str],
    ) -> None:
        """Log execution details to local files.

        Creates two files per execution:
        1. {execution_id}_meta.json - Metadata (timestamp, type, status, result)
        2. {execution_id}_code.{ext} - The actual code

        Args:
            execution_id: Unique identifier for this execution
            timestamp: ISO format timestamp
            code: The code/command being executed
            code_type: Type of code (python, bash, command, etc.)
            status: Execution status (started, success, error)
            result: Execution result or error message
        """
        extension = self._CODE_EXTENSION_MAPPING.get(code_type, "txt")

        # Write metadata file
        meta_file = self.log_dir / f"{execution_id}_meta.json"
        metadata = {
            "execution_id": execution_id,
            "timestamp": timestamp,
            "code_type": code_type,
            "status": status,
            "result": result,
            "code_file": f"{execution_id}_code.{extension}",
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Write code file (only on "started" to avoid duplication)
        if status == "started":
            code_file = self.log_dir / f"{execution_id}_code.{extension}"
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)

        logger.debug(f"Logged execution {execution_id} with status {status}")
