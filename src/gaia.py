import logging
import pandas as pd
import os, json
import copy
import re
import shutil
import string
import time
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Literal
import random
from pydantic import BaseModel, Field
from camel.messages import BaseMessage
from camel.agents import ChatAgent
from camel.retrievers import AutoRetriever
from camel.societies.workforce.workflow_memory_manager import WorkflowMemoryManager
from camel.societies.workforce.utils import WorkflowConfig
import uuid
from camel.embeddings import OpenAIEmbedding
from camel.types import StorageType
from tqdm.std import tqdm

logger = logging.getLogger(__name__)


class MemoryUpdateDecision(BaseModel):
    r"""Post-answer decision to create or update workflow memory."""

    action: Literal["create", "update"] = Field(
        description="Create a new workflow or update one existing workflow."
    )
    target_workflow_filename: Optional[str] = Field(
        default=None,
        description=(
            "Exact existing filename without .md when action is update; "
            "null when action is create."
        ),
    )
    reason: str = Field(
        description="Brief reason based on workflow similarity and reuse."
    )

class GAIARetriever(AutoRetriever):
    r"""Default retriever for the GAIA benchmark.
    This retriever uses AutoRetriever in camel to retrieve the content based on
    the query.
    """

    def retrieve(
        self, query: str, contents: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        r"""Retrieve the content based on the query.

        Args:
            query (str): The query to search for.
            contents (List[str]): The list of contents to search from.
            **kwargs (Any): The keyword arguments to pass to the
                retriever.

        Returns:
            Dict[str, Any]: The retrieved content.
        """
        return self.run_vector_retriever(query, 
                    contents,
                    similarity_threshold=0.3,
                    **kwargs)  # type: ignore[arg-type]

    def reset(self, **kwargs: Any) -> bool:
        r"""Reset the retriever.

        Args:
            **kwargs (Any): The keyword arguments to pass to the
                retriever.

        Returns:
            bool: Whether the reset was successful.
        """
        path = Path(self.vector_storage_local_path or os.getcwd())
        task_id = str(kwargs.get("task_id", uuid.uuid4()))
        retriever_dir = path / task_id
        if not retriever_dir.exists():
            try:
                retriever_dir.mkdir(parents=True)
            except Exception as e:
                logger.error(
                    "Error in creating directory: " + f"{retriever_dir}: {e!s}"
                )
                return False
        self.vector_storage_local_path = str(retriever_dir)
        return True


class GAIABenchmark:
    r"""
    The core class for setting up and running the GAIA benchmark.
    """
    def __init__(
        self,
        data_dir: str,
        save_to: Union[str, Path],
        retriever: Optional[GAIARetriever] = None,
        enable_attachment_retrieval: bool = True,
        save_workflow_memory: bool = False,
        workflow_memory_dir: Optional[str] = None,
        workflow_filename_prefix: Optional[str] = None,
        save_individual_results: bool = True,
        individual_results_dir: Optional[str] = None,
        progress_report_interval: int = 10,
        skip_finished_tasks: bool = True,
        blacklisted_task_ids: Optional[List[str]] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        adaptive_memory_updates: bool = False,
        paired_workflow_memory_dirs: Optional[Dict[str, str]] = None,
        ):
        r"""
        Initialize the GAIA benchmark.

        Args:
            data_dir: Directory containing the GAIA benchmark data.
            save_to: Path to save results to (will be converted to .jsonl).
            retriever: Optional retriever instance. If None, a default one is created.
            enable_attachment_retrieval: Whether attachment-backed tasks may use
                the vector retriever. Set to False for the attachment-free
                benchmark population so no embedding provider is initialized.
            save_workflow_memory: Whether to save workflow memory after each task.
            workflow_memory_dir: Directory to save workflow memories.
                Defaults to "workflow_memories/" if not specified.
            workflow_filename_prefix: Optional prefix added to workflow files.
                Use a model identifier to keep experiment artifacts distinct.
            save_individual_results: Whether to save individual task results as separate JSON files.
                Defaults to True.
            individual_results_dir: Directory to save individual task results.
                Defaults to "task_results/" if not specified.
            progress_report_interval: Report accuracy every N tasks.
                Defaults to 10.
            skip_finished_tasks: Whether to skip tasks that already have results in individual_results_dir.
                Defaults to True.
            blacklisted_task_ids: List of task IDs to skip (tasks that consistently fail).
                Defaults to empty list.
            adaptive_memory_updates: Whether the post-answer memory-writing
                phase may inspect existing workflow candidates and choose to
                create a new workflow or update one. Candidate memories are
                never added to the task-solving context.
        """
        # Set up variables
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self.results: List[Dict[str, Any]] = []

        self.enable_attachment_retrieval = enable_attachment_retrieval
        if not enable_attachment_retrieval:
            self.retriever = None
        elif retriever is None:
            self.retriever = self._setup_retriever()
        else:
            self.retriever = retriever

        # Workflow memory configuration
        self.save_workflow_memory = save_workflow_memory
        self.workflow_memory_dir = workflow_memory_dir or "workflow_memories/"
        self.workflow_filename_prefix = workflow_filename_prefix

        # Individual results configuration
        self.save_individual_results = save_individual_results
        self.individual_results_dir = individual_results_dir or "task_results/"
        if self.save_individual_results:
            Path(self.individual_results_dir).mkdir(parents=True, exist_ok=True)

        # Progress reporting configuration
        self.progress_report_interval = progress_report_interval

        # Skip finished tasks configuration
        self.skip_finished_tasks = skip_finished_tasks

        # Blacklisted task IDs
        self.blacklisted_task_ids = set(blacklisted_task_ids or [])
        self.experiment_metadata = dict(experiment_metadata or {})
        self.adaptive_memory_updates = adaptive_memory_updates
        self.paired_workflow_memory_dirs = dict(
            paired_workflow_memory_dirs or {}
        )
        unsupported_strategies = set(self.paired_workflow_memory_dirs) - {
            "structured",
            "self_organized",
        }
        if unsupported_strategies:
            raise ValueError(
                "Unsupported paired workflow-memory strategies: "
                + ", ".join(sorted(unsupported_strategies))
            )
        self.native_structured_output = bool(
            self.experiment_metadata.get("native_structured_output", True)
        )

        # set up save folder/file
        self.save_to = Path(save_to)
        if self.save_to.suffix != ".jsonl":
            self.save_to = self.save_to.with_suffix(".jsonl")
        if not self.save_to.parent.exists():
            self.save_to.parent.mkdir(parents=True, exist_ok=True)
        self.save_to: str = str(self.save_to)

        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.download()
        self.load()

    _USAGE_KEYS = (
        "prompt_tokens",
        "cached_prompt_tokens",
        "uncached_prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    )

    @classmethod
    def _empty_usage(cls) -> Dict[str, int]:
        return {key: 0 for key in cls._USAGE_KEYS}

    @classmethod
    def _normalize_usage(cls, usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        normalized = cls._empty_usage()
        raw = usage or {}
        for key in cls._USAGE_KEYS:
            normalized[key] = int(raw.get(key) or 0)
        if not normalized["cached_prompt_tokens"]:
            details = raw.get("prompt_tokens_details") or {}
            if isinstance(details, dict):
                normalized["cached_prompt_tokens"] = int(
                    details.get("cached_tokens") or 0
                )
        if not normalized["reasoning_tokens"]:
            details = raw.get("completion_tokens_details") or {}
            if isinstance(details, dict):
                normalized["reasoning_tokens"] = int(
                    details.get("reasoning_tokens") or 0
                )
        normalized["uncached_prompt_tokens"] = max(
            normalized["prompt_tokens"]
            - normalized["cached_prompt_tokens"],
            0,
        )
        return normalized

    @classmethod
    def _add_usage(cls, *usage_items: Optional[Dict[str, Any]]) -> Dict[str, int]:
        total = cls._empty_usage()
        for item in usage_items:
            normalized = cls._normalize_usage(item)
            for key in cls._USAGE_KEYS:
                total[key] += normalized[key]
        return total

    @staticmethod
    def _tool_call_name(tool_call: Any) -> str:
        for attribute in ("tool_name", "func_name", "name"):
            value = getattr(tool_call, attribute, None)
            if value:
                return str(value)
        if isinstance(tool_call, dict):
            for key in ("tool_name", "func_name", "name"):
                if tool_call.get(key):
                    return str(tool_call[key])
        return "unknown"

    @classmethod
    def _phase_efficiency(
        cls,
        response: Optional[Any] = None,
        *,
        usage: Optional[Dict[str, Any]] = None,
        model_calls: int = 0,
        tool_calls: Optional[List[Any]] = None,
        messages: int = 0,
        wall_seconds: float = 0.0,
        metrics_partial: bool = False,
    ) -> Dict[str, Any]:
        if response is not None:
            info = getattr(response, "info", {}) or {}
            usage = info.get("usage", usage)
            model_calls = int(info.get("model_calls", model_calls) or 0)
            tool_calls = info.get("tool_calls", tool_calls or [])
        calls = list(tool_calls or [])
        by_name: Dict[str, int] = {}
        for call in calls:
            name = cls._tool_call_name(call)
            by_name[name] = by_name.get(name, 0) + 1
        return {
            "usage": cls._normalize_usage(usage),
            "model_calls": model_calls,
            "tool_calls": len(calls),
            "tool_calls_by_name": by_name,
            "messages": messages,
            "wall_seconds": round(wall_seconds, 3),
            "metrics_partial": metrics_partial,
        }

    @classmethod
    def _add_phases(cls, *phases: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        present = [phase for phase in phases if phase]
        by_name: Dict[str, int] = {}
        for phase in present:
            for name, count in phase.get("tool_calls_by_name", {}).items():
                by_name[name] = by_name.get(name, 0) + int(count)
        return {
            "usage": cls._add_usage(
                *(phase.get("usage", {}) for phase in present)
            ),
            "model_calls": sum(int(p.get("model_calls", 0)) for p in present),
            "tool_calls": sum(int(p.get("tool_calls", 0)) for p in present),
            "tool_calls_by_name": by_name,
            "messages": sum(int(p.get("messages", 0)) for p in present),
            "wall_seconds": round(
                sum(float(p.get("wall_seconds", 0.0)) for p in present), 3
            ),
            "metrics_partial": any(
                bool(p.get("metrics_partial", False)) for p in present
            ),
        }

    @classmethod
    def _merge_response_metrics(cls, earlier: Any, later: Any) -> Any:
        earlier_info = getattr(earlier, "info", {}) or {}
        later_info = getattr(later, "info", {}) or {}
        later_info["usage"] = cls._add_usage(
            earlier_info.get("usage"), later_info.get("usage")
        )
        later_info["model_calls"] = int(
            earlier_info.get("model_calls", 0) or 0
        ) + int(later_info.get("model_calls", 0) or 0)
        later_info["tool_calls"] = list(
            earlier_info.get("tool_calls", [])
        ) + list(later_info.get("tool_calls", []))
        later.info = later_info
        return later

    def download(self):
        r"""
        Download the GAIA benchmark data.
        """
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading data to {self.data_dir}")
        snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir=self.data_dir,
            local_dir_use_symlinks=True,
        )
        logger.info(f"Download finished.")

    def load(self):
        r"""
        Load the GAIA benchmark data.
        """
        # Define validation and test directories
        valid_dir = self.data_dir / "2023/validation"
        test_dir = self.data_dir / "2023/test"

        if not valid_dir.exists() or not test_dir.exists():
            logger.info("Attempted loading while data not found. "
                        "Downloading data...")
            self.download()

        # Load metadata for both validation and test datasets
        for path, label in zip([valid_dir, test_dir], ["valid", "test"]):
            self.data[label] = []
            metadata_file = path / "metadata.parquet"
            df = pd.read_parquet(metadata_file)
            for _, row in df.iterrows():
                data = row.to_dict()
                if data["task_id"] == "0-0-0-0-0":
                    continue
                # convert level to int (parquet stores as string)
                data["Level"] = int(data["Level"])
                if data["file_name"]:
                    data["file_name"] = path / data["file_name"]
                self.data[label].append(data)
        return self

    async def run(self,
            agent: ChatAgent,
            test_valid:str="test",
            level:Union[int, "all"]=1,
            randomize:bool=False,
            num_samples:Optional[int]=None,
            ):
        r"""
        Run the GAIA benchmark asynchronously.

        This method uses agent.astep() to properly handle async tools like
        Crawl4AIToolkit.scrape.

        Args:
            agent: The agent to run the benchmark.
            test_valid: The dataset to run the benchmark on. "test" or "valid".
            level: The level of the benchmark to run. "all" to run all levels.
            randomize: Whether to randomize the benchmark.
            num_samples: The number of samples to run. None or -1 for all tasks.
        """
        if test_valid not in ["test", "valid"]:
            raise ValueError(f"test_valid must be 'test' or 'valid', got {test_valid}")
        if level not in [1,2,3, "all"]:
            raise ValueError(f"level must be 1, 2, 3, or 'all', got {level}")
        if num_samples is not None and num_samples != -1 and num_samples < 1:
            raise ValueError(f"num_samples must be greater than 0, -1, or None, got {num_samples}")

        if level == "all" or level is None:
            levels = [1,2,3]
        else:
            levels = [level]

        data = [data for data in self.data[test_valid] if data["Level"] in levels]

        # Shuffle and subset data if necessary
        if randomize:
            random.shuffle(data)
        if num_samples is not None and num_samples > 0:
            data = data[:num_samples]

        # Filter out blacklisted tasks
        if self.blacklisted_task_ids:
            original_count = len(data)
            data = [task for task in data if task['task_id'] not in self.blacklisted_task_ids]
            blacklisted_count = original_count - len(data)
            if blacklisted_count > 0:
                logger.info(f"Skipping {blacklisted_count} blacklisted tasks")

        # Filter out already finished tasks if skip_finished_tasks is enabled
        if self.skip_finished_tasks:
            original_count = len(data)
            data = [task for task in data if not self._is_task_finished(task['task_id'])]
            skipped_count = original_count - len(data)
            if skipped_count > 0:
                logger.info(f"Skipping {skipped_count} already finished tasks (results found in {self.individual_results_dir})")

        logger.info(f"Running benchmark on {test_valid} dataset "
                    f"with levels {levels} and {len(data)} samples")
        labels_available = test_valid == "valid"

        with open(self.save_to, "w") as f:
            for idx, task in enumerate(tqdm(data, desc="Running benchmark")):
                task_started_at = time.monotonic()
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Task {idx+1}/{len(data)}: {task['task_id']}")
                    logger.info(f"Level: {task['Level']}")
                    logger.info(f"Question: {task['Question'][:100]}...")
                    logger.info(f"{'='*60}")

                    # prepare task and prompt
                    if not self._prepare_task(task):
                        logger.warning(f"Skipping task {task['task_id']} - preparation failed")
                        continue
                    user_message = self._prepare_user_message(task)

                    # Set GAIA task ID if the agent supports it (e.g., WorkforceWrapper)
                    if hasattr(agent, 'set_gaia_task_id'):
                        agent.set_gaia_task_id(task['task_id'])

                    logger.info("Agent processing...")
                    solve_started_at = time.monotonic()
                    response = await agent.astep(user_message)
                    response_recovery_count = 0
                    if not response.msgs:
                        response_recovery_count = 1
                        prior_response = response
                        logger.warning(
                            "Model returned no final message after tool use; "
                            "requesting one final response"
                        )
                        recovery_message = BaseMessage.make_user_message(
                            role_name="User",
                            content=(
                                "Continue from the completed task and tool "
                                "results. Return your final answer now in the "
                                "required <final_answer> format. Do not restart "
                                "the research unless essential."
                            ),
                        )
                        response = self._merge_response_metrics(
                            prior_response,
                            await agent.astep(recovery_message),
                        )
                    if not response.msgs:
                        raise RuntimeError(
                            "Model produced no final message after one "
                            "recovery attempt"
                        )
                    solve_seconds = time.monotonic() - solve_started_at
                    logger.info("Agent finished processing")

                    result_data = self._process_result(
                        agent,
                        task,
                        response,
                        f,
                        timing_data={"solve_seconds": solve_seconds},
                        labels_available=labels_available,
                    )
                    result_data["response_recovery_count"] = (
                        response_recovery_count
                    )

                    # Save workflow memory if enabled
                    memory_build_seconds = 0.0
                    if self.save_workflow_memory:
                        memory_started_at = time.monotonic()
                        if self.paired_workflow_memory_dirs:
                            memory_build_efficiency = (
                                await self._build_paired_workflow_memories(
                                    agent, task, result_data
                                )
                            )
                        else:
                            reflection_efficiency = (
                                await self._reflect_on_result(
                                    agent, result_data
                                )
                            )
                            save_efficiency = await self._save_workflow_memory(
                                agent, task, result_data=result_data
                            )
                            memory_build_efficiency = self._add_phases(
                                reflection_efficiency, save_efficiency
                            )
                        memory_build_seconds = (
                            time.monotonic() - memory_started_at
                        )
                        memory_build_efficiency["wall_seconds"] = round(
                            memory_build_seconds, 3
                        )
                        result_data["efficiency"]["memory_build"] = (
                            memory_build_efficiency
                        )
                        result_data["efficiency"]["overall"] = (
                            self._add_phases(
                                result_data["efficiency"]["solve"],
                                memory_build_efficiency,
                            )
                        )

                    result_data["timing"].update({
                        "memory_build_seconds": round(
                            memory_build_seconds, 3
                        ),
                        "total_seconds": round(
                            time.monotonic() - task_started_at, 3
                        ),
                    })
                    self._save_individual_result(result_data)
                    logger.info("Timing: %s", result_data["timing"])

                    logger.info(f"Task {task['task_id']} completed")

                    # Report progress at configured interval
                    if self.progress_report_interval > 0 and (idx + 1) % self.progress_report_interval == 0:
                        self._log_progress_statistics(idx + 1, len(data))

                except Exception as e:
                    logger.error(f"Task {task['task_id']} failed: {e}")
                    self._handle_error(
                        task,
                        e,
                        f,
                        timing_data={
                            "total_seconds": round(
                                time.monotonic() - task_started_at, 3
                            )
                        },
                        labels_available=labels_available,
                    )
                finally:
                    agent.reset()

        # Rewrite from the in-memory rows so timing recorded after workflow
        # generation is included while preserving crash-safe per-task files.
        with open(self.save_to, "w", encoding="utf-8") as f:
            for result_data in self.results:
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

        # Calculate and log final statistics
        self._log_final_statistics()
    
    def _is_task_finished(self, task_id: str) -> bool:
        r"""Check if a task has already been completed.

        Args:
            task_id: The task ID to check.

        Returns:
            True if a result file exists for this task, False otherwise.
        """
        result_file = Path(self.individual_results_dir) / f"{task_id}.json"
        return result_file.exists()

    def _log_progress_statistics(self, tasks_completed: int, total_tasks: int) -> None:
        r"""Log progress statistics every N tasks.

        Args:
            tasks_completed: Number of tasks completed so far.
            total_tasks: Total number of tasks to complete.
        """
        if not self.results:
            return

        labeled_results = [
            result for result in self.results
            if result.get("score") in {0, 1}
        ]
        if not labeled_results:
            failed_tasks = sum(
                1 for result in self.results if result.get("error")
            )
            logger.info(
                "PROGRESS UPDATE: %d/%d tasks completed; labels hidden; "
                "%d execution failures",
                tasks_completed,
                total_tasks,
                failed_tasks,
            )
            return

        correct_tasks = sum(
            1 for result in labeled_results if result.get("score") == 1
        )
        incorrect_tasks = len(labeled_results) - correct_tasks
        accuracy = correct_tasks / len(labeled_results) * 100

        logger.info(f"\n{'='*60}")
        logger.info(f"PROGRESS UPDATE: {tasks_completed}/{total_tasks} tasks completed")
        logger.info(f"{'='*60}")
        logger.info(f"Correct: {correct_tasks}/{len(self.results)} ({accuracy:.2f}%)")
        logger.info(f"Incorrect: {incorrect_tasks}/{len(self.results)}")
        logger.info(f"{'='*60}\n")

    def _log_final_statistics(self) -> None:
        r"""Calculate and log final benchmark statistics."""
        if not self.results:
            logger.info("No results to calculate statistics from.")
            return

        labeled_results = [
            result for result in self.results
            if result.get("score") in {0, 1}
        ]
        if not labeled_results:
            failed_tasks = sum(
                1 for result in self.results if result.get("error")
            )
            logger.info("\n%s", "=" * 60)
            logger.info("FINAL BENCHMARK RESULTS (GROUND TRUTH HIDDEN)")
            logger.info("%s", "=" * 60)
            logger.info("Total Tasks: %d", len(self.results))
            logger.info("Completed without execution error: %d", len(self.results) - failed_tasks)
            logger.info("Execution failures: %d", failed_tasks)
            logger.info("Accuracy: unavailable")
            logger.info("%s", "=" * 60)
            return

        total_tasks = len(labeled_results)
        correct_tasks = sum(
            1 for result in labeled_results if result.get("score") == 1
        )
        incorrect_tasks = total_tasks - correct_tasks
        accuracy = (correct_tasks / total_tasks * 100) if total_tasks > 0 else 0

        # Calculate statistics by level
        level_stats = {}
        for result in self.results:
            level = result.get("level", "unknown")
            if level not in level_stats:
                level_stats[level] = {"total": 0, "correct": 0}
            level_stats[level]["total"] += 1
            if result.get("score", 0) == 1:
                level_stats[level]["correct"] += 1

        # Log overall statistics
        logger.info(f"\n{'='*60}")
        logger.info("FINAL BENCHMARK RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tasks: {total_tasks}")
        logger.info(f"Correct: {correct_tasks}")
        logger.info(f"Incorrect: {incorrect_tasks}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"{'='*60}")

        # Log statistics by level
        if level_stats:
            logger.info("RESULTS BY LEVEL:")
            for level in sorted(level_stats.keys()):
                stats = level_stats[level]
                level_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                logger.info(f"  Level {level}: {stats['correct']}/{stats['total']} ({level_accuracy:.2f}%)")
            logger.info(f"{'='*60}")

    def _prepare_task(self, task) -> bool:
        r"""Prepare the task by validating and enriching its data.
        Args:
            task: The task to prepare.
        Returns:
            True if the task was prepared successfully, False otherwise.
        """
        # Check annotator metadata for multimodal requirements
        multimodal_reqs = self._get_multimodal_requirements(task)
        if multimodal_reqs:
            logger.warning(
                f"Skipping task {task['task_id']} - requires multimodal "
                f"capabilities: {', '.join(multimodal_reqs)}"
            )
            return False

        if task["file_name"]:
            if not self.enable_attachment_retrieval:
                logger.warning(
                    "Skipping attachment-backed task %s because attachment "
                    "retrieval is disabled",
                    task["task_id"],
                )
                return False
            file_path = Path(task["file_name"])
            if not file_path.exists():
                logger.info(
                    f"Skipping task because file not found: {file_path}"
                )
                return False
            if file_path.suffix in [".pdf", ".docx", ".doc", ".txt"]:
                # set up retriever and get retrieved content
                if not self.retriever.reset(task_id=task["task_id"]):
                    return False
                retrieved_info = self.retriever.retrieve(
                    query=task["Question"], contents=[str(task["file_name"])]
                )
                # Handle both dict format (return_detailed_info=True) and
                # string format (return_detailed_info=False, the default)
                retrieved_context = retrieved_info.get("Retrieved Context", [])
                retrieved_content = []
                for item in retrieved_context:
                    if isinstance(item, dict):
                        retrieved_content.append(item.get("text", ""))
                    else:
                        # Item is already a string
                        retrieved_content.append(str(item))
                if retrieved_content:
                    task["retrieved_content"] = "\n".join(retrieved_content)
            else:
                logger.info(
                    f"Skipping task due to unsupported file "
                    f"format: {file_path.suffix}"
                )
                return False
        return True

    def _get_multimodal_requirements(self, task) -> List[str]:
        r"""Check if task requires multimodal capabilities (image/audio/video).

        Uses the 'Tools' field from Annotator Metadata to detect tasks that need
        image recognition, video processing, audio processing, or OCR.

        Returns:
            List[str]: List of detected multimodal keywords, empty if none found.
        """
        metadata = task.get("Annotator Metadata", {})
        if isinstance(metadata, str):
            import ast
            try:
                metadata = ast.literal_eval(metadata)
            except (ValueError, SyntaxError):
                metadata = {}

        tools_str = metadata.get("Tools", "").lower() if isinstance(metadata, dict) else ""

        multimodal_keywords = [
            "image recognition", "ocr", "video recognition", "video parsing",
            "video processing", "audio processing", "speech-to-text",
            "color recognition"
        ]

        return [kw for kw in multimodal_keywords if kw in tools_str]

    def _prepare_user_message(self, task) -> BaseMessage:
        r"""Create a user message from a task."""
        content = task["Question"]
        if "retrieved_content" in task:
            content += "\n" + task["retrieved_content"]

        return BaseMessage.make_user_message(
            role_name="User",
            content=content,
        )
    
    def _setup_retriever(self) -> GAIARetriever:
        r"""Setup the retriever."""
        print("Setting up retriever...")
        retriever = GAIARetriever(
            vector_storage_local_path="execution/retriever/",
            storage_type=StorageType.QDRANT,
            embedding_model=OpenAIEmbedding(),
        )
        print("Retriever created.")
        return retriever

    @staticmethod
    def _clone_memory_writer_agent(
        source_agent: ChatAgent,
        strategy: str,
    ) -> ChatAgent:
        r"""Clone only the solved conversation into an isolated no-tool agent."""
        context_creator = source_agent.memory.get_context_creator()
        writer = ChatAgent(
            system_message=None,
            model=source_agent.model_backend.models,
            tools=[],
            agent_id=f"{source_agent.agent_id}_{strategy}_memory_writer",
            max_iteration=source_agent.max_iteration,
            token_limit=getattr(context_creator, "token_limit", None),
            summarize_threshold=None,
            step_timeout=getattr(source_agent, "step_timeout", None),
            tool_execution_timeout=getattr(
                source_agent, "tool_execution_timeout", None
            ),
        )
        for context_record in source_agent.memory.retrieve():
            writer.memory.write_record(
                copy.deepcopy(context_record.memory_record)
            )
        writer._system_message = copy.deepcopy(
            getattr(source_agent, "_system_message", None)
        )
        writer._original_system_message = copy.deepcopy(
            getattr(source_agent, "_original_system_message", None)
        )
        return writer

    async def _build_paired_workflow_memories(
        self,
        agent: ChatAgent,
        task: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        r"""Write independent banks from the same fixed solved trajectory."""
        source_result = copy.deepcopy(result_data)
        branch_records: Dict[str, Any] = {}
        combined_efficiency = self._phase_efficiency()

        for strategy, memory_dir in self.paired_workflow_memory_dirs.items():
            branch_started_at = time.monotonic()
            branch_agent = self._clone_memory_writer_agent(agent, strategy)
            branch_result = copy.deepcopy(source_result)
            try:
                reflection_efficiency = await self._reflect_on_result(
                    branch_agent,
                    branch_result,
                    memory_strategy=strategy,
                )
                save_efficiency = await self._save_workflow_memory(
                    branch_agent,
                    task,
                    result_data=branch_result,
                    workflow_memory_dir=memory_dir,
                    memory_strategy=strategy,
                )
                branch_efficiency = self._add_phases(
                    reflection_efficiency, save_efficiency
                )
                branch_efficiency["wall_seconds"] = round(
                    time.monotonic() - branch_started_at, 3
                )
                branch_records[strategy] = {
                    "workflow_memory_dir": memory_dir,
                    "memory_operation": branch_result.get(
                        "memory_operation"
                    ),
                    "efficiency": branch_efficiency,
                }
                combined_efficiency = self._add_phases(
                    combined_efficiency, branch_efficiency
                )
            finally:
                branch_agent.reset()

        result_data["memory_branches"] = branch_records
        result_data["efficiency"]["memory_build_by_strategy"] = {
            strategy: record["efficiency"]
            for strategy, record in branch_records.items()
        }
        return combined_efficiency

    async def _save_workflow_memory(
        self,
        agent: ChatAgent,
        task: Dict[str, Any],
        result_data: Optional[Dict[str, Any]] = None,
        workflow_memory_dir: Optional[str] = None,
        memory_strategy: str = "structured",
    ) -> Dict[str, Any]:
        r"""Save workflow memory for the agent after completing a task.

        Uses a two-pass approach:
        1. Generate structured summary to extract agent_title
        2. Save to file named by agent_title

        Args:
            agent: The agent whose workflow memory to save.
            task: The task that was just completed.
            result_data: Stored answer and scoring metadata. Hidden-label runs
                use it to enforce outcome-blind workflow-summary constraints.
        """
        from camel.societies.workforce.structured_output_handler import (
            StructuredOutputHandler,
        )
        from camel.utils.context_utils import (
            ContextUtility,
            SelfOrganizedWorkflowSummary,
            WorkflowSummary,
        )

        active_memory_dir = workflow_memory_dir or self.workflow_memory_dir
        if memory_strategy == "structured":
            summary_schema = WorkflowSummary
        elif memory_strategy == "self_organized":
            summary_schema = SelfOrganizedWorkflowSummary
        else:
            raise ValueError(f"Unsupported memory strategy: {memory_strategy}")

        self._last_memory_selection_efficiency = self._phase_efficiency()
        try:
            # Create workflow config with custom directory
            config = WorkflowConfig(
                workflow_folder_name=active_memory_dir,
            )

            # Create workflow memory manager
            manager = WorkflowMemoryManager(
                worker=agent,
                description=f"GAIA Task: {task['task_id']}",
                role_identifier="gaia_agent",
                config=config,
            )

            memory_decision = MemoryUpdateDecision(
                action="create",
                target_workflow_filename=None,
                reason="Adaptive updates disabled or bank empty.",
            )
            available_workflow_count = 0
            if self.adaptive_memory_updates:
                memory_decision, available_workflow_count = (
                    await self._agent_select_memory_operation(
                        agent, manager, workflow_memory_dir=active_memory_dir
                    )
                )

            # PASS 1: Generate structured summary
            if memory_strategy == "structured":
                summary_prompt = manager._prepare_workflow_prompt()
            else:
                summary_prompt = (
                    SelfOrganizedWorkflowSummary.get_instruction_prompt()
                )
                if manager._loaded_workflow_contents:
                    summary_prompt += (
                        "\n\nAn existing self-organized memory was selected "
                        "for updating. Retain whatever earlier content you "
                        "judge useful while incorporating the completed "
                        "experience:\n\n--- Existing Memory ---"
                    )
                    summary_prompt += manager._format_workflow_list(
                        manager._loaded_workflow_contents
                    )
                    summary_prompt += "\n\n--- End Existing Memory ---"
                summary_prompt = (
                    StructuredOutputHandler.generate_structured_prompt(
                        base_prompt=summary_prompt,
                        schema=SelfOrganizedWorkflowSummary,
                    )
                )
            if self.adaptive_memory_updates:
                if memory_strategy == "structured":
                    summary_prompt += (
                        "\n\nTHE POST-TASK MEMORY DECISION HAS ALREADY BEEN "
                        "MADE BY THE AGENT. Follow it exactly:\n"
                        f"- operation_mode: {memory_decision.action}\n"
                        "- target_workflow_filename: "
                        f"{memory_decision.target_workflow_filename}\n"
                        "When updating, preserve useful existing steps, tool "
                        "lessons, failure modes, and prior-task coverage while "
                        "incorporating the new experience. The workflow may "
                        "grow longer; do not discard useful prior content "
                        "merely to keep it short."
                    )
                else:
                    summary_prompt += (
                        "\n\nTHE POST-TASK MEMORY DECISION HAS ALREADY BEEN "
                        "MADE. Follow it exactly:\n"
                        f"- operation_mode: {memory_decision.action}\n"
                        "- target_workflow_filename: "
                        f"{memory_decision.target_workflow_filename}\n"
                        "When updating, use your own judgment about what "
                        "earlier and new material is worth preserving."
                    )
            if result_data is not None and result_data.get("score") is None:
                if memory_strategy == "structured":
                    summary_prompt += (
                        "\n\nOUTCOME-BLIND MEMORY REQUIREMENTS:\n"
                        "- No authoritative ground truth or correctness "
                        "feedback exists for this task.\n"
                        "- Describe only workflow steps and tool outcomes that "
                        "are supported by the conversation history.\n"
                        "- Do not state or imply that the submitted answer was "
                        "correct or incorrect.\n"
                        "- Do not present a different post-hoc answer as a "
                        "correction. If self-review raised an alternative, "
                        "label it explicitly as unverified and preserve the "
                        f"submitted answer ({result_data.get('submitted_answer', '')}) "
                        "as the answer actually produced.\n"
                        "- In Failure And Recovery Strategies, include only "
                        "failures and recoveries directly observed in tool "
                        "output or the recorded attempt. If none occurred, "
                        "say that no failure was observed.\n"
                        "- Prefer reusable procedural lessons over claims "
                        "about the task's unknown outcome."
                    )
                else:
                    summary_prompt += (
                        "\n\nOUTCOME-BLIND MEMORY REQUIREMENTS:\n"
                        "- No authoritative ground truth or correctness "
                        "feedback exists for this task.\n"
                        "- Retain only claims supported by the recorded "
                        "attempt.\n"
                        "- Do not state or imply that the submitted answer was "
                        "correct or incorrect.\n"
                        "- Do not present a different post-hoc answer as a "
                        "correction. If an alternative arose, label it "
                        "unverified and preserve the submitted answer "
                        f"({result_data.get('submitted_answer', '')}) as the "
                        "answer actually produced.\n"
                        "- Decide for yourself what is useful to retain for "
                        "similar future tasks."
                    )
            gen_result = await agent.generate_workflow_summary_async(
                summary_prompt=summary_prompt,
                response_format=(
                    summary_schema
                    if getattr(self, "native_structured_output", True)
                    else None
                ),
            )

            if gen_result.get("status") != "success":
                logger.warning(
                    f"Failed to generate workflow summary for task "
                    f"{task['task_id']}: {gen_result.get('status')}"
                )
                return self._add_phases(
                    self._last_memory_selection_efficiency,
                    self._phase_efficiency(metrics_partial=True),
                )

            summary_efficiency = self._phase_efficiency(
                usage=gen_result.get("usage"),
                model_calls=gen_result.get("model_calls", 0),
                messages=2,
            )

            workflow_summary = gen_result.get("structured_summary")

            # If structured parsing failed, try to extract from raw content
            if workflow_summary is None:
                raw_content = gen_result.get("summary_content", "")
                workflow_summary = self._parse_workflow_summary(
                    raw_content, summary_schema=summary_schema
                )

            if workflow_summary is None:
                logger.warning(
                    f"Could not parse workflow summary for task {task['task_id']}"
                )
                return self._add_phases(
                    self._last_memory_selection_efficiency,
                    summary_efficiency,
                )

            if self.adaptive_memory_updates:
                workflow_summary.operation_mode = memory_decision.action
                workflow_summary.target_workflow_filename = (
                    memory_decision.target_workflow_filename
                )

            requested_operation = getattr(
                workflow_summary, "operation_mode", "create"
            )
            requested_target = getattr(
                workflow_summary, "target_workflow_filename", None
            )

            archived_version = None
            if (
                requested_operation == "update"
                and requested_target in manager._loaded_workflow_paths
            ):
                target_path = Path(
                    manager._loaded_workflow_paths[requested_target]
                )
                history_dir = (
                    Path(active_memory_dir)
                    / "_history"
                    / requested_target
                )
                history_dir.mkdir(parents=True, exist_ok=True)
                existing_metadata = manager._extract_existing_workflow_metadata(
                    target_path
                )
                existing_version = (
                    existing_metadata.workflow_version
                    if existing_metadata is not None else 1
                )
                archived_path = history_dir / f"v{existing_version}.md"
                if not archived_path.exists():
                    shutil.copy2(target_path, archived_path)
                archived_version = str(archived_path)

            # PASS 2: Save with agent_title as filename
            agent_title = getattr(workflow_summary, 'agent_title', 'gaia_agent')
            clean_title = ContextUtility.sanitize_workflow_filename(agent_title)
            filename_parts = [task['task_id'], clean_title]
            if self.workflow_filename_prefix:
                filename_parts.insert(0, self.workflow_filename_prefix)
            unique_filename = "_".join(filename_parts)

            # Create context utility pointing to base workflow directory
            # (not role-based subfolder) so file saves as workflows/agent_name.md
            base_context = ContextUtility(
                working_directory=active_memory_dir,
                create_folder=True,
                use_session_subfolder=False,
            )

            # Save using the manager with GAIA task ID
            # Use filename_override to name file by agent_title while preserving
            # original task_title in the file content
            result = await manager.save_workflow_content_async(
                workflow_summary=workflow_summary,
                context_utility=base_context,
                gaia_task_name=task['task_id'],
                filename_override=unique_filename,
            )

            if result.get("status") == "success":
                result_path = Path(result.get("file_path", ""))
                actual_operation = (
                    "update"
                    if requested_operation == "update"
                    and requested_target in manager._loaded_workflow_paths
                    and result_path
                    == Path(manager._loaded_workflow_paths[requested_target])
                    else "create"
                )
                if result_data is not None:
                    result_data["memory_operation"] = {
                        "adaptive": self.adaptive_memory_updates,
                        "available_workflow_count": available_workflow_count,
                        "agent_decision": memory_decision.model_dump(),
                        "requested_operation": requested_operation,
                        "requested_target": requested_target,
                        "actual_operation": actual_operation,
                        "file_path": str(result_path),
                        "archived_previous_version": archived_version,
                    }
                logger.info(
                    "Workflow memory %s for task %s at %s",
                    actual_operation,
                    task["task_id"],
                    result.get("file_path"),
                )
            else:
                logger.warning(
                    f"Failed to save workflow memory for task {task['task_id']}: "
                    f"{result.get('message', result.get('status'))}"
                )
            return self._add_phases(
                self._last_memory_selection_efficiency,
                summary_efficiency,
            )
        except Exception as e:
            logger.error(
                f"Error saving workflow memory for task {task['task_id']}: {e}"
            )
            return self._add_phases(
                self._last_memory_selection_efficiency,
                self._phase_efficiency(metrics_partial=True),
            )

    async def _agent_select_memory_operation(
        self,
        agent: ChatAgent,
        manager: WorkflowMemoryManager,
        workflow_memory_dir: Optional[str] = None,
    ) -> tuple[MemoryUpdateDecision, int]:
        r"""Let the task agent choose create or one exact update target.

        This runs only after the task answer is finalized. The agent receives
        filenames plus compact descriptions for every current workflow. If it
        chooses update, only that workflow's full content is loaded into the
        subsequent workflow-summary prompt.
        """
        from camel.utils.context_utils import ContextUtility

        memory_dir = Path(workflow_memory_dir or self.workflow_memory_dir)
        workflow_files = sorted(memory_dir.glob("*.md"))
        if not workflow_files:
            logger.info(
                "Adaptive memory: bank empty; next operation must create"
            )
            return (
                MemoryUpdateDecision(
                    action="create",
                    target_workflow_filename=None,
                    reason="No existing workflow memories are available.",
                ),
                0,
            )

        context_utility = ContextUtility(
            working_directory=str(memory_dir),
            create_folder=False,
            use_session_subfolder=False,
        )
        workflows_metadata = []
        for file_path in workflow_files:
            metadata = context_utility.extract_workflow_info(str(file_path))
            if metadata:
                workflows_metadata.append({
                    "filename": file_path.stem,
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "tags": metadata.get("tags", []),
                    "file_path": str(file_path),
                })

        if not workflows_metadata:
            return (
                MemoryUpdateDecision(
                    action="create",
                    target_workflow_filename=None,
                    reason="No readable workflow metadata is available.",
                ),
                0,
            )

        listing_parts = []
        for item in workflows_metadata:
            listing_parts.append(
                f"FILENAME: {item['filename']}\n"
                f"TITLE: {item['title']}\n"
                f"DESCRIPTION: {item['description']}\n"
                f"TAGS: {', '.join(item['tags']) or 'none'}"
            )
        selection_message = BaseMessage.make_user_message(
            role_name="User",
            content=(
                "POST-TASK MEMORY PHASE. The task answer is already final and "
                "must not be changed. Review the complete list of existing "
                "workflow-memory files below. Decide whether this completed "
                "experience should update exactly one existing workflow or "
                "create a new workflow. Choose update only when the existing "
                "workflow represents substantially the same reusable class "
                "of task, tools, and procedure. Otherwise choose create. Do "
                "not call tools. If updating, copy the filename exactly.\n\n"
                "Return only a valid JSON object with exactly these keys: "
                '\"action\" (\"create\" or \"update\"), '
                '\"target_workflow_filename\" (exact filename or null), and '
                '\"reason\" (a short string).\n\n'
                + "\n\n---\n\n".join(listing_parts)
            ),
        )

        try:
            response = await agent.astep(
                selection_message,
                response_format=(
                    MemoryUpdateDecision
                    if getattr(self, "native_structured_output", True)
                    else None
                ),
                disable_tools=True,
            )
            self._last_memory_selection_efficiency = self._phase_efficiency(
                response, messages=2
            )
            parsed = response.msgs[0].parsed if response.msgs else None
            if parsed is None and response.msgs:
                from camel.models._utils import strip_markdown_json_wrapper

                parsed = MemoryUpdateDecision.model_validate_json(
                    strip_markdown_json_wrapper(response.msgs[0].content)
                )
            decision = parsed
            if decision is None:
                raise ValueError("Memory selection returned no decision")
        except Exception as exc:
            logger.warning(
                "Adaptive memory selection failed; defaulting to create: %s",
                exc,
            )
            decision = MemoryUpdateDecision(
                action="create",
                target_workflow_filename=None,
                reason="Selection failed, so a new workflow is safer.",
            )

        available_by_filename = {
            item["filename"]: item for item in workflows_metadata
        }
        if (
            decision.action == "update"
            and decision.target_workflow_filename not in available_by_filename
        ):
            logger.warning(
                "Agent selected unavailable workflow %r; defaulting to create",
                decision.target_workflow_filename,
            )
            decision = MemoryUpdateDecision(
                action="create",
                target_workflow_filename=None,
                reason="The selected update filename was unavailable.",
            )
        elif decision.action == "create":
            decision.target_workflow_filename = None

        if decision.action == "update":
            selected = available_by_filename[
                decision.target_workflow_filename
            ]
            file_path = Path(selected["file_path"])
            content = file_path.read_text(encoding="utf-8")
            filtered_content = context_utility._filter_metadata_from_content(
                content
            )
            manager._loaded_workflow_paths[file_path.stem] = str(file_path)
            manager._loaded_workflow_contents = [{
                "filename": file_path.stem,
                "content": filtered_content,
            }]

        logger.info(
            "Adaptive memory agent decision: %s target=%s reason=%s",
            decision.action,
            decision.target_workflow_filename,
            decision.reason,
        )
        return decision, len(workflows_metadata)

    async def _reflect_on_result(
        self,
        agent: ChatAgent,
        result_data: Dict[str, Any],
        memory_strategy: str = "structured",
    ) -> Dict[str, Any]:
        r"""Add outcome feedback to memory before writing the workflow.

        The reflection is part of the memory-building phase only. It is not
        used to alter the already-scored answer for the current task.
        """
        outcome_known = result_data.get("score") in {0, 1}
        if not outcome_known:
            if memory_strategy == "self_organized":
                reflection_instruction = (
                    "No authoritative answer or correctness feedback is "
                    "available. In 2-3 sentences, reflect on what from this "
                    "attempt, if anything, may be useful to remember for "
                    "similar future tasks. Decide for yourself what matters. "
                    "Do not claim that the answer was correct or incorrect, "
                    "and do not call tools."
                )
            else:
                reflection_instruction = (
                    "No authoritative answer or correctness feedback is "
                    "available. In 2-3 sentences, record the workflow, tool "
                    "strategy, and practical lessons from the attempt. Do not "
                    "claim that the answer was correct or incorrect, and do "
                    "not call tools."
                )
            reflection_message = BaseMessage.make_user_message(
                role_name="User",
                content=(
                    "TASK RESULT: OUTCOME UNKNOWN\n"
                    f"Your raw answer: "
                    f"{result_data.get('raw_model_answer', '')}\n"
                    f"Submitted answer after the format guard: "
                    f"{result_data.get('submitted_answer', '')}\n"
                    f"Format repair: "
                    f"{result_data.get('format_guard', {})}\n\n"
                    f"{reflection_instruction}"
                ),
            )
            try:
                response = await agent.astep(
                    reflection_message, disable_tools=True
                )
                return self._phase_efficiency(response, messages=2)
            except Exception as exc:
                logger.warning(
                    "Failed to generate outcome-unknown reflection: %s", exc
                )
                return self._phase_efficiency(metrics_partial=True)

        is_correct = result_data.get("score") == 1
        status = "CORRECT" if is_correct else "INCORRECT"
        format_only_repair = (
            is_correct
            and result_data.get("raw_score") == 0
            and result_data.get("format_guard", {}).get("changed")
        )
        if memory_strategy == "self_organized":
            instruction = (
                "Reflect on what from this attempt, if anything, may be "
                "useful to remember for similar future tasks. Decide for "
                "yourself what matters and keep the reflection to 2-3 "
                "sentences."
            )
        elif format_only_repair:
            instruction = (
                "The substantive answer became correct only after the format "
                "guard repaired it. Focus the reflection on obeying the exact "
                "answer format next time, as well as preserving the successful "
                "solution approach. Keep the reflection to 2-3 sentences."
            )
        elif is_correct:
            instruction = (
                "Briefly identify the approach and key steps that worked. "
                "Keep the reflection to 2-3 sentences."
            )
        else:
            instruction = (
                "Briefly identify what likely went wrong and what should be "
                "done differently next time. Keep the reflection to 2-3 "
                "sentences."
            )

        reflection_message = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"TASK RESULT: {status}\n"
                f"Your raw answer: "
                f"{result_data.get('raw_model_answer', '')}\n"
                f"Submitted answer after the format guard: "
                f"{result_data.get('submitted_answer', '')}\n"
                f"Format repair: "
                f"{result_data.get('format_guard', {})}\n"
                f"Ground truth: {result_data.get('ground_truth', '')}\n\n"
                f"{instruction} Do not call tools; reflect only on the "
                "completed attempt and the feedback above."
            ),
        )
        try:
            response = await agent.astep(
                reflection_message, disable_tools=True
            )
            return self._phase_efficiency(response, messages=2)
        except Exception as exc:
            logger.warning(f"Failed to generate outcome reflection: {exc}")
            return self._phase_efficiency(metrics_partial=True)

    def _parse_workflow_summary(
        self,
        raw_content: str,
        summary_schema: Optional[type[BaseModel]] = None,
    ):
        """Parse WorkflowSummary from raw LLM response that may contain markdown.

        Handles responses wrapped in ```json ... ``` code blocks.

        Args:
            raw_content: Raw LLM response string

        Returns:
            WorkflowSummary instance or None if parsing fails
        """
        from camel.utils.context_utils import WorkflowSummary
        from camel.models._utils import strip_markdown_json_wrapper
        import json as json_module

        if not raw_content:
            return None

        schema = summary_schema or WorkflowSummary

        # Strip markdown wrapper and parse as JSON
        content = strip_markdown_json_wrapper(raw_content)

        try:
            data = json_module.loads(content)
            return schema(**data)
        except (json_module.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse workflow summary JSON: {e}")
            return None

    def _save_individual_result(self, result_data: Dict[str, Any]) -> None:
        r"""Save individual task result to a separate JSON file.

        Args:
            result_data: The task result data to save.
        """
        if not self.save_individual_results:
            return

        try:
            task_id = result_data["task_id"]
            output_file = Path(self.individual_results_dir) / f"{task_id}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved individual result to {output_file}")
        except Exception as e:
            logger.warning(f"Failed to save individual result for task {result_data.get('task_id')}: {e}")

    def _process_result(
        self,
        agent: ChatAgent,
        task: Dict[str, Any],
        result: Any,
        file_obj: Any,
        timing_data: Optional[Dict[str, float]] = None,
        labels_available: bool = True,
    ) -> Dict[str, Any]:
        r"""Process and store the result of a task."""
        raw_model_answer = self.get_final_answer(result.msgs[0].content)
        if raw_model_answer == "FINAL ANSWER not found":
            raise RuntimeError(
                "Model response omitted the required <final_answer> tag"
            )
        model_answer, guard_changed, guard_reason = (
            self.guard_final_answer(task["Question"], raw_model_answer)
        )
        final_answer = task["Final answer"] if labels_available else None
        raw_score = (
            self.question_scorer(raw_model_answer, final_answer)
            if labels_available else None
        )
        score = (
            self.question_scorer(model_answer, final_answer)
            if labels_available else None
        )
        tool_calls = result.info.get("tool_calls", [])

        # Extract detailed history from workforce agents if available
        # For workforce mode, use detailed_history which contains all worker messages
        # For single-agent mode, fall back to agent.memory.get_context()
        detailed_history = result.info.get("detailed_history", None)
        if detailed_history is not None:
            # Workforce mode: use the collected detailed history from all workers
            history = detailed_history
            message_count = len(detailed_history)
        else:
            # Single-agent mode: use agent's memory
            history = agent.memory.get_context()
            # get_context() returns (messages, token_count) tuple
            message_count = len(history[0]) if isinstance(history, tuple) and len(history) > 0 else len(history)

        logger.info(
            f"Raw model answer: {raw_model_answer[:100]}..."
            if len(raw_model_answer) > 100
            else f"Raw model answer: {raw_model_answer}"
        )
        if guard_changed:
            logger.info(
                "Format guard submitted %r (%s)",
                model_answer,
                guard_reason,
            )
        if labels_available:
            logger.info(f"Ground truth: {final_answer}")
            logger.info(f"Score: {'✓ CORRECT' if score else '✗ INCORRECT'}")
        else:
            logger.info("Ground truth: hidden")
            logger.info("Score: unavailable")
        logger.info(f"Tool calls made: {len(tool_calls)}")
        logger.info(f"Message count: {message_count}")

        solve_efficiency = self._phase_efficiency(
            result,
            messages=message_count,
            wall_seconds=(timing_data or {}).get("solve_seconds", 0.0),
        )
        empty_phase = self._phase_efficiency()

        result_data = {
            "task_id": task["task_id"],
            "question": task["Question"],
            "level": task["Level"],
            "raw_model_answer": raw_model_answer,
            "submitted_answer": model_answer,
            "model_answer": model_answer,
            "ground_truth": final_answer,
            "tool_calls": [tool.model_dump() for tool in tool_calls],
            "error": None,
            "score": int(score) if score is not None else None,
            "raw_score": int(raw_score) if raw_score is not None else None,
            "format_guard": {
                "changed": guard_changed,
                "reason": guard_reason,
            },
            "history": history,
            "message_count": message_count,
            "timing": {
                key: round(value, 3)
                for key, value in (timing_data or {}).items()
            },
            "experiment": self.experiment_metadata,
            "efficiency": {
                "solve": solve_efficiency,
                "memory_selection": empty_phase,
                "memory_build": self._phase_efficiency(),
                "overall": self._add_phases(solve_efficiency),
            },
        }
        self.results.append(result_data)

        # Write to main JSONL file
        file_obj.write(
            json.dumps(result_data, ensure_ascii=False) + "\n"
        )
        file_obj.flush()

        # Save individual result file
        self._save_individual_result(result_data)
        return result_data

    def _handle_error(
        self,
        task: Dict[str, Any],
        error: Exception,
        file_obj: Any,
        timing_data: Optional[Dict[str, float]] = None,
        labels_available: bool = True,
    ) -> None:
        r"""Handle errors encountered during task processing."""
        logger.warning(f"Error processing task {task['task_id']}: {error}")
        error_data = {
            "task_id": task["task_id"],
            "question": task["Question"],
            "level": task["Level"],
            "model_answer": "ERROR",
            "raw_model_answer": "ERROR",
            "submitted_answer": "ERROR",
            "ground_truth": (
                task["Final answer"] if labels_available else None
            ),
            "tool_calls": [],
            "error": str(error),
            "score": 0 if labels_available else None,
            "raw_score": 0 if labels_available else None,
            "format_guard": {"changed": False, "reason": None},
            "timing": dict(timing_data or {}),
            "experiment": self.experiment_metadata,
            "efficiency": {
                "solve": self._phase_efficiency(
                    wall_seconds=(timing_data or {}).get(
                        "solve_seconds",
                        (timing_data or {}).get("total_seconds", 0.0),
                    ),
                    metrics_partial=True,
                ),
                "memory_selection": self._phase_efficiency(),
                "memory_build": self._phase_efficiency(),
                "overall": self._phase_efficiency(
                    wall_seconds=(timing_data or {}).get(
                        "total_seconds", 0.0
                    ),
                    metrics_partial=True,
                ),
            },
        }
        self.results.append(error_data)

        # Write the error_data as a JSON line to the provided file object
        file_obj.write(json.dumps(error_data, ensure_ascii=False) + "\n")
        file_obj.flush()

        # Save individual error result file
        self._save_individual_result(error_data)
    
    def question_scorer(self, model_answer: str, ground_truth: str) -> bool:
        r"""Scorer for the GAIA benchmark.
        https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/
        scorer.py

        Args:
            model_answer (str): The model answer.
            ground_truth (str): The ground truth answer.

        Returns:
            bool: The score of the model
        """

        def is_float(element: Any) -> bool:
            try:
                float(element)
                return True
            except ValueError:
                return False

        if is_float(ground_truth):
            logger.info(f"Evaluating {model_answer} as a number.")
            normalized_answer = self.normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)

        elif any(char in ground_truth for char in [",", ";"]):
            logger.info(
                f"Evaluating {model_answer} as a comma separated list."
            )
            gt_elems = self.split_string(ground_truth)
            ma_elems = self.split_string(model_answer)

            if len(gt_elems) != len(ma_elems):
                logger.warning(
                    "Answer lists have different lengths, returning False."
                )
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = self.normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    ma_elem = self.normalize_str(ma_elem, remove_punct=False)
                    gt_elem = self.normalize_str(gt_elem, remove_punct=False)
                    comparisons.append(ma_elem == gt_elem)
            return all(comparisons)
        else:
            logger.info(f"Evaluating {model_answer} as a string.")
            ma_elem = self.normalize_str(model_answer)
            gt_elem = self.normalize_str(ground_truth)
            return ma_elem == gt_elem

    def guard_final_answer(
        self, question: str, raw_answer: str
    ) -> tuple[str, bool, Optional[str]]:
        r"""Repair submission formatting without consulting ground truth.

        The guard intentionally makes only conservative transformations based
        on the task wording. Native and guarded answers are both retained in
        result artifacts so instruction-following failures remain measurable.
        """
        answer = raw_answer.strip()
        reasons: List[str] = []

        # Remove generic prose that violates the answer-only contract.
        prefix_match = re.fullmatch(
            r"(?is)(?:the\s+)?answer\s*(?:is|:)\s*(.+)", answer
        )
        if prefix_match:
            answer = prefix_match.group(1).strip()
            reasons.append("removed generic answer prefix")

        question_lower = question.lower()
        numeric_request = bool(
            re.search(
                r"\b(?:how many|how much|number of|percentage|percent|"
                r"round(?:ed)?|absolute difference)\b",
                question_lower,
            )
        )
        explicitly_requires_unit = bool(
            re.search(
                r"\b(?:include|show|write|provide)\b.{0,30}"
                r"\b(?:the\s+)?units?\b|\bwith\s+(?:the\s+)?units?\b",
                question_lower,
            )
        )

        number_pattern = (
            r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        )
        numeric_with_label = re.fullmatch(
            rf"\s*({number_pattern})\s+([A-Za-z][A-Za-z .\-/]*)\s*",
            answer,
        )
        if (
            numeric_request
            and not explicitly_requires_unit
            and numeric_with_label
        ):
            answer = numeric_with_label.group(1)
            reasons.append("removed an unrequested label from a numeric answer")

        changed = answer != raw_answer.strip()
        return answer, changed, "; ".join(reasons) if reasons else None

    def normalize_number_str(self, number_str: str) -> float:
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            logger.error(
                f"String {number_str} cannot be normalized to number str."
            )
            return float("inf")

    def split_string(
        self, s: str, char_list: Optional[List[str]] = None
    ) -> list[str]:
        r"""Split a string based on a list of characters.

        Args:
            s (str): The string to split.
            char_list (Optional[List[str]], optional): T
                he list of characters to split on.
                (default: :obj:`None`)
        """
        if char_list is None:
            char_list = [",", ";"]
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def normalize_str(self, input_str, remove_punct=True) -> str:
        r"""Normalize a string.

        Args:
            input_str: The input string to normalize.
            remove_punct: Whether to remove punctuation.

        Returns:
            str: The normalized string.
        """
        no_spaces = re.sub(r"\s", "", input_str)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()

    def get_final_answer(self, content: str) -> str:
        r"""Get the final answer from the content.

        Args:
            content (str): The content to extract the final answer from.

        Returns:
            str: The final answer.
        """
        # Try tag format first: <final_answer>...</final_answer>
        # Use rfind to get the LAST occurrence (for multi-agent workforce responses)
        start_tag = "<final_answer>"
        end_tag = "</final_answer>"
        start_idx = content.rfind(start_tag)
        if start_idx != -1:
            start_idx += len(start_tag)
            end_idx = content.find(end_tag, start_idx)
            if end_idx != -1:
                return content[start_idx:end_idx].strip()

        # Fallback to old format: FINAL ANSWER: ...
        final_answer_index = content.find("FINAL ANSWER")
        if final_answer_index == -1:
            return "FINAL ANSWER not found"
        start_index = final_answer_index + len("FINAL ANSWER: ")
        final_answer_content = content[start_index:].strip()
        return final_answer_content
