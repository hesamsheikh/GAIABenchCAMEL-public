from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

# Suppress BeautifulSoup parser warning from wikipedia library
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

# Configure logging - only show our custom logs, silence camel's verbose output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Silence camel's verbose internal logging
logging.getLogger('camel').setLevel(logging.WARNING)
# Suppress "Format validation error" warnings from structured output parsing
# (we handle these with our own fallback parser)
logging.getLogger('camel.models.openai_compatible_model').setLevel(logging.ERROR)
# Suppress "Unknown model" context window warnings for custom model IDs
logging.getLogger('camel.types.unified_model_type').setLevel(logging.ERROR)
# Keep our gaia benchmark logs visible
logging.getLogger('src.gaia').setLevel(logging.INFO)
logging.getLogger('src.agents').setLevel(logging.INFO)

from src.gaia import GAIABenchmark
from src.experiment_utils import parse_level_counts, select_stratified_tasks
from src.toolkits import SafeCodeExecutionToolkit, WebToolkit, YouTubeToolkit
from src.agents import GAIAChatAgent
from camel.models import ModelFactory
from camel.toolkits import (
    SearchToolkit,
    FileToolkit,
    ExcelToolkit,
    Crawl4AIToolkit,  # Native async toolkit (works with astep)
)
from camel.types import ModelPlatformType

# Working directory for file operations
WORKING_DIRECTORY = os.environ.get("CAMEL_WORKDIR") or os.path.abspath("execution/files")
os.makedirs(WORKING_DIRECTORY, exist_ok=True)

# Experiment artifact configuration. Each invocation gets a fresh directory by
# default, so neither historical memories nor another Qwen run can be
# overwritten. Set GAIA_RUN_ID explicitly to name a planned run.
RUN_MODE = os.environ.get("GAIA_RUN_MODE", "smoke").lower()
if RUN_MODE not in {"smoke", "full"}:
    raise ValueError("GAIA_RUN_MODE must be 'smoke' or 'full'")
DEFAULT_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = os.environ.get("GAIA_RUN_ID", DEFAULT_RUN_ID)
TEST_VALID = os.environ.get("GAIA_SPLIT", "valid").lower()
if TEST_VALID not in {"valid", "test"}:
    raise ValueError("GAIA_SPLIT must be 'valid' or 'test'")
SPLIT_ARTIFACT_NAME = (
    "validation" if TEST_VALID == "valid" else "test"
)

MODEL_CONFIGS = {
    "qwen3_8_27b": {
        "model_id": "qwen/qwen3.8-27b-20260814",
        "provider": "akashml",
        "quantization": "bf16",
        "supports_temperature": True,
        "context_limit": 200_000,
    },
    "qwen3_7_flash": {
        "model_id": "qwen/qwen3.7-flash-20260727",
        "provider": "alibaba",
        "quantization": None,
        "supports_temperature": True,
        "context_limit": 200_000,
    },
    "gemini_3_7_flash": {
        "model_id": "google/gemini-3.7-flash-20260813",
        "provider": "google-ai-studio",
        "quantization": None,
        "supports_temperature": True,
        "context_limit": 200_000,
    },
    "muse_glimmer_30b": {
        "model_id": "meta/muse-glimmer-30b-20260810",
        "provider": "parasail",
        "quantization": "bf16",
        "supports_temperature": True,
        "context_limit": 120_000,
    },
    "deepseek_v4_flash_0731": {
        "model_id": "deepseek/deepseek-v4-flash-20260731",
        "provider": "coreweave",
        "quantization": "fp8",
        "supports_temperature": True,
        "context_limit": 200_000,
        "supports_native_structured_output": False,
    },
    "gpt_5_6_luna": {
        "model_id": "openai/gpt-5.6-luna-20260709",
        "provider": "openai",
        "quantization": None,
        "supports_temperature": False,
        "context_limit": 200_000,
    },
    "ox_alpha": {
        "model_id": "stealth/ox-alpha",
        "provider": "stealth",
        "quantization": None,
        "supports_temperature": True,
        "supports_seed": False,
        # OpenRouter advertises response_format, but the pinned stealth
        # endpoint rejects CAMEL's parsed-schema request. Use the same
        # plain-JSON reflection path as other incompatible endpoints.
        "supports_native_structured_output": False,
        # Keep the experimental cap fixed even though the endpoint advertises
        # a 1,048,576-token context window.
        "context_limit": 200_000,
    },
    "lfm_2_5_2_6b_free": {
        "model_id": "liquid/lfm-2.5-2.6b:free",
        "provider": "liquid",
        "quantization": None,
        "supports_temperature": True,
        "supports_seed": False,
        "supports_native_structured_output": False,
        "context_limit": 131_072,
    },
    "ministral_3b_2512": {
        "model_id": "mistralai/ministral-3b-2512",
        "provider": "mistral",
        "quantization": None,
        "supports_temperature": True,
        "supports_reasoning": False,
        # Reserve space inside the 131,072-token provider window for the
        # 8,192-token completion and serialized tool schemas.
        "context_limit": 120_000,
    },
    "nemotron_3_nano_30b_a3b": {
        "model_id": "nvidia/nemotron-3-nano-30b-a3b",
        "provider": "novita",
        "quantization": "fp4",
        "supports_temperature": True,
        # Novita accepts tools and tool_choice, but not reasoning_effort or
        # schema-constrained structured_outputs.
        "supports_reasoning": False,
        "supports_native_structured_output": False,
        "context_limit": 200_000,
    },
    "qwen3_30b_a3b_instruct_2507": {
        "model_id": "qwen/qwen3-30b-a3b-instruct-2507",
        "provider": "coreweave",
        "quantization": "bf16",
        "supports_temperature": True,
        "supports_reasoning": False,
        "context_limit": 200_000,
    },
}
MODEL_ARTIFACT_NAME = os.environ.get("GAIA_MODEL", "qwen3_8_27b")
if MODEL_ARTIFACT_NAME not in MODEL_CONFIGS:
    raise ValueError(
        "GAIA_MODEL must be one of: " + ", ".join(MODEL_CONFIGS)
    )
SELECTED_MODEL = MODEL_CONFIGS[MODEL_ARTIFACT_NAME]
RUN_NAME = f"{SPLIT_ARTIFACT_NAME}_{RUN_MODE}_{RUN_ID}"

# Workflow memory configuration
SAVE_WORKFLOW_MEMORY = os.environ.get(
    "GAIA_SAVE_WORKFLOW_MEMORY", "1"
).lower() not in {"0", "false", "no"}
ADAPTIVE_MEMORY_UPDATES = os.environ.get(
    "GAIA_ADAPTIVE_MEMORY_UPDATES", "0"
).lower() not in {"0", "false", "no"}
WORKFLOW_MEMORY_DIR = str(
    Path("workflows") / MODEL_ARTIFACT_NAME / RUN_NAME
)
WORKFLOW_FILENAME_PREFIX = MODEL_ARTIFACT_NAME
MEMORY_WRITING_MODE = os.environ.get(
    "GAIA_MEMORY_WRITING_MODE", "structured"
).lower()
if MEMORY_WRITING_MODE not in {"structured", "paired_schema_ablation"}:
    raise ValueError(
        "GAIA_MEMORY_WRITING_MODE must be 'structured' or "
        "'paired_schema_ablation'"
    )
PAIRED_WORKFLOW_MEMORY_DIRS = (
    {
        "structured": str(Path(WORKFLOW_MEMORY_DIR) / "structured"),
        "self_organized": str(
            Path(WORKFLOW_MEMORY_DIR) / "self_organized"
        ),
    }
    if MEMORY_WRITING_MODE == "paired_schema_ablation"
    else {}
)
if MEMORY_WRITING_MODE == "paired_schema_ablation":
    if TEST_VALID != "test" or not SAVE_WORKFLOW_MEMORY:
        raise ValueError(
            "paired_schema_ablation requires GAIA_SPLIT=test and "
            "GAIA_SAVE_WORKFLOW_MEMORY=1"
        )
    if not ADAPTIVE_MEMORY_UPDATES:
        raise ValueError(
            "paired_schema_ablation requires "
            "GAIA_ADAPTIVE_MEMORY_UPDATES=1"
        )

# Individual results configuration
SAVE_INDIVIDUAL_RESULTS = True  # Set to False to disable saving individual task results
INDIVIDUAL_RESULTS_DIR = str(
    Path("task_results") / MODEL_ARTIFACT_NAME / RUN_NAME
)
SKIP_FINISHED_TASKS = True  # Skip tasks that already have results in INDIVIDUAL_RESULTS_DIR

# Blacklisted task IDs (tasks that consistently fail and should be skipped)
BLACKLISTED_TASK_IDS = []

# Benchmark run configuration
LEVEL = "all"  # Run all difficulty levels with a fixed modality filter
RANDOMIZE = False  # Whether to randomize task order
NUM_SAMPLES = 3 if RUN_MODE == "smoke" else -1
RESULTS_FILE = str(
    Path("results") / MODEL_ARTIFACT_NAME / RUN_NAME
)
STRATIFIED_COUNTS_TEXT = os.environ.get("GAIA_STRATIFIED_COUNTS", "").strip()
STRATIFIED_COUNTS = (
    parse_level_counts(STRATIFIED_COUNTS_TEXT)
    if STRATIFIED_COUNTS_TEXT
    else {}
)
TASK_SUBSET_SEED = os.environ.get("GAIA_SUBSET_SEED", "gaia-subset-v1")
TASK_SUBSET_MANIFEST = Path(f"{RESULTS_FILE}_task_subset.json")
if MEMORY_WRITING_MODE == "paired_schema_ablation" and RUN_MODE != "full":
    raise ValueError("paired_schema_ablation requires GAIA_RUN_MODE=full")
PROGRESS_REPORT_INTERVAL = 10  # Report accuracy every N tasks

# Conservative attachment-free text population used for both memory building
# and final evaluation. Any task with a benchmark attachment is excluded,
# regardless of file type, so the experiment has no embedding/retriever
# dependency.

# Reproducible agent configuration. Keep these values identical across all
# experimental conditions (baseline, relevant memory, and irrelevant memory).
MODEL_MAX_OUTPUT_TOKENS = 8192
MODEL_TEMPERATURE = 0.0
MODEL_SEED = 42
AGENT_TOKEN_LIMIT = SELECTED_MODEL["context_limit"]
AGENT_STEP_TIMEOUT = 600.0
TOOL_EXECUTION_TIMEOUT = 120.0

# Pin the canonical dated OpenRouter model and one BF16 provider so model
# revision and numerical precision cannot change between conditions.
MODEL_ID = SELECTED_MODEL["model_id"]
OPENROUTER_PROVIDER = SELECTED_MODEL["provider"]
MODEL_REASONING_EFFORT = "low"

# Only send parameters supported by the pinned endpoint. In particular, the
# OpenAI endpoint serving GPT-5.6 Luna does not expose temperature.
model_config_dict = {
    "max_tokens": MODEL_MAX_OUTPUT_TOKENS,
    "tool_choice": "auto",
    "extra_body": {
        "provider": {
            "only": [OPENROUTER_PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    },
}
if SELECTED_MODEL.get("supports_reasoning", True):
    model_config_dict["extra_body"]["reasoning"] = {
        "effort": MODEL_REASONING_EFFORT,
        "exclude": True,
    }
if SELECTED_MODEL.get("supports_seed", True):
    model_config_dict["seed"] = MODEL_SEED
if SELECTED_MODEL["supports_temperature"]:
    model_config_dict["temperature"] = MODEL_TEMPERATURE
if SELECTED_MODEL["quantization"]:
    model_config_dict["extra_body"]["provider"]["quantizations"] = [
        SELECTED_MODEL["quantization"]
    ]

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENROUTER,
    model_type=MODEL_ID,
    model_config_dict=model_config_dict,
    timeout=AGENT_STEP_TIMEOUT,
)

system_message = (
    "You are a general AI assistant. "
    "Think step-by-step, critically analyze the question, "
    "and use your tools effectively. "
    "For information gathering, prioritize tools in this order: "
    "1) wikipedia_full_page for factual/encyclopedic info, "
    "2) search_tavily for general web searches, "
    "3) scrape_url ONLY when a specific non-Wikipedia URL must be accessed. "
    "Before giving your final answer, re-read the question to verify "
    "your answer matches the exact format and units requested. "
    "End with: <final_answer>answer only, no extra text</final_answer>"
)

# Initialize toolkits
code_toolkit = SafeCodeExecutionToolkit(
    verbose=False,
    log_dir="execution/code",
    network_mode="bridge",
)
search_toolkit = SearchToolkit()
web_toolkit = WebToolkit()
crawl_toolkit = Crawl4AIToolkit(timeout=60, max_content_length=50000)  # Native async toolkit (works with astep)
youtube_toolkit = YouTubeToolkit()
file_toolkit = FileToolkit(working_directory=WORKING_DIRECTORY)
excel_toolkit = ExcelToolkit(working_directory=WORKING_DIRECTORY)

# Collect all tools
all_tools = [
    *code_toolkit.get_tools(),           # execute_code, execute_command
    search_toolkit.search_tavily,        # web search (Tavily API)
    *web_toolkit.get_tools(),            # wikipedia_search_content (searches full article)
    *crawl_toolkit.get_tools(),          # scrape (fetch any webpage, native async)
    *youtube_toolkit.get_tools(),        # get_youtube_transcript, get_youtube_video_info
    *file_toolkit.get_tools(),           # read_file, write_to_file, etc.
    *excel_toolkit.get_tools(),          # excel operations
]

agent = GAIAChatAgent(
    system_message=system_message,
    model=model,
    tools=all_tools,
    max_iteration=30,
    token_limit=AGENT_TOKEN_LIMIT,
    summarize_threshold=None,
    step_timeout=AGENT_STEP_TIMEOUT,
    tool_execution_timeout=TOOL_EXECUTION_TIMEOUT,
)

async def main():
    artifact_paths = [
        Path(INDIVIDUAL_RESULTS_DIR),
        Path(RESULTS_FILE).with_suffix(".jsonl"),
    ]
    if SAVE_WORKFLOW_MEMORY:
        artifact_paths.append(Path(WORKFLOW_MEMORY_DIR))
    if STRATIFIED_COUNTS:
        artifact_paths.append(TASK_SUBSET_MANIFEST)
    existing_paths = [path for path in artifact_paths if path.exists()]
    if existing_paths:
        formatted_paths = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            "Refusing to overwrite an existing experiment run: "
            f"{formatted_paths}. Choose a fresh GAIA_RUN_ID."
        )

    benchmark = GAIABenchmark(
        data_dir="dataset",
        save_to=RESULTS_FILE,
        enable_attachment_retrieval=False,
        save_workflow_memory=SAVE_WORKFLOW_MEMORY,
        adaptive_memory_updates=ADAPTIVE_MEMORY_UPDATES,
        paired_workflow_memory_dirs=PAIRED_WORKFLOW_MEMORY_DIRS,
        workflow_memory_dir=WORKFLOW_MEMORY_DIR,
        workflow_filename_prefix=WORKFLOW_FILENAME_PREFIX,
        save_individual_results=SAVE_INDIVIDUAL_RESULTS,
        individual_results_dir=INDIVIDUAL_RESULTS_DIR,
        progress_report_interval=PROGRESS_REPORT_INTERVAL,
        skip_finished_tasks=SKIP_FINISHED_TASKS,
        blacklisted_task_ids=BLACKLISTED_TASK_IDS,
        experiment_metadata={
            "model_key": MODEL_ARTIFACT_NAME,
            "model_id": MODEL_ID,
            "provider": OPENROUTER_PROVIDER,
            "quantization": SELECTED_MODEL["quantization"],
            "reasoning_effort": MODEL_REASONING_EFFORT,
            "temperature": (
                MODEL_TEMPERATURE
                if SELECTED_MODEL["supports_temperature"]
                else None
            ),
            "seed": (
                MODEL_SEED
                if SELECTED_MODEL.get("supports_seed", True)
                else None
            ),
            "workflow_memory_saved": SAVE_WORKFLOW_MEMORY,
            "adaptive_memory_updates": ADAPTIVE_MEMORY_UPDATES,
            "memory_writing_mode": MEMORY_WRITING_MODE,
            "paired_workflow_memory_dirs": PAIRED_WORKFLOW_MEMORY_DIRS,
            "stratified_counts": STRATIFIED_COUNTS,
            "task_subset_seed": (
                TASK_SUBSET_SEED if STRATIFIED_COUNTS else None
            ),
            "population": "attachment_free_text",
            "attachment_retrieval_enabled": False,
            "malformed_json_policy": "count_as_failure_no_retry",
            "dataset_split": TEST_VALID,
            "ground_truth_available": TEST_VALID == "valid",
            "efficiency_instrumentation": "phase_v1",
            "token_usage_source": "provider_reported",
            "native_structured_output": SELECTED_MODEL.get(
                "supports_native_structured_output", True
            ),
        },
    )

    # Apply the reproducible attachment-free text definition before sampling.
    # Validation retains 106 tasks; test retains 229 real tasks after GAIA's
    # dummy 0-0-0-0-0 row is removed by the dataset loader.
    original_count = len(benchmark.data[TEST_VALID])
    benchmark.data[TEST_VALID] = [
        task for task in benchmark.data[TEST_VALID]
        if not benchmark._get_multimodal_requirements(task)
        and not task.get("file_name")
    ]
    logging.info(
        "Attachment-free text filter retained %d/%d %s tasks",
        len(benchmark.data[TEST_VALID]),
        original_count,
        TEST_VALID,
    )
    if STRATIFIED_COUNTS:
        selected_tasks = select_stratified_tasks(
            benchmark.data[TEST_VALID],
            STRATIFIED_COUNTS,
            TASK_SUBSET_SEED,
        )
        benchmark.data[TEST_VALID] = selected_tasks
        TASK_SUBSET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        TASK_SUBSET_MANIFEST.write_text(
            json.dumps(
                {
                    "split": TEST_VALID,
                    "seed": TASK_SUBSET_SEED,
                    "counts": STRATIFIED_COUNTS,
                    "task_ids": [
                        task["task_id"] for task in selected_tasks
                    ],
                    "tasks": [
                        {
                            "task_id": task["task_id"],
                            "level": task["Level"],
                        }
                        for task in selected_tasks
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logging.info(
            "Stratified subset retained %d tasks; manifest: %s",
            len(selected_tasks),
            TASK_SUBSET_MANIFEST,
        )
    logging.info("Dataset split: %s", TEST_VALID)
    logging.info("Run mode: %s", RUN_MODE)
    logging.info("Model: %s (%s)", MODEL_ARTIFACT_NAME, MODEL_ID)
    logging.info("Provider: %s", OPENROUTER_PROVIDER)
    logging.info("Results directory: %s", INDIVIDUAL_RESULTS_DIR)
    logging.info("Save workflow memory: %s", SAVE_WORKFLOW_MEMORY)
    logging.info("Memory writing mode: %s", MEMORY_WRITING_MODE)
    logging.info(
        "Adaptive create-or-update after answer: %s",
        ADAPTIVE_MEMORY_UPDATES,
    )
    if SAVE_WORKFLOW_MEMORY:
        logging.info("Workflow directory: %s", WORKFLOW_MEMORY_DIR)
    await benchmark.run(
        agent,
        test_valid=TEST_VALID,
        level=LEVEL,
        randomize=RANDOMIZE,
        num_samples=NUM_SAMPLES,
    )

    # Display final results summary
    logging.info("\n" + "="*60)
    logging.info("Benchmark run completed!")
    logging.info(f"Results saved to: {benchmark.save_to}")
    logging.info("="*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Clean up Docker resources
        code_toolkit.cleanup()
