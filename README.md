# GAIABenchCAMEL

GAIA Benchmark runner using the CAMEL-AI framework with Docker-based safe code execution.

## Repository contents

- `main_single_agent.py`, `main_single_agent_with_memory.py` and `src/`: the
  GAIA runners for the no-memory, memory-building and frozen-memory conditions
- `camel/`: the vendored CAMEL source with the workflow-memory changes; see
  [`CAMEL_PIN.md`](CAMEL_PIN.md) for the upstream revision
- `analyze_results.py`, `analyze_schema_ablation.py` and the `*_report.md`
  files: analysis scripts and aggregate reports
- [`data/`](data/README.md): cleaned run records and audit tables

The analysis scripts were written for the original, uncleaned run records; see
`data/README.md` for the fields that were removed.

The GAIA dataset is not included. It is available from Hugging Face
(`gaia-benchmark/GAIA`) after accepting its terms.

## Setup

### 1. Install Dependencies

Use the vendored, experiment-pinned CAMEL source rather than installing the
latest package from PyPI:

```bash
pip install -e './camel[all]'
```

The pinned CAMEL version and upstream provenance are recorded in
[`CAMEL_PIN.md`](CAMEL_PIN.md). Do not upgrade CAMEL between experimental
conditions.

### 2. Pre-build Docker Image

The code execution toolkit uses Docker for sandboxed execution. Pre-build the image before running:

```bash
docker build -t camel-interpreter:latest .venv/lib/python3.12/site-packages/camel/interpreters/docker/
```

The image includes Ubuntu 22.04 with Python 3.10, R, Node.js 22, and common build tools.

### 3. Configure Environment

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

For the pinned Qwen3.8-27B experiment, set `OPENROUTER_API_KEY`. The runners
use OpenRouter's canonical dated model slug and restrict routing to the pinned
AkashML BF16 endpoint; see `CAMEL_PIN.md`.

### 4. Run the Qwen validation memory-building evaluation

The baseline runner defaults to a three-task smoke test. Every run writes to a
fresh, identifiable path under `workflows/qwen3_8_27b/`,
`task_results/qwen3_8_27b/`, and `results/qwen3_8_27b/`. It refuses to start if
the selected run ID already exists, so historical workflow Markdown files are
never overwritten.

```bash
GAIA_RUN_MODE=smoke GAIA_RUN_ID=qwen_smoke_01 python main_single_agent.py
```

After the smoke test succeeds, run the full 106-task attachment-free text
validation population (39 Level 1, 53 Level 2, 14 Level 3):

```bash
GAIA_RUN_MODE=full GAIA_RUN_ID=qwen_memory_build_v1 python main_single_agent.py
```

Each workflow filename also starts with `qwen3_8_27b` and includes the GAIA
task UUID, making it unambiguous and collision-resistant.

For an adaptive outcome-blind memory-building run, enable post-answer
create-or-update decisions explicitly:

```bash
GAIA_MODEL=gpt_5_6_luna \
GAIA_SPLIT=test \
GAIA_RUN_MODE=full \
GAIA_RUN_ID=luna_test_adaptive_memory_v1 \
GAIA_SAVE_WORKFLOW_MEMORY=1 \
GAIA_ADAPTIVE_MEMORY_UPDATES=1 \
python main_single_agent.py
```

The task-solving phase receives no workflow memories. After the final answer
is fixed, the same agent receives the complete list of existing workflow
filenames, titles, descriptions, and tags. It chooses either `create` or one
exact file to `update`. Only the selected file's full content is then supplied
to the workflow-summary phase. Updates increment `workflow_version`, append
the contributing GAIA task ID, and archive the previous version under the
run's `_history/` directory.

For a no-memory smoke run with another pinned model, set `GAIA_MODEL` and turn
off workflow writing. Supported keys and provider pins are in `CAMEL_PIN.md`:

```bash
GAIA_MODEL=gemini_3_7_flash \
GAIA_RUN_MODE=smoke \
GAIA_RUN_ID=my_comparison \
GAIA_SAVE_WORKFLOW_MEMORY=0 \
python main_single_agent.py
```

Every result row records solve time, optional memory-building time, total task
time, model/provider metadata, and whether empty-response recovery was needed.
It also keeps the model's native `raw_model_answer`/`raw_score` separately from
the ground-truth-blind format guard's `submitted_answer`/`score`, including the
reason for any repair.

The memory-condition runner requires the frozen model-specific bank explicitly.
By default, the task agent sees the current question plus the metadata index
for every canonical workflow and selects exactly three relevant memories. The
selection exchange is flushed before solving; only the original system prompt
and the selected memories' full contents remain. The runner records the exact
filenames, selection response/method, fallback status, and selection time, and
never updates the frozen bank:

```bash
GAIA_MODEL=gpt_5_6_luna \
GAIA_SPLIT=valid \
GAIA_RUN_MODE=smoke \
GAIA_RUN_ID=luna_agentic_memory_smoke_01 \
GAIA_WORKFLOW_SELECTION=agentic \
GAIA_MEMORY_BANK_DIR=workflows/gpt_5_6_luna/test_full_luna_test_outcome_blind_agentic_adaptive_memory_build_v1 \
python main_single_agent_with_memory.py
```

The separate potentially-irrelevant control must be requested explicitly with
`GAIA_WORKFLOW_SELECTION=fixed_control`; it loads the first three canonical
files in deterministic filename order for every task.

## Code Execution Toolkit

The project includes a custom `SafeCodeExecutionToolkit` that extends CAMEL's Docker interpreter with:

- **Network access**: Containers run with `network_mode="bridge"` for internet connectivity
- **Execution logging**: All code executions are saved to `execution/code/`

### Components

| File | Description |
|------|-------------|
| `src/interpreters/logged_docker_interpreter.py` | Docker interpreter with network access and local logging |
| `src/toolkits/safe_code_execution.py` | Toolkit wrapper providing `execute_code` and `execute_command` tools |

### Usage

```python
from src.toolkits import SafeCodeExecutionToolkit

toolkit = SafeCodeExecutionToolkit(
    verbose=False,           # Print execution output
    log_dir="execution/code", # Directory for code logs
    network_mode="bridge",   # Enable internet access
)

tools = toolkit.get_tools()  # Returns [execute_code, execute_command]

# Use with ChatAgent
agent = ChatAgent(task_prompt, model, tools=tools)

# Cleanup when done
toolkit.cleanup()
```

### Execution Directory

All execution artifacts are stored in `execution/`:

```
execution/
├── code/                           # Code execution logs
│   ├── 20240115_103045_0000_meta.json
│   ├── 20240115_103045_0000_code.py
│   └── ...
└── retriever/                      # Vector storage for document retrieval
    └── {task_id}/
        └── ...
```

## Testing

Test the Docker code execution:

```bash
python test/test_docker_execution.py
```

## Running the Benchmark

```bash
python main.py
```
