# Run records

These are cleaned copies of the JSONL run records from the workflow-memory
experiments. Every row is one attempted GAIA task.

## What was removed

GAIA asks that its questions and answers are not reshared in crawlable form,
so the following fields were removed from every row:

- `question`, `ground_truth`
- `raw_model_answer`, `submitted_answer`, `model_answer`
- `tool_calls` (arguments and results) and `history` (full transcripts)
- free-text model output that can describe a task: the selector's
  `raw_response` and the memory writer's `agent_decision.reason`

Provider account identifiers were removed from error messages, and absolute
local paths were shortened to repository-relative paths or file names.

The workflow-memory banks are not included, because their records paraphrase
GAIA test tasks and sometimes quote the answers the models submitted.

## What was kept

- `task_id` and `level`, so rows can be joined with the GAIA dataset
- `score` (after the format guard) and `raw_score` (before it)
- `submitted_answer_present`: whether the run produced a final answer
- `error`, `format_guard.changed` and `format_guard.reason`
- `timing`, `efficiency` (tokens, model calls, tool calls by name, wall time)
  and `tool_call_names` (tools called, in order, without arguments)
- `experiment`: model, provider, decoding settings and memory-bank digest
- memory selection (`workflows_loaded`, `workflow_selection_audit`) and memory
  writing (`memory_operation`, `memory_branches`) metadata

All accuracy, paired-transition, token, tool-call and selection figures can be
recomputed from these fields.

## Files

`manifest.json` lists every file with its row count and SHA-256 hash.

Primary experiment (`runs/<model>/`):

| Model | Construction (229 test tasks) | Baseline (106 validation tasks) | Frozen memory (106 validation tasks) |
|---|---|---|---|
| GPT-5.6 Luna | `test_full_luna_test_outcome_blind_agentic_adaptive_memory_build_v1` | `validation_full_luna_validation_memory_build_v1_attachment_free` | `validation_with_test_memory_full_luna_validation_test_memory_agentic_v1` |
| Gemini 3.7 Flash | `test_full_gemini_test_outcome_blind_agentic_adaptive_memory_build_v1` | `validation_full_gemini_validation_baseline_v1` | `validation_with_test_memory_full_gemini_validation_test_memory_agentic_v1` |
| DeepSeek V4 Flash 0731 | `test_full_deepseek_test_outcome_blind_agentic_adaptive_memory_build_efficiency_v3` | `validation_full_deepseek_validation_baseline_efficiency_v1` | `validation_with_test_memory_full_deepseek_validation_test_memory_agentic_efficiency_v2` |
| Ministral 3B | `test_full_ministral_test_outcome_blind_agentic_adaptive_memory_build_toolfree_v2` | `validation_full_ministral_validation_baseline_efficiency_v1` | `validation_with_test_memory_full_ministral_validation_agentic_memory_toolfree_bank_v1` |

The Luna baseline file name contains `memory_build` because that run also wrote
memories after each answer; its writing time is excluded from the reported
validation time.

Recording-policy follow-up (50 construction tasks): the
`*_schema_ablation_50_*` files for Luna, Gemini and Ministral (two Ministral
replications, `v1` and `v2`). Each `*_task_subset.json` gives the sampled task
IDs and their order.

`audits/` holds the audit tables:

- `POPULATION_AUDIT.csv`: inclusion decision and exclusion reason for every
  GAIA test and validation task
- `TEST_TASK_ORDER.csv` and `VALIDATION_TASK_ORDER.csv`: task order shared by
  all models
- `RETRIEVAL_AUDIT.csv`: selection method and inserted context size for every
  memory-condition attempt
- `SCORING_GUARD_AUDIT.csv`: the 24 answers changed by the format guard, with
  official and logged scores (questions and answers removed)
- `CASE_AUDIT.md`: three task-linked example cases (answers withheld)
