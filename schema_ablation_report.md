# Luna workflow-memory recording-strategy ablation

## Research question

Does Luna benefit from being explicitly told which workflow-memory fields to
record, or can it write a better memory when it chooses the contents and
organization itself?

This experiment compares the complete recording strategies, not merely the
visual presence of Markdown headings. The structured writer was prompted to
record the normal workflow schema: task metadata, tools, steps, failure and
recovery strategies, notes, and tags. The self-organized writer received only
the retrieval/storage envelope—agent title, task title, task description,
tags, and a free-form `memory` field—and decided what the memory itself should
contain. Neither writer received a length constraint.

## Protocol

- Model: GPT-5.6 Luna, with the same model and evaluation settings in both
  conditions.
- Construction population: a deterministic stratified 50-task sample of the
  GAIA test split: 15 Level 1, 28 Level 2, and 7 Level 3 tasks. Test labels
  remained hidden.
- Shared trajectories: each construction task was solved once. After the
  answer, the identical conversation was cloned into the two no-tool memory
  writers.
- Isolated adaptation: the structured and self-organized writers had separate
  directories, indexes, create/update decisions, and histories. Each writer
  saw only its own bank.
- Validation population: the same ordered 106 attachment-free GAIA validation
  tasks in both conditions.
- Retrieval: the normal agentic metadata selector saw each task plus the task
  title, task description, and tags from the relevant bank, then loaded at
  most three memories. Every task loaded exactly three memories.
- Validation banks were frozen. Both runs verified their complete bank digest
  unchanged at exit.
- No baseline was rerun because the primary experiment already evaluates baseline versus
  memory; the estimand here is structured versus self-organized recording.

The implementation was committed before the full experiment at `4725632`.

## Construction audit

The shared 50-task solve produced 48 successful trajectories and two
pre-branch execution failures. Every successful trajectory generated one
operation in each bank, and neither writer made a tool call.

| Measure | Structured | Self-organized |
|---|---:|---:|
| Memory events | 48 | 48 |
| Creates | 42 | 43 |
| Updates | 6 | 5 |
| Canonical memories | 42 | 43 |
| Archived versions | 6 | 5 |
| Writer tool calls | 0 | 0 |
| Writer tokens | 2,479,511 | 2,410,544 |
| Writer wall time | 589.469 s | 454.890 s |

All canonical memories in both banks had nonempty retrieval titles,
descriptions, and tags. The writers made different consolidation decisions,
which is an intended consequence of keeping the adaptive strategies isolated.

## Primary validation result

| Measure | Structured | Self-organized | Structured minus self-organized |
|---|---:|---:|---:|
| Guarded score | **73/106 (68.87%)** | 66/106 (62.26%) | **+7 tasks, +6.60 pp** |
| Raw score | **68/106 (64.15%)** | 60/106 (56.60%) | **+8 tasks, +7.55 pp** |
| Execution failures | 1 | 3 | -2 |
| Wilson 95% interval, guarded accuracy | 59.52%–76.89% | 52.76%–70.91% | — |

The deterministic answer-format guard changed seven answers in each run. It
rescued five structured answers and six self-organized answers and harmed none.
The raw-score direction is therefore the same as the guarded-score direction.

## Paired comparison

| Paired guarded outcome | Tasks |
|---|---:|
| Both correct | 57 |
| Structured correct, self-organized incorrect | **16** |
| Self-organized correct, structured incorrect | 9 |
| Both incorrect | 24 |

The two-sided exact McNemar/binomial test over the 25 discordant pairs gives
`p = 0.2295`. A paired Wald 95% interval for the +6.60-point difference is
-2.56 to +15.76 points. The point estimate favors explicit structure, but the
interval includes no difference and the single run does not reach the usual
0.05 significance threshold.

The raw paired comparison has 16 structured-only versus 8 self-organized-only
successes (`p = 0.1516`). As a reliability sensitivity analysis, removing the
four tasks with an execution error in either condition leaves 102 tasks, 14
structured-only versus 8 self-organized-only successes, a +5.88-point
difference (`p = 0.2863`). The conclusion is unchanged.

## Result by GAIA level

| Level | Structured | Self-organized | Difference | Structured-only / self-only |
|---|---:|---:|---:|---:|
| Level 1 (39) | 29 (74.36%) | 27 (69.23%) | +5.13 pp | 3 / 1 |
| Level 2 (53) | 35 (66.04%) | 34 (64.15%) | +1.89 pp | 9 / 8 |
| Level 3 (14) | **9 (64.29%)** | 5 (35.71%) | **+28.57 pp** | 4 / 0 |

Most of the net difference came from Level 3, but that stratum contains only
14 tasks. It is a useful hypothesis—that explicit steps, tools, and recovery
fields help more on difficult tasks—not a reliable subgroup conclusion from
this sample alone.

## Memory length and retrieval behavior

The lack of a length constraint did not produce length-matched banks. Luna
spontaneously wrote much shorter self-organized memories.

| Measure | Structured | Self-organized | Self-organized change |
|---|---:|---:|---:|
| Total canonical-bank characters | 142,800 | 86,893 | -39.15% |
| Mean characters per canonical memory | 3,400 | 2,021 | -40.57% |
| Median characters per canonical memory | 3,294 | 1,920 | -41.70% |
| Mean memory context loaded per validation task | 9,586 chars | 5,875 chars | -38.72% |
| Estimated loaded context tokens per task | 2,397 | 1,469 | -38.71% |

This difference is part of the treatment: when Luna was not explicitly asked
for tools, steps, and recovery fields, it chose to preserve less information.
It also means the experiment cannot attribute the accuracy difference to
headings alone. The plausible mechanism is the combined effect of explicit
field requirements, greater retained procedural detail, and the retrieval
choices induced by the resulting metadata.

Retrieval was not held fixed, because the agreed experiment evaluates each
recording strategy end-to-end. The two conditions selected the same set of
source-memory task IDs on only 18/106 validation tasks. Their mean selected-set
Jaccard similarity was 0.463. Thus, the recording instruction changed both
what was injected and which memories the normal selector considered relevant.

## Efficiency and reliability

| Validation measure | Structured | Self-organized | Self-organized change |
|---|---:|---:|---:|
| Overall recorded tokens | 8,615,417 | 7,580,018 | **-12.02%** |
| Solve tokens | 8,243,458 | 7,221,456 | **-12.40%** |
| Selection tokens | 371,959 | 358,562 | -3.60% |
| Solve tool calls | 531 | 488 | **-8.10%** |
| Summed solve time | 2,757.900 s | 2,657.365 s | -3.65% |
| Summed selection time | 241.690 s | 234.175 s | -3.11% |
| Summed total task time | 3,054.090 s | 3,211.059 s | +5.14% |
| Selector failures | 0 | 0 | no change |
| Execution failures | 1 | 3 | +2 |

The self-organized strategy was more compact and used fewer tokens and tools,
but it scored lower. Its total wall time was higher despite lower summed solve
and selection time because three missing-final-answer failures consumed long
attempts; error-path time is retained in total task time. Wall time also
includes provider, network, and web-tool variation and should remain a
secondary metric.

## Interpretation

The defensible conclusion is:

> In this single paired Luna experiment, the normal explicitly structured
> workflow-memory writer achieved 73/106 versus 66/106 for a writer that chose
> its own free-form memory content. Structured memory produced 16 gains and 9
> losses relative to self-organized memory, but the paired difference was not
> statistically significant (`p = 0.2295`). Self-organized memories were about
> 39% shorter and reduced validation tokens by 12%, suggesting an
> accuracy–compactness trade-off rather than evidence that free-form recording
> is better.

This supports retaining explicit tools, steps, and recovery fields in the
current system. It is evidence that the recording instruction matters, but it
should be presented as an exploratory ablation rather than conclusive proof
that every heading is causally necessary.

For a stricter structure-only follow-up, construct a third condition that
matches the self-organized memories to the structured memories' token budget
and uses a fixed retrieval mapping. That would separate (1) required fields,
(2) memory length, and (3) retrieval differences. Repeating the current pair
with several seeds would estimate run-to-run variance.

## Integrity and artifacts

- Structured frozen-bank digest:
  `a6a2364ee952d68fc35f6ff8586947dd1f4ff3f5960f6442c69a2d3a64153c58`
- Self-organized frozen-bank digest:
  `79c6b7de6735b4ffcac408968e2cf3a2ce1ad879fd35468749c92d37a9152b3e`
- Fixed subset manifest:
  `results/gpt_5_6_luna/test_full_luna_schema_ablation_50_build_v1_task_subset.json`
- Shared construction results:
  `results/gpt_5_6_luna/test_full_luna_schema_ablation_50_build_v1.jsonl`
- Structured validation results:
  `results/gpt_5_6_luna/validation_with_test_memory_full_luna_schema_ablation_50_structured_validation_v1.jsonl`
- Self-organized validation results:
  `results/gpt_5_6_luna/validation_with_test_memory_full_luna_schema_ablation_50_self_organized_validation_v1.jsonl`
- Reproducible analysis: `analyze_schema_ablation.py`
