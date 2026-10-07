# Gemini workflow-memory recording-strategy ablation

## Research question

Does Gemini 3.7 Flash benefit from being explicitly told to record the normal
workflow-memory fields, or from choosing the contents and organization of the
memory body itself?

The structured writer recorded the normal schema: task metadata, tools, steps,
failure and recovery strategies, notes, and tags. The self-organized writer
received only the retrieval/storage envelope—agent title, task title, task
description, tags, and a free-form `memory` field—and decided what was worth
retaining. Neither writer received a length constraint.

This is an end-to-end recording-strategy ablation, not a headings-only
ablation. Allowing the writer to choose its own contents can change memory
length, consolidation decisions, retrieval metadata, selected memories, and
the injected content.

## Protocol

- Model: Gemini 3.7 Flash, with identical model and evaluation settings in
  both conditions.
- Construction population: the exact deterministic stratified 50-task test
  subset used for the Luna and Ministral ablations: 15 Level 1, 28 Level 2,
  and 7 Level 3 tasks. Test labels remained hidden.
- Shared trajectories: each construction task was solved once. Each successful
  post-answer conversation was then cloned into two no-tool memory writers.
- Isolated adaptation: the writers had separate directories, indexes,
  create/update decisions, and histories. Each writer saw only its own bank.
- Validation population: the same ordered 106 attachment-free validation
  tasks in both conditions and in the earlier ablations.
- Retrieval: the normal agentic selector saw the current task and the title,
  description, and tags from the relevant bank, then loaded at most three
  memories. Each arm loaded three memories on 105 tasks; one selector failure
  loaded none.
- Validation banks were frozen. Each run verified its complete bank digest
  unchanged at exit.
- No baseline was rerun because the primary experiment already evaluates baseline versus
  memory. The estimand here is structured versus self-organized recording.

## Construction audit

The shared 50-task solve produced 34 successful trajectories and 16
pre-branch execution failures. Every successful trajectory produced one
operation in each bank, and neither writer made a tool call.

| Measure | Structured | Self-organized |
|---|---:|---:|
| Memory events | 34 | 34 |
| Creates | 33 | 31 |
| Updates | 1 | 3 |
| Canonical memories | 33 | 31 |
| Archived versions | 1 | 3 |
| Writer tool calls | 0 | 0 |
| Writer tokens | 1,334,063 | 1,301,820 |
| Writer wall time | 317.671 s | 292.415 s |

All canonical files in both banks had nonempty retrieval titles,
descriptions, and tags. The different create/update choices are part of the
isolated adaptive strategies, not cross-bank contamination. Gemini produced
fewer usable construction trajectories than Luna (48) or Ministral (40), so
this comparison is based on a smaller memory bank despite using the same 50
construction tasks.

## Primary validation result

| Measure | Structured | Self-organized | Structured minus self-organized |
|---|---:|---:|---:|
| Guarded score | 67/106 (63.21%) | **75/106 (70.75%)** | **-8 tasks, -7.55 pp** |
| Raw score | 67/106 (63.21%) | **75/106 (70.75%)** | **-8 tasks, -7.55 pp** |
| Execution/selection failures | 13 | 13 | 0 |
| Wilson 95% interval | 53.72%–71.78% | 61.49%–78.57% | — |

The deterministic answer-format guard changed one structured answer and no
self-organized answers, but it rescued or harmed none. Guarded and raw scores
are therefore identical.

## Paired comparison

| Paired outcome | Tasks |
|---|---:|
| Both correct | 60 |
| Structured correct, self-organized incorrect | 7 |
| Self-organized correct, structured incorrect | **15** |
| Both incorrect | 24 |

The two-sided exact McNemar/binomial test over the 22 discordant pairs gives
`p = 0.1338`. A paired 95% interval for the structured-minus-self-organized
difference is -16.10 to +1.01 percentage points. The point estimate favors
self-organized recording, but the interval includes no difference.

Each arm had 13 error rows, with six errors shared by the same tasks. Removing
every task with an execution or selection error in either condition leaves 86
tasks. The clean subset has 3 structured-only versus 9 self-organized-only
successes: a 6.98-point self-organized advantage (`p = 0.1460`; paired 95%
interval -14.73 to +0.78 points). The direction is therefore not explained by
unequal failure counts, although the reduced comparison remains inconclusive.

## Result by GAIA level

| Level | Structured | Self-organized | Difference | Structured-only / self-only |
|---|---:|---:|---:|---:|
| Level 1 (39) | 31 (79.49%) | 32 (82.05%) | -2.56 pp | 3 / 4 |
| Level 2 (53) | 30 (56.60%) | 35 (66.04%) | -9.43 pp | 3 / 8 |
| Level 3 (14) | 6 (42.86%) | 8 (57.14%) | -14.29 pp | 1 / 3 |

All three strata favor self-organized recording, but Level 3 contains only 14
tasks. The consistency is descriptive evidence, not a reliable subgroup
effect from one run.

## Memory length and retrieval behavior

Gemini spontaneously wrote somewhat shorter memories when it was not required
to enumerate tools, steps, and recovery fields.

| Measure | Structured | Self-organized | Self-organized change |
|---|---:|---:|---:|
| Total canonical-bank characters | 88,692 | 75,483 | -14.89% |
| Total canonical-bank words | 11,310 | 8,767 | -22.48% |
| Mean characters per memory | 2,688 | 2,435 | -9.40% |
| Median characters per memory | 2,721 | 2,342 | -13.93% |
| Mean loaded context per validation task | 6,988 chars | 6,841 chars | -2.11% |
| Estimated loaded context tokens per task | 1,747 | 1,710 | -2.11% |

The complete self-organized bank was 14.9% smaller by characters, but its
average injected context was only 2.1% smaller. Gemini therefore provides a
less length-confounded comparison than Luna or Ministral: the selector largely
equalized the amount of context actually loaded even though the memory bodies
and source choices differed.

The recording instruction still changed retrieval. Both arms chose the same
set of source-task memories on 26/106 tasks, and their mean selected-set
Jaccard similarity was 0.572. On those 26 same-source tasks, the paired result
was 19 both correct, 3 self-organized-only, 0 structured-only, and 4 both
wrong. The same 3-to-0 result remains after excluding error rows (25 tasks).
On the 80 tasks with different source sets, the flips were 7 structured-only
and 12 self-organized-only.

The same-source result shows that the overall Gemini advantage is not solely
caused by retrieving different source tasks: when source identities matched,
the differently written bodies could still produce different outcomes. It
does not isolate Markdown formatting, because retained content, organization,
and consolidation still differ within those bodies.

## Efficiency and reliability

| Validation measure | Structured | Self-organized | Self-organized change |
|---|---:|---:|---:|
| Overall recorded tokens | 9,421,975 | 10,454,765 | **+10.96%** |
| Solve tokens | 9,149,832 | 10,189,543 | **+11.36%** |
| Selection tokens | 272,143 | 265,222 | -2.54% |
| Solve tool calls | 551 | 527 | -4.36% |
| Summed solve time | 2,804.000 s | 2,244.516 s | -19.95% |
| Summed selection time | 275.797 s | 259.481 s | -5.92% |
| Summed total task time | 3,755.310 s | 3,139.203 s | -16.41% |
| Selector failures | 1 | 1 | no change |
| Execution/selection failures | 13 | 13 | no change |

Self-organized recording scored higher, used fewer tools, and finished faster,
but used 11.0% more recorded validation tokens. It should not be described as
token-efficient for Gemini. Provider token accounting includes cached prompt
tokens, and runtime includes provider, network, web-tool, and error-path
variation; both remain secondary to the paired accuracy result.

During construction, the self-organized writer used 2.42% fewer tokens and
7.95% less time than the structured writer. Those writer costs are separate
from the validation totals above.

## Cross-model interpretation

The three matched replications do not identify a universally superior
recording strategy:

| Model | Structured | Self-organized | Structured advantage | Paired flips (structured / self) | Exact `p` |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | **73/106 (68.87%)** | 66/106 (62.26%) | +6.60 pp | 16 / 9 | 0.2295 |
| Ministral 3B | 18/106 (16.98%) | **24/106 (22.64%)** | -5.66 pp | 7 / 13 | 0.2632 |
| Gemini 3.7 Flash | 67/106 (63.21%) | **75/106 (70.75%)** | -7.55 pp | 7 / 15 | 0.1338 |

None of the individual differences reaches the usual 0.05 threshold, and
there is only one paired pass per model. Luna favors explicit procedural
fields, while Ministral and Gemini favor choosing their own contents. The
Gemini result is especially useful because loaded context length was nearly
matched and self-organized validation actually consumed more recorded tokens:
its accuracy gain cannot be explained as a simple shorter-prompt benefit.

The defensible conclusion is that the memory-writing instruction
materially changes downstream behavior, but its best form is model-dependent.
The current explicit schema is not inherently optimal across models. More
seeds are needed before making model-specific prescriptions, and the opposing
directions make an unqualified pooled effect inappropriate. A stricter
mechanism study would length-match memories and fix retrieval mappings while
changing only the required body fields.

## Integrity and artifacts

- Structured frozen-bank digest:
  `db2b10b2cdaa670301e20831e6c4872cf5ddcfe2fa987fdc49bddfc577e8b625`
- Self-organized frozen-bank digest:
  `bb81447641e563307435caba3ca9f872333e20893bba245c92a5a9c3c9e45707`
- Fixed subset manifest:
  `results/gemini_3_7_flash/test_full_gemini_schema_ablation_50_build_v1_task_subset.json`
- Shared construction results:
  `results/gemini_3_7_flash/test_full_gemini_schema_ablation_50_build_v1.jsonl`
- Structured validation results:
  `results/gemini_3_7_flash/validation_with_test_memory_full_gemini_schema_ablation_50_structured_validation_v1.jsonl`
- Self-organized validation results:
  `results/gemini_3_7_flash/validation_with_test_memory_full_gemini_schema_ablation_50_self_organized_validation_v1.jsonl`
- Reproducible analysis: `analyze_schema_ablation.py`

The three full phases took approximately 2 h 52 min in total, excluding the
one-task smoke test and bank-audit overhead.
