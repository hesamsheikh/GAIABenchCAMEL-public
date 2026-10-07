# Ministral workflow-memory recording-strategy ablation

## Research question

Does Ministral 3B benefit from being explicitly told to record the normal
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

- Model: Ministral 3B 2512, with identical model and evaluation settings in
  both conditions.
- Construction population: the exact deterministic stratified 50-task test
  subset used for the Luna ablation: 15 Level 1, 28 Level 2, and 7 Level 3
  tasks. Test labels remained hidden.
- Shared trajectories: each construction task was solved once. Each successful
  post-answer conversation was then cloned into two no-tool memory writers.
- Isolated adaptation: the writers had separate directories, indexes,
  create/update decisions, and histories. Each writer saw only its own bank.
- Validation population: the same ordered 106 attachment-free validation
  tasks in both conditions and in the Luna ablation.
- Retrieval: the normal agentic selector saw the current task and the title,
  description, and tags from the relevant bank, then loaded at most three
  memories. In replication 1, each arm loaded three memories on 105 tasks and
  had one selector failure. In replication 2, self-organized loaded three on
  all 106 tasks; structured loaded three on 105 and had one selector failure.
- Validation banks were frozen. Each run verified its complete bank digest
  unchanged at exit.
- No baseline was rerun because the primary experiment already evaluates baseline versus
  memory. The estimand here is structured versus self-organized recording.

The complete protocol was run twice. Replication 2 used new model executions
and newly built, separately frozen banks while preserving the exact same
ordered construction subset and validation population. It was not a resume or
reuse of replication 1's banks.

## Replication overview

| Independent run | Structured | Self-organized | Self-organized advantage | Structured-only / self-only | Exact paired `p` |
|---|---:|---:|---:|---:|---:|
| Replication 1 | 18/106 (16.98%) | **24/106 (22.64%)** | +5.66 pp | 7 / 13 | 0.2632 |
| Replication 2 | 15/106 (14.15%) | **27/106 (25.47%)** | **+11.32 pp** | 4 / 16 | **0.0118** |

Both independent bank builds therefore give the same directional result. The
second run is individually significant under the pre-specified task-paired
test; the first has the same sign but is inconclusive. Descriptively across
the two passes, structured recording solved 33/212 task-runs (15.57%) and
self-organized recording solved 51/212 (24.06%), an 8.49-point difference.
That aggregate is not treated as a new independent-task test because the same
106 task identities occur in both replications.

## Replication 1 construction audit

The shared 50-task solve produced 40 successful trajectories and 10
pre-branch execution failures. Every successful trajectory produced one
operation in each bank, and neither writer made a tool call.

| Measure | Structured | Self-organized |
|---|---:|---:|
| Memory events | 40 | 40 |
| Creates | 36 | 39 |
| Updates | 4 | 1 |
| Canonical memories | 36 | 39 |
| Archived versions | 4 | 1 |
| Writer tool calls | 0 | 0 |
| Writer tokens | 1,613,292 | 1,578,016 |
| Writer wall time | 290.123 s | 192.592 s |

All canonical files in both banks had nonempty retrieval titles,
descriptions, and tags. The different create/update choices are part of the
isolated adaptive strategies, not cross-bank contamination.

## Replication 1 validation result

| Measure | Structured | Self-organized | Structured minus self-organized |
|---|---:|---:|---:|
| Guarded score | 18/106 (16.98%) | **24/106 (22.64%)** | **-6 tasks, -5.66 pp** |
| Raw score | 18/106 (16.98%) | **24/106 (22.64%)** | **-6 tasks, -5.66 pp** |
| Execution/selection failures | 19 | 18 | +1 |
| Wilson 95% interval | 11.02%–25.25% | 15.71%–31.48% | — |

The deterministic answer-format guard changed two structured answers and one
self-organized answer but rescued or harmed none, so guarded and raw scores
are identical.

## Replication 1 paired comparison

| Paired outcome | Tasks |
|---|---:|
| Both correct | 11 |
| Structured correct, self-organized incorrect | 7 |
| Self-organized correct, structured incorrect | **13** |
| Both incorrect | 75 |

The two-sided exact McNemar/binomial test over the 20 discordant pairs gives
`p = 0.2632`. A paired 95% interval for the structured-minus-self-organized
difference is -13.86 to +2.54 percentage points. The point estimate favors
self-organized recording, but the interval includes no difference.

Removing every task with an execution or selection error in either condition
leaves 76 tasks. The clean subset has 6 structured-only versus 11
self-organized-only successes: a 6.58-point self-organized advantage
(`p = 0.3323`). The direction therefore does not depend only on malformed
outputs, although the reduced sample remains inconclusive.

## Replication 1 result by GAIA level

| Level | Structured | Self-organized | Difference | Structured-only / self-only |
|---|---:|---:|---:|---:|
| Level 1 (39) | 9 (23.08%) | 10 (25.64%) | -2.56 pp | 3 / 4 |
| Level 2 (53) | 9 (16.98%) | 12 (22.64%) | -5.66 pp | 4 / 7 |
| Level 3 (14) | 0 (0.00%) | 2 (14.29%) | -14.29 pp | 0 / 2 |

All three strata point in the same direction, but the model's low overall
accuracy creates a strong floor effect. In particular, the Level 3 result is
only two tasks and is not a reliable subgroup estimate.

## Replication 1 memory length and retrieval behavior

Ministral, like Luna, spontaneously wrote shorter memories when it was not
required to enumerate tools, steps, and recovery fields.

| Measure | Structured | Self-organized | Self-organized change |
|---|---:|---:|---:|
| Total canonical-bank characters | 118,914 | 78,948 | -33.61% |
| Total canonical-bank words | 15,341 | 8,699 | -43.30% |
| Mean characters per memory | 3,303 | 2,024 | -38.72% |
| Median characters per memory | 3,149 | 1,922 | -38.96% |
| Mean loaded context per validation task | 8,635 chars | 4,976 chars | -42.37% |
| Estimated loaded context tokens per task | 2,159 | 1,244 | -42.36% |

The recording instruction also changed retrieval. The arms selected the same
set of source-task memories on only 3/106 tasks, with a mean selected-set
Jaccard similarity of 0.255. This is expected in the agreed end-to-end design,
but it prevents attributing the result to formatting alone. The treatment
combines required fields, retained detail, consolidation, metadata, retrieval,
and injected context.

## Replication 1 efficiency and reliability

| Validation measure | Structured | Self-organized | Self-organized change |
|---|---:|---:|---:|
| Overall recorded tokens | 6,013,895 | 4,960,303 | **-17.52%** |
| Solve tokens | 5,733,808 | 4,635,183 | **-19.16%** |
| Selection tokens | 283,345 | 328,798 | +16.04% |
| Solve tool calls | 429 | 419 | -2.33% |
| Summed solve time | 1,312.947 s | 1,244.168 s | -5.24% |
| Summed selection time | 52.939 s | 58.278 s | +10.08% |
| Summed total task time | 2,106.737 s | 2,097.262 s | -0.45% |
| Selector failures | 1 | 1 | no change |
| Execution/selection failures | 19 | 18 | -1 |

The self-organized strategy both scored higher and used fewer tokens, although
its end-to-end time was effectively unchanged. Runtime is secondary because
provider, network, web-tool, and error-path variation are included.

## Replication 2 audit and result

The new shared construction pass produced 41 successful trajectories and 9
pre-branch failures. Structured writing completed all 41 events. The
self-organized writer completed 40; on one trajectory it returned an object
where the free-form memory field required a string, so that branch was counted
as a writer failure and was not retried. This treatment-specific reliability
event is retained rather than silently repaired.

| Construction measure | Structured | Self-organized |
|---|---:|---:|
| Attempted memory events | 41 | 41 |
| Completed memory events | 41 | 40 |
| Writer failures | 0 | 1 |
| Creates | 39 | 39 |
| Updates | 2 | 1 |
| Canonical memories | 39 | 39 |
| Archived versions | 2 | 1 |
| Writer tool calls | 0 | 0 |
| Writer tokens | 1,521,722 | 1,468,536 |
| Writer wall time | 269.298 s | 193.662 s |

One shared construction task made 353 solve-tool calls and ended in provider
context overflow before branching. It affects construction efficiency but
cannot favor either writer because neither branch was created. All canonical
memories in both completed banks had nonempty retrieval metadata.

| Validation measure | Structured | Self-organized | Structured minus self-organized |
|---|---:|---:|---:|
| Guarded score | 15/106 (14.15%) | **27/106 (25.47%)** | **-12 tasks, -11.32 pp** |
| Raw score | 15/106 (14.15%) | **27/106 (25.47%)** | **-12 tasks, -11.32 pp** |
| Execution/selection failures | 20 | 18 | +2 |
| Wilson 95% interval | 8.77%–22.04% | 18.14%–34.52% | — |

The paired outcomes were 11 both correct, 4 structured-only, 16
self-organized-only, and 75 both wrong. The exact two-sided McNemar/binomial
test gives `p = 0.0118`; the paired structured-minus-self-organized 95%
interval is -19.30 to -3.34 percentage points. Removing all 32 tasks with an
execution or selection error in either arm leaves 74 clean tasks. The same 4
versus 16 discordant successes remain, giving a 16.22-point self-organized
advantage (`p = 0.0118`). The result is therefore not produced by assigning
zero to malformed executions.

| Level | Structured | Self-organized | Difference | Structured-only / self-only |
|---|---:|---:|---:|---:|
| Level 1 (39) | 6 (15.38%) | 9 (23.08%) | -7.69 pp | 2 / 5 |
| Level 2 (53) | 8 (15.09%) | 15 (28.30%) | -13.21 pp | 2 / 9 |
| Level 3 (14) | 1 (7.14%) | 3 (21.43%) | -14.29 pp | 0 / 2 |

The self-organized bank again became substantially shorter: 80,285 versus
127,629 canonical characters (-37.10%) and 8,912 versus 16,307 words
(-45.35%). Mean injected context fell from 8,189 to 4,593 characters per task
(-43.92%). The selected source sets were identical on 7/106 tasks and their
mean Jaccard similarity was 0.285, again confirming that this is an
end-to-end recording-policy treatment rather than a headings-only mechanism
test.

Unlike replication 1, shorter self-organized context did not reduce total
recorded validation tokens: 5,632,927 versus 5,303,165 (+6.22%). It also made
508 versus 393 solve-tool calls (+29.26%), while summed end-to-end task time
was nearly unchanged (1,820.832 versus 1,838.020 seconds, -0.94%). Thus the
accuracy direction replicated, but the token-efficiency direction did not.
Provider, web-tool, and error-path variation make runtime secondary.

## Across-run stability

Exact per-task correctness is stochastic even when the aggregate direction is
stable. Between replications, the structured arm had 10 run-1-only and 7
run-2-only successes; the self-organized arm had 9 run-1-only and 12
run-2-only successes. This is why the evidence should be stated as a replicated
directional treatment effect, not as deterministic task-level behavior.

## Cross-model interpretation

Both Ministral runs reverse the Luna point estimate:

| Model | Structured | Self-organized | Structured advantage | Paired flips (structured / self) | Exact `p` |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | **73/106 (68.87%)** | 66/106 (62.26%) | +6.60 pp | 16 / 9 | 0.2295 |
| Ministral 3B, replication 1 | 18/106 (16.98%) | **24/106 (22.64%)** | -5.66 pp | 7 / 13 | 0.2632 |
| Ministral 3B, replication 2 | 15/106 (14.15%) | **27/106 (25.47%)** | **-11.32 pp** | 4 / 16 | **0.0118** |
| Gemini 3.7 Flash | 67/106 (63.21%) | **75/106 (70.75%)** | -7.55 pp | 7 / 15 | 0.1338 |

The second Ministral replication crosses the usual 0.05 threshold and the
first Ministral run agrees in direction. The defensible conclusion is still
not that either schema is universally superior: Luna points toward explicit
procedural fields, whereas two Ministral runs and Gemini point toward letting
the model choose its retained contents. Recording instructions materially
change what models preserve and later retrieve, and the downstream direction
is model-dependent.

This strengthens the claim that memory format and structure are design
variables that must be evaluated with the target model. It provides replicated
evidence against treating the current explicit schema as inherently optimal
for Ministral, but it does not isolate which changed component caused the
gain. A stricter mechanism study would length-match memories and hold
retrieval mappings fixed.

## Integrity and artifacts

- Structured frozen-bank digest:
  `b85a09630a8921b5afe282a07c446c3499cedb3f3abd076b307c656c5973b53e`
- Self-organized frozen-bank digest:
  `72da5a66f380dbf37b3d2769d26a0faa48788eca98c4f1c16691161907d3755e`
- Fixed subset manifest:
  `results/ministral_3b_2512/test_full_ministral_schema_ablation_50_build_v1_task_subset.json`
- Shared construction results:
  `results/ministral_3b_2512/test_full_ministral_schema_ablation_50_build_v1.jsonl`
- Structured validation results:
  `results/ministral_3b_2512/validation_with_test_memory_full_ministral_schema_ablation_50_structured_validation_v1.jsonl`
- Self-organized validation results:
  `results/ministral_3b_2512/validation_with_test_memory_full_ministral_schema_ablation_50_self_organized_validation_v1.jsonl`
- Reproducible analysis: `analyze_schema_ablation.py`

Replication 2 artifacts:

- Structured frozen-bank digest:
  `79920ae199e0005ab8c0e947cbca5c08cda2d126f54d719579f027e6f1f9c917`
- Self-organized frozen-bank digest:
  `06f4607e7281daab03a9d0cef8d80c14cb033f2c0ecd46407b14ebfa72fba017`
- Fixed subset manifest:
  `results/ministral_3b_2512/test_full_ministral_schema_ablation_50_build_v2_task_subset.json`
- Shared construction results:
  `results/ministral_3b_2512/test_full_ministral_schema_ablation_50_build_v2.jsonl`
- Structured validation results:
  `results/ministral_3b_2512/validation_with_test_memory_full_ministral_schema_ablation_50_structured_validation_v2.jsonl`
- Self-organized validation results:
  `results/ministral_3b_2512/validation_with_test_memory_full_ministral_schema_ablation_50_self_organized_validation_v2.jsonl`
