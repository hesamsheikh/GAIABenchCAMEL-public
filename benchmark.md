# Workflow-memory benchmark results

Consolidated on 2026-08-24 from the persisted JSONL results and canonical
workflow banks in this repository.

## Executive summary

The current evidence does **not** show a universal workflow-memory benefit.
It shows a strong, model-dependent effect:

- **GPT-5.6 Luna:** 56/106 to 72/106, a **+15.09 percentage-point** gain.
  Nineteen tasks changed from wrong to correct and only three changed from
  correct to wrong. This is the only clearly positive statistical result in
  the present single-run experiment (exact paired test, `p = 0.00086`).
- **Gemini 3.7 Flash:** 73/106 to 74/106, essentially neutral (**+0.94 pp**).
- **DeepSeek V4 Flash:** after removing its 19 hard memory-selector failures
  from both matched conditions, 55/87 to 60/87 (**+5.75 pp**). It also used
  fewer tokens and tools on comparable successful tasks, although the accuracy
  gain is not statistically conclusive.
- **Ox Alpha:** 59/106 to 56/106 (**-2.83 pp**) while using substantially more
  tokens on comparable successful tasks.
- **Ministral 3B:** 16/106 to 18/106 (**+1.89 pp**). It used far fewer tools and
  finished faster, but memory-selection overhead erased almost all of the
  solve-token reduction.

The main result is therefore: **workflow memory can provide a
large improvement, but the effect depends on how a model writes, selects, and
uses memories. Smaller size by itself does not guarantee a larger gain.**

### Recording-strategy ablation

A subsequent paired ablation tested whether Luna should keep the normal
explicit workflow schema or choose its own memory content. A shared stratified
50-task test run branched each successful post-answer trajectory into two
isolated adaptive banks. The normal writer recorded tools, steps, recovery
strategies, notes, and tags; the self-organized writer received only retrieval
metadata plus a free-form memory field. Neither writer had a length limit.

On the same 106 validation tasks, the structured bank scored **73/106
(68.87%)** and the self-organized bank scored **66/106 (62.26%)**, a
**+6.60-point structured advantage**. Structured memory uniquely solved 16
tasks, while self-organized memory uniquely solved 9; the exact paired test was
`p = 0.2295`, so the direction is favorable but not conclusive. The
self-organized bank was 39% shorter and used 12% fewer validation tokens,
indicating an accuracy–compactness trade-off. See
[`schema_ablation_report.md`](schema_ablation_report.md) for the full protocol,
integrity audit, level results, retrieval overlap, and efficiency analysis.

The identical experiment was then replicated twice with **Ministral 3B** on
the same ordered construction and validation subsets, using a newly built pair
of isolated banks each time. Both runs reversed Luna's direction. Replication
1 scored 18/106 structured versus 24/106 self-organized (+5.66 pp for
self-organized; flips 7/13, `p = 0.2632`). Replication 2 scored **15/106
structured versus 27/106 self-organized**, an **11.32-point self-organized
advantage** with 4/16 paired flips (**`p = 0.0118`**). The clean 74-task
intersection in replication 2 retained the same flips and a 16.22-point
advantage. Self-organized memories again loaded 43.9% less context, although
replication 2 used 6.2% more validation tokens, so the accuracy direction—but
not the token-efficiency direction—replicated.

The same pair was also run with **Gemini 3.7 Flash**. Structured memory scored
**67/106 (63.21%)**, while self-organized memory scored **75/106 (70.75%)**,
a **7.55-point self-organized advantage**. The paired flips were 7 versus 15
(`p = 0.1338`). This result is less confounded by memory length: the
self-organized bank was 14.9% smaller, but it loaded only 2.1% less context
and used 11.0% more recorded validation tokens. On the 26 tasks where both
arms retrieved the same source memories, the flips were 0 structured-only
versus 3 self-organized-only.

The second Ministral replication is individually significant and agrees with
the first Ministral run's direction. Luna's point estimate favors explicit
structure, whereas both Ministral runs and Gemini favor self-organized
contents. The recording instruction is therefore a model-dependent design
choice rather than a universally optimal template. See
[`ministral_schema_ablation_report.md`](ministral_schema_ablation_report.md)
and [`gemini_schema_ablation_report.md`](gemini_schema_ablation_report.md) for
the replication audits and cross-model interpretation.

## Experimental protocol

### Data split

- Memory construction used all **229 GAIA test tasks**. Their labels were
  hidden, so no test accuracy is reported or used while writing memories.
- Evaluation used the fixed **106-task attachment-free, text-compatible GAIA
  validation population**.
- Attachment retrieval was completely disabled.
- Every baseline solved without reading workflow memories. The Gemini,
  DeepSeek, Ox, and Ministral baselines also disabled writing. The historical
  Luna baseline wrote a reflection only after each answer; this could not
  affect that task's answer, and its reflection time is excluded from the
  efficiency comparison below.
- The memory condition read the model's own frozen test-built memory bank and
  did not modify it during validation.

This intentionally reverses the usual train/test use: unlabelled, real-world-
like test tasks supply experiences, while validation supplies the labels for a
controlled baseline-versus-memory comparison. The results are consequently
an internal workflow-memory experiment, **not an official GAIA leaderboard
score**.

### Memory behavior

After solving a test task, the same model performed an outcome-blind
reflection. It saw the available canonical memory filenames and descriptions,
then agentically chose to create a new memory or update a selected existing
memory. An update replaced the canonical file while archiving its previous
version under `_history/`.

For validation, a separate no-tool selection phase saw the current question
plus memory metadata and selected up to three memories. The selection exchange
was then discarded; the task-solving context retained only the selected memory
contents, wrapped as advisory recollections from past tasks whose relevance
and current correctness were not guaranteed.

### Controlled settings

- Reasoning effort: `low`
- Temperature: `0.0` where supported; omitted for Luna because its pinned
  endpoint did not expose temperature
- Seed: `42`
- Maximum output tokens per response: `8,192`
- Agent context limit: `200,000`
- Agent step timeout: `600 s`
- Tool timeout: `120 s`
- Maximum agent iterations: `30`
- Automatic context summarization: disabled
- Malformed JSON/final protocol failure: counted as failure, with no retry
- Provider/model/protocol failures: not filtered out. The sole exception is
  DeepSeek's 19 hard memory-selector failures, whose task IDs are removed from
  both DeepSeek conditions, leaving a matched 87-task comparison

The vendored CAMEL package is `camel-ai 0.2.83a1`, imported from upstream
commit `9c548266cc5aed567bf9abccb4800832070dd916`. See `CAMEL_PIN.md` for the full
pin. The historical result rows contain model/provider configuration and
memory-bank digests, but do not contain a parent-repository commit. The parent
repository was at `a0ac738eab2bde2e0cf699a28d265ddb569442c6` when this report was
generated, with uncommitted experiment changes; the exact parent commit of
each earlier run therefore cannot be recovered from the artifacts alone.

## Validation accuracy

The primary score is the deterministic format-guarded GAIA score. Raw score is
shown separately so the formatting guard's effect remains visible.

| Model | Baseline guarded (raw) | Memory guarded (raw) | Guarded change | Relative change | Errors, baseline -> memory |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | 56/106, 52.83% (47) | 72/106, 67.92% (69) | **+16, +15.09 pp** | +28.57% | 2 -> 1 |
| Gemini 3.7 Flash | 73/106, 68.87% (73) | 74/106, 69.81% (74) | +1, +0.94 pp | +1.37% | 11 -> 13 |
| DeepSeek V4 Flash (selector-success subset) | 55/87, 63.22% (53) | 60/87, 68.97% (59) | **+5, +5.75 pp** | +9.09% | 20 -> 11 |
| Ox Alpha | 59/106, 55.66% (57) | 56/106, 52.83% (56) | -3, -2.83 pp | -5.08% | 7 -> 13 |
| Ministral 3B | 16/106, 15.09% (16) | 18/106, 16.98% (18) | +2, +1.89 pp | +12.50% | 23 -> 19 |

### Paired task changes

| Model | Both correct | Memory gain: wrong -> correct | Memory loss: correct -> wrong | Both wrong | Exact paired `p` |
|---|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | 53 | **19** | 3 | 31 | **0.00086** |
| Gemini 3.7 Flash | 67 | 7 | 6 | 26 | 1.00000 |
| DeepSeek V4 Flash (selector-success subset) | 50 | **10** | 5 | 22 | 0.30176 |
| Ox Alpha | 44 | 12 | 15 | 35 | 0.70111 |
| Ministral 3B | 8 | 10 | 8 | 80 | 0.81453 |

The `p` values are two-sided exact McNemar/binomial tests over discordant
pairs. They are descriptive because there was one full pass per condition and
no correction for testing multiple models. Only Luna provides convincing
evidence of an improvement in this experiment. DeepSeek's filtered direction
is positive but inconclusive.

For transparency, the unfiltered DeepSeek intention-to-treat result is 68/106
at baseline versus 60/106 with memory. All 19 excluded rows are tasks where the
selector returned no usable selection and the task solver was never run; the
matched baseline solved 13 of those tasks correctly. Counting them as memory
failures therefore changes DeepSeek's result to -7.55 pp. The primary DeepSeek
analysis in this report instead measures outcomes when memory loading actually
ran successfully, as requested.

### Baseline-only small-model result

LFM 2.5 2.6B Free completed a baseline but has no matched memory condition. It
scored **36/106 (33.96%)**, with 13 error rows, 841 task-solving tool calls,
13,706,446 recorded solve tokens, and 7,870.665 seconds of summed task time.
It is excluded from all baseline-versus-memory deltas.

## Memory-bank construction

No accuracy is shown for this phase because test labels were hidden.

| Model bank | Test rows | Successful solves | Errors | Creates | Updates | No saved operation | Canonical memories | Archived versions | Reflection/selection tokens | Post-task tool calls | Total task wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | 229 | 218 | 11 | 128 | 90 | 11 | 128 | 90 | not instrumented | not instrumented | 11,280.869 s (3:08:01) |
| Gemini 3.7 Flash | 229 | 167 | 62 | 52 | **115** | 62 | 52 | 115 | not instrumented | not instrumented | 10,552.509 s (2:55:53) |
| DeepSeek V4 Flash | 229 | 151 | 78 | 115 | 34 | 80 | 115 | 34 | 15,958,209 | **31** | 35,656.694 s (9:54:17) |
| Ox Alpha | 229 | 169 | 60 | 114 | 51 | 64 | 114 | 51 | 9,317,035 | 0 | 32,737.750 s (9:05:38) |
| Ministral 3B | 229 | 180 | 49 | **153** | 24 | 52 | 153 | 24 | 10,263,346 | 0 | 9,845.917 s (2:44:06) |

`No saved operation` can be larger than `Errors`: DeepSeek completed two,
Ox Alpha four, and Ministral three tasks without successfully creating or
updating a memory.

The create/update distributions are themselves experimental observations.
Gemini updated existing memories for 115/167 successful reflections (68.9%),
producing the smallest and most consolidated bank. Luna updated 90/218
(41.3%). DeepSeek, Ox, and Ministral were progressively more create-heavy.
This did **not** translate monotonically into retrieval benefit: Gemini's
highly consolidated bank produced only a +0.94 pp validation change, while
Luna's larger bank produced +15.09 pp.

The DeepSeek bank was built before strict no-tool post-task isolation was
enforced. Its reflection phase made 31 tool calls and produced scratch
artifacts, so that bank has a protocol caveat. The final Ox and Ministral
builds made zero post-task tool calls. Older Luna and Gemini builds predate the
phase telemetry needed to prove the same property from counters.

Qualitative audits also found instances of over-merging in the Gemini and
Ministral banks: unrelated task experience was sometimes appended to a broad
existing workflow. Updates are therefore not intrinsically better than
creates; future analyses should measure bank size, update rate, memory length,
selection concentration, and downstream effect together.

## Agentic memory selection

| Model | Bank size | Agent selections | Fallbacks / failures | Tasks loading 3 memories | Distinct selected sets | Distinct memories used | Selector tool calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | 128 | 106/106 | 0 / 0 | 106 | 101 | 95 | 0 |
| Gemini 3.7 Flash | 52 | 105/106 | 1 filename-order fallback / 0 hard failures | 106 | 79 | 49 | 0 |
| DeepSeek V4 Flash | 115 | 87/106 | 0 / **19** | 87 | 83, including the empty set | 87 | 0 |
| Ox Alpha | 114 | 106/106 | 0 / 0 | 106 | 91 | 95 | 0 |
| Ministral 3B | 153 | 106/106 | 0 / 0 | 106 | 88 | 67 | 0 |

These distributions confirm that the primary memory runs were agentic; they
did not deterministically load the first three files. DeepSeek is the major
exception in reliability: 19 selector failures resulted in no loaded memory
for those tasks. Those task IDs are documented here but excluded from both
sides of DeepSeek's scored and full-run efficiency comparisons.

## Tool-call and runtime changes

These totals cover all 106 attempted validation tasks for Luna, Gemini, Ox,
and Ministral. DeepSeek uses the matched 87-task selector-success population.
Memory-condition wall time includes selection overhead. Selector tool calls
were zero, so the tool totals are task-solving calls.

| Model | Tool calls, baseline -> memory | Tool-call change | Wall time, baseline -> memory | Wall-time change |
|---|---:|---:|---:|---:|
| GPT-5.6 Luna | 474 -> 489 | +3.16% | 2,511.310 -> 2,964.310 s | **+18.04%** |
| Gemini 3.7 Flash | 575 -> 531 | **-7.65%** | 3,019.108 -> 2,806.266 s | **-7.05%** |
| DeepSeek V4 Flash (87 tasks) | 521 -> 509 | **-2.30%** | 8,644.800 -> 7,002.200 s | **-19.00%** |
| Ox Alpha | 312 -> 316 | +1.28% | 7,925.561 -> 7,501.925 s | -5.35% |
| Ministral 3B | 693 -> 444 | **-35.93%** | 2,530.041 -> 2,062.944 s | **-18.46%** |

Wall-clock changes include provider latency, network latency, and remote-tool
latency. They should not be interpreted as model-compute measurements. An
error can also terminate early, so runtime remains secondary to accuracy.

For Luna, the baseline figure is solve time only. The saved baseline total was
3,267.830 seconds, but 755.966 seconds came from post-answer validation-memory
writing that is not part of the no-memory solving condition. The memory figure
includes its 272.175-second agentic selection phase plus runner overhead. Luna
therefore improved accuracy strongly but did **not** demonstrate an end-to-end
speed improvement in the comparable accounting.

## Token-efficiency comparison

Reliable phase-level token telemetry is available for DeepSeek, Ox Alpha, and
Ministral. Luna and Gemini cannot be included without rerunning both conditions
with the newer instrumentation.

To avoid treating early failures as artificially cheap, the table below uses
the **paired common-success subset**: a task is included only when both its
baseline and memory run completed without an error. `Memory end-to-end` equals
task-solving tokens plus agentic-selection tokens. It excludes the one-time
cost of constructing the reusable bank.

| Model | Comparable tasks | Baseline solve tokens | Memory solve tokens | Selection tokens | Memory end-to-end tokens | Solve-token change | End-to-end token change | Solve tool-call change | End-to-end wall-time change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash | 63 | 10,412,664 | 7,919,814 | 696,976 | 8,616,790 | **-23.94%** | **-17.25%** | **-19.52%** (461 -> 371) | **-14.94%** |
| Ox Alpha | 91 | 3,185,930 | 6,813,776 | 916,907 | 7,730,683 | **+113.87%** | **+142.65%** | +22.62% (252 -> 309) | +3.08% |
| Ministral 3B | 72 | 5,340,336 | 4,424,943 | 986,229 | 5,411,172 | **-17.14%** | +1.33% | **-42.58%** (613 -> 352) | **-21.66%** |

Additional paired observations:

- DeepSeek's solve wall time fell 33.05%, but selection reduced the end-to-end
  saving to 14.94%. Its solve model calls fell 20.05%; after adding one
  selector call per task, end-to-end model calls fell 5.19%.
- Ox Alpha's solve wall time fell 4.78%, but selection made end-to-end time
  3.08% slower. Its solve model calls rose 20.45%, and end-to-end calls rose
  49.52% after selection.
- Ministral's solve wall time fell 25.98% and end-to-end time fell 21.66%.
  Its solve model calls fell 28.27%; including selection, calls still fell
  13.31%.

Thus, **fewer tool calls do not necessarily mean fewer tokens or higher
accuracy**. On its common-success subset, DeepSeek became cheaper and improved
slightly, though not significantly. Ministral became much less tool-intensive,
yet selection overhead left total tokens approximately flat. Ox Alpha became
dramatically more token-intensive. The only strong accuracy improvement,
Luna, lacks phase-token telemetry and must be rerun if a token-efficiency claim
is required for that model.

## Interpretation and limitations

1. Each condition currently has one run. Repeated runs are needed to estimate
   variance from sampling, providers, web search, and remote tools.
2. Every model built and read its own bank. The experiment measures the whole
   model-specific memory system, not just the effect of injecting an identical
   memory corpus.
3. Provider/model/protocol failures count as incorrect in the primary score,
   except for DeepSeek's 19 selector failures. Those exact task IDs are removed
   from both DeepSeek conditions; all other DeepSeek failures still count.
4. Filtering on successful selection changes the estimand and may select a
   systematically easier subset. The unfiltered 106-task DeepSeek result is
   retained above as a sensitivity analysis, and selector reliability must be
   reported separately from conditional memory effectiveness.
5. The format guard affects Luna most: its guarded baseline is 56 rather than
   the raw 47. Both values are retained so formatting recovery is not confused
   with workflow-memory reasoning.
6. The attachment-free 106-task population and reversed memory-build split
   prevent direct comparison with officially reported full GAIA results.
7. Memory-building cost is reported separately and is not charged entirely to
   one validation run. In a deployed persistent system it should be amortized
   over later tasks; future work should report the break-even number of tasks.
8. Token telemetry reflects provider-reported usage. It should be validated
   across repeated runs before making cost claims, especially where providers
   differ in caching and reasoning-token accounting.

## Primary artifact map

| Model | Baseline result | Test memory build | Agentic memory validation |
|---|---|---|---|
| GPT-5.6 Luna | `results/gpt_5_6_luna/validation_full_luna_validation_memory_build_v1.jsonl` (matched 106-task subset) | `results/gpt_5_6_luna/test_full_luna_test_outcome_blind_agentic_adaptive_memory_build_v1.jsonl` | `results/gpt_5_6_luna/validation_with_test_memory_full_luna_validation_test_memory_agentic_v1.jsonl` |
| Gemini 3.7 Flash | `results/gemini_3_7_flash/validation_full_gemini_validation_baseline_v1.jsonl` | `results/gemini_3_7_flash/test_full_gemini_test_outcome_blind_agentic_adaptive_memory_build_v1.jsonl` | `results/gemini_3_7_flash/validation_with_test_memory_full_gemini_validation_test_memory_agentic_v1.jsonl` |
| DeepSeek V4 Flash | `results/deepseek_v4_flash_0731/validation_full_deepseek_validation_baseline_efficiency_v1.jsonl` | `results/deepseek_v4_flash_0731/test_full_deepseek_test_outcome_blind_agentic_adaptive_memory_build_efficiency_v3.jsonl` | `results/deepseek_v4_flash_0731/validation_with_test_memory_full_deepseek_validation_test_memory_agentic_efficiency_v2.jsonl` |
| Ox Alpha | `results/ox_alpha/validation_full_ox_alpha_validation_baseline_efficiency_v1.jsonl` | `results/ox_alpha/test_full_ox_alpha_test_outcome_blind_agentic_adaptive_memory_build_efficiency_v1.jsonl` | `results/ox_alpha/validation_with_test_memory_full_ox_alpha_test_bank_agentic_readonly_full_v1.jsonl` |
| Ministral 3B | `results/ministral_3b_2512/validation_full_ministral_validation_baseline_efficiency_v1.jsonl` | `results/ministral_3b_2512/test_full_ministral_test_outcome_blind_agentic_adaptive_memory_build_toolfree_v2.jsonl` | `results/ministral_3b_2512/validation_with_test_memory_full_ministral_validation_agentic_memory_toolfree_bank_v1.jsonl` |

The earlier deterministic Luna first-three-memory run scored 64/106 and is a
control only, not the primary memory condition. Exploratory three-task model
smokes are documented separately in `results/model_smoke_comparison_20260819.md`.
