# CAMEL experiment pin

The GAIA workflow-memory experiments use the CAMEL source vendored in this
repository under `camel/`.

- Package: `camel-ai`
- Package version: `0.2.83a1`
- Upstream repository: `https://github.com/camel-ai/camel`
- Upstream commit imported into this repository: `9c548266cc5aed567bf9abccb4800832070dd916`
- Parent-repository commit that converted the submodule to vendored source: `3be5169`
- Dependency lock: `camel/uv.lock`
- Installation command: `pip install -e './camel[all]'`

The vendored CAMEL source contains experiment-specific changes made after the
upstream import. Those changes are versioned by the parent `GAIABenchCAMEL`
repository. Every published experiment must therefore record both:

1. the upstream CAMEL commit above; and
2. the final `GAIABenchCAMEL` commit used for the run.

Do not install an unpinned `camel-ai` package or upgrade the vendored source
between baseline and memory conditions. If CAMEL is intentionally upgraded,
create a new experiment version and rerun every condition.

## Fixed agent settings

The baseline and memory runners use the same controlled settings:

- temperature: `0.0`
- sampling seed: `42`
- maximum output tokens per model response: `8192`
- agent context limit: `200000`
- agent step timeout: `600` seconds
- individual tool execution timeout: `120` seconds
- maximum agent iterations: `30`
- automatic context summarization: disabled

The explicit context limit prevents CAMEL's unknown/custom-model fallback from
silently changing the effective context budget and leaves headroom below the
model's advertised context window.

## Artifact isolation

The Qwen memory-building runner writes only beneath model- and run-specific
paths:

- `workflows/qwen3_8_27b/validation_<mode>_<run-id>/`
- `task_results/qwen3_8_27b/validation_<mode>_<run-id>/`
- `results/qwen3_8_27b/validation_<mode>_<run-id>.jsonl`

It refuses to start when any selected output path already exists. Workflow
filenames begin with `qwen3_8_27b`, include the GAIA task UUID, and therefore
cannot collide with the legacy Markdown memories in the existing top-level
workflow directories.

The memory-condition runner accepts only an explicitly named existing bank
beneath `workflows/qwen3_8_27b/`. It reads that bank without saving or updating
memories and loads the same deterministic set of up to three memories for every
task; this keeps the first experiment non-adaptive.

## Pinned model condition

- Serving platform: OpenRouter
- API base URL: `https://openrouter.ai/api/v1`
- OpenRouter model slug: `qwen/qwen3.8-27b`
- Canonical dated model slug used by the runners: `qwen/qwen3.8-27b-20260814`
- Upstream model ID reported by OpenRouter: `Qwen/Qwen3.8-27B`
- Architecture: dense 27B vision-language model
- Advertised context window: `262144` tokens
- Experimental context cap: `200000` tokens
- Provider: AkashML only (`akashml`)
- Provider endpoint tag observed at pinning: `akashml/bf16`
- Quantization/precision constraint: `bf16`
- Provider fallbacks: disabled
- Required API-parameter support: enabled
- Reasoning: enabled at `low`, excluded from returned message content
- Tool choice: `auto`

The provider is pinned because OpenRouter exposes this model through providers
with different quantization levels and tool support. Allowing automatic
provider routing would mix BF16 and FP8 inference across experimental
conditions. If the pinned endpoint is temporarily unavailable, treat the run
as an infrastructure failure and retry the same condition later; do not enable
fallbacks during a run.

## Additional smoke-test model pins

The 2026-08-19 no-memory smoke comparison also supports:

| `GAIA_MODEL` | Canonical model slug | Provider | Precision |
|---|---|---|---|
| `gemini_3_7_flash` | `google/gemini-3.7-flash-20260813` | Google AI Studio | provider native/unspecified |
| `qwen3_7_flash` | `qwen/qwen3.7-flash-20260727` | Alibaba Cloud Int. | provider native/unspecified |
| `muse_glimmer_30b` | `meta/muse-glimmer-30b-20260810` | Parasail | BF16 |
| `deepseek_v4_flash_0731` | `deepseek/deepseek-v4-flash-20260731` | CoreWeave | FP8 |
| `gpt_5_6_luna` | `openai/gpt-5.6-luna-20260709` | OpenAI | provider native/unspecified |

All use seed `42`, low reasoning, disabled provider fallbacks, and required
parameter support. Temperature is `0.0` where the endpoint supports it. It is
omitted for GPT-5.6 Luna because the pinned OpenAI endpoint does not expose the
temperature parameter.
