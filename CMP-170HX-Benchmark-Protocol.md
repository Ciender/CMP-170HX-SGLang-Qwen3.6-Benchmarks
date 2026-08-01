# SGLang Model Benchmark and Publication Protocol

[English](CMP-170HX-Benchmark-Protocol.md) | [简体中文](CMP-170HX-Benchmark-Protocol.zh-CN.md)

This document defines the repository's standard benchmark contract. It applies
to CMP 170HX and other NVIDIA GPUs, and to Qwen3.6-27B or a user-supplied
SGLang checkpoint. The core matrix supports cross-system comparisons; the
extended matrix exposes long-context capacity on 64/80 GiB and larger systems.

Standard pipeline v1 is single-GPU and requires `TP_SIZE=1`. Multi-GPU tensor
parallel needs rank/GPU-indexed telemetry and capacity schema before it can be
published as a standard run; do not mix multiple GPU streams into one table.

## Principles

1. **Start the server before planning cases.** GPU memory size cannot be
   converted directly into a token limit. Weights, quantization, KV dtype,
   architecture, MTP draft state, GDN/Mamba state, and CUDA Graphs all affect
   capacity. Planning uses live `/server_info` context, token-pool, and
   effective-running-request values.
2. **Separate comparability from capacity.** Every machine attempts the `core`
   tier. The `extended` tier adds 32K-131K inputs only when actual capacity permits.
3. **A skip is not zero throughput.** Unsupported cases are recorded as
   `skipped-context` or `skipped-capacity`, never failure, zero tok/s, or a
   silently omitted row.
4. **Spell out MTP state.** Published labels use `MTP enabled`, `MTP disabled`,
   or `MTP unavailable`. MTP speedup is always enabled / disabled for the same
   environment and cell.
5. **Generate reports from structured artifacts.** Markdown and PDF use the
   same metadata, manifest, and CSV files; values are not manually copied.

## Standard pipeline

```text
benchmark.env
    │
    ├─ collect-run-metadata.py     host, GPU, software, model, weights, datasets
    ├─ serve-model.sh              model-neutral SGLang launcher
    ├─ capture-profile.py          live capacity, residency, KV/state/graphs
    ├─ plan-benchmark.py           core + extended planning and skip reasons
    ├─ bench-serving.sh            fixed-length requests
    ├─ monitor-gpu.sh              NVML telemetry
    ├─ summarize-expanded.py       serving.csv + power.csv
    └─ generate-report.py          report.md + report.pdf
```

Entry point:

```bash
cp configs/benchmark.example.env /tmp/my-benchmark.env
# Edit paths, model, context, MTP, and power settings.
sudo -E scripts/run-standard-suite.sh /tmp/my-benchmark.env
```

`scripts/test-pipeline.sh` currently covers offline planning for representative
32/40/64/80 GiB-like live capacities plus standardized tables and PDF output.
Treat the generic launcher and full runner as a candidate pipeline until its
first live-model end-to-end qualification. Existing 10G canonical data was
produced by the legacy pipeline and is unaffected.

Use a new `RUN_ID` for every changed configuration. Existing bundles are not
overwritten by default; use `RESUME=1` only when all configuration is unchanged.

## Model and server configuration

Required:

```text
MODEL_PATH
MODEL_NAME
RUN_ID
SGLANG_VENV
CONTEXT_LENGTH (may be `auto`)
```

Strongly recommended:

```text
MODEL_REPOSITORY
MODEL_REVISION
SGLANG_SOURCE
EXPECTED_POWER_LIMIT
```

Custom models must not inherit Qwen3.6-specific settings. Quantization,
attention/sampling backends, reasoning/tool parsers, and `trust_remote_code`
are explicit optional fields. A model without an integrated MTP/EAGLE head uses:

```bash
MTP_MODES="disabled"
```

A model with MTP support can compare:

```bash
MTP_MODES="disabled enabled"
MTP_ALGORITHM=EAGLE
MTP_STEPS=3
MTP_TOPK=1
MTP_DRAFT_TOKENS=4
```

If the MTP profile cannot start, report the profile as unavailable. Do not
silently change the draft model, concurrency, or token pool and claim the same
configuration.

## Capacity rules

Capture these values after every profile starts:

```text
context_length
max_total_tokens
effective_max_running_requests
GPU used/free memory after startup
target and draft load-phase deltas
target/draft KV cache size and dtype
GDN/Mamba/recurrent state size, when reported
CUDA Graph size
pool-end free memory
```

A fixed-length case is runnable only if:

```text
input_tokens + output_tokens < context_length
max_concurrency * (input_tokens + output_tokens) <= max_total_tokens
max_concurrency <= effective_max_running_requests
```

The first comparison is strict because generation needs a scheduling/token
slot. The second is the shared token-pool bound. Real datasets must be
tokenized first and checked per request; character counts are not capacity data.

### Different memory capacities

- **32/40 GiB:** run every `core` case that passes capacity checks and retain
  explicit skip rows for longer cases. Never shorten inputs or outputs to fill a table.
- **64/80 GiB:** complete the same core matrix, then run all capacity-safe
  `extended` cases. Long-context results supplement rather than replace core data.
- **Different models:** restart and recapture the token pool even on the same
  GPU. Capacity from another model or quantization is not reusable.

Standard matrix:

| Tier | Workload | Input tokens | Output tokens | Maximum concurrency | Repetition |
| --- | --- | --- | --- | ---: | ---: |
| core | Prefill | 512, 2048, 4096, 8192, 12288, 20480 | 1 | 1 | 3 requests/cell |
| core | Single-request generation | 1024, 4096, 8192, 20480 | 1024, 4096, 8192 | 1 | 3 requests/cell |
| core | Concurrent generation | 2048, 4096 | 512, 1024, 2048 | configured, default 4 | 5 complete waves |
| extended | Prefill | 32768, 49152, 65536, 98304, 131072 | 1 | 1 | 3 requests/cell |
| extended | Single-request generation | 32768, 49152, 65536, 98304, 131072 | 1024, 4096, 8192 | 1 | 3 requests/cell |
| extended | Concurrent generation | 8192, 16384, 32768 | 512, 1024, 2048 | configured | 5 complete waves |

A JSON `MATRIX_FILE` may add model-specific cases. Removing core rows means the
run is not a complete core reproduction. Cases beyond native model context are
naturally recorded as `skipped-context`.

## Required metadata

### Run identity

- RUN_ID, start time, timezone, tester, and data author;
- benchmark repository commit;
- SGLang commit, remote, dirty state, and patch;
- every manually changed server argument.

### GPU and host

- GPU name, UUID, PCI ID, memory, compute capability, and VBIOS;
- driver, power limit, maximum SM/memory clocks;
- CPU, logical CPUs, host memory, swap, kernel, and OS;
- tensor parallel size, visible GPUs, and other compute processes.

### Model

- local name, Hugging Face repository, and immutable revision;
- architecture, model type, dtype, quantization method/format;
- every safetensors filename and size, plus total weight bytes;
- SHA-256 for config, tokenizer, templates, and quantization recipe;
- native context, layer count, full-attention layers, KV heads, head dimension,
  and KV dtype;
- calculated target KV bytes/token only when configuration makes it reliable;
- MTP weights, layers, draft settings, and embedding-sharing behavior.

### Runtime GPU memory

On-disk weights and GPU residency are different metrics. Report target/draft
load deltas, KV caches, GDN/Mamba state, CUDA Graphs, startup GPU used/free, and
token pool separately. Do not force allocator/context/workspace values into a
false sum when logs do not separate them.

## Metrics and tables

- `input tok/s`: completed prompt tokens / complete request window;
- `output tok/s`: completed generated tokens / the same window; aggregate
  server throughput when concurrency exceeds one;
- `TTFT`: latency to first generated token; lower is better;
- `TPOT`: per-request time per output token after the first; lower is better;
- `1000 / mean TPOT`: single-request steady-decode approximation only;
- MTP acceptance: only for MTP-enabled data and interpreted with the draft ceiling.

Report order:

1. Run summary and model/GPU environment;
2. Live profile capacity and GPU memory/KV/state accounting;
3. Passed/skipped/failed counts;
4. Same-cell MTP enabled/disabled speedups;
5. Complete core table;
6. Extended long-context table;
7. Real datasets;
8. Telemetry and limitations.

Do not publish ambiguous labels such as only `off / 1024 / c1`. Use, for
example, “maximum concurrency 1, MTP disabled, end-to-end output throughput
(output tok/s).” Capacity-skip metric cells remain blank.

## Preflight and monitoring

Before testing, require zero swap, no other GPU compute process, a free server
port, the configured power limit, recorded model/SGLang/dataset identity, and a
new non-overwriting RUN_ID.

Each profile gets a separate server log and NVML stream. The default monitor
samples time, power, power limit, utilization, SM/memory clocks, temperature,
and P-state at approximately 200 ms. Manifest millisecond windows are the only
authoritative case-to-telemetry mapping.

## Publication bundle

```text
runs/<run-id>/
├── metadata.json
├── profiles/*.json
├── plans/*.csv
├── case-manifest.csv
├── serving.csv
├── power.csv
├── report.md
└── report.pdf

results/raw/<run-id>/*.jsonl
results/logs/<run-id>/*.log
results/telemetry/<run-id>/*.csv
```

`generate-report.py` creates Markdown and PDF from the structured bundle. PDF
is a reading artifact; CSV, JSONL, profile JSON, and metadata are the
recomputable sources. If large artifacts are not stored in Git, publish an
archive URL and SHA-256.

## Acceptance

Runs can contain different numbers of passed and skipped rows. The old fixed
“56 rows, 50 passed” rule is not valid for arbitrary models and memory sizes.
Before publication:

- every planned case has one terminal manifest row;
- terminal status is passed, an explicit skip, failed, or incomplete;
- every passed JSONL has `completed == requests`;
- skip notes include actual required tokens and context/token-pool capacity;
- live effective concurrency satisfies the declared profile;
- metadata identifies model, GPU, software, and SGLang source;
- every passed case window maps to telemetry;
- Markdown and PDF come from the same bundle;
- diff, Bash syntax, Python compilation, and capacity fixtures pass.

A run with failed/incomplete rows may be published as a partial run, but the
title and summary must say so. Missing cases must not be relabeled as capacity skips.

## Cross-system comparison

Join only exact matches for model repository/revision, profile settings,
workload, input, output, maximum concurrency, MTP state, sampling, and dataset
content. Then disclose GPU, SGLang, driver, power, clocks, KV dtype, and backend
differences.

More memory usually enables longer context or greater concurrency; it does not
automatically make an identical case faster. MTP speedup is also distinct from
a machine-B / machine-A throughput ratio.
