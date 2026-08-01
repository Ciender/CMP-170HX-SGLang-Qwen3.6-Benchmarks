# Complete Benchmark Pipeline and Publication Protocol

[English](BENCHMARK-PROTOCOL.md) | [简体中文](BENCHMARK-PROTOCOL.zh-CN.md)

This document is the operating contract for reproducers, reviewers, and AI
agents. It defines a complete CMP 170HX + SGLang + Qwen3.6 benchmark run, the
artifacts that must be retained, and when two runs may be compared.

## Current automation status

The repository has an executable core pipeline, but not a fully automated
run-to-publication pipeline.

| Stage | Current state | Entry point |
| --- | --- | --- |
| GPU/power/swap/process preflight | Automated | `scripts/run-250w-suite.sh` |
| Server startup and profile-capacity validation | Automated | `scripts/serve-qwen36-27b.sh`, `/server_info` |
| Fixed-length and real-dataset requests | Automated | `scripts/run-250w-suite.sh`, `scripts/bench-serving.sh` |
| 200 ms NVML telemetry | Automated | `scripts/monitor-gpu.sh` |
| Case manifest and raw JSONL | Automated | `scripts/run-250w-suite.sh` |
| Environment capture | Partially automated | `scripts/collect-environment.sh` |
| Serving/power aggregation | Tool exists; separate invocation required | `scripts/summarize-expanded.py` |
| Checkpoint revision, complete software diff, and dataset-hash gates | Partially missing | Must be recorded and checked |
| Integrity acceptance, cross-run comparison, publication bundle | Not one-command automated | Follow this document |
| Independent repetitions and randomized/interleaved profile order | Not automated | Required before strict hardware claims |

The repository therefore does need a complete pipeline protocol. Existing
scripts reproduce canonical v2, but another tester cannot obtain strictly
comparable numbers merely by running them without locking inputs and revisions.

## Two reproduction levels

### Level A: protocol reproduction

Use the same case matrix, metric definitions, and main serving arguments to
test whether another environment reproduces the trend. Hardware, driver, or a
clearly disclosed software patch may differ. Publish this as external
reproduction or a cross-system comparison; do not attribute the difference to
one variable.

### Level B: controlled A/B

Measure the causal effect of one variable, such as 40 GiB versus 64 GiB, patch
disabled versus enabled, or 120 W versus 250 W. Except for the target variable,
the following must be identical:

- the same card, or verified equivalent hardware revision, VBIOS, HBM bus/clock,
  and clock policy;
- the same SGLang commit and clean/dirty state, with the full patch retained;
- the same checkpoint repository and immutable revision;
- the same `config.json`, `generation_config.json`, `recipe.yaml`, and MTP file;
- the same driver, CUDA, Python, PyTorch, Triton, and FlashInfer;
- the same prompts, ordering, seed, output lengths, concurrency, and sampling;
- the same power limit, swap state, background GPU processes, and cache policy;
- at least three independent RUN_IDs with counterbalanced profile order.

The current 40 GiB run and external 64 GiB report are Level A, not Level B.

## Fixed test contract

### Checkpoint

```text
Repository: https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8
Base model: https://huggingface.co/Qwen/Qwen3.6-27B
```

A repository name is insufficient. A new run must record the Hugging Face
commit revision or SHA-256 for all critical files, including:

```text
config.json
generation_config.json
recipe.yaml
model.safetensors
mtp.safetensors
tokenizer/config/template files
```

When hashing large weights is impractical, record the immutable Hugging Face
revision and repository LFS oid for each safetensors file. A local directory
name alone is not an identity.

### SGLang and patches

Record:

```bash
git -C "$SGLANG_SOURCE" rev-parse HEAD
git -C "$SGLANG_SOURCE" status --short
git -C "$SGLANG_SOURCE" diff --binary
```

If the worktree is dirty, save the diff with the run artifacts. `SGLang 0.5.16`
alone cannot distinguish the upstream tag, a 170HX patch, a `topk1` fix, or
uncommitted local changes.

### Datasets

Canonical v2 uses:

```text
ShareGPT SHA-256:
35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4

Prepared LongBench-v2 JSONL SHA-256:
93ecd8a799ba868cfb2a1c28a38ce87653cb30eb211712030970020d39990058
```

Runs with different dataset hashes are not strict per-request comparisons.
Even when aggregate prompt/output token counts match, label them
distribution-matched unless normalized row-content hashes are also identical.

### Profiles

Published tables must spell out concurrency and MTP state. Do not use isolated
`baseline`, `mtp`, `c1`, or `c4` as the only reader-facing label.

| Maximum concurrency | MTP state | Context length | Total token pool | Static memory fraction | CUDA Graph batch sizes | Speculative decoding |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | MTP disabled | 24576 | 24576 | 0.88 | 1 | Not applicable |
| 1 | MTP enabled | 24576 | 24576 | 0.94 | 1 | EAGLE, 3 steps, top-k 1, 4 draft tokens |
| 4 | MTP disabled | 24576 | 20480 | 0.88 | 1, 2, 4 | Not applicable |
| 4 | MTP enabled | 24576 | 20480 | 0.98 | 1, 2, 3, 4 | EAGLE, 3 steps, top-k 1, 4 draft tokens |

All profiles use page size 64, 2048-token chunked prefill, a 4096-token prefill
ceiling, disabled prefill CUDA Graph, and disabled radix cache. Any deviation
must be present in run metadata rather than silently changed.

### Case matrix

Single-request generation completes three requests per cell:

```text
Prompt: 1024, 4096, 8192, 20480
Requested generation: 1024, 4096, 8192
Constraint: prompt + generation < 24576
```

Prefill completes three requests per cell and generates one token per request:

```text
Prompt: 512, 2048, 4096, 8192, 12288, 20480
```

Maximum concurrency four completes 20 requests per cell:

```text
Prompt: 2048, 4096
Requested generation: 512, 1024, 2048
Constraint: 4 * (prompt + generation) <= 20480
```

Real datasets:

```text
ShareGPT: 30 requests at maximum concurrency 1, natural output
ShareGPT: 20 requests at maximum concurrency 4, natural output
LongBench-v2: 20 requests at maximum concurrency 1, natural 10-token output
LongBench-v2: the same 20 requests, fixed generation 512
```

A complete run has 56 manifest rows: 50 passed and six skipped-capacity. A
capacity skip is not a failure and must not be reported as zero throughput.

## Complete execution flow

### 1. Select an immutable RUN_ID

```bash
export RUN_ID="cmp170hx-$(date +%Y%m%d-%H%M%S)"
export SGLANG_RUNTIME_HOME=/path/to/sglang-runtime
export SGLANG_VENV=/path/to/sglang-venv
export SGLANG_SOURCE=/path/to/sglang-source
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8
export SHAREGPT_PATH=/path/to/sharegpt.json
export LONGBENCH_PATH=/path/to/longbench-v2-prepared.jsonl
```

Never reuse a RUN_ID for a changed configuration. Resuming a run may reuse only
completed cases whose configuration is unchanged.

### 2. Preflight

```bash
nvidia-smi -q -d POWER,CLOCK,MEMORY,PCI
swapon --show
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
sha256sum "$SHAREGPT_PATH" "$LONGBENCH_PATH"
git -C "$SGLANG_SOURCE" status --short
```

Hard gates:

- power limit equals the declared value;
- swap is zero;
- no other GPU compute process is active;
- no server already owns port 8000;
- checkpoint, dataset, and SGLang revisions are recorded;
- host RAM is sufficient without swap or OOM retry.

### 3. Execute all cases

Current canonical 250 W pipeline:

```bash
sudo -E env \
  RUN_ID="$RUN_ID" \
  RUN_REAL_DATASETS=1 \
  SGLANG_RUNTIME_HOME="$SGLANG_RUNTIME_HOME" \
  SGLANG_VENV="$SGLANG_VENV" \
  SGLANG_SOURCE="$SGLANG_SOURCE" \
  MODEL_PATH="$MODEL_PATH" \
  SHAREGPT_PATH="$SHAREGPT_PATH" \
  LONGBENCH_PATH="$LONGBENCH_PATH" \
  scripts/run-250w-suite.sh
```

The suite starts each profile, queries `/server_info`, runs its cases, stops the
server, and waits for memory release. Every case uses `--flush-cache`; benchmark
warmup is zero while server-startup warmup remains enabled.

### 4. Monitor

`scripts/monitor-gpu.sh` samples every 200 ms by default:

```text
timestamp
power.draw
power.limit
utilization.gpu
clocks.sm
clocks.mem
temperature.gpu
pstate
```

Manifest fields `start_epoch_ms`, `end_epoch_ms`, and `telemetry_file` are the
only authoritative case-to-telemetry mapping. Do not approximate windows from
filenames or reuse another profile's telemetry.

### 5. Summarize

```bash
python3 scripts/summarize-expanded.py \
  "results/case-manifest-$RUN_ID.csv" \
  --repo-root "$PWD" \
  --serving-output "results/serving-$RUN_ID.csv" \
  --power-output "results/power-$RUN_ID.csv"
```

Keep the three metric classes separate:

- `e2e_output_tok_s`: completed output tokens / benchmark wall time;
- `input_tok_s`: completed input tokens / the same wall time;
- `1000 / mean TPOT`: steady-decode approximation for maximum concurrency one
  only.

At maximum concurrency four, never substitute `1000 / TPOT` for aggregate
output throughput.

### 6. Accept the run

Check every item before publication:

- manifest has exactly 56 data rows with unique case keys;
- 50 are `passed`, six are `skipped-capacity`, and none are `failed` or
  `incomplete`;
- the final summary record in every passed JSONL has `completed == requests`;
- fixed cases have the declared input/output token counts;
- MTP-disabled rows have no MTP acceptance length;
- MTP acceptance is recomputed from raw fields and lies in a plausible range;
  every row exactly equal to the draft ceiling requires manual review for a
  field-semantics or summary bug;
- `/server_info` token pool and effective maximum running requests match the
  profile;
- each case window is covered by its telemetry with acceptable intervals and
  gaps;
- environment records checkpoint revision, SGLang commit/diff state, and
  dataset hashes;
- README values can be recomputed from published CSVs, and manual transcription
  is labeled with its source.

### 7. Publish artifacts

An auditable run publishes at least:

```text
environment/<run-id>.txt
results/case-manifest-<run-id>.csv
results/serving-<run-id>.csv
results/power-<run-id>.csv
results/raw/<run-id>/*.jsonl
results/logs/<run-id>/*.log
results/telemetry/<run-id>/*.csv
patches/<run-id>-sglang.diff       # when source is dirty or patched
```

If raw artifacts are too large for Git, publish a downloadable archive, its
SHA-256, and the generation command. PDF- or Markdown-only external results may
be cited, but must be labeled as not recomputable from this repository.

## Cross-run comparison rules

A comparison script or AI must first join by these fields, not table row number:

```text
workload
prompt/input length
requested output length
maximum concurrency
MTP state
sampling parameters
dataset SHA
```

Then inspect environment differences and list every mismatch as a confounder.
Never:

- conflate `MTP enabled / MTP disabled` speedup with a `machine B / machine A`
  throughput ratio;
- compare input tok/s, output tok/s, and steady decode tok/s as one metric;
- treat LongBench 10-token natural output as a decode benchmark;
- encode a capacity skip as failure or 0 tok/s;
- assume identical files from an identical checkpoint name;
- ignore commit and patch because the SGLang version matches;
- interpret additional memory capacity directly as additional throughput.

## AI-agent operating constraints

1. Read this file, the main README, environment record, manifest, and summarizer
   before explaining results.
2. Do not start a long benchmark or overwrite an existing RUN_ID unless the
   user explicitly requests it.
3. Do not modify or delete existing raw/log/telemetry artifacts; use a new run
   directory.
4. When summarizing an external report, retain author, date, RUN_ID, source-file
   hash, and whether raw data is available.
5. Label every calculated value with its formula and every transcribed value
   with its source page/table or source CSV.
6. Use unambiguous labels such as “maximum concurrency 4, MTP enabled,
   aggregate output throughput.”
7. Before committing, check Markdown table widths, run `git diff --check`,
   `shellcheck scripts/*.sh`, and `python3 -m py_compile scripts/*.py`.

See [`COMPARISON-64GB.md`](COMPARISON-64GB.md) for the structured external
64 GiB report and comparability analysis.
