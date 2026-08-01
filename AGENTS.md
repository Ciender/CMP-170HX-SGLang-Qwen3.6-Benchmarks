# Benchmark Agent Instructions

These instructions apply to the entire repository. An AI agent running or
publishing a benchmark must follow them together with
`CMP-170HX-Benchmark-Protocol.md`.

## Scope and authorization

- Inspecting existing data is read-only work. Do not start a server or long
  benchmark unless the user explicitly asks for a run.
- Never overwrite an existing RUN_ID or delete raw JSONL, logs, telemetry,
  metadata, reports, or another tester's files. Use a new RUN_ID.
- Do not change GPU firmware, VBIOS, power limit, clocks, swap, drivers, or
  system packages unless the user separately authorizes that exact action.
- Preserve unrelated worktree changes. Do not commit generated raw model text
  or ignored local source documents.

## Before a run

1. Read `README.md`, `CMP-170HX-Benchmark-Protocol.md`, this file, the selected
   env config, `scripts/run-standard-suite.sh`, and `scripts/plan-benchmark.py`.
2. Identify the local checkpoint and record `MODEL_NAME`, `MODEL_REPOSITORY`,
   immutable `MODEL_REVISION`, `MODEL_PATH`, and SGLang source/commit. Never use
   only a local folder name as model identity when a repository exists.
   Keep SGLang clean and committed. Use `ALLOW_DIRTY_SGLANG=1` only for an
   explicitly labeled diagnostic run and retain its status/diff artifacts.
3. Inspect `config.json`. Set model-specific quantization/backends/parsers only
   when required. Do not apply Qwen3.6 parsers or compressed-tensors flags to an
   arbitrary model by habit.
4. Determine whether the checkpoint actually contains a compatible MTP/EAGLE
   head. Use `MTP_MODES="disabled"` when it does not. A failed MTP startup is
   `MTP unavailable`, not an MTP-disabled result.
5. Confirm zero swap, no competing GPU compute process, a free serving port,
   the declared power limit, sufficient host memory, and a new RUN_ID.
6. Do not estimate token capacity from 32/40/64/80 GiB. Start each serving
   profile and capture live `/server_info` first.
7. Standard pipeline v1 is single-GPU (`TP_SIZE=1`). Do not publish tensor
   parallel data with the single-GPU telemetry schema.

## Capacity planning

For every fixed-length case, calculate both:

```text
per_request_tokens = input_tokens + output_tokens
required_total_tokens = max_concurrency * per_request_tokens
```

Run only when:

```text
per_request_tokens < context_length
required_total_tokens <= max_total_tokens
max_concurrency <= effective_max_running_requests
```

Otherwise preserve a manifest row as `skipped-context` or `skipped-capacity`
with the actual numbers. Never shorten a case silently, report zero throughput,
or call a capacity skip a failure.

All systems attempt the core tier. Systems with additional measured capacity
also run the extended tier. Extended results supplement the core comparison;
they do not make a smaller-memory run incomplete.

For real datasets, tokenize first. Reject or separately group rows that do not
fit; never use character counts as token capacity.

## During a run

- Use `scripts/run-standard-suite.sh` unless diagnosing an individual stage.
- Keep MTP enabled and disabled inputs, outputs, seeds, sampling, concurrency,
  cache policy, and request counts identical within a comparison.
- Keep one server log and one telemetry stream per profile. Manifest epoch-ms
  windows are the only source for case-level power summaries.
- Record actual context, token pool, effective requests, GPU used/free memory,
  target/draft load deltas, KV dtype/size, recurrent state, CUDA Graph memory,
  and pool-end free memory when SGLang reports them.
- A benchmark command failure remains `failed` or `incomplete`. Diagnose it;
  do not rewrite it as a skip without a demonstrated capacity condition.
- Send the user progress updates before long profile starts and after material
  failures or capacity discoveries.

## Reporting

- Generate `serving.csv`, `power.csv`, `report.md`, and `report.pdf` from the
  same bundle. Never hand-edit benchmark values in generated reports.
- Put result highlights first, then environment, profile capacity/memory,
  complete tables, telemetry, and limitations.
- Use complete labels: `MTP enabled`, `MTP disabled`, maximum concurrency,
  prompt length, requested output length, and metric unit.
- Keep `input tok/s`, end-to-end or aggregate `output tok/s`, TTFT, TPOT, and
  single-request steady decode distinct.
- Compute MTP speedup only as same-cell enabled output tok/s divided by disabled
  output tok/s. Do not call a cross-machine throughput ratio MTP speedup.
- State that more memory enables capacity; do not claim it caused higher speed
  without a controlled A/B.
- Distinguish on-disk weight bytes from runtime GPU residency. Do not force
  allocator/context/workspace values into a fabricated total.
- A report with failed or incomplete cases must say `partial run` prominently.

## Validation before publication

Run at minimum:

```bash
git diff --check
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py
scripts/test-pipeline.sh
```

Also check Markdown table column counts, local links, PDF signature/page count,
unique manifest case keys, `completed == requests`, terminal status for every
plan row, and local/remote commit identity before claiming publication is done.

Git commits for this repository use:

```text
Ciender <26506811+Ciender@users.noreply.github.com>
```
