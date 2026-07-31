# CMP 170HX + SGLang + Qwen3.6-27B W8A8 Benchmark

[English](README.md) | [简体中文](README.zh-CN.md)

Single-GPU baseline versus MTP serving results for
`Qwen3.6-27B-INT8-W8A8` on a 40 GiB NVIDIA CMP 170HX (GA100, SM80), using
SGLang 0.5.16 at a verified 250 W power limit. The canonical run is
`expanded-250w-v2-20260731`; all tables below come from that run.

## Benchmark results

### Headline findings

- At concurrency one, MTP improves end-to-end output throughput by
  **1.79x-2.27x** across the valid 1K-20K input and 1K-8K output matrix.
- Steady decode, calculated as `1000 / mean TPOT`, improves by
  **1.91x-2.30x**. Mean MTP acceptance length is 2.91-3.50 tokens.
- With four concurrent requests, MTP improves aggregate output throughput by
  **1.65x-1.89x** on the five capacity-safe cases.
- MTP does not accelerate prefill. Median TTFT is 4.8%-12.7% higher than the
  non-MTP profile on the six-point prefill curve.
- On real ShareGPT requests, MTP improves output throughput by **1.87x** at
  concurrency one and **1.68x** at concurrency four. On LongBench-v2 with a
  fixed 512-token generation, it improves end-to-end throughput by **1.59x**.

### Single-request generation

Each cell contains three completed requests. `Off / MTP` columns compare the
same exact input length, output length, seed, and sampling parameters.

| Input/output | E2E output tok/s, Off / MTP | E2E speedup | Steady decode tok/s, Off / MTP | Mean TTFT, Off / MTP | Mean TPOT, Off / MTP | MTP accept length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024/1024 | 41.60 / 80.07 | 1.92x | 42.00 / 81.78 | 247.5 / 268.7 ms | 23.81 / 12.23 ms | 2.96 |
| 1024/4096 | 41.77 / 87.19 | 2.09x | 41.87 / 87.69 | 251.9 / 271.1 ms | 23.88 / 11.40 ms | 3.18 |
| 1024/8192 | 41.62 / 90.26 | 2.17x | 41.68 / 90.54 | 265.4 / 276.6 ms | 23.99 / 11.04 ms | 3.31 |
| 4096/1024 | 40.24 / 74.09 | 1.84x | 41.71 / 79.69 | 911.3 / 970.9 ms | 23.98 / 12.55 ms | 2.91 |
| 4096/4096 | 41.19 / 83.85 | 2.04x | 41.57 / 85.56 | 918.5 / 972.3 ms | 24.06 / 11.69 ms | 3.12 |
| 4096/8192 | 41.19 / 91.74 | 2.23x | 41.38 / 92.78 | 928.0 / 994.1 ms | 24.17 / 10.78 ms | 3.41 |
| 8192/1024 | 38.49 / 74.44 | 1.93x | 41.38 / 86.83 | 1869.6 / 1961.4 ms | 24.17 / 11.52 ms | 3.20 |
| 8192/4096 | 40.46 / 86.17 | 2.13x | 41.23 / 89.89 | 1890.3 / 1964.2 ms | 24.26 / 11.12 ms | 3.32 |
| 8192/8192 | 40.67 / 92.36 | 2.27x | 41.05 / 94.45 | 1874.4 / 1968.3 ms | 24.36 / 10.59 ms | 3.50 |
| 20480/1024 | 33.68 / 60.38 | 1.79x | 40.42 / 87.74 | 5083.8 / 5289.5 ms | 24.74 / 11.40 ms | 3.32 |

### Prefill curve

Each cell contains three requests and exactly one generated token. MTP loads a
draft worker but cannot use speculative decoding to accelerate prompt prefill.

| Input tokens | Input tok/s, Off / MTP | Median TTFT, Off / MTP | MTP TTFT change |
| ---: | ---: | ---: | ---: |
| 512 | 2326.17 / 2119.08 | 207.74 / 234.02 ms | +12.7% |
| 2048 | 4428.56 / 4122.16 | 444.77 / 486.97 ms | +9.5% |
| 4096 | 4561.56 / 4296.73 | 881.46 / 937.63 ms | +6.4% |
| 8192 | 4460.70 / 4245.31 | 1809.20 / 1905.02 ms | +5.3% |
| 12288 | 4356.72 / 4142.83 | 2800.57 / 2946.06 ms | +5.2% |
| 20480 | 4112.76 / 3911.91 | 4968.85 / 5207.51 ms | +4.8% |

### Four concurrent requests

Each cell contains 20 completed requests, or five full waves at the configured
maximum concurrency of four. `Actual concurrency` is the benchmark's
time-weighted average, so finite wave startup and drain make it lower than the
validated server limit in some cells.

| Input/output | Aggregate output tok/s, Off / MTP | Speedup | Mean TPOT, Off / MTP | Median TTFT, Off / MTP | Actual concurrency, Off / MTP | MTP accept length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048/512 | 134.87 / 222.57 | 1.65x | 27.24 / 16.10 ms | 1310.1 / 524.5 ms | 4.00 / 3.89 | 3.03 |
| 2048/1024 | 142.78 / 254.04 | 1.78x | 26.80 / 14.73 ms | 1313.6 / 521.3 ms | 4.00 / 3.92 | 3.05 |
| 2048/2048 | 146.73 / 269.31 | 1.84x | 26.65 / 13.91 ms | 1318.5 / 527.8 ms | 4.00 / 3.84 | 3.10 |
| 4096/512 | 92.11 / 174.34 | 1.89x | 27.44 / 17.62 ms | 2684.6 / 1133.3 ms | 3.73 / 3.73 | 3.14 |
| 4096/1024 | 100.23 / 179.24 | 1.79x | 26.75 / 14.74 ms | 2674.2 / 3193.2 ms | 3.73 / 3.75 | 3.04 |

## How to read throughput

SGLang reports two different views of generation performance:

- `E2E output tok/s = total generated tokens / benchmark request duration`.
  This includes TTFT, so a long prefill lowers the number.
- `Steady decode tok/s = 1000 / mean TPOT` for concurrency-one requests. This
  removes TTFT and shows the sustained decode rate after the first token.

For `20480/1024`, MTP has a 5289.5 ms mean TTFT and 11.40 ms mean TPOT. It
therefore reaches 60.38 tok/s end to end but 87.74 tok/s during steady decode.
The earlier `20480/128 = 18.81 tok/s` observation used only 128 output tokens:
the same roughly 5.3-second prefill was amortized over far less generation. It
was an end-to-end result, not a 18.81 tok/s decode ceiling.

The MTP setting `4 draft tokens` is also not a maximum output length. It is the
number of candidates proposed per speculative cycle; cycles repeat until the
requested 1024, 4096, or 8192 output tokens have been generated.

## Capacity and skipped cells

The single-request profiles expose a 24,576-token context and total-token pool.
This server build empirically requires `input + output < 24576`; equality was
rejected with HTTP 400. Consequently, `20480/4096` and `20480/8192` are
recorded as capacity skips rather than failed benchmarks.

The four-request profiles use a measured 20,480-token total pool. These five
cells fit:

```text
(2048 + 512)  * 4 = 10240
(2048 + 1024) * 4 = 12288
(2048 + 2048) * 4 = 16384
(4096 + 512)  * 4 = 18432
(4096 + 1024) * 4 = 20480
```

`(4096 + 2048) * 4 = 24576` exceeds that measured c4 pool and is skipped. The
c4 server was queried through `/server_info` before testing and reported both
`max_total_num_tokens=20480` and `effective_max_running_requests_per_dp=4`.

## Real datasets

ShareGPT uses natural response lengths. LongBench-v2 uses the official
10-token short-answer setting and a separate fixed 512-token generation on the
same prompts.

| Dataset | Requests / max concurrency | Median input / output | E2E output tok/s, Off / MTP | Speedup | Median TTFT, Off / MTP | Mean TPOT, Off / MTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 30 / 1 | 204 / 218.5 | 40.51 / 75.60 | 1.87x | 212.21 / 231.14 ms | 23.57 / 11.92 ms |
| ShareGPT | 20 / 4 | 239 / 280 | 133.59 / 224.17 | 1.68x | 258.70 / 279.87 ms | 26.80 / 14.76 ms |
| LongBench-v2 short answer | 20 / 1 | 17844 / 10 | 2.219 / 2.181 | 0.98x | 4401.83 / 4593.94 ms | 24.38 / 13.86 ms |
| LongBench-v2 fixed 512 | 20 / 1 | 17844 / 512 | 30.31 / 48.19 | 1.59x | 4438.87 / 4611.11 ms | 24.60 / 12.01 ms |

The LongBench-v2 source is the renamed
[`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2)
repository. Seventy `short` rows were considered; 28 fit the constraint
`prompt + 512 <= 24576`, and 20 were selected with seed 42. Their prompt range
is 10,954-22,842 tokens, median 17,844. The resulting local JSONL SHA-256 is
`93ecd8a799ba868cfb2a1c28a38ce87653cb30eb211712030970020d39990058`.
The ShareGPT source SHA-256 is
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.

## Is this an MTP port?

No SGLang inference kernel was ported for this result. Official SGLang v0.5.16
already includes `qwen3_5_mtp.py`, the Qwen3.5/Qwen3.6 model mapping, and EAGLE
speculative scheduling. Qwen3.6 exposes the internal architecture name
`Qwen3_5ForConditionalGeneration`.

The local SGLang branch is based on official tag `v0.5.16`, commit
`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`. Its two local commits add only
`profiles/170hx/*`; `git diff v0.5.16..HEAD -- python/sglang` is empty. The
tested head is `9a5776c3cf04e7e05fa7f7166b1b3a0f1a6226d1`. This repository's work is a
memory-safe launch profile, controlled measurement, capacity validation, and
telemetry for the CMP 170HX.

The checkpoint's `mtp.safetensors` contains the learned draft weights, but
weights alone are not engine support. The serving engine must still implement
the draft model, speculative scheduler, verification, and SM80-safe kernels.

[`vllm-dsa-mtp-sm80`](https://github.com/allover326/vllm-dsa-mtp-sm80) solves a
different vLLM problem. It composes an SM80 Triton sparse-MLA/DSA backend with
pipeline-parallel MTP fixes for GLM-5.2 and DeepSeek-family sparse models. This
Qwen3.6-27B checkpoint is a dense hybrid GDN model, runs on one GPU, and uses
SGLang's existing SM80 Triton GDN path. None of that vLLM repository's patches
were copied here.

SGLang support is selected by architecture and CUDA capability, not the
marketing name `CMP 170HX`. This card and A100 are both SM80, but this
repository only claims the exact card, model, and software combination tested
here.

## Test environment and time

Controlled fixed-length request window:

```text
2026-07-31 17:56:30 to 19:54:15 Asia/Shanghai (UTC+08:00)
```

Real-dataset request window:

```text
2026-07-31 22:08:11 to 22:35:41 Asia/Shanghai (UTC+08:00)
```

| Component | Measured value |
| --- | --- |
| GPU | NVIDIA GA100 `[CMP 170HX]`, PCI ID `10de:2082`, SM80 |
| GPU memory | 40960 MiB |
| Power limit | 250 W for every raw telemetry sample; allowed range 100-250 W |
| Maximum SM / memory clock | 1410 / 1215 MHz |
| PCIe during test | 2.5 GT/s x4; endpoint capability 5 GT/s x16 |
| Host | KVM VM, Debian 12, Linux 6.1.0-51-amd64 |
| CPU / RAM | 44 vCPU Intel Xeon E5-2680 v4 / 24 GiB |
| Swap | **0 B total, 0 B used** |
| Driver / toolkit | NVIDIA 610.43.02 / CUDA toolkit 13.3.73 |
| Python / PyTorch | 3.11.2 / 2.11.0+cu130 |
| SGLang / Triton / FlashInfer | 0.5.16 / 3.6.0 / 0.6.14 |
| Model | `Qwen3.6-27B-INT8-W8A8`, compressed-tensors dynamic W8A8 |
| Target / draft loading | 28.48 GiB target weights / 5.53 GiB draft-worker weights |

Radix/prefix cache was disabled for all v2 profiles. This does not mean GPU
state is zero: KV cache and Qwen's GDN/Mamba state are allocated as required by
serving. No other GPU compute process was present, and GPU memory returned to
zero after each profile.

### Selected 250 W telemetry

Power statistics use only samples inside a case window with GPU utilization at
or above 90%. The public CSV includes both active and total sample counts.

| Case | Active mean / P95 power | Mean GPU util | Mean SM clock | Max temperature |
| --- | ---: | ---: | ---: | ---: |
| MTP, 1024/8192, c1 | 239.57 / 255.56 W | 94.6% | 1392 MHz | 84 C |
| Off, 1024/8192, c1 | 247.42 / 251.95 W | 100.0% | 1409 MHz | 80 C |
| MTP, 4096/1024, c4 | 235.76 / 254.55 W | 94.6% | 1389 MHz | 78 C |
| Off, 4096/1024, c4 | 241.66 / 250.67 W | 99.8% | 1398 MHz | 81 C |
| MTP, LongBench fixed 512 | 241.18 / 258.18 W | 96.2% | 1360 MHz | 81 C |
| Off, LongBench fixed 512 | 246.84 / 255.90 W | 99.7% | 1376 MHz | 82 C |

The configured limit is not an instantaneous clamp on every NVML-reported
sample; short overshoot samples can exceed 250 W. All 40,172 raw rows report a
250.00 W control limit. Sampling cadence is 200-201 ms at the median/P95.

Earlier 120 W observations are not mixed into this dataset. Their raw
telemetry was not retained, so they are not suitable for a controlled power
comparison. The numbers in every table in this README were rerun at 250 W.

## Methodology

The driver calls `python -m sglang.benchmark.serving` with the `sglang-oai`
backend. Fixed cases use SGLang's random dataset generator with exact lengths:

- `--random-range-ratio 1.0`, seed 42, temperature 0.0, top-p 1.0;
- unlimited offered request rate, bounded by `--max-concurrency`;
- three requests per single-request and prefill cell;
- 20 requests per concurrency-four cell;
- zero benchmark warmup requests on an initialized server;
- `--flush-cache` for every case, in addition to server-side radix-cache
  disablement;
- full per-request JSONL output, including per-token latency arrays.

Zero benchmark warmups avoid a measured race in this SGLang release where the
client's streamed warmup can finish just before the server releases the
request, causing the immediate `/flush_cache` call to return HTTP 400. Server
startup warmup is still enabled.

`scripts/monitor-gpu.sh` runs `nvidia-smi -lms 200` and records timestamp with
milliseconds, power draw and limit, GPU utilization, SM and memory clocks,
temperature, and P-state. The manifest stores exact case start and end epoch
milliseconds and the unique telemetry segment used by each profile start.

## Serving profiles

| Profile | Context | Total token pool | Max running | Static memory | MTP |
| --- | ---: | ---: | ---: | ---: | --- |
| `mtp` | 24576 | 24576 | 1 | 0.94 | EAGLE, 3 steps, top-k 1, 4 draft |
| `baseline` | 24576 | 24576 | 1 | 0.88 | Off |
| `mtp-c4` | 24576 | 20480 | 4 | 0.98 | EAGLE, 3 steps, top-k 1, 4 draft |
| `baseline-c4` | 24576 | 20480 | 4 | 0.88 | Off |

All four profiles use page size 64, 2048-token chunked prefill, a 4096-token
prefill ceiling, disabled prefill CUDA Graph, and disabled radix cache. The c4
profiles capture decode graphs for batch sizes 1, 2, 3, and 4.

The high MTP c4 static-memory fraction is intentional. At 0.94, SGLang reduced
effective concurrency to one. At 0.98 with a 24,576-token pool, draft KV
allocation failed. A 0.98 fraction with a measured 20,480-token pool starts
cleanly, reports effective concurrency four, and completed every published c4
case.

## Reproduce

Prepare an SGLang 0.5.16 environment and the checkpoint:

```bash
export SGLANG_RUNTIME_HOME=/mnt/ss32t/SGLang
export SGLANG_VENV=/mnt/ss32t/SGLang/venv
export SGLANG_SOURCE=/path/to/sglang-v0.5.16
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8
```

The suite verifies a 250 W limit, zero swap, no competing GPU process, and the
server-reported token/request capacity. It does not unlock or change firmware:

```bash
nvidia-smi -q -d POWER
sudo -E env RUN_ID=expanded-250w-v2-20260731 RUN_REAL_DATASETS=0 \
  scripts/run-250w-suite.sh
```

Prepare the exact LongBench subset from two datasets-server responses:

```bash
curl -o /tmp/longbench-0.json \
  'https://datasets-server.huggingface.co/filter?dataset=zai-org%2FLongBench-v2&config=default&split=train&where=%22length%22%3D%27short%27&offset=0&length=50'
curl -o /tmp/longbench-50.json \
  'https://datasets-server.huggingface.co/filter?dataset=zai-org%2FLongBench-v2&config=default&split=train&where=%22length%22%3D%27short%27&offset=50&length=20'

$SGLANG_VENV/bin/python scripts/prepare-longbench-v2.py \
  /tmp/longbench-0.json /tmp/longbench-50.json \
  --model "$MODEL_PATH" --output /tmp/longbench-fit20.jsonl \
  --num-samples 20 --max-context 24576 --output-tokens 512 --seed 42
```

Append real-dataset measurements to the same run ID. Existing completed fixed
cases are reused:

```bash
sudo -E env RUN_ID=expanded-250w-v2-20260731 RUN_REAL_DATASETS=1 \
  SHAREGPT_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  LONGBENCH_PATH=/tmp/longbench-fit20.jsonl \
  scripts/run-250w-suite.sh
```

Regenerate the public CSVs:

```bash
scripts/summarize-expanded.py \
  results/case-manifest-expanded-250w-v2-20260731.csv \
  --repo-root . \
  --serving-output results/serving-expanded-250w.csv \
  --power-output results/power-expanded-250w.csv
```

The launch script temporarily sets `vm.overcommit_memory=1` while loading and
restores the original value after startup or shutdown. Root is required for
that restoration guarantee.

## Repository contents

```text
scripts/                         launch, benchmark, monitor, data preparation
environment/expanded-250w-v2-*  measured host/software/model metadata
results/case-manifest-*-v2-*     case windows, status, files, dataset hashes
results/serving-expanded-250w.csv
results/power-expanded-250w.csv
results/telemetry/*-v2-*         raw 200 ms NVML telemetry
```

Raw SGLang JSONL and server logs remain gitignored because they contain large
per-token arrays, generated text, absolute local paths, and complete server
configuration dumps. The compact public CSVs are derived from those raw files.

## Limitations

- This is one card, checkpoint, software build, and power state, not a general
  GPU ranking.
- Three repeats per fixed single-request cell are appropriate for profile
  selection, not cross-system confidence intervals.
- Fixed-token random prompts isolate serving behavior but do not represent
  every production prompt distribution; ShareGPT and LongBench are included
  as complementary workloads.
- The two sub-second 512-token prefill cases had no telemetry sample at or
  above 90% utilization, so their active-power fields are intentionally empty.
- A100 shares SM80 with CMP 170HX, but A100 memory size, clocks, PCIe, cooling,
  and firmware differ and were not measured here.
- Firmware modification and power-limit unlocking are outside this repository.

## License and acknowledgements

Code is Apache-2.0. SGLang is developed by the
[SGLang project](https://github.com/sgl-project/sglang). Qwen3.6, ShareGPT, and
LongBench-v2 are distributed under their respective licenses.
