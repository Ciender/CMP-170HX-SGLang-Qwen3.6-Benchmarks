# CMP 170HX 10G + SGLang + Qwen3.6-27B Benchmark

[English](CMP-170HX-10G.md) | [简体中文](CMP-170HX-10G.zh-CN.md) |
[Side-by-side comparison](README.md)

This page records the complete local CMP 170HX 10G (40 GiB) benchmark at a
configured 250 W power limit. It uses
[`Avesed/Qwen3.6-27B-INT8-W8A8`](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8)
with SGLang 0.5.16. The public dataset ID is `expanded-250w-v2-20260731`.

## Results summary

- With one request, **MTP enabled** delivers **1.79x-2.27x** the end-to-end
  generation throughput of **MTP disabled** across the valid matrix with
  1K-20K input and 1K-8K output.
- Across five valid maximum-concurrency-four cases, MTP enabled improves
  aggregate generation throughput by **1.65x-1.89x**.
- Mean MTP acceptance length is 2.91-3.50 tokens, and steady decode rate
  improves by **1.91x-2.30x**.
- MTP does not accelerate prefill. Median TTFT with MTP enabled is 4.8%-12.7%
  higher than with MTP disabled at all six prefill points.
- On ShareGPT, MTP enabled improves throughput by **1.87x** at maximum
  concurrency one and **1.68x** at maximum concurrency four. LongBench-v2
  with fixed 512-token generation improves by **1.59x**.

## Single-request generation

Each cell contains three completed requests. End-to-end throughput is total
generated tokens over the complete wall-clock request window and includes
TTFT. Steady decode rate is `1000 / mean TPOT`.

| Input tokens | Requested output tokens | MTP-disabled E2E throughput (output tok/s) | MTP-enabled E2E throughput (output tok/s) | MTP speedup | MTP-disabled steady decode (tok/s) | MTP-enabled steady decode (tok/s) | MTP-disabled mean TTFT (ms) | MTP-enabled mean TTFT (ms) | MTP acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 41.60 | 80.07 | 1.92x | 42.00 | 81.78 | 247.5 | 268.7 | 2.96 |
| 1024 | 4096 | 41.77 | 87.19 | 2.09x | 41.87 | 87.69 | 251.9 | 271.1 | 3.18 |
| 1024 | 8192 | 41.62 | 90.26 | 2.17x | 41.68 | 90.54 | 265.4 | 276.6 | 3.31 |
| 4096 | 1024 | 40.24 | 74.09 | 1.84x | 41.71 | 79.69 | 911.3 | 970.9 | 2.91 |
| 4096 | 4096 | 41.19 | 83.85 | 2.04x | 41.57 | 85.56 | 918.5 | 972.3 | 3.12 |
| 4096 | 8192 | 41.19 | 91.74 | 2.23x | 41.38 | 92.78 | 928.0 | 994.1 | 3.41 |
| 8192 | 1024 | 38.49 | 74.44 | 1.93x | 41.38 | 86.83 | 1869.6 | 1961.4 | 3.20 |
| 8192 | 4096 | 40.46 | 86.17 | 2.13x | 41.23 | 89.89 | 1890.3 | 1964.2 | 3.32 |
| 8192 | 8192 | 40.67 | 92.36 | 2.27x | 41.05 | 94.45 | 1874.4 | 1968.3 | 3.50 |
| 20480 | 1024 | 33.68 | 60.38 | 1.79x | 40.42 | 87.74 | 5083.8 | 5289.5 | 3.32 |

The 20,480-token input has approximately 5.3 seconds of TTFT, so its
end-to-end throughput is lower than its steady decode rate. The MTP setting
`4 draft tokens` is the candidates proposed in each speculative cycle, not a
maximum request output length. Speculative cycles repeat until the requested
1,024, 4,096, or 8,192 tokens have been generated.

## Prefill

Each cell contains three requests that generate one token each. Input
throughput is calculated over the complete request window and is not isolated
prefill-kernel peak throughput. TTFT change is
`(MTP-enabled TTFT - MTP-disabled TTFT) / MTP-disabled TTFT`, so a positive
value means MTP enabled is slower.

| Input tokens | MTP-disabled input throughput (input tok/s) | MTP-enabled input throughput (input tok/s) | MTP-disabled median TTFT (ms) | MTP-enabled median TTFT (ms) | TTFT change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2326.17 | 2119.08 | 207.74 | 234.02 | +12.7% |
| 2048 | 4428.56 | 4122.16 | 444.77 | 486.97 | +9.5% |
| 4096 | 4561.56 | 4296.73 | 881.46 | 937.63 | +6.4% |
| 8192 | 4460.70 | 4245.31 | 1809.20 | 1905.02 | +5.3% |
| 12288 | 4356.72 | 4142.83 | 2800.57 | 2946.06 | +5.2% |
| 20480 | 4112.76 | 3911.91 | 4968.85 | 5207.51 | +4.8% |

## Maximum concurrency four

Each cell contains 20 completed requests, or five full waves of four. The
throughput columns are aggregate server generation throughput. TPOT remains a
per-request metric; `1000 / TPOT` is not aggregate throughput in this table.

| Input tokens/request | Output tokens/request | MTP-disabled aggregate throughput (output tok/s) | MTP-enabled aggregate throughput (output tok/s) | MTP speedup | MTP-disabled mean TPOT (ms) | MTP-enabled mean TPOT (ms) | MTP-disabled median TTFT (ms) | MTP-enabled median TTFT (ms) | MTP acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 512 | 134.87 | 222.57 | 1.65x | 27.24 | 16.10 | 1310.1 | 524.5 | 3.03 |
| 2048 | 1024 | 142.78 | 254.04 | 1.78x | 26.80 | 14.73 | 1313.6 | 521.3 | 3.05 |
| 2048 | 2048 | 146.73 | 269.31 | 1.84x | 26.65 | 13.91 | 1318.5 | 527.8 | 3.10 |
| 4096 | 512 | 92.11 | 174.34 | 1.89x | 27.44 | 17.62 | 2684.6 | 1133.3 | 3.14 |
| 4096 | 1024 | 100.23 | 179.24 | 1.79x | 26.75 | 14.74 | 2674.2 | 3193.2 | 3.04 |

## Real datasets

| Workload | Requests | Maximum concurrency | Median input tokens | Median actual output tokens | MTP-disabled throughput (output tok/s) | MTP-enabled throughput (output tok/s) | MTP speedup | MTP-disabled median TTFT (ms) | MTP-enabled median TTFT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 30 | 1 | 204 | 218.5 | 40.51 | 75.60 | 1.87x | 212.21 | 231.14 |
| ShareGPT | 20 | 4 | 239 | 280 | 133.59 | 224.17 | 1.68x | 258.70 | 279.87 |
| LongBench-v2, 10-token generation | 20 | 1 | 17844 | 10 | 2.219 | 2.181 | 0.98x | 4401.83 | 4593.94 |
| LongBench-v2, fixed 512-token generation | 20 | 1 | 17844 | 512 | 30.31 | 48.19 | 1.59x | 4438.87 | 4611.11 |

The LongBench-v2 workload uses the `short` split from
[`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2).
With seed 42, it selects 20 requests satisfying
`prompt + 512 <= 24576`; input length ranges from 10,954 to 22,842 tokens.

## Capacity

Maximum concurrency one uses a 24,576-token total pool. The server empirically
requires `input + output < 24576`, so inputs of 20,480 with outputs of 4,096 or
8,192 are capacity skips, not performance failures.

Maximum concurrency four uses a validated 20,480-token total pool. These five
combinations fit:

```text
(2048 + 512)  * 4 = 10240
(2048 + 1024) * 4 = 12288
(2048 + 2048) * 4 = 16384
(4096 + 512)  * 4 = 18432
(4096 + 1024) * 4 = 20480
```

`(4096 + 2048) * 4 = 24576` exceeds that profile's total token pool and is
skipped.

## Test environment

The fixed-length matrix ran from 2026-07-31 17:56:30 to 19:54:15, and the real
datasets ran from 22:08:11 to 22:35:41. All times use Asia/Shanghai (UTC+08:00).

| Component | Measured value |
| --- | --- |
| GPU | NVIDIA GA100 `[CMP 170HX]` 10G, 40960 MiB final memory, SM80 |
| Configured power limit | 250 W |
| Maximum SM / memory clock | 1410 / 1215 MHz |
| PCIe during test | 2.5 GT/s x4; endpoint capability 5 GT/s x16 |
| Host | KVM VM, Debian 12, Linux 6.1.0-51-amd64 |
| CPU / host memory / swap | 44 vCPU Intel Xeon E5-2680 v4 / 24 GiB / 0 B |
| NVIDIA driver / CUDA | 610.43.02 / 13.3.73 |
| Python / PyTorch | 3.11.2 / 2.11.0+cu130 |
| SGLang / Triton / FlashInfer | 0.5.16 / 3.6.0 / 0.6.14 |
| Checkpoint | [`Avesed/Qwen3.6-27B-INT8-W8A8`](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8), compressed-tensors dynamic W8A8 |
| Base model | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Target weight file | 28.31 GiB |
| MTP weight file | 0.79 GiB |

### GPU memory and token pool

The table comes from the four server startup logs and uses GiB. Load-phase
deltas describe runtime loading and are not the on-disk weight size or a
component's final exclusive resident memory.

| Item | Max concurrency 1, MTP disabled | Max concurrency 1, MTP enabled | Max concurrency 4, MTP disabled | Max concurrency 4, MTP enabled |
| --- | ---: | ---: | ---: | ---: |
| Total token pool | 24,576 tokens | 24,576 tokens | 20,480 tokens | 20,480 tokens |
| Target load-phase delta | 28.48 | 28.48 | 28.48 | 28.48 |
| MTP draft-worker load-phase delta | Not applicable | 5.53 | Not applicable | 5.53 |
| Target GDN state | 0.29 | 0.29 | 0.71 | 0.71 |
| MTP verification intermediate state | Not applicable | 1.13 | Not applicable | 2.84 |
| Target full-attention KV | 1.50 | 1.50 | 1.26 | 1.26 |
| MTP draft KV | Not applicable | 0.10 | Not applicable | 0.08 |
| Decode/verify/draft CUDA Graph | 0.06 | 0.13 | 0.12 | 0.36 |
| Free memory after both KV pools | 8.97 | 2.21 | 8.80 | 0.35 |

Only 16 of Qwen3.6-27B's 64 layers use full attention. Target BF16 KV is
64 KiB/token, so 24,576 tokens use 1.50 GiB and 20,480 tokens use 1.25 GiB
(the log rounds K and V separately to 1.26 GiB). The MTP draft has one
full-attention layer and uses 4 KiB/token. The GDN layers use recurrent state
allocated by running-request capacity. Maximum concurrency four with MTP
enabled is the tightest configuration, leaving 0.35 GiB after pool allocation.

## Method

Fixed-length cases use the random dataset in
`python -m sglang.benchmark.serving`:

- seed 42, temperature 0.0, top-p 1.0, and `--random-range-ratio 1.0`;
- unlimited request rate, bounded only by maximum concurrency;
- three requests per maximum-concurrency-one and prefill cell, and 20 requests
  per maximum-concurrency-four cell;
- `--flush-cache` for every case and zero benchmark warmup requests;
- per-request JSONL output and NVML telemetry at approximately 200 ms intervals.

| Profile ID | Context | Total token pool | Maximum concurrency | Static memory fraction | MTP state | Speculative decoding settings |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `mtp` | 24576 | 24576 | 1 | 0.94 | MTP enabled | EAGLE, 3 steps, top-k 1, 4 draft tokens |
| `baseline` | 24576 | 24576 | 1 | 0.88 | MTP disabled | Not applicable |
| `mtp-c4` | 24576 | 20480 | 4 | 0.98 | MTP enabled | EAGLE, 3 steps, top-k 1, 4 draft tokens |
| `baseline-c4` | 24576 | 20480 | 4 | 0.88 | MTP disabled | Not applicable |

All profiles use page size 64, 2,048-token chunked prefill, a 4,096-token
prefill ceiling, and disabled prefill CUDA Graph.

## Reproduce

```bash
export SGLANG_RUNTIME_HOME=/mnt/ss32t/SGLang
export SGLANG_VENV=/mnt/ss32t/SGLang/venv
export SGLANG_SOURCE=/path/to/sglang-v0.5.16
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8

nvidia-smi -q -d POWER
sudo -E env RUN_ID=expanded-250w-v2-20260731 RUN_REAL_DATASETS=0 \
  scripts/run-250w-suite.sh
```

Append real-dataset tests:

```bash
sudo -E env RUN_ID=expanded-250w-v2-20260731 RUN_REAL_DATASETS=1 \
  SHAREGPT_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  LONGBENCH_PATH=/path/to/longbench-fit20.jsonl \
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

The suite verifies the 250 W setting, zero swap, and actual server-reported
token/request capacity. Full environment records and public data are in
`environment/` and `results/`.

## Selected 250 W telemetry

| MTP state | Workload | Input / output tokens | Maximum concurrency | Active mean power (W) | Active P95 power (W) | GPU utilization | Mean SM clock (MHz) | Maximum temperature (C) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP enabled | Fixed length | 1024 / 8192 | 1 | 239.57 | 255.56 | 94.6% | 1392 | 84 |
| MTP disabled | Fixed length | 1024 / 8192 | 1 | 247.42 | 251.95 | 100.0% | 1409 | 80 |
| MTP enabled | Fixed length | 4096 / 1024 | 4 | 235.76 | 254.55 | 94.6% | 1389 | 78 |
| MTP disabled | Fixed length | 4096 / 1024 | 4 | 241.66 | 250.67 | 99.8% | 1398 | 81 |
| MTP enabled | LongBench-v2, fixed generation 512 | median 17844 / 512 | 1 | 241.18 | 258.18 | 96.2% | 1360 | 81 |
| MTP disabled | LongBench-v2, fixed generation 512 | median 17844 / 512 | 1 | 246.84 | 255.90 | 99.7% | 1376 | 82 |

## Historical 120 W data

These early 120 W results are retained only as an appendix and are not mixed
with the primary 250 W data above.

| MTP state | Input / output tokens | Maximum concurrency | E2E or aggregate generation throughput (output tok/s) |
| --- | ---: | ---: | ---: |
| MTP disabled | 128 / 256 | 1 | 30.33 |
| MTP enabled | 128 / 256 | 1 | 66.15 |
| MTP disabled | 1024 / 256 | 4 | 99.79 |

## Limitations

- This is one card, checkpoint, software build, and power setting, not a
  general GPU ranking.
- Three requests per fixed single-request cell support profile selection, not
  a cross-system confidence interval.
- Random prompts isolate serving performance; ShareGPT and LongBench-v2 add
  real request distributions.
- Firmware modification and power unlocking are outside the repository scripts.
