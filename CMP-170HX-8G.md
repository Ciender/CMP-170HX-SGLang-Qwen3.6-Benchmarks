# CMP 170HX 8G + SGLang + Qwen3.6-27B Benchmark

[English](CMP-170HX-8G.md) | [简体中文](CMP-170HX-8G.zh-CN.md) |
[Side-by-side comparison](README.md)

This page records the complete CMP 170HX 8G (64 GiB) benchmark contributed by
9527 vincentdim. Every table explicitly distinguishes **MTP disabled** from
**MTP enabled**. Speedup always means
`MTP-enabled throughput / MTP-disabled throughput`.

## Results summary

- Across the valid single-request matrix, MTP enabled improves end-to-end
  generation throughput by **2.07x-2.82x**.
- Across the five valid maximum-concurrency-four cases, MTP enabled improves
  aggregate generation throughput by **2.07x-2.54x**.
- On ShareGPT, MTP enabled improves throughput by **2.58x** at maximum
  concurrency one and **2.25x** at maximum concurrency four.
- LongBench-v2 with fixed 512-token generation improves by **1.83x**. With only
  ten generated tokens, long-prompt prefill dominates end-to-end time and
  leaves little opportunity for MTP.
- Every completed MTP case records an acceptance length of 4.00 tokens.

## Single-request generation

Each cell contains three completed requests. End-to-end generation throughput
includes TTFT and is reported in `output tok/s`; TPOT is the per-request mean
time to produce one generated token.

| Input tokens | Requested output tokens | MTP-disabled throughput | MTP-enabled throughput | MTP speedup | MTP-disabled TPOT (ms) | MTP-enabled TPOT (ms) | MTP acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 45.93 | 120.53 | 2.62x | 21.55 | 8.05 | 4.00 |
| 1024 | 4096 | 42.97 | 121.05 | 2.82x | 23.21 | 8.20 | 4.00 |
| 1024 | 8192 | 42.15 | 116.29 | 2.76x | 23.69 | 8.56 | 4.00 |
| 4096 | 1024 | 42.46 | 109.61 | 2.58x | 22.57 | 8.20 | 4.00 |
| 4096 | 4096 | 41.80 | 114.68 | 2.74x | 23.65 | 8.46 | 4.00 |
| 4096 | 8192 | 41.61 | 113.97 | 2.74x | 23.89 | 8.64 | 4.00 |
| 8192 | 1024 | 40.31 | 98.07 | 2.43x | 22.77 | 8.28 | 4.00 |
| 8192 | 4096 | 41.07 | 110.40 | 2.69x | 23.77 | 8.53 | 4.00 |
| 8192 | 8192 | 41.00 | 111.54 | 2.72x | 24.10 | 8.69 | 4.00 |
| 20480 | 1024 | 34.82 | 72.21 | 2.07x | 23.18 | 8.54 | 4.00 |

## Prefill

Each request generates one token. Input throughput is calculated over the
complete request window in `input tok/s`; it is not isolated prefill-kernel
peak throughput. TTFT change is
`(MTP-enabled TTFT - MTP-disabled TTFT) / MTP-disabled TTFT`, so a positive
value means MTP enabled is slower.

| Input tokens | MTP-disabled input throughput | MTP-enabled input throughput | MTP-disabled median TTFT (ms) | MTP-enabled median TTFT (ms) | TTFT change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2934.21 | 2491.91 | 164.84 | 200.69 | +21.8% |
| 2048 | 4685.19 | 3564.12 | 417.03 | 462.56 | +10.9% |
| 4096 | 4783.56 | 4486.13 | 838.38 | 893.66 | +6.6% |
| 8192 | 4670.07 | 4454.42 | 1734.30 | 1827.68 | +5.4% |
| 12288 | 4539.20 | 4328.31 | 2694.70 | 2825.16 | +4.8% |
| 20480 | 4280.87 | 4099.29 | 4770.92 | 4982.14 | +4.4% |

TTFT is higher with MTP enabled at all six points. MTP should not be
interpreted as a prefill optimization.

## Maximum concurrency four

Each cell contains 20 completed requests. Throughput is aggregate server
generation throughput across four concurrent requests in `output tok/s`;
TPOT remains a per-request metric.

| Input tokens/request | Output tokens/request | MTP-disabled aggregate throughput | MTP-enabled aggregate throughput | MTP speedup | MTP-disabled TPOT (ms) | MTP-enabled TPOT (ms) | MTP acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 512 | 148.24 | 308.74 | 2.08x | 24.69 | 10.08 | 4.00 |
| 2048 | 1024 | 149.35 | 367.00 | 2.46x | 25.39 | 9.53 | 4.00 |
| 2048 | 2048 | 150.26 | 382.14 | 2.54x | 25.86 | 9.70 | 4.00 |
| 4096 | 512 | 95.06 | 196.39 | 2.07x | 26.02 | 10.28 | 4.00 |
| 4096 | 1024 | 102.58 | 239.23 | 2.33x | 25.69 | 9.77 | 4.00 |

## Real datasets

| Workload | Requests | Maximum concurrency | MTP-disabled throughput (output tok/s) | MTP-enabled throughput (output tok/s) | MTP speedup | MTP acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 30 | 1 | 43.097 | 111.273 | 2.58x | 4.000 |
| ShareGPT | 20 | 4 | 149.785 | 336.621 | 2.25x | 4.000 |
| LongBench-v2, 10-token generation | 20 | 1 | 2.114 | 2.185 | 1.03x | 4.000 |
| LongBench-v2, fixed 512-token generation | 20 | 1 | 29.912 | 54.706 | 1.83x | 4.000 |

## Selected telemetry

| MTP state | Workload | Maximum concurrency | Mean power (W) | P95 power (W) | GPU utilization | Mean SM clock (MHz) | Maximum temperature (C) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP enabled | ShareGPT | 1 | 244.21 | 262.58 | 95.5% | 1418 | 81 |
| MTP disabled | ShareGPT | 1 | 221.97 | 255.00 | 99.9% | 1261 | 87 |
| MTP enabled | LongBench-v2, 10-token generation | 1 | 211.28 | 252.38 | 98.6% | 1286 | 86 |
| MTP disabled | LongBench-v2, 10-token generation | 1 | 187.82 | 231.24 | 99.1% | 1217 | 87 |
| MTP enabled | LongBench-v2, fixed 512-token generation | 1 | 194.81 | 246.02 | 97.4% | 1209 | 88 |
| MTP disabled | LongBench-v2, fixed 512-token generation | 1 | 194.45 | 244.47 | 99.8% | 1161 | 87 |
| MTP enabled | ShareGPT | 4 | 238.96 | 260.29 | 95.8% | 1433 | 71 |
| MTP disabled | ShareGPT | 4 | 246.14 | 255.57 | 99.8% | 1422 | 74 |
