# CMP 170HX + SGLang + Qwen3.6-27B Benchmarks

[English](README.md) | [简体中文](README.zh-CN.md)

This repository collects single-GPU SGLang 0.5.16 results for
`Qwen3.6-27B-INT8-W8A8` on two CMP 170HX cards. This page shows only the most
useful side-by-side results; each card has a separate page for the complete
matrix, settings, and supporting details.

| Card | Complete results | Data author |
| --- | --- | --- |
| CMP 170HX 10G (40 GiB) | [10G benchmark details](CMP-170HX-10G.md) | Ciender |
| CMP 170HX 8G (64 GiB) | [8G benchmark details](CMP-170HX-8G.md) | 9527 vincentdim |

## Key findings

- On the CMP 170HX 10G, **MTP enabled** delivers **1.79x-2.27x** the
  end-to-end generation throughput of **MTP disabled** across the valid
  single-request matrix with 1K-20K input and 1K-8K output. The five valid
  maximum-concurrency-four cases improve by **1.65x-1.89x**.
- On the CMP 170HX 8G, the corresponding gain from **MTP disabled** to
  **MTP enabled** is **2.07x-2.82x** for one request and **2.07x-2.54x**
  at maximum concurrency four.
- MTP accelerates generation, not prompt prefill. Enabling MTP increases TTFT
  at all six prefill points on both cards.
- With MTP disabled, the cards' single-request results differ by only
  0.8%-10.4%. With MTP enabled, the 8G data is 19.6%-50.5% higher. The reported
  MTP acceptance length is 4.00 for the 8G data and 2.91-3.50 for the local 10G
  data, which directly indicates a difference in the MTP execution path.
- These results support enabled-versus-disabled MTP comparisons within each
  card and show differences between two test runs. They are not a controlled
  hardware A/B that isolates the original 10G or 8G memory configuration.

## Single-request comparison

All values are end-to-end generation throughput in `output tok/s`, including
time to first token. Each complete fixed-length cell contains three requests.

| Input / output tokens | 10G, MTP disabled | 10G, MTP enabled | 10G MTP speedup | 8G, MTP disabled | 8G, MTP enabled | 8G MTP speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 / 8192 | 41.62 | 90.26 | 2.17x | 42.15 | 116.29 | 2.76x |
| 4096 / 8192 | 41.19 | 91.74 | 2.23x | 41.61 | 113.97 | 2.74x |
| 8192 / 8192 | 40.67 | 92.36 | 2.27x | 41.00 | 111.54 | 2.72x |
| 20480 / 1024 | 33.68 | 60.38 | 1.79x | 34.82 | 72.21 | 2.07x |

See the complete ten-case matrices for the [10G card](CMP-170HX-10G.md#single-request-generation)
and [8G card](CMP-170HX-8G.md#single-request-generation).

## Maximum-concurrency-four comparison

Values are aggregate server generation throughput across four concurrent
requests in `output tok/s`. Each cell contains 20 completed requests.

| Input / output tokens per request | 10G, MTP disabled | 10G, MTP enabled | 10G MTP speedup | 8G, MTP disabled | 8G, MTP enabled | 8G MTP speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 / 512 | 134.87 | 222.57 | 1.65x | 148.24 | 308.74 | 2.08x |
| 2048 / 1024 | 142.78 | 254.04 | 1.78x | 149.35 | 367.00 | 2.46x |
| 2048 / 2048 | 146.73 | 269.31 | 1.84x | 150.26 | 382.14 | 2.54x |
| 4096 / 512 | 92.11 | 174.34 | 1.89x | 95.06 | 196.39 | 2.07x |
| 4096 / 1024 | 100.23 | 179.24 | 1.79x | 102.58 | 239.23 | 2.33x |

## Prefill comparison

Each request generates one token. Input throughput is prompt tokens processed
over the complete request window, not isolated prefill-kernel peak throughput.
All values are `input tok/s`.

| Input tokens | 10G, MTP disabled | 10G, MTP enabled | 8G, MTP disabled | 8G, MTP enabled |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2326.17 | 2119.08 | 2934.21 | 2491.91 |
| 2048 | 4428.56 | 4122.16 | 4685.19 | 3564.12 |
| 4096 | 4561.56 | 4296.73 | 4783.56 | 4486.13 |
| 8192 | 4460.70 | 4245.31 | 4670.07 | 4454.42 |
| 12288 | 4356.72 | 4142.83 | 4539.20 | 4328.31 |
| 20480 | 4112.76 | 3911.91 | 4280.87 | 4099.29 |

## Real-dataset comparison

Maximum concurrency one is end-to-end generation throughput; maximum
concurrency four is aggregate server generation throughput. All values are
`output tok/s`.

| Workload | Maximum concurrency | 10G, MTP disabled | 10G, MTP enabled | 10G MTP speedup | 8G, MTP disabled | 8G, MTP enabled | 8G MTP speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 1 | 40.513 | 75.597 | 1.87x | 43.097 | 111.273 | 2.58x |
| ShareGPT | 4 | 133.585 | 224.174 | 1.68x | 149.785 | 336.621 | 2.25x |
| LongBench-v2, 10-token generation | 1 | 2.219 | 2.181 | 0.98x | 2.114 | 2.185 | 1.03x |
| LongBench-v2, fixed 512-token generation | 1 | 30.308 | 48.194 | 1.59x | 29.912 | 54.706 | 1.83x |

The LongBench-v2 10-token case is dominated by long-prompt prefill and should
not be used as a sustained decode result. Its fixed 512-token counterpart
shows the generation-stage benefit of MTP more clearly.

## How to test

Copy the example configuration, set SGLang, model, context, MTP, and power
parameters, then run the standard pipeline:

```bash
cp configs/benchmark.example.env /tmp/my-benchmark.env
sudo -E scripts/run-standard-suite.sh /tmp/my-benchmark.env
```

The pipeline reads the live SGLang token pool and evaluates every
input/output/concurrency combination. Smaller-memory systems retain explicit
capacity skips; larger systems continue into the extended long-context matrix.
`runs/<RUN_ID>/` contains metadata, capacity/GPU-memory/KV records, CSV,
Markdown, and PDF. See the [benchmark protocol](CMP-170HX-Benchmark-Protocol.md)
for custom models and required fields; AI agents must also follow
[`AGENTS.md`](AGENTS.md). Capacity planning and CSV/Markdown/PDF generation
have offline fixture coverage; the new generic launcher and runner still need
their first live-model end-to-end qualification.

## Interpreting the difference

Final memory capacity determines whether the model, KV/GDN state, CUDA Graphs,
context, and concurrent requests fit. Once the same token pool already fits,
unused memory does not directly become compute throughput. Here the
MTP-disabled results are close, while the gap expands with MTP enabled. The 8G
data also has an acceptance length of 4.00 versus 2.91-3.50 for the 10G data.
Higher acceptance confirms more draft tokens per target verification and
requires fewer speculative cycles for the same output length.

The two tests may still differ in SGLang code, firmware, sustained clocks, HBM,
cooling, and platform state. The side-by-side tables show the measured
difference between two runs; they do not prove that the original 8G
specification or final 64 GiB capacity caused the corresponding speedup.

## Repository contents

```text
CMP-170HX-10G.md              complete 10G results, environment, memory, reproduction
CMP-170HX-8G.md               complete 8G results and telemetry
scripts/                      launch, benchmark, monitoring, and summary tools
configs/                      standard pipeline configuration template
runs/<run-id>/                generated metadata, plans, CSV, Markdown, PDF
environment/                  measured 10G environment records
results/                      10G summary CSVs and raw NVML telemetry
```

The complete execution and publication contract is in
[`CMP-170HX-Benchmark-Protocol.md`](CMP-170HX-Benchmark-Protocol.md).

## License and acknowledgements

Code is Apache-2.0. SGLang is developed by the
[SGLang project](https://github.com/sgl-project/sglang). Qwen3.6, ShareGPT, and
LongBench-v2 are distributed under their respective licenses. CMP 170HX 8G
benchmark data was contributed by 9527 vincentdim.
