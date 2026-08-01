# CMP 170HX 10G-to-40G vs 8G-to-64G Benchmark Comparison

[English](CMP-170HX-10G-to-40G-vs-8G-to-64G.md) | [简体中文](CMP-170HX-10G-to-40G-vs-8G-to-64G.zh-CN.md)

This document converts the 14-page report supplied by 9527 vincentdim,
`CMP-170HX_64GB_Qwen3.6-27B.pdf`, into searchable Markdown. It compares the
external CMP 170HX 8G-to-64G report with this repository's CMP 170HX
10G-to-40G canonical run, `expanded-250w-v2-20260731`. The PDF is
used only as a local input and is **not committed to this repository**. Its
local SHA-256 is recorded for source identification:

```text
ef140ebde6888482dc4838afc9ce103f606ac28b0fbd45c2c5a6cddb95410f0e
```

Naming is normalized throughout this repository:

- `CMP 170HX 10G-to-40G` is the local card, with 40 GiB final memory.
- `CMP 170HX 8G-to-64G` is the external card, with 64 GiB final memory.

The modification origins are supplied by the card owners. The PDF itself
confirms the external card's final 64 GB capacity but does not state its
original 8 GB capacity.

Only values that can be read unambiguously from the PDF are transcribed. The
external JSONL, CSV, server logs, and SGLang patch referenced by the PDF were
not supplied with it, so the external summaries cannot be recomputed from raw
records in this repository.

## Executive conclusions

- The external report completed 50 cases and recorded six capacity skips. Its
  matrix, request counts, result naming, and metric definitions match this
  repository's benchmark pipeline.
- Both reports identify one CMP 170HX, SGLang 0.5.16, a 250 W power limit, and
  a checkpoint named `Qwen3.6-27B-INT8-W8A8`. The external host contains two
  8G-to-64G cards, but the report used only `GPU 0`; this was not tensor parallel.
- This is useful same-protocol external reproduction evidence, but it is not a
  controlled 10G-to-40G versus 8G-to-64G hardware A/B. The external report explicitly
  used an additional SM80 `topk1` eager-equivalent fix and does not disclose
  the patch, SGLang source commit, checkpoint revision, or full environment.
- Capacity alone does not make the same token workload faster. External
  MTP-disabled results are generally only 0.8%-10.4% faster than the local
  card, while MTP-enabled results are 19.6%-50.5% faster. More importantly, the
  external report shows an MTP acceptance length of `4.000` for every completed
  MTP case, versus 2.91-3.50 in the local fixed matrix. That difference is a
  stronger explanation for the MTP-specific gain than memory capacity.
- The 8G-to-64G report still used the same 24,576/20,480-token profiles and skipped
  the same six capacity cases. It contains no successful longer-capacity cells
  without a matching local case.

## External environment

| Item | Value confirmed by the PDF |
| --- | --- |
| Data author | 9527 vincentdim |
| RUN_ID | `rep64g-20260801-052021` |
| PDF generation date | 2026-08-01 |
| GPU usage | Host with two CMP 170HX 8G-to-64G cards; benchmark used only `GPU 0` |
| Reported maximum SM clock | 1695 MHz |
| Power-limit field | 250.00 W for every recorded case |
| SGLang | 0.5.16 |
| Model name | `Qwen3.6-27B-INT8-W8A8` |
| Fixed-request counts | Three requests per single/prefill cell; 20 per maximum-concurrency-four cell |
| Completion | 50 passed, six skipped-capacity |
| LongBench-v2 | PDF states that its SHA-256 exactly matches this repository |
| ShareGPT | PDF states that file SHA-256 differs; aggregate token and distribution fields match the corresponding local rows |
| Disclosed code difference | SM80 `topk1` kernel fix using an eager-equivalent implementation |
| Not disclosed | SGLang source commit/patch, checkpoint repository revision, driver, CUDA, PyTorch, VBIOS, PCIe, HBM clock/bus, CPU, and swap |

The PDF says that `config.json` was aligned with the original report. It does
not provide a config SHA-256, so this remains a report claim rather than a
verified identity.

## Single-request generation

These values are transcribed from PDF appendices A2 and A3. End-to-end output
throughput includes TTFT; TPOT is the per-request mean. Speedup is
`MTP enabled / MTP disabled` on the same 8G-to-64G card.

| Prompt length (tokens) | Requested generation length (tokens) | E2E generation throughput, MTP disabled (output tok/s) | E2E generation throughput, MTP enabled (output tok/s) | MTP speedup | Mean TPOT, MTP disabled (ms) | Mean TPOT, MTP enabled (ms) | Reported MTP acceptance length |
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

### Difference from the local 10G-to-40G run

`8G-to-64G / 10G-to-40G` is the ratio of reported end-to-end generation throughput.
It is not a causal memory-capacity speedup because software and hardware state
were not fully controlled.

| Prompt length | Requested generation length | 8G-to-64G / 10G-to-40G, MTP disabled | 8G-to-64G / 10G-to-40G, MTP enabled | 8G-to-64G MTP speedup | 10G-to-40G MTP speedup | 8G-to-64G reported acceptance | 10G-to-40G measured acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1.104x | 1.505x | 2.62x | 1.92x | 4.00 | 2.96 |
| 1024 | 4096 | 1.029x | 1.388x | 2.82x | 2.09x | 4.00 | 3.18 |
| 1024 | 8192 | 1.013x | 1.288x | 2.76x | 2.17x | 4.00 | 3.31 |
| 4096 | 1024 | 1.055x | 1.479x | 2.58x | 1.84x | 4.00 | 2.91 |
| 4096 | 4096 | 1.015x | 1.368x | 2.74x | 2.04x | 4.00 | 3.12 |
| 4096 | 8192 | 1.010x | 1.242x | 2.74x | 2.23x | 4.00 | 3.41 |
| 8192 | 1024 | 1.047x | 1.317x | 2.43x | 1.93x | 4.00 | 3.20 |
| 8192 | 4096 | 1.015x | 1.281x | 2.69x | 2.13x | 4.00 | 3.32 |
| 8192 | 8192 | 1.008x | 1.208x | 2.72x | 2.27x | 4.00 | 3.50 |
| 20480 | 1024 | 1.034x | 1.196x | 2.07x | 1.79x | 4.00 | 3.32 |

## Prefill curve

Each request generates one token. Input throughput is the end-to-end system
prompt-processing metric, not isolated prefill-kernel throughput. TTFT change
is `(MTP enabled - MTP disabled) / MTP disabled`.

| Prompt length (tokens) | Input throughput, MTP disabled (input tok/s) | Input throughput, MTP enabled (input tok/s) | Median TTFT, MTP disabled (ms) | Median TTFT, MTP enabled (ms) | MTP TTFT change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2934.21 | 2491.91 | 164.84 | 200.69 | +21.8% |
| 2048 | 4685.19 | 3564.12 | 417.03 | 462.56 | +10.9% |
| 4096 | 4783.56 | 4486.13 | 838.38 | 893.66 | +6.6% |
| 8192 | 4670.07 | 4454.42 | 1734.30 | 1827.68 | +5.4% |
| 12288 | 4539.20 | 4328.31 | 2694.70 | 2825.16 | +4.8% |
| 20480 | 4280.87 | 4099.29 | 4770.92 | 4982.14 | +4.4% |

Apart from the short 512/2048 cases, where fixed overhead is prominent, the
external 4096-20480-token prefill throughput is about 1.04x-1.05x the local
result. Both reports show that MTP does not accelerate prefill and increases
TTFT.

## Maximum concurrency four

Each cell completed 20 requests. Throughput is aggregate server output across
four requests; TPOT remains a per-request metric.

| Prompt length (tokens/request) | Requested generation length (tokens/request) | Aggregate generation throughput, MTP disabled (output tok/s) | Aggregate generation throughput, MTP enabled (output tok/s) | MTP speedup | Mean TPOT, MTP disabled (ms) | Mean TPOT, MTP enabled (ms) | Reported MTP acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 512 | 148.24 | 308.74 | 2.08x | 24.69 | 10.08 | 4.00 |
| 2048 | 1024 | 149.35 | 367.00 | 2.46x | 25.39 | 9.53 | 4.00 |
| 2048 | 2048 | 150.26 | 382.14 | 2.54x | 25.86 | 9.70 | 4.00 |
| 4096 | 512 | 95.06 | 196.39 | 2.07x | 26.02 | 10.28 | 4.00 |
| 4096 | 1024 | 102.58 | 239.23 | 2.33x | 25.69 | 9.77 | 4.00 |

| Prompt length | Requested generation length | 8G-to-64G / 10G-to-40G, MTP disabled | 8G-to-64G / 10G-to-40G, MTP enabled | 8G-to-64G MTP speedup | 10G-to-40G MTP speedup | 8G-to-64G reported acceptance | 10G-to-40G measured acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 512 | 1.099x | 1.387x | 2.08x | 1.65x | 4.00 | 3.03 |
| 2048 | 1024 | 1.046x | 1.445x | 2.46x | 1.78x | 4.00 | 3.05 |
| 2048 | 2048 | 1.024x | 1.419x | 2.54x | 1.84x | 4.00 | 3.10 |
| 4096 | 512 | 1.032x | 1.126x | 2.07x | 1.89x | 4.00 | 3.14 |
| 4096 | 1024 | 1.023x | 1.335x | 2.33x | 1.79x | 4.00 | 3.04 |

## Real datasets

The table includes both the external 8G-to-64G report and the local 10G-to-40G canonical
run. The LongBench-v2 local JSONL SHA matches. ShareGPT aggregate input/output,
median, and maximum token counts match exactly, but the PDF says that the file
SHA differs. Without external per-request JSONL, byte-for-byte row identity
cannot be established.

| Dataset | Maximum concurrency | 8G-to-64G MTP disabled (output tok/s) | 8G-to-64G MTP enabled (output tok/s) | 8G-to-64G MTP speedup | 10G-to-40G MTP disabled (output tok/s) | 10G-to-40G MTP enabled (output tok/s) | 10G-to-40G MTP speedup | 8G-to-64G reported acceptance | 10G-to-40G measured acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 1 | 43.097 | 111.273 | 2.58x | 40.513 | 75.597 | 1.87x | 4.000 | 2.984 |
| LongBench-v2 natural 10-token output | 1 | 2.114 | 2.185 | 1.03x | 2.219 | 2.181 | 0.98x | 4.000 | 3.100 |
| LongBench-v2 fixed generation 512 | 1 | 29.912 | 54.706 | 1.83x | 30.308 | 48.194 | 1.59x | 4.000 | 3.121 |
| ShareGPT | 4 | 149.785 | 336.621 | 2.25x | 133.585 | 224.174 | 1.68x | 4.000 | 2.884 |

LongBench-v2 natural output is only 10 tokens, so end-to-end throughput is
dominated by long-prompt prefill and is not a useful decode ceiling.

## Selected external telemetry

These are the enlarged real-dataset telemetry rows in the PDF. Mean and P95
power use the report's active-sample definition.

| MTP state | Workload | Maximum concurrency | Mean power (W) | P95 power (W) | GPU utilization | Mean SM clock (MHz) | Maximum temperature (C) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP enabled | ShareGPT | 1 | 244.21 | 262.58 | 95.5% | 1418 | 81 |
| MTP enabled | LongBench-v2 natural output | 1 | 211.28 | 252.38 | 98.6% | 1286 | 86 |
| MTP enabled | LongBench-v2 fixed generation 512 | 1 | 194.81 | 246.02 | 97.4% | 1209 | 88 |
| MTP disabled | ShareGPT | 1 | 221.97 | 255.00 | 99.9% | 1261 | 87 |
| MTP disabled | LongBench-v2 natural output | 1 | 187.82 | 231.24 | 99.1% | 1217 | 87 |
| MTP disabled | LongBench-v2 fixed generation 512 | 1 | 194.45 | 244.47 | 99.8% | 1161 | 87 |
| MTP enabled | ShareGPT | 4 | 238.96 | 260.29 | 95.8% | 1433 | 71 |
| MTP disabled | ShareGPT | 4 | 246.14 | 255.57 | 99.8% | 1422 | 74 |

## Why the external report is faster

### It is not caused by the final 64 GiB capacity itself

Capacity determines whether weights, KV/GDN state, CUDA Graphs, context, and
concurrency fit. Once the same workload already fits and uses the same token
pool, unused memory does not turn directly into Tensor Core throughput. The
external report did not use its additional capacity for a larger pool; it ran
the same 24,576-token single-request and 20,480-token maximum-concurrency-four
profiles.

### The strongest evidence points to an MTP-path difference

If memory capacity, HBM, or general clock differences were the primary cause,
MTP-disabled and MTP-enabled cross-system ratios should be broadly similar.
They are not:

- Single-request MTP-disabled is 1.008x-1.104x local; MTP-enabled is
  1.196x-1.505x.
- Maximum-concurrency-four MTP-disabled is 1.023x-1.099x local; MTP-enabled is
  1.126x-1.445x.
- External acceptance is reported as 4.000 everywhere; local acceptance is
  2.91-3.50.

Higher acceptance confirms more tokens per target verification and requires
fewer speculative cycles for a fixed output length. Most of the external
advantage is therefore MTP-specific rather than an equal improvement to target
decode.

It is not yet possible to determine whether 4.000 represents true per-request
acceptance or a semantic change in the external patch/summary field. That
requires the external SGLang diff and raw JSONL acceptance counters.

### Clock, HBM, and platform differences can still contribute

The PDF labels 1695 MHz versus 1410 MHz for the local profile. Firmware, power
curves, cooling, and sustained clocks can differ. An 8G-to-64G memory modification
may also involve different HBM devices, channel state, or VBIOS. The PDF does
not disclose HBM clock/bus, sustained bandwidth, PCIe, driver, or VBIOS, so the
remaining difference cannot be assigned quantitatively. Long-output telemetry
in the PDF often shows actual SM clocks around 1200-1340 MHz, demonstrating
that “1695 MHz maximum” was not a sustained runtime clock.

## Can these runs be compared directly?

The metrics and reproduction trends can be compared. The difference cannot be
attributed directly to 10G-to-40G versus 8G-to-64G.

| Comparison use | Conclusion |
| --- | --- |
| Confirm that the same matrix runs | Yes |
| Compare enabled/disabled MTP benefit within each machine | Yes, and this is valuable |
| Compare reported throughput at matching prompt/output/concurrency | Yes, if labeled as a cross-system report ratio |
| Prove how much the 8G-to-64G modification accelerates the model | No |
| Prove the speedup from the `topk1` patch | No; a same-card patch-off/patch-on A/B is missing |
| Compare LongBench-v2 fixed 512 | Relatively strong same-dataset evidence, but software/hardware confounders remain |
| Compare ShareGPT | Approximate only because file SHA differs |

Using the same repository resolves case definitions and metric semantics. That
is necessary, not sufficient, for a controlled A/B. A strict comparison also
requires identical SGLang commit/diff, checkpoint revision, config hash,
driver/CUDA/PyTorch, MTP arguments, power/clock policy, and dataset SHA, with
raw JSONL, server logs, manifest, and per-case telemetry retained.

See [`CMP-170HX-Benchmark-Protocol.md`](CMP-170HX-Benchmark-Protocol.md) for the complete rerun and
publication contract.
