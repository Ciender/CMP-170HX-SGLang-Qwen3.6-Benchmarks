# CMP 170HX + SGLang + Qwen3.6-27B W8A8 Benchmark

[English](README.md) | [简体中文](README.zh-CN.md)

Single-GPU `baseline` (MTP disabled) versus `mtp` (MTP enabled) serving results for
`Qwen3.6-27B-INT8-W8A8` on a 40 GiB NVIDIA CMP 170HX (GA100, SM80), using
SGLang 0.5.16 at a configured 250 W power limit. The canonical run is
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

Each cell contains three completed requests. The `MTP disabled` columns come
from the `baseline` profile and the `MTP enabled` columns come from `mtp`.
Both profiles use identical input lengths, output lengths, seeds, and sampling
parameters. Every speedup is `MTP enabled / MTP disabled`.

| Prompt length (tokens) | Requested generation length (tokens) | E2E generation throughput, MTP disabled (output tok/s) | E2E generation throughput, MTP enabled (output tok/s) | MTP speedup | Steady decode rate, MTP disabled (tok/s) | Steady decode rate, MTP enabled (tok/s) | Mean TTFT, MTP disabled (ms) | Mean TTFT, MTP enabled (ms) | Mean TPOT, MTP disabled (ms) | Mean TPOT, MTP enabled (ms) | MTP accept length (tokens) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 41.60 | 80.07 | 1.92x | 42.00 | 81.78 | 247.5 | 268.7 | 23.81 | 12.23 | 2.96 |
| 1024 | 4096 | 41.77 | 87.19 | 2.09x | 41.87 | 87.69 | 251.9 | 271.1 | 23.88 | 11.40 | 3.18 |
| 1024 | 8192 | 41.62 | 90.26 | 2.17x | 41.68 | 90.54 | 265.4 | 276.6 | 23.99 | 11.04 | 3.31 |
| 4096 | 1024 | 40.24 | 74.09 | 1.84x | 41.71 | 79.69 | 911.3 | 970.9 | 23.98 | 12.55 | 2.91 |
| 4096 | 4096 | 41.19 | 83.85 | 2.04x | 41.57 | 85.56 | 918.5 | 972.3 | 24.06 | 11.69 | 3.12 |
| 4096 | 8192 | 41.19 | 91.74 | 2.23x | 41.38 | 92.78 | 928.0 | 994.1 | 24.17 | 10.78 | 3.41 |
| 8192 | 1024 | 38.49 | 74.44 | 1.93x | 41.38 | 86.83 | 1869.6 | 1961.4 | 24.17 | 11.52 | 3.20 |
| 8192 | 4096 | 40.46 | 86.17 | 2.13x | 41.23 | 89.89 | 1890.3 | 1964.2 | 24.26 | 11.12 | 3.32 |
| 8192 | 8192 | 40.67 | 92.36 | 2.27x | 41.05 | 94.45 | 1874.4 | 1968.3 | 24.36 | 10.59 | 3.50 |
| 20480 | 1024 | 33.68 | 60.38 | 1.79x | 40.42 | 87.74 | 5083.8 | 5289.5 | 24.74 | 11.40 | 3.32 |

### Prefill curve

Each cell contains three requests, and each request generates exactly one
token. `System prompt-processing throughput` is
`all prompt tokens / wall-clock time of the complete benchmark request window`.
It includes service request handling and returning the first generated token,
so it is an end-to-end prefill proxy rather than isolated prefill-kernel
throughput. MTP loads a draft worker but cannot use speculative decoding to
accelerate prompt prefill. TTFT change is
`(MTP enabled - MTP disabled) / MTP disabled`; positive values mean MTP is slower.

| Prompt length (tokens) | System prompt-processing throughput, MTP disabled (input tok/s) | System prompt-processing throughput, MTP enabled (input tok/s) | Median TTFT, MTP disabled (ms) | Median TTFT, MTP enabled (ms) | MTP TTFT change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2326.17 | 2119.08 | 207.74 | 234.02 | +12.7% |
| 2048 | 4428.56 | 4122.16 | 444.77 | 486.97 | +9.5% |
| 4096 | 4561.56 | 4296.73 | 881.46 | 937.63 | +6.4% |
| 8192 | 4460.70 | 4245.31 | 1809.20 | 1905.02 | +5.3% |
| 12288 | 4356.72 | 4142.83 | 2800.57 | 2946.06 | +5.2% |
| 20480 | 4112.76 | 3911.91 | 4968.85 | 5207.51 | +4.8% |

### Four concurrent requests

Each cell contains 20 completed requests, or five full waves at the configured
maximum concurrency of four. Aggregate generation throughput is the sum of
output tokens from every concurrent request divided by benchmark wall-clock
time. Mean TPOT is a per-request metric; at concurrency four, `1000 / TPOT`
must not be interpreted as aggregate throughput. `Actual concurrency` is the
benchmark's time-weighted average, so finite wave startup and drain make it
lower than the validated server limit in some cells.

| Prompt length (tokens/request) | Requested generation length (tokens/request) | Aggregate generation throughput, MTP disabled (output tok/s) | Aggregate generation throughput, MTP enabled (output tok/s) | MTP speedup | Mean TPOT, MTP disabled (ms) | Mean TPOT, MTP enabled (ms) | Median TTFT, MTP disabled (ms) | Median TTFT, MTP enabled (ms) | Actual concurrency, MTP disabled | Actual concurrency, MTP enabled | MTP accept length (tokens) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 512 | 134.87 | 222.57 | 1.65x | 27.24 | 16.10 | 1310.1 | 524.5 | 4.00 | 3.89 | 3.03 |
| 2048 | 1024 | 142.78 | 254.04 | 1.78x | 26.80 | 14.73 | 1313.6 | 521.3 | 4.00 | 3.92 | 3.05 |
| 2048 | 2048 | 146.73 | 269.31 | 1.84x | 26.65 | 13.91 | 1318.5 | 527.8 | 4.00 | 3.84 | 3.10 |
| 4096 | 512 | 92.11 | 174.34 | 1.89x | 27.44 | 17.62 | 2684.6 | 1133.3 | 3.73 | 3.73 | 3.14 |
| 4096 | 1024 | 100.23 | 179.24 | 1.79x | 26.75 | 14.74 | 2674.2 | 3193.2 | 3.73 | 3.75 | 3.04 |

## How to read throughput

This document uses three distinct throughput/rate metrics:

- CSV field `e2e_output_tok_s`, or end-to-end generation throughput, is
  `generated tokens from all completed requests / wall-clock time of the benchmark request window`.
  At concurrency one it includes TTFT; at concurrency four it is aggregate
  server output throughput, not a per-request decode rate.
- CSV field `input_tok_s`, or system prompt-processing throughput, is
  `prompt tokens from all completed requests / the same benchmark wall-clock window`.
  Prefill-curve requests still generate one token, so this is not kernel-only
  prefill throughput.
- The steady decode rate is used only for concurrency-one comparisons and is
  `1000 / mean TPOT`. It removes TTFT and approximates sustained generation
  after decode has started.

All throughput speedups are `MTP enabled / MTP disabled`. Higher throughput and
steady decode rates are better; lower TTFT and TPOT are better.

For the single request with a 20,480-token prompt and a requested generation
length of 1,024 tokens, MTP has a 5289.5 ms mean TTFT and 11.40 ms mean TPOT. It
therefore reaches 60.38 tok/s end to end but 87.74 tok/s during steady decode.
The historical result with a 20,480-token prompt and 128 generated tokens was
18.81 tok/s because it used only 128 output tokens:
the same roughly 5.3-second prefill was amortized over far less generation. It
was an end-to-end result, not a 18.81 tok/s decode ceiling.

The MTP setting `4 draft tokens` is also not a maximum output length. It is the
number of candidates proposed per speculative cycle; cycles repeat until the
requested 1024, 4096, or 8192 output tokens have been generated.

## Capacity and skipped cells

The single-request profiles expose a 24,576-token context and total-token pool.
This server build empirically requires `input + output < 24576`; equality was
rejected with HTTP 400. Consequently, prompt 20,480 + generation 4,096 and
prompt 20,480 + generation 8,192 are recorded as capacity skips rather than
failed benchmarks.

The four-request profiles use a measured 20,480-token total pool. These five
cells fit:

```text
(2048 + 512)  * 4 = 10240
(2048 + 1024) * 4 = 12288
(2048 + 2048) * 4 = 16384
(4096 + 512)  * 4 = 18432
(4096 + 1024) * 4 = 20480
```

`(4096 + 2048) * 4 = 24576` exceeds the measured token pool for the
maximum-concurrency-four profiles and is skipped. The server was queried
through `/server_info` before testing and reported both
`max_total_num_tokens=20480` and `effective_max_running_requests_per_dp=4`.

## Real datasets

ShareGPT uses natural response lengths. LongBench-v2 uses the official
10-token short-answer setting and a separate fixed 512-token generation on the
same prompts.

Both profiles use the same prompts, requested output lengths, and sampling
parameters. Median prompt length and median actual generation length are
identical for both profiles. Maximum-concurrency-one rows report end-to-end
throughput; the maximum-concurrency-four row reports aggregate throughput.

| Dataset | Requests | Maximum concurrency | Median prompt length (tokens) | Median actual generation length (tokens) | Generation throughput, MTP disabled (output tok/s) | Generation throughput, MTP enabled (output tok/s) | MTP speedup | Median TTFT, MTP disabled (ms) | Median TTFT, MTP enabled (ms) | Mean TPOT, MTP disabled (ms) | Mean TPOT, MTP enabled (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 30 | 1 | 204 | 218.5 | 40.51 | 75.60 | 1.87x | 212.21 | 231.14 | 23.57 | 11.92 |
| ShareGPT | 20 | 4 | 239 | 280 | 133.59 | 224.17 | 1.68x | 258.70 | 279.87 | 26.80 | 14.76 |
| LongBench-v2 short answer | 20 | 1 | 17844 | 10 | 2.219 | 2.181 | 0.98x | 4401.83 | 4593.94 | 24.38 | 13.86 |
| LongBench-v2 fixed generation 512 | 20 | 1 | 17844 | 512 | 30.31 | 48.19 | 1.59x | 4438.87 | 4611.11 | 24.60 | 12.01 |

The LongBench-v2 source is the renamed
[`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2)
repository. Seventy `short` rows were considered; 28 fit the constraint
`prompt + 512 <= 24576`, and 20 were selected with seed 42. Their prompt range
is 10,954-22,842 tokens, median 17,844. The resulting local JSONL SHA-256 is
`93ecd8a799ba868cfb2a1c28a38ce87653cb30eb211712030970020d39990058`.
The ShareGPT source SHA-256 is
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.

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
| Configured power limit for the canonical run | 250 W |
| Maximum SM clock | 1410 MHz |
| Maximum memory clock | 1215 MHz |
| PCIe during test | 2.5 GT/s x4; endpoint capability 5 GT/s x16 |
| Host | KVM VM, Debian 12, Linux 6.1.0-51-amd64 |
| CPU | 44 vCPU Intel Xeon E5-2680 v4 |
| Host memory | 24 GiB |
| Swap | **0 B total, 0 B used** |
| NVIDIA driver | 610.43.02 |
| CUDA toolkit | 13.3.73 |
| Python | 3.11.2 |
| PyTorch | 2.11.0+cu130 |
| SGLang | 0.5.16 |
| Triton | 3.6.0 |
| FlashInfer | 0.6.14 |
| Checkpoint | [`Avesed/Qwen3.6-27B-INT8-W8A8`](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8), compressed-tensors dynamic W8A8 |
| Base model | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Local target weight file | `model.safetensors`, 30,394,496,808 bytes (28.31 GiB) |
| Local MTP weight file | `mtp.safetensors`, 849,400,424 bytes (0.79 GiB) |
| Target runtime load-phase delta | 28.48 GiB |
| MTP draft-worker load-phase delta | 5.53 GiB; this is neither the size of `mtp.safetensors` nor a measurement of final exclusive resident weights |

### GPU memory accounting and token pool

The following values come directly from the four server startup logs. Units are
GiB and SGLang reports two decimal places. A load-phase delta is the difference
in available memory at the beginning and end of that phase. After initialization,
the MTP draft shares the large-vocabulary embedding and LM head with the target
(`mtp_use_dedicated_embeddings=false`). The 5.53 GiB delta therefore must not be
equated with the 0.79 GiB `mtp.safetensors` file or interpreted as strictly
isolated final draft residency.

| Memory item | Maximum concurrency 1, MTP disabled | Maximum concurrency 1, MTP enabled | Maximum concurrency 4, MTP disabled | Maximum concurrency 4, MTP enabled |
| --- | ---: | ---: | ---: | ---: |
| Total token pool | 24,576 tokens | 24,576 tokens | 20,480 tokens | 20,480 tokens |
| Target load-phase delta | 28.48 | 28.48 | 28.48 | 28.48 |
| MTP draft-worker load-phase delta | Not applicable | 5.53 | Not applicable | 5.53 |
| Target GDN convolution state | 0.01 | 0.01 | 0.01 | 0.01 |
| Target GDN SSM state | 0.28 | 0.28 | 0.70 | 0.70 |
| MTP-verification intermediate SSM state | Not applicable | 1.12 | Not applicable | 2.81 |
| MTP-verification intermediate convolution state | Not applicable | 0.01 | Not applicable | 0.03 |
| Target full-attention KV (K + V) | 0.75 + 0.75 = 1.50 | 0.75 + 0.75 = 1.50 | 0.63 + 0.63 = 1.26 | 0.63 + 0.63 = 1.26 |
| MTP draft KV (K + V) | Not applicable | 0.05 + 0.05 = 0.10 | Not applicable | 0.04 + 0.04 = 0.08 |
| Decode/verify/draft CUDA Graph total | 0.06 | 0.13 | 0.12 | 0.36 |
| Free memory after both KV pools were allocated | 8.97 | 2.21 | 8.80 | 0.35 |

The final row is the pool-end value recorded before CUDA Graph capture. SGLang
releases temporary pools before capture, so the later `available_gpu_mem` value
jumps upward and cannot be added to that row. The table also deliberately does
not force every entry to sum to 40 GiB: CUDA context, allocator reserves,
temporary workspaces, and shared objects cannot be fully separated from these
summary log lines.

The model's native `max_position_embeddings` is 262,144. The 24,576 value is
neither the native model limit nor an absolute limit of the card; it is the
measured, safe single-request profile selected for this project. With 64-token
pages, 24,576 is exactly 384 pages. Only 16 of Qwen3.6-27B's 64 layers use full
attention; the other 48 are GDN/linear-attention layers and do not all retain a
token-linear KV cache. The target BF16 KV calculation is:

```text
16 layers * 4 KV heads * 256 head_dim * 2 bytes * 2 (K + V)
= 65,536 bytes/token = 64 KiB/token

24,576 tokens * 64 KiB = 1.50 GiB
20,480 tokens * 64 KiB = 1.25 GiB
```

For 20,480 tokens, the startup log rounds K and V separately to 0.63 GiB, so
the displayed components sum to 1.26 GiB while the unrounded result is 1.25
GiB. The MTP draft has only one full-attention layer:

```text
1 layer * 4 KV heads * 256 head_dim * 2 bytes * 2 (K + V)
= 4 KiB/token

24,576 tokens = 96 MiB (0.094 GiB)
20,480 tokens = 80 MiB (0.078 GiB)
```

The 48 GDN layers instead use recurrent state allocated by running-request
capacity. With maximum concurrency four and MTP enabled, the draft worker, GDN
intermediate state, target/draft KV, and CUDA Graphs jointly create the tight
memory case. A 24,576-token pool failed to start, so the published profile uses
the verified 20,480-token pool and has only 0.35 GiB free after both KV pools
are allocated.

### W8A8, W8A16, and Fable Fusion 711

The tested checkpoint is genuinely W8A8 with INT8 activations, but W8A8 does
not mean that every operation in the model runs in INT8. Its `recipe.yaml`
applies symmetric per-channel INT8 weights (MSE observer) and dynamic per-token
INT8 input activations to matching Linear layers. Quality-sensitive GDN
`in_proj_a`/`in_proj_b`, `lm_head`, embeddings, the vision tower, and the MTP
head are excluded and remain BF16. Quantization covers the MLP, full-attention
projections, and non-excluded GDN Linear projections.

GA100/SM80 has roughly twice the dense theoretical Tensor Core operation rate
for INT8 as for BF16, but real Qwen3.6 throughput cannot simply double. BF16
layers, GDN kernels, dynamic quantization/dequantization, memory bandwidth,
batch size, and MTP verification all affect the result. W8A8 introduces
activation-quantization error that can change close logits and accumulate in
long reasoning, code, or low-probability tokens. Per-channel weights, dynamic
per-token activations, and BF16-sensitive layers reduce this risk. The Avesed
model card calls the checkpoint “near-lossless” but provides no BF16 comparison
table reproducible in this environment, so this report does not treat that
claim as an independently measured accuracy result.

W8A16 keeps INT8 weights but retains FP16/BF16 activations. It is normally
closer to BF16 and more robust to activation outliers, while avoiding dynamic
INT8 activation error. In exchange, it gives up the activation-INT8 GEMM path
that can benefit W8A8. W8A8 is more likely to lead in prefill and larger
batches. Single-token, small-batch decode is often limited by weight bandwidth
and kernel efficiency; W8A16 can still benefit from smaller INT8 weights and
may match or beat W8A8 with a favorable kernel. The format alone does not
determine which is faster. W8A16 was not measured on this CMP 170HX, so neither
speed nor accuracy can be compared directly with the tables above.

The user-referenced
[`lued/Qwen3.6-27B-Fable-Fusion-711-INT8-W8A16-MTP`](https://huggingface.co/lued/Qwen3.6-27B-Fable-Fusion-711-INT8-W8A16-MTP)
is such a W8A16 package: INT8 weights, FP16/BF16 activations, and a BF16 MTP
head. It is not another quantization of the official stock Qwen3.6-27B. Its
source is the multi-stage fine-tuned and merged
[`DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-MTP`](https://huggingface.co/DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-MTP).
“Fable Fusion” names that upstream fine-tune/merge lineage; “711” comes from
the upstream card's 0.711 ARC-Challenge score for its MXFP8 build. It is not an
INT8 kernel, an MTP method, or a 170HX optimization version. That repository
targets vLLM/Marlin and reports dual-RTX-3090 measurements, which cannot replace
the single-card SGLang measurements in this report.

### Selected 250 W telemetry

Power statistics use only samples inside a case window with GPU utilization at
or above 90%. The public CSV includes both active and total sample counts.

| Serving profile | MTP state | Workload | Prompt length (tokens) | Requested generation length (tokens) | Maximum concurrency | Active mean power (W) | Active P95 power (W) | Mean GPU utilization | Mean SM clock (MHz) | Maximum temperature (C) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mtp` | MTP enabled | Fixed length | 1024 | 8192 | 1 | 239.57 | 255.56 | 94.6% | 1392 | 84 |
| `baseline` | MTP disabled | Fixed length | 1024 | 8192 | 1 | 247.42 | 251.95 | 100.0% | 1409 | 80 |
| `mtp-c4` | MTP enabled | Fixed length | 4096 | 1024 | 4 | 235.76 | 254.55 | 94.6% | 1389 | 78 |
| `baseline-c4` | MTP disabled | Fixed length | 4096 | 1024 | 4 | 241.66 | 250.67 | 99.8% | 1398 | 81 |
| `mtp` | MTP enabled | LongBench-v2 fixed generation | Median 17844 | 512 | 1 | 241.18 | 258.18 | 96.2% | 1360 | 81 |
| `baseline` | MTP disabled | LongBench-v2 fixed generation | Median 17844 | 512 | 1 | 246.84 | 255.90 | 99.7% | 1376 | 82 |

## Methodology

The driver calls `python -m sglang.benchmark.serving` with the `sglang-oai`
backend. Fixed cases use SGLang's random dataset generator with exact lengths:

- `--random-range-ratio 1.0`, seed 42, temperature 0.0, top-p 1.0;
- unlimited offered request rate, bounded by `--max-concurrency`;
- three requests per single-request and prefill cell;
- 20 requests per concurrency-four cell;
- zero benchmark warmup requests on an initialized server;
- `--flush-cache` for every case;
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

| Profile | Context length | Total token pool | Maximum running requests | Static memory fraction | MTP state | Speculative decoding settings |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `mtp` | 24576 | 24576 | 1 | 0.94 | MTP enabled | EAGLE, 3 steps, top-k 1, 4 draft tokens |
| `baseline` | 24576 | 24576 | 1 | 0.88 | MTP disabled | Not applicable |
| `mtp-c4` | 24576 | 20480 | 4 | 0.98 | MTP enabled | EAGLE, 3 steps, top-k 1, 4 draft tokens |
| `baseline-c4` | 24576 | 20480 | 4 | 0.88 | MTP disabled | Not applicable |

All four profiles use page size 64, 2048-token chunked prefill, a 4096-token
prefill ceiling, and disabled prefill CUDA Graph. The maximum-concurrency-four
profiles, `mtp-c4` and `baseline-c4`, capture decode graphs for batch sizes 1,
2, 3, and 4.

The `mtp-c4` static-memory fraction of 0.98 is intentional. At 0.94, SGLang
reduced effective concurrency to one. At 0.98 with a 24,576-token pool, MTP
worker memory allocation failed. A 0.98 fraction with a measured 20,480-token
pool starts, reports effective concurrency four, and completed every published
maximum-concurrency-four case.

## Reproduce

Prepare an SGLang 0.5.16 environment and the checkpoint:

```bash
export SGLANG_RUNTIME_HOME=/mnt/ss32t/SGLang
export SGLANG_VENV=/mnt/ss32t/SGLang/venv
export SGLANG_SOURCE=/path/to/sglang-v0.5.16
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8
```

The suite verifies a 250 W limit, zero swap, and the server-reported
token/request capacity. It does not unlock or change firmware:

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

## Appendix: historical 120 W results

The table preserves the 120 W serving results recorded in Git commit
`79a164a`; all canonical result tables above are from the 250 W run.

| MTP state | Speculative decoding settings | Prompt length (tokens) | Requested generation length (tokens) | Maximum concurrency | End-to-end or aggregate generation throughput (output tok/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| MTP disabled | Not applicable | 128 | 256 | 1 | 30.33 |
| MTP enabled | EAGLE, 3 steps, 4 draft tokens | 128 | 256 | 1 | 66.15 |
| MTP disabled | Not applicable | 1024 | 256 | 4 | 99.79 |

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
