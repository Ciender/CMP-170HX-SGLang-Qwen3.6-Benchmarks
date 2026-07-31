# CMP 170HX + SGLang Qwen3.6-27B W8A8 Benchmarks

[English](README.md) | [简体中文](README.zh-CN.md)

Reproducible single-GPU serving benchmarks for Qwen3.6-27B-INT8-W8A8 on an
NVIDIA CMP 170HX (GA100, SM80, 40 GiB) with SGLang. This repository contains
the exact launch profiles, benchmark driver, 250 W result tables, and raw 200 ms
power telemetry used for the measurements on July 31, 2026.

## Scope

This is a hardware profile and benchmark repository, not a new MTP kernel port.
SGLang 0.5.16 already contains the Qwen3.5/Qwen3.6 MTP model path and EAGLE
speculative decoding support. Qwen3.6 uses the internal architecture name
`Qwen3_5ForConditionalGeneration`. The work here is:

- a memory-safe SM80 launch profile for the 40 GiB CMP 170HX;
- controlled MTP versus non-MTP measurements on identical requests;
- separate profiles for prefill and concurrency-four throughput;
- deterministic benchmark and cache handling;
- synchronized NVML power, clock, utilization, and temperature telemetry.

No source from `vllm-dsa-mtp-sm80` is copied. That project targets vLLM,
DeepSeek/GLM sparse attention, and related SM80 paths. This checkpoint is a
dense hybrid GDN model and uses SGLang's existing SM80 Triton GDN kernels.

## Main findings

- MTP delivers **1.96x to 2.03x** output throughput on 128-token prompts with
  128-512 generated tokens at concurrency one.
- MTP reduces mean TPOT from about **23.9 ms to 11.0-11.5 ms**. Short-prompt
  TTFT increases by 13-43 ms.
- At 10K and 20K input tokens, end-to-end output throughput improves by 1.40x
  and 1.22x. Prefill dominates those requests even though decode TPOT is still
  approximately halved.
- The non-MTP concurrency-four profile reaches **132.95 output tok/s** on 20
  fixed 1024/256 requests.
- The 8K prefill case sustains **4573.80 input tok/s**, or 98.6% of the 4K
  result, while spanning two 4096-token chunks.

## Test time

Measured request window:

```text
2026-07-31 14:56:44 to 15:25:32 Asia/Shanghai (UTC+08:00)
```

Each profile was started from a clean GPU state. No other GPU compute process
was present. After the suite, GPU memory usage returned to zero and
`vm.overcommit_memory` was restored to `0`.

## Test environment

### Hardware and host

| Component | Measured value |
| --- | --- |
| GPU PCI identity | NVIDIA GA100 `[CMP 170HX]`, device `10de:2082` |
| `nvidia-smi` name | `NVIDIA Graphics Device` (firmware enumeration) |
| Compute capability | 8.0 / SM80 |
| GPU memory | 40960 MiB |
| Maximum SM / memory clock | 1410 / 1215 MHz |
| Power state | current/requested/default 250 W; allowed range 100-250 W |
| PCIe link during test | 2.5 GT/s x4; endpoint capability 5 GT/s x16 |
| Host | KVM virtual machine, Debian 12, Linux 6.1.0-51-amd64 |
| CPU | 44 vCPU, Intel Xeon E5-2680 v4 |
| Host memory | 24 GiB RAM, 975 MiB swap |

The downgraded PCIe link affects checkpoint loading and host-device transfers.
The steady-state decode measurements keep weights and active state on the GPU,
but this environment detail should still be considered when comparing systems.

### Software

| Component | Version |
| --- | --- |
| NVIDIA driver / CUDA UMD | 610.43.02 / 13.3 |
| CUDA toolkit | 13.3.73 |
| Python | 3.11.2 |
| PyTorch | 2.11.0+cu130 |
| SGLang | 0.5.16 |
| SGLang engine commit | `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` |
| Triton | 3.6.0 |
| FlashInfer | 0.6.14 |

The NVIDIA user-mode driver/toolkit reports CUDA 13.3; this PyTorch wheel was
built against CUDA 13.0.

### Model

| Item | Value |
| --- | --- |
| Checkpoint | `Qwen3.6-27B-INT8-W8A8` |
| Architecture | `Qwen3_5ForConditionalGeneration`, dense hybrid GDN |
| Text configuration | 64 layers, hidden size 5120, full attention every 4 layers |
| Quantization | compressed-tensors W8A8: channel-wise INT8 weights, dynamic per-token INT8 activations |
| Target weight file | 30,394,496,808 bytes; 28.48 GiB loaded GPU weight memory |
| MTP file | 849,400,424 bytes; 5.53 GiB loaded draft-worker weight memory |
| `config.json` SHA-256 | `c3f53d4c340b3cac0c42dac9f66610acb26e2519b58a9fa20fcd87e706f3f979` |

The checkpoint itself is not distributed by this repository.

## Serving profiles

| Profile | Context | Running requests | MTP | Key settings |
| --- | ---: | ---: | --- | --- |
| `mtp` | 24,576 | 1 | EAGLE, 3 steps, top-k 1, 4 draft tokens | page 64, prefill graph off |
| `baseline` | 24,576 | 1 | Off | page 1, prefill graph off |
| `prefill` | 32,768 | 1 | Off | 4096 chunk, prefill CUDA Graph on |
| `throughput` | 32,768 | 4 | Off | mixed chunk, decode graphs for batch 1/2/4 |

MTP is limited to 24K because the target and draft workers together consume
approximately 34 GiB before KV cache, Mamba state, and CUDA Graph allocations.
Advertising a 32K MTP context on this 40 GiB card would exceed the measured
token capacity.

## Methodology

The serving benchmark uses `python -m sglang.benchmark.serving` with the
`sglang-oai` backend and SGLang's `random` dataset generator.

Common controls:

- exact fixed input and output lengths (`--random-range-ratio 1.0`);
- seed 42, temperature 0.0, top-p 1.0;
- unlimited offered request rate, constrained by `--max-concurrency`;
- cache flush before every measured run;
- zero benchmark warmup requests on an already initialized SGLang server;
- five requests for short decode and every prefill length;
- three requests for 10K/20K long-context cases;
- twenty requests, five waves, for concurrency four;
- full JSONL details written locally for each run.

Why zero benchmark warmups: in this SGLang release, the streamed warmup can
finish on the client immediately before the server releases the request. The
benchmark's immediate `/flush_cache` may then return HTTP 400 and retain the
warmup prefix. The published long-context runs confirmed a successful cache
flush and tightly grouped TTFT values with no prefix-cache hit.

The GPU monitor samples these fields every 200 ms:

```text
timestamp,power.draw,power.limit,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,pstate
```

Power summaries include samples with GPU utilization at or above 90%, excluding
server idle time and model loading. Percentiles use the nearest lower-rank
sample in sorted order.

## Reproduce

Install an SGLang 0.5.16 environment with its CUDA dependencies, obtain the
checkpoint, and set the paths. `SGLANG_SOURCE` is optional; it selects a source
checkout instead of the Python package installed in the virtual environment.

```bash
export SGLANG_RUNTIME_HOME=/mnt/ss32t/SGLang
export SGLANG_VENV=/mnt/ss32t/SGLang/venv
export SGLANG_SOURCE=/path/to/sglang-v0.5.16
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8
```

Confirm the power state before running. The suite verifies 250 W but does not
change the card's power limit or firmware:

```bash
nvidia-smi -q -d POWER
```

Run the complete matrix:

```bash
sudo -E scripts/run-250w-suite.sh
```

Or launch and measure one profile manually:

```bash
sudo -E env PROFILE=mtp scripts/serve-qwen36-27b.sh

RUN_TAG=mtp-128-256 WARMUP_REQUESTS=0 \
DECODE_INPUT_LEN=128 DECODE_OUTPUT_LEN=256 DECODE_PROMPTS=5 \
scripts/bench-serving.sh decode
```

The serving script temporarily sets `vm.overcommit_memory=1` while loading the
model and restores the original value after startup or shutdown. It must be run
as root for the restoration guarantee.

## Results

### Single-request decode

| Input/output | MTP | Output tok/s | Mean TPOT | Mean TTFT | Accept length | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128/128 | Off | 38.99 | 23.79 ms | 249.70 ms | - | - |
| 128/128 | 3 steps / 4 draft | 76.59 | 11.04 ms | 262.26 ms | 3.26 | 1.96x |
| 128/256 | Off | 40.42 | 23.93 ms | 223.86 ms | - | - |
| 128/256 | 3 steps / 4 draft | 81.17 | 11.30 ms | 266.53 ms | 3.23 | 2.01x |
| 128/512 | Off | 41.07 | 23.94 ms | 225.91 ms | - | - |
| 128/512 | 3 steps / 4 draft | 83.45 | 11.48 ms | 262.16 ms | 3.18 | 2.03x |

### Long context

| Input/output | MTP | Input tok/s | Output tok/s | Mean TTFT | Mean TPOT | Accept length |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10240/128 | Off | 1856.52 | 23.21 | 2404.65 ms | 24.37 ms | - |
| 10240/128 | 3 steps / 4 draft | 2599.55 | 32.49 | 2466.28 ms | 11.52 ms | 3.23 |
| 20480/128 | Off | 2476.88 | 15.48 | 5103.24 ms | 24.83 ms | - |
| 20480/128 | 3 steps / 4 draft | 3009.29 | 18.81 | 5274.62 ms | 11.97 ms | 3.16 |

### Prefill

Each request generates exactly one output token.

| Input tokens | Input tok/s | Mean TTFT |
| ---: | ---: | ---: |
| 128 | 890.24 | 136.40 ms |
| 256 | 1922.66 | 127.62 ms |
| 512 | 3254.82 | 150.71 ms |
| 1024 | 4020.76 | 248.31 ms |
| 2048 | 4418.20 | 457.41 ms |
| 4096 | 4639.95 | 876.43 ms |
| 8192 | 4573.80 | 1784.12 ms |

### Concurrency four

Twenty fixed 1024/256 requests completed in 38.51 seconds:

| Output tok/s | Input tok/s | Total tok/s | Mean TTFT | Mean TPOT | Mean concurrency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 132.95 | 531.80 | 664.75 | 764.86 ms | 27.12 ms | 3.99 |

### 250 W telemetry

| Profile workload | Mean power | Median power | P95 power | Mean SM clock | Max temperature |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP decode suite | 221.86 W | 216.06 W | 258.15 W | 1376 MHz | 66 C |
| Non-MTP decode suite | 229.52 W | 232.62 W | 253.24 W | 1394 MHz | 75 C |
| Prefill curve | 238.91 W | 249.40 W | 264.60 W | 1313 MHz | 60 C |
| Concurrency-four mixed | 220.43 W | 217.78 W | 253.93 W | 1402 MHz | 65 C |

The 250 W setting is a control limit, not a guarantee that every instantaneous
NVML sample is at or below 250 W. Short reported samples exceeded the limit;
mean and percentile values better describe sustained behavior. Prefill spends
more time power-limited, which explains its lower mean SM clock.

For context only, earlier fixed-length measurements under the previous 120 W
limit were 30.33 tok/s for baseline 128/256, 66.15 tok/s for MTP 128/256, and
99.79 tok/s for concurrency-four 1024/256. Raw 120 W telemetry was not retained,
so those values are not part of the controlled 250 W dataset.

## Repository contents

```text
scripts/                  launch, suite runner, benchmark, telemetry, summaries
environment/              measured host/software/model metadata
results/serving-250w.csv  machine-readable serving metrics
results/power-250w.csv    machine-readable active-load power summary
results/telemetry/        raw 200 ms NVML samples
```

## Limitations

- This is one card and one software/model combination, not a general GPU rank.
- Request counts are sufficient for profile selection but not confidence
  intervals across cards, drivers, temperatures, or repeated process starts.
- Random fixed-token prompts do not model every production prompt distribution.
- Maximum temperature differs with suite duration and starting temperature.
- MTP is optimized for concurrency one here. The four-way profile is non-MTP.
- Firmware modification and power-limit unlocking are outside this repository.

## License and acknowledgements

Code is provided under the Apache License 2.0. SGLang is developed by the
[SGLang project](https://github.com/sgl-project/sglang). The checkpoint and its
license are distributed separately by its model provider.
