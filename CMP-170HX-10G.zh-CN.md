# CMP 170HX 10G + SGLang + Qwen3.6-27B 测试

[English](CMP-170HX-10G.md) | [简体中文](CMP-170HX-10G.zh-CN.md) |
[两张卡并列对照](README.zh-CN.md)

本页记录本机 CMP 170HX 10G（40 GiB）在 250 W 设置下的完整测试结果。
测试 checkpoint 为
[`Avesed/Qwen3.6-27B-INT8-W8A8`](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8)，
服务框架为 SGLang 0.5.16。公开数据集 ID 为 `expanded-250w-v2-20260731`。

## 结果摘要

- 单请求下，与 **MTP 关闭**相比，**MTP 开启**在有效的 1K-20K 输入、
  1K-8K 输出矩阵中将端到端生成吞吐提升 **1.79x-2.27x**。
- 最大并发 4 的五个有效用例中，MTP 开启将聚合生成吞吐提升
  **1.65x-1.89x**。
- MTP 开启后的平均接受长度为 2.91-3.50 tokens，持续 decode 速率提升
  **1.91x-2.30x**。
- MTP 不加速 prefill。六个 prefill 点中，MTP 开启的中位 TTFT 比 MTP 关闭
  高 4.8%-12.7%。
- ShareGPT 上，MTP 开启在最大并发 1 和 4 时分别提升 **1.87x** 和 **1.68x**；
  LongBench-v2 固定生成 512 tokens 时提升 **1.59x**。

## 单请求生成

每个 cell 完成 3 个请求。端到端生成吞吐等于完整请求窗口内的总输出 token 数除以
墙钟时间，包含 TTFT；持续 decode 速率为 `1000 / 平均 TPOT`。

| 输入 tokens | 请求输出 tokens | MTP 关闭端到端吞吐 (output tok/s) | MTP 开启端到端吞吐 (output tok/s) | MTP 加速比 | MTP 关闭持续 decode (tok/s) | MTP 开启持续 decode (tok/s) | MTP 关闭平均 TTFT (ms) | MTP 开启平均 TTFT (ms) | MTP 接受长度 |
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

20,480-token 输入仍有约 5.3 秒 TTFT，因此端到端吞吐低于持续 decode 速率。
MTP 的 `4 draft tokens` 是每轮推测解码提出的候选数，不是请求最大输出长度；
推测解码会循环运行，直到生成请求指定的 1024、4096 或 8192 tokens。

## Prefill

每个 cell 完成 3 个请求，每个请求只生成 1 token。输入吞吐按完整请求窗口计算，
不是单独 prefill kernel 的峰值。TTFT 变化为
`(MTP 开启 TTFT - MTP 关闭 TTFT) / MTP 关闭 TTFT`，正值表示 MTP 开启更慢。

| 输入 tokens | MTP 关闭输入吞吐 (input tok/s) | MTP 开启输入吞吐 (input tok/s) | MTP 关闭中位 TTFT (ms) | MTP 开启中位 TTFT (ms) | TTFT 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2326.17 | 2119.08 | 207.74 | 234.02 | +12.7% |
| 2048 | 4428.56 | 4122.16 | 444.77 | 486.97 | +9.5% |
| 4096 | 4561.56 | 4296.73 | 881.46 | 937.63 | +6.4% |
| 8192 | 4460.70 | 4245.31 | 1809.20 | 1905.02 | +5.3% |
| 12288 | 4356.72 | 4142.83 | 2800.57 | 2946.06 | +5.2% |
| 20480 | 4112.76 | 3911.91 | 4968.85 | 5207.51 | +4.8% |

## 最大并发 4

每个 cell 完成 20 个请求，即 5 轮完整的四请求并发。吞吐为服务器聚合生成吞吐；
TPOT 是每请求指标，不能用 `1000 / TPOT` 代替服务器聚合吞吐。

| 每请求输入 tokens | 每请求输出 tokens | MTP 关闭聚合吞吐 (output tok/s) | MTP 开启聚合吞吐 (output tok/s) | MTP 加速比 | MTP 关闭平均 TPOT (ms) | MTP 开启平均 TPOT (ms) | MTP 关闭中位 TTFT (ms) | MTP 开启中位 TTFT (ms) | MTP 接受长度 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 512 | 134.87 | 222.57 | 1.65x | 27.24 | 16.10 | 1310.1 | 524.5 | 3.03 |
| 2048 | 1024 | 142.78 | 254.04 | 1.78x | 26.80 | 14.73 | 1313.6 | 521.3 | 3.05 |
| 2048 | 2048 | 146.73 | 269.31 | 1.84x | 26.65 | 13.91 | 1318.5 | 527.8 | 3.10 |
| 4096 | 512 | 92.11 | 174.34 | 1.89x | 27.44 | 17.62 | 2684.6 | 1133.3 | 3.14 |
| 4096 | 1024 | 100.23 | 179.24 | 1.79x | 26.75 | 14.74 | 2674.2 | 3193.2 | 3.04 |

## 真实数据集

| 工作负载 | 请求数 | 最大并发 | 中位输入 tokens | 中位实际输出 tokens | MTP 关闭吞吐 (output tok/s) | MTP 开启吞吐 (output tok/s) | MTP 加速比 | MTP 关闭中位 TTFT (ms) | MTP 开启中位 TTFT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 30 | 1 | 204 | 218.5 | 40.51 | 75.60 | 1.87x | 212.21 | 231.14 |
| ShareGPT | 20 | 4 | 239 | 280 | 133.59 | 224.17 | 1.68x | 258.70 | 279.87 |
| LongBench-v2，生成 10 tokens | 20 | 1 | 17844 | 10 | 2.219 | 2.181 | 0.98x | 4401.83 | 4593.94 |
| LongBench-v2，固定生成 512 tokens | 20 | 1 | 17844 | 512 | 30.31 | 48.19 | 1.59x | 4438.87 | 4611.11 |

LongBench-v2 使用
[`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2)
的 `short` 数据，以 seed 42 固定选择 20 条满足
`prompt + 512 <= 24576` 的请求；输入范围为 10,954-22,842 tokens。

## 容量

最大并发 1 使用 24,576-token 总池，并实测要求 `输入 + 输出 < 24576`。
因此 20,480 输入搭配 4,096 或 8,192 输出会被容量检查拒绝，没有记为性能失败。

最大并发 4 使用实测可启动的 20,480-token 总池。五个有效组合为：

```text
(2048 + 512)  * 4 = 10240
(2048 + 1024) * 4 = 12288
(2048 + 2048) * 4 = 16384
(4096 + 512)  * 4 = 18432
(4096 + 1024) * 4 = 20480
```

`(4096 + 2048) * 4 = 24576` 超过该 profile 的总 token 池，因此跳过。

## 测试环境

固定长度矩阵运行于 2026-07-31 17:56:30 至 19:54:15；真实数据集运行于
2026-07-31 22:08:11 至 22:35:41。时区均为 Asia/Shanghai（UTC+08:00）。

| 项目 | 实测值 |
| --- | --- |
| GPU | NVIDIA GA100 `[CMP 170HX]` 10G，最终显存 40960 MiB，SM80 |
| 功率设置 | 250 W |
| 最高 SM / 显存频率 | 1410 / 1215 MHz |
| 测试时 PCIe | 2.5 GT/s x4；端点能力 5 GT/s x16 |
| 主机 | KVM VM，Debian 12，Linux 6.1.0-51-amd64 |
| CPU / 内存 / Swap | 44 vCPU Intel Xeon E5-2680 v4 / 24 GiB / 0 B |
| NVIDIA 驱动 / CUDA | 610.43.02 / 13.3.73 |
| Python / PyTorch | 3.11.2 / 2.11.0+cu130 |
| SGLang / Triton / FlashInfer | 0.5.16 / 3.6.0 / 0.6.14 |
| Checkpoint | [`Avesed/Qwen3.6-27B-INT8-W8A8`](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8)，compressed-tensors 动态 W8A8 |
| 基础模型 | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Target 权重文件 | 28.31 GiB |
| MTP 权重文件 | 0.79 GiB |

### 显存与 token 池

下表来自四个服务启动日志，显存单位为 GiB。加载阶段增量描述运行时加载过程，
不等同于磁盘权重文件大小或某个组件最终独占的常驻显存。

| 项目 | 最大并发 1，MTP 关闭 | 最大并发 1，MTP 开启 | 最大并发 4，MTP 关闭 | 最大并发 4，MTP 开启 |
| --- | ---: | ---: | ---: | ---: |
| 总 token 池 | 24,576 tokens | 24,576 tokens | 20,480 tokens | 20,480 tokens |
| Target 加载阶段增量 | 28.48 | 28.48 | 28.48 | 28.48 |
| MTP draft worker 加载阶段增量 | 不适用 | 5.53 | 不适用 | 5.53 |
| Target GDN state | 0.29 | 0.29 | 0.71 | 0.71 |
| MTP 验证 intermediate state | 不适用 | 1.13 | 不适用 | 2.84 |
| Target full-attention KV | 1.50 | 1.50 | 1.26 | 1.26 |
| MTP draft KV | 不适用 | 0.10 | 不适用 | 0.08 |
| Decode/verify/draft CUDA Graph | 0.06 | 0.13 | 0.12 | 0.36 |
| 两级 KV pool 分配后的剩余显存 | 8.97 | 2.21 | 8.80 | 0.35 |

Qwen3.6-27B 的 64 层中只有 16 层使用 full attention。Target BF16 KV 为
64 KiB/token，因此 24,576 tokens 占 1.50 GiB，20,480 tokens 占 1.25 GiB
（日志将 K、V 分别四舍五入后显示为 1.26 GiB）。MTP draft 只有一个
full-attention layer，KV 为 4 KiB/token。其余 GDN 层使用按运行请求数分配的
recurrent state。最大并发 4 且 MTP 开启是最紧张的显存组合，分配完成后剩余
0.35 GiB。

## 测试方法

固定长度用例通过 `python -m sglang.benchmark.serving` 的 random 数据集运行：

- seed 42、temperature 0.0、top-p 1.0、`--random-range-ratio 1.0`；
- 请求发送速率不限制，只由最大并发数限流；
- 最大并发 1 和 prefill 每个 cell 3 个请求；最大并发 4 每个 cell 20 个请求；
- 每个 case 执行 `--flush-cache`，benchmark warmup 请求数为 0；
- 保存逐请求 JSONL，NVML 遥测以约 200 ms 间隔采集功耗、利用率、频率和温度。

| Profile ID | 上下文 | 总 token 池 | 最大并发 | 静态显存比例 | MTP 状态 | 推测解码设置 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `mtp` | 24576 | 24576 | 1 | 0.94 | MTP 开启 | EAGLE，3 steps，top-k 1，4 draft tokens |
| `baseline` | 24576 | 24576 | 1 | 0.88 | MTP 关闭 | 不适用 |
| `mtp-c4` | 24576 | 20480 | 4 | 0.98 | MTP 开启 | EAGLE，3 steps，top-k 1，4 draft tokens |
| `baseline-c4` | 24576 | 20480 | 4 | 0.88 | MTP 关闭 | 不适用 |

四个 profile 均使用 page size 64、2048-token chunked prefill、4096-token
prefill 上限，并关闭 prefill CUDA Graph。

## 复现

```bash
export SGLANG_RUNTIME_HOME=/mnt/ss32t/SGLang
export SGLANG_VENV=/mnt/ss32t/SGLang/venv
export SGLANG_SOURCE=/path/to/sglang-v0.5.16
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8

nvidia-smi -q -d POWER
sudo -E env RUN_ID=expanded-250w-v2-20260731 RUN_REAL_DATASETS=0 \
  scripts/run-250w-suite.sh
```

追加真实数据集测试：

```bash
sudo -E env RUN_ID=expanded-250w-v2-20260731 RUN_REAL_DATASETS=1 \
  SHAREGPT_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  LONGBENCH_PATH=/path/to/longbench-fit20.jsonl \
  scripts/run-250w-suite.sh
```

重新生成公开 CSV：

```bash
scripts/summarize-expanded.py \
  results/case-manifest-expanded-250w-v2-20260731.csv \
  --repo-root . \
  --serving-output results/serving-expanded-250w.csv \
  --power-output results/power-expanded-250w.csv
```

脚本会验证 250 W 设置、swap 为 0，以及服务实际报告的 token/request 容量。
完整环境记录和公开数据分别位于 `environment/` 与 `results/`。

## 250 W 遥测摘录

| MTP 状态 | 工作负载 | 输入 / 输出 tokens | 最大并发 | 活跃平均功耗 (W) | 活跃 P95 功耗 (W) | GPU 利用率 | 平均 SM 时钟 (MHz) | 最高温度 (C) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP 开启 | 固定长度 | 1024 / 8192 | 1 | 239.57 | 255.56 | 94.6% | 1392 | 84 |
| MTP 关闭 | 固定长度 | 1024 / 8192 | 1 | 247.42 | 251.95 | 100.0% | 1409 | 80 |
| MTP 开启 | 固定长度 | 4096 / 1024 | 4 | 235.76 | 254.55 | 94.6% | 1389 | 78 |
| MTP 关闭 | 固定长度 | 4096 / 1024 | 4 | 241.66 | 250.67 | 99.8% | 1398 | 81 |
| MTP 开启 | LongBench-v2，固定生成 512 | 中位 17844 / 512 | 1 | 241.18 | 258.18 | 96.2% | 1360 | 81 |
| MTP 关闭 | LongBench-v2，固定生成 512 | 中位 17844 / 512 | 1 | 246.84 | 255.90 | 99.7% | 1376 | 82 |

## 历史 120 W 数据

以下为早期 120 W 结果，仅作附表，不与本页 250 W 主结果混用。

| MTP 状态 | 输入 / 输出 tokens | 最大并发 | 端到端或聚合生成吞吐 (output tok/s) |
| --- | ---: | ---: | ---: |
| MTP 关闭 | 128 / 256 | 1 | 30.33 |
| MTP 开启 | 128 / 256 | 1 | 66.15 |
| MTP 关闭 | 1024 / 256 | 4 | 99.79 |

## 局限性

- 结果来自一张卡、一个 checkpoint、一套软件和一个功率设置，不是通用 GPU 排名。
- 固定单请求 cell 的 3 次重复适合 profile 选择，不足以给出跨系统置信区间。
- Random prompt 用于隔离服务性能；ShareGPT 和 LongBench-v2 用于补充真实分布。
- 固件修改与功率解锁不属于本仓库脚本范围。
