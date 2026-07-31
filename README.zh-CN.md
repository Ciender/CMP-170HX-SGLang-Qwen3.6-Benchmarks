# CMP 170HX + SGLang + Qwen3.6-27B W8A8 推理测试

[English](README.md) | [简体中文](README.zh-CN.md)

这是 `Qwen3.6-27B-INT8-W8A8` 在单张 40 GiB NVIDIA CMP 170HX
（GA100、SM80）上运行 SGLang 0.5.16 的 baseline 与 MTP 对照测试。测试期间
功率上限经原始遥测确认为 250 W。本文所有表格都来自标准运行
`expanded-250w-v2-20260731`。

## 测试结果

### 主要结论

- 单请求下，MTP 在有效的 1K-20K 输入、1K-8K 输出矩阵中，将端到端输出
  吞吐提升 **1.79x-2.27x**。
- 以 `1000 / 平均 TPOT` 计算的稳态 decode 提升 **1.91x-2.30x**；MTP
  平均接受长度为 2.91-3.50 tokens。
- 最大并发 4 时，MTP 在五个容量安全的用例中将聚合输出吞吐提升
  **1.65x-1.89x**。
- MTP 不会加速 prefill。在六点 prefill 曲线中，MTP 的中位 TTFT 比非 MTP
  高 4.8%-12.7%。
- 真实 ShareGPT 请求中，MTP 在单请求和 4 并发下分别提升 **1.87x** 和
  **1.68x**；LongBench-v2 固定生成 512 tokens 时提升 **1.59x**。

### 单请求长输出

每个 cell 完成 3 个请求。所有 `关闭 / MTP` 列都使用完全相同的输入长度、
输出长度、seed 和采样参数。

| 输入/输出 | 端到端 output tok/s，关闭 / MTP | 端到端加速 | 稳态 decode tok/s，关闭 / MTP | 平均 TTFT，关闭 / MTP | 平均 TPOT，关闭 / MTP | MTP 接受长度 |
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

### Prefill 曲线

每个 cell 完成 3 个请求，并且严格只生成 1 个 token。MTP 会加载 draft
worker，但 speculative decode 无法加速 prompt prefill。

| 输入 tokens | Input tok/s，关闭 / MTP | 中位 TTFT，关闭 / MTP | MTP TTFT 变化 |
| ---: | ---: | ---: | ---: |
| 512 | 2326.17 / 2119.08 | 207.74 / 234.02 ms | +12.7% |
| 2048 | 4428.56 / 4122.16 | 444.77 / 486.97 ms | +9.5% |
| 4096 | 4561.56 / 4296.73 | 881.46 / 937.63 ms | +6.4% |
| 8192 | 4460.70 / 4245.31 | 1809.20 / 1905.02 ms | +5.3% |
| 12288 | 4356.72 / 4142.83 | 2800.57 / 2946.06 ms | +5.2% |
| 20480 | 4112.76 / 3911.91 | 4968.85 / 5207.51 ms | +4.8% |

### 最大并发 4

每个 cell 完成 20 个请求，相当于按配置的最大并发 4 跑满 5 轮。`实际并发`
是 benchmark 的时间加权平均值，有限轮次的启动和排空会让部分数值低于服务端
已验证的并发上限 4。

| 输入/输出 | 聚合 output tok/s，关闭 / MTP | 加速 | 平均 TPOT，关闭 / MTP | 中位 TTFT，关闭 / MTP | 实际并发，关闭 / MTP | MTP 接受长度 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048/512 | 134.87 / 222.57 | 1.65x | 27.24 / 16.10 ms | 1310.1 / 524.5 ms | 4.00 / 3.89 | 3.03 |
| 2048/1024 | 142.78 / 254.04 | 1.78x | 26.80 / 14.73 ms | 1313.6 / 521.3 ms | 4.00 / 3.92 | 3.05 |
| 2048/2048 | 146.73 / 269.31 | 1.84x | 26.65 / 13.91 ms | 1318.5 / 527.8 ms | 4.00 / 3.84 | 3.10 |
| 4096/512 | 92.11 / 174.34 | 1.89x | 27.44 / 17.62 ms | 2684.6 / 1133.3 ms | 3.73 / 3.73 | 3.14 |
| 4096/1024 | 100.23 / 179.24 | 1.79x | 26.75 / 14.74 ms | 2674.2 / 3193.2 ms | 3.73 / 3.75 | 3.04 |

## 吞吐指标怎么理解

SGLang 报告的两种生成性能含义不同：

- `端到端 output tok/s = 总生成 tokens / 请求计时总时长`。它包含 TTFT，
  因此长 prefill 会压低这个数值。
- 单请求的 `稳态 decode tok/s = 1000 / 平均 TPOT`。它排除了首 token 之前
  的时间，反映开始 decode 后的持续速度。

以 `20480/1024` 为例，MTP 平均 TTFT 为 5289.5 ms，平均 TPOT 为 11.40 ms，
所以端到端速度是 60.38 tok/s，而稳态 decode 是 87.74 tok/s。此前看到的
`20480/128 = 18.81 tok/s` 只生成了 128 tokens，却同样承担了约 5.3 秒 prefill；
它是端到端结果，不代表 decode 上限只有 18.81 tok/s。

`4 draft tokens` 也不是最大输出长度。它表示每个 speculative cycle 提议的
候选数；cycle 会一直重复，直到真正生成请求指定的 1024、4096 或 8192 tokens。

## 容量与跳过项

单请求 profile 的上下文和总 token 池都是 24,576。实测该服务要求
`输入 + 输出 < 24576`，等于 24,576 时会返回 HTTP 400。因此
`20480/4096` 和 `20480/8192` 被明确记录为容量跳过，而不是测试失败。

4 并发 profile 使用实测可用的 20,480 总 token 池。以下五组可以运行：

```text
(2048 + 512)  * 4 = 10240
(2048 + 1024) * 4 = 12288
(2048 + 2048) * 4 = 16384
(4096 + 512)  * 4 = 18432
(4096 + 1024) * 4 = 20480
```

`(4096 + 2048) * 4 = 24576` 超过实测 c4 token 池，因此跳过。测试前通过
`/server_info` 验证了 c4 服务同时报告 `max_total_num_tokens=20480` 和
`effective_max_running_requests_per_dp=4`。

## 真实数据集

ShareGPT 使用数据集中的自然回答长度。LongBench-v2 分别运行 SGLang 官方的
10-token 短答案设置，以及同一批 prompt 固定生成 512 tokens 的设置。

| 数据集 | 请求数 / 最大并发 | 中位输入 / 输出 | 端到端 output tok/s，关闭 / MTP | 加速 | 中位 TTFT，关闭 / MTP | 平均 TPOT，关闭 / MTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 30 / 1 | 204 / 218.5 | 40.51 / 75.60 | 1.87x | 212.21 / 231.14 ms | 23.57 / 11.92 ms |
| ShareGPT | 20 / 4 | 239 / 280 | 133.59 / 224.17 | 1.68x | 258.70 / 279.87 ms | 26.80 / 14.76 ms |
| LongBench-v2 短答案 | 20 / 1 | 17844 / 10 | 2.219 / 2.181 | 0.98x | 4401.83 / 4593.94 ms | 24.38 / 13.86 ms |
| LongBench-v2 固定 512 | 20 / 1 | 17844 / 512 | 30.31 / 48.19 | 1.59x | 4438.87 / 4611.11 ms | 24.60 / 12.01 ms |

LongBench-v2 使用已改名的
[`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2)。
先读取 70 条 `short` 数据，其中 28 条满足 `prompt + 512 <= 24576`，再以
seed 42 固定选择 20 条。输入范围为 10,954-22,842 tokens，中位 17,844。
最终本地 JSONL 的 SHA-256 是
`93ecd8a799ba868cfb2a1c28a38ce87653cb30eb211712030970020d39990058`。
ShareGPT 源文件 SHA-256 是
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。

## 这是不是重新移植 MTP

不是。本次结果没有移植任何 SGLang 推理内核。官方 SGLang v0.5.16 已经包含
`qwen3_5_mtp.py`、Qwen3.5/Qwen3.6 模型映射和 EAGLE speculative 调度。
Qwen3.6 在引擎内部使用架构名 `Qwen3_5ForConditionalGeneration`。

本地 SGLang 分支基于官方 tag `v0.5.16`，对应 commit
`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`。本地两个 commit 只新增了
`profiles/170hx/*`；执行 `git diff v0.5.16..HEAD -- python/sglang` 没有输出。
实测 head 是 `9a5776c3cf04e7e05fa7f7166b1b3a0f1a6226d1`。本仓库真正完成的是
CMP 170HX 的显存安全启动参数、容量验证、受控对比和遥测。

模型内的 `mtp.safetensors` 是训练好的 draft 权重，但“模型带 MTP 权重”不等于
“所有推理引擎自动支持 MTP”。推理引擎仍需要实现 draft model、speculative
scheduler、verification 和 SM80 可用的内核。

[`vllm-dsa-mtp-sm80`](https://github.com/allover326/vllm-dsa-mtp-sm80) 处理的是
另一套 vLLM 问题：它把面向 SM80 的 Triton sparse-MLA/DSA 后端，与 GLM-5.2、
DeepSeek 系稀疏模型的 pipeline-parallel MTP 修复组合到一起。这里的
Qwen3.6-27B 是单卡稠密混合 GDN 模型，使用 SGLang 已有的 SM80 Triton GDN
路径。本仓库没有复制该 vLLM 项目的 patch。

SGLang 按模型架构和 CUDA capability 选择实现，并不按 `CMP 170HX` 这个商品名
写专门支持列表。CMP 170HX 和 A100 都是 SM80，但本文只对实际测试的显卡、
模型和软件组合负责，不直接外推 A100 的显存、频率和吞吐。

## 测试环境与时间

固定长度受控矩阵的请求窗口：

```text
2026-07-31 17:56:30 至 19:54:15，Asia/Shanghai（UTC+08:00）
```

真实数据集的请求窗口：

```text
2026-07-31 22:08:11 至 22:35:41，Asia/Shanghai（UTC+08:00）
```

| 项目 | 实测值 |
| --- | --- |
| GPU | NVIDIA GA100 `[CMP 170HX]`，PCI ID `10de:2082`，SM80 |
| 显存 | 40960 MiB |
| 功率上限 | 所有原始遥测都为 250 W；允许范围 100-250 W |
| 最高 SM / 显存频率 | 1410 / 1215 MHz |
| 测试时 PCIe | 2.5 GT/s x4；端点能力 5 GT/s x16 |
| 宿主环境 | KVM 虚拟机，Debian 12，Linux 6.1.0-51-amd64 |
| CPU / 内存 | 44 vCPU Intel Xeon E5-2680 v4 / 24 GiB |
| Swap | **总量 0 B，使用 0 B** |
| 驱动 / Toolkit | NVIDIA 610.43.02 / CUDA toolkit 13.3.73 |
| Python / PyTorch | 3.11.2 / 2.11.0+cu130 |
| SGLang / Triton / FlashInfer | 0.5.16 / 3.6.0 / 0.6.14 |
| 模型 | `Qwen3.6-27B-INT8-W8A8`，compressed-tensors 动态 W8A8 |
| Target / draft 加载 | target 权重 28.48 GiB / draft worker 权重 5.53 GiB |

v2 所有 profile 都关闭 radix/prefix cache，但这不表示 GPU state 为 0：服务仍会
按需分配 KV cache 和 Qwen 的 GDN/Mamba state。每个 profile 启动时都没有其他
GPU 计算进程，结束后显存恢复为 0。

### 250 W 遥测摘录

功耗统计只使用各 case 时间窗内 GPU 利用率不低于 90% 的采样；公开 CSV 同时
保留活跃样本数和总样本数。

| 用例 | 活跃平均 / P95 功耗 | 平均 GPU 利用率 | 平均 SM 频率 | 最高温度 |
| --- | ---: | ---: | ---: | ---: |
| MTP，1024/8192，c1 | 239.57 / 255.56 W | 94.6% | 1392 MHz | 84 C |
| 关闭，1024/8192，c1 | 247.42 / 251.95 W | 100.0% | 1409 MHz | 80 C |
| MTP，4096/1024，c4 | 235.76 / 254.55 W | 94.6% | 1389 MHz | 78 C |
| 关闭，4096/1024，c4 | 241.66 / 250.67 W | 99.8% | 1398 MHz | 81 C |
| MTP，LongBench 固定 512 | 241.18 / 258.18 W | 96.2% | 1360 MHz | 81 C |
| 关闭，LongBench 固定 512 | 246.84 / 255.90 W | 99.7% | 1376 MHz | 82 C |

设置的功率上限并不是对每个 NVML 瞬时上报值的硬截断，短时采样可能超过
250 W。40,172 行原始遥测的控制上限字段全部为 250.00 W。采样间隔的中位数和
P95 分别为 200-201 ms。

此前的 120 W 观察值没有混入本数据集。由于当时没有保留完整原始遥测，不能做
严格的受控功率比较。本文所有表格都已在 250 W 下重新运行。

## 测试方法

测试驱动调用 `python -m sglang.benchmark.serving` 和 `sglang-oai` 后端。固定
长度用例使用 SGLang random 数据集生成器，并遵守以下统一条件：

- `--random-range-ratio 1.0`、seed 42、temperature 0.0、top-p 1.0；
- 请求发送速率不限制，只由 `--max-concurrency` 限流；
- 单请求和 prefill 每个 cell 3 个请求；
- 4 并发每个 cell 20 个请求；
- 服务已初始化，但 benchmark warmup 请求数为 0；
- 每个 case 都执行 `--flush-cache`，同时服务端关闭 radix cache；
- 保存完整 JSONL，包括每个请求和逐 token 延迟数组。

benchmark warmup 设为 0，是为了规避该 SGLang 版本中实测到的一个竞态：客户端
流式 warmup 刚结束时，服务端可能尚未释放请求，紧接着执行 `/flush_cache` 会
返回 HTTP 400。服务启动阶段自己的 warmup 仍然启用。

`scripts/monitor-gpu.sh` 通过 `nvidia-smi -lms 200` 采集带毫秒的时间戳、功耗
与功率限制、GPU 利用率、SM/显存频率、温度和 P-state。manifest 保存每个 case
精确的开始/结束 epoch 毫秒，以及每次 profile 启动对应的唯一遥测文件。

## 启动 Profile

| Profile | 上下文 | 总 token 池 | 最大运行请求 | 静态显存比例 | MTP |
| --- | ---: | ---: | ---: | ---: | --- |
| `mtp` | 24576 | 24576 | 1 | 0.94 | EAGLE，3 steps，top-k 1，4 draft |
| `baseline` | 24576 | 24576 | 1 | 0.88 | 关闭 |
| `mtp-c4` | 24576 | 20480 | 4 | 0.98 | EAGLE，3 steps，top-k 1，4 draft |
| `baseline-c4` | 24576 | 20480 | 4 | 0.88 | 关闭 |

四个 profile 都使用 page size 64、2048-token chunked prefill、4096-token
prefill 上限，关闭 prefill CUDA Graph 和 radix cache。c4 profile 捕获 batch
1、2、3、4 的 decode graph。

MTP c4 的静态显存比例 0.98 是实测选择：使用 0.94 时，SGLang 会把有效并发从
4 自动降到 1；0.98 配合 24,576 token 池时，draft KV 分配会失败；0.98 配合
20,480 token 池可以干净启动，报告有效并发 4，并完成本文所有 c4 用例。

## 复现代码

准备 SGLang 0.5.16 环境和模型 checkpoint：

```bash
export SGLANG_RUNTIME_HOME=/mnt/ss32t/SGLang
export SGLANG_VENV=/mnt/ss32t/SGLang/venv
export SGLANG_SOURCE=/path/to/sglang-v0.5.16
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8
```

完整脚本会验证 250 W、swap 为 0、没有其他 GPU 进程，以及服务端实际 token/
请求容量，但不会修改固件或解锁功率：

```bash
nvidia-smi -q -d POWER
sudo -E env RUN_ID=expanded-250w-v2-20260731 RUN_REAL_DATASETS=0 \
  scripts/run-250w-suite.sh
```

从两个 datasets-server 响应准备完全相同的 LongBench 子集：

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

使用同一个 run ID 追加真实数据测试；已经完成的固定用例会直接复用：

```bash
sudo -E env RUN_ID=expanded-250w-v2-20260731 RUN_REAL_DATASETS=1 \
  SHAREGPT_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  LONGBENCH_PATH=/tmp/longbench-fit20.jsonl \
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

启动脚本在模型加载时临时设置 `vm.overcommit_memory=1`，服务启动完成或退出后
恢复原值。为了保证恢复操作成功，脚本要求 root 权限。

## 仓库内容

```text
scripts/                         启动、benchmark、监控、数据准备脚本
environment/expanded-250w-v2-*  实测宿主、软件和模型信息
results/case-manifest-*-v2-*     case 时间窗、状态、文件、数据集 SHA
results/serving-expanded-250w.csv
results/power-expanded-250w.csv
results/telemetry/*-v2-*         原始 200 ms NVML 遥测
```

原始 SGLang JSONL 和服务日志继续由 gitignore 排除，因为其中包含大型逐 token
数组、生成文本、本机绝对路径和完整服务配置。公开紧凑 CSV 由这些原始文件汇总。

## 局限性

- 这里只测试了一张显卡、一个 checkpoint、一套软件和一个功率状态，不能当成
  通用 GPU 排名。
- 固定单请求 cell 的 3 次重复适合选择 profile，不足以给出跨系统置信区间。
- 固定 token random prompt 用于隔离服务性能，不能代表所有生产分布，因此又
  补充了 ShareGPT 和 LongBench。
- 两个不到 1 秒的 512-token prefill case 没有利用率不低于 90% 的遥测采样，
  所以其活跃功耗字段按事实留空。
- A100 与 CMP 170HX 同为 SM80，但显存、频率、PCIe、散热和固件不同，本文没有
  实测 A100。
- 固件修改和功率上限解锁不属于本仓库范围。

## 许可证与致谢

代码使用 Apache-2.0。SGLang 由
[SGLang project](https://github.com/sgl-project/sglang) 开发。Qwen3.6、
ShareGPT 和 LongBench-v2 分别遵循其各自许可证。
