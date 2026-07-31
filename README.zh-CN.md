# CMP 170HX + SGLang + Qwen3.6-27B W8A8 推理测试

[English](README.md) | [简体中文](README.zh-CN.md)

这是 `Qwen3.6-27B-INT8-W8A8` 在单张 40 GiB NVIDIA CMP 170HX
（GA100、SM80）上运行 SGLang 0.5.16 的 `baseline`（MTP 关闭）与 `mtp`
（MTP 开启）对照测试。标准运行的设定功率上限为 250 W。本文所有表格都来自标准运行
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

每个 cell 完成 3 个请求。`MTP 关闭` 列来自 `baseline` profile，`MTP 开启`
列来自 `mtp` profile；两者使用完全相同的输入长度、输出长度、seed 和采样参数。
加速比均按 `MTP 开启值 / MTP 关闭值` 计算。

| Prompt 长度 (tokens) | 请求生成长度 (tokens) | 端到端生成吞吐，MTP 关闭 (output tok/s) | 端到端生成吞吐，MTP 开启 (output tok/s) | MTP 加速比 | 稳态解码速率，MTP 关闭 (tok/s) | 稳态解码速率，MTP 开启 (tok/s) | 平均 TTFT，MTP 关闭 (ms) | 平均 TTFT，MTP 开启 (ms) | 平均 TPOT，MTP 关闭 (ms) | 平均 TPOT，MTP 开启 (ms) | MTP 接受长度 (tokens) |
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

### Prefill 曲线

每个 cell 完成 3 个请求，并且每个请求严格只生成 1 个 token。表中的“系统级
Prompt 处理吞吐”是 `全部 prompt tokens / 整个 benchmark 请求窗口的墙钟时间`。
它包含服务请求处理和首个生成 token 的返回时间，是端到端 Prefill 代理指标，
不是单个 Prefill kernel 的纯吞吐。MTP 会加载 draft worker，但 speculative
decode 无法加速 prompt prefill。TTFT 变化按
`(MTP 开启 - MTP 关闭) / MTP 关闭` 计算，正数表示 MTP 更慢。

| Prompt 长度 (tokens) | 系统级 Prompt 处理吞吐，MTP 关闭 (input tok/s) | 系统级 Prompt 处理吞吐，MTP 开启 (input tok/s) | 中位 TTFT，MTP 关闭 (ms) | 中位 TTFT，MTP 开启 (ms) | MTP TTFT 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2326.17 | 2119.08 | 207.74 | 234.02 | +12.7% |
| 2048 | 4428.56 | 4122.16 | 444.77 | 486.97 | +9.5% |
| 4096 | 4561.56 | 4296.73 | 881.46 | 937.63 | +6.4% |
| 8192 | 4460.70 | 4245.31 | 1809.20 | 1905.02 | +5.3% |
| 12288 | 4356.72 | 4142.83 | 2800.57 | 2946.06 | +5.2% |
| 20480 | 4112.76 | 3911.91 | 4968.85 | 5207.51 | +4.8% |

### 最大并发 4

每个 cell 完成 20 个请求，相当于按配置的最大并发 4 跑满 5 轮。聚合生成吞吐
是所有并发请求的输出 tokens 总和除以 benchmark 墙钟时间。平均 TPOT 是
per-request 指标，四并发下不能用 `1000 / TPOT` 代替聚合吞吐。`实际并发` 是
benchmark 的时间加权平均值，有限轮次的启动和排空会让部分数值低于服务端
已验证的并发上限 4。

| Prompt 长度 (tokens/请求) | 请求生成长度 (tokens/请求) | 聚合生成吞吐，MTP 关闭 (output tok/s) | 聚合生成吞吐，MTP 开启 (output tok/s) | MTP 加速比 | 平均 TPOT，MTP 关闭 (ms) | 平均 TPOT，MTP 开启 (ms) | 中位 TTFT，MTP 关闭 (ms) | 中位 TTFT，MTP 开启 (ms) | 实际并发，MTP 关闭 | 实际并发，MTP 开启 | MTP 接受长度 (tokens) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 512 | 134.87 | 222.57 | 1.65x | 27.24 | 16.10 | 1310.1 | 524.5 | 4.00 | 3.89 | 3.03 |
| 2048 | 1024 | 142.78 | 254.04 | 1.78x | 26.80 | 14.73 | 1313.6 | 521.3 | 4.00 | 3.92 | 3.05 |
| 2048 | 2048 | 146.73 | 269.31 | 1.84x | 26.65 | 13.91 | 1318.5 | 527.8 | 4.00 | 3.84 | 3.10 |
| 4096 | 512 | 92.11 | 174.34 | 1.89x | 27.44 | 17.62 | 2684.6 | 1133.3 | 3.73 | 3.73 | 3.14 |
| 4096 | 1024 | 100.23 | 179.24 | 1.79x | 26.75 | 14.74 | 2674.2 | 3193.2 | 3.73 | 3.75 | 3.04 |

## 吞吐指标怎么理解

本文使用三个不同的吞吐/速率指标：

- CSV 中的 `e2e_output_tok_s`，即端到端生成吞吐，等于
  `所有已完成请求的生成 tokens 总数 / benchmark 请求窗口墙钟时间`。单请求结果
  包含 TTFT；四并发结果是服务器的聚合输出吞吐，不是单个请求的 decode 速率。
- CSV 中的 `input_tok_s`，即系统级 Prompt 处理吞吐，等于
  `所有已完成请求的 prompt tokens 总数 / 同一个 benchmark 请求窗口墙钟时间`。
  Prefill 曲线每个请求仍生成 1 个 token，所以它不是 kernel-only Prefill 吞吐。
- 仅用于单请求对比的稳态解码速率等于 `1000 / 平均 TPOT`。它排除了首 token
  之前的时间，近似反映进入 decode 后的持续生成速度。

所有吞吐加速比都按 `MTP 开启 / MTP 关闭` 计算；TTFT/TPOT 越低越好，吞吐和
稳态解码速率越高越好。

以 Prompt 长度 20,480 tokens、请求生成长度 1,024 tokens 的单请求为例，MTP
开启时平均 TTFT 为 5289.5 ms，平均 TPOT 为 11.40 ms，
所以端到端速度是 60.38 tok/s，而稳态 decode 是 87.74 tok/s。此前看到的
Prompt 长度 20,480、请求生成长度 128 的历史结果只有 18.81 tok/s，因为只生成了
128 tokens，却同样承担了约 5.3 秒 prefill；
它是端到端结果，不代表 decode 上限只有 18.81 tok/s。

`4 draft tokens` 也不是最大输出长度。它表示每个 speculative cycle 提议的
候选数；cycle 会一直重复，直到真正生成请求指定的 1024、4096 或 8192 tokens。

## 容量与跳过项

单请求 profile 的上下文和总 token 池都是 24,576。实测该服务要求
`输入 + 输出 < 24576`，等于 24,576 时会返回 HTTP 400。因此
Prompt 20,480 + 生成 4,096，以及 Prompt 20,480 + 生成 8,192，被明确记录为
容量跳过，而不是测试失败。

4 并发 profile 使用实测可用的 20,480 总 token 池。以下五组可以运行：

```text
(2048 + 512)  * 4 = 10240
(2048 + 1024) * 4 = 12288
(2048 + 2048) * 4 = 16384
(4096 + 512)  * 4 = 18432
(4096 + 1024) * 4 = 20480
```

`(4096 + 2048) * 4 = 24576` 超过最大并发 4 profile 的实测 token 池，因此
跳过。测试前通过 `/server_info` 验证了该服务同时报告
`max_total_num_tokens=20480` 和
`effective_max_running_requests_per_dp=4`。

## 真实数据集

ShareGPT 使用数据集中的自然回答长度。LongBench-v2 分别运行 SGLang 官方的
10-token 短答案设置，以及同一批 prompt 固定生成 512 tokens 的设置。

两种 profile 使用相同的 prompt、请求输出长度和采样参数。中位 Prompt 长度和
中位实际生成长度对两个 profile 相同；最大并发 1 的行是端到端吞吐，最大并发
4 的行是聚合吞吐。

| 数据集 | 请求数 | 最大并发 | 中位 Prompt 长度 (tokens) | 中位实际生成长度 (tokens) | 生成吞吐，MTP 关闭 (output tok/s) | 生成吞吐，MTP 开启 (output tok/s) | MTP 加速比 | 中位 TTFT，MTP 关闭 (ms) | 中位 TTFT，MTP 开启 (ms) | 平均 TPOT，MTP 关闭 (ms) | 平均 TPOT，MTP 开启 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 30 | 1 | 204 | 218.5 | 40.51 | 75.60 | 1.87x | 212.21 | 231.14 | 23.57 | 11.92 |
| ShareGPT | 20 | 4 | 239 | 280 | 133.59 | 224.17 | 1.68x | 258.70 | 279.87 | 26.80 | 14.76 |
| LongBench-v2 短答案 | 20 | 1 | 17844 | 10 | 2.219 | 2.181 | 0.98x | 4401.83 | 4593.94 | 24.38 | 13.86 |
| LongBench-v2 固定生成 512 | 20 | 1 | 17844 | 512 | 30.31 | 48.19 | 1.59x | 4438.87 | 4611.11 | 24.60 | 12.01 |

LongBench-v2 使用已改名的
[`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2)。
先读取 70 条 `short` 数据，其中 28 条满足 `prompt + 512 <= 24576`，再以
seed 42 固定选择 20 条。输入范围为 10,954-22,842 tokens，中位 17,844。
最终本地 JSONL 的 SHA-256 是
`93ecd8a799ba868cfb2a1c28a38ce87653cb30eb211712030970020d39990058`。
ShareGPT 源文件 SHA-256 是
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。

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
| 标准运行的设定功率上限 | 250 W |
| 最高 SM 频率 | 1410 MHz |
| 最高显存频率 | 1215 MHz |
| 测试时 PCIe | 2.5 GT/s x4；端点能力 5 GT/s x16 |
| 宿主环境 | KVM 虚拟机，Debian 12，Linux 6.1.0-51-amd64 |
| CPU | 44 vCPU Intel Xeon E5-2680 v4 |
| 主机内存 | 24 GiB |
| Swap | **总量 0 B，使用 0 B** |
| NVIDIA 驱动 | 610.43.02 |
| CUDA Toolkit | 13.3.73 |
| Python | 3.11.2 |
| PyTorch | 2.11.0+cu130 |
| SGLang | 0.5.16 |
| Triton | 3.6.0 |
| FlashInfer | 0.6.14 |
| Checkpoint | [`Avesed/Qwen3.6-27B-INT8-W8A8`](https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8)，compressed-tensors 动态 W8A8 |
| 基础模型 | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Target 权重加载 | 28.48 GiB |
| Draft worker 权重加载 | 5.53 GiB |

### 250 W 遥测摘录

功耗统计只使用各 case 时间窗内 GPU 利用率不低于 90% 的采样；公开 CSV 同时
保留活跃样本数和总样本数。

| Serving profile | MTP 状态 | 工作负载 | Prompt 长度 (tokens) | 请求生成长度 (tokens) | 最大并发 | 活跃平均功耗 (W) | 活跃 P95 功耗 (W) | 平均 GPU 利用率 | 平均 SM 频率 (MHz) | 最高温度 (C) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mtp` | MTP 开启 | 固定长度 | 1024 | 8192 | 1 | 239.57 | 255.56 | 94.6% | 1392 | 84 |
| `baseline` | MTP 关闭 | 固定长度 | 1024 | 8192 | 1 | 247.42 | 251.95 | 100.0% | 1409 | 80 |
| `mtp-c4` | MTP 开启 | 固定长度 | 4096 | 1024 | 4 | 235.76 | 254.55 | 94.6% | 1389 | 78 |
| `baseline-c4` | MTP 关闭 | 固定长度 | 4096 | 1024 | 4 | 241.66 | 250.67 | 99.8% | 1398 | 81 |
| `mtp` | MTP 开启 | LongBench-v2 固定生成 | 中位 17844 | 512 | 1 | 241.18 | 258.18 | 96.2% | 1360 | 81 |
| `baseline` | MTP 关闭 | LongBench-v2 固定生成 | 中位 17844 | 512 | 1 | 246.84 | 255.90 | 99.7% | 1376 | 82 |

## 测试方法

测试驱动调用 `python -m sglang.benchmark.serving` 和 `sglang-oai` 后端。固定
长度用例使用 SGLang random 数据集生成器，并遵守以下统一条件：

- `--random-range-ratio 1.0`、seed 42、temperature 0.0、top-p 1.0；
- 请求发送速率不限制，只由 `--max-concurrency` 限流；
- 单请求和 prefill 每个 cell 3 个请求；
- 4 并发每个 cell 20 个请求；
- 服务已初始化，但 benchmark warmup 请求数为 0；
- 每个 case 都执行 `--flush-cache`；
- 保存完整 JSONL，包括每个请求和逐 token 延迟数组。

benchmark warmup 设为 0，是为了规避该 SGLang 版本中实测到的一个竞态：客户端
流式 warmup 刚结束时，服务端可能尚未释放请求，紧接着执行 `/flush_cache` 会
返回 HTTP 400。服务启动阶段自己的 warmup 仍然启用。

`scripts/monitor-gpu.sh` 通过 `nvidia-smi -lms 200` 采集带毫秒的时间戳、功耗
与功率限制、GPU 利用率、SM/显存频率、温度和 P-state。manifest 保存每个 case
精确的开始/结束 epoch 毫秒，以及每次 profile 启动对应的唯一遥测文件。

## 启动 Profile

| Profile | 上下文长度 | 总 token 池 | 最大运行请求数 | 静态显存比例 | MTP 状态 | 推测解码设置 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `mtp` | 24576 | 24576 | 1 | 0.94 | MTP 开启 | EAGLE，3 steps，top-k 1，4 draft tokens |
| `baseline` | 24576 | 24576 | 1 | 0.88 | MTP 关闭 | 不适用 |
| `mtp-c4` | 24576 | 20480 | 4 | 0.98 | MTP 开启 | EAGLE，3 steps，top-k 1，4 draft tokens |
| `baseline-c4` | 24576 | 20480 | 4 | 0.88 | MTP 关闭 | 不适用 |

四个 profile 都使用 page size 64、2048-token chunked prefill、4096-token
prefill 上限，并关闭 prefill CUDA Graph。最大并发 4 的 `mtp-c4` 和
`baseline-c4` 捕获 batch size 1、2、3、4 的 decode graph。

`mtp-c4` 的静态显存比例 0.98 是实测选择：使用 0.94 时，SGLang 会把有效并发
从 4 自动降到 1；0.98 配合 24,576 token 池时，MTP worker 内存分配失败；
0.98 配合 20,480 token 池可以启动，报告有效并发 4，并完成本文所有最大并发
4 用例。

## 复现代码

准备 SGLang 0.5.16 环境和模型 checkpoint：

```bash
export SGLANG_RUNTIME_HOME=/mnt/ss32t/SGLang
export SGLANG_VENV=/mnt/ss32t/SGLang/venv
export SGLANG_SOURCE=/path/to/sglang-v0.5.16
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8
```

完整脚本会验证 250 W、swap 为 0，以及服务端实际 token/请求容量，但不会修改
固件或解锁功率：

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

## 附录：历史 120 W 结果

下表保留 Git 提交 `79a164a` 中记录的 120 W serving 结果；上面的标准结果表均为
250 W。

| MTP 状态 | 推测解码设置 | Prompt 长度 (tokens) | 请求生成长度 (tokens) | 最大并发 | 端到端或聚合生成吞吐 (output tok/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| MTP 关闭 | 不适用 | 128 | 256 | 1 | 30.33 |
| MTP 开启 | EAGLE，3 steps，4 draft tokens | 128 | 256 | 1 | 66.15 |
| MTP 关闭 | 不适用 | 1024 | 256 | 4 | 99.79 |

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
