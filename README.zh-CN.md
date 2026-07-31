# CMP 170HX + SGLang Qwen3.6-27B W8A8 推理测试

[English](README.md) | [简体中文](README.zh-CN.md)

这是 Qwen3.6-27B-INT8-W8A8 在单张 NVIDIA CMP 170HX（GA100、SM80、
40 GiB）上运行 SGLang 的可复现推理测试。仓库包含 2026 年 7 月 31 日
实测使用的启动配置、基准驱动、250W 结果表，以及原始 200 ms 功耗遥测。

## 工作范围

这不是一个新的 MTP 内核移植仓库。SGLang 0.5.16 已经包含 Qwen3.5/Qwen3.6
MTP 模型路径和 EAGLE 投机解码支持；Qwen3.6 在代码中的内部架构名称是
`Qwen3_5ForConditionalGeneration`。本仓库完成的工作是：

- 为 40 GiB CMP 170HX 提供不会超显存的 SM80 启动 profile；
- 在完全相同请求上控制变量对比 MTP 与非 MTP；
- 分离单请求 decode、prefill 和 4 并发吞吐配置；
- 固定随机种子、输入输出长度和缓存处理方法；
- 同步采集 NVML 功耗、频率、利用率和温度遥测。

本仓库没有复制 `vllm-dsa-mtp-sm80` 的代码。那个项目处理 vLLM、
DeepSeek/GLM 稀疏注意力及相关 SM80 路径；这里的 checkpoint 是稠密混合 GDN
模型，使用的是 SGLang 已有的 SM80 Triton GDN 内核。

## 主要结论

- 128 token 输入、生成 128-512 token、单并发时，MTP 输出吞吐提升
  **1.96x 到 2.03x**。
- MTP 把平均 TPOT 从约 **23.9 ms 降至 11.0-11.5 ms**；短输入的 TTFT
  增加 13-43 ms。
- 10K 和 20K 输入的端到端输出吞吐分别提升 1.40x 和 1.22x。此时请求主要
  消耗在 prefill，虽然纯 decode TPOT 仍然接近减半，但整体收益会被稀释。
- 非 MTP 的 4 并发 profile 在 20 个固定 1024/256 请求中达到
  **132.95 output tok/s**。
- 8K prefill 达到 **4573.80 input tok/s**，跨两个 4096-token chunk 后仍保留
  4K 档 98.6% 的吞吐。

## 测试时间

实际请求计时窗口：

```text
2026-07-31 14:56:44 至 15:25:32，Asia/Shanghai（UTC+08:00）
```

每个 profile 都从无其他 GPU 计算进程的干净状态启动。整套测试结束后，GPU
显存占用恢复为 0，`vm.overcommit_memory` 恢复为 `0`。

## 测试环境

### 硬件与宿主机

| 项目 | 实测值 |
| --- | --- |
| GPU PCI 身份 | NVIDIA GA100 `[CMP 170HX]`，设备 ID `10de:2082` |
| `nvidia-smi` 名称 | `NVIDIA Graphics Device`（当前固件枚举名称） |
| 计算能力 | 8.0 / SM80 |
| 显存 | 40960 MiB |
| 最高 SM / 显存频率 | 1410 / 1215 MHz |
| 功率状态 | 当前/请求/默认均为 250W；允许范围 100-250W |
| 测试时 PCIe 链路 | 2.5 GT/s x4；端点能力 5 GT/s x16 |
| 宿主环境 | KVM 虚拟机，Debian 12，Linux 6.1.0-51-amd64 |
| CPU | 44 vCPU，Intel Xeon E5-2680 v4 |
| 主机内存 | 24 GiB RAM，975 MiB swap |

降速后的 PCIe 链路会影响 checkpoint 加载和主机到设备的数据传输。稳态 decode
期间权重和活跃状态驻留在 GPU，但跨机器比较时仍应保留这项环境差异。

### 软件版本

| 组件 | 版本 |
| --- | --- |
| NVIDIA 驱动 / CUDA UMD | 610.43.02 / 13.3 |
| CUDA Toolkit | 13.3.73 |
| Python | 3.11.2 |
| PyTorch | 2.11.0+cu130 |
| SGLang | 0.5.16 |
| SGLang 引擎 commit | `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` |
| Triton | 3.6.0 |
| FlashInfer | 0.6.14 |

NVIDIA 用户态驱动和 Toolkit 报告 CUDA 13.3；本次 PyTorch wheel 是按 CUDA
13.0 构建的。

### 模型

| 项目 | 值 |
| --- | --- |
| Checkpoint | `Qwen3.6-27B-INT8-W8A8` |
| 架构 | `Qwen3_5ForConditionalGeneration`，稠密混合 GDN |
| 文本配置 | 64 层、hidden size 5120、每 4 层一次全注意力 |
| 量化 | compressed-tensors W8A8：权重按通道 INT8，激活动态逐 token INT8 |
| Target 权重文件 | 30,394,496,808 bytes；加载后 GPU 权重 28.48 GiB |
| MTP 文件 | 849,400,424 bytes；draft worker 加载后 GPU 权重 5.53 GiB |
| `config.json` SHA-256 | `c3f53d4c340b3cac0c42dac9f66610acb26e2519b58a9fa20fcd87e706f3f979` |

本仓库不分发模型 checkpoint。

## 启动 Profile

| Profile | 上下文 | 最大运行请求 | MTP | 关键设置 |
| --- | ---: | ---: | --- | --- |
| `mtp` | 24,576 | 1 | EAGLE，3 steps，top-k 1，4 draft tokens | page 64，关闭 prefill graph |
| `baseline` | 24,576 | 1 | 关闭 | page 1，关闭 prefill graph |
| `prefill` | 32,768 | 1 | 关闭 | 4096 chunk，开启 prefill CUDA Graph |
| `throughput` | 32,768 | 4 | 关闭 | mixed chunk，batch 1/2/4 decode graph |

MTP 只开放 24K，是因为 target 和 draft worker 在 KV cache、Mamba state 和
CUDA Graph 分配之前就合计消耗约 34 GiB。对这张 40 GiB 卡宣称 32K MTP
上下文会超过实测 token 容量。

## 测试方法

Serving benchmark 使用 `python -m sglang.benchmark.serving`、`sglang-oai`
后端和 SGLang 的 `random` 数据集生成器。

统一控制条件：

- 输入和输出 token 长度严格固定，`--random-range-ratio 1.0`；
- seed 42、temperature 0.0、top-p 1.0；
- 请求发送速率无限制，仅由 `--max-concurrency` 限流；
- 每个正式用例之前清空 cache；
- SGLang 服务已经完成初始化，但 benchmark warmup 请求数设为 0；
- 短 decode 和每个 prefill 长度各 5 个请求；
- 10K/20K 长上下文各 3 个请求；
- 4 并发使用 20 个请求，共 5 轮；
- 每个用例都在本地写出完整 JSONL 细节。

benchmark warmup 设为 0 的原因：在这个 SGLang 版本中，流式 warmup 可能刚在
客户端结束，服务端还没有释放请求；基准程序紧接着调用 `/flush_cache` 会返回
HTTP 400，从而保留 warmup prefix。本文发布的长上下文结果确认 cache flush
成功，三次 TTFT 分布紧密，没有 prefix cache 命中。

GPU 监控每 200 ms 采集以下字段：

```text
timestamp,power.draw,power.limit,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,pstate
```

功耗汇总只包含 GPU 利用率不低于 90% 的样本，排除服务空闲和模型加载时间。
百分位使用排序后向下取最近秩的样本。

## 复现方法

准备包含 CUDA 依赖的 SGLang 0.5.16 环境和模型 checkpoint，然后设置路径。
`SGLANG_SOURCE` 是可选变量，用来指定源代码 checkout；不设置时使用 venv
中安装的 SGLang Python 包。

```bash
export SGLANG_RUNTIME_HOME=/mnt/ss32t/SGLang
export SGLANG_VENV=/mnt/ss32t/SGLang/venv
export SGLANG_SOURCE=/path/to/sglang-v0.5.16
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8
```

运行前确认功率状态。整套脚本只校验 250W，不会修改显卡功率上限或固件：

```bash
nvidia-smi -q -d POWER
```

运行完整矩阵：

```bash
sudo -E scripts/run-250w-suite.sh
```

也可以手动启动和测试单个 profile：

```bash
sudo -E env PROFILE=mtp scripts/serve-qwen36-27b.sh

RUN_TAG=mtp-128-256 WARMUP_REQUESTS=0 \
DECODE_INPUT_LEN=128 DECODE_OUTPUT_LEN=256 DECODE_PROMPTS=5 \
scripts/bench-serving.sh decode
```

服务启动脚本在加载模型时临时设置 `vm.overcommit_memory=1`，启动完成或退出时
恢复原值。为了保证能够恢复 sysctl，服务脚本要求 root 权限。

## 测试结果

### 单请求 Decode

| 输入/输出 | MTP | Output tok/s | 平均 TPOT | 平均 TTFT | 接受长度 | 加速比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128/128 | 关闭 | 38.99 | 23.79 ms | 249.70 ms | - | - |
| 128/128 | 3 steps / 4 draft | 76.59 | 11.04 ms | 262.26 ms | 3.26 | 1.96x |
| 128/256 | 关闭 | 40.42 | 23.93 ms | 223.86 ms | - | - |
| 128/256 | 3 steps / 4 draft | 81.17 | 11.30 ms | 266.53 ms | 3.23 | 2.01x |
| 128/512 | 关闭 | 41.07 | 23.94 ms | 225.91 ms | - | - |
| 128/512 | 3 steps / 4 draft | 83.45 | 11.48 ms | 262.16 ms | 3.18 | 2.03x |

### 长上下文

| 输入/输出 | MTP | Input tok/s | Output tok/s | 平均 TTFT | 平均 TPOT | 接受长度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10240/128 | 关闭 | 1856.52 | 23.21 | 2404.65 ms | 24.37 ms | - |
| 10240/128 | 3 steps / 4 draft | 2599.55 | 32.49 | 2466.28 ms | 11.52 ms | 3.23 |
| 20480/128 | 关闭 | 2476.88 | 15.48 | 5103.24 ms | 24.83 ms | - |
| 20480/128 | 3 steps / 4 draft | 3009.29 | 18.81 | 5274.62 ms | 11.97 ms | 3.16 |

### Prefill

每个请求严格生成一个输出 token。

| 输入 token | Input tok/s | 平均 TTFT |
| ---: | ---: | ---: |
| 128 | 890.24 | 136.40 ms |
| 256 | 1922.66 | 127.62 ms |
| 512 | 3254.82 | 150.71 ms |
| 1024 | 4020.76 | 248.31 ms |
| 2048 | 4418.20 | 457.41 ms |
| 4096 | 4639.95 | 876.43 ms |
| 8192 | 4573.80 | 1784.12 ms |

### 4 并发

20 个固定 1024/256 请求在 38.51 秒内完成：

| Output tok/s | Input tok/s | Total tok/s | 平均 TTFT | 平均 TPOT | 实际平均并发 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 132.95 | 531.80 | 664.75 | 764.86 ms | 27.12 ms | 3.99 |

### 250W 功耗遥测

| Profile 工作负载 | 平均功耗 | 中位功耗 | P95 功耗 | 平均 SM 频率 | 最高温度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP decode 全套 | 221.86 W | 216.06 W | 258.15 W | 1376 MHz | 66 C |
| 非 MTP decode 全套 | 229.52 W | 232.62 W | 253.24 W | 1394 MHz | 75 C |
| Prefill 曲线 | 238.91 W | 249.40 W | 264.60 W | 1313 MHz | 60 C |
| 4 并发 mixed | 220.43 W | 217.78 W | 253.93 W | 1402 MHz | 65 C |

250W 是控制上限，并不保证每个 NVML 瞬时采样都不超过 250W。监控中出现过短时
过冲，持续行为应以平均值和百分位描述。Prefill 更长时间触及功率墙，因此平均
SM 频率更低。

仅供背景比较：此前 120W 下的固定长度结果为 baseline 128/256 30.33 tok/s、
MTP 128/256 66.15 tok/s、4 并发 1024/256 99.79 tok/s。120W 原始功耗遥测没有
保留，所以这些值不属于本文的受控 250W 数据集。

## 仓库内容

```text
scripts/                  启动、完整测试、benchmark、遥测和汇总代码
environment/              实测宿主机、软件和模型元数据
results/serving-250w.csv  可机读的 serving 指标
results/power-250w.csv    可机读的活跃负载功耗汇总
results/telemetry/        原始 200 ms NVML 采样
```

## 局限性

- 这里只测试了一张卡和一组软件/模型组合，不能当成通用 GPU 排名。
- 请求数量足够用于选择 profile，但不足以给出跨显卡、驱动、温度和多次进程启动
  的置信区间。
- 固定 token 的 random prompt 不能代表所有生产请求分布。
- 各 profile 运行时长和起始温度不同，最高温度不能直接横向评价散热。
- 本文 MTP 针对单并发优化；4 并发 profile 没有启用 MTP。
- 固件修改和功率上限解锁不属于本仓库范围。

## 许可证与致谢

代码使用 Apache License 2.0。SGLang 由
[SGLang project](https://github.com/sgl-project/sglang) 开发。模型及其许可证由
模型提供方单独分发。
