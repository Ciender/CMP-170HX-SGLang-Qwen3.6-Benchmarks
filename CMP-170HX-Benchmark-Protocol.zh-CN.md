# SGLang 模型推理测试与发布规范

[English](CMP-170HX-Benchmark-Protocol.md) | [简体中文](CMP-170HX-Benchmark-Protocol.zh-CN.md)

本文定义本仓库的标准测试合同。它既适用于 CMP 170HX，也适用于其他 NVIDIA GPU；
既可以测试 Qwen3.6-27B，也可以测试用户提供的其他 SGLang checkpoint。固定矩阵用于
横向比较，扩展矩阵用于发挥 64/80 GiB 或更大显存机器的长上下文能力。

当前标准管线 v1 限定单 GPU、`TP_SIZE=1`。多卡 tensor parallel 需要为遥测和容量
schema 增加 rank/GPU index 后才能作为标准结果发布，不能把多卡采样混进单卡表格。

## 设计原则

1. **先启动、再规划**：显存容量不能直接换算成 token 上限。权重大小、量化格式、
   KV dtype、模型层结构、MTP draft、GDN/Mamba state 和 CUDA Graph 都会占用显存。
   测试矩阵必须使用 SGLang 启动后 `/server_info` 返回的实际 context、token pool 和
   effective maximum running requests。
2. **可比矩阵与能力矩阵分开**：`core` tier 尽量由所有机器运行；`extended` tier
   包含 32K-131K 输入，只有实际容量允许时才运行。
3. **跳过不是零分**：不满足上下文或总 token pool 的 case 分别记录为
   `skipped-context` 或 `skipped-capacity`，不能记为失败、0 tok/s 或删除。
4. **MTP 开关写完整**：表格使用 `MTP 开启`、`MTP 关闭`或 `MTP 不可用`；MTP
   加速比始终是同一环境同一 cell 的 `MTP 开启 / MTP 关闭`。
5. **所有结论来自结构化产物**：Markdown 和 PDF 从相同的 metadata、manifest 和
   CSV 自动生成，不手工复制数值。

## 标准管线

```text
benchmark.env
    │
    ├─ collect-run-metadata.py     主机、GPU、软件、模型、权重、量化、数据集
    ├─ serve-model.sh              通用 SGLang 启动器
    ├─ capture-profile.py          实际 context/token pool/并发、显存、KV/state/graph
    ├─ plan-benchmark.py           core + extended 容量计算与跳过原因
    ├─ bench-serving.sh            固定长度请求
    ├─ monitor-gpu.sh              NVML 遥测
    ├─ summarize-expanded.py       serving.csv + power.csv
    └─ generate-report.py          report.md + report.pdf
```

一键入口：

```bash
cp configs/benchmark.example.env /tmp/my-benchmark.env
# 编辑路径、模型、context、MTP 和功率设置
sudo -E scripts/run-standard-suite.sh /tmp/my-benchmark.env
```

当前 `scripts/test-pipeline.sh` 已覆盖 32/40/64/80 GiB 类实际容量的离线规划、
标准表格和 PDF 生成；通用启动器和完整运行器在完成首个真实模型端到端 run 前应视为
候选管线。已有 10G canonical 数据由旧管线产生，不受此状态影响。

每次运行必须使用新的 `RUN_ID`。已有 bundle 默认拒绝覆盖；只有确认配置完全相同时
才允许用 `RESUME=1` 恢复未完成项。

## 模型与启动配置

必填项：

```text
MODEL_PATH
MODEL_NAME
RUN_ID
SGLANG_VENV
CONTEXT_LENGTH（可设为 `auto`）
```

强烈建议填写：

```text
MODEL_REPOSITORY
MODEL_REVISION
SGLANG_SOURCE
EXPECTED_POWER_LIMIT
```

自定义模型不应继承 Qwen3.6 专用参数。量化、attention/sampling backend、reasoning
parser、tool parser 和 `trust_remote_code` 都是显式可选项；只有模型确实需要时才设置。
没有集成 MTP/EAGLE head 的模型应使用：

```bash
MTP_MODES="disabled"
```

支持 MTP 且需要比较开关时使用：

```bash
MTP_MODES="disabled enabled"
MTP_ALGORITHM=EAGLE
MTP_STEPS=3
MTP_TOPK=1
MTP_DRAFT_TOKENS=4
```

如果 MTP profile 启动失败，记录为 profile unavailable；不能悄悄改成另一个 draft
模型、降低并发或减小 token pool 后继续声称参数相同。

## 容量规则

每个 profile 启动后记录：

```text
context_length
max_total_tokens
effective_max_running_requests
GPU used/free memory after startup
target and draft load-phase deltas
target/draft KV cache size and dtype
GDN/Mamba/recurrent state size（若日志提供）
CUDA Graph size
pool-end free memory
```

固定长度 case 必须同时满足：

```text
每请求 input_tokens + output_tokens < context_length
max_concurrency × (input_tokens + output_tokens) <= max_total_tokens
max_concurrency <= effective_max_running_requests
```

第一条使用严格小于，因为生成请求还需要至少一个调度/token slot。第二条是共享 token
pool 的上界。对于真实数据集，必须先 tokenizer 预处理每条请求，再按最长或逐请求
实际 token 数做相同检查，不能只用字符数估计。

### 不同显存容量如何测试

- **32/40 GiB**：运行所有能通过容量计算的 `core` case；不能运行的长输出保留 skip
  行。不要为了填满表格而降低实际输入/输出长度。
- **64/80 GiB**：先完成同样的 `core` case，再运行能通过容量计算的 `extended`
  case。更长上下文是额外能力数据，不替代 core 对照。
- **模型不同**：即使 GPU 显存相同，也必须重新启动并重新读取 token pool。不能复制
  另一模型或另一量化版本的容量。

标准固定矩阵：

| Tier | 工作负载 | 输入 tokens | 输出 tokens | 最大并发 | 重复 |
| --- | --- | --- | --- | ---: | ---: |
| core | Prefill | 512、2048、4096、8192、12288、20480 | 1 | 1 | 每点 3 请求 |
| core | 单请求生成 | 1024、4096、8192、20480 | 1024、4096、8192 | 1 | 每点 3 请求 |
| core | 并发生成 | 2048、4096 | 512、1024、2048 | 配置值，默认 4 | 5 个完整并发波次 |
| extended | Prefill | 32768、49152、65536、98304、131072 | 1 | 1 | 每点 3 请求 |
| extended | 单请求生成 | 32768、49152、65536、98304、131072 | 1024、4096、8192 | 1 | 每点 3 请求 |
| extended | 并发生成 | 8192、16384、32768 | 512、1024、2048 | 配置值 | 5 个完整并发波次 |

用户可用 JSON `MATRIX_FILE` 增加模型特定矩阵，但不能删除 core 行后仍称为完整 core
复现。模型原生 context 小于某些 core 点时，这些点自然成为 `skipped-context`。

## 必须采集的元数据

### 测试身份

- RUN_ID、开始时间、时区、测试者和数据作者；
- benchmark 仓库 commit；
- SGLang commit、remote、dirty 状态和额外 patch；
- 所有人工修改过的启动参数。

### GPU 与主机

- GPU 名称、UUID、PCI ID、显存、compute capability、VBIOS；
- 驱动、功率限制、最高 SM/显存频率；
- CPU、逻辑核、主机内存、swap、内核和操作系统；
- tensor parallel、可见 GPU 和测试期间其他 GPU 计算进程。

### 模型

- 本地模型名称、Hugging Face repository 和 immutable revision；
- architecture、model type、dtype、量化 method/format；
- 所有 safetensors 文件名、单文件大小和总权重大小；
- `config.json`、tokenizer、template、recipe 的 SHA-256；
- 模型原生 context、层数、full-attention 层数、KV heads、head dim、KV dtype；
- 能够从配置可靠计算时记录理论 target KV bytes/token；否则留空，不猜测；
- MTP 权重、层数、draft 参数和是否共享 embedding。

### 运行时显存

磁盘权重大小和 GPU 常驻显存是两个不同指标，必须分开。运行时表至少包括 target/draft
加载阶段增量、KV cache、GDN/Mamba state、CUDA Graph、启动后 GPU used/free 和
token pool。日志无法拆分的 allocator/context/workspace 不能强行凑成显存总和。

## 指标与表格

必须分开使用以下指标：

- `input tok/s`：完成的 prompt tokens / 完整 benchmark 请求窗口；
- `output tok/s`：完成的 generated tokens / 同一窗口；最大并发大于 1 时是服务器聚合吞吐；
- `TTFT`：首 token 延迟，越低越好；
- `TPOT`：首 token 后的每输出 token 延迟，越低越好；
- `1000 / mean TPOT`：只可称为单请求持续 decode 近似；并发时不能代替聚合吞吐；
- MTP acceptance：只出现在 MTP 开启数据，并应与 draft 上限一起解释。

标准报告顺序：

1. 运行摘要和模型/GPU 环境；
2. profile 实际 context、token pool、并发和显存/KV/state；
3. passed/skipped/failed 计数；
4. MTP 开启/关闭同 cell 加速比；
5. 完整 core 表；
6. extended 长上下文表；
7. 真实数据集；
8. 遥测与限制。

不得只写“关闭 / 1024 / c1”。应写成“最大并发 1，MTP 关闭，端到端输出吞吐
(output tok/s)”。容量 skip 的吞吐字段保持空白。

## 运行门禁与监控

开始前必须确认：

- swap 为 0；
- 没有其他 GPU compute process；
- 目标端口没有现存服务；
- 功率限制与配置一致；
- 模型、SGLang 和数据集身份已记录；
- 新 RUN_ID 不覆盖已有产物。

每个 profile 使用独立服务日志与独立 NVML 遥测。默认约 200 ms 采样时间、功耗、
功率限制、GPU 利用率、SM/显存频率、温度和 P-state。case manifest 中的毫秒时间窗
是遥测切片的唯一依据。

## 标准发布包

```text
runs/<run-id>/
├── metadata.json
├── profiles/*.json
├── plans/*.csv
├── case-manifest.csv
├── serving.csv
├── power.csv
├── report.md
└── report.pdf

results/raw/<run-id>/*.jsonl
results/logs/<run-id>/*.log
results/telemetry/<run-id>/*.csv
```

`report.md` 和 `report.pdf` 由 `generate-report.py` 从上述结构化数据生成。PDF 是便于
阅读的发布件；CSV、JSONL、profile JSON 和 metadata 才是可重算的数据源。大文件不
上传 Git 时，应发布归档下载地址与 SHA-256。

## 验收标准

一次 run 可以包含不同数量的 passed/skip 行，不能再用固定“56 行、50 passed”验收。
发布前必须满足：

- 每个 plan case 在 manifest 中有唯一终态；
- 终态只能是 `passed`、明确的 skip、`failed` 或 `incomplete`；
- 每个 passed JSONL 的 `completed == requests`；
- skip 原因包含实际 required tokens 与实际 context/token pool；
- profile JSON 的实际并发不少于该 profile 声明并发；
- metadata 有模型、GPU、软件和 SGLang 身份；
- 每个 passed case 的时间窗有对应遥测；
- Markdown 表和 PDF 来自同一 bundle；
- `git diff --check`、Bash 语法、Python 编译和容量规划 fixture 全部通过。

存在 `failed` 或 `incomplete` 时仍可发布为部分结果，但标题和摘要必须明确写
“partial run”，不能把缺失 cell 当作容量跳过。

## 跨机器比较

只连接完全匹配的：模型 repository/revision、profile 参数、workload、input、output、
最大并发、MTP 状态、sampling 参数和数据集内容。然后列出 GPU、SGLang、驱动、功率、
时钟、KV dtype 和 backend 差异。

更多显存通常意味着能运行更长上下文或更大并发，不自动意味着相同 case 更快。
MTP 加速比也不能与机器 B / 机器 A 的吞吐比混为一谈。
