# CMP 170HX 10G 改 40G / 8G 改 64G Benchmark 完整管线与发布协议

[English](CMP-170HX-Benchmark-Protocol.md) | [简体中文](CMP-170HX-Benchmark-Protocol.zh-CN.md)

本文供复现者、代码审阅者和 AI agent 使用。它定义什么是一次完整的 CMP 170HX +
SGLang + Qwen3.6 benchmark run，哪些 artifact 必须保留，以及什么时候两次 run 可以
横向比较。

## 当前自动化状态

仓库已经有可运行的核心管线，但还不是完全自动化的“运行到发布”流水线。

| 阶段 | 当前状态 | 入口 |
| --- | --- | --- |
| GPU/功率/swap/占用进程预检 | 已自动化 | `scripts/run-250w-suite.sh` |
| 服务启动和 profile 容量验证 | 已自动化 | `scripts/serve-qwen36-27b.sh`、`/server_info` |
| 固定长度和真实数据集请求 | 已自动化 | `scripts/run-250w-suite.sh`、`scripts/bench-serving.sh` |
| 200 ms NVML 遥测 | 已自动化 | `scripts/monitor-gpu.sh` |
| case manifest 和原始 JSONL | 已自动化 | `scripts/run-250w-suite.sh` |
| 环境采集 | 部分自动化 | `scripts/collect-environment.sh` |
| serving/power 汇总 | 已有工具，但需单独执行 | `scripts/summarize-expanded.py` |
| checkpoint revision、完整软件 diff 和数据集 hash 门禁 | 部分缺失 | 必须人工补录并检查 |
| 结果完整性验收、跨 run 比较、发布包生成 | 尚未一键自动化 | 按本文检查 |
| 多个独立 run、随机/交错 profile 顺序 | 尚未自动化 | 严格硬件结论前必须补做 |

因此答案是：**需要完整管线协议**。现有脚本足以复现 canonical v2，但其他测试者若
只运行脚本、不锁定输入和代码 revision，生成的数字不能自动与本仓库严格横比。

## 两个复现等级

### 等级 A：协议复现

目标是使用相同 case 矩阵、指标定义和主要启动参数，判断另一个环境能否复现趋势。
允许硬件、驱动或明确记录的软件 patch 不同。发布时必须称为“外部复现”或“跨系统
对照”，不能把差异归因给某一个变量。

### 等级 B：受控 A/B

目标是证明某一个变量的因果效果，例如 10G 改 40G 与 8G 改 64G、MTP patch 开启与关闭、
或 120 W 与 250 W。除目标变量外，下列项目必须相同：

- 同一张卡，或已证明等价的硬件 revision、VBIOS、HBM 总线/频率和时钟策略；
- 相同 SGLang commit 和 clean/dirty 状态；如果有 patch，保存完整 diff；
- 相同 checkpoint repository 和不可变 revision；
- 相同 `config.json`、`generation_config.json`、`recipe.yaml` 和 MTP 文件；
- 相同驱动、CUDA、Python、PyTorch、Triton 和 FlashInfer；
- 相同 prompt 内容、顺序、seed、输出长度、并发和采样参数；
- 相同功率限制、swap 状态、后台 GPU 进程和 cache 策略；
- 至少 3 个独立 RUN_ID，并交错或 counterbalance profile 执行顺序。

当前 10G 改 40G 与外部 8G 改 64G 报告属于等级 A，不属于等级 B。

## 固定测试契约

### Checkpoint

```text
Repository: https://huggingface.co/Avesed/Qwen3.6-27B-INT8-W8A8
Base model: https://huggingface.co/Qwen/Qwen3.6-27B
```

仅写 repository 名称不够。新 run 应记录 Hugging Face commit revision，或对所有
关键文件记录 SHA-256。至少包括：

```text
config.json
generation_config.json
recipe.yaml
model.safetensors
mtp.safetensors
tokenizer/config/template files
```

大权重文件计算 SHA-256 较慢时，可以使用 Hugging Face immutable revision 加上
safetensors 文件的仓库 LFS oid；不能只记录本地目录名。

### SGLang 和 patch

必须记录：

```bash
git -C "$SGLANG_SOURCE" rev-parse HEAD
git -C "$SGLANG_SOURCE" status --short
git -C "$SGLANG_SOURCE" diff --binary
```

如果工作树 dirty，必须把 diff 保存到 run artifact。只写 `SGLang 0.5.16` 无法区分
官方 tag、170HX patch、`topk1` 修复或本地未提交修改。

### 数据集

Canonical v2 使用：

```text
ShareGPT SHA-256:
35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4

LongBench-v2 prepared JSONL SHA-256:
93ecd8a799ba868cfb2a1c28a38ce87653cb30eb211712030970020d39990058
```

数据集 hash 不同的 run 不能称为逐请求严格对照。即使 prompt/output token 总数相同，
也只能标记为“分布匹配”，除非进一步比较规范化后的每行内容 hash。

### Profile

发布表格必须把并发和 MTP 状态写完整，不使用孤立的 `baseline`、`mtp`、`c1` 或
`c4` 作为面向读者的唯一标签。

| 最大并发 | MTP 状态 | 上下文长度 | 总 token 池 | 静态显存比例 | CUDA Graph batch size | 推测解码 |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | MTP 关闭 | 24576 | 24576 | 0.88 | 1 | 不适用 |
| 1 | MTP 开启 | 24576 | 24576 | 0.94 | 1 | EAGLE，3 steps，top-k 1，4 draft tokens |
| 4 | MTP 关闭 | 24576 | 20480 | 0.88 | 1、2、4 | 不适用 |
| 4 | MTP 开启 | 24576 | 20480 | 0.98 | 1、2、3、4 | EAGLE，3 steps，top-k 1，4 draft tokens |

四个 profile 使用 page size 64、2048-token chunked prefill、4096-token prefill
上限、关闭 prefill CUDA Graph 和关闭 radix cache。任何偏离都必须放进 run metadata，
不能静默修改。

### Case 矩阵

单请求长输出每个 cell 完成 3 个请求：

```text
Prompt: 1024, 4096, 8192, 20480
Requested generation: 1024, 4096, 8192
Constraint: prompt + generation < 24576
```

Prefill 每个 cell 完成 3 个请求，每个请求生成 1 token：

```text
Prompt: 512, 2048, 4096, 8192, 12288, 20480
```

最大并发 4 每个 cell 完成 20 个请求：

```text
Prompt: 2048, 4096
Requested generation: 512, 1024, 2048
Constraint: 4 * (prompt + generation) <= 20480
```

真实数据集：

```text
ShareGPT: 30 requests at maximum concurrency 1, natural output
ShareGPT: 20 requests at maximum concurrency 4, natural output
LongBench-v2: 20 requests at maximum concurrency 1, natural 10-token output
LongBench-v2: same 20 requests, fixed generation 512
```

完整 run 应得到 56 条 manifest row：50 passed 和 6 skipped-capacity。容量跳过不是
失败，也不能记为零吞吐。

## 完整执行流程

### 1. 选择不可变 RUN_ID

```bash
export RUN_ID="cmp170hx-$(date +%Y%m%d-%H%M%S)"
export SGLANG_RUNTIME_HOME=/path/to/sglang-runtime
export SGLANG_VENV=/path/to/sglang-venv
export SGLANG_SOURCE=/path/to/sglang-source
export MODEL_PATH=/path/to/Qwen3.6-27B-INT8-W8A8
export SHAREGPT_PATH=/path/to/sharegpt.json
export LONGBENCH_PATH=/path/to/longbench-v2-prepared.jsonl
```

RUN_ID 一旦产生数据就不能复用于不同配置。重跑必须使用新 RUN_ID；恢复同一个 run
只允许复用已完成且配置相同的 case。

### 2. 预检

```bash
nvidia-smi -q -d POWER,CLOCK,MEMORY,PCI
swapon --show
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
sha256sum "$SHAREGPT_PATH" "$LONGBENCH_PATH"
git -C "$SGLANG_SOURCE" status --short
```

硬门禁：

- power limit 必须等于声明值；
- swap 必须为 0；
- 不能有其他 GPU compute process；
- 8000 端口不能已有服务；
- checkpoint、数据集和 SGLang revision 必须记录；
- 可用主机内存必须足够，不能依赖 swap 或 OOM retry。

### 3. 执行全部 case

当前 canonical 250 W 管线：

```bash
sudo -E env \
  RUN_ID="$RUN_ID" \
  RUN_REAL_DATASETS=1 \
  SGLANG_RUNTIME_HOME="$SGLANG_RUNTIME_HOME" \
  SGLANG_VENV="$SGLANG_VENV" \
  SGLANG_SOURCE="$SGLANG_SOURCE" \
  MODEL_PATH="$MODEL_PATH" \
  SHAREGPT_PATH="$SHAREGPT_PATH" \
  LONGBENCH_PATH="$LONGBENCH_PATH" \
  scripts/run-250w-suite.sh
```

脚本会逐个 profile 启动服务、查询 `/server_info`、运行 case、停止服务并等待显存释放。
每个 case 执行 `--flush-cache`；benchmark warmup 为 0，服务启动 warmup 保持开启。

### 4. 监控

`scripts/monitor-gpu.sh` 默认每 200 ms 采集：

```text
timestamp
power.draw
power.limit
utilization.gpu
clocks.sm
clocks.mem
temperature.gpu
pstate
```

manifest 的 `start_epoch_ms`、`end_epoch_ms` 和 `telemetry_file` 是 case 到遥测的唯一
关联。不能按文件名大致截取时间窗，也不能混用另一个 profile 的 telemetry。

### 5. 汇总

```bash
python3 scripts/summarize-expanded.py \
  "results/case-manifest-$RUN_ID.csv" \
  --repo-root "$PWD" \
  --serving-output "results/serving-$RUN_ID.csv" \
  --power-output "results/power-$RUN_ID.csv"
```

三类指标必须保持分离：

- `e2e_output_tok_s`：完成的输出 tokens / benchmark 墙钟时间；
- `input_tok_s`：完成的输入 tokens / 同一墙钟时间；
- `1000 / mean TPOT`：只用于最大并发 1 的稳态 decode 近似。

最大并发 4 时不能用 `1000 / TPOT` 代替聚合输出吞吐。

### 6. 验收

发布前逐项检查：

- manifest 正好 56 条数据 row，case key 无重复；
- 50 条 `passed`、6 条 `skipped-capacity`，没有 `failed` 或 `incomplete`；
- 每个 passed row 的 JSONL 最后一条汇总记录 `completed == requests`；
- 固定长度 case 的输入/输出 token 数符合声明；
- MTP 关闭 row 不应出现 MTP 接受长度；
- MTP 接受长度必须在合理范围并从原始字段重算；恰好等于 draft 上限的所有 row
  必须人工复核，避免字段语义或汇总 bug；
- `/server_info` 中 token pool 和有效最大运行请求数与 profile 相符；
- 每个 case 时间窗被其 telemetry 覆盖，采样间隔和缺口可接受；
- 环境文件包含 checkpoint revision、SGLang commit/diff 状态和数据集 hash；
- README 数字可从发布 CSV 重算，手工转录值应有来源说明。

### 7. 发布 artifact

一个可审计 run 至少发布：

```text
environment/<run-id>.txt
results/case-manifest-<run-id>.csv
results/serving-<run-id>.csv
results/power-<run-id>.csv
results/raw/<run-id>/*.jsonl
results/logs/<run-id>/*.log
results/telemetry/<run-id>/*.csv
patches/<run-id>-sglang.diff       # 仅在 source dirty 或有额外 patch 时
```

如果 raw artifact 太大而不放 Git，必须发布可下载归档、SHA-256 和生成命令。只有 PDF
或 Markdown 摘要的外部结果可以引用，但必须标记为“不可从本仓库重算”。

## 跨 run 比较规则

比较脚本或 AI 必须先按以下字段连接，而不是按表格行号：

```text
workload
prompt/input length
requested output length
maximum concurrency
MTP state
sampling parameters
dataset SHA
```

然后检查环境差异。任何不相同字段都应列为 confounder。不得：

- 把 `MTP 开启 / MTP 关闭` 加速比和 `机器 B / 机器 A` 吞吐比混为一谈；
- 把 input tok/s、output tok/s 和 steady decode tok/s 放在同一列比较；
- 把 LongBench 10-token 自然输出当成 decode benchmark；
- 把容量跳过写成失败或 0 tok/s；
- 因为 checkpoint 名称相同就假定文件相同；
- 因为 SGLang version 相同就忽略 commit 和 patch；
- 将更多显存容量直接解释为更高吞吐。

## 面向 AI agent 的操作约束

1. 先读本文件、主 README、环境文件、manifest 和 summarizer，再解释数据。
2. 未经用户明确要求，不启动长 benchmark，也不覆盖已有 RUN_ID。
3. 不修改或删除用户已有 raw/log/telemetry；新 run 使用新目录。
4. 摘要外部报告时同时保留作者、日期、RUN_ID、源文件 hash 和“是否有 raw data”。
5. 所有计算值必须注明公式；所有转录值必须注明来源页/表或源 CSV。
6. 发现指标歧义时使用完整标签，例如“最大并发 4、MTP 开启、聚合输出吞吐”。
7. 提交前运行 Markdown 表格列数检查、`git diff --check`、`shellcheck scripts/*.sh`
   和 `python3 -m py_compile scripts/*.py`。

10G 改 40G 与外部 8G 改 64G 的结构化对照见
[`README.zh-CN.md`](README.zh-CN.md)。
