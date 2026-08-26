# GLM-5.2 BF16 DFlash2 交接说明

更新时间：2026-08-25

本文档面向接手本项目的服务器端 AI。开始工作前，请完整阅读本文和
[`README.md`](README.md)，不要根据文件名猜测训练语义。

## 1. 任务目标

本项目要在昇腾 910B 上，为 **GLM-5.2 BF16** target 训练一个 DFlash2
speculative-decoding drafter，主要面向 `cbyzju/vibe_coding_630k` 代码任务。

目标是：

1. 使用 GLM-5.2 的真实 sampling policy 生成 on-distribution agent 轨迹；
2. 沿固定 sampled token path 提取 GLM-5.2 中间 hidden states；
3. 训练阶段不重新加载完整 GLM-5.2 backbone，只读取离线 hidden cache 和冻结的
   target embedding/LM head；
4. 在 910B 上训练并导出 block size 16、五层的 DFlash2 drafter；
5. 最终通过 serving parity 和 speculative-decoding 评测验证模型。

当前仓库完成的是数据生成、hidden 提取、离线训练和训练侧验证。Ascend serving
kernel、最终 speculative-decoding runtime 和正式评测尚未在本仓库中完成。

## 2. 不要误解的关键语义

### 2.1 数据是 sampling，不是 greedy

Stage A 默认使用：

```text
temperature = 1.0
top_p       = 0.95
top_k       = -1   # 禁用额外 top-k 截断
```

对应调用链：

```text
scripts/run_stage_a_trajectories.sh
  -> tools/generate_trajectories.py
```

Stage A 生成 response 后，sampled response 和 token path 会被冻结。Stage B 沿该
固定路径 teacher forcing，不会重新 greedy decode。

`src/glm_dflash2/sglang_hidden_runner.py` 中的
`SamplingParams(temperature=0, max_new_tokens=1)` 只用于构造一次 SGLang prefill
请求；它不会生成训练答案，也不会把 sampling 数据变成 greedy 数据。

### 2.2 旧 vLLM response-only 路径已经删除

当前唯一生产路径是 SGLang 两阶段流程。不要重新创建或调用以下旧入口：

```text
tools/generate_responses.py
scripts/generate_glm52_910b.sh
src/glm_dflash2/backends.py
src/glm_dflash2/generate.py
```

旧链只保存 response schema，不能满足当前 Stage B 的完整 agent trajectory 和
hidden-cache 契约。

### 2.3 三个阶段必须使用同一模型 revision

以下内容必须来自同一个不可变 GLM-5.2 BF16 checkpoint：

- Stage A sampling endpoint；
- Stage A 本地 tokenizer/chat template；
- Stage B hidden extraction model；
- target `embed_tokens` 和 `lm_head`。

任意一项 revision 不同，cache 都不能视为有效训练数据。

## 3. 完整执行链

```text
cbyzju/vibe_coding_630k
        |
        v
Stage A: sampled coding-agent rollout
  - GLM-5.2 SGLang OpenAI endpoint
  - real workspace/tool execution
  - temperature=1.0, top_p=0.95
  - freeze messages/input_ids/loss_mask
        |
        v
frozen trajectory JSONL + manifest
        |
        v
Stage B: exact teacher-forced hidden extraction
  - fresh GLM-5.2 BF16 SGLang ModelRunner
  - no response regeneration
  - capture logical layers 1,20,38,56,75
        |
        v
frozen packed hidden cache
  - input_ids
  - loss_mask
  - hidden_states [T,5,6144] BF16
        |
        +---------------- target embed_tokens/lm_head extraction
        |                         |
        v                         v
Offline DFlash2 training (no target backbone)
        |
        v
draft-only model.safetensors + config.json
```

## 4. Stage A：sampling 轨迹生成

主要文件：

- `scripts/run_stage_a_trajectories.sh`
- `tools/generate_trajectories.py`
- `src/glm_dflash2/agent_trajectory.py`
- `src/glm_dflash2/vibe_coding.py`
- `src/glm_dflash2/workspaces.py`
- `src/glm_dflash2/open_swe_trajectories.py`

Stage A 不只是生成一段文本。它会：

1. 读取 vibe-coding 数据并保留消息、tool schema 和任务类型；
2. 对普通 repo/coding case 建立隔离 workspace，执行真实工具调用；
3. 对 Open-SWE prefix 路由恢复原始完整 trajectory；
4. 使用 GLM-5.2 sampling 进行多轮 agent rollout；
5. 用同一 tokenizer/chat template 冻结完整 `input_ids`；
6. 生成 target-token-position 语义的 `loss_mask`；
7. 逐条 fsync，并通过 manifest 固定 resume contract。

`loss_mask[i] = 1` 表示位置 `i` 是本次 rollout 中由 assistant 生成、允许作为训练
目标的位置。System、user、历史上下文和 tool observation 为 0。该 mask 不是普通
AR CE 的左移标签。

输出 manifest 只有在 shard 所有 ID 都成功提交且没有 unresolved error 时才是
`frozen`。`MAX_SAMPLES` smoke 生成的是 `partial`，不能冒充完整训练集。

推荐复用已经部署好的 GLM-5.2 SGLang endpoint：

```bash
cd /path/to/glm-dflash2

MODEL_PATH=/shared/models/GLM-5.2-bf16 \
ENDPOINT=http://glm52-sglang-service:30000 \
SERVED_MODEL_NAME=GLM-5.2 \
WORKSPACE_MAP=/shared/data/workspace_map.jsonl \
WORKSPACE_CACHE=/shared/cache/vibe-workspaces \
OPEN_SWE_STORE=/shared/data/open_swe_original.sqlite \
OUTPUT_JSONL=/shared/out/trajectories-shard-0-of-1.jsonl \
bash scripts/run_stage_a_trajectories.sh
```

并发分为两层，不能混为一谈：

- `WORKERS=8`：同时推进 trajectory、工具调用和 workspace 操作；
- `MAX_RUNNING_REQUESTS=2`：通过进程内共享 semaphore 限制本脚本同时发出的模型
  HTTP 请求；本地 SGLang 也收到同名限制；
- `MAX_TOTAL_TOKENS=131072`：本地 SGLang 的聚合 token-pool 上限。

第一次在新的 910B 环境上运行时先使用
`WORKERS=4 MAX_RUNNING_REQUESTS=1 MAX_SAMPLES=50`，确认长上下文峰值 HBM 后再切到
默认的 `8/2`。若 `ENDPOINT` 指向外部服务，本脚本无法修改外部服务自身的 token
pool，也无法限制其他客户端；外部 SGLang 启动参数必须单独对齐。

不要将 `TEMPERATURE=0` 作为默认值。若真实线上 GLM sampling policy 不是
`1.0/0.95/-1`，应显式传入线上参数，并重新生成完整 shard。

## 5. Stage B：hidden 提取

主要文件：

- `scripts/run_stage_b_hidden.sh`
- `tools/extract_hidden_sglang.py`
- `src/glm_dflash2/sglang_hidden_runner.py`
- `src/glm_dflash2/hidden_extraction.py`
- `src/glm_dflash2/hidden_cache.py`

Stage B 使用独立、全新的 SGLang internal `ModelRunner`，沿 Stage A 已冻结的
`input_ids` 做一次完整 teacher-forced prefill。

固定 logical layer IDs 为 `[1,20,38,56,75]`。在当前 GLM/SGLang capture 语义中，
对应 physical layer inputs `[2,21,39,57,76]`。兼容的 hook 名称为：

```text
set_eagle3_layers_to_capture
set_dflash_layers_to_capture
```

单条样本 tensor：

```text
input_ids              [T]          int64
loss_mask              [T]          bool/uint8
layer_hidden_states    [T,5,6144]   BF16
hidden_states          [T,30720]    BF16, flattened training view
```

BF16 两节点示例；两台机器运行相同命令，只修改 `NODE_RANK`：

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TRAJECTORY_JSONL=/shared/out/trajectories-shard-0-of-1.jsonl \
OUTPUT_DIR=/shared/out/hidden-shard-0-of-1 \
TP_SIZE=32 EP_SIZE=32 NNODES=2 NODE_RANK=0 \
DIST_INIT_ADDR=<rank-0-host> NCCL_PORT=29500 \
bash scripts/run_stage_b_hidden.sh
```

Stage B 为精确提取所有位置 hidden，当前禁用了 chunked prefill。超长单条轨迹可能
OOM，不能用静默截断规避。

## 6. Packed hidden cache 契约

```text
hidden-cache/
  manifest.json
  index.jsonl
  segment-00000/
    input_ids.bin
    loss_mask.bin
    hidden_states.bin
```

写入顺序为：数据 stream fsync，再写 index line 并 fsync。index 是 commit point，
manifest 可以从 index 恢复。每段数据都有 SHA-256。

训练热路径默认不重新计算每个 hidden slice 的 SHA-256，否则几十 TB cache 会产生
严重 I/O 开销。完整校验需显式运行：

```bash
python tools/validate_hidden_cache.py \
  --cache-dir /shared/out/hidden-shard-0-of-1 \
  --expected-layer-ids 1,20,38,56,75 \
  --full-scan
```

## 7. Target embedding/LM head

训练不会加载完整 GLM-5.2，只加载冻结的 target token I/O：

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TARGET_IO_DIR=/shared/out/glm52-target-io \
bash scripts/extract_glm52_io.sh
```

只接受 dense BF16/FP16/FP32 safetensors。当前不支持量化后的 ModelSlim/W8A8
embedding 或 LM head。

这两个 tensor 不注册进 draft module、不进入 optimizer、不写入 draft checkpoint。
训练结束还会检查它们没有被修改。

## 8. 当前训练结构与 loss

固定配置：

```text
block_size          = 16
anchors             = 64
draft layers        = 5
hidden_size         = 6144
intermediate_size   = 12288
attention heads     = 32
KV heads            = 8
target layers       = 5 x 6144 -> flatten 30720
dynamic conv        = kernel 2, group size 16
position gamma      = 7
selector top-k      = 16
selector rank       = 256
```

每个有效 anchor 构造长度 16 的 DFlash block。第一个位置是 anchor，后续位置为并行
预测 token。anchor 本身及至少一个 successor 必须位于 `loss_mask`，否则不参与训练。

训练目标：

1. **Base CE**：分块 full-vocabulary projection 计算精确 target-token NLL；
2. **Selector CE**：仅当真实 target 位于 base top-16 candidate 时，监督 selector
   选择对应 candidate slot。

```text
L = L_base_CE + L_selector_CE
```

两个 loss 使用 gamma=7 的 block-depth 权重。不同 rank 有效 token 数量不同时，loss
按全局有效权重归一化，避免 FSDP 简单平均 rank-local mean 造成偏差。

默认 optimizer：AdamW，`lr=6e-4`，`betas=(0.9,0.95)`，weight decay 0，warmup
1,000 optimizer steps，之后 cosine decay。

训练入口：

```bash
CACHE_DIR=/shared/out/hidden-frozen \
TARGET_IO_DIR=/shared/out/glm52-target-io \
OUTPUT_DIR=/shared/out/glm52-dflash2-b16 \
MASK_TOKEN_ID=<verified-real-mask-id> \
PAD_TOKEN_ID=<verified-pad-id> \
NUM_NPUS=8 \
bash scripts/train_glm52_dflash2_910b.sh
```

`MASK_TOKEN_ID` 必须从真实 tokenizer/正式 serving contract 中确认，不得用 EOS 或
PAD 猜测替代。

## 9. Checkpoint 与 resume

checkpoint 只允许在 optimizer-step boundary 保存，包含 draft model、optimizer、
scheduler、Python/NumPy/PyTorch/NPU RNG、anchor generator state，以及 epoch、cursor、
micro/global step 和训练语义配置。

payload、metadata 和最终 `COMPLETE` marker 使用原子 rename 与 fsync。resume 会拒绝
cache identity、target I/O、架构和 sampling/batch/scheduler 参数变化。

## 10. 当前验证状态

本地使用 Transformers 4.57.3，结果为：

```text
123 tests passed
CPU two-rank/FSDP2 checkpoint-resume tests passed
tiny offline training smoke passed
smoke loss: 4.295214 -> 3.987645
```

已静态或模拟验证：Stage A sampling 默认值、duplicate ID、token/loss-mask contract、
hidden hook 路由、cache recovery/checksum、block/attention/anchor shapes、chunked
LM-head、global weighted distributed loss、frozen target I/O、checkpoint exact resume
和 tiny forward/backward。

这些结果不等于真实 910B 硬件和正式 SGLang serving 已经通过。

## 11. 尚未解决或必须在服务器验证的问题

### P0：全量任务开始前必须完成

1. **真实 hidden 数值 gate**：用 1–2 条短轨迹验证 `[T,5,6144]` finite BF16，并将
   五层分别与同 checkpoint 直接 forward hook 对比；不能只检查 shape。
2. **确认真实 MASK token ID**：当前代码故意不自动猜测，必须与最终 runtime 一致。
3. **sampled response 原始 token IDs**：标准 SGLang OpenAI endpoint 可能只返回文本；
   当前使用同一 tokenizer canonical replay。必须抽样检查服务器原始 IDs 与 replay
   IDs，最稳妥是让 vendor endpoint 返回 response token IDs。
4. **模型 revision 一致性**：外部 endpoint 无法自动暴露权重 hash；Stage A manifest
   会记录 `weight_identity_verified=false`，部署方必须固定同一 BF16 revision。
5. **910B 两卡训练/resume gate**：

```bash
CACHE_DIR=/shared/gate/hidden-frozen \
TARGET_IO_DIR=/shared/out/glm52-target-io \
MASK_TOKEN_ID=<verified-id> \
OUTPUT_DIR=/shared/gate/train-2rank \
bash scripts/gate_train_2rank_910b.sh
```

必须得到 `gate-result.json` 中的 `"passed": true`。

6. **训练与 serving ABI parity**：对比 backbone logits、top-16 IDs/scores、selector
   pair scores 和最终 selected path，使用 `tools/compare_sglang_runtime.py`。

### P1：已知代码或调度问题

1. Stage A 输出锁目前为手动 `__enter__/__exit__`。正常退出和进程崩溃会释放，但
   同一 Python 进程内部异常后的释放应改成标准 `with`/`finally`。
2. scheduler 的 `total_steps` 按 DataLoader 上界估算；若整批无有效 anchor 而跳过，
   实际 optimizer steps 会略少，cosine 尾部可能不能精确走完。
3. 本仓库没有最终 Ascend DFlash2 serving 实现，只有训练侧 export 和 parity 工具。
4. 当前没有完整 speculative-decoding benchmark/evaluation launcher。

### P2：资源风险

五层 6144-wide BF16 hidden 每 token 为 `5 * 6144 * 2 = 61,440 bytes`。630K 样本
平均 1,000 token 时约 38.7 TB；平均 2,000 token 时约 77.4 TB，尚未包含索引和
文件系统开销。开始全量提取前必须确认容量、inode、吞吐、备份和清理策略。

Stage B 当前采用完整 prefill，超长 trajectory 还有 activation-memory 风险。

## 12. 建议执行顺序

不要直接启动 630K 全量任务。严格按以下顺序：

1. 检查 GLM-5.2 BF16 checkpoint、tokenizer、MASK/PAD ID；
2. 生成 1–2 条 Stage A sampling trajectory；
3. 检查 sampling manifest 和 sampled token replay；
4. 运行 Stage B 两样本 hidden extraction；
5. 与直接 GLM hook 做五层数值对齐；
6. 提取 dense BF16 target embedding/LM head；
7. 运行 `gate_train_2rank_910b.sh`；
8. 做 trainer/runtime ABI parity；
9. 估算真实 token 总量和 cache 存储；
10. 开始分 shard 的全量 Stage A；
11. 全量 Stage A frozen 后运行 Stage B；
12. 所有 cache shard 校验通过后开始正式训练。

## 13. 接手 AI 的操作边界

- 不要将 Stage A 改回 greedy。
- 不要在 Stage B 重新生成 response。
- 不要在 offline trainer 中加载完整 GLM-5.2 backbone。
- 不要将 W8A8/ModelSlim target I/O 混入 BF16 cache/训练。
- 不要把 EOS/PAD 当作 mask token。
- 不要跳过两样本 hidden 数值 gate。
- 不要因 OOM 静默截断 trajectory。
- 不要修改已经 frozen 的 trajectory/cache；新实验使用新目录。
- 修改代码后至少运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PY=/path/to/python bash scripts/smoke_train_no_npu.sh
```

## 14. 入口索引

| 任务 | 文件 |
|---|---|
| Stage A launcher | `scripts/run_stage_a_trajectories.sh` |
| Stage A implementation | `tools/generate_trajectories.py` |
| Agent rollout | `src/glm_dflash2/agent_trajectory.py` |
| Stage B launcher | `scripts/run_stage_b_hidden.sh` |
| Hidden extraction | `tools/extract_hidden_sglang.py` |
| SGLang hidden bridge | `src/glm_dflash2/sglang_hidden_runner.py` |
| Packed cache | `src/glm_dflash2/hidden_cache.py` |
| Target I/O extraction | `scripts/extract_glm52_io.sh` |
| Training launcher | `scripts/train_glm52_dflash2_910b.sh` |
| Training main | `tools/train_dflash2_offline.py` |
| Block construction | `src/glm_dflash2/dflash2_blocks.py` |
| Draft architecture | `src/glm_dflash2/dflash2_model.py` |
| Loss | `src/glm_dflash2/dflash2_objective.py` |
| Trainer wrapper | `src/glm_dflash2/offline_trainer.py` |
| Checkpoint | `src/glm_dflash2/checkpointing.py` |
| 910B gate | `scripts/gate_train_2rank_910b.sh` |
| Runtime parity | `tools/compare_sglang_runtime.py` |

如果服务器端 SGLang/torch-npu/CANN API 与本文不一致，应先记录精确版本、最小报错
和源码位置，再实现小范围 adapter。不要在未理解 tensor shape 和语义时重写主链。
