# GLM-5.2 三种 drafter 统一训练交接说明

更新时间：2026-08-26

## 目标与当前边界

本仓库在昇腾 910B 上为 GLM-5.2 BF16 target 构建统一训练数据，并离线训练
DFlash、DFlash2、DSpark 三种 drafter。三种方法必须使用同一批 sampled
trajectory、同一个 schema-v2 hidden cache、同一组 sample ID 和确定性 anchors，便于
做严格对比。

已实现：Stage A 轨迹生成、Stage B 单 forward hidden 抓取、冻结 target token I/O
提取、三种方法训练/FSDP2/checkpoint/export、本地数值与训练 smoke。

未宣称完成：真实 910B 数值 parity、正式多卡训练稳定性、Ascend serving ABI 和最终
投机解码评测。接手者不能把 CPU 测试通过写成真实硬件已经通过。

## 唯一生产数据链

```text
cbyzju/vibe_coding_630k
  -> Stage A: GLM-5.2 sampling agent rollout
  -> frozen input_ids + target-position loss_mask
  -> Stage B: same sampled path teacher forcing, one target forward
  -> aux hidden [T,5,6144] + final hidden [T,6144]
  -> one schema-v2 packed cache
  -> DFlash / DFlash2 / DSpark offline training
```

Stage A 默认 `temperature=1.0, top_p=0.95, top_k=-1`。Stage B 中用于构造 prefill
的 temperature=0 不会重生成答案；它严格沿 Stage A 冻结 token path 前向。

旧 vLLM response-only 路径已经删除，不要恢复。Stage A、Stage B、tokenizer、
embedding 和 LM head 必须来自同一不可变 GLM-5.2 BF16 revision。

## Stage A：rollout trajectory

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

## 固定模型契约

```text
target depth                 78
logical target layers       [1,20,38,56,75]
draft layers                5
hidden/intermediate         6144 / 12288
Q heads / KV heads          64 / 64
head_dim                    64
Q/K/V width                 4096
O projection                4096 -> 6144
attention                   full, no sliding window
RoPE theta                  8e6
RMS epsilon                 1e-5
block                       1 anchor + 15 mask positions
```

一个 block 内的 16 个 local query 互相全可见；context 只包含 anchor 之前的真实
prefix。位置 ID 是真实绝对位置，不允许重置成 0..15。

## Stage B：hidden 提取

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
auxiliary layers: model.model.layers_to_capture[physical_layer]
final hidden:     model.model.norm.forward_output
```

## Cache v2

```text
manifest.json
index.jsonl
segment-00000/
  input_ids.bin
  loss_mask.bin
  aux_hidden_states.bin
  target_final_hidden.bin
```

辅助层和 final hidden 在同一次 target forward 中抓取。final hidden 的语义必须是
`post_final_norm_lm_head_input`。v1 没有 final hidden，只允许显式 legacy 诊断读取，
三个统一训练入口全部拒绝 v1。

## 三种 loss

- DFlash：common backbone 的 full-vocab CE，深度权重 `exp(-d/7)`。
- DFlash2：DFlash base CE + dynamic-conv/top-16 selector CE；selector 只在真实 token
  位于 base top-16 时监督。
- DSpark：plain backbone + rank-256 Markov head + confidence head；用 final hidden
  重建 target 分布，loss 为 `0.1 CE + 0.9 full-vocab L1 + 1.0 BCE`。

DSpark 在 target token 位置 `p` 使用 final hidden `p-1`。首位置 predecessor 是 clean
anchor，后面是 teacher-forced previous target。target embedding/lm-head 完全冻结，
不进入 draft checkpoint。

## 关键入口

```text
Stage A: scripts/run_stage_a_trajectories.sh
Stage B: scripts/run_stage_b_hidden.sh
I/O:     scripts/extract_glm52_io.sh
Train:   scripts/train_glm52_drafter_910b.sh
CLI:     tools/train_drafter_offline.py
Gate:    tools/calibrate_hidden_capture_gate.py
         tools/validate_hidden_cache.py
```

训练示例：

```bash
METHOD=dflash2 \
CACHE_DIR=/shared/out/hidden-v2 \
TARGET_IO_DIR=/shared/out/glm52-target-io \
OUTPUT_DIR=/shared/out/glm52-dflash2 \
MASK_TOKEN_ID=<真实值> PAD_TOKEN_ID=<真实值> \
NUM_NPUS=8 \
bash scripts/train_glm52_drafter_910b.sh
```

旧 `train_glm52_dflash2_910b.sh` 只是兼容 wrapper。

## 接手后第一批必须做的事

1. 在真实 GLM-5.2 BF16 + 910B 环境对一个短序列做三次 direct capture；另外抓取
   shifted-layer 和 pre-final-norm 两个负对照。
2. 用 `calibrate_hidden_capture_gate.py` 固化误差阈值，再用
   `validate_hidden_cache.py` 验证 Stage B；严禁生产验证时现场放宽阈值。
3. 确认真实 MASK/PAD ID，并完成两 rank FSDP2 uninterrupted-vs-resume gate。
4. 在最终 Ascend serving fork 中逐方法验证 common logits；另验证 DFlash2 selector
   以及 DSpark Markov/confidence 输出。
5. 完成实际 speculative-decoding benchmark 后，才能宣称端到端可用。

更完整命令见 `docs/ASCEND_910B_RUNBOOK.md`。
