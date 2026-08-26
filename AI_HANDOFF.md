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
