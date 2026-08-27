# GLM-5.2 DFlash / DFlash2 / DSpark 服务器接管手册

更新时间：2026-08-26
仓库基线：`fix/stage-a-concurrency`，`d7b23bb fix: close production validation gaps`

> 这是服务器端 AI 的唯一接管入口。先完整读完本文，再看 `README.md` 和
> `docs/ASCEND_910B_RUNBOOK.md` 中的命令细节。本文记录的是“当前真实状态”，不把
> CPU smoke、离线 export 或已有 GLM 服务误写成 910B 端到端已经打通。

## 1. 最终目标

在昇腾 910B 上，以同一个 **GLM-5.2 BF16** target 和同一批 sampled vibe-coding
trajectory，完成一条可复现的统一实验链：

1. Stage A 生成真实 sampling agent trajectory，并保存服务端真实 sampled token IDs；
2. Stage B 沿冻结 token path 做一次 teacher-forced target forward，同时提取训练需要的
   五层 auxiliary hidden 和 final-norm hidden；
3. 提取并冻结 target 的 `embed_tokens` 与 `lm_head`；
4. 使用完全相同的 sample IDs、hidden cache 和 anchor 规则离线训练 DFlash、DFlash2、
   DSpark；
5. 导出 serving artifact，接入实际 Ascend 推理框架；
6. 串行测 target-only 与 speculative 的 acceptance length、TPS、speedup 和输出一致性。

预定的五组训练 setting：

| 方法 | physical block | 实际 proposal 数 | 训练轮数 | LR | gamma |
|---|---:|---:|---:|---:|---:|
| DFlash | 8 | 7 | 3 | `6e-4` | 4 |
| DFlash | 16 | 15 | 3 | `6e-4` | 7 |
| DFlash2 | 8 | 7 | 3 | `6e-4` | 4 |
| DFlash2 | 16 | 15 | 3 | `6e-4` | 7 |
| DSpark | 8 | 7 | 3 | `6e-4` | 4 |

physical block 的第一个位置是已知 anchor，不能把 B8/B16 错写成 8/16 个 proposal。

## 2. 当前结论：什么已完成，什么没有

### 2.1 本地已经实现

- Stage A：SGLang sampling、多轮工具调用、workspace 隔离、resume manifest、精确 token
  ID 冻结及 fail-closed 校验；
- Stage B：读取冻结 trajectory，使用一个 target forward 同时抓取五层 auxiliary hidden
  和 post-final-norm hidden，并写 schema-v2 packed cache；
- target token I/O：提取 dense BF16/FP16/FP32 embedding 与 LM head，并记录 identity；
- 统一数据与 anchor 构造：三个方法共享 cache row、sample ID、absolute position 和
  deterministic anchor；
- 三种离线训练 objective、FSDP2、梯度累积、完整 checkpoint/resume、标准 export；
- cache checksum、provenance、hidden parity gate、两 rank resume gate；
- target-only/speculative 串行 benchmark 驱动和 acceptance/TPS 统计；
- 旧的 vLLM response-only 数据生成入口已删除，生产路径只有 SGLang 两阶段方案。

### 2.2 本地能够证明的范围

- Python 单元测试、shell 静态检查、compileall、CPU tiny training smoke；
- schema-v2 cache 的结构、checksum、resume 和方法间数据一致性；
- 三个 objective 在 tiny 数据上 loss/gradient 有限，checkpoint 可 round-trip；
- export 自身的键、shape、checksum 和离线 round-trip。

这些检查不能证明真实 910B kernel、HCCL、SGLang hidden hook 或 serving ABI 正确。

2026-08-26 的最新验收记录：

```text
unit tests                     202 passed
all scripts bash -n            passed
src/tools/tests compileall     passed
git diff --check               passed
schema-v2 mock validation      passed (1 sample, 4 tokens, 5 layers)
tiny optimizer smoke           DFlash loss 2.586278
                               DFlash2 loss 3.063839
                               DSpark loss 1.230689
```

复现命令：

```bash
PYTHONPATH=src /tmp/glm-dflash2-test-py312/bin/python \
  -m unittest discover -s tests -p 'test_*.py'
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
PYTHONPATH=src /tmp/glm-dflash2-test-py312/bin/python -m compileall -q src tools tests
PY=/tmp/glm-dflash2-test-py312/bin/python bash scripts/smoke_no_model.sh
```

`/tmp/glm-dflash2-test-py312` 是当前机器的测试环境，不是可迁移依赖。服务器端应使用
厂商 torch-npu 环境重新执行相同 gate。

### 2.3 仍未完成，禁止提前宣称

- 真实 GLM-5.2 BF16 服务的 sampled token ID capability gate；
- 真实 910B 上 Stage B hidden/logits 对 direct Transformers/官方路径的数值 parity；
- 大规模 schema-v2 cache 正式生成；
- 真实 910B FSDP2 多卡长跑和中断恢复；
- DFlash/DFlash2/DSpark 的 **GLM-5.2 Ascend proposer adapter** 与 load/parity；
- 多节点 BF16 GLM-5.2 target-only/speculative 端到端 TPS 评测。

## 3. 不可更改的模型与数据契约

```text
target                          GLM-5.2 BF16
target depth                    78
logical target hidden layers    [1, 20, 38, 56, 75]
draft layers                    5
draft hidden/intermediate       6144 / 12288
query heads / KV heads          64 / 64
head_dim                        64
Q/K/V width                     4096
O projection                    4096 -> 6144
attention                       full attention, no sliding window
RoPE theta                      8e6
RMS epsilon                     1e-5
aux hidden                      [T, 5, 6144]
target final hidden             [T, 6144], post-final-norm LM-head input
```

三种方法必须统一使用 logical layers `[1,20,38,56,75]`。不要恢复曾讨论过的
`[8,23,39,55,70]`、三层 draft、sliding-window 或其他旧 GLM/DSpark 配置。

Stage A、Stage B、tokenizer、embedding 和 LM head 必须来自同一不可变 revision。
不得用 W8A8/W4A8C8 服务生成 trajectory，却用 BF16 权重提 hidden 并声称是同一个
BF16 实验。若必须研究量化，单独建 ablation，不污染主实验。

## 4. 唯一生产数据流

```text
cbyzju/vibe coding 630k
  -> Stage A: GLM-5.2 sampling agent rollout
     -> exact prompt_token_ids + sampled response_token_ids
     -> frozen input_ids + target-position loss_mask
  -> Stage B: same frozen path, one teacher-forced target forward
     -> auxiliary hidden [T,5,6144]
     -> final hidden [T,6144]
  -> schema-v2 cache + frozen target token I/O
  -> DFlash / DFlash2 / DSpark offline training
  -> runtime-specific export adapter
  -> acceptance/TPS evaluation
```

Stage A 默认 sampling policy：`temperature=1.0, top_p=0.95, top_k=-1`。如果真实线上
policy 不同，显式修改并重新生成 shard；不能在同一 shard resume 时静默改参数。

Stage B 中的 temperature=0 不表示重新 greedy 生成。Stage B 只沿 Stage A 已冻结的
`input_ids` teacher force，因此不会改变 response path。

## 5. Stage A：真实 trajectory 生成

主要文件：

- `scripts/run_stage_a_trajectories.sh`
- `tools/generate_trajectories.py`
- `src/glm_dflash2/agent_trajectory.py`
- `src/glm_dflash2/sglang_stage_a.py`
- `src/glm_dflash2/vibe_coding.py`
- `src/glm_dflash2/workspaces.py`
- `src/glm_dflash2/open_swe_trajectories.py`

### 5.1 为什么必须保存 sampled response token IDs

文本不是唯一 token path。空格、特殊 token、tool-call JSON 和 tokenizer normalization
都可能让“把文本重新 tokenize”得到另一条序列。Stage B 与离线训练必须复现 target
真正采样出的 IDs，因此新 rollout 必须同时返回：

- `prompt_token_ids`；
- `response_token_ids`；
- 对应 response 文本，仅作为可读记录。

缺失 IDs 或完整回放逐 token 不一致时整条样本失败，不能自动退回 text re-tokenize。

### 5.2 `loss_mask` 语义

`loss_mask[i]=1` 表示 token 位置 `i` 是本次 trajectory 中由 assistant 生成、可作为
drafter 训练目标的位置。System、user、历史上下文和 tool observation 为 0。它是
target-token-position mask，不是普通 causal LM 的 shifted-label mask。

### 5.3 现有 GLM 服务能否直接复用

服务器上已经有一个“像 Claude CLI 一样使用”的本地 GLM-5.2 服务。它只能在满足以下
条件后复用到 Stage A：

1. 有 OpenAI-compatible chat/completions HTTP endpoint，而不只是 CLI wrapper；
2. 非流式响应可返回 prompt IDs 和 sampled response IDs；
3. 能提供 endpoint manifest：模型路径/revision、tokenizer、BF16、SGLang/CANN/镜像、
   TP/EP 拓扑；
4. sampling 参数可以固定；
5. capability probe 成功。

若现有服务只返回文本，它可以用于人工体验，不能直接产生产训练 cache。先给服务补 token
IDs，或者从其内部 scheduler/response 对象接出真实 IDs。

### 5.4 Stage A 首次 smoke

```bash
cd /path/to/glm-dflash2

MODEL_PATH=/shared/models/GLM-5.2-bf16 \
ENDPOINT=http://glm52-sglang-service:30000 \
ENDPOINT_MANIFEST=/shared/identity/glm52-endpoint.json \
SERVED_MODEL_NAME=GLM-5.2 \
WORKSPACE_MAP=/shared/data/workspace_map.jsonl \
WORKSPACE_CACHE=/shared/cache/vibe-workspaces \
OPEN_SWE_STORE=/shared/data/open_swe_original.sqlite \
OUTPUT_JSONL=/shared/out/trajectory-smoke.jsonl \
WORKERS=4 MAX_RUNNING_REQUESTS=1 MAX_SAMPLES=50 \
bash scripts/run_stage_a_trajectories.sh
```

`WORKERS` 是 trajectory/tool/workspace 并发；`MAX_RUNNING_REQUESTS` 只限制本进程同时
发出的模型 HTTP 请求。外部 endpoint 的全局 token pool 和其他客户端不受该 semaphore
控制。先用 `4/1` 检查长上下文 HBM，再尝试默认 `8/2`，不要直接复制 Qwen3.8 的
`12/8` 并发。

Stage A smoke 通过标准：

- capability probe 返回两类 IDs；
- 50 条无 duplicate committed IDs；
- exact replay 全部通过；
- error ledger 中无未解释系统性错误；
- sampling/model/tokenizer identity 写入 manifest；
- `MAX_SAMPLES` 输出标为 `partial`，不能冒充完整 frozen dataset。

当前 patch `scripts/apply_sglang_v0516_token_ids_patch.sh` 只接受精确 SGLang 0.5.16。
服务器若使用其他厂商镜像或版本，**不要强行运行这个 patch**；先检查其响应 schema，再对
实际版本做最小补丁和 capability test。

Open-SWE 历史完整轨迹没有当时的 sampled token metadata，因此只能明确标为
`teacher_forced_original_trajectory`，不能伪装成 verified sampled IDs。

## 6. Stage B：同一路径 hidden 提取

主要文件：

- `scripts/run_stage_b_hidden.sh`
- `tools/extract_hidden_sglang.py`
- `src/glm_dflash2/sglang_hidden_runner.py`
- `src/glm_dflash2/hidden_extraction.py`
- `src/glm_dflash2/hidden_cache.py`

Stage B 需要模型内部 hidden hook。普通 CLI/OpenAI endpoint 不够；必须在实际 SGLang
fork 内运行 internal `ModelRunner`，或给服务增加等价的离线 hidden capture 入口。

当前代码的 capture 约定：

```text
logical layer IDs       [1,20,38,56,75]
physical layer inputs   [2,21,39,57,76]
aux hook                model.model.layers_to_capture[physical_layer]
final hook              model.model.norm.forward_output
```

这只是根据当前 GLM/SGLang 结构实现的映射，必须用真实权重做 direct-forward 数值 gate，
不能仅凭 shape 判定正确。

示例：

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TRAJECTORY_JSONL=/shared/out/trajectories-shard-0.jsonl \
OUTPUT_DIR=/shared/out/hidden-v2-shard-0 \
TP_SIZE=32 EP_SIZE=32 NNODES=2 NODE_RANK=0 \
DIST_INIT_ADDR=<rank-0-host> NCCL_PORT=29500 \
bash scripts/run_stage_b_hidden.sh
```

每个节点各运行一份，`NODE_RANK` 唯一。实际 TP/EP/节点数必须服从服务器已经验证的
GLM-5.2 BF16 部署拓扑，示例中的 32/32/2 不是对所有机器的强制值。

### 6.1 schema-v2 cache

```text
hidden-cache/
  manifest.json
  index.jsonl
  segment-00000/
    input_ids.bin               # int64 [T]
    loss_mask.bin               # uint8/bool [T]
    aux_hidden_states.bin       # BF16 [T,5,6144]
    target_final_hidden.bin     # BF16 [T,6144]
```

每个 slice 都有 SHA-256。stream 先 flush，index line 才是 sample commit point。v1 cache
缺 final hidden，只允许显式 legacy 诊断，所有统一训练入口均拒绝 v1。

六个 BF16 hidden 向量每 token 为 73,728 bytes。若 630K 样本平均 1,000 token，hidden
约 46.4 TB，还未包含 IDs、index 和文件系统开销。正式开跑前必须按真实 smoke token
分布估算容量，决定样本量、shard、保留策略和存储位置。

### 6.2 必须完成的 hidden/logit parity

在同一短 `input_ids` fixture 上保存：

1. 三次独立 direct-forward：`direct-1/2/3.pt`；
2. 一个 shifted-layer 负对照：`shifted.pt`；
3. 一个 pre-final-norm 负对照：`prenorm.pt`。

每个文件都要含 `input_ids`、`layer_ids`、`aux_hidden_states`、
`target_final_hidden`、`target_logits`。当前仓库提供校准/验证工具，但没有能够跨所有厂商
GLM fork 自动生成这五个 fixture 的万能脚本；服务器 AI 必须基于真实模型类接入 direct
forward，不能制造假 reference。

```bash
python tools/calibrate_hidden_capture_gate.py \
  --direct-run direct-1.pt --direct-run direct-2.pt --direct-run direct-3.pt \
  --shifted-layer-control shifted.pt --pre-norm-control prenorm.pt \
  --target-fingerprint <sha> --model-revision <revision> \
  --tokenizer-fingerprint <sha> --cann-version <version> \
  --torch-npu-version <version> --sglang-version <version> \
  --output hidden-parity-gate.json

python tools/validate_hidden_cache.py \
  --cache-dir /shared/gate/hidden-v2 \
  --reference-pt direct-1.pt \
  --parity-gate hidden-parity-gate.json \
  --runtime-identity-json runtime-identity.json \
  --target-io-dir /shared/out/glm52-target-io
```

阈值只允许由 direct-vs-direct 噪声与预先指定 floor 校准一次，必须显著低于两个负对照。
生产验证失败时修 hook/位置/精度，不得现场放宽阈值。

## 7. 冻结 target token I/O

```bash
MODEL_PATH=/shared/models/GLM-5.2-bf16 \
TARGET_IO_DIR=/shared/out/glm52-target-io \
bash scripts/extract_glm52_io.sh
```

artifact 必须是 dense BF16/FP16/FP32，并与 Stage A/B revision、tokenizer fingerprint
一致。量化权重、额外 logit scale/softcap、非 identity output path 均应 fail closed。

训练阶段不再加载 753B target transformer；只读取 cache 与冻结 token I/O。因此 DFlash、
DFlash2、DSpark 可以复用同一批数据与 hidden，主要显存来自 drafter、optimizer 和局部
full-vocab loss，而不是 target。

## 8. 三种训练逻辑

共同主干是五层 GLM DFlash shape，full attention，absolute positions。一个 block 内 local
query 互相可见，context 只含 anchor 之前的真实 prefix；位置 ID 不能重置为 `0..K-1`。

### DFlash

- common backbone；
- full-vocabulary CE；
- 深度位置权重 `exp(-depth/gamma)`；
- B8 gamma 4，B16 gamma 7。

### DFlash2

- DFlash base CE；
- identity-initialized two-tap grouped dynamic convolution；
- rank-256/top-16 candidate selector；
- target 位于 base top-16 时计算 selector CE。

这是本仓库的统一实现。是否与最终采用的官方 DFlash2 serving kernel 完全同 ABI 尚未被
真实环境证明，因此必须做 selector/top-k/logits parity。

### DSpark

- plain backbone；
- rank-256 Markov head：`Embedding(V,256) -> Linear(256,V)`；
- Markov-aware confidence head；
- final hidden 重建 target distribution；
- loss：`0.1 CE + 0.9 TV + 1.0 confidence BCE`；
- `TV=0.5*sum_v |p_target(v)-p_draft(v)|`；
- confidence soft target 为 `1-TV`。

target token 位置 `p` 使用 final hidden `p-1`。depth 0 predecessor 是 clean anchor，后续
depth 使用 teacher-forced previous target。LM-head matmul保持 BF16，logits 与归一化用
FP32；target embedding/lm-head 冻结且不进入 optimizer。

### 启动示例

```bash
METHOD=dflash2 \
BLOCK_SIZE=16 \
CACHE_DIR=/shared/out/hidden-v2-frozen \
TARGET_IO_DIR=/shared/out/glm52-target-io \
OUTPUT_DIR=/shared/out/glm52-dflash2-b16 \
MASK_TOKEN_ID=<真实 tokenizer/runtime ID> \
PAD_TOKEN_ID=<真实 ID> \
NUM_NPUS=8 \
bash scripts/train_glm52_drafter_910b.sh
```

入口：

- `scripts/train_glm52_drafter_910b.sh`：唯一主 launcher；
- `tools/train_drafter_offline.py`：统一 CLI；
- `scripts/train_glm52_dflash2_910b.sh`：仅为旧命令兼容 wrapper，不是另一套实现。

不要用 EOS/PAD 猜测 MASK ID。必须从实际 tokenizer/runtime 验证。

### 训练前硬 gate

```bash
PY=/path/to/vendor/python bash scripts/smoke_no_model.sh

CACHE_DIR=/shared/gate/hidden-v2-frozen \
TARGET_IO_DIR=/shared/out/glm52-target-io \
MASK_TOKEN_ID=<verified-id> \
OUTPUT_DIR=/shared/gate/train-2rank \
bash scripts/gate_train_2rank_910b.sh
```

要求：finite loss/gradient、target I/O hash 不变、FSDP2 两 rank 正常、完整 checkpoint
marker 存在、uninterrupted 与 resume 的下一步输出/状态一致。只从带 `COMPLETE` 标记的
checkpoint 恢复；cache、method、architecture、optimizer、scheduler identity 不得改变。

## 9. Export 不等于已经可 serving

训练会写：

```text
OUTPUT_DIR/
  step-N/                 # resumable training checkpoint
  export/
    config.json
    config.py
    model.safetensors
    export_manifest.json
```

export 含 drafter、冻结 `embed_tokens`、冻结 `lm_head` 和 checksum。它证明训练产物完整，
不自动证明某个推理框架已经认识其 GLM config、attention、Markov/confidence 或 DFlash2
selector。

本地模型代码借用 `Qwen3Config` 作为配置容器并实现固定的 GLM draft shape；这不表示
目标是 Qwen，也不表示 GLM runtime 会自动把 `model_type=qwen3` 当成正确 proposer。
serving adapter 必须显式完成 config/state 映射，不能仅靠 `trust_remote_code` 碰运气。

截至本文日期，vLLM-Ascend 官方 speculative decoding 文档虽已有 DFlash/DSpark 入口，
但明确注明 DSpark 当前只支持 Qwen，GLM/DeepSeek 仍在逐步适配。官方 Speculators 文档
存在 GLM-5.2 DSpark preview，并不等于所用 Ascend 版本已经完成 GLM 适配。因此：

- DFlash：需要在目标 vLLM-Ascend/厂商 fork 验证 GLM-5.2 draft config 和 proposer；
- DFlash2：需要 method-specific proposer/selector adapter；不能伪装成普通 DFlash；
- DSpark：需要合入并验证 GLM-5.2 Markov/confidence adapter；不能照搬 Qwen 路径宣称可用。

官方状态参考：

- <https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html>
- <https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md>
- <https://github.com/vllm-project/vllm-ascend/pull/11066>

服务器 AI 的首选策略不是另造模型，而是检查部署服务器正在使用的 vLLM-Ascend/SGLang
fork 是否已经包含 GLM-5.2 DFlash/DSpark patch；若有，按其实际 config/key/forward ABI
写 exporter adapter。若没有，在该 fork 内实现 proposer，并加 offline-vs-runtime parity。

## 10. Runtime parity 与最终评测

在启动大 benchmark 前，用固定短 batch 对比 offline 与 runtime：

- all：base logits、top-1 token、position IDs、proposal 数；
- DFlash2：base top-16 candidate IDs/scores、selector scores；
- DSpark：Markov chunk scores、confidence logits；
- greedy verify：最终输出必须与 target-only 逐 token 相同。

当前评测入口：

- `scripts/eval_vllm_ascend.sh`
- `tools/benchmark_vllm_ascend.py`
- `src/glm_dflash2/vllm_eval.py`

它会在同一组设备上串行启动 baseline 与 speculative server，读取 vLLM Prometheus
`spec_decode_num_drafts`、`spec_decode_num_draft_tokens`、
`spec_decode_num_accepted_tokens`，并计算：

```text
bonus-inclusive mean acceptance = 1 + accepted_tokens / drafts
TPS = completion tokens / wall time
speedup = speculative TPS / target-only TPS
```

示例：

```bash
TARGET_MODEL=/shared/models/GLM-5.2-bf16 \
DRAFTER_EXPORT=/shared/out/glm52-dspark-b8/export \
PROMPTS_JSONL=/shared/eval/fixed-prompts.jsonl \
OUT_DIR=/shared/eval/glm52-dspark-b8 \
TP_SIZE=16 MAX_SAMPLES=100 MAX_TOKENS=2048 \
bash scripts/eval_vllm_ascend.sh
```

注意：该 launcher 当前只会设置单节点可见设备并启动一个本地 `vllm serve`。GLM-5.2
BF16 通常无法用示例 TP16 单节点承载；必须按服务器已工作的多节点 TP/EP 启动方式改造
server launch 部分，benchmark client/compare 部分可以复用。不要因为脚本能 `bash -n`
就声称多节点评测已经支持。

Block Verify、Entropy Verify 会改变接受规则甚至采样精度，主结果默认禁用。若研究它们，
单列近似/有损 ablation，不能混入无损 speculative 主表。

## 11. 服务器到手后的严格执行顺序

### P0：先记录环境，不改代码猜测

保存到 `server_identity.md/json`：

- 节点数、每节点 910B 型号/数量/HBM；
- OS/架构、CANN、驱动、固件、torch、torch-npu；
- SGLang、vLLM、vLLM-Ascend commit/image digest；
- 已部署 GLM 服务的启动命令、模型 revision/精度、TP/EP、endpoint；
- tokenizer path/fingerprint、MASK/PAD/EOS IDs；
- 数据盘可用空间、inode、吞吐和共享盘语义。

不要仅记录包版本号；厂商 fork 的 git commit 和容器 digest 同样重要。

### P1：Stage A 50 条 capability + replay smoke

若失败，先修真实 token ID 返回，不准 text re-tokenize 绕过。

### P2：Stage B 1 条 shape smoke + direct numerical gate

依次确认 hook 存在、shape/dtype/position 正确，再做三 direct + 两 negative controls。

### P3：target I/O 与 schema-v2 100 条端到端 cache

校验 sample ID、token path、loss mask、checksum、final hidden、磁盘估算与 resume。

### P4：三方法单步 + 两 rank resume gate

先 B8 三方法，再 B16 两方法。比较相同 sample/anchor 上的输入完全一致。

### P5：小规模多步过拟合与方法 sanity

每种方法在固定小集训练 100～500 optimizer steps，确认 loss 可下降、proposal 不全是
MASK/PAD、DSpark confidence 非退化、DFlash2 selector 有监督覆盖率。

### P6：完整数据生成和正式训练

先冻结 Stage A manifests，再生成 Stage B。不要边生成 hidden 边改变 target service。

### P7：runtime adapter + offline/runtime parity

任何方法 parity 未过，都不能跑 TPS 主表。

### P8：串行 benchmark

相同硬件、prompt、max tokens、sampling、并发、warmup；先 target-only，彻底释放，再
speculative。报告 acceptance、TPS、speedup、P50/P95 latency、输出一致性和服务版本。

## 12. 明确阻点与处理原则

| 优先级 | 阻点 | 当前原因 | 正确处理 |
|---|---|---|---|
| P0 | 现有 GLM 服务身份未知 | CLI 可用不等于训练接口满足契约 | 获取启动命令、endpoint、revision、精度、token IDs capability |
| P0 | Stage B hook 未在真实 fork 校准 | 不同 GLM/SGLang 版本模块名/层语义可能变 | direct parity + negative controls，禁止只看 shape |
| P0 | GLM Ascend proposer 未验收 | 官方当前并非所有 GLM DFlash/DSpark 都 stock 支持 | 对实际 serving fork 做 adapter/load/logit parity |
| P0 | 多节点 eval launcher 缺失 | 当前脚本只启动单机 `vllm serve` | 复用现有 GLM 多节点 launch，保留 benchmark/compare |
| P1 | 630K hidden 可能数十 TB | 每 token 六个 6144-d BF16 向量 | 用 smoke 真实 token 分布估算，先定存储预算和样本规模 |
| P1 | SGLang token-ID patch 版本特定 | 当前 patch 只认 0.5.16 | 先 capability probe，再针对实际 fork 最小修改 |
| P1 | FSDP2/HCCL 只做了本地逻辑测试 | CPU/CUDA 不能代替 910B collectives | 两 rank gate 后再放大 |
| P1 | DFlash2 官方 ABI 可能不同 | 本地 objective/export 完整，但 runtime 未绑定 | 以最终 runtime kernel 的 tensor contract 为准做 parity |

遇到硬件/框架错误时，先保存完整命令、环境 identity、首个 root-cause traceback、HBM 和
rank 日志。不要反复改超参掩盖 adapter、position、dtype 或 distributed bug。

## 13. 接手 AI 每轮汇报格式

```text
阶段：Stage A / Stage B / I/O / Train / Export / Runtime / Eval
实际命令：完整可重放命令
环境：镜像、commit、CANN、torch-npu、拓扑
输入 identity：模型/tokenizer/cache fingerprint
通过的 gate：列证据文件及关键数值
失败点：第一处 root cause，不贴无关连锁报错
修改：文件、函数、为什么改
验证：正测试 + 负测试
下一步：唯一最小动作
```

任何“成功”都要附 artifact 路径和数值证据。不得只说“应该可以”“模型加载了”或“loss
在下降”。

## 14. 代码导航

```text
Stage A        scripts/run_stage_a_trajectories.sh
               tools/generate_trajectories.py
               src/glm_dflash2/agent_trajectory.py
Stage B        scripts/run_stage_b_hidden.sh
               tools/extract_hidden_sglang.py
               src/glm_dflash2/sglang_hidden_runner.py
Cache          src/glm_dflash2/hidden_cache.py
Parity         tools/calibrate_hidden_capture_gate.py
               tools/validate_hidden_cache.py
Target I/O     scripts/extract_glm52_io.sh
Training       scripts/train_glm52_drafter_910b.sh
               tools/train_drafter_offline.py
               src/glm_dflash2/offline_trainer.py
Models/loss    src/glm_dflash2/draft_backbone.py
               src/glm_dflash2/dflash2_model.py
               src/glm_dflash2/dspark_model.py
               src/glm_dflash2/method_objectives.py
Checkpoint     src/glm_dflash2/checkpointing.py
Export         src/glm_dflash2/speculator_export.py
Evaluation     scripts/eval_vllm_ascend.sh
               tools/benchmark_vllm_ascend.py
Tests          tests/
```

## 15. 最终完成定义

只有以下全部满足，项目才算“正确数据生成 + 训练 + 评测”完成：

- Stage A 完整 frozen manifests，exact sampled IDs，无 unresolved system error；
- Stage B schema-v2 与 target I/O identity 一致，direct hidden/logit parity 通过且负对照
  被拒绝；
- 五组训练均从同一数据契约启动，FSDP2/resume/checksum 正常；
- export 能被实际 Ascend runtime 加载，offline/runtime logits 与方法特有中间量 parity
  通过；
- greedy target-only/speculative 输出逐 token 一致；
- sampling 使用标准无损 rejection sampling，并记录 policy/seed；
- acceptance、TPS、speedup、latency 在相同硬件和负载下串行可复现；
- 所有命令、环境 identity、manifest、checkpoint、export 和 benchmark JSON 可追溯。

当前仓库已经把本地可实现的框架、校验和 fail-closed 逻辑补齐；剩余工作主要是**真实
服务器身份确认、910B 数值/分布式 gate、GLM-5.2 proposer runtime 适配和多节点实测**。
这些问题无法在没有目标服务器与实际厂商 fork 的机器上诚实地宣称完成。
