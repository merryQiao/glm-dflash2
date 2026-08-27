# Qwen3-Omni Thinker Ascend 服务器接管手册

更新时间：2026-08-27

> 这是 Omni 子项目的唯一接管入口。它只处理
> `Qwen3OmniMoeThinkerForConditionalGeneration` 的文本 token 流；Talker 和
> Code2Wav 不参与数据生成、hidden 提取或性能统计。

## 1. 目标与已选方案

采用方案 B：两个阶段均使用 **vLLM-Ascend**，但必须分开运行。

```text
accepted conditions
  -> Stage A: vLLM-Ascend Thinker sampling
     -> exact prompt_token_ids + response_token_ids
  -> Stage B: 回放同一 token path，vLLM extract_hidden_states
     -> target_hidden_states      [T, 5, 2048]
     -> target_last_hidden_states [T, 2048]
  -> 后续 drafter 训练数据
```

Stage B 是 teacher forcing，不再采样，不允许由文本重新 tokenize 构造目标。
它请求 logical layers `[1,12,24,36,47]` 以及 vLLM synthetic layer `48`；第 48
层是 raw post-decoder/pre-final-norm state，代码再用同一 checkpoint 的
`thinker.model.norm.weight` 执行 RMSNorm，才得到 LM-head input。

## 2. 本地已完成的实现

- A2/A3 配置校验，BF16 主实验拒绝 `quantization: ascend`；
- Stage A 精确 engine token IDs、condition-local seed、atomic shard 和 resume identity；
- Stage B 精确 token 回放、hidden connector 输出检查、final RMSNorm 和 provenance；
- text/image/audio/video accepted-condition 数据契约；
- A2/A3 launcher 和 `ASCEND_SMOKE_ATTESTATION.json` 验收入口；
- `inference_qwen3-omni.py` Thinker-only 性能入口，复用 Stage A 的 engine、
  processor、request builder 和 sampling provider；它已经补齐 preprocess / engine /
  end-to-end 三套时间、按模态汇总、可选 benchmark scorer、TP worker HBM、组件可用性和
  checksum success marker；
- CPU fake-engine 契约测试覆盖准确计时、精确 token IDs、scorer、HBM RPC/reduction 和
  原子发布。

上述证明代码契约，**不证明**目标 vLLM-Ascend 镜像的 NPU kernel、
multimodal hidden connector 和 layer-48 语义已经通过实机验收。

## 3. 必须保持的契约

```text
component               Thinker only
dtype                   BF16
sampling                temperature=0.7, top_p=0.8, top_k=20
hidden layers           [1,12,24,36,47]
final raw layer         synthetic 48
aux hidden              [T,5,2048]
final hidden            [T,2048], post_final_norm_lm_head_input
engine ownership        one Python process owns one complete TP/EP group
```

Stage A/Stage B 必须使用同一 model revision、processor revision、tokenizer、
sampling manifest 和 runtime identity。不要用 `torchrun` 启动一个 vLLM TP engine。多 worker
扩展时，每个 worker 必须使用不重叠的 NPU 集合和独立 shard。

## 4. 服务器上的最小执行顺序

### P0：固定环境

使用支持 Qwen3-Omni 的 vLLM-Ascend 镜像；当前配置基线是
`quay.io/ascend/vllm-ascend:v0.23.0`。记录镜像 digest、vLLM/vLLM-Ascend commit、
CANN、torch-npu、driver、firmware 和 A2/A3 型号。不要在厂商镜像里重装 torch。

修改 `configs/generate_thinker_data.yaml`：

- 模型与 processor 必须是不可变 commit/revision；
- 正确填写 accepted-condition 路径和 `expected_conditions`；
- `runtime.hardware` 为 `a2` 或 `a3`；
- `tensor_parallel_size` 与可见 NPU 数一致；
- 输出和 connector scratch 放在稳定存储上。

### P1：四模态 smoke

smoke accepted-condition 必须至少包含 text、image、audio、video 各一条。

```bash
cd /path/to/omni-sd-ascend
ASCEND_HARDWARE=a2 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
TP_SIZE=4 \
CONFIG=$PWD/configs/smoke_a2.yaml \
bash scripts/smoke_ascend.sh
```

A3 使用 `ASCEND_HARDWARE=a3`；代码只对 A3 设置
`HCCL_OP_EXPANSION_MODE=AIV`。smoke 成功后必须存在
`ASCEND_SMOKE_ATTESTATION.json`，并确认：样本数完整、token 路径一致、
hidden 全为 finite、final semantics 为 `post_final_norm_lm_head_input`。

### P2：正式两阶段数据

```bash
ASCEND_HARDWARE=a2 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/generate_thinker_trajectories_ascend.sh

ASCEND_HARDWARE=a2 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/extract_thinker_hidden_ascend.sh
```

只有 Stage A 完整冻结后才启动 Stage B。若修改 model/runtime/sampling 或输入身份，
建新输出目录，不要在原 shard 上强行 resume。

### P3：Thinker 性能

```bash
ASCEND_HARDWARE=a2 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 \
CONFIG=$PWD/configs/generate_thinker_data.yaml \
bash scripts/profile_thinker_ascend.sh \
  --conditions-parquet /path/to/accepted_conditions.parquet \
  --limit 128 --warmup 1 \
  --output-jsonl outputs/thinker_profile.jsonl \
  --profile-json outputs/thinker_profile.json
```

性能报告只包含 Thinker；必须报告 model-load/warmup 时间、实测 wall time、
engine-only 与 end-to-end 的 request/s、completion-token TPS、total-token TPS、按模态
latency/throughput，以及每个 TP worker 的 `torch_npu_allocator` HBM。默认缺 HBM 即失败；
`--allow-missing-hbm` 只允许用于诊断，不能用于正式表格。不要把 dry-run 的 plan JSON
当成性能结果，也不要把 batch latency 复制成每个 request 的 latency。

warmup 会为输入中每个实际 modality 运行第一个真实 batch shape，然后调用 pinned vLLM
公开的 `LLM.reset_mm_cache()` 清空多模态 processor cache，再进入正式测量。该 reset 失败
时必须终止，不能让第一批 measured media 因 warmup cache hit 获得虚假的低延迟。

正式结果必须同时存在 JSONL、profile JSON 和 `<profile>.SUCCESS.json`，且 marker 中
两个 SHA-256 与文件一致。Audio/Vision Encoder/Thinker 的内部 event timing 当前不可观测；
Talker、MTP/code predictor、Code2Wav 没有加载。profile 会明确记录 unavailable，不得填 0。
JSONL/profile/marker 三个 final path 会在整个 inference/publish 期间同时加锁，不能绕过锁
并发写共享输出路径。

若需要 benchmark accuracy，使用 JSONL 输入并为每条样本增加：

```json
{"evaluation":{"metric":"normalized_exact_match","reference":"expected"}}
```

同一次运行只能使用一种 `omni_eval_v1` metric；无 reference 的样本计入 skipped。

## 5. 目标机上仍需关闭的阻点

1. 确认目标 image 真正注册 Qwen3-Omni Thinker，而不只是能解析 config。
2. 验证 `extract_hidden_states` 在 multimodal + NPU 上可用，特别是
   `ExampleHiddenStatesConnector` 的 async host-copy 路径。
3. 用一条小 fixture 对比直接 forward，证明 logical layers 和 synthetic layer 48
   没有 off-by-one，且离线 RMSNorm 与官方 final norm 一致。
4. 确认 connector 实际输出 layout 为 `[tokens,layers,hidden]`；不得只凭 shape
   猜测 layer 轴。
5. 在正式数据量上验证长序列 HBM、scratch 盘、atomic rename 和 resume。
6. profile 实机 gate 必须包含 text、single-image、multi-image、audio、video、mixed
   各至少一条；确认 worker RPC 返回完整且物理 NPU 唯一的 TP ranks，HBM 值 finite，
   每种实际 payload 均进入对应 `performance.by_modality`，最后生成 success marker。

任一项失败时应修正目标 runtime 的 connector/adapter，不允许退化为文本重
tokenize、Transformers 第二份 target 副本，或未归一化的 final hidden。

## 6. 重要文件

- `README.md`：英文契约和完整命令；
- `configs/generate_thinker_data.yaml`：唯一运行配置；
- `scripts/generate_thinker_trajectories_ascend.sh`：Stage A；
- `scripts/extract_thinker_hidden_ascend.sh`：Stage B；
- `scripts/smoke_ascend.sh`：A2/A3 实机 gate；
- `inference_qwen3-omni.py` / `scripts/profile_thinker_ascend.sh`：Thinker 性能；
- `src/omni_sd/vllm_ascend_generation.py`：engine/request/sampling 生产路由；
- `src/omni_sd/vllm_ascend_hidden.py`：hidden provider 和 connector 契约；
- `scripts/data/attest_ascend_smoke.py`：实机 smoke attestation。

## 7. 接手 AI 每轮汇报格式

```text
阶段：Config / Stage A / Stage B / Parity / Profile
完整命令：可重放命令
环境：image digest、commits、CANN、torch-npu、A2/A3、TP/EP
输入 identity：model/processor revision、accepted-condition fingerprint
通过的 gate：证据文件与关键数值
失败：第一个 root-cause traceback，不贴连锁错误
修改：文件、函数、理由
验证：正测试 + 负测试
下一步：唯一最小动作
```
