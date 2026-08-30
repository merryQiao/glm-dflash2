# GLM-5.3-Flash Stage B and Offline Drafter Training Design

## Goal

Extend the standalone GLM-5.3-Flash Stage A directory with an Ascend-first
Stage B hidden-cache pipeline and aligned offline training for DFlash,
DFlash2, and DSpark. The implementation must not import the adjacent GLM-5.2
package and must not claim runtime deployability before real Ascend parity and
linear-state rollback validation.

## Fixed production contract

- Target: GLM-5.3-Flash BF16 text backbone, 45 decoder layers, hidden width
  4096, vocabulary 154880, final RMSNorm epsilon `1e-5`.
- DFlash auxiliary logical decoder layers: `[1, 11, 22, 32, 42]` (zero based,
  post-decoder-block). This is the official DFlash uniform selection rule with
  `start=1`, `end=num_target_layers-3`, and five draft layers.
- Transformers hidden-state indices are `[2, 12, 23, 33, 43]`; SGLang capture
  metadata must record both logical and concrete indices.
- In addition to the five auxiliary streams, Stage B captures the
  post-final-norm LM-head input for every token.
- Cache row shapes are `input_ids [T]`, `loss_mask [T]`,
  `aux_hidden_states [T,5,4096]`, and `target_final_hidden [T,4096]`.
- Stage B teacher-forces the exact Stage A token path. It never regenerates a
  response and never changes the Stage A loss mask.

## Stage B architecture

Stage B uses a fresh SGLang internal PyTorch ModelRunner on Ascend. Execution
is deterministic: DP=1, PP=1, one request, no radix cache, no chunked prefill,
and no execution graph. The adapter installs the runtime's supported DFlash or
Eagle hidden capture hook, verifies the concrete physical taps, and installs a
forward hook on the text backbone final norm. It fails closed if the GLM-5.3
wrapper path, hidden shape, capture ordering, or final-norm semantics cannot be
proved.

The packed cache is append-only and resumable. Binary streams are flushed
before an index row is committed. Each slice and the frozen source trajectory
have SHA-256 identities. The cache manifest binds the target model,
tokenizer/chat template, SGLang/CANN image, target layer mapping, dtype, and
token contract. An incomplete Stage A artifact may only be consumed through an
explicit bounded smoke flag and can never produce a production-frozen cache.

Target token I/O extraction is a separate step. It exports dense BF16
`embed_tokens.weight` and `lm_head.weight`, validates `[154880,4096]`, rejects
an LM-head bias, and binds the same model/tokenizer fingerprint as the hidden
cache. The revised v3 contract also resolves the exact special `[MASK]` token,
requires production ID 154821, and rejects legacy mask-less artifacts.

All five training variants consume the same immutable cache identity; no
method or block size is allowed to trigger a second hidden extraction.
DFlash/DFlash2 ignore `target_final_hidden`, while DSpark requires it. A
`smoke_unverified` Stage A artifact is accepted only through an explicit smoke
flag and a total 50-sample bound. It propagates `production_eligible=false`,
can never freeze a production cache, and `smoke_failed`/other partial artifacts
are rejected.

## Shared draft architecture

All methods share a five-layer Qwen3-shaped dense full-attention backbone:

- hidden size 4096;
- intermediate size 12288;
- 64 query heads, 64 KV heads, head dimension 64;
- full attention, no sliding window;
- RMSNorm epsilon `1e-5`;
- one projection `Linear(5*4096,4096,bias=False)` followed by RMSNorm;
- frozen target embedding and LM head, not copied into optimizer state.

The draft deliberately does not copy the target's KDA/DSA recurrent attention.
That keeps DFlash inference stateless with respect to the target's linear
attention state and preserves the standard DFlash model interface.

## Method recipes

### DFlash

- Physical block sizes 8 and 16 (one clean anchor plus 7/15 masks).
- Exact chunked full-vocabulary CE.
- Depth weight `exp(-depth/gamma)`, gamma 4 for B8 and 7 for B16.

### DFlash2

- Same B8/B16 base objective and backbone.
- Identity-initialized two-tap grouped dynamic causal convolution.
- Rank-256, top-16 candidate selector. Training and inference use the same
  base top-16 candidates. Selector CE is evaluated only where the target is
  already present; misses contribute no selector numerator or denominator.
- Convolution kernel size 2 and channel group size 16.
- Selector coefficient 1.0. The optimized loss is the sum of two independently
  normalized means, `base_CE.mean + selector_CE_on_hits.mean`, matching NeMo
  AutoModel DFlash2. `unary_recall` remains a separate diagnostic.

### DSpark

- External block size 8 only, corresponding to seven internal queries and
  seven proposed tokens. Query zero contains the verified anchor and predicts
  the first successor; unlike DFlash, no internal output is discarded.
- Rank-256 vanilla Markov residual head and Markov-aware confidence head.
- Target logits are reconstructed from cached post-final-norm hidden and the
  frozen BF16 LM head.
- Exact loss: `0.1*CE + 0.9*TV + 1.0*confidence_BCE`, gamma 4.

All recipes use 64 deterministic anchors per sample, three epochs, AdamW
`betas=(0.9,0.95)`, learning rate `6e-4`, zero weight decay, 1000 optimizer
warmup steps, BF16 parameters/activations, FP32 normalization/reductions, and
gradient clipping at 1.0. The production per-rank batch is 1 and gradient
accumulation is 8, so the effective sample batch is `8 * world_size`. After
warmup, learning rate follows cosine decay to `0.1 * initial_lr`. Every loss is
represented by additive FP32 numerator/denominator pairs, globally summed over
HCCL before the mean/gradient normalization is formed.

Anchor selection uses global seed 42 and a stable hash of `(sample_id, epoch,
seed)`, samples without replacement from positions whose entire physical block
is inside the assistant loss mask, and is independent of rank or dataloader
order. Fewer than 64 valid anchors are padded with an explicit keep mask rather
than duplicating anchors.

## Ascend 910B A2 contract

The production launcher defaults to `device=npu`, HCCL, `torch_npu`, BF16, and
FSDP2. Imports of SGLang and torch_npu are lazy so CPU tools can validate data
contracts without initializing an NPU runtime. Production code cannot require
FlashAttention2, NCCL, CUDA graphs, or a CUDA-only synchronization API.

CPU synthetic tests establish shapes, loss semantics, cache atomicity,
checkpoint resume, and one finite optimizer update. They do not establish NPU
kernel support. A real 910B A2 gate must validate:

1. GLM-5.3 SGLang hidden hooks and exact tap mapping;
2. post-final-norm parity against direct LM-head logits;
3. BF16 FSDP2/HCCL single-step and resume parity;
4. HBM headroom for representative sequence/anchor distributions.

A production cache additionally requires live, runtime-owned evidence from
`torch.npu.get_device_name` and `torch_npu.npu.get_cann_version("CANN")`, and
the resolved SGLang `ModelRunner` and Ascend attention backend must be the
actual imported classes. Caller-provided strings, environment version values,
CPU fakes, and placeholder versions cannot satisfy this gate. TP-aware logits
projection is invoked by every TP rank; only rank zero consumes and validates
the full-vocabulary result.

The repository owns an executable Ascend gate that records these four results
and refuses production eligibility when any is absent. Distributed resume also
has a two-rank CPU/gloo parity test comparing interrupted and uninterrupted
model, optimizer, scheduler, RNG, sampler cursor, and global step; the same
parity is repeated with FSDP2/HCCL on A2.

## Evaluation boundary

Stage B and offline training do not need speculative state rollback. Formal
runtime evaluation does: after verifying a draft block and accepting only a
prefix, all 34 KDA/linear-attention recurrent and short-convolution states must
match the state produced by committing exactly that prefix. A simple KV crop or
pointer rollback is invalid.

This work therefore emits training-complete candidate checkpoints with an
immutable `runtime_attested=false` capability record and no deployable export
or speculative evaluation entrypoint. Any future runtime/TPS entrypoint must
hard-reject the artifact unless a separate rollback-strategy attestation and
all-state parity result are present; there is no override. Runtime integration is a later gate
and must choose and validate one exact strategy (per-step state snapshots,
recompute from a committed checkpoint, or discard/re-extend from the committed
prefix) before acceptance length/TPS claims are allowed.
