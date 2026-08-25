# Offline GLM-5.2 DFlash2 Training Design

## Goal

Train a GLM-5.2-compatible DFlash2 drafter on Ascend 910B without loading the
753B target during training. The trainer consumes the frozen two-pass cache
(`input_ids`, token-position `loss_mask`, and five selected target hidden
layers), plus a small checkpoint containing only GLM-5.2 `embed_tokens` and
`lm_head` weights.

## Scope and compatibility boundary

This is an offline port of the official DFlash2 objective and Qwen3-shaped
draft stack, not a claim that upstream NeMo AutoModel already supports GLM-5.2
as a DFlash2 target. The produced draft uses the published serving key layout
for the grouped convolution and candidate selector. Actual SGLang-on-Ascend
serving still needs a hardware gate proving that the runtime supplies the same
five GLM hidden slices and interprets the exported config identically.

The first implementation is deliberately fixed to the normal DFlash2 setup:

- five dense Qwen3-style draft layers;
- block size 16, with one clean anchor and 15 mask positions;
- selected GLM layers `[1, 20, 38, 56, 75]` and hidden size 6144;
- position decay `exp(-k / 7)` for `k=0..14`;
- two-tap grouped dynamic convolution, group size 16;
- rank-256, top-16 pairwise selector;
- base block CE plus `1.0 * selector CE`;
- BF16 compute, HCCL, and FSDP2 for the trainable draft.

The serving-visible draft shape is fixed rather than inherited implicitly:
`hidden_size=6144`, `intermediate_size=12288`, 32 query heads, 8 KV heads,
`head_dim=128`, RMS epsilon `1e-5`, default RoPE with theta `8,000,000`, and a
2048-token draft sliding window. Attention is non-causal inside each draft
block, and all five layer types are `sliding_attention`. These values are
written in the export config and must be used by the runtime loader. The config
uses `model_type=qwen3`, `architectures=[Qwen3DFlash2DraftModel]`,
`is_causal=false`, `use_sliding_window=true`, `max_window_layers=5`, and a
`dflash_config` containing `block_size`, `mask_token_id`, `target_layer_ids`,
`conv_kernel_size`, `conv_group_size`, `selector_rank`, and `selector_top_k`.

The mask token ID is required at launch and is never guessed from pad/eos.

## Offline data contract

`PackedHiddenDataset` remains the single cache reader. Each sample supplies:

- `input_ids: int64[T]`;
- `loss_mask: bool[T]`, marking target-token positions belonging to generated
  assistant turns;
- `hidden_states: bf16[T, 5 * 6144]` in the exact configured layer order.

`hidden_states[t, i]` describes the same token position as `input_ids[t]`.
The archived IDs `[1,20,38,56,75]` are logical target-layer IDs. Stage B's
physical `+1` hook convention is provenance only; training and export never add
one again. At startup the trainer cross-checks cache and target-I/O manifests:
model/config/tokenizer/weight fingerprints, vocabulary and hidden dimensions,
logical layer order, cache dtype, and mask semantics must all agree.

An anchor is a supervised token position `a` with room for a complete block,
so `a <= T - 16`. The constructed block is
`[input_ids[a], MASK, ..., MASK]`. Block position `k` is aligned with source
position `a+k`; position zero is the clean anchor and is not supervised.
Positions 1 through 15 use source tokens `a+1:a+16` as labels and retain the
source `loss_mask`. This keeps the cached token-position convention unchanged;
there is no AR-style mask shift.

The dense SDPA mask has key/value layout
`[target context T | sampled blocks N*16]` and query layout `N*16`. For anchor
`a`, context K/V candidates are `hidden[0:a]`; `hidden[a]` is excluded. With
the fixed sliding window, block position `k` sees context positions `j` only
when `j<a` and `(a+k)-j<2048`. Block position `k` uses absolute RoPE position
`a+k`. Every query attends
bidirectionally to only its own block. Invalid/padded blocks retain self-block
attention to avoid all-masked softmax rows but carry zero loss. If no sample in
a microbatch has a valid full-block anchor, the microbatch is explicitly
skipped and counted rather than fabricating a label.

## Frozen target I/O checkpoint

A one-time extractor reads the safetensors index without constructing GLM-5.2
and copies only the dense floating-point tensors corresponding to
`model.embed_tokens.weight` and `lm_head.weight`. It validates vocabulary and
hidden dimensions against `config.json`, records source fingerprints and key
names, and writes a standalone safetensors file plus manifest.

The trainer loads these two tensors as frozen parameters. Embedding lookup
produces block inputs; the LM head projects draft hidden states to the target
vocabulary. They are held through non-registered references outside the FSDP2
root and excluded from optimizer groups, sharded state and gradient
synchronization. The LM-head matmul is not detached, so gradients still flow
into draft hidden states.
Quantized ModelSlim I/O tensors are intentionally rejected in this first
version: training uses BF16/FP16/FP32 source I/O weights even if serving uses a
W8A8 target.

## Draft model

The draft is a dense Qwen3-style transformer with GLM vocabulary and hidden
width. Its target feature projector maps `[T, 5*6144]` to `[T, 6144]`, followed
by RMSNorm. Each layer queries the block states and attends over projected
target context plus its own block states. RoPE uses absolute context positions
and repeated absolute block positions `anchor+k` from the training mask.

Each attention and MLP sublayer is wrapped by one identity-initialized two-tap
dynamic grouped causal convolution. Taps are reset at every 16-token block and
therefore never leak between sampled anchors. The selector retains the
backbone top-16 candidates and scores each candidate against the ground-truth
predecessor during training. Its successor codebook starts at zero, making a
fresh selector a no-op relative to the backbone logits.

Every sublayer convolution owns one shared prepare/finish projection and a
`base_kernel` of shape `[2,2,6144]`. Selector codebooks are direct Parameters
of shape `[154880,256]` (no `.weight` suffix). Export tests compare every key
and shape against the SGLang DFlash2 loader contract.

## Objective and metrics

For predicted depth `k=0..14`, let `m` be the gathered valid-token mask and
`w=m*exp(-k/7)`. The base term is exactly
`sum(w * full_vocab_nll) / sum(w)`. The selector uses the teacher-forced
predecessor `input_ids[a+k]` when predicting `input_ids[a+k+1]`; it participates
only if that true successor is in the same position's backbone top-16. With
`w_sel=w*has_target`, its term is
`sum(w_sel * selector_nll) / (sum(w_sel)+1e-6)` (zero if no target hits). The
total is

`loss = weighted_block_ce + selector_loss_weight * weighted_selector_ce`.

Metrics separately report base and selector token accuracy, bonus-inclusive
acceptance length, candidate recall, valid anchors/tokens, loss components,
learning rate, gradient norm, and throughput. Selector acceptance is the main
training metric because it is the path emitted at inference.

The LM projection never materializes `[batch, anchors*15, vocab]`. It is exact,
chunked over both predicted tokens and vocabulary: online log-sum-exp yields
the full-vocabulary CE while a running merge yields exact top-16 candidates.
Tests compare loss, candidate IDs/scores, and hidden gradients with a dense
reference. `num_anchors` is capped per sample by the real valid-anchor count;
padding blocks have zero loss. The first Ascend implementation fixes
`num_anchors=64` as both default and hard maximum for dense SDPA. For each
sample it draws `min(64, eligible_count)` eligible positions uniformly without
replacement with a dedicated CPU generator, sorts selected positions, and
pads to the batch maximum with anchor zero plus `keep=false`. The generator is
seeded from base seed, rank and epoch; its state and the sample cursor are
checkpointed for exact mid-epoch resume. Loss normalization matches the official local
microbatch objective; distributed metrics reduce additive numerators and
denominators rather than averaging per-rank means.

## Distributed training and checkpointing

Device selection is lazy: import `torch_npu` only for `--device npu`; otherwise
CPU/CUDA tests remain usable. Distributed training uses HCCL on NPU. FSDP2 is
applied bottom-up to each trainable draft layer and then the root draft; frozen
I/O modules stay outside the sharded draft. Gradient accumulation uses FSDP2
`set_requires_gradient_sync(False/True)` with matching reshard control, not the
FSDP1/DDP `no_sync()` API. The requirements document pins the validated
PyTorch/torch-npu/CANN tuple rather than installing generic PyTorch wheels.

Checkpoints are allowed only immediately after optimizer steps, never midway
through accumulation. They include sharded draft/optimizer/scheduler state,
global and micro steps, epoch and sample cursor, every rank's framework RNG,
the dedicated anchor RNG, and sampler state. All ranks join DCP/FSDP
collectives; only rank zero writes ordinary logs and the final consolidated
safetensors/config export. Resume must preserve LR, data order and sampled
anchors across accumulation and epoch boundaries.

## Verification gates

CPU tests use tiny dimensions and per-position/per-layer sentinels to detect
any +/-1 token shift, layer permutation, or wrong target-I/O checkpoint. They
also cover anchor/label alignment, exact mask semantics, no cross-block
convolution, identity initialization, selector
teacher forcing, weighted losses, frozen I/O gradients, save/resume, and
launcher defaults. The no-NPU smoke must overfit a deterministic tiny cache and
show finite gradients plus decreasing loss.

Before a full 910B run, the included executable two-rank NPU gate must prove:
HCCL/FSDP2
initialization, one optimizer step, finite trainable gradients, frozen I/O
weights unchanged, exact save/resume, and export reload equality. Before
serving, the included runtime-parity command compares the same cached block's
per-position backbone logits, selector top-k IDs/scores, pair scores and final
path between this implementation and SGLang. It also verifies whether the
artifact deliberately excludes frozen I/O and that the runtime shares the
target I/O weights. Both commands write machine-readable result artifacts;
until they pass on 910B, documentation calls the result "training-ready" and
does not claim serving compatibility.
