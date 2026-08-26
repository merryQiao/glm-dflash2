# Unified GLM-5.2 DFlash, DFlash2, and DSpark Design

## Goal

Build one GLM-5.2 BF16 trajectory and hidden-state dataset that can train fair,
method-aligned DFlash, DFlash2, and DSpark drafters on Ascend 910B. All three
methods use the same target data, auxiliary target features, five-layer draft
backbone, and token/block contract. Only the method-specific modules and losses
may differ.

## Experimental alignment

The common draft backbone follows the published GLM-5.2 DFlash checkpoint
rather than the Qwen3.8 DFlash2 checkpoint:

- target decoder depth: 78;
- target auxiliary layers: `[1, 20, 38, 56, 75]` in this exact order. These
  are zero-based decoder-block IDs in `0..77`; layer `k` means the residual
  stream immediately after decoder block `k`;
- target hidden size: 6144;
- draft layers: 5;
- draft hidden size: 6144;
- draft intermediate size: 12288;
- query heads: 64;
- KV heads: 64;
- head dimension: 64;
- attention: full attention in all five layers;
- sliding window: disabled;
- RoPE theta: 8,000,000;
- RMS norm epsilon: `1e-5`;
- physical block size is method-level: DFlash and DFlash2 use B8 and B16;
  DSpark uses only B8. Every physical block contains one known anchor, so B8
  predicts seven tokens and B16 predicts fifteen.

The attention projections do not derive `head_dim` from `hidden_size / heads`.
Q, K, and V each have width `64 * 64 = 4096`, and the output projection maps
`4096 -> 6144`. Tests enforce these nonstandard but published GLM DFlash
shapes.

DFlash2 adds only its two-tap grouped dynamic convolution and rank-256/top-16
path selector to this common backbone. DSpark adds only its Markov and
confidence heads and its distribution-aware objective. This makes comparisons
attributable to the method rather than to a different attention architecture or
target feature selection.

## Stage A: trajectory contract

Stage A continues to generate sampled GLM-5.2 tool-use trajectories with the
configured SGLang sampling parameters. The immutable record is rendered with
the local GLM-5.2 tokenizer and chat template. `loss_mask` marks target-token
positions belonging to generated assistant turns and remains unshifted.

The generated record and the tokenized record must retain their current
provenance fields. When the server returns response token IDs, they must match
the local frozen replay. When it does not, the manifest must continue to record
that exact response-token fidelity was unavailable. Stage B always consumes the
saved canonical `input_ids`, so the training labels and cached target features
remain internally aligned.

## Stage B: hidden cache schema v2

One target forward produces both auxiliary features and the final verifier
representation. No second GLM-5.2 forward is allowed.

Each sample stores four independent streams:

```text
input_ids:            int64[T]
loss_mask:            bool[T]
aux_hidden_states:    bfloat16[T, 5, 6144]
target_final_hidden:  bfloat16[T, 6144]
```

`aux_hidden_states[:, i]` is the output of logical target decoder layer
`[1, 20, 38, 56, 75][i]` at the same token position as `input_ids`. In a
Transformers hidden-state tuple whose element 0 is the embedding output, these
map to tuple indices `[2, 21, 39, 57, 76]`. Each backend stores a mapping record
containing its namespace, requested logical ID, concrete tap, and tap semantics;
a bare list of undocumented physical IDs is insufficient.

`target_final_hidden` has a stricter semantic contract: it is the
**post-final-norm tensor directly consumed by the frozen target LM head**. It is
not the raw output of decoder layer 77. This permits exact offline reconstruction
of target logits:

```python
target_logits = target_final_hidden @ lm_head.weight.T
```

The existing flattened compatibility view remains exactly five auxiliary
layers wide:

```text
hidden_states = aux_hidden_states.flatten(1)  # [T, 5 * 6144]
```

`target_final_hidden` must never be concatenated into this view because the
DFlash context projector consumes exactly `5 * 6144` features.

### Cache metadata

Schema v2 records:

- schema version;
- exact ordered logical auxiliary layer IDs;
- backend-specific capture mapping and tap semantics;
- target decoder depth;
- auxiliary and final hidden sizes and dtypes;
- `final_hidden_semantics = "post_final_norm_lm_head_input"`;
- target model revision or local fingerprint;
- tokenizer and chat-template fingerprints;
- generation sampling parameters.

Readers fail closed on missing streams, wrong dimensions, wrong layer order,
wrong final-hidden semantics, truncated files, or checksum mismatches.

Every schema-v2 stream is mandatory and validated, even when a particular
consumer ignores `target_final_hidden`. Schema v1 remains readable only through
an explicit legacy adapter for exploratory DFlash and DFlash2 runs. The aligned
three-method comparison pins one v2 manifest and identical sample IDs/checksums
for all consumers; it never mixes v1 and v2 data. There is no fake migration
from v1 to v2: reusing trajectories means rerunning Stage B for the same Stage-A
records. `aux_hidden_states` is persisted; the `hidden_states` flattened view is
derived by the reader and is not a fifth independent stream.

## Hidden extraction semantics

The SGLang bridge requests the five auxiliary decoder outputs and the final
post-norm output in one pass. Rank 0 writes both results after converting them
to CPU BF16; nonzero tensor-parallel ranks return empty host tensors as before.
The one-forward rule applies to production Stage B. A separate direct reference
forward is allowed only in the numerical validation gate.

Shape checks alone are insufficient. Before a full run, a real GLM-5.2 BF16
hardware gate compares several short sequences against a direct model hook:

- every auxiliary layer independently;
- the post-final-norm tensor;
- logits reconstructed from cached final hidden versus the target forward
  logits.

The comparison uses fixed token fixtures and reports three lower-is-better error
metrics: `cosine_error = 1 - cosine_similarity`, maximum absolute error, and mean
absolute error. A deterministic calibration runs each fixture three times
through the direct reference and capture paths. For each error metric, the
checked-in upper bound is the larger of an explicit numerical floor and twice
the worst observed direct-versus-direct variation, and must remain strictly
below the error produced by both negative controls. The floors, measured
variations, final upper bounds, model/runtime fingerprints, CANN, torch-npu, and
SGLang versions are stored in a versioned gate artifact. A run without the
matching artifact fails rather than recalibrating silently. Negative controls
deliberately substitute a pre-norm tensor and shift one layer ID; both must fail.

## Method consumers

### DFlash

Consumes `input_ids`, `loss_mask`, and the five auxiliary hidden layers. It uses
the common backbone and hard-token block CE. For physical block size `B` and
predicted depths `d=0..B-2`,
`y_d = input_ids[a+d+1]`, and

```text
w_d = valid_d * exp(-d / 7)
L_CE = sum_d w_d * -log q_d(y_d) / sum_d w_d
```

The global distributed numerator and denominator are reduced before division.
It does not read final hidden.

### DFlash2

Consumes the same streams as DFlash and uses the same common backbone. It adds
the two-tap dynamic convolution and path selector. Its base CE and selector CE
do not require final hidden. Base CE is identical to DFlash. At depth 0 the
selector predecessor is the clean anchor; at later depths it is the
teacher-forced previous target token. Selector CE is applied only when the
current target is present in the base top-16 candidate set, using the same
position weight and a separately reduced numerator/denominator. Its weight is
1.0. If the globally reduced selector denominator is zero, `L_selector` is a
differentiable zero and the term is skipped. The total objective is
`L = L_CE + 1.0 * L_selector`.

### DSpark

Consumes the same auxiliary hidden layers and additionally reads
`target_final_hidden`. For target token `y_d = input_ids[a+d+1]`, its aligned AR
teacher state is `target_final_hidden[a+d]`: the one-position shift is mandatory.
The frozen GLM-5.2 LM head reconstructs target logits in exact vocabulary chunks
to bound peak memory. The LM-head matmul uses the checkpoint BF16 dtype; draft
and target logits are cast to FP32 for full-vocabulary softmax. Target
probabilities are detached.

Let `q_d` and `p_d` be the draft and target FP32 softmax distributions. DSpark
uses the DeepSpec reference definitions:

```text
L_L1 = sum_d w_d * sum_v |q_d(v) - p_d(v)| / sum_d w_d
accept_target_d = clamp(1 - 0.5 * sum_v |q_d(v) - p_d(v)|, 0, 1)
L_conf = sum_d w_d * BCEWithLogits(c_d, stopgrad(accept_target_d)) / sum_d w_d
L = 0.1 * L_CE + 0.9 * L_L1 + 1.0 * L_conf
```

The Markov head is the official vanilla rank-256
`Embedding(V,256) -> Linear(256,V)` predecessor-token bias and applies before
`q_d` is computed; it does not gate on draft hidden. The confidence predictor
receives the concatenation of draft hidden and the same Markov predecessor
embedding. Predecessor IDs follow the same teacher-forced convention as the
DFlash2 selector. All numerators and denominators are reduced globally.
DSpark defaults to one epoch, learning rate `3e-4`, gamma 4, and physical B8.

## Block and mask contract

For physical block size `B` and anchor token index `a`:

```text
target: [input_ids[a], input_ids[a+1], ..., input_ids[a+B-1]]
noise:  [input_ids[a], MASK,           ..., MASK]
loss:   [ignore,       predict a+1,     ..., predict a+B-1]
```

An anchor is eligible iff `loss_mask[a] == 1`, `loss_mask[a+1] == 1`, and the
two indices exist. This includes the first token of an assistant turn when it
has a supervised successor, but excludes the prompt/tool token immediately
before that turn. For depth `d`, prediction validity is the cumulative product
of `loss_mask[a+1:a+d+2]` and the in-range mask. Consequently validity stops
permanently at the first turn boundary, hole, or sequence end; later tokens in
the same nominal block cannot become valid again.

Every draft query may attend to auxiliary context rows with absolute positions
`< a`, never row `a` or successor rows. The known anchor is supplied only by its
token embedding in local draft slot 0. Every draft query may attend to all `B`
local draft slots, whose contents are one anchor plus `B-1` masks. Raw future token
embeddings and auxiliary rows at positions `>= a` are forbidden.

RoPE always uses absolute sequence positions. Auxiliary row `t` uses position
ID `t`; local draft slot `j=0..B-1` uses position ID `a+j`. Block-relative RoPE is
forbidden. Training, exported configuration, and serving use this identical
convention.

All methods obtain identical anchors from a method-independent pure sampler
keyed only by `(global_seed, epoch, sample_id)`. It samples a uniform subset of
eligible indices without replacement. Model RNG use, data-parallel rank, and
method-specific forward calls cannot alter the selected anchors.

## Target I/O artifact

The small frozen target artifact continues to store GLM-5.2
`embed_tokens.weight` and `lm_head.weight`. Because the cached final hidden is
post-final-norm, no final-norm parameters are required by offline training.
The GLM path requires an LM head of shape `[vocab_size, 6144]` with no bias,
logit scaling, or soft-capping. If the inspected target violates any assumption,
the corresponding operation must be stored and reproduced rather than silently
ignored. The artifact manifest must match the cache's model and tokenizer
fingerprints, vocabulary size/order, hidden size, dtype, and weight checksum
before training starts.

## Export and runtime boundary

Exported DFlash, DFlash2, and DSpark configs must contain the common GLM-5.2
backbone fields and ordered target layer IDs. Method-specific config fields are
added without changing those fields. A serving checkpoint is not accepted based
on key shape alone: the exact Ascend SGLang/vLLM fork must load it and reproduce
one captured offline block's base logits and, where applicable, selector or
Markov outputs.

## Verification strategy

Implementation follows test-driven development:

1. configuration tests reject the old Qwen3.8-style 32/8/128 sliding setup and
   require the common GLM 64/64/64 full-attention setup;
2. cache tests prove schema-v2 round trips for both hidden streams, independent
   checksums, truncation detection, and schema-v1 compatibility boundaries;
3. extraction tests prove auxiliary and final outputs are separated and captured
   in one forward;
4. consumer tests prove DFlash/DFlash2 ignore final hidden and DSpark rejects a
   cache that lacks it;
5. numerical tests prove cached final hidden reconstructs the same logits as a
   direct frozen LM-head application;
6. hand-computed fixtures cover the one-token teacher shift, first assistant
   token, turn boundaries, holes, partial blocks, deterministic identical
   anchors, selector predecessors, Markov predecessors, and every loss term;
7. projection tests enforce Q/K/V width 4096, output width 6144, and explicit
   `head_dim=64` rather than deriving it from hidden size;
8. selector tests cover a globally empty top-16 eligibility set and require a
   finite differentiable zero selector term;
9. the existing complete CPU/distributed suite remains green under the pinned
   Transformers environment;
10. real 910B numerical and serving parity gates run before full data generation
   or training, including DFlash2 selector and DSpark Markov/confidence outputs.

## Non-goals

- Saving full-vocabulary target logits in the cache;
- saving every one of the 78 target layers;
- using the third-party RedHat GLM DSpark layer selection as the common main
  experiment setting;
- changing Stage A sampling policy or tool execution in this alignment change;
- claiming official DFlash2 or DSpark reproduction without final runtime and
  method-specific parity evidence.
