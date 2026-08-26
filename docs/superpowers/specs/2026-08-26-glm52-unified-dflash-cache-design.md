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
- target auxiliary layers: `[1, 20, 38, 56, 75]` in this exact order;
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
- block size: 16, represented by one known anchor and 15 predicted tokens.

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
`[1, 20, 38, 56, 75][i]` at the same token position as `input_ids`.

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
- physical capture IDs used by the runtime;
- target decoder depth;
- auxiliary and final hidden sizes and dtypes;
- `final_hidden_semantics = "post_final_norm_lm_head_input"`;
- target model revision or local fingerprint;
- tokenizer and chat-template fingerprints;
- generation sampling parameters.

Readers fail closed on missing streams, wrong dimensions, wrong layer order,
wrong final-hidden semantics, truncated files, or checksum mismatches.

Schema v1 remains readable for DFlash and DFlash2. DSpark must reject schema v1
because it cannot reconstruct the target distribution without final hidden
states. There is no fake migration from v1 to v2; existing trajectories may be
reused, but Stage B must be rerun to produce the missing final-hidden stream.

## Hidden extraction semantics

The SGLang bridge requests the five auxiliary decoder outputs and the final
post-norm output in one pass. Rank 0 writes both results after converting them
to CPU BF16; nonzero tensor-parallel ranks return empty host tensors as before.

Shape checks alone are insufficient. Before a full run, a real GLM-5.2 BF16
hardware gate compares several short sequences against a direct model hook:

- every auxiliary layer independently;
- the post-final-norm tensor;
- logits reconstructed from cached final hidden versus the target forward
  logits.

The comparison reports cosine similarity, maximum absolute error, and mean
absolute error and must fail on wrong layer ordering or pre-norm capture.

## Method consumers

### DFlash

Consumes `input_ids`, `loss_mask`, and the five auxiliary hidden layers. It uses
the common backbone and hard-token block CE. It does not read final hidden.

### DFlash2

Consumes the same streams as DFlash and uses the same common backbone. It adds
the two-tap dynamic convolution and path selector. Its base CE and selector CE
do not require final hidden.

### DSpark

Consumes the same auxiliary hidden layers and additionally reads
`target_final_hidden`. The frozen GLM-5.2 LM head reconstructs target logits in
chunks to bound peak memory. The method adds its Markov head and confidence
head and uses the configured CE, distribution-distance, and confidence losses.
The reference starting point is `0.1 * CE + 0.9 * TV/L1` with confidence-head
weight 1.0; these coefficients remain explicit experiment configuration rather
than hidden cache metadata.

## Block and mask contract

For block size 16:

```text
target: [anchor, token_1, ..., token_15]
noise:  [anchor, MASK,    ..., MASK]
loss:   [ignore, predict,  ..., predict]
```

The context may attend only to positions strictly before the anchor. Draft
positions use full non-causal attention within the block. Partial assistant
spans near the end of a sequence remain valid when at least one successor is
supervised; the per-position prediction mask excludes invalid suffix tokens.
All three methods use the same sampled anchors and validity masks.

## Target I/O artifact

The small frozen target artifact continues to store GLM-5.2
`embed_tokens.weight` and `lm_head.weight`. Because the cached final hidden is
post-final-norm, no final-norm parameters are required by offline training.
The artifact manifest must match the hidden-cache target fingerprint and hidden
size before training starts.

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
6. the existing complete CPU/distributed suite remains green under the pinned
   Transformers environment;
7. real 910B numerical and serving parity gates run before full data generation
   or training.

## Non-goals

- Saving full-vocabulary target logits in the cache;
- saving every one of the 78 target layers;
- using the third-party RedHat GLM DSpark layer selection as the common main
  experiment setting;
- changing Stage A sampling policy or tool execution in this alignment change;
- claiming official DFlash2 or DSpark reproduction without final runtime and
  method-specific parity evidence.
