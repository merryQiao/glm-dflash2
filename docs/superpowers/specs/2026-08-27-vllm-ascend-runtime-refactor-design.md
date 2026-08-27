# GLM-5.2 vLLM-Ascend export and runtime design

## Status and scope

This design deliberately leaves the proven data and training path in place:

1. SGLang Stage A generates sampled trajectories and freezes the exact prompt and
   response token IDs.
2. SGLang Stage B replays those immutable token paths and captures selected target
   hidden states plus the post-final-norm LM-head input.
3. The framework-neutral offline trainer consumes cache schema v2 and trains
   DFlash, DFlash2, or DSpark.

The refactor starts only after training. It splits method-specific export, binds
each candidate to the actual vLLM-Ascend runtime through a deploy attestation, and
uses one lossless evaluation entry point. It does not move hidden extraction to
vLLM, rewrite objectives, or introduce a second cache format.

## Goals

- Export DFlash, DFlash2, and DSpark with separate configs, state-key mappings,
  runtime compatibility, and parity gates.
- Align DSpark and DFlash to their real vLLM/vLLM-Ascend proposer ABIs.
- Keep DFlash2 distinct and provide a small version-pinned vLLM-Ascend adapter.
- Make vLLM-Ascend the only formal acceptance, TPS, and latency backend.
- Fail closed if the drafter is not active, output is not lossless, a checksum or
  environment identity differs, or speculative counters cannot be isolated.
- Preserve current caches, target I/O artifacts, checkpoints, and launchers until
  replacements pass regression tests.

## Non-goals

- Changing sampled trajectories, target layer IDs, Stage A/B sampling, or tool
  rollout behavior.
- Moving bulk hidden extraction to vLLM-Ascend.
- Changing cache schema v2 or any training loss.
- Pretending DFlash2 is ordinary DFlash.
- Claiming production compatibility before a real GLM-5.2 BF16 Ascend gate.

## Final structure

```text
src/glm_dflash2/
  sglang_stage_a.py
  sglang_hidden_runner.py
  hidden_cache.py
  offline_trainer.py
  draft_backbone.py
  dflash2_model.py
  dspark_model.py
  vllm_ascend/
    __init__.py
    export_common.py
    export_dflash.py
    export_dflash2.py
    export_dspark.py
    capability.py
    parity.py

integrations/vllm_ascend/
  dflash2_proposer.py
  dflash2_model_loader.py
  apply_patch.sh
  VERSION

scripts/
  generate_trajectories.sh
  extract_hidden_sglang.sh
  train_drafter.sh
  eval_vllm_ascend.sh
```

Compatibility wrappers may temporarily retain existing CLI names. They must call
the new modules and are deleted only after call-site tests prove them unused.

## Frozen upstream contract

### Stage A

Production trajectories require raw server-returned `prompt_token_ids` and
`response_token_ids` for every assistant turn. Re-tokenizing response text is not
an allowed fallback. The frozen manifest binds the model and tokenizer
fingerprints, sampling parameters, chat-template arguments, endpoint identity,
JSONL checksum, sample-ID checksum, and completion status.

### Stage B

Stage B consumes only a frozen Stage A artifact and teacher-forces its exact token
path. The supported production provider remains the existing SGLang internal
runner. It must:

- capture logical layers `[1,20,38,56,75]` with explicit physical tap metadata;
- capture `model.norm` output as `post_final_norm_lm_head_input` rather than using
  a pre-norm last-layer tensor;
- disable chunked prefill, prefix/radix cache, CUDA/NPU graphs, and concurrent
  requests during extraction;
- pin and record Model Runner generation, SGLang commit, CANN, torch-npu,
  dtype, target/tokenizer fingerprints, TP/EP/PP/DP, node count, device model, and
  layer mapping;
- reject unsupported runner versions before allocating the target;
- pass direct-forward parity for selected layers, final hidden, LM-head logits,
  and top-1 on the actual multi-node 910B topology.

The vLLM hidden extractor proposed in the previous design is removed. A future
provider is a separate project and may not write cache v2 until it proves the same
post-final-norm and topology contract.

## Common export contract

`export_common.py` owns only data shared by all methods:

- frozen target embedding and LM head identities;
- target and tokenizer fingerprints;
- ordered auxiliary layer IDs `[1,20,38,56,75]`;
- physical block size and proposal count;
- anchor policy;
- training checkpoint identity;
- canonical config and tensor checksums.

The common layer does not rename method-specific tensors. Each method exporter
must reject missing, unexpected, or shape-incompatible state keys.

Every export is first written as an immutable **candidate export** with status
`candidate-not-deployable`. Creating files successfully never implies runtime
support.

## Method-specific export

### DSpark

DSpark matches the vLLM runtime file, config, key, and forward ABI while retaining
this project's target layer selection:

```text
architecture: DSparkSpeculatorConfig
backbone: five-layer Qwen3-shaped draft backbone
aux_hidden_state_layer_ids: [1,20,38,56,75]
block_size: 8
num_speculative_tokens: 7
sample_from_anchor: false
markov_rank: 256
markov_head_type: vanilla
confidence_head: enabled
```

Parity uses a fixed anchor and compares every depth's predecessor ID, offline and
runtime logits, confidence values, proposal IDs, and seven-token proposal count.

### DFlash

DFlash exports the stock vLLM/speculators DFlash schema wherever the installed
vLLM-Ascend fork supports it:

```text
method: dflash
aux_hidden_state_layer_ids: [1,20,38,56,75]
block_size: 8 or 16
num_speculative_tokens: 7 or 15
```

The capability gate must prove that GLM-5.2 returns the requested five auxiliary
layers to the proposer. No target patch is added if the installed proposer already
drives target hidden selection from the draft config.

### DFlash2

DFlash2 retains its dynamic convolution, base top-k candidates, selector, and
candidate reranking. Its export uses a distinct architecture/config and cannot be
labelled DFlash. It supports physical block sizes 8 and 16 with respectively 7
and 15 speculative tokens, matching the corresponding training checkpoint.

The repository carries a small, version-pinned vLLM-Ascend integration that only:

1. loads the DFlash2 export;
2. receives target auxiliary hidden states;
3. produces base top-k candidates;
4. applies selector reranking;
5. returns draft token IDs through the proposer ABI.

Training and cache code never imports this integration.

## Candidate, parity, and deploy attestation

Capability is bound to a concrete candidate, not represented by reusable boolean
flags. The lifecycle is:

```text
training checkpoint
  -> candidate export (not deployable)
  -> load/logit/token/acceptance tests on installed runtime
  -> deploy attestation
  -> immutable deployable export
```

The attestation binds at least:

- export config and weights SHA-256;
- training checkpoint and target I/O SHA-256;
- target/model/tokenizer fingerprints;
- method, block size, proposal count, anchor policy, and method parameters;
- vLLM, vLLM-Ascend, adapter, CANN, torch-npu, driver, and firmware versions;
- TP/EP/PP/DP, nodes, device type, model-runner generation, attention backend,
  chunked-prefill and prefix-cache settings;
- fixture token-ID checksum and numerical thresholds;
- load, logits, proposal IDs, greedy token IDs, speculation-active, rejection
  sampling, and acceptance-counter results.

Unknown or missing fields, a non-passing status, a checksum mismatch, or a current
runtime identity mismatch causes startup failure. The attestation is created only
after testing the candidate, eliminating the previous export/test circularity.

## Formal evaluation

`scripts/eval_vllm_ascend.sh` is the only formal entry point:

```bash
METHOD=dspark \
TARGET_MODEL=/path/to/GLM-5.2-BF16 \
DRAFTER_EXPORT=/path/to/deployable-export \
PROMPTS=/path/to/eval.jsonl \
bash scripts/eval_vllm_ascend.sh
```

It performs serial measurements on the same allocated devices:

1. validate candidate checksums and deploy attestation against the current host;
2. start the target-only server and run warm-up;
3. snapshot counters, evaluate the fixed prompts, and record baseline metrics;
4. shut down the baseline server completely;
5. start target plus the selected proposer and run the same warm-up;
6. snapshot speculative counters, evaluate identical prompts and parameters, and
   record only counter deltas from the measured window;
7. require proposal/draft/accepted counters to exist and require proposals and
   draft tokens to be positive;
8. compute acceptance length, TPS, speedup, and request latency from the identical
   measured request set.

Greedy evaluation requires byte-identical output token IDs between baseline and
speculative runs. Sampling evaluation uses the runtime's standard rejection
sampling and validates its configured lossless mode; sampled outputs are not
required to be identical across two independent random streams.

Warm-up, health checks, and server startup are excluded from counters and TPS.

## Error handling

- Missing exact Stage A token IDs: stop before committing a trajectory.
- Unfrozen or identity-mismatched Stage A artifact: refuse Stage B.
- Pre-norm or unknown final-hidden semantics: refuse cache creation.
- Unsupported Model Runner/topology or extraction feature: refuse Stage B before
  model allocation.
- Export key/shape mismatch: fail with the exact missing/unexpected tensor.
- Missing or stale deploy attestation: refuse server startup.
- Drafter loaded but inactive counters: mark evaluation invalid.
- Greedy token mismatch: mark evaluation invalid even if TPS improves.

## Verification

### CPU tests

- Stage A exact sampled-ID capability and frozen-manifest resume stability;
- Stage B post-final-norm semantic enforcement and forbidden-runtime settings;
- separate DFlash, DFlash2, and DSpark export round trips and negative key tests;
- DSpark anchor/predecessor/confidence/proposal ABI;
- DFlash B8/B16 proposal count and auxiliary-layer contract;
- DFlash2 selector/base-candidate runtime adapter parity on deterministic tensors;
- candidate export cannot launch without a bound passing attestation;
- attestation fails after changing one export byte or one runtime/topology field;
- evaluation uses counter deltas, detects inactive speculation, and enforces greedy
  token-ID parity.

### Ascend gates

- SGLang Stage B direct-forward parity on the actual GLM-5.2 BF16 multi-node 910B
  topology, including varied sequence lengths and concurrent/resume fixtures;
- each method's candidate loads in the pinned vLLM-Ascend fork;
- offline/runtime logits and proposal token IDs match for all proposal depths;
- greedy target-only/speculative outputs are identical;
- sampling uses standard rejection sampling;
- baseline/speculative measurements run serially on identical hardware and report
  positive speculative counters.

## Completion criteria

- Existing Stage A data, cache v2, target I/O, and training checkpoints remain
  readable and unchanged.
- Three exporters contain no shared method-specific assumptions.
- DFlash2 runtime logic is isolated under `integrations/vllm_ascend`.
- A candidate cannot become deployable without a bound real-runtime attestation.
- `eval_vllm_ascend.sh` is the sole documented formal benchmark path.
- CPU tests pass; remaining 910B gates are stated truthfully and fail closed until
  executed on the target server.
