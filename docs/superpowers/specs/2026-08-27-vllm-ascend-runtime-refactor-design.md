# vLLM-Ascend-first runtime refactor

## Status

Approved direction: vLLM-Ascend is the only production serving and evaluation
backend. SGLang remains available for trajectory generation and as a fallback
hidden-state extractor.

## Motivation

The repository currently reflects its implementation history: SGLang-specific
names appear in the data pipeline, all three draft methods share one conservative
runtime compatibility marker, and the export layer cannot distinguish an upstream
GLM-5.2 DSpark path from DFlash or the custom DFlash2 path.

Recent upstream vLLM-Ascend adds an explicit GLM-5.2 DSpark end-to-end path and a
general hidden-state extraction mechanism. The repository should consume those
capabilities without coupling its data schema or offline trainer to a serving
framework.

## Goals

- Make vLLM-Ascend the sole production serving, acceptance, and TPS backend.
- Keep Stage A compatible with the existing SGLang OpenAI endpoint.
- Prefer vLLM-Ascend for Stage B hidden extraction when its exact GLM-5.2 layer
  contract passes numerical parity; retain the current SGLang hook as fallback.
- Keep cache schema v2, training objectives, checkpoint semantics, and existing
  generated data unchanged.
- Give DFlash, DFlash2, and DSpark separate export/runtime compatibility states.
- Fail closed when the installed runtime lacks a required method or tensor ABI.
- Remove duplicate or misleading launch paths after their replacements are tested.

## Non-goals

- Rewriting the three training objectives.
- Changing sampled trajectories, target layer IDs, anchor semantics, or block
  sizes.
- Claiming stock vLLM-Ascend support for GLM-5.2 DFlash or DFlash2 before a real
  Ascend parity gate passes.
- Supporting SGLang as a production speculative-decoding backend.
- Building a second cache format for vLLM.

## Architecture

The logical dependency direction is:

```text
OpenAI-compatible rollout endpoint
              |
              v
       framework-neutral data
              |
     +--------+---------+
     |                  |
vLLM hidden         SGLang hidden
(preferred)         (fallback)
     |                  |
     +--------+---------+
              v
        cache schema v2
              |
              v
 framework-neutral offline trainer
              |
      +-------+--------+
      |       |        |
   DFlash  DFlash2  DSpark
      |       |        |
      v       v        v
 method-specific vLLM-Ascend export
              |
              v
 vLLM-Ascend serving and benchmark only
```

### Package boundaries

The refactor uses focused subpackages while preserving small compatibility imports
during migration:

```text
src/glm_dflash2/
  core/
    cache_schema.py       cache-v2 validation and common records
    trajectory.py         sampled token path and loss-mask semantics
    target_io.py          frozen target embedding/head artifacts
    provenance.py         model, tokenizer, runtime and cache identity
  data/
    rollout_openai.py     framework-neutral Stage A client
    hidden_provider.py    HiddenProvider protocol and common result type
    hidden_vllm.py        preferred vLLM-Ascend extractor
    hidden_sglang.py      version-gated fallback extractor
  training/
    trainer.py            common FSDP/checkpoint/accumulation loop
    dflash.py             DFlash model and objective assembly
    dflash2.py            DFlash2 model and objective assembly
    dspark.py             DSpark model and objective assembly
  runtime/vllm_ascend/
    capabilities.py       installed-runtime capability probe
    export_common.py      hashes, target I/O, common metadata
    export_dflash.py      DFlash-specific config/key adapter
    export_dflash2.py     custom DFlash2 config/key adapter
    export_dspark.py      upstream GLM-5.2 DSpark-compatible adapter
    benchmark.py          lossless acceptance and TPS comparison
```

This is a responsibility split, not a requirement to move every class immediately.
Migration should be incremental and keep public CLI behavior stable until tests are
green.

## Data and hidden-state providers

### Stage A

Stage A speaks only the OpenAI-compatible chat API and records exact sampled token
IDs. The endpoint manifest records its implementation (`sglang`, `vllm`, or
unknown), but downstream code does not branch on that value. The production
launcher continues to target the existing SGLang deployment by default.

### Stage B

Both hidden providers implement one contract:

```python
class HiddenProvider(Protocol):
    def capabilities(self) -> HiddenCapabilities: ...
    def extract(self, sample: FrozenTrajectory) -> HiddenSample: ...
```

`HiddenSample` contains exact input token IDs, selected hidden streams, final hidden
states, logits or logit witnesses, layer semantics, dtype, and runtime identity.
It is serialized through the existing cache-v2 writer.

Provider selection is explicit:

- `--hidden-provider vllm` is the preferred production choice.
- `--hidden-provider sglang` selects the fallback.
- `--hidden-provider auto` may select vLLM only after its capability and numerical
  parity attestation exists; otherwise it selects SGLang and records the reason.

Shape agreement alone is insufficient. A provider must pass direct-forward token,
layer, hidden, and logits parity on the actual GLM-5.2 BF16 Ascend environment.

## Runtime capability contract

Hard-coded compatibility strings are replaced by a generated capability report:

```json
{
  "schema": "glm52-runtime-capabilities-v1",
  "runtime": "vllm-ascend",
  "runtime_revision": "<version-or-commit>",
  "target_architecture": "GLM-5.2",
  "methods": {
    "dspark": {
      "status": "upstream-path-available",
      "load_parity": false,
      "logit_parity": false,
      "acceptance_smoke": false
    },
    "dflash": {
      "status": "glm52-parity-required",
      "load_parity": false,
      "logit_parity": false,
      "acceptance_smoke": false
    },
    "dflash2": {
      "status": "custom-adapter-required",
      "load_parity": false,
      "logit_parity": false,
      "acceptance_smoke": false
    }
  }
}
```

An export is deployable only when the selected method has all three parity gates.
The repository may recognize an upstream path without claiming that the local
runtime version contains it.

## Method-specific export

- **DSpark:** match the public GLM-5.2 speculator configuration, Markov head,
  confidence head, weight keys, and proposal count. Do not inherit DFlash export
  assumptions when the upstream DSpark ABI differs.
- **DFlash:** use the stock vLLM/speculators DFlash schema where possible. Add only
  the minimum GLM-5.2 adapter needed by the installed vLLM-Ascend version.
- **DFlash2:** retain a distinct model type, selector tensors, and explicit custom
  proposer requirement. Never label it as ordinary DFlash to bypass capability
  checks.

Every export includes target/tokenizer fingerprints, block size, proposal count,
anchor policy, checksums, training method, and the capability report used at export
time.

## Serving and evaluation

Only vLLM-Ascend is accepted for formal evaluation. The launcher:

1. probes the runtime and validates the selected export;
2. starts target-only and speculative servers serially on the same devices;
3. runs identical prompts and sampling parameters;
4. requires exact output equality for greedy lossless evaluation;
5. reads vLLM speculative counters for acceptance length;
6. reports TPS, speedup, request latency, acceptance, versions, and topology.

SGLang serving commands are removed from formal-evaluation documentation. SGLang
remains documented only under data generation and fallback hidden extraction.

## Error handling

- Missing runtime capability: stop before loading a large model.
- Version mismatch: print the required method and detected vLLM/vLLM-Ascend
  revisions.
- Hidden parity failure: reject the provider; never silently fall back after cache
  generation has started.
- Export ABI mismatch: fail with missing/unexpected key and expected tensor shape.
- Inactive speculative metrics: fail the benchmark rather than report target-only
  TPS as speculative TPS.
- Output mismatch in greedy mode: mark the run invalid even if TPS improves.

## Migration and cleanup

1. Introduce interfaces and capability records without moving training code.
2. Make existing SGLang Stage A and Stage B implementations satisfy the new
   provider contracts.
3. Add the vLLM hidden provider and parity gate.
4. Split export by method and align DSpark first.
5. switch formal evaluation to the method-aware vLLM-Ascend gate.
6. Update launchers and documentation.
7. Delete compatibility wrappers and old imports only after call-site and CLI tests
   prove they are unused.

Existing schema-v2 caches and checkpoints must remain readable throughout.

## Verification

### CPU tests

- provider contract and deterministic selection;
- identical cache-v2 output from synthetic vLLM and SGLang providers;
- capability parsing and fail-closed states;
- separate DFlash, DFlash2, and DSpark export round trips;
- legacy cache/checkpoint compatibility;
- launcher tests proving SGLang cannot be selected for formal evaluation.

### Ascend gates

- GLM-5.2 BF16 target-only vLLM-Ascend smoke;
- selected-layer hidden and logits parity for vLLM extraction;
- SGLang fallback parity against the same frozen sample;
- DSpark load, logits, acceptance, and lossless greedy output parity;
- corresponding DFlash and DFlash2 gates as their adapters become available;
- serial baseline/speculative TPS benchmark on identical hardware.

## Completion criteria

- One framework-neutral cache and trainer path serves all three methods.
- vLLM-Ascend is the only formal serving/evaluation backend.
- SGLang is confined to Stage A and fallback Stage B modules.
- Each method has a truthful, independently tested runtime status.
- Existing caches and checkpoints still load.
- CPU tests pass and the server handoff document clearly separates locally proven
  behavior from pending Ascend validation.
