# GLM-5.2 on Ascend 910B: compatibility audit

Audit date: 2026-08-25.

## Result

The two-pass pipeline matches the official SGLang Ascend deployment path at
the service and topology levels. Stage A is low risk, especially when pointed
at an already-running official SGLang endpoint. Stage B is source-compatible
with the current GLM-5.2/DeepSeek hidden-capture path, but it relies on a
private SGLang `ModelRunner` API and therefore still requires a two-sample
hardware gate before a full cache build.

## Verified statically

- Official SGLang documents GLM-5.2 serving on Ascend with
  `--device npu --attention-backend ascend`.
- The launchers now pass those flags explicitly instead of relying on device
  auto-detection.
- TP, EP, multi-node rank and rendezvous settings are exposed by Stage B.
- The production recipe uses the GLM-5.2 BF16 checkpoint; optional load-format
  and DeepEP settings are passed through without hard-coding them.
- SGLang's `GlmMoeDsaForCausalLM` exposes the inherited Eagle3/DFlash capture
  method.
  Logical capture layers `[1,20,38,56,75]` map to physical layer inputs
  `[2,21,39,57,76]`, all within GLM-5.2's 78-layer backbone.
- Captured outputs are copied to CPU BF16 and packed as `[T,5,6144]`.
- Both observed `ForwardBatch.init_new` API variants are supported.

## Deployment topology

The official guide reports approximately 1.51 TB of BF16 weights. For 64 GB
910B cards, BF16 deployment needs at least 32 cards across two 16-card nodes.

KV cache, temporary activations and captured hidden states need additional
headroom. Stage A should normally reuse an external server. Stage B must load
the model separately after Stage A has finished.

## Required 910B gate

1. Run Stage A for one or two short examples and freeze the partial output.
2. Run Stage B on those exact token IDs.
3. Check finite BF16 output with exact shape `[T,5,6144]`.
4. On the same short input, compare each of the five slices against a direct
   model hook. Shape-only success is insufficient.
5. Record SGLang, torch-npu, CANN and model revision in the experiment log.

Only after this gate should the full hidden cache be built.

## Residual risks

1. **Internal API drift:** official serving can work while Stage B's private
   runner changes. This should be a small adapter change, not a redesign.
2. **Long-sequence memory:** exact hidden extraction disables chunked prefill.
   A 131K trajectory may exceed memory even when weights fit.
3. **Storage:** five BF16 6144-wide layers cost 61,440 bytes per token; the
   full cache can reach tens of terabytes.

## Official references

- SGLang Ascend GLM-5.2 deployment tutorial:
  <https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2.mdx>
- SGLang Ascend installation:
  <https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx>
- SGLang Ascend supported-model matrix:
  <https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/reference/support_models.mdx>
- GLM-5.2 model configuration:
  <https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json>
