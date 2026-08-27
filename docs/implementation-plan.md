# Maintained implementation map

The data/cache design remains documented in
[`docs/superpowers/plans/2026-08-25-hidden-cache.md`](superpowers/plans/2026-08-25-hidden-cache.md)
and offline training in
[`docs/superpowers/plans/2026-08-25-offline-glm-dflash2-training.md`](superpowers/plans/2026-08-25-offline-glm-dflash2-training.md).
The current export/runtime implementation follows
[`docs/superpowers/plans/2026-08-27-vllm-ascend-export-runtime.md`](superpowers/plans/2026-08-27-vllm-ascend-export-runtime.md).

The only production sequence is:

1. `scripts/generate_trajectories.sh`: SGLang Stage A sampled trajectories and
   exact token IDs.
2. `scripts/extract_hidden_sglang.sh`: SGLang Stage B teacher-forced replay and
   schema-v2 hidden cache.
3. `scripts/extract_glm52_io.sh`: immutable target embedding/LM-head artifact.
4. `scripts/train_drafter.sh`: DFlash/DFlash2/DSpark offline training.
5. Method-specific candidate export under `src/glm_dflash2/vllm_ascend/`.
6. Real Ascend runtime parity and
   `tools/attest_vllm_ascend_export.py attest`.
7. `scripts/eval_vllm_ascend.sh`: serial target-only/speculative formal
   evaluation using only the exact attested runtime.

New exports are immutable `candidate-not-deployable` artifacts. DFlash,
DFlash2 and DSpark have distinct configs and weight contracts; DFlash2's
runtime integration is isolated in `integrations/vllm_ascend/`. A candidate is
deployable only after a machine-readable parity artifact binds its exact bytes
to one fully pinned vLLM/vLLM-Ascend/Speculators/CANN runtime and creates
`deploy_attestation.json`. Legacy schema-v1 exports are permanently untrusted.

The remaining hardware gates cannot be completed locally:

- Stage B varied-length and final-norm numerical parity on the production
  SGLang/GLM-5.2 fork;
- two-rank HCCL/FSDP2 optimizer/save/resume;
- method-specific candidate load, logits, proposal IDs and intermediate
  tensor parity in the exact vLLM-Ascend fork;
- raw-token greedy equality, standard sampling rejection and positive
  speculative counter deltas;
- serial same-hardware TPS/latency measurement under the deployment topology.

CPU smoke, matching key names, or a successfully written export must never be
reported as proof that those real-hardware gates passed.
