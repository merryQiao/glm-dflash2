# Design

The authoritative design is
[`docs/superpowers/specs/2026-08-25-sglang-two-pass-data-design.md`](superpowers/specs/2026-08-25-sglang-two-pass-data-design.md).

The offline trainer design is
[`docs/superpowers/specs/2026-08-25-offline-glm-dflash2-training-design.md`](superpowers/specs/2026-08-25-offline-glm-dflash2-training-design.md).

Operational invariants:

1. Stage A runs real agent/tool rollouts, while the two Open-SWE routes restore
   their original full trajectories exactly as the reference generator does.
2. Stage B never re-tokenizes; it teacher-forces those exact IDs.
3. Only target layers `[1,20,38,56,75]` are archived.
4. DFlash masks use target-token positions, not shifted AR positions.
5. Stream fsync precedes index fsync; the index entry commits a sample.
6. The rollout server exits before the hidden extractor loads GLM-5.2.
7. A real Ascend per-layer identity check is required before scaling.
8. Offline training never loads the 753B backbone; frozen target token I/O is
   held outside the FSDP2 module/optimizer tree.
9. Anchor/label alignment is unshifted and every complete block begins with one
   clean anchor followed by 15 mask tokens.
10. Full-vocabulary CE and top-16 are exact despite token/vocabulary chunking.
11. The exported draft uses 78 as the GLM target depth and logical target-layer
    IDs `[1,20,38,56,75]`; Stage B's physical `+1` hook IDs are not reapplied.
12. Real two-rank HCCL/FSDP2 and SGLang runtime parity gates are mandatory.
