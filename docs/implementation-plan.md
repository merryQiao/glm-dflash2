# Implementation plan

The maintained plan is
[`docs/superpowers/plans/2026-08-25-hidden-cache.md`](superpowers/plans/2026-08-25-hidden-cache.md).

The offline training implementation plan is
[`docs/superpowers/plans/2026-08-25-offline-glm-dflash2-training.md`](superpowers/plans/2026-08-25-offline-glm-dflash2-training.md).

Implemented modules:

- `agent_trajectory.py`, `vibe_coding.py`, `web_tools.py`: copied behavioral
  baseline from the working SpecForge coding-agent generator;
- `trajectory_tokens.py`: exact replay token/mask freeze;
- `sglang_stage_a.py` and `tools/generate_trajectories.py`: GLM-5.2 rollout;
- `sglang_hidden_runner.py` and `tools/extract_hidden_sglang.py`: selected-layer
  SGLang internal teacher-forced pass;
- `hidden_cache.py`: packed durable cache and DFlash training reader;
- `tools/validate_hidden_cache.py`: integrity/shape/finite validation.
- `target_io.py`: selective frozen embedding/LM-head extraction and provenance;
- `dflash2_blocks.py`, `dflash2_objective.py`: exact anchors, masks and losses;
- `chunked_lm_head.py`: exact two-axis chunked full-vocabulary projection;
- `dflash2_model.py`: portable five-layer draft, dynamic conv and selector;
- `offline_trainer.py`: cache-to-loss training forward with unregistered I/O;
- `distributed.py`, `checkpointing.py`: HCCL/FSDP2 accumulation and exact resume;
- `tools/train_dflash2_offline.py`: offline training entrypoint and draft export.

Remaining hardware gates are a real GLM-5.2 layer-capture identity check, a
two-rank 910B optimizer/save/resume run, and tensor-by-tensor parity with the
actual SGLang-on-Ascend DFlash2 loader.  CPU tests cannot substitute for them.
