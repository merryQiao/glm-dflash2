# Server AI Handoff: Qwen3-Omni Stage B + Stage C

## Objective

Use completed Qwen3-Omni Stage A trajectories to create a deterministic v3
Thinker hidden cache on Ascend, then train DFlash/DFlash2/DSpark drafters
offline without loading the 30B target.

The authoritative design and commands are in
`docs/STAGE_B_STAGE_C_GUIDE.md`. Do not infer missing behavior from the GLM
pipeline; Qwen3-Omni requires three-axis mRoPE and 32Q/4KV GQA.

## Hard invariants

- Never re-tokenize saved responses; replay exact Stage A token IDs.
- Never accept `omni-thinker-hidden-cache-v2`; Stage C requires v3.
- Never synthesize flat 1-D positions; cache exact `[T,3]` mRoPE IDs.
- Never substitute raw layer-48 output for post-final-RMSNorm hidden.
- Never mix cache/model/processor revisions or incomplete shards.
- Never inject the DFlash2 target token into top-16 candidates.
- Never load Talker or Code2Wav for this task.
- Never claim Ascend production readiness from CPU smoke alone.

## First commands on the destination server

```bash
cd /path/to/this/branch
pip install -r requirements.txt
PYTHONPATH=$PWD/src:$PWD python -m pytest -q
bash scripts/smoke_stage_c_training.sh
```

Then edit `configs/generate_thinker_data.yaml`, run Stage B on a very small
completed Stage A shard, inspect `manifest.json`, extract target I/O, and run
one NPU optimizer step for each of the five routes before scaling up.

## Required report

For every real run, record:

- container/image digest, CANN, torch-npu, vLLM, vLLM-Ascend versions;
- hardware A2/A3, visible devices, TP/world size;
- model and processor immutable revisions;
- Stage A generation fingerprint and Stage B cache fingerprint;
- exact command, output directory, peak HBM, throughput, and failure log;
- one cache tensor shape/dtype audit;
- all five Stage C one-step results and one save/resume result.

If the native hidden connector or requested layer ABI differs in the target
image, stop and report the observed tensor keys/shapes. Do not silently patch
the cache into a different semantic contract.
