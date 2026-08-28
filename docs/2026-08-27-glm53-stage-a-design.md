# GLM-5.3-Flash Stage A Design

## Goal

Build a standalone GLM-5.3-Flash Stage A trajectory generator under
`glm-dflash2/glm-5.3-flash` without modifying or importing the parent GLM-5.2
implementation. The first version covers sampled coding-agent trajectory
generation only; hidden extraction, drafter training, export, and evaluation
remain out of scope.

## Architecture

The implementation copies only the transitive dependencies of the existing
Stage A path into a new `glm53_stage_a` package. A command-line driver loads the
vibe-coding dataset, materializes optional workspaces, calls an OpenAI-compatible
SGLang endpoint, executes bounded tools, and commits frozen trajectory records.
It may either launch a local SGLang server or reuse an externally managed
endpoint whose immutable identity is attested by a manifest.

The output remains schema-compatible with the existing Stage B reader:
assistant messages, tool calls and observations, exact prompt token IDs, exact
sampled response token IDs, flattened token IDs, and the assistant loss mask are
preserved. Stage A fails before rollout if the endpoint cannot return exact
token IDs.

## GLM-5.3-Flash Adaptation

- Default model identity: `GLM-5.3-Flash-BF16`.
- Sampling defaults remain `temperature=1.0`, `top_p=0.95`, `top_k=-1`.
- Reasoning and tool parsers default to `glm45` and `glm47`, but are explicit
  command-line/server-profile settings rather than hidden constants.
- Model metadata supports multimodal nested configuration, including
  `text_config.vocab_size`.
- Model and tokenizer identity includes `chat_template.jinja` and processor
  configuration files so a template update invalidates resume and endpoint
  manifests.
- The old SGLang 0.5.16 token-ID patch is not copied. The runtime is accepted
  only after a model-availability probe and an exact-token-ID capability probe.
- The initial path is text-only vibe coding. Image/video request construction is
  deliberately out of scope even though GLM-5.3-Flash is multimodal.

## Files

```text
glm-5.3-flash/
├── README.md
├── requirements.txt
├── src/glm53_stage_a/
│   ├── __init__.py
│   ├── agent_trajectory.py
│   ├── jsonl.py
│   ├── open_swe_trajectories.py
│   ├── provenance.py
│   ├── sglang_stage_a.py
│   ├── trajectory_tokens.py
│   ├── vibe_coding.py
│   ├── web_tools.py
│   └── workspaces.py
├── tools/generate_trajectories.py
├── tools/prepare_open_swe_trajectories.py
├── scripts/run_stage_a_trajectories.sh
└── tests/
```

No source file outside `glm-5.3-flash/` will be edited.

## Failure Handling and Identity

- A local model directory must contain model configuration, tokenizer/chat
  template artifacts, and weight files.
- External endpoints require an identity manifest except for an explicitly
  bounded smoke run. `--allow-unverified-endpoint` requires an explicit
  `--max-samples` between 1 and 50. Its manifest records
  `production_eligible=false` and must never transition to `status="frozen"`;
  successful completion uses `status="smoke_unverified"`. The resulting records
  are diagnostic artifacts and cannot pass the existing Stage B frozen-manifest
  gate.
- Resume compares the complete run contract and refuses changes to the model,
  tokenizer/template, dataset, sampling, sharding, workspace, or runtime
  identity.
- JSONL commits are flush-and-fsync boundaries; a truncated final record is
  repaired on resume.
- Errors are recorded separately and resolved only after a trajectory record is
  durably committed.
- Rows routed through existing Open-SWE trajectories require a prebuilt SQLite
  store. The standalone copy includes the preparation tool and fails before
  generation when those rows are selected but the store is absent.
- `model_revision` and tokenizer/chat-template fingerprint helpers live in the
  standalone `provenance.py`; the implementation must not import the parent's
  training-oriented `target_io.py`.

## Testing

Tests are written before each behavior change and cover:

1. GLM-5.3 server command construction and configurable parsers.
2. Nested `text_config.vocab_size` extraction.
3. Fingerprint changes when `chat_template.jinja` or processor metadata changes.
4. Exact prompt/response token-ID requests and capability failure.
5. External endpoint identity validation.
6. Tool ordering, trajectory freezing, loss masks, sharding, JSONL recovery,
   workspace routing, and resume-contract rejection.
7. Shell syntax and CLI `--help` smoke tests.
8. Import-isolation: all `glm53_stage_a` modules resolve below the standalone
   directory and no import resolves to the parent `glm_dflash2` package.

No unit test is treated as proof that the vendor Ascend SGLang image supports
`glm5_next`. The final handoff documents a required 10–50 sample 910B smoke gate.
