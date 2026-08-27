# Omni SD Ascend Data Pipeline Design

## Goal

Produce exact Qwen3-Omni Thinker sampling trajectories and teacher-forced target
hidden states on Atlas 800I A2/A3 without loading Talker or Code2Wav.

## Architecture

Stage A uses one vLLM-Ascend engine that owns tensor/expert parallelism. It
accepts text, image, audio, and video conditions and records the exact prompt
and sampled response token IDs returned by the engine. Sharding workers are
independent engines pinned to disjoint `ASCEND_RT_VISIBLE_DEVICES` sets; they
are not torchrun ranks inside one engine.

Stage B replays `prompt_token_ids + response_token_ids` through vLLM-Ascend's
native `extract_hidden_states` mode. The provider verifies that the connector's
token IDs exactly equal the requested sequence before committing a cache
sample. It preserves archival `[T, L, H]` auxiliary hidden states and a separate
`[T, H]` final normalized hidden state. Because upstream does not document the
final-normalized state ABI, support for that state is capability-gated: a run
fails closed unless the runtime returns it or an explicitly version-pinned
connector adapter is installed.

A Transformers provider remains available only for small parity tests. It uses
a device abstraction and never acts as the production multi-NPU extractor.

## Configuration

One YAML file defines model identity, sampling, vLLM-Ascend runtime, per-modality
batch sizes, target layers, hidden cache schema, and artifact paths. Validation
runs before model loading and rejects incomplete or internally inconsistent
configuration.

## Correctness gates

- Per-condition sampling seeds are independent of batch composition.
- Stage A records exact engine token IDs; decoded text is informational only.
- Stage B rejects any token mismatch.
- Hidden shapes, layer count, hidden width, dtype, and finite values are checked.
- Manifest fingerprints bind model/processor revisions, runtime versions,
  device topology, sampling policy, media identities, layer IDs, and shard
  layout.
- Text/image/audio/video smoke fixtures cover generation and replay contracts.
- Actual A2/A3 execution is represented by an explicit hardware attestation;
  CPU tests never claim hardware support.

## Non-goals

- Drafter architecture, training loss, or speculative serving.
- Talker or Code2Wav output generation.
- Hiding missing final-normalized hidden states by substituting a decoder-layer
  output.
