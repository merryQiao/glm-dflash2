from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .capability import normalize_runtime_identity, runtime_identities_match
from .export_common import (
    ATTESTATION_FILENAME,
    CANDIDATE_STATUS,
    load_candidate_export,
    sha256_file,
)


ATTESTATION_SCHEMA = "glm-vllm-ascend-deploy-attestation-v1"
PARITY_RESULTS_SCHEMA = "glm-vllm-ascend-parity-results-v1"
REQUIRED_GATES = (
    "candidate_load",
    "logits",
    "proposals",
    "token_ids",
    "rejection_sampling",
    "speculative_counters",
)


def _canonical_sha(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def candidate_binding(output_dir: str | Path) -> dict[str, Any]:
    candidate = load_candidate_export(output_dir)
    manifest = candidate.manifest
    binding = {
        "schema": manifest["schema"],
        "method": manifest["method"],
        "config_sha256": manifest["config_sha256"],
        "weights_sha256": manifest["weights_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "target_io_sha256": manifest["target_io_sha256"],
        "target_model_fingerprint": manifest["target_model_fingerprint"],
        "target_model_revision": manifest["target_model_revision"],
        "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
        "aux_hidden_state_layer_ids": manifest["aux_hidden_state_layer_ids"],
        "block_size": manifest["block_size"],
        "num_speculative_tokens": manifest["num_speculative_tokens"],
        "sample_from_anchor": manifest["sample_from_anchor"],
        "method_parameters": manifest["method_parameters"],
        "runtime_adapter": manifest["runtime_adapter"],
    }
    return {**binding, "binding_sha256": _canonical_sha(binding)}


def _validate_parity_results(
    results: Mapping[str, Any], *, binding: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    if results.get("schema") != PARITY_RESULTS_SCHEMA:
        raise ValueError("unsupported vLLM-Ascend parity result schema")
    if results.get("candidate_binding") != binding:
        raise ValueError("parity results are not bound to this candidate")
    reported_runtime = normalize_runtime_identity(
        results.get("runtime_identity") or {}, production=True
    )
    matched, drift = runtime_identities_match(runtime, reported_runtime)
    if not matched:
        raise ValueError(f"parity result runtime differs from active runtime: {drift}")
    fixture_id = str(results.get("fixture_id", "")).strip()
    thresholds = results.get("thresholds")
    if not fixture_id or not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("parity results require fixture_id and numerical thresholds")
    gates = results.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("parity results are missing gates")
    for name in REQUIRED_GATES:
        gate = gates.get(name)
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            raise ValueError(f"parity gate {name} did not pass")
    if gates["rejection_sampling"].get("mode") != "standard":
        raise ValueError("parity rejection_sampling gate must prove standard mode")
    if float(gates["speculative_counters"].get("draft_tokens", 0)) <= 0:
        raise ValueError("parity speculative_counters gate reported no draft tokens")
    return json.loads(json.dumps(dict(results)))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def attest_candidate(
    output_dir: str | Path,
    *,
    runtime_identity: Mapping[str, Any],
    parity_results: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    candidate = load_candidate_export(output_dir)
    if candidate.manifest.get("status") != CANDIDATE_STATUS:
        raise ValueError("only an unattested candidate may receive a deploy attestation")
    runtime = normalize_runtime_identity(runtime_identity, production=True)
    binding = candidate_binding(output_dir)
    results = _validate_parity_results(parity_results, binding=binding, runtime=runtime)
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "candidate_binding": binding,
        "runtime_identity": runtime,
        "fixture_id": results["fixture_id"],
        "thresholds": results["thresholds"],
        "gates": results["gates"],
        "parity_results_sha256": _canonical_sha(results),
    }
    attestation_path = output_dir / ATTESTATION_FILENAME
    _atomic_json(attestation_path, attestation)
    manifest = dict(candidate.manifest)
    manifest["status"] = "runtime-attested"
    manifest["deploy_attestation_sha256"] = sha256_file(attestation_path)
    _atomic_json(output_dir / "export_manifest.json", manifest)
    return attestation


def validate_deploy_attestation(
    output_dir: str | Path, *, active_runtime: Mapping[str, Any]
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    candidate = load_candidate_export(output_dir)
    if candidate.manifest.get("status") != "runtime-attested":
        raise ValueError("candidate is not runtime-attested")
    path = output_dir / ATTESTATION_FILENAME
    if not path.is_file():
        raise ValueError("runtime-attested candidate is missing deploy attestation")
    if sha256_file(path) != candidate.manifest.get("deploy_attestation_sha256"):
        raise ValueError("deploy attestation checksum mismatch")
    attestation = json.loads(path.read_text(encoding="utf-8"))
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise ValueError("unsupported deploy attestation schema")
    binding = candidate_binding(output_dir)
    if attestation.get("candidate_binding") != binding:
        raise ValueError("deploy attestation candidate binding mismatch")
    runtime = normalize_runtime_identity(active_runtime, production=True)
    matched, drift = runtime_identities_match(attestation.get("runtime_identity") or {}, runtime)
    if not matched:
        raise ValueError(f"deploy attestation runtime mismatch: {drift}")
    return attestation
