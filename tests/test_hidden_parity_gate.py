from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from glm_dflash2.hidden_cache import HiddenCacheSpec, PackedHiddenWriter
from glm_dflash2.parity_attestation import validate_training_parity_attestation
from tools.calibrate_hidden_capture_gate import (
    calibrate_parity_gate,
    capture_error_metrics,
    validate_capture_with_gate,
)
from tools.validate_hidden_cache import main as validate_cache


class HiddenParityGateTest(unittest.TestCase):
    def setUp(self):
        self.reference = {
            "aux_hidden_states": torch.tensor(
                [[[1.0, 2.0], [3.0, 4.0]], [[2.0, -1.0], [0.5, 3.0]]]
            ),
            "target_final_hidden": torch.tensor([[1.0, -2.0], [4.0, 0.5]]),
            "target_logits": torch.tensor(
                [[0.2, 1.3, -0.4], [2.0, -1.0, 0.3]]
            ),
            "input_ids": torch.tensor([101, 102]),
            "layer_ids": torch.tensor([1, 20]),
        }
        self.identity = {
            "target_fingerprint": "glm52-bf16-sha",
            "model_revision": "revision-1",
            "tokenizer_fingerprint": "tokenizer-sha",
            "cann_version": "9.0.0",
            "torch_npu_version": "2.7.1",
            "sglang_version": "0.5.16",
        }

    def _offset(self, amount: float):
        return {
            **{
                key: self.reference[key] + amount
                for key in (
                    "aux_hidden_states",
                    "target_final_hidden",
                    "target_logits",
                )
            },
            "input_ids": self.reference["input_ids"].clone(),
            "layer_ids": self.reference["layer_ids"].clone(),
        }

    def test_calibration_binds_identical_fixture_tokens_and_logical_layers(self):
        direct = [self.reference, self._offset(0.001), self._offset(-0.002)]
        controls = {
            "shifted_layer": self._offset(0.5),
            "pre_norm": self._offset(-0.4),
        }
        artifact = calibrate_parity_gate(
            direct_runs=direct,
            negative_controls=controls,
            floors={
                "cosine_error": 1e-8,
                "max_abs_error": 1e-4,
                "mean_abs_error": 1e-4,
            },
            identity=self.identity,
        )
        self.assertEqual(artifact["fixture"]["layer_ids"], [1, 20])
        self.assertEqual(artifact["fixture"]["tokens"], 2)
        self.assertRegex(artifact["fixture"]["input_ids_sha256"], r"^[0-9a-f]{64}$")
        direct[1] = dict(direct[1], input_ids=torch.tensor([101, 999]))
        with self.assertRaisesRegex(ValueError, "fixture input_ids"):
            calibrate_parity_gate(
                direct_runs=direct,
                negative_controls=controls,
                floors={
                    "cosine_error": 1e-8,
                    "max_abs_error": 1e-4,
                    "mean_abs_error": 1e-4,
                },
                identity=self.identity,
            )

    def test_error_metrics_use_one_minus_cosine_and_absolute_errors(self):
        candidate = self._offset(0.01)
        metrics = capture_error_metrics(self.reference, candidate)
        keys = ("aux_hidden_states", "target_final_hidden", "target_logits")
        left = torch.cat([self.reference[key].float().reshape(-1) for key in keys])
        right = torch.cat([candidate[key].float().reshape(-1) for key in keys])
        expected_cosine = 1.0 - torch.nn.functional.cosine_similarity(
            left[None], right[None]
        ).item()
        self.assertAlmostEqual(metrics["cosine_error"], expected_cosine, places=9)
        self.assertAlmostEqual(metrics["max_abs_error"], 0.01, places=6)
        self.assertAlmostEqual(metrics["mean_abs_error"], 0.01, places=6)

    def test_three_run_calibration_persists_exact_bounds_and_negative_controls(self):
        direct = [self.reference, self._offset(0.001), self._offset(-0.002)]
        controls = {
            "shifted_layer": self._offset(0.5),
            "pre_norm": self._offset(-0.4),
        }
        floors = {
            "cosine_error": 1e-8,
            "max_abs_error": 1e-4,
            "mean_abs_error": 1e-4,
        }
        artifact = calibrate_parity_gate(
            direct_runs=direct,
            negative_controls=controls,
            floors=floors,
            identity=self.identity,
        )
        self.assertEqual(artifact["schema"], "glm-hidden-capture-parity-gate-v2")
        self.assertEqual(artifact["calibration_runs"], 3)
        self.assertEqual(artifact["identity"], self.identity)
        self.assertEqual(
            tuple(artifact["streams"]),
            (
                "aux_hidden_states.layer_0",
                "aux_hidden_states.layer_1",
                "target_final_hidden",
                "target_logits",
            ),
        )
        for stream in artifact["streams"].values():
            for values in stream["metrics"].values():
                self.assertEqual(
                    values["bound"],
                    max(values["floor"], 2.0 * values["worst_direct_variation"]),
                )
                self.assertLess(
                    values["bound"], values["negative_controls"]["shifted_layer"]
                )
                self.assertLess(
                    values["bound"], values["negative_controls"]["pre_norm"]
                )

        result = validate_capture_with_gate(
            reference=self.reference,
            candidate=self._offset(0.001),
            artifact=artifact,
            identity=self.identity,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["target_top1"]["passed"])

    def test_one_bad_aux_layer_fails_even_when_other_streams_match(self):
        artifact = calibrate_parity_gate(
            direct_runs=[self.reference, self.reference, self.reference],
            negative_controls={
                "shifted_layer": self._offset(0.5),
                "pre_norm": self._offset(-0.4),
            },
            floors={
                "cosine_error": 1e-9,
                "max_abs_error": 1e-6,
                "mean_abs_error": 1e-6,
            },
            identity=self.identity,
        )
        candidate = {key: value.clone() for key, value in self.reference.items()}
        candidate["aux_hidden_states"][:, 1].add_(0.1)
        result = validate_capture_with_gate(
            reference=self.reference,
            candidate=candidate,
            artifact=artifact,
            identity=self.identity,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(result["streams"]["aux_hidden_states.layer_0"]["passed"])
        self.assertFalse(result["streams"]["aux_hidden_states.layer_1"]["passed"])

    def test_gate_fails_closed_on_identity_mismatch_or_weak_negative_control(self):
        direct = [self.reference, self.reference, self.reference]
        controls = {
            "shifted_layer": self._offset(0.5),
            "pre_norm": self._offset(-0.4),
        }
        floors = {
            "cosine_error": 1e-9,
            "max_abs_error": 1e-6,
            "mean_abs_error": 1e-6,
        }
        artifact = calibrate_parity_gate(
            direct_runs=direct,
            negative_controls=controls,
            floors=floors,
            identity=self.identity,
        )
        wrong = dict(self.identity, target_fingerprint="other")
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_capture_with_gate(
                reference=self.reference,
                candidate=self.reference,
                artifact=artifact,
                identity=wrong,
            )
        with self.assertRaisesRegex(ValueError, "negative control"):
            calibrate_parity_gate(
                direct_runs=direct,
                negative_controls={
                    "shifted_layer": self.reference,
                    "pre_norm": controls["pre_norm"],
                },
                floors=floors,
                identity=self.identity,
            )

    def test_cache_validator_applies_frozen_gate_to_aux_and_final_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            mapping = tuple(
                ("test", layer, f"tap-{layer}", "post_decoder_block")
                for layer in (1, 20, 38, 56, 75)
            )
            spec = HiddenCacheSpec(
                layer_ids=(1, 20, 38, 56, 75),
                hidden_size=2,
                capture_mapping=mapping,
            )
            aux = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2).to(
                torch.bfloat16
            )
            final = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
            with PackedHiddenWriter(cache, spec=spec) as writer:
                writer.append(
                    sample_id="gate-sample",
                    source_index=0,
                    input_ids=[1, 2],
                    loss_mask=[1, 0],
                    aux_hidden_states=aux,
                    target_final_hidden=final,
                )
                writer.freeze()
            reference = root / "reference.pt"
            torch.save(
                {
                    "input_ids": torch.tensor([1, 2]),
                    "aux_hidden_states": aux,
                    "target_final_hidden": final,
                    "target_logits": torch.tensor(
                        [[1.0, 2.0, -1.0], [3.0, 4.0, -2.0]]
                    ),
                    "layer_ids": [1, 20, 38, 56, 75],
                },
                reference,
            )
            capture = {
                "aux_hidden_states": aux.float(),
                "target_final_hidden": final.float(),
                "target_logits": torch.tensor(
                    [[1.0, 2.0, -1.0], [3.0, 4.0, -2.0]]
                ),
                "input_ids": torch.tensor([1, 2]),
                "layer_ids": torch.tensor([1, 20, 38, 56, 75]),
            }
            artifact = calibrate_parity_gate(
                direct_runs=[capture, capture, capture],
                negative_controls={
                    "shifted_layer": {
                        **{
                            key: capture[key] + 0.5
                            for key in (
                                "aux_hidden_states",
                                "target_final_hidden",
                                "target_logits",
                            )
                        },
                        "input_ids": capture["input_ids"],
                        "layer_ids": capture["layer_ids"],
                    },
                    "pre_norm": {
                        **{
                            key: capture[key] - 0.4
                            for key in (
                                "aux_hidden_states",
                                "target_final_hidden",
                                "target_logits",
                            )
                        },
                        "input_ids": capture["input_ids"],
                        "layer_ids": capture["layer_ids"],
                    },
                },
                floors={
                    "cosine_error": 1e-9,
                    "max_abs_error": 1e-6,
                    "mean_abs_error": 1e-6,
                },
                identity=self.identity,
            )
            artifact_path = root / "gate.json"
            identity_path = root / "identity.json"
            artifact_path.write_text(json.dumps(artifact))
            identity_path.write_text(json.dumps(self.identity))
            target_io_dir = root / "target-io"
            target_io_dir.mkdir()
            weights = target_io_dir / "model.safetensors"
            weights.write_bytes(b"target-io")
            (target_io_dir / "manifest.json").write_text(
                json.dumps({"weights_file": weights.name})
            )
            output = StringIO()
            head = torch.nn.Linear(2, 3, bias=False)
            with torch.no_grad():
                head.weight.copy_(
                    torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -0.5]])
                )
            with mock.patch(
                "tools.validate_hidden_cache.load_frozen_target_io",
                return_value=SimpleNamespace(lm_head=head),
            ), redirect_stdout(output):
                status = validate_cache(
                    [
                        "--cache-dir", str(cache),
                        "--reference-pt", str(reference),
                        "--parity-gate", str(artifact_path),
                        "--runtime-identity-json", str(identity_path),
                        "--target-io-dir", str(target_io_dir),
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["reference"], "matched_parity_gate")
            self.assertTrue(payload["parity"]["passed"])
            attestation = validate_training_parity_attestation(cache, target_io_dir)
            self.assertTrue(attestation["passed"])
            weights.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "target I/O"):
                validate_training_parity_attestation(cache, target_io_dir)

    def test_training_rejects_a_frozen_cache_without_parity_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            with PackedHiddenWriter(
                cache,
                spec=HiddenCacheSpec(
                    layer_ids=(1,),
                    hidden_size=2,
                    capture_mapping=(("test", 1, "tap", "post_decoder_block"),),
                ),
            ) as writer:
                writer.append(
                    sample_id="missing-attestation",
                    source_index=0,
                    input_ids=[1],
                    loss_mask=[1],
                    aux_hidden_states=torch.ones(1, 1, 2),
                    target_final_hidden=torch.ones(1, 2),
                )
                writer.freeze()
            target_io = root / "target-io"
            target_io.mkdir()
            with self.assertRaisesRegex(ValueError, "parity attestation"):
                validate_training_parity_attestation(cache, target_io)


if __name__ == "__main__":
    unittest.main()
