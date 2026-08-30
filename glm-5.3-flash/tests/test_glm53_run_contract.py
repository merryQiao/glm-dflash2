from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tools import generate_trajectories


class GLM53RunContractTest(unittest.TestCase):
    def _args(self, **updates):
        values = {
            "endpoint": None,
            "python": "/opt/glm53/bin/python",
            "host": "127.0.0.1",
            "port": 30000,
            "tp_size": 16,
            "dtype": "bfloat16",
            "device": "npu",
            "attention_backend": "ascend",
            "reasoning_parser": "glm45",
            "tool_call_parser": "glm47",
            "mem_fraction_static": 0.9,
            "context_length": 131072,
            "max_running_requests": 2,
            "max_total_tokens": 131072,
            "quantization": None,
            "moe_a2a_backend": None,
            "deepep_mode": None,
            "server_extra_arg": [],
            "web_search_provider": "searxng",
            "web_search_endpoint": "http://search-a/v1/search",
            "browser_provider": "direct",
            "browser_endpoint": "http://browser-a/fetch",
            "web_api_key_file": None,
            "web_search_max_calls": 2,
            "browser_max_calls": 8,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def test_service_contract_records_all_rollout_affecting_server_fields(self):
        service = generate_trajectories._service_contract(
            self._args(), endpoint_identity=None
        )
        self.assertEqual(service["dtype"], "bfloat16")
        self.assertEqual(service["device"], "npu")
        self.assertEqual(service["attention_backend"], "ascend")
        self.assertEqual(service["reasoning_parser"], "glm45")
        self.assertEqual(service["tool_call_parser"], "glm47")
        self.assertIn("quantization", service)
        self.assertIn("moe_a2a_backend", service)
        self.assertIn("deepep_mode", service)

    def test_external_contract_uses_attested_runtime_not_local_cli_defaults(self):
        runtime = {
            "sglang_version": "0.6.0",
            "cann_version": "9.0",
            "image_digest": "sha256:image",
            "tp_size": 32,
            "device": "npu",
            "attention_backend": "ascend",
            "reasoning_parser": "glm53-reasoning",
            "tool_call_parser": "glm53-tools",
            "context_length": 65536,
            "max_total_tokens": 65536,
            "quantization": None,
            "moe_a2a_backend": "deepep",
            "deepep_mode": "normal",
        }
        identity = {"manifest": {"dtype": "bfloat16", "runtime": runtime}}
        service = generate_trajectories._service_contract(
            self._args(
                endpoint="http://external:30000",
                reasoning_parser="misleading-local-default",
            ),
            identity,
        )
        self.assertEqual(service["reasoning_parser"], "glm53-reasoning")
        self.assertEqual(service["tp_size"], 32)
        self.assertNotIn("host", service)

    def test_web_contract_binds_endpoints_and_secret_identity(self):
        with tempfile.TemporaryDirectory() as td:
            key = Path(td) / "web.key"
            key.write_text("secret-a")
            first = generate_trajectories._web_contract(
                self._args(web_api_key_file=key)
            )
            self.assertEqual(first["search_endpoint"], "http://search-a/v1/search")
            self.assertEqual(first["browser_endpoint"], "http://browser-a/fetch")
            self.assertRegex(first["api_key_file_sha256"], r"^[0-9a-f]{64}$")
            second = generate_trajectories._web_contract(
                self._args(web_api_key_file=key, browser_endpoint="http://browser-b")
            )
            with self.assertRaisesRegex(ValueError, "resume contract mismatch for web_tools"):
                generate_trajectories._validate_resume_contract(
                    {"status": "partial", "web_tools": first},
                    {"status": "running", "web_tools": second},
                )

    def test_resume_rejects_changed_server_contract(self):
        current = {
            "status": "running",
            "service": generate_trajectories._service_contract(self._args(), None),
        }
        previous = {
            "status": "partial",
            "service": generate_trajectories._service_contract(
                self._args(reasoning_parser="different-parser"), None
            ),
        }
        with self.assertRaisesRegex(ValueError, "resume contract mismatch for service"):
            generate_trajectories._validate_resume_contract(previous, current)


if __name__ == "__main__":
    unittest.main()
