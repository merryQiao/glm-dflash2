from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src").resolve()
sys.path.insert(0, str(SRC))


class StandaloneImportTest(unittest.TestCase):
    def test_package_is_standalone_and_resolves_below_local_src(self) -> None:
        names = (
            "glm53_stage_a",
            "glm53_stage_a.agent_trajectory",
            "glm53_stage_a.jsonl",
            "glm53_stage_a.open_swe_trajectories",
            "glm53_stage_a.provenance",
            "glm53_stage_a.sglang_stage_a",
            "glm53_stage_a.trajectory_tokens",
            "glm53_stage_a.vibe_coding",
            "glm53_stage_a.web_tools",
            "glm53_stage_a.workspaces",
            "tools.generate_trajectories",
            "tools.prepare_open_swe_trajectories",
        )
        for name in names:
            module = importlib.import_module(name)
            module_path = Path(module.__file__).resolve()
            self.assertTrue(module_path.is_relative_to(ROOT))
        self.assertNotIn("glm_dflash2", sys.modules)


if __name__ == "__main__":
    unittest.main()
