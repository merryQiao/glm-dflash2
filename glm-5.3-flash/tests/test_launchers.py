from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def test_stage_a_launcher_is_valid_and_glm53_specific(self):
        launcher = ROOT / "scripts/run_stage_a_trajectories.sh"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("GLM-5.3-Flash-BF16", text)
        self.assertIn("REASONING_PARSER", text)
        self.assertIn("TOOL_CALL_PARSER", text)
        self.assertIn("ENDPOINT_MANIFEST", text)
        self.assertIn("MODEL_PATH", text)

    def test_runtime_sources_do_not_import_parent_glm52_package(self):
        forbidden = "glm" + "_dflash2"
        for directory in ("src", "tools", "scripts"):
            for path in (ROOT / directory).rglob("*"):
                if path.is_file() and path.suffix in {".py", ".sh"}:
                    with self.subTest(path=path):
                        self.assertNotIn(forbidden, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
