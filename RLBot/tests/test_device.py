"""Verify device selection and early CUDA failure without connecting to a match."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EXECUTABLE = Path(sys.argv[1]).resolve()
DEFAULT = sys.argv[2]

class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rocketforge device ")
        self.addCleanup(self.temp.cleanup)
        self.env = os.environ.copy()
        self.env.pop("RLBOT_AGENT_ID", None)
        self.env["CUDA_VISIBLE_DEVICES"] = "-1"
        self.missing_setup = str(Path(self.temp.name) / "missing.json")

    def launch(self, *args):
        result = subprocess.run([str(EXECUTABLE), *args, "--agent-setup", self.missing_setup],
            cwd=self.temp.name, env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_missing_value(self):
        result = subprocess.run([str(EXECUTABLE), "--device"], cwd=self.temp.name,
            env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertIn("--device requires gpu or cpu", result.stderr)

    def test_invalid_value(self):
        self.assertIn("--device must be gpu or cpu", self.launch("--device", "auto"))

    def test_cpu_does_not_initialize_cuda(self):
        output = self.launch("--device", "cpu")
        self.assertIn("RLBot inference device: CPU", output)
        self.assertIn("Failed to open agent setup file", output)

    def test_gpu_failure_never_falls_back(self):
        output = self.launch("--device", "gpu")
        self.assertIn("RLBot GPU initialization failed:", output)
        self.assertNotIn("RLBot inference device: CPU", output)
        self.assertNotIn("Failed to open agent setup file", output)

    def test_default_matches_build(self):
        output = self.launch()
        expected = "RLBot GPU initialization failed:" if DEFAULT == "gpu" else "RLBot inference device: CPU"
        self.assertIn(expected, output)

    def test_explicit_last_argument_wins(self):
        self.assertIn("RLBot inference device: CPU", self.launch("--device", "gpu", "--device", "cpu"))

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
