from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


surface = load_module("parity_public_surface", ROOT / "check_public_surface.py")


class PublicSurfaceTests(unittest.TestCase):
    def inspect_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            relative = PurePosixPath("tools/rlbot/parity_probe/sample.txt")
            path = repo / Path(*relative.parts)
            path.parent.mkdir(parents=True)
            path.write_text(text, encoding="utf-8")
            return surface.inspect_file(repo, relative)

    def test_windows_drive_path_is_rejected(self) -> None:
        separator = chr(92)
        findings = self.inspect_text(
            "source = 'D:" + separator + "project" + separator + "adapter'\n"
        )
        self.assertTrue(any(finding.rule == "absolute_path" for finding in findings))

    def test_unc_path_is_rejected(self) -> None:
        separator = chr(92)
        findings = self.inspect_text(
            "source = '" + separator * 2 + "host" + separator + "share'\n"
        )
        self.assertTrue(any(finding.rule == "absolute_path" for finding in findings))

    def test_escaped_windows_drive_path_is_rejected(self) -> None:
        separator = chr(92)
        findings = self.inspect_text(
            "source = 'E:"
            + separator * 2
            + "workspace"
            + separator * 2
            + "adapter'\n"
        )
        self.assertTrue(any(finding.rule == "absolute_path" for finding in findings))

    def test_escaped_unc_path_is_rejected(self) -> None:
        separator = chr(92)
        findings = self.inspect_text(
            "source = '"
            + separator * 4
            + "host"
            + separator * 2
            + "share'\n"
        )
        self.assertTrue(any(finding.rule == "absolute_path" for finding in findings))

    def test_detector_does_not_flag_its_own_source(self) -> None:
        relative = PurePosixPath("tools/rlbot/parity_probe/check_public_surface.py")
        findings = surface.inspect_file(REPO_ROOT, relative)
        self.assertFalse(
            [finding for finding in findings if finding.rule == "absolute_path"]
        )

    def test_local_directory_and_model_artifact_are_rejected_when_tracked(self) -> None:
        findings = surface.inspect_tracked_paths(
            [
                PurePosixPath("tools/rlbot/parity_probe/local/adapter.cpp"),
                PurePosixPath("tools/rlbot/parity_probe/examples/policy.lt"),
            ]
        )
        rules = {finding.rule for finding in findings}
        self.assertIn("tracked_local_directory", rules)
        self.assertIn("tracked_binary_artifact", rules)

    def test_local_directory_matching_is_case_insensitive(self) -> None:
        findings = surface.inspect_tracked_paths(
            [
                PurePosixPath("tools/rlbot/parity_probe/Local/adapter.cpp"),
                PurePosixPath("tools/rlbot/parity_probe/RESULTS/output.jsonl"),
                PurePosixPath("tools/rlbot/parity_probe/WoRk/fixture.json"),
            ]
        )
        local_findings = [
            finding
            for finding in findings
            if finding.rule == "tracked_local_directory"
        ]
        self.assertEqual(len(local_findings), 3)

    def test_generic_relative_paths_are_allowed(self) -> None:
        findings = self.inspect_text("fixture = 'examples/synthetic_fixture.json'\n")
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
