from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = load_module("parity_compare", ROOT / "compare_parity_outputs.py")


def refresh_float_hash(vector) -> None:
    payload = b"".join(struct.pack("<f", value) for value in vector["values"])
    vector["float32_le_sha256"] = hashlib.sha256(payload).hexdigest()


class CompareParityOutputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = compare.load_jsonl(
            ROOT / "examples" / "synthetic_reference.jsonl"
        )
        cls.candidate = compare.load_jsonl(
            ROOT / "examples" / "synthetic_candidate.jsonl"
        )

    def test_identical_example_passes_strictly(self) -> None:
        report = compare.compare_records(
            self.reference,
            self.candidate,
            compare.Tolerances(),
            strict_hashes=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["mismatch_count"], 0)
        self.assertEqual(report["warning_count"], 0)

    def test_observation_tolerance_can_pass_with_hash_warning(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["observation"]["values"][0] += 5e-7
        refresh_float_hash(candidate[0]["observation"])
        report = compare.compare_records(
            self.reference,
            candidate,
            compare.Tolerances(),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["warning_count"], 1)

    def test_observation_outside_tolerance_reports_index(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["observation"]["values"][2] += 1e-3
        refresh_float_hash(candidate[0]["observation"])
        report = compare.compare_records(
            self.reference,
            candidate,
            compare.Tolerances(),
        )
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                mismatch["path"] == "observation.values[2]"
                for mismatch in report["mismatches"]
            )
        )

    def test_parsed_action_is_exact_by_default(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["parsed_action"][1] = 1e-8
        candidate[0]["action_plan"][0]["action"][1] = 1e-8
        candidate[0]["runtime"]["queued_action"][1] = 1e-8
        report = compare.compare_records(
            self.reference,
            candidate,
            compare.Tolerances(),
        )
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                mismatch["path"] == "parsed_action[1]"
                for mismatch in report["mismatches"]
            )
        )

    def test_empty_programmatic_stream_is_an_integrity_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains no records"):
            compare.compare_records(
                self.reference,
                [],
                compare.Tolerances(),
            )

    def test_jsonl_rejects_non_finite_numbers(self) -> None:
        record = copy.deepcopy(self.reference[0])
        serialized = json.dumps(record).replace("0.25", "NaN", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(serialized + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compare.load_jsonl(path)

    def test_declared_vector_size_is_an_integrity_error(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["network"]["raw_logits"]["size"] += 1
        with self.assertRaisesRegex(ValueError, "declares 4 values"):
            compare.compare_records(
                self.reference,
                candidate,
                compare.Tolerances(),
            )

    def test_stale_action_mask_hash_is_an_integrity_error(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["action_mask"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "action_mask.sha256"):
            compare.compare_records(
                self.reference,
                candidate,
                compare.Tolerances(),
            )

    def test_matching_stale_hashes_cannot_pass_strict_comparison(self) -> None:
        reference = copy.deepcopy(self.reference)
        candidate = copy.deepcopy(self.candidate)
        reference[0]["observation"]["float32_le_sha256"] = "0" * 64
        candidate[0]["observation"]["float32_le_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "integrity error"):
            compare.compare_records(
                reference,
                candidate,
                compare.Tolerances(),
                strict_hashes=True,
            )

    def test_strict_cli_returns_two_for_stale_hash(self) -> None:
        reference = copy.deepcopy(self.reference)
        candidate = copy.deepcopy(self.candidate)
        reference[0]["observation"]["float32_le_sha256"] = "0" * 64
        candidate[0]["observation"]["float32_le_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.jsonl"
            candidate_path = Path(directory) / "candidate.jsonl"
            reference_path.write_text(json.dumps(reference[0]) + "\n", encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate[0]) + "\n", encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                result = compare.main(
                    [
                        str(reference_path),
                        str(candidate_path),
                        "--strict-hashes",
                    ]
                )
        self.assertEqual(result, 2)
        self.assertIn("integrity error", errors.getvalue())

    def test_empty_jsonl_inputs_return_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.jsonl"
            candidate_path = Path(directory) / "candidate.jsonl"
            reference_path.write_text("", encoding="utf-8")
            candidate_path.write_text("", encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                result = compare.main([str(reference_path), str(candidate_path)])
        self.assertEqual(result, 2)
        self.assertIn("contains no records", errors.getvalue())

    def test_identity_only_records_fail_by_default(self) -> None:
        record = {
            "schema": compare.RESULT_SCHEMA,
            "producer": "synthetic",
            "fixture_id": "fixture",
            "sequence_id": "sequence",
            "frame_id": "frame",
            "target_car_id": 1,
            "obs_rng_seed": 2,
        }
        with self.assertRaisesRegex(ValueError, "observation: missing required field"):
            compare.compare_records(
                [record],
                [copy.deepcopy(record)],
                compare.Tolerances(),
            )

    def test_unhashable_identity_is_a_clean_integrity_error(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["fixture_id"] = []
        with self.assertRaisesRegex(ValueError, "fixture_id"):
            compare.compare_records(
                self.reference,
                candidate,
                compare.Tolerances(),
            )

    def test_action_plan_ticks_are_compared_exactly(self) -> None:
        reference = copy.deepcopy(self.reference)
        candidate = copy.deepcopy(self.candidate)
        action = reference[0]["action_plan"][0]["action"]
        reference[0]["action_plan"] = [
            {"action": copy.deepcopy(action), "ticks": 0},
            {"action": copy.deepcopy(action), "ticks": 1},
        ]
        candidate[0]["action_plan"] = [
            {"action": copy.deepcopy(action), "ticks": 1},
            {"action": copy.deepcopy(action), "ticks": 0},
        ]
        report = compare.compare_records(
            reference,
            candidate,
            compare.Tolerances(),
        )
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                mismatch["path"] == "action_plan[0].ticks"
                for mismatch in report["mismatches"]
            )
        )

    def test_runtime_timing_is_compared_exactly(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["runtime"]["tick_skip"] = 9
        candidate[0]["runtime"]["action_delay"] = 8
        report = compare.compare_records(
            self.reference,
            candidate,
            compare.Tolerances(),
        )
        self.assertFalse(report["passed"])
        paths = {mismatch["path"] for mismatch in report["mismatches"]}
        self.assertIn("runtime.tick_skip", paths)
        self.assertIn("runtime.action_delay", paths)

    def test_selected_action_must_match_argmax(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["runtime"]["selected_action_id"] = 2
        with self.assertRaisesRegex(ValueError, "must equal network.argmax_action_id"):
            compare.compare_records(
                self.reference,
                candidate,
                compare.Tolerances(),
            )

    def test_fixture_completeness_catches_identical_truncation(self) -> None:
        fixture = json.loads(
            (ROOT / "examples" / "synthetic_fixture.json").read_text(
                encoding="utf-8"
            )
        )
        extra_frame = copy.deepcopy(fixture["sequences"][0]["frames"][0])
        extra_frame["frame_id"] = "missing-from-both-streams"
        fixture["sequences"][0]["frames"].append(extra_frame)
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "fixture.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                result = compare.main(
                    [
                        str(ROOT / "examples" / "synthetic_reference.jsonl"),
                        str(ROOT / "examples" / "synthetic_candidate.jsonl"),
                        "--fixture",
                        str(fixture_path),
                    ]
                )
        self.assertEqual(result, 2)
        self.assertIn("missing 1", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
