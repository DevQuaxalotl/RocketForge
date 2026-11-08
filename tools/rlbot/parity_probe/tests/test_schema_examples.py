from __future__ import annotations

import hashlib
import json
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class SchemaExampleTests(unittest.TestCase):
    def test_schemas_are_valid_json_documents(self) -> None:
        fixture_schema = json.loads(
            (ROOT / "schemas" / "fixture.schema.json").read_text(encoding="utf-8")
        )
        result_schema = json.loads(
            (ROOT / "schemas" / "result.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            fixture_schema["properties"]["schema"]["const"],
            "rocketforge.parity.fixture.v1",
        )
        self.assertEqual(
            result_schema["properties"]["schema"]["const"],
            "rocketforge.parity.result.v1",
        )

    def test_fixture_uses_physical_non_inverted_boost_arrays(self) -> None:
        fixture = json.loads(
            (ROOT / "examples" / "synthetic_fixture.json").read_text(
                encoding="utf-8"
            )
        )
        frame = fixture["sequences"][0]["frames"][0]
        state = frame["state"]
        self.assertEqual(frame["target_car_id"], state["players"][0]["car_id"])
        self.assertEqual(len(state["boost_pads"]), 34)
        self.assertEqual(len(state["boost_pad_timers"]), 34)
        self.assertNotEqual(
            state["boost_pad_timers"], list(reversed(state["boost_pad_timers"]))
        )

    def test_result_vector_hashes_match_float32_little_endian_values(self) -> None:
        records = read_jsonl(ROOT / "examples" / "synthetic_reference.jsonl")
        self.assertEqual(len(records), 1)
        record = records[0]
        vectors = [
            record["observation"],
            record["network"]["shared_features"],
            record["network"]["raw_logits"],
            record["network"]["masked_logits"],
        ]
        for vector in vectors:
            payload = b"".join(struct.pack("<f", value) for value in vector["values"])
            self.assertEqual(len(vector["values"]), vector["size"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), vector["float32_le_sha256"])

    def test_action_mask_hash_uses_one_byte_per_bit(self) -> None:
        record = read_jsonl(ROOT / "examples" / "synthetic_reference.jsonl")[0]
        mask = record["action_mask"]
        self.assertEqual(len(mask["bits"]), mask["size"])
        self.assertEqual(
            hashlib.sha256(bytes(mask["bits"])).hexdigest(), mask["sha256"]
        )

    def test_action_plan_and_runtime_timing_are_self_consistent(self) -> None:
        record = read_jsonl(ROOT / "examples" / "synthetic_reference.jsonl")[0]
        plan = record["action_plan"]
        runtime = record["runtime"]
        self.assertGreater(len(plan), 0)
        self.assertEqual(record["parsed_action"], plan[0]["action"])
        self.assertEqual(runtime["queued_action"], plan[0]["action"])
        self.assertEqual(runtime["action_ticks"], sum(item["ticks"] for item in plan))
        self.assertEqual(
            runtime["action_ticks"],
            runtime["tick_skip"] - runtime["action_delay"],
        )


if __name__ == "__main__":
    unittest.main()
