#!/usr/bin/env python3
"""Compare two RocketForge parity-probe JSONL result streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RESULT_SCHEMA = "rocketforge.parity.result.v1"
FIXTURE_SCHEMA = "rocketforge.parity.fixture.v1"
COMPARISON_SCHEMA = "rocketforge.parity.comparison.v1"
IDENTITY_FIELDS = (
    "fixture_id",
    "sequence_id",
    "frame_id",
    "target_car_id",
    "obs_rng_seed",
)
MISSING = object()


@dataclass(frozen=True)
class Tolerances:
    observation_atol: float = 1e-6
    network_atol: float = 1e-5
    control_atol: float = 0.0
    relative: float = 0.0


class Comparison:
    def __init__(self, max_details: int) -> None:
        self.max_details = max_details
        self.mismatch_count = 0
        self.warning_count = 0
        self.mismatches: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def mismatch(
        self,
        identity: Mapping[str, Any] | None,
        path: str,
        kind: str,
        reference: Any = MISSING,
        candidate: Any = MISSING,
        detail: str | None = None,
    ) -> None:
        self.mismatch_count += 1
        if len(self.mismatches) >= self.max_details:
            return
        self.mismatches.append(
            _difference(identity, path, kind, reference, candidate, detail)
        )

    def warning(
        self,
        identity: Mapping[str, Any] | None,
        path: str,
        kind: str,
        reference: Any = MISSING,
        candidate: Any = MISSING,
        detail: str | None = None,
    ) -> None:
        self.warning_count += 1
        if len(self.warnings) >= self.max_details:
            return
        self.warnings.append(
            _difference(identity, path, kind, reference, candidate, detail)
        )


def _difference(
    identity: Mapping[str, Any] | None,
    path: str,
    kind: str,
    reference: Any,
    candidate: Any,
    detail: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "identity": dict(identity) if identity is not None else None,
        "path": path,
        "kind": kind,
    }
    if reference is not MISSING:
        result["reference"] = reference
    if candidate is not MISSING:
        result["candidate"] = candidate
    if detail is not None:
        result["detail"] = detail
    return result


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_jsonl(
    path: Path,
    *,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line, parse_constant=_reject_non_finite)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")
            if record.get("schema") != RESULT_SCHEMA:
                raise ValueError(
                    f"{path}:{line_number}: expected schema {RESULT_SCHEMA!r}, "
                    f"got {record.get('schema')!r}"
                )
            missing = [field for field in IDENTITY_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing identity field(s): {', '.join(missing)}"
                )
            validate_result_integrity(
                record,
                f"{path}:{line_number}",
                allow_partial=allow_partial,
            )
            records.append(record)
    if not records:
        raise ValueError(f"{path}: result stream contains no records")
    return records


def _fixture_id(value: Any, location: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{location}: expected a string or integer identity")
    if isinstance(value, str) and not value:
        raise ValueError(f"{location}: identity strings must not be empty")
    return value


def load_fixture_identities(path: Path) -> set[tuple[Any, ...]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            fixture = json.load(stream, parse_constant=_reject_non_finite)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid fixture JSON: {error}") from error
    if not isinstance(fixture, Mapping):
        raise ValueError(f"{path}: fixture must be an object")
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(
            f"{path}: expected fixture schema {FIXTURE_SCHEMA!r}, "
            f"got {fixture.get('schema')!r}"
        )
    fixture_id = _fixture_id(fixture.get("fixture_id"), f"{path}:fixture_id")
    sequences = fixture.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"{path}: fixture must contain at least one sequence")

    identities: set[tuple[Any, ...]] = set()
    for sequence_index, sequence in enumerate(sequences):
        sequence_path = f"{path}:sequences[{sequence_index}]"
        if not isinstance(sequence, Mapping):
            raise ValueError(f"{sequence_path}: expected an object")
        sequence_id = _fixture_id(
            sequence.get("sequence_id"), f"{sequence_path}.sequence_id"
        )
        frames = sequence.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{sequence_path}: must contain at least one frame")
        for frame_index, frame in enumerate(frames):
            frame_path = f"{sequence_path}.frames[{frame_index}]"
            if not isinstance(frame, Mapping):
                raise ValueError(f"{frame_path}: expected an object")
            frame_id = _fixture_id(frame.get("frame_id"), f"{frame_path}.frame_id")
            target_car_id = frame.get("target_car_id")
            obs_rng_seed = frame.get("obs_rng_seed")
            for field, value, maximum in (
                ("target_car_id", target_car_id, 2**32 - 1),
                ("obs_rng_seed", obs_rng_seed, 2**64 - 1),
            ):
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    or value > maximum
                ):
                    raise ValueError(
                        f"{frame_path}.{field}: expected an integer from 0 through {maximum}"
                    )
            identity = (
                fixture_id,
                sequence_id,
                frame_id,
                target_car_id,
                obs_rng_seed,
            )
            if identity in identities:
                raise ValueError(f"{frame_path}: duplicate fixture frame identity")
            identities.add(identity)
    return identities


def validate_fixture_completeness(
    records: Sequence[Mapping[str, Any]],
    expected: set[tuple[Any, ...]],
    label: str,
) -> None:
    actual = {identity_of(record) for record in records}
    missing = expected - actual
    unexpected = actual - expected
    if not missing and not unexpected:
        return
    detail = f"missing {len(missing)}, unexpected {len(unexpected)} frame(s)"
    if missing:
        detail += f"; first missing identity: {identity_dict(sorted(missing, key=repr)[0])}"
    elif unexpected:
        detail += (
            "; first unexpected identity: "
            f"{identity_dict(sorted(unexpected, key=repr)[0])}"
        )
    raise ValueError(f"{label} result stream does not match fixture: {detail}")


def identity_of(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in IDENTITY_FIELDS)


def identity_dict(key: Sequence[Any]) -> dict[str, Any]:
    return dict(zip(IDENTITY_FIELDS, key))


def index_records(
    records: Iterable[dict[str, Any]],
    label: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = identity_of(record)
        if key in result:
            rendered = ", ".join(
                f"{field}={value!r}" for field, value in zip(IDENTITY_FIELDS, key)
            )
            raise ValueError(f"{label}: duplicate record identity: {rendered}")
        result[key] = record
    return result


def get_path(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return MISSING
        current = current[part]
    return current


def _compare_presence(
    comparison: Comparison,
    identity: Mapping[str, Any],
    path: str,
    reference: Any,
    candidate: Any,
    allow_missing: bool,
) -> bool:
    if reference is MISSING and candidate is MISSING:
        return False
    if reference is MISSING or candidate is MISSING:
        handler = comparison.warning if allow_missing else comparison.mismatch
        handler(
            identity,
            path,
            "missing_field",
            reference,
            candidate,
            "field is present in only one result stream",
        )
        return False
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _integrity_error(location: str, path: str, detail: str) -> None:
    raise ValueError(f"{location}: integrity error at {path}: {detail}")


def _require_fields(
    value: Any,
    location: str,
    path: str,
    fields: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _integrity_error(location, path, "expected an object")
    for field in fields:
        if field not in value:
            _integrity_error(location, f"{path}.{field}", "missing required field")
    return value


def _validate_action(action: Any, location: str, path: str) -> None:
    if not isinstance(action, list) or len(action) != 8:
        _integrity_error(location, path, "expected an eight-value action array")
    for index, value in enumerate(action):
        if not _is_number(value) or not math.isfinite(float(value)):
            _integrity_error(
                location,
                f"{path}[{index}]",
                "expected a finite number",
            )


def _validate_float_vector(
    vector: Any,
    location: str,
    path: str,
) -> None:
    if not isinstance(vector, Mapping):
        _integrity_error(location, path, "expected an object")
    for field in ("size", "float32_le_sha256", "values"):
        if field not in vector:
            _integrity_error(location, f"{path}.{field}", "missing field")

    size = vector["size"]
    values = vector["values"]
    declared_hash = vector["float32_le_sha256"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _integrity_error(location, f"{path}.size", "expected a non-negative integer")
    if not isinstance(values, list):
        _integrity_error(location, f"{path}.values", "expected an array")
    if size != len(values):
        _integrity_error(
            location,
            f"{path}.size",
            f"declares {size} values but the array contains {len(values)}",
        )

    payload = bytearray()
    for index, value in enumerate(values):
        if not _is_number(value) or not math.isfinite(float(value)):
            _integrity_error(
                location,
                f"{path}.values[{index}]",
                "expected a finite number",
            )
        try:
            payload.extend(struct.pack("<f", float(value)))
        except (OverflowError, struct.error) as error:
            _integrity_error(
                location,
                f"{path}.values[{index}]",
                f"value cannot be represented as float32: {error}",
            )

    expected_hash = hashlib.sha256(payload).hexdigest()
    if not isinstance(declared_hash, str) or declared_hash != expected_hash:
        _integrity_error(
            location,
            f"{path}.float32_le_sha256",
            f"expected {expected_hash}, got {declared_hash!r}",
        )


def _validate_action_mask(mask: Any, location: str) -> None:
    path = "action_mask"
    if not isinstance(mask, Mapping):
        _integrity_error(location, path, "expected an object")
    for field in ("size", "bits", "sha256"):
        if field not in mask:
            _integrity_error(location, f"{path}.{field}", "missing field")

    size = mask["size"]
    bits = mask["bits"]
    declared_hash = mask["sha256"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _integrity_error(location, f"{path}.size", "expected a non-negative integer")
    if not isinstance(bits, list) or not all(
        isinstance(bit, int) and not isinstance(bit, bool) and bit in (0, 1)
        for bit in bits
    ):
        _integrity_error(
            location,
            f"{path}.bits",
            "expected an array containing only integer 0 and 1 values",
        )
    if size != len(bits):
        _integrity_error(
            location,
            f"{path}.size",
            f"declares {size} bits but the array contains {len(bits)}",
        )

    expected_hash = hashlib.sha256(bytes(bits)).hexdigest()
    if not isinstance(declared_hash, str) or declared_hash != expected_hash:
        _integrity_error(
            location,
            f"{path}.sha256",
            f"expected {expected_hash}, got {declared_hash!r}",
        )


def validate_result_integrity(
    record: Mapping[str, Any],
    location: str,
    *,
    allow_partial: bool = False,
) -> None:
    if record.get("schema") != RESULT_SCHEMA:
        _integrity_error(location, "schema", f"expected {RESULT_SCHEMA!r}")
    producer = record.get("producer", MISSING)
    if not isinstance(producer, str) or not producer:
        _integrity_error(location, "producer", "expected a non-empty string")
    for field in IDENTITY_FIELDS:
        if field not in record:
            _integrity_error(location, field, "missing required identity field")
    for field in ("fixture_id", "sequence_id", "frame_id"):
        try:
            _fixture_id(record[field], f"{location}:{field}")
        except ValueError as error:
            _integrity_error(location, field, str(error).split(": ", 1)[-1])
    for field, maximum in (
        ("target_car_id", 2**32 - 1),
        ("obs_rng_seed", 2**64 - 1),
    ):
        value = record[field]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > maximum
        ):
            _integrity_error(
                location,
                field,
                f"expected an integer from 0 through {maximum}",
            )

    required_sections = (
        "observation",
        "action_mask",
        "network",
        "parsed_action",
        "action_plan",
        "runtime",
    )
    if not allow_partial:
        for field in required_sections:
            if field not in record:
                _integrity_error(location, field, "missing required field")

    for path in (
        "observation",
        "network.shared_features",
        "network.raw_logits",
        "network.masked_logits",
    ):
        vector = get_path(record, path)
        if vector is not MISSING:
            _validate_float_vector(vector, location, path)
    action_mask = get_path(record, "action_mask")
    if action_mask is not MISSING:
        _validate_action_mask(action_mask, location)

    network = get_path(record, "network")
    if network is not MISSING:
        if not isinstance(network, Mapping):
            _integrity_error(location, "network", "expected an object")
        if not allow_partial:
            _require_fields(
                network,
                location,
                "network",
                (
                    "shared_features",
                    "raw_logits",
                    "masked_logits",
                    "argmax_action_id",
                    "top1_margin",
                ),
            )
        argmax_action_id = network.get("argmax_action_id", MISSING)
        if argmax_action_id is not MISSING and (
            not isinstance(argmax_action_id, int)
            or isinstance(argmax_action_id, bool)
            or argmax_action_id < 0
        ):
            _integrity_error(
                location,
                "network.argmax_action_id",
                "expected a non-negative integer",
            )
        top1_margin = network.get("top1_margin", MISSING)
        if top1_margin is not MISSING and (
            not _is_number(top1_margin) or not math.isfinite(float(top1_margin))
        ):
            _integrity_error(
                location,
                "network.top1_margin",
                "expected a finite number",
            )

    parsed_action = get_path(record, "parsed_action")
    if parsed_action is not MISSING:
        _validate_action(parsed_action, location, "parsed_action")

    action_plan = get_path(record, "action_plan")
    if action_plan is not MISSING:
        if not isinstance(action_plan, list):
            _integrity_error(location, "action_plan", "expected an array")
        if not action_plan:
            _integrity_error(location, "action_plan", "must contain at least one segment")
        for index, segment in enumerate(action_plan):
            segment_path = f"action_plan[{index}]"
            segment = _require_fields(
                segment,
                location,
                segment_path,
                ("action", "ticks"),
            )
            _validate_action(segment["action"], location, f"{segment_path}.action")
            ticks = segment["ticks"]
            if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 0:
                _integrity_error(
                    location,
                    f"{segment_path}.ticks",
                    "expected a non-negative integer",
                )

    runtime = get_path(record, "runtime")
    if runtime is not MISSING:
        if not isinstance(runtime, Mapping):
            _integrity_error(location, "runtime", "expected an object")
        runtime_fields = (
            "selected_action_id",
            "tick_skip",
            "action_delay",
            "action_ticks",
            "queued_action",
            "applied_action",
        )
        if not allow_partial:
            runtime = _require_fields(
                runtime,
                location,
                "runtime",
                runtime_fields,
            )
        for field in ("selected_action_id", "action_delay", "action_ticks"):
            value = runtime.get(field, MISSING)
            if value is not MISSING and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                _integrity_error(
                    location,
                    f"runtime.{field}",
                    "expected a non-negative integer",
                )
        tick_skip = runtime.get("tick_skip", MISSING)
        if tick_skip is not MISSING and (
            not isinstance(tick_skip, int)
            or isinstance(tick_skip, bool)
            or tick_skip <= 0
        ):
            _integrity_error(
                location,
                "runtime.tick_skip",
                "expected a positive integer",
            )
        action_delay = runtime.get("action_delay", MISSING)
        action_ticks = runtime.get("action_ticks", MISSING)
        if tick_skip is not MISSING and action_delay is not MISSING:
            if action_delay > tick_skip:
                _integrity_error(
                    location,
                    "runtime.action_delay",
                    "must not exceed runtime.tick_skip",
                )
        if (
            tick_skip is not MISSING
            and action_delay is not MISSING
            and action_ticks is not MISSING
            and action_ticks != tick_skip - action_delay
        ):
            _integrity_error(
                location,
                "runtime.action_ticks",
                "must equal runtime.tick_skip minus runtime.action_delay",
            )
        for field in ("queued_action", "applied_action"):
            action = runtime.get(field, MISSING)
            if action is not MISSING:
                _validate_action(action, location, f"runtime.{field}")

        if isinstance(action_plan, list) and action_ticks is not MISSING:
            total_ticks = sum(segment["ticks"] for segment in action_plan)
            if total_ticks != action_ticks:
                _integrity_error(
                    location,
                    "action_plan",
                    f"segment ticks sum to {total_ticks}, expected {action_ticks}",
                )

    mask_size = (
        action_mask.get("size", MISSING)
        if isinstance(action_mask, Mapping)
        else MISSING
    )
    raw_logits = get_path(record, "network.raw_logits")
    masked_logits = get_path(record, "network.masked_logits")
    raw_size = raw_logits.get("size", MISSING) if isinstance(raw_logits, Mapping) else MISSING
    masked_size = (
        masked_logits.get("size", MISSING)
        if isinstance(masked_logits, Mapping)
        else MISSING
    )
    if mask_size is not MISSING and raw_size is not MISSING and raw_size != mask_size:
        _integrity_error(
            location,
            "network.raw_logits.size",
            f"must equal action_mask.size ({mask_size})",
        )
    if mask_size is not MISSING and masked_size is not MISSING and masked_size != mask_size:
        _integrity_error(
            location,
            "network.masked_logits.size",
            f"must equal action_mask.size ({mask_size})",
        )

    argmax_action_id = get_path(record, "network.argmax_action_id")
    selected_action_id = get_path(record, "runtime.selected_action_id")
    if (
        argmax_action_id is not MISSING
        and selected_action_id is not MISSING
        and selected_action_id != argmax_action_id
    ):
        _integrity_error(
            location,
            "runtime.selected_action_id",
            "must equal network.argmax_action_id",
        )
    if argmax_action_id is not MISSING and mask_size is not MISSING:
        if argmax_action_id >= mask_size:
            _integrity_error(
                location,
                "network.argmax_action_id",
                f"must be less than the action dimension ({mask_size})",
            )
        bits = action_mask.get("bits", [])
        if bits[argmax_action_id] != 1:
            _integrity_error(
                location,
                "network.argmax_action_id",
                "selects a disabled action-mask entry",
            )

    if isinstance(action_plan, list) and action_plan:
        first_action = action_plan[0]["action"]
        if parsed_action is not MISSING and parsed_action != first_action:
            _integrity_error(
                location,
                "parsed_action",
                "must equal action_plan[0].action",
            )
        queued_action = get_path(record, "runtime.queued_action")
        if queued_action is not MISSING and queued_action != first_action:
            _integrity_error(
                location,
                "runtime.queued_action",
                "must equal action_plan[0].action",
            )


def _compare_exact(
    comparison: Comparison,
    identity: Mapping[str, Any],
    path: str,
    reference_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    allow_missing: bool,
) -> None:
    reference = get_path(reference_record, path)
    candidate = get_path(candidate_record, path)
    if not _compare_presence(
        comparison, identity, path, reference, candidate, allow_missing
    ):
        return
    if reference != candidate:
        comparison.mismatch(identity, path, "exact_mismatch", reference, candidate)


def _numeric_error(reference: float, candidate: float) -> tuple[float, float]:
    absolute = abs(reference - candidate)
    scale = max(abs(reference), abs(candidate))
    relative = absolute / scale if scale > 0 else 0.0
    return absolute, relative


def _compare_numeric(
    comparison: Comparison,
    identity: Mapping[str, Any],
    path: str,
    reference_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    absolute_tolerance: float,
    relative_tolerance: float,
    allow_missing: bool,
) -> None:
    reference = get_path(reference_record, path)
    candidate = get_path(candidate_record, path)
    if not _compare_presence(
        comparison, identity, path, reference, candidate, allow_missing
    ):
        return
    if not _is_number(reference) or not _is_number(candidate):
        comparison.mismatch(
            identity,
            path,
            "invalid_type",
            reference,
            candidate,
            "both values must be finite JSON numbers",
        )
        return
    if not math.isfinite(float(reference)) or not math.isfinite(float(candidate)):
        comparison.mismatch(
            identity, path, "non_finite", reference, candidate
        )
        return
    if not math.isclose(
        float(reference),
        float(candidate),
        abs_tol=absolute_tolerance,
        rel_tol=relative_tolerance,
    ):
        absolute, relative = _numeric_error(float(reference), float(candidate))
        comparison.mismatch(
            identity,
            path,
            "numeric_mismatch",
            reference,
            candidate,
            f"absolute_error={absolute:.9g}, relative_error={relative:.9g}",
        )


def _compare_numeric_array(
    comparison: Comparison,
    identity: Mapping[str, Any],
    path: str,
    reference_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    absolute_tolerance: float,
    relative_tolerance: float,
    allow_missing: bool,
    display_path: str | None = None,
) -> None:
    reported_path = display_path or path
    reference = get_path(reference_record, path)
    candidate = get_path(candidate_record, path)
    if not _compare_presence(
        comparison, identity, reported_path, reference, candidate, allow_missing
    ):
        return
    if not isinstance(reference, list) or not isinstance(candidate, list):
        comparison.mismatch(
            identity,
            reported_path,
            "invalid_type",
            reference,
            candidate,
            "both values must be arrays",
        )
        return
    if len(reference) != len(candidate):
        comparison.mismatch(
            identity,
            reported_path,
            "length_mismatch",
            len(reference),
            len(candidate),
        )
        return

    worst_index: int | None = None
    worst_absolute = -1.0
    worst_relative = -1.0
    invalid_index: int | None = None
    for index, (reference_value, candidate_value) in enumerate(zip(reference, candidate)):
        if not _is_number(reference_value) or not _is_number(candidate_value):
            invalid_index = index
            break
        reference_float = float(reference_value)
        candidate_float = float(candidate_value)
        if not math.isfinite(reference_float) or not math.isfinite(candidate_float):
            invalid_index = index
            break
        if math.isclose(
            reference_float,
            candidate_float,
            abs_tol=absolute_tolerance,
            rel_tol=relative_tolerance,
        ):
            continue
        absolute, relative = _numeric_error(reference_float, candidate_float)
        if absolute > worst_absolute:
            worst_index = index
            worst_absolute = absolute
            worst_relative = relative

    if invalid_index is not None:
        comparison.mismatch(
            identity,
            f"{reported_path}[{invalid_index}]",
            "invalid_number",
            reference[invalid_index],
            candidate[invalid_index],
        )
    elif worst_index is not None:
        comparison.mismatch(
            identity,
            f"{reported_path}[{worst_index}]",
            "numeric_mismatch",
            reference[worst_index],
            candidate[worst_index],
            f"absolute_error={worst_absolute:.9g}, relative_error={worst_relative:.9g}",
        )


def _mask_bits(value: Any) -> list[int] | None:
    if isinstance(value, str):
        if any(character not in "01" for character in value):
            return None
        return [int(character) for character in value]
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and item in (0, 1)
        for item in value
    ):
        return value
    return None


def _compare_mask(
    comparison: Comparison,
    identity: Mapping[str, Any],
    reference_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    allow_missing: bool,
) -> None:
    path = "action_mask.bits"
    reference = get_path(reference_record, path)
    candidate = get_path(candidate_record, path)
    if not _compare_presence(
        comparison, identity, path, reference, candidate, allow_missing
    ):
        return
    reference_bits = _mask_bits(reference)
    candidate_bits = _mask_bits(candidate)
    if reference_bits is None or candidate_bits is None:
        comparison.mismatch(
            identity,
            path,
            "invalid_type",
            reference,
            candidate,
            "mask bits must be a bit string or an array containing only 0 and 1",
        )
        return
    if len(reference_bits) != len(candidate_bits):
        comparison.mismatch(
            identity,
            path,
            "length_mismatch",
            len(reference_bits),
            len(candidate_bits),
        )
        return
    for index, (reference_bit, candidate_bit) in enumerate(
        zip(reference_bits, candidate_bits)
    ):
        if reference_bit != candidate_bit:
            comparison.mismatch(
                identity,
                f"{path}[{index}]",
                "exact_mismatch",
                reference_bit,
                candidate_bit,
            )
            return


def _compare_action_plan(
    comparison: Comparison,
    identity: Mapping[str, Any],
    reference_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    control_tolerance: float,
    relative_tolerance: float,
    allow_missing: bool,
) -> None:
    path = "action_plan"
    reference = get_path(reference_record, path)
    candidate = get_path(candidate_record, path)
    if not _compare_presence(
        comparison, identity, path, reference, candidate, allow_missing
    ):
        return
    if not isinstance(reference, list) or not isinstance(candidate, list):
        comparison.mismatch(
            identity,
            path,
            "invalid_type",
            reference,
            candidate,
            "both values must be arrays",
        )
        return
    if len(reference) != len(candidate):
        comparison.mismatch(
            identity,
            path,
            "length_mismatch",
            len(reference),
            len(candidate),
        )
        return

    for index, (reference_segment, candidate_segment) in enumerate(
        zip(reference, candidate)
    ):
        segment_path = f"{path}[{index}]"
        reference_ticks = reference_segment.get("ticks", MISSING)
        candidate_ticks = candidate_segment.get("ticks", MISSING)
        if _compare_presence(
            comparison,
            identity,
            f"{segment_path}.ticks",
            reference_ticks,
            candidate_ticks,
            allow_missing,
        ) and reference_ticks != candidate_ticks:
            comparison.mismatch(
                identity,
                f"{segment_path}.ticks",
                "exact_mismatch",
                reference_ticks,
                candidate_ticks,
            )

        _compare_numeric_array(
            comparison,
            identity,
            "action",
            {"action": reference_segment.get("action", MISSING)},
            {"action": candidate_segment.get("action", MISSING)},
            control_tolerance,
            relative_tolerance,
            allow_missing,
            display_path=f"{segment_path}.action",
        )


def _compare_hash(
    comparison: Comparison,
    identity: Mapping[str, Any],
    path: str,
    value_path: str,
    reference_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    strict_hashes: bool,
    allow_missing: bool,
) -> None:
    reference = get_path(reference_record, path)
    candidate = get_path(candidate_record, path)
    if not _compare_presence(
        comparison, identity, path, reference, candidate, allow_missing
    ):
        return
    if reference == candidate:
        return
    values_available = (
        get_path(reference_record, value_path) is not MISSING
        and get_path(candidate_record, value_path) is not MISSING
    )
    if strict_hashes or not values_available:
        comparison.mismatch(
            identity,
            path,
            "hash_mismatch",
            reference,
            candidate,
            "hash mismatch is fatal because bitwise parity is required or values are absent",
        )
    else:
        comparison.warning(
            identity,
            path,
            "hash_mismatch",
            reference,
            candidate,
            "numeric values are compared with tolerance; use --strict-hashes for bitwise parity",
        )


def compare_record(
    comparison: Comparison,
    key: tuple[Any, ...],
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tolerances: Tolerances,
    strict_hashes: bool,
    allow_missing: bool,
) -> None:
    identity = identity_dict(key)

    for path in (
        "observation.size",
        "action_mask.size",
        "network.shared_features.size",
        "network.raw_logits.size",
        "network.masked_logits.size",
        "network.argmax_action_id",
        "runtime.selected_action_id",
        "runtime.tick_skip",
        "runtime.action_delay",
        "runtime.action_ticks",
    ):
        _compare_exact(
            comparison, identity, path, reference, candidate, allow_missing
        )

    _compare_numeric_array(
        comparison,
        identity,
        "observation.values",
        reference,
        candidate,
        tolerances.observation_atol,
        tolerances.relative,
        allow_missing,
    )
    _compare_mask(comparison, identity, reference, candidate, allow_missing)

    for path in (
        "network.shared_features.values",
        "network.raw_logits.values",
        "network.masked_logits.values",
    ):
        _compare_numeric_array(
            comparison,
            identity,
            path,
            reference,
            candidate,
            tolerances.network_atol,
            tolerances.relative,
            allow_missing,
        )
    _compare_numeric(
        comparison,
        identity,
        "network.top1_margin",
        reference,
        candidate,
        tolerances.network_atol,
        tolerances.relative,
        allow_missing,
    )

    for path in (
        "parsed_action",
        "runtime.queued_action",
        "runtime.applied_action",
    ):
        _compare_numeric_array(
            comparison,
            identity,
            path,
            reference,
            candidate,
            tolerances.control_atol,
            tolerances.relative,
            allow_missing,
        )
    _compare_action_plan(
        comparison,
        identity,
        reference,
        candidate,
        tolerances.control_atol,
        tolerances.relative,
        allow_missing,
    )

    _compare_hash(
        comparison,
        identity,
        "observation.float32_le_sha256",
        "observation.values",
        reference,
        candidate,
        strict_hashes,
        allow_missing,
    )
    for name in ("shared_features", "raw_logits", "masked_logits"):
        _compare_hash(
            comparison,
            identity,
            f"network.{name}.float32_le_sha256",
            f"network.{name}.values",
            reference,
            candidate,
            strict_hashes,
            allow_missing,
        )
    _compare_hash(
        comparison,
        identity,
        "action_mask.sha256",
        "action_mask.bits",
        reference,
        candidate,
        strict_hashes,
        allow_missing,
    )


def compare_records(
    reference_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    tolerances: Tolerances,
    *,
    strict_hashes: bool = False,
    allow_missing: bool = False,
    max_details: int = 50,
) -> dict[str, Any]:
    if not reference_records:
        raise ValueError("reference result stream contains no records")
    if not candidate_records:
        raise ValueError("candidate result stream contains no records")
    for index, record in enumerate(reference_records):
        validate_result_integrity(
            record,
            f"reference record {index}",
            allow_partial=allow_missing,
        )
    for index, record in enumerate(candidate_records):
        validate_result_integrity(
            record,
            f"candidate record {index}",
            allow_partial=allow_missing,
        )
    reference = index_records(reference_records, "reference")
    candidate = index_records(candidate_records, "candidate")
    comparison = Comparison(max_details)

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    for key in sorted(reference_keys - candidate_keys, key=repr):
        comparison.mismatch(
            identity_dict(key), "record", "missing_candidate_record"
        )
    for key in sorted(candidate_keys - reference_keys, key=repr):
        comparison.mismatch(
            identity_dict(key), "record", "unexpected_candidate_record"
        )
    common_keys = reference_keys & candidate_keys
    for key in sorted(common_keys, key=repr):
        compare_record(
            comparison,
            key,
            reference[key],
            candidate[key],
            tolerances,
            strict_hashes,
            allow_missing,
        )

    return {
        "schema": COMPARISON_SCHEMA,
        "passed": comparison.mismatch_count == 0,
        "records": {
            "reference": len(reference),
            "candidate": len(candidate),
            "matched": len(common_keys),
        },
        "mismatch_count": comparison.mismatch_count,
        "warning_count": comparison.warning_count,
        "mismatches": comparison.mismatches,
        "warnings": comparison.warnings,
        "details_truncated": (
            comparison.mismatch_count > len(comparison.mismatches)
            or comparison.warning_count > len(comparison.warnings)
        ),
        "tolerances": {
            "observation_atol": tolerances.observation_atol,
            "network_atol": tolerances.network_atol,
            "control_atol": tolerances.control_atol,
            "relative": tolerances.relative,
            "strict_hashes": strict_hashes,
            "allow_missing": allow_missing,
        },
    }


def format_human_report(report: Mapping[str, Any]) -> str:
    records = report["records"]
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        f"Parity comparison: {status}",
        (
            "Records: "
            f"{records['matched']} matched, "
            f"{records['reference']} reference, {records['candidate']} candidate"
        ),
        f"Mismatches: {report['mismatch_count']}; warnings: {report['warning_count']}",
    ]
    for label, entries in (
        ("Mismatch", report["mismatches"]),
        ("Warning", report["warnings"]),
    ):
        for entry in entries:
            identity = entry.get("identity")
            identity_text = ""
            if identity:
                identity_text = " [" + ", ".join(
                    f"{field}={identity[field]!r}" for field in IDENTITY_FIELDS
                ) + "]"
            detail = f": {entry['detail']}" if "detail" in entry else ""
            lines.append(
                f"{label}{identity_text} {entry['path']} ({entry['kind']}){detail}"
            )
    if report["details_truncated"]:
        lines.append("Additional details were truncated; increase --max-details to show more.")
    return "\n".join(lines)


def non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="reference probe JSONL")
    parser.add_argument("candidate", type=Path, help="candidate probe JSONL")
    parser.add_argument(
        "--fixture",
        type=Path,
        help="require both result streams to contain exactly the fixture's frame identities",
    )
    parser.add_argument(
        "--observation-atol",
        type=non_negative_float,
        default=1e-6,
        help="absolute tolerance for observation values (default: 1e-6)",
    )
    parser.add_argument(
        "--network-atol",
        type=non_negative_float,
        default=1e-5,
        help="absolute tolerance for features and logits (default: 1e-5)",
    )
    parser.add_argument(
        "--control-atol",
        type=non_negative_float,
        default=0.0,
        help="absolute tolerance for parsed and runtime actions (default: exact)",
    )
    parser.add_argument(
        "--rtol",
        type=non_negative_float,
        default=0.0,
        help="relative tolerance for every numeric comparison (default: 0)",
    )
    parser.add_argument(
        "--strict-hashes",
        action="store_true",
        help="require bitwise-identical observation and action-mask hashes",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="warn instead of failing when a measurement exists in only one stream",
    )
    parser.add_argument(
        "--max-details",
        type=positive_int,
        default=50,
        help="maximum stored details of each severity (default: 50)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="write the complete machine-readable comparison summary",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the machine-readable summary instead of the human report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reference = load_jsonl(args.reference, allow_partial=args.allow_missing)
        candidate = load_jsonl(args.candidate, allow_partial=args.allow_missing)
        if args.fixture is not None:
            expected = load_fixture_identities(args.fixture)
            validate_fixture_completeness(reference, expected, "reference")
            validate_fixture_completeness(candidate, expected, "candidate")
        report = compare_records(
            reference,
            candidate,
            Tolerances(
                observation_atol=args.observation_atol,
                network_atol=args.network_atol,
                control_atol=args.control_atol,
                relative=args.rtol,
            ),
            strict_hashes=args.strict_hashes,
            allow_missing=args.allow_missing,
            max_details=args.max_details,
        )
    except (OSError, ValueError) as error:
        print(f"parity comparison error: {error}", file=sys.stderr)
        return 2

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_human_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
