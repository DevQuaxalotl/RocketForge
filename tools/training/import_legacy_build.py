#!/usr/bin/env python3
"""Import a staged legacy model build into the RocketForge build/segment layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MODEL_FILE_NAMES = {"POLICY.lt", "SHARED_HEAD.lt"}
VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class LegacyCheckpoint:
    timesteps: int
    path: Path


@dataclass(frozen=True)
class LegacyLayout:
    source_root: Path
    training_root: Path
    build_root: Path


@dataclass
class LegacySegment:
    stage_name: str
    legacy_version_id: str
    checkpoints: list[LegacyCheckpoint]
    segment_id: str = ""
    folder_name: str = ""
    parent_segment_id: str = ""
    source_checkpoint_path: str = ""

    @property
    def start_timestep(self) -> int:
        return min(checkpoint.timesteps for checkpoint in self.checkpoints)

    @property
    def end_timestep(self) -> int:
        return max(checkpoint.timesteps for checkpoint in self.checkpoints)


def sanitize_folder_part(text: str) -> str:
    result: list[str] = []
    last_was_separator = False

    for char in text:
        if char.isalnum():
            result.append(char)
            last_was_separator = False
        elif char == "." and result and not last_was_separator:
            result.append(char)
            last_was_separator = False
        elif not last_was_separator:
            result.append("_")
            last_was_separator = True

    while result and result[0] in {"_", "."}:
        result.pop(0)
    while result and result[-1] in {"_", "."}:
        result.pop()

    return "".join(result) or "V1"


def make_segment_id(major: int, minor: int) -> str:
    return f"v{major}.{minor:02d}"


def parse_segment_id(text: str) -> tuple[int, int] | None:
    match = VERSION_RE.match(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def next_segment_id(existing_ids: set[str]) -> str:
    next_major = 1
    next_minor = 0
    has_existing = False

    for segment_id in existing_ids:
        parsed = parse_segment_id(segment_id)
        if not parsed:
            continue
        has_existing = True
        major, minor = parsed
        if major > next_major or (major == next_major and minor >= next_minor):
            next_major = major
            next_minor = minor + 1

    if not has_existing:
        return make_segment_id(1, 0)
    return make_segment_id(next_major, next_minor)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def relative_to_build(build_root: Path, path: Path) -> str:
    return path.relative_to(build_root).as_posix()


def is_model_checkpoint(folder: Path) -> bool:
    if not folder.is_dir() or not folder.name.isdigit():
        return False
    return any((folder / model_file).exists() for model_file in MODEL_FILE_NAMES)


def resolve_training_root(source_root: Path) -> Path:
    named_training_root = source_root / "Training Stages"
    if named_training_root.exists():
        return named_training_root

    return source_root


def resolve_legacy_layout(input_path: Path) -> LegacyLayout:
    input_path = input_path.resolve()

    if input_path.name in {"legacy_import", "legacy_models"}:
        source_root = input_path
        build_root = input_path.parent
    elif (input_path / "legacy_import").exists():
        source_root = input_path / "legacy_import"
        build_root = input_path
    elif (input_path / "legacy_models").exists():
        source_root = input_path / "legacy_models"
        build_root = input_path
    else:
        raise ValueError("Pass a build root, legacy_import folder, or legacy_models folder.")

    training_root = resolve_training_root(source_root)
    if not training_root.exists():
        raise ValueError(f"Legacy training root does not exist: {training_root}")

    return LegacyLayout(source_root, training_root, build_root)


def find_legacy_segments(training_root: Path) -> list[LegacySegment]:
    if not training_root.exists():
        return []

    segments: list[LegacySegment] = []
    for stage_root in sorted(path for path in training_root.iterdir() if path.is_dir()):
        for version_root in sorted(path for path in stage_root.iterdir() if path.is_dir()):
            if not VERSION_RE.match(version_root.name):
                continue

            checkpoints = [
                LegacyCheckpoint(int(checkpoint_root.name), checkpoint_root)
                for checkpoint_root in sorted(version_root.iterdir())
                if is_model_checkpoint(checkpoint_root)
            ]
            if not checkpoints:
                continue

            checkpoints.sort(key=lambda checkpoint: checkpoint.timesteps)
            segments.append(LegacySegment(stage_root.name, version_root.name, checkpoints))

    segments.sort(key=lambda segment: (segment.end_timestep, segment.stage_name, segment.legacy_version_id))
    return segments


def load_existing_segment_ids(build_root: Path) -> set[str]:
    ids: set[str] = set()
    segments_root = build_root / "segments"
    if not segments_root.exists():
        return ids

    for segment_json in segments_root.glob("*/segment.json"):
        data = read_json(segment_json, {})
        segment_id = data.get("segment_id")
        if isinstance(segment_id, str) and segment_id:
            ids.add(segment_id)
    return ids


def assign_segments(build_root: Path, segments: list[LegacySegment]) -> None:
    existing_ids = load_existing_segment_ids(build_root)
    previous_segment: LegacySegment | None = None

    for segment in segments:
        segment.segment_id = next_segment_id(existing_ids)
        existing_ids.add(segment.segment_id)
        segment.folder_name = f"{segment.segment_id}_{sanitize_folder_part(segment.stage_name)}"

        if previous_segment:
            segment.parent_segment_id = previous_segment.segment_id
            segment.source_checkpoint_path = (
                f"segments/{previous_segment.folder_name}/checkpoints/{previous_segment.end_timestep}"
            )

        previous_segment = segment


def copy_tree(source: Path, destination: Path, dry_run: bool, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        if not dry_run:
            shutil.rmtree(destination)

    if not dry_run:
        shutil.copytree(source, destination)


def import_segments(build_root: Path, build_name: str, segments: list[LegacySegment], dry_run: bool, overwrite: bool) -> None:
    imported_at = timestamp()
    for segment in segments:
        segment_root = build_root / "segments" / segment.folder_name
        checkpoints_root = segment_root / "checkpoints"
        if not dry_run:
            checkpoints_root.mkdir(parents=True, exist_ok=True)

        for checkpoint in segment.checkpoints:
            copy_tree(checkpoint.path, checkpoints_root / str(checkpoint.timesteps), dry_run, overwrite)

        checkpoint_entries = [
            {
                "timesteps": checkpoint.timesteps,
                "path": f"segments/{segment.folder_name}/checkpoints/{checkpoint.timesteps}",
            }
            for checkpoint in segment.checkpoints
        ]
        segment_json = {
            "schema_version": 1,
            "segment_id": segment.segment_id,
            "segment_revision": segment.segment_id,
            "folder_name": segment.folder_name,
            "build_name": build_name,
            "parent_segment_id": segment.parent_segment_id or None,
            "branch_name": "main",
            "status": "active",
            "stage_name": segment.stage_name,
            "setup_hash": "",
            "created_at": imported_at,
            "updated_at": imported_at,
            "start_timestep": segment.start_timestep,
            "end_timestep": segment.end_timestep,
            "source_checkpoint_path": segment.source_checkpoint_path or None,
            "wandb": {
                "project": "",
                "group": "",
                "run_name": "",
                "run_id": "",
            },
            "notes": f"Imported from legacy stage version {segment.legacy_version_id}.",
            "eligible_benchmark": False,
            "eligible_static_opponent": False,
            "eligible_rlbot": False,
            "pinned": False,
            "checkpoints": checkpoint_entries,
        }
        if not dry_run:
            write_json(segment_root / "segment.json", segment_json)


def update_build_metadata(models_root: Path, build_root: Path, build_name: str, segments: list[LegacySegment], dry_run: bool) -> None:
    imported_at = timestamp()
    folder_name = build_root.name
    active_segment = max(segments, key=lambda segment: segment.end_timestep, default=None)
    active_timesteps = active_segment.end_timestep if active_segment else 0
    active_segment_id = active_segment.segment_id if active_segment else ""

    build_json = {
        "schema_version": 1,
        "build_name": build_name,
        "folder_name": folder_name,
        "active_segment_id": active_segment_id or None,
        "created_at": imported_at,
        "updated_at": imported_at,
        "active_timesteps": active_timesteps,
        "total_trained_timesteps": active_timesteps,
        "notes": "Imported from legacy_import.",
    }

    builds_path = models_root / "builds.json"
    registry = read_json(builds_path, {"schema_version": 1, "builds": []})
    existing_builds = registry.setdefault("builds", [])
    replacement = {
        "build_name": build_name,
        "folder_name": folder_name,
        "created_at": imported_at,
        "updated_at": imported_at,
        "active": not any(item.get("active", False) for item in existing_builds),
    }

    for index, item in enumerate(existing_builds):
        if item.get("folder_name") == folder_name or item.get("build_name") == build_name:
            replacement["active"] = item.get("active", replacement["active"])
            existing_builds[index] = replacement
            break
    else:
        existing_builds.append(replacement)

    static_versions_path = build_root / "static_versions.json"
    if not dry_run:
        write_json(build_root / "build.json", build_json)
        write_json(builds_path, registry)
        if not static_versions_path.exists():
            write_json(static_versions_path, {"schema_version": 1, "static_versions": []})


def import_recent_policy_versions(legacy_import_root: Path, build_root: Path, dry_run: bool, overwrite: bool) -> int:
    count = 0
    for legacy_name in ("Recent Policy Versions", "recent_policy_versions"):
        legacy_root = legacy_import_root / legacy_name
        if not legacy_root.exists():
            continue

        destination_root = build_root / "recent_policy_versions"
        if not dry_run:
            destination_root.mkdir(parents=True, exist_ok=True)

        for entry in legacy_root.iterdir():
            destination = destination_root / entry.name
            if entry.is_dir():
                copy_tree(entry, destination, dry_run, overwrite)
                count += 1
            elif entry.is_file():
                if destination.exists() and not overwrite:
                    raise FileExistsError(f"Destination already exists: {destination}")
                if not dry_run:
                    shutil.copy2(entry, destination)
                count += 1
    return count


def parse_benchmark_source_from_name(folder_name: str) -> tuple[int, str, str] | None:
    match = re.match(r"^(\d+)_(.+)_(v\d+\.\d+)$", folder_name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), match.group(2), match.group(3)


def import_skill_benchmarks(
    legacy_import_root: Path,
    build_root: Path,
    agent_name: str,
    segment_map: dict[tuple[str, str], str],
    dry_run: bool,
    overwrite: bool,
) -> int:
    legacy_root = legacy_import_root / "Skill Benchmarks"
    if not legacy_root.exists():
        return 0

    destination_root = build_root / "skill_benchmarks"
    if not dry_run:
        destination_root.mkdir(parents=True, exist_ok=True)

    benchmark_entries = []
    for benchmark_folder in sorted(path for path in legacy_root.iterdir() if path.is_dir()):
        stats_path = benchmark_folder / "STATS.json"
        stats = read_json(stats_path, {}) if stats_path.exists() else {}
        parsed_name = parse_benchmark_source_from_name(benchmark_folder.name)

        timesteps = int(stats.get("timesteps") or (parsed_name[0] if parsed_name else 0))
        stage_name = str(stats.get("source_stage_name") or (parsed_name[1] if parsed_name else "unknown-stage"))
        legacy_version = str(stats.get("source_version_id") or (parsed_name[2] if parsed_name else "unversioned"))
        segment_id = segment_map.get((stage_name, legacy_version), legacy_version)
        folder_name = f"{segment_id}_{timesteps}_{sanitize_folder_part(stage_name)}"

        copy_tree(benchmark_folder, destination_root / folder_name, dry_run, overwrite)

        new_stats = dict(stats)
        new_stats["benchmark_folder_name"] = folder_name
        new_stats["benchmark_id"] = folder_name
        new_stats["source_stage_name"] = stage_name
        new_stats["source_version_id"] = segment_id
        new_stats["timesteps"] = timesteps
        if not dry_run:
            write_json(destination_root / folder_name / "STATS.json", new_stats)

        created_at = str(new_stats.get("created_at", ""))
        benchmark_entries.append({
            "id": folder_name,
            "source_stage_name": stage_name,
            "source_version_id": segment_id,
            "created_at": created_at,
            "timesteps": timesteps,
            "path": f"skill_benchmarks/{folder_name}",
        })

    if not dry_run:
        write_json(destination_root / "benchmarks.json", {
            "schema_version": 1,
            "agent": agent_name,
            "benchmarks": benchmark_entries,
        })

    return len(benchmark_entries)


def import_legacy_build(legacy_import_root: Path, dry_run: bool, overwrite: bool) -> None:
    layout = resolve_legacy_layout(legacy_import_root)
    legacy_import_root = layout.source_root
    build_root = layout.build_root
    models_root = build_root.parent
    agent_root = models_root.parent
    build_name = build_root.name
    agent_name = agent_root.name

    segments = find_legacy_segments(layout.training_root)
    assign_segments(build_root, segments)
    segment_map = {
        (segment.stage_name, segment.legacy_version_id): segment.segment_id
        for segment in segments
    }

    print(f"Agent: {agent_name}")
    print(f"Build: {build_name}")
    print(f"Legacy source: {legacy_import_root}")
    print(f"Training root: {layout.training_root}")
    print(f"Segments to import: {len(segments)}")
    for segment in segments:
        checkpoint_text = ", ".join(str(checkpoint.timesteps) for checkpoint in segment.checkpoints)
        print(f"  {segment.segment_id}: {segment.stage_name} {segment.legacy_version_id} -> {checkpoint_text}")

    if not dry_run:
        (build_root / "segments").mkdir(parents=True, exist_ok=True)
        (build_root / "recent_policy_versions").mkdir(parents=True, exist_ok=True)
        (build_root / "skill_benchmarks").mkdir(parents=True, exist_ok=True)

    import_segments(build_root, build_name, segments, dry_run, overwrite)
    update_build_metadata(models_root, build_root, build_name, segments, dry_run)
    recent_count = import_recent_policy_versions(legacy_import_root, build_root, dry_run, overwrite)
    benchmark_count = import_skill_benchmarks(legacy_import_root, build_root, agent_name, segment_map, dry_run, overwrite)

    print(f"Recent policy entries imported: {recent_count}")
    print(f"Skill benchmarks imported: {benchmark_count}")
    print("Dry run complete." if dry_run else "Import complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_import_root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned import without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace destination folders if they already exist.")
    args = parser.parse_args()

    import_legacy_build(args.legacy_import_root, args.dry_run, args.overwrite)


if __name__ == "__main__":
    main()
