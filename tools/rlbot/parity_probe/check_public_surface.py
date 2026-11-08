#!/usr/bin/env python3
"""Reject machine-local data and build artifacts from the public parity tool."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


PUBLIC_ROOT = PurePosixPath("tools/rlbot/parity_probe")
LOCAL_DIRECTORIES = {"local", "results", "work", "__pycache__"}
LOCAL_FILE_PATTERNS = (
    re.compile(r"(^|[._-])local([._-]|$)", re.IGNORECASE),
    re.compile(r"(^|[._-])private([._-]|$)", re.IGNORECASE),
)
ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".dll",
    ".dylib",
    ".engine",
    ".exe",
    ".lib",
    ".lt",
    ".onnx",
    ".pdb",
    ".pt",
    ".pth",
    ".so",
}
MAX_PUBLIC_FILE_BYTES = 1_000_000
_BACKSLASH = chr(92)
ABSOLUTE_PATH_PATTERNS = (
    (
        "Windows drive path",
        re.compile(
            r"(?<![A-Za-z0-9_])[A-Za-z]:"
            + rf"(?:{re.escape(_BACKSLASH)}+|/)"
        ),
    ),
    (
        "UNC path",
        re.compile(
            rf"(?<!{re.escape(_BACKSLASH)})"
            + rf"{re.escape(_BACKSLASH)}{{2,}}"
            + rf"[^{re.escape(_BACKSLASH)}/\s]+"
            + rf"(?:{re.escape(_BACKSLASH)}+|/)"
        ),
    ),
    (
        "POSIX home path",
        re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users|root)/[^\s\"'<>]+"),
    ),
    ("file URI", re.compile(r"\bfile://", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str
    detail: str
    line: int | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path": self.path,
            "rule": self.rule,
            "detail": self.detail,
        }
        if self.line is not None:
            result["line"] = self.line
        return result


def run_git(repo_root: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return process.stdout


def find_repo_root(start: Path) -> Path:
    output = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if output.returncode != 0:
        message = output.stderr.strip()
        raise RuntimeError(f"could not locate Git repository: {message}")
    return Path(output.stdout.strip()).resolve()


def decode_git_paths(output: bytes) -> list[PurePosixPath]:
    return [
        PurePosixPath(item.decode("utf-8", errors="surrogateescape"))
        for item in output.split(b"\0")
        if item
    ]


def git_tracked_paths(repo_root: Path) -> list[PurePosixPath]:
    output = run_git(
        repo_root,
        "ls-files",
        "-z",
        "--",
        PUBLIC_ROOT.as_posix(),
    )
    return decode_git_paths(output)


def public_files_on_disk(repo_root: Path) -> list[PurePosixPath]:
    root = repo_root / Path(*PUBLIC_ROOT.parts)
    if not root.exists():
        return []
    result: list[PurePosixPath] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(repo_root).as_posix())
        if is_local_path(relative):
            continue
        result.append(relative)
    return result


def relative_parts(path: PurePosixPath) -> tuple[str, ...]:
    try:
        return path.relative_to(PUBLIC_ROOT).parts
    except ValueError:
        return path.parts


def is_local_path(path: PurePosixPath) -> bool:
    parts = relative_parts(path)
    return any(part.casefold() in LOCAL_DIRECTORIES for part in parts[:-1])


def inspect_tracked_paths(paths: Iterable[PurePosixPath]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        parts = relative_parts(path)
        rendered = path.as_posix()
        if any(part.casefold() in LOCAL_DIRECTORIES for part in parts[:-1]):
            findings.append(
                Finding(
                    rendered,
                    "tracked_local_directory",
                    "machine-local parity directories must never be tracked",
                )
            )
        filename = parts[-1] if parts else ""
        if any(pattern.search(filename) for pattern in LOCAL_FILE_PATTERNS):
            findings.append(
                Finding(
                    rendered,
                    "tracked_local_filename",
                    "machine-local manifests and adapters must never be tracked",
                )
            )
        if PurePosixPath(filename).suffix.lower() in ARTIFACT_SUFFIXES:
            findings.append(
                Finding(
                    rendered,
                    "tracked_binary_artifact",
                    "models, executables, and build products must never be tracked",
                )
            )
    return findings


def inspect_file(repo_root: Path, relative: PurePosixPath) -> list[Finding]:
    path = repo_root / Path(*relative.parts)
    rendered = relative.as_posix()
    try:
        size = path.stat().st_size
        data = path.read_bytes()
    except OSError as error:
        return [Finding(rendered, "read_error", str(error))]

    findings: list[Finding] = []
    if size > MAX_PUBLIC_FILE_BYTES:
        findings.append(
            Finding(
                rendered,
                "oversized_public_file",
                f"file is {size} bytes; public parity files must be at most {MAX_PUBLIC_FILE_BYTES}",
            )
        )
    if b"\0" in data:
        findings.append(
            Finding(
                rendered,
                "binary_public_file",
                "public parity files must be reviewable text",
            )
        )
        return findings

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        findings.append(
            Finding(
                rendered,
                "non_utf8_public_file",
                f"public parity files must be UTF-8 text: {error}",
            )
        )
        return findings

    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in ABSOLUTE_PATH_PATTERNS:
            match = pattern.search(line)
            if match:
                findings.append(
                    Finding(
                        rendered,
                        "absolute_path",
                        f"{label} found near {match.group(0)!r}",
                        line_number,
                    )
                )
    return findings


def inspect_public_surface(
    repo_root: Path,
    *,
    include_untracked: bool = True,
) -> tuple[list[Finding], int, int]:
    tracked = git_tracked_paths(repo_root)
    findings = inspect_tracked_paths(tracked)
    paths = set(tracked)
    if include_untracked:
        paths.update(public_files_on_disk(repo_root))
    scanned = 0
    for relative in sorted(paths, key=PurePosixPath.as_posix):
        if is_local_path(relative):
            continue
        if not (repo_root / Path(*relative.parts)).is_file():
            continue
        findings.extend(inspect_file(repo_root, relative))
        scanned += 1
    return findings, len(tracked), scanned


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="repository root (normally discovered automatically)",
    )
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="scan only Git-tracked files instead of also checking unstaged public files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo_root = (
            args.repo_root.resolve()
            if args.repo_root is not None
            else find_repo_root(Path(__file__).resolve().parent)
        )
        findings, tracked_count, scanned_count = inspect_public_surface(
            repo_root, include_untracked=not args.tracked_only
        )
    except (OSError, RuntimeError) as error:
        print(f"parity public-surface check error: {error}", file=sys.stderr)
        return 2

    report = {
        "passed": not findings,
        "tracked_files": tracked_count,
        "scanned_files": scanned_count,
        "finding_count": len(findings),
        "findings": [finding.as_dict() for finding in findings],
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif findings:
        print("Parity public-surface check: FAIL")
        for finding in findings:
            location = finding.path
            if finding.line is not None:
                location += f":{finding.line}"
            print(f"- {location}: {finding.rule}: {finding.detail}")
    else:
        print(
            "Parity public-surface check: PASS "
            f"({tracked_count} tracked, {scanned_count} scanned)"
        )
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
