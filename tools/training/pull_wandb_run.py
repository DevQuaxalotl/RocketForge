"""
pull_wandb_run.py — Pull a single wandb run to disk for offline analysis.

Writes two files to an output directory:
  meta.json     — run config, summary, metadata, and available metric keys
  history.csv   — full per-step history via run.scan_history()

Usage:
    python tools/training/pull_wandb_run.py <ref> [--out DIR] [--project NAME]

<ref> accepts three forms:
  entity/project/run_id           Full path — used directly.
  https://wandb.ai/...            Run URL — entity/project/run_id extracted.
  bare-run-name                   Human display name — searched in the default
                                  project (--project, default: gigalearncpp)
                                  using the API's default entity. Names are not
                                  unique; if multiple runs match, a table is
                                  printed and the script exits non-zero asking
                                  you to re-run with an explicit path.

Authentication:
  Set the WANDB_API_KEY environment variable, or run `wandb login` first.
  The script checks auth before doing any work and exits non-zero if unauthenticated.

Self-bootstrapping:
  If wandb is not installed, the script installs it via pip into the current
  interpreter and retries. If installation fails, it exits non-zero.
"""

import argparse
import csv
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

# ---------------------------------------------------------------------------
# Pure parsing — no wandb import, fully unit-testable offline
# ---------------------------------------------------------------------------

def parse_run_reference(ref: str) -> dict:
    """Parse a wandb run reference into a structured dict.

    Returns one of:
        {"kind": "path", "entity": str, "project": str, "run_id": str}
        {"kind": "name", "name": str}

    Raises ValueError for malformed input.

    Three accepted forms:
        entity/project/run_id         — full path, returned as kind="path"
        https://wandb.ai/…/runs/…    — URL, parsed into kind="path"
        bare-name                     — no slashes, not a URL, kind="name"
    """
    ref = ref.strip()
    if not ref:
        raise ValueError("Run reference must not be empty.")

    # --- URL form ---
    if ref.startswith("http://") or ref.startswith("https://"):
        parsed = urllib.parse.urlparse(ref)
        if parsed.netloc != "wandb.ai":
            raise ValueError(
                f"URL does not point to wandb.ai (got netloc={parsed.netloc!r}). "
                "Only https://wandb.ai/... URLs are supported."
            )
        # path looks like: /entity/project[/groups/group-name]/runs/run_id[/][?...]
        # strip leading slash, split, ignore empties from trailing slash
        parts = [p for p in parsed.path.split("/") if p]
        # entity is always the first segment, project the second
        if len(parts) < 2:
            raise ValueError(
                f"Could not extract entity/project from wandb URL {ref!r}. "
                "Expected format: https://wandb.ai/<entity>/<project>/runs/<run_id>"
            )
        entity, project = parts[0], parts[1]
        # Locate "runs" anywhere in the path; run_id is the segment immediately after it
        try:
            runs_idx = parts.index("runs")
        except ValueError:
            raise ValueError(
                f"Could not find a 'runs' segment in wandb URL {ref!r}. "
                "Expected format: https://wandb.ai/<entity>/<project>/runs/<run_id>"
            )
        if runs_idx + 1 >= len(parts):
            raise ValueError(
                f"wandb URL {ref!r} has a 'runs' segment but no run_id after it. "
                "Expected format: https://wandb.ai/<entity>/<project>/runs/<run_id>"
            )
        run_id = parts[runs_idx + 1]
        return {"kind": "path", "entity": entity, "project": project, "run_id": run_id}

    # --- Slash-separated forms ---
    slash_parts = ref.split("/")
    if len(slash_parts) == 3:
        entity, project, run_id = slash_parts
        return {"kind": "path", "entity": entity, "project": project, "run_id": run_id}
    elif len(slash_parts) == 1:
        # Bare name — single token
        return {"kind": "name", "name": ref}
    else:
        raise ValueError(
            f"Invalid run reference {ref!r}. Expected one of:\n"
            "  entity/project/run_id\n"
            "  https://wandb.ai/<entity>/<project>/runs/<run_id>\n"
            "  bare-run-name"
        )


# ---------------------------------------------------------------------------
# wandb bootstrapping
# ---------------------------------------------------------------------------

def _ensure_wandb() -> None:
    """Try to import wandb; install it if missing. Exit on failure."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("wandb is not installed. Installing now via pip...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "wandb"],
            capture_output=False,
        )
        if result.returncode != 0:
            print(
                "\nERROR: Failed to install wandb. "
                "Please install it manually:\n"
                "    pip install wandb",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            import wandb  # noqa: F401
        except ImportError:
            print(
                "\nERROR: wandb was installed but still cannot be imported. "
                "Try running: pip install wandb",
                file=sys.stderr,
            )
            sys.exit(1)
        print("wandb installed successfully.")


def _check_auth() -> None:
    """Verify wandb authentication. Exit non-zero with instructions if not authenticated."""
    import wandb

    # wandb.api.api_key is a semi-internal attribute — verify it still exists on a wandb major-version bump.
    api_key = wandb.api.api_key
    if not api_key:
        print(
            "\nERROR: wandb is not authenticated.\n"
            "To authenticate, do one of the following:\n"
            "  1. Set the WANDB_API_KEY environment variable:\n"
            "         export WANDB_API_KEY=<your-key>   (Linux/macOS)\n"
            "         $env:WANDB_API_KEY = '<your-key>' (PowerShell)\n"
            "  2. Run:  wandb login\n"
            "\nGet your API key at: https://wandb.ai/authorize",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Run resolution
# ---------------------------------------------------------------------------

def _resolve_run(parsed_ref: dict, default_project: str):
    """Resolve a parsed reference to a wandb Run object.

    For kind="path" uses the path directly.
    For kind="name" searches the default project for matching display names.
    Returns a wandb.apis.public.Run.
    """
    import wandb

    api = wandb.Api()

    if parsed_ref["kind"] == "path":
        path = f"{parsed_ref['entity']}/{parsed_ref['project']}/{parsed_ref['run_id']}"
        try:
            run = api.run(path)
        except wandb.errors.CommError as exc:
            print(
                f"\nERROR: Could not fetch run at path {path!r}.\n"
                f"Details: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        return run

    # kind == "name": bare display-name search
    name = parsed_ref["name"]
    print(f"Searching for runs named {name!r} in project {default_project!r}...")
    try:
        runs = api.runs(
            path=default_project,
            filters={"display_name": name},
        )
        matches = list(runs)
    except wandb.errors.CommError as exc:
        print(
            f"\nERROR: Failed to search for runs named {name!r} "
            f"in project {default_project!r}.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(matches) == 0:
        print(
            f"\nERROR: No run found with display name {name!r} "
            f"in project {default_project!r}.\n"
            "Try using the full entity/project/run_id path instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(matches) > 1:
        print(
            f"\nERROR: Multiple runs found with display name {name!r}. "
            "Re-run with a specific entity/project/run_id path:\n",
            file=sys.stderr,
        )
        # Print a table of matches
        col_widths = [30, 20, 25, 12]
        headers = ["Name", "ID", "Created At", "State"]
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "  ".join("-" * w for w in col_widths)
        print(header_line, file=sys.stderr)
        print(sep_line, file=sys.stderr)
        for r in matches:
            row = [
                r.name,
                r.id,
                str(getattr(r, "created_at", "?")),
                r.state,
            ]
            print(
                "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)),
                file=sys.stderr,
            )
        print(
            "\nRe-run with:  python tools/training/pull_wandb_run.py "
            "<entity>/<project>/<run_id>",
            file=sys.stderr,
        )
        sys.exit(1)

    return matches[0]


# ---------------------------------------------------------------------------
# Writing output
# ---------------------------------------------------------------------------

def _write_meta(run, out_dir: Path, metric_keys: list, run_path: str) -> None:
    """Write meta.json for the run."""
    # Convert summary (wandb SummarySubDict) to a plain dict
    try:
        summary_dict = dict(run.summary)
    except Exception:
        summary_dict = {}

    meta = {
        "path": run_path,
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "created_at": str(getattr(run, "created_at", None)),
        "runtime": summary_dict.get("_runtime", None),
        "config": dict(run.config),
        "summary": summary_dict,
        "metric_keys": sorted(metric_keys),
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def _write_history(run, out_dir: Path) -> tuple[list, int]:
    """Write history.csv using run.scan_history(). Returns (metric_keys, row_count)."""
    print("Scanning full history (this may take a moment for long runs)...")
    rows = list(run.scan_history())

    # Collect union of all keys across rows for the header
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    # Stable ordering: put _step first if present, then alphabetical for the rest
    def _key_sort(k: str) -> tuple:
        return (0 if k == "_step" else 1, k)

    all_keys.sort(key=_key_sort)

    # Metric keys are those that don't start with _ (wandb internal fields)
    metric_keys = [k for k in all_keys if not k.startswith("_")]

    csv_path = out_dir / "history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return metric_keys, len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pull_wandb_run",
        description=(
            "Pull a single wandb run to disk for offline analysis.\n\n"
            "<ref> accepts: entity/project/run_id, a wandb.ai URL, or a bare run name."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ref",
        help=(
            "Run reference: entity/project/run_id, "
            "https://wandb.ai/<entity>/<project>/runs/<run_id>, "
            "or bare display name (searched in --project)."
        ),
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help=(
            "Output directory. Default: wandb/<run_id>/ relative to cwd. "
            "Created if it does not exist."
        ),
    )
    parser.add_argument(
        "--project",
        metavar="NAME",
        default="gigalearncpp",
        help=(
            "Default project (entity/project or just project) used when <ref> is a bare "
            "name. The API's configured default entity is used when no entity is included. "
            "Default: gigalearncpp."
        ),
    )
    args = parser.parse_args()

    # --- Self-bootstrap ---
    _ensure_wandb()

    # --- Auth check ---
    _check_auth()

    # --- Parse reference ---
    try:
        parsed_ref = parse_run_reference(args.ref)
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Resolve to a run object ---
    run = _resolve_run(parsed_ref, default_project=args.project)

    # Build the resolved path string (run.path is a list [entity, project, run_id])
    run_path_parts = run.path if isinstance(run.path, list) else run.path.split("/")
    run_path = "/".join(run_path_parts)
    run_id = run_path_parts[-1]

    # --- Determine output directory ---
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path("wandb") / run_id

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Write history ---
    metric_keys, row_count = _write_history(run, out_dir)

    # --- Write metadata (needs metric_keys) ---
    _write_meta(run, out_dir, metric_keys, run_path)

    if row_count == 0:
        print(
            "Note: The run history is empty. history.csv was written with headers only."
        )

    # --- Success summary ---
    print(
        f"\nDone.\n"
        f"  Resolved path : {run_path}\n"
        f"  Output dir    : {out_dir.resolve()}\n"
        f"  History rows  : {row_count}\n"
        f"  Metric keys   : {len(metric_keys)}"
    )


if __name__ == "__main__":
    main()
