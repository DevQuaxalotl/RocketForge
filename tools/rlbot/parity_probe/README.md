# RocketForge parity probe

The parity probe checks whether two independently built policy runtimes make the same decision from the same canonical game state. It records each stage of the decision path:

```text
game state -> observation -> action mask -> network logits -> action ID -> parser action plan
```

The result also echoes applied-action context supplied by the fixture or previous state. The probe does not advance physics or claim that this contextual action was scheduled by its parser.

This directory is intentionally generic. The tracked files contain the interchange protocol, RocketForge runner, comparator, schemas, synthetic examples, and privacy check. Machine-local adapters, source mappings, checkpoints, captures, results, and build products belong in ignored directories and must not be committed.

## Build the RocketForge runner

The runner is excluded from normal builds; the `build-parity` preset builds only it:

```powershell
cmake --preset forge
cmake --build --preset build-parity
```

List the agents compiled into that build:

```powershell
out/build/forge/RocketForgeParityProbe.exe --list-agents
```

Run a registered agent against a compatible checkpoint and fixture:

```powershell
out/build/forge/RocketForgeParityProbe.exe `
  --agent example-agent `
  --checkpoint tools/rlbot/parity_probe/local/checkpoint_folder `
  --fixture tools/rlbot/parity_probe/examples/synthetic_fixture.json `
  --producer rocketforge `
  --output tools/rlbot/parity_probe/results/rocketforge.jsonl
```

Pass `--device cuda` to run the same fixture through a CUDA-loaded policy. This is
useful for checking that a converted or portable model preserves action choices
and acceptable logit tolerances across CPU and GPU inference.

The synthetic fixture demonstrates the protocol; it is not expected to match every agent's observation or action dimensions. Real fixtures and their results should stay under ignored local or result directories.

`--setup` applies a saved public `agent_setup` before runtime construction. `--adapter` selects an optionally compiled frame adapter. Use `--list-adapters` to show the adapters in the current executable.

## Build an independent reference

An independent reference executable should consume `rocketforge.parity.fixture.v1` and emit one `rocketforge.parity.result.v1` JSON object per frame. Build old and current runtime implementations as separate processes when their dependencies or symbols conflict. They only need to agree on the JSON protocol.

The public [AdapterRegistry.h](include/RocketForge/ParityProbe/AdapterRegistry.h) provides hooks for state that a current RocketForge `GameState` cannot represent directly. Put machine-local adapter translation units directly in `local/`; the parity target discovers `local/*.cpp` at configure time. Reference projects and their dependency-specific build files can live below `local/` as well, but are deliberately not part of the public CMake graph.

Do not edit a canonical state to compensate for a suspected compatibility bug. For example, fixtures store boost pads and timers in normal, non-inverted physical order. Any legacy inversion belongs in the runtime being tested so a mismatch remains observable.

## Fixture protocol

[fixture.schema.json](schemas/fixture.schema.json) describes the complete structured fixture. A document contains one or more ordered sequences, and each sequence contains one or more frames.

The identity of a result frame is:

```text
fixture_id, sequence_id, frame_id, target_car_id, obs_rng_seed
```

Important sequencing rules:

- The first frame in a sequence resets the observation builder by default. Set `reset_obs_builder` explicitly when another reset boundary is needed.
- Consecutive frames preserve observation-builder state and link `GameState::prev` and matching `Player::prev` values.
- The runner seeds the observation RNG immediately before exactly one observation build. This makes shuffled player slots reproducible without hiding other stateful behavior.
- `ball_prediction` contains explicit prediction samples when an observation builder consumes them.
- `runtime.applied_action` is optional input evidence. If absent, the runner reports the target player's previous action as the applied action.
- The optional `extensions` objects are the only public escape hatch for adapter-specific data. Keep extension definitions and payloads machine-local if they disclose non-public implementation details.

Action arrays always use this eight-value order:

```text
[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
```

## Result protocol

[result.schema.json](schemas/result.schema.json) defines each JSONL record. The runner records:

- Observation values, size, and SHA-256.
- Action-mask bits, size, and SHA-256.
- Shared features, raw logits, masked logits, argmax action ID, and top-one margin.
- The parser-emitted action plan and exact ticks for every segment.
- Tick skip, action delay, action ticks, queued action, and applied-action context.

Float-vector hashes are calculated over contiguous IEEE-754 float32 little-endian bytes. Action-mask hashes use one unsigned byte per mask bit. Hashes make bitwise equality cheap to verify while retained numeric values allow useful tolerance-based diagnostics.

`parsed_action` is the first action emitted by the parser and is identical to `action_plan[0].action`. The full `action_plan` is authoritative for macro parsers. Its segment ticks sum to `runtime.action_ticks`, which equals `tick_skip - action_delay`. `runtime.queued_action` also names the first plan segment. `runtime.applied_action` is fixture input when supplied, otherwise the target player's previous action; it is evidence about runtime context, not a physics-simulation result.

When an agent has no shared head, `network.shared_features` equals the observation supplied directly to the policy. Masked logits use the same public inference rule as RocketForge: sanitize raw logits with `nan_to_num`, then add `-1e10` to disabled mask entries.

## Compare two results

Run the included exact synthetic example:

```powershell
python tools/rlbot/parity_probe/compare_parity_outputs.py `
  tools/rlbot/parity_probe/examples/synthetic_reference.jsonl `
  tools/rlbot/parity_probe/examples/synthetic_candidate.jsonl `
  --strict-hashes
```

For normal CPU FP32 comparisons, the defaults are:

- Observation absolute tolerance: `1e-6`
- Shared features and logits absolute tolerance: `1e-5`
- Parsed and runtime actions: exact
- Relative tolerance: zero

Before comparing the streams, the tool requires at least one record per input, verifies required fields and internal action/timing invariants, checks every declared vector size, and recomputes every float32 and action-mask hash from its accompanying values. An empty stream, stale hash, forged hash, or fixture-completeness failure is an input-integrity error, even if both streams contain the same bad evidence or the numeric values would pass tolerance. Input-integrity and parse errors exit with status `2`; they are never reported as policy mismatches.

After both streams pass integrity validation, a hash difference between them is a warning unless `--strict-hashes` is supplied. `--strict-hashes` therefore asks whether two valid vectors are bitwise identical; it does not make invalid evidence acceptable. Missing measurements fail by default, while `--allow-missing` downgrades them to warnings for an intentionally partial reference implementation.

Write a machine-readable report into the ignored results directory with:

```powershell
python tools/rlbot/parity_probe/compare_parity_outputs.py `
  tools/rlbot/parity_probe/results/reference.jsonl `
  tools/rlbot/parity_probe/results/rocketforge.jsonl `
  --fixture tools/rlbot/parity_probe/local/parity_fixture.json `
  --report-json tools/rlbot/parity_probe/results/comparison.json
```

The comparator matches records by identity rather than file order. Supplying `--fixture` is recommended for real comparisons: it requires both streams to contain exactly every fixture frame, catching the case where both producers stop at the same nonempty prefix. It reports missing frames, dimension changes, timing or plan changes, exact action differences, and a representative failing index with absolute and relative numeric error.

## Keep the public surface clean

Before staging or pushing parity-tool changes, run:

```powershell
python tools/rlbot/parity_probe/check_public_surface.py
```

The check uses `git ls-files` to reject tracked local directories, local manifests, model or binary artifacts, oversized files, non-UTF-8 data, and machine-specific absolute paths. Local-directory matching is case-insensitive. It also scans unstaged files in the public portion of this directory, while intentionally skipping ignored local work. This matters because an ignore rule does not protect a file that was force-added or tracked previously.

The following directories are ignored by this tool's own `.gitignore`:

```text
local/    adapters, reference projects, source mappings, local manifests
results/  probe output and comparison reports
work/     generated fixtures and temporary build inputs
```

If the privacy check reports a local file as tracked, remove it from the Git index before continuing. Do not weaken the check to accommodate machine-specific data in the public tree.

## Tests

The Python utilities use only the standard library:

```powershell
python -m unittest discover -s tools/rlbot/parity_probe/tests -v
```

The tests cover exact and tolerant comparisons, non-finite JSON rejection, size and hash integrity, schema/example consistency, vector hashes, and privacy-check regressions.

## What this proves

Exact observations, masks, action IDs, and parsed controls provide strong evidence that the policy runtime was reproduced correctly for the tested state corpus. Matching raw logits adds evidence that model architecture, parameters, activations, and preprocessing agree.

It does not prove that two full games remain frame-identical. Different physics engines, packet timing, tick scheduling, or floating-point execution can cause trajectories to diverge even when individual decisions match. Use asymmetric states, multiple teams and team sizes, aerial and grounded frames, unique boost timers, repeated sequence frames, and multiple observation RNG seeds to make the corpus representative.
