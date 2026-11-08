# Simulation engine internals

This page describes how RocketSim v2 and v3 sit behind one API, how ball prediction is shared,
and how saved engine settings load.
It's for contributors who change the simulator layer.
For choosing and building engines, see [Simulation engines](../Features/SimulationEngines.md).

## One API, two backends

Both engines sit behind the shared `RocketSim::Arena` API in `ForgeTraining/RLGym/Sim`.
Each arena is created with an `EngineId`,
and training, debug rendering, Param Testing, benchmarking, and RLBot all run in the same process,
on whichever engine each arena names.

The code is divided as follows:

| Path | Contents |
|---|---|
| `ForgeTraining/RLGym/RocketSimV3/upstream` | Unchanged public source: the ZealanL/RocketSim `v3-rust` branch, as a submodule. |
| `ForgeTraining/RLGym/RocketSimV3/bindings` | A standalone Rust C ABI and a generic C++ ownership wrapper, meant for upstream review. It has no RocketForge training types and doesn't depend on v2 headers. See its [README](../../ForgeTraining/RLGym/RocketSimV3/bindings/README.md). |
| `ForgeTraining/RLGym/Sim/src/Backend/V3Backend.cpp` | Maps the bindings onto the shared RocketSim API. It stays in RocketForge after the generic bindings move upstream. |
| `ForgeTraining/RLGym/Sim/src/Backend/V2Backend.cpp` | Wraps the vendored v2 engine, which lives in the `RocketSimV2` namespace. Only this backend includes v2 engine headers. |

For an exact in-memory copy of v3's physics and RNG state, use the bindings' `rsf_arena_clone` function;
setting a state is a teleport and doesn't restore solver caches.

Minimal RLBot source exports bundle the simulator sources,
so exported prediction uses the same engine as the exported agent.

`tools/build/test_v3_detection.ps1` checks the configure-time detection of v3 and its messages.

## Ball prediction

Each model's observations predict on a ball-only arena of its prediction engine.
When the prediction engine is the arena's own engine,
the arena's prediction is reused instead of simulating the ball twice,
and models that share an arena and a prediction engine share one predictor.

Framework code that consumes observations calls `MakeConfiguredObsBuilder()`,
which wraps both registered builders and custom `MakeObsBuilder()` overrides.
A scoped predictor makes `GameState::GetFutureBallState()` use the agent's prediction profile
ahead of packet or match-engine prediction.
Calling RocketSim arenas directly bypasses that profile.

Observation runtimes with different predictors stay separate,
and separate arenas and incompatible profiles never share a tracker.
A same-tick ball correction, or a tick counter that moves backward or resets, discards stale predictions.

## Saved engine settings

The agent metadata in `training_setup.json` records the resolved
`training_engine_version`, `ball_prediction_engine_version`, and `use_ball_prediction` values.
Loading follows these rules, in memory only; loading never rewrites saved history:

- Continuing, branching, and later queue stages inherit the checkpoint's simulator and prediction settings.
  The saved-or-current setup choice still applies to the other agent settings.
- Files that still hold a ball-prediction tick rate load, and the rate is ignored.
  The setup hash keeps hashing that retired rate as 0, so existing hashes stay valid.
- Snapshots without the engine fields inherit their recorded top-level `engine_version`.
  Snapshots with no engine provenance use v2.
- Checkpoints without simulator metadata resolve to v2, and malformed metadata is an error.

## Debug render controls

Debug controls, reward snapshots, and critic estimates all run in the ForgeGUI process.
The viewer's pause, speed, and focus commands go back to the render loop.
Coalesced GUI controls keep a revision for each field,
so an unrelated setting can't replay an old focus or speed command.

## Validation

The validation assets contain Soccar geometry.
Rust tests also exercise Heatseeker and Snowday state on that geometry;
Hoops and Dropshot gameplay aren't validated.
Tests compare the bindings with native public v3 behavior,
and check RocketForge's episode and reset behavior separately.
`ForgeTrainingTests/SimEngineDifferences_tests.cpp` pins the known engine differences.
