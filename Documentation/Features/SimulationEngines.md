# Simulation engines

RocketForge can train and play on two versions of the RocketSim simulator: RocketSim v2 and RocketSim v3.
Both are built into the same executables, and each agent chooses which one it trains on.
This page shows you how to build v3, how to choose an engine and a ball predictor for your agent,
how each tool picks its engine, and where the two engines behave differently.

For how the engines are wired in, see [Simulation engine internals](../Internals/SimulationEngines.md).

## Which engines your build has

v2 is always built.
v3 is built when two things are present:

- The v3 source, the `ForgeTraining/RLGym/RocketSimV3/upstream` submodule.
- A Rust toolchain.

Without them, the build still succeeds, with v2 only.

After it configures, `tools/build/build_and_test.ps1` prints an engines line, for example:

```text
Engines: v2 (built), v3 (built)
```

When v3 isn't built, the line gives the reason.
RocketForge never falls back from one engine to the other:
work that needs an engine your build lacks stops with a message that names the fix.
In ForgeGUI, that message is a **Simulator not available** dialog.

## Build v3

1. Initialize the v3 source, if you didn't clone with `--recursive`:

   ```powershell
   git submodule update --init ForgeTraining/RLGym/RocketSimV3/upstream
   ```

1. Install Rust.
   On Windows, run `tools/setup/install_rust.ps1`; on Linux, run `tools/setup/install_rust.sh`.
1. Configure and build again.
   The engines line now shows `v3 (built)`.

The pinned Rust toolchain is 1.99.0, set in `ForgeTraining/RLGym/RocketSimV3/bindings/rust-toolchain.toml`,
and the minimum is 1.95.
To build once with another installed rustup toolchain, without changing your default, set `ROCKETSIM_V3_RUST_TOOLCHAIN`.

The `ROCKETFORGE_ENABLE_V3` CMake option controls the build:

| Value | Effect |
|---|---|
| `AUTO` | Builds v3 when it can. This is the default. |
| `ON` | Fails the configure step when v3 can't be built. The Docker training image uses `ON`, so it never ships with v2 only. |
| `OFF` | Never builds v3. |

## Choose an engine for your agent

Set the engine in your agent's constructor.
`Bots/ExampleBot/ExampleAgent/src/ExampleAgent.cpp` shows these settings.

| Field | Values | Meaning |
|---|---|---|
| `trainingEngineVersion` | `2` or `3` | The engine that runs training games. |
| `ballPredictionEngineVersion` | `0`, `2`, or `3` | The engine that predicts the ball's path for observations. `0` follows the training engine. |
| `useBallPrediction` | `true` or `false` | Whether the agent's observations use ball prediction. |

These settings apply to new training.
Once a segment exists, its saved setup decides, as described in [What a checkpoint remembers](#what-a-checkpoint-remembers).

Ball prediction always runs at the engine's native 120 Hz.
Each model predicts with its own prediction engine,
so a model trained with v2 prediction keeps v2 prediction even in a v3 match.

If you write a custom observation builder that reads ball prediction,
call `GameState::GetFutureBallState()` rather than a RocketSim arena directly,
so that it uses the agent's prediction settings.

## How each tool picks its engine

| Tool | Engine |
|---|---|
| Training | The engine in the agent setup, or in the checkpoint that training continues from. Before it starts, the queue checks that every engine its stages need is built, including each stage's ball-prediction engine, and each stage logs the engines it uses. The **Training** tab has no engine control. |
| Debug rendering | The engine in the selected checkpoint's saved setup. Imported models use their current agent definition. |
| Param Testing | The baseline checkpoint's engine and ball predictor. |
| Model benchmarking | The engine you choose in **Test Setup > Simulation Engine**, for every model. Each model keeps its own ball predictor. For more information, see [Model benchmarking](ModelBenchmarking.md). |
| RLBot | The real game supplies the physics; the model keeps its own ball predictor. A runner built without an engine that the model needs refuses to start. |

Observations use the ball-prediction engine even when `useBallPrediction` is off,
so the queue checks that engine too.

## What a checkpoint remembers

Each segment's `training_setup.json` records the engine, the ball-prediction engine, and whether ball prediction is on.

- Continuing a checkpoint, branching from one, and later queue stages all keep the checkpoint's engine settings,
  even if you've since changed the agent's defaults.
- Checkpoints saved before RocketForge recorded an engine count as v2.
- A checkpoint with malformed engine settings is an error, rather than a guess.

Loading an old setup never rewrites it on disk.

## Engine differences

The two engines aren't physically identical, and their trajectories differ.
The following differences are known, and deliberately left as they are.
If your rewards or observations read one of these fields, treat it as engine-specific.

| Field or behavior | v2 | v3 |
|---|---|---|
| `obsOnGround` | `isOnGround && !isJumping`, the RLBot-style flag that turns false as soon as a jump starts. | `isOnGround`. |
| `supersonicTime` | Time spent supersonic. | Always 0. v3 keeps a grace timer for the time since dropping below supersonic speed, which isn't exposed. |
| `tickCountWhenExtraImpulseApplied` (ball hit info) | Set when the extra-impulse hit applies. | Never set (0). |
| Setting the ball in Hoops | Restarts the launch delay. | Doesn't restart the launch delay. |
| Prediction after a same-tick ball set | Stale until the next step. | Refreshed immediately. |
| Cars set overlapping each other | Stays finite. | Some overlaps turn both cars' physics state into NaN after one step. Space spawned cars at least two hitbox bounding radii apart. |
| `flipTime` | Time since the flip started; 0 outside a flip. | Keeps counting after the flip until the car lands. |
| Tick rate | Configurable per arena. | 120 Hz only. Creating a v3 arena at another rate throws an exception. |

No observation builder or reward in this repository reads `flipTime` or `supersonicTime`.
In RLBot play, `obsOnGround` comes from the game's `isOnGround`.

v3 also doesn't support v2-only solver options, custom boost-pad layouts, or mutators.
It rotates the ball natively by default.
Setting a game state teleports objects and doesn't restore the solver's caches.

`ForgeTrainingTests/SimEngineDifferences_tests.cpp` checks the first seven differences,
so a change in either engine shows up as a test failure.

## Supported game modes

Soccar is the validated game mode.
Heatseeker and Snowday are exercised in v3's Rust tests on Soccar geometry.
Hoops needs its own collision meshes, and Hoops and Dropshot gameplay aren't validated.
