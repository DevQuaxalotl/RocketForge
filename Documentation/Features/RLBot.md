# Play your bot in Rocket League with RLBot

[RLBot](https://rlbot.org/) runs bots in real Rocket League matches.
RocketForge ships an RLBot runner that loads a trained segment and plays it with the same observation builder,
action parser, and action delay that it trained with.

This page shows you how to build the runner, export a segment so that RLBot can find it,
choose the inference device, and package a bot for the RLBot bot pack.

## How it fits together

| Piece | Where it lives | What it is |
|---|---|---|
| RLBot runner | `RLBot.exe`, built by the `rlbot` preset | One executable that contains every agent compiled into RocketForge. |
| Agent RLBot template | `Bots/<bot>/<Agent>/rlbot/` | The `bot.toml`, `loadout.toml`, `device.txt`, and `run.bat` that every export of the agent starts from. Edit the bot's loadout and display details here. |
| Segment export | `Models/<Build>/segments/<Segment>/rlbot/` | A self-contained bot folder for one segment: the runner, the agent setup, and launch scripts that point at the segment's newest checkpoint. |
| Discovery configs | `Bots/<bot>/<Agent>/rlbot_exports/` | One `.bot.toml` per exported segment. Add this folder to RLBot, and every export of the agent appears in RLBot's bot list. |
| Submission package | A folder you choose | A package for the RLBot bot pack, built with Bob the Bot Builder. |

## Before you begin

- Install Rocket League and [RLBot v5](https://rlbot.org/).
- Build RocketForge and train at least one segment.
  For a walkthrough, see [Get started with RocketForge](../GettingStarted.md).

## Build the RLBot runner

The runner has its own configure preset, `rlbot`, so it builds in its own tree, `out/build/rlbot`.
Build it with the `build-rlbot-all-agents` build preset.
That preset builds the runner and then updates every agent's `rlbot` folder automatically:
it copies in the new `RLBot.exe` and regenerates `run.bat`.
Exports start from that folder, so every agent picks up the new runner without any manual copying.

Build it in one of the following ways:

- **Visual Studio or Rider:** open the repository folder, select the **RLBot** configure preset,
  and then build with the **RLBot, copied into every agent** build preset.
- **Command line:** in an **x64 Native Tools Command Prompt for VS 2022**, from the repository root, run:

  ```powershell
  cmake --preset rlbot
  cmake --build --preset build-rlbot-all-agents
  ```

Rebuild after you change agent code, or after you add a bot with `create-bot`,
so that the runner contains the current agents.
The `build-rlbot` preset builds only the runner in `out/build/rlbot`, and leaves the agents' folders unchanged.

The runner is built with CUDA support by default (`RLBOT_USE_GPU=ON`),
and that build still runs on the CPU when you ask it to.
To build a CPU-only runner, configure with `cmake --preset rlbot -DRLBOT_USE_GPU=OFF`.

## Export a segment

An export makes one segment playable in RLBot.

1. Export the segment.
   Without `--segment`, the command exports the build's active segment:

   ```powershell
   ForgeConsoleTrainer export-rlbot --agent example-agent
   ```

   To export another segment, find its id with `ForgeConsoleTrainer list segments`, and pass it with `--segment`.
   In ForgeGUI, open the **File Management** tab, select the segment, and click **Export to RLBot**.
1. Check the export.
   `ForgeConsoleTrainer list segments` shows exported segments in its `exported` column.

The export copies the runner and the agent's RLBot template into the segment's `rlbot` folder,
and writes a discovery config to the agent's `rlbot_exports` folder.

While the segment keeps training, each checkpoint save refreshes its export:
the launch script moves to the new checkpoint, and the bot's name updates to the new timesteps.
You don't need to export again after more training.

To remove an export, run `export-rlbot` with `--remove`.
Both directions take the build lock, so they refuse while a trainer holds the build.

## Add your bots to RLBot

You add the folder once per agent.
After that, every segment you export from that agent appears in RLBot.

1. Open RLBot.
1. Open the **Add/Remove** dialog.
1. Add the agent's `rlbot_exports` folder, for example `Bots\ExampleBot\ExampleAgent\rlbot_exports`.
1. Find your bot in the bot list.
   Its name combines the agent, the build, and the segment's timesteps.
1. Add the bot to a team, and start the match.

## Choose the inference device

Each export runs on the GPU by default.
The device comes from the export's `device.txt`, which holds `gpu` or `cpu`.

- To change the default for future exports, edit the agent's `Bots/<bot>/<Agent>/rlbot/device.txt`.
- To change one export, edit `device.txt` in that export's `rlbot` folder.
- To override the device for one launch, pass `--device cpu` or `--device gpu` to the export's `run.bat`.
- When you start `RLBot.exe` directly, the `RLBOT_USE_GPU` build option sets the default.

The runner validates the device before it connects to RLBot, and warms up the policy at the team's batch size.
If the GPU fails to initialize, the runner reports the error and exits; it never falls back to the CPU silently.

## Team batching (hive mind)

RLBot can run matching teammates in one process.
RocketForge then evaluates the shared policy once per decision for the whole team,
instead of once per car.
Each car still keeps its own observation, action mask, previous action, action delay, and kickoff playback.
Team batching changes how inference runs; it doesn't train a team policy.

Team batching is on by default in the runner, in RLBot exports, and in submission packages.
[RLBot groups bots with the same agent ID on the same team](https://wiki.rlbot.org/v5/botmaking/hiveminds/),
so give different models or versions different IDs.
Each opposing team gets its own process, and a single car works with batching on.

To turn it off for an agent, set `hivemind = false` in the agent's `rlbot/bot.toml`, and export again.
The export keeps an explicit `hivemind = false`, and passes `--hivemind off` to the runner.
For a submission package, clear the team batching option in the export dialog.

When you read development traces, note that `inference_us` is the whole batch's duration repeated for each car,
so don't add it up across teammates.

## Runner options

`run.bat` passes extra arguments on to `RLBot.exe`, so you can append any of these options to an export's launch command.

| Option | Effect |
|---|---|
| `--device gpu\|cpu` | Inference device for this launch. |
| `--hivemind on\|off` | Team batching. The default is `on`. |
| `--model-dir <folder>` | Checkpoint folder to load, or a folder of numbered checkpoints, from which the newest is used. |
| `--agent-setup <file>` | Agent setup snapshot to apply. Exports pass their `agent_setup.json`. |
| `--capture-dir <folder>` | Record the match for offline replay. Off by default. |
| `--replay <file>` | Replay a recording offline, without RLBot. Needs `--model-dir` and `--capture-dir`. |
| `--list-agents` | Print the agents compiled into the runner. |
| `--help` | Print the options. |

`ROCKETFORGE_MODEL_DIR` sets the checkpoint folder when you don't pass `--model-dir`.

For recording and replay, see [RLBot match recording and offline replay](../../tools/rlbot/capture/README.md).

## Package a bot for the RLBot bot pack

A submission package contains only your bot's inference sources, its checkpoint, and its build scripts,
with no dependency on a RocketForge checkout.
You build it with [Bob the Bot Builder](https://wiki.rlbot.org/).

1. Export the package:

   ```powershell
   ForgeConsoleTrainer export-submittal --agent example-agent --out C:\RLBotSubmissions
   ```

   The command prints the package path.
   In ForgeGUI, open the **File Management** tab, select the segment, and click **Export for Submittal**.
1. Build the package with Bob.
   A CPU package builds with a plain `bob build`, as the bot pack runs it.
   To build locally with a LibTorch of your own, run `python scripts/build_bob.py --libtorch "C:/path/to/libtorch"` in the package folder.

Packages target the CPU by default, because the RLBot bot pack builds only CPU bots,
and the agent's `device.txt` doesn't change that default.
CPU packages run on Windows and Linux.

To build a GPU package for Windows with an NVIDIA GPU,
pass `--device gpu`, or select the GPU option in the export dialog, and build it with `scripts/build_bob.py`.
GPU packages use a shared CUDA LibTorch, following
[GGLBot's bob-gpu branch](https://github.com/SubparN0va/GGLBot/tree/bob-gpu).
Packages never bundle the LibTorch DLLs.
The package's own README explains where the runtime looks for them, and the `LIBTORCH_CUDA_ROOT` and `LIBTORCH_CPU_ROOT` overrides.

To change a package's device after you export it, edit its `rlbot/device.txt`, and rebuild the package.

## Troubleshooting

### `RLBot export executable is missing for <agent>`

The agent's `rlbot` folder has no `RLBot.exe`.
Build the `build-rlbot-all-agents` preset, as described in [Build the RLBot runner](#build-the-rlbot-runner), and export again.

### The bot doesn't appear in RLBot

Check that you added the agent's `rlbot_exports` folder, not the agent's `rlbot` folder or a segment folder,
and that `rlbot_exports` contains a `.bot.toml` for the segment.

### `Failed to construct agent '<agent>'`

The runner was built before the agent existed.
Rebuild the `build-rlbot-all-agents` preset.

### `This bot can't run in this RLBot build`

The segment trained on a simulation engine that this runner wasn't built with, such as RocketSim v3.
Install that engine's prerequisites, and rebuild the runner.
For more information, see [Simulation engines](SimulationEngines.md).

### `RLBot GPU initialization failed`

CUDA or the CUDA build of LibTorch isn't available.
Install them, or set the export's `device.txt` to `cpu`.

## Run the RLBot regression tests

1. Configure the `rlbot` preset with tests enabled:

   ```powershell
   cmake --preset rlbot -DRLBOT_BUILD_TESTS=ON
   ```

1. Build, and run the tests with the `RLBot` CTest label:

   ```powershell
   cmake --build --preset build-rlbot
   ctest --test-dir out/build/rlbot -L RLBot
   ```

To also run the policy and packet checks on CUDA, set `ROCKETFORGE_TEST_GPU=1`.

To compile a generated submission package as a separate check,
set `ROCKETFORGE_SUBMITTAL_TEST_PACKAGE_DIR` to a new, absolute scratch folder,
and run the `SubmittalExport.WritesPortableBobPackage` test.
That package holds placeholder checkpoints for packaging checks, not a playable model.
