# Model benchmarking

Model benchmarking plays models against each other in the simulator and ranks them with Elo ratings.
Use it to check whether a new segment beats the last one,
or how your bot compares with public bots such as Nexto.

This page shows you how to run a benchmark in ForgeGUI, read and reuse its results, and run the same benchmark from the console.

## Key terms

- **Library:** a database file that holds your benchmark results and presets.
  Every game you play is stored there, and later runs reuse those games instead of playing them again.
  You can keep separate libraries for separate projects.
- **Comparison set:** the models in a benchmark, with a role for each one.
  A *preset* is a comparison set and its test setup, saved by name in the library.
- **Role:** what a model does in the next run.
  - **Target** models are the ones you're measuring. They play every other model.
  - **Opponent** models play the targets, and play each other only when **All Matchups** is on.
  - **Results only** models contribute their saved games but don't play new ones,
    so they don't need a local checkpoint.
- **Show:** whether a model appears in the displayed results, independently of its role.

## Run a benchmark in ForgeGUI

1. Open the **Model Benchmarking** tab.
1. Choose a library.
   If none is open, use the **Library** menu to create one or open an existing one.
1. Add models to the comparison set.
   Click **Add Models…**, filter by agent, build, branch, or stage, and click a row to add it.
   The picker also offers the latest autosave of each segment.
   To add the public bots, see [Public inference agents](../../Bots/PublicBots/README.md).
1. Set each model's role: **Target**, **Opponent**, or **Results only**.
   **All Primary** and **None Primary** set every model's role at once.
1. Click **Test Setup…**, review the settings, and click **Apply**:
   - **Simulation Engine:** the simulator that plays new games, RocketSim v2 or v3.
     For more information, see [Simulation engines](SimulationEngines.md).
   - **Mode:** 1v1, 2v2, 3v3, or all of them.
   - **Target Games per Matchup:** the game budget for each pair of models in each mode. The default is 500.
   - The seed and the execution settings, such as the number of parallel games.
   - **Compatibility:** which saved games count. For more information, see [Reuse saved games](#reuse-saved-games).
1. Optional: click **Review Schedule…** to see which matchups saved games already cover, which will be played, and which are blocked.
1. Click **Run Benchmark**.
   Before it starts, the benchmark checks that the match engine and every model's ball-prediction engine are built.
1. Watch the results update as games finish.
   To end early, click **Stop**.
   The run keeps every finished game.

Runs save to the library automatically when they complete or stop.
If a save fails, the results stay in memory, and **Retry Save** tries again.

## Read the results

The results combine the library's saved games with the live games of the current run.
Each view refreshes only while it's selected.

| View | What it shows |
|---|---|
| **Overall Ranking** | Each model's Elo rating and its confidence range. |
| **Kickoff Breakdown** | Results split by kickoff. |
| **Pairwise Summary** | Wins, losses, and draws for each pair of models. |
| **Detailed Stats** | Gameplay statistics, such as ball touches, bumps, and demolitions, when the run collects them. |
| **Match Coverage** | How many games each matchup has, against its budget. |
| **Environment Coverage** | Which simulators, environments, and ball-prediction setups the games came from. |
| **Details** | The individual game records. |

When models aren't connected through any chain of games, the ranking shows Elo as unavailable,
because their ratings can't be compared.

## Save and reuse a setup

- To save the comparison set and its test setup, click **Save** or **Save As…**.
  To switch presets, choose one under **Saved comparison** and click **Load**.
  If you have unsaved changes, ForgeGUI asks first.
- To start again from an earlier run, click **Use Previous Run…** and select the run.
  It restores the run's models, their roles, and its test setup.
- To share results, click **Export to Excel** or **Export Results…**.
  **Import Excel…** adds a supported export to the active library.

## Choose each model's setup

Each model plays with one of two agent setups:

- **Trained** restores the setup that the model was trained with, from its checkpoint.
  This is the usual choice.
- **Current** uses the agent definition compiled into the current build,
  including its ball-prediction settings, and ignores the checkpoint's saved setup.
  Use it to test a changed agent definition against an old checkpoint.
  Changing a C++ agent definition requires a rebuild.

Imported models without a saved training setup use **Current**.

A preset or previous run remembers the setup each model actually used.
If that setup no longer matches its source, the model's label adds "(saved version)",
and its tooltip lists the differences.
Choosing **Trained** or **Current** again replaces the remembered setup with the one the source has now.

The engine selected in **Test Setup** plays every new game.

### Kickoff sampling

Models play deterministic actions by default.
To add variety to kickoffs, enable **Stochastic kickoff inference** in **Test Setup**.
Each model then samples legal policy actions until the first car-ball touch,
and plays deterministic actions for the rest of the game.
Sampling stays off after that first touch, even if the ball returns to the center, and starts again for the next game.

**Kickoff temperature** controls how varied the sampled actions are:
lower values favor the policy's preferred actions, and higher values add variety.
The GUI accepts 0.01 to 10.0, and the default is 0.70.

Presets, run setups, and exports save both settings,
and the console uses the same `stochasticKickoff` and `kickoffTemperature` settings.
A fixed seed also seeds the sampling, but a run reproduces only with the same models, engine, timing, device, and execution settings.

## Reuse saved games

A saved game counts toward a matchup's budget only when it matches the exact checkpoint and setup that the run uses,
so a benchmark never plays the same game twice and never mixes in games from a different model.
**Fill Missing Games** plays only the games that the budget still needs.
Completed draws count toward the budget.

**Test Setup > Compatibility** controls which saved games count, and which appear in the results:

- By default, every environment of the selected simulator counts, including future revisions of it.
- Games from the other simulator count only when you select that simulator too.
- To narrow reuse to exact recorded environments, clear **Include all environments**, and choose them.
  For example, choose a recorded kickoff-sampling environment to reuse only games played with that sampling setup.

Results recorded before RocketForge supported RocketSim v3 count as v2 games.
Libraries from older versions upgrade automatically when you open them, and keep all of their games, models, and presets.
New runs record the simulator's revision, the car order, and each model's ball-prediction engine.
That record describes where a game came from; it doesn't make v2 and v3 physics equivalent.

### How models are identified

A benchmark identifies each model by a fingerprint of its checkpoint and setup files,
and verifies every fingerprint before a run.
For an imported scripted policy, the fingerprint covers `INFERENCE_POLICY.json` and the weight file it declares.
A public policy without a `.lt` file has no file fingerprint,
so the benchmark identifies it by its name and path instead, and marks it as unverified.

## Run a benchmark from the console

`ForgeConsoleTrainer model-benchmark` runs the benchmark described in `forge_model_benchmark_settings.json` at the repository root,
and writes the same workbook that ForgeGUI exports:

```powershell
ForgeConsoleTrainer model-benchmark --max-games-per-mode 200
```

For the file's fields, see [Settings files](../CLI/SettingsFiles.md).
For every option, see [`model-benchmark`](../CLI/ConsoleTrainer.md#model-benchmark).

## Related files

The `Documentation/ValidationRecords/` folder holds JSON records of earlier benchmark validation runs.
They're historical records from another machine, and they aren't maintained.
