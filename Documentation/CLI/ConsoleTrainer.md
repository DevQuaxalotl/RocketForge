# Console trainer

`ForgeConsoleTrainer` is the terminal front end to RocketForge.
It trains, inspects and manages models, renders and benchmarks them, and exports them, with no GUI.
It reads the same JSON settings files as ForgeGUI (see [Settings files](SettingsFiles.md)), so the two stay interchangeable.

This page is the command guide.
Each command's section repeats its `--help` text exactly, so `ForgeConsoleTrainer <command> --help` and this page always agree.

## Running it

| Platform | Executable |
|---|---|
| Windows | `out\build\forge\ForgeConsoleTrainer.exe` (build with `tools\build\build_and_test.ps1 -BuildGui`, or `cmake --build --preset build-console` inside the x64 VS2022 developer environment) |
| Linux / Docker | `ForgeConsoleTrainer` (on `PATH` inside the training image) |

The examples below write `ForgeConsoleTrainer` for either one.
Run them from the repository root.

## Grammar

```text
ForgeConsoleTrainer <command> [options]
```

- Options are named, and nothing depends on their order after the command name.
- With no command, or when the first argument is an option, the command is `train`.
  So `ForgeConsoleTrainer` and `ForgeConsoleTrainer --no-metrics` both run the saved queue.
- `ForgeConsoleTrainer --help` lists the commands, and `ForgeConsoleTrainer help <command>` or `<command> --help` prints one command's options.
- The old shapes (`<agent>`, `<agent> <stage>`, `new-stage`, `--debug`, `--list-agents`, `--list-stages`) no longer exist.
  Use `train --agent ...`, `train --stage ...`, `train --stage ... --from-stage ...`, `debug` and `list agents|stages` instead.

### Shared options

These names mean the same thing in every command that takes them.

| Option | Meaning |
|---|---|
| `--agent <name>` | Agent to work on. |
| `--build <name>` | Build within the agent. |
| `--segment <id>` | Segment id within the build, for example `v1.3`. |
| `--imported <folder>` | Imported-model folder name (an alternative to a build and segment). |
| `--stage <name>` | Training stage. |
| `--checkpoint <ts>` | Checkpoint timesteps within the resolved segment. |
| `--file <path>` | `forge_settings.json` to read instead of the agent's. |
| `--out <path>` | Where a command writes its output. |
| `--yes` | Skip a confirmation. |
| `--json` | Machine-readable output on stdout. |

### Defaults

Defaults are read-only: the CLI never writes `forge_app_state.json`.

| Option | Default |
|---|---|
| `--agent` | The agent last selected in the GUI (`forge_app_state.json`), else `example-agent`. |
| `--build` | The `buildName` in the agent's `forge_settings.json`; empty means the build named `V1`. |
| `--segment` | The build's active segment, for commands that need one. |
| `--checkpoint` | The newest checkpoint. |

Pass `--agent` explicitly in scripts, so a GUI selection can't change what they do.

### What every command prints

Before a command acts, it prints the values it resolved, for example:

```text
agent: example-agent  build: PreReleaseTesting  segment: v1.07
```

Commands whose stdout is read by a program (`list --json`, `export-rlbot`, `export-submittal` and the four plumbing commands) print this line on stderr instead,
and leave stdout untouched.

### Errors and exit codes

Errors print as `<command>: <message>` on stderr.

| Exit code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Runtime failure, such as a missing checkpoint or a failed save. |
| `2` | Usage or validation error, such as an unknown option, an unknown name, or a refused confirmation. |

An unknown value fails and names the `list` command that supplies it, with up to three closest matches.
For example, `ForgeConsoleTrainer list segments --agent exmple-agent` prints:

```text
list: unknown agent 'exmple-agent'. Did you mean: example-agent? Run `ForgeConsoleTrainer list agents`
```

### What gets written

Command-line flags are one-off.
They never write to any JSON file, so a `--no-metrics` or `--stage` run leaves your saved setup alone.

A normal queue run is different: it records its own progress in `forge_settings.json`,
namely the segment id each stage created, targets that a skip rewrote, and the reset on completion.
Remote training depends on that record (see `merge-segment-ids` below).
`train --stage`, `--new` and `--from-stage` never touch `forge_settings.json`.

## Commands

| Group | Command | What it does |
|---|---|---|
| Bots | `create-bot` | Creates a new bot from ExampleBot. |
| Training | `train` | Runs the saved stage queue, or one stage forever. |
| Inspect | `list` | Lists agents, stages, builds, segments, checkpoints, stage options, static opponents or imported models. |
| Models | `delete-segment`, `prune` | Deletes a segment, or all but the newest checkpoint of a segment or build. |
| Models | `export-rlbot` | Exports a segment into an RLBot folder, or removes the export. |
| Models | `export-submittal` | Packages a segment for Bob the Bot Builder. |
| Debug & analysis | `debug` | Renders a model in RocketSimVis. |
| Debug & analysis | `reward-table` | Prints a reward comparison matrix for a stage. |
| Debug & analysis | `reward-analysis` | Measures each reward's discounted return contribution. |
| Debug & analysis | `action-trace` | Traces a policy's actions to CSV. |
| Benchmarking | `training-benchmark` | Measures training speed with timing buckets. |
| Benchmarking | `model-benchmark` | Plays models against each other and exports the GUI workbook. |
| Script plumbing | `models-path`, `build-info`, `sync-manifest`, `merge-segment-ids` | Small commands the remote-training scripts call. |
| Help | `help [command]`, `--help`, `<command> --help` | Prints help. |

## Stop and skip

While `train` runs:

| Input | Effect |
|---|---|
| First Ctrl+C | Saves a checkpoint and stops the whole queue. On Linux this is SIGINT or SIGTERM. On Windows, Ctrl+C, Ctrl+Break and closing the console window all count. |
| Second Ctrl+C | Terminates immediately, without waiting for the save. |
| `Q` | Saves and stops the queue (interactive terminal only). |
| `S` | Saves and skips the current stage, then continues with the next queued stage (interactive terminal only). |
| `kill -USR1 <pid>` | Skips the current stage, like `S`, for runs with no keyboard. Linux only. In Docker, use `docker kill -s USR1 CONTAINER`; the entrypoint forwards it to the trainer. |

The same first-signal stop works in Docker: `docker stop -t 60 CONTAINER` sends SIGTERM, which the entrypoint forwards.
On Windows, closing the console window, logging off or shutting down requests the same stop and then waits up to about 4.5 seconds for the save,
because the operating system kills the process at about five.
That save is best-effort: one that takes longer is cut off, so prefer Ctrl+C there.

`debug` also stops on Ctrl+C, and on Windows on `q` while the viewer has focus.

## Command reference

Each usage block below is the command's `--help` output.

### Bots

#### `create-bot`

Copies ExampleBot into `Bots/<Name>`, renamed throughout, so a new bot starts from a working agent, stages and RLBot folder.
It refuses a name that matches an existing bot folder or registered agent, and writes nothing in that case.

```text
Usage: ForgeConsoleTrainer create-bot <Name>

Creates Bots/<Name> from ExampleBot: the agent <Name> and its stages
<name>-ball-chase, <name>-basic-score and <name>-advanced, renamed throughout.
Training state, models and RLBot binaries are not copied.

<Name> is letters and digits, starting with a letter (at most 32 characters),
and must not match an existing bot folder or agent.

Afterwards re-configure (cmake --preset forge) and rebuild before using the new agent.
```

### Training

#### `train`

Runs the saved stage queue, or one stage on its own.
The queue is the `stages` list in the agent's `forge_settings.json` ([Settings files](SettingsFiles.md#forge_settingsjson)).
Flags such as `--no-metrics` and `--performance-cores` apply to this run only.

```text
Usage: ForgeConsoleTrainer [train] [options]

Runs the agent's saved stage queue from forge_settings.json (created from the example template on first run).
With --stage, trains that one stage forever instead of running the queue.
A queue run records its own progress (created segment ids, completion) in forge_settings.json;
one-off flags and --stage runs never write it.

Options:
  --agent <name>        Agent to train (default: the GUI-selected agent, else example-agent)
  --file <path>         forge_settings.json to use (relative paths resolve from the working directory)
  --stage <name>        Train this stage alone, ignoring the queue; continues the stage's newest segment in the build
                        (a new segment if the stage has none)
  --new                 With --stage: start a new segment from scratch instead of continuing
  --from-stage <name>   With --stage: start a new segment of the stage from that stage's newest segment in the build
  --keep-run-id         With --from-stage: keep the source run id instead of clearing it
  --prune-source        Delete the source checkpoints after copying them (stage bootstrap)
  --preview             Print the resolved values and one line per queue entry, then exit without touching storage
  --no-metrics          Disable training metrics for this run
  --performance-cores   Windows hybrid CPUs: run on the performance cores only, with power throttling off

Stopping: the first Ctrl+C saves and stops the whole queue; a second one terminates immediately.
In an interactive terminal, Q saves and stops the queue and S saves and skips the current stage.
On Linux, `kill -USR1 <pid>` skips the current stage.

Run `ForgeConsoleTrainer list stages` for the stage names.
```

`--preview` resolves everything and prints one line per queue entry, then exits without creating or changing anything in storage.
A queue run starts from the saved setup, so edit the file (or the GUI queue) to change what it trains.
`--stage S` is the quick-test path: it never reads the queue and never writes `forge_settings.json`.
Its segment rules:

| Flags | Segment trained |
|---|---|
| `--stage S` | Continues S's newest segment in the build, or starts a new one if S has none. |
| `--stage S --new` | Starts a fresh segment. |
| `--stage S --from-stage T` | Starts a new S segment from T's newest segment in the build. |

`--keep-run-id` is valid only with `--from-stage`.
Stopping and skipping are covered under [Stop and skip](#stop-and-skip).

### Inspect

#### `list`

Prints what exists, so you know which value to pass to another command.
Every `Unknown ...` error from another command names the `list` subject that supplies the value.

```text
Usage: ForgeConsoleTrainer list <subject> [options]

Subjects:
  agents            Registered agents; * marks the default for --agent
  stages            Registered training stages
  builds            The agent's builds, with total timesteps
  segments          The build's segments; * marks the active segment and the ones exported to RLBot
  checkpoints       The segment's checkpoints and autosaves
  stage-options     A stage's state setters and team sizes, the values `debug --state-setter` and `--team-size` take
                    (team-size index = position in the stage's weighted list; -1 in forge_debug_settings.json means no override)
  static-opponents  The build's static opponent selection, with the keys `debug --opponent` takes
  imported-models   The agent's imported models, with the folder names `--imported` takes

Options:
  --agent <name>    Agent to read (default: the GUI-selected agent, else example-agent)
  --build <name>    Build to read (default: the build in forge_settings.json)
  --segment <id>    Segment for `checkpoints` (default: the build's active segment)
  --stage <name>    Stage for `stage-options` (required); filters `segments`
  --file <path>     forge_settings.json that names the default build
  --json            Print JSON on stdout and the resolved values on stderr
```

```text
ForgeConsoleTrainer list segments --agent example-agent --build PerfTesting --stage example-ball-chase
```

```text
id     stage               branch    timesteps  checkpoints  last trained     exported  active
-----  ------------------  --------  ---------  -----------  ---------------  --------  ------
v1.00  example-ball-chase  baseline  10003968   1            20260702-184954
v1.04  example-ball-chase  main      10006528   1            20260830-183635
```

- `list agents` marks the default with `*`; `list builds` shows each build's total timesteps.
- `list segments` marks the build's `active` segment and the ones that have an RLBot export on disk (`exported`).
- `list stage-options --stage S` is the source of the values `debug --state-setter` and `--team-size` take.
  The state-setter index counts the active (non-zero-weight) state setters from 0 and is what `forceStateSetterIndex` stores.
  The team-size index is the position in the stage's weighted list and is what `teamSizeOverrideIndex` stores; `-1` means no override.
- `list static-opponents` prints the keys `debug --opponent` takes.
- `list imported-models` prints the folder names `--imported` takes.
  Models are imported in the GUI; see [Importing models](../Features/ImportingModels.md).
- `--json` prints a JSON array on stdout and the resolved values on stderr, so `list ... --json | jq` works.

### Models

#### `delete-segment`

Removes one segment completely.
It prints what it will delete and asks `Continue? [y/N]`.

```text
Usage: ForgeConsoleTrainer delete-segment --segment <id> [options]

Permanently deletes one segment, its checkpoints and its RLBot export. Asks to confirm.

Options:
  --segment <id>    Segment to delete (required; see `list segments`)
  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)
  --build <name>    Build (default: the build in forge_settings.json)
  --file <path>     forge_settings.json that names the default build
  --yes             Skip the confirmation (required when stdin is not a terminal)

Refuses while a trainer holds the build.
```

Without a terminal on stdin and without `--yes`, it refuses and exits with code 2.
It also refuses while a trainer holds the build.

#### `prune`

Frees disk space by keeping only each segment's newest checkpoint.
Segment metadata, benchmarks and exports are kept.

```text
Usage: ForgeConsoleTrainer prune (--segment <id> | --all) [options]

Deletes every checkpoint of a segment except the newest, keeping its metadata. Asks to confirm.

Options:
  --segment <id>    Segment to prune
  --all             Prune every segment of the build
  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)
  --build <name>    Build (default: the build in forge_settings.json)
  --file <path>     forge_settings.json that names the default build
  --yes             Skip the confirmation (required when stdin is not a terminal)

Refuses while a trainer holds the build.
```

`prune --segment <id>` handles one segment and `prune --all` handles the whole build.
The confirmation, `--yes` and build-lock rules are the same as for `delete-segment`.

#### `export-rlbot`

Exports a segment into a self-contained RLBot folder.
`list segments` marks exported segments in its `exported` column.

```text
Usage: ForgeConsoleTrainer export-rlbot [options]

Exports a segment into a self-contained RLBot folder, or with --remove deletes that export.

Options:
  --segment <id>    Segment to export (default: the build's active segment)
  --remove          Remove the segment's RLBot export instead
  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)
  --build <name>    Build (default: the build in forge_settings.json)
  --file <path>     forge_settings.json that names the default build
```

Both directions take the build lock, so they refuse while a trainer holds the build.

#### `export-submittal`

Builds the package you upload to Bob the Bot Builder.
Stdout carries only the package path, so a script can capture it.
`--device` defaults to `cpu`, the only device the RLBot bot pack builds;
`gpu` packages must be built with the package's `scripts/build_bob.py`.

```text
Usage: ForgeConsoleTrainer export-submittal --out <folder> [options]

Packages a segment for Bob the Bot Builder inside the output folder and prints the package path on stdout.

Options:
  --out <folder>    Parent folder the package is created in (required)
  --overwrite       Replace an existing package
  --device <name>   cpu (default; the only device the RLBot bot pack builds) or gpu
  --segment <id>    Segment to package (default: the build's active segment)
  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)
  --build <name>    Build (default: the build in forge_settings.json)
  --file <path>     forge_settings.json that names the default build
```

### Debug and analysis

#### `debug`

Renders a model in the RocketSimVis viewer.
Which model and stage it renders comes from `forge_debug_settings.json` ([Settings files](SettingsFiles.md#forge_debug_settingsjson)),
and then your flags override it for this run.

```text
Usage: ForgeConsoleTrainer debug [options]

Renders a model in RocketSimVis, starting from the agent's forge_debug_settings.json (the GUI Debug tab's settings).
Without that file it renders the build's active segment on its trained stage.
Flags change this run only; the file is never written.
Reward and critic visualisation are GUI-only.

Options:
  --agent <name>           Agent to render (default: the GUI-selected agent, else example-agent)
  --build <name>           Build to render; alone, it renders that build's active segment
  --segment <id>           Segment to render (see `list segments`)
  --imported <folder>      Imported model to render instead of a segment (see `list imported-models`)
  --checkpoint <ts>        Checkpoint timesteps within the segment (default: newest; see `list checkpoints`)
  --stage <name>           Stage to render on (default: the segment's trained stage; required for imported models)
  --state-setter <i>       Force one state setter (index from `list stage-options`)
  --team-size <AvB>        Force one team size, e.g. 2v2 (from `list stage-options`)
  --player <i>             Player the camera starts on
  --opponent <key>         Static opponent key (see `list static-opponents`), `random` per episode, or `none`
  --episode-seconds <f>    End each episode after this much game time
  --speed <f>              Game speed, e.g. 2 for double speed
  --file <path>            forge_settings.json that names the default build

Press 'q' to quit (it works while the viewer has focus).
```

- On Linux the last line reads `Press Ctrl+C to quit.` instead, because the `q` key needs the Windows console.
- `--build` alone renders that build's active segment.
  `--imported` and `--stage` together render an imported model, which has no trained stage of its own.
- `--opponent <key>` pins a static opponent from the build's selection, `random` picks one per episode, and `none` turns static opponents off.
- Viewer keybinds (focus, speed, pause) go back to the render loop over UDP 9274.

#### `reward-table`

Shows each reward's weight, range and effective maximum, and how they compare.
It builds the stage without starting training.

```text
Usage: ForgeConsoleTrainer reward-table --stage <name> [options]

Prints a reward comparison matrix for a stage without starting training.

Options:
  --stage <name>       Stage to print (required; see `list stages`)
  --agent <name>       Agent the stage belongs to (default: the GUI-selected agent, else example-agent)
```

#### `reward-analysis`

Collects experience with the segment's newest durable checkpoint and reports each reward's contribution to the discounted return.
The result is written into the segment folder, where the GUI loads it.

```text
Usage: ForgeConsoleTrainer reward-analysis [options]

Runs a segment's newest durable checkpoint for N timesteps per team size and reports each reward's discounted return contribution.
The result is saved in the segment folder, where the GUI loads it.

Options:
  --agent <name>       Agent to analyse (default: the GUI-selected agent, else example-agent)
  --build <name>       Build to analyse (default: the build named by forge_settings.json)
  --segment <id>       Segment to analyse (default: the build's active segment; see `list segments`)
  --stage <name>       Stage to analyse on (default: the segment's trained stage)
  --timesteps <n>      Timesteps per team size (default: the stage's batch size)
  --file <path>        forge_settings.json that names the default build
```

#### `action-trace`

Replays a saved policy offline for debugging and parity checks against RLBot.
It writes a CSV of selected and applied actions plus a summary JSON.

```text
Usage: ForgeConsoleTrainer action-trace [options]

Runs a saved policy offline and writes selected policy actions plus per-tick applied controls to a CSV and a summary JSON.
The setup and checkpoint come from the resolved segment unless --setup and --checkpoint-path give them.

Options:
  --agent <name>           Agent to trace (default: the GUI-selected agent, else example-agent)
  --build <name>           Build to trace (default: the build named by forge_settings.json)
  --segment <id>           Segment to trace (default: the build's active segment; see `list segments`)
  --checkpoint <ts>        Checkpoint timesteps within the segment (default: newest; see `list checkpoints`)
  --file <path>            forge_settings.json that names the default build
  --setup <path>           training_setup.json or agent_setup JSON source, instead of the segment's
  --checkpoint-path <path> Folder containing POLICY.lt, instead of the segment's checkpoint
  --out <path>             CSV output path (default: action_trace.csv)
  --episodes <n>           Number of rollouts (default: 8)
  --max-ticks <n>          Physics ticks per rollout (default: 720)
  --team-size <n>          Cars per team for generated resets, 1-3 (default: 1)
  --seed <n>               Deterministic reset variation seed (default: 1234)
  --kickoff                Trace normal kickoff starts (default reset mode)
  --wall-probe             Use the original side-wall probe reset
  --stage-reset <name>     Use any training stage's state setter for each rollout
  --snapshot-csv <p>       Replay from an RLBot trace row's obs_state_snapshot
  --snapshot-tick <n>      physics_tick row to load from --snapshot-csv
  --snapshot-bot-index <n> Choose a bot_index when multiple rows share the snapshot tick
  --snapshot-car-id <n>    Choose a car_id when multiple rows share the snapshot tick
  --action-sequence-csv <p> Use selected_action_id rows from an RLBot trace CSV
  --control-replay-csv <p> Apply per-tick applied_* controls from an RLBot trace CSV
  --control-replay-input-delay-ticks <n> Delay replayed controls by n physics ticks
  --control-replay-use-last-input Use RLBot flatbuffer last_input controls for replay
  --control-replay-focus-policy   Drive the snapshot focus car with the offline policy
  --force-action <n>       Force one policy action id instead of model selection
  --stochastic             Sample actions instead of taking argmax
  --temperature <f>        Policy temperature when stochastic (default: 1)

Without a reset mode, action-trace uses a normal kickoff reset.
The CSV is generic; the summary JSON also includes wall-dash and speed-flip mining helpers when applicable.
```

Most options select a reset mode (`--kickoff`, `--wall-probe`, `--stage-reset`, or one of the RLBot snapshot and replay inputs).
`--checkpoint` is the checkpoint's timesteps within the resolved segment, and `--checkpoint-path` is a checkpoint folder on disk.

### Benchmarking

#### `training-benchmark`

Measures training throughput (steps per second) with detailed timing buckets.
Benchmark against a trained policy, not a fresh one; the method and its pitfalls are in [Performance testing](../PerformanceTesting.md).

```text
Usage: ForgeConsoleTrainer training-benchmark --stage <name> [options]

Runs a bounded training benchmark with detailed timing buckets.
Metrics, skill tracking, checkpoint writes and static opponents are always off.

Options:
  --stage <name>              Stage to benchmark (required; see `list stages`)
  --agent <name>              Agent the stage belongs to (default: the GUI-selected agent, else example-agent)
  --checkpoint-path <path>    Segment `checkpoints` folder to start from instead of a fresh policy
  --graphs                    Enable CUDA graphs (off by default so runs compare like for like)
  --recent-opponents          Train against the build's recent policy versions (needs --checkpoint-path)
  --warmup-iterations <n>     Iterations excluded from the steady-state figures (default: 3)
  --iterations <n>            Measured iterations (default: 4)
  --timesteps <n>             Timesteps per iteration (default: 8192)
  --arenas <n>                Parallel games (default: 128)
  --microbatch <n>            PPO micro-batch size; --timesteps must be a multiple of it (default: 4096)
  --seed <n>                  Random seed (default: 1234)
  --performance-cores         Windows hybrid CPUs: collect on P-cores only
```

#### `model-benchmark`

Runs the Model Benchmark from the console and writes the workbook the GUI exports.
The settings file format and the roster rules are in [Settings files](SettingsFiles.md#forge_model_benchmark_settingsjson).

```text
Usage: ForgeConsoleTrainer model-benchmark [options]

Runs the model benchmark described by forge_model_benchmark_settings.json and writes the same workbook the GUI exports.
Entries that give only agentName, buildName and versionId resolve to that segment's newest checkpoint.
Imported models use sourceLabel (or buildName) "Imported Model" and their folder name as versionId.
Models are named "agent / build / versionId", so each one can be selected once.

Options:
  --settings <path>           Benchmark settings file (default: forge_model_benchmark_settings.json at the repo root)
  --out <path>                Workbook to write (default: Model Benchmarks/model-benchmark-<timestamp>.xml)
  --arenas <n>                Override the number of parallel games
  --max-games-per-mode <n>    Override the game budget per team size
  --seed <n>                  Override the seed (0 picks a new one)
  --optional-stats            Collect gameplay statistics
  --no-optional-stats         Skip gameplay statistics
  --run-until-max             Play the full game budget
  --stop-early                Stop once the stopping rules are met
  --all-primary               Treat every model as primary
```

The override flags change this run only; they are never written to the settings file.

### Script plumbing

#### `models-path`

Prints the agent's `Models` folder, relative to the repository root, as the only line on stdout.
The sync scripts and the Docker entrypoint call it before a `forge_settings.json` exists.

```text
Usage: ForgeConsoleTrainer models-path [--agent <name>]

Prints the agent's Models directory relative to the repo root, as the only line on stdout.
Needs no forge_settings.json.

Options:
  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)
```

#### `build-info`

Prints `KEY=VALUE` lines the push scripts read.
The resolved values go to stderr, so stdout stays machine-readable.

```text
Usage: ForgeConsoleTrainer build-info [options]

Prints KEY=VALUE lines on stdout describing the build a queue run trains:
BUILD_NAME, BUILD_FOLDER, MODELS_REL, START_STAGE, START_SEGMENT, IS_NEW, TAG,
then one STATIC_OPPONENT_<i> line per opponent model folder outside the build.

Options:
  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)
  --build <name>    Build (default: the build in forge_settings.json)
  --file <path>     forge_settings.json to read (default: the agent's; it must exist)
```

#### `sync-manifest`

Prints the rclone filter list that push, pull and pod seeding copy.
A queue's autosaves under `temp_checkpoints/` are left out of the default list and listed by `--temp-checkpoints`.

```text
Usage: ForgeConsoleTrainer sync-manifest [options]

Prints an rclone filter list on stdout of the files the build's stage queue needs (for push, pull and pod seed).

Options:
  --agent <name>        Agent (default: the GUI-selected agent, else example-agent)
  --build <name>        Build to list, taken as given (default: the build in forge_settings.json)
  --file <path>         forge_settings.json to read (default: the agent's; it must exist)
  --metadata-only       List only build and segment metadata
  --temp-checkpoints    List only the queue segments' autosaves (temp_checkpoints/)
```

#### `merge-segment-ids`

After pulling a remote run's `forge_settings.json`, adopts the segment ids it recorded.
Every other local edit to the queue stays.

```text
Usage: ForgeConsoleTrainer merge-segment-ids --from <path> [options]

Copies the segment ids recorded in another copy of the agent's settings (a remote run's)
into the local forge_settings.json, keeping every other local edit.

Options:
  --from <path>     Settings file whose recorded segment ids are adopted (required)
  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)
  --file <path>     Local forge_settings.json to update (default: the agent's)
```

### Help

#### `help`

`ForgeConsoleTrainer help` prints the command list,
and `ForgeConsoleTrainer help <command>` prints that command's usage,
the same text as `<command> --help`.

```text
Usage: ForgeConsoleTrainer <command> [options]
With no command, or when the first argument is an option, runs `train`.

Bots:
  create-bot          Creates a new bot from ExampleBot

Training:
  train               Runs the saved stage queue, or one stage forever

Inspect:
  list                Lists agents, stages, builds, segments, checkpoints, stage options, static opponents or imported models

Models:
  delete-segment      Permanently deletes a segment and its checkpoints
  prune               Deletes all but the newest checkpoint of a segment, or of every segment
  export-rlbot        Exports a segment into a self-contained RLBot folder, or removes the export
  export-submittal    Packages a segment for Bob the Bot Builder

Debug & analysis:
  debug               Renders a model in RocketSimVis from forge_debug_settings.json, with one-run overrides
  reward-table        Prints a reward comparison matrix for a stage (does not start training)
  reward-analysis     Collects N timesteps of experience and reports the discounted return contribution of each reward
  action-trace        Traces a saved policy's selected and per-tick applied actions to CSV

Benchmarking:
  training-benchmark  Runs a bounded training benchmark with detailed timing buckets
  model-benchmark     Runs the model benchmark from forge_model_benchmark_settings.json and exports the GUI workbook

Script plumbing:
  models-path         Prints the agent's Models directory relative to the repo root
  build-info          Prints KEY=VALUE lines describing the build (folder, Models path, start segment, image tag)
  sync-manifest       Prints an rclone filter list of the files the build's stage queue needs (for push/pull/seed)
  merge-segment-ids   Copies the segment ids recorded in another settings file (--from) into the local forge_settings.json

Help:
  help [command]      Lists the commands, or prints one command's options (also: <command> --help)
```


## Workflows

### First run

```text
ForgeConsoleTrainer list agents
ForgeConsoleTrainer list stages
ForgeConsoleTrainer train --agent example-agent --preview
ForgeConsoleTrainer train --agent example-agent --no-metrics
```

1. `list agents` shows which agents are registered; `*` marks the default.
1. `list stages` shows the stage names a queue entry can use.
1. `train --preview` prints the queue as it would run, without touching storage.
1. `train` with no `--stage` runs the saved queue.
   The first time, with no `forge_settings.json`, it asks whether to create one from the agent's example template.
   Without a terminal, it creates the file without asking.
   `--no-metrics` skips W&B, so no API key is needed.

### Edit and preview a queue

The queue lives in the agent's `forge_settings.json`, one `stages` entry per stage (see [Settings files](SettingsFiles.md#forge_settingsjson)).
Edit the file, then check the plan:

```text
ForgeConsoleTrainer train --agent example-agent --preview
```

```text
agent: example-agent  build: PreReleaseTesting  settings: C:\RocketForge\Bots\ExampleBot\ExampleAgent\forge_settings.json
1. example-ball-chase  continue v1.07 (1.4M / 10.0M)
2. example-basic-score  new v1.08 from example-ball-chase v1.07
3. example-advanced  new v1.09 from example-basic-score v1.08
```

Each line is one queue entry: whether it continues a segment or starts a new one, which segment it starts from, and its progress against the target.
Settings errors (an unknown stage, targets that don't increase) are reported the same way, with exit code 2, before anything runs.
To try a different file without replacing the saved one, pass `--file <path>`.
That file must already exist: only the agent's own `forge_settings.json` is created from the example template.

### Quick-test one stage

```text
ForgeConsoleTrainer train --agent example-agent --stage example-ball-chase --no-metrics
```

`--stage` trains that stage forever, ignoring the queue, and never writes `forge_settings.json`.
By default it continues the stage's newest segment in the build, or starts a new segment if the stage has none.

```text
ForgeConsoleTrainer train --agent example-agent --stage example-ball-chase --new
ForgeConsoleTrainer train --agent example-agent --stage example-basic-score --from-stage example-ball-chase --keep-run-id
```

- `--new` starts a fresh segment.
- `--from-stage T` starts a new segment of the stage from stage T's newest segment in the build.
  `--keep-run-id` (only with `--from-stage`) keeps the source's W&B run id instead of clearing it,
  and `--prune-source` deletes the source checkpoints after copying.
- Add `--preview` to any of these to see the plan first.

### Debug a segment in another stage

Render a segment on a stage other than the one it trained on, pinning the state setter and team size:

```text
ForgeConsoleTrainer list stage-options --agent example-agent --stage example-basic-score
ForgeConsoleTrainer debug --agent example-agent --build PreReleaseTesting --segment v1.07 --stage example-basic-score --state-setter 1 --team-size 1v1 --speed 2
```

```text
kind          index  name                weight
------------  -----  ------------------  ------
state-setter  0      FuzzedKickoffState  1
state-setter  1      RandomState         0.5
team-size     0      1v1                 1
```

`--state-setter` takes the index from this listing, counted over the state setters that are active (non-zero weight).
`--team-size` takes the label (`1v1`).
Other flags pick a camera player, a static opponent (`--opponent <key>`, `random` or `none`; keys come from `list static-opponents`),
an episode length and a game speed.
The command starts from `forge_debug_settings.json` (the GUI Debug tab's settings) when it exists,
applies your flags on top for this run only, and never writes the file back.
Reward and critic visualisation are GUI-only.

To render an imported model, name its folder and a stage:

```text
ForgeConsoleTrainer list imported-models --agent example-agent
ForgeConsoleTrainer debug --agent example-agent --imported <folder> --stage example-ball-chase
```

### Benchmark two models and open the workbook

`model-benchmark` reads `forge_model_benchmark_settings.json` at the repository root,
plays the selected models against each other, and writes the same workbook the GUI exports.
A roster entry needs only an agent, a build and a segment id.
Write the file by hand, or let the GUI's Model Benchmark tab write it:

```json
{
  "run1v1": true,
  "run2v2": false,
  "run3v3": false,
  "maxGamesPerMode": 200,
  "numArenas": 64,
  "selectedModels": [
    { "agentName": "example-agent", "buildName": "PreReleaseTesting", "versionId": "v1.05" },
    { "agentName": "example-agent", "buildName": "PreReleaseTesting", "versionId": "v1.07" }
  ]
}
```

```text
ForgeConsoleTrainer model-benchmark
ForgeConsoleTrainer model-benchmark --settings my-benchmark.json --out results\two-models.xml --run-until-max
```

The ranking prints when the run ends:

```text
Rank  Model                                      Elo  Range        Record (W-L-NR)
----  -----------------------------------------  ---  -----------  ---------------
1     example-agent / PreReleaseTesting / v1.07  191  [-6, 387]    3-1-0
2     example-agent / PreReleaseTesting / v1.05  0    [-197, 197]  1-3-0
```

The workbook is an Excel XML file.
It lands in `Model Benchmarks/model-benchmark-<yyyyMMdd-HHmmss>.xml` at the repository root, or at `--out`.
Open it in Excel, or import it into the GUI's benchmark library with **Import Excel...** in the Model Benchmark tab.
Benchmarking on a rented pod isn't supported.
The roster rules, and the settings the console ignores, are in [Settings files](SettingsFiles.md#forge_model_benchmark_settingsjson).

### Export for RLBot and for Bob

```text
ForgeConsoleTrainer list segments --agent example-agent --build PreReleaseTesting
ForgeConsoleTrainer export-rlbot --agent example-agent --build PreReleaseTesting --segment v1.07
ForgeConsoleTrainer export-rlbot --agent example-agent --build PreReleaseTesting --segment v1.07 --remove
ForgeConsoleTrainer export-submittal --agent example-agent --build PreReleaseTesting --segment v1.07 --out submissions
```

- `list segments` marks the segment the build continues from (`active`) and the ones that already have an RLBot export (`exported`).
- `export-rlbot` writes a self-contained folder for the RLBot runtime; `--remove` deletes it again.
  Both take the build lock, so they refuse while a trainer is running on the build.
- `export-submittal` creates the Bob the Bot Builder package inside the `--out` folder
  and prints the package path on stdout, so a script can pick it up.
  `--overwrite` replaces an existing package.

### Create a bot

```text
ForgeConsoleTrainer create-bot MyBot
cmake --preset forge
cmake --build --preset build-console
ForgeConsoleTrainer list agents
ForgeConsoleTrainer train --agent MyBot --preview
```

1. `create-bot` copies ExampleBot to `Bots/MyBot` and names the agent `MyBot`.
1. Re-configure so the build picks up the new bot's folder, then rebuild.
1. `list agents` shows the new agent once the rebuilt executable runs.

### Prune a build

```text
ForgeConsoleTrainer list checkpoints --agent example-agent --build PerfTesting --segment v1.01
ForgeConsoleTrainer prune --agent example-agent --build PerfTesting --segment v1.01
ForgeConsoleTrainer prune --agent example-agent --build PerfTesting --all
ForgeConsoleTrainer delete-segment --agent example-agent --build PerfTesting --segment v1.00
```

`prune` keeps each segment's metadata and its newest checkpoint, and removes the rest.
`delete-segment` removes a segment entirely, including its checkpoints and its RLBot export.
Both print what they will remove and ask `Continue? [y/N]`.

- With no terminal on stdin (a pipe, a script, a pod), they refuse and exit with code 2 unless you pass `--yes`.
- They refuse while a trainer holds the build lock, so stop the queue first.
- `--yes` skips only the question; nothing checks whether another segment still refers to the one you delete.

### Remote training

On a rented GPU the same commands run inside the Docker image.
The pod's start command is the trainer's arguments,
so `train --agent example-agent --stage example-ball-chase --no-metrics` is a valid start command, and no command at all runs the saved queue.
Skip a stage with `docker kill -s USR1 CONTAINER`, and stop with `docker stop -t 60 CONTAINER`.
The sync scripts use `models-path`, `build-info`, `sync-manifest` and `merge-segment-ids` to decide what to copy.
See [Remote training](../RemoteTraining/RemoteTraining.md) for the full workflow.

## Related pages

- [Settings files](SettingsFiles.md): every JSON file the console reads, field by field.
- [Remote training](../RemoteTraining/RemoteTraining.md): Docker, R2 sync and the pod stop and skip controls.
- [Performance testing](../PerformanceTesting.md): measuring training speed with `training-benchmark`.
