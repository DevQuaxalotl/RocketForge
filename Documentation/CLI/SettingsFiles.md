# Settings files

The console trainer reads four JSON files.
ForgeGUI reads and writes the same files, so you can edit them by hand, with the GUI, or both.
The commands that use them are in the [console trainer guide](ConsoleTrainer.md).

| File | Location | Read by |
|---|---|---|
| `forge_settings.json` | `Bots/<bot>/<Agent>/forge_settings.json`, next to the agent's `Models/` folder | `train` and every command that resolves a default build |
| `forge_debug_settings.json` | `Bots/<bot>/<Agent>/forge_debug_settings.json`, next to `forge_settings.json` | `debug` |
| `forge_model_benchmark_settings.json` | The repository root | `model-benchmark` |
| `static_opponents.json` | `Bots/<bot>/<Agent>/Models/<BuildFolder>/static_opponents.json` | `train`, `debug --opponent`, `list static-opponents` |

Some rules apply to all of them:

- **Flags never write these files.**
  A one-off flag such as `--no-metrics`, `--stage` or `debug --speed 2` changes that run only.
- **A queue run records its own progress** in `forge_settings.json`:
  the segment each stage created, targets that a skip rewrote, and the reset on completion.
  Everything else in the file stays as you wrote it, except that unknown keys are dropped (see the next rule).
- **Unknown keys are ignored on load and lost on the next save.**
  This includes `_comment` keys: JSON has no comments, and nothing round-trips keys it doesn't know.
- **The GUI rewrites a file whenever you change it there.**
  Edit a file by hand while the GUI is closed, or reload afterwards.
- **A missing field takes its default.**
  The tables below give each default, so a hand-written file can stay short.
- **Values come from `list`.**
  The last column of each table names the `list` command that prints the legal values.

## `forge_settings.json`

The agent's training setup: the build it trains, global switches, and the queue of stages `train` runs.
`train --file <path>` reads a different copy; the agent's `forge_settings.example.json` is the template the first run copies.

A minimal two-stage queue (the file takes no comments, so the notes follow the block):

```json
{
  "schemaVersion": 1,
  "agentName": "example-agent",
  "buildName": "ExampleBuild",
  "globals": {
    "metricsEnabled": false
  },
  "stages": [
    {
      "stageName": "example-ball-chase",
      "targetTotalTimesteps": 100000000
    },
    {
      "stageName": "example-basic-score",
      "targetTotalTimesteps": 300000000
    }
  ]
}
```

- `buildName` selects the build the queue trains.
- `targetTotalTimesteps` is the total-timestep counter at which a stage hands over to the next.
  The counter carries across stages, so a stage that starts at 100M with a target of 300M trains for 200M steps.
- Each stage after the first bootstraps from the segment the stage before it created.
- `train --preview` checks the file and prints what each entry would do.

### Top level

| Field | Meaning | Allowed values | Default | Values from |
|---|---|---|---|---|
| `schemaVersion` | File format version. | `1` | `1` | |
| `agentName` | Agent the queue trains. | A registered agent | none (required) | `list agents` |
| `buildName` | Build the queue trains. | Any build name; empty is not allowed | `V1` | `list builds` |
| `globals` | Switches for the whole queue. | Object, below | see below | |
| `stages` | The queue, run in order. | At least one stage object | none (required) | |

### `globals`

| Field | Meaning | Allowed values | Default |
|---|---|---|---|
| `metricsEnabled` | Send metrics to W&B. `--no-metrics` turns it off for one run. | `true`, `false` | `true` |
| `skillTestingEnabled` | Track the model's skill rating during training; `false` turns the skill tracker off. | `true`, `false` | `true` |
| `pruneSourceCheckpoints` | After a stage bootstraps from an earlier one, delete all but the newest checkpoint of the source. Destructive. `train --prune-source` turns it on for one run. | `true`, `false` | `false` |
| `persistSegmentSnapshotsDuringTraining` | Keep every autosave as a permanent checkpoint instead of only the final save. | `true`, `false` | `false` |
| `performanceCoresOnly` | Windows hybrid CPUs: run on the performance cores only. `--performance-cores` turns it on for one run. | `true`, `false` | `false` |
| `render` | Rendered training in the GUI. The console never renders a queue run; use `debug`. | `true`, `false` | `false` |
| `debug` | The GUI's render and debug options, in the shape described under [`forge_debug_settings.json`](#debug-options). | Object | defaults |

### `stages[]`

| Field | Meaning | Allowed values | Default | Values from |
|---|---|---|---|---|
| `stageName` | Stage to train. | A registered stage | none (required) | `list stages` |
| `targetTotalTimesteps` | Total timesteps at which this stage ends. Must be greater than the previous stage's target. | Integer above 0 | none (required) | |
| `branchName` | Branch the stage's segments belong to. | Any name | `main` | |
| `warmStartPolicy` | Warm-start the policy on a new segment. | `true`, `false` | `true` | |
| `warmStartIterations` | Iterations the warm start lasts. | Integer, 0 or more | `20` | |
| `keepRunId` | Keep the source W&B run id when bootstrapping, instead of clearing it. | `true`, `false` | `false` | |
| `allowSegmentSetupMismatch` | Keep training a segment even when the stage's setup no longer matches the one it was created with, instead of starting a new segment from it. | `true`, `false` | `false` | |
| `versionIncrement` | How a new segment is numbered. | `none`, `minor`, `major` | `none` | |
| `versionOverride` | Force a specific segment number, for example `{ "major": 2, "minor": 0 }`. It must not collide with an existing segment. | `null` or `{ major, minor }` | `null` | `list segments` |
| `versionNotes` | Notes saved with a new segment. | Text | empty | |
| `staticOpponentsEnabled` | Train against the build's selected static opponents. | `true`, `false` | `false` | `list static-opponents` |
| `bootstrap` | How the queue's first stage picks its starting point. Ignored on later stages. | Object, below | `new` | |
| `parameterOverrides` | Reward weights or hyperparameters replaced after the stage builds its defaults. | Array of `{ kind, group, name, value }`; `kind` is `rewardWeight` or `hyperparameter` | none | |
| `createdSegmentId` | The segment this entry created or continues. The runner fills it in; edit it only to point the entry at another segment. | Segment id, or empty | empty | `list segments` |

### `stages[].bootstrap`

| Field | Meaning | Allowed values | Default | Values from |
|---|---|---|---|---|
| `mode` | Starting point of the first stage. `new` trains a fresh segment, `continueLatest` continues the build's active segment, `continueExisting` continues `selectedExistingVersionId`, and `loadFromOtherStage` starts a new segment from `sourceStage` + `sourceVersionId`. | `new`, `continueLatest`, `continueExisting`, `loadFromOtherStage` | `new` | |
| `branchMode` | Whether the first stage stays on its branch or starts another. | `continueExisting`, `continueSelected`, `createNew` | `continueExisting` | |
| `selectedExistingVersionId` | Segment to continue; required by `continueExisting`. | Segment id or `null` | `null` | `list segments` |
| `sourceStage` | Stage of the source segment; required by `loadFromOtherStage`. | A registered stage or `null` | `null` | `list stages` |
| `sourceVersionId` | Source segment id; required by `loadFromOtherStage`. | Segment id or `null` | `null` | `list segments` |

`continueExisting` and `loadFromOtherStage` check that their ids are present before anything runs.
A `branchMode` of `continueSelected` or `createNew` needs a non-empty `branchName`.

## `forge_debug_settings.json`

The GUI Debug tab's selection and options, one file per agent.
`debug` starts from this file and applies its flags on top for that run, so the file is the default `debug` renders when you pass nothing.
Without the file, `debug` renders the build's active segment on its trained stage.
The console never writes the file.

A minimal file that renders a trained segment, then one that renders an imported model:

```json
{
  "schemaVersion": 1,
  "buildName": "PreReleaseTesting",
  "segmentId": "v1.07",
  "renderStageName": "example-basic-score",
  "debug": {
    "gameSpeed": 2.0,
    "forceSingleStateSetter": true,
    "forceStateSetterIndex": 1,
    "teamSizeOverrideIndex": 0
  }
}
```

```json
{
  "schemaVersion": 1,
  "buildName": "",
  "segmentId": "Nexto",
  "renderStageName": "example-ball-chase"
}
```

An **imported model** is saved as an empty `buildName` plus the imported folder's name in `segmentId`.
`debug --imported <folder>` writes nothing, but produces the same selection for that run,
and `--segment` or `--build` switch back to a trained segment.

### Selection

| Field | Meaning | Allowed values | Default | Values from |
|---|---|---|---|---|
| `schemaVersion` | File format version. | `1` | `1` | |
| `buildName` | Build to render. Empty means an imported model. | Build name, or empty | empty | `list builds` |
| `segmentId` | Segment to render, or, with an empty `buildName`, the imported folder name. Empty means the build's active segment. | Segment id or folder name | empty | `list segments`, `list imported-models` |
| `renderStageName` | Stage to render on. Empty uses the segment's trained stage. An imported model has none, so it needs one. | A registered stage, or empty | empty | `list stages` |
| `showRewards` | GUI reward visualisation. The console ignores it and never shows rewards. | `true`, `false` | `false` | |

### Debug options

These live under `debug` (and under `globals.debug` in `forge_settings.json`).

| Field | Meaning | Allowed values | Default | Values from |
|---|---|---|---|---|
| `playerIndex` | Player the camera starts on. | Integer, 0 or more | `0` | |
| `gameSpeed` | Game speed, for example `2` for double speed. | Number above 0 | `1.0` | |
| `teamSizeOverrideIndex` | Force one team size. It is a position in the stage's weighted team-size list, not a size. `-1` means the stage's own mix. | `-1` or a listed index | `-1` | `list stage-options` |
| `forceSingleStateSetter` | Render with one state setter instead of the stage's mix. | `true`, `false` | `false` | |
| `forceStateSetterIndex` | The state setter to force. It counts the active (non-zero-weight) setters from 0. | Listed index | `0` | `list stage-options` |
| `maxEpisodeTimeEnabled` | Cut each episode off after `maxEpisodeTime`. | `true`, `false` | `false` | |
| `maxEpisodeTime` | Episode length in game seconds. | Number above 0 | `30.0` | |
| `includeStaticOpponent` | Play against the build's static opponents. `debug --opponent none` turns it off, any other value turns it on. | `true`, `false` | `false` | |
| `staticOpponentSelectionKey` | Pin one static opponent. Empty picks one at random each episode. | A key, or empty | empty | `list static-opponents` |
| `normMode`, `displayMode`, `forceDebugAll`, `debugEnabledRewards`, `hasRewardSelection`, `showCriticHeads`, `debugEnabledCriticHeads`, `hasCriticHeadSelection` | Reward and critic visualisation. GUI only; the console accepts them and does nothing with them. | As saved by the GUI | GUI defaults | |

## `forge_model_benchmark_settings.json`

The Model Benchmark tab's setup: which models play, how many games, and when to stop.
`model-benchmark` reads it, or the file named by `--settings <path>`, and writes the same workbook the GUI exports.
The GUI rewrites this file whenever you change the Model Benchmark tab, so keep a hand-written roster in a separate file and pass `--settings`.

A minimal file, with two models picked from the saved segments:

```json
{
  "run1v1": true,
  "run2v2": false,
  "run3v3": false,
  "numArenas": 64,
  "maxGamesPerMode": 200,
  "selectedModels": [
    { "agentName": "example-agent", "buildName": "PreReleaseTesting", "versionId": "v1.05" },
    { "agentName": "example-agent", "buildName": "PreReleaseTesting", "versionId": "v1.07" }
  ]
}
```

### Run options

| Field | Meaning | Allowed values | Default | Console flag |
|---|---|---|---|---|
| `engineVersion` | Simulator the games run on. | `2`, `3` | `2` | |
| `numArenas` | Games run in parallel. | Integer above 0 | `500` | `--arenas` |
| `maxGamesPerMode` | Game budget per team size. | Integer above 0 | `500` | `--max-games-per-mode` |
| `run1v1`, `run2v2`, `run3v3` | Which team sizes to play. | `true`, `false` | `true` | |
| `stochasticKickoff` | Sample legal policy actions until the first ball touch, then use deterministic actions. | `true`, `false` | `false` | |
| `kickoffTemperature` | Sampling temperature before the first touch. Lower favors preferred actions; higher adds variety. | Positive finite number when sampling is enabled; GUI range 0.01 to 10.0 | `0.70` | |
| `seed` | Reproducible seed. `0` picks a new one each run. | Integer | `0` | `--seed` |
| `runUntilMaxGames` | Play the full budget. `false` stops once the stopping rules below are met. | `true`, `false` | `true` | `--run-until-max`, `--stop-early` |
| `stopWhenRankingKnown` | A stopping rule used when `runUntilMaxGames` is off: stop once the ranking is known to `betterConfidence`. | `true`, `false` | `false` | |
| `betterConfidence` | Confidence level the ranking rule needs before it calls one model better. | 0 to 1 | `0.95` | |
| `eloRangeConfidence` | Confidence level of each reported Elo range. | 0 to 1 | `0.95` | |
| `maxEloUncertaintyRange` | A stopping rule used when `runUntilMaxGames` is off: the Elo range width to reach. | Number above 0 | `50` | |
| `collectOptionalStats` | Collect gameplay statistics. | `true`, `false` | `true` | `--optional-stats`, `--no-optional-stats` |
| `selectedModels` | The roster. | Array of entries, below | none | |

The console flags change one run and are never saved.

**Library-only options are ignored.**
`fillMissingGames`, `matchSavedConditions`, `libraryPath` (when it differs from the GUI's default),
`reuseSelectedEngine`, `reuseEngineVersions`, `reuseEnvironmentKeys` and `ratingTargetsOnly` belong to the GUI's results library, which the console doesn't use.
When one is set to a non-default value, the command prints a warning and carries on.
`allMatchups`, `presetName`, `stageFilter`, `hideImportedModels` and `includeOtherSimulatorResults` are GUI view settings,
and are ignored without a warning.

### `selectedModels[]`

An entry needs only `agentName`, `buildName` and `versionId`.
The command fills in the rest from storage: the segment's newest checkpoint and its `training_setup.json`.

| Field | Meaning | Allowed values | Default | Values from |
|---|---|---|---|---|
| `agentName` | Agent the model belongs to. | A registered agent | none (required) | `list agents` |
| `buildName` | Build the segment is in. An empty or missing value means the default build, `V1`. | Build name | `V1` | `list builds` |
| `versionId` | Segment id, or the folder name for an imported model. | Segment id or folder name | none (required) | `list segments`, `list imported-models` |
| `sourceLabel` | Set to `Imported Model` to mark an imported model. | Text | empty | |
| `included` | Set `false` to leave the entry out of this run. | `true`, `false` | `true` | |
| `resultsOnly` | A saved result with no model to run, from the GUI library. The console skips it. | `true`, `false` | `false` | |
| `primary` | Primary models play every selected model, and non-primary models play only primary ones. `--all-primary` overrides it. | `true`, `false` | `true` | |
| `checkpointPath` | Checkpoint folder to use instead of the segment's newest. | Path | empty | `list checkpoints` |
| `trainingSetupPath` | `training_setup.json` to use instead of the segment's. | Path | empty | |
| `agentSetupSource` | `savedSegment` uses the saved setup, and `currentAgent` uses the agent's code as it is now. | `savedSegment`, `currentAgent` | `savedSegment` | |
| `stageName`, `notes`, `lastTrainedAt`, `latestTimesteps`, `colorIndex`, `fingerprint`, `libraryModelId`, `agentSetupJson`, `skillRating`, `hasSkillRating` | Descriptive fields the GUI fills in. The console doesn't need them. | As saved | empty | |

Roster rules:

- **Imported models** are marked by `"sourceLabel": "Imported Model"` or by `"buildName": "Imported Model"`,
  with the imported folder's name as `versionId`.
  This differs from `forge_debug_settings.json`, which marks an imported model with an empty `buildName`.
  Here an empty `buildName` means the default build.
- **Names must be unique.**
  Each model is named `agent / build / versionId`, so one model can be selected once; a duplicate is an error (exit code 2).
- **A model with no saved setup** runs on the agent's current code, the way the GUI does.
- **A missing or unreadable settings file**, an unknown agent or segment, and a roster with fewer than two included models are errors (exit code 2).
- **Exports** land in `<repo>/Model Benchmarks/model-benchmark-<yyyyMMdd-HHmmss>.xml`, or at `--out`.
  Benchmarking on a rented pod isn't supported.

## `static_opponents.json`

The build's static opponents: other models the trained agent plays against during training and in `debug`.
One file per build, at `Bots/<bot>/<Agent>/Models/<BuildFolder>/static_opponents.json`.
The GUI's static opponent dialog writes it; you can also write entries by hand.

A segment opponent and an imported opponent, with only the `source_*` fields:

```json
{
  "static_opponents": [
    {
      "source_agent": "example-agent",
      "source_build": "PreReleaseTesting",
      "source_version_id": "v1.05"
    },
    {
      "source_agent": "example-agent",
      "source_build": "",
      "source_version_id": "Nexto"
    }
  ]
}
```

The `"source_build": ""` on the second entry is what marks it as an imported model.
Leaving the key out doesn't: a missing `source_build` defaults to `V1`, and the entry is read as a segment of the build named `V1`.

### Fields

| Field | Meaning | Allowed values | Default | Values from |
|---|---|---|---|---|
| `schema_version` | File format version. The runner writes it. | `2` | `2` | |
| `target_agent`, `target_build` | The agent and build the file belongs to. Written by the GUI; optional. | Names | empty | |
| `static_opponents` | The opponents. | Array of entries | none | |
| `source_agent` | Agent that owns the opponent's model. | A registered agent | none (required) | `list agents` |
| `source_build` | Build the opponent segment is in. Empty (`""`) marks an imported model. | Build name or `""` | `V1` when the key is missing | `list builds` |
| `source_version_id` | Segment id, or the folder name for an imported model. | Segment id or folder name | none (required) | `list segments`, `list imported-models` |
| `source_stage` | Stage the segment trained on. Filled in from the segment. For an imported model it is `__imported_models__`. | A registered stage | filled in | `list stages` |
| `display_name` | Name shown in lists. | Text | filled in | |
| `checkpoint_path` | Checkpoint the opponent plays with. | Path | newest checkpoint of the segment | |
| `agent_setup` | The opponent's network and observation setup. | Object | read from the segment's `training_setup.json` | |

Missing `source_stage`, `display_name`, `checkpoint_path` and `agent_setup` are filled in when the file is loaded.
The file on disk isn't changed by loading.
`list static-opponents` shows the resolved entries and the `key` that `debug --opponent <key>` takes.

When a stage that trains against static opponents starts, the trainer rewrites the file with the filled-in entries,
each pointing at its source's newest checkpoint.

An entry that can't be resolved is skipped with a warning, so training still starts without it:

```text
[StaticOpponents] Skipping entry <agent> / <build> / <id>: <reason>.
See `ForgeConsoleTrainer list segments --agent A --build B` (or `list imported-models --agent A`) for what exists.
```

An entry also needs `source_agent` and `source_version_id`, and the source agent must be registered in the build.
A skipped entry stays in the file exactly as you wrote it, after the filled-in ones, so it is used once its source is on the machine.
Training fails when no entry can be resolved.
`sync-manifest` lists a skipped entry's source folder too, so a remote run downloads it before it resolves the file.

## Where these files meet the CLI

- `train` reads `forge_settings.json`, and, for static opponents, `static_opponents.json`.
- `debug` reads `forge_debug_settings.json`, and `static_opponents.json` for `--opponent`.
- `model-benchmark` reads `forge_model_benchmark_settings.json`.
- `models-path`, `build-info`, `sync-manifest` and `merge-segment-ids` read `forge_settings.json` for the build and its queue.
  `merge-segment-ids --from <path>` copies the `createdSegmentId` values of a remote run's copy of `forge_settings.json` into the local file
  and leaves your other edits alone.
- The model and build folders themselves are described in `CLAUDE.md` under "Model / Build Storage".
