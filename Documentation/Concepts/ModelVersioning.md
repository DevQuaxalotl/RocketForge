# Model versioning

RocketForge versions trained agents build-first.
A *build* is the named training history of one agent version.
A *training segment* is one continuous stretch of training inside a build,
and most builds contain many segments over their lifespan.
Segments are numbered as revisions, own their checkpoints and setup snapshot,
and can point back to a parent segment when they branch from an existing checkpoint.

The versioning system answers the following questions:

- Which build produced this model?
- Which training segment and stage produced each checkpoint?
- What training setup did that segment use?
- Which segment is the active continuation?
- Which models are eligible for benchmarking, for use as static opponents, or for RLBot export?

## Agent models folder

Each agent has a stable `Models` folder, which holds the build storage:

```text
<agent>/
  forge_settings.json
  forge_settings.example.json
  Models/
    builds.json
    V1/
      build.json
      segments/
```

`ResolveAgentModelsFolder()` resolves this root from either the agent name or the agent's runtime model path.
That way, training metadata stays stable even when runtime paths point at generated build output.

## Build

A build is the top-level training history for an agent version.
A build typically starts at 0 timesteps and continues until you consider that bot version complete.
It can be mostly linear,
or it can branch when you want to test different training setups from the same checkpoint.

When `buildName` isn't set, the build is named `V1`.

A build stores the following items:

- `build.json`: persistent build metadata.
- `segments/`: one folder for each training segment.
- `recent_policy_versions/`: the registry and files of recent policy versions, for self-play and version play.
- `skill_benchmarks/`: benchmark artifacts.
- `static_opponents.json`: the registry of selected static opponents.
- `rlbot_exports.json`: the RLBot export registry.

`Models/builds.json` is the agent-level build registry.
It lists an agent's known builds and marks one as active for GUI selection.

## Training segment

A training segment is one continuous training span in a build.
It has one segment revision ID, one stage name, one setup hash, one checkpoint folder,
one optional parent segment, and one metadata file.

Segments are stored as follows:

```text
Models/
  V1/
    segments/
      1.0_BallChase/
        segment.json
        training_setup.json
        checkpoints/
          1000000000/
        temp_checkpoints/
```

The segment folder name is `<segment_id>_<stage_name>`, with unsafe folder characters sanitized.

### Segment revision

Segment IDs use the shared GigaLearn segment revision format:

```text
1.00
1.01
1.02
2.00
```

When RocketForge creates a segment, it normally allocates the next minor revision.
A stage queue entry can request a major increment instead,
or provide an explicit `versionOverride` with `{ major, minor }`.

### Training stage

A training stage is the curriculum implementation that a segment trains with,
such as `BallChase`, `BasicScore`, or another registered stage.
The segment's `training_setup.json` file and `setup_hash` capture the stage's exact setup.
For more information, see [Training stages](../Features/TrainingStages.md).

### Training setup snapshot

Every segment saves a `training_setup.json` file next to its `segment.json` file.
The snapshot records the following information:

- The agent, stage, segment revision, save time, and run name.
- The reward table entries, weights, ranges, categories, and effective maximum values.
- The environment, PPO and PPG, saving, metrics, normalization, and static-opponent settings.

RocketForge computes the `setup_hash` from the setup snapshot.
Volatile fields, such as the save time, run name, and version ID, are excluded from the hash,
so ordinary resume bookkeeping doesn't change it.

## Branch

Segment metadata represents a branch with the following fields:

- `parent_segment_id`: the segment that this segment branched from.
- `source_checkpoint_path`: the checkpoint that started the branch.
- `branch_name`: a user-facing label, `main` by default.

The checkpoint is the reproducible source of the branch.
Timesteps are useful for display, but the checkpoint path is the durable source reference.

## Active segment and active path

`build.json` stores `active_segment_id`, which updates when the queue saves progress for the current segment.
To reconstruct the active path, start at `active_segment_id`
and follow each segment's `parent_segment_id` back to the root segment.

A segment's `status` controls whether it appears in normal model lists:

| Status | Meaning |
|---|---|
| `active` | Visible and available in normal segment lists. |
| `inactive` | Kept on disk, but hidden unless a list requests inactive segments. |

`branch_name` and `parent_segment_id` describe lineage;
`status` describes curation and visibility.

## Storage layout

Schema version 1 uses the following build layout:

```text
Models/
  builds.json
  Imported Models/
    <ImportedModelFolder>/
      imported_model.json
  <BuildFolder>/
    build.json
    static_opponents.json
    rlbot_exports.json
    recent_policy_versions/
      recent_policy_versions.json
    skill_benchmarks/
      benchmarks.json
    segments/
      <SegmentId>_<StageName>/
        segment.json
        training_setup.json
        checkpoints/
          <TotalTimesteps>/
        temp_checkpoints/
          <TotalTimesteps>/
        rlbot/
```

Checkpoint folder names use the global total-timestep counter.
For example, a segment that starts at `6000000000` and ends at `7000000000` trained for `1000000000` timesteps.

## Summary

A build is the complete version history of one agent version,
and a training segment is one continuous training span inside that build.
Together, segment revisions, parent links, setup hashes, training setup snapshots, checkpoints, and curation flags
describe how a model was produced and how it can be used.

When the setup changes, RocketForge protects reproducibility by creating a child segment,
unless you explicitly allow the mismatch.
When you explore branches, their trained timesteps still count toward the build,
and the active path is reconstructed from `active_segment_id` and `parent_segment_id`.
