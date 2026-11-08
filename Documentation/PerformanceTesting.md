# Performance testing

This page shows you how to measure training speed (steps per second, SPS) reliably, and how to find where the time goes.
Everything here uses `ForgeConsoleTrainer training-benchmark`.

## Use a trained baseline

Benchmark against a trained policy, never a fresh one.
A fresh policy rarely touches the ball, so episodes are short and resets dominate the timing.
Train a ball-chase stage (such as ExampleBot's `example-ball-chase`) for roughly 50-100M steps first;
after that, episodes run about 900-1300 steps and the step cost matches real training.

Keep that build as your reference, for example a `PerfTesting` build whose segment `v1.00_example_ball_chase` holds the 50M-step checkpoint.

`--checkpoint-path` must point at the segment's `checkpoints` folder, not a numbered checkpoint folder.
A numbered folder prints "No checkpoints found, starting new model" and silently benchmarks a random policy.

## Recent opponents

`--recent-opponents` reads versions from `<build>/recent_policy_versions/v1.00_<timesteps>/`,
each holding a `POLICY.lt` (copy one from the segment's checkpoints) and a `STATS.json`.
Four versions are enough.

The stored `policy_compatibility` must match the running trainer exactly.
On MSVC that means the `class RLGC::...` typeid names, with the obs builder recorded as `class RLGC::ProfiledObsBuilder`.
The simplest template is an existing `STATS.json` from a real run; this one is from a 512×3 agent:

```json
{"policy_compatibility": {"action_delay":7,"action_parser_policy_key":"class RLGC::DefaultAction","action_parser_type":"class RLGC::DefaultAction","game_mode":0,"max_team_size":3,"models":[{"activation_type":0,"add_layer_norm":true,"add_output_layer":true,"layer_sizes":[512,512,512],"name":"policy","num_inputs":295,"num_outputs":90,"optimizer_type":0}],"num_actions":90,"obs_builder_type":"class RLGC::ProfiledObsBuilder","schema_version":1,"tick_skip":8,"uses_action_mask":true}, "source_version_id": "v1.00", "timesteps": 10000000}
```

## Windows scheduling

On hybrid CPUs (P-cores plus E-cores) Windows can demote the mostly-waiting collection thread to an E-core or throttle it,
which in some runs roughly halved collection SPS.
Pass `--performance-cores`:
it restricts the process to the performance cores, disables power throttling, and sizes the collection pool to match.
The startup log confirms it with `restricted process to 16 CPUs`, `power throttling disabled` and `Collection thread pool: 16 worker threads`.
It replaces manual `start /affinity` and `RLGC_NUM_THREADS` pinning.

## Running the benchmark

Build with the `build-console` preset (configure preset `forge`), and run from the repo root (the checkpoint path is relative to it):

```
out\build\forge\ForgeConsoleTrainer.exe training-benchmark --agent example-agent --stage example-ball-chase --graphs --recent-opponents --warmup-iterations 3 --iterations 4 --timesteps 250000 --arenas 256 --microbatch 50000 --performance-cores --checkpoint-path Bots\ExampleBot\ExampleAgent\Models\PerfTesting\segments\v1.00_example_ball_chase\checkpoints
```

`--warmup-iterations 3` drops the first iterations from the summary,
because every arena starts fresh and episode length is still ramping up.
Read the `Steady state` line: overall, collection and consumption SPS.
`training-benchmark` always disables static opponents, so a stage's static-opponent arenas are not measured.

## Comparing two builds

Run-to-run noise is large: overall SPS varied about 7% across three back-to-back runs,
consumption SPS occasionally dropped from about 340k to 280k, and pinned runs have spread about 17%.
Three sequential runs against a recorded baseline cannot resolve a gain of a few percent, so compare interleaved:

1. Copy the pre-change build's runtime folder aside as arm A, then rebuild as arm B.
2. Run A B A B A B A B (four per arm), one at a time, with nothing else CPU-heavy running.
3. Compare the arm means of the `Steady state` line.

Use collection SPS as the primary metric for collection-side changes;
consumption SPS is the noisy one and drags overall SPS with it.

Keep a change only when it shows a measured benefit.
The ball-prediction revalidation skip was implemented, measured at collection 177514 vs 177366 SPS (-0.1%), and dropped for that reason.

## Finding where time goes

Use the Visual Studio profiler, or temporary scoped timers around the sections in question.
Never commit the timers; remove them once you have your measurements.
