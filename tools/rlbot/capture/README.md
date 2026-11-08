# RLBot match recording and offline replay

Record what your bot sees and decides during a live RLBot match,
then replay those same inputs offline, without Rocket League,
and compare the decisions.
Use it to check that a bot makes the same choices on the CPU and the GPU,
or with and without team batching.

A recording holds, for each RLBot callback:
the game packet, the observation, the legal-action mask, the policy's action probabilities,
the selected action, the control schedule, and the controls returned to RLBot.
A replay runs those callbacks through the same RLBot runtime that played the match.

Recording is off by default.
It works in the development runner, in segment exports, and in submission packages.

## Record a match

1. Add `--capture-dir` and an output folder to the bot's launch command.
   In the bot's `bot.toml`, append it to the existing `run_command`, and keep the rest of the command as it is:

   ```toml
   run_command = 'run.bat --capture-dir "C:/rlbot-captures/my-match"'
   ```

   For a submission package, edit the package's `bot.toml` the same way.
1. Start the match from RLBot as usual.
   RLBot supplies the connection settings, so you can't record by starting the bot from a terminal on its own.
1. Stop the bots from RLBot when the match ends.
   A normal stop lets the recorder write the rest of its queue and a footer that marks the file complete.
1. Remove `--capture-dir` from `run_command` before you play normal matches.

Each car's runtime writes its own `callbacks.rfcapture` file in a new, uniquely named folder under the output folder.
The recorder never overwrites an existing file.

Recordings contain game packets, player names, and model observations.
Keep them in a local folder, and don't add them to source control or to a submission package.
Submission exports never include recordings.

## Replay a recording

Replay with the same agent and checkpoint that played the match,
using a build that has the same collision meshes and ball-prediction support.

1. Replay the recording on the CPU:

   ```bat
   RLBot.exe --replay "C:/rlbot-captures/my-match/capture-ID/callbacks.rfcapture" --model-dir "C:/models/checkpoint" --capture-dir "C:/rlbot-captures/replay-cpu" --device cpu
   ```

   `--model-dir` is the checkpoint folder.
   For a submission package, it's the package's `000-runtime` folder,
   and you run the launcher from a package built for the matching device.
1. To compare devices, replay again with `--device gpu` and a new output folder.
   The GPU replay needs CUDA; it never falls back to the CPU.

The replay picks the agent named in the recording, and applies the recorded agent setup unless you pass `--agent-setup`.
It refuses a different setup or checkpoint.
It reads the checkpoint only from the path you pass, never from a path inside the recording.

A team recording replays as one team-batched group by default.
To replay each car through its own runtime, add `--hivemind off`.
Each car then keeps its own policy state across the whole packet sequence,
which shows any real difference between team-batched and individual inference.
A single-car recording can't reconstruct a teammate that it didn't record.

The replay never connects to a server or starts the game.
It doesn't run development drills, so a match recorded with drills or other state-changing extensions shows expected differences.
Use an ordinary match to compare the standard runtime.

## Inspect and compare recordings

`capture.py` needs only the Python standard library.

- To summarize one recording:

  ```bat
  python tools/rlbot/capture/capture.py inspect "C:/rlbot-captures/my-match/capture-ID/callbacks.rfcapture"
  ```

- To compare a recording with a replay, and write JSON and HTML reports:

  ```bat
  python tools/rlbot/capture/capture.py compare "C:/rlbot-captures/my-match/capture-ID/callbacks.rfcapture" "C:/rlbot-captures/replay-cpu" --json "C:/rlbot-captures/comparison.json" --html "C:/rlbot-captures/comparison.html"
  ```

Submission packages include the same tool as `scripts/rlbot_capture.py`.

The second argument to `compare` can be one file, several single-car files,
or a folder that holds exactly one replay of each recorded car.
A missing or duplicate car, or different input packets, is an error,
so the tool never treats recordings of different matches as equivalent.
It matches decisions by callback number and car index, not by game frame number.

The comparison treats values as follows:

- Controls, masks, action IDs, and scheduling must match exactly.
- Observations must match within `1e-6`. To change the tolerance, pass `--obs-atol`.
- Probabilities must match within `1e-5`. To change the tolerance, pass `--probs-atol`.

Each report keeps the difference counts and the first 100 differences.
The HTML report is a single self-contained file.
The tool never overwrites an existing report.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | The comparison matched, or the inspection completed. |
| `1` | The comparison found differences. |
| `2` | A recording or the comparison is invalid. |
| `3` | The recording is an incomplete prefix, and the prefix matched or inspected cleanly. |

### Incomplete recordings

A recording that the bot couldn't finish, for example because the process was killed, has no footer.
Its verified start is still readable, and the tool always reports it as incomplete.

To replay or compare only that verified start, add `--allow-incomplete`.
It accepts a missing footer or a cut-off last line,
but never a checksum error, a missing callback, a lost queue entry, or a corrupt complete record.
A replay of an incomplete prefix produces a complete output for that prefix,
and the comparison still reports that the original recording is incomplete.

## What a replay proves

A replay checks the bot's *decisions* against recorded inputs.
It doesn't test physics:
the recorded game states stay fixed, so a different action in the replay doesn't change the states that follow it.

- Packets are copies of the decoded FlatBuffer data at callback entry,
  including RLBot's ball prediction and initialization data when present.
  They aren't the original socket bytes,
  and their order describes one car's runtime, not the network arrival order across all runtimes.
- The recorded controls are the ones returned to RLBot, not proof that Rocket League applied them.
- The observation RNG state is recorded at every callback, without reseeding or changing live behavior,
  so a replay needs the same RNG implementation.
  An agent that uses other global or random state can differ, and the report shows those differences.
- Each recording stores the simulator family, the public agent setup, the device, and checksums of the checkpoint files.
  CRC32 checksums and file sizes catch accidental changes; they aren't a cryptographic authenticity check.

## Recording overhead

Each car's recorder has a 16 MiB queue, and it writes to disk at least every 250 ms while the disk keeps up.
Memory use doesn't grow with match length.
If the queue overflows or a disk write fails, the bot keeps playing, the recorder logs a warning,
and the recording can't be replayed completely.

Encoding JSON and reading GPU diagnostics back still take time on the callback thread,
so the inference timings in a recording are diagnostic, not a clean speed benchmark.
Turn recording off for competitive matches and for performance measurements.

## Tests

- `RLBotCapture.RoundTripPreservesPolicyControlsAndTiming` records and replays through the production runtime,
  with synthetic checkpoints, action macros, missing players, and callbacks without a ball.
- `RLBot.CaptureReports` covers corrupt recordings, missing callbacks, interrupted files, row matching, tolerances, and report escaping.

No live match or real checkpoint ships as a test fixture.
To run the tests, see [Run the RLBot regression tests](../../../Documentation/Features/RLBot.md#run-the-rlbot-regression-tests).
