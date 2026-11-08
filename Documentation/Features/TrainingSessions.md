# Training sessions

Training that you start from the GUI runs in the GUI process, on a worker thread,
using the same queue runner as `ForgeConsoleTrainer`.
The agent and checkpoint select the simulator and ball predictor.
For more information, see [Simulation engines](SimulationEngines.md).

Closing the GUI ends training.
There's no background or detached session to reconnect to.

## Stop, skip, and close

- **Save and stop** asks the learner to reach its next safe boundary, write a checkpoint, and end the whole queue.
- **Skip** does the same for the current stage, and then continues with the next queued stage.
- If you close the window while training runs, the GUI asks "Save and stop training before closing?".
  **Yes** starts a save and stop and closes the window once the save finishes.
  **Cancel** keeps training.
- The console trainer has the same controls: the first Ctrl+C (SIGTERM or SIGINT on Linux and in Docker) runs the queue-level stop, and a second one terminates immediately.
  In an interactive terminal, `Q` saves and stops the queue and `S` saves and skips the stage.
  On Linux and in Docker, `kill -USR1` skips the stage.
  See [Console trainer](../CLI/ConsoleTrainer.md#stop-and-skip).

## One writer per build

While the queue runs, it holds a per-build operating-system writer lock,
the `.training-writer.lock` file in the build directory.
As a result, the GUI, the console trainer, and scripted launches can't train into the same build at the same time.
A second launch fails with a message that the build is already in use.

The lock is released when its owning process exits, including after a crash.
An inaccessible lock is an error, never permission to launch a second trainer.
The `TrainingBuildLock.RejectsOtherWritersAndReleasesOnExit` test checks this behavior.

## What a checkpoint preserves

Checkpoint files preserve the model weights, running statistics, training counters, and any available optimizer state.
Rollout state and the live arena's RNG and contact caches aren't saved,
so resuming isn't a bit-for-bit continuation of an interrupted rollout.
