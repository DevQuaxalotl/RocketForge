# Transfer learning

Transfer learning trains a new *student* policy to imitate a trained, frozen *teacher* policy.
Use it when you change something that makes a trained model unusable as a starting point,
such as the observation builder, the action parser, or the network size,
and you want the new model to start from the old model's behavior instead of from scratch.

The teacher and the student can have different network architectures, observation widths, and action spaces.

This page shows you how to set up a transfer run, choose its loss, map different action spaces,
and read its metrics.

## How it works

1. The student plays games and collects game states.
1. For the same states, the learner builds the teacher's observations and action masks,
   with the teacher's own observation builder and action parser,
   and records the teacher's action probabilities.
1. The student trains to match those probabilities for several epochs.
1. The cycle repeats with a fresh batch.

Only the policy learns.
The teacher stays frozen, and the critic doesn't train;
after transfer learning, continue with normal training so that the critic catches up.

## Set up a transfer run

Transfer learning is configured in C++, on a GigaLearn `Learner`.
It has no ForgeGUI or console mode.

1. Describe the teacher in a `GGL::TransferLearnConfig`:

   | Field | Meaning |
   |---|---|
   | `makeOldObsFn` | Returns a new instance of the teacher's observation builder. It can be the same as the student's. |
   | `makeOldActFn` | Returns a new instance of the teacher's action parser. It can be the same as the student's. |
   | `oldPolicyConfig` | The teacher policy's network configuration. |
   | `oldSharedHeadConfig` | The teacher's shared-head configuration. Leave it unset if the teacher has no shared layers. |
   | `oldModelsPath` | The checkpoint folder that holds the teacher's model files. |
   | `mapActsFn` | Maps student actions to teacher actions. Required unless the action spaces are identical; see [Map different action spaces](#map-different-action-spaces). |

1. Set the training settings:

   | Field | Default | Meaning |
   |---|---|---|
   | `batchSize` | `50'000` | Rows collected per update. Collection runs whole environment steps, so it can slightly exceed this value. |
   | `microBatchSize` | `0` | Maximum rows evaluated at once, to limit device memory. `0` evaluates the whole batch at once. |
   | `epochs` | `5` | Passes over each batch. Each epoch is one optimizer step. |
   | `lr` | `3e-4` | Learning rate. |
   | `useKLDiv`, `useHardActionLabels` | `false` | The loss. See [Choose a loss](#choose-a-loss). |
   | `lossScale` | `500` | Multiplies the loss. The default suits the absolute-error loss. |
   | `lossExponent` | `1` | Positive exponent applied before averaging and before `lossScale`. |

   The defaults suit a basic ball-chasing teacher.
   For a skilled teacher, use a larger batch and a much smaller learning rate.
1. Start the run:

   ```cpp
   GGL::TransferLearnConfig transfer;
   // Set the teacher factories, architecture, and checkpoint folder.
   transfer.batchSize = 50'000;
   transfer.microBatchSize = 2'048;
   transfer.epochs = 5;
   transfer.useKLDiv = true;
   transfer.lossScale = 1.f;
   transfer.lr = 3e-4f;
   learner.StartTransferLearn(transfer);
   ```

The learner validates the configuration, shapes, masks, and action maps before the first epoch changes the student.

Training saves and stops at the learner's iteration boundary, like normal training.
For unattended runs, use the training host's graceful stop.
Collected rows and cached teacher targets aren't checkpointed, so a resumed run collects a fresh batch.

RocketForge doesn't ship any teacher models or training data.

## Choose a loss

| Setting | Loss | Use it when |
|---|---|---|
| `useHardActionLabels = true` | Cross-entropy against the teacher's most probable action among the actions that both parsers allow. Takes priority over `useKLDiv`. | You want the student to copy the teacher's top choice. |
| `useKLDiv = true` | Categorical KL divergence, KL(teacher ‖ student), summed over actions and averaged over rows. | You want the student to match the teacher's whole distribution. Start with `lossScale = 1`, and tune the scale and the learning rate. |
| Both `false` | Mean absolute probability error, averaged over every student action. | The default. |

`lossExponent` applies per row to KL divergence and cross-entropy, and per action to absolute error.
Each loss has a different natural scale, so retune `lossScale` and `lr` when you switch losses.

## Map different action spaces

Omit `mapActsFn` only when both parsers have the same number of actions,
**with the same meanings in exactly the same order**.

Otherwise, `mapActsFn` returns one teacher action index for each student action,
or `-1` for a student action that the teacher doesn't have.
For example, if the student's first three actions are the teacher's actions 2, 0, and 1,
and its fourth action is new, return `{2, 0, 1, -1}`.
The map can change from state to state.
Teacher indices must be in range, and unique within the row.

The teacher's probabilities are renormalized over the mapped actions that are legal for **both** parsers,
and the student's loss uses the same set of actions.
As a result:

- The student learns the teacher's relative preferences among shared actions,
  not how likely a shared action is compared with a new one.
- New student actions get no direct gradient from the transfer loss.
  They can still change indirectly, through shared layers, optimizer momentum, or weight decay.
- Teacher actions outside the map can't be transferred.
- A row with no shared legal action contributes no loss, and counts as skipped.
  If a whole batch has no labeled rows, no optimizer step runs.

## Memory use

The full collection and the cached teacher targets stay in CPU memory, so CPU memory grows with `batchSize`.
Device memory for inputs, outputs, and activations grows with `microBatchSize`,
but the model weights and optimizer state must still fit on the device.

Gradients accumulate across microbatches into one optimizer step per epoch, weighted by row count,
so changing the microbatch size doesn't change the objective, apart from floating-point differences.
The batch size doesn't need to be a multiple of the microbatch size.
Sampled actions during collection can still differ when inference is split into chunks.

If you call `PPOLearner::TransferLearn` directly, pass CPU tensors,
so that your own code doesn't keep the full batch on the device.

## Read the metrics

| Metric | Meaning |
|---|---|
| `Transfer Learn Loss` | The loss on the batch, measured before the first epoch's update. |
| `Transfer Learn Accuracy` | How often the student's top action matches the teacher's, on labeled rows, before the first epoch's update. |
| `Transfer/Supervised Rows`, `Transfer/Skipped Rows` | Rows with and without shared teacher probability. |
| `Transfer/Microbatch Size`, `Transfer/Optimizer Steps` | The effective microbatch size and the number of optimizer steps. |

The report also includes the policy entropy and the policy update magnitude.

## Match the teacher's original conditions

The student only learns what the teacher's probabilities look like in this setup, so make that setup match how the teacher played:

- The teacher runs with the learner's policy temperature, exploration floor, and inference precision.
  Set these to the values the teacher was trained with.
- The teacher's observation builder must apply any preprocessing the teacher needs,
  because transfer learning doesn't load a separate observation-normalization state for the teacher.
- If the critic shares layers with the policy, its predictions can still change.

The PPO guiding-policy option is a different feature, and doesn't use this path.

## Tests

The `TransferLearning.*` and `LossModes/TransferBatching.*` tests check the following:

- Analytical losses and gradients.
- Empty action intersections, extra student actions, and invalid maps.
- That the teacher and critic don't change.
- That full-batch and uneven-microbatch updates match.

The batching tests check the row limit of both policy passes, with different teacher and student observation widths.
To run them on CUDA as well as the CPU, set `ROCKETFORGE_TEST_GPU=1`.
