# Training dashboard

The training dashboard in ForgeGUI's **Training** tab shows the live state of the stage that's training:
where it is in the queue, how fast it runs, and how the learner is doing.
This page walks through the **Training** tab, explains each dashboard card and its fields,
and describes what to look for while you train.

## The Training tab

The left side of the tab sets up a run:

- **Agent & Build:** the agent and build to train. **New Bot**, **New Build**, and **Rename Build** live here.
- **Global Settings:** switches for the whole queue, such as **Metrics** (send metrics to Weights & Biases)
  and **Performance Cores Only**. Each one matches a `globals` field in [`forge_settings.json`](../CLI/SettingsFiles.md#globals).
- **Training Queue:** the stages to train, in order, each with its timestep target.
- **Reward Table…**, **Static Opponents…**, and **Skill References…:** a comparison of the stage's rewards and their weights,
  the frozen opponents to train against, and the models that the skill rating is measured against.
- **Start Training**, **Stop and Save**, and **Save & Skip Segment:** start the queue, save and stop it,
  or save and move on to the next stage.
  For how stopping works, see [Training sessions](TrainingSessions.md).

The right side holds the dashboard cards and **Console Output**, the trainer's log.

To keep an eye on training while you work in another window, click **Mini Monitor**.
It opens a small, always-on-top window with the main progress and timing values.
You can also set it to open automatically when you minimize ForgeGUI during training.

## How the cards update

The cards update after each training iteration: one round of collecting experience and then learning from it.

- A value that the current iteration didn't report, or that isn't a finite number, shows as N/A.
  A real zero shows as zero.
- Counts show every digit, throughput is grouped by thousands,
  very small values use scientific notation, and timings are in seconds.
- With **Metrics** on, the same values, and more, also go to Weights & Biases, where you can see their history.

## Run and progress

**Run Info** describes the model and where it runs.

| Field | Meaning |
|---|---|
| Checkpoint | The checkpoint that this session started from or last saved. Hover for the full path. |
| Device | The device that trains the model: the GPU or the CPU. |
| Run ID | The Weights & Biases run ID, when metrics are on. |
| Obs Size | The number of values in one observation: the policy network's input size. |
| Actions | The number of discrete actions the policy chooses from. |
| Model Params | The total parameter count of the networks. |
| Learning Rate | The current learning rate. |

**Training Stats** counts the whole training history: total timesteps, total iterations,
total training time, and the time of the current session.

**Progress** follows the current stage toward its target.

| Field | Meaning |
|---|---|
| Target | The total-timestep count at which this stage ends. |
| Segment Steps | The timesteps this segment has trained. |
| Remaining | The timesteps left until the target. |
| ETA | The estimated time to reach the target, at the current speed. |
| Segment Time | How long this segment has trained, across all its sessions. |

## Speed

**Throughput** shows steps per second (SPS).
*Collection* is the phase that plays games to gather experience,
and *consumption* is the phase that learns from it.

| Field | Meaning |
|---|---|
| Overall | Timesteps per second for the whole iteration. This is the number that decides how long training takes. |
| Collection | Timesteps per second while collecting. |
| Consumption | Timesteps per second while learning. |

**Timing (s)** splits one iteration into its parts:
**Collection**, **Consumption**, **Inference** (running the policy during collection),
**Env Step** (stepping the simulator), **Reward** (computing rewards), and **Iteration** (the total).

If training is slower than you expect, compare these numbers.
Collection-bound runs depend mostly on CPU speed and the number of arenas;
consumption-bound runs depend mostly on the GPU and the batch size.
For how to measure speed reliably, see [Performance testing](../PerformanceTesting.md).

## Learning

These cards show how the learner is doing.
No single value says that training is going well; watch how the values trend over many iterations,
and compare them with an earlier run of the same stage.

### Collect

| Field | Meaning | What to look for |
|---|---|---|
| Timesteps | The timesteps collected in this iteration. | Stays steady; it's set by the stage's batch size. |
| Avg Reward | The average weighted reward per step. | Rises as the agent learns what the stage rewards. A sudden drop after a change usually means the change hurt. |
| Episode Len | The average episode length, in steps. | Depends on the stage: a stage that ends episodes on a goal or a touch gets shorter episodes as the agent improves. |

### GAE

GAE (generalized advantage estimation) turns rewards into the training targets for the policy and the critic.

| Field | Meaning | What to look for |
|---|---|---|
| Avg Adv | The average size of the advantages: how much better or worse actions turned out than the critic expected. | Shrinks as the critic gets better at predicting returns. |
| Avg Return | The average discounted return. | Follows the reward trend. |
| Val Target | The average size of the critic's training targets. | Moves with Avg Return. |
| Returns STD | The standard deviation of returns, used to normalize rewards. | Large swings mean the reward scale is changing. |
| Reward Clip | The fraction of normalized rewards that hit the reward clip range. | Stays near zero. A high value means rewards are often clipped, so the reward scale is too large for the clip range. |

### Loss

| Field | Meaning | What to look for |
|---|---|---|
| Policy | The policy loss. | Noisy by nature; watch the other fields instead. |
| Critic | The critic's prediction error. | Falls as the critic learns, and rises again when the rewards or the stage change. |
| Entropy | How spread out the policy's action choices are. | Falls slowly as the policy grows confident. A fast fall to near zero means the policy has stopped exploring. |
| KL Div | How far each update moved the policy. | Stays small and fairly steady. Spikes mean updates are too large. |
| Clip Frac | The fraction of samples where the PPO clip limited the update. | Stays fairly steady. A rising value, together with a rising KL Div, means updates are too aggressive. |

### Update

**Update** shows the size of each update to the networks' parameters:
**Policy Mag**, **Critic Mag**, and, when the networks share layers, **Shared Head**.
These magnitudes measure how much the parameters changed; they aren't a learning rate or a gradient norm.
Sudden spikes often appear together with KL Div spikes.

## Skill

The **Skill (segment delta)** card shows how strong the model is, measured in game.
While a segment trains, the skill tracker periodically plays the model against the build's skill references,
and rates it in each game mode.
To choose the references, click **Skill References…**.
To turn skill tracking off, clear `skillTestingEnabled` in [`forge_settings.json`](../CLI/SettingsFiles.md#globals).

For each mode, the card shows the latest saved or completed rating, and **Overall** is the mean of the mode ratings.
The value in parentheses is the change since the current segment began, including the segment's earlier sessions,
so it answers "has this segment made the model better?".
The card keeps its values between skill evaluations,
and it resets to the new segment's starting rating when the queue moves to the next stage.

Each segment records its starting rating when it first trains:
the configured initial rating for a segment trained from scratch,
or the source checkpoint's rating for a segment started from a checkpoint.
The ratings are stored as `skill_start_ratings` in the segment's `segment.json` file.

A dash in place of the change means the starting rating is unknown,
which happens for segments created before RocketForge recorded starting ratings.
The dashboard then looks for the rating in the segment's source checkpoint,
or in a checkpoint at the segment's exact starting timestep, and never substitutes a later one.
**Overall** shows a change only when every mode has a known starting rating.

## Related pages

- [Training sessions](TrainingSessions.md): stopping, skipping, and what a checkpoint keeps.
- [Training stages](TrainingStages.md): the stage settings behind batch size, rewards, and learner hyperparameters.
- [Phasic Policy Gradient](../Concepts/PPGAlgorithm.md): how RocketForge's learner trains the policy and critic.
- [Model benchmarking](ModelBenchmarking.md): rate models against each other on demand.
