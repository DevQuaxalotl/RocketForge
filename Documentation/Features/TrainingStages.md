# Training stages

A *training stage* is a `RocketForge::TrainingStage` subclass that defines what an agent trains on:
its rewards, state setters, terminal conditions, team sizes, and learner hyperparameters.
A bot's stages live in its training-plan module,
and the stage queue in the `forge_settings.json` file runs them in order.

`Bots/ExampleBot/ExampleTrainingPlan/` is the reference implementation of everything on this page.
For more information about reward wrappers and multi-part rewards, see [Rewards](Rewards.md).
For more information about weights that tune themselves during training, see [Adaptive reward weights](AdaptiveRewardWeights.md).

## Stage structure

We recommend one *parent stage* that declares everything any stage uses,
plus *child stages* that only raise the weights they train with.
The layout has the following parts:

- The parent declares every reward, every state setter, and their settings as members.
  Every weight member defaults to `0`.
- The parent's `GetRewardSpecs()` and `GetStateSetters()` methods return the full list,
  with each entry paired with its weight member.
- Each child's constructor sets the weights that it wants.
  Children don't override `GetRewardSpecs()` or `GetStateSetters()`.
- Before training, `TrainingStage` drops every reward and state setter whose weight is `0`,
  so an unused reward costs nothing and doesn't appear in metrics.

```cpp
// Parent
vector<RewardSpec> ExampleTrainingStage::GetRewardSpecs() const
{
	return {
	{ new TouchBallReward(), rw_touchBall },
	{ new GoalReward(), rw_goal },
	};
}

// Child
ExampleTrainingStage_BasicScore::ExampleTrainingStage_BasicScore(std::unique_ptr<Agent> agent) : ExampleTrainingStage(std::move(agent))
{
	stageName = NAME_BASIC_SCORE;

	rw_goal = 300.f;
	rw_touchBall = 5.f;
}
```

Because the reward list is defined once,
a reward keeps the same construction and the same name in every stage,
and you can compare the weights of a whole curriculum side by side.

Choose each child's parent by what the child inherits:

- A stage that builds on another stage parents from it, keeps its weights, and sets only what changes.
  For example, `example-advanced` parents from `example-basic-score`.
- A stage that starts clean parents from the base stage.

To drop an inherited reward or state setter, set its weight back to `0`.

The `CombineRewardSpecs()` and `CombineStateSetters()` functions append to a parent's list,
but we recommend the declare-everything-in-the-parent layout instead.

### Registration

Register each concrete stage under a kebab-case name, above its constructor:

```cpp
static constexpr string_view NAME_BASIC_SCORE = "example-basic-score";
REGISTER_STAGE(NAME_BASIC_SCORE.data(), ExampleTrainingStage_BasicScore, EXAMPLE_COMPAT_INFO);
```

`StageCompatInfo` describes which agents can use a stage.
`minAgentTeamSize` is the smallest `Agent::maxTeamSize` value that the stage supports.
For example, a stage that trains 3v3 needs an agent built for three players.

### Applying stage properties

A constructor sets stage properties, `learnerConfig` fields, and reward settings and weights in any order.
`REGISTER_STAGE` creates each stage through `TrainingStage::Create`,
which calls `UpdateConfigs()` once the most-derived constructor finishes.
`UpdateConfigs()` copies `stageName`, `runName`, `timestepsPerIteration`, and `learningRate` into `learnerConfig`,
so set those through the stage properties rather than on `learnerConfig` directly.
It also prepares `weightedTeamSizes` for team-size selection.

## Team sizes

`weightedTeamSizes` lists the matchups that a stage trains, each with a selection weight:

```cpp
// 1v1 only
weightedTeamSizes = { { 1, 1.f } };

// 1v1, 2v2 and 3v3 evenly, plus a little 1v2 and 2v3
weightedTeamSizes = { { 1, 1.f }, { 2, 1.f }, { 3, 1.f }, { 1, 2, 0.1f }, { 2, 3, 0.1f } };
```

The trainer balances the weights against the number of agents in each matchup,
so equal weights give each matchup equal training time rather than an equal arena count.
The agent's `maxTeamSize` must cover the largest team in the list.

To remove kickoff state setters from uneven matchups, where one team always reaches the ball first,
set `disableKickoffsForUnfairMatchups = true`.

### Per-team-size weights and targets

A bot that plays several modes usually needs different weights in each mode.
To set a weight per mode, declare it as `RocketForge::RewardWeight` instead of `float`,
and assign `{ 1v1, 2v2, 3v3 }`:

```cpp
RocketForge::RewardWeight rw_possession = 0.f;

rw_possession = { 5.f, 8.f, 10.f };
```

A matchup resolves the weight by its larger team, so 1v2 uses the 2v2 value.
A single value applies to every team size.

Adaptive weight settings work the same way through `AdaptiveWeightConfig`,
including per-size targets and turning off the controller for some sizes.
For more information, see [Per-team-size settings](AdaptiveRewardWeights.md#per-team-size-settings).

## State setters

`GetStateSetters()` returns `WeightedStateSetter` entries.
Each episode picks one setter at random, by weight.
The built-in setters are `KickoffState`, `FuzzedKickoffState`, and `RandomState`.
Bots can add their own `RLGC::StateSetter` subclasses.

## Terminal conditions

`GetTerminalConditions()` returns the conditions that end each episode.
The default is a 15-second `NoTouchCondition` (a truncation) and a `GoalScoreCondition` (a terminal).
Override the method to change either condition or to add bot-specific conditions.
If the method reads its timeout from a member, then child stages can change the timeout without overriding the method again.

`learnerConfig.ppo.maxEpisodeDuration`, in seconds, truncates every episode regardless of the conditions.

## Learner configuration

The following table lists the `learnerConfig` fields that stages change most often.

| Field | Purpose |
|---|---|
| `tsPerSave` | Timesteps between checkpoints. |
| `tsPerVersion` | Timesteps between skill-tracker and self-play versions. |
| `skillTracker.enabled` | Rates new versions against older ones. Usually off for the first stages. |
| `savePolicyVersions`, `trainAgainstRecentVersions`, `trainAgainstRecentPercent` | Self-play against saved older versions. |
| `trainAgainstStaticPercent` | Fraction of samples played against static opponents. |
| `ppo.entropyScale` | Exploration. Usually a constant; optionally a `ScheduledFloat`. For more information, see [Scheduled hyperparameters](#scheduled-hyperparameters). |
| `ppo.gaeGamma` | Credit horizon. 0.99 is about 100 steps; 0.995 is about 200 steps. |
| `ppo.normalizeAdvantages` | Normalizes advantages per batch. With this on, `entropyScale` and `policyGradClipNorm` need rescaling. |

### Scheduled hyperparameters

This feature is experimental.

A `GGL::ScheduledFloat` field can follow a sine wave instead of staying constant.
`ppo.entropyScale` is the only field that supports a schedule.
The value is fixed during an iteration, updates between iterations,
and is logged as `Scheduled/PPO/Entropy Scale`.

```cpp
// target 0.02, +/- 0.01, one full cycle per 10M timesteps
learnerConfig.ppo.entropyScale = GGL::ScheduledFloat(0.02f, 0.01f, 10'000'000);
```

Assigning a plain `float` makes the value constant again.

No evidence shows that a cycling entropy trains better than a constant one,
so we recommend a constant unless you're experimenting.
Before you use a schedule, consider the following behavior:

- **Each peak re-randomizes the policy.**
  Raising entropy pushes the policy back toward random actions,
  and the policy sharpens again as entropy falls.
  Play quality rises and falls with the cycle,
  which makes skill-tracker ratings noisy and leaves the stage's final checkpoint at an arbitrary point of the cycle.
- **The right values depend on advantage normalization.**
  With `ppo.normalizeAdvantages` on, an equivalent entropy scale is divided by the stage's advantage standard deviation,
  so a range that suits one setup can be far too strong or too weak in another.
- **The cycle runs on the build's total timesteps.**
  The phase is measured from `startTimestep` (default `0`) against the whole build's trained timesteps, not the stage's,
  so a stage that starts partway through a build also starts partway through the cycle.
- **The schedule only oscillates.**
  It has no one-way anneal that starts high and decays to a floor,
  so a schedule always swings back up.
