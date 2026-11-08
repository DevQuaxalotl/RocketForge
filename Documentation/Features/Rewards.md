# Rewards

A *reward* is an `RLGC::Reward` subclass that returns a value for each player on each step.
A stage pairs each reward with a weight in `GetRewardSpecs()`.
For more information, see [Training stages](TrainingStages.md).
The built-in rewards are in `RLGymCPP/Rewards/CommonRewards.h`.

Metrics, the reward debugger, and reward analysis report each registered reward under its name,
which is the class name by default.
Names also feed the segment's `setup_hash`, the critic heads, and the persisted adaptive weights,
so renaming a reward on a trained model starts all of those over.

## Write a reward

Override `GetReward(player, state, isFinal)`.
You can also override the following methods:

- `Reset` and `PreStep`, for per-episode or per-step state.
- `GetCategory()`, which returns `EVENT` for rewards that are 0 on most steps and spike on an event,
  and `CONTINUOUS` otherwise.
  The reward debugger and reward analysis handle the two categories differently.
- `GetValueRange()`, which returns the expected output range that the reward debugger uses for normalization.
- `GetDebugLines()`, which returns the 3D lines that the reward debugger draws.

Keep the raw output roughly within `[-1, 1]`, and let the weight set the scale.
That way, you can compare weights across rewards.

Rewards that emit negative values, such as `BumpedPenalty` and `DemoedPenalty`, are punishments with a positive weight.
To turn a reward that emits positive values into a punishment, give it a negative weight.

For the batch evaluation API and the buffer contract that custom rewards must follow,
see [Reward evaluation and buffer reuse](../Internals/RewardEvaluation.md).

## Zero-sum rewards

`ZeroSumReward` wraps another reward and redistributes it between teammates and opponents:

```
reward = own * (1 - teamSpirit) + agg(teammates) * teamSpirit - agg(opponents) * opponentScale
```

| Setting | Effect |
|---|---|
| `teamSpirit` | 0 is selfish; 1 shares the reward fully with teammates. Has no effect in 1v1. |
| `opponentScale` | How much an opponent earning the reward costs you. 0 turns off the zero-sum part. |
| `agg` | `Mean` or `Max` over a team's players. `Max` suits rewards that only one teammate earns at a time. |

```cpp
touchBallZeroSum.opponentScale = 0.5f;
touchBallZeroSum.teamSpirit = 0.75f;

{ new ZeroSumReward(new TouchBallReward(), touchBallZeroSum), rw_touchBall },
```

Don't wrap rewards that are already zero-sum by construction.
For example, `GoalReward` pays the conceding team `concedeScale`.

## Multi-part rewards

`RLGC::MultiPartReward` (`RLGymCPP/Rewards/MultiPartReward.h`) is one reward with several outputs that share its state.
Typically, the outputs are a continuous shaping part and an event part.

Each `Part(i)` is an ordinary `Reward*` that you register as its own spec,
so every part keeps its own per-reward settings.
`Reset` and `PreStep` run once per step regardless of how many parts are registered,
and `GetPartRewards` writes all parts for a player in one call.

```cpp
class BoostReward : public RLGC::MultiPartReward {
public:
	void GetPartRewards(const Player& player, const GameState& state, bool isFinal, std::span<float> outPerPart) override {
		outPerPart[CONT] = ...;
		outPerPart[EVENT] = ...;
	}
};

// In GetRewardSpecs(); owners must be created with make_shared
shared_ptr<BoostReward> boostReward = make_shared<BoostReward>();

{ boostReward->Part(BoostReward::CONT), rw_boost_cont },
{ boostReward->Part(BoostReward::EVENT), rw_boost_event, aw_boost_event },
```

Multi-part rewards follow these rules:

- Parts are reported as `<Owner>_<suffix>`, for example `BoostReward_Cont` and `BoostReward_Event`.
  To keep an existing reward's name when it becomes a part, set `RewardPart::nameOverride`.
- `GetParts()` defaults to a `Cont` and `Event` pair, indexed by `PartIndex { CONT, EVENT }`.
  A reward with other parts overrides `GetParts()` and declares its own `PartIndex`.
  Don't name it `enum Part`, which hides `Part()`.
- A part that isn't written on a step stays 0.
- A part with a weight of 0 is filtered out like any other reward.
  The shared work still runs for the remaining parts.

For a minimal example, see `BoostReward` in `RLGymCPP/Rewards/CommonRewards.h`.

## Reward cache

When several rewards need the same expensive per-step value, such as time-to-ball or goal predictions,
`RocketForge::RewardCacheBase` (`ForgeTraining/src/Rewards/RewardCache.h`) computes the value once per arena per tick.
Rewards fetch a typed slot by tag.
For the common case of one value per player, use `GetCachedPerPlayer`.

The cache lives on `GameState::userInfo`, and `TrainingStage` owns it.
The cache is absent during live RLBot play and offline reward analysis,
so code that uses the cache must handle a missing cache by computing only the requested player.

## Adaptive weights

To let a reward's weight tune itself during training, pass an `AdaptiveWeightConfig` as the spec's third argument.
For more information, see [Adaptive reward weights](AdaptiveRewardWeights.md).
