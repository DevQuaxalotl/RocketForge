# Adaptive reward weights

An *adaptive weight* is a reward weight that the trainer tunes during training,
based on how often the agent actually produces the reward.
Instead of guessing a weight that's strong enough to discover a behavior but not so strong that the agent farms it,
you set a target production rate and a weight range,
and a controller moves the weight within that range.

Adaptive weights are opt-in for each reward, through the third argument of a `RewardSpec`:

```cpp
RocketForge::AdaptiveWeightConfig aw_air = {};

aw_air = AdaptiveWeightSettings::TargetReward(AdaptiveTargetContinuous(1.f, 30.f), /*minWeight*/ 0.02f, /*maxWeight*/ 0.5f);

{ new AirReward(), rw_air, aw_air },
```

A default-constructed config is disabled,
and the weight stays fixed at its configured value.

## Targets

The target is the desired *unweighted* reward magnitude per step, per player.
The following helpers convert a target from units that are easier to reason about:

| Helper | Meaning |
|---|---|
| `AdaptiveTargetContinuous(fullRewardSeconds, overSeconds)` | `fullRewardSeconds` worth of full (1.0) reward every `overSeconds`. For continuous rewards. |
| `AdaptiveTargetEvent(fires, overSeconds, tickSkip)` | `fires` full events every `overSeconds`. For event rewards; pass the stage's `agentTickSkip`. |
| `AdaptiveTargetFromWeightedImpact(weightedPerStep, weight)` | Converts a reward-analysis `meanWeightedPerStep` value into a target. |

The controller measures production separately for each arena player count,
so 1v1 and 3v3 adapt independently.
`aggregation` chooses how players are combined:

- `Average` averages over all players.
- `Max` takes the top player in each arena.
  It suits sparse, role-concentrated rewards that only one teammate is expected to earn.

## Latch mode

Latch mode is the default, because it's expected to be less confusing for the critic.
Instead of tracking the rate every iteration, the weight moves through the following phases:

| Phase | Weight | Leaves when |
|---|---|---|
| Discovering | `maxWeight` | The rate reaches the target. Moves to Settled. |
| Settled | The configured weight, such as `rw_air` | The rate holds at `masteredRatio` × target, which moves to Mastered; or the rate falls below `relearnRatio` × target, which moves to Relearning. |
| Mastered | `minWeight` | The rate falls below the target. Moves to Settled. |
| Relearning | `maxWeight` | The rate reaches the target again. Moves to Settled. |

A phase changes only after its exit condition holds for `sustainSteps` collected timesteps,
and the weight eases toward each phase's value with a half-life of `weightHalfLifeSteps`.
The measured rate is itself an exponential moving average with a half-life of `emaHalfLifeSteps`.

```cpp
aw_air.masteredRatio = 3.f;              // demand more before dropping to minWeight
aw_air.sustainSteps = 2'000'000.f;       // how long an exit condition must hold
aw_air.weightHalfLifeSteps = 250'000.f;  // how quickly the weight eases toward each phase
```

## Regulate mode

`mode = AdaptiveMode::Regulate` makes the weight a continuous function of `measured / target` instead.
The weight takes the following values:

- `maxWeight` at or below `easeStartRatio`.
- The configured weight at the target.
- `minWeight` at or above `easeEndRatio`.
- A curve shaped by `easeExponent` in between.

A ratio within `regulateDeadband` of the target counts as exactly on the target.

## Punishments

`TargetPunishment` mirrors `TargetReward`, with the target as a ceiling.
The punishment holds at `maxWeight` until the bad behavior falls to the target,
settles at the configured weight,
and fades to `minWeight` once the behavior is rare.
A `minWeight` of 0 makes the punishment scaffolding that disappears once the agent learns.

Give the weight the same sign that you'd give a fixed weight:
negative for a reward that emits positive values,
and positive for a reward that already emits negative values, such as `BumpedPenalty`.

```cpp
aw_bumped = AdaptiveWeightSettings::TargetPunishment(AdaptiveTargetEvent(1.f, 60.f, agentTickSkip), 0.f, 50.f);
```

## Per-team-size settings

A bare `AdaptiveWeightSettings` value is shared across all team sizes.
To vary the settings by team size, use `AdaptiveWeightConfig`:

```cpp
// Only the target differs; clamps shared
aw_air = AdaptiveWeightConfig::TargetReward(
	{ AdaptiveTargetContinuous(1.f, 60.f), AdaptiveTargetContinuous(1.5f, 60.f), AdaptiveTargetContinuous(2.f, 60.f) },
	/*minWeight*/ 0.02f, /*maxWeight*/ 1.f);

// Fully independent settings: { 1v1, 2v2, 3v3 }; a default AdaptiveWeightSettings{} disables that size
aw_passing = AdaptiveWeightConfig({
	AdaptiveWeightSettings{},
	AdaptiveWeightSettings::TargetReward(AdaptiveTargetContinuous(2.f, 60.f), 3.f, 30.f, AdaptiveAggregation::Max),
	AdaptiveWeightSettings::TargetReward(AdaptiveTargetContinuous(2.75f, 60.f), 3.f, 30.f, AdaptiveAggregation::Max),
});
```

`AdaptiveWeightConfig::TargetPunishment` takes the same per-size targets for punishments.

## Monitoring and persistence

For each even matchup, every controlled reward logs `Adaptive Weights/<NvN>/<reward>/Weight`
and `Adaptive Weights/<NvN>/<reward>/Ratio`, which is measured divided by target.
Uneven matchups still adapt, but they aren't logged.

Each checkpoint's `RUNNING_STATS.json` file saves the controller state, and resuming restores it.
The state is keyed by reward name, so renaming a reward restarts its controller.
