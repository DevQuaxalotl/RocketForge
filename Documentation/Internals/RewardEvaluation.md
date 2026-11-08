# Reward evaluation and buffer reuse

Training and reward analysis keep a separate set of reward buffers for each arena.
The stock rewards, the plain reward wrapper, the zero-sum wrapper, and multi-part reward views write into those buffers.
Once the buffers reach the arena's player count,
these paths don't allocate an output vector for each reward on each step.

Buffer reuse doesn't change the meaning of weighted totals, evaluation order, final-step flags,
raw earned and debug values, or adaptive-weight samples.

For how to write and register rewards, see [Rewards](../Features/Rewards.md).

## Custom rewards

`Reward::GetAllRewardsInto(state, isFinal, out)` defaults to calling the virtual `GetAllRewards(state, isFinal)` method.
Rewards that override batch evaluation work without source changes,
including subclasses of stock rewards and wrappers.
This compatibility path can still allocate.
After you change the reward interface,
rebuild applications and custom native libraries together.

A reward that's implemented entirely through `GetReward` can opt into buffer reuse:

```cpp
#include <RLGymCPP/Rewards/PerPlayerReward.h>

class ExampleReward : public RLGC::PerPlayerReward<ExampleReward> {
public:
    float GetReward(const RLGC::Player& player, const RLGC::GameState&, bool) override {
        return player.isOnGround ? 0.f : 1.f;
    }
};
```

Pass the exact class as the template argument.
The adapter is for per-player implementations only;
a batch reward implements `GetAllRewardsInto` itself.
The exact-type check preserves batch overrides in further subclasses,
which keep the compatibility path until they explicitly opt in.

An in-place batch implementation must follow these rules:

- Resize `out` to the player count,
  and write every value in state-player order, including zeros.
- Produce the same values and state updates as `GetAllRewards`.
- Don't keep pointers or references to caller-owned storage.
  The caller reuses that storage for other rewards and later steps.
- Don't pass a zero-sum wrapper an output that aliases its `_lastRewards` vector.

## Share calculations between reward parts

`MultiPartReward` shares calculations explicitly:

1. Inside each arena's environment factory, create one owner with `std::make_shared`.
1. Register the owner's `Part(...)` views as independently weighted rewards.
1. In `GetPartRewards`, calculate the common quantities once and fill the related outputs together.

Views can have different weights and zero-sum settings while reusing the owner's computed values,
and each view writes its part into the caller's reusable output buffer.

Keep the owner local to one arena, because arenas evaluate rewards in parallel.
Call `Reset` and `PreStep` on each registered view as usual;
the owner handles one hook per round.
The owner invalidates its values on a new step, a reset, a state or tick change,
a player-count change, or a final-step flag change.
Reusing the same tick number still needs `PreStep` for a new step.
Separate instances are never merged implicitly,
so reward configuration and state stay independent.

## Tests

The `RewardEvaluation` tests cover the following behavior:

- Numerical equivalence of the stock rewards.
- Retained buffer storage.
- Batch-only custom rewards and further subclasses.
- Plain and nested wrappers.
- Raw debug values.
- Shared multi-part evaluation.

The arena integration test checks separate buffers for different arena sizes,
repeated steps and resets, weighted totals, reward heads, and raw versus wrapped debug values.
The multi-part, reward-analysis, and adaptive-weight tests cover those features' own behavior.

## Measured reward-loop performance

These measurements were taken on October 17, 2026, against public `main` commit `558c94a`,
with MSVC 19.44 Release builds.
Both the v2 and v3 applications were rebuilt,
and all 117 focused regression tests passed on each simulator with no skips.

The isolated v2 benchmark evaluates eight public reward components,
including three zero-sum wrappers and two shared multi-part outputs,
and accumulates their weighted totals.
The following table shows median milliseconds for 200,000 steps, over seven samples after warm-up.
The versions before and after buffer reuse were compiled separately,
with the same harness and simulator math library.

| Players | Before | With buffer reuse | Reduction in evaluation time |
| --- | ---: | ---: | ---: |
| 2 | 83.13 ms | 36.67 ms | 55.9% |
| 4 | 90.58 ms | 50.77 ms | 44.0% |
| 6 | 105.87 ms | 66.31 ms | 37.4% |

The weighted-output checksums matched exactly in all three cases.
A separate allocation-counting pass over 1,000 warm steps recorded nine allocations per step before and zero after.
The multi-part calculations were already shared before this change,
so these numbers isolate the benefit of buffer reuse.

These are reward-loop measurements, not end-to-end training throughput.
Simulation, observations, policy inference, optimization, and logging are outside the timed loop.
Custom batch rewards that use the compatibility path can still allocate,
and the gain varies with reward cost and player count.
