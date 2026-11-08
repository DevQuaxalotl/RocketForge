# Phasic Policy Gradient

*Phasic Policy Gradient* (PPG) is a reinforcement learning algorithm in the same family as PPO.
It keeps PPO's clipped policy update,
but it separates policy learning and value learning into different phases.
The goal is to keep PPO's stability while reducing the interference
that can happen when the policy and critic are trained too tightly together.

PPO's loop is to collect experience, estimate advantages, update the policy, update the critic, and repeat.
Many PPO implementations train the policy and critic in the same update loop,
and the two can even share early network layers.
That can work well, but the policy and critic don't always benefit from the same representation.
The policy needs features that help it choose good actions,
and the critic needs features that help it predict future return.
Those goals are related but not identical,
so shared training can pull the shared layers in two directions.

PPG splits training into a *policy phase* and an *auxiliary phase*.
The policy phase looks much like PPO.
The auxiliary phase runs less often:
it trains the policy network to contain useful value information,
while it also keeps the policy from changing too much.
In RocketForge's PPG implementation, the policy and the true critic are separate models,
and the auxiliary critic trains on the policy network's hidden features.

```text
Policy:
observation -> policy -> action logits

Auxiliary value:
observation -> policy feature layers -> aux_critic -> value

True critic:
observation -> critic -> value
```

The true critic provides value predictions and advantage estimates.
The auxiliary critic is used only during PPG's auxiliary training.
The deployed bot needs only the policy network.

## Comparison with PPO

PPO and PPG share the same foundation:

- Both are on-policy algorithms,
  which means that they train on data collected by a recent version of the current policy.
- Both use advantage estimates to decide whether sampled actions were better or worse than expected.
- Both use a clipped policy objective to keep the policy from moving too far in one update.

The main difference is how value learning affects the policy network.
PPO usually updates the policy and critic together on each rollout.
If the policy and critic share layers, then critic gradients directly update part of the policy network.
PPG doesn't use the true critic that way.
It keeps a separate critic for value prediction,
and it lets value learning shape the policy representation only during a controlled auxiliary phase.

That extra structure adds complexity and compute cost.
In exchange, it can reduce interference between the policy and the critic,
which can help when shared layers make PPO less stable or less effective.

## Purpose of the auxiliary phase

When the true critic is fully separate from the policy,
critic gradients no longer help shape the policy network's hidden features.
That avoids interference, but it can also remove a useful learning signal.
Value prediction often teaches a model about important state structure,
such as which situations are promising or dangerous.
PPG keeps that benefit without letting the critic drive the policy update directly.

The auxiliary phase relies on two ideas:

- An auxiliary critic trains on the policy network's hidden features,
  so the policy's representation learns value-relevant information.
- A *policy clone loss* keeps the current policy distribution close to the distribution from just before the auxiliary phase.
  The representation can improve, while the actual action policy is less likely to drift.

## Training process

1. **Collect experience.**

   RocketForge runs the current policy in the environment and collects a rollout of a configured number of timesteps.
   Each sample includes the observation, action mask, sampled action, old action log probability, reward,
   and terminal or truncation information.
   The rollout is fresh on-policy data from the policy that's about to be improved.

1. **Estimate advantages and target values.**

   RocketForge uses the true critic to estimate values for the visited states.
   From the rewards and values, it computes advantages and target values,
   typically through Generalized Advantage Estimation (GAE).
   The advantage tells the policy whether an action was better or worse than expected.
   The target value tells the critic what return to predict.

1. **Run the policy phase.**

   The policy phase is the PPO-like part of PPG.
   RocketForge updates the policy with PPO's clipped objective,
   which compares each sampled action's new probability with the old probability recorded during collection.
   If the new policy moves too far from the old one, clipping limits the update,
   which keeps policy learning stable and avoids destructive jumps.

   During this phase, RocketForge also trains the true critic against the target values.
   In PPG, the true critic is its own model and doesn't share a head with the policy,
   so these critic updates don't modify the policy network.

1. **Save data for the auxiliary phase.**

   PPG doesn't run the auxiliary phase after every rollout.
   Instead, RocketForge saves selected rollout data into an *auxiliary buffer* over several policy phases.
   The buffer stores observations, action masks, and target values.
   The larger dataset makes the auxiliary phase less dependent on any single rollout.

1. **Wait for the auxiliary interval.**

   After each policy phase, RocketForge increments a phase counter.
   When the counter reaches `auxPhaseInterval`, it runs the auxiliary phase.
   For example, if `auxPhaseInterval` is 32,
   then RocketForge runs 32 policy phases before each auxiliary phase.
   This is what makes PPG *phasic*:
   policy learning happens frequently,
   and representation-focused auxiliary learning happens periodically.

1. **Cache the current policy distribution.**

   Immediately before auxiliary training starts,
   RocketForge runs the current policy over the auxiliary buffer and saves its action probabilities.
   The timing matters:
   the saved probabilities must represent the policy after the recent policy updates,
   right before auxiliary training begins.
   These cached probabilities are the target of the policy clone loss.

1. **Run the auxiliary phase.**

   RocketForge optimizes three losses:

   - The *auxiliary value loss* trains `aux_critic` to predict target values from policy features,
     which updates both the auxiliary critic and the policy feature layers.
   - The *policy clone loss* compares the current policy distribution with the cached distribution,
     which reduces unwanted policy drift.
   - The *true critic loss* continues training the separate critic against the same target values.

   The goal of this step is to improve the policy network's representation
   without changing the policy's behavior too much.

1. **Clear the auxiliary buffer and repeat.**

   After the auxiliary phase, RocketForge clears the auxiliary buffer and resets the phase counter.
   Training then returns to collecting rollouts and running policy phases on fresh on-policy data.

## Hyperparameters

PPG uses the normal PPO hyperparameters plus the following PPG-specific settings:

| Setting | Effect |
|---|---|
| `policyEpochs` | PPO-style policy epochs per rollout. |
| `criticEpochs` | True critic training during the policy phase. |
| `auxPhaseInterval` | Number of policy phases before each auxiliary phase. |
| `auxEpochs` | Number of passes over the auxiliary buffer. |
| `auxBatchSize` | Auxiliary optimizer batch size. 0 inherits the PPO batch size. |
| `auxPolicyCloneLossScale` | How strongly the auxiliary phase preserves the policy distribution. |
| `auxValueLossScale` | Strength of auxiliary value learning from policy features. |
| `auxTrueCriticLossScale` | Strength of continued true critic learning during the auxiliary phase. |

The PPO `microBatchSize` setting still limits memory per microbatch for GPU training.
The auxiliary phase also has optional learning-rate overrides.

Because these settings are tightly coupled,
RocketForge selects PPO or PPG in code rather than through a GUI toggle.

## Summary

PPO is the simpler algorithm: collect data, update the policy and critic, and repeat.
PPG adds a second training rhythm:
collect data, update the policy and true critic, save auxiliary data,
and periodically run an auxiliary phase that trains value-aware policy features while preserving policy behavior.

The auxiliary phase gets some of the benefit of value-shaped policy features
without forcing the policy and true critic to share a network.
That makes PPG a good fit when PPO's shared policy and critic representation causes instability or weaker training.
