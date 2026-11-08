#pragma once

#include <RLGymCPP/EnvSet/AdaptiveRewardWeightController.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <filesystem>
#include <vector>

namespace RocketForge
{
	/**
	 * Reads the adaptive controller entries a checkpoint persisted in its RUNNING_STATS.json
	 * (the "adaptive_reward_weights" section Learner::SaveStats writes).
	 * A missing file, unparsable JSON, or missing section yields an empty list.
	 */
	std::vector<RLGC::AdaptivePersistEntry> LoadPersistedAdaptiveWeights(const std::filesystem::path& checkpointFolder);

	/**
	 * Overwrites each adaptive reward's weight with the checkpoint's effective weight for an arena of
	 * (blue, orange) cars, selecting entries the way the live controller keys arenas.
	 * Static rewards, and adaptive rewards with no applicable entry, keep their current weight.
	 */
	void ApplyPersistedAdaptiveWeights(
		const std::vector<RLGC::AdaptivePersistEntry>& entries, int blue, int orange, std::vector<RLGC::WeightedReward>& rewards);
}
