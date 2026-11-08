#pragma once

#include "RewardWeight.h"
#include "AdaptiveWeightConfig.h"

#include <RLGymCPP/Rewards/Reward.h>

#include <initializer_list>
#include <memory>

namespace RocketForge
{
	/** Owns the raw reward instance until a specific matchup resolves it into a runtime WeightedReward. */
	struct RewardSpecState {
		RLGC::Reward* reward = nullptr;

		explicit RewardSpecState(RLGC::Reward* reward) : reward(reward) {}

		~RewardSpecState() {
			delete reward;
		}

		RLGC::Reward* ReleaseReward() {
			RLGC::Reward* releasedReward = reward;
			reward = nullptr;
			return releasedReward;
		}
	};

	/** Pairs a reward instance with a team-size-aware weight configuration before runtime resolution. */
	struct RewardSpec {
		std::shared_ptr<RewardSpecState> state;
		RewardWeight weight;

		/**
		 * Opt-in closed-loop adaptive weight control (see RLGC::AdaptiveWeightSettings),
		 * optionally per team size; disabled by default.
		 */
		AdaptiveWeightConfig adaptiveSettings = {};

		RewardSpec() = default;
		RewardSpec(RLGC::Reward* reward, float weight)
			: state(std::make_shared<RewardSpecState>(reward)), weight(weight) {}
		RewardSpec(RLGC::Reward* reward, int weight)
			: state(std::make_shared<RewardSpecState>(reward)), weight(weight) {}
		RewardSpec(RLGC::Reward* reward, std::initializer_list<float> teamSizeWeights)
			: state(std::make_shared<RewardSpecState>(reward)), weight(teamSizeWeights) {}
		RewardSpec(RLGC::Reward* reward, const RewardWeight& weight)
			: state(std::make_shared<RewardSpecState>(reward)), weight(weight) {}

		/** Overloads below additionally opt this reward into closed-loop adaptive weight control. */
		RewardSpec(RLGC::Reward* reward, float weight, const AdaptiveWeightConfig& adaptive)
			: state(std::make_shared<RewardSpecState>(reward)), weight(weight), adaptiveSettings(adaptive) {}
		RewardSpec(RLGC::Reward* reward, std::initializer_list<float> teamSizeWeights, const AdaptiveWeightConfig& adaptive)
			: state(std::make_shared<RewardSpecState>(reward)), weight(teamSizeWeights), adaptiveSettings(adaptive) {}
		RewardSpec(RLGC::Reward* reward, const RewardWeight& weight, const AdaptiveWeightConfig& adaptive)
			: state(std::make_shared<RewardSpecState>(reward)), weight(weight), adaptiveSettings(adaptive) {}

		/** Releases ownership of the reward so it can be used in a resolved WeightedReward. */
		RLGC::Reward* ReleaseReward() const {
			return state ? state->ReleaseReward() : nullptr;
		}

		/** Resolves this spec's weight for the provided max team size. */
		float ResolveWeight(int maxTeamSize) const {
			return weight.Resolve(maxTeamSize);
		}

		/** Returns whether this spec has any non-zero weight. */
		bool HasAnyWeight() const {
			return weight.HasAnyWeight();
		}
	};
}
