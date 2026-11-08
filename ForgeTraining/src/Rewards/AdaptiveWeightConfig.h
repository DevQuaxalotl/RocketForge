#pragma once

#include "PerTeamSize.h"

#include <RLGymCPP/Rewards/Reward.h>

#include <initializer_list>
#include <stdexcept>

namespace RocketForge {

	/**
	 * Either one shared AdaptiveWeightSettings applied to every team size, or three independent
	 * settings for 1v1, 2v2, and 3v3. Resolved per matchup by max team size, mirroring RewardWeight.
	 * A default-constructed config resolves to disabled settings for every size.
	 */
	struct AdaptiveWeightConfig : PerTeamSize<RLGC::AdaptiveWeightSettings> {
		AdaptiveWeightConfig() = default;

		/** Implicit: one settings object shared across all team sizes (back-compat with a bare AdaptiveWeightSettings). */
		AdaptiveWeightConfig(const RLGC::AdaptiveWeightSettings& settings) { SetShared(settings); }

		/** Three independent settings: { 1v1, 2v2, 3v3 }. */
		AdaptiveWeightConfig(std::initializer_list<RLGC::AdaptiveWeightSettings> teamSizeSettings) { SetPerTeamSize(teamSizeSettings); }

		/**
		 * Convenience for the common case where only the target differs per team size while the weight
		 * clamps (and the step/EMA defaults) are shared. perSizeTargets is { 1v1, 2v2, 3v3 }.
		 */
		static AdaptiveWeightConfig TargetReward(
			std::initializer_list<float> perSizeTargets,
			float minWeight,
			float maxWeight,
			RLGC::AdaptiveAggregation aggregation = RLGC::AdaptiveAggregation::Average) {
			return Build(perSizeTargets, minWeight, maxWeight, aggregation, /*punishment*/ false);
		}

		/** Same as TargetReward, but each resolved AdaptiveWeightSettings is built via TargetPunishment. */
		static AdaptiveWeightConfig TargetPunishment(
			std::initializer_list<float> perSizeTargets,
			float minWeight,
			float maxWeight,
			RLGC::AdaptiveAggregation aggregation = RLGC::AdaptiveAggregation::Average) {
			return Build(perSizeTargets, minWeight, maxWeight, aggregation, /*punishment*/ true);
		}

		/** Returns whether adaptive control is enabled for any team size. */
		bool AnyEnabled() const {
			return Any([](const RLGC::AdaptiveWeightSettings& s) { return s.enabled; });
		}

	private:
		static AdaptiveWeightConfig Build(
			std::initializer_list<float> perSizeTargets,
			float minWeight,
			float maxWeight,
			RLGC::AdaptiveAggregation aggregation,
			bool punishment) {
			if (perSizeTargets.size() != TEAM_SIZE_COUNT)
				throw std::runtime_error("AdaptiveWeightConfig::TargetReward/TargetPunishment requires exactly three targets: { 1v1, 2v2, 3v3 }");

			const float* targets = perSizeTargets.begin();
			auto make = [&](float target) {
				return punishment
					? RLGC::AdaptiveWeightSettings::TargetPunishment(target, minWeight, maxWeight, aggregation)
					: RLGC::AdaptiveWeightSettings::TargetReward(target, minWeight, maxWeight, aggregation);
			};

			AdaptiveWeightConfig config;
			config.SetPerTeamSize({ make(targets[0]), make(targets[1]), make(targets[2]) });
			return config;
		}
	};
}
