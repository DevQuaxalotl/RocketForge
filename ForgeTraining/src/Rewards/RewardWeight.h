#pragma once

#include "PerTeamSize.h"

#include <initializer_list>

namespace RocketForge {

	/** Describes either one shared reward weight or per-team-size weights for 1v1, 2v2, and 3v3. */
	struct RewardWeight : PerTeamSize<float> {
		RewardWeight() = default;
		RewardWeight(float weight);
		RewardWeight(int weight);
		RewardWeight(std::initializer_list<float> teamSizeWeights);

		RewardWeight& operator=(float weight);
		RewardWeight& operator=(int weight);
		RewardWeight& operator=(std::initializer_list<float> teamSizeWeights);

		/** Returns whether this weight contributes to any supported team size. */
		bool HasAnyWeight() const;
	};
}
