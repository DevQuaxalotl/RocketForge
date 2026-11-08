#include "RewardWeight.h"

namespace RocketForge {

	RewardWeight::RewardWeight(float weight) { SetShared(weight); }
	RewardWeight::RewardWeight(int weight) { SetShared((float)weight); }
	RewardWeight::RewardWeight(std::initializer_list<float> teamSizeWeights) { SetPerTeamSize(teamSizeWeights); }

	RewardWeight& RewardWeight::operator=(float weight) { SetShared(weight); return *this; }
	RewardWeight& RewardWeight::operator=(int weight) { SetShared((float)weight); return *this; }
	RewardWeight& RewardWeight::operator=(std::initializer_list<float> teamSizeWeights) { SetPerTeamSize(teamSizeWeights); return *this; }

	bool RewardWeight::HasAnyWeight() const {
		return Any([](float weight) { return weight != 0.f; });
	}
}
