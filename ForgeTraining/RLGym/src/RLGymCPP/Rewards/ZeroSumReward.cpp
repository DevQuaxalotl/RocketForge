#include "ZeroSumReward.h"

RLGC::ZeroSumReward::ZeroSumReward(Reward* child, float teamSpirit, float opponentScale, AggregationMethod agg)
	: RewardWrapper(child), settings{ teamSpirit, opponentScale, agg } {}

RLGC::ZeroSumReward::ZeroSumReward(Reward* child, ZeroSumSettings settings)
	: RewardWrapper(child), settings(settings) {}

float RLGC::ZeroSumReward::AggregateTeam(const std::vector<float>& rewards, const GameState& state, Team team, AggregationMethod agg) {
	float aggregate = 0.f;
	int teamSize = 0;

	for (int i = 0; i < (int)state.players.size(); i++) {
		if (state.players[i].team != team)
			continue;

		if (agg == AggregationMethod::Max)
			aggregate = (teamSize == 0 || aggregate < rewards[i]) ? rewards[i] : aggregate;
		else
			aggregate += rewards[i];

		teamSize++;
	}

	return agg == AggregationMethod::Max ? aggregate : aggregate / static_cast<float>(teamSize);
}

std::vector<float> RLGC::ZeroSumReward::GetAllRewards(const GameState& state, bool final) {
	std::vector<float> rewards;
	FillRewards(state, final, rewards);

	return rewards;
}

void RLGC::ZeroSumReward::GetAllRewardsInto(const GameState& state, bool final, std::vector<float>& out) {
	if (typeid(*this) != typeid(ZeroSumReward)) {
		Reward::GetAllRewardsInto(state, final, out);
		return;
	}

	FillRewards(state, final, out);
}

void RLGC::ZeroSumReward::FillRewards(const GameState& state, bool final, std::vector<float>& rewards) {
	child->GetAllRewardsInto(state, final, rewards);
	RG_ASSERT(rewards.size() == state.players.size());
	_lastRewards = rewards;

	float teamAgg[2] = {
		AggregateTeam(rewards, state, Team::BLUE, settings.agg),
		AggregateTeam(rewards, state, Team::ORANGE, settings.agg)
	};

	for (int i = 0; i < (int)state.players.size(); i++) {
		int teamIdx = (int)state.players[i].team;
		rewards[i] =
			rewards[i] * (1 - settings.teamSpirit)
			+ teamAgg[teamIdx] * settings.teamSpirit
			- teamAgg[1 - teamIdx] * settings.opponentScale;
	}

}
