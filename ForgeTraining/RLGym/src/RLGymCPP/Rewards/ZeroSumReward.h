#pragma once
#include "RewardWrapper.h"

namespace RLGC {
	enum class AggregationMethod { Mean, Max };

	struct ZeroSumSettings {
		float teamSpirit = 0.f;
		float opponentScale = 1.f;
		AggregationMethod agg = AggregationMethod::Mean;
	};

	/**
	* Wraps a reward function to be zero-sum and team-distributed.
	* Per-player reward: ownReward*(1-teamSpirit) + agg(teamRewards)*teamSpirit - agg(oppRewards)*opponentScale
	*/
	class ZeroSumReward : public RewardWrapper {
	public:
		ZeroSumSettings settings;

		/** For reward logging */
		std::vector<float> _lastRewards = {};

		ZeroSumReward(Reward* child, float teamSpirit, float opponentScale = 1, AggregationMethod agg = AggregationMethod::Mean);

		ZeroSumReward(Reward* child, ZeroSumSettings settings);

		/** Writes wrapped values to caller-owned storage; out must not alias _lastRewards. */
		void GetAllRewardsInto(const GameState& state, bool final, std::vector<float>& out) override;

	protected:
		virtual std::vector<float> GetAllRewards(const GameState& state, bool final) override;

	private:
		void FillRewards(const GameState& state, bool final, std::vector<float>& out);

		/** Mean or max of the rewards of one team's players, taken in player order. */
		static float AggregateTeam(const std::vector<float>& rewards, const GameState& state, Team team, AggregationMethod agg);
	};
}
