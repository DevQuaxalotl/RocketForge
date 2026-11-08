#pragma once

#include "Reward.h"

namespace RLGC {

	/**
	 * Optional speed-up base for a reward that only implements GetReward for one player at a time.
	 * Each step, plain Reward builds a new vector of every player's reward through GetAllRewards;
	 * this fills the trainer's existing buffer instead, skipping that allocation.
	 * Derive as `class MyReward : public PerPlayerReward<MyReward>`, naming your own class.
	 * A class derived from MyReward falls back to the plain path, so it can still override GetAllRewards.
	 * Deriving from Reward directly is fine when the speed doesn't matter.
	 **/
	template<typename ExactReward>
	class PerPlayerReward : public Reward {
	public:
		void GetAllRewardsInto(const GameState& state, bool isFinal, std::vector<float>& out) override {
			// NOTE: A subclass may override GetAllRewards, so bypass it only for the opted-in class.
			if (typeid(*this) != typeid(ExactReward)) {
				Reward::GetAllRewardsInto(state, isFinal, out);
				return;
			}

			out.resize(state.players.size());
			for (size_t i = 0; i < state.players.size(); i++)
				out[i] = this->GetReward(state.players[i], state, isFinal);
		}
	};
}
