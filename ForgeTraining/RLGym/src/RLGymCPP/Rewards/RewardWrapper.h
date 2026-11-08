#pragma once
#include "Reward.h"

namespace RLGC {
	// This is a base wrapper class that wraps around an existing reward
	class RewardWrapper : public Reward {
	public:

		Reward* child;

		// Assumes ownership
		RewardWrapper(Reward* child)
			: child(child) {}

		~RewardWrapper() {
			delete child;
		}

		void GetAllRewardsInto(const GameState& state, bool isFinal, std::vector<float>& out) override {
			if (typeid(*this) != typeid(RewardWrapper)) {
				Reward::GetAllRewardsInto(state, isFinal, out);
				return;
			}

			// NOTE: Matches the base GetAllRewards: per-player calls, not the child's batch override.
			out.resize(state.players.size());
			for (size_t i = 0; i < state.players.size(); i++)
				out[i] = GetReward(state.players[i], state, isFinal);
		}

	protected:
		virtual void Reset(const GameState& initialState) override {
			child->Reset(initialState);
		}

		virtual void PreStep(const GameState& state) override {
			child->PreStep(state);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return child->GetReward(player, state, isFinal);
		}

		virtual std::string GetName() {
			return child->GetName();
		}

		RewardCategory GetCategory() const override { return child->GetCategory(); }
		RewardValueRange GetValueRange() const override { return child->GetValueRange(); }
		std::vector<DebugLine> GetDebugLines(const Player& player, const GameState& state) const override {
			return child->GetDebugLines(player, state);
		}
	};
}