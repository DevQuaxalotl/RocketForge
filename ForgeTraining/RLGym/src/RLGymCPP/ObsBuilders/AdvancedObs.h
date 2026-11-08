#pragma once

#include "ObsBuilder.h"

namespace RLGC {
	// NOTE: This not based off of AdvancedObs in Python RLGym, and is specific to GigaLearn
	class AdvancedObs : public ObsBuilder {
	public:

		constexpr static float
			POS_COEF = 1 / 5000.f,
			VEL_COEF = 1 / 2300.f,
			ANG_VEL_COEF = 1 / 3.f;

		virtual void AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball);

		virtual FList BuildObs(const Player& player, const GameState& state) override;

		/** Reuses caller storage for the stock builder; subclasses keep virtual BuildObs dispatch. */
		void BuildObsInto(const Player& player, const GameState& state, FList& out) override;

	protected:
		/** Appends the ball, the player's previous action and the boost pads: everything before the per-player blocks. */
		void AddSharedObs(FList& obs, const Player& player, const GameState& state, bool inv, const PhysState& ball);

	private:
		void FillObs(const Player& player, const GameState& state, FList& out, bool directGroups);
	};
}