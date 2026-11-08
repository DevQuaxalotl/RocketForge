#pragma once

#include "AdvancedObs.h"

namespace RLGC {
	/**
	 * AdvancedObs with zero-filled slots for missing teammates and opponents,
	 * so the observation size depends only on maxPlayers and one network can play every team size up to it.
	 * Real players fill slots in state.players order without shuffling, so at maxPlayers = 1 the output equals AdvancedObs's.
	 */
	class AdvancedObsPadded : public AdvancedObs {
	public:
		/** Largest team size this builder accepts; usually the Agent's maxTeamSize. */
		int maxPlayers;

		explicit AdvancedObsPadded(int maxPlayers);

		FList BuildObs(const Player& player, const GameState& state) override;

		/** Reuses caller storage for the stock builder; subclasses keep virtual BuildObs dispatch. */
		void BuildObsInto(const Player& player, const GameState& state, FList& out) override;

	private:
		void FillPaddedObs(const Player& player, const GameState& state, FList& obs);
	};
}
