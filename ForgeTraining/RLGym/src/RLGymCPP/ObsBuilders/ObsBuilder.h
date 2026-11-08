#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"
#include "../BasicTypes/Lists.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/obs_builder.py
namespace RLGC {
	class ObsBuilder {
	public:
		virtual ~ObsBuilder() = default;
		virtual void Reset(const GameState& initialState) {}

		// NOTE: May be called once during environment initialization to determine policy neuron size
		virtual FList BuildObs(const Player& player, const GameState& state) = 0;

		virtual void BuildObsInto(const Player& player, const GameState& state, FList& out) {
			out = BuildObs(player, state);
		}
	};
}