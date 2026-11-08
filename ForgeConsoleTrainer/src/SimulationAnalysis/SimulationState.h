#pragma once

#include <RLGymCPP/BasicTypes/Action.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/StateSetters/StateSetter.h>

#include <cstdint>
#include <vector>

namespace RocketForge::SimulationAnalysis
{
	/** Resets the arena to the original side-wall probe used for wall-dash mining. **/
	void ResetWallProbe(RocketSim::Arena* arena, RocketSim::Car* car, int episode, uint64_t seed);
	/** Resets the arena to a deterministic kickoff probe. **/
	void ResetKickoffProbe(RocketSim::Arena* arena, RocketSim::Car* car, int episode, uint64_t seed);
	/** Resets the arena through an arbitrary training stage state setter. **/
	void ResetStageProbe(RLGC::StateSetter& stateSetter, RocketSim::Arena* arena);
	/** Finds the traced player in the observation state or throws an action-trace error. **/
	const RLGC::Player& FindPlayerByCarIdOrThrow(const RLGC::GameState& state, uint32_t carId);
	/** Builds a full multi-car action list for GameState::UpdateFromArena. **/
	std::vector<RLGC::Action> BuildFullStepActions(
		RocketSim::Arena* arena,
		RocketSim::Car* tracedCar,
		const RLGC::Action& selectedAction);
}