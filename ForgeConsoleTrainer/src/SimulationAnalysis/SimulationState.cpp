#include "SimulationState.h"

#include <RLGymCPP/CommonValues.h>

#include <stdexcept>

namespace RocketForge::SimulationAnalysis
{
	using namespace RocketSim;
	using namespace RLGC;

	void ResetWallProbe(Arena* arena, Car* car, int episode, uint64_t seed)
	{
		// Purpose-built reset retained for mining wall-dash action windows.
		// This is not the default trace mode because it is mechanism-specific.
		arena->ResetToRandomKickoff((int)(seed + (uint64_t)episode));

		BallState ball = {};
		ball.pos = Vec(0.f, 0.f, 1200.f);
		ball.vel = {};
		ball.angVel = {};
		arena->ball->SetState(ball);

		const float side = (episode % 2 == 0) ? 1.f : -1.f;
		const float yOffset = -2100.f + 700.f * (float)(episode % 7);
		const Vec forward(side, 0.f, 0.f);

		CarState carState = {};
		carState.pos = Vec(side * 2850.f, yOffset, RLConst::CAR_SPAWN_REST_Z);
		carState.rotMat = Angle::FromVec(forward).ToRotMat();
		carState.vel = forward * 950.f;
		carState.angVel = {};
		carState.boost = 100.f;
		carState.isOnGround = true;
		carState.obsOnGround = true;
		car->SetState(carState);
		car->controls = {};
	}

	void ResetKickoffProbe(Arena* arena, Car* car, int episode, uint64_t seed)
	{
		// Generic default reset: no stage dependency, no bot-specific state setter.
		arena->ResetToRandomKickoff((int)(seed + (uint64_t)episode));
		car->controls = {};
	}

	void ResetStageProbe(StateSetter& stateSetter, Arena* arena)
	{
		stateSetter.ResetArena(arena);
		for (Car* car : arena->_cars)
			car->controls = {};
	}

	const Player& FindPlayerByCarIdOrThrow(const GameState& state, uint32_t carId)
	{
		for (const Player& player : state.players)
			if (player.carId == carId)
				return player;

		throw std::runtime_error("traced car was missing from game state");
	}

	std::vector<Action> BuildFullStepActions(Arena* arena, Car* tracedCar, const Action& selectedAction)
	{
		std::vector<Action> result;
		result.reserve(arena->_cars.size());
		for (Car* car : arena->_cars)
			result.push_back(car == tracedCar ? selectedAction : Action{});
		return result;
	}
}