#include "GameState.h"
#include "../BallPrediction/PredictionContext.h"

#include "../Math.h"

#include <cmath>

using namespace RLGC;

void RLGC::GameState::BuildBoostPadIndexMap() {
	constexpr const char* ERROR_PREFIX = "BuildBoostPadIndexMap(): ";
#ifdef RG_VERBOSE
	RG_LOG("Building boost pad index map...");
#endif

	const int boostLocationAmount = GetBoostLocationAmount();

	if (lastArena->_boostPads.size() != boostLocationAmount) {
		RG_ERR_CLOSE(
			ERROR_PREFIX << "Arena boost pad count does not match GetBoostLocationAmount() " <<
			"(" << lastArena->_boostPads.size() << "/" << boostLocationAmount << ")"
		);
	}

	boostPadIndexMap.resize(boostLocationAmount);
	
	std::vector<bool> found = std::vector<bool>(boostLocationAmount);
	const Vec* boostLocations = GetBoostLocations();
	for (int i = 0; i < boostLocationAmount; i++) {
		Vec targetPos = boostLocations[i];
		for (int j = 0; j < lastArena->_boostPads.size(); j++) {
			Vec padPos = lastArena->_boostPads[j]->config.pos;

			if (padPos.DistSq2D(targetPos) < 10) {
				if (!found[i]) {
					found[i] = true;
					boostPadIndexMap[i] = j;
				} else {
					RG_ERR_CLOSE(
						ERROR_PREFIX << "Matched duplicate boost pad at " << targetPos << "=" << padPos
					);
				}
				break;
			}
		}

		if (!found[i])
			RS_ERR_CLOSE(ERROR_PREFIX << "Failed to find matching pad at " << targetPos);
	}

#ifdef RG_VERBOSE
	RG_LOG(" > Done");
#endif
	boostPadIndexMapBuilt = true;
}

void RLGC::GameState::ResetBeforeStep() {
	for (auto& player : players)
		player.ResetBeforeStep();
}

void RLGC::GameState::UpdateFromArena(Arena* arena, const std::vector<Action>& actions, GameState* prev) {
	this->prev = prev;
	if (prev)
		prev->prev = NULL;

	gameMode = arena->gameMode;

	lastArena = arena;
	int tickSkip = RS_MAX(arena->tickCount - lastTickCount, 0);
	deltaTime = tickSkip * (1 / 120.f);
	episodeTime += deltaTime;

	ball = arena->ball->GetState();

	players.resize(arena->_cars.size());

	auto carItr = arena->_cars.begin();
	for (int i = 0; i < players.size(); i++) {
		auto& player = players[i];
		player.index = i;
		player.UpdateFromCar(*carItr, arena->tickCount, tickSkip, actions[i], prev ? &prev->players[i] : NULL);
		if (player.ballTouchedStep)
			lastTouchCarID = player.carId;

		carItr++;
	}

	if (!boostPadIndexMapBuilt)
		BuildBoostPadIndexMap();

	int numBoostPads = arena->_boostPads.size();
	boostPads.resize(numBoostPads);
	boostPadsInv.resize(numBoostPads);
	boostPadTimers.resize(numBoostPads);
	boostPadTimersInv.resize(numBoostPads);

	// Per thread rather than per state, since game states are copied every step.
	thread_local std::vector<RocketSim::BoostPadTimers> padTimers;
	padTimers.resize(numBoostPads);
	arena->GetBoostPadTimers(padTimers.data());

	for (int i = 0; i < numBoostPads; i++) {
		const RocketSim::BoostPadTimers& state = padTimers[boostPadIndexMap[i]];
		const RocketSim::BoostPadTimers& stateInv = padTimers[boostPadIndexMap[GetBoostLocationAmount() - i - 1]];

		boostPads[i] = state.isActive;
		boostPadsInv[i] = stateInv.isActive;

		boostPadTimers[i] = state.cooldown;
		boostPadTimersInv[i] = stateInv.cooldown;
	}

	// Update goal scoring
	// If you don't have a GoalScoreCondition then that's not my problem lmao
	goalScored = arena->IsBallScored();

	lastTickCount = arena->tickCount;
}

BallState GameState::GetFutureBallState(const float time) const
{
	// The observing model's own predictor takes precedence over packet/match-simulator prediction.
	if (predictionContext) {
		// The arena's tracker is shared with every other caller on this arena (e.g. rewards),
		// so reusing it avoids simulating the same ball a second time.
		if (CanUseArenaBallPrediction(predictionContext->Engine()))
			return lastArena->GetFutureBallState(time);

		return predictionContext->Predict(ball, lastTickCount, time);
	}

	// If there is ball prediction data from RLBot, use that
	if (!ballPredData.empty()) {
		const int index = time / (1.f / 120.f);
		if (index < ballPredData.size())
			return ballPredData[index];
		else {
			RS_WARN("Arena::GetFutureBallState Requested time is outside of predicted range. Requested time: " << time << " Max time: " << ballPredData.size() * (1.f / 120.f));
			return BallState();
		}
	}
	// Otherwise, obtain predicted ball state from RocketSim arena
	else if (lastArena)
		return lastArena->GetFutureBallState(time);

	return BallState();
}

bool GameState::CanUseArenaBallPrediction(RocketSim::EngineId engine) const
{
	if (!lastArena || engine != lastArena->GetEngine() || lastTickCount != lastArena->tickCount)
		return false;

	// Prediction contexts step at 120 Hz; an arena at another rate predicts a different trajectory.
	if (std::abs(lastArena->GetTickRate() - 120.f) > 0.01f)
		return false;

	// A state edited away from its arena (e.g. inverted or state-set) must not read the arena's prediction.
	const BallState arenaBall = lastArena->ball->GetState();
	return arenaBall.pos == ball.pos && arenaBall.vel == ball.vel && arenaBall.angVel == ball.angVel;
}

int GameState::GetBoostLocationAmount() const {
	switch (gameMode)
	{
	case GameMode::HOOPS:
		return CommonValues::BOOST_LOCATIONS_AMOUNT_HOOPS;
	default:
	case GameMode::SOCCAR:
		return CommonValues::BOOST_LOCATIONS_AMOUNT;
		break;
	}
}

const Vec* GameState::GetBoostLocations() const {
	switch (gameMode)
	{
	case GameMode::HOOPS:
		return CommonValues::BOOST_LOCATIONS_HOOPS;
	default:
	case GameMode::SOCCAR:
		return CommonValues::BOOST_LOCATIONS;
		break;
	}
}

const float GameState::GetXWall() const {
	switch (gameMode)
	{
	case GameMode::HOOPS:
		return CommonValues::SIDE_WALL_X_HOOPS;
	default:
	case GameMode::SOCCAR:
		return CommonValues::SIDE_WALL_X;
		break;
	}
}

const float GameState::GetYWall() const {
	switch (gameMode)
	{
	case GameMode::HOOPS:
		return CommonValues::BACK_WALL_Y_HOOPS;
	default:
	case GameMode::SOCCAR:
		return CommonValues::BACK_WALL_Y;
		break;
	}
}

const float GameState::GetCeiling() const {
	switch (gameMode)
	{
	case GameMode::HOOPS:
		return CommonValues::CEILING_Z_HOOPS;
	default:
	case GameMode::SOCCAR:
		return CommonValues::CEILING_Z;
		break;
	}
}

const Vec GameState::GetGoalPos(const RocketSim::Team team) const {
	switch (gameMode)
	{
	case GameMode::HOOPS:
		return team == RocketSim::Team::BLUE ? CommonValues::BLUE_GOAL_CENTER_HOOPS : CommonValues::ORANGE_GOAL_CENTER_HOOPS;
	default:
	case GameMode::SOCCAR:
		return team == RocketSim::Team::BLUE ? CommonValues::BLUE_GOAL_CENTER : CommonValues::ORANGE_GOAL_CENTER;
		break;
	}
}