#include <RocketSim/Arena.h>
#include <RocketSim/BallPredTracker.h>

#include "Backend/ArenaBackend.h"
#include "Backend/V2Backend.h"

#include <EngineConfig.h>

#if RF_ENGINE_V3_AVAILABLE
#include "Backend/V3Backend.h"
#endif

#include <cfloat>
#include <stdexcept>

namespace RocketSim {

using EngineBackend::ArenaBackend;

void Init(std::filesystem::path collisionMeshesFolder, bool silent) {
	EngineBackend::InitV2(collisionMeshesFolder, silent);
#if RF_ENGINE_V3_AVAILABLE
	EngineBackend::InitV3(collisionMeshesFolder, silent);
#endif
}

std::filesystem::path CollisionMeshesFolder() {
	return std::filesystem::path(RF_COLLISION_MESHES_DIR);
}

std::filesystem::path CollisionMeshesRelativePath() {
	return std::filesystem::path(RF_COLLISION_MESHES_RELATIVE_PATH);
}

RocketSimStage GetStage() {
	const RocketSimStage v2Stage = EngineBackend::GetV2Stage();
#if RF_ENGINE_V3_AVAILABLE
	if (v2Stage == RocketSimStage::INITIALIZED && !EngineBackend::V3IsInitialized())
		return RocketSimStage::UNINITIALIZED;
#endif
	return v2Stage;
}

bool HasCollisionMeshes(EngineId engine, GameMode gameMode) {
	if (engine == EngineId::V2)
		return EngineBackend::V2HasCollisionMeshes(gameMode);

#if RF_ENGINE_V3_AVAILABLE
	if (engine == EngineId::V3)
		return EngineBackend::V3HasCollisionMeshes(gameMode);
#endif

	return false;
}

/////////////// Car ///////////////

CarState Car::GetState() {
	CarState state = _arena->_backend->GetCarState(_idx);

	// Backends refer to other cars by index; ids belong to the arena and callers may reassign them.
	uint32_t& otherCarID = state.carContact.otherCarID;
	if (otherCarID != 0)
		otherCarID = otherCarID <= _arena->_cars.size() ? _arena->_cars[otherCarID - 1]->id : 0;

	return state;
}

void Car::SetState(const CarState& state) {
	CarState backendState = state;

	uint32_t& otherCarID = backendState.carContact.otherCarID;
	if (otherCarID != 0) {
		Car* otherCar = _arena->GetCar(otherCarID);
		otherCarID = otherCar ? otherCar->_idx + 1 : 0;
	}

	_arena->_backend->SetCarState(_idx, backendState);
}

void Car::ResetStepContactState() {
	_arena->_backend->ResetCarStepContact(_idx);
}

Vec Car::GetForwardDir() const {
	return _arena->_backend->GetCarState(_idx).rotMat.forward;
}

Vec Car::GetRightDir() const {
	return _arena->_backend->GetCarState(_idx).rotMat.right;
}

Vec Car::GetUpDir() const {
	return _arena->_backend->GetCarState(_idx).rotMat.up;
}

/////////////// Ball ///////////////

BallState Ball::GetState() {
	return _arena->_backend->GetBallState();
}

void Ball::SetState(const BallState& state) {
	_arena->_backend->SetBallState(state);
}

/////////////// BoostPad ///////////////

BoostPadState BoostPad::GetState() const {
	const ArenaBackend::BoostPadStateRaw raw = _arena->_backend->GetPadState(_idx);

	BoostPadState state;
	state.isActive = raw.isActive;
	state.cooldown = raw.cooldown;
	state.curLockedCar = raw.lockedCarIdx == UINT32_MAX ? nullptr : _arena->_cars[raw.lockedCarIdx];
	state.prevLockedCarID = raw.prevLockedCarIdx == UINT32_MAX ? 0 : _arena->_cars[raw.prevLockedCarIdx]->id;
	return state;
}

void BoostPad::SetState(const BoostPadState& state) {
	const Car* prevLockedCar = state.prevLockedCarID ? _arena->GetCar(state.prevLockedCarID) : nullptr;

	ArenaBackend::BoostPadStateRaw raw;
	raw.isActive = state.isActive;
	raw.cooldown = state.cooldown;
	raw.lockedCarIdx = state.curLockedCar ? state.curLockedCar->_idx : UINT32_MAX;
	raw.prevLockedCarIdx = prevLockedCar ? prevLockedCar->_idx : UINT32_MAX;
	_arena->_backend->SetPadState(_idx, raw);
}

/////////////// Arena ///////////////

Arena::Arena(GameMode gameMode, const ArenaConfig& config, float tickRate)
	: gameMode(gameMode), tickTime(1 / tickRate), _config(config), _mutatorConfig(gameMode),
	  _stepResult(std::make_unique<EngineBackend::StepResult>()) {}

Arena* Arena::Create(GameMode gameMode, const ArenaConfig& config, float tickRate) {
	std::unique_ptr<Arena> arena(new Arena(gameMode, config, tickRate));
	arena->_backend = EngineBackend::CreateBackend(config.engine, gameMode, config, tickRate, arena->tickCount);

	arena->ball = new Ball();
	arena->ball->_arena = arena.get();

	const uint32_t numPads = arena->_backend->NumPads();
	arena->_boostPads.reserve(numPads);
	for (uint32_t i = 0; i < numPads; i++) {
		BoostPad* pad = new BoostPad();
		pad->_arena = arena.get();
		pad->_idx = i;
		pad->config = arena->_backend->GetPadConfig(i);
		arena->_boostPads.push_back(pad);
	}

	return arena.release();
}

Arena::~Arena() {
	delete _ballPredTracker;

	for (Car* car : _cars)
		delete car;
	for (BoostPad* pad : _boostPads)
		delete pad;
	delete ball;
}

Car* Arena::AddCar(Team team, const CarConfig& config) {
	const uint32_t idx = _backend->AddCar(team, config);
	if (idx != _cars.size())
		RS_ERR_CLOSE("Arena::AddCar(): engine car index " << idx << " doesn't match car count " << _cars.size());

	Car* car = new Car();
	car->config = config;
	car->team = team;
	car->id = ++_lastCarID;
	car->_arena = this;
	car->_idx = idx;

	_cars.push_back(car);
	_carIDMap[car->id] = car;
	return car;
}

Car* Arena::GetCar(uint32_t id) {
	auto itr = _carIDMap.find(id);
	return itr == _carIDMap.end() ? nullptr : itr->second;
}

void Arena::GetBoostPadTimers(BoostPadTimers* out) const {
	_backend->GetPadTimers(out);
}

void Arena::Step(int ticksToSimulate) {
	if (ticksToSimulate <= 0)
		return;

	_stepControls.resize(_cars.size());
	for (size_t i = 0; i < _cars.size(); i++) {
		// Matches the clamp engines apply internally, so callers that read controls back see what was simulated.
		_cars[i]->controls.ClampFix();
		_stepControls[i] = _cars[i]->controls;
	}

	EngineBackend::StepResult& result = *_stepResult;
	_backend->Step(ticksToSimulate, tickCount, _stepControls.data(), result);
	tickCount += ticksToSimulate;

	if (_bumpCallback.func)
		for (const EngineBackend::StepResult::Bump& bump : result.bumps)
			_bumpCallback.func(this, _cars[bump.bumperIdx], _cars[bump.victimIdx], bump.isDemo, _bumpCallback.userInfo);
}

void Arena::ResetToRandomKickoff(int seed) {
	_backend->ResetToRandomKickoff(seed);
}

bool Arena::IsBallScored() const {
	return _backend->IsBallScored();
}

// Returns negative inside the hoop; the margin is squared.
static float BallWithinHoopsGoalXYMarginSq(float x, float y) {
	constexpr float
		SCALE_Y = 0.9f,
		OFFSET_Y = 2770.f,
		RADIUS_SQ = 716 * 716;

	float dy = abs(y) * SCALE_Y - OFFSET_Y;
	float distSq = x * x + dy * dy;
	return distSq - RADIUS_SQ;
}

bool Arena::IsBallProbablyGoingIn(float maxTime, float extraMargin, Team* goalTeamOut) const {
	const BallState ballState = ball->GetState();
	const Vec ballPos = ballState.pos;
	const Vec ballVel = ballState.vel;

	if (gameMode == GameMode::SOCCAR || gameMode == GameMode::SNOWDAY) {
		if (abs(ballVel.y) < FLT_EPSILON)
			return false;

		float scoreDirSgn = RS_SGN(ballVel.y);
		float goalY = _mutatorConfig.goalBaseThresholdY * scoreDirSgn;
		float distToGoal = abs(ballPos.y - goalY);

		float timeToGoal = distToGoal / abs(ballVel.y);

		if (timeToGoal > maxTime)
			return false;

		Vec extrapPosWhenScore = ballPos + (ballVel * timeToGoal) + (_mutatorConfig.gravity * timeToGoal * timeToGoal) / 2;

		// From: https://github.com/RLBot/RLBot/wiki/Useful-Game-Values
		constexpr float
			APPROX_GOAL_HALF_WIDTH = 892.755f,
			APPROX_GOAL_HEIGHT = 642.775;

		float scoreMargin = _mutatorConfig.ballRadius * 0.1f + extraMargin;

		if (extrapPosWhenScore.z > APPROX_GOAL_HEIGHT + scoreMargin)
			return false;

		if (abs(extrapPosWhenScore.x) > APPROX_GOAL_HALF_WIDTH + scoreMargin)
			return false;

		if (goalTeamOut)
			*goalTeamOut = RS_TEAM_FROM_Y(scoreDirSgn);

		return true;
	} else if (gameMode == GameMode::HOOPS) {

		constexpr float
			APPROX_RIM_HEIGHT = 365;

		float minHeight = APPROX_RIM_HEIGHT + _mutatorConfig.ballRadius * 1.2f;

		if (ballVel.z < -FLT_EPSILON && ballPos.z < minHeight) {
			if (BallWithinHoopsGoalXYMarginSq(ballPos.x, ballPos.y) < 0) {
				if (goalTeamOut)
					*goalTeamOut = RS_TEAM_FROM_Y(ballPos.y);
				return true;
			}
		}

		float margin = _mutatorConfig.ballRadius * 1.0f;
		float marginSq = margin * margin;

		float upQuadIntercept;
		float downQuadIntercept;

		// Time to reach the rim height, from the quadratic intercept
		{
			float g = _mutatorConfig.gravity.z;
			if (g > -FLT_EPSILON)
				return false;

			float v = ballVel.z;
			float h = ballPos.z - minHeight;

			float sqrtInput = v * v - 2 * g * h;
			if (sqrtInput > 0) {
				float sqrtOutput = sqrtf(sqrtInput);
				upQuadIntercept = (-v + sqrtOutput) / g;
				downQuadIntercept = (-v - sqrtOutput) / g;
			} else {
				// Never reaches the rim height, so it scores only if it is already within the hoop
				return BallWithinHoopsGoalXYMarginSq(ballPos.x, ballPos.y) < -marginSq;
			}
		}

		if (upQuadIntercept >= 0) {
			// The ball rises through the rim height first; it must not clip the rim on the way up
			Vec extrapPosUp = ballPos + (ballVel * upQuadIntercept);
			float upMarginSq = BallWithinHoopsGoalXYMarginSq(extrapPosUp.x, extrapPosUp.y);

			float minClearanceMargin = 60 + _mutatorConfig.ballRadius;

			if (upMarginSq > -marginSq && upMarginSq < (minClearanceMargin * minClearanceMargin))
				return false;
		}

		Vec extrapPosDown = ballPos + (ballVel * downQuadIntercept);
		extrapPosDown.y = abs(extrapPosDown.y);

		{ // Very approximate prediction of a backboard bounce
			float wallBounceY = RLConst::ARENA_EXTENT_Y_HOOPS - _mutatorConfig.ballRadius;
			if (extrapPosDown.y > wallBounceY) {
				float bounceMargin = extrapPosDown.y - wallBounceY;
				extrapPosDown.y -= bounceMargin * (1 + _mutatorConfig.ballWorldRestitution);
			}
		}

		if (BallWithinHoopsGoalXYMarginSq(extrapPosDown.x, extrapPosDown.y) < -marginSq) {
			if (goalTeamOut)
				*goalTeamOut = RS_TEAM_FROM_Y(extrapPosDown.y);
			return true;
		} else {
			return false;
		}

	} else {
		RS_ERR_CLOSE("Arena::IsBallProbablyGoingIn() is not supported for gamemode " << GAMEMODE_STRS[(int)gameMode]);
		return false;
	}
}

BallState Arena::GetFutureBallState(float time) {
	if (time <= 0.f)
		return ball->GetState();

	// 1 / GetTickRate() rather than tickTime: the two can differ in the last bit, which can shift the horizon by a tick.
	const float predTickTime = 1.f / GetTickRate();
	const int predTicks = static_cast<int>(time / predTickTime);

	if (!_ballPredTracker)
		_ballPredTracker = new BallPredTracker(this, predTicks);

	if (predTicks > static_cast<int>(_ballPredTracker->numPredTicks))
		_ballPredTracker->SetNumPredTicks(predTicks);

	_ballPredTracker->UpdatePredFromArena(this);
	return _ballPredTracker->GetBallStateForTime(time);
}

bool Arena::GetPredictedScore(float& outScoreTime, Vec& outScorePos) const {
	if (!_ballPredTracker)
		return false;

	const int predScoreStepIndex = _ballPredTracker->predScoreStepIndex;
	if (predScoreStepIndex > 0 && _ballPredTracker->predData.size() > predScoreStepIndex) {
		outScoreTime = predScoreStepIndex * _ballPredTracker->ballPredArena->tickTime;
		outScorePos = _ballPredTracker->predData[predScoreStepIndex].pos;
		return true;
	}

	return false;
}

DropshotTilesState Arena::GetDropshotTilesState() const {
	if (gameMode != GameMode::DROPSHOT)
		throw std::logic_error("Arena::GetDropshotTilesState() is only valid in Dropshot");

	return _backend->GetTiles();
}

void Arena::SetDropshotTilesState(const DropshotTilesState& tilesState) {
	if (gameMode != GameMode::DROPSHOT)
		throw std::logic_error("Arena::SetDropshotTilesState() is only valid in Dropshot");

	_backend->SetTiles(tilesState);
}

}
