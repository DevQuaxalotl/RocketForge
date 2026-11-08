#include "../../../RocketSim/src/RocketSim.h"

#include "V2Backend.h"

namespace RocketSim::EngineBackend {

void InitV2(const std::filesystem::path& collisionMeshesFolder, bool silent) {
	RocketSimV2::Init(collisionMeshesFolder, silent);
}

RocketSimStage GetV2Stage() {
	switch (RocketSimV2::GetStage()) {
	case RocketSimV2::RocketSimStage::INITIALIZING:
		return RocketSimStage::INITIALIZING;
	case RocketSimV2::RocketSimStage::INITIALIZED:
		return RocketSimStage::INITIALIZED;
	default:
		return RocketSimStage::UNINITIALIZED;
	}
}

bool V2HasCollisionMeshes(GameMode gameMode) {
	return !RocketSimV2::GetArenaCollisionShapes(gameMode).empty();
}

V2Backend::V2Backend(GameMode mode, const ArenaConfig& config, float tickRate) {
	RocketSimV2::ArenaConfig v2Config;
	v2Config.memWeightMode = config.memWeightMode == ArenaMemWeightMode::LIGHT
		? RocketSimV2::ArenaMemWeightMode::LIGHT
		: RocketSimV2::ArenaMemWeightMode::HEAVY;
	v2Config.minPos = config.minPos;
	v2Config.maxPos = config.maxPos;
	v2Config.maxAABBLen = config.maxAABBLen;
	v2Config.noBallRot = config.noBallRot;
	v2Config.useCustomBroadphase = config.useCustomBroadphase;
	v2Config.maxObjects = config.maxObjects;
	v2Config.useCustomBoostPads = config.useCustomBoostPads;
	for (const BoostPadConfig& pad : config.customBoostPads)
		v2Config.customBoostPads.push_back({ pad.pos, pad.isBig });

	_arena = RocketSimV2::Arena::Create(mode, v2Config, tickRate);
	_arena->SetCarBumpCallback(OnBump, this);
}

V2Backend::~V2Backend() {
	delete _arena;
}

uint32_t V2Backend::AddCar(Team team, const CarConfig& config) {
	_cars.push_back(_arena->AddCar(team, config));
	return (uint32_t)_cars.size() - 1;
}

void V2Backend::Step(int ticks, uint64_t startTick, const CarControls* controls, StepResult& out) {
	const size_t numCars = _cars.size();
	for (size_t i = 0; i < numCars; i++)
		_cars[i]->controls = controls[i];

	out.bumps.clear();
	_arena->tickCount = startTick;
	_stepResult = &out;
	_arena->Step(ticks);
	_stepResult = nullptr;
}

CarState V2Backend::GetCarState(uint32_t idx) {
	return _cars[idx]->GetState();
}

void V2Backend::SetCarState(uint32_t idx, const CarState& state) {
	_cars[idx]->SetState(state);
}

void V2Backend::ResetCarStepContact(uint32_t idx) {
	_cars[idx]->ResetStepContactState();
}

BallState V2Backend::GetBallState() {
	return _arena->ball->GetState();
}

void V2Backend::SetBallState(const BallState& state) {
	_arena->ball->SetState(state);
}

uint32_t V2Backend::NumPads() {
	return (uint32_t)_arena->_boostPads.size();
}

BoostPadConfig V2Backend::GetPadConfig(uint32_t idx) {
	const RocketSimV2::BoostPadConfig& config = _arena->_boostPads[idx]->config;
	return { config.pos, config.isBig };
}

ArenaBackend::BoostPadStateRaw V2Backend::GetPadState(uint32_t idx) {
	const RocketSimV2::BoostPadState& state = _arena->_boostPads[idx]->_internalState;
	return { state.isActive, state.cooldown, IndexOf(state.curLockedCar), IndexOfId(state.prevLockedCarID) };
}

void V2Backend::GetPadTimers(BoostPadTimers* out) {
	for (RocketSimV2::BoostPad* pad : _arena->_boostPads)
		*out++ = { pad->_internalState.isActive, pad->_internalState.cooldown };
}

void V2Backend::SetPadState(uint32_t idx, const BoostPadStateRaw& state) {
	RocketSimV2::BoostPadState v2State;
	v2State.isActive = state.isActive;
	v2State.cooldown = state.cooldown;
	v2State.curLockedCar = state.lockedCarIdx == UINT32_MAX ? nullptr : _cars[state.lockedCarIdx];
	v2State.prevLockedCarID = state.prevLockedCarIdx == UINT32_MAX ? 0 : _cars[state.prevLockedCarIdx]->id;
	_arena->_boostPads[idx]->SetState(v2State);
}

void V2Backend::ResetToRandomKickoff(int seed) {
	_arena->ResetToRandomKickoff(seed);
}

bool V2Backend::IsBallScored() {
	return _arena->IsBallScored();
}

DropshotTilesState V2Backend::GetTiles() {
	return _arena->GetDropshotTilesState();
}

void V2Backend::SetTiles(const DropshotTilesState& tiles) {
	_arena->SetDropshotTilesState(tiles);
}

int V2Backend::PredictBall(const BallState& start, int numTicks, BallState* out) {
	// Steps the v2 arena directly so its tick counter keeps advancing across predictions; Dropshot tile damage compares the ball's lastDamageTick against it.
	_arena->ball->SetState(start);

	int firstScoringIndex = -1;
	for (int i = 0; i < numTicks; i++) {
		_arena->Step(1);
		out[i] = _arena->ball->GetState();
		if (firstScoringIndex < 0 && _arena->IsBallScored())
			firstScoringIndex = i;
	}

	return firstScoringIndex;
}

void V2Backend::OnBump(RocketSimV2::Arena*, RocketSimV2::Car* bumper, RocketSimV2::Car* victim, bool isDemo, void* userInfo) {
	V2Backend* backend = static_cast<V2Backend*>(userInfo);
	if (backend->_stepResult)
		backend->_stepResult->bumps.push_back({ backend->IndexOf(bumper), backend->IndexOf(victim), isDemo });
}

uint32_t V2Backend::IndexOf(const RocketSimV2::Car* car) const {
	return car ? IndexOfId(car->id) : UINT32_MAX;
}

uint32_t V2Backend::IndexOfId(uint32_t carId) const {
	return carId ? carId - 1 : UINT32_MAX;
}

}
