#include "V3Backend.h"

#include <climits>
#include <cmath>
#include <stdexcept>

namespace RocketSim::EngineBackend {

namespace {
	void CheckRsf(int status) {
		if (status != 0)
			throw std::runtime_error(std::string("RocketSim v3: ") + rsf_last_error());
	}

	uint32_t GameModeToRsf(GameMode mode) {
		switch (mode) {
		case GameMode::SOCCAR:
			return 0;
		case GameMode::HOOPS:
			return 1;
		case GameMode::HEATSEEKER:
			return 2;
		case GameMode::SNOWDAY:
			return 3;
		case GameMode::DROPSHOT:
			return 4;
		case GameMode::THE_VOID:
			return 5;
		default:
			throw std::invalid_argument("RocketSim v3: unsupported GameMode " + std::to_string((int)mode));
		}
	}

	uint32_t MemoryModeToRsf(ArenaMemWeightMode mode) {
		switch (mode) {
		case ArenaMemWeightMode::LIGHT:
			return 2;
		case ArenaMemWeightMode::BALANCED:
			return 1;
		default:
			return 0;
		}
	}

	Vec FromRsf(const RsfVec3& v) { return Vec(v.x, v.y, v.z); }
	RsfVec3 ToRsf(const Vec& v) { return RsfVec3{ v.x, v.y, v.z }; }

	RotMat FromRsf(const RsfRotMat& m) { return RotMat(FromRsf(m.forward), FromRsf(m.right), FromRsf(m.up)); }
	RsfRotMat ToRsf(const RotMat& m) { return RsfRotMat{ ToRsf(m.forward), ToRsf(m.right), ToRsf(m.up) }; }

	CarControls FromRsf(const RsfCarControls& c) {
		CarControls out;
		out.throttle = c.throttle;
		out.steer = c.steer;
		out.pitch = c.pitch;
		out.yaw = c.yaw;
		out.roll = c.roll;
		out.jump = c.jump;
		out.boost = c.boost;
		out.handbrake = c.handbrake;
		return out;
	}

	RsfCarControls ToRsf(const CarControls& c) {
		return RsfCarControls{ c.throttle, c.steer, c.pitch, c.yaw, c.roll, (uint8_t)c.jump, (uint8_t)c.boost, (uint8_t)c.handbrake };
	}

	RsfWheelPairConfig ToRsf(const WheelPairConfig& config) {
		return RsfWheelPairConfig{ config.wheelRadius, config.suspensionRestLength, ToRsf(config.connectionPointOffset) };
	}

	RsfCarConfig ToRsf(const CarConfig& config) {
		return RsfCarConfig{
			ToRsf(config.hitboxSize),
			ToRsf(config.hitboxPosOffset),
			ToRsf(config.frontWheels),
			ToRsf(config.backWheels),
			(uint8_t)config.threeWheels,
			config.dodgeDeadzone
		};
	}

	void PhysFromRsf(PhysState& out, const RsfPhysState& phys) {
		out.pos = FromRsf(phys.pos);
		out.rotMat = FromRsf(phys.rotMat);
		out.vel = FromRsf(phys.vel);
		out.angVel = FromRsf(phys.angVel);
	}

	RsfPhysState PhysToRsf(const PhysState& phys) {
		return RsfPhysState{ ToRsf(phys.pos), ToRsf(phys.rotMat), ToRsf(phys.vel), ToRsf(phys.angVel) };
	}

	/** Every ball field except the Dropshot damage tick, whose clock depends on the caller. **/
	BallState BallFromRsf(const RsfBallState& rs) {
		BallState state;
		PhysFromRsf(state, rs.phys);
		state.hsInfo.yTargetDir = rs.hsTarget;
		state.hsInfo.curTargetSpeed = rs.hsSpeed;
		state.hsInfo.timeSinceHit = rs.hsTime;
		state.dsInfo.chargeLevel = rs.dsCharge;
		state.dsInfo.accumulatedHitForce = rs.dsForce;
		state.dsInfo.yTargetDir = rs.dsTarget;
		state.dsInfo.hasDamaged = false;
		state.dsInfo.lastDamageTick = 0;
		return state;
	}

	RsfBallState BallToRsf(const BallState& state) {
		RsfBallState rs = {};
		rs.phys = PhysToRsf(state);
		rs.hsTarget = (int8_t)state.hsInfo.yTargetDir;
		rs.hsSpeed = state.hsInfo.curTargetSpeed;
		rs.hsTime = state.hsInfo.timeSinceHit;
		rs.dsCharge = (uint8_t)state.dsInfo.chargeLevel;
		rs.dsForce = state.dsInfo.accumulatedHitForce;
		rs.dsTarget = (int8_t)state.dsInfo.yTargetDir;
		return rs;
	}

	/** Ticks between `state`'s last damage and `now`; false when it never damaged or the damage tick is after `now`. **/
	bool DamageAge(const BallState& state, uint64_t now, uint64_t& outAge) {
		if (!state.dsInfo.hasDamaged || state.dsInfo.lastDamageTick > now)
			return false;

		outAge = now - state.dsInfo.lastDamageTick;
		return true;
	}

	void SetDamageFromAge(BallState& state, uint64_t now, uint64_t age) {
		state.dsInfo.hasDamaged = true;
		state.dsInfo.lastDamageTick = now >= age ? now - age : 0;
	}
}

void InitV3(const std::filesystem::path& collisionMeshesFolder, bool silent) {
	if (V3IsInitialized())
		return;

	if (!silent)
		RS_LOG("Initializing RocketSim v3, ported to Rust by VirxEC...");
	const uint64_t startMS = RS_CUR_MS();

	// The Rust logger writes straight to the process's stderr, which the GUI log never shows, so it stays silent and v3 logs here instead.
	const std::u8string utf8Path = collisionMeshesFolder.u8string();
	if (rsf_init(reinterpret_cast<const char*>(utf8Path.c_str()), 1) != 0)
		throw std::runtime_error(RS_STR("RocketSim v3 failed to load collision meshes from " << collisionMeshesFolder << ": " << rsf_last_error()));

	if (!silent)
		RS_LOG("Finished initializing RocketSim v3 in " << ((RS_CUR_MS() - startMS) / 1000.f) << "s!");
}

bool V3IsInitialized() {
	return rsf_is_initialized() != 0;
}

bool V3HasCollisionMeshes(GameMode gameMode) {
	return rsf_has_meshes(GameModeToRsf(gameMode)) != 0;
}

V3Backend::V3Backend(GameMode mode, const ArenaConfig& config, float tickRate, const uint64_t& arenaTickCount)
	: _gameMode(mode), _noBallRot(config.noBallRot), _arenaTickCount(arenaTickCount) {
	const MutatorConfig mutators(mode);
	_padCooldownBig = mutators.boostPadCooldown_Big;
	_padCooldownSmall = mutators.boostPadCooldown_Small;

	if (!V3IsInitialized())
		throw std::logic_error("RocketSim v3: Arena::Create() before Init()");

	// Callers pass rates rebuilt from a tick time, such as 1 / (1 / 120.f), which can miss 120 by a rounding step.
	if (std::abs(tickRate - 120.f) > 0.01f)
		throw std::invalid_argument(RS_STR("RocketSim v3: only 120Hz tick rate is supported (got " << tickRate << ")"));
	if (config.useCustomBoostPads)
		throw std::invalid_argument("RocketSim v3: custom boost-pad layouts are not supported");

	const ArenaConfig defaults;
	if (config.minPos != defaults.minPos || config.maxPos != defaults.maxPos || config.maxAABBLen != defaults.maxAABBLen ||
		config.maxObjects != defaults.maxObjects || config.useCustomBroadphase != defaults.useCustomBroadphase)
		throw std::invalid_argument("RocketSim v3: v2-specific broadphase settings are unsupported");

	const int64_t randomSeed = -1;
	_arena = rsf_arena_new(GameModeToRsf(mode), randomSeed, (uint8_t)config.noBallRot, MemoryModeToRsf(config.memWeightMode));
	if (!_arena)
		throw std::runtime_error(std::string("RocketSim v3: ") + rsf_last_error());

	_engineTick = rsf_arena_tick_count(_arena);

	const uint32_t numPads = rsf_arena_num_pads(_arena);
	_padConfigs.resize(numPads);
	for (uint32_t i = 0; i < numPads; i++)
		CheckRsf(rsf_arena_get_pad_config(_arena, i, &_padConfigs[i]));
	_pads.resize(numPads);

	const uint32_t numTiles = rsf_arena_num_tiles(_arena);
	if (numTiles == UINT32_MAX)
		throw std::runtime_error(std::string("RocketSim v3: ") + rsf_last_error());
	_tiles.resize(numTiles);

	_bumps.resize(16);
	RefreshSnapshot();
}

V3Backend::~V3Backend() {
	rsf_ball_predictor_free(_predictor);
	rsf_arena_free(_arena);
}

void V3Backend::PrepareStepOutput(RsfStepOutput& out, uint32_t bumpsCap) {
	out.cars = _cars.data();
	out.carResults = _carResults.data();
	out.numCars = (uint32_t)_cars.size();
	out.ball = &_ball;
	out.pads = _pads.data();
	out.numPads = (uint32_t)_pads.size();
	out.tiles = _tiles.data();
	out.numTiles = (uint32_t)_tiles.size();
	out.bumps = _bumps.data();
	out.bumpsCap = bumpsCap;
	out.bumpsWritten = 0;
	out.isBallScored = 0;
}

void V3Backend::RefreshSnapshot() {
	RsfStepOutput out;
	const uint32_t noBumps = 0;
	PrepareStepOutput(out, noBumps);

	const uint32_t zeroTicks = 0;
	CheckRsf(rsf_arena_step_snapshot(_arena, _controls.data(), zeroTicks, &out));

	_isBallScored = out.isBallScored != 0;
	_isBallScoredStale = false;
}

uint32_t V3Backend::AddCar(Team team, const CarConfig& config) {
	const RsfCarConfig rsConfig = ToRsf(config);
	const uint32_t idx = rsf_arena_add_car(_arena, (uint32_t)team, &rsConfig);
	if (idx == UINT32_MAX)
		throw std::runtime_error(std::string("RocketSim v3: ") + rsf_last_error());

	_controls.push_back({});
	_cars.push_back({});
	_carResults.push_back({});
	_carExtras.push_back({});
	RefreshSnapshot();
	return idx;
}

void V3Backend::Step(int ticks, uint64_t startTick, const CarControls* controls, StepResult& out) {
	const size_t numCars = _cars.size();
	for (size_t i = 0; i < numCars; i++)
		_controls[i] = ToRsf(controls[i]);

	// Each ordered pair of cars registers at most one bump per tick.
	const size_t orderedCarPairs = numCars > 1 ? numCars * (numCars - 1) : 0;
	const size_t maxBumps = std::max<size_t>(16, orderedCarPairs * ticks);
	if (_bumps.size() < maxBumps)
		_bumps.resize(maxBumps);

	RsfStepOutput snapshot;
	PrepareStepOutput(snapshot, (uint32_t)_bumps.size());
	CheckRsf(rsf_arena_step_snapshot(_arena, _controls.data(), (uint32_t)ticks, &snapshot));
	if (snapshot.bumpsWritten > snapshot.bumpsCap)
		throw std::runtime_error(RS_STR("RocketSim v3: a step reported " << snapshot.bumpsWritten << " bumps, more than the " << snapshot.bumpsCap << " it had room for"));

	_engineTick += ticks;
	_isBallScored = snapshot.isBallScored != 0;
	_isBallScoredStale = false;
	_ballTicksSinceSet += ticks;

	for (size_t i = 0; i < numCars; i++) {
		const RsfStepCarResult& result = _carResults[i];
		CarExtras& extras = _carExtras[i];

		extras.ticksSinceSet += ticks;
		extras.wasOnGroundDuringStep = extras.wasOnGroundDuringStep || result.wasOnGround;
		extras.groundContactGraceUsed = false;
		extras.jumpResetBlockTicks = 0;

		if (result.hitBall) {
			BallHitInfo& hit = extras.ballHitInfo;
			hit.isValid = true;
			hit.tickCountWhenHit = startTick + result.hitTick;
			hit.ballPos = FromRsf(result.hitBallPos);
			hit.relativePosOnBall = FromRsf(result.hitContactPoint) - hit.ballPos;
			hit.extraHitVel = FromRsf(result.hitExtraVel);
			hit.tickCountWhenExtraImpulseApplied = 0;
		}
	}

	out.bumps.clear();
	for (uint32_t i = 0; i < snapshot.bumpsWritten; i++) {
		const RsfBump& bump = _bumps[i];
		out.bumps.push_back({ bump.bumper, bump.victim, bump.isDemo != 0 });
	}
}

CarState V3Backend::GetCarState(uint32_t idx) {
	const RsfCarState& rs = _cars.at(idx);
	const CarExtras& extras = _carExtras[idx];

	CarState state;
	PhysFromRsf(state, rs.phys);
	state.tickCountSinceUpdate = extras.ticksSinceSet;
	state.isOnGround = rs.isOnGround;
	state.wasOnGroundDuringStep = extras.wasOnGroundDuringStep;
	state.groundContactGraceUsed = extras.groundContactGraceUsed;
	state.obsOnGround = rs.isOnGround;
	for (int i = 0; i < 4; i++)
		state.wheelsWithContact[i] = rs.wheelsWithContact[i];
	state.hasJumped = rs.hasJumped;
	state.hasDoubleJumped = rs.hasDoubleJumped;
	state.hasFlipped = rs.hasFlipped;
	state.flipRelTorque = FromRsf(rs.flipRelTorque);
	state.jumpTime = rs.jumpTicks / 120.f;
	state.flipTime = rs.flipTime;
	state.isFlipping = rs.isFlipping;
	state.isJumping = rs.isJumping;
	state.airTime = rs.airTime;
	state.airTimeSinceJump = rs.airTimeSinceJump;
	state.boost = rs.boost;
	state.timeSinceBoosted = rs.timeSinceBoosted;
	state.isBoosting = rs.isBoosting;
	state.boostingTime = rs.boostingTime;
	state.isSupersonic = rs.isSupersonic;

	// v3 keeps a grace timer since dropping below supersonic instead of v2's time spent supersonic.
	state.supersonicTime = 0;

	state.handbrakeVal = rs.handbrakeVal;
	state.isAutoFlipping = rs.isAutoFlipping;
	state.autoFlipTimer = rs.autoFlipTimer;
	state.autoFlipTorqueScale = rs.autoFlipTorqueScale;
	state.worldContact.hasContact = rs.hasWorldContact;
	state.worldContact.contactNormal = FromRsf(rs.worldContactNormal);
	state.carContact.otherCarID = rs.hasBumpCooldownVictim ? rs.bumpCooldownVictim + 1 : 0;
	state.carContact.cooldownTimer = rs.bumpCooldownTimer;
	state.isDemoed = rs.isDemoed;
	state.demoRespawnTimer = rs.demoRespawnTimer;
	state.ballHitInfo = extras.ballHitInfo;
	state.lastControls = FromRsf(rs.prevControls);
	state.jumpResetBlockTicks = extras.jumpResetBlockTicks;
	return state;
}

void V3Backend::SetCarState(uint32_t idx, const CarState& state) {
	const RsfCarState& previous = _cars.at(idx);

	RsfCarState rs = {};
	rs.phys = PhysToRsf(state);
	rs.controls = previous.controls;
	rs.prevControls = ToRsf(state.lastControls);
	rs.isOnGround = state.isOnGround;
	for (int i = 0; i < 4; i++)
		rs.wheelsWithContact[i] = state.wheelsWithContact[i];
	rs.hasJumped = state.hasJumped;
	rs.hasDoubleJumped = state.hasDoubleJumped;
	rs.hasFlipped = state.hasFlipped;
	rs.flipRelTorque = ToRsf(state.flipRelTorque);
	rs.jumpTicks = (uint32_t)std::max(0L, std::lround(state.jumpTime * 120.f));
	rs.flipTime = state.flipTime;
	rs.isFlipping = state.isFlipping;
	rs.isJumping = state.isJumping;
	rs.airTime = state.airTime;
	rs.airTimeSinceJump = state.airTimeSinceJump;
	rs.boost = state.boost;
	rs.timeSinceBoosted = state.timeSinceBoosted;
	rs.isBoosting = state.isBoosting;
	rs.boostingTime = state.boostingTime;
	rs.isSupersonic = state.isSupersonic;
	rs.supersonicGraceTimer = 0;
	rs.handbrakeVal = state.handbrakeVal;
	rs.isAutoFlipping = state.isAutoFlipping;
	rs.autoFlipTimer = state.autoFlipTimer;
	rs.autoFlipTorqueScale = state.autoFlipTorqueScale;
	rs.bumpCooldownTimer = state.carContact.cooldownTimer;
	if (state.carContact.otherCarID != 0 && state.carContact.otherCarID <= _cars.size()) {
		rs.bumpCooldownVictim = state.carContact.otherCarID - 1;
		rs.hasBumpCooldownVictim = 1;
	}
	rs.hasWorldContact = state.worldContact.hasContact;
	rs.worldContactNormal = ToRsf(state.worldContact.contactNormal);
	rs.isDemoed = state.isDemoed;
	rs.demoRespawnTimer = state.demoRespawnTimer;

	CheckRsf(rsf_arena_set_car_state_preserving(_arena, idx, &rs));

	// Mirrors what the preserving setter keeps, so the cache matches the engine without reading the car back.
	rs.jumpHoldBroken = previous.jumpHoldBroken;
	rs.lastExtraHitTick = previous.lastExtraHitTick;
	rs.hasLastExtraHitTick = previous.hasLastExtraHitTick;
	for (int i = 0; i < 4; i++)
		if (rs.wheelsWithContact[i] && previous.wheelsWithContact[i])
			rs.wheelContacts[i] = previous.wheelContacts[i];
	_cars[idx] = rs;

	CarExtras& extras = _carExtras[idx];
	extras.ballHitInfo = state.ballHitInfo;
	extras.wasOnGroundDuringStep = state.wasOnGroundDuringStep;
	extras.groundContactGraceUsed = state.groundContactGraceUsed;
	extras.jumpResetBlockTicks = state.jumpResetBlockTicks;
	extras.ticksSinceSet = 0;
}

void V3Backend::ResetCarStepContact(uint32_t idx) {
	_carExtras.at(idx).wasOnGroundDuringStep = false;
}

BallState V3Backend::GetBallState() {
	BallState state = BallFromRsf(_ball);
	state.tickCountSinceUpdate = _ballTicksSinceSet;

	if (_ball.dsHasDamageTick) {
		const uint64_t age = _engineTick >= _ball.dsLastDamageTick ? _engineTick - _ball.dsLastDamageTick : 0;
		SetDamageFromAge(state, _arenaTickCount, age);
	}

	return state;
}

void V3Backend::SetBallState(const BallState& state) {
	RsfBallState rs = BallToRsf(state);

	// A damage tick older than the engine's whole run is outside the damage rate limit, the same as never having damaged.
	uint64_t age;
	if (DamageAge(state, _arenaTickCount, age) && age <= _engineTick) {
		rs.dsHasDamageTick = 1;
		rs.dsLastDamageTick = _engineTick - age;
	}

	rs.tickCountSinceKickoff = _ball.tickCountSinceKickoff;
	CheckRsf(rsf_arena_set_ball_state_preserving(_arena, &rs));

	_ball = rs;
	_ballTicksSinceSet = 0;
	_isBallScoredStale = true;
}

uint32_t V3Backend::NumPads() {
	return (uint32_t)_padConfigs.size();
}

BoostPadConfig V3Backend::GetPadConfig(uint32_t idx) {
	const RsfPadConfig& config = _padConfigs.at(idx);
	return { FromRsf(config.pos), config.isBig != 0 };
}

ArenaBackend::BoostPadStateRaw V3Backend::GetPadState(uint32_t idx) {
	const RsfPadState& state = _pads.at(idx);
	return { state.isActive != 0, state.cooldown, UINT32_MAX, UINT32_MAX };
}

void V3Backend::GetPadTimers(BoostPadTimers* out) {
	for (const RsfPadState& state : _pads)
		*out++ = { state.isActive != 0, state.cooldown };
}

void V3Backend::SetPadState(uint32_t idx, const BoostPadStateRaw& state) {
	// v3 derives isActive from cooldown <= 0, so keep the pair consistent.
	RsfPadState rs;
	rs.cooldown = state.isActive ? 0.f : std::max(state.cooldown, 1 / 120.f);
	rs.isActive = state.isActive;
	CheckRsf(rsf_arena_set_pad_state(_arena, idx, &rs));

	// The engine stores a cooldown as whole ticks since pickup, clamped to the pad's maximum, and rebuilds seconds from that.
	if (rs.cooldown > 0) {
		const float maxCooldown = _padConfigs.at(idx).isBig ? _padCooldownBig : _padCooldownSmall;
		const float ticksSincePickup = std::round(std::max(maxCooldown - rs.cooldown, 0.f) * 120.f);
		rs.cooldown = std::max(maxCooldown - ticksSincePickup * (1.f / 120.f), 0.f);
		rs.isActive = rs.cooldown <= 0;
	}

	_pads.at(idx) = rs;
}

void V3Backend::ResetToRandomKickoff(int seed) {
	CheckRsf(rsf_arena_reset_kickoff(_arena, seed));
	RefreshSnapshot();

	for (CarExtras& extras : _carExtras)
		extras = {};
	_ballTicksSinceSet = 0;
}

bool V3Backend::IsBallScored() {
	if (_isBallScoredStale) {
		const int scored = rsf_arena_is_ball_scored(_arena);
		if (scored < 0)
			throw std::runtime_error(std::string("RocketSim v3: ") + rsf_last_error());

		_isBallScored = scored != 0;
		_isBallScoredStale = false;
	}

	return _isBallScored;
}

DropshotTilesState V3Backend::GetTiles() {
	DropshotTilesState tiles;
	if (_tiles.empty())
		return tiles;

	for (int team = 0; team < RLConst::Dropshot::TEAM_AMOUNT; team++)
		for (int tile = 0; tile < RLConst::Dropshot::NUM_TILES_PER_TEAM; tile++)
			tiles.states[team][tile].damageState = _tiles[team * RLConst::Dropshot::NUM_TILES_PER_TEAM + tile];

	return tiles;
}

void V3Backend::SetTiles(const DropshotTilesState& tiles) {
	std::vector<uint8_t> states(RLConst::Dropshot::TEAM_AMOUNT * RLConst::Dropshot::NUM_TILES_PER_TEAM);
	for (int team = 0; team < RLConst::Dropshot::TEAM_AMOUNT; team++)
		for (int tile = 0; tile < RLConst::Dropshot::NUM_TILES_PER_TEAM; tile++)
			states[team * RLConst::Dropshot::NUM_TILES_PER_TEAM + tile] = tiles.states[team][tile].damageState;

	CheckRsf(rsf_arena_set_tiles(_arena, states.data(), (uint32_t)states.size()));
	_tiles = std::move(states);
}

int V3Backend::PredictBall(const BallState& start, int numTicks, BallState* out) {
	if (numTicks <= 0)
		return -1;

	if (!_predictor) {
		_predictor = rsf_ball_predictor_new(GameModeToRsf(_gameMode), (uint8_t)_noBallRot);
		if (!_predictor)
			throw std::runtime_error(std::string("RocketSim v3: ") + rsf_last_error());
	}

	if (!_tiles.empty() && _tiles != _predictorTiles) {
		CheckRsf(rsf_ball_predictor_set_tiles(_predictor, _tiles.data(), (uint32_t)_tiles.size()));
		_predictorTiles = _tiles;
	}

	// The predictor reads and reports damage ticks as ages.
	RsfBallState rsStart = BallToRsf(start);
	uint64_t startAge;
	if (DamageAge(start, _predictTick, startAge)) {
		rsStart.dsHasDamageTick = 1;
		rsStart.dsLastDamageTick = startAge;
	}

	_predictScratch.resize(numTicks);
	const int32_t firstScoringIndex = rsf_ball_predict(_predictor, &rsStart, (uint32_t)numTicks, _predictScratch.data());
	if (firstScoringIndex == INT32_MIN)
		throw std::runtime_error(std::string("RocketSim v3: ") + rsf_last_error());

	for (int i = 0; i < numTicks; i++) {
		const RsfBallState& predicted = _predictScratch[i];
		out[i] = BallFromRsf(predicted);
		out[i].tickCountSinceUpdate = i + 1;
		if (predicted.dsHasDamageTick)
			SetDamageFromAge(out[i], _predictTick + i + 1, predicted.dsLastDamageTick);
	}

	_predictTick += numTicks;
	return firstScoringIndex;
}

}
