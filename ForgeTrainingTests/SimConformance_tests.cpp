#include <gtest/gtest.h>

#include "TestSupport.h"

#include <RocketSim/Includes.h>

#include <map>
#include <memory>
#include <vector>

using namespace RocketSim;

namespace {
	struct BumpEvent {
		Arena* arena;
		Car* bumper;
		Car* victim;
		bool isDemo;
	};

	void CaptureBump(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo) {
		static_cast<std::vector<BumpEvent>*>(userInfo)->push_back({ arena, bumper, victim, isDemo });
	}

	CarState MakeGroundedCar(const Vec& pos, const Vec& forward, const Vec& vel) {
		CarState state;
		state.pos = pos;
		state.rotMat = RotMat::LookAt(forward, Vec(0, 0, 1));
		state.vel = vel;
		state.isOnGround = true;
		state.obsOnGround = true;
		return state;
	}

	BallState MakeBall(const Vec& pos, const Vec& vel) {
		BallState state;
		state.pos = pos;
		state.vel = vel;
		return state;
	}

	void ExpectVecNear(const Vec& actual, const Vec& expected, float tolerance) {
		EXPECT_NEAR(actual.x, expected.x, tolerance);
		EXPECT_NEAR(actual.y, expected.y, tolerance);
		EXPECT_NEAR(actual.z, expected.z, tolerance);
	}

	void ExpectRotMatNear(const RotMat& actual, const RotMat& expected, float tolerance) {
		ExpectVecNear(actual.forward, expected.forward, tolerance);
		ExpectVecNear(actual.right, expected.right, tolerance);
		ExpectVecNear(actual.up, expected.up, tolerance);
	}

	/** Places a car driving at full throttle into a resting ball 400 uu ahead of it. **/
	Car* SetUpTouch(Arena& arena) {
		Car* car = arena.AddCar(Team::BLUE);
		car->SetState(MakeGroundedCar(Vec(0, -400, 17), Vec(0, 1, 0), Vec(0, 1500, 0)));
		car->controls.throttle = 1;
		arena.ball->SetState(MakeBall(Vec(0, 0, 93), Vec()));
		return car;
	}

	/** A charged ball dropping fast onto Orange's half, which it is allowed to damage. **/
	BallState MakeChargedDropshotBall() {
		BallState ball = MakeBall(Vec(0, 2500, 400), Vec(0, 0, -2500));
		ball.dsInfo.chargeLevel = 2;
		ball.dsInfo.yTargetDir = 1;
		return ball;
	}

	int CountDamagedTiles(const DropshotTilesState& tiles) {
		int damaged = 0;
		for (int team = 0; team < RLConst::Dropshot::TEAM_AMOUNT; team++)
			for (int tile = 0; tile < RLConst::Dropshot::NUM_TILES_PER_TEAM; tile++)
				damaged += tiles.states[team][tile].damageState != DropshotTileState::STATE_FULL;
		return damaged;
	}

	DropshotTilesState AllTiles(uint8_t damageState) {
		DropshotTilesState tiles;
		for (int team = 0; team < RLConst::Dropshot::TEAM_AMOUNT; team++)
			for (int tile = 0; tile < RLConst::Dropshot::NUM_TILES_PER_TEAM; tile++)
				tiles.states[team][tile].damageState = damageState;
		return tiles;
	}

	/** Steps one tick at a time until the first bump event or `maxTicks`. **/
	std::vector<BumpEvent> DriveIntoVictim(Arena& arena, float speed, int maxTicks) {
		Car* bumper = arena.AddCar(Team::BLUE);
		Car* victim = arena.AddCar(Team::ORANGE);

		CarState bumperState = MakeGroundedCar(Vec(0, 0, 17), Vec(1, 0, 0), Vec(speed, 0, 0));
		bumperState.isSupersonic = speed >= RLConst::SUPERSONIC_START_SPEED;
		bumper->SetState(bumperState);
		victim->SetState(MakeGroundedCar(Vec(250, 0, 17), Vec(1, 0, 0), Vec()));

		bumper->controls.throttle = 1;
		bumper->controls.boost = bumperState.isSupersonic;

		std::vector<BumpEvent> events;
		arena.SetCarBumpCallback(CaptureBump, &events);
		for (int tick = 0; tick < maxTicks && events.empty(); tick++)
			arena.Step(1);

		return events;
	}
}

class SimConformance : public ::testing::TestWithParam<EngineId> {
protected:
	void SetUp() override {
		ROCKETFORGE_REQUIRE_ROCKETSIM();
		ROCKETFORGE_REQUIRE_ENGINE(GetParam());
	}

	std::unique_ptr<Arena> Make(GameMode mode = GameMode::SOCCAR) {
		ArenaConfig config;
		config.engine = GetParam();
		return std::unique_ptr<Arena>(Arena::Create(mode, config));
	}
};

TEST_P(SimConformance, CarIdsAreSequentialInAddOrder) {
	std::unique_ptr<Arena> arena = Make();
	Car* first = arena->AddCar(Team::BLUE);
	Car* second = arena->AddCar(Team::ORANGE, CAR_CONFIG_DOMINUS);
	Car* third = arena->AddCar(Team::BLUE);

	EXPECT_EQ(first->id, 1u);
	EXPECT_EQ(second->id, 2u);
	EXPECT_EQ(third->id, 3u);
	EXPECT_EQ(second->team, Team::ORANGE);
	EXPECT_EQ(second->config.hitboxSize, CAR_CONFIG_DOMINUS.hitboxSize);

	ASSERT_EQ(arena->_cars.size(), size_t(3));
	EXPECT_EQ(arena->_cars[0], first);
	EXPECT_EQ(arena->_cars[1], second);
	EXPECT_EQ(arena->_cars[2], third);

	EXPECT_EQ(arena->GetCar(2), second);
	EXPECT_EQ(arena->GetCar(99), nullptr);
	EXPECT_EQ(arena->GetEngine(), GetParam());
}

TEST_P(SimConformance, StepAdvancesTickCountAndWritesStick) {
	std::unique_ptr<Arena> arena = Make();
	arena->AddCar(Team::BLUE);

	arena->Step(8);
	EXPECT_EQ(arena->tickCount, 8u);

	arena->tickCount = 1000;
	arena->Step(1);
	EXPECT_EQ(arena->tickCount, 1001u);
}

TEST_P(SimConformance, CarBallPadStateRoundTrip) {
	std::unique_ptr<Arena> arena = Make();
	Car* car = arena->AddCar(Team::BLUE);

	CarState carState = MakeGroundedCar(Vec(500, -1200, 17), Vec(0.6f, 0.8f, 0), Vec(300, 400, 0));
	carState.angVel = Vec(0.1f, -0.2f, 0.3f);
	carState.boost = 42;
	car->SetState(carState);

	CarState carRead = car->GetState();
	ExpectVecNear(carRead.pos, carState.pos, 1e-3f);
	ExpectVecNear(carRead.vel, carState.vel, 1e-3f);
	ExpectVecNear(carRead.angVel, carState.angVel, 1e-4f);
	ExpectRotMatNear(carRead.rotMat, carState.rotMat, 1e-4f);
	EXPECT_FLOAT_EQ(carRead.boost, 42);

	BallState ballState = MakeBall(Vec(-300, 800, 400), Vec(100, -200, 300));
	ballState.angVel = Vec(1, 2, -1);
	arena->ball->SetState(ballState);

	BallState ballRead = arena->ball->GetState();
	ExpectVecNear(ballRead.pos, ballState.pos, 1e-3f);
	ExpectVecNear(ballRead.vel, ballState.vel, 1e-3f);
	ExpectVecNear(ballRead.angVel, ballState.angVel, 1e-4f);

	ASSERT_FALSE(arena->_boostPads.empty());
	BoostPad* pad = arena->_boostPads[3];
	BoostPadState padState;
	padState.isActive = false;
	padState.cooldown = 3.5f;
	pad->SetState(padState);

	BoostPadState padRead = pad->GetState();
	EXPECT_FALSE(padRead.isActive);
	EXPECT_FLOAT_EQ(padRead.cooldown, 3.5f);
	EXPECT_EQ(padRead.curLockedCar, nullptr);
}

TEST_P(SimConformance, TouchRecordsTickAndBallPosAtTouch) {
	constexpr uint64_t START_TICK = 500;
	constexpr int STEP_TICKS = 40;

	// The same setup stepped tick by tick records where the ball was at the start of every tick.
	std::unique_ptr<Arena> tickByTick = Make();
	Car* tickByTickCar = SetUpTouch(*tickByTick);
	tickByTick->tickCount = START_TICK;

	std::map<uint64_t, Vec> ballPosAtTickStart;
	for (int tick = 0; tick < STEP_TICKS; tick++) {
		ballPosAtTickStart[tickByTick->tickCount] = tickByTick->ball->GetState().pos;
		tickByTick->Step(1);
	}

	std::unique_ptr<Arena> arena = Make();
	Car* car = SetUpTouch(*arena);
	arena->tickCount = START_TICK;
	arena->Step(STEP_TICKS);

	const BallHitInfo hit = car->GetState().ballHitInfo;
	ASSERT_TRUE(hit.isValid);
	EXPECT_GE(hit.tickCountWhenHit, START_TICK);
	EXPECT_LT(hit.tickCountWhenHit, START_TICK + STEP_TICKS);
	EXPECT_EQ(hit.tickCountWhenHit, tickByTickCar->GetState().ballHitInfo.tickCountWhenHit);

	ASSERT_TRUE(ballPosAtTickStart.count(hit.tickCountWhenHit));
	ExpectVecNear(hit.ballPos, ballPosAtTickStart[hit.tickCountWhenHit], 1e-3f);
}

TEST_P(SimConformance, WasOnGroundCoversWholeStep) {
	std::unique_ptr<Arena> arena = Make();
	Car* car = arena->AddCar(Team::BLUE);
	car->SetState(MakeGroundedCar(Vec(0, 0, 17), Vec(1, 0, 0), Vec()));
	arena->Step(30);
	ASSERT_TRUE(car->GetState().isOnGround);

	car->ResetStepContactState();
	car->controls.jump = true;
	arena->Step(8);

	const CarState state = car->GetState();
	EXPECT_FALSE(state.isOnGround);
	EXPECT_TRUE(state.wasOnGroundDuringStep);
}

TEST_P(SimConformance, SupersonicHitDemosAndSlowHitBumps) {
	std::unique_ptr<Arena> fastArena = Make();
	std::vector<BumpEvent> fast = DriveIntoVictim(*fastArena, 2300, 60);
	ASSERT_EQ(fast.size(), size_t(1));
	EXPECT_EQ(fast[0].arena, fastArena.get());
	EXPECT_EQ(fast[0].bumper, fastArena->_cars[0]);
	EXPECT_EQ(fast[0].victim, fastArena->_cars[1]);
	EXPECT_TRUE(fast[0].isDemo);

	std::unique_ptr<Arena> slowArena = Make();
	std::vector<BumpEvent> slow = DriveIntoVictim(*slowArena, 500, 120);
	ASSERT_EQ(slow.size(), size_t(1));
	EXPECT_EQ(slow[0].bumper, slowArena->_cars[0]);
	EXPECT_EQ(slow[0].victim, slowArena->_cars[1]);
	EXPECT_FALSE(slow[0].isDemo);
}

TEST_P(SimConformance, GoalIsDetected) {
	std::unique_ptr<Arena> arena = Make();
	arena->ball->SetState(MakeBall(Vec(0, 4900, 300), Vec(0, 2000, 0)));
	ASSERT_FALSE(arena->IsBallScored());

	for (int tick = 0; tick < 60 && !arena->IsBallScored(); tick++)
		arena->Step(1);

	EXPECT_TRUE(arena->IsBallScored());
}

TEST_P(SimConformance, SeededKickoffIsDeterministic) {
	// One car per team: v2 hands same-team cars their kickoff spots in pointer-hash order.
	std::unique_ptr<Arena> first = Make();
	std::unique_ptr<Arena> second = Make();
	for (Arena* arena : { first.get(), second.get() }) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
		arena->ResetToRandomKickoff(7);
	}

	for (size_t i = 0; i < first->_cars.size(); i++) {
		const CarState expected = first->_cars[i]->GetState();
		const CarState actual = second->_cars[i]->GetState();
		ExpectVecNear(actual.pos, expected.pos, 0);
		ExpectRotMatNear(actual.rotMat, expected.rotMat, 0);
		EXPECT_EQ(actual.boost, expected.boost);
	}

	ExpectVecNear(second->ball->GetState().pos, first->ball->GetState().pos, 0);
}

TEST_P(SimConformance, FutureBallMatchesSteppingABallOnlyArena) {
	constexpr int PREDICTED_TICKS = 120;
	const BallState start = MakeBall(Vec(-500, 1000, 600), Vec(800, -1200, 400));

	std::unique_ptr<Arena> arena = Make();
	arena->AddCar(Team::BLUE);
	arena->ball->SetState(start);

	// The prediction index is floor(time / tickTime), so ask for the middle of the tick.
	arena->GetFutureBallState(2.f);
	const BallState predicted = arena->GetFutureBallState((PREDICTED_TICKS + 0.5f) * arena->tickTime);

	std::unique_ptr<Arena> ballOnly = Make();
	ballOnly->ball->SetState(start);
	ballOnly->Step(PREDICTED_TICKS);

	EXPECT_TRUE(predicted.Matches(ballOnly->ball->GetState()));
}

TEST_P(SimConformance, SetStatePreservesUnmappedState) {
	const Vec offset(600, 400, 0);

	std::unique_ptr<Arena> teleported = Make();
	std::unique_ptr<Arena> twin = Make();
	Car* teleportedCar = teleported->AddCar(Team::BLUE);
	Car* twinCar = twin->AddCar(Team::BLUE);
	for (Car* car : { teleportedCar, twinCar })
		car->SetState(MakeGroundedCar(Vec(-1500, -2000, 17), Vec(1, 0, 0), Vec()));

	auto stepBoth = [&](bool jump, int ticks) {
		teleportedCar->controls.jump = jump;
		twinCar->controls.jump = jump;
		teleported->Step(ticks);
		twin->Step(ticks);
	};

	stepBoth(false, 30);
	stepBoth(true, 6);

	CarState moved = teleportedCar->GetState();
	ASSERT_TRUE(moved.isJumping);
	moved.pos += offset;
	teleportedCar->SetState(moved);

	// Hold the jump, release it, then press again: the jump's hold and the follow-up jump both depend on state the setter can't see.
	stepBoth(true, 10);
	stepBoth(false, 3);
	stepBoth(true, 20);

	const CarState teleportedState = teleportedCar->GetState();
	const CarState twinState = twinCar->GetState();
	ExpectVecNear(teleportedState.pos - offset, twinState.pos, 0.05f);
	ExpectVecNear(teleportedState.vel, twinState.vel, 0.05f);
	ExpectVecNear(teleportedState.angVel, twinState.angVel, 1e-3f);
	EXPECT_EQ(teleportedState.hasJumped, twinState.hasJumped);
	EXPECT_EQ(teleportedState.hasDoubleJumped, twinState.hasDoubleJumped);
	EXPECT_EQ(teleportedState.isOnGround, twinState.isOnGround);
}

TEST_P(SimConformance, PadStateReadsBackAsTheEngineKeepsIt) {
	std::unique_ptr<Arena> arena = Make();

	// v2 only ticks pad cooldowns in an arena with cars; this one is parked in the air, away from every pad.
	Car* car = arena->AddCar(Team::BLUE);
	CarState parked;
	parked.pos = Vec(0, 0, 1500);
	car->SetState(parked);

	BoostPad* smallPad = nullptr;
	BoostPad* bigPad = nullptr;
	for (BoostPad* pad : arena->_boostPads)
		(pad->config.isBig ? bigPad : smallPad) = pad;
	ASSERT_NE(smallPad, nullptr);
	ASSERT_NE(bigPad, nullptr);

	// Longer than a small pad's cooldown, and between ticks for the big pad.
	const float cooldowns[] = { 10.f, 3.3337f };
	BoostPad* pads[] = { smallPad, bigPad };
	float setReads[2];
	for (int i = 0; i < 2; i++) {
		BoostPadState state;
		state.isActive = false;
		state.cooldown = cooldowns[i];
		pads[i]->SetState(state);
		setReads[i] = pads[i]->GetState().cooldown;
	}

	// A step reads the engine's own value; one tick later it must be one tick shorter than what the setter reported.
	arena->Step(1);
	for (int i = 0; i < 2; i++) {
		SCOPED_TRACE(i);
		EXPECT_NEAR(pads[i]->GetState().cooldown, setReads[i] - arena->tickTime, 1e-4f);
	}
}

TEST_P(SimConformance, ReusedArenaSetStateMatchesFreshArena) {
	std::unique_ptr<Arena> reused = Make();
	std::unique_ptr<Arena> fresh = Make();
	Car* reusedCar = reused->AddCar(Team::BLUE);
	Car* freshCar = fresh->AddCar(Team::BLUE);

	reusedCar->SetState(MakeGroundedCar(Vec(-2000, -2000, 17), Vec(1, 0, 0), Vec()));
	reusedCar->controls.throttle = 1;
	reused->Step(120);

	const CarState state = MakeGroundedCar(Vec(1500, 1500, 17), Vec(0, 1, 0), Vec());
	reusedCar->SetState(state);
	freshCar->SetState(state);
	reusedCar->controls = {};
	reused->Step(30);
	fresh->Step(30);

	const CarState reusedState = reusedCar->GetState();
	const CarState freshState = freshCar->GetState();
	// Suspension and solver state outside CarState survive a teleport, so the cars measured up to 0.03 uu and 0.12 uu/s apart.
	// The bounds leave room for that and still catch state that leaks through SetState.
	ExpectVecNear(reusedState.pos, freshState.pos, 0.5f);
	ExpectVecNear(reusedState.vel, freshState.vel, 1.f);
}

TEST_P(SimConformance, DamageTickSurvivesStepsAndTickRewrites) {
	std::unique_ptr<Arena> arena = Make();
	arena->Step(10);
	arena->tickCount = 1000;

	BallState ball = MakeBall(Vec(0, 0, 300), Vec());
	ball.dsInfo.hasDamaged = true;
	ball.dsInfo.lastDamageTick = 995;
	arena->ball->SetState(ball);
	arena->Step(3);

	const BallState read = arena->ball->GetState();
	EXPECT_TRUE(read.dsInfo.hasDamaged);
	EXPECT_EQ(read.dsInfo.lastDamageTick, 995u);

	ball.dsInfo.lastDamageTick = 5000;
	EXPECT_NO_THROW(arena->ball->SetState(ball));
	EXPECT_NO_THROW(arena->Step(1));

	arena->tickCount = 2;
	EXPECT_NO_THROW(arena->ball->SetState(arena->ball->GetState()));
	EXPECT_NO_THROW(arena->Step(1));
}

TEST(V3Backend, DamageTickTranslationEdges) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	ROCKETFORGE_REQUIRE_ENGINE(EngineId::V3);

	ArenaConfig config;
	config.engine = EngineId::V3;
	std::unique_ptr<Arena> arena(Arena::Create(GameMode::SOCCAR, config));

	BallState ball = MakeBall(Vec(0, 0, 300), Vec());
	ball.dsInfo.hasDamaged = true;

	// Older than the engine's whole run (no ticks yet): outside the rate limit, so it reads back as never damaged.
	arena->tickCount = 1000;
	ball.dsInfo.lastDamageTick = 995;
	arena->ball->SetState(ball);
	EXPECT_FALSE(arena->ball->GetState().dsInfo.hasDamaged);

	arena->Step(10);
	ball.dsInfo.lastDamageTick = 1005;
	arena->ball->SetState(ball);
	ASSERT_TRUE(arena->ball->GetState().dsInfo.hasDamaged);

	// Rewriting tickCount below the age of 5 clamps the damage tick to 0; setting that back keeps an age of 2.
	arena->tickCount = 2;
	BallState clamped = arena->ball->GetState();
	EXPECT_TRUE(clamped.dsInfo.hasDamaged);
	EXPECT_EQ(clamped.dsInfo.lastDamageTick, 0u);
	arena->ball->SetState(clamped);
	arena->tickCount = 1010;
	EXPECT_EQ(arena->ball->GetState().dsInfo.lastDamageTick, 1008u);

	// A damage tick after tickCount is set as never having damaged.
	ball.dsInfo.lastDamageTick = 5000;
	arena->ball->SetState(ball);
	EXPECT_FALSE(arena->ball->GetState().dsInfo.hasDamaged);
}

TEST_P(SimConformance, DropshotTilesRoundTrip) {
	ROCKETFORGE_REQUIRE_DROPSHOT_MESHES();
	std::unique_ptr<Arena> arena = Make(GameMode::DROPSHOT);

	DropshotTilesState tiles = arena->GetDropshotTilesState();
	for (int team = 0; team < RLConst::Dropshot::TEAM_AMOUNT; team++)
		for (int tile = 0; tile < RLConst::Dropshot::NUM_TILES_PER_TEAM; tile++)
			EXPECT_EQ(tiles.states[team][tile].damageState, DropshotTileState::STATE_FULL);

	tiles.states[0][3].damageState = DropshotTileState::STATE_DAMAGED;
	tiles.states[1][40].damageState = DropshotTileState::STATE_BROKEN;
	arena->SetDropshotTilesState(tiles);
	arena->Step(1);

	const DropshotTilesState read = arena->GetDropshotTilesState();
	for (int team = 0; team < RLConst::Dropshot::TEAM_AMOUNT; team++)
		for (int tile = 0; tile < RLConst::Dropshot::NUM_TILES_PER_TEAM; tile++)
			EXPECT_EQ(read.states[team][tile].damageState, tiles.states[team][tile].damageState) << team << "/" << tile;
}

TEST_P(SimConformance, ChargedBallDamagesTiles) {
	ROCKETFORGE_REQUIRE_DROPSHOT_MESHES();
	std::unique_ptr<Arena> arena = Make(GameMode::DROPSHOT);
	arena->ball->SetState(MakeChargedDropshotBall());

	for (int tick = 0; tick < 60 && CountDamagedTiles(arena->GetDropshotTilesState()) == 0; tick++)
		arena->Step(1);

	// Charge level 2 damages the hit tile and its six neighbours.
	EXPECT_GT(CountDamagedTiles(arena->GetDropshotTilesState()), 1);
	EXPECT_TRUE(arena->ball->GetState().dsInfo.hasDamaged);
}

TEST_P(SimConformance, BallScoresThroughBrokenTiles) {
	ROCKETFORGE_REQUIRE_DROPSHOT_MESHES();
	std::unique_ptr<Arena> arena = Make(GameMode::DROPSHOT);
	arena->SetDropshotTilesState(AllTiles(DropshotTileState::STATE_BROKEN));
	arena->ball->SetState(MakeBall(Vec(0, 2500, 400), Vec(0, 0, -1000)));

	for (int tick = 0; tick < 240 && !arena->IsBallScored(); tick++)
		arena->Step(1);

	EXPECT_TRUE(arena->IsBallScored());
}

TEST_P(SimConformance, PredictionSeesBrokenTiles) {
	ROCKETFORGE_REQUIRE_DROPSHOT_MESHES();
	std::unique_ptr<Arena> arena = Make(GameMode::DROPSHOT);
	arena->ball->SetState(MakeBall(Vec(0, 2500, 400), Vec(0, 0, -1000)));
	const BallState overFullTiles = arena->GetFutureBallState(1.5f);

	arena->SetDropshotTilesState(AllTiles(DropshotTileState::STATE_BROKEN));
	const BallState overBrokenTiles = arena->GetFutureBallState(1.5f);

	EXPECT_GT(overFullTiles.pos.z, 0);
	EXPECT_LT(overBrokenTiles.pos.z, 0);
}

INSTANTIATE_TEST_SUITE_P(Engines, SimConformance, ::testing::Values(EngineId::V2, EngineId::V3),
	[](const auto& info) { return info.param == EngineId::V2 ? "V2" : "V3"; });
