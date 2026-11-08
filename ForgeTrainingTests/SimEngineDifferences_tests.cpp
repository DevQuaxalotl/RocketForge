#include <gtest/gtest.h>

#include "TestSupport.h"

#include <RocketSim/Includes.h>

#include <cmath>
#include <memory>

using namespace RocketSim;

// Each test pins down a documented difference between the engines, so an upstream change to one of them shows up here.

namespace {
	std::unique_ptr<Arena> MakeArena(EngineId engine, GameMode mode = GameMode::SOCCAR) {
		ArenaConfig config;
		config.engine = engine;
		return std::unique_ptr<Arena>(Arena::Create(mode, config));
	}

	Car* AddGroundedCar(Arena& arena, const Vec& pos, const Vec& forward, float speed) {
		Car* car = arena.AddCar(Team::BLUE);

		CarState state;
		state.pos = pos;
		state.rotMat = RotMat::LookAt(forward, Vec(0, 0, 1));
		state.vel = forward * speed;
		car->SetState(state);
		return car;
	}

	/** Ticks from a Hoops kickoff until the ball launches, optionally setting the ball to its own state `setAfterTicks` in. **/
	int HoopsLaunchTick(EngineId engine, int setAfterTicks) {
		std::unique_ptr<Arena> arena = MakeArena(engine, GameMode::HOOPS);
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
		arena->ResetToRandomKickoff(3);

		const int maxTicks = 120;
		for (int tick = 0; tick < maxTicks; tick++) {
			if (tick == setAfterTicks)
				arena->ball->SetState(arena->ball->GetState());

			if (arena->ball->GetState().vel.z > 0)
				return tick;

			arena->Step(1);
		}

		return -1;
	}
}

class SimEngineDifferences : public ::testing::Test {
protected:
	void SetUp() override {
		ROCKETFORGE_REQUIRE_ROCKETSIM();
		ROCKETFORGE_REQUIRE_ENGINE(EngineId::V2);
		ROCKETFORGE_REQUIRE_ENGINE(EngineId::V3);
	}
};

TEST_F(SimEngineDifferences, ObsOnGroundDiffersWhileJumping) {
	for (EngineId engine : { EngineId::V2, EngineId::V3 }) {
		SCOPED_TRACE(static_cast<int>(engine));
		std::unique_ptr<Arena> arena = MakeArena(engine);
		Car* car = AddGroundedCar(*arena, Vec(0, -2000, 17), Vec(1, 0, 0), 0);
		arena->Step(30);

		car->controls.jump = true;
		arena->Step(1);

		// v2 reports the RLBot-style flag, which drops as soon as a jump starts; v3 reports isOnGround.
		const CarState state = car->GetState();
		ASSERT_TRUE(state.isJumping);
		EXPECT_EQ(state.obsOnGround, engine == EngineId::V3);
	}
}

TEST_F(SimEngineDifferences, SupersonicTimeStaysZeroOnV3) {
	for (EngineId engine : { EngineId::V2, EngineId::V3 }) {
		SCOPED_TRACE(static_cast<int>(engine));
		std::unique_ptr<Arena> arena = MakeArena(engine);
		Car* car = AddGroundedCar(*arena, Vec(-3500, -3000, 17), Vec(1, 0, 0), 2300);
		car->controls.throttle = 1;
		car->controls.boost = true;
		arena->Step(240);

		const CarState state = car->GetState();
		ASSERT_TRUE(state.isSupersonic);
		if (engine == EngineId::V2)
			EXPECT_GT(state.supersonicTime, 0);
		else
			EXPECT_EQ(state.supersonicTime, 0);
	}
}

TEST_F(SimEngineDifferences, ExtraImpulseTickUnsetOnV3) {
	for (EngineId engine : { EngineId::V2, EngineId::V3 }) {
		SCOPED_TRACE(static_cast<int>(engine));
		std::unique_ptr<Arena> arena = MakeArena(engine);
		Car* car = AddGroundedCar(*arena, Vec(0, -400, 17), Vec(0, 1, 0), 1500);
		car->controls.throttle = 1;

		BallState ball;
		ball.pos = Vec(0, 0, 93);
		arena->ball->SetState(ball);
		arena->tickCount = 500;

		for (int tick = 0; tick < 60 && !car->GetState().ballHitInfo.isValid; tick++)
			arena->Step(1);

		const BallHitInfo hit = car->GetState().ballHitInfo;
		ASSERT_TRUE(hit.isValid);
		if (engine == EngineId::V2)
			EXPECT_GT(hit.tickCountWhenExtraImpulseApplied, 0u);
		else
			EXPECT_EQ(hit.tickCountWhenExtraImpulseApplied, 0u);
	}
}

TEST_F(SimEngineDifferences, HoopsBallSetRestartsLaunchDelayOnlyOnV2) {
	const int setAfterTicks = 10;
	const int noSet = -1;

	// v2 counts the launch delay with the ball's tickCountSinceUpdate, which every ball set resets.
	const int v2Untouched = HoopsLaunchTick(EngineId::V2, noSet);
	ASSERT_GT(v2Untouched, setAfterTicks);
	EXPECT_EQ(HoopsLaunchTick(EngineId::V2, setAfterTicks), v2Untouched + setAfterTicks);

	const int v3Untouched = HoopsLaunchTick(EngineId::V3, noSet);
	ASSERT_GT(v3Untouched, setAfterTicks);
	EXPECT_EQ(HoopsLaunchTick(EngineId::V3, setAfterTicks), v3Untouched);
}

TEST_F(SimEngineDifferences, OverlappingCarSpawnTurnsNonFiniteOnlyOnV3) {
	for (EngineId engine : { EngineId::V2, EngineId::V3 }) {
		SCOPED_TRACE(static_cast<int>(engine));
		std::unique_ptr<Arena> arena = MakeArena(engine);

		// Two stationary cars about 29 uu apart, well inside each other's hitboxes.
		// v3 only fails for some overlaps (closer ones stay finite), so these are exact values from a failed training arena.
		const Vec forward(1.19249e-08f, -1, 0);
		Car* first = AddGroundedCar(*arena, Vec(177.046f, 4212.06f, 17), forward, 0);
		Car* second = AddGroundedCar(*arena, Vec(159.443f, 4189.31f, 17), forward, 0);
		arena->Step(8);

		for (Car* car : { first, second }) {
			const CarState state = car->GetState();
			const bool finite = std::isfinite(state.pos.x) && std::isfinite(state.vel.x) && std::isfinite(state.rotMat.forward.x);
			EXPECT_EQ(finite, engine == EngineId::V2);
		}
	}
}

TEST_F(SimEngineDifferences, SameTickBallSetRefreshesPredictionOnlyOnV3) {
	for (EngineId engine : { EngineId::V2, EngineId::V3 }) {
		SCOPED_TRACE(static_cast<int>(engine));
		std::unique_ptr<Arena> arena = MakeArena(engine);
		BallState state;
		state.pos = Vec(0, 0, 500);
		arena->ball->SetState(state);
		const BallState before = arena->GetFutureBallState(0.25f);

		state.pos.x = 1200;
		arena->ball->SetState(state);
		const BallState sameTick = arena->GetFutureBallState(0.25f);

		// v2 keeps the prediction from before the set until the arena steps.
		const float expectedShift = engine == EngineId::V3 ? 1200.f : 0.f;
		EXPECT_NEAR(sameTick.pos.x - before.pos.x, expectedShift, 0.01f);

		arena->Step(1);
		const BallState nextTick = arena->GetFutureBallState(0.25f);
		EXPECT_NEAR(nextTick.pos.x, 1200, 0.01f);
	}
}
