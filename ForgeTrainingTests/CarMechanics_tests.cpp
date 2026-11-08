#include <gtest/gtest.h>

#include "TestSupport.h"

#include "../ForgeTraining/RLGym/RocketSim/src/RocketSim.h"

#include <memory>

namespace {
	RocketSimV2::CarState MakeAirborneDodgeState(bool hasJumped, float airTimeSinceJump) {
		RocketSimV2::CarState state;
		state.pos = RocketSimV2::Vec(0.f, 0.f, 1000.f);
		state.rotMat = RocketSimV2::RotMat::LookAt(
			RocketSimV2::Vec(1.f, 0.f, 0.f),
			RocketSimV2::Vec(0.f, 0.f, 1.f));
		state.isOnGround = false;
		state.obsOnGround = false;
		state.hasJumped = hasJumped;
		state.hasDoubleJumped = false;
		state.hasFlipped = false;
		state.isJumping = false;
		state.isFlipping = false;
		state.airTime = 2.f;
		state.airTimeSinceJump = airTimeSinceJump;
		state.jumpResetBlockTicks = 0;
		state.lastControls = {};
		return state;
	}

	RocketSimV2::CarState AttemptBackwardDodge(const RocketSimV2::CarState& initialState) {
		std::unique_ptr<RocketSimV2::Arena> arena(
			RocketSimV2::Arena::Create(RocketSimV2::GameMode::SOCCAR));
		RocketSimV2::Car* car = arena->AddCar(RocketSimV2::Team::BLUE);
		car->SetState(initialState);
		car->controls.pitch = 1.f;
		car->controls.jump = true;
		arena->Step(1);
		return car->GetState();
	}
}

TEST(CarMechanics, HeldFlipResetDoesNotExpire) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	const RocketSimV2::CarState result = AttemptBackwardDodge(
		MakeAirborneDodgeState(false, 0.f));

	ASSERT_TRUE(result.hasFlipped);
	ASSERT_TRUE(result.isFlipping);
}

TEST(CarMechanics, NormalAirDodgeStillExpires) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	const RocketSimV2::CarState result = AttemptBackwardDodge(
		MakeAirborneDodgeState(true, 1.5f));

	ASSERT_FALSE(result.hasFlipped);
	ASSERT_FALSE(result.isFlipping);
}
