#include <gtest/gtest.h>

#include "TestSupport.h"

#include "../ForgeTraining/RLGym/RocketSim/src/RocketSim.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace {
	struct BumpEvent {
		uint32_t attackerId;
		uint32_t victimId;
		bool isDemo;
	};

	void CaptureBumpEvent(RocketSimV2::Arena*, RocketSimV2::Car* attacker, RocketSimV2::Car* victim, bool isDemo, void* userInfo) {
		auto events = static_cast<std::vector<BumpEvent>*>(userInfo);
		events->push_back({ attacker->id, victim->id, isDemo });
	}

	RocketSimV2::CarState MakeCarState(const RocketSimV2::Vec& pos, const RocketSimV2::Vec& forward, const RocketSimV2::Vec& vel, bool isSupersonic) {
		RocketSimV2::CarState state;
		state.pos = pos;
		state.rotMat = RocketSimV2::RotMat::LookAt(forward, RocketSimV2::Vec(0, 0, 1));
		state.vel = vel;
		state.isOnGround = true;
		state.obsOnGround = true;
		state.isSupersonic = isSupersonic;
		return state;
	}

	void RunCarCollision(RocketSimV2::Arena* arena, RocketSimV2::Car* car1, RocketSimV2::Car* car2) {
		btManifoldPoint manifold;
		arena->_BtCallback_OnCarCarCollision(car1, car2, manifold);
	}
}

TEST(ArenaDemo, BackwardsSupersonicCanDemolish) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	std::unique_ptr<RocketSimV2::Arena> arena(RocketSimV2::Arena::Create(RocketSimV2::GameMode::SOCCAR));
	RocketSimV2::Car* attacker = arena->AddCar(RocketSimV2::Team::BLUE);
	RocketSimV2::Car* victim = arena->AddCar(RocketSimV2::Team::ORANGE);

	attacker->SetState(MakeCarState(RocketSimV2::Vec(0, 0, 17), RocketSimV2::Vec(1, 0, 0), RocketSimV2::Vec(-2200, 0, 0), true));
	victim->SetState(MakeCarState(RocketSimV2::Vec(-150, 0, 17), RocketSimV2::Vec(1, 0, 0), RocketSimV2::Vec(0, 0, 0), false));

	std::vector<BumpEvent> events;
	arena->SetCarBumpCallback(CaptureBumpEvent, &events);

	RunCarCollision(arena.get(), attacker, victim);

	ASSERT_EQ(events.size(), size_t(1));
	ASSERT_EQ(events[0].attackerId, attacker->id);
	ASSERT_EQ(events[0].victimId, victim->id);
	ASSERT_TRUE(events[0].isDemo);
	ASSERT_TRUE(victim->GetState().isDemoed);
}

TEST(ArenaDemo, SupersonicSidewaysDriftDemotesToBump) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	std::unique_ptr<RocketSimV2::Arena> arena(RocketSimV2::Arena::Create(RocketSimV2::GameMode::SOCCAR));
	RocketSimV2::Car* attacker = arena->AddCar(RocketSimV2::Team::BLUE);
	RocketSimV2::Car* victim = arena->AddCar(RocketSimV2::Team::ORANGE);

	attacker->SetState(MakeCarState(RocketSimV2::Vec(0, 0, 17), RocketSimV2::Vec(1, 0, 0), RocketSimV2::Vec(1600, 1600, 0), true));
	victim->SetState(MakeCarState(RocketSimV2::Vec(150, 0, 17), RocketSimV2::Vec(-1, 0, 0), RocketSimV2::Vec(0, 0, 0), false));

	std::vector<BumpEvent> events;
	arena->SetCarBumpCallback(CaptureBumpEvent, &events);

	RunCarCollision(arena.get(), attacker, victim);

	ASSERT_EQ(events.size(), size_t(1));
	ASSERT_EQ(events[0].attackerId, attacker->id);
	ASSERT_EQ(events[0].victimId, victim->id);
	ASSERT_FALSE(events[0].isDemo);
	ASSERT_FALSE(victim->GetState().isDemoed);
}

TEST(ArenaDemo, HeadOnSupersonicCollisionAllowsMutualDemos) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	std::unique_ptr<RocketSimV2::Arena> arena(RocketSimV2::Arena::Create(RocketSimV2::GameMode::SOCCAR));
	RocketSimV2::Car* blue = arena->AddCar(RocketSimV2::Team::BLUE);
	RocketSimV2::Car* orange = arena->AddCar(RocketSimV2::Team::ORANGE);

	blue->SetState(MakeCarState(RocketSimV2::Vec(-100, 0, 17), RocketSimV2::Vec(1, 0, 0), RocketSimV2::Vec(2200, 0, 0), true));
	orange->SetState(MakeCarState(RocketSimV2::Vec(100, 0, 17), RocketSimV2::Vec(-1, 0, 0), RocketSimV2::Vec(-2200, 0, 0), true));

	std::vector<BumpEvent> events;
	arena->SetCarBumpCallback(CaptureBumpEvent, &events);

	RunCarCollision(arena.get(), blue, orange);

	ASSERT_EQ(events.size(), size_t(2));
	ASSERT_TRUE(events[0].isDemo);
	ASSERT_TRUE(events[1].isDemo);
	ASSERT_TRUE(blue->GetState().isDemoed);
	ASSERT_TRUE(orange->GetState().isDemoed);
}

TEST(ArenaDemo, OneWayDemoSuppressesVictimBumpAction) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	std::unique_ptr<RocketSimV2::Arena> arena(RocketSimV2::Arena::Create(RocketSimV2::GameMode::SOCCAR));
	RocketSimV2::Car* attacker = arena->AddCar(RocketSimV2::Team::BLUE);
	RocketSimV2::Car* victim = arena->AddCar(RocketSimV2::Team::ORANGE);

	attacker->SetState(MakeCarState(RocketSimV2::Vec(0, 0, 17), RocketSimV2::Vec(1, 0, 0), RocketSimV2::Vec(2200, 0, 0), true));
	victim->SetState(MakeCarState(RocketSimV2::Vec(150, 0, 17), RocketSimV2::Vec(-1, 0, 0), RocketSimV2::Vec(-1500, 0, 0), false));

	std::vector<BumpEvent> events;
	arena->SetCarBumpCallback(CaptureBumpEvent, &events);

	RunCarCollision(arena.get(), attacker, victim);

	ASSERT_EQ(events.size(), size_t(1));
	ASSERT_EQ(events[0].attackerId, attacker->id);
	ASSERT_EQ(events[0].victimId, victim->id);
	ASSERT_TRUE(events[0].isDemo);
	ASSERT_TRUE(victim->GetState().isDemoed);
}
