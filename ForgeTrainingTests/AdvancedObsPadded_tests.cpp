#include <gtest/gtest.h>
#include "TestSupport.h"

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RocketSim/Arena.h>

#include <memory>

namespace {
	/** Arena with the given team sizes; cars are spread out so their encodings differ. */
	std::unique_ptr<RocketSim::Arena> MakeArena(int blueSize, int orangeSize) {
		std::unique_ptr<RocketSim::Arena> arena(RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR));
		for (int i = 0; i < blueSize; i++)
			arena->AddCar(RocketSim::Team::BLUE);
		for (int i = 0; i < orangeSize; i++)
			arena->AddCar(RocketSim::Team::ORANGE);
		arena->ResetToRandomKickoff(7);
		return arena;
	}

	size_t PaddedObsSize(int maxPlayers, int blueSize, int orangeSize) {
		std::unique_ptr<RocketSim::Arena> arena = MakeArena(blueSize, orangeSize);
		RLGC::GameState state(arena.get());
		RLGC::AdvancedObsPadded obs(maxPlayers);
		return obs.BuildObs(state.players[0], state).size();
	}
}

TEST(AdvancedObsPadded, OneVOneMatchesAdvancedObs) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	std::unique_ptr<RocketSim::Arena> arena = MakeArena(1, 1);
	RLGC::GameState state(arena.get());

	RLGC::AdvancedObs advanced;
	RLGC::AdvancedObsPadded padded(1);
	for (const RLGC::Player& player : state.players) {
		const RLGC::FList expected = advanced.BuildObs(player, state);
		const RLGC::FList actual = padded.BuildObs(player, state);
		ASSERT_EQ(actual.size(), expected.size());
		for (size_t i = 0; i < expected.size(); i++)
			EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
	}
}

TEST(AdvancedObsPadded, SizeIsConstantAcrossTeamSizes) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const int maxPlayers = 3;
	const size_t oneVOne = PaddedObsSize(maxPlayers, 1, 1);
	EXPECT_EQ(PaddedObsSize(maxPlayers, 2, 2), oneVOne);
	EXPECT_EQ(PaddedObsSize(maxPlayers, 3, 3), oneVOne);
	EXPECT_EQ(PaddedObsSize(maxPlayers, 1, 2), oneVOne);
}

TEST(AdvancedObsPadded, TooManyPlayersFails) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	std::unique_ptr<RocketSim::Arena> arena = MakeArena(2, 2);
	RLGC::GameState state(arena.get());
	RLGC::AdvancedObsPadded obs(1);
	EXPECT_ANY_THROW(obs.BuildObs(state.players[0], state));
}

TEST(AdvancedObsPadded, NonPositiveMaxPlayersFails) {
	EXPECT_ANY_THROW(RLGC::AdvancedObsPadded(0));
}

TEST(AdvancedObsPadded, BuildObsIntoMatchesBuildObs) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	std::unique_ptr<RocketSim::Arena> arena = MakeArena(2, 2);
	RLGC::GameState state(arena.get());
	RLGC::AdvancedObsPadded obs(3);

	const RLGC::FList expected = obs.BuildObs(state.players[1], state);
	RLGC::FList actual = { 99.f };
	obs.BuildObsInto(state.players[1], state, actual);
	ASSERT_EQ(actual, expected);
}
