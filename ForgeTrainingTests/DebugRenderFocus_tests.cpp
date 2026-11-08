#include <gtest/gtest.h>

#include <DebugRender/DebugRenderFocus.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <optional>

using namespace RocketForge;
using namespace RLGC;

namespace {
	GameState MakeState(const std::vector<std::pair<uint32_t, Team>>& cars) {
		GameState gameState;
		for (const auto& [carId, team] : cars) {
			Player player = {};
			player.index = (int)gameState.players.size();
			player.carId = carId;
			player.team = team;
			gameState.players.push_back(player);
		}

		return gameState;
	}
}

TEST(DebugRenderFocus, FindPlayerIndexByCarId) {
	GameState gameState = MakeState({ { 7, Team::BLUE }, { 9, Team::ORANGE } });

	ASSERT_EQ(FindPlayerIndexByCarId(gameState, 9), 1);
	ASSERT_EQ(FindPlayerIndexByCarId(gameState, 7), 0);
	ASSERT_EQ(FindPlayerIndexByCarId(gameState, 42), -1);
	ASSERT_EQ(FindPlayerIndexByCarId(gameState, -1), -1);
}

TEST(DebugRenderFocus, ChooseStaticTeamOppositeFocus) {
	GameState gameState = MakeState({ { 1, Team::BLUE }, { 2, Team::ORANGE } });

	ASSERT_TRUE(ChooseStaticTeam(gameState, 0) == Team::ORANGE);
	ASSERT_TRUE(ChooseStaticTeam(gameState, 1) == Team::BLUE);
	ASSERT_TRUE(ChooseStaticTeam(gameState, -1) == Team::ORANGE);
}

TEST(DebugRenderFocus, PickPrefersFocusCar) {
	GameState gameState = MakeState({ { 1, Team::BLUE }, { 2, Team::ORANGE } });

	ASSERT_EQ(PickEpisodeFocusIndex(gameState, 1, std::nullopt, 0), 1);
	ASSERT_EQ(PickEpisodeFocusIndex(gameState, 1, Team::BLUE, 0), 1);
}

TEST(DebugRenderFocus, PickFallsBackToNonStaticTeam) {
	GameState gameState = MakeState({ { 1, Team::ORANGE }, { 2, Team::ORANGE }, { 3, Team::BLUE } });

	ASSERT_EQ(PickEpisodeFocusIndex(gameState, -1, Team::ORANGE, 0), 2);
	ASSERT_EQ(PickEpisodeFocusIndex(gameState, -1, Team::BLUE, 2), 0);
}

TEST(DebugRenderFocus, PickKeepsCurrentWithoutFocusOrStatic) {
	GameState gameState = MakeState({ { 1, Team::BLUE }, { 2, Team::ORANGE } });

	ASSERT_EQ(PickEpisodeFocusIndex(gameState, -1, std::nullopt, 1), 1);
	ASSERT_EQ(PickEpisodeFocusIndex(gameState, -1, std::nullopt, 5), 1);
}

TEST(DebugRenderFocus, StepGameSpeed) {
	struct Case {
		const char* label;
		float speed;
		int direction;
		float expected;
	};

	const Case cases[] = {
		{ "preset steps up one notch", 1.f, 1, 2.f },
		{ "preset steps down one notch", 1.f, -1, 0.5f },
		{ "clamps at the fastest preset", 10.f, 1, 10.f },
		{ "clamps at the slowest preset", 0.1f, -1, 0.1f },
		{ "off-preset snaps up", 1.5f, 1, 2.f },
		{ "off-preset snaps down", 1.5f, -1, 1.f },
		{ "off-preset snaps down to the nearest lower preset", 0.3f, -1, 0.25f },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(StepGameSpeed(testCase.speed, testCase.direction), testCase.expected);
	}
}

TEST(DebugRenderFocus, ParseViewerCommands) {
	std::optional<ViewerCommand> focus = ParseViewerCommand(R"({"cmd":"focus_set","index":3})");
	ASSERT_TRUE(focus.has_value());
	ASSERT_TRUE(focus->type == ViewerCommand::Type::FocusSet);
	ASSERT_EQ(focus->value, 3);

	std::optional<ViewerCommand> speed = ParseViewerCommand(R"({"cmd":"speed_step","dir":-1})");
	ASSERT_TRUE(speed.has_value());
	ASSERT_TRUE(speed->type == ViewerCommand::Type::SpeedStep);
	ASSERT_EQ(speed->value, -1);

	ASSERT_TRUE(ParseViewerCommand(R"({"cmd":"speed_reset"})")->type == ViewerCommand::Type::SpeedReset);
	ASSERT_TRUE(ParseViewerCommand(R"({"cmd":"pause_toggle"})")->type == ViewerCommand::Type::PauseToggle);
}

TEST(DebugRenderFocus, ParseRejectsMalformed) {
	ASSERT_FALSE(ParseViewerCommand("not json").has_value());
	ASSERT_FALSE(ParseViewerCommand(R"({"cmd":"explode"})").has_value());
	ASSERT_FALSE(ParseViewerCommand(R"({"cmd":"focus_set"})").has_value());
	ASSERT_FALSE(ParseViewerCommand(R"({"cmd":"focus_set","index":"two"})").has_value());
	ASSERT_FALSE(ParseViewerCommand(R"([1,2])").has_value());
}
