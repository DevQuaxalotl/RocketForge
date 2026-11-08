#include "DebugRenderFocus.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>

using namespace RLGC;

namespace
{
	constexpr std::array<float, 7> GAME_SPEED_PRESETS = { 0.1f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 10.f };

	// Speeds within this of a preset count as that preset, so float round-off never skips or repeats a step.
	constexpr float SPEED_EPSILON = 1e-3f;
}

int RocketForge::FindPlayerIndexByCarId(const GameState& gameState, int carId)
{
	if (carId < 0)
		return -1;

	for (int i = 0; i < (int)gameState.players.size(); i++)
		if (gameState.players[i].carId == (uint32_t)carId)
			return i;

	return -1;
}

Team RocketForge::ChooseStaticTeam(const GameState& gameState, int focusPlayerIndex)
{
	if (focusPlayerIndex < 0 || focusPlayerIndex >= (int)gameState.players.size())
		return Team::ORANGE;

	return gameState.players[focusPlayerIndex].team == Team::BLUE ? Team::ORANGE : Team::BLUE;
}

int RocketForge::PickEpisodeFocusIndex(
	const GameState& gameState,
	int focusPlayerIndex,
	std::optional<Team> staticTeam,
	int currentIndex)
{
	if (gameState.players.empty())
		return 0;

	if (focusPlayerIndex >= 0 && focusPlayerIndex < (int)gameState.players.size())
		return focusPlayerIndex;

	if (staticTeam.has_value()) {
		for (int i = 0; i < (int)gameState.players.size(); i++)
			if (gameState.players[i].team != *staticTeam)
				return i;
	}

	return std::clamp(currentIndex, 0, (int)gameState.players.size() - 1);
}

float RocketForge::StepGameSpeed(float currentSpeed, int direction)
{
	if (direction > 0) {
		for (float preset : GAME_SPEED_PRESETS)
			if (preset > currentSpeed + SPEED_EPSILON)
				return preset;

		return GAME_SPEED_PRESETS.back();
	}

	for (auto preset = GAME_SPEED_PRESETS.rbegin(); preset != GAME_SPEED_PRESETS.rend(); ++preset)
		if (*preset < currentSpeed - SPEED_EPSILON)
			return *preset;

	return GAME_SPEED_PRESETS.front();
}

std::optional<RocketForge::ViewerCommand> RocketForge::ParseViewerCommand(const std::string& datagram)
{
	const nlohmann::json json = nlohmann::json::parse(datagram, nullptr, false);
	if (!json.is_object() || !json.contains("cmd") || !json["cmd"].is_string())
		return std::nullopt;

	const std::string name = json["cmd"].get<std::string>();

	if (name == "focus_set") {
		if (!json.contains("index") || !json["index"].is_number_integer())
			return std::nullopt;

		return ViewerCommand{ ViewerCommand::Type::FocusSet, json["index"].get<int>() };
	}

	if (name == "speed_step") {
		if (!json.contains("dir") || !json["dir"].is_number_integer())
			return std::nullopt;

		return ViewerCommand{ ViewerCommand::Type::SpeedStep, json["dir"].get<int>() > 0 ? 1 : -1 };
	}

	if (name == "speed_reset")
		return ViewerCommand{ ViewerCommand::Type::SpeedReset };

	if (name == "pause_toggle")
		return ViewerCommand{ ViewerCommand::Type::PauseToggle };

	return std::nullopt;
}
