#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <optional>
#include <string>

namespace RocketForge
{
	inline constexpr float DEFAULT_GAME_SPEED = 1.f;

	/** A control request sent from the render viewer back to the render loop. */
	struct ViewerCommand
	{
		enum class Type { FocusSet, SpeedStep, SpeedReset, PauseToggle };

		Type type;

		/** Player index for FocusSet, +1/-1 for SpeedStep, unused otherwise. */
		int value = 0;
	};

	/** Index of the player driving carId, or -1 when carId is negative or absent. */
	int FindPlayerIndexByCarId(const RLGC::GameState& gameState, int carId);

	/** The team opposite the focus player, or ORANGE when there is no focus player. */
	RocketSim::Team ChooseStaticTeam(const RLGC::GameState& gameState, int focusPlayerIndex);

	/**
	 * Player to view at episode start: the focus player when there is one,
	 * else the first player off the static team, else currentIndex clamped to the player count.
	 */
	int PickEpisodeFocusIndex(
		const RLGC::GameState& gameState,
		int focusPlayerIndex,
		std::optional<RocketSim::Team> staticTeam,
		int currentIndex);

	/** Next speed preset strictly above (direction > 0) or below the current speed, clamped at the ends. */
	float StepGameSpeed(float currentSpeed, int direction);

	/** Parses one viewer datagram. Returns nullopt for malformed or unknown commands. */
	std::optional<ViewerCommand> ParseViewerCommand(const std::string& datagram);
}
