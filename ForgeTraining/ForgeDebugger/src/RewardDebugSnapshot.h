#pragma once
#include "RewardDebugData.h"
#include "NormalizationMode.h"
#include "RewardDisplayMode.h"
#include "CriticDebugData.h"
#include <vector>
#include <cstdint>
#include <optional>
#include <string>

namespace ForgeDebug {

	/** One entry per reward in the stage — used by the GUI to list ALL rewards
	 *  (including those not currently debug-enabled) so each can be toggled.
	 *  Independent of snapshot.rewards, which only holds effective-enabled rewards. */
	struct RewardRosterEntry {
		std::string name;
		RLGC::RewardCategory category;
		int index;             // collector reward index, stable for the stage
		bool desiredEnabled;   // the user's per-reward selection (independent of forceDebugAll)
	};

	struct RewardDebugSnapshot {
		/** Every reward in the stage, regardless of enabled state. */
		std::vector<RewardRosterEntry> roster;

		/** Only contains debug-enabled rewards */
		std::vector<RewardDebugData> rewards;

		/** Sum of weighted continuous reward values this frame */
		float totalContinuousWeighted;

		/** Sum of weighted event reward values this frame */
		float totalEventWeighted;

		/** Sum of all weighted reward values this frame */
		float totalWeighted;

		NormalizationMode normMode;
		RewardDisplayMode rewardDisplayMode;

		/** Every reward is debugged regardless of its roster selection. */
		bool forceDebugAll = false;

		uint64_t frameNumber;

		/** Index into GameState::players for the player being debugged */
		int playerIndex;
		int playerCarId;
		int playerTeam;

		/** Game time of the recorded steps since Initialize() or Reset(), in seconds. */
		double gameTimeSeconds = 0.0;

		/** Increments at every episode start. */
		uint64_t episodeIndex = 0;

		/** Critic value prediction for the debugged player; empty when no critic is available. */
		std::optional<CriticDebugData> critic;
	};

}
