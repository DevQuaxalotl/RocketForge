#pragma once
#include "Player.h"
#include "../CommonValues.h"
#include "../BasicTypes/Action.h"

namespace RLGC::BallPrediction {
	class Context;
}

namespace RLGC {
	struct ScoreLine {
		int teamGoals[2] = { 0,0 };

		int operator[](size_t index) const {
			return teamGoals[index];
		}

		int& operator[](size_t index) {
			return teamGoals[index];
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/game_state.py
	struct GameState {
		
		GameState* prev = NULL;

		RocketSim::GameMode gameMode = RocketSim::GameMode::SOCCAR;

		float deltaTime = 0; // Time that has passed since last update

		/** Seconds elapsed since the episode this state belongs to began **/
		float episodeTime = 0;

		bool goalScored = false; // If the ball is in the goal
		int lastTouchCarID = -1;
		std::vector<Player> players;

		BallState ball;

		std::vector<int> boostPadIndexMap;
		std::vector<bool> boostPads, boostPadsInv;
		std::vector<float> boostPadTimers, boostPadTimersInv;

		// Ball prediction data set by RLBot
		std::vector<BallState> ballPredData;

		// Last arena we updated with
		// Can be used to determine current arena from within reward function, for example
		// NOTE: Could be null
		Arena* lastArena = NULL;

		// Last tick count when updated
		uint64_t lastTickCount = 0;

		/**
		 * The predictor of the model whose observation is being built, set only for the duration of that build.
		 * Mutable because builders receive the state as const; one state's observations are built on one thread.
		 */
		mutable BallPrediction::Context* predictionContext = nullptr;

		void* userInfo = NULL;

		GameState() {
			boostPadIndexMap = std::vector<int>(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
			boostPads = std::vector<bool>(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
			boostPadsInv = std::vector<bool>(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
			boostPadTimers = std::vector<float>(CommonValues::BOOST_LOCATIONS_AMOUNT, 0);
			boostPadTimersInv = std::vector<float>(CommonValues::BOOST_LOCATIONS_AMOUNT, 0);
		}
		explicit GameState(Arena* arena) {
			UpdateFromArena(arena, std::vector<Action>(arena->_cars.size()), NULL);

			// A state built straight from an arena starts an episode;
			// the update above had no prior tick count, so its deltaTime is not elapsed episode time
			episodeTime = 0;
		}

		const auto& GetBoostPads(bool inverted) const {
			return inverted ? boostPadsInv : boostPads;
		}

		const auto& GetBoostPadTimers(bool inverted) const {
			return inverted ? boostPadTimersInv : boostPadTimers;
		}

		// Called before updating to reset the per-step state
		void ResetBeforeStep();

		void UpdateFromArena(Arena* arena, const std::vector<Action>& actions, GameState* prev);

		bool IsEmpty() const {
			return players.empty();
		}

		void MakeEmpty() {
			players.clear();
		}

		// Get ball state at given amount of seconds into the future
		BallState GetFutureBallState(const float time) const;

		/** Whether lastArena's own ball prediction is the same predictor as `engine` would be for this exact state. **/
		bool CanUseArenaBallPrediction(RocketSim::EngineId engine) const;

		// Helper functions to replace CommonValues for multi mode support
		int GetBoostLocationAmount() const;
		const Vec* GetBoostLocations() const;
		const float GetXWall() const;
		const float GetYWall() const;
		const float GetCeiling() const;
		const Vec GetGoalPos(const RocketSim::Team team) const;

	protected:
		bool boostPadIndexMapBuilt = false;

		void BuildBoostPadIndexMap();
	};
}