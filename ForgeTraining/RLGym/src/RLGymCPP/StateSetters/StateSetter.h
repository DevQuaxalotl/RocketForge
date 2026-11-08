#pragma once
#include "../Gamestates/GameState.h"

namespace RLGC {
	/** The kind of situation a state setter create */
	enum class StateSetterType { Kickoff, Random, Scenario };

	/** Returns whether an episode represents normal gameplay rather than a bespoke scenario. **/
	inline bool IsNormalGameplayEpisode(StateSetterType type) {
		return type == StateSetterType::Kickoff || type == StateSetterType::Random;
	}

	class StateSetter {
	public:
		virtual ~StateSetter() = default;

		virtual void ResetArena(Arena* arena) = 0;

		/**
		 * Optional car ID whose trajectory should be collected for this scenario.
		 * A negative value keeps the normal behavior of collecting every learner-controlled car.
		 **/
		int trainingFocusCarId = -1;

		/**
		 * Optional cap, in seconds, on how long an episode started by this setter may run.
		 * Zero or less leaves the episode to the terminal conditions alone.
		 **/
		float maxEpisodeSeconds = -1.f;

		bool EpisodeCapElapsed(const GameState& state) const {
			return maxEpisodeSeconds > 0.f && state.episodeTime >= maxEpisodeSeconds;
		}

		StateSetterType setterType = StateSetterType::Scenario;
		StateSetterType GetSetterType() const { return setterType; }
	};
}