#pragma once

#include <RocketSim/Includes.h>

#include <cstdint>
#include <vector>

namespace GGL {

	/**
	 * One scheduled game of a fixed series: which opponent, which side the subject takes, and its kickoff seed.
	 * The subject is the current policy unless another subject is given.
	 */
	struct FixedSeriesGame {
		int opponentIdx = 0;
		RocketSim::Team currentTeam = RocketSim::Team::BLUE;
		uint64_t kickoffSeed = 0;
	};

	/** Scored and unscored results of the subject (the current policy unless another subject is given) against one fixed opponent. */
	struct FixedSeriesRecord {
		int wins = 0;
		int losses = 0;
		int noScoreGames = 0;
	};

	/**
	 * Schedules gamesPerOpponent games (rounded up to even) against each opponent, as side-swapped pairs sharing one kickoff seed.
	 * Pair k uses seed baseSeed + a fixed per-mode offset + k, so every opponent and every test of a mode replays the same kickoffs.
	 * Returns an empty schedule when gamesPerOpponent <= 0.
	 */
	std::vector<FixedSeriesGame> BuildFixedSeriesGames(int opponentCount, int gamesPerOpponent, uint64_t baseSeed, int teamSize);

	/** Adds one finished game to the subject's record; scoringTeam is ignored when no goal was scored. */
	void TallyFixedSeriesGame(FixedSeriesRecord& record, RocketSim::Team subjectTeam, bool goalScored, RocketSim::Team scoringTeam);
}
