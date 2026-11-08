#include "SkillFixedSeries.h"

std::vector<GGL::FixedSeriesGame> GGL::BuildFixedSeriesGames(int opponentCount, int gamesPerOpponent, uint64_t baseSeed, int teamSize) {
	if (opponentCount <= 0 || gamesPerOpponent <= 0)
		return {};

	constexpr uint64_t TEAM_SIZE_SEED_STRIDE = 1'000'003;
	const int pairsPerOpponent = (gamesPerOpponent + 1) / 2;
	const uint64_t modeBaseSeed = baseSeed + (uint64_t)teamSize * TEAM_SIZE_SEED_STRIDE;

	std::vector<FixedSeriesGame> games;
	games.reserve((size_t)opponentCount * pairsPerOpponent * 2);

	// Pair-major order keeps every opponent's series progressing together while slots run in parallel.
	for (int pairIdx = 0; pairIdx < pairsPerOpponent; pairIdx++) {
		const uint64_t kickoffSeed = modeBaseSeed + (uint64_t)pairIdx;
		for (int opponentIdx = 0; opponentIdx < opponentCount; opponentIdx++) {
			games.push_back({ opponentIdx, RocketSim::Team::BLUE, kickoffSeed });
			games.push_back({ opponentIdx, RocketSim::Team::ORANGE, kickoffSeed });
		}
	}

	return games;
}

void GGL::TallyFixedSeriesGame(FixedSeriesRecord& record, RocketSim::Team subjectTeam, bool goalScored, RocketSim::Team scoringTeam) {
	if (!goalScored)
		record.noScoreGames++;
	else if (scoringTeam == subjectTeam)
		record.wins++;
	else
		record.losses++;
}
