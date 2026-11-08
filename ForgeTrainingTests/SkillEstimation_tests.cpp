#include <gtest/gtest.h>

#include <GigaLearnCPP/SkillEstimation.h>
#include <GigaLearnCPP/SkillFixedSeries.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

TEST(SkillEstimation, EstimateRatingGapIsZeroForEvenRecord)
{
	GGL::SkillTrackerConfig config = {};
	GGL::SkillEstimate gap = GGL::EstimateRatingGap(32, 32, config);
	ASSERT_NEAR(gap.mu, 0.0f, 5.0f);
	ASSERT_LT(gap.sigma, config.initialSigma);
}

TEST(SkillEstimation, EstimateRatingGapIsAntisymmetric)
{
	GGL::SkillTrackerConfig config = {};
	ASSERT_NEAR(GGL::EstimateRatingGap(40, 24, config).mu, -GGL::EstimateRatingGap(24, 40, config).mu, 1.0f);
	ASSERT_GT(GGL::EstimateRatingGap(40, 24, config).mu, 0.0f);
}

TEST(SkillEstimation, EstimateRatingGapNarrowsWithMoreGames)
{
	GGL::SkillTrackerConfig config = {};
	ASSERT_LT(GGL::EstimateRatingGap(64, 64, config).sigma, GGL::EstimateRatingGap(16, 16, config).sigma);
}

TEST(SkillEstimation, EstimateRatingGapStaysFiniteForSweep)
{
	GGL::SkillTrackerConfig config = {};
	GGL::SkillEstimate gap = GGL::EstimateRatingGap(64, 0, config);
	ASSERT_TRUE(std::isfinite(gap.mu));
	ASSERT_GT(gap.mu, 0.0f);
}

TEST(SkillEstimation, EstimateRatingGapReturnsPriorWithNoGames)
{
	GGL::SkillTrackerConfig config = {};
	GGL::SkillEstimate gap = GGL::EstimateRatingGap(0, 0, config);
	ASSERT_EQ(gap.mu, 0.0f);
	ASSERT_EQ(gap.sigma, config.initialSigma);
}

TEST(SkillEstimation, IsWinProbabilityLopsidedIsOneSided)
{
	ASSERT_TRUE(GGL::IsWinProbabilityLopsided(0.95f, 0.90f));
	ASSERT_TRUE(GGL::IsWinProbabilityLopsided(0.05f, 0.90f));
	ASSERT_FALSE(GGL::IsWinProbabilityLopsided(0.70f, 0.90f));
}

TEST(SkillEstimation, CombineCalibrationSamplesIgnoresLopsidedSamples)
{
	GGL::SkillTrackerConfig config = {};
	const float evenGapSigma = GGL::EstimateRatingGap(32, 32, config).sigma;
	const std::vector<GGL::CalibrationSample> samples = {
		{ 1000.0f, GGL::SkillEstimate{ 20.0f, evenGapSigma } },
		{ 1500.0f, GGL::EstimateRatingGap(64, 0, config) },
	};

	const std::optional<float> rating = GGL::CombineCalibrationSamples(samples, config.anchorRetireExpectedWinProb, config);

	ASSERT_TRUE(rating.has_value());
	ASSERT_NEAR(*rating, 1020.0f, 1e-3f);
}

TEST(SkillEstimation, CombineCalibrationSamplesIsEmptyWhenAllLopsided)
{
	GGL::SkillTrackerConfig config = {};
	const std::vector<GGL::CalibrationSample> samples = {
		{ 1000.0f, GGL::EstimateRatingGap(64, 0, config) },
		{ 1200.0f, GGL::EstimateRatingGap(0, 64, config) },
	};

	ASSERT_FALSE(GGL::CombineCalibrationSamples(samples, config.anchorRetireExpectedWinProb, config).has_value());
}

TEST(SkillEstimation, EstimateFixedOpponentMMRIsAboveOpponentForWinningRecord)
{
	GGL::SkillTrackerConfig config = {};
	const GGL::FixedSeriesRecord currentPolicyRecord = { 40, 24, 0 };

	const std::optional<GGL::SkillEstimate> mmr = GGL::EstimateFixedOpponentMMR(1000.0f, currentPolicyRecord, config);

	ASSERT_TRUE(mmr.has_value());
	ASSERT_GT(mmr->mu, 1000.0f);
	ASSERT_FALSE(GGL::EstimateFixedOpponentMMR(1000.0f, GGL::FixedSeriesRecord{ 0, 0, 8 }, config).has_value());
}

TEST(SkillEstimation, CalibrationRatesWinningReferenceAboveBenchmark)
{
	GGL::SkillTrackerConfig config = {};
	const GGL::FixedSeriesRecord referenceRecord = { 40, 24, 0 };

	const std::optional<GGL::CalibrationSample> sample = GGL::MakeCalibrationSample(1000.0f, referenceRecord, config);
	ASSERT_TRUE(sample.has_value());

	const std::optional<float> rating = GGL::CombineCalibrationSamples({ *sample }, config.anchorRetireExpectedWinProb, config);
	ASSERT_TRUE(rating.has_value());
	ASSERT_GT(*rating, 1000.0f);
	ASSERT_FALSE(GGL::MakeCalibrationSample(1000.0f, GGL::FixedSeriesRecord{ 0, 0, 8 }, config).has_value());
}

TEST(SkillFixedSeries, BuildFixedSeriesGamesBalancesSidesAndRoundsUpToEven)
{
	std::vector<GGL::FixedSeriesGame> games = GGL::BuildFixedSeriesGames(2, 5, 7, 1);
	ASSERT_EQ(games.size(), size_t(12));
	for (int opponent = 0; opponent < 2; opponent++) {
		int blue = 0, orange = 0;
		for (const GGL::FixedSeriesGame& game : games)
			if (game.opponentIdx == opponent)
				(game.currentTeam == RocketSim::Team::BLUE ? blue : orange)++;
		ASSERT_EQ(blue, 3);
		ASSERT_EQ(orange, 3);
	}
}

TEST(SkillFixedSeries, BuildFixedSeriesGamesUsesSameSeedsForEveryOpponentAndCall)
{
	const int opponentCount = 3;
	std::vector<GGL::FixedSeriesGame> games = GGL::BuildFixedSeriesGames(opponentCount, 6, 7, 2);
	std::vector<GGL::FixedSeriesGame> repeated = GGL::BuildFixedSeriesGames(opponentCount, 6, 7, 2);
	ASSERT_EQ(games.size(), repeated.size());
	for (size_t i = 0; i < games.size(); i++) {
		ASSERT_EQ(games[i].opponentIdx, repeated[i].opponentIdx);
		ASSERT_EQ(games[i].currentTeam, repeated[i].currentTeam);
		ASSERT_EQ(games[i].kickoffSeed, repeated[i].kickoffSeed);
	}

	std::vector<std::vector<uint64_t>> seedsByOpponent(opponentCount);
	for (int opponent = 0; opponent < opponentCount; opponent++) {
		std::map<uint64_t, std::pair<int, int>> sidesBySeed;
		for (const GGL::FixedSeriesGame& game : games) {
			if (game.opponentIdx != opponent)
				continue;

			seedsByOpponent[opponent].push_back(game.kickoffSeed);
			std::pair<int, int>& sides = sidesBySeed[game.kickoffSeed];
			(game.currentTeam == RocketSim::Team::BLUE ? sides.first : sides.second)++;
		}

		ASSERT_EQ(sidesBySeed.size(), size_t(3));
		for (const auto& [seed, sides] : sidesBySeed) {
			ASSERT_EQ(sides.first, 1);
			ASSERT_EQ(sides.second, 1);
		}

		std::sort(seedsByOpponent[opponent].begin(), seedsByOpponent[opponent].end());
	}

	for (int opponent = 1; opponent < opponentCount; opponent++)
		ASSERT_EQ(seedsByOpponent[opponent], seedsByOpponent[0]);

	const uint64_t firstSeed = 7 + 2 * 1'000'003;
	ASSERT_EQ(seedsByOpponent[0].front(), firstSeed);
	ASSERT_EQ(seedsByOpponent[0].back(), firstSeed + 2);
}

TEST(SkillFixedSeries, BuildFixedSeriesGamesIsEmptyWhenDisabled)
{
	ASSERT_TRUE(GGL::BuildFixedSeriesGames(3, 0, 7, 1).empty());
}

TEST(SkillFixedSeries, TallyFixedSeriesGameCountsFromSubjectSide)
{
	using RocketSim::Team;

	struct Case {
		std::string label;
		Team subjectTeam;
		bool goalScored;
		Team scoringTeam;
		GGL::FixedSeriesRecord expected;
	};
	const Case cases[] = {
		{ "blue scores for", Team::BLUE, true, Team::BLUE, { 1, 0, 0 } },
		{ "blue concedes", Team::BLUE, true, Team::ORANGE, { 0, 1, 0 } },
		{ "blue no score", Team::BLUE, false, Team::ORANGE, { 0, 0, 1 } },
		{ "orange scores for", Team::ORANGE, true, Team::ORANGE, { 1, 0, 0 } },
		{ "orange concedes", Team::ORANGE, true, Team::BLUE, { 0, 1, 0 } },
		{ "orange no score", Team::ORANGE, false, Team::ORANGE, { 0, 0, 1 } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		GGL::FixedSeriesRecord record = {};
		GGL::TallyFixedSeriesGame(record, testCase.subjectTeam, testCase.goalScored, testCase.scoringTeam);
		ASSERT_EQ(record.wins, testCase.expected.wins);
		ASSERT_EQ(record.losses, testCase.expected.losses);
		ASSERT_EQ(record.noScoreGames, testCase.expected.noScoreGames);
	}
}
