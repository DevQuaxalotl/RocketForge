#include "SkillEstimation.h"

#include <algorithm>
#include <cmath>
#include <vector>

float GGL::NormalPDF(float x) {
	static constexpr float INV_SQRT_2PI = 0.3989422804014327f;
	return INV_SQRT_2PI * expf(-0.5f * x * x);
}

float GGL::NormalCDF(float x) {
	return 0.5f * erfcf(-x / sqrtf(2.0f));
}

float GGL::GetExpectedWinProbability(const SkillEstimate& a, const SkillEstimate& b, const SkillTrackerConfig& config) {
	float variance = a.sigma * a.sigma + b.sigma * b.sigma + 2.0f * config.beta * config.beta;
	if (variance <= 0)
		return a.mu > b.mu ? 1.0f : 0.0f;

	return NormalCDF((a.mu - b.mu) / sqrtf(variance));
}

GGL::SkillEstimate GGL::EstimateRatingGap(int wins, int losses, const SkillTrackerConfig& config) {
	SkillEstimate gap = { 0.0f, config.initialSigma };
	const SkillEstimate exactOpponent = { 0.0f, 0.0f };

	const int totalGames = wins + losses;
	const int minorityGames = std::min(wins, losses);
	const bool minorityWon = wins < losses;

	// Spread the rarer result evenly, centered in its slot, so a record and its mirror replay as mirrored sequences.
	std::vector<bool> outcomes(totalGames, !minorityWon);
	for (int minorityIndex = 0; minorityIndex < minorityGames; minorityIndex++) {
		const int position = (int)((int64_t)(2 * minorityIndex + 1) * totalGames / (2 * minorityGames));
		outcomes[position] = minorityWon;
	}

	for (int gameIndex = 0; gameIndex < totalGames; gameIndex++)
		PolicyVersionManager::UpdateSkillEstimateAgainstFixedOpponent(gap, exactOpponent, outcomes[gameIndex], config);

	return gap;
}

bool GGL::IsWinProbabilityLopsided(float winProbability, float threshold) {
	const float clampedThreshold = std::clamp(threshold, 0.5f, 0.999999f);
	return std::max(winProbability, 1.0f - winProbability) >= clampedThreshold;
}

float GGL::GetGapWinProbability(const SkillEstimate& gap, const SkillTrackerConfig& config) {
	return GetExpectedWinProbability(gap, SkillEstimate{ 0.0f, 0.0f }, config);
}

std::optional<float> GGL::CombineCalibrationSamples(
	const std::vector<CalibrationSample>& samples,
	float lopsidedThreshold,
	const SkillTrackerConfig& config) {

	double weightedRatingSum = 0.0;
	double totalWeight = 0.0;

	for (const CalibrationSample& sample : samples) {
		if (IsWinProbabilityLopsided(GetGapWinProbability(sample.gap, config), lopsidedThreshold))
			continue;

		// A zero sigma would claim infinite certainty and take all of the weight.
		const double sigma = std::max({ sample.gap.sigma, config.minSigma, 1e-3f });
		const double weight = 1.0 / (sigma * sigma);
		weightedRatingSum += weight * (sample.benchmarkRating + sample.gap.mu);
		totalWeight += weight;
	}

	if (totalWeight <= 0.0)
		return std::nullopt;

	return (float)(weightedRatingSum / totalWeight);
}

std::optional<GGL::CalibrationSample> GGL::MakeCalibrationSample(
	float benchmarkRating,
	const FixedSeriesRecord& subjectRecord,
	const SkillTrackerConfig& config) {

	if (subjectRecord.wins + subjectRecord.losses == 0)
		return std::nullopt;

	return CalibrationSample{ benchmarkRating, EstimateRatingGap(subjectRecord.wins, subjectRecord.losses, config) };
}

std::optional<GGL::SkillEstimate> GGL::EstimateFixedOpponentMMR(
	float opponentRating,
	const FixedSeriesRecord& record,
	const SkillTrackerConfig& config) {

	if (record.wins + record.losses == 0)
		return std::nullopt;

	const SkillEstimate gap = EstimateRatingGap(record.wins, record.losses, config);
	return SkillEstimate{ opponentRating + gap.mu, gap.sigma };
}
