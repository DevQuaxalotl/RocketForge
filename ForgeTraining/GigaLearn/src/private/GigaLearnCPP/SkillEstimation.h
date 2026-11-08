#pragma once

#include "PolicyVersionManager.h"

#include <optional>
#include <vector>

namespace GGL {

	/** Standard normal probability density at x. **/
	float NormalPDF(float x);

	/** Standard normal cumulative distribution at x. **/
	float NormalCDF(float x);

	/** Probability that a beats b, given both rating uncertainties and the per-game performance spread (config.beta). **/
	float GetExpectedWinProbability(const SkillEstimate& a, const SkillEstimate& b, const SkillTrackerConfig& config);

	/**
	 * Rating gap (current minus opponent) implied by a win/loss record alone, with the opponent treated as exact.
	 * Starts from a {0, initialSigma} prior and replays the games interleaved evenly, so the result depends only on the counts.
	 **/
	SkillEstimate EstimateRatingGap(int wins, int losses, const SkillTrackerConfig& config);

	/** True when the likelier side's win probability is at least threshold (clamped to [0.5, 0.999999]). **/
	bool IsWinProbabilityLopsided(float winProbability, float threshold);

	/** Probability that a player with the given rating gap beats an exactly rated opponent. **/
	float GetGapWinProbability(const SkillEstimate& gap, const SkillTrackerConfig& config);

	/** One calibration result: the benchmark's rating and the reference-minus-benchmark gap from their games. **/
	struct CalibrationSample {
		float benchmarkRating = 0.0f;
		SkillEstimate gap;
	};

	/** Calibration sample from the reference's record (as subject) against a benchmark; empty when no game was scored. **/
	std::optional<CalibrationSample> MakeCalibrationSample(
		float benchmarkRating,
		const FixedSeriesRecord& subjectRecord,
		const SkillTrackerConfig& config);

	/**
	 * MMR = opponentRating + the current policy's rating gap from its record against that opponent.
	 * Empty when no game was scored.
	 **/
	std::optional<SkillEstimate> EstimateFixedOpponentMMR(
		float opponentRating,
		const FixedSeriesRecord& record,
		const SkillTrackerConfig& config);

	/** Inverse-variance mean of benchmarkRating + gap.mu over non-lopsided samples; nullopt when none qualify. **/
	std::optional<float> CombineCalibrationSamples(
		const std::vector<CalibrationSample>& samples,
		float lopsidedThreshold,
		const SkillTrackerConfig& config);
}
