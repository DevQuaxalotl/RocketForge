#pragma once

#include <GigaLearnInfer/Framework.h>

namespace GGL {
	struct SkillTrackerConfig {
		bool enabled = false;

		// Maximum reusable arena slots available during skill tests
		int maxArenas = 500;

		// Target confidence for each skill mode, from 0-1.
		// A test stops at sigma = minSigma + (1 - targetConfidence) * (initialSigma - minSigma), so this sets the rating's final precision.
		float targetConfidence = 0.95f;

		// Minimum scored games per mode before confidence can stop a skill test
		int minGames = 64;

		// Maximum finished games per mode in one skill test
		int maxGames = 1000;

		// Maximum in-game seconds to wait for a goal before discarding the episode as no-result
		float maxNoScoreGameSeconds = 150.0f;

		// Cancel the skill test as inconclusive if no-score episodes dominate (after minGames no-score games)
		float inconclusiveNoScoreRatio = 0.5f;

		// Min and max time steps to run between skill tests. The actual time steps will be interpolated based on the number of completed games
		uint64_t minTimeSteps =  20'000'000;
		uint64_t maxTimeSteps = 100'000'000;

		// Finished-game interval between progress prints
		int progressPrintInterval = 500;

		// Initial Bayesian skill estimate
		float initialMu = 0.0f;
		float initialSigma = 300.0f;

		// TrueSkill-style update tuning
		float beta = 150.0f;
		float minSigma = 15.0f;

		// Rating sigma growth (added in quadrature) per 100M timesteps elapsed between skill tests
		float ratingDriftSigmaPer100MTimesteps = 50.0f;

		// Games against each fixed reference opponent, per mode (0 disables references and the anchor)
		int referenceGamesPerOpponent = 64;

		// Games per calibration benchmark when rating a reference opponent, per mode
		int referenceCalibrationGames = 64;

		// Expected win probability against the anchor reference above which the anchor is retired
		float anchorRetireExpectedWinProb = 0.90f;

		// Base seed for reference kickoffs
		uint64_t fixedMatchSeed = 1;

		// Maximum persistent benchmark anchors to keep
		int maxSkillBenchmarks = 32;

		// Benchmarks outside this one-sided expected win/loss probability are pruned and excluded from skill tests
		float benchmarkPruneMaxExpectedWinProb = 0.995f;

		// Policies will be inferred deterministically by default during skill tests
		// Turn off to test the stochastic policy
		bool deterministic = true;
	};
}