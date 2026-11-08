#pragma once

#include <GigaLearnInfer/FrameworkTorch.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace GGL {
	/** Critic and advantage statistics over the samples of one even arena player count. */
	struct ModeCriticStats {
		int playerCount = 0;
		int64_t sampleCount = 0;

		/** 1 - Var(target - valPred) / Var(target); absent below 2 samples or with zero target variance. */
		std::optional<float> explainedVariance;

		/** mean((target - valPred)^2), in the units of the targets passed in. */
		float valueError = 0.f;

		float advantageStd = 0.f;
		float avgAbsReturn = 0.f;
		float meanAdvantage = 0.f;
	};

	/**
	 * Splits a rollout batch by each sample's arena player count and summarizes every even count present,
	 * ordered by player count.
	 * All tensors are 1-D CPU tensors of equal length; playerCounts is any integer type.
	 */
	std::vector<ModeCriticStats> ComputePerModeCriticStats(
		const torch::Tensor& valPreds, const torch::Tensor& targetValues,
		const torch::Tensor& advantages, const torch::Tensor& returns, const torch::Tensor& playerCounts);

	/** One value head's statistics over the present samples of one even player count. */
	struct HeadModeStats {
		int playerCount = 0;

		/** Absent below 2 present samples or with zero target variance. */
		std::optional<float> explainedVariance;

		/** Population std of the raw targets; absent when the head has no present samples in this mode. */
		std::optional<float> targetStd;

		/** This head's Σ|w|·σ over this mode's samples, as a fraction of the same sum over all heads. */
		float contributionShare = 0.f;
	};

	struct ValueHeadStats {
		/** This head's Σ|w|·σ over the batch, as a fraction of the same sum over all heads. */
		float contributionShare = 0.f;

		/** Mean of ((target - valPred) / σ)² over present cells: the critic loss in normalized units; 0 with none present. */
		float normalizedLoss = 0.f;

		/** One entry per even player count in the batch, ordered by player count. */
		std::vector<HeadModeStats> modes;
	};

	/**
	 * Summarizes each value head of a decomposed critic. All tensors are CPU tensors.
	 * headValPreds and headTargets are [samples × heads] in raw units; presence and sampleWeights are [samples × heads],
	 * sigmas is [heads], and playerCounts is [samples] of any integer type.
	 */
	std::vector<ValueHeadStats> ComputeValueHeadStats(
		const torch::Tensor& headValPreds,
		const torch::Tensor& headTargets,
		const torch::Tensor& presence,
		const torch::Tensor& sampleWeights,
		const torch::Tensor& sigmas,
		const torch::Tensor& playerCounts);
}
