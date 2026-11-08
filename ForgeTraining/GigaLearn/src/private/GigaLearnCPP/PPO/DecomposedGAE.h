#pragma once

#include <GigaLearnInfer/FrameworkTorch.h>

namespace GGL {

	struct DecomposedGAEResult {
		/** [samples × heads] raw per-head advantages and value targets. */
		torch::Tensor headAdvantages, headTargets;

		/** [samples] each sample's heads combined with its weights, Σᵢ wᵢ·xᵢ. */
		torch::Tensor advantages, valPreds, targetValues, returns;
	};

	/**
	 * Runs GAE once per value head in raw reward units (no return scaling, no reward clip),
	 * then combines the heads with each sample's weights [samples × heads].
	 * GAE is linear in rewards and values, so the combined advantage equals single-stream GAE on Σ w·r and Σ w·V
	 * as long as each sample's weights are constant within its episode.
	 * headTruncValPreds is [truncations × heads] and may be undefined when nothing truncated.
	 */
	DecomposedGAEResult ComputeDecomposedGAE(
		const torch::Tensor& headRewards,
		const torch::Tensor& terminals,
		const torch::Tensor& headValPreds,
		const torch::Tensor& headTruncValPreds,
		const torch::Tensor& sampleWeights,
		float gamma,
		float lambda
	);
}
