#include "DecomposedGAE.h"
#include "GAE.h"

#include <vector>

GGL::DecomposedGAEResult GGL::ComputeDecomposedGAE(
	const torch::Tensor& headRewards,
	const torch::Tensor& terminals,
	const torch::Tensor& headValPreds,
	const torch::Tensor& headTruncValPreds,
	const torch::Tensor& sampleWeights,
	float gamma,
	float lambda
) {

	const int64_t numSamples = headRewards.size(0);
	const int64_t numHeads = headRewards.size(1);
	RG_ASSERT(sampleWeights.size(0) == numSamples && sampleWeights.size(1) == numHeads);
	RG_ASSERT(headValPreds.size(0) == numSamples);
	RG_ASSERT(terminals.size(0) == numSamples);

	std::vector<torch::Tensor> advantagesPerHead, targetsPerHead, returnsPerHead;
	advantagesPerHead.reserve(numHeads);
	targetsPerHead.reserve(numHeads);
	returnsPerHead.reserve(numHeads);

	// Raw rewards with no clip keep every head linear, which the weighted combination relies on.
	const float rawRewardScale = 0.f;
	const float noRewardClip = 0.f;

	for (int64_t headIdx = 0; headIdx < numHeads; headIdx++) {
		torch::Tensor advantages, targets, returns;
		float rewardClipPortion = 0.f;
		const torch::Tensor truncValPreds = headTruncValPreds.defined()
			? headTruncValPreds.select(1, headIdx).contiguous()
			: torch::Tensor();

		GAE::Compute(
			headRewards.select(1, headIdx).contiguous(), terminals,
			headValPreds.select(1, headIdx).contiguous(), truncValPreds,
			advantages, targets, returns, rewardClipPortion,
			gamma, lambda, rawRewardScale, noRewardClip);

		advantagesPerHead.push_back(advantages);
		targetsPerHead.push_back(targets);
		returnsPerHead.push_back(returns);
	}

	DecomposedGAEResult result = {};
	result.headAdvantages = torch::stack(advantagesPerHead, 1);
	result.headTargets = torch::stack(targetsPerHead, 1);

	result.advantages = (sampleWeights * result.headAdvantages).sum(1);
	result.valPreds = (sampleWeights * headValPreds).sum(1);
	result.targetValues = (sampleWeights * result.headTargets).sum(1);
	result.returns = (sampleWeights * torch::stack(returnsPerHead, 1)).sum(1);

	return result;
}
