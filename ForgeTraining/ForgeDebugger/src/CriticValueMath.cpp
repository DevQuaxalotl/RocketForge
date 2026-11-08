#include "CriticValueMath.h"

#include <RLGymCPP/Framework.h>

#include <algorithm>
#include <cmath>

namespace ForgeDebug {

	float DenormalizeHeadValue(float normalized, const ValueHeadStats& stats, float sigmaFloor) {
		if (!stats.initialized)
			return normalized;

		const double unflooredSigma = std::sqrt(std::max(stats.meanSq - stats.mean * stats.mean, 0.0));
		const double sigma = std::max(unflooredSigma, (double)sigmaFloor);

		return (float)(normalized * sigma + stats.mean);
	}

	CriticDebugData CombineDecomposedCritic(
		const std::vector<float>& normalizedOutputs, const std::vector<DecomposedCriticHead>& heads, float sigmaFloor) {
		RG_ASSERT(normalizedOutputs.size() == heads.size());

		CriticDebugData result;
		result.isDecomposed = true;
		result.heads.reserve(heads.size());

		for (size_t headIdx = 0; headIdx < heads.size(); headIdx++) {
			const DecomposedCriticHead& head = heads[headIdx];

			CriticHeadDebugData headData;
			headData.name = head.stats.name;
			headData.raw = DenormalizeHeadValue(normalizedOutputs[headIdx], head.stats, sigmaFloor);
			headData.weight = head.weight;
			headData.contribution = head.weight ? *head.weight * headData.raw : 0.f;

			result.total += headData.contribution;
			result.heads.push_back(headData);
		}

		return result;
	}

	CriticDebugData MakeSingleCriticEstimate(float output, float returnStd) {
		CriticDebugData result;
		result.total = output * returnStd;
		return result;
	}

	float WelfordStd(double runningVariance, int64_t count) {
		if (count < 2)
			return 1.f;

		double variance = runningVariance / (double)(count - 1);
		if (variance == 0.0)
			variance = 1.0;

		return (float)std::sqrt(variance);
	}
}
