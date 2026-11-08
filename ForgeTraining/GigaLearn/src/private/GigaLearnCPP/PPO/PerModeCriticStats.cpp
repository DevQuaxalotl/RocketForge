#include "PerModeCriticStats.h"

#include <RLGymCPP/ThreadPool.h>

#include <algorithm>
#include <cmath>
#include <set>

namespace {
	std::set<int64_t> FindEvenPlayerCounts(const torch::Tensor& counts) {
		std::set<int64_t> evenCounts;
		const int64_t* countData = counts.data_ptr<int64_t>();
		for (int64_t i = 0; i < counts.size(0); i++)
			if (countData[i] > 0 && countData[i] % 2 == 0)
				evenCounts.insert(countData[i]);

		return evenCounts;
	}

	std::optional<float> ExplainedVariance(const torch::Tensor& targets, const torch::Tensor& errors) {
		if (targets.size(0) < 2)
			return std::nullopt;

		const float targetVariance = targets.var(/*unbiased*/ false).item<float>();
		if (targetVariance <= 0.f)
			return std::nullopt;

		return 1.f - errors.var(/*unbiased*/ false).item<float>() / targetVariance;
	}

	/**
	 * Sums per-sample terms over [0, numSamples) in parallel.
	 * Samples are split into fixed chunks whose partial sums are added in chunk order, so the result does not depend on scheduling.
	 */
	template <typename AccumulateChunk>
	std::vector<double> SumOverSampleChunks(int64_t numSamples, int64_t numSums, const AccumulateChunk& accumulateChunk) {
		const int64_t numChunks = std::max<int64_t>(1, std::min<int64_t>(numSamples, (int64_t)RLGC::g_ThreadPool.GetNumThreads() * 4));
		std::vector<std::vector<double>> partials((size_t)numChunks, std::vector<double>((size_t)numSums, 0.0));
		const bool runAsync = false;
		RLGC::g_ThreadPool.StartBatchedJobs([&](int chunk) {
			accumulateChunk(chunk * numSamples / numChunks, (chunk + 1) * numSamples / numChunks, partials[chunk].data());
		}, (int)numChunks, runAsync);

		std::vector<double> totals((size_t)numSums, 0.0);
		for (const std::vector<double>& partial : partials)
			for (int64_t sumIdx = 0; sumIdx < numSums; sumIdx++)
				totals[sumIdx] += partial[sumIdx];

		return totals;
	}
}

std::vector<GGL::ModeCriticStats> GGL::ComputePerModeCriticStats(
	const torch::Tensor& valPreds, const torch::Tensor& targetValues,
	const torch::Tensor& advantages, const torch::Tensor& returns, const torch::Tensor& playerCounts)
{
	const torch::Tensor counts = playerCounts.reshape({ -1 }).to(torch::kInt64).contiguous();
	const torch::Tensor values = valPreds.reshape({ -1 }).to(torch::kFloat32);
	const torch::Tensor targets = targetValues.reshape({ -1 }).to(torch::kFloat32);
	const torch::Tensor advantagesFlat = advantages.reshape({ -1 }).to(torch::kFloat32);
	const torch::Tensor returnsFlat = returns.reshape({ -1 }).to(torch::kFloat32);

	const std::set<int64_t> evenCounts = FindEvenPlayerCounts(counts);

	std::vector<ModeCriticStats> result;
	for (const int64_t playerCount : evenCounts) {
		const torch::Tensor mask = counts.eq(playerCount);
		const torch::Tensor modeTargets = targets.masked_select(mask);
		const torch::Tensor modeErrors = modeTargets - values.masked_select(mask);
		const torch::Tensor modeAdvantages = advantagesFlat.masked_select(mask);

		ModeCriticStats stats;
		stats.playerCount = (int)playerCount;
		stats.sampleCount = modeTargets.size(0);
		stats.valueError = modeErrors.square().mean().item<float>();
		stats.advantageStd = modeAdvantages.std(/*unbiased*/ false).item<float>();
		stats.avgAbsReturn = returnsFlat.masked_select(mask).abs().mean().item<float>();
		stats.meanAdvantage = modeAdvantages.mean().item<float>();
		stats.explainedVariance = ExplainedVariance(modeTargets, modeErrors);

		result.push_back(stats);
	}
	return result;
}

std::vector<GGL::ValueHeadStats> GGL::ComputeValueHeadStats(
	const torch::Tensor& headValPreds,
	const torch::Tensor& headTargets,
	const torch::Tensor& presence,
	const torch::Tensor& sampleWeights,
	const torch::Tensor& sigmas,
	const torch::Tensor& playerCounts)
{
	const torch::Tensor counts = playerCounts.reshape({ -1 }).to(torch::kInt64).contiguous();
	const std::set<int64_t> evenCounts = FindEvenPlayerCounts(counts);
	const std::vector<int64_t> modePlayerCounts(evenCounts.begin(), evenCounts.end());
	const int64_t numSamples = headValPreds.size(0);
	const int64_t numHeads = headValPreds.size(1);
	const int64_t numModes = (int64_t)modePlayerCounts.size();
	const int64_t numCells = numModes * numHeads;

	const torch::Tensor predsFloat = headValPreds.to(torch::kFloat32).contiguous();
	const torch::Tensor targetsFloat = headTargets.to(torch::kFloat32).contiguous();
	const torch::Tensor presenceFloat = presence.to(torch::kFloat32).contiguous();
	const torch::Tensor weightsFloat = sampleWeights.to(torch::kFloat32).contiguous();
	const torch::Tensor sigmasFloat = sigmas.to(torch::kFloat32).contiguous();
	const float* predData = predsFloat.data_ptr<float>();
	const float* targetData = targetsFloat.data_ptr<float>();
	const float* presenceData = presenceFloat.data_ptr<float>();
	const float* weightData = weightsFloat.data_ptr<float>();
	const float* sigmaData = sigmasFloat.data_ptr<float>();
	const int64_t* countData = counts.data_ptr<int64_t>();

	// -1 for samples whose player count is not one of the reported modes.
	std::vector<int32_t> sampleModes((size_t)numSamples, -1);
	for (int64_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
		for (int64_t modeIdx = 0; modeIdx < numModes; modeIdx++)
			if (countData[sampleIdx] == modePlayerCounts[modeIdx])
				sampleModes[sampleIdx] = (int32_t)modeIdx;

	// First pass: counts, sums and contributions, as [heads] arrays followed by [modes × heads] arrays.
	enum FirstPassArray { HEAD_PRESENT, HEAD_SQUARED_ERROR, HEAD_CONTRIBUTION, MODE_COUNT, MODE_TARGET_SUM, MODE_ERROR_SUM, MODE_CONTRIBUTION };
	const int64_t modeArraysOffset = 3 * numHeads;
	auto firstPassIndex = [&](FirstPassArray array, int64_t modeIdx, int64_t headIdx) {
		if (array < MODE_COUNT)
			return array * numHeads + headIdx;
		return modeArraysOffset + (array - MODE_COUNT) * numCells + modeIdx * numHeads + headIdx;
	};

	const std::vector<double> sums = SumOverSampleChunks(numSamples, modeArraysOffset + 4 * numCells, [&](int64_t begin, int64_t end, double* partial) {
		for (int64_t sampleIdx = begin; sampleIdx < end; sampleIdx++) {
			const int32_t modeIdx = sampleModes[sampleIdx];
			const int64_t rowOffset = sampleIdx * numHeads;

			for (int64_t headIdx = 0; headIdx < numHeads; headIdx++) {
				if (presenceData[rowOffset + headIdx] == 0.f)
					continue;

				const float target = targetData[rowOffset + headIdx];
				const float error = target - predData[rowOffset + headIdx];
				const double contribution = (double)std::fabs(weightData[rowOffset + headIdx]) * sigmaData[headIdx];
				partial[firstPassIndex(HEAD_PRESENT, 0, headIdx)] += 1;
				partial[firstPassIndex(HEAD_SQUARED_ERROR, 0, headIdx)] += (double)error * error;
				partial[firstPassIndex(HEAD_CONTRIBUTION, 0, headIdx)] += contribution;

				if (modeIdx < 0)
					continue;

				partial[firstPassIndex(MODE_COUNT, modeIdx, headIdx)] += 1;
				partial[firstPassIndex(MODE_TARGET_SUM, modeIdx, headIdx)] += target;
				partial[firstPassIndex(MODE_ERROR_SUM, modeIdx, headIdx)] += error;
				partial[firstPassIndex(MODE_CONTRIBUTION, modeIdx, headIdx)] += contribution;
			}
		}
	});

	std::vector<double> targetMeans((size_t)numCells), errorMeans((size_t)numCells);
	for (int64_t modeIdx = 0; modeIdx < numModes; modeIdx++) {
		for (int64_t headIdx = 0; headIdx < numHeads; headIdx++) {
			const int64_t cell = modeIdx * numHeads + headIdx;
			const double sampleCount = std::max(1.0, sums[firstPassIndex(MODE_COUNT, modeIdx, headIdx)]);
			targetMeans[cell] = sums[firstPassIndex(MODE_TARGET_SUM, modeIdx, headIdx)] / sampleCount;
			errorMeans[cell] = sums[firstPassIndex(MODE_ERROR_SUM, modeIdx, headIdx)] / sampleCount;
		}
	}

	// Second pass: squared deviations from the per-mode means, which avoids the cancellation of Σx² - n·mean².
	const std::vector<double> squaredDeviations = SumOverSampleChunks(numSamples, 2 * numCells, [&](int64_t begin, int64_t end, double* partial) {
		for (int64_t sampleIdx = begin; sampleIdx < end; sampleIdx++) {
			const int32_t modeIdx = sampleModes[sampleIdx];
			if (modeIdx < 0)
				continue;

			const int64_t rowOffset = sampleIdx * numHeads;
			const int64_t cellOffset = modeIdx * numHeads;
			for (int64_t headIdx = 0; headIdx < numHeads; headIdx++) {
				if (presenceData[rowOffset + headIdx] == 0.f)
					continue;

				const float target = targetData[rowOffset + headIdx];
				const double targetDeviation = target - targetMeans[cellOffset + headIdx];
				const double errorDeviation = (target - predData[rowOffset + headIdx]) - errorMeans[cellOffset + headIdx];
				partial[cellOffset + headIdx] += targetDeviation * targetDeviation;
				partial[numCells + cellOffset + headIdx] += errorDeviation * errorDeviation;
			}
		}
	});

	double totalContribution = 0;
	std::vector<double> modeTotals((size_t)numModes, 0.0);
	for (int64_t headIdx = 0; headIdx < numHeads; headIdx++) {
		totalContribution += sums[firstPassIndex(HEAD_CONTRIBUTION, 0, headIdx)];
		for (int64_t modeIdx = 0; modeIdx < numModes; modeIdx++)
			modeTotals[modeIdx] += sums[firstPassIndex(MODE_CONTRIBUTION, modeIdx, headIdx)];
	}

	std::vector<ValueHeadStats> result((size_t)numHeads);
	for (int64_t headIdx = 0; headIdx < numHeads; headIdx++) {
		ValueHeadStats& stats = result[headIdx];
		if (totalContribution > 0)
			stats.contributionShare = (float)(sums[firstPassIndex(HEAD_CONTRIBUTION, 0, headIdx)] / totalContribution);

		const double presentCount = sums[firstPassIndex(HEAD_PRESENT, 0, headIdx)];
		if (presentCount > 0) {
			const double sigma = sigmaData[headIdx];
			stats.normalizedLoss = (float)(sums[firstPassIndex(HEAD_SQUARED_ERROR, 0, headIdx)] / (presentCount * sigma * sigma));
		}

		for (int64_t modeIdx = 0; modeIdx < numModes; modeIdx++) {
			const int64_t cell = modeIdx * numHeads + headIdx;
			const double sampleCount = sums[firstPassIndex(MODE_COUNT, modeIdx, headIdx)];
			const double targetVariance = squaredDeviations[cell] / std::max(1.0, sampleCount);
			const double errorVariance = squaredDeviations[numCells + cell] / std::max(1.0, sampleCount);

			HeadModeStats mode;
			mode.playerCount = (int)modePlayerCounts[modeIdx];
			if (sampleCount >= 2 && targetVariance > 0)
				mode.explainedVariance = (float)(1 - errorVariance / targetVariance);
			if (sampleCount > 0)
				mode.targetStd = (float)std::sqrt(targetVariance);
			if (modeTotals[modeIdx] > 0)
				mode.contributionShare = (float)(sums[firstPassIndex(MODE_CONTRIBUTION, modeIdx, headIdx)] / modeTotals[modeIdx]);

			stats.modes.push_back(mode);
		}
	}

	return result;
}
