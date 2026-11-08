#include <gtest/gtest.h>

#include <GigaLearnCPP/PPO/PerModeCriticStats.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace {
	bool Near(float a, float b, float tolerance = 1e-5f) {
		return std::fabs(a - b) <= tolerance;
	}

	double RelativeTolerance(double expected) {
		return 1e-5 * std::fabs(expected);
	}

	struct HeadStatsBatch {
		torch::Tensor valPreds, targets, presence, sampleWeights, sigmas, playerCounts;
	};

	HeadStatsBatch MakeRandomHeadStatsBatch(int64_t numSamples, int64_t numHeads) {
		torch::manual_seed(7);

		const std::vector<int64_t> choosablePlayerCounts = { 2, 3, 4, 6 };
		const torch::Tensor countChoices = torch::randint(0, (int64_t)choosablePlayerCounts.size(), { numSamples });
		const torch::Tensor countTable = torch::tensor(choosablePlayerCounts);

		HeadStatsBatch batch;
		batch.valPreds = torch::randn({ numSamples, numHeads }) * 2.f;
		batch.targets = torch::randn({ numSamples, numHeads }) * 3.f + 1.f;
		batch.presence = (torch::rand({ numSamples, numHeads }) < 0.7).to(torch::kFloat32);
		batch.sampleWeights = torch::rand({ numSamples, numHeads }) * 4.f - 2.f;
		batch.sigmas = torch::rand({ numHeads }) * 2.5f + 0.5f;
		batch.playerCounts = countTable.index_select(0, countChoices);
		return batch;
	}

	std::vector<GGL::ValueHeadStats> ComputeStats(const HeadStatsBatch& batch) {
		return GGL::ComputeValueHeadStats(batch.valPreds, batch.targets, batch.presence, batch.sampleWeights, batch.sigmas, batch.playerCounts);
	}

	uint32_t FloatBits(float value) {
		return std::bit_cast<uint32_t>(value);
	}

	bool SameOptionalBits(const std::optional<float>& a, const std::optional<float>& b) {
		return a.has_value() == b.has_value() && (!a.has_value() || FloatBits(*a) == FloatBits(*b));
	}
}

TEST(PerModeCriticStats, SplitsByEvenPlayerCount) {
	// 1v1 samples: perfect critic. 2v2 samples: constant error of 1. One 3-player sample is ignored.
	torch::Tensor targets   = torch::tensor(std::vector<float>{ 1.f, 2.f, 3.f,   1.f, 2.f, 3.f,   5.f });
	torch::Tensor valPreds  = torch::tensor(std::vector<float>{ 1.f, 2.f, 3.f,   0.f, 1.f, 2.f,   0.f });
	torch::Tensor advantages = torch::tensor(std::vector<float>{ 1.f, -1.f, 0.f,  2.f, 2.f, 2.f,   9.f });
	torch::Tensor returns   = torch::tensor(std::vector<float>{ -1.f, 1.f, 1.f,   4.f, 4.f, -4.f,  9.f });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 2, 2, 2,  4, 4, 4,  3 });

	std::vector<GGL::ModeCriticStats> stats =
		GGL::ComputePerModeCriticStats(valPreds, targets, advantages, returns, playerCounts);

	ASSERT_EQ(stats.size(), (size_t)2);
	const GGL::ModeCriticStats& oneVOne = stats[0];
	const GGL::ModeCriticStats& twoVTwo = stats[1];

	ASSERT_EQ(oneVOne.playerCount, 2);
	ASSERT_EQ(oneVOne.sampleCount, (int64_t)3);
	ASSERT_TRUE(Near(*oneVOne.explainedVariance, 1.f));
	ASSERT_TRUE(Near(oneVOne.valueError, 0.f));
	ASSERT_TRUE(Near(oneVOne.meanAdvantage, 0.f));
	ASSERT_TRUE(Near(oneVOne.advantageStd, std::sqrt(2.f / 3.f)));
	ASSERT_TRUE(Near(oneVOne.avgAbsReturn, 1.f));

	// A constant offset leaves the error variance at 0, so explained variance is 1 while value error is 1.
	ASSERT_EQ(twoVTwo.playerCount, 4);
	ASSERT_TRUE(Near(*twoVTwo.explainedVariance, 1.f));
	ASSERT_TRUE(Near(twoVTwo.valueError, 1.f));
	ASSERT_TRUE(Near(twoVTwo.advantageStd, 0.f));
	ASSERT_TRUE(Near(twoVTwo.avgAbsReturn, 4.f));
}

TEST(PerModeCriticStats, ExplainedVarianceOfAMeanPredictorIsZero) {
	torch::Tensor targets  = torch::tensor(std::vector<float>{ 0.f, 2.f, 4.f });
	torch::Tensor valPreds = torch::tensor(std::vector<float>{ 2.f, 2.f, 2.f });
	torch::Tensor zeros = torch::zeros({ 3 });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 6, 6, 6 });

	std::vector<GGL::ModeCriticStats> stats = GGL::ComputePerModeCriticStats(valPreds, targets, zeros, zeros, playerCounts);

	ASSERT_EQ(stats.size(), (size_t)1);
	ASSERT_EQ(stats[0].playerCount, 6);
	ASSERT_TRUE(Near(*stats[0].explainedVariance, 0.f));
}

TEST(PerModeCriticStats, SkipsExplainedVarianceWithoutTargetVariance) {
	torch::Tensor constant = torch::full({ 4 }, 3.f);
	torch::Tensor single = torch::tensor(std::vector<float>{ 1.f });
	torch::Tensor fourPlayers = torch::tensor(std::vector<int8_t>{ 4, 4, 4, 4 });
	torch::Tensor onePlayerPair = torch::tensor(std::vector<int8_t>{ 2 });

	ASSERT_FALSE(GGL::ComputePerModeCriticStats(constant, constant, constant, constant, fourPlayers)[0].explainedVariance.has_value());
	ASSERT_FALSE(GGL::ComputePerModeCriticStats(single, single, single, single, onePlayerPair)[0].explainedVariance.has_value());
}

TEST(PerModeCriticStats, ValueHeadExplainedVariancePerMode) {
	// Head 0: perfect in 1v1 (count 2), constant error of 1 in 2v2 (count 4).
	// Head 1: absent in 1v1, perfect in 2v2.
	torch::Tensor targets = torch::tensor(std::vector<float>{
		1.f, 0.f,   2.f, 0.f,   3.f, 0.f,
		1.f, 5.f,   2.f, 6.f,   3.f, 7.f }).reshape({ 6, 2 });
	torch::Tensor valPreds = torch::tensor(std::vector<float>{
		1.f, 9.f,   2.f, 9.f,   3.f, 9.f,
		0.f, 5.f,   1.f, 6.f,   2.f, 7.f }).reshape({ 6, 2 });
	torch::Tensor presence = torch::tensor(std::vector<float>{
		1.f, 0.f,   1.f, 0.f,   1.f, 0.f,
		1.f, 1.f,   1.f, 1.f,   1.f, 1.f }).reshape({ 6, 2 });
	torch::Tensor sampleWeights = presence.clone();
	torch::Tensor sigmas = torch::tensor(std::vector<float>{ 1.f, 1.f });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 2, 2, 2, 4, 4, 4 });

	std::vector<GGL::ValueHeadStats> stats =
		GGL::ComputeValueHeadStats(valPreds, targets, presence, sampleWeights, sigmas, playerCounts);

	ASSERT_EQ(stats.size(), (size_t)2);
	ASSERT_EQ(stats[0].modes.size(), (size_t)2);
	ASSERT_EQ(stats[0].modes[0].playerCount, 2);
	ASSERT_TRUE(Near(stats[0].modes[0].explainedVariance.value(), 1.f));

	// A constant offset leaves the error variance at 0, so EV is still 1.
	ASSERT_TRUE(Near(stats[0].modes[1].explainedVariance.value(), 1.f));
	ASSERT_FALSE(stats[1].modes[0].explainedVariance.has_value());
	ASSERT_TRUE(Near(stats[1].modes[1].explainedVariance.value(), 1.f));
}

TEST(PerModeCriticStats, ValueHeadContributionShare) {
	// Two samples; |w|·σ sums: head 0 = (2 + 2)·1 = 4, head 1 = (1 + 1)·4 = 8.
	torch::Tensor zeros = torch::zeros({ 2, 2 });
	torch::Tensor presence = torch::ones({ 2, 2 });
	torch::Tensor sampleWeights = torch::tensor(std::vector<float>{ -2.f, 1.f, 2.f, 1.f }).reshape({ 2, 2 });
	torch::Tensor sigmas = torch::tensor(std::vector<float>{ 1.f, 4.f });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 2, 2 });

	std::vector<GGL::ValueHeadStats> stats =
		GGL::ComputeValueHeadStats(zeros, zeros, presence, sampleWeights, sigmas, playerCounts);

	ASSERT_TRUE(Near(stats[0].contributionShare, 1.f / 3.f));
	ASSERT_TRUE(Near(stats[1].contributionShare, 2.f / 3.f));
}

TEST(PerModeCriticStats, ValueHeadDetailedStats) {
	// Rows are samples (player counts 2, 2, 4, 4); columns are heads. Head 1 is absent in sample 0.
	torch::Tensor targets = torch::tensor(std::vector<float>{
		1.f, 9.f,   3.f, 4.f,   2.f, 1.f,   6.f, 3.f }).reshape({ 4, 2 });
	torch::Tensor valPreds = torch::tensor(std::vector<float>{
		0.f, 0.f,   3.f, 2.f,   2.f, 1.f,   4.f, 3.f }).reshape({ 4, 2 });
	torch::Tensor presence = torch::tensor(std::vector<float>{
		1.f, 0.f,   1.f, 1.f,   1.f, 1.f,   1.f, 1.f }).reshape({ 4, 2 });
	torch::Tensor sampleWeights = presence.clone();
	torch::Tensor sigmas = torch::tensor(std::vector<float>{ 2.f, 1.f });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 2, 2, 4, 4 });

	std::vector<GGL::ValueHeadStats> stats =
		GGL::ComputeValueHeadStats(valPreds, targets, presence, sampleWeights, sigmas, playerCounts);

	// Head 0 errors 1, 0, 0, 2 over σ = 2; head 1 errors 2, 0, 0 over its three present cells.
	ASSERT_TRUE(Near(stats[0].normalizedLoss, (0.25f + 1.f) / 4.f));
	ASSERT_TRUE(Near(stats[1].normalizedLoss, 4.f / 3.f));

	ASSERT_TRUE(Near(stats[0].modes[0].targetStd.value(), 1.f));
	ASSERT_TRUE(Near(stats[0].modes[1].targetStd.value(), 2.f));
	ASSERT_TRUE(Near(stats[1].modes[0].targetStd.value(), 0.f));
	ASSERT_TRUE(Near(stats[1].modes[1].targetStd.value(), 1.f));

	// |w|·σ per mode: 2-player head 0 = 2·2, head 1 = 1·1; 4-player head 0 = 2·2, head 1 = 2·1.
	ASSERT_TRUE(Near(stats[0].modes[0].contributionShare, 0.8f));
	ASSERT_TRUE(Near(stats[1].modes[0].contributionShare, 0.2f));
	ASSERT_TRUE(Near(stats[0].modes[1].contributionShare, 2.f / 3.f));
	ASSERT_TRUE(Near(stats[1].modes[1].contributionShare, 1.f / 3.f));
}

TEST(PerModeCriticStats, ValueHeadAbsentInModeHasNoTargetStd) {
	torch::Tensor values = torch::zeros({ 2, 1 });
	torch::Tensor presence = torch::zeros({ 2, 1 });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 2, 2 });

	std::vector<GGL::ValueHeadStats> stats =
		GGL::ComputeValueHeadStats(values, values, presence, presence, torch::ones({ 1 }), playerCounts);

	ASSERT_FALSE(stats[0].modes[0].targetStd.has_value());
	ASSERT_EQ(stats[0].modes[0].contributionShare, 0.f);
	ASSERT_EQ(stats[0].normalizedLoss, 0.f);
}

TEST(PerModeCriticStats, ValueHeadStatsMatchesNaiveReferenceOnRandomBatch) {
	const int64_t numSamples = 2000;
	const int64_t numHeads = 7;
	const HeadStatsBatch batch = MakeRandomHeadStatsBatch(numSamples, numHeads);

	const float* predData = batch.valPreds.data_ptr<float>();
	const float* targetData = batch.targets.data_ptr<float>();
	const float* presenceData = batch.presence.data_ptr<float>();
	const float* weightData = batch.sampleWeights.data_ptr<float>();
	const float* sigmaData = batch.sigmas.data_ptr<float>();
	const int64_t* countData = batch.playerCounts.data_ptr<int64_t>();

	const std::vector<int64_t> evenCounts = { 2, 4, 6 };

	// Whole-batch totals per head, and per (even count, head) lists of the present targets and errors.
	std::vector<double> headContribution((size_t)numHeads, 0.0), headSquaredError((size_t)numHeads, 0.0), headPresent((size_t)numHeads, 0.0);
	std::map<int64_t, std::vector<double>> modeContribution;
	std::map<int64_t, std::vector<std::vector<double>>> modeTargets, modeErrors;
	for (const int64_t playerCount : evenCounts) {
		modeContribution[playerCount].assign((size_t)numHeads, 0.0);
		modeTargets[playerCount].resize((size_t)numHeads);
		modeErrors[playerCount].resize((size_t)numHeads);
	}

	for (int64_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
		for (int64_t headIdx = 0; headIdx < numHeads; headIdx++) {
			const int64_t cell = sampleIdx * numHeads + headIdx;
			if (presenceData[cell] == 0.f)
				continue;

			const double error = (double)targetData[cell] - predData[cell];
			const double contribution = std::fabs((double)weightData[cell]) * sigmaData[headIdx];
			headContribution[headIdx] += contribution;
			headSquaredError[headIdx] += error * error;
			headPresent[headIdx] += 1;

			if (modeTargets.count(countData[sampleIdx]) == 0)
				continue;

			modeContribution[countData[sampleIdx]][headIdx] += contribution;
			modeTargets[countData[sampleIdx]][headIdx].push_back(targetData[cell]);
			modeErrors[countData[sampleIdx]][headIdx].push_back(error);
		}
	}

	auto populationVariance = [](const std::vector<double>& values) {
		double mean = 0;
		for (const double value : values)
			mean += value;
		mean /= (double)values.size();

		double variance = 0;
		for (const double value : values)
			variance += (value - mean) * (value - mean);
		return variance / (double)values.size();
	};

	double totalContribution = 0;
	for (const double contribution : headContribution)
		totalContribution += contribution;

	const std::vector<GGL::ValueHeadStats> stats = ComputeStats(batch);
	ASSERT_EQ(stats.size(), (size_t)numHeads);

	for (int64_t headIdx = 0; headIdx < numHeads; headIdx++) {
		SCOPED_TRACE("head " + std::to_string(headIdx));
		const GGL::ValueHeadStats& headStats = stats[headIdx];

		const double expectedShare = headContribution[headIdx] / totalContribution;
		ASSERT_NEAR(headStats.contributionShare, expectedShare, RelativeTolerance(expectedShare));

		const double sigma = sigmaData[headIdx];
		const double expectedLoss = headSquaredError[headIdx] / (headPresent[headIdx] * sigma * sigma);
		ASSERT_NEAR(headStats.normalizedLoss, expectedLoss, RelativeTolerance(expectedLoss));

		ASSERT_EQ(headStats.modes.size(), evenCounts.size());
		for (size_t modeIdx = 0; modeIdx < evenCounts.size(); modeIdx++) {
			const int64_t playerCount = evenCounts[modeIdx];
			SCOPED_TRACE("players " + std::to_string(playerCount));
			const GGL::HeadModeStats& modeStats = headStats.modes[modeIdx];
			const std::vector<double>& targets = modeTargets[playerCount][headIdx];
			const std::vector<double>& errors = modeErrors[playerCount][headIdx];
			ASSERT_EQ(modeStats.playerCount, (int)playerCount);

			double modeTotal = 0;
			for (const double contribution : modeContribution[playerCount])
				modeTotal += contribution;

			const double expectedModeShare = modeContribution[playerCount][headIdx] / modeTotal;
			ASSERT_NEAR(modeStats.contributionShare, expectedModeShare, RelativeTolerance(expectedModeShare));

			ASSERT_EQ(modeStats.targetStd.has_value(), !targets.empty());
			const double targetVariance = targets.empty() ? 0.0 : populationVariance(targets);
			if (!targets.empty()) {
				const double expectedStd = std::sqrt(targetVariance);
				ASSERT_NEAR(*modeStats.targetStd, expectedStd, RelativeTolerance(expectedStd));
			}

			const bool hasExplainedVariance = targets.size() >= 2 && targetVariance > 0;
			ASSERT_EQ(modeStats.explainedVariance.has_value(), hasExplainedVariance);
			if (hasExplainedVariance)
				ASSERT_NEAR(*modeStats.explainedVariance, 1.0 - populationVariance(errors) / targetVariance, 1e-5);
		}
	}
}

TEST(PerModeCriticStats, ValueHeadStatsOddPlayerCountsOnlyAffectWholeBatchStats) {
	// Samples 0 and 1 are count 2; sample 2 is count 3 and gets no mode entry.
	// Head 0 errors are 1, 0, 4 (σ = 1); head 1 errors are 2, 0, -1 (σ = 2).
	torch::Tensor targets = torch::tensor(std::vector<float>{ 1.f, 2.f,   3.f, 4.f,   10.f, 0.f }).reshape({ 3, 2 });
	torch::Tensor valPreds = torch::tensor(std::vector<float>{ 0.f, 0.f,   3.f, 4.f,   6.f, 1.f }).reshape({ 3, 2 });
	torch::Tensor presence = torch::ones({ 3, 2 });
	torch::Tensor sampleWeights = torch::tensor(std::vector<float>{ 1.f, 1.f,   1.f, 1.f,   2.f, 3.f }).reshape({ 3, 2 });
	torch::Tensor sigmas = torch::tensor(std::vector<float>{ 1.f, 2.f });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 2, 2, 3 });

	std::vector<GGL::ValueHeadStats> stats =
		GGL::ComputeValueHeadStats(valPreds, targets, presence, sampleWeights, sigmas, playerCounts);

	ASSERT_EQ(stats.size(), (size_t)2);

	// Whole batch: Σ|w|·σ is 4 for head 0 and 10 for head 1.
	ASSERT_TRUE(Near(stats[0].contributionShare, 4.f / 14.f));
	ASSERT_TRUE(Near(stats[1].contributionShare, 10.f / 14.f));
	ASSERT_TRUE(Near(stats[0].normalizedLoss, 17.f / 3.f, 1e-4f));
	ASSERT_TRUE(Near(stats[1].normalizedLoss, 5.f / 12.f));

	// Count 2 only: Σ|w|·σ is 2 for head 0 and 4 for head 1.
	ASSERT_EQ(stats[0].modes.size(), (size_t)1);
	ASSERT_EQ(stats[1].modes.size(), (size_t)1);
	ASSERT_EQ(stats[0].modes[0].playerCount, 2);
	ASSERT_TRUE(Near(stats[0].modes[0].contributionShare, 1.f / 3.f));
	ASSERT_TRUE(Near(stats[1].modes[0].contributionShare, 2.f / 3.f));
	ASSERT_TRUE(Near(stats[0].modes[0].targetStd.value(), 1.f));
	ASSERT_TRUE(Near(stats[1].modes[0].targetStd.value(), 1.f));
	ASSERT_TRUE(Near(stats[0].modes[0].explainedVariance.value(), 0.75f));
	ASSERT_TRUE(Near(stats[1].modes[0].explainedVariance.value(), 0.f));
}

TEST(PerModeCriticStats, ValueHeadStatsAllOddPlayerCountsHaveNoModes) {
	// Head 0 errors are 2, -2 (σ = 1); head 1 errors are 3, 0 (σ = 2).
	torch::Tensor targets = torch::tensor(std::vector<float>{ 2.f, 4.f,   1.f, 0.f }).reshape({ 2, 2 });
	torch::Tensor valPreds = torch::tensor(std::vector<float>{ 0.f, 1.f,   3.f, 0.f }).reshape({ 2, 2 });
	torch::Tensor presence = torch::ones({ 2, 2 });
	torch::Tensor sampleWeights = torch::tensor(std::vector<float>{ 1.f, 2.f,   3.f, 1.f }).reshape({ 2, 2 });
	torch::Tensor sigmas = torch::tensor(std::vector<float>{ 1.f, 2.f });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 3, 5 });

	std::vector<GGL::ValueHeadStats> stats =
		GGL::ComputeValueHeadStats(valPreds, targets, presence, sampleWeights, sigmas, playerCounts);

	ASSERT_EQ(stats.size(), (size_t)2);
	ASSERT_TRUE(stats[0].modes.empty());
	ASSERT_TRUE(stats[1].modes.empty());

	// Σ|w|·σ is 4 for head 0 and 6 for head 1.
	ASSERT_TRUE(Near(stats[0].contributionShare, 0.4f));
	ASSERT_TRUE(Near(stats[1].contributionShare, 0.6f));
	ASSERT_TRUE(Near(stats[0].normalizedLoss, 4.f));
	ASSERT_TRUE(Near(stats[1].normalizedLoss, 1.125f));
}

TEST(PerModeCriticStats, ValueHeadStatsConstantTargetsHaveNoExplainedVariance) {
	torch::Tensor targets = torch::full({ 3, 1 }, 5.f);
	torch::Tensor valPreds = torch::tensor(std::vector<float>{ 4.f, 5.f, 7.f }).reshape({ 3, 1 });
	torch::Tensor presence = torch::ones({ 3, 1 });
	torch::Tensor playerCounts = torch::tensor(std::vector<int8_t>{ 4, 4, 4 });

	std::vector<GGL::ValueHeadStats> stats =
		GGL::ComputeValueHeadStats(valPreds, targets, presence, presence, torch::ones({ 1 }), playerCounts);

	ASSERT_EQ(stats[0].modes.size(), (size_t)1);
	ASSERT_FALSE(stats[0].modes[0].explainedVariance.has_value());
	ASSERT_EQ(stats[0].modes[0].targetStd.value(), 0.f);
}

TEST(PerModeCriticStats, ValueHeadStatsEmptyBatch) {
	const int64_t numHeads = 3;
	torch::Tensor empty = torch::zeros({ 0, numHeads });
	torch::Tensor playerCounts = torch::zeros({ 0 }, torch::kInt64);

	std::vector<GGL::ValueHeadStats> stats =
		GGL::ComputeValueHeadStats(empty, empty, empty, empty, torch::ones({ numHeads }), playerCounts);

	ASSERT_EQ(stats.size(), (size_t)numHeads);
	for (const GGL::ValueHeadStats& headStats : stats) {
		ASSERT_TRUE(headStats.modes.empty());
		ASSERT_EQ(headStats.contributionShare, 0.f);
		ASSERT_EQ(headStats.normalizedLoss, 0.f);
	}
}

TEST(PerModeCriticStats, ValueHeadStatsIsDeterministic) {
	const HeadStatsBatch batch = MakeRandomHeadStatsBatch(2000, 7);

	const std::vector<GGL::ValueHeadStats> first = ComputeStats(batch);
	const std::vector<GGL::ValueHeadStats> second = ComputeStats(batch);

	ASSERT_EQ(first.size(), second.size());
	for (size_t headIdx = 0; headIdx < first.size(); headIdx++) {
		ASSERT_EQ(FloatBits(first[headIdx].contributionShare), FloatBits(second[headIdx].contributionShare));
		ASSERT_EQ(FloatBits(first[headIdx].normalizedLoss), FloatBits(second[headIdx].normalizedLoss));

		ASSERT_EQ(first[headIdx].modes.size(), second[headIdx].modes.size());
		for (size_t modeIdx = 0; modeIdx < first[headIdx].modes.size(); modeIdx++) {
			const GGL::HeadModeStats& a = first[headIdx].modes[modeIdx];
			const GGL::HeadModeStats& b = second[headIdx].modes[modeIdx];
			ASSERT_EQ(FloatBits(a.contributionShare), FloatBits(b.contributionShare));
			ASSERT_TRUE(SameOptionalBits(a.explainedVariance, b.explainedVariance));
			ASSERT_TRUE(SameOptionalBits(a.targetStd, b.targetStd));
		}
	}
}
