#include <gtest/gtest.h>

#include <GigaLearnCPP/PPO/DecomposedCritic.h>

#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

using namespace GGL;

namespace {
	template <typename Fn>
	bool Throws(Fn&& fn) {
		try {
			fn();
		} catch (const std::runtime_error&) {
			return true;
		}
		return false;
	}

	std::unique_ptr<Model> MakeTestCritic(int numInputs, int numHeads) {
		PartialModelConfig partialConfig = {};
		partialConfig.layerSizes = { 8 };
		ModelConfig config(partialConfig);
		config.numInputs = numInputs;
		config.numOutputs = numHeads;
		return std::make_unique<Model>("critic", config, torch::Device(torch::kCPU));
	}

	LearnerConfig MakeDecomposedConfig() {
		LearnerConfig config = {};
		config.ppo.decomposedCritic.enabled = true;
		config.ppo.normalizeAdvantages = true;
		return config;
	}
}

TEST(DecomposedCritic, ValidateAcceptsSupportedConfig) {
	LearnerConfig off = {};
	off.algorithm = LearnerAlgorithm::PPG;

	ASSERT_FALSE(Throws([&] { ValidateDecomposedCriticConfig(off); }));
	ASSERT_FALSE(Throws([&] { ValidateDecomposedCriticConfig(MakeDecomposedConfig()); }));
}

TEST(DecomposedCritic, ValidateRejectsUnsupportedConfig) {
	struct Case {
		const char* label;
		void (*mutate)(LearnerConfig&);
	};

	const Case cases[] = {
		{ "PPG", [](LearnerConfig& config) { config.algorithm = LearnerAlgorithm::PPG; } },
		{ "zero sigma floor", [](LearnerConfig& config) { config.ppo.decomposedCritic.popArtSigmaFloor = 0.f; } },
		{ "zero beta", [](LearnerConfig& config) { config.ppo.decomposedCritic.popArtBeta = 0.f; } },
		{ "beta above one", [](LearnerConfig& config) { config.ppo.decomposedCritic.popArtBeta = 1.5f; } },
		{ "advantage normalization off", [](LearnerConfig& config) { config.ppo.normalizeAdvantages = false; } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		LearnerConfig config = MakeDecomposedConfig();
		testCase.mutate(config);

		ASSERT_TRUE(Throws([&] { ValidateDecomposedCriticConfig(config); }));
	}
}

TEST(DecomposedCritic, ApplyPopArtUpdatePreservesRawOutputsAndMarksHalfCopyStale) {
	torch::manual_seed(3);
	std::unique_ptr<Model> critic = MakeTestCritic(4, 2);
	PopArtStats stats({ "A", "B" }, 0.3f, 0.01f);
	torch::Tensor presence = torch::ones({ 32, 2 });
	torch::Tensor inputs = torch::randn({ 10, 4 });
	ApplyPopArtUpdate(stats, *critic, torch::randn({ 32, 2 }) * 2.f, presence);

	torch::Tensor before = stats.Denormalize(critic->Forward(inputs, false)).detach();
	critic->_seqHalfOutdated = false;
	torch::Tensor headTargets = torch::randn({ 32, 2 }) * 50.f + 10.f;
	torch::Tensor normalizedTargets = ApplyPopArtUpdate(stats, *critic, headTargets, presence);
	torch::Tensor after = stats.Denormalize(critic->Forward(inputs, false)).detach();

	const float scale = before.abs().max().item<float>() + 1.f;
	ASSERT_TRUE((after - before).abs().max().item<float>() <= 1e-4f * scale);
	ASSERT_TRUE(critic->_seqHalfOutdated);
	ASSERT_TRUE((stats.Denormalize(normalizedTargets) - headTargets).abs().max().item<float>() < 1e-3f);
}

TEST(DecomposedCritic, EqualLossWeightsAverageOverPresentCells) {
	torch::Tensor presence = torch::tensor(std::vector<float>{ 1.f, 0.f, 1.f, 1.f }).reshape({ 2, 2 });
	torch::Tensor sampleWeights = torch::tensor(std::vector<float>{ 5.f, 0.f, 1.f, 2.f }).reshape({ 2, 2 });
	torch::Tensor sigmas = torch::tensor(std::vector<float>{ 1.f, 1.f });

	torch::Tensor weights = ComputeCriticLossWeights(presence, sampleWeights, sigmas, CriticHeadLossWeighting::Equal);
	torch::Tensor squaredErrors = torch::tensor(std::vector<float>{ 1.f, 100.f, 2.f, 3.f }).reshape({ 2, 2 });

	ASSERT_EQ(weights[0][1].item<float>(), 0.f);
	ASSERT_TRUE(std::fabs((weights * squaredErrors).mean().item<float>() - 2.f) < 1e-5f);
}

TEST(DecomposedCritic, ContributionLossWeightsFollowWeightTimesSigma) {
	torch::Tensor presence = torch::ones({ 1, 2 });
	torch::Tensor sampleWeights = torch::tensor(std::vector<float>{ -3.f, 1.f }).reshape({ 1, 2 });
	torch::Tensor sigmas = torch::tensor(std::vector<float>{ 1.f, 2.f });

	torch::Tensor weights = ComputeCriticLossWeights(presence, sampleWeights, sigmas, CriticHeadLossWeighting::Contribution);

	// Raw weights |w|·σ are 3 and 2; scaled so they sum to the cell count (2).
	ASSERT_TRUE(std::fabs(weights[0][0].item<float>() - 1.2f) < 1e-5f);
	ASSERT_TRUE(std::fabs(weights[0][1].item<float>() - 0.8f) < 1e-5f);
}

TEST(DecomposedCritic, ContributionLossWeightsWithZeroTotalAreZero) {
	torch::Tensor weights = ComputeCriticLossWeights(
		torch::ones({ 2, 2 }), torch::zeros({ 2, 2 }), torch::ones({ 2 }), CriticHeadLossWeighting::Contribution);

	ASSERT_FALSE(torch::isnan(weights).any().item<bool>());
	ASSERT_EQ(weights.abs().sum().item<float>(), 0.f);
}

TEST(DecomposedCritic, LoadValueHeadsRejectsSingleOutputMismatch) {
	PopArtStats stats({ "A", "B" }, 0.02f, 0.01f);
	nlohmann::json singleOutput = { { "total_timesteps", 100 } };
	nlohmann::json decomposed = singleOutput;
	SaveValueHeads(decomposed, stats);

	ASSERT_FALSE(Throws([&] { LoadValueHeads(singleOutput, nullptr); }));
	ASSERT_TRUE(Throws([&] { LoadValueHeads(decomposed, nullptr); }));
	ASSERT_TRUE(Throws([&] { LoadValueHeads(singleOutput, &stats); }));
}

TEST(DecomposedCritic, LoadValueHeadsRemapsChangedLayout) {
	PopArtStats saved({ "A", "B" }, 0.02f, 0.01f);
	saved.Update(torch::tensor(std::vector<float>{ 1.f, 4.f, 3.f, 8.f }).reshape({ 2, 2 }), torch::ones({ 2, 2 }));
	nlohmann::json runningStats = { { "total_timesteps", 100 } };
	SaveValueHeads(runningStats, saved);

	PopArtStats sameLayout({ "A", "B" }, 0.02f, 0.01f);
	const ValueHeadsLoadResult unchanged = LoadValueHeads(runningStats, &sameLayout);
	ASSERT_FALSE(unchanged.layoutChanged);
	ASSERT_TRUE(sameLayout.GetHeads()[1].initialized);
	ASSERT_EQ(sameLayout.GetHeads()[1].mean, saved.GetHeads()[1].mean);

	PopArtStats otherLayout({ "B", "C" }, 0.02f, 0.01f);
	const ValueHeadsLoadResult changed = LoadValueHeads(runningStats, &otherLayout);
	ASSERT_TRUE(changed.layoutChanged);
	ASSERT_EQ(changed.savedHeadNames.size(), (size_t)2);
	ASSERT_EQ(changed.savedHeadNames[0], std::string("A"));
	ASSERT_EQ(otherLayout.GetHeads()[0].mean, saved.GetHeads()[1].mean);
	ASSERT_FALSE(otherLayout.GetHeads()[1].initialized);
}
