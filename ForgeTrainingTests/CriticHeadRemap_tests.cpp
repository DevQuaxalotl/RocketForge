#include <gtest/gtest.h>

#include "TestSupport.h"

#include <GigaLearnCPP/PPO/DecomposedCritic.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
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

	std::unique_ptr<Model> MakeTestCritic(int numHeads, int hiddenSize = 8) {
		PartialModelConfig partialConfig = {};
		partialConfig.layerSizes = { hiddenSize };
		ModelConfig config(partialConfig);
		config.numInputs = 4;
		config.numOutputs = numHeads;
		return std::make_unique<Model>("critic", config, torch::Device(torch::kCPU));
	}

	PPOLearnerConfig MakeTinyPPOConfig() {
		PPOLearnerConfig config = {};
		config.tsPerItr = 8;
		config.batchSize = 8;
		config.microBatchSize = 4;
		config.epochs = 1;
		config.policyLR = 1e-3f;
		config.criticLR = 1e-3f;

		config.sharedHead.layerSizes = { 8 };
		config.sharedHead.addOutputLayer = false;
		config.policy.layerSizes = { 8 };
		config.critic.layerSizes = { 8 };
		return config;
	}
}

TEST(CriticHeadRemap, CopiesTrunkAndRowsByName) {
	torch::manual_seed(5);
	std::unique_ptr<Model> saved = MakeTestCritic(2);
	std::unique_ptr<Model> critic = MakeTestCritic(3);
	critic->_seqHalfOutdated = false;

	RemapCriticHeads(*critic, *saved, { "A", "B" }, { "B", "C", "A" });

	torch::NoGradGuard noGrad;
	torch::Tensor inputs = torch::randn({ 6, 4 });
	torch::Tensor savedValues = saved->Forward(inputs, false);
	torch::Tensor values = critic->Forward(inputs, false);

	// Matching outputs on random inputs means both the trunk and each kept row were copied.
	ASSERT_TRUE(torch::allclose(values.select(1, 0), savedValues.select(1, 1)));
	ASSERT_TRUE(torch::allclose(values.select(1, 2), savedValues.select(1, 0)));
	ASSERT_EQ(values.select(1, 1).abs().max().item<float>(), 0.f);
	ASSERT_TRUE(critic->_seqHalfOutdated);
}

TEST(CriticHeadRemap, TrunkShapeMismatchIsFatal) {
	std::unique_ptr<Model> saved = MakeTestCritic(2, 16);
	std::unique_ptr<Model> critic = MakeTestCritic(2, 8);

	ASSERT_TRUE(Throws([&] { RemapCriticHeads(*critic, *saved, { "A", "B" }, { "A", "B" }); }));
}

TEST(CriticHeadRemap, DescribeCriticHeadRemap) {
	struct Case {
		const char* label;
		std::vector<std::string> savedHeads;
		std::vector<std::string> currentHeads;
		const char* expected;
	};

	const Case cases[] = {
		{ "added and removed", { "A", "B", "D" }, { "C", "A", "E" }, "+2 added (C, E), -2 removed (B, D), critic optimizer reset" },
		{ "added only", { "A" }, { "A", "B" }, "+1 added (B), critic optimizer reset" },
		{ "reordered only", { "A", "B" }, { "B", "A" }, "reordered, critic optimizer reset" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(DescribeCriticHeadRemap(testCase.savedHeads, testCase.currentHeads), std::string(testCase.expected));
	}
}

TEST(CriticHeadRemap, LearnerLoadsPolicyAndRemapsCriticWithoutItsOptimizer) {
	const std::filesystem::path folder = RocketForgeTests::TestTempDir();

	torch::manual_seed(11);
	PPOLearner saved(4, 3, MakeTinyPPOConfig(), torch::kCPU, true, 2);
	saved.SaveTo(folder);

	// The saved critic optimizer belongs to the old output rows; loading it would throw on this content.
	{
		std::ofstream criticOptimizer(folder / "CRITIC_OPTIM.lt", std::ios::binary | std::ios::trunc);
		criticOptimizer << "not an optimizer";
	}

	PPOLearner loaded(4, 3, MakeTinyPPOConfig(), torch::kCPU, true, 3);
	loaded.LoadFromRemappingCriticHeads(folder, { "A", "B" }, { "B", "C", "A" });

	ASSERT_TRUE(torch::equal(loaded.models["policy"]->CopyParams(), saved.models["policy"]->CopyParams()));
	ASSERT_TRUE(torch::equal(loaded.models["shared_head"]->CopyParams(), saved.models["shared_head"]->CopyParams()));

	torch::NoGradGuard noGrad;
	torch::Tensor features = torch::randn({ 5, saved.models["critic"]->config.numInputs });
	torch::Tensor savedValues = saved.models["critic"]->Forward(features, false);
	torch::Tensor values = loaded.models["critic"]->Forward(features, false);
	ASSERT_TRUE(torch::allclose(values.select(1, 0), savedValues.select(1, 1)));
	ASSERT_TRUE(torch::allclose(values.select(1, 2), savedValues.select(1, 0)));
	ASSERT_EQ(values.select(1, 1).abs().max().item<float>(), 0.f);

	std::filesystem::remove_all(folder);
}
