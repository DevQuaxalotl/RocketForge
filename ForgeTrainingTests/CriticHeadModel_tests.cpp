#include <gtest/gtest.h>

#include <GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>

#include <vector>

using namespace GGL;

namespace {
	void MakeTestModels(int numCriticOutputs, ModelSet& outModels) {
		PartialModelConfig policyConfig = {};
		policyConfig.layerSizes = { 8 };
		PartialModelConfig criticConfig = {};
		criticConfig.layerSizes = { 8 };

		const bool makeCritic = true;
		ModelManager::MakeModels(
			makeCritic, 5, 3, {}, policyConfig, criticConfig, torch::Device(torch::kCPU), outModels, numCriticOutputs);
	}
}

TEST(CriticHeadModel, InferCriticHeadsReturnsOneColumnPerHead) {
	ModelSet models = {};
	MakeTestModels(4, models);

	torch::Tensor values = ModelManager::InferCriticHeads(torch::randn({ 6, 5 }), models, false);

	ASSERT_EQ(values.size(0), (int64_t)6);
	ASSERT_EQ(values.size(1), (int64_t)4);
	models.Free();
}

TEST(CriticHeadModel, SingleOutputCriticStillFlattens) {
	ModelSet models = {};
	MakeTestModels(1, models);

	torch::Tensor values = ModelManager::InferCritic(torch::randn({ 6, 5 }), models, false);

	ASSERT_EQ(values.dim(), (int64_t)1);
	ASSERT_EQ(values.size(0), (int64_t)6);
	models.Free();
}

TEST(CriticHeadModel, ExperienceBatchesSkipUndefinedCriticLossWeights) {
	ExperienceBuffer experience(1, torch::Device(torch::kCPU));
	experience.data.states = torch::randn({ 8, 5 });
	experience.data.actions = torch::zeros({ 8 }, torch::kInt64);
	experience.data.logProbs = torch::zeros({ 8 });
	experience.data.targetValues = torch::zeros({ 8 });
	experience.data.actionMasks = torch::ones({ 8, 3 }, torch::kUInt8);
	experience.data.advantages = torch::zeros({ 8 });

	std::vector<ExperienceTensors> batches = experience.GetAllBatchesShuffled(4, false);

	ASSERT_EQ(batches.size(), (size_t)2);
	ASSERT_FALSE(batches[0].criticLossWeights.defined());

	experience.data.targetValues = torch::zeros({ 8, 2 });
	experience.data.criticLossWeights = torch::ones({ 8, 2 });
	batches = experience.GetAllBatchesShuffled(4, false);

	ASSERT_EQ(batches[0].criticLossWeights.size(0), (int64_t)4);
	ASSERT_EQ(batches[0].criticLossWeights.size(1), (int64_t)2);
}
