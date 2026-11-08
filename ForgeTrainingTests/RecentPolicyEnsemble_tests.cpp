#include <gtest/gtest.h>

#include "TestSupport.h"

#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>
#include <GigaLearnInfer/RecentPolicyEnsemble.h>

#include <torch/cuda.h>

namespace {
	GGL::ModelSet MakePolicy(
		const torch::Device& device,
		int obsSize,
		int numActions,
		float parameterOffset,
		bool sharedHead)
	{
		GGL::PartialModelConfig sharedConfig = {};
		if (sharedHead) {
			sharedConfig.layerSizes = { 7 };
			sharedConfig.addOutputLayer = false;
			sharedConfig.addLayerNorm = true;
		}

		GGL::PartialModelConfig policyConfig = {};
		policyConfig.layerSizes = { 11, 9 };
		policyConfig.addLayerNorm = true;
		policyConfig.activationType = GGL::ModelActivationType::LEAKY_RELU;

		GGL::ModelSet models = {};
		GGL::ModelManager::MakeModels(
			false, obsSize, numActions, sharedConfig, policyConfig, {}, device, models);
		for (torch::Tensor& parameter : models["policy"]->parameters())
			parameter.add_(parameterOffset);
		if (models["shared_head"]) {
			for (torch::Tensor& parameter : models["shared_head"]->parameters())
				parameter.add_(parameterOffset * 0.5f);
		}
		return models;
	}
}

TEST(RecentPolicyEnsemble, MatchesIndependentPoliciesOnCpu)
{
	torch::NoGradGuard noGrad;
	torch::manual_seed(42);
	const torch::Device device(torch::kCPU);
	const int slots = 4;
	const int rows = 6;
	const int obsSize = 5;
	const int numActions = 8;

	std::vector<GGL::ModelSet> models;
	GGL::RecentPolicyEnsemble ensemble(slots);
	for (int slot = 0; slot < slots; slot++) {
		models.push_back(MakePolicy(device, obsSize, numActions, slot * 0.01f, true));
		ensemble.SetSlot(slot, models.back());
	}

	torch::Tensor obs = torch::randn({ slots, rows, obsSize }, device);
	torch::Tensor masks = torch::rand({ slots, rows, numActions }, device).gt(0.25).to(torch::kUInt8);
	masks.select(2, 0).fill_(1);
	torch::Tensor ensembleProbs = ensemble.InferPolicyProbs(obs, masks, 0.9f, 0.001f, false);

	for (int slot = 0; slot < slots; slot++) {
		torch::Tensor expected = GGL::ModelManager::InferPolicyProbsFromModels(
			models[slot], obs[slot], masks[slot], 0.9f, false, 0.001f, false);
		ASSERT_TRUE(torch::allclose(ensembleProbs[slot], expected, 2e-5, 2e-5));
	}

	// The fourth weight slot stays allocated while the normal non-draining graph
	// executes only the three slots that currently have arena rows.
	torch::Tensor compactProbs = ensemble.InferPolicyProbs(
		obs.narrow(0, 0, 3), masks.narrow(0, 0, 3), 0.9f, 0.001f, false);
	ASSERT_TRUE(torch::allclose(compactProbs, ensembleProbs.narrow(0, 0, 3), 2e-5, 2e-5));

	for (GGL::ModelSet& modelSet : models)
		modelSet.Free();
}

TEST(RecentPolicyEnsemble, CudaGraphReusesCaptureAfterWeightRefresh)
{
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(17);
	const torch::Device device(torch::kCUDA);
	const int slots = 4;
	const int rows = 13;
	const int obsSize = 5;
	const int numActions = 8;

	std::vector<GGL::ModelSet> models;
	GGL::RecentPolicyEnsemble ensemble(slots);
	for (int slot = 0; slot < slots; slot++) {
		models.push_back(MakePolicy(device, obsSize, numActions, slot * 0.01f, true));
		ensemble.SetSlot(slot, models.back());
	}

	torch::Tensor obs = torch::randn({ slots, rows, obsSize }, device);
	torch::Tensor masks = torch::ones(
		{ slots, rows, numActions }, torch::TensorOptions().dtype(torch::kUInt8).device(device));
	torch::Tensor first = ensemble.InferPolicyProbs(obs, masks, 1.0f, 0.0f, true).cpu();
	torch::Tensor eagerFirst = ensemble.InferPolicyProbs(obs, masks, 1.0f, 0.0f, false).cpu();
	ASSERT_TRUE(torch::allclose(first, eagerFirst, 2e-5, 2e-5));

	GGL::ModelSet replacement = MakePolicy(device, obsSize, numActions, 0.25f, true);
	ensemble.SetSlot(2, replacement);
	torch::Tensor refreshed = ensemble.InferPolicyProbs(obs, masks, 1.0f, 0.0f, true).cpu();
	torch::Tensor expected = GGL::ModelManager::InferPolicyProbsFromModels(
		replacement, obs[2], masks[2], 1.0f, false, 0.0f, false).cpu();
	ASSERT_TRUE(torch::allclose(refreshed[2], expected, 2e-5, 2e-5));
	ASSERT_FALSE(torch::allclose(first[2], refreshed[2], 1e-5, 1e-5));

	const GGL::RecentPolicyEnsembleStats stats = ensemble.GetStats();
	ASSERT_EQ(stats.captures, (uint64_t)1);
	ASSERT_TRUE(stats.replays >= 2);
	ASSERT_EQ(stats.captureFailures, (uint64_t)0);
	ASSERT_EQ(stats.graphEntries, (uint64_t)1);

	replacement.Free();
	for (GGL::ModelSet& modelSet : models)
		modelSet.Free();
#endif
}

TEST(RecentPolicyEnsemble, CudaGraphEvictsColdVariantsAfterActiveSlotChange)
{
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(23);
	const torch::Device device(torch::kCUDA);
	const int slots = 3;
	const int obsSize = 5;
	const int numActions = 8;

	std::vector<GGL::ModelSet> models;
	GGL::RecentPolicyEnsemble ensemble(slots);
	for (int slot = 0; slot < slots; slot++) {
		models.push_back(MakePolicy(device, obsSize, numActions, slot * 0.01f, true));
		ensemble.SetSlot(slot, models.back());
	}

	// Fill every cache entry with one-slot row buckets, matching the topology
	// growth that previously made two- and three-version training permanently eager.
	for (int rows : { 1, 17, 33, 65, 129 }) {
		torch::Tensor obs = torch::randn({ 1, rows, obsSize }, device);
		torch::Tensor masks = torch::ones(
			{ 1, rows, numActions },
			torch::TensorOptions().dtype(torch::kUInt8).device(device));
		torch::Tensor graphProbs = ensemble.InferPolicyProbs(obs, masks, 1.0f, 0.0f, true).cpu();
		torch::Tensor eagerProbs = ensemble.InferPolicyProbs(obs, masks, 1.0f, 0.0f, false).cpu();
		ASSERT_TRUE(torch::allclose(graphProbs, eagerProbs, 2e-5, 2e-5));
	}

	torch::Tensor twoSlotObs = torch::randn({ 2, 13, obsSize }, device);
	torch::Tensor twoSlotMasks = torch::ones(
		{ 2, 13, numActions },
		torch::TensorOptions().dtype(torch::kUInt8).device(device));
	torch::Tensor graphProbs = ensemble.InferPolicyProbs(
		twoSlotObs, twoSlotMasks, 1.0f, 0.0f, true).cpu();
	torch::Tensor eagerProbs = ensemble.InferPolicyProbs(
		twoSlotObs, twoSlotMasks, 1.0f, 0.0f, false).cpu();
	ASSERT_TRUE(torch::allclose(graphProbs, eagerProbs, 2e-5, 2e-5));

	const GGL::RecentPolicyEnsembleStats stats = ensemble.GetStats();
	ASSERT_EQ(stats.captures, (uint64_t)6);
	ASSERT_EQ(stats.replays, (uint64_t)6);
	ASSERT_EQ(stats.fallbacks, (uint64_t)0);
	ASSERT_EQ(stats.captureFailures, (uint64_t)0);
	ASSERT_EQ(stats.graphEntries, (uint64_t)5);
	ASSERT_EQ(stats.graphEvictions, (uint64_t)1);

	for (GGL::ModelSet& modelSet : models)
		modelSet.Free();
#endif
}
