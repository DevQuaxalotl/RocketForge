#include <gtest/gtest.h>

#include "TestSupport.h"

#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>

#include <ATen/ops/index_select.h>
#include <torch/cuda.h>

TEST(ModelManager, CudaGraphBatchCapacityMatchesRuntimeBuckets)
{
	ASSERT_EQ(GGL::ModelManager::GetCudaGraphBatchCapacity(
		0, GGL::CudaGraphBatchPolicy::PowerOfTwo), (int64_t)0);
	ASSERT_EQ(GGL::ModelManager::GetCudaGraphBatchCapacity(
		1, GGL::CudaGraphBatchPolicy::PowerOfTwo), (int64_t)16);
	ASSERT_EQ(GGL::ModelManager::GetCudaGraphBatchCapacity(
		17, GGL::CudaGraphBatchPolicy::PowerOfTwo), (int64_t)32);
	ASSERT_EQ(GGL::ModelManager::GetCudaGraphBatchCapacity(
		17, GGL::CudaGraphBatchPolicy::Exact), (int64_t)17);
}

TEST(ModelManager, CudaGraphDeterministicPolicyMatchesEager) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(123);

	const int obsSize = 4;
	const int numActions = 5;
	const bool makeCritic = false;
	const bool useHalfPrecision = false;
	const bool noGraph = false;
	const bool useGraph = true;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 16 };
	policyConfig.addLayerNorm = false;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(makeCritic, obsSize, numActions, {}, policyConfig, {}, device, models);

	torch::Tensor obs = torch::randn({ 16, obsSize }, device);
	torch::Tensor actionMasks = torch::ones({ 16, numActions }, torch::TensorOptions().dtype(torch::kUInt8).device(device));

	torch::Tensor eagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		models,
		obs,
		actionMasks,
		1.0f,
		useHalfPrecision,
		0.0f,
		noGraph
	).cpu();

	GGL::ModelManager::ResetCudaGraphStats();
	torch::Tensor graphProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		models,
		obs,
		actionMasks,
		1.0f,
		useHalfPrecision,
		0.0f,
		useGraph
	).cpu();

	torch::Tensor eagerActions;
	torch::Tensor graphActions;
	const bool deterministic = true;
	GGL::ModelManager::InferActionsFromModels(
		models,
		obs,
		actionMasks,
		deterministic,
		1.0f,
		useHalfPrecision,
		&eagerActions,
		nullptr,
		0.0f,
		noGraph
	);
	GGL::ModelManager::InferActionsFromModels(
		models,
		obs,
		actionMasks,
		deterministic,
		1.0f,
		useHalfPrecision,
		&graphActions,
		nullptr,
		0.0f,
		useGraph
	);

	GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();

	ASSERT_TRUE(torch::allclose(eagerProbs, graphProbs, 1e-5, 1e-5));
	ASSERT_TRUE(torch::equal(eagerActions.cpu(), graphActions.cpu()));
	ASSERT_TRUE(stats.captures >= 1);
	ASSERT_TRUE(stats.replays >= 2);
	ASSERT_EQ(stats.captureFailures, (uint64_t)0);

	models.Free();
#endif
}

TEST(ModelManager, CudaGraphRecapturesAfterModelSetReuse) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;

	const int obsSize = 4;
	const int numActions = 5;
	const bool makeCritic = false;
	const bool useHalfPrecision = false;
	const bool noGraph = false;
	const bool useGraph = true;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 16 };
	policyConfig.addLayerNorm = false;

	torch::Tensor obs = torch::randn({ 16, obsSize }, device);
	torch::Tensor actionMasks = torch::ones({ 16, numActions }, torch::TensorOptions().dtype(torch::kUInt8).device(device));

	// Capture a graph for the first set of weights, then free the models.
	torch::manual_seed(123);
	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(makeCritic, obsSize, numActions, {}, policyConfig, {}, device, models);

	torch::Tensor firstEagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		models, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, noGraph).cpu();
	GGL::ModelManager::InferPolicyProbsFromModels(models, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, useGraph);

	models.Free();

	// Rebuild different weights in the same ModelSet object. The graphed result must come
	// from a fresh capture of the new weights, not a stale replay of the freed set.
	torch::manual_seed(456);
	GGL::ModelManager::MakeModels(makeCritic, obsSize, numActions, {}, policyConfig, {}, device, models);

	torch::Tensor secondEagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		models, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, noGraph).cpu();
	torch::Tensor secondGraphProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		models, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, useGraph).cpu();

	ASSERT_TRUE(torch::allclose(secondGraphProbs, secondEagerProbs, 1e-5, 1e-5));
	ASSERT_FALSE(torch::allclose(firstEagerProbs, secondEagerProbs, 1e-5, 1e-5));

	models.Free();
#endif
}

TEST(ModelManager, CudaGraphCacheEvictsFreedModels) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(123);

	const int obsSize = 4;
	const int numActions = 5;
	const bool makeCritic = false;
	const bool useHalfPrecision = false;
	const bool useGraph = true;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 16 };
	policyConfig.addLayerNorm = false;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(makeCritic, obsSize, numActions, {}, policyConfig, {}, device, models);

	torch::Tensor obs = torch::randn({ 16, obsSize }, device);
	torch::Tensor actionMasks = torch::ones({ 16, numActions }, torch::TensorOptions().dtype(torch::kUInt8).device(device));

	const uint64_t entriesBefore = GGL::ModelManager::GetCudaGraphStats().cacheEntries;
	GGL::ModelManager::InferPolicyProbsFromModels(models, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, useGraph);
	ASSERT_TRUE(GGL::ModelManager::GetCudaGraphStats().cacheEntries > entriesBefore);

	models.Free();
	ASSERT_EQ(GGL::ModelManager::GetCudaGraphStats().cacheEntries, entriesBefore);
#endif
}

TEST(ModelManager, CudaGraphTwoModelSetsKeepSeparateGraphs) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;

	const int obsSize = 4;
	const int numActions = 5;
	const bool makeCritic = false;
	const bool useHalfPrecision = false;
	const bool noGraph = false;
	const bool useGraph = true;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 16 };
	policyConfig.addLayerNorm = false;

	torch::Tensor obs = torch::randn({ 16, obsSize }, device);
	torch::Tensor actionMasks = torch::ones({ 16, numActions }, torch::TensorOptions().dtype(torch::kUInt8).device(device));

	// Two live model sets with different weights, mirroring the learner-vs-opponent
	// inference pattern during trainAgainstRecentVersions iterations.
	torch::manual_seed(123);
	GGL::ModelSet learnerModels = {};
	GGL::ModelManager::MakeModels(makeCritic, obsSize, numActions, {}, policyConfig, {}, device, learnerModels);

	torch::manual_seed(456);
	GGL::ModelSet opponentModels = {};
	GGL::ModelManager::MakeModels(makeCritic, obsSize, numActions, {}, policyConfig, {}, device, opponentModels);

	torch::Tensor learnerEagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		learnerModels, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, noGraph).cpu();
	torch::Tensor opponentEagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		opponentModels, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, noGraph).cpu();

	torch::Tensor learnerGraphProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		learnerModels, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, useGraph).cpu();
	torch::Tensor opponentGraphProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		opponentModels, obs, actionMasks, 1.0f, useHalfPrecision, 0.0f, useGraph).cpu();

	ASSERT_TRUE(torch::allclose(learnerGraphProbs, learnerEagerProbs, 1e-5, 1e-5));
	ASSERT_TRUE(torch::allclose(opponentGraphProbs, opponentEagerProbs, 1e-5, 1e-5));
	ASSERT_FALSE(torch::allclose(learnerEagerProbs, opponentEagerProbs, 1e-5, 1e-5));

	learnerModels.Free();
	opponentModels.Free();
#endif
}

TEST(ModelManager, CudaGraphStochasticPolicySamplesOnCuda) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(123);

	const int obsSize = 4;
	const int numActions = 5;
	const bool makeCritic = false;
	const bool deterministic = false;
	const bool useHalfPrecision = false;
	const bool useGraph = true;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 16 };
	policyConfig.addLayerNorm = false;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(makeCritic, obsSize, numActions, {}, policyConfig, {}, device, models);

	torch::Tensor obs = torch::randn({ 32, obsSize }, device);
	torch::Tensor actionMasks = torch::ones({ 32, numActions }, torch::TensorOptions().dtype(torch::kUInt8).device(device));
	actionMasks.index_put_({ torch::indexing::Slice(), 0 }, 0);

	torch::Tensor actions;
	torch::Tensor logProbs;
	GGL::ModelManager::ResetCudaGraphStats();
	GGL::ModelManager::InferActionsFromModels(
		models,
		obs,
		actionMasks,
		deterministic,
		1.0f,
		useHalfPrecision,
		&actions,
		&logProbs,
		0.0f,
		useGraph
	);
	GGL::ModelManager::InferActionsFromModels(
		models,
		obs,
		actionMasks,
		deterministic,
		1.0f,
		useHalfPrecision,
		&actions,
		&logProbs,
		0.0f,
		useGraph
	);
	torch::cuda::synchronize();

	torch::Tensor chosenMask = actionMasks.gather(1, actions.view({ -1, 1 })).flatten();
	GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();

	ASSERT_TRUE(actions.is_cuda());
	ASSERT_TRUE(logProbs.is_cuda());
	ASSERT_TRUE(torch::all(chosenMask.cpu().to(torch::kBool)).item<bool>());
	ASSERT_TRUE(torch::all(torch::isfinite(logProbs.cpu())).item<bool>());
	ASSERT_TRUE(stats.captures >= 1);
	ASSERT_TRUE(stats.replays >= 2);
	ASSERT_EQ(stats.fallbacks, (uint64_t)0);
	ASSERT_EQ(stats.captureFailures, (uint64_t)0);

	models.Free();
#endif
}

TEST(ModelManager, CudaGraphPowerOfTwoBucketsMatchEager) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(123);

	const int obsSize = 4;
	const int numActions = 5;
	const torch::Device device = torch::Device(torch::kCUDA);

	for (bool addLayerNorm : { false, true }) {
		GGL::PartialModelConfig policyConfig = {};
		policyConfig.layerSizes = { 16, 16 };
		policyConfig.addLayerNorm = addLayerNorm;

		GGL::ModelSet models = {};
		GGL::ModelManager::MakeModels(false, obsSize, numActions, {}, policyConfig, {}, device, models);
		GGL::ModelManager::ClearCudaGraphCache();
		GGL::ModelManager::ResetCudaGraphStats();

		// Seventeen and thirty-two rows share the thirty-two-row graph bucket. This exercises both the
		// padded prefix and the full bucket without capturing a second graph.
		for (int64_t batchSize : { 17, 32 }) {
			torch::Tensor obs = torch::randn({ batchSize, obsSize }, device);
			torch::Tensor actionMasks = torch::ones(
				{ batchSize, numActions },
				torch::TensorOptions().dtype(torch::kUInt8).device(device)
			);
			actionMasks.index_put_({ torch::indexing::Slice(), 0 }, 0);

			torch::Tensor eagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
				models, obs, actionMasks, 1.0f, false, 0.0f, false).cpu();
			torch::Tensor graphProbs = GGL::ModelManager::InferPolicyProbsFromModels(
				models, obs, actionMasks, 1.0f, false, 0.0f, true).cpu();

			ASSERT_EQ(graphProbs.size(0), batchSize);
			ASSERT_TRUE(torch::allclose(eagerProbs, graphProbs, 1e-5, 1e-5));
		}

		GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();
		ASSERT_EQ(stats.captures, (uint64_t)1);
		ASSERT_EQ(stats.replays, (uint64_t)2);
		ASSERT_EQ(stats.cacheEntries, (uint64_t)1);
		ASSERT_EQ(stats.fallbacks, (uint64_t)0);
		ASSERT_EQ(stats.captureFailures, (uint64_t)0);

		models.Free();
	}
#endif
}

TEST(ModelManager, CudaGraphExactBatchesCaptureSeparately) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(123);

	const int obsSize = 4;
	const int numActions = 5;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 16, 16 };
	policyConfig.addLayerNorm = true;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(false, obsSize, numActions, {}, policyConfig, {}, device, models);
	GGL::ModelManager::ClearCudaGraphCache();
	GGL::ModelManager::ResetCudaGraphStats();

	for (int64_t batchSize : { 17, 32 }) {
		torch::Tensor obs = torch::randn({ batchSize, obsSize }, device);
		torch::Tensor actionMasks = torch::ones(
			{ batchSize, numActions },
			torch::TensorOptions().dtype(torch::kUInt8).device(device)
		);
		actionMasks.index_put_({ torch::indexing::Slice(), 0 }, 0);

		torch::Tensor eagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
			models, obs, actionMasks, 1.0f, false, 0.0f, false).cpu();
		torch::Tensor graphProbs = GGL::ModelManager::InferPolicyProbsFromModels(
			models, obs, actionMasks, 1.0f, false, 0.0f, true,
			GGL::CudaGraphBatchPolicy::Exact).cpu();

		ASSERT_TRUE(torch::allclose(eagerProbs, graphProbs, 1e-5, 1e-5));
	}

	GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();
	ASSERT_EQ(stats.captures, (uint64_t)2);
	ASSERT_EQ(stats.replays, (uint64_t)2);
	ASSERT_EQ(stats.cacheEntries, (uint64_t)2);
	ASSERT_EQ(stats.fallbacks, (uint64_t)0);
	ASSERT_EQ(stats.captureFailures, (uint64_t)0);

	models.Free();
#endif
}

TEST(ModelManager, CudaGraphPersistentInputsReplayWithoutStagingCopy) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(123);

	const int obsSize = 4;
	const int numActions = 5;
	const int64_t batchSize = 32;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 16, 16 };
	policyConfig.addLayerNorm = true;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(false, obsSize, numActions, {}, policyConfig, {}, device, models);
	GGL::ModelManager::ClearCudaGraphCache();
	GGL::ModelManager::ResetCudaGraphStats();

	torch::Tensor persistentObs = torch::empty({ batchSize, obsSize }, device);
	torch::Tensor persistentMasks = torch::empty(
		{ batchSize, numActions },
		torch::TensorOptions().dtype(torch::kUInt8).device(device));

	for (int replay = 0; replay < 2; replay++) {
		persistentObs.copy_(torch::randn_like(persistentObs));
		persistentMasks.fill_(1);
		persistentMasks.index_put_({ torch::indexing::Slice(), replay }, 0);

		torch::Tensor eagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
			models, persistentObs, persistentMasks, 1.0f, false, 0.0f, false).cpu();
		torch::Tensor graphProbs = GGL::ModelManager::InferPolicyProbsFromModels(
			models, persistentObs, persistentMasks, 1.0f, false, 0.0f, true,
			GGL::CudaGraphBatchPolicy::Exact, true).cpu();

		ASSERT_TRUE(torch::allclose(eagerProbs, graphProbs, 1e-5, 1e-5));
	}

	GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();
	ASSERT_EQ(stats.captures, (uint64_t)1);
	ASSERT_EQ(stats.replays, (uint64_t)2);
	ASSERT_EQ(stats.persistentInputReplays, (uint64_t)2);
	ASSERT_EQ(stats.fallbacks, (uint64_t)0);
	ASSERT_EQ(stats.captureFailures, (uint64_t)0);

	models.Free();
#endif
}

TEST(ModelManager, CudaGraphPersistentBucketAcceptsGatheredSubsets) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(321);

	const int obsSize = 4;
	const int numActions = 5;
	const int64_t sourceRows = 48;
	const int64_t bucketSize = 32;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 16, 16 };
	policyConfig.addLayerNorm = true;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(false, obsSize, numActions, {}, policyConfig, {}, device, models);
	GGL::ModelManager::ClearCudaGraphCache();
	GGL::ModelManager::ResetCudaGraphStats();

	torch::Tensor sourceObs = torch::randn({ sourceRows, obsSize }, device);
	torch::Tensor sourceMasks = torch::ones(
		{ sourceRows, numActions },
		torch::TensorOptions().dtype(torch::kUInt8).device(device));
	torch::Tensor persistentObs = torch::zeros({ bucketSize, obsSize }, device);
	torch::Tensor persistentMasks = torch::ones(
		{ bucketSize, numActions },
		torch::TensorOptions().dtype(torch::kUInt8).device(device));

	for (int replay = 0; replay < 2; replay++) {
		torch::Tensor indices = replay == 0
			? torch::arange(0, 23, torch::TensorOptions().dtype(torch::kInt64).device(device))
			: torch::arange(17, 40, torch::TensorOptions().dtype(torch::kInt64).device(device));
		const int64_t activeRows = indices.size(0);
		torch::Tensor activeObs = persistentObs.narrow(0, 0, activeRows);
		torch::Tensor activeMasks = persistentMasks.narrow(0, 0, activeRows);
		at::index_select_out(activeObs, sourceObs, 0, indices);
		at::index_select_out(activeMasks, sourceMasks, 0, indices);

		torch::Tensor eagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
			models,
			sourceObs.index_select(0, indices),
			sourceMasks.index_select(0, indices),
			1.0f, false, 0.0f, false).cpu();
		torch::Tensor graphProbs = GGL::ModelManager::InferPolicyProbsFromModels(
			models, persistentObs, persistentMasks, 1.0f, false, 0.0f, true,
			GGL::CudaGraphBatchPolicy::Exact, true).narrow(0, 0, activeRows).cpu();

		ASSERT_TRUE(torch::allclose(eagerProbs, graphProbs, 1e-5, 1e-5));
	}

	GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();
	ASSERT_EQ(stats.captures, (uint64_t)1);
	ASSERT_EQ(stats.replays, (uint64_t)2);
	ASSERT_EQ(stats.persistentInputReplays, (uint64_t)2);
	ASSERT_EQ(stats.fallbacks, (uint64_t)0);
	ASSERT_EQ(stats.captureFailures, (uint64_t)0);

	models.Free();
#endif
}

TEST(ModelManager, CudaGraphSmallBatchesShareMinimumBucketAndLargeBatchStaysGraphed) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;

	const int obsSize = 4;
	const int numActions = 5;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 8 };
	policyConfig.addLayerNorm = false;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(false, obsSize, numActions, {}, policyConfig, {}, device, models);
	GGL::ModelManager::ClearCudaGraphCache();
	GGL::ModelManager::ResetCudaGraphStats();

	for (int64_t batchSize : { 1, 8, 16, 2049 }) {
		torch::Tensor obs = torch::randn({ batchSize, obsSize }, device);
		torch::Tensor actionMasks = torch::ones(
			{ batchSize, numActions },
			torch::TensorOptions().dtype(torch::kUInt8).device(device)
		);
		torch::Tensor eagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
			models, obs, actionMasks, 1.0f, false, 0.0f, false).cpu();
		torch::Tensor fallbackProbs = GGL::ModelManager::InferPolicyProbsFromModels(
			models, obs, actionMasks, 1.0f, false, 0.0f, true).cpu();

		ASSERT_TRUE(torch::allclose(eagerProbs, fallbackProbs, 1e-5, 1e-5));
	}

	GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();
	ASSERT_EQ(stats.cacheEntries, (uint64_t)2);
	ASSERT_EQ(stats.captures, (uint64_t)2);
	ASSERT_EQ(stats.replays, (uint64_t)4);
	ASSERT_EQ(stats.fallbacks, (uint64_t)0);

	models.Free();
#endif
}

TEST(ModelManager, CudaGraphVariantLimitFallsBackWithoutGrowingCache) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;
	torch::manual_seed(123);

	const int obsSize = 4;
	const int numActions = 5;
	const int maxVariantsPerPolicy = 8;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 8 };
	policyConfig.addLayerNorm = false;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(false, obsSize, numActions, {}, policyConfig, {}, device, models);
	GGL::ModelManager::ClearCudaGraphCache();
	GGL::ModelManager::ResetCudaGraphStats();

	const int64_t batchSize = 16;
	torch::Tensor obs = torch::randn({ batchSize, obsSize }, device);
	torch::Tensor actionMasks = torch::ones(
		{ batchSize, numActions },
		torch::TensorOptions().dtype(torch::kUInt8).device(device)
	);
	for (int variant = 0; variant < maxVariantsPerPolicy; variant++) {
		const float temperature = 0.5f + 0.1f * variant;
		torch::Tensor probs = GGL::ModelManager::InferPolicyProbsFromModels(
			models, obs, actionMasks, temperature, false, 0.0f, true);
		ASSERT_EQ(probs.size(0), batchSize);
	}

	GGL::CudaGraphStats atLimit = GGL::ModelManager::GetCudaGraphStats();
	ASSERT_EQ(atLimit.captures, (uint64_t)maxVariantsPerPolicy);
	ASSERT_EQ(atLimit.cacheEntries, (uint64_t)maxVariantsPerPolicy);
	ASSERT_EQ(atLimit.variantLimitFallbacks, (uint64_t)0);

	const float overflowTemperature = 0.5f + 0.1f * maxVariantsPerPolicy;
	torch::Tensor eagerProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		models, obs, actionMasks, overflowTemperature, false, 0.0f, false).cpu();
	torch::Tensor fallbackProbs = GGL::ModelManager::InferPolicyProbsFromModels(
		models, obs, actionMasks, overflowTemperature, false, 0.0f, true).cpu();

	GGL::CudaGraphStats afterFallback = GGL::ModelManager::GetCudaGraphStats();
	ASSERT_TRUE(torch::allclose(eagerProbs, fallbackProbs, 1e-5, 1e-5));
	ASSERT_EQ(afterFallback.captures, (uint64_t)maxVariantsPerPolicy);
	ASSERT_EQ(afterFallback.cacheEntries, (uint64_t)maxVariantsPerPolicy);
	ASSERT_EQ(afterFallback.variantLimitFallbacks, (uint64_t)1);
	ASSERT_EQ(afterFallback.fallbacks, (uint64_t)1);

	models.Free();
#endif
}

TEST(ModelManager, CudaGraphCaptureFailureDoesNotRemainCached) {
#ifndef RG_CUDA_SUPPORT
	GTEST_SKIP() << "Built without CUDA support";
#else
	ROCKETFORGE_REQUIRE_CUDA();

	torch::NoGradGuard noGrad;

	const int obsSize = 4;
	const int numActions = 5;
	const torch::Device device = torch::Device(torch::kCUDA);

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 8 };
	policyConfig.addLayerNorm = false;

	GGL::ModelSet models = {};
	GGL::ModelManager::MakeModels(false, obsSize, numActions, {}, policyConfig, {}, device, models);
	GGL::ModelManager::ClearCudaGraphCache();
	GGL::ModelManager::ResetCudaGraphStats();

	torch::Tensor obs = torch::randn({ 16, obsSize }, device);
	// Deliberately use the wrong action count. The graph warmup fails before capture,
	// then the eager fallback reports the same invalid input to the caller.
	torch::Tensor invalidMasks = torch::ones(
		{ 16, numActions - 1 },
		torch::TensorOptions().dtype(torch::kUInt8).device(device)
	);
	bool threw = false;
	try {
		GGL::ModelManager::InferPolicyProbsFromModels(
			models, obs, invalidMasks, 1.0f, false, 0.0f, true);
	} catch (const std::exception&) {
		threw = true;
	}

	GGL::CudaGraphStats stats = GGL::ModelManager::GetCudaGraphStats();
	ASSERT_TRUE(threw);
	ASSERT_EQ(stats.cacheEntries, (uint64_t)0);
	ASSERT_EQ(stats.captureFailures, (uint64_t)1);

	models.Free();
#endif
}
