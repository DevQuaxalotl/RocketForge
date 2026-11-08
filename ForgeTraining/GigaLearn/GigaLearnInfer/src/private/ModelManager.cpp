#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>

#ifdef RG_CUDA_SUPPORT
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#endif

namespace {
	struct AtomicCudaGraphStats {
		std::atomic<uint64_t> captures = 0;
		std::atomic<uint64_t> replays = 0;
		std::atomic<uint64_t> persistentInputReplays = 0;
		std::atomic<uint64_t> fallbacks = 0;
		std::atomic<uint64_t> captureFailures = 0;
		std::atomic<uint64_t> variantLimitFallbacks = 0;
		std::atomic<uint64_t> cacheEntries = 0;

		GGL::CudaGraphStats Snapshot() const {
			return {
				captures.load(std::memory_order_relaxed),
				replays.load(std::memory_order_relaxed),
				persistentInputReplays.load(std::memory_order_relaxed),
				fallbacks.load(std::memory_order_relaxed),
				captureFailures.load(std::memory_order_relaxed),
				variantLimitFallbacks.load(std::memory_order_relaxed),
				cacheEntries.load(std::memory_order_relaxed)
			};
		}

		void ResetCounters() {
			captures.store(0, std::memory_order_relaxed);
			replays.store(0, std::memory_order_relaxed);
			persistentInputReplays.store(0, std::memory_order_relaxed);
			fallbacks.store(0, std::memory_order_relaxed);
			captureFailures.store(0, std::memory_order_relaxed);
			variantLimitFallbacks.store(0, std::memory_order_relaxed);
		}
	};

	AtomicCudaGraphStats g_graphStats = {};
	// Power-of-two buckets start at 16 rows. A per-policy variant cap bounds cache
	// growth without forcing large training-inference batches back to eager mode.
	constexpr int64_t MIN_CUDA_GRAPH_BATCH_BUCKET = 16;
	constexpr size_t MAX_CUDA_GRAPH_VARIANTS_PER_POLICY = 8;

	int64_t ResolveCudaGraphBatchCapacity(int64_t batchSize, GGL::CudaGraphBatchPolicy batchPolicy) {
		if (batchSize <= 0)
			return 0;
		if (batchPolicy == GGL::CudaGraphBatchPolicy::Exact)
			return batchSize;

		int64_t bucketSize = MIN_CUDA_GRAPH_BATCH_BUCKET;
		while (bucketSize < batchSize && bucketSize <= INT64_MAX / 2)
			bucketSize *= 2;

		return bucketSize >= batchSize ? bucketSize : batchSize;
	}

	// NOTE: Keyed by the individual Model pointers (stable heap addresses), not the ModelSet address.
	// ModelSet objects move around inside PolicyVersionManager's version vector, so an address there
	// can be reused by a different version's models, which would replay a graph with stale weights.
	struct GraphKey {
		const GGL::Model* sharedHeadModel = nullptr;
		const GGL::Model* policyModel = nullptr;
		int64_t batchBucketSize = 0;
		int obsSize = 0;
		int numActions = 0;
		int deviceIndex = 0;
		bool halfPrec = false;
		bool deterministicActions = false;
		float temperature = 1;
		float minActionProbability = 0;
		uintptr_t persistentObsStorage = 0;
		uintptr_t persistentMaskStorage = 0;

		bool operator<(const GraphKey& other) const {
			return std::tie(
				sharedHeadModel, policyModel, batchBucketSize, obsSize, numActions, deviceIndex,
				halfPrec, temperature, minActionProbability, persistentObsStorage, persistentMaskStorage, deterministicActions)
				< std::tie(
					other.sharedHeadModel,
					other.policyModel,
					other.batchBucketSize,
					other.obsSize,
					other.numActions,
					other.deviceIndex,
					other.halfPrec,
					other.temperature,
					other.minActionProbability,
					other.persistentObsStorage,
					other.persistentMaskStorage,
					other.deterministicActions
				);
		}
	};

	torch::Tensor InferPolicyProbsFromModelsEager(
		GGL::ModelSet& models,
		torch::Tensor obs, torch::Tensor actionMasks,
		float temperature, bool halfPrec,
		float minActionProbability) {

		actionMasks = actionMasks.to(torch::kBool);

		constexpr float ACTION_ZERO_PROB = 1e-11f;
		constexpr float ACTION_DISABLED_LOGIT = -1e10f;

		if (models["shared_head"])
			obs = models["shared_head"]->Forward(obs, halfPrec);

		torch::Tensor logits = models["policy"]->Forward(obs, halfPrec) / temperature;
		logits = torch::nan_to_num(logits, 0.0, 50.0, -50.0);
		torch::Tensor validActionMask = actionMasks.to(logits.scalar_type());

		torch::Tensor result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
		result = result * validActionMask;
		result = result / (result.sum(-1, true) + ACTION_ZERO_PROB);

		if (minActionProbability > 0) {
			torch::Tensor validActionCounts = validActionMask.sum(-1, true);
			float maxPossibleFloor = 1.f / models["policy"]->config.numOutputs;
			float actionFloor = RS_MIN(minActionProbability, maxPossibleFloor);

			torch::Tensor learnedMass = 1 - actionFloor * validActionCounts;
			result = result * learnedMass + validActionMask * actionFloor;
		}

		return result.view({ -1, models["policy"]->config.numOutputs });
	}

#ifdef RG_CUDA_SUPPORT
	struct PolicyProbsGraph {
		torch::Tensor staticObs;
		torch::Tensor staticMasks;
		torch::Tensor staticOutput;
		std::optional<c10::cuda::CUDAStream> captureStream;
		at::cuda::CUDAEvent inputsReady;
		at::cuda::CUDAEvent graphDone;
		at::cuda::CUDAGraph graph;
		bool captured = false;

		torch::Tensor Run(
			GGL::ModelSet& models,
			torch::Tensor obs,
			torch::Tensor actionMasks,
			int64_t batchBucketSize,
			float temperature,
			bool halfPrec,
			float minActionProbability,
			bool inputsHavePersistentStorage,
			bool deterministicActions) {
			if (!captureStream)
				captureStream.emplace(c10::cuda::getStreamFromPool(false, obs.device().index()));
			c10::cuda::CUDAStream& stream = *captureStream;
			c10::cuda::CUDAStream callerStream = c10::cuda::getCurrentCUDAStream(obs.device().index());

			// NOTE: Events preserve caller-stream ordering without forcing a host-side sync.
			inputsReady.record(callerStream);
			inputsReady.block(stream);

			c10::cuda::CUDAStreamGuard streamGuard(stream.unwrap());

			bool captureStarted = false;
			auto EndFailedCapture = [&]() {
				if (!captureStarted)
					return;

				try {
					graph.capture_end();
				} catch (...) {
				}
				captureStarted = false;
			};
			try {
				const int64_t actualBatchSize = obs.size(0);
				if (!captured) {
					if (inputsHavePersistentStorage) {
						staticObs = obs;
						staticMasks = actionMasks;
					} else {
						staticObs = torch::zeros({ batchBucketSize, obs.size(1) }, obs.options());
						staticMasks = torch::ones({ batchBucketSize, actionMasks.size(1) }, actionMasks.options());
					}

					// Models contain row-independent Linear and LayerNorm operations. Padding lets nearby
					// batch sizes reuse one graph without changing any returned player's result.
					if (!inputsHavePersistentStorage) {
						staticObs.narrow(0, 0, actualBatchSize).copy_(obs, true);
						staticMasks.narrow(0, 0, actualBatchSize).copy_(actionMasks, true);
					}

					auto infer = [&]() {
						torch::Tensor probabilities = InferPolicyProbsFromModelsEager(
							models, staticObs, staticMasks, temperature, halfPrec, minActionProbability);
						return deterministicActions ? probabilities.argmax(-1).to(torch::kInt32) : probabilities;
					};

					const int warmupRuns = 3;
					for (int i = 0; i < warmupRuns; i++)
						staticOutput = infer();

					graph.capture_begin(at::cuda::graph_pool_handle(), cudaStreamCaptureModeThreadLocal);
					captureStarted = true;
					staticOutput = infer();
					graph.capture_end();
					captureStarted = false;
					captured = true;
					g_graphStats.captures.fetch_add(1, std::memory_order_relaxed);
				} else if (!inputsHavePersistentStorage) {
					staticObs.narrow(0, 0, actualBatchSize).copy_(obs, true);
					staticMasks.narrow(0, 0, actualBatchSize).copy_(actionMasks, true);
				}

				graph.replay();
				graphDone.record(stream);
				graphDone.block(callerStream);
				g_graphStats.replays.fetch_add(1, std::memory_order_relaxed);
				if (inputsHavePersistentStorage)
					g_graphStats.persistentInputReplays.fetch_add(1, std::memory_order_relaxed);
				return staticOutput.narrow(0, 0, actualBatchSize);
			} catch (const c10::OutOfMemoryError&) {
				EndFailedCapture();
				g_graphStats.captureFailures.fetch_add(1, std::memory_order_relaxed);
				throw;
			} catch (const c10::AcceleratorError& error) {
				EndFailedCapture();
				g_graphStats.captureFailures.fetch_add(1, std::memory_order_relaxed);
				if (error.get_error_code() == cudaErrorMemoryAllocation)
					throw;
				return {};
			} catch (const std::exception&) {
				EndFailedCapture();
				g_graphStats.captureFailures.fetch_add(1, std::memory_order_relaxed);
				return {};
			}
		}
	};

	struct PolicyGraphCache {
		std::mutex mutex;
		std::map<GraphKey, std::shared_ptr<PolicyProbsGraph>> entries;
	};

	// NOTE: Intentionally leaked so Model destructors can safely evict entries during static teardown.
	PolicyGraphCache& GetPolicyGraphCache() {
		static PolicyGraphCache* cache = new PolicyGraphCache();
		return *cache;
	}

	void IncrementCudaGraphFallbacks() {
		g_graphStats.fallbacks.fetch_add(1, std::memory_order_relaxed);
	}

	bool CanUseGraph(torch::Tensor obs, bool halfPrec) {
		return obs.defined()
			&& obs.is_cuda()
			&& obs.dim() == 2
			&& obs.size(0) > 0
			&& !halfPrec
			&& !torch::GradMode::is_enabled();
	}

	torch::Tensor TryInferPolicyProbsGraph(
		GGL::ModelSet& models,
		torch::Tensor obs,
		torch::Tensor actionMasks,
		float temperature,
		bool halfPrec,
		float minActionProbability,
		GGL::CudaGraphBatchPolicy batchPolicy,
		bool inputsHavePersistentStorage,
		size_t maxVariants = MAX_CUDA_GRAPH_VARIANTS_PER_POLICY,
		bool deterministicActions = false) {
		if (!CanUseGraph(obs, halfPrec)
			|| !actionMasks.defined()
			|| !actionMasks.is_cuda()
			|| actionMasks.dim() != 2
			|| actionMasks.size(0) != obs.size(0)) {
			return {};
		}
		if (inputsHavePersistentStorage && batchPolicy != GGL::CudaGraphBatchPolicy::Exact)
			return {};

		GraphKey key = {};
		key.sharedHeadModel = models["shared_head"];
		key.policyModel = models["policy"];
		key.batchBucketSize = ResolveCudaGraphBatchCapacity(obs.size(0), batchPolicy);
		key.obsSize = (int)obs.size(1);
		key.numActions = (int)actionMasks.size(1);
		key.deviceIndex = obs.device().index();
		key.halfPrec = halfPrec;
		key.deterministicActions = deterministicActions;
		key.temperature = temperature;
		key.minActionProbability = minActionProbability;
		if (inputsHavePersistentStorage) {
			key.persistentObsStorage = reinterpret_cast<uintptr_t>(obs.data_ptr());
			key.persistentMaskStorage = reinterpret_cast<uintptr_t>(actionMasks.data_ptr());
		}

		PolicyGraphCache& cache = GetPolicyGraphCache();
		std::shared_ptr<PolicyProbsGraph> graphEntry;
		{
			std::lock_guard<std::mutex> lock(cache.mutex);
			auto existing = cache.entries.find(key);
			if (existing == cache.entries.end()) {
				const size_t policyVariantCount = std::count_if(
					cache.entries.begin(), cache.entries.end(), [&key](const auto& entry) {
						return entry.first.policyModel == key.policyModel;
					});
				if (policyVariantCount >= maxVariants) {
					g_graphStats.variantLimitFallbacks.fetch_add(1, std::memory_order_relaxed);
					return {};
				}

				graphEntry = std::make_shared<PolicyProbsGraph>();
				cache.entries.emplace(key, graphEntry);
			} else {
				graphEntry = existing->second;
			}
			g_graphStats.cacheEntries.store(cache.entries.size(), std::memory_order_relaxed);
		}

		auto eraseGraphEntry = [&]() {
			std::lock_guard<std::mutex> lock(cache.mutex);
			auto existing = cache.entries.find(key);
			if (existing != cache.entries.end() && existing->second == graphEntry)
				cache.entries.erase(existing);
			g_graphStats.cacheEntries.store(cache.entries.size(), std::memory_order_relaxed);
		};
		torch::Tensor result;
		try {
			result = graphEntry->Run(
				models,
				obs,
				actionMasks,
				key.batchBucketSize,
				temperature,
				halfPrec,
				minActionProbability,
				inputsHavePersistentStorage,
				deterministicActions
			);
		} catch (...) {
			eraseGraphEntry();
			throw;
		}
		if (!result.defined())
			eraseGraphEntry();
		return result;
	}
#endif
}

int64_t GGL::ModelManager::GetCudaGraphBatchCapacity(
	int64_t batchSize,
	CudaGraphBatchPolicy cudaGraphBatchPolicy) {
	return ResolveCudaGraphBatchCapacity(batchSize, cudaGraphBatchPolicy);
}

void GGL::ModelManager::MakeModels(
	bool makeCritic,
	int obsSize, int numActions,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device,
	ModelSet& outModels,
	int numCriticOutputs) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = numCriticOutputs;

	if (sharedHeadConfig.IsValid()) {

		ModelConfig fullSharedHeadConfig = sharedHeadConfig;
		fullSharedHeadConfig.numInputs = obsSize;
		fullSharedHeadConfig.numOutputs = 0;

		RG_ASSERT(!sharedHeadConfig.addOutputLayer);

		fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back();
		fullCriticConfig.numInputs = fullSharedHeadConfig.layerSizes.back();

		outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
	}

	outModels.Add(new Model("policy", fullPolicyConfig, device));

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

torch::Tensor GGL::ModelManager::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec,
	float minActionProbability,
	bool useCudaGraph,
	CudaGraphBatchPolicy cudaGraphBatchPolicy,
	bool inputsHavePersistentStorage) {

#ifdef RG_CUDA_SUPPORT
	if (useCudaGraph) {
		torch::Tensor graphResult = TryInferPolicyProbsGraph(
			models, obs, actionMasks, temperature, halfPrec, minActionProbability,
			cudaGraphBatchPolicy, inputsHavePersistentStorage);
		if (graphResult.defined())
			return graphResult;
		IncrementCudaGraphFallbacks();
	}
#else
	if (useCudaGraph)
		g_graphStats.fallbacks.fetch_add(1, std::memory_order_relaxed);
#endif

	return InferPolicyProbsFromModelsEager(models, obs, actionMasks, temperature, halfPrec, minActionProbability);
}

void GGL::ModelManager::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs,
	float minActionProbability,
	bool useCudaGraph,
	CudaGraphBatchPolicy cudaGraphBatchPolicy,
	bool inputsHavePersistentStorage,
	std::optional<at::Generator> actionGenerator) {

	torch::Tensor probs = InferPolicyProbsFromModels(
		models, obs, actionMasks, temperature, halfPrec,
		minActionProbability, useCudaGraph, cudaGraphBatchPolicy, inputsHavePersistentStorage);

	if (deterministic) {
		torch::Tensor action = probs.argmax(1);
		if (outActions)
			*outActions = action.flatten();
	}
	else {
		torch::Tensor action = torch::multinomial(probs, 1, true, actionGenerator);
		if (outActions)
			*outActions = action.flatten();

		if (outLogProbs)
			*outLogProbs = torch::log(probs).gather(-1, action).flatten();
	}
}

void GGL::ModelManager::InferActionsFromModelsValidRows(
	ModelSet& models, torch::Tensor obs, torch::Tensor actionMasks,
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs,
	float minActionProbability, bool useCudaGraph,
	CudaGraphBatchPolicy batchPolicy, bool persistentStorage, int64_t validRows,
	std::optional<at::Generator> actionGenerator) {
	TORCH_CHECK(validRows >= 0 && validRows <= obs.size(0), "Invalid inference row count");

#ifdef RG_CUDA_SUPPORT
	// The graph also captures the argmax, so only compact action indices are read back.
	// Evaluation batches shrink as games finish, so this path allows more cached shapes.
	if (deterministic && useCudaGraph && outActions && !outLogProbs) {
		const size_t maxShrinkingBatchVariants = 16;
		const bool captureArgmax = true;
		torch::Tensor actions = TryInferPolicyProbsGraph(
			models, obs, actionMasks, temperature, halfPrec, minActionProbability,
			batchPolicy, persistentStorage, maxShrinkingBatchVariants, captureArgmax);
		if (actions.defined()) {
			*outActions = actions.narrow(0, 0, validRows);
			return;
		}

		IncrementCudaGraphFallbacks();
		useCudaGraph = false;
	}
#endif

	// Restrict actual decisions to valid rows, including on the eager fallback.
	InferActionsFromModels(
		models, obs.narrow(0, 0, validRows), actionMasks.narrow(0, 0, validRows),
		deterministic, temperature, halfPrec, outActions, outLogProbs,
		minActionProbability, useCudaGraph, batchPolicy, persistentStorage && validRows == obs.size(0), actionGenerator);
}

torch::Tensor GGL::ModelManager::InferCritic(torch::Tensor obs, ModelSet& models, bool useHalfPrecision) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, useHalfPrecision);

	return models["critic"]->Forward(obs, useHalfPrecision).flatten();
}

torch::Tensor GGL::ModelManager::InferCriticHeads(torch::Tensor obs, ModelSet& models, bool useHalfPrecision) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, useHalfPrecision);

	return models["critic"]->Forward(obs, useHalfPrecision);
}

void GGL::ModelManager::ReleaseCudaGraphs(const Model* model) {
#ifdef RG_CUDA_SUPPORT
	PolicyGraphCache& cache = GetPolicyGraphCache();
	std::lock_guard<std::mutex> lock(cache.mutex);

	std::erase_if(cache.entries, [model](const auto& entry) {
		return entry.first.sharedHeadModel == model || entry.first.policyModel == model;
	});
	g_graphStats.cacheEntries.store(cache.entries.size(), std::memory_order_relaxed);
#endif
}

GGL::CudaGraphStats GGL::ModelManager::GetCudaGraphStats() {
	return g_graphStats.Snapshot();
}

void GGL::ModelManager::ResetCudaGraphStats() {
	g_graphStats.ResetCounters();
}

void GGL::ModelManager::ClearCudaGraphCache() {
#ifdef RG_CUDA_SUPPORT
	PolicyGraphCache& cache = GetPolicyGraphCache();
	std::lock_guard<std::mutex> lock(cache.mutex);
	cache.entries.clear();
#endif
	g_graphStats.cacheEntries.store(0, std::memory_order_relaxed);
}