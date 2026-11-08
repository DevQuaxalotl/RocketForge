#include <GigaLearnInfer/RecentPolicyEnsemble.h>

#include <GigaLearnInfer/Models.h>

#include <ATen/ops/layer_norm.h>

#include <algorithm>
#include <climits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>

#ifdef RG_CUDA_SUPPORT
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#endif

namespace {
	constexpr int64_t MIN_GRAPH_ROWS_PER_SLOT = 16;
	constexpr int64_t GRAPH_ROWS_PER_SLOT_QUANTUM = 16;
	constexpr size_t MAX_GRAPH_VARIANTS = 5;
	constexpr float LAYER_NORM_EPSILON = 1e-5f;

	bool SameConfig(const GGL::ModelConfig& a, const GGL::ModelConfig& b) {
		return a.numInputs == b.numInputs
			&& a.numOutputs == b.numOutputs
			&& a.layerSizes == b.layerSizes
			&& a.activationType == b.activationType
			&& a.addLayerNorm == b.addLayerNorm
			&& a.addOutputLayer == b.addOutputLayer;
	}

	int64_t GetGraphCapacity(int64_t rows) {
		if (rows <= MIN_GRAPH_ROWS_PER_SLOT)
			return MIN_GRAPH_ROWS_PER_SLOT;
		if (rows > INT64_MAX - (GRAPH_ROWS_PER_SLOT_QUANTUM - 1))
			return rows;
		return ((rows + GRAPH_ROWS_PER_SLOT_QUANTUM - 1)
			/ GRAPH_ROWS_PER_SLOT_QUANTUM) * GRAPH_ROWS_PER_SLOT_QUANTUM;
	}
}

struct GGL::RecentPolicyEnsemble::Impl {
	struct StackedModel {
		std::optional<ModelConfig> config;
		std::vector<torch::Tensor> parameters;

		void Initialize(Model& model, int slotCount) {
			config = model.config;
			parameters.clear();
			for (const torch::Tensor& parameter : model.parameters()) {
				std::vector<int64_t> sizes = { slotCount };
				sizes.insert(sizes.end(), parameter.sizes().begin(), parameter.sizes().end());
				parameters.push_back(torch::empty(sizes, parameter.options()));
			}
		}

		bool IsCompatible(Model& model) const {
			if (!config.has_value() || !SameConfig(config.value(), model.config))
				return false;
			const std::vector<torch::Tensor> source = model.parameters();
			if (source.size() != parameters.size())
				return false;
			for (size_t i = 0; i < source.size(); i++) {
				if (parameters[i].slice(0, 0, 1).squeeze(0).sizes() != source[i].sizes())
					return false;
			}
			return true;
		}

		void CopySlot(int slot, Model& model) {
			const std::vector<torch::Tensor> source = model.parameters();
			for (size_t i = 0; i < source.size(); i++)
				parameters[i][slot].copy_(source[i], true);
		}

		torch::Tensor Activate(torch::Tensor value) const {
			switch (config->activationType) {
			case ModelActivationType::RELU:
				return torch::relu(value);
			case ModelActivationType::LEAKY_RELU:
				return torch::leaky_relu(value);
			case ModelActivationType::SIGMOID:
				return torch::sigmoid(value);
			case ModelActivationType::TANH:
				return torch::tanh(value);
			}
			throw std::runtime_error("RecentPolicyEnsemble encountered an unknown activation");
		}

		torch::Tensor Forward(torch::Tensor value) const {
			const int64_t activeSlots = value.size(0);
			size_t parameterIdx = 0;
			for (size_t layerIdx = 0; layerIdx < config->layerSizes.size(); layerIdx++) {
				torch::Tensor weight = parameters[parameterIdx++].narrow(0, 0, activeSlots);
				torch::Tensor bias = parameters[parameterIdx++].narrow(0, 0, activeSlots);
				value = torch::baddbmm(bias.unsqueeze(1), value, weight.transpose(1, 2));

				if (config->addLayerNorm) {
					torch::Tensor normWeight = parameters[parameterIdx++].narrow(0, 0, activeSlots);
					torch::Tensor normBias = parameters[parameterIdx++].narrow(0, 0, activeSlots);
					value = at::layer_norm(
						value, { value.size(-1) }, std::nullopt, std::nullopt,
						LAYER_NORM_EPSILON, true);
					value = torch::addcmul(
						normBias.unsqueeze(1), value, normWeight.unsqueeze(1));
				}
				value = Activate(value);
			}

			if (config->addOutputLayer) {
				torch::Tensor weight = parameters[parameterIdx++].narrow(0, 0, activeSlots);
				torch::Tensor bias = parameters[parameterIdx++].narrow(0, 0, activeSlots);
				value = torch::baddbmm(bias.unsqueeze(1), value, weight.transpose(1, 2));
			}
			return value;
		}
	};

#ifdef RG_CUDA_SUPPORT
	struct GraphKey {
		int64_t activeSlots = 0;
		int64_t rowsPerSlot = 0;
		float temperature = 1;
		float minActionProbability = 0;

		bool operator<(const GraphKey& other) const {
			return std::tie(activeSlots, rowsPerSlot, temperature, minActionProbability)
				< std::tie(other.activeSlots, other.rowsPerSlot, other.temperature, other.minActionProbability);
		}
	};

	struct GraphEntry {
		torch::Tensor staticObs;
		torch::Tensor staticMasks;
		torch::Tensor staticOutput;
		std::optional<c10::cuda::CUDAStream> captureStream;
		at::cuda::CUDAEvent inputsReady;
		at::cuda::CUDAEvent graphDone;
		at::cuda::CUDAGraph graph;
		uint64_t lastUsed = 0;
		bool captured = false;
	};
#endif

	explicit Impl(int slotCount) : slotCount(slotCount) {
		if (slotCount <= 0)
			throw std::invalid_argument("RecentPolicyEnsemble slot count must be positive");
	}

	int slotCount;
	std::vector<uint8_t> populatedSlots;
	StackedModel sharedHead;
	StackedModel policy;
	bool hasSharedHead = false;
	RecentPolicyEnsembleStats stats = {};
#ifdef RG_CUDA_SUPPORT
	std::map<GraphKey, std::unique_ptr<GraphEntry>> graphs;
	uint64_t graphUseSequence = 0;
#endif

	bool IsInitialized() const {
		return policy.config.has_value();
	}

	bool IsCompatible(ModelSet& models) const {
		if (!IsInitialized() || !models["policy"])
			return false;
		if ((models["shared_head"] != nullptr) != hasSharedHead)
			return false;
		return policy.IsCompatible(*models["policy"])
			&& (!hasSharedHead || sharedHead.IsCompatible(*models["shared_head"]));
	}

	void Initialize(ModelSet& models) {
		if (!models["policy"])
			throw std::invalid_argument("RecentPolicyEnsemble requires a policy model");
		hasSharedHead = models["shared_head"] != nullptr;
		if (hasSharedHead)
			sharedHead.Initialize(*models["shared_head"], slotCount);
		policy.Initialize(*models["policy"], slotCount);
		populatedSlots.assign(slotCount, 0);
	}

	torch::Tensor InferEager(
		torch::Tensor obs,
		torch::Tensor actionMasks,
		float temperature,
		float minActionProbability) {
		constexpr float ACTION_ZERO_PROB = 1e-11f;
		constexpr float ACTION_DISABLED_LOGIT = -1e10f;

		actionMasks = actionMasks.to(torch::kBool);
		if (hasSharedHead)
			obs = sharedHead.Forward(obs);

		torch::Tensor logits = policy.Forward(obs) / temperature;
		logits = torch::nan_to_num(logits, 0.0, 50.0, -50.0);
		torch::Tensor validActionMask = actionMasks.to(logits.scalar_type());
		torch::Tensor result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
		result = result * validActionMask;
		result = result / (result.sum(-1, true) + ACTION_ZERO_PROB);

		if (minActionProbability > 0) {
			torch::Tensor validActionCounts = validActionMask.sum(-1, true);
			const float actionFloor = std::min(minActionProbability, 1.f / policy.config->numOutputs);
			torch::Tensor learnedMass = 1 - actionFloor * validActionCounts;
			result = result * learnedMass + validActionMask * actionFloor;
		}
		return result;
	}

#ifdef RG_CUDA_SUPPORT
	torch::Tensor TryInferGraph(
		torch::Tensor obs,
		torch::Tensor actionMasks,
		float temperature,
		float minActionProbability) {
		if (!obs.is_cuda() || torch::GradMode::is_enabled())
			return {};

		const int64_t actualRows = obs.size(1);
		const int64_t capacity = GetGraphCapacity(actualRows);
		const GraphKey key = { obs.size(0), capacity, temperature, minActionProbability };
		auto graphItr = graphs.find(key);
		if (graphItr == graphs.end()) {
			if (graphs.size() >= MAX_GRAPH_VARIANTS) {
				// Active-slot changes produce disjoint keys. Evict the coldest variant
				// instead of permanently forcing every new topology back to eager mode.
				auto evictionItr = std::min_element(
					graphs.begin(), graphs.end(), [](const auto& a, const auto& b) {
						return a.second->lastUsed < b.second->lastUsed;
					});
				// An ensemble replay may still be running after Run() has returned because
				// the caller stream only waits asynchronously. Retire the graph safely.
				if (evictionItr->second->captured)
					evictionItr->second->graphDone.synchronize();
				graphs.erase(evictionItr);
				stats.graphEvictions++;
			}
			graphItr = graphs.emplace(key, std::make_unique<GraphEntry>()).first;
		}
		GraphEntry& entry = *graphItr->second;
		entry.lastUsed = ++graphUseSequence;

		if (!entry.captureStream)
			entry.captureStream.emplace(c10::cuda::getStreamFromPool(false, obs.device().index()));
		c10::cuda::CUDAStream& stream = *entry.captureStream;
		c10::cuda::CUDAStream callerStream = c10::cuda::getCurrentCUDAStream(obs.device().index());
		entry.inputsReady.record(callerStream);
		entry.inputsReady.block(stream);
		c10::cuda::CUDAStreamGuard streamGuard(stream.unwrap());

		bool captureStarted = false;
		auto endFailedCapture = [&]() {
			if (!captureStarted)
				return;
			try {
				entry.graph.capture_end();
			} catch (...) {
			}
			captureStarted = false;
		};

		try {
			if (!entry.captured) {
				entry.staticObs = torch::zeros({ obs.size(0), capacity, obs.size(2) }, obs.options());
				entry.staticMasks = torch::ones({ obs.size(0), capacity, actionMasks.size(2) }, actionMasks.options());
				entry.staticObs.narrow(1, 0, actualRows).copy_(obs, true);
				entry.staticMasks.narrow(1, 0, actualRows).copy_(actionMasks, true);
				for (int i = 0; i < 3; i++)
					entry.staticOutput = InferEager(
						entry.staticObs, entry.staticMasks, temperature, minActionProbability);

				entry.graph.capture_begin(at::cuda::graph_pool_handle(), cudaStreamCaptureModeThreadLocal);
				captureStarted = true;
				entry.staticOutput = InferEager(
					entry.staticObs, entry.staticMasks, temperature, minActionProbability);
				entry.graph.capture_end();
				captureStarted = false;
				entry.captured = true;
				stats.captures++;
			} else {
				entry.staticObs.narrow(1, 0, actualRows).copy_(obs, true);
				entry.staticMasks.narrow(1, 0, actualRows).copy_(actionMasks, true);
			}

			entry.graph.replay();
			entry.graphDone.record(stream);
			entry.graphDone.block(callerStream);
			stats.replays++;
			stats.graphEntries = graphs.size();
			stats.actualRows += obs.size(0) * actualRows;
			stats.evaluatedRows += obs.size(0) * capacity;
			return entry.staticOutput.narrow(1, 0, actualRows);
		} catch (const c10::OutOfMemoryError&) {
			endFailedCapture();
			graphs.erase(key);
			stats.captureFailures++;
			throw;
		} catch (const std::exception&) {
			endFailedCapture();
			graphs.erase(key);
			stats.captureFailures++;
			return {};
		}
	}
#endif
};

GGL::RecentPolicyEnsemble::RecentPolicyEnsemble(int slotCount) : impl(std::make_unique<Impl>(slotCount)) {
}

GGL::RecentPolicyEnsemble::~RecentPolicyEnsemble() = default;

int GGL::RecentPolicyEnsemble::GetSlotCount() const {
	return impl->slotCount;
}

bool GGL::RecentPolicyEnsemble::IsInitialized() const {
	return impl->IsInitialized();
}

bool GGL::RecentPolicyEnsemble::IsCompatible(ModelSet& models) const {
	return impl->IsCompatible(models);
}

void GGL::RecentPolicyEnsemble::SetSlot(int slot, ModelSet& models) {
	if (slot < 0 || slot >= impl->slotCount)
		throw std::out_of_range("RecentPolicyEnsemble slot is out of range");
	if (!impl->IsInitialized())
		impl->Initialize(models);
	if (!impl->IsCompatible(models))
		throw std::invalid_argument("RecentPolicyEnsemble model architecture does not match existing slots");

	impl->policy.CopySlot(slot, *models["policy"]);
	if (impl->hasSharedHead)
		impl->sharedHead.CopySlot(slot, *models["shared_head"]);
	impl->populatedSlots[slot] = 1;
	impl->stats.weightCopies++;
}

void GGL::RecentPolicyEnsemble::ClearSlot(int slot) {
	if (slot < 0 || slot >= impl->slotCount)
		throw std::out_of_range("RecentPolicyEnsemble slot is out of range");
	if (!impl->populatedSlots.empty())
		impl->populatedSlots[slot] = 0;
}

torch::Tensor GGL::RecentPolicyEnsemble::InferPolicyProbs(
	torch::Tensor obs,
	torch::Tensor actionMasks,
	float temperature,
	float minActionProbability,
	bool useCudaGraph) {
	if (!impl->IsInitialized())
		throw std::runtime_error("RecentPolicyEnsemble has no populated slots");
	if (obs.dim() != 3 || actionMasks.dim() != 3
		|| obs.size(0) != actionMasks.size(0)
		|| obs.size(0) <= 0 || obs.size(0) > impl->slotCount
		|| obs.size(1) != actionMasks.size(1))
		throw std::invalid_argument("RecentPolicyEnsemble inputs have invalid shapes");

#ifdef RG_CUDA_SUPPORT
	if (useCudaGraph) {
		torch::Tensor result = impl->TryInferGraph(
			obs, actionMasks, temperature, minActionProbability);
		if (result.defined())
			return result;
		impl->stats.fallbacks++;
	}
#else
	if (useCudaGraph)
		impl->stats.fallbacks++;
#endif

	impl->stats.actualRows += obs.size(0) * obs.size(1);
	impl->stats.evaluatedRows += obs.size(0) * obs.size(1);
	return impl->InferEager(obs, actionMasks, temperature, minActionProbability);
}

void GGL::RecentPolicyEnsemble::InferActions(
	torch::Tensor obs,
	torch::Tensor actionMasks,
	bool deterministic,
	float temperature,
	torch::Tensor* outActions,
	float minActionProbability,
	bool useCudaGraph) {
	torch::Tensor probs = InferPolicyProbs(
		obs, actionMasks, temperature, minActionProbability, useCudaGraph);
	const int64_t numActions = probs.size(2);
	torch::Tensor flatProbs = probs.view({ -1, numActions });
	torch::Tensor actions = deterministic
		? flatProbs.argmax(1)
		: torch::multinomial(flatProbs, 1, true).flatten();
	if (outActions)
		*outActions = actions.view({ probs.size(0), probs.size(1) });
}

GGL::RecentPolicyEnsembleStats GGL::RecentPolicyEnsemble::GetStats() const {
	RecentPolicyEnsembleStats result = impl->stats;
#ifdef RG_CUDA_SUPPORT
	result.graphEntries = impl->graphs.size();
#endif
	return result;
}

void GGL::RecentPolicyEnsemble::ResetStats() {
	const uint64_t weightCopies = impl->stats.weightCopies;
	impl->stats = {};
	impl->stats.weightCopies = weightCopies;
#ifdef RG_CUDA_SUPPORT
	impl->stats.graphEntries = impl->graphs.size();
#endif
}
