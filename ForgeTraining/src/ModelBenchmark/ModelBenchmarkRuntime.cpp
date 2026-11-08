#include "ModelBenchmarkInternal.h"

#include <GigaLearnInfer/ModelManager.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/cuda.h>
#endif

namespace RocketForgeTraining::ModelBenchmarkInternal
{
	namespace
	{
		bool SameVec(const Vec& a, const Vec& b) {
			return a.x == b.x && a.y == b.y && a.z == b.z && a._w == b._w;
		}

		bool SameWheelPairConfig(const WheelPairConfig& a, const WheelPairConfig& b) {
			return a.wheelRadius == b.wheelRadius
				&& a.suspensionRestLength == b.suspensionRestLength
				&& SameVec(a.connectionPointOffset, b.connectionPointOffset);
		}

		bool SameCarConfig(const CarConfig& a, const CarConfig& b) {
			return SameVec(a.hitboxSize, b.hitboxSize)
				&& SameVec(a.hitboxPosOffset, b.hitboxPosOffset)
				&& SameWheelPairConfig(a.frontWheels, b.frontWheels)
				&& SameWheelPairConfig(a.backWheels, b.backWheels)
				&& a.threeWheels == b.threeWheels
				&& a.dodgeDeadzone == b.dodgeDeadzone;
		}

		bool SamePartialModelConfig(const GGL::PartialModelConfig& a, const GGL::PartialModelConfig& b) {
			return a.layerSizes == b.layerSizes
				&& a.activationType == b.activationType
				&& a.optimType == b.optimType
				&& a.addLayerNorm == b.addLayerNorm
				&& a.addOutputLayer == b.addOutputLayer;
		}

		std::string VecSignature(const Vec& vec) {
			std::ostringstream out;
			out << std::setprecision(9) << vec.x << "," << vec.y << "," << vec.z << "," << vec._w;
			return out.str();
		}

		std::string WheelPairConfigSignature(const WheelPairConfig& config) {
			std::ostringstream out;
			out << std::setprecision(9)
				<< config.wheelRadius << "|"
				<< config.suspensionRestLength << "|"
				<< VecSignature(config.connectionPointOffset);
			return out.str();
		}

		std::string CarConfigSignature(const CarConfig& config) {
			std::ostringstream out;
			out << std::setprecision(9)
				<< VecSignature(config.hitboxSize) << "|"
				<< VecSignature(config.hitboxPosOffset) << "|"
				<< WheelPairConfigSignature(config.frontWheels) << "|"
				<< WheelPairConfigSignature(config.backWheels) << "|"
				<< config.threeWheels << "|"
				<< config.dodgeDeadzone;
			return out.str();
		}

		std::string PartialModelConfigSignature(const GGL::PartialModelConfig& config) {
			std::ostringstream out;
			for (int layerSize : config.layerSizes)
				out << layerSize << ",";
			out << "|"
				<< (int)config.activationType << "|"
				<< (int)config.optimType << "|"
				<< config.addLayerNorm << "|"
				<< config.addOutputLayer;
			return out.str();
		}
	}

	void BenchmarkModelSetDeleter::operator()(GGL::ModelSet* modelSet) const noexcept {
		if (modelSet == nullptr)
			return;

		modelSet->Free();
		delete modelSet;
	}

	BenchmarkCudaCleanupGuard::~BenchmarkCudaCleanupGuard() noexcept {
		Cleanup();
	}

	void BenchmarkCudaCleanupGuard::Cleanup() noexcept {
		if (cleaned)
			return;
		cleaned = true;

		// Cleanup must never replace the benchmark result or an exception already
		// being unwound. Keep graph and allocator cleanup independent so one failure
		// cannot prevent the other cleanup attempt.
		try {
			GGL::ModelManager::ClearCudaGraphCache();
		} catch (...) {
		}

#ifdef RG_CUDA_SUPPORT
		try {
			if (torch::cuda::is_available())
				c10::cuda::CUDACachingAllocator::emptyCache();
		} catch (...) {
		}
#endif
	}

	ModelBenchmarkRuntimeSignature RuntimeSignature(const LoadedModel& model) {
		ModelBenchmarkRuntimeSignature signature = {};
		signature.agentName = model.agentName;
		signature.ballPredictionEngineVersion = model.agent->GetBallPredictionEngineVersion();
		signature.tickSkip = model.tickSkip;
		signature.actionDelay = model.actionDelay;
		signature.obsSize = model.obsSize;
		signature.numActions = model.numActions;
		signature.maxTeamSize = model.maxTeamSize;
		signature.obsBuilderType = model.obsBuilderType;
		signature.actionParserType = model.actionParserType;
		signature.carConfigSignature = CarConfigSignature(model.carConfig);
		signature.policyConfigSignature =
			PartialModelConfigSignature(model.agent->sharedHead)
			+ "/"
			+ PartialModelConfigSignature(model.agent->policy);
		return signature;
	}

	std::string SameRuntimeSlotMismatchReason(const std::vector<LoadedModel>& models) {
		if (models.empty())
			return "no models were loaded";

		const LoadedModel& first = models.front();
		for (int i = 1; i < (int)models.size(); i++) {
			const LoadedModel& model = models[i];
			if (model.agent->GetBallPredictionEngineVersion() != first.agent->GetBallPredictionEngineVersion())
				return "selected models use different ball prediction settings";
			if (model.tickSkip != first.tickSkip || model.actionDelay != first.actionDelay)
				return "selected models use different tick skips or action delays";
			if (model.agentName != first.agentName)
				return "selected models use different agents";
			if (model.obsSize != first.obsSize)
				return "selected models use different observation sizes";
			if (model.numActions != first.numActions)
				return "selected models use different action counts";
			if (model.obsBuilderType != first.obsBuilderType)
				return "selected models use different observation builders";
			if (model.actionParserType != first.actionParserType)
				return "selected models use different action parsers";
			if (model.maxTeamSize != first.maxTeamSize)
				return "selected models support different max team sizes";
			if (!SameCarConfig(model.carConfig, first.carConfig))
				return "selected models use different car hitbox configs";
			if (!SamePartialModelConfig(model.agent->sharedHead, first.agent->sharedHead)
				|| !SamePartialModelConfig(model.agent->policy, first.agent->policy)) {
				return "selected models use different policy network configs";
			}
		}

		return {};
	}

	BenchmarkModelSetPtr LoadModelSet(const LoadedModel& model, torch::Device device) {
		BenchmarkModelSetPtr modelSet(new GGL::ModelSet());
		model.agent->MakeInferenceModels(
			model.obsSize,
			model.numActions,
			device,
			*modelSet);
		const bool allowNotExist = false;
		modelSet->Load(model.checkpointPath, allowNotExist);
		return modelSet;
	}

	std::vector<int> SelectedModes(const ModelBenchmarkConfig& config, int maxTeamSize) {
		std::vector<int> modes;
		if (config.run1v1 && maxTeamSize >= 1) modes.push_back(1);
		if (config.run2v2 && maxTeamSize >= 2) modes.push_back(2);
		if (config.run3v3 && maxTeamSize >= 3) modes.push_back(3);
		if (modes.empty())
			throw std::runtime_error("ModelBenchmark: select at least one supported benchmark mode.");
		return modes;
	}

	std::vector<PairKey> BuildEligiblePairModes(
		const std::vector<std::pair<int, int>>& matchups,
		const std::vector<int>& modelMaxTeamSizes,
		const std::vector<int>& selectedModes) {
		std::vector<PairKey> pairModes;
		for (const auto& matchup : matchups) {
			if (matchup.first < 0 || matchup.second < 0
				|| matchup.first >= (int)modelMaxTeamSizes.size()
				|| matchup.second >= (int)modelMaxTeamSizes.size()) {
				throw std::invalid_argument("ModelBenchmark: matchup references an unknown model.");
			}

			const int sharedMaxTeamSize = std::min(
				modelMaxTeamSizes[matchup.first],
				modelMaxTeamSizes[matchup.second]);
			for (int mode : selectedModes) {
				if (mode <= sharedMaxTeamSize) {
					pairModes.push_back({
						std::min(matchup.first, matchup.second),
						std::max(matchup.first, matchup.second),
						mode
					});
				}
			}
		}
		return pairModes;
	}

	ModelBenchmarkEligibilityAnalysis AnalyzeModelBenchmarkEligibility(
		const std::vector<ModelBenchmarkModel>& models,
		const std::vector<int>& modelMaxTeamSizes,
		const std::vector<int>& selectedModes) {
		if (models.size() != modelMaxTeamSizes.size())
			throw std::invalid_argument("ModelBenchmark: model capability count does not match model count.");

		std::vector<int> activeModelIndices(models.size());
		std::iota(activeModelIndices.begin(), activeModelIndices.end(), 0);
		std::vector<PairKey> activePairModes;

		// Removing a zero-game Primary model can also remove the only matchups
		// available to non-primary models, so prune until participation is stable.
		while (!activeModelIndices.empty()) {
			std::vector<ModelBenchmarkModel> activeModels;
			std::vector<int> activeMaxTeamSizes;
			activeModels.reserve(activeModelIndices.size());
			activeMaxTeamSizes.reserve(activeModelIndices.size());
			for (int modelIndex : activeModelIndices) {
				activeModels.push_back(models[modelIndex]);
				activeMaxTeamSizes.push_back(modelMaxTeamSizes[modelIndex]);
			}

			activePairModes = BuildEligiblePairModes(
				BuildModelBenchmarkMatchups(activeModels),
				activeMaxTeamSizes,
				selectedModes);
			std::vector<bool> hasEligibleGame(activeModels.size(), false);
			for (const PairKey& pairMode : activePairModes) {
				hasEligibleGame[pairMode.a] = true;
				hasEligibleGame[pairMode.b] = true;
			}

			std::vector<int> participatingModelIndices;
			for (int activeIndex = 0; activeIndex < (int)activeModelIndices.size(); activeIndex++)
				if (hasEligibleGame[activeIndex])
					participatingModelIndices.push_back(activeModelIndices[activeIndex]);
			if (participatingModelIndices.size() == activeModelIndices.size())
				break;
			activeModelIndices = std::move(participatingModelIndices);
		}

		ModelBenchmarkEligibilityAnalysis analysis = {};
		analysis.includedModelIndices = activeModelIndices;
		std::vector<bool> included(models.size(), false);
		for (int modelIndex : activeModelIndices)
			included[modelIndex] = true;
		for (int modelIndex = 0; modelIndex < (int)models.size(); modelIndex++)
			if (!included[modelIndex])
				analysis.excludedModelIndices.push_back(modelIndex);

		analysis.eligiblePairModeCount = (int)activePairModes.size();
		std::set<int> modes;
		std::set<std::pair<int, int>> matchups;
		for (const PairKey& pairMode : activePairModes) {
			analysis.eligiblePairModes.push_back({activeModelIndices[pairMode.a], activeModelIndices[pairMode.b], pairMode.mode, 0});
			modes.insert(pairMode.mode);
			matchups.insert({ pairMode.a, pairMode.b });
		}
		analysis.eligibleModes.assign(modes.begin(), modes.end());
		analysis.eligibleMatchupCount = (int)matchups.size();
		analysis.comparisonGraphConnected = HasConnectedPairModeGraph(
			(int)activeModelIndices.size(),
			activePairModes);
		return analysis;
	}

	bool HasConnectedPairModeGraph(int numModels, const std::vector<PairKey>& pairModes) {
		if (numModels < 2)
			return false;

		std::vector<std::vector<int>> neighbors(numModels);
		for (const PairKey& pairMode : pairModes) {
			if (pairMode.a < 0 || pairMode.b < 0
				|| pairMode.a >= numModels || pairMode.b >= numModels
				|| pairMode.a == pairMode.b) {
				continue;
			}
			neighbors[pairMode.a].push_back(pairMode.b);
			neighbors[pairMode.b].push_back(pairMode.a);
		}

		std::vector<bool> visited(numModels, false);
		std::vector<int> pending = { 0 };
		visited[0] = true;
		while (!pending.empty()) {
			const int model = pending.back();
			pending.pop_back();
			for (int neighbor : neighbors[model]) {
				if (visited[neighbor])
					continue;
				visited[neighbor] = true;
				pending.push_back(neighbor);
			}
		}
		return std::all_of(visited.begin(), visited.end(), [](bool value) { return value; });
	}

	int CountEligiblePairsForMode(const std::vector<PairKey>& pairModes, int mode) {
		return (int)std::count_if(
			pairModes.begin(),
			pairModes.end(),
			[mode](const PairKey& pairMode) { return pairMode.mode == mode; });
	}

	double OverallOutcomeWeightForPair(
		const std::vector<PairKey>& pairModes,
		int modelA,
		int modelB) {
		std::map<std::pair<int, int>, int> modeCountsByPair;
		for (const PairKey& pairMode : pairModes)
			modeCountsByPair[{ pairMode.a, pairMode.b }]++;

		const int a = std::min(modelA, modelB);
		const int b = std::max(modelA, modelB);
		const auto modeCount = modeCountsByPair.find({ a, b });
		if (modeCount == modeCountsByPair.end())
			throw std::invalid_argument("ModelBenchmark: cannot weight an ineligible model pair.");

		int minimumModeCount = std::numeric_limits<int>::max();
		for (const auto& entry : modeCountsByPair)
			minimumModeCount = std::min(minimumModeCount, entry.second);
		return (double)minimumModeCount / modeCount->second;
	}
}

std::string RocketForgeTraining::ModelBenchmarkStrategyName(ModelBenchmarkStrategy strategy) {
	switch (strategy) {
	case ModelBenchmarkStrategy::SameRuntimeSlot:
		return "SameRuntimeSlot";
	case ModelBenchmarkStrategy::Heterogeneous:
		return "Heterogeneous";
	default:
		return "Unknown";
	}
}

RocketForgeTraining::ModelBenchmarkStrategy RocketForgeTraining::SelectModelBenchmarkStrategy(
	const std::vector<ModelBenchmarkRuntimeSignature>& signatures) {
	if (signatures.empty())
		throw std::invalid_argument("ModelBenchmark: at least one runtime signature is required.");

	const ModelBenchmarkRuntimeSignature& first = signatures.front();
	for (int i = 1; i < (int)signatures.size(); i++) {
		const ModelBenchmarkRuntimeSignature& signature = signatures[i];
		if (signature.tickSkip != first.tickSkip || signature.actionDelay != first.actionDelay) {
			return ModelBenchmarkStrategy::Heterogeneous;
		}
		if (signature.agentName != first.agentName
			|| signature.obsSize != first.obsSize
			|| signature.numActions != first.numActions
			|| signature.obsBuilderType != first.obsBuilderType
			|| signature.actionParserType != first.actionParserType
			|| signature.maxTeamSize != first.maxTeamSize
			|| signature.carConfigSignature != first.carConfigSignature
			|| signature.policyConfigSignature != first.policyConfigSignature
			|| signature.ballPredictionEngineVersion != first.ballPredictionEngineVersion) {
			return ModelBenchmarkStrategy::Heterogeneous;
		}
	}

	return ModelBenchmarkStrategy::SameRuntimeSlot;
}
