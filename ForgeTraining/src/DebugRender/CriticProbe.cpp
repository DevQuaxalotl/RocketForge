#include "CriticProbe.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/JsonArchive.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

using namespace std;
using namespace RLGC;
using namespace ForgeDebug;

namespace fs = std::filesystem;

void ForgeDebug::ValueHeadStats::Serialize(GGL::JsonArchive& ar) {
	ar.Field("name", name);
	ar.Field("initialized", initialized);
	ar.Field("mean", mean);
	ar.Field("mean_sq", meanSq);
}

namespace {
	/** The saved fields of GGL::WelfordStat, which is private to GigaLearn. */
	struct SavedReturnStat {
		double mean = 0.0;
		double var = 0.0;
		int64_t count = 0;

		void Serialize(GGL::JsonArchive& ar) {
			ar.Field("mean", mean);
			ar.Field("var", var);
			ar.Field("count", count);
		}
	};
}

namespace RocketForge
{
	optional<CriticCheckpointStats> LoadCriticCheckpointStats(const fs::path& checkpointFolder)
	{
		ifstream file(checkpointFolder / "RUNNING_STATS.json");
		if (!file.is_open())
			return nullopt;

		const bool allowExceptions = false;
		nlohmann::json json = nlohmann::json::parse(file, nullptr, allowExceptions);
		if (json.is_discarded() || !json.is_object())
			return nullopt;

		CriticCheckpointStats stats;
		optional<SavedReturnStat> returnStat;

		GGL::JsonArchive ar = GGL::JsonArchive::MakeLoader(json);
		ar.Field("value_heads", stats.valueHeads);
		ar.Field("return_stat", returnStat);
		if (!ar.GetErrors().empty())
			return nullopt;

		if (returnStat)
			stats.returnStd = WelfordStd(returnStat->var, returnStat->count);

		return stats;
	}

	unique_ptr<CriticProbe> CriticProbe::TryLoad(const CriticProbeLoadConfig& config)
	{
		if (!fs::exists(config.checkpointFolder / "CRITIC.lt")) {
			RG_LOG("Critic display: no CRITIC.lt in " << config.checkpointFolder << ", critic hidden");
			return nullptr;
		}

		optional<CriticCheckpointStats> stats = LoadCriticCheckpointStats(config.checkpointFolder);
		if (!stats) {
			RG_LOG("Critic display: no readable RUNNING_STATS.json in " << config.checkpointFolder << ", critic hidden");
			return nullptr;
		}

		unique_ptr<CriticProbe> probe(new CriticProbe());
		probe->_obsSize = config.agent->GetObsSize();
		probe->_useGPU = config.useGPU;
		probe->_sigmaFloor = config.popArtSigmaFloor;
		probe->_returnStd = stats->returnStd;
		probe->_models = make_unique<GGL::ModelSet>();

		const int numCriticOutputs = stats->valueHeads.empty() ? 1 : (int)stats->valueHeads.size();
		const bool makeCritic = true;
		const bool allowNotExist = false;
		const torch::Device device = config.useGPU ? torch::kCUDA : torch::kCPU;

		// MakeModels always builds the policy too, so its checkpoint file is loaded alongside the critic.
		// A critic whose output count doesn't match the saved value heads fails here as a shape mismatch.
		try {
			GGL::ModelManager::MakeModels(
				makeCritic, probe->_obsSize, config.numActions,
				config.agent->sharedHead, config.agent->policy, config.agent->critic,
				device, *probe->_models, numCriticOutputs);
			probe->_models->Load(config.checkpointFolder, allowNotExist);
		} catch (const exception& e) {
			RG_LOG("Critic display: failed to load the critic from " << config.checkpointFolder << ": " << e.what());
			return nullptr;
		}

		for (const ValueHeadStats& headStats : stats->valueHeads) {
			DecomposedCriticHead head;
			head.stats = headStats;
			probe->_heads.push_back(head);
		}
		probe->RemapHeads(*config.rewards);

		return probe;
	}

	void CriticProbe::RemapHeads(const vector<WeightedReward>& rewards)
	{
		_headRewardIndices.clear();
		for (const DecomposedCriticHead& head : _heads) {
			int rewardIdx = -1;
			for (int i = 0; i < (int)rewards.size(); i++) {
				if (rewards[i].reward->GetName() == head.stats.name) {
					rewardIdx = i;
					break;
				}
			}
			_headRewardIndices.push_back(rewardIdx);
		}
	}

	CriticProbe::~CriticProbe()
	{
		if (_models)
			_models->Free();
	}

	CriticDebugData CriticProbe::Evaluate(const vector<float>& obs, const vector<WeightedReward>& rewards) const
	{
		RG_ASSERT((int)obs.size() == _obsSize);
		RG_NO_GRAD;

		const torch::Device device = _useGPU ? torch::kCUDA : torch::kCPU;
		const bool useHalfPrecision = false;

		torch::Tensor obsTensor = torch::tensor(obs).reshape({ 1, (int64_t)_obsSize }).to(device);
		torch::Tensor outputs = GGL::ModelManager::InferCriticHeads(obsTensor, *_models, useHalfPrecision).flatten();
		const vector<float> outputValues = GGL::TENSOR_TO_VEC<float>(outputs);

		if (_heads.empty())
			return MakeSingleCriticEstimate(outputValues[0], _returnStd);

		vector<DecomposedCriticHead> heads = _heads;
		for (size_t headIdx = 0; headIdx < heads.size(); headIdx++) {
			const int rewardIdx = _headRewardIndices[headIdx];
			if (rewardIdx >= 0 && rewardIdx < (int)rewards.size())
				heads[headIdx].weight = rewards[rewardIdx].weight;
		}

		return CombineDecomposedCritic(outputValues, heads, _sigmaFloor);
	}
}
