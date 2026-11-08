#include "DecomposedCritic.h"

#include <torch/nn/modules/linear.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
	torch::nn::LinearImpl& GetOutputLayer(GGL::Model& critic) {
		RG_ASSERT(critic.config.addOutputLayer && critic.seq->size() > 0);

		torch::nn::LinearImpl* outputLayer = critic.seq->ptr(critic.seq->size() - 1)->as<torch::nn::LinearImpl>();
		RG_ASSERT(outputLayer != nullptr);
		return *outputLayer;
	}

	bool SameHeadNames(const std::vector<GGL::PopArtHeadState>& a, const std::vector<GGL::PopArtHeadState>& b) {
		if (a.size() != b.size())
			return false;

		for (size_t i = 0; i < a.size(); i++)
			if (a[i].name != b[i].name)
				return false;

		return true;
	}

	std::vector<torch::Tensor> GetTrunkParameters(GGL::Model& model) {
		std::vector<torch::Tensor> parameters;
		for (size_t moduleIdx = 0; moduleIdx + 1 < model.seq->size(); moduleIdx++)
			for (const torch::Tensor& parameter : model.seq[moduleIdx]->parameters())
				parameters.push_back(parameter);

		return parameters;
	}

	bool SameShapes(const std::vector<torch::Tensor>& a, const std::vector<torch::Tensor>& b) {
		if (a.size() != b.size())
			return false;

		for (size_t i = 0; i < a.size(); i++)
			if (a[i].sizes() != b[i].sizes())
				return false;

		return true;
	}

	std::string DescribeShapes(const std::vector<torch::Tensor>& parameters) {
		std::ostringstream shapes;
		for (const torch::Tensor& parameter : parameters)
			shapes << parameter.sizes() << ' ';

		return shapes.str();
	}

	std::vector<std::string> NamesNotIn(const std::vector<std::string>& names, const std::vector<std::string>& other) {
		std::vector<std::string> missing;
		for (const std::string& name : names)
			if (std::find(other.begin(), other.end(), name) == other.end())
				missing.push_back(name);

		return missing;
	}

	std::string JoinNames(const std::vector<std::string>& names) {
		std::ostringstream joined;
		for (size_t i = 0; i < names.size(); i++)
			joined << (i ? ", " : "") << names[i];

		return joined.str();
	}
}

void GGL::ValidateDecomposedCriticConfig(const LearnerConfig& config) {
	if (!config.ppo.decomposedCritic.enabled)
		return;

	if (config.algorithm == LearnerAlgorithm::PPG)
		RG_ERR_CLOSE("The decomposed critic does not support the PPG learner");

	// Per-head GAE runs in raw reward units, so the policy needs its advantages normalized to a fixed scale.
	if (!config.ppo.normalizeAdvantages)
		RG_ERR_CLOSE("The decomposed critic requires ppo.normalizeAdvantages");

	if (!(config.ppo.decomposedCritic.popArtSigmaFloor > 0.f))
		RG_ERR_CLOSE("The decomposed critic requires popArtSigmaFloor > 0");

	if (!(config.ppo.decomposedCritic.popArtBeta > 0.f && config.ppo.decomposedCritic.popArtBeta <= 1.f))
		RG_ERR_CLOSE("The decomposed critic requires popArtBeta in (0, 1]");
}

torch::Tensor GGL::ApplyPopArtUpdate(
	PopArtStats& stats,
	Model& critic,
	const torch::Tensor& headTargets,
	const torch::Tensor& presence
) {
	const PopArtUpdateResult update = stats.Update(headTargets, presence);

	torch::nn::LinearImpl& outputLayer = GetOutputLayer(critic);
	stats.PreserveOutputs(outputLayer.weight, outputLayer.bias, update);
	critic._seqHalfOutdated = true;

	return stats.Normalize(headTargets);
}

torch::Tensor GGL::ComputeCriticLossWeights(
	const torch::Tensor& presence,
	const torch::Tensor& sampleWeights,
	const torch::Tensor& sigmas,
	CriticHeadLossWeighting weighting
) {
	torch::Tensor rawWeights = presence.to(torch::kFloat32);
	if (weighting == CriticHeadLossWeighting::Contribution)
		rawWeights = rawWeights * sampleWeights.abs() * sigmas.unsqueeze(0);

	const float total = rawWeights.sum().item<float>();
	if (total <= 0.f)
		return torch::zeros_like(rawWeights);

	return rawWeights * ((float)rawWeights.numel() / total);
}

void GGL::SaveValueHeads(nlohmann::json& runningStats, const PopArtStats& stats) {
	std::vector<PopArtHeadState> heads = stats.GetHeads();
	JsonArchive ar = JsonArchive::MakeSaver(runningStats);
	ar.Field("value_heads", heads);
}

GGL::ValueHeadsLoadResult GGL::LoadValueHeads(const nlohmann::json& runningStats, PopArtStats* stats) {
	const bool checkpointHasHeads = runningStats.contains("value_heads");

	if (!stats) {
		if (checkpointHasHeads)
			RG_ERR_CLOSE("Checkpoint has a decomposed critic; enable Agent::decomposedCritic or load a different build");
		return {};
	}

	if (!checkpointHasHeads)
		RG_ERR_CLOSE("Checkpoint has a single-output critic; the decomposed critic needs a new build");

	std::vector<PopArtHeadState> heads = {};
	JsonArchive ar = JsonArchive::MakeLoader(runningStats);
	ar.Field("value_heads", heads);
	if (!ar.GetErrors().empty())
		RG_ERR_CLOSE("Failed to load value_heads: " << ar.GetErrors().front());

	ValueHeadsLoadResult result = {};
	for (const PopArtHeadState& head : heads)
		result.savedHeadNames.push_back(head.name);

	result.layoutChanged = !SameHeadNames(heads, stats->GetHeads());
	if (result.layoutChanged)
		stats->RemapFrom(heads);
	else
		stats->SetHeads(std::move(heads));

	return result;
}

void GGL::RemapCriticHeads(
	Model& critic,
	Model& savedCritic,
	const std::vector<std::string>& savedHeadNames,
	const std::vector<std::string>& headNames
) {
	torch::nn::LinearImpl& outputLayer = GetOutputLayer(critic);
	torch::nn::LinearImpl& savedOutputLayer = GetOutputLayer(savedCritic);
	RG_ASSERT(outputLayer.weight.size(0) == (int64_t)headNames.size());
	RG_ASSERT(savedOutputLayer.weight.size(0) == (int64_t)savedHeadNames.size());

	const std::vector<torch::Tensor> trunk = GetTrunkParameters(critic);
	const std::vector<torch::Tensor> savedTrunk = GetTrunkParameters(savedCritic);
	if (!SameShapes(trunk, savedTrunk))
		RG_ERR_CLOSE(
			"The checkpoint's critic layers differ from this agent's critic, so its value heads can't be remapped."
			<< "\n\tCheckpoint critic: " << DescribeShapes(savedTrunk)
			<< "\n\tThis critic:       " << DescribeShapes(trunk));

	torch::NoGradGuard noGrad;
	for (size_t i = 0; i < trunk.size(); i++)
		trunk[i].copy_(savedTrunk[i]);

	outputLayer.weight.zero_();
	outputLayer.bias.zero_();
	for (size_t headIdx = 0; headIdx < headNames.size(); headIdx++) {
		const auto saved = std::find(savedHeadNames.begin(), savedHeadNames.end(), headNames[headIdx]);
		if (saved == savedHeadNames.end())
			continue;

		const int64_t savedIdx = saved - savedHeadNames.begin();
		outputLayer.weight[(int64_t)headIdx].copy_(savedOutputLayer.weight[savedIdx]);
		outputLayer.bias[(int64_t)headIdx].copy_(savedOutputLayer.bias[savedIdx]);
	}

	critic._seqHalfOutdated = true;
}

std::string GGL::DescribeCriticHeadRemap(const std::vector<std::string>& savedHeadNames, const std::vector<std::string>& headNames) {
	const std::vector<std::string> added = NamesNotIn(headNames, savedHeadNames);
	const std::vector<std::string> removed = NamesNotIn(savedHeadNames, headNames);

	std::ostringstream summary;
	if (!added.empty())
		summary << "+" << added.size() << " added (" << JoinNames(added) << "), ";
	if (!removed.empty())
		summary << "-" << removed.size() << " removed (" << JoinNames(removed) << "), ";
	if (added.empty() && removed.empty())
		summary << "reordered, ";

	summary << "critic optimizer reset";
	return summary.str();
}
