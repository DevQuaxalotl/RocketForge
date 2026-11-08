#include "PPGLearner.h"

#include <GigaLearnCPP/Util/AvgTracker.h>
#include <GigaLearnInfer/ModelManager.h>

#include <torch/nn/modules/loss.h>
#include <torch/nn/utils/clip_grad.h>

using namespace torch;

namespace {
	constexpr float ACTION_ZERO_PROB = 1e-11f;

	torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
		auto entropy = -((probs + ACTION_ZERO_PROB).log() * probs).sum(-1);

		if (maskEntropy) {
			entropy /= actionMasks.to(torch::kFloat32).sum(-1).log();
		} else {
			entropy /= logf(actionMasks.size(-1));
		}

		return entropy.mean();
	}
}

void GGL::PPGLearner::AuxBuffer::Clear() {
	states.clear();
	actionMasks.clear();
	targetValues.clear();
	size = 0;
}

GGL::PPGLearner::PPGLearner(
	int obsSize, int numActions,
	PPOLearnerConfig ppoConfig,
	PPGLearnerConfig ppgConfig,
	torch::Device device
) : PPOLearner(obsSize, numActions, ppoConfig, device, false), ppgConfig(ppgConfig) {
	if (this->ppgConfig.auxPhaseInterval <= 0)
		RG_ERR_CLOSE("PPGLearner: auxPhaseInterval must be greater than zero");
	if (this->ppgConfig.policyEpochs <= 0)
		RG_ERR_CLOSE("PPGLearner: policyEpochs must be greater than zero");
	if (this->ppgConfig.criticEpochs <= 0)
		RG_ERR_CLOSE("PPGLearner: criticEpochs must be greater than zero");
	if (this->ppgConfig.auxEpochs < 0)
		RG_ERR_CLOSE("PPGLearner: auxEpochs cannot be negative");
	if (this->ppgConfig.auxBatchSize < 0)
		RG_ERR_CLOSE("PPGLearner: auxBatchSize cannot be negative");
	if (this->ppgConfig.auxGradClip <= 0)
		RG_ERR_CLOSE("PPGLearner: auxGradClip must be greater than zero");
	if (
		this->ppgConfig.auxValueLossScale < 0 ||
		this->ppgConfig.auxPolicyCloneLossScale < 0 ||
		this->ppgConfig.auxTrueCriticLossScale < 0
		)
		RG_ERR_CLOSE("PPGLearner: auxiliary loss scales cannot be negative");

	MakePPGModels(obsSize, numActions);
	SetLearningRates(config.policyLR, config.criticLR);
	LogModelParameterCounts();
	LoadGuidingPolicyIfEnabled(obsSize, numActions);
	RG_LOG("PPGLearner: full dual-network PPG enabled");
}

void GGL::PPGLearner::MakePPGModels(int obsSize, int numActions) {
	ModelConfig fullPolicyConfig = config.policy;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = config.critic;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = 1;

	models.Add(new Model("policy", fullPolicyConfig, device));
	models.Add(new Model("critic", fullCriticConfig, device));

	ModelConfig auxConfig = PartialModelConfig{};
	auxConfig.numInputs = models["policy"]->GetFeatureSize();
	auxConfig.numOutputs = 1;
	auxConfig.optimType = config.critic.optimType;
	auxConfig.activationType = config.critic.activationType;
	auxConfig.addLayerNorm = false;
	auxConfig.addOutputLayer = true;
	models.Add(new Model("aux_critic", auxConfig, device));

	optimizers.Attach(models);
}

void GGL::PPGLearner::LoadGuidingPolicyIfEnabled(int obsSize, int numActions) {
	if (!config.useGuidingPolicy)
		return;

	RG_LOG("Guiding policy enabled, loading from " << config.guidingPolicyPath << "...");
	ModelManager::MakeModels(false, obsSize, numActions, {}, config.policy, {}, device, guidingPolicyModels);
	guidingPolicyModels.Load(config.guidingPolicyPath, false);
}

void GGL::PPGLearner::LogModelParameterCounts() const {
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : const_cast<ModelSet&>(models)) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));
}

torch::Tensor GGL::PPGLearner::InferTrueCritic(torch::Tensor obs) {
	return models["critic"]->Forward(obs, false).flatten();
}

torch::Tensor GGL::PPGLearner::InferAuxValue(torch::Tensor obs) {
	obs = models["policy"]->ForwardFeatures(obs, false);
	return models["aux_critic"]->Forward(obs, false).flatten();
}

void GGL::PPGLearner::SetLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	optimizers.SetLearningRate(*models["policy"], policyLR);
	optimizers.SetLearningRate(*models["critic"], criticLR);
	optimizers.SetLearningRate(*models["aux_critic"], policyLR);

	RG_LOG("PPGLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPGLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		std::string name = model->modelName;
		if (name == "critic" || name == "aux_critic")
			continue;

		result.Add(model);
	}
	return result;
}

void GGL::PPGLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	Timer policyTimer = {};
	RunPolicyPhase(experience, report, isFirstIteration);
	report["PPG/Policy Phase Time"] = policyTimer.Elapsed();

	AppendAuxExperience(experience);
	policyPhasesSinceAux++;
	report["PPG/Aux Buffer Timesteps"] = (double)auxBuffer.size;
	report["PPG/Policy Phases Since Aux"] = policyPhasesSinceAux;

	if (policyPhasesSinceAux >= ppgConfig.auxPhaseInterval) {
		Timer auxTimer = {};
		RunAuxiliaryPhase(report);
		report["PPG/Aux Phase Time"] = auxTimer.Elapsed();
		policyPhasesSinceAux = 0;
		auxBuffer.Clear();
	} else {
		report["PPG/Aux Phase Ran"] = 0.0;
	}
}

void GGL::PPGLearner::RunPolicyPhase(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	auto mseLoss = torch::nn::MSELoss();

	MutAvgTracker
		avgEntropy,
		avgDivergence,
		avgPolicyLoss,
		avgPolicyLossAbs,
		avgEntropyBonus,
		avgRelEntropyLoss,
		avgCriticLoss,
		avgGuidingLoss,
		avgClip;

	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;

	for (int epoch = 0; epoch < ppgConfig.policyEpochs; epoch++) {
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);

		for (auto& batch : batches) {
			int64_t batchSize = batch.states.size(0);
			auto fnRunPolicyMicroBatch = [&](int64_t start, int64_t stop) {
				// Accumulate memory microbatches into one optimizer step per PPO batch.
				float batchSizeRatio = (stop - start) / (float)batchSize;
				auto acts = batch.actions.slice(0, start, stop);
				auto obs = batch.states.slice(0, start, stop);
				auto actionMasks = batch.actionMasks.slice(0, start, stop);
				auto advantages = batch.advantages.slice(0, start, stop);
				auto oldProbs = batch.logProbs.slice(0, start, stop);

				auto probs = ModelManager::InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false, config.minActionProbability);
				auto logProbs = (probs + ACTION_ZERO_PROB).log().gather(-1, acts.unsqueeze(-1)).view_as(oldProbs);
				auto entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);

				auto logRatio = torch::clamp(logProbs - oldProbs, -20, 20);
				auto ratio = torch::exp(logRatio);
				auto clipped = torch::clamp(ratio, 1 - config.clipRange, 1 + config.clipRange);
				auto policyLoss = -torch::min(ratio * advantages, clipped * advantages).mean();
				const float entropyScale = config.entropyScale.Get();
				auto ppoLoss = (policyLoss - entropy * entropyScale) * batchSizeRatio;

				if (config.useGuidingPolicy) {
					torch::Tensor guidingProbs;
					{
						RG_NO_GRAD;
						guidingProbs = ModelManager::InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision, config.minActionProbability);
					}

					auto guidingLoss = (guidingProbs - probs).abs().mean();
					avgGuidingLoss.Add(guidingLoss.detach().cpu().item<float>());
					ppoLoss = ppoLoss + guidingLoss * config.guidingStrength * batchSizeRatio;
				}

				ppoLoss.backward();

				float curEntropy = entropy.detach().cpu().item<float>();
				float curPolicyLoss = policyLoss.detach().cpu().item<float>();
				float curPolicyLossAbs = curPolicyLoss < 0 ? -curPolicyLoss : curPolicyLoss;
				float curEntropyBonus = curEntropy * entropyScale;

				avgEntropy += curEntropy;
				avgPolicyLoss += curPolicyLoss;
				avgPolicyLossAbs += curPolicyLossAbs;
				avgEntropyBonus += curEntropyBonus;
				if (curPolicyLoss != 0)
					avgRelEntropyLoss += curEntropyBonus / curPolicyLoss;

				{
					RG_NO_GRAD;
					auto klTensor = (torch::exp(logRatio) - 1) - logRatio;
					avgDivergence += klTensor.mean().detach().cpu().item<float>();
					auto clipFraction = torch::mean((torch::abs(ratio - 1) > config.clipRange).to(torch::kFloat));
					avgClip += clipFraction.detach().cpu().item<float>();
				}
			};

			if (trainPolicy) {
				if (device.is_cpu()) {
					fnRunPolicyMicroBatch(0, batchSize);
				} else {
					for (int64_t mbs = 0; mbs < batchSize; mbs += config.microBatchSize)
						fnRunPolicyMicroBatch(mbs, RS_MIN(mbs + config.microBatchSize, batchSize));
				}

				nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
				optimizers.Step(*models["policy"]);
			}
		}
	}

	for (int epoch = 0; epoch < ppgConfig.criticEpochs; epoch++) {
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);

		for (auto& batch : batches) {
			int64_t batchSize = batch.states.size(0);
			auto fnRunCriticMicroBatch = [&](int64_t start, int64_t stop) {
				// Accumulate memory microbatches into one optimizer step per PPO batch.
				float batchSizeRatio = (stop - start) / (float)batchSize;
				auto obs = batch.states.slice(0, start, stop);
				auto targetValues = batch.targetValues.slice(0, start, stop);
				auto vals = InferTrueCritic(obs).view_as(targetValues);
				auto criticLoss = mseLoss(vals, targetValues);
				(criticLoss * batchSizeRatio).backward();

				avgCriticLoss += criticLoss.detach().cpu().item<float>();
			};

			if (trainCritic) {
				if (device.is_cpu()) {
					fnRunCriticMicroBatch(0, batchSize);
				} else {
					for (int64_t mbs = 0; mbs < batchSize; mbs += config.microBatchSize)
						fnRunCriticMicroBatch(mbs, RS_MIN(mbs + config.microBatchSize, batchSize));
				}

				nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);
				optimizers.Step(*models["critic"]);
			}
		}
	}

	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();
	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	report["LossAndEntropy/Policy Entropy"] = avgEntropy.Get();
	report["LossAndEntropy/Mean KL Divergence"] = avgDivergence.Get();
	report["PPG/Policy Mean KL Divergence"] = avgDivergence.Get();
	report["PPG/Policy Phase Policy Epochs"] = ppgConfig.policyEpochs;
	report["PPG/Policy Phase Critic Epochs"] = ppgConfig.criticEpochs;

	if (!isFirstIteration) {
		float policyLossAbs = avgPolicyLossAbs.Get();
		float entropyBonus = avgEntropyBonus.Get();
		float policyLossAbsDenom = policyLossAbs > 1e-6f ? policyLossAbs : 1e-6f;

		report["LossAndEntropy/Policy Loss"] = avgPolicyLoss.Get();
		report["LossAndEntropy/Policy Loss Abs"] = policyLossAbs;
		report["LossAndEntropy/Entropy Bonus"] = entropyBonus;
		report["LossAndEntropy/Entropy Bonus to Policy Loss Ratio"] = entropyBonus / policyLossAbsDenom;
		report["LossAndEntropy/Policy Relative Entropy Loss"] = avgRelEntropyLoss.Get();
		report["LossAndEntropy/Critic Loss"] = avgCriticLoss.Get();
		report["LossAndEntropy/SB3 Clip Fraction"] = avgClip.Get();
		report["LossAndEntropy/Policy Update Magnitude"] = policyUpdateMagnitude;
		report["LossAndEntropy/Critic Update Magnitude"] = criticUpdateMagnitude;
		if (config.useGuidingPolicy)
			report["LossAndEntropy/Guiding Loss"] = avgGuidingLoss.Get();
	}
}

void GGL::PPGLearner::AppendAuxExperience(const ExperienceBuffer& experience) {
	auxBuffer.states.push_back(experience.data.states.detach().cpu());
	auxBuffer.actionMasks.push_back(experience.data.actionMasks.detach().cpu());
	auxBuffer.targetValues.push_back(experience.data.targetValues.detach().cpu());
	auxBuffer.size += experience.data.states.size(0);
}

torch::Tensor GGL::PPGLearner::CacheCurrentPolicyProbs(torch::Tensor states, torch::Tensor actionMasks, int64_t microBatchSize) {
	std::vector<torch::Tensor> chunks = {};
	const int64_t size = states.size(0);
	for (int64_t start = 0; start < size; start += microBatchSize) {
		int64_t stop = RS_MIN(start + microBatchSize, size);
		RG_NO_GRAD;
		auto probs = ModelManager::InferPolicyProbsFromModels(
			models,
			states.slice(0, start, stop).to(device, true),
			actionMasks.slice(0, start, stop).to(device, true),
			config.policyTemperature,
			config.useHalfPrecision,
			config.minActionProbability
		).cpu();
		chunks.push_back(probs);
	}

	return torch::cat(chunks, 0);
}

void GGL::PPGLearner::RunAuxiliaryPhase(Report& report) {
	report["PPG/Aux Phase Ran"] = 1.0;

	if (ppgConfig.auxEpochs == 0 || auxBuffer.size == 0) {
		report["PPG/Aux Phase Skipped"] = 1.0;
		return;
	}

	auto mseLoss = torch::nn::MSELoss();
	torch::Tensor states = torch::cat(auxBuffer.states, 0);
	torch::Tensor actionMasks = torch::cat(auxBuffer.actionMasks, 0);
	torch::Tensor targetValues = torch::cat(auxBuffer.targetValues, 0);

	int64_t auxBatchSize = ppgConfig.auxBatchSize > 0 ? ppgConfig.auxBatchSize : config.batchSize;
	if (auxBatchSize <= 0)
		auxBatchSize = auxBuffer.size;
	auxBatchSize = RS_MIN(auxBatchSize, auxBuffer.size);

	int64_t auxMicroBatchSize = device.is_cpu() ? auxBatchSize : config.microBatchSize;
	if (auxMicroBatchSize <= 0)
		auxMicroBatchSize = auxBatchSize;
	auxMicroBatchSize = RS_MIN(auxMicroBatchSize, auxBatchSize);

	// auxBatchSize is the optimizer batch; config.microBatchSize only caps GPU memory.
	// NOTE: This is the PPG clone target. It must be captured after the last
	// policy phase and immediately before auxiliary optimization begins.
	torch::Tensor oldPolicyProbs = CacheCurrentPolicyProbs(states, actionMasks, auxMicroBatchSize);

	const float prevPolicyLR = config.policyLR;
	const float prevCriticLR = config.criticLR;
	if (ppgConfig.auxPolicyLR > 0) {
		optimizers.SetLearningRate(*models["policy"], ppgConfig.auxPolicyLR);
		optimizers.SetLearningRate(*models["aux_critic"], ppgConfig.auxPolicyLR);
	}
	if (ppgConfig.auxCriticLR > 0) {
		optimizers.SetLearningRate(*models["critic"], ppgConfig.auxCriticLR);
	}

	MutAvgTracker avgAuxValueLoss;
	MutAvgTracker avgCloneKL;
	MutAvgTracker avgTrueCriticLoss;
	MutAvgTracker avgTotalLoss;
	int64_t optimSteps = 0;

	for (int epoch = 0; epoch < ppgConfig.auxEpochs; epoch++) {
		auto indices = torch::randperm(auxBuffer.size, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

		for (int64_t start = 0; start < auxBuffer.size; start += auxBatchSize) {
			int64_t stop = RS_MIN(start + auxBatchSize, auxBuffer.size);
			const float auxBatchSizeInv = 1.0f / (float)(stop - start);

			for (int64_t mbs = start; mbs < stop; mbs += auxMicroBatchSize) {
				int64_t microStop = RS_MIN(mbs + auxMicroBatchSize, stop);
				float microBatchRatio = (microStop - mbs) * auxBatchSizeInv;
				auto batchIdx = indices.slice(0, mbs, microStop);

				auto obs = states.index_select(0, batchIdx).to(device, true);
				auto masks = actionMasks.index_select(0, batchIdx).to(device, true);
				auto targets = targetValues.index_select(0, batchIdx).to(device, true);
				auto oldProbs = oldPolicyProbs.index_select(0, batchIdx).to(device, true);

				auto curProbs = ModelManager::InferPolicyProbsFromModels(models, obs, masks, config.policyTemperature, false, config.minActionProbability);
				auto cloneKL = (oldProbs * ((oldProbs + ACTION_ZERO_PROB).log() - (curProbs + ACTION_ZERO_PROB).log())).sum(-1).mean();
				auto auxValues = InferAuxValue(obs).view_as(targets);
				auto auxValueLoss = mseLoss(auxValues, targets);
				auto trueCriticValues = InferTrueCritic(obs).view_as(targets);
				auto trueCriticLoss = mseLoss(trueCriticValues, targets);

				auto totalLoss =
					cloneKL * ppgConfig.auxPolicyCloneLossScale +
					auxValueLoss * ppgConfig.auxValueLossScale +
					trueCriticLoss * ppgConfig.auxTrueCriticLossScale;

				(totalLoss * microBatchRatio).backward();

				avgCloneKL += cloneKL.detach().cpu().item<float>();
				avgAuxValueLoss += auxValueLoss.detach().cpu().item<float>();
				avgTrueCriticLoss += trueCriticLoss.detach().cpu().item<float>();
				avgTotalLoss += totalLoss.detach().cpu().item<float>();
			}

			nn::utils::clip_grad_norm_(models["policy"]->parameters(), ppgConfig.auxGradClip);
			nn::utils::clip_grad_norm_(models["aux_critic"]->parameters(), ppgConfig.auxGradClip);
			nn::utils::clip_grad_norm_(models["critic"]->parameters(), ppgConfig.auxGradClip);

			optimizers.Step(*models["policy"]);
			optimizers.Step(*models["aux_critic"]);
			optimizers.Step(*models["critic"]);

			optimSteps++;
		}
	}

	if (ppgConfig.auxPolicyLR > 0 || ppgConfig.auxCriticLR > 0)
		SetLearningRates(prevPolicyLR, prevCriticLR);

	report["PPG/Aux Phase Skipped"] = 0.0;
	report["PPG/Aux Epochs"] = ppgConfig.auxEpochs;
	report["PPG/Aux Batch Size"] = (double)auxBatchSize;
	report["PPG/Aux MicroBatch Size"] = (double)auxMicroBatchSize;
	report["PPG/Aux Optim Steps"] = (double)optimSteps;
	report["PPG/Aux Policy Clone KL"] = avgCloneKL.Get();
	report["PPG/Aux Value Loss"] = avgAuxValueLoss.Get();
	report["PPG/Aux True Critic Loss"] = avgTrueCriticLoss.Get();
	report["PPG/Aux Total Loss"] = avgTotalLoss.Get();
}
