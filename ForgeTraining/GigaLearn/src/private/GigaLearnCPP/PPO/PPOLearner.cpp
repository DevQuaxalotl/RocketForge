#include "PPOLearner.h"
#include "AdvantageNormalization.h"
#include "DecomposedCritic.h"
#include "TransferLearning.h"

#include <string_view>
#include <torch/nn/utils/convert_parameters.h>
#include <torch/csrc/api/include/torch/serialize.h>

#include <GigaLearnInfer/ModelManager.h>

using namespace torch;

namespace {
	constexpr float ACTION_ZERO_PROB = 1e-11f;
	constexpr float GRAD_CLIP_MAX_NORM = 0.5f;

	struct TensorAvgTracker {
		torch::Tensor total;
		torch::Tensor count;

		void Add(const torch::Tensor& value) {
			AddIf(value, torch::logical_not(torch::isnan(value)));
		}

		void AddIf(const torch::Tensor& value, const torch::Tensor& condition) {
			torch::Tensor detached = value.detach();
			torch::Tensor valid = condition.detach().logical_and(torch::logical_not(torch::isnan(detached)));

			if (!total.defined()) {
				total = torch::zeros_like(detached);
				count = torch::zeros_like(detached);
			}

			total = total + torch::where(valid, detached, torch::zeros_like(detached));
			count = count + valid.to(detached.scalar_type());
		}

		float Get(double& metricSyncTime) const {
			if (!total.defined())
				return 0;

			GGL::Timer timer = {};
			float result = (total / torch::clamp(count, 1)).cpu().item<float>();
			metricSyncTime += timer.Elapsed();
			return result;
		}
	};

	struct GradClipMetrics {
		torch::Tensor norm;
		torch::Tensor appliedScale;
	};

	GradClipMetrics ClipGradNormNoSync(const std::vector<torch::Tensor>& parameters, double maxNorm) {
		std::vector<torch::Tensor> paramsWithGrad;
		paramsWithGrad.reserve(parameters.size());

		for (const auto& param : parameters) {
			if (param.grad().defined())
				paramsWithGrad.push_back(param);
		}

		if (paramsWithGrad.empty())
			return {};

		std::vector<torch::Tensor> norms;
		norms.reserve(paramsWithGrad.size());
		for (const auto& param : paramsWithGrad)
			norms.emplace_back(param.grad().data().norm(2.0));

		torch::Tensor totalNorm = norms.size() == 1 ? norms[0] : torch::stack(norms).norm(2.0);
		torch::Tensor clipCoef = maxNorm / (totalNorm + 1e-6);
		torch::Tensor clipCoefClamped = torch::clamp(clipCoef, 0.0, 1.0);

		for (const auto& param : paramsWithGrad)
			param.grad().data().mul_(clipCoefClamped);

		return { totalNorm, clipCoefClamped };
	}

	torch::Tensor CopyParamsNoSync(const GGL::Model* model) {
		return torch::nn::utils::parameters_to_vector(model->parameters()).detach();
	}
}

GGL::PPOLearner::~PPOLearner() {
	optimizers.Clear();
	models.Free();
	guidingPolicyModels.Free();
}

GGL::PPOLearner::PPOLearner(
	int obsSize,
	int numActions,
	PPOLearnerConfig _config,
	Device _device,
	bool initializeModels,
	int numCriticOutputs)
	: config(_config), device(_device) {

	if (config.microBatchSize == 0)
		config.microBatchSize = config.batchSize;

	if (config.batchSize % config.microBatchSize != 0)
		RG_ERR_CLOSE(
			"PPOLearner: config.batchSize (" << config.batchSize
			<< ") must be a multiple of config.microBatchSize (" << config.microBatchSize << ")");

	if (!initializeModels)
		return;

	const bool makeCritic = true;
	ModelManager::MakeModels(
		makeCritic,
		obsSize,
		numActions,
		config.sharedHead,
		config.policy,
		config.critic,
		device,
		models,
		numCriticOutputs
	);
	optimizers.Attach(models);

	SetLearningRates(config.policyLR, config.criticLR);

	// Print param counts
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : this->models) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));

	if (config.useGuidingPolicy) {
		RG_LOG("Guiding policy enabled, loading from " << config.guidingPolicyPath << "...");
		const bool makeGuidingCritic = false;
		ModelManager::MakeModels(
			makeGuidingCritic, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, guidingPolicyModels);
		guidingPolicyModels.Load(config.guidingPolicyPath, false);
	}
}



torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	// Compute log probs and entropy
	auto entropy = -((probs + ACTION_ZERO_PROB).log() * probs).sum(-1);

	if (maskEntropy) {
		// Account for action masking in entropy
		// We will effectively narrow the entropy to the scope of the valid actions
		// This way states with more masked actions don't just have inherently lower entropy
		entropy /= actionMasks.to(torch::kFloat32).sum(-1).log();
	} else {
		entropy /= logf(actionMasks.size(-1));
	}

	return entropy.mean();
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	auto mseLoss = torch::nn::MSELoss();

	TensorAvgTracker
		avgEntropy,
		avgDivergence,
		avgPolicyLoss,
		avgPolicyLossAbs,
		avgEntropyBonus,
		avgRelEntropyLoss,
		avgCriticLoss,
		avgGuidingLoss,
		avgRatio,
		avgClip,
		avgPolicyGradientNorm,
		avgPolicyGradientAppliedScale;

	double metricSyncTime = 0;

	// Save parameters first
	torch::Tensor policyBefore = CopyParamsNoSync(models["policy"]);
	torch::Tensor criticBefore = CopyParamsNoSync(models["critic"]);

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchActionMasks = batch.actionMasks;
			auto batchTargetValues = batch.targetValues;
			auto batchAdvantages = batch.advantages;
			auto batchCriticLossWeights = batch.criticLossWeights;

			// Normalized over the whole optimizer batch so microbatch size never changes the statistics.
			if (config.normalizeAdvantages)
				batchAdvantages = NormalizeAdvantages(batchAdvantages, config.advantageNormEpsilon);

			auto fnRunMicroBatch = [&](int start, int stop) {

				float batchSizeRatio = (stop - start) / (float)config.batchSize;

				// Experience already lives on the training device (moved once in Learner.cpp
				// after the consumption build), so slicing is sufficient -- no transfer needed.
				auto acts = batchActs.slice(0, start, stop);
				auto obs = batchObs.slice(0, start, stop);
				auto actionMasks = batchActionMasks.slice(0, start, stop);

				auto advantages = batchAdvantages.slice(0, start, stop);
				auto oldProbs = batchOldProbs.slice(0, start, stop);
				auto targetValues = batchTargetValues.slice(0, start, stop);

				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {

					// Get policy log probs and entropy
					{
						const bool useHalfPrecision = false;
						probs = ModelManager::InferPolicyProbsFromModels(
							models, obs, actionMasks, config.policyTemperature, useHalfPrecision, config.minActionProbability);
						logProbs = (probs + ACTION_ZERO_PROB).log().gather(-1, acts.unsqueeze(-1));
						entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);
						avgEntropy.Add(entropy);
					}

					logProbs = logProbs.view_as(oldProbs);

					// Compute PPO loss
					ratio = exp(logProbs - oldProbs);
					avgRatio.Add(ratio.mean());
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss
					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					avgPolicyLoss.Add(policyLoss);
					avgPolicyLossAbs.Add(torch::abs(policyLoss));

					const float entropyScale = config.entropyScale.Get();
					torch::Tensor entropyBonus = entropy * entropyScale;
					avgEntropyBonus.Add(entropyBonus);
					avgRelEntropyLoss.AddIf(entropyBonus / policyLoss, policyLoss != 0);

					ppoLoss = (policyLoss - entropy * entropyScale) * batchSizeRatio;

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = ModelManager::InferPolicyProbsFromModels(
								guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision, config.minActionProbability);
						}

						torch::Tensor guidingLoss = (guidingProbs - probs).abs().mean();
						avgGuidingLoss.Add(guidingLoss);
						guidingLoss = guidingLoss * config.guidingStrength;
						ppoLoss = ppoLoss + guidingLoss;
					}
				}

				torch::Tensor criticLoss;
				if (trainCritic) {
					if (config.decomposedCritic.enabled) {
						torch::Tensor headValues = ModelManager::InferCriticHeads(obs, models, config.useHalfPrecision);
						torch::Tensor lossWeights = batchCriticLossWeights.slice(0, start, stop);
						RG_ASSERT(headValues.sizes() == targetValues.sizes() && lossWeights.sizes() == targetValues.sizes());

						criticLoss = (lossWeights * (headValues - targetValues).square()).mean();
					} else {
						auto vals = ModelManager::InferCritic(obs, models, config.useHalfPrecision);

						// Compute value loss
						vals = vals.view_as(targetValues);
						criticLoss = mseLoss(vals, targetValues);
					}

					avgCriticLoss.Add(criticLoss);
					criticLoss = criticLoss * batchSizeRatio;
				}

				if (trainPolicy) {
					// Compute KL divergence & clip fraction using SB3 method for reporting;
					{
						RG_NO_GRAD;

						auto logRatio = logProbs - oldProbs;
						auto klTensor = (exp(logRatio) - 1) - logRatio;
						avgDivergence.Add(klTensor.mean());

						auto clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat));
						avgClip.Add(clipFraction);
					}
				}

				if (trainPolicy && trainCritic) {
					auto combinedLoss = ppoLoss + criticLoss;
					combinedLoss.backward();
				} else {
					if (trainPolicy)
						ppoLoss.backward();
					if (trainCritic)
						criticLoss.backward();
				}
			};

			
			if (device.is_cpu()) {
				// Run the full optimizer batch in one pass.
				fnRunMicroBatch(0, config.batchSize);
			} else {
				for (int mbs = 0; mbs < config.batchSize; mbs += config.microBatchSize) {
					int start = mbs;
					int stop = start + config.microBatchSize;
					fnRunMicroBatch(start, stop);
				}
			}

			if (trainPolicy) {
				GradClipMetrics policyGradClip = ClipGradNormNoSync(models["policy"]->parameters(), config.policyGradClipNorm);
				if (policyGradClip.norm.defined()) {
					avgPolicyGradientNorm.Add(policyGradClip.norm);
					avgPolicyGradientAppliedScale.Add(policyGradClip.appliedScale);
				}
			}
			if (trainCritic)
				ClipGradNormNoSync(models["critic"]->parameters(), GRAD_CLIP_MAX_NORM);

			if (trainSharedHead)
				ClipGradNormNoSync(models["shared_head"]->parameters(), GRAD_CLIP_MAX_NORM);

			optimizers.StepAll(models);
		}
	}

	// Compute magnitude of updates made to the policy and value estimator
	torch::Tensor policyAfter = CopyParamsNoSync(models["policy"]);
	torch::Tensor criticAfter = CopyParamsNoSync(models["critic"]);

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	// Assemble and return report
	report["LossAndEntropy/Policy Entropy"] = avgEntropy.Get(metricSyncTime);
	report["LossAndEntropy/Mean KL Divergence"] = avgDivergence.Get(metricSyncTime);
	if (!isFirstIteration) {
		// These metrics give bad data on the first iteration, which will mess up graph scaling
		// So we'll just skip them for the first iteration
		float policyLossAbs = avgPolicyLossAbs.Get(metricSyncTime);
		float entropyBonus = avgEntropyBonus.Get(metricSyncTime);
		float policyLossAbsDenom = policyLossAbs > 1e-6f ? policyLossAbs : 1e-6f;

		report["LossAndEntropy/Policy Loss"] = avgPolicyLoss.Get(metricSyncTime);
		report["LossAndEntropy/Policy Loss Abs"] = policyLossAbs;
		report["LossAndEntropy/Entropy Bonus"] = entropyBonus;
		report["LossAndEntropy/Entropy Bonus to Policy Loss Ratio"] = entropyBonus / policyLossAbsDenom;
		report["LossAndEntropy/Policy Relative Entropy Loss"] = avgRelEntropyLoss.Get(metricSyncTime);
		report["LossAndEntropy/Critic Loss"] = avgCriticLoss.Get(metricSyncTime);

		if (config.useGuidingPolicy)
			report["LossAndEntropy/Guiding Loss"] = avgGuidingLoss.Get(metricSyncTime);

		report["LossAndEntropy/SB3 Clip Fraction"] = avgClip.Get(metricSyncTime);
		report["LossAndEntropy/Policy Gradient Norm Before Clip"] = avgPolicyGradientNorm.Get(metricSyncTime);
		report["LossAndEntropy/Policy Gradient Applied Scale"] = avgPolicyGradientAppliedScale.Get(metricSyncTime);
		report["LossAndEntropy/Policy Update Magnitude"] = policyUpdateMagnitude;
		report["LossAndEntropy/Critic Update Magnitude"] = criticUpdateMagnitude;
	}
}

void GGL::PPOLearner::TransferLearn(
	ModelSet& oldModels,
	torch::Tensor newObs, torch::Tensor oldObs,
	torch::Tensor newActionMasks, torch::Tensor oldActionMasks,
	torch::Tensor actionMaps,
	Report& report,
	const TransferLearnConfig& tlConfig
) {

	TransferLearning::ValidateConfig(tlConfig);
	TORCH_CHECK(std::isfinite(config.policyTemperature) && config.policyTemperature > 0
		&& std::isfinite(config.minActionProbability) && config.minActionProbability >= 0,
		"Transfer policy temperature must be positive and exploration floor nonnegative");
	TORCH_CHECK(models["policy"] && oldModels["policy"], "Transfer requires teacher and student policies");
	TORCH_CHECK(newObs.dim() == 2 && oldObs.dim() == 2 && newObs.size(0) > 0
		&& newObs.size(0) == oldObs.size(0), "Transfer requires nonempty paired observation rows");

	const int64_t rowCount = newObs.size(0);
	const int64_t studentActions = models["policy"]->config.numOutputs;
	const int64_t teacherActions = oldModels["policy"]->config.numOutputs;
	auto inputWidth = [](ModelSet& set) {
		return (set["shared_head"] ? set["shared_head"] : set["policy"])->config.numInputs;
	};

	TORCH_CHECK(newObs.size(1) == inputWidth(models) && oldObs.size(1) == inputWidth(oldModels),
		"Transfer observation width does not match its policy");
	TORCH_CHECK(newActionMasks.dim() == 2 && oldActionMasks.dim() == 2
		&& newActionMasks.size(0) == rowCount && oldActionMasks.size(0) == rowCount
		&& newActionMasks.size(1) == studentActions && oldActionMasks.size(1) == teacherActions,
		"Transfer action masks must match the paired rows and policy action widths");

	if (actionMaps.defined()) {
		TORCH_CHECK(actionMaps.sizes() == newActionMasks.sizes()
			&& (actionMaps.scalar_type() == torch::kInt32 || actionMaps.scalar_type() == torch::kInt64),
			"Transfer action maps must be integer teacher indices in student action order");
		actionMaps = actionMaps.detach().to(torch::kCPU).to(torch::kInt64);
	} else {
		TORCH_CHECK(studentActions == teacherActions, "Different action spaces require a transfer action map");
	}

	// Keep the collection and cached targets on the host.
	// Only one chunk's inputs, teacher outputs or student autograd graph reside on the training device at once.
	newObs = newObs.detach().to(torch::kCPU);
	oldObs = oldObs.detach().to(torch::kCPU);
	newActionMasks = newActionMasks.detach().to(torch::kCPU);
	oldActionMasks = oldActionMasks.detach().to(torch::kCPU);

	const int64_t chunkSize = tlConfig.microBatchSize > 0 ? std::min<int64_t>(tlConfig.microBatchSize, rowCount) : rowCount;
	torch::Tensor targets = torch::empty({ rowCount, studentActions }, torch::kFloat32);
	torch::Tensor sharedMasks = torch::empty({ rowCount, studentActions }, torch::kBool);
	auto entropy = [&](torch::Tensor probs, torch::Tensor mask) {
		torch::Tensor raw = -(probs * probs.clamp_min(1e-30).log()).sum(1);
		torch::Tensor count = config.maskEntropy
			? mask.sum(1).to(torch::kFloat32)
			: torch::full_like(raw, static_cast<double>(probs.size(1)));
		return torch::where(count > 1, raw / count.clamp_min(2).log(), torch::zeros_like(raw));
	};

	double teacherEntropy = 0;
	int64_t supervisedRows = 0;
	{
		RG_NO_GRAD;
		for (int64_t start = 0; start < rowCount; start += chunkSize) {
			const int64_t end = std::min(start + chunkSize, rowCount);
			torch::Tensor newMask = newActionMasks.slice(0, start, end);
			torch::Tensor oldMask = oldActionMasks.slice(0, start, end);
			TORCH_CHECK(torch::isfinite(newObs.slice(0, start, end)).all().item<bool>()
				&& torch::isfinite(oldObs.slice(0, start, end)).all().item<bool>(), "Non-finite transfer observations");
			TORCH_CHECK(((newMask == 0) | (newMask == 1)).all().item<bool>()
				&& ((oldMask == 0) | (oldMask == 1)).all().item<bool>(), "Transfer action masks must contain only zero or one");

			torch::Tensor map;
			if (actionMaps.defined()) {
				map = actionMaps.slice(0, start, end);
				TORCH_CHECK(((map >= -1) & (map < teacherActions)).all().item<bool>(), "Transfer action map index out of range");
				torch::Tensor sorted = std::get<0>(map.sort(1));
				torch::Tensor left = sorted.slice(1, 0, studentActions - 1);
				torch::Tensor right = sorted.slice(1, 1, studentActions);
				TORCH_CHECK(!((left == right) & (left >= 0)).any().item<bool>(), "Transfer action map repeats a teacher index");
				map = map.to(device);
			}

			oldMask = oldMask.to(device, torch::kBool);
			torch::Tensor teacher = ModelManager::InferPolicyProbsFromModels(
				oldModels, oldObs.slice(0, start, end).to(device), oldMask,
				config.policyTemperature, config.useHalfPrecision, config.minActionProbability).to(torch::kFloat32);
			TransferLearning::Targets mapped = TransferLearning::MapTargets(teacher, oldMask, newMask.to(device, torch::kBool), map);
			targets.slice(0, start, end).copy_(mapped.probabilities);
			sharedMasks.slice(0, start, end).copy_(mapped.mask);
			supervisedRows += mapped.mask.any(1).sum().item<int64_t>();
			teacherEntropy += entropy(teacher, oldMask).sum().item<double>();
		}
	}

	report["LossAndEntropy/Old Policy Entropy"] = teacherEntropy / rowCount;
	report["Transfer/Supervised Rows"] = supervisedRows;
	report["Transfer/Skipped Rows"] = rowCount - supervisedRows;
	report["Transfer/Microbatch Size"] = chunkSize;
	report["Transfer/Optimizer Steps"] = 0;

	// No optimizer momentum or weight decay may run on an entirely unlabeled batch.
	if (supervisedRows == 0) {
		report["Transfer Learn Loss"] = 0;
		report["Transfer Learn Accuracy"] = 0;
		report["LossAndEntropy/Policy Update Magnitude"] = 0;
		return;
	}

	ModelSet policyModels = GetPolicyModels();
	for (Model* model : policyModels) {
		optimizers.SetLearningRate(*model, tlConfig.lr);
		model->zero_grad();
	}

	torch::Tensor policyBefore = models["policy"]->CopyParams();
	double initialLoss = 0, initialMatches = 0, studentEntropy = 0;

	try {
		for (int epoch = 0; epoch < tlConfig.epochs; epoch++) {
			for (int64_t start = 0; start < rowCount; start += chunkSize) {
				const int64_t end = std::min(start + chunkSize, rowCount);
				TransferLearning::Targets target = { targets.slice(0, start, end).to(device), sharedMasks.slice(0, start, end).to(device) };

				torch::Tensor input = newObs.slice(0, start, end).to(device);
				if (models["shared_head"])
					input = models["shared_head"]->Forward(input, false);

				torch::Tensor logits = models["policy"]->Forward(input, false) / config.policyTemperature;
				TORCH_CHECK(torch::isfinite(logits).all().item<bool>(), "Non-finite student logits during transfer");

				torch::Tensor logProbs = TransferLearning::ConditionalLogProbs(logits, target.mask, config.minActionProbability);
				torch::Tensor rows = TransferLearning::LossRows(logProbs, target, tlConfig);
				if (epoch == 0) {
					RG_NO_GRAD;
					initialLoss += rows.sum().item<double>();

					torch::Tensor predicted =
						logProbs.masked_fill(target.mask.logical_not(), -std::numeric_limits<float>::infinity()).argmax(1);
					initialMatches += ((predicted == target.probabilities.argmax(1)) & target.mask.any(1)).sum().item<double>();

					torch::Tensor mask = newActionMasks.slice(0, start, end).to(device, torch::kBool);
					torch::Tensor probs =
						TransferLearning::ConditionalLogProbs(logits.detach(), mask, config.minActionProbability).exp() * mask;
					studentEntropy += entropy(probs, mask).sum().item<double>();
				}

				// Accumulate the full-batch mean, including a shorter final chunk, then step once per epoch.
				(rows.sum() / rowCount).backward();
			}

			optimizers.StepAll(policyModels);
			report["Transfer/Optimizer Steps"] += 1;
		}
	} catch (...) {
		for (Model* model : policyModels)
			model->zero_grad();
		throw;
	}

	report["Transfer Learn Loss"] = initialLoss / rowCount;
	report["Transfer Learn Accuracy"] = initialMatches / supervisedRows;
	report["LossAndEntropy/Policy Entropy"] = studentEntropy / rowCount;
	report["LossAndEntropy/Policy Update Magnitude"] = (policyBefore - models["policy"]->CopyParams()).norm().item<float>();
}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
	optimizers.Save(models, folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true);
	optimizers.Load(models, folderPath);

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::LoadFromRemappingCriticHeads(
	std::filesystem::path folderPath,
	const std::vector<std::string>& savedHeadNames,
	const std::vector<std::string>& headNames
) {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner::LoadFromRemappingCriticHeads(): Path " << folderPath << " is not a valid directory");

	ModelSet policyModels = GetPolicyModels();
	policyModels.Load(folderPath, true);

	Model& critic = *models["critic"];
	ModelConfig savedCriticConfig = critic.config;
	savedCriticConfig.numOutputs = (int)savedHeadNames.size();
	Model savedCritic(critic.modelName, savedCriticConfig, device);
	savedCritic.Load(folderPath, false);
	RemapCriticHeads(critic, savedCritic, savedHeadNames, headNames);

	optimizers.Load(policyModels, folderPath);

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::SetLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	optimizers.SetLearningRate(*models["policy"], policyLR);
	optimizers.SetLearningRate(*models["critic"], criticLR);

	if (models["shared_head"])
		optimizers.SetLearningRate(*models["shared_head"], RS_MIN(policyLR, criticLR));

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		if (std::string_view(model->modelName) == "critic")
			continue;
		
		result.Add(model);
	}
	return result;
}
