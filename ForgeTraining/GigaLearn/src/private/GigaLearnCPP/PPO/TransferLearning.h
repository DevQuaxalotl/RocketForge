#pragma once

#include <GigaLearnInfer/FrameworkTorch.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>
#include <cmath>
#include <limits>

namespace GGL::TransferLearning {
	/** Reject invalid settings before collecting data or changing parameters. **/
	inline void ValidateConfig(const TransferLearnConfig& config) {
		TORCH_CHECK(config.batchSize > 0 && config.epochs > 0 && config.microBatchSize >= 0,
			"Transfer batch size and epochs must be positive; microbatch size cannot be negative");
		TORCH_CHECK(std::isfinite(config.lr) && config.lr > 0
			&& std::isfinite(config.lossScale) && config.lossScale > 0
			&& std::isfinite(config.lossExponent) && config.lossExponent > 0,
			"Transfer learning rate, loss scale and exponent must be finite and positive");
	}

	/** Log probabilities conditional on the supplied legal actions, before any loss reduction. **/
	inline torch::Tensor ConditionalLogProbs(torch::Tensor logits, torch::Tensor mask, float floor = 0) {
		mask = mask.to(torch::kBool);

		// NOTE: Empty rows need a finite softmax, but must contribute no gradient or probability mass.
		torch::Tensor safeMask = mask.logical_or(mask.any(1, true).logical_not());
		torch::Tensor logProbs = torch::log_softmax(logits.masked_fill(safeMask.logical_not(), -std::numeric_limits<float>::infinity()), 1);

		if (floor > 0) {
			const float actionFloor = std::min(floor, 1.f / static_cast<float>(logits.size(1)));
			torch::Tensor learnedMass = 1 - actionFloor * mask.sum(1, true).to(torch::kFloat32);
			logProbs = (logProbs.exp() * learnedMass + actionFloor).clamp_min(1e-30).log();
		}

		return logProbs.masked_fill(mask.logical_not(), 0);
	}

	/** Paired teacher probabilities and the legal intersection, in student action order. **/
	struct Targets {
		torch::Tensor probabilities;
		torch::Tensor mask;
	};

	/** Maps teacher actions without inventing labels for additional student actions. **/
	inline Targets MapTargets(torch::Tensor teacher, torch::Tensor teacherMask, torch::Tensor studentMask, torch::Tensor actionMap) {
		torch::Tensor mask = studentMask.to(torch::kBool);
		if (actionMap.defined()) {
			torch::Tensor indices = actionMap.clamp_min(0);
			mask = mask.logical_and(actionMap >= 0).logical_and(teacherMask.gather(1, indices).to(torch::kBool));
			teacher = teacher.gather(1, indices);
		} else {
			mask = mask.logical_and(teacherMask.to(torch::kBool));
		}

		teacher = teacher.detach() * mask;
		torch::Tensor mass = teacher.sum(1, true);
		mask = mask.logical_and(mass > 0);

		return { teacher / torch::where(mass > 0, mass, torch::ones_like(mass)), mask };
	}

	/** Per-row loss; empty shared-action rows are zero and retain a finite zero gradient. **/
	inline torch::Tensor LossRows(torch::Tensor logProbs, const Targets& targets, const TransferLearnConfig& config) {
		torch::Tensor valid = targets.mask.any(1);
		torch::Tensor loss;
		if (config.useHardActionLabels) {
			loss = -logProbs.gather(1, targets.probabilities.argmax(1, true)).squeeze(1);
		} else if (config.useKLDiv) {
			loss = (targets.probabilities * (targets.probabilities.clamp_min(1e-30).log() - logProbs)).sum(1);
		} else {
			torch::Tensor probs = logProbs.exp() * targets.mask;
			loss = (targets.probabilities - probs).abs();
		}

		// NOTE: Roundoff may make KL slightly negative; fractional powers also need a safe derivative at zero.
		loss = torch::where(loss > 0, loss.clamp_min(1e-30).pow(config.lossExponent), torch::zeros_like(loss));
		if (loss.dim() == 2)
			loss = loss.mean(1);

		return loss * valid * config.lossScale;
	}
}
