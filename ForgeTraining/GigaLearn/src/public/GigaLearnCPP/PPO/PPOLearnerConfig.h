#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "GigaLearnCPP/ScheduledValue.h"
#include "GigaLearnCPP/PPO/DecomposedCriticConfig.h"
#include "GigaLearnInfer/JsonArchive.h"
#include "GigaLearnInfer/ModelConfig.h"

namespace GGL {

	/** PPO learning settings, held by LearnerConfig::ppo. */
	struct PPOLearnerConfig {

		/**************** Commonly tuned ****************/

		/** Scale of the normalized entropy bonus; higher keeps the policy exploring longer. */
		ScheduledFloat entropyScale = 0.018f;

		/** Passes over each iteration's experience. */
		int epochs = 2;

		/** Discount factor: how far into the future rewards are credited. */
		float gaeGamma = 0.99f;
		float gaeLambda = 0.95f;

		float clipRange = 0.2f;

		/** In seconds. */
		double maxEpisodeDuration = 120;

		/**
		 * Clip range for normalized rewards; 0 disables.
		 * Ignored with the decomposed critic, whose per-head GAE must stay unclipped to combine linearly.
		 */
		float rewardClipRange = 10;

		/**
		 * Normalizes policy-loss advantages to zero mean and unit std over each optimizer batch.
		 * Critic targets are unaffected.
		 * Advantages grow by 1 / (pre-normalization advantage std),
		 * so settings such as entropyScale and guidingStrength may need adjusting when switching.
		 */
		bool normalizeAdvantages = false;

		/**
		 * Max gradient norm for the policy network; the critic and shared head always clip at 0.5.
		 * With normalizeAdvantages on, 0.5 / (pre-normalization advantage std) keeps the previous clip frequency.
		 */
		float policyGradClipNorm = 0.5f;

		/**
		 * GPU memory microbatch size; 0 uses batchSize.
		 * Splits each batch into smaller forward/backward passes to reduce peak VRAM,
		 * accumulating gradients into one optimizer step.
		 */
		int64_t microBatchSize = 0;

		/** Experimental; see DecomposedCriticConfig. */
		DecomposedCriticConfig decomposedCritic = {};

		/**************** Less commonly changed ****************/

		/** Temperature of the policy's softmax distribution. */
		float policyTemperature = 1;

		/**
		 * Minimum probability for each currently valid action during PPO sampling and training; <= 0 disables.
		 * Masked actions still get zero probability.
		 */
		float minActionProbability = 0.0001f;

		/**
		 * True computes entropy only over the actions available in each state.
		 * False gives unavailable actions zero entropy, which caps a state's entropy at its fraction of available actions.
		 */
		bool maskEntropy = false;

		/** Added to the advantage std when normalizeAdvantages is on. */
		float advantageNormEpsilon = 1e-8f;

		/**
		 * On the last batch of an iteration, uses all remaining experience as one larger batch
		 * when less than two batches' worth is left, so batch-size rounding never drops experience.
		 */
		bool overbatching = true;

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/";
		float guidingStrength = 0.03f;

		/**
		 * Always picks the most likely action.
		 * Usually plays better but cannot learn; a PPO learn iteration throws in this mode.
		 */
		bool deterministic = false;

		/** Half-precision inference; much faster on GPU, little gain on CPU. */
		bool useHalfPrecision = false;

		/** Uses CUDA graph replay for graph-safe GPU inference paths. */
		bool useCudaGraphs = true;

		/**************** Set by TrainingStage ****************/

		/** Set from TrainingStage::timestepsPerIteration. */
		int64_t tsPerItr = 50'000;

		/** Optimizer batch size for each PPO update. Set from TrainingStage::timestepsPerIteration. */
		int64_t batchSize = 50'000;

		/** Set from TrainingStage::learningRate. */
		float policyLR = 3e-4f;

		/** Set from TrainingStage::learningRate. */
		float criticLR = 3e-4f;

		/** Set from the Agent's network settings. */
		PartialModelConfig policy, critic, sharedHead;
	};
}
