#pragma once

#include <cstdint>

namespace GGL {

	/**
	 * Full dual-network Phasic Policy Gradient configuration.
	 * PPG is currently not functioning; train with PPO.
	 * These values are only read when LearnerConfig::algorithm is LearnerAlgorithm::PPG.
	 **/
	struct PPGLearnerConfig {
		/** PPO-style clipped policy epochs per rollout. **/
		int policyEpochs = 1;

		/** True critic value epochs per rollout. **/
		int criticEpochs = 1;

		/** Number of policy phases to accumulate before one auxiliary phase. **/
		int auxPhaseInterval = 32;

		/** Auxiliary epochs over the accumulated auxiliary buffer. **/
		int auxEpochs = 6;

		/** Loss scale for fitting aux_critic values from policy-network features. **/
		float auxValueLossScale = 1.0f;

		/** Loss scale for keeping auxiliary training close to the pre-auxiliary policy distribution. **/
		float auxPolicyCloneLossScale = 1.0f;

		/** Loss scale for continuing true critic value training during the auxiliary phase. **/
		float auxTrueCriticLossScale = 1.0f;

		/** Maximum gradient norm applied to auxiliary-phase policy, aux_critic, and critic updates. **/
		float auxGradClip = 0.5f;

		/**
		 * Optional auxiliary-phase overrides.
		 * Zero uses the corresponding PPO batch size or learning rate.
		 * PPO microBatchSize remains a memory microbatch limit.
		 **/

		/** Optimizer batch size used when training over the accumulated auxiliary buffer. **/
		int64_t auxBatchSize = 0;

		/** Policy and auxiliary-critic learning rate used during the auxiliary phase. **/
		float auxPolicyLR = 0;

		/** True critic learning rate used during the auxiliary phase. **/
		float auxCriticLR = 0;
	};
}
