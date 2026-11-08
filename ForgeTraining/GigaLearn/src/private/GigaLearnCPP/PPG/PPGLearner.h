#pragma once

#include <GigaLearnCPP/PPG/PPGLearnerConfig.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>

namespace GGL {

	/**
	 * Full dual-network Phasic Policy Gradient learner.
	 * PPG uses independent full policy and critic networks. PPG-only value
	 * models are excluded from policy versioning and runtime inference.
	 **/
	class GGL_TRAIN_IMEXPORT PPGLearner : public PPOLearner {
	public:
		PPGLearner(
			int obsSize, int numActions,
			PPOLearnerConfig ppoConfig,
			PPGLearnerConfig ppgConfig,
			torch::Device device
		);

		void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) override;
		void SetLearningRates(float policyLR, float criticLR) override;
		ModelSet GetPolicyModels() override;

	private:
		struct AuxBuffer {
			std::vector<torch::Tensor> states;
			std::vector<torch::Tensor> actionMasks;
			std::vector<torch::Tensor> targetValues;
			int64_t size = 0;

			void Clear();
		};

		PPGLearnerConfig ppgConfig;
		AuxBuffer auxBuffer;
		int policyPhasesSinceAux = 0;

		void MakePPGModels(int obsSize, int numActions);
		void LoadGuidingPolicyIfEnabled(int obsSize, int numActions);
		void LogModelParameterCounts() const;

		torch::Tensor InferTrueCritic(torch::Tensor obs);
		torch::Tensor InferAuxValue(torch::Tensor obs);

		void RunPolicyPhase(ExperienceBuffer& experience, Report& report, bool isFirstIteration);
		void AppendAuxExperience(const ExperienceBuffer& experience);
		void RunAuxiliaryPhase(Report& report);
		torch::Tensor CacheCurrentPolicyProbs(torch::Tensor states, torch::Tensor actionMasks, int64_t microBatchSize);
	};
}
