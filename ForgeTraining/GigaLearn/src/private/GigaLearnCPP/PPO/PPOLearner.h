#pragma once
#include "ExperienceBuffer.h"
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/PPO/PPOLearnerConfig.h>
#include <GigaLearnCPP/PPO/TransferLearnConfig.h>
#include <GigaLearnCPP/Optimizers/ModelOptimizer.h>

#include <GigaLearnInfer/Models.h>

#include <torch/nn/modules/loss.h>
#include <torch/nn/modules/container/sequential.h>

#include <string>
#include <vector>

#include "ExperienceBuffer.h"

namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	class GGL_TRAIN_IMEXPORT PPOLearner {
	public:
		ModelSet models = {};
		ModelSet guidingPolicyModels = {};
		ModelOptimizerSet optimizers;

		PPOLearnerConfig config;
		torch::Device device;

		PPOLearner(
			int obsSize, int numActions,
			PPOLearnerConfig config, torch::Device device,
			bool initializeModels = true,
			int numCriticOutputs = 1
		);

		virtual void Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration);

		void TransferLearn(
			ModelSet& oldModels, 
			torch::Tensor newObs, torch::Tensor oldObs, 
			torch::Tensor newActionMasks, torch::Tensor oldActionMasks, 
			torch::Tensor actionMaps,
			Report& report, 
			const TransferLearnConfig& transferLearnConfig
		);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);

		/**
		 * Loads like LoadFrom, for a checkpoint whose critic was saved with different value heads:
		 * the critic's trunk loads unchanged, its output rows are matched to headNames by name,
		 * and its optimizer state is not loaded, since that state belongs to the saved rows.
		 */
		void LoadFromRemappingCriticHeads(
			std::filesystem::path folderPath,
			const std::vector<std::string>& savedHeadNames,
			const std::vector<std::string>& headNames
		);

		virtual void SetLearningRates(float policyLR, float criticLR);

		virtual ModelSet GetPolicyModels();

		virtual ~PPOLearner();
	};
}