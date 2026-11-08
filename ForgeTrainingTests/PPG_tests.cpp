#include <gtest/gtest.h>

#include "TestSupport.h"

#include <GigaLearnCPP/PPG/PPGLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnInfer/Models.h>

#include <cmath>
#include <filesystem>

namespace {
	GGL::PPOLearnerConfig MakeTinyPPOConfig() {
		GGL::PPOLearnerConfig config = {};
		config.tsPerItr = 8;
		config.batchSize = 8;
		config.microBatchSize = 4;
		config.epochs = 1;
		config.policyLR = 1e-3f;
		config.criticLR = 1e-3f;
		config.entropyScale = 0.0f;
		config.minActionProbability = 0.0f;

		config.sharedHead.layerSizes = { 8 };
		config.sharedHead.addOutputLayer = false;

		config.policy.layerSizes = { 8 };
		config.critic.layerSizes = { 8 };
		return config;
	}

	GGL::PPGLearnerConfig MakeTinyPPGConfig() {
		GGL::PPGLearnerConfig config = {};
		config.policyEpochs = 1;
		config.criticEpochs = 1;
		config.auxPhaseInterval = 1;
		config.auxEpochs = 1;
		config.auxBatchSize = 4;
		return config;
	}
}

TEST(PPG, OutputOnlyModelHeadIsValid) {
	GGL::PartialModelConfig partial = {};
	partial.addOutputLayer = true;
	partial.addLayerNorm = false;

	GGL::ModelConfig config = partial;
	config.numInputs = 4;
	config.numOutputs = 1;

	GGL::Model model("aux_critic", config, torch::kCPU);
	torch::Tensor input = torch::randn({ 2, 4 });
	torch::Tensor output = model.Forward(input, false);
	torch::Tensor features = model.ForwardFeatures(input, false);

	ASSERT_EQ(output.size(0), 2);
	ASSERT_EQ(output.size(1), 1);
	ASSERT_EQ(features.size(0), 2);
	ASSERT_EQ(features.size(1), 4);
	ASSERT_EQ(model.GetFeatureSize(), 4);
}

TEST(PPG, UsesFullPolicyAndCriticNetworksWithoutSharedHeads) {
	GGL::PPGLearner learner(4, 3, MakeTinyPPOConfig(), MakeTinyPPGConfig(), torch::kCPU);
	GGL::ModelSet policyModels = learner.GetPolicyModels();

	ASSERT_FALSE(learner.models["shared_head"] != nullptr);
	ASSERT_FALSE(learner.models["critic_shared_head"] != nullptr);
	ASSERT_TRUE(policyModels["policy"] != nullptr);
	ASSERT_FALSE(policyModels["critic"] != nullptr);
	ASSERT_FALSE(policyModels["aux_critic"] != nullptr);
	ASSERT_EQ(learner.models["policy"]->config.numInputs, 4);
	ASSERT_EQ(learner.models["critic"]->config.numInputs, 4);
	ASSERT_EQ(learner.models["aux_critic"]->config.numInputs, 8);
	ASSERT_TRUE(learner.optimizers.Has(*learner.models["policy"]));
	ASSERT_TRUE(learner.optimizers.Has(*learner.models["critic"]));
	ASSERT_TRUE(learner.optimizers.Has(*learner.models["aux_critic"]));
}

TEST(PPG, CheckpointPersistsTrainingOptimizersOutsideModels) {
	const std::filesystem::path folder = RocketForgeTests::TestTempDir();

	GGL::PPGLearner learner(4, 3, MakeTinyPPOConfig(), MakeTinyPPGConfig(), torch::kCPU);
	learner.SaveTo(folder);

	ASSERT_TRUE(std::filesystem::file_size(folder / "POLICY_OPTIM.lt") > 0);
	ASSERT_TRUE(std::filesystem::file_size(folder / "CRITIC_OPTIM.lt") > 0);
	ASSERT_TRUE(std::filesystem::file_size(folder / "AUX_CRITIC_OPTIM.lt") > 0);
	learner.LoadFrom(folder);

	std::filesystem::remove_all(folder);
}

TEST(PPG, LearnRunsPolicyAndAuxiliaryPhases) {
	const int64_t batchSize = 8;
	const int64_t obsSize = 4;
	const int64_t numActions = 3;

	GGL::PPGLearner learner((int)obsSize, (int)numActions, MakeTinyPPOConfig(), MakeTinyPPGConfig(), torch::kCPU);

	GGL::ExperienceBuffer experience(123, torch::kCPU);
	experience.data.states = torch::randn({ batchSize, obsSize });
	experience.data.actionMasks = torch::ones({ batchSize, numActions }, torch::kBool);
	experience.data.actions = torch::randint(0, numActions, { batchSize }, torch::kInt64);
	experience.data.logProbs = torch::full({ batchSize }, -std::log((float)numActions));
	experience.data.advantages = torch::randn({ batchSize });
	experience.data.targetValues = torch::randn({ batchSize });

	GGL::Report report = {};
	learner.Learn(experience, report, false);

	ASSERT_TRUE(report.Has("PPG/Aux Phase Ran"));
	ASSERT_EQ(report["PPG/Aux Phase Ran"], 1.0);
	ASSERT_TRUE(report.Has("PPG/Aux Total Loss"));
	ASSERT_TRUE(std::isfinite(report["PPG/Aux Total Loss"]));
	ASSERT_TRUE(report.Has("LossAndEntropy/Policy Entropy"));
	ASSERT_TRUE(std::isfinite(report["LossAndEntropy/Policy Entropy"]));
	ASSERT_TRUE(report.Has("LossAndEntropy/Policy Loss Abs"));
	ASSERT_TRUE(std::isfinite(report["LossAndEntropy/Policy Loss Abs"]));
	ASSERT_TRUE(report.Has("LossAndEntropy/Entropy Bonus"));
	ASSERT_TRUE(std::isfinite(report["LossAndEntropy/Entropy Bonus"]));
	ASSERT_TRUE(report.Has("LossAndEntropy/Entropy Bonus to Policy Loss Ratio"));
	ASSERT_TRUE(std::isfinite(report["LossAndEntropy/Entropy Bonus to Policy Loss Ratio"]));
}

TEST(PPG, DefaultAuxBatchUsesPPOBatchSize) {
	const int64_t batchSize = 8;
	const int64_t obsSize = 4;
	const int64_t numActions = 3;

	GGL::PPOLearnerConfig ppoConfig = MakeTinyPPOConfig();
	GGL::PPGLearnerConfig ppgConfig = MakeTinyPPGConfig();
	ppgConfig.auxBatchSize = 0;
	GGL::PPGLearner learner((int)obsSize, (int)numActions, ppoConfig, ppgConfig, torch::kCPU);

	GGL::ExperienceBuffer experience(123, torch::kCPU);
	experience.data.states = torch::randn({ batchSize, obsSize });
	experience.data.actionMasks = torch::ones({ batchSize, numActions }, torch::kBool);
	experience.data.actions = torch::randint(0, numActions, { batchSize }, torch::kInt64);
	experience.data.logProbs = torch::full({ batchSize }, -std::log((float)numActions));
	experience.data.advantages = torch::randn({ batchSize });
	experience.data.targetValues = torch::randn({ batchSize });

	GGL::Report report = {};
	learner.Learn(experience, report, false);

	ASSERT_TRUE(report.Has("PPG/Aux Batch Size"));
	ASSERT_EQ(report["PPG/Aux Batch Size"], (double)ppoConfig.batchSize);
	ASSERT_TRUE(report.Has("PPG/Aux Optim Steps"));
	ASSERT_EQ(report["PPG/Aux Optim Steps"], 1.0);
}
