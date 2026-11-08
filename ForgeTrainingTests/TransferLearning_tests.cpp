#include <gtest/gtest.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/TransferLearning.h>
#include <GigaLearnInfer/ModelManager.h>
#include <torch/cuda.h>
#include <cstdlib>

namespace {
	GGL::PPOLearnerConfig TinyConfig() {
		GGL::PPOLearnerConfig config;
		config.batchSize = 12;
		config.microBatchSize = 4;
		config.useHalfPrecision = false;
		config.sharedHead.layerSizes = { 8 };
		config.sharedHead.addOutputLayer = false;
		config.policy.layerSizes = { 8 };
		config.critic.layerSizes = { 8 };
		config.minActionProbability = 0;
		return config;
	}

	void CopyModels(GGL::PPOLearner& from, GGL::PPOLearner& to) {
		torch::NoGradGuard noGrad;
		for (auto* model : from.models) {
			std::vector<torch::Tensor> sourceParams = model->parameters(), destinationParams = to.models[model->modelName]->parameters();
			for (size_t i = 0; i < sourceParams.size(); i++) destinationParams[i].copy_(sourceParams[i]);
		}
	}

	std::vector<torch::Device> Devices() {
		std::vector<torch::Device> devices{ torch::kCPU };
		if (const char* enabled = std::getenv("ROCKETFORGE_TEST_GPU"); enabled && std::string(enabled) == "1") {
			TORCH_CHECK(torch::cuda::is_available(), "GPU transfer checks requested without CUDA");
			devices.emplace_back(torch::kCUDA);
		}

		return devices;
	}

	class CountingModel : public GGL::Model {
	public:
		int64_t largestBatch = 0, calls = 0;

		CountingModel(GGL::Model& source) : Model(source.modelName, source.config, source.device) {
			torch::NoGradGuard noGrad;
			std::vector<torch::Tensor> sourceParams = source.parameters(), destinationParams = parameters();
			for (size_t i = 0; i < sourceParams.size(); i++) destinationParams[i].copy_(sourceParams[i]);
		}

		torch::Tensor Forward(torch::Tensor input, bool half) override {
			largestBatch = std::max(largestBatch, input.size(0));
			calls++;
			return Model::Forward(input, half);
		}
	};

	CountingModel* Track(GGL::PPOLearner& learner) {
		GGL::Model* original = learner.models["policy"];
		CountingModel* tracked = new CountingModel(*original);

		learner.optimizers.Clear();
		learner.models.map["policy"] = tracked;
		delete original;
		learner.optimizers.Attach(learner.models);
		return tracked;
	}
}

TEST(TransferLearning, KnownLossesUseCategoricalTargets) {
	torch::Tensor logits = torch::tensor({ { 0.f, 0.f, 12.f } }).requires_grad_(true);
	GGL::TransferLearning::Targets target{
		torch::tensor({ { .75f, .25f, 0.f } }), torch::tensor({ { true, true, false } }) };
	torch::Tensor logProbs = GGL::TransferLearning::ConditionalLogProbs(logits, target.mask);
	GGL::TransferLearnConfig config;
	config.lossScale = 1;
	config.useKLDiv = true;

	const torch::Tensor klLoss = GGL::TransferLearning::LossRows(logProbs, target, config);
	EXPECT_NEAR(klLoss.item<float>(), .75 * std::log(1.5) + .25 * std::log(.5), 1e-6);
	klLoss.sum().backward();
	EXPECT_NEAR(logits.grad()[0][0].item<float>(), -.25, 1e-6);
	EXPECT_NEAR(logits.grad()[0][1].item<float>(), .25, 1e-6);
	EXPECT_EQ(logits.grad()[0][2].item<float>(), 0);

	config.useHardActionLabels = true;
	EXPECT_NEAR(GGL::TransferLearning::LossRows(logProbs, target, config).item<float>(), std::log(2.), 1e-6);

	config.useHardActionLabels = false;
	config.useKLDiv = false;
	EXPECT_NEAR(GGL::TransferLearning::LossRows(logProbs, target, config).item<float>(), .5 / 3, 1e-6);
}

TEST(TransferLearning, MappingRespectsBothMasksAndExtraActions) {
	torch::Tensor teacher = torch::tensor({ { .2f, .3f, .5f }, { .2f, .3f, .5f } });
	torch::Tensor map = torch::tensor({ { 2, -1, 0, 1 }, { -1, -1, -1, -1 } }, torch::kInt64);
	torch::Tensor teacherMask = torch::tensor({ { true, false, true }, { true, true, true } });
	torch::Tensor studentMask = torch::ones({ 2, 4 }, torch::kBool);

	GGL::TransferLearning::Targets target = GGL::TransferLearning::MapTargets(teacher, teacherMask, studentMask, map);
	EXPECT_TRUE(torch::allclose(target.probabilities[0], torch::tensor({ 5.f / 7, 0.f, 2.f / 7, 0.f })));
	EXPECT_EQ(target.mask[0].sum().item<int>(), 2);
	EXPECT_EQ(target.mask[1].sum().item<int>(), 0);
	EXPECT_EQ(target.probabilities[1].sum().item<float>(), 0);

	for (int mode = 0; mode < 3; mode++) {
		torch::Tensor logits = torch::tensor({ { -1000.f, 2000.f, 1000.f, 20.f }, { 5.f, 4.f, 3.f, 2.f } }).requires_grad_(true);
		GGL::TransferLearnConfig config;
		config.useKLDiv = mode == 1;
		config.useHardActionLabels = mode == 2;
		config.lossExponent = .5f;

		torch::Tensor logProbs = GGL::TransferLearning::ConditionalLogProbs(logits, target.mask);
		torch::Tensor loss = GGL::TransferLearning::LossRows(logProbs, target, config);
		EXPECT_TRUE(torch::isfinite(loss).all().item<bool>());
		EXPECT_EQ(loss[1].item<float>(), 0);

		loss.sum().backward();
		EXPECT_TRUE(torch::isfinite(logits.grad()).all().item<bool>());
		EXPECT_EQ(logits.grad()[0][1].item<float>(), 0);
		EXPECT_EQ(logits.grad()[0][3].item<float>(), 0);
		EXPECT_EQ(logits.grad()[1].abs().sum().item<float>(), 0);
	}
}

TEST(TransferLearning, TinySharedTeacherMassStillNormalizes) {
	torch::Tensor teacher = torch::tensor({ { 1e-35f, 1.f } });
	torch::Tensor mask = torch::tensor({ { true, false } });
	GGL::TransferLearning::Targets target = GGL::TransferLearning::MapTargets(teacher, torch::ones_like(mask), mask, {});
	EXPECT_EQ(target.probabilities[0][0].item<float>(), 1);
	EXPECT_EQ(target.probabilities.sum().item<float>(), 1);
}

TEST(TransferLearning, ExactMatchAndSingleLegalActionHaveFiniteZeroLoss) {
	for (float exponent : { .5f, 1.f, 2.f }) {
		torch::Tensor logits = torch::zeros({ 2, 3 }).requires_grad_(true);
		torch::Tensor mask = torch::tensor({ { true, true, false }, { false, true, false } });
		torch::Tensor teacher = torch::tensor({ { .5f, .5f, 0.f }, { 0.f, 1.f, 0.f } });
		GGL::TransferLearning::Targets target = GGL::TransferLearning::MapTargets(teacher, mask, mask, {});
		GGL::TransferLearnConfig config;
		config.useKLDiv = true;
		config.lossExponent = exponent;

		torch::Tensor loss = GGL::TransferLearning::LossRows(GGL::TransferLearning::ConditionalLogProbs(logits, mask), target, config);
		EXPECT_EQ(loss.sum().item<float>(), 0);
		loss.sum().backward();
		EXPECT_TRUE(torch::isfinite(logits.grad()).all().item<bool>());
	}
}

class TransferBatching : public ::testing::TestWithParam<int> {};

TEST_P(TransferBatching, MatchesFullBatchAndBoundsBothModels) {
	for (const torch::Device& device : Devices()) {
		SCOPED_TRACE(device.str());
		torch::manual_seed(712);

		GGL::PPOLearnerConfig studentConfig = TinyConfig();
		studentConfig.minActionProbability = .01f;
		GGL::PPOLearner full(4, 4, studentConfig, device), micro(4, 4, studentConfig, device), teacher(6, 3, TinyConfig(), device);
		CopyModels(full, micro);
		CountingModel* trackedStudent = Track(micro);
		CountingModel* trackedTeacher = Track(teacher);

		GGL::ModelSet oldModels = teacher.GetPolicyModels();
		torch::Tensor teacherBefore = teacher.models["policy"]->CopyParams();
		torch::Tensor criticBefore = micro.models["critic"]->CopyParams();
		torch::Tensor studentBefore = micro.models["policy"]->CopyParams();
		torch::Tensor sharedBefore = micro.models["shared_head"]->CopyParams();

		torch::Tensor newObs = torch::randn({ 11, 4 }), oldObs = torch::randn({ 11, 6 });
		torch::Tensor newMask = torch::ones({ 11, 4 }, torch::kBool), oldMask = torch::ones({ 11, 3 }, torch::kBool);
		torch::Tensor map = torch::tensor({ 2, -1, 0, 1 }, torch::kInt64).repeat({ 11, 1 });
		map[2].fill_(-1);
		oldMask[4][1] = false;
		newMask[5][0] = false;

		GGL::TransferLearnConfig config;
		config.useKLDiv = GetParam() == 1;
		config.useHardActionLabels = GetParam() == 2;
		config.lossScale = 1;
		config.lr = .002f;
		config.epochs = 3;

		GGL::Report fullReport, microReport;
		full.TransferLearn(oldModels, newObs, oldObs, newMask, oldMask, map, fullReport, config);
		trackedTeacher->largestBatch = trackedTeacher->calls = 0;
		config.microBatchSize = 3;
		micro.TransferLearn(oldModels, newObs, oldObs, newMask, oldMask, map, microReport, config);

		EXPECT_EQ(trackedStudent->largestBatch, 3);
		EXPECT_EQ(trackedTeacher->largestBatch, 3);
		EXPECT_EQ(trackedStudent->calls, 12);
		EXPECT_EQ(trackedTeacher->calls, 4);
		EXPECT_EQ(microReport["Transfer/Optimizer Steps"], 3);
		EXPECT_EQ(microReport["Transfer/Supervised Rows"], 10);

		for (const char* metric : { "Transfer Learn Loss", "Transfer Learn Accuracy", "LossAndEntropy/Policy Entropy", "LossAndEntropy/Old Policy Entropy" })
			EXPECT_NEAR(fullReport[metric], microReport[metric], 2e-6) << metric;

		for (auto* model : full.GetPolicyModels())
			EXPECT_TRUE(torch::allclose(model->CopyParams(), micro.models[model->modelName]->CopyParams(), 2e-5, 2e-6)) << model->modelName;

		EXPECT_TRUE(torch::equal(teacherBefore, teacher.models["policy"]->CopyParams()));
		EXPECT_TRUE(torch::equal(criticBefore, micro.models["critic"]->CopyParams()));
		EXPECT_FALSE(torch::equal(studentBefore, micro.models["policy"]->CopyParams()));
		EXPECT_FALSE(torch::equal(sharedBefore, micro.models["shared_head"]->CopyParams()));

		for (auto* model : oldModels)
			for (auto& parameter : model->parameters()) EXPECT_FALSE(parameter.grad().defined());
	}
}

INSTANTIATE_TEST_SUITE_P(LossModes, TransferBatching, ::testing::Values(0, 1, 2));

TEST(TransferLearning, ConditionalProbabilitiesMatchRuntimeForIdenticalSpaces) {
	GGL::PPOLearner student(4, 3, TinyConfig(), torch::kCPU);
	torch::Tensor obs = torch::randn({ 5, 4 });
	torch::Tensor masks = torch::ones({ 5, 3 }, torch::kBool);
	masks[1][2] = false;
	masks[2].fill_(false);
	masks[2][1] = true;
	masks[3].fill_(false);

	torch::NoGradGuard noGrad;
	const bool halfPrecision = false;
	const float temperature = 1.3f;
	torch::Tensor hidden = student.models["shared_head"]->Forward(obs, halfPrecision);
	torch::Tensor logits = student.models["policy"]->Forward(hidden, halfPrecision) / temperature;

	for (float floor : { 0.f, .01f }) {
		torch::Tensor expected =
			GGL::ModelManager::InferPolicyProbsFromModels(student.models, obs, masks, temperature, halfPrecision, floor);
		torch::Tensor actual = GGL::TransferLearning::ConditionalLogProbs(logits, masks, floor).exp() * masks;
		EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-7));
	}
}

TEST(TransferLearning, UnlabeledBatchDoesNotAdvanceMomentumOrDecay) {
	GGL::PPOLearnerConfig studentConfig = TinyConfig();
	studentConfig.policy.optimType = GGL::ModelOptimType::ADAMW;
	GGL::PPOLearner student(4, 3, studentConfig, torch::kCPU), teacher(4, 3, TinyConfig(), torch::kCPU);
	torch::Tensor obs = torch::randn({ 5, 4 });
	torch::Tensor masks = torch::ones({ 5, 3 }, torch::kBool);
	GGL::ModelSet oldModels = teacher.GetPolicyModels();
	GGL::TransferLearnConfig config;
	config.microBatchSize = 2;
	config.epochs = 1;

	GGL::Report report;
	student.TransferLearn(oldModels, obs, obs, masks, masks, {}, report, config);
	torch::Tensor before = student.models["policy"]->CopyParams();
	student.TransferLearn(oldModels, obs, obs, masks, masks, torch::full({ 5, 3 }, -1, torch::kInt64), report, config);

	EXPECT_TRUE(torch::equal(before, student.models["policy"]->CopyParams()));
	EXPECT_EQ(report["Transfer/Optimizer Steps"], 0);
	EXPECT_EQ(report["Transfer/Skipped Rows"], 5);
}

TEST(TransferLearning, RejectsInvalidLaterRowsBeforeChangingParameters) {
	GGL::PPOLearner student(4, 3, TinyConfig(), torch::kCPU), teacher(4, 3, TinyConfig(), torch::kCPU);
	torch::Tensor obs = torch::randn({ 5, 4 }), masks = torch::ones({ 5, 3 }, torch::kBool);
	GGL::ModelSet oldModels = teacher.GetPolicyModels();
	GGL::TransferLearnConfig config;
	config.microBatchSize = 2;
	GGL::Report report;
	torch::Tensor before = student.models["policy"]->CopyParams();

	for (int badIndex : { -2, 3, 0 }) {
		torch::Tensor map = torch::arange(3, torch::kInt64).repeat({ 5, 1 });
		map[4][2] = badIndex;
		EXPECT_THROW(student.TransferLearn(oldModels, obs, obs, masks, masks, map, report, config), c10::Error);
	}

	EXPECT_THROW(student.TransferLearn(oldModels, obs, obs.slice(0, 0, 4), masks, masks, {}, report, config), c10::Error);

	config.microBatchSize = -1;
	EXPECT_THROW(student.TransferLearn(oldModels, obs, obs, masks, masks, {}, report, config), c10::Error);

	config.microBatchSize = 2;
	config.lossExponent = std::numeric_limits<float>::quiet_NaN();
	EXPECT_THROW(student.TransferLearn(oldModels, obs, obs, masks, masks, {}, report, config), c10::Error);

	EXPECT_TRUE(torch::equal(before, student.models["policy"]->CopyParams()));
}
