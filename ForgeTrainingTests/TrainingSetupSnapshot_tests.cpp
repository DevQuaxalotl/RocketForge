#include <gtest/gtest.h>
#include "TestSupport.h"

#include "Storage/BuildStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <TrainingStage.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>

TEST(TrainingSetupSnapshot, SerializeRoundTrip) {
	RocketForge::TrainingSetupSnapshot snapshot = {};
	snapshot.agentName = "example-agent";
	snapshot.stageName = "ball-chase";
	snapshot.versionId = "v1.0";
	snapshot.savedAt = "2026-07-19";

	RocketForge::RewardTableEntry reward = {};
	reward.name = "TouchBall";
	reward.weight = 2.f;
	reward.effectiveMax = 4.f;
	reward.range.minValue = -1.f;
	reward.range.maxValue = 1.f;
	reward.category = RLGC::RewardCategory::EVENT;
	snapshot.rewards.push_back(reward);

	snapshot.hyperparameters.push_back({ "PPO", "entropy", "0.01" });

	nlohmann::json j;
	GGL::JsonSave(snapshot, j);

	ASSERT_EQ(j["agent"].get<std::string>(), "example-agent");
	ASSERT_EQ(j["rewards"][0]["category"].get<std::string>(), "event");
	ASSERT_EQ(j["rewards"][0]["range"]["min"].get<float>(), -1.f);
	ASSERT_EQ(j["hyperparameters"][0]["group"].get<std::string>(), "PPO");
	ASSERT_TRUE(j["agent_setup"].is_null());

	RocketForge::TrainingSetupSnapshot loaded = {};
	std::vector<std::string> errors = GGL::JsonLoad(j, loaded);
	ASSERT_TRUE(errors.empty());
	ASSERT_TRUE(loaded.rewards[0].category == RLGC::RewardCategory::EVENT);
	ASSERT_EQ(loaded.rewards[0].range.maxValue, 1.f);
}

namespace {
	constexpr const char* SNAPSHOT_TEST_AGENT_NAME = "test-advantage-norm-agent";

	std::unique_ptr<Agent> MakeSnapshotTestAgent() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = SNAPSHOT_TEST_AGENT_NAME;
		return agent;
	}

	class SnapshotTestStage : public RocketForge::TrainingStage {
	public:
		SnapshotTestStage() : TrainingStage(MakeSnapshotTestAgent()) {
			AgentRegistry::Register(SNAPSHOT_TEST_AGENT_NAME, MakeSnapshotTestAgent);
		}
	};

	constexpr const char* DECOMPOSED_TEST_AGENT_NAME = "test-decomposed-critic-agent";

	std::unique_ptr<Agent> MakeDecomposedCriticTestAgent() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = DECOMPOSED_TEST_AGENT_NAME;
		agent->decomposedCritic = true;
		return agent;
	}

	class DecomposedCriticTestStage : public RocketForge::TrainingStage {
	public:
		DecomposedCriticTestStage() : TrainingStage(MakeDecomposedCriticTestAgent()) {
			AgentRegistry::Register(DECOMPOSED_TEST_AGENT_NAME, MakeDecomposedCriticTestAgent);
		}
	};

	std::optional<std::string> FindTrainingParameter(const RocketForge::TrainingSetupSnapshot& snapshot, const std::string& name) {
		for (const RocketForge::TrainingSetupParameter& parameter : snapshot.hyperparameters)
			if (parameter.group == "Training" && parameter.name == name)
				return parameter.value;

		return std::nullopt;
	}
}

TEST(TrainingSetupSnapshot, OmitsAdvantageNormalizationParametersAtDefaults) {
	SnapshotTestStage stage;
	RocketForge::TrainingSetupSnapshot snapshot = RocketForge::MakeTrainingSetupSnapshot(&stage);

	ASSERT_FALSE(FindTrainingParameter(snapshot, "Normalize Advantages").has_value());
	ASSERT_FALSE(FindTrainingParameter(snapshot, "Advantage Norm Epsilon").has_value());
	ASSERT_FALSE(FindTrainingParameter(snapshot, "Policy Grad Clip Norm").has_value());
}

TEST(TrainingSetupSnapshot, RecordsAdvantageNormalizationWhenEnabled) {
	SnapshotTestStage stage;
	stage.learnerConfig.ppo.normalizeAdvantages = true;
	stage.learnerConfig.ppo.policyGradClipNorm = 1.7f;
	RocketForge::TrainingSetupSnapshot snapshot = RocketForge::MakeTrainingSetupSnapshot(&stage);

	ASSERT_EQ(FindTrainingParameter(snapshot, "Normalize Advantages").value_or(""), std::string("true"));
	ASSERT_EQ(FindTrainingParameter(snapshot, "Advantage Norm Epsilon").value_or(""), std::string("1e-08"));
	ASSERT_EQ(FindTrainingParameter(snapshot, "Policy Grad Clip Norm").value_or(""), std::string("1.7"));
}

TEST(TrainingStage, CopiesDecomposedCriticFromAgent) {
	SnapshotTestStage defaultStage;
	DecomposedCriticTestStage decomposedStage;

	ASSERT_FALSE(defaultStage.learnerConfig.ppo.decomposedCritic.enabled);
	ASSERT_TRUE(decomposedStage.learnerConfig.ppo.decomposedCritic.enabled);
}

TEST(TrainingSetupSnapshot, OmitsDecomposedCriticParametersWhenOff) {
	SnapshotTestStage stage;
	RocketForge::TrainingSetupSnapshot snapshot = RocketForge::MakeTrainingSetupSnapshot(&stage);

	ASSERT_FALSE(FindTrainingParameter(snapshot, "Critic Head Loss Weighting").has_value());
	ASSERT_FALSE(FindTrainingParameter(snapshot, "PopArt Beta").has_value());
	ASSERT_FALSE(FindTrainingParameter(snapshot, "PopArt Sigma Floor").has_value());
}

TEST(TrainingSetupSnapshot, RecordsDecomposedCriticParametersWhenOn) {
	SnapshotTestStage stage;
	stage.learnerConfig.ppo.decomposedCritic.enabled = true;
	stage.learnerConfig.ppo.decomposedCritic.lossWeighting = GGL::CriticHeadLossWeighting::Contribution;
	RocketForge::TrainingSetupSnapshot snapshot = RocketForge::MakeTrainingSetupSnapshot(&stage);

	ASSERT_EQ(FindTrainingParameter(snapshot, "Critic Head Loss Weighting").value_or(""), std::string("Contribution"));
	ASSERT_EQ(FindTrainingParameter(snapshot, "PopArt Beta").value_or(""), std::string("0.02"));
	ASSERT_EQ(FindTrainingParameter(snapshot, "PopArt Sigma Floor").value_or(""), std::string("0.01"));
}

TEST(TrainingSetupSnapshot, DetailedHeadMetricsDoNotChangeSetupHash) {
	SnapshotTestStage stage;
	stage.learnerConfig.ppo.decomposedCritic.enabled = true;
	const std::string hashBefore = RocketForgeTraining::BuildStorage::MakeSetupHash(RocketForge::MakeTrainingSetupSnapshot(&stage));

	stage.learnerConfig.ppo.decomposedCritic.logDetailedHeadMetrics = true;
	const std::string hashAfter = RocketForgeTraining::BuildStorage::MakeSetupHash(RocketForge::MakeTrainingSetupSnapshot(&stage));

	ASSERT_EQ(hashBefore, hashAfter);
}

TEST(TrainingSetupSnapshot, SkillTestParametersDoNotAffectIdentity) {
	RocketForge::TrainingSetupParameter parameter = {};
	parameter.group = "Skill Test";
	parameter.name = "Target Confidence";
	ASSERT_FALSE(RocketForge::TrainingSetupParameterAffectsIdentity(parameter));
}

TEST(TrainingSetupSnapshot, SkillTestTuningDoesNotChangeSetupHash) {
	SnapshotTestStage stage;
	const std::string hashBefore = RocketForgeTraining::BuildStorage::MakeSetupHash(RocketForge::MakeTrainingSetupSnapshot(&stage));

	stage.learnerConfig.skillTracker.targetConfidence = 0.98f;
	stage.learnerConfig.skillTracker.referenceGamesPerOpponent = 8;
	stage.learnerConfig.skillTracker.referenceCalibrationGames = 8;
	stage.learnerConfig.skillTracker.ratingDriftSigmaPer100MTimesteps = 10.0f;
	stage.learnerConfig.skillTracker.anchorRetireExpectedWinProb = 0.8f;
	const std::string hashAfter = RocketForgeTraining::BuildStorage::MakeSetupHash(RocketForge::MakeTrainingSetupSnapshot(&stage));

	ASSERT_EQ(hashBefore, hashAfter);
}

TEST(TrainingSetupSnapshot, SetupHashDistinguishesSimulatorsButNotRevisions) {
	RocketForge::TrainingSetupSnapshot first, second;
	first.engineVersion = second.engineVersion = 2;
	first.engineRevision = "before";
	second.engineRevision = "after";
	EXPECT_EQ(RocketForgeTraining::BuildStorage::MakeSetupHash(first), RocketForgeTraining::BuildStorage::MakeSetupHash(second));

	second.engineVersion = 3;
	EXPECT_NE(RocketForgeTraining::BuildStorage::MakeSetupHash(first), RocketForgeTraining::BuildStorage::MakeSetupHash(second));
	EXPECT_FALSE(RocketForge::DiffTrainingSetupSnapshots(first, second).empty());
}

TEST(TrainingSetupSnapshot, LoadRejectsCorruptEngineFields) {
	RocketForge::TrainingSetupSnapshot snapshot;
	snapshot.engineVersion = 3;
	snapshot.agentSetup = RocketForge::CaptureAgentSetupSnapshot(Agent{});
	nlohmann::json saved;
	GGL::JsonSave(snapshot, saved);

	struct Case {
		const char* label;
		std::function<void(nlohmann::json&)> corrupt;
	};
	const Case cases[] = {
		{ "null engine_version", [](nlohmann::json& j) { j["engine_version"] = nullptr; } },
		{ "string engine_version", [](nlohmann::json& j) { j["engine_version"] = "3"; } },
		{ "null training engine", [](nlohmann::json& j) { j["agent_setup"]["training_engine_version"] = nullptr; } },
		{ "string prediction engine", [](nlohmann::json& j) { j["agent_setup"]["ball_prediction_engine_version"] = "2"; } },
	};

	const std::filesystem::path path = RocketForgeTests::TestTempDir() / RocketForge::TrainingSetupSnapshot::FILE_NAME;
	std::filesystem::create_directories(path.parent_path());
	std::ofstream(path) << saved.dump();
	ASSERT_TRUE(RocketForge::LoadTrainingSetupSnapshot(path));

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		nlohmann::json corrupted = saved;
		testCase.corrupt(corrupted);
		std::ofstream(path) << corrupted.dump();

		std::string error;
		EXPECT_FALSE(RocketForge::LoadTrainingSetupSnapshot(path, &error));
		EXPECT_NE(error.find("engine_version"), std::string::npos) << error;
	}
}

TEST(TrainingSetupSnapshot, SavedSetupsFailLoudlyAndLegacyOnesResolveTheirEngine) {
	const std::filesystem::path path = RocketForgeTests::TestTempDir() / RocketForge::TrainingSetupSnapshot::FILE_NAME;
	std::filesystem::create_directories(path.parent_path());
	RocketForge::AgentSetupSnapshot current = RocketForge::CaptureAgentSetupSnapshot(Agent{});
	current.trainingEngineVersion = 3;
	current.ballPredictionEngineVersion = 3;

	EXPECT_FALSE(RocketForge::LoadExistingTrainingSetupSnapshot(path));
	const RocketForge::AgentSetupSnapshot unrecorded = RocketForge::ResolveSavedAgentSetup(std::nullopt, current);
	EXPECT_EQ(unrecorded.trainingEngineVersion, 2);
	EXPECT_EQ(unrecorded.ballPredictionEngineVersion, 2);

	std::ofstream(path) << "broken";
	EXPECT_THROW(RocketForge::LoadExistingTrainingSetupSnapshot(path), std::runtime_error);

	current.trainingEngineVersion = 2;
	current.ballPredictionEngineVersion = 2;
	std::ofstream(path) << nlohmann::json{ { "engine_version", 3 } }.dump();
	const std::optional<RocketForge::TrainingSetupSnapshot> legacy = RocketForge::LoadExistingTrainingSetupSnapshot(path);
	ASSERT_TRUE(legacy);
	const RocketForge::AgentSetupSnapshot resolved = RocketForge::ResolveSavedAgentSetup(legacy, current);
	EXPECT_EQ(resolved.trainingEngineVersion, 3);
	EXPECT_EQ(resolved.ballPredictionEngineVersion, 3);
}
