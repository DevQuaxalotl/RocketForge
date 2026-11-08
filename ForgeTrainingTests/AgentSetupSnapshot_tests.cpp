#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Storage/BuildStorage.h"
#include "ForgeTraining.h"
#include "StageRegistry.h"
#include "StaticOpponents/StaticOpponentSelectionInternal.h"
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include "TrainingSetup/TrainingSetupSnapshot.h"
#include "TrainingSetup/VersionManagement.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace RocketForge;
using namespace RocketForgeTraining;

namespace
{
	namespace fs = std::filesystem;

	Agent MakeConfiguredAgent()
	{
		Agent agent;
		agent.agentName = "snapshot-agent";
		agent.gameMode = RocketSim::GameMode::HOOPS;
		agent.carConfig = RocketSim::CAR_CONFIG_DOMINUS;
		agent.tickSkip = 4;
		agent.actionDelay = 3;
		agent.maxTeamSize = 3;
		agent.useBallPrediction = true;
		agent.addLayerNorm = false;
		agent.otimType = GGL::ModelOptimType::ADAMW;
		agent.modelActivationType = GGL::ModelActivationType::TANH;
		agent.sharedHead.layerSizes = { 128, 64 };
		agent.sharedHead.addOutputLayer = false;
		agent.policy.layerSizes = { 256, 128 };
		agent.policy.activationType = GGL::ModelActivationType::LEAKY_RELU;
		agent.critic.layerSizes = { 512, 256 };
		agent.critic.optimType = GGL::ModelOptimType::RMSPROP;
		return agent;
	}

	class ExportTestStage : public TrainingStage
	{
	public:
		ExportTestStage(std::unique_ptr<Agent> agent, const std::string& name)
			: TrainingStage(std::move(agent))
		{
			stageName = name;
		}
	};

	fs::path TempAgentRoot(const std::string& name)
	{
		return RocketForgeTests::TestTempDir(name);
	}

	void WriteTextFile(const fs::path& path, const std::string& text)
	{
		fs::create_directories(path.parent_path());
		std::ofstream fOut(path);
		fOut << text;
	}

	std::string ReadTextFile(const fs::path& path)
	{
		std::ifstream fIn(path);
		std::stringstream buffer;
		buffer << fIn.rdbuf();
		return buffer.str();
	}

}

TEST(AgentSetupSnapshot, CaptureApplyAndJsonRoundTrip)
{
	Agent source = MakeConfiguredAgent();
	AgentSetupSnapshot captured = CaptureAgentSetupSnapshot(source);
	nlohmann::json capturedJson = AgentSetupSnapshotToJson(captured);
	ASSERT_EQ(capturedJson.value("car_config", std::string()), std::string("Dominus"));
	ASSERT_EQ(capturedJson.value("obs_builder", std::string()), std::string("DefaultObsPadded"));
	ASSERT_EQ(capturedJson.value("action_parser", std::string()), std::string("DefaultAction"));

	std::optional<AgentSetupSnapshot> loaded =
		AgentSetupSnapshotFromJson(capturedJson);
	ASSERT_TRUE(loaded.has_value());
	ASSERT_TRUE(SameAgentSetupSnapshot(captured, loaded.value()));

	Agent target;
	ApplyAgentSetupSnapshot(target, loaded.value());
	ASSERT_TRUE(SameAgentSetupSnapshot(captured, CaptureAgentSetupSnapshot(target)));
}

TEST(AgentSetupSnapshot, RetiredPredictionRateOnlyFeedsTheSetupHash)
{
	const AgentSetupSnapshot captured = CaptureAgentSetupSnapshot(MakeConfiguredAgent());

	const nlohmann::json snapshotJson = AgentSetupSnapshotToJson(captured);
	EXPECT_TRUE(snapshotJson.find("ball_pred_tick_rate") == snapshotJson.end());

	const nlohmann::json hashJson = AgentSetupSnapshotToHashJson(captured);
	ASSERT_TRUE(hashJson.contains("ball_pred_tick_rate"));
	EXPECT_EQ(hashJson.at("ball_pred_tick_rate").dump(), "0.0");

	nlohmann::json legacyJson = snapshotJson;
	legacyJson["ball_pred_tick_rate"] = 60.f;
	const std::optional<AgentSetupSnapshot> legacy = AgentSetupSnapshotFromJson(legacyJson);
	ASSERT_TRUE(legacy.has_value());
	EXPECT_TRUE(SameAgentSetupSnapshot(captured, legacy.value()));
	EXPECT_EQ(AgentSetupSnapshotToJson(legacy.value()), snapshotJson);
}

TEST(StaticOpponentSelection, PersistsExactSourceAgentSetup)
{
	const fs::path root = TempAgentRoot("static_opponent_selection");
	const fs::path selectionPath = root / "static_opponents.json";

	Agent source = MakeConfiguredAgent();
	RocketForgeTraining::StaticOpponentSelectionEntry entry = {};
	entry.sourceAgentName = source.agentName;
	entry.sourceBuildName = "V2";
	entry.sourceStageName = "ground";
	entry.sourceVersionId = "v2.05_ground";
	entry.displayName = "Snapshot source";
	entry.checkpointPath = root / "205000000";
	entry.agentSetup = CaptureAgentSetupSnapshot(source);

	RocketForgeTraining::StaticOpponentSelectionInternal::SaveSelectionToPath(
		selectionPath,
		"target-agent",
		"V1",
		{ entry });

	nlohmann::json savedJson;
	std::ifstream(selectionPath) >> savedJson;
	ASSERT_EQ(savedJson.value("schema_version", 0), 2);
	ASSERT_EQ(
		savedJson["static_opponents"][0].value("runtime", std::string()),
		std::string("source_agent"));
	ASSERT_TRUE(savedJson["static_opponents"][0].contains("agent_setup"));

	const auto loaded = RocketForgeTraining::StaticOpponentSelectionInternal::LoadSelectionFromPath(
		selectionPath);
	ASSERT_EQ(loaded.size(), size_t(1));
	ASSERT_TRUE(loaded.front().agentSetup.has_value());
	ASSERT_TRUE(SameAgentSetupSnapshot(
		entry.agentSetup.value(),
		loaded.front().agentSetup.value()));

	fs::remove_all(root);
}

TEST(AgentSetupSnapshot, DiffReportsSavedToCurrentValues)
{
	Agent savedAgent = MakeConfiguredAgent();
	Agent currentAgent = savedAgent;
	currentAgent.tickSkip = 8;
	currentAgent.policy.layerSizes = { 1024, 512 };

	std::vector<AgentSetupDiff> diffs = DiffAgentSetupSnapshots(
		CaptureAgentSetupSnapshot(savedAgent),
		CaptureAgentSetupSnapshot(currentAgent));

	bool sawTickSkip = false;
	bool sawPolicyLayers = false;
	for (const AgentSetupDiff& diff : diffs) {
		if (diff.field == "Tick Skip") {
			sawTickSkip = true;
			ASSERT_EQ(diff.savedValue, std::string("4"));
			ASSERT_EQ(diff.currentValue, std::string("8"));
		}
		if (diff.field == "Policy Layers") {
			sawPolicyLayers = true;
			ASSERT_EQ(diff.savedValue, std::string("[256, 128]"));
			ASSERT_EQ(diff.currentValue, std::string("[1024, 512]"));
		}
	}

	ASSERT_TRUE(sawTickSkip);
	ASSERT_TRUE(sawPolicyLayers);
}

TEST(AgentSetupSnapshot, DiffReportsSavedComponentNames)
{
	Agent savedAgent = MakeConfiguredAgent();
	Agent currentAgent = savedAgent;
	currentAgent.obsBuilderName = "AdvancedObs";
	currentAgent.actionParserName = "DifferentAction";

	std::vector<AgentSetupDiff> diffs = DiffAgentSetupSnapshots(
		CaptureAgentSetupSnapshot(savedAgent),
		CaptureAgentSetupSnapshot(currentAgent));

	bool sawObsBuilder = false;
	bool sawActionParser = false;
	for (const AgentSetupDiff& diff : diffs) {
		if (diff.field == "Observation Builder") {
			sawObsBuilder = true;
			ASSERT_EQ(diff.savedValue, std::string("DefaultObsPadded"));
			ASSERT_EQ(diff.currentValue, std::string("AdvancedObs"));
		}
		if (diff.field == "Action Parser") {
			sawActionParser = true;
			ASSERT_EQ(diff.savedValue, std::string("DefaultAction"));
			ASSERT_EQ(diff.currentValue, std::string("DifferentAction"));
		}
	}

	ASSERT_TRUE(sawObsBuilder);
	ASSERT_TRUE(sawActionParser);
}

TEST(AgentSetupSnapshot, MissingComponentNamesIsInvalid)
{
	Agent source = MakeConfiguredAgent();
	const nlohmann::json completeJson = AgentSetupSnapshotToJson(CaptureAgentSetupSnapshot(source));

	// Either component name missing makes the snapshot unusable, not only both.
	const std::vector<std::vector<std::string>> missingKeySets = { { "obs_builder", "action_parser" }, { "obs_builder" } };

	for (const std::vector<std::string>& missingKeys : missingKeySets) {
		nlohmann::json legacyJson = completeJson;
		for (const std::string& key : missingKeys)
			legacyJson.erase(key);

		std::optional<AgentSetupSnapshot> loaded = AgentSetupSnapshotFromJson(legacyJson);
		ASSERT_FALSE(loaded.has_value());
	}
}

TEST(AgentSetupSnapshot, LoadTrainingSetupTreatsMissingComponentNamesAsMissingSetup)
{
	fs::path root = TempAgentRoot("missing_component_names_load");
	fs::path setupPath = root / TrainingSetupSnapshot::FILE_NAME;

	Agent source = MakeConfiguredAgent();
	nlohmann::json setupJson = {
		{ "schema_version", 1 },
		{ "agent", source.agentName },
		{ "stage", "snapshot-stage" },
		{ "version_id", "v1.00" },
		{ "saved_at", "test" },
		{ "agent_setup", AgentSetupSnapshotToJson(CaptureAgentSetupSnapshot(source)) },
		{ "rewards", nlohmann::json::array() },
		{ "hyperparameters", nlohmann::json::array() },
	};
	setupJson["agent_setup"].erase("obs_builder");
	setupJson["agent_setup"].erase("action_parser");
	WriteTextFile(setupPath, setupJson.dump(4));

	std::optional<TrainingSetupSnapshot> loaded = RocketForge::LoadTrainingSetupSnapshot(setupPath);
	ASSERT_TRUE(loaded.has_value());
	ASSERT_FALSE(loaded->agentSetup.has_value());

	fs::remove_all(root);
}

TEST(AgentSetupSnapshot, SetupHashTracksAgentSetupButIgnoresModelPath)
{
	TrainingSetupSnapshot snapshot = {};
	snapshot.agentName = "snapshot-agent";
	snapshot.stageName = "snapshot-stage";
	snapshot.versionId = "v1.00";

	Agent agent = MakeConfiguredAgent();
	snapshot.agentSetup = CaptureAgentSetupSnapshot(agent);
	const std::string baseline = BuildStorage::MakeSetupHash(snapshot);

	agent.modelPath = "C:/different/model/path";
	snapshot.agentSetup = CaptureAgentSetupSnapshot(agent);
	ASSERT_EQ(BuildStorage::MakeSetupHash(snapshot), baseline);

	agent.tickSkip = 6;
	snapshot.agentSetup = CaptureAgentSetupSnapshot(agent);
	ASSERT_FALSE(BuildStorage::MakeSetupHash(snapshot) == baseline);

	agent.tickSkip = 4;
	agent.actionParserName = "DifferentAction";
	snapshot.agentSetup = CaptureAgentSetupSnapshot(agent);
	ASSERT_FALSE(BuildStorage::MakeSetupHash(snapshot) == baseline);
}

TEST(TrainingSetupSnapshot, SetupHashAndDiffIgnoreMetricsTransportFields)
{
	TrainingSetupSnapshot saved = {};
	saved.agentName = "snapshot-agent";
	saved.stageName = "snapshot-stage";
	saved.versionId = "v1.00";
	saved.hyperparameters.push_back({ "Metrics", "Send Metrics", "true" });
	saved.hyperparameters.push_back({ "Metrics", "Project Name", "ExampleBot" });
	saved.hyperparameters.push_back({ "Metrics", "Group Name", "Example-Ent" });
	saved.hyperparameters.push_back({ "Metrics", "Run Name", "run-a" });
	saved.hyperparameters.push_back({ "Metrics", "Add Rewards To Metrics", "true" });
	saved.hyperparameters.push_back({ "Metrics", "Skill Tracker Enabled", "true" });

	TrainingSetupSnapshot current = saved;
	current.hyperparameters[0].value = "false";
	current.hyperparameters[1].value = "Render";
	current.hyperparameters[2].value = "example-stage-2-ts8";
	current.hyperparameters[3].value = "run-b";
	current.hyperparameters[4].value = "false";

	const std::string baseline = BuildStorage::MakeSetupHash(saved);
	ASSERT_EQ(BuildStorage::MakeSetupHash(current), baseline);
	ASSERT_TRUE(DiffTrainingSetupSnapshots(saved, current).empty());

	current.hyperparameters[5].value = "false";

	const std::vector<TrainingSetupDiff> diffs = DiffTrainingSetupSnapshots(saved, current);
	bool sawSkillTracker = false;
	for (const TrainingSetupDiff& diff : diffs) {
		if (diff.field == "Metrics / Skill Tracker Enabled") {
			sawSkillTracker = true;
			ASSERT_EQ(diff.savedValue, std::string("true"));
			ASSERT_EQ(diff.currentValue, std::string("false"));
		}
	}

	ASSERT_FALSE(BuildStorage::MakeSetupHash(current) == baseline);
	ASSERT_TRUE(sawSkillTracker);
}

TEST(AgentSetupSnapshot, SegmentSetupHashRequiresExactCurrentHash)
{
	fs::path root = TempAgentRoot("strict_match");
	BuildStorage storage(root / "Models", "Build");
	storage.EnsureBuildExists();

	TrainingSetupSnapshot current = {};
	current.agentName = "snapshot-agent";
	current.stageName = "snapshot-stage";
	current.versionId = "v1.00";
	Agent agent = MakeConfiguredAgent();
	current.agentSetup = CaptureAgentSetupSnapshot(agent);

	TrainingSegmentMetadata segment = {};
	segment.segmentId = current.versionId;
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, current.stageName);
	segment.setupHash = "old-hash";

	const std::string currentHash = BuildStorage::MakeSetupHash(current);
	ASSERT_FALSE(storage.SegmentSetupHashMatches(segment, current, currentHash));

	segment.setupHash = currentHash;
	ASSERT_TRUE(storage.SegmentSetupHashMatches(segment, current, currentHash));

	fs::remove_all(root);
}

TEST(AgentSetupSnapshot, SegmentSetupHashUsesSavedSnapshotForStaleMetricHash)
{
	fs::path root = TempAgentRoot("stale_metric_hash");
	BuildStorage storage(root / "Models", "Build");
	storage.EnsureBuildExists();

	TrainingSetupSnapshot saved = {};
	saved.agentName = "snapshot-agent";
	saved.stageName = "snapshot-stage";
	saved.versionId = "v1.00";
	saved.hyperparameters.push_back({ "Metrics", "Send Metrics", "true" });
	saved.hyperparameters.push_back({ "Metrics", "Group Name", "Example-Ent" });
	saved.hyperparameters.push_back({ "Training", "Entropy Scale", "0.02" });

	TrainingSetupSnapshot current = saved;
	current.hyperparameters[0].value = "false";
	current.hyperparameters[1].value = "example-stage-2-ts8";

	TrainingSegmentMetadata segment = {};
	segment.segmentId = current.versionId;
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, current.stageName);
	segment.stageName = current.stageName;
	segment.setupHash = "legacy-hash";

	ASSERT_TRUE(GGL::JsonSaveToFile(
		saved,
		storage.SegmentRoot(segment.folderName) / TrainingSetupSnapshot::FILE_NAME));

	const std::string currentHash = BuildStorage::MakeSetupHash(current);
	ASSERT_FALSE(currentHash == segment.setupHash);
	ASSERT_TRUE(storage.SegmentSetupHashMatches(segment, current, currentHash));

	current.hyperparameters[2].value = "0.03";
	ASSERT_FALSE(storage.SegmentSetupHashMatches(
		segment,
		current,
		BuildStorage::MakeSetupHash(current)));

	fs::remove_all(root);
}

TEST(AgentSetupSnapshot, SerializeRoundTripAndPresetCarConfig)
{
	RocketForge::AgentSetupSnapshot snapshot = {};
	snapshot.agentName = "example-agent";
	snapshot.obsBuilderName = "default-obs";
	snapshot.actionParserName = "default-parser";
	snapshot.tickSkip = 6;
	snapshot.actionDelay = 5;
	snapshot.carConfig = RocketSim::CAR_CONFIG_DOMINUS;
	snapshot.policy.layerSizes = { 256, 256 };

	nlohmann::json j = RocketForge::AgentSetupSnapshotToJson(snapshot);

	// Preset car configs serialize as their preset name string, not an object.
	ASSERT_TRUE(j["car_config"].is_string());
	ASSERT_EQ(j["car_config"].get<std::string>(), "Dominus");

	std::optional<RocketForge::AgentSetupSnapshot> loaded = RocketForge::AgentSetupSnapshotFromJson(j);
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->tickSkip, 6);
	ASSERT_EQ(loaded->policy.layerSizes.size(), size_t(2));

	// Compare against OCTANE too, so a broken preset lookup that just returns the default
	// (OCTANE) config would fail this test instead of passing it by coincidence.
	ASSERT_EQ(loaded->carConfig.hitboxSize.x, RocketSim::CAR_CONFIG_DOMINUS.hitboxSize.x);
	ASSERT_TRUE(loaded->carConfig.hitboxSize.x != RocketSim::CAR_CONFIG_OCTANE.hitboxSize.x);
	ASSERT_EQ(loaded->carConfig.frontWheels.wheelRadius, RocketSim::CAR_CONFIG_DOMINUS.frontWheels.wheelRadius);
}

TEST(AgentSetupSnapshot, SerializeRoundTripCustomCarConfig)
{
	RocketSim::CarConfig customConfig = RocketSim::CAR_CONFIG_OCTANE;
	customConfig.hitboxSize.x += 12.5f;
	customConfig.hitboxSize.y += 3.f;
	customConfig.frontWheels.wheelRadius += 1.f;
	customConfig.dodgeDeadzone = 0.75f;
	customConfig.threeWheels = true;

	RocketForge::AgentSetupSnapshot snapshot = {};
	snapshot.obsBuilderName = "default-obs";
	snapshot.actionParserName = "default-parser";
	snapshot.carConfig = customConfig;

	nlohmann::json j = RocketForge::AgentSetupSnapshotToJson(snapshot);

	// A perturbed hitbox must be recognized as custom and serialize as a full object rather
	// than a preset string, or this test would silently exercise the preset-string path instead.
	ASSERT_TRUE(j["car_config"].is_object());
	ASSERT_TRUE(j["car_config"].contains("hitbox_size"));
	ASSERT_TRUE(j["car_config"].contains("hitbox_pos_offset"));
	ASSERT_TRUE(j["car_config"].contains("front_wheels"));
	ASSERT_TRUE(j["car_config"].contains("back_wheels"));
	ASSERT_TRUE(j["car_config"].contains("three_wheels"));
	ASSERT_TRUE(j["car_config"].contains("dodge_deadzone"));

	std::optional<RocketForge::AgentSetupSnapshot> loaded = RocketForge::AgentSetupSnapshotFromJson(j);
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->carConfig.hitboxSize.x, customConfig.hitboxSize.x);
	ASSERT_EQ(loaded->carConfig.hitboxSize.y, customConfig.hitboxSize.y);
	ASSERT_EQ(loaded->carConfig.hitboxPosOffset.x, customConfig.hitboxPosOffset.x);
	ASSERT_EQ(loaded->carConfig.frontWheels.wheelRadius, customConfig.frontWheels.wheelRadius);
	ASSERT_EQ(loaded->carConfig.backWheels.wheelRadius, customConfig.backWheels.wheelRadius);
	ASSERT_EQ(loaded->carConfig.threeWheels, customConfig.threeWheels);
	ASSERT_EQ(loaded->carConfig.dodgeDeadzone, customConfig.dodgeDeadzone);
}

TEST(TrainingSetupSnapshot, DiffReportsSavedToCurrentValues)
{
	TrainingSetupSnapshot saved = {};
	saved.agentName = "snapshot-agent";
	saved.stageName = "stage-a";
	saved.versionId = "v1.00";
	saved.rewards.push_back({
		"TouchBallReward",
		1.f,
		1.f,
		{ 0.f, 1.f },
		RLGC::RewardCategory::EVENT
	});
	saved.hyperparameters.push_back({ "Training", "Entropy Scale", "0.02" });
	saved.hyperparameters.push_back({ "Versioning", "Saved At", "old-time" });

	TrainingSetupSnapshot current = saved;
	current.stageName = "stage-b";
	current.rewards[0].weight = 2.f;
	current.hyperparameters[0].value = "0.03";
	current.hyperparameters[1].value = "new-time";

	const std::vector<TrainingSetupDiff> diffs = DiffTrainingSetupSnapshots(saved, current);

	bool sawStage = false;
	bool sawRewardWeight = false;
	bool sawEntropy = false;
	bool sawSavedAt = false;
	for (const TrainingSetupDiff& diff : diffs) {
		if (diff.field == "Training Stage") {
			sawStage = true;
			ASSERT_EQ(diff.savedValue, std::string("stage-a"));
			ASSERT_EQ(diff.currentValue, std::string("stage-b"));
		}
		if (diff.field == "Reward / TouchBallReward / Weight") {
			sawRewardWeight = true;
			ASSERT_EQ(diff.savedValue, std::string("1"));
			ASSERT_EQ(diff.currentValue, std::string("2"));
		}
		if (diff.field == "Training / Entropy Scale") {
			sawEntropy = true;
			ASSERT_EQ(diff.savedValue, std::string("0.02"));
			ASSERT_EQ(diff.currentValue, std::string("0.03"));
		}
		if (diff.field == "Versioning / Saved At")
			sawSavedAt = true;
	}

	ASSERT_TRUE(sawStage);
	ASSERT_TRUE(sawRewardWeight);
	ASSERT_TRUE(sawEntropy);
	ASSERT_FALSE(sawSavedAt);
}

TEST(AgentSetupSnapshot, CreateTrainingStageAppliesOverrideBeforeLearnerConfig)
{
	AgentRegistry::Register("test-agent-setup-override", []() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = "test-agent-setup-override";
		agent->tickSkip = 8;
		agent->actionDelay = 7;
		agent->policy.layerSizes = { 256 };
		return agent;
	});

	RocketForge::StageRegistry::Register(
		"test-agent-setup-stage",
		[](std::unique_ptr<Agent> agent) {
			return TrainingStage::Create<TrainingStage>(std::move(agent));
		},
		RocketForge::StageCompatInfo{});

	Agent savedAgent;
	savedAgent.agentName = "test-agent-setup-override";
	savedAgent.tickSkip = 5;
	savedAgent.actionDelay = 4;
	savedAgent.obsBuilderName = "AdvancedObs";
	savedAgent.actionParserName = "DefaultAction";
	savedAgent.policy.layerSizes = { 64, 64 };

	StageSettings settings = {};
	settings.stageName = "test-agent-setup-stage";
	settings.agentSetupOverride = CaptureAgentSetupSnapshot(savedAgent);

	std::unique_ptr<TrainingStage> stage = CreateTrainingStage(
		"test-agent-setup-override",
		"test-agent-setup-stage",
		GlobalSettings{},
		settings);

	ASSERT_TRUE(stage != nullptr);
	ASSERT_EQ(stage->agent->tickSkip, 5);
	ASSERT_EQ(stage->agent->actionDelay, 4);
	ASSERT_EQ(stage->learnerConfig.tickSkip, 5);
	ASSERT_EQ(stage->learnerConfig.actionDelay, 4);
	ASSERT_EQ(stage->agent->obsBuilderName, std::string("AdvancedObs"));
	ASSERT_EQ(stage->agent->actionParserName, std::string("DefaultAction"));
	ASSERT_EQ(stage->learnerConfig.ppo.policy.layerSizes.size(), size_t(2));
	ASSERT_EQ(stage->learnerConfig.ppo.policy.layerSizes[0], 64);
}

TEST(AgentSetupSnapshot, RLBotExportWritesSavedSetupOverride)
{
	fs::path root = TempAgentRoot("rlbot_export");
	fs::path agentRoot = root / "Agent";
	fs::path modelsRoot = agentRoot / "Models";
	WriteTextFile(agentRoot / "rlbot" / "bot.toml",
		"[settings]\n"
		"agent_id = \"Test/Agent\"\n"
		"name = \"Test Agent\"\n"
		"loadout_file = \"loadout.toml\"\n"
		"root_dir = \".\"\n"
		"run_command = \"cmd /C run.bat\"\n");
	WriteTextFile(agentRoot / "rlbot" / "loadout.toml", "[loadout]\n");
	WriteTextFile(agentRoot / "rlbot" / "logo.png", "logo-bytes");
	WriteTextFile(agentRoot / "rlbot" / "RLBot.exe", "rlbot-binary");


	const std::string buildName = "Export Build";
	BuildStorage storage(modelsRoot, buildName);
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "export-stage");
	segment.buildName = buildName;
	segment.stageName = "export-stage";
	segment.status = "active";
	storage.SaveSegment(segment);
	WriteTextFile(storage.SegmentCheckpointsRoot(segment.folderName) / "1234" / "POLICY.lt", "policy");

	Agent savedAgent = MakeConfiguredAgent();
	savedAgent.agentName = "saved-agent-setup";
	savedAgent.tickSkip = 5;
	savedAgent.actionDelay = 2;
	savedAgent.obsBuilderName = "AdvancedObs";
	savedAgent.actionParserName = "DefaultAction";
	savedAgent.policy.layerSizes = { 64, 32 };

	TrainingSetupSnapshot setup = {};
	setup.agentName = savedAgent.agentName;
	setup.stageName = segment.stageName;
	setup.versionId = segment.segmentId;
	setup.agentSetup = CaptureAgentSetupSnapshot(savedAgent);
	ASSERT_TRUE(GGL::JsonSaveToFile(
		setup,
		storage.SegmentRoot(segment.folderName) / TrainingSetupSnapshot::FILE_NAME));

	AgentRegistry::Register("base-agent", [modelsRoot]() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = "base-agent";
		agent->modelPath = modelsRoot;
		return agent;
	});

	auto baseAgent = std::make_unique<Agent>();
	baseAgent->agentName = "base-agent";
	baseAgent->modelPath = modelsRoot;
	ExportTestStage stage(std::move(baseAgent), segment.stageName);
	stage.learnerConfig.runtime.buildName = buildName;

	RocketForgeTraining::ExportVersionToRLBot(&stage, segment.segmentId);

	const fs::path exportRoot = storage.SegmentRoot(segment.folderName) / "rlbot";
	const std::string runBat = [] (const fs::path& path) {
		std::ifstream fIn(path);
		std::stringstream buffer;
		buffer << fIn.rdbuf();
		return buffer.str();
	}(exportRoot / "run.bat");
	const fs::path checkpointRoot = storage.SegmentCheckpointsRoot(segment.folderName) / "1234";
	fs::path expectedModelRoot = fs::relative(checkpointRoot, exportRoot);
	expectedModelRoot.make_preferred();

	ASSERT_TRUE(runBat.find("set \"MODEL_ROOT=" + expectedModelRoot.string() + "\"") != std::string::npos);
	ASSERT_TRUE(runBat.find(checkpointRoot.string()) == std::string::npos);
	ASSERT_TRUE(runBat.find("\"base-agent\" --model-dir \"%MODEL_ROOT%\" --agent-setup \"%AGENT_SETUP%\"") != std::string::npos);
	ASSERT_TRUE(runBat.find("set \"AGENT_SETUP=%~dp0agent_setup.json\"") != std::string::npos);
	ASSERT_EQ(ReadTextFile(exportRoot / "logo.png"), std::string("logo-bytes"));
	ASSERT_EQ(ReadTextFile(exportRoot / "device.txt"), "gpu\n");
	ASSERT_NE(runBat.find("--hivemind on --device \"%RLBOT_DEVICE%\" %*"), std::string::npos);
	ASSERT_NE(ReadTextFile(exportRoot / "bot.toml").find("hivemind = true"), std::string::npos);

	const fs::path discoveryConfig = agentRoot / "rlbot_exports" / "base-agentexport-build-v1-00.bot.toml";
	ASSERT_NE(ReadTextFile(discoveryConfig).find("hivemind = true"), std::string::npos);

	{
		std::ifstream fIn(exportRoot / "agent_setup.json");
		ASSERT_TRUE(fIn.good());
		nlohmann::json exportedSetup = nlohmann::json::parse(fIn, nullptr, false);
		ASSERT_EQ(exportedSetup.value("car_config", std::string()), std::string("Dominus"));
		std::optional<AgentSetupSnapshot> loaded = AgentSetupSnapshotFromJson(exportedSetup);
		ASSERT_TRUE(loaded.has_value());
		ASSERT_EQ(loaded->agentName, std::string("saved-agent-setup"));
		ASSERT_EQ(loaded->obsBuilderName, std::string("AdvancedObs"));
		ASSERT_EQ(loaded->actionParserName, std::string("DefaultAction"));
		ASSERT_EQ(loaded->tickSkip, 5);
		ASSERT_EQ(loaded->actionDelay, 2);
		ASSERT_EQ(loaded->policy.layerSizes.size(), size_t(2));
		ASSERT_EQ(loaded->policy.layerSizes[0], 64);
	}

	// Re-exporting a template that opts out must keep both RLBot configurations and the launcher aligned.
	WriteTextFile(agentRoot / "rlbot" / "bot.toml",
		"[settings]\nhivemind = false\nagent_id = \"Test/Agent\"\nname = \"Test Agent\"\n"
		"run_command = \"cmd /C run.bat\"\n");
	RocketForgeTraining::ExportVersionToRLBot(&stage, segment.segmentId);
	WriteTextFile(agentRoot / "rlbot" / "device.txt", "cpu\n");
	RocketForgeTraining::ExportVersionToRLBot(&stage, segment.segmentId);

	ASSERT_EQ(ReadTextFile(exportRoot / "device.txt"), "cpu\n");
	const std::string optedOut = ReadTextFile(exportRoot / "bot.toml");
	ASSERT_NE(optedOut.find("hivemind = false"), std::string::npos);
	ASSERT_EQ(optedOut.find("hivemind = true"), std::string::npos);
	ASSERT_NE(ReadTextFile(discoveryConfig).find("hivemind = false"), std::string::npos);
	ASSERT_NE(ReadTextFile(exportRoot / "run.bat").find("--hivemind off --device \"%RLBOT_DEVICE%\" %*"), std::string::npos);

	fs::remove_all(root);
}

TEST(AgentSetupSnapshot, RLBotExportRefreshesAfterContinuedTraining)
{
	fs::path root = TempAgentRoot("rlbot_refresh");
	fs::path agentRoot = root / "Agent";
	fs::path modelsRoot = agentRoot / "Models";
	WriteTextFile(agentRoot / "rlbot" / "bot.toml",
		"[settings]\n"
		"agent_id = \"Test/Agent\"\n"
		"name = \"Test Agent\"\n"
		"loadout_file = \"loadout.toml\"\n"
		"root_dir = \".\"\n"
		"run_command = \"cmd /C run.bat\"\n");
	WriteTextFile(agentRoot / "rlbot" / "loadout.toml", "[loadout]\n");
	WriteTextFile(agentRoot / "rlbot" / "RLBot.exe", "rlbot-binary");

	const std::string buildName = "Export Build";
	BuildStorage storage(modelsRoot, buildName);
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "export-stage");
	segment.buildName = buildName;
	segment.stageName = "export-stage";
	segment.status = "active";
	storage.SaveSegment(segment);
	WriteTextFile(storage.SegmentCheckpointsRoot(segment.folderName) / "2000000" / "POLICY.lt", "policy");

	AgentRegistry::Register("base-agent", [modelsRoot]() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = "base-agent";
		agent->modelPath = modelsRoot;
		return agent;
	});

	auto baseAgent = std::make_unique<Agent>();
	baseAgent->agentName = "base-agent";
	baseAgent->modelPath = modelsRoot;
	ExportTestStage stage(std::move(baseAgent), segment.stageName);
	stage.learnerConfig.runtime.buildName = buildName;

	RocketForgeTraining::ExportVersionToRLBot(&stage, segment.segmentId);

	const fs::path exportRoot = storage.SegmentRoot(segment.folderName) / "rlbot";
	const fs::path laterCheckpoint = storage.SegmentCheckpointsRoot(segment.folderName) / "3000000";
	WriteTextFile(laterCheckpoint / "POLICY.lt", "policy");
	RocketForgeTraining::RefreshRLBotExportAfterCheckpoint(&stage, segment.segmentId, 3000000, laterCheckpoint);

	const std::string runBat = ReadTextFile(exportRoot / "run.bat");
	fs::path expectedModelRoot = fs::relative(laterCheckpoint, exportRoot);
	expectedModelRoot.make_preferred();
	ASSERT_TRUE(runBat.find("set \"MODEL_ROOT=" + expectedModelRoot.string() + "\"") != std::string::npos);
	ASSERT_TRUE(runBat.find(laterCheckpoint.string()) == std::string::npos);

	const std::string botToml = ReadTextFile(exportRoot / "bot.toml");
	ASSERT_TRUE(botToml.find("name = \"base-agentExport Build v1.00 3M\"") != std::string::npos);

	const fs::path discoveryConfig = agentRoot / "rlbot_exports" / "base-agentexport-build-v1-00.bot.toml";
	const std::string discoveryToml = ReadTextFile(discoveryConfig);
	ASSERT_TRUE(discoveryToml.find("name = \"base-agentExport Build v1.00 3M\"") != std::string::npos);

	{
		std::ifstream fIn(storage.RLBotExportsPath());
		nlohmann::json registry = nlohmann::json::parse(fIn, nullptr, false);
		ASSERT_TRUE(registry.is_object());
		ASSERT_EQ(registry["exports"].at(0).value("latest_timesteps", uint64_t(0)), uint64_t(3000000));
	}

	fs::remove_all(root);
}

TEST(VersionSummary, ReportsNewerAutosave)
{
	fs::path root = TempAgentRoot("summary_autosave");
	fs::path modelsRoot = root / "Agent" / "Models";

	const std::string buildName = "Summary Build";
	BuildStorage storage(modelsRoot, buildName);
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "summary-stage");
	segment.buildName = buildName;
	segment.stageName = "summary-stage";
	segment.status = "active";
	storage.SaveSegment(segment);
	WriteTextFile(storage.SegmentCheckpointsRoot(segment.folderName) / "1000" / "POLICY.lt", "policy");
	WriteTextFile(storage.SegmentTempCheckpointsRoot(segment.folderName) / "5000" / "POLICY.lt", "policy");

	AgentRegistry::Register("summary-agent", [modelsRoot]() {
		auto registered = std::make_unique<Agent>();
		registered->agentName = "summary-agent";
		registered->modelPath = modelsRoot;
		return registered;
	});

	auto agent = std::make_unique<Agent>();
	agent->agentName = "summary-agent";
	agent->modelPath = modelsRoot;
	ExportTestStage stage(std::move(agent), segment.stageName);
	stage.learnerConfig.runtime.buildName = buildName;

	const bool includeInactive = true;
	std::vector<GGL::SegmentVersionSummary> versions = RocketForgeTraining::ListVersions(&stage, includeInactive);
	ASSERT_EQ(versions.size(), size_t(1));
	ASSERT_EQ(versions[0].latestTimesteps, uint64_t(5000));
	ASSERT_TRUE(versions[0].latestIsAutosave);
	ASSERT_EQ(versions[0].modelCount, size_t(1));

	fs::remove_all(root);
}

TEST(AgentSetupSnapshot, RLBotExportSupportsStoredImportedStages)
{
	fs::path root = TempAgentRoot("rlbot_stored_imported");
	fs::path agentRoot = root / "Agent";
	fs::path modelsRoot = agentRoot / "Models";
	WriteTextFile(agentRoot / "rlbot" / "bot.toml",
		"[settings]\n"
		"agent_id = \"Test/StoredAgent\"\n"
		"name = \"Stored Agent\"\n"
		"loadout_file = \"loadout.toml\"\n"
		"root_dir = \".\"\n"
		"run_command = \"cmd /C run.bat\"\n");
	WriteTextFile(agentRoot / "rlbot" / "loadout.toml", "[loadout]\n");
	WriteTextFile(agentRoot / "rlbot" / "RLBot.exe", "rlbot-binary");

	const std::string agentName = "stored-imported-export-agent";
	AgentRegistry::Register(agentName, [agentName, modelsRoot]() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = agentName;
		// Exercise an agent whose compiled default points at one deep checkpoint.
		agent->modelPath = modelsRoot / "Build8" / "segments" / "v1.00_imported" / "checkpoints" / "0";
		return agent;
	});

	for (const std::pair<std::string, int>& build : {
		std::pair<std::string, int>{ "Build4", 4 },
		std::pair<std::string, int>{ "Build8", 8 } }) {
		BuildStorage storage(modelsRoot, build.first);
		storage.EnsureBuildExists();

		TrainingSegmentMetadata segment = {};
		segment.segmentId = "v1.00";
		segment.folderName = "v1.00_imported";
		segment.buildName = build.first;
		segment.stageName = "Imported";
		segment.status = "active";
		segment.eligibleRLBot = true;
		storage.SaveSegment(segment);
		WriteTextFile(storage.SegmentCheckpointsRoot(segment.folderName) / "0" / "POLICY.lt", "policy");

		Agent savedAgent;
		savedAgent.agentName = agentName;
		savedAgent.tickSkip = build.second;
		savedAgent.actionDelay = build.second == 4 ? 3 : 0;
		savedAgent.obsBuilderName = build.second == 4 ? "StoredObs4" : "StoredObs8";
		savedAgent.actionParserName = build.second == 4 ? "DefaultAction" : "StoredAction";

		TrainingSetupSnapshot setup = {};
		setup.agentName = agentName;
		setup.stageName = segment.stageName;
		setup.versionId = segment.segmentId;
		setup.agentSetup = CaptureAgentSetupSnapshot(savedAgent);
		ASSERT_TRUE(GGL::JsonSaveToFile(
			setup,
			storage.SegmentRoot(segment.folderName) / TrainingSetupSnapshot::FILE_NAME));

		RocketForgeTraining::ExportVersionToRLBot(agentName, build.first, segment.segmentId);

		const fs::path exportRoot = storage.SegmentRoot(segment.folderName) / "rlbot";
		ASSERT_EQ(ReadTextFile(exportRoot / "RLBot.exe"), std::string("rlbot-binary"));
		ASSERT_TRUE(ReadTextFile(exportRoot / "run.bat").find(
			"\"" + agentName + "\" --model-dir \"%MODEL_ROOT%\" --agent-setup \"%AGENT_SETUP%\"")
			!= std::string::npos);

		std::ifstream setupIn(exportRoot / "agent_setup.json");
		nlohmann::json exportedSetup = nlohmann::json::parse(setupIn, nullptr, false);
		ASSERT_EQ(exportedSetup.value("tick_skip", 0), build.second);
	}

	ASSERT_TRUE(fs::exists(agentRoot / "rlbot_exports" / "stored-imported-export-agentbuild4-v1-00.bot.toml"));
	ASSERT_TRUE(fs::exists(agentRoot / "rlbot_exports" / "stored-imported-export-agentbuild8-v1-00.bot.toml"));

	fs::remove_all(root);
}

TEST(AgentSetupSnapshot, RLBotExportTakesTheEngineFromTheSegmentNotTheCompiledAgent)
{
	fs::path root = TempAgentRoot("rlbot_export_engine");
	fs::path agentRoot = root / "Agent";
	fs::path modelsRoot = agentRoot / "Models";
	WriteTextFile(agentRoot / "rlbot" / "bot.toml", "[settings]\nagent_id = \"Test/EngineAgent\"\nname = \"Engine Agent\"\n");
	WriteTextFile(agentRoot / "rlbot" / "RLBot.exe", "rlbot-binary");

	const std::string agentName = "rlbot-export-engine-agent";
	AgentRegistry::Register(agentName, [agentName, modelsRoot]() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = agentName;
		agent->trainingEngineVersion = 3;
		agent->modelPath = modelsRoot;
		return agent;
	});

	const std::string buildName = "Engine Build";
	BuildStorage storage(modelsRoot, buildName);
	storage.EnsureBuildExists();
	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "engine-stage");
	segment.buildName = buildName;
	segment.stageName = "engine-stage";
	segment.status = "active";
	storage.SaveSegment(segment);
	WriteTextFile(storage.SegmentCheckpointsRoot(segment.folderName) / "1000" / "POLICY.lt", "policy");
	const fs::path setupPath = storage.SegmentRoot(segment.folderName) / TrainingSetupSnapshot::FILE_NAME;

	WriteTextFile(setupPath, "broken");
	EXPECT_THROW(RocketForgeTraining::ExportVersionToRLBot(agentName, buildName, segment.segmentId), std::runtime_error);

	// Segments saved before setups were recorded trained on v2, whatever the agent now defaults to.
	fs::remove(setupPath);
	RocketForgeTraining::ExportVersionToRLBot(agentName, buildName, segment.segmentId);
	const fs::path exportRoot = storage.SegmentRoot(segment.folderName) / "rlbot";
	const nlohmann::json exportedSetup = nlohmann::json::parse(ReadTextFile(exportRoot / "agent_setup.json"));
	EXPECT_EQ(exportedSetup.at("training_engine_version"), 2);
	EXPECT_NE(ReadTextFile(exportRoot / "run.bat").find("--agent-setup"), std::string::npos);

	fs::remove_all(root);
}

namespace
{
	/** Sets up an RLBot template, a build with one active segment, and a stage registered for export. */
	struct AutosaveExportFixture
	{
		fs::path root;
		fs::path modelsRoot;
		std::string buildName = "Export Build";
		static constexpr const char* STAGE_NAME = "export-stage";
		BuildStorage storage;
		TrainingSegmentMetadata segment = {};
		ExportTestStage stage;

		static std::unique_ptr<Agent> MakeBaseAgent(const fs::path& modelsRoot)
		{
			auto agent = std::make_unique<Agent>();
			agent->agentName = "base-agent";
			agent->modelPath = modelsRoot;
			return agent;
		}

		explicit AutosaveExportFixture(const std::string& rootName)
			: root(TempAgentRoot(rootName))
			, modelsRoot(root / "Agent" / "Models")
			, storage(modelsRoot, buildName)
			, stage(MakeBaseAgent(modelsRoot), STAGE_NAME)
		{
			fs::path agentRoot = root / "Agent";
			WriteTextFile(agentRoot / "rlbot" / "bot.toml",
				"[settings]\n"
				"agent_id = \"Test/Agent\"\n"
				"name = \"Test Agent\"\n"
				"loadout_file = \"loadout.toml\"\n"
				"root_dir = \".\"\n"
				"run_command = \"cmd /C run.bat\"\n");
			WriteTextFile(agentRoot / "rlbot" / "loadout.toml", "[loadout]\n");
			WriteTextFile(agentRoot / "rlbot" / "RLBot.exe", "rlbot-binary");

			storage.EnsureBuildExists();

			segment.segmentId = "v1.00";
			segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, STAGE_NAME);
			segment.buildName = buildName;
			segment.stageName = STAGE_NAME;
			segment.status = "active";
			storage.SaveSegment(segment);

			fs::path capturedModelsRoot = modelsRoot;
			AgentRegistry::Register("base-agent", [capturedModelsRoot]() { return MakeBaseAgent(capturedModelsRoot); });

			stage.learnerConfig.runtime.buildName = buildName;
		}
	};
}

TEST(AgentSetupSnapshot, RLBotExportSnapshotsNewerAutosave)
{
	AutosaveExportFixture fixture("rlbot_autosave_export");
	BuildStorage& storage = fixture.storage;
	TrainingSegmentMetadata& segment = fixture.segment;
	WriteTextFile(storage.SegmentCheckpointsRoot(segment.folderName) / "1000000" / "POLICY.lt", "policy");

	const fs::path autosave = storage.SegmentTempCheckpointsRoot(segment.folderName) / "4000000";
	WriteTextFile(autosave / "POLICY.lt", "autosave-policy");
	WriteTextFile(autosave / "SHARED_HEAD.lt", "autosave-head");
	WriteTextFile(autosave / "RUNNING_STATS.json", "{}");
	WriteTextFile(autosave / "CRITIC.lt", "critic");
	WriteTextFile(autosave / "POLICY_OPTIM.lt", "optim");
	WriteTextFile(autosave / "CRITIC_OPTIM.lt", "optim");

	RocketForgeTraining::ExportVersionToRLBot(&fixture.stage, segment.segmentId);

	const fs::path exportRoot = storage.SegmentRoot(segment.folderName) / "rlbot";
	const fs::path snapshot = exportRoot / "model_snapshot" / "4000000";
	ASSERT_EQ(ReadTextFile(snapshot / "POLICY.lt"), std::string("autosave-policy"));
	ASSERT_EQ(ReadTextFile(snapshot / "SHARED_HEAD.lt"), std::string("autosave-head"));
	ASSERT_TRUE(fs::exists(snapshot / "RUNNING_STATS.json"));
	ASSERT_FALSE(fs::exists(snapshot / "CRITIC.lt"));
	ASSERT_FALSE(fs::exists(snapshot / "POLICY_OPTIM.lt"));
	ASSERT_FALSE(fs::exists(snapshot / "CRITIC_OPTIM.lt"));

	fs::path expectedModelRoot = fs::relative(snapshot, exportRoot);
	expectedModelRoot.make_preferred();
	const std::string runBat = ReadTextFile(exportRoot / "run.bat");
	ASSERT_TRUE(runBat.find("set \"MODEL_ROOT=" + expectedModelRoot.string() + "\"") != std::string::npos);

	const std::string botToml = ReadTextFile(exportRoot / "bot.toml");
	ASSERT_TRUE(botToml.find("4M") != std::string::npos);

	fs::remove_all(fixture.root);
}

TEST(AgentSetupSnapshot, RLBotRefreshRemovesStaleAutosaveSnapshot)
{
	AutosaveExportFixture fixture("rlbot_autosave_refresh");
	BuildStorage& storage = fixture.storage;
	TrainingSegmentMetadata& segment = fixture.segment;
	WriteTextFile(storage.SegmentTempCheckpointsRoot(segment.folderName) / "4000000" / "POLICY.lt", "autosave-policy");

	RocketForgeTraining::ExportVersionToRLBot(&fixture.stage, segment.segmentId);

	const fs::path exportRoot = storage.SegmentRoot(segment.folderName) / "rlbot";
	ASSERT_TRUE(fs::exists(exportRoot / "model_snapshot"));

	const fs::path laterCheckpoint = storage.SegmentCheckpointsRoot(segment.folderName) / "5000000";
	WriteTextFile(laterCheckpoint / "POLICY.lt", "policy");
	RocketForgeTraining::RefreshRLBotExportAfterCheckpoint(&fixture.stage, segment.segmentId, 5000000, laterCheckpoint);

	ASSERT_FALSE(fs::exists(exportRoot / "model_snapshot"));

	fs::remove_all(fixture.root);
}

TEST(AgentSetupSnapshot, OmitsDecomposedCriticWhenOff)
{
	Agent agent = MakeConfiguredAgent();
	agent.decomposedCritic = false;

	nlohmann::json j = AgentSetupSnapshotToJson(CaptureAgentSetupSnapshot(agent));

	ASSERT_FALSE(j.contains("decomposed_critic"));
}

TEST(AgentSetupSnapshot, RoundTripsDecomposedCritic)
{
	Agent agent = MakeConfiguredAgent();
	agent.decomposedCritic = true;

	nlohmann::json j = AgentSetupSnapshotToJson(CaptureAgentSetupSnapshot(agent));
	std::optional<AgentSetupSnapshot> loaded = AgentSetupSnapshotFromJson(j);

	ASSERT_TRUE(loaded.has_value());
	ASSERT_TRUE(loaded->decomposedCritic);

	Agent applied = MakeConfiguredAgent();
	ApplyAgentSetupSnapshot(applied, loaded.value());
	ASSERT_TRUE(applied.decomposedCritic);
}

TEST(AgentSetupSnapshot, DiffReportsDecomposedCritic)
{
	Agent savedAgent = MakeConfiguredAgent();
	Agent currentAgent = savedAgent;
	currentAgent.decomposedCritic = true;

	std::vector<AgentSetupDiff> diffs = DiffAgentSetupSnapshots(
		CaptureAgentSetupSnapshot(savedAgent),
		CaptureAgentSetupSnapshot(currentAgent));

	bool sawDecomposedCritic = false;
	for (const AgentSetupDiff& diff : diffs) {
		if (diff.field == "Decomposed Critic") {
			sawDecomposedCritic = true;
			ASSERT_EQ(diff.savedValue, std::string("false"));
			ASSERT_EQ(diff.currentValue, std::string("true"));
		}
	}

	ASSERT_TRUE(sawDecomposedCritic);
}
