#include <gtest/gtest.h>

#include "TestSupport.h"

#include <GigaLearnCPP/PolicyVersionManager.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/SkillEstimation.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <GigaLearnInfer/ModelManager.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>

namespace fs = std::filesystem;

namespace {
	fs::path TempPolicyVersionManagerRoot(const std::string& name)
	{
		return RocketForgeTests::TestTempDir(name);
	}

	GGL::SkillRating MakeSkillRating(float mu)
	{
		GGL::SkillRating rating = {};
		for (const std::string& mode : GGL::SkillRating::ModeNames())
			rating.data[mode] = GGL::SkillEstimate{ mu, 15.0f };
		return rating;
	}

	GGL::SkillBenchmark MakeBenchmark(
		const fs::path& benchmarkRoot,
		const std::string& id,
		const std::string& createdAt,
		float mu,
		uint64_t timesteps)
	{
		GGL::SkillBenchmark benchmark = {};
		benchmark.id = id;
		benchmark.folderName = id;
		benchmark.createdAt = createdAt;
		benchmark.timesteps = timesteps;
		benchmark.folder = benchmarkRoot / id;
		benchmark.ratings = MakeSkillRating(mu);
		fs::create_directories(benchmark.folder);
		return benchmark;
	}

	bool HasBenchmark(const std::vector<GGL::SkillBenchmark>& benchmarks, const std::string& id)
	{
		return std::find_if(benchmarks.begin(), benchmarks.end(), [&](const GGL::SkillBenchmark& benchmark) {
			return benchmark.id == id;
		}) != benchmarks.end();
	}

	bool HasBenchmarkIndex(
		const std::vector<GGL::SkillBenchmark>& benchmarks,
		const std::vector<int>& indices,
		const std::string& id)
	{
		for (int idx : indices)
			if (idx >= 0 && idx < (int)benchmarks.size() && benchmarks[idx].id == id)
				return true;
		return false;
	}

	void WriteSkillBenchmark(
		const fs::path& benchmarkRoot,
		const std::string& id,
		const std::string& sourceVersionId,
		uint64_t timesteps)
	{
		fs::path folder = benchmarkRoot / id;
		fs::create_directories(folder);

		std::ofstream fOut(folder / "STATS.json");
		fOut << "{\n"
			<< "    \"benchmark_id\": \"" << id << "\",\n"
			<< "    \"benchmark_folder_name\": \"" << id << "\",\n"
			<< "    \"created_at\": \"20260101-000000\",\n"
			<< "    \"source_stage_name\": \"stage\",\n";
		if (!sourceVersionId.empty())
			fOut << "    \"source_version_id\": \"" << sourceVersionId << "\",\n";
		fOut << "    \"timesteps\": " << timesteps << "\n"
			<< "}";
	}

	fs::path MakeRecentPolicyVersionFolder(
		const fs::path& root,
		const std::string& sourceVersionId,
		uint64_t timesteps)
	{
		return root / GGL::TrainingStorage::MakeRecentPolicyVersionFolderName(sourceVersionId, timesteps);
	}

	void WriteRecentPolicyVersion(
		const fs::path& root,
		const std::string& sourceVersionId,
		uint64_t timesteps,
		const std::string& policyCompatibilityJson = {})
	{
		fs::path folder = MakeRecentPolicyVersionFolder(root, sourceVersionId, timesteps);
		fs::create_directories(folder);

		std::ofstream fOut(folder / "STATS.json");
		fOut << "{\n"
			<< "    \"timesteps\": " << timesteps << ",\n"
			<< "    \"source_version_id\": \"" << sourceVersionId << "\"";
		if (!policyCompatibilityJson.empty())
			fOut << ",\n    \"policy_compatibility\": " << policyCompatibilityJson << "\n";
		else
			fOut << "\n";
		fOut
			<< "}";
	}

	bool HasRecentPolicyVersion(
		const fs::path& root,
		const std::string& sourceVersionId,
		uint64_t timesteps)
	{
		return fs::exists(MakeRecentPolicyVersionFolder(root, sourceVersionId, timesteps));
	}

	std::unique_ptr<Agent> CreateRegisteredAgent(const std::string& name)
	{
		return AgentRegistry::Create(name);
	}

	struct FixedSeriesTinyObsBuilder : RLGC::ObsBuilder {
		RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override
		{
			return { 0.0f, 1.0f };
		}
	};

	struct FixedSeriesIdleActionParser : RLGC::ActionParser {
		RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override
		{
			return {};
		}

		int GetActionAmount() override
		{
			return 1;
		}
	};

	RLGC::EnvSetConfig MakeFixedSeriesEnvConfig(int tickSkip)
	{
		RLGC::EnvSetConfig config = {};
		config.numArenas = 1;
		config.tickSkip = tickSkip;
		config.actionDelay = 0;
		config.saveRewards = false;
		config.envCreateCfg = RLGC::EnvCreateConfig(1, (uint8_t)RocketSim::GameMode::SOCCAR);
		config.envCreateCfg.maxTeamSize = 1;
		config.envCreateFn = [](const int, const RLGC::EnvCreateConfig& createConfig) {
			const int teamSize = createConfig.teamSizeOverride > 0 ? createConfig.teamSizeOverride : 1;
			RLGC::EnvCreateResult result = {};
			result.arena = RocketSim::Arena::Create(createConfig.mode);
			for (int i = 0; i < teamSize; i++) {
				result.arena->AddCar(RocketSim::Team::BLUE);
				result.arena->AddCar(RocketSim::Team::ORANGE);
			}
			result.obsBuilder = new FixedSeriesTinyObsBuilder();
			result.actionParser = new FixedSeriesIdleActionParser();
			return result;
		};
		return config;
	}

	GGL::PPOLearnerConfig MakeFixedSeriesPPOConfig()
	{
		GGL::PPOLearnerConfig config = {};
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

	GGL::FixedSeriesOpponent MakeFixedSeriesOpponent(GGL::ModelSet* models, const std::string& label, int tickSkip)
	{
		GGL::FixedSeriesOpponent opponent = {};
		opponent.label = label;
		opponent.runtime.models = models;
		opponent.runtime.obsSize = 2;
		opponent.runtime.numActions = 1;
		opponent.runtime.tickSkip = tickSkip;
		opponent.runtime.actionDelay = 0;
		opponent.runtime.makeRuntimeComponents = []() {
			return GGL::PolicyMatchRuntimeComponents{
				std::make_unique<FixedSeriesTinyObsBuilder>(),
				std::make_unique<FixedSeriesIdleActionParser>()
			};
		};
		return opponent;
	}
}

TEST(PolicyVersionManager, LoadStaticOpponentsUsesSourceAgentRuntime)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	fs::path root = TempPolicyVersionManagerRoot("public_static_opponent");
	fs::path registryPath = root / "static_opponents.json";
	std::unique_ptr<Agent> nexto = AgentRegistry::Create("Nexto");

	nlohmann::json registry = {
		{ "static_opponents", nlohmann::json::array({
			{
				{ "source_agent", "Nexto" },
				{ "source_stage", "Imported Models" },
				{ "source_version_id", "RLBotPack" },
				{ "display_name", "Nexto (RLBotPack)" },
				{ "checkpoint_path", nexto->modelPath.string() }
			}
		}) }
	};
	std::ofstream(registryPath) << registry.dump(2);

	GGL::ModelSet learnerModels = {};
	GGL::PartialModelConfig policy = {};
	policy.layerSizes = { 8 };
	GGL::ModelManager::MakeModels(
		false,
		4,
		5,
		{},
		policy,
		{},
		torch::kCPU,
		learnerModels);

	GGL::SkillTrackerConfig skillConfig = {};
	RLGC::EnvSetConfig envConfig = {};
	envConfig.tickSkip = 8;
	envConfig.actionDelay = 7;
	GGL::PolicyVersionManager manager(
		root / "Recent Policy Versions",
		4,
		1,
		skillConfig,
		envConfig,
		false);
	manager.staticOpponentsRegistryPath = registryPath;
	manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);

	ASSERT_EQ(manager.versions.size(), size_t(1));
	GGL::PolicyVersion& loaded = manager.versions.front();
	ASSERT_TRUE(loaded.inferenceAgent != nullptr);
	ASSERT_EQ(loaded.inferenceAgent->agentName, std::string("Nexto"));
	ASSERT_EQ(loaded.inferenceObsSize, 1057);
	ASSERT_EQ(loaded.inferenceNumActions, 90);
	ASSERT_TRUE(loaded.inferenceDeterministic);
	ASSERT_EQ(loaded.sourceBuildName, std::string("V1"));
	ASSERT_EQ(loaded.staticSelectionKey, std::string("Nexto\nV1\nImported Models\nRLBotPack"));

	torch::Tensor logits = loaded.models["policy"]->Forward(torch::zeros({ 1, 1057 }), false);
	ASSERT_EQ(logits.size(1), int64_t(90));

	learnerModels.Free();
	fs::remove_all(root);
}

TEST(PolicyVersionManager, LoadStaticOpponentsSkipsUnresolvedEntries)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	fs::path root = TempPolicyVersionManagerRoot("unresolved_static_opponent");
	fs::path registryPath = root / "static_opponents.json";
	std::unique_ptr<Agent> nexto = AgentRegistry::Create("Nexto");

	const nlohmann::json unresolved = { { "source_agent", "Nexto" }, { "source_build", "V1" }, { "source_version_id", "v9.9" } };
	const nlohmann::json resolved = {
		{ "source_agent", "Nexto" },
		{ "source_stage", "Imported Models" },
		{ "source_version_id", "RLBotPack" },
		{ "checkpoint_path", nexto->modelPath.string() }
	};

	GGL::ModelSet learnerModels = {};
	GGL::PartialModelConfig policy = {};
	policy.layerSizes = { 8 };
	GGL::ModelManager::MakeModels(false, 4, 5, {}, policy, {}, torch::kCPU, learnerModels);

	GGL::SkillTrackerConfig skillConfig = {};
	RLGC::EnvSetConfig envConfig = {};
	envConfig.tickSkip = 8;
	envConfig.actionDelay = 7;
	GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 4, 1, skillConfig, envConfig, false);
	manager.staticOpponentsRegistryPath = registryPath;

	std::ofstream(registryPath) << nlohmann::json({ { "static_opponents", { resolved, unresolved } } }).dump(2);
	manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);
	ASSERT_EQ(manager.versions.size(), size_t(1));
	ASSERT_EQ(manager.versions.front().sourceVersionId, std::string("RLBotPack"));

	std::ofstream(registryPath) << nlohmann::json({ { "static_opponents", { unresolved } } }).dump(2);
	ASSERT_THROW(manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent), std::runtime_error);

	learnerModels.Free();
	fs::remove_all(root);
}

TEST(PolicyVersionManager, LoadSkillReferencesReadsStoredRatings)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	fs::path root = TempPolicyVersionManagerRoot("skill_reference_ratings");
	fs::path registryPath = root / "skill_references.json";
	std::unique_ptr<Agent> nexto = AgentRegistry::Create("Nexto");

	nlohmann::json registry = {
		{ "skill_references", nlohmann::json::array({
			{
				{ "source_agent", "Nexto" },
				{ "source_stage", "Imported Models" },
				{ "source_version_id", "RLBotPack" },
				{ "display_name", "Nexto (RLBotPack)" },
				{ "checkpoint_path", nexto->modelPath.string() },
				{ "skill_ratings", { { "1v1", 1400.0 }, { "3v3", -10.0 }, { "2v2", "abc" } } }
			}
		}) }
	};
	std::ofstream(registryPath) << registry.dump(2);

	GGL::ModelSet learnerModels = {};
	GGL::PartialModelConfig policy = {};
	policy.layerSizes = { 8 };
	GGL::ModelManager::MakeModels(
		false,
		4,
		5,
		{},
		policy,
		{},
		torch::kCPU,
		learnerModels);

	GGL::SkillTrackerConfig skillConfig = {};
	RLGC::EnvSetConfig envConfig = {};
	envConfig.tickSkip = 8;
	envConfig.actionDelay = 7;
	GGL::PolicyVersionManager manager(root, -1, 0, skillConfig, envConfig, false);
	manager.staticOpponentsRegistryPath = registryPath;
	manager.registryListKey = "skill_references";
	manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);

	ASSERT_EQ(manager.versions.size(), size_t(1));
	const GGL::PolicyVersion& loaded = manager.versions.front();
	ASSERT_EQ(loaded.staticSelectionKey, std::string("Nexto\nV1\nImported Models\nRLBotPack"));
	ASSERT_TRUE(loaded.ratings.data.contains("1v1"));
	ASSERT_NEAR(loaded.ratings.data.at("1v1").mu, 1400.0f, 1e-3f);
	ASSERT_NEAR(loaded.ratings.data.at("1v1").sigma, 0.0f, 1e-6f);
	ASSERT_FALSE(loaded.ratings.data.contains("2v2"));
	ASSERT_TRUE(loaded.ratings.data.contains("3v3"));
	ASSERT_NEAR(loaded.ratings.data.at("3v3").mu, -10.0f, 1e-3f);

	learnerModels.Free();
	fs::remove_all(root);
}

TEST(PolicyVersionManager, WriteSkillReferenceRatingPreservesOtherFields)
{
	fs::path root = TempPolicyVersionManagerRoot("skill_reference_write");
	fs::create_directories(root);
	fs::path registryPath = root / "skill_references.json";

	const nlohmann::json untouched = {
		{ "source_agent", "Nexto" },
		{ "source_build", "V1" },
		{ "source_stage", "Imported Models" },
		{ "source_version_id", "RLBotPack" },
		{ "display_name", "Nexto (RLBotPack)" },
		{ "skill_ratings", { { "1v1", 1400.0 } } }
	};
	const nlohmann::json malformed = {
		{ "source_agent", 5 },
		{ "source_stage", "ground" }
	};
	const nlohmann::json target = {
		{ "source_agent", "source-agent" },
		{ "source_build", "Main 6M" },
		{ "source_stage", "ground" },
		{ "source_version_id", "v1.04" },
		{ "display_name", "Main ground" },
		{ "checkpoint_path", "C:/models/source/checkpoints/6000000" },
		{ "skill_ratings", { { "1v1", 1100.0 } } }
	};
	nlohmann::json registry = {
		{ "schema_version", 1 },
		{ "target_agent", "example-agent" },
		{ "skill_references", nlohmann::json::array({ untouched, malformed, target }) }
	};
	std::ofstream(registryPath) << registry.dump(2);

	GGL::PolicyVersionManager::WriteSkillReferenceRating(registryPath, "source-agent\nMain 6M\nground\nv1.04", "2v2", 1350.0f);

	std::ifstream fIn(registryPath);
	const nlohmann::json written = nlohmann::json::parse(fIn);
	fIn.close();

	const nlohmann::json& references = written.at("skill_references");
	ASSERT_EQ(references.size(), size_t(3));
	ASSERT_EQ(references[0], untouched);
	ASSERT_EQ(references[1], malformed);
	ASSERT_EQ(written.value("target_agent", std::string()), std::string("example-agent"));
	ASSERT_FALSE(fs::exists(registryPath.string() + ".tmp"));

	const nlohmann::json& updated = references[2];
	ASSERT_EQ(updated.value("display_name", std::string()), std::string("Main ground"));
	ASSERT_EQ(updated.value("checkpoint_path", std::string()), std::string("C:/models/source/checkpoints/6000000"));
	ASSERT_NEAR(updated.at("skill_ratings").at("1v1").get<float>(), 1100.0f, 1e-3f);
	ASSERT_NEAR(updated.at("skill_ratings").at("2v2").get<float>(), 1350.0f, 1e-3f);

	GGL::PolicyVersionManager::WriteSkillReferenceRating(registryPath, "Missing\nV1\nground\nv9.00", "1v1", 1.0f);

	std::ifstream fInAfterMiss(registryPath);
	ASSERT_EQ(nlohmann::json::parse(fInAfterMiss), written);
	fInAfterMiss.close();

	fs::remove_all(root);
}

TEST(PolicyVersionManager, StaticOpponentBallPredictionEngineOverridesSavedPredictor)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	ROCKETFORGE_REQUIRE_ENGINE(RocketSim::EngineId::V3);

	fs::path root = TempPolicyVersionManagerRoot("static_opponent_prediction_engine");
	fs::path registryPath = root / "static_opponents.json";
	std::unique_ptr<Agent> nexto = AgentRegistry::Create("Nexto");
	ASSERT_EQ(nexto->GetBallPredictionEngineVersion(), 2);

	nlohmann::json registry = {
		{ "static_opponents", nlohmann::json::array({
			{
				{ "source_agent", "Nexto" },
				{ "source_stage", "Imported Models" },
				{ "source_version_id", "RLBotPack" },
				{ "display_name", "Nexto (RLBotPack)" },
				{ "checkpoint_path", nexto->modelPath.string() }
			}
		}) }
	};
	std::ofstream(registryPath) << registry.dump(2);

	GGL::ModelSet learnerModels = {};
	GGL::PartialModelConfig policy = {};
	policy.layerSizes = { 8 };
	GGL::ModelManager::MakeModels(false, 4, 5, {}, policy, {}, torch::kCPU, learnerModels);

	struct Case {
		std::string label;
		std::optional<RocketSim::EngineId> overrideEngine;
		int expectedEngine;
	};
	const Case cases[] = {
		{ "own predictor", std::nullopt, 2 },
		{ "learner predictor", RocketSim::EngineId::V3, 3 },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		RLGC::EnvSetConfig envConfig = {};
		envConfig.tickSkip = 8;
		envConfig.actionDelay = 7;
		GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 4, 1, {}, envConfig, false);
		manager.staticOpponentsRegistryPath = registryPath;
		manager.staticOpponentBallPredictionEngine = testCase.overrideEngine;
		manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);

		ASSERT_EQ(manager.versions.size(), size_t(1));
		const GGL::PolicyVersion& loaded = manager.versions.front();
		EXPECT_EQ(loaded.inferenceAgent->GetBallPredictionEngineVersion(), testCase.expectedEngine);
		EXPECT_EQ(loaded.inferenceAgent->trainingEngineVersion, 2);
		EXPECT_EQ(loaded.inferenceObsSize, 1057);
	}

	learnerModels.Free();
	fs::remove_all(root);
}

TEST(PolicyVersionManager, NormalStaticOpponentKeepsLearnerRuntime)
{
	fs::path root = TempPolicyVersionManagerRoot("normal_static_opponent");
	fs::path modelFolder = root / "checkpoint";
	fs::path registryPath = root / "static_opponents.json";
	fs::create_directories(modelFolder);

	GGL::ModelSet learnerModels = {};
	GGL::PartialModelConfig policy = {};
	policy.layerSizes = { 8 };
	GGL::ModelManager::MakeModels(
		false,
		4,
		5,
		{},
		policy,
		{},
		torch::kCPU,
		learnerModels);
	learnerModels.Save(modelFolder);

	nlohmann::json registry = {
		{ "static_opponents", nlohmann::json::array({
			{
				{ "source_agent", "example-agent" },
				{ "source_stage", "Imported Models" },
				{ "source_version_id", "legacy" },
				{ "display_name", "Registered static opponent" },
				{ "checkpoint_path", modelFolder.string() }
			},
			{
				{ "source_agent", "unregistered-external-agent" },
				{ "source_stage", "Imported Models" },
				{ "source_version_id", "external" },
				{ "display_name", "Unregistered static opponent" },
				{ "checkpoint_path", modelFolder.string() }
			}
		}) }
	};
	std::ofstream(registryPath) << registry.dump(2);

	GGL::SkillTrackerConfig skillConfig = {};
	RLGC::EnvSetConfig envConfig = {};
	envConfig.tickSkip = 8;
	envConfig.actionDelay = 7;
	GGL::PolicyVersionManager manager(
		root / "Recent Policy Versions",
		4,
		1,
		skillConfig,
		envConfig,
		false);
	manager.staticOpponentsRegistryPath = registryPath;
	manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);

	ASSERT_EQ(manager.versions.size(), size_t(2));
	for (GGL::PolicyVersion& version : manager.versions) {
		ASSERT_TRUE(version.inferenceAgent == nullptr);
		ASSERT_EQ(version.models["policy"]->config.numInputs, 4);
		ASSERT_EQ(version.models["policy"]->config.numOutputs, 5);
	}

	manager.ClearVersions();
	learnerModels.Free();
	fs::remove_all(root);
}

TEST(PolicyVersionManager, SavedSourceSetupOverridesCurrentAgentAndSupportsSlowerCadence)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	const std::string agentName = "saved-static-runtime-test-agent";
	AgentRegistry::Register(agentName, [agentName]() {
		auto agent = std::make_unique<Agent>();
		agent->agentName = agentName;
		agent->tickSkip = 4;
		agent->actionDelay = 3;
		agent->maxTeamSize = 3;
		agent->sharedHead.layerSizes = { 31 };
		agent->policy.layerSizes = { 29 };
		return agent;
	});

	Agent savedAgent;
	savedAgent.agentName = agentName;
	savedAgent.tickSkip = 8;
	savedAgent.actionDelay = 7;
	savedAgent.maxTeamSize = 2;
	savedAgent.sharedHead.layerSizes = { 13 };
	savedAgent.policy.layerSizes = { 17, 11 };
	const RocketForge::AgentSetupSnapshot savedSetup =
		RocketForge::CaptureAgentSetupSnapshot(savedAgent);

	fs::path root = TempPolicyVersionManagerRoot("saved_source_setup");
	fs::path modelFolder = root / "checkpoint";
	fs::path registryPath = root / "static_opponents.json";
	fs::create_directories(modelFolder);

	std::unique_ptr<RLGC::ActionParser> sourceActionParser(savedAgent.MakeActionParser());
	const int sourceObsSize = savedAgent.GetObsSize();
	const int sourceNumActions = sourceActionParser->GetActionAmount();
	GGL::ModelSet sourceModels = {};
	savedAgent.MakeInferenceModels(
		sourceObsSize,
		sourceNumActions,
		torch::kCPU,
		sourceModels);
	sourceModels.Save(modelFolder);

	nlohmann::json registry = {
		{ "schema_version", 2 },
		{ "static_opponents", nlohmann::json::array({
			{
				{ "source_agent", agentName },
				{ "source_stage", "test-stage" },
				{ "source_version_id", "saved-setup" },
				{ "display_name", "Saved setup opponent" },
				{ "checkpoint_path", modelFolder.string() },
				{ "runtime", "source_agent" },
				{ "agent_setup", RocketForge::AgentSetupSnapshotToJson(savedSetup) }
			}
		}) }
	};
	std::ofstream(registryPath) << registry.dump(2);

	GGL::ModelSet learnerModels = {};
	GGL::PartialModelConfig learnerPolicy = {};
	learnerPolicy.layerSizes = { 7 };
	GGL::ModelManager::MakeModels(
		false,
		4,
		5,
		{},
		learnerPolicy,
		{},
		torch::kCPU,
		learnerModels);

	RLGC::EnvSetConfig envConfig = {};
	envConfig.tickSkip = 4;
	envConfig.actionDelay = 3;
	envConfig.envCreateCfg.maxTeamSize = 3;
	GGL::PolicyVersionManager manager(
		root / "Recent Policy Versions",
		4,
		1,
		{},
		envConfig,
		false);
	manager.staticOpponentsRegistryPath = registryPath;
	manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);

	ASSERT_EQ(manager.versions.size(), size_t(1));
	GGL::PolicyVersion& loaded = manager.versions.front();
	ASSERT_TRUE(loaded.inferenceAgent != nullptr);
	ASSERT_EQ(loaded.inferenceAgent->tickSkip, 8);
	ASSERT_EQ(loaded.inferenceAgent->actionDelay, 7);
	ASSERT_EQ(loaded.inferenceAgent->maxTeamSize, 2);
	ASSERT_EQ(loaded.inferenceObsSize, sourceObsSize);
	ASSERT_EQ(loaded.inferenceNumActions, sourceNumActions);
	ASSERT_EQ(loaded.models["policy"]->config.layerSizes, std::vector<int>({ 17, 11 }));
	ASSERT_EQ(loaded.inferenceCadenceRatio, 2);
	ASSERT_EQ(loaded.inferenceActionApplyStep, 1);
	ASSERT_EQ(loaded.inferenceActionApplyTick, 3);
	ASSERT_FALSE(loaded.inferenceRequiresEarlyAction);

	savedAgent.actionDelay = 0;
	registry["static_opponents"][0]["agent_setup"] =
		RocketForge::AgentSetupSnapshotToJson(RocketForge::CaptureAgentSetupSnapshot(savedAgent));
	std::ofstream(registryPath) << registry.dump(2);
	manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);

	ASSERT_EQ(manager.versions.size(), size_t(1));
	ASSERT_EQ(manager.versions.front().inferenceCadenceRatio, 2);
	ASSERT_EQ(manager.versions.front().inferenceActionApplyStep, 0);
	ASSERT_EQ(manager.versions.front().inferenceActionApplyTick, 0);
	ASSERT_TRUE(manager.versions.front().inferenceRequiresEarlyAction);

	// A native boundary after the learner's application boundary would require
	// mid-second-half inference scheduling. Reject it instead of approximating it.
	manager.baseEnvSetConfig.actionDelay = 2;
	savedAgent.actionDelay = 3;
	registry["static_opponents"][0]["agent_setup"] =
		RocketForge::AgentSetupSnapshotToJson(RocketForge::CaptureAgentSetupSnapshot(savedAgent));
	std::ofstream(registryPath) << registry.dump(2);
	bool rejectedUnsupportedExactTiming = false;
	try {
		manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);
	} catch (const std::runtime_error& e) {
		rejectedUnsupportedExactTiming =
			std::string(e.what()).find("exact timing combination is not supported") != std::string::npos;
	}
	ASSERT_TRUE(rejectedUnsupportedExactTiming);
	ASSERT_TRUE(manager.versions.empty());

	manager.ClearVersions();
	learnerModels.Free();
	sourceModels.Free();
	fs::remove_all(root);
}

TEST(PolicyVersionManager, StaticOpponentRegistryFailsClosed)
{
	fs::path root = TempPolicyVersionManagerRoot("static_registry_fails_closed");
	fs::path modelFolder = root / "checkpoint";
	fs::path registryPath = root / "static_opponents.json";
	fs::create_directories(modelFolder);

	GGL::ModelSet learnerModels = {};
	GGL::PartialModelConfig policy = {};
	policy.layerSizes = { 8 };
	GGL::ModelManager::MakeModels(false, 4, 5, {}, policy, {}, torch::kCPU, learnerModels);
	learnerModels.Save(modelFolder);

	nlohmann::json registry = {
		{ "schema_version", 2 },
		{ "static_opponents", nlohmann::json::array({
			{
				{ "source_agent", "example-agent" },
				{ "source_stage", "legacy" },
				{ "source_version_id", "loadable" },
				{ "checkpoint_path", modelFolder.string() }
			},
			{
				{ "source_agent", "missing-source-agent" },
				{ "source_stage", "test" },
				{ "source_version_id", "broken" },
				{ "checkpoint_path", modelFolder.string() },
				{ "runtime", "source_agent" }
			}
		}) }
	};
	std::ofstream(registryPath) << registry.dump(2);

	RLGC::EnvSetConfig envConfig = {};
	envConfig.tickSkip = 8;
	envConfig.actionDelay = 7;
	GGL::PolicyVersionManager manager(
		root / "Recent Policy Versions",
		4,
		1,
		{},
		envConfig,
		false);
	manager.staticOpponentsRegistryPath = registryPath;

	bool threw = false;
	try {
		manager.LoadStaticOpponents(learnerModels, CreateRegisteredAgent);
	} catch (const std::runtime_error& e) {
		threw = std::string(e.what()).find("missing-source-agent") != std::string::npos;
	}
	ASSERT_TRUE(threw);
	ASSERT_TRUE(manager.versions.empty());

	learnerModels.Free();
	fs::remove_all(root);
}

TEST(SkillRating, SeedFromPrevious_CarriesPreviousRatings)
{
	GGL::SkillTrackerConfig config = {};
	config.initialMu = -50.0f;
	config.initialSigma = 400.0f;

	GGL::SkillRating previous = {};
	previous.data["1v1"] = GGL::SkillEstimate{ 2000.0f, 30.0f };
	previous.data["3v3"] = GGL::SkillEstimate{ 1500.0f, 45.0f };

	GGL::SkillRating seeded = {};
	seeded.SeedFromPrevious(previous, config);

	ASSERT_EQ(seeded.data["1v1"].mu, 2000.0f);
	ASSERT_EQ(seeded.data["1v1"].sigma, 30.0f);
	ASSERT_EQ(seeded.data["2v2"].mu, config.initialMu);
	ASSERT_EQ(seeded.data["2v2"].sigma, config.initialSigma);
	ASSERT_EQ(seeded.data["3v3"].mu, 1500.0f);
	ASSERT_EQ(seeded.data["3v3"].sigma, 45.0f);
}

TEST(SkillRating, SeedForSkillTest_ResetsSigmaAndCarriesMu)
{
	GGL::SkillTrackerConfig config = {};
	config.initialMu = -50.0f;
	config.initialSigma = 400.0f;

	GGL::SkillRating previous = {};
	previous.data["1v1"] = GGL::SkillEstimate{ 2000.0f, 30.0f };
	previous.data["3v3"] = GGL::SkillEstimate{ 1500.0f, 15.0f };

	GGL::SkillRating seeded = {};
	seeded.SeedForSkillTest(previous, config);

	ASSERT_EQ(seeded.data["1v1"].mu, 2000.0f);
	ASSERT_EQ(seeded.data["1v1"].sigma, config.initialSigma);
	ASSERT_EQ(seeded.data["2v2"].mu, config.initialMu);
	ASSERT_EQ(seeded.data["2v2"].sigma, config.initialSigma);
	ASSERT_EQ(seeded.data["3v3"].mu, 1500.0f);
	ASSERT_EQ(seeded.data["3v3"].sigma, config.initialSigma);
}

TEST(SkillRating, SeedForSkillTestWithDriftGrowsSigmaWithElapsedTimesteps)
{
	GGL::SkillTrackerConfig config = {};
	config.initialSigma = 300.0f;
	config.ratingDriftSigmaPer100MTimesteps = 40.0f;

	GGL::SkillRating previous = {};
	previous.data["1v1"] = GGL::SkillEstimate{ 2000.0f, 30.0f };

	GGL::SkillRating seeded = {};
	seeded.SeedForSkillTestWithDrift(previous, 100'000'000, config);

	ASSERT_EQ(seeded.data["1v1"].mu, 2000.0f);
	ASSERT_NEAR(seeded.data["1v1"].sigma, 50.0f, 0.01f);
	ASSERT_EQ(seeded.data["2v2"].sigma, config.initialSigma);
}

TEST(SkillRating, SeedForSkillTestWithDriftKeepsSigmaWhenNoTimestepsElapsed)
{
	GGL::SkillTrackerConfig config = {};

	GGL::SkillRating previous = {};
	previous.data["1v1"] = GGL::SkillEstimate{ 2000.0f, 30.0f };

	GGL::SkillRating seeded = {};
	seeded.SeedForSkillTestWithDrift(previous, 0, config);

	ASSERT_EQ(seeded.data["1v1"].mu, 2000.0f);
	ASSERT_NEAR(seeded.data["1v1"].sigma, 30.0f, 0.001f);
}

TEST(PolicyVersionManager, GetTimestepsSinceLastSkillTest)
{
	ASSERT_EQ(GGL::PolicyVersionManager::GetTimestepsSinceLastSkillTest(500, 200), uint64_t(300));
	ASSERT_EQ(GGL::PolicyVersionManager::GetTimestepsSinceLastSkillTest(500, 0), uint64_t(0));
	ASSERT_EQ(GGL::PolicyVersionManager::GetTimestepsSinceLastSkillTest(100, 200), uint64_t(0));
	ASSERT_EQ(GGL::PolicyVersionManager::GetTimestepsSinceLastSkillTest(200, 200), uint64_t(0));
}

TEST(SkillRating, SeedForSkillTestWithDriftCapsAtInitialSigma)
{
	GGL::SkillTrackerConfig config = {};
	config.initialSigma = 300.0f;

	GGL::SkillRating previous = {};
	previous.data["1v1"] = GGL::SkillEstimate{ 0.0f, 290.0f };

	GGL::SkillRating seeded = {};
	seeded.SeedForSkillTestWithDrift(previous, 10'000'000'000ull, config);

	ASSERT_EQ(seeded.data["1v1"].sigma, 300.0f);
}

TEST(PolicyVersionManager, RunningStatsRoundTripLastRunTimestep)
{
	fs::path root = TempPolicyVersionManagerRoot("running_stats_last_run");
	fs::path saveFolder = root / "Recent Policy Versions";

	GGL::SkillTrackerConfig config = {};
	config.enabled = true;
	RLGC::EnvSetConfig envConfig = {};

	GGL::PolicyVersionManager saved(saveFolder, 1, 1, config, envConfig, false);
	saved.skill.lastRunTimestep = 123456;

	nlohmann::json json = {};
	saved.AddRunningStatsToJSON(json);

	GGL::PolicyVersionManager loaded(saveFolder, 1, 1, config, envConfig, false);
	loaded.LoadRunningStatsFromJSON(json);
	ASSERT_EQ(loaded.skill.lastRunTimestep, uint64_t(123456));

	GGL::PolicyVersionManager legacy(saveFolder, 1, 1, config, envConfig, false);
	legacy.skill.lastRunTimestep = 99;
	legacy.LoadRunningStatsFromJSON(nlohmann::json::object());
	ASSERT_EQ(legacy.skill.lastRunTimestep, uint64_t(0));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, UpdateSkillEstimateAgainstFixedOpponent_MutatesOnlyCurrent)
{
	GGL::SkillTrackerConfig config = {};
	config.beta = 150.0f;
	config.minSigma = 15.0f;

	GGL::SkillEstimate current = { 1000.0f, 300.0f };
	const GGL::SkillEstimate opponent = { 900.0f, 15.0f };

	GGL::PolicyVersionManager::UpdateSkillEstimateAgainstFixedOpponent(current, opponent, false, config);

	ASSERT_TRUE(current.mu < 1000.0f);
	ASSERT_TRUE(current.sigma < 300.0f);
	ASSERT_EQ(opponent.mu, 900.0f);
	ASSERT_EQ(opponent.sigma, 15.0f);
}

TEST(PolicyVersionManager, BuildSkillBenchmarkPool_IncludesRatingCoverage)
{
	fs::path root = TempPolicyVersionManagerRoot("benchmark_pool_coverage");
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	GGL::SkillTrackerConfig config = {};
	config.benchmarkPruneMaxExpectedWinProb = 0.995f;
	GGL::SkillRating current = MakeSkillRating(1000.0f);

	std::vector<GGL::SkillBenchmark> benchmarks = {
		MakeBenchmark(benchmarkRoot, "close_even", "20260101-000000", 1000.0f, 1000),
		MakeBenchmark(benchmarkRoot, "close_low", "20260101-000001", 980.0f, 2000),
		MakeBenchmark(benchmarkRoot, "close_high", "20260101-000002", 1020.0f, 3000),
		MakeBenchmark(benchmarkRoot, "lower_near", "20260101-000003", 900.0f, 4000),
		MakeBenchmark(benchmarkRoot, "lower_mid", "20260101-000004", 700.0f, 5000),
		MakeBenchmark(benchmarkRoot, "lower_extra", "20260101-000005", 600.0f, 6000),
		MakeBenchmark(benchmarkRoot, "lower_far", "20260101-000006", 500.0f, 7000),
		MakeBenchmark(benchmarkRoot, "higher_near", "20260101-000007", 1100.0f, 8000),
		MakeBenchmark(benchmarkRoot, "higher_mid", "20260101-000008", 1300.0f, 9000),
		MakeBenchmark(benchmarkRoot, "higher_extra", "20260101-000009", 1400.0f, 10000),
		MakeBenchmark(benchmarkRoot, "higher_far", "20260101-000010", 1500.0f, 11000)
	};

	std::vector<int> selected = GGL::PolicyVersionManager::BuildSkillBenchmarkPool(
		current,
		"1v1",
		benchmarks,
		config,
		8);

	ASSERT_EQ(selected.size(), size_t(8));
	ASSERT_TRUE(HasBenchmarkIndex(benchmarks, selected, "close_even"));
	ASSERT_TRUE(HasBenchmarkIndex(benchmarks, selected, "lower_far"));
	ASSERT_TRUE(HasBenchmarkIndex(benchmarks, selected, "higher_far"));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, LoadBenchmarks_IgnoresNonLineageBenchmarksForScratchSegment)
{
	fs::path root = TempPolicyVersionManagerRoot("scratch_benchmark_lineage");
	fs::path saveFolder = root / "Recent Policy Versions";
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	WriteSkillBenchmark(benchmarkRoot, "v10.05_100_stage", "v10.05", 100);
	WriteSkillBenchmark(benchmarkRoot, "v10.06_200_stage", "", 200);

	GGL::SkillTrackerConfig config = {};
	config.enabled = true;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.versionId = "v11.00";

	GGL::ModelSet models;
	manager.LoadBenchmarks(models, 0);

	ASSERT_TRUE(manager.skill.benchmarks.empty());

	fs::remove_all(root);
}

TEST(PolicyVersionManager, LoadBenchmarks_KeepsCurrentLineageBenchmarks)
{
	fs::path root = TempPolicyVersionManagerRoot("current_benchmark_lineage");
	fs::path saveFolder = root / "Recent Policy Versions";
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	WriteSkillBenchmark(benchmarkRoot, "v11.00_100_stage", "v11.00", 100);
	WriteSkillBenchmark(benchmarkRoot, "v11.01_200_stage", "v11.01", 200);
	WriteSkillBenchmark(benchmarkRoot, "v10.05_300_stage", "v10.05", 300);

	GGL::SkillTrackerConfig config = {};
	config.enabled = true;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.versionId = "v11.01";
	manager.recentPolicyVersionSourceIds = { "v11.00" };

	GGL::ModelSet models;
	manager.LoadBenchmarks(models, 200);

	ASSERT_EQ(manager.skill.benchmarks.size(), size_t(2));
	ASSERT_TRUE(HasBenchmark(manager.skill.benchmarks, "v11.00_100_stage"));
	ASSERT_TRUE(HasBenchmark(manager.skill.benchmarks, "v11.01_200_stage"));
	ASSERT_FALSE(HasBenchmark(manager.skill.benchmarks, "v10.05_300_stage"));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, LoadBenchmarks_PrunesFutureLineageBenchmarks)
{
	fs::path root = TempPolicyVersionManagerRoot("future_benchmark_lineage");
	fs::path saveFolder = root / "Recent Policy Versions";
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	WriteSkillBenchmark(benchmarkRoot, "v11.00_100_stage", "v11.00", 100);
	WriteSkillBenchmark(benchmarkRoot, "v11.00_300_stage", "v11.00", 300);

	GGL::SkillTrackerConfig config = {};
	config.enabled = true;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.versionId = "v11.00";

	GGL::ModelSet models;
	manager.LoadBenchmarks(models, 200);

	ASSERT_EQ(manager.skill.benchmarks.size(), size_t(1));
	ASSERT_TRUE(HasBenchmark(manager.skill.benchmarks, "v11.00_100_stage"));
	ASSERT_TRUE(fs::exists(benchmarkRoot / "v11.00_100_stage"));
	ASSERT_FALSE(fs::exists(benchmarkRoot / "v11.00_300_stage"));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, EnsureInitialSkillBenchmark_ReplacesInvalidStartupBenchmarks)
{
	fs::path root = TempPolicyVersionManagerRoot("startup_initial_anchor");
	fs::path saveFolder = root / "Recent Policy Versions";
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	WriteSkillBenchmark(benchmarkRoot, "v10.00_100_old_stage", "v10.00", 100);

	GGL::SkillTrackerConfig config = {};
	config.enabled = true;
	config.minTimeSteps = 123;
	config.maxTimeSteps = 456;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.versionId = "v11.00";
	manager.sourceStageName = "stage";

	GGL::ModelSet models;
	manager.LoadBenchmarks(models, 0);
	bool addedInitialAnchor = manager.EnsureInitialSkillBenchmark(models, 0);

	ASSERT_TRUE(addedInitialAnchor);
	ASSERT_EQ(manager.skill.benchmarks.size(), size_t(1));
	ASSERT_FALSE(fs::exists(benchmarkRoot / "v10.00_100_old_stage"));
	ASSERT_TRUE(fs::exists(manager.skill.benchmarks[0].folder / "STATS.json"));
	ASSERT_EQ(manager.skill.nextRunTimestep, uint64_t(123));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, SaveBenchmarks_DoesNotRewriteLoadedBenchmarkStats)
{
	fs::path root = TempPolicyVersionManagerRoot("fixed_anchor_stats");
	fs::path saveFolder = root / "Recent Policy Versions";
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;
	fs::path benchmarkFolder = benchmarkRoot / "fixed_anchor";
	fs::create_directories(benchmarkFolder);

	const std::string originalStats = "{ \"sentinel\": \"keep\" }";
	{
		std::ofstream fOut(benchmarkFolder / "STATS.json");
		fOut << originalStats;
	}

	GGL::SkillTrackerConfig config = {};
	config.enabled = true;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);

	GGL::SkillBenchmark benchmark = MakeBenchmark(benchmarkRoot, "fixed_anchor", "20260101-000000", 1000.0f, 1000);
	benchmark.modelsSaved = true;
	manager.skill.benchmarks = { benchmark };

	manager.SaveBenchmarks();

	{
		std::ifstream fIn(benchmarkFolder / "STATS.json");
		std::string savedStats((std::istreambuf_iterator<char>(fIn)), std::istreambuf_iterator<char>());
		ASSERT_EQ(savedStats, originalStats);
	}

	fs::remove_all(root);
}

TEST(PolicyVersionManager, PruneSavedVersions_LimitsSharedCacheAcrossSegments)
{
	fs::path root = TempPolicyVersionManagerRoot("shared_cache_limit");
	fs::path saveFolder = root / "Recent Policy Versions";

	WriteRecentPolicyVersion(saveFolder, "v1.00", 25);
	WriteRecentPolicyVersion(saveFolder, "v1.01", 50);
	WriteRecentPolicyVersion(saveFolder, "v1.02", 75);
	WriteRecentPolicyVersion(saveFolder, "v1.03", 100);

	GGL::SkillTrackerConfig config = {};
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 3, 1, config, envConfig, true);

	manager.PruneSavedVersions(1000);

	ASSERT_FALSE(HasRecentPolicyVersion(saveFolder, "v1.00", 25));
	ASSERT_TRUE(HasRecentPolicyVersion(saveFolder, "v1.01", 50));
	ASSERT_TRUE(HasRecentPolicyVersion(saveFolder, "v1.02", 75));
	ASSERT_TRUE(HasRecentPolicyVersion(saveFolder, "v1.03", 100));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, PruneSavedVersions_RemovesNewerUnusableBeforeOlderUsable)
{
	fs::path root = TempPolicyVersionManagerRoot("prune_unusable_first");
	fs::path saveFolder = root / "Recent Policy Versions";

	WriteRecentPolicyVersion(saveFolder, "v1.00", 25);
	WriteRecentPolicyVersion(saveFolder, "v1.00", 50);
	WriteRecentPolicyVersion(saveFolder, "v2.00", 150);
	WriteRecentPolicyVersion(saveFolder, "v2.00", 200);

	GGL::SkillTrackerConfig config = {};
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 2, 1, config, envConfig, true);

	manager.PruneSavedVersions(100);

	ASSERT_TRUE(HasRecentPolicyVersion(saveFolder, "v1.00", 25));
	ASSERT_TRUE(HasRecentPolicyVersion(saveFolder, "v1.00", 50));
	ASSERT_FALSE(HasRecentPolicyVersion(saveFolder, "v2.00", 150));
	ASSERT_FALSE(HasRecentPolicyVersion(saveFolder, "v2.00", 200));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, LoadVersions_PrunesLegacyRecentVersionsWithoutCompatibilityMetadata)
{
	fs::path root = TempPolicyVersionManagerRoot("legacy_recent_policy");
	fs::path saveFolder = root / "Recent Policy Versions";

	WriteRecentPolicyVersion(saveFolder, "v1.00", 50);

	GGL::SkillTrackerConfig config = {};
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 3, 1, config, envConfig, true);
	manager.versionId = "v1.00";

	GGL::ModelSet models;
	manager.LoadVersions(models, 100);

	ASSERT_EQ(manager.versions.size(), size_t(0));
	ASSERT_FALSE(HasRecentPolicyVersion(saveFolder, "v1.00", 50));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, LoadVersions_PrunesIncompatibleRecentVersions)
{
	fs::path root = TempPolicyVersionManagerRoot("incompatible_recent_policy");
	fs::path saveFolder = root / "Recent Policy Versions";

	WriteRecentPolicyVersion(saveFolder, "v1.00", 50, "{ \"schema_version\": 999 }");

	GGL::SkillTrackerConfig config = {};
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 3, 1, config, envConfig, true);
	manager.versionId = "v1.00";

	GGL::ModelSet models;
	manager.LoadVersions(models, 100);

	ASSERT_EQ(manager.versions.size(), size_t(0));
	ASSERT_FALSE(HasRecentPolicyVersion(saveFolder, "v1.00", 50));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, SaveVersions_WritesLoadableCompatibilityMetadata)
{
	fs::path root = TempPolicyVersionManagerRoot("loadable_recent_policy");
	fs::path saveFolder = root / "Recent Policy Versions";

	GGL::SkillTrackerConfig config = {};
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(saveFolder, 3, 1, config, envConfig, true);
	manager.versionId = "v1.00";

	GGL::ModelSet models;
	GGL::PolicyVersion& saved = manager.AddVersion(models, 50);
	saved.sourceVersionId = "v1.00";
	saved.saveWithCurrentVersion = true;
	manager.SaveVersions(100);

	{
		fs::path statsPath = MakeRecentPolicyVersionFolder(saveFolder, "v1.00", 50) / "STATS.json";
		std::ifstream fIn(statsPath);
		nlohmann::json stats = nlohmann::json::parse(fIn);
		ASSERT_TRUE(stats.contains("policy_compatibility"));
	}

	manager.ClearVersions();
	manager.LoadVersions(models, 100);

	ASSERT_EQ(manager.versions.size(), size_t(1));
	ASSERT_EQ(manager.versions[0].sourceVersionId, std::string("v1.00"));

	manager.ClearVersions();
	fs::remove_all(root);
}

TEST(PolicyVersionManager, PruneInMemoryVersions_PreservesActiveEpisodeVersion)
{
	fs::path root = TempPolicyVersionManagerRoot("protected_active_version");
	GGL::SkillTrackerConfig config = {};
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 1, 1, config, envConfig, true);
	GGL::ModelSet models;

	GGL::PolicyVersion& active = manager.AddVersion(models, 100);
	active.sourceVersionId = "v1.00";
	manager.SetProtectedVersions({ GGL::PolicyVersionKey{ "v1.00", 100 } });

	GGL::PolicyVersion& newest = manager.AddVersion(models, 200);
	newest.sourceVersionId = "v1.00";
	manager.PruneInMemoryVersions(GGL::PolicyVersionKey{ "v1.00", 200 });

	// The cap may be exceeded while both versions are protected by a live episode
	// or by the just-created snapshot.
	ASSERT_EQ(manager.versions.size(), size_t(2));
	ASSERT_TRUE(manager.FindVersion({ "v1.00", 100 }) != nullptr);
	ASSERT_TRUE(manager.FindVersion({ "v1.00", 200 }) != nullptr);

	manager.SetProtectedVersions({});
	ASSERT_EQ(manager.versions.size(), size_t(1));
	ASSERT_TRUE(manager.FindVersion({ "v1.00", 100 }) == nullptr);
	ASSERT_TRUE(manager.FindVersion({ "v1.00", 200 }) != nullptr);

	manager.ClearVersions();
	fs::remove_all(root);
}

TEST(PolicyVersionManager, PruneBenchmarks_RemovesOutOfRangeBenchmark)
{
	struct Case {
		const char* name;
		float rating;
	};

	const Case cases[] = {
		{ "too_strong", 800.0f },
		{ "too_weak", -800.0f },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.name);
		fs::path root = TempPolicyVersionManagerRoot(testCase.name);
		fs::path saveFolder = root / "Recent Policy Versions";
		fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

		GGL::SkillTrackerConfig config = {};
		config.maxSkillBenchmarks = 2;
		config.benchmarkPruneMaxExpectedWinProb = 0.995f;
		RLGC::EnvSetConfig envConfig = {};
		GGL::PolicyVersionManager manager(saveFolder, 1, 1, config, envConfig, false);
		manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
		manager.skill.curRatings = MakeSkillRating(0.0f);
		manager.skill.benchmarks = {
			MakeBenchmark(benchmarkRoot, testCase.name, "20260101-000000", testCase.rating, 1000),
			MakeBenchmark(benchmarkRoot, "near_current", "20260102-000000", 0.0f, 2000),
			MakeBenchmark(benchmarkRoot, "newest_near_current", "20260103-000000", 20.0f, 3000)
		};

		manager.PruneBenchmarks();

		ASSERT_EQ(manager.skill.benchmarks.size(), size_t(2));
		ASSERT_FALSE(HasBenchmark(manager.skill.benchmarks, testCase.name));
		ASSERT_FALSE(fs::exists(benchmarkRoot / testCase.name));

		fs::remove_all(root);
	}
}

TEST(PolicyVersionManager, PruneBenchmarks_RemovesOneSidedBenchmarkUnderCap)
{
	fs::path root = TempPolicyVersionManagerRoot("one_sided_under_cap");
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	GGL::SkillTrackerConfig config = {};
	config.maxSkillBenchmarks = 10;
	config.benchmarkPruneMaxExpectedWinProb = 0.995f;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.skill.curRatings = MakeSkillRating(0.0f);
	manager.skill.benchmarks = {
		MakeBenchmark(benchmarkRoot, "too_strong", "20260101-000000", 800.0f, 1000),
		MakeBenchmark(benchmarkRoot, "near_current", "20260102-000000", 0.0f, 2000)
	};

	manager.PruneBenchmarks();

	ASSERT_EQ(manager.skill.benchmarks.size(), size_t(1));
	ASSERT_FALSE(HasBenchmark(manager.skill.benchmarks, "too_strong"));
	ASSERT_TRUE(HasBenchmark(manager.skill.benchmarks, "near_current"));
	ASSERT_FALSE(fs::exists(benchmarkRoot / "too_strong"));
	ASSERT_TRUE(fs::exists(benchmarkRoot / "near_current"));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, PruneBenchmarks_KeepsHighRatedCloseBenchmark)
{
	fs::path root = TempPolicyVersionManagerRoot("high_rated_close");
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	GGL::SkillTrackerConfig config = {};
	config.maxSkillBenchmarks = 10;
	config.benchmarkPruneMaxExpectedWinProb = 0.995f;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.skill.curRatings = MakeSkillRating(3550.0f);
	manager.skill.benchmarks = {
		MakeBenchmark(benchmarkRoot, "close_high_rating", "20260101-000000", 3550.0f, 1000)
	};

	manager.PruneBenchmarks();

	ASSERT_EQ(manager.skill.benchmarks.size(), size_t(1));
	ASSERT_TRUE(HasBenchmark(manager.skill.benchmarks, "close_high_rating"));
	ASSERT_TRUE(fs::exists(benchmarkRoot / "close_high_rating"));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, PruneBenchmarks_RemovesBenchmarkWhenAnyModeIsOneSided)
{
	fs::path root = TempPolicyVersionManagerRoot("one_sided_mode");
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	GGL::SkillTrackerConfig config = {};
	config.maxSkillBenchmarks = 10;
	config.benchmarkPruneMaxExpectedWinProb = 0.995f;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.skill.curRatings = MakeSkillRating(0.0f);

	GGL::SkillBenchmark mixed = MakeBenchmark(benchmarkRoot, "mixed_mode", "20260101-000000", 0.0f, 1000);
	mixed.ratings.data["1v1"] = GGL::SkillEstimate{ 800.0f, 15.0f };
	manager.skill.benchmarks = {
		mixed,
		MakeBenchmark(benchmarkRoot, "near_current", "20260102-000000", 0.0f, 2000)
	};

	manager.PruneBenchmarks();

	ASSERT_EQ(manager.skill.benchmarks.size(), size_t(1));
	ASSERT_FALSE(HasBenchmark(manager.skill.benchmarks, "mixed_mode"));
	ASSERT_TRUE(HasBenchmark(manager.skill.benchmarks, "near_current"));
	ASSERT_FALSE(fs::exists(benchmarkRoot / "mixed_mode"));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, RunFixedSkillSeriesPlaysEveryScheduledGame)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	struct Case {
		std::string label;
		int opponentTickSkip;
		bool explicitSubject;
	};
	const Case cases[] = {
		{ "learner timing", 8, false },
		{ "independent opponent timing", 4, false },
		{ "explicit subject", 8, true },
	};

	const int learnerTickSkip = 8;
	const int teamSize = 1;
	const int gamesPerOpponent = 3;
	const int scheduledGamesPerOpponent = 4;

	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 4 };
	GGL::ModelSet opponentModels = {};
	GGL::ModelManager::MakeModels(false, 2, 1, {}, policyConfig, {}, torch::kCPU, opponentModels);
	GGL::PPOLearner learner(2, 1, MakeFixedSeriesPPOConfig(), torch::kCPU);

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		GGL::SkillTrackerConfig skillConfig = {};
		skillConfig.maxArenas = 4;
		skillConfig.maxNoScoreGameSeconds = 1.0f;
		const bool managePolicyVersions = false;
		GGL::PolicyVersionManager manager(
			RocketForgeTests::TestTempDir("fixed_series") / "Recent Policy Versions",
			4,
			1,
			skillConfig,
			MakeFixedSeriesEnvConfig(learnerTickSkip),
			managePolicyVersions);

		const std::vector<GGL::FixedSeriesOpponent> opponents = {
			MakeFixedSeriesOpponent(&opponentModels, "first", testCase.opponentTickSkip),
			MakeFixedSeriesOpponent(&opponentModels, "second", testCase.opponentTickSkip),
		};
		const GGL::FixedSeriesOpponent subject = MakeFixedSeriesOpponent(&opponentModels, "subject", learnerTickSkip);

		const std::vector<GGL::FixedSeriesRecord> records = manager.RunFixedSkillSeries(
			&learner,
			opponents,
			teamSize,
			gamesPerOpponent,
			testCase.explicitSubject ? &subject : nullptr);

		ASSERT_EQ(records.size(), opponents.size());
		for (const GGL::FixedSeriesRecord& record : records)
			ASSERT_EQ(record.wins + record.losses + record.noScoreGames, scheduledGamesPerOpponent);
	}

	opponentModels.Free();
}

TEST(PolicyVersionManager, FindBenchmarkByIdReturnsMinusOneOncePruned)
{
	std::vector<GGL::SkillBenchmark> benchmarks(2);
	benchmarks[0].id = "older";
	benchmarks[1].id = "previous";

	ASSERT_EQ(GGL::PolicyVersionManager::FindBenchmarkById(benchmarks, "previous"), 1);

	benchmarks.erase(benchmarks.begin() + 1);

	ASSERT_EQ(GGL::PolicyVersionManager::FindBenchmarkById(benchmarks, "previous"), -1);
}

TEST(PolicyVersionManager, SelectAnchorBenchmarkKeepsUsefulAnchor)
{
	fs::path benchmarkRoot = RocketForgeTests::TestTempDir("anchor_keep");
	GGL::SkillTrackerConfig config = {};
	const std::vector<GGL::SkillBenchmark> benchmarks = {
		MakeBenchmark(benchmarkRoot, "anchor", "20260101-000000", 1000.0f, 1000),
		MakeBenchmark(benchmarkRoot, "higher", "20260102-000000", 1040.0f, 2000),
	};

	ASSERT_EQ(GGL::PolicyVersionManager::SelectAnchorBenchmark(benchmarks, "anchor", MakeSkillRating(1050.0f), config), 0);

	fs::remove_all(benchmarkRoot);
}

TEST(PolicyVersionManager, SelectAnchorBenchmarkRetiresLopsidedAnchorToHighestRated)
{
	fs::path benchmarkRoot = RocketForgeTests::TestTempDir("anchor_retire");
	GGL::SkillTrackerConfig config = {};
	config.anchorRetireExpectedWinProb = 0.90f;
	const std::vector<GGL::SkillBenchmark> benchmarks = {
		MakeBenchmark(benchmarkRoot, "a", "20260101-000000", 500.0f, 1000),
		MakeBenchmark(benchmarkRoot, "b", "20260102-000000", 900.0f, 2000),
		MakeBenchmark(benchmarkRoot, "c", "20260103-000000", 1000.0f, 3000),
	};

	ASSERT_EQ(GGL::PolicyVersionManager::SelectAnchorBenchmark(benchmarks, "a", MakeSkillRating(1100.0f), config), 2);

	fs::remove_all(benchmarkRoot);
}

TEST(PolicyVersionManager, SelectAnchorBenchmarkPicksNewAnchorWhenMissing)
{
	fs::path benchmarkRoot = RocketForgeTests::TestTempDir("anchor_missing");
	GGL::SkillTrackerConfig config = {};
	config.anchorRetireExpectedWinProb = 0.90f;
	const std::vector<GGL::SkillBenchmark> benchmarks = {
		MakeBenchmark(benchmarkRoot, "low", "20260101-000000", 900.0f, 1000),
		MakeBenchmark(benchmarkRoot, "high", "20260102-000000", 1000.0f, 2000),
		MakeBenchmark(benchmarkRoot, "too_strong", "20260103-000000", 2000.0f, 3000),
	};

	struct Case {
		std::string label;
		std::string currentAnchorId;
		float currentRating;
		int expectedIdx;
	};
	const Case cases[] = {
		{ "anchor not loaded", "gone", 1000.0f, 1 },
		{ "no anchor yet", "", 1000.0f, 1 },
		{ "every benchmark lopsided", "gone", 5000.0f, -1 },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		const int selectedIdx = GGL::PolicyVersionManager::SelectAnchorBenchmark(
			benchmarks,
			testCase.currentAnchorId,
			MakeSkillRating(testCase.currentRating),
			config);
		ASSERT_EQ(selectedIdx, testCase.expectedIdx);
	}

	fs::remove_all(benchmarkRoot);
}

TEST(PolicyVersionManager, PruneBenchmarksKeepsAnchorBenchmark)
{
	fs::path root = TempPolicyVersionManagerRoot("anchor_prune");
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	// The anchor sits outside the 0.995 prune window but inside the 0.9999 retire window.
	GGL::SkillTrackerConfig config = {};
	config.maxSkillBenchmarks = 10;
	config.benchmarkPruneMaxExpectedWinProb = 0.995f;
	config.anchorRetireExpectedWinProb = 0.9999f;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.skill.curRatings = MakeSkillRating(0.0f);
	manager.skill.anchorBenchmarkId = "strong_anchor";
	manager.skill.benchmarks = {
		MakeBenchmark(benchmarkRoot, "strong_anchor", "20260101-000000", 650.0f, 1000),
		MakeBenchmark(benchmarkRoot, "near_current", "20260102-000000", 0.0f, 2000)
	};

	manager.PruneBenchmarks();

	ASSERT_EQ(manager.skill.benchmarks.size(), size_t(2));
	ASSERT_TRUE(HasBenchmark(manager.skill.benchmarks, "strong_anchor"));
	ASSERT_TRUE(fs::exists(benchmarkRoot / "strong_anchor"));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, PruneBenchmarksRemovesRetiredAnchorBenchmark)
{
	fs::path root = TempPolicyVersionManagerRoot("anchor_prune_retired");
	fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

	GGL::SkillTrackerConfig config = {};
	config.maxSkillBenchmarks = 10;
	config.benchmarkPruneMaxExpectedWinProb = 0.995f;
	config.anchorRetireExpectedWinProb = 0.90f;
	RLGC::EnvSetConfig envConfig = {};
	GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 1, 1, config, envConfig, false);
	manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
	manager.skill.curRatings = MakeSkillRating(0.0f);
	manager.skill.anchorBenchmarkId = "too_strong";
	manager.skill.benchmarks = {
		MakeBenchmark(benchmarkRoot, "too_strong", "20260101-000000", 800.0f, 1000),
		MakeBenchmark(benchmarkRoot, "near_current", "20260102-000000", 0.0f, 2000)
	};

	manager.PruneBenchmarks();

	ASSERT_EQ(manager.skill.benchmarks.size(), size_t(1));
	ASSERT_FALSE(HasBenchmark(manager.skill.benchmarks, "too_strong"));
	ASSERT_FALSE(fs::exists(benchmarkRoot / "too_strong"));

	fs::remove_all(root);
}

TEST(PolicyVersionManager, PruneBenchmarksIgnoresAnchorWhileAnchorPhaseIsUnused)
{
	struct Case {
		std::string label;
		int referenceGamesPerOpponent;
		bool hasUserReferences;
	};
	const Case cases[] = {
		{ "reference games off", 0, false },
		{ "user references replace the anchor", 64, true },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		fs::path root = TempPolicyVersionManagerRoot(testCase.referenceGamesPerOpponent == 0 ? "games_off" : "user_references");
		fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

		// Same windows as PruneBenchmarksKeepsAnchorBenchmark, where the anchor would be kept.
		GGL::SkillTrackerConfig config = {};
		config.maxSkillBenchmarks = 10;
		config.benchmarkPruneMaxExpectedWinProb = 0.995f;
		config.anchorRetireExpectedWinProb = 0.9999f;
		config.referenceGamesPerOpponent = testCase.referenceGamesPerOpponent;
		RLGC::EnvSetConfig envConfig = {};
		GGL::PolicyVersionManager manager(root / "Recent Policy Versions", 1, 1, config, envConfig, false);
		manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);
		manager.skill.curRatings = MakeSkillRating(0.0f);
		manager.skill.anchorBenchmarkId = "strong_anchor";
		manager.skill.benchmarks = {
			MakeBenchmark(benchmarkRoot, "strong_anchor", "20260101-000000", 650.0f, 1000),
			MakeBenchmark(benchmarkRoot, "near_current", "20260102-000000", 0.0f, 2000)
		};

		std::vector<GGL::PolicyVersion> userReferences(testCase.hasUserReferences ? 1 : 0);
		manager.skillReferences = &userReferences;

		manager.PruneBenchmarks();

		ASSERT_EQ(manager.skill.benchmarks.size(), size_t(1));
		ASSERT_FALSE(HasBenchmark(manager.skill.benchmarks, "strong_anchor"));

		fs::remove_all(root);
	}
}

TEST(PolicyVersionManager, ReportFixedOpponentMMRWritesKeysOnlyForScoredRecords)
{
	GGL::SkillTrackerConfig config = {};
	const std::string ratingKey = "SkillTest/Anchor/1v1";
	const std::string sigmaKey = "SkillTest/AnchorSigma/1v1";
	const float baseRating = 1000.0f;

	GGL::Report unscoredReport = {};
	const GGL::FixedSeriesRecord unscored = { 0, 0, 8 };
	const bool reportedUnscored = GGL::PolicyVersionManager::ReportFixedOpponentMMR(
		unscoredReport,
		ratingKey,
		sigmaKey,
		"anchor",
		"1v1",
		baseRating,
		unscored,
		config);

	ASSERT_FALSE(reportedUnscored);
	ASSERT_TRUE(unscoredReport.data.empty());

	GGL::Report scoredReport = {};
	const GGL::FixedSeriesRecord scored = { 20, 12, 2 };
	const bool reportedScored = GGL::PolicyVersionManager::ReportFixedOpponentMMR(
		scoredReport,
		ratingKey,
		sigmaKey,
		"anchor",
		"1v1",
		baseRating,
		scored,
		config);

	ASSERT_TRUE(reportedScored);
	ASSERT_EQ(scoredReport.data.size(), size_t(2));
	ASSERT_TRUE(scoredReport.Has(ratingKey));
	ASSERT_TRUE(scoredReport.Has(sigmaKey));
	ASSERT_TRUE(std::isfinite(scoredReport[ratingKey]));
	ASSERT_TRUE(std::isfinite(scoredReport[sigmaKey]));
	ASSERT_GT(scoredReport[ratingKey], baseRating);
	ASSERT_GT(scoredReport[sigmaKey], 0.0);
}

TEST(PolicyVersionManager, RunSkillMatchesSchedulesFromEveryPhaseAndOmitsPendingReferences)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	// Idle cars never score, so the one main-phase game ends the test as inconclusive and the fixed phases still run.
	const int mainPhaseGames = 1;
	const int anchorGames = 2;
	const int calibrationGames = 4;

	struct Case {
		std::string label;
		bool pendingReference;
		int expectedGames;
	};
	const Case cases[] = {
		{ "auto anchor", false, mainPhaseGames + anchorGames },
		{ "pending reference", true, mainPhaseGames + calibrationGames },
	};

	const int learnerTickSkip = 8;
	GGL::PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 4 };
	GGL::PPOLearner learner(2, 1, MakeFixedSeriesPPOConfig(), torch::kCPU);

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		fs::path root = TempPolicyVersionManagerRoot(testCase.pendingReference ? "pending_reference" : "auto_anchor");
		fs::path benchmarkRoot = root / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME;

		// The interval grows 1000 timesteps per game past minGames, so it reads back as the game count.
		GGL::SkillTrackerConfig config = {};
		config.enabled = true;
		config.maxArenas = 1;
		config.maxNoScoreGameSeconds = 1.0f;
		config.minGames = 1;
		config.maxGames = 1001;
		config.minTimeSteps = 1'000;
		config.maxTimeSteps = 1'001'000;
		config.referenceGamesPerOpponent = anchorGames;
		config.referenceCalibrationGames = calibrationGames;

		const bool managePolicyVersions = false;
		GGL::PolicyVersionManager manager(
			root / "Recent Policy Versions",
			1,
			1,
			config,
			MakeFixedSeriesEnvConfig(learnerTickSkip),
			managePolicyVersions);
		manager.storage.SetSkillBenchmarkFolder(benchmarkRoot);

		GGL::SkillBenchmark previous = MakeBenchmark(benchmarkRoot, "previous", "20260101-000000", 0.0f, 1000);
		GGL::ModelManager::MakeModels(false, 2, 1, {}, policyConfig, {}, torch::kCPU, previous.models);
		manager.skill.benchmarks.push_back(previous);

		std::vector<GGL::PolicyVersion> references(testCase.pendingReference ? 1 : 0);
		for (GGL::PolicyVersion& reference : references) {
			reference.label = "pending reference";
			GGL::ModelManager::MakeModels(false, 2, 1, {}, policyConfig, {}, torch::kCPU, reference.models);
		}
		manager.skillReferences = &references;

		const uint64_t totalTimesteps = 5000;
		GGL::Report report = {};
		manager.RunSkillMatches(&learner, report, totalTimesteps);

		const uint64_t expectedInterval = config.minTimeSteps + 1000 * (uint64_t)(testCase.expectedGames - config.minGames);
		ASSERT_EQ(manager.skill.nextRunTimestep, totalTimesteps + expectedInterval);
		for (const auto& [key, value] : report.data) {
			ASSERT_NE(key.rfind("SkillTest/Reference/", 0), size_t(0)) << key;
			ASSERT_NE(key.rfind("SkillTest/ReferenceSigma/", 0), size_t(0)) << key;
		}

		for (GGL::PolicyVersion& reference : references)
			reference.models.Free();
		fs::remove_all(root);
	}
}

TEST(PolicyVersionManager, MakeReferenceMetricLabelReplacesSeparators)
{
	const std::string label = GGL::PolicyVersionManager::MakeReferenceMetricLabel("Nexto / V1 / Imported Models / RLBotPack");
	ASSERT_EQ(label, "Nexto-V1-Imported-Models-RLBotPack");
}

TEST(PolicyVersionManager, MakeSkillReferenceOpponentPlaysWithSourceAgentTiming)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	fs::path root = TempPolicyVersionManagerRoot("skill_reference_runtime");
	fs::path registryPath = root / "skill_references.json";
	std::unique_ptr<Agent> nexto = AgentRegistry::Create("Nexto");

	nlohmann::json registry = {
		{ "skill_references", nlohmann::json::array({
			{
				{ "source_agent", "Nexto" },
				{ "source_stage", "Imported Models" },
				{ "source_version_id", "RLBotPack" },
				{ "display_name", "Nexto (RLBotPack)" },
				{ "checkpoint_path", nexto->modelPath.string() }
			}
		}) }
	};
	std::ofstream(registryPath) << registry.dump(2);

	// Nexto plays at 8/7, a whole multiple of this learner's 4-tick step.
	const int learnerTickSkip = 4;
	RLGC::EnvSetConfig envConfig = MakeFixedSeriesEnvConfig(learnerTickSkip);
	envConfig.actionDelay = 3;

	GGL::SkillTrackerConfig skillConfig = {};
	skillConfig.maxArenas = 4;
	skillConfig.maxNoScoreGameSeconds = 1.0f;
	const bool managePolicyVersions = false;
	GGL::PolicyVersionManager manager(root, -1, 0, skillConfig, envConfig, managePolicyVersions);
	manager.staticOpponentsRegistryPath = registryPath;
	manager.registryListKey = "skill_references";

	GGL::PPOLearner learner(2, 1, MakeFixedSeriesPPOConfig(), torch::kCPU);
	manager.LoadStaticOpponents(learner.GetPolicyModels(), CreateRegisteredAgent);
	ASSERT_EQ(manager.versions.size(), size_t(1));

	const int teamSize = 1;
	const GGL::FixedSeriesOpponent opponent = manager.MakeSkillReferenceOpponent(&learner, manager.versions.front(), teamSize);
	ASSERT_EQ(opponent.runtime.tickSkip, 8);
	ASSERT_EQ(opponent.runtime.actionDelay, 7);
	ASSERT_EQ(opponent.runtime.obsSize, 1057);
	ASSERT_EQ(opponent.runtime.numActions, 90);
	ASSERT_TRUE(opponent.runtime.makeRuntimeComponents != nullptr);

	const int gamesPerOpponent = 2;
	const std::vector<GGL::FixedSeriesRecord> records = manager.RunFixedSkillSeries(&learner, { opponent }, teamSize, gamesPerOpponent);

	ASSERT_EQ(records.size(), size_t(1));
	ASSERT_EQ(records[0].wins + records[0].losses + records[0].noScoreGames, gamesPerOpponent);

	manager.ClearVersions();
	fs::remove_all(root);
}
