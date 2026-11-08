#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Settings/ForgeSettings.h"
#include "StageRegistry.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>

namespace fs = std::filesystem;
using namespace RocketForgeTraining;

namespace {

	fs::path TempPath(const std::string& name) {
		return RocketForgeTests::TestTempDir() / name;
	}

	ForgeSettings MakeSampleSettings() {
		ForgeSettings s;
		s.schemaVersion = kForgeSettingsSchemaVersion;
		s.agentName     = "example-agent";
		s.buildName     = "V2";
		s.globals.metricsEnabled         = false;
		s.globals.pruneSourceCheckpoints = true;
		s.globals.persistSegmentSnapshotsDuringTraining = true;

		StageSettings a;
		a.stageName            = "ball-chase";
		a.targetTotalTimesteps = 100'000'000;
		a.branchName           = "main";
		a.warmStartPolicy      = false;
		a.warmStartIterations  = 5;
		a.keepRunId            = true;
		a.allowSegmentSetupMismatch = true;
		a.versionIncrement     = VersionIncrement::Major;
		a.versionNotes         = "first stage";
		a.bootstrap.mode       = BootstrapConfig::Mode::New;
		a.createdSegmentId     = "v1.00";
		s.stages.push_back(a);

		StageSettings b;
		b.stageName            = "tactics";
		b.targetTotalTimesteps = 250'000'000;
		s.stages.push_back(b);

		return s;
	}
}

TEST(ForgeSettings, RoundTripPreservesAllFields) {
	ForgeSettings original = MakeSampleSettings();
	fs::path path = TempPath("roundtrip.json");

	ASSERT_TRUE(GGL::JsonSaveToFile(original, path));
	ForgeSettingsLoadResult loaded = LoadForgeSettings(path);
	ASSERT_TRUE(loaded.ok);

	ASSERT_EQ(loaded.value.agentName, original.agentName);
	ASSERT_EQ(loaded.value.buildName, original.buildName);
	ASSERT_EQ(loaded.value.globals.metricsEnabled, original.globals.metricsEnabled);
	ASSERT_EQ(loaded.value.globals.pruneSourceCheckpoints, original.globals.pruneSourceCheckpoints);
	ASSERT_EQ(
		loaded.value.globals.persistSegmentSnapshotsDuringTraining,
		original.globals.persistSegmentSnapshotsDuringTraining);
	ASSERT_EQ(loaded.value.stages.size(), original.stages.size());
	ASSERT_EQ(loaded.value.stages[0].stageName, std::string("ball-chase"));
	ASSERT_EQ(loaded.value.stages[0].targetTotalTimesteps, uint64_t(100'000'000));
	ASSERT_EQ(loaded.value.stages[0].branchName, std::string("main"));
	ASSERT_EQ(loaded.value.stages[0].warmStartIterations, 5);
	ASSERT_TRUE(loaded.value.stages[0].keepRunId);
	ASSERT_TRUE(loaded.value.stages[0].allowSegmentSetupMismatch);
	ASSERT_TRUE(loaded.value.stages[0].versionIncrement == VersionIncrement::Major);
	ASSERT_EQ(loaded.value.stages[0].versionNotes, std::string("first stage"));
	ASSERT_EQ(loaded.value.stages[0].createdSegmentId, std::string("v1.00"));

	{
		std::ifstream saved(path);
		const std::string savedJson((std::istreambuf_iterator<char>(saved)), std::istreambuf_iterator<char>());
		ASSERT_TRUE(savedJson.find("createdVersionId") == std::string::npos);
	}

	fs::remove(path);
}

TEST(ForgeSettings, RoundTripPreservesBootstrapModes)
{
	struct Case {
		const char* fileName;
		BootstrapConfig::Mode mode;
		BootstrapConfig::BranchMode branchMode;
		const char* branchName;
		const char* selectedExistingVersionId;
	};

	const Case cases[] = {
		{ "continue_latest_branch_mode.json", BootstrapConfig::Mode::ContinueLatest, BootstrapConfig::BranchMode::CreateNew, "branch-test", "v1.02" },
		{ "selected_branch_mode.json", BootstrapConfig::Mode::New, BootstrapConfig::BranchMode::ContinueSelected, "Triple Entropy", "" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.fileName);
		ForgeSettings original;
		original.schemaVersion = kForgeSettingsSchemaVersion;
		original.agentName = "example-agent";
		original.buildName = "V2";
		StageSettings stage;
		stage.stageName = "ball-chase";
		stage.targetTotalTimesteps = 1000;
		stage.branchName = testCase.branchName;
		stage.bootstrap.mode = testCase.mode;
		stage.bootstrap.branchMode = testCase.branchMode;
		stage.bootstrap.selectedExistingVersionId = testCase.selectedExistingVersionId;
		original.stages.push_back(stage);

		fs::path path = TempPath(testCase.fileName);
		ASSERT_TRUE(GGL::JsonSaveToFile(original, path));
		ForgeSettingsLoadResult loaded = LoadForgeSettings(path);
		ASSERT_TRUE(loaded.ok);
		ASSERT_TRUE(loaded.value.stages[0].bootstrap.mode == testCase.mode);
		ASSERT_TRUE(loaded.value.stages[0].bootstrap.branchMode == testCase.branchMode);
		ASSERT_EQ(loaded.value.stages[0].bootstrap.selectedExistingVersionId, std::string(testCase.selectedExistingVersionId));
		ASSERT_EQ(loaded.value.stages[0].branchName, std::string(testCase.branchName));

		fs::remove(path);
	}
}

TEST(ForgeSettings, LoadMigratesLegacyCreatedVersionId)
{
	fs::path path = TempPath("legacy_created_version_id.json");
	std::ofstream out(path);
	out << R"({
		"schemaVersion": 1,
		"agentName": "example-agent",
		"buildName": "V2",
		"globals": {},
		"stages": [
			{
				"stageName": "ball-chase",
				"targetTotalTimesteps": 1000,
				"bootstrap": { "mode": "new" },
				"createdVersionId": "v1.00"
			}
		]
	})";
	out.close();

	ForgeSettingsLoadResult loaded = LoadForgeSettings(path);
	ASSERT_TRUE(loaded.ok);
	ASSERT_EQ(loaded.value.stages[0].createdSegmentId, std::string("v1.00"));

	fs::remove(path);
}

TEST(ForgeSettings, OldEngineVersionKeyStillLoads)
{
	fs::path path = TempPath("old_engine_version.json");
	std::ofstream out(path);
	out << R"({
		"schemaVersion": 1,
		"agentName": "example-agent",
		"buildName": "V2",
		"globals": { "engineVersion": 3, "metricsEnabled": false },
		"stages": [ { "stageName": "ball-chase", "targetTotalTimesteps": 1000 } ]
	})";
	out.close();

	ForgeSettingsLoadResult loaded = LoadForgeSettings(path);
	EXPECT_TRUE(loaded.errors.empty());
	ASSERT_TRUE(loaded.ok);
	EXPECT_FALSE(loaded.value.globals.metricsEnabled);
	ASSERT_EQ(loaded.value.stages.size(), 1u);

	nlohmann::json saved;
	GGL::JsonSave(loaded.value, saved);
	EXPECT_FALSE(saved.at("globals").contains("engineVersion"));

	fs::remove(path);
}

TEST(ModelBenchmarkSettings, RoundTripPreservesAllFields) {
	ModelBenchmarkSettings original;
	original.stageFilter = "ball-chase";
	original.hideImportedModels = true;
	original.libraryPath = fs::path("Model Benchmarks") / "shared.sqlite";
	original.numArenas = 128;
	original.maxGamesPerMode = 250;
	original.fillMissingGames = false;
	original.run2v2 = false;
	original.betterConfidence = 0.90f;
	original.stopWhenRankingKnown = true;
	original.eloRangeConfidence = 0.99f;
	original.maxEloUncertaintyRange = 25.0f;
	original.runUntilMaxGames = true;
	original.collectOptionalStats = false;
	original.seed = uint64_t{ 17375808098319191535ULL };
	ModelBenchmarkSelectedModelSettings selected;
	selected.agentName = "example-agent";
	selected.buildName = "V2";
	selected.stageName = "ball-chase";
	selected.sourceLabel = "RocketForge";
	selected.versionId = "v1.0";
	selected.checkpointPath = fs::path("Models") / "V2" / "segments" / "v1.00_ball_chase" / "checkpoints" / "123456";
	selected.trainingSetupPath = fs::path("Models") / "V2" / "segments" / "v1.00_ball_chase" / "training_setup.json";
	selected.agentSetupSource = ModelBenchmarkAgentSetupSource::CurrentAgent;
	selected.primary = false;
	selected.colorIndex = 4;
	selected.latestTimesteps = 123456;
	selected.notes = "benchmark model";
	selected.skillRating = 42.5;
	selected.hasSkillRating = true;
	original.selectedModels.push_back(selected);

	fs::path path = TempPath("model_benchmark_settings.json");
	ASSERT_TRUE(GGL::JsonSaveToFile(original, path));
	ModelBenchmarkSettingsLoadResult loaded = LoadModelBenchmarkSettings(path);
	ASSERT_TRUE(loaded.ok);

	ASSERT_EQ(loaded.value.stageFilter, std::string("ball-chase"));
	ASSERT_TRUE(loaded.value.hideImportedModels);
	ASSERT_EQ(loaded.value.libraryPath.string(), (fs::path("Model Benchmarks") / "shared.sqlite").string());
	ASSERT_EQ(loaded.value.numArenas, 128);
	ASSERT_EQ(loaded.value.maxGamesPerMode, 250);
	ASSERT_FALSE(loaded.value.fillMissingGames);
	ASSERT_FALSE(loaded.value.run2v2);
	ASSERT_EQ(loaded.value.betterConfidence, 0.90f);
	ASSERT_TRUE(loaded.value.stopWhenRankingKnown);
	ASSERT_EQ(loaded.value.eloRangeConfidence, 0.99f);
	ASSERT_EQ(loaded.value.maxEloUncertaintyRange, 25.0f);
	ASSERT_TRUE(loaded.value.runUntilMaxGames);
	ASSERT_FALSE(loaded.value.collectOptionalStats);
	ASSERT_EQ(loaded.value.seed, uint64_t{ 17375808098319191535ULL });
	ASSERT_EQ(loaded.value.selectedModels.size(), size_t(1));
	ASSERT_EQ(loaded.value.selectedModels[0].agentName, std::string("example-agent"));
	ASSERT_EQ(loaded.value.selectedModels[0].buildName, std::string("V2"));
	ASSERT_EQ(loaded.value.selectedModels[0].stageName, std::string("ball-chase"));
	ASSERT_EQ(loaded.value.selectedModels[0].sourceLabel, std::string("RocketForge"));
	ASSERT_EQ(loaded.value.selectedModels[0].versionId, std::string("v1.0"));
	ASSERT_EQ(
		loaded.value.selectedModels[0].checkpointPath.string(),
		(fs::path("Models") / "V2" / "segments" / "v1.00_ball_chase" / "checkpoints" / "123456").string());
	ASSERT_EQ(
		loaded.value.selectedModels[0].trainingSetupPath.string(),
		(fs::path("Models") / "V2" / "segments" / "v1.00_ball_chase" / "training_setup.json").string());
	ASSERT_TRUE(loaded.value.selectedModels[0].agentSetupSource == ModelBenchmarkAgentSetupSource::CurrentAgent);
	ASSERT_FALSE(loaded.value.selectedModels[0].primary);
	ASSERT_EQ(loaded.value.selectedModels[0].colorIndex, 4);
	ASSERT_EQ(loaded.value.selectedModels[0].latestTimesteps, uint64_t(123456));
	ASSERT_EQ(loaded.value.selectedModels[0].notes, std::string("benchmark model"));
	ASSERT_EQ(loaded.value.selectedModels[0].skillRating, 42.5);
	ASSERT_TRUE(loaded.value.selectedModels[0].hasSkillRating);

	fs::remove(path);
}

TEST(ModelBenchmarkSettings, LoadMissingPrimaryDefaultsTrue) {
	fs::path path = TempPath("model_benchmark_settings_legacy_primary.json");
	std::ofstream out(path);
	out << R"({
		"selectedModels": [
			{
				"agentName": "example-agent",
				"versionId": "v1.0",
				"checkpointPath": "Models/V1/checkpoints/123"
			}
		]
	})";
	out.close();

	ModelBenchmarkSettingsLoadResult loaded = LoadModelBenchmarkSettings(path);
	ASSERT_TRUE(loaded.ok);
	ASSERT_EQ(loaded.value.selectedModels.size(), size_t(1));
	ASSERT_TRUE(loaded.value.selectedModels[0].primary);
	ASSERT_TRUE(loaded.value.fillMissingGames);
	ASSERT_EQ(loaded.value.maxGamesPerMode, 500);
	ASSERT_EQ(loaded.value.selectedModels[0].colorIndex, -1);

	fs::remove(path);
}

TEST(ForgeSettings, LoadMissingFileReportsError) {
	fs::path nonexistent = TempPath("does-not-exist.json");
	fs::remove(nonexistent);
	ForgeSettingsLoadResult r = LoadForgeSettings(nonexistent);
	ASSERT_FALSE(r.ok);
	ASSERT_EQ(r.errors.empty(), false);
}

TEST(ForgeSettings, LoadWrongSchemaVersionReportsError) {
	fs::path path = TempPath("wrong_schema.json");
	std::ofstream out(path);
	out << R"({ "schemaVersion": 999, "agentName": "x", "globals": {}, "stages": [] })";
	out.close();
	ForgeSettingsLoadResult r = LoadForgeSettings(path);
	ASSERT_FALSE(r.ok);
	fs::remove(path);
}

TEST(ForgeSettings, ValidationRejectsEmptyStages) {
	ForgeSettings s;
	s.schemaVersion = kForgeSettingsSchemaVersion;
	s.agentName     = "example-agent";
	auto errors = ValidateForgeSettings(s);
	ASSERT_FALSE(errors.empty());
}

TEST(ForgeSettings, ValidationRejectsNonIncreasingTargets) {
	ForgeSettings s;
	s.schemaVersion = kForgeSettingsSchemaVersion;
	s.agentName     = "example-agent";
	StageSettings a; a.stageName = "example-ball-chase"; a.targetTotalTimesteps = 100;
	StageSettings b; b.stageName = "tactics";            b.targetTotalTimesteps = 50;
	s.stages = { a, b };
	auto errors = ValidateForgeSettings(s);
	ASSERT_FALSE(errors.empty());
}

TEST(ForgeSettings, ValidationAllowsDuplicateStageNames) {
	RocketForge::StageRegistry::Register(
		"test-duplicate-stage",
		nullptr,
		RocketForge::StageCompatInfo{});
	AgentRegistry::Register(
		"test-settings-agent",
		[]() { return std::make_unique<Agent>(); });

	ForgeSettings s;
	s.schemaVersion = kForgeSettingsSchemaVersion;
	s.agentName     = "test-settings-agent";
	StageSettings a; a.stageName = "test-duplicate-stage"; a.targetTotalTimesteps = 100;
	StageSettings b; b.stageName = "test-duplicate-stage"; b.targetTotalTimesteps = 200;
	s.stages = { a, b };
	auto errors = ValidateForgeSettings(s);
	ASSERT_TRUE(errors.empty());
}

TEST(ForgeSettings, SerializeRoundTripPreservesAllFields) {
	using namespace RocketForgeTraining;

	ForgeSettings settings;
	settings.agentName = "example-agent";
	settings.buildName = "V2";
	settings.globals.metricsEnabled = false;
	settings.globals.debug.gameSpeed = 2.f;

	StageSettings stage;
	stage.stageName = "example-ball-chase";
	stage.targetTotalTimesteps = 1000000;
	stage.branchName = "experiment";
	stage.versionIncrement = VersionIncrement::Major;
	stage.versionOverride = GGL::SegmentVersionNumber{ 3, 1 };
	stage.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
	stage.bootstrap.sourceStage = "other-stage";
	stage.parameterOverrides.push_back({ StageParameterOverrideKind::RewardWeight, "Rewards", "TouchBall", "2.5" });
	stage.createdSegmentId = "v3.1";
	settings.stages.push_back(stage);

	nlohmann::json j;
	GGL::JsonSave(settings, j);

	// Spot-check exact on-disk key names (backward compatibility).
	ASSERT_EQ(j["stages"][0]["stageName"].get<std::string>(), "example-ball-chase");
	ASSERT_EQ(j["stages"][0]["versionIncrement"].get<std::string>(), "major");
	ASSERT_EQ(j["stages"][0]["bootstrap"]["mode"].get<std::string>(), "loadFromOtherStage");
	ASSERT_EQ(j["stages"][0]["versionOverride"]["major"].get<int>(), 3);
	ASSERT_EQ(j["stages"][0]["parameterOverrides"][0]["kind"].get<std::string>(), "rewardWeight");
	ASSERT_TRUE(j["stages"][0]["bootstrap"]["sourceVersionId"].is_null());   // empty string -> null

	ForgeSettings loaded;
	std::vector<std::string> errors = GGL::JsonLoad(j, loaded);
	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(loaded.stages[0].branchName, "experiment");
	ASSERT_TRUE(loaded.stages[0].versionOverride.has_value());
	ASSERT_EQ(loaded.stages[0].versionOverride->minor, 1);
	ASSERT_EQ(loaded.stages[0].parameterOverrides.size(), size_t(1));
}

TEST(ForgeSettings, LegacyCreatedVersionIdWinsWhenPrimaryIsEmptyString) {
	using namespace RocketForgeTraining;

	// The old loader fell back to the legacy key whenever the resulting VALUE was empty,
	// not only when the primary key was absent — an explicit empty "createdSegmentId" must
	// still defer to "createdVersionId".
	nlohmann::json stageJson = { { "stageName", "s" }, { "createdSegmentId", "" }, { "createdVersionId", "v1.2" } };

	StageSettings stage;
	std::vector<std::string> errors = GGL::JsonLoad(stageJson, stage);
	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(stage.createdSegmentId, "v1.2");
}

TEST(ForgeSettings, PerformanceCoresOnlyRoundTripsAndDefaultsOff) {
	using namespace RocketForgeTraining;

	GlobalSettings defaults;
	ASSERT_FALSE(defaults.performanceCoresOnly);

	GlobalSettings enabled;
	enabled.performanceCoresOnly = true;

	nlohmann::json saved;
	GGL::JsonSave(enabled, saved);
	ASSERT_TRUE(saved["performanceCoresOnly"].get<bool>());

	GlobalSettings reloaded;
	ASSERT_TRUE(GGL::JsonLoad(saved, reloaded).empty());
	ASSERT_TRUE(reloaded.performanceCoresOnly);

	nlohmann::json withoutKey = { { "metricsEnabled", false } };
	GlobalSettings loadedWithoutKey;
	GGL::JsonLoad(withoutKey, loadedWithoutKey);
	ASSERT_FALSE(loadedWithoutKey.performanceCoresOnly);
}

TEST(ModelBenchmarkSettings, KickoffSamplingRoundTripsAndDefaultsForLegacyFiles)
{
	ModelBenchmarkSettings settings;
	settings.stochasticKickoff = true;
	settings.kickoffTemperature = 0.65f;
	const auto loaded = DeserializeModelBenchmarkSettings(SerializeModelBenchmarkSettings(settings));
	ASSERT_TRUE(loaded.ok);
	EXPECT_TRUE(loaded.value.stochasticKickoff);
	EXPECT_FLOAT_EQ(loaded.value.kickoffTemperature, 0.65f);

	const auto legacy = DeserializeModelBenchmarkSettings("{}");
	ASSERT_TRUE(legacy.ok);
	EXPECT_FALSE(legacy.value.stochasticKickoff);
	EXPECT_FLOAT_EQ(legacy.value.kickoffTemperature, 0.70f);
}
