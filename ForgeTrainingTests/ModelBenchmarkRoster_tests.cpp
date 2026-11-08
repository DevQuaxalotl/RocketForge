#include <gtest/gtest.h>

#include "ModelBenchmark/ModelBenchmarkExport.h"
#include "ModelBenchmark/ModelBenchmarkRoster.h"
#include "Storage/BuildStorage.h"
#include "TestSupport.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const ROSTER_AGENT = "model-benchmark-roster-test-agent";
	const char* const ROSTER_BUILD = "V1";
	const char* const ROSTER_STAGE = "example-ball-chase";

	void WriteFile(const fs::path& path, const std::string& text)
	{
		fs::create_directories(path.parent_path());
		std::ofstream(path) << text;
	}

	/** Registers the test agent over a fresh temp Models folder holding build V1 with segments v1.0 and v1.1, each with a checkpoint at 400. */
	fs::path MakeRosterModels(const std::string& label)
	{
		const fs::path modelsRoot = RocketForgeTests::TestTempDir(label) / "Agent" / "Models";
		fs::create_directories(modelsRoot);

		AgentRegistry::Register(ROSTER_AGENT, [modelsRoot] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = ROSTER_AGENT;
			agent->modelPath = modelsRoot;
			return agent;
		});

		BuildStorage storage(modelsRoot, ROSTER_BUILD);
		storage.EnsureBuildExists();

		for (const char* segmentId : { "v1.0", "v1.1" }) {
			TrainingSegmentMetadata segment;
			segment.segmentId = segmentId;
			segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, ROSTER_STAGE);
			segment.buildName = ROSTER_BUILD;
			segment.stageName = ROSTER_STAGE;

			WriteFile(storage.SegmentCheckpointsRoot(segment.folderName) / "400" / "POLICY.lt", "policy weights");
			WriteFile(storage.SegmentRoot(segment.folderName) / "training_setup.json", "{}");
			storage.SaveSegment(segment);
		}

		return modelsRoot;
	}

	ModelBenchmarkSelectedModelSettings SegmentEntry(const std::string& versionId)
	{
		ModelBenchmarkSelectedModelSettings entry;
		entry.agentName = ROSTER_AGENT;
		entry.buildName = ROSTER_BUILD;
		entry.versionId = versionId;
		return entry;
	}

}

TEST(ModelBenchmarkRoster, ResolvesPathlessSegmentEntry)
{
	MakeRosterModels("roster-pathless");

	ModelBenchmarkSettings settings;
	settings.stochasticKickoff = true;
	settings.kickoffTemperature = 0.65f;
	settings.selectedModels = { SegmentEntry("v1.0"), SegmentEntry("v1.1") };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_TRUE(roster.errors.empty()) << roster.errors.front();
	ASSERT_EQ(roster.config.models.size(), 2u);
	ASSERT_EQ(roster.exportModels.size(), 2u);

	EXPECT_TRUE(roster.config.stochasticKickoff);
	EXPECT_FLOAT_EQ(roster.config.kickoffTemperature, 0.65f);
	EXPECT_EQ(roster.config.models[0].checkpointPath.filename().string(), "400");
	EXPECT_EQ(roster.config.models[0].trainingSetupPath.filename().string(), "training_setup.json");
	EXPECT_EQ(roster.config.models[0].stageName, ROSTER_STAGE);
	EXPECT_FALSE(roster.exportModels[0].fingerprint.empty());
	EXPECT_EQ(roster.exportModels[0].latestTimesteps, 400u);
	EXPECT_EQ(roster.exportModels[0].buildName, ROSTER_BUILD);
}

TEST(ModelBenchmarkRoster, SkipsExcludedAndResultsOnlyEntries)
{
	MakeRosterModels("roster-skips");

	ModelBenchmarkSelectedModelSettings excluded = SegmentEntry("v1.0");
	excluded.included = false;

	ModelBenchmarkSelectedModelSettings resultsOnly = SegmentEntry("v1.0");
	resultsOnly.resultsOnly = true;

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0"), SegmentEntry("v1.1"), excluded, resultsOnly };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	EXPECT_TRUE(roster.errors.empty());
	EXPECT_EQ(roster.config.models.size(), 2u);
}

TEST(ModelBenchmarkRoster, WarnsAboutEachLibraryOnlyOption)
{
	MakeRosterModels("roster-warnings");

	struct Case {
		const char* field;
		void (*change)(ModelBenchmarkSettings&);
	};
	const Case cases[] = {
		{ "fillMissingGames", [](ModelBenchmarkSettings& settings) { settings.fillMissingGames = false; } },
		{ "matchSavedConditions", [](ModelBenchmarkSettings& settings) { settings.matchSavedConditions = false; } },
		{ "libraryPath", [](ModelBenchmarkSettings& settings) { settings.libraryPath = "x.db"; } },
		{ "reuseSelectedEngine", [](ModelBenchmarkSettings& settings) { settings.reuseSelectedEngine = false; } },
		{ "reuseEngineVersions", [](ModelBenchmarkSettings& settings) { settings.reuseEngineVersions = { 3 }; } },
		{ "reuseEnvironmentKeys", [](ModelBenchmarkSettings& settings) { settings.reuseEnvironmentKeys[2] = { "key" }; } },
		{ "ratingTargetsOnly", [](ModelBenchmarkSettings& settings) { settings.ratingTargetsOnly = true; } },
	};

	ModelBenchmarkSettings defaults;
	defaults.selectedModels = { SegmentEntry("v1.0"), SegmentEntry("v1.1") };
	EXPECT_TRUE(ResolveModelBenchmarkRoster(defaults, false).warnings.empty());

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.field);

		ModelBenchmarkSettings settings = defaults;
		testCase.change(settings);

		const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

		ASSERT_EQ(roster.warnings.size(), 1u);
		EXPECT_EQ(roster.warnings[0], std::string(testCase.field) + " needs the GUI's results library and is ignored by the console benchmark");
	}
}

TEST(ModelBenchmarkRoster, GuiDefaultLibraryPathDoesNotWarn)
{
	MakeRosterModels("roster-default-library");

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0"), SegmentEntry("v1.1") };
	settings.libraryPath = DefaultModelBenchmarkLibraryPath();

	EXPECT_TRUE(ResolveModelBenchmarkRoster(settings, false).warnings.empty());
}

TEST(ModelBenchmarkRoster, UnresolvableEntryNamesTheListCommand)
{
	MakeRosterModels("roster-unresolvable");

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0"), SegmentEntry("v9.9") };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_FALSE(roster.errors.empty());
	EXPECT_NE(roster.errors[0].find("v9.9"), std::string::npos) << roster.errors[0];
	EXPECT_NE(roster.errors[0].find("list segments"), std::string::npos) << roster.errors[0];
}

TEST(ModelBenchmarkRoster, NeedsAtLeastTwoRunnableModels)
{
	MakeRosterModels("roster-too-few");

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0") };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_EQ(roster.errors.size(), 1u);
	EXPECT_NE(roster.errors[0].find("at least 2"), std::string::npos) << roster.errors[0];
}

TEST(ModelBenchmarkRoster, UnknownImportedModelNamesTheListCommand)
{
	MakeRosterModels("roster-imported");

	ModelBenchmarkSelectedModelSettings imported = SegmentEntry("missing-model");
	imported.sourceLabel = "Imported Model";

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0"), imported };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_FALSE(roster.errors.empty());
	EXPECT_NE(roster.errors[0].find("list imported-models"), std::string::npos) << roster.errors[0];
}

TEST(ModelBenchmarkRoster, AllPrimaryOverridesEntryRoles)
{
	MakeRosterModels("roster-all-primary");

	ModelBenchmarkSelectedModelSettings secondary = SegmentEntry("v1.1");
	secondary.primary = false;

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0"), secondary };

	EXPECT_FALSE(ResolveModelBenchmarkRoster(settings, false).config.models[1].primary);
	EXPECT_TRUE(ResolveModelBenchmarkRoster(settings, true).config.models[1].primary);
}

TEST(ModelBenchmarkRoster, ResolvedModelsRunOnTheCurrentAgentWithoutASavedSetup)
{
	const fs::path modelsRoot = MakeRosterModels("roster-current-agent");
	WriteFile(modelsRoot / "Imported Models" / "imp1" / "POLICY.lt", "imported weights");

	ModelBenchmarkSelectedModelSettings imported = SegmentEntry("imp1");
	imported.sourceLabel = "Imported Model";

	ModelBenchmarkSelectedModelSettings importedByBuild = SegmentEntry("imp1");
	importedByBuild.buildName = "Imported Model";

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0"), imported };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_TRUE(roster.errors.empty()) << roster.errors.front();
	ASSERT_EQ(roster.config.models.size(), 2u);

	// training_setup.json holds no agent setup, so the saved-setup path would make the engine throw.
	EXPECT_EQ(roster.config.models[0].agentSetupSource, ModelBenchmarkAgentSetupSource::CurrentAgent);

	const ModelBenchmarkModel& importedModel = roster.config.models[1];
	EXPECT_EQ(importedModel.checkpointPath, (modelsRoot / "Imported Models" / "imp1"));
	EXPECT_EQ(importedModel.agentSetupSource, ModelBenchmarkAgentSetupSource::CurrentAgent);
	EXPECT_EQ(importedModel.displayName, std::string(ROSTER_AGENT) + " / Imported Model / imp1");
	EXPECT_EQ(roster.exportModels[1].buildName, "Imported Model");
	EXPECT_FALSE(roster.exportModels[1].fingerprint.empty());

	settings.selectedModels = { SegmentEntry("v1.0"), importedByBuild };
	EXPECT_TRUE(ResolveModelBenchmarkRoster(settings, false).errors.empty());
}

TEST(ModelBenchmarkRoster, StalePathReResolvesAndRefreshesTheFingerprint)
{
	const fs::path modelsRoot = MakeRosterModels("roster-stale-path");

	ModelBenchmarkSelectedModelSettings stale = SegmentEntry("v1.0");
	stale.checkpointPath = modelsRoot / "gone" / "checkpoints" / "100";
	stale.fingerprint = "saved-for-the-old-checkpoint";

	ModelBenchmarkSettings settings;
	settings.selectedModels = { stale, SegmentEntry("v1.1") };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_TRUE(roster.errors.empty()) << roster.errors.front();
	EXPECT_EQ(roster.config.models[0].checkpointPath.filename().string(), "400");
	EXPECT_NE(roster.exportModels[0].fingerprint, stale.fingerprint);
	EXPECT_FALSE(roster.exportModels[0].fingerprint.empty());

	ASSERT_EQ(roster.warnings.size(), 1u);
	EXPECT_NE(roster.warnings[0].find("not found"), std::string::npos) << roster.warnings[0];
}

TEST(ModelBenchmarkRoster, ExistingPathKeepsItsSavedFingerprint)
{
	const fs::path modelsRoot = MakeRosterModels("roster-saved-fingerprint");

	ModelBenchmarkSelectedModelSettings entry = SegmentEntry("v1.0");
	entry.checkpointPath = modelsRoot / "Imported Models" / "elsewhere";
	fs::create_directories(entry.checkpointPath);
	entry.fingerprint = "saved";

	ModelBenchmarkSettings settings;
	settings.selectedModels = { entry, SegmentEntry("v1.1") };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_TRUE(roster.errors.empty()) << roster.errors.front();
	EXPECT_EQ(roster.exportModels[0].fingerprint, "saved");
	EXPECT_TRUE(roster.warnings.empty());
}

TEST(ModelBenchmarkRoster, SameModelTwiceIsAnError)
{
	MakeRosterModels("roster-duplicate");

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0"), SegmentEntry("v1.0"), SegmentEntry("v1.1") };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_EQ(roster.errors.size(), 1u);
	EXPECT_NE(roster.errors[0].find("twice"), std::string::npos) << roster.errors[0];
	EXPECT_NE(roster.errors[0].find(std::string(ROSTER_AGENT) + " / V1 / v1.0"), std::string::npos) << roster.errors[0];
}

TEST(ModelBenchmarkRoster, StaleExternalPathHasItsOwnError)
{
	MakeRosterModels("roster-external");

	ModelBenchmarkSelectedModelSettings external;
	external.agentName = ROSTER_AGENT;
	external.sourceLabel = "External";
	external.checkpointPath = "Z:/no/such/model";

	ModelBenchmarkSettings settings;
	settings.selectedModels = { SegmentEntry("v1.0"), external };

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, false);

	ASSERT_FALSE(roster.errors.empty());
	EXPECT_NE(roster.errors[0].find("external model folder not found"), std::string::npos) << roster.errors[0];
	EXPECT_EQ(roster.errors[0].find("list segments"), std::string::npos) << roster.errors[0];
}
