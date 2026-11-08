#include <gtest/gtest.h>

#include "Cli/CliErrors.h"
#include "Commands/Commands.h"
#include "Commands/TrainCommand.h"
#include "Storage/BuildStorage.h"
#include "TestSupport.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace {

	const std::filesystem::path SETTINGS_PATH = "Bots/Test/Agent/forge_settings.json";

	CliArgs ParseTrainArgs(const std::vector<std::string>& arguments)
	{
		return CliArgs::Parse(arguments, TrainingCommands().front().options);
	}

	ForgeSettings MakeLoadedSettings()
	{
		ForgeSettings loaded;
		loaded.agentName = "test-agent";

		StageSettings queued;
		queued.stageName = "queued-stage";
		queued.targetTotalTimesteps = 1000;
		loaded.stages = { queued, queued };

		return loaded;
	}

}

TEST(TrainCommand, QueueRunKeepsOverridesOutOfPersistedGlobals)
{
	ForgeSettings loaded = MakeLoadedSettings();
	loaded.globals.metricsEnabled = true;
	loaded.globals.performanceCoresOnly = false;

	const TrainRequest request = BuildTrainRequest(ParseTrainArgs({ "--no-metrics", "--performance-cores" }), loaded, SETTINGS_PATH);

	EXPECT_FALSE(request.settings.globals.metricsEnabled);
	EXPECT_TRUE(request.settings.globals.performanceCoresOnly);
	EXPECT_EQ(request.settings.stages.size(), loaded.stages.size());
	EXPECT_EQ(request.persistence.settingsPath, SETTINGS_PATH);
	ASSERT_TRUE(request.persistence.persistedGlobals.has_value());
	EXPECT_TRUE(request.persistence.persistedGlobals->metricsEnabled);
	EXPECT_FALSE(request.persistence.persistedGlobals->performanceCoresOnly);
	EXPECT_FALSE(request.preview);
}

TEST(TrainCommand, AdHocStageNeverPersists)
{
	struct Case {
		std::string label;
		std::vector<std::string> arguments;
		AdHocSegments segments;
		BootstrapConfig::Mode expectedMode;
		std::string expectedSelectedId;
		std::string expectedSourceStage;
		std::string expectedSourceId;
		bool expectedKeepRunId;
	};

	const std::vector<Case> cases = {
		{ "continues the newest segment", { "--stage", "s" }, { "v1.7", "" }, BootstrapConfig::Mode::ContinueExisting, "v1.7", "", "", false },
		{ "no segment starts new", { "--stage", "s" }, { "", "" }, BootstrapConfig::Mode::New, "", "", "", false },
		{ "new ignores the segment", { "--stage", "s", "--new" }, { "v1.7", "" }, BootstrapConfig::Mode::New, "", "", "", false },
		{ "from another stage", { "--stage", "s", "--from-stage", "t", "--keep-run-id" }, { "v1.7", "v2.3" },
			BootstrapConfig::Mode::LoadFromOtherStage, "", "t", "v2.3", true },
	};

	const ForgeSettings loaded = MakeLoadedSettings();

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		const TrainRequest request = BuildTrainRequest(ParseTrainArgs(testCase.arguments), loaded, SETTINGS_PATH, testCase.segments);

		ASSERT_EQ(request.settings.stages.size(), 1u);
		const StageSettings& stage = request.settings.stages[0];
		EXPECT_EQ(stage.stageName, "s");
		EXPECT_EQ(stage.targetTotalTimesteps, UINT64_MAX);
		EXPECT_EQ(stage.bootstrap.mode, testCase.expectedMode);
		EXPECT_EQ(stage.bootstrap.selectedExistingVersionId, testCase.expectedSelectedId);
		EXPECT_EQ(stage.bootstrap.sourceStage, testCase.expectedSourceStage);
		EXPECT_EQ(stage.bootstrap.sourceVersionId, testCase.expectedSourceId);
		EXPECT_EQ(stage.keepRunId, testCase.expectedKeepRunId);
		EXPECT_TRUE(request.persistence.settingsPath.empty());
		EXPECT_FALSE(request.persistence.persistedGlobals.has_value());
		EXPECT_EQ(loaded.stages.size(), 2u);
		EXPECT_EQ(loaded.stages[0].stageName, "queued-stage");
	}
}

TEST(TrainCommand, RejectsBootstrapFlagsWithoutStage)
{
	const std::vector<std::vector<std::string>> cases = {
		{ "--new" },
		{ "--from-stage", "t" },
		{ "--keep-run-id" },
		{ "--stage", "s", "--new", "--from-stage", "t" },
		{ "--stage", "s", "--keep-run-id" },
	};

	const ForgeSettings loaded = MakeLoadedSettings();
	const AdHocSegments segments = { "v1.1", "v1.2" };

	for (const std::vector<std::string>& arguments : cases) {
		SCOPED_TRACE(arguments.back());

		EXPECT_THROW(BuildTrainRequest(ParseTrainArgs(arguments), loaded, SETTINGS_PATH, segments), CliUsageError);
	}
}

TEST(TrainCommand, MissingNamedSettingsFileIsAUsageError)
{
	const std::filesystem::path missing = RocketForgeTests::TestTempDir("train_missing_file") / "forge_settngs.json";

	const std::vector<std::vector<std::string>> cases = {
		{ "--file", missing.string() },
		{ "--file", missing.string(), "--preview" },
		{ "--file", missing.string(), "--stage", "example-ball-chase" },
	};

	for (const std::vector<std::string>& arguments : cases) {
		SCOPED_TRACE(arguments.back());

		try {
			std::vector<std::string> withAgent = arguments;
			withAgent.insert(withAgent.end(), { "--agent", "example-agent" });
			RunTrainCommand(ParseTrainArgs(withAgent));
			ADD_FAILURE() << "expected CliUsageError";
		} catch (const CliUsageError& error) {
			EXPECT_NE(std::string(error.what()).find("settings file not found"), std::string::npos) << error.what();
		}

		EXPECT_FALSE(std::filesystem::exists(missing));
	}
}

TEST(TrainCommand, FromStageWithoutSourceSegmentNamesTheListCommand)
{
	const ForgeSettings loaded = MakeLoadedSettings();

	try {
		BuildTrainRequest(ParseTrainArgs({ "--stage", "s", "--from-stage", "t" }), loaded, SETTINGS_PATH, { "v1.1", "" });
		FAIL() << "expected CliUsageError";
	} catch (const CliUsageError& error) {
		EXPECT_NE(std::string(error.what()).find("list segments"), std::string::npos);
	}
}

TEST(TrainCommand, AdHocRequestsPlanToTheirSegments)
{
	struct Case {
		std::string label;
		std::vector<std::string> arguments;
		AdHocSegments segments;
		ActionKind expectedKind;
		std::string expectedSegmentId;
		std::string expectedSourceStage;
		std::string expectedSourceSegmentId;
	};

	const std::vector<Case> cases = {
		{ "continues the stage's segment", { "--stage", "s" }, { "v1.7", "" }, ActionKind::Continue, "v1.7", "", "" },
		{ "starts new without a segment", { "--stage", "s" }, { "", "" }, ActionKind::StartNew, "", "", "" },
		{ "bootstraps a new segment from the source", { "--stage", "s", "--from-stage", "t" }, { "v1.7", "v2.3" }, ActionKind::StartNew, "", "t", "v2.3" },
	};

	const SegmentInspector inspect = [](const std::string&, const std::string&, const std::string& segmentId) {
		CheckpointInfo info;
		info.segmentExists = true;
		info.hasCheckpoints = true;
		info.latestTimesteps = 5'000'000;
		info.stageName = segmentId == "v2.3" ? "t" : "s";
		return info;
	};

	ForgeSettings loaded = MakeLoadedSettings();
	loaded.buildName = "V1";

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		const TrainRequest request = BuildTrainRequest(ParseTrainArgs(testCase.arguments), loaded, SETTINGS_PATH, testCase.segments);
		BuildStorage storage(RocketForgeTests::TestTempDir("train_plan"), "V1");

		const std::vector<QueuePlanEntry> plan = PlanQueue(request.settings, storage, inspect);

		ASSERT_EQ(plan.size(), 1u);
		EXPECT_EQ(plan[0].kind, testCase.expectedKind);
		EXPECT_EQ(plan[0].sourceStage, testCase.expectedSourceStage);
		EXPECT_EQ(plan[0].sourceSegmentId, testCase.expectedSourceSegmentId);

		if (!testCase.expectedSegmentId.empty())
			EXPECT_EQ(plan[0].segmentId, testCase.expectedSegmentId);
	}
}

TEST(TrainCommand, PruneSourceAndPreviewFlowIntoTheRequest)
{
	const TrainRequest request =
		BuildTrainRequest(ParseTrainArgs({ "--prune-source", "--preview" }), MakeLoadedSettings(), SETTINGS_PATH);

	EXPECT_TRUE(request.settings.globals.pruneSourceCheckpoints);
	EXPECT_TRUE(request.preview);
}

TEST(TrainCommand, FormatsEachPlanKind)
{
	QueuePlanEntry continued;
	continued.stageIndex = 0;
	continued.stageName = "ball-chase";
	continued.kind = ActionKind::Continue;
	continued.segmentId = "v1.3";
	continued.currentTimesteps = 12'000'000;
	continued.targetTotalTimesteps = 50'000'000;

	QueuePlanEntry unlimited = continued;
	unlimited.stageIndex = 4;
	unlimited.stageName = "one-off";
	unlimited.targetTotalTimesteps = UINT64_MAX;

	QueuePlanEntry fresh;
	fresh.stageIndex = 1;
	fresh.stageName = "kickoff";
	fresh.kind = ActionKind::StartNew;
	fresh.segmentId = "v1.4";

	QueuePlanEntry bootstrapped = fresh;
	bootstrapped.stageIndex = 2;
	bootstrapped.stageName = "aerials";
	bootstrapped.sourceStage = "kickoff";
	bootstrapped.sourceSegmentId = "v1.3";

	QueuePlanEntry skipped;
	skipped.stageIndex = 3;
	skipped.stageName = "done-stage";
	skipped.kind = ActionKind::Skip;
	skipped.skipReason = "already reached its target";

	EXPECT_EQ(
		FormatQueuePlan({ continued, fresh, bootstrapped, skipped, unlimited }),
		"1. ball-chase  continue v1.3 (12.0M / 50.0M)\n"
		"2. kickoff  new v1.4\n"
		"3. aerials  new v1.4 from kickoff v1.3\n"
		"4. done-stage  skip: already reached its target\n"
		"5. one-off  continue v1.3 (12.0M / no limit)\n");
}
