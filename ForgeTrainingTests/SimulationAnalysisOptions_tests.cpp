#include <gtest/gtest.h>

#include "Cli/CliErrors.h"
#include "Commands/Commands.h"
#include "SimulationAnalysis/Options.h"
#include "Storage/BuildStorage.h"
#include "TestSupport.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace RocketForge::Cli;
using namespace RocketForge::SimulationAnalysis;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const ANALYSIS_AGENT = "analysis-options-test-agent";
	const char* const ANALYSIS_BUILD = "V1";
	const char* const ANALYSIS_SEGMENT = "v1.0";
	const char* const ANALYSIS_STAGE = "example-ball-chase";

	CliArgs ParseActionTraceArgs(const std::vector<std::string>& arguments)
	{
		for (const CliCommand& command : AnalysisCommands()) {
			if (command.name == "action-trace")
				return CliArgs::Parse(arguments, command.options);
		}

		throw std::runtime_error("no action-trace command registered");
	}

	/** Registers the test agent over a temp Models folder holding build V1 with segment v1.0, its training_setup.json and checkpoint 300. */
	fs::path MakeSegmentFixture(const std::string& label)
	{
		const fs::path modelsRoot = RocketForgeTests::TestTempDir(label) / "Agent" / "Models";
		fs::create_directories(modelsRoot);

		AgentRegistry::Register(ANALYSIS_AGENT, [modelsRoot] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = ANALYSIS_AGENT;
			agent->modelPath = modelsRoot;
			return agent;
		});

		BuildStorage storage(modelsRoot, ANALYSIS_BUILD);
		storage.EnsureBuildExists();

		TrainingSegmentMetadata segment;
		segment.segmentId = ANALYSIS_SEGMENT;
		segment.folderName = storage.MakeSegmentFolderName(ANALYSIS_SEGMENT, ANALYSIS_STAGE);
		segment.buildName = ANALYSIS_BUILD;
		segment.stageName = ANALYSIS_STAGE;

		fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / "100");
		fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / "300");
		storage.SaveSegment(segment);

		const fs::path segmentRoot = storage.SegmentRoot(segment.folderName);
		std::ofstream(segmentRoot / "training_setup.json") << "{}";

		return segmentRoot;
	}

}

TEST(SimulationAnalysisOptions, ResolvesSetupAndCheckpointFromSegment)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const fs::path segmentRoot = MakeSegmentFixture("analysis_options_segment");

	const SimulationAnalysisOptions newest =
		ParseSimulationAnalysisOptions(ParseActionTraceArgs({ "--agent", ANALYSIS_AGENT, "--segment", ANALYSIS_SEGMENT }));

	EXPECT_EQ(newest.agentName, ANALYSIS_AGENT);
	EXPECT_EQ(fs::weakly_canonical(newest.setupPath), fs::weakly_canonical(segmentRoot / "training_setup.json"));
	EXPECT_EQ(newest.checkpointPath.filename(), "300");

	const SimulationAnalysisOptions earlier = ParseSimulationAnalysisOptions(
		ParseActionTraceArgs({ "--agent", ANALYSIS_AGENT, "--segment", ANALYSIS_SEGMENT, "--checkpoint", "100" }));

	EXPECT_EQ(earlier.checkpointPath.filename(), "100");
}

TEST(SimulationAnalysisOptions, ExplicitPathsWin)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	MakeSegmentFixture("analysis_options_explicit");

	const SimulationAnalysisOptions options = ParseSimulationAnalysisOptions(ParseActionTraceArgs(
		{ "--agent", ANALYSIS_AGENT, "--segment", ANALYSIS_SEGMENT, "--setup", "a.json", "--checkpoint-path", "c" }));

	EXPECT_EQ(options.setupPath, ResolvePath("a.json"));
	EXPECT_EQ(options.checkpointPath, ResolvePath("c"));
	EXPECT_TRUE(options.setupPath.is_absolute());
	EXPECT_TRUE(options.checkpointPath.is_absolute());

	EXPECT_THROW(
		ParseSimulationAnalysisOptions(ParseActionTraceArgs(
			{ "--agent", ANALYSIS_AGENT, "--setup", "a.json", "--checkpoint-path", "c", "--checkpoint", "100" })),
		CliUsageError);
}

TEST(SimulationAnalysisOptions, KeepsExistingValidation)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	struct Case {
		const char* label;
		std::vector<std::string> arguments;
	};

	const Case cases[] = {
		{ "two reset modes", { "--kickoff", "--wall-probe" } },
		{ "reset mode and stage reset", { "--kickoff", "--stage-reset", "example-ball-chase" } },
		{ "snapshot csv without a tick", { "--snapshot-csv", "trace.csv" } },
		{ "replay filter without a snapshot", { "--snapshot-bot-index", "1" } },
		{ "control replay input delay without a replay", { "--control-replay-input-delay-ticks", "3" } },
		{ "control replay with a forced action",
		  { "--snapshot-csv", "t.csv", "--snapshot-tick", "1", "--control-replay-csv", "t.csv", "--force-action", "2" } },
		{ "zero episodes", { "--episodes", "0" } },
		{ "team size out of range", { "--team-size", "4" } },
		{ "non-numeric value", { "--max-ticks", "many" } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		std::vector<std::string> arguments = { "--agent", "example-agent", "--setup", "a.json", "--checkpoint-path", "c" };
		arguments.insert(arguments.end(), testCase.arguments.begin(), testCase.arguments.end());

		EXPECT_THROW(ParseSimulationAnalysisOptions(ParseActionTraceArgs(arguments)), CliUsageError);
	}
}
