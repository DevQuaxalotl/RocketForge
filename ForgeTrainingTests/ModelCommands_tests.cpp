#include <gtest/gtest.h>

#include "Cli/CliErrors.h"
#include "Cli/CommandTable.h"
#include "Commands/Commands.h"
#include "Storage/BuildStorage.h"
#include "Storage/TrainingBuildLock.h"
#include "TestSupport.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const MODEL_AGENT = "model-command-test-agent";
	const char* const MODEL_BUILD = "V1";
	const char* const MODEL_SEGMENT = "v1.0";
	const char* const MODEL_STAGE = "example-ball-chase";

	struct ModelFixture {
		fs::path modelsRoot;
		fs::path segmentRoot;
		fs::path checkpointsRoot;
		fs::path buildRoot;
	};

	/** Saves segment v1.0 in the build, with one durable checkpoint per entry. */
	ModelFixture AddBuildWithSegment(const fs::path& modelsRoot, const std::string& buildName, const std::vector<int>& checkpoints)
	{
		BuildStorage storage(modelsRoot, buildName);
		storage.EnsureBuildExists();

		TrainingSegmentMetadata segment;
		segment.segmentId = MODEL_SEGMENT;
		segment.folderName = storage.MakeSegmentFolderName(MODEL_SEGMENT, MODEL_STAGE);
		segment.buildName = buildName;
		segment.stageName = MODEL_STAGE;

		for (int timesteps : checkpoints)
			fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / std::to_string(timesteps));

		storage.SaveSegment(segment);

		ModelFixture fixture;
		fixture.modelsRoot = modelsRoot;
		fixture.segmentRoot = storage.SegmentRoot(segment.folderName);
		fixture.checkpointsRoot = storage.SegmentCheckpointsRoot(segment.folderName);
		fixture.buildRoot = storage.BuildRoot();

		return fixture;
	}

	/** Registers the test agent over a fresh temp Models folder holding build V1 with segment v1.0. */
	ModelFixture MakeModelFixture(const std::string& label, const std::vector<int>& checkpoints)
	{
		const fs::path modelsRoot = RocketForgeTests::TestTempDir(label) / "Agent" / "Models";
		fs::create_directories(modelsRoot);

		AgentRegistry::Register(MODEL_AGENT, [modelsRoot] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = MODEL_AGENT;
			agent->modelPath = modelsRoot;
			return agent;
		});

		return AddBuildWithSegment(modelsRoot, MODEL_BUILD, checkpoints);
	}

	/** Parses a command's arguments with the test agent and the given build appended as its target. */
	CliArgs ParseModelArgs(const std::string& commandName, std::vector<std::string> arguments, const char* buildName = MODEL_BUILD)
	{
		for (const char* option : { "--agent", MODEL_AGENT, "--build", buildName })
			arguments.push_back(option);

		for (const CliCommand& command : ModelCommands()) {
			if (command.name == commandName)
				return CliArgs::Parse(arguments, command.options);
		}

		throw std::runtime_error("no " + commandName + " command registered");
	}

	size_t CountCheckpoints(const fs::path& checkpointsRoot)
	{
		size_t count = 0;
		for (const fs::directory_entry& entry : fs::directory_iterator(checkpointsRoot)) {
			if (entry.is_directory())
				count++;
		}

		return count;
	}

	int RunDelete(
		const std::vector<std::string>& arguments,
		bool interactive,
		const std::string& input,
		std::string* output = nullptr,
		const char* buildName = MODEL_BUILD)
	{
		std::istringstream in(input);
		std::ostringstream out;
		const int exitCode = RunDeleteSegmentCommand(ParseModelArgs("delete-segment", arguments, buildName), interactive, in, out);

		if (output)
			*output = out.str();

		return exitCode;
	}

	int RunPrune(const std::vector<std::string>& arguments, bool interactive, const std::string& input, std::string* output = nullptr)
	{
		std::istringstream in(input);
		std::ostringstream out;
		const int exitCode = RunPruneCommand(ParseModelArgs("prune", arguments), interactive, in, out);

		if (output)
			*output = out.str();

		return exitCode;
	}

}

TEST(ModelCommands, DeleteRefusesWithoutYesWhenNonInteractive)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_delete_refuse", { 100, 200 });

	std::string output;
	const bool interactive = false;

	EXPECT_EQ(RunDelete({ "--segment", MODEL_SEGMENT }, interactive, "", &output), 2);
	EXPECT_TRUE(fs::exists(fixture.segmentRoot));
	EXPECT_NE(output.find("Delete segment v1.0"), std::string::npos) << output;
}

TEST(ModelCommands, DeleteRemovesTheSegmentWhenConfirmed)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_delete_yes", { 100, 200 });

	const bool interactive = false;

	EXPECT_EQ(RunDelete({ "--segment", MODEL_SEGMENT, "--yes" }, interactive, ""), 0);
	EXPECT_FALSE(fs::exists(fixture.segmentRoot));
}

TEST(ModelCommands, DeclinedConfirmationDeletesNothing)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_delete_decline", { 100, 200 });

	const bool interactive = true;

	EXPECT_EQ(RunDelete({ "--segment", MODEL_SEGMENT }, interactive, "n\n"), 1);
	EXPECT_TRUE(fs::exists(fixture.segmentRoot));
}

TEST(ModelCommands, DeleteRequiresAKnownSegment)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_delete_unknown", { 100 });

	struct Case {
		const char* label;
		std::vector<std::string> arguments;
		const char* expectedMessage;
	};

	const Case cases[] = {
		{ "segment omitted", { "--yes" }, "--segment is required" },
		{ "segment unknown", { "--segment", "v9.9", "--yes" }, "list segments" },
	};

	const bool interactive = false;
	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		try {
			RunDelete(testCase.arguments, interactive, "");
			ADD_FAILURE() << "expected a CliUsageError";
		} catch (const CliUsageError& error) {
			EXPECT_NE(std::string(error.what()).find(testCase.expectedMessage), std::string::npos) << error.what();
		}
	}

	EXPECT_TRUE(fs::exists(fixture.segmentRoot));
}

TEST(ModelCommands, DeleteWithAnEmptySegmentIsAUsageError)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_delete_empty_segment", { 100, 200 });

	std::vector<std::string> arguments = {
		"ForgeConsoleTrainer", "delete-segment", "--agent", MODEL_AGENT, "--build", MODEL_BUILD, "--segment", "", "--yes",
	};
	std::vector<char*> argv;
	for (std::string& argument : arguments)
		argv.push_back(argument.data());
	argv.push_back(nullptr);

	testing::internal::CaptureStderr();
	const int exitCode = RunCli(static_cast<int>(arguments.size()), argv.data());
	const std::string errors = testing::internal::GetCapturedStderr();

	EXPECT_EQ(exitCode, 2);
	EXPECT_NE(errors.find("--segment requires a non-empty value"), std::string::npos) << errors;
	EXPECT_TRUE(fs::exists(fixture.segmentRoot));
	EXPECT_EQ(CountCheckpoints(fixture.checkpointsRoot), 2u);
}

TEST(ModelCommands, DeleteOnlyTouchesTheTargetBuild)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture target = MakeModelFixture("model_delete_cross_build", { 100, 200 });
	const ModelFixture other = AddBuildWithSegment(target.modelsRoot, "V2", { 100, 200 });

	const bool interactive = false;

	EXPECT_EQ(RunDelete({ "--segment", MODEL_SEGMENT, "--yes" }, interactive, "", nullptr, "V2"), 0);
	EXPECT_FALSE(fs::exists(other.segmentRoot));
	EXPECT_TRUE(fs::exists(target.segmentRoot));
	EXPECT_EQ(CountCheckpoints(target.checkpointsRoot), 2u);
}

TEST(ModelCommands, PruneKeepsOnlyTheNewestCheckpoint)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_prune_keep", { 100, 200 });

	const bool interactive = false;

	EXPECT_EQ(RunPrune({ "--segment", MODEL_SEGMENT, "--yes" }, interactive, ""), 0);
	EXPECT_EQ(CountCheckpoints(fixture.checkpointsRoot), 1u);
	EXPECT_TRUE(fs::exists(fixture.checkpointsRoot / "200"));
}

TEST(ModelCommands, PruneWithNothingToRemoveSucceeds)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_prune_nothing", { 100 });

	std::string output;
	const bool interactive = false;

	EXPECT_EQ(RunPrune({ "--all" }, interactive, "", &output), 0);
	EXPECT_EQ(CountCheckpoints(fixture.checkpointsRoot), 1u);
	EXPECT_NE(output.find("Nothing to prune"), std::string::npos) << output;
}

TEST(ModelCommands, PruneRefusesWithoutYesWhenNonInteractive)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_prune_refuse", { 100, 200 });

	const bool interactive = false;

	EXPECT_EQ(RunPrune({ "--all" }, interactive, ""), 2);
	EXPECT_EQ(CountCheckpoints(fixture.checkpointsRoot), 2u);
}

TEST(ModelCommands, DestructiveCommandsRefuseWhileTheBuildIsLocked)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_locked", { 100, 200 });
	const TrainingBuildLock trainerLock(fixture.buildRoot);

	const bool interactive = false;

	EXPECT_THROW(RunPrune({ "--all", "--yes" }, interactive, ""), CliRuntimeError);
	EXPECT_THROW(RunDelete({ "--segment", MODEL_SEGMENT, "--yes" }, interactive, ""), CliRuntimeError);

	EXPECT_EQ(CountCheckpoints(fixture.checkpointsRoot), 2u);
	EXPECT_TRUE(fs::exists(fixture.segmentRoot));
}

TEST(ModelCommands, PruneRequiresExactlyOneTarget)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_prune_target", { 100, 200 });

	struct Case {
		const char* label;
		std::vector<std::string> arguments;
	};

	const Case cases[] = {
		{ "neither", { "--yes" } },
		{ "both", { "--segment", MODEL_SEGMENT, "--all", "--yes" } },
	};

	const bool interactive = false;
	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		EXPECT_THROW(RunPrune(testCase.arguments, interactive, ""), CliUsageError);
	}

	EXPECT_EQ(CountCheckpoints(fixture.checkpointsRoot), 2u);
}

TEST(ModelCommands, PruneNothingMessageNamesTheTarget)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	MakeModelFixture("model_prune_message", { 100 });

	std::string output;
	const bool interactive = false;

	EXPECT_EQ(RunPrune({ "--segment", MODEL_SEGMENT }, interactive, "", &output), 0);
	EXPECT_NE(output.find("segment v1.0 has at most one checkpoint"), std::string::npos) << output;
}

TEST(ModelCommands, DeleteSummaryCountsAutosaves)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_delete_summary", { 100 });
	fs::create_directories(fixture.segmentRoot / BuildStorage::TEMP_CHECKPOINTS_FOLDER_NAME / "150");

	std::string output;
	const bool interactive = false;

	EXPECT_EQ(RunDelete({ "--segment", MODEL_SEGMENT }, interactive, "", &output), 2);
	EXPECT_NE(output.find("1 checkpoint, 1 autosave"), std::string::npos) << output;
}

TEST(ModelCommands, ExportRlbotRefusesASegmentWithoutCheckpoints)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	MakeModelFixture("model_export_empty", {});

	try {
		RunExportRlbotCommand(ParseModelArgs("export-rlbot", { "--segment", MODEL_SEGMENT }));
		FAIL() << "a segment with no checkpoints must not export";
	} catch (const CliRuntimeError& error) {
		EXPECT_NE(std::string(error.what()).find("no checkpoints to export"), std::string::npos) << error.what();
	}
}

TEST(ModelCommands, ExportRlbotRefusesWhileTheBuildIsLocked)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_export_locked", { 100 });
	const TrainingBuildLock trainerLock(fixture.buildRoot);

	EXPECT_THROW(RunExportRlbotCommand(ParseModelArgs("export-rlbot", { "--segment", MODEL_SEGMENT })), CliRuntimeError);
	EXPECT_THROW(RunExportRlbotCommand(ParseModelArgs("export-rlbot", { "--segment", MODEL_SEGMENT, "--remove" })), CliRuntimeError);
}

TEST(ModelCommands, ExportSubmittalRequiresAnOutputFolder)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	MakeModelFixture("model_submittal_out", { 100 });

	EXPECT_THROW(RunExportSubmittalCommand(ParseModelArgs("export-submittal", { "--segment", MODEL_SEGMENT })), CliUsageError);
}

TEST(ModelCommands, ExportSubmittalRejectsAnUnknownDevice)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();
	const ModelFixture fixture = MakeModelFixture("model_submittal_device", { 100 });

	try {
		RunExportSubmittalCommand(ParseModelArgs(
			"export-submittal",
			{ "--segment", MODEL_SEGMENT, "--out", fixture.modelsRoot.parent_path().string(), "--device", "cuda" }));
		FAIL() << "only cpu and gpu are devices";
	} catch (const CliUsageError& error) {
		EXPECT_NE(std::string(error.what()).find("--device must be cpu or gpu"), std::string::npos) << error.what();
	}
}
