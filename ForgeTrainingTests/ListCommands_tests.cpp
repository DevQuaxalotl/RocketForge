#include <gtest/gtest.h>

#include "Cli/CliErrors.h"
#include "Commands/Commands.h"
#include "Storage/BuildStorage.h"
#include "TestSupport.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const LIST_AGENT = "list-command-test-agent";
	const char* const LIST_BUILD = "V1";

	/** Registers an agent whose Models folder is a fresh temp tree, and returns that folder. */
	fs::path RegisterListFixtureAgent()
	{
		const fs::path modelsRoot = RocketForgeTests::TestTempDir("list_commands") / "Agent" / "Models";
		fs::create_directories(modelsRoot);

		AgentRegistry::Register(LIST_AGENT, [modelsRoot] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = LIST_AGENT;
			agent->modelPath = modelsRoot;
			return agent;
		});

		return modelsRoot;
	}

	/** Saves a segment whose checkpoints/ holds one folder per entry of durable and temp_checkpoints/ one per entry of autosaves. */
	void SaveSegment(
		const BuildStorage& storage,
		const std::string& segmentId,
		const std::vector<int>& durable,
		const std::vector<int>& autosaves)
	{
		TrainingSegmentMetadata segment;
		segment.segmentId = segmentId;
		segment.folderName = storage.MakeSegmentFolderName(segmentId, "example-ball-chase");
		segment.buildName = LIST_BUILD;
		segment.stageName = "example-ball-chase";

		for (int timesteps : durable)
			fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / std::to_string(timesteps));
		for (int timesteps : autosaves)
			fs::create_directories(storage.SegmentTempCheckpointsRoot(segment.folderName) / std::to_string(timesteps));

		storage.SaveSegment(segment);
	}

	/** Points stream at target until destroyed, so a throwing test can't leave the stream redirected. */
	class StreamRedirect {
	public:
		StreamRedirect(std::ostream& stream, std::ostream& target) : _stream(stream), _previous(stream.rdbuf(target.rdbuf())) {}
		~StreamRedirect() { _stream.rdbuf(_previous); }

		StreamRedirect(const StreamRedirect&) = delete;
		StreamRedirect& operator=(const StreamRedirect&) = delete;

	private:
		std::ostream& _stream;
		std::streambuf* _previous;
	};

	CliArgs ParseListArgs(const std::vector<std::string>& arguments)
	{
		for (const CliCommand& command : InspectCommands()) {
			if (command.name == "list")
				return CliArgs::Parse(arguments, command.options);
		}

		throw std::runtime_error("no list command registered");
	}

}

TEST(ListCommands, SegmentsMarkTheActiveSegment)
{
	const fs::path modelsRoot = RegisterListFixtureAgent();
	BuildStorage storage(modelsRoot, LIST_BUILD);
	storage.EnsureBuildExists();
	SaveSegment(storage, "v1.00", { 100 }, {});
	SaveSegment(storage, "v1.01", { 200 }, {});

	BuildMetadata build = storage.LoadBuild();
	build.activeSegmentId = "v1.01";
	storage.SaveBuild(build);

	TrainingSegmentMetadata exported = *storage.LoadSegmentById("v1.00");
	exported.eligibleRLBot = true;
	storage.SaveSegment(exported);

	// The summary only reports an export whose files are all on disk.
	const fs::path rlbotRoot = storage.SegmentRoot(exported.folderName) / "rlbot";
	fs::create_directories(rlbotRoot);
	for (const char* fileName : { "bot.toml", "run.bat", "RLBot.exe" })
		std::ofstream(rlbotRoot / fileName) << "x";

	const ListOutput output = ListSubject("segments", ParseListArgs({ "segments", "--agent", LIST_AGENT, "--build", LIST_BUILD }));

	ASSERT_EQ(output.rows.size(), 2u);
	ASSERT_EQ(output.headers.size(), output.rows[0].size());
	EXPECT_EQ(output.rows[0][0], "v1.00");
	EXPECT_EQ(output.rows[1][0], "v1.01");
	EXPECT_EQ(output.rows[1].back(), "*");
	EXPECT_EQ(output.rows[0].back(), "");

	ASSERT_EQ(output.json.size(), 2u);
	EXPECT_FALSE(output.json[0]["active"].get<bool>());
	EXPECT_TRUE(output.json[1]["active"].get<bool>());
	EXPECT_TRUE(output.json[0]["exported"].get<bool>());
	EXPECT_FALSE(output.json[1]["exported"].get<bool>());
}

TEST(ListCommands, CheckpointsSeparateAutosaves)
{
	const fs::path modelsRoot = RegisterListFixtureAgent();
	BuildStorage storage(modelsRoot, LIST_BUILD);
	storage.EnsureBuildExists();
	SaveSegment(storage, "v1.00", { 100 }, { 200 });

	const ListOutput output = ListSubject(
		"checkpoints", ParseListArgs({ "checkpoints", "--agent", LIST_AGENT, "--build", LIST_BUILD, "--segment", "v1.00" }));

	ASSERT_EQ(output.rows.size(), 2u);
	EXPECT_EQ(output.rows[0][0], "100");
	EXPECT_EQ(output.rows[0][1], "checkpoint");
	EXPECT_EQ(output.rows[1][0], "200");
	EXPECT_EQ(output.rows[1][1], "autosave");
	EXPECT_EQ(output.json[1]["kind"].get<std::string>(), "autosave");
}

TEST(ListCommands, StageOptionsListSettersAndTeamSizes)
{
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	const ListOutput output = ListSubject("stage-options", ParseListArgs({ "stage-options", "--stage", "example-ball-chase" }));

	bool foundSetter = false;
	bool foundOneVsOne = false;
	for (const std::vector<std::string>& row : output.rows) {
		foundSetter = foundSetter || row[0] == "state-setter";
		foundOneVsOne = foundOneVsOne || (row[0] == "team-size" && row[1] == "0" && row[2] == "1v1");
	}

	EXPECT_TRUE(foundSetter);
	EXPECT_TRUE(foundOneVsOne);
}

TEST(ListCommands, StageOptionsRequireAStage)
{
	EXPECT_THROW(ListSubject("stage-options", ParseListArgs({ "stage-options" })), CliUsageError);
}

TEST(ListCommands, JsonModeKeepsStdoutMachineReadable)
{
	std::ostringstream out;
	std::ostringstream err;
	{
		const StreamRedirect redirectOut(std::cout, out);
		const StreamRedirect redirectErr(std::cerr, err);
		EXPECT_EQ(RunListCommand(ParseListArgs({ "imported-models", "--json", "--agent", "example-agent" })), 0);
	}

	EXPECT_TRUE(nlohmann::json::parse(out.str(), nullptr, false).is_array()) << out.str();
	EXPECT_EQ(out.str().find("agent:"), std::string::npos) << out.str();
	EXPECT_NE(err.str().find("agent: example-agent"), std::string::npos) << err.str();
}

TEST(ListCommands, UnknownSubjectSuggestsOne)
{
	try {
		ListSubject("segment", ParseListArgs({ "segment" }));
		FAIL() << "expected CliUsageError";
	} catch (const CliUsageError& error) {
		EXPECT_NE(std::string(error.what()).find("segments"), std::string::npos) << error.what();
	}
}

TEST(ListCommands, ListsRegisteredAgentsAndStages)
{
	const ListOutput agents = ListSubject("agents", ParseListArgs({ "agents" }));
	const ListOutput stages = ListSubject("stages", ParseListArgs({ "stages" }));

	bool foundExampleAgent = false;
	for (const std::vector<std::string>& row : agents.rows)
		foundExampleAgent = foundExampleAgent || row[0] == "example-agent";

	bool foundExampleStage = false;
	for (const std::vector<std::string>& row : stages.rows)
		foundExampleStage = foundExampleStage || row[0] == "example-ball-chase";

	EXPECT_TRUE(foundExampleAgent);
	EXPECT_TRUE(foundExampleStage);
}
