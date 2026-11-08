#include <gtest/gtest.h>

#include "AgentModelPaths.h"
#include "Cli/CommandTable.h"
#include "Settings/ForgeSettings.h"
#include "Storage/BuildStorage.h"
#include "TestSupport.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const PLUMBING_AGENT = "plumbing-command-test-agent";
	const char* const PLUMBING_BUILD = "V1";
	const char* const PLUMBING_STAGE = "example-ball-chase";

	struct CliRun {
		int exitCode = 0;
		std::string out;
		std::string err;
	};

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

	CliRun RunCliWith(const std::vector<std::string>& arguments)
	{
		std::vector<std::string> storage = { "prog" };
		storage.insert(storage.end(), arguments.begin(), arguments.end());

		std::vector<char*> argv;
		for (std::string& argument : storage)
			argv.push_back(argument.data());
		argv.push_back(nullptr);

		std::ostringstream out;
		std::ostringstream err;
		CliRun run;

		{
			const StreamRedirect redirectOut(std::cout, out);
			const StreamRedirect redirectErr(std::cerr, err);
			run.exitCode = RunCli(static_cast<int>(storage.size()), argv.data());
		}

		run.out = out.str();
		run.err = err.str();
		return run;
	}

	/** Registers the test agent over a fresh temp Models folder holding an empty build V1; returns the agent's settings path. */
	fs::path RegisterPlumbingAgent(const std::string& label)
	{
		const fs::path modelsRoot = RocketForgeTests::TestTempDir(label) / "Agent" / "Models";
		fs::create_directories(modelsRoot);

		AgentRegistry::Register(PLUMBING_AGENT, [modelsRoot] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = PLUMBING_AGENT;
			agent->modelPath = modelsRoot;
			return agent;
		});

		BuildStorage(modelsRoot, PLUMBING_BUILD).EnsureBuildExists();

		return GetDefaultForgeSettingsPath(PLUMBING_AGENT);
	}

	/** Saves a one-stage queue on build V1 at path. */
	void SavePlumbingSettings(const fs::path& path)
	{
		ForgeSettings settings = MakeDefaultForgeSettings(PLUMBING_AGENT, PLUMBING_STAGE);
		settings.buildName = PLUMBING_BUILD;
		ASSERT_TRUE(GGL::JsonSaveToFile(settings, path));
	}

	std::vector<std::string> SplitLines(const std::string& text)
	{
		std::vector<std::string> lines;
		std::istringstream stream(text);
		for (std::string line; std::getline(stream, line);)
			lines.push_back(line);

		return lines;
	}

}

TEST(PlumbingCommands, BuildInfoKeepsItsKeyValueContract)
{
	SavePlumbingSettings(RegisterPlumbingAgent("PlumbingBuildInfo"));

	const CliRun run = RunCliWith({ "build-info", "--agent", PLUMBING_AGENT });
	ASSERT_EQ(run.exitCode, 0) << run.err;

	const std::vector<std::string> lines = SplitLines(run.out);
	const std::vector<std::string> keys = { "BUILD_NAME=", "BUILD_FOLDER=", "MODELS_REL=", "START_STAGE=", "START_SEGMENT=", "IS_NEW=", "TAG=" };
	ASSERT_GE(lines.size(), keys.size()) << run.out;

	for (size_t i = 0; i < keys.size(); i++) {
		SCOPED_TRACE(keys[i]);
		EXPECT_EQ(lines[i].rfind(keys[i], 0), 0u) << lines[i];
	}

	EXPECT_EQ(lines[0], std::string("BUILD_NAME=") + PLUMBING_BUILD);
	EXPECT_EQ(lines[3], std::string("START_STAGE=") + PLUMBING_STAGE);
	EXPECT_EQ(lines[5], "IS_NEW=1");

	EXPECT_NE(run.err.find(std::string("agent: ") + PLUMBING_AGENT), std::string::npos) << run.err;
	EXPECT_NE(run.err.find(std::string("build: ") + PLUMBING_BUILD), std::string::npos) << run.err;
}

TEST(PlumbingCommands, SyncManifestDoesNotCreateSettings)
{
	struct Case {
		const char* label;
		std::vector<std::string> arguments;
	};

	const Case cases[] = {
		{ "sync-manifest", { "sync-manifest", "--agent", PLUMBING_AGENT } },
		{ "sync-manifest --metadata-only", { "sync-manifest", "--agent", PLUMBING_AGENT, "--metadata-only" } },
		{ "build-info", { "build-info", "--agent", PLUMBING_AGENT } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		const fs::path settingsPath = RegisterPlumbingAgent(std::string("PlumbingNoSettings_") + testCase.label);
		ASSERT_FALSE(fs::exists(settingsPath));

		const CliRun run = RunCliWith(testCase.arguments);

		EXPECT_EQ(run.exitCode, 1) << run.out << run.err;
		EXPECT_TRUE(run.out.empty()) << run.out;
		EXPECT_NE(run.err.find(std::string("no forge_settings.json for ") + PLUMBING_AGENT), std::string::npos) << run.err;
		EXPECT_FALSE(fs::exists(settingsPath));
	}
}

TEST(PlumbingCommands, RejectsInvalidArgumentsAsUsageErrors)
{
	const fs::path settingsPath = RegisterPlumbingAgent("PlumbingUsage");
	SavePlumbingSettings(settingsPath);

	const std::string missing = (settingsPath.parent_path() / "missing_settings.json").string();
	const std::string existing = settingsPath.string();

	struct Case {
		const char* label;
		std::vector<std::string> arguments;
	};

	const Case cases[] = {
		{ "merge-segment-ids without --from", { "merge-segment-ids", "--agent", PLUMBING_AGENT } },
		{ "both manifest scopes", { "sync-manifest", "--agent", PLUMBING_AGENT, "--metadata-only", "--temp-checkpoints" } },
		{ "positional agent", { "models-path", PLUMBING_AGENT } },
		{ "build-info --file missing", { "build-info", "--agent", PLUMBING_AGENT, "--file", missing } },
		{ "sync-manifest --file missing", { "sync-manifest", "--agent", PLUMBING_AGENT, "--file", missing } },
		{ "sync-manifest --build --file missing", { "sync-manifest", "--agent", PLUMBING_AGENT, "--build", PLUMBING_BUILD, "--file", missing } },
		{ "merge-segment-ids --from missing", { "merge-segment-ids", "--agent", PLUMBING_AGENT, "--from", missing } },
		{ "merge-segment-ids --file missing", { "merge-segment-ids", "--agent", PLUMBING_AGENT, "--file", missing, "--from", existing } },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		const CliRun run = RunCliWith(testCase.arguments);

		EXPECT_EQ(run.exitCode, 2) << run.out << run.err;
		EXPECT_TRUE(run.out.empty()) << run.out;
	}
}

TEST(PlumbingCommands, ModelsPathPrintsOnlyThePathOnStdout)
{
	RegisterPlumbingAgent("PlumbingModelsPath");

	const CliRun run = RunCliWith({ "models-path", "--agent", PLUMBING_AGENT });
	ASSERT_EQ(run.exitCode, 0) << run.err;

	EXPECT_EQ(run.out, ResolveAgentModelsRelativePath(PLUMBING_AGENT) + "\n");
	EXPECT_NE(run.err.find(std::string("agent: ") + PLUMBING_AGENT), std::string::npos) << run.err;
}

TEST(PlumbingCommands, SyncManifestListsABuildThatIsNotLocal)
{
	SavePlumbingSettings(RegisterPlumbingAgent("PlumbingRemoteBuild"));

	const CliRun run = RunCliWith({ "sync-manifest", "--agent", PLUMBING_AGENT, "--build", "RemoteOnly" });

	ASSERT_EQ(run.exitCode, 0) << run.err;
	EXPECT_NE(run.out.find("/RemoteOnly/"), std::string::npos) << run.out;
	EXPECT_NE(run.err.find("build: RemoteOnly"), std::string::npos) << run.err;
}

TEST(PlumbingCommands, MergeSegmentIdsRewritesTheLocalFileOnlyWhenSomethingChanged)
{
	const fs::path localPath = RegisterPlumbingAgent("PlumbingMerge");
	SavePlumbingSettings(localPath);

	ForgeSettings remote = MakeDefaultForgeSettings(PLUMBING_AGENT, PLUMBING_STAGE);
	remote.buildName = PLUMBING_BUILD;
	remote.stages[0].createdSegmentId = "v1.3";
	const fs::path remotePath = localPath.parent_path() / "remote_settings.json";
	ASSERT_TRUE(GGL::JsonSaveToFile(remote, remotePath));

	const CliRun adopted = RunCliWith({ "merge-segment-ids", "--agent", PLUMBING_AGENT, "--from", remotePath.string() });
	ASSERT_EQ(adopted.exitCode, 0) << adopted.err;
	EXPECT_EQ(adopted.out, "Adopted 1 recorded segment id(s) into " + localPath.string() + "\n");

	const ForgeSettingsLoadResult merged = LoadForgeSettings(localPath);
	ASSERT_TRUE(merged.ok);
	EXPECT_EQ(merged.value.stages[0].createdSegmentId, "v1.3");

	const fs::file_time_type writtenAt = fs::last_write_time(localPath);

	const CliRun unchanged = RunCliWith({ "merge-segment-ids", "--agent", PLUMBING_AGENT, "--from", remotePath.string() });
	ASSERT_EQ(unchanged.exitCode, 0) << unchanged.err;
	EXPECT_EQ(unchanged.out, "Adopted 0 recorded segment id(s) into " + localPath.string() + "\n");
	EXPECT_EQ(fs::last_write_time(localPath), writtenAt);
}
