#include <gtest/gtest.h>

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"
#include "Settings/ForgeSettings.h"
#include "TestSupport.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace RocketForge::Cli;

TEST(CliResolve, UnknownNameNamesTheListCommand)
{
	try {
		RequireKnownName("stage", "ball-chas", { "example-ball-chase" }, "list stages");
		FAIL() << "expected CliUsageError";
	} catch (const CliUsageError& error) {
		const std::string message = error.what();
		EXPECT_NE(message.find("example-ball-chase"), std::string::npos) << message;
		EXPECT_NE(message.find("ForgeConsoleTrainer list stages"), std::string::npos) << message;
	}

	EXPECT_EQ(RequireKnownName("stage", "example-ball-chase", { "example-ball-chase" }, "list stages"), "example-ball-chase");
}

TEST(CliResolve, AgentFallsBackFromFlagToSavedToExample)
{
	const CliOptionSpec spec{ { "--agent" }, {}, 0 };

	struct Case {
		const char* label;
		std::vector<std::string> arguments;
		std::string savedAgent;
		std::vector<std::string> knownAgents;
		std::string expectedAgent;
		std::string expectedErrorText;
	};
	const Case cases[] = {
		{ "flag wins over saved", { "--agent", "b" }, "a", { "a", "b" }, "b", "" },
		{ "saved when no flag", {}, "a", { "a", "b" }, "a", "" },
		{ "example when nothing saved", {}, "", { "example-agent" }, "example-agent", "" },
		{ "unknown flag value", { "--agent", "zz" }, "", { "a" }, "", "list agents" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		const CliArgs args = CliArgs::Parse(testCase.arguments, spec);

		if (testCase.expectedErrorText.empty()) {
			EXPECT_EQ(ResolveAgentName(args, testCase.savedAgent, testCase.knownAgents), testCase.expectedAgent);
			continue;
		}

		try {
			ResolveAgentName(args, testCase.savedAgent, testCase.knownAgents);
			ADD_FAILURE() << "expected CliUsageError";
		} catch (const CliUsageError& error) {
			EXPECT_NE(std::string(error.what()).find(testCase.expectedErrorText), std::string::npos) << error.what();
		}
	}
}

TEST(CliResolve, PrintsResolvedFieldsOnOneLine)
{
	std::ostringstream out;

	PrintResolved(out, { { "agent", "x" }, { "build", "y" }, { "segment", "z" } });

	EXPECT_EQ(out.str(), "agent: x  build: y  segment: z\n");
}

TEST(CliResolve, BuildComesFromSettingsOnlyWhenTheyLoad)
{
	namespace fs = std::filesystem;

	const fs::path root = RocketForgeTests::TestTempDir("CliResolveBuild");
	const fs::path missingPath = root / "missing.json";
	const fs::path malformedPath = root / "malformed.json";
	const fs::path validPath = root / "valid.json";

	std::ofstream(malformedPath) << "{ not json";

	RocketForgeTraining::ForgeSettings settings;
	settings.agentName = "example-agent";
	settings.buildName = "Chosen";
	ASSERT_TRUE(GGL::JsonSaveToFile(settings, validPath));

	enum class Expect { Build, UsageError, RuntimeError };
	struct Case {
		const char* label;
		std::vector<std::string> arguments;
		fs::path settingsPath;
		Expect expect;
		std::string expectedBuild;
	};
	const Case cases[] = {
		{ "missing default settings", {}, missingPath, Expect::Build, GGL::DEFAULT_BUILD_NAME },
		{ "missing explicit --file", { "--file", missingPath.string() }, missingPath, Expect::UsageError, "" },
		{ "malformed settings", {}, malformedPath, Expect::RuntimeError, "" },
		{ "settings name a build", {}, validPath, Expect::Build, "Chosen" },
	};

	const CliOptionSpec spec{ { "--file", "--build" }, {}, 0 };

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		const CliArgs args = CliArgs::Parse(testCase.arguments, spec);

		switch (testCase.expect) {
		case Expect::Build:
			EXPECT_EQ(ResolveBuild(args, "example-agent", testCase.settingsPath), testCase.expectedBuild);
			break;
		case Expect::UsageError:
			EXPECT_THROW(ResolveBuild(args, "example-agent", testCase.settingsPath), CliUsageError);
			break;
		case Expect::RuntimeError:
			EXPECT_THROW(ResolveBuild(args, "example-agent", testCase.settingsPath), CliRuntimeError);
			break;
		}
	}
}

TEST(CliResolve, CheckpointPicksRequestedOrNewest)
{
	namespace fs = std::filesystem;

	const fs::path segmentRoot = RocketForgeTests::TestTempDir("CliResolveCheckpoint") / "v1.0_stage";
	for (const char* timesteps : { "100", "200" })
		fs::create_directories(segmentRoot / "checkpoints" / timesteps);
	fs::create_directories(segmentRoot / "temp_checkpoints" / "150");

	const CliOptionSpec spec{ { "--checkpoint" }, {}, 0 };

	const RocketForgeTraining::ResolvedCheckpoint newest = ResolveCheckpoint(CliArgs::Parse({}, spec), segmentRoot);
	EXPECT_EQ(newest.timesteps, 200u);
	EXPECT_FALSE(newest.isAutosave);

	const RocketForgeTraining::ResolvedCheckpoint autosave = ResolveCheckpoint(CliArgs::Parse({ "--checkpoint", "150" }, spec), segmentRoot);
	EXPECT_EQ(autosave.timesteps, 150u);
	EXPECT_TRUE(autosave.isAutosave);
	EXPECT_EQ(autosave.folder.filename(), "150");

	try {
		ResolveCheckpoint(CliArgs::Parse({ "--checkpoint", "999" }, spec), segmentRoot);
		ADD_FAILURE() << "expected CliUsageError";
	} catch (const CliUsageError& error) {
		EXPECT_NE(std::string(error.what()).find("list checkpoints"), std::string::npos) << error.what();
	}

	const fs::path emptySegment = RocketForgeTests::TestTempDir("CliResolveCheckpointEmpty") / "v2.0_stage";
	fs::create_directories(emptySegment);
	EXPECT_THROW(ResolveCheckpoint(CliArgs::Parse({}, spec), emptySegment), CliRuntimeError);
}
