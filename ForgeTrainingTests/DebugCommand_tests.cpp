#include <gtest/gtest.h>

#include "Cli/CliErrors.h"
#include "Commands/Commands.h"
#include "TestSupport.h"
#include "TrainingStage.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace RocketForge;
using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace {

	CliArgs ParseDebugArgs(const std::vector<std::string>& arguments)
	{
		for (const CliCommand& command : DebugCommands()) {
			if (command.name == "debug")
				return CliArgs::Parse(arguments, command.options);
		}

		throw std::runtime_error("no debug command registered");
	}

	/** A file selection the overrides start from: a trained segment with a pinned opponent. */
	ForgeDebugSettings MakeLoadedSettings()
	{
		ForgeDebugSettings loaded;
		loaded.buildName = "Saved";
		loaded.segmentId = "v1.2";
		loaded.debug.includeStaticOpponent = true;
		loaded.debug.staticOpponentSelectionKey = "pinned";

		return loaded;
	}

}

TEST(DebugCommand, OverridesMapToDebugSettingsEncoding)
{
	const std::vector<WeightedTeamSize> stageTeamSizes = { WeightedTeamSize(1, 1.f), WeightedTeamSize(2, 1.f) };

	struct Case {
		const char* label;
		std::vector<std::string> arguments;
		std::function<void(const ForgeDebugSettings&)> check;
	};

	const Case cases[] = {
		{ "no flags keep the file", {}, [](const ForgeDebugSettings& settings) {
			EXPECT_EQ(settings.buildName, "Saved");
			EXPECT_EQ(settings.segmentId, "v1.2");
			EXPECT_EQ(settings.debug.teamSizeOverrideIndex, -1);
		} },
		{ "segment keeps the build", { "--segment", "v1.5" }, [](const ForgeDebugSettings& settings) {
			EXPECT_EQ(settings.buildName, "Saved");
			EXPECT_EQ(settings.segmentId, "v1.5");
		} },
		{ "build and segment", { "--build", "Other", "--segment", "v2.0" }, [](const ForgeDebugSettings& settings) {
			EXPECT_EQ(settings.buildName, "Other");
			EXPECT_EQ(settings.segmentId, "v2.0");
		} },
		{ "another build alone uses its active segment", { "--build", "Other" }, [](const ForgeDebugSettings& settings) {
			EXPECT_EQ(settings.buildName, "Other");
			EXPECT_TRUE(settings.segmentId.empty());
		} },
		{ "imported", { "--imported", "F" }, [](const ForgeDebugSettings& settings) {
			EXPECT_TRUE(settings.buildName.empty());
			EXPECT_EQ(settings.segmentId, "F");
		} },
		{ "stage", { "--stage", "example-basic-score" }, [](const ForgeDebugSettings& settings) {
			EXPECT_EQ(settings.renderStageName, "example-basic-score");
		} },
		{ "state setter", { "--state-setter", "2" }, [](const ForgeDebugSettings& settings) {
			EXPECT_TRUE(settings.debug.forceSingleStateSetter);
			EXPECT_EQ(settings.debug.forceStateSetterIndex, 2);
		} },
		{ "team size", { "--team-size", "2v2" }, [](const ForgeDebugSettings& settings) {
			EXPECT_EQ(settings.debug.teamSizeOverrideIndex, 1);
		} },
		{ "player", { "--player", "1" }, [](const ForgeDebugSettings& settings) {
			EXPECT_EQ(settings.debug.playerIndex, 1);
		} },
		{ "opponent key", { "--opponent", "key-a" }, [](const ForgeDebugSettings& settings) {
			EXPECT_TRUE(settings.debug.includeStaticOpponent);
			EXPECT_EQ(settings.debug.staticOpponentSelectionKey, "key-a");
		} },
		{ "random opponent", { "--opponent", "random" }, [](const ForgeDebugSettings& settings) {
			EXPECT_TRUE(settings.debug.includeStaticOpponent);
			EXPECT_TRUE(settings.debug.staticOpponentSelectionKey.empty());
		} },
		{ "no opponent", { "--opponent", "none" }, [](const ForgeDebugSettings& settings) {
			EXPECT_FALSE(settings.debug.includeStaticOpponent);
		} },
		{ "episode seconds", { "--episode-seconds", "12.5" }, [](const ForgeDebugSettings& settings) {
			EXPECT_TRUE(settings.debug.maxEpisodeTimeEnabled);
			EXPECT_FLOAT_EQ(settings.debug.maxEpisodeTime, 12.5f);
		} },
		{ "speed", { "--speed", "2" }, [](const ForgeDebugSettings& settings) {
			EXPECT_FLOAT_EQ(settings.debug.gameSpeed, 2.f);
		} },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		testCase.check(ApplyDebugOverrides(ParseDebugArgs(testCase.arguments), MakeLoadedSettings(), stageTeamSizes));
	}

	struct UsageErrorCase {
		const char* label;
		std::vector<std::string> arguments;
		const char* expectedText;
	};

	const UsageErrorCase usageErrors[] = {
		{ "unknown team size", { "--team-size", "3v3" }, "list stage-options" },
		{ "malformed team size", { "--team-size", "big" }, "list stage-options" },
		{ "segment with imported", { "--segment", "v1.0", "--imported", "F" }, "--imported" },
		{ "build with imported", { "--build", "Other", "--imported", "F" }, "--imported" },
	};

	for (const UsageErrorCase& testCase : usageErrors) {
		SCOPED_TRACE(testCase.label);

		try {
			ApplyDebugOverrides(ParseDebugArgs(testCase.arguments), MakeLoadedSettings(), stageTeamSizes);
			ADD_FAILURE() << "expected CliUsageError";
		} catch (const CliUsageError& error) {
			EXPECT_NE(std::string(error.what()).find(testCase.expectedText), std::string::npos) << error.what();
		}
	}
}
