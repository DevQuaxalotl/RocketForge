#include <gtest/gtest.h>

#include <Utils/BotScaffold.h>
#include "Cli/CliErrors.h"
#include "Cli/CommandTable.h"
#include "Commands/Commands.h"
#include "TestSupport.h"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

using namespace RocketForge::Cli;

namespace fs = std::filesystem;

namespace {

	CliArgs ParseCreateBotArgs(const std::vector<std::string>& arguments)
	{
		CliOptionSpec spec;
		spec.maxPositionals = 1;
		return CliArgs::Parse(arguments, spec);
	}
}

TEST(CreateBotCommand, RequiresAName)
{
	std::ostringstream out;
	EXPECT_THROW(RunCreateBotCommand(ParseCreateBotArgs({}), RocketForge::DefaultExampleBotRoot(),
		RocketForgeTests::TestTempDir("create_bot_no_name"), {}, out), CliUsageError);
}

TEST(CreateBotCommand, RejectsTwoNames)
{
	EXPECT_THROW(ParseCreateBotArgs({ "A", "B" }), CliUsageError);
}

TEST(CreateBotCommand, InvalidNameIsUsageError)
{
	std::ostringstream out;

	try {
		RunCreateBotCommand(ParseCreateBotArgs({ "Sky-Hawk" }), RocketForge::DefaultExampleBotRoot(),
			RocketForgeTests::TestTempDir("create_bot_invalid"), {}, out);
		FAIL() << "expected CliUsageError";
	} catch (const CliUsageError& error) {
		EXPECT_STREQ(error.what(), "'Sky-Hawk' must start with a letter and contain only letters and digits");
	}

	EXPECT_TRUE(out.str().empty());
}

TEST(CreateBotCommand, CreatesAndPrintsNextSteps)
{
	const fs::path botsRoot = RocketForgeTests::TestTempDir("create_bot_creates");
	std::ostringstream out;

	const int exitCode = RunCreateBotCommand(ParseCreateBotArgs({ "MyBot" }), RocketForge::DefaultExampleBotRoot(), botsRoot, {}, out);

	EXPECT_EQ(exitCode, 0);
	EXPECT_TRUE(fs::is_directory(botsRoot / "MyBot"));

	const std::string text = out.str();
	const size_t creating = text.find("Creating Bots/MyBot from Bots/ExampleBot");
	const size_t created = text.find("Created ");
	ASSERT_NE(creating, std::string::npos);
	ASSERT_NE(created, std::string::npos);
	EXPECT_LT(creating, created);
	EXPECT_NE(text.find("train --agent MyBot"), std::string::npos);
}

TEST(CreateBotCommand, IsRegisteredInBotsGroup)
{
	const std::vector<CliCommand>& commands = AllCommands();
	const auto found =
		std::find_if(commands.begin(), commands.end(), [](const CliCommand& command) { return command.name == "create-bot"; });

	ASSERT_NE(found, commands.end());
	EXPECT_EQ(found->group, "Bots");
}
