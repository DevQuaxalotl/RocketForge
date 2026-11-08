#include <gtest/gtest.h>

#include "TestSupport.h"

#include <Utils/BotScaffold.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace RocketForge;

TEST(BotScaffold, MakeBotNameFormsDerivesSpellings)
{
	BotNameForms forms = MakeBotNameForms("MyBot");

	ASSERT_EQ(forms.asGiven, "MyBot");
	ASSERT_EQ(forms.lower, "mybot");
	ASSERT_EQ(forms.upper, "MYBOT");
	ASSERT_EQ(forms.AgentName(), "MyBot");
}

TEST(BotScaffold, ValidateAcceptsNewName)
{
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_accepts");

	ASSERT_EQ(ValidateNewBotName("MyBot", botsRoot, {}), "");
}

TEST(BotScaffold, ValidateAcceptsMissingBotsRoot)
{
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_missing_root") / "NoSuchFolder";

	ASSERT_EQ(ValidateNewBotName("MyBot", botsRoot, {}), "");
}

TEST(BotScaffold, ValidateRejectsBadNames)
{
	struct Case {
		std::string label;
		std::string name;
		std::string expected;
	};

	const std::string lettersAndDigits = "must start with a letter and contain only letters and digits";
	const Case cases[] = {
		{ "empty", "", "a bot name is required" },
		{ "leading space", " MyBot", "' MyBot' " + lettersAndDigits },
		{ "inner space", "Sky Hawk", "'Sky Hawk' " + lettersAndDigits },
		{ "hyphen", "Sky-Hawk", "'Sky-Hawk' " + lettersAndDigits },
		{ "leading digit", "9Lives", "'9Lives' " + lettersAndDigits },
		{ "non-ascii", "Sk\xC3\xBF", "'Sk\xC3\xBF' " + lettersAndDigits },
		{ "too long", std::string(33, 'A'), "'" + std::string(33, 'A') + "' is longer than 32 characters" },
		{ "template name", "Example", "'Example' is the template bot's name; choose another" },
		{ "template name upper", "EXAMPLE", "'EXAMPLE' is the template bot's name; choose another" },
	};

	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_bad_names");

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		ASSERT_EQ(ValidateNewBotName(testCase.name, botsRoot, {}), testCase.expected);
	}
}

TEST(BotScaffold, ValidateRejectsCollisions)
{
	struct Case {
		std::string label;
		std::string existingDir;
		std::vector<std::string> registeredAgents;
		std::string name;
		std::string expected;
	};

	const Case cases[] = {
		{ "bot folder, other case", "Falcon", {}, "falcon", "Bots/Falcon already exists" },
		{ "bot folder wins over target dir", "Taken/TakenAgent", {}, "taken", "Bots/Taken already exists" },
		{ "target dir in another bot", "Odd/HawkAgent", {}, "Hawk", "Bots/Odd/HawkAgent already defines the HawkAgent target" },
		{ "training plan dir in another bot", "Odd/hawktrainingplan", {}, "Hawk",
		  "Bots/Odd/hawktrainingplan already defines the hawktrainingplan target" },
		{ "registered agent, other case", "", { "hawk" }, "Hawk", "the agent 'Hawk' is already registered" },
		{ "different registered agent", "", { "hawk-agent" }, "Hawk", "" },
		{ "staging folder", "Hawk.creating", {}, "Hawk", "" },
		{ "staging folder target dir", "Hawk.creating/HawkAgent", {}, "Hawk", "" },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_collisions");
		fs::remove_all(botsRoot);
		fs::create_directories(botsRoot);

		if (!testCase.existingDir.empty())
			fs::create_directories(botsRoot / testCase.existingDir);

		ASSERT_EQ(ValidateNewBotName(testCase.name, botsRoot, testCase.registeredAgents), testCase.expected);
	}
}

TEST(BotScaffold, ValidateFailsWhenBotsRootIsNotReadable)
{
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_root_is_file") / "Bots";
	std::ofstream(botsRoot).put('x');

	std::string message = ValidateNewBotName("Hawk", botsRoot, {});

	ASSERT_EQ(message.rfind("cannot read " + botsRoot.generic_string() + ": ", 0), 0u) << message;
}

TEST(BotScaffold, ValidateIgnoresFilesInBotsRoot)
{
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_files");
	std::ofstream(botsRoot / "Hawk").put('x');

	ASSERT_EQ(ValidateNewBotName("Hawk", botsRoot, {}), "");
}

TEST(BotScaffold, DefaultBotsRootIsProjectBots)
{
	ASSERT_EQ(DefaultBotsRoot().filename(), "Bots");
	ASSERT_TRUE(fs::is_directory(DefaultBotsRoot()));
}

namespace
{
	std::string ReadText(const fs::path& path)
	{
		std::ifstream file(path, std::ios::binary);
		return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
	}

	void WriteText(const fs::path& path, const std::string& text)
	{
		fs::create_directories(path.parent_path());
		std::ofstream file(path, std::ios::binary);
		file << text;
	}

	std::string Lowercase(std::string text)
	{
		for (char& c : text)
			c = char(std::tolower((unsigned char)c));

		return text;
	}

	// A template with only the files a new bot copies, so tests can break it without copying Models/.
	fs::path MakeTemplateCopy(const std::string& label, const fs::path& skippedFile = "")
	{
		fs::path templateRoot = RocketForgeTests::TestTempDir(label) / "ExampleBot";
		fs::path realRoot = DefaultExampleBotRoot();

		for (const fs::path& relativePath : ExampleBotCopyList()) {
			if (relativePath == skippedFile)
				continue;

			WriteText(templateRoot / relativePath, ReadText(realRoot / relativePath));
		}

		return templateRoot;
	}

	std::vector<std::string> ListRelativeFiles(const fs::path& root)
	{
		std::vector<std::string> files;
		for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
			if (entry.is_regular_file())
				files.push_back(fs::relative(entry.path(), root).generic_string());
		}

		std::sort(files.begin(), files.end());
		return files;
	}

	bool IsEmptyDirectory(const fs::path& path)
	{
		return fs::is_directory(path) && fs::directory_iterator(path) == fs::directory_iterator();
	}
}

TEST(BotScaffold, ReplacementIsSinglePass)
{
	ASSERT_EQ(ApplyBotNameReplacements("ExampleTrainingAgent example-agent EXAMPLE_AGENT ExampleBot", MakeBotNameForms("Reexample")),
		"ReexampleTrainingAgent Reexample REEXAMPLE_AGENT Reexample");
	ASSERT_EQ(ApplyBotNameReplacements("Example", MakeBotNameForms("MyExampleBot")), "MyExampleBot");
}

TEST(BotScaffold, ExampleBotFilesAreAllClassified)
{
	fs::path root = DefaultExampleBotRoot();
	ASSERT_TRUE(fs::is_directory(root));

	const std::vector<fs::path>& copyList = ExampleBotCopyList();
	for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
		if (!entry.is_regular_file())
			continue;

		fs::path relativePath = fs::relative(entry.path(), root);
		bool listed = std::find(copyList.begin(), copyList.end(), relativePath) != copyList.end();
		ASSERT_TRUE(listed || IsNeverCopiedExampleBotFile(relativePath))
			<< "unclassified ExampleBot file: " << relativePath.generic_string();
	}

	for (const fs::path& relativePath : copyList)
		ASSERT_TRUE(fs::is_regular_file(root / relativePath)) << "missing copy-list file: " << relativePath.generic_string();
}

TEST(BotScaffold, CreateCopiesRenamedTemplate)
{
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_create");
	CreateBotResult result = CreateBotFromExample("MyBot", DefaultExampleBotRoot(), botsRoot, {});

	ASSERT_TRUE(result.Ok()) << result.error;
	ASSERT_EQ(result.botFolder, botsRoot / "MyBot");

	const std::vector<std::string> expectedFiles = {
		"CMakeLists.txt",
		"MyBotAgent/CMakeLists.txt",
		"MyBotAgent/forge_settings.example.json",
		"MyBotAgent/rlbot/bot.toml",
		"MyBotAgent/rlbot/device.txt",
		"MyBotAgent/rlbot/loadout.toml",
		"MyBotAgent/rlbot/run.bat",
		"MyBotAgent/src/MyBotAgent.cpp",
		"MyBotAgent/src/MyBotAgent.h",
		"MyBotTrainingPlan/CMakeLists.txt",
		"MyBotTrainingPlan/src/MyBotTrainingPlan.cpp",
		"MyBotTrainingPlan/src/MyBotTrainingPlan.h",
	};
	std::vector<std::string> actualFiles = ListRelativeFiles(result.botFolder);
	ASSERT_EQ(actualFiles, expectedFiles);

	for (const std::string& relativePath : actualFiles)
		ASSERT_EQ(Lowercase(ReadText(result.botFolder / relativePath)).find("example"), std::string::npos) << relativePath;

	std::string topCMake = ReadText(result.botFolder / "CMakeLists.txt");
	ASSERT_NE(topCMake.find("project(MyBot)"), std::string::npos);
	ASSERT_NE(topCMake.find("add_subdirectory(MyBotAgent)"), std::string::npos);

	std::string agentSource = ReadText(result.botFolder / "MyBotAgent/src/MyBotAgent.cpp");
	ASSERT_NE(agentSource.find("REGISTER_AGENT(\"MyBot\", MyBotTrainingAgent)"), std::string::npos);
	ASSERT_NE(agentSource.find("agentName = \"MyBot\";"), std::string::npos);
	ASSERT_NE(ReadText(result.botFolder / "MyBotAgent/forge_settings.example.json").find("\"agentName\": \"MyBot\""), std::string::npos);
	ASSERT_NE(ReadText(result.botFolder / "MyBotAgent/rlbot/run.bat").find("set \"AGENT=MyBot\""), std::string::npos);
	ASSERT_NE(agentSource.find("MYBOT_AGENT_MODELS_PATH"), std::string::npos);

	std::string botToml = ReadText(result.botFolder / "MyBotAgent/rlbot/bot.toml");
	ASSERT_NE(botToml.find("agent_id = \"RocketForge/MyBot\""), std::string::npos);
	ASSERT_NE(botToml.find("name = \"MyBot\""), std::string::npos);
	ASSERT_EQ(botToml.find("RFBot"), std::string::npos);

	ASSERT_FALSE(fs::exists(botsRoot / "MyBot.creating"));
}

TEST(BotScaffold, CreateRefusesInvalidNameWithoutWriting)
{
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_create_invalid");
	CreateBotResult result = CreateBotFromExample("Sky-Hawk", DefaultExampleBotRoot(), botsRoot, {});

	ASSERT_FALSE(result.Ok());
	ASSERT_TRUE(IsEmptyDirectory(botsRoot));
}

TEST(BotScaffold, CreateFailsCleanlyWhenTemplateIncomplete)
{
	fs::path templateRoot = MakeTemplateCopy("bot_scaffold_incomplete", "ExampleAgent/src/ExampleAgent.h");
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_incomplete_bots");

	CreateBotResult result = CreateBotFromExample("MyBot", templateRoot, botsRoot, {});

	ASSERT_EQ(result.error, "ExampleBot is missing ExampleAgent/src/ExampleAgent.h");
	ASSERT_FALSE(fs::exists(botsRoot / "MyBot"));
	ASSERT_FALSE(fs::exists(botsRoot / "MyBot.creating"));
}

TEST(BotScaffold, CreateFailsWhenBotTomlLinesMissing)
{
	fs::path templateRoot = MakeTemplateCopy("bot_scaffold_toml");
	fs::path botTomlPath = templateRoot / "ExampleAgent/rlbot/bot.toml";
	std::string botToml = ReadText(botTomlPath);
	const std::string agentIdLine = "agent_id = \"RocketForge/RFBot\"";
	size_t position = botToml.find(agentIdLine);
	ASSERT_NE(position, std::string::npos);
	botToml.replace(position, agentIdLine.size(), "agent_id = \"Someone/Else\"");
	WriteText(botTomlPath, botToml);

	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_toml_bots");
	CreateBotResult result = CreateBotFromExample("MyBot", templateRoot, botsRoot, {});

	ASSERT_EQ(result.error, "ExampleBot's rlbot/bot.toml no longer has the line 'agent_id = \"RocketForge/RFBot\"'");
	ASSERT_TRUE(IsEmptyDirectory(botsRoot));
}

TEST(BotScaffold, CreateReplacesStaleStagingFolder)
{
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_stale");
	WriteText(botsRoot / "MyBot.creating/junk.txt", "stale");

	CreateBotResult result = CreateBotFromExample("MyBot", DefaultExampleBotRoot(), botsRoot, {});

	ASSERT_TRUE(result.Ok()) << result.error;
	ASSERT_FALSE(fs::exists(result.botFolder / "junk.txt"));
	ASSERT_FALSE(fs::exists(botsRoot / "MyBot.creating"));
}

TEST(BotScaffold, CreateFailsCleanlyForReservedDeviceName)
{
	fs::path botsRoot = RocketForgeTests::TestTempDir("bot_scaffold_reserved");
	CreateBotResult result = CreateBotFromExample("NUL", DefaultExampleBotRoot(), botsRoot, {});

#ifdef _WIN32
	ASSERT_FALSE(result.Ok());
#endif
	if (!result.Ok())
		ASSERT_TRUE(IsEmptyDirectory(botsRoot));
}
