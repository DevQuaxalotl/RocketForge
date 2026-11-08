#include "BotScaffold.h"

#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace
{
	constexpr size_t MAX_BOT_NAME_LENGTH = 32;
	const std::string TEMPLATE_BOT_NAME = "example";
	const std::string STAGING_SUFFIX = ".creating";
	const std::string EXAMPLE_AGENT_DIR = "ExampleAgent";
	const std::string TOP_CMAKE_FILE = "CMakeLists.txt";
	const std::string KEPT_NAME_FILE = "forge_settings.example.json";
	const fs::path BOT_TOML_PATH = "ExampleAgent/rlbot/bot.toml";

	char AsciiLower(char c)
	{
		return (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
	}

	char AsciiUpper(char c)
	{
		return (c >= 'a' && c <= 'z') ? char(c - 'a' + 'A') : c;
	}

	bool IsAsciiLetter(char c)
	{
		return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
	}

	bool IsAsciiDigit(char c)
	{
		return c >= '0' && c <= '9';
	}

	std::string ToLower(std::string text)
	{
		for (char& c : text)
			c = AsciiLower(c);

		return text;
	}

	std::string ToUpper(std::string text)
	{
		for (char& c : text)
			c = AsciiUpper(c);

		return text;
	}

	bool EqualsIgnoreCase(const std::string& a, const std::string& b)
	{
		return ToLower(a) == ToLower(b);
	}

	bool HasValidCharacters(const std::string& name)
	{
		if (!IsAsciiLetter(name[0]))
			return false;

		for (char c : name) {
			if (!IsAsciiLetter(c) && !IsAsciiDigit(c))
				return false;
		}

		return true;
	}

	bool EndsWith(const std::string& text, const std::string& suffix)
	{
		return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
	}

	/**
	 * Fills `directories` with the directories directly under `parent`, skipping the staging folders a scaffold in progress leaves behind.
	 * A missing `parent` counts as empty; any other failure returns a sentence saying what could not be read, and the list is incomplete.
	 */
	std::string ListChildDirectories(const fs::path& parent, std::vector<fs::path>& directories)
	{
		std::error_code ec;
		fs::directory_iterator it(parent, ec);
		fs::directory_iterator end;

		for (; !ec && it != end; it.increment(ec)) {
			std::error_code statusEc;
			bool isDirectory = it->is_directory(statusEc);
			if (statusEc && statusEc != std::errc::no_such_file_or_directory)
				return "cannot read " + it->path().generic_string() + ": " + statusEc.message();

			if (!isDirectory || EndsWith(it->path().filename().string(), STAGING_SUFFIX))
				continue;

			directories.push_back(it->path());
		}

		if (ec && ec != std::errc::no_such_file_or_directory)
			return "cannot read " + parent.generic_string() + ": " + ec.message();

		return "";
	}

	/** Reads a whole file in binary so line endings and BOMs survive. */
	std::string ReadBinaryFile(const fs::path& path, std::string& error)
	{
		std::ifstream file(path, std::ios::binary);
		std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		if (file.bad() || (!file.is_open()))
			error = "cannot read " + path.generic_string();

		return contents;
	}

	std::string WriteBinaryFile(const fs::path& path, const std::string& contents)
	{
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
		file.write(contents.data(), std::streamsize(contents.size()));
		file.close();
		if (!file)
			return "cannot write " + path.generic_string();

		return "";
	}

	/** Replaces the whole line `oldLine` with `newLine`; false when no line matches exactly. */
	bool ReplaceLine(std::string& text, const std::string& oldLine, const std::string& newLine)
	{
		size_t position = 0;
		while ((position = text.find(oldLine, position)) != std::string::npos) {
			size_t end = position + oldLine.size();
			bool startsLine = position == 0 || text[position - 1] == '\n';
			bool endsLine = end == text.size() || text[end] == '\n' || text[end] == '\r';
			if (startsLine && endsLine) {
				text.replace(position, oldLine.size(), newLine);
				return true;
			}

			position = end;
		}

		return false;
	}

	/**
	 * Points a copied bot.toml at the new bot.
	 * Its agent_id and name use the RFBot placeholder, which the token replacement does not touch.
	 */
	std::string RenameBotTomlIdentity(std::string& text, const std::string& botName)
	{
		const std::pair<std::string, std::string> lines[] = {
			{ "agent_id = \"RocketForge/RFBot\"", "agent_id = \"RocketForge/" + botName + "\"" },
			{ "name = \"RFBot\"", "name = \"" + botName + "\"" },
		};

		for (const auto& [oldLine, newLine] : lines) {
			if (!ReplaceLine(text, oldLine, newLine))
				return "ExampleBot's rlbot/bot.toml no longer has the line '" + oldLine + "'";
		}

		return "";
	}

	/** The copied file's path under the new bot: every component renamed, except the settings template whose name is fixed. */
	fs::path RenamedRelativePath(const fs::path& relativePath, const RocketForge::BotNameForms& forms)
	{
		fs::path renamed;
		for (const fs::path& part : relativePath) {
			if (part == KEPT_NAME_FILE)
				renamed /= part;
			else
				renamed /= RocketForge::ApplyBotNameReplacements(part.string(), forms);
		}

		return renamed;
	}

	/** Writes the renamed copy list into `stagingFolder`; returns an error sentence, empty on success. */
	std::string WriteStagedBot(const RocketForge::BotNameForms& forms, const fs::path& exampleBotRoot, const fs::path& stagingFolder)
	{
		for (const fs::path& relativePath : RocketForge::ExampleBotCopyList()) {
			std::error_code ec;
			if (!fs::is_regular_file(exampleBotRoot / relativePath, ec))
				return "ExampleBot is missing " + relativePath.generic_string();

			std::string error;
			std::string contents = ReadBinaryFile(exampleBotRoot / relativePath, error);
			if (!error.empty())
				return error;

			contents = RocketForge::ApplyBotNameReplacements(contents, forms);
			if (relativePath == BOT_TOML_PATH) {
				error = RenameBotTomlIdentity(contents, forms.asGiven);
				if (!error.empty())
					return error;
			}

			fs::path destination = stagingFolder / RenamedRelativePath(relativePath, forms);
			if (relativePath == TOP_CMAKE_FILE)
				destination += STAGING_SUFFIX;

			fs::create_directories(destination.parent_path(), ec);
			if (ec)
				return "cannot create " + destination.parent_path().generic_string() + ": " + ec.message();

			error = WriteBinaryFile(destination, contents);
			if (!error.empty())
				return error;
		}

		return "";
	}
}

std::string RocketForge::BotNameForms::AgentName() const
{
	return asGiven;
}

RocketForge::BotNameForms RocketForge::MakeBotNameForms(const std::string& name)
{
	BotNameForms forms;
	forms.asGiven = name;
	forms.lower = ToLower(name);
	forms.upper = ToUpper(name);

	return forms;
}

std::string RocketForge::ValidateNewBotName(
	const std::string& name,
	const fs::path& botsRoot,
	const std::vector<std::string>& registeredAgents)
{
	if (name.empty())
		return "a bot name is required";

	if (name.size() > MAX_BOT_NAME_LENGTH)
		return "'" + name + "' is longer than " + std::to_string(MAX_BOT_NAME_LENGTH) + " characters";

	if (!HasValidCharacters(name))
		return "'" + name + "' must start with a letter and contain only letters and digits";

	BotNameForms forms = MakeBotNameForms(name);
	if (forms.lower == TEMPLATE_BOT_NAME)
		return "'" + name + "' is the template bot's name; choose another";

	std::vector<fs::path> botFolders;
	std::string readError = ListChildDirectories(botsRoot, botFolders);
	if (!readError.empty())
		return readError;

	for (const fs::path& botFolder : botFolders) {
		std::string existingName = botFolder.filename().string();
		if (EqualsIgnoreCase(existingName, name))
			return "Bots/" + existingName + " already exists";
	}

	// The Agent and TrainingPlan folders each register a target named after the folder.
	const std::string agentDir = name + "Agent";
	const std::string trainingPlanDir = name + "TrainingPlan";

	for (const fs::path& botFolder : botFolders) {
		std::vector<fs::path> targetFolders;
		readError = ListChildDirectories(botFolder, targetFolders);
		if (!readError.empty())
			return readError;

		for (const fs::path& targetFolder : targetFolders) {
			std::string existingDir = targetFolder.filename().string();
			if (EqualsIgnoreCase(existingDir, agentDir) || EqualsIgnoreCase(existingDir, trainingPlanDir))
				return "Bots/" + botFolder.filename().string() + "/" + existingDir + " already defines the " + existingDir + " target";
		}
	}

	const std::string agentName = forms.AgentName();
	for (const std::string& registeredAgent : registeredAgents) {
		if (EqualsIgnoreCase(registeredAgent, agentName))
			return "the agent '" + agentName + "' is already registered";
	}

	return "";
}

fs::path RocketForge::DefaultBotsRoot()
{
	return fs::path(PROJECT_ROOT) / "Bots";
}

fs::path RocketForge::DefaultExampleBotRoot()
{
	return DefaultBotsRoot() / "ExampleBot";
}

const std::vector<fs::path>& RocketForge::ExampleBotCopyList()
{
	static const std::vector<fs::path> copyList = {
		"CMakeLists.txt",
		"ExampleAgent/CMakeLists.txt",
		"ExampleAgent/forge_settings.example.json",
		"ExampleAgent/rlbot/bot.toml",
		"ExampleAgent/rlbot/device.txt",
		"ExampleAgent/rlbot/loadout.toml",
		"ExampleAgent/rlbot/run.bat",
		"ExampleAgent/src/ExampleAgent.cpp",
		"ExampleAgent/src/ExampleAgent.h",
		"ExampleTrainingPlan/CMakeLists.txt",
		"ExampleTrainingPlan/src/ExampleTrainingPlan.cpp",
		"ExampleTrainingPlan/src/ExampleTrainingPlan.h",
	};

	return copyList;
}

bool RocketForge::IsNeverCopiedExampleBotFile(const fs::path& relativePath)
{
	std::vector<std::string> parts;
	for (const fs::path& part : relativePath)
		parts.push_back(part.string());

	if (parts.size() < 2 || parts[0] != EXAMPLE_AGENT_DIR)
		return false;

	if (parts[1] == "Models" && parts.size() > 2)
		return true;

	if (parts.size() == 2)
		return parts[1] == "forge_settings.json" || parts[1] == "forge_debug_settings.json";

	if (parts.size() == 3 && parts[1] == "rlbot") {
		std::string extension = ToLower(relativePath.extension().string());
		return extension == ".exe" || extension == ".dll";
	}

	return false;
}

std::string RocketForge::ApplyBotNameReplacements(const std::string& text, const BotNameForms& forms)
{
	// Longest token first, so ExampleBot is not consumed as Example followed by a stray "Bot",
	// and the agent name becomes the bot name rather than "<lower>-agent".
	const std::string agentName = forms.AgentName();
	const struct {
		std::string token;
		const std::string& replacement;
	} replacements[] = {
		{ "example-agent", agentName },
		{ "ExampleBot", forms.asGiven },
		{ "Example", forms.asGiven },
		{ "example", forms.lower },
		{ "EXAMPLE", forms.upper },
	};

	std::string result;
	result.reserve(text.size());

	size_t position = 0;
	while (position < text.size()) {
		bool replaced = false;
		for (const auto& replacement : replacements) {
			if (text.compare(position, replacement.token.size(), replacement.token) == 0) {
				result += replacement.replacement;
				position += replacement.token.size();
				replaced = true;
				break;
			}
		}

		if (!replaced)
			result += text[position++];
	}

	return result;
}

RocketForge::CreateBotResult RocketForge::CreateBotFromExample(
	const std::string& name,
	const fs::path& exampleBotRoot,
	const fs::path& botsRoot,
	const std::vector<std::string>& registeredAgents)
{
	CreateBotResult result;

	result.error = ValidateNewBotName(name, botsRoot, registeredAgents);
	if (!result.error.empty())
		return result;

	BotNameForms forms = MakeBotNameForms(name);
	const fs::path finalFolder = botsRoot / forms.asGiven;
	const fs::path stagingFolder = botsRoot / (forms.asGiven + STAGING_SUFFIX);
	bool finalFolderCreated = false;

	try {
		std::error_code ec;
		fs::remove_all(stagingFolder, ec);
		if (ec) {
			result.error = "cannot clear " + stagingFolder.generic_string() + ": " + ec.message();
			return result;
		}

		result.error = WriteStagedBot(forms, exampleBotRoot, stagingFolder);

		if (result.error.empty()) {
			fs::rename(stagingFolder, finalFolder, ec);
			if (ec)
				result.error =
					"cannot rename " + stagingFolder.generic_string() + " to " + finalFolder.generic_string() + ": " + ec.message();
			else
				finalFolderCreated = true;
		}

		if (result.error.empty()) {
			fs::rename(finalFolder / (TOP_CMAKE_FILE + STAGING_SUFFIX), finalFolder / TOP_CMAKE_FILE, ec);
			if (ec)
				result.error = "cannot finish " + finalFolder.generic_string() + ": " + ec.message();
		}
	}
	catch (const std::exception& exception) {
		result.error = "cannot create " + finalFolder.generic_string() + ": " + exception.what();
	}

	if (!result.error.empty()) {
		// Best effort: the failure being reported is the original one.
		std::error_code cleanupEc;
		fs::remove_all(finalFolderCreated ? finalFolder : stagingFolder, cleanupEc);
		return result;
	}

	result.botFolder = finalFolder;
	return result;
}
