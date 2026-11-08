#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace RocketForge
{
	/** A bot name in the three spellings ExampleBot uses: `Example`, `example`, `EXAMPLE`. */
	struct BotNameForms {
		std::string asGiven;
		std::string lower;
		std::string upper;

		/** The registered agent name: the bot name as given, e.g. "MyBot". */
		std::string AgentName() const;
	};

	BotNameForms MakeBotNameForms(const std::string& name);

	/**
	 * Empty when `name` can be created under botsRoot; otherwise one sentence saying why not.
	 * registeredAgents is normally AgentRegistry::GetAvailableAgents().
	 */
	std::string ValidateNewBotName(
		const std::string& name,
		const std::filesystem::path& botsRoot,
		const std::vector<std::string>& registeredAgents);

	/** PROJECT_ROOT/Bots. */
	std::filesystem::path DefaultBotsRoot();

	/** DefaultBotsRoot()/ExampleBot. */
	std::filesystem::path DefaultExampleBotRoot();

	/** ExampleBot files a new bot gets, relative to the ExampleBot folder. */
	const std::vector<std::filesystem::path>& ExampleBotCopyList();

	/** True for ExampleBot files a new bot must never get: local training state and build outputs. */
	bool IsNeverCopiedExampleBotFile(const std::filesystem::path& relativePath);

	/**
	 * One left-to-right pass, longest token first at each position:
	 * example-agent -> AgentName(), ExampleBot -> asGiven, Example -> asGiven, example -> lower, EXAMPLE -> upper.
	 */
	std::string ApplyBotNameReplacements(const std::string& text, const BotNameForms& forms);

	struct CreateBotResult {
		std::filesystem::path botFolder;
		std::string error;

		bool Ok() const { return error.empty(); }
	};

	/**
	 * Validates, then copies ExampleBotCopyList() from exampleBotRoot into botsRoot/<name>, renamed.
	 * On failure nothing is left under botsRoot, including the staging folder.
	 */
	CreateBotResult CreateBotFromExample(
		const std::string& name,
		const std::filesystem::path& exampleBotRoot,
		const std::filesystem::path& botsRoot,
		const std::vector<std::string>& registeredAgents);
}
