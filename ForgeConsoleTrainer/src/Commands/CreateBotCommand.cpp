#include "Commands.h"

#include "Cli/CliErrors.h"

#include "Utils/BotScaffold.h"

#include <GigaLearnInfer/AgentRegistry.h>

#include <iostream>

using namespace std;
using namespace RocketForge;
using namespace RocketForge::Cli;

namespace fs = std::filesystem;

namespace {

	const char* const CREATE_BOT_USAGE =
		"Usage: ForgeConsoleTrainer create-bot <Name>\n"
		"\n"
		"Creates Bots/<Name> from ExampleBot: the agent <Name> and its stages\n"
		"<name>-ball-chase, <name>-basic-score and <name>-advanced, renamed throughout.\n"
		"Training state, models and RLBot binaries are not copied.\n"
		"\n"
		"<Name> is letters and digits, starting with a letter (at most 32 characters),\n"
		"and must not match an existing bot folder or agent.\n"
		"\n"
		"Afterwards re-configure (cmake --preset forge) and rebuild before using the new agent.\n";

}

int RocketForge::Cli::RunCreateBotCommand(const CliArgs& args)
{
	return RunCreateBotCommand(args, DefaultExampleBotRoot(), DefaultBotsRoot(), AgentRegistry::GetAvailableAgents(), cout);
}

int RocketForge::Cli::RunCreateBotCommand(
	const CliArgs& args,
	const fs::path& exampleBotRoot,
	const fs::path& botsRoot,
	const vector<string>& registeredAgents,
	ostream& out)
{
	if (args.Positionals().empty())
		throw CliUsageError("a bot name is required: ForgeConsoleTrainer create-bot <Name>");

	const string& name = args.Positionals()[0];

	// Validated here so a bad name is a usage error; CreateBotFromExample's own errors are then runtime failures.
	const string invalid = ValidateNewBotName(name, botsRoot, registeredAgents);
	if (!invalid.empty())
		throw CliUsageError(invalid);

	out << "Creating Bots/" << name << " from Bots/ExampleBot\n";

	const CreateBotResult result = CreateBotFromExample(name, exampleBotRoot, botsRoot, registeredAgents);
	if (!result.Ok())
		throw CliRuntimeError(result.error);

	out << "Created " << result.botFolder.string() << "\n";
	out << "Next: re-configure (cmake --preset forge), rebuild, then run\n";
	out << "  ForgeConsoleTrainer train --agent " << MakeBotNameForms(name).AgentName() << "\n";

	return 0;
}

vector<CliCommand> RocketForge::Cli::BotCommands()
{
	CliCommand createBot;
	createBot.name = "create-bot";
	createBot.group = "Bots";
	createBot.summary = "Creates a new bot from ExampleBot";
	createBot.usage = CREATE_BOT_USAGE;
	createBot.options.maxPositionals = 1;
	createBot.run = [](const CliArgs& args) { return RunCreateBotCommand(args); };

	return { createBot };
}
