#include "CommandTable.h"

#include "CliErrors.h"
#include "CliResolve.h"

#include "Commands/Commands.h"

#include <RocketSim/Engine.h>

#include <algorithm>
#include <iostream>

using namespace std;
using namespace RocketForge::Cli;

namespace {

	const char* const PROGRAM_NAME = "ForgeConsoleTrainer";
	const char* const DEFAULT_COMMAND_NAME = "train";
	const char* const HELP_COMMAND_NAME = "help";
	const char* const HELP_SWITCH = "--help";

	bool IsOptionToken(const string& token)
	{
		return token.rfind("--", 0) == 0;
	}

	const CliCommand* FindCommand(const string& name)
	{
		const vector<CliCommand>& commands = AllCommands();
		const auto found = std::find_if(commands.begin(), commands.end(), [&](const CliCommand& command) { return command.name == name; });
		return found == commands.end() ? nullptr : &*found;
	}

	/** Throws CliUsageError naming the closest commands. */
	[[noreturn]] void ThrowUnknownCommand(const string& name)
	{
		vector<string> names = { HELP_COMMAND_NAME };
		for (const CliCommand& command : AllCommands())
			names.push_back(command.name);

		RequireKnownName("command", name, names, HELP_SWITCH);
		throw CliUsageError("unknown command '" + name + "'");
	}

	int PrintHelp(const vector<string>& topics)
	{
		try {
			if (topics.size() > 1)
				throw CliUsageError("unexpected argument '" + topics[1] + "'");

			if (topics.empty() || topics[0] == HELP_COMMAND_NAME) {
				cout << FormatGlobalHelp();
				return 0;
			}

			const CliCommand* command = FindCommand(topics[0]);
			if (command == nullptr)
				ThrowUnknownCommand(topics[0]);

			cout << command->usage;
			return 0;
		} catch (const CliUsageError& error) {
			cerr << HELP_COMMAND_NAME << ": " << error.what() << "\n";
			return 2;
		}
	}

	int RunCommand(const CliCommand& command, const vector<string>& arguments)
	{
		try {
			return command.run(CliArgs::Parse(arguments, command.options));
		} catch (const CliUsageError& error) {
			cerr << command.name << ": " << error.what() << "\n";
			return 2;
		} catch (const RocketSim::EngineUnavailableError&) {
			// main reports a missing simulation engine in its own format.
			throw;
		} catch (const std::exception& error) {
			cerr << command.name << ": " << error.what() << "\n";
			return 1;
		}
	}

}

const vector<CliCommand>& RocketForge::Cli::AllCommands()
{
	static const vector<CliCommand> commands = [] {
		vector<CliCommand> table = BotCommands();
		for (const CliCommand& command : TrainingCommands())
			table.push_back(command);
		for (const CliCommand& command : InspectCommands())
			table.push_back(command);
		for (const CliCommand& command : ModelCommands())
			table.push_back(command);
		for (const CliCommand& command : DebugCommands())
			table.push_back(command);
		for (const CliCommand& command : AnalysisCommands())
			table.push_back(command);
		for (const CliCommand& command : BenchmarkCommands())
			table.push_back(command);
		for (const CliCommand& command : PlumbingCommands())
			table.push_back(command);

		return table;
	}();

	return commands;
}

string RocketForge::Cli::FormatGlobalHelp()
{
	const string helpUsage = string(HELP_COMMAND_NAME) + " [command]";

	size_t nameWidth = helpUsage.size();
	for (const CliCommand& command : AllCommands())
		nameWidth = std::max(nameWidth, command.name.size());

	const auto formatLine = [nameWidth](const string& name, const string& summary) {
		return "  " + name + string(nameWidth - name.size() + 2, ' ') + summary + "\n";
	};

	string help = string("Usage: ") + PROGRAM_NAME + " <command> [options]\n"
		+ "With no command, or when the first argument is an option, runs `" + DEFAULT_COMMAND_NAME + "`.\n";

	for (const string& group : COMMAND_GROUPS) {
		help += "\n" + group + ":\n";

		for (const CliCommand& command : AllCommands()) {
			if (command.group == group)
				help += formatLine(command.name, command.summary);
		}
	}

	help += "\nHelp:\n";
	help += formatLine(helpUsage, "Lists the commands, or prints one command's options (also: <command> --help)");

	return help;
}

int RocketForge::Cli::RunCli(int argc, char* argv[])
{
	const vector<string> arguments(argv + std::min(argc, 1), argv + argc);

	// A registered name wins over the option-means-train rule, so commands may be named like options.
	string commandName = DEFAULT_COMMAND_NAME;
	vector<string> commandArguments = arguments;

	if (!arguments.empty() && FindCommand(arguments[0]) != nullptr) {
		commandName = arguments[0];
		commandArguments.erase(commandArguments.begin());
	} else if (!arguments.empty() && (arguments[0] == HELP_COMMAND_NAME || arguments[0] == HELP_SWITCH)) {
		return PrintHelp(vector<string>(arguments.begin() + 1, arguments.end()));
	} else if (!arguments.empty() && !IsOptionToken(arguments[0])) {
		try {
			ThrowUnknownCommand(arguments[0]);
		} catch (const CliUsageError& error) {
			cerr << PROGRAM_NAME << ": " << error.what() << "\n";
			return 2;
		}
	}

	const CliCommand* command = FindCommand(commandName);
	if (command == nullptr) {
		cerr << PROGRAM_NAME << ": no '" << commandName << "' command is registered\n";
		return 1;
	}

	if (std::find(commandArguments.begin(), commandArguments.end(), HELP_SWITCH) != commandArguments.end()) {
		cout << command->usage;
		return 0;
	}

	return RunCommand(*command, commandArguments);
}
