#pragma once

#include "CliArgs.h"

#include <functional>
#include <string>
#include <vector>

namespace RocketForge::Cli {

/** One `ForgeConsoleTrainer <name> [options]` command. */
struct CliCommand {
	std::string name;

	/** One of COMMAND_GROUPS; --help lists commands under their group's heading. */
	std::string group;

	/** One line for the global --help list. */
	std::string summary;

	/** Full help text printed by `help <name>` and `<name> --help`. */
	std::string usage;

	CliOptionSpec options;
	std::function<int(const CliArgs&)> run;
};

/** Group headings in --help order. */
inline const std::vector<std::string> COMMAND_GROUPS = { "Bots", "Training", "Inspect", "Models", "Debug & analysis", "Benchmarking", "Script plumbing" };

/** Every registered command, built once from the Commands.h groups. */
const std::vector<CliCommand>& AllCommands();

/**
 * Dispatches argv to its command and returns the process exit code.
 * No arguments, or a first argument that is an option and not a command name, runs `train`.
 * Usage errors print `<command>: <message>` and return 2; runtime failures do the same and return 1.
 */
int RunCli(int argc, char* argv[]);

std::string FormatGlobalHelp();

}
