#pragma once

#include "Cli/CommandTable.h"

#include "Settings/ForgeDebugSettings.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace RocketForge {
	struct WeightedTeamSize;
}

namespace RocketForge::Cli {

/** The Bots group: `create-bot`. */
std::vector<CliCommand> BotCommands();

/** `create-bot <Name>`: copies ExampleBot into Bots/<Name>. Exit 2 for a missing or invalid name. */
int RunCreateBotCommand(const CliArgs& args);
int RunCreateBotCommand(
	const CliArgs& args,
	const std::filesystem::path& exampleBotRoot,
	const std::filesystem::path& botsRoot,
	const std::vector<std::string>& registeredAgents,
	std::ostream& out);

/** The Training group: `train`. */
std::vector<CliCommand> TrainingCommands();

/** The Inspect group: `list`. */
std::vector<CliCommand> InspectCommands();

/** The Models group: `delete-segment`, `prune`, `export-rlbot` and `export-submittal`. */
std::vector<CliCommand> ModelCommands();

/** `delete-segment`: confirms, then removes the segment. Exit 2 when it refuses to ask, 1 when declined. */
int RunDeleteSegmentCommand(const CliArgs& args);
int RunDeleteSegmentCommand(const CliArgs& args, bool interactive, std::istream& in, std::ostream& out);

/** `prune`: confirms, then keeps only the newest checkpoint of the segment or of every segment. */
int RunPruneCommand(const CliArgs& args);
int RunPruneCommand(const CliArgs& args, bool interactive, std::istream& in, std::ostream& out);

int RunExportRlbotCommand(const CliArgs& args);
int RunExportSubmittalCommand(const CliArgs& args);

/** One `list` subject's data: the table, the same rows as JSON, and the values the subject resolved to get there. */
struct ListOutput {
	std::vector<std::string> headers;
	std::vector<std::vector<std::string>> rows;
	nlohmann::json json = nlohmann::json::array();
	std::vector<std::pair<std::string, std::string>> resolved;
};

/** Collects one `list` subject's data without printing. Throws CliUsageError for an unknown subject or unresolvable option. */
ListOutput ListSubject(const std::string& subject, const CliArgs& args);

/** `list <subject>`: the table on stdout, or with --json the JSON on stdout and the resolved values on stderr. */
int RunListCommand(const CliArgs& args);

/** The Debug & analysis group's JSON-configured render: `debug`. */
std::vector<CliCommand> DebugCommands();

/**
 * Applies `debug`'s one-run flags to the loaded forge_debug_settings.json, in that file's encoding.
 * stageTeamSizes is the rendered stage's weighted list, which --team-size labels index into.
 * Throws CliUsageError for a conflicting or unknown value.
 */
RocketForgeTraining::ForgeDebugSettings ApplyDebugOverrides(
	const CliArgs& args,
	RocketForgeTraining::ForgeDebugSettings loaded,
	const std::vector<RocketForge::WeightedTeamSize>& stageTeamSizes);

/** `debug`: renders a model in RocketSimVis until 'q' (Windows) or a stop signal. Never writes the debug settings. */
int RunDebugCommand(const CliArgs& args);

/** The rest of the Debug & analysis group: `reward-table`, `reward-analysis` and `action-trace`. */
std::vector<CliCommand> AnalysisCommands();

/** The Benchmarking group: `training-benchmark` and `model-benchmark`. */
std::vector<CliCommand> BenchmarkCommands();

/** `training-benchmark`: a bounded training run reporting steady-state steps per second. Never writes settings or checkpoints. */
int RunTrainingBenchmarkCommand(const CliArgs& args);

/** `model-benchmark`: runs the persisted benchmark setup and writes the GUI-importable workbook. Exit 2 when the setup cannot resolve. */
int RunModelBenchmarkCommand(const CliArgs& args);

/**
 * Creates the workbook's folder and checks that it can be written, so a bad --out fails before a long run.
 * Throws CliUsageError when the path is an existing folder, CliRuntimeError when the folder can't be created or written.
 */
void PrepareModelBenchmarkExportPath(const std::filesystem::path& exportPath);

/** The Script plumbing group: `models-path`, `build-info`, `sync-manifest` and `merge-segment-ids`; stdout is machine-read. */
std::vector<CliCommand> PlumbingCommands();

}
