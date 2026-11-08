#pragma once

#include "CliArgs.h"

#include "Storage/BuildStorage.h"

#include <GigaLearnCPP/SegmentVersioning.h>

#include <filesystem>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace RocketForge::Cli {

/**
 * Returns value when it is in known.
 * Otherwise throws CliUsageError: "unknown <kind> '<value>'. Did you mean: a, b? Run `ForgeConsoleTrainer <listCommand>`".
 */
std::string RequireKnownName(const std::string& kind, const std::string& value, const std::vector<std::string>& known, const std::string& listCommand);

/** savedAgent, else "example-agent": the agent --agent defaults to. */
std::string DefaultAgentName(const std::string& savedAgent);

/** --agent, else savedAgent, else "example-agent"; validated against known. */
std::string ResolveAgentName(const CliArgs& args, const std::string& savedAgent, const std::vector<std::string>& known);

/** ResolveAgentName over the GUI-selected agent and the registered agents. Never writes the selection. */
std::string ResolveAgent(const CliArgs& args);

/** --file (relative paths resolve from the working directory), else the agent's forge_settings.json. */
std::filesystem::path ResolveSettingsPath(const CliArgs& args, const std::string& agentName);

/** --stage validated against the registered stages; empty when absent and not required. */
std::string ResolveStage(const CliArgs& args, bool required);

/**
 * --build validated against the agent's builds, else the settings file's buildName, else GGL::DEFAULT_BUILD_NAME.
 * Throws CliUsageError when an explicit --file is missing, and CliRuntimeError when the settings file fails to load.
 */
std::string ResolveBuild(const CliArgs& args, const std::string& agentName, const std::filesystem::path& settingsPath);

/** --segment, else the build's active segment; throws CliRuntimeError when the build has neither. */
GGL::SegmentVersionSummary ResolveSegment(const CliArgs& args, const std::string& agentName, const std::string& buildName);

/** --checkpoint timesteps, else the newest checkpoint; autosaves count, and a durable checkpoint wins a tie. */
RocketForgeTraining::ResolvedCheckpoint ResolveCheckpoint(const CliArgs& args, const std::filesystem::path& segmentRoot);

/** Prints "name: value" pairs on one line, e.g. "agent: x  build: y  segment: z". */
void PrintResolved(std::ostream& out, const std::vector<std::pair<std::string, std::string>>& fields);

}
