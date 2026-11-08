#pragma once

#include "Cli/CliArgs.h"
#include "SimulationAnalysisTypes.h"

#include <filesystem>
#include <string>

namespace RocketForge::SimulationAnalysis
{
	/** Resolves a command-line path against the current working directory. **/
	std::filesystem::path ResolvePath(std::filesystem::path path);
	/** Returns the JSON summary path paired with a CSV output path. **/
	std::filesystem::path SummaryPathForCsv(const std::filesystem::path& csvPath);
	/**
	 * Parses and validates action-trace options, throwing CliUsageError on bad input.
	 * --setup and --checkpoint-path win; whatever they leave out comes from the resolved segment.
	 **/
	SimulationAnalysisOptions ParseSimulationAnalysisOptions(const RocketForge::Cli::CliArgs& args);
	/** Returns a compact display name for the selected reset/replay mode. **/
	std::string TraceModeName(const SimulationAnalysisOptions& options);
}