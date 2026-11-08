#include "Options.h"

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"

#include <climits>
#include <optional>

namespace RocketForge::SimulationAnalysis
{
	bool IsTraceFlag(const std::string& arg)
	{
		return arg.rfind("--", 0) == 0;
	}

	std::filesystem::path ResolvePath(std::filesystem::path path)
	{
		if (path.is_relative())
			path = std::filesystem::current_path() / path;
		path = std::filesystem::absolute(path);
		path.make_preferred();
		return path;
	}

	std::filesystem::path SummaryPathForCsv(const std::filesystem::path& csvPath)
	{
		std::filesystem::path result = csvPath;
		result.replace_extension(".summary.json");
		return result;
	}

	namespace
	{
		using RocketForge::Cli::CliArgs;
		using RocketForge::Cli::CliUsageError;

		const char* const TRAINING_SETUP_FILE_NAME = "training_setup.json";

		int GetInt(const CliArgs& args, const std::string& option, int defaultValue)
		{
			const std::optional<uint64_t> value = args.GetUint64(option);
			if (!value.has_value())
				return defaultValue;

			if (*value > static_cast<uint64_t>(INT_MAX))
				throw CliUsageError(option + " is too large (got " + std::to_string(*value) + ")");

			return static_cast<int>(*value);
		}

		std::filesystem::path GetPath(const CliArgs& args, const std::string& option)
		{
			const std::optional<std::string> value = args.Get(option);
			return value.has_value() ? std::filesystem::path(*value) : std::filesystem::path();
		}

		/** The flag values and their cross-flag rules, before any agent, segment or path resolution. */
		SimulationAnalysisOptions ParseTraceFlags(const CliArgs& args)
		{
			SimulationAnalysisOptions options = {};
			options.setupPath = GetPath(args, "--setup");
			options.checkpointPath = GetPath(args, "--checkpoint-path");
			options.outPath = args.Get("--out").value_or("action_trace.csv");
			options.episodes = GetInt(args, "--episodes", options.episodes);
			options.maxTicks = GetInt(args, "--max-ticks", options.maxTicks);
			options.teamSize = GetInt(args, "--team-size", options.teamSize);
			options.seed = args.GetUint64("--seed").value_or(options.seed);
			options.kickoffMode = args.Has("--kickoff");
			options.wallProbeMode = args.Has("--wall-probe");
			options.stageResetName = args.Get("--stage-reset").value_or("");
			options.snapshotCsvPath = GetPath(args, "--snapshot-csv");
			options.snapshotTick = GetInt(args, "--snapshot-tick", options.snapshotTick);
			options.snapshotBotIndex = GetInt(args, "--snapshot-bot-index", options.snapshotBotIndex);
			options.snapshotCarId = GetInt(args, "--snapshot-car-id", static_cast<int>(options.snapshotCarId));
			options.actionSequenceCsvPath = GetPath(args, "--action-sequence-csv");
			options.controlReplayCsvPath = GetPath(args, "--control-replay-csv");
			options.controlReplayInputDelayTicks = GetInt(args, "--control-replay-input-delay-ticks", options.controlReplayInputDelayTicks);
			options.controlReplayUseLastInput = args.Has("--control-replay-use-last-input");
			options.controlReplayFocusPolicy = args.Has("--control-replay-focus-policy");
			options.forcedActionId = GetInt(args, "--force-action", options.forcedActionId);
			options.deterministic = !args.Has("--stochastic");
			options.temperature = static_cast<float>(args.GetDouble("--temperature").value_or(options.temperature));

			if (options.episodes <= 0)
				throw CliUsageError("--episodes must be positive");
			if (options.maxTicks <= 0)
				throw CliUsageError("--max-ticks must be positive");
			if (options.teamSize < 1 || options.teamSize > 3)
				throw CliUsageError("--team-size must be between 1 and 3");
			if (options.temperature <= 0.f)
				throw CliUsageError("--temperature must be positive");

			const bool snapshotMode = !options.snapshotCsvPath.empty();
			const int resetModes = (options.kickoffMode ? 1 : 0) + (options.wallProbeMode ? 1 : 0) +
				(!options.stageResetName.empty() ? 1 : 0) + (snapshotMode ? 1 : 0);
			if (resetModes > 1)
				throw CliUsageError("choose only one reset mode");
			if (snapshotMode && options.snapshotTick < 0)
				throw CliUsageError("--snapshot-tick is required with --snapshot-csv");
			if (!snapshotMode && (!options.controlReplayCsvPath.empty() || options.snapshotBotIndex >= 0 || options.snapshotCarId >= 0))
				throw CliUsageError("replay filters require --snapshot-csv");
			if (!options.controlReplayCsvPath.empty() && (options.forcedActionId >= 0 || !options.actionSequenceCsvPath.empty()))
				throw CliUsageError("--control-replay-csv cannot be combined with --force-action or --action-sequence-csv");
			if (options.controlReplayInputDelayTicks > 0 && options.controlReplayCsvPath.empty())
				throw CliUsageError("--control-replay-input-delay-ticks requires --control-replay-csv");
			if (options.controlReplayUseLastInput && options.controlReplayCsvPath.empty())
				throw CliUsageError("--control-replay-use-last-input requires --control-replay-csv");
			if (options.controlReplayFocusPolicy && options.controlReplayCsvPath.empty())
				throw CliUsageError("--control-replay-focus-policy requires --control-replay-csv");

			return options;
		}
	}

	SimulationAnalysisOptions ParseSimulationAnalysisOptions(const RocketForge::Cli::CliArgs& args)
	{
		using namespace RocketForge::Cli;

		SimulationAnalysisOptions options = ParseTraceFlags(args);
		options.agentName = ResolveAgent(args);

		const bool hasSetupPath = !options.setupPath.empty();
		const bool hasCheckpointPath = !options.checkpointPath.empty();
		if (hasCheckpointPath && args.Get("--checkpoint").has_value())
			throw CliUsageError("--checkpoint and --checkpoint-path can't be combined");

		// A segment is only consulted for what the explicit paths leave out.
		if (!hasSetupPath || !hasCheckpointPath) {
			const std::filesystem::path settingsPath = ResolveSettingsPath(args, options.agentName);
			options.buildName = ResolveBuild(args, options.agentName, settingsPath);

			const GGL::SegmentVersionSummary segment = ResolveSegment(args, options.agentName, options.buildName);
			options.segmentId = segment.id;

			if (!hasSetupPath)
				options.setupPath = std::filesystem::path(segment.rootPath) / TRAINING_SETUP_FILE_NAME;
			if (!hasCheckpointPath)
				options.checkpointPath = ResolveCheckpoint(args, std::filesystem::path(segment.rootPath)).folder;
		}

		options.setupPath = ResolvePath(options.setupPath);
		options.checkpointPath = ResolvePath(options.checkpointPath);
		options.outPath = ResolvePath(options.outPath);
		if (!options.snapshotCsvPath.empty())
			options.snapshotCsvPath = ResolvePath(options.snapshotCsvPath);
		if (!options.actionSequenceCsvPath.empty())
			options.actionSequenceCsvPath = ResolvePath(options.actionSequenceCsvPath);
		if (!options.controlReplayCsvPath.empty())
			options.controlReplayCsvPath = ResolvePath(options.controlReplayCsvPath);

		return options;
	}

	std::string TraceModeName(const SimulationAnalysisOptions& options)
	{
		if (!options.controlReplayCsvPath.empty())
		{
			std::string mode = options.controlReplayFocusPolicy
				? "control_replay_focus_policy"
				: options.controlReplayUseLastInput
				? "control_replay_last_input"
				: "control_replay";
			if (options.controlReplayInputDelayTicks > 0)
				mode += "_delay" + std::to_string(options.controlReplayInputDelayTicks);
			return mode;
		}
		if (!options.snapshotCsvPath.empty())
			return "snapshot";
		if (!options.stageResetName.empty())
			return "stage:" + options.stageResetName;
		return options.wallProbeMode ? "wall_probe" : "kickoff";
	}
}