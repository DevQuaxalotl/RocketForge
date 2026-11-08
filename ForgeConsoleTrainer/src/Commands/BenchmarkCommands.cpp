#include "Commands.h"

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"
#include "Cli/TextTable.h"

#include "ForgeTraining.h"
#include "ModelBenchmark/ModelBenchmarkExport.h"
#include "ModelBenchmark/ModelBenchmarkRoster.h"
#include "Settings/ForgeSettings.h"
#include "StageRegistry.h"
#include "Storage/BuildStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"
#include "TrainingStage.h"
#include "Utils/SteadyStateSps.h"

#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <RLGymCPP/ThreadPool.h>
#include <RocketSim/Engine.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <set>
#include <sstream>

using namespace std;
using namespace RocketForge;
using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const TRAINING_BENCHMARK_USAGE =
		"Usage: ForgeConsoleTrainer training-benchmark --stage <name> [options]\n"
		"\n"
		"Runs a bounded training benchmark with detailed timing buckets.\n"
		"Metrics, skill tracking, checkpoint writes and static opponents are always off.\n"
		"\n"
		"Options:\n"
		"  --stage <name>              Stage to benchmark (required; see `list stages`)\n"
		"  --agent <name>              Agent the stage belongs to (default: the GUI-selected agent, else example-agent)\n"
		"  --checkpoint-path <path>    Segment `checkpoints` folder to start from instead of a fresh policy\n"
		"  --graphs                    Enable CUDA graphs (off by default so runs compare like for like)\n"
		"  --recent-opponents          Train against the build's recent policy versions (needs --checkpoint-path)\n"
		"  --warmup-iterations <n>     Iterations excluded from the steady-state figures (default: 3)\n"
		"  --iterations <n>            Measured iterations (default: 4)\n"
		"  --timesteps <n>             Timesteps per iteration (default: 8192)\n"
		"  --arenas <n>                Parallel games (default: 128)\n"
		"  --microbatch <n>            PPO micro-batch size; --timesteps must be a multiple of it (default: 4096)\n"
		"  --seed <n>                  Random seed (default: 1234)\n"
		"  --performance-cores         Windows hybrid CPUs: collect on P-cores only\n";

	const char* const MODEL_BENCHMARK_USAGE =
		"Usage: ForgeConsoleTrainer model-benchmark [options]\n"
		"\n"
		"Runs the model benchmark described by forge_model_benchmark_settings.json and writes the same workbook the GUI exports.\n"
		"Entries that give only agentName, buildName and versionId resolve to that segment's newest checkpoint.\n"
		"Imported models use sourceLabel (or buildName) \"Imported Model\" and their folder name as versionId.\n"
		"Models are named \"agent / build / versionId\", so each one can be selected once.\n"
		"\n"
		"Options:\n"
		"  --settings <path>           Benchmark settings file (default: forge_model_benchmark_settings.json at the repo root)\n"
		"  --out <path>                Workbook to write (default: Model Benchmarks/model-benchmark-<timestamp>.xml)\n"
		"  --arenas <n>                Override the number of parallel games\n"
		"  --max-games-per-mode <n>    Override the game budget per team size\n"
		"  --seed <n>                  Override the seed (0 picks a new one)\n"
		"  --optional-stats            Collect gameplay statistics\n"
		"  --no-optional-stats         Skip gameplay statistics\n"
		"  --run-until-max             Play the full game budget\n"
		"  --stop-early                Stop once the stopping rules are met\n"
		"  --all-primary               Treat every model as primary\n";

	optional<uint64_t> GetPositive(const CliArgs& args, const string& option)
	{
		const optional<uint64_t> value = args.GetUint64(option);
		if (value.has_value() && (*value == 0 || *value > static_cast<uint64_t>(INT64_MAX)))
			throw CliUsageError(option + " must be a positive number that fits in 63 bits (got " + to_string(*value) + ")");

		return value;
	}

	/** A random version-4 UUID in the GUI's run-id format (lowercase, no braces). */
	string MakeRunId()
	{
		random_device entropy;
		mt19937_64 generator((static_cast<uint64_t>(entropy()) << 32) | entropy());

		const uint64_t high = (generator() & ~0xF000ULL) | 0x4000ULL;
		const uint64_t low = (generator() & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

		const unsigned timeLow = static_cast<unsigned>(high >> 32);
		const unsigned timeMid = static_cast<unsigned>((high >> 16) & 0xFFFF);
		const unsigned timeHigh = static_cast<unsigned>(high & 0xFFFF);
		const unsigned clockSequence = static_cast<unsigned>(low >> 48);
		const unsigned long long node = low & 0xFFFFFFFFFFFFULL;

		char text[37];
		snprintf(text, sizeof(text), "%08x-%04x-%04x-%04x-%012llx", timeLow, timeMid, timeHigh, clockSequence, node);

		return text;
	}

	string FormatDuration(double seconds)
	{
		if (seconds < 0.0)
			return "--";

		const int64_t total = static_cast<int64_t>(seconds + 0.5);
		const int64_t hours = total / 3600;
		const int64_t minutes = (total / 60) % 60;

		if (hours > 0)
			return to_string(hours) + "h" + to_string(minutes) + "m" + to_string(total % 60) + "s";
		if (minutes > 0)
			return to_string(minutes) + "m" + to_string(total % 60) + "s";

		return to_string(total) + "s";
	}

	string FormatNumber(double value, int decimals)
	{
		char text[32];
		snprintf(text, sizeof(text), "%.*f", decimals, value);

		return text;
	}

	string FormatProgressLine(const ModelBenchmarkProgress& progress)
	{
		ostringstream line;
		line << "[benchmark] " << (progress.label.empty() ? "all" : progress.label);
		if (progress.maxGames > 0)
			line << " scored " << progress.scoredGames << "/" << progress.maxGames;
		line << "  " << FormatNumber(progress.snapshotGamesPerSecond, 1) << " games/s";
		line << "  eta " << FormatDuration(progress.estimatedRemainingSeconds);

		return line.str();
	}

	string JoinLines(const vector<string>& lines)
	{
		string joined;
		for (size_t i = 0; i < lines.size(); i++)
			joined += (i == 0 ? "" : "\n  ") + lines[i];

		return joined;
	}

	ModelBenchmarkSettings LoadSettingsWithOverrides(const CliArgs& args, const fs::path& settingsPath)
	{
		if (args.Has("--optional-stats") && args.Has("--no-optional-stats"))
			throw CliUsageError("--optional-stats and --no-optional-stats can't be combined");
		if (args.Has("--run-until-max") && args.Has("--stop-early"))
			throw CliUsageError("--run-until-max and --stop-early can't be combined");

		if (!fs::exists(settingsPath))
			throw CliUsageError("settings file not found: " + settingsPath.string());

		const ModelBenchmarkSettingsLoadResult loaded = LoadModelBenchmarkSettings(settingsPath);
		if (!loaded.ok)
			throw CliUsageError("could not load " + settingsPath.string() + "\n  " + JoinLines(loaded.errors));

		ModelBenchmarkSettings settings = loaded.value;

		if (const optional<uint64_t> arenas = GetPositive(args, "--arenas"))
			settings.numArenas = static_cast<int>(min<uint64_t>(*arenas, INT_MAX));
		if (const optional<uint64_t> maxGames = GetPositive(args, "--max-games-per-mode"))
			settings.maxGamesPerMode = static_cast<int>(min<uint64_t>(*maxGames, INT_MAX));
		if (const optional<uint64_t> seed = args.GetUint64("--seed"))
			settings.seed = *seed;

		if (args.Has("--optional-stats"))
			settings.collectOptionalStats = true;
		if (args.Has("--no-optional-stats"))
			settings.collectOptionalStats = false;
		if (args.Has("--run-until-max"))
			settings.runUntilMaxGames = true;
		if (args.Has("--stop-early"))
			settings.runUntilMaxGames = false;

		return settings;
	}

	void PrintSummary(const ModelBenchmarkResult& result)
	{
		cout << "\nScored games: " << result.scoredGames << "  No-result games: " << result.noResultGames
			<< "  Elapsed: " << FormatDuration(result.elapsedSeconds) << "\n\n";

		const bool eloAvailable = result.overallEloAvailable;

		vector<vector<string>> rows;
		for (const ModelBenchmarkRankingRow& row : result.rankings) {
			rows.push_back({
				eloAvailable ? to_string(row.rank) : "N/A",
				row.modelName,
				eloAvailable ? FormatNumber(row.relativeElo, 0) : "N/A",
				eloAvailable ? "[" + FormatNumber(row.rangeLow, 0) + ", " + FormatNumber(row.rangeHigh, 0) + "]" : "N/A",
				to_string(row.totalRecord.wins) + "-" + to_string(row.totalRecord.losses) + "-" + to_string(row.totalRecord.noResults),
			});
		}

		cout << FormatTable({ "Rank", "Model", "Elo", "Range", "Record (W-L-NR)" }, rows);
	}

	/** The newest numbered checkpoint's total_iterations, so the benchmark can run its iterations on top of a loaded checkpoint. */
	uint64_t ReadCheckpointTotalIterations(const fs::path& checkpointRoot)
	{
		const set<int64_t> timesteps = GGL::TrainingStorage::FindNumberedDirs(checkpointRoot);
		if (timesteps.empty())
			return 0;

		ifstream in(checkpointRoot / to_string(*timesteps.rbegin()) / "RUNNING_STATS.json");
		if (!in.good())
			return 0;

		const nlohmann::json stats = nlohmann::json::parse(in, nullptr, false);
		if (stats.is_discarded())
			return 0;

		return stats.value("total_iterations", uint64_t{ 0 });
	}

	/** The segment folder a --checkpoint-path points into: the `checkpoints` folder itself or one numbered folder inside it. */
	fs::path SegmentRootOfCheckpointPath(const fs::path& checkpointPath)
	{
		if (checkpointPath.filename() == "checkpoints")
			return checkpointPath.parent_path();

		if (checkpointPath.parent_path().filename() == "checkpoints")
			return checkpointPath.parent_path().parent_path();

		return {};
	}

}

int RocketForge::Cli::RunTrainingBenchmarkCommand(const CliArgs& args)
{
	const string agentName = ResolveAgent(args);
	const string stageName = ResolveStage(args, true);

	const bool useGraphs = args.Has("--graphs");
	const bool useRecentOpponents = args.Has("--recent-opponents");
	const uint64_t iterations = GetPositive(args, "--iterations").value_or(4);
	const uint64_t warmupIterations = args.GetUint64("--warmup-iterations").value_or(3);
	const int64_t timesteps = static_cast<int64_t>(GetPositive(args, "--timesteps").value_or(8192));
	const int64_t arenas = static_cast<int64_t>(GetPositive(args, "--arenas").value_or(128));
	const int64_t microBatch = static_cast<int64_t>(GetPositive(args, "--microbatch").value_or(4096));
	const int64_t seed = static_cast<int64_t>(args.GetUint64("--seed").value_or(1234));

	if (arenas > INT_MAX)
		throw CliUsageError("--arenas is too large (got " + to_string(arenas) + ")");
	if (timesteps % microBatch != 0)
		throw CliUsageError("--timesteps must be a multiple of --microbatch");

	const optional<string> checkpointPathValue = args.Get("--checkpoint-path");
	const fs::path checkpointLoadFolder = checkpointPathValue.has_value() ? fs::absolute(*checkpointPathValue) : fs::path();
	if (!checkpointLoadFolder.empty() && !fs::exists(checkpointLoadFolder))
		throw CliUsageError("--checkpoint-path does not exist: " + checkpointLoadFolder.string());
	if (useRecentOpponents && checkpointLoadFolder.empty())
		throw CliUsageError("--recent-opponents requires --checkpoint-path");

	const uint64_t initialIterations = checkpointLoadFolder.empty() ? 0 : ReadCheckpointTotalIterations(checkpointLoadFolder);

	GlobalSettings globals = {};
	globals.metricsEnabled = false;
	globals.performanceCoresOnly = args.Has("--performance-cores");

	// This path skips RunTraining, which would otherwise apply the setting before the thread pool is first used.
	if (globals.performanceCoresOnly)
		RLGC::RestrictProcessToPerformanceCores();

	StageSettings stageSettings = {};
	stageSettings.stageName = stageName;
	stageSettings.targetTotalTimesteps = UINT64_MAX;

	const fs::path benchmarkSegmentRoot = SegmentRootOfCheckpointPath(checkpointLoadFolder);
	optional<TrainingSetupSnapshot> savedTrainingSetup = nullopt;

	const fs::path setupPath = benchmarkSegmentRoot / TrainingSetupSnapshot::FILE_NAME;
	if (!benchmarkSegmentRoot.empty() && fs::exists(setupPath)) {
		string setupError = {};
		savedTrainingSetup = LoadTrainingSetupSnapshot(setupPath, &setupError);
		if (!savedTrainingSetup.has_value())
			throw CliRuntimeError("failed to load the checkpoint's saved training setup: " + setupError);
		if (!savedTrainingSetup->agentSetup.has_value())
			throw CliRuntimeError("the checkpoint's saved training setup has no agent setup metadata");

		stageSettings.agentSetupOverride = savedTrainingSetup->agentSetup;
	}

	if (useRecentOpponents && !savedTrainingSetup.has_value())
		throw CliRuntimeError("--recent-opponents requires a checkpoint with saved training setup metadata");

	const unique_ptr<TrainingStage> stage = CreateTrainingStage(agentName, stageName, globals, stageSettings);
	if (!stage)
		throw CliRuntimeError("could not create stage '" + stageName + "' for agent '" + agentName + "'");

	if (!savedTrainingSetup.has_value())
		RG_LOG("training-benchmark: no saved training setup, so using " << agentName << "'s own simulator "
			<< RocketSim::GetEngineInfo(stage->learnerConfig.engine).displayName
			<< " and ball predictor RocketSim v" << stage->agent->GetBallPredictionEngineVersion());

	for (RocketSim::EngineId engine : RequiredEngines(CaptureAgentSetupSnapshot(*stage->agent)))
		RocketSim::RequireEngine(engine);

	InitializeTraining();

	GGL::LearnerConfig& config = stage->learnerConfig;
	GGL::LearnerRuntimeConfig& runtimeConfig = config.runtime;
	fs::path recentPolicyVersionsFolder = runtimeConfig.recentPolicyVersionsFolder;
	if (useRecentOpponents) {
		// Stage checkpoints live under <build>/segments/<segment>/checkpoints while
		// recent policies are shared by the build at <build>/recent_policy_versions.
		const fs::path buildRoot = benchmarkSegmentRoot.parent_path().parent_path();
		recentPolicyVersionsFolder = buildRoot / BuildStorage::RECENT_POLICY_VERSIONS_FOLDER_NAME;
	}

	config.sendMetrics = false;
	config.savePolicyVersions = false;
	config.skillTracker.enabled = false;
	config.trainAgainstRecentVersions = useRecentOpponents;
	runtimeConfig.recentPolicyVersionsFolder = recentPolicyVersionsFolder;
	if (useRecentOpponents) {
		runtimeConfig.versionId = savedTrainingSetup->versionId;
		runtimeConfig.segmentId = savedTrainingSetup->versionId;
	}

	config.trainAgainstStaticOpponents = false;
	runtimeConfig.checkpointFolder = fs::path();
	runtimeConfig.checkpointLoadFolder = checkpointLoadFolder;
	runtimeConfig.stopAtTotalTimesteps = 0;
	runtimeConfig.stopAfterIterations = initialIterations + warmupIterations + iterations;
	config.randomSeed = seed;
	config.numGames = static_cast<int>(arenas);
	config.ppo.tsPerItr = timesteps;
	config.ppo.batchSize = timesteps;
	config.ppo.microBatchSize = microBatch;
	config.ppo.useCudaGraphs = useGraphs;

	cout << "Training benchmark: agent=" << agentName
		<< " stage=" << stageName
		<< " graphs=" << (useGraphs ? "on" : "off")
		<< " recentOpponents=" << (useRecentOpponents ? "on" : "off")
		<< " warmupIterations=" << warmupIterations
		<< " iterations=" << iterations
		<< " timesteps=" << timesteps
		<< " arenas=" << arenas
		<< " microbatch=" << microBatch
		<< " seed=" << seed;
	if (!runtimeConfig.checkpointLoadFolder.empty())
		cout << " checkpointPath=" << runtimeConfig.checkpointLoadFolder << " initialIterations=" << initialIterations;
	cout << "\n";

	SteadyStateSps steadyStateSps = {};
	steadyStateSps.warmupIterations = static_cast<int>(warmupIterations);
	runtimeConfig.onIterationReport = [&steadyStateSps](const GGL::Report& report) {
		steadyStateSps.AddIteration(static_cast<uint64_t>(report["Collected Timesteps"]), report["Collection Time"], report["Consumption Time"]);
	};

	GGL::Learner learner(stage->environmentCreateFunc, config, stage->stepCallbackFunc);
	learner.Start();

	cout << steadyStateSps.Summary() << '\n';
	return 0;
}

int RocketForge::Cli::RunModelBenchmarkCommand(const CliArgs& args)
{
	const optional<string> settingsOption = args.Get("--settings");
	const fs::path settingsPath = settingsOption.has_value() ? fs::absolute(*settingsOption) : GetDefaultModelBenchmarkSettingsPath();
	const ModelBenchmarkSettings settings = LoadSettingsWithOverrides(args, settingsPath);

	const ModelBenchmarkRoster roster = ResolveModelBenchmarkRoster(settings, args.Has("--all-primary"));
	for (const string& warning : roster.warnings)
		cerr << "model-benchmark: warning: " << warning << "\n";
	if (!roster.errors.empty())
		throw CliUsageError(JoinLines(roster.errors));

	const optional<string> outOption = args.Get("--out");
	const fs::path defaultExportPath = DefaultModelBenchmarkExportFolder() / DefaultModelBenchmarkExportName();
	const fs::path exportPath = outOption.has_value() ? fs::absolute(*outOption) : defaultExportPath;
	PrepareModelBenchmarkExportPath(exportPath);

	PrintResolved(cout, {
		{ "settings", settingsPath.string() },
		{ "models", to_string(roster.config.models.size()) },
		{ "arenas", to_string(roster.config.numArenas) },
		{ "max games per mode", to_string(roster.config.maxGamesPerMode) },
		{ "seed", to_string(roster.config.seed) },
		{ "optional stats", roster.config.collectOptionalStats ? "on" : "off" },
		{ "run until max", roster.config.runUntilMaxGames ? "on" : "off" },
	});

	vector<string> consoleLines;
	const ModelBenchmarkResult result = RunModelBenchmark(roster.config, [&consoleLines](const ModelBenchmarkProgress& progress) {
		consoleLines.push_back(FormatProgressLine(progress));
		cout << consoleLines.back() << endl;
	});

	PrintSummary(result);

	ModelBenchmarkExportInput exportInput;
	exportInput.result = &result;
	exportInput.runId = MakeRunId();
	exportInput.models = roster.exportModels;
	exportInput.consoleLines = consoleLines;

	string exportError;
	if (!WriteModelBenchmarkExport(exportPath, exportInput, &exportError)) {
		cerr << "model-benchmark: failed to write " << exportPath.string() << ": " << exportError << "\n";
		return 1;
	}

	cout << "\nSaved results: " << exportPath.string() << "\n";
	return 0;
}

void RocketForge::Cli::PrepareModelBenchmarkExportPath(const fs::path& exportPath)
{
	if (fs::is_directory(exportPath))
		throw CliUsageError("--out names a folder: " + exportPath.string() + ". Give the workbook's file path");

	const fs::path folder = exportPath.parent_path();
	error_code directoryError;
	fs::create_directories(folder, directoryError);
	if (directoryError)
		throw CliRuntimeError("could not create " + folder.string() + ": " + directoryError.message());

	const fs::path probePath = folder / (exportPath.filename().string() + ".write-check.tmp");
	const bool writable = ofstream(probePath).good();
	error_code removeError;
	fs::remove(probePath, removeError);
	if (!writable)
		throw CliRuntimeError("can't write to " + folder.string());
}

vector<CliCommand> RocketForge::Cli::BenchmarkCommands()
{
	CliCommand trainingBenchmark;
	trainingBenchmark.name = "training-benchmark";
	trainingBenchmark.group = "Benchmarking";
	trainingBenchmark.summary = "Runs a bounded training benchmark with detailed timing buckets";
	trainingBenchmark.usage = TRAINING_BENCHMARK_USAGE;
	trainingBenchmark.options.valueOptions = {
		"--agent", "--stage", "--checkpoint-path", "--warmup-iterations", "--iterations", "--timesteps", "--arenas", "--microbatch", "--seed",
	};
	trainingBenchmark.options.switches = { "--graphs", "--recent-opponents", "--performance-cores" };
	trainingBenchmark.run = RunTrainingBenchmarkCommand;

	CliCommand modelBenchmark;
	modelBenchmark.name = "model-benchmark";
	modelBenchmark.group = "Benchmarking";
	modelBenchmark.summary = "Runs the model benchmark from forge_model_benchmark_settings.json and exports the GUI workbook";
	modelBenchmark.usage = MODEL_BENCHMARK_USAGE;
	modelBenchmark.options.valueOptions = { "--settings", "--out", "--arenas", "--max-games-per-mode", "--seed" };
	modelBenchmark.options.switches = { "--optional-stats", "--no-optional-stats", "--run-until-max", "--stop-early", "--all-primary" };
	modelBenchmark.run = RunModelBenchmarkCommand;

	return { trainingBenchmark, modelBenchmark };
}
