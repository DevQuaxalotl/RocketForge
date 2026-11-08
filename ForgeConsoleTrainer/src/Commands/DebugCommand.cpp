#include "Commands.h"

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"
#include "Cli/ConsoleControl.h"

#include "AgentModelPaths.h"
#include "DebugRender/DebugRenderTarget.h"
#include "DebugRender/InferenceDebugRenderLoop.h"
#include "ForgeTraining.h"
#include "Settings/ForgeSettings.h"
#include "StageQueue/StageQueueSignals.h"
#include "StageRegistry.h"
#include "StaticOpponents/StaticOpponentSelection.h"
#include "Storage/BuildStorage.h"
#include "Storage/ImportedModelStorage.h"
#include "TrainingSetup/VersionManagement.h"
#include "TrainingStage.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <iostream>

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

using namespace std;
using namespace RocketForge;
using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const OPPONENT_RANDOM = "random";
	const char* const OPPONENT_NONE = "none";

	const char* const DEBUG_USAGE =
		"Usage: ForgeConsoleTrainer debug [options]\n"
		"\n"
		"Renders a model in RocketSimVis, starting from the agent's forge_debug_settings.json (the GUI Debug tab's settings).\n"
		"Without that file it renders the build's active segment on its trained stage.\n"
		"Flags change this run only; the file is never written.\n"
		"Reward and critic visualisation are GUI-only.\n"
		"\n"
		"Options:\n"
		"  --agent <name>           Agent to render (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>           Build to render; alone, it renders that build's active segment\n"
		"  --segment <id>           Segment to render (see `list segments`)\n"
		"  --imported <folder>      Imported model to render instead of a segment (see `list imported-models`)\n"
		"  --checkpoint <ts>        Checkpoint timesteps within the segment (default: newest; see `list checkpoints`)\n"
		"  --stage <name>           Stage to render on (default: the segment's trained stage; required for imported models)\n"
		"  --state-setter <i>       Force one state setter (index from `list stage-options`)\n"
		"  --team-size <AvB>        Force one team size, e.g. 2v2 (from `list stage-options`)\n"
		"  --player <i>             Player the camera starts on\n"
		"  --opponent <key>         Static opponent key (see `list static-opponents`), `random` per episode, or `none`\n"
		"  --episode-seconds <f>    End each episode after this much game time\n"
		"  --speed <f>              Game speed, e.g. 2 for double speed\n"
		"  --file <path>            forge_settings.json that names the default build\n"
		"\n"
#ifdef _MSC_VER
		"Press 'q' to quit (it works while the viewer has focus).\n";
#else
		"Press Ctrl+C to quit.\n";
#endif

	string FormatTeamSize(const WeightedTeamSize& teamSize)
	{
		return to_string(teamSize.teamSizeA) + "v" + to_string(teamSize.teamSizeB);
	}

	optional<int> GetIndex(const CliArgs& args, const string& option)
	{
		const optional<uint64_t> value = args.GetUint64(option);
		if (!value.has_value())
			return nullopt;

		if (*value > static_cast<uint64_t>(INT_MAX))
			throw CliUsageError(option + " is too large (got " + to_string(*value) + ")");

		return static_cast<int>(*value);
	}

	optional<float> GetPositive(const CliArgs& args, const string& option)
	{
		const optional<double> value = args.GetDouble(option);
		if (value.has_value() && *value <= 0)
			throw CliUsageError(option + " must be greater than 0");

		return value.has_value() ? optional<float>(static_cast<float>(*value)) : nullopt;
	}

	/** The model selection flags alone: --build, --segment, --imported and --stage. */
	ForgeDebugSettings ApplySelectionOverrides(const CliArgs& args, ForgeDebugSettings settings)
	{
		const optional<string> build = args.Get("--build");
		const optional<string> segment = args.Get("--segment");
		const optional<string> imported = args.Get("--imported");

		if (imported.has_value() && (build.has_value() || segment.has_value()))
			throw CliUsageError("--imported can't be combined with --build or --segment");

		if (imported.has_value()) {
			settings.buildName.clear();
			settings.segmentId = *imported;
		}

		// The saved segment belongs to the saved build, so another build starts from its own active segment.
		if (build.has_value()) {
			if (*build != settings.buildName && !segment.has_value())
				settings.segmentId.clear();
			settings.buildName = *build;
		}

		if (segment.has_value())
			settings.segmentId = *segment;

		if (const optional<string> stage = args.Get("--stage"))
			settings.renderStageName = *stage;

		return settings;
	}

	ForgeDebugSettings LoadDebugSettingsOrDefaults(const string& agentName)
	{
		const fs::path path = GetDefaultForgeDebugSettingsPath(agentName);

		std::error_code ignored;
		if (!fs::exists(path, ignored))
			return ForgeDebugSettings();

		const ForgeDebugSettingsLoadResult loaded = LoadForgeDebugSettings(path);
		if (!loaded.ok) {
			string message = "could not load " + path.string();
			for (const string& error : loaded.errors)
				message += "\n  " + error;
			throw CliRuntimeError(message);
		}

		return loaded.value;
	}

	/** Checks the selection by name first, so a typo gets the CLI's suggestions instead of the shared resolver's plain error. */
	void RequireKnownSelection(const string& agentName, const ForgeDebugSettings& selection)
	{
		if (selection.buildName.empty()) {
			vector<string> folderNames;
			for (const DiscoveredImportedModel& model : EnumerateImportedModels(agentName))
				folderNames.push_back(model.folderName);

			RequireKnownName("imported model", selection.segmentId, folderNames, "list imported-models --agent " + agentName);
			return;
		}

		const bool includeInactive = true;
		vector<string> segmentIds;
		for (const GGL::SegmentVersionSummary& segment : ListBuildVersions(agentName, selection.buildName, {}, includeInactive))
			segmentIds.push_back(segment.id);

		const string listCommand = "list segments --agent " + agentName + " --build " + selection.buildName;
		RequireKnownName("segment", selection.segmentId, segmentIds, listCommand);
	}

	void RequireKnownStaticOpponent(const string& agentName, const ForgeDebugSettings& settings)
	{
		if (settings.buildName.empty())
			throw CliUsageError("--opponent needs a build's static opponents; imported models have none");

		const vector<StaticOpponentSelectionEntry> entries = LoadStaticOpponentSelection(agentName, settings.buildName);
		if (entries.empty())
			throw CliUsageError("build '" + settings.buildName + "' has no static opponents selected");

		if (settings.debug.staticOpponentSelectionKey.empty())
			return;

		vector<string> keys;
		for (const StaticOpponentSelectionEntry& entry : entries)
			keys.push_back(MakeStaticOpponentSelectionKey(entry));

		const string listCommand = "list static-opponents --agent " + agentName + " --build " + settings.buildName;
		RequireKnownName("static opponent", settings.debug.staticOpponentSelectionKey, keys, listCommand);
	}

	/**
	 * Global key state rather than console input, so 'q' registers while the viewer window has focus.
	 * Elsewhere Ctrl+C is the quit path, through the stop handler.
	 */
	bool ConsoleQuitKeyPressed()
	{
#ifdef _MSC_VER
		return (GetAsyncKeyState('Q') & 0x8000) != 0;
#else
		return false;
#endif
	}

}

ForgeDebugSettings RocketForge::Cli::ApplyDebugOverrides(
	const CliArgs& args,
	ForgeDebugSettings loaded,
	const vector<WeightedTeamSize>& stageTeamSizes)
{
	ForgeDebugSettings settings = ApplySelectionOverrides(args, std::move(loaded));
	ForgeDebug::DebugSettings& debug = settings.debug;

	if (const optional<int> stateSetter = GetIndex(args, "--state-setter")) {
		debug.forceSingleStateSetter = true;
		debug.forceStateSetterIndex = *stateSetter;
	}

	if (const optional<string> teamSize = args.Get("--team-size")) {
		vector<string> labels;
		for (const WeightedTeamSize& stageTeamSize : stageTeamSizes)
			labels.push_back(FormatTeamSize(stageTeamSize));

		const string stageName = settings.renderStageName.empty() ? "<stage>" : settings.renderStageName;
		RequireKnownName("team size", *teamSize, labels, "list stage-options --stage " + stageName);
		debug.teamSizeOverrideIndex = static_cast<int>(std::find(labels.begin(), labels.end(), *teamSize) - labels.begin());
	}

	if (const optional<int> player = GetIndex(args, "--player"))
		debug.playerIndex = *player;

	if (const optional<string> opponent = args.Get("--opponent")) {
		debug.includeStaticOpponent = *opponent != OPPONENT_NONE;
		debug.staticOpponentSelectionKey = *opponent == OPPONENT_RANDOM || *opponent == OPPONENT_NONE ? "" : *opponent;
	}

	if (const optional<float> episodeSeconds = GetPositive(args, "--episode-seconds")) {
		debug.maxEpisodeTimeEnabled = true;
		debug.maxEpisodeTime = *episodeSeconds;
	}

	if (const optional<float> speed = GetPositive(args, "--speed"))
		debug.gameSpeed = *speed;

	return settings;
}

int RocketForge::Cli::RunDebugCommand(const CliArgs& args)
{
	const string agentName = ResolveAgent(args);
	const fs::path settingsPath = ResolveSettingsPath(args, agentName);

	if (args.Get("--imported").has_value() && args.Get("--checkpoint").has_value())
		throw CliUsageError("--checkpoint applies to segments; an imported model has a single checkpoint");

	// Called only for its error on an unknown --build; the result is discarded and resolved again below when the selection needs it.
	if (args.Get("--build").has_value())
		ResolveBuild(args, agentName, settingsPath);

	// The file's empty build marks an imported selection, unless --segment chose a trained one or nothing is selected at all.
	ForgeDebugSettings selection = ApplySelectionOverrides(args, LoadDebugSettingsOrDefaults(agentName));
	if (selection.buildName.empty() && (args.Get("--segment").has_value() || selection.segmentId.empty()))
		selection.buildName = ResolveBuild(args, agentName, settingsPath);

	if (!selection.buildName.empty() && selection.segmentId.empty()) {
		selection.segmentId = BuildStorage(ResolveAgentModelsFolder(agentName), selection.buildName).LoadBuild().activeSegmentId;
		if (selection.segmentId.empty())
			throw CliRuntimeError("build '" + selection.buildName + "' has no active segment; train it first or pass --segment");
	}

	RequireKnownSelection(agentName, selection);

	const uint64_t checkpointTimesteps = args.GetUint64("--checkpoint").value_or(0);
	const DebugRenderTarget target = ResolveDebugRenderTarget(agentName, selection.buildName, selection.segmentId, checkpointTimesteps);

	if (!target.error.empty()) {
		if (checkpointTimesteps == 0)
			throw CliRuntimeError(target.error);

		const string listCommand =
			"list checkpoints --agent " + agentName + " --build " + selection.buildName + " --segment " + selection.segmentId;
		throw CliUsageError(target.error + " Run `ForgeConsoleTrainer " + listCommand + "`");
	}

	const string stageName = selection.renderStageName.empty() ? target.trainedStageName : selection.renderStageName;
	if (stageName.empty())
		throw CliUsageError("--stage is required for imported models. Run `ForgeConsoleTrainer list stages`");

	RequireKnownName("stage", stageName, StageRegistry::GetAvailableStages(), "list stages");

	const unique_ptr<TrainingStage> stage = CreateTrainingStage(agentName, stageName, GlobalSettings(), StageSettings());
	if (!stage)
		throw CliRuntimeError("could not create stage '" + stageName + "' for agent '" + agentName + "'");

	ForgeDebugSettings settings = ApplyDebugOverrides(args, selection, stage->GetWeightedTeamSizes());

	const int stateSetterCount = static_cast<int>(stage->GetDebugStateSetterOptions().size());
	if (args.Get("--state-setter").has_value() && settings.debug.forceStateSetterIndex >= stateSetterCount) {
		const string listCommand = "list stage-options --agent " + agentName + " --stage " + stageName;
		const string index = to_string(settings.debug.forceStateSetterIndex);
		throw CliUsageError("unknown state setter " + index + ". Run `ForgeConsoleTrainer " + listCommand + "`");
	}

	if (args.Get("--opponent").has_value() && settings.debug.includeStaticOpponent)
		RequireKnownStaticOpponent(agentName, settings);

	if (target.isImported)
		PrintResolved(cout, { { "agent", agentName }, { "imported", selection.segmentId }, { "stage", stageName } });
	else
		PrintResolved(cout, {
			{ "agent", agentName },
			{ "build", selection.buildName },
			{ "segment", selection.segmentId },
			{ "checkpoint", target.checkpointFolder.filename().string() },
			{ "stage", stageName },
		});

	RocketForge::InferenceDebugRenderConfig config;
	config.agentName = agentName;
	config.stageName = stageName;
	config.checkpointFolder = target.checkpointFolder;
	config.isImported = target.isImported;
	config.buildName = settings.buildName;
	config.includeStaticOpponent = settings.debug.includeStaticOpponent;
	config.staticOpponentSelectionKey = settings.debug.staticOpponentSelectionKey;
	config.playerIndex = settings.debug.playerIndex;
	config.debugSettings = &settings.debug;
	config.showRewards = false;

	// Zero defers team size to debugSettings->teamSizeOverrideIndex, else the stage's weighted mix.
	config.teamSizeOverride = 0;

	// The viewer's speed keys write this live, starting from the configured game speed.
	static atomic<float> timeScale{ 1.f };
	timeScale.store(settings.debug.gameSpeed);
	config.timeScale = &timeScale;

	config.shouldStop = [] { return ConsoleQuitKeyPressed() || StageQueueSignalState::ConsumeStop(); };

	InstallStopAndSkipHandlers();

#ifdef _MSC_VER
	cout << "Rendering; press 'q' to quit.\n";
#else
	cout << "Rendering; Ctrl+C to quit.\n";
#endif

	RunInferenceDebugRender(config);
	NotifyRunFinished();

	return 0;
}

vector<CliCommand> RocketForge::Cli::DebugCommands()
{
	CliCommand debug;
	debug.name = "debug";
	debug.group = "Debug & analysis";
	debug.summary = "Renders a model in RocketSimVis from forge_debug_settings.json, with one-run overrides";
	debug.usage = DEBUG_USAGE;
	debug.options.valueOptions = {
		"--agent", "--build", "--segment", "--imported", "--checkpoint", "--stage", "--state-setter", "--team-size",
		"--player", "--opponent", "--episode-seconds", "--speed", "--file",
	};
	debug.run = RunDebugCommand;

	return { debug };
}
