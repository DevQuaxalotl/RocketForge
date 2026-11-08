#include "TrainCommand.h"

#include "Commands.h"

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"
#include "Cli/Confirm.h"
#include "Cli/ConsoleControl.h"

#include "StageRegistry.h"
#include "TrainingSetup/VersionManagement.h"

#include <GigaLearnCPP/TrainingStorage.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <tuple>

using namespace std;
using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace {

	const char* const STAGE_OPTION = "--stage";
	const char* const NEW_OPTION = "--new";
	const char* const FROM_STAGE_OPTION = "--from-stage";
	const char* const KEEP_RUN_ID_OPTION = "--keep-run-id";

	string FormatMillions(uint64_t timesteps)
	{
		if (timesteps == UINT64_MAX)
			return "no limit";

		char text[32];
		snprintf(text, sizeof(text), "%.1fM", static_cast<double>(timesteps) / 1'000'000.0);
		return text;
	}

	string FormatPlanEntry(const QueuePlanEntry& entry)
	{
		const string prefix = to_string(entry.stageIndex + 1) + ". " + entry.stageName + "  ";

		switch (entry.kind) {
		case ActionKind::Continue:
			return prefix + "continue " + entry.segmentId + " (" + FormatMillions(entry.currentTimesteps) + " / "
				+ FormatMillions(entry.targetTotalTimesteps) + ")";

		case ActionKind::StartNew: {
			string line = prefix + "new " + entry.segmentId;
			if (!entry.sourceStage.empty()) {
				line += " from " + entry.sourceStage;
				if (!entry.sourceSegmentId.empty())
					line += " " + entry.sourceSegmentId;
			}

			return line;
		}

		case ActionKind::Skip:
			break;
		}

		return prefix + "skip: " + entry.skipReason;
	}

	StageSettings MakeAdHocStage(const CliArgs& args, const string& stageName, const AdHocSegments& segments)
	{
		StageSettings stage;
		stage.stageName = stageName;
		stage.targetTotalTimesteps = UINT64_MAX;
		stage.keepRunId = args.Has(KEEP_RUN_ID_OPTION);

		const optional<string> sourceStage = args.Get(FROM_STAGE_OPTION);
		if (sourceStage) {
			if (segments.sourceSegmentId.empty())
				throw CliUsageError("stage '" + *sourceStage + "' has no saved segment. Run `ForgeConsoleTrainer list segments --stage " + *sourceStage + "`");

			// CreateNew makes the runner start a new segment from the source's checkpoint instead of continuing the source's own.
			stage.bootstrap.mode = BootstrapConfig::Mode::LoadFromOtherStage;
			stage.bootstrap.branchMode = BootstrapConfig::BranchMode::CreateNew;
			stage.bootstrap.sourceStage = *sourceStage;
			stage.bootstrap.sourceVersionId = segments.sourceSegmentId;
		} else if (args.Has(NEW_OPTION) || segments.stageSegmentId.empty()) {
			stage.bootstrap.mode = BootstrapConfig::Mode::New;
		} else {
			stage.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
			stage.bootstrap.selectedExistingVersionId = segments.stageSegmentId;
		}

		return stage;
	}

	/** The id of the highest-versioned segment of stageName in the build, or empty when it has none. */
	string NewestSegmentId(const string& agentName, const string& buildName, const string& stageName)
	{
		const bool includeInactive = true;
		const vector<GGL::SegmentVersionSummary> segments = ListBuildVersions(agentName, buildName, stageName, includeInactive);

		const auto newest = max_element(segments.begin(), segments.end(), [](const auto& left, const auto& right) {
			return tie(left.major, left.minor) < tie(right.major, right.minor);
		});

		return newest == segments.end() ? string() : newest->id;
	}

	/** Blank input, an unrecognised answer, or EOF all accept. */
	bool PromptYesNo(const string& question)
	{
		cout << question << " [Y/n] ";

		string line;
		if (!getline(cin, line))
			return true;

		for (char character : line) {
			if (character == 'n' || character == 'N')
				return false;
			if (character == 'y' || character == 'Y')
				return true;
		}

		return true;
	}

	string NumberedErrors(const vector<string>& errors)
	{
		string text;
		for (size_t i = 0; i < errors.size(); ++i)
			text += "\n  " + to_string(i + 1) + ". " + errors[i];

		return text;
	}

	/** Loads the agent's settings; a missing file yields the example template (or defaults) in memory and writes nothing. */
	ForgeSettings LoadSettingsOrTemplate(const string& agentName, const filesystem::path& settingsPath, bool& outFileExists)
	{
		outFileExists = filesystem::exists(settingsPath);

		const filesystem::path loadPath = outFileExists ? settingsPath : GetExampleForgeSettingsPath(agentName);
		if (!filesystem::exists(loadPath))
			return MakeDefaultForgeSettings(agentName, "");

		ForgeSettingsLoadResult result = LoadForgeSettings(loadPath);
		if (!result.ok)
			throw CliRuntimeError("could not load " + loadPath.string() + NumberedErrors(result.errors));

		return result.value;
	}

	/** First queue run: creates the settings file from the example template, asking first when a person is at the terminal. */
	bool CreateMissingSettingsFile(const string& agentName, const filesystem::path& settingsPath)
	{
		if (StdinIsInteractive()) {
			cout << "No saved settings found for agent [" << agentName << "].\n";

			if (!PromptYesNo("Populate from example defaults and start training?")) {
				cout << "Aborted. Configure the training queue (e.g. via ForgeGUI) and re-run.\n";
				return false;
			}
		} else {
			cout << "No saved settings found for agent [" << agentName
				 << "] and no interactive terminal: creating defaults from the example template.\n";
		}

		const filesystem::path example = GetExampleForgeSettingsPath(agentName);
		if (filesystem::exists(example)) {
			if (settingsPath.has_parent_path())
				filesystem::create_directories(settingsPath.parent_path());

			filesystem::copy_file(example, settingsPath);
		}

		return true;
	}

}

TrainRequest RocketForge::Cli::BuildTrainRequest(
	const CliArgs& args,
	const ForgeSettings& loaded,
	const filesystem::path& settingsPath,
	const AdHocSegments& segments)
{
	const optional<string> stageName = args.Get(STAGE_OPTION);

	if (!stageName) {
		for (const char* bootstrapOption : { NEW_OPTION, FROM_STAGE_OPTION, KEEP_RUN_ID_OPTION }) {
			if (args.Has(bootstrapOption) || args.Get(bootstrapOption))
				throw CliUsageError(string(bootstrapOption) + " only applies to a one-off run: add --stage <name>");
		}
	}

	if (args.Has(NEW_OPTION) && args.Get(FROM_STAGE_OPTION))
		throw CliUsageError("--new and --from-stage can't be combined");

	if (args.Has(KEEP_RUN_ID_OPTION) && !args.Get(FROM_STAGE_OPTION))
		throw CliUsageError("--keep-run-id only applies with --from-stage");

	TrainRequest request;
	request.settings = loaded;
	request.preview = args.Has("--preview");

	if (args.Has("--no-metrics"))
		request.settings.globals.metricsEnabled = false;
	if (args.Has("--performance-cores"))
		request.settings.globals.performanceCoresOnly = true;
	if (args.Has("--prune-source"))
		request.settings.globals.pruneSourceCheckpoints = true;

	if (stageName) {
		request.settings.stages = { MakeAdHocStage(args, *stageName, segments) };
		return request;
	}

	request.persistence.settingsPath = settingsPath;
	request.persistence.persistedGlobals = loaded.globals;

	return request;
}

string RocketForge::Cli::FormatQueuePlan(const vector<QueuePlanEntry>& plan)
{
	string text;
	for (const QueuePlanEntry& entry : plan)
		text += FormatPlanEntry(entry) + "\n";

	return text;
}

int RocketForge::Cli::RunTrainCommand(const CliArgs& args)
{
	const string agentName = ResolveAgent(args);
	const filesystem::path settingsPath = ResolveSettingsPath(args, agentName);
	const string stageName = ResolveStage(args, false);
	const bool adHoc = !stageName.empty();

	// Only the agent's own default file is created from the example template on a first run.
	if (args.Get("--file").has_value() && !filesystem::exists(settingsPath))
		throw CliUsageError("settings file not found: " + settingsPath.string());

	bool settingsFileExists = false;
	ForgeSettings loaded = LoadSettingsOrTemplate(agentName, settingsPath, settingsFileExists);
	loaded.agentName = agentName;

	const string buildName = loaded.buildName.empty() ? string(GGL::DEFAULT_BUILD_NAME) : loaded.buildName;

	AdHocSegments segments;
	if (adHoc) {
		segments.stageSegmentId = NewestSegmentId(agentName, buildName, stageName);

		if (const optional<string> sourceStage = args.Get(FROM_STAGE_OPTION)) {
			RequireKnownName("stage", *sourceStage, StageRegistry::GetAvailableStages(), "list stages");
			segments.sourceSegmentId = NewestSegmentId(agentName, buildName, *sourceStage);
		}
	}

	TrainRequest request = BuildTrainRequest(args, loaded, settingsPath, segments);

	const vector<string> errors = ValidateForgeSettings(request.settings);
	if (!errors.empty())
		throw CliUsageError("invalid settings in " + settingsPath.string() + ":" + NumberedErrors(errors));

	vector<pair<string, string>> resolved = { { "agent", agentName }, { "build", buildName } };
	if (adHoc)
		resolved.push_back({ "stage", stageName + " (one-off, settings file not written)" });
	else
		resolved.push_back({ "settings", settingsPath.string() });

	PrintResolved(cout, resolved);

	if (request.preview) {
		cout << FormatQueuePlan(PlanQueue(request.settings));
		return 0;
	}

	if (!adHoc && !settingsFileExists && !CreateMissingSettingsFile(agentName, settingsPath))
		return 0;

	InstallStopAndSkipHandlers();
	StartTrainingKeyListener();

	RunResult result = RunTraining(request.settings, request.persistence);
	NotifyRunFinished();

	if (result.status == RunStatus::Error)
		throw CliRuntimeError(result.errorMessage);

	return 0;
}

vector<CliCommand> RocketForge::Cli::TrainingCommands()
{
	CliCommand train;
	train.name = "train";
	train.group = "Training";
	train.summary = "Runs the saved stage queue, or one stage forever";
	train.usage =
		"Usage: ForgeConsoleTrainer [train] [options]\n"
		"\n"
		"Runs the agent's saved stage queue from forge_settings.json (created from the example template on first run).\n"
		"With --stage, trains that one stage forever instead of running the queue.\n"
		"A queue run records its own progress (created segment ids, completion) in forge_settings.json;\n"
		"one-off flags and --stage runs never write it.\n"
		"\n"
		"Options:\n"
		"  --agent <name>        Agent to train (default: the GUI-selected agent, else example-agent)\n"
		"  --file <path>         forge_settings.json to use (relative paths resolve from the working directory)\n"
		"  --stage <name>        Train this stage alone, ignoring the queue; continues the stage's newest segment in the build\n"
		"                        (a new segment if the stage has none)\n"
		"  --new                 With --stage: start a new segment from scratch instead of continuing\n"
		"  --from-stage <name>   With --stage: start a new segment of the stage from that stage's newest segment in the build\n"
		"  --keep-run-id         With --from-stage: keep the source run id instead of clearing it\n"
		"  --prune-source        Delete the source checkpoints after copying them (stage bootstrap)\n"
		"  --preview             Print the resolved values and one line per queue entry, then exit without touching storage\n"
		"  --no-metrics          Disable training metrics for this run\n"
		"  --performance-cores   Windows hybrid CPUs: run on the performance cores only, with power throttling off\n"
		"\n"
		"Stopping: the first Ctrl+C saves and stops the whole queue; a second one terminates immediately.\n"
		"In an interactive terminal, Q saves and stops the queue and S saves and skips the current stage.\n"
		"On Linux, `kill -USR1 <pid>` skips the current stage.\n"
		"\n"
		"Run `ForgeConsoleTrainer list stages` for the stage names.\n";
	train.options.valueOptions = { "--agent", "--file", STAGE_OPTION, FROM_STAGE_OPTION };
	train.options.switches = { NEW_OPTION, KEEP_RUN_ID_OPTION, "--prune-source", "--preview", "--no-metrics", "--performance-cores" };
	train.run = RunTrainCommand;

	return { train };
}
