#include "Commands.h"

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"
#include "Cli/Confirm.h"

#include "AgentModelPaths.h"
#include "ForgeTraining.h"
#include "Settings/ForgeSettings.h"
#include "Storage/BuildStorage.h"
#include "Storage/TrainingBuildLock.h"
#include "TrainingSetup/SubmittalExport/SubmittalExport.h"
#include "TrainingSetup/VersionManagement.h"
#include "TrainingStage.h"

#include <GigaLearnCPP/TrainingStorage.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>

using namespace std;
using namespace RocketForge;
using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const size_t CHECKPOINTS_TO_KEEP = 1;

	struct BuildTarget {
		string agentName;
		string buildName;
	};

	BuildTarget ResolveBuildTarget(const CliArgs& args)
	{
		BuildTarget target;
		target.agentName = ResolveAgent(args);
		target.buildName = ResolveBuild(args, target.agentName, ResolveSettingsPath(args, target.agentName));

		return target;
	}

	/** Holds the build's writer lock, so a running trainer or GUI session keeps the build to itself. */
	unique_ptr<TrainingBuildLock> LockBuild(const BuildTarget& target)
	{
		const fs::path buildRoot = BuildStorage(ResolveAgentModelsFolder(target.agentName), target.buildName).BuildRoot();

		try {
			return make_unique<TrainingBuildLock>(buildRoot);
		} catch (const runtime_error& error) {
			throw CliRuntimeError(string("build is in use by a running trainer (") + error.what() + ")");
		}
	}

	/**
	 * The stage the version functions find segments through.
	 * Scoped to the target build so another build's same-numbered segment is never touched.
	 */
	unique_ptr<TrainingStage> CreateStageFor(const BuildTarget& target, const GGL::SegmentVersionSummary& segment)
	{
		unique_ptr<TrainingStage> stage = CreateTrainingStage(target.agentName, segment.stageName, {}, {});
		if (!stage)
			throw CliRuntimeError("segment " + segment.id + " was trained with stage '" + segment.stageName + "', which is not registered");

		stage->learnerConfig.runtime.buildName = target.buildName;

		return stage;
	}

	string CheckpointCount(size_t count)
	{
		return to_string(count) + (count == 1 ? " checkpoint" : " checkpoints");
	}

	/** Everything DeleteVersion removes: durable checkpoints, autosaves and the RLBot export. */
	string DescribeSegmentContents(const GGL::SegmentVersionSummary& segment)
	{
		const fs::path tempRoot = fs::path(segment.rootPath) / BuildStorage::TEMP_CHECKPOINTS_FOLDER_NAME;
		const size_t autosaves = GGL::TrainingStorage::FindNumberedDirs(tempRoot).size();

		string contents = CheckpointCount(segment.modelCount) + ", " + to_string(autosaves) + (autosaves == 1 ? " autosave" : " autosaves");
		if (segment.eligibleRLBot)
			contents += ", RLBot export";

		return contents;
	}

	fs::path CheckpointsRootOf(const GGL::SegmentVersionSummary& segment)
	{
		return fs::path(segment.rootPath) / BuildStorage::CHECKPOINTS_FOLDER_NAME;
	}

	/** Maps a refused or declined confirmation to its exit code; nothing is changed for either. */
	int ExitCodeFor(ConfirmResult result, ostream& out)
	{
		if (result == ConfirmResult::RefusedNonInteractive)
			return 2;

		out << "Cancelled. Nothing was changed.\n";

		return 1;
	}

	struct PrunePlanEntry {
		GGL::SegmentVersionSummary segment;
		size_t checkpointsToRemove = 0;
	};

	vector<PrunePlanEntry> PlanPrune(const BuildTarget& target, const CliArgs& args)
	{
		vector<GGL::SegmentVersionSummary> segments;
		if (args.Has("--all")) {
			const bool includeInactive = true;
			segments = ListBuildVersions(target.agentName, target.buildName, {}, includeInactive);
		} else {
			segments.push_back(ResolveSegment(args, target.agentName, target.buildName));
		}

		vector<PrunePlanEntry> plan;
		for (const GGL::SegmentVersionSummary& segment : segments) {
			PrunePlanEntry entry;
			entry.segment = segment;
			entry.checkpointsToRemove = GGL::TrainingStorage::GetCheckpointFoldersToPrune(CheckpointsRootOf(segment), CHECKPOINTS_TO_KEEP).size();
			plan.push_back(entry);
		}

		return plan;
	}

	string DescribePrunePlan(const vector<PrunePlanEntry>& plan)
	{
		ostringstream text;
		text << "Prune checkpoints, keeping the newest of each segment:\n";

		for (const PrunePlanEntry& entry : plan) {
			const size_t total = entry.checkpointsToRemove + CHECKPOINTS_TO_KEEP;

			if (entry.checkpointsToRemove == 0)
				text << "  " << entry.segment.id << "  skipped (nothing beyond the newest checkpoint)\n";
			else
				text << "  " << entry.segment.id << "  remove " << entry.checkpointsToRemove << " of " << CheckpointCount(total) << "\n";
		}

		return text.str();
	}

	const char* const DELETE_SEGMENT_USAGE =
		"Usage: ForgeConsoleTrainer delete-segment --segment <id> [options]\n"
		"\n"
		"Permanently deletes one segment, its checkpoints and its RLBot export. Asks to confirm.\n"
		"\n"
		"Options:\n"
		"  --segment <id>    Segment to delete (required; see `list segments`)\n"
		"  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>    Build (default: the build in forge_settings.json)\n"
		"  --file <path>     forge_settings.json that names the default build\n"
		"  --yes             Skip the confirmation (required when stdin is not a terminal)\n"
		"\n"
		"Refuses while a trainer holds the build.\n";

	const char* const PRUNE_USAGE =
		"Usage: ForgeConsoleTrainer prune (--segment <id> | --all) [options]\n"
		"\n"
		"Deletes every checkpoint of a segment except the newest, keeping its metadata. Asks to confirm.\n"
		"\n"
		"Options:\n"
		"  --segment <id>    Segment to prune\n"
		"  --all             Prune every segment of the build\n"
		"  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>    Build (default: the build in forge_settings.json)\n"
		"  --file <path>     forge_settings.json that names the default build\n"
		"  --yes             Skip the confirmation (required when stdin is not a terminal)\n"
		"\n"
		"Refuses while a trainer holds the build.\n";

	const char* const EXPORT_RLBOT_USAGE =
		"Usage: ForgeConsoleTrainer export-rlbot [options]\n"
		"\n"
		"Exports a segment into a self-contained RLBot folder, or with --remove deletes that export.\n"
		"\n"
		"Options:\n"
		"  --segment <id>    Segment to export (default: the build's active segment)\n"
		"  --remove          Remove the segment's RLBot export instead\n"
		"  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>    Build (default: the build in forge_settings.json)\n"
		"  --file <path>     forge_settings.json that names the default build\n";

	const char* const EXPORT_SUBMITTAL_USAGE =
		"Usage: ForgeConsoleTrainer export-submittal --out <folder> [options]\n"
		"\n"
		"Packages a segment for Bob the Bot Builder inside the output folder and prints the package path on stdout.\n"
		"\n"
		"Options:\n"
		"  --out <folder>    Parent folder the package is created in (required)\n"
		"  --overwrite       Replace an existing package\n"
		"  --device <name>   cpu (default; the only device the RLBot bot pack builds) or gpu\n"
		"  --segment <id>    Segment to package (default: the build's active segment)\n"
		"  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>    Build (default: the build in forge_settings.json)\n"
		"  --file <path>     forge_settings.json that names the default build\n";

}

int RocketForge::Cli::RunDeleteSegmentCommand(const CliArgs& args, bool interactive, istream& in, ostream& out)
{
	if (!args.Get("--segment").has_value())
		throw CliUsageError("--segment is required: deleting has no default segment. Run `ForgeConsoleTrainer list segments`");

	const BuildTarget target = ResolveBuildTarget(args);
	ResolveSegment(args, target.agentName, target.buildName);

	const unique_ptr<TrainingBuildLock> lock = LockBuild(target);

	// Read again under the lock: a trainer may have written checkpoints before the lock was free.
	const GGL::SegmentVersionSummary segment = ResolveSegment(args, target.agentName, target.buildName);
	const unique_ptr<TrainingStage> stage = CreateStageFor(target, segment);

	PrintResolved(out, { { "agent", target.agentName }, { "build", target.buildName }, { "segment", segment.id } });

	const string summary =
		"Delete segment " + segment.id + " (" + segment.stageName + ", " + DescribeSegmentContents(segment) + ") at " + segment.rootPath + "?";
	const ConfirmResult confirmed = ConfirmDestructive(summary, args.Has("--yes"), interactive, in, out);
	if (confirmed != ConfirmResult::Confirmed)
		return ExitCodeFor(confirmed, out);

	DeleteVersion(stage.get(), segment.id);
	out << "Deleted segment " << segment.id << ".\n";

	return 0;
}

int RocketForge::Cli::RunDeleteSegmentCommand(const CliArgs& args)
{
	return RunDeleteSegmentCommand(args, StdinIsInteractive(), cin, cerr);
}

int RocketForge::Cli::RunPruneCommand(const CliArgs& args, bool interactive, istream& in, ostream& out)
{
	if (args.Has("--all") == args.Get("--segment").has_value())
		throw CliUsageError("give exactly one of --segment <id> or --all");

	const BuildTarget target = ResolveBuildTarget(args);
	if (args.Get("--segment").has_value())
		ResolveSegment(args, target.agentName, target.buildName);

	const unique_ptr<TrainingBuildLock> lock = LockBuild(target);

	vector<PrunePlanEntry> plan = PlanPrune(target, args);

	// Every stage is created before anything is removed, so an unregistered one fails before a partial prune.
	vector<unique_ptr<TrainingStage>> stages;
	for (const PrunePlanEntry& entry : plan) {
		if (entry.checkpointsToRemove > 0)
			stages.push_back(CreateStageFor(target, entry.segment));
	}

	PrintResolved(out, { { "agent", target.agentName }, { "build", target.buildName } });

	if (stages.empty()) {
		if (args.Has("--all"))
			out << "Nothing to prune: every segment has at most one checkpoint.\n";
		else
			out << "Nothing to prune: segment " << plan.front().segment.id << " has at most one checkpoint.\n";

		return 0;
	}

	const ConfirmResult confirmed = ConfirmDestructive(DescribePrunePlan(plan), args.Has("--yes"), interactive, in, out);
	if (confirmed != ConfirmResult::Confirmed)
		return ExitCodeFor(confirmed, out);

	size_t stageIndex = 0;
	for (const PrunePlanEntry& entry : plan) {
		if (entry.checkpointsToRemove == 0)
			continue;

		PruneVersionModels(stages[stageIndex++].get(), entry.segment.id);
		out << "Pruned " << entry.segment.id << ": removed " << CheckpointCount(entry.checkpointsToRemove) << ".\n";
	}

	return 0;
}

int RocketForge::Cli::RunPruneCommand(const CliArgs& args)
{
	return RunPruneCommand(args, StdinIsInteractive(), cin, cerr);
}

int RocketForge::Cli::RunExportRlbotCommand(const CliArgs& args)
{
	const BuildTarget target = ResolveBuildTarget(args);
	ResolveSegment(args, target.agentName, target.buildName);

	// Both directions rewrite segment.json, so they wait for any trainer to release the build.
	const unique_ptr<TrainingBuildLock> lock = LockBuild(target);

	const GGL::SegmentVersionSummary segment = ResolveSegment(args, target.agentName, target.buildName);

	PrintResolved(cerr, { { "agent", target.agentName }, { "build", target.buildName }, { "segment", segment.id } });

	const bool removing = args.Has("--remove");
	if (!removing && !BuildStorage::LatestSegmentCheckpointIn(segment.rootPath).has_value())
		throw CliRuntimeError("segment " + segment.id + " has no checkpoints to export");

	if (removing) {
		RemoveVersionFromRLBot(target.agentName, target.buildName, segment.id);
		cout << "Removed the RLBot export of " << segment.id << ".\n";
	} else {
		ExportVersionToRLBot(target.agentName, target.buildName, segment.id);
		cout << "Exported " << segment.id << " for RLBot.\n";
	}

	return 0;
}

int RocketForge::Cli::RunExportSubmittalCommand(const CliArgs& args)
{
	const optional<string> outputParent = args.Get("--out");
	if (!outputParent.has_value())
		throw CliUsageError("--out is required: the folder the package is created in");

	const string device = args.Get("--device").value_or("cpu");
	if (device != "cpu" && device != "gpu")
		throw CliUsageError("--device must be cpu or gpu, not '" + device + "'");

	const BuildTarget target = ResolveBuildTarget(args);
	const GGL::SegmentVersionSummary segment = ResolveSegment(args, target.agentName, target.buildName);

	PrintResolved(cerr, { { "agent", target.agentName }, { "build", target.buildName }, { "segment", segment.id }, { "device", device } });

	const bool useGPU = device == "gpu";
	const SubmittalExportResult result =
		ExportVersionForSubmittal(target.agentName, target.buildName, segment.id, fs::path(*outputParent), useGPU, args.Has("--overwrite"));
	cout << result.packageRoot.string() << "\n";

	return 0;
}

vector<CliCommand> RocketForge::Cli::ModelCommands()
{
	CliOptionSpec buildOptions;
	buildOptions.valueOptions = { "--agent", "--build", "--segment", "--file" };

	CliCommand deleteSegment;
	deleteSegment.name = "delete-segment";
	deleteSegment.group = "Models";
	deleteSegment.summary = "Permanently deletes a segment and its checkpoints";
	deleteSegment.usage = DELETE_SEGMENT_USAGE;
	deleteSegment.options = buildOptions;
	deleteSegment.options.switches = { "--yes" };
	deleteSegment.run = [](const CliArgs& args) { return RunDeleteSegmentCommand(args); };

	CliCommand prune;
	prune.name = "prune";
	prune.group = "Models";
	prune.summary = "Deletes all but the newest checkpoint of a segment, or of every segment";
	prune.usage = PRUNE_USAGE;
	prune.options = buildOptions;
	prune.options.switches = { "--all", "--yes" };
	prune.run = [](const CliArgs& args) { return RunPruneCommand(args); };

	CliCommand exportRlbot;
	exportRlbot.name = "export-rlbot";
	exportRlbot.group = "Models";
	exportRlbot.summary = "Exports a segment into a self-contained RLBot folder, or removes the export";
	exportRlbot.usage = EXPORT_RLBOT_USAGE;
	exportRlbot.options = buildOptions;
	exportRlbot.options.switches = { "--remove" };
	exportRlbot.run = RunExportRlbotCommand;

	CliCommand exportSubmittal;
	exportSubmittal.name = "export-submittal";
	exportSubmittal.group = "Models";
	exportSubmittal.summary = "Packages a segment for Bob the Bot Builder";
	exportSubmittal.usage = EXPORT_SUBMITTAL_USAGE;
	exportSubmittal.options = buildOptions;
	exportSubmittal.options.valueOptions.push_back("--out");
	exportSubmittal.options.valueOptions.push_back("--device");
	exportSubmittal.options.switches = { "--overwrite" };
	exportSubmittal.run = RunExportSubmittalCommand;

	return { deleteSegment, prune, exportRlbot, exportSubmittal };
}
