#include "Commands.h"

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"
#include "Cli/TextTable.h"

#include "AgentModelPaths.h"
#include "ForgeTraining.h"
#include "Settings/ForgeSettings.h"
#include "StageRegistry.h"
#include "StaticOpponents/StaticOpponentSelection.h"
#include "Storage/BuildStorage.h"
#include "Storage/ImportedModelStorage.h"
#include "TrainingSetup/VersionManagement.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/AgentRegistry.h>

#include <iostream>
#include <sstream>

using namespace std;
using namespace RocketForge;
using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const MARK = "*";

	const vector<string> SUBJECTS = {
		"agents", "stages", "builds", "segments", "checkpoints", "stage-options", "static-opponents", "imported-models",
	};

	string Mark(bool marked)
	{
		return marked ? MARK : "";
	}

	string FormatWeight(float weight)
	{
		ostringstream text;
		text << weight;
		return text.str();
	}

	string FormatTeamSize(int teamSizeA, int teamSizeB)
	{
		return to_string(teamSizeA) + "v" + to_string(teamSizeB);
	}

	/** The agent and build a subject reads, resolved the way every other command resolves them. */
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

	void SetResolved(ListOutput& output, const BuildTarget& target)
	{
		output.resolved = { { "agent", target.agentName }, { "build", target.buildName } };
	}

	ListOutput ListAgents()
	{
		const string defaultAgent = DefaultAgentName(LoadLastSelectedAgent());

		ListOutput output;
		output.headers = { "name", "default" };

		for (const string& name : AgentRegistry::GetAvailableAgents()) {
			const bool isDefault = name == defaultAgent;
			output.rows.push_back({ name, Mark(isDefault) });
			output.json.push_back({ { "name", name }, { "default", isDefault } });
		}

		return output;
	}

	ListOutput ListStages()
	{
		ListOutput output;
		output.headers = { "name" };

		for (const string& name : StageRegistry::GetAvailableStages()) {
			output.rows.push_back({ name });
			output.json.push_back({ { "name", name } });
		}

		return output;
	}

	ListOutput ListBuilds(const CliArgs& args)
	{
		const string agentName = ResolveAgent(args);

		ListOutput output;
		output.headers = { "name", "folder", "active", "updated", "total timesteps" };
		output.resolved = { { "agent", agentName } };

		for (const BuildSummary& build : ListAgentBuilds(agentName)) {
			const uint64_t totalTimesteps = BuildStorage(ResolveAgentModelsFolder(agentName), build.buildName).LoadBuild().totalTrainedTimesteps;

			output.rows.push_back({ build.buildName, build.folderName, Mark(build.active), build.updatedAt, to_string(totalTimesteps) });
			output.json.push_back({
				{ "name", build.buildName },
				{ "folder", build.folderName },
				{ "active", build.active },
				{ "updated", build.updatedAt },
				{ "total_timesteps", totalTimesteps },
			});
		}

		return output;
	}

	ListOutput ListSegments(const CliArgs& args)
	{
		const BuildTarget target = ResolveBuildTarget(args);
		const string stageFilter = args.Get("--stage").value_or("");

		const string activeSegmentId = BuildStorage(ResolveAgentModelsFolder(target.agentName), target.buildName).LoadBuild().activeSegmentId;

		ListOutput output;
		output.headers = { "id", "stage", "branch", "timesteps", "checkpoints", "last trained", "exported", "active" };
		SetResolved(output, target);
		if (!stageFilter.empty())
			output.resolved.push_back({ "stage", stageFilter });

		const bool includeInactive = true;
		for (const GGL::SegmentVersionSummary& segment : ListBuildVersions(target.agentName, target.buildName, stageFilter, includeInactive)) {
			const bool isActive = segment.id == activeSegmentId;

			output.rows.push_back({
				segment.id,
				segment.stageName,
				segment.branchName,
				to_string(segment.latestTimesteps),
				to_string(segment.modelCount),
				segment.lastTrainedAt,
				Mark(segment.eligibleRLBot),
				Mark(isActive),
			});
			output.json.push_back({
				{ "id", segment.id },
				{ "stage", segment.stageName },
				{ "branch", segment.branchName },
				{ "timesteps", segment.latestTimesteps },
				{ "checkpoints", segment.modelCount },
				{ "last_trained", segment.lastTrainedAt },
				{ "exported", segment.eligibleRLBot },
				{ "active", isActive },
			});
		}

		return output;
	}

	void AddCheckpointRows(ListOutput& output, const fs::path& root, const char* kind)
	{
		for (int64_t timesteps : GGL::TrainingStorage::FindNumberedDirs(root)) {
			const string folder = (root / to_string(timesteps)).string();

			output.rows.push_back({ to_string(timesteps), kind, folder });
			output.json.push_back({ { "timesteps", timesteps }, { "kind", kind }, { "path", folder } });
		}
	}

	ListOutput ListCheckpoints(const CliArgs& args)
	{
		const BuildTarget target = ResolveBuildTarget(args);
		const GGL::SegmentVersionSummary segment = ResolveSegment(args, target.agentName, target.buildName);

		ListOutput output;
		output.headers = { "timesteps", "kind", "path" };
		SetResolved(output, target);
		output.resolved.push_back({ "segment", segment.id });

		const fs::path segmentRoot = segment.rootPath;
		AddCheckpointRows(output, segmentRoot / BuildStorage::CHECKPOINTS_FOLDER_NAME, "checkpoint");
		AddCheckpointRows(output, segmentRoot / BuildStorage::TEMP_CHECKPOINTS_FOLDER_NAME, "autosave");

		return output;
	}

	/** The state setters and team sizes of a stage, as the values `debug --state-setter` and `--team-size` take. */
	ListOutput ListStageOptions(const CliArgs& args)
	{
		const string agentName = ResolveAgent(args);
		const string stageName = ResolveStage(args, true);

		unique_ptr<TrainingStage> stage = CreateTrainingStage(agentName, stageName, GlobalSettings(), StageSettings());
		if (!stage)
			throw CliRuntimeError("could not create stage '" + stageName + "' for agent '" + agentName + "'");

		ListOutput output;
		output.headers = { "kind", "index", "name", "weight" };
		output.resolved = { { "agent", agentName }, { "stage", stageName } };

		nlohmann::json stateSetters = nlohmann::json::array();
		for (const DebugStateSetterOption& option : stage->GetDebugStateSetterOptions()) {
			output.rows.push_back({ "state-setter", to_string(option.index), option.name, FormatWeight(option.weight) });
			stateSetters.push_back({ { "index", option.index }, { "name", option.name }, { "weight", option.weight } });
		}

		nlohmann::json teamSizes = nlohmann::json::array();
		int teamSizeIndex = 0;
		for (const WeightedTeamSize& teamSize : stage->GetWeightedTeamSizes()) {
			const string label = FormatTeamSize(teamSize.teamSizeA, teamSize.teamSizeB);

			output.rows.push_back({ "team-size", to_string(teamSizeIndex), label, FormatWeight(teamSize.weight) });
			teamSizes.push_back({ { "index", teamSizeIndex }, { "name", label }, { "weight", teamSize.weight } });
			teamSizeIndex++;
		}

		output.json = { { "state_setters", stateSetters }, { "team_sizes", teamSizes } };

		return output;
	}

	ListOutput ListStaticOpponents(const CliArgs& args)
	{
		const BuildTarget target = ResolveBuildTarget(args);

		ListOutput output;
		output.headers = { "key", "name", "agent", "build", "version" };
		SetResolved(output, target);

		for (const StaticOpponentSelectionEntry& entry : LoadStaticOpponentSelection(target.agentName, target.buildName)) {
			const string key = MakeStaticOpponentSelectionKey(entry);

			output.rows.push_back({ key, entry.displayName, entry.sourceAgentName, entry.sourceBuildName, entry.sourceVersionId });
			output.json.push_back({
				{ "key", key },
				{ "name", entry.displayName },
				{ "agent", entry.sourceAgentName },
				{ "build", entry.sourceBuildName },
				{ "version", entry.sourceVersionId },
			});
		}

		return output;
	}

	ListOutput ListImportedModels(const CliArgs& args)
	{
		const string agentName = ResolveAgent(args);

		ListOutput output;
		output.headers = { "folder", "name", "usable for inference" };
		output.resolved = { { "agent", agentName } };

		for (const DiscoveredImportedModel& model : EnumerateImportedModels(agentName)) {
			output.rows.push_back({ model.folderName, model.metadata.displayName, Mark(model.canUseForInference) });
			output.json.push_back({
				{ "folder", model.folderName },
				{ "name", model.metadata.displayName },
				{ "usable_for_inference", model.canUseForInference },
			});
		}

		return output;
	}

	const char* const LIST_USAGE =
		"Usage: ForgeConsoleTrainer list <subject> [options]\n"
		"\n"
		"Subjects:\n"
		"  agents            Registered agents; * marks the default for --agent\n"
		"  stages            Registered training stages\n"
		"  builds            The agent's builds, with total timesteps\n"
		"  segments          The build's segments; * marks the active segment and the ones exported to RLBot\n"
		"  checkpoints       The segment's checkpoints and autosaves\n"
		"  stage-options     A stage's state setters and team sizes, the values `debug --state-setter` and `--team-size` take\n"
		"                    (team-size index = position in the stage's weighted list; -1 in forge_debug_settings.json means no override)\n"
		"  static-opponents  The build's static opponent selection, with the keys `debug --opponent` takes\n"
		"  imported-models   The agent's imported models, with the folder names `--imported` takes\n"
		"\n"
		"Options:\n"
		"  --agent <name>    Agent to read (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>    Build to read (default: the build in forge_settings.json)\n"
		"  --segment <id>    Segment for `checkpoints` (default: the build's active segment)\n"
		"  --stage <name>    Stage for `stage-options` (required); filters `segments`\n"
		"  --file <path>     forge_settings.json that names the default build\n"
		"  --json            Print JSON on stdout and the resolved values on stderr\n";

}

ListOutput RocketForge::Cli::ListSubject(const string& subject, const CliArgs& args)
{
	const string known = RequireKnownName("subject", subject, SUBJECTS, "list --help");

	if (known == "agents")
		return ListAgents();
	if (known == "stages")
		return ListStages();
	if (known == "builds")
		return ListBuilds(args);
	if (known == "segments")
		return ListSegments(args);
	if (known == "checkpoints")
		return ListCheckpoints(args);
	if (known == "stage-options")
		return ListStageOptions(args);
	if (known == "static-opponents")
		return ListStaticOpponents(args);

	return ListImportedModels(args);
}

int RocketForge::Cli::RunListCommand(const CliArgs& args)
{
	if (args.Positionals().empty())
		throw CliUsageError("missing subject. Run `ForgeConsoleTrainer list --help`");

	const ListOutput output = ListSubject(args.Positionals()[0], args);

	// JSON mode keeps stdout machine-readable, so the resolved values go to stderr.
	ostream& resolvedStream = args.Has("--json") ? cerr : cout;
	if (!output.resolved.empty())
		PrintResolved(resolvedStream, output.resolved);

	if (args.Has("--json"))
		cout << output.json.dump(2) << "\n";
	else
		cout << FormatTable(output.headers, output.rows);

	return 0;
}

vector<CliCommand> RocketForge::Cli::InspectCommands()
{
	CliCommand list;
	list.name = "list";
	list.group = "Inspect";
	list.summary = "Lists agents, stages, builds, segments, checkpoints, stage options, static opponents or imported models";
	list.usage = LIST_USAGE;
	list.options.valueOptions = { "--agent", "--build", "--segment", "--stage", "--file" };
	list.options.switches = { "--json" };
	list.options.maxPositionals = 1;
	list.run = RunListCommand;

	return { list };
}
