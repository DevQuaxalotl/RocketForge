#include "Commands.h"

#include "Cli/CliErrors.h"
#include "Cli/CliResolve.h"

#include "AgentModelPaths.h"
#include "Settings/ForgeSettings.h"
#include "StageQueue/StageQueueRunner.h"
#include "StageQueue/SyncManifest.h"
#include "StaticOpponents/SkillReferenceSelection.h"
#include "Storage/BuildStorage.h"

#include <GigaLearnInfer/JsonArchive.h>

#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>

using namespace std;
using namespace RocketForge;
using namespace RocketForge::Cli;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const size_t MAX_DOCKER_TAG_LENGTH = 128;

	/**
	 * Never creates a missing settings file: plumbing reads settings, and only merge-segment-ids writes back an existing one.
	 * A missing path the user named is a usage error; a missing default file means the agent has no settings yet.
	 */
	ForgeSettings LoadExistingSettings(const fs::path& settingsPath, const string& agentName, bool userNamedPath)
	{
		if (!fs::exists(settingsPath)) {
			if (userNamedPath)
				throw CliUsageError("settings file not found: " + settingsPath.string());
			throw CliRuntimeError("no forge_settings.json for " + agentName + " at " + settingsPath.string());
		}

		const ForgeSettingsLoadResult loaded = LoadForgeSettings(settingsPath);
		if (!loaded.ok) {
			string message = "could not load " + settingsPath.string();
			for (const string& error : loaded.errors)
				message += "\n  " + error;
			throw CliRuntimeError(message);
		}

		return loaded.value;
	}

	/** Valid Docker image tag: [A-Za-z0-9_][A-Za-z0-9_.-]*, at most 128 characters. */
	string SanitizeDockerTag(const string& text)
	{
		string tag;
		for (char c : text) {
			const unsigned char uc = static_cast<unsigned char>(c);
			tag.push_back(std::isalnum(uc) || c == '_' || c == '.' || c == '-' ? c : '-');
		}

		while (!tag.empty() && (tag.front() == '.' || tag.front() == '-'))
			tag.erase(tag.begin());

		if (tag.size() > MAX_DOCKER_TAG_LENGTH)
			tag.resize(MAX_DOCKER_TAG_LENGTH);

		return tag.empty() ? "build" : tag;
	}

	int RunModelsPathCommand(const CliArgs& args)
	{
		const string agentName = ResolveAgent(args);
		PrintResolved(cerr, { { "agent", agentName } });

		cout << ResolveAgentModelsRelativePath(agentName) << "\n";

		return 0;
	}

	int RunBuildInfoCommand(const CliArgs& args)
	{
		const string agentName = ResolveAgent(args);
		const fs::path settingsPath = ResolveSettingsPath(args, agentName);
		const string buildName = ResolveBuild(args, agentName, settingsPath);

		ForgeSettings settings = LoadExistingSettings(settingsPath, agentName, args.Get("--file").has_value());
		settings.agentName = agentName;
		settings.buildName = buildName;

		PrintResolved(cerr, { { "agent", agentName }, { "build", buildName } });

		BuildStorage storage(ResolveAgentModelsFolder(agentName), buildName);
		const string buildFolder = storage.BuildRoot().filename().string();
		const QueueStart start = ResolveQueueStart(settings);

		// The queue's start segment, else the last-trained segment, else a never-trained build.
		string segment = start.resolved ? start.segmentId : string();
		if (segment.empty())
			segment = storage.LoadBuild().activeSegmentId;
		if (segment.empty())
			segment = "new";

		cout << "BUILD_NAME=" << buildName << "\n";
		cout << "BUILD_FOLDER=" << buildFolder << "\n";
		cout << "MODELS_REL=" << ResolveAgentModelsRelativePath(agentName) << "\n";
		cout << "START_STAGE=" << start.stageName << "\n";
		cout << "START_SEGMENT=" << segment << "\n";
		cout << "IS_NEW=" << (start.isNew ? 1 : 0) << "\n";
		cout << "TAG=" << SanitizeDockerTag(buildFolder + "-" + segment) << "\n";

		// Static opponents and skill references live under their own source agent's Models tree,
		// outside the build folder, so a machine syncing this build needs them listed separately.
		const vector<string> staticOpponentPaths = ListOpponentModelSyncPaths(agentName, buildName);
		for (size_t i = 0; i < staticOpponentPaths.size(); i++)
			cout << "STATIC_OPPONENT_" << i << "=" << staticOpponentPaths[i] << "\n";

		return 0;
	}

	int RunSyncManifestCommand(const CliArgs& args)
	{
		if (args.Has("--metadata-only") && args.Has("--temp-checkpoints"))
			throw CliUsageError("--metadata-only and --temp-checkpoints are mutually exclusive");

		SyncManifestScope scope = SyncManifestScope::Full;
		if (args.Has("--metadata-only"))
			scope = SyncManifestScope::MetadataOnly;
		else if (args.Has("--temp-checkpoints"))
			scope = SyncManifestScope::TempCheckpoints;

		const string agentName = ResolveAgent(args);
		const fs::path settingsPath = ResolveSettingsPath(args, agentName);

		// --build is taken as given: a pull lists a build that may so far exist only on the remote.
		const optional<string> buildOverride = args.Get("--build");
		const string buildName = buildOverride.has_value() ? *buildOverride : ResolveBuild(args, agentName, settingsPath);

		ForgeSettings settings = LoadExistingSettings(settingsPath, agentName, args.Get("--file").has_value());
		settings.agentName = agentName;
		settings.buildName = buildName;

		PrintResolved(cerr, { { "agent", agentName }, { "build", buildName } });

		if (settings.stages.empty())
			throw CliRuntimeError("the stage queue for '" + agentName + "' is empty");

		BuildStorage storage(ResolveAgentModelsFolder(agentName), buildName);
		SegmentInspector inspect = [&buildName](const string& agent, const string&, const string& segmentId) {
			return InspectSegmentFromDisk(agent, buildName, segmentId);
		};

		SyncManifestContext context;
		context.modelsRel = ResolveAgentModelsRelativePath(agentName);
		context.staticOpponentGlobs = ListOpponentModelSyncPaths(agentName, buildName);

		const SyncManifest manifest = BuildSyncManifest(settings, storage, inspect, context);
		cout << FormatRcloneFilter(manifest, scope);

		return 0;
	}

	int RunMergeSegmentIdsCommand(const CliArgs& args)
	{
		const optional<string> from = args.Get("--from");
		if (!from.has_value())
			throw CliUsageError("--from <path> is required: the settings file whose recorded segment ids are adopted");

		const string agentName = ResolveAgent(args);
		const fs::path localPath = ResolveSettingsPath(args, agentName);

		PrintResolved(cerr, { { "agent", agentName }, { "settings", localPath.string() } });

		const bool fromIsUserNamed = true;
		ForgeSettings local = LoadExistingSettings(localPath, agentName, args.Get("--file").has_value());
		const ForgeSettings remote = LoadExistingSettings(fs::path(*from), agentName, fromIsUserNamed);

		const int updated = MergeRecordedSegmentIds(local, remote);
		if (updated > 0) {
			string saveError;
			if (!GGL::JsonSaveToFile(local, localPath, &saveError))
				throw CliRuntimeError("failed to save " + localPath.string() + ": " + saveError);
		}

		cout << "Adopted " << updated << " recorded segment id(s) into " << localPath.string() << "\n";

		return 0;
	}

	const char* const MODELS_PATH_USAGE =
		"Usage: ForgeConsoleTrainer models-path [--agent <name>]\n"
		"\n"
		"Prints the agent's Models directory relative to the repo root, as the only line on stdout.\n"
		"Needs no forge_settings.json.\n"
		"\n"
		"Options:\n"
		"  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)\n";

	const char* const BUILD_INFO_USAGE =
		"Usage: ForgeConsoleTrainer build-info [options]\n"
		"\n"
		"Prints KEY=VALUE lines on stdout describing the build a queue run trains:\n"
		"BUILD_NAME, BUILD_FOLDER, MODELS_REL, START_STAGE, START_SEGMENT, IS_NEW, TAG,\n"
		"then one STATIC_OPPONENT_<i> line per opponent model folder outside the build.\n"
		"\n"
		"Options:\n"
		"  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>    Build (default: the build in forge_settings.json)\n"
		"  --file <path>     forge_settings.json to read (default: the agent's; it must exist)\n";

	const char* const SYNC_MANIFEST_USAGE =
		"Usage: ForgeConsoleTrainer sync-manifest [options]\n"
		"\n"
		"Prints an rclone filter list on stdout of the files the build's stage queue needs (for push, pull and pod seed).\n"
		"\n"
		"Options:\n"
		"  --agent <name>        Agent (default: the GUI-selected agent, else example-agent)\n"
		"  --build <name>        Build to list, taken as given (default: the build in forge_settings.json)\n"
		"  --file <path>         forge_settings.json to read (default: the agent's; it must exist)\n"
		"  --metadata-only       List only build and segment metadata\n"
		"  --temp-checkpoints    List only the queue segments' autosaves (temp_checkpoints/)\n";

	const char* const MERGE_SEGMENT_IDS_USAGE =
		"Usage: ForgeConsoleTrainer merge-segment-ids --from <path> [options]\n"
		"\n"
		"Copies the segment ids recorded in another copy of the agent's settings (a remote run's)\n"
		"into the local forge_settings.json, keeping every other local edit.\n"
		"\n"
		"Options:\n"
		"  --from <path>     Settings file whose recorded segment ids are adopted (required)\n"
		"  --agent <name>    Agent (default: the GUI-selected agent, else example-agent)\n"
		"  --file <path>     Local forge_settings.json to update (default: the agent's)\n";

}

vector<CliCommand> RocketForge::Cli::PlumbingCommands()
{
	const string group = "Script plumbing";

	CliCommand modelsPath;
	modelsPath.name = "models-path";
	modelsPath.group = group;
	modelsPath.summary = "Prints the agent's Models directory relative to the repo root";
	modelsPath.usage = MODELS_PATH_USAGE;
	modelsPath.options.valueOptions = { "--agent" };
	modelsPath.run = RunModelsPathCommand;

	CliCommand buildInfo;
	buildInfo.name = "build-info";
	buildInfo.group = group;
	buildInfo.summary = "Prints KEY=VALUE lines describing the build (folder, Models path, start segment, image tag)";
	buildInfo.usage = BUILD_INFO_USAGE;
	buildInfo.options.valueOptions = { "--agent", "--build", "--file" };
	buildInfo.run = RunBuildInfoCommand;

	CliCommand syncManifest;
	syncManifest.name = "sync-manifest";
	syncManifest.group = group;
	syncManifest.summary = "Prints an rclone filter list of the files the build's stage queue needs (for push/pull/seed)";
	syncManifest.usage = SYNC_MANIFEST_USAGE;
	syncManifest.options.valueOptions = { "--agent", "--build", "--file" };
	syncManifest.options.switches = { "--metadata-only", "--temp-checkpoints" };
	syncManifest.run = RunSyncManifestCommand;

	CliCommand mergeSegmentIds;
	mergeSegmentIds.name = "merge-segment-ids";
	mergeSegmentIds.group = group;
	mergeSegmentIds.summary = "Copies the segment ids recorded in another settings file (--from) into the local forge_settings.json";
	mergeSegmentIds.usage = MERGE_SEGMENT_IDS_USAGE;
	mergeSegmentIds.options.valueOptions = { "--agent", "--file", "--from" };
	mergeSegmentIds.run = RunMergeSegmentIdsCommand;

	return { modelsPath, buildInfo, syncManifest, mergeSegmentIds };
}
