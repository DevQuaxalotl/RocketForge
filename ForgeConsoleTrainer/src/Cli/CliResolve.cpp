#include "CliResolve.h"

#include "CliErrors.h"
#include "ClosestMatches.h"

#include "AgentModelPaths.h"
#include "Settings/ForgeSettings.h"
#include "StageRegistry.h"
#include "TrainingSetup/VersionManagement.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/AgentRegistry.h>

#include <algorithm>

using namespace std;
using namespace RocketForge;
using namespace RocketForgeTraining;

namespace fs = std::filesystem;

namespace {

	const char* const DEFAULT_AGENT_NAME = "example-agent";

	vector<ResolvedCheckpoint> ListCheckpointsIn(const fs::path& root, bool isAutosave)
	{
		vector<ResolvedCheckpoint> checkpoints;
		for (int64_t timesteps : GGL::TrainingStorage::FindNumberedDirs(root))
			checkpoints.push_back({ root, root / to_string(timesteps), static_cast<uint64_t>(timesteps), isAutosave });

		return checkpoints;
	}

}

string RocketForge::Cli::RequireKnownName(const string& kind, const string& value, const vector<string>& known, const string& listCommand)
{
	if (std::find(known.begin(), known.end(), value) != known.end())
		return value;

	string message = "unknown " + kind + " '" + value + "'.";

	const vector<string> suggestions = ClosestMatches(value, known);
	if (!suggestions.empty()) {
		message += " Did you mean: ";
		for (size_t i = 0; i < suggestions.size(); i++)
			message += (i == 0 ? "" : ", ") + suggestions[i];
		message += "?";
	}

	message += " Run `ForgeConsoleTrainer " + listCommand + "`";
	throw CliUsageError(message);
}

string RocketForge::Cli::DefaultAgentName(const string& savedAgent)
{
	return savedAgent.empty() ? DEFAULT_AGENT_NAME : savedAgent;
}

string RocketForge::Cli::ResolveAgentName(const CliArgs& args, const string& savedAgent, const vector<string>& known)
{
	const string agentName = args.Get("--agent").value_or(DefaultAgentName(savedAgent));

	return RequireKnownName("agent", agentName, known, "list agents");
}

string RocketForge::Cli::ResolveAgent(const CliArgs& args)
{
	return ResolveAgentName(args, LoadLastSelectedAgent(), AgentRegistry::GetAvailableAgents());
}

fs::path RocketForge::Cli::ResolveSettingsPath(const CliArgs& args, const string& agentName)
{
	const optional<string> file = args.Get("--file");
	if (!file.has_value())
		return GetDefaultForgeSettingsPath(agentName);

	const fs::path path = *file;
	return path.is_relative() ? fs::current_path() / path : path;
}

string RocketForge::Cli::ResolveStage(const CliArgs& args, bool required)
{
	const optional<string> stage = args.Get("--stage");
	if (!stage.has_value()) {
		if (required)
			throw CliUsageError("--stage is required. Run `ForgeConsoleTrainer list stages`");
		return {};
	}

	return RequireKnownName("stage", *stage, StageRegistry::GetAvailableStages(), "list stages");
}

string RocketForge::Cli::ResolveBuild(const CliArgs& args, const string& agentName, const fs::path& settingsPath)
{
	const optional<string> build = args.Get("--build");
	if (build.has_value()) {
		vector<string> buildNames;
		for (const BuildSummary& summary : ListAgentBuilds(agentName))
			buildNames.push_back(summary.buildName);

		return RequireKnownName("build", *build, buildNames, "list builds --agent " + agentName);
	}

	// No default settings file means the agent hasn't chosen a build yet; any other failure must not guess one.
	if (!fs::exists(settingsPath)) {
		if (args.Get("--file").has_value())
			throw CliUsageError("settings file not found: " + settingsPath.string());
		return GGL::DEFAULT_BUILD_NAME;
	}

	const ForgeSettingsLoadResult settings = LoadForgeSettings(settingsPath);
	if (!settings.ok) {
		string message = "could not load " + settingsPath.string();
		for (const string& error : settings.errors)
			message += "\n  " + error;
		throw CliRuntimeError(message);
	}

	return settings.value.buildName.empty() ? string(GGL::DEFAULT_BUILD_NAME) : settings.value.buildName;
}

GGL::SegmentVersionSummary RocketForge::Cli::ResolveSegment(const CliArgs& args, const string& agentName, const string& buildName)
{
	string segmentId = args.Get("--segment").value_or("");
	if (segmentId.empty()) {
		segmentId = BuildStorage(ResolveAgentModelsFolder(agentName), buildName).LoadBuild().activeSegmentId;
		if (segmentId.empty())
			throw CliRuntimeError("build '" + buildName + "' has no active segment; train it first or pass --segment");
	}

	const bool includeInactive = true;
	const vector<GGL::SegmentVersionSummary> segments = ListBuildVersions(agentName, buildName, {}, includeInactive);

	vector<string> segmentIds;
	for (const GGL::SegmentVersionSummary& segment : segments)
		segmentIds.push_back(segment.id);

	RequireKnownName("segment", segmentId, segmentIds, "list segments --agent " + agentName + " --build " + buildName);

	return *std::find_if(segments.begin(), segments.end(), [&](const GGL::SegmentVersionSummary& segment) { return segment.id == segmentId; });
}

ResolvedCheckpoint RocketForge::Cli::ResolveCheckpoint(const CliArgs& args, const fs::path& segmentRoot)
{
	const optional<uint64_t> requested = args.GetUint64("--checkpoint");
	if (!requested.has_value()) {
		const optional<ResolvedCheckpoint> latest = BuildStorage::LatestSegmentCheckpointIn(segmentRoot);
		if (!latest.has_value())
			throw CliRuntimeError("segment folder '" + segmentRoot.filename().string() + "' has no checkpoints");
		return *latest;
	}

	// Durable checkpoints come first so they win over an autosave at the same timesteps.
	const bool isAutosave = true;
	vector<ResolvedCheckpoint> checkpoints = ListCheckpointsIn(segmentRoot / BuildStorage::CHECKPOINTS_FOLDER_NAME, !isAutosave);
	const vector<ResolvedCheckpoint> autosaves = ListCheckpointsIn(segmentRoot / BuildStorage::TEMP_CHECKPOINTS_FOLDER_NAME, isAutosave);
	checkpoints.insert(checkpoints.end(), autosaves.begin(), autosaves.end());

	for (const ResolvedCheckpoint& checkpoint : checkpoints) {
		if (checkpoint.timesteps == *requested)
			return checkpoint;
	}

	vector<string> timesteps;
	for (const ResolvedCheckpoint& checkpoint : checkpoints)
		timesteps.push_back(to_string(checkpoint.timesteps));

	RequireKnownName("checkpoint", to_string(*requested), timesteps, "list checkpoints");
	return {};
}

void RocketForge::Cli::PrintResolved(ostream& out, const vector<pair<string, string>>& fields)
{
	for (size_t i = 0; i < fields.size(); i++)
		out << (i == 0 ? "" : "  ") << fields[i].first << ": " << fields[i].second;

	out << "\n";
}
