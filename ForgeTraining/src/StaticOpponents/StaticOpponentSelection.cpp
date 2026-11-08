#include "StaticOpponentSelection.h"

#include "StaticOpponentSelectionInternal.h"

#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"
#include "ForgeTraining.h"
#include "StageQueue/SyncManifest.h"

#include <GigaLearnInfer/AgentRegistry.h>
#include <StageRegistry.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <set>

namespace fs = std::filesystem;

std::string RocketForgeTraining::MakeStaticOpponentSelectionKey(const StaticOpponentSelectionEntry& entry)
{
	return StaticOpponentSelectionInternal::MakeSelectionKey(entry);
}

std::vector<RocketForgeTraining::StaticOpponentCandidate> RocketForgeTraining::ListStaticOpponentCandidates(
	const std::string& targetAgentName)
{
	return ListStaticOpponentCandidates(targetAgentName, GGL::DEFAULT_BUILD_NAME);
}

std::vector<RocketForgeTraining::StaticOpponentCandidate> RocketForgeTraining::ListStaticOpponentCandidates(
	const std::string& targetAgentName,
	const std::string& targetBuildName)
{
	using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

	std::vector<StaticOpponentSelectionEntry> selectedEntries = LoadStaticOpponentSelection(targetAgentName, targetBuildName);
	std::set<std::string> selectedKeys = {};
	for (const StaticOpponentSelectionEntry& entry : selectedEntries)
		selectedKeys.insert(MakeSelectionKey(entry));

	std::vector<StaticOpponentCandidate> candidates = {};
	for (const std::string& agentName : AgentRegistry::GetAvailableAgents()) {
		const fs::path agentModelsFolder = ResolveAgentModelsFolder(agentName);
		BuildStorage probe(agentModelsFolder, GGL::DEFAULT_BUILD_NAME);
		std::vector<BuildSummary> builds = probe.ListBuilds();

		if (builds.empty()) {
			BuildSummary defaultBuild = {};
			defaultBuild.buildName = GGL::DEFAULT_BUILD_NAME;
			defaultBuild.folderName = GGL::DEFAULT_BUILD_NAME;
			defaultBuild.active = true;
			builds.push_back(defaultBuild);
		}

		for (const BuildSummary& build : builds) {
			BuildStorage buildStorage(agentModelsFolder, build.buildName);
			for (const TrainingSegmentMetadata& segment : buildStorage.ListSegments()) {
				const fs::path latestCheckpoint = GGL::TrainingStorage::GetMostRecentCheckpointIn(
					buildStorage.SegmentCheckpointsRoot(segment.folderName));

				if (latestCheckpoint.empty())
					continue;

				StaticOpponentCandidate candidate = {};
				candidate.selection.sourceAgentName = agentName;
				candidate.selection.sourceBuildName = build.buildName;
				candidate.selection.sourceStageName = segment.stageName;
				candidate.selection.sourceVersionId = segment.segmentId;
				candidate.selection.checkpointPath = latestCheckpoint;
				candidate.selection.displayName = MakeDisplayName(candidate.selection);
				candidate.notes = segment.notes;
				candidate.latestTimesteps = segment.endTimestep;
				candidate.selected = selectedKeys.contains(MakeSelectionKey(candidate.selection));
				candidates.push_back(candidate);
			}
		}
	}

	AppendDirectStaticCandidates(targetAgentName, selectedKeys, candidates);

	std::sort(candidates.begin(), candidates.end(), [](const StaticOpponentCandidate& a, const StaticOpponentCandidate& b) {
		if (a.selection.sourceAgentName != b.selection.sourceAgentName)
			return a.selection.sourceAgentName < b.selection.sourceAgentName;
		if (a.selection.sourceBuildName != b.selection.sourceBuildName)
			return a.selection.sourceBuildName < b.selection.sourceBuildName;
		if (a.selection.sourceStageName != b.selection.sourceStageName)
			return a.selection.sourceStageName < b.selection.sourceStageName;
		if (a.selection.sourceVersionId != b.selection.sourceVersionId)
			return a.selection.sourceVersionId < b.selection.sourceVersionId;
		return a.selection.displayName < b.selection.displayName;
	});

	return candidates;
}

std::vector<std::string> RocketForgeTraining::ListStaticOpponentSyncPaths(
	const std::string& targetAgentName,
	const std::string& targetBuildName)
{
	using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

	BuildStorage storage(ResolveAgentModelsFolder(targetAgentName), targetBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : targetBuildName);
	return ListSelectionSyncPaths(LoadRawSelectionFromPath(storage.StaticOpponentsPath()));
}

std::vector<std::string> RocketForgeTraining::StaticOpponentSelectionInternal::ListSelectionSyncPaths(
	const std::vector<StaticOpponentSelectionEntry>& entries)
{
	std::vector<std::string> paths = {};
	std::set<std::string> seen = {};

	// Neither kind of opponent lives inside the target build's synced set,
	// so each entry contributes its own folder.
	for (const StaticOpponentSelectionEntry& entry : entries) {
		const std::string modelsRel = ResolveAgentModelsRelativePath(entry.sourceAgentName);
		if (modelsRel.empty())
			continue;

		const std::string buildName = entry.sourceBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : entry.sourceBuildName;
		const BuildStorage sourceStorage(ResolveAgentModelsFolder(entry.sourceAgentName), buildName);

		std::string path = MakeStaticOpponentSyncPath(entry, modelsRel, sourceStorage);
		if (!path.empty() && seen.insert(path).second)
			paths.push_back(std::move(path));
	}

	return paths;
}

std::string RocketForgeTraining::StaticOpponentSelectionInternal::MakeStaticOpponentSyncPath(
	const StaticOpponentSelectionEntry& entry,
	const std::string& sourceModelsRel,
	const BuildStorage& sourceStorage)
{
	if (entry.sourceVersionId.empty())
		return {};

	const std::string modelsGlob = RcloneGlobLiteral(sourceModelsRel);
	if (IsImportedModelEntry(entry))
		return modelsGlob + "/" + GGL::TrainingStorage::IMPORTED_MODELS_FOLDER_NAME + "/" + RcloneGlobLiteral(entry.sourceVersionId);

	const std::optional<TrainingSegmentMetadata> segment = sourceStorage.LoadSegmentById(entry.sourceVersionId);
	std::string segmentFolderGlob = {};
	if (segment.has_value())
		segmentFolderGlob = RcloneGlobLiteral(segment->folderName);
	else if (!entry.sourceStageName.empty())
		segmentFolderGlob = RcloneGlobLiteral(sourceStorage.MakeSegmentFolderName(entry.sourceVersionId, entry.sourceStageName));
	else
		segmentFolderGlob = RcloneGlobLiteral(entry.sourceVersionId) + "_*";

	const std::string buildFolderGlob = RcloneGlobLiteral(sourceStorage.BuildRoot().filename().string());

	return modelsGlob + "/" + buildFolderGlob + "/" + BuildStorage::SEGMENTS_FOLDER_NAME + "/" + segmentFolderGlob;
}

std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> RocketForgeTraining::LoadStaticOpponentSelection(
	const std::string& targetAgentName)
{
	return LoadStaticOpponentSelection(targetAgentName, GGL::DEFAULT_BUILD_NAME);
}

std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> RocketForgeTraining::LoadStaticOpponentSelection(
	const std::string& targetAgentName,
	const std::string& targetBuildName)
{
	using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

	BuildStorage storage(ResolveAgentModelsFolder(targetAgentName), targetBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : targetBuildName);
	return DeduplicateEntries(FillHandWrittenEntries(LoadRawSelectionFromPath(storage.StaticOpponentsPath())));
}

void RocketForgeTraining::SaveStaticOpponentSelection(
	const std::string& targetAgentName,
	const std::vector<StaticOpponentSelectionEntry>& rawEntries)
{
	SaveStaticOpponentSelection(targetAgentName, GGL::DEFAULT_BUILD_NAME, rawEntries);
}

void RocketForgeTraining::SaveStaticOpponentSelection(
	const std::string& targetAgentName,
	const std::string& targetBuildName,
	const std::vector<StaticOpponentSelectionEntry>& rawEntries)
{
	using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

	BuildStorage storage(ResolveAgentModelsFolder(targetAgentName), targetBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : targetBuildName);
	const fs::path selectionPath = storage.StaticOpponentsPath();
	const std::vector<StaticOpponentSelectionEntry> savedEntries = ResolveEntriesForSave(DeduplicateEntries(rawEntries));

	SaveSelectionToPath(selectionPath, targetAgentName, targetBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : targetBuildName, savedEntries);
}

void RocketForgeTraining::SyncStaticOpponentSelection(const std::string& targetAgentName)
{
	SyncStaticOpponentSelection(targetAgentName, GGL::DEFAULT_BUILD_NAME);
}

void RocketForgeTraining::SyncStaticOpponentSelection(const std::string& targetAgentName, const std::string& targetBuildName)
{
	using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

	const std::string buildName = targetBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : targetBuildName;
	BuildStorage storage(ResolveAgentModelsFolder(targetAgentName), buildName);
	const fs::path selectionPath = storage.StaticOpponentsPath();

	std::vector<StaticOpponentSelectionEntry> resolvable = {};
	std::vector<nlohmann::json> unresolvable = {};
	for (const StoredSelectionEntry& fileEntry : LoadStoredSelectionFromPath(selectionPath)) {
		if (std::optional<StaticOpponentSelectionEntry> filled = FillHandWrittenEntry(fileEntry.entry))
			resolvable.push_back(std::move(filled.value()));
		else
			unresolvable.push_back(fileEntry.stored);
	}

	// An entry this machine can't resolve is written back exactly as stored:
	// its source may be missing only here, and no automatic rewrite removes what the user wrote.
	const std::vector<StaticOpponentSelectionEntry> refreshed = ResolveEntriesForSave(DeduplicateEntries(resolvable));
	SaveSelectionToPath(selectionPath, targetAgentName, buildName, refreshed, unresolvable);
}

std::vector<RocketSim::EngineId> RocketForgeTraining::ResolveStaticOpponentEngines(
	const std::string& targetAgentName,
	const std::string& targetBuildName)
{
	using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

	std::vector<RocketSim::EngineId> engines;
	const std::string buildName = targetBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : targetBuildName;

	for (const SourceVersionSnapshot& source : ResolveSourceVersions(LoadStaticOpponentSelection(targetAgentName, buildName))) {
		for (RocketSim::EngineId engine : RocketForge::RequiredEngines(source.entry.agentSetup.value())) {
			if (std::find(engines.begin(), engines.end(), engine) == engines.end())
				engines.push_back(engine);
		}
	}

	return engines;
}
