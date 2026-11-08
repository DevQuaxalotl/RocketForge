#include "StaticOpponentSelectionInternal.h"

#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"
#include "Storage/ImportedModelStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"

#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <GigaLearnInfer/AgentRegistry.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>

namespace fs = std::filesystem;

namespace
{
	// The model folder's explicit agent_setup.json, or nullopt when it has none. Throws when the file exists but is invalid.
	std::optional<RocketForge::AgentSetupSnapshot> LoadAgentSetupFile(const fs::path& path)
	{
		using nlohmann::json;

		if (path.empty() || !fs::exists(path))
			return std::nullopt;

		std::ifstream fIn(path);
		const json j = fIn.good() ? json::parse(fIn, nullptr, false) : json();
		std::optional<RocketForge::AgentSetupSnapshot> setup = j.is_object() ? RocketForge::AgentSetupSnapshotFromJson(j) : std::nullopt;
		if (!setup)
			throw std::runtime_error("Cannot read static opponent agent setup " + path.string());

		return setup;
	}

	RocketForge::AgentSetupSnapshot CaptureSourceAgentSetup(const RocketForgeTraining::StaticOpponentSelectionEntry& entry)
	{
		const std::vector<std::string> registeredAgents = AgentRegistry::GetAvailableAgents();
		if (std::find(registeredAgents.begin(), registeredAgents.end(), entry.sourceAgentName) == registeredAgents.end())
			throw std::runtime_error("Static opponent source agent is not registered: " + entry.sourceAgentName);

		std::unique_ptr<Agent> agent = AgentRegistry::Create(entry.sourceAgentName);
		if (!agent)
			throw std::runtime_error("Failed to create static opponent source agent: " + entry.sourceAgentName);

		return RocketForge::CaptureAgentSetupSnapshot(*agent);
	}

	// A saved segment runs with its training_setup.json, or its recorded engine (v2 when none) for one saved without a setup.
	// An imported model has no training history, so a setup captured when it was selected stands in for files it lacks.
	RocketForge::AgentSetupSnapshot ResolveAgentSetup(
		const RocketForgeTraining::StaticOpponentSelectionEntry& entry,
		const fs::path& versionRoot,
		const fs::path& modelFolder,
		bool isSegment)
	{
		if (std::optional<RocketForge::AgentSetupSnapshot> setup = LoadAgentSetupFile(modelFolder / "agent_setup.json"))
			return setup.value();

		const std::optional<RocketForge::TrainingSetupSnapshot> saved =
			RocketForge::LoadExistingTrainingSetupSnapshot(versionRoot / RocketForge::TrainingSetupSnapshot::FILE_NAME);
		if (isSegment || saved)
			return RocketForge::ResolveSavedAgentSetup(saved, CaptureSourceAgentSetup(entry));

		if (entry.agentSetup.has_value())
			return entry.agentSetup.value();
		return CaptureSourceAgentSetup(entry);
	}

	uint64_t ReadStaticTimesteps(const fs::path& modelFolder)
	{
		using nlohmann::json;

		const fs::path statsPath = modelFolder / "RUNNING_STATS.json";
		if (!fs::exists(statsPath))
			return 0;

		std::ifstream fIn(statsPath);
		if (!fIn.good())
			return 0;

		const json j = json::parse(fIn, nullptr, false);
		if (!j.is_object())
			return 0;

		return j.value("total_timesteps", static_cast<uint64_t>(0));
	}

	fs::path ResolveDirectStaticSourceRoot(const RocketForgeTraining::StaticOpponentSelectionEntry& entry)
	{
		using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

		if (!IsDirectStaticSelectionEntry(entry))
			return {};

		return RocketForgeTraining::ResolveImportedModelRoot(entry.sourceAgentName, entry.sourceVersionId);
	}
}

RocketForgeTraining::StaticOpponentSelectionInternal::SourceVersionSnapshot RocketForgeTraining::StaticOpponentSelectionInternal::ResolveSourceVersion(
	const StaticOpponentSelectionEntry& rawEntry)
{
	StaticOpponentSelectionEntry entry = NormalizeEntry(rawEntry);

	if (IsDirectStaticSelectionEntry(entry)) {
		const fs::path versionRoot = ResolveDirectStaticSourceRoot(entry);
		if (versionRoot.empty() || !fs::exists(versionRoot))
			throw std::runtime_error("Imported model folder not found: " + MakeDisplayName(entry));

		const fs::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(versionRoot);
		if (modelFolder.empty())
			throw std::runtime_error("Imported model folder has no loadable models: " + MakeDisplayName(entry));

		const bool isSegment = false;
		entry.agentSetup = ResolveAgentSetup(entry, versionRoot, modelFolder, isSegment);

		return { entry, versionRoot, modelFolder };
	}

	if (BuildStorage::TryParseSegmentId(entry.sourceVersionId).has_value()) {
		BuildStorage buildStorage(
			ResolveAgentModelsFolder(entry.sourceAgentName),
			entry.sourceBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : entry.sourceBuildName);
		std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(entry.sourceVersionId);
		if (!segment.has_value())
			throw std::runtime_error("Static opponent source segment not found: " + MakeDisplayName(entry));

		const fs::path latestCheckpoint = GGL::TrainingStorage::GetMostRecentCheckpointIn(
			buildStorage.SegmentCheckpointsRoot(segment->folderName));
		if (latestCheckpoint.empty() || !fs::exists(latestCheckpoint))
			throw std::runtime_error("Static opponent source checkpoint not found: " + MakeDisplayName(entry));

		const fs::path versionRoot = buildStorage.SegmentRoot(segment->folderName);
		const bool isSegment = true;
		entry.agentSetup = ResolveAgentSetup(entry, versionRoot, latestCheckpoint, isSegment);

		return { entry, versionRoot, latestCheckpoint };
	}

	throw std::runtime_error("Static opponent source segment id is invalid: " + MakeDisplayName(entry));
}

std::vector<RocketForgeTraining::StaticOpponentSelectionInternal::SourceVersionSnapshot> RocketForgeTraining::StaticOpponentSelectionInternal::ResolveSourceVersions(
	const std::vector<StaticOpponentSelectionEntry>& entries)
{
	std::vector<SourceVersionSnapshot> snapshots = {};
	snapshots.reserve(entries.size());
	for (const StaticOpponentSelectionEntry& rawEntry : entries)
		snapshots.push_back(ResolveSourceVersion(rawEntry));

	return snapshots;
}

std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> RocketForgeTraining::StaticOpponentSelectionInternal::ResolveEntriesForSave(
	const std::vector<StaticOpponentSelectionEntry>& entries)
{
	std::vector<StaticOpponentSelectionEntry> resolved = {};
	resolved.reserve(entries.size());
	for (const SourceVersionSnapshot& source : ResolveSourceVersions(entries)) {
		StaticOpponentSelectionEntry entry = source.entry;
		entry.checkpointPath = source.latestCheckpoint;
		resolved.push_back(std::move(entry));
	}

	return resolved;
}

std::optional<RocketForgeTraining::StaticOpponentSelectionEntry> RocketForgeTraining::StaticOpponentSelectionInternal::FillHandWrittenEntry(
	const StaticOpponentSelectionEntry& rawEntry)
{
	const bool isComplete = !rawEntry.sourceStageName.empty() && !rawEntry.checkpointPath.empty() && rawEntry.agentSetup.has_value();
	if (isComplete)
		return rawEntry;

	StaticOpponentSelectionEntry entry = rawEntry;
	const bool isImported = IsImportedModelEntry(entry);
	const std::string buildName = entry.sourceBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : entry.sourceBuildName;

	try {
		if (entry.sourceAgentName.empty() || entry.sourceVersionId.empty())
			throw std::runtime_error("it needs source_agent and source_version_id");

		if (isImported) {
			entry.sourceStageName = IMPORTED_MODEL_STAGE_KEY;
		} else if (entry.sourceStageName.empty()) {
			const BuildStorage buildStorage(ResolveAgentModelsFolder(entry.sourceAgentName), buildName);
			const std::optional<TrainingSegmentMetadata> segment = buildStorage.LoadSegmentById(entry.sourceVersionId);
			if (!segment.has_value())
				throw std::runtime_error("no such segment");

			entry.sourceStageName = segment->stageName;
		}

		const SourceVersionSnapshot source = ResolveSourceVersion(entry);
		if (entry.checkpointPath.empty())
			entry.checkpointPath = source.latestCheckpoint;
		if (!entry.agentSetup.has_value())
			entry.agentSetup = source.entry.agentSetup;
		if (entry.displayName.empty())
			entry.displayName = MakeDisplayName(entry);

		return entry;
	} catch (const std::exception& error) {
		const std::string agentFlag = " --agent " + entry.sourceAgentName;
		const std::string listCommand = isImported
			? "ForgeConsoleTrainer list imported-models" + agentFlag
			: "ForgeConsoleTrainer list segments" + agentFlag + " --build " + buildName;

		std::cerr << "[StaticOpponents] Skipping entry " << entry.sourceAgentName << " / " << buildName << " / "
				  << entry.sourceVersionId << ": " << error.what() << ". See `" << listCommand << "` for what exists." << std::endl;
		return std::nullopt;
	}
}

std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> RocketForgeTraining::StaticOpponentSelectionInternal::FillHandWrittenEntries(
	const std::vector<StaticOpponentSelectionEntry>& entries)
{
	std::vector<StaticOpponentSelectionEntry> filled = {};
	filled.reserve(entries.size());

	for (const StaticOpponentSelectionEntry& rawEntry : entries) {
		if (std::optional<StaticOpponentSelectionEntry> entry = FillHandWrittenEntry(rawEntry))
			filled.push_back(std::move(entry.value()));
	}

	return filled;
}

void RocketForgeTraining::StaticOpponentSelectionInternal::AppendDirectStaticCandidates(
	const std::string& targetAgentName,
	const std::set<std::string>& selectedKeys,
	std::vector<StaticOpponentCandidate>& candidates)
{
	(void)targetAgentName;

	for (const std::string& agentName : AgentRegistry::GetAvailableAgents()) {
		std::set<std::string> seenFolderNames = {};
		for (const fs::path& importedRoot : RocketForgeTraining::GetAgentImportedModelSearchFolders(agentName)) {
			if (importedRoot.empty() || !fs::exists(importedRoot))
				continue;

			for (const fs::directory_entry& folderEntry : fs::directory_iterator(importedRoot)) {
				if (!folderEntry.is_directory())
					continue;

				const std::string folderName = folderEntry.path().filename().string();
				if (folderName.empty() || seenFolderNames.contains(folderName))
					continue;

				const fs::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(folderEntry.path());
				if (modelFolder.empty())
					continue;

				seenFolderNames.insert(folderName);
				std::optional<ImportedModelMetadata> metadata = RocketForgeTraining::LoadImportedModelMetadata(folderEntry.path());

				StaticOpponentCandidate candidate = {};
				candidate.selection.sourceAgentName = agentName;
				candidate.selection.sourceBuildName = metadata ? metadata->sourceBuildName : std::string();
				candidate.selection.sourceStageName = DIRECT_IMPORTED_MODEL_STAGE_KEY;
				candidate.selection.sourceVersionId = folderName;
				candidate.selection.displayName = metadata && !metadata->displayName.empty()
					? metadata->displayName
					: MakeDisplayName(candidate.selection);
				candidate.notes = metadata && !metadata->notes.empty() ? metadata->notes : std::string("Imported model folder");
				candidate.latestTimesteps = metadata && metadata->latestTimesteps > 0
					? metadata->latestTimesteps
					: ReadStaticTimesteps(modelFolder);
				candidate.selected = selectedKeys.contains(MakeSelectionKey(candidate.selection));
				candidates.push_back(std::move(candidate));
			}
		}
	}
}
