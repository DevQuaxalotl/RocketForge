#include "SkillReferenceSelection.h"

#include "StaticOpponentSelectionInternal.h"

#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"

#include <GigaLearnCPP/SkillReferenceRatings.h>
#include <nlohmann/json.hpp>

#include <optional>
#include <stdexcept>

namespace fs = std::filesystem;

using namespace RocketForgeTraining;
using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

namespace
{
	const char* const SKILL_REFERENCE_FILE_KIND = "skill reference";

	// On-disk document for a skill reference file; callers only see its entry list.
	struct SkillReferenceDocument
	{
		int schemaVersion = 1;
		std::string targetAgent;
		std::string targetBuild;
		std::vector<SkillReferenceEntry> skillReferences;

		void Serialize(GGL::JsonArchive& ar)
		{
			ar.Field("schema_version", schemaVersion);
			ar.Field("target_agent", targetAgent);
			ar.Field("target_build", targetBuild);
			ar.Field("skill_references", skillReferences);
		}
	};

	// Deduplicates by source selection; the first entry for a source keeps its ratings.
	std::vector<SkillReferenceEntry> DeduplicateReferences(const std::vector<SkillReferenceEntry>& entries)
	{
		std::vector<StaticOpponentSelectionEntry> selections = {};
		std::map<std::string, std::map<std::string, float>> ratingsByKey = {};
		for (const SkillReferenceEntry& entry : entries) {
			selections.push_back(entry.selection);
			ratingsByKey.try_emplace(MakeSelectionKey(entry.selection), entry.skillRatings);
		}

		std::vector<SkillReferenceEntry> deduped = {};
		for (const StaticOpponentSelectionEntry& selection : DeduplicateEntries(selections))
			deduped.push_back({ selection, ratingsByKey.at(MakeSelectionKey(selection)) });

		return deduped;
	}

	std::vector<StaticOpponentSelectionEntry> SelectionsOf(const std::vector<SkillReferenceEntry>& entries)
	{
		std::vector<StaticOpponentSelectionEntry> selections = {};
		selections.reserve(entries.size());
		for (const SkillReferenceEntry& entry : entries)
			selections.push_back(entry.selection);

		return selections;
	}

	std::string BuildNameOrDefault(const std::string& buildName)
	{
		return buildName.empty() ? std::string(GGL::DEFAULT_BUILD_NAME) : buildName;
	}

	// The last component of a path saved on any OS; fs::path only splits on this OS's separators.
	std::string StoredPathLeaf(const fs::path& storedPath)
	{
		const std::string text = storedPath.string();
		const size_t separator = text.find_last_of("/\\");
		return separator == std::string::npos ? text : text.substr(separator + 1);
	}
}

void RocketForgeTraining::SkillReferenceEntry::Serialize(GGL::JsonArchive& ar)
{
	selection.Serialize(ar);

	// Ratings load through GigaLearn's reader so training and this file agree on which modes are pending.
	if (ar.IsSaving())
		(*ar.Raw())[GGL::SKILL_REFERENCE_RATINGS_KEY] = skillRatings;
	else if (ar.Contains(GGL::SKILL_REFERENCE_RATINGS_KEY))
		skillRatings = GGL::ReadSkillReferenceRatings(ar.RawLoad()->at(GGL::SKILL_REFERENCE_RATINGS_KEY), MakeDisplayName(selection));
}

std::vector<SkillReferenceEntry> RocketForgeTraining::StaticOpponentSelectionInternal::LoadSkillReferencesFromPath(const fs::path& path)
{
	const std::optional<nlohmann::json> j = ReadSelectionFile(path, SKILL_REFERENCE_FILE_KIND);
	if (!j)
		return {};

	SkillReferenceDocument document = {};
	GGL::JsonLoad(*j, document);

	return DeduplicateReferences(document.skillReferences);
}

void RocketForgeTraining::StaticOpponentSelectionInternal::SaveSkillReferencesToPath(
	const fs::path& path,
	const std::string& targetAgentName,
	const std::string& targetBuildName,
	const std::vector<SkillReferenceEntry>& entries)
{
	SkillReferenceDocument document = {};
	document.targetAgent = targetAgentName;
	document.targetBuild = targetBuildName;
	document.skillReferences = DeduplicateReferences(entries);
	RequireResolvedSetups(SelectionsOf(document.skillReferences), "Skill reference");

	nlohmann::json j;
	GGL::JsonSave(document, j);
	WriteSelectionFile(path, j, SKILL_REFERENCE_FILE_KIND);
}

std::vector<SkillReferenceEntry> RocketForgeTraining::MergeSkillReferenceEdits(
	const std::vector<SkillReferenceEntry>& onDisk,
	const std::vector<SkillReferenceEdit>& checkedRows)
{
	std::map<std::string, const SkillReferenceEntry*> onDiskByKey = {};
	for (const SkillReferenceEntry& entry : onDisk)
		onDiskByKey.try_emplace(MakeSelectionKey(entry.selection), &entry);

	std::vector<SkillReferenceEntry> merged = {};
	for (const SkillReferenceEdit& row : checkedRows) {
		SkillReferenceEntry entry = {};
		entry.selection = row.selection;

		const auto found = onDiskByKey.find(MakeSelectionKey(row.selection));
		if (found != onDiskByKey.end())
			entry.skillRatings = found->second->skillRatings;

		for (const auto& [mode, rating] : row.editedRatings) {
			if (rating.has_value())
				entry.skillRatings[mode] = *rating;
			else
				entry.skillRatings.erase(mode);
		}

		merged.push_back(std::move(entry));
	}

	return merged;
}

std::vector<SkillReferenceEntry> RocketForgeTraining::LoadSkillReferenceSelection(
	const std::string& targetAgentName,
	const std::string& targetBuildName)
{
	BuildStorage storage(ResolveAgentModelsFolder(targetAgentName), BuildNameOrDefault(targetBuildName));
	return LoadSkillReferencesFromPath(storage.SkillReferencesPath());
}

void RocketForgeTraining::SaveSkillReferenceSelection(
	const std::string& targetAgentName,
	const std::string& targetBuildName,
	const std::vector<SkillReferenceEntry>& entries)
{
	const std::string buildName = BuildNameOrDefault(targetBuildName);
	BuildStorage storage(ResolveAgentModelsFolder(targetAgentName), buildName);

	const fs::path path = storage.SkillReferencesPath();
	const std::vector<SkillReferenceEntry> onDisk = LoadSkillReferencesFromPath(path);
	SaveSkillReferencesToPath(path, targetAgentName, buildName, PinSkillReferencesForSave(onDisk, entries, ResolveEntriesForSave));
}

std::vector<SkillReferenceEntry> RocketForgeTraining::StaticOpponentSelectionInternal::PinSkillReferencesForSave(
	const std::vector<SkillReferenceEntry>& onDisk,
	const std::vector<SkillReferenceEntry>& entries,
	const SelectionResolver& resolveNew)
{
	std::map<std::string, const StaticOpponentSelectionEntry*> pinnedByKey = {};
	for (const SkillReferenceEntry& entry : onDisk) {
		if (!entry.selection.checkpointPath.empty() && entry.selection.agentSetup.has_value())
			pinnedByKey.try_emplace(MakeSelectionKey(entry.selection), &entry.selection);
	}

	std::vector<SkillReferenceEntry> pinned = DeduplicateReferences(entries);
	std::vector<size_t> newIndices = {};
	std::vector<StaticOpponentSelectionEntry> newSelections = {};
	for (size_t i = 0; i < pinned.size(); i++) {
		const auto found = pinnedByKey.find(MakeSelectionKey(pinned[i].selection));
		if (found == pinnedByKey.end()) {
			newIndices.push_back(i);
			newSelections.push_back(pinned[i].selection);
			continue;
		}

		pinned[i].selection.checkpointPath = found->second->checkpointPath;
		pinned[i].selection.agentSetup = found->second->agentSetup;
	}

	const std::vector<StaticOpponentSelectionEntry> resolved = newSelections.empty() ? newSelections : resolveNew(newSelections);
	for (size_t i = 0; i < newIndices.size(); i++)
		pinned[newIndices[i]].selection = resolved[i];

	return pinned;
}

std::vector<SkillReferenceEntry> RocketForgeTraining::StaticOpponentSelectionInternal::RelocateSkillReferences(
	const std::vector<SkillReferenceEntry>& entries,
	const SourceResolver& resolveSource)
{
	std::vector<SkillReferenceEntry> relocated = entries;
	for (SkillReferenceEntry& entry : relocated) {
		const SourceVersionSnapshot source = resolveSource(entry.selection);
		const std::string pinnedCheckpoint = StoredPathLeaf(entry.selection.checkpointPath);
		if (IsDirectStaticSelectionEntry(entry.selection) || pinnedCheckpoint.empty()) {
			entry.selection.checkpointPath = source.latestCheckpoint;
			continue;
		}

		const fs::path localCheckpoint = source.latestCheckpoint.parent_path() / pinnedCheckpoint;
		if (!fs::exists(localCheckpoint))
			throw std::runtime_error("Skill reference checkpoint not found: " + MakeDisplayName(entry.selection) + " (" + localCheckpoint.string() + ")");

		entry.selection.checkpointPath = localCheckpoint;
	}

	return relocated;
}

void RocketForgeTraining::SyncSkillReferenceSelection(const std::string& targetAgentName, const std::string& targetBuildName)
{
	const std::string buildName = BuildNameOrDefault(targetBuildName);
	BuildStorage storage(ResolveAgentModelsFolder(targetAgentName), buildName);

	const fs::path path = storage.SkillReferencesPath();
	const std::vector<SkillReferenceEntry> entries = LoadSkillReferencesFromPath(path);
	if (entries.empty())
		return;

	SaveSkillReferencesToPath(path, targetAgentName, buildName, RelocateSkillReferences(entries, ResolveSourceVersion));
}

std::vector<std::string> RocketForgeTraining::ListOpponentModelSyncPaths(
	const std::string& targetAgentName,
	const std::string& targetBuildName)
{
	BuildStorage storage(ResolveAgentModelsFolder(targetAgentName), BuildNameOrDefault(targetBuildName));

	// Raw entries, so an opponent whose source hasn't reached this machine yet still has its folder listed.
	std::vector<StaticOpponentSelectionEntry> selections = LoadRawSelectionFromPath(storage.StaticOpponentsPath());
	const std::vector<StaticOpponentSelectionEntry> references =
		SelectionsOf(LoadSkillReferenceSelection(targetAgentName, targetBuildName));
	selections.insert(selections.end(), references.begin(), references.end());

	return ListSelectionSyncPaths(selections);
}
