#include "StaticOpponentSelectionInternal.h"

#include <GigaLearnInfer/JsonArchive.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace
{
	const char* const STATIC_OPPONENT_FILE_KIND = "static opponent selection";

	// On-disk document for a static opponent selection file; callers only see its entry list.
	struct StaticOpponentSelectionDocument
	{
		int schemaVersion = 2;
		std::string targetAgent;
		std::string targetBuild;
		std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> staticOpponents;

		void Serialize(GGL::JsonArchive& ar)
		{
			// Saving always writes the current schema version rather than round-tripping the loaded one.
			if (ar.IsSaving())
				(*ar.Raw())["schema_version"] = schemaVersion;
			else
				ar.Field("schema_version", schemaVersion);

			ar.Field("target_agent", targetAgent);
			ar.Field("target_build", targetBuild);
			ar.Field("static_opponents", staticOpponents);
		}
	};
}

void RocketForgeTraining::StaticOpponentSelectionEntry::Serialize(GGL::JsonArchive& ar)
{
	ar.Field("source_agent", sourceAgentName);
	ar.Field("source_build", sourceBuildName, std::string(GGL::DEFAULT_BUILD_NAME));
	ar.Field("source_stage", sourceStageName);
	ar.Field("source_version_id", sourceVersionId);
	ar.Field("display_name", displayName);
	ar.Field("checkpoint_path", checkpointPath);

	// agent_setup goes through the snapshot's own JSON helpers,
	// so a snapshot missing its required component names loads as absent rather than as a half-filled setup.
	if (ar.IsSaving()) {
		(*ar.Raw())["runtime"] = "source_agent";
		if (agentSetup.has_value())
			(*ar.Raw())["agent_setup"] = RocketForge::AgentSetupSnapshotToJson(agentSetup.value());
	} else {
		const nlohmann::json& loaded = *ar.RawLoad();
		if (loaded.contains("agent_setup") && !loaded.at("agent_setup").is_null())
			agentSetup = RocketForge::AgentSetupSnapshotFromJson(loaded.at("agent_setup"));
	}
}

std::string RocketForgeTraining::StaticOpponentSelectionInternal::MakeSelectionKey(const StaticOpponentSelectionEntry& entry)
{
	return entry.sourceAgentName + "\n"
		+ (entry.sourceBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : entry.sourceBuildName) + "\n"
		+ entry.sourceStageName + "\n"
		+ entry.sourceVersionId;
}

std::string RocketForgeTraining::StaticOpponentSelectionInternal::MakeDisplayName(const StaticOpponentSelectionEntry& entry)
{
	if (!entry.sourceAgentName.empty() && !entry.sourceStageName.empty() && !entry.sourceVersionId.empty()) {
		const std::string stageName = IsDirectStaticSelectionEntry(entry) ? IMPORTED_MODEL_STAGE_DISPLAY : entry.sourceStageName;
		const std::string buildName = entry.sourceBuildName.empty() ? GGL::DEFAULT_BUILD_NAME : entry.sourceBuildName;
		return entry.sourceAgentName + " / " + buildName + " / " + stageName + " / " + entry.sourceVersionId;
	}

	return entry.displayName;
}

RocketForgeTraining::StaticOpponentSelectionEntry RocketForgeTraining::StaticOpponentSelectionInternal::NormalizeEntry(
	const StaticOpponentSelectionEntry& entry)
{
	StaticOpponentSelectionEntry normalized = entry;
	if (normalized.sourceBuildName.empty())
		normalized.sourceBuildName = GGL::DEFAULT_BUILD_NAME;
	normalized.displayName = MakeDisplayName(normalized);

	return normalized;
}

std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> RocketForgeTraining::StaticOpponentSelectionInternal::DeduplicateEntries(
	const std::vector<StaticOpponentSelectionEntry>& entries)
{
	std::vector<StaticOpponentSelectionEntry> deduped = {};
	std::set<std::string> seen = {};
	for (const StaticOpponentSelectionEntry& rawEntry : entries) {
		const StaticOpponentSelectionEntry entry = NormalizeEntry(rawEntry);
		if (entry.sourceAgentName.empty() || entry.sourceStageName.empty() || entry.sourceVersionId.empty())
			continue;

		const std::string key = MakeSelectionKey(entry);
		if (seen.contains(key))
			continue;

		seen.insert(key);
		deduped.push_back(entry);
	}

	return deduped;
}

bool RocketForgeTraining::StaticOpponentSelectionInternal::IsDirectStaticSelectionEntry(const StaticOpponentSelectionEntry& entry)
{
	return entry.sourceStageName == IMPORTED_MODEL_STAGE_KEY;
}

bool RocketForgeTraining::StaticOpponentSelectionInternal::IsImportedModelEntry(const StaticOpponentSelectionEntry& entry)
{
	return IsDirectStaticSelectionEntry(entry) || (entry.sourceStageName.empty() && entry.sourceBuildName.empty());
}

std::optional<nlohmann::json> RocketForgeTraining::StaticOpponentSelectionInternal::ReadSelectionFile(
	const fs::path& path,
	const std::string& fileKind)
{
	using nlohmann::json;

	if (!fs::exists(path))
		return std::nullopt;

	std::ifstream fIn(path);
	if (!fIn.good())
		throw std::runtime_error("Failed to open " + fileKind + " file: " + path.string());

	json j = json::parse(fIn, nullptr, false);
	if (j.is_discarded() || !j.is_object())
		throw std::runtime_error("Failed to parse " + fileKind + " file: " + path.string());

	return j;
}

void RocketForgeTraining::StaticOpponentSelectionInternal::WriteSelectionFile(
	const fs::path& path,
	const nlohmann::json& document,
	const std::string& fileKind)
{
	fs::create_directories(path.parent_path());
	std::ofstream fOut(path);
	if (!fOut.good())
		throw std::runtime_error("Failed to write " + fileKind + " file: " + path.string());
	fOut << document.dump(4);
}

void RocketForgeTraining::StaticOpponentSelectionInternal::RequireResolvedSetups(
	const std::vector<StaticOpponentSelectionEntry>& entries,
	const std::string& entryNoun)
{
	for (const StaticOpponentSelectionEntry& entry : entries) {
		if (!entry.agentSetup.has_value())
			throw std::runtime_error(entryNoun + " is missing resolved source-agent setup: " + MakeDisplayName(entry));
	}
}

std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> RocketForgeTraining::StaticOpponentSelectionInternal::LoadRawSelectionFromPath(
	const fs::path& path)
{
	const std::optional<nlohmann::json> j = ReadSelectionFile(path, STATIC_OPPONENT_FILE_KIND);
	if (!j)
		return {};

	StaticOpponentSelectionDocument document = {};
	GGL::JsonLoad(*j, document);

	return document.staticOpponents;
}

std::vector<RocketForgeTraining::StaticOpponentSelectionInternal::StoredSelectionEntry> RocketForgeTraining::StaticOpponentSelectionInternal::LoadStoredSelectionFromPath(
	const fs::path& path)
{
	const std::optional<nlohmann::json> j = ReadSelectionFile(path, STATIC_OPPONENT_FILE_KIND);
	if (!j)
		return {};

	StaticOpponentSelectionDocument document = {};
	GGL::JsonLoad(*j, document);

	// The archive loads the array element by element, so the parsed entries line up with the stored objects.
	std::vector<StoredSelectionEntry> entries = {};
	entries.reserve(document.staticOpponents.size());
	for (size_t i = 0; i < document.staticOpponents.size(); i++)
		entries.push_back({ document.staticOpponents[i], j->at("static_opponents").at(i) });

	return entries;
}

std::vector<RocketForgeTraining::StaticOpponentSelectionEntry> RocketForgeTraining::StaticOpponentSelectionInternal::LoadSelectionFromPath(
	const fs::path& path)
{
	return DeduplicateEntries(LoadRawSelectionFromPath(path));
}

void RocketForgeTraining::StaticOpponentSelectionInternal::SaveSelectionToPath(
	const fs::path& path,
	const std::string& targetAgentName,
	const std::string& targetBuildName,
	const std::vector<StaticOpponentSelectionEntry>& entries,
	const std::vector<nlohmann::json>& keptAsStored)
{
	StaticOpponentSelectionDocument document = {};
	document.targetAgent = targetAgentName;
	document.targetBuild = targetBuildName;
	document.staticOpponents = DeduplicateEntries(entries);
	RequireResolvedSetups(document.staticOpponents, "Static opponent");

	nlohmann::json j;
	GGL::JsonSave(document, j);
	for (const nlohmann::json& stored : keptAsStored)
		j["static_opponents"].push_back(stored);

	WriteSelectionFile(path, j, STATIC_OPPONENT_FILE_KIND);
}
