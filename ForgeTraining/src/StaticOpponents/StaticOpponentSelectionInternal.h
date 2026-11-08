#pragma once

#include "SkillReferenceSelection.h"
#include "StaticOpponentSelection.h"

#include "Storage/BuildStorage.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace RocketForgeTraining::StaticOpponentSelectionInternal
{
	inline constexpr const char* IMPORTED_MODEL_STAGE_KEY = "__imported_models__";
	inline constexpr const char* DIRECT_IMPORTED_MODEL_STAGE_KEY = IMPORTED_MODEL_STAGE_KEY;
	inline constexpr const char* IMPORTED_MODEL_STAGE_DISPLAY = "Imported Models";

	struct SourceVersionSnapshot
	{
		StaticOpponentSelectionEntry entry;
		std::filesystem::path versionRoot;
		std::filesystem::path latestCheckpoint;
	};

	std::string MakeSelectionKey(const StaticOpponentSelectionEntry& entry);
	std::string MakeDisplayName(const StaticOpponentSelectionEntry& entry);
	StaticOpponentSelectionEntry NormalizeEntry(const StaticOpponentSelectionEntry& entry);
	std::vector<StaticOpponentSelectionEntry> DeduplicateEntries(const std::vector<StaticOpponentSelectionEntry>& entries);
	bool IsDirectStaticSelectionEntry(const StaticOpponentSelectionEntry& entry);

	/** True for an imported-model entry: one marked with the imported stage key, or a hand-written one with an empty source build and no stage. */
	bool IsImportedModelEntry(const StaticOpponentSelectionEntry& entry);

	/** The parsed JSON object of a selection file, or nullopt when there is none. Throws when it exists but is unreadable. */
	std::optional<nlohmann::json> ReadSelectionFile(const std::filesystem::path& path, const std::string& fileKind);

	/** Writes a selection file's JSON, creating its folder. Throws on IO failure. */
	void WriteSelectionFile(const std::filesystem::path& path, const nlohmann::json& document, const std::string& fileKind);

	/** Throws, naming the entry with entryNoun, when an entry has no resolved source-agent setup to save. */
	void RequireResolvedSetups(const std::vector<StaticOpponentSelectionEntry>& entries, const std::string& entryNoun);

	/** The entries exactly as the file stores them: no normalization, deduplication or dropping of incomplete ones. */
	std::vector<StaticOpponentSelectionEntry> LoadRawSelectionFromPath(const std::filesystem::path& path);

	/** One entry as the selection file stores it: parsed, and the JSON object it was parsed from. */
	struct StoredSelectionEntry
	{
		StaticOpponentSelectionEntry entry;
		nlohmann::json stored;
	};

	/** LoadRawSelectionFromPath's entries, each paired with its stored JSON so it can be written back untouched. */
	std::vector<StoredSelectionEntry> LoadStoredSelectionFromPath(const std::filesystem::path& path);

	std::vector<StaticOpponentSelectionEntry> LoadSelectionFromPath(const std::filesystem::path& path);

	/** Writes entries, then each of keptAsStored exactly as given, which needs no resolved setup. */
	void SaveSelectionToPath(
		const std::filesystem::path& path,
		const std::string& targetAgentName,
		const std::string& targetBuildName,
		const std::vector<StaticOpponentSelectionEntry>& entries,
		const std::vector<nlohmann::json>& keptAsStored = {});

	std::vector<SkillReferenceEntry> LoadSkillReferencesFromPath(const std::filesystem::path& path);
	void SaveSkillReferencesToPath(
		const std::filesystem::path& path,
		const std::string& targetAgentName,
		const std::string& targetBuildName,
		const std::vector<SkillReferenceEntry>& entries);

	using SelectionResolver = std::function<std::vector<StaticOpponentSelectionEntry>(const std::vector<StaticOpponentSelectionEntry>&)>;

	/**
	 * The skill references as their file stores them.
	 * A reference already in onDisk keeps its stored checkpoint and setup, because its MMR was measured against that checkpoint;
	 * only references new to the file go through resolveNew.
	 */
	std::vector<SkillReferenceEntry> PinSkillReferencesForSave(
		const std::vector<SkillReferenceEntry>& onDisk,
		const std::vector<SkillReferenceEntry>& entries,
		const SelectionResolver& resolveNew);

	/**
	 * Repo-relative folder (forward slashes, rclone glob syntax with literal names escaped) holding one selected opponent's model files:
	 * the imported-model folder for imported entries, the source segment folder otherwise.
	 * sourceStorage is the source agent's build; a segment not present there is named by prediction,
	 * as "<id>_*" when the entry names no stage.
	 * Works from the entry's own fields, so it names the folder of an entry that can't be resolved on this machine.
	 * Returns empty for entries with no version id.
	 */
	std::string MakeStaticOpponentSyncPath(
		const StaticOpponentSelectionEntry& entry,
		const std::string& sourceModelsRel,
		const BuildStorage& sourceStorage);

	/** Repo-relative model folder globs for the entries, in entry order with duplicates removed (see MakeStaticOpponentSyncPath). */
	std::vector<std::string> ListSelectionSyncPaths(const std::vector<StaticOpponentSelectionEntry>& entries);

	SourceVersionSnapshot ResolveSourceVersion(const StaticOpponentSelectionEntry& rawEntry);
	std::vector<SourceVersionSnapshot> ResolveSourceVersions(const std::vector<StaticOpponentSelectionEntry>& entries);

	using SourceResolver = std::function<SourceVersionSnapshot(const StaticOpponentSelectionEntry&)>;

	/**
	 * The skill references with each stored checkpoint path rebuilt under its source as resolveSource finds it on this machine.
	 * A segment reference keeps its pinned checkpoint folder, since its MMR was measured against it; an imported model takes its local folder.
	 * Throws when a pinned checkpoint is not present locally.
	 */
	std::vector<SkillReferenceEntry> RelocateSkillReferences(
		const std::vector<SkillReferenceEntry>& entries,
		const SourceResolver& resolveSource);

	/** The entries as a selection file stores them: each with its resolved source setup and the source's latest checkpoint. */
	std::vector<StaticOpponentSelectionEntry> ResolveEntriesForSave(const std::vector<StaticOpponentSelectionEntry>& entries);

	/**
	 * Fills, in memory only, whatever a hand-written entry leaves out from the source agent's storage:
	 * stage name, checkpoint path and agent setup.
	 * An entry with an empty source build is an imported-model folder, any other is a segment.
	 * A complete entry passes through untouched; one that cannot be resolved yields nullopt with a warning on stderr.
	 */
	std::optional<StaticOpponentSelectionEntry> FillHandWrittenEntry(const StaticOpponentSelectionEntry& entry);

	/** FillHandWrittenEntry over every entry, dropping the ones that cannot be resolved. */
	std::vector<StaticOpponentSelectionEntry> FillHandWrittenEntries(const std::vector<StaticOpponentSelectionEntry>& entries);

	void AppendDirectStaticCandidates(
		const std::string& targetAgentName,
		const std::set<std::string>& selectedKeys,
		std::vector<StaticOpponentCandidate>& candidates);
}
