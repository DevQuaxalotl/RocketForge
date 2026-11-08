#pragma once

#include "StaticOpponentSelection.h"

#include <GigaLearnInfer/JsonArchive.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace RocketForgeTraining
{
	/** A skill reference: a static-opponent-style source plus its MMR per mode. Absent modes are pending calibration. */
	struct SkillReferenceEntry
	{
		StaticOpponentSelectionEntry selection;
		std::map<std::string, float> skillRatings;

		/** Writes the selection's fields flat in the entry object, beside a "skill_ratings" mode-to-MMR object. */
		void Serialize(GGL::JsonArchive& ar);
	};

	/** A checked picker row: the source plus the MMR cells the user touched; an untouched mode is absent from the map. */
	struct SkillReferenceEdit
	{
		StaticOpponentSelectionEntry selection;

		/** A typed value replaces the stored rating; nullopt means the user cleared the cell. */
		std::map<std::string, std::optional<float>> editedRatings;
	};

	/**
	 * Builds the list to save from the picker's checked rows and the references currently on disk.
	 * Training writes calibrated ratings into that file while the picker is open,
	 * so a mode the user did not edit keeps its on-disk rating, and an unchecked source is dropped.
	 */
	std::vector<SkillReferenceEntry> MergeSkillReferenceEdits(
		const std::vector<SkillReferenceEntry>& onDisk,
		const std::vector<SkillReferenceEdit>& checkedRows);

	/** Loads the Build's skill references; a hand-edited rating that is not a valid MMR loads as pending. */
	std::vector<SkillReferenceEntry> LoadSkillReferenceSelection(const std::string& targetAgentName, const std::string& targetBuildName);

	/**
	 * Saves the Build's skill references.
	 * A reference already in the file keeps its pinned checkpoint; a new one is resolved the way SaveStaticOpponentSelection does.
	 */
	void SaveSkillReferenceSelection(
		const std::string& targetAgentName,
		const std::string& targetBuildName,
		const std::vector<SkillReferenceEntry>& entries);

	/**
	 * Rewrites the Build's skill reference file so each stored checkpoint path points into this machine's copy of its source.
	 * The file stores absolute paths from the machine that saved it, so run this before training reads it.
	 * Does nothing when the Build has no skill reference file.
	 */
	void SyncSkillReferenceSelection(const std::string& targetAgentName, const std::string& targetBuildName);

	/** The Build's static-opponent and skill-reference model folders together, as rclone folder globs, each listed once. */
	std::vector<std::string> ListOpponentModelSyncPaths(const std::string& targetAgentName, const std::string& targetBuildName);
}
