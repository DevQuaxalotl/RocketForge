#pragma once

#include "Storage/BuildStorage.h"

#include <GigaLearnCPP/SegmentVersioning.h>

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace RocketForge { class TrainingStage; }

namespace RocketForgeTraining {

	/** Lists Builds under the given agent's Models folder. */
	std::vector<BuildSummary> ListAgentBuilds(const std::string& agentName);

	/** Lists training stage names used by active segments in a Build. Empty buildName lists all Builds. */
	std::vector<std::string> ListBuildTrainingStages(const std::string& agentName, const std::string& buildName = {});

	/** Lists branch names used by segments in a Build. Empty buildName lists all Builds. */
	std::vector<std::string> ListBuildBranches(
		const std::string& agentName,
		const std::string& buildName = {},
		bool includeInactive = true);

	/** Returns the Build's active segment, which represents the last trained segment. */
	std::optional<GGL::SegmentVersionSummary> LatestBuildVersion(const std::string& agentName, const std::string& buildName);

	/** Lists saved Build segments using optional Build and stage filters. */
	std::vector<GGL::SegmentVersionSummary> ListBuildVersions(
		const std::string& agentName,
		const std::string& buildName = {},
		const std::string& stageFilter = {},
		bool includeInactive = false);

	/** Lists known saved segments for the given stage. Inactive segments are hidden unless includeInactive=true. */
	std::vector<GGL::SegmentVersionSummary> ListVersions(const RocketForge::TrainingStage* stage, bool includeInactive = false);

	/** Updates the notes string stored for the given segment. */
	void SetVersionNotes(const RocketForge::TrainingStage* stage, const std::string& versionId, const std::string& notes);

	/** Updates the eligibility flags (benchmark, rlbot, pinned) for the given segment. */
	void SetVersionEligibility(
		const RocketForge::TrainingStage* stage,
		const std::string& versionId,
		std::optional<bool> benchmark,
		std::optional<bool> rlbot,
		std::optional<bool> pinned);

	/** Updates whether a segment is active. */
	void SetVersionActive(const RocketForge::TrainingStage* stage, const std::string& versionId, bool active);

	/** Moves the given segment metadata to another branch label. */
	void SetVersionBranch(const RocketForge::TrainingStage* stage, const std::string& versionId, const std::string& branchName);

	/** Exports the given segment's model to the RLBot bot folder. */
	void ExportVersionToRLBot(const RocketForge::TrainingStage* stage, const std::string& versionId);

	/** Exports a stored segment without requiring its historical training stage to be registered. */
	void ExportVersionToRLBot(const std::string& agentName, const std::string& buildName, const std::string& versionId);

	/** Refreshes an existing RLBot export after continued training saves a newer checkpoint. */
	void RefreshRLBotExportAfterCheckpoint(
		const RocketForge::TrainingStage* stage,
		const std::string& versionId,
		uint64_t latestTimesteps,
		const std::filesystem::path& latestCheckpoint,
		const std::string& segmentFolderName = {});

	/** Removes the given segment's export from the RLBot bot folder. */
	void RemoveVersionFromRLBot(const RocketForge::TrainingStage* stage, const std::string& versionId);

	/** Removes a stored segment export without requiring its historical training stage to be registered. */
	void RemoveVersionFromRLBot(const std::string& agentName, const std::string& buildName, const std::string& versionId);

	/** Permanently deletes the given segment's data; callers confirm with the user first. */
	void DeleteVersion(const RocketForge::TrainingStage* stage, const std::string& versionId);

	/** Deletes saved model checkpoints for the given segment, keeping only metadata. */
	void PruneVersionModels(const RocketForge::TrainingStage* stage, const std::string& versionId);
}
