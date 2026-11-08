#pragma once

#include "TrainingSetup/TrainingSetupSnapshot.h"

#include <GigaLearnCPP/SegmentVersioning.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <map>
#include <string>
#include <vector>

namespace RocketForgeTraining
{
	/** One saved checkpoint inside a Training Segment. **/
	struct SegmentCheckpoint
	{
		uint64_t timesteps = 0;
		std::string path;

		void Serialize(GGL::JsonArchive& ar);
	};

	/** The newest checkpoint of a segment, from either its durable or its autosave folder. **/
	struct ResolvedCheckpoint
	{
		std::filesystem::path root;
		std::filesystem::path folder;
		uint64_t timesteps = 0;

		/** True when the checkpoint is a tsPerSave autosave under temp_checkpoints/. **/
		bool isAutosave = false;
	};

	/** Persistent metadata for one continuous Training Segment. **/
	struct TrainingSegmentMetadata
	{
		int schemaVersion = 1;
		std::string segmentId;
		std::string folderName;
		std::string buildName;
		std::string parentSegmentId;
		std::string branchName = "main";
		std::string status = "active";
		std::string stageName;
		std::string setupHash;
		std::string createdAt;
		std::string updatedAt;
		std::string lastTrainedAt;
		uint64_t startTimestep = 0;
		uint64_t endTimestep = 0;
		double totalTrainingSeconds = 0.0;
		std::string sourceCheckpointPath;

		/** Per-mode skill at segment start, retained after restarts and source pruning. **/
		std::map<std::string, double> skillStartRatings;
		std::string wandbProject;
		std::string wandbGroup;
		std::string wandbRunName;
		std::string wandbRunId;
		std::string notes;
		bool eligibleBenchmark = false;
		bool eligibleRLBot = false;
		bool pinned = false;
		std::vector<SegmentCheckpoint> checkpoints;

		void Serialize(GGL::JsonArchive& ar);
	};

	/** Persistent metadata for a Build. **/
	struct BuildMetadata
	{
		int schemaVersion = 1;
		std::string buildName;
		std::string folderName;
		std::string activeSegmentId;
		std::string createdAt;
		std::string updatedAt;
		uint64_t activeTimesteps = 0;
		uint64_t totalTrainedTimesteps = 0;
		std::string notes;

		void Serialize(GGL::JsonArchive& ar);
	};

	/** Lightweight entry stored in Models/builds.json. **/
	struct BuildSummary
	{
		std::string buildName;
		std::string folderName;
		std::string createdAt;
		std::string updatedAt;
		bool active = false;

		void Serialize(GGL::JsonArchive& ar);
	};

	/**
	 * Owns the Build-first storage layout under an agent's Models folder.
	 * The class does not move model files; it creates paths and persists metadata.
	 **/
	class BuildStorage
	{
	public:
		static constexpr const char* BUILD_REGISTRY_FILE_NAME = "builds.json";
		static constexpr const char* BUILD_FILE_NAME = "build.json";
		static constexpr const char* SEGMENTS_FOLDER_NAME = "segments";
		static constexpr const char* SEGMENT_FILE_NAME = "segment.json";
		static constexpr const char* CHECKPOINTS_FOLDER_NAME = "checkpoints";
		static constexpr const char* TEMP_CHECKPOINTS_FOLDER_NAME = "temp_checkpoints";
		static constexpr const char* RLBOT_EXPORTS_FILE_NAME = "rlbot_exports.json";
		static constexpr const char* STATIC_OPPONENTS_FILE_NAME = "static_opponents.json";
		static constexpr const char* SKILL_REFERENCES_FILE_NAME = "skill_references.json";
		static constexpr const char* RECENT_POLICY_VERSIONS_FOLDER_NAME = "recent_policy_versions";
		static constexpr const char* RECENT_POLICY_VERSIONS_FILE_NAME = "recent_policy_versions.json";
		static constexpr const char* SKILL_BENCHMARKS_FOLDER_NAME = "skill_benchmarks";
		static constexpr const char* SKILL_BENCHMARKS_FILE_NAME = "benchmarks.json";

		BuildStorage(std::filesystem::path agentModelsFolder, std::string buildName);

		static std::string SanitizeFolderPart(const std::string& text);
		static std::string MakeTimestamp();
		static std::string MakeSetupHash(const RocketForge::TrainingSetupSnapshot& snapshot);
		/** Returns whether the current setup matches the saved segment identity. **/
		bool SegmentSetupHashMatches(
			const TrainingSegmentMetadata& segment,
			const RocketForge::TrainingSetupSnapshot& currentSnapshot,
			const std::string& currentHash) const;
		static std::string MakeSegmentId(int major, int minor = 0);
		static std::optional<GGL::SegmentVersionNumber> TryParseSegmentRevision(const std::string& segmentId);
		static std::optional<int> TryParseSegmentId(const std::string& segmentId);

		std::filesystem::path AgentModelsFolder() const;
		std::filesystem::path BuildRegistryPath() const;
		std::filesystem::path BuildRoot() const;
		std::filesystem::path BuildMetadataPath() const;
		std::filesystem::path SegmentsRoot() const;
		std::filesystem::path SegmentRoot(const std::string& segmentFolderName) const;
		std::filesystem::path SegmentCheckpointsRoot(const std::string& segmentFolderName) const;
		std::filesystem::path SegmentTempCheckpointsRoot(const std::string& segmentFolderName) const;

		/** Newest checkpoint across checkpoints/ and temp_checkpoints/ under a segment root; the durable one wins a tie. **/
		static std::optional<ResolvedCheckpoint> LatestSegmentCheckpointIn(const std::filesystem::path& segmentRoot);
		std::optional<ResolvedCheckpoint> LatestSegmentCheckpoint(const std::string& segmentFolderName) const;

		/**
		 * Copies the segment's newest autosave into checkpoints/ when it is newer than every durable checkpoint,
		 * then refreshes and saves the segment's checkpoint list.
		 * Autosaves are excluded from the sync manifest and pruned remotely at each pod launch, so this is how they outlive a run.
		 * Returns the new durable folder, or empty when there was nothing newer to promote.
		 **/
		std::optional<std::filesystem::path> PromoteNewestAutosave(TrainingSegmentMetadata& segment) const;

		std::filesystem::path RecentPolicyVersionsRoot() const;
		std::filesystem::path RecentPolicyVersionsRegistryPath() const;
		std::filesystem::path SkillBenchmarksRoot() const;
		std::filesystem::path SkillBenchmarksRegistryPath() const;
		std::filesystem::path RLBotExportsPath() const;
		std::filesystem::path StaticOpponentsPath() const;
		std::filesystem::path SkillReferencesPath() const;
		std::filesystem::path ResolveBuildPath(const std::string& relativeOrAbsolutePath) const;
		std::string MakeRelativeToBuild(const std::filesystem::path& path) const;

		/** Optionally reuse the segments already read during Build reconciliation. **/
		std::vector<BuildSummary> ListBuilds(
			std::map<std::string, std::vector<TrainingSegmentMetadata>>* reconciledSegments = nullptr) const;
		void SaveBuilds(const std::vector<BuildSummary>& builds) const;
		BuildMetadata LoadBuild() const;
		void SaveBuild(const BuildMetadata& build) const;
		void EnsureBuildExists() const;
		bool RenameBuild(const std::string& newBuildName, std::string* error = nullptr) const;

		std::vector<TrainingSegmentMetadata> ListSegments() const;
		std::optional<TrainingSegmentMetadata> LoadSegmentById(const std::string& segmentId) const;

		/** Read a known segment without reconciling historical segments; invalid identity returns empty. **/
		std::optional<TrainingSegmentMetadata> LoadSegmentByFolderName(const std::string& segmentFolderName) const;

		/**
		 * Record a completed durable save or autosave without scanning historical segments.
		 * False requests reconciliation; autosaves never become durable checkpoint entries.
		 **/
		bool RecordSavedCheckpoint(
			const std::string& segmentId,
			const std::string& segmentFolderName,
			uint64_t timesteps,
			const std::filesystem::path& checkpointFolder) const;

		void SaveSegment(const TrainingSegmentMetadata& segment) const;

		/** Recover only the exact starting ratings; initialMu applies only to an untrained scratch segment. **/
		std::map<std::string, double> ResolveSegmentSkillStartRatings(
			const TrainingSegmentMetadata& segment, std::optional<double> initialMu = std::nullopt) const;
		std::string AllocateSegmentId() const;
		std::string MakeSegmentFolderName(const std::string& segmentId, const std::string& stageName) const;
		void RefreshSegmentCheckpoints(TrainingSegmentMetadata& segment, bool markTrained = false) const;

	private:
		std::filesystem::path agentModelsFolder;
		std::string buildName;
		std::string buildFolderName;
	};
}
