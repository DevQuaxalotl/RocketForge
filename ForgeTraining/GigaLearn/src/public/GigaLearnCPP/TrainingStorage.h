#pragma once

#include <GigaLearnInfer/Framework.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>


namespace GGL {

	/** Filesystem location for a saved skill benchmark snapshot. **/
	struct BenchmarkStorageLocation {
		std::string id;
		std::string folderName;
		std::filesystem::path folder;
	};

	/**
	 * Centralizes filesystem layout for training data.
	 * This keeps training, versioning, and benchmark code focused on behavior
	 * instead of repeatedly rebuilding the same paths.
	 **/
	class GGL_TRAIN_IMEXPORT TrainingStorage {
	public:
		static constexpr const char* SKILL_BENCHMARK_REGISTRY_FILE_NAME = "benchmarks.json";
		static constexpr const char* TEMP_CHECKPOINTS_FOLDER_NAME = "Temp Checkpoints";
		static constexpr const char* TRAINING_STAGES_FOLDER_NAME = "Training Stages";
		static constexpr const char* RECENT_POLICY_VERSIONS_FOLDER_NAME = "Recent Policy Versions";
		static constexpr const char* SKILL_BENCHMARKS_FOLDER_NAME = "Skill Benchmarks";
		static constexpr const char* IMPORTED_MODELS_FOLDER_NAME = "Imported Models";
		static constexpr const char* STATIC_OPPONENT_SELECTION_FILE_NAME = "static_opponents.json";

		TrainingStorage(
			std::filesystem::path stageRootFolder = {},
			std::filesystem::path agentModelsFolder = {});

		std::filesystem::path GetAgentModelsFolder() const;
		std::filesystem::path GetStageRootFolder() const;
		std::string GetStageName() const;

		std::filesystem::path GetSkillBenchmarkFolder() const;
		std::filesystem::path GetSkillBenchmarkRegistryPath() const;
		std::vector<std::filesystem::path> GetSkillBenchmarkSearchRoots() const;
		std::filesystem::path GetImportedModelsFolder() const;
		std::filesystem::path GetStaticOpponentSelectionPath() const;
		void SetRecentPolicyVersionsFolder(const std::filesystem::path& path);
		void SetSkillBenchmarkFolder(const std::filesystem::path& path);

		std::filesystem::path GetAgentCheckpointFolder() const;
		std::filesystem::path GetRecentPolicyVersionsFolder() const;
		std::vector<std::filesystem::path> GetRecentPolicyVersionSearchRoots(const std::string& versionId) const;
		std::filesystem::path GetPreferredRecentPolicyVersionsRoot(const std::string& versionId) const;

		std::string MakeStageRelativePath(const std::filesystem::path& path) const;

		static bool IsAllDigits(const std::string& text);
		static std::set<int64_t> FindNumberedDirs(const std::filesystem::path& basePath);
		static bool HasNumberedDirs(const std::filesystem::path& basePath);
		static bool HasDirectModelFiles(const std::filesystem::path& folder);
		/** Resolves a loadable model folder from direct files or nested checkpoints. **/
		static std::filesystem::path ResolveModelFolder(const std::filesystem::path& rootFolder);
		static std::filesystem::path GetMostRecentCheckpointIn(const std::filesystem::path& rootFolder);
		/** Returns checkpoint folders that would be deleted when keeping the newest `checkpointsToKeep`. **/
		static std::vector<std::filesystem::path> GetCheckpointFoldersToPrune(
			const std::filesystem::path& rootFolder,
			size_t checkpointsToKeep = 1);
		/** Prefixes shared recent-policy folders with the owning version or segment id. **/
		static std::string MakeRecentPolicyVersionFolderName(const std::string& versionId, uint64_t timesteps);
		/** Reads timesteps from segment-prefixed recent-policy folder names. **/
		static std::optional<uint64_t> TryParseRecentPolicyVersionTimesteps(
			const std::string& folderName,
			const std::string& versionId);
		static BenchmarkStorageLocation MakeBenchmarkStorageLocation(
			const std::filesystem::path& benchmarkRoot,
			const std::string& stageName,
			uint64_t timesteps,
			const std::string& versionId);

	private:
		std::filesystem::path stageRootFolder;
		std::filesystem::path agentModelsFolder;
		std::filesystem::path recentPolicyVersionsFolderOverride;
		std::filesystem::path skillBenchmarkFolderOverride;
	};
}
