#pragma once

#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnInfer/JsonArchive.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace RocketForgeTraining
{
	inline constexpr const char* IMPORTED_MODEL_METADATA_FILE_NAME = "imported_model.json";

	/** File-presence summary for an imported model's resolved loadable checkpoint folder. **/
	struct ImportedModelTrainingFileStatus
	{
		bool hasPolicy = false;
		bool hasSharedHead = false;
		bool hasCritic = false;
		bool hasAuxCritic = false;
		bool hasPolicyOptim = false;
		bool hasSharedHeadOptim = false;
		bool hasCriticOptim = false;
		bool hasAuxCriticOptim = false;
		bool hasRunningStats = false;
		bool hasInferencePolicy = false;

		std::vector<std::string> MissingTrainingFiles() const;

		void Serialize(GGL::JsonArchive& ar) {
			ar.Field("policy", hasPolicy);
			ar.Field("shared_head", hasSharedHead);
			ar.Field("critic", hasCritic);
			ar.Field("aux_critic", hasAuxCritic);
			ar.Field("policy_optim", hasPolicyOptim);
			ar.Field("shared_head_optim", hasSharedHeadOptim);
			ar.Field("critic_optim", hasCriticOptim);
			ar.Field("aux_critic_optim", hasAuxCriticOptim);
			ar.Field("running_stats", hasRunningStats);
		}
	};

	/** Persistent metadata for a frozen model under an agent's Imported Models folder. **/
	struct ImportedModelMetadata
	{
		int schemaVersion = 1;
		std::string folderName;
		std::string displayName;
		std::string targetAgentName;
		std::string sourceType = "external";
		std::string sourceBuildName;
		std::string sourceStageName;
		std::string sourceVersionId;
		std::string sourcePath;
		std::string importedAt;
		std::string notes;
		uint64_t latestTimesteps = 0;
		bool canUseForInference = false;
		bool canUseForTrainingResume = false;
		bool canUseForTrainingWarmStart = false;
		ImportedModelTrainingFileStatus files;

		void Serialize(GGL::JsonArchive& ar) {
			ar.Field("schema_version", schemaVersion, 1);
			ar.Field("folder_name", folderName);
			ar.Field("display_name", displayName);
			ar.Field("target_agent", targetAgentName);
			ar.Field("source_type", sourceType);
			ar.FieldNullIfEmpty("source_build", sourceBuildName);
			ar.FieldNullIfEmpty("source_stage", sourceStageName);
			ar.FieldNullIfEmpty("source_version_id", sourceVersionId);

			// Old code loaded source_path/imported_at via a null-tolerant helper but saved them
			// as plain (non-null) strings — an asymmetry that predates this migration. Preserve
			// it deliberately rather than switching to FieldNullIfEmpty, which would also make
			// save start emitting null for these two fields.
			if (ar.Contains("source_path") && ar.RawLoad()->at("source_path").is_null())
				sourcePath.clear();
			else
				ar.Field("source_path", sourcePath);

			if (ar.Contains("imported_at") && ar.RawLoad()->at("imported_at").is_null())
				importedAt.clear();
			else
				ar.Field("imported_at", importedAt);

			ar.Field("latest_timesteps", latestTimesteps);
			ar.Field("can_use_for_inference", canUseForInference);
			ar.Field("can_use_for_training_resume", canUseForTrainingResume);
			ar.Field("can_use_for_training_warm_start", canUseForTrainingWarmStart);

			// missing_training_files is derived on save and ignored on load.
			if (ar.IsSaving()) {
				nlohmann::json missing = nlohmann::json::array();
				for (const std::string& fileName : files.MissingTrainingFiles())
					missing.push_back(fileName);
				(*ar.Raw())["missing_training_files"] = missing;
			}

			ar.Field("files", files);
			ar.FieldNullIfEmpty("notes", notes);
		}
	};

	/** Input options for importing a model into Imported Models. **/
	struct ImportedModelImportConfig
	{
		std::string targetAgentName;
		std::filesystem::path sourceFolder;
		std::string folderName;
		std::string displayName;
		std::string sourceType = "external";
		std::string sourceBuildName;
		std::string sourceStageName;
		std::string sourceVersionId;
		std::string notes;
		bool overwrite = false;
	};

	/** Result of a completed imported model copy. **/
	struct ImportedModelImportResult
	{
		std::filesystem::path importedModelRoot;
		std::filesystem::path resolvedModelFolder;
		ImportedModelMetadata metadata;
	};

	/** What a folder offered for import contains, read before anything is copied. **/
	struct ImportSourceInspection
	{
		/** Folder the model files come from: the picked folder, or its newest numbered checkpoint. Empty when none loads. **/
		std::filesystem::path modelFolder;

		/** Names of the model files inside modelFolder that an import copies. **/
		std::vector<std::string> filesToCopy;
		uint64_t bytesToCopy = 0;
		uint64_t timesteps = 0;
		ImportedModelTrainingFileStatus files;

		/** The checkpoint's segment metadata when the source is a checkpoint inside a RocketForge build. **/
		std::optional<TrainingSegmentMetadata> segment;

		bool IsLoadable() const { return !modelFolder.empty(); }
	};

	/**
	 * The files an import copies when they are present.
	 * A scripted policy's INFERENCE_POLICY.json also brings the artifacts it names.
	 **/
	inline constexpr const char* IMPORTABLE_MODEL_FILE_NAMES[] = {
		"POLICY.lt",
		"SHARED_HEAD.lt",
		"CRITIC.lt",
		"AUX_CRITIC.lt",
		"POLICY_OPTIM.lt",
		"SHARED_HEAD_OPTIM.lt",
		"CRITIC_OPTIM.lt",
		"AUX_CRITIC_OPTIM.lt",
		"RUNNING_STATS.json",
		"INFERENCE_POLICY.json",
		"agent_setup.json",
	};

	/**
	 * Reads a candidate import folder without copying anything.
	 * Accepts a folder holding model files, a folder of numbered checkpoints (the newest wins),
	 * or a RocketForge segment folder (its newest checkpoint).
	 **/
	ImportSourceInspection InspectImportSource(const std::filesystem::path& sourceFolder);

	std::string SanitizeImportedModelFolderName(const std::string& text);
	std::filesystem::path GetAgentImportedModelsFolder(const std::string& agentName);
	std::vector<std::filesystem::path> GetAgentImportedModelSearchFolders(const std::string& agentName);
	std::filesystem::path ResolveImportedModelRoot(const std::string& agentName, const std::string& folderName);
	ImportedModelTrainingFileStatus InspectImportedModelTrainingFiles(const std::filesystem::path& modelFolder);
	std::optional<ImportedModelMetadata> LoadImportedModelMetadata(const std::filesystem::path& importedModelRoot);
	void SaveImportedModelMetadata(const std::filesystem::path& importedModelRoot, const ImportedModelMetadata& metadata);
	ImportedModelImportResult ImportExternalModel(const ImportedModelImportConfig& config);

	/** One discovered imported model under an agent's Imported Models folder(s). **/
	struct DiscoveredImportedModel
	{
		std::string folderName;
		std::filesystem::path importedRoot;
		std::filesystem::path resolvedModelFolder;  
		ImportedModelMetadata metadata;             
		bool canUseForInference = false;
	};

	/**
	 * Enumerates imported models directly under the given search roots, skipping non-directories,
	 * "__"-prefixed and duplicate folder names, and folders that don't resolve.
	 **/
	std::vector<DiscoveredImportedModel> EnumerateImportedModelsInRoots(const std::vector<std::filesystem::path>& searchRoots);

	/** Enumerates the agent's imported models across its configured search folders. **/
	std::vector<DiscoveredImportedModel> EnumerateImportedModels(const std::string& agentName);
}

namespace RocketForgeTraining::ImportedModelStorageInternal
{
	inline std::filesystem::path MakePreferredCopy(std::filesystem::path path)
	{
		path.make_preferred();
		return path;
	}

	inline std::optional<nlohmann::json> ReadJsonObject(const std::filesystem::path& path)
	{
		std::ifstream fIn(path);
		if (!fIn.good())
			return std::nullopt;

		nlohmann::json j = nlohmann::json::parse(fIn, nullptr, false);
		if (!j.is_object())
			return std::nullopt;
		return j;
	}

	inline std::filesystem::path ResolveImportSourceModelFolder(const std::filesystem::path& sourceFolder)
	{
		const std::filesystem::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(sourceFolder);
		if (!modelFolder.empty())
			return modelFolder;

		return GGL::TrainingStorage::ResolveModelFolder(sourceFolder / BuildStorage::CHECKPOINTS_FOLDER_NAME);
	}

	/** The owning segment of a <segment>/checkpoints/<timesteps> folder, or nullopt for any other layout. **/
	inline std::optional<TrainingSegmentMetadata> LoadCheckpointSegment(const std::filesystem::path& modelFolder)
	{
		const std::filesystem::path checkpointsFolder = modelFolder.parent_path();
		if (checkpointsFolder.filename() != BuildStorage::CHECKPOINTS_FOLDER_NAME)
			return std::nullopt;

		const std::optional<nlohmann::json> j = ReadJsonObject(checkpointsFolder.parent_path() / BuildStorage::SEGMENT_FILE_NAME);
		if (!j.has_value())
			return std::nullopt;

		TrainingSegmentMetadata segment = {};
		if (!GGL::JsonLoad(*j, segment).empty())
			return std::nullopt;
		return segment;
	}

	/** Adds the weight artifacts a scripted INFERENCE_POLICY.json names, when they sit beside it. **/
	inline void AppendInferencePolicyArtifacts(const std::filesystem::path& modelFolder, std::vector<std::string>& fileNames)
	{
		const std::optional<nlohmann::json> policy = ReadJsonObject(modelFolder / "INFERENCE_POLICY.json");
		if (!policy.has_value())
			return;

		for (const char* key : { "artifact", "original_artifact" }) {
			if (!policy->contains(key) || !(*policy)[key].is_string())
				continue;

			// Only plain file names: an artifact path that leaves the folder is never copied.
			const std::string artifactName = (*policy)[key].get<std::string>();
			const std::filesystem::path artifactPath(artifactName);
			if (artifactName.empty() || artifactPath != artifactPath.filename())
				continue;

			const bool alreadyListed = std::find(fileNames.begin(), fileNames.end(), artifactName) != fileNames.end();
			if (!alreadyListed && std::filesystem::is_regular_file(modelFolder / artifactPath))
				fileNames.push_back(artifactName);
		}
	}

	inline std::filesystem::path WeaklyCanonicalOrAbsolute(const std::filesystem::path& path)
	{
		std::error_code error = {};
		std::filesystem::path result = std::filesystem::weakly_canonical(path, error);
		if (!error)
			return result;

		error = {};
		result = std::filesystem::absolute(path, error);
		return error ? path : result;
	}

	inline bool IsPathWithinOrEqual(const std::filesystem::path& parent, const std::filesystem::path& child)
	{
		const std::filesystem::path parentPath = WeaklyCanonicalOrAbsolute(parent);
		const std::filesystem::path childPath = WeaklyCanonicalOrAbsolute(child);
		if (parentPath == childPath)
			return true;

		std::error_code error = {};
		const std::filesystem::path relative = std::filesystem::relative(childPath, parentPath, error);
		if (error || relative.empty() || relative.is_absolute())
			return false;

		for (const std::filesystem::path& part : relative) {
			if (part == "..")
				return false;
		}
		return true;
	}

	inline uint64_t ReadRunningStatsTimesteps(const std::filesystem::path& modelFolder)
	{
		using nlohmann::json;

		const std::filesystem::path statsPath = modelFolder / "RUNNING_STATS.json";
		if (!std::filesystem::exists(statsPath))
			return 0;

		std::ifstream fIn(statsPath);
		if (!fIn.good())
			return 0;

		const json j = json::parse(fIn, nullptr, false);
		if (!j.is_object())
			return 0;

		return j.value("total_timesteps", static_cast<uint64_t>(0));
	}

	inline bool CanUseForTrainingResume(const ImportedModelTrainingFileStatus& files)
	{
		return files.hasPolicy
			&& files.hasCritic
			&& files.hasRunningStats
			&& files.hasPolicyOptim
			&& files.hasCriticOptim
			&& (!files.hasSharedHead || files.hasSharedHeadOptim)
			&& (!files.hasAuxCritic || files.hasAuxCriticOptim);
	}

}

inline std::vector<std::string> RocketForgeTraining::ImportedModelTrainingFileStatus::MissingTrainingFiles() const
{
	std::vector<std::string> missing = {};
	if (!hasCritic)
		missing.push_back("CRITIC.lt");
	if (!hasRunningStats)
		missing.push_back("RUNNING_STATS.json");
	if (!hasPolicyOptim)
		missing.push_back("POLICY_OPTIM.lt");
	if (!hasCriticOptim)
		missing.push_back("CRITIC_OPTIM.lt");
	if (hasSharedHead && !hasSharedHeadOptim)
		missing.push_back("SHARED_HEAD_OPTIM.lt");
	if (hasAuxCritic && !hasAuxCriticOptim)
		missing.push_back("AUX_CRITIC_OPTIM.lt");
	return missing;
}

inline RocketForgeTraining::ImportSourceInspection RocketForgeTraining::InspectImportSource(const std::filesystem::path& sourceFolder)
{
	using namespace ImportedModelStorageInternal;

	ImportSourceInspection inspection = {};
	if (sourceFolder.empty() || !std::filesystem::is_directory(sourceFolder))
		return inspection;

	inspection.modelFolder = ResolveImportSourceModelFolder(MakePreferredCopy(sourceFolder));
	if (inspection.modelFolder.empty())
		return inspection;

	for (const char* fileName : IMPORTABLE_MODEL_FILE_NAMES) {
		if (std::filesystem::is_regular_file(inspection.modelFolder / fileName))
			inspection.filesToCopy.push_back(fileName);
	}

	AppendInferencePolicyArtifacts(inspection.modelFolder, inspection.filesToCopy);

	for (const std::string& fileName : inspection.filesToCopy) {
		std::error_code error = {};
		const uintmax_t size = std::filesystem::file_size(inspection.modelFolder / fileName, error);
		if (!error)
			inspection.bytesToCopy += size;
	}

	inspection.files = InspectImportedModelTrainingFiles(inspection.modelFolder);
	inspection.timesteps = ReadRunningStatsTimesteps(inspection.modelFolder);
	if (inspection.timesteps == 0 && GGL::TrainingStorage::IsAllDigits(inspection.modelFolder.filename().string()))
		inspection.timesteps = static_cast<uint64_t>(std::stoull(inspection.modelFolder.filename().string()));

	inspection.segment = LoadCheckpointSegment(inspection.modelFolder);
	return inspection;
}

inline std::string RocketForgeTraining::SanitizeImportedModelFolderName(const std::string& text)
{
	std::string result = {};
	bool lastWasSeparator = false;

	for (unsigned char c : text) {
		if (std::isalnum(c)) {
			result.push_back((char)c);
			lastWasSeparator = false;
		} else if ((c == '-' || c == '_' || c == '.') && !result.empty() && !lastWasSeparator) {
			result.push_back((char)c);
			lastWasSeparator = false;
		} else if (!lastWasSeparator) {
			result.push_back('-');
			lastWasSeparator = true;
		}
	}

	while (!result.empty() && (result.front() == '-' || result.front() == '_' || result.front() == '.'))
		result.erase(result.begin());
	while (!result.empty() && (result.back() == '-' || result.back() == '_' || result.back() == '.'))
		result.pop_back();

	return result.empty() ? "imported-model" : result;
}

inline std::filesystem::path RocketForgeTraining::GetAgentImportedModelsFolder(const std::string& agentName)
{
	using namespace ImportedModelStorageInternal;

	const std::filesystem::path modelsRoot = ResolveAgentModelsFolder(agentName);
	if (modelsRoot.empty())
		return {};

	return MakePreferredCopy(modelsRoot / GGL::TrainingStorage::IMPORTED_MODELS_FOLDER_NAME);
}

inline std::vector<std::filesystem::path> RocketForgeTraining::GetAgentImportedModelSearchFolders(const std::string& agentName)
{
	using namespace ImportedModelStorageInternal;

	const std::filesystem::path importedRoot = GetAgentImportedModelsFolder(agentName);
	if (importedRoot.empty())
		return {};

	return { importedRoot };
}

inline std::filesystem::path RocketForgeTraining::ResolveImportedModelRoot(
	const std::string& agentName,
	const std::string& folderName)
{
	using namespace ImportedModelStorageInternal;

	if (folderName.empty())
		return {};

	for (const std::filesystem::path& root : GetAgentImportedModelSearchFolders(agentName)) {
		const std::filesystem::path candidate = root / folderName;
		if (!root.empty() && std::filesystem::exists(candidate))
			return MakePreferredCopy(candidate);
	}

	return {};
}

inline RocketForgeTraining::ImportedModelTrainingFileStatus RocketForgeTraining::InspectImportedModelTrainingFiles(
	const std::filesystem::path& modelFolder)
{
	ImportedModelTrainingFileStatus files = {};
	if (modelFolder.empty())
		return files;

	files.hasPolicy = std::filesystem::exists(modelFolder / "POLICY.lt");
	files.hasSharedHead = std::filesystem::exists(modelFolder / "SHARED_HEAD.lt");
	files.hasCritic = std::filesystem::exists(modelFolder / "CRITIC.lt");
	files.hasAuxCritic = std::filesystem::exists(modelFolder / "AUX_CRITIC.lt");
	files.hasPolicyOptim = std::filesystem::exists(modelFolder / "POLICY_OPTIM.lt");
	files.hasSharedHeadOptim = std::filesystem::exists(modelFolder / "SHARED_HEAD_OPTIM.lt");
	files.hasCriticOptim = std::filesystem::exists(modelFolder / "CRITIC_OPTIM.lt");
	files.hasAuxCriticOptim = std::filesystem::exists(modelFolder / "AUX_CRITIC_OPTIM.lt");
	files.hasRunningStats = std::filesystem::exists(modelFolder / "RUNNING_STATS.json");
	files.hasInferencePolicy = files.hasPolicy || std::filesystem::exists(modelFolder / "INFERENCE_POLICY.json");
	return files;
}

inline std::optional<RocketForgeTraining::ImportedModelMetadata> RocketForgeTraining::LoadImportedModelMetadata(
	const std::filesystem::path& importedModelRoot)
{
	using nlohmann::json;
	using namespace ImportedModelStorageInternal;

	const std::filesystem::path metadataPath = importedModelRoot / IMPORTED_MODEL_METADATA_FILE_NAME;
	if (!std::filesystem::exists(metadataPath))
		return std::nullopt;

	std::ifstream fIn(metadataPath);
	if (!fIn.good())
		return std::nullopt;

	const json j = json::parse(fIn, nullptr, false);
	if (!j.is_object())
		return std::nullopt;

	ImportedModelMetadata metadata = {};
	const std::vector<std::string> loadErrors = GGL::JsonLoad(j, metadata);
	if (!loadErrors.empty())
		throw std::runtime_error("Malformed " + metadataPath.string() + ": " + loadErrors.front());

	// folder_name/display_name default to values derived from context that the
	// struct's own Serialize cannot see; apply them only when the key was absent
	// (mirrors the old j.value() fallback), with display_name chained off folder_name.
	if (!j.contains("folder_name"))
		metadata.folderName = importedModelRoot.filename().string();
	if (!j.contains("display_name"))
		metadata.displayName = metadata.folderName;

	const std::filesystem::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(importedModelRoot);
	if (!modelFolder.empty()) {
		metadata.files = InspectImportedModelTrainingFiles(modelFolder);
		metadata.canUseForInference = metadata.files.hasInferencePolicy;
		metadata.canUseForTrainingWarmStart = metadata.files.hasPolicy;
		metadata.canUseForTrainingResume = ImportedModelStorageInternal::CanUseForTrainingResume(metadata.files);
		if (metadata.latestTimesteps == 0 && GGL::TrainingStorage::IsAllDigits(modelFolder.filename().string()))
			metadata.latestTimesteps = static_cast<uint64_t>(std::stoull(modelFolder.filename().string()));
	}
	return metadata;
}

inline void RocketForgeTraining::SaveImportedModelMetadata(
	const std::filesystem::path& importedModelRoot,
	const ImportedModelMetadata& metadata)
{
	using namespace ImportedModelStorageInternal;

	std::filesystem::create_directories(importedModelRoot);
	std::ofstream fOut(importedModelRoot / IMPORTED_MODEL_METADATA_FILE_NAME);
	if (!fOut.good())
		throw std::runtime_error("Failed to write imported model metadata.");

	nlohmann::json j;
	GGL::JsonSave(metadata, j);
	fOut << j.dump(4);
}

inline RocketForgeTraining::ImportedModelImportResult RocketForgeTraining::ImportExternalModel(
	const ImportedModelImportConfig& config)
{
	using namespace ImportedModelStorageInternal;

	if (config.targetAgentName.empty())
		throw std::runtime_error("Target agent is required.");
	if (config.sourceFolder.empty() || !std::filesystem::is_directory(config.sourceFolder))
		throw std::runtime_error("Source model folder does not exist.");

	const std::filesystem::path sourceRoot = MakePreferredCopy(config.sourceFolder);
	const ImportSourceInspection inspection = InspectImportSource(sourceRoot);
	if (!inspection.IsLoadable()) {
		throw std::runtime_error(
			"No model found in " + sourceRoot.string() + ".\n"
			"Pick a folder that contains POLICY.lt or INFERENCE_POLICY.json, a folder of numbered checkpoints, or a segment folder.");
	}

	const std::string folderName = SanitizeImportedModelFolderName(
		config.folderName.empty() ? sourceRoot.filename().string() : config.folderName);
	const std::filesystem::path targetRoot = GetAgentImportedModelsFolder(config.targetAgentName);
	if (targetRoot.empty())
		throw std::runtime_error("Agent " + config.targetAgentName + " has no Models folder.");

	const std::filesystem::path importedModelRoot = MakePreferredCopy(targetRoot / folderName);

	if (IsPathWithinOrEqual(inspection.modelFolder, importedModelRoot) || IsPathWithinOrEqual(importedModelRoot, inspection.modelFolder))
		throw std::runtime_error("Cannot import a model into itself.");

	if (std::filesystem::exists(importedModelRoot) && !config.overwrite)
		throw std::runtime_error("Imported model folder already exists: " + importedModelRoot.string());

	// Copy into a "__"-prefixed staging folder, which discovery skips,
	// so a failed copy never leaves a partial model or loses the one being replaced.
	const std::filesystem::path stagingRoot = MakePreferredCopy(targetRoot / ("__importing_" + folderName));
	try {
		std::filesystem::remove_all(stagingRoot);
		std::filesystem::create_directories(stagingRoot);
		for (const std::string& fileName : inspection.filesToCopy)
			std::filesystem::copy_file(inspection.modelFolder / fileName, stagingRoot / fileName);

		std::filesystem::remove_all(importedModelRoot);
		std::filesystem::rename(stagingRoot, importedModelRoot);
	} catch (...) {
		std::error_code error = {};
		std::filesystem::remove_all(stagingRoot, error);
		throw;
	}

	const std::filesystem::path resolvedModelFolder = GGL::TrainingStorage::ResolveModelFolder(importedModelRoot);
	if (resolvedModelFolder.empty())
		throw std::runtime_error("Imported model has no loadable model folder after copy.");

	const std::optional<TrainingSegmentMetadata>& segment = inspection.segment;
	const std::string defaultVersionId = segment ? segment->segmentId : sourceRoot.filename().string();

	ImportedModelMetadata metadata = {};
	metadata.folderName = folderName;
	metadata.displayName = config.displayName.empty() ? folderName : config.displayName;
	metadata.targetAgentName = config.targetAgentName;
	metadata.sourceType = config.sourceType.empty() ? "external" : config.sourceType;
	metadata.sourceBuildName = config.sourceBuildName.empty() && segment ? segment->buildName : config.sourceBuildName;
	metadata.sourceStageName = config.sourceStageName.empty() && segment ? segment->stageName : config.sourceStageName;
	metadata.sourceVersionId = config.sourceVersionId.empty() ? defaultVersionId : config.sourceVersionId;
	metadata.sourcePath = inspection.modelFolder.string();
	metadata.importedAt = BuildStorage::MakeTimestamp();
	metadata.notes = config.notes;
	metadata.latestTimesteps = inspection.timesteps;
	metadata.files = InspectImportedModelTrainingFiles(resolvedModelFolder);
	metadata.canUseForInference = metadata.files.hasInferencePolicy;
	metadata.canUseForTrainingWarmStart = metadata.files.hasPolicy;
	metadata.canUseForTrainingResume = CanUseForTrainingResume(metadata.files);

	SaveImportedModelMetadata(importedModelRoot, metadata);

	return {
		importedModelRoot,
		resolvedModelFolder,
		metadata
	};
}

inline std::vector<RocketForgeTraining::DiscoveredImportedModel>
RocketForgeTraining::EnumerateImportedModelsInRoots(
	const std::vector<std::filesystem::path>& searchRoots)
{
	std::vector<DiscoveredImportedModel> discovered = {};
	std::set<std::string> seenFolderNames = {};

	for (const std::filesystem::path& importedRoot : searchRoots) {
		if (importedRoot.empty() || !std::filesystem::exists(importedRoot))
			continue;

		std::error_code error = {};
		for (const auto& entry : std::filesystem::directory_iterator(importedRoot, error)) {
			if (!entry.is_directory())
				continue;

			const std::string folderName = entry.path().filename().string();
			if (folderName.empty() || folderName.rfind("__", 0) == 0 || seenFolderNames.contains(folderName))
				continue;

			const std::filesystem::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(entry.path());
			if (modelFolder.empty())
				continue;

			seenFolderNames.insert(folderName);

			try {
				DiscoveredImportedModel model = {};
				model.folderName = folderName;
				model.importedRoot = entry.path();
				model.resolvedModelFolder = modelFolder;
				if (std::optional<ImportedModelMetadata> metadata = LoadImportedModelMetadata(entry.path()))
					model.metadata = std::move(*metadata);
				model.canUseForInference = InspectImportedModelTrainingFiles(modelFolder).hasInferencePolicy;
				discovered.push_back(std::move(model));
			} catch (const std::exception&) {
				// A single malformed imported_model.json must not abort discovery of the rest.
				continue;
			}
		}
	}

	return discovered;
}

inline std::vector<RocketForgeTraining::DiscoveredImportedModel>
RocketForgeTraining::EnumerateImportedModels(const std::string& agentName)
{
	return EnumerateImportedModelsInRoots(GetAgentImportedModelSearchFolders(agentName));
}
