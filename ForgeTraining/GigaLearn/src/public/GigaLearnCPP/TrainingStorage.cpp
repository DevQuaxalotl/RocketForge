#include "TrainingStorage.h"

#include <cctype>
#include <optional>
#include <utility>

namespace {
	void AddUniquePath(std::vector<std::filesystem::path>& paths, const std::filesystem::path& path) {
		if (path.empty())
			return;

		for (const auto& existing : paths) {
			if (existing == path)
				return;
		}

		paths.push_back(path);
	}

	std::filesystem::path MakePreferredCopy(std::filesystem::path path) {
		path.make_preferred();
		return path;
	}

	std::string SanitizeFolderPart(const std::string& text) {
		std::string result = {};
		bool lastWasSeparator = false;

		for (unsigned char c : text) {
			if (std::isalnum(c)) {
				result.push_back((char)c);
				lastWasSeparator = false;
			} else if (c == '.' && !result.empty() && !lastWasSeparator) {
				result.push_back('.');
				lastWasSeparator = false;
			} else if (!lastWasSeparator) {
				result.push_back('_');
				lastWasSeparator = true;
			}
		}

		while (!result.empty() && (result.front() == '_' || result.front() == '.'))
			result.erase(result.begin());
		while (!result.empty() && (result.back() == '_' || result.back() == '.'))
			result.pop_back();

		return result.empty() ? "unknown" : result;
	}
}

GGL::TrainingStorage::TrainingStorage(
	std::filesystem::path stageRootFolder,
	std::filesystem::path agentModelsFolder) :
	stageRootFolder(std::move(stageRootFolder)),
	agentModelsFolder(std::move(agentModelsFolder)) {}

std::filesystem::path GGL::TrainingStorage::GetAgentModelsFolder() const {
	if (!agentModelsFolder.empty())
		return MakePreferredCopy(agentModelsFolder);

	if (!stageRootFolder.empty()) {
		std::filesystem::path parent = stageRootFolder.parent_path();
		if (!parent.empty() && parent.filename() == TRAINING_STAGES_FOLDER_NAME)
			return MakePreferredCopy(parent.parent_path());
		return MakePreferredCopy(parent);
	}

	return {};
}

std::filesystem::path GGL::TrainingStorage::GetStageRootFolder() const {
	return stageRootFolder.empty() ? std::filesystem::path() : MakePreferredCopy(stageRootFolder);
}

std::string GGL::TrainingStorage::GetStageName() const {
	if (!stageRootFolder.empty() && !stageRootFolder.filename().empty())
		return stageRootFolder.filename().string();

	return {};
}

std::filesystem::path GGL::TrainingStorage::GetSkillBenchmarkFolder() const {
	if (!skillBenchmarkFolderOverride.empty())
		return MakePreferredCopy(skillBenchmarkFolderOverride);

	std::filesystem::path root = GetAgentModelsFolder();
	if (root.empty())
		return {};

	return MakePreferredCopy(root / SKILL_BENCHMARKS_FOLDER_NAME);
}

std::filesystem::path GGL::TrainingStorage::GetSkillBenchmarkRegistryPath() const {
	std::filesystem::path benchmarkRoot = GetSkillBenchmarkFolder();
	if (benchmarkRoot.empty())
		return {};

	return MakePreferredCopy(benchmarkRoot / SKILL_BENCHMARK_REGISTRY_FILE_NAME);
}

std::vector<std::filesystem::path> GGL::TrainingStorage::GetSkillBenchmarkSearchRoots() const {
	std::vector<std::filesystem::path> roots = {};
	std::filesystem::path currentRoot = GetSkillBenchmarkFolder();
	if (std::filesystem::exists(currentRoot))
		AddUniquePath(roots, currentRoot);

	std::filesystem::path agentRoot = GetAgentModelsFolder();
	if (agentRoot.empty() || !std::filesystem::exists(agentRoot))
		return roots;

	for (const auto& entry : std::filesystem::recursive_directory_iterator(agentRoot)) {
		if (!entry.is_directory())
			continue;
		if (entry.path() == currentRoot)
			continue;
		if (entry.path().filename() == SKILL_BENCHMARKS_FOLDER_NAME)
			AddUniquePath(roots, MakePreferredCopy(entry.path()));
	}

	return roots;
}

std::filesystem::path GGL::TrainingStorage::GetImportedModelsFolder() const {
	std::filesystem::path root = GetAgentModelsFolder();
	if (root.empty())
		return {};

	return MakePreferredCopy(root / IMPORTED_MODELS_FOLDER_NAME);
}

std::filesystem::path GGL::TrainingStorage::GetStaticOpponentSelectionPath() const {
	std::filesystem::path root = GetAgentModelsFolder();
	if (root.empty())
		return {};

	return MakePreferredCopy(root / STATIC_OPPONENT_SELECTION_FILE_NAME);
}

void GGL::TrainingStorage::SetRecentPolicyVersionsFolder(const std::filesystem::path& path) {
	recentPolicyVersionsFolderOverride = MakePreferredCopy(path);
}

void GGL::TrainingStorage::SetSkillBenchmarkFolder(const std::filesystem::path& path) {
	skillBenchmarkFolderOverride = MakePreferredCopy(path);
}

std::filesystem::path GGL::TrainingStorage::GetAgentCheckpointFolder() const {
	std::filesystem::path root = GetAgentModelsFolder();
	if (root.empty())
		return {};

	return MakePreferredCopy(root / TEMP_CHECKPOINTS_FOLDER_NAME);
}

std::filesystem::path GGL::TrainingStorage::GetRecentPolicyVersionsFolder() const {
	if (!recentPolicyVersionsFolderOverride.empty())
		return MakePreferredCopy(recentPolicyVersionsFolderOverride);

	std::filesystem::path root = GetAgentModelsFolder();
	if (root.empty())
		return {};

	return MakePreferredCopy(root / RECENT_POLICY_VERSIONS_FOLDER_NAME);
}

std::vector<std::filesystem::path> GGL::TrainingStorage::GetRecentPolicyVersionSearchRoots(const std::string& versionId) const {
	(void)versionId;
	std::vector<std::filesystem::path> roots = {};
	AddUniquePath(roots, GetRecentPolicyVersionsFolder());

	return roots;
}

std::filesystem::path GGL::TrainingStorage::GetPreferredRecentPolicyVersionsRoot(const std::string& versionId) const {
	std::filesystem::path sharedPolicyRoot = GetRecentPolicyVersionsFolder();
	if (HasNumberedDirs(sharedPolicyRoot))
		return sharedPolicyRoot;

	for (const auto& root : GetRecentPolicyVersionSearchRoots(versionId)) {
		if (root == sharedPolicyRoot)
			continue;
		if (HasNumberedDirs(root))
			return root;
	}

	return sharedPolicyRoot;
}

std::string GGL::TrainingStorage::MakeStageRelativePath(const std::filesystem::path& path) const {
	std::filesystem::path stageRoot = GetStageRootFolder();
	if (stageRoot.empty())
		return path.generic_string();

	std::error_code ec;
	std::filesystem::path rel = std::filesystem::relative(path, stageRoot, ec);
	if (ec)
		rel = path;
	return rel.generic_string();
}

bool GGL::TrainingStorage::IsAllDigits(const std::string& text) {
	if (text.empty())
		return false;

	for (char c : text) {
		if (!std::isdigit(static_cast<unsigned char>(c)))
			return false;
	}

	return true;
}

std::set<int64_t> GGL::TrainingStorage::FindNumberedDirs(const std::filesystem::path& basePath) {
	std::set<int64_t> results = {};
	if (!std::filesystem::exists(basePath) || !std::filesystem::is_directory(basePath))
		return results;

	for (const auto& entry : std::filesystem::directory_iterator(basePath)) {
		if (!entry.is_directory())
			continue;

		std::string name = entry.path().filename().string();
		if (!IsAllDigits(name))
			continue;

		results.insert(std::stoll(name));
	}

	return results;
}

bool GGL::TrainingStorage::HasNumberedDirs(const std::filesystem::path& basePath) {
	return !FindNumberedDirs(basePath).empty();
}

bool GGL::TrainingStorage::HasDirectModelFiles(const std::filesystem::path& folder) {
	return std::filesystem::exists(folder / "POLICY.lt")
		|| std::filesystem::exists(folder / "SHARED_HEAD.lt")
		|| std::filesystem::exists(folder / "CRITIC.lt")
		|| std::filesystem::exists(folder / "INFERENCE_POLICY.json");
}

std::filesystem::path GGL::TrainingStorage::ResolveModelFolder(const std::filesystem::path& rootFolder) {
	if (rootFolder.empty() || !std::filesystem::exists(rootFolder))
		return {};

	// External exports can include stale or empty numbered folders beside the
	// loadable root model files, so direct files take precedence.
	if (HasDirectModelFiles(rootFolder))
		return rootFolder;

	if (HasNumberedDirs(rootFolder))
		return GetMostRecentCheckpointIn(rootFolder);

	return {};
}

std::filesystem::path GGL::TrainingStorage::GetMostRecentCheckpointIn(const std::filesystem::path& rootFolder) {
	std::optional<int64_t> newestTimesteps = {};
	for (int64_t timesteps : FindNumberedDirs(rootFolder)) {
		if (!newestTimesteps.has_value() || timesteps > newestTimesteps.value())
			newestTimesteps = timesteps;
	}

	if (!newestTimesteps.has_value())
		return {};

	return MakePreferredCopy(rootFolder / std::to_string(newestTimesteps.value()));
}

std::vector<std::filesystem::path> GGL::TrainingStorage::GetCheckpointFoldersToPrune(
	const std::filesystem::path& rootFolder,
	size_t checkpointsToKeep) {

	std::vector<std::filesystem::path> toPrune = {};
	if (checkpointsToKeep < 1)
		checkpointsToKeep = 1;

	std::set<int64_t> allTimesteps = FindNumberedDirs(rootFolder);
	if (allTimesteps.size() <= checkpointsToKeep)
		return toPrune;

	const size_t pruneCount = allTimesteps.size() - checkpointsToKeep;
	size_t pruned = 0;
	for (int64_t timesteps : allTimesteps) {
		if (pruned >= pruneCount)
			break;

		toPrune.push_back(MakePreferredCopy(rootFolder / std::to_string(timesteps)));
		pruned++;
	}

	return toPrune;
}

std::string GGL::TrainingStorage::MakeRecentPolicyVersionFolderName(const std::string& versionId, uint64_t timesteps) {
	if (versionId.empty())
		return std::to_string(timesteps);

	return versionId + "_" + std::to_string(timesteps);
}

std::optional<uint64_t> GGL::TrainingStorage::TryParseRecentPolicyVersionTimesteps(
	const std::string& folderName,
	const std::string& versionId) {

	if (versionId.empty() && IsAllDigits(folderName))
		return static_cast<uint64_t>(std::stoull(folderName));

	if (versionId.empty())
		return std::nullopt;

	const std::string prefix = versionId + "_";
	if (!folderName.starts_with(prefix))
		return std::nullopt;

	const std::string timestepText = folderName.substr(prefix.size());
	if (!IsAllDigits(timestepText))
		return std::nullopt;

	return static_cast<uint64_t>(std::stoull(timestepText));
}

GGL::BenchmarkStorageLocation GGL::TrainingStorage::MakeBenchmarkStorageLocation(
	const std::filesystem::path& benchmarkRoot,
	const std::string& stageName,
	uint64_t timesteps,
	const std::string& versionId) {

	const std::string benchmarkVersionId = versionId.empty() ? "unversioned" : versionId;
	const std::string safeStageName = SanitizeFolderPart(stageName.empty() ? "unknown-stage" : stageName);

	BenchmarkStorageLocation result = {};
	result.id = benchmarkVersionId + "_" + std::to_string(timesteps) + "_" + safeStageName;
	result.folderName = result.id;
	result.folder = MakePreferredCopy(benchmarkRoot / result.folderName);

	int collisionIndex = 1;
	while (std::filesystem::exists(result.folder)) {
		result.id = benchmarkVersionId + "_" + std::to_string(timesteps) + "_" + safeStageName + "_" + std::to_string(collisionIndex);
		result.folderName = result.id;
		result.folder = MakePreferredCopy(benchmarkRoot / result.folderName);
		collisionIndex++;
	}

	return result;
}
