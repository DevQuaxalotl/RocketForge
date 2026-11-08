#include "PolicyVersionManager.h"
#include "SkillEstimation.h"
#include <nlohmann/json.hpp>

#include <GigaLearnCPP/Eval/PolicyMatchRunner.h>
#include <GigaLearnCPP/SegmentVersioning.h>
#include <GigaLearnCPP/Util/Utils.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <GigaLearnCPP/SkillBenchmarkRegistry.h>
#include <GigaLearnCPP/SkillReferenceRatings.h>

#include <GigaLearnInfer/AgentSetupSnapshot.h>

#include <RLGymCPP/StateSetters/FuzzedKickoffState.h>

#include <GigaLearnCPP/PPO/PPOLearner.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <typeinfo>

using namespace nlohmann;
using GGL::GetExpectedWinProbability;

namespace {
	constexpr const char* STATS_FILE_NAME = "STATS.json";
	constexpr const char* IMPORTED_MODEL_METADATA_FILE_NAME = "imported_model.json";
	constexpr const char* SKILL_TEST_TIME_METRIC = "SkillTest/TestTimeSeconds";
	constexpr const char* POLICY_COMPATIBILITY_JSON_KEY = "policy_compatibility";
	// NOTE: Bump when policy obs/action semantics change without changing model tensor shapes.
	constexpr int POLICY_COMPATIBILITY_SCHEMA_VERSION = 1;

	std::filesystem::path MakePreferredCopy(std::filesystem::path path) {
		path.make_preferred();
		return path;
	}

	void AddUniquePath(std::vector<std::filesystem::path>& paths, const std::filesystem::path& path) {
		if (path.empty())
			return;

		for (const auto& existing : paths) {
			if (existing == path)
				return;
		}
		paths.push_back(path);
	}

	void AddUniqueString(std::vector<std::string>& strings, const std::string& value) {
		if (value.empty())
			return;

		for (const auto& existing : strings) {
			if (existing == value)
				return;
		}
		strings.push_back(value);
	}

	std::string ReadOptionalString(const nlohmann::json& j, const char* key) {
		if (!j.contains(key) || j.at(key).is_null())
			return {};
		return j.at(key).get<std::string>();
	}

	std::string MakeStaticSelectionKey(const nlohmann::json& entry) {
		const std::string buildName = entry.value("source_build", std::string(GGL::DEFAULT_BUILD_NAME));
		return entry.value("source_agent", std::string()) + "\n"
			+ (buildName.empty() ? GGL::DEFAULT_BUILD_NAME : buildName) + "\n"
			+ entry.value("source_stage", std::string()) + "\n"
			+ entry.value("source_version_id", std::string());
	}

	// A hand-edited entry whose source fields are not strings has no key, rather than throwing.
	std::optional<std::string> TryMakeStaticSelectionKey(const nlohmann::json& entry) {
		try {
			return MakeStaticSelectionKey(entry);
		} catch (const nlohmann::json::exception&) {
			return std::nullopt;
		}
	}

	struct RecentPolicyVersionFolder {
		std::string sourceVersionId;
		uint64_t timesteps = 0;
		std::filesystem::path path;
		std::optional<json> policyCompatibility;
	};

	std::optional<RecentPolicyVersionFolder> TryParseRecentPolicyVersionFolderName(const std::filesystem::path& folder) {
		const std::string folderName = folder.filename().string();
		if (GGL::TrainingStorage::IsAllDigits(folderName)) {
			return RecentPolicyVersionFolder{
				{},
				static_cast<uint64_t>(std::stoull(folderName)),
				folder
			};
		}

		const size_t separator = folderName.find_last_of('_');
		if (separator == std::string::npos || separator + 1 >= folderName.size())
			return std::nullopt;

		const std::string timestepText = folderName.substr(separator + 1);
		if (!GGL::TrainingStorage::IsAllDigits(timestepText))
			return std::nullopt;

		return RecentPolicyVersionFolder{
			folderName.substr(0, separator),
			static_cast<uint64_t>(std::stoull(timestepText)),
			folder
		};
	}

	std::optional<RecentPolicyVersionFolder> ReadRecentPolicyVersionFolder(const std::filesystem::path& folder) {
		std::optional<RecentPolicyVersionFolder> parsed = TryParseRecentPolicyVersionFolderName(folder);

		RecentPolicyVersionFolder result = parsed.value_or(RecentPolicyVersionFolder{});
		result.path = folder;

		const std::filesystem::path jsonPath = folder / STATS_FILE_NAME;
		if (std::filesystem::exists(jsonPath)) {
			std::ifstream fIn(jsonPath);
			if (fIn.good()) {
				const json j = json::parse(fIn, nullptr, false);
				if (!j.is_discarded() && j.is_object()) {
					const std::string sourceVersionId = ReadOptionalString(j, "source_version_id");
					if (!sourceVersionId.empty())
						result.sourceVersionId = sourceVersionId;
					if (j.contains("timesteps") && j.at("timesteps").is_number_unsigned())
						result.timesteps = j.at("timesteps").get<uint64_t>();
					else if (j.contains("timesteps") && j.at("timesteps").is_number_integer()) {
						const int64_t signedTimesteps = j.at("timesteps").get<int64_t>();
						if (signedTimesteps >= 0)
							result.timesteps = static_cast<uint64_t>(signedTimesteps);
					}
					if (j.contains(POLICY_COMPATIBILITY_JSON_KEY) && j.at(POLICY_COMPATIBILITY_JSON_KEY).is_object())
						result.policyCompatibility = j.at(POLICY_COMPATIBILITY_JSON_KEY);
				}
			}
		}

		if (result.timesteps == 0 && !parsed.has_value())
			return std::nullopt;
		if (result.sourceVersionId.empty() && !parsed.has_value())
			return std::nullopt;

		return result;
	}

	std::vector<std::filesystem::path> GetRecentPolicyVersionSearchRoots(
		const std::filesystem::path& saveFolder,
		const GGL::TrainingStorage& storage,
		const std::string& versionId) {

		std::vector<std::filesystem::path> roots = {};
		AddUniquePath(roots, saveFolder);
		for (const auto& root : storage.GetRecentPolicyVersionSearchRoots(versionId))
			AddUniquePath(roots, root);

		return roots;
	}

	std::vector<std::string> GetRecentPolicyVersionSourceIds(
		const std::string& versionId,
		const std::vector<std::string>& configuredSourceIds) {

		std::vector<std::string> sourceIds = {};
		AddUniqueString(sourceIds, versionId);
		for (const std::string& sourceId : configuredSourceIds)
			AddUniqueString(sourceIds, sourceId);
		return sourceIds;
	}

	std::set<std::string> GetLineageSourceIdSet(
		const std::string& versionId,
		const std::vector<std::string>& configuredSourceIds) {

		const std::vector<std::string> sourceIds = GetRecentPolicyVersionSourceIds(versionId, configuredSourceIds);
		return std::set<std::string>(sourceIds.begin(), sourceIds.end());
	}

	std::string TryParseBenchmarkSourceVersionId(const std::string& folderName) {
		const size_t separator = folderName.find('_');
		if (separator == std::string::npos || separator == 0)
			return {};

		const size_t timestepEnd = folderName.find('_', separator + 1);
		if (timestepEnd == std::string::npos || timestepEnd == separator + 1)
			return {};

		const std::string timestepText = folderName.substr(separator + 1, timestepEnd - separator - 1);
		if (!GGL::TrainingStorage::IsAllDigits(timestepText))
			return {};

		return folderName.substr(0, separator);
	}

	bool IsSkillBenchmarkSourceEligible(
		const std::string& sourceVersionId,
		const std::set<std::string>& lineageSourceIds) {

		if (lineageSourceIds.empty())
			return true;

		return !sourceVersionId.empty() && lineageSourceIds.contains(sourceVersionId);
	}

	std::vector<RecentPolicyVersionFolder> FindRecentPolicyVersionFolders(
		const std::filesystem::path& root,
		const std::vector<std::string>& sourceVersionIds) {

		std::vector<RecentPolicyVersionFolder> folders = {};
		if (root.empty() || !std::filesystem::exists(root) || !std::filesystem::is_directory(root))
			return folders;

		std::set<std::string> allowedSourceIds(sourceVersionIds.begin(), sourceVersionIds.end());
		for (const auto& entry : std::filesystem::directory_iterator(root)) {
			if (!entry.is_directory())
				continue;

			std::optional<RecentPolicyVersionFolder> folder = ReadRecentPolicyVersionFolder(entry.path());
			if (!folder.has_value())
				continue;
			if (!allowedSourceIds.empty() && !allowedSourceIds.contains(folder->sourceVersionId))
				continue;

			folders.push_back(folder.value());
		}

		std::sort(folders.begin(), folders.end(), [](const auto& a, const auto& b) {
			if (a.timesteps != b.timesteps)
				return a.timesteps < b.timesteps;
			return a.sourceVersionId < b.sourceVersionId;
		});
		return folders;
	}

	std::vector<RecentPolicyVersionFolder> FindAllRecentPolicyVersionFolders(const std::filesystem::path& root) {
		return FindRecentPolicyVersionFolders(root, {});
	}

	std::filesystem::path ResolveRegistryPath(
		const std::filesystem::path& registryPath,
		const std::string& storedPath) {

		if (storedPath.empty())
			return {};

		std::filesystem::path path(storedPath);
		if (path.is_relative())
			path = registryPath.parent_path() / path;
		path.make_preferred();
		return path;
	}

	std::filesystem::path ResolveBenchmarkFolder(
		const std::filesystem::path& saveFolder,
		const GGL::TrainingStorage& storage) {

		std::filesystem::path benchmarkRoot = storage.GetSkillBenchmarkFolder();
		if (!benchmarkRoot.empty())
			return benchmarkRoot;

		std::filesystem::path versionRoot = saveFolder.parent_path();
		std::filesystem::path stageRoot = versionRoot.parent_path();
		std::filesystem::path agentRoot = stageRoot.parent_path();
		if (!stageRoot.empty() && stageRoot.filename() == GGL::TrainingStorage::TRAINING_STAGES_FOLDER_NAME)
			agentRoot = stageRoot.parent_path();
		if (!agentRoot.empty())
			return MakePreferredCopy(agentRoot / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME);
		if (!stageRoot.empty())
			return MakePreferredCopy(stageRoot / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME);
		return MakePreferredCopy(versionRoot / GGL::TrainingStorage::SKILL_BENCHMARKS_FOLDER_NAME);
	}

	std::string GetStageName(const std::filesystem::path& saveFolder, const GGL::TrainingStorage& storage) {
		std::string stageName = storage.GetStageName();
		if (!stageName.empty())
			return stageName;

		std::filesystem::path stageRoot = saveFolder.parent_path();
		if (!stageRoot.empty() && !stageRoot.filename().empty())
			return stageRoot.filename().string();

		return "unknown-stage";
	}

	struct ImportedModelMetadata {
		std::string label;
		std::string sourceAgentName;
		std::string sourceStageName;
		std::string sourceVersionId;
	};

	ImportedModelMetadata LoadImportedModelMetadata(const std::filesystem::path& rootFolder) {
		ImportedModelMetadata metadata = {};
		metadata.label = rootFolder.filename().string();

		const std::filesystem::path metadataPath = rootFolder / IMPORTED_MODEL_METADATA_FILE_NAME;
		if (!std::filesystem::exists(metadataPath))
			return metadata;

		std::ifstream fIn(metadataPath);
		if (!fIn.good())
			return metadata;

		const json j = json::parse(fIn, nullptr, false);
		if (j.is_discarded() || !j.is_object())
			return metadata;

		metadata.label = j.value("display_name", metadata.label);
		metadata.sourceAgentName = ReadOptionalString(j, "target_agent");
		metadata.sourceStageName = ReadOptionalString(j, "source_stage");
		if (metadata.sourceStageName.empty())
			metadata.sourceStageName = "Imported Models";
		metadata.sourceVersionId = ReadOptionalString(j, "source_version_id");
		if (metadata.sourceVersionId.empty())
			metadata.sourceVersionId = j.value("folder_name", rootFolder.filename().string());

		return metadata;
	}

	std::string GetSourceStageName(const nlohmann::json& benchmarkJson, const std::string& fallbackStageName) {
		std::string stageName = benchmarkJson.value("source_stage_name", std::string());
		if (!stageName.empty())
			return stageName;

		return fallbackStageName;
	}

	float Clamp01(float v) {
		return RS_CLAMP(v, 0.0f, 1.0f);
	}

	float GetConfidence(const GGL::SkillEstimate& estimate, const GGL::SkillTrackerConfig& config) {
		float denom = config.initialSigma - config.minSigma;
		if (denom <= 0)
			return 1.0f;

		return Clamp01(1.0f - ((estimate.sigma - config.minSigma) / denom));
	}

	uint64_t GetSkillRunIntervalTimesteps(const GGL::SkillTrackerConfig& config, int completedGames) {
		int minGames = RS_MAX(1, config.minGames);
		int maxGames = RS_MAX(minGames, config.maxGames);
		uint64_t minTimeSteps = RS_MAX((uint64_t)1, config.minTimeSteps);
		uint64_t maxTimeSteps = RS_MAX(minTimeSteps, config.maxTimeSteps);

		if (maxGames == minGames)
			return minTimeSteps;

		float t = (completedGames - minGames) / (float)(maxGames - minGames);
		t = Clamp01(t);
		return minTimeSteps + (uint64_t)roundf((maxTimeSteps - minTimeSteps) * t);
	}

	GGL::SkillEstimate GetEstimateOrDefault(const GGL::SkillRating& ratings, const std::string& mode, const GGL::SkillTrackerConfig& config) {
		auto itr = ratings.data.find(mode);
		if (itr != ratings.data.end())
			return itr->second;

		return GGL::SkillEstimate{ config.initialMu, config.initialSigma };
	}

	float GetOneSidedExpectedWinProbability(float currentWinProbability) {
		return RS_MAX(currentWinProbability, 1.0f - currentWinProbability);
	}

	float GetBenchmarkExpectedWinWindowThreshold(const GGL::SkillTrackerConfig& config) {
		return RS_CLAMP(config.benchmarkPruneMaxExpectedWinProb, 0.5f, 0.999999f);
	}

	bool IsBenchmarkOutsideExpectedWinWindow(float currentWinProbability, const GGL::SkillTrackerConfig& config) {
		return GetOneSidedExpectedWinProbability(currentWinProbability) >= GetBenchmarkExpectedWinWindowThreshold(config);
	}

	std::string FormatBenchmarkExpectedWinWindowThreshold(const GGL::SkillTrackerConfig& config) {
		std::stringstream stream;
		stream << std::fixed << std::setprecision(1)
			<< (GetBenchmarkExpectedWinWindowThreshold(config) * 100.0f);
		return stream.str();
	}

	std::string FormatProbabilityPercent(float probability) {
		std::stringstream stream;
		stream << std::fixed << std::setprecision(1) << (probability * 100.0f) << "%";
		return stream.str();
	}

	float GetMostOneSidedCurrentWinProbabilityAgainstBenchmark(
		const GGL::SkillRating& currentRatings, const GGL::SkillBenchmark& benchmark,
		const GGL::SkillTrackerConfig& config) {

		float mostOneSidedWinProbability = 0.5f;

		for (const std::string& mode : GGL::SkillRating::ModeNames()) {
			GGL::SkillEstimate current = GetEstimateOrDefault(currentRatings, mode, config);
			GGL::SkillEstimate opponent = GetEstimateOrDefault(benchmark.ratings, mode, config);
			const float currentWinProbability = GetExpectedWinProbability(current, opponent, config);
			if (
				GetOneSidedExpectedWinProbability(currentWinProbability) >
				GetOneSidedExpectedWinProbability(mostOneSidedWinProbability)) {
				mostOneSidedWinProbability = currentWinProbability;
			}
		}

		return mostOneSidedWinProbability;
	}

	/** True when benchmark is the anchor and the current policy is not lopsided against it; a retired anchor is prunable. */
	bool IsProtectedSkillAnchor(
		const GGL::SkillBenchmark& benchmark,
		const std::string& anchorBenchmarkId,
		const GGL::SkillRating& currentRatings,
		const GGL::SkillTrackerConfig& config) {

		if (anchorBenchmarkId.empty() || benchmark.id != anchorBenchmarkId)
			return false;

		const float currentWinProbability = GetMostOneSidedCurrentWinProbabilityAgainstBenchmark(currentRatings, benchmark, config);
		return !GGL::IsWinProbabilityLopsided(currentWinProbability, config.anchorRetireExpectedWinProb);
	}

	/** True when the skill test measures anchor MMR: reference games are on and no user skill reference replaces the anchor. */
	bool UsesAutoSkillAnchor(const GGL::PolicyVersionManager& manager) {
		const bool hasUserReferences = manager.skillReferences && !manager.skillReferences->empty();
		return manager.skill.config.referenceGamesPerOpponent > 0 && !hasUserReferences;
	}

	/** The anchor id pruning must protect; empty while the anchor phase is not in use, so a stale id protects nothing. */
	std::string GetProtectedSkillAnchorId(const GGL::PolicyVersionManager& manager) {
		return UsesAutoSkillAnchor(manager) ? manager.skill.anchorBenchmarkId : std::string();
	}

	struct SkillBenchmarkCandidate {
		int index = -1;
		float distanceFromEven = 0.0f;
		float ratingDelta = 0.0f;
		uint64_t timesteps = 0;
	};

	bool AddSkillBenchmarkCandidate(
		std::vector<int>& selected,
		std::set<int>& selectedIndices,
		const SkillBenchmarkCandidate& candidate,
		int targetCount) {

		if ((int)selected.size() >= targetCount)
			return false;
		if (selectedIndices.contains(candidate.index))
			return false;

		selected.push_back(candidate.index);
		selectedIndices.insert(candidate.index);
		return true;
	}

	void AddClosestSkillBenchmarkCandidates(
		std::vector<int>& selected,
		std::set<int>& selectedIndices,
		std::vector<SkillBenchmarkCandidate> candidates,
		int count,
		int targetCount) {

		std::sort(candidates.begin(), candidates.end(), [](const SkillBenchmarkCandidate& a, const SkillBenchmarkCandidate& b) {
			if (a.distanceFromEven != b.distanceFromEven)
				return a.distanceFromEven < b.distanceFromEven;
			return a.timesteps > b.timesteps;
		});

		for (const SkillBenchmarkCandidate& candidate : candidates) {
			if (count <= 0 || (int)selected.size() >= targetCount)
				break;
			if (AddSkillBenchmarkCandidate(selected, selectedIndices, candidate, targetCount))
				count--;
		}
	}

	void AddRatingCoverageSkillBenchmarkCandidates(
		std::vector<int>& selected,
		std::set<int>& selectedIndices,
		const std::vector<SkillBenchmarkCandidate>& candidates,
		int count,
		int targetCount) {

		std::vector<SkillBenchmarkCandidate> remaining = {};
		for (const SkillBenchmarkCandidate& candidate : candidates)
			if (!selectedIndices.contains(candidate.index))
				remaining.push_back(candidate);

		if (remaining.empty() || count <= 0)
			return;

		if ((int)remaining.size() <= count) {
			for (const SkillBenchmarkCandidate& candidate : remaining)
				AddSkillBenchmarkCandidate(selected, selectedIndices, candidate, targetCount);
			return;
		}

		for (int i = 0; i < count && (int)selected.size() < targetCount; i++) {
			const int idx = count == 1 ? 0 : (int)roundf(i * (remaining.size() - 1) / (float)(count - 1));
			AddSkillBenchmarkCandidate(selected, selectedIndices, remaining[idx], targetCount);
		}
	}

	void AddRandomSkillBenchmarkCandidate(
		std::vector<int>& selected,
		std::set<int>& selectedIndices,
		const std::vector<SkillBenchmarkCandidate>& candidates,
		int targetCount) {

		std::vector<SkillBenchmarkCandidate> remaining = {};
		for (const SkillBenchmarkCandidate& candidate : candidates)
			if (!selectedIndices.contains(candidate.index))
				remaining.push_back(candidate);

		if (remaining.empty() || (int)selected.size() >= targetCount)
			return;

		AddSkillBenchmarkCandidate(
			selected,
			selectedIndices,
			remaining[Math::RandInt(0, (int)remaining.size())],
			targetCount);
	}

	std::vector<int> BuildSkillBenchmarkPoolImpl(
		const GGL::SkillRating& currentRatings,
		const std::string& mode,
		const std::vector<GGL::SkillBenchmark>& benchmarks,
		const GGL::SkillTrackerConfig& config,
		int targetCount) {

		std::vector<SkillBenchmarkCandidate> candidates = {};
		std::vector<SkillBenchmarkCandidate> lowerRated = {};
		std::vector<SkillBenchmarkCandidate> higherRated = {};
		GGL::SkillEstimate current = GetEstimateOrDefault(currentRatings, mode, config);
		for (int i = 0; i < (int)benchmarks.size(); i++) {
			GGL::SkillEstimate estimate = GetEstimateOrDefault(benchmarks[i].ratings, mode, config);
			float currentWinProbability = GetExpectedWinProbability(current, estimate, config);
			if (IsBenchmarkOutsideExpectedWinWindow(currentWinProbability, config))
				continue;

			SkillBenchmarkCandidate candidate = {};
			candidate.index = i;
			candidate.distanceFromEven = fabsf(currentWinProbability - 0.5f);
			candidate.ratingDelta = estimate.mu - current.mu;
			candidate.timesteps = benchmarks[i].timesteps;
			candidates.push_back(candidate);
			if (candidate.ratingDelta < 0.0f)
				lowerRated.push_back(candidate);
			else if (candidate.ratingDelta > 0.0f)
				higherRated.push_back(candidate);
		}

		if (candidates.empty())
			return {};

		targetCount = RS_CLAMP(targetCount, 1, (int)candidates.size());
		if ((int)candidates.size() <= targetCount) {
			std::vector<int> indices = {};
			for (const SkillBenchmarkCandidate& candidate : candidates)
				indices.push_back(candidate.index);
			return indices;
		}

		std::sort(lowerRated.begin(), lowerRated.end(), [](const SkillBenchmarkCandidate& a, const SkillBenchmarkCandidate& b) {
			if (a.ratingDelta != b.ratingDelta)
				return a.ratingDelta > b.ratingDelta;
			return a.timesteps > b.timesteps;
		});
		std::sort(higherRated.begin(), higherRated.end(), [](const SkillBenchmarkCandidate& a, const SkillBenchmarkCandidate& b) {
			if (a.ratingDelta != b.ratingDelta)
				return a.ratingDelta < b.ratingDelta;
			return a.timesteps > b.timesteps;
		});

		std::vector<int> selected = {};
		std::set<int> selectedIndices = {};

		AddClosestSkillBenchmarkCandidates(selected, selectedIndices, candidates, RS_MIN(3, targetCount), targetCount);
		AddRatingCoverageSkillBenchmarkCandidates(selected, selectedIndices, lowerRated, RS_MIN(2, targetCount), targetCount);
		AddRatingCoverageSkillBenchmarkCandidates(selected, selectedIndices, higherRated, RS_MIN(2, targetCount), targetCount);
		AddRandomSkillBenchmarkCandidate(selected, selectedIndices, candidates, targetCount);
		AddClosestSkillBenchmarkCandidates(selected, selectedIndices, candidates, targetCount - (int)selected.size(), targetCount);

		return selected;
	}

	bool IsOlderHalfBenchmark(const std::vector<GGL::SkillBenchmark>& benchmarks, int idx) {
		int newerOrEqualCount = 0;
		const auto& benchmark = benchmarks[idx];
		for (int i = 0; i < benchmarks.size(); i++) {
			if (i == idx)
				continue;

			const auto& other = benchmarks[i];
			if (
				other.createdAt > benchmark.createdAt ||
				(other.createdAt == benchmark.createdAt && other.timesteps >= benchmark.timesteps)
				)
				newerOrEqualCount++;
		}

		return newerOrEqualCount >= (int)benchmarks.size() / 2;
	}

	std::string FormatTimesteps(uint64_t timesteps) {
		std::stringstream stream;

		if (timesteps >= 1'000'000'000ull) {
			stream << std::fixed << std::setprecision(3);
			stream << (timesteps / 1'000'000'000.0) << "B";
		} else if (timesteps >= 1'000'000ull) {
			stream << std::fixed << std::setprecision(1);
			stream << (timesteps / 1'000'000.0) << "M";
		} else if (timesteps >= 1'000ull) {
			stream << std::fixed << std::setprecision(1);
			stream << (timesteps / 1'000.0) << "K";
		} else {
			stream << timesteps;
		}

		return stream.str();
	}

	std::string FormatSeconds(double seconds) {
		std::stringstream stream;
		stream << std::fixed << std::setprecision(2) << seconds << "s";
		return stream.str();
	}

	std::string MakeTimestamp() {
		std::time_t now = std::time(nullptr);
		std::tm tm = {};
#ifdef _WIN32
		localtime_s(&tm, &now);
#else
		localtime_r(&now, &tm);
#endif

		std::stringstream stream;
		stream << std::put_time(&tm, "%Y%m%d-%H%M%S");
		return stream.str();
	}

	float GetBenchmarkSkillPoint(const GGL::SkillBenchmark& benchmark, const GGL::SkillTrackerConfig& config) {
		float total = 0;
		int count = 0;

		for (const std::string& mode : GGL::SkillRating::ModeNames()) {
			auto itr = benchmark.ratings.data.find(mode);
			if (itr == benchmark.ratings.data.end())
				continue;

			total += itr->second.mu;
			count++;
		}

		return count > 0 ? total / count : config.initialMu;
	}

	struct SkillBenchmarkRecord {
		int games = 0;
		int wins = 0;
		int losses = 0;
		int noScoreGames = 0;
	};

	struct SkillMode {
		std::string name;
		int teamSize;
		int games = 0;
		int wins = 0;
		int losses = 0;
		int noScoreGames = 0;
		std::map<std::string, SkillBenchmarkRecord> benchmarkRecords = {};
	};

	struct SkillMatchResult {
		int benchmarkIdx = -1;
		int modeIdx = -1;
		bool scored = false;
		bool currentWon = false;
	};

	using SkillBenchmarkModeKey = std::pair<int, int>;

	std::vector<SkillMode> GetSupportedSkillModes(const RLGC::EnvSetConfig& envSetConfig) {
		int maxTeamSize = envSetConfig.envCreateCfg.maxTeamSize > 0 ? envSetConfig.envCreateCfg.maxTeamSize : 1;
		maxTeamSize = RS_CLAMP(maxTeamSize, 1, 3);

		std::vector<SkillMode> modes = {};
		for (int teamSize = 1; teamSize <= maxTeamSize; teamSize++)
			modes.push_back({ RS_STR(teamSize << "v" << teamSize), teamSize });
		return modes;
	}

	struct SkillPolicyRuntimeInfo {
		RocketSim::CarConfig blueCarConfig = RocketSim::CAR_CONFIG_OCTANE;
		RocketSim::CarConfig orangeCarConfig = RocketSim::CAR_CONFIG_OCTANE;
		int obsSize = 0;
		int numActions = 0;
		bool usesActionMask = false;
	};

	struct SkillPolicyEnvironment {
		std::unique_ptr<RocketSim::Arena> arena;
		std::vector<std::unique_ptr<RLGC::Reward>> rewards;
		std::vector<std::unique_ptr<RLGC::TerminalCondition>> terminalConditions;
		std::unique_ptr<RLGC::ObsBuilder> obsBuilder;
		std::unique_ptr<RLGC::ActionParser> actionParser;
		std::unique_ptr<RLGC::StateSetter> stateSetter;
		void* userInfo = nullptr;
	};

	RLGC::EnvCreateConfig MakeSkillEnvCreateConfig(const RLGC::EnvSetConfig& envSetConfig, int teamSize) {
		RLGC::EnvCreateConfig createConfig = envSetConfig.envCreateCfg;
		createConfig.numArenas = 1;
		createConfig.teamSizeOverride = teamSize;
		return createConfig;
	}

	SkillPolicyEnvironment CreateSkillPolicyEnvironment(const RLGC::EnvSetConfig& envSetConfig, int teamSize) {
		RLGC::EnvCreateResult result = envSetConfig.envCreateFn(0, MakeSkillEnvCreateConfig(envSetConfig, teamSize));
		if (!result.arena || !result.obsBuilder || !result.actionParser)
			RG_ERR_CLOSE("Skill tracker failed to create benchmark runtime components");

		SkillPolicyEnvironment environment = {};
		environment.arena.reset(result.arena);
		environment.rewards.reserve(result.rewards.size());
		for (RLGC::WeightedReward& reward : result.rewards)
			environment.rewards.emplace_back(reward.reward);
		environment.terminalConditions.reserve(result.terminalConditions.size());
		for (RLGC::TerminalCondition* terminalCondition : result.terminalConditions)
			environment.terminalConditions.emplace_back(terminalCondition);
		environment.obsBuilder.reset(result.obsBuilder);
		environment.actionParser.reset(result.actionParser);
		environment.stateSetter.reset(result.stateSetter);
		environment.userInfo = result.userInfo;
		return environment;
	}

	SkillPolicyRuntimeInfo InspectSkillPolicyRuntime(const RLGC::EnvSetConfig& envSetConfig, int teamSize) {
		SkillPolicyEnvironment created = CreateSkillPolicyEnvironment(envSetConfig, teamSize);

		RLGC::FuzzedKickoffState setter;
		setter.ResetArena(created.arena.get());
		RLGC::GameState testState(created.arena.get());
		testState.userInfo = created.userInfo;
		created.obsBuilder->Reset(testState);

		int playersOnTeams[2] = { 0, 0 };
		bool foundBlueConfig = false;
		bool foundOrangeConfig = false;
		SkillPolicyRuntimeInfo info = {};
		for (RocketSim::Car* car : created.arena->_cars) {
			playersOnTeams[(int)car->team]++;
			if (car->team == RocketSim::Team::BLUE && !foundBlueConfig) {
				info.blueCarConfig = car->config;
				foundBlueConfig = true;
			} else if (car->team == RocketSim::Team::ORANGE && !foundOrangeConfig) {
				info.orangeCarConfig = car->config;
				foundOrangeConfig = true;
			}
		}

		if (playersOnTeams[0] != teamSize || playersOnTeams[1] != teamSize) {
			RG_ERR_CLOSE(
				"Skill tracker requested " << teamSize << "v" << teamSize <<
				", but the environment created " << playersOnTeams[0] << "v" << playersOnTeams[1]);
		}

		RLGC::FList obsScratch;
		created.obsBuilder->BuildObsInto(testState.players[0], testState, obsScratch);
		info.obsSize = (int)obsScratch.size();
		info.numActions = created.actionParser->GetActionAmount();
		info.usesActionMask = created.actionParser->UsesActionMask();
		return info;
	}

	json ModelCompatibilityJSON(const GGL::Model& model) {
		json j = {};
		j["name"] = model.modelName ? model.modelName : "";
		j["num_inputs"] = model.config.numInputs;
		j["num_outputs"] = model.config.numOutputs;
		j["layer_sizes"] = model.config.layerSizes;
		j["activation_type"] = (int)model.config.activationType;
		j["optimizer_type"] = (int)model.config.optimType;
		j["add_layer_norm"] = model.config.addLayerNorm;
		j["add_output_layer"] = model.config.addOutputLayer;
		return j;
	}

	json BuildPolicyCompatibilityJSON(GGL::ModelSet& models, const RLGC::EnvSetConfig& envSetConfig) {
		json j = {};
		j["schema_version"] = POLICY_COMPATIBILITY_SCHEMA_VERSION;
		j["tick_skip"] = envSetConfig.tickSkip;
		j["action_delay"] = envSetConfig.actionDelay;
		j["game_mode"] = (int)envSetConfig.envCreateCfg.mode;
		j["max_team_size"] = envSetConfig.envCreateCfg.maxTeamSize;

		j["models"] = json::array();
		for (GGL::Model* model : models)
			j["models"].push_back(ModelCompatibilityJSON(*model));

		if (envSetConfig.envCreateFn) {
			const int teamSize = RS_CLAMP(
				envSetConfig.envCreateCfg.maxTeamSize > 0 ? envSetConfig.envCreateCfg.maxTeamSize : 1,
				1,
				3);
			SkillPolicyEnvironment created = CreateSkillPolicyEnvironment(envSetConfig, teamSize);
			j["obs_builder_type"] = typeid(*created.obsBuilder).name();
			j["action_parser_type"] = typeid(*created.actionParser).name();
			j["action_parser_policy_key"] = created.actionParser->GetPolicyCompatibilityKey();
			j["num_actions"] = created.actionParser->GetActionAmount();
			j["uses_action_mask"] = created.actionParser->UsesActionMask();
		}

		return j;
	}

	GGL::PolicyMatchRuntimeComponents MakeSkillRuntimeComponents(const RLGC::EnvSetConfig& envSetConfig, int teamSize) {
		SkillPolicyEnvironment created = CreateSkillPolicyEnvironment(envSetConfig, teamSize);
		return GGL::PolicyMatchRuntimeComponents{
			std::move(created.obsBuilder),
			std::move(created.actionParser)
		};
	}

	int GetSkillMaxArenas(const GGL::SkillTrackerConfig& config) {
		return RS_MAX(1, config.maxArenas);
	}

	int GetSkillMaxGameSteps(const GGL::SkillTrackerConfig& config, const RLGC::EnvSetConfig& envSetConfig) {
		return RS_MAX(1, (int)ceilf(config.maxNoScoreGameSeconds * 120.0f / envSetConfig.tickSkip));
	}

	/** Builds a skill-test arena from the learner environment, optionally handing its obs/action components to runtime group 0. */
	GGL::PolicyMatchArena MakeSkillPolicyMatchArena(const RLGC::EnvSetConfig& envSetConfig, int teamSize, bool attachRuntimeComponents) {
		SkillPolicyEnvironment created = CreateSkillPolicyEnvironment(envSetConfig, teamSize);
		GGL::PolicyMatchArena arena = {};
		arena.arena = std::move(created.arena);
		arena.userInfo = created.userInfo;
		if (attachRuntimeComponents) {
			arena.runtimeComponents = {
				std::move(created.obsBuilder),
				std::move(created.actionParser)
			};
		}
		arena.ownedRewards = std::move(created.rewards);
		arena.ownedTerminalConditions = std::move(created.terminalConditions);
		arena.ownedStateSetter = std::move(created.stateSetter);
		return arena;
	}

	/** Forwards one of every maxArenas steps to the render sender; empty when nothing is rendering. */
	std::function<void(const GGL::PolicyMatchStep&)> MakeSkillRenderThrottle(GGL::RenderSender* renderSender, int maxArenas) {
		if (!renderSender)
			return {};

		return [renderSender, maxArenas, renderStepCountdown = 0](const GGL::PolicyMatchStep& step) mutable {
			if (renderStepCountdown <= 0) {
				renderSender->Send(step.state);
				renderStepCountdown = maxArenas;
			} else {
				renderStepCountdown--;
			}
		};
	}

	/** Builds policy-match runtimes that play through the learner's own obs builder, action parser and timing. */
	struct SkillRuntimeFactory {
		RLGC::EnvSetConfig envSetConfig;
		SkillPolicyRuntimeInfo runtimeInfo;
		int teamSize = 1;
		bool deterministic = true;
		bool useHalfPrecision = false;
		float policyTemperature = 1.0f;

		GGL::PolicyMatchRuntime Make(GGL::ModelSet* models, int runtimeGroup) const {
			GGL::PolicyMatchRuntime runtime = {};
			runtime.models = models;
			runtime.carConfig = runtimeInfo.blueCarConfig;
			runtime.obsSize = runtimeInfo.obsSize;
			runtime.numActions = runtimeInfo.numActions;
			runtime.usesActionMask = runtimeInfo.usesActionMask;
			runtime.deterministic = deterministic;
			runtime.useHalfPrecision = useHalfPrecision;
			runtime.policyTemperature = policyTemperature;
			runtime.minActionProbability = 0.0f;
			runtime.tickSkip = envSetConfig.tickSkip;
			runtime.actionDelay = envSetConfig.actionDelay;
			runtime.runtimeGroup = runtimeGroup;
			runtime.makeRuntimeComponents = [envSetConfig = envSetConfig, teamSize = teamSize]() {
				return MakeSkillRuntimeComponents(envSetConfig, teamSize);
			};
			return runtime;
		}

		/** A runtime for a fixed series, which assigns every runtime its own group. */
		GGL::PolicyMatchRuntime Make(GGL::ModelSet* models) const {
			const int unassignedGroup = 0;
			return Make(models, unassignedGroup);
		}
	};

	SkillRuntimeFactory MakeSkillRuntimeFactory(
		const RLGC::EnvSetConfig& envSetConfig,
		const GGL::SkillTrackerConfig& config,
		const GGL::PPOLearner& ppo,
		int teamSize) {

		SkillRuntimeFactory factory = {};
		factory.envSetConfig = envSetConfig;
		factory.runtimeInfo = InspectSkillPolicyRuntime(envSetConfig, teamSize);
		factory.teamSize = teamSize;
		factory.deterministic = config.deterministic;
		factory.useHalfPrecision = ppo.config.useHalfPrecision;
		factory.policyTemperature = ppo.config.policyTemperature;
		return factory;
	}

	/** Index of the newest benchmark by creation time, then timesteps; -1 when there are none. */
	int FindNewestBenchmarkIdx(const std::vector<GGL::SkillBenchmark>& benchmarks) {
		int newestIdx = -1;
		for (int i = 0; i < (int)benchmarks.size(); i++) {
			if (newestIdx < 0) {
				newestIdx = i;
				continue;
			}

			const GGL::SkillBenchmark& candidate = benchmarks[i];
			const GGL::SkillBenchmark& newest = benchmarks[newestIdx];
			const bool newerCreatedAt = candidate.createdAt > newest.createdAt;
			const bool sameCreatedAtMoreTimesteps = candidate.createdAt == newest.createdAt && candidate.timesteps > newest.timesteps;
			if (newerCreatedAt || sameCreatedAtMoreTimesteps)
				newestIdx = i;
		}

		return newestIdx;
	}

	bool IsSkillTestInconclusive(const GGL::SkillTrackerConfig& config, int scoredGames, int noScoreGames) {
		int threshold = RS_MAX(1, config.minGames);
		if (noScoreGames < threshold)
			return false;

		int totalGames = scoredGames + noScoreGames;
		if (totalGames <= 0)
			return false;

		float noScoreRatio = noScoreGames / (float)totalGames;
		return noScoreRatio >= Clamp01(config.inconclusiveNoScoreRatio);
	}

	/** PolicyVersionManager::RunFixedSkillSeries for the team size runtimeFactory was built for. */
	std::vector<GGL::FixedSeriesRecord> RunFixedSeriesWithFactory(
		GGL::PolicyVersionManager& manager,
		GGL::PPOLearner* ppo,
		const SkillRuntimeFactory& runtimeFactory,
		const std::vector<GGL::FixedSeriesOpponent>& opponents,
		int gamesPerOpponent,
		const GGL::FixedSeriesOpponent* subject) {

		RG_NO_GRAD;

		const GGL::SkillTrackerConfig& config = manager.skill.config;
		const RLGC::EnvSetConfig& envSetConfig = manager.baseEnvSetConfig;
		const int teamSize = runtimeFactory.teamSize;

		std::vector<GGL::FixedSeriesRecord> records(opponents.size());
		const std::vector<GGL::FixedSeriesGame> games = GGL::BuildFixedSeriesGames(
			(int)opponents.size(),
			gamesPerOpponent,
			config.fixedMatchSeed,
			teamSize);
		if (games.empty())
			return records;

		const int subjectModel = 0;

		std::vector<GGL::PolicyMatchRuntime> runtimes;
		runtimes.reserve(opponents.size() + 1);
		runtimes.push_back(subject ? subject->runtime : runtimeFactory.Make(&ppo->models));
		for (const GGL::FixedSeriesOpponent& opponent : opponents)
			runtimes.push_back(opponent.runtime);

		bool needsIndependentTiming = false;
		for (int runtimeIdx = 0; runtimeIdx < (int)runtimes.size(); runtimeIdx++) {
			GGL::PolicyMatchRuntime& runtime = runtimes[runtimeIdx];

			// Slots cache obs/action components by runtime group, so each runtime gets its own group whatever the caller set.
			runtime.runtimeGroup = runtimeIdx;

			needsIndependentTiming = needsIndependentTiming
				|| runtime.tickSkip != envSetConfig.tickSkip
				|| runtime.actionDelay != envSetConfig.actionDelay;
		}

		std::map<uint64_t, GGL::PolicyMatchSnapshot> snapshotsBySeed = {};
		for (const GGL::FixedSeriesGame& game : games) {
			if (snapshotsBySeed.contains(game.kickoffSeed))
				continue;

			snapshotsBySeed[game.kickoffSeed] = GGL::SamplePolicyMatchStartState(
				GGL::PolicyMatchStartType::Kickoff,
				teamSize,
				runtimeFactory.runtimeInfo.blueCarConfig,
				game.kickoffSeed,
				envSetConfig.envCreateCfg.engine);
		}

		size_t nextGameIdx = 0;
		auto fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pendingTasks, int activeSlots, int targetQueueSize) {
			while (nextGameIdx < games.size() && (int)(pendingTasks.size() + activeSlots) < targetQueueSize) {
				const GGL::FixedSeriesGame& game = games[nextGameIdx];
				const int opponentModel = game.opponentIdx + 1;

				GGL::PolicyMatchTask task = {};
				task.blueModel = game.currentTeam == Team::BLUE ? subjectModel : opponentModel;
				task.orangeModel = game.currentTeam == Team::BLUE ? opponentModel : subjectModel;
				task.blueArenaKey = 0;
				task.orangeArenaKey = 0;
				task.teamSize = teamSize;
				task.snapshot = snapshotsBySeed.at(game.kickoffSeed);
				task.userValues = { game.opponentIdx, (int)game.currentTeam, 0, 0 };
				task.gameId = nextGameIdx;
				pendingTasks.push_back(std::move(task));
				nextGameIdx++;
			}
		};

		// The learner environment's own components only fit runtime group 0 when the current policy is the subject.
		const bool attachRuntimeComponents = subject == nullptr;
		const int maxSkillArenas = GetSkillMaxArenas(config);
		GGL::PolicyMatchRunnerParams params = {
			runtimes,
			ppo->device,
			maxSkillArenas,
			envSetConfig.tickSkip,
			envSetConfig.actionDelay,
			GetSkillMaxGameSteps(config, envSetConfig),
			[&](const GGL::PolicyMatchTask& task) {
				return MakeSkillPolicyMatchArena(envSetConfig, task.teamSize, attachRuntimeComponents);
			},
			fillPendingTasks,
			[&]() { return nextGameIdx >= games.size(); },
			{},
			MakeSkillRenderThrottle(manager.renderSender, maxSkillArenas),
			[&](const GGL::PolicyMatchFinish& finish) {
				const Team subjectTeam = (Team)finish.task.userValues[1];
				GGL::TallyFixedSeriesGame(records[finish.task.userValues[0]], subjectTeam, finish.goalScored, finish.scoringTeam);

				const bool stopRequested = false;
				return stopRequested;
			}
		};
		params.independentSlotTiming = needsIndependentTiming;

		if (needsIndependentTiming)
			GGL::RunPolicyMatchVariableTimingSlots(params);
		else
			GGL::RunPolicyMatchSlots(params);

		return records;
	}

	/** PolicyVersionManager::MakeSkillReferenceOpponent for the team size runtimeFactory was built for. */
	GGL::FixedSeriesOpponent MakeSkillReferenceOpponentWithFactory(
		const SkillRuntimeFactory& runtimeFactory,
		GGL::PolicyVersion& reference) {

		GGL::FixedSeriesOpponent opponent = { reference.label, runtimeFactory.Make(&reference.models) };
		if (!reference.inferenceAgent)
			return opponent;

		const Agent* agent = reference.inferenceAgent.get();
		const std::unique_ptr<RLGC::ActionParser> actionParser(agent->MakeActionParser());

		GGL::PolicyMatchRuntime& runtime = opponent.runtime;
		runtime.obsSize = reference.inferenceObsSize;
		runtime.numActions = reference.inferenceNumActions;
		runtime.usesActionMask = actionParser->UsesActionMask();
		runtime.deterministic = reference.inferenceDeterministic || runtimeFactory.deterministic;
		runtime.tickSkip = agent->tickSkip;
		runtime.actionDelay = agent->actionDelay;
		runtime.makeRuntimeComponents = [agent]() {
			return GGL::PolicyMatchRuntimeComponents{
				std::unique_ptr<RLGC::ObsBuilder>(agent->MakeConfiguredObsBuilder()),
				std::unique_ptr<RLGC::ActionParser>(agent->MakeActionParser())
			};
		};
		return opponent;
	}
}

std::vector<int> GGL::PolicyVersionManager::BuildSkillBenchmarkPool(
	const GGL::SkillRating& currentRatings,
	const std::string& mode,
	const std::vector<GGL::SkillBenchmark>& benchmarks,
	const GGL::SkillTrackerConfig& config,
	int targetCount) {

	return BuildSkillBenchmarkPoolImpl(currentRatings, mode, benchmarks, config, targetCount);
}

void GGL::PolicyVersionManager::UpdateSkillEstimateAgainstFixedOpponent(
	GGL::SkillEstimate& current,
	const GGL::SkillEstimate& opponent,
	bool currentWon,
	const GGL::SkillTrackerConfig& config) {

	float currentSigmaSq = current.sigma * current.sigma;
	float opponentSigmaSq = opponent.sigma * opponent.sigma;
	float betaSq = config.beta * config.beta;
	float c = sqrtf(currentSigmaSq + opponentSigmaSq + 2.0f * betaSq);
	float t = currentWon ? (current.mu - opponent.mu) / c : (opponent.mu - current.mu) / c;
	float cdf = RS_MAX(NormalCDF(t), 1e-6f);
	float v = NormalPDF(t) / cdf;
	float w = v * (v + t);
	float direction = currentWon ? 1.0f : -1.0f;

	current.mu += direction * (currentSigmaSq / c) * v;

	float cSq = c * c;
	float minSigmaSq = config.minSigma * config.minSigma;
	current.sigma = sqrtf(RS_MAX(minSigmaSq, currentSigmaSq * (1.0f - (currentSigmaSq / cSq) * w)));
}

GGL::PolicyVersionManager::PolicyVersionManager(
	std::filesystem::path saveFolder, int maxVersions, uint64_t tsPerVersion,
	const SkillTrackerConfig& skillTrackerConfig, const RLGC::EnvSetConfig& envSetConfig,
	bool managePolicyVersions, RenderSender* renderSender) :
	saveFolder(saveFolder),
	maxVersions(maxVersions), tsPerVersion(tsPerVersion), managePolicyVersions(managePolicyVersions),
	baseEnvSetConfig(envSetConfig), renderSender(renderSender) {

	skill.config = skillTrackerConfig;
	skill.curRatings.EnsureDefaultModes(skill.config);

	if (managePolicyVersions && !std::filesystem::exists(saveFolder))
		std::filesystem::create_directories(saveFolder);
}

namespace {
	std::filesystem::path GetBenchmarkRegistryPath(const std::filesystem::path& benchmarkRoot) {
		return MakePreferredCopy(benchmarkRoot / GGL::TrainingStorage::SKILL_BENCHMARK_REGISTRY_FILE_NAME);
	}
}

GGL::PolicyVersionManager::~PolicyVersionManager() {
	ClearVersions();

	for (auto& benchmark : skill.benchmarks)
		benchmark.models.Free();
}

void GGL::PolicyVersionManager::ClearVersions() {
	for (auto& version : versions)
		version.models.Free();
	versions.clear();
	protectedVersionKeys.clear();
}

GGL::PolicyVersion* GGL::PolicyVersionManager::FindVersion(const PolicyVersionKey& key) {
	for (auto& version : versions) {
		if (version.timesteps == key.timesteps && version.sourceVersionId == key.sourceVersionId)
			return &version;
	}
	return nullptr;
}

void GGL::PolicyVersionManager::SetProtectedVersions(const std::vector<PolicyVersionKey>& keys) {
	protectedVersionKeys = std::set<PolicyVersionKey>(keys.begin(), keys.end());
	PruneInMemoryVersions();
}

void GGL::PolicyVersionManager::PruneInMemoryVersions(const std::optional<PolicyVersionKey>& preserveKey) {
	while (maxVersions >= 0 && versions.size() > static_cast<size_t>(maxVersions)) {
		auto removable = std::find_if(versions.begin(), versions.end(), [&](const PolicyVersion& version) {
			const PolicyVersionKey key = { version.sourceVersionId, version.timesteps };
			return !protectedVersionKeys.contains(key)
				&& (!preserveKey.has_value() || key != preserveKey.value());
		});
		if (removable == versions.end())
			break;

		removable->models.Free();
		versions.erase(removable);
	}
}

GGL::PolicyVersion& GGL::PolicyVersionManager::AddVersion(ModelSet modelsToClone, uint64_t timesteps) {
	RG_NO_GRAD;

	auto models = modelsToClone.CloneAll();
	if (maxVersions > 0) {
		while (versions.size() >= static_cast<size_t>(maxVersions)) {
			auto removable = std::find_if(versions.begin(), versions.end(), [&](const PolicyVersion& version) {
				return !protectedVersionKeys.contains({ version.sourceVersionId, version.timesteps });
			});
			if (removable == versions.end())
				break;

			removable->models.Free();
			versions.erase(removable);
		}
	}

	// Keep the newly added version at the back until the caller finishes filling
	// in metadata. Sorting here can invalidate the intent of the returned
	// reference when multiple versions share the same timestep value.
	versions.push_back(PolicyVersion{
		timesteps,
		models
	});

	return versions.back();
}

void GGL::PolicyVersionManager::PruneSavedVersions(uint64_t curTimesteps) {
	if (!managePolicyVersions || maxVersions < 0)
		return;

	std::vector<RecentPolicyVersionFolder> savedFolders = FindAllRecentPolicyVersionFolders(saveFolder);
	if (savedFolders.size() <= static_cast<size_t>(maxVersions))
		return;

	std::sort(savedFolders.begin(), savedFolders.end(), [curTimesteps](const auto& a, const auto& b) {
		const bool aNewerThanCurrent = a.timesteps > curTimesteps;
		const bool bNewerThanCurrent = b.timesteps > curTimesteps;
		if (aNewerThanCurrent != bNewerThanCurrent)
			return aNewerThanCurrent;
		if (a.timesteps != b.timesteps)
			return a.timesteps < b.timesteps;
		return a.sourceVersionId < b.sourceVersionId;
	});

	size_t savedCount = savedFolders.size();
	size_t removeIndex = 0;
	while (savedCount > static_cast<size_t>(maxVersions) && removeIndex < savedFolders.size()) {
		std::filesystem::remove_all(savedFolders[removeIndex].path);
		savedCount--;
		removeIndex++;
	}
}

void GGL::PolicyVersionManager::SaveVersions(uint64_t curTimesteps) {
	if (!managePolicyVersions)
		return;

	RG_NO_GRAD;

	json policyCompatibility = {};
	bool hasPolicyCompatibility = false;
	for (auto& version : versions) {
		if (!version.saveWithCurrentVersion)
			continue;

		auto versionSaveFolder = saveFolder / TrainingStorage::MakeRecentPolicyVersionFolderName(versionId, version.timesteps);
		if (!std::filesystem::exists(versionSaveFolder)) {
			std::filesystem::create_directories(versionSaveFolder);
			version.models.Save(versionSaveFolder);
		}

		auto jsonPath = versionSaveFolder / STATS_FILE_NAME;
		std::ofstream fOut(jsonPath);
		RG_ASSERT(fOut.good());

		json j = {};
		j["timesteps"] = version.timesteps;
		j["source_version_id"] = versionId;
		if (!hasPolicyCompatibility) {
			policyCompatibility = BuildPolicyCompatibilityJSON(version.models, baseEnvSetConfig);
			hasPolicyCompatibility = true;
		}
		j[POLICY_COMPATIBILITY_JSON_KEY] = policyCompatibility;
		fOut << j.dump(4);
	}

	PruneSavedVersions(curTimesteps);
}

void GGL::PolicyVersionManager::LoadVersions(ModelSet modelsTemplate, uint64_t curTimesteps) {
	RG_NO_GRAD;

	RG_LOG("PolicyVersionManager::LoadVersions():");

	ClearVersions();
	if (managePolicyVersions)
		PruneSavedVersions(curTimesteps);

	json currentCompatibility = BuildPolicyCompatibilityJSON(modelsTemplate, baseEnvSetConfig);
	const std::vector<std::string> sourceIds =
		GetRecentPolicyVersionSourceIds(versionId, recentPolicyVersionSourceIds);
	std::vector<RecentPolicyVersionFolder> savedFolders = {};
	std::set<std::filesystem::path> loadedPaths = {};
	for (const auto& root : GetRecentPolicyVersionSearchRoots(saveFolder, storage, versionId)) {
		for (const auto& folder : FindRecentPolicyVersionFolders(root, sourceIds)) {
			if (loadedPaths.contains(folder.path))
				continue;
			loadedPaths.insert(folder.path);
			savedFolders.push_back(folder);
		}
	}

	std::sort(savedFolders.begin(), savedFolders.end(), [](const auto& a, const auto& b) {
		if (a.timesteps != b.timesteps)
			return a.timesteps < b.timesteps;
		return a.sourceVersionId < b.sourceVersionId;
	});

	uint64_t prunedFutureVersions = 0;
	uint64_t prunedLegacyVersions = 0;
	uint64_t prunedIncompatibleVersions = 0;
	uint64_t prunedBadModelVersions = 0;
	for (const auto& saved : savedFolders) {
		uint64_t savedTimesteps = saved.timesteps;
		if (savedTimesteps > curTimesteps) {
			prunedFutureVersions++;
			if (managePolicyVersions)
				std::filesystem::remove_all(saved.path);
			continue;
		}

		if (!saved.policyCompatibility.has_value()) {
			prunedLegacyVersions++;
			if (managePolicyVersions)
				std::filesystem::remove_all(saved.path);
			continue;
		}

		if (saved.policyCompatibility.value() != currentCompatibility) {
			prunedIncompatibleVersions++;
			if (managePolicyVersions)
				std::filesystem::remove_all(saved.path);
			continue;
		}

		ModelSet loadedModels = modelsTemplate.CloneAll();
		try {
			loadedModels.Load(saved.path, false);
		} catch (const std::exception& e) {
			RG_LOG(" > Skipping bad recent policy version " << saved.path.filename() << ": " << e.what());
			loadedModels.Free();
			prunedBadModelVersions++;
			if (managePolicyVersions)
				std::filesystem::remove_all(saved.path);
			continue;
		} catch (...) {
			RG_LOG(" > Skipping bad recent policy version " << saved.path.filename());
			loadedModels.Free();
			prunedBadModelVersions++;
			if (managePolicyVersions)
				std::filesystem::remove_all(saved.path);
			continue;
		}

		PolicyVersion version = {};
		version.timesteps = savedTimesteps;
		version.models = std::move(loadedModels);
		version.sourceVersionId = saved.sourceVersionId;
		version.rootFolder = saved.path;
		versions.push_back(std::move(version));
		PruneInMemoryVersions();
	}

	SortVersions();

	if (prunedFutureVersions > 0)
		RG_LOG(" > Pruned " << prunedFutureVersions << " future recent policy version(s) from the current segment lineage");
	if (prunedLegacyVersions > 0)
		RG_LOG(" > Pruned " << prunedLegacyVersions << " legacy recent policy version(s) without compatibility metadata");
	if (prunedIncompatibleVersions > 0)
		RG_LOG(" > Pruned " << prunedIncompatibleVersions << " recent policy version(s) with incompatible policy metadata");
	if (prunedBadModelVersions > 0)
		RG_LOG(" > Pruned " << prunedBadModelVersions << " unreadable recent policy version(s)");
	RG_LOG(" > Loaded " << versions.size() << " versions(s)");
}

void GGL::PolicyVersionManager::LoadStaticOpponents(
	ModelSet modelsTemplate,
	const std::function<std::unique_ptr<Agent>(const std::string&)>& agentFactory) {
	RG_NO_GRAD;

	RG_LOG("PolicyVersionManager::LoadStaticOpponents():");

	const bool loadingSkillReferences = registryListKey == SKILL_REFERENCES_LIST_KEY;
	const std::string entryNoun = loadingSkillReferences ? "Skill reference" : "Static opponent";
	const std::string entryNounLower = loadingSkillReferences ? "skill reference" : "static opponent";

	auto addStaticVersion = [&](const std::string& sourceAgentName,
		const std::filesystem::path& modelFolder,
		const json* registryEntry = nullptr) -> PolicyVersion& {
		const bool entryRequestsSourceRuntime = registryEntry
			&& (registryEntry->value("runtime", std::string()) == "source_agent"
				|| (registryEntry->contains("agent_setup") && !(*registryEntry)["agent_setup"].is_null()));
		bool useSourceRuntime = entryRequestsSourceRuntime;
		std::unique_ptr<Agent> sourceAgent;
		if (agentFactory) {
			try {
				sourceAgent = agentFactory(sourceAgentName);
			} catch (...) {
				if (entryRequestsSourceRuntime)
					throw std::runtime_error(entryNoun + " source agent is not registered: " + sourceAgentName);
			}
		}
		if (sourceAgent)
			useSourceRuntime = useSourceRuntime || sourceAgent->inferenceOnly;

		// Legacy registries did not preserve source setup. Keep their old learner-runtime
		// behavior unless the registered source is explicitly inference-only.
		if (!useSourceRuntime) {
			PolicyVersion& version = AddVersion(modelsTemplate, 0);
			version.models.Load(modelFolder, false);
			return version;
		}
		if (!sourceAgent)
			throw std::runtime_error(entryNoun + " source agent is not registered: " + sourceAgentName);
		if (entryRequestsSourceRuntime && registryEntry
			&& (!registryEntry->contains("agent_setup") || (*registryEntry)["agent_setup"].is_null())) {
			throw std::runtime_error(
				entryNoun + " " + sourceAgentName + " is missing saved agent_setup metadata");
		}

		if (registryEntry && registryEntry->contains("agent_setup")
			&& !(*registryEntry)["agent_setup"].is_null()) {
			std::optional<RocketForge::AgentSetupSnapshot> savedSetup =
				RocketForge::AgentSetupSnapshotFromJson((*registryEntry)["agent_setup"]);
			if (!savedSetup.has_value())
				throw std::runtime_error(entryNoun + " has invalid saved agent_setup metadata");
			if (!savedSetup->agentName.empty() && savedSetup->agentName != sourceAgentName) {
				throw std::runtime_error(
					entryNoun + " agent_setup belongs to " + savedSetup->agentName
					+ ", not " + sourceAgentName);
			}
			RocketForge::ApplyAgentSetupSnapshot(*sourceAgent, savedSetup.value());
		}
		sourceAgent->modelPath = modelFolder;
		if (staticOpponentBallPredictionEngine.has_value())
			sourceAgent->ballPredictionEngineVersion = static_cast<int>(staticOpponentBallPredictionEngine.value());

		const bool usesLearnerCadence = sourceAgent->tickSkip == baseEnvSetConfig.tickSkip;
		const bool usesSlowerWholeMultiple =
			baseEnvSetConfig.tickSkip > 0
			&& sourceAgent->tickSkip > baseEnvSetConfig.tickSkip
			&& sourceAgent->tickSkip % baseEnvSetConfig.tickSkip == 0;
		if ((!usesLearnerCadence && !usesSlowerWholeMultiple)
			|| sourceAgent->actionDelay < 0
			|| sourceAgent->actionDelay >= sourceAgent->tickSkip) {
			throw std::runtime_error(
				entryNoun + " " + sourceAgentName + " uses unsupported tickSkip/actionDelay "
				+ std::to_string(sourceAgent->tickSkip) + "/"
				+ std::to_string(sourceAgent->actionDelay)
				+ "; the learner uses "
				+ std::to_string(baseEnvSetConfig.tickSkip) + "/"
				+ std::to_string(baseEnvSetConfig.actionDelay)
				+ " and supports the learner tick skip or a slower whole multiple");
		}
		const int cadenceRatio = std::max(1, sourceAgent->tickSkip / baseEnvSetConfig.tickSkip);
		const int actionApplyStep = sourceAgent->actionDelay / baseEnvSetConfig.tickSkip;
		const int actionApplyTick = sourceAgent->actionDelay % baseEnvSetConfig.tickSkip;
		if (actionApplyTick > baseEnvSetConfig.actionDelay) {
			throw std::runtime_error(
				entryNoun + " " + sourceAgentName + " needs to apply its native action at tick "
				+ std::to_string(actionApplyTick) + " within the learner's "
				+ std::to_string(baseEnvSetConfig.tickSkip) + "-tick step, after the learner action "
				"boundary at tick " + std::to_string(baseEnvSetConfig.actionDelay)
				+ "; this exact timing combination is not supported");
		}
		if (sourceAgent->gameMode != baseEnvSetConfig.envCreateCfg.mode) {
			throw std::runtime_error(
				entryNoun + " " + sourceAgentName
				+ " uses a different game mode from the learner environment");
		}
		const int learnerMaxTeamSize = std::max(1, baseEnvSetConfig.envCreateCfg.maxTeamSize);
		if (learnerMaxTeamSize > sourceAgent->maxTeamSize) {
			RG_LOG(" > " << entryNoun << " " << sourceAgentName
				<< " supports team sizes up to " << sourceAgent->maxTeamSize
				<< "; it will only be sampled in compatible arenas (learner maximum: "
				<< learnerMaxTeamSize << ")");
		}

		std::unique_ptr<RLGC::ActionParser> actionParser(sourceAgent->MakeActionParser());
		const int obsSize = sourceAgent->GetObsSize();
		const int numActions = actionParser->GetActionAmount();
		ModelSet loadedModels = {};
		try {
			sourceAgent->MakeInferenceModels(
				obsSize,
				numActions,
				modelsTemplate["policy"]->device,
				loadedModels);
			loadedModels.Load(modelFolder, false);
		} catch (...) {
			loadedModels.Free();
			throw;
		}

		PolicyVersion version = {};
		version.models = std::move(loadedModels);
		version.inferenceObsSize = obsSize;
		version.inferenceNumActions = numActions;
		version.inferenceDeterministic = sourceAgent->inferenceDeterministic;
		version.inferenceCadenceRatio = cadenceRatio;
		version.inferenceActionApplyStep = actionApplyStep;
		version.inferenceActionApplyTick = actionApplyTick;
		version.inferenceRequiresEarlyAction = actionApplyTick < baseEnvSetConfig.actionDelay;
		version.inferenceAgent = std::move(sourceAgent);
		versions.push_back(std::move(version));

		while (maxVersions >= 0 && versions.size() > static_cast<size_t>(maxVersions)) {
			versions.front().models.Free();
			versions.erase(versions.begin());
		}
		return versions.back();
	};

	if (!staticOpponentsRegistryPath.empty()) {
		ClearVersions();

		if (!std::filesystem::exists(staticOpponentsRegistryPath))
			throw std::runtime_error(entryNoun + " registry not found: " + staticOpponentsRegistryPath.string());

		std::ifstream fIn(staticOpponentsRegistryPath);
		if (!fIn.good())
			RG_ERR_CLOSE("Failed to open " << entryNounLower << " registry: " << staticOpponentsRegistryPath);

		json j = json::parse(fIn, nullptr, false);
		if (j.is_discarded() || !j.is_object())
			RG_ERR_CLOSE("Failed to parse " << entryNounLower << " registry: " << staticOpponentsRegistryPath);

		const json entries = j.value(registryListKey, json::array());
		if (!entries.is_array() || entries.empty())
			throw std::runtime_error(entryNoun + " registry contains no entries: " + staticOpponentsRegistryPath.string());

		std::vector<std::string> loadFailures;
		size_t unresolvedCount = 0;

		for (const json& entry : entries) {
			// The registry keeps hand-written entries its sync could not resolve on this machine, unfilled and without a checkpoint.
			if (entry.value("checkpoint_path", std::string()).empty()) {
				RG_LOG(" > Skipping unresolved " << entryNounLower << " " << entry.value("source_agent", std::string()) << " / "
					<< entry.value("source_version_id", std::string()) << ": it has no checkpoint on this machine");
				unresolvedCount++;
				continue;
			}

			const std::filesystem::path configuredPath =
				ResolveRegistryPath(staticOpponentsRegistryPath, entry.value("checkpoint_path", std::string()));
			const std::filesystem::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(configuredPath);
			if (modelFolder.empty()) {
				loadFailures.push_back(configuredPath.string() + " - no loadable checkpoint");
				continue;
			}

			const size_t previousSize = versions.size();
			try {
				const std::string sourceAgentName = entry.value("source_agent", std::string());
				PolicyVersion& version = addStaticVersion(sourceAgentName, modelFolder, &entry);
				version.label = entry.value("display_name", modelFolder.filename().string());
				version.sourceAgentName = sourceAgentName;
				version.sourceBuildName = entry.value("source_build", std::string(GGL::DEFAULT_BUILD_NAME));
				version.sourceStageName = entry.value("source_stage", std::string());
				version.sourceVersionId = entry.value("source_version_id", std::string());
				version.staticSelectionKey = MakeStaticSelectionKey(entry);
				version.rootFolder = configuredPath;

				// A stored reference rating is exact by definition, so it carries no uncertainty.
				const std::map<std::string, float> storedRatings =
					ReadSkillReferenceRatings(entry.value(SKILL_REFERENCE_RATINGS_KEY, json()), version.label);
				for (const auto& [mode, rating] : storedRatings)
					version.ratings.data[mode] = SkillEstimate{ rating, 0.0f };

				if (loadingSkillReferences) {
					for (const std::string& mode : SkillRating::ModeNames()) {
						if (!storedRatings.contains(mode))
							RG_LOG(" > Skill reference " << version.label << " has no " << mode << " rating; it is pending calibration");
					}
				}

				version.timesteps = 0;
				if (GGL::TrainingStorage::IsAllDigits(modelFolder.filename().string()))
					version.timesteps = (uint64_t)std::stoull(modelFolder.filename().string());
			} catch (const std::exception& e) {
				loadFailures.push_back(configuredPath.string() + " - " + e.what());
				while (versions.size() > previousSize) {
					versions.back().models.Free();
					versions.pop_back();
				}
			} catch (...) {
				loadFailures.push_back(configuredPath.string() + " - unknown load error");
				while (versions.size() > previousSize) {
					versions.back().models.Free();
					versions.pop_back();
				}
			}
		}
		if (!loadFailures.empty()) {
			std::string message = "Failed to load selected " + entryNounLower + "(s):";
			for (const std::string& failure : loadFailures)
				message += "\n - " + failure;
			ClearVersions();
			throw std::runtime_error(message);
		}

		if (unresolvedCount == entries.size())
			throw std::runtime_error(entryNoun + " registry has no entry that resolves on this machine: " + staticOpponentsRegistryPath.string());

		SortVersions();
		RG_LOG(" > Loaded " << versions.size() << " " << entryNounLower << "(s)");
		return;
	}

	if (saveFolder.empty())
		RG_ERR_CLOSE("Cannot load imported models with no save folder");

	if (!std::filesystem::exists(saveFolder))
		std::filesystem::create_directories(saveFolder);

	ClearVersions();

	std::vector<std::filesystem::path> versionFolders = {};
	for (const auto& entry : std::filesystem::directory_iterator(saveFolder)) {
		if (entry.is_directory())
			versionFolders.push_back(entry.path());
	}

	std::sort(versionFolders.begin(), versionFolders.end(), [](const auto& a, const auto& b) {
		return a.filename().string() < b.filename().string();
	});

	for (const auto& rootFolder : versionFolders) {
		const std::filesystem::path modelFolder = GGL::TrainingStorage::ResolveModelFolder(rootFolder);
		if (modelFolder.empty()) {
			RG_LOG(" > Skipping imported model with no loadable checkpoint: " << rootFolder);
			continue;
		}

		try {
			const ImportedModelMetadata metadata = LoadImportedModelMetadata(rootFolder);
			PolicyVersion& version = addStaticVersion(metadata.sourceAgentName, modelFolder);
			version.label = metadata.label;
			version.sourceAgentName = metadata.sourceAgentName;
			version.sourceStageName = metadata.sourceStageName;
			version.sourceVersionId = metadata.sourceVersionId;
			version.rootFolder = rootFolder;
			version.timesteps = 0;
			if (GGL::TrainingStorage::IsAllDigits(modelFolder.filename().string()))
				version.timesteps = (uint64_t)std::stoull(modelFolder.filename().string());
		} catch (const std::exception& e) {
			RG_LOG(" > Skipping incompatible imported model " << rootFolder << " - " << e.what());
			for (auto it = versions.begin(); it != versions.end(); ++it) {
				if (it->rootFolder == rootFolder) {
					it->models.Free();
					versions.erase(it);
					break;
				}
			}
		} catch (...) {
			RG_LOG(" > Skipping incompatible imported model " << rootFolder);
			for (auto it = versions.begin(); it != versions.end(); ++it) {
				if (it->rootFolder == rootFolder) {
					it->models.Free();
					versions.erase(it);
					break;
				}
			}
		}
	}

	SortVersions();

	RG_LOG(" > Loaded " << versions.size() << " imported model(s)");
}

GGL::SkillBenchmark& GGL::PolicyVersionManager::AddBenchmark(ModelSet modelsToClone, uint64_t timesteps) {
	RG_NO_GRAD;

	std::string timestamp = MakeTimestamp();
	std::filesystem::path benchmarkRoot = ResolveBenchmarkFolder(saveFolder, storage);
	std::string stageName = GetStageName(saveFolder, storage);
	if (!sourceStageName.empty())
		stageName = sourceStageName;
	BenchmarkStorageLocation location = TrainingStorage::MakeBenchmarkStorageLocation(
		benchmarkRoot,
		stageName,
		timesteps,
		versionId);

	SkillBenchmark benchmark = {};
	benchmark.id = location.id;
	benchmark.folderName = location.folderName;
	benchmark.createdAt = timestamp;
	benchmark.sourceStageName = stageName;
	benchmark.sourceVersionId = versionId.empty() ? "unversioned" : versionId;
	benchmark.timesteps = timesteps;
	benchmark.folder = location.folder;
	benchmark.models = modelsToClone.CloneAll();
	benchmark.ratings = skill.curRatings;
	benchmark.ratings.EnsureDefaultModes(skill.config);
	benchmark.modelsSaved = false;

	skill.benchmarks.push_back(benchmark);
	return skill.benchmarks.back();
}

void GGL::PolicyVersionManager::SaveBenchmarks() {
	if (!skill.config.enabled)
		return;

	RG_NO_GRAD;

	std::filesystem::path benchmarkRoot = ResolveBenchmarkFolder(saveFolder, storage);
	if (!std::filesystem::exists(benchmarkRoot))
		std::filesystem::create_directories(benchmarkRoot);
	SkillBenchmarkRegistry benchmarkRegistry(
		GetBenchmarkRegistryPath(benchmarkRoot),
		benchmarkRoot.parent_path());

	std::set<std::string> liveFolderNames = {};
	for (auto& benchmark : skill.benchmarks)
		liveFolderNames.insert(benchmark.folderName);

	{
		std::vector<std::filesystem::path> benchmarkEntries;
		for (auto& entry : std::filesystem::directory_iterator(benchmarkRoot))
			if (entry.is_directory())
				benchmarkEntries.push_back(entry.path());
		for (auto& entryPath : benchmarkEntries) {
			if (!liveFolderNames.contains(entryPath.filename().string())) {
				benchmarkRegistry.RemoveByPath(entryPath);
				std::filesystem::remove_all(entryPath);
			}
		}
	}

	json policyCompatibility = {};
	bool hasPolicyCompatibility = false;
	for (auto& benchmark : skill.benchmarks) {
		benchmark.folder = benchmarkRoot / benchmark.folderName;
		std::filesystem::create_directories(benchmark.folder);
		auto jsonPath = benchmark.folder / STATS_FILE_NAME;
		bool shouldWriteStats = !std::filesystem::exists(jsonPath);

		if (!benchmark.modelsSaved) {
			benchmark.models.Save(benchmark.folder);
			benchmark.modelsSaved = true;
			shouldWriteStats = true;
		}

		if (shouldWriteStats) {
			std::ofstream fOut(jsonPath);
			RG_ASSERT(fOut.good());

			json j = {};
			j["benchmark_id"] = benchmark.id;
			j["benchmark_folder_name"] = benchmark.folderName;
			j["created_at"] = benchmark.createdAt;
			j["source_stage_name"] = benchmark.sourceStageName;
			j["source_version_id"] = benchmark.sourceVersionId;
			j["timesteps"] = benchmark.timesteps;
			j["skill_ratings"] = benchmark.ratings.ToJSON();
			if (!hasPolicyCompatibility) {
				policyCompatibility = BuildPolicyCompatibilityJSON(benchmark.models, baseEnvSetConfig);
				hasPolicyCompatibility = true;
			}
			j[POLICY_COMPATIBILITY_JSON_KEY] = policyCompatibility;
			fOut << j.dump(4);
		}

		benchmarkRegistry.UpsertBenchmark({
			benchmark.id,
			benchmark.sourceStageName,
			benchmark.sourceVersionId,
			benchmark.createdAt,
			benchmark.folder.generic_string(),
			benchmark.timesteps
			});
	}
}

void GGL::PolicyVersionManager::LoadBenchmarks(ModelSet modelsTemplate, uint64_t curTimesteps) {
	if (!skill.config.enabled)
		return;

	RG_NO_GRAD;

	RG_LOG("PolicyVersionManager::LoadBenchmarks():");

	for (auto& benchmark : skill.benchmarks)
		benchmark.models.Free();
	skill.benchmarks.clear();

	std::filesystem::path benchmarkRoot = ResolveBenchmarkFolder(saveFolder, storage);
	SkillBenchmarkRegistry benchmarkRegistry(
		GetBenchmarkRegistryPath(benchmarkRoot),
		benchmarkRoot.parent_path());
	skill.anchorBenchmarkId = benchmarkRegistry.GetAnchorBenchmarkId();

	std::vector<std::filesystem::path> benchmarkRoots = storage.GetSkillBenchmarkSearchRoots();
	if (benchmarkRoots.empty() && std::filesystem::exists(benchmarkRoot))
		benchmarkRoots.push_back(benchmarkRoot);

	if (benchmarkRoots.empty()) {
		RG_LOG(" > Loaded 0 benchmark(s)");
		return;
	}

	std::set<std::string> loadedIds = {};
	const std::string fallbackStageName = sourceStageName.empty() ? GetStageName(saveFolder, storage) : sourceStageName;
	const std::string fallbackVersionId = versionId.empty() ? "unversioned" : versionId;
	const std::set<std::string> lineageSourceIds = GetLineageSourceIdSet(versionId, recentPolicyVersionSourceIds);
	json currentCompatibility = BuildPolicyCompatibilityJSON(modelsTemplate, baseEnvSetConfig);
	int skippedOutsideLineage = 0;
	int skippedIncompatible = 0;
	int prunedFutureLineage = 0;
	for (const auto& root : benchmarkRoots) {
		std::vector<std::filesystem::path> benchmarkEntries;
		for (auto& entry : std::filesystem::directory_iterator(root))
			if (entry.is_directory())
				benchmarkEntries.push_back(entry.path());
		for (auto& path : benchmarkEntries) {
			auto jsonPath = path / STATS_FILE_NAME;
			if (!std::filesystem::exists(jsonPath))
				continue;

			json j;
			{
				std::ifstream fIn(jsonPath);
				RG_ASSERT(fIn.good());
				j = json::parse(fIn);
			}

			SkillBenchmark benchmark = {};
			benchmark.id = j.value("benchmark_id", path.filename().string());
			if (loadedIds.contains(benchmark.id))
				continue;

			benchmark.folderName = j.value("benchmark_folder_name", path.filename().string());
			benchmark.sourceVersionId = j.value("source_version_id", std::string());
			if (benchmark.sourceVersionId.empty())
				benchmark.sourceVersionId = TryParseBenchmarkSourceVersionId(benchmark.folderName);
			if (!IsSkillBenchmarkSourceEligible(benchmark.sourceVersionId, lineageSourceIds)) {
				skippedOutsideLineage++;
				continue;
			}

			benchmark.timesteps = j.value("timesteps", (uint64_t)0);
			if (benchmark.timesteps > curTimesteps) {
				prunedFutureLineage++;
				benchmarkRegistry.RemoveByPath(path);
				std::filesystem::remove_all(path);
				continue;
			}
			if (
				j.contains(POLICY_COMPATIBILITY_JSON_KEY) &&
				j.at(POLICY_COMPATIBILITY_JSON_KEY).is_object() &&
				j.at(POLICY_COMPATIBILITY_JSON_KEY) != currentCompatibility) {
				skippedIncompatible++;
				continue;
			}

			benchmark.createdAt = j.value("created_at", std::string());
			benchmark.sourceStageName = GetSourceStageName(j, fallbackStageName);
			if (benchmark.sourceVersionId.empty())
				benchmark.sourceVersionId = fallbackVersionId;
			benchmark.folder = path;
			benchmark.models = modelsTemplate.CloneAll();
			try {
				benchmark.models.Load(path, false);
			} catch (const std::exception& e) {
				RG_LOG("Warning: Skipping benchmark " << path.filename() << " — incompatible model architecture: " << e.what());
				benchmark.models.Free();
				continue;
			}
			if (j.contains("skill_ratings"))
				benchmark.ratings.ReadFromJSON(j["skill_ratings"], skill.config);
			benchmark.ratings.EnsureDefaultModes(skill.config);
			benchmark.modelsSaved = true;

			loadedIds.insert(benchmark.id);
			skill.benchmarks.push_back(benchmark);
			benchmarkRegistry.UpsertBenchmark({
				benchmark.id,
				benchmark.sourceStageName,
				benchmark.sourceVersionId,
				benchmark.createdAt,
				benchmark.folder.generic_string(),
				benchmark.timesteps
				});
		}
	}

	std::sort(skill.benchmarks.begin(), skill.benchmarks.end(), [](const SkillBenchmark& a, const SkillBenchmark& b) {
		if (a.createdAt != b.createdAt)
			return a.createdAt < b.createdAt;
		return a.timesteps < b.timesteps;
	});

	RG_LOG(" > Loaded " << skill.benchmarks.size() << " benchmark(s)");
	if (skippedOutsideLineage > 0) {
		RG_LOG(" > Skipped " << skippedOutsideLineage << " benchmark(s) outside the current segment lineage");
	}
	if (skippedIncompatible > 0) {
		RG_LOG(" > Skipped " << skippedIncompatible << " benchmark(s) with incompatible policy metadata");
	}
	if (prunedFutureLineage > 0) {
		RG_LOG(" > Pruned " << prunedFutureLineage << " future benchmark(s) from the current segment lineage");
	}
}

bool GGL::PolicyVersionManager::EnsureInitialSkillBenchmark(ModelSet modelsToClone, uint64_t curTimesteps) {
	if (!skill.config.enabled)
		return false;

	RG_NO_GRAD;

	skill.curRatings.EnsureDefaultModes(skill.config);
	PruneBenchmarks();
	if (!skill.benchmarks.empty())
		return false;

	RG_LOG("Skill tracker has no valid startup benchmarks; saving current model as the initial benchmark anchor...");
	AddBenchmark(modelsToClone, curTimesteps);
	SaveBenchmarks();

	skill.nextRunTimestep = curTimesteps + GetSkillRunIntervalTimesteps(skill.config, 0);
	RG_LOG(
		"Next skill test scheduled at " << FormatTimesteps(skill.nextRunTimestep) <<
		" (in " << FormatTimesteps(skill.nextRunTimestep - curTimesteps) << " timesteps)");
	return true;
}

void GGL::PolicyVersionManager::PruneBenchmarks() {
	int maxBenchmarks = RS_MAX(1, skill.config.maxSkillBenchmarks);
	std::filesystem::path benchmarkRoot = ResolveBenchmarkFolder(saveFolder, storage);
	SkillBenchmarkRegistry benchmarkRegistry(
		GetBenchmarkRegistryPath(benchmarkRoot),
		benchmarkRoot.parent_path());

	auto removeBenchmarkAt = [&](int removeIdx) {
		auto removePath = skill.benchmarks[removeIdx].folder;
		skill.benchmarks[removeIdx].models.Free();
		skill.benchmarks.erase(skill.benchmarks.begin() + removeIdx);
		benchmarkRegistry.RemoveByPath(removePath);
		std::filesystem::remove_all(removePath);
	};

	const std::string protectedAnchorId = GetProtectedSkillAnchorId(*this);
	auto isAnchor = [&](int benchmarkIdx) {
		return IsProtectedSkillAnchor(skill.benchmarks[benchmarkIdx], protectedAnchorId, skill.curRatings, skill.config);
	};

	for (int i = 0; i < (int)skill.benchmarks.size();) {
		float currentWinProb = GetMostOneSidedCurrentWinProbabilityAgainstBenchmark(skill.curRatings, skill.benchmarks[i], skill.config);
		if (!isAnchor(i) && IsBenchmarkOutsideExpectedWinWindow(currentWinProb, skill.config)) {
			RG_LOG(
				"Pruning SkillTest benchmark " << skill.benchmarks[i].id <<
				": expected result is outside the " <<
				FormatBenchmarkExpectedWinWindowThreshold(skill.config) <<
				"% expected win/loss window.");
			removeBenchmarkAt(i);
			continue;
		}
		i++;
	}

	while ((int)skill.benchmarks.size() > maxBenchmarks) {
		const int newestIdx = FindNewestBenchmarkIdx(skill.benchmarks);
		int lowestIdx = 0;
		int highestIdx = 0;

		std::vector<float> skillPoints(skill.benchmarks.size());
		for (int i = 0; i < skill.benchmarks.size(); i++) {
			skillPoints[i] = GetBenchmarkSkillPoint(skill.benchmarks[i], skill.config);

			if (skillPoints[i] < skillPoints[lowestIdx])
				lowestIdx = i;
			if (skillPoints[i] > skillPoints[highestIdx])
				highestIdx = i;
		}

		int removeIdx = -1;
		float bestDistance = FLT_MAX;

		float bestStaleOneSidedWinProb = -1.0f;
		for (int i = 0; i < skill.benchmarks.size(); i++) {
			if (i == newestIdx || isAnchor(i))
				continue;

			float currentWinProb = GetMostOneSidedCurrentWinProbabilityAgainstBenchmark(skill.curRatings, skill.benchmarks[i], skill.config);
			const float oneSidedWinProb = GetOneSidedExpectedWinProbability(currentWinProb);
			bool tooPredictable = IsBenchmarkOutsideExpectedWinWindow(currentWinProb, skill.config);
			if (!tooPredictable || !IsOlderHalfBenchmark(skill.benchmarks, i))
				continue;

			bool betterCandidate = oneSidedWinProb > bestStaleOneSidedWinProb;
			if (!betterCandidate && oneSidedWinProb == bestStaleOneSidedWinProb && removeIdx != -1) {
				const auto& cur = skill.benchmarks[i];
				const auto& best = skill.benchmarks[removeIdx];
				betterCandidate =
					cur.createdAt < best.createdAt ||
					(cur.createdAt == best.createdAt && skillPoints[i] < skillPoints[removeIdx]);
			}

			if (betterCandidate) {
				bestStaleOneSidedWinProb = oneSidedWinProb;
				removeIdx = i;
			}
		}

		for (int i = 0; removeIdx == -1 && i < skill.benchmarks.size(); i++) {
			if (i == newestIdx || i == lowestIdx || i == highestIdx || isAnchor(i))
				continue;

			float nearestDistance = FLT_MAX;
			for (int j = 0; j < skill.benchmarks.size(); j++) {
				if (i == j)
					continue;

				nearestDistance = RS_MIN(nearestDistance, fabsf(skillPoints[i] - skillPoints[j]));
			}

			bool betterCandidate = nearestDistance < bestDistance;
			if (!betterCandidate && nearestDistance == bestDistance && removeIdx != -1) {
				const auto& cur = skill.benchmarks[i];
				const auto& best = skill.benchmarks[removeIdx];
				betterCandidate =
					cur.createdAt < best.createdAt ||
					(cur.createdAt == best.createdAt && skillPoints[i] < skillPoints[removeIdx]);
			}

			if (betterCandidate) {
				bestDistance = nearestDistance;
				removeIdx = i;
			}
		}

		if (removeIdx == -1) {
			for (int i = 0; i < skill.benchmarks.size(); i++) {
				if (i != newestIdx && !isAnchor(i)) {
					removeIdx = i;
					break;
				}
			}
		}

		if (removeIdx == -1)
			return;

		removeBenchmarkAt(removeIdx);
	}
}

void GGL::PolicyVersionManager::SortVersions() {
	auto fnCompareVersions = [](const PolicyVersion& a, const PolicyVersion& b) {
		return a.timesteps < b.timesteps;
	};

	std::sort(versions.begin(), versions.end(), fnCompareVersions);
}

uint64_t GGL::PolicyVersionManager::GetTimestepsSinceLastSkillTest(uint64_t totalTimesteps, uint64_t lastRunTimestep) {
	if (lastRunTimestep == 0 || totalTimesteps <= lastRunTimestep)
		return 0;

	return totalTimesteps - lastRunTimestep;
}

namespace {
	/** Lowest, median and highest benchmark by the mode's rating, without repeats. */
	std::vector<int> SelectCalibrationBenchmarks(
		const std::vector<GGL::SkillBenchmark>& benchmarks,
		const std::string& mode,
		const GGL::SkillTrackerConfig& config) {

		std::vector<int> byRating(benchmarks.size());
		std::iota(byRating.begin(), byRating.end(), 0);
		std::sort(byRating.begin(), byRating.end(), [&](int a, int b) {
			return GetEstimateOrDefault(benchmarks[a].ratings, mode, config).mu <
				GetEstimateOrDefault(benchmarks[b].ratings, mode, config).mu;
		});

		std::vector<int> selected = {};
		if (byRating.empty())
			return selected;

		for (int position : { (int)byRating.size() - 1, (int)byRating.size() / 2, 0 }) {
			if (std::find(selected.begin(), selected.end(), byRating[position]) == selected.end())
				selected.push_back(byRating[position]);
		}

		return selected;
	}

	/**
	 * Rates a skill reference in one mode from its games against the calibration benchmarks and stores the result.
	 * Returns nullopt, after logging why, while calibration is pending.
	 */
	std::optional<float> CalibrateSkillReference(
		GGL::PolicyVersionManager& manager,
		GGL::PPOLearner* ppo,
		const SkillRuntimeFactory& runtimeFactory,
		GGL::PolicyVersion& reference,
		const SkillMode& mode,
		int& gamesPlayed) {

		const GGL::SkillTrackerConfig& config = manager.skill.config;
		const std::vector<int> benchmarkIndices = SelectCalibrationBenchmarks(manager.skill.benchmarks, mode.name, config);

		std::vector<GGL::FixedSeriesOpponent> opponents = {};
		for (int benchmarkIdx : benchmarkIndices) {
			GGL::SkillBenchmark& benchmark = manager.skill.benchmarks[benchmarkIdx];
			opponents.push_back({ benchmark.id, runtimeFactory.Make(&benchmark.models) });
		}

		const size_t scheduledGames = GGL::BuildFixedSeriesGames(
			(int)opponents.size(),
			config.referenceCalibrationGames,
			config.fixedMatchSeed,
			mode.teamSize).size();
		RG_LOG(
			"Calibrating " << reference.label << " " << mode.name << " against " << opponents.size() <<
			" benchmark(s) (" << scheduledGames << " games)...");

		const GGL::FixedSeriesOpponent subject = MakeSkillReferenceOpponentWithFactory(runtimeFactory, reference);
		const std::vector<GGL::FixedSeriesRecord> records =
			RunFixedSeriesWithFactory(manager, ppo, runtimeFactory, opponents, config.referenceCalibrationGames, &subject);

		std::vector<GGL::CalibrationSample> samples = {};
		for (int opponentIdx = 0; opponentIdx < (int)records.size(); opponentIdx++) {
			const GGL::FixedSeriesRecord& record = records[opponentIdx];
			gamesPlayed += record.wins + record.losses + record.noScoreGames;
			RG_LOG(
				"  vs " << opponents[opponentIdx].label <<
				": Record=" << record.wins << "-" << record.losses << " | No-score=" << record.noScoreGames);

			const GGL::SkillBenchmark& benchmark = manager.skill.benchmarks[benchmarkIndices[opponentIdx]];
			const float benchmarkRating = GetEstimateOrDefault(benchmark.ratings, mode.name, config).mu;
			if (const std::optional<GGL::CalibrationSample> sample = GGL::MakeCalibrationSample(benchmarkRating, record, config))
				samples.push_back(*sample);
		}

		if (samples.empty()) {
			RG_LOG(
				reference.label << " " << mode.name <<
				": calibration pending — no calibration game was scored; retrying next skill test.");
			return std::nullopt;
		}

		const std::optional<float> rating = GGL::CombineCalibrationSamples(samples, config.anchorRetireExpectedWinProb, config);
		if (!rating.has_value()) {
			RG_LOG(
				reference.label << " " << mode.name <<
				": calibration pending — every benchmark is lopsided against it; retrying next skill test.");
			return std::nullopt;
		}

		const std::filesystem::path& registryPath = manager.skillReferencesRegistryPath;
		GGL::PolicyVersionManager::WriteSkillReferenceRating(registryPath, reference.staticSelectionKey, mode.name, *rating);
		reference.ratings.data[mode.name] = GGL::SkillEstimate{ *rating, 0.0f };
		RG_LOG(
			"SkillTest " << mode.name << " calibrated reference " << reference.label <<
			" at " << std::fixed << std::setprecision(1) << *rating << " MMR from " << samples.size() << " benchmark(s)");
		return rating;
	}

	/**
	 * Plays the current policy against each user skill reference per mode, calibrating references with no rating yet.
	 * runtimeFactories holds one factory per mode, in mode order. Returns the games played.
	 */
	int RunSkillReferenceSeries(
		GGL::PolicyVersionManager& manager,
		GGL::PPOLearner* ppo,
		GGL::Report& report,
		const std::vector<SkillMode>& modes,
		const std::vector<SkillRuntimeFactory>& runtimeFactories) {

		const GGL::SkillTrackerConfig& config = manager.skill.config;
		int gamesPlayed = 0;

		for (int modeIdx = 0; modeIdx < (int)modes.size(); modeIdx++) {
			const SkillMode& mode = modes[modeIdx];
			const SkillRuntimeFactory& runtimeFactory = runtimeFactories[modeIdx];

			std::vector<GGL::PolicyVersion*> measuredReferences = {};
			for (GGL::PolicyVersion& reference : *manager.skillReferences) {
				if (reference.inferenceAgent && reference.inferenceAgent->maxTeamSize < mode.teamSize) {
					RG_LOG(
						"Skipping SkillTest " << mode.name << " reference " << reference.label <<
						": it supports team sizes up to " << reference.inferenceAgent->maxTeamSize);
					continue;
				}

				const bool hasRating = reference.ratings.data.contains(mode.name);
				if (!hasRating && !CalibrateSkillReference(manager, ppo, runtimeFactory, reference, mode, gamesPlayed).has_value())
					continue;

				measuredReferences.push_back(&reference);
			}

			if (measuredReferences.empty())
				continue;

			std::vector<GGL::FixedSeriesOpponent> opponents = {};
			for (GGL::PolicyVersion* reference : measuredReferences)
				opponents.push_back(MakeSkillReferenceOpponentWithFactory(runtimeFactory, *reference));

			const std::vector<GGL::FixedSeriesRecord> records =
				RunFixedSeriesWithFactory(manager, ppo, runtimeFactory, opponents, config.referenceGamesPerOpponent, nullptr);

			std::set<std::string> reportedLabels = {};
			for (int referenceIdx = 0; referenceIdx < (int)measuredReferences.size(); referenceIdx++) {
				const GGL::PolicyVersion& reference = *measuredReferences[referenceIdx];
				const GGL::FixedSeriesRecord& record = records[referenceIdx];
				gamesPlayed += record.wins + record.losses + record.noScoreGames;

				const std::string label = GGL::PolicyVersionManager::MakeReferenceMetricLabel(reference.label);
				if (!reportedLabels.insert(label).second) {
					RG_LOG(
						"Warning: SkillTest " << mode.name << " reference " << reference.label << " shares the metric label " << label <<
						" with another reference; its MMR overwrites theirs. Give the references distinct display names.");
				}

				GGL::PolicyVersionManager::ReportFixedOpponentMMR(
					report,
					"SkillTest/Reference/" + label + "/" + mode.name,
					"SkillTest/ReferenceSigma/" + label + "/" + mode.name,
					"reference " + reference.label,
					mode.name,
					reference.ratings.data.at(mode.name).mu,
					record,
					config);
			}
		}

		return gamesPlayed;
	}

	/**
	 * Selects and persists the anchor benchmark, then plays the current policy against it per mode.
	 * runtimeFactories holds one factory per mode, in mode order. Returns the games played.
	 */
	int RunSkillAnchorSeries(
		GGL::PolicyVersionManager& manager,
		GGL::PPOLearner* ppo,
		GGL::Report& report,
		const std::vector<SkillMode>& modes,
		const std::vector<SkillRuntimeFactory>& runtimeFactories) {

		const GGL::SkillTrackerConfig& config = manager.skill.config;
		const int anchorIdx = GGL::PolicyVersionManager::SelectAnchorBenchmark(
			manager.skill.benchmarks,
			manager.skill.anchorBenchmarkId,
			manager.skill.curRatings,
			config);

		const std::string anchorId = anchorIdx >= 0 ? manager.skill.benchmarks[anchorIdx].id : std::string();
		if (anchorId != manager.skill.anchorBenchmarkId) {
			manager.skill.anchorBenchmarkId = anchorId;

			std::filesystem::path benchmarkRoot = ResolveBenchmarkFolder(manager.saveFolder, manager.storage);
			GGL::SkillBenchmarkRegistry benchmarkRegistry(GetBenchmarkRegistryPath(benchmarkRoot), benchmarkRoot.parent_path());
			benchmarkRegistry.SetAnchorBenchmarkId(anchorId);
		}

		if (anchorIdx < 0)
			return 0;

		GGL::SkillBenchmark& anchor = manager.skill.benchmarks[anchorIdx];
		int gamesPlayed = 0;

		for (int modeIdx = 0; modeIdx < (int)modes.size(); modeIdx++) {
			const SkillMode& mode = modes[modeIdx];
			const SkillRuntimeFactory& runtimeFactory = runtimeFactories[modeIdx];
			const std::vector<GGL::FixedSeriesOpponent> opponents = { { anchor.id, runtimeFactory.Make(&anchor.models) } };

			const GGL::FixedSeriesRecord record =
				RunFixedSeriesWithFactory(manager, ppo, runtimeFactory, opponents, config.referenceGamesPerOpponent, nullptr)[0];
			gamesPlayed += record.wins + record.losses + record.noScoreGames;

			const bool reported = GGL::PolicyVersionManager::ReportFixedOpponentMMR(
				report,
				"SkillTest/Anchor/" + mode.name,
				"SkillTest/AnchorSigma/" + mode.name,
				"anchor " + anchor.id,
				mode.name,
				GetEstimateOrDefault(anchor.ratings, mode.name, config).mu,
				record,
				config);
			if (reported)
				report["SkillTest/AnchorTimesteps/" + mode.name] = anchor.timesteps;
		}

		return gamesPlayed;
	}
}

bool GGL::PolicyVersionManager::ReportFixedOpponentMMR(
	Report& report,
	const std::string& ratingKey,
	const std::string& sigmaKey,
	const std::string& opponentName,
	const std::string& modeName,
	float baseRating,
	const FixedSeriesRecord& record,
	const SkillTrackerConfig& config) {

	std::stringstream line;
	line << "SkillTest " << modeName << " vs " << opponentName <<
		": Record=" << record.wins << "-" << record.losses << " | No-score=" << record.noScoreGames;

	const std::optional<SkillEstimate> mmr = EstimateFixedOpponentMMR(baseRating, record, config);
	if (!mmr.has_value()) {
		RG_LOG(line.str() << " | MMR=n/a (no scored games)");
		return false;
	}

	report[ratingKey] = mmr->mu;
	report[sigmaKey] = mmr->sigma;
	RG_LOG(line.str() << " | MMR=" << std::fixed << std::setprecision(1) << mmr->mu << " +/- " << mmr->sigma);

	const SkillEstimate gap = { mmr->mu - baseRating, mmr->sigma };
	const float winProbability = GetGapWinProbability(gap, config);
	if (IsWinProbabilityLopsided(winProbability, config.anchorRetireExpectedWinProb)) {
		RG_LOG(
			"Warning: SkillTest " << modeName << " " << opponentName <<
			" has lost resolution: the current policy's expected win probability is " << FormatProbabilityPercent(winProbability) <<
			"; a closer opponent would measure MMR more precisely.");
	}

	return true;
}

void GGL::PolicyVersionManager::RunSkillMatches(PPOLearner* ppo, Report& report, uint64_t totalTimesteps) {
	RG_NO_GRAD;

	skill.curRatings.EnsureDefaultModes(skill.config);

	PruneBenchmarks();

	if (skill.benchmarks.empty()) {
		RG_LOG("Skill tracker has no benchmarks; saving current model as the initial benchmark anchor...");
		AddBenchmark(ppo->GetPolicyModels(), totalTimesteps);
		SaveBenchmarks();
		std::vector<SkillMode> modes = GetSupportedSkillModes(baseEnvSetConfig);
		for (const SkillMode& mode : modes) {
			SkillEstimate& estimate = skill.curRatings.GetEstimate(mode.name, skill.config);
			report["SkillTest/Rating/" + mode.name] = estimate.mu;
			report["SkillTest/RatingSigma/" + mode.name] = estimate.sigma;
			report["SkillTest/Games/" + mode.name] = 0;
			report["SkillTest/Draws/" + mode.name] = 0;
		}
		report["SkillTest/Benchmarks/Count"] = skill.benchmarks.size();
		skill.nextRunTimestep = totalTimesteps + GetSkillRunIntervalTimesteps(skill.config, 0);
		report["SkillTest/NextRunTimestep"] = skill.nextRunTimestep;
		RG_LOG(
			"Next skill test scheduled at " << FormatTimesteps(skill.nextRunTimestep) <<
			" (in " << FormatTimesteps(skill.nextRunTimestep - totalTimesteps) << " timesteps)");
		return;
	}

	// Every phase of the test adds each game it played here, scored or not, before the interval is computed.
	int totalSkillTestGames = 0;

	std::vector<SkillMode> modes = GetSupportedSkillModes(baseEnvSetConfig);
	std::vector<SkillRuntimeFactory> runtimeFactories = {};
	for (const SkillMode& mode : modes)
		runtimeFactories.push_back(MakeSkillRuntimeFactory(baseEnvSetConfig, skill.config, *ppo, mode.teamSize));

	SkillRating prevCurRatings = skill.curRatings;
	std::vector<SkillRating> prevBenchmarkRatings = {};
	for (auto& benchmark : skill.benchmarks)
		prevBenchmarkRatings.push_back(benchmark.ratings);

	// A first test (or legacy stats without a last-run timestep) has no carried uncertainty to grow from.
	const uint64_t elapsedTimesteps = GetTimestepsSinceLastSkillTest(totalTimesteps, skill.lastRunTimestep);
	auto seedCurRatings = [&]() {
		if (skill.lastRunTimestep == 0)
			skill.curRatings.SeedForSkillTest(prevCurRatings, skill.config);
		else
			skill.curRatings.SeedForSkillTestWithDrift(prevCurRatings, elapsedTimesteps, skill.config);
	};
	seedCurRatings();

	int totalScoredGames = 0;
	int totalNoScoreGames = 0;

	// Unlike the totals above, this keeps games finished after a mode completed and games the replay later discards.
	int mainPhaseGamesPlayed = 0;
	std::vector<SkillMatchResult> skillMatchResults = {};
	bool inconclusive = false;
	bool discardedAllScoredGames = false;
	std::string inconclusiveReason = {};
	int minGames = RS_MAX(1, skill.config.minGames);
	int maxGames = RS_MAX(minGames, skill.config.maxGames);
	Timer skillTestTimer = {};
	const int maxSkillArenas = GetSkillMaxArenas(skill.config);
	const int initialSkillWindow = RS_MIN(maxSkillArenas, RS_MAX(64, minGames * 4));
	const int minSkillWindow = initialSkillWindow;
	const int skillBenchmarkPoolTarget = RS_MIN(8, (int)skill.benchmarks.size());

	std::stringstream modeList;
	for (int i = 0; i < (int)modes.size(); i++) {
		if (i > 0)
			modeList << ", ";
		modeList << modes[i].name;
	}
	RG_LOG(
		"Running skill test at " << FormatTimesteps(totalTimesteps) <<
		" (Modes=" << modeList.str() << ")\n" <<
		"Max Arenas=" << maxSkillArenas <<
		", Initial In-Flight=" << initialSkillWindow <<
		", Min Games=" << minGames << ", Max Games=" << maxGames <<
		", Target Confidence=" << std::fixed << std::setprecision(1) << (skill.config.targetConfidence * 100.0f) << "%");

	auto fnPrintProgress = [&]() {
		RG_LOG(
			"----- SkillTest Progress: " <<
			" (Scored Games=" << totalScoredGames << ") -----");

		for (auto& mode : modes) {
			SkillEstimate& estimate = skill.curRatings.GetEstimate(mode.name, skill.config);
			std::stringstream line;
			line << std::left << std::setw(3) << mode.name << ": Games=" << std::setw(5) << mode.games
				<< " | Record: " << mode.wins << "-" << mode.losses
				<< " | Skill rating=" << std::fixed << std::setprecision(1) << estimate.mu
				<< " | Confidence=" << std::fixed << std::setprecision(1) << (GetConfidence(estimate, skill.config) * 100.0f);
			RG_LOG(line.str());
		}
	};

	auto fnPrintAnchorRecords = [&]() {
		for (auto& mode : modes) {
			if (mode.benchmarkRecords.empty())
				continue;

			RG_LOG("----- SkillTest Anchor Records: " << mode.name << " -----");
			for (const auto& pair : mode.benchmarkRecords) {
				const std::string& benchmarkId = pair.first;
				const GGL::SkillBenchmark* benchmark = nullptr;
				for (const GGL::SkillBenchmark& candidate : skill.benchmarks) {
					if (candidate.id == benchmarkId) {
						benchmark = &candidate;
						break;
					}
				}
				if (!benchmark)
					continue;

				const SkillBenchmarkRecord& record = pair.second;
				if (record.games == 0 && record.noScoreGames == 0)
					continue;

				std::string observedWin = record.games > 0 ?
					FormatProbabilityPercent(record.wins / (float)record.games) :
					"n/a";
				const SkillEstimate currentEstimate = GetEstimateOrDefault(skill.curRatings, mode.name, skill.config);
				const SkillEstimate benchmarkEstimate = GetEstimateOrDefault(benchmark->ratings, mode.name, skill.config);
				RG_LOG(
					"  " << benchmark->id <<
					": Rating=" << std::fixed << std::setprecision(1) << benchmarkEstimate.mu <<
					" | Record=" << record.wins << "-" << record.losses <<
					" | Observed win=" << observedWin <<
					" | Final rating-implied win=" <<
					FormatProbabilityPercent(GetExpectedWinProbability(currentEstimate, benchmarkEstimate, skill.config)));
			}
		}
	};

	int nextProgressPrint = skill.config.progressPrintInterval > 0 ? skill.config.progressPrintInterval : INT_MAX;

	for (int modeIdx = 0; modeIdx < (int)modes.size(); modeIdx++) {
		auto& mode = modes[modeIdx];
		if (inconclusive)
			break;

		std::vector<int> benchmarkPool = BuildSkillBenchmarkPool(
			prevCurRatings,
			mode.name,
			skill.benchmarks,
			skill.config,
			skillBenchmarkPoolTarget);
		if (benchmarkPool.empty()) {
			skill.curRatings.GetEstimate(mode.name, skill.config) =
				GetEstimateOrDefault(prevCurRatings, mode.name, skill.config);
			RG_LOG(
				"Skipping SkillTest mode " << mode.name << ": all " << skill.benchmarks.size() <<
				" benchmark(s) are outside the " <<
				FormatBenchmarkExpectedWinWindowThreshold(skill.config) <<
				"% expected win/loss window.");
			continue;
		}

		RG_LOG("SkillTest " << mode.name << " selected " << benchmarkPool.size() << " fixed benchmark anchor(s):");
		for (int benchmarkIdx : benchmarkPool) {
			const SkillEstimate currentEstimate = GetEstimateOrDefault(prevCurRatings, mode.name, skill.config);
			const SkillEstimate benchmarkEstimate = GetEstimateOrDefault(skill.benchmarks[benchmarkIdx].ratings, mode.name, skill.config);
			RG_LOG(
				"  - " << skill.benchmarks[benchmarkIdx].id <<
				" | Rating=" << std::fixed << std::setprecision(1) << benchmarkEstimate.mu <<
				" | Sigma=" << std::fixed << std::setprecision(1) << benchmarkEstimate.sigma <<
				" | Prior rating-implied win=" <<
				FormatProbabilityPercent(GetExpectedWinProbability(currentEstimate, benchmarkEstimate, skill.config)));
		}

		const SkillRuntimeFactory& runtimeFactory = runtimeFactories[modeIdx];
		const int startSnapshotCount = RS_CLAMP(initialSkillWindow, 1, 32);
		std::vector<GGL::PolicyMatchSnapshot> startSnapshots = {};
		startSnapshots.reserve(startSnapshotCount);
		for (int i = 0; i < startSnapshotCount; i++) {
			startSnapshots.push_back(GGL::SamplePolicyMatchStartState(
				GGL::PolicyMatchStartType::Kickoff,
				mode.teamSize,
				runtimeFactory.runtimeInfo.blueCarConfig,
				baseEnvSetConfig.envCreateCfg.engine));
		}
		int nextStartSnapshot = 0;
		int benchmarkPoolCursor = benchmarkPool.empty() ? 0 : Math::RandInt(0, (int)benchmarkPool.size());
		std::map<int, int> benchmarkAssignmentCounts = {};

		auto selectBenchmarkForAssignment = [&]() {
			int bestBenchmarkIdx = benchmarkPool[benchmarkPoolCursor % benchmarkPool.size()];
			int bestOffset = 0;
			int bestCount = benchmarkAssignmentCounts[bestBenchmarkIdx];
			for (int offset = 1; offset < (int)benchmarkPool.size(); offset++) {
				const int candidateIdx = benchmarkPool[(benchmarkPoolCursor + offset) % benchmarkPool.size()];
				const int candidateCount = benchmarkAssignmentCounts[candidateIdx];
				if (candidateCount < bestCount) {
					bestBenchmarkIdx = candidateIdx;
					bestOffset = offset;
					bestCount = candidateCount;
				}
			}

			benchmarkPoolCursor = (benchmarkPoolCursor + bestOffset + 1) % benchmarkPool.size();
			return bestBenchmarkIdx;
		};

		auto modeComplete = [&]() {
			const bool hasEnoughGames = mode.games >= minGames;
			const SkillEstimate& estimate = skill.curRatings.GetEstimate(mode.name, skill.config);
			const bool reachedConfidence = GetConfidence(estimate, skill.config) >= skill.config.targetConfidence;
			return mode.games >= maxGames || (hasEnoughGames && reachedConfidence) || inconclusive;
		};

		auto desiredInFlightTarget = [&]() {
			if (modeComplete())
				return 0;

			if (mode.games < minGames)
				return initialSkillWindow;

			SkillEstimate& estimate = skill.curRatings.GetEstimate(mode.name, skill.config);
			float confidence = GetConfidence(estimate, skill.config);
			float targetConfidence = RS_MAX(0.0001f, skill.config.targetConfidence);
			float missingConfidence = Clamp01((skill.config.targetConfidence - confidence) / targetConfidence);
			int window = minSkillWindow + (int)roundf((maxSkillArenas - minSkillWindow) * missingConfidence);
			window = RS_CLAMP(window, minSkillWindow, maxSkillArenas);

			int remainingScoredGames = maxGames - mode.games;
			if (remainingScoredGames <= 0)
				return 0;
			return RS_MAX(1, RS_MIN(window, remainingScoredGames));
		};

		std::vector<GGL::PolicyMatchRuntime> runtimes;
		runtimes.reserve(skill.benchmarks.size() + 1);
		runtimes.push_back(runtimeFactory.Make(&ppo->models, 0));
		for (SkillBenchmark& benchmark : skill.benchmarks)
			runtimes.push_back(runtimeFactory.Make(&benchmark.models, (int)runtimes.size()));

		auto fillPendingTasks = [&](std::vector<GGL::PolicyMatchTask>& pendingTasks, int activeSlots, int targetQueueSize) {
			int dynamicTarget = RS_MIN(targetQueueSize, desiredInFlightTarget());
			while (!modeComplete() && (int)(pendingTasks.size() + activeSlots) < dynamicTarget) {
				const int benchmarkIdx = selectBenchmarkForAssignment();
				const int perBenchmarkAssignment = benchmarkAssignmentCounts[benchmarkIdx]++;
				const Team currentTeam = (perBenchmarkAssignment % 2) == 0 ? Team::BLUE : Team::ORANGE;
				const int currentModel = 0;
				const int benchmarkModel = benchmarkIdx + 1;

				GGL::PolicyMatchTask task = {};
				task.blueModel = currentTeam == Team::BLUE ? currentModel : benchmarkModel;
				task.orangeModel = currentTeam == Team::BLUE ? benchmarkModel : currentModel;
				task.blueArenaKey = 0;
				task.orangeArenaKey = 0;
				task.teamSize = mode.teamSize;
				task.snapshot = startSnapshots[nextStartSnapshot++ % startSnapshots.size()];
				task.userValues = {
					benchmarkIdx,
					(int)currentTeam,
					modeIdx,
					0
				};
				pendingTasks.push_back(std::move(task));
			}
		};

		const bool attachRuntimeComponents = true;
		GGL::RunPolicyMatchSlots({
			runtimes,
			ppo->device,
			maxSkillArenas,
			baseEnvSetConfig.tickSkip,
			baseEnvSetConfig.actionDelay,
			GetSkillMaxGameSteps(skill.config, baseEnvSetConfig),
			[&](const GGL::PolicyMatchTask& task) {
				return MakeSkillPolicyMatchArena(baseEnvSetConfig, task.teamSize, attachRuntimeComponents);
			},
			fillPendingTasks,
			modeComplete,
			{},
			MakeSkillRenderThrottle(renderSender, maxSkillArenas),
			[&](const GGL::PolicyMatchFinish& finish) {
				const int benchmarkIdx = finish.task.userValues[0];
				const Team currentTeam = (Team)finish.task.userValues[1];
				const int resultModeIdx = finish.task.userValues[2];
				mainPhaseGamesPlayed++;

				if (modeComplete())
					return true;

				if (finish.goalScored) {
					const bool currentWon = finish.scoringTeam == currentTeam;
					skillMatchResults.push_back({ benchmarkIdx, resultModeIdx, true, currentWon });
					SkillEstimate& currentEstimate = skill.curRatings.GetEstimate(mode.name, skill.config);
					const SkillEstimate benchmarkEstimate = GetEstimateOrDefault(skill.benchmarks[benchmarkIdx].ratings, mode.name, skill.config);
					UpdateSkillEstimateAgainstFixedOpponent(currentEstimate, benchmarkEstimate, currentWon, skill.config);
					SkillBenchmarkRecord& record = mode.benchmarkRecords[skill.benchmarks[benchmarkIdx].id];
					record.games++;
					if (currentWon) {
						record.wins++;
						mode.wins++;
					} else {
						record.losses++;
						mode.losses++;
					}

					mode.games++;
					totalScoredGames++;
				} else {
					skillMatchResults.push_back({ benchmarkIdx, resultModeIdx, false, false });
					mode.benchmarkRecords[skill.benchmarks[benchmarkIdx].id].noScoreGames++;
					mode.noScoreGames++;
					totalNoScoreGames++;

					if (IsSkillTestInconclusive(skill.config, totalScoredGames, totalNoScoreGames)) {
						inconclusive = true;
						inconclusiveReason =
							"too many games reached the " +
							std::to_string((int)skill.config.maxNoScoreGameSeconds) +
							"s no-score limit (" + std::to_string(totalNoScoreGames) +
							" no-score, " + std::to_string(totalScoredGames) + " scored)";
					}
				}

				int totalCompletedGames = totalScoredGames + totalNoScoreGames;
				if (totalCompletedGames >= nextProgressPrint) {
					fnPrintProgress();
					nextProgressPrint += skill.config.progressPrintInterval;
				}

				return modeComplete();
			}
		});
	}

	if (inconclusive) {
		skill.curRatings = prevCurRatings;
		for (int i = 0; i < (int)skill.benchmarks.size() && i < (int)prevBenchmarkRatings.size(); i++)
			skill.benchmarks[i].ratings = prevBenchmarkRatings[i];

		RG_LOG("Skill test inconclusive: " << inconclusiveReason << ". Ratings and benchmarks were left unchanged.");
	}

	if (!inconclusive) {
		std::set<int> excludedBenchmarkIndices = {};

		auto replaySkillResults = [&]() {
			std::vector<SkillMode> replayedModes = GetSupportedSkillModes(baseEnvSetConfig);
			std::vector<bool> modeHasScoredGame(replayedModes.size(), false);
			int replayedScoredGames = 0;
			int replayedNoScoreGames = 0;

			skill.curRatings = prevCurRatings;
			for (int i = 0; i < (int)skill.benchmarks.size() && i < (int)prevBenchmarkRatings.size(); i++)
				skill.benchmarks[i].ratings = prevBenchmarkRatings[i];
			seedCurRatings();

			for (const SkillMatchResult& result : skillMatchResults) {
				if (result.benchmarkIdx < 0 || result.benchmarkIdx >= (int)skill.benchmarks.size())
					continue;
				if (result.modeIdx < 0 || result.modeIdx >= (int)replayedModes.size())
					continue;
				if (excludedBenchmarkIndices.contains(result.benchmarkIdx))
					continue;

				SkillMode& mode = replayedModes[result.modeIdx];
				if (result.scored) {
					SkillEstimate& currentEstimate = skill.curRatings.GetEstimate(mode.name, skill.config);
					const SkillEstimate benchmarkEstimate = GetEstimateOrDefault(skill.benchmarks[result.benchmarkIdx].ratings, mode.name, skill.config);
					UpdateSkillEstimateAgainstFixedOpponent(currentEstimate, benchmarkEstimate, result.currentWon, skill.config);
					SkillBenchmarkRecord& record = mode.benchmarkRecords[skill.benchmarks[result.benchmarkIdx].id];
					record.games++;
					if (result.currentWon) {
						record.wins++;
						mode.wins++;
					} else {
						record.losses++;
						mode.losses++;
					}

					mode.games++;
					replayedScoredGames++;
					modeHasScoredGame[result.modeIdx] = true;
				} else {
					mode.benchmarkRecords[skill.benchmarks[result.benchmarkIdx].id].noScoreGames++;
					mode.noScoreGames++;
					replayedNoScoreGames++;
				}
			}

			for (int i = 0; i < (int)replayedModes.size(); i++) {
				if (!modeHasScoredGame[i]) {
					skill.curRatings.GetEstimate(replayedModes[i].name, skill.config) =
						GetEstimateOrDefault(prevCurRatings, replayedModes[i].name, skill.config);
				}
			}

			modes = replayedModes;
			totalScoredGames = replayedScoredGames;
			totalNoScoreGames = replayedNoScoreGames;
		};

		auto findNewlyInvalidBenchmarks = [&]() {
			bool foundNewInvalidBenchmark = false;
			std::set<SkillBenchmarkModeKey> checkedPairs = {};
			for (const SkillMatchResult& result : skillMatchResults) {
				if (!result.scored)
					continue;
				if (result.benchmarkIdx < 0 || result.benchmarkIdx >= (int)skill.benchmarks.size())
					continue;
				if (result.modeIdx < 0 || result.modeIdx >= (int)modes.size())
					continue;
				if (excludedBenchmarkIndices.contains(result.benchmarkIdx))
					continue;

				SkillBenchmarkModeKey key = { result.benchmarkIdx, result.modeIdx };
				if (checkedPairs.contains(key))
					continue;
				checkedPairs.insert(key);

				const std::string& modeName = modes[result.modeIdx].name;
				SkillEstimate current = GetEstimateOrDefault(skill.curRatings, modeName, skill.config);
				SkillEstimate opponent = GetEstimateOrDefault(skill.benchmarks[result.benchmarkIdx].ratings, modeName, skill.config);
				float currentWinProb = GetExpectedWinProbability(current, opponent, skill.config);
				if (IsBenchmarkOutsideExpectedWinWindow(currentWinProb, skill.config)) {
					excludedBenchmarkIndices.insert(result.benchmarkIdx);
					foundNewInvalidBenchmark = true;
				}
			}
			return foundNewInvalidBenchmark;
		};

		while (findNewlyInvalidBenchmarks())
			replaySkillResults();

		if (!excludedBenchmarkIndices.empty()) {
			int discardedScoredGames = 0;
			int discardedNoScoreGames = 0;
			for (const SkillMatchResult& result : skillMatchResults) {
				if (!excludedBenchmarkIndices.contains(result.benchmarkIdx))
					continue;
				if (result.scored)
					discardedScoredGames++;
				else
					discardedNoScoreGames++;
			}

			RG_LOG(
				"Discarding SkillTest data from " << excludedBenchmarkIndices.size() <<
				" benchmark(s): post-test expected result was outside the " <<
				FormatBenchmarkExpectedWinWindowThreshold(skill.config) <<
				"% expected win/loss window (" << discardedScoredGames <<
				" scored, " << discardedNoScoreGames << " no-score game(s) removed).");

			std::filesystem::path benchmarkRoot = ResolveBenchmarkFolder(saveFolder, storage);
			SkillBenchmarkRegistry benchmarkRegistry(
				GetBenchmarkRegistryPath(benchmarkRoot),
				benchmarkRoot.parent_path());
			const std::string protectedAnchorId = GetProtectedSkillAnchorId(*this);
			for (int i = (int)skill.benchmarks.size() - 1; i >= 0; i--) {
				if (!excludedBenchmarkIndices.contains(i))
					continue;

				if (IsProtectedSkillAnchor(skill.benchmarks[i], protectedAnchorId, skill.curRatings, skill.config)) {
					RG_LOG(
						"Keeping SkillTest anchor benchmark " << skill.benchmarks[i].id <<
						" although its post-test data was discarded.");
					continue;
				}

				std::filesystem::path removePath = skill.benchmarks[i].folder;
				RG_LOG(
					"Pruning SkillTest benchmark " << skill.benchmarks[i].id <<
					": post-test data was outside the " <<
					FormatBenchmarkExpectedWinWindowThreshold(skill.config) <<
					"% expected win/loss window.");
				skill.benchmarks[i].models.Free();
				skill.benchmarks.erase(skill.benchmarks.begin() + i);
				benchmarkRegistry.RemoveByPath(removePath);
				std::filesystem::remove_all(removePath);
			}

			if (discardedScoredGames > 0 && totalScoredGames == 0) {
				discardedAllScoredGames = true;
				skill.curRatings = prevCurRatings;
				RG_LOG("Skill test discarded all scored benchmark data; ratings were left unchanged and no new benchmark will be saved.");
			}
		}
	}

	if (skill.config.referenceGamesPerOpponent > 0) {
		Timer referenceTimer = {};
		totalSkillTestGames += UsesAutoSkillAnchor(*this)
			? RunSkillAnchorSeries(*this, ppo, report, modes, runtimeFactories)
			: RunSkillReferenceSeries(*this, ppo, report, modes, runtimeFactories);
		report["SkillTest/ReferenceSeconds"] = referenceTimer.Elapsed();
	}

	fnPrintProgress();
	fnPrintAnchorRecords();

	for (const SkillMode& mode : modes) {
		SkillEstimate& estimate = skill.curRatings.GetEstimate(mode.name, skill.config);
		report["SkillTest/Rating/" + mode.name] = estimate.mu;
		report["SkillTest/RatingSigma/" + mode.name] = estimate.sigma;
	}

	for (auto& mode : modes)
		report["SkillTest/Games/" + mode.name] = mode.games;
	for (auto& mode : modes)
		report["SkillTest/Draws/" + mode.name] = mode.noScoreGames;

	if (!inconclusive && !discardedAllScoredGames) {
		AddBenchmark(ppo->GetPolicyModels(), totalTimesteps);
		PruneBenchmarks();
		SaveBenchmarks();
	}
	if (inconclusive || discardedAllScoredGames)
		report["SkillTest/Inconclusive"] = 1;

	const double skillTestTimeSeconds = skillTestTimer.Elapsed();
	report[SKILL_TEST_TIME_METRIC] = skillTestTimeSeconds;
	RG_LOG("----------------------------\n" <<
		"Skill test completed in " << FormatSeconds(skillTestTimeSeconds));

	skill.lastRunTimestep = totalTimesteps;

	totalSkillTestGames += mainPhaseGamesPlayed;
	skill.nextRunTimestep = totalTimesteps + GetSkillRunIntervalTimesteps(skill.config, totalSkillTestGames);
	report["SkillTest/NextRunTimestep"] = skill.nextRunTimestep;
	RG_LOG(
		"Next skill test scheduled at " << FormatTimesteps(skill.nextRunTimestep) <<
		" (in " << FormatTimesteps(skill.nextRunTimestep - totalTimesteps) << " timesteps)");
}

int GGL::PolicyVersionManager::FindBenchmarkById(const std::vector<SkillBenchmark>& benchmarks, const std::string& id) {
	for (int i = 0; i < (int)benchmarks.size(); i++) {
		if (benchmarks[i].id == id)
			return i;
	}

	return -1;
}

int GGL::PolicyVersionManager::SelectAnchorBenchmark(
	const std::vector<SkillBenchmark>& benchmarks,
	const std::string& currentAnchorId,
	const SkillRating& currentRatings,
	const SkillTrackerConfig& config) {

	auto isLopsided = [&](const SkillBenchmark& benchmark) {
		const float currentWinProbability = GetMostOneSidedCurrentWinProbabilityAgainstBenchmark(currentRatings, benchmark, config);
		return IsWinProbabilityLopsided(currentWinProbability, config.anchorRetireExpectedWinProb);
	};

	const int currentAnchorIdx = currentAnchorId.empty() ? -1 : FindBenchmarkById(benchmarks, currentAnchorId);
	if (currentAnchorIdx >= 0 && !isLopsided(benchmarks[currentAnchorIdx]))
		return currentAnchorIdx;

	int selectedIdx = -1;
	for (int i = 0; i < (int)benchmarks.size(); i++) {
		if (isLopsided(benchmarks[i]))
			continue;

		if (selectedIdx < 0 || GetBenchmarkSkillPoint(benchmarks[i], config) > GetBenchmarkSkillPoint(benchmarks[selectedIdx], config))
			selectedIdx = i;
	}

	const std::string selectedId = selectedIdx >= 0 ? benchmarks[selectedIdx].id : "none (every benchmark is lopsided)";
	if (currentAnchorId.empty()) {
		RG_LOG("SkillTest anchor: selecting " << selectedId);
	} else if (currentAnchorIdx < 0) {
		RG_LOG("SkillTest anchor " << currentAnchorId << " is not loaded on this lineage; selecting " << selectedId);
	} else {
		RG_LOG(
			"SkillTest anchor " << currentAnchorId << " retired: the current policy's expected result is outside the " <<
			std::fixed << std::setprecision(1) << (config.anchorRetireExpectedWinProb * 100.0f) <<
			"% win/loss window; selecting " << selectedId);
	}

	return selectedIdx;
}

std::string GGL::PolicyVersionManager::MakeReferenceMetricLabel(const std::string& displayName) {
	std::string label = {};
	bool pendingSeparator = false;
	for (char c : displayName) {
		if (c == '/' || std::isspace((unsigned char)c)) {
			pendingSeparator = true;
			continue;
		}

		if (pendingSeparator && !label.empty())
			label += '-';
		pendingSeparator = false;
		label += c;
	}

	return label;
}

GGL::FixedSeriesOpponent GGL::PolicyVersionManager::MakeSkillReferenceOpponent(
	PPOLearner* ppo,
	PolicyVersion& reference,
	int teamSize) const {

	return MakeSkillReferenceOpponentWithFactory(MakeSkillRuntimeFactory(baseEnvSetConfig, skill.config, *ppo, teamSize), reference);
}

void GGL::PolicyVersionManager::WriteSkillReferenceRating(
	const std::filesystem::path& registryPath,
	const std::string& selectionKey,
	const std::string& mode,
	float rating) {
	std::ifstream fIn(registryPath);
	json registry = fIn.good() ? json::parse(fIn, nullptr, false) : json();
	fIn.close();

	if (!registry.is_object() || !registry.contains(SKILL_REFERENCES_LIST_KEY) || !registry[SKILL_REFERENCES_LIST_KEY].is_array()) {
		RG_LOG("PolicyVersionManager::WriteSkillReferenceRating(): cannot read skill reference registry " << registryPath);
		return;
	}

	for (json& entry : registry[SKILL_REFERENCES_LIST_KEY]) {
		if (!entry.is_object() || TryMakeStaticSelectionKey(entry) != selectionKey)
			continue;

		json& ratings = entry[SKILL_REFERENCE_RATINGS_KEY];
		if (!ratings.is_object())
			ratings = json::object();
		ratings[mode] = rating;

		// Write beside the registry and rename over it, so a crash mid-write never leaves a truncated registry.
		const std::filesystem::path tempPath = registryPath.string() + ".tmp";
		std::ofstream fOut(tempPath);
		fOut << registry.dump(DetectJsonIndent(registryPath));
		fOut.close();
		if (fOut.fail()) {
			RG_LOG("PolicyVersionManager::WriteSkillReferenceRating(): failed to write " << tempPath);
			std::error_code removeError;
			std::filesystem::remove(tempPath, removeError);
			return;
		}

		std::error_code renameError;
		std::filesystem::rename(tempPath, registryPath, renameError);
		if (renameError) {
			RG_LOG(
				"PolicyVersionManager::WriteSkillReferenceRating(): failed to replace " << registryPath <<
				" - " << renameError.message());
			std::error_code removeError;
			std::filesystem::remove(tempPath, removeError);
		}
		return;
	}

	std::string readableKey = selectionKey;
	std::replace(readableKey.begin(), readableKey.end(), '\n', '/');
	RG_LOG("PolicyVersionManager::WriteSkillReferenceRating(): no skill reference " << readableKey
		<< " in " << registryPath << "; its " << mode << " rating was not saved");
}

std::vector<GGL::FixedSeriesRecord> GGL::PolicyVersionManager::RunFixedSkillSeries(
	PPOLearner* ppo,
	const std::vector<FixedSeriesOpponent>& opponents,
	int teamSize,
	int gamesPerOpponent,
	const FixedSeriesOpponent* subject) {

	const SkillRuntimeFactory runtimeFactory = MakeSkillRuntimeFactory(baseEnvSetConfig, skill.config, *ppo, teamSize);
	return RunFixedSeriesWithFactory(*this, ppo, runtimeFactory, opponents, gamesPerOpponent, subject);
}

void GGL::PolicyVersionManager::OnIteration(class PPOLearner* ppo, Report& report, int64_t totalTimesteps, int64_t prevTotalTimesteps) {
	if (managePolicyVersions && ((totalTimesteps / tsPerVersion > prevTotalTimesteps / tsPerVersion) || (prevTotalTimesteps == 0))) {
		PolicyVersion& version = AddVersion(ppo->GetPolicyModels(), totalTimesteps);
		version.sourceStageName = sourceStageName;
		version.sourceVersionId = versionId;
		version.saveWithCurrentVersion = true;
		PruneInMemoryVersions(PolicyVersionKey{ version.sourceVersionId, version.timesteps });
	}

	if (skill.config.enabled) {
		if ((uint64_t)totalTimesteps >= skill.nextRunTimestep) {
			RunSkillMatches(ppo, report, totalTimesteps);
		}

		report["SkillTest/NextRunTimestep"] = skill.nextRunTimestep;
	}
}

void GGL::PolicyVersionManager::AddRunningStatsToJSON(nlohmann::json& json) {
	if (skill.config.enabled) {
		json["skill_ratings"] = skill.curRatings.ToJSON();
		json["skill_next_run_timestep"] = skill.nextRunTimestep;
		json["skill_last_run_timestep"] = skill.lastRunTimestep;
	}
}

void GGL::PolicyVersionManager::LoadRunningStatsFromJSON(const nlohmann::json& json) {
	if (skill.config.enabled) {
		if (json.contains("skill_ratings"))
			skill.curRatings.ReadFromJSON(json["skill_ratings"], skill.config);
		skill.nextRunTimestep = json.value("skill_next_run_timestep", (uint64_t)0);
		skill.lastRunTimestep = json.value("skill_last_run_timestep", (uint64_t)0);
		skill.curRatings.EnsureDefaultModes(skill.config);
	}
}
