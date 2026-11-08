#include "BuildStorage.h"

#include <GigaLearnCPP/TrainingStorage.h>
#include <GigaLearnCPP/SegmentVersioning.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cwctype>
#include <mutex>
#include <stdexcept>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using nlohmann::json;

namespace
{
	std::recursive_mutex StorageMetadataMutex;

	uint64_t StableStoragePathHash(const fs::path& path)
	{
		std::error_code error;
		fs::path normalized = fs::weakly_canonical(path, error);
		if (error) {
			error.clear();
			normalized = fs::absolute(path, error);
		}
		if (error)
			normalized = path;

		normalized = normalized.lexically_normal();

#ifdef _WIN32
		std::wstring text = normalized.native();
		std::transform(text.begin(), text.end(), text.begin(), [](wchar_t value) {
			return (wchar_t)std::towlower(value);
		});
#else
		const std::string text = normalized.native();
#endif

		// FNV-1a: stable across processes and runs, unlike std::hash.
		const uint64_t fnvOffsetBasis = 1469598103934665603ULL;
		const uint64_t fnvPrime = 1099511628211ULL;
		uint64_t hash = fnvOffsetBasis;
		for (const auto value : text) {
			hash ^= (uint64_t)value;
			hash *= fnvPrime;
		}

		return hash;
	}

	class StorageMetadataGuard
	{
	public:
		explicit StorageMetadataGuard(const fs::path& storageRoot) :
			_processLock(StorageMetadataMutex)
		{
			// All BuildStorage instances for one agent share this OS lock,
			// including instances owned by separate GUI and trainer processes.
			std::ostringstream hashText;
			hashText << std::hex << StableStoragePathHash(storageRoot);
			const std::string hashString = hashText.str();

#ifdef _WIN32
			const std::wstring name = L"Local\\RocketForge.BuildStorage." + std::wstring(hashString.begin(), hashString.end());
			_mutex = CreateMutexW(nullptr, FALSE, name.c_str());
			if (!_mutex)
				throw std::runtime_error("BuildStorage: failed to create storage mutex");

			const DWORD waitResult = WaitForSingleObject(_mutex, INFINITE);
			if (waitResult != WAIT_OBJECT_0 && waitResult != WAIT_ABANDONED) {
				CloseHandle(_mutex);
				_mutex = nullptr;
				throw std::runtime_error("BuildStorage: failed to acquire storage mutex");
			}
#else
			const fs::path lockPath = fs::temp_directory_path() / ("rocketforge-buildstorage-" + hashString + ".lock");
			if (_lockDepth == 0) {
				_lockFileDescriptor = open(lockPath.c_str(), O_CREAT | O_RDWR, 0600);
				if (_lockFileDescriptor < 0 || flock(_lockFileDescriptor, LOCK_EX) != 0) {
					if (_lockFileDescriptor >= 0)
						close(_lockFileDescriptor);
					_lockFileDescriptor = -1;
					throw std::runtime_error("BuildStorage: failed to acquire storage file lock");
				}

				_lockedPathHash = StableStoragePathHash(storageRoot);
			} else if (_lockedPathHash != StableStoragePathHash(storageRoot)) {
				throw std::runtime_error("BuildStorage: nested transaction used a different storage root");
			}

			++_lockDepth;
#endif
		}

		~StorageMetadataGuard()
		{
#ifdef _WIN32
			if (_mutex) {
				ReleaseMutex(_mutex);
				CloseHandle(_mutex);
			}
#else
			if (--_lockDepth == 0) {
				flock(_lockFileDescriptor, LOCK_UN);
				close(_lockFileDescriptor);
				_lockFileDescriptor = -1;
				_lockedPathHash = 0;
			}
#endif
		}

		StorageMetadataGuard(const StorageMetadataGuard&) = delete;
		StorageMetadataGuard& operator=(const StorageMetadataGuard&) = delete;

	private:
		std::unique_lock<std::recursive_mutex> _processLock;
#ifdef _WIN32
		HANDLE _mutex = nullptr;
#else
		static thread_local int _lockDepth;
		static thread_local int _lockFileDescriptor;
		static thread_local uint64_t _lockedPathHash;
#endif
	};

#ifndef _WIN32
	thread_local int StorageMetadataGuard::_lockDepth = 0;
	thread_local int StorageMetadataGuard::_lockFileDescriptor = -1;
	thread_local uint64_t StorageMetadataGuard::_lockedPathHash = 0;
#endif

	fs::path MakePreferredCopy(fs::path path)
	{
		path.make_preferred();
		return path;
	}

	std::optional<RocketForgeTraining::ResolvedCheckpoint> NewestCheckpointIn(const fs::path& root, bool isAutosave)
	{
		const std::set<int64_t> timesteps = GGL::TrainingStorage::FindNumberedDirs(root);
		if (timesteps.empty())
			return std::nullopt;

		const int64_t newest = *timesteps.rbegin();
		return RocketForgeTraining::ResolvedCheckpoint{ root, MakePreferredCopy(root / std::to_string(newest)), static_cast<uint64_t>(newest), isAutosave };
	}

	uint64_t TryParseTimestepsFromPath(const std::string& pathText)
	{
		if (pathText.empty())
			return 0;

		const std::string name = fs::path(pathText).filename().string();
		if (!GGL::TrainingStorage::IsAllDigits(name))
			return 0;

		return static_cast<uint64_t>(std::stoull(name));
	}

	bool PathExists(const fs::path& path)
	{
		std::error_code error;
		return fs::exists(path, error);
	}

	bool IsDirectory(const fs::path& path)
	{
		std::error_code error;
		return fs::is_directory(path, error);
	}

	std::optional<RocketForgeTraining::TrainingSegmentMetadata> LoadSegmentMetadataFile(const fs::path& path, bool strict = false)
	{
		if (!PathExists(path))
			return std::nullopt;

		std::ifstream fIn(path);
		if (!fIn.good())
			return std::nullopt;

		json j = json::parse(fIn, nullptr, false);
		if (!j.is_object())
			return std::nullopt;

		RocketForgeTraining::TrainingSegmentMetadata segment = {};
		const std::vector<std::string> errors = GGL::JsonLoad(j, segment);
		if (strict && !errors.empty())
			return std::nullopt;

		return segment;
	}

	bool HasSegmentMetadata(const fs::path& folder)
	{
		return PathExists(folder / RocketForgeTraining::BuildStorage::SEGMENT_FILE_NAME);
	}

	bool DirectoryHasSegmentFolders(const fs::path& root)
	{
		if (!IsDirectory(root))
			return false;

		std::error_code error;
		for (const fs::directory_entry& entry : fs::directory_iterator(root, error)) {
			if (error)
				return false;
			if (!entry.is_directory(error) || error)
				continue;
			if (HasSegmentMetadata(entry.path()))
				return true;
		}

		return false;
	}

	bool IsReservedBuildFolderName(const std::string& folderName)
	{
		return folderName == RocketForgeTraining::BuildStorage::SEGMENTS_FOLDER_NAME ||
			folderName == RocketForgeTraining::BuildStorage::RECENT_POLICY_VERSIONS_FOLDER_NAME ||
			folderName == RocketForgeTraining::BuildStorage::SKILL_BENCHMARKS_FOLDER_NAME ||
			folderName == GGL::TrainingStorage::IMPORTED_MODELS_FOLDER_NAME ||
			folderName == "Static Models";
	}

	bool NormalizeDirectSegmentFolders(const fs::path& buildRoot)
	{
		if (!IsDirectory(buildRoot))
			return false;

		// NOTE: Manual copies often drop a segment directly under the build root.
		// Normalize to the standard segments/ layout so existing path helpers stay valid.
		bool changed = false;
		const fs::path segmentsRoot = buildRoot / RocketForgeTraining::BuildStorage::SEGMENTS_FOLDER_NAME;
		std::error_code error;
		for (const fs::directory_entry& entry : fs::directory_iterator(buildRoot, error)) {
			if (error)
				return changed;
			if (!entry.is_directory(error) || error)
				continue;

			const std::string folderName = entry.path().filename().string();
			if (IsReservedBuildFolderName(folderName) || !HasSegmentMetadata(entry.path()))
				continue;

			const fs::path target = segmentsRoot / folderName;
			if (PathExists(target))
				continue;

			fs::create_directories(segmentsRoot, error);
			if (error)
				continue;

			error.clear();
			fs::rename(entry.path(), target, error);
			if (!error)
				changed = true;
		}

		return changed;
	}

	bool BuildRootHasSegments(const fs::path& buildRoot)
	{
		return DirectoryHasSegmentFolders(buildRoot / RocketForgeTraining::BuildStorage::SEGMENTS_FOLDER_NAME) ||
			DirectoryHasSegmentFolders(buildRoot);
	}

	bool CheckpointsEqual(
		const std::vector<RocketForgeTraining::SegmentCheckpoint>& a,
		const std::vector<RocketForgeTraining::SegmentCheckpoint>& b)
	{
		if (a.size() != b.size())
			return false;

		for (size_t i = 0; i < a.size(); i++) {
			if (a[i].timesteps != b[i].timesteps || a[i].path != b[i].path)
				return false;
		}

		return true;
	}

	std::vector<RocketForgeTraining::SegmentCheckpoint> ScanSegmentCheckpoints(
		const RocketForgeTraining::BuildStorage& storage,
		const std::string& segmentFolderName)
	{
		std::vector<RocketForgeTraining::SegmentCheckpoint> checkpoints = {};
		const fs::path checkpointRoot = storage.SegmentCheckpointsRoot(segmentFolderName);
		const std::set<int64_t> checkpointTimesteps = GGL::TrainingStorage::FindNumberedDirs(checkpointRoot);

		for (int64_t timesteps : checkpointTimesteps) {
			RocketForgeTraining::SegmentCheckpoint checkpoint = {};
			checkpoint.timesteps = static_cast<uint64_t>(timesteps);
			checkpoint.path = storage.MakeRelativeToBuild(checkpointRoot / std::to_string(timesteps));
			checkpoints.push_back(checkpoint);
		}

		return checkpoints;
	}

	std::string InferSegmentIdFromFolderName(const std::string& folderName)
	{
		if (RocketForgeTraining::BuildStorage::TryParseSegmentRevision(folderName).has_value() ||
			RocketForgeTraining::BuildStorage::TryParseSegmentId(folderName).has_value())
			return folderName;

		const size_t separator = folderName.find('_');
		if (separator == std::string::npos)
			return {};

		const std::string prefix = folderName.substr(0, separator);
		if (RocketForgeTraining::BuildStorage::TryParseSegmentRevision(prefix).has_value() ||
			RocketForgeTraining::BuildStorage::TryParseSegmentId(prefix).has_value())
			return prefix;

		return {};
	}

	bool ReconcileSegmentCheckpoints(
		const RocketForgeTraining::BuildStorage& storage,
		RocketForgeTraining::TrainingSegmentMetadata& segment)
	{
		bool changed = false;

		const std::vector<RocketForgeTraining::SegmentCheckpoint> checkpoints =
			ScanSegmentCheckpoints(storage, segment.folderName);
		if (!CheckpointsEqual(segment.checkpoints, checkpoints)) {
			segment.checkpoints = checkpoints;
			changed = true;
		}

		if (!segment.checkpoints.empty()) {
			if (segment.startTimestep == 0 && !segment.sourceCheckpointPath.empty()) {
				segment.startTimestep = TryParseTimestepsFromPath(segment.sourceCheckpointPath);
				changed = true;
			}

			const uint64_t latestTimesteps = segment.checkpoints.back().timesteps;
			if (segment.endTimestep != latestTimesteps) {
				segment.endTimestep = latestTimesteps;
				changed = true;
			}
		} else if (segment.endTimestep != 0) {
			segment.endTimestep = 0;
			changed = true;
		}

		return changed;
	}

	bool RepairSegmentMetadata(
		const RocketForgeTraining::BuildStorage& storage,
		RocketForgeTraining::TrainingSegmentMetadata& segment,
		const std::string& folderName)
	{
		bool changed = false;
		const std::string timestamp = RocketForgeTraining::BuildStorage::MakeTimestamp();

		if (segment.segmentId.empty()) {
			segment.segmentId = InferSegmentIdFromFolderName(folderName);
			if (segment.segmentId.empty())
				segment.segmentId = folderName;
			changed = true;
		}

		if (segment.folderName != folderName) {
			segment.folderName = folderName;
			changed = true;
		}

		const RocketForgeTraining::BuildMetadata build = storage.LoadBuild();
		if (segment.buildName != build.buildName) {
			segment.buildName = build.buildName;
			changed = true;
		}

		if (segment.branchName.empty()) {
			segment.branchName = "main";
			changed = true;
		}

		if (segment.status.empty()) {
			segment.status = "active";
			changed = true;
		}

		if (segment.createdAt.empty()) {
			segment.createdAt = timestamp;
			changed = true;
		}

		// Segment metadata is synced to machines that hold only some segments' checkpoints,
		// so an absent checkpoints folder means "not synced here", not "no checkpoints".
		if (IsDirectory(storage.SegmentCheckpointsRoot(segment.folderName)))
			changed |= ReconcileSegmentCheckpoints(storage, segment);

		if (segment.updatedAt.empty()) {
			segment.updatedAt = timestamp;
			changed = true;
		} else if (changed) {
			segment.updatedAt = timestamp;
		}

		return changed;
	}

	void RefreshBuildMetadataFromSegments(
		const RocketForgeTraining::BuildStorage& storage,
		const std::vector<RocketForgeTraining::TrainingSegmentMetadata>& segments)
	{
		if (!PathExists(storage.BuildMetadataPath()))
			return;

		RocketForgeTraining::BuildMetadata build = storage.LoadBuild();
		bool changed = false;
		const RocketForgeTraining::TrainingSegmentMetadata* activeSegment = nullptr;
		for (const RocketForgeTraining::TrainingSegmentMetadata& segment : segments) {
			if (segment.segmentId == build.activeSegmentId) {
				activeSegment = &segment;
				break;
			}
		}

		if (!activeSegment && !segments.empty()) {
			for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
				if (it->status != "inactive") {
					activeSegment = &(*it);
					break;
				}
			}
			if (!activeSegment)
				activeSegment = &segments.back();
		}

		const std::string activeSegmentId = activeSegment ? activeSegment->segmentId : std::string();
		const uint64_t activeTimesteps = activeSegment ? activeSegment->endTimestep : 0;
		uint64_t totalTrainedTimesteps = 0;
		for (const RocketForgeTraining::TrainingSegmentMetadata& segment : segments) {
			if (segment.endTimestep >= segment.startTimestep)
				totalTrainedTimesteps += segment.endTimestep - segment.startTimestep;
		}

		if (build.activeSegmentId != activeSegmentId) {
			build.activeSegmentId = activeSegmentId;
			changed = true;
		}
		if (build.activeTimesteps != activeTimesteps) {
			build.activeTimesteps = activeTimesteps;
			changed = true;
		}
		if (build.totalTrainedTimesteps != totalTrainedTimesteps) {
			build.totalTrainedTimesteps = totalTrainedTimesteps;
			changed = true;
		}

		if (changed) {
			build.updatedAt = RocketForgeTraining::BuildStorage::MakeTimestamp();
			storage.SaveBuild(build);
		}
	}

	bool BuildSummaryEquals(
		const RocketForgeTraining::BuildSummary& a,
		const RocketForgeTraining::BuildSummary& b)
	{
		return a.buildName == b.buildName &&
			a.folderName == b.folderName &&
			a.createdAt == b.createdAt &&
			a.updatedAt == b.updatedAt &&
			a.active == b.active;
	}

	RocketForgeTraining::BuildSummary MakeBuildSummary(
		const RocketForgeTraining::BuildMetadata& metadata,
		bool active)
	{
		RocketForgeTraining::BuildSummary summary = {};
		summary.buildName = metadata.buildName;
		summary.folderName = metadata.folderName;
		summary.createdAt = metadata.createdAt;
		summary.updatedAt = metadata.updatedAt;
		summary.active = active;
		return summary;
	}

	bool WriteJsonFile(const fs::path& path, const json& j)
	{
		fs::create_directories(path.parent_path());
		std::ofstream fOut(path);
		if (!fOut.good())
			return false;
		fOut << j.dump(4);
		return true;
	}

#ifdef _WIN32
	/** Errors from replacing a file that another handle holds open without FILE_SHARE_DELETE. **/
	bool IsFileInUseError(DWORD error)
	{
		return error == ERROR_ACCESS_DENIED || error == ERROR_SHARING_VIOLATION || error == ERROR_LOCK_VIOLATION;
	}
#endif

	/** Writes through a temp file and a rename, so readers never see a partial file. Returns the OS error on failure. **/
	std::error_code WriteJsonFileAtomically(const fs::path& path, const json& j)
	{
		static std::atomic_uint64_t nextTempId = 1;
		fs::create_directories(path.parent_path());

		fs::path tempPath = path;
#ifdef _WIN32
		const uint64_t processId = GetCurrentProcessId();
#else
		const uint64_t processId = (uint64_t)getpid();
#endif
		tempPath += ".tmp." + std::to_string(processId) + "." + std::to_string(nextTempId.fetch_add(1));

		{
			errno = 0;
			std::ofstream out(tempPath, std::ios::binary | std::ios::trunc);
			if (!out.good())
				return std::error_code(errno ? errno : EIO, std::generic_category());

			out << j.dump(4);
			out.flush();
			if (!out.good()) {
				const std::error_code writeError(errno ? errno : EIO, std::generic_category());
				out.close();
				std::error_code cleanupError;
				fs::remove(tempPath, cleanupError);
				return writeError;
			}
		}

#ifdef _WIN32
		// Readers outside the storage lock, such as the GUI's model catalog, antivirus scanners, or file sync,
		// hold the target open for moments at a time, and Windows refuses the replace while they do.
		constexpr int REPLACE_ATTEMPTS = 40;
		constexpr DWORD REPLACE_RETRY_DELAY_MS = 50;

		DWORD replaceError = ERROR_SUCCESS;
		for (int attempt = 0; attempt < REPLACE_ATTEMPTS; attempt++) {
			if (MoveFileExW(tempPath.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
				return {};

			replaceError = GetLastError();
			if (!IsFileInUseError(replaceError))
				break;

			Sleep(REPLACE_RETRY_DELAY_MS);
		}

		std::error_code cleanupError;
		fs::remove(tempPath, cleanupError);

		return std::error_code((int)replaceError, std::system_category());
#else
		std::error_code error;
		fs::rename(tempPath, path, error);
		if (error) {
			std::error_code cleanupError;
			fs::remove(tempPath, cleanupError);
		}

		return error;
#endif
	}

	std::string DescribeSaveFailure(const std::string& what, const fs::path& path, const std::error_code& error)
	{
		std::string message = "BuildStorage: failed to save " + what + ": " + path.string() + ": " + error.message();

#ifdef _WIN32
		if (error.category() == std::system_category() && IsFileInUseError((DWORD)error.value()))
			message += " (another program may be holding the file open, or it may be read-only)";
#endif

		return message;
	}

	void UpdateBuildLocalJsonNames(
		const fs::path& path,
		const std::string& oldBuildName,
		const std::string& newBuildName)
	{
		if (!fs::exists(path))
			return;

		std::ifstream fIn(path);
		if (!fIn.good())
			return;

		json j = json::parse(fIn, nullptr, false);
		if (!j.is_object())
			return;

		bool changed = false;
		if (j.contains("target_build") && j["target_build"].is_string()) {
			j["target_build"] = newBuildName;
			changed = true;
		}

		if (j.contains("exports") && j["exports"].is_array()) {
			for (json& item : j["exports"]) {
				if (!item.is_object())
					continue;
				if (item.value("build_name", oldBuildName) == oldBuildName) {
					item["build_name"] = newBuildName;
					changed = true;
				}
			}
		}

		if (changed)
			(void)WriteJsonFile(path, j);
	}
}

void RocketForgeTraining::SegmentCheckpoint::Serialize(GGL::JsonArchive& ar)
{
	ar.Field("timesteps", timesteps);
	ar.Field("path", path);
}

void RocketForgeTraining::TrainingSegmentMetadata::Serialize(GGL::JsonArchive& ar)
{
	ar.Field("schema_version", schemaVersion, 1);
	ar.Field("segment_id", segmentId);
	
	if (ar.IsLoading() && segmentId.empty())
		ar.Field("segment_revision", segmentId);

	// Legacy readers key off segment_revision; keep writing both until none remain.
	if (ar.IsSaving())
		(*ar.Raw())["segment_revision"] = segmentId;

	ar.Field("folder_name", folderName);
	ar.Field("build_name", buildName);
	ar.FieldNullIfEmpty("parent_segment_id", parentSegmentId);
	ar.Field("branch_name", branchName, std::string("main"));
	ar.Field("status", status, std::string("active"));
	ar.Field("stage_name", stageName);
	ar.Field("setup_hash", setupHash);
	ar.Field("created_at", createdAt);
	ar.Field("updated_at", updatedAt);
	ar.Field("last_trained_at", lastTrainedAt);
	ar.Field("start_timestep", startTimestep);
	ar.Field("end_timestep", endTimestep);
	ar.Field("total_training_seconds", totalTrainingSeconds);
	ar.FieldNullIfEmpty("source_checkpoint_path", sourceCheckpointPath);

	// Invalid optional ratings remain unknown, rather than becoming a zero baseline.
	json ratings = skillStartRatings;
	ar.Field("skill_start_ratings", ratings);
	if (ar.IsLoading()) {
		skillStartRatings.clear();
		if (ratings.is_object()) {
			for (const auto& [mode, rating] : ratings.items()) {
				if (!mode.empty() && rating.is_number() && std::isfinite(rating.get<double>()))
					skillStartRatings.emplace(mode, rating.get<double>());
			}
		}
	}

	GGL::JsonArchive wandb = ar.Child("wandb");
	wandb.Field("project", wandbProject);
	wandb.Field("group", wandbGroup);
	wandb.Field("run_name", wandbRunName);
	wandb.Field("run_id", wandbRunId);

	ar.Field("notes", notes);
	ar.Field("eligible_benchmark", eligibleBenchmark);
	ar.Field("eligible_rlbot", eligibleRLBot);
	ar.Field("pinned", pinned);
	ar.Field("checkpoints", checkpoints);

	// Old files may predate last_trained_at and empty-string branch names.
	if (ar.IsLoading()) {
		if (!ar.Contains("last_trained_at"))
			lastTrainedAt = updatedAt;
		if (branchName.empty())
			branchName = "main";
	}
}

void RocketForgeTraining::BuildMetadata::Serialize(GGL::JsonArchive& ar)
{
	ar.Field("schema_version", schemaVersion, 1);
	ar.Field("build_name", buildName);
	ar.Field("folder_name", folderName);
	ar.FieldNullIfEmpty("active_segment_id", activeSegmentId);
	ar.Field("created_at", createdAt);
	ar.Field("updated_at", updatedAt);
	ar.Field("active_timesteps", activeTimesteps);
	ar.Field("total_trained_timesteps", totalTrainedTimesteps);
	ar.Field("notes", notes);
}

void RocketForgeTraining::BuildSummary::Serialize(GGL::JsonArchive& ar)
{
	ar.Field("build_name", buildName);
	ar.Field("folder_name", folderName);
	ar.Field("created_at", createdAt);
	ar.Field("updated_at", updatedAt);
	ar.Field("active", active);
}

RocketForgeTraining::BuildStorage::BuildStorage(fs::path agentModelsFolder, std::string buildName) :
	agentModelsFolder(MakePreferredCopy(std::move(agentModelsFolder))),
	buildName(std::move(buildName)),
	buildFolderName(SanitizeFolderPart(this->buildName))
{
	if (this->buildName.empty())
		this->buildName = GGL::DEFAULT_BUILD_NAME;
	if (buildFolderName.empty())
		buildFolderName = GGL::DEFAULT_BUILD_NAME;

	const fs::path directRoot = this->agentModelsFolder / this->buildName;
	const fs::path sanitizedRoot = this->agentModelsFolder / buildFolderName;
	std::error_code error;
	if (this->buildName != buildFolderName && fs::is_directory(directRoot, error) && !fs::exists(sanitizedRoot, error))
		buildFolderName = this->buildName;
}

std::string RocketForgeTraining::BuildStorage::SanitizeFolderPart(const std::string& text)
{
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

	return result.empty() ? GGL::DEFAULT_BUILD_NAME : result;
}

std::string RocketForgeTraining::BuildStorage::MakeTimestamp()
{
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

std::string RocketForgeTraining::BuildStorage::MakeSetupHash(const RocketForge::TrainingSetupSnapshot& snapshot)
{
	json j = {
		{ "agent", snapshot.agentName },
		{ "stage", snapshot.stageName },
		{ "version_id", snapshot.versionId },
		{ "rewards", json::array() },
		{ "hyperparameters", json::array() }
	};

	// Simulator choice is part of a training setup; source revisions are provenance only.
	if (snapshot.engineVersion != 0)
		j["engine_version"] = snapshot.engineVersion;

	if (snapshot.agentSetup.has_value())
		j["agent_setup"] = RocketForge::AgentSetupSnapshotToHashJson(snapshot.agentSetup.value());

	for (const RocketForge::RewardTableEntry& reward : snapshot.rewards) {
		j["rewards"].push_back({
			{ "name", reward.name },
			{ "weight", reward.weight },
			{ "effective_max", reward.effectiveMax },
			{ "range_min", reward.range.minValue },
			{ "range_max", reward.range.maxValue },
			{ "category", static_cast<int>(reward.category) }
		});
	}

	for (const RocketForge::TrainingSetupParameter& parameter : snapshot.hyperparameters) {
		if (!RocketForge::TrainingSetupParameterAffectsIdentity(parameter))
			continue;

		j["hyperparameters"].push_back({
			{ "group", parameter.group },
			{ "name", parameter.name },
			{ "value", parameter.value }
		});
	}

	const std::string serialized = j.dump();
	uint64_t hash = 14695981039346656037ull;
	for (unsigned char c : serialized) {
		hash ^= c;
		hash *= 1099511628211ull;
	}

	std::stringstream stream;
	stream << std::hex << hash;
	return stream.str();
}

bool RocketForgeTraining::BuildStorage::SegmentSetupHashMatches(
	const TrainingSegmentMetadata& segment,
	const RocketForge::TrainingSetupSnapshot& currentSnapshot,
	const std::string& currentHash) const
{
	if (currentHash == segment.setupHash)
		return true;

	// Check if the difference is a field that doesn't matter to the training setup
	const std::optional<RocketForge::TrainingSetupSnapshot> savedSnapshot =
		RocketForge::LoadTrainingSetupSnapshot(
			SegmentRoot(segment.folderName) / RocketForge::TrainingSetupSnapshot::FILE_NAME);
	return savedSnapshot.has_value() &&
		RocketForge::DiffTrainingSetupSnapshots(savedSnapshot.value(), currentSnapshot).empty();
}

std::string RocketForgeTraining::BuildStorage::MakeSegmentId(int major, int minor)
{
	return GGL::MakeSegmentVersionId(major, minor);
}

std::optional<GGL::SegmentVersionNumber> RocketForgeTraining::BuildStorage::TryParseSegmentRevision(const std::string& segmentId)
{
	GGL::SegmentVersionNumber parsed = {};
	if (!GGL::TryParseSegmentVersionId(segmentId, parsed))
		return std::nullopt;

	return parsed;
}

std::optional<int> RocketForgeTraining::BuildStorage::TryParseSegmentId(const std::string& segmentId)
{
	if (std::optional<GGL::SegmentVersionNumber> parsed = TryParseSegmentRevision(segmentId))
		return parsed->major * 100000 + parsed->minor;

	if (segmentId.size() >= 2 && segmentId[0] == 'S') {
		for (size_t i = 1; i < segmentId.size(); i++) {
			if (!std::isdigit(static_cast<unsigned char>(segmentId[i])))
				return std::nullopt;
		}

		return std::stoi(segmentId.substr(1));
	}

	return std::nullopt;
}

fs::path RocketForgeTraining::BuildStorage::AgentModelsFolder() const
{
	return MakePreferredCopy(agentModelsFolder);
}

fs::path RocketForgeTraining::BuildStorage::BuildRegistryPath() const
{
	return MakePreferredCopy(agentModelsFolder / BUILD_REGISTRY_FILE_NAME);
}

fs::path RocketForgeTraining::BuildStorage::BuildRoot() const
{
	return MakePreferredCopy(agentModelsFolder / buildFolderName);
}

fs::path RocketForgeTraining::BuildStorage::BuildMetadataPath() const
{
	return MakePreferredCopy(BuildRoot() / BUILD_FILE_NAME);
}

fs::path RocketForgeTraining::BuildStorage::SegmentsRoot() const
{
	return MakePreferredCopy(BuildRoot() / SEGMENTS_FOLDER_NAME);
}

fs::path RocketForgeTraining::BuildStorage::SegmentRoot(const std::string& segmentFolderName) const
{
	return MakePreferredCopy(SegmentsRoot() / segmentFolderName);
}

fs::path RocketForgeTraining::BuildStorage::SegmentCheckpointsRoot(const std::string& segmentFolderName) const
{
	return MakePreferredCopy(SegmentRoot(segmentFolderName) / CHECKPOINTS_FOLDER_NAME);
}

fs::path RocketForgeTraining::BuildStorage::SegmentTempCheckpointsRoot(const std::string& segmentFolderName) const
{
	return MakePreferredCopy(SegmentRoot(segmentFolderName) / TEMP_CHECKPOINTS_FOLDER_NAME);
}

std::optional<RocketForgeTraining::ResolvedCheckpoint> RocketForgeTraining::BuildStorage::LatestSegmentCheckpointIn(const fs::path& segmentRoot)
{
	const bool isAutosave = true;
	const std::optional<ResolvedCheckpoint> durable = NewestCheckpointIn(MakePreferredCopy(segmentRoot / CHECKPOINTS_FOLDER_NAME), !isAutosave);
	const std::optional<ResolvedCheckpoint> autosave = NewestCheckpointIn(MakePreferredCopy(segmentRoot / TEMP_CHECKPOINTS_FOLDER_NAME), isAutosave);

	if (autosave.has_value() && (!durable.has_value() || autosave->timesteps > durable->timesteps))
		return autosave;

	return durable;
}

std::optional<RocketForgeTraining::ResolvedCheckpoint> RocketForgeTraining::BuildStorage::LatestSegmentCheckpoint(const std::string& segmentFolderName) const
{
	return LatestSegmentCheckpointIn(SegmentRoot(segmentFolderName));
}

std::optional<fs::path> RocketForgeTraining::BuildStorage::PromoteNewestAutosave(TrainingSegmentMetadata& segment) const
{
	const std::optional<ResolvedCheckpoint> latest = LatestSegmentCheckpoint(segment.folderName);
	if (!latest.has_value() || !latest->isAutosave)
		return std::nullopt;

	const fs::path destination = SegmentCheckpointsRoot(segment.folderName) / latest->folder.filename();

	// Copy under a non-numeric name and rename, so an interrupted copy is never mistaken for a checkpoint
	fs::path staging = destination;
	staging += ".promoting";
	fs::remove_all(staging);
	fs::create_directories(staging.parent_path());
	fs::copy(latest->folder, staging, fs::copy_options::recursive);
	fs::rename(staging, destination);

	RefreshSegmentCheckpoints(segment, true);
	SaveSegment(segment);

	return destination;
}

fs::path RocketForgeTraining::BuildStorage::RecentPolicyVersionsRoot() const
{
	return MakePreferredCopy(BuildRoot() / RECENT_POLICY_VERSIONS_FOLDER_NAME);
}

fs::path RocketForgeTraining::BuildStorage::RecentPolicyVersionsRegistryPath() const
{
	return MakePreferredCopy(RecentPolicyVersionsRoot() / RECENT_POLICY_VERSIONS_FILE_NAME);
}

fs::path RocketForgeTraining::BuildStorage::SkillBenchmarksRoot() const
{
	return MakePreferredCopy(BuildRoot() / SKILL_BENCHMARKS_FOLDER_NAME);
}

fs::path RocketForgeTraining::BuildStorage::SkillBenchmarksRegistryPath() const
{
	return MakePreferredCopy(SkillBenchmarksRoot() / SKILL_BENCHMARKS_FILE_NAME);
}

fs::path RocketForgeTraining::BuildStorage::RLBotExportsPath() const
{
	return MakePreferredCopy(BuildRoot() / RLBOT_EXPORTS_FILE_NAME);
}

fs::path RocketForgeTraining::BuildStorage::StaticOpponentsPath() const
{
	return MakePreferredCopy(BuildRoot() / STATIC_OPPONENTS_FILE_NAME);
}

fs::path RocketForgeTraining::BuildStorage::SkillReferencesPath() const
{
	return MakePreferredCopy(BuildRoot() / SKILL_REFERENCES_FILE_NAME);
}

fs::path RocketForgeTraining::BuildStorage::ResolveBuildPath(const std::string& relativeOrAbsolutePath) const
{
	if (relativeOrAbsolutePath.empty())
		return {};

	fs::path path(relativeOrAbsolutePath);
	if (path.is_absolute())
		return MakePreferredCopy(path);

	return MakePreferredCopy(BuildRoot() / path);
}

std::vector<RocketForgeTraining::BuildSummary> RocketForgeTraining::BuildStorage::ListBuilds(
	std::map<std::string, std::vector<TrainingSegmentMetadata>>* reconciledSegments) const
{
	StorageMetadataGuard lock(agentModelsFolder);
	if (reconciledSegments)
		reconciledSegments->clear();

	std::vector<BuildSummary> registryBuilds = {};
	const fs::path path = BuildRegistryPath();
	if (PathExists(path)) {
		std::ifstream fIn(path);
		if (fIn.good()) {
			json j = json::parse(fIn, nullptr, false);
			if (j.is_object()) {
				for (const json& item : j.value("builds", json::array())) {
					BuildSummary summary = {};
					GGL::JsonLoad(item, summary);
					registryBuilds.push_back(summary);
				}
			}
		}
	}

	std::vector<BuildSummary> result = {};
	std::set<std::string> knownFolders = {};
	bool changed = false;

	// Registry reads reconcile back to the filesystem so manual moves/deletes
	// become durable metadata updates on the next refresh.
	const auto addBuild = [&](const BuildSummary* registrySummary, const std::string& buildName) {
		BuildStorage storage(agentModelsFolder, buildName);
		const fs::path root = storage.BuildRoot();
		if (!IsDirectory(root))
			return false;

		changed = NormalizeDirectSegmentFolders(root) || changed;
		const bool hasBuildMetadata = PathExists(storage.BuildMetadataPath());
		const bool hasSegments = BuildRootHasSegments(root);
		if (!hasBuildMetadata && !hasSegments)
			return false;

		if (!hasBuildMetadata) {
			BuildMetadata build = storage.LoadBuild();
			if (registrySummary && !registrySummary->createdAt.empty())
				build.createdAt = registrySummary->createdAt;
			if (registrySummary && !registrySummary->updatedAt.empty())
				build.updatedAt = registrySummary->updatedAt;
			storage.SaveBuild(build);
			changed = true;
		}

		std::vector<TrainingSegmentMetadata> segments = storage.ListSegments();
		BuildMetadata metadata = storage.LoadBuild();
		if (metadata.buildName.empty()) {
			metadata.buildName = buildName;
			changed = true;
		}
		if (metadata.folderName.empty()) {
			metadata.folderName = storage.BuildRoot().filename().string();
			changed = true;
		}
		if (metadata.createdAt.empty()) {
			metadata.createdAt = MakeTimestamp();
			changed = true;
		}
		if (metadata.updatedAt.empty()) {
			metadata.updatedAt = metadata.createdAt;
			changed = true;
		}
		if (changed)
			storage.SaveBuild(metadata);

		const bool active = registrySummary ? registrySummary->active : false;
		const BuildSummary summary = MakeBuildSummary(metadata, active);
		if (reconciledSegments)
			(*reconciledSegments)[summary.buildName] = std::move(segments);
		if (registrySummary && !BuildSummaryEquals(*registrySummary, summary))
			changed = true;
		result.push_back(summary);
		knownFolders.insert(storage.BuildRoot().filename().string());
		return true;
	};

	for (const BuildSummary& summary : registryBuilds) {
		const std::string buildName = summary.buildName.empty() ? summary.folderName : summary.buildName;
		if (buildName.empty() || !addBuild(&summary, buildName))
			changed = true;
	}

	std::vector<BuildSummary> discovered = {};
	if (IsDirectory(agentModelsFolder)) {
		std::error_code error;
		for (const fs::directory_entry& entry : fs::directory_iterator(agentModelsFolder, error)) {
			if (error)
				break;
			if (!entry.is_directory(error) || error)
				continue;

			const std::string folderName = entry.path().filename().string();
			if (folderName.empty() || knownFolders.count(folderName) > 0 || IsReservedBuildFolderName(folderName))
				continue;

			const size_t before = result.size();
			if (addBuild(nullptr, folderName)) {
				discovered.push_back(result.back());
				result.pop_back();
				changed = true;
			}
			while (result.size() > before)
				result.pop_back();
		}
	}

	std::sort(discovered.begin(), discovered.end(), [](const BuildSummary& a, const BuildSummary& b) {
		return a.buildName < b.buildName;
	});
	result.insert(result.end(), discovered.begin(), discovered.end());

	if (!result.empty() && std::none_of(result.begin(), result.end(), [](const BuildSummary& build) {
		return build.active;
	})) {
		result.front().active = true;
		changed = true;
	}

	if (changed)
		SaveBuilds(result);

	return result;
}

void RocketForgeTraining::BuildStorage::SaveBuilds(const std::vector<BuildSummary>& builds) const
{
	StorageMetadataGuard lock(agentModelsFolder);
	json j = {
		{ "schema_version", 1 },
		{ "builds", json::array() }
	};

	for (const BuildSummary& build : builds) {
		json item;
		GGL::JsonSave(build, item);
		j["builds"].push_back(item);
	}

	if (const std::error_code error = WriteJsonFileAtomically(BuildRegistryPath(), j))
		throw std::runtime_error(DescribeSaveFailure("metadata", BuildRegistryPath(), error));
}

RocketForgeTraining::BuildMetadata RocketForgeTraining::BuildStorage::LoadBuild() const
{
	StorageMetadataGuard lock(agentModelsFolder);
	const fs::path path = BuildMetadataPath();
	if (!fs::exists(path)) {
		BuildMetadata build = {};
		build.buildName = buildName;
		build.folderName = buildFolderName;
		build.createdAt = MakeTimestamp();
		build.updatedAt = build.createdAt;
		return build;
	}

	std::ifstream fIn(path);
	if (!fIn.good())
		return {};

	json j = json::parse(fIn, nullptr, false);
	if (!j.is_object())
		return {};

	BuildMetadata build = {};
	GGL::JsonLoad(j, build);

	// build_name/folder_name default to the identity this BuildStorage was constructed
	// with, but only when the key is genuinely absent (mirrors the old j.value() fallback).
	if (!j.contains("build_name"))
		build.buildName = buildName;
	if (!j.contains("folder_name"))
		build.folderName = buildFolderName;

	return build;
}

void RocketForgeTraining::BuildStorage::SaveBuild(const BuildMetadata& build) const
{
	StorageMetadataGuard lock(agentModelsFolder);
	json j;
	GGL::JsonSave(build, j);

	if (const std::error_code error = WriteJsonFileAtomically(BuildMetadataPath(), j))
		throw std::runtime_error(DescribeSaveFailure("metadata", BuildMetadataPath(), error));
}

void RocketForgeTraining::BuildStorage::EnsureBuildExists() const
{
	StorageMetadataGuard lock(agentModelsFolder);
	fs::create_directories(BuildRoot());
	fs::create_directories(SegmentsRoot());
	fs::create_directories(RecentPolicyVersionsRoot());
	fs::create_directories(SkillBenchmarksRoot());

	BuildMetadata build = LoadBuild();
	if (build.buildName.empty()) {
		build.buildName = buildName;
		build.folderName = buildFolderName;
		build.createdAt = MakeTimestamp();
	}
	build.updatedAt = MakeTimestamp();
	SaveBuild(build);

	std::vector<BuildSummary> builds = ListBuilds();
	const auto existing = std::find_if(builds.begin(), builds.end(), [&](const BuildSummary& summary) {
		return summary.folderName == buildFolderName;
	});
	if (existing == builds.end()) {
		BuildSummary summary = {};
		summary.buildName = buildName;
		summary.folderName = buildFolderName;
		summary.createdAt = build.createdAt;
		summary.updatedAt = build.updatedAt;
		summary.active = builds.empty();
		builds.push_back(summary);
	} else {
		existing->buildName = buildName;
		existing->updatedAt = build.updatedAt;
	}
	SaveBuilds(builds);
}

bool RocketForgeTraining::BuildStorage::RenameBuild(const std::string& newBuildName, std::string* error) const
{
	StorageMetadataGuard lock(agentModelsFolder);
	const auto fail = [&](const std::string& message) {
		if (error)
			*error = message;
		return false;
	};

	if (newBuildName.empty())
		return fail("Build name cannot be empty.");

	const fs::path oldRoot = BuildRoot();
	if (!fs::exists(BuildMetadataPath()))
		return fail("Build does not exist: " + buildName);

	const std::string newFolderName = SanitizeFolderPart(newBuildName);
	BuildStorage renamed(agentModelsFolder, newBuildName);
	const fs::path newRoot = renamed.BuildRoot();

	std::vector<BuildSummary> builds = ListBuilds();
	for (const BuildSummary& summary : builds) {
		const std::string summaryName = summary.buildName.empty() ? summary.folderName : summary.buildName;
		const std::string summaryFolder = summary.folderName.empty()
			? SanitizeFolderPart(summaryName)
			: summary.folderName;
		const bool isCurrent = summaryFolder == buildFolderName || summaryName == buildName;
		if (isCurrent)
			continue;
		if (summaryName == newBuildName)
			return fail("A build named \"" + newBuildName + "\" already exists.");
		if (summaryFolder == newFolderName)
			return fail("A build folder named \"" + newFolderName + "\" already exists.");
	}

	if (newRoot != oldRoot && fs::exists(newRoot))
		return fail("A build folder named \"" + newFolderName + "\" already exists.");

	BuildMetadata build = LoadBuild();
	const std::string timestamp = MakeTimestamp();

	if (newRoot != oldRoot) {
		std::error_code renameError;
		fs::rename(oldRoot, newRoot, renameError);
		if (renameError)
			return fail("Failed to rename build folder: " + renameError.message());
	}

	build.buildName = newBuildName;
	build.folderName = newFolderName;
	build.updatedAt = timestamp;
	renamed.SaveBuild(build);

	for (TrainingSegmentMetadata segment : renamed.ListSegments()) {
		segment.buildName = newBuildName;
		segment.updatedAt = timestamp;
		renamed.SaveSegment(segment);
	}

	bool found = false;
	for (BuildSummary& summary : builds) {
		const std::string summaryName = summary.buildName.empty() ? summary.folderName : summary.buildName;
		const std::string summaryFolder = summary.folderName.empty()
			? SanitizeFolderPart(summaryName)
			: summary.folderName;
		if (summaryFolder != buildFolderName && summaryName != buildName)
			continue;

		summary.buildName = newBuildName;
		summary.folderName = newFolderName;
		summary.updatedAt = timestamp;
		found = true;
	}

	if (!found) {
		BuildSummary summary = {};
		summary.buildName = newBuildName;
		summary.folderName = newFolderName;
		summary.createdAt = build.createdAt.empty() ? timestamp : build.createdAt;
		summary.updatedAt = timestamp;
		summary.active = builds.empty();
		builds.push_back(summary);
	}

	renamed.SaveBuilds(builds);
	UpdateBuildLocalJsonNames(renamed.StaticOpponentsPath(), buildName, newBuildName);
	UpdateBuildLocalJsonNames(renamed.SkillReferencesPath(), buildName, newBuildName);
	UpdateBuildLocalJsonNames(renamed.RLBotExportsPath(), buildName, newBuildName);
	return true;
}

std::vector<RocketForgeTraining::TrainingSegmentMetadata> RocketForgeTraining::BuildStorage::ListSegments() const
{
	StorageMetadataGuard lock(agentModelsFolder);
	std::vector<TrainingSegmentMetadata> result = {};
	(void)NormalizeDirectSegmentFolders(BuildRoot());

	const fs::path root = SegmentsRoot();
	if (!IsDirectory(root)) {
		RefreshBuildMetadataFromSegments(*this, result);
		return result;
	}

	std::error_code error;
	for (const fs::directory_entry& entry : fs::directory_iterator(root, error)) {
		if (error)
			break;
		if (!entry.is_directory(error) || error)
			continue;

		const fs::path segmentPath = entry.path() / SEGMENT_FILE_NAME;
		std::optional<TrainingSegmentMetadata> loaded = LoadSegmentMetadataFile(segmentPath);
		if (!loaded.has_value())
			continue;

		TrainingSegmentMetadata segment = loaded.value();
		const std::string folderName = entry.path().filename().string();
		if (RepairSegmentMetadata(*this, segment, folderName))
			SaveSegment(segment);
		result.push_back(segment);
	}

	std::sort(result.begin(), result.end(), [](const TrainingSegmentMetadata& a, const TrainingSegmentMetadata& b) {
		const std::optional<GGL::SegmentVersionNumber> aRevision = BuildStorage::TryParseSegmentRevision(a.segmentId);
		const std::optional<GGL::SegmentVersionNumber> bRevision = BuildStorage::TryParseSegmentRevision(b.segmentId);
		if (aRevision.has_value() && bRevision.has_value()) {
			if (aRevision->major != bRevision->major)
				return aRevision->major < bRevision->major;
			if (aRevision->minor != bRevision->minor)
				return aRevision->minor < bRevision->minor;
		}
		if (aRevision.has_value() != bRevision.has_value())
			return aRevision.has_value();
		return a.segmentId < b.segmentId;
	});

	RefreshBuildMetadataFromSegments(*this, result);
	return result;
}

std::optional<RocketForgeTraining::TrainingSegmentMetadata> RocketForgeTraining::BuildStorage::LoadSegmentById(const std::string& segmentId) const
{
	for (TrainingSegmentMetadata segment : ListSegments()) {
		if (segment.segmentId == segmentId)
			return segment;
	}
	return std::nullopt;
}

std::map<std::string, double> RocketForgeTraining::BuildStorage::ResolveSegmentSkillStartRatings(
	const TrainingSegmentMetadata& segment, std::optional<double> initialMu) const
{
	if (!segment.skillStartRatings.empty())
		return segment.skillStartRatings;

	if (initialMu && std::isfinite(*initialMu) && segment.startTimestep == 0 && segment.endTimestep == 0
		&& segment.sourceCheckpointPath.empty() && segment.parentSegmentId.empty()
		&& segment.checkpoints.empty() && !LatestSegmentCheckpoint(segment.folderName))
		return { { "1v1", *initialMu }, { "2v2", *initialMu }, { "3v3", *initialMu } };

	std::vector<fs::path> candidates;
	if (!segment.sourceCheckpointPath.empty())
		candidates.push_back(ResolveBuildPath(segment.sourceCheckpointPath));

	const uint64_t start = segment.startTimestep != 0 ? segment.startTimestep : TryParseTimestepsFromPath(segment.sourceCheckpointPath);
	if (start > 0) {
		candidates.push_back(SegmentCheckpointsRoot(segment.folderName) / std::to_string(start));
		candidates.push_back(SegmentTempCheckpointsRoot(segment.folderName) / std::to_string(start));
	}

	for (const fs::path& checkpoint : candidates) {
		std::ifstream input(checkpoint / "RUNNING_STATS.json");
		if (!input)
			continue;

		const json stats = json::parse(input, nullptr, false);
		if (!stats.is_object() || !stats.contains("skill_ratings") || !stats["skill_ratings"].is_object())
			continue;

		std::map<std::string, double> ratings;
		for (const auto& [mode, estimate] : stats["skill_ratings"].items()) {
			if (!mode.empty() && estimate.is_object() && estimate.contains("mu") && estimate["mu"].is_number()) {
				const double mu = estimate["mu"].get<double>();
				if (std::isfinite(mu))
					ratings.emplace(mode, mu);
			}
		}

		if (!ratings.empty())
			return ratings;
	}

	// A later checkpoint cannot reconstruct a missing historical baseline.
	return {};
}

void RocketForgeTraining::BuildStorage::SaveSegment(const TrainingSegmentMetadata& segment) const
{
	StorageMetadataGuard lock(agentModelsFolder);
	json j;
	GGL::JsonSave(segment, j);

	const fs::path path = SegmentRoot(segment.folderName) / SEGMENT_FILE_NAME;
	if (const std::error_code error = WriteJsonFileAtomically(path, j))
		throw std::runtime_error(DescribeSaveFailure("segment metadata", path, error));
}

std::optional<RocketForgeTraining::TrainingSegmentMetadata> RocketForgeTraining::BuildStorage::LoadSegmentByFolderName(
	const std::string& segmentFolderName) const
{
	StorageMetadataGuard lock(agentModelsFolder);
	const fs::path folder(segmentFolderName);
	if (folder.empty() || folder.has_parent_path() || folder == "." || folder == "..")
		return std::nullopt;

	const bool strict = true;
	const fs::path segmentFile = SegmentRoot(segmentFolderName) / SEGMENT_FILE_NAME;
	const std::optional<TrainingSegmentMetadata> segment = LoadSegmentMetadataFile(segmentFile, strict);
	if (!segment || segment->folderName != segmentFolderName || segment->buildName != buildName)
		return std::nullopt;

	return segment;
}

bool RocketForgeTraining::BuildStorage::RecordSavedCheckpoint(
	const std::string& segmentId,
	const std::string& segmentFolderName,
	uint64_t timesteps,
	const fs::path& checkpointFolder) const
{
	StorageMetadataGuard lock(agentModelsFolder);

	std::optional<TrainingSegmentMetadata> loaded = LoadSegmentByFolderName(segmentFolderName);
	if (!loaded.has_value() || loaded->segmentId != segmentId)
		return false;

	std::error_code error;
	if (!fs::is_directory(checkpointFolder, error) || error)
		return false;

	const fs::path expectedRoot = fs::weakly_canonical(SegmentCheckpointsRoot(segmentFolderName), error);
	if (error)
		return false;

	error.clear();
	const fs::path actualFolder = fs::weakly_canonical(checkpointFolder, error);
	if (error)
		return false;

	error.clear();
	const fs::path autosaveRoot = fs::weakly_canonical(SegmentTempCheckpointsRoot(segmentFolderName), error);
	if (error)
		return false;

	const bool isAutosave = actualFolder.parent_path() == autosaveRoot;
	if (actualFolder.parent_path() != expectedRoot && !isAutosave)
		return false;
	if (actualFolder.filename() != std::to_string(timesteps))
		return false;

	TrainingSegmentMetadata segment = std::move(loaded.value());

	// Check only this segment's recorded durable checkpoints.
	// A pruned or malformed list is reconciled by the caller, never used for an incremental total.
	uint64_t previous = 0;
	for (size_t i = 0; i < segment.checkpoints.size(); ++i) {
		const SegmentCheckpoint& checkpoint = segment.checkpoints[i];
		error.clear();
		const fs::path saved = fs::weakly_canonical(ResolveBuildPath(checkpoint.path), error);
		if (error || saved.parent_path() != expectedRoot || saved.filename() != std::to_string(checkpoint.timesteps)
			|| !IsDirectory(saved) || (i && checkpoint.timesteps <= previous))
			return false;

		previous = checkpoint.timesteps;
	}

	if (segment.endTimestep != previous)
		return false;

	// Legacy source counters need the full reconciliation path.
	if (segment.startTimestep == 0 && !segment.sourceCheckpointPath.empty())
		return false;

	const uint64_t oldEndTimestep = segment.endTimestep;

	// Autosaves remain separate: they are resume candidates, not durable snapshots.
	if (!isAutosave) {
		const std::string relativePath = MakeRelativeToBuild(actualFolder);
		auto checkpoint = std::lower_bound(
			segment.checkpoints.begin(),
			segment.checkpoints.end(),
			timesteps,
			[](const SegmentCheckpoint& value, uint64_t target) { return value.timesteps < target; });
		if (checkpoint != segment.checkpoints.end() && checkpoint->timesteps == timesteps)
			checkpoint->path = relativePath;
		else
			segment.checkpoints.insert(checkpoint, { timesteps, relativePath });
	}

	segment.endTimestep = segment.checkpoints.empty() ? 0 : segment.checkpoints.back().timesteps;
	const std::string timestamp = MakeTimestamp();
	segment.updatedAt = timestamp;
	segment.lastTrainedAt = timestamp;

	BuildMetadata build = LoadBuild();
	const bool incrementalTotalsAreSafe = build.activeSegmentId == segment.segmentId && build.activeTimesteps == oldEndTimestep;
	SaveSegment(segment);

	// A process may have stopped after replacing segment.json but before replacing build.json.
	// Repair the complete Build when the two sides do not describe the same prior generation;
	// applying only the new delta would preserve stale totals.
	if (!incrementalTotalsAreSafe) {
		const std::vector<TrainingSegmentMetadata> reconciled = ListSegments();
		BuildMetadata repairedBuild = LoadBuild();
		repairedBuild.activeSegmentId = segment.segmentId;
		for (const TrainingSegmentMetadata& current : reconciled)
			if (current.segmentId == segment.segmentId)
				repairedBuild.activeTimesteps = current.endTimestep;

		repairedBuild.updatedAt = timestamp;
		SaveBuild(repairedBuild);
		return true;
	}

	build.activeSegmentId = segment.segmentId;
	build.activeTimesteps = segment.endTimestep;

	const uint64_t oldDuration = oldEndTimestep >= segment.startTimestep ? oldEndTimestep - segment.startTimestep : 0;
	const uint64_t newDuration = segment.endTimestep >= segment.startTimestep ? segment.endTimestep - segment.startTimestep : 0;
	if (newDuration >= oldDuration)
		build.totalTrainedTimesteps += newDuration - oldDuration;
	else
		build.totalTrainedTimesteps -= std::min(build.totalTrainedTimesteps, oldDuration - newDuration);

	build.updatedAt = timestamp;
	SaveBuild(build);

	return true;
}

std::string RocketForgeTraining::BuildStorage::AllocateSegmentId() const
{
	GGL::SegmentVersionNumber next = { 1, 0 };
	bool hasRevisionSegments = false;

	for (const TrainingSegmentMetadata& segment : ListSegments()) {
		const std::optional<GGL::SegmentVersionNumber> parsed = TryParseSegmentRevision(segment.segmentId);
		if (!parsed.has_value())
			continue;

		hasRevisionSegments = true;
		if (parsed->major > next.major || (parsed->major == next.major && parsed->minor >= next.minor)) {
			next.major = parsed->major;
			next.minor = parsed->minor + 1;
		}
	}

	if (!hasRevisionSegments)
		return MakeSegmentId(1, 0);
	return MakeSegmentId(next.major, next.minor);
}

std::string RocketForgeTraining::BuildStorage::MakeSegmentFolderName(
	const std::string& segmentId,
	const std::string& stageName) const
{
	std::string folderName = segmentId;
	if (!stageName.empty())
		folderName += "_" + SanitizeFolderPart(stageName);
	return folderName;
}

void RocketForgeTraining::BuildStorage::RefreshSegmentCheckpoints(TrainingSegmentMetadata& segment, bool markTrained) const
{
	segment.checkpoints = ScanSegmentCheckpoints(*this, segment.folderName);

	if (!segment.checkpoints.empty()) {
		if (segment.startTimestep == 0 && !segment.sourceCheckpointPath.empty())
			segment.startTimestep = TryParseTimestepsFromPath(segment.sourceCheckpointPath);
		segment.endTimestep = segment.checkpoints.back().timesteps;
	} else {
		segment.endTimestep = 0;
	}

	const std::string timestamp = MakeTimestamp();
	segment.updatedAt = timestamp;
	if (markTrained && !segment.checkpoints.empty())
		segment.lastTrainedAt = timestamp;
}

std::string RocketForgeTraining::BuildStorage::MakeRelativeToBuild(const fs::path& path) const
{
	std::error_code error;
	fs::path relative = fs::relative(path, BuildRoot(), error);
	if (error)
		relative = path;
	return relative.generic_string();
}
