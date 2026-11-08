#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Storage/BuildStorage.h"
#include "Storage/TrainingBuildLock.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <fstream>
#include <iterator>
#include <optional>
#include <thread>
#ifdef ROCKETFORGE_TESTS_HAVE_QT
#include <QProcess>
#endif

namespace fs = std::filesystem;
using namespace RocketForgeTraining;

namespace {
	fs::path TempBuildStorageRoot(const std::string& name)
	{
		return RocketForgeTests::TestTempDir(name);
	}
}

TEST(BuildStorage, UsesBuildNameDirectlyUnderModels)
{
	fs::path root = TempBuildStorageRoot("layout");
	BuildStorage storage(root, "V2");

	ASSERT_EQ(storage.BuildRoot(), (root / "V2").make_preferred());
	ASSERT_EQ(storage.SegmentsRoot(), (root / "V2" / "segments").make_preferred());
	ASSERT_EQ(storage.SegmentTempCheckpointsRoot("v1.00_starter"), (root / "V2" / "segments" / "v1.00_starter" / "temp_checkpoints").make_preferred());
	ASSERT_EQ(storage.RecentPolicyVersionsRoot(), (root / "V2" / "recent_policy_versions").make_preferred());
	ASSERT_EQ(storage.SkillBenchmarksRoot(), (root / "V2" / "skill_benchmarks").make_preferred());
	ASSERT_EQ(storage.StaticOpponentsPath(), (root / "V2" / "static_opponents.json").make_preferred());

	fs::remove_all(root);
}

TEST(BuildStorage, SkillReferencesPathIsInBuildFolder)
{
	fs::path root = TempBuildStorageRoot("skill_references");
	BuildStorage storage(root, "V2");

	ASSERT_EQ(storage.SkillReferencesPath(), (root / "V2" / "skill_references.json").make_preferred());

	fs::remove_all(root);
}

TEST(BuildStorage, SavesSegmentMetadataWithEligibilityFlags)
{
	fs::path root = TempBuildStorageRoot("segment");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.buildName = "V1";
	segment.stageName = "starter";
	segment.branchName = "experiment";
	segment.status = "inactive";
	segment.lastTrainedAt = "20260101-010101";
	segment.eligibleBenchmark = true;
	segment.eligibleRLBot = true;
	segment.pinned = true;

	storage.SaveSegment(segment);
	std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentById("v1.00");
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->folderName, std::string("v1.00_starter"));
	ASSERT_EQ(loaded->branchName, std::string("experiment"));
	ASSERT_EQ(loaded->status, std::string("inactive"));
	ASSERT_EQ(loaded->lastTrainedAt, std::string("20260101-010101"));
	ASSERT_TRUE(loaded->eligibleBenchmark);
	ASSERT_TRUE(loaded->eligibleRLBot);
	ASSERT_TRUE(loaded->pinned);

	fs::remove_all(root);
}

TEST(BuildStorage, BlankBranchNameLoadsAsMain)
{
	fs::path root = TempBuildStorageRoot("blank_branch");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.buildName = "V1";
	segment.stageName = "starter";
	segment.branchName.clear();

	storage.SaveSegment(segment);
	std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentById("v1.00");
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->branchName, std::string("main"));

	fs::remove_all(root);
}

TEST(BuildStorage, LoadsLegacyLastTrainedAtFromUpdatedAt)
{
	fs::path root = TempBuildStorageRoot("legacy_last_trained");
	BuildStorage storage(root, "V1");
	fs::path segmentRoot = storage.SegmentRoot("v1.00_starter");
	fs::create_directories(segmentRoot);

	std::ofstream out(segmentRoot / BuildStorage::SEGMENT_FILE_NAME);
	out << R"({
		"schema_version": 1,
		"segment_id": "v1.00",
		"folder_name": "v1.00_starter",
		"build_name": "V1",
		"stage_name": "starter",
		"updated_at": "20260101-010203"
	})";
	out.close();

	std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentById("v1.00");
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->lastTrainedAt, std::string("20260101-010203"));

	fs::remove_all(root);
}

TEST(BuildStorage, LoadsBlankLastTrainedAtAsBlank)
{
	fs::path root = TempBuildStorageRoot("blank_last_trained");
	BuildStorage storage(root, "V1");
	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.updatedAt = "20260101-010203";
	segment.lastTrainedAt.clear();

	storage.SaveSegment(segment);
	std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentById("v1.00");
	ASSERT_TRUE(loaded.has_value());
	ASSERT_TRUE(loaded->lastTrainedAt.empty());

	fs::remove_all(root);
}

TEST(BuildStorage, RefreshCheckpointsKeepsSourceStartTimestep)
{
	fs::path root = TempBuildStorageRoot("checkpoints");
	BuildStorage storage(root, "V1");
	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.sourceCheckpointPath = "segments/v0.99_starter/checkpoints/1000";
	fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / "1500");
	fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / "2000");

	storage.RefreshSegmentCheckpoints(segment);
	ASSERT_EQ(segment.startTimestep, uint64_t(1000));
	ASSERT_EQ(segment.endTimestep, uint64_t(2000));
	ASSERT_EQ(segment.checkpoints.size(), size_t(2));

	fs::remove_all(root);
}

TEST(BuildStorage, RefreshCheckpointsMarksLastTrainedOnlyWhenTraining)
{
	fs::path root = TempBuildStorageRoot("last_trained_refresh");
	BuildStorage storage(root, "V1");
	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.lastTrainedAt = "20260101-000000";
	fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / "1000");

	storage.RefreshSegmentCheckpoints(segment);
	ASSERT_EQ(segment.lastTrainedAt, std::string("20260101-000000"));

	storage.RefreshSegmentCheckpoints(segment, true);
	ASSERT_FALSE(segment.lastTrainedAt.empty());
	ASSERT_EQ(segment.lastTrainedAt, segment.updatedAt);

	fs::remove_all(root);
}

TEST(BuildStorage, ListSegmentsKeepsCheckpointListWhenCheckpointsFolderAbsent)
{
	fs::path root = TempBuildStorageRoot("absent_checkpoints_folder");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.buildName = "V1";
	segment.stageName = "starter";
	segment.endTimestep = 200;
	segment.checkpoints = {
		{ 100, "segments/v1.00_starter/checkpoints/100" },
		{ 200, "segments/v1.00_starter/checkpoints/200" },
	};
	storage.SaveSegment(segment);
	ASSERT_FALSE(fs::exists(storage.SegmentCheckpointsRoot(segment.folderName)));

	std::vector<TrainingSegmentMetadata> segments = storage.ListSegments();
	ASSERT_EQ(segments.size(), size_t(1));

	std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentById("v1.00");
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->endTimestep, uint64_t(200));
	ASSERT_EQ(loaded->checkpoints.size(), size_t(2));
	ASSERT_EQ(loaded->checkpoints.back().timesteps, uint64_t(200));
	ASSERT_EQ(loaded->checkpoints.back().path, std::string("segments/v1.00_starter/checkpoints/200"));

	BuildMetadata build = storage.LoadBuild();
	ASSERT_EQ(build.activeSegmentId, std::string("v1.00"));
	ASSERT_EQ(build.activeTimesteps, uint64_t(200));
	ASSERT_EQ(build.totalTrainedTimesteps, uint64_t(200));

	fs::remove_all(root);
}

TEST(BuildStorage, ListSegmentsClearsCheckpointListWhenCheckpointsFolderEmpty)
{
	fs::path root = TempBuildStorageRoot("empty_checkpoints_folder");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.buildName = "V1";
	segment.stageName = "starter";
	segment.endTimestep = 200;
	segment.checkpoints = {
		{ 100, "segments/v1.00_starter/checkpoints/100" },
		{ 200, "segments/v1.00_starter/checkpoints/200" },
	};
	storage.SaveSegment(segment);
	fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName));

	std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentById("v1.00");
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->endTimestep, uint64_t(0));
	ASSERT_TRUE(loaded->checkpoints.empty());

	fs::remove_all(root);
}

TEST(BuildStorage, RenameBuildMovesFolderAndUpdatesMetadata)
{
	fs::path root = TempBuildStorageRoot("rename");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.buildName = "V1";
	segment.stageName = "starter";
	segment.lastTrainedAt = "20260101-010101";
	storage.SaveSegment(segment);

	std::string error;
	ASSERT_TRUE(storage.RenameBuild("Renamed Build", &error));

	BuildStorage renamed(root, "Renamed Build");
	ASSERT_FALSE(fs::exists(storage.BuildMetadataPath()));
	ASSERT_TRUE(fs::exists(renamed.BuildMetadataPath()));

	BuildMetadata metadata = renamed.LoadBuild();
	ASSERT_EQ(metadata.buildName, std::string("Renamed Build"));
	ASSERT_EQ(metadata.folderName, std::string("Renamed_Build"));

	std::optional<TrainingSegmentMetadata> loaded = renamed.LoadSegmentById("v1.00");
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->buildName, std::string("Renamed Build"));
	ASSERT_EQ(loaded->lastTrainedAt, std::string("20260101-010101"));

	std::vector<BuildSummary> builds = renamed.ListBuilds();
	ASSERT_EQ(builds.size(), size_t(1));
	ASSERT_EQ(builds.front().buildName, std::string("Renamed Build"));
	ASSERT_EQ(builds.front().folderName, std::string("Renamed_Build"));

	fs::remove_all(root);
}

TEST(BuildStorage, ListBuildsDiscoversDirectSegmentBuildAndRepairsMetadata)
{
	fs::path root = TempBuildStorageRoot("discover_direct_segment");
	fs::path segmentRoot = root / "Quax" / "v1.04_late_ground_stage";
	fs::create_directories(segmentRoot / "checkpoints" / "100");
	fs::create_directories(segmentRoot / "checkpoints" / "200");

	std::ofstream out(segmentRoot / BuildStorage::SEGMENT_FILE_NAME);
	out << R"({
		"schema_version": 1,
		"segment_id": "v1.04",
		"folder_name": "v1.04_late_ground_stage",
		"build_name": "Main6M",
		"stage_name": "late-ground-stage",
		"updated_at": "20260101-010101",
		"end_timestep": 100,
		"checkpoints": [
			{
				"timesteps": 100,
				"path": "segments/v1.04_late_ground_stage/checkpoints/100"
			}
		]
	})";
	out.close();

	BuildStorage probe(root, "V1");
	std::vector<BuildSummary> builds = probe.ListBuilds();
	auto quax = std::find_if(builds.begin(), builds.end(), [](const BuildSummary& build) {
		return build.buildName == "Quax";
	});
	ASSERT_TRUE(quax != builds.end());
	ASSERT_EQ(quax->folderName, std::string("Quax"));
	ASSERT_TRUE(quax->active);
	ASSERT_FALSE(fs::exists(segmentRoot));

	BuildStorage storage(root, "Quax");
	ASSERT_TRUE(fs::exists(storage.BuildMetadataPath()));
	ASSERT_TRUE(fs::exists(storage.SegmentRoot("v1.04_late_ground_stage")));

	BuildMetadata build = storage.LoadBuild();
	ASSERT_EQ(build.buildName, std::string("Quax"));
	ASSERT_EQ(build.activeSegmentId, std::string("v1.04"));
	ASSERT_EQ(build.activeTimesteps, uint64_t(200));

	std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentById("v1.04");
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->buildName, std::string("Quax"));
	ASSERT_EQ(loaded->lastTrainedAt, std::string("20260101-010101"));
	ASSERT_EQ(loaded->endTimestep, uint64_t(200));
	ASSERT_EQ(loaded->checkpoints.size(), size_t(2));
	ASSERT_EQ(loaded->checkpoints.back().path, std::string("segments/v1.04_late_ground_stage/checkpoints/200"));

	fs::remove_all(root);
}

TEST(BuildStorage, ListBuildsPrunesDeletedBuildRegistryEntries)
{
	fs::path root = TempBuildStorageRoot("prune_deleted_build");
	BuildStorage keep(root, "Keep");
	keep.EnsureBuildExists();
	BuildStorage gone(root, "Gone");
	gone.EnsureBuildExists();
	fs::remove_all(gone.BuildRoot());

	BuildStorage probe(root, "V1");
	std::vector<BuildSummary> builds = probe.ListBuilds();
	ASSERT_EQ(builds.size(), size_t(1));
	ASSERT_EQ(builds.front().buildName, std::string("Keep"));

	std::ifstream fIn(probe.BuildRegistryPath());
	std::string registry((std::istreambuf_iterator<char>(fIn)), std::istreambuf_iterator<char>());
	fIn.close();
	ASSERT_TRUE(registry.find("Keep") != std::string::npos);
	ASSERT_TRUE(registry.find("Gone") == std::string::npos);

	fs::remove_all(root);
}

TEST(BuildStorage, SaveWaitsOutBriefReaderOfRegistry)
{
	fs::path root = TempBuildStorageRoot("save_brief_reader");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	// A reader outside the storage lock, like the GUI's model catalog, holds the file open briefly.
	// On Windows an open handle without FILE_SHARE_DELETE blocks replacing the file.
	std::promise<void> readerOpened;
	std::thread reader([&]() {
		std::ifstream fIn(storage.BuildRegistryPath());
		readerOpened.set_value();
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
	});
	readerOpened.get_future().wait();

	BuildSummary renamed = {};
	renamed.buildName = "Renamed";
	renamed.folderName = "V1";
	renamed.active = true;
	EXPECT_NO_THROW(storage.SaveBuilds({ renamed }));
	reader.join();

	std::ifstream fIn(storage.BuildRegistryPath());
	std::string registry((std::istreambuf_iterator<char>(fIn)), std::istreambuf_iterator<char>());
	fIn.close();
	EXPECT_TRUE(registry.find("Renamed") != std::string::npos);

	fs::remove_all(root);
}

#ifdef _WIN32
TEST(BuildStorage, SaveFailureNamesTheOsReason)
{
	fs::path root = TempBuildStorageRoot("save_failure_reason");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	std::string message;
	{
		std::ifstream heldOpen(storage.BuildRegistryPath());
		try {
			storage.SaveBuilds({});
		} catch (const std::runtime_error& error) {
			message = error.what();
		}
	}

	EXPECT_NE(message.find("failed to save metadata"), std::string::npos) << message;
	EXPECT_NE(message.find(storage.BuildRegistryPath().string()), std::string::npos) << message;
	EXPECT_NE(message.find("another program"), std::string::npos) << message;

	fs::remove_all(root);
}
#endif

TEST(BuildStorage, SegmentMetadataSerializeMatchesLegacyShape)
{
	using namespace RocketForgeTraining;

	TrainingSegmentMetadata segment = {};
	segment.schemaVersion = 1;
	segment.segmentId = "v2.3";
	segment.folderName = "v2.3_stage";
	segment.buildName = "V1";
	segment.parentSegmentId = "";            // must save as null
	segment.branchName = "main";
	segment.status = "active";
	segment.stageName = "ball-chase";
	segment.updatedAt = "2026-07-19T00:00:00Z";
	segment.wandbProject = "rocket";
	segment.wandbRunId = "abc";
	segment.checkpoints.push_back({ 1000, "checkpoints/1000" });

	nlohmann::json j;
	GGL::JsonSave(segment, j);

	ASSERT_EQ(j["segment_id"].get<std::string>(), "v2.3");
	ASSERT_EQ(j["segment_revision"].get<std::string>(), "v2.3");   // written under BOTH keys
	ASSERT_TRUE(j["parent_segment_id"].is_null());
	ASSERT_EQ(j["wandb"]["project"].get<std::string>(), "rocket");
	ASSERT_EQ(j["checkpoints"][0]["timesteps"].get<uint64_t>(), 1000ull);

	// last_trained_at is always written on save; erase it to exercise the load-time fallback.
	j.erase("last_trained_at");

	TrainingSegmentMetadata loaded = {};
	std::vector<std::string> errors = GGL::JsonLoad(j, loaded);
	ASSERT_TRUE(errors.empty());
	ASSERT_EQ(loaded.segmentId, "v2.3");
	ASSERT_EQ(loaded.lastTrainedAt, segment.updatedAt);   // missing last_trained_at falls back to updated_at
	ASSERT_EQ(loaded.wandbRunId, "abc");
}

TEST(BuildStorage, SegmentRevisionOnlyFileStillLoads)
{
	using namespace RocketForgeTraining;

	nlohmann::json j = { { "segment_revision", "v1.0" } };
	TrainingSegmentMetadata loaded = {};
	GGL::JsonLoad(j, loaded);
	ASSERT_EQ(loaded.segmentId, "v1.0");
}

TEST(BuildStorage, SegmentIdPresentButEmptyFallsBackToSegmentRevision)
{
	using namespace RocketForgeTraining;

	nlohmann::json j = { { "segment_id", "" }, { "segment_revision", "v1.0" } };
	TrainingSegmentMetadata loaded = {};
	GGL::JsonLoad(j, loaded);
	ASSERT_EQ(loaded.segmentId, "v1.0");
}

TEST(BuildStorage, ListSegmentsRepairsMissingActiveSegment)
{
	fs::path root = TempBuildStorageRoot("missing_active_segment");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata first = {};
	first.segmentId = "v1.00";
	first.folderName = storage.MakeSegmentFolderName(first.segmentId, "starter");
	first.buildName = "V1";
	first.stageName = "starter";
	first.startTimestep = 0;
	first.endTimestep = 100;
	storage.SaveSegment(first);

	TrainingSegmentMetadata second = {};
	second.segmentId = "v1.01";
	second.folderName = storage.MakeSegmentFolderName(second.segmentId, "starter");
	second.buildName = "V1";
	second.stageName = "starter";
	second.startTimestep = 100;
	second.endTimestep = 200;
	storage.SaveSegment(second);

	BuildMetadata build = storage.LoadBuild();
	build.activeSegmentId = "v1.01";
	build.activeTimesteps = 200;
	build.totalTrainedTimesteps = 200;
	storage.SaveBuild(build);

	fs::remove_all(storage.SegmentRoot(second.folderName));
	std::vector<TrainingSegmentMetadata> segments = storage.ListSegments();
	ASSERT_EQ(segments.size(), size_t(1));
	ASSERT_EQ(segments.front().segmentId, std::string("v1.00"));

	BuildMetadata repaired = storage.LoadBuild();
	ASSERT_EQ(repaired.activeSegmentId, std::string("v1.00"));
	ASSERT_EQ(repaired.activeTimesteps, uint64_t(100));

	fs::remove_all(root);
}

namespace {
	void MakeCheckpointFolder(const fs::path& root, uint64_t timesteps)
	{
		fs::create_directories(root / std::to_string(timesteps));
	}
}

TEST(BuildStorage, LatestSegmentCheckpoint_PicksNewestPreferringDurable)
{
	struct Case {
		const char* label;
		std::vector<uint64_t> durableTimesteps;
		std::vector<uint64_t> autosaveTimesteps;
		uint64_t expectedTimesteps;
		bool expectedAutosave;
	};

	const Case cases[] = {
		{ "newer_autosave", { 1000 }, { 500, 2000 }, 2000, true },
		{ "newer_durable", { 3000 }, { 2000 }, 3000, false },
		{ "tie_prefers_durable", { 4000 }, { 4000 }, 4000, false },
		{ "autosave_only", {}, { 700 }, 700, true },
		{ "durable_only", { 900 }, {}, 900, false },
	};

	fs::path root = TempBuildStorageRoot("latest_segment_checkpoint");
	BuildStorage storage(root, "V1");

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		const std::string folder = std::string("v1.00_") + testCase.label;
		for (uint64_t timesteps : testCase.durableTimesteps)
			MakeCheckpointFolder(storage.SegmentCheckpointsRoot(folder), timesteps);
		for (uint64_t timesteps : testCase.autosaveTimesteps)
			MakeCheckpointFolder(storage.SegmentTempCheckpointsRoot(folder), timesteps);

		std::optional<ResolvedCheckpoint> latest = storage.LatestSegmentCheckpoint(folder);
		const fs::path expectedRoot =
			testCase.expectedAutosave ? storage.SegmentTempCheckpointsRoot(folder) : storage.SegmentCheckpointsRoot(folder);

		ASSERT_TRUE(latest.has_value());
		ASSERT_EQ(latest->timesteps, testCase.expectedTimesteps);
		ASSERT_EQ(latest->isAutosave, testCase.expectedAutosave);
		ASSERT_EQ(latest->root, expectedRoot);
		ASSERT_EQ(latest->folder, (expectedRoot / std::to_string(testCase.expectedTimesteps)).make_preferred());
	}

	fs::remove_all(root);
}

TEST(BuildStorage, LatestSegmentCheckpoint_NoneOrNonNumeric)
{
	fs::path root = TempBuildStorageRoot("latest_none");
	BuildStorage storage(root, "V1");
	const std::string folder = "v1.00_starter";
	ASSERT_FALSE(storage.LatestSegmentCheckpoint(folder).has_value());

	fs::create_directories(storage.SegmentTempCheckpointsRoot(folder) / "not-a-checkpoint");
	fs::create_directories(storage.SegmentCheckpointsRoot(folder) / "latest");
	ASSERT_FALSE(storage.LatestSegmentCheckpoint(folder).has_value());

	fs::remove_all(root);
}

TEST(BuildStorage, LatestSegmentCheckpointIn_MatchesMemberOverload)
{
	fs::path root = TempBuildStorageRoot("latest_static");
	BuildStorage storage(root, "V1");
	const std::string folder = "v1.00_starter";
	MakeCheckpointFolder(storage.SegmentTempCheckpointsRoot(folder), 1500);

	std::optional<ResolvedCheckpoint> latest = BuildStorage::LatestSegmentCheckpointIn(storage.SegmentRoot(folder));
	ASSERT_TRUE(latest.has_value());
	ASSERT_TRUE(latest->isAutosave);
	ASSERT_EQ(latest->timesteps, uint64_t(1500));

	fs::remove_all(root);
}

TEST(BuildStorage, PromoteNewestAutosaveCopiesOnlyNewerAutosave)
{
	struct Case {
		const char* label;
		std::vector<uint64_t> durableTimesteps;
		std::vector<uint64_t> autosaveTimesteps;
		std::optional<uint64_t> expectedPromoted;
	};

	const Case cases[] = {
		{ "newer_autosave", { 1000 }, { 500, 2000 }, 2000 },
		{ "autosave_only", {}, { 700 }, 700 },
		{ "newer_durable", { 3000 }, { 2000 }, std::nullopt },
		{ "tie", { 4000 }, { 4000 }, std::nullopt },
		{ "no_autosaves", { 900 }, {}, std::nullopt },
	};

	fs::path root = TempBuildStorageRoot("promote_autosave");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		TrainingSegmentMetadata segment = {};
		segment.segmentId = "v1.00";
		segment.folderName = std::string("v1.00_") + testCase.label;
		segment.buildName = "V1";
		segment.stageName = testCase.label;
		storage.SaveSegment(segment);

		for (uint64_t timesteps : testCase.durableTimesteps)
			MakeCheckpointFolder(storage.SegmentCheckpointsRoot(segment.folderName), timesteps);
		for (uint64_t timesteps : testCase.autosaveTimesteps) {
			const fs::path autosave = storage.SegmentTempCheckpointsRoot(segment.folderName) / std::to_string(timesteps);
			fs::create_directories(autosave);
			std::ofstream(autosave / "POLICY.lt", std::ios::binary) << "weights" << timesteps;
		}

		const std::optional<fs::path> promoted = storage.PromoteNewestAutosave(segment);
		ASSERT_EQ(promoted.has_value(), testCase.expectedPromoted.has_value());
		if (!testCase.expectedPromoted)
			continue;

		const fs::path expectedFolder = storage.SegmentCheckpointsRoot(segment.folderName) / std::to_string(*testCase.expectedPromoted);
		ASSERT_EQ(promoted->lexically_normal(), expectedFolder.lexically_normal());

		std::ifstream policy(expectedFolder / "POLICY.lt", std::ios::binary);
		const std::string contents((std::istreambuf_iterator<char>(policy)), std::istreambuf_iterator<char>());
		ASSERT_EQ(contents, "weights" + std::to_string(*testCase.expectedPromoted));

		ASSERT_TRUE(fs::exists(storage.SegmentTempCheckpointsRoot(segment.folderName) / std::to_string(*testCase.expectedPromoted)));
		ASSERT_FALSE(segment.checkpoints.empty());
		ASSERT_EQ(segment.endTimestep, *testCase.expectedPromoted);

		const std::optional<TrainingSegmentMetadata> saved = storage.LoadSegmentByFolderName(segment.folderName);
		ASSERT_TRUE(saved.has_value());
		ASSERT_EQ(saved->endTimestep, *testCase.expectedPromoted);

		std::optional<ResolvedCheckpoint> latest = storage.LatestSegmentCheckpoint(segment.folderName);
		ASSERT_TRUE(latest.has_value());
		ASSERT_FALSE(latest->isAutosave);
	}

	fs::remove_all(root);
}

TEST(BuildStorage, SkillBaselineSurvivesResumeAndSourcePruning)
{
	const fs::path root = TempBuildStorageRoot("skill_baseline");
	BuildStorage storage(root, "Learning");

	TrainingSegmentMetadata segment;
	segment.segmentId = "v1.01";
	segment.folderName = "v1.01_starter";
	segment.parentSegmentId = "v1.00";
	segment.startTimestep = 100;
	segment.sourceCheckpointPath = "segments/v1.00_starter/checkpoints/100";

	const fs::path source = storage.ResolveBuildPath(segment.sourceCheckpointPath);
	fs::create_directories(source);
	std::ofstream(source / "RUNNING_STATS.json") << R"({"skill_ratings":{"1v1":{"mu":2100.25},"2v2":{"mu":2500.5}}})";
	segment.skillStartRatings = storage.ResolveSegmentSkillStartRatings(segment);
	ASSERT_EQ(segment.skillStartRatings.at("1v1"), 2100.25);

	storage.SaveSegment(segment);
	fs::remove_all(source);

	const fs::path latest = storage.SegmentTempCheckpointsRoot(segment.folderName) / "500";
	fs::create_directories(latest);
	std::ofstream(latest / "RUNNING_STATS.json") << R"({"skill_ratings":{"1v1":{"mu":2400}}})";

	const std::optional<TrainingSegmentMetadata> resumed = storage.LoadSegmentById(segment.segmentId);
	ASSERT_TRUE(resumed.has_value());

	const double initialMu = 999.0;
	const std::map<std::string, double> baseline = storage.ResolveSegmentSkillStartRatings(*resumed, initialMu);
	ASSERT_EQ(baseline.at("1v1"), 2100.25);
	ASSERT_EQ(baseline.at("2v2"), 2500.5);
	ASSERT_EQ(baseline.size(), 2u);

	fs::remove_all(root);
}

TEST(BuildStorage, SkillBaselineRecoversOnlyExactStartCheckpoint)
{
	const fs::path root = TempBuildStorageRoot("skill_recovery");
	BuildStorage storage(root, "Learning");

	TrainingSegmentMetadata segment;
	segment.folderName = "v1.01_starter";
	segment.startTimestep = 100;
	segment.sourceCheckpointPath = "pruned/100";

	const fs::path checkpoints = storage.SegmentCheckpointsRoot(segment.folderName);
	fs::create_directories(checkpoints / "200");
	std::ofstream(checkpoints / "200" / "RUNNING_STATS.json") << R"({"skill_ratings":{"1v1":{"mu":9999}}})";
	const double initialMu = 123.0;
	ASSERT_TRUE(storage.ResolveSegmentSkillStartRatings(segment, initialMu).empty());

	const fs::path exact = storage.SegmentTempCheckpointsRoot(segment.folderName) / "100";
	fs::create_directories(exact);
	std::ofstream(exact / "RUNNING_STATS.json") << R"({"skill_ratings":{"1v1":{"mu":2100.25},"bad":{"mu":"invalid"}}})";
	const std::map<std::string, double> recovered = storage.ResolveSegmentSkillStartRatings(segment);
	ASSERT_EQ(recovered.at("1v1"), 2100.25);
	ASSERT_EQ(recovered.size(), 1u);

	std::ofstream(exact / "RUNNING_STATS.json") << "broken";
	ASSERT_TRUE(storage.ResolveSegmentSkillStartRatings(segment).empty());

	fs::remove_all(root);
}

TEST(BuildStorage, InitialSkillBaselineAppliesOnlyBeforeScratchTraining)
{
	const fs::path root = TempBuildStorageRoot("skill_scratch");
	BuildStorage storage(root, "Learning");

	TrainingSegmentMetadata segment;
	segment.folderName = "v1.00_starter";
	ASSERT_TRUE(storage.ResolveSegmentSkillStartRatings(segment).empty());

	const double initialMu = 123.5;
	const std::map<std::string, double> baseline = storage.ResolveSegmentSkillStartRatings(segment, initialMu);
	ASSERT_EQ(baseline.size(), 3u);
	ASSERT_EQ(baseline.at("3v3"), initialMu);

	segment.endTimestep = 500;
	ASSERT_TRUE(storage.ResolveSegmentSkillStartRatings(segment, initialMu).empty());

	segment.endTimestep = 0;
	fs::create_directories(storage.SegmentTempCheckpointsRoot(segment.folderName) / "500");
	ASSERT_TRUE(storage.ResolveSegmentSkillStartRatings(segment, initialMu).empty());

	fs::remove_all(root);
}

TEST(BuildStorage, SkillBaselineInvalidValuesRemainUnknown)
{
	TrainingSegmentMetadata segment;
	nlohmann::json document = {{"skill_start_ratings", {{"1v1", 123.5}, {"2v2", "invalid"}, {"3v3", nullptr}}}};
	GGL::JsonLoad(document, segment);
	ASSERT_EQ(segment.skillStartRatings.size(), 1u);
	ASSERT_EQ(segment.skillStartRatings.at("1v1"), 123.5);

	nlohmann::json saved;
	GGL::JsonSave(segment, saved);
	ASSERT_EQ(saved["skill_start_ratings"].size(), 1u);

	TrainingSegmentMetadata legacy;
	GGL::JsonLoad(nlohmann::json::object(), legacy);
	ASSERT_TRUE(legacy.skillStartRatings.empty());
}

TEST(BuildStorage, RecordSavedCheckpointUpdatesOnlyKnownSegment)
{
	fs::path root = TempBuildStorageRoot("direct_checkpoint_update");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata active = {};
	active.segmentId = "v1.00";
	active.folderName = storage.MakeSegmentFolderName(active.segmentId, "starter");
	active.buildName = "V1";
	active.stageName = "starter";
	active.startTimestep = 100;
	storage.SaveSegment(active);

	TrainingSegmentMetadata historical = {};
	historical.segmentId = "v1.01";
	historical.folderName = storage.MakeSegmentFolderName(historical.segmentId, "other");
	historical.buildName = "V1";
	historical.stageName = "other";
	historical.updatedAt = "unchanged";
	storage.SaveSegment(historical);
	(void)storage.ListSegments();

	BuildMetadata initialBuild = storage.LoadBuild();
	initialBuild.activeSegmentId = active.segmentId;
	initialBuild.activeTimesteps = active.endTimestep;
	storage.SaveBuild(initialBuild);

	const fs::path historicalPath = storage.SegmentRoot(historical.folderName) / BuildStorage::SEGMENT_FILE_NAME;
	std::ifstream beforeIn(historicalPath, std::ios::binary);
	const std::string historicalBefore((std::istreambuf_iterator<char>(beforeIn)), std::istreambuf_iterator<char>());
	beforeIn.close();

	// If a save scans history, this deliberately unreconciled folder would change it.
	fs::create_directories(storage.SegmentCheckpointsRoot(historical.folderName) / "999");

	const fs::path checkpoint = storage.SegmentCheckpointsRoot(active.folderName) / "250";
	fs::create_directories(checkpoint);
	std::ofstream(checkpoint / "POLICY.lt", std::ios::binary) << "weights";
	ASSERT_TRUE(storage.RecordSavedCheckpoint(active.segmentId, active.folderName, 250, checkpoint));
	ASSERT_TRUE(storage.RecordSavedCheckpoint(active.segmentId, active.folderName, 250, checkpoint));

	const fs::path checkpoint275 = storage.SegmentCheckpointsRoot(active.folderName) / "275";
	const fs::path checkpoint300 = storage.SegmentCheckpointsRoot(active.folderName) / "300";
	fs::create_directories(checkpoint275);
	fs::create_directories(checkpoint300);

	bool recorded275 = false;
	bool recorded300 = false;
	std::thread first([&]() { recorded275 = storage.RecordSavedCheckpoint(active.segmentId, active.folderName, 275, checkpoint275); });
	std::thread second([&]() { recorded300 = storage.RecordSavedCheckpoint(active.segmentId, active.folderName, 300, checkpoint300); });
	first.join();
	second.join();
	ASSERT_TRUE(recorded275);
	ASSERT_TRUE(recorded300);

	const std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentByFolderName(active.folderName);
	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->checkpoints.size(), size_t(3));
	ASSERT_EQ(loaded->checkpoints.front().timesteps, uint64_t(250));
	ASSERT_EQ(loaded->checkpoints.back().timesteps, uint64_t(300));
	ASSERT_EQ(loaded->endTimestep, uint64_t(300));
	ASSERT_FALSE(loaded->lastTrainedAt.empty());

	const BuildMetadata build = storage.LoadBuild();
	ASSERT_EQ(build.activeSegmentId, active.segmentId);
	ASSERT_EQ(build.activeTimesteps, uint64_t(300));
	ASSERT_EQ(build.totalTrainedTimesteps, uint64_t(200));

	std::ifstream afterIn(historicalPath, std::ios::binary);
	const std::string historicalAfter((std::istreambuf_iterator<char>(afterIn)), std::istreambuf_iterator<char>());
	afterIn.close();
	ASSERT_EQ(historicalAfter, historicalBefore);

	std::map<std::string, std::vector<TrainingSegmentMetadata>> catalog;
	BuildStorage probe(root, "V1");
	const std::vector<BuildSummary> builds = probe.ListBuilds(&catalog);
	ASSERT_EQ(builds.size(), size_t(1));
	ASSERT_EQ(catalog.at("V1").size(), size_t(2));

	fs::remove_all(root);
}

TEST(BuildStorage, RecordSavedCheckpointRepairsPartialPriorTransaction)
{
	const fs::path root = TempBuildStorageRoot("partial_checkpoint_transaction");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.00";
	segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, "starter");
	segment.buildName = "V1";
	segment.stageName = "starter";
	segment.startTimestep = 100;
	storage.SaveSegment(segment);

	const fs::path checkpoint250 = storage.SegmentCheckpointsRoot(segment.folderName) / "250";
	fs::create_directories(checkpoint250);
	ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 250, checkpoint250));

	// Simulate termination between the atomic segment and Build replacements.
	BuildMetadata staleBuild = storage.LoadBuild();
	staleBuild.activeTimesteps = 0;
	staleBuild.totalTrainedTimesteps = 0;
	storage.SaveBuild(staleBuild);

	const fs::path checkpoint300 = storage.SegmentCheckpointsRoot(segment.folderName) / "300";
	fs::create_directories(checkpoint300);
	ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 300, checkpoint300));

	const BuildMetadata repaired = storage.LoadBuild();
	ASSERT_EQ(repaired.activeSegmentId, segment.segmentId);
	ASSERT_EQ(repaired.activeTimesteps, uint64_t(300));
	ASSERT_EQ(repaired.totalTrainedTimesteps, uint64_t(200));

	fs::remove_all(root);
}

TEST(BuildStorage, AutosavesRemainSeparateFromDurableMetadata)
{
	const fs::path root = TempBuildStorageRoot("direct_autosave");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment;
	segment.segmentId = "v1.00";
	segment.folderName = "v1.00_starter";
	segment.buildName = "V1";
	segment.stageName = "starter";
	segment.skillStartRatings = {{"1v1", 1234}};
	segment.notes = "keep notes";
	segment.eligibleBenchmark = true;
	storage.SaveSegment(segment);
	(void)storage.ListSegments();

	const fs::path autosave = storage.SegmentTempCheckpointsRoot(segment.folderName) / "200";
	fs::create_directories(autosave);
	ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 200, autosave));

	std::optional<TrainingSegmentMetadata> loaded = storage.LoadSegmentByFolderName(segment.folderName);
	ASSERT_TRUE(loaded.has_value());
	ASSERT_TRUE(loaded->checkpoints.empty());
	ASSERT_EQ(loaded->endTimestep, 0u);
	ASSERT_FALSE(loaded->lastTrainedAt.empty());
	ASSERT_EQ(storage.LoadBuild().totalTrainedTimesteps, 0u);

	std::optional<ResolvedCheckpoint> latest = storage.LatestSegmentCheckpoint(segment.folderName);
	ASSERT_TRUE(latest.has_value());
	ASSERT_TRUE(latest->isAutosave);
	ASSERT_EQ(latest->timesteps, 200u);

	const fs::path durable = storage.SegmentCheckpointsRoot(segment.folderName) / "200";
	fs::create_directories(durable);
	ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 200, durable));
	ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 200, durable));

	loaded = storage.LoadSegmentByFolderName(segment.folderName);
	ASSERT_EQ(loaded->checkpoints.size(), 1u);
	ASSERT_EQ(loaded->skillStartRatings.at("1v1"), 1234);
	ASSERT_EQ(loaded->notes, "keep notes");
	ASSERT_TRUE(loaded->eligibleBenchmark);
	ASSERT_EQ(storage.LoadBuild().totalTrainedTimesteps, 200u);
	ASSERT_FALSE(storage.LatestSegmentCheckpoint(segment.folderName)->isAutosave);

	fs::remove_all(autosave);
	const fs::path next = storage.SegmentTempCheckpointsRoot(segment.folderName) / "300";
	fs::create_directories(next);
	ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 300, next));
	ASSERT_EQ(storage.LoadSegmentByFolderName(segment.folderName)->checkpoints.size(), 1u);
	ASSERT_EQ(storage.LoadBuild().activeTimesteps, 200u);
	ASSERT_EQ(storage.LatestSegmentCheckpoint(segment.folderName)->timesteps, 300u);
}

TEST(BuildStorage, DirectCheckpointRejectsInvalidIdentityAndPrunedHistory)
{
	const fs::path root = TempBuildStorageRoot("direct_invalid");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment;
	segment.segmentId = "v1.00";
	segment.folderName = "v1.00_starter";
	segment.buildName = "V1";
	storage.SaveSegment(segment);
	(void)storage.ListSegments();

	const fs::path first = storage.SegmentCheckpointsRoot(segment.folderName) / "100";
	const fs::path next = storage.SegmentCheckpointsRoot(segment.folderName) / "200";
	fs::create_directories(first);
	fs::create_directories(next);
	ASSERT_FALSE(storage.RecordSavedCheckpoint("v9.99", segment.folderName, 100, first));
	ASSERT_FALSE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 200, first));
	ASSERT_FALSE(storage.LoadSegmentByFolderName("../v1.00_starter"));
	ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 100, first));

	fs::remove_all(first);
	ASSERT_FALSE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 200, next));
	ASSERT_EQ(storage.LoadSegmentByFolderName(segment.folderName)->endTimestep, 100u);

	// The caller's reconciliation discovers pruning and missing saved metadata.
	(void)storage.ListSegments();
	ASSERT_TRUE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 200, next));
	ASSERT_EQ(storage.LoadBuild().totalTrainedTimesteps, 200u);

	std::ofstream(storage.SegmentRoot(segment.folderName) / "segment.json")
		<< R"({"segment_id":"v1.00","folder_name":"v1.00_starter","build_name":"V1","end_timestep":"broken"})";
	ASSERT_FALSE(storage.LoadSegmentByFolderName(segment.folderName));
	ASSERT_FALSE(storage.RecordSavedCheckpoint(segment.segmentId, segment.folderName, 200, next));
}

TEST(BuildStorage, RegistryReturnsReconciledSegmentsForReuse)
{
	const fs::path root = TempBuildStorageRoot("reconciled_catalog");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment;
	segment.segmentId = "v1.00";
	segment.folderName = "v1.00_starter";
	segment.buildName = "V1";
	storage.SaveSegment(segment);
	fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / "100");

	std::map<std::string, std::vector<TrainingSegmentMetadata>> catalog;
	catalog["obsolete"] = {};
	const std::vector<BuildSummary> builds = storage.ListBuilds(&catalog);
	ASSERT_EQ(builds.size(), 1u);
	ASSERT_EQ(catalog.size(), 1u);
	ASSERT_EQ(catalog.at("V1").size(), 1u);
	ASSERT_EQ(catalog.at("V1").front().endTimestep, 100u);
}

TEST(BuildStorage, CrossProcessCheckpointAndRepairAreSerialized)
{
#ifdef ROCKETFORGE_TESTS_HAVE_QT
	const fs::path root = TempBuildStorageRoot("process_storage");
	BuildStorage storage(root, "V1");
	storage.EnsureBuildExists();

	TrainingSegmentMetadata segment;
	segment.segmentId = "v1.00";
	segment.folderName = "v1.00_starter";
	segment.buildName = "V1";
	segment.startTimestep = 100;
	storage.SaveSegment(segment);
	for (const int timesteps : {250, 300})
		fs::create_directories(storage.SegmentCheckpointsRoot(segment.folderName) / std::to_string(timesteps));

	const fs::path gate = root / "start.gate";
	QProcess first;
	QProcess second;
	QProcess repair;
	const auto start = [&](QProcess& process, const QString& mode, const QString& timesteps) {
		const QStringList arguments = {mode, QString::fromStdWString(root.wstring()), QString::fromStdWString(gate.wstring()), timesteps};
		process.start(QString::fromUtf8(BUILD_STORAGE_WORKER_PATH), arguments);
		ASSERT_TRUE(process.waitForStarted(10000));
	};
	start(first, "record", "250");
	start(second, "record", "300");
	start(repair, "repair", "0");
	std::ofstream(gate).put('1');

	for (QProcess* process : {&first, &second, &repair}) {
		ASSERT_TRUE(process->waitForFinished(30000));
		ASSERT_EQ(process->exitStatus(), QProcess::NormalExit) << process->readAllStandardError().toStdString();
		ASSERT_EQ(process->exitCode(), 0) << process->readAllStandardError().toStdString();
	}

	const std::optional<TrainingSegmentMetadata> saved = storage.LoadSegmentByFolderName(segment.folderName);
	ASSERT_TRUE(saved.has_value());
	ASSERT_EQ(saved->checkpoints.size(), 2u);
	ASSERT_EQ(saved->endTimestep, 300u);
	ASSERT_EQ(storage.LoadBuild().totalTrainedTimesteps, 200u);
	for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root))
		ASSERT_EQ(entry.path().filename().string().find(".tmp."), std::string::npos);
#else
	GTEST_SKIP() << "Qt process support is unavailable";
#endif
}

TEST(TrainingBuildLock, RejectsOtherWritersAndReleasesOnExit)
{
	const fs::path root = RocketForgeTests::TestTempDir("writer-lock");
	{
		TrainingBuildLock first(root);
		EXPECT_THROW(TrainingBuildLock second(root), std::runtime_error);
	}
	EXPECT_NO_THROW(TrainingBuildLock next(root));
}
