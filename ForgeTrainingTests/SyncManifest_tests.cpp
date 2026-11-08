#include <gtest/gtest.h>

#include "TestSupport.h"

#include "StageQueue/SyncManifest.h"
#include "StageQueue/StageQueueRunner.h"
#include "StaticOpponents/StaticOpponentSelectionInternal.h"
#include "Storage/BuildStorage.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace RocketForgeTraining;

namespace {
	const std::string MODELS_REL = "Bots/test-bot/TestAgent/Models";
	const std::string BUILD_REL = MODELS_REL + "/V1";
	const std::string NEXTO_REL = "Bots/PublicBots/Nexto/Models/Imported Models/RLBotPack";

	fs::path TempManifestStorageRoot(const std::string& name)
	{
		return RocketForgeTests::TestTempDir(name);
	}

	void SaveSegment(BuildStorage& storage, const std::string& segmentId, const std::string& stageName, const std::string& parentId)
	{
		TrainingSegmentMetadata segment = {};
		segment.segmentId = segmentId;
		segment.stageName = stageName;
		segment.parentSegmentId = parentId;
		segment.folderName = storage.MakeSegmentFolderName(segmentId, stageName);
		storage.SaveSegment(segment);
	}

	SegmentInspector NullInspector()
	{
		return [](const std::string&, const std::string&, const std::string&) {
			return CheckpointInfo{};
		};
	}

	SegmentInspector InspectorFor(const std::string& segmentId, uint64_t timesteps)
	{
		return [=](const std::string&, const std::string&, const std::string& id) {
			CheckpointInfo info;
			if (id == segmentId) {
				info.segmentExists = true;
				info.hasCheckpoints = true;
				info.latestTimesteps = timesteps;
			}
			return info;
		};
	}

	bool Contains(const std::vector<std::string>& values, const std::string& value)
	{
		return std::find(values.begin(), values.end(), value) != values.end();
	}

	SyncManifestContext TestContext()
	{
		SyncManifestContext context;
		context.modelsRel = MODELS_REL;
		context.staticOpponentGlobs = { NEXTO_REL };
		return context;
	}

	StageSettings ContinueStage(const std::string& stageName, const std::string& segmentId)
	{
		StageSettings st;
		st.stageName = stageName;
		st.targetTotalTimesteps = 1000;
		st.createdSegmentId = segmentId;
		st.bootstrap.mode = BootstrapConfig::Mode::ContinueExisting;
		st.bootstrap.selectedExistingVersionId = segmentId;
		return st;
	}
}

TEST(SyncManifest, ListsFixedMetadataFiles)
{
	ForgeSettings s;
	s.stages = { ContinueStage("score", "v1.01") };

	fs::path root = TempManifestStorageRoot("metadata");
	BuildStorage storage(root, "V1");

	SyncManifest manifest = BuildSyncManifest(s, storage, NullInspector(), TestContext());

	ASSERT_TRUE(Contains(manifest.metadataPaths, "Bots/test-bot/TestAgent/forge_settings.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, MODELS_REL + "/builds.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, BUILD_REL + "/build.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, BUILD_REL + "/static_opponents.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, BUILD_REL + "/skill_references.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, BUILD_REL + "/rlbot_exports.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, BUILD_REL + "/recent_policy_versions/recent_policy_versions.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, BUILD_REL + "/skill_benchmarks/benchmarks.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, BUILD_REL + "/segments/*/segment.json"));
	ASSERT_TRUE(Contains(manifest.metadataPaths, BUILD_REL + "/segments/*/training_setup.json"));
	ASSERT_EQ(manifest.metadataPaths.size(), size_t(10));

	fs::remove_all(root);
}

TEST(SyncManifest, ContinuedSegment_IncludesSegmentAndLineagePools)
{
	ForgeSettings s;
	s.stages = { ContinueStage("score", "v1.01") };

	fs::path root = TempManifestStorageRoot("lineage");
	BuildStorage storage(root, "V1");
	SaveSegment(storage, "v1.00", "warmup", "");
	SaveSegment(storage, "v1.01", "score", "v1.00");
	SaveSegment(storage, "v1.02", "aerial", "v1.01");

	SyncManifest manifest = BuildSyncManifest(s, storage, InspectorFor("v1.01", 500), TestContext());

	// Only the continued segment syncs whole; its parent contributes pools only.
	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/segments/v1.01_*/**"));
	ASSERT_FALSE(Contains(manifest.folderGlobs, BUILD_REL + "/segments/v1.00_*/**"));
	ASSERT_FALSE(Contains(manifest.folderGlobs, BUILD_REL + "/segments/v1.02_*/**"));

	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/recent_policy_versions/v1.01_*/**"));
	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/recent_policy_versions/v1.00_*/**"));
	ASSERT_FALSE(Contains(manifest.folderGlobs, BUILD_REL + "/recent_policy_versions/v1.02_*/**"));

	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/skill_benchmarks/v1.01_*/**"));
	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/skill_benchmarks/v1.00_*/**"));
	ASSERT_FALSE(Contains(manifest.folderGlobs, BUILD_REL + "/skill_benchmarks/v1.02_*/**"));

	fs::remove_all(root);
}

TEST(SyncManifest, TempCheckpointGlobs_CoverQueueSegmentsOnly)
{
	ForgeSettings s;
	s.stages = { ContinueStage("score", "v1.01") };

	fs::path root = TempManifestStorageRoot("temp_checkpoints");
	BuildStorage storage(root, "V1");
	SaveSegment(storage, "v1.00", "warmup", "");
	SaveSegment(storage, "v1.01", "score", "v1.00");
	SaveSegment(storage, "v1.02", "aerial", "v1.01");

	SyncManifest manifest = BuildSyncManifest(s, storage, InspectorFor("v1.01", 500), TestContext());

	ASSERT_TRUE(Contains(manifest.tempCheckpointGlobs, BUILD_REL + "/segments/v1.01_*/temp_checkpoints/**"));
	ASSERT_FALSE(Contains(manifest.tempCheckpointGlobs, BUILD_REL + "/segments/v1.00_*/temp_checkpoints/**"));
	ASSERT_FALSE(Contains(manifest.tempCheckpointGlobs, BUILD_REL + "/segments/v1.02_*/temp_checkpoints/**"));
	ASSERT_EQ(manifest.tempCheckpointGlobs.size(), size_t(1));

	// The default folder set must keep excluding temp checkpoints.
	ASSERT_FALSE(Contains(manifest.folderGlobs, BUILD_REL + "/segments/v1.01_*/temp_checkpoints/**"));

	fs::remove_all(root);
}

TEST(SyncManifest, SecondStageWithoutSegment_IncludesPredictedNewId)
{
	ForgeSettings s;
	StageSettings s1;
	s1.stageName = "aerial";
	s1.targetTotalTimesteps = 1000;
	s.stages = { ContinueStage("score", "v1.01"), s1 };

	fs::path root = TempManifestStorageRoot("predicted");
	BuildStorage storage(root, "V1");
	SaveSegment(storage, "v1.00", "warmup", "");
	SaveSegment(storage, "v1.01", "score", "v1.00");

	SyncManifest manifest = BuildSyncManifest(s, storage, InspectorFor("v1.01", 500), TestContext());

	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/segments/v1.02_*/**"));
	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/recent_policy_versions/v1.02_*/**"));

	fs::remove_all(root);
}

TEST(SyncManifest, CreatedSegmentMissingLocally_StillIncludedById)
{
	ForgeSettings s;
	s.stages = { ContinueStage("score", "v1.05") };

	fs::path root = TempManifestStorageRoot("missing");
	BuildStorage storage(root, "V1");

	SyncManifest manifest = BuildSyncManifest(s, storage, NullInspector(), TestContext());

	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/segments/v1.05_*/**"));
	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/recent_policy_versions/v1.05_*/**"));
	ASSERT_TRUE(Contains(manifest.folderGlobs, BUILD_REL + "/skill_benchmarks/v1.05_*/**"));

	fs::remove_all(root);
}

TEST(SyncManifest, StaticOpponents_AppearAsFolderGlobs)
{
	ForgeSettings s;
	s.stages = { ContinueStage("score", "v1.01") };

	fs::path root = TempManifestStorageRoot("static");
	BuildStorage storage(root, "V1");

	SyncManifest manifest = BuildSyncManifest(s, storage, NullInspector(), TestContext());

	ASSERT_TRUE(Contains(manifest.folderGlobs, NEXTO_REL + "/**"));

	fs::remove_all(root);
}

TEST(SelectionSyncPaths, SkillReferencesShareStaticOpponentFolders)
{
	StaticOpponentSelectionEntry imported;
	imported.sourceAgentName = "example-agent";
	imported.sourceStageName = StaticOpponentSelectionInternal::IMPORTED_MODEL_STAGE_KEY;
	imported.sourceVersionId = "RLBotPack";

	StaticOpponentSelectionEntry referenceOnly = imported;
	referenceOnly.sourceVersionId = "Reference";

	const std::vector<std::string> staticPaths = StaticOpponentSelectionInternal::ListSelectionSyncPaths({ imported });
	const std::vector<std::string> combinedPaths =
		StaticOpponentSelectionInternal::ListSelectionSyncPaths({ imported, imported, referenceOnly });

	const std::string modelsRel = "Bots/ExampleBot/ExampleAgent/Models";
	ASSERT_EQ(staticPaths, std::vector<std::string>({ modelsRel + "/Imported Models/RLBotPack" }));
	ASSERT_EQ(combinedPaths, std::vector<std::string>({
		modelsRel + "/Imported Models/RLBotPack",
		modelsRel + "/Imported Models/Reference" }));
}

TEST(StaticOpponentSyncPath, ImportedEntry_UsesImportedModelsFolder)
{
	fs::path root = TempManifestStorageRoot("static_imported");
	BuildStorage storage(root, "V1");

	StaticOpponentSelectionEntry entry;
	entry.sourceAgentName = "nexto";
	entry.sourceStageName = StaticOpponentSelectionInternal::IMPORTED_MODEL_STAGE_KEY;
	entry.sourceVersionId = "RLBotPack";

	const std::string path = StaticOpponentSelectionInternal::MakeStaticOpponentSyncPath(entry, MODELS_REL, storage);
	ASSERT_EQ(path, MODELS_REL + "/Imported Models/RLBotPack");

	fs::remove_all(root);
}

TEST(StaticOpponentSyncPath, SegmentEntry_UsesOnDiskSegmentFolder)
{
	fs::path root = TempManifestStorageRoot("static_segment");
	BuildStorage storage(root, "Main 6M");
	SaveSegment(storage, "v1.04", "ground", "");

	StaticOpponentSelectionEntry entry;
	entry.sourceAgentName = "source-agent";
	entry.sourceBuildName = "Main 6M";
	entry.sourceStageName = "ground";
	entry.sourceVersionId = "v1.04";

	const std::string path = StaticOpponentSelectionInternal::MakeStaticOpponentSyncPath(entry, MODELS_REL, storage);
	ASSERT_EQ(path, MODELS_REL + "/Main_6M/segments/v1.04_ground");

	fs::remove_all(root);
}

TEST(StaticOpponentSyncPath, SegmentEntryMissingLocally_PredictsFolderName)
{
	fs::path root = TempManifestStorageRoot("static_segment_missing");
	BuildStorage storage(root, "V1");

	StaticOpponentSelectionEntry entry;
	entry.sourceAgentName = "source-agent";
	entry.sourceBuildName = "V1";
	entry.sourceStageName = "aerial-stage";
	entry.sourceVersionId = "v2.01";

	const std::string path = StaticOpponentSelectionInternal::MakeStaticOpponentSyncPath(entry, MODELS_REL, storage);
	ASSERT_EQ(path, MODELS_REL + "/V1/segments/" + storage.MakeSegmentFolderName("v2.01", "aerial-stage"));

	fs::remove_all(root);
}

TEST(StaticOpponentSyncPath, EmptyVersionId_ContributesNothing)
{
	fs::path root = TempManifestStorageRoot("static_empty_version");
	BuildStorage storage(root, "V1");

	StaticOpponentSelectionEntry entry;
	entry.sourceAgentName = "source-agent";
	entry.sourceStageName = "ground";

	ASSERT_TRUE(StaticOpponentSelectionInternal::MakeStaticOpponentSyncPath(entry, MODELS_REL, storage).empty());

	fs::remove_all(root);
}

TEST(StaticOpponentSyncPath, EscapesGlobMetacharactersInNames)
{
	fs::path root = TempManifestStorageRoot("escape");
	BuildStorage storage(root, "V1");

	StaticOpponentSelectionEntry entry;
	entry.sourceAgentName = "odd";
	entry.sourceStageName = StaticOpponentSelectionInternal::IMPORTED_MODEL_STAGE_KEY;
	entry.sourceVersionId = "Pack*";

	const std::string path = StaticOpponentSelectionInternal::MakeStaticOpponentSyncPath(entry, "Bots/Odd[1]/Models", storage);
	ASSERT_EQ(path, "Bots/Odd\\[1\\]/Models/Imported Models/Pack\\*");

	fs::remove_all(root);
}

TEST(StaticOpponentSyncPath, SegmentEntryWithoutStageMatchesAnyStageFolder)
{
	fs::path root = TempManifestStorageRoot("static_segment_no_stage");
	BuildStorage storage(root, "V1");

	StaticOpponentSelectionEntry entry;
	entry.sourceAgentName = "source-agent";
	entry.sourceBuildName = "V1";
	entry.sourceVersionId = "v2.01";

	const std::string path = StaticOpponentSelectionInternal::MakeStaticOpponentSyncPath(entry, MODELS_REL, storage);
	ASSERT_EQ(path, MODELS_REL + "/V1/segments/v2.01_*");

	fs::remove_all(root);
}

TEST(FormatRcloneFilter, FullMode_OrdersExcludesIncludesTerminator)
{
	SyncManifest manifest;
	manifest.metadataPaths = { "Bots/test-bot/TestAgent/forge_settings.json", BUILD_REL + "/segments/*/segment.json" };
	manifest.folderGlobs = { BUILD_REL + "/segments/v1.01_*/**", NEXTO_REL + "/**" };

	const std::string expected =
		"- temp_checkpoints/**\n"
		"- Temp Checkpoints/**\n"
		"+ /test-bot/TestAgent/forge_settings.json\n"
		"+ /test-bot/TestAgent/Models/V1/segments/*/segment.json\n"
		"+ /test-bot/TestAgent/Models/V1/segments/v1.01_*/**\n"
		"+ /PublicBots/Nexto/Models/Imported Models/RLBotPack/**\n"
		"- **\n";

	ASSERT_EQ(FormatRcloneFilter(manifest, SyncManifestScope::Full), expected);
}

TEST(FormatRcloneFilter, MetadataOnly_OmitsFolderRules)
{
	SyncManifest manifest;
	manifest.metadataPaths = { "Bots/test-bot/TestAgent/forge_settings.json" };
	manifest.folderGlobs = { BUILD_REL + "/segments/v1.01_*/**" };

	const std::string expected =
		"- temp_checkpoints/**\n"
		"- Temp Checkpoints/**\n"
		"+ /test-bot/TestAgent/forge_settings.json\n"
		"- **\n";

	ASSERT_EQ(FormatRcloneFilter(manifest, SyncManifestScope::MetadataOnly), expected);
}

TEST(FormatRcloneFilter, TempCheckpoints_ListsOnlyTempGlobs)
{
	SyncManifest manifest;
	manifest.metadataPaths = { "Bots/test-bot/TestAgent/forge_settings.json" };
	manifest.folderGlobs = { BUILD_REL + "/segments/v1.01_*/**" };
	manifest.tempCheckpointGlobs = { BUILD_REL + "/segments/v1.01_*/temp_checkpoints/**" };

	const std::string expected =
		"+ /test-bot/TestAgent/Models/V1/segments/v1.01_*/temp_checkpoints/**\n"
		"- **\n";

	ASSERT_EQ(FormatRcloneFilter(manifest, SyncManifestScope::TempCheckpoints), expected);
}

TEST(FormatRcloneFilter, PathOutsideBots_IsRootedAsIs)
{
	SyncManifest manifest;
	manifest.metadataPaths = { "Other/file.json" };

	const std::string expected =
		"- temp_checkpoints/**\n"
		"- Temp Checkpoints/**\n"
		"+ /Other/file.json\n"
		"- **\n";

	ASSERT_EQ(FormatRcloneFilter(manifest, SyncManifestScope::MetadataOnly), expected);
}

namespace {
	StageSettings NewMinorStage(const std::string& stageName)
	{
		StageSettings st;
		st.stageName = stageName;
		st.targetTotalTimesteps = 1000;
		st.bootstrap.mode = BootstrapConfig::Mode::New;
		st.versionIncrement = VersionIncrement::Minor;
		return st;
	}
}

TEST(MergeRecordedSegmentIds, AdoptsRemoteSegmentAndKeepsLocalEdits)
{
	ForgeSettings local;
	local.stages = { ContinueStage("stage-a", "v1.00"), NewMinorStage("stage-b") };
	local.stages[1].targetTotalTimesteps = 5000;
	local.stages[1].versionNotes = "local notes";

	ForgeSettings remote;
	remote.stages = { ContinueStage("stage-a", "v1.00"), ContinueStage("stage-b", "v1.01") };
	remote.stages[1].branchName = "side";

	ASSERT_EQ(MergeRecordedSegmentIds(local, remote), 1);

	const StageSettings& merged = local.stages[1];
	ASSERT_EQ(merged.createdSegmentId, "v1.01");
	ASSERT_EQ(merged.bootstrap.mode, BootstrapConfig::Mode::ContinueExisting);
	ASSERT_EQ(merged.bootstrap.selectedExistingVersionId, "v1.01");
	ASSERT_EQ(merged.versionIncrement, VersionIncrement::None);
	ASSERT_EQ(merged.branchName, "side");
	ASSERT_EQ(merged.targetTotalTimesteps, 5000u);
	ASSERT_EQ(merged.versionNotes, "local notes");
}

TEST(MergeRecordedSegmentIds, LeavesEntriesTheRemoteCannotVouchFor)
{
	struct Case {
		std::string label;
		StageSettings remoteStage;
	};

	const Case cases[] = {
		{ "different stage at that index", ContinueStage("stage-other", "v1.01") },
		{ "remote has no recorded id", NewMinorStage("stage-b") },
		{ "same id already recorded", ContinueStage("stage-b", "v1.00") },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		ForgeSettings local;
		local.stages = { ContinueStage("stage-b", "v1.00") };
		local.stages[0].versionNotes = "local notes";
		const StageSettings before = local.stages[0];

		ForgeSettings remote;
		remote.stages = { testCase.remoteStage };

		ASSERT_EQ(MergeRecordedSegmentIds(local, remote), 0);
		ASSERT_EQ(local.stages[0].createdSegmentId, before.createdSegmentId);
		ASSERT_EQ(local.stages[0].versionNotes, before.versionNotes);
	}
}

TEST(MergeRecordedSegmentIds, IgnoresRemoteStagesBeyondTheLocalQueue)
{
	ForgeSettings local;
	local.stages = { ContinueStage("stage-a", "v1.00") };

	ForgeSettings remote;
	remote.stages = { ContinueStage("stage-a", "v1.00"), ContinueStage("stage-b", "v1.01") };

	ASSERT_EQ(MergeRecordedSegmentIds(local, remote), 0);
	ASSERT_EQ(local.stages.size(), 1u);
}
