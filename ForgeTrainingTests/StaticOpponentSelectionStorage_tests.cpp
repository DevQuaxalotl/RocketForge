#include <gtest/gtest.h>

#include "TestSupport.h"

#include "StageQueue/SyncManifest.h"
#include "StaticOpponents/StaticOpponentSelectionInternal.h"

#include "AgentModelPaths.h"
#include "Storage/BuildStorage.h"

#include <GigaLearnCPP/SkillReferenceRatings.h>
#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <limits>
#include <map>

namespace fs = std::filesystem;
using namespace RocketForgeTraining;
using namespace RocketForgeTraining::StaticOpponentSelectionInternal;

namespace {
	const char* const HAND_WRITTEN_AGENT = "static-opponent-hand-written-agent";
	const char* const HAND_WRITTEN_BUILD = "V1";
	const char* const HAND_WRITTEN_STAGE = "example-ball-chase";

	fs::path TempSelectionPath(const std::string& name)
	{
		return RocketForgeTests::TestTempDir() / (name + ".json");
	}

	void WriteTextFile(const fs::path& path, const std::string& text)
	{
		fs::create_directories(path.parent_path());
		std::ofstream(path, std::ios::binary) << text;
	}

	std::string ReadTextFile(const fs::path& path)
	{
		std::ifstream fIn(path, std::ios::binary);
		return std::string(std::istreambuf_iterator<char>(fIn), std::istreambuf_iterator<char>());
	}

	/** Registers the test agent over a fresh temp Models folder holding segment v1.0 with a checkpoint at 500 and an imported model folder. */
	BuildStorage MakeHandWrittenModels(const std::string& label)
	{
		const fs::path modelsRoot = RocketForgeTests::TestTempDir(label) / "Agent" / "Models";
		fs::create_directories(modelsRoot);

		AgentRegistry::Register(HAND_WRITTEN_AGENT, [modelsRoot] {
			auto agent = std::make_unique<Agent>();
			agent->agentName = HAND_WRITTEN_AGENT;
			agent->modelPath = modelsRoot;
			return agent;
		});

		BuildStorage storage(modelsRoot, HAND_WRITTEN_BUILD);
		storage.EnsureBuildExists();

		TrainingSegmentMetadata segment;
		segment.segmentId = "v1.0";
		segment.folderName = storage.MakeSegmentFolderName(segment.segmentId, HAND_WRITTEN_STAGE);
		segment.buildName = HAND_WRITTEN_BUILD;
		segment.stageName = HAND_WRITTEN_STAGE;
		WriteTextFile(storage.SegmentCheckpointsRoot(segment.folderName) / "500" / "POLICY.lt", "policy weights");
		WriteTextFile(storage.SegmentRoot(segment.folderName) / "training_setup.json", "{}");
		storage.SaveSegment(segment);

		WriteTextFile(modelsRoot / "Imported Models" / "PackModel" / "POLICY.lt", "policy weights");

		return storage;
	}
}

TEST(StaticOpponentSelectionStorage, SaveLoadRoundTripsEntries)
{
	fs::path path = TempSelectionPath("round_trip");
	fs::remove(path);

	StaticOpponentSelectionEntry entry = {};
	entry.sourceAgentName = "example-agent";
	entry.sourceBuildName = "V2";
	entry.sourceStageName = "starter";
	entry.sourceVersionId = "v1.00";
	entry.displayName = "example-agent / V2 / starter / v1.00";
	entry.checkpointPath = "C:/models/example/checkpoints/1000";
	entry.agentSetup = RocketForge::AgentSetupSnapshot{};

	SaveSelectionToPath(path, "target-agent", "V1", { entry });
	std::vector<StaticOpponentSelectionEntry> loaded = LoadSelectionFromPath(path);

	ASSERT_EQ(loaded.size(), size_t(1));
	ASSERT_EQ(loaded[0].sourceAgentName, std::string("example-agent"));
	ASSERT_EQ(loaded[0].sourceBuildName, std::string("V2"));
	ASSERT_EQ(loaded[0].sourceStageName, std::string("starter"));
	ASSERT_EQ(loaded[0].sourceVersionId, std::string("v1.00"));
	ASSERT_EQ(loaded[0].checkpointPath.string(), entry.checkpointPath.string());

	fs::remove(path);
}

TEST(StaticOpponentSelectionStorage, MissingSourceBuildDefaultsToV1OnLoad)
{
	fs::path path = TempSelectionPath("default_build");
	fs::remove(path);

	// A hand-written file with no "source_build" key at all (older format) must still load
	// with the "V1" default that StaticOpponentSelectionEntry::Serialize's Field call applies.
	std::ofstream fOut(path);
	fOut << R"({
		"schema_version": 1,
		"target_agent": "target-agent",
		"target_build": "V1",
		"static_opponents": [
			{
				"source_agent": "legacy-agent",
				"source_stage": "starter",
				"source_version_id": "v1.00",
				"display_name": "legacy-agent / starter / v1.00",
				"checkpoint_path": "C:/models/legacy/checkpoints/500"
			}
		]
	})";
	fOut.close();

	std::vector<StaticOpponentSelectionEntry> loaded = LoadSelectionFromPath(path);

	ASSERT_EQ(loaded.size(), size_t(1));
	ASSERT_EQ(loaded[0].sourceBuildName, std::string("V1"));

	fs::remove(path);
}

TEST(StaticOpponentSelectionStorage, SavedDocumentHasExpectedTopLevelShape)
{
	fs::path path = TempSelectionPath("document_shape");
	fs::remove(path);

	StaticOpponentSelectionEntry entry = {};
	entry.sourceAgentName = "example-agent";
	entry.sourceBuildName = "V2";
	entry.sourceStageName = "starter";
	entry.sourceVersionId = "v1.00";
	entry.agentSetup = RocketForge::AgentSetupSnapshot{};

	SaveSelectionToPath(path, "target-agent", "V3", { entry });

	std::ifstream fIn(path);
	nlohmann::json j = nlohmann::json::parse(fIn);
	fIn.close();

	ASSERT_TRUE(j.contains("schema_version"));
	ASSERT_EQ(j["schema_version"].get<int>(), 2);
	ASSERT_EQ(j.value("target_agent", std::string()), std::string("target-agent"));
	ASSERT_EQ(j.value("target_build", std::string()), std::string("V3"));
	ASSERT_TRUE(j.contains("static_opponents"));
	ASSERT_TRUE(j["static_opponents"].is_array());
	ASSERT_EQ(j["static_opponents"].size(), size_t(1));
	ASSERT_EQ(j["static_opponents"][0].value("source_agent", std::string()), std::string("example-agent"));

	fs::remove(path);
}

TEST(SkillReferenceSelection, SaveLoadRoundTripsRatings)
{
	fs::path path = TempSelectionPath("skill_reference_round_trip");
	fs::remove(path);

	SkillReferenceEntry rated = {};
	rated.selection.sourceAgentName = "example-agent";
	rated.selection.sourceBuildName = "V2";
	rated.selection.sourceStageName = "starter";
	rated.selection.sourceVersionId = "v1.00";
	rated.selection.checkpointPath = "C:/models/example/checkpoints/1000";
	rated.selection.agentSetup = RocketForge::AgentSetupSnapshot{};
	rated.skillRatings = { { "1v1", 1500.0f } };

	SkillReferenceEntry pending = rated;
	pending.selection.sourceVersionId = "v1.01";
	pending.skillRatings = {};

	SaveSkillReferencesToPath(path, "target-agent", "V1", { rated, pending });
	std::vector<SkillReferenceEntry> loaded = LoadSkillReferencesFromPath(path);

	ASSERT_EQ(loaded.size(), size_t(2));
	ASSERT_EQ(loaded[0].selection.sourceAgentName, std::string("example-agent"));
	ASSERT_EQ(loaded[0].selection.sourceBuildName, std::string("V2"));
	ASSERT_EQ(loaded[0].selection.sourceStageName, std::string("starter"));
	ASSERT_EQ(loaded[0].selection.sourceVersionId, std::string("v1.00"));
	ASSERT_EQ(loaded[0].selection.checkpointPath.string(), rated.selection.checkpointPath.string());
	ASSERT_EQ(loaded[0].skillRatings.size(), size_t(1));
	ASSERT_NEAR(loaded[0].skillRatings.at("1v1"), 1500.0f, 1e-3f);

	ASSERT_EQ(loaded[1].selection.sourceVersionId, std::string("v1.01"));
	ASSERT_TRUE(loaded[1].skillRatings.empty());

	fs::remove(path);
}

TEST(SkillReferenceSelection, MergeEditsKeepsUntouchedRatingsFromDisk)
{
	SkillReferenceEntry stored = {};
	stored.selection.sourceAgentName = "example-agent";
	stored.selection.sourceBuildName = "V2";
	stored.selection.sourceStageName = "starter";
	stored.selection.sourceVersionId = "v1.00";
	stored.skillRatings = { { "1v1", 1500.0f }, { "2v2", -40.0f }, { "3v3", 900.0f } };

	SkillReferenceEntry unchecked = stored;
	unchecked.selection.sourceVersionId = "v1.01";

	SkillReferenceEdit row = {};
	row.selection = stored.selection;
	row.editedRatings = { { "1v1", 1600.0f }, { "3v3", std::nullopt } };

	SkillReferenceEdit newRow = {};
	newRow.selection = stored.selection;
	newRow.selection.sourceVersionId = "v1.02";
	newRow.editedRatings = { { "2v2", -5.0f } };

	std::vector<SkillReferenceEntry> merged = MergeSkillReferenceEdits({ stored, unchecked }, { row, newRow });

	ASSERT_EQ(merged.size(), size_t(2));
	ASSERT_EQ(merged[0].selection.sourceVersionId, std::string("v1.00"));
	ASSERT_EQ(merged[0].skillRatings.size(), size_t(2));
	ASSERT_NEAR(merged[0].skillRatings.at("1v1"), 1600.0f, 1e-3f);
	ASSERT_NEAR(merged[0].skillRatings.at("2v2"), -40.0f, 1e-3f);
	ASSERT_FALSE(merged[0].skillRatings.contains("3v3"));

	ASSERT_EQ(merged[1].selection.sourceVersionId, std::string("v1.02"));
	ASSERT_EQ(merged[1].skillRatings.size(), size_t(1));
	ASSERT_NEAR(merged[1].skillRatings.at("2v2"), -5.0f, 1e-3f);
}

TEST(SkillReferenceSelection, ResaveKeepsPinnedCheckpointAndRatings)
{
	fs::path path = TempSelectionPath("skill_reference_resave");
	fs::remove(path);

	SkillReferenceEntry stored = {};
	stored.selection.sourceAgentName = "example-agent";
	stored.selection.sourceBuildName = "V2";
	stored.selection.sourceStageName = "starter";
	stored.selection.sourceVersionId = "v1.00";
	stored.selection.checkpointPath = "C:/models/example/checkpoints/1000";
	stored.selection.agentSetup = RocketForge::CaptureAgentSetupSnapshot(Agent{});
	stored.skillRatings = { { "1v1", 1500.0f } };
	SaveSkillReferencesToPath(path, "target-agent", "V1", { stored });

	SkillReferenceEntry added = stored;
	added.selection.sourceVersionId = "v1.01";
	added.selection.checkpointPath.clear();
	added.skillRatings = {};

	const fs::path newerCheckpoint = "C:/models/example/checkpoints/9000";
	std::vector<std::string> resolvedVersionIds = {};
	const SelectionResolver resolveToNewer = [&](const std::vector<StaticOpponentSelectionEntry>& selections) {
		std::vector<StaticOpponentSelectionEntry> resolved = selections;
		for (StaticOpponentSelectionEntry& selection : resolved) {
			resolvedVersionIds.push_back(selection.sourceVersionId);
			selection.checkpointPath = newerCheckpoint;
			selection.agentSetup = RocketForge::AgentSetupSnapshot{};
		}
		return resolved;
	};

	SkillReferenceEntry reselected = stored;
	reselected.selection.checkpointPath.clear();
	reselected.selection.agentSetup.reset();
	const std::vector<SkillReferenceEntry> pinned =
		PinSkillReferencesForSave(LoadSkillReferencesFromPath(path), { reselected, added }, resolveToNewer);
	SaveSkillReferencesToPath(path, "target-agent", "V1", pinned);
	std::vector<SkillReferenceEntry> loaded = LoadSkillReferencesFromPath(path);

	ASSERT_EQ(resolvedVersionIds, std::vector<std::string>({ "v1.01" }));
	ASSERT_EQ(loaded.size(), size_t(2));
	ASSERT_EQ(loaded[0].selection.checkpointPath.string(), stored.selection.checkpointPath.string());
	ASSERT_EQ(loaded[0].skillRatings.size(), size_t(1));
	ASSERT_NEAR(loaded[0].skillRatings.at("1v1"), 1500.0f, 1e-3f);
	ASSERT_EQ(loaded[1].selection.checkpointPath.string(), newerCheckpoint.string());

	fs::remove(path);
}

TEST(SkillReferenceSelection, InvalidRatingsLoadAsPending)
{
	fs::path path = TempSelectionPath("skill_reference_invalid_ratings");
	fs::remove(path);

	std::ofstream fOut(path);
	fOut << R"({
		"schema_version": 1,
		"target_agent": "target-agent",
		"target_build": "V1",
		"skill_references": [
			{
				"source_agent": "example-agent",
				"source_build": "V1",
				"source_stage": "starter",
				"source_version_id": "v1.00",
				"checkpoint_path": "C:/models/example/checkpoints/500",
				"skill_ratings": { "1v1": "abc", "2v2": null, "3v3": 1200, "4v4": -35.5 }
			}
		]
	})";
	fOut.close();

	std::vector<SkillReferenceEntry> loaded = LoadSkillReferencesFromPath(path);

	ASSERT_EQ(loaded.size(), size_t(1));
	ASSERT_EQ(loaded[0].skillRatings.size(), size_t(2));
	ASSERT_NEAR(loaded[0].skillRatings.at("3v3"), 1200.0f, 1e-3f);
	ASSERT_NEAR(loaded[0].skillRatings.at("4v4"), -35.5f, 1e-3f);

	fs::remove(path);
}

TEST(SkillReferenceRatings, DropsNonFiniteValuesAndKeepsNegativeOnes)
{
	nlohmann::json ratings = {
		{ "1v1", -5.0 },
		{ "2v2", std::numeric_limits<double>::quiet_NaN() },
		{ "3v3", std::numeric_limits<double>::infinity() },
		{ "4v4", 0.0 },
		{ "5v5", 1e300 }
	};

	std::map<std::string, float> parsed = GGL::ReadSkillReferenceRatings(ratings, "test reference");

	ASSERT_EQ(parsed.size(), size_t(2));
	ASSERT_NEAR(parsed.at("1v1"), -5.0f, 1e-6f);
	ASSERT_NEAR(parsed.at("4v4"), 0.0f, 1e-6f);
}

TEST(SkillReferenceSelection, RelocateKeepsPinnedCheckpointUnderLocalSource)
{
	const fs::path checkpointsRoot = RocketForgeTests::TestTempDir() / "skill_reference_relocate" / "checkpoints";
	fs::create_directories(checkpointsRoot / "1000");
	fs::create_directories(checkpointsRoot / "9000");

	SkillReferenceEntry stored = {};
	stored.selection.sourceAgentName = "example-agent";
	stored.selection.sourceBuildName = "V2";
	stored.selection.sourceStageName = "starter";
	stored.selection.sourceVersionId = "v1.00";
	stored.selection.checkpointPath = R"(D:\OtherMachine\Models\V2\segments\v1.00_starter\checkpoints\1000)";
	stored.selection.agentSetup = RocketForge::CaptureAgentSetupSnapshot(Agent{});
	stored.skillRatings = { { "1v1", 1500.0f } };

	const SourceResolver resolveLocal = [&](const StaticOpponentSelectionEntry& selection) {
		SourceVersionSnapshot source = {};
		source.entry = selection;
		source.latestCheckpoint = checkpointsRoot / "9000";
		return source;
	};

	const std::vector<SkillReferenceEntry> relocated = RelocateSkillReferences({ stored }, resolveLocal);

	ASSERT_EQ(relocated.size(), size_t(1));
	ASSERT_EQ(relocated[0].selection.checkpointPath, checkpointsRoot / "1000");
	ASSERT_TRUE(relocated[0].selection.agentSetup.has_value());
	ASSERT_NEAR(relocated[0].skillRatings.at("1v1"), 1500.0f, 1e-3f);
}

TEST(SkillReferenceSelection, RelocateThrowsWhenPinnedCheckpointIsMissing)
{
	const fs::path checkpointsRoot = RocketForgeTests::TestTempDir() / "skill_reference_relocate_missing" / "checkpoints";
	fs::create_directories(checkpointsRoot / "9000");

	SkillReferenceEntry stored = {};
	stored.selection.sourceAgentName = "example-agent";
	stored.selection.sourceStageName = "starter";
	stored.selection.sourceVersionId = "v1.00";
	stored.selection.checkpointPath = "C:/models/example/checkpoints/1000";
	stored.selection.agentSetup = RocketForge::AgentSetupSnapshot{};

	const SourceResolver resolveLocal = [&](const StaticOpponentSelectionEntry& selection) {
		SourceVersionSnapshot source = {};
		source.entry = selection;
		source.latestCheckpoint = checkpointsRoot / "9000";
		return source;
	};

	ASSERT_THROW(RelocateSkillReferences({ stored }, resolveLocal), std::runtime_error);
}

TEST(SkillReferenceSelection, RelocateImportedModelUsesLocalModelFolder)
{
	SkillReferenceEntry stored = {};
	stored.selection.sourceAgentName = "Nexto";
	stored.selection.sourceStageName = IMPORTED_MODEL_STAGE_KEY;
	stored.selection.sourceVersionId = "RLBotPack";
	stored.selection.checkpointPath = R"(D:\OtherMachine\Bots\PublicBots\Nexto\Models\Imported Models\RLBotPack)";
	stored.selection.agentSetup = RocketForge::AgentSetupSnapshot{};

	const fs::path localFolder = RocketForgeTests::TestTempDir() / "Imported Models" / "RLBotPack";
	const SourceResolver resolveLocal = [&](const StaticOpponentSelectionEntry& selection) {
		SourceVersionSnapshot source = {};
		source.entry = selection;
		source.latestCheckpoint = localFolder;
		return source;
	};

	const std::vector<SkillReferenceEntry> relocated = RelocateSkillReferences({ stored }, resolveLocal);

	ASSERT_EQ(relocated.size(), size_t(1));
	ASSERT_EQ(relocated[0].selection.checkpointPath, localFolder);
}

TEST(StaticOpponentSelectionStorage, FillsMissingFieldsForHandWrittenEntries)
{
	const BuildStorage storage = MakeHandWrittenModels("static-opponent-hand-written");

	const std::string document = R"({
		"static_opponents": [
			{ "source_agent": "static-opponent-hand-written-agent", "source_build": "V1", "source_version_id": "v1.0" },
			{ "source_agent": "static-opponent-hand-written-agent", "source_build": "", "source_version_id": "PackModel" }
		]
	})";
	WriteTextFile(storage.StaticOpponentsPath(), document);

	const std::vector<StaticOpponentSelectionEntry> loaded = LoadStaticOpponentSelection(HAND_WRITTEN_AGENT, HAND_WRITTEN_BUILD);

	ASSERT_EQ(loaded.size(), size_t(2));

	EXPECT_EQ(loaded[0].checkpointPath.filename().string(), "500");
	EXPECT_EQ(loaded[0].sourceStageName, HAND_WRITTEN_STAGE);
	EXPECT_TRUE(loaded[0].agentSetup.has_value());

	EXPECT_EQ(loaded[1].sourceStageName, IMPORTED_MODEL_STAGE_KEY);
	EXPECT_EQ(loaded[1].checkpointPath.filename().string(), "PackModel");
	EXPECT_TRUE(loaded[1].agentSetup.has_value());

	EXPECT_EQ(ReadTextFile(storage.StaticOpponentsPath()), document);
}

TEST(StaticOpponentSelectionStorage, DropsUnresolvableHandWrittenEntryWithHint)
{
	const BuildStorage storage = MakeHandWrittenModels("static-opponent-unresolvable");

	WriteTextFile(storage.StaticOpponentsPath(), R"({
		"static_opponents": [
			{ "source_agent": "static-opponent-hand-written-agent", "source_build": "V1", "source_version_id": "v9.9" },
			{ "source_agent": "static-opponent-hand-written-agent", "source_build": "", "source_version_id": "NoSuchModel" }
		]
	})");

	testing::internal::CaptureStderr();
	const std::vector<StaticOpponentSelectionEntry> loaded = LoadStaticOpponentSelection(HAND_WRITTEN_AGENT, HAND_WRITTEN_BUILD);
	const std::string warnings = testing::internal::GetCapturedStderr();

	EXPECT_TRUE(loaded.empty());
	EXPECT_NE(warnings.find("v9.9"), std::string::npos) << warnings;
	EXPECT_NE(warnings.find("list segments"), std::string::npos) << warnings;
	EXPECT_NE(warnings.find("NoSuchModel"), std::string::npos) << warnings;
	EXPECT_NE(warnings.find("list imported-models"), std::string::npos) << warnings;
}

TEST(StaticOpponentSelectionStorage, SyncKeepsUnresolvableHandWrittenEntryAsStored)
{
	const BuildStorage storage = MakeHandWrittenModels("static-opponent-sync-keeps");

	const nlohmann::json unresolvable = {
		{ "source_agent", HAND_WRITTEN_AGENT },
		{ "source_build", HAND_WRITTEN_BUILD },
		{ "source_version_id", "v9.9" },
		{ "note", "kept for another machine" },
	};
	nlohmann::json document = { { "static_opponents", nlohmann::json::array() } };
	document["static_opponents"].push_back({ { "source_agent", HAND_WRITTEN_AGENT }, { "source_build", HAND_WRITTEN_BUILD }, { "source_version_id", "v1.0" } });
	document["static_opponents"].push_back(unresolvable);
	WriteTextFile(storage.StaticOpponentsPath(), document.dump(4));

	testing::internal::CaptureStderr();
	SyncStaticOpponentSelection(HAND_WRITTEN_AGENT, HAND_WRITTEN_BUILD);
	const std::string firstSync = ReadTextFile(storage.StaticOpponentsPath());
	SyncStaticOpponentSelection(HAND_WRITTEN_AGENT, HAND_WRITTEN_BUILD);
	const std::string warnings = testing::internal::GetCapturedStderr();

	const nlohmann::json synced = nlohmann::json::parse(firstSync);
	ASSERT_EQ(synced.at("static_opponents").size(), size_t(2));

	const nlohmann::json& resolved = synced.at("static_opponents").at(0);
	EXPECT_EQ(resolved.at("source_stage"), HAND_WRITTEN_STAGE);
	EXPECT_EQ(fs::path(resolved.at("checkpoint_path").get<std::string>()).filename().string(), "500");
	EXPECT_TRUE(resolved.contains("agent_setup"));

	EXPECT_EQ(synced.at("static_opponents").at(1), unresolvable);
	EXPECT_EQ(ReadTextFile(storage.StaticOpponentsPath()), firstSync);
	EXPECT_NE(warnings.find("v9.9"), std::string::npos) << warnings;
}

TEST(StaticOpponentSelectionStorage, SyncPathsListUnresolvableHandWrittenEntries)
{
	const BuildStorage storage = MakeHandWrittenModels("static-opponent-sync-paths");

	WriteTextFile(storage.StaticOpponentsPath(), R"({
		"static_opponents": [
			{ "source_agent": "static-opponent-hand-written-agent", "source_build": "V1", "source_version_id": "v1.0" },
			{ "source_agent": "static-opponent-hand-written-agent", "source_build": "V1", "source_version_id": "v9.9" },
			{ "source_agent": "static-opponent-hand-written-agent", "source_build": "", "source_version_id": "NotSeededYet" }
		]
	})");

	const std::vector<std::string> paths = ListOpponentModelSyncPaths(HAND_WRITTEN_AGENT, HAND_WRITTEN_BUILD);

	const std::string modelsGlob = RcloneGlobLiteral(ResolveAgentModelsRelativePath(HAND_WRITTEN_AGENT));
	const std::string segmentsGlob = modelsGlob + "/" + RcloneGlobLiteral(storage.BuildRoot().filename().string()) + "/segments/";
	EXPECT_EQ(paths, std::vector<std::string>({
		segmentsGlob + RcloneGlobLiteral(storage.MakeSegmentFolderName("v1.0", HAND_WRITTEN_STAGE)),
		segmentsGlob + "v9.9_*",
		modelsGlob + "/Imported Models/NotSeededYet" }));
}
