#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Storage/ImportedModelStorage.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace RocketForgeTraining;

namespace
{
	struct TempFolder
	{
		fs::path path;
		~TempFolder()
		{
			std::error_code ec;
			fs::remove_all(path, ec);
		}
	};

	void Touch(const fs::path& p)
	{
		fs::create_directories(p.parent_path());
		std::ofstream(p, std::ios::binary) << "x";
	}

	fs::path TempImportedModelStorageRoot(const std::string& name)
	{
		return RocketForgeTests::TestTempDir(name);
	}
}

TEST(ImportedModelStorage, InferencePolicyCapability)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	const fs::path policyOnly = temp.path / "policy";
	Touch(policyOnly / "POLICY.lt");
	ASSERT_TRUE(InspectImportedModelTrainingFiles(policyOnly).hasInferencePolicy);

	const fs::path scriptedOnly = temp.path / "scripted";
	Touch(scriptedOnly / "INFERENCE_POLICY.json");
	ASSERT_TRUE(InspectImportedModelTrainingFiles(scriptedOnly).hasInferencePolicy);

	const fs::path criticOnly = temp.path / "critic";
	Touch(criticOnly / "CRITIC.lt");
	ASSERT_FALSE(InspectImportedModelTrainingFiles(criticOnly).hasInferencePolicy);

	const fs::path empty = temp.path / "empty";
	fs::create_directories(empty);
	ASSERT_FALSE(InspectImportedModelTrainingFiles(empty).hasInferencePolicy);
}

TEST(ImportedModelStorage, ScriptedModelIsInferenceCapable)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	const fs::path root = temp.path / "RLBotPack";
	Touch(root / "INFERENCE_POLICY.json");
	Touch(root / "nexto-model.pt");
	Touch(root / IMPORTED_MODEL_METADATA_FILE_NAME);
	std::ofstream(root / IMPORTED_MODEL_METADATA_FILE_NAME, std::ios::binary)
		<< R"({"schema_version":1,"folder_name":"RLBotPack","can_use_for_inference":true})";

	std::optional<ImportedModelMetadata> metadata = LoadImportedModelMetadata(root);
	ASSERT_TRUE(metadata.has_value());
	ASSERT_TRUE(metadata->canUseForInference);
	ASSERT_FALSE(metadata->canUseForTrainingResume);
}

TEST(ImportSourceInspection, CopiesOnlyKnownModelFiles)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	// A bot release folder: model files next to RLBot config and art.
	const fs::path source = temp.path / "rlbot";
	Touch(source / "POLICY.lt");
	Touch(source / "SHARED_HEAD.lt");
	Touch(source / "bot.toml");
	Touch(source / "logo.png");

	const ImportSourceInspection inspection = InspectImportSource(source);

	ASSERT_TRUE(inspection.IsLoadable());
	ASSERT_EQ(inspection.modelFolder, source);
	ASSERT_EQ(inspection.filesToCopy, (std::vector<std::string>{ "POLICY.lt", "SHARED_HEAD.lt" }));
	ASSERT_EQ(inspection.bytesToCopy, (uint64_t)2);
	ASSERT_TRUE(inspection.files.hasInferencePolicy);
	ASSERT_FALSE(inspection.segment.has_value());
}

TEST(ImportSourceInspection, RunFolderUsesNewestCheckpointWithTrainingState)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	const fs::path run = temp.path / "late-stage-run";
	Touch(run / "100" / "POLICY.lt");

	const char* trainingFileNames[] = {
		"POLICY.lt", "SHARED_HEAD.lt", "CRITIC.lt", "POLICY_OPTIM.lt", "SHARED_HEAD_OPTIM.lt", "CRITIC_OPTIM.lt",
	};
	for (const char* fileName : trainingFileNames)
		Touch(run / "200" / fileName);

	std::ofstream(run / "200" / "RUNNING_STATS.json") << R"({"total_timesteps": 0})";
	Touch(run / "policy_versions" / "150" / "POLICY.lt");

	const ImportSourceInspection inspection = InspectImportSource(run);

	ASSERT_EQ(inspection.modelFolder.filename().string(), std::string("200"));
	ASSERT_EQ(inspection.timesteps, (uint64_t)200);
	ASSERT_EQ(inspection.filesToCopy.size(), (size_t)7);
	ASSERT_TRUE(inspection.files.MissingTrainingFiles().empty());
}

TEST(ImportSourceInspection, SegmentFolderResolvesCheckpointAndReadsSegment)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	const fs::path segmentRoot = temp.path / "Models" / "V1" / "segments" / "v1.2_ball-chase";

	TrainingSegmentMetadata segment = {};
	segment.segmentId = "v1.2";
	segment.buildName = "V1";
	segment.stageName = "ball-chase";

	nlohmann::json segmentJson;
	GGL::JsonSave(segment, segmentJson);
	fs::create_directories(segmentRoot);
	std::ofstream(segmentRoot / "segment.json") << segmentJson.dump();
	Touch(segmentRoot / "checkpoints" / "500" / "POLICY.lt");

	const ImportSourceInspection inspection = InspectImportSource(segmentRoot);

	ASSERT_EQ(inspection.modelFolder.filename().string(), std::string("500"));
	ASSERT_TRUE(inspection.segment.has_value());
	ASSERT_EQ(inspection.segment->segmentId, std::string("v1.2"));
	ASSERT_EQ(inspection.segment->buildName, std::string("V1"));
	ASSERT_EQ(inspection.segment->stageName, std::string("ball-chase"));
}

TEST(ImportSourceInspection, ScriptedPolicyBringsOnlyArtifactsBesideIt)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	const fs::path source = temp.path / "scripted";
	fs::create_directories(source);
	std::ofstream(source / "INFERENCE_POLICY.json")
		<< R"({"engine":"torchscript","artifact":"model.pt","original_artifact":"../outside.pt"})";
	Touch(source / "model.pt");
	Touch(temp.path / "outside.pt");

	const ImportSourceInspection inspection = InspectImportSource(source);

	ASSERT_EQ(inspection.filesToCopy, (std::vector<std::string>{ "INFERENCE_POLICY.json", "model.pt" }));
}

TEST(ImportSourceInspection, FolderWithoutModelIsNotLoadable)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	Touch(temp.path / "notes.txt");

	const ImportSourceInspection inspection = InspectImportSource(temp.path);

	ASSERT_FALSE(inspection.IsLoadable());
	ASSERT_TRUE(inspection.filesToCopy.empty());
}

TEST(ImportedModelStorage, EnumerateImportedModelsInRoots)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	const fs::path nexto = temp.path / "RLBotPack";
	Touch(nexto / "INFERENCE_POLICY.json");
	Touch(nexto / "nexto-model.pt");

	Touch(temp.path / "__scratch" / "INFERENCE_POLICY.json");

	fs::create_directories(temp.path / "not-a-model");

	std::vector<DiscoveredImportedModel> found = EnumerateImportedModelsInRoots({ temp.path });

	ASSERT_EQ(found.size(), (size_t)1);
	ASSERT_EQ(found[0].folderName, std::string("RLBotPack"));
	ASSERT_TRUE(found[0].canUseForInference);
	ASSERT_EQ(found[0].resolvedModelFolder.filename().string(), std::string("RLBotPack"));
	ASSERT_EQ(found[0].importedRoot.filename().string(), std::string("RLBotPack"));
}

TEST(ImportedModelStorage, EnumerateSkipsMalformedMetadata)
{
	TempFolder temp{ RocketForgeTests::TestTempDir() };

	// Good scripted model.
	const fs::path good = temp.path / "Good";
	Touch(good / "INFERENCE_POLICY.json");

	// Resolvable folder whose imported_model.json has a type-incompatible field
	// (latest_timesteps as a string) that makes LoadImportedModelMetadata throw.
	const fs::path bad = temp.path / "Bad";
	Touch(bad / "INFERENCE_POLICY.json");
	std::ofstream(bad / IMPORTED_MODEL_METADATA_FILE_NAME, std::ios::binary)
		<< R"({"schema_version":1,"latest_timesteps":"not-a-number"})";

	std::vector<DiscoveredImportedModel> found = EnumerateImportedModelsInRoots({ temp.path });

	// The malformed folder is skipped; the good one still enumerates.
	ASSERT_EQ(found.size(), (size_t)1);
	ASSERT_EQ(found[0].folderName, std::string("Good"));
}

TEST(ImportedModelStorage, SaveLoadRoundTripsSourcePathAndImportedAt)
{
	fs::path root = TempImportedModelStorageRoot("round_trip");

	ImportedModelMetadata metadata = {};
	metadata.folderName = "example-import";
	metadata.displayName = "Example Import";
	metadata.targetAgentName = "example-agent";
	metadata.sourcePath = "C:/models/example";
	metadata.importedAt = "20260101-010101";

	SaveImportedModelMetadata(root, metadata);
	std::optional<ImportedModelMetadata> loaded = LoadImportedModelMetadata(root);

	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->sourcePath, std::string("C:/models/example"));
	ASSERT_EQ(loaded->importedAt, std::string("20260101-010101"));

	fs::remove_all(root);
}

TEST(ImportedModelStorage, ExplicitNullSourcePathAndImportedAtLoadAsEmpty)
{
	fs::path root = TempImportedModelStorageRoot("null_tolerant");
	fs::create_directories(root);

	// Some existing files on disk carry explicit JSON null for these two keys rather than
	// omitting them or writing an empty string; loading must tolerate that (clearing the
	// field to "") without throwing.
	std::ofstream fOut(root / IMPORTED_MODEL_METADATA_FILE_NAME);
	fOut << R"({
		"folder_name": "null-import",
		"display_name": "Null Import",
		"target_agent": "example-agent",
		"source_path": null,
		"imported_at": null
	})";
	fOut.close();

	std::optional<ImportedModelMetadata> loaded = LoadImportedModelMetadata(root);

	ASSERT_TRUE(loaded.has_value());
	ASSERT_EQ(loaded->sourcePath, std::string(""));
	ASSERT_EQ(loaded->importedAt, std::string(""));
	ASSERT_EQ(loaded->folderName, std::string("null-import"));

	fs::remove_all(root);
}

TEST(ImportedModelStorage, SaveAlwaysWritesPlainStringsNeverNull)
{
	fs::path root = TempImportedModelStorageRoot("save_never_null");

	// Save always writes plain (possibly empty) strings for source_path/imported_at -- unlike
	// FieldNullIfEmpty fields elsewhere -- this is the deliberate asymmetry documented on
	// ImportedModelMetadata::Serialize.
	ImportedModelMetadata metadata = {};
	metadata.folderName = "empty-fields";
	metadata.sourcePath = "";
	metadata.importedAt = "";

	SaveImportedModelMetadata(root, metadata);

	std::ifstream fIn(root / IMPORTED_MODEL_METADATA_FILE_NAME);
	nlohmann::json j = nlohmann::json::parse(fIn);
	fIn.close();

	ASSERT_TRUE(j.contains("source_path"));
	ASSERT_TRUE(j["source_path"].is_string());
	ASSERT_TRUE(j.contains("imported_at"));
	ASSERT_TRUE(j["imported_at"].is_string());

	fs::remove_all(root);
}
