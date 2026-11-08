#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Storage/BuildStorage.h"
#include "TrainingSetup/TrainingSetupSnapshot.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace RocketForge;
using namespace RocketForgeTraining;

namespace
{
	namespace fs = std::filesystem;

	// Golden values pin the hashes saved setups already carry; never regenerate them to make this pass.
	const std::vector<std::pair<std::string, std::string>> kGoldenHashes = {
		{ "legacy_ball_chase.json", "736ea1ceb0cc89b9" },
		{ "legacy_basic_score.json", "6b10740efb3037eb" },
		{ "perf_basic_score.json", "e7302cef7c41673d" },
	};

	fs::path SourceRoot() { return fs::path(FORGE_SOURCE_DIR); }
	fs::path LocalGoldenPath() { return SourceRoot() / "ForgeTrainingTests" / "fixtures" / "local" / "setup_hash_golden.json"; }

	TrainingSetupSnapshot LoadSnapshotFile(const fs::path& path)
	{
		std::string error;
		const std::optional<TrainingSetupSnapshot> snapshot = LoadTrainingSetupSnapshot(path, &error);
		if (!snapshot.has_value())
			throw std::runtime_error("Could not load " + path.string() + ": " + error);
		return snapshot.value();
	}

	/** True when a setup saved a non-native prediction rate; retiring that setting deliberately changed such hashes. **/
	bool RecordsNonNativePredictionRate(const fs::path& setupFile)
	{
		const nlohmann::json setup = nlohmann::json::parse(std::ifstream(setupFile), nullptr, false);
		const nlohmann::json* agentSetup = setup.is_object() && setup.contains("agent_setup") ? &setup["agent_setup"] : nullptr;
		return agentSetup && agentSetup->is_object() && agentSetup->value("ball_pred_tick_rate", 0.f) != 0.f;
	}

	/** Every <Models>/<build>/segments/<segment>/training_setup.json under Bots/ whose hash is meant to be unchanged. **/
	std::vector<fs::path> FindLocalSetupFiles()
	{
		std::vector<fs::path> found;
		const fs::path botsRoot = SourceRoot() / "Bots";
		for (fs::recursive_directory_iterator it(botsRoot), end; it != end; ++it) {
			if (!it->is_regular_file() || it->path().filename() != TrainingSetupSnapshot::FILE_NAME)
				continue;

			const fs::path segmentFolder = it->path().parent_path();
			if (segmentFolder.parent_path().filename() != BuildStorage::SEGMENTS_FOLDER_NAME)
				continue;
			if (!fs::is_regular_file(segmentFolder / BuildStorage::SEGMENT_FILE_NAME))
				continue;
			if (RecordsNonNativePredictionRate(it->path()))
				continue;
			found.push_back(it->path());
		}
		std::sort(found.begin(), found.end());
		return found;
	}

	std::string RelativeKey(const fs::path& path)
	{
		return fs::relative(path, SourceRoot()).generic_string();
	}

	/** Builds the storage and segment metadata that own a segment's training_setup.json. **/
	std::pair<BuildStorage, std::optional<TrainingSegmentMetadata>> OpenSegment(const fs::path& setupFile)
	{
		const fs::path segmentFolder = setupFile.parent_path();
		const fs::path buildFolder = segmentFolder.parent_path().parent_path();
		const fs::path modelsFolder = buildFolder.parent_path();

		// The build folder is the sanitized build name, and sanitizing is idempotent.
		BuildStorage storage(modelsFolder, buildFolder.filename().string());
		std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentByFolderName(segmentFolder.filename().string());
		return { std::move(storage), std::move(segment) };
	}

	std::map<std::string, std::string> LoadLocalGolden()
	{
		std::map<std::string, std::string> golden;
		std::ifstream stream(LocalGoldenPath());
		if (!stream)
			return golden;
		const nlohmann::json parsed = nlohmann::json::parse(stream);
		for (const auto& [path, hash] : parsed.items())
			golden[path] = hash.get<std::string>();
		return golden;
	}

	void WriteLocalGolden()
	{
		nlohmann::json golden = nlohmann::json::object();
		for (const fs::path& setupFile : FindLocalSetupFiles())
			golden[RelativeKey(setupFile)] = BuildStorage::MakeSetupHash(LoadSnapshotFile(setupFile));

		fs::create_directories(LocalGoldenPath().parent_path());
		std::ofstream(LocalGoldenPath()) << golden.dump(1) << "\n";
	}
}

TEST(SetupHashStability, TrackedFixturesKeepTheirHashes)
{
	for (const auto& [file, golden] : kGoldenHashes) {
		const TrainingSetupSnapshot snapshot =
			LoadSnapshotFile(SourceRoot() / "ForgeTrainingTests" / "fixtures" / "setup_hash" / file);
		EXPECT_EQ(BuildStorage::MakeSetupHash(snapshot), golden) << file;
	}
}

TEST(SetupHashStability, LocalSegmentsKeepTheirHashes)
{
	if (std::getenv("ROCKETFORGE_WRITE_SETUP_HASH_GOLDEN") != nullptr)
		WriteLocalGolden();

	const std::map<std::string, std::string> localGolden = LoadLocalGolden();
	if (localGolden.empty())
		GTEST_SKIP() << "No local golden at " << LocalGoldenPath().string();

	for (const auto& [relativePath, golden] : localGolden) {
		const fs::path setupFile = SourceRoot() / relativePath;
		const TrainingSetupSnapshot snapshot = LoadSnapshotFile(setupFile);
		const std::string hash = BuildStorage::MakeSetupHash(snapshot);
		EXPECT_EQ(hash, golden) << relativePath;

		auto [storage, segment] = OpenSegment(setupFile);
		ASSERT_TRUE(segment.has_value()) << relativePath;
		EXPECT_TRUE(storage.SegmentSetupHashMatches(segment.value(), snapshot, hash)) << relativePath;
	}
}
