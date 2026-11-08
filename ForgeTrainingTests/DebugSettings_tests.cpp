#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Settings/ForgeSettings.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace RocketForgeTraining;

namespace {
	fs::path TempPath(const std::string& name) {
		return RocketForgeTests::TestTempDir() / name;
	}
}

TEST(DebugSettings, DefaultsWhenMissingFromJson) {
	fs::path path = TempPath("debug_missing.json");
	{
		std::ofstream out(path);
		out << R"({"schemaVersion":1,"agentName":"x","globals":{"metricsEnabled":true,"render":false,"showRewards":false,"pruneSourceCheckpoints":false},"stages":[{"stageName":"s","targetTotalTimesteps":100000}]})";
	}

	ForgeSettingsLoadResult loaded = LoadForgeSettings(path);
	ASSERT_TRUE(loaded.ok);

	const auto& d = loaded.value.globals.debug;
	ASSERT_EQ((int)d.normMode,            (int)ForgeDebug::NormalizationMode::RAW_WEIGHTED);
	ASSERT_EQ((int)d.displayMode,         (int)ForgeDebug::RewardDisplayMode::EARNED);
	ASSERT_EQ(d.playerIndex,              0);
	ASSERT_FALSE(d.forceDebugAll);
	ASSERT_EQ(d.gameSpeed,               1.0f);
	ASSERT_EQ(d.teamSizeOverrideIndex,   -1);
	ASSERT_FALSE(d.forceSingleStateSetter);
	ASSERT_EQ(d.forceStateSetterIndex,   0);
	ASSERT_FALSE(d.maxEpisodeTimeEnabled);
	ASSERT_EQ(d.maxEpisodeTime,           30.0f);
	ASSERT_FALSE(d.hasRewardSelection);
	ASSERT_TRUE(d.debugEnabledRewards.empty());

	fs::remove(path);
}
