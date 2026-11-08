#include <gtest/gtest.h>

#include "TestSupport.h"

#include "Settings/ForgeDebugSettings.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace RocketForgeTraining;

namespace {
    fs::path DbgTempPath(const std::string& name) {
        return RocketForgeTests::TestTempDir() / name;
    }
}

TEST(ForgeDebugSettings, RoundTripPreservesAllFields) {
    ForgeDebugSettings original;
    original.buildName  = "V3";
    original.segmentId  = "v2.01";
    original.renderStageName = "SomeStage";
    original.showRewards = true;
    original.debug.gameSpeed              = 2.5f;
    original.debug.teamSizeOverrideIndex  = 1;
    original.debug.forceSingleStateSetter = true;
    original.debug.forceStateSetterIndex  = 2;
    original.debug.maxEpisodeTimeEnabled  = true;
    original.debug.maxEpisodeTime         = 12.0f;
    original.debug.includeStaticOpponent  = true;
    original.debug.staticOpponentSelectionKey = "test-agent\nmodel-b\ntest-stage\nv1.04";
    original.debug.playerIndex            = 3;
    original.debug.forceDebugAll          = true;
    original.debug.normMode               = ForgeDebug::NormalizationMode::PER_REWARD_NORMALIZED;
    original.debug.displayMode            = ForgeDebug::RewardDisplayMode::TRAINING;
    original.debug.debugEnabledRewards = { "SpeedReward", "TouchReward" };
    original.debug.hasRewardSelection  = true;
    original.debug.showCriticHeads = false;
    original.debug.debugEnabledCriticHeads = { "TouchReward", "GoalReward" };
    original.debug.hasCriticHeadSelection = true;

    fs::path path = DbgTempPath("roundtrip.json");
    ASSERT_TRUE(GGL::JsonSaveToFile(original, path));

    ForgeDebugSettingsLoadResult loaded = LoadForgeDebugSettings(path);
    ASSERT_TRUE(loaded.ok);
    ASSERT_EQ(loaded.value.buildName, std::string("V3"));
    ASSERT_EQ(loaded.value.segmentId, std::string("v2.01"));
    ASSERT_EQ(loaded.value.renderStageName, std::string("SomeStage"));
    ASSERT_TRUE(loaded.value.showRewards);
    ASSERT_EQ(loaded.value.debug.gameSpeed, 2.5f);
    ASSERT_EQ(loaded.value.debug.teamSizeOverrideIndex, 1);
    ASSERT_TRUE(loaded.value.debug.forceSingleStateSetter);
    ASSERT_EQ(loaded.value.debug.forceStateSetterIndex, 2);
    ASSERT_TRUE(loaded.value.debug.maxEpisodeTimeEnabled);
    ASSERT_EQ(loaded.value.debug.maxEpisodeTime, 12.0f);
    ASSERT_TRUE(loaded.value.debug.includeStaticOpponent);
    ASSERT_EQ(loaded.value.debug.staticOpponentSelectionKey, std::string("test-agent\nmodel-b\ntest-stage\nv1.04"));
    ASSERT_EQ(loaded.value.debug.playerIndex, 3);
    ASSERT_TRUE(loaded.value.debug.forceDebugAll);
    ASSERT_TRUE(loaded.value.debug.normMode == ForgeDebug::NormalizationMode::PER_REWARD_NORMALIZED);
    ASSERT_TRUE(loaded.value.debug.displayMode == ForgeDebug::RewardDisplayMode::TRAINING);
    ASSERT_TRUE(loaded.value.debug.hasRewardSelection);
    ASSERT_EQ(loaded.value.debug.debugEnabledRewards.size(), size_t(2));
    ASSERT_EQ(loaded.value.debug.debugEnabledRewards[0], std::string("SpeedReward"));
    ASSERT_EQ(loaded.value.debug.debugEnabledRewards[1], std::string("TouchReward"));
    ASSERT_FALSE(loaded.value.debug.showCriticHeads);
    ASSERT_EQ(loaded.value.debug.debugEnabledCriticHeads.size(), size_t(2));
    ASSERT_EQ(loaded.value.debug.debugEnabledCriticHeads[1], std::string("GoalReward"));
    ASSERT_TRUE(loaded.value.debug.hasCriticHeadSelection);

    fs::remove(path);
}

TEST(ForgeDebugSettings, LoadMissingFileReportsError) {
    fs::path nonexistent = DbgTempPath("missing.json");
    fs::remove(nonexistent);
    ForgeDebugSettingsLoadResult r = LoadForgeDebugSettings(nonexistent);
    ASSERT_FALSE(r.ok);
    ASSERT_FALSE(r.errors.empty());
}

TEST(ForgeDebugSettings, LoadWrongSchemaVersionReportsError) {
    fs::path path = DbgTempPath("wrong_schema.json");
    std::ofstream out(path);
    out << R"({ "schemaVersion": 999 })";
    out.close();
    ForgeDebugSettingsLoadResult r = LoadForgeDebugSettings(path);
    ASSERT_FALSE(r.ok);
    fs::remove(path);
}

TEST(ForgeDebugSettings, LoadMalformedJsonReportsError) {
    fs::path path = DbgTempPath("malformed.json");
    std::ofstream out(path);
    out << "{ this is not valid json";
    out.close();
    ForgeDebugSettingsLoadResult r = LoadForgeDebugSettings(path);
    ASSERT_FALSE(r.ok);
    ASSERT_FALSE(r.errors.empty());
    fs::remove(path);
}

TEST(ForgeDebugSettings, OldFileDefaultsCriticFields) {
    fs::path path = DbgTempPath("critic_old_file.json");
    {
        std::ofstream out(path);
        out << R"({ "schemaVersion": )" << kForgeDebugSettingsSchemaVersion
            << R"(, "showRewards": true, "debug": { "playerIndex": 1 } })";
    }

    ForgeDebugSettingsLoadResult loaded = LoadForgeDebugSettings(path);
    ASSERT_TRUE(loaded.ok);
    ASSERT_TRUE(loaded.value.debug.showCriticHeads);
    ASSERT_TRUE(loaded.value.debug.debugEnabledCriticHeads.empty());
    ASSERT_FALSE(loaded.value.debug.hasCriticHeadSelection);
    fs::remove(path);
}
