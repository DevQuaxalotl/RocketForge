#include <gtest/gtest.h>
#include <GigaLearnCPP/TrainingStorage.h>

#include <optional>

// Returns agentModelsFolder / "Temp Checkpoints" regardless of stageRoot.
TEST(TrainingStorage, GetAgentCheckpointFolder_UsesAgentRoot) {
    GGL::TrainingStorage storage(
        std::filesystem::path("C:/Agents/TestAgent/Models/Training Stages/test-stage"),
        std::filesystem::path("C:/Agents/TestAgent/Models"));

    std::filesystem::path expected =
        std::filesystem::path("C:/Agents/TestAgent/Models/Temp Checkpoints").make_preferred();

    ASSERT_EQ(storage.GetAgentCheckpointFolder(), expected);
}

// Returns empty when agentModelsFolder is not set.
TEST(TrainingStorage, GetAgentCheckpointFolder_EmptyWhenNoAgentRoot) {
    GGL::TrainingStorage storage({}, {});
    ASSERT_TRUE(storage.GetAgentCheckpointFolder().empty());
}

// When agentModelsFolder is not supplied explicitly, it is inferred from stageRootFolder
// by walking up and stripping the "Training Stages" directory component.
TEST(TrainingStorage, GetAgentCheckpointFolder_InfersAgentRootFromStageRoot) {
    GGL::TrainingStorage storage(
        std::filesystem::path("C:/Agents/TestAgent/Models/Training Stages/test-stage"),
        {});  // no explicit agentModelsFolder

    std::filesystem::path expected =
        std::filesystem::path("C:/Agents/TestAgent/Models/Temp Checkpoints").make_preferred();

    ASSERT_EQ(storage.GetAgentCheckpointFolder(), expected);
}

TEST(TrainingStorage, RecentPolicyVersionFolderName_PrefixesSegmentId) {
    ASSERT_EQ(
        GGL::TrainingStorage::MakeRecentPolicyVersionFolderName("v1.00", 1200),
        std::string("v1.00_1200"));
    ASSERT_EQ(
        GGL::TrainingStorage::MakeRecentPolicyVersionFolderName("", 1200),
        std::string("1200"));
}

TEST(TrainingStorage, TryParseRecentPolicyVersionTimesteps_ReadsSegmentPrefixedNames) {
    std::optional<uint64_t> current =
        GGL::TrainingStorage::TryParseRecentPolicyVersionTimesteps("v1.00_1200", "v1.00");
    std::optional<uint64_t> unprefixed =
        GGL::TrainingStorage::TryParseRecentPolicyVersionTimesteps("900", "v1.00");
    std::optional<uint64_t> sibling =
        GGL::TrainingStorage::TryParseRecentPolicyVersionTimesteps("v1.01_1200", "v1.00");

    ASSERT_TRUE(current.has_value());
    ASSERT_EQ(current.value(), uint64_t(1200));
    ASSERT_FALSE(unprefixed.has_value());
    ASSERT_FALSE(sibling.has_value());
}

TEST(TrainingStorage, MakeBenchmarkStorageLocation_PrefixesSegmentAndStage) {
    std::filesystem::path root = std::filesystem::path("C:/Agents/TestAgent/Models/V1/skill_benchmarks");
    GGL::BenchmarkStorageLocation location =
        GGL::TrainingStorage::MakeBenchmarkStorageLocation(root, "kickoff", 1200, "v1.00");

    ASSERT_EQ(location.id, std::string("v1.00_1200_kickoff"));
    ASSERT_EQ(location.folderName, std::string("v1.00_1200_kickoff"));
    ASSERT_EQ(location.folder, (root / "v1.00_1200_kickoff").make_preferred());
}
