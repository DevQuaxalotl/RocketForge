#include <gtest/gtest.h>

#include <StageRegistry.h>
#include <GigaLearnInfer/Agent.h>

#include <algorithm>

TEST(StageRegistry, GetCompatibleStages_FiltersIncompatible) {
	RocketForge::StageRegistry::Register(
		"test-compat-teamsize1", nullptr,
		RocketForge::StageCompatInfo{ .minAgentTeamSize = 1 });
	RocketForge::StageRegistry::Register(
		"test-compat-teamsize3", nullptr,
		RocketForge::StageCompatInfo{ .minAgentTeamSize = 3 });

	Agent agent1; agent1.maxTeamSize = 1;
	Agent agent3; agent3.maxTeamSize = 3;

	auto for1 = RocketForge::StageRegistry::GetCompatibleStages(agent1);
	auto for3 = RocketForge::StageRegistry::GetCompatibleStages(agent3);

	auto contains = [](const std::vector<std::string>& v, const std::string& s) {
		return std::find(v.begin(), v.end(), s) != v.end();
	};

	// agent with maxTeamSize=3 sees both stages
	ASSERT_TRUE(contains(for3, "test-compat-teamsize1"));
	ASSERT_TRUE(contains(for3, "test-compat-teamsize3"));

	// agent with maxTeamSize=1 sees only the size-1 stage
	ASSERT_TRUE(contains(for1, "test-compat-teamsize1"));
	ASSERT_FALSE(contains(for1, "test-compat-teamsize3"));
}

TEST(StageRegistry, GetCompatInfo_ReturnsDefaultForUnknown) {
	RocketForge::StageCompatInfo info = RocketForge::StageRegistry::GetCompatInfo("nonexistent-stage");
	ASSERT_EQ(info.minAgentTeamSize, 1);
}

namespace {
	class RegistryNoUpdateCallStage : public RocketForge::TrainingStage {
	public:
		explicit RegistryNoUpdateCallStage(std::unique_ptr<Agent> agent) : TrainingStage(std::move(agent)) {
			timestepsPerIteration = 123'000;
		}
	};
}

REGISTER_STAGE("test-registry-update-configs", RegistryNoUpdateCallStage, RocketForge::StageCompatInfo{});

TEST(StageRegistry, CreateAppliesStagePropertiesWithoutExplicitUpdateConfigs) {
	std::unique_ptr<RocketForge::TrainingStage> stage =
		RocketForge::StageRegistry::Create("test-registry-update-configs", std::make_unique<Agent>());
	ASSERT_EQ(stage->learnerConfig.ppo.tsPerItr, 123'000);
}
