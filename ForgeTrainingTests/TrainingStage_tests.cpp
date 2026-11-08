#include <gtest/gtest.h>

#include <TrainingStage.h>

namespace {
	struct CountingStateSetter : RLGC::StateSetter {
		explicit CountingStateSetter(int& count, RLGC::StateSetterType type = RLGC::StateSetterType::Scenario)
			: count(count) { setterType = type; }

		void ResetArena(RocketSim::Arena*) override {
			count++;
		}

		int& count;
	};

	class TestStateSetterStage : public RocketForge::TrainingStage {
	public:
		TestStateSetterStage(int& regularCount, int& kickoffCount, float regularWeight)
			: TrainingStage(std::make_unique<Agent>())
			, regularCount(regularCount)
			, kickoffCount(kickoffCount)
			, regularWeight(regularWeight)
		{
			disableKickoffsForUnfairMatchups = true;
		}

	protected:
		std::vector<RocketForge::WeightedStateSetter> GetStateSetters() const override {
			return {
				{ new CountingStateSetter(regularCount), regularWeight },
				{ new CountingStateSetter(kickoffCount, RLGC::StateSetterType::Kickoff), 1.f },
			};
		}

	private:
		int& regularCount;
		int& kickoffCount;
		float regularWeight;
	};
}

TEST(TrainingStage, FiltersKickoffStateSettersForUnfairMatchups) {
	int regularCount = 0;
	int kickoffCount = 0;
	TestStateSetterStage stage(regularCount, kickoffCount, 1.f);

	auto setter = stage.GetStateSetter(2, 1);
	for (int i = 0; i < 5; i++)
		setter->ResetArena(nullptr);

	ASSERT_EQ(regularCount, 5);
	ASSERT_EQ(kickoffCount, 0);
}

TEST(TrainingStage, KeepsKickoffStateSettersForFairMatchups) {
	int regularCount = 0;
	int kickoffCount = 0;
	TestStateSetterStage stage(regularCount, kickoffCount, 0.f);

	auto setter = stage.GetStateSetter(2, 2);
	setter->ResetArena(nullptr);

	ASSERT_EQ(regularCount, 0);
	ASSERT_EQ(kickoffCount, 1);
}

TEST(TrainingStage, ForceSingleStateSetterUsesSelectedOption) {
	int regularCount = 0;
	int kickoffCount = 0;
	TestStateSetterStage stage(regularCount, kickoffCount, 1.f);
	ForgeDebug::DebugSettings debug;
	debug.forceSingleStateSetter = true;
	debug.forceStateSetterIndex = 1;
	stage.SetDebugSettings(&debug);

	auto setter = stage.GetStateSetter(2, 1);
	for (int i = 0; i < 5; i++)
		setter->ResetArena(nullptr);

	ASSERT_EQ(regularCount, 0);
	ASSERT_EQ(kickoffCount, 5);
}

TEST(TrainingStage, NormalGameplaySetterExcludesScenario) {
	// Fair matchup: both a Scenario setter (weight 1) and a Kickoff setter (weight 1) are eligible,
	// but the normal-gameplay-only setter must reset via the Kickoff one exclusively.
	int scenarioCount = 0;
	int kickoffCount = 0;
	TestStateSetterStage stage(scenarioCount, kickoffCount, 1.f);

	auto setter = stage.GetNormalGameplayStateSetter(2, 2);
	ASSERT_TRUE(setter != nullptr);
	for (int i = 0; i < 5; i++)
		setter->ResetArena(nullptr);

	ASSERT_EQ(scenarioCount, 0);
	ASSERT_EQ(kickoffCount, 5);
}

TEST(TrainingStage, NormalGameplaySetterNullWhenNoneRemain) {
	// 2v1 filters out the Kickoff setter (disableKickoffsForUnfairMatchups); the only survivor is
	// Scenario, which is not normal-gameplay -> nullptr so the caller skips adaptive sampling.
	int scenarioCount = 0;
	int kickoffCount = 0;
	TestStateSetterStage stage(scenarioCount, kickoffCount, 1.f);

	auto setter = stage.GetNormalGameplayStateSetter(2, 1);
	ASSERT_TRUE(setter == nullptr);
}

TEST(StateSetterType, ClassifiesNormalGameplayEpisodes) {
	ASSERT_TRUE(RLGC::IsNormalGameplayEpisode(RLGC::StateSetterType::Kickoff));
	ASSERT_TRUE(RLGC::IsNormalGameplayEpisode(RLGC::StateSetterType::Random));
	ASSERT_FALSE(RLGC::IsNormalGameplayEpisode(RLGC::StateSetterType::Scenario));
}

TEST(TrainingStage, AppliesAdvantageNormalizationOverrides) {
	int regularCount = 0;
	int kickoffCount = 0;
	TestStateSetterStage stage(regularCount, kickoffCount, 1.f);

	RocketForgeTraining::StageParameterOverride normalizeOverride;
	normalizeOverride.group = "Training";
	normalizeOverride.name = "Normalize Advantages";
	normalizeOverride.value = "true";

	RocketForgeTraining::StageParameterOverride clipOverride;
	clipOverride.group = "Training";
	clipOverride.name = "Policy Grad Clip Norm";
	clipOverride.value = "1.25";

	stage.SetParameterOverrides({ normalizeOverride, clipOverride });

	ASSERT_TRUE(stage.learnerConfig.ppo.normalizeAdvantages);
	ASSERT_EQ(stage.learnerConfig.ppo.policyGradClipNorm, 1.25f);
}

TEST(TrainingStage, AppliesCriticHeadLossWeightingOverride) {
	int regularCount = 0;
	int kickoffCount = 0;
	TestStateSetterStage stage(regularCount, kickoffCount, 1.f);

	RocketForgeTraining::StageParameterOverride weightingOverride;
	weightingOverride.group = "Training";
	weightingOverride.name = "Critic Head Loss Weighting";
	weightingOverride.value = "Contribution";
	stage.SetParameterOverrides({ weightingOverride });

	ASSERT_TRUE(stage.learnerConfig.ppo.decomposedCritic.lossWeighting == GGL::CriticHeadLossWeighting::Contribution);
}

TEST(TrainingStage, RejectsUnknownCriticHeadLossWeighting) {
	int regularCount = 0;
	int kickoffCount = 0;
	TestStateSetterStage stage(regularCount, kickoffCount, 1.f);

	RocketForgeTraining::StageParameterOverride weightingOverride;
	weightingOverride.group = "Training";
	weightingOverride.name = "Critic Head Loss Weighting";
	weightingOverride.value = "Squared";

	std::string message;
	try {
		stage.SetParameterOverrides({ weightingOverride });
	} catch (const std::exception& e) {
		message = e.what();
	}

	ASSERT_TRUE(message.find("must be Equal or Contribution") != std::string::npos);
}

namespace {
	class NoUpdateCallStage : public RocketForge::TrainingStage {
	public:
		explicit NoUpdateCallStage(std::unique_ptr<Agent> agent) : TrainingStage(std::move(agent)) {
			stageName = "no-update-call";
			timestepsPerIteration = 123'000;
			learningRate = 2e-4f;
		}
	};

	class DerivedNoUpdateCallStage : public NoUpdateCallStage {
	public:
		explicit DerivedNoUpdateCallStage(std::unique_ptr<Agent> agent) : NoUpdateCallStage(std::move(agent)) {
			timestepsPerIteration = 456'000;
		}
	};
}

TEST(TrainingStage, CreateAppliesStagePropertiesWithoutExplicitUpdateConfigs) {
	std::unique_ptr<NoUpdateCallStage> stage = RocketForge::TrainingStage::Create<NoUpdateCallStage>(std::make_unique<Agent>());
	EXPECT_EQ(stage->learnerConfig.ppo.tsPerItr, 123'000);
	EXPECT_EQ(stage->learnerConfig.ppo.batchSize, 123'000);
	EXPECT_FLOAT_EQ(stage->learnerConfig.ppo.policyLR, 2e-4f);
	EXPECT_EQ(stage->learnerConfig.metricsGroupName, "no-update-call");
}

TEST(TrainingStage, CreateAppliesMostDerivedStageProperties) {
	std::unique_ptr<DerivedNoUpdateCallStage> stage = RocketForge::TrainingStage::Create<DerivedNoUpdateCallStage>(std::make_unique<Agent>());
	EXPECT_EQ(stage->learnerConfig.ppo.tsPerItr, 456'000);
}

TEST(TrainingStage, CreateAppliesStageNameSetInConstructor) {
	std::unique_ptr<NoUpdateCallStage> stage = RocketForge::TrainingStage::Create<NoUpdateCallStage>(std::make_unique<Agent>());
	EXPECT_EQ(stage->learnerConfig.runtime.trainingStageName, "no-update-call");
}
