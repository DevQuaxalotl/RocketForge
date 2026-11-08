#include <gtest/gtest.h>

#include "Rewards/AdaptiveWeightConfig.h"

TEST(AdaptiveWeightConfig, ImplicitFromSingleSettingsSharedAcrossSizes) {
	RLGC::AdaptiveWeightSettings s = RLGC::AdaptiveWeightSettings::TargetReward(0.02f, 1.f, 10.f);
	RocketForge::AdaptiveWeightConfig cfg = s;   // implicit conversion

	ASSERT_FALSE(cfg.usePerTeamSize);
	for (int size = 1; size <= 3; size++) {
		RLGC::AdaptiveWeightSettings r = cfg.Resolve(size);
		ASSERT_TRUE(r.enabled);
		ASSERT_EQ(r.normalizedTargetPerStep, 0.02f);
		ASSERT_EQ(r.minWeight, 1.f);
		ASSERT_EQ(r.maxWeight, 10.f);
	}
}

TEST(AdaptiveWeightConfig, TargetRewardVariesTargetSharesClamps) {
	RocketForge::AdaptiveWeightConfig cfg =
		RocketForge::AdaptiveWeightConfig::TargetReward({ 0.01f, 0.02f, 0.03f }, 0.5f, 20.f);

	ASSERT_TRUE(cfg.usePerTeamSize);

	RLGC::AdaptiveWeightSettings r1 = cfg.Resolve(1);
	RLGC::AdaptiveWeightSettings r2 = cfg.Resolve(2);
	RLGC::AdaptiveWeightSettings r3 = cfg.Resolve(3);

	ASSERT_EQ(r1.normalizedTargetPerStep, 0.01f);
	ASSERT_EQ(r2.normalizedTargetPerStep, 0.02f);
	ASSERT_EQ(r3.normalizedTargetPerStep, 0.03f);

	ASSERT_EQ(r1.minWeight, 0.5f);
	ASSERT_EQ(r3.minWeight, 0.5f);
	ASSERT_EQ(r1.maxWeight, 20.f);
	ASSERT_EQ(r3.maxWeight, 20.f);
	ASSERT_EQ(r1.easeStartRatio, r3.easeStartRatio);
	ASSERT_EQ(r1.emaHalfLifeSteps, r3.emaHalfLifeSteps);
	ASSERT_TRUE(r1.enabled && r2.enabled && r3.enabled);
}

TEST(AdaptiveWeightConfig, FullPerSizeIncludingDisabledSlot) {
	RLGC::AdaptiveWeightSettings a = RLGC::AdaptiveWeightSettings::TargetReward(0.02f, 1.f, 10.f);
	RLGC::AdaptiveWeightSettings b = RLGC::AdaptiveWeightSettings::TargetReward(0.05f, 2.f, 15.f);
	RocketForge::AdaptiveWeightConfig cfg = { a, b, RLGC::AdaptiveWeightSettings{} }; // 3v3 disabled

	ASSERT_TRUE(cfg.Resolve(1).enabled);
	ASSERT_EQ(cfg.Resolve(1).normalizedTargetPerStep, 0.02f);
	ASSERT_TRUE(cfg.Resolve(2).enabled);
	ASSERT_EQ(cfg.Resolve(2).maxWeight, 15.f);
	ASSERT_FALSE(cfg.Resolve(3).enabled);
}

TEST(AdaptiveWeightConfig, AnyEnabledAndDefaultDisabled) {
	RocketForge::AdaptiveWeightConfig def;                 // default-constructed
	ASSERT_FALSE(def.AnyEnabled());
	ASSERT_FALSE(def.Resolve(2).enabled);

	RocketForge::AdaptiveWeightConfig shared = RLGC::AdaptiveWeightSettings::TargetReward(0.02f, 1.f, 10.f);
	ASSERT_TRUE(shared.AnyEnabled());

	RocketForge::AdaptiveWeightConfig oneOn = {
		RLGC::AdaptiveWeightSettings{},
		RLGC::AdaptiveWeightSettings{},
		RLGC::AdaptiveWeightSettings::TargetReward(0.02f, 1.f, 10.f)
	};
	ASSERT_TRUE(oneOn.AnyEnabled());
}

TEST(AdaptiveWeightConfig, TargetPunishmentSetsPunishmentAllSizes) {
	RocketForge::AdaptiveWeightConfig cfg =
		RocketForge::AdaptiveWeightConfig::TargetPunishment({ 0.01f, 0.02f, 0.03f }, 0.5f, 20.f);

	ASSERT_TRUE(cfg.usePerTeamSize);
	for (int i = 0; i < RocketForge::AdaptiveWeightConfig::TEAM_SIZE_COUNT; i++) {
		ASSERT_TRUE(cfg.perTeamSize[i].enabled);
		ASSERT_TRUE(cfg.perTeamSize[i].punishment);
	}

	// TargetReward must leave punishment false on every size.
	RocketForge::AdaptiveWeightConfig rewardCfg =
		RocketForge::AdaptiveWeightConfig::TargetReward({ 0.01f, 0.02f, 0.03f }, 0.5f, 20.f);
	for (int i = 0; i < RocketForge::AdaptiveWeightConfig::TEAM_SIZE_COUNT; i++)
		ASSERT_FALSE(rewardCfg.perTeamSize[i].punishment);
}

TEST(AdaptiveWeightConfig, TargetRewardRequiresExactlyThree) {
	bool threw = false;
	try {
		RocketForge::AdaptiveWeightConfig::TargetReward({ 0.01f, 0.02f }, 1.f, 10.f);
	} catch (const std::exception&) { threw = true; }
	ASSERT_TRUE(threw);
}
