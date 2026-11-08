#include <gtest/gtest.h>

#include "TestSupport.h"
#include "TrainingStage.h"
#include "Rewards/AdaptiveWeightConfig.h"

#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/EnvSet/AdaptiveRewardWeightController.h>

#include <cmath>
#include <optional>

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <RLGymCPP/StateSetters/StateSetter.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>


TEST(AdaptiveWeight, TargetRewardFactorySetsFields) {
	RLGC::AdaptiveWeightSettings s = RLGC::AdaptiveWeightSettings::TargetReward(0.02f, 2.f, 60.f);

	ASSERT_TRUE(s.enabled);
	ASSERT_EQ(s.normalizedTargetPerStep, 0.02f);
	ASSERT_EQ(s.minWeight, 2.f);
	ASSERT_EQ(s.maxWeight, 60.f);

	// Ease-shape defaults left in place.
	ASSERT_EQ(s.easeStartRatio, 0.5f);
	ASSERT_EQ(s.easeEndRatio, 1.5f);
	ASSERT_EQ(s.easeExponent, 2.f);
	ASSERT_EQ(s.emaHalfLifeSteps, 2000000.f);
	ASSERT_TRUE(s.mode == RLGC::AdaptiveMode::Latch);
	ASSERT_EQ(s.masteredRatio, 2.f);
	ASSERT_EQ(s.relearnRatio, 0.5f);
	ASSERT_EQ(s.sustainSteps, 1000000.f);
	ASSERT_EQ(s.weightHalfLifeSteps, 500000.f);
	ASSERT_EQ(s.regulateDeadband, 0.1f);
}

TEST(AdaptiveWeight, TargetRewardAcceptsMaxAggregation) {
	RLGC::AdaptiveWeightSettings s =
		RLGC::AdaptiveWeightSettings::TargetReward(0.02f, 2.f, 60.f, RLGC::AdaptiveAggregation::Max);

	ASSERT_TRUE(s.enabled);
	ASSERT_TRUE(s.aggregation == RLGC::AdaptiveAggregation::Max);
}

TEST(AdaptiveWeight, TargetContinuousHelper) {
	// 2 seconds of full reward per 30s = 2/30 per step; tickSkip cancels.
	float target = RLGC::AdaptiveTargetContinuous(2.f, 30.f);
	ASSERT_TRUE(target > 0.0666f && target < 0.0667f);
}

TEST(AdaptiveWeight, TargetEventHelper) {
	// tickSkip 8 -> 15 steps/sec. 5 fires per 30 sec = 5 / (30*15) = 0.01111.. per step.
	float target = RLGC::AdaptiveTargetEvent(5.f, 30.f, 8);
	ASSERT_TRUE(target > 0.0111f && target < 0.0112f);
}

TEST(AdaptiveWeight, TargetFromWeightedImpactHelper) {
	// A weighted per-step of 0.4 at weight 20 normalizes to 0.4 / 20 = 0.02 per step.
	float target = RLGC::AdaptiveTargetFromWeightedImpact(0.4f, 20.f);
	ASSERT_TRUE(target > 0.0199f && target < 0.0201f);
}

namespace {
	/** Minimal reward whose only meaningful behavior is a fixed name. */
	struct NamedReward : RLGC::Reward {
		std::string name;
		explicit NamedReward(std::string name) : name(std::move(name)) {}
		std::string GetName() override { return name; }
	};

	const RLGC::AdaptivePersistEntry* FindEntry(
		const std::vector<RLGC::AdaptivePersistEntry>& entries, const std::string& name, int playerCount) {
		for (const RLGC::AdaptivePersistEntry& e : entries)
			if (e.name == name && e.playerCount == playerCount)
				return &e;
		return nullptr;
	}

	// These controller tests pin Regulate's curve and the aggregation/borrow plumbing,
	// so they run Regulate with the deadband and weight lag off.
	RLGC::AdaptiveWeightSettings AsRegulateCurve(RLGC::AdaptiveWeightSettings settings) {
		settings.mode = RLGC::AdaptiveMode::Regulate;
		settings.regulateDeadband = 0.f;
		settings.weightHalfLifeSteps = 0.f;
		return settings;
	}

	// Runs one iteration: `perStepAbs` accumulated once per step for `steps` steps in arena 0, reward 0.
	void RunIteration(RLGC::AdaptiveRewardWeightController& c,
		std::vector<std::vector<RLGC::WeightedReward>>& rewards, float perStepAbs, int steps) {
		for (int s = 0; s < steps; s++)
			c.Accumulate(0, 0, 0, perStepAbs);
		c.UpdateAndApply(rewards, (double)steps);
	}

	// Runs one iteration across two arenas of the SAME player count, feeding an explicit per-player
	// absolute reward each step. playerAbs0/playerAbs1 are the per-player values for arena 0 / arena 1.
	void RunTwoArenaIteration(RLGC::AdaptiveRewardWeightController& c,
		std::vector<std::vector<RLGC::WeightedReward>>& rewards,
		const std::vector<float>& playerAbs0, const std::vector<float>& playerAbs1, int steps) {

		for (int step = 0; step < steps; step++) {
			for (int p = 0; p < (int)playerAbs0.size(); p++)
				c.Accumulate(0, 0, p, playerAbs0[p]);

			for (int p = 0; p < (int)playerAbs1.size(); p++)
				c.Accumulate(1, 0, p, playerAbs1[p]);
		}

		c.UpdateAndApply(rewards, (double)steps * (playerAbs0.size() + playerAbs1.size()));
	}
}

TEST(AdaptiveController, EmaSeedsThenFolds) {
	NamedReward reward("Boost");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1e9f, /*min*/ 0.01f, /*max*/ 1e9f));
	s.emaHalfLifeSteps = 1.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	RunIteration(controller, rewards, 10.f, 1); // first sample seeds: ema = 10
	RunIteration(controller, rewards, 0.f, 1);  // one half-life toward 0: ema = 5

	const RLGC::AdaptivePersistEntry* e = FindEntry(controller.ExportState(), "Boost", 2);
	ASSERT_TRUE(e != nullptr);
	ASSERT_TRUE(e->rawRateEma > 4.99f && e->rawRateEma < 5.01f);
}

TEST(AdaptiveController, AverageMergesAcrossPlayers) {
	// One arena, two players. Average aggregation must divide the summed |raw| by the total
	// player-steps: player 0 contributes 10/step, player 1 contributes 2/step -> rawRate = 6.
	// A tiny half-life makes the EMA settle on rawRate within a single 50-step iteration.
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1e9f, /*min*/ 0.01f, /*max*/ 1e9f));
	s.emaHalfLifeSteps = 1.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	for (int step = 0; step < 50; step++) {
		controller.Accumulate(0, 0, 0, 10.0f);
		controller.Accumulate(0, 0, 1, 2.0f);
	}
	controller.UpdateAndApply(rewards, 100.0);

	std::vector<RLGC::AdaptivePersistEntry> exported = controller.ExportState();
	const RLGC::AdaptivePersistEntry* e = FindEntry(exported, "Flip", 2);
	ASSERT_TRUE(e != nullptr);
	ASSERT_TRUE(e->rawRateEma > 5.99f && e->rawRateEma < 6.01f);
}

TEST(AdaptiveController, OverProducedWeightFallsToMin) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 0.5f, /*min*/ 1.f, /*max*/ 100.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 20.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// rawRate ~1 vs target 0.5 -> ratio ~2 >= easeEndRatio(1.5) -> weight rests at min.
	for (int it = 0; it < 12; it++)
		RunIteration(controller, rewards, 1.f, 50);

	ASSERT_TRUE(rewards[0][0].weight > 0.99f && rewards[0][0].weight < 1.01f);
}

TEST(AdaptiveController, UnderProducedWeightRisesToMax) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 5.f, /*min*/ 0.1f, /*max*/ 40.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 20.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Very small production, target large -> measured << target -> ramp up to max and hold.
	for (int it = 0; it < 40; it++)
		RunIteration(controller, rewards, 0.001f, 50);

	ASSERT_TRUE(rewards[0][0].weight > 39.99f && rewards[0][0].weight < 40.01f);
}

TEST(AdaptiveController, UndiscoveredRampsToMaxAndStops) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 30.f));
	s.emaHalfLifeSteps = 2.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	for (int it = 0; it < 60; it++)
		RunIteration(controller, rewards, 0.f, 50); // zero production every step

	ASSERT_TRUE(rewards[0][0].weight > 29.99f && rewards[0][0].weight < 30.01f);
}

TEST(AdaptiveController, PenaltyKeepsNegativeSign) {
	// Non-punishment path: a negative-weight reward (punishment=false) still adapts in the REWARD
	// direction (over-production lowers magnitude). Sign preservation only; see AdaptiveController_Punishment* for inverted direction.
	NamedReward reward("Bump");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 0.2f, /*min*/ 1.f, /*max*/ 50.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, -20.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	for (int it = 0; it < 10; it++)
		RunIteration(controller, rewards, 1.f, 50); // over-produced -> magnitude drops toward min

	ASSERT_TRUE(rewards[0][0].weight < 0.f);           // still a penalty
	ASSERT_TRUE(std::fabs(rewards[0][0].weight) < 20.f); // magnitude decreased
}

TEST(AdaptiveController, PunishmentAboveTargetRisesToMax) {
	NamedReward reward("Bump");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetPunishment(/*target*/ 0.5f, /*min*/ 1.f, /*max*/ 100.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 20.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// rawRate ~1 vs ceiling 0.5 -> ratio ~2 >= easeEndRatio(1.5) -> punishment magnitude rests at max.
	for (int it = 0; it < 12; it++)
		RunIteration(controller, rewards, 1.f, 50);

	ASSERT_TRUE(rewards[0][0].weight > 99.99f && rewards[0][0].weight < 100.01f);
}

TEST(AdaptiveController, PunishmentBelowTargetFallsToMin) {
	NamedReward reward("Bump");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetPunishment(/*target*/ 5.f, /*min*/ 0.1f, /*max*/ 40.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 20.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Tiny (but non-zero) production well under target -> measured << target -> ramp DOWN to min and hold.
	for (int it = 0; it < 60; it++)
		RunIteration(controller, rewards, 0.001f, 50);

	ASSERT_TRUE(rewards[0][0].weight > 0.099f && rewards[0][0].weight < 0.101f);
}

TEST(AdaptiveController, PunishmentZeroRateRelaxesToMin) {
	NamedReward reward("Bump");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetPunishment(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 30.f));
	s.emaHalfLifeSteps = 2.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Behavior never happens (rawRate ~ 0) -> punishment relaxes toward min, not up to max.
	for (int it = 0; it < 100; it++)
		RunIteration(controller, rewards, 0.f, 50);

	ASSERT_TRUE(rewards[0][0].weight > 0.099f && rewards[0][0].weight < 0.101f);
}

TEST(AdaptiveController, PunishmentKeepsNegativeSignWhileRising) {
	NamedReward reward("Bump");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetPunishment(/*target*/ 0.2f, /*min*/ 1.f, /*max*/ 50.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, -10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	for (int it = 0; it < 10; it++)
		RunIteration(controller, rewards, 1.f, 50); // over target -> magnitude rises

	ASSERT_TRUE(rewards[0][0].weight < 0.f);             // still a penalty (sign preserved)
	ASSERT_TRUE(std::fabs(rewards[0][0].weight) > 10.f); // magnitude increased past its start
}

TEST(AdaptiveController, PerTeamSizeIndependentWeights) {
	// Two arenas, different player counts, same reward name/start/settings.
	NamedReward flip1("Flip"); // arena 0 (1 player)
	NamedReward flip2("Flip"); // arena 1 (2 players)
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 5.f, /*min*/ 0.1f, /*max*/ 200.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = {
		{ { &flip1, 20.f, s } },
		{ { &flip2, 20.f, s } }
	};
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1}, {2,2} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Drive the two keys in opposite directions to prove independent keying:
	// arena 0 (1 player) over target -> weight sinks; arena 1 (2 players) under target -> weight rises.
	for (int it = 0; it < 30; it++) {
		for (int step = 0; step < 50; step++) {
			controller.Accumulate(0, 0, 0, 10.0f); // arena0 over target -> weight sinks
			controller.Accumulate(1, 0, 0, 1.0f);  // arena1 under target -> weight rises
		}
		controller.UpdateAndApply(rewards, 50.0);
	}

	ASSERT_TRUE(rewards[1][0].weight > rewards[0][0].weight);

	std::vector<RLGC::AdaptivePersistEntry> exported = controller.ExportState();
	ASSERT_EQ(exported.size(), (size_t)2);
	ASSERT_TRUE(FindEntry(exported, "Flip", 2) != nullptr);
	ASSERT_TRUE(FindEntry(exported, "Flip", 4) != nullptr);
}

TEST(AdaptiveController, MaxUsesMeanOfPerArenaMaxes) {
	// Two 3-player arenas, same key. Per-player split: arena0 = {6,0,0}, arena1 = {0,4,0}.
	// Population mean = (6+4)/6 = 1.6667. Mean of per-arena maxes = (6+4)/2 = 5.0.
	// A large emaHalfLife-relative sample count makes the EMA settle near the aggregated rate.
	NamedReward flipA("Flip");
	NamedReward flipB("Flip");
	RLGC::AdaptiveWeightSettings s =
		AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f, RLGC::AdaptiveAggregation::Max));
	s.emaHalfLifeSteps = 50.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = {
		{ { &flipA, 10.f, s } },
		{ { &flipB, 10.f, s } }
	};
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {3,3}, {3,3} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	std::vector<float> arena0 = { 6.f, 0.f, 0.f, 0.f, 0.f, 0.f };
	std::vector<float> arena1 = { 0.f, 4.f, 0.f, 0.f, 0.f, 0.f };
	for (int it = 0; it < 200; it++)
		RunTwoArenaIteration(controller, rewards, arena0, arena1, 50);

	std::vector<RLGC::AdaptivePersistEntry> exported = controller.ExportState();
	const RLGC::AdaptivePersistEntry* e = FindEntry(exported, "Flip", 6);
	ASSERT_TRUE(e != nullptr);

	// Settled near mean-of-maxes (5.0), clearly above the population mean (1.6667).
	ASSERT_TRUE(e->rawRateEma > 4.5f && e->rawRateEma < 5.5f);
}

TEST(AdaptiveController, MaxEqualsAverageForSinglePlayer) {
	// One player per arena: per-arena max == that player's rate == population mean.
	// Max and Average must produce the same measured EMA given identical input.
	NamedReward flipAvg("Flip");
	NamedReward flipMax("Flip");

	RLGC::AdaptiveWeightSettings avg =
		AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f, RLGC::AdaptiveAggregation::Average));
	avg.emaHalfLifeSteps = 10.f;
	RLGC::AdaptiveWeightSettings max =
		AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f, RLGC::AdaptiveAggregation::Max));
	max.emaHalfLifeSteps = 10.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewardsAvg = { { { &flipAvg, 10.f, avg } } };
	std::vector<std::vector<RLGC::WeightedReward>> rewardsMax = { { { &flipMax, 10.f, max } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controllerAvg;
	RLGC::AdaptiveRewardWeightController controllerMax;
	controllerAvg.Init(rewardsAvg, teamSizes);
	controllerMax.Init(rewardsMax, teamSizes);

	for (int it = 0; it < 30; it++) {
		RunIteration(controllerAvg, rewardsAvg, 0.7f, 50);
		RunIteration(controllerMax, rewardsMax, 0.7f, 50);
	}

	const RLGC::AdaptivePersistEntry* ea = FindEntry(controllerAvg.ExportState(), "Flip", 2);
	const RLGC::AdaptivePersistEntry* em = FindEntry(controllerMax.ExportState(), "Flip", 2);
	ASSERT_TRUE(ea != nullptr && em != nullptr);

	ASSERT_TRUE(std::fabs(ea->rawRateEma - em->rawRateEma) < 1e-4f);
	ASSERT_TRUE(std::fabs(rewardsAvg[0][0].weight - rewardsMax[0][0].weight) < 1e-4f);
}

TEST(AdaptiveController, ReportEntryRatioOnTarget) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f));
	s.emaHalfLifeSteps = 3.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 5.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Feed a raw rate equal to the target; the smoothed measured converges to target -> ratio ~1.
	std::vector<RLGC::AdaptiveRewardWeightController::ReportEntry> last;
	for (int it = 0; it < 80; it++) {
		for (int step = 0; step < 50; step++)
			controller.Accumulate(0, 0, 0, 1.0f); // rawRate ~1 == target
		last = controller.UpdateAndApply(rewards, 50.0);
	}

	ASSERT_EQ(last.size(), (size_t)1);
	ASSERT_EQ(last[0].name, std::string("Flip"));
	ASSERT_EQ(last[0].playerCount, 2);
	ASSERT_TRUE(last[0].ratio > 0.9f && last[0].ratio < 1.1f);
}

TEST(AdaptiveController, WeightPassesThroughAnchorAtTarget) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.01f, /*max*/ 100.f));
	s.emaHalfLifeSteps = 3.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Feed a raw rate equal to target -> smoothed ratio ~1 -> weight settles at the configured anchor (10).
	for (int it = 0; it < 80; it++)
		RunIteration(controller, rewards, 1.f, 50);

	ASSERT_TRUE(rewards[0][0].weight > 9.5f && rewards[0][0].weight < 10.5f);
}

TEST(AdaptiveController, WeightReversesWithRate) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 1.f, /*max*/ 100.f));
	s.emaHalfLifeSteps = 2.f; // settles fast so the smoothed rate tracks the fed value

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 20.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Over target -> weight sinks toward min.
	for (int it = 0; it < 40; it++)
		RunIteration(controller, rewards, 2.f, 50);
	float low = rewards[0][0].weight;
	ASSERT_TRUE(low < 2.f);

	// Rate falls back below target -> weight climbs back up the same curve toward max.
	for (int it = 0; it < 40; it++)
		RunIteration(controller, rewards, 0.f, 50);
	float high = rewards[0][0].weight;

	ASSERT_TRUE(high > 99.f);
	ASSERT_TRUE(high > low);
}

TEST(AdaptiveController, ExportImportRoundTrip) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 2.f, /*min*/ 0.1f, /*max*/ 100.f));
	s.emaHalfLifeSteps = 5.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 8.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController source;
	source.Init(rewards, teamSizes);
	for (int it = 0; it < 25; it++)
		RunIteration(source, rewards, 0.6f, 50);
	std::vector<RLGC::AdaptivePersistEntry> exported = source.ExportState();

	NamedReward reward2("Flip");
	std::vector<std::vector<RLGC::WeightedReward>> rewards2 = { { { &reward2, 8.f, s } } };
	RLGC::AdaptiveRewardWeightController restored;
	restored.Init(rewards2, teamSizes);
	restored.ImportState(exported, rewards2);

	ASSERT_TRUE(rewards2[0][0].weight > rewards[0][0].weight - 1e-3f
		&& rewards2[0][0].weight < rewards[0][0].weight + 1e-3f);

	std::vector<RLGC::AdaptivePersistEntry> restoredState = restored.ExportState();
	const RLGC::AdaptivePersistEntry* re = FindEntry(restoredState, "Flip", 2);
	ASSERT_TRUE(re != nullptr);
	ASSERT_TRUE(re->rawRateEma > 0.f);
}

TEST(AdaptiveController, ImportRestoresClampedWeightAndIgnoresUnknownKeys) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 2.f, /*min*/ 0.1f, /*max*/ 100.f));
	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 8.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	std::vector<RLGC::AdaptivePersistEntry> saved = {
		{ "OldMechanic", 3, 12.f, 0.9f }, // unknown name/count -> ignored
		{ "Flip", 2, 999.f, 2.0f }         // restored, clamped into [min, max]
	};
	controller.ImportState(saved, rewards);

	ASSERT_EQ(controller.ExportState().size(), (size_t)1);
	ASSERT_EQ(rewards[0][0].weight, 100.f);
}

TEST(AdaptiveController, UnevenArenaBorrowsLargerTeamKey) {
	// Arena 0 = 2v2 (fair, owns key total 4). Arena 1 = 1v2 (uneven, larger team 2 -> borrows total 4).
	NamedReward fair("Flip");
	NamedReward uneven("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = {
		{ { &fair, 20.f, s } },
		{ { &uneven, 20.f, s } }
	};
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {2,2}, {1,2} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Only arena 0 (fair) collects; drive it under target so its weight rises.
	for (int it = 0; it < 40; it++) {
		for (int step = 0; step < 50; step++)
			controller.Accumulate(0, 0, 0, 0.0f); // fair owner collects zero -> ramps to max
		controller.UpdateAndApply(rewards, 50.0);
	}

	// Exactly one key exists (the 2v2 total-4 key); the uneven arena borrowed it.
	std::vector<RLGC::AdaptivePersistEntry> exported = controller.ExportState();
	ASSERT_EQ(exported.size(), (size_t)1);
	ASSERT_TRUE(FindEntry(exported, "Flip", 4) != nullptr);

	// The uneven arena received the borrowed (risen) weight, identical to the fair arena's.
	ASSERT_TRUE(rewards[1][0].weight > 99.f);
	ASSERT_TRUE(std::fabs(rewards[0][0].weight - rewards[1][0].weight) < 1e-4f);
}

TEST(AdaptiveController, UnevenArenaDoesNotCollect) {
	NamedReward fair("Flip");
	NamedReward uneven("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f));
	s.emaHalfLifeSteps = 4.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = {
		{ { &fair, 20.f, s } },
		{ { &uneven, 20.f, s } }
	};
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {2,2}, {1,2} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	ASSERT_TRUE(controller.IsCollecting(0, 0));   // fair owner
	ASSERT_FALSE(controller.IsCollecting(1, 0));  // uneven borrower
	ASSERT_TRUE(controller.IsEnabled(1, 0));      // borrower still receives an applied weight
}

TEST(AdaptiveController, BorrowResolvesRegardlessOfArenaOrder) {
	// Uneven arena appears FIRST; the only fair (2v2) arena appears later. Borrow must still resolve.
	NamedReward uneven("Flip");
	NamedReward fair("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f));

	std::vector<std::vector<RLGC::WeightedReward>> rewards = {
		{ { &uneven, 20.f, s } },  // arena 0: 1v2 (uneven)
		{ { &fair, 20.f, s } }     // arena 1: 2v2 (fair)
	};
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,2}, {2,2} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	ASSERT_TRUE(controller.IsEnabled(0, 0));       // uneven arena mapped to the (later) 2v2 key
	ASSERT_FALSE(controller.IsCollecting(0, 0));   // but does not collect
	ASSERT_TRUE(controller.IsCollecting(1, 0));    // fair arena owns
}

TEST(AdaptiveController, BorrowFallsBackToNearestBelow) {
	// Run has 1v1 (total 2) only; a 1v2 (target total 4) has no exact key -> falls back to total 2.
	NamedReward fair("Flip");
	NamedReward uneven("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f));

	std::vector<std::vector<RLGC::WeightedReward>> rewards = {
		{ { &fair, 20.f, s } },   // arena 0: 1v1 -> key total 2
		{ { &uneven, 20.f, s } }  // arena 1: 1v2 -> target total 4, falls back to total 2
	};
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1}, {1,2} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	std::vector<RLGC::AdaptivePersistEntry> exported = controller.ExportState();
	ASSERT_EQ(exported.size(), (size_t)1);
	ASSERT_TRUE(FindEntry(exported, "Flip", 2) != nullptr); // only the 1v1 key exists
	ASSERT_TRUE(controller.IsEnabled(1, 0));                 // 1v2 borrowed it
}

TEST(AdaptiveController, NoEvenKeyLeavesStaticWeight) {
	// Only an uneven matchup exists -> no key created -> weight stays at its configured value.
	NamedReward uneven("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1.f, /*min*/ 0.1f, /*max*/ 100.f));

	std::vector<std::vector<RLGC::WeightedReward>> rewards = {
		{ { &uneven, 20.f, s } }  // 1v2 only
	};
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,2} };

	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	ASSERT_EQ(controller.ExportState().size(), (size_t)0);
	ASSERT_FALSE(controller.IsEnabled(0, 0));
	ASSERT_EQ(rewards[0][0].weight, 20.f); // unchanged from configured
}

TEST(AdaptiveController, FindEntryForArena_FairMatchupUsesOwnKey) {
	std::vector<RLGC::AdaptivePersistEntry> entries = {
		{ "Flip", 2, 11.f, 0.f },
		{ "Flip", 4, 22.f, 0.f },
		{ "Flip", 6, 33.f, 0.f }
	};

	const RLGC::AdaptivePersistEntry* found = RLGC::AdaptiveRewardWeightController::FindEntryForArena(entries, "Flip", 2, 2);
	ASSERT_TRUE(found != nullptr);
	ASSERT_EQ(found->weight, 22.f);

	// A fair matchup never borrows: no exact key -> nothing.
	ASSERT_TRUE(RLGC::AdaptiveRewardWeightController::FindEntryForArena(entries, "Flip", 4, 4) == nullptr);
}

TEST(AdaptiveController, FindEntryForArena_UnevenBorrowsLargerTeamKey) {
	std::vector<RLGC::AdaptivePersistEntry> entries = {
		{ "Flip", 2, 11.f, 0.f },
		{ "Flip", 4, 22.f, 0.f },
		{ "Flip", 6, 33.f, 0.f }
	};

	// 1v2 -> larger team 2 -> even key total 4.
	const RLGC::AdaptivePersistEntry* found = RLGC::AdaptiveRewardWeightController::FindEntryForArena(entries, "Flip", 1, 2);
	ASSERT_TRUE(found != nullptr);
	ASSERT_EQ(found->weight, 22.f);

	// 3v1 -> larger team 3 -> even key total 6; team order is irrelevant.
	found = RLGC::AdaptiveRewardWeightController::FindEntryForArena(entries, "Flip", 3, 1);
	ASSERT_TRUE(found != nullptr);
	ASSERT_EQ(found->weight, 33.f);
}

TEST(AdaptiveController, FindEntryForArena_UnevenFallsBackLikeInit) {
	std::vector<RLGC::AdaptivePersistEntry> entries = {
		{ "Flip", 2, 11.f, 0.f },
		{ "Flip", 6, 33.f, 0.f }
	};

	// 1v2 wants total 4: no such key -> nearest below (2).
	const RLGC::AdaptivePersistEntry* found = RLGC::AdaptiveRewardWeightController::FindEntryForArena(entries, "Flip", 1, 2);
	ASSERT_TRUE(found != nullptr);
	ASSERT_EQ(found->weight, 11.f);

	// Nothing at or below the target -> smallest tracked key.
	std::vector<RLGC::AdaptivePersistEntry> onlyLarge = { { "Flip", 6, 33.f, 0.f } };
	found = RLGC::AdaptiveRewardWeightController::FindEntryForArena(onlyLarge, "Flip", 1, 2);
	ASSERT_TRUE(found != nullptr);
	ASSERT_EQ(found->weight, 33.f);
}

TEST(AdaptiveController, FindEntryForArena_IgnoresOtherNames) {
	std::vector<RLGC::AdaptivePersistEntry> entries = { { "Boost", 4, 22.f, 0.f } };

	ASSERT_TRUE(RLGC::AdaptiveRewardWeightController::FindEntryForArena(entries, "Flip", 2, 2) == nullptr);
	ASSERT_TRUE(RLGC::AdaptiveRewardWeightController::FindEntryForArena(entries, "Flip", 1, 2) == nullptr);
}

namespace {
	struct TinyObs : RLGC::ObsBuilder {
		RLGC::FList BuildObs(const RLGC::Player& player, const RLGC::GameState& state) override { return { 0.f }; }
	};
	struct TinyParser : RLGC::ActionParser {
		RLGC::Action ParseAction(int actionIdx, const RLGC::Player& player, const RLGC::GameState& state) override { return {}; }
		int GetActionAmount() override { return 1; }
	};
	struct KickoffSetter : RLGC::StateSetter {
		KickoffSetter() { setterType = RLGC::StateSetterType::Kickoff; }
		void ResetArena(RocketSim::Arena* arena) override { arena->ResetToRandomKickoff(); }
	};
	struct StepTerminal : RLGC::TerminalCondition {
		bool IsTerminal(const RLGC::GameState& currentState) override { return true; }
		bool IsTruncation() override { return true; }
	};
	struct ConstantReward : RLGC::Reward {
		std::string name; float value;
		ConstantReward(std::string name, float value) : name(std::move(name)), value(value) {}
		std::string GetName() override { return name; }
		RLGC::RewardCategory GetCategory() const override { return RLGC::RewardCategory::EVENT; }
		float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override { return value; }
	};

	// Runs one "iteration": several stepped frames then the serial UpdateAndApply.
	void StepEnvIteration(RLGC::EnvSet& envSet, int frames) {
		for (int f = 0; f < frames; f++) {
			envSet.Reset();
			envSet.StepFirstHalf(false);
			envSet.StepSecondHalf(RLGC::IList(envSet.state.numPlayers, 0), false);
			envSet.Sync();
		}
		envSet.adaptiveWeightController.UpdateAndApply(envSet.rewards, (double)frames * envSet.state.numPlayers);
	}
}

TEST(AdaptiveController, EnvSetOverProducedWeightFalls) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 0.5f, /*min*/ 1.f, /*max*/ 60.f));
	s.emaHalfLifeSteps = 20.f;

	RLGC::EnvSetConfig cfg;
	cfg.numArenas = 1;
	cfg.tickSkip = 8;
	cfg.actionDelay = 0;
	cfg.saveRewards = false;
	cfg.envCreateCfg = RLGC::EnvCreateConfig(1, (uint8_t)RocketSim::GameMode::SOCCAR);
	cfg.envCreateFn = [&](const int index, const RLGC::EnvCreateConfig& c) {
		RLGC::EnvCreateResult result;
		RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
		arena->AddCar(RocketSim::Team::BLUE);
		arena->AddCar(RocketSim::Team::ORANGE);
		result.arena = arena;
		result.rewards = { { new ConstantReward("Flip", 1.f), 20.f, s } };
		result.terminalConditions = { new StepTerminal() };
		result.obsBuilder = new TinyObs();
		result.actionParser = new TinyParser();
		result.stateSetter = new KickoffSetter();
		result.userInfo = nullptr;
		return result;
	};

	RLGC::EnvSet envSet(cfg);

	float startWeight = envSet.rewards[0][0].weight; // starts at max (60): undiscovered at init
	for (int it = 0; it < 8; it++)
		StepEnvIteration(envSet, 30);

	// rawRate ~1 -> measured (normalized) ~1 >> target 0.5 -> weight ramps down toward min.
	ASSERT_TRUE(envSet.rewards[0][0].weight < startWeight);

	const RLGC::AdaptivePersistEntry* e = nullptr;
	std::vector<RLGC::AdaptivePersistEntry> exported = envSet.adaptiveWeightController.ExportState();
	for (const auto& entry : exported)
		if (entry.name == "Flip" && entry.playerCount == 2) e = &entry;
	ASSERT_TRUE(e != nullptr);
	ASSERT_TRUE(e->rawRateEma > 0.5f);
}

TEST(AdaptiveController, EnvSetUsesPreWrapperValueForZeroSum) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 1e9f, /*min*/ 0.01f, /*max*/ 1e9f));
	s.emaHalfLifeSteps = 5.f;

	RLGC::EnvSetConfig cfg;
	cfg.numArenas = 1;
	cfg.tickSkip = 8;
	cfg.actionDelay = 0;
	cfg.saveRewards = false;
	cfg.envCreateCfg = RLGC::EnvCreateConfig(1, (uint8_t)RocketSim::GameMode::SOCCAR);
	cfg.envCreateFn = [&](const int index, const RLGC::EnvCreateConfig& c) {
		RLGC::EnvCreateResult result;
		RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
		arena->AddCar(RocketSim::Team::BLUE);
		arena->AddCar(RocketSim::Team::ORANGE);
		result.arena = arena;
		// Both players earn 1; zero-sum OUTPUT ~0 but pre-wrapper magnitude is 1/player.
		result.rewards = { { new RLGC::ZeroSumReward(new ConstantReward("Demo", 1.f), 0.f), 30.f, s } };
		result.terminalConditions = { new StepTerminal() };
		result.obsBuilder = new TinyObs();
		result.actionParser = new TinyParser();
		result.stateSetter = new KickoffSetter();
		result.userInfo = nullptr;
		return result;
	};

	RLGC::EnvSet envSet(cfg);
	for (int it = 0; it < 5; it++)
		StepEnvIteration(envSet, 10);

	const RLGC::AdaptivePersistEntry* e = nullptr;
	std::vector<RLGC::AdaptivePersistEntry> exported = envSet.adaptiveWeightController.ExportState();
	for (const auto& entry : exported)
		if (entry.name == "Demo" && entry.playerCount == 2) e = &entry;
	ASSERT_TRUE(e != nullptr);
	ASSERT_TRUE(e->rawRateEma > 0.5f); // ~0 if post-wrapper had been used
}

namespace {
	struct ScenarioSetter : RLGC::StateSetter {
		ScenarioSetter() { setterType = RLGC::StateSetterType::Scenario; }
		void ResetArena(RocketSim::Arena* arena) override { arena->ResetToRandomKickoff(); }
	};
}

TEST(AdaptiveController, EnvSetScenarioEpisodesDoNotCollect) {
	ROCKETFORGE_REQUIRE_ROCKETSIM();

	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(/*target*/ 0.5f, /*min*/ 1.f, /*max*/ 60.f));
	s.emaHalfLifeSteps = 20.f;

	RLGC::EnvSetConfig cfg;
	cfg.numArenas = 1;
	cfg.tickSkip = 8;
	cfg.actionDelay = 0;
	cfg.saveRewards = false;
	cfg.envCreateCfg = RLGC::EnvCreateConfig(1, (uint8_t)RocketSim::GameMode::SOCCAR);
	cfg.envCreateFn = [&](const int index, const RLGC::EnvCreateConfig& c) {
		RLGC::EnvCreateResult result;
		RocketSim::Arena* arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
		arena->AddCar(RocketSim::Team::BLUE);
		arena->AddCar(RocketSim::Team::ORANGE);
		result.arena = arena;
		result.rewards = { { new ConstantReward("Flip", 1.f), 20.f, s } };
		result.terminalConditions = { new StepTerminal() };
		result.obsBuilder = new TinyObs();
		result.actionParser = new TinyParser();
		result.stateSetter = new ScenarioSetter();  // scenario -> excluded from collection
		result.userInfo = nullptr;
		return result;
	};

	RLGC::EnvSet envSet(cfg);
	float startWeight = envSet.rewards[0][0].weight; // starts at max (60)
	for (int it = 0; it < 8; it++)
		StepEnvIteration(envSet, 30);

	// Scenario episodes never accumulate -> rawRateEma stays 0 -> weight holds at its undiscovered max.
	ASSERT_TRUE(envSet.rewards[0][0].weight > startWeight - 1e-3f);

	std::vector<RLGC::AdaptivePersistEntry> exported = envSet.adaptiveWeightController.ExportState();
	const RLGC::AdaptivePersistEntry* e = nullptr;
	for (const auto& entry : exported)
		if (entry.name == "Flip" && entry.playerCount == 2) e = &entry;
	ASSERT_TRUE(e != nullptr);
	ASSERT_TRUE(e->rawRateEma < 1e-4f);
}

namespace {
	struct SpecTestStage : RocketForge::TrainingStage {
		SpecTestStage() : RocketForge::TrainingStage(nullptr) {}
		std::vector<RLGC::WeightedReward> ResolvePublic(const std::vector<RocketForge::RewardSpec>& specs, int maxTeamSize) {
			return ResolveRewardSpecs(specs, maxTeamSize);
		}
	};
}

TEST(AdaptiveWeight, RewardSpecCarriesSettingsThroughResolution) {
	RLGC::AdaptiveWeightSettings s = RLGC::AdaptiveWeightSettings::TargetReward(0.02f, 2.f, 60.f);

	std::vector<RocketForge::RewardSpec> specs;
	specs.emplace_back(new NamedReward("Flip"), 20.f, s);
	specs.emplace_back(new NamedReward("Goal"), 100.f); // no adaptive settings

	SpecTestStage stage;
	std::vector<RLGC::WeightedReward> resolved = stage.ResolvePublic(specs, 1);

	ASSERT_EQ(resolved.size(), (size_t)2);
	ASSERT_EQ(resolved[0].weight, 20.f);
	ASSERT_TRUE(resolved[0].adaptiveSettings.enabled);
	ASSERT_EQ(resolved[0].adaptiveSettings.normalizedTargetPerStep, 0.02f);
	ASSERT_FALSE(resolved[1].adaptiveSettings.enabled);
}

TEST(AdaptiveWeight, TargetPunishmentFactorySetsPunishment) {
	RLGC::AdaptiveWeightSettings p = RLGC::AdaptiveWeightSettings::TargetPunishment(0.02f, 2.f, 60.f);
	ASSERT_TRUE(p.enabled);
	ASSERT_TRUE(p.punishment);
	ASSERT_EQ(p.normalizedTargetPerStep, 0.02f);
	ASSERT_EQ(p.minWeight, 2.f);
	ASSERT_EQ(p.maxWeight, 60.f);

	// TargetReward must leave punishment false.
	RLGC::AdaptiveWeightSettings r = RLGC::AdaptiveWeightSettings::TargetReward(0.02f, 2.f, 60.f);
	ASSERT_FALSE(r.punishment);
}

TEST(AdaptiveWeight, PerTeamSizeSettingsResolvePerMaxTeamSize) {
	// ResolveRewardSpecs releases the reward out of each spec, so build fresh specs per resolution.
	auto makeSpecs = []() {
		std::vector<RocketForge::RewardSpec> specs;
		RocketForge::AdaptiveWeightConfig cfg =
			RocketForge::AdaptiveWeightConfig::TargetReward({ 0.01f, 0.02f, 0.03f }, /*min*/ 1.f, /*max*/ 50.f);
		specs.emplace_back(new NamedReward("Flip"), 20.f, cfg);
		return specs;
	};

	SpecTestStage stage;

	auto specs1 = makeSpecs();
	std::vector<RLGC::WeightedReward> r1 = stage.ResolvePublic(specs1, 1);
	auto specs2 = makeSpecs();
	std::vector<RLGC::WeightedReward> r2 = stage.ResolvePublic(specs2, 2);
	auto specs3 = makeSpecs();
	std::vector<RLGC::WeightedReward> r3 = stage.ResolvePublic(specs3, 3);

	ASSERT_EQ(r1.size(), (size_t)1);
	ASSERT_TRUE(r1[0].adaptiveSettings.enabled);
	ASSERT_EQ(r1[0].adaptiveSettings.normalizedTargetPerStep, 0.01f);
	ASSERT_EQ(r2[0].adaptiveSettings.normalizedTargetPerStep, 0.02f);
	ASSERT_EQ(r3[0].adaptiveSettings.normalizedTargetPerStep, 0.03f);

	// Shared clamps carried through regardless of size.
	ASSERT_EQ(r1[0].adaptiveSettings.minWeight, 1.f);
	ASSERT_EQ(r3[0].adaptiveSettings.maxWeight, 50.f);
}

TEST(AdaptiveController, SharedClockSmoothsKeysAlike) {
	// Arena 0 (1v1) logs 50 samples, arena 1 (2v2) 400; the shared clock gives both the same alpha.
	NamedReward flip1("Flip");
	NamedReward flip2("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(1.f, 0.1f, 100.f));
	s.emaHalfLifeSteps = 1000.f;

	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &flip1, 10.f, s } }, { { &flip2, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1}, {2,2} };
	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	auto feed = [&](float value) {
		for (int step = 0; step < 50; step++) {
			controller.Accumulate(0, 0, 0, value);
			for (int p = 0; p < 4; p++) {
				controller.Accumulate(1, 0, p, value);
				controller.Accumulate(1, 0, p, value);
			}
		}
	};

	feed(0.f);
	controller.UpdateAndApply(rewards, 1000.0); // seeds both at 0
	feed(1.f);
	controller.UpdateAndApply(rewards, 1000.0); // one half-life toward 1

	std::vector<RLGC::AdaptivePersistEntry> exported = controller.ExportState();
	ASSERT_TRUE(std::fabs(FindEntry(exported, "Flip", 2)->rawRateEma - 0.5f) < 1e-4f);
	ASSERT_TRUE(std::fabs(FindEntry(exported, "Flip", 4)->rawRateEma - 0.5f) < 1e-4f);
}

TEST(AdaptiveController, NoSampleUpdateLeavesKeyUntouched) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };
	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	RunIteration(controller, rewards, 0.4f, 50);
	const RLGC::AdaptivePersistEntry before = controller.ExportState()[0];

	controller.UpdateAndApply(rewards, 250000.0); // nothing accumulated

	const RLGC::AdaptivePersistEntry after = controller.ExportState()[0];
	ASSERT_EQ(after.rawRateEma, before.rawRateEma);
	ASSERT_EQ(after.weight, before.weight);
}

TEST(AdaptiveController, FreshLatchKeyHoldsMaxBelowTarget) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };
	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	controller.Accumulate(0, 0, 0, 0.2f);
	std::vector<RLGC::AdaptiveRewardWeightController::ReportEntry> report = controller.UpdateAndApply(rewards, 250000.0);

	ASSERT_EQ(report[0].weight, 100.f);
	ASSERT_TRUE(std::fabs(report[0].ratio - 0.2f) < 1e-6f);
	ASSERT_TRUE(*controller.ExportState()[0].phase == RLGC::AdaptivePhase::Discovering);
}

TEST(AdaptiveController, ImportInfersPhaseForPrePhaseEntries) {
	NamedReward mastered("Mastered");
	NamedReward settled("Settled");
	NamedReward discovering("Discovering");
	RLGC::AdaptiveWeightSettings s = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
	std::vector<std::vector<RLGC::WeightedReward>> rewards = { {
		{ &mastered, 10.f, s }, { &settled, -10.f, s }, { &discovering, 10.f, s }
	} };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };
	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	// Pre-phase entries: name, player count, weight, raw rate EMA only.
	controller.ImportState({
		{ "Mastered", 2, 7.f, 2.5f },
		{ "Settled", 2, 999.f, 0.8f },
		{ "Discovering", 2, 50.f, 0.3f }
	}, rewards);

	std::vector<RLGC::AdaptivePersistEntry> exported = controller.ExportState();
	ASSERT_TRUE(*FindEntry(exported, "Mastered", 2)->phase == RLGC::AdaptivePhase::Mastered);
	ASSERT_TRUE(*FindEntry(exported, "Settled", 2)->phase == RLGC::AdaptivePhase::Settled);
	ASSERT_TRUE(*FindEntry(exported, "Discovering", 2)->phase == RLGC::AdaptivePhase::Discovering);

	// Weight is restored, not re-derived: magnitude from the file (clamped), sign from the config.
	ASSERT_EQ(rewards[0][0].weight, 7.f);
	ASSERT_EQ(rewards[0][1].weight, -100.f);
	ASSERT_EQ(rewards[0][2].weight, 50.f);
}

TEST(AdaptiveController, ImportUsesRecordedPhaseAndTimer) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };
	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	RLGC::AdaptivePersistEntry saved = { "Flip", 2, 60.f, 2.5f };
	saved.phase = RLGC::AdaptivePhase::Relearning;
	saved.pendingExitSteps = 300000.0;
	controller.ImportState({ saved }, rewards);

	const RLGC::AdaptivePersistEntry exported = controller.ExportState()[0];
	ASSERT_TRUE(*exported.phase == RLGC::AdaptivePhase::Relearning);
	ASSERT_EQ(exported.pendingExitSteps, 300000.0);
	ASSERT_EQ(exported.weight, 60.f);
}

TEST(AdaptiveController, RegulateKeyIgnoresRecordedPhase) {
	NamedReward reward("Flip");
	RLGC::AdaptiveWeightSettings s = AsRegulateCurve(RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f));
	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &reward, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {1,1} };
	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	RLGC::AdaptivePersistEntry saved = { "Flip", 2, 60.f, 1.f };
	saved.phase = RLGC::AdaptivePhase::Mastered;
	controller.ImportState({ saved }, rewards);

	ASSERT_FALSE(controller.ExportState()[0].phase.has_value());
}

TEST(AdaptiveController, BorrowerFollowsOwnerLatchWeight) {
	// Arena 0 = 2v2 owns the key; arena 1 = 1v2 borrows it.
	NamedReward fair("Flip");
	NamedReward uneven("Flip");
	RLGC::AdaptiveWeightSettings s = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
	s.emaHalfLifeSteps = 0.f;
	s.weightHalfLifeSteps = 0.f;
	std::vector<std::vector<RLGC::WeightedReward>> rewards = { { { &fair, 10.f, s } }, { { &uneven, 10.f, s } } };
	std::vector<RLGC::AdaptiveRewardWeightController::ArenaTeamSizes> teamSizes = { {2,2}, {1,2} };
	RLGC::AdaptiveRewardWeightController controller;
	controller.Init(rewards, teamSizes);

	for (int it = 0; it < 4; it++) {
		for (int p = 0; p < 4; p++)
			controller.Accumulate(0, 0, p, 1.2f);
		controller.UpdateAndApply(rewards, 250000.0);
	}

	ASSERT_EQ(controller.ExportState().size(), (size_t)1);
	ASSERT_TRUE(std::fabs(rewards[0][0].weight - 10.f) < 1e-3f);
	ASSERT_EQ(rewards[1][0].weight, rewards[0][0].weight);
}

namespace {
	RLGC::AdaptiveWeightSettings MakeScheduleSettings(float target, float minW, float maxW, bool punishment) {
		// Defaults: easeStartRatio 0.5, easeEndRatio 1.5, easeExponent 2.
		return punishment
			? RLGC::AdaptiveWeightSettings::TargetPunishment(target, minW, maxW)
			: RLGC::AdaptiveWeightSettings::TargetReward(target, minW, maxW);
	}
}

TEST(Schedule, RewardAnchorsAtThreePoints) {
	RLGC::AdaptiveWeightSettings s = MakeScheduleSettings(1.f, 1.f, 100.f, false);
	const float anchor = 20.f;

	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(0.0f, s, anchor) - 100.f) < 1e-4f);  // r=0    -> max
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(0.5f, s, anchor) - 100.f) < 1e-4f);  // r=start-> max
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(1.0f, s, anchor) - anchor) < 1e-4f); // r=1    -> anchor
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(1.5f, s, anchor) - 1.f) < 1e-4f);    // r=end  -> min
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(3.0f, s, anchor) - 1.f) < 1e-4f);    // r>end  -> min
}

TEST(Schedule, RewardMonotonicNonIncreasing) {
	RLGC::AdaptiveWeightSettings s = MakeScheduleSettings(1.f, 1.f, 100.f, false);
	const float anchor = 20.f;

	float prev = RLGC::ScheduleWeightMag(0.f, s, anchor);
	for (int i = 1; i <= 30; i++) {
		float weight = RLGC::ScheduleWeightMag(i / 15.f, s, anchor); // r sweeps 0 .. 2
		ASSERT_TRUE(weight <= prev + 1e-5f);
		prev = weight;
	}
}

TEST(Schedule, PunishmentMirrorsReward) {
	RLGC::AdaptiveWeightSettings s = MakeScheduleSettings(1.f, 1.f, 100.f, true);
	const float anchor = 20.f;

	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(0.0f, s, anchor) - 1.f) < 1e-4f);    // below -> min
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(1.0f, s, anchor) - anchor) < 1e-4f); // target-> anchor
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(2.0f, s, anchor) - 100.f) < 1e-4f);  // above -> max

	float prev = RLGC::ScheduleWeightMag(0.f, s, anchor);
	for (int i = 1; i <= 30; i++) {
		float weight = RLGC::ScheduleWeightMag(i / 15.f, s, anchor);
		ASSERT_TRUE(weight >= prev - 1e-5f);
		prev = weight;
	}
}

TEST(Schedule, AnchorClampedIntoRange) {
	RLGC::AdaptiveWeightSettings s = MakeScheduleSettings(1.f, 5.f, 50.f, false);

	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(1.f, s, 999.f) - 50.f) < 1e-4f); // above max -> clamps to max
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(1.f, s, 0.f) - 5.f) < 1e-4f);    // below min -> clamps to min
}

TEST(Schedule, ExponentControlsShape) {
	RLGC::AdaptiveWeightSettings linear = MakeScheduleSettings(1.f, 0.f, 100.f, false);
	linear.easeExponent = 1.f;
	RLGC::AdaptiveWeightSettings quad = MakeScheduleSettings(1.f, 0.f, 100.f, false);
	quad.easeExponent = 2.f;

	const float anchor = 0.f;   // below-target curve runs max -> 0
	const float rMid   = 0.75f; // u = (0.75 - 0.5) / (1 - 0.5) = 0.5

	// Linear: 100 + (0 - 100) * 0.5   = 50.  Quadratic: 100 + (0 - 100) * 0.25 = 75 (holds higher).
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(rMid, linear, anchor) - 50.f) < 1e-3f);
	ASSERT_TRUE(std::fabs(RLGC::ScheduleWeightMag(rMid, quad,   anchor) - 75.f) < 1e-3f);
}

namespace {
	struct TypedNoopSetter : RLGC::StateSetter {
		explicit TypedNoopSetter(RLGC::StateSetterType type) { setterType = type; }
		void ResetArena(RocketSim::Arena*) override {}
	};
}

TEST(CombinedState, ReportsChosenChildType) {
	// Single-child combined setters are deterministic (real usage filters 0-weight children before
	// wrapping, so this mirrors production while avoiding any RNG-boundary flake).
	RLGC::CombinedState kickoffOnly({ { new TypedNoopSetter(RLGC::StateSetterType::Kickoff), 1.f } });
	ASSERT_TRUE(kickoffOnly.setterType == RLGC::StateSetterType::Scenario); // default before any reset
	kickoffOnly.ResetArena(nullptr);
	ASSERT_TRUE(kickoffOnly.setterType == RLGC::StateSetterType::Kickoff);

	RLGC::CombinedState scenarioOnly({ { new TypedNoopSetter(RLGC::StateSetterType::Scenario), 1.f } });
	scenarioOnly.ResetArena(nullptr);
	ASSERT_TRUE(scenarioOnly.setterType == RLGC::StateSetterType::Scenario);

	RLGC::CombinedState randomOnly({ { new TypedNoopSetter(RLGC::StateSetterType::Random), 1.f } });
	randomOnly.ResetArena(nullptr);
	ASSERT_TRUE(randomOnly.setterType == RLGC::StateSetterType::Random);

	// Nested: outer's only child is an inner CombinedState (kickoff) -> outer adopts Kickoff,
	// proving the child's setterType is read AFTER its own ResetArena updated it.
	RLGC::CombinedState nested({
		{ new RLGC::CombinedState({ { new TypedNoopSetter(RLGC::StateSetterType::Kickoff), 1.f } }), 1.f },
	});
	nested.ResetArena(nullptr);
	ASSERT_TRUE(nested.setterType == RLGC::StateSetterType::Kickoff);
}
