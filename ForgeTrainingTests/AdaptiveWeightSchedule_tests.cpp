#include <gtest/gtest.h>

#include <RLGymCPP/EnvSet/AdaptiveWeightSchedule.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {
	constexpr double ITERATION_STEPS = 250000.0;

	bool Near(float a, float b, float tolerance = 1e-3f) {
		return std::fabs(a - b) <= tolerance;
	}

	// Regulate with the deadband, EMA smoothing and weight lag off, so a step lands exactly on the curve.
	RLGC::AdaptiveWeightSettings ExactRegulate(RLGC::AdaptiveWeightSettings settings) {
		settings.mode = RLGC::AdaptiveMode::Regulate;
		settings.regulateDeadband = 0.f;
		settings.weightHalfLifeSteps = 0.f;
		settings.emaHalfLifeSteps = 0.f;
		return settings;
	}
}

TEST(AdaptiveSchedule, GoodnessRatioOrientsPunishments) {
	RLGC::AdaptiveWeightSettings reward = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 1.f, 100.f);
	RLGC::AdaptiveWeightSettings punishment = RLGC::AdaptiveWeightSettings::TargetPunishment(1.f, 1.f, 100.f);

	ASSERT_TRUE(Near(RLGC::GoodnessRatio(2.f, reward), 2.f));
	ASSERT_TRUE(Near(RLGC::GoodnessRatio(2.f, punishment), 0.5f));
	ASSERT_TRUE(std::isinf(RLGC::GoodnessRatio(0.f, punishment)));
}

TEST(AdaptiveSchedule, FreshLatchKeyStartsDiscoveringAtMax) {
	RLGC::AdaptiveWeightSettings reward = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
	RLGC::AdaptiveWeightSettings punishment = RLGC::AdaptiveWeightSettings::TargetPunishment(1.f, 2.f, 100.f);

	for (const RLGC::AdaptiveWeightSettings& settings : { reward, punishment }) {
		RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, 10.f);
		ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Discovering);
		ASSERT_EQ(state.weightMag, 100.f);
		ASSERT_FALSE(state.rateSeeded);
	}
}

TEST(AdaptiveSchedule, FreshRegulateKeyStartsOnCurveAtZeroRate) {
	RLGC::AdaptiveWeightSettings reward = ExactRegulate(RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f));
	RLGC::AdaptiveWeightSettings punishment = ExactRegulate(RLGC::AdaptiveWeightSettings::TargetPunishment(1.f, 2.f, 100.f));

	ASSERT_EQ(RLGC::MakeFreshKeyState(reward, 10.f).weightMag, 100.f);
	ASSERT_EQ(RLGC::MakeFreshKeyState(punishment, 10.f).weightMag, 2.f);
}

TEST(AdaptiveSchedule, FirstStepSeedsEma) {
	RLGC::AdaptiveWeightSettings settings = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
	RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, 10.f);

	RLGC::StepKey(state, settings, 10.f, 0.7f, ITERATION_STEPS);

	ASSERT_TRUE(state.rateSeeded);
	ASSERT_TRUE(Near(state.rawRateEma, 0.7f, 1e-6f));
}

TEST(AdaptiveSchedule, EmaHalfLifeUsesUpdateTimesteps) {
	RLGC::AdaptiveWeightSettings settings = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
	settings.emaHalfLifeSteps = 1000.f;
	RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, 10.f);

	RLGC::StepKey(state, settings, 10.f, 0.f, 1.0);   // seeds at 0
	RLGC::StepKey(state, settings, 10.f, 1.f, 1000.0); // one half-life toward 1

	ASSERT_TRUE(Near(state.rawRateEma, 0.5f, 1e-5f));
}

TEST(AdaptiveSchedule, WeightLagHalvesDistancePerHalfLife) {
	RLGC::AdaptiveWeightSettings settings = ExactRegulate(RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f));
	settings.weightHalfLifeSteps = 1000.f;
	RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, 10.f); // starts at 100

	RLGC::StepKey(state, settings, 10.f, 1.f, 1000.0); // ratio 1 -> target is the anchor (10)

	ASSERT_TRUE(Near(state.weightMag, 55.f));
}

TEST(AdaptiveSchedule, WeightLagIsIndependentOfUpdateSize) {
	RLGC::AdaptiveWeightSettings settings = ExactRegulate(RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f));
	settings.weightHalfLifeSteps = 500000.f;

	RLGC::AdaptiveKeyState oneUpdate = RLGC::MakeFreshKeyState(settings, 10.f);
	RLGC::AdaptiveKeyState fourUpdates = oneUpdate;

	RLGC::StepKey(oneUpdate, settings, 10.f, 1.f, 1000000.0);
	for (int i = 0; i < 4; i++)
		RLGC::StepKey(fourUpdates, settings, 10.f, 1.f, 250000.0);

	ASSERT_TRUE(Near(oneUpdate.weightMag, fourUpdates.weightMag, 1e-3f));
}

TEST(AdaptiveSchedule, ZeroWeightHalfLifeJumpsToTarget) {
	RLGC::AdaptiveWeightSettings settings = ExactRegulate(RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f));
	RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, 10.f);

	RLGC::StepKey(state, settings, 10.f, 1.f, 1.0);

	ASSERT_TRUE(Near(state.weightMag, 10.f));
}

TEST(AdaptiveSchedule, RegulateDeadbandHoldsAnchor) {
	RLGC::AdaptiveWeightSettings settings = ExactRegulate(RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f));
	settings.regulateDeadband = 0.1f;

	for (float rate : { 0.92f, 1.f, 1.08f }) {
		RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, 10.f);
		RLGC::StepKey(state, settings, 10.f, rate, 1.0);
		ASSERT_TRUE(Near(state.weightMag, 10.f));
	}
}

TEST(AdaptiveSchedule, RegulateDeadbandIsContinuousAtEdge) {
	RLGC::AdaptiveWeightSettings settings = ExactRegulate(RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f));
	settings.regulateDeadband = 0.1f;

	RLGC::AdaptiveKeyState atEdge = RLGC::MakeFreshKeyState(settings, 10.f);
	RLGC::AdaptiveKeyState pastEdge = atEdge;
	RLGC::StepKey(atEdge, settings, 10.f, 1.1f, 1.0);
	RLGC::StepKey(pastEdge, settings, 10.f, 1.1001f, 1.0);

	ASSERT_TRUE(Near(atEdge.weightMag, pastEdge.weightMag, 0.01f));
}

namespace {
	constexpr float LATCH_ANCHOR = 10.f;

	// Target 1, min 2, max 100; the EMA and lag are off so each step reads the fed rate and lands on its target.
	RLGC::AdaptiveWeightSettings ExactLatch(bool punishment = false) {
		RLGC::AdaptiveWeightSettings settings = punishment
			? RLGC::AdaptiveWeightSettings::TargetPunishment(1.f, 2.f, 100.f)
			: RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);
		settings.emaHalfLifeSteps = 0.f;
		settings.weightHalfLifeSteps = 0.f;
		return settings;
	}

	void StepRepeatedly(RLGC::AdaptiveKeyState& state, const RLGC::AdaptiveWeightSettings& settings, float rate, int count) {
		for (int i = 0; i < count; i++)
			RLGC::StepKey(state, settings, LATCH_ANCHOR, rate, ITERATION_STEPS);
	}

	RLGC::AdaptiveKeyState SettledState(float rate) {
		RLGC::AdaptiveKeyState state;
		state.phase = RLGC::AdaptivePhase::Settled;
		state.weightMag = LATCH_ANCHOR;
		state.rawRateEma = rate;
		state.rateSeeded = true;
		return state;
	}
}

TEST(AdaptiveLatch, DiscoveringHoldsMaxBelowTarget) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch();
	RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, LATCH_ANCHOR);

	StepRepeatedly(state, settings, 0.9f, 20);

	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Discovering);
	ASSERT_EQ(state.weightMag, 100.f);
}

TEST(AdaptiveLatch, SettlesAtAnchorOnlyAfterSustain) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch(); // sustain 1M = four 250k updates
	RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, LATCH_ANCHOR);

	StepRepeatedly(state, settings, 1.2f, 3);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Discovering);

	StepRepeatedly(state, settings, 1.2f, 1);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Settled);
	ASSERT_TRUE(Near(state.weightMag, LATCH_ANCHOR));
}

TEST(AdaptiveLatch, BrokenConditionResetsSustain) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch();
	RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, LATCH_ANCHOR);

	StepRepeatedly(state, settings, 1.2f, 3);
	StepRepeatedly(state, settings, 0.9f, 1);
	StepRepeatedly(state, settings, 1.2f, 3);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Discovering);

	StepRepeatedly(state, settings, 1.2f, 1);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Settled);
}

TEST(AdaptiveLatch, MastersPastMasteredRatio) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch();
	RLGC::AdaptiveKeyState state = SettledState(1.f);

	StepRepeatedly(state, settings, 2.5f, 4);

	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Mastered);
	ASSERT_TRUE(Near(state.weightMag, 2.f));
}

TEST(AdaptiveLatch, MasteredReachesZeroAndRelearnClimbsBack) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch();
	settings.minWeight = 0.f;
	settings.weightHalfLifeSteps = 250000.f;
	RLGC::AdaptiveKeyState state = SettledState(1.f);

	StepRepeatedly(state, settings, 2.5f, 80);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Mastered);
	ASSERT_EQ(state.weightMag, 0.f);

	StepRepeatedly(state, settings, 0.3f, 4);  // back to Settled
	StepRepeatedly(state, settings, 0.3f, 80); // then Relearning, climbing from ~anchor toward max
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Relearning);
	ASSERT_EQ(state.weightMag, 100.f);
}

TEST(AdaptiveLatch, LagStepWithinEpsilonSnapsOntoTarget) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch();
	settings.weightHalfLifeSteps = 250000.f;
	RLGC::AdaptiveKeyState state = SettledState(1.f);
	state.weightMag = LATCH_ANCHOR + 0.005f; // inside 1e-4 of the 100 weight range

	RLGC::StepKey(state, settings, LATCH_ANCHOR, 1.f, ITERATION_STEPS);

	ASSERT_EQ(state.weightMag, LATCH_ANCHOR);
}

TEST(AdaptiveLatch, MasteredReturnsToSettledBelowTarget) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch();
	RLGC::AdaptiveKeyState state = SettledState(1.f);
	StepRepeatedly(state, settings, 2.5f, 4);

	StepRepeatedly(state, settings, 1.5f, 20); // between 1 and masteredRatio: stays Mastered
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Mastered);

	StepRepeatedly(state, settings, 0.9f, 4);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Settled);
	ASSERT_TRUE(Near(state.weightMag, LATCH_ANCHOR));
}

TEST(AdaptiveLatch, SettledRelearnsBelowRelearnRatioAndRecovers) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch();
	RLGC::AdaptiveKeyState state = SettledState(1.f);

	StepRepeatedly(state, settings, 0.6f, 20); // between relearnRatio and 1: stays Settled
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Settled);

	StepRepeatedly(state, settings, 0.4f, 4);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Relearning);
	ASSERT_EQ(state.weightMag, 100.f);

	StepRepeatedly(state, settings, 0.9f, 20); // below 1: keeps relearning
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Relearning);

	StepRepeatedly(state, settings, 1.f, 4);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Settled);
}

TEST(AdaptiveLatch, PunishmentMirrorsThroughGoodness) {
	RLGC::AdaptiveWeightSettings settings = ExactLatch(/*punishment*/ true);
	RLGC::AdaptiveKeyState state = RLGC::MakeFreshKeyState(settings, LATCH_ANCHOR);

	StepRepeatedly(state, settings, 2.f, 20); // bad rate 2x the ceiling: keeps teaching at max
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Discovering);
	ASSERT_EQ(state.weightMag, 100.f);

	StepRepeatedly(state, settings, 0.8f, 4); // g = 1.25
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Settled);
	ASSERT_TRUE(Near(state.weightMag, LATCH_ANCHOR));

	StepRepeatedly(state, settings, 0.4f, 4); // g = 2.5
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Mastered);
	ASSERT_TRUE(Near(state.weightMag, 2.f));

	StepRepeatedly(state, settings, 3.f, 4); // g = 0.33: Mastered -> Settled first
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Settled);
	StepRepeatedly(state, settings, 3.f, 4);
	ASSERT_TRUE(state.phase == RLGC::AdaptivePhase::Relearning);
}

TEST(AdaptiveLatch, NoisyRateNearEachThresholdTransitionsAtMostOnce) {
	// Default EMA (2M) and sustain (1M); uniform noise of +-0.3 around each threshold.
	RLGC::AdaptiveWeightSettings settings = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);

	for (float center : { 1.f, 2.f, 0.5f }) {
		std::mt19937 rng(7);
		std::uniform_real_distribution<float> noise(-0.3f, 0.3f);

		RLGC::AdaptiveKeyState state = SettledState(center);
		int transitions = 0;
		for (int i = 0; i < 400; i++) {
			const RLGC::AdaptivePhase before = state.phase;
			RLGC::StepKey(state, settings, LATCH_ANCHOR, std::max(0.f, center + noise(rng)), ITERATION_STEPS);
			if (state.phase != before)
				transitions++;
		}
		ASSERT_TRUE(transitions <= 1);
	}
}

TEST(AdaptiveLatch, InferPhaseFromGoodness) {
	RLGC::AdaptiveWeightSettings settings = RLGC::AdaptiveWeightSettings::TargetReward(1.f, 2.f, 100.f);

	ASSERT_TRUE(RLGC::InferLatchPhase(2.5f, settings) == RLGC::AdaptivePhase::Mastered);
	ASSERT_TRUE(RLGC::InferLatchPhase(2.0f, settings) == RLGC::AdaptivePhase::Mastered);
	ASSERT_TRUE(RLGC::InferLatchPhase(1.0f, settings) == RLGC::AdaptivePhase::Settled);
	ASSERT_TRUE(RLGC::InferLatchPhase(0.5f, settings) == RLGC::AdaptivePhase::Settled);
	ASSERT_TRUE(RLGC::InferLatchPhase(0.49f, settings) == RLGC::AdaptivePhase::Discovering);
}
