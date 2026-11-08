#include <gtest/gtest.h>

#include <GigaLearnCPP/RecentOpponentScheduler.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

namespace {
	bool Near(float actual, float expected, float tolerance = 1e-5f)
	{
		return std::abs(actual - expected) <= tolerance;
	}
}

TEST(OpponentSampleTracker, EpisodeProbabilityFromSampleTarget)
{
	struct Case {
		const char* label;
		int normalLearnerPlayers;
		int opponentLearnerPlayers;
		float expected;
	};

	// A normal 1v1 episode contributes two learner streams, while a recent-opponent
	// episode contributes one. About 26.1% of episodes therefore yields 15% of samples.
	const Case cases[] = {
		{ "1v1", 2, 1, 0.26086956f },
		{ "equal streams", 1, 1, 0.15f },
		{ "3v3", 6, 2, 0.34615386f },
	};

	GGL::OpponentSampleTracker tracker(0.15f, 1000);
	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);
		const float probability = tracker.GetEpisodeProbability(testCase.normalLearnerPlayers, testCase.opponentLearnerPlayers);
		ASSERT_TRUE(Near(probability, testCase.expected));
	}
}

TEST(OpponentSampleTracker, FeedbackCorrectsObservedSampleShare)
{
	GGL::OpponentSampleTracker tracker(0.15f, 1000);
	const float baselineProbability = tracker.GetEpisodeProbability(2, 1);

	tracker.AddSamples(30, 100);
	ASSERT_TRUE(tracker.GetEpisodeProbability(2, 1) < baselineProbability);

	tracker.Reset();
	tracker.AddSamples(0, 100);
	ASSERT_TRUE(tracker.GetEpisodeProbability(2, 1) > baselineProbability);
}

TEST(OpponentSampleTracker, BoundsHistoricalSampleWeight)
{
	GGL::OpponentSampleTracker tracker(0.15f, 100);

	tracker.AddSamples(100, 100);
	tracker.AddSamples(0, 100);
	ASSERT_TRUE(Near(tracker.GetSampleFraction(), 0.5f));

	tracker.AddSamples(0, 100);
	ASSERT_TRUE(Near(tracker.GetSampleFraction(), 0.25f));
}

TEST(OpponentSampleTracker, SplitsCombinedTargetsWithoutChangingSampleShares)
{
	const float recentTarget = 0.15f;
	const float staticTarget = 0.10f;
	const float combinedTarget = recentTarget + staticTarget;
	const float opponentEpisodeProbability =
		GGL::OpponentSampleTracker::SampleFractionToEpisodeProbability(combinedTarget, 2, 1);
	const float recentEpisodeProbability = opponentEpisodeProbability * recentTarget / combinedTarget;
	const float staticEpisodeProbability = opponentEpisodeProbability * staticTarget / combinedTarget;
	const float normalEpisodeProbability = 1.f - opponentEpisodeProbability;
	const float expectedSamplesPerEpisode = normalEpisodeProbability * 2.f
		+ (recentEpisodeProbability + staticEpisodeProbability);

	ASSERT_TRUE(Near(recentEpisodeProbability / expectedSamplesPerEpisode, recentTarget));
	ASSERT_TRUE(Near(staticEpisodeProbability / expectedSamplesPerEpisode, staticTarget));
}

namespace {
	GGL::PolicyVersionKey Version(uint64_t timesteps)
	{
		return { "test", timesteps };
	}
}

TEST(RecentOpponentScheduler, StackStartsWithRandomSelectableVersions)
{
	bool observedCohortWithoutNewest = false;
	for (uint64_t seed = 0; seed < 100; seed++) {
		GGL::RecentOpponentStack stack(3, 100, seed);
		stack.Update({ Version(10), Version(20), Version(30), Version(40) }, {}, 0);
		ASSERT_EQ(stack.GetSelectable().size(), (size_t)3);
		ASSERT_FALSE(stack.GetDraining().has_value());
		observedCohortWithoutNewest = observedCohortWithoutNewest
			|| std::find(stack.GetSelectable().begin(), stack.GetSelectable().end(), Version(40))
				== stack.GetSelectable().end();
	}
	ASSERT_TRUE(observedCohortWithoutNewest);
}

TEST(RecentOpponentScheduler, StackWaitsForDrainBeforeNextRotation)
{
	const std::vector<GGL::PolicyVersionKey> versions = {
		Version(10), Version(20), Version(30), Version(40), Version(50)
	};
	GGL::RecentOpponentStack stack(3, 100, 5);
	stack.Update(versions, {}, 0);
	std::map<GGL::PolicyVersionKey, int> active;
	for (const GGL::PolicyVersionKey& version : stack.GetSelectable())
		active[version] = 1;
	stack.Update(versions, active, 100);

	ASSERT_TRUE(stack.GetDraining().has_value());
	const GGL::PolicyVersionKey draining = stack.GetDraining().value();
	ASSERT_TRUE(active.contains(draining));
	ASSERT_TRUE(std::find(stack.GetSelectable().begin(), stack.GetSelectable().end(), draining)
		== stack.GetSelectable().end());
	ASSERT_EQ(stack.GetRotationCount(), (uint64_t)1);

	stack.Update(versions, { { draining, 1 } }, 200);
	ASSERT_EQ(stack.GetDraining().value(), draining);
	ASSERT_EQ(stack.GetRotationCount(), (uint64_t)1);
	ASSERT_EQ(stack.GetPendingRotationCount(), (uint64_t)1);

	stack.Update(versions, {}, 200);
	ASSERT_EQ(stack.GetRotationCount(), (uint64_t)2);
}

TEST(RecentOpponentScheduler, StackUsesTimestepRotationBoundaries)
{
	const std::vector<GGL::PolicyVersionKey> versions = {
		Version(10), Version(20), Version(30), Version(40)
	};
	GGL::RecentOpponentStack stack(2, 100, 7);
	stack.Update(versions, {}, 0);
	stack.Update(versions, {}, 99);
	ASSERT_EQ(stack.GetRotationCount(), (uint64_t)0);
	stack.Update(versions, {}, 100);
	ASSERT_EQ(stack.GetRotationCount(), (uint64_t)1);
	ASSERT_EQ(stack.GetNextRotationTimestep(), (uint64_t)200);
}

TEST(RecentOpponentScheduler, StackSelectsEpisodesRandomly)
{
	GGL::RecentOpponentStack stack(3, 100, 11);
	stack.Update({ Version(10), Version(20), Version(30) }, {}, 0);
	std::set<GGL::PolicyVersionKey> selected;
	for (int i = 0; i < 100; i++)
		selected.insert(stack.SelectForEpisode().value());
	ASSERT_TRUE(selected.size() > 1);
}

TEST(StaticOpponentScheduler, NativeCadenceOnlyInfersAtBoundary)
{
	int cadenceStep = 0;
	ASSERT_TRUE(GGL::ShouldInferStaticOpponentAction(cadenceStep));

	cadenceStep = GGL::AdvanceStaticOpponentCadence(cadenceStep, 2);
	ASSERT_FALSE(GGL::ShouldInferStaticOpponentAction(cadenceStep));

	cadenceStep = GGL::AdvanceStaticOpponentCadence(cadenceStep, 2);
	ASSERT_TRUE(GGL::ShouldInferStaticOpponentAction(cadenceStep));
}

TEST(StaticOpponentScheduler, CohortIsBoundedAndPreservesModeCoverage)
{
	GGL::StaticOpponentCohort cohort(3);
	cohort.Update({ { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 1 }, { 4, 3 } }, {}, 0);

	ASSERT_EQ(cohort.GetSelectable().size(), (size_t)3);
	ASSERT_TRUE(cohort.HasCompatibleVersion(1));
	ASSERT_TRUE(cohort.HasCompatibleVersion(2));
	ASSERT_TRUE(cohort.HasCompatibleVersion(3));
	ASSERT_FALSE(cohort.GetDraining().has_value());
}

TEST(StaticOpponentScheduler, WaitsForRemovedPolicyToDrain)
{
	const std::vector<GGL::StaticOpponentCandidate> candidates = {
		{ 0, 1 }, { 1, 1 }, { 2, 3 }, { 3, 1 }, { 4, 1 }
	};
	GGL::StaticOpponentCohort cohort(3, 100, 13);
	cohort.Update(candidates, {}, 0);
	std::map<int, int> active;
	for (int versionIndex : cohort.GetSelectable())
		active[versionIndex] = 1;
	cohort.Update(candidates, active, 100);

	ASSERT_TRUE(cohort.GetDraining().has_value());
	const int draining = cohort.GetDraining().value();
	ASSERT_TRUE(active.contains(draining));
	ASSERT_TRUE(std::find(cohort.GetSelectable().begin(), cohort.GetSelectable().end(), draining)
		== cohort.GetSelectable().end());
	ASSERT_EQ(cohort.GetRotationCount(), (uint64_t)1);
	ASSERT_TRUE(cohort.HasCompatibleVersion(3));

	cohort.Update(candidates, { { draining, 1 } }, 200);
	ASSERT_EQ(cohort.GetRotationCount(), (uint64_t)1);
	ASSERT_EQ(cohort.GetPendingRotationCount(), (uint64_t)1);

	cohort.Update(candidates, {}, 200);
	ASSERT_EQ(cohort.GetRotationCount(), (uint64_t)2);
}

TEST(StaticOpponentScheduler, SelectsCompatibleEpisodesRandomly)
{
	GGL::StaticOpponentCohort cohort(2, 100, 17);
	cohort.Update({ { 0, 3 }, { 1, 1 } }, {}, 0);
	std::set<int> selected;
	for (int i = 0; i < 100; i++)
		selected.insert(cohort.SelectForEpisode(1).value());
	ASSERT_EQ(selected.size(), (size_t)2);
	ASSERT_EQ(cohort.SelectForEpisode(3).value(), 0);
}

TEST(StaticOpponentScheduler, AvoidsPreviousVersionWhenAnotherIsCompatible)
{
	GGL::StaticOpponentCohort cohort(3, 100, 23);
	cohort.Update({ { 0, 3 }, { 1, 3 }, { 2, 1 } }, {}, 0);

	for (int i = 0; i < 50; i++) {
		ASSERT_TRUE(cohort.SelectForEpisode(3, 0).has_value());
		ASSERT_EQ(cohort.SelectForEpisode(3, 0).value(), 1);
	}
	ASSERT_EQ(cohort.SelectForEpisode(3, 1).value(), 0);
}

TEST(StaticOpponentScheduler, UsesTimestepRotationBoundaries)
{
	const std::vector<GGL::StaticOpponentCandidate> candidates = {
		{ 0, 3 }, { 1, 3 }, { 2, 3 }
	};
	GGL::StaticOpponentCohort cohort(2, 100, 19);
	cohort.Update(candidates, {}, 0);
	cohort.Update(candidates, {}, 99);
	ASSERT_EQ(cohort.GetRotationCount(), (uint64_t)0);
	cohort.Update(candidates, {}, 100);
	ASSERT_EQ(cohort.GetRotationCount(), (uint64_t)1);
	ASSERT_EQ(cohort.GetNextRotationTimestep(), (uint64_t)200);
}
