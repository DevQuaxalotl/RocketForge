#include <gtest/gtest.h>

#include "Rewards/RewardWeight.h"

TEST(RewardWeight, SharedConstructionAndResolve) {
	RocketForge::RewardWeight w(4.f);
	ASSERT_FALSE(w.usePerTeamSize);
	ASSERT_EQ(w.Resolve(1), 4.f);
	ASSERT_EQ(w.Resolve(3), 4.f);
	ASSERT_TRUE(w.HasAnyWeight());

	RocketForge::RewardWeight iw(7);   // int overload
	ASSERT_EQ(iw.Resolve(2), 7.f);
}

TEST(RewardWeight, PerTeamSizeConstructionAndResolve) {
	RocketForge::RewardWeight w({ 50.f, 45.f, 40.f });
	ASSERT_TRUE(w.usePerTeamSize);
	ASSERT_EQ(w.Resolve(1), 50.f);
	ASSERT_EQ(w.Resolve(2), 45.f);
	ASSERT_EQ(w.Resolve(3), 40.f);
	ASSERT_TRUE(w.HasAnyWeight());
}

TEST(RewardWeight, AssignmentOperators) {
	RocketForge::RewardWeight w;
	w = 3.f;
	ASSERT_EQ(w.Resolve(1), 3.f);
	w = { 1.f, 2.f, 3.f };
	ASSERT_EQ(w.Resolve(2), 2.f);
	w = 9;                 // int overload
	ASSERT_EQ(w.Resolve(3), 9.f);
}

TEST(RewardWeight, HasAnyWeightFalseWhenAllZero) {
	RocketForge::RewardWeight shared(0.f);
	ASSERT_FALSE(shared.HasAnyWeight());

	RocketForge::RewardWeight perSize({ 0.f, 0.f, 0.f });
	ASSERT_FALSE(perSize.HasAnyWeight());

	RocketForge::RewardWeight oneNonZero({ 0.f, 0.f, 0.1f });
	ASSERT_TRUE(oneNonZero.HasAnyWeight());
}
