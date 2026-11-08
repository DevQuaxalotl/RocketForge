#include <gtest/gtest.h>

#include <CriticValueMath.h>

#include <cmath>
#include <vector>

using namespace ForgeDebug;

namespace {
	bool Near(float a, float b) { return std::abs(a - b) < 1e-5f; }

	ValueHeadStats MakeStats(const std::string& name, double mean, double meanSq) {
		ValueHeadStats stats;
		stats.name = name;
		stats.initialized = true;
		stats.mean = mean;
		stats.meanSq = meanSq;
		return stats;
	}
}

TEST(CriticValueMath, DenormalizeUsesMeanAndSpread) {
	// mean 2, meanSq 13 -> variance 9, sigma 3 -> 0.5 * 3 + 2
	ASSERT_TRUE(Near(DenormalizeHeadValue(0.5f, MakeStats("a", 2.0, 13.0), 0.01f), 3.5f));
}

TEST(CriticValueMath, DenormalizeUninitializedHeadIsIdentity) {
	ValueHeadStats stats;
	stats.name = "a";
	stats.mean = 50.0;
	stats.meanSq = 5000.0;
	ASSERT_TRUE(Near(DenormalizeHeadValue(0.75f, stats, 0.01f), 0.75f));
}

TEST(CriticValueMath, DenormalizeClampsSigmaToFloor) {
	// mean 1, meanSq 1 -> zero spread, floored to 0.25 -> 2 * 0.25 + 1
	ASSERT_TRUE(Near(DenormalizeHeadValue(2.f, MakeStats("a", 1.0, 1.0), 0.25f), 1.5f));
}

TEST(CriticValueMath, CombineWeightsMatchedHeadsOnly) {
	DecomposedCriticHead matched;
	matched.stats = MakeStats("touch", 0.0, 1.0);   // sigma 1, mean 0
	matched.weight = 2.f;

	DecomposedCriticHead unmatched;
	unmatched.stats = MakeStats("gone", 0.0, 1.0);
	unmatched.weight = std::nullopt;

	const CriticDebugData result = CombineDecomposedCritic({ 1.5f, 4.f }, { matched, unmatched }, 0.01f);

	ASSERT_TRUE(result.isDecomposed);
	ASSERT_EQ(result.heads.size(), size_t(2));
	ASSERT_EQ(result.heads[0].name, std::string("touch"));
	ASSERT_TRUE(Near(result.heads[0].raw, 1.5f));
	ASSERT_TRUE(Near(result.heads[0].contribution, 3.f));
	ASSERT_FALSE(result.heads[1].weight.has_value());
	ASSERT_TRUE(Near(result.heads[1].raw, 4.f));
	ASSERT_TRUE(Near(result.heads[1].contribution, 0.f));
	ASSERT_TRUE(Near(result.total, 3.f));
}

TEST(CriticValueMath, SingleCriticScalesByReturnStd) {
	const CriticDebugData result = MakeSingleCriticEstimate(0.5f, 4.f);
	ASSERT_FALSE(result.isDecomposed);
	ASSERT_TRUE(result.heads.empty());
	ASSERT_TRUE(Near(result.total, 2.f));
}

TEST(CriticValueMath, WelfordStdMatchesWelfordStat) {
	ASSERT_TRUE(Near(WelfordStd(18.0, 0), 1.f));
	ASSERT_TRUE(Near(WelfordStd(18.0, 1), 1.f));
	ASSERT_TRUE(Near(WelfordStd(0.0, 10), 1.f));
	// runningVariance 18 over count 3 -> 18 / 2 = 9 -> 3
	ASSERT_TRUE(Near(WelfordStd(18.0, 3), 3.f));
}
