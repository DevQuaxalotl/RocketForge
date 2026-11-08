#include <gtest/gtest.h>

#include <Utils/SteadyStateSps.h>

#include <string>

using namespace RocketForge;

namespace {
	struct Iteration {
		uint64_t timesteps;
		double collectionSeconds;
		double consumptionSeconds;
	};
}

TEST(SteadyStateSps, ExcludesWarmupIterations)
{
	const Iteration iterations[] = { { 100, 1, 1 }, { 100, 1, 1 }, { 1000, 2, 3 }, { 2000, 3, 2 } };

	SteadyStateSps sps = {};
	sps.warmupIterations = 2;
	for (const Iteration& iteration : iterations)
		sps.AddIteration(iteration.timesteps, iteration.collectionSeconds, iteration.consumptionSeconds);

	ASSERT_EQ(sps.MeasuredIterations(), 2);
	ASSERT_NEAR(sps.OverallSps(), 300.0, 1e-9);
	ASSERT_NEAR(sps.CollectionSps(), 600.0, 1e-9);
	ASSERT_NEAR(sps.ConsumptionSps(), 600.0, 1e-9);
	ASSERT_EQ(
		sps.Summary(),
		"Steady state (2 iterations after 2 warmup): overall 300 SPS, collection 600 SPS, consumption 600 SPS, collection 50% of time");
}

TEST(SteadyStateSps, ReportsZeroWhenNothingIsMeasured)
{
	struct Case {
		const char* label;
		int warmupIterations;
		int iterationsAdded;
	};
	const Case cases[] = {
		{ "warmup exceeds iterations", 5, 3 },
		{ "warmup equals iterations", 3, 3 },
		{ "no iterations", 0, 0 },
	};

	for (const Case& testCase : cases) {
		SCOPED_TRACE(testCase.label);

		SteadyStateSps sps = {};
		sps.warmupIterations = testCase.warmupIterations;
		for (int i = 0; i < testCase.iterationsAdded; i++)
			sps.AddIteration(1000, 1, 1);

		ASSERT_EQ(sps.MeasuredIterations(), 0);
		ASSERT_EQ(sps.OverallSps(), 0.0);
		ASSERT_EQ(sps.CollectionSps(), 0.0);
		ASSERT_EQ(sps.ConsumptionSps(), 0.0);
		ASSERT_NE(sps.Summary().find("no measured iterations"), std::string::npos);
	}
}
