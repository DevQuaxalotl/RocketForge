#include <gtest/gtest.h>

#include <GigaLearnCPP/ScheduledValue.h>

#include <cmath>

namespace {
	void AssertNear(float actual, float expected, float tolerance) {
		ASSERT_TRUE(std::abs(actual - expected) <= tolerance);
	}
}

TEST(ScheduledValue, SineOscillatesAroundTarget) {
	GGL::ScheduledFloat schedule(0.02f, 0.01f, 100);

	AssertNear(schedule.ValueAt(0), 0.02f, 0.000001f);
	AssertNear(schedule.ValueAt(25), 0.03f, 0.000001f);
	AssertNear(schedule.ValueAt(50), 0.02f, 0.000001f);
	AssertNear(schedule.ValueAt(75), 0.01f, 0.000001f);
	AssertNear(schedule.ValueAt(100), 0.02f, 0.000001f);
}

TEST(ScheduledValue, MinMaxComesFromTargetAndAmplitude) {
	GGL::ScheduledFloat schedule(0.02f, 0.01f, 100);

	AssertNear(schedule.MinValue(), 0.01f, 0.000001f);
	AssertNear(schedule.MaxValue(), 0.03f, 0.000001f);
}

TEST(ScheduledValue, AssignmentFromConfiguredScheduleReadsAsTarget) {
	GGL::ScheduledFloat source;
	source.target = 0.02f;
	source.amplitude = 0.01f;
	source.period = 100;

	GGL::ScheduledFloat assigned = 0.f;
	assigned = source;

	AssertNear((float)assigned, 0.02f, 0.000001f);
	AssertNear(assigned.Update(25), 0.03f, 0.000001f);
}

TEST(ScheduledValue, GetAndResolveReadCachedUpdatedValue) {
	GGL::ScheduledFloat schedule(0.02f, 0.01f, 100);

	schedule.Update(25);

	AssertNear(schedule.Get(), 0.03f, 0.000001f);
	AssertNear((float)schedule, 0.03f, 0.000001f);
	AssertNear(GGL::Resolve(schedule), 0.03f, 0.000001f);
	AssertNear(GGL::Resolve(0.5f), 0.5f, 0.000001f);
}
