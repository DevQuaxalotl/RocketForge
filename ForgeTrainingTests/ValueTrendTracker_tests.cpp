#include <gtest/gtest.h>
#include "RewardVisualizer/ValueTrendTracker.h"

#include <cmath>

using namespace ForgeGUI;

namespace {
    bool Near(float a, float b) { return std::abs(a - b) < 1e-5f; }

    // Samples t = 0, 1/16, ..., 1 with value = slope * t.
    void RecordRamp(ValueTrendTracker& tracker, const std::string& key, float slope) {
        for (int i = 0; i <= 16; i++) {
            const double time = i / 16.0;
            tracker.Record(key, time, slope * (float)time);
        }
    }

    // Samples t = 0, 1/16, ..., 1 with value = baseline + slope * t.
    void RecordBaselineRamp(ValueTrendTracker& tracker, const std::string& key, float baseline, float slope) {
        for (int i = 0; i <= 16; i++) {
            const double time = i / 16.0;
            tracker.Record(key, time, baseline + slope * (float)time);
        }
    }
}

TEST(ValueTrendTracker, NotEnoughHistoryReadsFlat) {
    ValueTrendTracker tracker;
    tracker.Record("a", 0.0, 1.f);
    tracker.Record("a", 0.25, 5.f);

    const TrendReading reading = tracker.Get("a");
    ASSERT_EQ(reading.direction, 0);
    ASSERT_TRUE(Near(reading.delta, 0.f));
}

TEST(ValueTrendTracker, RisingSeriesReportsSmoothedDelta) {
    ValueTrendTracker tracker;
    RecordRamp(tracker, "a", 1.f);

    // recent [0.9, 1.0]: 0.9375, 1.0 -> 0.96875; past [0.4, 0.5]: 0.4375, 0.5 -> 0.46875
    const TrendReading reading = tracker.Get("a");
    ASSERT_TRUE(Near(reading.delta, 0.5f));
    ASSERT_EQ(reading.direction, 1);
}

TEST(ValueTrendTracker, FallingSeriesReportsNegativeDirection) {
    ValueTrendTracker tracker;
    RecordRamp(tracker, "a", -1.f);

    const TrendReading reading = tracker.Get("a");
    ASSERT_TRUE(Near(reading.delta, -0.5f));
    ASSERT_EQ(reading.direction, -1);
}

TEST(ValueTrendTracker, ChangeInsideDeadBandReadsFlat) {
    // value = 10 + 0.01 * t, so max |value| = 10.01 and the dead-band is 0.01 * 10.01 = 0.1001.
    // delta = (10 + 0.01 * 0.96875) - (10 + 0.01 * 0.46875) = 0.005, which is below 0.1001 -> flat.
    // Without the dead-band the same delta would read +1, so this test pins the rule.
    ValueTrendTracker tracker;
    RecordBaselineRamp(tracker, "a", 10.f, 0.01f);

    const TrendReading reading = tracker.Get("a");
    ASSERT_EQ(reading.direction, 0);
    ASSERT_TRUE(Near(reading.delta, 0.005f));
}

TEST(ValueTrendTracker, ChangeAboveDeadBandReadsRising) {
    // value = 10 + 0.25 * t, so max |value| = 10.25 and the dead-band is 0.01 * 10.25 = 0.1025.
    // delta = 0.5 * 0.25 = 0.125, which is above 0.1025 -> rising.
    ValueTrendTracker tracker;
    RecordBaselineRamp(tracker, "a", 10.f, 0.25f);

    const TrendReading reading = tracker.Get("a");
    ASSERT_EQ(reading.direction, 1);
    ASSERT_TRUE(Near(reading.delta, 0.125f));
}

TEST(ValueTrendTracker, EpisodeChangeClearsSeries) {
    ValueTrendTracker tracker;
    tracker.ObserveEpisode(0);
    RecordRamp(tracker, "a", 1.f);
    ASSERT_EQ(tracker.Get("a").direction, 1);

    tracker.ObserveEpisode(0);
    ASSERT_EQ(tracker.Get("a").direction, 1);

    tracker.ObserveEpisode(1);
    ASSERT_EQ(tracker.Get("a").direction, 0);
    ASSERT_TRUE(Near(tracker.Get("a").delta, 0.f));
}

TEST(ValueTrendTracker, IgnoresSamplesNotNewerThanLatest) {
    ValueTrendTracker tracker;
    RecordRamp(tracker, "a", 1.f);

    tracker.Record("a", 1.0, 100.f);
    tracker.Record("a", 0.5, -100.f);

    ASSERT_TRUE(Near(tracker.Get("a").delta, 0.5f));
}

TEST(ValueTrendTracker, UnknownKeyReadsFlat) {
    ValueTrendTracker tracker;
    ASSERT_EQ(tracker.Get("missing").direction, 0);
}
