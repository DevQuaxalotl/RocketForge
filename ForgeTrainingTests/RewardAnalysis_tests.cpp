#include <gtest/gtest.h>

#include "TestSupport.h"

#include <RewardAnalysis/RewardAnalysis.h>
#include <RewardAnalysis/RewardAnalysisJson.h>

#include <cmath>
#include <vector>

using namespace RocketForge;

TEST(RewardAnalysis, AdaptiveRate_NoSamplesIsUndefined) {
    struct Case {
        const char* label;
        std::vector<std::vector<double>> sums;
        std::vector<std::vector<long long>> counts;
    };

    const Case cases[] = {
        { "no arenas", {}, {} },
        { "players without samples", { { 0.0, 0.0 } }, { { 0, 0 } } },
    };

    for (const Case& testCase : cases) {
        SCOPED_TRACE(testCase.label);
        AdaptiveRateResult r = ComputeAdaptiveGameplayRate(testCase.sums, testCase.counts);
        ASSERT_EQ(r.sampleCount, 0LL);
        ASSERT_EQ(r.avgPerStep, 0.f);
        ASSERT_EQ(r.maxPerStep, 0.f);
    }
}

TEST(RewardAnalysis, AdaptiveRate_AverageOverAllPlayerSteps) {
    // Arena 0: p0 sum=4 over 2 steps (rate 2), p1 sum=2 over 2 steps (rate 1).
    // Average = (4+2) / (2+2) = 1.5. Max = max(2,1) = 2 (single arena).
    std::vector<std::vector<double>>    sums   = { { 4.0, 2.0 } };
    std::vector<std::vector<long long>> counts = { { 2, 2 } };
    AdaptiveRateResult r = ComputeAdaptiveGameplayRate(sums, counts);
    ASSERT_EQ(r.sampleCount, 4LL);
    ASSERT_TRUE(std::fabs(r.avgPerStep - 1.5f) < 1e-6f);
    ASSERT_TRUE(std::fabs(r.maxPerStep - 2.0f) < 1e-6f);
}

TEST(RewardAnalysis, AdaptiveRate_MaxMeansOverArenasWithSteps) {
    // Arena 0: top rate 2. Arena 1: only p0 logged, sum=6 over 2 = rate 3.
    // Arena 2: no steps -> excluded from Max mean.
    // Max = mean(2, 3) = 2.5. Average = (4+2+6) / (2+2+2) = 12/6 = 2.
    std::vector<std::vector<double>>    sums   = { { 4.0, 2.0 }, { 6.0, 0.0 }, { 0.0, 0.0 } };
    std::vector<std::vector<long long>> counts = { { 2, 2 },     { 2, 0 },     { 0, 0 }     };
    AdaptiveRateResult r = ComputeAdaptiveGameplayRate(sums, counts);
    ASSERT_EQ(r.sampleCount, 6LL);
    ASSERT_TRUE(std::fabs(r.avgPerStep - 2.0f) < 1e-6f);
    ASSERT_TRUE(std::fabs(r.maxPerStep - 2.5f) < 1e-6f);
}

TEST(RewardAnalysis, AdaptiveWindow_ShorterWindowRaisesMaxForSpikySignal) {
    // One arena, two players. The reward "spikes" on a different player each window:
    //   window A: p0 hot (10 over 10 -> rate 1), p1 cold (0 over 10 -> rate 0) -> arena max 1
    //   window B: p0 cold (0 over 10 -> rate 0), p1 hot (10 over 10 -> rate 1) -> arena max 1
    // Windowed Max = mean(1, 1) = 1.0.
    AdaptiveRateAccumulator acc;
    FoldAdaptiveWindow({ { 10.0, 0.0 } }, { { 10, 10 } }, acc);
    FoldAdaptiveWindow({ { 0.0, 10.0 } }, { { 10, 10 } }, acc);
    AdaptiveRateResult windowed = FinalizeWindowedAdaptiveRate(acc);

    // The SAME data collapsed into one whole-run window smooths each player to rate 0.5,
    // so the whole-run Max is only 0.5 -- the under-report the windowing fix removes.
    AdaptiveRateResult wholeRun =
        ComputeAdaptiveGameplayRate({ { 10.0, 10.0 } }, { { 20, 20 } });

    ASSERT_TRUE(std::fabs(windowed.maxPerStep - 1.0f) < 1e-6f);
    ASSERT_TRUE(std::fabs(wholeRun.maxPerStep - 0.5f) < 1e-6f);
    ASSERT_TRUE(windowed.maxPerStep > wholeRun.maxPerStep);

    // Average and sample count are window-invariant: identical either way.
    ASSERT_TRUE(std::fabs(windowed.avgPerStep - 0.5f) < 1e-6f);
    ASSERT_TRUE(std::fabs(windowed.avgPerStep - wholeRun.avgPerStep) < 1e-6f);
    ASSERT_EQ(windowed.sampleCount, 40LL);
    ASSERT_EQ(windowed.sampleCount, wholeRun.sampleCount);
}

TEST(RewardAnalysis, AdaptiveWindow_EmptyWindowsDoNotContribute) {
    // A window in which no arena logged a step must not add a maxWindowCount entry
    // (otherwise it would dilute the Max mean with a spurious 0).
    AdaptiveRateAccumulator acc;
    FoldAdaptiveWindow({ { 0.0, 0.0 } }, { { 0, 0 } }, acc);   // empty window: skipped
    FoldAdaptiveWindow({ { 6.0, 0.0 } }, { { 2, 0 } }, acc);   // p0 rate 3, p1 no steps
    AdaptiveRateResult r = FinalizeWindowedAdaptiveRate(acc);

    ASSERT_EQ(r.sampleCount, 2LL);
    ASSERT_TRUE(std::fabs(r.avgPerStep - 3.0f) < 1e-6f);   // 6 / 2
    ASSERT_TRUE(std::fabs(r.maxPerStep - 3.0f) < 1e-6f);   // single contributing window
}

TEST(RewardAnalysis, FoldAdaptiveRate_AccumulatesCollectorsOnly) {
    float avgWS = 0.f, maxWS = 0.f, collectedWeight = 0.f;
    long long sampleCount = 0;
    // Collector: weight 0.5, avg 0.02, max 0.05, 100 samples.
    FoldAdaptiveRate(0.02f, 0.05f, 100, 0.5f, avgWS, maxWS, collectedWeight, sampleCount);
    // Non-collector (uneven matchup): 0 samples -> must NOT touch avg/max/weight, but sampleCount sums.
    FoldAdaptiveRate(0.0f, 0.0f, 0, 0.5f, avgWS, maxWS, collectedWeight, sampleCount);
    ASSERT_TRUE(std::fabs(avgWS - 0.01f) < 1e-6f);          // 0.02 * 0.5 only
    ASSERT_TRUE(std::fabs(maxWS - 0.025f) < 1e-6f);         // 0.05 * 0.5 only
    ASSERT_TRUE(std::fabs(collectedWeight - 0.5f) < 1e-6f); // only the collector's weight
    ASSERT_EQ(sampleCount, 100LL);                          // still sums the 0
}

TEST(RewardAnalysis, FinalizeAdaptiveRate_RenormalizesOverCollectors) {
    // Continuation of the scenario above: one collector (w 0.5), one non-collector (w 0.5).
    // Renormalized avg = 0.01 / 0.5 = 0.02 (the collector's true rate), NOT diluted to 0.01.
    float dstAvg = 0.f, dstMax = 0.f;
    FinalizeAdaptiveRate(0.01f, 0.025f, 0.5f, dstAvg, dstMax);
    ASSERT_TRUE(std::fabs(dstAvg - 0.02f) < 1e-6f);
    ASSERT_TRUE(std::fabs(dstMax - 0.05f) < 1e-6f);
}

TEST(RewardAnalysis, FinalizeAdaptiveRate_ZeroCollectedIsUndefined) {
    // No team size collected -> avg/max stay 0 (GUI renders N/A alongside sampleCount 0).
    float dstAvg = 0.f, dstMax = 0.f;
    FinalizeAdaptiveRate(0.f, 0.f, 0.f, dstAvg, dstMax);
    ASSERT_EQ(dstAvg, 0.f);
    ASSERT_EQ(dstMax, 0.f);
}

TEST(RewardAnalysis, FoldFinalize_TwoEqualCollectorsAverages) {
    // Two collectors, equal weight 0.5 each -> renormalized weighted average.
    float avgWS = 0.f, maxWS = 0.f, cw = 0.f;
    long long sc = 0;
    FoldAdaptiveRate(0.02f, 0.05f, 100, 0.5f, avgWS, maxWS, cw, sc);
    FoldAdaptiveRate(0.04f, 0.09f, 300, 0.5f, avgWS, maxWS, cw, sc);
    float dstAvg = 0.f, dstMax = 0.f;
    FinalizeAdaptiveRate(avgWS, maxWS, cw, dstAvg, dstMax);
    ASSERT_TRUE(std::fabs(dstAvg - 0.03f) < 1e-6f); // (0.02+0.04)/2
    ASSERT_TRUE(std::fabs(dstMax - 0.07f) < 1e-6f); // (0.05+0.09)/2
    ASSERT_EQ(sc, 400LL);
}

TEST(RewardAnalysis, Json_RoundTripsNewFields) {
    RewardImpactEntry in;
    in.name = "AirDribbleReward";
    in.normalGameplayAvgPerStep = 0.0123f;
    in.normalGameplayMaxPerStep = 0.0456f;
    in.normalGameplaySampleCount = 987654LL;

    nlohmann::json j = in;              // to_json
    RewardImpactEntry out = j.get<RewardImpactEntry>();  // from_json

    ASSERT_TRUE(std::fabs(out.normalGameplayAvgPerStep - 0.0123f) < 1e-7f);
    ASSERT_TRUE(std::fabs(out.normalGameplayMaxPerStep - 0.0456f) < 1e-7f);
    ASSERT_EQ(out.normalGameplaySampleCount, 987654LL);
}

TEST(RewardAnalysis, Json_DefaultsNewFieldsWhenAbsent) {
    // A v1-style entry with no new keys must load with zeroed new fields.
    nlohmann::json j = { { "name", "Old" }, { "category", "CONTINUOUS" } };
    RewardImpactEntry out = j.get<RewardImpactEntry>();
    ASSERT_EQ(out.normalGameplaySampleCount, 0LL);
    ASSERT_EQ(out.normalGameplayAvgPerStep, 0.f);
    ASSERT_EQ(out.normalGameplayMaxPerStep, 0.f);
}

TEST(RewardAnalysis, CacheFileName_IsPerStageAndSanitized) {
    const std::string a = RewardAnalysisCacheFileName("Air Dribble / v2");
    const std::string b = RewardAnalysisCacheFileName("Kickoff");
    ASSERT_TRUE(a != b);
    // Must not contain path separators after sanitizing.
    ASSERT_TRUE(a.find('/') == std::string::npos);
    ASSERT_TRUE(a.find('\\') == std::string::npos);
    // Recognisable prefix/suffix.
    ASSERT_TRUE(a.rfind("reward_analysis_", 0) == 0);
    ASSERT_TRUE(a.size() > std::string("reward_analysis_.json").size());
}

TEST(RewardAnalysis, SaveLoad_PerStageFilesCoexist) {
    namespace fs = std::filesystem;
    fs::path dir = RocketForgeTests::TestTempDir();
    fs::create_directories(dir);

    RewardAnalysisResult ra; ra.stageName = "StageA"; ra.timestepsPerTeamSize = 111;
    RewardAnalysisResult rb; rb.stageName = "StageB"; rb.timestepsPerTeamSize = 222;
    // Need at least one team size so a load is considered valid content.
    ra.teamSizes.push_back({}); rb.teamSizes.push_back({});

    std::string err;
    ASSERT_TRUE(SaveRewardAnalysisResult(ra, dir, "StageA", &err));
    ASSERT_TRUE(SaveRewardAnalysisResult(rb, dir, "StageB", &err));

    auto la = TryLoadRewardAnalysisResult(dir, "StageA");
    auto lb = TryLoadRewardAnalysisResult(dir, "StageB");
    ASSERT_TRUE(la.has_value());
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(la->timestepsPerTeamSize, 111);
    ASSERT_EQ(lb->timestepsPerTeamSize, 222);

    fs::remove_all(dir);
}
