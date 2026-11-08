// ForgeTraining/src/RewardAnalysis/RewardAnalysis.h
#pragma once

#include <RLGymCPP/Rewards/Reward.h>

#include <GigaLearnInfer/JsonArchive.h>

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace RocketForge
{

/**
 * Per-reward adaptive-matching gameplay rate: mean unweighted |raw| per player-step over
 * normal-gameplay (Kickoff/Random) episodes — the adaptive controller's normalizedTargetPerStep.
 * sampleCount == 0 => none observed, so avg/max are undefined.
 **/
struct AdaptiveRateResult
{
    /** Σ|raw| / Σcount over all normal-gameplay player-steps. **/
    float avgPerStep = 0.f;

    /** Mean over arenas (with steps) of max_player(sum/count). **/
    float maxPerStep = 0.f;

    long long sampleCount = 0;
};

/**
 * Running accumulator for the windowed Average/Max aggregations. Fold one window at a time with
 * FoldAdaptiveWindow, then read the result via FinalizeWindowedAdaptiveRate.
 *
 * Windowing exists to match AdaptiveRewardWeightController::UpdateAndApply, which recomputes each
 * arena's top-player rate once per PPO iteration (a short window that spans multiple episodes,
 * ~tsPerItr/(numGames*numPlayers) steps) and averages those maxima over iterations. Because max is
 * convex, E[max over a short window] >= max[over the whole run], so a single whole-run Max
 * under-reports the signal the controller regulates. Average is window-invariant, so it is folded
 * the same way and unaffected.
 **/
struct AdaptiveRateAccumulator
{
    /** Σ|raw| over every folded player-step (window-invariant => Average). **/
    double    totalSum       = 0.0;
    /** Σ player-steps folded (== sampleCount). **/
    long long totalCount     = 0;
    /** Σ, over each (window, arena-with-steps), of that window's top-player rate. **/
    double    maxRateSum     = 0.0;
    /** Number of (window, arena) pairs that contributed a max (i.e. had >=1 step). **/
    long long maxWindowCount = 0;
};

/**
 * Folds one window's per-(arena,player) accumulators into acc. windowSum/windowCount are indexed
 * [arena][player]. Each arena with >=1 player-step contributes its top player's sum/count to the
 * running Max; every player-step contributes to the running Average.
 **/
void FoldAdaptiveWindow(
    const std::vector<std::vector<double>>& windowSum,
    const std::vector<std::vector<long long>>& windowCount,
    AdaptiveRateAccumulator& acc);

/**
 * Reads a windowed accumulator: avg = totalSum/totalCount; max = maxRateSum/maxWindowCount;
 * sampleCount = totalCount. avg/max stay 0 when their denominator is 0 (undefined).
 **/
AdaptiveRateResult FinalizeWindowedAdaptiveRate(const AdaptiveRateAccumulator& acc);

/**
 * Single-window convenience over per-(arena,player) normal-gameplay accumulators (the whole run as
 * one window), mirroring AdaptiveRewardWeightController for one iteration: Average = Σsum/Σcount;
 * Max = mean over arenas of the top player's sum/count. Empty input => sampleCount 0. Equivalent to
 * one FoldAdaptiveWindow + FinalizeWindowedAdaptiveRate; prefer the windowed API to match training.
 **/
AdaptiveRateResult ComputeAdaptiveGameplayRate(
    const std::vector<std::vector<double>>& perPlayerSum,
    const std::vector<std::vector<long long>>& perPlayerCount);

/**
 * Folds one team size's rate into the running "All (average)" sums. Only collecting team sizes
 * (sampleCount > 0 — fair matchups) contribute weight, so uneven matchups don't dilute the
 * average; FinalizeAdaptiveRate renormalizes over that weight. sampleCount always sums.
 **/
void FoldAdaptiveRate(
    float srcAvg, float srcMax, long long srcSampleCount, float teamWeight,
    float& dstAvgWeightedSum, float& dstMaxWeightedSum,
    float& dstCollectedWeight, long long& dstSampleCount);

/**
 * Divides the weighted sums by the collecting weight; leaves avg/max at 0 when nothing collected
 * (rendered as N/A alongside sampleCount 0).
 **/
void FinalizeAdaptiveRate(
    float avgWeightedSum, float maxWeightedSum, float collectedWeight,
    float& dstAvg, float& dstMax);

/** Persisted JSON layout version. Bumped when RewardAnalysisResult or its members
    change in a way that breaks backward compatibility with existing files. **/
constexpr int REWARD_ANALYSIS_SCHEMA_VERSION = 2;

struct RewardImpactEntry
{
    std::string name;
    RLGC::RewardCategory category = RLGC::RewardCategory::CONTINUOUS;
    float weight = 0.f;

    /** Mean across episodes of the per-player-mean discounted return G0.
        G0 = sum(gamma^t * r_t) for t = 0..T-1, computed per-player then averaged. **/
    float meanDiscountedReturn = 0.f;

    /** Std across episodes of the per-player-mean G0. **/
    float stdDiscountedReturn = 0.f;

    /** Mean across episodes of the per-player-mean |G0|. Gross learning signal magnitude.
        Non-zero for ZeroSum rewards even when meanDiscountedReturn == 0,
        because individual player signals cancel in the signed mean. **/
    float meanAbsDiscountedReturn = 0.f;

    /** Std across episodes of the per-player-mean |G0|. Note: this is the std of an
        already-averaged statistic, so it understates per-player variability by ~sqrt(N). **/
    float stdAbsDiscountedReturn = 0.f;

    /** Magnitude share of this reward across all rewards: mean(|G0_i|) / sum_j(mean(|G0_j|)).
        Always in [0, 1] and sums to ~1.0 across rewards. The sign of the underlying
        contribution is reflected by sign(meanDiscountedReturn). **/
    float impactFraction = 0.f;

    /** CONTINUOUS only (zero for EVENT). Per-step weighted reward averaged across all steps. **/
    float meanWeightedPerStep = 0.f;

    /** EVENT only (zero for CONTINUOUS). Average number of fires per episode. **/
    float meanFiresPerEpisode = 0.f;

    /** EVENT only (zero for CONTINUOUS). Magnitude per fire (always non-negative;
        ZeroSum rewards still report their gross magnitude here even when the signed
        mean ~ 0). Conditional on at least one fire per episode. **/
    float meanWeightedPerFire = 0.f;

    /** EVENT only. Total number of fires observed across the run. Useful sample-size
        signal — small values mean the reported per-fire stats are noisy. **/
    int totalFires = 0;

    /**
     * Unweighted |raw| per player-step over normal-gameplay (Kickoff/Random) episodes, Average
     * aggregation — the adaptive controller's normalizedTargetPerStep. Valid iff sampleCount > 0.
     **/
    float normalGameplayAvgPerStep = 0.f;

    /**
     * As above, Max aggregation: per-arena top player, averaged over training-sized windows
     * (matches AdaptiveRewardWeightController's per-iteration Max; see AdaptiveRateAccumulator).
     **/
    float normalGameplayMaxPerStep = 0.f;

    /** Normal-gameplay player-steps sampled; 0 => adaptive rate undefined (pure-Scenario or uneven-only stage). **/
    long long normalGameplaySampleCount = 0;

    /** Serializes this entry into a "rewards" array entry of reward_analysis.json. **/
    void Serialize(GGL::JsonArchive& ar);
};

struct TeamSizeImpactResult
{
    /** e.g. "1v1", "2v2" **/
    std::string label;
    int   teamSizeA = 0;
    int   teamSizeB = 0;

    /** Normalized training weight for this team size (from stage config). **/
    float weight = 1.f;

    int   episodesRun               = 0;
    /** Steps per episode. **/
    float meanEpisodeLength         = 0.f;
    float meanTotalDiscountedReturn = 0.f;
    float stdTotalDiscountedReturn  = 0.f;

    std::vector<RewardImpactEntry> rewards;

    /** Serializes this entry into a "team_sizes" array entry of reward_analysis.json. **/
    void Serialize(GGL::JsonArchive& ar);
};

struct RewardAnalysisResult
{
    /** Schema version of the loaded file. Set automatically on save. **/
    int schemaVersion = REWARD_ANALYSIS_SCHEMA_VERSION;

    std::string agentName;
    std::string stageName;
    std::string checkpointPath;

    /** Target timesteps used for collection. **/
    int   timestepsPerTeamSize = 0;
    float gaeGamma = 0.99f;

    std::vector<TeamSizeImpactResult> teamSizes;

    /** Pre-computed weighted average across all team sizes, weighted by each team size's
        configured training weight. Std fields are intentionally zero — standard deviations
        do not pool linearly across groups. Refer to the per-team-size rows for variability. **/
    TeamSizeImpactResult average;

    /** Serializes this result to/from the root of reward_analysis.json. **/
    void Serialize(GGL::JsonArchive& ar);
};

class TrainingStage;

/**
 * Runs numTimesteps of experience per supported team size and returns the breakdown.
 * progressCallback(stepsCollected, totalTarget, teamSizeLabel) is called after each episode.
 * If numTimesteps == 0, reads the target from stage->GetTimestepsPerIteration().
 *
 * isCancelled (optional): polled periodically inside the step loop. When it returns true
 * the analysis exits as soon as practical and may return a result with episodesRun == 0
 * for any team size that hadn't completed an episode. Callers should check the same
 * cancel flag before consuming the result.
 *
 * On a non-cancelled run that completes zero episodes for any team size (e.g. numTimesteps
 * is too small to fit a single episode), this function calls RG_ERR_CLOSE.
 **/
RewardAnalysisResult RunRewardAnalysis(
    TrainingStage* stage,
    const std::filesystem::path& modelsPath,
    int numTimesteps,
    const std::function<void(int, int, const std::string&)>& progressCallback,
    const std::function<bool()>& isCancelled = {});

/** Cache filename for a given stage: "reward_analysis_<sanitized stage>.json". **/
std::string RewardAnalysisCacheFileName(const std::string& stageName);

/**
 * Saves a result to savePath/reward_analysis_<sanitized stageName>.json, so a segment
 * can hold cached results for multiple stages side by side.
 * savePath should be the version folder, not a numbered checkpoint subfolder.
 * Returns false and writes an error message on failure.
 **/
bool SaveRewardAnalysisResult(const RewardAnalysisResult& result,
                              const std::filesystem::path& savePath,
                              const std::string& stageName,
                              std::string* error = nullptr);

/**
 * Loads a previously saved result from savePath/reward_analysis_<sanitized stageName>.json.
 * savePath should be the version folder, not a numbered checkpoint subfolder.
 * Returns nullopt if the file does not exist, cannot be parsed, or is from a
 * future schema version.
 **/
std::optional<RewardAnalysisResult> TryLoadRewardAnalysisResult(
    const std::filesystem::path& savePath,
    const std::string& stageName);

} // namespace RocketForge
