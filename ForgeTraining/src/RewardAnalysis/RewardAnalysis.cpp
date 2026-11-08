#include "RewardAnalysis.h"
#include "RewardAnalysisJson.h"

#include "Storage/BuildStorage.h"
#include "TrainingStage.h"
#include <GigaLearnInfer/InferUnit.h>
#include <Inference/InferenceModelSource.h>
#include <Inference/PersistedAdaptiveWeights.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RocketSim/Includes.h>

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/StateSetters/StateSetter.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <iomanip>
#include <memory>
#include <mutex>
#include <utility>

using namespace std;
using namespace RLGC;
using namespace RocketForge;

// ============================================================================
// JSON serialization
// ============================================================================

namespace
{
    const GGL::EnumNames<RewardCategory> REWARD_ANALYSIS_CATEGORY_NAMES = { {
        { RewardCategory::EVENT,      "EVENT" },
        { RewardCategory::CONTINUOUS, "CONTINUOUS" },
    } };
}

void RocketForge::RewardImpactEntry::Serialize(GGL::JsonArchive& ar)
{
    ar.Field("name", name);
    ar.EnumField("category", category, REWARD_ANALYSIS_CATEGORY_NAMES);
    ar.Field("weight", weight);
    ar.Field("mean_discounted_return", meanDiscountedReturn);
    ar.Field("std_discounted_return", stdDiscountedReturn);
    ar.Field("mean_abs_discounted_return", meanAbsDiscountedReturn);
    ar.Field("std_abs_discounted_return", stdAbsDiscountedReturn);
    ar.Field("impact_fraction", impactFraction);
    ar.Field("mean_weighted_per_step", meanWeightedPerStep);
    ar.Field("mean_fires_per_episode", meanFiresPerEpisode);
    ar.Field("mean_weighted_per_fire", meanWeightedPerFire);
    ar.Field("total_fires", totalFires);
    ar.Field("normal_gameplay_avg_per_step", normalGameplayAvgPerStep);
    ar.Field("normal_gameplay_max_per_step", normalGameplayMaxPerStep);
    ar.Field("normal_gameplay_sample_count", normalGameplaySampleCount);
}

void RocketForge::TeamSizeImpactResult::Serialize(GGL::JsonArchive& ar)
{
    ar.Field("label", label);
    ar.Field("team_size_a", teamSizeA);
    ar.Field("team_size_b", teamSizeB);
    ar.Field("weight", weight);
    ar.Field("episodes_run", episodesRun);
    ar.Field("mean_episode_length", meanEpisodeLength);
    ar.Field("mean_total_discounted_return", meanTotalDiscountedReturn);
    ar.Field("std_total_discounted_return", stdTotalDiscountedReturn);
    ar.Field("rewards", rewards);
}

void RocketForge::RewardAnalysisResult::Serialize(GGL::JsonArchive& ar)
{
    if (ar.IsSaving())
        (*ar.Raw())["schema_version"] = REWARD_ANALYSIS_SCHEMA_VERSION;
    else
        ar.Field("schema_version", schemaVersion, 0);

    ar.Field("agent_name", agentName);
    ar.Field("stage_name", stageName);
    ar.Field("checkpoint_path", checkpointPath);
    ar.Field("timesteps_per_team_size", timestepsPerTeamSize);
    ar.Field("gae_gamma", gaeGamma, 0.99f);
    ar.Field("team_sizes", teamSizes);
    ar.Field("average", average);
}

void RocketForge::to_json(nlohmann::json& j, const RewardImpactEntry& entry)
{
    GGL::JsonSave(entry, j);
}

void RocketForge::from_json(const nlohmann::json& j, RewardImpactEntry& entry)
{
    GGL::JsonLoad(j, entry);
}

void RocketForge::to_json(nlohmann::json& j, const TeamSizeImpactResult& result)
{
    GGL::JsonSave(result, j);
}

void RocketForge::from_json(const nlohmann::json& j, TeamSizeImpactResult& result)
{
    GGL::JsonLoad(j, result);
}

void RocketForge::to_json(nlohmann::json& j, const RewardAnalysisResult& result)
{
    GGL::JsonSave(result, j);
}

void RocketForge::from_json(const nlohmann::json& j, RewardAnalysisResult& result)
{
    GGL::JsonLoad(j, result);
}

// ============================================================================
// Implementation
// ============================================================================

namespace
{

constexpr float PHYSICS_TICK_RATE_HZ      = 120.f;
constexpr int   DEFAULT_MAX_EPISODE_STEPS = 2700;   // ~225s at tickSkip=8
constexpr float ZERO_GUARD                = 1e-6f;
constexpr int   ARENAS_PER_THREAD         = 4;

string MakeLabel(int teamSizeA, int teamSizeB)
{
    return to_string(teamSizeA) + "v" + to_string(teamSizeB);
}

/**
 * One arena's in-progress episode, reduced forward in time so no per-step trajectory is stored.
 * G0 = Σ γ^t · r_t from the episode start; discount holds γ^t for the step being added.
 **/
struct EpisodeAccumulator
{
    double discount = 1.0;

    /** G0 of the player-mean weighted reward, sized [numRewards]. **/
    vector<double> g0;

    /** Per-player G0, flattened as [rewardIndex * numPlayers + playerIndex]. **/
    vector<double> g0PerPlayer;

    double totalG0 = 0.0;

    /** Undiscounted sum of the player-mean weighted reward, sized [numRewards]. **/
    vector<double> rawSum;

    /** EVENT rewards only: steps that fired, and the summed mean per-player |weighted| on those steps. **/
    vector<int>    fires;
    vector<double> firedWeighted;

    int length = 0;

    EpisodeAccumulator(size_t numRewards, int numPlayers)
        : g0(numRewards, 0.0)
        , g0PerPlayer(numRewards * numPlayers, 0.0)
        , rawSum(numRewards, 0.0)
        , fires(numRewards, 0)
        , firedWeighted(numRewards, 0.0) {}

    void Clear() {
        discount = 1.0;
        totalG0  = 0.0;
        length   = 0;
        std::fill(g0.begin(), g0.end(), 0.0);
        std::fill(g0PerPlayer.begin(), g0PerPlayer.end(), 0.0);
        std::fill(rawSum.begin(), rawSum.end(), 0.0);
        std::fill(fires.begin(), fires.end(), 0);
        std::fill(firedWeighted.begin(), firedWeighted.end(), 0.0);
    }
};

/** Per-arena owning containers for components EnvSet does NOT take ownership of.
    Wrapping in unique_ptr makes cleanup exception-safe. **/
struct ArenaResources
{
    vector<unique_ptr<ObsBuilder>>                                 obsBuilders;
    vector<unique_ptr<ActionParser>>                               actionParsers;
    vector<unique_ptr<StateSetter>>                                stateSetters;
    vector<vector<unique_ptr<TerminalCondition>>>             terminalConditions;
    vector<vector<unique_ptr<Reward>>>                        rewardOwners;
    /** Non-owning view of rewardOwners with the runtime weights, used for fast iteration. **/
    vector<vector<WeightedReward>>                                 arenaRewards;
};

/** Cross-episode accumulators. doubles to avoid catastrophic cancellation in the
    two-pass variance computation `var = E[X^2] - E[X]^2`. **/
struct AccumulatorState
{
    vector<double> sumG0;
    vector<double> sumG0sq;
    /** mean(|G0_per_player|) per episode — non-zero for ZeroSum rewards. **/
    vector<double> sumAbsG0;
    vector<double> sumAbsG0sq;
    vector<double> sumRawPerStep;
    vector<double> sumFires;
    vector<double> sumWeightedPerFire;
    vector<int>    fireEpisodes;

    // Per-(arena,player) unweighted |raw| sums/counts for the CURRENT window, sampled only on
    // normal-gameplay (Kickoff/Random) episodes; indexed [reward][arena][player]. Folded into
    // normalGameplayAccum and cleared at each window boundary by FlushAdaptiveWindow().
    vector<vector<vector<double>>>    normalGameplaySum;
    vector<vector<vector<long long>>> normalGameplayCount;

    // Running windowed adaptive-rate accumulators, one per reward. Fed by FlushAdaptiveWindow();
    // read via FinalizeWindowedAdaptiveRate() in BuildTeamSizeResult.
    vector<AdaptiveRateAccumulator>  normalGameplayAccum;

    double sumTotalG0       = 0.0;
    double sumTotalG0sq     = 0.0;
    /** Player-steps; matches the learner's timestep definition. **/
    int    totalStepsCollected = 0;
    int    totalGameSteps      = 0;
    int    episodesRun         = 0;

    AccumulatorState(size_t numRewards, int numArenas, int numPlayers)
        : sumG0(numRewards, 0.0)
        , sumG0sq(numRewards, 0.0)
        , sumAbsG0(numRewards, 0.0)
        , sumAbsG0sq(numRewards, 0.0)
        , sumRawPerStep(numRewards, 0.0)
        , sumFires(numRewards, 0.0)
        , sumWeightedPerFire(numRewards, 0.0)
        , fireEpisodes(numRewards, 0)
        , normalGameplaySum(numRewards,
              vector<vector<double>>(numArenas, vector<double>(numPlayers, 0.0)))
        , normalGameplayCount(numRewards,
              vector<vector<long long>>(numArenas, vector<long long>(numPlayers, 0)))
        , normalGameplayAccum(numRewards) {}

    // Folds the current window's per-(arena,player) samples into the running accumulators, then
    // clears the window buffers. Called at each window boundary and once at end of run.
    void FlushAdaptiveWindow() {
        for (size_t r = 0; r < normalGameplayAccum.size(); r++) {
            FoldAdaptiveWindow(normalGameplaySum[r], normalGameplayCount[r], normalGameplayAccum[r]);
            for (vector<double>&    arenaRow : normalGameplaySum[r])   std::fill(arenaRow.begin(), arenaRow.end(), 0.0);
            for (vector<long long>& arenaRow : normalGameplayCount[r]) std::fill(arenaRow.begin(), arenaRow.end(), 0LL);
        }
    }
};

/** Builds the EnvSet factory function, populating an ArenaResources record under a mutex
    (the factory is invoked concurrently, once per arena, by the EnvSet constructor). **/
EnvSetConfig BuildEnvSetConfig(
    TrainingStage* stage,
    int numArenas,
    int teamSizeA,
    int teamSizeB,
    int tickSkip,
    int actionDelay,
    ArenaResources& arenaResources,
    mutex& factoryMutex)
{
    EnvSetConfig envCfg;
    envCfg.numArenas         = numArenas;
    envCfg.tickSkip          = tickSkip;
    envCfg.actionDelay       = actionDelay;
    envCfg.saveRewards       = false;
    envCfg.envCreateCfg.mode = stage->agent->gameMode;

    arenaResources.terminalConditions.resize(numArenas);

    envCfg.envCreateFn = [stage, teamSizeA, teamSizeB, &arenaResources, &factoryMutex]
        (const int arenaIndex, const EnvCreateConfig&) -> EnvCreateResult
    {
        EnvCreateResult envCreateResult;
        RocketSim::ArenaConfig arenaConfig = {};
        arenaConfig.engine = RocketSim::EngineFromVersion(stage->agent->trainingEngineVersion);
        envCreateResult.arena = RocketSim::Arena::Create(stage->agent->gameMode, arenaConfig);

        for (int i = 0; i < teamSizeA; i++) envCreateResult.arena->AddCar(RocketSim::Team::BLUE, stage->agent->carConfig);
        for (int i = 0; i < teamSizeB; i++) envCreateResult.arena->AddCar(RocketSim::Team::ORANGE, stage->agent->carConfig);

        envCreateResult.rewards            = {};  // empty — we call arenaRewards manually
        envCreateResult.obsBuilder         = stage->agent->MakeConfiguredObsBuilder();
        envCreateResult.actionParser       = stage->agent->MakeActionParser();
        envCreateResult.userInfo           = nullptr;

        auto ownedStateSetter   = stage->GetStateSetter();
        auto ownedTerminalConds = stage->GetTerminalConditions();
        envCreateResult.stateSetter = ownedStateSetter.get();
        envCreateResult.terminalConditions.reserve(ownedTerminalConds.size());

        for (auto& tc : ownedTerminalConds)
            envCreateResult.terminalConditions.push_back(tc.get());
        {
            lock_guard<mutex> lock(factoryMutex);
            arenaResources.obsBuilders.emplace_back(envCreateResult.obsBuilder);
            arenaResources.actionParsers.emplace_back(envCreateResult.actionParser);

            // Transfer ownership of state setter and terminal conditions into arenaResources.
            arenaResources.stateSetters.emplace_back(ownedStateSetter.release());

            vector<unique_ptr<TerminalCondition>> tcOwners;
            tcOwners.reserve(ownedTerminalConds.size());
            for (auto& tc : ownedTerminalConds)
                tcOwners.emplace_back(tc.release());

            arenaResources.terminalConditions[arenaIndex] = std::move(tcOwners);
        }
        return envCreateResult;
    };

    return envCfg;
}

/** Folds one completed episode into the cross-episode accumulators. **/
void FoldEpisodeStats(const EpisodeAccumulator& episode, int numPlayers, size_t numRewards, AccumulatorState& accumulator)
{
    for (size_t rewardIndex = 0; rewardIndex < numRewards; rewardIndex++) {
        const double g0 = episode.g0[rewardIndex];
        accumulator.sumG0[rewardIndex]   += g0;
        accumulator.sumG0sq[rewardIndex] += g0 * g0;

        double meanAbsG0 = 0.0;
        for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++)
            meanAbsG0 += std::abs(episode.g0PerPlayer[rewardIndex * numPlayers + playerIndex]);
        meanAbsG0 /= numPlayers;
        accumulator.sumAbsG0[rewardIndex]   += meanAbsG0;
        accumulator.sumAbsG0sq[rewardIndex] += meanAbsG0 * meanAbsG0;

        accumulator.sumRawPerStep[rewardIndex] += episode.rawSum[rewardIndex];
        accumulator.sumFires[rewardIndex]      += episode.fires[rewardIndex];
        if (episode.fires[rewardIndex] > 0) {
            accumulator.sumWeightedPerFire[rewardIndex] += episode.firedWeighted[rewardIndex] / episode.fires[rewardIndex];
            accumulator.fireEpisodes[rewardIndex]++;
        }
    }

    accumulator.sumTotalG0   += episode.totalG0;
    accumulator.sumTotalG0sq += episode.totalG0 * episode.totalG0;
    accumulator.totalStepsCollected += episode.length * numPlayers;
    accumulator.totalGameSteps      += episode.length;
    accumulator.episodesRun++;
}

/** Builds the final TeamSizeImpactResult from accumulated stats. Returns a result with
    episodesRun == 0 and zeroed fields when no episode completed (no division performed). **/
TeamSizeImpactResult BuildTeamSizeResult(
    const AccumulatorState& accumulator,
    const vector<WeightedReward>& referenceRewards,
    int teamSizeA,
    int teamSizeB)
{
    TeamSizeImpactResult result;
    result.label       = MakeLabel(teamSizeA, teamSizeB);
    result.teamSizeA   = teamSizeA;
    result.teamSizeB   = teamSizeB;
    result.episodesRun = accumulator.episodesRun;

    const size_t numRewards = referenceRewards.size();
    result.rewards.resize(numRewards);

    if (accumulator.episodesRun == 0)
        return result;  // No episodes completed — caller decides what to do (cancel vs error).

    const double episodes  = static_cast<double>(accumulator.episodesRun);
    const double gameSteps = static_cast<double>(accumulator.totalGameSteps);

    result.meanEpisodeLength = static_cast<float>(gameSteps / episodes);

    const double meanG0 = accumulator.sumTotalG0 / episodes;
    const double varG0  = accumulator.sumTotalG0sq / episodes - meanG0 * meanG0;
    result.meanTotalDiscountedReturn = static_cast<float>(meanG0);
    result.stdTotalDiscountedReturn  = static_cast<float>(std::sqrt(std::max(0.0, varG0)));

    // First pass: per-reward means + raw stats.
    for (size_t rewardIndex = 0; rewardIndex < numRewards; rewardIndex++) {
        const Reward* reward = referenceRewards[rewardIndex].reward;
        RewardImpactEntry&  entry  = result.rewards[rewardIndex];

        entry.name     = const_cast<Reward*>(reward)->GetName();
        entry.category = reward->GetCategory();
        entry.weight   = referenceRewards[rewardIndex].weight;

        const double meanG0Reward = accumulator.sumG0[rewardIndex] / episodes;
        const double varG0Reward  = accumulator.sumG0sq[rewardIndex] / episodes - meanG0Reward * meanG0Reward;
        entry.meanDiscountedReturn = static_cast<float>(meanG0Reward);
        entry.stdDiscountedReturn  = static_cast<float>(std::sqrt(std::max(0.0, varG0Reward)));

        const double meanAbsReward = accumulator.sumAbsG0[rewardIndex] / episodes;
        const double varAbsReward  = accumulator.sumAbsG0sq[rewardIndex] / episodes - meanAbsReward * meanAbsReward;
        entry.meanAbsDiscountedReturn = static_cast<float>(meanAbsReward);
        entry.stdAbsDiscountedReturn  = static_cast<float>(std::sqrt(std::max(0.0, varAbsReward)));

        if (entry.category == RewardCategory::CONTINUOUS) {
            entry.meanWeightedPerStep = static_cast<float>(accumulator.sumRawPerStep[rewardIndex] / gameSteps);
        } else {
            entry.meanFiresPerEpisode = static_cast<float>(accumulator.sumFires[rewardIndex] / episodes);
            entry.meanWeightedPerFire = (accumulator.fireEpisodes[rewardIndex] > 0)
                ? static_cast<float>(accumulator.sumWeightedPerFire[rewardIndex] / accumulator.fireEpisodes[rewardIndex])
                : 0.f;
            entry.totalFires = static_cast<int>(accumulator.sumFires[rewardIndex]);
        }

        // Windowed accumulators were filled by FlushAdaptiveWindow() during collection; the
        // per-window buffers are already drained, so read the folded result here.
        const AdaptiveRateResult adaptiveRate =
            FinalizeWindowedAdaptiveRate(accumulator.normalGameplayAccum[rewardIndex]);
        entry.normalGameplayAvgPerStep  = adaptiveRate.avgPerStep;
        entry.normalGameplayMaxPerStep  = adaptiveRate.maxPerStep;
        entry.normalGameplaySampleCount = adaptiveRate.sampleCount;
    }

    // Second pass: magnitude-share impact fraction. Always non-negative; sums to ~1.0.
    double totalMagnitude = 0.0;
    for (const RewardImpactEntry& entry : result.rewards)
        totalMagnitude += entry.meanAbsDiscountedReturn;
    if (totalMagnitude > ZERO_GUARD) {
        for (RewardImpactEntry& entry : result.rewards)
            entry.impactFraction = static_cast<float>(entry.meanAbsDiscountedReturn / totalMagnitude);
    }

    return result;
}

/** Weighted average across team sizes using each team size's training weight.
    Standard deviations are intentionally omitted (left at zero) — combining stds
    requires the law of total variance, which needs per-group means re-derived from
    accumulators not carried here. The per-team-size rows show variability instead. **/
TeamSizeImpactResult ComputeAverage(
    const vector<TeamSizeImpactResult>& results)
{
    if (results.empty()) return {};

    float totalWeight = 0.f;
    for (const TeamSizeImpactResult& teamResult : results)
        totalWeight += teamResult.weight;
    if (totalWeight < ZERO_GUARD)
        totalWeight = static_cast<float>(results.size());

    TeamSizeImpactResult average;
    average.label     = "average";
    average.teamSizeA = 0;
    average.teamSizeB = 0;

    // Track exact float for episodesRun then round once at the end.
    float episodesRunFloat = 0.f;

    for (const TeamSizeImpactResult& teamResult : results) {
        const float normalizedWeight = teamResult.weight / totalWeight;
        episodesRunFloat                  += static_cast<float>(teamResult.episodesRun) * normalizedWeight;
        average.meanEpisodeLength         += teamResult.meanEpisodeLength             * normalizedWeight;
        average.meanTotalDiscountedReturn += teamResult.meanTotalDiscountedReturn     * normalizedWeight;
    }
    average.episodesRun = static_cast<int>(std::round(episodesRunFloat));

    const size_t numRewards = results[0].rewards.size();
    average.rewards.resize(numRewards);
    for (size_t rewardIndex = 0; rewardIndex < numRewards; rewardIndex++) {
        average.rewards[rewardIndex].name     = results[0].rewards[rewardIndex].name;
        average.rewards[rewardIndex].category = results[0].rewards[rewardIndex].category;
        average.rewards[rewardIndex].weight   = results[0].rewards[rewardIndex].weight;

        // Adaptive-rate scratch: weighted sums over ONLY collecting (fair) team sizes,
        // renormalized by FinalizeAdaptiveRate so uneven matchups don't dilute the average.
        float adaptiveAvgWeightedSum  = 0.f;
        float adaptiveMaxWeightedSum  = 0.f;
        float adaptiveCollectedWeight = 0.f;

        for (const TeamSizeImpactResult& teamResult : results) {
            RG_ASSERT(teamResult.rewards.size() == numRewards);
            const float normalizedWeight = teamResult.weight / totalWeight;
            const RewardImpactEntry& src = teamResult.rewards[rewardIndex];
            RewardImpactEntry&       dst = average.rewards[rewardIndex];
            dst.meanDiscountedReturn    += src.meanDiscountedReturn    * normalizedWeight;
            dst.meanAbsDiscountedReturn += src.meanAbsDiscountedReturn * normalizedWeight;
            dst.impactFraction          += src.impactFraction          * normalizedWeight;
            dst.meanWeightedPerStep     += src.meanWeightedPerStep     * normalizedWeight;
            dst.meanFiresPerEpisode     += src.meanFiresPerEpisode     * normalizedWeight;
            dst.meanWeightedPerFire     += src.meanWeightedPerFire     * normalizedWeight;
            dst.totalFires              += src.totalFires;

            FoldAdaptiveRate(
                src.normalGameplayAvgPerStep, src.normalGameplayMaxPerStep,
                src.normalGameplaySampleCount, teamResult.weight,
                adaptiveAvgWeightedSum, adaptiveMaxWeightedSum,
                adaptiveCollectedWeight, average.rewards[rewardIndex].normalGameplaySampleCount);
        }

        FinalizeAdaptiveRate(
            adaptiveAvgWeightedSum, adaptiveMaxWeightedSum, adaptiveCollectedWeight,
            average.rewards[rewardIndex].normalGameplayAvgPerStep,
            average.rewards[rewardIndex].normalGameplayMaxPerStep);
    }
    return average;
}

// Collects at least numTimesteps player-steps of complete episodes for one team-size config.
// Each step: inference is batched across all arenas from the EnvSet's per-arena obs rows,
// then every arena's second half (physics, state update, per-reward collection, next obs) runs in parallel on the thread pool.
// Rewards are passed empty to EnvSet so they are never called internally — we call them
// manually after each step to get a per-reward breakdown for analysis.
//
// Once numTimesteps player-steps have been taken, arenas stop starting new episodes and the in-flight ones run to completion.
// Discarding them instead would keep only the episodes that happened to finish early, biasing the stats toward short episodes.
TeamSizeImpactResult RunTeamSizeAnalysis(
    TrainingStage* stage,
    GGL::InferUnit& inferUnit,
    int teamSizeA,
    int teamSizeB,
    int numTimesteps,
    float gaeGamma,
    const function<void(int, int, bool)>& progressCallback,
    const function<bool()>& isCancelled,
    const vector<AdaptivePersistEntry>& adaptiveWeights)
{
    const int numPlayers  = teamSizeA + teamSizeB;
    const int tickSkip    = stage->agent->tickSkip;
    const int actionDelay = stage->agent->actionDelay;

    // The adaptive controller only measures fair matchups (blue == orange); uneven matchups
    // borrow a weight and never collect. Match that: measure the rate only when fair.
    const bool fairMatchup = (teamSizeA == teamSizeB);

    // Several arenas per thread keep every worker busy and give inference a batch large enough to amortize its GPU round trip.
    // numGames caps it at what training itself runs.
    const int arenaBudget = std::max(1, g_ThreadPool.GetNumThreads()) * ARENAS_PER_THREAD;
    const int trainNumGames = stage->learnerConfig.numGames;
    const int numArenas = (trainNumGames > 0) ? std::min(arenaBudget, trainNumGames) : arenaBudget;

    // Each arena needs its own reward instances so their internal state doesn't interfere.
    // GetRewards() transfers ownership — store as unique_ptr for exception-safe cleanup
    // and build a non-owning WeightedReward view alongside.
    ArenaResources arenaResources;
    arenaResources.rewardOwners.resize(numArenas);
    arenaResources.arenaRewards.resize(numArenas);
    for (int i = 0; i < numArenas; i++) {
        // GetRewards now takes a single max-team-size argument; mirror the resolution
        // the previous two-arg overload did internally (max of A and B).
        vector<WeightedReward> rewards = stage->GetConfiguredRewards(std::max(teamSizeA, teamSizeB));
        arenaResources.rewardOwners[i].reserve(rewards.size());
        arenaResources.arenaRewards[i].reserve(rewards.size());
        for (WeightedReward& weighted : rewards) {
            arenaResources.rewardOwners[i].emplace_back(weighted.reward);
            arenaResources.arenaRewards[i].push_back(weighted);  // raw pointer view, owner above
        }
    }

    // Reflect the model's effective adaptive weights so the analysis matches the weights
    // actually in force at this checkpoint.
    for (int i = 0; i < numArenas; i++)
        ApplyPersistedAdaptiveWeights(adaptiveWeights, teamSizeA, teamSizeB, arenaResources.arenaRewards[i]);

    const size_t numRewards = arenaResources.arenaRewards[0].size();
    if (numRewards == 0)
        RG_ERR_CLOSE("RunTeamSizeAnalysis: stage produced zero rewards for matchup " << MakeLabel(teamSizeA, teamSizeB));

    // Resolved once so the per-step loop doesn't dynamic_cast every reward on every step.
    vector<vector<ZeroSumReward*>> arenaZeroSumRewards(numArenas, vector<ZeroSumReward*>(numRewards, nullptr));
    for (int arenaIndex = 0; arenaIndex < numArenas; arenaIndex++) {
        for (size_t rewardIndex = 0; rewardIndex < numRewards; rewardIndex++)
            arenaZeroSumRewards[arenaIndex][rewardIndex] = dynamic_cast<ZeroSumReward*>(arenaResources.arenaRewards[arenaIndex][rewardIndex].reward);
    }

    // Target normal-gameplay sample count for the adaptive-rate top-up = the largest EMA half-life
    // (emaHalfLifeSteps) among this matchup's adaptive rewards.
    // Training's half-life runs on the iteration's collected timesteps across every matchup,
    // so a full half-life of this matchup's samples alone is a conservative bound that keeps even the longest-memory key well-sampled.
    // 0 => no adaptive rewards / no smoothing, so no top-up.
    long long adaptiveTargetSamples = 0;
    for (const WeightedReward& weighted : arenaResources.arenaRewards[0]) {
        if (!weighted.adaptiveSettings.enabled)
            continue;
        const float halfLife = weighted.adaptiveSettings.emaHalfLifeSteps;
        if (halfLife > 0.f)
            adaptiveTargetSamples = std::max(adaptiveTargetSamples,
                static_cast<long long>(std::ceil(halfLife)));
    }

    // Match the learner's truncation threshold: maxEpisodeDuration seconds * steps/second.
    int maxEpisodeSteps = static_cast<int>(
        stage->learnerConfig.ppo.maxEpisodeDuration * (PHYSICS_TICK_RATE_HZ / static_cast<float>(tickSkip)));
    if (maxEpisodeSteps <= 0)
        maxEpisodeSteps = DEFAULT_MAX_EPISODE_STEPS;

    const bool  deterministic     = stage->learnerConfig.ppo.deterministic;
    const float policyTemperature = stage->learnerConfig.ppo.policyTemperature;

    // EnvSet destructor only deletes arenas/eventTrackers/eventCallbackInfos. The factory
    // is invoked concurrently (one call per arena), so the cleanup vectors are protected
    // by a mutex — see BuildEnvSetConfig.
    mutex factoryMutex;
    EnvSetConfig envCfg = BuildEnvSetConfig(
        stage, numArenas, teamSizeA, teamSizeB, tickSkip, actionDelay,
        arenaResources, factoryMutex);

    AccumulatorState accumulator(numRewards, numArenas, numPlayers);

    // Per-arena state written only by the job stepping that arena; uint8_t rather than the bit-packed vector<bool>
    // so concurrent writes to neighbouring arenas don't race.
    vector<EpisodeAccumulator> arenaEpisodes(numArenas, EpisodeAccumulator(numRewards, numPlayers));
    vector<vector<float>> arenaRewardScratch(numArenas);
    vector<uint8_t> arenaEpisodeCompleted(numArenas, 0);
    vector<uint8_t> arenaActive(numArenas, 1);
    vector<uint8_t> arenaEpisodeIsNormalGameplay(numArenas, 0);

    {   // Scope envSet so its destructor runs before ArenaResources cleanup.
        EnvSet envSet(envCfg);
        RG_ASSERT(envSet.obsSize == inferUnit.obsSize);
        RG_ASSERT(envSet.state.numPlayers == numArenas * numPlayers);

        // EnvSet constructor calls ResetArena for all arenas. Reset our per-arena rewards
        // with the post-construction game states to align reward internal state.
        for (int arenaIndex = 0; arenaIndex < numArenas; arenaIndex++) {
            for (WeightedReward& weightedReward : arenaResources.arenaRewards[arenaIndex])
                weightedReward.reward->Reset(envSet.state.gameStates[arenaIndex]);
        }

        // Setter type of each arena's in-progress episode, captured at reset. Only Kickoff/Random
        // episodes feed the adaptive-rate accumulators — mirrors EnvSet.cpp's IsNormalGameplayEpisode.
        for (int arenaIndex = 0; arenaIndex < numArenas; arenaIndex++)
            arenaEpisodeIsNormalGameplay[arenaIndex] = IsNormalGameplayEpisode(envSet.stateSetters[arenaIndex]->setterType);

        // Match training's Max-aggregation window. AdaptiveRewardWeightController recomputes each
        // arena's top-player rate once per PPO iteration — a short window that spans several episodes
        // — and averages those maxima. One while-iteration below advances every arena exactly one
        // step, so counting iterations counts per-arena steps. Windowing is derived from the TRAINING
        // config (not this run's length), so the reported Max reproduces the signal the controller
        // regulated instead of the far-longer whole-run window (which under-reports).
        //
        // The true per-arena iteration length is tsPerItr / (total players across all training
        // arenas). We approximate that denominator as numGames*numPlayers, i.e. a homogeneous mix of
        // this matchup's size. For a symmetric 1v1/2v2/3v3 mix this is exact at the middle size and
        // within ~2x at the extremes — far better than the whole-run window either way. 0/invalid
        // config -> a single whole-run window (legacy behavior).
        const int trainTsPerItr = stage->GetTimestepsPerIteration();
        const int windowSteps = (trainTsPerItr > 0 && trainNumGames > 0 && numPlayers > 0)
            ? std::max(1, trainTsPerItr / (trainNumGames * numPlayers))
            : std::numeric_limits<int>::max();
        int stepsSinceFlush = 0;

        // Second half of one step for one arena: apply controls, step, update state, collect rewards, and either
        // build the next obs or end the episode. Touches only this arena's slots, so it runs on the thread pool.
        auto stepArena = [&](int arenaIndex, const vector<int>& actionIndices, bool recordReturnStats, bool startNewEpisodes) {
            const int playerOffset = arenaIndex * numPlayers;
            GameState& gameState = envSet.state.gameStates[arenaIndex];

            // Mirror EnvSet::StepSecondHalf's zero-delay path. StepFirstHalf is a no-op
            // when actionDelay is 0, so per-step event flags must be cleared here.
            if (actionDelay == 0) {
                envSet.state.prevGameStates[arenaIndex] = gameState;
                gameState.ResetBeforeStep();
                for (RocketSim::Car* car : envSet.arenas[arenaIndex]->_cars)
                    car->ResetStepContactState();
            }

            ActionParser* actionParser = envSet.actionParsers[arenaIndex];
            const int actionTicks = tickSkip - actionDelay;
            vector<Action> arenaActions(numPlayers);
            vector<vector<ActionMacro>> actionMacros(numPlayers);
            bool hasMacroAction = false;

            for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++) {
                const int actionIndex = actionIndices[playerOffset + playerIndex];
                const Player& player = gameState.players[playerIndex];
                vector<ActionMacro>& macros = actionMacros[playerIndex];
                if (actionDelay != 0 && actionParser->IsMacroAction(actionIndex, player, gameState)) {
                    RG_ERR_CLOSE(
                        "RunTeamSizeAnalysis: action macros require actionDelay == 0 "
                        << "(tickSkip=" << tickSkip << ", actionDelay=" << actionDelay << ")"
                    );
                }
                actionParser->ParseActionMacros(actionIndex, player, gameState, actionTicks, macros);

                if (macros.empty())
                    macros.push_back({ actionParser->ParseAction(actionIndex, player, gameState), actionTicks });

                int macroTicks = 0;
                for (const ActionMacro& macro : macros) {
                    RG_ASSERT(macro.ticks >= 0);
                    macroTicks += macro.ticks;
                }
                RG_ASSERT(macroTicks == actionTicks);

                arenaActions[playerIndex] = macros.back().action;
                if (macros.size() != 1 || macros.front().ticks != actionTicks)
                    hasMacroAction = true;
            }

            // Step remaining ticks, including parser-provided action macros.
            if (!hasMacroAction) {
                for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++) {
                    RocketSim::Car* car = envSet.arenas[arenaIndex]->GetCar(gameState.players[playerIndex].carId);
                    if (car)
                        car->controls = static_cast<RocketSim::CarControls>(arenaActions[playerIndex]);
                }
                envSet.arenas[arenaIndex]->Step(actionTicks);
            } else {
                vector<int> macroIndices(numPlayers, 0);
                vector<int> macroTicksLeft(numPlayers, 0);
                for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++)
                    macroTicksLeft[playerIndex] = actionMacros[playerIndex].front().ticks;

                for (int tick = 0; tick < actionTicks; tick++) {
                    for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++) {
                        vector<ActionMacro>& macros = actionMacros[playerIndex];
                        while (macroTicksLeft[playerIndex] <= 0 &&
                               macroIndices[playerIndex] + 1 < static_cast<int>(macros.size())) {
                            macroIndices[playerIndex]++;
                            macroTicksLeft[playerIndex] = macros[macroIndices[playerIndex]].ticks;
                        }

                        RocketSim::Car* car = envSet.arenas[arenaIndex]->GetCar(gameState.players[playerIndex].carId);
                        if (car) {
                            car->controls = static_cast<RocketSim::CarControls>(
                                macros[macroIndices[playerIndex]].action);
                        }
                    }

                    envSet.arenas[arenaIndex]->Step(1);

                    for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++)
                        if (macroTicksLeft[playerIndex] > 0)
                            macroTicksLeft[playerIndex]--;
                }
            }

            // Update event tracker (populates player.eventState: shot/save/goal/bump/demo).
            if (envSet.eventTrackers[arenaIndex])
                envSet.eventTrackers[arenaIndex]->Update(envSet.arenas[arenaIndex]);

            // Advance game state from the stepped arena (mirrors StepSecondHalf's UpdateFromArena).
            // prevGameStates[arenaIndex] was set by StepFirstHalf or the zero-delay branch
            // above; IsEmpty() matches the same guard used in StepSecondHalf.
            GameState* prevGameStatePtr = envSet.state.prevGameStates[arenaIndex].IsEmpty()
                ? nullptr : &envSet.state.prevGameStates[arenaIndex];
            gameState.UpdateFromArena(envSet.arenas[arenaIndex], arenaActions, prevGameStatePtr);

            // Check terminal conditions.
            envSet.state.terminals[arenaIndex] = TerminalType::NOT_TERMINAL;
            for (auto& condition : arenaResources.terminalConditions[arenaIndex]) {
                if (condition->IsTerminal(gameState)) {
                    envSet.state.terminals[arenaIndex] = condition->IsTruncation()
                        ? TerminalType::TRUNCATED
                        : TerminalType::NORMAL;
                    break;
                }
            }

            if (envSet.state.terminals[arenaIndex] == TerminalType::NOT_TERMINAL
                && envSet.stateSetters[arenaIndex]->EpisodeCapElapsed(gameState))
                envSet.state.terminals[arenaIndex] = TerminalType::TRUNCATED;

            const bool isFinal = (envSet.state.terminals[arenaIndex] != TerminalType::NOT_TERMINAL);

            // Collect per-reward data for this step.
            vector<WeightedReward>& arenaRewards = arenaResources.arenaRewards[arenaIndex];
            for (size_t rewardIndex = 0; rewardIndex < numRewards; rewardIndex++)
                arenaRewards[rewardIndex].reward->PreStep(gameState);

            EpisodeAccumulator& episode = arenaEpisodes[arenaIndex];
            const bool sampleAdaptiveRate = fairMatchup && arenaEpisodeIsNormalGameplay[arenaIndex];
            double totalStep = 0.0;

            for (size_t rewardIndex = 0; rewardIndex < numRewards; rewardIndex++) {
                WeightedReward& weightedReward = arenaRewards[rewardIndex];
                vector<float>& stepRewards = arenaRewardScratch[arenaIndex];
                weightedReward.reward->GetAllRewardsInto(gameState, isFinal, stepRewards);
                RG_ASSERT(stepRewards.size() == gameState.players.size());

                float playerAverage = 0.f;
                float playerAbsSum  = 0.f;
                for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++) {
                    const float weightedValue = stepRewards[playerIndex] * weightedReward.weight;
                    playerAverage += weightedValue;
                    playerAbsSum  += std::abs(weightedValue);
                    episode.g0PerPlayer[rewardIndex * numPlayers + playerIndex] += episode.discount * weightedValue;
                }
                playerAverage /= static_cast<float>(numPlayers);

                episode.g0[rewardIndex]     += episode.discount * playerAverage;
                episode.rawSum[rewardIndex] += playerAverage;
                totalStep += playerAverage;

                if (weightedReward.reward->GetCategory() == RewardCategory::EVENT) {
                    float rawSum = 0.f;
                    for (float rewardValue : stepRewards) rawSum += std::abs(rewardValue);

                    // Per-player absolute values so ZeroSum rewards show gross magnitude.
                    if (rawSum > 0.f) {
                        episode.fires[rewardIndex]++;
                        episode.firedWeighted[rewardIndex] += playerAbsSum / static_cast<float>(numPlayers);
                    }
                }

                // Adaptive-rate sample: unweighted |raw|, pre-zero-sum inner value, on
                // normal-gameplay episodes and fair matchups only. Mirrors EnvSet.cpp.
                if (sampleAdaptiveRate) {
                    ZeroSumReward* zeroSum = arenaZeroSumRewards[arenaIndex][rewardIndex];
                    for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++) {
                        const float rawValue = zeroSum ? zeroSum->_lastRewards[playerIndex] : stepRewards[playerIndex];
                        accumulator.normalGameplaySum[rewardIndex][arenaIndex][playerIndex] += std::fabs(rawValue);
                        accumulator.normalGameplayCount[rewardIndex][arenaIndex][playerIndex] += 1;
                    }
                }
            }

            episode.totalG0 += episode.discount * totalStep;
            episode.discount *= gaeGamma;
            episode.length++;

            // Timeout guard — mirrors the learner's maxEpisodeLength cap.
            // Rewards on this step were collected with isFinal=false (TC didn't fire),
            // matching learner behaviour (timeout is not signalled to rewards).
            const bool episodeDone = isFinal || episode.length >= maxEpisodeSteps;
            if (!episodeDone) {
                envSet.BuildArenaObs(arenaIndex);
                return;
            }

            // Top-up episodes exist purely to gather adaptive samples; keep them out of the return/impact stats.
            // A completed episode stays in arenaEpisodes until the serial fold reads and clears it.
            if (recordReturnStats)
                arenaEpisodeCompleted[arenaIndex] = 1;
            else
                episode.Clear();

            if (!startNewEpisodes) {
                arenaActive[arenaIndex] = 0;
                return;
            }

            envSet.ResetArena(arenaIndex);
            arenaEpisodeIsNormalGameplay[arenaIndex] = IsNormalGameplayEpisode(envSet.stateSetters[arenaIndex]->setterType);
            envSet.state.terminals[arenaIndex] = TerminalType::NOT_TERMINAL;
            for (WeightedReward& weightedReward : arenaRewards)
                weightedReward.reward->Reset(gameState);
        };

        // One collection step across all active arenas, then the serial fold of any finished episodes and the window flush.
        // Inactive (drained) arenas still occupy their inference rows; their actions are simply ignored.
        auto collectOneStep = [&](bool recordReturnStats, bool startNewEpisodes) {
            envSet.StepFirstHalf(true);
            envSet.Sync();

            const vector<int> actionIndices = inferUnit.BatchInferActionIndicesFromObs(
                envSet.state.obs.data, envSet.state.actionMasks.data, envSet.state.numPlayers, deterministic, policyTemperature);

            g_ThreadPool.StartBatchedJobs([&](int arenaIndex) {
                if (arenaActive[arenaIndex])
                    stepArena(arenaIndex, actionIndices, recordReturnStats, startNewEpisodes);
            }, numArenas, false);

            // Fold in arena order so results don't depend on which worker finished first.
            bool anyCompleted = false;
            for (int arenaIndex = 0; arenaIndex < numArenas; arenaIndex++) {
                if (!arenaEpisodeCompleted[arenaIndex])
                    continue;

                FoldEpisodeStats(arenaEpisodes[arenaIndex], numPlayers, numRewards, accumulator);
                arenaEpisodes[arenaIndex].Clear();
                arenaEpisodeCompleted[arenaIndex] = 0;
                anyCompleted = true;
            }

            // Every active arena advanced one step this iteration; close the window when it fills.
            if (++stepsSinceFlush >= windowSteps) {
                accumulator.FlushAdaptiveWindow();
                stepsSinceFlush = 0;
            }

            return anyCompleted;
        };

        // Phase 1: natural episode mix. Drives the return/impact stats and whatever adaptive samples
        // the stage's normal-gameplay episodes happen to produce.
        // Progress runs against the target until it is met, then against everything taken so the drain still advances the bar.
        const long long playerStepsPerCollect = static_cast<long long>(numArenas) * numPlayers;
        long long playerStepsTaken = 0;
        while (std::find(arenaActive.begin(), arenaActive.end(), 1) != arenaActive.end()) {
            if (isCancelled && isCancelled()) break;

            const bool startNewEpisodes = playerStepsTaken < numTimesteps;
            const long long activeArenas = std::count(arenaActive.begin(), arenaActive.end(), 1);
            playerStepsTaken += activeArenas * numPlayers;

            if (collectOneStep(true, startNewEpisodes)) {
                const long long progressTotal = std::max<long long>(numTimesteps, playerStepsTaken);
                progressCallback(accumulator.totalStepsCollected, static_cast<int>(progressTotal), /*topUp*/ false);
            }
        }
        accumulator.FlushAdaptiveWindow();
        stepsSinceFlush = 0;

        // Smallest per-key adaptive sample count so far (all rewards accumulate on the same
        // normal-gameplay steps, so these are equal in practice; min is a safe lower bound).
        auto adaptiveSampleCount = [&]() -> long long {
            if (accumulator.normalGameplayAccum.empty())
                return 0;
            long long minCount = std::numeric_limits<long long>::max();
            for (const AdaptiveRateAccumulator& acc : accumulator.normalGameplayAccum)
                minCount = std::min(minCount, acc.totalCount);
            return minCount;
        };

        // Phase 2: adaptive-rate top-up. Scenario-heavy stages collect almost no normal-gameplay in
        // phase 1, leaving the windowed Max estimated from a handful of samples (noisy, inflated). If
        // we're under the target (the controller's EMA half-life), run extra kickoff/random-only
        // episodes — which never touch the return stats — until we reach it, so the Max is stable and
        // comparable to the controller's per-iteration Max during training. Only fair matchups collect
        // adaptive samples, and only when the stage has adaptive rewards (adaptiveTargetSamples > 0).
        const bool cancelled = isCancelled && isCancelled();
        if (fairMatchup && !cancelled && adaptiveSampleCount() < adaptiveTargetSamples) {
            // One setter per arena: arenas reset concurrently, and setters carry per-instance state.
            vector<unique_ptr<StateSetter>> topUpSetters(numArenas);
            for (int arenaIndex = 0; arenaIndex < numArenas; arenaIndex++)
                topUpSetters[arenaIndex] = stage->GetNormalGameplayStateSetter(teamSizeA, teamSizeB);

            // nullptr => the stage has no kickoff/random setters for this matchup; leave the
            // adaptive rate at whatever phase 1 gathered (possibly undefined) rather than fake it.
            if (topUpSetters[0]) {
                // Point every arena at a normal-gameplay setter and restart episodes.
                for (int arenaIndex = 0; arenaIndex < numArenas; arenaIndex++) {
                    envSet.stateSetters[arenaIndex] = topUpSetters[arenaIndex].get();
                    envSet.ResetArena(arenaIndex);
                    arenaEpisodeIsNormalGameplay[arenaIndex] =
                        IsNormalGameplayEpisode(envSet.stateSetters[arenaIndex]->setterType);
                    envSet.state.terminals[arenaIndex] = TerminalType::NOT_TERMINAL;
                    for (WeightedReward& weightedReward : arenaResources.arenaRewards[arenaIndex])
                        weightedReward.reward->Reset(envSet.state.gameStates[arenaIndex]);
                    arenaEpisodes[arenaIndex].Clear();
                    arenaActive[arenaIndex] = 1;
                }

                // Bound the pass so a misconfigured stage can't loop forever.
                const long long maxTopUpPlayerSteps = adaptiveTargetSamples * 5;
                long long topUpPlayerSteps = 0;
                long long collected = adaptiveSampleCount();
                while (collected < adaptiveTargetSamples && topUpPlayerSteps < maxTopUpPlayerSteps) {
                    if (isCancelled && isCancelled()) break;
                    collectOneStep(false, true);
                    topUpPlayerSteps += playerStepsPerCollect;
                    collected = adaptiveSampleCount();
                    // Report top-up progress against the sample target (distinct label -> the GUI/
                    // console shows the extra phase instead of an apparently-finished, frozen bar).
                    progressCallback(static_cast<int>(std::min<long long>(collected, adaptiveTargetSamples)),
                        static_cast<int>(adaptiveTargetSamples), /*topUp*/ true);
                }
                accumulator.FlushAdaptiveWindow();
            }
        }
    }   // envSet destructor: deletes arenas, eventTrackers, eventCallbackInfos.

    return BuildTeamSizeResult(
        accumulator, arenaResources.arenaRewards[0], teamSizeA, teamSizeB);
    // arenaResources unique_ptrs clean up on scope exit (exception-safe).
}

} // anonymous namespace

namespace RocketForge
{

void FoldAdaptiveWindow(
    const vector<vector<double>>& windowSum,
    const vector<vector<long long>>& windowCount,
    AdaptiveRateAccumulator& acc)
{
    for (size_t arena = 0; arena < windowSum.size(); arena++) {
        double arenaMaxRate = 0.0;
        bool arenaHadSteps = false;

        const vector<double>&    sums   = windowSum[arena];
        const vector<long long>& counts = windowCount[arena];
        for (size_t player = 0; player < sums.size(); player++) {
            const long long c = counts[player];
            if (c <= 0)
                continue;
            acc.totalSum   += sums[player];
            acc.totalCount += c;
            const double rate = sums[player] / static_cast<double>(c);
            arenaMaxRate = std::max(arenaMaxRate, rate);
            arenaHadSteps = true;
        }

        // An arena with no steps this window contributes no max sample: it must not dilute
        // the Max mean with a spurious 0 (mirrors UpdateAndApply's arenaHadSteps gate).
        if (arenaHadSteps) {
            acc.maxRateSum     += arenaMaxRate;
            acc.maxWindowCount += 1;
        }
    }
}

AdaptiveRateResult FinalizeWindowedAdaptiveRate(const AdaptiveRateAccumulator& acc)
{
    AdaptiveRateResult result;
    result.sampleCount = acc.totalCount;
    if (acc.totalCount > 0)
        result.avgPerStep = static_cast<float>(acc.totalSum / static_cast<double>(acc.totalCount));
    if (acc.maxWindowCount > 0)
        result.maxPerStep = static_cast<float>(acc.maxRateSum / static_cast<double>(acc.maxWindowCount));
    return result;
}

AdaptiveRateResult ComputeAdaptiveGameplayRate(
    const vector<vector<double>>& perPlayerSum,
    const vector<vector<long long>>& perPlayerCount)
{
    // The whole run treated as a single window.
    AdaptiveRateAccumulator acc;
    FoldAdaptiveWindow(perPlayerSum, perPlayerCount, acc);
    return FinalizeWindowedAdaptiveRate(acc);
}

void FoldAdaptiveRate(
    float srcAvg, float srcMax, long long srcSampleCount, float teamWeight,
    float& dstAvgWeightedSum, float& dstMaxWeightedSum,
    float& dstCollectedWeight, long long& dstSampleCount)
{
    dstSampleCount += srcSampleCount;
    if (srcSampleCount <= 0)
        return;

    dstAvgWeightedSum  += srcAvg * teamWeight;
    dstMaxWeightedSum  += srcMax * teamWeight;
    dstCollectedWeight += teamWeight;
}

void FinalizeAdaptiveRate(
    float avgWeightedSum, float maxWeightedSum, float collectedWeight,
    float& dstAvg, float& dstMax)
{
    if (collectedWeight <= 0.f)
        return;

    dstAvg = avgWeightedSum / collectedWeight;
    dstMax = maxWeightedSum / collectedWeight;
}

RewardAnalysisResult RunRewardAnalysis(
    TrainingStage* stage,
    const filesystem::path& modelsPath,
    int numTimesteps,
    const function<void(int, int, const string&)>& progressCallback,
    const function<bool()>& isCancelled)
{
    if (numTimesteps <= 0)
        numTimesteps = stage->GetTimestepsPerIteration();

    // Resolve an inference-ready agent (applies the segment's saved agent setup if present,
    // sets modelPath) via the shared helper, so mismatched/imported models load uniformly.
    ResolvedInferenceAgent resolved = ResolveInferenceAgent(FromSegmentCheckpoint(stage->agent->agentName, modelsPath));

    if (resolved.appliedSnapshot.has_value())
        RG_LOG("RunRewardAnalysis: applied saved agent setup from " << resolved.checkpointFolder);

    // Build obs/action + InferUnit from the resolved agent before moving it into the stage.
    // The InferUnit keeps no Agent reference after construction (it only calls MakeInferenceModels
    // during the ctor), so the later move is safe. obsBuilder/actionParser are borrowed by the
    // InferUnit and must outlive it — they do (locals for the whole function).
    unique_ptr<ObsBuilder>   obsBuilder(resolved.agent->MakeConfiguredObsBuilder());
    unique_ptr<ActionParser> actionParser(resolved.agent->MakeActionParser());

    const bool useGPU = ResolveInferenceUseGPU();
    unique_ptr<GGL::InferUnit> inferUnitPtr = MakeInferUnit(resolved, obsBuilder.get(), actionParser.get(), useGPU);
    GGL::InferUnit& inferUnit = *inferUnitPtr;

    // The analysis rollout drives stage->agent (obs/action/team sizes). Swap in the resolved
    // (snapshot-adjusted) agent so the env matches the saved weights.
    stage->agent = std::move(resolved.agent);

    const float gaeGamma = stage->learnerConfig.ppo.gaeGamma;
    const vector<WeightedTeamSize>& teamSizes = stage->GetWeightedTeamSizes();

    RewardAnalysisResult result;
    result.schemaVersion        = REWARD_ANALYSIS_SCHEMA_VERSION;
    result.agentName            = stage->agent->agentName;
    result.stageName            = stage->GetStageName();
    result.checkpointPath       = modelsPath.string();
    result.timestepsPerTeamSize = numTimesteps;
    result.gaeGamma             = gaeGamma;

    const vector<AdaptivePersistEntry> adaptiveWeights = LoadPersistedAdaptiveWeights(modelsPath);
    if (!adaptiveWeights.empty())
        RG_LOG("RunRewardAnalysis: loaded " << adaptiveWeights.size()
            << " effective adaptive reward weight(s) from RUNNING_STATS.json");

    for (const WeightedTeamSize& teamSize : teamSizes) {
        if (isCancelled && isCancelled()) break;
        const string label = MakeLabel(teamSize.teamSizeA, teamSize.teamSizeB);
        TeamSizeImpactResult teamSizeResult = RunTeamSizeAnalysis(
            stage, inferUnit, teamSize.teamSizeA, teamSize.teamSizeB,
            numTimesteps, gaeGamma,
            [&](int stepsCollected, int total, bool topUp) {
                progressCallback(stepsCollected, total,
                    topUp ? (label + " (sampling gameplay for adaptive rates)") : label);
            },
            isCancelled, adaptiveWeights);
        teamSizeResult.weight = teamSize.weight;
        result.teamSizes.push_back(std::move(teamSizeResult));
    }

    // Hard-error if a non-cancelled run failed to complete a single episode for any matchup —
    // the result would contain NaN/inf otherwise. Cancelled runs are allowed to be empty.
    const bool wasCancelled = isCancelled && isCancelled();
    if (!wasCancelled) {
        for (const TeamSizeImpactResult& teamResult : result.teamSizes) {
            if (teamResult.episodesRun == 0)
                RG_ERR_CLOSE("RunRewardAnalysis: zero episodes completed for matchup "
                    << teamResult.label << ". Increase --timesteps or shorten the episode length.");
        }
    }

    if (!result.teamSizes.empty())
        result.average = ComputeAverage(result.teamSizes);

    return result;
}

string RewardAnalysisCacheFileName(const string& stageName)
{
    return "reward_analysis_" + RocketForgeTraining::BuildStorage::SanitizeFolderPart(stageName) + ".json";
}

bool SaveRewardAnalysisResult(const RewardAnalysisResult& result,
                              const filesystem::path& savePath,
                              const string& stageName,
                              string* error)
{
    filesystem::path filePath = savePath / RewardAnalysisCacheFileName(stageName);
    ofstream out(filePath);
    if (!out.good()) {
        if (error) *error = "Cannot open file for writing: " + filePath.string();
        return false;
    }
    nlohmann::json j = result;
    out << setw(4) << j;
    return true;
}

optional<RewardAnalysisResult> TryLoadRewardAnalysisResult(
    const filesystem::path& savePath,
    const string& stageName)
{
    filesystem::path filePath = savePath / RewardAnalysisCacheFileName(stageName);
    if (!filesystem::exists(filePath))
        return nullopt;

    ifstream in(filePath);
    if (!in.good())
        return nullopt;

    // Non-throwing parse — codebase convention is to avoid exceptions.
    nlohmann::json j = nlohmann::json::parse(in, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded())
        return nullopt;

    RewardAnalysisResult result;
    from_json(j, result);

    // Reject files from a future schema version we don't know how to read.
    if (result.schemaVersion > REWARD_ANALYSIS_SCHEMA_VERSION)
        return nullopt;

    return result;
}

} // namespace RocketForge
