#pragma once

#include <RLGymCPP/Rewards/Reward.h>

#include <GigaLearnInfer/JsonArchive.h>

#include <string>
#include <vector>

namespace RocketForge {

struct RewardTableEntry {
    std::string name;

    /** The larger-magnitude bound of the reward's value range times its weight, keeping its sign. */
    float effectiveMax;

    float weight;
    RLGC::RewardValueRange range;
    RLGC::RewardCategory category;

    void Serialize(GGL::JsonArchive& ar);
};

struct RewardComparisonTable {
    std::vector<RewardTableEntry> rewards;

    /**
     * ratios[i][j] = effectiveMax[i] / effectiveMax[j], read as "1 [row] = ratios[i][j] [col]".
     * NaN when the column's effectiveMax is zero.
     */
    std::vector<std::vector<float>> ratios;

    static RewardComparisonTable Build(const std::vector<RLGC::WeightedReward>& rewards);
    static RewardComparisonTable Build(const std::vector<RewardTableEntry>& rewards);

    void PrintToConsole() const;
};

} // namespace RocketForge
