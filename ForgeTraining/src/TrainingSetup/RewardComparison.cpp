#include "RewardComparison.h"
#include "TrainingSetupSnapshot.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace RocketForge {

void RewardTableEntry::Serialize(GGL::JsonArchive& ar)
{
    ar.Field("name", name);
    ar.Field("effective_max", effectiveMax);
    ar.Field("weight", weight);
    ar.Field("range", range, RLGC::RewardValueRange{ 0.f, 0.f });
    ar.EnumField("category", category, REWARD_CATEGORY_NAMES);
}

namespace {

// clang-format off
static constexpr int TABLE_WIDTH      = 80;
static constexpr int COLUMNS_PER_PAGE = 12;
static constexpr int ROW_LABEL_WIDTH  = 22;
static constexpr int COL_LABEL_WIDTH  = 7;
static constexpr int CELL_WIDTH       = 7;
// clang-format on

std::string TruncateName(const std::string& name, int maxLength)
{
    return (int)name.size() <= maxLength ? name : name.substr(0, maxLength - 3) + "...";
}

std::string FormatRatio(float ratio)
{
    if (std::isnan(ratio))
        return "  N/A  ";

    std::ostringstream out;
    const float absRatio = std::abs(ratio);
    if (absRatio >= 1000.f)
        out << std::setw(CELL_WIDTH) << std::setprecision(2) << std::scientific << ratio;
    else if (absRatio >= 100.f)
        out << std::setw(CELL_WIDTH) << std::fixed << std::setprecision(0) << ratio;
    else
        out << std::setw(CELL_WIDTH) << std::fixed << std::setprecision(2) << ratio;

    return out.str();
}

} // namespace

RewardComparisonTable RewardComparisonTable::Build(const std::vector<RewardTableEntry>& rewardEntries)
{
    RewardComparisonTable result;
    const int rewardCount = static_cast<int>(rewardEntries.size());
    result.rewards = rewardEntries;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    result.ratios.assign(rewardCount, std::vector<float>(rewardCount, nan));

    for (int i = 0; i < rewardCount; i++) {
        for (int j = 0; j < rewardCount; j++) {
            if (i == j) {
                result.ratios[i][j] = 1.0f;
                continue;
            }

            const float rowMax = result.rewards[i].effectiveMax;
            const float colMax = result.rewards[j].effectiveMax;
            if (colMax == 0.f)
                continue;

            result.ratios[i][j] = rowMax / colMax;
        }
    }

    return result;
}

RewardComparisonTable RewardComparisonTable::Build(const std::vector<RLGC::WeightedReward>& weightedRewards)
{
    std::vector<RewardTableEntry> rewardEntries;
    rewardEntries.reserve(weightedRewards.size());

    for (const RLGC::WeightedReward& weightedReward : weightedRewards) {
        RewardTableEntry entry;
        entry.name     = weightedReward.reward->GetName();
        entry.weight   = weightedReward.weight;
        entry.range    = weightedReward.reward->GetValueRange();
        entry.category = weightedReward.reward->GetCategory();

        // Use whichever bound has the higher magnitude, preferring positive on a tie,
        // so a punishment with range [-1, 0] isn't measured by its zero maxValue.
        const float absMax = std::abs(entry.range.maxValue);
        const float absMin = std::abs(entry.range.minValue);
        const float peak   = (absMax >= absMin) ? entry.range.maxValue : entry.range.minValue;
        entry.effectiveMax = peak * weightedReward.weight;

        rewardEntries.push_back(std::move(entry));
    }

    return Build(rewardEntries);
}

void RewardComparisonTable::PrintToConsole() const
{
    const int rewardCount = static_cast<int>(rewards.size());
    const std::string separator(TABLE_WIDTH, '=');
    const std::string divider(TABLE_WIDTH, '-');

    std::cout << separator << "\n";
    std::cout << " REWARD COMPARISON TABLE\n";
    std::cout << separator << "\n\n";

    // --- Section 1: Effective Max legend ---
    std::cout << " Effective Max per Reward  (abs peak of range × weight, signed)\n";
    std::cout << divider << "\n";
    std::cout << std::left
              << "  " << std::setw(3) << "#"
              << std::setw(28) << "Name"
              << std::right
              << std::setw(10) << "Weight"
              << std::setw(16) << "Range"
              << std::setw(10) << "EffMax"
              << "\n";
    std::cout << divider << "\n";

    for (int i = 0; i < rewardCount; i++) {
        const RewardTableEntry& entry = rewards[i];
        std::ostringstream rangeStr;
        rangeStr << std::fixed << std::setprecision(2)
                 << "[" << entry.range.minValue << ", " << entry.range.maxValue << "]";

        std::cout << std::left
                  << "  " << std::setw(3) << i
                  << std::setw(28) << TruncateName(entry.name, 27)
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(10) << entry.weight
                  << std::setw(16) << rangeStr.str()
                  << std::setw(10) << entry.effectiveMax
                  << "\n";
    }

    if (rewardCount == 0) {
        std::cout << "  (no rewards)\n";
        std::cout << separator << "\n";
        return;
    }

    // --- Section 2: Ratio matrix (paged) ---
    std::cout << "\n";
    std::cout << " Ratio Matrix  ratios[row][col] = EffMax(row) / EffMax(col)\n";
    std::cout << " Read: \"1 [row] = [value] [col]\"\n";
    std::cout << divider << "\n";

    const int pages = (rewardCount + COLUMNS_PER_PAGE - 1) / COLUMNS_PER_PAGE;
    for (int page = 0; page < pages; page++) {
        const int colStart = page * COLUMNS_PER_PAGE;
        const int colEnd   = std::min(colStart + COLUMNS_PER_PAGE, rewardCount);

        if (pages > 1)
            std::cout << " -- Columns " << colStart << " to " << (colEnd - 1) << " --\n";

        // Header row
        std::cout << std::left << std::setw(ROW_LABEL_WIDTH + 2) << "";
        for (int j = colStart; j < colEnd; j++)
            std::cout << std::right << std::setw(CELL_WIDTH) << TruncateName(rewards[j].name, COL_LABEL_WIDTH);
        std::cout << "\n";

        // Data rows
        for (int i = 0; i < rewardCount; i++) {
            std::cout << " " << std::left << std::setw(ROW_LABEL_WIDTH) << TruncateName(rewards[i].name, ROW_LABEL_WIDTH) << " ";
            for (int j = colStart; j < colEnd; j++)
                std::cout << FormatRatio(ratios[i][j]);
            std::cout << "\n";
        }

        if (page + 1 < pages)
            std::cout << "\n";
    }

    std::cout << separator << "\n";
}

} // namespace RocketForge
