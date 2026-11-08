#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace ForgeGUI {

/** Sort keys of the reward visualizer's Sort control; the values are the combo's item data. */
enum class RewardSortKey { Name = 0, Weighted = 1, AbsSum = 2, Norm = 3 };

/**
 * One table row with its values already resolved by the coordinator (e.g. held event values).
 * Header-only and Qt-free so ForgeTrainingTests can unit-test the sort.
 */
struct RewardDisplayRow {
    /** Stable id reported back when the row's checkbox is toggled. */
    int index = -1;

    std::string name;

    /** Checkbox state from the authoritative selection. */
    bool desiredEnabled = false;

    /** Values are shown; a disabled row is greyed out with empty value cells. */
    bool enabled = false;

    /** False for an enabled row with nothing to show; its value cells read "—". */
    bool hasValue = true;

    float displayWeighted = 0.f;
    float displayNorm = 0.f;

    /** Bar opacity, 0..1. */
    float decayAlpha = 0.f;

    /** Sum of |displayWeighted| over the episode so far; also what RewardSortKey::AbsSum orders by. */
    float absSum = 0.f;

    /** Recent change of displayWeighted, as ValueTrendTracker measures it. */
    float trendDelta = 0.f;

    /** +1 rising, -1 falling, 0 flat. */
    int trendDirection = 0;

    /** |trendDelta| relative to the table's largest, 0..1; see ScaleTrendIntensities. */
    float trendIntensity = 0.f;
};

/** Sets each enabled row's trendIntensity relative to the largest |trendDelta| among enabled rows with a value. */
inline void ScaleTrendIntensities(std::vector<RewardDisplayRow>& rows) {
    float maxAbsDelta = 0.f;
    for (const RewardDisplayRow& row : rows)
        if (row.enabled && row.hasValue)
            maxAbsDelta = std::max(maxAbsDelta, std::abs(row.trendDelta));

    for (RewardDisplayRow& row : rows)
        if (row.enabled && row.hasValue)
            row.trendIntensity = maxAbsDelta > 0.f ? std::abs(row.trendDelta) / maxAbsDelta : 0.f;
}

/** Enabled rows first, then by key; ties fall back to index so the order holds still frame to frame. */
inline void SortRewardDisplayRows(std::vector<RewardDisplayRow>& rows, RewardSortKey key, bool descending) {
    const auto rowLess = [key, descending](const RewardDisplayRow& a, const RewardDisplayRow& b) {
        if (a.enabled != b.enabled)
            return a.enabled;

        const auto valueLess = [descending](float l, float r) {
            return descending ? l > r : l < r;
        };
        const auto differs = [](float l, float r) {
            return std::abs(l - r) >= 1e-6f;
        };

        switch (key) {
        case RewardSortKey::Name:
            if (a.name != b.name)
                return descending ? a.name > b.name : a.name < b.name;
            break;
        case RewardSortKey::Weighted:
            if (differs(a.displayWeighted, b.displayWeighted))
                return valueLess(a.displayWeighted, b.displayWeighted);
            break;
        case RewardSortKey::AbsSum:
            if (differs(a.absSum, b.absSum))
                return valueLess(a.absSum, b.absSum);
            break;
        case RewardSortKey::Norm:
            if (differs(a.displayNorm, b.displayNorm))
                return valueLess(a.displayNorm, b.displayNorm);
            break;
        }

        return a.index < b.index;
    };

    std::stable_sort(rows.begin(), rows.end(), rowLess);
}

} // namespace ForgeGUI
