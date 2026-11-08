#pragma once

#include "RewardDisplayRow.h"
#include "ValueTrendTracker.h"

#include <CriticDebugData.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace ForgeGUI {

/**
 * One display row per decomposed-critic head, in critic row order.
 * A head is enabled when no selection was saved yet, its name is selected, or showAllHeads is set;
 * its checkbox still reflects the selection alone.
 * Norm is relative to peakAbsContribution, the largest magnitude among enabled heads with a value on any frame so far,
 * which this call raises to cover the current frame; a frame-local peak would pin a lone head at ±1.
 */
inline std::vector<RewardDisplayRow> BuildCriticHeadRows(
    const ForgeDebug::CriticDebugData& critic,
    const std::vector<std::string>& selectedHeads,
    bool hasSelection,
    bool showAllHeads,
    const ValueTrendTracker& trends,
    float& peakAbsContribution)
{
    std::vector<RewardDisplayRow> rows;
    rows.reserve(critic.heads.size());

    for (int headIdx = 0; headIdx < (int)critic.heads.size(); headIdx++) {
        const ForgeDebug::CriticHeadDebugData& head = critic.heads[headIdx];
        const bool selected = !hasSelection || std::find(selectedHeads.begin(), selectedHeads.end(), head.name) != selectedHeads.end();
        const TrendReading trend = trends.Get(head.name);

        RewardDisplayRow row;
        row.index = headIdx;
        row.name = head.name;
        row.desiredEnabled = selected;
        row.enabled = selected || showAllHeads;
        row.hasValue = head.weight.has_value();
        row.displayWeighted = head.contribution;
        row.decayAlpha = 1.f;
        row.absSum = head.episodeAbsContributionSum;
        row.trendDelta = trend.delta;
        row.trendDirection = trend.direction;

        if (row.enabled && row.hasValue)
            peakAbsContribution = std::max(peakAbsContribution, std::abs(head.contribution));

        rows.push_back(row);
    }

    for (RewardDisplayRow& row : rows)
        if (row.enabled && row.hasValue)
            row.displayNorm = peakAbsContribution > 0.f ? row.displayWeighted / peakAbsContribution : 0.f;

    ScaleTrendIntensities(rows);

    return rows;
}

/** The selection after one head's checkbox changes. Without a saved selection every head counts as selected. */
inline std::vector<std::string> ToggleCriticHeadSelection(
    const std::vector<std::string>& selectedHeads,
    bool hasSelection,
    const std::vector<std::string>& allHeads,
    const std::string& head,
    bool enabled)
{
    std::vector<std::string> selection = hasSelection ? selectedHeads : allHeads;
    const auto existing = std::find(selection.begin(), selection.end(), head);

    if (enabled && existing == selection.end())
        selection.push_back(head);
    else if (!enabled && existing != selection.end())
        selection.erase(existing);

    return selection;
}

} // namespace ForgeGUI
