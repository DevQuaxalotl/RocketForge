#pragma once

#include <algorithm>
#include <vector>

namespace ForgeGUI {

// Pure, Qt-free layout math for the reward-debug boxes.
// Kept header-only and dependency-free so ForgeTrainingTests can unit-test it
// without linking Qt/ForgeGUI.

struct RewardBoxMetrics {
    /** Rewards currently enabled in this category (shown without scrolling). **/
    int enabledRows = 0;
    /** All rewards in this category (enabled + disabled). **/
    int totalRows   = 0;
};

struct RewardBoxLayoutConfig {
    /** Per-row height in pixels. The live GUI passes the table's *measured* row
     *  height; the default matches the QTableWidget default section size. **/
    int rowHeight    = 20;
    /** Table horizontal-header height in pixels (measured by the GUI at runtime). **/
    int headerHeight = 24;
    /** Rows each box keeps visible before leftover space is shared, when space allows. **/
    int minRows      = 3;
    /** Disabled rows kept visible below the enabled ones in the desired height.
     *  The box never grows past enabled + this (the remaining disabled rows scroll), so quiet boxes stay compact.
     *  Default 0 keeps the pure unit tests stable; the live GUI sets this to 2. **/
    int extraVisibleRows = 0;
};

// Decide each box's table height given the available vertical space.
//
// Precondition: for each metric, enabledRows <= totalRows. If violated, the
// clamp below still yields a well-defined (capped) result rather than UB.
//
// Rule:
//   desired_X = clamp(header + (enabled_X + extra)*row, min(minBox, full_X), full_X)
//               i.e. show every enabled row plus a small disabled buffer,
//               never more than the box actually has.
//   if desired sum fits -> each gets desired (leftover -> caller's stretch)
//   else                -> reserve min for each, distribute the remainder
//                          proportionally to unmet desire (greedy, capped).
//   if even the minimums don't fit -> split proportionally by desire.
// The last box takes any integer remainder so the heights sum to exactly `available`.
inline std::vector<int> AllocateBoxHeights(
    int available,
    const std::vector<RewardBoxMetrics>& boxes,
    const RewardBoxLayoutConfig& cfg = {})
{
    std::vector<int> heights(boxes.size(), 0);
    if (available <= 0 || boxes.empty())
        return heights;

    const int minBox = cfg.headerHeight + cfg.minRows * cfg.rowHeight;

    const auto fullH = [&](const RewardBoxMetrics& m) {
        return cfg.headerHeight + m.totalRows * cfg.rowHeight;
    };
    const auto desiredH = [&](const RewardBoxMetrics& m) {
        const int full = fullH(m);
        const int want = cfg.headerHeight + (m.enabledRows + cfg.extraVisibleRows) * cfg.rowHeight;
        return std::clamp(want, std::min(minBox, full), full);
    };

    std::vector<int> desired;
    std::vector<int> minimum;
    int desiredSum = 0;
    int minimumSum = 0;
    for (const RewardBoxMetrics& box : boxes) {
        desired.push_back(desiredH(box));
        minimum.push_back(std::min(minBox, fullH(box)));
        desiredSum += desired.back();
        minimumSum += minimum.back();
    }

    if (desiredSum <= available)
        return desired;

    const int boxCount = (int)boxes.size();
    const int lastIdx = boxCount - 1;
    const int remaining = available - minimumSum;

    if (remaining <= 0) {
        int assigned = 0;
        for (int i = 0; i < lastIdx; i++) {
            heights[i] = desiredSum > 0 ? available * desired[i] / desiredSum : available / boxCount;
            assigned += heights[i];
        }
        heights[lastIdx] = available - assigned;
        return heights;
    }

    int wantTotal = 0;
    for (int i = 0; i < boxCount; i++)
        wantTotal += desired[i] - minimum[i];

    int extraAssigned = 0;
    for (int i = 0; i < lastIdx; i++) {
        const int want = desired[i] - minimum[i];
        const int extra = std::min(wantTotal > 0 ? remaining * want / wantTotal : remaining / boxCount, want);
        heights[i] = minimum[i] + extra;
        extraAssigned += extra;
    }
    heights[lastIdx] = minimum[lastIdx] + (remaining - extraAssigned);

    return heights;
}

} // namespace ForgeGUI
