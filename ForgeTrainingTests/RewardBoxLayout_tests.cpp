#include <gtest/gtest.h>
#include "RewardVisualizer/RewardBoxLayout.h"

#include <vector>

using namespace ForgeGUI;

// cfg defaults: row=20, header=24, minRows=3, extraVisibleRows=0 -> minBox=84

TEST(RewardBoxLayout, PlentyOfRoom_ReturnsDesired) {
    // cont: 5 enabled / 10 total -> desired = 24 + 5*20 = 124
    // evt:  4 enabled / 6  total -> desired = 24 + 4*20 = 104
    std::vector<int> h = AllocateBoxHeights(400, { { 5, 10 }, { 4, 6 } });
    ASSERT_EQ(h[0], 124);
    ASSERT_EQ(h[1], 104);
}

TEST(RewardBoxLayout, Constrained_GreedyWithMin) {
    // desiredC=124, desiredE=104, sum=228 > 180
    // minC=minE=84, remaining = 180-168 = 12
    // wantC=40, wantE=20 -> extraC = 12*40/60 = 8
    // hC = 92, hE = 84 + (12-8) = 88  (busier continuous gets more)
    std::vector<int> h = AllocateBoxHeights(180, { { 5, 10 }, { 4, 6 } });
    ASSERT_EQ(h[0], 92);
    ASSERT_EQ(h[1], 88);
    ASSERT_EQ(h[0] + h[1], 180);
}

TEST(RewardBoxLayout, TooShortForBothMins_SplitsByDesire) {
    // minC+minE = 168 > 150 -> remaining < 0
    // hC = 150 * 124 / 228 = 81, hE = 69
    std::vector<int> h = AllocateBoxHeights(150, { { 5, 10 }, { 4, 6 } });
    ASSERT_EQ(h[0], 81);
    ASSERT_EQ(h[1], 69);
    ASSERT_EQ(h[0] + h[1], 150);
}

TEST(RewardBoxLayout, EmptyCategory_GetsHeaderOnly) {
    // cont: 0/0 -> full=24, desired=clamp(24, min(84,24)=24, 24)=24
    // evt:  3/8 -> desired = 24 + 3*20 = 84
    std::vector<int> h = AllocateBoxHeights(400, { { 0, 0 }, { 3, 8 } });
    ASSERT_EQ(h[0], 24);
    ASSERT_EQ(h[1], 84);
}

TEST(RewardBoxLayout, NonPositiveAvailable_ReturnsZero) {
    for (int available : { 0, -1 }) {
        std::vector<int> h = AllocateBoxHeights(available, { { 5, 10 }, { 4, 6 } });
        ASSERT_EQ(h[0], 0);
        ASSERT_EQ(h[1], 0);
    }
}

TEST(RewardBoxLayout, BufferShowsFewDisabled_NotAll) {
    // With extraVisibleRows=2 and lots of disabled rewards, the box must size to
    // enabled+2, NOT to all rows -- the extra disabled scroll.
    // cont: 3 enabled / 12 total -> want = 24 + (3+2)*20 = 124 (full would be 264)
    // evt:  0/0 -> 24
    RewardBoxLayoutConfig cfg;
    cfg.extraVisibleRows = 2;
    std::vector<int> h = AllocateBoxHeights(600, { { 3, 12 }, { 0, 0 } }, cfg);
    ASSERT_EQ(h[0], 124);
    ASSERT_EQ(h[1], 24);
}

TEST(RewardBoxLayout, BufferCappedByTotalRows) {
    // The disabled buffer never invents rows: with only 1 disabled reward, the
    // box stops at the real total (full), not enabled+2.
    // cont: 4 enabled / 5 total -> want=24+(4+2)*20=144, clamped to full=124
    RewardBoxLayoutConfig cfg;
    cfg.extraVisibleRows = 2;
    std::vector<int> h = AllocateBoxHeights(600, { { 4, 5 }, { 0, 0 } }, cfg);
    ASSERT_EQ(h[0], 124);
    ASSERT_EQ(h[1], 24);
}

TEST(RewardBoxLayout, MeasuredMetricsDriveHeight) {
    // The GUI passes measured pixel sizes; a taller real row height yields a
    // taller table so enabled rows are not clipped.
    // row=28, header=30, 4 enabled / 9 total, extra=2 -> 30 + (4+2)*28 = 198
    RewardBoxLayoutConfig cfg;
    cfg.rowHeight        = 28;
    cfg.headerHeight     = 30;
    cfg.extraVisibleRows = 2;
    std::vector<int> h = AllocateBoxHeights(800, { { 4, 9 }, { 0, 0 } }, cfg);
    ASSERT_EQ(h[0], 198);
}

// Third box {2, 8}: desired = clamp(24 + 2*20 = 64, min(84, 184) = 84, 184) = 84.

TEST(RewardBoxLayout, ThreeBoxes_PlentyOfRoom_ReturnsDesired) {
    std::vector<int> h = AllocateBoxHeights(400, { { 5, 10 }, { 4, 6 }, { 2, 8 } });
    ASSERT_EQ(h.size(), size_t(3));
    ASSERT_EQ(h[0], 124);
    ASSERT_EQ(h[1], 104);
    ASSERT_EQ(h[2], 84);
}

TEST(RewardBoxLayout, ThreeBoxes_Constrained_GreedyWithMin) {
    // desired 124 + 104 + 84 = 312 > 270; mins 84 * 3 = 252, remaining 18
    // wants 40, 20, 0 -> extra0 = 18*40/60 = 12, extra1 = 18*20/60 = 6, last keeps 84
    std::vector<int> h = AllocateBoxHeights(270, { { 5, 10 }, { 4, 6 }, { 2, 8 } });
    ASSERT_EQ(h[0], 96);
    ASSERT_EQ(h[1], 90);
    ASSERT_EQ(h[2], 84);
    ASSERT_EQ(h[0] + h[1] + h[2], 270);
}

TEST(RewardBoxLayout, ThreeBoxes_TooShortForMins_SplitsByDesire) {
    // 200 < 252: h0 = 200*124/312 = 79, h1 = 200*104/312 = 66, last = 55
    std::vector<int> h = AllocateBoxHeights(200, { { 5, 10 }, { 4, 6 }, { 2, 8 } });
    ASSERT_EQ(h[0], 79);
    ASSERT_EQ(h[1], 66);
    ASSERT_EQ(h[2], 55);
}

TEST(RewardBoxLayout, NoBoxes_ReturnsEmpty) {
    ASSERT_TRUE(AllocateBoxHeights(400, {}).empty());
}
