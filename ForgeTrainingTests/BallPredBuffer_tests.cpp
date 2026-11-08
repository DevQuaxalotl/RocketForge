#include <gtest/gtest.h>

#include <RLGymCPP/Framework.h>

namespace {
    BallState MakeBallStateAt(float x) {
        BallState s;
        s.pos = Vec(x, 0.f, 0.f);
        return s;
    }
}

TEST(BallPredBuffer, Reset_SetsCapacityAndClearsCount) {
    BallPredBuffer buf;
    buf.Reset(8);
    ASSERT_EQ(buf.capacity(), 8u);
    ASSERT_EQ(buf.size(), 0u);
    ASSERT_TRUE(buf.empty());
}

TEST(BallPredBuffer, Push_FillsInOrder) {
    BallPredBuffer buf;
    buf.Reset(4);
    for (int i = 0; i < 4; ++i)
        buf.Push(MakeBallStateAt(static_cast<float>(i)));

    ASSERT_EQ(buf.size(), 4u);
    ASSERT_FALSE(buf.empty());
    for (int i = 0; i < 4; ++i)
        ASSERT_EQ(buf[i].pos.x, static_cast<float>(i));
}

TEST(BallPredBuffer, AdvanceThenPush_WrapsAround) {
    BallPredBuffer buf;
    buf.Reset(4);
    for (int i = 0; i < 4; ++i)
        buf.Push(MakeBallStateAt(static_cast<float>(i)));

    buf.Advance(2); // logical [0,1] dropped; size 2; capacity 4
    ASSERT_EQ(buf.size(), 2u);
    ASSERT_EQ(buf[0].pos.x, 2.0f);
    ASSERT_EQ(buf[1].pos.x, 3.0f);

    buf.Push(MakeBallStateAt(4.0f));
    buf.Push(MakeBallStateAt(5.0f));
    ASSERT_EQ(buf.size(), 4u);
    ASSERT_EQ(buf[0].pos.x, 2.0f);
    ASSERT_EQ(buf[1].pos.x, 3.0f);
    ASSERT_EQ(buf[2].pos.x, 4.0f);
    ASSERT_EQ(buf[3].pos.x, 5.0f);
}

TEST(BallPredBuffer, AdvanceAll_LeavesEmpty) {
    BallPredBuffer buf;
    buf.Reset(3);
    for (int i = 0; i < 3; ++i)
        buf.Push(MakeBallStateAt(static_cast<float>(i)));
    buf.Advance(3);
    ASSERT_EQ(buf.size(), 0u);
    ASSERT_TRUE(buf.empty());

    buf.Push(MakeBallStateAt(99.0f));
    ASSERT_EQ(buf.size(), 1u);
    ASSERT_EQ(buf[0].pos.x, 99.0f);
}

TEST(BallPredBuffer, FrontAndBack_TrackLogicalEnds) {
    BallPredBuffer buf;
    buf.Reset(4);
    buf.Push(MakeBallStateAt(10.f));
    buf.Push(MakeBallStateAt(20.f));
    buf.Push(MakeBallStateAt(30.f));
    ASSERT_EQ(buf.front().pos.x, 10.f);
    ASSERT_EQ(buf.back().pos.x, 30.f);

    buf.Advance(1);
    ASSERT_EQ(buf.front().pos.x, 20.f);
    ASSERT_EQ(buf.back().pos.x, 30.f);
}
