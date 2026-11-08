#include <gtest/gtest.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/MultiPartReward.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

#include <bit>
#include <cmath>
#include <memory>
#include <random>

using namespace RLGC;

namespace {
    GameState MakeRewardState(int count) {
        GameState state;
        for (int i = 0; i < count; i++) {
            Player player = {};
            player.index = i;
            player.carId = i + 1;
            player.team = i % 2 ? Team::ORANGE : Team::BLUE;
            state.players.push_back(player);
        }

        return state;
    }

    void ExpectSame(const std::vector<float>& expected, const std::vector<float>& actual) {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < expected.size(); i++) {
            if (std::isnan(expected[i]))
                EXPECT_TRUE(std::isnan(actual[i])) << i;
            else
                EXPECT_EQ(std::bit_cast<uint32_t>(expected[i]), std::bit_cast<uint32_t>(actual[i])) << i;
        }
    }

    class BatchOnly : public Reward {
    public:
        int calls = 0;
        std::vector<float> GetAllRewards(const GameState& state, bool final) override {
            calls++;
            return std::vector<float>(state.players.size(), final ? 8.f : -3.f);
        }
    };

    class DerivedSpeed : public SpeedReward {
    public:
        std::vector<float> GetAllRewards(const GameState& state, bool) override {
            return std::vector<float>(state.players.size(), 91.f);
        }
    };

    class DerivedPlayerSpeed : public SpeedReward {
    public:
        float GetReward(const Player&, const GameState&, bool) override { return 17.f; }
    };

    class DifferentBatch : public BatchOnly {
    public:
        float GetReward(const Player&, const GameState&, bool) override { return 2.f; }
    };

    template<typename Base>
    class DerivedBatch : public Base {
    public:
        using Base::Base;
        std::vector<float> GetAllRewards(const GameState& state, bool) override {
            return std::vector<float>(state.players.size(), 53.f);
        }
    };

    class RawReward : public PerPlayerReward<RawReward> {
    public:
        float GetReward(const Player& player, const GameState&, bool final) override {
            return player.boost * (final ? -1.f : 1.f);
        }
    };

    // Independent reference implementation of the zero-sum arithmetic, including player order.
    std::vector<float> ReferenceZeroSum(std::vector<float> values, const GameState& state, ZeroSumSettings settings) {
        float aggregate[2] = {};

        for (int team = 0; team < 2; team++) {
            int count = 0;
            for (size_t i = 0; i < values.size(); i++) {
                if ((int)state.players[i].team != team)
                    continue;
                if (settings.agg == AggregationMethod::Max)
                    aggregate[team] = (count == 0 || aggregate[team] < values[i]) ? values[i] : aggregate[team];
                else
                    aggregate[team] += values[i];
                count++;
            }

            if (settings.agg == AggregationMethod::Mean)
                aggregate[team] /= (float)count;
        }

        for (size_t i = 0; i < values.size(); i++) {
            int team = (int)state.players[i].team;
            values[i] = values[i] * (1 - settings.teamSpirit)
                + aggregate[team] * settings.teamSpirit - aggregate[1 - team] * settings.opponentScale;
        }

        return values;
    }

    class SharedParts : public MultiPartReward {
    public:
        int evaluations = 0;
        void GetPartRewards(const Player& player, const GameState&, bool final, std::span<float> out) override {
            evaluations++;
            out[CONT] = player.boost;
            if (final)
                out[EVENT] = -player.boost;
        }
    };
}

TEST(RewardEvaluation, LegacyBatchOverrideAndFurtherSubclassesKeepDispatch) {
    GameState state = MakeRewardState(4);
    std::vector<float> out(19, 999.f);

    BatchOnly batch;
    batch.GetAllRewardsInto(state, false, out);
    EXPECT_EQ(out, std::vector<float>(4, -3.f));
    batch.GetAllRewardsInto(state, true, out);
    EXPECT_EQ(out, std::vector<float>(4, 8.f));
    EXPECT_EQ(batch.calls, 2);

    DerivedSpeed speed;
    speed.GetAllRewardsInto(state, false, out);
    EXPECT_EQ(out, std::vector<float>(4, 91.f));

    DerivedPlayerSpeed perPlayer;
    perPlayer.GetAllRewardsInto(state, false, out);
    EXPECT_EQ(out, std::vector<float>(4, 17.f));
}

TEST(RewardEvaluation, PlainWrapperRetainsPerPlayerSemanticsAndSubclassBatchOverrides) {
    GameState state = MakeRewardState(2);
    RewardWrapper wrapper(new DifferentBatch());
    std::vector<float> out;
    wrapper.GetAllRewardsInto(state, false, out);
    EXPECT_EQ(out, std::vector<float>(2, 2.f));

    const float* storage = out.data();
    wrapper.GetAllRewardsInto(state, true, out);
    EXPECT_EQ(storage, out.data());

    DerivedBatch<RewardWrapper> derived(new DifferentBatch());
    derived.GetAllRewardsInto(state, false, out);
    EXPECT_EQ(out, std::vector<float>(2, 53.f));

    DerivedBatch<ZeroSumReward> zero(new DifferentBatch(), 0.5f);
    zero.GetAllRewardsInto(state, false, out);
    EXPECT_EQ(out, std::vector<float>(2, 53.f));
}

TEST(RewardEvaluation, StockRewardsMatchReturningApiAndReuseStorage) {
    std::vector<std::unique_ptr<Reward>> rewards;
    rewards.emplace_back(new GoalReward());
    rewards.emplace_back(new VelocityReward());
    rewards.emplace_back(new VelocityReward(true));
    rewards.emplace_back(new VelocityBallToGoalReward());
    rewards.emplace_back(new VelocityBallToGoalReward(true));
    rewards.emplace_back(new VelocityPlayerToBallReward());
    rewards.emplace_back(new FaceBallReward());
    rewards.emplace_back(new TouchBallReward());
    rewards.emplace_back(new SpeedReward());
    rewards.emplace_back(new WavedashReward());
    rewards.emplace_back(new PickupBoostReward());
    rewards.emplace_back(new SaveBoostReward());
    rewards.emplace_back(new AirReward());
    rewards.emplace_back(new TouchAccelReward());
    rewards.emplace_back(new StrongTouchReward());
    rewards.emplace_back(new PlayerGoalReward());
    rewards.emplace_back(new AssistReward());
    rewards.emplace_back(new ShotReward());
    rewards.emplace_back(new ShotPassReward());
    rewards.emplace_back(new SaveReward());
    rewards.emplace_back(new BumpReward());
    rewards.emplace_back(new BumpedPenalty());
    rewards.emplace_back(new DemoReward());
    rewards.emplace_back(new DemoedPenalty());

    std::mt19937 random(371);
    std::uniform_real_distribution<float> position(-4000.f, 4000.f), boost(0.f, 100.f);
    std::vector<float> out;
    out.reserve(6);
    const float* storage = out.data();

    for (int step = 0; step < 100; step++) {
        GameState previous = MakeRewardState(6);
        GameState state = MakeRewardState(step % 7);
        state.prev = step % 3 ? &previous : nullptr;
        state.ball.pos = Vec(position(random), position(random), 400.f);
        state.ball.vel = Vec(position(random), position(random), 100.f);
        previous.ball.vel = Vec(0.f, 400.f, 0.f);
        state.goalScored = step % 2;

        for (size_t i = 0; i < state.players.size(); i++) {
            Player& player = state.players[i];
            player.prev = step % 3 ? &previous.players[i] : nullptr;
            player.pos = Vec(position(random), position(random), 300.f);
            player.vel = Vec(position(random), position(random), 100.f);
            player.boost = boost(random);
            player.isOnGround = (step + i) % 2;
            player.ballTouchedStep = (step + i) % 3;
            previous.players[i].boost = boost(random);
            previous.players[i].isFlipping = true;
            previous.players[i].isOnGround = false;
            player.eventState.goal = player.eventState.save = player.eventState.assist = step % 2;
            player.eventState.shot = player.eventState.shotPass = player.eventState.bump = step % 2;
            player.eventState.bumped = player.eventState.demo = player.eventState.demoed = step % 2;
        }

        for (std::unique_ptr<Reward>& reward : rewards) {
            SCOPED_TRACE(reward->GetName());
            std::vector<float> expected = reward->GetAllRewards(state, step % 2);
            reward->GetAllRewardsInto(state, step % 2, out);
            ExpectSame(expected, out);
            EXPECT_EQ(storage, out.data());
        }
    }
}

TEST(RewardEvaluation, ZeroSumMatchesOriginalArithmeticIncludingRawDebugAndNestedWrappers) {
    std::mt19937 random(927);
    std::uniform_real_distribution<float> value(-200.f, 200.f);

    for (AggregationMethod aggregation : { AggregationMethod::Mean, AggregationMethod::Max }) {
        ZeroSumSettings settings{0.35f, 0.8f, aggregation};
        ZeroSumReward reward(new RawReward(), settings);
        ZeroSumReward nested(new ZeroSumReward(new RawReward(), settings), 0.7f);
        std::vector<float> out;
        out.reserve(6);
        const float* storage = out.data();

        for (int step = 0; step < 140; step++) {
            GameState state = MakeRewardState(step % 7);
            std::vector<float> raw;
            for (Player& player : state.players) {
                player.boost = value(random);
                if (step % 11 == 0) player.boost = -0.f;
                if (step % 13 == 0) player.boost = std::numeric_limits<float>::quiet_NaN();
                if (step % 17 == 0) player.boost = std::numeric_limits<float>::infinity();
                raw.push_back(player.boost * (step % 2 ? -1.f : 1.f));
            }

            std::vector<float> expected = ReferenceZeroSum(raw, state, settings);
            reward.GetAllRewardsInto(state, step % 2, out);
            ExpectSame(expected, out);
            ExpectSame(raw, reward._lastRewards);
            EXPECT_EQ(storage, out.data());

            nested.GetAllRewardsInto(state, step % 2, out);
            ExpectSame(ReferenceZeroSum(expected, state, nested.settings), out);
            ExpectSame(expected, nested._lastRewards);
            EXPECT_EQ(storage, out.data());
        }
    }
}

TEST(RewardEvaluation, ZeroSumRetainsBatchOnlyChildrenAndDerivedBaseCalls) {
    GameState state = MakeRewardState(2);
    ZeroSumReward wrapped(new BatchOnly(), 0.f, 0.5f);
    std::vector<float> out;
    wrapped.GetAllRewardsInto(state, false, out);
    EXPECT_EQ(out, std::vector<float>(2, -1.5f));

    class DelegatingWrapper : public ZeroSumReward {
    public:
        using ZeroSumReward::ZeroSumReward;
        std::vector<float> GetAllRewards(const GameState& gameState, bool final) override {
            std::vector<float> values = ZeroSumReward::GetAllRewards(gameState, final);
            for (float& value : values) value += 1.f;
            return values;
        }
    } delegated(new BatchOnly(), 0.f, 0.5f);

    delegated.GetAllRewardsInto(state, false, out);
    EXPECT_EQ(out, std::vector<float>(2, -0.5f));
}

TEST(RewardEvaluation, SharedPartsReuseCalculationAndCallerBufferAcrossWrappersAndResets) {
    auto owner = std::make_shared<SharedParts>();
    std::unique_ptr<Reward> continuous(owner->Part(MultiPartReward::CONT));
    ZeroSumReward event(owner->Part(MultiPartReward::EVENT), 0.f);
    std::vector<float> out;
    out.reserve(6);
    const float* storage = out.data();

    for (int step = 0; step < 12; step++) {
        GameState state = MakeRewardState(2 + 2 * (step % 3));
        for (Player& player : state.players) player.boost = (float)player.carId;

        // Reuse tick zero; PreStep, finality, and Reset must still invalidate the shared cache.
        continuous->Reset(state);
        static_cast<Reward&>(event).Reset(state);
        continuous->PreStep(state);
        static_cast<Reward&>(event).PreStep(state);
        int before = owner->evaluations;

        for (bool final : {false, true, false}) {
            continuous->GetAllRewardsInto(state, final, out);
            std::vector<float> raw;
            for (Player& player : state.players) raw.push_back(player.boost);
            ExpectSame(raw, out);

            event.GetAllRewardsInto(state, final, out);
            for (float& value : raw) value = final ? -value : 0.f;
            ExpectSame(raw, event._lastRewards);
            ExpectSame(ReferenceZeroSum(raw, state, event.settings), out);
            EXPECT_EQ(storage, out.data());
        }

        EXPECT_EQ(owner->evaluations - before, 3 * state.players.size());
    }

    DerivedBatch<MultiPartRewardView> derived(owner, 0, "custom");
    derived.GetAllRewardsInto(MakeRewardState(2), false, out);
    EXPECT_EQ(out, std::vector<float>(2, 53.f));
}
