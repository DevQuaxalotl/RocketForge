#include <gtest/gtest.h>
#include "TestSupport.h"

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/DefaultObsPadded.h>

#include <bit>
#include <memory>
#include <stdexcept>

using namespace RLGC;

namespace {
    GameState SampleState(int count, int step) {
        GameState state;
        state.lastTickCount = step;
        state.ball.pos = Vec(100.f + step, -200.f, 300.f);
        state.ball.vel = Vec(-500.f, 200.f + step, 100.f);
        state.ball.angVel = Vec(0.1f, -0.2f, 0.3f);
        for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
            state.boostPads[i] = (i + step) % 2;
            state.boostPadsInv[i] = (i + step) % 3;
            state.boostPadTimers[i] = i * 0.1f;
            state.boostPadTimersInv[i] = i * 0.2f;
        }

        for (int i = 0; i < count; i++) {
            Player player = {};
            player.index = i;
            player.carId = i + 1;
            player.team = i % 2 ? Team::ORANGE : Team::BLUE;
            player.pos = Vec(100.f * i, -500.f + step, 20.f);
            player.vel = Vec(200.f + step, -120.f * i, 80.f);
            player.angVel = Vec(0.2f * i, -0.4f, 0.8f);
            player.boost = (float)((i * 17 + step) % 101);
            player.isOnGround = (i + step) % 2;
            player.isDemoed = (i + step) % 5 == 0;
            player.hasJumped = (i + step) % 3 == 0;
            player.prevAction = { 0.2f, -0.3f, 0.4f, -0.5f, 0.6f, 1.f, 0.f, 1.f };
            state.players.push_back(player);
        }

        return state;
    }

    void ExpectExact(const FList& expected, const FList& actual) {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t i = 0; i < expected.size(); i++)
            ASSERT_EQ(std::bit_cast<uint32_t>(expected[i]), std::bit_cast<uint32_t>(actual[i])) << i;
    }

    template<typename Base>
    struct CustomBuilder : Base {
        FList BuildObs(const Player&, const GameState&) override { return { 91.f, -7.f }; }
    };

    struct StatefulDefault : DefaultObs {
        std::vector<uint32_t> calls;
        void AddPlayerToObs(FList& out, const Player& player, bool) override {
            calls.push_back(player.carId);
            out.push_back((float)player.carId + (float)out.size());
        }
    };

    struct StatefulAdvanced : AdvancedObs {
        std::vector<uint32_t> calls;
        void AddPlayerToObs(FList& out, const Player& player, bool, const PhysState&) override {
            calls.push_back(player.carId);
            out.push_back((float)player.carId + (float)out.size());
        }
    };

    struct ProbeObs : ObsBuilder {
        bool firstWasEmpty = true;
        bool failNext = false;
        const float* storage = nullptr;

        FList BuildObs(const Player& player, const GameState& state) override {
            return { (float)player.carId, (float)state.lastTickCount, player.prevAction.throttle };
        }

        void BuildObsInto(const Player& player, const GameState& state, FList& out) override {
            if (player.index == 0 && !out.empty()) firstWasEmpty = false;

            if (failNext) {
                failNext = false;
                out.assign(64, 999.f);
                throw std::runtime_error("synthetic observation failure");
            }

            out.resize(3);
            out[0] = (float)player.carId;
            out[1] = (float)state.lastTickCount;
            out[2] = player.prevAction.throttle;
            storage = out.data();
        }
    };

    struct ProbeParser : ActionParser {
        bool useMask = true;
        bool overrideTicks = false;
        bool masksStartedEmpty = true;
        bool macrosStartedEmpty = true;
        int overrideCalls = 0;
        const uint8_t* maskStorage = nullptr;
        std::vector<const ActionMacro*> macroStorage;

        Action ParseAction(int action, const Player&, const GameState&) override {
            return { action == 2 ? -0.5f : 0.25f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        }

        void ParseActionMacros(
            int action,
            const Player& player,
            const GameState& state,
            int ticks,
            std::vector<ActionMacro>& out) override {
            macrosStartedEmpty = macrosStartedEmpty && out.empty();

            if (action == 1) {
                out.push_back({ Action{}, 0 });
                out.push_back({ Action{}, ticks / 2 });
                out.push_back({ ParseAction(action, player, state), ticks - ticks / 2 });
            } else if (action != 2) {
                out.push_back({ ParseAction(action, player, state), ticks });
            }
            // Action 2 intentionally appends nothing, exercising EnvSet's ordinary-action fallback.

            macroStorage.resize(state.players.size());
            macroStorage[player.index] = out.data();
        }

        int GetActionAmount() override { return 4; }
        bool UsesActionMask() override { return useMask; }

        void GetActionMaskInto(const Player& player, const GameState& state, std::vector<uint8_t>& out) override {
            if (player.index == 0 && !out.empty()) masksStartedEmpty = false;
            out.assign(4, false);
            out[(player.index + state.lastTickCount) % 4] = true;
            maskStorage = out.data();
        }

        bool HasRuntimeActionOverride() const override { return overrideTicks; }
        bool RequiresPerTickActionOverride(const Player&, const GameState&) override { return overrideTicks; }

        Action OverrideRuntimeAction(const Action& action, const Player&, const GameState&, int) override {
            overrideCalls++;
            Action result = action;
            result.throttle *= 0.5f;
            return result;
        }
    };

    struct FixedSetter : StateSetter {
        void ResetArena(Arena* arena) override {
            int i = 0;
            for (Car* car : arena->_cars) {
                CarState state = car->GetState();
                state.pos = Vec(500.f * i++, -1500.f, 20.f);
                state.vel = Vec(0, 0, 0);
                car->SetState(state);
            }
        }
    };

    struct BufferEnvironment {
        std::vector<std::unique_ptr<ProbeObs>> obs;
        std::vector<std::unique_ptr<ProbeParser>> parsers;
        std::vector<std::unique_ptr<FixedSetter>> setters;
        std::unique_ptr<EnvSet> env;

        explicit BufferEnvironment(int delay = 0) {
            for (int i = 0; i < 3; i++) {
                obs.emplace_back(new ProbeObs());
                parsers.emplace_back(new ProbeParser());
                setters.emplace_back(new FixedSetter());
            }

            EnvSetConfig config = {};
            config.numArenas = 3;
            config.tickSkip = 8;
            config.actionDelay = delay;
            config.envCreateCfg = EnvCreateConfig(3, (uint8_t)GameMode::SOCCAR);
            config.envCreateFn = [this](int index, const EnvCreateConfig& createConfig) {
                EnvCreateResult result = {};
                result.arena = Arena::Create(createConfig.mode);
                for (int p = 0; p < 2 * (index + 1); p++)
                    result.arena->AddCar(p % 2 ? Team::ORANGE : Team::BLUE);

                result.obsBuilder = obs[index].get();
                result.actionParser = parsers[index].get();
                result.stateSetter = setters[index].get();
                return result;
            };

            env = std::make_unique<EnvSet>(config);
        }

        void CheckRows() {
            for (int arena = 0; arena < 3; arena++) {
                const GameState& state = env->state.gameStates[arena];
                for (const Player& player : state.players) {
                    int row = env->state.arenaPlayerStartIdx[arena] + player.index;
                    ExpectExact(obs[arena]->BuildObs(player, state), env->state.obs.GetRow(row));

                    auto mask = env->state.actionMasks.GetRow(row);
                    for (int action = 0; action < 4; action++)
                        EXPECT_EQ(mask[action], !parsers[arena]->useMask || action == (player.index + state.lastTickCount) % 4);
                }

                EXPECT_TRUE(obs[arena]->firstWasEmpty);
                EXPECT_TRUE(parsers[arena]->masksStartedEmpty);
                EXPECT_TRUE(parsers[arena]->macrosStartedEmpty);
            }
        }
    };
}

TEST(SimulationBuffers, StockObservationValuesAndOrderingMatchExactlyWithReusableCapacity) {
    DefaultObs standard;
    AdvancedObs advanced;

    for (ObsBuilder* builder : { static_cast<ObsBuilder*>(&standard), static_cast<ObsBuilder*>(&advanced) }) {
        FList out;
        out.reserve(1000);
        const float* storage = out.data();

        for (int step = 0; step < 30; step++) {
            GameState state = SampleState(step % 6 + 1, step);
            for (const Player& player : state.players) {
                FList expected = builder->BuildObs(player, state);
                out.assign(1000, 999.f);
                builder->BuildObsInto(player, state, out);
                ExpectExact(expected, out);
                EXPECT_EQ(storage, out.data());
            }
        }
    }
}

TEST(SimulationBuffers, LiveStockObservationsMatchReturningApiAcrossStepsAndResets) {
    ROCKETFORGE_REQUIRE_ROCKETSIM();
    DefaultObs standard;
    AdvancedObs advanced;
    const bool async = false;

    for (ObsBuilder* builder : { static_cast<ObsBuilder*>(&standard), static_cast<ObsBuilder*>(&advanced) }) {
        for (int count : {2, 4, 6}) {
            ProbeParser parser;
            FixedSetter setter;
            EnvSetConfig config = {};
            config.numArenas = 1;
            config.tickSkip = 8;
            config.actionDelay = 0;
            config.envCreateCfg = EnvCreateConfig(1, (uint8_t)GameMode::SOCCAR);
            config.envCreateFn = [&](int, const EnvCreateConfig& createConfig) {
                EnvCreateResult result = {};
                result.arena = Arena::Create(createConfig.mode);
                for (int p = 0; p < count; p++)
                    result.arena->AddCar(p % 2 ? Team::ORANGE : Team::BLUE);

                result.obsBuilder = builder;
                result.actionParser = &parser;
                result.stateSetter = &setter;
                return result;
            };

            EnvSet env(config);
            IList actions(count, 0);

            for (int step = 0; step < 32; step++) {
                if (step == 16) env.ResetArena(0);
                for (int p = 0; p < count; p++) actions[p] = (step + p) % 3;
                env.StepSecondHalf(actions, async);

                const GameState& state = env.state.gameStates[0];
                for (const Player& player : state.players)
                    ExpectExact(builder->BuildObs(player, state), env.state.obs.GetRow(player.index));
            }
        }
    }
}

TEST(SimulationBuffers, CustomObservationOverridesAndStatefulPlayerHooksKeepLegacyBehavior) {
    GameState state = SampleState(6, 7);
    FList out;
    CustomBuilder<DefaultObs> customDefault;
    CustomBuilder<AdvancedObs> customAdvanced;
    customDefault.BuildObsInto(state.players[1], state, out);
    EXPECT_EQ(out, (FList{91.f, -7.f}));
    customAdvanced.BuildObsInto(state.players[1], state, out);
    EXPECT_EQ(out, (FList{91.f, -7.f}));

    StatefulDefault legacyDefault, bufferedDefault;
    StatefulAdvanced legacyAdvanced, bufferedAdvanced;
    FList expected = legacyDefault.BuildObs(state.players[1], state);
    bufferedDefault.BuildObsInto(state.players[1], state, out);
    ExpectExact(expected, out);
    EXPECT_EQ(legacyDefault.calls, bufferedDefault.calls);

    expected = legacyAdvanced.BuildObs(state.players[1], state);
    bufferedAdvanced.BuildObsInto(state.players[1], state, out);
    ExpectExact(expected, out);
    EXPECT_EQ(legacyAdvanced.calls, bufferedAdvanced.calls);
}

TEST(SimulationBuffers, PaddedObservationFallbackPreservesShuffleAndRandomGeneratorState) {
    DefaultObsPadded padded(3);

    for (int count : {2, 4, 6}) {
        GameState state = SampleState(count, 3);
        const std::default_random_engine initial = ::Math::GetRandEngine();
        const FList expected = padded.BuildObs(state.players[0], state);
        const std::default_random_engine final = ::Math::GetRandEngine();
        ::Math::GetRandEngine() = initial;

        FList actual;
        padded.BuildObsInto(state.players[0], state, actual);
        ExpectExact(expected, actual);
        EXPECT_TRUE(::Math::GetRandEngine() == final);
    }
}

TEST(SimulationBuffers, ParallelArenasReuseBuffersAcrossMasksMacrosFallbacksAndResets) {
    ROCKETFORGE_REQUIRE_ROCKETSIM();
    BufferEnvironment fixture;

    // Warm the maximum macro count before checking retained storage.
    IList actions(fixture.env->state.numPlayers, 1);
    const bool warmupAsync = false;
    fixture.env->StepSecondHalf(actions, warmupAsync);

    std::vector<std::vector<const ActionMacro*>> macroPointers;
    for (std::unique_ptr<ProbeParser>& parser : fixture.parsers) macroPointers.push_back(parser->macroStorage);

    std::vector<const float*> obsPointers;
    std::vector<const uint8_t*> maskPointers;
    for (int i = 0; i < 3; i++) {
        obsPointers.push_back(fixture.obs[i]->storage);
        maskPointers.push_back(fixture.parsers[i]->maskStorage);
    }

    const bool async = true;
    for (int step = 0; step < 12; step++) {
        for (std::unique_ptr<ProbeParser>& parser : fixture.parsers) parser->useMask = step % 3 != 1;

        if (step == 6) {
            for (int arena = 0; arena < 3; arena++) fixture.env->ResetArena(arena);
        }

        int action = step % 3;
        actions.assign(actions.size(), action);
        fixture.env->StepSecondHalf(actions, async);
        fixture.env->Sync();
        fixture.CheckRows();

        for (int arena = 0; arena < 3; arena++) {
            EXPECT_EQ(obsPointers[arena], fixture.obs[arena]->storage);
            EXPECT_EQ(maskPointers[arena], fixture.parsers[arena]->maskStorage);
            EXPECT_EQ(macroPointers[arena], fixture.parsers[arena]->macroStorage);
            for (const Player& player : fixture.env->state.gameStates[arena].players)
                EXPECT_FLOAT_EQ(player.prevAction.throttle, action == 2 ? -0.5f : 0.25f);
        }
    }

    EXPECT_NE(obsPointers[0], obsPointers[1]);
    EXPECT_NE(maskPointers[0], maskPointers[1]);
    EXPECT_NE(macroPointers[0][0], macroPointers[1][0]);
}

TEST(SimulationBuffers, PerPlayerOverridesAndScheduledDelaysKeepTheirTimingAcrossReuse) {
    ROCKETFORGE_REQUIRE_ROCKETSIM();
    const int actionDelay = 3;
    BufferEnvironment fixture(actionDelay);
    std::vector<ActionParser*> playerParsers;
    for (int arena = 0; arena < 3; arena++) {
        fixture.parsers[arena]->overrideTicks = true;
        for (const Player& player : fixture.env->state.gameStates[arena].players)
            playerParsers.push_back(fixture.parsers[arena].get());
    }

    IList actions(fixture.env->state.numPlayers, 0), applyTicks(actions.size(), 1);
    const bool async = false;

    for (int step = 0; step < 6; step++) {
        if (step == 3) {
            for (int arena = 0; arena < 3; arena++) fixture.env->ResetArena(arena);
        }

        if (step % 2)
            fixture.env->StepFirstHalf(async, playerParsers, actions, applyTicks);
        else
            fixture.env->StepFirstHalf(async, playerParsers);

        fixture.env->StepSecondHalf(actions, playerParsers, async);
        fixture.CheckRows();

        for (int arena = 0; arena < 3; arena++) {
            int playerCount = 2 * (arena + 1);
            EXPECT_EQ(fixture.parsers[arena]->overrideCalls, (step + 1) * playerCount * 8);
            for (const Player& player : fixture.env->state.gameStates[arena].players)
                EXPECT_FLOAT_EQ(player.prevAction.throttle, 0.125f);
        }
    }
}

TEST(SimulationBuffers, ObservationFailureDoesNotLeaveStaleRowsOnTheNextBuild) {
    ROCKETFORGE_REQUIRE_ROCKETSIM();
    BufferEnvironment fixture;
    fixture.obs[1]->failNext = true;
    EXPECT_THROW(fixture.env->BuildArenaObs(1), std::runtime_error);
    fixture.env->BuildArenaObs(1);
    fixture.CheckRows();
}
