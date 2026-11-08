#include <gtest/gtest.h>
#include "TestSupport.h"
#include "Runtime/RLBotClient.h"
#include "Runtime/RLBotCapture.h"
#include "Runtime/RunRLBot.h"
#include <GigaLearnInfer/AgentRegistry.h>
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include "Development/RLBotDevelopmentExtension.h"
#include "Runtime/RLBotPolicyRuntime.h"
#include <GigaLearnInfer/InferUnit.h>
#include <GigaLearnInfer/ModelManager.h>
#include <GigaLearnInfer/Models.h>
#include <ATen/Parallel.h>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <optional>
#include <mutex>

using namespace RLGC;
namespace flat = rlbot::flat;

namespace {
    bool TestGPU() { return std::getenv("ROCKETFORGE_TEST_GPU") != nullptr; }

    struct TestObs final : ObsBuilder {
        FList BuildObs(const Player& player, const GameState& state) override {
            float total = 0;
            for (const Player& other : state.players) total += other.lastControls.throttle;
            return {static_cast<float>(player.index), player.prevAction.throttle, player.lastControls.throttle, total};
        }
    };

    struct TestActions final : ActionParser {
        int GetActionAmount() override { return 3; }

        std::vector<uint8_t> GetActionMask(const Player& player, const GameState&) override {
            std::vector<uint8_t> mask(3, 0);
            mask.at(player.index % 3) = 1;
            return mask;
        }

        Action ParseAction(int action, const Player&, const GameState&) override {
            Action result{};
            result.throttle = (action + 1) / 3.f;
            return result;
        }

        void ParseActionMacros(
            int action,
            const Player& player,
            const GameState& state,
            int ticks,
            std::vector<ActionMacro>& out) override {
            Action first = ParseAction(action, player, state);
            Action second = first;
            second.throttle = -first.throttle;
            out = {{first, 2}, {second, ticks - 2}};
        }

        bool IsMacroAction(int, const Player&, const GameState&) override { return true; }
    };

    struct TestAgent final : Agent {
        mutable int modelFactoryCalls = 0;
        mutable bool lastUsedGPU = false;

        TestAgent() {
            // Batching doesn't depend on the engine; v2 is the one every build has.
            trainingEngineVersion = 2;
            tickSkip = 4;
            actionDelay = 0;
            maxTeamSize = 3;
            sharedHead.layerSizes = {8};
            sharedHead.addOutputLayer = false;
            policy.layerSizes = {8};
        }

        ObsBuilder* MakeObsBuilder() const override { return new TestObs; }
        ActionParser* MakeActionParser() const override { return new TestActions; }
        int GetObsSize() const override { return 4; }

        void MakeInferenceModels(int obsSize, int numActions, const c10::Device& device, GGL::ModelSet& models) const override {
            ++modelFactoryCalls;
            lastUsedGPU = device.is_cuda();
            Agent::MakeInferenceModels(obsSize, numActions, device, models);
        }
    };

    struct Capture final : RLBotClientExtension, RLBotPolicyObserver {
        int calls = 0;
        std::vector<unsigned> indices;
        std::vector<int> actions;
        std::vector<float> obs;

        RLBotPolicyObserver* GetPolicyObserver() override { return this; }
        bool CapturePolicyDiagnostics() const override { return true; }
        void OnPolicyRuntimeReady(double) override {}
        void OnPolicyInference(int, double, const GGL::InferUnit&) override {}

        void OnPolicyBatchInference(
            const std::vector<unsigned>& rows,
            const std::vector<int>& result,
            double,
            const GGL::InferUnit& unit) override {
            ++calls;
            indices = rows;
            actions = result;
            obs = unit.lastObs;
        }
    };

    struct Fixture {
        std::shared_ptr<TestAgent> agent = std::make_shared<TestAgent>();
        std::filesystem::path checkpoint = RocketForgeTests::TestTempDir("model");

        Fixture() {
            static std::once_flag init;
            std::call_once(init, [] {
                at::set_num_threads(1);
                RocketSim::Init(RocketSim::CollisionMeshesFolder(), true);
            });

            std::filesystem::create_directories(checkpoint);
            GGL::ModelSet models;
            GGL::ModelManager::MakeModels(false, 4, 3, agent->sharedHead, agent->policy, {}, torch::kCPU, models);
            models.Save(checkpoint);
            models.Free();
        }

        std::shared_ptr<SharedBotContext> Context(Capture*& capture) {
            std::shared_ptr<SharedBotContext> context = std::make_shared<SharedBotContext>();
            context->agent = agent;
            context->useGPU = TestGPU();
            context->modelsFolder = checkpoint;
            context->extensionFactory = [&capture](unsigned, const std::unordered_set<unsigned>&) {
                std::unique_ptr<Capture> extension = std::make_unique<Capture>();
                capture = extension.get();
                return extension;
            };

            return context;
        }
    };

    void Step(RLBotBot& bot, int tick, int count = 6) {
        flat::GamePacketT packet;
        packet.match_info = std::make_unique<flat::MatchInfoT>();
        packet.match_info->seconds_elapsed = tick / 120.f;
        packet.match_info->frame_num = tick;
        packet.boost_pads.assign(34, flat::BoostPadState(true, 0.f));

        std::unique_ptr<flat::BallInfoT> ball = std::make_unique<flat::BallInfoT>();
        ball->physics = std::make_unique<flat::Physics>();
        ball->shape.Set(flat::SphereShapeT{});
        packet.balls.push_back(std::move(ball));

        for (int i = 0; i < count; ++i) {
            std::unique_ptr<flat::PlayerInfoT> player = std::make_unique<flat::PlayerInfoT>();
            player->physics = std::make_unique<flat::Physics>();
            player->score_info = std::make_unique<flat::ScoreInfo>();
            player->hitbox = std::make_unique<flat::BoxShapeT>();
            player->hitbox_offset = std::make_unique<flat::Vector3>();
            player->last_input = std::make_unique<flat::ControllerState>();
            player->dodge_dir = std::make_unique<flat::Vector2>();
            player->player_id = i + 1;
            player->team = i < 3 ? 0 : 1;
            player->demolished_timeout = -1;
            packet.players.push_back(std::move(player));
        }

        flatbuffers::FlatBufferBuilder buffer;
        buffer.Finish(flat::GamePacket::Pack(buffer, &packet));
        bot.update(flatbuffers::GetRoot<flat::GamePacket>(buffer.GetBufferPointer()), nullptr);
    }
}

TEST(RLBotHivemind, RowsMasksAndSingletonParity) {
    ROCKETFORGE_REQUIRE_COLLISION_MESHES();
    Fixture fixture;
    Capture capture;
    const unsigned warmupBatchSize = 3;
    RLBotPolicyRuntime runtime(*fixture.agent, fixture.checkpoint, TestGPU(), &capture, warmupBatchSize);

    GameState state;
    state.players.resize(6);
    for (unsigned i = 0; i < 6; ++i) {
        state.players[i].index = i;
        state.players[i].prevAction.throttle = (i + 1) / 10.f;
    }

    runtime.PrepareObservationState(state);
    const std::vector<unsigned> indices{5, 1, 3};
    const std::vector<int> batch = runtime.InferBatch(indices, state);
    ASSERT_EQ(batch, (std::vector<int>{2, 1, 0}));
    ASSERT_EQ(fixture.agent->modelFactoryCalls, 1);
    ASSERT_EQ(fixture.agent->lastUsedGPU, TestGPU());
    ASSERT_EQ(capture.indices, indices);
    ASSERT_EQ(capture.obs.size(), size_t(12));

    for (size_t row = 0; row < indices.size(); ++row) {
        ASSERT_EQ(capture.obs[row * 4], static_cast<float>(indices[row]));
        ASSERT_EQ(capture.obs[row * 4 + 1], state.players[indices[row]].prevAction.throttle);
        ASSERT_EQ(runtime.Infer(state.players[indices[row]], state), batch[row]);
    }

    ASSERT_TRUE(runtime.InferBatch({}, state).empty());
    ASSERT_EQ(capture.calls, 1);

    bool rejected = false;
    try {
        runtime.InferBatch({6}, state);
    } catch (const std::out_of_range&) {
        rejected = true;
    }

    ASSERT_TRUE(rejected);
}

TEST(RLBotHivemind, LiveBatchPreservesMacrosAndPreviousActions) {
    ROCKETFORGE_REQUIRE_COLLISION_MESHES();
    Fixture fixture;
    Capture* capture = nullptr;
    RLBotBot bot({2, 0, 1}, 0, "test", fixture.Context(capture));

    Step(bot, 0);
    ASSERT_EQ(capture->calls, 1);
    ASSERT_EQ(capture->indices, (std::vector<unsigned>{0, 1, 2}));
    for (unsigned i = 0; i < 3; ++i) ASSERT_EQ(bot.getOutput(i).throttle(), (i + 1) / 3.f);

    Step(bot, 2);
    ASSERT_EQ(capture->calls, 1);
    for (unsigned i = 0; i < 3; ++i) ASSERT_EQ(bot.getOutput(i).throttle(), -static_cast<float>(i + 1) / 3.f);

    Step(bot, 4);
    ASSERT_EQ(capture->calls, 2);
    for (unsigned i = 0; i < 3; ++i) {
        ASSERT_EQ(capture->obs[i * 4 + 1], -static_cast<float>(i + 1) / 3.f);
        ASSERT_EQ(capture->obs[i * 4 + 2], -static_cast<float>(i + 1) / 3.f);
    }

    for (unsigned i = 0; i < 3; ++i) ASSERT_FLOAT_EQ(capture->obs[i * 4 + 3], -2.f);

    Step(bot, 8, 2);
    ASSERT_EQ(capture->indices, (std::vector<unsigned>{0, 1}));
    ASSERT_EQ(bot.getOutput(2).throttle(), 0.f);
}

TEST(RLBotHivemind, LiveBatchPreservesDelayedActions) {
    ROCKETFORGE_REQUIRE_COLLISION_MESHES();
    Fixture fixture;
    fixture.agent->actionDelay = 3;
    Capture* capture = nullptr;
    RLBotBot bot({5, 3, 4}, 1, "test", fixture.Context(capture));

    Step(bot, 0);
    for (unsigned i = 3; i < 6; ++i) ASSERT_EQ(bot.getOutput(i).throttle(), 0.f);

    Step(bot, 2);
    ASSERT_EQ(capture->calls, 1);

    Step(bot, 3);
    for (unsigned i = 3; i < 6; ++i) ASSERT_EQ(bot.getOutput(i).throttle(), (i % 3 + 1) / 3.f);

    Step(bot, 4);
    ASSERT_EQ(capture->calls, 2);
    for (unsigned i = 0; i < 3; ++i) {
        ASSERT_EQ(capture->obs[i * 4 + 1], (i + 1) / 3.f);
        ASSERT_EQ(capture->obs[i * 4 + 2], (i + 1) / 3.f);
    }
}

namespace {
    struct ScopedEnv {
        std::string name;
        std::optional<std::string> previous;

        static void Set(const std::string& name, const char* value) {
#ifdef _WIN32
            _putenv_s(name.c_str(), value ? value : "");
#else
            if (value)
                setenv(name.c_str(), value, 1);
            else
                unsetenv(name.c_str());
#endif
        }

        ScopedEnv(std::string key, const std::string& value) : name(std::move(key)) {
            if (const char* old = std::getenv(name.c_str())) previous = old;
            Set(name, value.c_str());
        }

        ~ScopedEnv() { Set(name, previous ? previous->c_str() : nullptr); }
    };

    std::vector<std::string> Split(const std::string& value, char delimiter) {
        std::vector<std::string> parts;
        std::istringstream stream(value);
        std::string part;
        while (std::getline(stream, part, delimiter)) parts.push_back(part);
        return parts;
    }
}

TEST(RLBotHivemind, BatchedControlsMatchSeparateCars) {
    ROCKETFORGE_REQUIRE_COLLISION_MESHES();
    Fixture fixture;

    for (int delay : {0, 3}) {
        fixture.agent->actionDelay = delay;
        Capture* capture = nullptr;
        RLBotBot batch({3, 4, 5}, 1, "batch", fixture.Context(capture));
        std::vector<std::unique_ptr<RLBotBot>> singles;
        for (unsigned i : {3, 4, 5})
            singles.push_back(std::make_unique<RLBotBot>(std::unordered_set<unsigned>{i}, 1, "single", fixture.Context(capture)));

        for (int tick = 0; tick <= 16; ++tick) {
            Step(batch, tick);
            for (unsigned row = 0; row < singles.size(); ++row) {
                Step(*singles[row], tick);
                ASSERT_EQ(batch.getOutput(row + 3).throttle(), singles[row]->getOutput(row + 3).throttle())
                    << "delay=" << delay << ", tick=" << tick << ", row=" << row;
            }
        }
    }
}

TEST(RLBotHivemind, DiagnosticsStayWithTheirCar) {
    ROCKETFORGE_REQUIRE_COLLISION_MESHES();
    Fixture fixture;
    const std::filesystem::path traces = RocketForgeTests::TestTempDir("trace");
    ScopedEnv enabled("ROCKETFORGE_RLBOT_TRACE", "1");
    ScopedEnv output("ROCKETFORGE_RLBOT_TRACE_PATH", (traces / "trace.csv").string());
    ScopedEnv diagnostics("ROCKETFORGE_RLBOT_TRACE_POLICY_DIAGNOSTICS", "1");
    ScopedEnv snapshots("ROCKETFORGE_RLBOT_TRACE_STATE_SNAPSHOTS", "0");

    {
        Capture* ignored = nullptr;
        std::shared_ptr<SharedBotContext> context = fixture.Context(ignored);
        context->extensionFactory = CreateRLBotDevelopmentExtension;
        RLBotBot bot({5, 3, 4}, 1, "trace", context);
        Step(bot, 0);
        Step(bot, 4);
    }

    std::ifstream trace(traces / "trace_team1_idx3_idx4_idx5.csv");
    ASSERT_TRUE(trace.good());
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(trace, line)));

    const std::vector<std::string> headers = Split(line, ',');
    const auto column = [&headers](const std::string& name) {
        const auto found = std::find(headers.begin(), headers.end(), name);
        EXPECT_NE(found, headers.end());
        return static_cast<size_t>(found - headers.begin());
    };

    const size_t indexColumn = column("bot_index"), obsColumn = column("obs_values"), maskColumn = column("mask_bits");
    const size_t probabilitiesColumn = column("action_probs"), actionColumn = column("selected_action_id");
    int rows = 0;

    while (std::getline(trace, line)) {
        const std::vector<std::string> fields = Split(line, ',');
        ASSERT_GT(fields.size(), probabilitiesColumn);

        const unsigned index = std::stoul(fields[indexColumn]);
        ASSERT_EQ(index, 3 + rows % 3);

        const std::vector<std::string> observation = Split(fields[obsColumn], ';');
        ASSERT_EQ(observation.size(), 4);
        ASSERT_EQ(std::stof(observation[0]), static_cast<float>(index));

        const std::vector<std::string> probabilities = Split(fields[probabilitiesColumn], ';');
        ASSERT_EQ(probabilities.size(), 3);
        ASSERT_FLOAT_EQ(std::stof(probabilities[index % 3]), 1.f);

        std::string mask(3, '0');
        mask[index % 3] = '1';
        ASSERT_EQ(fields[maskColumn], mask);
        ASSERT_EQ(std::stoi(fields[actionColumn]), index % 3);
        ++rows;
    }

    ASSERT_EQ(rows, 6);
}

TEST(RLBotHivemind, SingleCarObserversRemainCompatible) {
    ROCKETFORGE_REQUIRE_COLLISION_MESHES();
    Fixture fixture;

    struct SingleObserver final : RLBotPolicyObserver {
        int calls = 0;
        bool CapturePolicyDiagnostics() const override { return false; }
        void OnPolicyRuntimeReady(double) override {}
        void OnPolicyInference(int, double, const GGL::InferUnit&) override { ++calls; }
    } observer;

    RLBotPolicyRuntime runtime(*fixture.agent, fixture.checkpoint, TestGPU(), &observer);
    GameState state;
    state.players.resize(1);
    state.players[0].index = 0;
    runtime.PrepareObservationState(state);
    ASSERT_EQ(runtime.InferBatch({0}, state), (std::vector<int>{0}));
    ASSERT_EQ(observer.calls, 1);

    const unsigned emptyWarmupBatch = 0;
    EXPECT_THROW(RLBotPolicyRuntime(*fixture.agent, fixture.checkpoint, TestGPU(), nullptr, emptyWarmupBatch), std::invalid_argument);
}

TEST(RLBotCapture, RoundTripPreservesPolicyControlsAndTiming) {
    ROCKETFORGE_REQUIRE_COLLISION_MESHES();
    Fixture fixture;
    const std::filesystem::path output = RocketForgeTests::TestTempDir("capture");
    Capture* observer = nullptr;
    std::shared_ptr<SharedBotContext> context = fixture.Context(observer);
    context->captureDirectory = output;

    {
        RLBotBot bot({0, 1, 2}, 0, "capture", context);
        bot.initialize(nullptr, nullptr, nullptr);
        for (int tick = 0; tick < 17; ++tick) Step(bot, tick, tick == 8 ? 2 : 6);

        // Early-return callbacks must still record controls.
        bot.update(nullptr, nullptr);
        ASSERT_EQ(observer->calls, 5);
    }

    std::filesystem::path path;
    for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(output))
        if (entry.path().extension() == ".rfcapture") path = entry.path();
    ASSERT_FALSE(path.empty());

    std::vector<nlohmann::json> original, replay;
    nlohmann::json header = ReadRLBotCapture(path, false, [&](const auto& row) {
        if (row.at("kind") == "callback") original.push_back(row);
    });
    ASSERT_TRUE(header.at("complete"));
    ASSERT_EQ(header.at("callbacks"), 18);

    context->captureDirectory.clear();
    context->captureSink = [&](const auto& row) {
        if (row.at("kind") == "callback") replay.push_back(row);
    };
    ReplayRLBotCapture(path, context, false, false);
    ASSERT_EQ(original.size(), replay.size());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_EQ(original[i].at("policy"), replay[i].at("policy"));
        EXPECT_EQ(original[i].at("controls"), replay[i].at("controls"));
        EXPECT_EQ(original[i].at("schedule"), replay[i].at("schedule"));
    }

    // Single-car replay can change observations (teammates' previous controls),
    // but this synthetic policy's decisions must still match.
    replay.clear();
    ReplayRLBotCapture(path, context, true, false);
    ASSERT_EQ(replay.size(), original.size() * 3);
    for (size_t i = 0; i < original.size(); ++i)
        for (size_t car = 0; car < 3; ++car)
            EXPECT_EQ(original[i].at("controls")[car], replay[i * 3 + car].at("controls")[0]);

    // Incomplete files are usable only as explicitly requested verified prefixes.
    std::ifstream input(path, std::ios::binary);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) lines.push_back(line);

    const std::filesystem::path partial = output / "partial.rfcapture";
    {
        std::ofstream file(partial, std::ios::binary);
        for (size_t i = 0; i + 1 < lines.size(); ++i) file << lines[i] << '\n';
        file << "{incomplete";
    }

    EXPECT_THROW(ReadRLBotCapture(partial, false), std::runtime_error);
    EXPECT_FALSE(ReadRLBotCapture(partial, true).at("complete"));
    replay.clear();
    ReplayRLBotCapture(partial, context, false, true);
    EXPECT_EQ(replay.size(), original.size());

    const std::filesystem::path corrupt = output / "corrupt.rfcapture";
    {
        std::ofstream file(corrupt, std::ios::binary);
        nlohmann::json envelope = nlohmann::json::parse(lines[0]);
        envelope["crc32"] = 0;
        file << envelope.dump() << '\n';
    }

    EXPECT_THROW(ReadRLBotCapture(corrupt, true), std::runtime_error);

    // New captures in the same output directory must never overwrite old files.
    context->captureSink = {};
    context->captureDirectory = output;
    {
        RLBotBot bot({0}, 0, "another", context);
        bot.initialize(nullptr, nullptr, nullptr);
        Step(bot, 0);
    }

    EXPECT_EQ(ReadRLBotCapture(path, false).at("callbacks"), 18);

    std::ofstream changed(fixture.checkpoint / "POLICY.lt", std::ios::app | std::ios::binary);
    changed << 'x';
    changed.close();
    EXPECT_THROW(ReplayRLBotCapture(path, context, false, false), std::runtime_error);
}

TEST(RLBotCapture, CliRestoresSetupAndReplaysDelayedControlsOnCPU) {
    ROCKETFORGE_REQUIRE_COLLISION_MESHES();
    Fixture fixture;
    fixture.agent->agentName = "capture-cli-agent";
    fixture.agent->actionDelay = 3;
    AgentRegistry::Register("capture-cli-agent", [] {
        std::unique_ptr<TestAgent> agent = std::make_unique<TestAgent>();
        agent->agentName = "capture-cli-agent";
        return agent;
    });

    const std::filesystem::path output = RocketForgeTests::TestTempDir("cli-capture");
    Capture* observer = nullptr;
    std::shared_ptr<SharedBotContext> context = fixture.Context(observer);
    context->captureDirectory = output / "original";
    {
        RLBotBot bot({0, 1, 2}, 0, "capture", context);
        bot.initialize(nullptr, nullptr, nullptr);
        for (int tick = 0; tick <= 16; ++tick) Step(bot, tick);
    }

    const auto findCapture = [](const std::filesystem::path& root) {
        for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(root))
            if (entry.path().extension() == ".rfcapture") return entry.path();
        return std::filesystem::path{};
    };

    const std::filesystem::path original = findCapture(output / "original");
    ASSERT_FALSE(original.empty());

    std::vector<std::string> args{"RLBot", "--replay", original.string(),
        "--model-dir", fixture.checkpoint.string(), "--capture-dir", (output / "replayed").string(),
        "--device", "cpu"};
    std::vector<char*> argv;
    for (std::string& arg : args) argv.push_back(arg.data());
    ASSERT_EQ(RunRLBot(static_cast<int>(argv.size()), argv.data(), RLBotLaunchMode::ROCKET_FORGE_DEVELOPMENT, {}), 0);

    const std::filesystem::path replayed = findCapture(output / "replayed");
    std::vector<nlohmann::json> expected, actual;
    nlohmann::json header = ReadRLBotCapture(original, false, [&](const auto& row) {
        if (row.at("kind") == "callback") expected.push_back(row);
    });
    nlohmann::json replayHeader = ReadRLBotCapture(replayed, false, [&](const auto& row) {
        if (row.at("kind") == "callback") actual.push_back(row);
    });
    EXPECT_EQ(replayHeader.at("setup"), header.at("setup"));
    EXPECT_EQ(replayHeader.at("device"), "cpu");
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i].at("controls"), actual[i].at("controls"));
        EXPECT_EQ(expected[i].at("schedule"), actual[i].at("schedule"));
        EXPECT_EQ(expected[i].at("policy"), actual[i].at("policy"));
    }

    // A portable package's neighboring default setup must not silently replace the recorded setup.
    // An explicit --agent-setup remains a checked override.
    const std::filesystem::path portableRoot = output / "portable";
    std::filesystem::create_directories(portableRoot);
    std::unique_ptr<Agent> defaults = AgentRegistry::Create("capture-cli-agent");
    std::ofstream(portableRoot / "agent_setup.json")
        << RocketForge::AgentSetupSnapshotToJson(RocketForge::CaptureAgentSetupSnapshot(*defaults));

    args[0] = (portableRoot / "RLBotCore.exe").string();
    args[6] = (output / "portable-replay").string();
    argv.clear();
    for (std::string& arg : args) argv.push_back(arg.data());
    ASSERT_EQ(RunRLBot(static_cast<int>(argv.size()), argv.data(), RLBotLaunchMode::PORTABLE, {}), 0);
    EXPECT_EQ(ReadRLBotCapture(findCapture(output / "portable-replay"), false).at("setup"), header.at("setup"));

    if (const char* artifact = std::getenv("ROCKETFORGE_CAPTURE_TEST_DIR")) {
        ASSERT_FALSE(std::filesystem::exists(artifact));
        std::filesystem::copy(output, artifact, std::filesystem::copy_options::recursive);
    }
}
