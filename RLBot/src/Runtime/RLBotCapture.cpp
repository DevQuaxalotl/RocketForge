#include "RLBotCapture.h"
#include "RLBotClient.h"
#include <GigaLearnInfer/AgentSetupSnapshot.h>
#include <GigaLearnInfer/InferUnit.h>
#include <RLGymCPP/Framework.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

using nlohmann::json;
namespace fs = std::filesystem;

namespace {
    constexpr size_t MAX_RECORD_BYTES = 64 * 1024 * 1024;
    constexpr size_t MAX_QUEUED_BYTES = 16 * 1024 * 1024;

    uint32_t CRC32(const char* data, size_t size, uint32_t crc = 0) {
        static const auto table = [] {
            std::array<uint32_t, 256> result{};
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t value = i;
                for (int bit = 0; bit < 8; ++bit)
                    value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1u)));
                result[i] = value;
            }

            return result;
        }();

        crc = ~crc;
        for (size_t i = 0; i < size; ++i)
            crc = table[(crc ^ static_cast<unsigned char>(data[i])) & 255u] ^ (crc >> 8);

        return ~crc;
    }

    std::string Envelope(const json& record) {
        const std::string payload = record.dump();
        return json{{"payload", payload}, {"crc32", CRC32(payload.data(), payload.size())}}.dump() + '\n';
    }

    template<class T> json Pack(const T* value) {
        if (!value) return nullptr;

        const std::unique_ptr<typename T::NativeTableType> native(value->UnPack());
        flatbuffers::FlatBufferBuilder buffer;
        buffer.Finish(T::Pack(buffer, native.get()));

        static const char digits[] = "0123456789abcdef";
        std::string hex(buffer.GetSize() * 2, '0');
        for (size_t i = 0; i < buffer.GetSize(); ++i) {
            hex[i * 2] = digits[buffer.GetBufferPointer()[i] >> 4];
            hex[i * 2 + 1] = digits[buffer.GetBufferPointer()[i] & 15];
        }

        return hex;
    }

    template<class T> struct Unpack {
        std::vector<uint8_t> bytes;

        explicit Unpack(const json& value) {
            if (value.is_null()) return;

            const std::string& hex = value.get_ref<const std::string&>();
            if (hex.size() % 2) throw std::runtime_error("odd packet hex length");

            auto digit = [](char ch) -> unsigned {
                if (ch >= '0' && ch <= '9') return ch - '0';
                if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
                throw std::runtime_error("invalid packet hex");
            };

            bytes.resize(hex.size() / 2);
            for (size_t i = 0; i < bytes.size(); ++i)
                bytes[i] = static_cast<uint8_t>(digit(hex[i * 2]) * 16 + digit(hex[i * 2 + 1]));

            flatbuffers::Verifier verifier(bytes.data(), bytes.size());
            if (!verifier.VerifyBuffer<T>(nullptr)) throw std::runtime_error("invalid captured FlatBuffer");
        }

        const T* Get() const { return bytes.empty() ? nullptr : flatbuffers::GetRoot<T>(bytes.data()); }
    };

    json ActionToJson(const RLGC::Action& action) {
        return {action.throttle, action.steer, action.pitch, action.yaw, action.roll, action.jump, action.boost, action.handbrake};
    }

    json PlanToJson(const std::vector<RLGC::ActionMacro>& macros) {
        json result = json::array();
        for (const RLGC::ActionMacro& macro : macros) result.push_back({{"ticks", macro.ticks}, {"action", ActionToJson(macro.action)}});
        return result;
    }

    void ValidatePacket(const rlbot::flat::GamePacket* packet) {
        if (!packet || !packet->match_info() || !packet->balls() || packet->balls()->size() == 0) return;

        if (!std::isfinite(packet->match_info()->seconds_elapsed()) || !packet->boost_pads() || !packet->balls()->Get(0)->physics())
            throw std::runtime_error("capture packet lacks required runtime fields");

        if (const auto players = packet->players()) {
            if (players->size() > 64) throw std::runtime_error("too many captured players");
            for (const auto player : *players)
                if (!player || !player->physics()) throw std::runtime_error("captured player has no physics");
        }
    }

    void RestoreRNG(const json& record) {
        std::istringstream stream(record.at("rng").get<std::string>());
        if (!(stream >> ::Math::GetRandEngine())) throw std::runtime_error("unsupported captured RNG state");
    }
}

json RLBotCheckpointChecksums(const fs::path& path) {
    json result = json::object();
    for (const fs::directory_entry& file : fs::directory_iterator(path)) {
        if (!file.is_regular_file() || file.path().extension() != ".lt") continue;

        // Critic/optimizer files do not participate in inference or portable exports.
        const std::string name = file.path().filename().string();
        if (name.find("CRITIC") != std::string::npos || name.find("OPTIM") != std::string::npos) continue;

        std::ifstream input(file.path(), std::ios::binary);
        if (!input) throw std::runtime_error("cannot checksum checkpoint: " + name);

        std::array<char, 65536> buffer;
        uint32_t crc = 0;
        uint64_t size = 0;
        while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
            crc = CRC32(buffer.data(), static_cast<size_t>(input.gcount()), crc);
            size += input.gcount();
        }

        if (!input.eof()) throw std::runtime_error("checkpoint checksum read failed");
        result[name] = {{"bytes", size}, {"crc32", crc}};
    }

    if (result.empty()) throw std::runtime_error("no inference checkpoint files to record");
    return result;
}

struct RLBotCapture::Impl {
    std::function<void(const json&)> sink;
    std::ofstream file;
    std::thread writer;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::string> queue;
    size_t queuedBytes = 0;
    bool stopping = false;
    std::atomic<bool> failed{false};
    uint64_t sequence = 0, callback = 0, lost = 0;
    json current;
    std::vector<unsigned> indices;

    void Failure() noexcept {
        ++lost;
        if (!failed.exchange(true))
            std::fprintf(stderr, "RLBot capture failed; recording is incomplete. Bot controls continue.\n");
    }

    void Submit(json value) {
        value["sequence"] = sequence++;
        if (sink) {
            sink(value);
            return;
        }

        if (failed) {
            ++lost;
            return;
        }

        std::string encoded = Envelope(value);
        std::lock_guard lock(mutex);
        if (queuedBytes + encoded.size() > MAX_QUEUED_BYTES) {
            ++lost;
            if (lost == 1) std::fprintf(stderr, "RLBot capture queue full; recording has missing callbacks.\n");
            return;
        }

        queuedBytes += encoded.size();
        queue.push_back(std::move(encoded));
        wake.notify_one();
    }

    void RunWriter() noexcept {
        std::chrono::steady_clock::time_point nextFlush = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        try {
            for (;;) {
                std::string data;
                bool done;
                {
                    std::unique_lock lock(mutex);
                    wake.wait_until(lock, nextFlush, [&] { return stopping || !queue.empty(); });
                    if (!queue.empty()) {
                        data = std::move(queue.front());
                        queue.pop_front();
                        queuedBytes -= data.size();
                    }

                    done = stopping && queue.empty();
                }

                if (!data.empty()) file.write(data.data(), data.size());

                if (done || std::chrono::steady_clock::now() >= nextFlush) {
                    file.flush();
                    nextFlush = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
                }

                if (!file) throw std::runtime_error("capture write failed");
                if (done) return;
            }
        } catch (...) {
            // The callback thread owns the loss counter.
            // This atomic flag also prevents further queue growth if a disk fills or disappears.
            failed = true;
            std::fprintf(stderr, "RLBot capture disk write failed; recording is incomplete.\n");
        }
    }

    ~Impl() {
        try {
            Submit({{"kind", "footer"}, {"callbacks", callback}, {"lost", lost}, {"write_failed", failed.load()}});
        } catch (...) {
            Failure();
        }

        if (writer.joinable()) {
            {
                std::lock_guard lock(mutex);
                stopping = true;
            }

            wake.notify_one();
            writer.join();
        }
    }
};

RLBotCapture::RLBotCapture(const SharedBotContext& context, const rlbot::Bot& bot, RLBotPolicyObserver* delegate)
    : _impl(std::make_unique<Impl>()), _delegate(delegate) {
    Impl& impl = *_impl;
    impl.sink = context.captureSink;
    impl.indices.assign(bot.indices.begin(), bot.indices.end());
    std::sort(impl.indices.begin(), impl.indices.end());

    if (!impl.sink) {
        fs::create_directories(context.captureDirectory);
        static std::atomic<uint64_t> counter{0};
        fs::path directory;
        for (int attempt = 0; ; ++attempt) {
            if (attempt == 100) throw std::runtime_error("cannot reserve unique capture directory");

            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            directory = context.captureDirectory / ("capture-" + std::to_string(stamp) + "-" + std::to_string(counter++));
            if (fs::create_directory(directory)) break;
        }

        const fs::path path = directory / "callbacks.rfcapture";
        impl.file.open(path, std::ios::binary);
        if (!impl.file) throw std::runtime_error("cannot open capture output: " + path.string());

        impl.writer = std::thread([&impl] { impl.RunWriter(); });
        std::printf("RLBot recording: %s\n", path.string().c_str());
    }

    impl.Submit({{"simulator", context.agent->trainingEngineVersion}, {"rng_type", typeid(::Math::GetRandEngine()).name()},
        {"kind", "header"}, {"schema", "rocketforge.rlbot.capture.v1"},
        {"agent", context.agent->agentName}, {"indices", impl.indices}, {"team", bot.team},
        {"device", context.useGPU ? "gpu" : "cpu"},
        {"setup", RocketForge::AgentSetupSnapshotToJson(RocketForge::CaptureAgentSetupSnapshot(*context.agent))},
        {"checkpoint", context.captureChecksums.empty() ? RLBotCheckpointChecksums(context.modelsFolder) : context.captureChecksums},
        {"replay_of", context.replaySource}});
}

RLBotCapture::~RLBotCapture() = default;

void RLBotCapture::Initialize(
    const rlbot::flat::ControllableTeamInfo* team,
    const rlbot::flat::FieldInfo* field,
    const rlbot::flat::MatchConfiguration* match) noexcept {
    try {
        _impl->Submit({{"kind", "initialize"}, {"team_info", Pack(team)}, {"field", Pack(field)}, {"match", Pack(match)}});
    } catch (...) {
        _impl->Failure();
    }
}

void RLBotCapture::Begin(const rlbot::flat::GamePacket* packet, const rlbot::flat::BallPrediction* prediction) noexcept {
    try {
        std::ostringstream rng;
        rng << ::Math::GetRandEngine();
        _impl->current = {{"kind", "callback"}, {"callback", _impl->callback++}, {"rng", rng.str()},
            {"packet", Pack(packet)}, {"prediction", Pack(prediction)},
            {"policy", json::array()}, {"schedule", json::array()}};
    } catch (...) {
        _impl->current = nullptr;
        _impl->Failure();
    }
}

void RLBotCapture::End(rlbot::Bot& bot) noexcept {
    try {
        if (_impl->current.is_null()) return;

        json controls = json::array();
        for (unsigned index : _impl->indices) {
            const auto output = bot.getOutput(index);
            controls.push_back({{"index", index}, {"values", {output.throttle(), output.steer(), output.pitch(), output.yaw(),
                output.roll(), output.jump() ? 1.f : 0.f, output.boost() ? 1.f : 0.f, output.handbrake() ? 1.f : 0.f}}});
        }

        _impl->current["controls"] = std::move(controls);
        _impl->Submit(std::move(_impl->current));
        _impl->current = nullptr;
    } catch (...) {
        _impl->Failure();
    }
}

void RLBotCapture::Frame(const RLBotFrameEvent& frame) noexcept {
    try {
        if (_impl->current.is_null()) return;

        _impl->current["schedule"].push_back({{"index", frame.index}, {"frame", frame.frameNum},
            {"infer", frame.inferAction}, {"tick_skip", frame.tickSkip}, {"delay", frame.actionDelay},
            {"ticks_elapsed", frame.ticksElapsed}, {"policy_tick", frame.policyTick},
            {"selected", ActionToJson(frame.selectedAction)}, {"previous", ActionToJson(frame.prevAction)},
            {"pending_plan", PlanToJson(frame.pendingActionMacros)},
            {"active_plan", PlanToJson(frame.activeActionMacros)}, {"macro_index", frame.activeMacroIndex},
            {"macro_ticks_left", frame.activeMacroTicksLeft}});
    } catch (...) {
        _impl->Failure();
    }
}

void RLBotCapture::OnPolicyRuntimeReady(double micros) {
    if (_delegate) _delegate->OnPolicyRuntimeReady(micros);
}

void RLBotCapture::OnPolicyInference(int action, double micros, const GGL::InferUnit& unit) {
    OnPolicyBatchInference(_impl->indices, {action}, micros, unit);
}

void RLBotCapture::OnPolicyBatchInference(
    const std::vector<unsigned>& indices, const std::vector<int>& actions, double micros, const GGL::InferUnit& unit) {
    try {
        if (!_impl->current.is_null()) {
            const size_t obsSize = unit.obsSize;
            const size_t actionSize = unit.actionParser->GetActionAmount();
            for (size_t row = 0; row < indices.size(); ++row) {
                auto slice = [row](const auto& values, size_t size) {
                    if (values.size() < (row + 1) * size) throw std::runtime_error("missing policy diagnostics");
                    return json(std::vector(values.begin() + row * size, values.begin() + (row + 1) * size));
                };

                _impl->current["policy"].push_back({{"index", indices[row]}, {"action", actions.at(row)},
                    {"observation", slice(unit.lastObs, obsSize)}, {"mask", slice(unit.lastActionMasks, actionSize)},
                    {"probabilities", slice(unit.lastActionProbs, actionSize)}});
            }

            _impl->current["inference_us"] = micros;
        }
    } catch (...) {
        _impl->Failure();
    }

    if (_delegate) _delegate->OnPolicyBatchInference(indices, actions, micros, unit);
}

json ReadRLBotCapture(const fs::path& path, bool allowIncomplete, const std::function<void(const json&)>& visitor) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open recording: " + path.string());

    json header;
    uint64_t sequence = 0, callbacks = 0;
    bool footer = false, initialized = false;
    std::string line;

    while (file.peek() != EOF) {
        line.clear();
        char character;
        bool newline = false;
        while (file.get(character)) {
            if (character == '\n') {
                newline = true;
                break;
            }

            if (line.size() >= MAX_RECORD_BYTES) throw std::runtime_error("capture record exceeds size limit");
            line += character;
        }

        if (!newline) {
            if (allowIncomplete && !footer) break;
            throw std::runtime_error("truncated capture record");
        }

        if (footer) throw std::runtime_error("data after capture footer");

        const json envelope = json::parse(line);
        const std::string payload = envelope.at("payload").get<std::string>();
        if (envelope.at("crc32").get<uint32_t>() != CRC32(payload.data(), payload.size()))
            throw std::runtime_error("capture checksum mismatch");

        const json record = json::parse(payload);
        if (record.at("sequence").get<uint64_t>() != sequence++) throw std::runtime_error("capture sequence gap");

        const std::string kind = record.at("kind").get<std::string>();
        if (sequence == 1) {
            if (kind != "header" || record.at("schema") != "rocketforge.rlbot.capture.v1")
                throw std::runtime_error("unsupported capture header");

            const std::vector<unsigned> indices = record.at("indices").get<std::vector<unsigned>>();
            if (indices.empty() || indices.size() > 64 || indices.back() >= 64 ||
                !std::is_sorted(indices.begin(), indices.end()) ||
                std::adjacent_find(indices.begin(), indices.end()) != indices.end() || record.at("team").get<unsigned>() > 1)
                throw std::runtime_error("invalid capture car indices or team");

            if (!record.at("agent").is_string() || !record.at("rng_type").is_string() ||
                !record.at("checkpoint").is_object() || record.at("checkpoint").empty() ||
                !RocketForge::AgentSetupSnapshotFromJson(record.at("setup")))
                throw std::runtime_error("invalid capture setup or checkpoint metadata");

            header = record;
        } else if (kind == "initialize") {
            if (initialized || callbacks) throw std::runtime_error("unexpected capture initialization");

            Unpack<rlbot::flat::ControllableTeamInfo> team(record.at("team_info"));
            Unpack<rlbot::flat::FieldInfo> field(record.at("field"));
            Unpack<rlbot::flat::MatchConfiguration> match(record.at("match"));
            initialized = true;
        } else if (kind == "callback") {
            if (!initialized) throw std::runtime_error("capture has no initialization");
            if (record.at("callback").get<uint64_t>() != callbacks++) throw std::runtime_error("missing callback");

            Unpack<rlbot::flat::GamePacket> packet(record.at("packet"));
            Unpack<rlbot::flat::BallPrediction> prediction(record.at("prediction"));
            ValidatePacket(packet.Get());
            if (!record.at("rng").is_string() || !record.at("policy").is_array() ||
                !record.at("schedule").is_array() || !record.at("controls").is_array())
                throw std::runtime_error("capture callback is missing diagnostics");
        } else if (kind == "footer") {
            if (record.at("lost") != 0 || record.at("write_failed") != false || record.at("callbacks") != callbacks)
                throw std::runtime_error("capture reports lost data");

            footer = true;
        } else {
            throw std::runtime_error("unknown capture record");
        }

        if (visitor) visitor(record);
    }

    if (!file.eof() && file.fail()) throw std::runtime_error("capture read failed");
    if (header.is_null() || !callbacks) throw std::runtime_error("capture has no complete callbacks");
    if (!footer && !allowIncomplete) throw std::runtime_error("capture is incomplete; use --allow-incomplete for its verified prefix");

    header["complete"] = footer;
    header["callbacks"] = callbacks;
    return header;
}

void ReplayRLBotCapture(const fs::path& path, std::shared_ptr<SharedBotContext> context, bool separateCars, bool allowIncomplete) {
    // Validate the entire stream before running any policy.
    // Even prefix replay must never cross a missing/corrupt callback and invent persistent state.
    const json header = ReadRLBotCapture(path, allowIncomplete);
    if (header.at("rng_type") != typeid(::Math::GetRandEngine()).name())
        throw std::runtime_error("recording uses an incompatible RNG implementation");
    if (RLBotCheckpointChecksums(context->modelsFolder) != header.at("checkpoint"))
        throw std::runtime_error("replay checkpoint differs from the recording");
    if (RocketForge::AgentSetupSnapshotToJson(RocketForge::CaptureAgentSetupSnapshot(*context->agent)) != header.at("setup"))
        throw std::runtime_error("replay agent setup differs from the recording");
    if (context->captureDirectory.empty() && !context->captureSink)
        throw std::runtime_error("replay needs --capture-dir for its output");

    context->captureChecksums = header.at("checkpoint");
    context->replaySource = fs::absolute(path).string();

    // A replay never activates development drills or live controls.
    context->extensionFactory = {};

    const std::vector<unsigned> indices = header.at("indices").get<std::vector<unsigned>>();
    if (indices.empty() || indices.size() > 64) throw std::runtime_error("invalid captured car indices");

    const unsigned team = header.at("team").get<unsigned>();
    std::vector<std::unique_ptr<RLBotBot>> bots;
    if (separateCars) {
        for (unsigned index : indices)
            bots.push_back(std::make_unique<RLBotBot>(std::unordered_set<unsigned>{index}, team, "replay", context));
    } else {
        bots.push_back(std::make_unique<RLBotBot>(std::unordered_set<unsigned>(indices.begin(), indices.end()), team, "replay", context));
    }

    ReadRLBotCapture(path, allowIncomplete, [&](const json& record) {
        if (record.at("kind") == "initialize") {
            Unpack<rlbot::flat::ControllableTeamInfo> teamInfo(record.at("team_info"));
            Unpack<rlbot::flat::FieldInfo> field(record.at("field"));
            Unpack<rlbot::flat::MatchConfiguration> match(record.at("match"));
            for (std::unique_ptr<RLBotBot>& bot : bots) bot->initialize(teamInfo.Get(), field.Get(), match.Get());
        } else if (record.at("kind") == "callback") {
            Unpack<rlbot::flat::GamePacket> packet(record.at("packet"));
            Unpack<rlbot::flat::BallPrediction> prediction(record.at("prediction"));
            RestoreRNG(record);
            for (std::unique_ptr<RLBotBot>& bot : bots) bot->update(packet.Get(), prediction.Get());
        }
    });

    std::printf("Replayed %llu callbacks (%s recording).\n", header.at("callbacks").get<unsigned long long>(),
        header.at("complete").get<bool>() ? "complete" : "incomplete prefix");
}
