#pragma once

#include "RLBotPolicyObserver.h"
#include <rlbot/Bot.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <functional>
#include <memory>

struct SharedBotContext;
struct RLBotFrameEvent;

/** Optional callback recorder; disk I/O runs on a bounded background queue. **/
class RLBotCapture final : public RLBotPolicyObserver {
public:
    RLBotCapture(const SharedBotContext& context, const rlbot::Bot& bot, RLBotPolicyObserver* delegate);
    ~RLBotCapture();

    void Initialize(
        const rlbot::flat::ControllableTeamInfo*,
        const rlbot::flat::FieldInfo*,
        const rlbot::flat::MatchConfiguration*) noexcept;
    void Begin(const rlbot::flat::GamePacket*, const rlbot::flat::BallPrediction*) noexcept;
    void End(rlbot::Bot&) noexcept;
    void Frame(const RLBotFrameEvent&) noexcept;

    bool CapturePolicyDiagnostics() const override { return true; }
    void OnPolicyRuntimeReady(double) override;
    void OnPolicyInference(int, double, const GGL::InferUnit&) override;
    void OnPolicyBatchInference(const std::vector<unsigned>&, const std::vector<int>&, double, const GGL::InferUnit&) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    RLBotPolicyObserver* _delegate;
};

/** Reads verified records in order; an explicit opt-in permits a truncated final record. **/
nlohmann::json ReadRLBotCapture(const std::filesystem::path& path, bool allowIncomplete,
    const std::function<void(const nlohmann::json&)>& visitor = {});

/** Replays callback inputs through the real runtime without a network connection. **/
void ReplayRLBotCapture(
    const std::filesystem::path& path,
    std::shared_ptr<SharedBotContext> context,
    bool separateCars,
    bool allowIncomplete);

/** File size and CRC32 identify local checkpoint inputs; not a security signature. **/
nlohmann::json RLBotCheckpointChecksums(const std::filesystem::path& path);
