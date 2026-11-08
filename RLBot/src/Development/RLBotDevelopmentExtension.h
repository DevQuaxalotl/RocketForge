#pragma once

#include "RLBotAnalysis.h"
#include "RLBotWallDashDrill.h"
#include "Runtime/RLBotClientExtension.h"

#include <chrono>
#include <optional>
#include <unordered_map>

/** Connects RocketForge's local analysis and drill tools to the portable client. **/
class RLBotDevelopmentExtension final :
    public RLBotClientExtension,
    public RLBotPolicyObserver {
public:
    RLBotDevelopmentExtension(
        unsigned team,
        const std::unordered_set<unsigned>& indices);

    RLBotPolicyObserver* GetPolicyObserver() override { return this; }
    void BeginCallback() override;
    void CapturePlayerPacket(unsigned index, const rlbot::flat::PlayerInfo* playerInfo) override;
    RLBotClientExtensionStep Step(
        const RLGC::GameState& state,
        const std::unordered_set<unsigned>& indices,
        float curTime) override;
    void ResetBot(unsigned index) override;
    void ResetCapturedPlayerPacket(unsigned index) override;
    void RecordInference(
        unsigned index,
        const std::vector<RLGC::ActionMacro>& actionMacros,
        int actionId) override;
    void ActivatePendingAction(unsigned index) override;
    void RecordFrame(const RLBotFrameEvent& event) override;
    void AdvancePolicyStep(unsigned index) override;
    void FinishCallback() override;

    bool CapturePolicyDiagnostics() const override;
    void OnPolicyRuntimeReady(double startupMicros) override;
    void OnPolicyInference(
        int actionId,
        double inferenceMicros,
        const GGL::InferUnit& inferUnit) override;

    void OnPolicyBatchInference(
        const std::vector<unsigned>& indices,
        const std::vector<int>& actions,
        double inferenceMicros,
        const GGL::InferUnit& inferUnit) override;

private:
    struct RawPlayerPacket {
        RLGC::Action lastInput = {};
        int airState = -1;
        float dodgeTimeout = -1.f;
        float dodgeElapsed = -1.f;
        float dodgeDirX = 0.f;
        float dodgeDirY = 0.f;
        float demoTimeout = -1.f;
    };

    RLBotAnalysis analysis_;
    std::unique_ptr<RLBotWallDashDrill> wallDashDrill_;
    std::unordered_map<unsigned, RawPlayerPacket> rawPlayerPackets_;
    std::optional<RLBotPolicyInferenceResult> pendingInference_;
    std::unordered_map<unsigned, RLBotPolicyInferenceResult> pendingBatch_;
    std::chrono::steady_clock::time_point callbackStart_ = {};
    double runtimeStartupMicros_ = -1.0;
    bool callbackAnalysisActive_ = false;
};

std::unique_ptr<RLBotClientExtension> CreateRLBotDevelopmentExtension(
    unsigned team,
    const std::unordered_set<unsigned>& indices);
