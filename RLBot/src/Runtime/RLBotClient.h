#pragma once

#include "RLBotClientExtension.h"

#include <rlbot/BotManager.h>
#include <GigaLearnInfer/Agent.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>

#include <nlohmann/json.hpp>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace RLGC { struct GameState; }
namespace RocketSim { class Arena; }
class RLBotPolicyRuntime;
class RLBotCapture;

struct SharedBotContext {
    std::shared_ptr<const Agent> agent;
    std::filesystem::path modelsFolder;
    bool useGPU = false;
    RLBotClientExtensionFactory extensionFactory;
    std::filesystem::path captureDirectory;
    nlohmann::json captureChecksums;
    std::string replaySource;

    /** In-memory capture sink for deterministic offline checks. **/
    std::function<void(const nlohmann::json&)> captureSink;
};

struct PlayerTimingState {
    float jumpTime = 0.f;
    float airTime = 0.f;
    float airTimeSinceJump = 0.f;
    bool hadJumped = false;
    bool wasOnGroundDuringStep = false;
};

class RLBotBot : public rlbot::Bot {
public:
    struct PerBotState {
        bool initialized = false;

        // Queued policy output and currently applied controls.
        RLGC::Action
            action = {},
            controls = {};
        std::vector<RLGC::ActionMacro> pendingActionMacros;
        std::vector<RLGC::ActionMacro> activeActionMacros;
        RLGC::Action obsPrevAction = {};
        int activeMacroIndex = 0;
        int activeMacroTicksLeft = 0;
    };

    // Persistent info
    int ticks = -1;
    float prevTime = 0;

    RLBotBot() noexcept = delete;
    ~RLBotBot() noexcept override;

    RLBotBot(std::unordered_set<unsigned> indices_,
        unsigned team_,
        std::string name_,
        std::shared_ptr<const SharedBotContext> ctx_);

    void initialize(rlbot::flat::ControllableTeamInfo const* controllableTeamInfo_,
        rlbot::flat::FieldInfo const* fieldInfo_,
        rlbot::flat::MatchConfiguration const* matchConfiguration_) noexcept override;

    void update(rlbot::flat::GamePacket const* packet_,
        rlbot::flat::BallPrediction const* ballPrediction_) noexcept override;

private:
    void ClearExtensionControls(bool resetExtensionState);
    void ResetDrillObservedPlayers(RLGC::GameState& gs);
    bool HandleExtensionStep(RLGC::GameState& gs, float curTime, bool& forceInferAction);

    std::shared_ptr<const SharedBotContext> ctx_;
    std::unique_ptr<RLBotClientExtension> extension_;
    std::unique_ptr<RLBotCapture> capture_;
    std::unique_ptr<RLBotPolicyRuntime> policyRuntime_;
    std::unordered_map<unsigned, PerBotState> m_botState;
    std::vector<PlayerTimingState> m_playerTiming;
    std::unique_ptr<RocketSim::Arena> m_ballPredArena;
    std::uint64_t m_ballPredTickCount = 0;

    // Resolved from RLBot's match config in initialize(), or the agent's mode as fallback.
    RocketSim::GameMode m_gameMode = RocketSim::GameMode::SOCCAR;
};

class RLBotBotManager final : public rlbot::BotManagerBase {
public:
    explicit RLBotBotManager(
        std::shared_ptr<const SharedBotContext> ctx_,
        bool batchHivemind_ = true) noexcept;

private:
    static std::unique_ptr<rlbot::Bot> spawn(
        std::unordered_set<unsigned> indices_,
        unsigned team_,
        std::string name_) noexcept;
};
