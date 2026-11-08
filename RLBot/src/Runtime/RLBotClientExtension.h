#pragma once

#include "RLBotPolicyObserver.h"

#include <rlbot/Bot.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace RLGC {
    struct GameState;
    struct Player;
}

/** Optional packet interception result produced by an RLBot client extension. **/
struct RLBotClientExtensionStep {
    bool consumedPacket = false;
    bool clearControls = false;
    bool resetExtensionState = false;
    bool resetTickCounter = false;
    bool resetObservedPlayerState = false;
    bool forceInferAction = false;
    std::optional<rlbot::flat::DesiredGameStateT> desiredGameState;
};

/** Runtime state exposed after one bot's controls have been updated. **/
struct RLBotFrameEvent {
    unsigned index;
    const RLGC::Player& player;
    const RLGC::GameState& gameState;
    RLGC::ActionParser& actionParser;
    const RLGC::Action& selectedAction;
    const RLGC::Action& appliedControls;
    const RLGC::Action& prevAction;
    const std::vector<RLGC::ActionMacro>& activeActionMacros;
    const std::vector<RLGC::ActionMacro>& pendingActionMacros;
    int activeMacroIndex;
    int activeMacroTicksLeft;
    int tickSkip;
    int actionDelay;
    int ticksElapsed;
    int policyTick;
    float secondsElapsed;
    float gameTimeRemaining;
    int matchPhase;
    float deltaTime;
    bool useGPU;
    bool inferAction;
    bool activatedPendingAction;
    std::uint32_t frameNum;

    /** The engine the model trained on, which a replay of this frame simulates with. **/
    int trainingEngineVersion;
};

/**
 * Extension point for local diagnostics and drills. The packaged runtime uses
 * this no-op implementation and does not link any development sources.
 **/
class RLBotClientExtension {
public:
    virtual ~RLBotClientExtension() = default;

    virtual RLBotPolicyObserver* GetPolicyObserver() { return nullptr; }
    virtual void BeginCallback() {}
    virtual void CapturePlayerPacket(unsigned, const rlbot::flat::PlayerInfo*) {}
    virtual RLBotClientExtensionStep Step(
        const RLGC::GameState&,
        const std::unordered_set<unsigned>&,
        float) { return {}; }
    virtual void ResetBot(unsigned) {}
    virtual void ResetCapturedPlayerPacket(unsigned) {}
    virtual void RecordInference(
        unsigned,
        const std::vector<RLGC::ActionMacro>&,
        int) {}
    virtual void ActivatePendingAction(unsigned) {}
    virtual void RecordFrame(const RLBotFrameEvent&) {}
    virtual void AdvancePolicyStep(unsigned) {}
    virtual void FinishCallback() {}
};

using RLBotClientExtensionFactory = std::function<std::unique_ptr<RLBotClientExtension>(
    unsigned,
    const std::unordered_set<unsigned>&)>;
