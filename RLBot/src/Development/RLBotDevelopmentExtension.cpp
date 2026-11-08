#include "RLBotDevelopmentExtension.h"

#include <GigaLearnInfer/InferUnit.h>
#include <RLGymCPP/Framework.h>

#include <utility>

namespace
{
    RLGC::Action ToAction(const rlbot::flat::ControllerState* input)
    {
        RLGC::Action action = {};
        if (!input)
            return action;

        action.throttle = input->throttle();
        action.steer = input->steer();
        action.pitch = input->pitch();
        action.yaw = input->yaw();
        action.roll = input->roll();
        action.jump = input->jump() ? 1.f : 0.f;
        action.boost = input->boost() ? 1.f : 0.f;
        action.handbrake = input->handbrake() ? 1.f : 0.f;
        return action;
    }
}

RLBotDevelopmentExtension::RLBotDevelopmentExtension(
    unsigned team,
    const std::unordered_set<unsigned>& indices)
    : analysis_(team, indices)
{
    analysis_.Configure();
    wallDashDrill_ = RLBotWallDashDrill::CreateFromEnv();
    callbackAnalysisActive_ = analysis_.IsEnabled();
}

void RLBotDevelopmentExtension::BeginCallback()
{
    pendingBatch_.clear();
    pendingInference_.reset();
    callbackAnalysisActive_ = analysis_.IsEnabled();
    if (callbackAnalysisActive_)
        callbackStart_ = std::chrono::steady_clock::now();
}

void RLBotDevelopmentExtension::CapturePlayerPacket(
    unsigned index,
    const rlbot::flat::PlayerInfo* playerInfo)
{
    if (!callbackAnalysisActive_ || !playerInfo)
        return;

    const auto dodgeDir = playerInfo->dodge_dir();
    rawPlayerPackets_[index] = {
        .lastInput = ToAction(playerInfo->last_input()),
        .airState = static_cast<int>(playerInfo->air_state()),
        .dodgeTimeout = playerInfo->dodge_timeout(),
        .dodgeElapsed = playerInfo->dodge_elapsed(),
        .dodgeDirX = dodgeDir ? dodgeDir->x() : 0.f,
        .dodgeDirY = dodgeDir ? dodgeDir->y() : 0.f,
        .demoTimeout = playerInfo->demolished_timeout(),
    };
}

RLBotClientExtensionStep RLBotDevelopmentExtension::Step(
    const RLGC::GameState& state,
    const std::unordered_set<unsigned>& indices,
    float curTime)
{
    if (!wallDashDrill_)
        return {};

    RLBotWallDashDrill::StepResult result = wallDashDrill_->Step(state, indices, curTime);
    return {
        .consumedPacket = result.consumedPacket,
        .clearControls = result.clearControls,
        .resetExtensionState = result.resetAnalysis,
        .resetTickCounter = result.resetTickCounter,
        .resetObservedPlayerState = result.resetObservedPlayerState,
        .forceInferAction = result.forceInferAction,
        .desiredGameState = std::move(result.desiredGameState),
    };
}

void RLBotDevelopmentExtension::ResetBot(unsigned index)
{
    if (callbackAnalysisActive_)
        analysis_.ResetBot(index);
}

void RLBotDevelopmentExtension::ResetCapturedPlayerPacket(unsigned index)
{
    rawPlayerPackets_.erase(index);
}

void RLBotDevelopmentExtension::RecordInference(
    unsigned index,
    const std::vector<RLGC::ActionMacro>& actionMacros,
    int actionId)
{
    if (!callbackAnalysisActive_)
        return;

    if (auto found = pendingBatch_.find(index); found != pendingBatch_.end()) {
        pendingInference_ = std::move(found->second);
        pendingBatch_.erase(found);
    }

    if (!pendingInference_.has_value())
        pendingInference_ = RLBotPolicyInferenceResult{ .actionId = actionId };

    analysis_.RecordInference(index, actionMacros, std::move(pendingInference_.value()));
    pendingInference_.reset();
}

void RLBotDevelopmentExtension::ActivatePendingAction(unsigned index)
{
    if (callbackAnalysisActive_)
        analysis_.ActivatePendingAction(index);
}

void RLBotDevelopmentExtension::RecordFrame(const RLBotFrameEvent& event)
{
    if (!callbackAnalysisActive_)
        return;

    const auto raw = rawPlayerPackets_.find(event.index);
    const RawPlayerPacket* packet = raw != rawPlayerPackets_.end() ? &raw->second : nullptr;
    const double callbackWorkMicros = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - callbackStart_).count();
    analysis_.WriteSample({
        .index = event.index,
        .player = event.player,
        .gameState = event.gameState,
        .actionParser = event.actionParser,
        .selectedAction = event.selectedAction,
        .appliedControls = event.appliedControls,
        .prevAction = event.prevAction,
        .activeActionMacros = event.activeActionMacros,
        .activeMacroIndex = event.activeMacroIndex,
        .activeMacroTicksLeft = event.activeMacroTicksLeft,
        .tickSkip = event.tickSkip,
        .actionDelay = event.actionDelay,
        .ticksElapsed = event.ticksElapsed,
        .episode = wallDashDrill_ ? wallDashDrill_->GetEpisode() : 0,
        .policyTick = event.policyTick,
        .secondsElapsed = event.secondsElapsed,
        .gameTimeRemaining = event.gameTimeRemaining,
        .matchPhase = event.matchPhase,
        .deltaTime = event.deltaTime,
        .callbackWorkMicros = callbackWorkMicros,
        .runtimeStartupMicros = runtimeStartupMicros_,
        .useGPU = event.useGPU,
        .inferAction = event.inferAction,
        .activatedPendingAction = event.activatedPendingAction,
        .frameNum = event.frameNum,
        .rlbotLastInput = packet ? packet->lastInput : RLGC::Action{},
        .rawAirState = packet ? packet->airState : -1,
        .rawDodgeTimeout = packet ? packet->dodgeTimeout : -1.f,
        .rawDodgeElapsed = packet ? packet->dodgeElapsed : -1.f,
        .rawDodgeDirX = packet ? packet->dodgeDirX : 0.f,
        .rawDodgeDirY = packet ? packet->dodgeDirY : 0.f,
        .rawDemoTimeout = packet ? packet->demoTimeout : -1.f,
        .trainingEngineVersion = event.trainingEngineVersion,
    });
}

void RLBotDevelopmentExtension::AdvancePolicyStep(unsigned index)
{
    if (callbackAnalysisActive_)
        analysis_.AdvancePolicyStep(index);
}

void RLBotDevelopmentExtension::FinishCallback()
{
    if (!callbackAnalysisActive_)
        return;

    analysis_.FinishCallback(std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - callbackStart_).count());
}

bool RLBotDevelopmentExtension::CapturePolicyDiagnostics() const
{
    return callbackAnalysisActive_ && analysis_.CapturePolicyDiagnostics();
}

void RLBotDevelopmentExtension::OnPolicyRuntimeReady(double startupMicros)
{
    runtimeStartupMicros_ = startupMicros;
}

void RLBotDevelopmentExtension::OnPolicyInference(
    int actionId,
    double inferenceMicros,
    const GGL::InferUnit& inferUnit)
{
    if (!callbackAnalysisActive_)
        return;

    pendingInference_ = {
        .actionId = actionId,
        .inferenceMicros = inferenceMicros,
    };
    if (CapturePolicyDiagnostics()) {
        pendingInference_->observation = inferUnit.lastObs;
        pendingInference_->actionMask = inferUnit.lastActionMasks;
        pendingInference_->actionProbabilities = inferUnit.lastActionProbs;
    }
}

void RLBotDevelopmentExtension::OnPolicyBatchInference(
    const std::vector<unsigned>& indices,
    const std::vector<int>& actions,
    double inferenceMicros,
    const GGL::InferUnit& inferUnit)
{
    if (!callbackAnalysisActive_)
        return;

    const size_t obsSize = inferUnit.obsSize;
    const size_t actionSize = inferUnit.actionParser->GetActionAmount();

    for (size_t row = 0; row < indices.size(); ++row) {
        RLBotPolicyInferenceResult& result = pendingBatch_[indices[row]];
        result = { .actionId = actions[row], .inferenceMicros = inferenceMicros };
        if (CapturePolicyDiagnostics()) {
            result.observation.assign(inferUnit.lastObs.begin() + row * obsSize,
                inferUnit.lastObs.begin() + (row + 1) * obsSize);
            result.actionMask.assign(inferUnit.lastActionMasks.begin() + row * actionSize,
                inferUnit.lastActionMasks.begin() + (row + 1) * actionSize);
            result.actionProbabilities.assign(inferUnit.lastActionProbs.begin() + row * actionSize,
                inferUnit.lastActionProbs.begin() + (row + 1) * actionSize);
        }
    }
}

std::unique_ptr<RLBotClientExtension> CreateRLBotDevelopmentExtension(
    unsigned team,
    const std::unordered_set<unsigned>& indices)
{
    return std::make_unique<RLBotDevelopmentExtension>(team, indices);
}
