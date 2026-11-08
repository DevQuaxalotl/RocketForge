#include "RLBotPolicyRuntime.h"

#include <GigaLearnInfer/Agent.h>
#include <GigaLearnInfer/InferUnit.h>
#include <RLGymCPP/Framework.h>

#include <chrono>
#include <stdexcept>

using namespace RLGC;

RLBotPolicyRuntime::RLBotPolicyRuntime(
    const Agent& agent,
    const std::filesystem::path& modelsFolder,
    bool useGPU,
    RLBotPolicyObserver* observer,
    unsigned warmupBatchSize)
    : observer_(observer)
{
    // Complete model loading and one live-shaped forward pass before RLBot sends
    // InitComplete. Large CUDA policies otherwise stall their first packet.
    if (warmupBatchSize < 1)
        throw std::invalid_argument("Policy warmup requires at least one car");

    const auto startup = std::chrono::steady_clock::now();
    obsBuilder_.reset(agent.MakeConfiguredObsBuilder());
    actionParser_.reset(agent.MakeActionParser());
    const int obsSize = agent.GetObsSize();
    inferUnit_ = std::make_unique<GGL::InferUnit>(
        obsBuilder_.get(),
        obsSize,
        actionParser_.get(),
        agent,
        modelsFolder,
        useGPU);

    RocketSim::ArenaConfig warmupConfig = {};
    warmupConfig.engine = RocketSim::EngineFromVersion(agent.trainingEngineVersion);
    if (RocketSim::Arena* warmupArena = RocketSim::Arena::Create(agent.gameMode, warmupConfig)) {
        startupWarmupArena_.reset(warmupArena);
        std::vector<unsigned> warmupIndices;
        for (unsigned i = 0; i < warmupBatchSize; ++i) {
            warmupArena->AddCar(RocketSim::Team::BLUE, agent.carConfig);
            warmupIndices.push_back(i);
        }

        for (unsigned i = 0; i < warmupBatchSize; ++i)
            warmupArena->AddCar(RocketSim::Team::ORANGE, agent.carConfig);

        GameState warmupState(warmupArena);
        obsBuilder_->Reset(warmupState);
        const bool deterministic = true;
        inferUnit_->BatchInferActionIndices(warmupIndices, warmupState, deterministic);
    }

    inferUnit_->captureDiagnostics = observer_ && observer_->CapturePolicyDiagnostics();
    const double startupMicros = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - startup).count();
    if (observer_)
        observer_->OnPolicyRuntimeReady(startupMicros);
}

RLBotPolicyRuntime::~RLBotPolicyRuntime() = default;

RLGC::ActionParser& RLBotPolicyRuntime::GetActionParser()
{
    return *actionParser_;
}

void RLBotPolicyRuntime::RequestObservationReset()
{
    observationNeedsReset_ = true;
}

void RLBotPolicyRuntime::PrepareObservationState(RLGC::GameState& gameState)
{
    if (!observationNeedsReset_)
        return;

    obsBuilder_->Reset(gameState);
    startupWarmupArena_.reset();
    observationNeedsReset_ = false;
}

int RLBotPolicyRuntime::Infer(
    const RLGC::Player& player,
    const RLGC::GameState& gameState)
{
    inferUnit_->captureDiagnostics = observer_ && observer_->CapturePolicyDiagnostics();
    const auto inferenceStart = observer_
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const int actionId = inferUnit_->InferActionIndex(player, gameState, true);
    if (observer_) {
        const double inferenceMicros = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - inferenceStart).count();
        observer_->OnPolicyInference(actionId, inferenceMicros, *inferUnit_);
    }
    return actionId;
}

std::vector<int> RLBotPolicyRuntime::InferBatch(const std::vector<unsigned>& indices, const RLGC::GameState& gameState)
{
    if (indices.empty())
        return {};

    inferUnit_->captureDiagnostics = observer_ && observer_->CapturePolicyDiagnostics();
    const auto start = observer_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool deterministic = true;
    std::vector<int> actions = inferUnit_->BatchInferActionIndices(indices, gameState, deterministic);

    if (observer_) {
        const double micros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        observer_->OnPolicyBatchInference(indices, actions, micros, *inferUnit_);
    }

    return actions;
}
