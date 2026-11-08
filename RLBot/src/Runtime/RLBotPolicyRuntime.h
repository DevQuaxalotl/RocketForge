#pragma once

#include "RLBotPolicyObserver.h"

#include <filesystem>
#include <memory>
#include <vector>

class Agent;

namespace GGL { struct InferUnit; }
namespace RLGC {
    class ActionParser;
    class ObsBuilder;
    struct GameState;
    struct Player;
}
namespace RocketSim { class Arena; }

/**
 * Owns the policy components and their RLBot-specific initialization lifecycle.
 * Packet scheduling and control application remain the client's responsibility.
 **/
class RLBotPolicyRuntime {
public:
    RLBotPolicyRuntime(
        const Agent& agent,
        const std::filesystem::path& modelsFolder,
        bool useGPU,
        RLBotPolicyObserver* observer = nullptr,
        unsigned warmupBatchSize = 1);
    ~RLBotPolicyRuntime();

    RLBotPolicyRuntime(const RLBotPolicyRuntime&) = delete;
    RLBotPolicyRuntime& operator=(const RLBotPolicyRuntime&) = delete;

    RLGC::ActionParser& GetActionParser();
    void RequestObservationReset();
    void PrepareObservationState(RLGC::GameState& gameState);
    int Infer(
        const RLGC::Player& player,
        const RLGC::GameState& gameState);

    /** Returns one action per packet index, in the supplied order. **/
    std::vector<int> InferBatch(const std::vector<unsigned>& indices, const RLGC::GameState& gameState);

private:
    // NOTE: The warmup arena must outlive observation state until the first real reset.
    std::unique_ptr<RocketSim::Arena> startupWarmupArena_;
    std::unique_ptr<RLGC::ObsBuilder> obsBuilder_;
    std::unique_ptr<RLGC::ActionParser> actionParser_;
    std::unique_ptr<GGL::InferUnit> inferUnit_;
    RLBotPolicyObserver* observer_ = nullptr;
    bool observationNeedsReset_ = true;
};
