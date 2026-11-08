#pragma once

#include <rlbot/Bot.h>

#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace RLGC { struct GameState; }

/**
 * Development-only drill state setter for replaying wall-dash starts through
 * the live RLBot process without putting the setup logic on the inference path.
 **/
class RLBotWallDashDrill {
public:
    struct StepResult {
        bool consumedPacket = false;
        bool clearControls = false;
        bool resetAnalysis = false;
        bool resetTickCounter = false;
        bool resetObservedPlayerState = false;
        bool forceInferAction = false;
        std::optional<rlbot::flat::DesiredGameStateT> desiredGameState;
    };

    static std::unique_ptr<RLBotWallDashDrill> CreateFromEnv();

    int GetEpisode() const { return _episode; }

    StepResult Step(const RLGC::GameState& state, const std::unordered_set<unsigned>& indices, float curTime);

private:
    struct WallDashDrillTarget;

    WallDashDrillTarget MakeTarget(int caseIndex) const;
    WallDashDrillTarget MakeTargetForOrdinal(int caseIndex, int ordinal) const;
    bool IsPlayerSettled(const RLGC::GameState& state, unsigned index, int ordinal) const;
    bool IsResetSettled(const RLGC::GameState& state, const std::unordered_set<unsigned>& indices) const;
    rlbot::flat::DesiredGameStateT BuildGameState(const std::unordered_set<unsigned>& indices, int caseIndex) const;

    bool _enabled = false;
    bool _pendingReset = false;
    float _resetInterval = 2.25f;
    float _startSpeed = 400.f;
    float _startY = 0.f;
    float _startZ = 650.f;
    float _lastResetTime = -10000.f;
    float _pendingResetTime = -10000.f;
    int _episode = 0;
    int _nextCase = 0;
    int _pendingCase = 0;
    std::vector<float> _angleDegrees;
};
