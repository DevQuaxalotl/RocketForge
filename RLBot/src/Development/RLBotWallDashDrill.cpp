#include "RLBotWallDashDrill.h"

#include <RLGymCPP/Framework.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace RLGC;

namespace
{
    constexpr float WALL_DASH_DRILL_WALL_OFFSET = 24.f;
    constexpr float WALL_DASH_DRILL_MIN_Z = 650.f;
    constexpr float WALL_DASH_DRILL_MAX_Z = 1600.f;
    constexpr float WALL_DASH_DRILL_STRAIGHT_RUNWAY = 1000.f;
    constexpr float WALL_DASH_DRILL_MULTI_CAR_OFFSET = 350.f;

    const char* GetEnv(const char* name)
    {
        const char* value = std::getenv(name);
        return value && value[0] != '\0' ? value : nullptr;
    }

    bool EnvFlagEnabled(const char* name)
    {
        const char* value = GetEnv(name);
        if (!value)
            return false;

        std::string normalized(value);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
            return (char)std::tolower(c);
        });

        return normalized != "0" && normalized != "false" && normalized != "off";
    }

    float EnvFloatOrDefault(const char* name, float fallback)
    {
        const char* value = GetEnv(name);
        if (!value)
            return fallback;

        char* end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end == value || *end != '\0')
            return fallback;

        return parsed;
    }

    std::vector<float> EnvAngleListOrDefault(const char* name)
    {
        std::vector<float> result;
        const char* value = GetEnv(name);
        if (value) {
            std::stringstream input(value);
            std::string token;
            while (std::getline(input, token, ',')) {
                char* end = nullptr;
                const float parsed = std::strtof(token.c_str(), &end);
                if (end != token.c_str())
                    result.push_back(parsed);
            }
        }

        if (result.empty())
            result = { 0.f, 45.f, -45.f };

        return result;
    }

    std::vector<unsigned> SortedIndices(const std::unordered_set<unsigned>& indices)
    {
        std::vector<unsigned> sorted(indices.begin(), indices.end());
        std::sort(sorted.begin(), sorted.end());
        return sorted;
    }

    std::unique_ptr<rlbot::flat::Float> MakeFloat(float value)
    {
        return std::make_unique<rlbot::flat::Float>(value);
    }

    std::unique_ptr<rlbot::flat::Vector3PartialT> MakeVectorPartial(const Vec& value)
    {
        auto result = std::make_unique<rlbot::flat::Vector3PartialT>();
        result->x = MakeFloat(value.x);
        result->y = MakeFloat(value.y);
        result->z = MakeFloat(value.z);
        return result;
    }

    std::unique_ptr<rlbot::flat::RotatorPartialT> MakeRotatorPartial(const Angle& value)
    {
        auto result = std::make_unique<rlbot::flat::RotatorPartialT>();
        result->pitch = MakeFloat(value.pitch);
        result->yaw = MakeFloat(value.yaw);
        result->roll = MakeFloat(value.roll);
        return result;
    }

    std::unique_ptr<rlbot::flat::DesiredPhysicsT> MakeDesiredPhysics(
        const Vec& pos,
        const Angle& rot,
        const Vec& vel,
        const Vec& angVel)
    {
        auto result = std::make_unique<rlbot::flat::DesiredPhysicsT>();
        result->location = MakeVectorPartial(pos);
        result->rotation = MakeRotatorPartial(rot);
        result->velocity = MakeVectorPartial(vel);
        result->angular_velocity = MakeVectorPartial(angVel);
        return result;
    }
}

struct RLBotWallDashDrill::WallDashDrillTarget
{
    Vec pos = {};
    Vec forward = {};
    Vec wallNormal = {};
    Angle rotation = {};
};

std::unique_ptr<RLBotWallDashDrill> RLBotWallDashDrill::CreateFromEnv()
{
    auto drill = std::make_unique<RLBotWallDashDrill>();
    drill->_enabled = EnvFlagEnabled("ROCKETFORGE_RLBOT_WALL_DASH_DRILL");
    if (!drill->_enabled)
        return nullptr;

    drill->_resetInterval = std::max(0.25f, EnvFloatOrDefault("ROCKETFORGE_RLBOT_WALL_DASH_DRILL_INTERVAL", drill->_resetInterval));
    drill->_startSpeed = std::max(0.f, EnvFloatOrDefault("ROCKETFORGE_RLBOT_WALL_DASH_DRILL_SPEED", drill->_startSpeed));
    drill->_startY = EnvFloatOrDefault("ROCKETFORGE_RLBOT_WALL_DASH_DRILL_Y", drill->_startY);
    drill->_startZ = std::clamp(
        EnvFloatOrDefault("ROCKETFORGE_RLBOT_WALL_DASH_DRILL_Z", drill->_startZ),
        WALL_DASH_DRILL_MIN_Z,
        WALL_DASH_DRILL_MAX_Z);
    drill->_angleDegrees = EnvAngleListOrDefault("ROCKETFORGE_RLBOT_WALL_DASH_DRILL_ANGLES");

    std::printf(
        "RLBot wall-dash drill enabled: interval %.2fs, speed %.1f, y %.1f, z %.1f, %zu angle cases\n",
        drill->_resetInterval,
        drill->_startSpeed,
        drill->_startY,
        drill->_startZ,
        drill->_angleDegrees.size());

    return drill;
}

RLBotWallDashDrill::WallDashDrillTarget RLBotWallDashDrill::MakeTarget(int caseIndex) const
{
    const int angleCount = std::max(1, (int)_angleDegrees.size());
    const float angleDeg = _angleDegrees.empty() ? 0.f : _angleDegrees[caseIndex % angleCount];
    const float wallSide = ((caseIndex / angleCount) % 2 == 0) ? 1.f : -1.f;
    const float wallTravelAngle = angleDeg * 3.1415926535f / 180.f;
    const float forwardSide = -wallSide;

    WallDashDrillTarget target = {};
    target.wallNormal = Vec(-wallSide, 0.f, 0.f);
    target.forward = Vec(0.f, forwardSide * std::cos(wallTravelAngle), std::sin(wallTravelAngle));

    float minZ = WALL_DASH_DRILL_MIN_Z;
    float maxZ = WALL_DASH_DRILL_MAX_Z;
    if (target.forward.z > 0.01f)
        maxZ = std::min(maxZ, WALL_DASH_DRILL_MAX_Z - WALL_DASH_DRILL_STRAIGHT_RUNWAY * target.forward.z);
    else if (target.forward.z < -0.01f)
        minZ = std::max(minZ, WALL_DASH_DRILL_MIN_Z + WALL_DASH_DRILL_STRAIGHT_RUNWAY * -target.forward.z);

    const float startZ = minZ <= maxZ
        ? std::clamp(_startZ, minZ, maxZ)
        : std::clamp((WALL_DASH_DRILL_MIN_Z + WALL_DASH_DRILL_MAX_Z) * 0.5f, WALL_DASH_DRILL_MIN_Z, WALL_DASH_DRILL_MAX_Z);

    target.pos = Vec(wallSide * (CommonValues::SIDE_WALL_X - WALL_DASH_DRILL_WALL_OFFSET), _startY, startZ);
    target.rotation = Angle::FromRotMat(RotMat::LookAt(target.forward, target.wallNormal));
    return target;
}

RLBotWallDashDrill::WallDashDrillTarget RLBotWallDashDrill::MakeTargetForOrdinal(int caseIndex, int ordinal) const
{
    const int angleCount = std::max(1, (int)_angleDegrees.size());
    WallDashDrillTarget target = MakeTarget(caseIndex + ordinal * angleCount);

    // Cars beyond the first pair reuse a wall side, so stagger them along the wall plane.
    const int sameWallOrdinal = ordinal / 2;
    if (sameWallOrdinal > 0)
        target.pos -= target.forward * (WALL_DASH_DRILL_MULTI_CAR_OFFSET * sameWallOrdinal);

    return target;
}

bool RLBotWallDashDrill::IsPlayerSettled(const GameState& state, unsigned index, int ordinal) const
{
    constexpr float POS_X_TOLERANCE = 25.f;
    constexpr float POS_Y_TOLERANCE = 250.f;
    constexpr float POS_Z_TOLERANCE = 150.f;
    constexpr float MAX_SETTLED_SPEED = 900.f;

    if (index >= state.players.size())
        return false;

    const Player& player = state.players[index];
    const WallDashDrillTarget target = MakeTargetForOrdinal(_pendingCase, ordinal);

    if (std::abs(player.pos.x - target.pos.x) > POS_X_TOLERANCE)
        return false;
    if (std::abs(player.pos.y - target.pos.y) > POS_Y_TOLERANCE)
        return false;
    if (std::abs(player.pos.z - target.pos.z) > POS_Z_TOLERANCE)
        return false;
    if (player.vel.Length() > MAX_SETTLED_SPEED)
        return false;

    const float minForwardSpeed = std::min(100.f, _startSpeed * 0.25f);
    if (player.vel.Dot(target.forward) < minForwardSpeed)
        return false;

    return true;
}

bool RLBotWallDashDrill::IsResetSettled(const GameState& state, const std::unordered_set<unsigned>& indices) const
{
    const std::vector<unsigned> sorted = SortedIndices(indices);
    for (int ordinal = 0; ordinal < (int)sorted.size(); ordinal++) {
        if (!IsPlayerSettled(state, sorted[ordinal], ordinal))
            return false;
    }

    return true;
}

rlbot::flat::DesiredGameStateT RLBotWallDashDrill::BuildGameState(const std::unordered_set<unsigned>& indices, int caseIndex) const
{
    rlbot::flat::DesiredGameStateT state;

    auto ballState = std::make_unique<rlbot::flat::DesiredBallStateT>();
    ballState->physics = MakeDesiredPhysics(Vec(0.f, 0.f, CommonValues::BALL_RADIUS), Angle(), Vec(), Vec());
    state.ball_states.emplace_back(std::move(ballState));

    const unsigned maxIndex = *std::max_element(indices.begin(), indices.end());
    state.car_states.resize(maxIndex + 1);
    for (std::unique_ptr<rlbot::flat::DesiredCarStateT>& carState : state.car_states)
        carState = std::make_unique<rlbot::flat::DesiredCarStateT>();

    const std::vector<unsigned> sorted = SortedIndices(indices);
    for (int ordinal = 0; ordinal < (int)sorted.size(); ordinal++) {
        const WallDashDrillTarget target = MakeTargetForOrdinal(caseIndex, ordinal);

        auto carState = std::make_unique<rlbot::flat::DesiredCarStateT>();
        carState->physics = MakeDesiredPhysics(target.pos, target.rotation, target.forward * _startSpeed, Vec());
        carState->boost_amount = MakeFloat(15.f);
        state.car_states[sorted[ordinal]] = std::move(carState);
    }

    return state;
}

RLBotWallDashDrill::StepResult RLBotWallDashDrill::Step(
    const GameState& state,
    const std::unordered_set<unsigned>& indices,
    float curTime)
{
    StepResult result = {};
    if (!_enabled || indices.empty())
        return result;

    if (_pendingReset) {
        if (!IsResetSettled(state, indices)) {
            if (curTime - _pendingResetTime > 0.5f) {
                result.desiredGameState.emplace(BuildGameState(indices, _pendingCase));
                _pendingResetTime = curTime;
            }

            result.clearControls = true;
            result.consumedPacket = true;
            return result;
        }

        result.clearControls = true;
        result.resetAnalysis = true;
        result.resetTickCounter = true;
        result.resetObservedPlayerState = true;
        result.forceInferAction = true;

        _pendingReset = false;
        _episode++;
        _nextCase = _pendingCase + 1;
        _lastResetTime = curTime;
        return result;
    }

    if (curTime - _lastResetTime < _resetInterval)
        return result;

    _pendingCase = _nextCase;
    _pendingReset = true;
    _pendingResetTime = curTime;

    result.desiredGameState.emplace(BuildGameState(indices, _pendingCase));
    result.clearControls = true;
    result.consumedPacket = true;
    return result;
}
