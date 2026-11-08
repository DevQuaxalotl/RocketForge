#include "RLBotAnalysis.h"

#include <RLGymCPP/Framework.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

using namespace RLGC;

namespace
{
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

    bool EnvFlagOrDefault(const char* name, bool fallback)
    {
        return GetEnv(name) ? EnvFlagEnabled(name) : fallback;
    }

    int EnvIntOrDefault(const char* name, int fallback)
    {
        const char* value = GetEnv(name);
        if (!value)
            return fallback;

        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0')
            return fallback;
        return (int)parsed;
    }

    void WriteActionCsv(std::ostream& out, const RLGC::Action& action)
    {
        out << action.throttle << ','
            << action.steer << ','
            << action.pitch << ','
            << action.yaw << ','
            << action.roll << ','
            << action.jump << ','
            << action.boost << ','
            << action.handbrake;
    }

    bool ActionsEqual(const RLGC::Action& a, const RLGC::Action& b)
    {
        return a.throttle == b.throttle &&
            a.steer == b.steer &&
            a.pitch == b.pitch &&
            a.yaw == b.yaw &&
            a.roll == b.roll &&
            a.jump == b.jump &&
            a.boost == b.boost &&
            a.handbrake == b.handbrake;
    }

    int CountAllowedActions(const std::vector<uint8_t>& mask)
    {
        return (int)std::count_if(mask.begin(), mask.end(), [](uint8_t value) {
            return value != 0;
        });
    }

    uint64_t HashActionMask(const std::vector<uint8_t>& mask)
    {
        uint64_t hash = 1469598103934665603ull;
        for (uint8_t value : mask)
        {
            hash ^= value ? 1ull : 0ull;
            hash *= 1099511628211ull;
        }
        return hash;
    }

    std::string MaskToBits(const std::vector<uint8_t>& mask)
    {
        std::string result;
        result.reserve(mask.size());
        for (uint8_t value : mask)
            result.push_back(value ? '1' : '0');
        return result;
    }

    uint64_t HashObsValues(const std::vector<float>& obs)
    {
        uint64_t hash = 1469598103934665603ull;
        for (float value : obs)
        {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int byte = 0; byte < 4; byte++)
            {
                hash ^= (uint64_t)((bits >> (byte * 8)) & 0xffu);
                hash *= 1099511628211ull;
            }
        }
        return hash;
    }

    std::string ObsToValues(const std::vector<float>& obs)
    {
        std::ostringstream ss;
        ss << std::setprecision(9);
        for (int i = 0; i < (int)obs.size(); i++)
        {
            if (i > 0)
                ss << ';';
            ss << obs[i];
        }
        return ss.str();
    }

    std::string TopActionProbsToValues(const std::vector<float>& probs, int topN)
    {
        std::vector<int> indices(probs.size());
        for (int i = 0; i < (int)indices.size(); i++)
            indices[i] = i;

        const int count = std::min(topN, (int)indices.size());
        std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
            [&](int a, int b) {
                return probs[a] > probs[b];
            });

        std::ostringstream ss;
        ss << std::setprecision(9);
        for (int i = 0; i < count; i++)
        {
            if (i > 0)
                ss << '|';
            const int actionId = indices[i];
            ss << actionId << ':' << probs[actionId];
        }
        return ss.str();
    }

    bool IsNearWall(const RLGC::Player& player)
    {
        return std::abs(player.pos.x) > CommonValues::SIDE_WALL_X - 550.f ||
            std::abs(player.pos.y) > CommonValues::BACK_WALL_Y - 650.f;
    }

    int CountWheelContacts(const RLGC::Player& player)
    {
        int count = 0;
        for (bool hasContact : player.wheelsWithContact)
            if (hasContact)
                count++;
        return count;
    }

    std::string WheelContactBits(const RLGC::Player& player)
    {
        std::string bits;
        bits.reserve(4);
        for (bool hasContact : player.wheelsWithContact)
            bits.push_back(hasContact ? '1' : '0');
        return bits;
    }

    void AppendVecSnapshot(std::ostream& out, const Vec& vec)
    {
        out << vec.x << ':' << vec.y << ':' << vec.z;
    }

    void AppendRotMatSnapshot(std::ostream& out, const RotMat& rotMat)
    {
        AppendVecSnapshot(out, rotMat.forward);
        out << ':';
        AppendVecSnapshot(out, rotMat.right);
        out << ':';
        AppendVecSnapshot(out, rotMat.up);
    }

    void AppendPhysSnapshot(std::ostream& out, const PhysState& state)
    {
        AppendVecSnapshot(out, state.pos);
        out << ':';
        AppendRotMatSnapshot(out, state.rotMat);
        out << ':';
        AppendVecSnapshot(out, state.vel);
        out << ':';
        AppendVecSnapshot(out, state.angVel);
    }

    void AppendActionSnapshot(std::ostream& out, const RLGC::Action& action)
    {
        out << action.throttle << ':' << action.steer << ':' << action.pitch << ':' << action.yaw << ':'
            << action.roll << ':' << action.jump << ':' << action.boost << ':' << action.handbrake;
    }

    void AppendControlsSnapshot(std::ostream& out, const CarControls& controls)
    {
        out << controls.throttle << ':' << controls.steer << ':' << controls.pitch << ':' << controls.yaw << ':'
            << controls.roll << ':' << controls.jump << ':' << controls.boost << ':' << controls.handbrake;
    }

    void AppendPlayerSnapshot(std::ostream& out, const Player& player)
    {
        out << player.index << ':' << player.carId << ':' << (int)player.team << ':';
        AppendPhysSnapshot(out, player);
        out << ':'
            << player.boost << ':'
            << (player.isOnGround ? 1 : 0) << ':'
            << (player.wasOnGroundDuringStep ? 1 : 0) << ':'
            << (player.obsOnGround ? 1 : 0) << ':'
            << (player.wheelsWithContact[0] ? 1 : 0) << ':'
            << (player.wheelsWithContact[1] ? 1 : 0) << ':'
            << (player.wheelsWithContact[2] ? 1 : 0) << ':'
            << (player.wheelsWithContact[3] ? 1 : 0) << ':'
            << (player.hasJumped ? 1 : 0) << ':'
            << (player.hasDoubleJumped ? 1 : 0) << ':'
            << (player.hasFlipped ? 1 : 0) << ':';
        AppendVecSnapshot(out, player.flipRelTorque);
        out << ':'
            << player.jumpTime << ':'
            << (player.isFlipping ? 1 : 0) << ':'
            << player.flipTime << ':'
            << (player.isJumping ? 1 : 0) << ':'
            << player.airTime << ':'
            << player.airTimeSinceJump << ':'
            << (player.isBoosting ? 1 : 0) << ':'
            << player.boostingTime << ':'
            << player.timeSinceBoosted << ':'
            << (player.isSupersonic ? 1 : 0) << ':'
            << player.supersonicTime << ':'
            << player.handbrakeVal << ':'
            << (player.isAutoFlipping ? 1 : 0) << ':'
            << player.autoFlipTimer << ':'
            << player.autoFlipTorqueScale << ':'
            << (player.worldContact.hasContact ? 1 : 0) << ':';
        AppendVecSnapshot(out, player.worldContact.contactNormal);
        out << ':'
            << player.carContact.otherCarID << ':'
            << player.carContact.cooldownTimer << ':'
            << (player.isDemoed ? 1 : 0) << ':'
            << player.demoRespawnTimer << ':';
        AppendControlsSnapshot(out, player.lastControls);
        out << ':';
        AppendActionSnapshot(out, player.prevAction);
        out << ':'
            << (int)player.airState << ':'
            << (player.hasDodged ? 1 : 0) << ':'
            << player.dodgeTimeout << ':'
            << player.demolishedTimeout << ':'
            << (player.ballTouchedStep ? 1 : 0) << ':'
            << (player.ballTouchedTick ? 1 : 0);
    }

    // Live play has no match arena, so the snapshot names the engine a replay should simulate with.
    std::string SerializeGameStateSnapshot(const GameState& state, int trainingEngineVersion)
    {
        std::ostringstream ss;
        ss << std::setprecision(9);
        ss << "v=2;gm=" << (int)state.gameMode
            << ";dt=" << state.deltaTime
            << ";goal=" << (state.goalScored ? 1 : 0)
            << ";last_touch=" << state.lastTouchCarID
            << ";last_tick=" << state.lastTickCount
            << ";engine=" << trainingEngineVersion
            << ";ball=";
        AppendPhysSnapshot(ss, state.ball);

        ss << ";pads=";
        for (int i = 0; i < (int)state.boostPads.size(); i++)
        {
            if (i > 0)
                ss << '|';
            const float timer = i < (int)state.boostPadTimers.size() ? state.boostPadTimers[i] : 0.f;
            ss << (state.boostPads[i] ? 1 : 0) << ':' << timer;
        }

        ss << ";players=" << state.players.size();
        for (const Player& player : state.players)
        {
            ss << '|';
            AppendPlayerSnapshot(ss, player);
        }
        return ss.str();
    }

    int FindActionIdForControls(
        RLGC::ActionParser& actionParser,
        const RLGC::Action& action,
        const RLGC::Player& player,
        const RLGC::GameState& state)
    {
        const int actionAmount = actionParser.GetActionAmount();
        for (int i = 0; i < actionAmount; i++)
            if (ActionsEqual(actionParser.ParseAction(i, player, state), action))
                return i;
        return -1;
    }

    std::string BuildMacroPlanString(
        RLGC::ActionParser& actionParser,
        const std::vector<RLGC::ActionMacro>& macros,
        const RLGC::Player& player,
        const RLGC::GameState& state)
    {
        std::ostringstream ss;
        for (int i = 0; i < (int)macros.size(); i++)
        {
            if (i > 0)
                ss << '|';
            ss << FindActionIdForControls(actionParser, macros[i].action, player, state)
                << 'x' << macros[i].ticks;
        }
        return ss.str();
    }

    bool IsSelectedMacro(const std::vector<RLGC::ActionMacro>& macros, int actionTicks)
    {
        return macros.size() != 1 || (!macros.empty() && macros.front().ticks != actionTicks);
    }

    void WriteTraceHeader(std::ostream& out)
    {
        out << "source,bot_index,bot_team,car_id,episode,physics_tick,policy_step,policy_tick,tick_skip,action_delay,ticks_elapsed,"
            << "seconds_elapsed,game_time_remaining,match_phase,delta_time,infer_action,apply_queued_action,"
            << "selected_action_id,selected_is_macro,selected_macro_index,applied_action_id,applied_base_action_id,"
            << "macro_segment_index,macro_segment_tick,macro_segment_ticks,macro_ticks_left,macro_plan,"
            << "mask_allowed_count,mask_hash,mask_bits,"
            << "mask_boost,mask_on_ground,mask_was_on_ground_during_step,mask_obs_on_ground,mask_is_jumping,mask_is_flipping,"
            << "mask_has_jumped,mask_has_flipped,mask_has_dodged,mask_has_double_jumped,"
            << "mask_air_state,mask_jump_time,mask_air_time,mask_air_time_since_jump,mask_dodge_timeout,"
            << "obs_size,obs_hash,obs_values,action_probs,action_prob_top,"
            << "inference_us,callback_work_us,previous_callback_us,runtime_startup_us,use_gpu,"
            << "selected_throttle,selected_steer,selected_pitch,selected_yaw,selected_roll,selected_jump,selected_boost,selected_handbrake,"
            << "applied_throttle,applied_steer,applied_pitch,applied_yaw,applied_roll,applied_jump,applied_boost,applied_handbrake,"
            << "prev_throttle,prev_steer,prev_pitch,prev_yaw,prev_roll,prev_jump,prev_boost,prev_handbrake,"
            << "x,y,z,vx,vy,vz,ang_vel_x,ang_vel_y,ang_vel_z,speed,boost,on_ground,obs_on_ground,wheels_contact_count,wheels_contact_bits,"
            << "world_contact,wall_contact,near_wall,contact_normal_x,contact_normal_y,contact_normal_z,"
            << "is_jumping,is_flipping,has_jumped,has_flipped,has_dodged,has_double_jumped,"
            << "air_state,jump_time,air_time,air_time_since_jump,dodge_timeout,raw_air_state,raw_dodge_timeout,raw_demo_timeout,"
            << "frame_num,rlbot_last_input_throttle,rlbot_last_input_steer,rlbot_last_input_pitch,rlbot_last_input_yaw,"
            << "rlbot_last_input_roll,rlbot_last_input_jump,rlbot_last_input_boost,rlbot_last_input_handbrake,"
            << "raw_dodge_elapsed,raw_dodge_dir_x,raw_dodge_dir_y,"
            << "jump_edge,ball_touched_tick,obs_state_snapshot\n";
    }
} // anonymous namespace

RLBotAnalysis::RLBotAnalysis(unsigned team, const std::unordered_set<unsigned>& indices)
    : team_(team)
    , indices_(indices)
{
}

bool RLBotAnalysis::IsEnabled() const
{
    return enabled_;
}

bool RLBotAnalysis::CapturePolicyDiagnostics() const
{
    return enabled_ && capturePolicyDiagnostics_;
}

void RLBotAnalysis::Configure()
{
    if (configured_)
        return;

    configured_ = true;
    if (!EnvFlagEnabled("ROCKETFORGE_RLBOT_TESTING") && !EnvFlagEnabled("ROCKETFORGE_RLBOT_TRACE"))
        return;

    try {
        const char* tracePath = GetEnv("ROCKETFORGE_RLBOT_TRACE_PATH");
        std::filesystem::path baseTracePath = tracePath ? std::filesystem::path(tracePath) : std::filesystem::path("rlbot_action_trace.csv");
        std::ostringstream suffix;
        suffix << "_team" << team_;
        std::set<unsigned> sortedIndices(std::begin(indices_), std::end(indices_));
        for (unsigned index : sortedIndices)
            suffix << "_idx" << index;

        const std::filesystem::path parentPath = baseTracePath.parent_path();
        const std::string extension = baseTracePath.has_extension() ? baseTracePath.extension().string() : std::string();
        const std::string stem = baseTracePath.has_extension()
            ? baseTracePath.stem().string()
            : baseTracePath.filename().string();
        tracePath_ = parentPath / (stem + suffix.str() + extension);
        maxRows_ = EnvIntOrDefault("ROCKETFORGE_RLBOT_TRACE_MAX_ROWS", -1);
        flushRows_ = std::max(1, EnvIntOrDefault("ROCKETFORGE_RLBOT_TRACE_FLUSH_ROWS", 1));
        capturePolicyDiagnostics_ = EnvFlagOrDefault(
            "ROCKETFORGE_RLBOT_TRACE_POLICY_DIAGNOSTICS", true);
        captureStateSnapshots_ = EnvFlagOrDefault(
            "ROCKETFORGE_RLBOT_TRACE_STATE_SNAPSHOTS", true);

        const std::filesystem::path parent = tracePath_.parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                RG_LOG("RLBotAnalysis: failed to create trace directory " << parent << ": " << ec.message());
                return;
            }
        }

        traceCsv_.open(tracePath_, std::ios::out | std::ios::trunc);
        if (!traceCsv_.good()) {
            RG_LOG("RLBotAnalysis: failed to open action trace CSV: " << tracePath_);
            return;
        }

        traceCsv_ << std::fixed << std::setprecision(6);
        WriteTraceHeader(traceCsv_);
        traceCsv_.flush();
        enabled_ = true;
        RG_LOG("RLBotAnalysis: writing action trace to " << tracePath_);
    }
    catch (const std::exception& e) {
        RG_LOG("RLBotAnalysis: failed to configure action trace: " << e.what());
        enabled_ = false;
    }
}

void RLBotAnalysis::ResetBot(unsigned index)
{
    if (!enabled_)
        return;

    botStates_[index] = BotState{};
}

void RLBotAnalysis::RecordInference(
    unsigned index,
    const std::vector<RLGC::ActionMacro>& actionMacros,
    RLBotPolicyInferenceResult&& inference)
{
    if (!enabled_)
        return;

    BotState& state = botStates_[index];
    state.selectedObs = std::move(inference.observation);
    state.selectedActionMask = std::move(inference.actionMask);
    state.selectedActionProbs = std::move(inference.actionProbabilities);
    state.selectedActionId = inference.actionId;
    state.pendingActionId = inference.actionId;
    state.lastInferenceMicros = inference.inferenceMicros;
    state.selectedActionMacros = actionMacros;
}

void RLBotAnalysis::ActivatePendingAction(unsigned index)
{
    if (!enabled_)
        return;

    BotState& state = botStates_[index];
    state.activeActionId = state.pendingActionId;
    state.pendingActionId = -1;
}

void RLBotAnalysis::WriteSample(const RLBotAnalysisSample& sample)
{
    if (!enabled_ || !traceCsv_.good())
        return;
    if (maxRows_ >= 0 && rows_ >= maxRows_)
        return;

    const RLGC::Player& player = sample.player;
    const RLGC::GameState& gameState = sample.gameState;
    RLGC::ActionParser& actionParser = sample.actionParser;
    BotState& state = botStates_[sample.index];
    const bool outputJump = sample.appliedControls.jump > 0.5f;
    const bool jumpEdge = outputJump && !state.lastOutputJump;
    state.lastOutputJump = outputJump;

    const int actionTicks = sample.tickSkip - sample.actionDelay;
    const bool selectedIsMacro = IsSelectedMacro(state.selectedActionMacros, actionTicks);
    const int selectedMacroIndex = selectedIsMacro ? state.selectedActionId : -1;
    const int appliedBaseActionId = FindActionIdForControls(actionParser, sample.appliedControls, player, gameState);
    const int macroSegmentTicks =
        !sample.activeActionMacros.empty() && sample.activeMacroIndex < (int)sample.activeActionMacros.size()
        ? sample.activeActionMacros[sample.activeMacroIndex].ticks
        : -1;
    const int macroSegmentTick =
        macroSegmentTicks > 0
        ? std::max(0, macroSegmentTicks - sample.activeMacroTicksLeft)
        : -1;
    const std::string macroPlan = BuildMacroPlanString(actionParser, state.selectedActionMacros, player, gameState);
    const int gameTick = (int)std::round(sample.secondsElapsed * 120.f);
    const bool nearWall = IsNearWall(player);
    // The selected obs/mask fields are inference-time diagnostics, but replay seeding needs
    // the current packet state so queued-action rows can be used as snapshot ticks.
    const std::string currentStateSnapshot = captureStateSnapshots_
        ? SerializeGameStateSnapshot(gameState, sample.trainingEngineVersion)
        : std::string();

    traceCsv_ << "rlbot,"
        << sample.index << ','
        << (int)player.team << ','
        << player.carId << ','
        << sample.episode << ','
        << gameTick << ','
        << state.policyStep << ','
        << sample.policyTick << ','
        << sample.tickSkip << ','
        << sample.actionDelay << ','
        << sample.ticksElapsed << ','
        << sample.secondsElapsed << ','
        << sample.gameTimeRemaining << ','
        << sample.matchPhase << ','
        << sample.deltaTime << ','
        << (sample.inferAction ? 1 : 0) << ','
        << (sample.activatedPendingAction ? 1 : 0) << ','
        << state.selectedActionId << ','
        << (selectedIsMacro ? 1 : 0) << ','
        << selectedMacroIndex << ','
        << state.activeActionId << ','
        << appliedBaseActionId << ','
        << sample.activeMacroIndex << ','
        << macroSegmentTick << ','
        << macroSegmentTicks << ','
        << sample.activeMacroTicksLeft << ','
        << macroPlan << ','
        << CountAllowedActions(state.selectedActionMask) << ','
        << HashActionMask(state.selectedActionMask) << ','
        << MaskToBits(state.selectedActionMask) << ','
        << player.boost << ','
        << (player.isOnGround ? 1 : 0) << ','
        << (player.wasOnGroundDuringStep ? 1 : 0) << ','
        << (player.obsOnGround ? 1 : 0) << ','
        << (player.isJumping ? 1 : 0) << ','
        << (player.isFlipping ? 1 : 0) << ','
        << (player.hasJumped ? 1 : 0) << ','
        << (player.hasFlipped ? 1 : 0) << ','
        << (player.hasDodged ? 1 : 0) << ','
        << (player.hasDoubleJumped ? 1 : 0) << ','
        << (int)player.airState << ','
        << player.jumpTime << ','
        << player.airTime << ','
        << player.airTimeSinceJump << ','
        << player.dodgeTimeout << ','
        << state.selectedObs.size() << ','
        << HashObsValues(state.selectedObs) << ','
        << ObsToValues(state.selectedObs) << ','
        << ObsToValues(state.selectedActionProbs) << ','
        << TopActionProbsToValues(state.selectedActionProbs, 12) << ','
        << (sample.inferAction ? state.lastInferenceMicros : -1.0) << ','
        << sample.callbackWorkMicros << ','
        << previousCallbackMicros_ << ','
        << sample.runtimeStartupMicros << ','
        << (sample.useGPU ? 1 : 0) << ',';
    WriteActionCsv(traceCsv_, sample.selectedAction);
    traceCsv_ << ',';
    WriteActionCsv(traceCsv_, sample.appliedControls);
    traceCsv_ << ',';
    WriteActionCsv(traceCsv_, sample.prevAction);
    traceCsv_ << ','
        << player.pos.x << ','
        << player.pos.y << ','
        << player.pos.z << ','
        << player.vel.x << ','
        << player.vel.y << ','
        << player.vel.z << ','
        << player.angVel.x << ','
        << player.angVel.y << ','
        << player.angVel.z << ','
        << player.vel.Length() << ','
        << player.boost << ','
        << (player.isOnGround ? 1 : 0) << ','
        << (player.obsOnGround ? 1 : 0) << ','
        << CountWheelContacts(player) << ','
        << WheelContactBits(player) << ','
        << 0 << ','
        << 0 << ','
        << (nearWall ? 1 : 0) << ','
        << 0.f << ','
        << 0.f << ','
        << 0.f << ','
        << (player.isJumping ? 1 : 0) << ','
        << (player.isFlipping ? 1 : 0) << ','
        << (player.hasJumped ? 1 : 0) << ','
        << (player.hasFlipped ? 1 : 0) << ','
        << (player.hasDodged ? 1 : 0) << ','
        << (player.hasDoubleJumped ? 1 : 0) << ','
        << (int)player.airState << ','
        << player.jumpTime << ','
        << player.airTime << ','
        << player.airTimeSinceJump << ','
        << player.dodgeTimeout << ','
        << sample.rawAirState << ','
        << sample.rawDodgeTimeout << ','
        << sample.rawDemoTimeout << ','
        << sample.frameNum << ',';
    WriteActionCsv(traceCsv_, sample.rlbotLastInput);
    traceCsv_ << ','
        << sample.rawDodgeElapsed << ','
        << sample.rawDodgeDirX << ','
        << sample.rawDodgeDirY << ','
        << (jumpEdge ? 1 : 0) << ','
        << (player.ballTouchedTick ? 1 : 0) << ','
        << currentStateSnapshot << '\n';
    rows_++;
    if (maxRows_ >= 0 && rows_ >= maxRows_) {
        traceCsv_.flush();
        traceCsv_.close();
        enabled_ = false;
        RG_LOG("RLBotAnalysis: reached trace row limit; policy diagnostics disabled for the rest of the run.");
    } else if (rows_ % flushRows_ == 0) {
        traceCsv_.flush();
    }
}

void RLBotAnalysis::FinishCallback(double callbackMicros)
{
    previousCallbackMicros_ = callbackMicros;
}

void RLBotAnalysis::AdvancePolicyStep(unsigned index)
{
    if (!enabled_)
        return;

    botStates_[index].policyStep++;
}
