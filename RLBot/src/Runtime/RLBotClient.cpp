#include "RLBotClient.h"
#include "RLBotClientExtension.h"
#include "RLBotPolicyRuntime.h"
#include "RLBotCapture.h"

#include <RLGymCPP/Framework.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <set>
#include <utility>

using namespace RLGC;

namespace
{
    std::shared_ptr<const SharedBotContext>& SpawnContextStorage() {
        static std::shared_ptr<const SharedBotContext> ctx;
        return ctx;
    }

    void ResetPerBotState(RLBotBot::PerBotState& state)
    {
        state.initialized = true;
        state.action = {};
        state.controls = {};
        state.pendingActionMacros.clear();
        state.activeActionMacros.clear();
        state.obsPrevAction = {};
        state.activeMacroIndex = 0;
        state.activeMacroTicksLeft = 0;
    }

    void SetActiveMacro(RLBotBot::PerBotState& state)
    {
        while (state.activeMacroIndex < (int)state.activeActionMacros.size())
        {
            const RLGC::ActionMacro& macro = state.activeActionMacros[state.activeMacroIndex];
            state.activeMacroTicksLeft = macro.ticks;
            state.controls = macro.action;
            if (state.activeMacroTicksLeft > 0)
                return;

            state.activeMacroIndex++;
        }

        state.activeActionMacros.clear();
        state.activeMacroIndex = 0;
        state.activeMacroTicksLeft = 0;
    }

    void ActivatePendingMacros(RLBotBot::PerBotState& state)
    {
        if (state.pendingActionMacros.empty())
            return;

        state.activeActionMacros = std::move(state.pendingActionMacros);
        state.pendingActionMacros.clear();
        state.activeMacroIndex = 0;
        state.activeMacroTicksLeft = 0;
        SetActiveMacro(state);
    }

    void AdvanceActiveMacros(RLBotBot::PerBotState& state, int ticksElapsed)
    {
        while (ticksElapsed > 0 && !state.activeActionMacros.empty())
        {
            if (state.activeMacroTicksLeft > ticksElapsed)
            {
                state.activeMacroTicksLeft -= ticksElapsed;
                return;
            }

            ticksElapsed -= std::max(0, state.activeMacroTicksLeft);
            state.activeMacroIndex++;
            SetActiveMacro(state);
        }
    }

    std::vector<RLGC::ActionMacro> BuildActionMacros(
        RLGC::ActionParser& actionParser,
        int actionId,
        const RLGC::Player& player,
        const RLGC::GameState& state,
        int tickCount,
        const RLGC::Action& fallbackAction)
    {
        std::vector<RLGC::ActionMacro> macros;
        if (tickCount <= 0)
            return macros;

        actionParser.ParseActionMacros(actionId, player, state, tickCount, macros);
        if (macros.empty())
            macros.push_back({ fallbackAction, tickCount });

        int macroTicks = 0;
        bool validMacros = true;
        for (const RLGC::ActionMacro& macro : macros)
        {
            if (macro.ticks < 0)
                validMacros = false;
            macroTicks += macro.ticks;
        }

        if (!validMacros || macroTicks != tickCount)
        {
            RG_LOG("RLBotClient: action parser returned invalid macros for action " << actionId
                << "; falling back to a single action macro.");
            macros.clear();
            macros.push_back({ fallbackAction, tickCount });
        }

        return macros;
    }

    bool IsSegmentedActionMacro(const std::vector<RLGC::ActionMacro>& macros, int actionTicks)
    {
        return macros.size() != 1 || (!macros.empty() && macros.front().ticks != actionTicks);
    }

    // Not a cast: the two mode enums are ordered differently. Modes RocketSim can't
    // simulate (Rumble/Gridiron/Knockout) resolve to the fallback.
    RocketSim::GameMode ToRocketSimGameMode(rlbot::flat::GameMode mode, RocketSim::GameMode fallback) {
        switch (mode) {
        case rlbot::flat::GameMode::Soccar:     return RocketSim::GameMode::SOCCAR;
        case rlbot::flat::GameMode::Hoops:      return RocketSim::GameMode::HOOPS;
        case rlbot::flat::GameMode::Dropshot:   return RocketSim::GameMode::DROPSHOT;
        case rlbot::flat::GameMode::Snowday:    return RocketSim::GameMode::SNOWDAY;
        case rlbot::flat::GameMode::Heatseeker: return RocketSim::GameMode::HEATSEEKER;
        default:                                return fallback;
        }
    }

    Vec ToVec(const rlbot::flat::Vector3 rlbotVec) {
        return Vec(rlbotVec.x(), rlbotVec.y(), rlbotVec.z());
    }

    PhysState ToPhysObj(const rlbot::flat::Physics* phys) {
        PhysState obj = {};
        obj.pos = ToVec(phys->location());

        Angle ang = Angle(phys->rotation().yaw(), phys->rotation().pitch(), phys->rotation().roll());
        obj.rotMat = ang.ToRotMat();

        obj.vel = ToVec(phys->velocity());
        obj.angVel = ToVec(phys->angular_velocity());
        return obj;
    }

    Player ToPlayer(
        const rlbot::flat::PlayerInfo* playerInfo,
        float dtSec,
        PlayerTimingState& timing)
    {
        Player pd = {};

        static_cast<PhysState&>(pd) = ToPhysObj(playerInfo->physics());

        pd.carId = playerInfo->player_id();
        pd.team = (Team)playerInfo->team();

        pd.boost = playerInfo->boost();

        // Raw values from RLBot
        const auto airState = playerInfo->air_state();
        const float rawDemoTimeout = playerInfo->demolished_timeout();     // RLBot: -1 when not demoed

        pd.isOnGround = (playerInfo->air_state() == rlbot::flat::AirState::OnGround);
        timing.wasOnGroundDuringStep = timing.wasOnGroundDuringStep || pd.isOnGround;
        pd.wasOnGroundDuringStep = timing.wasOnGroundDuringStep;
        pd.obsOnGround = pd.isOnGround;
        pd.isJumping = (airState == rlbot::flat::AirState::Jumping);
        pd.isFlipping = (airState == rlbot::flat::AirState::Dodging);

        pd.hasJumped = playerInfo->has_jumped();
        pd.hasDoubleJumped = playerInfo->has_double_jumped();
        pd.hasFlipped = playerInfo->has_dodged();
        // Training sets hasDodged = hasFlipped || isFlipping
        pd.hasDodged = (pd.hasFlipped || pd.isFlipping);

        if (pd.hasJumped) {
            timing.jumpTime = timing.hadJumped ? timing.jumpTime + dtSec : dtSec;
        } else {
            timing.jumpTime = 0.f;
        }
        timing.hadJumped = pd.hasJumped;
        pd.jumpTime = timing.jumpTime;

        pd.isSupersonic = playerInfo->is_supersonic();
        pd.isDemoed = playerInfo->demolished_timeout() >= 0.f;
        pd.demoRespawnTimer = pd.isDemoed ? rawDemoTimeout : 0.f;

        // Approximate airtime timers (used for HasFlipOrJump behavior)
        if (pd.hasJumped && !pd.isJumping) {
            timing.airTime += dtSec;

            timing.airTimeSinceJump = pd.hasJumped ? timing.airTime : 0.f;
        }
        else {
            timing.airTime = 0.f;
            timing.airTimeSinceJump = 0.f;
        }

        pd.airTime = timing.airTime;
        pd.airTimeSinceJump = timing.airTimeSinceJump;

        // Training's airState mapping:
        // 0 OnGround, 1 Jumping, 2 DoubleJumping, 3 Dodging, 4 InAir
        pd.airState =
            pd.isOnGround ? 0 :
            pd.isJumping ? 1 :
            pd.hasDoubleJumped ? 2 :
            pd.isFlipping ? 3 :
            4;

        const bool flipAvailable =
            pd.hasJumped &&              // <- key: require first jump used
            !pd.hasDoubleJumped &&
            !pd.hasDodged &&
            !pd.isOnGround;

        pd.dodgeTimeout = flipAvailable
            ? std::max(0.f, RLConst::DOUBLEJUMP_MAX_DELAY - pd.airTimeSinceJump)
            : 0.f;

        // Training's demolishedTimeout: 0 when not demoed
        pd.demolishedTimeout = pd.isDemoed ? pd.demoRespawnTimer : 0.f;

        return pd;
    }

    GameState ToGameState(
        rlbot::flat::GamePacket const* packet,
        float dtSec,
        std::vector<PlayerTimingState>& playerTiming,
        RLBotClientExtension& extension,
        RocketSim::GameMode gameMode)
    {
        GameState gs = {};
        gs.deltaTime = dtSec;
        gs.gameMode = gameMode;

        auto players = packet->players();
        if (players) {
            const int n = (int)players->size();
            if ((int)playerTiming.size() < n)
                playerTiming.resize(n);

            gs.players.reserve(n);
            for (int i = 0; i < n; i++) {
                const rlbot::flat::PlayerInfo* playerInfo = players->Get(i);
                extension.CapturePlayerPacket(i, playerInfo);
                Player player = ToPlayer(playerInfo, dtSec, playerTiming[i]);
                player.index = i;
                gs.players.push_back(player);
            }
        }

        static_cast<PhysState&>(gs.ball) = ToPhysObj(packet->balls()->Get(0)->physics());

        // GameState's constructor sizes these for soccar; resize to this mode's count.
        // Defaulting to "on" also serves the bad-pad-count fallback below.
        const int boostLocationAmount = gs.GetBoostLocationAmount();
        gs.boostPads.assign(boostLocationAmount, true);
        gs.boostPadsInv.assign(boostLocationAmount, true);
        gs.boostPadTimers.assign(boostLocationAmount, 0.f);
        gs.boostPadTimersInv.assign(boostLocationAmount, 0.f);

        auto boostPadStates = packet->boost_pads();
        if ((int)boostPadStates->size() != boostLocationAmount) {
            if (rand() % 20 == 0) { // Don't spam-log as that will lag the bot
                RG_LOG(
                    "RLBotClient ToGameState(): Bad boost pad amount, expected "
                    << boostLocationAmount << " but got " << boostPadStates->size()
                );
            }
        }
        else {
            for (int i = 0; i < boostLocationAmount; i++) {
                gs.boostPads[i] = boostPadStates->Get(i)->is_active();
                gs.boostPadsInv[boostLocationAmount - i - 1] = gs.boostPads[i];

                gs.boostPadTimers[i] = boostPadStates->Get(i)->timer();
                gs.boostPadTimersInv[boostLocationAmount - i - 1] = gs.boostPadTimers[i];
            }
        }

        return gs;
    }
} // anonymous namespace

RLBotBotManager::RLBotBotManager(
    std::shared_ptr<const SharedBotContext> ctx,
    bool const batchHivemind) noexcept
    : rlbot::BotManagerBase(batchHivemind, RLBotBotManager::spawn)
{
    SpawnContextStorage() = std::move(ctx);
}

std::unique_ptr<rlbot::Bot> RLBotBotManager::spawn(
    std::unordered_set<unsigned> indices_,
    unsigned const team_,
    std::string name_) noexcept
{
    auto ctx = SpawnContextStorage();
    if (!ctx || !ctx->agent) {
        std::fprintf(stderr, "RLBotBotManager: shared bot context was not initialized\n");
        std::exit(EXIT_FAILURE);
    }

    try {
        return std::make_unique<RLBotBot>(
            std::move(indices_), team_, std::move(name_), std::move(ctx));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RLBotBotManager: failed to initialize bot: %s\n", error.what());
        std::exit(EXIT_FAILURE);
    }
}

RLBotBot::RLBotBot(std::unordered_set<unsigned> indices_,
    unsigned const team_,
    std::string name_,
    std::shared_ptr<const SharedBotContext> ctx)
    : rlbot::Bot(std::move(indices_), team_, std::move(name_))
    , ctx_(std::move(ctx))
{
    // Default to the agent's declared mode until initialize() supplies the live match mode.
    m_gameMode = ctx_->agent->gameMode;

    extension_ = ctx_->extensionFactory
        ? ctx_->extensionFactory(team_, indices)
        : std::make_unique<RLBotClientExtension>();
    if (!extension_)
        extension_ = std::make_unique<RLBotClientExtension>();

    if (!ctx_->captureDirectory.empty() || ctx_->captureSink)
        capture_ = std::make_unique<RLBotCapture>(*ctx_, *this, extension_->GetPolicyObserver());

    policyRuntime_ = std::make_unique<RLBotPolicyRuntime>(
        *ctx_->agent,
        ctx_->modelsFolder,
        ctx_->useGPU,
        capture_ ? capture_.get() : extension_->GetPolicyObserver(),
        static_cast<unsigned>(indices.size()));

    std::set<unsigned> sorted(std::begin(indices), std::end(indices));
    for (auto const& index : sorted)
        std::printf("Team %u Index %u: %s created\n", team_, index, name.c_str());
}

RLBotBot::~RLBotBot() {}

void RLBotBot::initialize(
    rlbot::flat::ControllableTeamInfo const* controllableTeamInfo_,
    rlbot::flat::FieldInfo const* fieldInfo_,
    rlbot::flat::MatchConfiguration const* matchConfiguration_) noexcept
{
    if (capture_) capture_->Initialize(controllableTeamInfo_, fieldInfo_, matchConfiguration_);

    // Runs once before the first update(), so update() can trust m_gameMode.
    if (matchConfiguration_)
        m_gameMode = ToRocketSimGameMode(matchConfiguration_->game_mode(), ctx_->agent->gameMode);
}

void RLBotBot::ClearExtensionControls(bool resetExtensionState)
{
    m_ballPredArena.reset();
    m_ballPredTickCount = 0;
    policyRuntime_->RequestObservationReset();

    for (unsigned index : indices) {
        ResetPerBotState(m_botState[index]);
        if (index < m_playerTiming.size()) {
            m_playerTiming[index] = {};
            extension_->ResetCapturedPlayerPacket(index);
        }
        if (resetExtensionState)
            extension_->ResetBot(index);

        setOutput(index, {});
    }
}

void RLBotBot::ResetDrillObservedPlayers(GameState& gs)
{
    for (unsigned index : indices) {
        if (index < gs.players.size()) {
            Player& player = gs.players[index];
            player.lastControls = {};
            player.prevAction = {};
            player.wasOnGroundDuringStep = player.isOnGround;
            player.obsOnGround = player.isOnGround;
        }
    }
}

bool RLBotBot::HandleExtensionStep(GameState& gs, float curTime, bool& forceInferAction)
{
    RLBotClientExtensionStep step = extension_->Step(gs, indices, curTime);
    if (step.desiredGameState)
        sendDesiredGameState(std::move(*step.desiredGameState));

    if (step.clearControls)
        ClearExtensionControls(step.resetExtensionState);

    if (step.resetTickCounter)
        ticks = 0;

    if (step.resetObservedPlayerState)
        ResetDrillObservedPlayers(gs);

    forceInferAction = forceInferAction || step.forceInferAction;

    return step.consumedPacket;
}

void RLBotBot::update(rlbot::flat::GamePacket const* packet,
    rlbot::flat::BallPrediction const* ballPrediction_) noexcept
{
    if (capture_) capture_->Begin(packet, ballPrediction_);

    // Record every return path, including ball-absent replay/teardown packets.
    struct CaptureEnd {
        RLBotCapture* capture;
        RLBotBot& bot;
        ~CaptureEnd() { if (capture) capture->End(bot); }
    } captureEnd{capture_.get(), *this};

    if (!packet || !packet->match_info() || !packet->balls() || packet->balls()->size() == 0) {
        return;
    }

    extension_->BeginCallback();

    const float curTime = packet->match_info()->seconds_elapsed();
    const bool firstPacket = ticks < 0;
    float deltaTime = firstPacket ? 0.f : curTime - prevTime;
    prevTime = curTime;

    if (deltaTime < 0.f)
        deltaTime = 0.f;

    const int tickSkip = std::max(1, ctx_->agent->tickSkip);
    const int actionDelay = std::clamp(ctx_->agent->actionDelay, 0, tickSkip);
    const int ticksElapsed = firstPacket ? 0 : std::max(0, (int)roundf(deltaTime * 120.f));

    if (firstPacket)
        ticks = 0;
    else
        ticks += ticksElapsed;

    GameState gs = ToGameState(
        packet,
        deltaTime,
        m_playerTiming,
        *extension_,
        m_gameMode);

    bool forceInferAction = false;
    if (HandleExtensionStep(gs, curTime, forceInferAction))
        return;

    const bool inferAction = forceInferAction || firstPacket || ticks >= tickSkip;
    const bool queuedActionDue = !firstPacket && ticks >= actionDelay;
    
    if (inferAction)
        ticks = 0;

    if (ctx_->agent->useBallPrediction) {
        if (!m_ballPredArena || m_ballPredArena->gameMode != gs.gameMode) {
            RocketSim::ArenaConfig predictionConfig = {};
            predictionConfig.engine = RocketSim::EngineFromVersion(ctx_->agent->GetBallPredictionEngineVersion());
            m_ballPredArena.reset(RocketSim::Arena::Create(gs.gameMode, predictionConfig));

            m_ballPredTickCount = 0;
        }

        if (!firstPacket)
            m_ballPredTickCount += (std::uint64_t)ticksElapsed;

        m_ballPredArena->tickCount = m_ballPredTickCount;
        m_ballPredArena->ball->SetState(gs.ball);
        gs.lastArena = m_ballPredArena.get();
        gs.lastTickCount = m_ballPredArena->tickCount;
    }

    policyRuntime_->PrepareObservationState(gs);
    RLGC::ActionParser& actionParser = policyRuntime_->GetActionParser();

    // Prepare every controlled car before observing any car.
    // Stable rows keep observations, masks, actions and diagnostics tied to the same player.
    std::vector<unsigned> activeIndices;
    for (unsigned index : indices) {
        if (index < gs.players.size()) {
            activeIndices.push_back(index);
        } else {
            setOutput(index, {});
            m_botState.erase(index);
        }
    }

    std::sort(activeIndices.begin(), activeIndices.end());

    std::unordered_set<unsigned> activatedPendingIndices;
    for (unsigned index : activeIndices) {
        auto& st = m_botState[index];
        bool activatedPendingAction = false;
        if (!st.initialized) {
            st.initialized = true;

            st.action = RLGC::Action{};
            st.controls = RLGC::Action{};
            st.pendingActionMacros.clear();
            st.activeActionMacros.clear();
            st.obsPrevAction = {};
            st.activeMacroIndex = 0;
            st.activeMacroTicksLeft = 0;
            extension_->ResetBot(index);
        }

        if (!firstPacket)
            AdvanceActiveMacros(st, ticksElapsed);

        if (queuedActionDue) {
            const bool hadPendingMacros = !st.pendingActionMacros.empty();
            ActivatePendingMacros(st);
            if (hadPendingMacros) {
                activatedPendingAction = true;
                extension_->ActivatePendingAction(index);
            }
        }

        auto& localPlayer = gs.players[index];
        localPlayer.lastControls = (CarControls)st.controls;
        localPlayer.prevAction = st.obsPrevAction;

        if (activatedPendingAction)
            activatedPendingIndices.insert(index);
    }

    const std::vector<int> actions = inferAction ? policyRuntime_->InferBatch(activeIndices, gs) : std::vector<int>{};

    for (size_t row = 0; row < activeIndices.size(); ++row) {
        const unsigned index = activeIndices[row];
        auto& st = m_botState[index];
        auto& localPlayer = gs.players[index];
        bool activatedPendingAction = activatedPendingIndices.contains(index);
        if (inferAction) {
            const int actionId = actions[row];
            st.action = actionParser.ParseAction(actionId, localPlayer, gs);
            if (actionDelay != 0 && actionParser.IsMacroAction(actionId, localPlayer, gs)) {
                RG_LOG("RLBotClient: action macros require actionDelay == 0; falling back to a single parsed action.");
                st.pendingActionMacros.clear();
                st.pendingActionMacros.push_back({ st.action, tickSkip - actionDelay });
            } else {
                st.pendingActionMacros = BuildActionMacros(
                    actionParser,
                    actionId,
                    localPlayer,
                    gs,
                    tickSkip - actionDelay,
                    st.action);
                if (actionDelay != 0 && IsSegmentedActionMacro(st.pendingActionMacros, tickSkip - actionDelay)) {
                    RG_LOG("RLBotClient: action parser returned segmented macros with actionDelay != 0; falling back to a single parsed action.");
                    st.pendingActionMacros.clear();
                    st.pendingActionMacros.push_back({ st.action, tickSkip - actionDelay });
                }
            }
            st.obsPrevAction = st.pendingActionMacros.empty() ? st.action : st.pendingActionMacros.back().action;
            extension_->RecordInference(index, st.pendingActionMacros, actionId);
        }

        // If actionDelay is zero, apply action immediately
        if (inferAction && actionDelay == 0) {
            const bool hadPendingMacros = !st.pendingActionMacros.empty();
            ActivatePendingMacros(st);
            if (hadPendingMacros) {
                activatedPendingAction = true;
                extension_->ActivatePendingAction(index);
            }
        }

        const auto& c = st.controls;
        setOutput(index, {
            c.throttle,
            c.steer,
            c.pitch,
            c.yaw,
            c.roll,
            c.jump > 0.5f,
            c.boost > 0.5f,
            c.handbrake > 0.5f,
            false,
            });

        const RLBotFrameEvent frame{
            .index = index,
            .player = localPlayer,
            .gameState = gs,
            .actionParser = actionParser,
            .selectedAction = st.action,
            .appliedControls = st.controls,
            .prevAction = localPlayer.prevAction,
            .activeActionMacros = st.activeActionMacros,
            .pendingActionMacros = st.pendingActionMacros,
            .activeMacroIndex = st.activeMacroIndex,
            .activeMacroTicksLeft = st.activeMacroTicksLeft,
            .tickSkip = tickSkip,
            .actionDelay = actionDelay,
            .ticksElapsed = ticksElapsed,
            .policyTick = firstPacket || inferAction ? 0 : ticks,
            .secondsElapsed = curTime,
            .gameTimeRemaining = packet->match_info()->game_time_remaining(),
            .matchPhase = (int)packet->match_info()->match_phase(),
            .deltaTime = deltaTime,
            .useGPU = ctx_->useGPU,
            .inferAction = inferAction,
            .activatedPendingAction = activatedPendingAction,
            .frameNum = packet->match_info()->frame_num(),
            .trainingEngineVersion = ctx_->agent->trainingEngineVersion,
            };

        extension_->RecordFrame(frame);
        if (capture_) capture_->Frame(frame);

        if (inferAction)
            extension_->AdvancePolicyStep(index);
    }

    if (inferAction)
        for (PlayerTimingState& timing : m_playerTiming)
            timing.wasOnGroundDuringStep = false;

    extension_->FinishCallback();

}
