#include "DebugBridge.h"

#include <ForgeDebugger.h>
#include <RLGymCPP/Gamestates/GameState.h>

DebugBridge::DebugBridge(QObject* parent) : QObject(parent) {}

void DebugBridge::InitializeFromSettings(const ForgeDebug::DebugSettings& settings)
{
    _current = settings;
    _pendingNormMode.store((int)settings.normMode);
    _pendingDisplayMode.store((int)settings.displayMode);
    _pendingPlayerIndex.store(settings.playerIndex);
    _pendingForceDebugAll.store(settings.forceDebugAll ? 1 : 0);
    _pendingGameSpeed.store(settings.gameSpeed);
    _pendingTeamSizeOverride.store(settings.teamSizeOverrideIndex);
    _pendingForceSetter.store(settings.forceSingleStateSetter ? 1 : 0);
    _pendingForceSetterIndex.store(settings.forceStateSetterIndex);
    _pendingMaxTimeEnabled.store(settings.maxEpisodeTimeEnabled ? 1 : 0);
    _pendingMaxTime.store(settings.maxEpisodeTime);

    _initialSelection      = settings.debugEnabledRewards;
    _hasInitialSelection   = settings.hasRewardSelection;
    _initialSelectionApplied.store(false);
}

int DebugBridge::TakePendingPlayerIndex()
{
    return _pendingPlayerIndex.exchange(-1);
}

void DebugBridge::FlushGameControlPending()
{
    _current.gameSpeed = _pendingGameSpeed.load(std::memory_order_relaxed);

    int tsOverride = _pendingTeamSizeOverride.exchange(-2);
    if (tsOverride != -2)
        _current.teamSizeOverrideIndex = tsOverride;

    int forceSetter = _pendingForceSetter.exchange(-1);
    if (forceSetter >= 0)
        _current.forceSingleStateSetter = (forceSetter != 0);

    int forceSetterIndex = _pendingForceSetterIndex.exchange(-2);
    if (forceSetterIndex != -2)
        _current.forceStateSetterIndex = forceSetterIndex;

    int maxTimeEnabled = _pendingMaxTimeEnabled.exchange(-1);
    if (maxTimeEnabled >= 0)
        _current.maxEpisodeTimeEnabled = (maxTimeEnabled != 0);

    float maxTime = _pendingMaxTime.exchange(-1.0f);
    if (maxTime >= 0.0f)
        _current.maxEpisodeTime = maxTime;
}

void DebugBridge::ApplyPendingSettings(ForgeDebug::RewardDebugCollector& collector,
                                        const RLGC::GameState& gs)
{
    // Apply persisted selection exactly once on first apply
    if (_hasInitialSelection && !_initialSelectionApplied.exchange(true))
        collector.SetDebugSelection(_initialSelection);

    int normMode = _pendingNormMode.exchange(-1);
    if (normMode >= 0 && normMode < ForgeDebug::NORMALIZATION_MODE_COUNT) {
        _current.normMode = static_cast<ForgeDebug::NormalizationMode>(normMode);
        collector.SetNormalizationMode(_current.normMode);
    }

    int displayMode = _pendingDisplayMode.exchange(-1);
    if (displayMode >= 0 && displayMode < static_cast<int>(ForgeDebug::RewardDisplayMode::COUNT)) {
        _current.displayMode = static_cast<ForgeDebug::RewardDisplayMode>(displayMode);
        collector.SetRewardDisplayMode(_current.displayMode);
    }

    int forceAll = _pendingForceDebugAll.exchange(-1);
    if (forceAll >= 0) {
        _current.forceDebugAll = (forceAll != 0);
        collector.SetForceDebugAll(_current.forceDebugAll);
    }

    // Drain per-reward toggles queue
    {
        std::lock_guard<std::mutex> lock(_toggleMutex);
        for (const auto& [name, on] : _pendingRewardToggles)
            collector.SetRewardDebugEnabled(name, on);
        _pendingRewardToggles.clear();
    }

    int playerIdx = _pendingPlayerIndex.exchange(-1);
    if (playerIdx >= 0) {
        int clamped = gs.players.empty() ? 0
                    : std::min(playerIdx, static_cast<int>(gs.players.size()) - 1);
        _current.playerIndex = clamped;
        collector.SetPlayerIndex(clamped);
    }

    // Game control — update _current so TrainingStage reads latest values each episode
    _current.gameSpeed = _pendingGameSpeed.load(std::memory_order_relaxed);

    int tsOverride = _pendingTeamSizeOverride.exchange(-2);
    if (tsOverride != -2)
        _current.teamSizeOverrideIndex = tsOverride;

    int forceSetter = _pendingForceSetter.exchange(-1);
    if (forceSetter >= 0)
        _current.forceSingleStateSetter = (forceSetter != 0);

    int forceSetterIndex = _pendingForceSetterIndex.exchange(-2);
    if (forceSetterIndex != -2)
        _current.forceStateSetterIndex = forceSetterIndex;

    int maxTimeEnabled = _pendingMaxTimeEnabled.exchange(-1);
    if (maxTimeEnabled >= 0)
        _current.maxEpisodeTimeEnabled = (maxTimeEnabled != 0);

    float maxTime = _pendingMaxTime.exchange(-1.0f);
    if (maxTime >= 0.0f)
        _current.maxEpisodeTime = maxTime;
}

void DebugBridge::Render(const ForgeDebug::RewardDebugSnapshot& snapshot)
{
    if (_snapshotPending.exchange(true))
        return;

    QMetaObject::invokeMethod(this, [this, snap = snapshot]() {
        _snapshotPending.store(false);
        emit SnapshotReady(snap);
    }, Qt::QueuedConnection);
}

void DebugBridge::RequestNormMode(int modeIndex)         { _pendingNormMode.store(modeIndex); }
void DebugBridge::RequestDisplayMode(int modeIndex)      { _pendingDisplayMode.store(modeIndex); }
void DebugBridge::RequestForceDebugAll(bool enabled)     { _pendingForceDebugAll.store(enabled ? 1 : 0); }
void DebugBridge::RequestPlayerIndex(int idx)            { _pendingPlayerIndex.store(std::max(0, idx)); }
void DebugBridge::RequestGameSpeed(float speed)          { _pendingGameSpeed.store(std::max(0.1f, speed)); }
void DebugBridge::RequestTeamSizeOverride(int index)     { _pendingTeamSizeOverride.store(index); }
void DebugBridge::RequestForceSingleStateSetter(bool en) { _pendingForceSetter.store(en ? 1 : 0); }
void DebugBridge::RequestForceStateSetterIndex(int idx)  { _pendingForceSetterIndex.store(std::max(-1, idx)); }
void DebugBridge::RequestMaxEpisodeTimeEnabled(bool en)  { _pendingMaxTimeEnabled.store(en ? 1 : 0); }
void DebugBridge::RequestMaxEpisodeTime(float seconds)   { _pendingMaxTime.store(std::max(0.5f, seconds)); }

void DebugBridge::RequestRewardDebugEnabled(const std::string& rewardName, bool enabled)
{
    std::lock_guard<std::mutex> lock(_toggleMutex);
    _pendingRewardToggles.emplace_back(rewardName, enabled);
}
