#pragma once

#include <QObject>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <utility>

#include <ForgeDebugger.h>

/**
 * DebugBridge
 *
 * Thread-safe bridge consolidating all debug state.
 *
 * GUI thread writes pending values via Request*() slots.
 * Trainer thread applies them via ApplyPendingSettings() each frame and reads
 * CurrentSettings() for per-episode overrides (team size, force setter, max time).
 * Snapshots are posted to the main thread via SnapshotReady.
 *
 * Must live on the main thread (owned by MainWindow).
 **/
class DebugBridge : public QObject, public ForgeDebug::IDebugRenderer
{
    Q_OBJECT

public:
    explicit DebugBridge(QObject* parent = nullptr);

    /** Initialize _current and all pending atomics from persisted settings.
        Call from main thread before dispatching RunTraining. **/
    void InitializeFromSettings(const ForgeDebug::DebugSettings& settings);

    /** Render-thread only: returns a newly requested player index once, or -1 when unchanged. **/
    int TakePendingPlayerIndex();

    /** Current applied settings. Trainer-thread-only read after ApplyPendingSettings(). **/
    const ForgeDebug::DebugSettings& CurrentSettings() const { return _current; }

    /** Returns the game speed atomic for the inference render loop's time scale. **/
    std::atomic<float>& GameSpeedAtomic() { return _pendingGameSpeed; }

    /** Flush game-control pending values (team size, force setter, max time, game speed)
        into _current. Call from trainer thread in onStageCreated before reading
        CurrentSettings() — these don't need a collector or game state. **/
    void FlushGameControlPending();

    /** IDebugRenderer — trainer thread **/
    void ApplyPendingSettings(ForgeDebug::RewardDebugCollector& collector,
                              const RLGC::GameState& gs) override;
    void Render(const ForgeDebug::RewardDebugSnapshot& snapshot) override;

signals:
    void SnapshotReady(ForgeDebug::RewardDebugSnapshot snapshot);

public slots:
    /*** GUI-side requests — main thread ***/
    void RequestNormMode(int modeIndex);
    void RequestDisplayMode(int modeIndex);
    void RequestForceDebugAll(bool enabled);
    void RequestPlayerIndex(int idx);
    void RequestGameSpeed(float speed);
    void RequestTeamSizeOverride(int index);          // -1 = no override
    void RequestForceSingleStateSetter(bool enabled);
    void RequestForceStateSetterIndex(int index);
    void RequestMaxEpisodeTimeEnabled(bool enabled);
    void RequestMaxEpisodeTime(float seconds);
    void RequestRewardDebugEnabled(const std::string& rewardName, bool enabled);

private:
    std::atomic<bool>  _snapshotPending{ false };     // frame-drop guard

    /** Reward visualizer pending(-1 = no change for int slots) */
    std::atomic<int>   _pendingNormMode{ -1 };
    std::atomic<int>   _pendingDisplayMode{ -1 };
    std::atomic<int>   _pendingPlayerIndex{ -1 };
    std::atomic<int>   _pendingForceDebugAll{ -1 };   // -1=no change, 0=false, 1=true

    /** Game control pending (-2 = no change for teamSizeOverride; -1.0f = no change for maxTime) */
    std::atomic<float> _pendingGameSpeed{ 1.0f };     // read every frame, not consumed
    std::atomic<int>   _pendingTeamSizeOverride{ -2 };// -2=no change, -1=no override, >=0=index
    std::atomic<int>   _pendingForceSetter{ -1 };     // -1=no change, 0=false, 1=true
    std::atomic<int>   _pendingForceSetterIndex{ -2 };// -2=no change, otherwise the index
    std::atomic<int>   _pendingMaxTimeEnabled{ -1 };  // -1=no change, 0=false, 1=true
    std::atomic<float> _pendingMaxTime{ -1.0f };      // -1=no change

    /** Per-reward toggles: variable-length, so guarded by a mutex rather than an atomic. */
    std::mutex _toggleMutex;
    std::vector<std::pair<std::string, bool>> _pendingRewardToggles;

    /** One-shot seeding of the persisted selection on the first trainer-thread apply. */
    std::vector<std::string> _initialSelection;
    bool _hasInitialSelection = false;
    std::atomic<bool> _initialSelectionApplied{ false };

    ForgeDebug::DebugSettings _current;               // trainer-thread-only after init
};
