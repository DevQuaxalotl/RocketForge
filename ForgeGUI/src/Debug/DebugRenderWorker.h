#pragma once

#include <QObject>
#include <QString>

#include <atomic>

class DebugBridge;

/**
 * DebugRenderWorker
 *
 * Runs RocketForge::RunInferenceDebugRender on its own QThread for the GUI Debug tab.
 * Mirrors the TrainingWorker pattern: created without a parent, moved onto a dedicated
 * thread, jobs dispatched via a queued Run() invocation. The blocking render loop runs
 * until Stop() flips the atomic, then Finished/Failed is emitted back to the main thread.
 */
class DebugRenderWorker : public QObject
{
    Q_OBJECT

public:
    explicit DebugRenderWorker(QObject* parent = nullptr);

    /** Share the main-thread DebugBridge (renderer + live time-scale/settings source). */
    void SetDebugBridge(DebugBridge* bridge);

public slots:
    /** Blocks on the render loop until Stop() flips the atomic. Emits Finished/Failed. */
    void Run(QString agentName, QString stageName, QString checkpointFolder,
             bool isImported, bool showRewards, int playerIndex,
             QString buildName, bool includeStaticOpponent, QString staticOpponentSelectionKey);

    /** Thread-safe request to end the current render (atomic flag, safe to call cross-thread). */
    void Stop();

signals:
    void Finished();

    /** simulatorUnavailable marks a RocketSim::EngineUnavailableError reason. */
    void Failed(const QString& error, bool simulatorUnavailable);

private:
    DebugBridge* m_bridge = nullptr;
    std::atomic<bool> m_stop{ false };
};
