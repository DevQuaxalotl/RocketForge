#include "DebugRenderWorker.h"

#include "DebugBridge.h"

#include <DebugRender/InferenceDebugRenderLoop.h>
#include <RocketSim/Engine.h>

#include <exception>

DebugRenderWorker::DebugRenderWorker(QObject* parent) : QObject(parent) {}

void DebugRenderWorker::SetDebugBridge(DebugBridge* bridge)
{
    m_bridge = bridge;
}

void DebugRenderWorker::Run(QString agentName, QString stageName, QString checkpointFolder,
                            bool isImported, bool showRewards, int playerIndex,
                            QString buildName, bool includeStaticOpponent, QString staticOpponentSelectionKey)
{
    m_stop.store(false, std::memory_order_relaxed);

    try {
        RocketForge::InferenceDebugRenderConfig cfg;
        cfg.agentName        = agentName.toStdString();
        cfg.stageName        = stageName.toStdString();
        cfg.checkpointFolder = checkpointFolder.toStdString();
        cfg.isImported       = isImported;
        cfg.showRewards      = showRewards;
        cfg.playerIndex      = playerIndex;

        cfg.buildName                    = buildName.toStdString();
        cfg.includeStaticOpponent        = includeStaticOpponent;
        cfg.staticOpponentSelectionKey   = staticOpponentSelectionKey.toStdString();

        // Team size is driven by the DebugBridge's live DebugSettings (teamSizeOverrideIndex,
        // applied through TrainingStage::ChooseTeamSize). Leaving teamSizeOverride at 0 makes the
        // render loop defer to that path rather than forcing a fixed size.
        cfg.teamSizeOverride = 0;

        if (m_bridge) {
            cfg.debugRenderer = m_bridge;
            cfg.timeScale     = &m_bridge->GameSpeedAtomic();
            cfg.debugSettings = &m_bridge->CurrentSettings();

            // Game controls and focus must apply even when no reward collector exists to call ApplyPendingSettings.
            cfg.pollControls       = [this] { m_bridge->FlushGameControlPending(); };
            cfg.consumePlayerIndex = [this] { return m_bridge->TakePendingPlayerIndex(); };
        }

        cfg.shouldStop = [this]() { return m_stop.load(std::memory_order_relaxed); };

        RocketForge::RunInferenceDebugRender(cfg);

        emit Finished();
    } catch (const RocketSim::EngineUnavailableError& e) {
        emit Failed(QString::fromUtf8(e.what()), true);
    } catch (const std::exception& e) {
        emit Failed(QString::fromUtf8(e.what()), false);
    } catch (...) {
        emit Failed("Unknown error during debug render.", false);
    }
}

void DebugRenderWorker::Stop()
{
    m_stop.store(true, std::memory_order_relaxed);
}
