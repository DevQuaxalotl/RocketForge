#pragma once

#include <QWidget>
#include <QString>
#include <QStringList>
#include <string>
#include <vector>

#include <Settings/ForgeDebugSettings.h>

class AgentComboBox;
class BuildComboBox;
class SegmentComboBox;
class DebugOptionsWidget;
class DebugBridge;
class QCheckBox;
class QComboBox;
class QPushButton;
class QLabel;
class QTimer;

/**
 * DebugTabWidget
 *
 * Top-level "Debug" mode page. Lets the user pick agent -> build -> segment and render
 * that segment's latest checkpoint (trains like normal rendering, metrics off).
 * Owns no worker: it emits StartDebugRequested / StopRequested and MainWindow dispatches.
 * Hosts the relocated DebugOptionsWidget. Persists selection + options to the agent's
 * forge_debug_settings.json (sticky, debounced auto-save).
 */
class DebugTabWidget : public QWidget
{
    Q_OBJECT
public:
    explicit DebugTabWidget(DebugBridge* bridge, QWidget* parent = nullptr);

    /** Reload the build/segment lists for the current agent (called when the tab is shown). */
    void RefreshLists();

    /** Switch to the training agent unless a debug run holds the selection. */
    void FollowTrainingAgent(const QString& agentName);

public slots:
    /** Flip controls for a debug run started/stopped from this tab. */
    void SetRunActive(bool active);
    /** Disable Start while an external run (queue training / benchmark) is active. */
    void SetExternalRunActive(bool active);
    /** Update the persisted per-reward debug selection (from the reward viz window). */
    void SetRewardSelection(QStringList enabledRewardNames);
    /** Update the persisted reward-viz controls (norm/display/force-all) from the window. */
    void SetRewardVizConfig(int normMode, int displayMode, bool forceDebugAll);
    /** Update the persisted critic head selection (from the reward viz window). */
    void SetCriticHeadSelection(QStringList enabledHeadNames);
    /** Update the persisted "Critic Heads" toggle (from the reward viz window). */
    void SetShowCriticHeads(bool show);

signals:
    void StartDebugRequested(RocketForgeTraining::ForgeDebugSettings settings, QString agentName,
                             QString stageName, QString checkpointFolder, bool isImported);
    void StopRequested();

private slots:
    void OnAgentChanged();
    void OnBuildChanged();
    void OnSegmentChanged();
    void OnStageChanged();
    void OnOptionsChanged();
    void OnStartClicked();
    void OnStopClicked();

private:
    void LoadForAgent(const QString& agentName);
    void SaveCurrent();
    void RefreshStageCombo();
    void RefreshTeamSizes();
    void RefreshStaticOpponentOptions(const QString& preferredSelectionKey = {});
    void UpdateStaticOpponentControls();
    void UpdateStartEnabled();
    void ScheduleSave();
    RocketForgeTraining::ForgeDebugSettings CurrentToSettings() const;

    AgentComboBox*      m_agentCombo   = nullptr;
    BuildComboBox*      m_buildCombo   = nullptr;
    SegmentComboBox*    m_segmentCombo = nullptr;
    QComboBox*          m_stageCombo   = nullptr;
    QLabel*             m_stageTrainedHint = nullptr;
    QCheckBox*          m_staticOpponentCheck = nullptr;
    QComboBox*          m_staticOpponentCombo = nullptr;
    DebugOptionsWidget* m_options      = nullptr;
    QPushButton*        m_startBtn     = nullptr;
    QPushButton*        m_stopBtn      = nullptr;
    QLabel*             m_statusLabel  = nullptr;
    QTimer*             m_saveTimer    = nullptr;

    bool    m_runActive          = false;
    bool    m_externalRunActive  = false;
    bool    m_suspendSave        = false;
    bool    m_selectedStageAvailable = true;
    bool    m_staticOpponentOptionsAvailable = false;
    QString m_currentAgent;
    QString m_stageUnavailableText;
    QString m_staticOpponentUnavailableText;

    std::vector<std::string> m_debugEnabledRewards;
    bool                     m_hasRewardSelection = false;
    std::vector<std::string> m_debugEnabledCriticHeads;
    bool                     m_hasCriticHeadSelection = false;
    bool                     m_showCriticHeads = true;
    int                      m_normMode      = 0;     // default: RAW_WEIGHTED
    int                      m_displayMode   = 0;     // default: EARNED
    bool                     m_forceDebugAll = false;
};
