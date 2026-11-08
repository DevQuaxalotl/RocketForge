#pragma once

#include <QElapsedTimer>
#include <QMainWindow>
#include <QMap>
#include <QSet>
#include <QString>
#include <memory>
#include <vector>

#include <ParamTesting/ParamTesting.h>
#include "RemoteSync/RemoteSyncCommand.h"
#include "RewardVisualizer/TrainingLogParser.h"
#include "Settings/ForgeSettings.h"
#include "Settings/ForgeDebugSettings.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

namespace ForgeGUI { class RemoteSyncPanel; }

class ConsoleCapture;
class TrainingQueueWidget;
class TrainingWorker;
class DebugRenderWorker;
class DebugBridge;
class DebugTabWidget;
class RewardVisualizerWindow;
class TrainingSetupWindow;
class ModelBenchmarkWidget;
class ParamTestingWidget;
class ImpactAnalysisWidget;
class VersionManagementDialog;
class TrainingMonitorWindow;
class SkillRatingsPanel;
struct TrainingMonitorSnapshot;
class QCheckBox;
class QCloseEvent;
class QEvent;
class QGroupBox;
class QLabel;
class QPushButton;
class QTabBar;
class QStackedWidget;
class QThread;
class QTimer;

/**
 * MainWindow
 *
 * The application's primary window. It provides a GUI for the ForgeSettings-based
 * stage-queue training workflow. The left pane contains the agent selector, global
 * settings, and the TrainingQueueWidget. The right pane shows live training stats
 * and the console log.
 *
 * Architecture overview:
 *   - Training runs on a background QThread via TrainingWorker so the UI stays
 *     responsive throughout a run.
 *   - Stage configuration is owned by TrainingQueueWidget / TrainingStageRowWidget.
 *   - Global settings (metrics, prune) live in the main window; debug/render options live in the Debug tab (DebugTabWidget).
 *   - std::cout is redirected through ConsoleCapture, which emits each line as
 *     a Qt signal. TrainingLogParser classifies the lines into typed events
 *     (TrainingStat, RunInfo, SkillRating, Error, Raw) that are routed to the
 *     appropriate panel by OnLogEvent().
 *   - The Skill card (SkillRatingsPanel) displays the latest parsed skill-rating result for each mode.
 *   - Settings are auto-saved (debounced 500 ms) whenever the user changes anything
 *     in the queue or global settings.
 *
 * The member functions are defined across several files, one per section below:
 *   mainwindow.cpp                    window construction, theme, window events, training monitor, log panel
 *   MainWindowSettings.cpp            agent/build selection and settings persistence
 *   MainWindowRunControl.cpp          queue training and debug render runs, and locking other tabs while one is active
 *   MainWindowParamTestingPreview.cpp showing a Param Testing run in the Training tab
 *   MainWindowDashboard.cpp           the live stats cards, log event routing, and ETA tracking
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event) override;
    void changeEvent(QEvent* event) override;

private slots:
    /**************** Settings (MainWindowSettings.cpp) ****************/
    void OnAgentChanged();
    void OnBuildChanged();
    void OnNewBuildClicked();
    void OnNewBotClicked();
    void OnRenameBuildClicked();
    void OnQueueChanged();
    void OnPruneToggleClicked(bool checked);
    void OnStaticOpponentsClicked();
    void OnSkillReferencesClicked();
    void OnRewardTableClicked();
    void SaveSettingsNow();
    void FlushPendingSettingsSave();

    /**************** Run control (MainWindowRunControl.cpp) ****************/
    void OnStartTrainingClicked();
    void OnSaveAndStopClicked();
    void OnSaveAndSkipClicked();
    void OnTrainingFinished();
    void OnTrainingError(const QString& message, bool simulatorUnavailable);
    void OnStageStarted(int index, QString stageName, QString versionId);
    void OnStageCompleted(int index);
    void OnStageSkipped(int index, QString reason);
    void OnStartDebugRequested(RocketForgeTraining::ForgeDebugSettings settings, QString agentName,
                               QString stageName, QString checkpointFolder, bool isImported);
    void OnDebugRenderFinished();
    void OnDebugRenderFailed(const QString& error, bool simulatorUnavailable);
    void OnModelBenchmarkRunningChanged(bool running);
    void OnParamTestingRunningChanged(bool running);
    void OnRemoteSyncRunningChanged(bool running, ForgeGUI::RemoteSyncKind kind);
    void OnRemoteSyncFinished(ForgeGUI::RemoteSyncKind kind, ForgeGUI::RemoteSyncOutcome outcome, int exitCode);
    void UpdateRemoteSyncPanelState();

    /**************** Param Testing preview (MainWindowParamTestingPreview.cpp) ****************/
    void OnParamTestingTrainingPreviewRequested(RocketForgeTraining::ParamTestingConfig config);
    void OnParamTestingPreviewStageStarted(int index, QString stageName, QString versionId);
    void OnParamTestingPreviewStageCompleted(int index, QString versionId);
    void OnParamTestingPreviewFinished();
    void OnParamTestingBenchmarkRunRequested();

    /**************** Dashboard (MainWindowDashboard.cpp) ****************/
    void OnLogEvent(TrainingLogEvent event);

private:
    friend class TrainingDashboardTests;

    MainWindow(QWidget* parent, bool loadSettings);

    /**************** Window (mainwindow.cpp) ****************/
    void SetupModeTabs();
    void SetupTrainingPane();
    void SetupLogCapture();
    void SetupWorkerThreads();
    void SetupDebugTab();
    void ConnectTrainingControls();
    void ConfigurePaneSizing();
    void ApplyTheme();
    void SetColorsEnabled(bool enabled, bool savePreference);
    void RestoreWindowState();
    void SaveWindowState() const;
    void SetupTrainingMonitor();
    void SetTrainingMonitorVisible(bool visible);
    void AppendLog(const QString& text, const QString& colorHex = {});

    /**************** Settings (MainWindowSettings.cpp) ****************/
    void LoadOrCreateForgeSettings();
    RocketForgeTraining::ForgeSettings BuildForgeSettingsFromUI() const;
    QString CurrentBuildName() const;
    void RefreshBuildCombo(const QString& preferredBuild = {});
    void UpdateStaticOpponentsButtonText();
    void SyncFileManagementContext();

    /**************** Run control (MainWindowRunControl.cpp) ****************/
    void SetTrainingActive(bool active);

    /** Tells the File Management tab which segment is training, so it can mark that row, then refreshes the tab. */
    void RefreshFileManagementRunContext(bool runActive, const QString& agentName);

    void UpdateBuildPromptState();
    void UpdateStatusBarState();

    /**************** Param Testing preview (MainWindowParamTestingPreview.cpp) ****************/
    void SetTrainingTabLockedForParamTesting(bool locked);

    /**************** Dashboard (MainWindowDashboard.cpp) ****************/
    void ConfigureDashboardBlocks();
    void UpdateProgressBlock(const QString& etaText = {});
    void LoadActiveTrainingDurationBaselines(const QString& segmentId);

    /** Resets the cards and clocks for a run that is about to start. */
    void StartTrainingSessionClock();

    /** Stops the clocks and clears the active stage once a run ends. */
    void StopTrainingSessionClock();

    /** Points the cards and ETA tracking at a stage that just started. */
    void BeginActiveStage(int index, const QString& buildName, const QString& stageName, const QString& versionId);

    /** Clears the active stage once it completes or is skipped. */
    void EndActiveStage();

    double CurrentRunTrainingSeconds() const;
    double CurrentSessionTrainingSeconds() const;
    void UpdateTrainingDurationDisplays();
    void ResetIterationMetrics(const QSet<QLabel*>& keep = {});
    void ResetSkillRatings();
    void SeedSkillRatingsFromActiveSegment();
    void UpdateSkillTestTimingRows();
    void UpdateSkillTestEtaStateFromRawLine(const QString& line);
    double EstimateRemainingSkillTestSeconds(quint64 target, quint64 current) const;
    void UpdateActiveStageEta();
    TrainingMonitorSnapshot BuildTrainingMonitorSnapshot() const;

    Ui::MainWindow*    ui        = nullptr;
    ConsoleCapture*    m_capture = nullptr;
    TrainingLogParser* m_parser  = nullptr;
    TrainingWorker*    m_worker  = nullptr;
    QThread*           m_thread  = nullptr;

    DebugRenderWorker* m_debugWorker = nullptr;
    QThread*           m_debugThread = nullptr;

    DebugBridge*            m_bridge              = nullptr;
    DebugTabWidget*         m_debugTab            = nullptr;
    RewardVisualizerWindow* m_vizWindow           = nullptr;
    TrainingSetupWindow*    m_trainingSetupWindow = nullptr;
    TrainingQueueWidget*    m_queueWidget         = nullptr;
    ModelBenchmarkWidget*   m_modelBenchmarkWidget = nullptr;
    ParamTestingWidget*     m_paramTestingWidget  = nullptr;
    ImpactAnalysisWidget*   m_impactAnalysisWidget = nullptr;
    VersionManagementDialog* m_fileManagementWidget = nullptr;
    QTabBar*                m_modeSelector        = nullptr;
    QPushButton*            m_colorToggleButton   = nullptr;
    QStackedWidget*         m_modeStack           = nullptr;
    QLabel*                 m_statusStateLabel    = nullptr;
    QGroupBox*              m_progressMetricsGroup = nullptr;

    /**************** Run state ****************/
    enum class ActiveRunMode { None, Queue, Debug };
    ActiveRunMode m_activeRunMode = ActiveRunMode::None;

    bool    m_trainingActive = false;
    bool    m_modelBenchmarkActive = false;
    bool    m_paramTestingActive = false;
    bool    m_remoteSyncActive = false;
    ForgeGUI::RemoteSyncPanel* m_remoteSyncPanel = nullptr;
    bool    m_paramTestingPreviewActive = false;
    bool    m_colorsEnabled = true;

    // NOTE: Always equals (m_activeRunMode == ActiveRunMode::Debug); the bool is a convenience for guard expressions.
    bool    m_debugRenderActive = false;

    /** Agent that owns the in-flight debug render, which the Debug tab selects independently of the Training tab. */
    QString m_debugRenderAgent;

    QString m_savedParamTestingQueueAgent;
    QString m_savedParamTestingQueueBuild;
    QString m_paramTestingPreviewAgent;
    QString m_paramTestingPreviewBuild;
    std::vector<RocketForgeTraining::StageSettings> m_savedParamTestingQueueStages;

    QString m_currentAgent;

    bool    m_stopRequested = false;
    bool    m_closeAfterTraining = false;
    QString m_runOutcome = "Idle";
    QString m_lastTrainingError;

    /**************** Dashboard cards ****************/
    std::unique_ptr<SkillRatingsPanel> m_skillRatings;

    /** Report key -> the iteration metric label it updates. */
    QMap<QString, QLabel*>     m_iterationMetricLabels;

    /** Labels the report being received has updated so far; the rest are cleared when it finishes. */
    QSet<QLabel*>              m_reportedMetricLabels;

    QLabel*                    m_progressTargetValueLabel = nullptr;
    QLabel*                    m_progressSegmentStepsValueLabel = nullptr;
    QLabel*                    m_progressRemainingValueLabel = nullptr;
    QLabel*                    m_progressEtaValueLabel = nullptr;
    QLabel*                    m_progressSegmentTimeValueLabel = nullptr;
    QLabel*                    m_totalTimeValueLabel = nullptr;
    QLabel*                    m_sessionTimeValueLabel = nullptr;

    /**************** Training monitor ****************/
    TrainingMonitorWindow* m_monitorWindow         = nullptr;
    QPushButton*           m_monitorToggleButton   = nullptr;
    QCheckBox*             m_monitorAutoShowCheck  = nullptr;

    /** True while the monitor is showing only because the main window was minimized; restoring hides it again. */
    bool m_monitorAutoShown = false;

    /** Set when the user closes the monitor while minimized, so it stays hidden until the next minimize. */
    bool m_monitorDismissedWhileMinimized = false;

    /**************** Settings auto-save ****************/
    QTimer* m_saveDebounceTimer  = nullptr;
    bool    m_autoSaveSuspended  = false;

    /**************** Live progress and ETA ****************/
    QTimer* m_trainingDurationRefreshTimer = nullptr;
    int     m_activeStageIndex   = -1;
    QString m_activeBuildName;
    QString m_activeStageName;
    QString m_activeVersionId;
    double  m_rollingStepsPerSec = 0.0;
    double  m_smoothedEtaSecs    = 0.0;
    double  m_avgSkillTestSecs   = 0.0;
    double  m_lastSkillTestSecs  = 0.0;
    double  m_activeSegmentTrainingSecondsBase = 0.0;
    double  m_activeLineageTrainingSecondsBase = 0.0;
    QElapsedTimer m_trainingDurationTimer;
    QElapsedTimer m_trainingSessionTimer;
    bool    m_trainingDurationTimerActive = false;
    bool    m_trainingSessionTimerActive = false;
    quint64 m_lastTotalTimesteps = 0;
    quint64 m_activeSegmentStartTimesteps = 0;
    quint64 m_nextSkillTestTimestep      = 0;
    quint64 m_skillTestIntervalTimesteps = 0;
    bool    m_hasNextSkillTest           = false;
    bool    m_hasLastSkillTest           = false;
    bool    m_skillTestInProgress        = false;
};
