#include <RocketSim/Engine.h>
#include "MainWindow/mainwindow.h"
#include "ui_mainwindow.h"

#include "DebugBridge.h"
#include "Debug/DebugRenderWorker.h"
#include "Debug/DebugTabWidget.h"
#include "MainWindow/SkillRatingsPanel.h"
#include "MainWindow/TrainingStartPreflight.h"
#include "ModelBenchmark/ModelBenchmarkWidget.h"
#include "ParamTesting/ParamTestingWidget.h"
#include "RemoteSync/RemoteSyncPanel.h"
#include "RewardVisualizer/RewardVisualizerWindow.h"
#include "SimulatorUnavailableDialog.h"
#include "ThemeUtils.h"
#include "TrainingSetup/TrainingQueueWidget.h"
#include "TrainingSetup/TrainingWorker.h"
#include "VersionManagementDialog.h"
#include "Widgets/TrainingMonitorWindow.h"

#include <StageQueue/StageQueueRunner.h>
#include <Storage/TrainingBuildLock.h>

#include <QLabel>
#include <QMessageBox>
#include <QTimer>

#include <exception>
#include <memory>

using namespace RocketForgeTraining;
using namespace ForgeGUI;

namespace
{
    constexpr const char* BuildRequiredStatusText = "Create a build to start training";
}

// ---------------------------------------------------------------------------
// Queue training
// ---------------------------------------------------------------------------

void MainWindow::OnStartTrainingClicked()
{
    if (m_trainingActive) return;

    if (CurrentBuildName().isEmpty()) {
        QMessageBox::warning(
            this,
            "Cannot start training",
            "Create a build before starting training.");
        UpdateBuildPromptState();
        return;
    }

    if (m_modelBenchmarkActive || (m_modelBenchmarkWidget && m_modelBenchmarkWidget->IsRunning())) {
        QMessageBox::warning(
            this,
            "Cannot start training",
            "Stop the model benchmark before starting training.");
        return;
    }

    if (m_paramTestingActive || (m_paramTestingWidget && m_paramTestingWidget->IsRunning())) {
        QMessageBox::warning(
            this,
            "Cannot start training",
            "Stop Param Testing before starting training.");
        return;
    }

    if (m_debugRenderActive) {
        QMessageBox::warning(
            this,
            "Cannot start training",
            "Stop the debug render before starting training.");
        return;
    }

    ForgeSettings settings = BuildForgeSettingsFromUI();
    const std::vector<std::string> errors = ValidateForgeSettings(settings);

    if (!errors.empty()) {
        QString combined;
        for (const std::string& error : errors)
            combined += QString::fromStdString(error) + "\n";
        QMessageBox::warning(this, "Cannot start training", combined);
        return;
    }

    const QString staticOpponentWarning = StaticOpponentSelectionWarning(settings);

    if (!staticOpponentWarning.isEmpty()) {
        QMessageBox::warning(this, "Cannot start training", staticOpponentWarning);
        return;
    }

    // Protect continuation metadata until training is dispatched. The queue takes its own lock.
    std::unique_ptr<TrainingBuildLock> preparationLock;

    try {
        preparationLock = std::make_unique<TrainingBuildLock>(TrainingBuildRoot(settings));
    } catch (const std::exception& error) {
        QMessageBox::warning(this, "Cannot start training", QString::fromUtf8(error.what()));
        return;
    }

    if (!ResolveAgentSetupOverridesForContinuation(this, settings))
        return;

    try {
        for (RocketSim::EngineId engine : ResolveTrainingSimulation(settings))
            RocketSim::RequireEngine(engine);
    } catch (const RocketSim::EngineUnavailableError& error) {
        ShowSimulatorUnavailable(this, QString::fromUtf8(error.what()));
        return;
    } catch (const std::exception& error) {
        QMessageBox::warning(this, "Cannot start training", QString::fromUtf8(error.what()));
        return;
    }

    if (!ConfirmContinueSegmentMetadataOverwrite(this, settings))
        return;

    const QString settingsPath = QString::fromStdString(GetDefaultForgeSettingsPath(settings.agentName).string());
    std::string saveError;

    if (!GGL::JsonSaveToFile(settings, settingsPath.toStdString(), &saveError)) {
        QMessageBox::warning(this, "Cannot start training", QString::fromStdString(saveError));
        return;
    }

    preparationLock.reset();

    m_avgSkillTestSecs = 0.0;
    m_lastSkillTestSecs = 0.0;
    m_hasLastSkillTest = false;
    m_nextSkillTestTimestep = 0;
    m_skillTestIntervalTimesteps = 0;
    m_hasNextSkillTest = false;
    m_skillTestInProgress = false;
    ResetSkillRatings();
    StartTrainingSessionClock();
    m_skillRatings->SeedFromTrainingStart(settings);

    m_stopRequested = false;
    m_lastTrainingError.clear();

    m_activeRunMode = ActiveRunMode::Queue;
    SetTrainingActive(true);
    m_queueWidget->SetReadOnly(true);

    m_closeAfterTraining = false;
    m_worker->PrepareRun();
    const bool dispatched = QMetaObject::invokeMethod(m_worker, "RunTraining",
        Qt::QueuedConnection,
        Q_ARG(RocketForgeTraining::ForgeSettings, settings),
        Q_ARG(QString, settingsPath));

    if (!dispatched) {
        m_activeRunMode = ActiveRunMode::None;
        SetTrainingActive(false);
        m_queueWidget->SetReadOnly(false);
        AppendLog("Error: Failed to dispatch training task — worker thread not running.", "#ff4444");
    }
}

void MainWindow::OnSaveAndSkipClicked()
{
    const QMessageBox::StandardButton answer = QMessageBox::question(this, "Skip Segment",
        "Skipping will save the current checkpoint and advance to the next segment. "
        "The target timestep count for this segment will be updated to its current progress. Continue?",
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);

    if (answer != QMessageBox::Yes)
        return;

    // Call directly (not via QueuedConnection on m_worker). The worker thread is blocked inside
    // RunTraining, so a queued invocation would not run until training already finished. Skip is
    // an atomic-flag-flip + learner->RequestStop(), both thread-safe to call from the UI thread.
    m_worker->RequestSkipStage();
    ui->saveAndSkipButton->setEnabled(false);
    AppendLog("Skip segment requested - waiting for learner to finish current iteration...");
}

void MainWindow::OnSaveAndStopClicked()
{
    m_worker->RequestStop();
    m_stopRequested = true;
    ui->saveAndStopBtn->setEnabled(false);
    AppendLog("Stop and save requested — waiting for learner to finish current iteration...");
}

void MainWindow::OnTrainingFinished()
{
    // Debug renders run on m_debugWorker, not m_worker, so this only ever fires for queue training.
    AppendLog("--- Training complete ---");

    // Reload settings from disk so the GUI reflects any mutations made during training
    // (e.g. FlipBootstrapToContinueExisting, skip-rewritten targets, completion reset to New).
    LoadOrCreateForgeSettings();
    m_runOutcome = m_stopRequested ? "Stopped" : "Finished";
    m_activeRunMode = ActiveRunMode::None;
    SetTrainingActive(false);
    m_queueWidget->SetReadOnly(false);
    if (m_closeAfterTraining) QTimer::singleShot(0, this, &QWidget::close);
}

void MainWindow::OnTrainingError(const QString& message, bool simulatorUnavailable)
{
    m_closeAfterTraining = false;

    // Debug renders report through OnDebugRenderFailed, so this only fires for queue training.
    // Reload settings from disk before re-enabling editing — partial training may have
    // mutated bootstrap or createdSegmentId fields that the GUI doesn't know about yet.
    LoadOrCreateForgeSettings();
    m_runOutcome = "Error";
    m_lastTrainingError = message;
    m_activeRunMode = ActiveRunMode::None;
    SetTrainingActive(false);
    m_queueWidget->SetReadOnly(false);
    AppendLog("Error: " + message, "#ff4444");
    if (simulatorUnavailable)
        ShowSimulatorUnavailable(this, message);
    else
        QMessageBox::critical(this, "Training Error", message);
}

void MainWindow::OnStageStarted(int index, QString stageName, QString versionId)
{
    BeginActiveStage(index, CurrentBuildName(), stageName, versionId);
    if (m_lastTotalTimesteps > 0)
        m_queueWidget->UpdateLiveTotalTimesteps(index, m_lastTotalTimesteps);

    // Re-enable the Save & Skip button for the freshly started stage. It is disabled
    // by OnSaveAndSkipClicked while a skip request is in flight to prevent double-clicks.
    if (m_trainingActive)
        ui->saveAndSkipButton->setEnabled(true);
    AppendLog(QString("Segment %1 (%2) started [%3]").arg(index + 1).arg(stageName).arg(versionId));
    RefreshFileManagementRunContext(m_trainingActive, m_currentAgent);
}

void MainWindow::OnStageCompleted(int index)
{
    m_queueWidget->MarkStageCompleted(index, index == m_activeStageIndex ? m_activeVersionId : QString());
    if (index == m_activeStageIndex)
        EndActiveStage();

    AppendLog(QString("Segment %1 completed").arg(index + 1));
    RefreshFileManagementRunContext(m_trainingActive, m_currentAgent);
}

void MainWindow::OnStageSkipped(int index, QString reason)
{
    m_queueWidget->UpdateLiveEta(index, "");
    if (index == m_activeStageIndex)
        EndActiveStage();

    AppendLog(QString("Segment %1 skipped: %2").arg(index + 1).arg(reason));
    RefreshFileManagementRunContext(m_trainingActive, m_currentAgent);
}

void MainWindow::SetTrainingActive(bool active)
{
    m_trainingActive        = active;
    m_autoSaveSuspended     = active;

    if (!active) {
        StopTrainingSessionClock();
        if (m_queueWidget)
            m_queueWidget->SetActiveStage(-1);
        UpdateProgressBlock();
    } else if (m_activeBuildName.isEmpty()) {
        m_activeBuildName = CurrentBuildName();
    }

    ui->agentCombo->setEnabled(!active);
    ui->buildCombo->setEnabled(!active);
    ui->newBuildBtn->setEnabled(!active);
    ui->newBotBtn->setEnabled(!active);
    ui->staticOpponentsBtn->setEnabled(true);
    ui->skillReferencesBtn->setEnabled(true);
    ui->metricsCheck->setEnabled(!active);
    ui->pruneSourceCheck->setEnabled(!active);
    ui->persistSnapshotsCheck->setEnabled(!active);
    ui->performanceCoresCheck->setEnabled(!active);
    ui->saveAndStopBtn->setEnabled(active);
    ui->saveAndSkipButton->setEnabled(active);
    if (m_debugTab)
        m_debugTab->SetExternalRunActive(active);
    if (m_modelBenchmarkWidget)
        m_modelBenchmarkWidget->SetTrainingActive(active);
    if (m_paramTestingWidget)
        m_paramTestingWidget->SetExternalRunActive(active);
    if (m_fileManagementWidget)
        m_fileManagementWidget->SetTrainingContext(
            active, m_currentAgent, m_activeBuildName, m_activeStageName, m_activeVersionId);
    UpdateBuildPromptState();

    UpdateStatusBarState();
    if (m_monitorWindow && m_monitorWindow->isVisible())
        m_monitorWindow->Refresh();
}

void MainWindow::RefreshFileManagementRunContext(bool runActive, const QString& agentName)
{
    if (!m_fileManagementWidget)
        return;

    m_fileManagementWidget->SetTrainingContext(runActive, agentName, m_activeBuildName, m_activeStageName, m_activeVersionId);
    m_fileManagementWidget->Refresh();
}

// ---------------------------------------------------------------------------
// Debug render
// ---------------------------------------------------------------------------

void MainWindow::OnStartDebugRequested(RocketForgeTraining::ForgeDebugSettings settings, QString agentName,
                                       QString stageName, QString checkpointFolder, bool isImported)
{
    if (m_trainingActive || m_debugRenderActive ||
        (m_modelBenchmarkWidget && m_modelBenchmarkWidget->IsRunning()) ||
        (m_paramTestingWidget && m_paramTestingWidget->IsRunning())) {
        QMessageBox::warning(this, "Cannot start debug render",
            "Another run is already active. Stop it first.");
        return;
    }

    if (stageName.isEmpty() || checkpointFolder.isEmpty()) {
        QMessageBox::warning(this, "Cannot start debug render",
            "Select a segment (or imported model) and a stage to render.");
        return;
    }

    m_activeRunMode     = ActiveRunMode::Debug;
    m_debugRenderActive = true;
    m_debugRenderAgent  = agentName;
    m_debugTab->SetRunActive(true);

    // Lock out the model benchmark / param testing for the duration — both drive RocketSim/GPU
    // and must not run concurrently. (Queue training does this via SetTrainingActive; debug
    // renders bypass that path, so disable them explicitly here.)
    if (m_modelBenchmarkWidget)
        m_modelBenchmarkWidget->SetTrainingActive(true);
    if (m_paramTestingWidget)
        m_paramTestingWidget->SetExternalRunActive(true);
    UpdateBuildPromptState();
    UpdateStatusBarState();

    // Seed the bridge + reward viz from the persisted debug settings before the worker reads them.
    if (m_bridge)
        m_bridge->InitializeFromSettings(settings.debug);
    if (m_vizWindow)
        m_vizWindow->InitFromSettings(settings.debug);

    if (settings.showRewards && m_vizWindow) {
        m_vizWindow->show();
        m_vizWindow->raise();
    } else if (m_vizWindow) {
        m_vizWindow->hide();
    }

    const bool dispatched = QMetaObject::invokeMethod(m_debugWorker, "Run",
        Qt::QueuedConnection,
        Q_ARG(QString, agentName),
        Q_ARG(QString, stageName),
        Q_ARG(QString, checkpointFolder),
        Q_ARG(bool,    isImported),
        Q_ARG(bool,    settings.showRewards),
        Q_ARG(int,     settings.debug.playerIndex),
        Q_ARG(QString, CurrentBuildName()),
        Q_ARG(bool,    settings.debug.includeStaticOpponent),
        Q_ARG(QString, QString::fromStdString(settings.debug.staticOpponentSelectionKey)));

    if (!dispatched) {
        m_activeRunMode     = ActiveRunMode::None;
        m_debugRenderActive = false;
        m_debugTab->SetRunActive(false);
        if (m_modelBenchmarkWidget)
            m_modelBenchmarkWidget->SetTrainingActive(false);
        if (m_paramTestingWidget)
            m_paramTestingWidget->SetExternalRunActive(m_remoteSyncActive);
        UpdateBuildPromptState();
        UpdateStatusBarState();
        AppendLog("Error: Failed to dispatch debug render — worker thread not running.", "#ff4444");
    }
}

void MainWindow::OnDebugRenderFinished()
{
    m_debugRenderActive = false;
    m_activeRunMode = ActiveRunMode::None;

    if (m_vizWindow)
        m_vizWindow->hide();
    if (m_debugTab)
        m_debugTab->SetRunActive(false);
    if (m_modelBenchmarkWidget)
        m_modelBenchmarkWidget->SetTrainingActive(false);
    if (m_paramTestingWidget)
        m_paramTestingWidget->SetExternalRunActive(m_remoteSyncActive);

    UpdateBuildPromptState();
    UpdateStatusBarState();
}

void MainWindow::OnDebugRenderFailed(const QString& error, bool simulatorUnavailable)
{
    OnDebugRenderFinished();   // identical teardown, plus a message

    AppendLog("Debug render error: " + error, "#ff4444");
    if (simulatorUnavailable)
        ShowSimulatorUnavailable(this, error);
    else
        QMessageBox::warning(this, "Debug Render Error", error);
}

// ---------------------------------------------------------------------------
// Runs in other tabs
// ---------------------------------------------------------------------------

void MainWindow::OnModelBenchmarkRunningChanged(bool running)
{
    m_modelBenchmarkActive = running;
    UpdateBuildPromptState();
    if (m_debugTab)
        m_debugTab->SetExternalRunActive(running || m_trainingActive || m_paramTestingActive);
    if (m_paramTestingWidget)
        m_paramTestingWidget->SetExternalRunActive(running || m_trainingActive || m_debugRenderActive || m_remoteSyncActive);

    if (!m_trainingActive)
        UpdateStatusBarState();
}

void MainWindow::OnParamTestingRunningChanged(bool running)
{
    m_paramTestingActive = running;
    UpdateBuildPromptState();
    if (m_debugTab)
        m_debugTab->SetExternalRunActive(running || m_trainingActive || m_modelBenchmarkActive);
    if (m_modelBenchmarkWidget)
        m_modelBenchmarkWidget->SetTrainingActive(running || m_trainingActive || m_debugRenderActive);

    if (!m_trainingActive)
        UpdateStatusBarState();
}

void MainWindow::OnRemoteSyncRunningChanged(bool running, RemoteSyncKind kind)
{
    m_remoteSyncActive = running;
    const bool pullRunning = running && kind == RemoteSyncKind::Pull;

    // The agent, build and queue name what the sync transfers, and a pull reloads them afterwards.
    m_autoSaveSuspended = running;
    m_queueWidget->SetReadOnly(running);
    ui->agentCombo->setEnabled(!running);
    ui->buildCombo->setEnabled(!running);
    ui->newBuildBtn->setEnabled(!running);
    ui->newBotBtn->setEnabled(!running);
    ui->staticOpponentsBtn->setEnabled(!running);
    ui->skillReferencesBtn->setEnabled(!running);
    ui->metricsCheck->setEnabled(!running);
    ui->pruneSourceCheck->setEnabled(!running);
    ui->persistSnapshotsCheck->setEnabled(!running);
    ui->performanceCoresCheck->setEnabled(!running);

    if (m_debugTab)
        m_debugTab->SetExternalRunActive(pullRunning || m_trainingActive || m_modelBenchmarkActive || m_paramTestingActive);
    if (m_modelBenchmarkWidget)
        m_modelBenchmarkWidget->SetTrainingActive(pullRunning || m_trainingActive || m_debugRenderActive || m_paramTestingActive);

    // Param testing swaps a preview into the queue and re-enables the agent/build controls when it ends.
    if (m_paramTestingWidget)
        m_paramTestingWidget->SetExternalRunActive(running || m_trainingActive || m_debugRenderActive || m_modelBenchmarkActive);

    UpdateBuildPromptState();
}

void MainWindow::OnRemoteSyncFinished(RemoteSyncKind kind, RemoteSyncOutcome outcome, int exitCode)
{
    switch (outcome) {
    case RemoteSyncOutcome::Succeeded:
        AppendLog("[sync] Done.", Theme::POSITIVE.name());
        break;
    case RemoteSyncOutcome::Failed:
        AppendLog(QString("[sync] Failed (exit %1).").arg(exitCode), Theme::NEGATIVE.name());
        break;
    case RemoteSyncOutcome::Cancelled:
        AppendLog("[sync] Cancelled. Re-running is safe: rclone --update never deletes.", Theme::TEXT_MUTED.name());
        break;
    }

    // A pull's first pass rewrites forge_settings.json, so even one that stopped partway may have changed it;
    // without a reload, the next auto-save would write the stale in-memory queue back over the pulled segment ids.
    if (kind == RemoteSyncKind::Pull) {
        LoadOrCreateForgeSettings();
        AppendLog("[sync] Reloaded settings from disk.");
    }
}

// ---------------------------------------------------------------------------
// Availability and status
// ---------------------------------------------------------------------------

void MainWindow::UpdateBuildPromptState()
{
    if (!ui || !ui->newBuildBtn || !ui->continueTrainingBtn)
        return;

    const bool needsBuild = CurrentBuildName().isEmpty();
    ui->newBuildBtn->setStyleSheet(QString());

    const bool anyRunActive =
        m_trainingActive || m_modelBenchmarkActive || m_paramTestingActive || m_debugRenderActive || m_remoteSyncActive;
    if (ui->renameBuildBtn)
        ui->renameBuildBtn->setEnabled(!needsBuild && !anyRunActive);
    ui->continueTrainingBtn->setEnabled(!needsBuild && !anyRunActive);

    UpdateRemoteSyncPanelState();
    UpdateStatusBarState();
}

void MainWindow::UpdateRemoteSyncPanelState()
{
    if (!m_remoteSyncPanel)
        return;

    const bool canPush = !m_trainingActive && !m_paramTestingActive;

    // A pull may overwrite checkpoints a debug render or benchmark has loaded.
    const bool canPull = canPush && !m_debugRenderActive && !m_modelBenchmarkActive;

    m_remoteSyncPanel->SetTarget(m_currentAgent, CurrentBuildName());
    m_remoteSyncPanel->SetAvailability(canPush, canPull);
}

void MainWindow::UpdateStatusBarState()
{
    if (!ui || !m_statusStateLabel)
        return;

    if (m_trainingActive) {
        m_statusStateLabel->setText("Training: " + m_currentAgent);
        return;
    }

    if (m_debugRenderActive) {
        m_statusStateLabel->setText("Debug render: " + m_debugRenderAgent);
        return;
    }

    if (m_modelBenchmarkActive) {
        m_statusStateLabel->setText("Model benchmarking");
        return;
    }

    if (m_paramTestingActive) {
        m_statusStateLabel->setText("Param testing");
        return;
    }

    if (CurrentBuildName().isEmpty()) {
        m_statusStateLabel->setText(BuildRequiredStatusText);
        return;
    }

    m_statusStateLabel->setText("Ready");
}
