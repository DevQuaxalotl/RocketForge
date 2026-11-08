#include "MainWindow/mainwindow.h"
#include "ui_mainwindow.h"

#include "Debug/DebugTabWidget.h"
#include "ModelBenchmark/ModelBenchmarkWidget.h"
#include "TrainingSetup/TrainingQueueWidget.h"
#include "VersionManagementDialog.h"

#include <QStackedWidget>
#include <QTabBar>

#include <exception>

using namespace RocketForgeTraining;

void MainWindow::OnParamTestingTrainingPreviewRequested(ParamTestingConfig config)
{
    if (!m_queueWidget)
        return;

    std::vector<StageSettings> stages;

    try {
        stages = BuildParamTestingPreviewStages(config);
    } catch (const std::exception& error) {
        AppendLog("Param Testing preview failed: " + QString::fromStdString(error.what()), "#ff4444");
        return;
    }

    if (!m_paramTestingPreviewActive) {
        m_savedParamTestingQueueAgent = m_currentAgent;
        m_savedParamTestingQueueBuild = CurrentBuildName();
        m_savedParamTestingQueueStages = m_queueWidget->Stages();
    }

    m_paramTestingPreviewActive = true;
    m_stopRequested = false;
    m_lastTrainingError.clear();
    m_paramTestingPreviewAgent = QString::fromStdString(config.agentName);
    m_paramTestingPreviewBuild = QString::fromStdString(config.buildName);

    if (m_modeSelector)
        m_modeSelector->setCurrentIndex(0);

    m_queueWidget->SetValidationStylesSuppressed(true);
    m_queueWidget->SetAgentName(m_paramTestingPreviewAgent);
    m_queueWidget->SetBuildName(m_paramTestingPreviewBuild);
    m_queueWidget->SetStages(stages);
    m_queueWidget->SetReadOnly(true);
    m_queueWidget->SetActiveStage(-1);

    ResetSkillRatings();
    m_avgSkillTestSecs = 0.0;
    StartTrainingSessionClock();

    SetTrainingTabLockedForParamTesting(true);
    UpdateProgressBlock();
    AppendLog(QString("Param Testing loaded %1 segment%2 into the Training tab.")
        .arg(static_cast<int>(stages.size()))
        .arg(stages.size() == 1 ? "" : "s"));
}

void MainWindow::OnParamTestingPreviewStageStarted(int index, QString stageName, QString versionId)
{
    if (!m_paramTestingPreviewActive || !m_queueWidget)
        return;

    BeginActiveStage(index, m_paramTestingPreviewBuild, stageName, versionId);
    AppendLog(QString("Param Testing segment %1 (%2) started [%3]").arg(index + 1).arg(stageName).arg(versionId));
    const bool previewRunning = true;
    RefreshFileManagementRunContext(previewRunning, m_paramTestingPreviewAgent);
}

void MainWindow::OnParamTestingPreviewStageCompleted(int index, QString versionId)
{
    if (!m_paramTestingPreviewActive || !m_queueWidget)
        return;

    m_queueWidget->MarkStageCompleted(index, versionId);
    if (index == m_activeStageIndex)
        EndActiveStage();

    AppendLog(QString("Param Testing segment %1 completed").arg(index + 1));
    const bool previewRunning = true;
    RefreshFileManagementRunContext(previewRunning, m_paramTestingPreviewAgent);
}

void MainWindow::OnParamTestingPreviewFinished()
{
    if (!m_paramTestingPreviewActive || !m_queueWidget)
        return;

    StopTrainingSessionClock();

    m_queueWidget->SetValidationStylesSuppressed(false);
    m_queueWidget->SetAgentName(m_savedParamTestingQueueAgent);
    m_queueWidget->SetBuildName(m_savedParamTestingQueueBuild);
    m_queueWidget->SetStages(m_savedParamTestingQueueStages);
    m_queueWidget->SetActiveStage(-1);

    m_paramTestingPreviewActive = false;
    m_runOutcome = m_stopRequested ? "Stopped" : "Finished";
    m_paramTestingPreviewAgent.clear();
    m_paramTestingPreviewBuild.clear();
    m_savedParamTestingQueueAgent.clear();
    m_savedParamTestingQueueBuild.clear();
    m_savedParamTestingQueueStages.clear();

    SetTrainingTabLockedForParamTesting(false);
    UpdateProgressBlock();
    AppendLog("Param Testing restored the previous Training tab queue.");
    const bool previewRunning = false;
    RefreshFileManagementRunContext(previewRunning, m_currentAgent);
}

void MainWindow::OnParamTestingBenchmarkRunRequested()
{
    if (!m_modelBenchmarkWidget)
        return;

    if (m_modeSelector && m_modeStack)
        m_modeSelector->setCurrentIndex(m_modeStack->indexOf(m_modelBenchmarkWidget));
    m_modelBenchmarkWidget->StartPersistedBenchmark();
}

void MainWindow::SetTrainingTabLockedForParamTesting(bool locked)
{
    if (!ui)
        return;

    m_autoSaveSuspended = locked;
    if (m_queueWidget)
        m_queueWidget->SetReadOnly(locked);

    ui->agentCombo->setEnabled(!locked);
    ui->buildCombo->setEnabled(!locked);
    ui->newBuildBtn->setEnabled(!locked);
    ui->newBotBtn->setEnabled(!locked);
    if (ui->renameBuildBtn)
        ui->renameBuildBtn->setEnabled(!locked && !CurrentBuildName().isEmpty());
    ui->continueTrainingBtn->setEnabled(!locked && !CurrentBuildName().isEmpty());
    ui->staticOpponentsBtn->setEnabled(!locked);
    ui->skillReferencesBtn->setEnabled(!locked);
    ui->rewardTableBtn->setEnabled(!locked);
    ui->metricsCheck->setEnabled(!locked);
    ui->pruneSourceCheck->setEnabled(!locked);
    ui->persistSnapshotsCheck->setEnabled(!locked);
    ui->performanceCoresCheck->setEnabled(!locked);
    ui->saveAndStopBtn->setEnabled(false);
    ui->saveAndSkipButton->setEnabled(false);

    if (m_debugTab)
        m_debugTab->SetExternalRunActive(locked || m_trainingActive || m_modelBenchmarkActive);
    if (!locked)
        UpdateBuildPromptState();
    UpdateStatusBarState();
}
