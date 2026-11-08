#include "MainWindow/mainwindow.h"
#include "ui_mainwindow.h"

#include "Debug/DebugTabWidget.h"
#include "MainWindow/TrainingStartPreflight.h"
#include "ModelBenchmark/ModelBenchmarkWidget.h"
#include "ParamTesting/ParamTestingWidget.h"
#include "RewardVisualizer/ImpactAnalysisWidget.h"
#include "StaticOpponentSelectionDialog.h"
#include "TrainingSetup/TrainingQueueWidget.h"
#include "TrainingSetup/TrainingSetupWindow.h"
#include "VersionManagementDialog.h"
#include "Widgets/AgentComboBox.h"
#include "Widgets/BuildComboBox.h"

#include <AgentModelPaths.h>
#include <Utils/BotScaffold.h>
#include <ForgeTraining.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <StageRegistry.h>
#include <TrainingStage.h>
#include <StaticOpponents/SkillReferenceSelection.h>
#include <StaticOpponents/StaticOpponentSelection.h>
#include <Storage/BuildStorage.h>
#include <Storage/TrainingBuildLock.h>
#include <TrainingSetup/TrainingSetupSnapshot.h>
#include <TrainingSetup/VersionManagement.h>

#include <QCoreApplication>
#include <QInputDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QStatusBar>
#include <QTimer>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <memory>

using namespace RocketForgeTraining;
using namespace ForgeGUI;

namespace
{
    constexpr int StatusMessageMilliseconds = 2000;

    QString ProposedNextBuildName(const std::string& agentName)
    {
        int next = 1;
        static const QRegularExpression rxVersion(R"(^V(\d+)$)", QRegularExpression::CaseInsensitiveOption);

        for (const BuildSummary& build : ListAgentBuilds(agentName)) {
            const QRegularExpressionMatch match = rxVersion.match(QString::fromStdString(build.buildName));

            if (!match.hasMatch())
                continue;

            bool ok = false;
            const int number = match.captured(1).toInt(&ok);
            if (ok)
                next = std::max(next, number + 1);
        }

        return QString("V%1").arg(next);
    }
}

// ---------------------------------------------------------------------------
// Loading and saving
// ---------------------------------------------------------------------------

void MainWindow::LoadOrCreateForgeSettings()
{
    const std::string agent = m_currentAgent.toStdString();

    if (agent.empty())
        return;

    const std::filesystem::path path = GetDefaultForgeSettingsPath(agent);
    ForgeSettingsLoadResult result = LoadForgeSettings(path);

    ForgeSettings settings;

    if (result.ok) {
        settings = result.value;
    } else {
        const std::filesystem::path examplePath = GetExampleForgeSettingsPath(agent);

        if (std::filesystem::exists(examplePath)) {
            std::filesystem::copy_file(examplePath, path, std::filesystem::copy_options::skip_existing);
            result   = LoadForgeSettings(path);
            settings = result.ok ? result.value : MakeDefaultForgeSettings(agent, "");
        } else {
            settings = MakeDefaultForgeSettings(agent, "");
        }
    }

    // Populating the UI fires its change signals, which must not schedule a save of what was just loaded.
    m_autoSaveSuspended = true;

    RefreshBuildCombo(QString::fromStdString(settings.buildName));
    m_queueWidget->SetAgentName(QString::fromStdString(settings.agentName));
    m_queueWidget->SetBuildName(CurrentBuildName());
    m_queueWidget->SetStages(settings.stages);

    ui->metricsCheck->setChecked(settings.globals.metricsEnabled);
    ui->pruneSourceCheck->setChecked(settings.globals.pruneSourceCheckpoints);
    ui->persistSnapshotsCheck->setChecked(settings.globals.persistSegmentSnapshotsDuringTraining);
    ui->performanceCoresCheck->setChecked(settings.globals.performanceCoresOnly);

    m_autoSaveSuspended = false;
    SyncFileManagementContext();
}

ForgeSettings MainWindow::BuildForgeSettingsFromUI() const
{
    ForgeSettings settings;
    settings.schemaVersion                  = kForgeSettingsSchemaVersion;
    settings.agentName                      = m_currentAgent.toStdString();
    settings.buildName                      = CurrentBuildName().toStdString();
    settings.globals.metricsEnabled         = ui->metricsCheck->isChecked();
    settings.globals.render                 = false;   // Queue training never renders; use the Debug tab.
    settings.globals.pruneSourceCheckpoints = ui->pruneSourceCheck->isChecked();
    settings.globals.persistSegmentSnapshotsDuringTraining = ui->persistSnapshotsCheck->isChecked();
    settings.globals.performanceCoresOnly   = ui->performanceCoresCheck->isChecked();
    settings.stages                         = m_queueWidget->Stages();

    return settings;
}

void MainWindow::OnQueueChanged()
{
    m_saveDebounceTimer->start();
}

void MainWindow::SaveSettingsNow()
{
    if (m_autoSaveSuspended || m_trainingActive)
        return;

    ForgeSettings settings = BuildForgeSettingsFromUI();

    if (settings.agentName.empty())
        return;

    std::unique_ptr<TrainingBuildLock> settingsLock;

    if (!settings.buildName.empty()) {
        try {
            settingsLock = std::make_unique<TrainingBuildLock>(TrainingBuildRoot(settings));
        } catch (const std::exception& error) {
            AppendLog("Settings were not saved: " + QString::fromUtf8(error.what()), "#cc6666");
            return;
        }
    }

    const std::filesystem::path path = GetDefaultForgeSettingsPath(settings.agentName);
    std::string error;
    if (!GGL::JsonSaveToFile(settings, path, &error))
        AppendLog(QString("Auto-save failed: %1").arg(QString::fromStdString(error)), "#cc6666");
    else
        statusBar()->showMessage("Settings saved.", StatusMessageMilliseconds);
}

void MainWindow::FlushPendingSettingsSave()
{
    if (!m_saveDebounceTimer->isActive())
        return;

    m_saveDebounceTimer->stop();
    SaveSettingsNow();
}

void MainWindow::OnPruneToggleClicked(bool checked)
{
    if (!checked) {
        OnQueueChanged();
        return;
    }

    const QMessageBox::StandardButton answer = QMessageBox::question(this, "Prune source checkpoints",
        "Pruning source checkpoints will permanently delete old checkpoints from the previous segment "
        "after each transition. This cannot be undone. Enable?");

    if (answer != QMessageBox::Yes) {
        ui->pruneSourceCheck->setChecked(false);
        return;
    }

    OnQueueChanged();
}

// ---------------------------------------------------------------------------
// Agent and build selection
// ---------------------------------------------------------------------------

QString MainWindow::CurrentBuildName() const
{
    if (!ui || !ui->buildCombo)
        return {};

    return ui->buildCombo->CurrentBuildName();
}

void MainWindow::RefreshBuildCombo(const QString& preferredBuild)
{
    if (!ui || !ui->buildCombo)
        return;

    ui->buildCombo->SetAgent(m_currentAgent);
    ui->buildCombo->Refresh(preferredBuild);

    UpdateBuildPromptState();
}

void MainWindow::OnAgentChanged()
{
    m_currentAgent = ui->agentCombo->currentText();
    SaveLastSelectedAgent(m_currentAgent.toStdString());
    LoadOrCreateForgeSettings();

    m_modelBenchmarkWidget->FollowTrainingAgent(m_currentAgent);
    m_paramTestingWidget->FollowTrainingAgent(m_currentAgent);
    m_impactAnalysisWidget->FollowTrainingAgent(m_currentAgent);
    m_debugTab->FollowTrainingAgent(m_currentAgent);
}

void MainWindow::OnBuildChanged()
{
    m_queueWidget->SetBuildName(CurrentBuildName());
    m_queueWidget->ClearCreatedIds();
    UpdateBuildPromptState();
    SyncFileManagementContext();
    OnQueueChanged();
}

void MainWindow::OnNewBuildClicked()
{
    const std::string agentName = m_currentAgent.toStdString();
    bool ok = false;
    const QString buildName = QInputDialog::getText(
        this,
        "New Build",
        "Build name:",
        QLineEdit::Normal,
        ProposedNextBuildName(agentName),
        &ok).trimmed();

    if (!ok || buildName.isEmpty())
        return;

    BuildStorage storage(ResolveAgentModelsFolder(agentName), buildName.toStdString());
    storage.EnsureBuildExists();

    RefreshBuildCombo(buildName);
    m_queueWidget->SetBuildName(buildName);
    m_queueWidget->ClearCreatedIds();
    UpdateBuildPromptState();
    SyncFileManagementContext();
    OnQueueChanged();
}

void MainWindow::OnNewBotClicked()
{
    bool ok = false;
    const QString botName = QInputDialog::getText(
        this,
        "New Bot",
        "Bot name:",
        QLineEdit::Normal,
        QString(),
        &ok).trimmed();

    if (!ok || botName.isEmpty())
        return;

    const std::string name = botName.toStdString();
    const RocketForge::CreateBotResult result = RocketForge::CreateBotFromExample(
        name,
        RocketForge::DefaultExampleBotRoot(),
        RocketForge::DefaultBotsRoot(),
        AgentRegistry::GetAvailableAgents());

    if (!result.Ok()) {
        QMessageBox::warning(this, "Cannot create bot", QString::fromStdString(result.error));
        return;
    }

    QMessageBox createdBox(this);
    createdBox.setWindowTitle("Bot created");
    createdBox.setText(
        QString("Created %1.\n\nRocketForge has to be re-configured and rebuilt before %2 appears in the agent list.")
            .arg(QString::fromStdString(result.botFolder.string()),
                 QString::fromStdString(RocketForge::MakeBotNameForms(name).AgentName())));
    QPushButton* rebuildButton = createdBox.addButton("Rebuild and restart", QMessageBox::AcceptRole);
    createdBox.addButton("Later", QMessageBox::RejectRole);
    createdBox.exec();

    if (createdBox.clickedButton() != rebuildButton)
        return;

    // closeEvent can refuse the close; the helper waits for this process to exit, so it must not start while the GUI stays open.
    if (!close()) {
        QMessageBox::information(this, "Rebuild not started", "Rebuild not started; rebuild after RocketForge has closed.");
        return;
    }

    const std::filesystem::path repoRoot = std::filesystem::path(TRAINING_ROOT).parent_path();
    const std::filesystem::path script = repoRoot / "tools" / "build" / "rebuild_and_relaunch.ps1";

    // The script only hands the rebuild off to a process outside this one's job and exits, so wait for it: the GUI must not
    // exit (and take the job down) before the hand-off is done.
    QProcess helper;
    helper.setProgram("powershell.exe");
    helper.setArguments({
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", QString::fromStdString(script.string()),
        "-WaitPid", QString::number(QCoreApplication::applicationPid()),
        "-Relaunch", QCoreApplication::applicationFilePath()});
    helper.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* arguments) {
        arguments->flags |= CREATE_NO_WINDOW;
    });
    helper.start();

    constexpr int HANDOFF_TIMEOUT_MS = 30000;
    if (!helper.waitForFinished(HANDOFF_TIMEOUT_MS) || helper.exitStatus() != QProcess::NormalExit || helper.exitCode() != 0) {
        const QString details = QString::fromLocal8Bit(helper.readAllStandardError()).trimmed();
        QMessageBox::warning(nullptr, "Rebuild not started",
            "Could not start the rebuild. Rebuild RocketForge manually.\n\n" + details);
    }
}

void MainWindow::OnRenameBuildClicked()
{
    const QString oldBuildName = CurrentBuildName();

    if (oldBuildName.isEmpty()) {
        UpdateBuildPromptState();
        return;
    }

    if (m_trainingActive || m_modelBenchmarkActive || m_paramTestingActive || m_debugRenderActive) {
        QMessageBox::warning(
            this,
            "Cannot rename build",
            "Stop training, Param Testing, or model benchmarking before renaming a build.");
        UpdateBuildPromptState();
        return;
    }

    const std::string agentName = m_currentAgent.toStdString();
    bool ok = false;
    const QString newBuildName = QInputDialog::getText(
        this,
        "Rename Build",
        "Build name:",
        QLineEdit::Normal,
        oldBuildName,
        &ok).trimmed();

    if (!ok || newBuildName.isEmpty() || newBuildName == oldBuildName)
        return;

    BuildStorage storage(ResolveAgentModelsFolder(agentName), oldBuildName.toStdString());
    std::string error;

    if (!storage.RenameBuild(newBuildName.toStdString(), &error)) {
        QMessageBox::warning(
            this,
            "Cannot rename build",
            QString::fromStdString(error.empty() ? "Failed to rename build." : error));
        RefreshBuildCombo(oldBuildName);
        return;
    }

    RefreshBuildCombo(newBuildName);
    m_queueWidget->SetBuildName(newBuildName);
    m_queueWidget->ClearCreatedIds();
    UpdateBuildPromptState();
    SyncFileManagementContext();
    OnQueueChanged();
    statusBar()->showMessage("Build renamed.", StatusMessageMilliseconds);
}

void MainWindow::SyncFileManagementContext()
{
    UpdateStaticOpponentsButtonText();

    if (!m_fileManagementWidget)
        return;

    QString initialStage;

    if (m_queueWidget) {
        const std::vector<StageSettings> stages = m_queueWidget->Stages();
        if (!stages.empty())
            initialStage = QString::fromStdString(stages.front().stageName);
    }

    m_fileManagementWidget->SetTrainingContext(m_trainingActive, m_currentAgent, m_activeBuildName, m_activeStageName, m_activeVersionId);
    m_fileManagementWidget->SetContext(m_currentAgent, CurrentBuildName(), initialStage);
}

// ---------------------------------------------------------------------------
// Queue tools
// ---------------------------------------------------------------------------

void MainWindow::UpdateStaticOpponentsButtonText()
{
    if (!ui || !ui->staticOpponentsBtn)
        return;

    int selectedCount = 0;
    int referenceCount = 0;
    const QString agentName = ui->agentCombo ? ui->agentCombo->currentText().trimmed() : QString();
    const QString buildName = CurrentBuildName();

    if (!agentName.isEmpty() && !buildName.isEmpty()) {
        try {
            selectedCount = static_cast<int>(LoadStaticOpponentSelection(agentName.toStdString(), buildName.toStdString()).size());
        } catch (...) {
            selectedCount = 0;
        }

        try {
            referenceCount = static_cast<int>(LoadSkillReferenceSelection(agentName.toStdString(), buildName.toStdString()).size());
        } catch (...) {
            referenceCount = 0;
        }
    }

    ui->staticOpponentsBtn->setText(selectedCount > 0
        ? QString("Static Opponents (%1)...").arg(selectedCount)
        : QString("Static Opponents..."));

    ui->skillReferencesBtn->setText(referenceCount > 0
        ? QString("Skill References (%1)...").arg(referenceCount)
        : QString("Skill References..."));
}

void MainWindow::OnStaticOpponentsClicked()
{
    StaticOpponentSelectionDialog dialog(ui->agentCombo->currentText(), CurrentBuildName(), m_trainingActive, this);
    dialog.exec();
    UpdateStaticOpponentsButtonText();
}

void MainWindow::OnSkillReferencesClicked()
{
    StaticOpponentSelectionDialog dialog(
        ui->agentCombo->currentText(), CurrentBuildName(), m_trainingActive, this, SelectionPurpose::SkillReferences);
    dialog.exec();
    UpdateStaticOpponentsButtonText();
}

void MainWindow::OnRewardTableClicked()
{
    const std::string agentName = ui->agentCombo->currentText().toStdString();
    std::string stageName;
    const std::vector<StageSettings> stages = m_queueWidget->Stages();

    if (!stages.empty()) {
        stageName = stages.front().stageName;
    } else {
        const std::vector<std::string> availableStages = RocketForge::StageRegistry::GetAvailableStages();
        if (!availableStages.empty())
            stageName = availableStages.front();
    }

    if (stageName.empty()) {
        AppendLog("Error: No training stages are registered - cannot open reward table.", "#ff4444");
        return;
    }

    auto stage = CreateTrainingStage(agentName, stageName, {}, {});

    if (!stage) {
        AppendLog("Error: Failed to create training stage for reward table.", "#ff4444");
        return;
    }

    const RocketForge::TrainingSetupSnapshot snapshot = CaptureTrainingSetupSnapshot(stage.get());
    const std::filesystem::path checkpointPath = stage->GetMostRecentCheckpoint();
    const QString checkpointPathText = QString::fromStdString(checkpointPath.string());
    const QString versionPathText = QString::fromStdString(checkpointPath.parent_path().string());
    const QString versionId;

    delete m_trainingSetupWindow;
    m_trainingSetupWindow = new TrainingSetupWindow(snapshot, versionId, checkpointPathText, versionPathText, CurrentBuildName());
    m_trainingSetupWindow->setWindowTitle(QString("Training Setup - %1 / %2").arg(ui->agentCombo->currentText(), CurrentBuildName()));
    m_trainingSetupWindow->show();
    m_trainingSetupWindow->raise();
}
