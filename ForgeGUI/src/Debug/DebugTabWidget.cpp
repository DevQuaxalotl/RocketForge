#include "DebugTabWidget.h"

#include "Widgets/AgentComboBox.h"
#include "Widgets/BuildComboBox.h"
#include "Widgets/SegmentComboBox.h"
#include "DebugOptionsWidget.h"
#include "ThemeUtils.h"

#include <ForgeTraining.h>
#include <TrainingStage.h>
#include <StageRegistry.h>
#include <DebugRender/DebugRenderTarget.h>
#include <Settings/ForgeSettings.h>
#include <StaticOpponents/StaticOpponentSelection.h>
#include <Storage/BuildStorage.h>
#include <TrainingSetup/VersionManagement.h>
#include <GigaLearnCPP/SegmentVersioning.h>

#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <optional>
#include <vector>

using namespace RocketForgeTraining;

namespace
{
    constexpr int SIDEBAR_WIDTH = 520;
    constexpr int HINT_MAX_WIDTH = 720;
}

DebugTabWidget::DebugTabWidget(DebugBridge* bridge, QWidget* parent) : QWidget(parent)
{
    auto* root = new QHBoxLayout(this);

    auto* modelGroup = new QGroupBox("Model", this);
    auto* form = new QFormLayout(modelGroup);
    m_agentCombo   = new AgentComboBox(this);
    m_buildCombo   = new BuildComboBox(this);
    m_segmentCombo = new SegmentComboBox(this);
    m_stageCombo   = new QComboBox(this);

    // Imported bots have no build; let the segment combo list them when no build is selected.
    m_segmentCombo->SetIncludeImportedModels(true);

    m_staticOpponentCheck = new QCheckBox("Include static opponent", this);
    m_staticOpponentCheck->setToolTip(
        "Use one of the static models selected for this Build as the opposing policy.");
    m_staticOpponentCombo = new QComboBox(this);
    m_staticOpponentCombo->setToolTip(
        "Choose one model, or reselect a random compatible model at every episode reset.");

    form->addRow("Agent:",   m_agentCombo);
    form->addRow("Build:",   m_buildCombo);
    form->addRow("Segment:", m_segmentCombo);

    m_stageTrainedHint = new QLabel(this);
    m_stageTrainedHint->setProperty(ForgeGUI::TextRoleProperty, "muted");
    form->addRow("Stage:", m_stageCombo);
    form->addRow(QString(), m_stageTrainedHint);

    form->addRow(m_staticOpponentCheck);
    form->addRow("Static model:", m_staticOpponentCombo);

    auto* optionsGroup = new QGroupBox("Render Options", this);
    auto* optionsLayout = new QVBoxLayout(optionsGroup);
    m_options = new DebugOptionsWidget(bridge, optionsGroup);
    optionsLayout->addWidget(m_options);

    auto* btnRow = new QHBoxLayout();
    m_startBtn = new QPushButton("Start Debug Render", this);
    m_stopBtn  = new QPushButton("Stop", this);
    m_stopBtn->setEnabled(false);
    btnRow->addWidget(m_startBtn);
    btnRow->addWidget(m_stopBtn);
    btnRow->addStretch(1);

    auto* sidebar = new QWidget(this);
    sidebar->setFixedWidth(SIDEBAR_WIDTH);
    auto* sidebarLayout = new QVBoxLayout(sidebar);
    sidebarLayout->setContentsMargins(0, 0, 0, 0);
    sidebarLayout->addWidget(modelGroup);
    sidebarLayout->addWidget(optionsGroup);
    sidebarLayout->addLayout(btnRow);
    sidebarLayout->addStretch(1);

    auto* sessionGroup = new QGroupBox("Session", this);
    auto* sessionLayout = new QVBoxLayout(sessionGroup);

    m_statusLabel = new QLabel("Idle", this);
    m_statusLabel->setWordWrap(true);
    auto* statusRow = new QHBoxLayout();
    statusRow->addWidget(new QLabel("Status:", this));
    statusRow->addWidget(m_statusLabel, 1);

    auto* hintLabel = new QLabel(
        "Start a debug render to watch the selected model in the RocketSim viewer. "
        "With Show Rewards on, the Reward Debugger window opens alongside it "
        "with live per-reward values and the critic's value estimate.",
        this);
    hintLabel->setWordWrap(true);
    hintLabel->setMaximumWidth(HINT_MAX_WIDTH);
    hintLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");

    sessionLayout->addLayout(statusRow);
    sessionLayout->addSpacing(8);
    sessionLayout->addWidget(hintLabel);
    sessionLayout->addStretch(1);

    root->addWidget(sidebar);
    root->addWidget(sessionGroup, 1);

    m_saveTimer = new QTimer(this);
    m_saveTimer->setSingleShot(true);
    m_saveTimer->setInterval(500);
    connect(m_saveTimer, &QTimer::timeout, this, [this]() {
        if (!m_suspendSave) SaveCurrent();
    });

    connect(m_agentCombo,   &AgentComboBox::AgentChanged,     this, [this](const QString&) { OnAgentChanged(); });
    connect(m_buildCombo,   &BuildComboBox::BuildChanged,     this, [this](const QString&) { OnBuildChanged(); });
    connect(m_segmentCombo, &SegmentComboBox::SegmentChanged, this, [this](const QString&) { OnSegmentChanged(); });
    connect(m_stageCombo,   qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { OnStageChanged(); });
    connect(m_staticOpponentCheck, &QCheckBox::toggled, this, [this]() {
        UpdateStaticOpponentControls();
        ScheduleSave();
        UpdateStartEnabled();
    });
    connect(m_staticOpponentCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this]() {
        ScheduleSave();
    });
    connect(m_options,      &DebugOptionsWidget::Changed,     this, &DebugTabWidget::OnOptionsChanged);
    connect(m_startBtn,     &QPushButton::clicked,            this, &DebugTabWidget::OnStartClicked);
    connect(m_stopBtn,      &QPushButton::clicked,            this, &DebugTabWidget::OnStopClicked);

    m_suspendSave = true;
    m_currentAgent = m_agentCombo->CurrentAgentName();
    LoadForAgent(m_currentAgent);
    m_suspendSave = false;

    UpdateStartEnabled();
}

void DebugTabWidget::RefreshLists()
{
    // Refreshing the lists re-emits SegmentChanged; suppress the debounced save so
    // merely showing the tab doesn't rewrite the settings file.
    m_suspendSave = true;

    const QString     build   = m_buildCombo->CurrentBuildName();
    const std::string segment = m_segmentCombo->CurrentSegmentId();
    const QString      stage   = m_stageCombo->currentText();
    const QString staticSelectionKey = m_staticOpponentCombo->currentData().toString();

    m_buildCombo->Refresh(build);
    m_segmentCombo->SetAgentBuild(m_currentAgent, m_buildCombo->CurrentBuildName());
    if (!segment.empty())
        m_segmentCombo->SetCurrentSegment(segment);

    // The segment refresh reset the stage combo to the trained default; restore the prior pick
    // and re-derive team sizes for it.
    if (!stage.isEmpty()) {
        const int stageIdx = m_stageCombo->findText(stage);
        if (stageIdx >= 0) {
            {
                QSignalBlocker blocker(m_stageCombo);
                m_stageCombo->setCurrentIndex(stageIdx);
            }
            RefreshTeamSizes();
        }
    }

    RefreshStaticOpponentOptions(staticSelectionKey);

    m_suspendSave = false;

    UpdateStartEnabled();
}

void DebugTabWidget::FollowTrainingAgent(const QString& agentName)
{
    m_agentCombo->FollowAgent(agentName);
}

void DebugTabWidget::OnAgentChanged()
{
    m_currentAgent = m_agentCombo->CurrentAgentName();
    m_suspendSave = true;
    LoadForAgent(m_currentAgent);
    m_suspendSave = false;
    UpdateStartEnabled();
}

void DebugTabWidget::OnBuildChanged()
{
    m_segmentCombo->SetAgentBuild(m_currentAgent, m_buildCombo->CurrentBuildName());
    RefreshStaticOpponentOptions();
    ScheduleSave();
    UpdateStartEnabled();
}

void DebugTabWidget::OnSegmentChanged()
{
    RefreshStageCombo();
    RefreshTeamSizes();
    ScheduleSave();
    UpdateStartEnabled();
}

void DebugTabWidget::OnStageChanged()
{
    RefreshTeamSizes();
    ScheduleSave();
    UpdateStartEnabled();
}

void DebugTabWidget::OnOptionsChanged()
{
    ScheduleSave();
}

void DebugTabWidget::RefreshStageCombo()
{
    const QString trained = m_segmentCombo->CurrentStageName();

    m_stageCombo->blockSignals(true);
    m_stageCombo->clear();

    for (const std::string& name : RocketForge::StageRegistry::GetAvailableStages())
        m_stageCombo->addItem(QString::fromStdString(name));

    // Default to the segment's trained-with stage when present (empty for imported models
    // without a source stage — the user must then pick a stage before rendering).
    const int idx = trained.isEmpty() ? -1 : m_stageCombo->findText(trained);
    m_stageCombo->setCurrentIndex(idx);

    m_stageCombo->blockSignals(false);

    m_stageTrainedHint->setText(trained.isEmpty()
        ? QString()
        : QString("trained with: %1").arg(trained));
}

void DebugTabWidget::LoadForAgent(const QString& agentName)
{
    m_buildCombo->SetAgent(agentName);

    ForgeDebugSettings settings;   // defaults
    if (!agentName.isEmpty()) {
        auto path = GetDefaultForgeDebugSettingsPath(agentName.toStdString());
        auto result = LoadForgeDebugSettings(path);
        if (result.ok)
            settings = result.value;
    }

    if (!settings.buildName.empty())
        m_buildCombo->SetCurrentBuild(QString::fromStdString(settings.buildName));
    m_segmentCombo->SetAgentBuild(agentName, m_buildCombo->CurrentBuildName());
    if (!settings.segmentId.empty())
        m_segmentCombo->SetCurrentSegment(settings.segmentId);

    // Repopulate stages for the resolved segment, then restore the persisted stage override
    // (defaults to the segment's trained-with stage when no override was saved).
    RefreshStageCombo();
    if (!settings.renderStageName.empty()) {
        const int stageIdx = m_stageCombo->findText(QString::fromStdString(settings.renderStageName));
        if (stageIdx >= 0) {
            QSignalBlocker blocker(m_stageCombo);
            m_stageCombo->setCurrentIndex(stageIdx);
        }
    }

    m_staticOpponentCheck->setChecked(settings.debug.includeStaticOpponent);
    RefreshStaticOpponentOptions(QString::fromStdString(settings.debug.staticOpponentSelectionKey));

    m_options->LoadSettings(settings.debug, settings.showRewards);
    m_debugEnabledRewards = settings.debug.debugEnabledRewards;
    m_hasRewardSelection  = settings.debug.hasRewardSelection;
    m_debugEnabledCriticHeads = settings.debug.debugEnabledCriticHeads;
    m_hasCriticHeadSelection  = settings.debug.hasCriticHeadSelection;
    m_showCriticHeads         = settings.debug.showCriticHeads;
    m_normMode      = (int)settings.debug.normMode;
    m_displayMode   = (int)settings.debug.displayMode;
    m_forceDebugAll = settings.debug.forceDebugAll;
    RefreshTeamSizes();
}

void DebugTabWidget::RefreshStaticOpponentOptions(const QString& preferredSelectionKey)
{
    const QSignalBlocker blocker(m_staticOpponentCombo);
    m_staticOpponentCombo->clear();
    m_staticOpponentOptionsAvailable = false;
    m_staticOpponentUnavailableText.clear();

    if (m_currentAgent.isEmpty() || m_buildCombo->CurrentBuildName().isEmpty()) {
        m_staticOpponentCombo->addItem("No Build selected");
        UpdateStaticOpponentControls();
        return;
    }

    try {
        const auto entries = LoadStaticOpponentSelection(
            m_currentAgent.toStdString(),
            m_buildCombo->CurrentBuildName().toStdString());
        if (entries.empty()) {
            m_staticOpponentCombo->addItem("No static models selected for this Build");
            m_staticOpponentUnavailableText =
                "Select at least one model in the Static Opponents tab for this Build";
            UpdateStaticOpponentControls();
            return;
        }

        m_staticOpponentCombo->addItem("Random (new each episode)", QString());
        for (const StaticOpponentSelectionEntry& entry : entries) {
            const QString label = QString::fromStdString(entry.displayName.empty()
                ? entry.checkpointPath.filename().string()
                : entry.displayName);
            const QString key = QString::fromStdString(MakeStaticOpponentSelectionKey(entry));
            m_staticOpponentCombo->addItem(label, key);
        }
        m_staticOpponentOptionsAvailable = true;

        if (!preferredSelectionKey.isEmpty()) {
            const int preferredIndex = m_staticOpponentCombo->findData(preferredSelectionKey);
            if (preferredIndex >= 0)
                m_staticOpponentCombo->setCurrentIndex(preferredIndex);
        }
    } catch (const std::exception& e) {
        m_staticOpponentCombo->addItem("Static model list unavailable");
        m_staticOpponentUnavailableText = QString("Static model list unavailable: %1")
            .arg(QString::fromUtf8(e.what()));
    }

    UpdateStaticOpponentControls();
}

void DebugTabWidget::UpdateStaticOpponentControls()
{
    const bool canEdit = !m_runActive;
    m_staticOpponentCheck->setEnabled(canEdit);
    m_staticOpponentCombo->setEnabled(
        canEdit && m_staticOpponentOptionsAvailable && m_staticOpponentCheck->isChecked());
}

void DebugTabWidget::RefreshTeamSizes()
{
    m_selectedStageAvailable = true;
    m_stageUnavailableText.clear();

    const QString stageName = m_stageCombo->currentText();
    if (m_currentAgent.isEmpty() || stageName.isEmpty()) {
        m_options->SetTeamSizes({});   // clear stale sizes from a previous selection
        m_options->SetStateSetterOptions({});
        return;
    }

    GlobalSettings minimalGlobals;
    StageSettings  minimalStage;
    minimalStage.stageName = stageName.toStdString();

    std::vector<RocketForge::WeightedTeamSize> teamSizes;
    try {
        auto stage = CreateTrainingStage(m_currentAgent.toStdString(),
                                         stageName.toStdString(),
                                         minimalGlobals, minimalStage);
        if (!stage) {
            m_selectedStageAvailable = false;
            m_stageUnavailableText = QString("Stage %1 is unavailable for %2")
                .arg(stageName, m_currentAgent);
            m_options->SetTeamSizes({});
            m_options->SetStateSetterOptions({});
            return;
        }

        teamSizes = stage->GetWeightedTeamSizes();
        m_options->SetStateSetterOptions(stage->GetDebugStateSetterOptions());
    } catch (const std::exception& e) {
        m_selectedStageAvailable = false;
        m_stageUnavailableText = QString("Stage %1 is unavailable: %2")
            .arg(stageName, QString::fromUtf8(e.what()));
        m_options->SetTeamSizes({});
        m_options->SetStateSetterOptions({});
        return;
    } catch (...) {
        m_selectedStageAvailable = false;
        m_stageUnavailableText = QString("Stage %1 is unavailable").arg(stageName);
        m_options->SetTeamSizes({});
        m_options->SetStateSetterOptions({});
        return;
    }

    m_options->SetTeamSizes(teamSizes);
}

RocketForgeTraining::ForgeDebugSettings DebugTabWidget::CurrentToSettings() const
{
    ForgeDebugSettings s;
    s.buildName   = m_buildCombo->CurrentBuildName().toStdString();
    s.segmentId   = m_segmentCombo->CurrentSegmentId();
    s.renderStageName = m_stageCombo->currentText().toStdString();
    s.showRewards = m_options->IsShowRewardsChecked();
    s.debug       = m_options->ToDebugSettings();
    s.debug.debugEnabledRewards = m_debugEnabledRewards;
    s.debug.hasRewardSelection  = m_hasRewardSelection;
    s.debug.debugEnabledCriticHeads = m_debugEnabledCriticHeads;
    s.debug.hasCriticHeadSelection  = m_hasCriticHeadSelection;
    s.debug.showCriticHeads         = m_showCriticHeads;
    s.debug.normMode      = (ForgeDebug::NormalizationMode)m_normMode;
    s.debug.displayMode   = (ForgeDebug::RewardDisplayMode)m_displayMode;
    s.debug.forceDebugAll = m_forceDebugAll;
    s.debug.includeStaticOpponent = m_staticOpponentCheck->isChecked();
    s.debug.staticOpponentSelectionKey = m_staticOpponentCombo->currentData().toString().toStdString();
    // playerIndex intentionally not persisted; leaves the ToDebugSettings default.
    return s;
}

void DebugTabWidget::SaveCurrent()
{
    if (m_currentAgent.isEmpty())
        return;
    auto path = GetDefaultForgeDebugSettingsPath(m_currentAgent.toStdString());
    std::string err;
    GGL::JsonSaveToFile(CurrentToSettings(), path, &err);
}

void DebugTabWidget::ScheduleSave()
{
    if (!m_suspendSave)
        m_saveTimer->start();
}

void DebugTabWidget::UpdateStartEnabled()
{
    const bool hasStage = !m_stageCombo->currentText().isEmpty();
    const bool canStart = !m_runActive && !m_externalRunActive
        && m_segmentCombo->HasRenderableSelection()
        && hasStage
        && m_selectedStageAvailable
        && (!m_staticOpponentCheck->isChecked() || m_staticOpponentOptionsAvailable);
    m_startBtn->setEnabled(canStart);
    if (m_runActive)
        m_statusLabel->setText(QString("Rendering %1 / %2%3")
            .arg(m_buildCombo->CurrentBuildName(),
                 QString::fromStdString(m_segmentCombo->CurrentSegmentId()),
                 m_staticOpponentCheck->isChecked()
                    ? QString(" | Static: %1").arg(m_staticOpponentCombo->currentText())
                    : QString()));
    else if (m_externalRunActive)
        m_statusLabel->setText("Unavailable — another run is active");
    else if (!m_selectedStageAvailable)
        m_statusLabel->setText(m_stageUnavailableText.isEmpty()
            ? QString("Selected stage is unavailable")
            : m_stageUnavailableText);
    else if (m_staticOpponentCheck->isChecked() && !m_staticOpponentOptionsAvailable)
        m_statusLabel->setText(m_staticOpponentUnavailableText.isEmpty()
            ? QString("No static models are available for this Build")
            : m_staticOpponentUnavailableText);
    else if (!m_segmentCombo->HasRenderableSelection())
        m_statusLabel->setText("Select a build and a segment with a checkpoint");
    else if (m_stageCombo->currentText().isEmpty())
        m_statusLabel->setText("Select a stage to render");
    else
        m_statusLabel->setText("Idle");
}

void DebugTabWidget::OnStartClicked()
{
    const QString stage = m_stageCombo->currentText();

    if (!m_segmentCombo->HasRenderableSelection()
        || stage.isEmpty()
        || !m_selectedStageAvailable
        || (m_staticOpponentCheck->isChecked() && !m_staticOpponentOptionsAvailable))
        return;

    // Imported models carry their resolved paths as item data, so only trained segments need resolving.
    QString checkpointFolder;
    bool    isImported = false;

    if (m_segmentCombo->CurrentIsImported()) {
        isImported = true;

        // FromImportedModel resolves the model folder under the imported root (CurrentVersionPath).
        checkpointFolder = m_segmentCombo->CurrentVersionPath();
        if (checkpointFolder.isEmpty()) {
            m_statusLabel->setText("This imported model has no loadable policy to render.");
            return;
        }
    } else {
        const DebugRenderTarget target = ResolveDebugRenderTarget(
            m_currentAgent.toStdString(), m_buildCombo->CurrentBuildName().toStdString(), m_segmentCombo->CurrentSegmentId());

        if (!target.error.empty()) {
            m_statusLabel->setText(QString::fromStdString(target.error));
            return;
        }

        checkpointFolder = QString::fromStdString(target.checkpointFolder.string());
    }

    emit StartDebugRequested(CurrentToSettings(), m_currentAgent, stage, checkpointFolder, isImported);
}

void DebugTabWidget::OnStopClicked()
{
    emit StopRequested();
}

void DebugTabWidget::SetRewardSelection(QStringList enabledRewardNames)
{
    m_debugEnabledRewards.clear();
    m_debugEnabledRewards.reserve(enabledRewardNames.size());
    for (const QString& n : enabledRewardNames)
        m_debugEnabledRewards.push_back(n.toStdString());
    m_hasRewardSelection = true;
    ScheduleSave();
}

void DebugTabWidget::SetRewardVizConfig(int normMode, int displayMode, bool forceDebugAll)
{
    m_normMode      = normMode;
    m_displayMode   = displayMode;
    m_forceDebugAll = forceDebugAll;
    ScheduleSave();
}

void DebugTabWidget::SetCriticHeadSelection(QStringList enabledHeadNames)
{
    m_debugEnabledCriticHeads.clear();
    m_debugEnabledCriticHeads.reserve(enabledHeadNames.size());
    for (const QString& n : enabledHeadNames)
        m_debugEnabledCriticHeads.push_back(n.toStdString());
    m_hasCriticHeadSelection = true;
    ScheduleSave();
}

void DebugTabWidget::SetShowCriticHeads(bool show)
{
    m_showCriticHeads = show;
    ScheduleSave();
}

void DebugTabWidget::SetRunActive(bool active)
{
    m_runActive = active;
    m_agentCombo->setEnabled(!active);
    m_buildCombo->setEnabled(!active);
    m_segmentCombo->setEnabled(!active);
    UpdateStaticOpponentControls();
    m_options->SetTrainingActive(active);
    m_stopBtn->setEnabled(active);
    UpdateStartEnabled();
}

void DebugTabWidget::SetExternalRunActive(bool active)
{
    m_externalRunActive = active;
    UpdateStartEnabled();
}
