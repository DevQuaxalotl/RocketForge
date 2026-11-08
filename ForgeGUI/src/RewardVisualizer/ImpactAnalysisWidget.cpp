#include "ImpactAnalysisWidget.h"

#include "RewardImpactWidget.h"
#include "ThemeUtils.h"
#include "Widgets/AgentComboBox.h"
#include "Widgets/BuildComboBox.h"
#include "Widgets/SegmentComboBox.h"

#include <StageRegistry.h>
#include <Storage/BuildStorage.h>
#include <TrainingSetup/VersionManagement.h>
#include <GigaLearnCPP/SegmentVersioning.h>

#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <vector>

namespace
{
    constexpr int SIDEBAR_WIDTH = 520;
}

ImpactAnalysisWidget::ImpactAnalysisWidget(QWidget* parent)
    : QWidget(parent)
{
    QHBoxLayout* root = new QHBoxLayout(this);

    QGroupBox* modelGroup = new QGroupBox("Model", this);
    QFormLayout* form = new QFormLayout(modelGroup);
    m_agentCombo   = new AgentComboBox(this);
    m_buildCombo   = new BuildComboBox(this);
    m_segmentCombo = new SegmentComboBox(this);
    m_stageCombo   = new QComboBox(this);

    // Imported bots have no build; let the segment combo list them when no build is selected.
    m_segmentCombo->SetIncludeImportedModels(true);

    form->addRow("Agent:",   m_agentCombo);
    form->addRow("Build:",   m_buildCombo);
    form->addRow("Segment:", m_segmentCombo);

    m_trainedHint = new QLabel(this);
    m_trainedHint->setProperty(ForgeGUI::TextRoleProperty, "muted");
    form->addRow("Stage:", m_stageCombo);
    form->addRow(QString(), m_trainedHint);

    m_statusLabel = new QLabel(this);
    m_statusLabel->setWordWrap(true);
    m_statusLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");

    QWidget* sidebar = new QWidget(this);
    sidebar->setFixedWidth(SIDEBAR_WIDTH);
    QVBoxLayout* sidebarLayout = new QVBoxLayout(sidebar);
    sidebarLayout->setContentsMargins(0, 0, 0, 0);
    sidebarLayout->addWidget(modelGroup);
    sidebarLayout->addWidget(m_statusLabel);
    sidebarLayout->addStretch(1);

    // Results engine. Constructed with an empty context; SetContext drives it.
    QGroupBox* resultsGroup = new QGroupBox("Reward Impact", this);
    QVBoxLayout* resultsLayout = new QVBoxLayout(resultsGroup);
    m_results = new RewardImpactWidget(QString(), QString(), QString(), QString(), QString(), resultsGroup);
    resultsLayout->addWidget(m_results);

    root->addWidget(sidebar);
    root->addWidget(resultsGroup, 1);

    connect(m_agentCombo,   &AgentComboBox::AgentChanged,     this, &ImpactAnalysisWidget::OnAgentChanged);
    connect(m_buildCombo,   &BuildComboBox::BuildChanged,     this, &ImpactAnalysisWidget::OnBuildChanged);
    connect(m_segmentCombo, &SegmentComboBox::SegmentChanged, this, &ImpactAnalysisWidget::OnSegmentChanged);
    connect(m_stageCombo,   QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ImpactAnalysisWidget::OnStageChanged);

    Refresh();
}

void ImpactAnalysisWidget::Refresh()
{
    // Populate() blocks signals, so drive the dependent selectors manually.
    m_agentCombo->Populate();
    OnAgentChanged();
}

void ImpactAnalysisWidget::FollowTrainingAgent(const QString& agentName)
{
    m_agentCombo->FollowAgent(agentName);
}

void ImpactAnalysisWidget::OnAgentChanged()
{
    // SetAgent() repopulates with signals blocked, so cascade manually.
    m_buildCombo->SetAgent(m_agentCombo->CurrentAgentName());
    OnBuildChanged();
}

void ImpactAnalysisWidget::OnBuildChanged()
{
    // SetAgentBuild() repopulates and always emits SegmentChanged -> OnSegmentChanged.
    m_segmentCombo->SetAgentBuild(m_agentCombo->CurrentAgentName(), m_buildCombo->CurrentBuildName());
}

void ImpactAnalysisWidget::OnSegmentChanged()
{
    RefreshStageCombo();
    PushContext();
}

void ImpactAnalysisWidget::OnStageChanged()
{
    PushContext();
}

void ImpactAnalysisWidget::RefreshStageCombo()
{
    const QString trained = m_segmentCombo->CurrentStageName();

    m_stageCombo->blockSignals(true);
    m_stageCombo->clear();

    for (const std::string& name : RocketForge::StageRegistry::GetAvailableStages())
        m_stageCombo->addItem(QString::fromStdString(name));

    // Default to the segment's trained-with stage when present.
    const int idx = trained.isEmpty() ? -1 : m_stageCombo->findText(trained);
    if (idx >= 0)
        m_stageCombo->setCurrentIndex(idx);

    m_stageCombo->blockSignals(false);

    m_trainedHint->setText(trained.isEmpty()
        ? QString()
        : QString("trained with: %1").arg(trained));
}

void ImpactAnalysisWidget::ClearResults(const QString& statusText)
{
    m_statusLabel->setText(statusText);
    m_results->SetContext(QString(), QString(), QString(), QString(), QString());
}

void ImpactAnalysisWidget::PushContext()
{
    const QString agent         = m_agentCombo->CurrentAgentName();
    const std::string segmentId = m_segmentCombo->CurrentSegmentId();
    const QString stage         = m_stageCombo->currentText();

    if (agent.isEmpty() || segmentId.empty() || stage.isEmpty()) {
        ClearResults("Select an agent, build, segment, and stage.");
        return;
    }

    if (m_segmentCombo->CurrentIsImported()) {
        // Imported bot: the combo already resolved the paths; no build/ListBuildVersions needed.
        const QString checkpointPath = m_segmentCombo->CurrentCheckpointPath();
        const QString versionPath    = m_segmentCombo->CurrentVersionPath();

        if (checkpointPath.isEmpty() || versionPath.isEmpty()) {
            ClearResults("This imported model has no loadable GigaLearn policy (POLICY.lt), "
                         "so reward analysis can't run on it. Import a model with a trainable "
                         "policy, or use a trained build.");
            return;
        }

        const QString trained = m_segmentCombo->CurrentStageName();
        const bool offStage = !trained.isEmpty() && stage != trained;
        m_statusLabel->setText(offStage
            ? QString("Running %1 against stage '%2' (imported; source stage '%3').")
                  .arg(QString::fromStdString(segmentId), stage, trained)
            : QString("Ready: %1 / %2 (imported)").arg(QString::fromStdString(segmentId), stage));

        m_results->SetContext(agent, stage, QString::fromStdString(segmentId), checkpointPath, versionPath);
        return;
    }

    // Trained bot: resolve via the selected build. Imported bots have no build, so this
    // guard only gates the trained path.
    const QString build = m_buildCombo->CurrentBuildName();

    if (build.isEmpty()) {
        ClearResults("Select an agent, build, segment, and stage.");
        return;
    }

    // Resolve the segment's root + newest checkpoint, durable or autosave.
    // includeInactive=true so any segment the combo can show is resolvable here.
    std::vector<GGL::SegmentVersionSummary> segments =
        RocketForgeTraining::ListBuildVersions(agent.toStdString(), build.toStdString(), {}, true);

    auto found = std::find_if(segments.begin(), segments.end(),
        [&](const GGL::SegmentVersionSummary& summary) { return summary.id == segmentId; });

    if (found == segments.end()) {
        ClearResults("Could not resolve the selected segment.");
        return;
    }

    const std::filesystem::path segmentRoot = found->rootPath;
    const std::optional<RocketForgeTraining::ResolvedCheckpoint> latest =
        RocketForgeTraining::BuildStorage::LatestSegmentCheckpointIn(segmentRoot);
    const std::filesystem::path checkpointPath = latest ? latest->folder : std::filesystem::path();

    const bool offStage = !found->stageName.empty() && stage.toStdString() != found->stageName;
    m_statusLabel->setText(offStage
        ? QString("Running %1 against stage '%2' (trained with '%3').")
              .arg(QString::fromStdString(segmentId), stage, QString::fromStdString(found->stageName))
        : QString("Ready: %1 / %2").arg(QString::fromStdString(segmentId), stage));

    m_results->SetContext(
        agent,
        stage,
        QString::fromStdString(segmentId),
        QString::fromStdString(checkpointPath.string()),
        QString::fromStdString(segmentRoot.string()));
}
