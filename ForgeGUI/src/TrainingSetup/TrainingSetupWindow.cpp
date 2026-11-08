#include "TrainingSetupWindow.h"

#include "FormatUtils.h"
#include "ThemeUtils.h"
#include "RewardVisualizer/RewardComparisonWidget.h"

#include <ForgeTraining.h>
#include <GigaLearnInfer/AgentRegistry.h>
#include <StageRegistry.h>
#include <TrainingStage.h>
#include <TrainingSetup/VersionManagement.h>
#include <GigaLearnCPP/TrainingStorage.h>

#include <QAbstractItemView>
#include <QComboBox>
#include <QDateTime>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <QSignalBlocker>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <vector>

namespace
{
    constexpr const char* NoBuildLabel = "N/A";

    QString FormatMaybeTimestamp(const std::string& rawValue)
    {
        if (rawValue.empty())
            return {};

        const QString raw = QString::fromStdString(rawValue);
        const QDateTime parsed = QDateTime::fromString(raw, "yyyyMMdd-HHmmss");
        if (!parsed.isValid())
            return raw;
        return parsed.toString("MM/dd/yy hh:mm");
    }

    QTableWidgetItem* MakeReadOnlyItem(const QString& text)
    {
        auto* item = new QTableWidgetItem(text);
        item->setFlags(Qt::ItemIsEnabled);
        item->setTextAlignment(Qt::AlignCenter);
        return item;
    }

    QString SegmentDisplayLabel(const GGL::SegmentVersionSummary& segment)
    {
        QString label = QString::fromStdString(segment.id);
        if (segment.latestTimesteps > 0)
            label += QString(" - %1").arg(ForgeGUI::FormatTimesteps(segment.latestTimesteps));
        if (!segment.stageName.empty())
            label += QString(" - %1").arg(QString::fromStdString(segment.stageName));
        return label;
    }
}

TrainingSetupWindow::TrainingSetupWindow(const RocketForge::TrainingSetupSnapshot& snapshot,
                                          const QString& versionId,
                                          const QString& checkpointPath,
                                          const QString& versionPath,
                                          const QString& buildName,
                                          QWidget* parent)
    : QWidget(parent, Qt::Window)
    , m_snapshot(snapshot)
    , m_agentName(QString::fromStdString(snapshot.agentName))
    , m_buildName(buildName.trimmed())
    , m_versionId(versionId.isEmpty() ? QString::fromStdString(snapshot.versionId) : versionId)
    , m_checkpointPath(checkpointPath)
    , m_versionPath(versionPath)
{
    setMinimumSize(760, 560);
    resize(980, 720);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->setSpacing(6);

    m_titleLabel = new QLabel(this);
    root->addWidget(m_titleLabel);

    m_savedAtLabel = new QLabel(this);
    m_savedAtLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");
    root->addWidget(m_savedAtLabel);

    auto* selectorRow = new QHBoxLayout;
    selectorRow->addWidget(new QLabel("Agent:", this));
    m_agentCombo = new QComboBox(this);
    selectorRow->addWidget(m_agentCombo);
    selectorRow->addSpacing(12);
    selectorRow->addWidget(new QLabel("Build:", this));
    m_buildCombo = new QComboBox(this);
    selectorRow->addWidget(m_buildCombo);
    selectorRow->addSpacing(12);
    selectorRow->addWidget(new QLabel("Segment:", this));
    m_segmentCombo = new QComboBox(this);
    selectorRow->addWidget(m_segmentCombo);
    selectorRow->addStretch();
    root->addLayout(selectorRow);

    m_statusLabel = new QLabel(this);
    m_statusLabel->setProperty(ForgeGUI::TextRoleProperty, "muted");
    m_statusLabel->setWordWrap(true);
    root->addWidget(m_statusLabel);

    auto* modeRow = new QHBoxLayout;
    modeRow->addWidget(new QLabel("View:", this));
    m_modeCombo = new QComboBox(this);
    m_modeCombo->addItem("Rewards");
    m_modeCombo->addItem("Hyperparameters");
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &TrainingSetupWindow::OnModeChanged);
    modeRow->addWidget(m_modeCombo);
    modeRow->addStretch();
    root->addLayout(modeRow);

    m_stack = new QStackedWidget(this);

    root->addWidget(m_stack);

    PopulateAgentSelector();
    const int initialAgentIndex = m_agentCombo->findText(m_agentName);
    if (initialAgentIndex >= 0)
        m_agentCombo->setCurrentIndex(initialAgentIndex);
    RefreshBuildSelector(m_buildName);
    RefreshSegmentSelector(m_versionId);

    connect(m_agentCombo, &QComboBox::currentTextChanged,
            this, &TrainingSetupWindow::OnAgentChanged);
    connect(m_buildCombo, &QComboBox::currentTextChanged,
            this, &TrainingSetupWindow::OnBuildChanged);
    connect(m_segmentCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &TrainingSetupWindow::OnSegmentChanged);

    const QString selectedVersionId = m_segmentCombo->currentData().toString();
    const bool loadedSelectedSnapshot =
        !selectedVersionId.isEmpty()
        && selectedVersionId != m_versionId
        && LoadSelectedSnapshot();
    if (!loadedSelectedSnapshot) {
        RebuildPages();
        UpdateTitleLabels();
    }
    OnModeChanged(0);
}

void TrainingSetupWindow::OnModeChanged(int index)
{
    if (!m_stack)
        return;
    if (index >= 0 && index < m_stack->count())
        m_stack->setCurrentIndex(index);
}

void TrainingSetupWindow::OnAgentChanged(const QString& agentName)
{
    if (m_updatingSelectors)
        return;

    m_agentName = agentName;
    RefreshBuildSelector();
    RefreshSegmentSelector();
    if (!LoadSelectedSnapshot())
        LoadCurrentStageSnapshot();
}

void TrainingSetupWindow::OnBuildChanged(const QString& buildName)
{
    if (m_updatingSelectors)
        return;

    m_buildName = buildName.trimmed() == NoBuildLabel ? QString() : buildName.trimmed();
    RefreshSegmentSelector();
    if (!LoadSelectedSnapshot())
        LoadCurrentStageSnapshot();
}

void TrainingSetupWindow::OnSegmentChanged(int /*index*/)
{
    if (m_updatingSelectors)
        return;

    LoadSelectedSnapshot();
}

void TrainingSetupWindow::PopulateAgentSelector()
{
    if (!m_agentCombo)
        return;

    QSignalBlocker blocker(m_agentCombo);
    m_agentCombo->clear();
    for (const std::string& agentName : AgentRegistry::GetAvailableAgents())
        m_agentCombo->addItem(QString::fromStdString(agentName));
}

void TrainingSetupWindow::RefreshBuildSelector(const QString& preferredBuildName)
{
    if (!m_buildCombo)
        return;

    m_updatingSelectors = true;
    const QString previous = preferredBuildName.trimmed().isEmpty() ? m_buildName : preferredBuildName.trimmed();
    QSignalBlocker blocker(m_buildCombo);
    m_buildCombo->clear();

    for (const RocketForgeTraining::BuildSummary& build : RocketForgeTraining::ListAgentBuilds(m_agentName.toStdString()))
        m_buildCombo->addItem(QString::fromStdString(build.buildName));
    if (m_buildCombo->count() == 0)
        m_buildCombo->addItem(NoBuildLabel);

    const int index = m_buildCombo->findText(previous);
    if (index >= 0)
        m_buildCombo->setCurrentIndex(index);
    else
        m_buildCombo->setCurrentIndex(0);
    m_buildName = m_buildCombo->currentText().trimmed() == NoBuildLabel ? QString() : m_buildCombo->currentText().trimmed();
    m_updatingSelectors = false;
}

void TrainingSetupWindow::RefreshSegmentSelector(const QString& preferredSegmentId)
{
    if (!m_segmentCombo)
        return;

    m_updatingSelectors = true;
    QSignalBlocker blocker(m_segmentCombo);
    m_segmentCombo->clear();
    m_segmentCombo->setEnabled(false);
    m_segments.clear();

    m_segments = RocketForgeTraining::ListBuildVersions(
        m_agentName.toStdString(),
        m_buildName.toStdString(),
        {},
        false);
    std::sort(m_segments.begin(), m_segments.end(), [](const GGL::SegmentVersionSummary& a, const GGL::SegmentVersionSummary& b) {
        if (a.latestTimesteps != b.latestTimesteps)
            return a.latestTimesteps < b.latestTimesteps;
        return a.id < b.id;
    });

    for (const GGL::SegmentVersionSummary& segment : m_segments) {
        if (!std::filesystem::exists(std::filesystem::path(segment.rootPath) / RocketForge::TrainingSetupSnapshot::FILE_NAME))
            continue;

        m_segmentCombo->addItem(SegmentDisplayLabel(segment), QString::fromStdString(segment.id));
    }

    const bool hasSegments = m_segmentCombo->count() > 0;
    m_segmentCombo->setEnabled(hasSegments);
    if (hasSegments) {
        int segmentIndex = preferredSegmentId.isEmpty() ? -1 : m_segmentCombo->findData(preferredSegmentId);
        if (segmentIndex < 0)
            segmentIndex = 0;
        m_segmentCombo->setCurrentIndex(segmentIndex);
        if (m_statusLabel)
            m_statusLabel->setText("Showing the saved training setup for the selected segment.");
    } else if (m_statusLabel) {
        m_statusLabel->setText("No saved training setup snapshots were found for this build.");
    }

    m_updatingSelectors = false;
}

bool TrainingSetupWindow::LoadSelectedSnapshot()
{
    if (!m_segmentCombo || m_segmentCombo->currentIndex() < 0)
        return false;

    const QString segmentId = m_segmentCombo->currentData().toString();
    if (segmentId.isEmpty())
        return false;

    auto selected = std::find_if(m_segments.begin(), m_segments.end(), [&](const GGL::SegmentVersionSummary& segment) {
        return segment.id == segmentId.toStdString();
    });
    if (selected == m_segments.end())
        return false;

    const QString stageName = QString::fromStdString(selected->stageName);
    auto stage = RocketForgeTraining::CreateTrainingStage(
        m_agentName.toStdString(),
        stageName.toStdString(),
        RocketForgeTraining::GlobalSettings{},
        RocketForgeTraining::StageSettings{});
    if (!stage) {
        if (m_statusLabel)
            m_statusLabel->setText("Failed to create that training stage.");
        return false;
    }
    stage->learnerConfig.runtime.buildName = m_buildName.toStdString();

    std::optional<RocketForge::TrainingSetupSnapshot> snapshot =
        RocketForge::LoadTrainingSetupSnapshot(
            std::filesystem::path(selected->rootPath) / RocketForge::TrainingSetupSnapshot::FILE_NAME);
    if (!snapshot.has_value()) {
        if (m_statusLabel)
            m_statusLabel->setText("No saved training setup was found for that segment.");
        return false;
    }

    const std::filesystem::path segmentRoot = std::filesystem::path(selected->rootPath);
    const std::filesystem::path checkpointPath = GGL::TrainingStorage::GetMostRecentCheckpointIn(segmentRoot / "checkpoints");

    m_snapshot = snapshot.value();
    m_versionId = segmentId;
    m_checkpointPath = QString::fromStdString(checkpointPath.string());
    m_versionPath = QString::fromStdString(segmentRoot.string());
    RebuildPages();
    UpdateTitleLabels();
    if (m_statusLabel)
        m_statusLabel->setText("Showing the saved training setup for the selected segment.");
    return true;
}

bool TrainingSetupWindow::LoadCurrentStageSnapshot()
{
    RebuildPages();
    UpdateTitleLabels();
    if (m_statusLabel)
        m_statusLabel->setText("No saved segment snapshot is selected. Showing the current queue snapshot.");
    return true;
}

void TrainingSetupWindow::RebuildPages()
{
    if (!m_stack)
        return;

    const int modeIndex = m_modeCombo ? m_modeCombo->currentIndex() : 0;
    while (m_stack->count() > 0) {
        QWidget* page = m_stack->widget(0);
        m_stack->removeWidget(page);
        delete page;
    }

    m_rewardWidget = new RewardComparisonWidget(m_snapshot.BuildRewardTable(), m_stack);
    m_stack->addWidget(m_rewardWidget);

    m_hyperTable = new QTableWidget(m_stack);
    BuildHyperparameterTable();
    m_stack->addWidget(m_hyperTable);

    OnModeChanged(modeIndex);
}

void TrainingSetupWindow::UpdateTitleLabels()
{
    if (m_titleLabel) {
        m_titleLabel->setText(
            QString("%1 / %2 / %3")
                .arg(QString::fromStdString(m_snapshot.agentName))
                .arg(m_buildName)
                .arg(QString::fromStdString(m_snapshot.versionId)));
    }

    if (m_savedAtLabel) {
        if (m_snapshot.savedAt.empty()) {
            m_savedAtLabel->clear();
        } else {
            m_savedAtLabel->setText(
                QString("Saved: %1 | Training Stage: %2")
                    .arg(FormatMaybeTimestamp(m_snapshot.savedAt))
                    .arg(QString::fromStdString(m_snapshot.stageName)));
        }
    }
}

void TrainingSetupWindow::BuildHyperparameterTable()
{
    m_hyperTable->setColumnCount(3);
    m_hyperTable->setHorizontalHeaderLabels({ "Group", "Parameter", "Value" });
    m_hyperTable->setRowCount(static_cast<int>(m_snapshot.hyperparameters.size()));
    m_hyperTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_hyperTable->setSelectionMode(QAbstractItemView::NoSelection);
    m_hyperTable->setFocusPolicy(Qt::NoFocus);
    m_hyperTable->setAlternatingRowColors(true);
    m_hyperTable->verticalHeader()->setVisible(false);
    m_hyperTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_hyperTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_hyperTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);

    for (int row = 0; row < static_cast<int>(m_snapshot.hyperparameters.size()); ++row) {
        const auto& parameter = m_snapshot.hyperparameters[row];
        auto* groupItem = MakeReadOnlyItem(QString::fromStdString(parameter.group));
        auto* nameItem = MakeReadOnlyItem(QString::fromStdString(parameter.name));
        const QString formattedValue = FormatMaybeTimestamp(parameter.value);
        auto* valueItem = MakeReadOnlyItem(formattedValue);
        valueItem->setToolTip(formattedValue);

        m_hyperTable->setItem(row, 0, groupItem);
        m_hyperTable->setItem(row, 1, nameItem);
        m_hyperTable->setItem(row, 2, valueItem);
    }
}
