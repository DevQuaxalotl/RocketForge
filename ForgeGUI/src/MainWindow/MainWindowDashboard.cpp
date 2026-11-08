#include "MainWindow/mainwindow.h"
#include "ui_mainwindow.h"

#include "FormatUtils.h"
#include "MainWindow/DashboardFormatting.h"
#include "MainWindow/MetricCardLayout.h"
#include "MainWindow/SkillRatingsPanel.h"
#include "TrainingSetup/TrainingQueueWidget.h"
#include "Widgets/TrainingMonitorWindow.h"

#include <AgentModelPaths.h>
#include <Storage/BuildStorage.h>

#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLayout>
#include <QLocale>
#include <QRegularExpression>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <set>

using namespace RocketForgeTraining;
using namespace ForgeGUI;

namespace
{
    double SegmentLineageTrainingSeconds(const BuildStorage& storage, const std::string& segmentId)
    {
        double totalSeconds = 0.0;
        std::set<std::string> seen;
        std::string currentId = segmentId;

        while (!currentId.empty() && !seen.contains(currentId)) {
            seen.insert(currentId);

            std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(currentId);

            if (!segment.has_value())
                break;

            totalSeconds += std::max(0.0, segment->totalTrainingSeconds);
            currentId = segment->parentSegmentId;
        }

        return totalSeconds;
    }
}

// ---------------------------------------------------------------------------
// Card layout
// ---------------------------------------------------------------------------

void MainWindow::ConfigureDashboardBlocks()
{
    if (!ui || !ui->metricsRowsLayout || !ui->rightPaneLayout)
        return;

    m_iterationMetricLabels = {
        { "Collected Timesteps",                           ui->collectTimestepsValueLabel },
        { "Average Step Reward",                           ui->collectAvgRewardValueLabel },
        { "Episode Length",                                ui->collectEpisodeLengthValueLabel },
        { "Average Episode Length",                        ui->collectEpisodeLengthValueLabel },
        { "Avg Episode Length",                            ui->collectEpisodeLengthValueLabel },
        { "GAE/Avg Advantage",                             ui->gaeAvgAdvantageValueLabel },
        { "GAE/Avg Return",                                ui->gaeAvgReturnValueLabel },
        { "GAE/Avg Val Target",                            ui->gaeAvgTargetValueLabel },
        { "GAE/Returns STD",                               ui->gaeReturnStdValueLabel },
        { "Clipped Reward Portion",                        ui->gaeRewardClipValueLabel },
        { "LossAndEntropy/Policy Entropy",                 ui->lossEntropyValueLabel },
        { "Policy Entropy",                                ui->lossEntropyValueLabel },
        { "LossAndEntropy/Mean KL Divergence",             ui->lossKlValueLabel },
        { "LossAndEntropy/SB3 Clip Fraction",              ui->lossClipValueLabel },
        { "LossAndEntropy/Policy Loss",                    ui->lossPolicyValueLabel },
        { "LossAndEntropy/Critic Loss",                    ui->lossCriticValueLabel },
        { "LossAndEntropy/Policy Update Magnitude",        ui->updatePolicyValueLabel },
        { "LossAndEntropy/Critic Update Magnitude",        ui->updateCriticValueLabel },
        { "LossAndEntropy/Shared Head Update Magnitude",   ui->updateSharedHeadValueLabel },
        { "Steps Per Second/Overall",                      ui->throughputOverallValueLabel },
        { "Overall Steps/Second",                          ui->throughputOverallValueLabel },
        { "Steps Per Second/Collection",                   ui->throughputCollectionValueLabel },
        { "Steps Per Second/Consumption",                  ui->throughputConsumptionValueLabel },
        { "Collection Time",                               ui->timingCollectionValueLabel },
        { "Consumption Time",                              ui->timingConsumptionValueLabel },
        { "Inference Time",                                ui->timingInferenceValueLabel },
        { "Env Step Time",                                 ui->timingEnvStepValueLabel },
        { "Reward Time",                                   ui->timingRewardValueLabel },
        { "Iteration Time",                                ui->timingIterationValueLabel },
    };

    // One grid keeps all three rows aligned while retaining the existing cards.
    while (QLayoutItem* item = ui->metricsRowsLayout->takeAt(0)) delete item;

    auto* cardGrid = new QGridLayout;
    cardGrid->setContentsMargins(0, 0, 0, 0);
    cardGrid->setSpacing(6);
    for (int column = 0; column < 3; ++column) cardGrid->setColumnStretch(column, 1);
    ui->metricsRowsLayout->addLayout(cardGrid);

    ui->rightPaneLayout->removeWidget(ui->trainingStatsGroup);
    ConfigureMetricGroup(ui->trainingStatsGroup);
    PinGridRowsToTop(ui->trainingStatsGridLayout);
    m_totalTimeValueLabel = ui->totalTimeValueLabel;
    m_sessionTimeValueLabel = ui->sessionTimeValueLabel;
    cardGrid->addWidget(ui->trainingStatsGroup, 0, 0);

    m_progressMetricsGroup = new QGroupBox("Progress", ui->metricsRowsWidget);
    m_progressMetricsGroup->setObjectName("progressMetricsGroup");
    ConfigureMetricGroup(m_progressMetricsGroup);

    auto* progressLayout = new QGridLayout(m_progressMetricsGroup);
    progressLayout->setVerticalSpacing(2);
    progressLayout->setContentsMargins(8, 8, 8, 8);
    m_progressTargetValueLabel = MakeMetricValueLabel("progressTargetValueLabel", m_progressMetricsGroup);
    m_progressSegmentStepsValueLabel = MakeMetricValueLabel("progressSegmentStepsValueLabel", m_progressMetricsGroup);
    m_progressRemainingValueLabel = MakeMetricValueLabel("progressRemainingValueLabel", m_progressMetricsGroup);
    m_progressEtaValueLabel = MakeMetricValueLabel("progressEtaValueLabel", m_progressMetricsGroup);
    m_progressSegmentTimeValueLabel = MakeMetricValueLabel("progressSegmentTimeValueLabel", m_progressMetricsGroup);
    AddMetricRow(progressLayout, 0, "Target:", m_progressTargetValueLabel);
    AddMetricRow(progressLayout, 1, "Segment Steps:", m_progressSegmentStepsValueLabel);
    AddMetricRow(progressLayout, 2, "Remaining:", m_progressRemainingValueLabel);
    AddMetricRow(progressLayout, 3, "ETA:", m_progressEtaValueLabel);
    AddMetricRow(progressLayout, 4, "Segment Time:", m_progressSegmentTimeValueLabel);
    PinGridRowsToTop(progressLayout);
    cardGrid->addWidget(m_progressMetricsGroup, 0, 1);

    ui->rightPaneLayout->removeWidget(ui->skillRatingsGroup);
    ui->skillRatingsGroup->setTitle("Skill (segment delta)");
    ui->skillRatingsGroup->setToolTip(
        "Latest saved or completed skill rating. "
        "Parentheses show the change since this segment began, including earlier sessions. "
        "A dash means the original baseline is unavailable.");
    ui->skillRatingsGroup->setVisible(true);
    ConfigureMetricGroup(ui->skillRatingsGroup);

    m_skillRatings = std::make_unique<SkillRatingsPanel>(ui->skillRatingsContainer);
    UpdateSkillTestTimingRows();
    cardGrid->addWidget(ui->skillRatingsGroup, 0, 2);

    ConfigureMetricGroup(ui->collectMetricsGroup);
    ConfigureMetricGroup(ui->gaeMetricsGroup);
    ConfigureMetricGroup(ui->lossMetricsGroup);
    ConfigureMetricGroup(ui->timingMetricsGroup);
    ConfigureMetricGroup(ui->throughputMetricsGroup);
    ConfigureMetricGroup(ui->updateMetricsGroup);
    PinGridRowsToTop(ui->collectMetricsGridLayout);
    PinGridRowsToTop(ui->gaeMetricsGridLayout);
    PinGridRowsToTop(ui->lossMetricsGridLayout);
    PinGridRowsToTop(ui->timingMetricsGridLayout);
    PinGridRowsToTop(ui->throughputMetricsGridLayout);
    PinGridRowsToTop(ui->updateMetricsGridLayout);

    cardGrid->addWidget(ui->collectMetricsGroup, 1, 0);
    cardGrid->addWidget(ui->gaeMetricsGroup, 1, 1);
    cardGrid->addWidget(ui->lossMetricsGroup, 1, 2);
    cardGrid->addWidget(ui->timingMetricsGroup, 2, 0);
    cardGrid->addWidget(ui->throughputMetricsGroup, 2, 1);
    cardGrid->addWidget(ui->updateMetricsGroup, 2, 2);

    ui->timingMetricsGroup->setTitle("Timing (s)");
    ui->updateMetricsGroup->setToolTip(
        "L2 magnitude of parameter changes during learning. "
        "This is not the learning rate or gradient norm; its scale depends on the number of parameters. "
        "N/A means no value was reported this iteration.");
    for (QGroupBox* group : { ui->collectMetricsGroup, ui->gaeMetricsGroup, ui->lossMetricsGroup,
            ui->timingMetricsGroup, ui->throughputMetricsGroup })
        group->setToolTip("Values from the latest iteration report. N/A means this iteration did not report a finite value.");

    UpdateProgressBlock();
}

// ---------------------------------------------------------------------------
// Active stage and run clocks
// ---------------------------------------------------------------------------

void MainWindow::StartTrainingSessionClock()
{
    m_activeSegmentTrainingSecondsBase = 0.0;
    m_activeLineageTrainingSecondsBase = 0.0;
    m_trainingSessionTimer.restart();
    m_trainingSessionTimerActive = true;
    m_trainingDurationTimerActive = false;
    if (m_totalTimeValueLabel)
        m_totalTimeValueLabel->setText("N/A");
    if (m_sessionTimeValueLabel)
        m_sessionTimeValueLabel->setText(FormatElapsedTrainingDuration(0.0));
    if (m_progressSegmentTimeValueLabel)
        m_progressSegmentTimeValueLabel->setText("N/A");
    if (m_trainingDurationRefreshTimer)
        m_trainingDurationRefreshTimer->start();
}

void MainWindow::StopTrainingSessionClock()
{
    UpdateTrainingDurationDisplays();
    m_trainingSessionTimerActive = false;
    m_trainingDurationTimerActive = false;
    if (m_trainingDurationRefreshTimer)
        m_trainingDurationRefreshTimer->stop();
    if (m_sessionTimeValueLabel)
        m_sessionTimeValueLabel->setText("N/A");
    m_activeStageIndex = -1;
    m_activeBuildName.clear();
    m_activeStageName.clear();
    m_activeVersionId.clear();
}

void MainWindow::BeginActiveStage(int index, const QString& buildName, const QString& stageName, const QString& versionId)
{
    m_activeStageIndex             = index;
    m_activeBuildName              = buildName;
    m_activeStageName              = stageName;
    m_activeVersionId              = versionId;
    m_rollingStepsPerSec           = 0.0;
    m_smoothedEtaSecs              = 0.0;
    m_lastTotalTimesteps           = 0;
    m_nextSkillTestTimestep        = 0;
    m_skillTestIntervalTimesteps   = 0;
    m_hasNextSkillTest             = false;
    m_skillTestInProgress          = false;

    LoadActiveTrainingDurationBaselines(versionId);
    ResetIterationMetrics();
    ui->totalIterationsLabel->setText("N/A");
    ui->totalTimestepsLabel->setText(QLocale(QLocale::English).toString(m_lastTotalTimesteps));
    SeedSkillRatingsFromActiveSegment();

    m_trainingDurationTimer.restart();
    m_trainingDurationTimerActive = true;
    if (m_trainingDurationRefreshTimer)
        m_trainingDurationRefreshTimer->start();

    m_queueWidget->SetActiveStage(index);
    UpdateProgressBlock();
    UpdateSkillTestTimingRows();
    UpdateTrainingDurationDisplays();
}

void MainWindow::EndActiveStage()
{
    UpdateTrainingDurationDisplays();
    m_trainingDurationTimerActive = false;
    m_activeStageIndex = -1;
    m_activeStageName.clear();
    m_activeVersionId.clear();
    UpdateProgressBlock();
}

void MainWindow::LoadActiveTrainingDurationBaselines(const QString& segmentId)
{
    m_activeSegmentTrainingSecondsBase = 0.0;
    m_activeLineageTrainingSecondsBase = 0.0;
    m_activeSegmentStartTimesteps = 0;

    const QString agentName = m_paramTestingPreviewActive && !m_paramTestingPreviewAgent.isEmpty()
        ? m_paramTestingPreviewAgent
        : m_currentAgent;

    if (agentName.isEmpty() || m_activeBuildName.isEmpty() || segmentId.isEmpty())
        return;

    BuildStorage storage(ResolveAgentModelsFolder(agentName.toStdString()), m_activeBuildName.toStdString());
    const std::string segmentIdStd = segmentId.toStdString();
    std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(segmentIdStd);

    if (!segment.has_value())
        return;

    m_activeSegmentTrainingSecondsBase = std::max(0.0, segment->totalTrainingSeconds);
    m_activeLineageTrainingSecondsBase = SegmentLineageTrainingSeconds(storage, segmentIdStd);
    m_activeSegmentStartTimesteps = segment->startTimestep;

    if (m_activeSegmentStartTimesteps == 0 && !segment->sourceCheckpointPath.empty()) {
        const std::optional<int64_t> sourceTimesteps = CheckpointFolderTimesteps(std::filesystem::path(segment->sourceCheckpointPath));
        if (sourceTimesteps.has_value() && sourceTimesteps.value() > 0)
            m_activeSegmentStartTimesteps = static_cast<quint64>(sourceTimesteps.value());
    }

    quint64 latestSegmentTimesteps = segment->endTimestep;
    if (!segment->checkpoints.empty())
        latestSegmentTimesteps = std::max(latestSegmentTimesteps, segment->checkpoints.back().timesteps);
    m_lastTotalTimesteps = std::max(latestSegmentTimesteps, m_activeSegmentStartTimesteps);
}

double MainWindow::CurrentRunTrainingSeconds() const
{
    if (!m_trainingDurationTimerActive || !m_trainingDurationTimer.isValid())
        return 0.0;
    return (double)m_trainingDurationTimer.elapsed() / 1000.0;
}

double MainWindow::CurrentSessionTrainingSeconds() const
{
    if (!m_trainingSessionTimerActive || !m_trainingSessionTimer.isValid())
        return 0.0;
    return (double)m_trainingSessionTimer.elapsed() / 1000.0;
}

// ---------------------------------------------------------------------------
// Card updates
// ---------------------------------------------------------------------------

void MainWindow::UpdateProgressBlock(const QString& etaText)
{
    if (!m_progressTargetValueLabel || !m_progressSegmentStepsValueLabel ||
        !m_progressRemainingValueLabel || !m_progressEtaValueLabel || !m_progressSegmentTimeValueLabel)
        return;

    const bool hasActiveStage = m_activeStageIndex >= 0 && m_queueWidget;
    const std::vector<StageSettings> stages = hasActiveStage ? m_queueWidget->Stages() : std::vector<StageSettings>{};

    if (m_activeStageIndex < 0 || m_activeStageIndex >= (int)stages.size()) {
        m_progressTargetValueLabel->setText("N/A");
        m_progressSegmentStepsValueLabel->setText("N/A");
        m_progressRemainingValueLabel->setText("N/A");
        m_progressEtaValueLabel->setText("N/A");
        m_progressSegmentTimeValueLabel->setText("N/A");
        return;
    }

    const quint64 target = stages[m_activeStageIndex].targetTotalTimesteps;
    const quint64 current = m_lastTotalTimesteps;
    const quint64 segmentSteps = current > m_activeSegmentStartTimesteps
        ? current - m_activeSegmentStartTimesteps
        : 0;
    m_progressSegmentStepsValueLabel->setText(FormatTimesteps(segmentSteps));

    if (target == std::numeric_limits<quint64>::max()) {
        m_progressTargetValueLabel->setText("Unlimited");
        m_progressRemainingValueLabel->setText("Unlimited");
    } else {
        m_progressTargetValueLabel->setText(FormatTimesteps(target));
        m_progressRemainingValueLabel->setText(
            target > current ? FormatTimesteps(target - current) : QString("Done"));
    }

    if (!etaText.isEmpty())
        m_progressEtaValueLabel->setText(EtaValueText(etaText));

    UpdateTrainingDurationDisplays();
}

void MainWindow::UpdateTrainingDurationDisplays()
{
    if (m_sessionTimeValueLabel) {
        m_sessionTimeValueLabel->setText(
            m_trainingSessionTimerActive
                ? FormatElapsedTrainingDuration(CurrentSessionTrainingSeconds())
                : QString("N/A"));
    }

    const bool hasActiveStage = m_activeStageIndex >= 0 && m_trainingDurationTimerActive;

    if (!hasActiveStage) {
        if (m_progressSegmentTimeValueLabel)
            m_progressSegmentTimeValueLabel->setText("N/A");
        return;
    }

    const double runSeconds = CurrentRunTrainingSeconds();
    if (m_totalTimeValueLabel)
        m_totalTimeValueLabel->setText(
            FormatElapsedTrainingDuration(m_activeLineageTrainingSecondsBase + runSeconds));
    if (m_progressSegmentTimeValueLabel)
        m_progressSegmentTimeValueLabel->setText(
            FormatElapsedTrainingDuration(m_activeSegmentTrainingSecondsBase + runSeconds));
}

void MainWindow::ResetIterationMetrics(const QSet<QLabel*>& keep)
{
    // Skill results survive between tests; iteration-specific values must not.
    for (QGridLayout* layout : { ui->collectMetricsGridLayout, ui->gaeMetricsGridLayout, ui->lossMetricsGridLayout,
            ui->timingMetricsGridLayout, ui->throughputMetricsGridLayout, ui->updateMetricsGridLayout }) {
        for (int row = 0; row < layout->rowCount(); ++row) {
            if (QLayoutItem* item = layout->itemAtPosition(row, 1)) {
                auto* label = qobject_cast<QLabel*>(item->widget());
                if (label && !keep.contains(label)) label->setText("N/A");
            }
        }
    }
}

void MainWindow::ResetSkillRatings()
{
    if (!m_skillRatings)
        return;

    m_skillRatings->Reset();
    UpdateSkillTestTimingRows();
}

void MainWindow::SeedSkillRatingsFromActiveSegment()
{
    m_hasLastSkillTest = false;
    ResetSkillRatings();

    const QString agent = m_paramTestingPreviewActive ? m_paramTestingPreviewAgent : m_currentAgent;
    if (!m_skillRatings || agent.isEmpty() || m_activeBuildName.isEmpty() || m_activeVersionId.isEmpty()) return;

    BuildStorage storage(ResolveAgentModelsFolder(agent.toStdString()), m_activeBuildName.toStdString());
    if (const std::optional<TrainingSegmentMetadata> segment = storage.LoadSegmentById(m_activeVersionId.toStdString()))
        m_skillRatings->SeedFromSegment(storage, *segment);
}

void MainWindow::UpdateSkillTestTimingRows()
{
    if (!m_skillRatings)
        return;

    QString nextTestText = "N/A";

    if (m_skillTestInProgress) {
        nextTestText = "Running";
    } else if (m_hasNextSkillTest) {
        const quint64 remaining = m_nextSkillTestTimestep > m_lastTotalTimesteps
            ? m_nextSkillTestTimestep - m_lastTotalTimesteps
            : 0;
        nextTestText = QString("%1 (in %2)")
            .arg(FormatSkillTestTimesteps(m_nextSkillTestTimestep), FormatSkillTestTimesteps(remaining));
    }

    const QString previousTestText = m_hasLastSkillTest ? FormatElapsedTrainingDuration(m_lastSkillTestSecs) : QString("N/A");
    m_skillRatings->SetTestTimingRows(nextTestText, previousTestText);
}

// ---------------------------------------------------------------------------
// Log event routing
// ---------------------------------------------------------------------------

void MainWindow::OnLogEvent(TrainingLogEvent event)
{
    switch (event.type) {

    // The log can deliver a report in pieces, so stale values are cleared only once the whole report has arrived.
    case TrainingLogEvent::Type::ReportStarted:
        m_reportedMetricLabels.clear();
        break;

    case TrainingLogEvent::Type::ReportFinished:
        ResetIterationMetrics(m_reportedMetricLabels);
        break;

    case TrainingLogEvent::Type::TrainingStat: {
        const QString key = NormalizeReportKey(event.fields["key"]);
        const QString rawValue = event.fields["value"];
        const QString valueText = FormatTrainingMetric(key, rawValue);
        ui->trainingStatsGroup->setVisible(true);

        if (key == "Total Timesteps")
            ui->totalTimestepsLabel->setText(valueText);
        else if (key == "Total Iterations")
            ui->totalIterationsLabel->setText(valueText);
        else if (QLabel* metricLabel = m_iterationMetricLabels.value(key)) {
            metricLabel->setText(valueText);
            m_reportedMetricLabels.insert(metricLabel);
        } else if (key.startsWith("SkillTest/Rating/")) {
            m_skillRatings->UpdateRating(key.mid(QString("SkillTest/Rating/").length()), rawValue);
        } else if (key == "SkillTest/TestTimeSeconds") {
            bool ok = false;
            const double seconds = QString(rawValue).remove(',').toDouble(&ok);

            if (ok && std::isfinite(seconds) && seconds > 0.0) {
                m_lastSkillTestSecs = seconds;
                m_hasLastSkillTest = true;
                UpdateSkillTestTimingRows();
            }
        } else if (key == "SkillTest/NextRunTimestep") {
            quint64 nextRun = 0;

            if (ParseMetricTimesteps(rawValue, nextRun)) {
                m_nextSkillTestTimestep = nextRun;
                m_skillTestIntervalTimesteps = nextRun > m_lastTotalTimesteps ? nextRun - m_lastTotalTimesteps : 0;
                m_hasNextSkillTest = true;
                UpdateSkillTestTimingRows();
            }
        }

        if (key == "Total Timesteps" && m_activeStageIndex >= 0) {
            quint64 totalTimesteps = 0;

            if (ParseMetricTimesteps(rawValue, totalTimesteps)) {
                m_lastTotalTimesteps = totalTimesteps;
                m_queueWidget->UpdateLiveTotalTimesteps(m_activeStageIndex, totalTimesteps);
                UpdateSkillTestTimingRows();
                UpdateProgressBlock();
                UpdateActiveStageEta();
            }
        }

        if ((key == "Steps Per Second/Overall" || key == "Overall Steps/Second") && m_activeStageIndex >= 0) {
            bool ok = false;
            const double stepsPerSecond = QString(rawValue).remove(',').toDouble(&ok);

            if (ok && std::isfinite(stepsPerSecond) && stepsPerSecond > 0.0) {
                const double smoothing = 0.15;
                m_rollingStepsPerSec = (m_rollingStepsPerSec == 0.0)
                    ? stepsPerSecond
                    : (smoothing * stepsPerSecond + (1.0 - smoothing) * m_rollingStepsPerSec);

                UpdateActiveStageEta();
            }
        }

        break;
    }

    case TrainingLogEvent::Type::RunInfo: {
        const QString& key = event.fields["key"];
        const QString& value = event.fields["value"];
        ui->runInfoGroup->setVisible(true);

        if (key == "checkpoint") {
            QString displayPath = value;
            displayPath.replace('\\', QString('\\') + QChar(0x200b));
            displayPath.replace('/', QString('/') + QChar(0x200b));
            ui->checkpointValueLabel->setText(displayPath);
            ui->checkpointValueLabel->setToolTip(value);
            m_skillRatings->SeedFromCheckpointRoot(value, !m_skillRatings->HasCurrentRatings());
        }
        else if (key == "checkpoint_resume") {
            m_skillRatings->SeedFromCheckpointRoot(value, !m_skillRatings->HasCurrentRatings());
        }
        else if (key == "device")        ui->deviceValueLabel->setText(value);
        else if (key == "obs_size")      ui->obsSizeValueLabel->setText(value);
        else if (key == "action_count")  ui->actionCountValueLabel->setText(value);
        else if (key == "model_params")  ui->modelParamsValueLabel->setText(value);
        else if (key == "run_id") {
            ui->runIdValueLabel->setText(value);
            ui->runIdValueLabel->setToolTip(value);
        }
        else if (key == "learning_rate") ui->learningRateValueLabel->setText(value);
        break;
    }

    case TrainingLogEvent::Type::SkillRating: {
        const QString mode    = event.fields["mode"];
        const QString rating  = event.fields["rating"];
        m_skillRatings->UpdateRating(mode, rating);
        break;
    }

    case TrainingLogEvent::Type::Error:
        m_lastTrainingError = event.rawLine.trimmed();
        AppendLog(event.rawLine, "#ff4444");
        break;

    case TrainingLogEvent::Type::Raw:
    default:
        UpdateSkillTestEtaStateFromRawLine(event.rawLine);
        AppendLog(event.rawLine);
        break;
    }
}

// ---------------------------------------------------------------------------
// ETA tracking
// ---------------------------------------------------------------------------

void MainWindow::UpdateSkillTestEtaStateFromRawLine(const QString& line)
{
    const QString trimmed = line.trimmed();

    static const QRegularExpression rxStarted(
        R"(^Running skill test at\s+([0-9.,]+[KMB]?)\b)",
        QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression rxCompleted(
        R"(^Skill test completed in\s+([0-9]+(?:\.[0-9]+)?)s\s*$)",
        QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression rxScheduled(
        R"(^Next skill test scheduled at\s+([0-9.,]+[KMB]?)\s+\(in\s+([0-9.,]+[KMB]?)\s+timesteps\)\s*$)",
        QRegularExpression::CaseInsensitiveOption);

    QRegularExpressionMatch match;

    if ((match = rxStarted.match(trimmed)).hasMatch()) {
        quint64 startedAt = 0;

        if (ParseCompactTimesteps(match.captured(1), startedAt) && startedAt > m_lastTotalTimesteps) {
            m_lastTotalTimesteps = startedAt;
            if (m_activeStageIndex >= 0)
                m_queueWidget->UpdateLiveTotalTimesteps(m_activeStageIndex, startedAt);
        }

        m_skillTestInProgress = true;
        m_hasNextSkillTest = false;
        UpdateSkillTestTimingRows();
        UpdateActiveStageEta();
        return;
    }

    if ((match = rxCompleted.match(trimmed)).hasMatch()) {
        bool ok = false;
        const double seconds = match.captured(1).toDouble(&ok);

        if (ok && seconds > 0.0) {
            const double smoothing = 0.25;
            m_avgSkillTestSecs = (m_avgSkillTestSecs == 0.0)
                ? seconds
                : (smoothing * seconds + (1.0 - smoothing) * m_avgSkillTestSecs);
            m_lastSkillTestSecs = seconds;
            m_hasLastSkillTest = true;
        }

        m_skillTestInProgress = false;
        UpdateSkillTestTimingRows();
        UpdateActiveStageEta();
        return;
    }

    if ((match = rxScheduled.match(trimmed)).hasMatch()) {
        quint64 nextRun = 0;
        quint64 interval = 0;
        if (ParseCompactTimesteps(match.captured(1), nextRun) &&
            ParseCompactTimesteps(match.captured(2), interval)) {
            const quint64 scheduledAt = nextRun >= interval ? nextRun - interval : 0;

            if (scheduledAt > m_lastTotalTimesteps) {
                m_lastTotalTimesteps = scheduledAt;
                if (m_activeStageIndex >= 0)
                    m_queueWidget->UpdateLiveTotalTimesteps(m_activeStageIndex, scheduledAt);
            }

            m_nextSkillTestTimestep = nextRun;
            m_skillTestIntervalTimesteps = interval;
            m_hasNextSkillTest = true;
        }

        UpdateSkillTestTimingRows();
        UpdateActiveStageEta();
    }
}

double MainWindow::EstimateRemainingSkillTestSeconds(quint64 target, quint64 current) const
{
    if (m_avgSkillTestSecs <= 0.0 || target <= current)
        return 0.0;

    quint64 estimatedTests = m_skillTestInProgress ? 1 : 0;

    if (m_hasNextSkillTest && m_nextSkillTestTimestep > current && m_nextSkillTestTimestep <= target) {
        estimatedTests++;
        if (m_skillTestIntervalTimesteps > 0 && target > m_nextSkillTestTimestep)
            estimatedTests += (target - m_nextSkillTestTimestep) / m_skillTestIntervalTimesteps;
    }

    return (double)estimatedTests * m_avgSkillTestSecs;
}

void MainWindow::UpdateActiveStageEta()
{
    if (m_activeStageIndex < 0)
        return;

    const std::vector<StageSettings> stages = m_queueWidget->Stages();

    if (m_activeStageIndex >= (int)stages.size())
        return;

    const quint64 target = stages[m_activeStageIndex].targetTotalTimesteps;
    const quint64 current = m_lastTotalTimesteps;

    if (target == std::numeric_limits<quint64>::max()) {
        UpdateProgressBlock();
        return;
    }

    if (target <= current) {
        const QString etaText = FormatEta(0.0);
        m_queueWidget->UpdateLiveEta(m_activeStageIndex, etaText);
        UpdateProgressBlock(etaText);
        return;
    }

    if (m_rollingStepsPerSec <= 1.0) {
        UpdateProgressBlock();
        return;
    }

    const double trainingSecs = (double)(target - current) / m_rollingStepsPerSec;
    const double smoothing = 0.1;
    m_smoothedEtaSecs = (m_smoothedEtaSecs == 0.0)
        ? trainingSecs
        : (smoothing * trainingSecs + (1.0 - smoothing) * m_smoothedEtaSecs);

    const double totalSecs = m_smoothedEtaSecs + EstimateRemainingSkillTestSeconds(target, current);
    const QString etaText = FormatEta(totalSecs);
    m_queueWidget->UpdateLiveEta(m_activeStageIndex, etaText);
    UpdateProgressBlock(etaText);
}

TrainingMonitorSnapshot MainWindow::BuildTrainingMonitorSnapshot() const
{
    TrainingMonitorSnapshot snapshot;

    if (m_trainingActive || m_paramTestingPreviewActive) {
        if (m_stopRequested)
            snapshot.status = "Stopping...";
        else if (m_skillTestInProgress)
            snapshot.status = "Skill test running";
        else
            snapshot.status = "Training: " + (m_paramTestingPreviewActive ? m_paramTestingPreviewAgent : m_currentAgent);
    } else {
        snapshot.status = m_runOutcome;
    }

    snapshot.lastError = m_lastTrainingError;

    snapshot.stage = m_activeStageName.isEmpty() ? QString("No active segment") : m_activeStageName + "  |  " + m_activeVersionId;
    snapshot.totalTimesteps = m_lastTotalTimesteps > 0 ? FormatTimesteps(m_lastTotalTimesteps) : QString("N/A");
    snapshot.stepsPerSecond = m_rollingStepsPerSec > 0.0 ? QLocale().toString(qRound64(m_rollingStepsPerSec)) : QString("N/A");

    // The dashboard labels already hold the formatted ETA and durations; reusing them keeps both views in agreement.
    snapshot.eta = m_progressEtaValueLabel ? m_progressEtaValueLabel->text() : QString("N/A");
    snapshot.segmentTime = m_progressSegmentTimeValueLabel ? m_progressSegmentTimeValueLabel->text() : QString("N/A");
    snapshot.sessionTime = m_sessionTimeValueLabel ? m_sessionTimeValueLabel->text() : QString("N/A");
    snapshot.totalTime = m_totalTimeValueLabel ? m_totalTimeValueLabel->text() : QString("N/A");

    snapshot.progressFallbackText = "N/A";
    const std::vector<StageSettings> stages = m_queueWidget ? m_queueWidget->Stages() : std::vector<StageSettings>{};

    if (m_activeStageIndex >= 0 && m_activeStageIndex < (int)stages.size()) {
        const quint64 target = stages[m_activeStageIndex].targetTotalTimesteps;
        const quint64 segmentSteps = m_lastTotalTimesteps > m_activeSegmentStartTimesteps
            ? m_lastTotalTimesteps - m_activeSegmentStartTimesteps
            : 0;
        if (target == std::numeric_limits<quint64>::max())
            snapshot.progressFallbackText = "Unlimited";
        else if (target > m_activeSegmentStartTimesteps)
            snapshot.progressFraction = (double)segmentSteps / (double)(target - m_activeSegmentStartTimesteps);
    }

    return snapshot;
}
