#include "RewardImpactWidget.h"
#include "AnalysisWorker.h"
#include "ThemeUtils.h"

#include <RLGymCPP/Rewards/Reward.h>

#include <QComboBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QMetaObject>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QThread>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

using namespace std;
using namespace RLGC;
using namespace RocketForge;

namespace
{

constexpr float ZERO_GUARD = 1e-6f;
constexpr int HINT_MAX_WIDTH = 720;
/** Maximum time to wait for the worker thread to finish on shutdown. **/
constexpr int WORKER_SHUTDOWN_TIMEOUT_MS = 5000;
/** Stylesheets cached at file scope so we don't allocate per cell. **/
const QString POSITIVE_BAR_STYLE = QString("QProgressBar::chunk { background-color: %1; }").arg(ForgeGUI::Theme::POSITIVE.name());
const QString NEGATIVE_BAR_STYLE = QString("QProgressBar::chunk { background-color: %1; }").arg(ForgeGUI::Theme::NEGATIVE.name());

QTableWidgetItem* ReadOnlyItem(const QString& text)
{
    QTableWidgetItem* item = new QTableWidgetItem(text);
    item->setFlags(Qt::ItemIsEnabled);
    item->setTextAlignment(Qt::AlignCenter);
    return item;
}

} // namespace

RewardImpactWidget::RewardImpactWidget(const QString& agentName,
                                       const QString& stageName,
                                       const QString& versionId,
                                       const QString& checkpointPath,
                                       const QString& versionPath,
                                       QWidget* parent)
    : QWidget(parent)
    , m_agentName(agentName)
    , m_stageName(stageName)
    , m_versionId(versionId)
    , m_checkpointPath(checkpointPath)
    , m_versionPath(versionPath)
{
    // Worker lives on its own thread. Standard QThread pattern: worker has no QObject
    // parent (parents and threads don't mix), and is auto-deleted when the thread exits.
    // The thread itself also has no parent — Qt must not auto-delete a running QThread via
    // parent-child destruction, as that races the thread's event loop. We delete it
    // explicitly in the destructor after quit() + wait().
    m_workerThread = new QThread();
    m_worker       = new AnalysisWorker;
    m_worker->moveToThread(m_workerThread);
    connect(m_workerThread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(m_worker, &AnalysisWorker::progressUpdated,  this, &RewardImpactWidget::OnProgressUpdated);
    connect(m_worker, &AnalysisWorker::analysisComplete, this, &RewardImpactWidget::OnAnalysisComplete);
    connect(m_worker, &AnalysisWorker::analysisFailed,   this, &RewardImpactWidget::OnAnalysisFailed);
    m_workerThread->start();

    m_stack = new QStackedWidget(this);
    QVBoxLayout* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->addWidget(m_stack);

    // -- Ready page --
    m_readyPage = new QWidget(m_stack);
    QVBoxLayout* readyLayout = new QVBoxLayout(m_readyPage);
    readyLayout->setAlignment(Qt::AlignTop);

    QHBoxLayout* tsRow = new QHBoxLayout;
    tsRow->addWidget(new QLabel("Timesteps per team size:", m_readyPage));
    m_timestepSpin = new QSpinBox(m_readyPage);
    m_timestepSpin->setRange(0, 10'000'000);
    m_timestepSpin->setSingleStep(1000);
    m_timestepSpin->setValue(0);
    m_timestepSpin->setSpecialValueText("Segment default");
    m_timestepSpin->setToolTip("0 = use the segment's training batch size");
    tsRow->addWidget(m_timestepSpin);
    tsRow->addStretch();
    readyLayout->addLayout(tsRow);

    QPushButton* runBtn = new QPushButton("Run Analysis", m_readyPage);
    connect(runBtn, &QPushButton::clicked, this, &RewardImpactWidget::OnRunClicked);
    readyLayout->addWidget(runBtn, 0, Qt::AlignLeft);

    QLabel* readyHint = new QLabel(
        "Plays the selected model on the chosen stage for each team size, then ranks every reward by its impact "
        "on the return, alongside its signal magnitude and adaptive rates. Results are cached per segment and stage.",
        m_readyPage);
    readyHint->setWordWrap(true);
    readyHint->setMaximumWidth(HINT_MAX_WIDTH);
    readyHint->setProperty(ForgeGUI::TextRoleProperty, "muted");
    readyLayout->addSpacing(8);
    readyLayout->addWidget(readyHint);
    readyLayout->addStretch();
    m_stack->addWidget(m_readyPage);

    // -- Running page --
    m_runningPage = new QWidget(m_stack);
    QVBoxLayout* runningLayout = new QVBoxLayout(m_runningPage);
    runningLayout->setAlignment(Qt::AlignTop);

    m_progressLabel = new QLabel("Starting...", m_runningPage);
    runningLayout->addWidget(m_progressLabel);

    m_progressBar = new QProgressBar(m_runningPage);
    m_progressBar->setRange(0, 100);
    runningLayout->addWidget(m_progressBar);

    QPushButton* cancelBtn = new QPushButton("Cancel", m_runningPage);
    connect(cancelBtn, &QPushButton::clicked, this, &RewardImpactWidget::OnCancelClicked);
    runningLayout->addWidget(cancelBtn, 0, Qt::AlignLeft);
    runningLayout->addStretch();
    m_stack->addWidget(m_runningPage);

    // -- Results page --
    m_resultsPage = new QWidget(m_stack);
    QVBoxLayout* resultsLayout = new QVBoxLayout(m_resultsPage);
    resultsLayout->setAlignment(Qt::AlignTop);

    m_summaryLabel = new QLabel(m_resultsPage);
    m_summaryLabel->setWordWrap(true);
    resultsLayout->addWidget(m_summaryLabel);

    QHBoxLayout* selectorRow = new QHBoxLayout;
    selectorRow->addWidget(new QLabel("View:", m_resultsPage));
    m_teamSizeCombo = new QComboBox(m_resultsPage);
    connect(m_teamSizeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &RewardImpactWidget::OnTeamSizeChanged);
    selectorRow->addWidget(m_teamSizeCombo);

    selectorRow->addSpacing(16);
    selectorRow->addWidget(new QLabel("Sort:", m_resultsPage));
    m_sortCombo = new QComboBox(m_resultsPage);
    m_sortCombo->addItem("By impact");
    m_sortCombo->addItem("Events first");
    m_sortCombo->addItem("Continuous first");
    connect(m_sortCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &RewardImpactWidget::OnSortChanged);
    selectorRow->addWidget(m_sortCombo);
    selectorRow->addStretch();
    resultsLayout->addLayout(selectorRow);

    m_impactTable = new QTableWidget(m_resultsPage);
    m_impactTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_impactTable->setSelectionMode(QAbstractItemView::NoSelection);
    m_impactTable->setFocusPolicy(Qt::NoFocus);
    m_impactTable->setAlternatingRowColors(true);
    m_impactTable->verticalHeader()->setVisible(false);
    resultsLayout->addWidget(m_impactTable);

    QPushButton* rerunBtn = new QPushButton("Re-run", m_resultsPage);
    connect(rerunBtn, &QPushButton::clicked, this, &RewardImpactWidget::ShowReady);
    resultsLayout->addWidget(rerunBtn, 0, Qt::AlignLeft);

    m_stack->addWidget(m_resultsPage);

    // Show cached results immediately if a previous run was saved for this checkpoint.
    TryLoadCachedResult();
}

RewardImpactWidget::~RewardImpactWidget()
{
    // Shutdown sequence:
    // 1. Cancel() unblocks RunRewardAnalysis between episodes (sets atomic flag).
    // 2. quit() asks the event loop to exit after the current slot returns.
    // 3. wait() blocks until the thread's run() returns.
    // The QThread::finished -> deleteLater connection then frees the worker.
    if (m_worker) m_worker->Cancel();
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait(WORKER_SHUTDOWN_TIMEOUT_MS);
        delete m_workerThread;
        m_workerThread = nullptr;
    }
}

void RewardImpactWidget::SetContext(const QString& agentName,
                                    const QString& stageName,
                                    const QString& versionId,
                                    const QString& checkpointPath,
                                    const QString& versionPath)
{
    // Cancel() is a non-blocking atomic-flag set. Bump the run id so a signal still queued from
    // an in-flight run can't land after we repoint and clobber the new context (the On* slots
    // all filter stale runIds).
    if (m_worker) m_worker->Cancel();
    ++m_currentRunId;

    m_agentName      = agentName;
    m_stageName      = stageName;
    m_versionId      = versionId;
    m_checkpointPath = checkpointPath;
    m_versionPath    = versionPath;

    ShowReady();
    m_result = RewardAnalysisResult{};
    TryLoadCachedResult();  // shows results page if a cached file exists for this stage
}

void RewardImpactWidget::TryLoadCachedResult()
{
    if (m_versionPath.isEmpty())
        return;

    optional<RewardAnalysisResult> cached =
        TryLoadRewardAnalysisResult(m_versionPath.toStdString(), m_stageName.toStdString());
    if (!cached.has_value())
        return;

    // Reuse OnAnalysisComplete so the result is displayed identically to a fresh run.
    // Passing m_currentRunId (not a literal run id) keeps this valid whether we're loading
    // on construction or after SetContext() has bumped the id.
    // QWidget has its own ::move(QPoint) — qualify explicitly to disambiguate.
    OnAnalysisComplete(m_currentRunId, std::move(*cached));
}

void RewardImpactWidget::ShowReady()   { m_stack->setCurrentWidget(m_readyPage); }
void RewardImpactWidget::ShowRunning() { m_stack->setCurrentWidget(m_runningPage); }
void RewardImpactWidget::ShowResults() { m_stack->setCurrentWidget(m_resultsPage); }

void RewardImpactWidget::OnRunClicked()
{
    const int timesteps = m_timestepSpin->value();

    // Bump the run id so any stale signals from a prior run get filtered out.
    const int runId = ++m_currentRunId;

    ShowRunning();
    m_progressBar->setValue(0);
    m_progressLabel->setText("Starting...");

    // Dispatch onto the worker thread. Capture by value so the lambda is self-contained.
    // The worker echoes runId back in every signal — see OnProgressUpdated/OnAnalysisComplete.
    AnalysisWorker* worker    = m_worker;
    const QString agentName   = m_agentName;
    const QString stageName   = m_stageName;
    const QString versionId   = m_versionId;
    const QString checkpoint  = m_checkpointPath;
    const QString versionPath = m_versionPath;
    QMetaObject::invokeMethod(worker, [worker, runId, agentName, stageName, versionId,
                                       checkpoint, versionPath, timesteps]() {
        worker->Start(runId, agentName, stageName, versionId, checkpoint, versionPath, timesteps);
    }, Qt::QueuedConnection);
}

void RewardImpactWidget::OnCancelClicked()
{
    if (m_worker) m_worker->Cancel();
    ShowReady();
}

void RewardImpactWidget::OnProgressUpdated(int runId, int current, int total, const QString& label)
{
    if (runId != m_currentRunId) return;  // stale signal from a previous run
    m_progressLabel->setText(
        QString("%1 - %2 / %3 timesteps").arg(label).arg(current).arg(total));
    m_progressBar->setMaximum(total);
    m_progressBar->setValue(std::min(current, total));
}

void RewardImpactWidget::OnAnalysisFailed(int runId, const QString& error)
{
    if (runId != m_currentRunId) return;  // stale
    ShowReady();
    QMessageBox::warning(this, "Reward Analysis Failed", error);
}

void RewardImpactWidget::OnAnalysisComplete(int runId, RewardAnalysisResult result)
{
    if (runId != m_currentRunId) return;  // stale

    if (result.teamSizes.empty()) {
        // Cancelled or otherwise produced no data — return to ready without showing results.
        ShowReady();
        return;
    }

    // Update the cached checkpoint path in case it changed (e.g. new checkpoint was created).
    if (!result.checkpointPath.empty())
        m_checkpointPath = QString::fromStdString(result.checkpointPath);

    m_result = std::move(result);  // qualified to avoid clash with QWidget::move

    m_teamSizeCombo->blockSignals(true);
    m_teamSizeCombo->clear();
    if (m_result.teamSizes.size() > 1)
        m_teamSizeCombo->addItem("All (average)");
    for (const TeamSizeImpactResult& teamSize : m_result.teamSizes)
        m_teamSizeCombo->addItem(QString::fromStdString(teamSize.label));
    m_teamSizeCombo->setVisible(m_result.teamSizes.size() > 1);
    m_teamSizeCombo->blockSignals(false);

    const TeamSizeImpactResult& first =
        (m_result.teamSizes.size() > 1) ? m_result.average : m_result.teamSizes.front();
    UpdateSummaryLabel(first);
    BuildResultsTable(first);
    ShowResults();
}

void RewardImpactWidget::OnSortChanged(int /*index*/)
{
    OnTeamSizeChanged(m_teamSizeCombo->currentIndex());
}

void RewardImpactWidget::OnTeamSizeChanged(int index)
{
    if (m_result.teamSizes.empty()) return;

    const TeamSizeImpactResult* data = nullptr;
    if (m_result.teamSizes.size() > 1 && index == 0)
        data = &m_result.average;
    else {
        const int teamSizeIndex = (m_result.teamSizes.size() > 1) ? index - 1 : index;
        if (teamSizeIndex >= 0 && teamSizeIndex < static_cast<int>(m_result.teamSizes.size()))
            data = &m_result.teamSizes[teamSizeIndex];
    }

    if (data) {
        UpdateSummaryLabel(*data);
        BuildResultsTable(*data);
    }
}

void RewardImpactWidget::UpdateSummaryLabel(const TeamSizeImpactResult& data)
{
    m_summaryLabel->setText(
        QString("%1 timesteps - %2 episodes - g=%3 - avg ep length: %4 steps\n"
                "Total discounted return: %5 +/- %6")
            .arg(m_result.timestepsPerTeamSize)
            .arg(data.episodesRun)
            .arg(m_result.gaeGamma, 0, 'f', 2)
            .arg(static_cast<int>(data.meanEpisodeLength))
            .arg(data.meanTotalDiscountedReturn, 0, 'f', 1)
            .arg(data.stdTotalDiscountedReturn, 0, 'f', 1));
}

void RewardImpactWidget::BuildResultsTable(const TeamSizeImpactResult& data)
{
    // setRowCount(0) before reassigning so QProgressBar cell widgets from a previous
    // run are destroyed via Qt's child-cleanup. Without this, re-runs leak N progress bars.
    m_impactTable->setRowCount(0);
    m_impactTable->clear();
    m_impactTable->setColumnCount(7);
    m_impactTable->setHorizontalHeaderLabels(
        { "Name", "Category", "Impact", "Signal Magnitude",
          "Adaptive rate (avg)", "Adaptive rate (max)", "Detail" });
    m_impactTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Interactive);
    m_impactTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_impactTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    m_impactTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    m_impactTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    m_impactTable->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    m_impactTable->horizontalHeader()->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    m_impactTable->setColumnWidth(0, 160);

    m_impactTable->horizontalHeaderItem(3)->setToolTip(
        "Average total reward this function contributed per player per episode,\n"
        "with rewards further in the future counting for less (same discount used in training).\n"
        "A larger value means the agent sees a stronger signal from this reward.\n"
        "Sign shows whether it trends positive or negative; +/- shows episode-to-episode variability.\n"
        "The 'All (average)' view omits +/- because standard deviations don't pool across\n"
        "team sizes - select a specific team size to see variability.");

    const QString adaptiveTip =
        "Mean unweighted |raw reward| per player-step, sampled only from kickoff/random\n"
        "episodes (the same signal the adaptive weight controller measures).\n"
        "Set an adaptive reward's target directly to this value to reproduce this\n"
        "segment's production rate. '—' = no kickoff/random episodes were sampled\n"
        "(pure-scenario stage, or an uneven matchup that never collects).";
    m_impactTable->horizontalHeaderItem(4)->setToolTip(adaptiveTip + "\n\nAverage aggregation.");
    m_impactTable->horizontalHeaderItem(5)->setToolTip(adaptiveTip +
        "\n\nMax aggregation: per-arena top player, averaged over training-sized windows\n"
        "(reproduces the controller's per-iteration Max; a whole-run max would under-report).");

    m_impactTable->horizontalHeaderItem(6)->setToolTip(
        "Continuous rewards: average weighted reward earned per game step.\n"
        "Event rewards: how many times the event fires per episode, and the average reward per firing.\n"
        "'(N events total)' shows how many times the event was observed across the whole run -\n"
        "small numbers mean the per-fire stats are noisy.");

    const SortMode sortMode = m_sortCombo
        ? static_cast<SortMode>(m_sortCombo->currentIndex())
        : SortMode::EventsFirst;

    vector<RewardImpactEntry> sorted = data.rewards;
    std::stable_sort(sorted.begin(), sorted.end(),
        [sortMode](const RewardImpactEntry& a,
                   const RewardImpactEntry& b) {
            if (sortMode != SortMode::ByImpact && a.category != b.category) {
                const bool aIsEvent = a.category == RewardCategory::EVENT;
                return sortMode == SortMode::EventsFirst ? aIsEvent : !aIsEvent;
            }
            return a.meanAbsDiscountedReturn > b.meanAbsDiscountedReturn;
        });

    // Compute total absolute return for bar scaling — use gross per-player signal, not signed mean.
    float totalAbsReturn = 0.f;
    for (const RewardImpactEntry& entry : sorted)
        totalAbsReturn += entry.meanAbsDiscountedReturn;

    m_impactTable->setRowCount(static_cast<int>(sorted.size()));
    for (int row = 0; row < static_cast<int>(sorted.size()); row++) {
        const RewardImpactEntry& entry = sorted[row];

        m_impactTable->setItem(row, 0, ReadOnlyItem(QString::fromStdString(entry.name)));
        m_impactTable->setItem(row, 1, ReadOnlyItem(
            entry.category == RewardCategory::EVENT ? "Event" : "Continuous"));

        const float absContrib = (totalAbsReturn > ZERO_GUARD)
            ? entry.meanAbsDiscountedReturn / totalAbsReturn * 100.f
            : 0.f;
        QProgressBar* bar = new QProgressBar();
        bar->setRange(0, 100);
        bar->setValue(static_cast<int>(absContrib));
        bar->setFormat(QString::number(static_cast<double>(absContrib), 'f', 1) + "%");
        bar->setAlignment(Qt::AlignCenter);
        bar->setStyleSheet(entry.meanDiscountedReturn >= 0.f ? POSITIVE_BAR_STYLE : NEGATIVE_BAR_STYLE);
        m_impactTable->setCellWidget(row, 2, bar);

        // Use the abs magnitude (non-zero for ZeroSum) with sign from the signed mean.
        // copysign gives + when meanDiscountedReturn is 0 (ZeroSum symmetric case).
        const float signedMag = std::copysign(entry.meanAbsDiscountedReturn, entry.meanDiscountedReturn);
        // Skip the "+/- N" suffix when std is zero. This happens in the 'average' row
        // (stds aren't pooled across team sizes) and for any reward with a perfectly
        // constant signal — showing "+/- 0" is misleading in both cases.
        QString magnitudeText = QString::number(static_cast<double>(signedMag), 'f', 2);
        if (entry.stdAbsDiscountedReturn > ZERO_GUARD) {
            magnitudeText += QString(" +/- %1")
                .arg(static_cast<double>(entry.stdAbsDiscountedReturn), 0, 'f', 2);
        }
        m_impactTable->setItem(row, 3, ReadOnlyItem(magnitudeText));

        // 0 sample count means no kickoff/random episodes were collected for this reward -
        // the adaptive rate is undefined, so show an em dash instead of a misleading 0.
        auto adaptiveCell = [&entry](float value) -> QTableWidgetItem* {
            if (entry.normalGameplaySampleCount <= 0)
                return ReadOnlyItem(QStringLiteral("—"));
            return ReadOnlyItem(QString::number(static_cast<double>(value), 'f', 4));
        };
        m_impactTable->setItem(row, 4, adaptiveCell(entry.normalGameplayAvgPerStep));
        m_impactTable->setItem(row, 5, adaptiveCell(entry.normalGameplayMaxPerStep));

        QString detail;
        if (entry.category == RewardCategory::CONTINUOUS) {
            detail = QString("%1/step").arg(static_cast<double>(entry.meanWeightedPerStep), 0, 'f', 4);
        } else {
            detail = QString("%1 fires/ep, %2/fire (%3 events total)")
                .arg(static_cast<double>(entry.meanFiresPerEpisode), 0, 'f', 2)
                .arg(static_cast<double>(entry.meanWeightedPerFire), 0, 'f', 2)
                .arg(entry.totalFires);
        }
        m_impactTable->setItem(row, 6, ReadOnlyItem(detail));
    }
}
