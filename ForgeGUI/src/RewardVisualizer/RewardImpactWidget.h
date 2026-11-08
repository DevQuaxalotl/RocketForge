#pragma once

#include <RewardAnalysis/RewardAnalysis.h>
#include <QWidget>
#include <QString>

class AnalysisWorker;
class QComboBox;
class QLabel;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QStackedWidget;
class QThread;
class QTableWidget;

/**
 * Three-state widget for the Reward Impact Analysis tool: Ready -> Running -> Results.
 * Owns an AnalysisWorker that calls RocketForge::RunRewardAnalysis() directly on a
 * background QThread (no subprocess).
 **/
class RewardImpactWidget : public QWidget
{
    Q_OBJECT

public:
    explicit RewardImpactWidget(const QString& agentName,
                                const QString& stageName,
                                const QString& versionId,
                                const QString& checkpointPath,
                                const QString& versionPath,
                                QWidget* parent = nullptr);
    ~RewardImpactWidget() override;

    /**
     * Re-points the widget at a new target: resets the ready page, clears results, and loads
     * the cached result for (versionPath, stageName) if any. Safe to call repeatedly.
     **/
    void SetContext(const QString& agentName,
                    const QString& stageName,
                    const QString& versionId,
                    const QString& checkpointPath,
                    const QString& versionPath);

private slots:
    void OnRunClicked();
    void OnCancelClicked();
    void OnProgressUpdated(int runId, int current, int total, const QString& teamSizeLabel);
    void OnAnalysisComplete(int runId, RocketForge::RewardAnalysisResult result);
    void OnAnalysisFailed(int runId, const QString& error);
    void OnTeamSizeChanged(int index);
    void OnSortChanged(int index);

private:
    /** Sort modes for the impact table. **/
    enum class SortMode {
        ByImpact        = 0,
        EventsFirst     = 1,
        ContinuousFirst = 2
    };

    void ShowReady();
    void ShowRunning();
    void ShowResults();
    void BuildResultsTable(const RocketForge::TeamSizeImpactResult& data);
    void UpdateSummaryLabel(const RocketForge::TeamSizeImpactResult& data);
    void TryLoadCachedResult();

    QString m_agentName;
    QString m_stageName;
    QString m_versionId;
    QString m_checkpointPath;
    QString m_versionPath;

    RocketForge::RewardAnalysisResult m_result;

    /** Background worker. Lives on m_workerThread; must be deleteLater'd by the thread. **/
    AnalysisWorker* m_worker       = nullptr;
    QThread*        m_workerThread = nullptr;

    /** Bumped each time OnRunClicked() dispatches a new analysis. Late-arriving signals
        from a previous run are filtered out by comparing against this. **/
    int m_currentRunId = 0;

    QStackedWidget* m_stack = nullptr;

    // Ready page
    QWidget*  m_readyPage    = nullptr;
    QSpinBox* m_timestepSpin = nullptr;

    // Running page
    QWidget*      m_runningPage   = nullptr;
    QLabel*       m_progressLabel = nullptr;
    QProgressBar* m_progressBar   = nullptr;

    // Results page
    QWidget*      m_resultsPage   = nullptr;
    QLabel*       m_summaryLabel  = nullptr;
    QComboBox*    m_teamSizeCombo = nullptr;
    QComboBox*    m_sortCombo     = nullptr;
    QTableWidget* m_impactTable   = nullptr;
};
